# Adapted from the LVSM (ICLR 2025) and RayZer (ICCV 2025) DL3DV loaders.
"""DL3DV scenes in the `opencv_cameras.json` layout.

Each line of a scene list is a scene id; the scene is read from
`<data_root>/<scene_id>/opencv_cameras.json`, whose `frames` carry `file_path` (relative to the
scene directory), `fx, fy, cx, cy` (pixels) and `w2c`.

Training (variable views): the batch sampler hands over `(scene_idx, bucket_idx)`; the bucket sets
the view count, the frame span (scaled with the view count, optional curriculum) and the split
into context / held-out views (`data/view_buckets.py`).
Evaluation: `training.num_views` frames at `linspace(0, max_frame_dist, num_views)`, even positions
as context (`even_odd_split_positions`).
"""

import json
import os
import random
import traceback

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset

from data.view_buckets import (
    build_buckets,
    even_odd_split_positions,
    random_split_positions,
    resolve_frame_dist,
    stratified_middle_frames,
    uses_view_buckets,
)


class Dataset(TorchDataset):
    def __init__(self, config, inference=False):
        super().__init__()
        self.config = config
        self.current_iteration = 0
        self.inference = inference
        self.buckets = build_buckets(config) if uses_view_buckets(config) else None
        assert inference or self.buckets is not None, "training needs training.num_views_buckets"

        self.data_root = config.training.data_root
        with open(config.training.dataset_path, "r") as f:
            self.scene_ids = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.scene_ids)

    def update_iteration(self, iteration):
        self.current_iteration = iteration
        vs = self.config.training.view_selector
        if not vs.get("use_curriculum", False):
            return
        spans = ", ".join(
            "{}v: {}-{}".format(b["num_views"], *resolve_frame_dist(vs, b["num_views"], b["max_views"], iteration, scale=True))
            for b in self.buckets
        )
        print(f"Curriculum progress {min(iteration / vs.get('curriculum_iter', 30000), 1.0):.3f}, frame span per bucket: {spans}")

    # ------------------------------------------------------------------
    def preprocess_frames(self, frames_chosen, image_paths_chosen):
        """Resize to height image_size_h (width rounded to the patch size), centre-crop to a square,
        and update the pixel intrinsics accordingly."""
        resize_h = self.config.model.image_tokenizer.image_size_h
        patch_size = self.config.model.image_tokenizer.patch_size
        square_crop = self.config.training.get("square_crop", True)

        images, intrinsics = [], []
        for frame, path in zip(frames_chosen, image_paths_chosen):
            image = PIL.Image.open(path)
            orig_w, orig_h = image.size
            resize_w = int(resize_h / orig_h * orig_w)
            resize_w = int(round(resize_w / patch_size) * patch_size)
            image = image.resize((resize_w, resize_h), resample=PIL.Image.LANCZOS)
            if square_crop:
                min_size = min(resize_h, resize_w)
                start_h = (resize_h - min_size) // 2
                start_w = (resize_w - min_size) // 2
                image = image.crop((start_w, start_h, start_w + min_size, start_h + min_size))

            image = torch.from_numpy(np.array(image) / 255.0).permute(2, 0, 1).float()
            fxfycxcy = np.array([frame["fx"], frame["fy"], frame["cx"], frame["cy"]])
            rx, ry = resize_w / orig_w, resize_h / orig_h
            fxfycxcy *= (rx, ry, rx, ry)
            if square_crop:
                fxfycxcy[2] -= start_w
                fxfycxcy[3] -= start_h
            images.append(image)
            intrinsics.append(torch.from_numpy(fxfycxcy).float())

        w2cs = np.stack([np.array(frame["w2c"]) for frame in frames_chosen])
        c2ws = torch.from_numpy(np.linalg.inv(w2cs)).float()
        return torch.stack(images), torch.stack(intrinsics), c2ws

    @staticmethod
    def preprocess_poses(in_c2ws, scene_scale_factor=1.35):
        """Express the cameras relative to their mean pose and rescale the scene (used for the
        ground-truth poses only; the model never sees them)."""
        center = in_c2ws[:, :3, 3].mean(0)
        avg_forward = F.normalize(in_c2ws[:, :3, 2].mean(0), dim=-1)
        avg_down = in_c2ws[:, :3, 1].mean(0)
        avg_right = F.normalize(torch.cross(avg_down, avg_forward, dim=-1), dim=-1)
        avg_down = F.normalize(torch.cross(avg_forward, avg_right, dim=-1), dim=-1)

        avg_pose = torch.eye(4, device=in_c2ws.device)
        avg_pose[:3, :3] = torch.stack([avg_right, avg_down, avg_forward], dim=-1)
        avg_pose[:3, 3] = center
        in_c2ws = torch.linalg.inv(avg_pose) @ in_c2ws

        scene_scale = scene_scale_factor * torch.max(torch.abs(in_c2ws[:, :3, 3]))
        in_c2ws[:, :3, 3] /= scene_scale
        return in_c2ws

    def view_selector(self, frames, bucket):
        """Frame indices for one training sample of this bucket, or None if the clip is too short."""
        vs = self.config.training.view_selector
        num_views = bucket["num_views"]
        if len(frames) < num_views:
            return None
        min_frame_dist, max_frame_dist = resolve_frame_dist(
            vs, num_views, bucket["max_views"], self.current_iteration, scale=True)
        max_frame_dist = min(len(frames) - 1, max_frame_dist)
        min_frame_dist = min(min_frame_dist, max_frame_dist)
        if max_frame_dist < num_views - 1:
            return None
        return two_frame_selector(frames, num_views, min_frame_dist, max_frame_dist,
                                  stratified=vs.get("stratified", False))

    def _retry(self, bucket):
        nxt = random.randint(0, len(self) - 1)
        return self._load(nxt if bucket is None else (nxt, bucket["index"]))

    def __getitem__(self, idx):
        # Safety net for hard failures (missing / corrupt files): redraw a random scene.
        retries = self.config.training.get("dataset_getitem_retries", 8)
        for attempt in range(retries + 1):
            try:
                return self._load(idx)
            except Exception:
                if attempt == retries:
                    raise
                traceback.print_exc()
                nxt = random.randint(0, len(self) - 1)
                idx = nxt if not isinstance(idx, (tuple, list)) else (nxt, idx[1])

    def _load(self, idx):
        bucket = None
        if isinstance(idx, (tuple, list)):
            idx, bucket_idx = int(idx[0]), int(idx[1])
            bucket = self.buckets[bucket_idx]

        scene_dir = os.path.join(self.data_root, self.scene_ids[idx])
        data_json = json.load(open(os.path.join(scene_dir, "opencv_cameras.json"), "r"))
        frames = data_json["frames"]
        scene_name = data_json["scene_name"]

        if self.inference:
            num_views = self.config.training.num_views
            if len(frames) < num_views:
                return self._load((idx + 1) % len(self))
            max_frame = min(len(frames) - 1, self.config.training.view_selector.get("max_frame_dist", 100))
            image_indices = np.linspace(0, max_frame, num=num_views, dtype=int).tolist()
            context_indices, target_indices = even_odd_split_positions(num_views)
        else:
            image_indices = self.view_selector(frames, bucket)
            if not image_indices:
                return self._retry(bucket)
            image_indices = sorted(image_indices)
            context_indices, target_indices = random_split_positions(
                bucket["num_views"], bucket["num_input_views"], random)

        frames_chosen = [frames[i] for i in image_indices]
        image_paths = [os.path.join(scene_dir, f["file_path"]) for f in frames_chosen]
        try:
            images, intrinsics, c2ws = self.preprocess_frames(frames_chosen, image_paths)
        except FileNotFoundError as e:
            print(f"File not found for scene {scene_name}: {e}")
            return self._retry(bucket)

        det = torch.det(c2ws[:, :3, :3])
        if (c2ws[:, :3, 3] > 1e3).any() or torch.isnan(det).any() or not torch.allclose(det, det.new_tensor(1.0)):
            print(f"Invalid camera poses in scene {scene_name}, resampling")
            return self._retry(bucket)
        c2ws = self.preprocess_poses(c2ws, self.config.training.get("scene_scale_factor", 1.35))

        image_indices = torch.tensor(image_indices).long().unsqueeze(-1)
        return {
            "image": images,                      # [v, 3, h, w] in [0, 1]
            "c2w": c2ws,                          # ground truth, evaluation only
            "fxfycxcy": intrinsics,               # ground truth (pixels), evaluation only
            "index": torch.cat([image_indices, torch.full_like(image_indices, idx)], dim=-1),
            "scene_name": scene_name,
            "context_indices": context_indices,
            "target_indices": target_indices,
        }


def two_frame_selector(frames, num_views, min_frame_dist, max_frame_dist, stratified=False):
    """A random clip of `frame_dist` in [min, max] frames: both end frames plus num_views - 2 interior
    frames, either one per equal-width bin (stratified) or uniformly at random."""
    frame_dist = random.randint(min_frame_dist, max_frame_dist)
    if len(frames) <= frame_dist:
        return []
    start_frame = random.randint(0, len(frames) - frame_dist - 1)
    end_frame = start_frame + frame_dist
    if stratified:
        rest = stratified_middle_frames(start_frame + 1, end_frame, num_views - 2)
    else:
        rest = random.sample(range(start_frame + 1, end_frame), num_views - 2)
    return [start_frame] + rest + [end_frame]
