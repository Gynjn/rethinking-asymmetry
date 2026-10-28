"""Camera pose and depth on ScanNet v2 / NRGBD (ground-truth depth), 10 views.

    python evaluate_indoor.py --ckpt CKPT --dataset scannet --data_root /path/to/scannetv2 [--depth_views input]

Per scene, 10 frames (data/indoor_eval.py) go through one forward pass:
  pose         all 10 cameras from the joint pass, scored pairwise in temporal order (as evaluate_pose.py)
  multi-view   depth decoded by the DPT head for --depth_views:
                 input  the 5 context views [0, 2, 4, 6, 9] (E-RayZer's 10-view split; paper Table 2)
                 all    all 10 views
  single-view  each of those views alone (v = 1, camera token kept); --no_single_view skips it
Depth is the distance along each ray, converted to planar z with the predicted intrinsics of the same
pass, and scored per view with median scaling against GT (utils/depth_metrics.py); the reported value
is the mean over all scored views.
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

from data.dataset_dl3dv import Dataset
from data.indoor_eval import depth_to_image_grid, frame_ids, load_depth, load_scene
from model.mvssl import ray_to_z
from utils.depth_metrics import KEYS, mean_metrics, per_view_metrics
from utils.model_io import load_model
from utils.pose_metrics import THRESHOLDS, aggregate, pair_errors

INPUT_POS = [0, 2, 4, 6, 9]
SCENE_LISTS = {"scannet": "data/scannetv2_eval_scenes.txt", "nrgbd": "data/nrgbd_eval_scenes.txt"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/dl3dv.yaml")
    p.add_argument("--ckpt", default="", help="default: the released model_dl3dv.pt (downloaded from Hugging Face)")
    p.add_argument("--weights", default="model")
    p.add_argument("--dataset", choices=["scannet", "nrgbd"], required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--scene_list", default="", help="default: data/<dataset>_eval_scenes.txt")
    p.add_argument("--num_views", type=int, default=10)
    p.add_argument("--depth_views", choices=["input", "all"], default="input")
    p.add_argument("--no_single_view", action="store_true")
    p.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--out", default="", help="optional json dump of per-scene poses and depth metrics")
    args = p.parse_args()
    assert args.depth_views == "all" or args.num_views == 10, "--depth_views input is defined for 10 views"

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model, config = load_model(args.config, args.ckpt, args.weights)
    size, patch = config.model.image_tokenizer.image_size_h, config.model.image_tokenizer.patch_size
    prep = Dataset.__new__(Dataset)   # only its image preprocessing is used
    prep.config = config

    def run(images, depth_idx):
        """images [v, 3, h, w] -> c2w [v, 4, 4], z-depth [len(depth_idx), h, w]."""
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"):
            out = model.predict(images[None].cuda(), torch.tensor([depth_idx]).cuda())
        fx = out.fxfycxcy[0].float()[depth_idx]
        z = ray_to_z(out.depth[0].float(), fx)[:, 0]
        return out.c2w[0].float().cpu().numpy(), z.cpu().numpy()

    scenes = [s.strip() for s in open(args.scene_list or SCENE_LISTS[args.dataset]) if s.strip()]
    ids = frame_ids(args.dataset, args.num_views)
    depth_pos = INPUT_POS if args.depth_views == "input" else list(range(args.num_views))
    pose_err, mv_rows, sv_rows, dump = {}, [], [], {}
    for scene in scenes:
        paths, frames, gt_c2w = load_scene(args.dataset, args.data_root, scene, ids)
        images, _, _ = prep.preprocess_frames(frames, paths)
        W, H = Image.open(paths[0]).size
        gts = [depth_to_image_grid(d, W, H, size, patch) for d in load_depth(args.dataset, args.data_root, scene, ids)]

        c2w, z = run(images, depth_pos)
        pose_err[scene] = pair_errors(gt_c2w, c2w)
        mv = [per_view_metrics(z[i], gts[j]) for i, j in enumerate(depth_pos)]
        sv = [] if args.no_single_view else [per_view_metrics(run(images[j:j + 1], [0])[1][0], gts[j]) for j in depth_pos]
        mv_rows += mv
        sv_rows += sv
        dump[scene] = {"ids": ids, "gt_c2w": gt_c2w.tolist(), "pred_c2w": c2w.tolist(), "depth_pos": depth_pos, "mv": mv, "sv": sv}

    res = aggregate(pose_err)
    print(f"{args.dataset} | {len(scenes)} scenes | {args.num_views} views | {args.precision} | depth on {args.depth_views} views")
    print("  pose        " + "  ".join(f"RPA@{t} {100 * res[f'RPA@{t}']:.1f}" for t in THRESHOLDS) + f"  AUC@30 {100 * res['AUC@30']:.1f}")
    summary = {"pose": res}
    for name, rows in (("multi-view", mv_rows), ("single-view", sv_rows)):
        if rows:
            m, n = mean_metrics(rows)
            summary[name] = {**m, "num_views": n}
            print(f"  {name:11s} " + "  ".join(f"{k} {m[k]:.3f}" for k in KEYS) + f"  ({n} views)")
    if args.out:
        json.dump({"summary": summary, "scenes": dump}, open(args.out, "w"))


if __name__ == "__main__":
    main()
