"""Cameras and depth from unposed images.

    python inference.py --ckpt CKPT --images path/to/frames --num_views 10 --out out/

Takes `--num_views` frames evenly spaced over the sorted image folder (or all of them), predicts
every camera relative to the middle frame and the depth of every frame in one forward pass, and
writes cameras.json, depth_XX.npy (planar z-depth), depth previews and a coloured point cloud
points.ply (pixels lifted with the predicted cameras). Scale is arbitrary (self-supervised).
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
from PIL import Image

from model.mvssl import cam_info_to_plucker, ray_to_z
from utils.model_io import load_images, load_model


def colorize(z):
    import matplotlib
    lo, hi = np.percentile(z, 2), np.percentile(z, 98)
    rgba = matplotlib.colormaps["turbo"](np.clip((z - lo) / max(hi - lo, 1e-8), 0, 1))
    return (rgba[..., :3] * 255).astype(np.uint8)


def write_ply(path, xyz, rgb):
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
        rec = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
        rec["x"], rec["y"], rec["z"] = xyz.T
        rec["r"], rec["g"], rec["b"] = rgb.T
        f.write(rec.tobytes())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/dl3dv.yaml")
    p.add_argument("--ckpt", default="", help="default: the released model_dl3dv.pt (downloaded from Hugging Face)")
    p.add_argument("--weights", default="model", help="weight set to load from a training checkpoint")
    p.add_argument("--images", required=True, help="folder of frames (sorted by name) or a glob")
    p.add_argument("--num_views", type=int, default=10, help="0 = use every image")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    paths = sorted(glob.glob(os.path.join(args.images, "*")) if os.path.isdir(args.images) else glob.glob(args.images))
    paths = [q for q in paths if q.lower().endswith((".png", ".jpg", ".jpeg"))]
    if args.num_views and len(paths) > args.num_views:
        paths = [paths[i] for i in np.linspace(0, len(paths) - 1, args.num_views).round().astype(int)]
    assert len(paths) >= 2, "need at least two images"

    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model, config = load_model(args.config, args.ckpt, args.weights, device)
    images = load_images(paths, config.model.image_tokenizer.image_size_h, config.model.image_tokenizer.patch_size)
    out = model.predict(images[None].to(device))
    c2w, fx, ray_depth = out.c2w[0].float(), out.fxfycxcy[0].float(), out.depth[0].float()
    z = ray_to_z(ray_depth, fx)[:, 0].cpu().numpy()                     # [v, h, w]

    os.makedirs(args.out, exist_ok=True)
    v, _, h, w = images.shape
    cams = [{"image": os.path.basename(q), "c2w": c2w[i].cpu().tolist(),
             "fx": float(fx[i, 0] * w), "fy": float(fx[i, 1] * h), "cx": float(fx[i, 2] * w), "cy": float(fx[i, 3] * h)}
            for i, q in enumerate(paths)]
    json.dump({"width": w, "height": h, "cameras": cams}, open(os.path.join(args.out, "cameras.json"), "w"), indent=1)
    for i in range(v):
        np.save(os.path.join(args.out, f"depth_{i:02d}.npy"), z[i])
        Image.fromarray(colorize(z[i])).save(os.path.join(args.out, f"depth_{i:02d}.png"))

    rays = cam_info_to_plucker(c2w[None], fx[None], config.model.image_tokenizer)   # [v, 6, h, w]
    xyz = (rays[:, :3] + rays[:, 3:] * ray_depth).permute(0, 2, 3, 1).reshape(-1, 3).cpu().numpy()
    rgb = (images.permute(0, 2, 3, 1).reshape(-1, 3).numpy() * 255).astype(np.uint8)
    write_ply(os.path.join(args.out, "points.ply"), xyz, rgb)
    print(f"{v} views -> {args.out} (cameras.json, depth_*.npy/png, points.ply)")


if __name__ == "__main__":
    main()
