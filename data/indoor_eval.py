"""ScanNet v2 and NRGBD test scenes with ground-truth poses and depth (out-of-domain evaluation).

Both follow ZipMap's (MonST3R) preprocessing and readers:

ScanNet v2   <root>/<scene>/color_90/frame_XXXX.jpg   first 270 frames at stride 3 (1296 x 968)
             <root>/<scene>/depth_90/frame_XXXX.png   uint16 / 5000 (ZipMap's reader), 0 = invalid
             <root>/<scene>/pose_90.txt               flattened 4x4 camera-to-world (OpenCV)
             <root>/<scene>/intrinsic/intrinsic_color.txt
             10 views at linspace(0, 64, 10) of the stride-3 list.
NRGBD        <root>/<scene>/images/img<i>.png         640 x 480
             <root>/<scene>/depth/depth<i>.png        uint16 / 1000, outside [1e-3, 10] m = invalid
             <root>/<scene>/poses.txt                 4x4 camera-to-world, OpenGL axes (flipped to OpenCV)
             10 views at linspace(0, 256, 10) from the first frame.
"""
import os

import numpy as np
from PIL import Image

SPAN = {"scannet": 64, "nrgbd": 256}
NRGBD_K = (554.2562584220408, 554.2562584220408, 320.0, 240.0)   # fx, fy, cx, cy (pixels)


def frame_ids(dataset, num_views=10):
    return np.linspace(0, SPAN[dataset], num_views, dtype=int).tolist()


def load_scene(dataset, root, scene, ids):
    """-> image paths, frames ({fx, fy, cx, cy} in pixels, w2c), GT camera-to-world [v, 4, 4]."""
    if dataset == "scannet":
        c2w = np.loadtxt(os.path.join(root, scene, "pose_90.txt")).reshape(-1, 4, 4)[ids].astype(np.float64)
        k = np.loadtxt(os.path.join(root, scene, "intrinsic", "intrinsic_color.txt"))[:3, :3]
        fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
        paths = [os.path.join(root, scene, "color_90", f"frame_{i:04d}.jpg") for i in ids]
    elif dataset == "nrgbd":
        with open(os.path.join(root, scene, "poses.txt")) as f:
            vals = [[float(x) for x in line.split()] for line in f if line.strip()]
        c2w = np.array(vals, dtype=np.float64).reshape(-1, 4, 4)[ids]
        c2w[:, :3, 1:3] *= -1.0   # OpenGL -> OpenCV camera axes
        fx, fy, cx, cy = NRGBD_K
        paths = [os.path.join(root, scene, "images", f"img{i}.png") for i in ids]
    else:
        raise ValueError(dataset)
    frames = [{"fx": fx, "fy": fy, "cx": cx, "cy": cy, "w2c": np.linalg.inv(c).tolist()} for c in c2w]
    return paths, frames, c2w


def load_depth(dataset, root, scene, ids):
    """GT depth maps at their native resolution, 0 = invalid."""
    out = []
    for i in ids:
        if dataset == "scannet":
            d = np.asarray(Image.open(os.path.join(root, scene, "depth_90", f"frame_{i:04d}.png"))).astype(np.float32) / 5000.0
        else:
            import imageio.v2 as iio
            d = np.nan_to_num(iio.imread(os.path.join(root, scene, "depth", f"depth{i}.png")).astype(np.float32), nan=0.0) / 1000.0
            d[(d > 10) | (d < 1e-3)] = 0.0
        out.append(d)
    return out


def depth_to_image_grid(depth, image_w, image_h, size=256, patch_size=16):
    """Put a GT depth map through the colour image's preprocessing (height -> size, width rounded to the
    patch size, centre crop), with nearest-neighbour resampling, so GT and predicted pixels correspond."""
    resize_w = int(round(int(size / image_h * image_w) / patch_size) * patch_size)
    d = np.asarray(Image.fromarray(depth.astype(np.float32)).resize((resize_w, size), Image.NEAREST))
    left = (resize_w - size) // 2
    return d[:, left:left + size]
