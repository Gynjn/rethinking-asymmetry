"""3D reconstruction figure: the model's own point cloud and cameras from unposed images, one forward pass.

    python tools/render_reconstruction.py --ckpt CKPT --images path/to/frames --num_views 10 --out recon.png
    python tools/render_reconstruction.py --ckpt CKPT --images "scene/images_undistort/*.png" \
        --frame_ids 0,10,21,32,42,53,64,74,85,96 --out recon.png --orbit 120

Geometry (nothing fitted): every pixel of every view is placed at camera centre + unit ray * predicted
ray depth, with the predicted pose and intrinsics, coloured from its input image. A multi-view
consistency filter keeps a point only if the other views that see it (and where it is not occluded)
agree with its depth within 5% in at least two of them; then the far tail and isolated speckles are
removed. Rendered on white with a z-buffered point splatter, the predicted frusta and the camera path.
--orbit N also writes an N-frame orbit video (<out>.mp4) around the same view.
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.model_io import load_images, load_model  # noqa: E402

REL_TOL, MIN_AGREE = 0.05, 2
ACCENT = np.array([124, 77, 235]) / 255.0        # cameras (Ours purple)
PATH = np.array([170, 145, 245]) / 255.0         # camera path
SS = 3                                           # supersampling factor


def _rays(fxfycxcy, h, w):
    fx, fy, cx, cy = fxfycxcy * np.array([w, h, w, h])
    y, x = np.meshgrid(np.arange(h) + 0.5, np.arange(w) + 0.5, indexing="ij")
    return np.stack([(x - cx) / fx, (y - cy) / fy, np.ones_like(x)], -1)


def lift(image, c2w, ray_depth, fxfycxcy):
    """image [V,3,h,w] in [0,1], c2w [V,4,4], ray_depth [V,h,w], normalized fxfycxcy [V,4] -> points, colours."""
    V, _, h, w = image.shape
    zmaps = [ray_depth[j] / np.linalg.norm(_rays(fxfycxcy[j], h, w), axis=-1) for j in range(V)]
    pts, cols = [], []
    for i in range(V):
        di = _rays(fxfycxcy[i], h, w)
        dw = (di / np.linalg.norm(di, axis=-1, keepdims=True)) @ c2w[i, :3, :3].T
        X = (c2w[i, :3, 3] + dw * ray_depth[i][..., None]).reshape(-1, 3)
        C = image[i].transpose(1, 2, 0).reshape(-1, 3)
        vis = np.zeros(len(X), int)
        agree = np.zeros(len(X), int)
        for j in range(V):
            if j == i:
                continue
            Xc = (X - c2w[j, :3, 3]) @ c2w[j, :3, :3]
            z = Xc[:, 2]
            fx, fy, cx, cy = fxfycxcy[j] * np.array([w, h, w, h])
            ok = z > 1e-6
            u = np.full(len(X), -1.0)
            v = np.full(len(X), -1.0)
            u[ok] = fx * Xc[ok, 0] / z[ok] + cx
            v[ok] = fy * Xc[ok, 1] / z[ok] + cy
            ok &= (u >= 0) & (u < w) & (v >= 0) & (v < h)
            ui, vi = u[ok].astype(int), v[ok].astype(int)
            rel = (z[ok] - zmaps[j][vi, ui]) / zmaps[j][vi, ui]
            idx = np.where(ok)[0]
            vis[idx] += rel <= REL_TOL            # behind view j's surface (occluded there): no evidence
            agree[idx] += np.abs(rel) < REL_TOL
        keep = (vis > 0) & (agree >= np.minimum(MIN_AGREE, vis))
        pts.append(X[keep])
        cols.append(C[keep])
    return np.concatenate(pts), (np.clip(np.concatenate(cols), 0, 1) * 255).astype(np.uint8)


def clean(P, C, keep_pct=95, vox_div=120, min_pts=3):
    """Drop the far tail (distance to the median) and points in sparse voxels (isolated speckles)."""
    dist = np.linalg.norm(P - np.median(P, 0), axis=1)
    m = dist < np.percentile(dist, keep_pct)
    P, C = P[m], C[m]
    vox = np.percentile(np.linalg.norm(P - np.median(P, 0), axis=1), 80) * 2 / vox_div
    key = np.floor((P - P.min(0)) / vox).astype(np.int64)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    m = cnt[inv.ravel()] >= min_pts
    return P[m], C[m]


def look_at(eye, center, up):
    """OpenCV-style camera (x right, y down, z forward): camera coords = (X - t) @ R."""
    z = center - eye
    z = z / np.linalg.norm(z)
    y = -(up - up.dot(z) * z)
    y = y / np.linalg.norm(y)
    return np.stack([np.cross(y, z), y, z], 1), np.asarray(eye, dtype=np.float64)


def render(P, C, c2w, az=12.0, el=12.0, dscale=1.0, size=(1100, 820)):
    """Points [N,3] + colours [N,3] uint8 and cameras c2w [V,4,4] -> RGB image [H, W, 3].
    The view looks along the mean input viewing direction, rotated by az / el degrees."""
    PW, PH = size
    W, H = PW * SS, PH * SS
    center = np.median(P, 0)
    extent = np.percentile(np.linalg.norm(P - center, axis=1), 80)
    radius = np.percentile(np.linalg.norm(P - center, axis=1), 95)
    focal = 1.3 * W / 2
    fit = radius / np.tan(np.arctan(H / 2 / focal)) * 1.05 * dscale
    fwd = c2w[:, :3, 2].mean(0)
    fwd /= np.linalg.norm(fwd)
    up = -c2w[:, :3, 1].mean(0)
    up -= up.dot(fwd) * fwd
    up /= np.linalg.norm(up)
    side = np.cross(fwd, up)
    a, e = np.radians(az), np.radians(el)
    back = -np.cos(a) * np.cos(e) * fwd + np.sin(a) * np.cos(e) * side + np.sin(e) * up
    cam_d = ((c2w[:, :3, 3] - center) @ back).max()           # stay behind every input camera
    R, t = look_at(center + max(fit, 1.6 * cam_d + 0.3 * radius) * back, center, up)

    def nproj(X):
        Xc = (X - t) @ R
        z = np.maximum(Xc[:, 2], 1e-6)
        return Xc[:, 0] / z, Xc[:, 1] / z, Xc[:, 2]

    # auto-frame: robust point extent + all cameras inside the image, with a margin
    nu, nv, _ = nproj(P)
    cu0, cv0, _ = nproj(c2w[:, :3, 3])
    lo = np.minimum([np.percentile(nu, 2), np.percentile(nv, 2)], [cu0.min(), cv0.min()])
    hi = np.maximum([np.percentile(nu, 98), np.percentile(nv, 98)], [cu0.max(), cv0.max()])
    focal = 0.92 * min(W / (hi[0] - lo[0]), H / (hi[1] - lo[1]))
    ox, oy = W / 2 - focal * (lo[0] + hi[0]) / 2, H / 2 - focal * (lo[1] + hi[1]) / 2

    def proj(X):
        u, v, z = nproj(X)
        return focal * u + ox, focal * v + oy, z

    # z-buffered splats, far to near
    u, v, z = proj(P)
    ok = (z > 1e-3) & (u > -10) & (u < W + 10) & (v > -10) & (v < H + 10)
    u, v, z, Cc = u[ok], v[ok], z[ok], C[ok]
    o = np.argsort(-z)
    u, v, z, Cc = u[o], v[o], z[o], Cc[o]
    rad = np.clip(np.round(2.2 * SS * np.median(z) / z / 2), 1, 3 * SS).astype(int)
    img = np.full((H, W, 3), 255, np.uint8)
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    r = rad.max()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            m = (dx * dx + dy * dy) <= rad * rad
            x, y = ui[m] + dx, vi[m] + dy
            inb = (x >= 0) & (x < W) & (y >= 0) & (y < H)
            img[y[inb], x[inb]] = Cc[m][inb]

    # cameras: path + translucent frusta, drawn over the points
    col = tuple(int(c) for c in (ACCENT * 255)[::-1])
    pcol = tuple(int(c) for c in (PATH * 255)[::-1])
    bgr = img[..., ::-1].copy()
    cu, cv_, _ = proj(c2w[:, :3, 3])
    q = np.stack([cu, cv_], 1).astype(int)
    for k in range(len(q) - 1):
        cv2.line(bgr, tuple(q[k]), tuple(q[k + 1]), pcol, 2 * SS, cv2.LINE_AA)
    s = extent * 0.08
    h = np.tan(np.radians(25)) * s
    for T in c2w:
        cam = np.array([[0, 0, 0], [h, h, s], [h, -h, s], [-h, -h, s], [-h, h, s]])
        fu, fv, fz = proj(cam @ T[:3, :3].T + T[:3, 3])
        if (fz <= 1e-3).any():
            continue
        p = np.stack([fu, fv], 1).astype(int)
        ov = bgr.copy()
        cv2.fillPoly(ov, [p[1:].reshape(-1, 1, 2)], col, cv2.LINE_AA)
        bgr = cv2.addWeighted(ov, 0.25, bgr, 0.75, 0)
        for a_, b_ in [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]:
            cv2.line(bgr, tuple(p[a_]), tuple(p[b_]), col, int(1.6 * SS), cv2.LINE_AA)
    return cv2.resize(bgr, (PW, PH), interpolation=cv2.INTER_AREA)[..., ::-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/dl3dv.yaml")
    p.add_argument("--ckpt", default="", help="default: the released model_dl3dv.pt (downloaded from Hugging Face)")
    p.add_argument("--weights", default="model")
    p.add_argument("--images", required=True, help="folder of frames (sorted by name) or a glob")
    p.add_argument("--num_views", type=int, default=10, help="frames spaced evenly over the sequence")
    p.add_argument("--frame_ids", default="", help="explicit comma-separated indices into the sorted frames")
    p.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--azim", type=float, default=12.0)
    p.add_argument("--elev", type=float, default=12.0)
    p.add_argument("--dist", type=float, default=1.0, help="viewing distance scale")
    p.add_argument("--orbit", type=int, default=0, help="also write an orbit video with this many frames")
    p.add_argument("--out", required=True, help="output .png")
    args = p.parse_args()

    paths = sorted(glob.glob(os.path.join(args.images, "*")) if os.path.isdir(args.images) else glob.glob(args.images))
    paths = [q for q in paths if q.lower().endswith((".png", ".jpg", ".jpeg"))]
    if args.frame_ids:
        paths = [paths[int(i)] for i in args.frame_ids.split(",")]
    elif args.num_views and len(paths) > args.num_views:
        paths = [paths[i] for i in np.linspace(0, len(paths) - 1, args.num_views).round().astype(int)]

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model, config = load_model(args.config, args.ckpt, args.weights)
    images = load_images(paths, config.model.image_tokenizer.image_size_h, config.model.image_tokenizer.patch_size)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"):
        out = model.predict(images[None].cuda())
    c2w = out.c2w[0].float().cpu().numpy().astype(np.float64)
    fx = out.fxfycxcy[0].float().cpu().numpy()
    depth = out.depth[0, :, 0].float().cpu().numpy().astype(np.float64)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    P, C = clean(*lift(images.numpy(), c2w, depth, fx))
    cv2.imwrite(args.out, render(P, C, c2w, args.azim, args.elev, args.dist)[..., ::-1])
    print(f"{len(paths)} views, {len(P)} points -> {args.out}")
    if args.orbit:
        video = os.path.splitext(args.out)[0] + ".mp4"
        frames = [render(P, C, c2w, args.azim + 35 * np.sin(2 * np.pi * k / args.orbit), args.elev, args.dist)
                  for k in range(args.orbit)]
        h, w = frames[0].shape[:2]
        wr = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
        for f in frames:
            wr.write(f[..., ::-1])
        wr.release()
        print(f"orbit video ({args.orbit} frames) -> {video}")


if __name__ == "__main__":
    main()
