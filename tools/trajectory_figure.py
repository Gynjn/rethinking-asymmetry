"""Camera-trajectory panels and APE for the paper's trajectory figures.

    python evaluate_pose.py   --ckpt CKPT --data_root DL3DV_ROOT --out ours_dl3dv.json
    python evaluate_indoor.py --ckpt CKPT --dataset scannet --data_root SCANNET_ROOT --out ours_scannet.json
    python tools/trajectory_figure.py --group dl3dv_main --pred Ours=ours_dl3dv.json --out figs/
    python tools/trajectory_figure.py --group dl3dv_main --pred E-RayZer=erayzer.json --pred Ours=ours_dl3dv.json --out figs/

Each prediction is aligned to the ground-truth cameras with one Sim(3) (Umeyama on camera centres);
APE is the RMSE of the aligned camera centres. A panel shows GT (black) and the aligned prediction
(colour) as frusta along the trajectory, with grey lines between corresponding cameras. All
methods of a scene share one axis cube, so their panels are directly comparable.

Pose dumps: {"scenes": {scene: {"gt_c2w": [V,4,4], "pred_c2w": [V,4,4]}}} as written by the
evaluation scripts (also accepted: {scene: ...} or {dataset: {scene: ...}}), views in temporal order.
Scenes: --group names a list in data/figure_scenes.json, or pass --scenes explicitly.
Writes <out>/<group>_<scene[:16]>_<method>.png and <out>/<group>_ape.json.
"""
import argparse
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image, ImageChops  # noqa: E402

GT_COLOR, LINK_COLOR = "#111111", "#a8a8b0"
COLORS = {"E-RayZer": "#e4453a", "Ours": "#8b5cf6"}
FALLBACK = ["#e4453a", "#2f9e44", "#1c7ed6", "#8b5cf6"]
PANEL_W, PANEL_H = 1152, 1248
EDGES = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1), (0, 5)]


def umeyama_sim3(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s_c, d_c = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(d_c.T @ s_c / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    Rm = U @ S @ Vt
    var = (s_c ** 2).sum() / len(src)
    scale = float(np.trace(np.diag(D) @ S) / var) if var > 1e-12 else 1.0
    return Rm, mu_d - scale * Rm @ mu_s, scale


def align_poses(pred, gt):
    """Sim(3)-align predicted camera-to-world poses onto GT; returns (aligned poses, APE)."""
    Rm, t, s = umeyama_sim3(pred[:, :3, 3], gt[:, :3, 3])
    out = np.tile(np.eye(4), (len(pred), 1, 1))
    out[:, :3, :3] = Rm @ pred[:, :3, :3]
    out[:, :3, 3] = s * (Rm @ pred[:, :3, 3].T).T + t
    return out, float(np.sqrt(((out[:, :3, 3] - gt[:, :3, 3]) ** 2).sum(1).mean()))


def frustum_pts(c2w, zoom, fov_deg=50.0, aspect=4 / 3):
    h = np.tan(np.radians(fov_deg) / 2) * zoom
    w = h * aspect
    cam = np.array([[0, 0, 0], [w, h, zoom], [w, -h, zoom], [-w, -h, zoom], [-w, h, zoom], [0, h * 1.5, zoom]])
    return (c2w[:3, :3] @ cam.T).T + c2w[:3, 3]


def draw(ax, poses, color, zoom, lw=1.0):
    for p in poses:
        pts = frustum_pts(p, zoom)
        for a, b in EDGES:
            ax.plot(*zip(pts[a], pts[b]), color=color, linewidth=lw, solid_capstyle="round")
    c = poses[:, :3, 3]
    ax.plot(c[:, 0], c[:, 1], c[:, 2], color=color, linewidth=2.0, marker="o", markersize=2.6)


def load_dump(path):
    d = json.load(open(path))
    if "scenes" in d:
        return d["scenes"]
    first = next(iter(d.values()))
    return first if "gt_c2w" not in first else d     # {dataset: {scene: ...}} or {scene: ...}


def find(dump, scene):
    keys = [k for k in dump if k == scene or k.startswith(scene) or k.split("@")[0] == scene]
    assert len(keys) == 1, f"scene {scene}: {len(keys)} matches"
    return dump[keys[0]]


def render_scene(gt, preds, colors, out_prefix):
    """gt [V,4,4]; preds {method: [V,4,4]} -> {method: APE}; one panel PNG per method."""
    aligned, ape = {}, {}
    for name, p in preds.items():
        aligned[name], ape[name] = align_poses(p, gt)
    allc = np.concatenate([gt[:, :3, 3]] + [a[:, :3, 3] for a in aligned.values()])
    zoom = max(float(np.ptp(allc, axis=0).max()) * 0.038, 1e-3)
    drawn = np.concatenate([frustum_pts(q, zoom) for q in gt] +
                           [frustum_pts(q, zoom) for a in aligned.values() for q in a] + [allc])
    lo, hi = drawn.min(0), drawn.max(0)
    ctr, half = (lo + hi) / 2, float((hi - lo).max()) / 2 * 1.04 + 1e-6

    for name in preds:
        fig = plt.figure(figsize=(6.0, 6.0))
        ax = fig.add_subplot(111, projection="3d")
        for g, q in zip(gt[:, :3, 3], aligned[name][:, :3, 3]):
            ax.plot(*zip(g, q), color=LINK_COLOR, linewidth=1.0, zorder=1)
        draw(ax, gt, GT_COLOR, zoom, lw=1.0)
        draw(ax, aligned[name], colors[name], zoom, lw=1.2)
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.set_major_locator(plt.MaxNLocator(4))
            a.set_ticklabels([])
            a.set_pane_color((0.965, 0.965, 0.972, 1.0))
            a.line.set_color((1, 1, 1, 0))
            a._axinfo["tick"]["inward_factor"] = 0.0
            a._axinfo["tick"]["outward_factor"] = 0.0
            for tl in a.get_ticklines():
                tl.set_visible(False)
        ax.tick_params(axis="both", which="both", length=0, width=0)
        ax.set_xlim(ctr[0] - half, ctr[0] + half)
        ax.set_ylim(ctr[1] - half, ctr[1] + half)
        ax.set_zlim(ctr[2] - half, ctr[2] + half)
        ax.set_box_aspect((1, 1, 1), zoom=1.02)
        ax.view_init(elev=26, azim=-64)
        fig.subplots_adjust(0, 0, 1, 1)
        tmp = f"{out_prefix}_{name}_raw.png"
        fig.savefig(tmp, dpi=200, facecolor="white")
        plt.close(fig)
        # crop to the drawn cube, pad to the panel aspect, resize
        im = Image.open(tmp).convert("RGB")
        im = im.crop(ImageChops.difference(im, Image.new("RGB", im.size, "white")).getbbox())
        w, h = im.size
        tw = max(w, round(h * PANEL_W / PANEL_H))
        th = max(h, round(tw * PANEL_H / PANEL_W))
        canvas = Image.new("RGB", (tw, th), "white")
        canvas.paste(im, ((tw - w) // 2, (th - h) // 2))
        canvas.resize((PANEL_W, PANEL_H), Image.LANCZOS).save(f"{out_prefix}_{name}.png")
        os.remove(tmp)
    return ape


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred", action="append", required=True, help="METHOD=dump.json (repeatable; panel order)")
    p.add_argument("--group", default="", help="scene list name in data/figure_scenes.json")
    p.add_argument("--scenes", nargs="*", default=[])
    p.add_argument("--scene_file", default=os.path.join(os.path.dirname(__file__), "..", "data", "figure_scenes.json"))
    p.add_argument("--out", required=True)
    args = p.parse_args()

    methods = [s.split("=", 1) for s in args.pred]
    dumps = {m: load_dump(f) for m, f in methods}
    colors = {m: COLORS.get(m, FALLBACK[i % len(FALLBACK)]) for i, (m, _) in enumerate(methods)}
    scenes = args.scenes or json.load(open(args.scene_file))[args.group]
    tag = args.group or "scenes"
    os.makedirs(args.out, exist_ok=True)

    result = {}
    for scene in scenes:
        entries = {m: find(dumps[m], scene) for m, _ in methods}
        gt = np.asarray(next(iter(entries.values()))["gt_c2w"], dtype=np.float64)
        preds = {m: np.asarray(e["pred_c2w"], dtype=np.float64) for m, e in entries.items()}
        ape = render_scene(gt, preds, colors, os.path.join(args.out, f"{tag}_{scene[:16]}"))
        result[scene] = ape
        print(f"{scene[:16]:16s}  " + "  ".join(f"{m} APE {v:.3f}" for m, v in ape.items()))
    json.dump(result, open(os.path.join(args.out, f"{tag}_ape.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
