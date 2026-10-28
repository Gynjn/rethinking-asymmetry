"""Camera pose on the DL3DV benchmark (140 scenes).

    python evaluate_pose.py --ckpt CKPT --data_root /path/to/dl3dv_benchmark [--num_views 10 --span 96]

Protocol (E-RayZer's): per scene, `num_views` frames at linspace(0, span, num_views), all of them
as input in one forward pass, scored with pairwise RRA / RTA / RPA and AUC in temporal order
(utils/pose_metrics.py). Scenes shorter than span + 1 frames are skipped.
The forward runs under bf16 autocast by default, as in E-RayZer's evaluate.py (`--precision fp32`
for full precision); poses and metrics are fp32 either way.
"""
import argparse
import json
import os

import numpy as np
import torch

from data.dataset_dl3dv import Dataset
from utils.model_io import load_model
from utils.pose_metrics import THRESHOLDS, aggregate, pair_errors


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/dl3dv.yaml")
    p.add_argument("--ckpt", default="", help="default: the released model_dl3dv.pt (downloaded from Hugging Face)")
    p.add_argument("--weights", default="model")
    p.add_argument("--data_root", required=True)
    p.add_argument("--scene_list", default="data/dl3dv10k_benchmark.txt")
    p.add_argument("--num_views", type=int, default=10)
    p.add_argument("--span", type=int, default=96)
    p.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--out", default="", help="optional json dump of per-scene poses")
    args = p.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model, config = load_model(args.config, args.ckpt, args.weights)
    config.training.data_root = args.data_root
    config.training.dataset_path = args.scene_list
    config.training.num_views = args.num_views
    data = Dataset(config, inference=True)   # only its frame loader is used here

    ids = np.linspace(0, args.span, args.num_views, dtype=int).tolist()
    per_scene, dump = {}, {}
    for scene_id in data.scene_ids:
        scene_dir = os.path.join(args.data_root, scene_id)
        frames = json.load(open(os.path.join(scene_dir, "opencv_cameras.json")))["frames"]
        if len(frames) <= args.span:
            continue
        chosen = [frames[i] for i in ids]
        images, _, gt_c2w = data.preprocess_frames(chosen, [os.path.join(scene_dir, f["file_path"]) for f in chosen])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"):
            pred = model.predict(images[None].cuda()).c2w[0]
        pred = pred.float().cpu().numpy()
        per_scene[scene_id] = pair_errors(gt_c2w.numpy(), pred)
        dump[scene_id] = {"ids": ids, "gt_c2w": gt_c2w.tolist(), "pred_c2w": pred.tolist()}

    res = aggregate(per_scene)
    print(f"DL3DV benchmark | {len(per_scene)} scenes | {args.num_views} views over {args.span} frames | {args.precision}")
    print("  " + "  ".join(f"RPA@{t} {100 * res[f'RPA@{t}']:.1f}" for t in THRESHOLDS) + f"  AUC@30 {100 * res['AUC@30']:.1f}")
    if args.out:
        json.dump({"metrics": res, "scenes": dump}, open(args.out, "w"))


if __name__ == "__main__":
    main()
