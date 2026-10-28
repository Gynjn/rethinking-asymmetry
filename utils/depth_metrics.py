"""Per-view depth metrics with median scaling (ZipMap / MonST3R `depth_evaluation`, default branch).

Valid pixels: 0 < gt < max_depth. The prediction is scaled by median(gt) / median(pred) over them,
then AbsRel and the delta < 1.25^k accuracies are computed on the same pixels.
"""
import numpy as np
import torch

KEYS = ["Abs Rel", "δ < 1.25", "δ < 1.25^2"]


def per_view_metrics(pred, gt, max_depth=10.0, min_valid=100):
    """pred, gt: [h, w] z-depth maps. Returns a dict of metrics, or None with fewer than min_valid GT pixels."""
    if (gt > 0).sum() < min_valid:
        return None
    pred = torch.from_numpy(np.asarray(pred, dtype=np.float32))
    gt = torch.from_numpy(np.asarray(gt, dtype=np.float32))
    mask = (gt > 0) & (gt < max_depth)
    p, g = pred[mask], gt[mask]
    p = p * (torch.median(g) / torch.median(p))
    abs_rel = torch.mean(torch.abs(p - g) / g).item()
    ratio = torch.maximum(p.clamp(min=1e-5) / g, g / p.clamp(min=1e-5))
    return {"Abs Rel": abs_rel,
            "δ < 1.25": torch.mean((ratio < 1.25).float()).item(),
            "δ < 1.25^2": torch.mean((ratio < 1.25 ** 2).float()).item()}


def mean_metrics(rows):
    rows = [r for r in rows if r is not None]
    return {k: float(np.mean([r[k] for r in rows])) for k in KEYS}, len(rows)
