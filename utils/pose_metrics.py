"""Pairwise relative-pose metrics (scale-invariant): RRA / RTA / RPA at angular thresholds and AUC.

For every pair (i, j), i < j, in temporal order, the rotation error is the geodesic angle of the
relative rotation and the translation error the angle between relative translation directions
(min(e, 180 - e)). RPA counts a pair when both are within the threshold; accuracies are averaged
per scene, then over scenes. AUC integrates the recall of max(e_R, e_t) over all pairs.
"""
import numpy as np

_trapezoid = getattr(np, "trapezoid", None) or np.trapz   # numpy >= 2 renamed trapz

THRESHOLDS = [3, 5, 10, 15, 30]


def _proj_so3(R):
    """Nearest rotation matrix (SVD), so tiny non-orthonormality does not floor small angles to 0."""
    U, _, Vt = np.linalg.svd(R)
    d = np.sign(np.linalg.det(np.matmul(U, Vt)))
    D = np.tile(np.eye(3), (R.shape[0], 1, 1))
    D[:, 2, 2] = d
    return np.matmul(np.matmul(U, D), Vt)


def _rot_angle_deg(R_gt, R_pred, eps=1e-15):
    R_gt, R_pred = _proj_so3(R_gt), _proj_so3(R_pred)
    trace = np.trace(np.einsum("nji,njk->nik", R_gt, R_pred), axis1=-2, axis2=-1)
    return np.degrees(np.arccos(np.clip((trace - 1) / 2, -1 + eps, 1 - eps)))


def _trans_angle_deg(t_gt, t_pred, eps=1e-15):
    a = t_gt / (np.linalg.norm(t_gt, axis=1, keepdims=True) + eps)
    b = t_pred / (np.linalg.norm(t_pred, axis=1, keepdims=True) + eps)
    ang = np.arccos(np.clip((a * b).sum(1), -1 + eps, 1 - eps))
    return np.degrees(np.minimum(ang, np.pi - ang))


def _inv(T):
    R, t = T[:, :3, :3], T[:, :3, 3:]
    out = np.tile(np.eye(4), (T.shape[0], 1, 1))
    out[:, :3, :3] = np.transpose(R, (0, 2, 1))
    out[:, :3, 3:] = -np.matmul(np.transpose(R, (0, 2, 1)), t)
    return out


def pose_auc(errors, thresholds=THRESHOLDS):
    errors = np.sort(np.asarray(errors, dtype=np.float64))
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors, recall = np.r_[0.0, errors], np.r_[0.0, recall]
    out = []
    for t in thresholds:
        last = np.searchsorted(errors, t)
        r = np.r_[recall[:last], recall[last - 1]]
        e = np.r_[errors[:last], t]
        out.append(float(_trapezoid(r, x=e) / t))
    return out


def pair_errors(gt_c2w, pred_c2w):
    """gt / pred camera-to-world [v, 4, 4] in temporal order -> (rotation errors, translation errors) per pair."""
    gt, pr = np.asarray(gt_c2w, np.float64), np.asarray(pred_c2w, np.float64)
    i1, i2 = np.triu_indices(gt.shape[0], k=1)
    gw, pw = _inv(gt), _inv(pr)
    rg = np.matmul(gw[i1], _inv(gw[i2]))
    rp = np.matmul(pw[i1], _inv(pw[i2]))
    return _rot_angle_deg(rg[:, :3, :3], rp[:, :3, :3]), _trans_angle_deg(rg[:, :3, 3], rp[:, :3, 3])


def aggregate(per_scene):
    """per_scene: {scene: (e_R, e_t)} -> {RRA@t, RTA@t, RPA@t, AUC@t}."""
    res = {}
    for kind in ("RRA", "RTA", "RPA"):
        for t in THRESHOLDS:
            vals = []
            for eR, et in per_scene.values():
                e = {"RRA": eR, "RTA": et, "RPA": np.maximum(eR, et)}[kind]
                vals.append(float((e <= t).mean()))
            res[f"{kind}@{t}"] = float(np.mean(vals))
    pmax = np.concatenate([np.maximum(eR, et) for eR, et in per_scene.values()])
    for t, a in zip(THRESHOLDS, pose_auc(pmax)):
        res[f"AUC@{t}"] = a
    return res
