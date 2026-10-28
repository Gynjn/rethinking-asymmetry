"""Differentiable Gaussian splatting (gsplat) for the predicted pixel-aligned Gaussians."""

from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from einops import rearrange, repeat
from gsplat import rasterization


# ---------------------------------------------------------------------------
# Quaternion helpers. NOTE: `quat_to_mat` / `mat_to_quat` use the scalar-last (xyzw)
# convention, so `cam_quat_wxyz_to_world_quat_wxyz` effectively reads the predicted
# quaternion as xyzw and hands gsplat the world rotation in that same layout. The released
# checkpoints were trained with exactly this mapping, so it is kept as-is.
# ---------------------------------------------------------------------------
def quat_to_mat(quaternions: torch.Tensor) -> torch.Tensor:
    i, j, k, r = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    return torch.where(quaternions[..., 3:4] < 0, -quaternions, quaternions)


def mat_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )
    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))
    out = quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(
        batch_dim + (4,)
    )
    out = out[..., [1, 2, 3, 0]]
    return standardize_quaternion(out)


def cam_quat_wxyz_to_world_quat_wxyz(cam_quat_wxyz, c2w):
    """Rotate per-Gaussian camera-frame quaternions [b, n, 4] into the world frame by c2w [b, n, 4, 4]."""
    b, n = cam_quat_wxyz.shape[:2]
    rotmat_cam = quat_to_mat(cam_quat_wxyz.reshape(-1, 4)).reshape(b, n, 3, 3)
    rotmat_world = torch.matmul(c2w[..., :3, :3], rotmat_cam)
    return mat_to_quat(rotmat_world.reshape(-1, 3, 3)).reshape(b, n, 4)


# ---------------------------------------------------------------------------
# Rasterization
# ---------------------------------------------------------------------------
def _invert_SE3(transforms: torch.Tensor) -> torch.Tensor:
    Rinv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = Rinv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def render_opencv_cam_gsplat(
    means3D: torch.Tensor,
    opacity: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    shs: torch.Tensor,
    height: int,
    width: int,
    C2W: torch.Tensor,
    fxfycxcy: torch.Tensor,
    sh_degree: Union[int, None] = None,
    near_plane=0.2,
    bg_color=(1.0, 1.0, 1.0),
):
    """Render one scene's Gaussians into v OpenCV cameras (C2W [v, 4, 4], pixel fxfycxcy [v, 4])."""
    bg_color = torch.tensor(list(bg_color), dtype=torch.float32, device=C2W.device).unsqueeze(0).expand(C2W.size(0), -1)
    W2C = _invert_SE3(C2W)
    intr = torch.zeros(fxfycxcy.size(0), 3, 3, device=fxfycxcy.device, dtype=fxfycxcy.dtype)
    intr[:, 0, 0] = fxfycxcy[:, 0]
    intr[:, 1, 1] = fxfycxcy[:, 1]
    intr[:, 0, 2] = fxfycxcy[:, 2]
    intr[:, 1, 2] = fxfycxcy[:, 3]
    intr[:, 2, 2] = 1.0
    render_colors, _, _ = rasterization(
        means3D, rotations, scales, opacity.squeeze(), shs, W2C, intr, width, height,
        near_plane=near_plane, sh_degree=sh_degree, backgrounds=bg_color, render_mode="RGB+ED",
    )  # [v, h, w, 4]
    return {
        "render": render_colors[..., :3].permute(0, 3, 1, 2),
        "depth": render_colors[..., 3:].permute(0, 3, 1, 2),
    }


class Renderer(nn.Module):
    """Activates the raw Gaussian attributes and splats every context-view Gaussian into each target view."""

    def __init__(self, config):
        super().__init__()
        self.sh_degree = config.model.gaussians.sh_degree
        self.near_plane = config.model.get("near_plane", 0.2)

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    def forward(self, xyz, features, scaling, rotation, opacity, height, width, C2W, C2W_input, fxfycxcy):
        """
        xyz:       [b, n, 3] world-space Gaussian centers
        features:  [b, n, 3, (sh_degree+1)^2]
        scaling, rotation, opacity: [b, n, 3] / [b, n, 4] / [b, n, 1], pre-activation
        C2W:       [b, v, 4, 4] target cameras;  fxfycxcy: [b, v, 4] in pixels
        C2W_input: [b, v_in, 4, 4] context cameras (rotations are predicted in their frames)

        Returns edict(render=[b, v, 3, h, w], depth=[b, v, 1, h, w]).
        """
        b, v = C2W.size(0), C2W.size(1)
        renderings = torch.zeros(b, v, 3, height, width, dtype=torch.float32, device=xyz.device)
        depth = torch.zeros(b, v, 1, height, width, dtype=torch.float32, device=xyz.device)

        scaling = torch.exp(scaling)
        rotation = F.normalize(rotation, dim=-1)
        opacity = torch.sigmoid(opacity)
        c2w_mat = repeat(C2W_input, "b v ... -> b (v h w) ...", h=height, w=width)
        rotation_world = cam_quat_wxyz_to_world_quat_wxyz(rotation, c2w_mat)
        features_world = rearrange(features, "b n shd c -> b n c shd").contiguous()
        for i in range(b):
            buffers = render_opencv_cam_gsplat(
                xyz[i], opacity[i], scaling[i], rotation_world[i], features_world[i],
                height, width, C2W[i], fxfycxcy[i], self.sh_degree, near_plane=self.near_plane,
            )
            renderings[i] = buffers["render"][:, :3]
            depth[i] = buffers["depth"]
        return edict(render=renderings, depth=depth)


def get_point_range_func(gaussians_config):
    """Depth activation: depth = sigmoid(t) * (far - near) + near (`linear_depth`)."""
    range_setting = gaussians_config.range_setting
    assert range_setting.type == "linear_depth", f"unsupported range_setting {range_setting.type}"
    near = range_setting.get("near", 0.0)
    far = range_setting.get("far", 500.0)
    return lambda t: torch.sigmoid(t) * (far - near) + near
