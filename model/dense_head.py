# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the FAIR Noncommercial Research License found in
# licenses/LICENSE_VGGT-Omega.
#
# Adapted from VGGT-Omega (https://github.com/facebookresearch/vggt-omega):
#   vggt_omega/models/heads/dense_head.py  (DenseHead, fusion blocks)
#   vggt_omega/models/heads/utils.py       (create_uv_grid, position_grid_to_embed)
# Modifications: configurable output channels (pixel-aligned Gaussian attributes instead of
# depth + confidence), the confidence branch and frame chunking removed, the two helper
# modules merged into this file.
#
# DPT head inspired by https://github.com/DepthAnything/Depth-Anything-V2

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseHead(nn.Module):
    """DPT head: four transformer layers -> per-pixel outputs (here the Gaussian attributes).

    Each layer's patch tokens are projected and resized to 1/4, 1/8, 1/16 and 1/32 of the image,
    fused coarse-to-fine, decoded at 1/4 scale and pixel-shuffled to full resolution.
    """

    def __init__(
        self,
        dim_in: int = 768,
        patch_size: int = 16,
        features: int = 128,
        out_channels: Tuple[int, ...] = (96, 192, 384, 768),
        out_dim: int = 14,
    ) -> None:
        super().__init__()
        assert patch_size % 4 == 0, f"patch_size must be divisible by 4, got {patch_size}"
        self.patch_size = patch_size
        self.final_shuffle_factor = patch_size // 4
        self.norm = nn.LayerNorm(dim_in, eps=1e-5)

        self.projects = nn.ModuleList([nn.Conv2d(dim_in, oc, kernel_size=1) for oc in out_channels])
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4),   # x4
            nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2),   # x2
            nn.Identity(),                                                                    # x1
            nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),  # x0.5
        ])

        self.scratch = nn.Module()
        for i, oc in enumerate(out_channels):
            setattr(self.scratch, f"layer{i + 1}_rn", nn.Conv2d(oc, features, kernel_size=3, padding=1, bias=False))
        self.scratch.refinenet1 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet4 = FeatureFusionBlock(features, has_residual=False)

        self.proj = nn.Conv2d(features, self.final_shuffle_factor ** 2 * out_dim, kernel_size=1)

    def forward(self, tokens_list: list, images: torch.Tensor) -> torch.Tensor:
        """
        tokens_list: four [b, v, n_patches, dim_in] patch-token tensors
        images:      [b, v, 3, h, w] (only its shape is used)
        returns:     [b, v, h, w, out_dim]
        """
        b, v, _, height, width = images.shape
        patch_h, patch_w = height // self.patch_size, width // self.patch_size

        features = []
        for i, x in enumerate(tokens_list):
            x = self.norm(x.reshape(b * v, -1, x.shape[-1]))
            x = x.permute(0, 2, 1).reshape(b * v, x.shape[-1], patch_h, patch_w)
            x = self.projects[i](x)
            x = self._apply_pos_embed(x, width, height)
            features.append(self.resize_layers[i](x))

        fused = self._apply_pos_embed(self._fuse(features), width, height)
        out = F.pixel_shuffle(self.proj(fused), self.final_shuffle_factor).permute(0, 2, 3, 1)
        return out.view(b, v, *out.shape[1:])

    def _fuse(self, features: list) -> torch.Tensor:
        l1, l2, l3, l4 = (getattr(self.scratch, f"layer{i + 1}_rn")(f) for i, f in enumerate(features))
        out = self.scratch.refinenet4(l4, size=l3.shape[2:])
        out = self.scratch.refinenet3(out, l3, size=l2.shape[2:])
        out = self.scratch.refinenet2(out, l2, size=l1.shape[2:])
        return self.scratch.refinenet1(out, l1, size=l1.shape[2:])

    @staticmethod
    def _apply_pos_embed(x: torch.Tensor, width: int, height: int, ratio: float = 0.1) -> torch.Tensor:
        uv = create_uv_grid(x.shape[-1], x.shape[-2], aspect_ratio=width / height, dtype=x.dtype, device=x.device)
        pos_embed = position_grid_to_embed(uv, x.shape[1]) * ratio
        return x + pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)


class ResidualConvUnit(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.activation = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.activation(self.conv1(self.activation(x)))) + x


class FeatureFusionBlock(nn.Module):
    def __init__(self, features: int, has_residual: bool = True) -> None:
        super().__init__()
        self.out_conv = nn.Conv2d(features, features, kernel_size=1)
        self.has_residual = has_residual
        if has_residual:
            self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, x: torch.Tensor, residual: torch.Tensor = None, size=None) -> torch.Tensor:
        if self.has_residual:
            x = x + self.resConfUnit1(residual)
        x = self.resConfUnit2(x)
        if tuple(x.shape[-2:]) != tuple(size):
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=True)
        return self.out_conv(x)


# ---------------------------------------------------------------------------
# Positional embedding of the feature maps
# ---------------------------------------------------------------------------
def create_uv_grid(width: int, height: int, aspect_ratio: float = None, dtype=None, device=None) -> torch.Tensor:
    """(height, width, 2) UV grid spanning the image plane, normalized by its diagonal
    (inspired by https://github.com/microsoft/moge)."""
    if aspect_ratio is None:
        aspect_ratio = float(width) / float(height)
    diag = (aspect_ratio ** 2 + 1.0) ** 0.5
    span_x, span_y = aspect_ratio / diag, 1.0 / diag
    x = torch.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, steps=width, dtype=dtype, device=device)
    y = torch.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, steps=height, dtype=dtype, device=device)
    uu, vv = torch.meshgrid(x, y, indexing="xy")
    return torch.stack((uu, vv), dim=-1)


def position_grid_to_embed(pos_grid: torch.Tensor, embed_dim: int, omega_0: float = 100) -> torch.Tensor:
    """(H, W, 2) positions -> (H, W, embed_dim) sinusoidal embedding (half x, half y)."""
    H, W, _ = pos_grid.shape
    pos = pos_grid.reshape(-1, 2)
    emb = torch.cat([_sincos_embed(embed_dim // 2, pos[:, 0], omega_0),
                     _sincos_embed(embed_dim // 2, pos[:, 1], omega_0)], dim=-1)
    return emb.view(H, W, embed_dim)


def _sincos_embed(embed_dim: int, pos: torch.Tensor, omega_0: float = 100) -> torch.Tensor:
    omega = torch.arange(embed_dim // 2, dtype=torch.float32 if pos.device.type == "mps" else torch.double, device=pos.device)
    omega = 1.0 / omega_0 ** (omega / (embed_dim / 2.0))
    out = torch.einsum("m,d->md", pos.reshape(-1), omega)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1).float()
