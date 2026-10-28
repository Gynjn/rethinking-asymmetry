# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the FAIR Noncommercial Research License found in
# licenses/LICENSE_Pixio.
#
# From Pixio (https://github.com/facebookresearch/pixio): evaluation/common/pixio.py
# Modifications: forward() returns the final-LayerNorm CLS and patch tokens (no per-block feature extraction).
# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

from functools import partial
from typing import Type

import torch
import torch.nn as nn

from .layers.attention import SelfAttentionBlock
from .layers.mlp import Mlp
from .layers.patch_embed import PatchEmbed

class PixioViT(nn.Module):
    """Pixio ViT encoder"""
    def __init__(
        self,
        img_size: int = 256, 
        patch_size: int = 16, 
        in_chans: int = 3,
        embed_dim: int = 1280, 
        depth: int = 32, 
        num_heads: int = 16,
        mlp_ratio: int = 4., 
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        n_cls_tokens: int = 8
    ):
        super().__init__()
        
        self.n_cls_tokens = n_cls_tokens
        
        self.patch_embed = PatchEmbed(
            img_size, 
            patch_size, 
            in_chans,
            embed_dim
        )
        
        self.cls_token = nn.Parameter(
            torch.zeros(1, n_cls_tokens, embed_dim)
        )
        
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + n_cls_tokens, embed_dim)
        )
        
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(
                embed_dim, 
                num_heads, 
                mlp_ratio, 
                qkv_bias=True, 
                norm_layer=norm_layer, 
                mlp_layer=Mlp
            ) for _ in range(depth)
        ])
        
        self.norm = norm_layer(embed_dim)
        
    def _interpolate_pos_emb(
        self, 
        x: torch.Tensor
    ):
        """Interpolate the pre-trained positional embeddings to match the input x"""
        assert x.shape[-2] % self.patch_embed.patch_size[0] == 0, \
            f'height {x.shape[-2]} must be divisible by patch size {self.patch_embed.patch_size[0]}'
        assert x.shape[-1] % self.patch_embed.patch_size[1] == 0, \
            f'width {x.shape[-1]} must be divisible by patch size {self.patch_embed.patch_size[1]}'
        
        H = x.shape[-2] // self.patch_embed.patch_size[0]
        W = x.shape[-1] // self.patch_embed.patch_size[1]
        
        cls_pos_embed = self.pos_embed[:, :self.n_cls_tokens]
        patch_pos_embed = self.pos_embed[:, self.n_cls_tokens:]
        
        pt_size = int(patch_pos_embed.shape[1] ** 0.5)
        
        if pt_size == H == W:
            return self.pos_embed
        
        patch_pos_embed = patch_pos_embed.reshape(1, pt_size, pt_size, -1).permute(0, 3, 1, 2)
        patch_pos_embed = torch.nn.functional.interpolate(
            patch_pos_embed, size=(H, W), mode='bicubic', align_corners=False
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, H * W, -1)
        
        new_pos_embed = torch.cat((cls_pos_embed, patch_pos_embed), dim=1)
        
        return new_pos_embed
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, 3, H, W] ImageNet-normalized images.

        Returns:
            (cls_tokens [B, n_cls, D], patch_tokens [B, N, D]), both after the final LayerNorm.
        """
        pos_embed = self._interpolate_pos_emb(x)
        x = self.patch_embed(x)
        x = x + pos_embed[:, self.n_cls_tokens:, :]

        cls_token = self.cls_token + pos_embed[:, :self.n_cls_tokens, :]
        x = torch.cat((cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, :self.n_cls_tokens], x[:, self.n_cls_tokens:]


def pixio_vitb16(pretrained=None):
    model = PixioViT(
        img_size=256,
        patch_size=16, 
        embed_dim=768, 
        depth=12, 
        num_heads=12,
        mlp_ratio=4, 
        n_cls_tokens=8,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )

    if pretrained:
        state_dict = torch.load(pretrained, map_location='cpu', weights_only=False)
        model.load_state_dict(state_dict)
    
    return model
