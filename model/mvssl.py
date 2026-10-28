"""Self-supervised multi-view geometry model.

Pixio encoder -> 12-layer multi-view transformer (alternating view-wise / global attention)
-> camera head (camera tokens) + DPT head (pixel-aligned Gaussians).

Asymmetry lives in the 3D representation, not in attention: all views (context and held-out)
attend to each other, but the Gaussian scene is built from the context views only and rendered
at the predicted held-out poses.

Training objective (paper Eq. 5):
    L = L_nvs + lambda_align * L_align + lambda_sem * L_sem
  L_nvs    MSE + perceptual loss on the held-out renderings
  L_align  context alignment: an EMA copy sees all views, the online network a symmetric
           subset; per-view patch tokens at the DPT layers are matched with a cosine loss
           through a per-layer projector (online side only)
  L_sem    semantic alignment: encoder patch tokens matched to a frozen Pixio's (cosine)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from easydict import EasyDict as edict
from einops import rearrange

from utils.pose_utils import rot6d2mat
from .dense_head import DenseHead
from .layers.attention import SelfAttentionBlock
from .layers.mlp import Mlp
from .loss import LossComputer, cosine_alignment_loss
from .pixio import pixio_vitb16
from .renderer import Renderer, get_point_range_func
from .rope import PositionGetter, RotaryPositionEmbedding2D


def _init_weights(module, std=0.02):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        if isinstance(module, nn.Linear) and module.bias is not None:
            torch.nn.init.zeros_(module.bias)


def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
        nn.Linear(hidden_size, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, z_dim),
    )


class MVSSL(nn.Module):
    def __init__(self, config, for_training=True):
        """for_training=False skips the modules only the losses need (frozen Pixio target,
        VGG perceptual network, alignment projectors), so inference needs no extra weights."""
        super().__init__()
        self.config = config
        self.for_training = for_training
        self.embed_dim = config.model.transformer.d
        self.num_heads = config.model.transformer.d // config.model.transformer.d_head
        self.patch_size = config.model.image_tokenizer.patch_size
        self.patch_start_idx = 1    # one camera token per view
        self.used_by_DPT_idx = list(config.model.transformer.used_by_DPT_idx)
        # Gaussian channels: 3 (depth logit, averaged) + SH colour + 3 scale + 4 rotation + 1 opacity
        self.gs_channels = 3 + (config.model.gaussians.sh_degree + 1) ** 2 * 3 + 3 + 4 + 1

        pixio_ckpt = config.training.get("pixio_ckpt", None) if for_training else None
        self.vit = pixio_vitb16(pretrained=pixio_ckpt)
        if for_training:
            # Frozen copy of the pretrained encoder: target of the semantic alignment.
            self.reg_vit = pixio_vitb16(pretrained=pixio_ckpt)
            for param in self.reg_vit.parameters():
                param.requires_grad = False
            self.reg_vit.eval()

        self.pose_predictor = PoseEstimator(config)
        self.dense_head = DenseHead(out_dim=self.gs_channels)

        # camera_token[:, 0] marks the canonical (middle) view, camera_token[:, 1] every other view
        self.camera_token = nn.Parameter(torch.zeros(1, 2, self.embed_dim))
        nn.init.normal_(self.camera_token, std=1e-6)
        self.rope = RotaryPositionEmbedding2D()
        self.position_getter = PositionGetter()
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(
                self.embed_dim,
                self.num_heads,
                qkv_bias=True,
                norm_layer=nn.LayerNorm,
                mlp_layer=Mlp,
                qk_norm=True,
                rope=self.rope if idx % 2 == 0 else None,   # RoPE on the view-wise layers
            ) for idx in range(config.model.transformer.n_layer)
        ])
        self.blocks.apply(_init_weights)

        self.range_func = get_point_range_func(config.model.gaussians)
        self.renderer = Renderer(config)
        self.scaling_bias = config.model.gaussians.get("scaling_bias", -2.3)
        self.scaling_max = config.model.gaussians.get("scaling_max", -1.2)
        self.opacity_bias = config.model.gaussians.get("opacity_bias", -2.0)
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        tc = config.training
        if for_training:
            self.loss_computer = LossComputer(config)

        # Context alignment (L_align)
        self.align_loss_weight = tc.get("align_loss_weight", 0.0)
        self.align_warmup_start = tc.get("align_warmup_start", 0)
        self.align_warmup_iter = tc.get("align_warmup_iter", 0)
        self.align_layer_idx = sorted({int(i) for i in tc.get("align_layer_idx", self.used_by_DPT_idx)})
        self.align_layer_set = set(self.align_layer_idx)
        self.num_views_buckets = [int(x) for x in tc.get("num_views_buckets", None) or []]
        self.sparse_views_buckets = [int(x) for x in tc.get("sparse_views_buckets", None) or []]
        assert not self.sparse_views_buckets or len(self.sparse_views_buckets) == len(self.num_views_buckets), \
            "sparse_views_buckets must align index-by-index with num_views_buckets"
        for v_all, v_few in zip(self.num_views_buckets, self.sparse_views_buckets):
            assert 1 <= v_few <= v_all and (v_all - v_few) % 2 == 0, \
                f"sparse view count {v_few} for bucket {v_all} must drop an even number of views"
        if for_training and self.align_loss_weight > 0.0:
            # One projector per aligned layer, applied to the online (sparse) side only.
            self.feat_projectors = nn.ModuleList([build_mlp(768, 2048, 768) for _ in self.align_layer_idx])
            self.feat_projectors.apply(_init_weights)
        else:
            self.feat_projectors = None

        # Semantic alignment (L_sem)
        self.sem_loss_weight = tc.get("sem_loss_weight", 0.0) if for_training else 0.0

    def train(self, mode=True):
        super().train(mode)
        if self.for_training:   # frozen modules stay in eval mode
            self.loss_computer.eval()
            self.reg_vit.eval()
        return self

    # ------------------------------------------------------------------
    # Backbone
    # ------------------------------------------------------------------
    def _prepare_rope(self, B, V, H, W, device):
        pos = self.position_getter(B * V, H // self.patch_size, W // self.patch_size, device=device)
        pos = rearrange(pos, "(b v) n c -> b v n c", b=B) + 1
        pos_special = torch.zeros(B, V, self.patch_start_idx, 2, device=device, dtype=pos.dtype)
        return torch.cat([pos_special, pos], dim=2)

    def process_attention(self, x, block, attn_type="global", pos=None):
        b, s = x.shape[:2]
        if attn_type == "local":
            x = rearrange(x, "b s n c -> (b s) n c")
            if pos is not None:
                pos = rearrange(pos, "b s n c -> (b s) n c")
            x = block(x, pos=pos)
            return rearrange(x, "(b s) n c -> b s n c", b=b, s=s)
        x = rearrange(x, "b s n c -> b (s n) c")
        x = block(x, pos=None)
        return rearrange(x, "b (s n) c -> b s n c", b=b, s=s)

    def encode(self, image):
        """image [b, v, 3, h, w] in [0, 1] -> (normalized image [b*v, 3, h, w], patch tokens [b, v, n, d])."""
        b, v = image.shape[:2]
        image_norm = self.normalize(image.flatten(0, 1))
        _, img_tokens = self.vit(image_norm)
        return image_norm, rearrange(img_tokens, "(b v) n d -> b v n d", b=b, v=v)

    def run_trunk(self, img_tokens, h, w, stop_after=None):
        """Multi-view transformer over [b, v, n, d] patch tokens.

        The canonical view is v // 2 (camera_token[0]); every other view gets camera_token[1].
        Returns (per-layer patch tokens {block_idx: [b, v, n, d]} for the DPT / aligned layers,
        camera tokens of the last block [b, v, d]). `stop_after` ends the pass early at that block.
        """
        b, v = img_tokens.shape[:2]
        middle = v // 2
        ref_cam = self.camera_token[:, :1].expand(b, -1, -1)
        src_cam = self.camera_token[:, 1:].expand(b, v - 1, -1)
        cam_tokens = torch.cat([src_cam[:, :middle], ref_cam, src_cam[:, middle:]], dim=1)
        x = torch.cat([cam_tokens.unsqueeze(2), img_tokens], dim=2)
        pos = self._prepare_rope(b, v, h, w, img_tokens.device)

        taps = {}
        keep = set(self.used_by_DPT_idx) | self.align_layer_set
        for i, blk in enumerate(self.blocks):
            if i % 2 == 0:
                x = self.process_attention(x, blk, attn_type="local", pos=pos)
            else:
                x = self.process_attention(x, blk, attn_type="global")
            if i in keep:
                taps[i] = x[..., self.patch_start_idx:, :]
            if stop_after is not None and i == stop_after:
                break
        return taps, x[..., 0, :]

    def to_gs(self, gaussians):
        """gaussians [b, n, c] -> (depth logit [b, n, 3], sh [b, n, 3, k], scale, rotation, opacity)."""
        sh_k = (self.config.model.gaussians.sh_degree + 1) ** 2
        xyz, features, scaling, rotation, opacity = gaussians.split([3, sh_k * 3, 3, 4, 1], dim=2)
        xyz = xyz.clamp(-500.0, 500.0)
        features = features.reshape(features.size(0), features.size(1), 3, sh_k).contiguous()
        scaling = (scaling + self.scaling_bias).clamp(max=self.scaling_max).clamp(min=-10.0)
        opacity = (opacity + self.opacity_bias).clamp(min=-10.0)
        return xyz, features, scaling, rotation, opacity

    def decode_gaussians(self, taps, image, view_idx=None):
        """DPT head on (a subset of) the views. Returns ray depth [b, v', 1, h, w] and Gaussian attributes."""
        b = image.shape[0]
        h, w = image.shape[-2:]
        feats = [taps[i] for i in self.used_by_DPT_idx]
        if view_idx is not None:
            batch_idx = torch.arange(b, device=image.device).unsqueeze(1)
            feats = [f[batch_idx, view_idx] for f in feats]
            image = image[batch_idx, view_idx]
        v = image.shape[1]
        gaussians = rearrange(self.dense_head(feats, image), "b v h w c -> b (v h w) c")
        xyz, features, scaling, rotation, opacity = self.to_gs(gaussians)
        depth_logit = rearrange(xyz, "b (v h w) c -> b v c h w", v=v, h=h, w=w).mean(dim=2, keepdim=True)
        depth = self.range_func(depth_logit)    # distance along the unit ray
        return depth, features, scaling, rotation, opacity

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, image, context_idx=None):
        """
        image:       [b, v, 3, h, w] in [0, 1], unposed
        context_idx: [b, v'] views to decode depth for (default: all views)

        Returns edict with
          c2w       [b, v, 4, 4]  cameras relative to the middle view (index v // 2)
          fxfycxcy  [b, v, 4]     intrinsics normalized by width / height
          depth     [b, v', 1, h, w]  depth along the unit ray (`ray_to_z` converts to z-depth)
        """
        b, v, _, h, w = image.shape
        if context_idx is None:
            context_idx = torch.arange(v, device=image.device).unsqueeze(0).expand(b, -1)
        _, img_tokens = self.encode(image)
        taps, cam_tokens = self.run_trunk(img_tokens, h, w)
        c2w, fxfycxcy = get_cam_se3(self.pose_predictor(cam_tokens, v))
        depth = self.decode_gaussians(taps, image, context_idx)[0]
        return edict(
            c2w=rearrange(c2w, "(b v) c d -> b v c d", b=b),
            fxfycxcy=rearrange(fxfycxcy, "(b v) c -> b v c", b=b),
            depth=depth,
        )

    @torch.no_grad()
    def teacher_features(self, image):
        """Dense-context alignment targets (run on the EMA copy): patch tokens at each aligned layer, all views."""
        h, w = image.shape[-2:]
        _, img_tokens = self.encode(image)
        taps, _ = self.run_trunk(img_tokens, h, w, stop_after=self.align_layer_idx[-1])
        return [taps[i] for i in self.align_layer_idx]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def align_gamma(self, step):
        """lambda_align at `step`: 0 before align_warmup_start, then a linear ramp."""
        if self.align_loss_weight <= 0.0 or step < self.align_warmup_start:
            return 0.0
        if self.align_warmup_iter <= 0:
            return self.align_loss_weight
        progress = (step - self.align_warmup_start) / self.align_warmup_iter
        return self.align_loss_weight * min(1.0, max(0.0, progress))

    def sparse_view_count(self, v_all):
        for va, vf in zip(self.num_views_buckets, self.sparse_views_buckets):
            if va == v_all:
                return vf
        return max(1, v_all - 2)

    def sparse_view_indices(self, b, v_all, device):
        """Per-scene sparse subset [b, v_few], sorted. The canonical view v_all // 2 and the outermost
        views 0 and v_all - 1 are always kept, so the subset spans the full baseline of the dense pass;
        the same number of interior views is dropped at random on each side. The canonical view lands
        at v_few // 2 of the subset and every view keeps the camera token it had in the dense pass."""
        cano = v_all // 2
        v_few = self.sparse_view_count(v_all)
        k_left = v_few // 2                 # views kept left of the canonical one (incl. view 0)
        k_right = v_few - 1 - k_left        # views kept right of it (incl. view v_all - 1)
        n_right = v_all - cano - 1
        assert (k_left <= cano and k_right <= n_right and cano - k_left == n_right - k_right) or v_few == 1
        cano_col = torch.full((b, 1), cano, device=device, dtype=torch.long)
        if v_few == 1:
            return cano_col
        assert k_left >= 1 and k_right >= 1, f"sparse view count {v_few} cannot keep both outermost views"
        first_col = torch.zeros((b, 1), device=device, dtype=torch.long)
        last_col = torch.full((b, 1), v_all - 1, device=device, dtype=torch.long)
        # argsort of uniform noise = an independent random permutation per row; the remaining picks
        # come from the interior candidates 1..cano-1 and cano+1..v_all-2
        left = torch.rand(b, cano - 1, device=device).argsort(dim=1)[:, :k_left - 1] + 1
        right = torch.rand(b, n_right - 1, device=device).argsort(dim=1)[:, :k_right - 1] + cano + 1
        return torch.cat([first_col, left, cano_col, right, last_col], dim=1).sort(dim=1).values

    def forward(self, data, pose_eval=False, iter=0, teacher_feat=None):
        """
        data: {"image": [b, v, 3, h, w], "context_indices": [b, v_in], "target_indices": [b, v_tgt]}
        pose_eval=True returns only the predicted c2w [b, v, 4, 4].
        teacher_feat: dense-context targets from `ema.teacher_features(data["image"])`.
        """
        image = data["image"]
        b, v_all, _, h, w = image.shape
        device = image.device
        input_idx = data["context_indices"].to(device)
        target_idx = data["target_indices"].to(device)
        batch_idx = torch.arange(b, device=device).unsqueeze(1)
        target_image = image[batch_idx, target_idx]

        image_norm, img_tokens = self.encode(image)
        taps, cam_tokens = self.run_trunk(img_tokens, h, w)
        c2w, fxfycxcy = get_cam_se3(self.pose_predictor(cam_tokens, v_all))
        c2w = rearrange(c2w, "(b v) c d -> b v c d", b=b, v=v_all)
        if pose_eval:
            return c2w
        fxfycxcy = rearrange(fxfycxcy, "(b v) c -> b v c", b=b, v=v_all)

        # Gaussians from the context views only, lifted along the predicted rays
        c2w_input, fxfycxcy_input = c2w[batch_idx, input_idx], fxfycxcy[batch_idx, input_idx]
        c2w_target, fxfycxcy_target = c2w[batch_idx, target_idx], fxfycxcy[batch_idx, target_idx]
        canonical_depth, features, scaling, rotation, opacity = self.decode_gaussians(taps, image, input_idx)
        rays = cam_info_to_plucker(c2w_input, fxfycxcy_input, self.config.model.image_tokenizer)
        ray_o, ray_d = rearrange(rays, "(b v) c h w -> b v c h w", b=b).split([3, 3], dim=2)
        xyz_world = rearrange(ray_o + ray_d * canonical_depth, "b v c h w -> b (v h w) c")
        fxfycxcy_target = fxfycxcy_target * torch.tensor([w, h, w, h], device=device, dtype=fxfycxcy_target.dtype)

        render = self.renderer(
            xyz_world, features, scaling, rotation, opacity, h, w,
            C2W=c2w_target, C2W_input=c2w_input, fxfycxcy=fxfycxcy_target,
        )

        # L_nvs: MSE + perceptual on the held-out views
        loss_metrics = edict(loss=0.0)
        err = ((render.render - target_image) ** 2).mean(dim=2, keepdim=True)
        loss_mse = err.mean()
        loss_metrics.psnr = -10.0 * torch.log10(loss_mse.clamp(min=1e-8))
        loss_metrics.loss_mse = loss_mse
        loss_metrics.loss = loss_metrics.loss + self.config.training.get("mse_loss_weight", 1.0) * loss_mse
        perc_w = self.config.training.get("perceptual_loss_weight", 0.0)
        if perc_w > 0.0:
            loss_perc = self.loss_computer.perceptual_loss_module(
                render.render.reshape(-1, 3, h, w), target_image.reshape(-1, 3, h, w))
            loss_metrics.loss_perc = loss_perc
            loss_metrics.loss = loss_metrics.loss + perc_w * loss_perc

        # L_align: sparse-context online features -> dense-context EMA features
        gamma = self.align_gamma(iter)
        if self.training and gamma > 0.0:
            assert teacher_feat is not None, "context alignment needs the EMA teacher's features"
            few_idx = self.sparse_view_indices(b, v_all, device)
            student_taps, _ = self.run_trunk(img_tokens[batch_idx, few_idx], h, w,
                                             stop_after=self.align_layer_idx[-1])
            terms = []
            for j, layer in enumerate(self.align_layer_idx):
                l_j, cos_j, spread_j = cosine_alignment_loss(
                    self.feat_projectors[j](student_taps[layer]), teacher_feat[j][batch_idx, few_idx])
                terms.append(l_j)
                loss_metrics[f"align_cos_l{layer}"] = cos_j
                loss_metrics[f"align_spread_l{layer}"] = spread_j
            loss_align = torch.stack(terms).mean()
            loss_metrics.loss_align = loss_align
            loss_metrics.align_gamma = torch.as_tensor(gamma, device=device)
            loss_metrics.sparse_views = torch.as_tensor(float(few_idx.shape[1]), device=device)
            loss_metrics.loss = loss_metrics.loss + gamma * loss_align

        # L_sem: encoder patch tokens of the context views -> frozen Pixio
        if self.sem_loss_weight > 0.0:
            image_in = rearrange(rearrange(image_norm, "(b v) c h w -> b v c h w", b=b)[batch_idx, input_idx],
                                 "b v c h w -> (b v) c h w")
            with torch.no_grad():
                _, sem_target = self.reg_vit(image_in)
            tokens_in = rearrange(img_tokens[batch_idx, input_idx], "b v n d -> (b v) n d")
            loss_sem = -(F.normalize(tokens_in, dim=-1) * F.normalize(sem_target, dim=-1)).sum(dim=-1).mean()
            loss_metrics.loss_sem = loss_sem
            loss_metrics.loss = loss_metrics.loss + self.sem_loss_weight * loss_sem

        return edict(
            loss_metrics=loss_metrics,
            render=render.render,
            target_image=target_image,
            input_image=image[batch_idx, input_idx],
            canonical_depth=canonical_depth,
            c2w=c2w,
            fxfycxcy=fxfycxcy,
            input_idx=input_idx,
            target_idx=target_idx,
        )


# ----------------------------------------------------------------------
# Camera utilities
# ----------------------------------------------------------------------
def get_cam_se3(cam_info):
    """cam_info [n, 13] = 6D rotation, translation, normalized fxfycxcy -> (c2w [n, 4, 4], fxfycxcy [n, 4])."""
    n = cam_info.shape[0]
    R = rot6d2mat(cam_info[:, :6])
    t = cam_info[:, 6:9].unsqueeze(-1)
    fxfycxcy = cam_info[:, 9:]
    bottom = torch.tensor([0, 0, 0, 1], dtype=R.dtype, device=R.device).view(1, 1, 4).repeat(n, 1, 1)
    return torch.cat([torch.cat([R, t], dim=2), bottom], dim=1), fxfycxcy


def cam_info_to_plucker(c2w, fxfycxcy, imgs_info):
    """Per-pixel ray origins and unit directions. c2w [b, v, 4, 4], normalized fxfycxcy [b, v, 4]
    -> [b*v, 6, h, w] (origin, direction)."""
    c2w = rearrange(c2w, "b v n d -> (b v) n d")
    fxfycxcy = rearrange(fxfycxcy, "b v d -> (b v) d").clone()
    n = c2w.shape[0]
    h, w = imgs_info.image_size_h, imgs_info.image_size_w
    fxfycxcy[:, 0] *= w
    fxfycxcy[:, 1] *= h
    fxfycxcy[:, 2] *= w
    fxfycxcy[:, 3] *= h

    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    y, x = y.to(c2w), x.to(c2w)
    x = x[None].expand(n, -1, -1).reshape(n, -1)
    y = y[None].expand(n, -1, -1).reshape(n, -1)
    x = (x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]
    y = (y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]
    ray_d = torch.stack([x, y, torch.ones_like(x)], dim=2)
    ray_d = torch.bmm(ray_d, c2w[:, :3, :3].transpose(1, 2))
    ray_d = ray_d / torch.norm(ray_d, dim=2, keepdim=True)
    ray_o = c2w[:, :3, 3][:, None, :].expand_as(ray_d)
    ray_o = ray_o.reshape(n, h, w, 3).permute(0, 3, 1, 2)
    ray_d = ray_d.reshape(n, h, w, 3).permute(0, 3, 1, 2)
    return torch.cat([ray_o, ray_d], dim=1)


def ray_to_z(ray_depth, fxfycxcy):
    """Depth along the unit ray [..., v, 1, h, w] -> planar z-depth, using normalized fxfycxcy [..., v, 4]."""
    h, w = ray_depth.shape[-2:]
    k = fxfycxcy * torch.tensor([w, h, w, h], device=fxfycxcy.device, dtype=fxfycxcy.dtype)
    yy, xx = torch.meshgrid(torch.arange(h, device=k.device, dtype=k.dtype),
                            torch.arange(w, device=k.device, dtype=k.dtype), indexing="ij")
    x = (xx + 0.5 - k[..., 2, None, None]) / k[..., 0, None, None]
    y = (yy + 0.5 - k[..., 3, None, None]) / k[..., 1, None, None]
    return ray_depth / torch.sqrt(x ** 2 + y ** 2 + 1.0).unsqueeze(-3)


class PoseEstimator(nn.Module):
    """Camera head: each view's pose relative to the middle (canonical) view, plus one shared focal length."""

    def __init__(self, config):
        super().__init__()
        d = config.model.transformer.d
        self.num_pose_element = 6   # 6D rotation

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=1e-3)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.rel_head = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, self.num_pose_element + 3))
        self.rel_head.apply(init_weights)
        self.canonical_k_head = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1))
        self.canonical_k_head.apply(init_weights)
        self.f_bias = 1.25

    def forward(self, x, v):
        """x: camera tokens [b, v, d] -> [b*v, 13] (6D rotation, translation, fx, fy, cx, cy)."""
        b = x.shape[0]
        cano_idx = v // 2
        rel_indices = torch.cat([torch.arange(cano_idx), torch.arange(cano_idx + 1, v)])
        x_canonical = x[:, cano_idx:cano_idx + 1]
        x_rel = x[:, rel_indices]

        fxfy = (self.canonical_k_head(x_canonical[:, 0]) + self.f_bias).unsqueeze(1).repeat(1, 1, 2)   # [b, 1, 2]
        rt_canonical = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 0]).reshape(1, 1, 9).to(fxfy).repeat(b, 1, 1)
        info_all = torch.cat([rt_canonical, fxfy], dim=-1).repeat(1, v, 1)                             # [b, v, 11]
        info_all[:, rel_indices, :self.num_pose_element + 3] += self.rel_head(x_rel)

        cxcy = torch.tensor([0.5, 0.5]).reshape(1, 1, 2).repeat(b, v, 1).to(info_all)
        return rearrange(torch.cat([info_all, cxcy], dim=-1), "b v d -> (b v) d")
