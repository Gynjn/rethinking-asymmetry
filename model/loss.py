import os
from pathlib import Path

import scipy.io
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19

VGG_URL = "https://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat"


def cosine_alignment_loss(z_student, z_target):
    """Negative cosine similarity between token sets matched token-by-token ([..., n, d]).

    Returns (loss, mean cosine, token spread). The spread (std across tokens of the normalized
    student features) is a collapse diagnostic: it goes to 0 if every token maps to one direction.
    """
    z_s = F.normalize(z_student, dim=-1)
    z_t = F.normalize(z_target.detach(), dim=-1)
    cos = (z_s * z_t).sum(dim=-1)
    return -cos.mean(), cos.mean().detach(), z_s.std(dim=-2).mean().detach()


# The perceptual loss is modified from
# https://github.com/zhengqili/Crowdsampling-the-Plenoptic-Function/blob/f5216f312cf82d77f8d20454b5eeb3930324630a/models/networks.py#L1478
# and https://github.com/arthurhero/Long-LRM/blob/main/model/loss.py
class PerceptualLoss(nn.Module):
    """VGG-19 feature-matching loss (Chen & Koltun, 2017) on MatConvNet weights."""

    def __init__(self, weight_path):
        super().__init__()
        self.weight_path = weight_path
        self.vgg = self._build_vgg()
        self._load_weights()
        self._setup_feature_blocks()

    def _build_vgg(self):
        model = vgg19()
        for i, layer in enumerate(model.features):   # average pooling instead of max pooling
            if isinstance(layer, nn.MaxPool2d):
                model.features[i] = nn.AvgPool2d(kernel_size=2, stride=2)
        return model.eval()

    def _load_weights(self):
        weight_file = Path(self.weight_path)
        weight_file.parent.mkdir(exist_ok=True, parents=True)
        if not dist.is_initialized() or dist.get_rank() == 0:
            if not weight_file.exists():
                os.system(f"wget {VGG_URL} -O {weight_file}")
        if dist.is_initialized():
            dist.barrier()

        vgg_layers = scipy.io.loadmat(weight_file)["layers"][0]
        layer_indices = [0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]
        filter_sizes = [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512]
        with torch.no_grad():
            for i, layer_idx in enumerate(layer_indices):
                weights = torch.from_numpy(vgg_layers[layer_idx][0][0][2][0][0]).permute(3, 2, 0, 1)
                self.vgg.features[layer_idx].weight = nn.Parameter(weights, requires_grad=False)
                biases = torch.from_numpy(vgg_layers[layer_idx][0][0][2][0][1]).view(filter_sizes[i])
                self.vgg.features[layer_idx].bias = nn.Parameter(biases, requires_grad=False)

    def _setup_feature_blocks(self):
        output_indices = [0, 4, 9, 14, 23, 32]
        self.blocks = nn.ModuleList()
        for i in range(len(output_indices) - 1):
            block = nn.Sequential(*list(self.vgg.features[output_indices[i]:output_indices[i + 1]]))
            self.blocks.append(block.eval())
        for param in self.vgg.parameters():
            param.requires_grad = False

    def _extract_features(self, x):
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features

    @staticmethod
    def _preprocess_images(images):
        mean = torch.tensor([123.6800, 116.7790, 103.9390]).reshape(1, 3, 1, 1).to(images.device)
        return images * 255.0 - mean

    @staticmethod
    def _error(real, fake):
        return torch.abs(real - fake).mean(dim=1, keepdim=True).mean()

    def forward(self, pred_img, target_img):
        """pred_img, target_img: [N, 3, H, W] in [0, 1]."""
        target_p = self._preprocess_images(target_img)
        pred_p = self._preprocess_images(pred_img)
        tf = self._extract_features(target_p)
        pf = self._extract_features(pred_p)

        e0 = self._error(target_p, pred_p)
        e1 = self._error(tf[0], pf[0]) / 2.6
        e2 = self._error(tf[1], pf[1]) / 4.8
        e3 = self._error(tf[2], pf[2]) / 3.7
        e4 = self._error(tf[3], pf[3]) / 5.6
        e5 = self._error(tf[4], pf[4]) * 10 / 1.5
        return (e0 + e1 + e2 + e3 + e4 + e5) / 255.0


class LossComputer(nn.Module):
    """Frozen loss networks (kept as a module so checkpoint keys stay `loss_computer.*`)."""

    def __init__(self, config):
        super().__init__()
        self.perceptual_loss_module = PerceptualLoss(weight_path=config.training.vgg_ckpt)
        self.perceptual_loss_module.eval()
        for param in self.perceptual_loss_module.parameters():
            param.requires_grad = False
