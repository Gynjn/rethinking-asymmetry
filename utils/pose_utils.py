import torch
import torch.nn.functional as F


def rot6d2mat(x):
    """6D rotation representation [B, 6] -> rotation matrix [B, 3, 3] (Zhou et al., CVPR 2019)."""
    a1 = x[:, 0:3]
    a2 = x[:, 3:6]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack((b1, b2, b3), dim=-1)
