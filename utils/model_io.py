import os

import numpy as np
import PIL.Image
import torch

from model.mvssl import MVSSL
from utils.launch import load_config

# Only needed by the training losses; inference builds the model without them.
TRAINING_ONLY = ("reg_vit.", "loss_computer.", "feat_projectors.")


# Released DL3DV-trained weights, stored under the key "model".
RELEASE_URL = "https://huggingface.co/Gynjn/rethinking-asymmetry/resolve/main/model_dl3dv.pt"
RELEASE_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "model_dl3dv.pt")


def resolve_checkpoint(ckpt_path=None):
    """A local checkpoint path, or (when empty) the released weights, downloaded once to checkpoints/."""
    if ckpt_path:
        return ckpt_path
    if not os.path.exists(RELEASE_CACHE):
        os.makedirs(os.path.dirname(RELEASE_CACHE), exist_ok=True)
        print(f"Downloading {RELEASE_URL} -> {RELEASE_CACHE}")
        torch.hub.download_url_to_file(RELEASE_URL, RELEASE_CACHE)
    return RELEASE_CACHE


def load_model(config_path, ckpt_path=None, weights="model", device="cuda"):
    """Inference model from the released weights (ckpt_path empty: model_dl3dv.pt from Hugging Face)
    or from a checkpoint written by train.py (`weights` selects which stored weight set to load)."""
    config = load_config(config_path)
    model = MVSSL(config, for_training=False)
    ckpt = torch.load(resolve_checkpoint(ckpt_path), map_location="cpu", weights_only=True)
    state = ckpt[weights] if weights in ckpt else ckpt["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not missing, f"missing keys: {missing[:5]}"
    assert all(k.startswith(TRAINING_ONLY) for k in unexpected), f"unexpected keys: {unexpected[:5]}"
    return model.to(device).eval(), config


def load_images(paths, size=256, patch_size=16):
    """Resize to height `size` (width rounded to the patch size) and centre-crop to size x size,
    as in training. Returns [v, 3, size, size] in [0, 1]."""
    images = []
    for path in paths:
        image = PIL.Image.open(path).convert("RGB")
        w, h = image.size
        new_w = int(round(int(size / h * w) / patch_size) * patch_size)
        image = image.resize((new_w, size), resample=PIL.Image.LANCZOS)
        left, top = (new_w - min(size, new_w)) // 2, 0
        if new_w < size:   # portrait: crop vertically instead
            image = image.resize((size, int(round(int(size / w * h) / patch_size) * patch_size)),
                                 resample=PIL.Image.LANCZOS)
            left, top = 0, (image.size[1] - size) // 2
        image = image.crop((left, top, left + size, top + size))
        images.append(torch.from_numpy(np.array(image) / 255.0).permute(2, 0, 1).float())
    return torch.stack(images)
