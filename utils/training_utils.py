import os
import traceback

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup


def print_rank0(*args, **kwargs):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)


def create_optimizer(model, weight_decay, learning_rate, betas):
    """AdamW; no weight decay on 1-D parameters (norms, biases, tokens)."""
    params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    decay = [p for p in params.values() if not (p.dim() == 1 or getattr(p, "_no_weight_decay", False))]
    nodecay = [p for p in params.values() if p.dim() == 1 or getattr(p, "_no_weight_decay", False)]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": nodecay, "weight_decay": 0.0}],
        lr=learning_rate, betas=betas, fused=False,
    )
    total = sum(p.numel() for p in model.parameters())
    print_rank0(f"AdamW lr {learning_rate} wd {weight_decay} betas {betas} | "
                f"{sum(p.numel() for p in params.values()) / 1e6:.1f}M trainable / {total / 1e6:.1f}M total")
    return optimizer, params


def create_lr_scheduler(optimizer, param_update_steps, warm_up_steps, scheduler_type="cosine"):
    if scheduler_type == "linear":
        return get_linear_schedule_with_warmup(optimizer, warm_up_steps, param_update_steps)
    if scheduler_type == "cosine":
        return get_cosine_schedule_with_warmup(optimizer, warm_up_steps, param_update_steps)
    if scheduler_type == "constant":
        return get_constant_schedule_with_warmup(optimizer, warm_up_steps)
    raise ValueError(f"Invalid scheduler type: {scheduler_type}")


def find_checkpoints(load_path):
    if os.path.isdir(load_path):
        return [os.path.join(load_path, f) for f in sorted(os.listdir(load_path)) if f.endswith(".pt")]
    return [load_path] if load_path.endswith(".pt") else []


@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    """ema = decay * ema + (1 - decay) * model for parameters; buffers are copied.
    Call only after an optimizer step that actually happened."""
    model = model.module if hasattr(model, "module") else model
    ema_params = dict(ema_model.named_parameters())
    for name, param in model.named_parameters():
        ema_params[name].mul_(decay).add_(param.detach(), alpha=1 - decay)
    ema_buffers = dict(ema_model.named_buffers())
    for name, buf in model.named_buffers():
        if name in ema_buffers:
            ema_buffers[name].copy_(buf)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def auto_resume_job(load_path, model, optimizer, lr_scheduler, reset_training_state, ema=None):
    """Resume from `load_path` (a .pt file, or the latest .pt in a directory).
    Returns (optimizer, lr_scheduler, fwdbwd_pass_step, param_update_step)."""
    ckpt_paths = find_checkpoints(load_path)
    if not ckpt_paths:
        print_rank0(f"No checkpoint found in {load_path}, starting from scratch")
        return optimizer, lr_scheduler, 0, 0
    ckpt_path = ckpt_paths[-1]
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    module = model.module if isinstance(model, DDP) else model
    print_rank0(f"Loaded model from {ckpt_path}: {module.load_state_dict(checkpoint['model'], strict=False)}")
    if ema is not None:
        if "ema" in checkpoint:
            print_rank0(f"Loaded EMA from {ckpt_path}: {ema.load_state_dict(checkpoint['ema'], strict=False)}")
        else:
            update_ema(ema, model, decay=0.0)   # no EMA state: start the EMA from the loaded model
            print_rank0("No EMA state in checkpoint; EMA initialized from the loaded model")

    fwd_step, update_step = 0, 0
    if not reset_training_state:
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            fwd_step = checkpoint["fwdbwd_pass_step"]
            update_step = checkpoint["param_update_step"]
            print_rank0(f"Resumed optimizer and lr_scheduler from {ckpt_path} at step {fwd_step}")
        except Exception:
            traceback.print_exc()
            print_rank0(f"Failed to load optimizer / lr_scheduler from {ckpt_path}")
    return optimizer, lr_scheduler, fwd_step, update_step
