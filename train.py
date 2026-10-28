"""Distributed training.

    torchrun --nproc_per_node 8 train.py --config configs/dl3dv.yaml [key=value ...]
"""
import importlib
import os
import time
from copy import deepcopy

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from data.loaders import build_eval_loader, build_train_loader, set_loader_epoch, uses_curriculum
from utils.camera_eval import evaluate_camera_pose_metrics
from utils.launch import init_config, init_distributed, init_wandb
from utils.training_utils import (auto_resume_job, create_lr_scheduler, create_optimizer, print_rank0,
                                  requires_grad, update_ema)

config = init_config()
os.environ["OMP_NUM_THREADS"] = str(config.training.get("num_threads", 1))
ddp_info = init_distributed(seed=777)
run = init_wandb(config) if ddp_info.is_main_process else None
dist.barrier()

torch.backends.cuda.matmul.allow_tf32 = config.training.use_tf32
torch.backends.cudnn.allow_tf32 = config.training.use_tf32
amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[config.training.amp_dtype]

# Data: one view-count bucket per batch, the same bucket on every rank at a given step
train_dataset, dataloader = build_train_loader(config, world_size=ddp_info.world_size, rank=ddp_info.global_rank)
print_rank0(f"Variable-view training | {dataloader.batch_sampler.describe()}")
dataloader_iter = iter(dataloader)
use_curriculum = uses_curriculum(config)
eval_loaders = []
if ddp_info.is_main_process:
    for entry in config.training.get("eval_datasets", []):
        for v in config.training.get("eval_num_views", [10]):
            eval_loaders.append((f"{entry.get('tag', 'eval')}_v{v}", build_eval_loader(config, entry, v)))

# Model, EMA (dense-context teacher for the context alignment; always stepped, from step 1)
module, class_name = config.model.class_name.rsplit(".", 1)
model = importlib.import_module(module).__dict__[class_name](config).to(ddp_info.device)
ema = deepcopy(model).to(ddp_info.device)
requires_grad(ema, False)
ema.eval()
ema_decay = config.training.get("ema_decay", 0.999)
# The alignment projectors receive no gradient before the alignment ramp starts.
model = DDP(model, device_ids=[ddp_info.local_rank],
            find_unused_parameters=config.training.get("align_loss_weight", 0.0) > 0.0)

optimizer, optimized_params = create_optimizer(
    model, config.training.weight_decay, config.training.lr, (config.training.beta1, config.training.beta2))
optim_param_list = list(optimized_params.values())
total_train_steps = config.training.train_steps
lr_scheduler = create_lr_scheduler(optimizer, total_train_steps, config.training.warmup,
                                   scheduler_type=config.training.get("scheduler_type", "cosine"))
load_path = config.training.get("resume_ckpt", "") or config.training.checkpoint_dir
optimizer, lr_scheduler, cur_train_step, cur_param_update_step = auto_resume_job(
    load_path, model, optimizer, lr_scheduler, config.training.get("reset_training_state", False), ema)

scaler = torch.amp.GradScaler("cuda", enabled=config.training.use_amp and config.training.amp_dtype == "fp16")
dist.barrier()
start_train_step = cur_train_step
model.train()
steps_per_epoch = max(1, len(dataloader))


@torch.no_grad()
def evaluate_pose(eval_model, step):
    eval_model.eval()
    for tag, loader in eval_loaders:
        totals, n = {}, 0
        with torch.autocast(enabled=config.training.use_amp, device_type="cuda", dtype=amp_dtype):
            for batch in loader:
                batch = {k: v.to(ddp_info.device) if torch.is_tensor(v) else v for k, v in batch.items()}
                pred = eval_model(batch, pose_eval=True)[0].float()
                for k, v in evaluate_camera_pose_metrics(batch["c2w"][0], pred, thresholds=[3, 5, 10]).items():
                    totals[k] = totals.get(k, 0.0) + v
                n += 1
        metrics = {f"eval/{tag}/{k}": v / max(n, 1) for k, v in totals.items()}
        print(f"[eval {tag} @ {step}] " + " ".join(f"{k.split('/')[-1]} {v:.3f}" for k, v in metrics.items()))
        if run is not None:
            run.log(metrics, step=step)
    eval_model.train()
    torch.cuda.empty_cache()


while cur_train_step < total_train_steps:
    tic = time.time()
    cur_epoch = cur_train_step // steps_per_epoch
    if cur_train_step % steps_per_epoch == 0:
        if use_curriculum:
            train_dataset.update_iteration(cur_train_step)
            _, dataloader = build_train_loader(config, world_size=ddp_info.world_size, rank=ddp_info.global_rank,
                                               epoch=cur_epoch, dataset=train_dataset)
        else:
            set_loader_epoch(dataloader, cur_epoch)
        dataloader_iter = iter(dataloader)
    try:
        data = next(dataloader_iter)
    except StopIteration:   # bucket draws make epoch lengths vary slightly
        dataloader_iter = iter(dataloader)
        data = next(dataloader_iter)
    batch = {k: v.to(ddp_info.device) if torch.is_tensor(v) else v for k, v in data.items()}

    with torch.autocast(enabled=config.training.use_amp, device_type="cuda", dtype=amp_dtype):
        teacher_feat = None
        if model.module.align_gamma(cur_train_step) > 0.0:
            teacher_feat = ema.teacher_features(batch["image"])
        ret = model(batch, iter=cur_train_step, teacher_feat=teacher_feat)
    loss = ret.loss_metrics.loss
    scaler.scale(loss).backward()
    cur_train_step += 1

    # NaN / exploding-gradient guards; the EMA only follows steps that were actually taken
    skip = bool(torch.isnan(loss) or torch.isinf(loss))
    grad_norm = None
    if skip:
        print(f"NaN or Inf loss at step {cur_train_step}, skipping")
    else:
        scaler.unscale_(optimizer)
        with torch.no_grad():
            for p in optim_param_list:
                if p.grad is not None:
                    p.grad.nan_to_num_(nan=0.0, posinf=1e-6, neginf=-1e-6)
        clip = config.training.grad_clip_norm
        grad_norm = torch.nn.utils.clip_grad_norm_(optim_param_list, max_norm=clip).item()
        allowed = clip * config.training.get("allowed_gradnorm_factor", 5)
        if grad_norm > allowed and cur_train_step > config.training.get("no_pass_steps", -1):
            print(f"Step {cur_train_step}: grad norm {grad_norm:.2f} > {allowed}, skipping optimizer step")
            skip = True
        if not skip:
            scaler.step(optimizer)
            cur_param_update_step += 1
            update_ema(ema, model, decay=ema_decay)
    scaler.update()
    lr_scheduler.step()
    optimizer.zero_grad(set_to_none=True)

    is_ckpt_step = cur_train_step % config.training.checkpoint_every == 0 or cur_train_step == total_train_steps
    # Same value on every rank (eval_loaders only exist on rank 0), so the barrier below matches
    is_eval_step = bool(config.training.get("eval_datasets")) and (
        cur_train_step % config.training.get("eval_every", 10000) == 0 or cur_train_step == total_train_steps)
    if ddp_info.is_main_process:
        losses = {k: float(v) for k, v in ret.loss_metrics.items()}
        if cur_train_step % config.training.print_every == 0 or cur_train_step < start_train_step + 100:
            print(f"[epoch {cur_epoch}] step {cur_train_step} (updates {cur_param_update_step}) | "
                  f"views {batch['image'].shape[1]} x bs {batch['image'].shape[0]}/gpu | "
                  f"{time.time() - tic:.2f}s | lr {optimizer.param_groups[0]['lr']:.2e}\n  "
                  + " | ".join(f"{k} {v:.4f}" for k, v in losses.items()))
        if run is not None and cur_train_step % config.training.get("wandb_log_every", 50) == 0:
            run.log({"lr": optimizer.param_groups[0]["lr"], "grad_norm": grad_norm,
                     "num_views": batch["image"].shape[1], **{f"train/{k}": v for k, v in losses.items()}},
                    step=cur_train_step)
        if is_eval_step:
            evaluate_pose(model.module, cur_train_step)
        if is_ckpt_step:
            os.makedirs(config.training.checkpoint_dir, exist_ok=True)
            path = os.path.join(config.training.checkpoint_dir, f"ckpt_{cur_train_step:016}.pt")
            torch.save({
                "model": model.module.state_dict(),
                "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "fwdbwd_pass_step": cur_train_step,
                "param_update_step": cur_param_update_step,
            }, path)
            print(f"Saved checkpoint to {os.path.abspath(path)}")
    if is_ckpt_step or is_eval_step:
        dist.barrier()

dist.barrier()
dist.destroy_process_group()
