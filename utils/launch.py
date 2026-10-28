"""Config loading (YAML + `key=value` CLI overrides), torch.distributed setup and wandb."""
import argparse
import copy
import datetime
import os
import random
import re

import numpy as np
import torch
import torch.distributed as dist
import yaml
from easydict import EasyDict as edict
from omegaconf import OmegaConf


def _process_overrides(overrides):
    """Accept `key = value` as well as `key=value`."""
    fixed = re.sub(r'(\S+)\s*=\s*(\S+)', r'\1=\2', ' '.join(overrides))
    return re.findall(r'[^\s=]+=\S+|\S+', fixed)


def load_config(path, overrides=()):
    config = OmegaConf.load(path)
    config = OmegaConf.merge(config, OmegaConf.from_cli(_process_overrides(list(overrides))))
    return edict(OmegaConf.to_container(config, resolve=True))


def init_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("overrides", nargs="*", help="key=value config overrides")
    args = parser.parse_args()
    return load_config(args.config, args.overrides)


def init_distributed(seed=42):
    """Init NCCL from the torchrun environment and seed every RNG with seed + rank."""
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    process_seed = seed + global_rank
    torch.manual_seed(process_seed)
    torch.cuda.manual_seed_all(process_seed)
    np.random.seed(process_seed)
    random.seed(process_seed)
    torch.backends.cudnn.benchmark = True
    return edict(local_rank=local_rank, global_rank=global_rank, world_size=world_size,
                 device=device, is_main_process=global_rank == 0, seed=process_seed)


def init_wandb(config):
    """Log to wandb if `training.wandb_project` is set (key from `training.api_key_path` or WANDB_API_KEY);
    always save the resolved config next to the checkpoints."""
    os.makedirs(config.training.checkpoint_dir, exist_ok=True)
    with open(os.path.join(config.training.checkpoint_dir, "config.yaml"), "w") as f:
        yaml.dump(dict(config), f)
    if not config.training.get("wandb_project"):
        return None
    import wandb
    key_path = config.training.get("api_key_path", "")
    if key_path and os.path.exists(key_path):
        key = (yaml.safe_load(open(key_path)) or {}).get("wandb")
        if key and key != "YOUR_WANDB_API_KEY":      # placeholder: fall back to WANDB_API_KEY / wandb login
            os.environ["WANDB_API_KEY"] = str(key)
    return wandb.init(
        project=config.training.wandb_project,
        name=config.training.get("wandb_exp_name"),
        entity=config.training.get("wandb_entity"),
        config=copy.deepcopy(config),
    )
