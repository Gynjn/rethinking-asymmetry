"""Train / eval DataLoaders for one dataset entry ({class_name, dataset_path, data_root, view_selector})."""

import copy
import importlib

from torch.utils.data import DataLoader

from data.batched_sampler import ViewBucketBatchSampler
from data.view_buckets import build_buckets, split_counts


def build_dataset(config, entry, inference=False):
    """The dataset classes read `training.{dataset_path, data_root, view_selector}`; set them from the entry."""
    sub_config = copy.deepcopy(config)
    sub_config.training.dataset_path = entry["dataset_path"]
    sub_config.training.data_root = entry["data_root"]
    sub_config.training.view_selector = copy.deepcopy(entry.get("view_selector", {}))
    module, class_name = entry["class_name"].rsplit(".", 1)
    return importlib.import_module(module).__dict__[class_name](sub_config, inference=inference)


def uses_curriculum(config):
    return config.training.train_dataset.get("view_selector", {}).get("use_curriculum", False)


def build_train_loader(config, world_size=1, rank=0, epoch=0, seed=777, dataset=None):
    """One view-count bucket per batch (data/view_buckets.py); every rank runs the same bucket at a step.
    Pass `dataset` to rebuild only the loader (workers snapshot the curriculum state)."""
    if dataset is None:
        dataset = build_dataset(config, config.training.train_dataset)
    batch_sampler = ViewBucketBatchSampler(
        num_scenes=len(dataset), buckets=build_buckets(config),
        world_size=world_size, rank=rank, drop_last=True, seed=seed,
    )
    batch_sampler.set_epoch(epoch)
    workers = config.training.num_workers
    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=False,
        prefetch_factor=config.training.prefetch_factor if workers > 0 else None,
    )
    return dataset, loader


def set_loader_epoch(dataloader, epoch):
    """Reshuffle the bucket plan; must be called with the same epoch on every rank."""
    dataloader.batch_sampler.set_epoch(epoch)


def build_eval_loader(config, entry, num_views):
    """Fixed `num_views` frames per scene over the entry's `view_selector.max_frame_dist`."""
    eval_config = copy.deepcopy(config)
    eval_config.training.num_views = num_views
    eval_config.training.num_input_views, eval_config.training.num_target_views = split_counts(num_views)
    workers = config.training.num_workers
    return DataLoader(
        build_dataset(eval_config, entry, inference=True),
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        persistent_workers=False,
        pin_memory=False,
        drop_last=False,
        prefetch_factor=1 if workers > 0 else None,
    )
