"""Batch sampler that draws one view-count bucket per batch.

Adapted from a `DynamicBatchedMultiFeatureRandomSampler`, reduced to the
single feature this repo needs (view count) and with two deliberate changes:

1. **The epoch's batch plan is materialised up front**, so `__len__` is exact
   instead of a lower bound estimated from the largest batch size. The training
   loop uses that number as "forward passes per epoch", so an estimate would
   silently truncate epochs.
2. **Ranks are split *within* a batch, not round-robin *across* batches.** The
   reference hands batch `k` to rank `k % world_size`; since each batch draws its
   own bucket, ranks would then run different view counts at the same step —
   different batch sizes and different token counts per rank, which skews the
   all-reduced gradient. Here every rank shares the plan (same seed, same epoch)
   and takes its own contiguous slice of a `batch_size * world_size` chunk, so all
   ranks always agree on the view count for the current step.

Yields lists of `(scene_idx, bucket_idx)` tuples. `bucket_idx` indexes
`data/view_buckets.py:build_buckets`, and the dataset resolves it into a view count
and a frame-distance range.
"""

import numpy as np


class ViewBucketBatchSampler:
    def __init__(
        self,
        num_scenes,
        buckets,
        world_size=1,
        rank=0,
        drop_last=True,
        seed=777,
    ):
        self.num_scenes = int(num_scenes)
        self.buckets = buckets
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0
        self._plan_cache = None
        assert self.num_scenes > 0, "empty dataset"
        assert len(self.buckets) > 0, "no view buckets configured"

    def set_epoch(self, epoch):
        """Reshuffle for a new epoch. Must be called on every rank with the same
        value, since the plan (and therefore each step's view count) is derived
        from it."""
        if epoch != self.epoch:
            self.epoch = int(epoch)
            self._plan_cache = None

    def _plan(self):
        """List of `(bucket_idx, scene_indices)` batches for this rank."""
        if self._plan_cache is not None:
            return self._plan_cache

        rng = np.random.default_rng(self.seed + self.epoch)
        order = rng.permutation(self.num_scenes)

        plan = []
        pos = 0
        while pos < len(order):
            bucket = self.buckets[int(rng.integers(len(self.buckets)))]
            chunk_size = bucket["batch_size"] * self.world_size
            remaining = len(order) - pos
            if remaining < chunk_size:
                # A ragged tail cannot be split evenly across ranks, so it is only
                # kept in the single-process case.
                if self.drop_last or self.world_size > 1:
                    break
                chunk_size = remaining
            chunk = order[pos : pos + chunk_size]
            pos += chunk_size
            per_rank = len(chunk) // self.world_size if self.world_size > 1 else len(chunk)
            start = self.rank * per_rank
            plan.append((bucket["index"], chunk[start : start + per_rank]))

        self._plan_cache = plan
        return plan

    def __len__(self):
        return len(self._plan())

    def __iter__(self):
        for bucket_idx, scene_indices in self._plan():
            yield [(int(scene_idx), int(bucket_idx)) for scene_idx in scene_indices]

    def describe(self):
        """One-line summary of the current epoch's bucket mix, for startup logs."""
        plan = self._plan()
        counts = {}
        samples = {}
        for bucket_idx, scene_indices in plan:
            views = self.buckets[bucket_idx]["num_views"]
            counts[views] = counts.get(views, 0) + 1
            samples[views] = samples.get(views, 0) + len(scene_indices)
        parts = [
            f"{views}v: {counts[views]} batches / {samples[views]} samples"
            for views in sorted(counts)
        ]
        return f"{len(plan)} batches per epoch (" + ", ".join(parts) + ")"
