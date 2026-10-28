"""Variable-view-count buckets for the training dataloader.

A *bucket* is one `(num_views, num_input_views, num_target_views, batch_size)`
combination. `data/batched_sampler.py` draws one bucket per batch, so every sample
inside a batch has the same view count and the batch still collates into single
`[b, v, ...]` tensors.

Config (see `configs/dl3dv.yaml`):

    training:
      num_views_buckets:  [5,  7,  9, 11]   # total views per sample
      batch_size_per_gpu: [24, 20, 16, 12]  # aligned to the bucket list

The input/target split follows from the view count alone (`split_counts`):
  odd  v: `num_input = ceil(v/2)`,  `num_target = floor(v/2)` -> 5: 3+2, 7: 4+3, 9: 5+4, 11: 6+5
  even v: `num_input = v/2 + 1`,    `num_target = v/2 - 1`    -> 6: 4+2, 8: 5+3, 10: 6+4
The model pins its canonical camera to view `v // 2`: the exact middle for odd v,
the right-of-centre frame for even v (10 -> 5). The sparse-context student of the
context alignment keeps that same frame as its canonical view (`MVSSL.sparse_view_indices`).

**Frame-distance scaling (train).** `view_selector.{min,max}_frame_dist` describe
the *largest* bucket. A smaller bucket covers a proportionally shorter clip, so
the average baseline between neighbouring views stays roughly constant:
`dist_v = round(dist_max * v / v_max)` (5 of 11 views -> ~5/11 of the span). The
curriculum is applied first (at the `v_max` scale), then scaled.

**Frame-distance scaling (eval): none.** Every eval view count spans the full
`max_frame_dist`; a higher view count just samples that same span more densely,
so fewer views give sparser coverage of the same clip.
"""

import random

import torch


def split_counts(num_views):
    """(num_input, num_target) for a view count.

    Odd v: ceil(v/2) input, floor(v/2) target. Even v: v/2 + 1 input, v/2 - 1 target
    (10 -> 6+4, 8 -> 5+3, 6 -> 4+2), i.e. both ends plus every other interior frame.
    """
    if num_views % 2 == 0:
        return num_views // 2 + 1, num_views // 2 - 1
    return (num_views + 1) // 2, num_views // 2


def uses_view_buckets(config):
    """True when the config asks for variable-view training."""
    return config.training.get("num_views_buckets", None) is not None


def build_buckets(config):
    """Resolve `training.num_views_buckets` + `training.batch_size_per_gpu` into
    a list of bucket dicts, ordered as written in the config. A bucket's position
    in this list is the `bucket_idx` carried by every sampler index."""
    views = list(config.training.num_views_buckets)
    batch_sizes = config.training.batch_size_per_gpu
    if not isinstance(batch_sizes, (list, tuple)):
        # a scalar batch size means "same batch size for every bucket"
        batch_sizes = [batch_sizes] * len(views)
    assert len(batch_sizes) == len(views), (
        "training.batch_size_per_gpu must have one entry per training.num_views_buckets "
        f"entry (got {len(batch_sizes)} batch sizes for {len(views)} buckets)"
    )
    assert all(v >= 3 for v in views), f"view buckets must be >= 3 views, got {views}"

    max_views = max(views)
    buckets = []
    for i, (v, bs) in enumerate(zip(views, batch_sizes)):
        n_in, n_tgt = split_counts(v)
        buckets.append(
            {
                "index": i,
                "num_views": v,
                "num_input_views": n_in,
                "num_target_views": n_tgt,
                "batch_size": int(bs),
                "max_views": max_views,
            }
        )
    return buckets


def resolve_frame_dist(
    view_selector, num_views, max_views, current_iteration=0, scale=True
):
    """`(min_frame_dist, max_frame_dist)` for one bucket.

    Applies the curriculum at the `max_views` scale first (same interpolation the
    fixed-view datasets use), then scales the span by `num_views / max_views` when
    `scale=True` (training). Eval passes `scale=False`.

    The result is clamped to `>= num_views - 1`, the shortest clip
    `two_frame_selector` can draw `num_views` distinct frames from.
    """
    min_dist = view_selector.get("min_frame_dist", 25)
    max_dist = view_selector.get("max_frame_dist", 100)

    if view_selector.get("use_curriculum", False):
        progress = min(current_iteration / view_selector.get("curriculum_iter", 30000), 1.0)
        start_min = view_selector.get("curriculum_start_min_frame_dist", 48)
        start_max = view_selector.get("curriculum_start_max_frame_dist", 64)
        min_dist = int(start_min + (min_dist - start_min) * progress)
        max_dist = int(start_max + (max_dist - start_max) * progress)

    if scale and num_views != max_views:
        ratio = num_views / max_views
        min_dist = int(round(min_dist * ratio))
        max_dist = int(round(max_dist * ratio))

    min_dist = max(min_dist, num_views - 1)
    max_dist = max(max_dist, min_dist)
    return min_dist, max_dist


def random_split_positions(num_views, num_input_views, rng):
    """Random disjoint input/target split over view *positions* `0..num_views-1`.

    Positions 0 and `num_views-1` always go to the input set (so targets are always interpolations, never extrapolations)
    and the remaining input positions are drawn uniformly from the interior.

    `rng` is a `random.Random`-compatible object. Returns sorted LongTensors.
    """
    assert 2 <= num_input_views <= num_views
    interior = list(range(1, num_views - 1))
    middle = rng.sample(interior, num_input_views - 2)
    input_positions = sorted([0, num_views - 1] + middle)
    input_set = set(input_positions)
    target_positions = [p for p in range(num_views) if p not in input_set]
    return (
        torch.tensor(input_positions, dtype=torch.long),
        torch.tensor(target_positions, dtype=torch.long),
    )


def stratified_middle_frames(lo, hi, k, rng=random):
    """`k` frame indices from the half-open interior range `[lo, hi)`, one drawn
    uniformly from each of `k` contiguous, near-equal-width bins (widths
    `hi-lo` split by `divmod`, so they differ by at most one frame) instead of
    `k` uniform draws from the whole range.

    Used by `two_frame_selector`'s `stratified=True` path as the alternative to
    `random.sample(range(lo, hi), k)`: plain sampling can (and, over many draws,
    routinely does) cluster several of the `k` middle views close together and
    leave a long gap elsewhere in the clip; one draw per bin guarantees every
    sub-interval of the span contributes exactly one view. `lo`/`hi` are the
    frame indices just after `start_frame` / at `end_frame` (exclusive), so the
    span here is `end_frame - start_frame`, i.e. `frame_dist`, and requires
    `hi - lo >= k` (guaranteed by the `min_frame_dist >= num_views - 1` clamp in
    `resolve_frame_dist` and the fixed-view callers).
    """
    span = hi - lo
    assert span >= k, f"stratified sampling needs span >= k ({span} < {k})"
    base, rem = divmod(span, k)
    positions = []
    cursor = lo
    for i in range(k):
        width = base + (1 if i < rem else 0)
        positions.append(rng.randint(cursor, cursor + width - 1))
        cursor += width
    return positions


def even_odd_split_positions(num_views):
    """Deterministic eval split: even positions are input, odd are target.

    For odd v the sizes come out as `ceil(v/2)` / `floor(v/2)`. For even v the last position `v-1` (odd) is moved to the input set so
    targets stay interpolations, e.g. 10 -> `[0,2,4,6,8,9]` / `[1,3,5,7]`. Both match
    `split_counts`.
    """
    inputs = list(range(0, num_views, 2))
    targets = list(range(1, num_views, 2))
    if num_views % 2 == 0:
        targets.remove(num_views - 1)
        inputs.append(num_views - 1)
    return (
        torch.tensor(inputs, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
    )
