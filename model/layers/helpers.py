# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the FAIR Noncommercial Research License found in
# licenses/LICENSE_Pixio.
#
# From Pixio (https://github.com/facebookresearch/pixio): evaluation/common/layers/helpers.py
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models
# --------------------------------------------------------

from itertools import repeat
import collections.abc


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_2tuple = _ntuple(2)
