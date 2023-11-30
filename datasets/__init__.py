#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from .fourier import FourierDataset, FourierLightfieldDataset

from .random import (
    RandomPixelDataset,
    RandomRayDataset,
    RandomRayLightfieldDataset,
    RandomViewSubsetDataset,
)

from .ent_video_full_mask import EntVideoDatasetFull
from .ent_video_importance_mask import EntVideoDatasetImportance
from.ent_video import EntVideoDatasetNew
from. ent_video_random_mask import EntVideoDatasetRandom
from. ent_video_new_mask_small import EntVideoDatasetNewSmall
from .ent_video_new_importance_mask import EntVideoDatasetImportanceNew
from .ent_video_new_importance_mask_tool import EntVideoDatasetImportanceNewTool
from .ent_video_new_importance_mask_small import EntVideoDatasetImportanceNewSmall

dataset_dict = {
    "fourier": FourierDataset,
    "fourier_lightfield": FourierLightfieldDataset,
    "random_ray": RandomRayDataset,
    "random_pixel": RandomPixelDataset,
    "random_lightfield": RandomRayLightfieldDataset,
    "random_view": RandomViewSubsetDataset,
    "ent_video_full_mask": EntVideoDatasetFull,
    "ent_video_importance_mask": EntVideoDatasetImportance,
    "ent_video_new_importance_mask": EntVideoDatasetImportanceNew,
    "ent_video_new_importance_mask_small": EntVideoDatasetImportanceNewSmall,
    "ent_video_new_mask":EntVideoDatasetNew,
    "ent_video_random_mask":EntVideoDatasetRandom,
    "ent_video_new_mask_small":EntVideoDatasetNewSmall,
    "ent_video_new_importance_mask_tool": EntVideoDatasetImportanceNewTool,
}
