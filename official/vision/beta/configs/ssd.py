# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""SSD configuration definition."""

import os
from typing import List, Optional
import dataclasses
from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import hyperparams
from official.modeling import optimization
from official.vision.beta.configs import backbones
from official.vision.beta.configs import common
from official.vision.beta.configs import decoders


# pylint: disable=missing-class-docstring
@dataclasses.dataclass
class TfExampleDecoder(hyperparams.Config):
  regenerate_source_id: bool = False


@dataclasses.dataclass
class TfExampleDecoderLabelMap(hyperparams.Config):
  regenerate_source_id: bool = False
  label_map: str = ''


@dataclasses.dataclass
class DataDecoder(hyperparams.OneOfConfig):
  type: Optional[str] = 'simple_decoder'
  simple_decoder: TfExampleDecoder = TfExampleDecoder()
  label_map_decoder: TfExampleDecoderLabelMap = TfExampleDecoderLabelMap()


@dataclasses.dataclass
class Parser(hyperparams.Config):
  match_threshold: float = 0.5
  unmatched_threshold: float = 0.5
  aug_rand_hflip: bool = False
  aug_scale_min: float = 1.0
  aug_scale_max: float = 1.0
  skip_crowd_during_training: bool = True
  max_num_instances: int = 100


@dataclasses.dataclass
class DataConfig(cfg.DataConfig):
  """Input config for training."""
  input_path: str = ''
  global_batch_size: int = 0
  is_training: bool = False
  dtype: str = 'bfloat16'
  decoder: DataDecoder = DataDecoder()
  parser: Parser = Parser()
  shuffle_buffer_size: int = 10000
  file_type: str = 'tfrecord'


@dataclasses.dataclass
class SSDAnchor(hyperparams.Config):
  min_scale: float = 0.2
  max_scale: float = 0.95
  scales: List[float] = None
  aspect_ratios: List[float] = dataclasses.field(
      default_factory=lambda: [1.0, 2.0, 3.0, 1.0 / 2, 1.0 / 3])
  interpolated_scale_aspect_ratio: float = 1.0
  anchor_size: float = 4.0


@dataclasses.dataclass
class Losses(hyperparams.Config):
  focal_loss_alpha: float = 0.25
  focal_loss_gamma: float = 1.5
  huber_loss_delta: float = 0.1
  box_loss_weight: int = 50
  l2_weight_decay: float = 0.0


@dataclasses.dataclass
class SSDHead(hyperparams.Config):
  kernel_size: int = 3
  use_depthwise: bool = True
  activation: str = 'relu6'
  use_sync_bn: bool = False
  freeze_norm: bool = False
  add_background_class: bool = True
  class_prediction_bias_init: float = 0.0


@dataclasses.dataclass
class DetectionGenerator(hyperparams.Config):
  pre_nms_top_k: int = 5000
  pre_nms_score_threshold: float = 0.05
  nms_iou_threshold: float = 0.5
  max_num_detections: int = 100
  use_batched_nms: bool = False
  

@dataclasses.dataclass
class SSDModel(hyperparams.Config):
  num_classes: int = 0
  input_size: List[int] = dataclasses.field(default_factory=list)
  min_level: int = 3
  num_layers: int = 6
  anchor: SSDAnchor = SSDAnchor()
  backbone: backbones.Backbone = backbones.Backbone(
      type='mobiledet', mobiledet=backbones.MobileDet())
  decoder: decoders.Decoder = decoders.Decoder(
      type='mrfm', mrfm=decoders.MRFM())
  head: SSDHead = SSDHead()
  detection_generator: DetectionGenerator = DetectionGenerator()
  norm_activation: common.NormActivation = common.NormActivation()


@dataclasses.dataclass
class SSDTask(cfg.TaskConfig):
  model: SSDModel = SSDModel()
  train_data: DataConfig = DataConfig(is_training=True)
  validation_data: DataConfig = DataConfig(is_training=False)
  losses: Losses = Losses()
  init_checkpoint: Optional[str] = None
  init_checkpoint_modules: str = 'all'  # all or backbone
  annotation_file: Optional[str] = None
  per_category_metrics: bool = False


@exp_factory.register_config_factory('ssd')
def retinanet() -> cfg.ExperimentConfig:
  """RetinaNet general config."""
  return cfg.ExperimentConfig(
      task=SSDTask(),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])


COCO_INPUT_PATH_BASE = 'coco'
COCO_TRAIN_EXAMPLES = 118287
COCO_VAL_EXAMPLES = 5000
