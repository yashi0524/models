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

"""Multibox/SSD detection models"""
from typing import Any, Mapping, List, Optional, Union

# Import libraries
import tensorflow as tf

from official.vision.beta.ops import anchor


@tf.keras.utils.register_keras_serializable(package='Vision')
class SSDModel(tf.keras.Model):
  """The SSD model class."""

  def __init__(self,
               backbone: tf.keras.Model,
               decoder: tf.keras.Model,
               head: tf.keras.layers.Layer,
               detection_generator: tf.keras.layers.Layer,
               min_level: Optional[int] = None,
               num_layers: Optional[int] = None,
               min_scale: Optional[float] = 0.2,
               max_scale: Optional[float] = 0.95,
               scales: Optional[List[float]] = None,
               aspect_ratios: Optional[List[float]] = None,
               anchor_size: Optional[float] = None,
               interpolated_scale_aspect_ratio: Optional[float] = 1.0,
               **kwargs):
    """SSD model initialization function.

    Args:
      backbone: `tf.keras.Model` a backbone network.
      decoder: `tf.keras.Model` a decoder network.
      head: `RetinaNetHead`, the RetinaNet head.
      detection_generator: the detection generator.
      min_level: An `int` number of minimum feature level.
      num_layers: An `int` number of grid layers to create anchors for (actual
        grid sizes passed in at generation time)
      min_scale: scale of anchors corresponding to finest resolution (float)
      max_scale: scale of anchors corresponding to coarsest resolution (float)
      scales: As list of anchor scales to use. When not None and not empty,
        min_scale and max_scale are not used.
      aspect_ratios: A list representing the aspect raito
        anchors added on each level. The number indicates the ratio of width to
        height. For instances, aspect_ratios=[1.0, 2.0, 0.5] adds three anchors
        on each scale level.
      anchor_size: A number representing the scale of size of the base
        anchor to the feature stride 2^level.
      interpolated_scale_aspect_ratio: An additional anchor is added with this
        aspect ratio and a scale interpolated between the scale for a layer
        and the scale for the next layer (1.0 for the last layer).
        This anchor is not included if this value is 0.
      **kwargs: keyword arguments to be passed.
    """
    super(SSDModel, self).__init__(**kwargs)
    
    self._backbone = backbone
    self._decoder = decoder
    self._head = head
    self._detection_generator = detection_generator
    self._min_level = min_level
    self._num_layers = num_layers
    self._scales = scales
    self._min_scale = min_scale
    self._max_scale = max_scale
    self._aspect_ratios = aspect_ratios
    self._anchor_size = anchor_size
    self._interpolated_scale_aspect_ratio = interpolated_scale_aspect_ratio

  def call(self,
           images: tf.Tensor,
           image_shape: Optional[tf.Tensor] = None,
           anchor_boxes: Optional[Mapping[str, tf.Tensor]] = None,
           training: bool = None) -> Mapping[str, tf.Tensor]:
    """Forward pass of the RetinaNet model.

    Args:
      images: `Tensor`, the input batched images, whose shape is
        [batch, height, width, 3].
      image_shape: `Tensor`, the actual shape of the input images, whose shape
        is [batch, 2] where the last dimension is [height, width]. Note that
        this is the actual image shape excluding paddings. For example, images
        in the batch may be resized into different shapes before padding to the
        fixed size.
      anchor_boxes: a dict of tensors which includes multilevel anchors.
        - key: `str`, the level of the multilevel predictions.
        - values: `Tensor`, the anchor coordinates of a particular feature
            level, whose shape is [height_l, width_l, num_anchors_per_location].
      training: `bool`, indicating whether it is in training mode.

    Returns:
      scores: a dict of tensors which includes scores of the predictions.
        - key: `str`, the level of the multilevel predictions.
        - values: `Tensor`, the box scores predicted from a particular feature
            level, whose shape is
            [batch, height_l, width_l, num_classes * num_anchors_per_location].
      boxes: a dict of tensors which includes coordinates of the predictions.
        - key: `str`, the level of the multilevel predictions.
        - values: `Tensor`, the box coordinates predicted from a particular
            feature level, whose shape is
            [batch, height_l, width_l, 4 * num_anchors_per_location].
    """
    # Feature extraction.
    features = self.backbone(images)
    if self.decoder:
      features = self.decoder(features)

    raw_scores, raw_boxes = self.head(features)

    if training:
      outputs = {
          'cls_outputs': raw_scores,
          'box_outputs': raw_boxes,
      }
      return outputs
    
    # Generate anchor boxes for this batch if not provided.
    if anchor_boxes is None:
      _, image_height, image_width, _ = images.get_shape().as_list()
      anchor_boxes = anchor.SSDAnchor(
          min_level=self._min_level,
          num_layers=self._num_layers,
          min_scale=self._min_scale,
          max_scale=self._max_scale,
          scales=self._scales,
          aspect_ratios=self._aspect_ratios,
          anchor_size=self._anchor_size,
          interpolated_scale_aspect_ratio=self._interpolated_scale_aspect_ratio,
          image_size=(image_height, image_width)).multilevel_boxes
      for l in anchor_boxes:
        anchor_boxes[l] = tf.tile(
            tf.expand_dims(anchor_boxes[l], axis=0),
            [tf.shape(images)[0], 1, 1, 1])

    # Post-processing.
    final_results = self.detection_generator(
        raw_boxes, raw_scores, anchor_boxes, image_shape)
    outputs = {
        'detection_boxes': final_results['detection_boxes'],
        'detection_scores': final_results['detection_scores'],
        'detection_classes': final_results['detection_classes'],
        'num_detections': final_results['num_detections'],
        'cls_outputs': raw_scores,
        'box_outputs': raw_boxes,
    }
    return outputs

  @property
  def checkpoint_items(
      self) -> Mapping[str, Union[tf.keras.Model, tf.keras.layers.Layer]]:
    """Returns a dictionary of items to be additionally checkpointed."""
    items = dict(backbone=self.backbone, head=self.head)
    if self.decoder is not None:
      items.update(decoder=self.decoder)

    return items

  @property
  def backbone(self) -> tf.keras.Model:
    return self._backbone

  @property
  def decoder(self) -> tf.keras.Model:
    return self._decoder

  @property
  def head(self) -> tf.keras.layers.Layer:
    return self._head

  @property
  def detection_generator(self) -> tf.keras.layers.Layer:
    return self._detection_generator

  def get_config(self) -> Mapping[str, Any]:
    config_dict = {
        'backbone': self._backbone,
        'decoder': self._decoder,
        'head': self._head,
        'detection_generator': self._detection_generator,
        'min_level': self._min_level,
        'num_layers': self._num_layers,
        'scales': self._scales,
        'min_scale': self._min_scale,
        'max_scale': self._max_scale,
        'aspect_ratios': self._aspect_ratios,
        'anchor_size': self._anchor_size,
        'interpolated_scale_aspect_ratio': self._interpolated_scale_aspect_ratio,
    }
    return config_dict

  @classmethod
  def from_config(cls, config):
    return cls(**config)
