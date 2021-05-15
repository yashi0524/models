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

"""Contains definitions of dense prediction heads."""

from typing import Any, Dict, List, Mapping, Optional, Union

# Import libraries

import numpy as np
import tensorflow as tf

from official.modeling import tf_utils
from official.vision.beta.modeling.layers import nn_layers


@tf.keras.utils.register_keras_serializable(package='Vision')
class RetinaNetHead(tf.keras.layers.Layer):
  """Creates a RetinaNet head."""

  def __init__(
      self,
      min_level: int,
      max_level: int,
      num_classes: int,
      num_anchors_per_location: int,
      num_convs: int = 4,
      num_filters: int = 256,
      attribute_heads: Optional[List[Dict[str, Any]]] = None,
      use_separable_conv: bool = False,
      activation: str = 'relu',
      use_sync_bn: bool = False,
      norm_momentum: float = 0.99,
      norm_epsilon: float = 0.001,
      kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      **kwargs):
    """Initializes a RetinaNet head.

    Args:
      min_level: An `int` number of minimum feature level.
      max_level: An `int` number of maximum feature level.
      num_classes: An `int` number of classes to predict.
      num_anchors_per_location: An `int` number of number of anchors per pixel
        location.
      num_convs: An `int` number that represents the number of the intermediate
        conv layers before the prediction.
      num_filters: An `int` number that represents the number of filters of the
        intermediate conv layers.
      attribute_heads: If not None, a list that contains a dict for each
        additional attribute head. Each dict consists of 3 key-value pairs:
        `name`, `type` ('regression' or 'classification'), and `size` (number
        of predicted values for each instance).
      use_separable_conv: A `bool` that indicates whether the separable
        convolution layers is used.
      activation: A `str` that indicates which activation is used, e.g. 'relu',
        'swish', etc.
      use_sync_bn: A `bool` that indicates whether to use synchronized batch
        normalization across different replicas.
      norm_momentum: A `float` of normalization momentum for the moving average.
      norm_epsilon: A `float` added to variance to avoid dividing by zero.
      kernel_regularizer: A `tf.keras.regularizers.Regularizer` object for
        Conv2D. Default is None.
      bias_regularizer: A `tf.keras.regularizers.Regularizer` object for Conv2D.
      **kwargs: Additional keyword arguments to be passed.
    """
    super(RetinaNetHead, self).__init__(**kwargs)
    self._config_dict = {
        'min_level': min_level,
        'max_level': max_level,
        'num_classes': num_classes,
        'num_anchors_per_location': num_anchors_per_location,
        'num_convs': num_convs,
        'num_filters': num_filters,
        'attribute_heads': attribute_heads,
        'use_separable_conv': use_separable_conv,
        'activation': activation,
        'use_sync_bn': use_sync_bn,
        'norm_momentum': norm_momentum,
        'norm_epsilon': norm_epsilon,
        'kernel_regularizer': kernel_regularizer,
        'bias_regularizer': bias_regularizer,
    }

    if tf.keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1
    self._activation = tf_utils.get_activation(activation)

  def build(self, input_shape: Union[tf.TensorShape, List[tf.TensorShape]]):
    """Creates the variables of the head."""
    conv_op = (tf.keras.layers.SeparableConv2D
               if self._config_dict['use_separable_conv']
               else tf.keras.layers.Conv2D)
    conv_kwargs = {
        'filters': self._config_dict['num_filters'],
        'kernel_size': 3,
        'padding': 'same',
        'bias_initializer': tf.zeros_initializer(),
        'bias_regularizer': self._config_dict['bias_regularizer'],
    }
    if not self._config_dict['use_separable_conv']:
      conv_kwargs.update({
          'kernel_initializer': tf.keras.initializers.RandomNormal(
              stddev=0.01),
          'kernel_regularizer': self._config_dict['kernel_regularizer'],
      })
    bn_op = (tf.keras.layers.experimental.SyncBatchNormalization
             if self._config_dict['use_sync_bn']
             else tf.keras.layers.BatchNormalization)
    bn_kwargs = {
        'axis': self._bn_axis,
        'momentum': self._config_dict['norm_momentum'],
        'epsilon': self._config_dict['norm_epsilon'],
    }

    # Class net.
    self._cls_convs = []
    self._cls_norms = []
    for level in range(
        self._config_dict['min_level'], self._config_dict['max_level'] + 1):
      this_level_cls_norms = []
      for i in range(self._config_dict['num_convs']):
        if level == self._config_dict['min_level']:
          cls_conv_name = 'classnet-conv_{}'.format(i)
          self._cls_convs.append(conv_op(name=cls_conv_name, **conv_kwargs))
        cls_norm_name = 'classnet-conv-norm_{}_{}'.format(level, i)
        this_level_cls_norms.append(bn_op(name=cls_norm_name, **bn_kwargs))
      self._cls_norms.append(this_level_cls_norms)

    classifier_kwargs = {
        'filters': (
            self._config_dict['num_classes'] *
            self._config_dict['num_anchors_per_location']),
        'kernel_size': 3,
        'padding': 'same',
        'bias_initializer': tf.constant_initializer(-np.log((1 - 0.01) / 0.01)),
        'bias_regularizer': self._config_dict['bias_regularizer'],
    }
    if not self._config_dict['use_separable_conv']:
      classifier_kwargs.update({
          'kernel_initializer': tf.keras.initializers.RandomNormal(stddev=1e-5),
          'kernel_regularizer': self._config_dict['kernel_regularizer'],
      })
    self._classifier = conv_op(name='scores', **classifier_kwargs)

    # Box net.
    self._box_convs = []
    self._box_norms = []
    for level in range(
        self._config_dict['min_level'], self._config_dict['max_level'] + 1):
      this_level_box_norms = []
      for i in range(self._config_dict['num_convs']):
        if level == self._config_dict['min_level']:
          box_conv_name = 'boxnet-conv_{}'.format(i)
          self._box_convs.append(conv_op(name=box_conv_name, **conv_kwargs))
        box_norm_name = 'boxnet-conv-norm_{}_{}'.format(level, i)
        this_level_box_norms.append(bn_op(name=box_norm_name, **bn_kwargs))
      self._box_norms.append(this_level_box_norms)

    box_regressor_kwargs = {
        'filters': 4 * self._config_dict['num_anchors_per_location'],
        'kernel_size': 3,
        'padding': 'same',
        'bias_initializer': tf.zeros_initializer(),
        'bias_regularizer': self._config_dict['bias_regularizer'],
    }
    if not self._config_dict['use_separable_conv']:
      box_regressor_kwargs.update({
          'kernel_initializer': tf.keras.initializers.RandomNormal(
              stddev=1e-5),
          'kernel_regularizer': self._config_dict['kernel_regularizer'],
      })
    self._box_regressor = conv_op(name='boxes', **box_regressor_kwargs)

    # Attribute learning nets.
    if self._config_dict['attribute_heads']:
      self._att_predictors = {}
      self._att_convs = {}
      self._att_norms = {}

      for att_config in self._config_dict['attribute_heads']:
        att_name = att_config['name']
        att_type = att_config['type']
        att_size = att_config['size']
        att_convs_i = []
        att_norms_i = []

        # Build conv and norm layers.
        for level in range(self._config_dict['min_level'],
                           self._config_dict['max_level'] + 1):
          this_level_att_norms = []
          for i in range(self._config_dict['num_convs']):
            if level == self._config_dict['min_level']:
              att_conv_name = '{}-conv_{}'.format(att_name, i)
              att_convs_i.append(conv_op(name=att_conv_name, **conv_kwargs))
            att_norm_name = '{}-conv-norm_{}_{}'.format(att_name, level, i)
            this_level_att_norms.append(bn_op(name=att_norm_name, **bn_kwargs))
          att_norms_i.append(this_level_att_norms)
        self._att_convs[att_name] = att_convs_i
        self._att_norms[att_name] = att_norms_i

        # Build the final prediction layer.
        att_predictor_kwargs = {
            'filters':
                (att_size * self._config_dict['num_anchors_per_location']),
            'kernel_size': 3,
            'padding': 'same',
            'bias_initializer': tf.zeros_initializer(),
            'bias_regularizer': self._config_dict['bias_regularizer'],
        }
        if att_type == 'regression':
          att_predictor_kwargs.update(
              {'bias_initializer': tf.zeros_initializer()})
        elif att_type == 'classification':
          att_predictor_kwargs.update({
              'bias_initializer':
                  tf.constant_initializer(-np.log((1 - 0.01) / 0.01))
          })
        else:
          raise ValueError(
              'Attribute head type {} not supported.'.format(att_type))

        if not self._config_dict['use_separable_conv']:
          att_predictor_kwargs.update({
              'kernel_initializer':
                  tf.keras.initializers.RandomNormal(stddev=1e-5),
              'kernel_regularizer':
                  self._config_dict['kernel_regularizer'],
          })

        self._att_predictors[att_name] = conv_op(
            name='{}_attributes'.format(att_name), **att_predictor_kwargs)

    super(RetinaNetHead, self).build(input_shape)

  def call(self, features: Mapping[str, tf.Tensor]):
    """Forward pass of the RetinaNet head.

    Args:
      features: A `dict` of `tf.Tensor` where
        - key: A `str` of the level of the multilevel features.
        - values: A `tf.Tensor`, the feature map tensors, whose shape is
            [batch, height_l, width_l, channels].

    Returns:
      scores: A `dict` of `tf.Tensor` which includes scores of the predictions.
        - key: A `str` of the level of the multilevel predictions.
        - values: A `tf.Tensor` of the box scores predicted from a particular
            feature level, whose shape is
            [batch, height_l, width_l, num_classes * num_anchors_per_location].
      boxes: A `dict` of `tf.Tensor` which includes coordinates of the
        predictions.
        - key: A `str` of the level of the multilevel predictions.
        - values: A `tf.Tensor` of the box scores predicted from a particular
            feature level, whose shape is
            [batch, height_l, width_l, 4 * num_anchors_per_location].
      attributes: a dict of (attribute_name, attribute_prediction). Each
        `attribute_prediction` is a dict of:
        - key: `str`, the level of the multilevel predictions.
        - values: `Tensor`, the box scores predicted from a particular feature
            level, whose shape is
            [batch, height_l, width_l,
            attribute_size * num_anchors_per_location].
        Can be an empty dictionary if no attribute learning is required.
    """
    scores = {}
    boxes = {}
    if self._config_dict['attribute_heads']:
      attributes = {
          att_config['name']: {}
          for att_config in self._config_dict['attribute_heads']
      }
    else:
      attributes = {}

    for i, level in enumerate(
        range(self._config_dict['min_level'],
              self._config_dict['max_level'] + 1)):
      this_level_features = features[str(level)]

      # class net.
      x = this_level_features
      for conv, norm in zip(self._cls_convs, self._cls_norms[i]):
        x = conv(x)
        x = norm(x)
        x = self._activation(x)
      scores[str(level)] = self._classifier(x)

      # box net.
      x = this_level_features
      for conv, norm in zip(self._box_convs, self._box_norms[i]):
        x = conv(x)
        x = norm(x)
        x = self._activation(x)
      boxes[str(level)] = self._box_regressor(x)

      # attribute nets.
      if self._config_dict['attribute_heads']:
        for att_config in self._config_dict['attribute_heads']:
          att_name = att_config['name']
          x = this_level_features
          for conv, norm in zip(self._att_convs[att_name],
                                self._att_norms[att_name][i]):
            x = conv(x)
            x = norm(x)
            x = self._activation(x)
          attributes[att_name][str(level)] = self._att_predictors[att_name](x)

    return scores, boxes, attributes

  def get_config(self):
    return self._config_dict

  @classmethod
  def from_config(cls, config):
    return cls(**config)


@tf.keras.utils.register_keras_serializable(package='Vision')
class RPNHead(tf.keras.layers.Layer):
  """Creates a Region Proposal Network (RPN) head."""

  def __init__(
      self,
      min_level: int,
      max_level: int,
      num_anchors_per_location: int,
      num_convs: int = 1,
      num_filters: int = 256,
      use_separable_conv: bool = False,
      activation: str = 'relu',
      use_sync_bn: bool = False,
      norm_momentum: float = 0.99,
      norm_epsilon: float = 0.001,
      kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      **kwargs):
    """Initializes a Region Proposal Network head.

    Args:
      min_level: An `int` number of minimum feature level.
      max_level: An `int` number of maximum feature level.
      num_anchors_per_location: An `int` number of number of anchors per pixel
        location.
      num_convs: An `int` number that represents the number of the intermediate
        convolution layers before the prediction.
      num_filters: An `int` number that represents the number of filters of the
        intermediate convolution layers.
      use_separable_conv: A `bool` that indicates whether the separable
        convolution layers is used.
      activation: A `str` that indicates which activation is used, e.g. 'relu',
        'swish', etc.
      use_sync_bn: A `bool` that indicates whether to use synchronized batch
        normalization across different replicas.
      norm_momentum: A `float` of normalization momentum for the moving average.
      norm_epsilon: A `float` added to variance to avoid dividing by zero.
      kernel_regularizer: A `tf.keras.regularizers.Regularizer` object for
        Conv2D. Default is None.
      bias_regularizer: A `tf.keras.regularizers.Regularizer` object for Conv2D.
      **kwargs: Additional keyword arguments to be passed.
    """
    super(RPNHead, self).__init__(**kwargs)
    self._config_dict = {
        'min_level': min_level,
        'max_level': max_level,
        'num_anchors_per_location': num_anchors_per_location,
        'num_convs': num_convs,
        'num_filters': num_filters,
        'use_separable_conv': use_separable_conv,
        'activation': activation,
        'use_sync_bn': use_sync_bn,
        'norm_momentum': norm_momentum,
        'norm_epsilon': norm_epsilon,
        'kernel_regularizer': kernel_regularizer,
        'bias_regularizer': bias_regularizer,
    }

    if tf.keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1
    self._activation = tf_utils.get_activation(activation)

  def build(self, input_shape):
    """Creates the variables of the head."""
    conv_op = (tf.keras.layers.SeparableConv2D
               if self._config_dict['use_separable_conv']
               else tf.keras.layers.Conv2D)
    conv_kwargs = {
        'filters': self._config_dict['num_filters'],
        'kernel_size': 3,
        'padding': 'same',
        'bias_initializer': tf.zeros_initializer(),
        'bias_regularizer': self._config_dict['bias_regularizer'],
    }
    if not self._config_dict['use_separable_conv']:
      conv_kwargs.update({
          'kernel_initializer': tf.keras.initializers.RandomNormal(
              stddev=0.01),
          'kernel_regularizer': self._config_dict['kernel_regularizer'],
      })
    bn_op = (tf.keras.layers.experimental.SyncBatchNormalization
             if self._config_dict['use_sync_bn']
             else tf.keras.layers.BatchNormalization)
    bn_kwargs = {
        'axis': self._bn_axis,
        'momentum': self._config_dict['norm_momentum'],
        'epsilon': self._config_dict['norm_epsilon'],
    }

    self._convs = []
    self._norms = []
    for level in range(
        self._config_dict['min_level'], self._config_dict['max_level'] + 1):
      this_level_norms = []
      for i in range(self._config_dict['num_convs']):
        if level == self._config_dict['min_level']:
          conv_name = 'rpn-conv_{}'.format(i)
          self._convs.append(conv_op(name=conv_name, **conv_kwargs))
        norm_name = 'rpn-conv-norm_{}_{}'.format(level, i)
        this_level_norms.append(bn_op(name=norm_name, **bn_kwargs))
      self._norms.append(this_level_norms)

    classifier_kwargs = {
        'filters': self._config_dict['num_anchors_per_location'],
        'kernel_size': 1,
        'padding': 'valid',
        'bias_initializer': tf.zeros_initializer(),
        'bias_regularizer': self._config_dict['bias_regularizer'],
    }
    if not self._config_dict['use_separable_conv']:
      classifier_kwargs.update({
          'kernel_initializer': tf.keras.initializers.RandomNormal(
              stddev=1e-5),
          'kernel_regularizer': self._config_dict['kernel_regularizer'],
      })
    self._classifier = conv_op(name='rpn-scores', **classifier_kwargs)

    box_regressor_kwargs = {
        'filters': 4 * self._config_dict['num_anchors_per_location'],
        'kernel_size': 1,
        'padding': 'valid',
        'bias_initializer': tf.zeros_initializer(),
        'bias_regularizer': self._config_dict['bias_regularizer'],
    }
    if not self._config_dict['use_separable_conv']:
      box_regressor_kwargs.update({
          'kernel_initializer': tf.keras.initializers.RandomNormal(
              stddev=1e-5),
          'kernel_regularizer': self._config_dict['kernel_regularizer'],
      })
    self._box_regressor = conv_op(name='rpn-boxes', **box_regressor_kwargs)

    super(RPNHead, self).build(input_shape)

  def call(self, features: Mapping[str, tf.Tensor]):
    """Forward pass of the RPN head.

    Args:
      features: A `dict` of `tf.Tensor` where
        - key: A `str` of the level of the multilevel features.
        - values: A `tf.Tensor`, the feature map tensors, whose shape is [batch,
          height_l, width_l, channels].

    Returns:
      scores: A `dict` of `tf.Tensor` which includes scores of the predictions.
        - key: A `str` of the level of the multilevel predictions.
        - values: A `tf.Tensor` of the box scores predicted from a particular
            feature level, whose shape is
            [batch, height_l, width_l, num_classes * num_anchors_per_location].
      boxes: A `dict` of `tf.Tensor` which includes coordinates of the
        predictions.
        - key: A `str` of the level of the multilevel predictions.
        - values: A `tf.Tensor` of the box scores predicted from a particular
            feature level, whose shape is
            [batch, height_l, width_l, 4 * num_anchors_per_location].
    """
    scores = {}
    boxes = {}
    for i, level in enumerate(
        range(self._config_dict['min_level'],
              self._config_dict['max_level'] + 1)):
      x = features[str(level)]
      for conv, norm in zip(self._convs, self._norms[i]):
        x = conv(x)
        x = norm(x)
        x = self._activation(x)
      scores[str(level)] = self._classifier(x)
      boxes[str(level)] = self._box_regressor(x)
    return scores, boxes

  def get_config(self):
    return self._config_dict

  @classmethod
  def from_config(cls, config):
    return cls(**config)


@tf.keras.utils.register_keras_serializable(package='Vision')
class SSDHead(tf.keras.layers.Layer):
  """Creates a SSD head."""

  def __init__(
      self,
      min_level: int,
      max_level: int,
      num_classes: int,
      num_anchors_per_location_list: List[int],
      is_training: bool,
      kernel_size: int = 3,
      use_depthwise: bool = True,
      activation: str = 'relu6',
      use_sync_bn: bool = False,
      freeze_norm: bool = False,
      norm_momentum: float = 0.99,
      norm_epsilon: float = 0.001,
      add_background_class: bool = True,
      class_prediction_bias_init: float = 0.0,
      kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      **kwargs):
    """Initializes a SSD head.

    Args:
      min_level: An `int` number of minimum feature level.
      max_level: An `int` number of maximum feature level.
      num_classes: An `int` number of classes to predict.
      num_anchors_per_location_list: A list of integers representing the number of
        box predictions to be made per spatial location for each feature map.
      is_training: Indicates whether the BoxPredictor is in training mode.
      kernel_size: An `int` kernel size used for building prediction heads.
      use_depthwise: A `bool` indicating whether depthwise convolution is used
        for building extra feature layers.
      activation: A `str` that indicates which activation is used, e.g. 'relu',
        'swish', etc.
      use_sync_bn: A `bool` that indicates whether to use synchronized batch
        normalization across different replicas.
      norm_momentum: A `float` of normalization momentum for the moving average.
      norm_epsilon: A `float` added to variance to avoid dividing by zero.
      freeze_norm: A `bool`. Whether to freeze batch norm parameters during
        training or not. When training with a small batch size (e.g. 1), it is
        desirable to freeze batch norm update and use pretrained batch norm
        params.
      add_background_class: Whether to add an implicit background class.
      class_prediction_bias_init: constant value to initialize bias of the last
        conv2d layer before class prediction.
      kernel_regularizer: A `tf.keras.regularizers.Regularizer` object for
        Conv2D. Default is None.
      bias_regularizer: A `tf.keras.regularizers.Regularizer` object for Conv2D.
      **kwargs: Additional keyword arguments to be passed.
    """
    super(SSDHead, self).__init__(**kwargs)
    self._min_level = min_level
    self._max_level = max_level
    self._num_classes = num_classes
    self._num_anchors_per_location_list = num_anchors_per_location_list
    self._is_training = is_training
    self._kernel_size = kernel_size
    self._use_depthwise = use_depthwise
    self._activation = activation
    self._use_sync_bn = use_sync_bn
    self._freeze_norm = freeze_norm
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._add_background_class = add_background_class
    self._class_prediction_bias_init = class_prediction_bias_init
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer

    if tf.keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1
    self._activation = tf_utils.get_activation(activation)

    self._conv_hyperparams = {
        'kernel_initializer': tf.keras.initializers.RandomNormal(stddev=0.03),
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer
    }
    
    self._bn_hyperparams = {
        'axis': self._bn_axis,
        'momentum': self._norm_momentum,
        'epsilon': self._norm_epsilon,
    }
    self._cls_convs = []
    self._box_convs = []

  def _build_box_head(self, num_pred_per_location: int):
    box_encoder_layers = []
    if self._use_depthwise:
      box_encoder_layers.extend([
          tf.keras.layers.DepthwiseConv2D(
              kernel_size=[self._kernel_size, self._kernel_size],
              padding='SAME',
              depth_multiplier=1,
              strides=1,
              dilation_rate=1,
              **self._conv_hyperparams),
          nn_layers.build_freezable_batch_norm(
              use_bn=True,
              use_sync_bn=self._use_sync_bn,
              training=(self._is_training and not self._freeze_norm),
              **self._bn_hyperparams),
          self._activation_fn,
          tf.keras.layers.Conv2D(
              filters=num_pred_per_location * 4,
              kernel_size=[1, 1],
              use_bias=True,
              **self._conv_hyperparams)
      ])
    else:
      box_encoder_layers.append(
          tf.keras.layers.Conv2D(
              filters=num_pred_per_location * 4,
              kernel_size=[self._kernel_size, self._kernel_size],
              padding='SAME',
              use_bias=True,
              bias_initializer=tf.zeros_initializer(),
              **self._conv_hyperparams)
      )
    return box_encoder_layers
  
  def _build_class_head(self,
                        num_class_slots: int,
                        num_pred_per_location: int):
    class_encoder_layers = []
    if self._use_depthwise:
      class_encoder_layers.extend([
          tf.keras.layers.DepthwiseConv2D(
              kernel_size=[self._kernel_size, self._kernel_size],
              padding='SAME',
              depth_multiplier=1,
              strides=1,
              dilation_rate=1,
              **self._conv_hyperparams),
          nn_layers.build_freezable_batch_norm(
              use_bn=True,
              use_sync_bn=self._use_sync_bn,
              training=(self._is_training and not self._freeze_norm),
              **self._bn_hyperparams),
          self._activation_fn,
          tf.keras.layers.Conv2D(
              filters=num_pred_per_location * num_class_slots,
              kernel_size=[1, 1],
              use_bias=True,
              **self._conv_hyperparams)
      ])
    else:
      class_encoder_layers.append(
          tf.keras.layers.Conv2D(
              filters=num_pred_per_location * num_class_slots,
              kernel_size=[self._kernel_size, self._kernel_size],
              padding='SAME',
              use_bias=True,
              bias_initializer=tf.constant_initializer(
                  self._class_prediction_bias_init),
              **self._conv_hyperparams)
      )
    return class_encoder_layers

  def build(self, input_shape: Union[tf.TensorShape, List[tf.TensorShape]]):
    """Creates the variables of the head."""
    # Class net
    num_class_slots = (
      self._num_classes + 1 if self._add_background_class else self._num_classes)
    min_level = self._config_dict['min_level']
    max_level = self._config_dict['max_level']

    for index, level in enumerate(range(min_level,  max_level + 1)):
      self._cls_convs.append(
          self._build_class_head(
              num_class_slots=num_class_slots,
              num_pred_per_location=self._num_anchors_per_location_list[index])
      )
      self._box_convs.append(
          self._build_box_head(
              num_pred_per_location=self._num_anchors_per_location_list[index]
          )
      )
    
    super(SSDHead, self).build(input_shape)
    
  def call(self, features: Mapping[str, tf.Tensor], **kwargs):
    """Forward pass of the RetinaNet head.

    Args:
      features: A `dict` of `tf.Tensor` where
        - key: A `str` of the level of the multilevel features.
        - values: A `tf.Tensor`, the feature map tensors, whose shape is
            [batch, height_l, width_l, channels].

    Returns:
      scores: A `dict` of `tf.Tensor` which includes scores of the predictions.
        - key: A `str` of the level of the multilevel predictions.
        - values: A `tf.Tensor` of the box scores predicted from a particular
            feature level, whose shape is
            [batch, height_l, width_l, num_classes * num_anchors_per_location].
      boxes: A `dict` of `tf.Tensor` which includes coordinates of the
        predictions.
        - key: A `str` of the level of the multilevel predictions.
        - values: A `tf.Tensor` of the box scores predicted from a particular
            feature level, whose shape is
            [batch, height_l, width_l, 4 * num_anchors_per_location].
    """
    scores = {}
    boxes = {}
    
    min_level = self._config_dict['min_level']
    max_level = self._config_dict['max_level']
    for index, level in enumerate(range(min_level,  max_level + 1)):
      this_level_features = features[str(level)]
      x = this_level_features
      for head in self._box_convs[index]:
        for layer in head:
          x = layer(x)
      boxes[str(level)] = x
    
    for index, level in enumerate(range(min_level,  max_level + 1)):
      this_level_features = features[str(level)]
      x = this_level_features
      for head in self._cls_convs[index]:
        for layer in head:
          x = layer(x)
      scores[str(level)] = x
    
    return scores, boxes

  def get_config(self) -> Mapping[str, Any]:
    config = {
        'min_level': self._min_level,
        'max_level': self._max_level,
        'num_classes': self._num_classes,
        'num_anchors_per_location_list': self._num_anchors_per_location_list,
        'is_training': self._is_training,
        'kernel_size': self._kernel_size,
        'use_depthwise': self._use_depthwise,
        'activation': self._activation,
        'use_sync_bn': self._use_sync_bn,
        'freeze_norm': self._freeze_norm,
        'add_background_class': self._add_background_class,
        'class_prediction_bias_init': self._class_prediction_bias_init,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
    }
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)
