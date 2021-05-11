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

"""Contains the definitions of Multi-resolution Feature Map."""
from typing import Any, Mapping, Optional, List, Callable, Dict

# Import libraries
from functools import partial
import tensorflow as tf

from official.modeling import tf_utils
from official.vision.beta.modeling.layers import nn_layers
from official.vision.beta.ops import preprocess_ops

layers = tf.keras.layers


def get_depth_fn(
    depth_multiplier: float,
    min_depth: int) -> Callable[[int], int]:
  """Builds a callable to compute depth (output channels) of conv filters.

  Args:
    depth_multiplier: A `float` multiplier for the nominal depth.
    min_depth: An `int` lower bound on the depth of filters.

  Returns:
    A callable that takes in a nominal depth and returns the depth to use.
  """
  
  def multiply_depth(depth: int) -> int:
    new_depth = int(depth * depth_multiplier)
    return max(new_depth, min_depth)
  
  return multiply_depth


def validate_feature_map_layout(
    feature_map_layout: Mapping[str, List]) -> (bool, str):
  """Validate whether the provided layout is valid."""
  if not feature_map_layout:
    return False, 'The feature_map_layout cannot be empty.'
  
  if ('from_layer' not in feature_map_layout
      or 'layer_depth' not in feature_map_layout):
    return False, 'Both from_layer and layer_depth should be provided.'
  
  if (len(feature_map_layout['from_layer'])
      != len(feature_map_layout['layer_depth'])):
    return False, 'Lengths of from_layer and layer_depth are not equal.'
  
  return True, ''


@tf.keras.utils.register_keras_serializable(package='Vision')
class MRFM(tf.keras.Model):
  """Creates a multi resolution feature maps from input image features.

  This implements the paper:
  A Keras model that generates multi-scale feature maps for detection as in the
  SSD papers by Liu et al: https://arxiv.org/pdf/1512.02325v2.pdf, See Sec 2.1.

  More specifically, when called on inputs it performs the following two tasks:
  1) If a layer name is provided in the configuration, returns that layer as a
     feature map.
  2) If a layer name is left as an empty string, constructs a new feature map
     based on the spatial shape and depth configuration. Note that the current
     implementation only supports generating new layers using convolution of
     stride 2 resulting in a spatial resolution reduction by a factor of 2.
     By default convolution kernel size is set to 3, and it can be customized
     by caller.

  An example of the configuration for Inception V3:
  {
    'from_layer': ['Mixed_5d', 'Mixed_6e', 'Mixed_7c', '', '', ''],
    'layer_depth': [-1, -1, -1, 512, 256, 128]
  }

  When this feature generator object is called on input image_features:
    Args:
      image_features: A dictionary of handles to activation tensors from the
        base feature extractor.

    Returns:
      feature_maps: an OrderedDict mapping keys (feature map names) to
        tensors where each tensor has shape [batch, height_i, width_i, depth_i].
  """
  
  def __init__(
      self,
      input_specs: Mapping[str, tf.TensorShape],
      feature_map_layout: Mapping[str, List],
      depth_multiplier: float,
      min_depth: int,
      is_training: bool = True,
      insert_1x1_conv: bool = True,
      kernel_size: int = 3,
      use_explicit_padding: bool = False,
      use_depthwise: bool = False,
      activation: str = 'relu',
      use_sync_bn: bool = False,
      freeze_norm: bool = False,
      norm_momentum: float = 0.99,
      norm_epsilon: float = 0.001,
      kernel_initializer: str = 'VarianceScaling',
      kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      **kwargs):
    """Initializes a multi resolution feature maps.

    Args:
      input_specs: A `dict` of input specifications. A dictionary consists of
        {level: TensorShape} from a backbone.
      feature_map_layout: Dictionary of specifications for the feature map
        layouts in the following format:
        {
          'from_layer': ['2', '3', '4', '', '', ''],
          'layer_depth': [-1, -1, -1, 512, 256, 128]
        }
        If 'from_layer' is specified, the specified feature map is directly used
        as a box predictor layer, and the layer_depth is directly inferred from
        the feature map (instead of using the provided 'layer_depth' parameter).
        In this case, the convention is to set 'layer_depth' to -1 for clarity.
        Otherwise, if 'from_layer' is an empty string, then the box predictor
        layer will be built from the previous layer using convolution
        operations. Note that the current implementation only supports
        generating new layers using convolutions of stride 2 (resulting in a
        spatial resolution reduction by a factor of 2), and will be extended to
        a more flexible design. Convolution kernel size is set to 3 by default,
        and can be customized by 'kernel_size' parameter.
        The created convolution operation will be a normal 2D convolution by
        default, and a depthwise convolution followed by 1x1 convolution if
        'use_depthwise' is set to True.
      depth_multiplier: A `float` depth multiplier for convolutional layers.
      min_depth: An `int` minimum depth for convolutional layers.
      is_training: A `bool` indicating whether the feature generator is
        in training mode.
      insert_1x1_conv: A `bool` indicating whether an additional 1x1
        convolution should be inserted before shrinking the feature map.
      kernel_size: An `int` kernel size used for building extra feature layers.
      use_explicit_padding: A `bool` indicating whether explicit padding used.
      use_depthwise: A `bool` indicating whether depthwise convolution is used
        for building extra feature layers.
      activation: A `str` name of the activation function.
      use_sync_bn: A `bool`. If True, use synchronized batch normalization.
      freeze_norm: A `bool`. Whether to freeze batch norm parameters during
        training or not. When training with a small batch size (e.g. 1), it is
        desirable to freeze batch norm update and use pretrained batch norm
        params.
      norm_momentum: A `float` of normalization momentum for the moving average.
      norm_epsilon: A `float` added to variance to avoid dividing by zero.
      kernel_initializer: A `str` name of kernel_initializer for convolutional
        layers.
      kernel_regularizer: A `tf.keras.regularizers.Regularizer` object for
        Conv2D. Default is None.
      bias_regularizer: A `tf.keras.regularizers.Regularizer` object for Conv2D.
      **kwargs: Additional keyword arguments to be passed.
    """
    
    is_valid, detail = validate_feature_map_layout(feature_map_layout)
    if not is_valid:
      raise ValueError(detail)
    
    self._input_specs = input_specs
    self._feature_map_layout = feature_map_layout
    self._depth_multiplier = depth_multiplier
    self._min_depth = min_depth
    self._is_training = is_training
    self._insert_1x1_conv = insert_1x1_conv
    self._kernel_size = kernel_size
    self._use_explicit_padding = use_explicit_padding
    self._use_depthwise = use_depthwise
    self._activation = activation
    self._use_sync_bn = use_sync_bn
    self._freeze_norm = freeze_norm
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    
    if tf.keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1
    
    self._conv_hyperparams = {
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer
    }
    self._bn_hyperparams = {
        'axis': self._bn_axis,
        'momentum': self._norm_momentum,
        'epsilon': self._norm_epsilon
    }
    self._depth_fn = get_depth_fn(depth_multiplier, min_depth)
    self._activation_fn = tf.keras.layers.Activation(
        tf_utils.get_activation(activation))
    
    # Get input feature pyramid from backbone.
    inputs = self._build_input_pyramid(input_specs)
    
    feats = {}
    
    # The feature_map_layout is in the format of
    # {
    #     'from_layer': ['2', '3', '4', '', '', ''],
    #     'layer_depth': [-1, -1, -1, 512, 256, 128]
    # }
    pre_layer = 0
    base_from_layer = ''
    for index, from_layer in enumerate(feature_map_layout['from_layer']):
      if from_layer:
        feats[from_layer] = inputs[from_layer]
        pre_layer = int(from_layer)
        base_from_layer = from_layer
      else:
        # Build extra feature layer
        x = self._build_extra_feature_layer(
            inputs=feats[str(pre_layer)],
            layer_depth=feature_map_layout['layer_depth'][index],
            layer_index=index,
            base_from_layer=base_from_layer)
        feats[str(pre_layer + 1)] = x
        pre_layer = pre_layer + 1
    
    self._output_specs = {
        str(level): feats[str(level)].get_shape()
        for level in feats.keys()
    }
    
    super(MRFM, self).__init__(inputs=inputs, outputs=feats, **kwargs)
  
  def _build_input_pyramid(
      self,
      input_specs: Mapping[str, tf.TensorShape]) -> Dict[str, tf.keras.Input]:
    assert isinstance(input_specs, dict)
    
    inputs = {}
    for level, spec in input_specs.items():
      inputs[level] = tf.keras.Input(shape=spec[1:])
    return inputs
  
  def _build_extra_feature_layer(
      self,
      inputs: tf.Tensor,
      layer_depth: int,
      layer_index: int,
      base_from_layer: str) -> tf.Tensor:
    """Build the extra feature layer."""
    layer_name = '{}_Conv2d_{}_{}x{}_s2_{}'.format(
        base_from_layer, layer_index, self._kernel_size, self._kernel_size,
        self._depth_fn(layer_depth))
    x = inputs
    if self._insert_1x1_conv:
      x = layers.Conv2D(self._depth_fn(layer_depth // 2),
                        [1, 1],
                        padding='SAME',
                        strides=1,
                        **self._conv_hyperparams)(x)
      x = nn_layers.build_freezable_batch_norm(
          use_bn=True,
          use_sync_bn=self._use_sync_bn,
          training=(self._is_training and not self._freeze_norm),
          **self._bn_hyperparams)(x)
      x = self._activation_fn(x)
    
    stride = 2
    padding = 'VALID' if self._use_explicit_padding else 'SAME'
    if self._use_explicit_padding:
      kernel_fixed_padding = partial(preprocess_ops.fixed_padding,
                                     self._kernel_size)
      x = layers.Lambda(kernel_fixed_padding)(x)
    if self._use_depthwise:
      x = layers.DepthwiseConv2D(
          kernel_size=[self._kernel_size, self._kernel_size],
          depth_multiplier=1,
          padding=padding,
          strides=stride,
          **self._conv_hyperparams)(x)
      x = nn_layers.build_freezable_batch_norm(
          use_bn=True,
          use_sync_bn=self._use_sync_bn,
          training=(self._is_training and not self._freeze_norm),
          **self._bn_hyperparams)(x)
      x = self._activation_fn(x)
      
      x = layers.Conv2D(
          filters=self._depth_fn(layer_depth),
          kernel_size=[1, 1],
          padding='SAME',
          strides=1,
          **self._conv_hyperparams)(x)
      x = nn_layers.build_freezable_batch_norm(
          use_bn=True,
          use_sync_bn=self._use_sync_bn,
          training=(self._is_training and not self._freeze_norm),
          **self._bn_hyperparams)(x)
      x = self._activation_fn(x)
    else:
      x = layers.Conv2D(
          filters=self._depth_fn(layer_depth),
          kernel_size=[self._kernel_size, self._kernel_size],
          padding=padding,
          strides=stride,
          **self._conv_hyperparams)(x)
      x = nn_layers.build_freezable_batch_norm(
          use_bn=True,
          use_sync_bn=self._use_sync_bn,
          training=(self._is_training and not self._freeze_norm),
          **self._bn_hyperparams)(x)
      x = self._activation_fn(x)
    return tf.identity(x, name=layer_name)
  
  def get_config(self) -> Mapping[str, Any]:
    config = {
        'input_specs': self._input_specs,
        'feature_map_layout': self._feature_map_layout,
        'depth_multiplier': self._depth_multiplier,
        'min_depth': self._min_depth,
        'is_training': self._is_training,
        'insert_1x1_conv': self._insert_1x1_conv,
        'kernel_size': self._kernel_size,
        'use_explicit_padding': self._use_explicit_padding,
        'use_depthwise': self._use_depthwise,
        'activation': self._activation,
        'use_sync_bn': self._use_sync_bn,
        'freeze_norm': self._freeze_norm,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
    }
    base_config = super(MRFM, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
  
  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)
  
  @property
  def output_specs(self) -> Mapping[str, tf.TensorShape]:
    """A dict of {level: TensorShape} pairs for the model output."""
    return self._output_specs
