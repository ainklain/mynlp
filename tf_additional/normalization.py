

from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer

from tensorflow.python.ops import nn
from tensorflow.python.ops import array_ops
from tensorflow.python.util.tf_export import keras_export

@keras_export('keras.layers.experimental.LayerNormalization')
class LayerNormalization(Layer):
  """Layer normalization layer (Ba et al., 2016).
  Normalize the activations of the previous layer for each given example in a
  batch independently, rather than across a batch like Batch Normalization.
  i.e. applies a transformation that maintains the mean activation within each
  example close to 0 and the activation standard deviation close to 1.
  Given a tensor `inputs` of rank `R`, moments are calculated and normalization
  is performed over all axes in norm_axis.  Scaling and centering,
  if requested, is performed over all axes in params_axis.
  By default, normalization is performed over all but the first axis
  (the `HWC` if `inputs` is `NHWC`), while the `beta` and `gamma` trainable
  parameters are calculated for the rightmost axis (the `C` if `inputs` is
  `NHWC`).  Scaling and recentering is performed via broadcast of the
  `beta` and `gamma` parameters with the normalized tensor.
  The shapes of `beta` and `gamma` are
  `[inputs.shape[i] for i in (param axes)]`,
  and this part of the inputs' shape must be fully defined.
  Arguments:
    norm_axis: Integer or List. normalization will be
      performed along these dimensions. If unspecified (None), it will default
      to the dimensions `begin_norm_axis : rank(inputs)`
    params_axis: Integer or List. The (beta, gamma) dimensions: scale
      and centering parameters will have take their shapes from these axes and
      will be broadcast with the normalized inputs accordingly. If unspecified
      (None), it will default to the last dimension
    epsilon: Small float added to variance to avoid dividing by zero.
    center: If True, add offset of `beta` to normalized tensor.
        If False, `beta` is ignored.
    scale: If True, multiply by `gamma`.
      If False, `gamma` is not used.
      When the next layer is linear (also e.g. `nn.relu`),
      this can be disabled since the scaling
      will be done by the next layer.
    beta_initializer: Initializer for the beta weight.
    gamma_initializer: Initializer for the gamma weight.
    beta_regularizer: Optional regularizer for the beta weight.
    gamma_regularizer: Optional regularizer for the gamma weight.
    beta_constraint: Optional constraint for the beta weight.
    gamma_constraint: Optional constraint for the gamma weight.
    trainable: Boolean, if `True` the variables will be marked as trainable.
  Input shape:
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the samples axis)
    when using this layer as the first layer in a model.
  Output shape:
    Same shape as input.
  References:
    - [Layer Normalization](https://arxiv.org/abs/1607.06450)
  """

  def __init__(self,
               norm_axis=None,
               params_axis=-1,
               epsilon=1e-12,
               center=True,
               scale=True,
               beta_initializer='zeros',
               gamma_initializer='ones',
               beta_regularizer=None,
               gamma_regularizer=None,
               beta_constraint=None,
               gamma_constraint=None,
               trainable=True,
               name=None,
               **kwargs):
    super(LayerNormalization, self).__init__(
        name=name, trainable=trainable, **kwargs)
    if isinstance(norm_axis, list):
      self.norm_axis = norm_axis[:]
    elif isinstance(norm_axis, int):
      self.norm_axis = norm_axis
    elif norm_axis is None:
      self.norm_axis = None
    else:
      raise TypeError('norm_axis must be int or list or None, type given: %s'
                      % type(norm_axis))

    if isinstance(params_axis, list):
      self.params_axis = params_axis[:]
    elif isinstance(params_axis, int):
      self.params_axis = params_axis
    else:
      raise TypeError('params_axis must be int or list, type given: %s'
                      % type(params_axis))

    self.epsilon = epsilon
    self.center = center
    self.scale = scale
    self.beta_initializer = initializers.get(beta_initializer)
    self.gamma_initializer = initializers.get(gamma_initializer)
    self.beta_regularizer = regularizers.get(beta_regularizer)
    self.gamma_regularizer = regularizers.get(gamma_regularizer)
    self.beta_constraint = constraints.get(beta_constraint)
    self.gamma_constraint = constraints.get(gamma_constraint)

    self.supports_masking = True

  def build(self, input_shape):
    ndims = len(input_shape)
    if ndims is None:
      raise ValueError('Input shape %s has undefined rank.' % input_shape)

    # Handle an unspecified norm_axis
    if self.norm_axis is None:
      self.norm_axis = list(range(1, ndims))

    # Convert axes to lists and resolve negatives
    if isinstance(self.norm_axis, int):
      self.norm_axis = [self.norm_axis]
    for idx, x in enumerate(self.norm_axis):
      if x < 0:
        self.norm_axis[idx] = ndims + x

    if isinstance(self.params_axis, int):
      self.params_axis = [self.params_axis]
    for idx, x in enumerate(self.params_axis):
      if x < 0:
        self.params_axis[idx] = ndims + x

    # Validate axes
    for x in self.norm_axis:
      if x < 0 or x >= ndims:
        raise ValueError('Invalid axis: %d' % x)
    if len(self.norm_axis) != len(set(self.norm_axis)):
      raise ValueError('Duplicate axis: %s' % self.norm_axis)

    for x in self.params_axis:
      if x < 0 or x >= ndims:
        raise ValueError('Invalid axis: %d' % x)
    if len(self.params_axis) != len(set(self.params_axis)):
      raise ValueError('Duplicate axis: %s' % self.params_axis)

    param_shape = [input_shape[dim] for dim in self.params_axis]

    if self.scale:
      self.gamma = self.add_weight(
          name='gamma',
          shape=param_shape,
          initializer=self.gamma_initializer,
          regularizer=self.gamma_regularizer,
          constraint=self.gamma_constraint,
          trainable=True,
          experimental_autocast=False)
    else:
      self.gamma = None

    if self.center:
      self.beta = self.add_weight(
          name='beta',
          shape=param_shape,
          initializer=self.beta_initializer,
          regularizer=self.beta_regularizer,
          constraint=self.beta_constraint,
          trainable=True,
          experimental_autocast=False)
    else:
      self.beta = None

  def call(self, inputs):
    # Compute the axes along which to reduce the mean / variance
    input_shape = inputs.get_shape()
    ndims = len(input_shape)

    # Calculate the moments on the last axis (layer activations).
    mean, variance = nn.moments(inputs, self.norm_axis, keep_dims=True)

    # Broadcasting only necessary for norm where the params axes aren't just
    # the last dimension
    broadcast_shape = [1] * ndims
    for dim in self.params_axis:
      broadcast_shape[dim] = input_shape.dims[dim].value
    def _broadcast(v):
      if (v is not None and
          len(v.get_shape()) != ndims and
          self.params_axis != [ndims - 1]):
        return array_ops.reshape(v, broadcast_shape)
      return v
    scale, offset = _broadcast(self.gamma), _broadcast(self.beta)

    # Compute layer normalization using the batch_normalization function.
    outputs = nn.batch_normalization(
        inputs,
        mean,
        variance,
        offset=offset,
        scale=scale,
        variance_epsilon=self.epsilon)

    # If some components of the shape got lost due to adjustments, fix that.
    outputs.set_shape(input_shape)

    return outputs

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {
        'norm_axis': self.norm_axis,
        'params_axis': self.params_axis,
        'epsilon': self.epsilon,
        'center': self.center,
        'scale': self.scale,
        'beta_initializer': initializers.serialize(self.beta_initializer),
        'gamma_initializer': initializers.serialize(self.gamma_initializer),
        'beta_regularizer': regularizers.serialize(self.beta_regularizer),
        'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
        'beta_constraint': constraints.serialize(self.beta_constraint),
        'gamma_constraint': constraints.serialize(self.gamma_constraint)
    }
    base_config = super(LayerNormalization, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))