<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.InverseGaussian" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.InverseGaussian

Create a random variable for InverseGaussian.

``` python
tfp.edward2.InverseGaussian(
    *args,
    **kwargs
)
```



Defined in [`python/edward2/interceptor.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/edward2/interceptor.py).

<!-- Placeholder for "Used in" -->

See InverseGaussian for more details.

#### Returns:
RandomVariable.


#### Original Docstring for Distribution

Constructs inverse Gaussian distribution with `loc` and `concentration`.

#### Args:

* <b>`loc`</b>: Floating-point `Tensor`, the loc params. Must contain only positive
  values.
* <b>`concentration`</b>: Floating-point `Tensor`, the concentration params.
  Must contain only positive values.
* <b>`validate_args`</b>: Python `bool`, default `False`. When `True` distribution
  parameters are checked for validity despite possibly degrading runtime
  performance. When `False` invalid inputs may silently render incorrect
  outputs.
  Default value: `False` (i.e. do not validate args).
* <b>`allow_nan_stats`</b>: Python `bool`, default `True`. When `True`, statistics
  (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
  result is undefined. When `False`, an exception is raised if one or
  more of the statistic's batch members are undefined.
  Default value: `True`.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this class.
  Default value: 'InverseGaussian'.