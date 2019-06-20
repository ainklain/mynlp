<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.vi.pearson" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.vi.pearson

The Pearson Csiszar-function in log-space.

``` python
tfp.vi.pearson(
    logu,
    name=None
)
```



Defined in [`python/vi/csiszar_divergence.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/vi/csiszar_divergence.py).

<!-- Placeholder for "Used in" -->

A Csiszar-function is a member of,

```none
F = { f:R_+ to R : f convex }.
```

The Pearson Csiszar-function is:

```none
f(u) = (u - 1)**2
```

Warning: this function makes non-log-space calculations and may therefore be
numerically unstable for `|logu| >> 0`.

#### Args:

* <b>`logu`</b>: `float`-like `Tensor` representing `log(u)` from above.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this function.


#### Returns:

* <b>`pearson_of_u`</b>: `float`-like `Tensor` of the Csiszar-function evaluated at
  `u = exp(logu)`.