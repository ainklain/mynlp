<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.vi.squared_hellinger" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.vi.squared_hellinger

The Squared-Hellinger Csiszar-function in log-space.

``` python
tfp.vi.squared_hellinger(
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

The Squared-Hellinger Csiszar-function is:

```none
f(u) = (sqrt(u) - 1)**2
```

This Csiszar-function induces a symmetric f-Divergence, i.e.,
`D_f[p, q] = D_f[q, p]`.

Warning: this function makes non-log-space calculations and may therefore be
numerically unstable for `|logu| >> 0`.

#### Args:

* <b>`logu`</b>: `float`-like `Tensor` representing `log(u)` from above.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this function.


#### Returns:

* <b>`squared_hellinger_of_u`</b>: `float`-like `Tensor` of the Csiszar-function
  evaluated at `u = exp(logu)`.