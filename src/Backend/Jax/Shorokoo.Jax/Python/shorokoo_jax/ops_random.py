"""ONNX random draws: RandomNormal, RandomUniform, their Like forms, Bernoulli, Multinomial, and
Dropout.

The values come from JAX's generator, so they cannot match another runtime's draw for draw; what
the spec fixes is kept: the distribution, the shape and element type, and that a node with a `seed`
draws the same values every run -- and as every other node with that seed and the same parameters.
A node without one draws from the run's key (`runtime.next_key`).
"""

import jax
import jax.numpy as jnp
import numpy as np

from . import runtime as _rt

_FLOATS = (np.dtype(np.float16), _rt.jax_dtype(16), np.dtype(np.float32), np.dtype(np.float64))


def _draw(sample, shape, dtype, seed):
    """`sample(key, shape, dtype)` -- a normal or uniform draw -- in `dtype` where JAX draws it,
    and in float32 then converted where it does not."""
    draw_dtype = dtype if dtype in _FLOATS else np.dtype(np.float32)
    return sample(_rt.next_key(seed), tuple(int(d) for d in shape), draw_dtype).astype(dtype)


def _normal(mean, scale):
    def sample(key, shape, dtype):
        values = jax.random.normal(key, shape, dtype)
        return values * np.asarray(scale, dtype=dtype) + np.asarray(mean, dtype=dtype)
    return sample


def _uniform(low, high):
    def sample(key, shape, dtype):
        return jax.random.uniform(key, shape, dtype, minval=np.asarray(low, dtype=dtype),
                                  maxval=np.asarray(high, dtype=dtype))
    return sample


def _like(x, dtype):
    return np.dtype(x.dtype) if dtype is None else _rt.jax_dtype(dtype)


def random_normal(*, shape, dtype=1, mean=0.0, scale=1.0, seed=None):
    return _draw(_normal(mean, scale), shape, _rt.jax_dtype(dtype), seed)


def random_uniform(*, shape, dtype=1, high=1.0, low=0.0, seed=None):
    return _draw(_uniform(low, high), shape, _rt.jax_dtype(dtype), seed)


def random_normal_like(x, /, *, dtype=None, mean=0.0, scale=1.0, seed=None):
    return _draw(_normal(mean, scale), x.shape, _like(x, dtype), seed)


def random_uniform_like(x, /, *, dtype=None, high=1.0, low=0.0, seed=None):
    return _draw(_uniform(low, high), x.shape, _like(x, dtype), seed)


def bernoulli(x, /, *, dtype=None, seed=None):
    """Bernoulli: 1 with probability `x`, element by element, of `dtype` or `x`'s."""
    uniform = _draw(_uniform(0.0, 1.0), x.shape, np.dtype(np.float64), seed)
    return (uniform < x.astype(np.float64)).astype(_like(x, dtype))


def multinomial(x, /, *, dtype=6, sample_size=1, seed=None):
    """Multinomial: `sample_size` class indices per row of the [batch, class] unnormalized
    log-probabilities `x`, drawn with replacement."""
    logits = jnp.asarray(x, dtype=np.float64)
    samples = jax.random.categorical(_rt.next_key(seed), logits, axis=-1, shape=(sample_size, x.shape[0]))
    return samples.T.astype(_rt.jax_dtype(dtype))


def dropout(data, ratio_in=None, training_mode=None, /, *, seed=None, ratio=None, _outputs):
    """Dropout, returning its output and its mask. Outside training -- no `training_mode`, or a
    false one -- or at a ratio of 0 it is the identity and the mask is all true; in training each
    element is kept with probability 1 - ratio, and the kept ones are scaled by 1 / (1 - ratio).
    `ratio` is an input from opset 12, an attribute before it. A training mode or a ratio the graph
    computes from an input is applied as a value: the draw is made and selected by it."""
    everything = np.ones(data.shape, dtype=bool)
    if training_mode is None or (_rt.concrete(training_mode) and not np.asarray(training_mode).reshape(-1)[0]):
        return (data, everything)[:_outputs]
    if ratio_in is None:
        rate = np.asarray(0.5 if ratio is None else ratio, dtype=np.float32)
    else:
        rate = ratio_in.reshape(-1)[0].astype(np.float32)
    if _rt.concrete(rate) and rate == 0:
        return (data, everything)[:_outputs]
    # A uniform draw is below 1, so at a ratio of 0 every element is kept, scaled by 1.
    uniform = _draw(_uniform(0.0, 1.0), data.shape, np.dtype(np.float32), seed)
    mask = uniform >= rate
    scale = (1 / (1 - rate.astype(np.float64))).astype(data.dtype)
    output = jnp.where(mask, data * scale, jnp.zeros((), dtype=data.dtype))
    if not _rt.concrete(training_mode):
        training = training_mode.reshape(-1)[0].astype(bool)
        output = jnp.where(training, output, data)
        mask = jnp.where(training, mask, True)
    return (output, mask)[:_outputs]
