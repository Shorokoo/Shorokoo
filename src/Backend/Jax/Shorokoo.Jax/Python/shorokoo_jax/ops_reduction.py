"""ONNX reductions over numpy and jax arrays: the Reduce* family, ArgMax and ArgMin.

Axes arrive as an input from opset 13 (ReduceSum) or 18 (the others) and as an attribute before
that; both are accepted and the input wins, and an input must be concrete (`runtime.ints`). An empty
axes list reduces everything unless noop_with_empty_axes is set, in which case the input comes back
unchanged. Every reduction keeps the input's element type, as ONNX does and numpy's own integer sums
do not. A 16-bit float is reduced in float32, as the other backends accumulate it.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np

from . import runtime as _rt
from .ops_elementwise import div as _div

_HALF = (np.dtype(np.float16), np.dtype(jnp.bfloat16))


def _dims(operator, data, axes_input, axes, noop_with_empty_axes):
    """The dimensions to reduce, or None for the identity."""
    if axes_input is not None:
        axes = _rt.ints(axes_input, operator, "its axes")
    if not axes:
        return None if noop_with_empty_axes else list(range(data.ndim))
    rank = data.ndim
    return sorted({a % rank for a in axes}) if rank else []


def _filled(xp, data, dims, keepdims, value):
    shape = [1 if d in dims else n for d, n in enumerate(data.shape)] if keepdims \
        else [n for d, n in enumerate(data.shape) if d not in dims]
    return xp.full(shape, value, dtype=data.dtype)


def _reduce(operator, fn, empty_value, data, axes_input, axes, keepdims, noop_with_empty_axes):
    """`fn(xp, x, dims, keep)` over the dimensions the node names, cast back to data's type."""
    dims = _dims(operator, data, axes_input, axes, noop_with_empty_axes)
    if dims is None:
        return data
    xp = _rt.xp(data)
    if not dims:
        # A scalar reduced over no axes: reducing a leading axis of one gives each reduction its
        # own answer for a single element (the element, its square, its absolute value...).
        return fn(xp, data.reshape(1), (0,), False).astype(data.dtype)
    if any(data.shape[d] == 0 for d in dims):
        value = empty_value(data.dtype) if callable(empty_value) else empty_value
        return _filled(xp, data, dims, keepdims, value)
    work = data.astype(np.float32) if np.dtype(data.dtype) in _HALF else data
    return fn(xp, work, tuple(dims), bool(keepdims)).astype(data.dtype)


def _accumulator(x):
    """The type an integer sum accumulates in: 64 bits, which wrap as the narrower types do once
    cast back."""
    if not _rt.is_integral(x):
        return None
    return np.uint64 if jnp.issubdtype(x.dtype, jnp.unsignedinteger) else np.int64


def _sum(xp, x, dims, keep):
    return xp.sum(x, axis=dims, keepdims=keep, dtype=_accumulator(x))


def _lowest(dtype):
    if dtype == np.bool_:
        return False
    return -math.inf if _rt.is_floating(dtype) else jnp.iinfo(dtype).min


def _highest(dtype):
    if dtype == np.bool_:
        return True
    return math.inf if _rt.is_floating(dtype) else jnp.iinfo(dtype).max


def _mean(xp, x, dims, keep):
    if not _rt.is_integral(x):
        return xp.mean(x, axis=dims, keepdims=keep)
    total = _sum(xp, x, dims, keep)
    return _div(total, np.asarray(math.prod(x.shape[d] for d in dims), dtype=total.dtype))


def _empty_mean(dtype):
    return math.nan if _rt.is_floating(dtype) else 0


def reduce_sum(data, axes_input=None, /, *, axes=None, keepdims=1, noop_with_empty_axes=0):
    return _reduce("ReduceSum", _sum, 0, data, axes_input, axes, keepdims, noop_with_empty_axes)


def reduce_mean(data, axes_input=None, /, *, axes=None, keepdims=1, noop_with_empty_axes=0):
    return _reduce("ReduceMean", _mean, _empty_mean, data, axes_input, axes, keepdims, noop_with_empty_axes)


# A maximum's or minimum's gradient is shared equally among the elements that tie for it, as
# Shorokoo's own rule shares it and jax.numpy's does.

def reduce_max(data, axes_input=None, /, *, axes=None, keepdims=1, noop_with_empty_axes=0):
    return _reduce("ReduceMax", lambda xp, x, d, k: xp.max(x, axis=d, keepdims=k), _lowest,
                   data, axes_input, axes, keepdims, noop_with_empty_axes)


def reduce_min(data, axes_input=None, /, *, axes=None, keepdims=1, noop_with_empty_axes=0):
    return _reduce("ReduceMin", lambda xp, x, d, k: xp.min(x, axis=d, keepdims=k), _highest,
                   data, axes_input, axes, keepdims, noop_with_empty_axes)


def reduce_prod(data, axes_input=None, /, *, axes=None, keepdims=1, noop_with_empty_axes=0):
    return _reduce("ReduceProd", lambda xp, x, d, k: xp.prod(x, axis=d, keepdims=k, dtype=_accumulator(x)), 1,
                   data, axes_input, axes, keepdims, noop_with_empty_axes)


def reduce_l1(data, axes_input=None, /, *, axes=None, keepdims=1, noop_with_empty_axes=0):
    return _reduce("ReduceL1", lambda xp, x, d, k: _sum(xp, xp.abs(x), d, k), 0,
                   data, axes_input, axes, keepdims, noop_with_empty_axes)


def reduce_l2(data, axes_input=None, /, *, axes=None, keepdims=1, noop_with_empty_axes=0):
    def l2(xp, x, dims, keep):
        if _rt.is_integral(x):
            return xp.sqrt(_sum(xp, x * x, dims, keep).astype(np.float64))
        return xp.sqrt(_sum(xp, x * x, dims, keep))
    return _reduce("ReduceL2", l2, 0, data, axes_input, axes, keepdims, noop_with_empty_axes)


def reduce_sum_square(data, axes_input=None, /, *, axes=None, keepdims=1, noop_with_empty_axes=0):
    return _reduce("ReduceSumSquare", lambda xp, x, d, k: _sum(xp, x * x, d, k), 0,
                   data, axes_input, axes, keepdims, noop_with_empty_axes)


def reduce_log_sum(data, axes_input=None, /, *, axes=None, keepdims=1, noop_with_empty_axes=0):
    return _reduce("ReduceLogSum", lambda xp, x, d, k: xp.log(_sum(xp, x, d, k)), -math.inf,
                   data, axes_input, axes, keepdims, noop_with_empty_axes)


def _log_sum_exp(xp, x, dims, keep):
    if xp is np:
        top = np.max(x, axis=dims, keepdims=True)
        top = np.where(np.isfinite(top), top, 0)
        total = np.log(np.sum(np.exp(x - top), axis=dims, keepdims=True)) + top
        return total if keep else np.squeeze(total, axis=dims)
    return jax.scipy.special.logsumexp(x, axis=dims, keepdims=keep)


def reduce_log_sum_exp(data, axes_input=None, /, *, axes=None, keepdims=1, noop_with_empty_axes=0):
    return _reduce("ReduceLogSumExp", _log_sum_exp, -math.inf, data, axes_input, axes, keepdims, noop_with_empty_axes)


def _arg(fn, data, axis, keepdims, select_last_index):
    xp = _rt.xp(data)
    work = data.reshape(1) if data.ndim == 0 else data
    axis = axis % work.ndim
    if select_last_index:
        flipped = xp.flip(work, axis)
        index = work.shape[axis] - 1 - getattr(xp, fn)(flipped, axis=axis, keepdims=bool(keepdims))
    else:
        index = getattr(xp, fn)(work, axis=axis, keepdims=bool(keepdims))
    return index.astype(np.int64)


def arg_max(data, *, axis=0, keepdims=1, select_last_index=0):
    return _arg("argmax", data, axis, keepdims, select_last_index)


def arg_min(data, *, axis=0, keepdims=1, select_last_index=0):
    return _arg("argmin", data, axis, keepdims, select_last_index)
