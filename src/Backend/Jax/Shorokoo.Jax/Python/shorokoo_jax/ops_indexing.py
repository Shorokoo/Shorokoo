"""ONNX operators that index into a tensor, over numpy and jax arrays: Gather, GatherElements,
GatherND, Slice, ScatterElements, ScatterND and TopK. (Compress, Unique and NonZero make a shape
their input's values decide, and the JAX backend refuses them.)

Indices may be negative, counting from the end of their axis, as ONNX allows. Slice's starts, ends,
axes and steps and TopK's k decide the output's shape, so they must be concrete (`runtime.ints`);
indices into the data need not be. A scatter writes into a copy: `.at[...]` of a jax array, fancy
assignment into a copy of a numpy one.
"""

import math

import jax.numpy as jnp
import numpy as np

from . import runtime as _rt


def _normalized(xp, indices, length):
    indices = indices.astype(np.int64)
    return xp.where(indices < 0, indices + length, indices)


def gather(data, indices, /, *, axis=0):
    xp = _rt.xp(data, indices)
    axis = axis % data.ndim
    return xp.take(data, _normalized(xp, indices, data.shape[axis]), axis=axis)


def _cut(data, shape, axis):
    """`data` cut to `shape` on every axis but `axis`: an index tensor may be shorter than the
    data on those, and take_along_axis wants them equal."""
    return data[tuple(slice(None) if a == axis else slice(0, n) for a, n in enumerate(shape))]


def gather_elements(data, indices, /, *, axis=0):
    xp = _rt.xp(data, indices)
    axis = axis % data.ndim
    index = _normalized(xp, indices, data.shape[axis])
    return xp.take_along_axis(_cut(data, indices.shape, axis), index, axis=axis)


def _clamped(value, length, lowest, highest):
    """A Slice start or end counted from the front, clamped to [lowest, highest]."""
    if value < 0:
        value += length
    return min(max(value, lowest), highest)


def slice_(data, starts_in=None, ends_in=None, axes_in=None, steps_in=None, /, *, starts=None, ends=None, axes=None):
    """Slice, from opset 10 (inputs) or before it (attributes)."""
    starts = _rt.ints(starts_in, "Slice", "its starts") if starts_in is not None else list(starts)
    ends = _rt.ints(ends_in, "Slice", "its ends") if ends_in is not None else list(ends)
    if axes_in is not None:
        axes = _rt.ints(axes_in, "Slice", "its axes")
    axes = list(range(len(starts))) if axes is None else list(axes)
    steps = _rt.ints(steps_in, "Slice", "its steps") if steps_in is not None else [1] * len(starts)

    index = [slice(None)] * data.ndim
    for start, end, axis, step in zip(starts, ends, axes, steps):
        axis = axis % data.ndim
        length = data.shape[axis]
        if step > 0:
            first = _clamped(start, length, 0, length)
            last = _clamped(end, length, 0, length)
            index[axis] = slice(first, max(last, first), step)
        else:
            # A negative step starts at an element, and ends at one or at -1, before the first.
            first = _clamped(start, length, 0, length - 1)
            last = _clamped(end, length, -1, length - 1)
            index[axis] = slice(first, last if last >= 0 else None, step) if first > last else slice(0, 0)
    return data[tuple(index)]


def _offsets(xp, indices, shape):
    """The row-major offset, into `shape`, of each index tuple along `indices`' last axis."""
    index = indices.astype(np.int64)
    index = xp.where(index < 0, index + np.array(shape, dtype=np.int64), index)
    strides = np.array([math.prod(shape[i + 1:]) for i in range(len(shape))], dtype=np.int64)
    return (index * strides).sum(-1)


def gather_nd(data, indices, /, *, batch_dims=0):
    """GatherND: each index tuple along `indices`' last axis picks a slice of `data`, below its
    first `batch_dims` axes, which `data` and `indices` share."""
    xp = _rt.xp(data, indices)
    b = batch_dims
    k = indices.shape[-1]
    batch = list(data.shape[:b])
    batch_count = math.prod(batch)
    picked = list(data.shape[b:b + k])
    rest = list(data.shape[b + k:])
    lookups = list(indices.shape[b:-1])
    offsets = _offsets(xp, indices, picked).reshape(batch_count, -1)
    offsets = offsets + np.arange(batch_count, dtype=np.int64)[:, None] * math.prod(picked)
    flat = data.reshape([batch_count * math.prod(picked)] + rest)
    return xp.take(flat, offsets.reshape(-1), axis=0).reshape(batch + lookups + rest)


# ---- scatters ------------------------------------------------------------------------------

_AT = {"none": "set", "add": "add", "mul": "multiply", "max": "max", "min": "min"}
_UFUNCS = {"add": np.add, "mul": np.multiply, "max": np.maximum, "min": np.minimum}


def _scattered(xp, data, where, updates, reduction):
    """`data` with `updates` written at `where` (an index tuple), or combined into it by
    `reduction`; repeated positions accumulate."""
    if xp is np:
        result = np.array(data, copy=True)
        if reduction == "none":
            result[where] = updates
        else:
            _UFUNCS[reduction].at(result, where, updates)
        return result
    return getattr(jnp.asarray(data).at[where], _AT[reduction])(updates)


def scatter_elements(data, indices, updates, /, *, axis=0, reduction="none"):
    xp = _rt.xp(data, indices, updates)
    axis = axis % data.ndim
    where = list(np.indices(indices.shape, sparse=True))
    where[axis] = _normalized(xp, indices, data.shape[axis])
    return _scattered(xp, data, tuple(where), updates, reduction)


def scatter_nd(data, indices, updates, /, *, reduction="none"):
    """ScatterND: each index tuple along `indices`' last axis names a slice of `data` that the
    matching slice of `updates` replaces or, with a reduction, is combined into."""
    xp = _rt.xp(data, indices, updates)
    k = indices.shape[-1]
    picked = list(data.shape[:k])
    rest = list(data.shape[k:])
    offsets = _offsets(xp, indices.reshape(-1, k), picked)
    flat = data.reshape([math.prod(picked)] + rest)
    source = updates.reshape([offsets.shape[0]] + rest)
    return _scattered(xp, flat, offsets, source, reduction).reshape(data.shape)


# ---- selections ----------------------------------------------------------------------------

def top_k(x, k, /, *, axis=-1, largest=1, sorted=1):
    """TopK: the k largest (smallest) elements along `axis` and their indices, in order, equal
    values in the order of their indices, as the spec requires. The order is kept even where
    `sorted` is 0, since any order is allowed then. The values are gathered from `x` by the
    indices, so they carry its gradient."""
    count = int(_rt.number(k, "TopK", "its k"))
    xp = _rt.xp(x)
    axis = axis % x.ndim
    if largest:
        # A stable descending order: a stable ascending sort of the reversed axis, reversed, puts
        # equal values back in the order of their indices.
        length = x.shape[axis]
        order = length - 1 - xp.flip(xp.argsort(xp.flip(x, axis), axis=axis, stable=True), axis)
    else:
        order = xp.argsort(x, axis=axis, stable=True)
    indices = _rt.detach(order[tuple(slice(0, count) if a == axis else slice(None) for a in range(x.ndim))]).astype(np.int64)
    return xp.take_along_axis(x, indices, axis=axis), indices
