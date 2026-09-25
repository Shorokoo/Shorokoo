"""ONNX shape and data-movement operators that rearrange or make tensors without indexing into
them: Reshape, Transpose, Concat, Split, Squeeze/Unsqueeze, Shape, Size, Flatten, Expand, Tile,
Identity, ConstantOfShape, Range, Trilu, Pad, OneHot, EyeLike, ReverseSequence, TensorScatter.

Shape and Size read only their input's shape, which is known while the model is traced, so their
outputs are always concrete; an operator that takes a shape, a count or an axis as an input needs
it concrete (`runtime.ints`).
"""

import math

import jax
import jax.numpy as jnp
import numpy as np

from . import runtime as _rt


def reshape(data, shape, /, *, allowzero=0):
    target = _rt.ints(shape, "Reshape", "its target shape")
    if not allowzero:
        target = [data.shape[i] if d == 0 else d for i, d in enumerate(target)]
    return _rt.xp(data).reshape(data, target)


def transpose(data, *, perm=None):
    perm = list(reversed(range(data.ndim))) if perm is None else list(perm)
    return _rt.xp(data).transpose(data, perm)


def concat(*inputs, axis):
    return _rt.xp(*inputs).concatenate(inputs, axis=axis)


def split(data, split_input=None, /, *, axis=0, num_outputs=None, split=None, _outputs):
    sizes = (_rt.ints(split_input, "Split", "its split sizes") if split_input is not None
             else (list(split) if split is not None else None))
    length = data.shape[axis]
    if sizes is None:
        count = num_outputs if num_outputs is not None else _outputs
        # ceil(length / count) each, as far as the axis reaches: the last ones may be shorter, or
        # empty where a full chunk for every output but the last overruns it.
        chunk = -(-length // count)
        sizes = [min(chunk, max(length - chunk * i, 0)) for i in range(count)]
    xp = _rt.xp(data)
    return tuple(xp.split(data, np.cumsum(sizes)[:-1].tolist(), axis=axis))


def squeeze(data, axes_input=None, /, *, axes=None):
    if axes_input is not None:
        axes = _rt.ints(axes_input, "Squeeze", "its axes")
    if axes is None:
        axes = [d for d, n in enumerate(data.shape) if n == 1]
    rank = data.ndim
    keep = [n for d, n in enumerate(data.shape) if d not in {a % rank for a in axes}]
    return _rt.xp(data).reshape(data, keep)


def unsqueeze(data, axes_input=None, /, *, axes=None):
    if axes_input is not None:
        axes = _rt.ints(axes_input, "Unsqueeze", "its axes")
    rank = data.ndim + len(axes)
    shape = list(data.shape)
    for axis in sorted(a % rank for a in axes):
        shape.insert(axis, 1)
    return _rt.xp(data).reshape(data, shape)


def shape(data, *, start=0, end=None):
    return np.array(list(data.shape)[slice(start, end)], dtype=np.int64)


def size(data):
    return np.array(math.prod(data.shape), dtype=np.int64)


def flatten(data, *, axis=1):
    rank = data.ndim
    axis = axis % rank if rank and axis < 0 else axis
    rows = math.prod(data.shape[:axis])
    return _rt.xp(data).reshape(data, (rows, math.prod(data.shape[axis:])))


def expand(data, shape):
    target = np.broadcast_shapes(tuple(data.shape), tuple(_rt.ints(shape, "Expand", "its target shape")))
    return _rt.xp(data).broadcast_to(data, target)


def tile(data, repeats):
    return _rt.xp(data).tile(data, _rt.ints(repeats, "Tile", "its repeat counts"))


def identity(data):
    return data


def constant_of_shape(shape, /, *, value=None):
    dims = _rt.ints(shape, "ConstantOfShape", "its shape")
    if value is None:
        return np.zeros(dims, dtype=np.float32)
    return np.full(dims, np.asarray(value).reshape(-1)[0], dtype=value.dtype)


def range_(start, limit, delta):
    s = _rt.number(start, "Range", "its start")
    l = _rt.number(limit, "Range", "its limit")
    d = _rt.number(delta, "Range", "its delta")
    count = max(math.ceil((l - s) / d), 0)
    floating = _rt.is_floating(start)
    steps = np.arange(count, dtype=np.float64 if floating else np.int64)
    return (np.asarray(s, dtype=steps.dtype) + steps * np.asarray(d, dtype=steps.dtype)).astype(start.dtype)


def trilu(data, k=None, /, *, upper=1):
    diagonal = 0 if k is None else (int(_rt.number(k, "Trilu", "")) if _rt.concrete(k) else jnp.reshape(k, (-1,))[0])
    xp = _rt.xp(data, diagonal)
    rows, cols = data.shape[-2], data.shape[-1]
    offsets = xp.asarray(np.arange(cols)[None, :] - np.arange(rows)[:, None])
    keep = offsets >= diagonal if upper else offsets <= diagonal
    return xp.where(keep, data, xp.zeros((), dtype=data.dtype))


def _padding_indices(length, before, after, mode):
    """The source index of every position of a padded axis, for the modes that copy from the
    input: edge repeats the end elements, reflect mirrors about them, wrap cycles."""
    positions = np.arange(-before, length + after)
    if mode == "edge":
        return np.clip(positions, 0, length - 1)
    if mode == "wrap":
        return np.remainder(positions, length)
    if length == 1:
        return np.zeros_like(positions)
    period = 2 * (length - 1)
    folded = np.remainder(positions, period)
    return np.where(folded >= length, period - folded, folded)


def pad(data, pads_in=None, constant_value=None, axes_in=None, /, *, mode="constant", pads=None, value=0.0):
    """Pad, from opset 11 (pads, constant value and, from 18, axes as inputs) or before it
    (attributes). A negative pad crops."""
    rank = data.ndim
    widths = _rt.ints(pads_in, "Pad", "its pads") if pads_in is not None else list(pads)
    axes = [a % rank for a in _rt.ints(axes_in, "Pad", "its axes")] if axes_in is not None else list(range(rank))
    before, after = [0] * rank, [0] * rank
    for i, axis in enumerate(axes):
        before[axis], after[axis] = widths[i], widths[i + len(axes)]

    xp = _rt.xp(data, constant_value)
    cropped = tuple(slice(max(-b, 0), data.shape[a] - max(-e, 0)) for a, (b, e) in enumerate(zip(before, after)))
    result = data[cropped]
    before = [max(b, 0) for b in before]
    after = [max(e, 0) for e in after]
    if not any(before) and not any(after):
        return result

    if mode == "constant":
        widths = list(zip(before, after))
        if constant_value is None:
            return xp.pad(result, widths, mode="constant", constant_values=np.asarray(value, dtype=data.dtype))
        fill = xp.reshape(constant_value, (-1,))[0].astype(data.dtype)
        # The fill is placed as a tensor, not a number, so that where it is differentiated its
        # gradient -- the sum over the positions it fills -- reaches it.
        inside = np.pad(np.ones(result.shape, dtype=bool), widths, mode="constant", constant_values=False)
        return xp.where(inside, xp.pad(result, widths, mode="constant"), fill)
    for axis in range(rank):
        if before[axis] or after[axis]:
            index = _padding_indices(result.shape[axis], before[axis], after[axis], mode)
            result = xp.take(result, index, axis=axis)
    return result


def one_hot(indices, depth, values, /, *, axis=-1):
    """OneHot: `values[1]` where the new `axis` position is the index, `values[0]` elsewhere. A
    negative index counts back from `depth`; one still out of range selects nothing."""
    count = int(_rt.number(depth, "OneHot", "its depth"))
    xp = _rt.xp(indices, values)
    index = indices.astype(np.int64)
    index = xp.where(index < 0, index + count, index)
    axis = axis % (index.ndim + 1)
    positions = [1] * (index.ndim + 1)
    positions[axis] = count
    hot = xp.expand_dims(index, axis) == np.arange(count).reshape(positions)
    # The values are constants of the encoding, as Shorokoo's own autodiff treats them: no
    # gradient reaches them.
    values = _rt.detach(values)
    return xp.where(hot, values[1], values[0])


def eye_like(x, /, *, dtype=None, k=0):
    """EyeLike: ones on the k-th diagonal of a matrix shaped like `x`, of `dtype` or `x`'s."""
    rows, cols = x.shape
    target = x.dtype if dtype is None else _rt.jax_dtype(dtype)
    offsets = np.arange(cols)[None, :] - np.arange(rows)[:, None]
    return (offsets == k).astype(target)


def reverse_sequence(x, sequence_lens, /, *, batch_axis=1, time_axis=0):
    """ReverseSequence: the first `sequence_lens[b]` steps of each batch entry reversed along the
    time axis, the rest kept."""
    xp = _rt.xp(x, sequence_lens)
    steps = x.shape[time_axis]
    lengths = sequence_lens.astype(np.int64).reshape(1, -1)
    time = np.arange(steps).reshape(-1, 1)
    source = xp.where(time < lengths, lengths - 1 - time, time)
    if time_axis == 1:
        source = source.T
    source = xp.broadcast_to(source.reshape(list(source.shape) + [1] * (x.ndim - 2)), x.shape)
    return xp.take_along_axis(x, source, axis=time_axis)


def tensor_scatter(past_cache, update, write_indices=None, /, *, mode="linear", axis=-2):
    """TensorScatter: `update` written into each batch entry's cache along `axis`, starting at its
    write index (0 without one), wrapping around the cache in circular mode."""
    xp = _rt.xp(past_cache, update, write_indices)
    axis = axis % past_cache.ndim
    length, window = past_cache.shape[axis], update.shape[axis]
    batch = past_cache.shape[0]
    if window == 0:
        return past_cache
    starts = np.zeros(batch, dtype=np.int64) if write_indices is None else write_indices.astype(np.int64).reshape(-1)
    offsets = np.arange(length)[None, :] - starts[:, None]
    if mode == "circular":
        offsets = xp.remainder(offsets, length)
    written = (offsets >= 0) & (offsets < window)
    shape = [1] * past_cache.ndim
    shape[0], shape[axis] = batch, length
    written = xp.broadcast_to(written.reshape(shape), past_cache.shape)
    positions = xp.broadcast_to(xp.clip(offsets, 0, window - 1).reshape(shape), past_cache.shape)
    gathered = xp.take_along_axis(update, _fit(xp, positions, update.shape, axis), axis=axis)
    return xp.where(written, gathered, past_cache)


def _fit(xp, positions, shape, axis):
    """`positions` cut to `shape` on every axis but `axis`, as take_along_axis needs them."""
    return positions[tuple(slice(None) if a == axis else slice(0, n) for a, n in enumerate(shape))]
