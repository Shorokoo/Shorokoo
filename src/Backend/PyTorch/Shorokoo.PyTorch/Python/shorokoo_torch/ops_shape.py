"""ONNX shape and data-movement operators that rearrange or make tensors without indexing into
them: Reshape, Transpose, Concat, Split, Squeeze/Unsqueeze, Shape, Size, Flatten, Expand, Tile,
Identity, ConstantOfShape, Range, Trilu, Pad, OneHot, EyeLike, ReverseSequence, TensorScatter.

A result may be a view of an input; the run hands every output over in memory of its own, so a
view never escapes to the caller.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

from . import runtime as _rt


def _ints(tensor):
    return [int(v) for v in tensor.reshape(-1).tolist()]


def _shape_of(value):
    return list(value.shape)


def reshape(data, shape, /, *, allowzero=0):
    target = _ints(shape)
    if not allowzero:
        target = [data.shape[i] if d == 0 else d for i, d in enumerate(target)]
    if _rt.is_strings(data):
        return data.reshape(target)
    return data.reshape(target)


def transpose(data, *, perm=None):
    perm = list(reversed(range(data.ndim))) if perm is None else list(perm)
    if _rt.is_strings(data):
        return data.transpose(perm)
    return data.permute(perm)


def concat(*inputs, axis):
    if _rt.is_strings(inputs[0]):
        return np.concatenate(inputs, axis=axis)
    return torch.cat(inputs, axis)


def split(data, split_input=None, /, *, axis=0, num_outputs=None, split=None, _outputs):
    sizes = _ints(split_input) if split_input is not None else (list(split) if split is not None else None)
    length = data.shape[axis]
    if sizes is None:
        count = num_outputs if num_outputs is not None else _outputs
        chunk = -(-length // count)
        sizes = [chunk] * (count - 1) + [length - chunk * (count - 1)]
    if _rt.is_strings(data):
        return tuple(np.split(data, np.cumsum(sizes)[:-1], axis=axis))
    return tuple(torch.split(data, sizes, axis))


def squeeze(data, axes_input=None, /, *, axes=None):
    if axes_input is not None:
        axes = _ints(axes_input)
    if axes is None:
        axes = [d for d, n in enumerate(data.shape) if n == 1]
    rank = data.ndim
    keep = [n for d, n in enumerate(data.shape) if d not in {a % rank for a in axes}]
    return data.reshape(keep)


def unsqueeze(data, axes_input=None, /, *, axes=None):
    if axes_input is not None:
        axes = _ints(axes_input)
    rank = data.ndim + len(axes)
    shape = list(data.shape)
    for axis in sorted(a % rank for a in axes):
        shape.insert(axis, 1)
    return data.reshape(shape)


def shape(data, *, start=0, end=None):
    dims = _shape_of(data)
    return torch.tensor(dims[slice(start, end)], dtype=torch.int64, device=_rt.device())


def size(data):
    return torch.tensor(math.prod(_shape_of(data)), dtype=torch.int64, device=_rt.device())


def flatten(data, *, axis=1):
    rank = data.ndim
    axis = axis % rank if rank and axis < 0 else axis
    rows = math.prod(data.shape[:axis])
    return data.reshape(rows, math.prod(data.shape[axis:]))


def expand(data, shape):
    target = torch.broadcast_shapes(tuple(data.shape), tuple(_ints(shape)))
    return data.expand(target)


def tile(data, repeats):
    counts = _ints(repeats)
    if _rt.is_strings(data):
        return np.tile(data, counts)
    return data.repeat(counts) if counts else data.clone()


def identity(data):
    return data


def constant_of_shape(shape, /, *, value=None):
    dims = _ints(shape)
    if value is None:
        return torch.zeros(dims, dtype=torch.float32, device=_rt.device())
    return torch.full(dims, value.reshape(-1)[0].item(), dtype=value.dtype, device=_rt.device())


def range_(start, limit, delta):
    s, l, d = start.item(), limit.item(), delta.item()
    count = max(math.ceil((l - s) / d), 0)
    steps = torch.arange(count, device=_rt.device(), dtype=torch.float64 if start.dtype.is_floating_point else torch.int64)
    return (s + steps * d).to(start.dtype)


def trilu(data, k=None, /, *, upper=1):
    diagonal = int(k.reshape(-1)[0]) if k is not None else 0
    return torch.triu(data, diagonal) if upper else torch.tril(data, diagonal)


def _padding_indices(length, before, after, mode, device):
    """The source index of every position of a padded axis, for the modes that copy from the
    input: edge repeats the end elements, reflect mirrors about them, wrap cycles."""
    positions = torch.arange(-before, length + after, device=device)
    if mode == "edge":
        return positions.clamp(0, length - 1)
    if mode == "wrap":
        return torch.remainder(positions, length)
    if length == 1:
        return torch.zeros_like(positions)
    period = 2 * (length - 1)
    folded = torch.remainder(positions, period)
    return torch.where(folded >= length, period - folded, folded)


def pad(data, pads_in=None, constant_value=None, axes_in=None, /, *, mode="constant", pads=None, value=0.0):
    """Pad, from opset 11 (pads, constant value and, from 18, axes as inputs) or before it
    (attributes). A negative pad crops."""
    rank = data.ndim
    widths = _ints(pads_in) if pads_in is not None else list(pads)
    axes = [a % rank for a in _ints(axes_in)] if axes_in is not None else list(range(rank))
    before, after = [0] * rank, [0] * rank
    for i, axis in enumerate(axes):
        before[axis], after[axis] = widths[i], widths[i + len(axes)]

    result = data
    for axis in range(rank):
        start, stop = max(-before[axis], 0), result.shape[axis] - max(-after[axis], 0)
        if start or stop != result.shape[axis]:
            result = result.narrow(axis, start, max(stop - start, 0))
    before = [max(b, 0) for b in before]
    after = [max(a, 0) for a in after]
    if not any(before) and not any(after):
        return result

    if mode == "constant":
        fill = constant_value.reshape(-1)[0].item() if constant_value is not None else value
        widths = []
        for axis in reversed(range(rank)):
            widths += [before[axis], after[axis]]
        return F.pad(result, widths, mode="constant", value=fill)
    for axis in range(rank):
        if before[axis] or after[axis]:
            index = _padding_indices(result.shape[axis], before[axis], after[axis], mode, result.device)
            result = torch.index_select(result, axis, index)
    return result


def one_hot(indices, depth, values, /, *, axis=-1):
    """OneHot: `values[1]` where the new `axis` position is the index, `values[0]` elsewhere. A
    negative index counts back from `depth`; one still out of range selects nothing."""
    count = int(depth.reshape(-1)[0].item())
    index = indices.to(torch.int64)
    index = torch.where(index < 0, index + count, index)
    axis = axis % (index.ndim + 1)
    positions = [1] * (index.ndim + 1)
    positions[axis] = count
    hot = index.unsqueeze(axis) == torch.arange(count, device=index.device).reshape(positions)
    if _rt.is_strings(values):
        return np.where(hot.cpu().numpy(), values[1], values[0]).astype(object)
    return torch.where(hot, values[1].to(hot.device), values[0].to(hot.device))


def eye_like(x, /, *, dtype=None, k=0):
    """EyeLike: ones on the k-th diagonal of a matrix shaped like `x`, of `dtype` or `x`'s."""
    rows, cols = x.shape
    target = x.dtype if dtype is None else _rt.torch_dtype(dtype)
    device = x.device if isinstance(x, torch.Tensor) else _rt.device()
    offsets = torch.arange(cols, device=device).unsqueeze(0) - torch.arange(rows, device=device).unsqueeze(1)
    return (offsets == k).to(target)


def reverse_sequence(x, sequence_lens, /, *, batch_axis=1, time_axis=0):
    """ReverseSequence: the first `sequence_lens[b]` steps of each batch entry reversed along the
    time axis, the rest kept."""
    steps = x.shape[time_axis]
    lengths = sequence_lens.to(torch.int64).to(_device_of(x)).reshape(1, -1)
    time = torch.arange(steps, device=lengths.device).reshape(-1, 1)
    source = torch.where(time < lengths, lengths - 1 - time, time)
    if time_axis == 1:
        source = source.transpose(0, 1)
    source = source.reshape(list(source.shape) + [1] * (x.ndim - 2)).expand(x.shape)
    if _rt.is_strings(x):
        return np.take_along_axis(x, source.cpu().numpy(), time_axis)
    return torch.gather(x, time_axis, source.contiguous())


def _device_of(value):
    return value.device if isinstance(value, torch.Tensor) else _rt.device()


def tensor_scatter(past_cache, update, write_indices=None, /, *, mode="linear", axis=-2):
    """TensorScatter: `update` written into each batch entry's cache along `axis`, starting at its
    write index (0 without one), wrapping around the cache in circular mode."""
    axis = axis % past_cache.ndim
    length, window = past_cache.shape[axis], update.shape[axis]
    batch = past_cache.shape[0]
    device = _device_of(past_cache)
    if write_indices is None:
        starts = torch.zeros(batch, dtype=torch.int64, device=device)
    else:
        starts = write_indices.to(torch.int64).to(device).reshape(-1)
    offsets = torch.arange(length, device=device).unsqueeze(0) - starts.unsqueeze(1)
    if mode == "circular":
        offsets = torch.remainder(offsets, length)
    written = (offsets >= 0) & (offsets < window)
    shape = [1] * past_cache.ndim
    shape[0], shape[axis] = batch, length
    written = written.reshape(shape).expand(past_cache.shape)
    positions = offsets.clamp(0, max(window - 1, 0)).reshape(shape).expand(past_cache.shape)
    if window == 0:
        return past_cache.clone()
    return torch.where(written, torch.gather(update, axis, positions.contiguous()), past_cache)
