"""ONNX shape and data-movement operators that rearrange or make tensors without indexing into
them: Reshape, Transpose, Concat, Split, Squeeze/Unsqueeze, Shape, Size, Flatten, Expand, Tile,
Identity, ConstantOfShape, Range, Trilu.

A result may be a view of an input; the run hands every output over in memory of its own, so a
view never escapes to the caller.
"""

import math

import numpy as np
import torch

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
