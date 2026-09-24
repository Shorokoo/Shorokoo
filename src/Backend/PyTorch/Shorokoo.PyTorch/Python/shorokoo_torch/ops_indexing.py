"""ONNX operators that index into a tensor: Gather, GatherElements, Slice.

Indices may be negative, counting from the end of their axis, as ONNX allows.
"""

import numpy as np
import torch

from . import runtime as _rt


def _normalized(indices, length):
    return torch.where(indices < 0, indices + length, indices).to(torch.int64)


def gather(data, indices, /, *, axis=0):
    rank = data.ndim
    axis = axis % rank
    if _rt.is_strings(data):
        index = indices.cpu().numpy()
        index = np.where(index < 0, index + data.shape[axis], index)
        return np.take(data, index, axis=axis)
    flat = _normalized(indices, data.shape[axis]).reshape(-1)
    picked = torch.index_select(data, axis, flat.to(data.device))
    return picked.reshape(list(data.shape[:axis]) + list(indices.shape) + list(data.shape[axis + 1:]))


def gather_elements(data, indices, /, *, axis=0):
    axis = axis % data.ndim
    return torch.gather(data, axis, _normalized(indices, data.shape[axis]))


def _clamped(value, length, step):
    if value < 0:
        value += length
    if step > 0:
        return min(max(value, 0), length)
    return min(max(value, -1), length - 1)


def slice_(data, starts_in=None, ends_in=None, axes_in=None, steps_in=None, /, *, starts=None, ends=None, axes=None):
    """Slice, from opset 10 (inputs) or before it (attributes)."""
    starts = [int(v) for v in starts_in.reshape(-1).tolist()] if starts_in is not None else list(starts)
    ends = [int(v) for v in ends_in.reshape(-1).tolist()] if ends_in is not None else list(ends)
    if axes_in is not None:
        axes = [int(v) for v in axes_in.reshape(-1).tolist()]
    axes = list(range(len(starts))) if axes is None else list(axes)
    steps = [int(v) for v in steps_in.reshape(-1).tolist()] if steps_in is not None else [1] * len(starts)

    result = data
    for start, end, axis, step in zip(starts, ends, axes, steps):
        axis = axis % data.ndim
        length = data.shape[axis]
        first = _clamped(start, length, step)
        last = _clamped(end, length, step)
        if _rt.is_strings(result):
            index = [slice(None)] * result.ndim
            index[axis] = slice(first, last if last >= 0 else None, step)
            result = result[tuple(index)]
        elif step > 0:
            result = result.narrow(axis, first, max(last - first, 0))[
                (slice(None),) * axis + (slice(None, None, step),)]
        else:
            positions = torch.arange(first, last, step, device=result.device)
            result = torch.index_select(result, axis, positions)
    return result
