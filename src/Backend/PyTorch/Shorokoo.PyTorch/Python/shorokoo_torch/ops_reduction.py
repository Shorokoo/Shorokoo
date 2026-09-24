"""ONNX reductions: the Reduce* family, ArgMax and ArgMin.

Axes arrive as an input from opset 13 (ReduceSum) or 18 (the others) and as an attribute before
that; both are accepted and the input wins. An empty axes list reduces everything unless
noop_with_empty_axes is set, in which case the input comes back unchanged. Every reduction keeps
the input's element type, as ONNX does and torch's own integer sums do not.
"""

import math

import torch

from .ops_elementwise import _NARROW_UNSIGNED


def _dims(data, axes_input, axes, noop_with_empty_axes):
    """The dimensions to reduce, or None for the identity."""
    if axes_input is not None:
        axes = [int(a) for a in axes_input.reshape(-1).tolist()]
    if not axes:
        return None if noop_with_empty_axes else list(range(data.dim()))
    rank = data.dim()
    return sorted({a % rank for a in axes}) if rank else []


def _filled(data, dims, keepdims, value):
    shape = [1 if d in dims else n for d, n in enumerate(data.shape)] if keepdims \
        else [n for d, n in enumerate(data.shape) if d not in dims]
    return torch.full(shape, value, dtype=data.dtype, device=data.device)


def _reduce(fn, empty_value, data, axes_input, axes, keepdims, noop_with_empty_axes):
    dims = _dims(data, axes_input, axes, noop_with_empty_axes)
    if dims is None:
        return data.clone()
    if not dims:
        # A scalar reduced over no axes: reducing a leading axis of one gives each reduction its
        # own answer for a single element (the element, its square, its absolute value...).
        return fn(data.unsqueeze(0), [0], False).to(data.dtype)
    if any(data.shape[d] == 0 for d in dims):
        value = empty_value(data.dtype) if callable(empty_value) else empty_value
        return _filled(data, dims, keepdims, value)
    work = data.to(torch.int64) if data.dtype in _NARROW_UNSIGNED else data
    return fn(work, dims, bool(keepdims)).to(data.dtype)


def _lowest(dtype):
    return -math.inf if dtype.is_floating_point else torch.iinfo(dtype).min


def _highest(dtype):
    return math.inf if dtype.is_floating_point else torch.iinfo(dtype).max


def _mean(x, dims, keep):
    if x.dtype.is_floating_point or x.dtype.is_complex:
        return torch.mean(x, dims, keepdim=keep)
    count = math.prod(x.shape[d] for d in dims)
    return torch.div(torch.sum(x, dims, keepdim=keep), count, rounding_mode="trunc")


def reduce_sum(data, axes_input=None, /, *, axes=None, keepdims=1, noop_with_empty_axes=0):
    return _reduce(lambda x, d, k: torch.sum(x, d, keepdim=k), 0, data, axes_input, axes, keepdims, noop_with_empty_axes)


def reduce_mean(data, axes_input=None, /, *, axes=None, keepdims=1, noop_with_empty_axes=0):
    return _reduce(_mean, math.nan, data, axes_input, axes, keepdims, noop_with_empty_axes)


def reduce_max(data, axes_input=None, /, *, axes=None, keepdims=1, noop_with_empty_axes=0):
    return _reduce(lambda x, d, k: torch.amax(x, d, keepdim=k), _lowest, data, axes_input, axes, keepdims, noop_with_empty_axes)


def reduce_min(data, axes_input=None, /, *, axes=None, keepdims=1, noop_with_empty_axes=0):
    return _reduce(lambda x, d, k: torch.amin(x, d, keepdim=k), _highest, data, axes_input, axes, keepdims, noop_with_empty_axes)


def reduce_prod(data, axes_input=None, /, *, axes=None, keepdims=1, noop_with_empty_axes=0):
    def prod(x, dims, keep):
        for d in sorted(dims, reverse=True):
            x = torch.prod(x, d, keepdim=keep)
        return x
    return _reduce(prod, 1, data, axes_input, axes, keepdims, noop_with_empty_axes)


def reduce_l1(data, axes_input=None, /, *, axes=None, keepdims=1, noop_with_empty_axes=0):
    return _reduce(lambda x, d, k: torch.sum(torch.abs(x), d, keepdim=k), 0, data, axes_input, axes, keepdims, noop_with_empty_axes)


def reduce_l2(data, axes_input=None, /, *, axes=None, keepdims=1, noop_with_empty_axes=0):
    def l2(x, dims, keep):
        if not (x.dtype.is_floating_point or x.dtype.is_complex):
            return torch.sqrt(torch.sum(x * x, dims, keepdim=keep).to(torch.float64))
        return torch.sqrt(torch.sum(x * x, dims, keepdim=keep))
    return _reduce(l2, 0, data, axes_input, axes, keepdims, noop_with_empty_axes)


def reduce_sum_square(data, axes_input=None, /, *, axes=None, keepdims=1, noop_with_empty_axes=0):
    return _reduce(lambda x, d, k: torch.sum(x * x, d, keepdim=k), 0, data, axes_input, axes, keepdims, noop_with_empty_axes)


def reduce_log_sum(data, axes_input=None, /, *, axes=None, keepdims=1, noop_with_empty_axes=0):
    return _reduce(lambda x, d, k: torch.log(torch.sum(x, d, keepdim=k)), -math.inf, data, axes_input, axes, keepdims, noop_with_empty_axes)


def reduce_log_sum_exp(data, axes_input=None, /, *, axes=None, keepdims=1, noop_with_empty_axes=0):
    return _reduce(lambda x, d, k: torch.logsumexp(x, d, keepdim=k), -math.inf, data, axes_input, axes, keepdims, noop_with_empty_axes)


def _arg(fn, data, axis, keepdims, select_last_index):
    work = data.to(torch.int64) if data.dtype in _NARROW_UNSIGNED else data
    if work.dim() == 0:
        work = work.reshape(1)
    axis = axis % work.dim()
    if select_last_index:
        flipped = torch.flip(work, [axis])
        index = work.shape[axis] - 1 - fn(flipped, axis, keepdim=bool(keepdims))
    else:
        index = fn(work, axis, keepdim=bool(keepdims))
    return index.to(torch.int64)


def arg_max(data, *, axis=0, keepdims=1, select_last_index=0):
    return _arg(torch.argmax, data, axis, keepdims, select_last_index)


def arg_min(data, *, axis=0, keepdims=1, select_last_index=0):
    return _arg(torch.argmin, data, axis, keepdims, select_last_index)
