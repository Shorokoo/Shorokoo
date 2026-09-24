"""ONNX operators that index into a tensor: Gather, GatherElements, GatherND, Slice, Compress,
ScatterElements, ScatterND, TopK, Unique and NonZero.

Indices may be negative, counting from the end of their axis, as ONNX allows. Where torch has no
kernel for an element type -- the unsigned ones beyond uint8, mostly -- the values are moved in a
type it has one for and moved back, which is exact: a scatter or a gather only copies them, and a
reduction is done where it keeps its meaning (see `_workable`).
"""

import math

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


def compress(data, condition, /, *, axis=None):
    """The slices along `axis` (of the flattened input where there is none) whose condition is
    true. A condition shorter than the axis selects from the leading slices only."""
    if axis is None:
        data = data.reshape(-1)
        axis = 0
    axis = axis % data.ndim
    keep = condition.reshape(-1)[: data.shape[axis]].to(torch.bool)
    index = torch.nonzero(keep).reshape(-1).to(data.device)
    return torch.index_select(data, axis, index)


# ---- the unsigned types torch has few kernels for ------------------------------------------

_UNSIGNED = (torch.uint16, torch.uint32, torch.uint64)
_SIGN_BIT = -(2 ** 63)


def _workable(values, ordered=False):
    """(values in a dtype torch's indexing kernels take, the function that brings a result back).

    uint16 and uint32 widen to int64, which holds them exactly. uint64 is carried as its int64 bit
    pattern, which is exact for copying and for the ring operations (add, mul); where the result
    depends on the order of the values (`ordered`: min, max, sort), the sign bit is flipped too, so
    that int64 order is the unsigned order."""
    dtype = values.dtype
    if dtype in (torch.uint16, torch.uint32):
        return values.to(torch.int64), lambda r: r.to(dtype)
    if dtype == torch.uint64:
        bits = values.view(torch.int64)
        if not ordered:
            return bits, lambda r: r.view(torch.uint64)
        return bits ^ _SIGN_BIT, lambda r: (r ^ _SIGN_BIT).view(torch.uint64)
    if dtype == torch.bool:
        return values.to(torch.uint8), lambda r: r.to(torch.bool)
    return values, lambda r: r


# ---- gathers -------------------------------------------------------------------------------

def gather_nd(data, indices, /, *, batch_dims=0):
    """GatherND: each index tuple along `indices`' last axis picks a slice of `data`, below its
    first `batch_dims` axes, which `data` and `indices` share."""
    b = batch_dims
    k = indices.shape[-1]
    batch = list(data.shape[:b])
    batch_count = math.prod(batch)
    picked = list(data.shape[b:b + k])
    rest = list(data.shape[b + k:])
    index = indices.to(torch.int64).to(data.device)
    index = torch.where(index < 0, index + torch.tensor(picked, dtype=torch.int64, device=data.device), index)
    strides = [math.prod(picked[i + 1:]) for i in range(k)]
    offsets = (index * torch.tensor(strides, dtype=torch.int64, device=data.device)).sum(-1)
    lookups = list(indices.shape[b:-1])
    offsets = offsets.reshape(batch_count, -1)
    offsets = offsets + torch.arange(batch_count, device=data.device).unsqueeze(1) * math.prod(picked)
    if _rt.is_strings(data):
        flat = data.reshape([batch_count * math.prod(picked)] + rest)
        return flat[offsets.reshape(-1).cpu().numpy()].reshape(batch + lookups + rest)
    flat = data.reshape([batch_count * math.prod(picked)] + rest)
    return torch.index_select(flat, 0, offsets.reshape(-1)).reshape(batch + lookups + rest)


# ---- scatters ------------------------------------------------------------------------------

_REDUCTIONS = {"add": "sum", "mul": "prod", "max": "amax", "min": "amin"}


def scatter_elements(data, indices, updates, /, *, axis=0, reduction="none"):
    axis = axis % data.ndim
    index = _normalized(indices, data.shape[axis]).to(data.device)
    if _rt.is_strings(data):
        result = data.copy()
        np.put_along_axis(result, index.cpu().numpy(), updates, axis)
        return result
    work, back = _workable(data, ordered=reduction in ("max", "min"))
    source, _ = _workable(updates, ordered=reduction in ("max", "min"))
    if reduction == "none":
        return back(work.scatter(axis, index, source))
    return back(work.scatter_reduce(axis, index, source, _REDUCTIONS[reduction], include_self=True))


def scatter_nd(data, indices, updates, /, *, reduction="none"):
    """ScatterND: each index tuple along `indices`' last axis names a slice of `data` that the
    matching slice of `updates` replaces or, with a reduction, is combined into -- in index order,
    so repeated indices accumulate."""
    k = indices.shape[-1]
    picked = list(data.shape[:k])
    rest = list(data.shape[k:])
    index = indices.to(torch.int64).to(data.device).reshape(-1, k)
    index = torch.where(index < 0, index + torch.tensor(picked, dtype=torch.int64, device=data.device), index)
    strides = [math.prod(picked[i + 1:]) for i in range(k)]
    offsets = (index * torch.tensor(strides, dtype=torch.int64, device=data.device)).sum(-1)
    if _rt.is_strings(data):
        flat = data.reshape([math.prod(picked)] + rest).copy()
        flat[offsets.cpu().numpy()] = updates.reshape([offsets.shape[0]] + rest)
        return flat.reshape(data.shape)
    work, back = _workable(data, ordered=reduction in ("max", "min"))
    source, _ = _workable(updates, ordered=reduction in ("max", "min"))
    flat = work.reshape([math.prod(picked)] + rest)
    source = source.reshape([offsets.shape[0]] + rest)
    if reduction == "none":
        result = flat.index_put((offsets,), source)
    else:
        spread = offsets.reshape([-1] + [1] * len(rest)).expand(source.shape)
        result = flat.scatter_reduce(0, spread, source, _REDUCTIONS[reduction], include_self=True)
    return back(result).reshape(data.shape)


# ---- selections ----------------------------------------------------------------------------

def top_k(x, k, /, *, axis=-1, largest=1, sorted=1):
    """TopK: the k largest (smallest) elements along `axis` and their indices, in order, equal
    values in the order of their indices, as the spec requires. The order is kept even where
    `sorted` is 0, since any order is allowed then. The values are gathered from `x` by the
    indices, so they carry its gradient."""
    count = int(k.reshape(-1)[0].item())
    axis = axis % x.ndim
    work, back = _workable(x, ordered=True)
    order = torch.sort(work, dim=axis, descending=bool(largest), stable=True).indices
    indices = order.narrow(axis, 0, count)
    if x.dtype in _UNSIGNED:
        return back(torch.gather(work, axis, indices)), indices
    return torch.gather(x, axis, indices), indices


def _host_numpy(x):
    """`x` as a numpy array on the host that orders and compares its values as `x` does."""
    if _rt.is_strings(x):
        return x
    if x.dtype in (torch.bfloat16,) or x.dtype in _FLOAT8:
        return x.detach().cpu().to(torch.float32).numpy()
    return x.detach().cpu().numpy()


_FLOAT8 = (torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2, torch.float8_e5m2fnuz)


def unique(x, /, *, axis=None, sorted=1):
    """Unique: the distinct elements (or slices along `axis`), ascending or in order of first
    occurrence, with the index of each one's first occurrence, the inverse map, and the counts.
    The distinct values are gathered from `x` by those first indices, so they carry its
    gradient."""
    host = _host_numpy(x)
    if axis is None:
        host = host.reshape(-1)
        dim = 0
    else:
        dim = axis % x.ndim
    if _rt.is_strings(x):
        flat = host if axis is None else None
        if flat is None:
            raise NotImplementedError("Unique with an axis over a string tensor")
        keys = flat.astype(str)
        _, first, inverse, counts = np.unique(keys, return_index=True, return_inverse=True, return_counts=True)
    elif axis is None:
        _, first, inverse, counts = np.unique(host, return_index=True, return_inverse=True, return_counts=True)
    else:
        _, first, inverse, counts = np.unique(host, axis=dim, return_index=True, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    if not sorted:
        order = np.argsort(first, kind="stable")
        rank = np.empty_like(order)
        rank[order] = np.arange(len(order))
        first, counts, inverse = first[order], counts[order], rank[inverse]
    device = _rt.device() if _rt.is_strings(x) else x.device

    def int64s(values):
        return torch.from_numpy(np.ascontiguousarray(values, dtype=np.int64)).to(device)

    if _rt.is_strings(x):
        y = host[first]
    else:
        source = x.reshape(-1) if axis is None else x
        y = torch.index_select(source, dim, int64s(first))
    return y, int64s(first), int64s(inverse), int64s(counts)


def non_zero(x):
    """NonZero: the index of every non-zero element, one row per axis, in row-major order."""
    if _rt.is_strings(x):
        found = np.array(np.nonzero(x != ""), dtype=np.int64).reshape(x.ndim, -1)
        return torch.from_numpy(found).to(_rt.device())
    mask = x != 0 if x.dtype != torch.bool else x
    if x.ndim == 0:
        return torch.zeros((0, int(mask.item())), dtype=torch.int64, device=x.device)
    return torch.nonzero(mask).transpose(0, 1).contiguous()
