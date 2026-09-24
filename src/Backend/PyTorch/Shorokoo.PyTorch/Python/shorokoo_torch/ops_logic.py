"""ONNX comparisons, logic and bitwise operators, and Where."""

import numpy as np
import torch

from . import runtime as _rt
from .ops_elementwise import _NARROW_UNSIGNED, _emulated


def _comparable(a, b):
    """a and b as tensors torch can compare: narrow unsigned ones widened to int64, and uint64
    ones as int64 with the sign bit flipped, which orders them as unsigned."""
    if a.dtype in _NARROW_UNSIGNED:
        return a.to(torch.int64), b.to(torch.int64)
    if a.dtype == torch.uint64:
        flip = torch.tensor(-(2 ** 63), dtype=torch.int64, device=a.device)
        return a.view(torch.int64) ^ flip, b.view(torch.int64) ^ flip
    return a, b


def _compare(fn, npfn, a, b):
    if _rt.is_strings(a):
        return torch.from_numpy(np.asarray(npfn(a, b), dtype=bool)).to(_rt.device())
    return fn(*_comparable(a, b))


def equal(a, b):
    return _compare(torch.eq, np.equal, a, b)


def greater(a, b):
    return _compare(torch.gt, np.greater, a, b)


def greater_or_equal(a, b):
    return _compare(torch.ge, np.greater_equal, a, b)


def less(a, b):
    return _compare(torch.lt, np.less, a, b)


def less_or_equal(a, b):
    return _compare(torch.le, np.less_equal, a, b)


def and_(a, b):
    return torch.logical_and(a, b)


def or_(a, b):
    return torch.logical_or(a, b)


def xor(a, b):
    return torch.logical_xor(a, b)


def not_(x):
    return torch.logical_not(x)


def is_nan(x):
    return torch.isnan(x)


def is_inf(x, *, detect_negative=1, detect_positive=1):
    if detect_negative and detect_positive:
        return torch.isinf(x)
    if detect_negative:
        return torch.isneginf(x)
    if detect_positive:
        return torch.isposinf(x)
    return torch.zeros_like(x, dtype=torch.bool)


def where(condition, x, y):
    if _rt.is_strings(x):
        return np.where(condition.cpu().numpy(), x, y).astype(object)
    return torch.where(condition, x, y)


def bitwise_and(a, b):
    return _emulated(torch.bitwise_and, a, b)


def bitwise_or(a, b):
    return _emulated(torch.bitwise_or, a, b)


def bitwise_xor(a, b):
    return _emulated(torch.bitwise_xor, a, b)


def bitwise_not(x):
    return _emulated(torch.bitwise_not, x)


def bit_shift(x, y, *, direction):
    if direction == "LEFT":
        return _emulated(torch.bitwise_left_shift, x, y)
    if x.dtype == torch.uint64:
        # An arithmetic shift of the int64 bit pattern would drag the sign bit in; masking it off
        # makes the shift logical, which is what an unsigned one is.
        bits, count = x.view(torch.int64), y.view(torch.int64)
        one = torch.ones_like(count)
        mask = (one << (64 - count).clamp(max=63)) - 1
        return torch.where(count == 0, bits, (bits >> count) & mask).view(torch.uint64)
    return _emulated(torch.bitwise_right_shift, x, y)
