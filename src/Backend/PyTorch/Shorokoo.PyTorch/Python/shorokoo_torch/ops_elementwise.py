"""ONNX elementwise math and activations, over torch tensors.

Every helper is functional -- it never writes into a tensor it is handed -- and takes the node's
inputs positionally and its attributes as keywords named as ONNX names them, with ONNX's defaults.
"""

import functools
import math

import numpy as np
import torch
import torch.nn.functional as F

from . import runtime as _rt

_NARROW_UNSIGNED = {torch.uint16: 0xFFFF, torch.uint32: 0xFFFFFFFF}


def _emulated(fn, *xs):
    """`fn` over xs, for the unsigned dtypes torch has few kernels for: 16- and 32-bit ones are
    computed in int64 and wrapped back, and 64-bit ones as int64 bit patterns, which is exact for
    the ring operations (add, sub, mul, bitwise) it is used for."""
    dtype = xs[0].dtype
    if dtype in _NARROW_UNSIGNED:
        return (fn(*[x.to(torch.int64) for x in xs]) & _NARROW_UNSIGNED[dtype]).to(dtype)
    if dtype == torch.uint64:
        return fn(*[x.view(torch.int64) for x in xs]).view(torch.uint64)
    return fn(*xs)


def _widened(fn, *xs):
    """`fn` over xs with narrow unsigned dtypes widened to int64 and the result cast back."""
    dtype = xs[0].dtype
    if dtype in _NARROW_UNSIGNED:
        return fn(*[x.to(torch.int64) for x in xs]).to(dtype)
    return fn(*xs)


def _is_integral(tensor):
    return not (tensor.dtype.is_floating_point or tensor.dtype.is_complex)


# ---- arithmetic ------------------------------------------------------------------------------

def add(a, b):
    return _emulated(torch.add, a, b)


def sub(a, b):
    return _emulated(torch.sub, a, b)


def mul(a, b):
    return _emulated(torch.mul, a, b)


def div(a, b):
    if _is_integral(a):
        return _widened(lambda x, y: torch.div(x, y, rounding_mode="trunc"), a, b)
    return torch.div(a, b)


def mod(a, b, *, fmod=0):
    if fmod:
        return _widened(torch.fmod, a, b)
    return _widened(torch.remainder, a, b)


def pow_(x, y):
    result = torch.pow(x, y) if x.dtype == y.dtype or not _is_integral(x) else torch.pow(x.to(torch.float64), y)
    return result.to(x.dtype)


def neg(x):
    return torch.neg(x)


def abs_(x):
    if x.dtype in (torch.uint8, torch.uint16, torch.uint32, torch.uint64):
        return x.clone()
    return torch.abs(x)


def sign(x):
    return _widened(torch.sign, x)


def reciprocal(x):
    return torch.reciprocal(x)


def floor(x):
    return torch.floor(x)


def ceil(x):
    return torch.ceil(x)


def round_(x):
    # Half to even, as ONNX specifies and torch.round does.
    return torch.round(x)


def sqrt(x):
    return torch.sqrt(x)


def exp(x):
    return torch.exp(x)


def log(x):
    return torch.log(x)


def sin(x):
    return torch.sin(x)


def cos(x):
    return torch.cos(x)


def tan(x):
    return torch.tan(x)


def asin(x):
    return torch.asin(x)


def acos(x):
    return torch.acos(x)


def atan(x):
    return torch.atan(x)


def sinh(x):
    return torch.sinh(x)


def cosh(x):
    return torch.cosh(x)


def tanh(x):
    return torch.tanh(x)


def asinh(x):
    return torch.asinh(x)


def acosh(x):
    return torch.acosh(x)


def atanh(x):
    return torch.atanh(x)


def erf(x):
    return torch.erf(x)


def max_(*xs):
    return functools.reduce(lambda a, b: _widened(torch.maximum, a, b), xs)


def min_(*xs):
    return functools.reduce(lambda a, b: _widened(torch.minimum, a, b), xs)


def sum_(*xs):
    return functools.reduce(add, xs)


def mean(*xs):
    return div(sum_(*xs), torch.tensor(len(xs), dtype=xs[0].dtype, device=xs[0].device))


def clip(x, lo=None, hi=None, /, *, min=None, max=None):
    if lo is None and min is not None:
        lo = torch.tensor(min, dtype=x.dtype, device=x.device)
    if hi is None and max is not None:
        hi = torch.tensor(max, dtype=x.dtype, device=x.device)
    work = x.to(torch.int64) if x.dtype in _NARROW_UNSIGNED else x
    result = work.clone()
    if lo is not None:
        result = torch.maximum(result, lo.to(work.dtype))
    if hi is not None:
        # After the lower bound, so that a lower bound above the upper one yields the upper one,
        # as ONNX specifies.
        result = torch.minimum(result, hi.to(work.dtype))
    return result.to(x.dtype)


def cumsum(x, axis, /, *, exclusive=0, reverse=0):
    return _cumulative(torch.cumsum, 0, x, axis, exclusive, reverse)


def cumprod(x, axis, /, *, exclusive=0, reverse=0):
    return _cumulative(torch.cumprod, 1, x, axis, exclusive, reverse)


def _cumulative(fn, identity, x, axis, exclusive, reverse):
    dim = int(axis.reshape(-1)[0])
    source = torch.flip(x, [dim]) if reverse else x
    result = fn(source, dim).to(x.dtype)
    if exclusive:
        pad_shape = list(result.shape)
        pad_shape[dim] = 1
        head = torch.full(pad_shape, identity, dtype=x.dtype, device=x.device)
        result = torch.cat([head, result.narrow(dim, 0, result.shape[dim] - 1)], dim) if result.shape[dim] else result
    return torch.flip(result, [dim]) if reverse else result


# ---- activations -----------------------------------------------------------------------------

def sigmoid(x):
    return torch.sigmoid(x)


def relu(x):
    return torch.relu(x)


def leaky_relu(x, *, alpha=0.01):
    return torch.where(x >= 0, x, x * alpha)


# The branch torch.where does not select still has its gradient taken, multiplied by zero: an
# exponential there must see only the values it is selected for, or an overflow to inf in it makes
# the gradient 0 * inf = NaN. Hence the clamps inside expm1.

def elu(x, *, alpha=1.0):
    return torch.where(x > 0, x, alpha * torch.expm1(torch.clamp(x, max=0)))


def selu(x, *, alpha=1.67326319217681884765625, gamma=1.05070102214813232421875):
    return gamma * torch.where(x > 0, x, alpha * torch.expm1(torch.clamp(x, max=0)))


def celu(x, *, alpha=1.0):
    return torch.where(x > 0, x, alpha * torch.expm1(torch.clamp(x, max=0) / alpha))


def thresholded_relu(x, *, alpha=1.0):
    return torch.where(x > alpha, x, torch.zeros_like(x))


def hard_sigmoid(x, *, alpha=0.2, beta=0.5):
    return torch.clamp(alpha * x + beta, 0, 1)


def hard_swish(x):
    return x * hard_sigmoid(x, alpha=1.0 / 6, beta=0.5)


def softplus(x):
    # log(e^x + e^0): max(x, 0) + log1p(exp(-|x|)), without overflow, and with a gradient (the
    # sigmoid) that stays finite everywhere.
    return torch.logaddexp(x, torch.zeros_like(x))


def softsign(x):
    return x / (1 + torch.abs(x))


def mish(x):
    return x * torch.tanh(softplus(x))


def gelu(x, *, approximate="none"):
    return F.gelu(x, approximate=approximate)


def swish(x, *, alpha=1.0):
    return x * torch.sigmoid(alpha * x)


def shrink(x, *, lambd=0.5, bias=0.0):
    return torch.where(x < -lambd, x + bias, torch.where(x > lambd, x - bias, torch.zeros_like(x))).to(x.dtype)


def prelu(x, slope):
    return torch.where(x < 0, x * slope, x)


def _flattened(fn, x, axis, opset):
    """Softmax-like `fn` over `axis`: before opset 13 over the input coerced to 2D at `axis`
    (default 1), from 13 over `axis` alone (default -1)."""
    if opset >= 13:
        return fn(x, -1 if axis is None else axis)
    axis = 1 if axis is None else axis
    if x.dim() == 0:
        return fn(x.reshape(1, 1), 1).reshape(())
    axis = axis % x.dim()
    rows = math.prod(x.shape[:axis])
    return fn(x.reshape(rows, -1), 1).reshape(x.shape)


def softmax(x, *, axis=None, _opset):
    return _flattened(lambda v, d: torch.softmax(v, d), x, axis, _opset)


def log_softmax(x, *, axis=None, _opset):
    return _flattened(lambda v, d: torch.log_softmax(v, d), x, axis, _opset)


def hardmax(x, *, axis=None, _opset):
    def one_hot_of_first_max(v, d):
        index = torch.argmax(v, d, keepdim=True)
        return torch.zeros_like(v).scatter(d, index, torch.ones_like(index, dtype=v.dtype))
    return _flattened(one_hot_of_first_max, x, axis, _opset)


# ---- casts -----------------------------------------------------------------------------------

_FLOAT8 = (torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2, torch.float8_e5m2fnuz)


def _format_number(value):
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "INF" if value > 0 else "-INF"
        return np.format_float_positional(np.float32(value), trim="-") if abs(value) < 1e16 else repr(value)
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def cast(x, *, to, saturate=1, round_mode="up"):
    to = int(to)
    if to == _rt.STRING:
        if _rt.is_strings(x):
            return x.copy()
        values = x.detach().cpu().reshape(-1).tolist()
        return _rt.strings([_format_number(v) for v in values], list(x.shape))
    target = _rt.torch_dtype(to)
    if _rt.is_strings(x):
        parsed = [float(s) if target.is_floating_point else int(float(s)) for s in x.reshape(-1).tolist()]
        return torch.tensor(parsed, device=_rt.device()).to(target).reshape(x.shape)
    if target in _FLOAT8 and saturate and x.dtype.is_floating_point:
        limit = torch.finfo(target).max
        bounded = torch.clamp(x, -limit, limit)
        if target in (torch.float8_e5m2,):
            bounded = torch.where(torch.isinf(x), x, bounded)
        return bounded.to(target)
    if target == torch.bool:
        return x != 0
    if x.dtype in (torch.uint32, torch.uint64) or target in (torch.uint16, torch.uint32):
        return x.to(torch.int64).to(target) if x.dtype != torch.uint64 else x.to(target)
    return x.to(target)


def cast_like(x, like, *, saturate=1, round_mode="up"):
    return cast(x, to=_rt.dtype_code(like), saturate=saturate, round_mode=round_mode)
