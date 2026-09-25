"""ONNX elementwise math and activations, over torch tensors.

Every helper is functional -- it never writes into a tensor it is handed -- and takes the node's
inputs positionally and its attributes as keywords named as ONNX names them, with ONNX's defaults.
"""

import functools
import math
import re

import numpy as np
import torch
import torch.nn.functional as F

from . import runtime as _rt

_NARROW_UNSIGNED = {torch.uint16: 0xFFFF, torch.uint32: 0xFFFFFFFF}
_SIGN_BIT = -(2 ** 63)


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


def _unsigned_at_least(x, y):
    """x >= y for int64 tensors holding uint64 bit patterns."""
    return (x ^ _SIGN_BIT) >= (y ^ _SIGN_BIT)


def _divmod_uint64(a, b):
    """Quotient and remainder of uint64 tensors, which torch has no division kernel for, exactly,
    in int64 arithmetic on their bit patterns: a divisor of 2**63 or more goes at most once; a
    smaller one divides half the dividend -- which is below 2**63 -- and the quotient doubled is
    corrected by the one step the remainder can still hold."""
    x, y = torch.broadcast_tensors(a.view(torch.int64), b.view(torch.int64))
    large = y < 0
    divisor = torch.where(large, torch.ones_like(y), y)
    quotient = torch.div((x >> 1) & 0x7FFFFFFFFFFFFFFF, divisor, rounding_mode="floor") << 1
    quotient = quotient + _unsigned_at_least(x - quotient * divisor, divisor).to(torch.int64)
    quotient = torch.where(large, _unsigned_at_least(x, y).to(torch.int64), quotient)
    return quotient.view(torch.uint64), (x - quotient * y).view(torch.uint64)


def div(a, b):
    if a.dtype == torch.uint64:
        return _divmod_uint64(a, b)[0]
    if _is_integral(a):
        return _widened(lambda x, y: torch.div(x, y, rounding_mode="trunc"), a, b)
    return torch.div(a, b)


def mod(a, b, *, fmod=0):
    if a.dtype == torch.uint64:
        return _divmod_uint64(a, b)[1]
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
    if x.dtype == torch.uint64:
        return (x.view(torch.int64) != 0).to(torch.int64).view(torch.uint64)
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


def _ordered(fn, a, b):
    """`fn` (maximum or minimum) of a and b; uint64 ones, which torch cannot order, as int64 bit
    patterns with the sign bit flipped, which int64 orders as unsigned."""
    if a.dtype == torch.uint64:
        return (fn(a.view(torch.int64) ^ _SIGN_BIT, b.view(torch.int64) ^ _SIGN_BIT) ^ _SIGN_BIT).view(torch.uint64)
    return _widened(fn, a, b)


def _extreme(fn, pairwise, xs):
    """Max or Min of xs. A floating-point one is `fn` (amax or amin) over the inputs stacked, whose
    gradient is shared equally among every input that ties for the result, as Shorokoo's own rule
    shares it; torch's pairwise maximum and minimum would halve it at each tie instead."""
    if len(xs) > 2 and xs[0].dtype.is_floating_point:
        return fn(torch.stack(torch.broadcast_tensors(*xs)), 0)
    return functools.reduce(lambda a, b: _ordered(pairwise, a, b), xs)


def max_(*xs):
    return _extreme(torch.amax, torch.maximum, xs)


def min_(*xs):
    return _extreme(torch.amin, torch.minimum, xs)


def sum_(*xs):
    return functools.reduce(add, xs)


def mean(*xs):
    return div(sum_(*xs), torch.tensor(len(xs), dtype=xs[0].dtype, device=xs[0].device))


def clip(x, lo=None, hi=None, /, *, min=None, max=None):
    """Clip, whose gradient is Shorokoo's rule: 1 where x lies within the bounds, the bounds
    themselves included, 0 elsewhere, and none to the bounds."""
    if lo is None and min is not None:
        lo = torch.tensor(min, dtype=x.dtype, device=x.device)
    if hi is None and max is not None:
        hi = torch.tensor(max, dtype=x.dtype, device=x.device)
    if x.dtype == torch.uint64:
        # As int64 bit patterns with the sign bit flipped, which int64 orders as unsigned.
        work = x.view(torch.int64) ^ _SIGN_BIT
        lo = None if lo is None else lo.view(torch.int64) ^ _SIGN_BIT
        hi = None if hi is None else hi.view(torch.int64) ^ _SIGN_BIT
        return (clip(work, lo, hi) ^ _SIGN_BIT).view(torch.uint64)
    work = x.to(torch.int64) if x.dtype in _NARROW_UNSIGNED else x
    inside = torch.ones_like(work, dtype=torch.bool)
    result = work.detach()
    if lo is not None:
        lo = lo.detach().to(work.dtype)
        inside = inside & ~(work < lo)
        result = torch.maximum(result, lo)
    if hi is not None:
        # After the lower bound, so that a lower bound above the upper one yields the upper one,
        # as ONNX specifies.
        hi = hi.detach().to(work.dtype)
        inside = inside & ~(work > hi)
        result = torch.minimum(result, hi)
    return torch.where(inside, work, result).to(x.dtype)


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


# Where an activation's pieces meet, the gradient is that of the piece Shorokoo's own rule takes:
# the negative one at 0 for LeakyRelu and PRelu, and 0 where HardSigmoid meets its bounds.

def leaky_relu(x, *, alpha=0.01):
    return torch.where(x > 0, x, x * alpha)


# The branch torch.where does not select still has its gradient taken, multiplied by zero: an
# exponential there must see only the values it is selected for, or an overflow to inf in it makes
# the gradient 0 * inf = NaN. Hence the clamp inside expm1 -- a `where`, since torch.clamp passes no
# gradient at its bound, and at x = 0 the negative piece's gradient is Shorokoo's rule.

def _nonpositive(x):
    return torch.where(x > 0, torch.zeros_like(x), x)


def elu(x, *, alpha=1.0):
    return torch.where(x > 0, x, alpha * torch.expm1(_nonpositive(x)))


def selu(x, *, alpha=1.67326319217681884765625, gamma=1.05070102214813232421875):
    return gamma * torch.where(x > 0, x, alpha * torch.expm1(_nonpositive(x)))


def celu(x, *, alpha=1.0):
    return torch.where(x > 0, x, alpha * torch.expm1(_nonpositive(x) / alpha))


def thresholded_relu(x, *, alpha=1.0):
    return torch.where(x > alpha, x, torch.zeros_like(x))


def hard_sigmoid(x, *, alpha=0.2, beta=0.5):
    t = alpha * x + beta
    return torch.where((t > 0) & (t < 1), t, torch.clamp(t, 0, 1).detach())


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
    if x.dtype in _NARROW_UNSIGNED or x.dtype == torch.uint64:
        # Never negative, and torch has no comparison for these types.
        return x.clone()
    return torch.where(x > 0, x, x * slope)


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
    """A number as ONNX Runtime writes it into a string: an integer in decimal, a bool as 1 or 0,
    a float as printf's %.8g (which turns to exponent notation outside 1e-5..1e8), and the
    non-finite ones as NaN, INF and -INF."""
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "INF" if value > 0 else "-INF"
        return "%.8g" % value
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


_LEADING_INTEGER = re.compile(r"\s*[+-]?\d+")


def _parse_number(text, target):
    """A string read as a number of dtype `target`: a float in plain or exponent notation (INF,
    -INF and NaN in any case), and an integer from its leading digits, as C's strtoll does."""
    if target.is_floating_point:
        return float(text)
    if target == torch.bool:
        return bool(_parse_number(text, torch.int64))
    match = _LEADING_INTEGER.match(text)
    if match is None:
        raise ValueError(f"Cast: '{text}' is not an integer")
    return int(match.group(0))


def cast(x, *, to, saturate=1, round_mode="up"):
    to = int(to)
    if to == _rt.STRING:
        if _rt.is_strings(x):
            return x.copy()
        values = x.detach().cpu().reshape(-1).tolist()
        return _rt.strings([_format_number(v) for v in values], list(x.shape))
    target = _rt.torch_dtype(to)
    if _rt.is_strings(x):
        parsed = [_parse_number(s, target) for s in x.reshape(-1).tolist()]
        if target.is_floating_point:
            return torch.tensor(parsed, dtype=torch.float64).to(target).reshape(x.shape).to(_rt.device())
        wide = torch.tensor([v & 0xFFFFFFFFFFFFFFFF for v in parsed], dtype=torch.uint64)
        return cast(wide.reshape(x.shape), to=to).to(_rt.device())
    if target in _FLOAT8 and saturate and x.dtype.is_floating_point:
        # Beyond the largest finite value, infinities included, saturates to it.
        limit = torch.finfo(target).max
        return torch.clamp(x, -limit, limit).to(target)
    if target == torch.float8_e4m3fn and x.dtype.is_floating_point:
        # Unsaturated, a value that rounds past 448 -- beyond 464, halfway to the next step -- has
        # no encoding and is NaN; torch saturates it.
        return torch.where(torch.abs(x) > 464, torch.full_like(x, math.nan), x).to(target)
    if target == torch.bool:
        return x != 0
    if x.dtype in (torch.uint32, torch.uint64) or target in (torch.uint16, torch.uint32):
        return x.to(torch.int64).to(target) if x.dtype != torch.uint64 else x.to(target)
    return x.to(target)


def cast_like(x, like, *, saturate=1, round_mode="up"):
    return cast(x, to=_rt.dtype_code(like), saturate=saturate, round_mode=round_mode)
