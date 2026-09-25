"""ONNX elementwise math and activations, over numpy and jax arrays.

Every helper takes the node's inputs positionally and its attributes as keywords named as ONNX
names them, with ONNX's defaults. A helper whose inputs are all concrete computes with numpy, so
that shape arithmetic stays concrete (`runtime.xp`); one numpy has no function for computes with
jax.numpy whatever its inputs.
"""

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np

from . import runtime as _rt


def _is_unsigned(x):
    return jnp.issubdtype(x.dtype, jnp.unsignedinteger)


# ---- arithmetic ------------------------------------------------------------------------------

def add(a, b):
    return _rt.xp(a, b).add(a, b)


def sub(a, b):
    return _rt.xp(a, b).subtract(a, b)


def mul(a, b):
    return _rt.xp(a, b).multiply(a, b)


def div(a, b):
    xp = _rt.xp(a, b)
    if not _rt.is_integral(a):
        return xp.divide(a, b)
    if _is_unsigned(a):
        return xp.floor_divide(a, b)
    # Integer division truncates toward zero; floor division rounds down, so a quotient of operands
    # of different signs that leaves a remainder is one too low.
    quotient = xp.floor_divide(a, b)
    behind = (a - quotient * b != 0) & ((a < 0) != (b < 0))
    return (quotient + behind.astype(quotient.dtype)).astype(a.dtype)


def mod(a, b, *, fmod=0):
    xp = _rt.xp(a, b)
    return xp.fmod(a, b) if fmod else xp.remainder(a, b)


def pow_(x, y):
    xp = _rt.xp(x, y)
    if _rt.is_integral(x) and not _rt.is_integral(y):
        return xp.power(x.astype(np.float64), y.astype(np.float64)).astype(x.dtype)
    if not _rt.is_integral(x):
        return xp.power(x, y.astype(x.dtype))
    return xp.power(x, y.astype(x.dtype)).astype(x.dtype)


def neg(x):
    return _rt.xp(x).negative(x)


def abs_(x):
    return x if _is_unsigned(x) else _rt.xp(x).abs(x)


def sign(x):
    return _rt.xp(x).sign(x).astype(x.dtype)


def reciprocal(x):
    return _rt.xp(x).reciprocal(x) if not _rt.is_integral(x) else div(_rt.xp(x).ones_like(x), x)


def floor(x):
    return _rt.xp(x).floor(x)


def ceil(x):
    return _rt.xp(x).ceil(x)


def round_(x):
    # Half to even, as ONNX specifies and numpy and jax.numpy do.
    return _rt.xp(x).round(x)


def sqrt(x):
    return _rt.xp(x).sqrt(x)


def exp(x):
    return _rt.xp(x).exp(x)


def log(x):
    return _rt.xp(x).log(x)


def sin(x):
    return _rt.xp(x).sin(x)


def cos(x):
    return _rt.xp(x).cos(x)


def tan(x):
    return _rt.xp(x).tan(x)


def asin(x):
    return _rt.xp(x).arcsin(x)


def acos(x):
    return _rt.xp(x).arccos(x)


def atan(x):
    return _rt.xp(x).arctan(x)


def sinh(x):
    return _rt.xp(x).sinh(x)


def cosh(x):
    return _rt.xp(x).cosh(x)


def tanh(x):
    return _rt.xp(x).tanh(x)


def asinh(x):
    return _rt.xp(x).arcsinh(x)


def acosh(x):
    return _rt.xp(x).arccosh(x)


def atanh(x):
    return _rt.xp(x).arctanh(x)


def erf(x):
    return jax.scipy.special.erf(x).astype(x.dtype)


def _extreme(reduce, pairwise, xs):
    """Max or Min of xs. A floating-point one is `reduce` (max or min) over the inputs stacked, whose
    gradient is shared equally among every input that ties for the result, as Shorokoo's own rule
    shares it."""
    xp = _rt.xp(*xs)
    if len(xs) > 1 and not _rt.is_integral(xs[0]):
        return reduce(xp.stack(xp.broadcast_arrays(*xs)), axis=0)
    return functools.reduce(pairwise, xs)


def max_(*xs):
    xp = _rt.xp(*xs)
    return _extreme(xp.max, xp.maximum, xs)


def min_(*xs):
    xp = _rt.xp(*xs)
    return _extreme(xp.min, xp.minimum, xs)


def sum_(*xs):
    return functools.reduce(add, xs)


def mean(*xs):
    xp = _rt.xp(*xs)
    return div(sum_(*xs), xp.asarray(len(xs), dtype=xs[0].dtype))


def clip(x, lo=None, hi=None, /, *, min=None, max=None):
    """Clip, whose gradient is Shorokoo's rule: 1 where x lies within the bounds, the bounds
    themselves included, 0 elsewhere, and none to the bounds."""
    xp = _rt.xp(x, lo, hi)
    if lo is None and min is not None:
        lo = np.asarray(min, dtype=x.dtype)
    if hi is None and max is not None:
        hi = np.asarray(max, dtype=x.dtype)
    inside = xp.ones(x.shape, dtype=bool)
    result = _rt.detach(x)
    if lo is not None:
        lo = _rt.detach(lo).astype(x.dtype)
        inside = inside & ~(x < lo)
        result = xp.maximum(result, lo)
    if hi is not None:
        # After the lower bound, so that a lower bound above the upper one yields the upper one, as
        # ONNX specifies.
        hi = _rt.detach(hi).astype(x.dtype)
        inside = inside & ~(x > hi)
        result = xp.minimum(result, hi)
    return xp.where(inside, x, result).astype(x.dtype)


def cumsum(x, axis, /, *, exclusive=0, reverse=0):
    return _cumulative("cumsum", 0, x, axis, exclusive, reverse)


def cumprod(x, axis, /, *, exclusive=0, reverse=0):
    return _cumulative("cumprod", 1, x, axis, exclusive, reverse)


def _cumulative(name, identity, x, axis, exclusive, reverse):
    xp = _rt.xp(x)
    dim = _rt.ints(axis, "CumSum" if name == "cumsum" else "CumProd", "its axis")[0] % max(x.ndim, 1)
    source = xp.flip(x, dim) if reverse else x
    result = getattr(xp, name)(source, axis=dim, dtype=x.dtype)
    if exclusive and result.shape[dim]:
        head_shape = list(result.shape)
        head_shape[dim] = 1
        head = xp.full(head_shape, identity, dtype=x.dtype)
        kept = [slice(None)] * result.ndim
        kept[dim] = slice(0, result.shape[dim] - 1)
        result = xp.concatenate([head, result[tuple(kept)]], axis=dim)
    return xp.flip(result, dim) if reverse else result


# ---- activations -----------------------------------------------------------------------------

def sigmoid(x):
    return jax.nn.sigmoid(x)


def relu(x):
    xp = _rt.xp(x)
    return xp.maximum(x, xp.zeros((), dtype=x.dtype)) if _rt.is_integral(x) else xp.where(x > 0, x, xp.zeros_like(x))


# Where an activation's pieces meet, the gradient is that of the piece Shorokoo's own rule takes:
# the negative one at 0 for LeakyRelu and PRelu, and 0 where HardSigmoid meets its bounds.

def leaky_relu(x, *, alpha=0.01):
    xp = _rt.xp(x)
    return xp.where(x > 0, x, x * np.asarray(alpha, dtype=x.dtype))


# The branch `where` does not select still has its gradient taken, multiplied by zero: an
# exponential there must see only the values it is selected for, or an overflow to inf in it makes
# the gradient 0 * inf = NaN. Hence the clamps inside expm1.

def _alpha(value, x):
    return np.asarray(value, dtype=x.dtype)


def elu(x, *, alpha=1.0):
    xp = _rt.xp(x)
    return xp.where(x > 0, x, _alpha(alpha, x) * xp.expm1(xp.minimum(x, 0)))


def selu(x, *, alpha=1.67326319217681884765625, gamma=1.05070102214813232421875):
    xp = _rt.xp(x)
    return _alpha(gamma, x) * xp.where(x > 0, x, _alpha(alpha, x) * xp.expm1(xp.minimum(x, 0)))


def celu(x, *, alpha=1.0):
    xp = _rt.xp(x)
    return xp.where(x > 0, x, _alpha(alpha, x) * xp.expm1(xp.minimum(x, 0) / _alpha(alpha, x)))


def thresholded_relu(x, *, alpha=1.0):
    xp = _rt.xp(x)
    return xp.where(x > _alpha(alpha, x), x, xp.zeros_like(x))


def hard_sigmoid(x, *, alpha=0.2, beta=0.5):
    xp = _rt.xp(x)
    t = _alpha(alpha, x) * x + _alpha(beta, x)
    return xp.where((t > 0) & (t < 1), t, _rt.detach(xp.clip(t, 0, 1)))


def hard_swish(x):
    return x * hard_sigmoid(x, alpha=1.0 / 6, beta=0.5)


def softplus(x):
    # log(e^x + e^0), without overflow, and with a gradient (the sigmoid) that stays finite.
    return _rt.xp(x).logaddexp(x, _rt.xp(x).zeros_like(x))


def softsign(x):
    return x / (1 + _rt.xp(x).abs(x))


def mish(x):
    return x * _rt.xp(x).tanh(softplus(x))


def gelu(x, *, approximate="none"):
    return jax.nn.gelu(x, approximate=approximate == "tanh").astype(x.dtype)


def swish(x, *, alpha=1.0):
    return x * jax.nn.sigmoid(_alpha(alpha, x) * x)


def shrink(x, *, lambd=0.5, bias=0.0):
    xp = _rt.xp(x)
    lambd, bias = np.asarray(lambd, dtype=np.float64), np.asarray(bias, dtype=np.float64)
    shrunk = xp.where(x < -lambd, x + bias, xp.where(x > lambd, x - bias, xp.zeros_like(x)))
    return shrunk.astype(x.dtype)


def prelu(x, slope):
    xp = _rt.xp(x, slope)
    if _is_unsigned(x):
        return x
    return xp.where(x > 0, x, x * slope)


def _flattened(fn, x, axis, opset):
    """Softmax-like `fn` over `axis`: before opset 13 over the input coerced to 2D at `axis`
    (default 1), from 13 over `axis` alone (default -1)."""
    if opset >= 13:
        return fn(x, -1 if axis is None else axis)
    axis = 1 if axis is None else axis
    if x.ndim == 0:
        return fn(x.reshape(1, 1), 1).reshape(())
    axis = axis % x.ndim
    rows = math.prod(x.shape[:axis])
    return fn(x.reshape(rows, -1), 1).reshape(x.shape)


def softmax(x, *, axis=None, _opset):
    return _flattened(lambda v, d: jax.nn.softmax(v, axis=d), x, axis, _opset)


def log_softmax(x, *, axis=None, _opset):
    return _flattened(lambda v, d: jax.nn.log_softmax(v, axis=d), x, axis, _opset)


def hardmax(x, *, axis=None, _opset):
    def one_hot_of_first_max(v, d):
        xp = _rt.xp(v)
        index = xp.argmax(v, axis=d, keepdims=True)
        positions = xp.arange(v.shape[d]).reshape([-1 if i == d % v.ndim else 1 for i in range(v.ndim)])
        return (positions == index).astype(v.dtype)
    return _flattened(one_hot_of_first_max, x, axis, _opset)


# ---- casts -----------------------------------------------------------------------------------

def cast(x, *, to, saturate=1, round_mode="up"):
    target = _rt.jax_dtype(int(to))
    xp = _rt.xp(x)
    if target in _rt.FLOAT8 and saturate and _rt.is_floating(x):
        # Beyond the largest finite value, infinities included, saturates to it.
        limit = float(jnp.finfo(target).max)
        return xp.clip(x, -limit, limit).astype(target)
    if target == _rt.jax_dtype(17) and _rt.is_floating(x):
        # Unsaturated, a value that rounds past 448 -- beyond 464, halfway to the next step -- has
        # no encoding and is NaN.
        return xp.where(xp.abs(x) > 464, xp.asarray(math.nan, dtype=x.dtype), x).astype(target)
    if target == np.bool_:
        return x != 0
    return x.astype(target)


def cast_like(x, like, *, saturate=1, round_mode="up"):
    return cast(x, to=_rt.dtype_code(like), saturate=saturate, round_mode=round_mode)
