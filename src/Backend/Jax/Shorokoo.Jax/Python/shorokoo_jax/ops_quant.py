"""ONNX quantization: QuantizeLinear, DequantizeLinear, DynamicQuantizeLinear, QLinearMatMul and
QLinearConv. (MatMulInteger and ConvInteger live with the matrix products and the convolutions.)

Rounding is half to even throughout, as ONNX specifies and numpy and jax.numpy round. A quantized
value is computed the way ONNX Runtime computes it -- the scaled value in float32, rounded, the
zero point added, then saturated to the target type -- and an integer product or convolution
exactly, before it is requantized.
"""

import jax
import numpy as np

from . import ops_elementwise as _elementwise
from . import runtime as _rt

_INTEGER_RANGES = {
    np.dtype(np.uint8): (0, 255),
    np.dtype(np.int8): (-128, 127),
    np.dtype(np.uint16): (0, 65535),
    np.dtype(np.int16): (-32768, 32767),
    np.dtype(np.int32): (-2147483648, 2147483647),
}


def _computed(x):
    return np.float64 if x.dtype == np.float64 else np.float32


def _parameter(value, x, axis, block_size):
    """A scale or zero point broadcast against x: per tensor (one element), per axis (1-D along
    `axis`) or blocked (x's rank, `block_size` elements of x per element along `axis`)."""
    if value.size == 1 and value.ndim <= 1 and not block_size:
        return value.reshape(())
    axis = axis % x.ndim
    xp = _rt.xp(value)
    if block_size:
        kept = [slice(None)] * x.ndim
        kept[axis] = slice(0, x.shape[axis])
        return xp.repeat(value, block_size, axis=axis)[tuple(kept)]
    shape = [1] * x.ndim
    shape[axis] = value.shape[0]
    return value.reshape(shape)


def _saturated(q, target):
    """q, an integral float array, saturated to the integer type `target` and cast to it."""
    lo, hi = _INTEGER_RANGES[np.dtype(target)]
    xp = _rt.xp(q)
    # Clipped again in int64: a float32 bound of int32's range rounds past it.
    return xp.clip(xp.clip(q, lo, hi).astype(np.int64), lo, hi).astype(target)


def _target(zero_point, output_dtype):
    if zero_point is not None:
        return np.dtype(zero_point.dtype)
    if output_dtype:
        return _rt.jax_dtype(output_dtype)
    return np.dtype(np.uint8)


def quantize_linear(x, y_scale, y_zero_point=None, /, *, axis=1, block_size=0, output_dtype=0, saturate=1):
    target = _target(y_zero_point, output_dtype)
    work = _computed(x)
    scale = _parameter(y_scale, x, axis, block_size).astype(work)
    scaled = _rt.divide(x.astype(work), scale)
    if target in _rt.FLOAT8:
        if y_zero_point is not None:
            scaled = scaled + _parameter(y_zero_point, x, axis, block_size).astype(work)
        return _elementwise.cast(scaled, to=_rt.dtype_code(target), saturate=saturate)
    q = _rt.xp(scaled).round(scaled)
    if y_zero_point is not None:
        q = q + _parameter(y_zero_point, x, axis, block_size).astype(work)
    return _saturated(q, target)


def dequantize_linear(x, x_scale, x_zero_point=None, /, *, axis=1, block_size=0, output_dtype=0):
    out = _rt.jax_dtype(output_dtype) if output_dtype else np.dtype(x_scale.dtype)
    scale = _parameter(x_scale, x, axis, block_size)
    wide = np.float32 if np.dtype(x.dtype) in _rt.FLOAT8 else np.int64
    value = x.astype(wide)
    if x_zero_point is not None:
        value = value - _parameter(x_zero_point, x, axis, block_size).astype(wide)
    work = np.float64 if out == np.float64 else np.float32
    return (value.astype(work) * scale.astype(work)).astype(out)


def dynamic_quantize_linear(x, /, *, _outputs):
    """uint8 y, its scale and zero point, from x's range widened to take in zero. A constant-zero
    x has scale 1, as ONNX Runtime gives it (the spec's formula divides by zero)."""
    x = _rt.detach(x)
    xp = _rt.xp(x)
    zero = np.zeros((), dtype=np.float32)
    lo = xp.minimum(xp.min(x), 0).astype(np.float32) if x.size else zero
    hi = xp.maximum(xp.max(x), 0).astype(np.float32) if x.size else zero
    scale = xp.where(hi == lo, np.float32(1), _rt.divide(hi - lo, np.float32(255)))
    zero_point = xp.round(xp.clip(0 - _rt.divide(lo, scale), 0, 255))
    y = xp.clip(xp.round(_rt.divide(x.astype(np.float32), scale)) + zero_point, 0, 255)
    return (y.astype(np.uint8), scale, zero_point.astype(np.uint8))[:_outputs]


def _requantized(accumulated, scale, zero_point):
    """An int64 accumulator requantized: scaled in float32, rounded, zero point added, saturated
    to the zero point's type."""
    xp = _rt.xp(accumulated, scale)
    q = xp.round(accumulated.astype(np.float32) * scale)
    q = q + zero_point.astype(np.float32)
    return _saturated(q, zero_point.dtype)


def _shifted(x, zero_point):
    value = x.astype(np.int64)
    return value if zero_point is None else value - zero_point.astype(np.int64)


def qlinear_matmul(a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point, /):
    """A per-row scale or zero point of a is one per row of a; a per-column one of b, one per
    column of b. The product of the shifted integers is exact in int64."""
    def per_row(v):
        return v.reshape(()) if v.size == 1 else v.reshape(v.shape + (1,))

    def per_column(v):
        return v.reshape(()) if v.size == 1 else v.reshape(v.shape[:-1] + (1, v.shape[-1]))

    left = _shifted(a, per_row(a_zero_point))
    right = _shifted(b, per_column(b_zero_point))
    accumulated = _rt.xp(left, right).matmul(left, right)
    scale = (per_row(a_scale).astype(np.float32) * per_column(b_scale).astype(np.float32)
             / y_scale.reshape(()).astype(np.float32))
    return _requantized(accumulated, scale, y_zero_point.reshape(()))


def _conv_pads(sizes, kernel, strides, dilations, pads, auto_pad):
    """(begin, end) padding per spatial axis."""
    count = len(sizes)
    if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
        result = []
        for size, k, s, d in zip(sizes, kernel, strides, dilations):
            out = -(-size // s)
            total = max(0, (out - 1) * s + (k - 1) * d + 1 - size)
            small = total // 2
            result.append((small, total - small) if auto_pad == "SAME_UPPER" else (total - small, small))
        return result
    if auto_pad == "VALID" or pads is None:
        return [(0, 0)] * count
    return [(pads[i], pads[i + count]) for i in range(count)]


def qlinear_conv(x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point, b=None, /, *,
                 auto_pad="NOTSET", dilations=None, group=1, kernel_shape=None, pads=None, strides=None):
    """The convolution of the zero-point-shifted integers, exact in float64 (a product of 8- or
    16-bit integers summed over any realistic kernel is far inside its 53-bit mantissa), plus the
    int32 bias, requantized with a per-output-channel scale x_scale * w_scale / y_scale."""
    spatial = x.ndim - 2
    kernel = list(kernel_shape) if kernel_shape is not None else list(w.shape[2:])
    strides = list(strides) if strides is not None else [1] * spatial
    dilations = list(dilations) if dilations is not None else [1] * spatial
    channels = w.shape[0]

    def per_channel(v):
        return v.reshape(()) if v.size == 1 else v.reshape((channels,) + (1,) * spatial)

    shifted_x = _shifted(x, x_zero_point.reshape(())).astype(np.float64)
    shifted_w = _shifted(w, None if w_zero_point is None else
                         (w_zero_point.reshape(()) if w_zero_point.size == 1
                          else w_zero_point.reshape((channels,) + (1,) * (spatial + 1)))).astype(np.float64)
    padding = _conv_pads(list(x.shape[2:]), kernel, strides, dilations, pads, auto_pad)
    accumulated = jax.lax.conv_general_dilated(
        shifted_x, shifted_w, window_strides=strides, padding=padding, rhs_dilation=dilations,
        feature_group_count=group, precision=_rt.PRECISION).astype(np.int64)
    if b is not None:
        accumulated = accumulated + b.astype(np.int64).reshape((channels,) + (1,) * spatial)
    scale = (x_scale.reshape(()).astype(np.float32) * per_channel(w_scale).astype(np.float32)
             / y_scale.reshape(()).astype(np.float32))
    return _requantized(accumulated, scale, y_zero_point.reshape(()))
