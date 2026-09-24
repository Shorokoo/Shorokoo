"""ONNX quantization: QuantizeLinear, DequantizeLinear, DynamicQuantizeLinear, QLinearMatMul and
QLinearConv. (MatMulInteger and ConvInteger live with the matrix products and the convolutions.)

Rounding is half to even throughout, as ONNX specifies and torch.round does. A quantized value is
computed the way ONNX Runtime computes it -- the scaled value in float32, rounded, the zero point
added, then saturated to the target type -- and an integer product or convolution exactly, in
64-bit arithmetic, before it is requantized.
"""

import torch
import torch.nn.functional as F

from . import ops_elementwise as _elementwise
from . import runtime as _rt

_FLOAT8 = (torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2, torch.float8_e5m2fnuz)
_INTEGER_RANGES = {
    torch.uint8: (0, 255),
    torch.int8: (-128, 127),
    torch.uint16: (0, 65535),
    torch.int16: (-32768, 32767),
    torch.int32: (-2147483648, 2147483647),
}


def _computed(x):
    return torch.float64 if x.dtype == torch.float64 else torch.float32


def _parameter(value, x, axis, block_size):
    """A scale or zero point broadcast against x: per tensor (one element), per axis (1-D along
    `axis`) or blocked (x's rank, `block_size` elements of x per element along `axis`)."""
    if value.numel() == 1 and value.dim() <= 1 and not block_size:
        return value.reshape(())
    axis = axis % x.dim()
    if block_size:
        value = value.repeat_interleave(block_size, dim=axis).narrow(axis, 0, x.shape[axis])
        return value
    shape = [1] * x.dim()
    shape[axis] = value.shape[0]
    return value.reshape(shape)


def _saturated(q, target):
    """q, an integral float tensor, saturated to the integer type `target` and cast to it."""
    lo, hi = _INTEGER_RANGES[target]
    q = torch.clamp(q, lo, hi)
    if target == torch.uint16:
        return q.to(torch.int64).to(target)
    return q.to(target)


def _target(zero_point, output_dtype):
    if zero_point is not None:
        return zero_point.dtype
    if output_dtype:
        return _rt.torch_dtype(output_dtype)
    return torch.uint8


def quantize_linear(x, y_scale, y_zero_point=None, /, *, axis=1, block_size=0, output_dtype=0, saturate=1):
    target = _target(y_zero_point, output_dtype)
    work = _computed(x)
    scale = _parameter(y_scale, x, axis, block_size).to(work)
    scaled = x.to(work) / scale
    if target in _FLOAT8:
        if y_zero_point is not None:
            scaled = scaled + _parameter(y_zero_point, x, axis, block_size).to(work)
        return _elementwise.cast(scaled, to=_rt.dtype_code(torch.empty((), dtype=target)), saturate=saturate)
    q = torch.round(scaled)
    if y_zero_point is not None:
        q = q + _parameter(y_zero_point, x, axis, block_size).to(work)
    return _saturated(q, target)


def dequantize_linear(x, x_scale, x_zero_point=None, /, *, axis=1, block_size=0, output_dtype=0):
    out = _rt.torch_dtype(output_dtype) if output_dtype else x_scale.dtype
    scale = _parameter(x_scale, x, axis, block_size)
    if x.dtype in _FLOAT8:
        value = x.to(torch.float32)
        if x_zero_point is not None:
            value = value - _parameter(x_zero_point, x, axis, block_size).to(torch.float32)
    else:
        value = x.to(torch.int64)
        if x_zero_point is not None:
            value = value - _parameter(x_zero_point, x, axis, block_size).to(torch.int64)
    work = torch.float64 if out == torch.float64 else torch.float32
    return (value.to(work) * scale.to(work)).to(out)


def dynamic_quantize_linear(x, /, *, _outputs):
    """uint8 y, its scale and zero point, from x's range widened to take in zero. A constant-zero
    x has scale 1, as ONNX Runtime gives it (the spec's formula divides by zero)."""
    x = x.detach()
    lo = torch.clamp(x.min(), max=0).to(torch.float32) if x.numel() else torch.zeros((), device=x.device)
    hi = torch.clamp(x.max(), min=0).to(torch.float32) if x.numel() else torch.zeros((), device=x.device)
    scale = torch.where(hi == lo, torch.ones_like(hi), (hi - lo) / 255)
    zero_point = torch.round(torch.clamp(0 - lo / scale, 0, 255))
    y = torch.clamp(torch.round(x.to(torch.float32) / scale) + zero_point, 0, 255)
    return (y.to(torch.uint8), scale, zero_point.to(torch.uint8))[:_outputs]


def _requantized(accumulated, scale, zero_point):
    """An int64 accumulator requantized: scaled in float32, rounded, zero point added, saturated
    to the zero point's type."""
    q = torch.round(accumulated.to(torch.float32) * scale)
    q = q + zero_point.to(torch.float32)
    return _saturated(q, zero_point.dtype)


def _shifted(x, zero_point):
    value = x.to(torch.int64)
    return value if zero_point is None else value - zero_point.to(torch.int64)


def qlinear_matmul(a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point, /):
    """A per-row scale or zero point of a is one per row of a; a per-column one of b, one per
    column of b."""
    def per_row(v):
        return v.reshape(()) if v.numel() == 1 else v.reshape(v.shape + (1,))

    def per_column(v):
        return v.reshape(()) if v.numel() == 1 else v.reshape(v.shape[:-1] + (1, v.shape[-1]))

    left = _shifted(a, per_row(a_zero_point)).to(torch.float64)
    right = _shifted(b, per_column(b_zero_point)).to(torch.float64)
    accumulated = torch.matmul(left, right).to(torch.int64)
    scale = per_row(a_scale).to(torch.float32) * per_column(b_scale).to(torch.float32) / y_scale.reshape(()).to(torch.float32)
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


_CONVOLUTIONS = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}


def qlinear_conv(x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point, b=None, /, *,
                 auto_pad="NOTSET", dilations=None, group=1, kernel_shape=None, pads=None, strides=None):
    """The convolution of the zero-point-shifted integers, exact in float64 (a product of 8- or
    16-bit integers summed over any realistic kernel is far inside its 53-bit mantissa), plus the
    int32 bias, requantized with a per-output-channel scale x_scale * w_scale / y_scale."""
    spatial = x.dim() - 2
    kernel = list(kernel_shape) if kernel_shape is not None else list(w.shape[2:])
    strides = list(strides) if strides is not None else [1] * spatial
    dilations = list(dilations) if dilations is not None else [1] * spatial
    channels = w.shape[0]

    def per_channel(v):
        return v.reshape(()) if v.numel() == 1 else v.reshape((channels,) + (1,) * spatial)

    shifted_x = _shifted(x, x_zero_point.reshape(())).to(torch.float64)
    shifted_w = _shifted(w, None if w_zero_point is None else
                         (w_zero_point.reshape(()) if w_zero_point.numel() == 1
                          else w_zero_point.reshape((channels,) + (1,) * (spatial + 1)))).to(torch.float64)
    padding = _conv_pads(list(x.shape[2:]), kernel, strides, dilations, pads, auto_pad)
    flat = [p for pair in reversed(padding) for p in pair]
    if any(flat):
        shifted_x = F.pad(shifted_x, flat)
    if spatial not in _CONVOLUTIONS:
        raise NotImplementedError(f"QLinearConv over {spatial} spatial axes")
    accumulated = _CONVOLUTIONS[spatial](shifted_x, shifted_w, None, strides, 0, dilations, group).to(torch.int64)
    if b is not None:
        accumulated = accumulated + b.to(torch.int64).reshape((channels,) + (1,) * spatial)
    scale = x_scale.reshape(()).to(torch.float32) * per_channel(w_scale).to(torch.float32) / y_scale.reshape(()).to(torch.float32)
    return _requantized(accumulated, scale, y_zero_point.reshape(()))
