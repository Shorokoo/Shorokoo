"""ONNX convolution and pooling: Conv, ConvTranspose, ConvInteger, DeformConv, the max, average
and Lp pools with their global forms, MaxUnpool and MaxRoiPool.

Every helper pads explicitly (ONNX's pads may be asymmetric or, cropping, negative, and an auto_pad
of SAME_UPPER or SAME_LOWER puts the odd element at a chosen end), then runs the kernel unpadded
over exactly the windows ONNX defines. A convolution runs in full precision (`runtime.PRECISION`),
and a 16-bit float one, like an average or Lp pool of one, is computed in float32 and rounded back.
"""

import itertools
import math

import jax
import jax.numpy as jnp
import numpy as np

from . import runtime as _rt

_NARROW_FLOATS = (np.dtype(np.float16), _rt.jax_dtype(16))


# ---- geometry --------------------------------------------------------------------------------

def _pads(sizes, kernel, strides, dilations, pads, auto_pad):
    """(begins, ends): the explicit pads, or those auto_pad resolves to."""
    n = len(sizes)
    if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
        begins, ends = [], []
        for size, k, s, d in zip(sizes, kernel, strides, dilations):
            out = -(-size // s)
            total = max(0, (out - 1) * s + (k - 1) * d + 1 - size)
            small = total // 2
            if auto_pad == "SAME_UPPER":
                begins.append(small)
                ends.append(total - small)
            else:
                begins.append(total - small)
                ends.append(small)
        return begins, ends
    if auto_pad == "VALID" or not pads:
        return [0] * n, [0] * n
    return list(pads[:n]), list(pads[n:])


def _pool_out(size, k, s, d, begin, end, ceil_mode):
    span = size + begin + end - ((k - 1) * d + 1)
    out = (-(-span // s) if ceil_mode else span // s) + 1
    if ceil_mode and (out - 1) * s >= size + begin:
        out -= 1
    return out


def _pad_spatial(x, begins, ends, value=0):
    """x padded with `value` (or, for a negative pad, cropped) along its spatial dims."""
    if not any(begins) and not any(ends):
        return x
    config = [(0, 0, 0)] * (x.ndim - len(begins)) + [(b, e, 0) for b, e in zip(begins, ends)]
    return jax.lax.pad(x, np.asarray(value, dtype=x.dtype), config)


def _windows(x, kernel_shape, strides, dilations, pads, auto_pad, ceil_mode):
    """The geometry of a pool: (kernel, strides, dilations, begins, explicit ends, output sizes,
    ends that make exactly those windows fit)."""
    n = x.ndim - 2
    sizes = list(x.shape[2:])
    kernel = list(kernel_shape)
    strides = list(strides) if strides else [1] * n
    dilations = list(dilations) if dilations else [1] * n
    begins, ends = _pads(sizes, kernel, strides, dilations, pads, auto_pad)
    outs = [_pool_out(*g, ceil_mode) for g in zip(sizes, kernel, strides, dilations, begins, ends)]
    fits = [(o - 1) * s + (k - 1) * d + 1 - b - size
            for o, s, k, d, b, size in zip(outs, strides, kernel, dilations, begins, sizes)]
    return kernel, strides, dilations, begins, ends, outs, fits


def _lowest(dtype):
    return -math.inf if jnp.issubdtype(dtype, jnp.floating) else jnp.iinfo(dtype).min


def _computable(x):
    """x in the dtype a float computation runs in -- float32 for a 16-bit float -- and the dtype
    to return, or None where it is x's own."""
    if np.dtype(x.dtype) in _NARROW_FLOATS:
        return x.astype(np.float32), x.dtype
    return x, None


def _back(y, dtype):
    return y if dtype is None else y.astype(dtype)


def _window_reduce(values, init, op, kernel, strides, dilations):
    """`op` (add or max) over every window of `values` [N, C, ...], already padded to fit."""
    ones = (1, 1)
    return jax.lax.reduce_window(values, np.asarray(init, dtype=values.dtype), op, ones + tuple(kernel),
                                 ones + tuple(strides), "VALID", window_dilation=ones + tuple(dilations))


def _taps(values, kernel, strides, dilations, outs):
    """Every tap of the kernel over the output windows of `values` [N, C, ...], already padded to
    fit, stacked in the kernel's row-major order: [K, N, C, *outs]."""
    xp = _rt.xp(values)
    taps = []
    for offset in itertools.product(*(range(k) for k in kernel)):
        index = tuple(slice(o * d, o * d + (out - 1) * s + 1, s)
                      for o, d, s, out in zip(offset, dilations, strides, outs))
        taps.append(values[(slice(None), slice(None)) + index])
    return xp.stack(taps)


# ---- convolution -----------------------------------------------------------------------------

def _conv(x, w, strides, padding, dilations, group, lhs_dilation=None):
    return jax.lax.conv_general_dilated(
        x, w.astype(x.dtype), tuple(strides), padding, lhs_dilation=lhs_dilation,
        rhs_dilation=tuple(dilations), feature_group_count=group, precision=_rt.PRECISION)


def conv(x, w, b=None, /, *, auto_pad="NOTSET", dilations=None, group=1, kernel_shape=None, pads=None,
         strides=None):
    n = x.ndim - 2
    x, dtype = _computable(x)
    kernel = list(kernel_shape) if kernel_shape else list(w.shape[2:])
    strides = list(strides) if strides else [1] * n
    dilations = list(dilations) if dilations else [1] * n
    begins, ends = _pads(list(x.shape[2:]), kernel, strides, dilations, pads, auto_pad)
    y = _conv(x, w, strides, list(zip(begins, ends)), dilations, group)
    if b is not None:
        y = y + b.astype(y.dtype).reshape([-1] + [1] * n)
    return _back(y, dtype)


def conv_integer(x, w, x_zero_point=None, w_zero_point=None, /, *, auto_pad="NOTSET", dilations=None,
                 group=1, kernel_shape=None, pads=None, strides=None):
    # Exact in float64: every product of two 8-bit values, and any sum of them a tensor can hold,
    # is an integer below 2**53.
    xs = x.astype(np.float64)
    ws = w.astype(np.float64)
    if x_zero_point is not None:
        xs = xs - x_zero_point.astype(np.float64)
    if w_zero_point is not None:
        zp = w_zero_point.astype(np.float64)
        if zp.ndim == 1 and zp.size > 1:
            zp = zp.reshape([-1] + [1] * (ws.ndim - 1))
        ws = ws - zp
    y = conv(xs, ws, auto_pad=auto_pad, dilations=dilations, group=group, kernel_shape=kernel_shape,
             pads=pads, strides=strides)
    return jnp.round(y).astype(np.int32)


def conv_transpose(x, w, b=None, /, *, auto_pad="NOTSET", dilations=None, group=1, kernel_shape=None,
                   output_padding=None, output_shape=None, pads=None, strides=None):
    n = x.ndim - 2
    x, dtype = _computable(x)
    sizes = list(x.shape[2:])
    kernel = list(kernel_shape) if kernel_shape else list(w.shape[2:])
    strides = list(strides) if strides else [1] * n
    dilations = list(dilations) if dilations else [1] * n
    extra = list(output_padding) if output_padding else [0] * n
    # The full transposed convolution covers (in-1)*s + (k-1)*d + 1 positions per dim; output
    # padding extends it at the end with positions no input reaches, and pads crop it.
    full = [(size - 1) * s + (k - 1) * d + 1 for size, s, k, d in zip(sizes, strides, kernel, dilations)]
    if output_shape:
        target = list(output_shape)[-n:]
        begins, ends = [], []
        for f, e, t in zip(full, extra, target):
            # An output_shape past the full extent is reached with zeros at the end.
            total = f + e - t
            if total < 0:
                begins.append(0)
                ends.append(total)
            elif auto_pad == "SAME_UPPER":
                begins.append(total // 2)
                ends.append(total - total // 2)
            else:
                begins.append(total - total // 2)
                ends.append(total // 2)
    elif auto_pad in ("SAME_UPPER", "SAME_LOWER"):
        begins, ends = [], []
        for size, s, f, e in zip(sizes, strides, full, extra):
            total = max(0, f + e - size * s)
            if auto_pad == "SAME_UPPER":
                begins.append(total // 2)
                ends.append(total - total // 2)
            else:
                begins.append(total - total // 2)
                ends.append(total // 2)
    elif auto_pad == "VALID" or not pads:
        begins, ends = [0] * n, [0] * n
    else:
        begins, ends = list(pads[:n]), list(pads[n:])
    # The transposed convolution as a convolution of the input dilated by the strides with the
    # kernel flipped and its input and output channels swapped within each group.
    c_in = w.shape[0]
    rhs = w.reshape([group, c_in // group, -1] + kernel)
    rhs = jnp.swapaxes(rhs, 1, 2).reshape([-1, c_in // group] + kernel)
    rhs = jnp.flip(rhs, tuple(range(2, 2 + n)))
    reach = [(k - 1) * d for k, d in zip(kernel, dilations)]
    y = _conv(x, rhs, [1] * n, [(r, r) for r in reach], dilations, group, lhs_dilation=tuple(strides))
    y = _pad_spatial(y, [-p for p in begins], [e - p for e, p in zip(extra, ends)])
    if b is not None:
        y = y + b.astype(y.dtype).reshape([-1] + [1] * n)
    return _back(y, dtype)


def _bilinear_gather(image, ys, xs):
    """Bilinear samples of `image` [N, G, C, H, W] at the points (ys, xs) [N, G, P]: [N, G, C, P].
    A corner outside the image contributes zero."""
    n, g, c, h, w = image.shape
    flat = image.reshape(n, g, c, h * w)
    y0 = jnp.floor(ys)
    x0 = jnp.floor(xs)
    ly = ys - y0
    lx = xs - x0
    total = 0
    for dy, wy in ((0, 1 - ly), (1, ly)):
        for dx, wx in ((0, 1 - lx), (1, lx)):
            yi = y0 + dy
            xi = x0 + dx
            inside = (yi >= 0) & (yi <= h - 1) & (xi >= 0) & (xi <= w - 1)
            index = jnp.where(inside, yi * w + xi, jnp.zeros_like(yi)).astype(np.int64)
            values = jnp.take_along_axis(flat, index[:, :, None, :], axis=3)
            weight = jnp.where(inside, wy * wx, jnp.zeros_like(wy))
            total = total + values * weight[:, :, None, :]
    return total


def deform_conv(x, w, offset, b=None, mask=None, /, *, dilations=None, group=1, kernel_shape=None,
                offset_group=1, pads=None, strides=None):
    if x.ndim != 4:
        raise NotImplementedError("DeformConv over other than 2 spatial dimensions")
    n, c, h, wd = x.shape
    kh, kw = list(kernel_shape) if kernel_shape else list(w.shape[2:])
    sh, sw = list(strides) if strides else [1, 1]
    dh, dw = list(dilations) if dilations else [1, 1]
    pt, pl, _, _ = list(pads) if pads else [0, 0, 0, 0]
    oh, ow = offset.shape[2], offset.shape[3]
    k = kh * kw
    og = offset_group
    off = offset.reshape(n, og, k, 2, oh, ow).astype(x.dtype)
    ki = np.repeat(np.arange(kh), kw).astype(x.dtype)
    kj = np.tile(np.arange(kw), kh).astype(x.dtype)
    base_y = (np.arange(oh) * sh - pt).astype(x.dtype).reshape(1, oh, 1) + (ki * dh).reshape(k, 1, 1)
    base_x = (np.arange(ow) * sw - pl).astype(x.dtype).reshape(1, 1, ow) + (kj * dw).reshape(k, 1, 1)
    ys = (base_y + off[:, :, :, 0]).reshape(n, og, k * oh * ow)
    xs = (base_x + off[:, :, :, 1]).reshape(n, og, k * oh * ow)
    cols = _bilinear_gather(x.reshape(n, og, c // og, h, wd), ys, xs)  # [N, OG, C/OG, K*OH*OW]
    cols = cols.reshape(n, og, c // og, k, oh, ow)
    if mask is not None:
        cols = cols * mask.astype(x.dtype).reshape(n, og, 1, k, oh, ow)
    cols = cols.reshape(n, group, (c // group) * k, oh * ow)
    oc = w.shape[0]
    weights = w.astype(x.dtype).reshape(group, oc // group, (c // group) * k)
    y = jnp.matmul(weights[None], cols, precision=_rt.PRECISION).reshape(n, oc, oh, ow)
    if b is not None:
        y = y + b.astype(y.dtype).reshape(1, oc, 1, 1)
    return y


# ---- pooling ---------------------------------------------------------------------------------

def max_pool(x, /, *, auto_pad="NOTSET", ceil_mode=0, dilations=None, kernel_shape, pads=None,
             storage_order=0, strides=None, _outputs):
    """MaxPool, whose gradient goes to the first maximum of each window, in the kernel's row-major
    order, as its indices output names it."""
    n = x.ndim - 2
    kernel, strides, dilations, begins, _, outs, fits = _windows(
        x, kernel_shape, strides, dilations, pads, auto_pad, ceil_mode)
    padded = _pad_spatial(x, begins, fits, _lowest(x.dtype))
    if _outputs < 2 and all(d == 1 for d in dilations):
        return (_window_reduce(padded, _lowest(x.dtype), jax.lax.max, kernel, strides, dilations),)
    # JAX differentiates a windowed max only undilated; the kernel's taps stacked are, and name
    # the first maximum as well.
    xp = _rt.xp(x)
    taps = _taps(padded, kernel, strides, dilations, outs)
    first = xp.argmax(_rt.detach(taps), axis=0)
    y = xp.take_along_axis(taps, first[None], axis=0)[0]
    if _outputs < 2:
        return (y,)
    # ONNX's indices count positions in the whole unpadded tensor, row-major, or with the spatial
    # dims column-major for storage_order 1. A padded tap holds the lowest value, so it is a
    # window's first maximum only before a tap that holds it too.
    sizes = list(x.shape[2:])
    strides_of = [0] * n
    scale = 1
    for i in (range(n) if storage_order else reversed(range(n))):
        strides_of[i] = scale
        scale *= sizes[i]
    position = np.zeros(sizes, dtype=np.int64)
    for i in range(n):
        position = position + (np.arange(sizes[i], dtype=np.int64) * strides_of[i]).reshape(
            [-1 if j == i else 1 for j in range(n)])
    channel = np.arange(x.shape[0] * x.shape[1], dtype=np.int64).reshape(list(x.shape[:2]) + [1] * n)
    position = _pad_spatial(position + channel * math.prod(sizes), begins, fits)
    indices = xp.take_along_axis(_taps(position, kernel, strides, dilations, outs), first[None], axis=0)[0]
    return y, indices


def average_pool(x, /, *, auto_pad="NOTSET", ceil_mode=0, count_include_pad=0, dilations=None,
                 kernel_shape, pads=None, strides=None):
    kernel, strides, dilations, begins, ends, _, fits = _windows(
        x, kernel_shape, strides, dilations, pads, auto_pad, ceil_mode)
    source, dtype = _computable(x)
    sums = _window_reduce(_pad_spatial(source, begins, fits), 0, jax.lax.add, kernel, strides, dilations)
    # Each window's divisor: its taps inside the input, and with count_include_pad those in the
    # pads as well -- never those past the end pad, which only ceil_mode's last window reaches.
    ones = np.ones([1, 1] + list(x.shape[2:]), dtype=source.dtype)
    if count_include_pad:
        mask = _pad_spatial(_pad_spatial(ones, begins, ends, 1), [0] * len(fits),
                            [f - e for f, e in zip(fits, ends)])
    else:
        mask = _pad_spatial(ones, begins, fits)
    counts = _window_reduce(mask, 0, jax.lax.add, kernel, strides, dilations)
    return _back(sums / counts, dtype)


def lp_pool(x, /, *, auto_pad="NOTSET", ceil_mode=0, dilations=None, kernel_shape, p=2, pads=None,
            strides=None):
    kernel, strides, dilations, begins, _, _, fits = _windows(
        x, kernel_shape, strides, dilations, pads, auto_pad, ceil_mode)
    source, dtype = _computable(x)
    powered = jnp.power(jnp.abs(_pad_spatial(source, begins, fits)), np.asarray(p, dtype=source.dtype))
    sums = _window_reduce(powered, 0, jax.lax.add, kernel, strides, dilations)
    return _back(jnp.power(sums, np.asarray(1.0 / p, dtype=source.dtype)), dtype)


def _spatial(x):
    return tuple(range(2, x.ndim))


def global_average_pool(x):
    return _rt.xp(x).mean(x, axis=_spatial(x), keepdims=True).astype(x.dtype)


def global_max_pool(x):
    return _rt.xp(x).max(x, axis=_spatial(x), keepdims=True)


def global_lp_pool(x, *, p=2):
    xp = _rt.xp(x)
    sums = xp.sum(xp.power(xp.abs(x), np.asarray(p, dtype=x.dtype)), axis=_spatial(x), keepdims=True)
    return xp.power(sums, np.asarray(1.0 / p, dtype=x.dtype))


def max_unpool(x, indices, output_shape=None, /, *, kernel_shape, pads=None, strides=None):
    n = x.ndim - 2
    strides = list(strides) if strides else [1] * n
    pads = list(pads) if pads else [0] * (2 * n)
    if output_shape is not None:
        shape = _rt.ints(output_shape, "MaxUnpool", "its output shape")
    else:
        shape = list(x.shape[:2]) + [(x.shape[2 + i] - 1) * strides[i] + kernel_shape[i] - pads[i] - pads[i + n]
                                     for i in range(n)]
    out = jnp.zeros(math.prod(shape), dtype=x.dtype)
    return out.at[indices.reshape(-1)].set(x.reshape(-1)).reshape(shape)


def max_roi_pool(x, rois, /, *, pooled_shape, spatial_scale=1.0):
    ph, pw = pooled_shape
    h, w = x.shape[2], x.shape[3]
    r = rois.astype(np.float32)
    scale = np.float32(spatial_scale)

    def rounded(v):
        # Half away from zero, as C's round() does.
        return jnp.sign(v) * jnp.floor(jnp.abs(v) + np.float32(0.5))

    batch = r[:, 0].astype(np.int64)
    x1 = rounded(r[:, 1] * scale)
    y1 = rounded(r[:, 2] * scale)
    x2 = rounded(r[:, 3] * scale)
    y2 = rounded(r[:, 4] * scale)

    def bins(start, end, count, size):
        length = jnp.maximum(end - start + 1, np.float32(1))
        step = _rt.divide(length, np.float32(count))
        k = np.arange(count, dtype=np.float32)
        lo = jnp.clip(jnp.floor(k * step[:, None]) + start[:, None], 0, size)
        hi = jnp.clip(jnp.ceil((k + 1) * step[:, None]) + start[:, None], 0, size)
        pos = np.arange(size, dtype=np.float32)
        member = (pos >= lo[:, :, None]) & (pos < hi[:, :, None])  # [R, count, size]
        return member, hi <= lo

    rows, empty_rows = bins(y1, y2, ph, h)
    cols, empty_cols = bins(x1, x2, pw, w)
    image = jnp.take(x, batch, axis=0)  # [R, C, H, W]
    low = np.asarray(_lowest(x.dtype), dtype=x.dtype)
    across = jnp.where(cols[:, None, None, :, :], image[:, :, :, None, :], low).max(-1)  # [R, C, H, PW]
    pooled = jnp.where(rows[:, None, :, :, None], across[:, :, None, :, :], low).max(-2)  # [R, C, PH, PW]
    empty = empty_rows[:, :, None] | empty_cols[:, None, :]
    return jnp.where(empty[:, None], jnp.zeros_like(pooled), pooled)
