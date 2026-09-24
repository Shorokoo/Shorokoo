"""ONNX convolution and pooling: Conv, ConvTranspose, ConvInteger, DeformConv, the max, average
and Lp pools with their global forms, MaxUnpool and MaxRoiPool.

torch's own kernels pad symmetrically and round a ceil-mode output differently, so every helper
here pads explicitly (ONNX's pads may be asymmetric, and an auto_pad of SAME_UPPER or SAME_LOWER
puts the odd element at a chosen end), then runs the kernel unpadded over exactly the windows ONNX
defines. Every helper is functional and autograd-safe: no input is written to, and float data never
leaves torch.
"""

import math

import torch
import torch.nn.functional as F

_CONV = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}
_CONV_TRANSPOSE = {1: F.conv_transpose1d, 2: F.conv_transpose2d, 3: F.conv_transpose3d}
_MAX_POOL = {1: F.max_pool1d, 2: F.max_pool2d, 3: F.max_pool3d}


# ---- geometry --------------------------------------------------------------------------------

def _rank_check(x, what):
    n = x.dim() - 2
    if n not in _CONV:
        raise NotImplementedError(f"{what} over {n} spatial dimensions (the PyTorch backend runs 1 to 3)")
    return n


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


def _pad_spatial(x, begins, ends, value=0.0):
    """x padded (or, for a negative pad, cropped) along its spatial dims."""
    flat = []
    for b, e in zip(reversed(begins), reversed(ends)):
        flat += [b, e]
    if not any(flat):
        return x
    return F.pad(x, flat, value=value)


def _windows(x, kernel_shape, strides, dilations, pads, auto_pad, ceil_mode):
    """The geometry of a pool: (kernel, strides, dilations, begins, explicit ends, output sizes,
    ends that make exactly those windows fit)."""
    n = x.dim() - 2
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
    return -math.inf if dtype.is_floating_point else torch.iinfo(dtype).min


def _computable(x):
    """x in a dtype torch's CPU pooling and convolution kernels accept, and the dtype to return."""
    if x.dtype.is_floating_point and (x.dtype == torch.float32 or x.dtype == torch.float64 or x.is_cuda):
        return x, None
    return x.to(torch.float64 if not x.dtype.is_floating_point else torch.float32), x.dtype


def _back(y, dtype):
    if dtype is None:
        return y
    if not dtype.is_floating_point:
        y = torch.round(y)
    return y.to(dtype)


# ---- convolution -----------------------------------------------------------------------------

def conv(x, w, b=None, /, *, auto_pad="NOTSET", dilations=None, group=1, kernel_shape=None, pads=None,
         strides=None):
    n = _rank_check(x, "Conv")
    x, dtype = _computable(x)
    w = w.to(x.dtype)
    b = None if b is None else b.to(x.dtype)
    kernel = list(kernel_shape) if kernel_shape else list(w.shape[2:])
    strides = list(strides) if strides else [1] * n
    dilations = list(dilations) if dilations else [1] * n
    begins, ends = _pads(list(x.shape[2:]), kernel, strides, dilations, pads, auto_pad)
    y = _CONV[n](_pad_spatial(x, begins, ends), w, b, stride=strides, dilation=dilations, groups=group)
    return _back(y, dtype)


def conv_integer(x, w, x_zero_point=None, w_zero_point=None, /, *, auto_pad="NOTSET", dilations=None,
                 group=1, kernel_shape=None, pads=None, strides=None):
    # Exact in float64: every product of two 8-bit values, and any sum of them a tensor can hold,
    # is an integer below 2**53.
    xs = x.to(torch.float64)
    ws = w.to(torch.float64)
    if x_zero_point is not None:
        xs = xs - x_zero_point.to(torch.float64)
    if w_zero_point is not None:
        zp = w_zero_point.to(torch.float64)
        if zp.dim() == 1 and zp.numel() > 1:
            zp = zp.reshape([-1] + [1] * (ws.dim() - 1))
        ws = ws - zp
    y = conv(xs, ws, auto_pad=auto_pad, dilations=dilations, group=group, kernel_shape=kernel_shape,
             pads=pads, strides=strides)
    return torch.round(y).to(torch.int32)


def conv_transpose(x, w, b=None, /, *, auto_pad="NOTSET", dilations=None, group=1, kernel_shape=None,
                   output_padding=None, output_shape=None, pads=None, strides=None):
    n = _rank_check(x, "ConvTranspose")
    x, dtype = _computable(x)
    w = w.to(x.dtype)
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
    y = _CONV_TRANSPOSE[n](x, w, None, stride=strides, dilation=dilations, groups=group)
    y = _pad_spatial(y, [-p for p in begins], [e - p for e, p in zip(extra, ends)])
    if b is not None:
        y = y + b.to(y.dtype).reshape([-1] + [1] * n)
    return _back(y, dtype)


def _bilinear_gather(image, ys, xs):
    """Bilinear samples of `image` [N, G, C, H, W] at the points (ys, xs) [N, G, P]: [N, G, C, P].
    A corner outside the image contributes zero."""
    n, g, c, h, w = image.shape
    flat = image.reshape(n, g, c, h * w)
    y0 = torch.floor(ys)
    x0 = torch.floor(xs)
    ly = ys - y0
    lx = xs - x0
    total = 0
    for dy, wy in ((0, 1 - ly), (1, ly)):
        for dx, wx in ((0, 1 - lx), (1, lx)):
            yi = y0 + dy
            xi = x0 + dx
            inside = (yi >= 0) & (yi <= h - 1) & (xi >= 0) & (xi <= w - 1)
            index = torch.where(inside, yi * w + xi, torch.zeros_like(yi)).to(torch.int64)
            values = torch.gather(flat, 3, index.unsqueeze(2).expand(n, g, c, index.shape[-1]))
            weight = torch.where(inside, wy * wx, torch.zeros_like(wy))
            total = total + values * weight.unsqueeze(2)
    return total


def deform_conv(x, w, offset, b=None, mask=None, /, *, dilations=None, group=1, kernel_shape=None,
                offset_group=1, pads=None, strides=None):
    if x.dim() != 4:
        raise NotImplementedError("DeformConv over other than 2 spatial dimensions")
    n, c, h, wd = x.shape
    kh, kw = list(kernel_shape) if kernel_shape else list(w.shape[2:])
    sh, sw = list(strides) if strides else [1, 1]
    dh, dw = list(dilations) if dilations else [1, 1]
    pt, pl, _, _ = list(pads) if pads else [0, 0, 0, 0]
    oh, ow = offset.shape[2], offset.shape[3]
    k = kh * kw
    og = offset_group
    off = offset.reshape(n, og, k, 2, oh, ow).to(x.dtype)
    ki = torch.arange(kh, device=x.device, dtype=x.dtype).repeat_interleave(kw)
    kj = torch.arange(kw, device=x.device, dtype=x.dtype).repeat(kh)
    base_y = (torch.arange(oh, device=x.device, dtype=x.dtype) * sh - pt).reshape(1, oh, 1) + (ki * dh).reshape(k, 1, 1)
    base_x = (torch.arange(ow, device=x.device, dtype=x.dtype) * sw - pl).reshape(1, 1, ow) + (kj * dw).reshape(k, 1, 1)
    ys = (base_y + off[:, :, :, 0]).reshape(n, og, k * oh * ow)
    xs = (base_x + off[:, :, :, 1]).reshape(n, og, k * oh * ow)
    cols = _bilinear_gather(x.reshape(n, og, c // og, h, wd), ys, xs)  # [N, OG, C/OG, K*OH*OW]
    cols = cols.reshape(n, og, c // og, k, oh, ow)
    if mask is not None:
        cols = cols * mask.to(x.dtype).reshape(n, og, 1, k, oh, ow)
    cols = cols.reshape(n, group, (c // group) * k, oh * ow)
    oc = w.shape[0]
    weights = w.to(x.dtype).reshape(group, oc // group, (c // group) * k)
    y = torch.matmul(weights.unsqueeze(0), cols).reshape(n, oc, oh, ow)
    if b is not None:
        y = y + b.to(y.dtype).reshape(1, oc, 1, 1)
    return y


# ---- pooling ---------------------------------------------------------------------------------

def max_pool(x, /, *, auto_pad="NOTSET", ceil_mode=0, dilations=None, kernel_shape, pads=None,
             storage_order=0, strides=None, _outputs):
    n = _rank_check(x, "MaxPool")
    kernel, strides, dilations, begins, _, _, fits = _windows(
        x, kernel_shape, strides, dilations, pads, auto_pad, ceil_mode)
    source, dtype = _computable(x)
    padded = _pad_spatial(source, begins, fits, _lowest(source.dtype))
    y, flat = _MAX_POOL[n](padded, kernel, strides, 0, dilations, False, True)
    y = _back(y, dtype)
    if _outputs < 2:
        return (y,)
    # torch's indices count positions in the padded plane of one (n, c); ONNX's count positions
    # in the whole unpadded tensor, row-major, or with the spatial dims column-major for
    # storage_order 1.
    sizes = list(x.shape[2:])
    padded_sizes = list(padded.shape[2:])
    position = torch.zeros_like(flat)
    rest = flat
    scale = 1
    order = range(n) if storage_order else reversed(range(n))
    coords = [None] * n
    for i in reversed(range(n)):
        coords[i] = rest % padded_sizes[i] - begins[i]
        rest = rest // padded_sizes[i]
    for i in order:
        position = position + coords[i] * scale
        scale *= sizes[i]
    plane = math.prod(sizes)
    channel = torch.arange(x.shape[0] * x.shape[1], device=x.device, dtype=torch.int64)
    channel = channel.reshape([x.shape[0], x.shape[1]] + [1] * n) * plane
    return y, position + channel


def _window_sums(values, kernel, strides, dilations):
    """The sum of every window of `values` [N, C, ...] (already padded to fit)."""
    n = values.dim() - 2
    c = values.shape[1]
    ones = torch.ones([c, 1] + kernel, dtype=values.dtype, device=values.device)
    return _CONV[n](values, ones, None, stride=strides, dilation=dilations, groups=c)


def average_pool(x, /, *, auto_pad="NOTSET", ceil_mode=0, count_include_pad=0, dilations=None,
                 kernel_shape, pads=None, strides=None):
    _rank_check(x, "AveragePool")
    kernel, strides, dilations, begins, ends, _, fits = _windows(
        x, kernel_shape, strides, dilations, pads, auto_pad, ceil_mode)
    source, dtype = _computable(x)
    sums = _window_sums(_pad_spatial(source, begins, fits), kernel, strides, dilations)
    # Each window's divisor: its taps inside the input, and with count_include_pad those in the
    # pads as well -- never those past the end pad, which only ceil_mode's last window reaches.
    ones = torch.ones([1, 1] + list(x.shape[2:]), dtype=source.dtype, device=x.device)
    if count_include_pad:
        mask = _pad_spatial(_pad_spatial(ones, begins, ends, 1.0), [0] * len(fits),
                            [f - e for f, e in zip(fits, ends)])
    else:
        mask = _pad_spatial(ones, begins, fits)
    counts = _window_sums(mask, kernel, strides, dilations)
    return _back(sums / counts, dtype)


def lp_pool(x, /, *, auto_pad="NOTSET", ceil_mode=0, dilations=None, kernel_shape, p=2, pads=None,
            strides=None):
    _rank_check(x, "LpPool")
    kernel, strides, dilations, begins, _, _, fits = _windows(
        x, kernel_shape, strides, dilations, pads, auto_pad, ceil_mode)
    source, dtype = _computable(x)
    powered = torch.pow(torch.abs(_pad_spatial(source, begins, fits)), p)
    return _back(torch.pow(_window_sums(powered, kernel, strides, dilations), 1.0 / p), dtype)


def _spatial(x):
    return list(range(2, x.dim()))


def global_average_pool(x):
    return torch.mean(x, dim=_spatial(x), keepdim=True)


def global_max_pool(x):
    return torch.amax(x, dim=_spatial(x), keepdim=True)


def global_lp_pool(x, *, p=2):
    return torch.pow(torch.sum(torch.pow(torch.abs(x), p), dim=_spatial(x), keepdim=True), 1.0 / p)


def max_unpool(x, indices, output_shape=None, /, *, kernel_shape, pads=None, strides=None):
    n = x.dim() - 2
    strides = list(strides) if strides else [1] * n
    pads = list(pads) if pads else [0] * (2 * n)
    if output_shape is not None:
        shape = [int(v) for v in output_shape.tolist()]
    else:
        shape = list(x.shape[:2]) + [(x.shape[2 + i] - 1) * strides[i] + kernel_shape[i] - pads[i] - pads[i + n]
                                     for i in range(n)]
    out = torch.zeros(math.prod(shape), dtype=x.dtype, device=x.device)
    return out.scatter(0, indices.reshape(-1), x.reshape(-1)).reshape(shape)


def max_roi_pool(x, rois, /, *, pooled_shape, spatial_scale=1.0):
    ph, pw = pooled_shape
    h, w = x.shape[2], x.shape[3]
    r = rois.to(torch.float32)
    # Region corners rounded half away from zero, as C's round() does.
    def rounded(v):
        return torch.sign(v) * torch.floor(torch.abs(v) + 0.5)
    batch = r[:, 0].to(torch.int64)
    x1 = rounded(r[:, 1] * spatial_scale)
    y1 = rounded(r[:, 2] * spatial_scale)
    x2 = rounded(r[:, 3] * spatial_scale)
    y2 = rounded(r[:, 4] * spatial_scale)
    def bins(start, end, count, size):
        length = torch.clamp(end - start + 1, min=1)
        step = length / count
        k = torch.arange(count, device=x.device, dtype=torch.float32)
        lo = torch.clamp(torch.floor(k * step.unsqueeze(1)) + start.unsqueeze(1), 0, size)
        hi = torch.clamp(torch.ceil((k + 1) * step.unsqueeze(1)) + start.unsqueeze(1), 0, size)
        pos = torch.arange(size, device=x.device, dtype=torch.float32)
        member = (pos >= lo.unsqueeze(2)) & (pos < hi.unsqueeze(2))  # [R, count, size]
        return member, hi <= lo
    rows, empty_rows = bins(y1, y2, ph, h)
    cols, empty_cols = bins(x1, x2, pw, w)
    image = x[batch]  # [R, C, H, W]
    low = _lowest(x.dtype)
    across = torch.where(cols[:, None, None, :, :], image[:, :, :, None, :], low).amax(-1)  # [R, C, H, PW]
    pooled = torch.where(rows[:, None, :, :, None], across[:, :, None, :, :], low).amax(-2)  # [R, C, PH, PW]
    empty = empty_rows[:, :, None] | empty_cols[:, None, :]
    return torch.where(empty[:, None], torch.zeros_like(pooled), pooled)
