"""ONNX image and geometry: Resize (and the Upsample it replaced), GridSample, AffineGrid, RoiAlign,
CenterCropPad, Col2Im, DepthToSpace and SpaceToDepth. (ImageDecoder and NonMaxSuppression, whose
output shapes their inputs' values decide, are refused before anything here runs.)

Resize runs one axis at a time: a nearest resize gathers along the axis, and a linear or cubic one
multiplies by an [output, input] matrix of interpolation weights, which covers antialiasing,
exclude_outside and edge clamping alike and lets JAX differentiate it. Its scales, sizes and
region of interest must be concrete; the coordinates and the weights are computed from them on the
host, output coordinates mapped to input ones in float32, as ONNX Runtime maps them, so that a
nearest resize rounds the same way.
"""

import itertools
import math

import jax.numpy as jnp
import numpy as np

from . import runtime as _rt


def _floats(value, operator, what):
    if not _rt.concrete(value):
        raise _rt.DataDependentShape(operator, what)
    return [float(v) for v in np.asarray(value).reshape(-1).tolist()]


def _present(value):
    return value is not None and value.size > 0


def _matmul(a, b):
    if _rt.concrete(a, b):
        return np.matmul(a, b)
    return jnp.matmul(a, b, precision=_rt.PRECISION)


# ---- Resize ----------------------------------------------------------------------------------

_f32 = np.float32


def _original(out_size, in_size, scale, mode, roi_start, roi_end):
    """The input coordinate (float32) of every output position along one axis."""
    x = np.arange(out_size, dtype=np.float32)
    s = _f32(scale)
    if mode == "asymmetric":
        return x / s
    if mode == "half_pixel":
        return (x + _f32(0.5)) / s - _f32(0.5)
    if mode == "pytorch_half_pixel":
        return (x + _f32(0.5)) / s - _f32(0.5) if out_size > 1 else np.zeros_like(x)
    if mode == "align_corners":
        return x * _f32(in_size - 1) / _f32(out_size - 1) if out_size > 1 else np.zeros_like(x)
    if mode == "half_pixel_symmetric":
        # The offset in float32, the rest in float64, as ONNX Runtime writes it.
        adjustment = _f32(out_size) / (s * _f32(in_size))
        offset = _f32(in_size) / _f32(2) * (_f32(1) - adjustment)
        return (np.float64(offset) + (x.astype(np.float64) + 0.5) / np.float64(s) - 0.5).astype(np.float32)
    if mode == "tf_crop_and_resize":
        start, end = _f32(roi_start), _f32(roi_end)
        span = _f32(in_size - 1)
        if out_size > 1:
            return start * span + x * (end - start) * span / _f32(out_size - 1)
        return np.full_like(x, float(_f32(0.5) * (start + end) * span))
    raise NotImplementedError(f"Resize coordinate_transformation_mode '{mode}'")


def _round_half_away(x):
    xp = _rt.xp(x)
    whole = xp.trunc(x)
    return whole + xp.where(xp.abs(x - whole) >= 0.5, xp.sign(x), xp.zeros_like(x))


def _nearest(x, rule, downsampling):
    if rule == "round_prefer_floor":
        return np.where(x - np.floor(x) == 0.5, np.floor(x), _round_half_away(x))
    if rule == "round_prefer_ceil":
        return _round_half_away(x)
    if rule == "floor":
        return np.floor(x)
    if rule == "ceil":
        return np.ceil(x)
    if rule == "simple":
        return np.ceil(x) if downsampling else np.trunc(x)
    raise NotImplementedError(f"Resize nearest_mode '{rule}'")


def _cubic(t, a):
    t = np.abs(t)
    near = ((a + 2) * t - (a + 3)) * t * t + 1
    far = ((a * t - 5 * a) * t + 8 * a) * t - 4 * a
    return np.where(t <= 1, near, np.where(t < 2, far, 0.0))


def _linear(t):
    return np.maximum(1 - np.abs(t), 0)


def _weights(coords, in_size, mode, scale, antialias, exclude_outside, a):
    """The [output, input] interpolation matrix (float64) of one axis."""
    x = coords.astype(np.float64)
    if mode == "linear" and not antialias:
        x = np.clip(x, 0, in_size - 1)
    support = 1.0 if mode == "linear" else 2.0
    stretch = 1.0
    if antialias and scale < 1:
        stretch = float(scale)
        support /= stretch
    reach = int(math.ceil(support)) + 1
    taps = np.floor(x)[:, None] + np.arange(-reach, reach + 1, dtype=np.float64)
    distance = (taps - x[:, None]) * stretch
    w = _linear(distance) if mode == "linear" else _cubic(distance, a)
    inside = (taps >= 0) & (taps <= in_size - 1)
    if exclude_outside:
        w = np.where(inside, w, 0.0)
    total = w.sum(axis=1, keepdims=True)
    if exclude_outside or antialias:
        w = w / np.where(total == 0, 1.0, total)
    index = np.clip(taps, 0, in_size - 1).astype(np.int64)
    matrix = np.zeros((coords.shape[0], in_size), dtype=np.float64)
    np.add.at(matrix, (np.arange(coords.shape[0])[:, None], index), w)
    return matrix


def resize(x, *inputs, antialias=0, axes=None, coordinate_transformation_mode="half_pixel",
           cubic_coeff_a=-0.75, exclude_outside=0, extrapolation_value=0.0, keep_aspect_ratio_policy="stretch",
           mode="nearest", nearest_mode="round_prefer_floor", _opset):
    if _opset < 11:
        roi, scales, sizes = None, inputs[0], None
        coordinate_transformation_mode = "asymmetric"
        nearest_mode = "simple"
    else:
        roi, scales, sizes = (list(inputs) + [None, None, None])[:3]
    rank = x.ndim
    axes = [a % rank for a in axes] if axes is not None else list(range(rank))
    in_shape = list(x.shape)
    out_shape = list(in_shape)
    axis_scale = [1.0] * rank
    if _present(scales):
        for axis, s in zip(axes, _floats(scales, "Resize", "its scales")):
            s = float(_f32(s))
            axis_scale[axis] = s
            out_shape[axis] = int(float(_f32(in_shape[axis]) * _f32(s)))
    elif _present(sizes):
        wanted = _rt.ints(sizes, "Resize", "its sizes")
        ratios = [float(_f32(size) / _f32(in_shape[axis])) for axis, size in zip(axes, wanted)]
        if keep_aspect_ratio_policy == "stretch":
            for axis, size, r in zip(axes, wanted, ratios):
                out_shape[axis] = size
                axis_scale[axis] = r
        else:
            common = min(ratios) if keep_aspect_ratio_policy == "not_larger" else max(ratios)
            for axis in axes:
                axis_scale[axis] = common
                out_shape[axis] = int(_round_half_away(_f32(common) * _f32(in_shape[axis])))
    else:
        raise ValueError("Resize needs scales or sizes")
    roi_values = _floats(roi, "Resize", "its region of interest") if _present(roi) else None

    xp = _rt.xp(x)
    cropping = coordinate_transformation_mode == "tf_crop_and_resize"
    outside = None
    y = x
    if mode != "nearest":
        y = y.astype(x.dtype if _rt.is_floating(x) else np.float32)
    for axis in range(rank):
        n_in, n_out, s = in_shape[axis], out_shape[axis], axis_scale[axis]
        if n_in == n_out and s == 1.0 and not cropping:
            continue
        start, end = 0.0, 1.0
        if roi_values is not None:
            if len(roi_values) == 2 * rank:
                start, end = roi_values[axis], roi_values[axis + rank]
            elif axis in axes:
                k = axes.index(axis)
                start, end = roi_values[k], roi_values[k + len(axes)]
        coords = _original(n_out, n_in, s, coordinate_transformation_mode, start, end)
        if cropping:
            shape = [1] * rank
            shape[axis] = n_out
            off = ((coords < 0) | (coords > n_in - 1)).reshape(shape)
            outside = off if outside is None else outside | off
        if mode == "nearest":
            index = np.clip(_nearest(coords, nearest_mode, s < 1), 0, n_in - 1).astype(np.int64)
            y = xp.take(y, index, axis=axis)
        else:
            matrix = _weights(coords, n_in, mode, s, antialias, exclude_outside, cubic_coeff_a)
            y = xp.moveaxis(_matmul(xp.moveaxis(y, axis, -1), matrix.T.astype(y.dtype)), -1, axis)
    if outside is not None:
        y = xp.where(outside, np.asarray(extrapolation_value, dtype=y.dtype), y)
    if y.dtype != x.dtype:
        # Antialiasing rounds to the nearest integer and plain interpolation truncates, as ONNX
        # Runtime does, and either saturates where a cubic overshoots the type's range.
        y = _round_half_away(y) if antialias and mode != "nearest" else xp.trunc(y)
        info = np.iinfo(x.dtype)
        y = xp.clip(y, info.min, info.max).astype(x.dtype)
    return y


def upsample(x, scales, /, *, mode="nearest"):
    return resize(x, scales, mode=mode, _opset=10)


# ---- sampling --------------------------------------------------------------------------------

def _grid_taps(coords, size, mode, padding_mode, align_corners):
    """ONNX Runtime's taps along one spatial axis of every sampled point: (index, weight) pairs,
    the index already brought inside [0, size) by the padding mode, the weight zero for a tap that
    falls outside under zero padding."""
    xp = _rt.xp(coords)
    if mode == "nearest":
        base = xp.round(coords)
        taps = [(base, xp.ones_like(coords))]
    elif mode == "linear":
        base = xp.floor(coords)
        taps = [(base, base + 1 - coords), (base + 1, coords - base)]
    else:
        base = xp.floor(coords)
        t = coords - base
        a = np.asarray(-0.75, dtype=coords.dtype)

        def near(d):
            return ((a + 2) * d - (a + 3)) * d * d + 1

        def far(d):
            return ((a * d - 5 * a) * d + 8 * a) * d - 4 * a

        taps = [(base - 1, far(t + 1)), (base, near(t)), (base + 1, near(1 - t)), (base + 2, far(2 - t))]
    result = []
    for position, weight in taps:
        position = _rt.detach(position)
        if padding_mode == "zeros":
            inside = (position >= 0) & (position <= size - 1)
            weight = xp.where(inside, weight, xp.zeros_like(weight))
        elif padding_mode == "reflection":
            position = _reflected(position, 0.0 if align_corners else -0.5,
                                  size - 1.0 if align_corners else size - 0.5)
        index = xp.clip(position, 0, size - 1).astype(np.int64)
        result.append((index, weight))
    return result


def _reflected(x, low, high):
    """ONNX Runtime's reflection of the integral coordinates x into [low, high]."""
    xp = _rt.xp(x)
    span = high - low
    if span <= 0:
        return xp.zeros_like(x)
    below = low - x
    above = x - high
    n_below = xp.floor(below / span)
    n_above = xp.floor(above / span)
    r_below = below - n_below * span
    r_above = above - n_above * span
    from_below = xp.where(n_below % 2 == 0, low + r_below, high - r_below)
    from_above = xp.where(n_above % 2 == 0, high - r_above, low + r_above)
    return xp.trunc(xp.where(x < low, from_below, xp.where(x > high, from_above, x)))


def grid_sample(x, grid, /, *, align_corners=0, mode="linear", padding_mode="zeros"):
    """Samples x ([N, C, *spatial]) at grid ([N, *out, rank]), whose last axis holds normalized
    coordinates in [-1, 1], x (the last spatial axis) first. Interpolated as ONNX Runtime does:
    each tap an integer pixel index, brought inside the image by the padding mode."""
    mode = {"bilinear": "linear", "bicubic": "cubic"}.get(mode, mode)
    xp = _rt.xp(x, grid)
    grid = grid.astype(x.dtype)
    sizes = list(x.shape[2:])
    rank = len(sizes)
    per_axis = []
    for axis, size in enumerate(sizes):
        g = grid[..., rank - 1 - axis]
        if align_corners:
            coords = (g + 1) / 2 * (size - 1)
        else:
            coords = ((g + 1) * size - 1) / 2
        per_axis.append(_grid_taps(coords, size, mode, padding_mode, align_corners))
    batch = np.arange(x.shape[0]).reshape((-1,) + (1,) * rank)
    result = None
    for combination in itertools.product(*per_axis):
        weight = combination[0][1]
        for _, w in combination[1:]:
            weight = weight * w
        values = x[(batch, slice(None)) + tuple(index for index, _ in combination)]  # [N, *out, C]
        term = values * weight[..., None]
        result = term if result is None else result + term
    return xp.moveaxis(result, -1, 1)


def affine_grid(theta, size, /, *, align_corners=0):
    # In ONNX Runtime's float arithmetic, so that a sampler rounding a grid point to the nearest
    # pixel sees the same point; a one-pixel axis is at -1 under align_corners, as ONNX's reference
    # puts it.
    dims = _rt.ints(size, "AffineGrid", "its size")[2:]
    dtype = np.dtype(theta.dtype)
    axes = []
    for n in dims:
        if n == 1:
            base = np.full((1,), -1.0 if align_corners else 0.0, dtype=dtype)
        else:
            step = np.asarray(2.0, dtype=dtype) / np.asarray(n - 1, dtype=dtype)
            base = -1 + np.arange(n, dtype=dtype) * step
            base[-1] = 1.0
            if not align_corners:
                base = base * np.asarray(n - 1, dtype=dtype) / np.asarray(n, dtype=dtype)
        axes.append(base)
    # Grid points (x, y[, z]), x varying fastest: the reverse of the size's spatial order.
    mesh = np.meshgrid(*axes, indexing="ij")
    points = [m.reshape(-1) for m in reversed(mesh)]
    rank = len(dims)
    xp = _rt.xp(theta)
    rows = []
    for i in range(rank):
        row = theta[:, i, rank - 1:rank] * points[rank - 1]
        for k in reversed(range(rank - 1)):
            row = theta[:, i, k:k + 1] * points[k] + row
        rows.append(row + theta[:, i, rank:rank + 1])
    return xp.stack(rows, axis=-1).reshape([theta.shape[0]] + dims + [rank])


def roi_align(x, rois, batch_indices, /, *, coordinate_transformation_mode="half_pixel", mode="avg",
              output_height=1, output_width=1, sampling_ratio=0, spatial_scale=1.0):
    """Each region pooled from its image. Where sampling_ratio is 0 the samples per bin follow from
    the region's size, which must then be concrete."""
    xp = _rt.xp(x, rois, batch_indices)
    height, width = x.shape[2], x.shape[3]
    offset = _f32(0.5 if coordinate_transformation_mode == "half_pixel" else 0.0)
    scale = _f32(spatial_scale)
    boxes = rois.astype(np.float32)
    results = []
    for r in range(rois.shape[0]):
        x1, y1, x2, y2 = (boxes[r, k] for k in range(4))
        start_w = x1 * scale - offset
        start_h = y1 * scale - offset
        roi_w = x2 * scale - offset - start_w
        roi_h = y2 * scale - offset - start_h
        if offset == 0:
            roi_w = _rt.xp(rois).maximum(roi_w, _f32(1))
            roi_h = _rt.xp(rois).maximum(roi_h, _f32(1))
        bin_h = roi_h / _f32(output_height)
        bin_w = roi_w / _f32(output_width)
        if sampling_ratio > 0:
            grid_h = grid_w = sampling_ratio
        else:
            grid_h = int(math.ceil(_rt.number(bin_h, "RoiAlign", "a region's size (sampling_ratio 0)")))
            grid_w = int(math.ceil(_rt.number(bin_w, "RoiAlign", "a region's size (sampling_ratio 0)")))
        count = max(grid_h * grid_w, 1)
        ys = _roi_samples(start_h, bin_h, output_height, grid_h)
        xs = _roi_samples(start_w, bin_w, output_width, grid_w)
        results.append(_roi_bin_values(xp.take(x, batch_indices[r], axis=0), ys, xs, height, width, mode, count))
    if not results:
        return xp.zeros((0, x.shape[1], output_height, output_width), dtype=x.dtype)
    return xp.stack(results)


def _roi_samples(start, bin_size, bins, grid):
    """The sample coordinates [bins, grid] along one axis of a region."""
    p = np.arange(bins, dtype=np.float32)[:, None]
    i = np.arange(grid, dtype=np.float32)[None, :]
    return start + p * bin_size + (i + _f32(0.5)) * bin_size / _f32(grid)


def _roi_corners(coords, size):
    """ONNX Runtime's bilinear taps of each sample: (low index, high index, low weight, high weight),
    the weights zero for a sample outside [-1, size]."""
    xp = _rt.xp(coords)
    valid = (coords >= -1.0) & (coords <= size)
    c = xp.maximum(coords, _f32(0))
    low = xp.floor(c).astype(np.int64)
    at_end = low >= size - 1
    low = xp.where(at_end, size - 1, low)
    high = xp.where(at_end, low, low + 1)
    c = xp.where(at_end, low.astype(c.dtype), c)
    frac = c - low.astype(c.dtype)
    zero = xp.zeros_like(frac)
    return low, high, xp.where(valid, 1 - frac, zero), xp.where(valid, frac, zero)


def _roi_bin_values(image, ys, xs, height, width, mode, count):
    """One region's pooled values [C, bins_h, bins_w]."""
    xp = _rt.xp(image, ys, xs)
    ylo, yhi, wyl, wyh = _roi_corners(ys, height)
    xlo, xhi, wxl, wxh = _roi_corners(xs, width)
    terms = []
    for yi, wy in ((ylo, wyl), (yhi, wyh)):
        rows = image[:, yi]  # [C, BH, GH, W]
        for xi, wx in ((xlo, wxl), (xhi, wxh)):
            values = rows[:, :, :, xi]  # [C, BH, GH, BW, GW]
            weight = wy[:, :, None, None] * wx[None, None, :, :]
            terms.append(values * weight.astype(image.dtype))
    if mode == "avg":
        return sum(terms).sum(axis=(2, 4)) / np.asarray(count, dtype=image.dtype)
    if ys.shape[1] == 0 or xs.shape[1] == 0:
        return xp.zeros((image.shape[0], ys.shape[0], xs.shape[0]), dtype=image.dtype)
    # ONNX Runtime's max mode takes the largest weighted corner term of any sample.
    return xp.stack(terms).max(axis=(0, 3, 5))


# ---- rearrangement ---------------------------------------------------------------------------

def center_crop_pad(x, shape, /, *, axes=None):
    rank = x.ndim
    axes = [a % rank for a in axes] if axes is not None else list(range(rank))
    kept = [slice(None)] * rank
    padding = [(0, 0)] * rank
    for axis, target in zip(axes, _rt.ints(shape, "CenterCropPad", "its target shape")):
        diff = target - x.shape[axis]
        if diff >= 0:
            padding[axis] = (diff // 2, diff - diff // 2)
        else:
            begin = (-diff) // 2
            kept[axis] = slice(begin, begin + target)
    y = x[tuple(kept)]
    return _rt.xp(y).pad(y, padding) if any(p != (0, 0) for p in padding) else y


def depth_to_space(x, /, *, blocksize, mode="DCR"):
    n, c, h, w = x.shape
    b = blocksize
    xp = _rt.xp(x)
    if mode == "DCR":
        y = xp.transpose(x.reshape(n, b, b, c // (b * b), h, w), (0, 3, 4, 1, 5, 2))
    else:
        y = xp.transpose(x.reshape(n, c // (b * b), b, b, h, w), (0, 1, 4, 2, 5, 3))
    return y.reshape(n, c // (b * b), h * b, w * b)


def space_to_depth(x, /, *, blocksize):
    n, c, h, w = x.shape
    b = blocksize
    y = _rt.xp(x).transpose(x.reshape(n, c, h // b, b, w // b, b), (0, 3, 5, 1, 2, 4))
    return y.reshape(n, c * b * b, h // b, w // b)


def col2im(data, image_shape, block_shape, /, *, dilations=None, pads=None, strides=None):
    image = _rt.ints(image_shape, "Col2Im", "its image shape")
    block = _rt.ints(block_shape, "Col2Im", "its block shape")
    n = len(image)
    dilations = list(dilations) if dilations else [1] * n
    strides = list(strides) if strides else [1] * n
    pads = list(pads) if pads else [0] * (2 * n)
    padded = [image[i] + pads[i] + pads[i + n] for i in range(n)]
    counts = [(padded[i] - dilations[i] * (block[i] - 1) - 1) // strides[i] + 1 for i in range(n)]
    # The flat position in the padded image of every (kernel offset, block) pair, both row-major.
    index = np.zeros([1] * (2 * n), dtype=np.int64)
    stride = 1
    for i in reversed(range(n)):
        shape = [1] * (2 * n)
        shape[i] = block[i]
        kernel_pos = (np.arange(block[i]) * dilations[i]).reshape(shape)
        shape = [1] * (2 * n)
        shape[n + i] = counts[i]
        block_pos = (np.arange(counts[i]) * strides[i]).reshape(shape)
        index = index + (kernel_pos + block_pos) * stride
        stride *= padded[i]
    batch = data.shape[0]
    channels = data.shape[1] // math.prod(block)
    values = data.reshape(batch, channels, -1)
    flat = index.reshape(-1)
    if _rt.concrete(data):
        out = np.zeros((batch, channels, math.prod(padded)), dtype=data.dtype)
        np.add.at(out, (slice(None), slice(None), flat), values)
    else:
        out = jnp.zeros((batch, channels, math.prod(padded)), dtype=data.dtype).at[:, :, flat].add(values)
    out = out.reshape([batch, channels] + padded)
    kept = tuple(slice(pads[i], pads[i] + image[i]) for i in range(n))
    return out[(slice(None), slice(None)) + kept]
