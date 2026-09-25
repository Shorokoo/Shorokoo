"""ONNX image and geometry: Resize (and the Upsample it replaced), GridSample, AffineGrid, RoiAlign,
NonMaxSuppression, CenterCropPad, Col2Im, DepthToSpace and SpaceToDepth, and ImageDecoder.

Resize runs one axis at a time: a nearest resize gathers along the axis, and a linear or cubic one
multiplies by an [output, input] matrix of interpolation weights, which covers antialiasing,
exclude_outside and edge clamping alike and lets autograd differentiate it. Output coordinates are
mapped to input coordinates in float32, as ONNX Runtime maps them, so that a nearest resize rounds
the same way.
"""

import io
import math

import numpy as np
import torch
import torch.nn.functional as F

from . import runtime as _rt


def _ints(tensor):
    return [int(v) for v in tensor.reshape(-1).tolist()]


def _floats(tensor):
    return [float(v) for v in tensor.reshape(-1).tolist()]


def _present(tensor):
    return tensor is not None and tensor.numel() > 0


# ---- Resize ----------------------------------------------------------------------------------

def _f32(value):
    return torch.tensor(value, dtype=torch.float32)


def _original(out_size, in_size, scale, mode, roi_start, roi_end):
    """The input coordinate (float32) of every output position along one axis."""
    x = torch.arange(out_size, dtype=torch.float32)
    s = _f32(scale)
    if mode == "asymmetric":
        return x / s
    if mode == "half_pixel":
        return (x + 0.5) / s - 0.5
    if mode == "pytorch_half_pixel":
        return (x + 0.5) / s - 0.5 if out_size > 1 else torch.zeros_like(x)
    if mode == "align_corners":
        return x * _f32(in_size - 1) / _f32(out_size - 1) if out_size > 1 else torch.zeros_like(x)
    if mode == "half_pixel_symmetric":
        # The offset in float32, the rest in float64, as ONNX Runtime writes it.
        adjustment = _f32(out_size) / (s * _f32(in_size))
        offset = _f32(in_size) / 2 * (1 - adjustment)
        return (offset.double() + (x.double() + 0.5) / s.double() - 0.5).float()
    if mode == "tf_crop_and_resize":
        start, end = _f32(roi_start), _f32(roi_end)
        span = _f32(in_size - 1)
        if out_size > 1:
            return start * span + x * (end - start) * span / _f32(out_size - 1)
        return torch.full_like(x, float(0.5 * (start + end) * span))
    raise NotImplementedError(f"Resize coordinate_transformation_mode '{mode}'")


def _round_half_away(x):
    whole = torch.trunc(x)
    return whole + torch.where(torch.abs(x - whole) >= 0.5, torch.sign(x), torch.zeros_like(x))


def _nearest(x, rule, downsampling):
    if rule == "round_prefer_floor":
        return torch.where(x - torch.floor(x) == 0.5, torch.floor(x), _round_half_away(x))
    if rule == "round_prefer_ceil":
        return _round_half_away(x)
    if rule == "floor":
        return torch.floor(x)
    if rule == "ceil":
        return torch.ceil(x)
    if rule == "simple":
        return torch.ceil(x) if downsampling else torch.trunc(x)
    raise NotImplementedError(f"Resize nearest_mode '{rule}'")


def _cubic(t, a):
    t = torch.abs(t)
    near = ((a + 2) * t - (a + 3)) * t * t + 1
    far = ((a * t - 5 * a) * t + 8 * a) * t - 4 * a
    return torch.where(t <= 1, near, torch.where(t < 2, far, torch.zeros_like(t)))


def _linear(t):
    return torch.clamp(1 - torch.abs(t), min=0)


def _weights(coords, in_size, mode, scale, antialias, exclude_outside, a):
    """The [output, input] interpolation matrix (float64) of one axis."""
    x = coords.to(torch.float64)
    if mode == "linear" and not antialias:
        x = torch.clamp(x, 0, in_size - 1)
    support = 1.0 if mode == "linear" else 2.0
    stretch = 1.0
    if antialias and scale < 1:
        stretch = float(scale)
        support /= stretch
    reach = int(math.ceil(support)) + 1
    base = torch.floor(x)
    taps = base.unsqueeze(1) + torch.arange(-reach, reach + 1, dtype=torch.float64)
    distance = (taps - x.unsqueeze(1)) * stretch
    w = _linear(distance) if mode == "linear" else _cubic(distance, a)
    inside = (taps >= 0) & (taps <= in_size - 1)
    if exclude_outside:
        w = torch.where(inside, w, torch.zeros_like(w))
    total = w.sum(dim=1, keepdim=True)
    if exclude_outside or antialias:
        w = w / torch.where(total == 0, torch.ones_like(total), total)
    index = torch.clamp(taps, 0, in_size - 1).to(torch.int64)
    matrix = torch.zeros(coords.shape[0], in_size, dtype=torch.float64)
    return matrix.scatter_add(1, index, w)


def resize(x, *inputs, antialias=0, axes=None, coordinate_transformation_mode="half_pixel",
           cubic_coeff_a=-0.75, exclude_outside=0, extrapolation_value=0.0, keep_aspect_ratio_policy="stretch",
           mode="nearest", nearest_mode="round_prefer_floor", _opset):
    if _opset < 11:
        roi, scales, sizes = None, inputs[0], None
        coordinate_transformation_mode = "asymmetric"
        nearest_mode = "simple"
    else:
        roi, scales, sizes = (list(inputs) + [None, None, None])[:3]
    rank = x.dim()
    axes = [a % rank for a in axes] if axes is not None else list(range(rank))
    in_shape = list(x.shape)
    out_shape = list(in_shape)
    axis_scale = [1.0] * rank
    if _present(scales):
        for axis, s in zip(axes, _floats(scales)):
            s = float(torch.tensor(s, dtype=torch.float32))
            axis_scale[axis] = s
            out_shape[axis] = int(float(torch.tensor(in_shape[axis], dtype=torch.float32) * _f32(s)))
    elif _present(sizes):
        wanted = _ints(sizes)
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
    roi_values = _floats(roi) if _present(roi) else None

    cropping = coordinate_transformation_mode == "tf_crop_and_resize"
    outside = None
    y = x
    if mode != "nearest":
        compute = x.dtype if x.dtype.is_floating_point else torch.float32
        y = y.to(compute)
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
            off = (coords < 0) | (coords > n_in - 1)
            shape = [1] * rank
            shape[axis] = n_out
            off = off.reshape(shape).to(x.device)
            outside = off if outside is None else outside | off
        if mode == "nearest":
            index = torch.clamp(_nearest(coords, nearest_mode, s < 1), 0, n_in - 1).to(torch.int64)
            y = torch.index_select(y, axis, index.to(x.device))
        else:
            matrix = _weights(coords, n_in, mode, s, antialias, exclude_outside, cubic_coeff_a)
            y = torch.movedim(torch.matmul(torch.movedim(y, axis, -1), matrix.t().to(y.dtype).to(x.device)), -1, axis)
    if outside is not None:
        y = torch.where(outside, torch.tensor(extrapolation_value, dtype=y.dtype, device=y.device), y)
    if y.dtype != x.dtype:
        if not x.dtype.is_floating_point:
            # Antialiasing rounds to the nearest integer and plain interpolation truncates, as ONNX
            # Runtime does, and either saturates where a cubic overshoots the type's range.
            y = _round_half_away(y) if antialias and mode != "nearest" else torch.trunc(y)
            info = torch.iinfo(x.dtype)
            y = torch.clamp(y, info.min, info.max)
        y = y.to(x.dtype)
    return y


def upsample(x, scales, /, *, mode="nearest"):
    return resize(x, scales, mode=mode, _opset=10)


# ---- sampling --------------------------------------------------------------------------------

_GRID_MODES = {"linear": "bilinear", "bilinear": "bilinear", "nearest": "nearest", "cubic": "bicubic",
               "bicubic": "bicubic"}


def grid_sample(x, grid, /, *, align_corners=0, mode="linear", padding_mode="zeros"):
    return F.grid_sample(x, grid.to(x.dtype), mode=_GRID_MODES[mode], padding_mode=padding_mode,
                         align_corners=bool(align_corners))


def affine_grid(theta, size, /, *, align_corners=0):
    # Written out rather than F.affine_grid, which puts a one-pixel axis at 0 where ONNX's
    # reference puts it at -1 under align_corners; and in ONNX Runtime's float arithmetic, so that
    # a sampler rounding a grid point to the nearest pixel sees the same point.
    dims = _ints(size)[2:]
    axes = []
    for n in dims:
        if n == 1:
            base = torch.full((1,), -1.0 if align_corners else 0.0, dtype=theta.dtype, device=theta.device)
        else:
            step = torch.tensor(2.0, dtype=theta.dtype) / (n - 1)
            base = -1 + torch.arange(n, dtype=theta.dtype) * step
            base[-1] = 1.0
            if not align_corners:
                base = base * (n - 1) / n
            base = base.to(theta.device)
        axes.append(base)
    # Grid points (x, y[, z]), x varying fastest: the reverse of the size's spatial order.
    mesh = torch.meshgrid(*axes, indexing="ij")
    points = [m.reshape(-1) for m in reversed(mesh)]
    rank = len(dims)
    rows = []
    for i in range(rank):
        row = theta[:, i, rank - 1:rank] * points[rank - 1]
        for k in reversed(range(rank - 1)):
            row = theta[:, i, k:k + 1] * points[k] + row
        rows.append(row + theta[:, i, rank:rank + 1])
    return torch.stack(rows, dim=-1).reshape([theta.shape[0]] + dims + [rank])


def roi_align(x, rois, batch_indices, /, *, coordinate_transformation_mode="half_pixel", mode="avg",
              output_height=1, output_width=1, sampling_ratio=0, spatial_scale=1.0):
    height, width = x.shape[2], x.shape[3]
    offset = 0.5 if coordinate_transformation_mode == "half_pixel" else 0.0
    results = []
    boxes = rois.to(torch.float32).tolist()
    batches = _ints(batch_indices)
    for (x1, y1, x2, y2), b in zip(boxes, batches):
        start_w = _f32(x1) * _f32(spatial_scale) - offset
        start_h = _f32(y1) * _f32(spatial_scale) - offset
        roi_w = _f32(x2) * _f32(spatial_scale) - offset - start_w
        roi_h = _f32(y2) * _f32(spatial_scale) - offset - start_h
        if offset == 0.0:
            roi_w = torch.clamp(roi_w, min=1.0)
            roi_h = torch.clamp(roi_h, min=1.0)
        bin_h = roi_h / output_height
        bin_w = roi_w / output_width
        grid_h = sampling_ratio if sampling_ratio > 0 else int(math.ceil(float(roi_h / output_height)))
        grid_w = sampling_ratio if sampling_ratio > 0 else int(math.ceil(float(roi_w / output_width)))
        count = max(grid_h * grid_w, 1)
        ys = _roi_samples(start_h, bin_h, output_height, grid_h)
        xs = _roi_samples(start_w, bin_w, output_width, grid_w)
        results.append(_roi_bin_values(x[b], ys, xs, height, width, mode, count))
    if not results:
        return torch.zeros(0, x.shape[1], output_height, output_width, dtype=x.dtype, device=x.device)
    return torch.stack(results)


def _roi_samples(start, bin_size, bins, grid):
    """The sample coordinates [bins, grid] along one axis of a region."""
    p = torch.arange(bins, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(grid, dtype=torch.float32).unsqueeze(0)
    return start + p * bin_size + (i + 0.5) * bin_size / grid


def _roi_corners(coords, size):
    """ONNX Runtime's bilinear taps of each sample: (low index, high index, low weight, high weight),
    the weights zero for a sample outside [-1, size]."""
    valid = (coords >= -1.0) & (coords <= size)
    c = torch.clamp(coords, min=0.0)
    low = torch.floor(c).to(torch.int64)
    at_end = low >= size - 1
    low = torch.where(at_end, torch.full_like(low, size - 1), low)
    high = torch.where(at_end, low, low + 1)
    c = torch.where(at_end, low.to(c.dtype), c)
    frac = c - low.to(c.dtype)
    zero = torch.zeros_like(frac)
    return low, high, torch.where(valid, 1 - frac, zero), torch.where(valid, frac, zero)


def _roi_bin_values(image, ys, xs, height, width, mode, count):
    """One region's pooled values [C, bins_h, bins_w]."""
    device = image.device
    ylo, yhi, wyl, wyh = (t.to(device) for t in _roi_corners(ys, height))
    xlo, xhi, wxl, wxh = (t.to(device) for t in _roi_corners(xs, width))
    terms = []
    for yi, wy in ((ylo, wyl), (yhi, wyh)):
        rows = image[:, yi]  # [C, BH, GH, W]
        for xi, wx in ((xlo, wxl), (xhi, wxh)):
            values = rows[:, :, :, xi]  # [C, BH, GH, BW, GW]
            weight = wy[:, :, None, None] * wx[None, None, :, :]
            terms.append(values * weight.to(image.dtype))
    if mode == "avg":
        return sum(terms).sum(dim=(2, 4)) / count
    if ys.shape[1] == 0 or xs.shape[1] == 0:
        return torch.zeros(image.shape[0], ys.shape[0], xs.shape[0], dtype=image.dtype, device=device)
    # ONNX Runtime's max mode takes the largest weighted corner term of any sample.
    return torch.stack(terms).amax(dim=(0, 3, 5))


# ---- NonMaxSuppression -----------------------------------------------------------------------

def _iou_exceeds(a, b, threshold, center_point_box):
    if center_point_box:
        ax1, ax2 = a[0] - a[2] / 2, a[0] + a[2] / 2
        ay1, ay2 = a[1] - a[3] / 2, a[1] + a[3] / 2
        bx1, bx2 = b[0] - b[2] / 2, b[0] + b[2] / 2
        by1, by2 = b[1] - b[3] / 2, b[1] + b[3] / 2
    else:
        ay1, ay2 = min(a[0], a[2]), max(a[0], a[2])
        ax1, ax2 = min(a[1], a[3]), max(a[1], a[3])
        by1, by2 = min(b[0], b[2]), max(b[0], b[2])
        bx1, bx2 = min(b[1], b[3]), max(b[1], b[3])
    area_a = (ay2 - ay1) * (ax2 - ax1)
    area_b = (by2 - by1) * (bx2 - bx1)
    if area_a <= 0 or area_b <= 0:
        return False
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter = inter_h * inter_w
    if inter <= 0:
        return False
    return inter / (area_a + area_b - inter) > threshold


def non_max_suppression(boxes, scores, max_output_boxes_per_class=None, iou_threshold=None,
                        score_threshold=None, /, *, center_point_box=0):
    limit = _ints(max_output_boxes_per_class)[0] if _present(max_output_boxes_per_class) else 0
    iou = _floats(iou_threshold)[0] if _present(iou_threshold) else 0.0
    floor = _floats(score_threshold)[0] if _present(score_threshold) else None
    all_boxes = boxes.to(torch.float32).tolist()
    all_scores = scores.to(torch.float32).tolist()
    selected = []
    if limit > 0:
        for b, per_class in enumerate(all_scores):
            for c, class_scores in enumerate(per_class):
                order = sorted((i for i, s in enumerate(class_scores) if floor is None or s > floor),
                               key=lambda i: (-class_scores[i], i))
                kept = []
                for i in order:
                    if len(kept) == limit:
                        break
                    if all(not _iou_exceeds(all_boxes[b][i], all_boxes[b][k], iou, center_point_box) for k in kept):
                        kept.append(i)
                selected += [[b, c, i] for i in kept]
    result = torch.tensor(selected, dtype=torch.int64).reshape(-1, 3)
    return result.to(_rt.device())


# ---- rearrangement ---------------------------------------------------------------------------

def center_crop_pad(x, shape, /, *, axes=None):
    rank = x.dim()
    axes = [a % rank for a in axes] if axes is not None else list(range(rank))
    begins = [0] * rank
    ends = [0] * rank
    for axis, target in zip(axes, _ints(shape)):
        diff = target - x.shape[axis]
        if diff >= 0:
            begins[axis] = diff // 2
        else:
            begins[axis] = -((-diff) // 2)
        ends[axis] = diff - begins[axis]
    flat = []
    for b, e in zip(reversed(begins), reversed(ends)):
        flat += [b, e]
    if x.dtype == torch.bool:
        return F.pad(x.to(torch.uint8), flat).to(torch.bool)
    return F.pad(x, flat)


def depth_to_space(x, /, *, blocksize, mode="DCR"):
    n, c, h, w = x.shape
    b = blocksize
    if mode == "DCR":
        y = x.reshape(n, b, b, c // (b * b), h, w).permute(0, 3, 4, 1, 5, 2)
    else:
        y = x.reshape(n, c // (b * b), b, b, h, w).permute(0, 1, 4, 2, 5, 3)
    return y.reshape(n, c // (b * b), h * b, w * b)


def space_to_depth(x, /, *, blocksize):
    n, c, h, w = x.shape
    b = blocksize
    y = x.reshape(n, c, h // b, b, w // b, b).permute(0, 3, 5, 1, 2, 4)
    return y.reshape(n, c * b * b, h // b, w // b)


def col2im(data, image_shape, block_shape, /, *, dilations=None, pads=None, strides=None):
    image = _ints(image_shape)
    block = _ints(block_shape)
    n = len(image)
    dilations = list(dilations) if dilations else [1] * n
    strides = list(strides) if strides else [1] * n
    pads = list(pads) if pads else [0] * (2 * n)
    padded = [image[i] + pads[i] + pads[i + n] for i in range(n)]
    counts = [(padded[i] - dilations[i] * (block[i] - 1) - 1) // strides[i] + 1 for i in range(n)]
    device = data.device
    # The flat position in the padded image of every (kernel offset, block) pair, both row-major.
    index = torch.zeros([1] * (2 * n), dtype=torch.int64, device=device)
    stride = 1
    for i in reversed(range(n)):
        shape = [1] * (2 * n)
        shape[i] = block[i]
        kernel_pos = (torch.arange(block[i], device=device) * dilations[i]).reshape(shape)
        shape = [1] * (2 * n)
        shape[n + i] = counts[i]
        block_pos = (torch.arange(counts[i], device=device) * strides[i]).reshape(shape)
        index = index + (kernel_pos + block_pos) * stride
        stride *= padded[i]
    batch = data.shape[0]
    channels = data.shape[1] // math.prod(block)
    out = torch.zeros(batch, channels, math.prod(padded), dtype=data.dtype, device=device)
    out = out.index_add(2, index.reshape(-1), data.reshape(batch, channels, -1))
    out = out.reshape([batch, channels] + padded)
    crop = []
    for i in reversed(range(n)):
        crop += [-pads[i], -pads[i + n]]
    return F.pad(out, crop) if any(crop) else out


# ---- ImageDecoder ----------------------------------------------------------------------------

def image_decoder(encoded, /, *, pixel_format="RGB"):
    """Decodes with Pillow, as the ONNX reference implementation does, so every format the spec
    names (BMP, JPEG, JPEG 2000, TIFF, PNG, WebP, the portable any-maps) is read. An image of any
    other mode (palette, alpha, greyscale, CMYK) is converted to the requested pixel format."""
    try:
        from PIL import Image
    except ImportError as e:
        raise RuntimeError("ImageDecoder needs Pillow in the Python environment") from e
    with Image.open(io.BytesIO(encoded.detach().cpu().numpy().tobytes())) as image:
        pixels = np.asarray(image.convert("L" if pixel_format == "Grayscale" else "RGB"))
    if pixel_format == "Grayscale":
        pixels = pixels[:, :, None]
    elif pixel_format == "BGR":
        pixels = pixels[:, :, ::-1]
    elif pixel_format != "RGB":
        raise NotImplementedError(f"ImageDecoder pixel_format '{pixel_format}'")
    return torch.from_numpy(np.ascontiguousarray(pixels)).to(_rt.device())
