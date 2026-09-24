"""ONNX normalization and losses: BatchNormalization (inference and training mode), Instance-,
Layer-, Group-, RMS-, Lp- and MeanVarianceNormalization, LRN, NegativeLogLikelihoodLoss
and SoftmaxCrossEntropyLoss.

Every helper is functional -- no input is written to, the running statistics of a training-mode
BatchNormalization included, which come back as new tensors -- so autograd can differentiate
through all of them.
"""

import torch
import torch.nn.functional as F

_NARROW_FLOATS = (torch.float16, torch.bfloat16)


def _stashed(x, stash_type):
    """The dtype a normalization computes in: float32 for a 16-bit input when stash_type asks for
    it (the default), otherwise the input's own."""
    if stash_type == 1 and x.dtype in _NARROW_FLOATS:
        return torch.float32
    return x.dtype


def _channel_shape(x):
    return [1, -1] + [1] * (x.dim() - 2)


def batch_normalization(x, scale, bias, mean, var, /, *, epsilon=1e-5, momentum=0.9, training_mode=0,
                        _outputs):
    shape = _channel_shape(x)
    compute = torch.float32 if x.dtype in _NARROW_FLOATS else x.dtype
    xs = x.to(compute)
    if training_mode:
        dims = [0] + list(range(2, x.dim()))
        batch_mean = torch.mean(xs, dim=dims)
        batch_var = torch.mean(torch.square(xs - batch_mean.reshape(shape)), dim=dims)
        use_mean, use_var = batch_mean, batch_var
    else:
        use_mean, use_var = mean.to(compute), var.to(compute)
    y = (xs - use_mean.reshape(shape)) / torch.sqrt(use_var.reshape(shape) + epsilon)
    y = (y * scale.to(compute).reshape(shape) + bias.to(compute).reshape(shape)).to(x.dtype)
    if _outputs == 1:
        return (y,)
    if not training_mode:
        raise ValueError("BatchNormalization has running-statistics outputs only in training mode")
    running_mean = (mean.to(compute) * momentum + batch_mean * (1 - momentum)).to(mean.dtype)
    running_var = (var.to(compute) * momentum + batch_var * (1 - momentum)).to(var.dtype)
    return (y, running_mean, running_var)[:_outputs]


def instance_normalization(x, scale, bias, /, *, epsilon=1e-5):
    compute = torch.float32 if x.dtype in _NARROW_FLOATS else x.dtype
    xs = x.to(compute)
    dims = list(range(2, x.dim()))
    mean = torch.mean(xs, dim=dims, keepdim=True)
    var = torch.mean(torch.square(xs - mean), dim=dims, keepdim=True)
    shape = _channel_shape(x)
    y = (xs - mean) / torch.sqrt(var + epsilon) * scale.to(compute).reshape(shape) + bias.to(compute).reshape(shape)
    return y.to(x.dtype)


def layer_normalization(x, scale, bias=None, /, *, axis=-1, epsilon=1e-5, stash_type=1, _outputs):
    axis = axis % x.dim()
    compute = _stashed(x, stash_type)
    xs = x.to(compute)
    dims = list(range(axis, x.dim()))
    mean = torch.mean(xs, dim=dims, keepdim=True)
    # The variance as E[(x - E[x])^2], as the function body ONNX defines computes it. The form
    # E[x^2] - E[x]^2 cancels away every digit of a variance that is small next to the square of
    # the mean, down to a negative one and a NaN.
    centered = xs - mean
    std = torch.sqrt(torch.mean(torch.square(centered), dim=dims, keepdim=True) + epsilon)
    inv_std = 1 / std
    y = centered / std * scale.to(compute)
    if bias is not None:
        y = y + bias.to(compute)
    return (y.to(x.dtype), mean, inv_std)[:_outputs]


def group_normalization(x, scale, bias, /, *, epsilon=1e-5, num_groups, stash_type=1, _opset):
    compute = _stashed(x, stash_type)
    n, c = x.shape[0], x.shape[1]
    grouped = x.to(compute).reshape(n, num_groups, -1)
    mean = torch.mean(grouped, dim=2, keepdim=True)
    var = torch.mean(torch.square(grouped - mean), dim=2, keepdim=True)
    y = ((grouped - mean) / torch.sqrt(var + epsilon)).reshape(x.shape)
    shape = _channel_shape(x)
    scale = scale.to(compute)
    bias = bias.to(compute)
    if _opset < 21:
        # Before opset 21 the scale and bias held one value per group.
        scale = scale.repeat_interleave(c // num_groups)
        bias = bias.repeat_interleave(c // num_groups)
    return (y * scale.reshape(shape) + bias.reshape(shape)).to(x.dtype)


def rms_normalization(x, scale, /, *, axis=-1, epsilon=1e-5, stash_type=1):
    axis = axis % x.dim()
    compute = _stashed(x, stash_type)
    xs = x.to(compute)
    mean_square = torch.mean(torch.square(xs), dim=list(range(axis, x.dim())), keepdim=True)
    return (xs * torch.rsqrt(mean_square + epsilon) * scale.to(compute)).to(x.dtype)


def lrn(x, /, *, alpha=0.0001, beta=0.75, bias=1.0, size):
    squares = torch.square(x)
    before = (size - 1) // 2
    after = size - 1 - before
    padded = F.pad(squares.transpose(1, -1), (before, after)).transpose(1, -1)
    window = padded.unfold(1, size, 1).sum(-1)
    return x / torch.pow(bias + (alpha / size) * window, beta)


def lp_normalization(x, /, *, axis=-1, p=2):
    if p == 1:
        norm = torch.sum(torch.abs(x), dim=axis, keepdim=True)
    else:
        squares = torch.sum(torch.square(x), dim=axis, keepdim=True)
        # The square root only of a nonzero sum, whose gradient is finite.
        norm = torch.where(squares == 0, squares, torch.sqrt(torch.where(squares == 0, torch.ones_like(squares), squares)))
    # A zero norm leaves its zeros as they are, as ONNX Runtime's kernel does, and divides the
    # gradient by 1e-12, as Shorokoo's own rule does, which floors the norm there (by the smallest
    # normal number of a type that cannot hold 1e-12).
    floor = max(1e-12, torch.finfo(x.dtype).tiny)
    return x / torch.where(norm == 0, torch.full_like(norm, floor), norm)


def mean_variance_normalization(x, /, *, axes=None):
    axes = list(axes) if axes is not None else [0, 2, 3]
    mean = torch.mean(x, dim=axes, keepdim=True)
    variance = torch.mean(torch.square(x), dim=axes, keepdim=True) - torch.square(mean)
    return (x - mean) / (torch.sqrt(variance) + 1e-9)


def _nll(log_prob, target, weight, ignore_index, reduction):
    """ONNX's negative log-likelihood of `target` under `log_prob` [N, C, d...]."""
    if ignore_index is not None:
        ignored = target == ignore_index
        safe = torch.where(ignored, torch.zeros_like(target), target)
    else:
        ignored = None
        safe = target
    picked = torch.gather(log_prob, 1, safe.unsqueeze(1).to(torch.int64)).squeeze(1)
    if weight is not None:
        w = weight[safe.to(torch.int64)].to(log_prob.dtype)
    else:
        w = torch.ones_like(picked)
    if ignored is not None:
        w = torch.where(ignored, torch.zeros_like(w), w)
    loss = -picked * w
    if reduction == "none":
        return loss
    if reduction == "sum":
        return torch.sum(loss)
    return torch.sum(loss) / torch.sum(w)


def negative_log_likelihood_loss(input, target, weight=None, /, *, ignore_index=None, reduction="mean"):
    return _nll(input, target, weight, ignore_index, reduction)


def softmax_cross_entropy_loss(scores, labels, weights=None, /, *, ignore_index=None, reduction="mean",
                               _outputs):
    log_prob = torch.log_softmax(scores, dim=1)
    loss = _nll(log_prob, labels, weights, ignore_index, reduction)
    return (loss, log_prob)[:_outputs]

