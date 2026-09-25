"""ONNX normalization and losses: BatchNormalization (inference and training mode), Instance-,
Layer-, Group-, RMS-, Lp- and MeanVarianceNormalization, LRN, NegativeLogLikelihoodLoss
and SoftmaxCrossEntropyLoss.

A 16-bit float input is normalized in float32 (for Layer-, Group- and RMSNormalization, where
stash_type asks for it, as it does by default) and rounded back. The running statistics of a
training-mode BatchNormalization come back as new values, like every other output.
"""

import jax
import jax.numpy as jnp
import numpy as np

from . import runtime as _rt

_NARROW_FLOATS = (np.dtype(np.float16), _rt.jax_dtype(16))


def _widened(x):
    """The dtype a normalization of x computes in: float32 for a 16-bit float, else x's own."""
    return np.dtype(np.float32) if np.dtype(x.dtype) in _NARROW_FLOATS else np.dtype(x.dtype)


def _stashed(x, stash_type):
    return _widened(x) if stash_type == 1 else np.dtype(x.dtype)


def _channel_shape(x):
    return [1, -1] + [1] * (x.ndim - 2)


def _constant(value, dtype):
    return np.asarray(value, dtype=dtype)


def batch_normalization(x, scale, bias, mean, var, /, *, epsilon=1e-5, momentum=0.9, training_mode=0,
                        _outputs):
    xp = _rt.xp(x, scale, bias, mean, var)
    shape = _channel_shape(x)
    compute = _widened(x)
    xs = x.astype(compute)
    if training_mode:
        dims = (0,) + tuple(range(2, x.ndim))
        batch_mean = xp.mean(xs, axis=dims)
        batch_var = xp.mean(xp.square(xs - batch_mean.reshape(shape)), axis=dims)
        use_mean, use_var = batch_mean, batch_var
    else:
        use_mean, use_var = mean.astype(compute), var.astype(compute)
    y = (xs - use_mean.reshape(shape)) / xp.sqrt(use_var.reshape(shape) + _constant(epsilon, compute))
    y = (y * scale.astype(compute).reshape(shape) + bias.astype(compute).reshape(shape)).astype(x.dtype)
    if _outputs == 1:
        return (y,)
    if not training_mode:
        raise ValueError("BatchNormalization has running-statistics outputs only in training mode")
    kept, fresh = _constant(momentum, compute), _constant(1 - momentum, compute)
    running_mean = (mean.astype(compute) * kept + batch_mean * fresh).astype(mean.dtype)
    running_var = (var.astype(compute) * kept + batch_var * fresh).astype(var.dtype)
    return (y, running_mean, running_var)[:_outputs]


def instance_normalization(x, scale, bias, /, *, epsilon=1e-5):
    xp = _rt.xp(x, scale, bias)
    compute = _widened(x)
    xs = x.astype(compute)
    dims = tuple(range(2, x.ndim))
    mean = xp.mean(xs, axis=dims, keepdims=True)
    var = xp.mean(xp.square(xs - mean), axis=dims, keepdims=True)
    shape = _channel_shape(x)
    y = ((xs - mean) / xp.sqrt(var + _constant(epsilon, compute)) * scale.astype(compute).reshape(shape)
         + bias.astype(compute).reshape(shape))
    return y.astype(x.dtype)


def layer_normalization(x, scale, bias=None, /, *, axis=-1, epsilon=1e-5, stash_type=1, _outputs):
    xp = _rt.xp(x, scale, bias)
    axis = axis % x.ndim
    compute = _stashed(x, stash_type)
    xs = x.astype(compute)
    dims = tuple(range(axis, x.ndim))
    mean = xp.mean(xs, axis=dims, keepdims=True)
    # The variance as E[(x - E[x])^2], as the function body ONNX defines computes it. The form
    # E[x^2] - E[x]^2 cancels away every digit of a variance that is small next to the square of
    # the mean, down to a negative one and a NaN.
    centered = xs - mean
    std = xp.sqrt(xp.mean(xp.square(centered), axis=dims, keepdims=True) + _constant(epsilon, compute))
    inv_std = _constant(1, compute) / std
    y = centered / std * scale.astype(compute)
    if bias is not None:
        y = y + bias.astype(compute)
    return (y.astype(x.dtype), mean, inv_std)[:_outputs]


def group_normalization(x, scale, bias, /, *, epsilon=1e-5, num_groups, stash_type=1, _opset):
    xp = _rt.xp(x, scale, bias)
    compute = _stashed(x, stash_type)
    n, c = x.shape[0], x.shape[1]
    grouped = x.astype(compute).reshape(n, num_groups, -1)
    mean = xp.mean(grouped, axis=2, keepdims=True)
    var = xp.mean(xp.square(grouped - mean), axis=2, keepdims=True)
    y = ((grouped - mean) / xp.sqrt(var + _constant(epsilon, compute))).reshape(x.shape)
    shape = _channel_shape(x)
    scale = scale.astype(compute)
    bias = bias.astype(compute)
    if _opset < 21:
        # Before opset 21 the scale and bias held one value per group.
        scale = xp.repeat(scale, c // num_groups)
        bias = xp.repeat(bias, c // num_groups)
    return (y * scale.reshape(shape) + bias.reshape(shape)).astype(x.dtype)


def rms_normalization(x, scale, /, *, axis=-1, epsilon=1e-5, stash_type=1):
    xp = _rt.xp(x, scale)
    axis = axis % x.ndim
    compute = _stashed(x, stash_type)
    xs = x.astype(compute)
    mean_square = xp.mean(xp.square(xs), axis=tuple(range(axis, x.ndim)), keepdims=True)
    root = xp.sqrt(mean_square + _constant(epsilon, compute))
    return (xs / root * scale.astype(compute)).astype(x.dtype)


def lrn(x, /, *, alpha=0.0001, beta=0.75, bias=1.0, size):
    xp = _rt.xp(x)
    squares = xp.square(x)
    before = (size - 1) // 2
    after = size - 1 - before
    padded = xp.pad(squares, [(0, 0), (before, after)] + [(0, 0)] * (x.ndim - 2))
    channels = x.shape[1]
    window = sum(padded[:, i:i + channels] for i in range(size))
    base = _constant(bias, x.dtype) + _constant(alpha / size, x.dtype) * window
    return x / xp.power(base, _constant(beta, x.dtype))


def lp_normalization(x, /, *, axis=-1, p=2):
    xp = _rt.xp(x)
    if p == 1:
        norm = xp.sum(xp.abs(x), axis=axis, keepdims=True)
    else:
        squares = xp.sum(xp.square(x), axis=axis, keepdims=True)
        # The square root only of a nonzero sum, whose gradient is finite.
        norm = xp.where(squares == 0, squares, xp.sqrt(xp.where(squares == 0, xp.ones_like(squares), squares)))
    # A zero norm leaves its zeros as they are, as ONNX Runtime's kernel does, and divides the
    # gradient by 1e-12, as Shorokoo's own rule does, which floors the norm there (by the smallest
    # normal number of a type that cannot hold 1e-12).
    floor = _constant(max(1e-12, float(jnp.finfo(x.dtype).tiny)), x.dtype)
    return x / xp.where(norm == 0, floor, norm)


def mean_variance_normalization(x, /, *, axes=None):
    xp = _rt.xp(x)
    axes = tuple(axes) if axes is not None else (0, 2, 3)
    mean = xp.mean(x, axis=axes, keepdims=True)
    variance = xp.mean(xp.square(x), axis=axes, keepdims=True) - xp.square(mean)
    return (x - mean) / (xp.sqrt(variance) + _constant(1e-9, x.dtype))


def _nll(log_prob, target, weight, ignore_index, reduction):
    """ONNX's negative log-likelihood of `target` under `log_prob` [N, C, d...]."""
    xp = _rt.xp(log_prob, target, weight)
    if ignore_index is not None:
        ignored = target == ignore_index
        safe = xp.where(ignored, xp.zeros_like(target), target)
    else:
        ignored = None
        safe = target
    safe = safe.astype(np.int64)
    picked = xp.take_along_axis(log_prob, safe[:, None], axis=1)[:, 0]
    if weight is not None:
        w = xp.take(weight, safe, axis=0).astype(log_prob.dtype)
    else:
        w = xp.ones_like(picked)
    if ignored is not None:
        w = xp.where(ignored, xp.zeros_like(w), w)
    loss = -picked * w
    if reduction == "none":
        return loss
    if reduction == "sum":
        return xp.sum(loss)
    return xp.sum(loss) / xp.sum(w)


def negative_log_likelihood_loss(input, target, weight=None, /, *, ignore_index=None, reduction="mean"):
    return _nll(input, target, weight, ignore_index, reduction)


def softmax_cross_entropy_loss(scores, labels, weights=None, /, *, ignore_index=None, reduction="mean",
                               _outputs):
    log_prob = jax.nn.log_softmax(scores, axis=1)
    loss = _nll(log_prob, labels, weights, ignore_index, reduction)
    return (loss, log_prob)[:_outputs]
