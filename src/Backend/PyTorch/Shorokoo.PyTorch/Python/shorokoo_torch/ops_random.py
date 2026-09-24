"""ONNX random draws: RandomNormal, RandomUniform, their Like forms, Bernoulli, Multinomial, and
Dropout.

The values come from torch's generator, so they cannot match another runtime's draw for draw;
what the spec fixes is kept: the distribution, the shape and element type, and that a node with a
`seed` draws the same values every time -- and as every other node with that seed and the same
parameters. A node without one draws from torch's default generator.
"""

import struct

import torch

from . import runtime as _rt

_FLOATS = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def _generator(seed, device):
    """A generator seeded by the float `seed`'s bit pattern, or None for the default one."""
    if seed is None:
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(struct.unpack("<I", struct.pack("<f", seed))[0])
    return generator


def _draw(fill, shape, dtype, seed, device):
    """`fill` (normal or uniform) of `shape`, drawn in `dtype` where torch draws it and in float32
    then converted where it does not."""
    draw_dtype = dtype if dtype in _FLOATS else torch.float32
    values = torch.empty(tuple(shape), dtype=draw_dtype, device=device)
    return fill(values, _generator(seed, device)).to(dtype)


def _like(x, dtype):
    return x.dtype if dtype is None else _rt.torch_dtype(dtype)


def random_normal(*, shape, dtype=1, mean=0.0, scale=1.0, seed=None):
    return _draw(lambda v, g: v.normal_(mean, scale, generator=g), shape, _rt.torch_dtype(dtype), seed, _rt.device())


def random_uniform(*, shape, dtype=1, high=1.0, low=0.0, seed=None):
    return _draw(lambda v, g: v.uniform_(low, high, generator=g), shape, _rt.torch_dtype(dtype), seed, _rt.device())


def random_normal_like(x, /, *, dtype=None, mean=0.0, scale=1.0, seed=None):
    return _draw(lambda v, g: v.normal_(mean, scale, generator=g), x.shape, _like(x, dtype), seed, x.device)


def random_uniform_like(x, /, *, dtype=None, high=1.0, low=0.0, seed=None):
    return _draw(lambda v, g: v.uniform_(low, high, generator=g), x.shape, _like(x, dtype), seed, x.device)


def bernoulli(x, /, *, dtype=None, seed=None):
    """Bernoulli: 1 with probability `x`, element by element, of `dtype` or `x`'s."""
    uniform = _draw(lambda v, g: v.uniform_(0.0, 1.0, generator=g), x.shape, torch.float64, seed, x.device)
    return (uniform < x.to(torch.float64)).to(_like(x, dtype))


def multinomial(x, /, *, dtype=6, sample_size=1, seed=None):
    """Multinomial: `sample_size` class indices per row of the [batch, class] unnormalized
    log-probabilities `x`, drawn with replacement."""
    probabilities = torch.softmax(x.to(torch.float64), -1)
    samples = torch.multinomial(probabilities, sample_size, replacement=True, generator=_generator(seed, x.device))
    return samples.to(_rt.torch_dtype(dtype))


def dropout(data, ratio_in=None, training_mode=None, /, *, seed=None, ratio=None, _outputs):
    """Dropout, returning its output and its mask. Outside training -- no `training_mode`, or a
    false one -- or at a ratio of 0 it is the identity and the mask is all true; in training each
    element is kept with probability 1 - ratio, and the kept ones are scaled by 1 / (1 - ratio).
    `ratio` is an input from opset 12, an attribute before it."""
    rate = float(ratio_in.reshape(-1)[0].item()) if ratio_in is not None else (0.5 if ratio is None else ratio)
    training = training_mode is not None and bool(training_mode.reshape(-1)[0].item())
    if not training or rate == 0.0:
        output, mask = data, torch.ones(data.shape, dtype=torch.bool, device=data.device)
    else:
        uniform = _draw(lambda v, g: v.uniform_(0.0, 1.0, generator=g), data.shape, torch.float32, seed, data.device)
        mask = uniform >= rate
        output = torch.where(mask, data * (1.0 / (1.0 - rate)), torch.zeros_like(data))
    return (output, mask)[:_outputs]
