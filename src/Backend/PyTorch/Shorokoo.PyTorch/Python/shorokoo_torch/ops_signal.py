"""ONNX signal processing: DFT, STFT, the Hann, Hamming and Blackman windows, MelWeightMatrix.

A complex signal is ONNX's: a real tensor whose last axis holds (real, imaginary) -- or a single
real part. The transforms go through torch.fft on the complex view, which autograd
differentiates. The windows and the mel matrix are made from scalar sizes and parameters, so they
are computed on the host from plain numbers and carry no gradient.
"""

import math

import torch

from . import runtime as _rt


def _int(value):
    return int(value.reshape(-1)[0].item())


def _float(value):
    return float(value.reshape(-1)[0].item())


def _complex(x):
    """x ([..., 1|2]) as a complex tensor without its last axis."""
    if x.shape[-1] == 1:
        return torch.complex(x[..., 0], torch.zeros_like(x[..., 0]))
    return torch.complex(x[..., 0], x[..., 1])


def _real_pair(z, dtype):
    return torch.view_as_real(z).to(dtype)


def _computed(x):
    """The dtype a transform of x is computed in: torch.fft has no half-precision kernels on CPU."""
    return torch.float64 if x.dtype == torch.float64 else torch.float32


def dft(x, dft_length=None, axis_input=None, /, *, axis=None, inverse=0, onesided=0, _opset):
    rank = x.dim()
    if _opset >= 20:
        dim = -2 if axis_input is None else _int(axis_input)
    else:
        dim = 1 if axis is None else axis
    dim = dim % rank
    if dim == rank - 1:
        raise ValueError("DFT's axis cannot be the last, which holds the complex components")
    signal = _complex(x.to(_computed(x)))
    length = signal.shape[dim] if dft_length is None else _int(dft_length)
    if inverse:
        z = torch.fft.ifft(signal, n=length, dim=dim)
    else:
        z = torch.fft.fft(signal, n=length, dim=dim)
    if onesided:
        z = z.narrow(dim, 0, length // 2 + 1)
    return _real_pair(z, x.dtype)


def stft(signal, frame_step, window=None, frame_length=None, /, *, onesided=1):
    step = _int(frame_step)
    if window is not None:
        length = window.shape[0]
    elif frame_length is not None:
        length = _int(frame_length)
    else:
        raise ValueError("STFT needs a window or a frame_length")
    x = _complex(signal.to(_computed(signal)))
    frames = x.unfold(1, length, step)
    if window is not None:
        frames = frames * window.to(frames.real.dtype)
    z = torch.fft.fft(frames, dim=-1)
    if onesided:
        z = z[..., :length // 2 + 1]
    return _real_pair(z, signal.dtype)


def _cosine_window(size, periodic, output_datatype, coefficients):
    n = _int(size)
    denominator = n if periodic else n - 1
    values = []
    for i in range(n):
        angle = 2 * math.pi * i / denominator if denominator else 0.0
        values.append(sum(
            (-1) ** k * a * math.cos(k * angle) for k, a in enumerate(coefficients)))
    window = torch.tensor(values, dtype=torch.float64, device=_rt.device())
    return window.to(_rt.torch_dtype(output_datatype))


def hann_window(size, /, *, output_datatype=1, periodic=1):
    return _cosine_window(size, periodic, output_datatype, (0.5, 0.5))


def hamming_window(size, /, *, output_datatype=1, periodic=1):
    return _cosine_window(size, periodic, output_datatype, (25 / 46, 1 - 25 / 46))


def blackman_window(size, /, *, output_datatype=1, periodic=1):
    return _cosine_window(size, periodic, output_datatype, (0.42, 0.5, 0.08))


def mel_weight_matrix(num_mel_bins, dft_length, sample_rate, lower_edge_hertz, upper_edge_hertz, /, *,
                      output_datatype=1):
    bins = _int(num_mel_bins)
    length = _int(dft_length)
    rate = _float(sample_rate)
    spectrogram_bins = length // 2 + 1

    def to_mel(hz):
        return 2595 * math.log10(1 + hz / 700)

    low = to_mel(_float(lower_edge_hertz))
    step = (to_mel(_float(upper_edge_hertz)) - low) / (bins + 2)
    points = [
        int(math.floor((length + 1) * (700 * (10 ** ((low + step * i) / 2595) - 1)) / rate))
        for i in range(bins + 2)]

    target = _rt.torch_dtype(output_datatype)
    integral = not (target.is_floating_point or target.is_complex)
    matrix = [[0.0] * bins for _ in range(spectrogram_bins)]

    def ratio(numerator, denominator):
        return numerator // denominator if integral else numerator / denominator

    for m in range(bins):
        left, center, right = points[m], points[m + 1], points[m + 2]
        if center == left:
            if center < spectrogram_bins:
                matrix[center][m] = 1.0
        else:
            for j in range(left, min(center + 1, spectrogram_bins)):
                matrix[j][m] = ratio(j - left, center - left)
        if right > center:
            for j in range(center, min(right, spectrogram_bins)):
                matrix[j][m] = ratio(right - j, right - center)
    return torch.tensor(matrix, dtype=torch.float64, device=_rt.device()).reshape(spectrogram_bins, bins).to(target)
