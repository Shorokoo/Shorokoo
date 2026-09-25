"""ONNX signal processing: DFT, STFT, the Hann, Hamming and Blackman windows, MelWeightMatrix.

A complex signal is ONNX's: a real tensor whose last axis holds (real, imaginary) -- or a single
real part. The transforms go through the FFT of numpy or jax.numpy on the complex view, which JAX
differentiates. The windows and the mel matrix are made from scalar sizes and parameters, which
must be concrete, so they are computed on the host and carry no gradient.
"""

import math

import jax
import numpy as np

from . import runtime as _rt


def _complex(x):
    """x ([..., 1|2]) as a complex array without its last axis."""
    xp = _rt.xp(x)
    real = x[..., 0]
    imaginary = xp.zeros_like(real) if x.shape[-1] == 1 else x[..., 1]
    if xp is np:
        return (real + 1j * imaginary).astype(np.complex128 if real.dtype == np.float64 else np.complex64)
    return jax.lax.complex(real, imaginary)


def _real_pair(z, dtype):
    xp = _rt.xp(z)
    return xp.stack([xp.real(z), xp.imag(z)], axis=-1).astype(dtype)


def _computed(x):
    """The dtype a transform of x is computed in: there are no half-precision FFTs."""
    return np.float64 if x.dtype == np.float64 else np.float32


def dft(x, dft_length=None, axis_input=None, /, *, axis=None, inverse=0, onesided=0, _opset):
    rank = x.ndim
    if _opset >= 20:
        dim = -2 if axis_input is None else int(_rt.number(axis_input, "DFT", "its axis"))
    else:
        dim = 1 if axis is None else axis
    dim = dim % rank
    if dim == rank - 1:
        raise ValueError("DFT's axis cannot be the last, which holds the complex components")
    signal = _complex(x.astype(_computed(x)))
    length = signal.shape[dim] if dft_length is None else int(_rt.number(dft_length, "DFT", "its length"))
    fft = _rt.xp(x).fft
    z = fft.ifft(signal, n=length, axis=dim) if inverse else fft.fft(signal, n=length, axis=dim)
    if onesided:
        kept = [slice(None)] * z.ndim
        kept[dim] = slice(0, length // 2 + 1)
        z = z[tuple(kept)]
    return _real_pair(z, x.dtype)


def stft(signal, frame_step, window=None, frame_length=None, /, *, onesided=1):
    step = int(_rt.number(frame_step, "STFT", "its frame step"))
    if window is not None:
        length = window.shape[0]
    elif frame_length is not None:
        length = int(_rt.number(frame_length, "STFT", "its frame length"))
    else:
        raise ValueError("STFT needs a window or a frame_length")
    x = _complex(signal.astype(_computed(signal)))
    count = (x.shape[1] - length) // step + 1
    frames = x[:, np.arange(count)[:, None] * step + np.arange(length)[None, :]]
    xp = _rt.xp(x, window)
    if window is not None:
        frames = frames * window.astype(_computed(signal))
    z = xp.fft.fft(frames, axis=-1)
    if onesided:
        z = z[..., :length // 2 + 1]
    return _real_pair(z, signal.dtype)


def _cosine_window(operator, size, periodic, output_datatype, coefficients):
    n = int(_rt.number(size, operator, "its size"))
    denominator = n if periodic else n - 1
    values = []
    for i in range(n):
        angle = 2 * math.pi * i / denominator if denominator else 0.0
        values.append(sum(
            (-1) ** k * a * math.cos(k * angle) for k, a in enumerate(coefficients)))
    return np.array(values, dtype=np.float64).astype(_rt.jax_dtype(output_datatype))


def hann_window(size, /, *, output_datatype=1, periodic=1):
    return _cosine_window("HannWindow", size, periodic, output_datatype, (0.5, 0.5))


def hamming_window(size, /, *, output_datatype=1, periodic=1):
    return _cosine_window("HammingWindow", size, periodic, output_datatype, (25 / 46, 1 - 25 / 46))


def blackman_window(size, /, *, output_datatype=1, periodic=1):
    return _cosine_window("BlackmanWindow", size, periodic, output_datatype, (0.42, 0.5, 0.08))


def mel_weight_matrix(num_mel_bins, dft_length, sample_rate, lower_edge_hertz, upper_edge_hertz, /, *,
                      output_datatype=1):
    def parameter(value, what):
        return _rt.number(value, "MelWeightMatrix", what)

    bins = int(parameter(num_mel_bins, "its number of mel bins"))
    length = int(parameter(dft_length, "its DFT length"))
    rate = float(parameter(sample_rate, "its sample rate"))
    spectrogram_bins = length // 2 + 1

    def to_mel(hz):
        return 2595 * math.log10(1 + hz / 700)

    low = to_mel(float(parameter(lower_edge_hertz, "its lower edge")))
    step = (to_mel(float(parameter(upper_edge_hertz, "its upper edge"))) - low) / (bins + 2)
    points = [
        int(math.floor((length + 1) * (700 * (10 ** ((low + step * i) / 2595) - 1)) / rate))
        for i in range(bins + 2)]

    target = _rt.jax_dtype(output_datatype)
    integral = _rt.is_integral(target)
    matrix = np.zeros((spectrogram_bins, bins), dtype=np.float64)

    def ratio(numerator, denominator):
        return numerator // denominator if integral else numerator / denominator

    for m in range(bins):
        left, center, right = points[m], points[m + 1], points[m + 2]
        if center == left:
            if center < spectrogram_bins:
                matrix[center, m] = 1.0
        else:
            for j in range(left, min(center + 1, spectrogram_bins)):
                matrix[j, m] = ratio(j - left, center - left)
        if right > center:
            for j in range(center, min(right, spectrogram_bins)):
                matrix[j, m] = ratio(right - j, right - center)
    return matrix.astype(target)
