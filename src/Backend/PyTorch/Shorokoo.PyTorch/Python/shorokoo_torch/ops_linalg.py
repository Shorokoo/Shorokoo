"""ONNX matrix products: MatMul and Gemm."""

import torch

from .ops_elementwise import _NARROW_UNSIGNED


def _integral(x):
    return not (x.dtype.is_floating_point or x.dtype.is_complex)


def matmul(a, b):
    if _integral(a):
        # torch's integer matmul kernels cover the signed types; the rest go through int64,
        # which wraps exactly as the narrower type would.
        return torch.matmul(a.to(torch.int64), b.to(torch.int64)).to(a.dtype)
    return torch.matmul(a, b)


def gemm(a, b, c=None, /, *, alpha=1.0, beta=1.0, transA=0, transB=0):
    a = a.transpose(0, 1) if transA else a
    b = b.transpose(0, 1) if transB else b
    product = matmul(a, b)
    if alpha != 1.0:
        product = (product * alpha).to(product.dtype)
    if c is None:
        return product
    addend = c if beta == 1.0 else (c * beta).to(c.dtype)
    return product + addend
