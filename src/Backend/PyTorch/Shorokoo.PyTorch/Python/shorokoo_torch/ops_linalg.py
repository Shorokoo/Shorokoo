"""ONNX matrix products and linear algebra: MatMul, Gemm, MatMulInteger, Einsum and Det."""

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


def einsum(*inputs, equation):
    equation = equation.replace(" ", "")
    if _integral(inputs[0]):
        return torch.einsum(equation, *[x.to(torch.int64) for x in inputs]).to(inputs[0].dtype)
    return torch.einsum(equation, *inputs)


def det(x):
    return torch.linalg.det(x)


def matmul_integer(a, b, a_zero_point=None, b_zero_point=None, /):
    a = a.to(torch.int64)
    b = b.to(torch.int64)
    if a_zero_point is not None:
        zero = a_zero_point.to(torch.int64)
        # A 1-D zero point of A holds one value per row.
        if zero.dim() == 1 and zero.numel() > 1 and a.dim() >= 2:
            zero = zero.unsqueeze(-1)
        a = a - zero
    if b_zero_point is not None:
        b = b - b_zero_point.to(torch.int64)
    return torch.matmul(a, b).to(torch.int32)
