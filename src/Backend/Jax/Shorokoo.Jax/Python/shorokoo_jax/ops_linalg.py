"""ONNX matrix products and linear algebra: MatMul, Gemm, MatMulInteger, Einsum and Det.

A floating-point product is computed in the operands' full precision (`runtime.PRECISION`). An
integer product is exact, wrapping as the element type would: it is computed in int64, which XLA
multiplies and sums exactly and wraps as the narrower types do once cast back.
"""

import jax.numpy as jnp
import numpy as np

from . import runtime as _rt
from .ops_elementwise import add as _add


def _product(fn, operands):
    """`fn` -- a matmul or an einsum -- of `operands`, exact for integers."""
    if _rt.is_integral(operands[0]):
        wide = [x.astype(np.int64) for x in operands]
        return fn(*wide).astype(operands[0].dtype)
    return fn(*operands)


def matmul(a, b):
    if _rt.xp(a, b) is np:
        return _product(np.matmul, [a, b])
    return _product(lambda x, y: jnp.matmul(x, y, precision=_rt.PRECISION), [a, b])


def gemm(a, b, c=None, /, *, alpha=1.0, beta=1.0, transA=0, transB=0):
    a = a.T if transA else a
    b = b.T if transB else b
    product = matmul(a, b)
    if alpha != 1.0:
        product = _scaled(product, alpha)
    if c is None:
        return product
    addend = c if beta == 1.0 else _scaled(c, beta)
    return _add(product, addend.astype(product.dtype))


def _scaled(x, factor):
    """x times a float attribute, in x's type: an integer one scaled in floating point and truncated,
    as torch scales it, rather than by the factor truncated first."""
    work = np.float64 if _rt.is_integral(x) else x.dtype
    return (x.astype(work) * np.asarray(factor, dtype=work)).astype(x.dtype)


def einsum(*inputs, equation):
    equation = equation.replace(" ", "")
    if _rt.xp(*inputs) is np:
        return _product(lambda *xs: np.einsum(equation, *xs), list(inputs))
    return _product(lambda *xs: jnp.einsum(equation, *xs, precision=_rt.PRECISION), list(inputs))


def det(x):
    if x.dtype in (np.float16, _rt.jax_dtype(16)):
        # Neither is factorized; float32 holds both exactly.
        return jnp.linalg.det(x.astype(np.float32)).astype(x.dtype)
    return jnp.linalg.det(x)


def matmul_integer(a, b, a_zero_point=None, b_zero_point=None, /):
    xp = _rt.xp(a, b, a_zero_point, b_zero_point)
    a = a.astype(np.int64)
    b = b.astype(np.int64)
    if a_zero_point is not None:
        zero = a_zero_point.astype(np.int64)
        # A 1-D zero point of A holds one value per row.
        if zero.ndim == 1 and zero.size > 1 and a.ndim >= 2:
            zero = zero[:, None]
        a = a - zero
    if b_zero_point is not None:
        b = b - b_zero_point.astype(np.int64)
    return xp.matmul(a, b).astype(np.int32)
