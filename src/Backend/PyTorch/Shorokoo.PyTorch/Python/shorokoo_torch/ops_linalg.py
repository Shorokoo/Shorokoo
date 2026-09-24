"""ONNX matrix products and linear algebra: MatMul, Gemm, MatMulInteger, Einsum and Det.

An integer product is exact, wrapping as the element type would. torch's integer matmul kernels
cover int64 on the CPU only; on CUDA a product goes through float64 where that holds every partial
sum exactly, and back through the CPU where it cannot (see `_integer_route`).
"""

import math

import torch

from .ops_elementwise import add as _add

# Every integer up to 2**53 is a float64.
_EXACT = 2 ** 53


def _integral(x):
    return not (x.dtype.is_floating_point or x.dtype.is_complex)


def _largest(dtype, shifted=False):
    """The largest magnitude a value of the integer `dtype` has -- or, where `shifted`, a value less
    a zero point of the same type."""
    info = torch.iinfo(dtype)
    return info.max - info.min if shifted else max(info.max, -info.min)


def _integer_route(largest, terms, device_type):
    """Where an integer product of factors at most `largest` in magnitude (one bound per operand),
    `terms` of them summed into each element, is computed: 'int64' in torch's integer kernels,
    which the CPU has; 'float64' on CUDA, where every partial sum is an integer float64 holds
    exactly; 'cpu' on CUDA where one might not be."""
    if device_type != "cuda":
        return "int64"
    return "float64" if math.prod(largest) * terms <= _EXACT else "cpu"


def _integer_product(fn, operands, largest, terms):
    """`fn` -- a matmul or an einsum -- of integer operands, as int64, exactly."""
    device = operands[0].device
    route = _integer_route(largest, terms, device.type)
    if route == "float64":
        return fn(*[x.to(torch.float64) for x in operands]).to(torch.int64)
    if route == "cpu":
        return fn(*[x.cpu().to(torch.int64) for x in operands]).to(device)
    return fn(*[x.to(torch.int64) for x in operands])


def _summed_terms(equation, shapes):
    """How many products an einsum sums into each element of its output: the product of the
    extents of the letters its inputs name and its output does not."""
    equation = equation.replace(" ", "")
    inputs, arrow, output = equation.partition("->")
    if not arrow:
        letters = [c for c in inputs if c.isalpha()]
        output = "".join(c for c in letters if letters.count(c) == 1)
    extents = {}
    for spec, shape in zip(inputs.split(","), shapes):
        head, _, tail = spec.partition("...")
        named = list(zip(head, shape)) + list(zip(tail, shape[len(shape) - len(tail):]))
        for letter, extent in named:
            extents[letter] = max(extents.get(letter, 1), extent)
    return math.prod(n for letter, n in extents.items() if letter not in output)


def matmul(a, b):
    if _integral(a):
        # The narrower and unsigned types go through int64 too, which wraps exactly as they would.
        largest = [_largest(a.dtype), _largest(b.dtype)]
        return _integer_product(torch.matmul, [a, b], largest, a.shape[-1]).to(a.dtype)
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
    return _add(product, addend.to(product.dtype))


def einsum(*inputs, equation):
    equation = equation.replace(" ", "")
    if _integral(inputs[0]):
        terms = _summed_terms(equation, [list(x.shape) for x in inputs])
        largest = [_largest(x.dtype) for x in inputs]
        return _integer_product(lambda *xs: torch.einsum(equation, *xs), list(inputs), largest, terms).to(inputs[0].dtype)
    return torch.einsum(equation, *inputs)


def det(x):
    if x.dtype in (torch.float16, torch.bfloat16):
        # torch factorizes neither; float32 holds both exactly.
        return torch.linalg.det(x.to(torch.float32)).to(x.dtype)
    return torch.linalg.det(x)


def matmul_integer(a, b, a_zero_point=None, b_zero_point=None, /):
    largest = [_largest(a.dtype, a_zero_point is not None), _largest(b.dtype, b_zero_point is not None)]
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
    return _integer_product(torch.matmul, [a, b], largest, a.shape[-1]).to(torch.int32)
