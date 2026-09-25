"""ONNX comparisons, logic and bitwise operators, and Where, over numpy and jax arrays.

numpy and jax.numpy compare and shift every integer type, the unsigned 64-bit one included, as
ONNX does, so nothing here is emulated.
"""

from . import runtime as _rt


def equal(a, b):
    return _rt.xp(a, b).equal(a, b)


def greater(a, b):
    return _rt.xp(a, b).greater(a, b)


def greater_or_equal(a, b):
    return _rt.xp(a, b).greater_equal(a, b)


def less(a, b):
    return _rt.xp(a, b).less(a, b)


def less_or_equal(a, b):
    return _rt.xp(a, b).less_equal(a, b)


def and_(a, b):
    return _rt.xp(a, b).logical_and(a, b)


def or_(a, b):
    return _rt.xp(a, b).logical_or(a, b)


def xor(a, b):
    return _rt.xp(a, b).logical_xor(a, b)


def not_(x):
    return _rt.xp(x).logical_not(x)


def is_nan(x):
    return _rt.xp(x).isnan(x)


def is_inf(x, *, detect_negative=1, detect_positive=1):
    xp = _rt.xp(x)
    if detect_negative and detect_positive:
        return xp.isinf(x)
    if detect_negative:
        return xp.isneginf(x)
    if detect_positive:
        return xp.isposinf(x)
    return xp.zeros(x.shape, dtype=bool)


def where(condition, x, y):
    return _rt.xp(condition, x, y).where(condition, x, y)


def bitwise_and(a, b):
    return _rt.xp(a, b).bitwise_and(a, b)


def bitwise_or(a, b):
    return _rt.xp(a, b).bitwise_or(a, b)


def bitwise_xor(a, b):
    return _rt.xp(a, b).bitwise_xor(a, b)


def bitwise_not(x):
    return _rt.xp(x).invert(x)


def bit_shift(x, y, *, direction):
    # ONNX shifts unsigned integers only, so a right shift is a logical one.
    xp = _rt.xp(x, y)
    return (xp.left_shift(x, y) if direction == "LEFT" else xp.right_shift(x, y)).astype(x.dtype)
