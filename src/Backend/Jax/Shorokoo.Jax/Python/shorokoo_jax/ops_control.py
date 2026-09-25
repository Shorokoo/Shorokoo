"""ONNX control flow: If and Loop.

A branch or a loop body is a Python function the translator writes, nested in the function of the
graph that holds it, so it reads outer values by closure exactly as an ONNX subgraph reads its
enclosing scope. A branch takes nothing and returns its outputs as a tuple; a loop body takes the
iteration number, the condition and the loop-carried values, and returns the condition, the
carried values and then the scan outputs.

Where the condition, the trip count and every iteration's condition are known while the model is
traced -- computed from shapes and constants -- the branch is picked and the loop unrolled then, so
the body sees its iteration number as a number. Otherwise they become XLA control flow: `lax.cond`,
whose branches must agree in their outputs' shapes and types, and `lax.scan` or `lax.while_loop`,
whose carried values must keep theirs.
"""

import jax
import jax.numpy as jnp
import numpy as np

from . import runtime as _rt

# The most iterations a loop whose count is known is unrolled into: beyond it, the body is compiled
# once, as XLA control flow.
_UNROLLED = 64


def _truth(value):
    return bool(np.asarray(value).reshape(-1)[0])


def _predicate(value):
    return jnp.reshape(value, (-1,))[0].astype(bool)


def if_(condition, then_branch, else_branch):
    if _rt.concrete(condition):
        return then_branch() if _truth(condition) else else_branch()
    try:
        return jax.lax.cond(_predicate(condition), lambda: tuple(then_branch()), lambda: tuple(else_branch()))
    except TypeError as ex:
        raise _rt.DataDependentShape("If", None, (
            "An If whose condition is computed from the values of an input is compiled with both branches, "
            f"which must then produce outputs of the same shapes and types, and they do not: {ex}")) from ex


def loop(trip_count, condition, carried, body, scan_types):
    """`scan_types` is one (element type code, element dims or None) per scan output: what the body
    declares of it, which an output of a loop that runs no iteration is made empty of."""
    count = None if trip_count is None else (_rt.number(trip_count, "Loop", "") if _rt.concrete(trip_count) else trip_count)
    go = True if condition is None else (_truth(condition) if _rt.concrete(condition) else condition)
    known = not isinstance(count, jax.Array) and not isinstance(go, jax.Array)

    if known and count is not None and count <= _UNROLLED or known and count is None:
        return _unrolled(count, go, list(carried), body, scan_types)
    if known and go and count is not None:
        scanned = _scanned(count, list(carried), body, len(scan_types))
        if scanned is not None:
            return scanned
    if not scan_types:
        return _while(count, go, list(carried), body)
    raise _rt.DataDependentShape("Loop", None, (
        "A Loop with scan outputs is compiled with as many iterations as its trip count, and this one's "
        "count or condition is computed from the values of an input, so its scan outputs' length is not "
        "known when the model is compiled"))


def _unrolled(count, go, values, body, scan_types):
    scans = [[] for _ in scan_types]
    iteration = 0
    while go and (count is None or iteration < count):
        outputs = body(np.array(iteration, dtype=np.int64), np.array(True), *values)
        next_values = list(outputs[1:1 + len(values)])
        for scan, value in zip(scans, outputs[1 + len(values):]):
            scan.append(value)
        iteration += 1
        if not _rt.concrete(outputs[0]):
            if scan_types:
                raise _rt.DataDependentShape("Loop", None, (
                    "A Loop's scan outputs are as long as the iterations it runs, and this one's condition "
                    "is computed from the values of an input, so their length is not known when the model is "
                    "compiled"))
            return tuple(_while(None if count is None else count - iteration, outputs[0], next_values, body, iteration))
        values = next_values
        go = _truth(outputs[0])
        if count is None and iteration > 100000:
            raise _rt.DataDependentShape("Loop", None, "A Loop with no trip count ran more than 100000 iterations while it was traced")
    stacked = [
        _rt.xp(*scan).stack(scan) if scan
        else np.empty((0, *(dims or ())), dtype=_rt.jax_dtype(dtype))
        for scan, (dtype, dims) in zip(scans, scan_types)
    ]
    return tuple(values + stacked)


def _scanned(count, values, body, scan_count):
    """The loop as one `lax.scan` over `count` iterations, where the body's condition stays true
    while it is traced; None where it does not, for the caller to compile it otherwise."""
    stays_true = [True]

    def step(state, iteration):
        outputs = body(iteration, np.array(True), *state)
        stays_true[0] = stays_true[0] and _rt.concrete(outputs[0]) and _truth(outputs[0])
        carried = tuple(jnp.asarray(v) for v in outputs[1:1 + len(state)])
        return carried, tuple(outputs[1 + len(state):])

    try:
        carried, scans = jax.lax.scan(step, tuple(jnp.asarray(v) for v in values), jnp.arange(count, dtype=jnp.int64))
    except TypeError as ex:
        raise _rt.DataDependentShape("Loop", None, (
            "A Loop that is not unrolled is compiled with its body once, so its carried values must keep their "
            f"shapes and types from one iteration to the next, and they do not: {ex}")) from ex
    if not stays_true[0]:
        return None
    return tuple(list(carried) + list(scans))


def _while(count, go, values, body, start=0):
    """The loop as one `lax.while_loop`, for a loop with no scan outputs."""
    limit = None if count is None else jnp.asarray(count, dtype=jnp.int64) + start

    def more(state):
        iteration, keep_going = state[0], state[1]
        return keep_going if limit is None else keep_going & (iteration < limit)

    def step(state):
        iteration, keep_going, *carried = state
        outputs = body(iteration, keep_going, *carried)
        return (iteration + 1, _predicate(outputs[0]), *(jnp.asarray(v) for v in outputs[1:1 + len(carried)]))

    initial = (jnp.asarray(start, dtype=jnp.int64), _predicate(jnp.asarray(go)), *(jnp.asarray(v) for v in values))
    try:
        final = jax.lax.while_loop(more, step, initial)
    except TypeError as ex:
        raise _rt.DataDependentShape("Loop", None, (
            "A Loop that is not unrolled is compiled with its body once, so its carried values must keep their "
            f"shapes and types from one iteration to the next, and they do not: {ex}")) from ex
    return tuple(final[2:])
