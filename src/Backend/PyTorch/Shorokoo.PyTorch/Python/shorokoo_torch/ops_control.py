"""ONNX control flow: If and Loop.

A branch or a loop body is a Python function the translator writes, nested in the function of the
graph that holds it, so it reads outer values by closure exactly as an ONNX subgraph reads its
enclosing scope. A branch takes nothing and returns its outputs as a tuple; a loop body takes the
iteration number, the condition and the loop-carried values, and returns the condition, the
carried values and then the scan outputs.
"""

import torch

from . import runtime as _rt


def _truth(value):
    return bool(value.reshape(-1)[0].item())


def if_(condition, then_branch, else_branch):
    return then_branch() if _truth(condition) else else_branch()


def loop(trip_count, condition, carried, body, scan_types):
    """`scan_types` is one (element type code, element dims or None) per scan output: what the body
    declares of it, which an output of a loop that runs no iteration is made empty of."""
    limit = None if trip_count is None else int(trip_count.reshape(-1)[0].item())
    keep_going = True if condition is None else _truth(condition)
    values = list(carried)
    scans = [[] for _ in scan_types]
    iteration = 0
    while keep_going and (limit is None or iteration < limit):
        outputs = body(
            torch.tensor(iteration, dtype=torch.int64, device=_rt.device()),
            torch.tensor(keep_going, device=_rt.device()),
            *values)
        keep_going = _truth(outputs[0])
        values = list(outputs[1:1 + len(values)])
        for scan, value in zip(scans, outputs[1 + len(values):]):
            scan.append(value)
        iteration += 1
    stacked = [
        torch.stack(scan) if scan
        else torch.empty((0, *(dims or ())), dtype=_rt.torch_dtype(dtype), device=_rt.device())
        for scan, (dtype, dims) in zip(scans, scan_types)
    ]
    return tuple(values + stacked)
