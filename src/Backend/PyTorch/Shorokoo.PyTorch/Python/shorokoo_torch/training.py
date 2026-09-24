"""Training steps whose gradient torch computes: the ai.shorokoo.training::AutoGrad node.

A translated step makes each tensor it differentiates with respect to a leaf (`leaf`), runs the
forward pass up to the AutoGrad node under torch.enable_grad(), takes the gradient there
(`autograd`), and runs the rest -- the optimizer update -- under the run's torch.no_grad(). Every
output is handed back detached (`detach`), so nothing a step returns carries its autograd graph.
"""

import torch

_DIFFERENTIABLE = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def leaf(value):
    """`value` as a leaf of the step's autograd graph: the same memory, no history, requiring grad."""
    if not isinstance(value, torch.Tensor) or value.dtype not in _DIFFERENTIABLE:
        kind = value.dtype if isinstance(value, torch.Tensor) else type(value).__name__
        raise TypeError(f"AutoGrad differentiates with respect to floating-point tensors only, not {kind}")
    return value.detach().requires_grad_(True)


def autograd(loss, wrt):
    """The gradient of the sum of `loss` with respect to each tensor of `wrt`, each shaped and typed
    as that tensor, and zeros for one the loss does not depend on."""
    if not wrt:
        return ()
    if not loss.requires_grad:
        return tuple(torch.zeros_like(w) for w in wrt)
    return torch.autograd.grad(
        loss, wrt, grad_outputs=torch.ones_like(loss), allow_unused=True, materialize_grads=True)


def detach(value):
    """An output with no autograd history: a tensor detached, a sequence element by element. A tensor
    that has none is handed back as it is -- the very object -- since the run recognises an output it
    wrote into an input's memory by identity."""
    if isinstance(value, torch.Tensor):
        return value.detach() if value.requires_grad else value
    if isinstance(value, list):
        return [detach(item) for item in value]
    return value
