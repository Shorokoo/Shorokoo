"""Training steps whose gradient JAX computes: the ai.shorokoo.training::AutoGrad node.

A translated step writes its forward pass -- everything before the AutoGrad node -- as a function
of the tensors it differentiates with respect to, returning the loss and the values the rest of the
step reads, and hands it to `value_and_grad`. The step is compiled whole, so XLA compiles the
forward pass, the backward pass JAX derives from it and the optimizer update as one program.
"""

import jax
import jax.numpy as jnp

from . import runtime as _rt


def value_and_grad(forward, wrt):
    """(the values `forward` carries out, the gradient of the sum of its loss with respect to each
    tensor of `wrt`): each gradient shaped and typed as its tensor, and zeros for one the loss does
    not depend on."""
    if not wrt:
        return forward()[1], ()

    def objective(*values):
        loss, carried = forward(*values)
        return jnp.sum(loss), carried

    argnums = tuple(range(len(wrt)))
    try:
        (_, carried), gradients = jax.value_and_grad(objective, argnums=argnums, has_aux=True)(*wrt)
    except ValueError as ex:
        if "while_loop" not in str(ex):
            raise
        raise _rt.DataDependentShape("Loop", None, (
            "A Loop whose trip count is computed from the values of an input is compiled as a while loop, "
            "which JAX cannot differentiate, and this one lies between the loss and a tensor the step "
            "differentiates it with respect to")) from ex
    return carried, gradients
