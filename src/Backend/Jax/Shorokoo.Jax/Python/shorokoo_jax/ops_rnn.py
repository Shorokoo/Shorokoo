"""ONNX recurrent networks: RNN, GRU and LSTM.

Each is an explicit loop over the time steps in ONNX's own gate order and with ONNX's activation
functions. The loop is a Python loop over the sequence length, which the input's shape fixes, so it
is unrolled into the compiled program and JAX differentiates it as written.

The helpers take the node's inputs positionally (an omitted optional input as None) and its
attributes as keywords, and return as many of (Y, Y_h[, Y_c]) as the node declares.

Every step is computed for the whole batch; `sequence_lens`, which may be computed from an input,
masks it per batch entry, so a finished sequence keeps its last state and writes zeros to Y. The
reverse direction runs each sequence backwards within its own length: the time axis is permuted per
batch entry before the loop and the same permutation, being its own inverse, puts Y back in order
after it.
"""

import jax
import jax.numpy as jnp
import numpy as np

from . import runtime as _rt

_LSTM_DEFAULTS = ("Sigmoid", "Tanh", "Tanh")
_GRU_DEFAULTS = ("Sigmoid", "Tanh")
_RNN_DEFAULTS = ("Tanh",)

# How many of activation_alpha / activation_beta each activation consumes, and the defaults it
# falls back to when the lists run out -- the defaults of the ONNX operator of the same name. ONNX
# states none for ScaledTanh, and Shorokoo hands over a node whose ScaledTanh is given no value as
# it was written; ONNX Runtime reads 0 for the missing alpha and beta of every recurrent operator,
# so that is what is read here too.
_ARGUMENTS = {
    "relu": (0, 0, 0.0, 0.0),
    "tanh": (0, 0, 0.0, 0.0),
    "sigmoid": (0, 0, 0.0, 0.0),
    "softsign": (0, 0, 0.0, 0.0),
    "softplus": (0, 0, 0.0, 0.0),
    "affine": (1, 1, 1.0, 0.0),
    "leakyrelu": (1, 0, 0.01, 0.0),
    "thresholdedrelu": (1, 0, 1.0, 0.0),
    "scaledtanh": (1, 1, 0.0, 0.0),
    "hardsigmoid": (1, 1, 0.2, 0.5),
    "elu": (1, 0, 1.0, 0.0),
}


def _function(name, alpha, beta):
    """The activation `name`, whose alpha and beta take the dtype of the value it is applied to."""
    a = lambda x: np.asarray(alpha, dtype=x.dtype)
    b = lambda x: np.asarray(beta, dtype=x.dtype)
    if name == "relu":
        return lambda x: jnp.where(x > 0, x, jnp.zeros_like(x))
    if name == "tanh":
        return jnp.tanh
    if name == "sigmoid":
        return jax.nn.sigmoid
    if name == "softsign":
        return lambda x: x / (1 + jnp.abs(x))
    if name == "softplus":
        return lambda x: jnp.logaddexp(x, jnp.zeros_like(x))
    if name == "affine":
        return lambda x: a(x) * x + b(x)
    if name == "leakyrelu":
        return lambda x: jnp.where(x >= 0, x, a(x) * x)
    if name == "thresholdedrelu":
        return lambda x: jnp.where(x > a(x), x, jnp.zeros_like(x))
    if name == "scaledtanh":
        return lambda x: a(x) * jnp.tanh(b(x) * x)
    if name == "hardsigmoid":
        return lambda x: jnp.clip(a(x) * x + b(x), 0, 1)
    if name == "elu":
        # Clamped inside expm1: the branch `where` does not select still has its gradient taken.
        return lambda x: jnp.where(x > 0, x, a(x) * jnp.expm1(jnp.minimum(x, 0)))
    raise NotImplementedError(f"the recurrent activation '{name}' is not an ONNX one")


def _activations(names, alphas, betas, defaults, directions):
    """Per direction, the activation functions, consuming the alpha and beta lists in order."""
    names = list(names) if names else list(defaults) * directions
    alphas = list(alphas or [])
    betas = list(betas or [])
    functions = []
    for name in names:
        key = name.lower()
        if key not in _ARGUMENTS:
            raise NotImplementedError(f"the recurrent activation '{name}' is not an ONNX one")
        alpha_count, beta_count, alpha, beta = _ARGUMENTS[key]
        if alpha_count and alphas:
            alpha = alphas.pop(0)
        if beta_count and betas:
            beta = betas.pop(0)
        functions.append(_function(key, alpha, beta))
    per = len(defaults)
    return [functions[d * per:(d + 1) * per] for d in range(directions)]


def _directions(direction):
    if direction == "forward":
        return [False]
    if direction == "reverse":
        return [True]
    if direction == "bidirectional":
        return [False, True]
    raise NotImplementedError(f"the recurrent direction '{direction}' is not an ONNX one")


def _clipped(clip):
    if clip is None:
        return lambda x: x
    return lambda x: jnp.clip(x, np.asarray(-clip, dtype=x.dtype), np.asarray(clip, dtype=x.dtype))


def _time_permutation(lengths, steps, batch):
    """The index that runs each batch entry's sequence backwards within its length and leaves the
    padding after it in place: its own inverse."""
    t = np.broadcast_to(np.arange(steps, dtype=np.int64)[:, None], (steps, batch))
    if lengths is None:
        return steps - 1 - t
    return jnp.where(t < lengths, lengths - 1 - t, t)


def _run(x, lengths, reverse, state, step):
    """Runs `step` over the time axis of x ([seq, batch, input]) from `state`, a tuple whose first
    element is the hidden state. Returns (Y [seq, batch, hidden], final state)."""
    steps, batch = x.shape[0], x.shape[1]
    if reverse:
        order = _time_permutation(lengths, steps, batch)
        x = jnp.take_along_axis(x, order[:, :, None], axis=0)
    outputs = []
    for t in range(steps):
        new = step(x[t], state)
        if lengths is None:
            state = new
            outputs.append(new[0])
        else:
            active = (t < lengths)[:, None]
            state = tuple(jnp.where(active, n, s) for n, s in zip(new, state))
            outputs.append(jnp.where(active, new[0], jnp.zeros_like(new[0])))
    y = jnp.stack(outputs) if outputs else jnp.zeros((0, batch, state[0].shape[1]), dtype=x.dtype)
    if reverse:
        y = jnp.take_along_axis(y, order[:, :, None], axis=0)
    return y, state


def _recurrent(x, w, sequence_lens, initial_states, layout, direction, hidden_size, run_direction, outputs):
    """The shared frame: layout, directions, the initial states and stacking the outputs."""
    if layout:
        x = jnp.swapaxes(x, 0, 1)
        initial_states = [None if s is None else jnp.swapaxes(s, 0, 1) for s in initial_states]
    steps, batch = x.shape[0], x.shape[1]
    lengths = None
    if sequence_lens is not None:
        lengths = sequence_lens.astype(np.int64)
        # Lengths known to be the full sequence mask nothing; traced ones mask whatever they hold.
        if _rt.concrete(lengths) and bool((lengths == steps).all()):
            lengths = None
    ys = []
    finals = [[] for _ in initial_states]
    for d, reverse in enumerate(_directions(direction)):
        state = tuple(
            jnp.zeros((batch, hidden_size), dtype=x.dtype) if s is None else s[d]
            for s in initial_states)
        y, state = run_direction(d, x, lengths, reverse, state)
        if lengths is not None:
            # A sequence of length 0 ends in zero states, as ONNX Runtime has it, not its initial ones.
            empty = (lengths == 0)[:, None]
            state = tuple(jnp.where(empty, jnp.zeros_like(s), s) for s in state)
        ys.append(y)
        for kept, s in zip(finals, state):
            kept.append(s)
    y = jnp.stack(ys, 1)
    finals = [jnp.stack(kept, 0) for kept in finals]
    if layout:
        y = jnp.transpose(y, (2, 0, 1, 3))
        finals = [jnp.swapaxes(f, 0, 1) for f in finals]
    return tuple([y, *finals][:outputs])


def _hidden(r, hidden_size):
    return int(hidden_size) if hidden_size is not None else r.shape[-1]


def _matmul_t(v, m):
    return jnp.matmul(v, m.T, precision=_rt.PRECISION)


def rnn(x, w, r, b=None, sequence_lens=None, initial_h=None, /, *, activation_alpha=None,
        activation_beta=None, activations=None, clip=None, direction="forward", hidden_size=None,
        layout=0, _outputs):
    hidden = _hidden(r, hidden_size)
    acts = _activations(activations, activation_alpha, activation_beta, _RNN_DEFAULTS, len(_directions(direction)))
    clipped = _clipped(clip)

    def run_direction(d, xs, lengths, reverse, state):
        wd, rd = w[d], r[d]
        bias = 0 if b is None else b[d, :hidden] + b[d, hidden:]
        f, = acts[d]

        def step(xt, s):
            return (f(clipped(_matmul_t(xt, wd) + _matmul_t(s[0], rd) + bias)),)
        return _run(xs, lengths, reverse, state, step)

    return _recurrent(x, w, sequence_lens, [initial_h], layout, direction, hidden, run_direction, _outputs)


def gru(x, w, r, b=None, sequence_lens=None, initial_h=None, /, *, activation_alpha=None,
        activation_beta=None, activations=None, clip=None, direction="forward", hidden_size=None,
        layout=0, linear_before_reset=0, _outputs):
    hidden = _hidden(r, hidden_size)
    acts = _activations(activations, activation_alpha, activation_beta, _GRU_DEFAULTS, len(_directions(direction)))
    clipped = _clipped(clip)

    def run_direction(d, xs, lengths, reverse, state):
        wz, wr, wh = jnp.split(w[d], 3)
        rz, rr, rh = jnp.split(r[d], 3)
        if b is None:
            wbz = wbr = wbh = rbz = rbr = rbh = 0
        else:
            wbz, wbr, wbh, rbz, rbr, rbh = jnp.split(b[d], 6)
        f, g = acts[d]

        def step(xt, s):
            h = s[0]
            z = f(clipped(_matmul_t(xt, wz) + _matmul_t(h, rz) + wbz + rbz))
            reset = f(clipped(_matmul_t(xt, wr) + _matmul_t(h, rr) + wbr + rbr))
            if linear_before_reset:
                candidate = g(clipped(_matmul_t(xt, wh) + reset * (_matmul_t(h, rh) + rbh) + wbh))
            else:
                candidate = g(clipped(_matmul_t(xt, wh) + _matmul_t(reset * h, rh) + rbh + wbh))
            return ((1 - z) * candidate + z * h,)
        return _run(xs, lengths, reverse, state, step)

    return _recurrent(x, w, sequence_lens, [initial_h], layout, direction, hidden, run_direction, _outputs)


def lstm(x, w, r, b=None, sequence_lens=None, initial_h=None, initial_c=None, p=None, /, *,
         activation_alpha=None, activation_beta=None, activations=None, clip=None, direction="forward",
         hidden_size=None, input_forget=0, layout=0, _outputs):
    hidden = _hidden(r, hidden_size)
    acts = _activations(activations, activation_alpha, activation_beta, _LSTM_DEFAULTS, len(_directions(direction)))
    clipped = _clipped(clip)

    def run_direction(d, xs, lengths, reverse, state):
        wd, rd = w[d], r[d]
        bias = 0 if b is None else b[d, :4 * hidden] + b[d, 4 * hidden:]
        if p is None:
            pi = po = pf = 0
        else:
            pi, po, pf = jnp.split(p[d], 3)
        f, g, h_act = acts[d]

        def step(xt, s):
            h, c = s
            gates = _matmul_t(xt, wd) + _matmul_t(h, rd) + bias
            gi, go, gf, gc = jnp.split(gates, 4, axis=1)
            i = f(clipped(gi + pi * c))
            forget = 1 - i if input_forget else f(clipped(gf + pf * c))
            candidate = g(clipped(gc))
            c_new = forget * c + i * candidate
            o = f(clipped(go + po * c_new))
            return (o * h_act(c_new), c_new)
        return _run(xs, lengths, reverse, state, step)

    return _recurrent(x, w, sequence_lens, [initial_h, initial_c], layout, direction, hidden, run_direction, _outputs)
