"""ONNX recurrent networks: RNN, GRU and LSTM.

Each is an explicit loop over the time steps in ONNX's own gate order and with ONNX's activation
functions, rather than a call of torch.nn.RNN/GRU/LSTM, whose gate order, bias layout and fixed
activations differ from ONNX's. The loop is built of ordinary differentiable tensor operations, so
autograd differentiates it as written.

The helpers take the node's inputs positionally (an omitted optional input as None) and its
attributes as keywords, and return as many of (Y, Y_h[, Y_c]) as the node declares.

Every step is computed for the whole batch; `sequence_lens` masks it per batch entry, so a
finished sequence keeps its last state and writes zeros to Y. The reverse direction runs each
sequence backwards within its own length: the time axis is permuted per batch entry before the
loop and the same permutation, being its own inverse, puts Y back in order after it.
"""

import torch

_LSTM_DEFAULTS = ("Sigmoid", "Tanh", "Tanh")
_GRU_DEFAULTS = ("Sigmoid", "Tanh")
_RNN_DEFAULTS = ("Tanh",)

# How many of activation_alpha / activation_beta each activation consumes, and the defaults it
# falls back to when the lists run out -- the defaults of the ONNX operator of the same name.
_ARGUMENTS = {
    "relu": (0, 0, 0.0, 0.0),
    "tanh": (0, 0, 0.0, 0.0),
    "sigmoid": (0, 0, 0.0, 0.0),
    "softsign": (0, 0, 0.0, 0.0),
    "softplus": (0, 0, 0.0, 0.0),
    "affine": (1, 1, 1.0, 0.0),
    "leakyrelu": (1, 0, 0.01, 0.0),
    "thresholdedrelu": (1, 0, 1.0, 0.0),
    "scaledtanh": (1, 1, 1.0, 1.0),
    "hardsigmoid": (1, 1, 0.2, 0.5),
    "elu": (1, 0, 1.0, 0.0),
}


def _function(name, alpha, beta):
    if name == "relu":
        return torch.relu
    if name == "tanh":
        return torch.tanh
    if name == "sigmoid":
        return torch.sigmoid
    if name == "softsign":
        return lambda x: x / (1 + torch.abs(x))
    if name == "softplus":
        return lambda x: torch.where(x > 0, x + torch.log1p(torch.exp(-x)), torch.log1p(torch.exp(x)))
    if name == "affine":
        return lambda x: alpha * x + beta
    if name == "leakyrelu":
        return lambda x: torch.where(x >= 0, x, alpha * x)
    if name == "thresholdedrelu":
        return lambda x: torch.where(x > alpha, x, torch.zeros_like(x))
    if name == "scaledtanh":
        return lambda x: alpha * torch.tanh(beta * x)
    if name == "hardsigmoid":
        return lambda x: torch.clamp(alpha * x + beta, 0, 1)
    if name == "elu":
        return lambda x: torch.where(x > 0, x, alpha * torch.expm1(x))
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
    return lambda x: torch.clamp(x, -clip, clip)


def _time_permutation(lengths, steps, batch, device):
    """The index that runs each batch entry's sequence backwards within its length and leaves the
    padding after it in place: its own inverse."""
    t = torch.arange(steps, device=device).unsqueeze(1).expand(steps, batch)
    if lengths is None:
        return steps - 1 - t
    return torch.where(t < lengths, lengths - 1 - t, t)


def _run(x, lengths, reverse, state, step):
    """Runs `step` over the time axis of x ([seq, batch, input]) from `state`, a tuple whose first
    element is the hidden state. Returns (Y [seq, batch, hidden], final state)."""
    steps, batch = x.shape[0], x.shape[1]
    if reverse:
        order = _time_permutation(lengths, steps, batch, x.device)
        x = torch.gather(x, 0, order.unsqueeze(2).expand(-1, -1, x.shape[2]))
    outputs = []
    for t in range(steps):
        new = step(x[t], state)
        if lengths is None:
            state = new
            outputs.append(new[0])
        else:
            active = (t < lengths).unsqueeze(1)
            state = tuple(torch.where(active, n, s) for n, s in zip(new, state))
            outputs.append(torch.where(active, new[0], torch.zeros_like(new[0])))
    y = torch.stack(outputs) if outputs else x.new_zeros((0, batch, state[0].shape[1]))
    if reverse:
        y = torch.gather(y, 0, order.unsqueeze(2).expand(-1, -1, y.shape[2]))
    return y, state


def _recurrent(x, w, sequence_lens, initial_states, layout, direction, hidden_size, run_direction, outputs):
    """The shared frame: layout, directions, the initial states and stacking the outputs."""
    if layout:
        x = x.transpose(0, 1)
        initial_states = [None if s is None else s.transpose(0, 1) for s in initial_states]
    steps, batch = x.shape[0], x.shape[1]
    lengths = None
    if sequence_lens is not None:
        lengths = sequence_lens.to(device=x.device, dtype=torch.int64)
        if bool((lengths == steps).all()):
            lengths = None
    ys = []
    finals = [[] for _ in initial_states]
    for d, reverse in enumerate(_directions(direction)):
        state = tuple(
            x.new_zeros((batch, hidden_size)) if s is None else s[d]
            for s in initial_states)
        y, state = run_direction(d, x, lengths, reverse, state)
        ys.append(y)
        for kept, s in zip(finals, state):
            kept.append(s)
    y = torch.stack(ys, 1)
    finals = [torch.stack(kept, 0) for kept in finals]
    if layout:
        y = y.permute(2, 0, 1, 3)
        finals = [f.transpose(0, 1) for f in finals]
    return tuple([y, *finals][:outputs])


def _hidden(r, hidden_size):
    return int(hidden_size) if hidden_size is not None else r.shape[-1]


def _matmul_t(v, m):
    return torch.matmul(v, m.transpose(0, 1))


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
        wz, wr, wh = w[d].split(hidden)
        rz, rr, rh = r[d].split(hidden)
        if b is None:
            wbz = wbr = wbh = rbz = rbr = rbh = 0
        else:
            wbz, wbr, wbh, rbz, rbr, rbh = b[d].split(hidden)
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
            pi, po, pf = p[d].split(hidden)
        f, g, h_act = acts[d]

        def step(xt, s):
            h, c = s
            gates = _matmul_t(xt, wd) + _matmul_t(h, rd) + bias
            gi, go, gf, gc = gates.split(hidden, dim=1)
            i = f(clipped(gi + pi * c))
            forget = 1 - i if input_forget else f(clipped(gf + pf * c))
            candidate = g(clipped(gc))
            c_new = forget * c + i * candidate
            o = f(clipped(go + po * c_new))
            return (o * h_act(c_new), c_new)
        return _run(xs, lengths, reverse, state, step)

    return _recurrent(x, w, sequence_lens, [initial_h, initial_c], layout, direction, hidden, run_direction, _outputs)
