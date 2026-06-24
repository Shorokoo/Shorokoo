# NN library: layers, losses, optimizers, initializers

Related: [defining-models.md](defining-models.md) · [training.md](training.md) · [core-types.md](core-types.md)

## Facts

- `Shorokoo.Modules` is the baseline neural-network library: ready-made
  initializers, layers, losses, and optimizers, all built from ordinary
  Shorokoo `[Module]`s / `[TrainableParamInitializer]`s — nothing privileged.
- Namespaces: `Shorokoo.Modules.Initializers`, `.Layers`, `.Losses`, `.Optimizers`.
- Layers are `[Module]` classes: use them like any module —
  `Linear.Call(hypers..., x)` inline, or `Linear.Model(hypers...).Call(x)` to fix
  the hyperparameters once (see [defining-models.md](defining-models.md)).
- Exceptions to the `[Module]` rule: pooling, the generalized convolution
  helpers, and the recurrent layers are plain C#-argument static helpers
  (`Pooling.MaxPool2d(x, 2)`, `Convolution.Conv(...)`, `Recurrent.RNN(x, 16)`),
  and plain activations are tensor one-liners (`x.Relu()`) — none needs a module.

```bash
dotnet add package Shorokoo.Modules
```

## Initializers (`Shorokoo.Modules.Initializers`)

All are shape-only `[TrainableParamInitializer]`s — call sites are one-liners
like `Zeros.Init([outFeatures])` or `KaimingUniform.Init([outC, inC, k, k])`.

| Initializer | Fills with | Notes |
|---|---|---|
| `Zeros` | 0.0 | biases, BatchNorm beta |
| `Ones` | 1.0 | BatchNorm/LayerNorm gamma |
| `Constant` | `value` (every element) | deterministic (no RNG); any rank; the parameterized generalization of `Zeros`/`Ones` (`Constant(0)`/`Constant(1)`); `value` is an `Init` arg (`Constant.Init([shape], Scalar(v))`), à la `RecurrentUniform`; PyTorch `constant_` / Keras `Constant` |
| `Uniform` | U(0, 1) | seeded; the fixed U(0, 1) default form (use `UniformRange` for a configurable range) |
| `Normal` | N(0, 1) | seeded; PyTorch's `nn.Embedding` default; the fixed N(0, 1) default form (use `NormalDist` for configurable mean/std) |
| `UniformRange` | U(low, high) | seeded; any rank; the parameterized generalization of `Uniform` (`Uniform` retained as the U(0, 1) default); `low`/`high` are `Init` args (`UniformRange.Init([shape], Scalar(lo), Scalar(hi))`), built as the affine transform of a standard draw; expects `low ≤ high`; PyTorch `uniform_(a, b)` / Keras `RandomUniform(minval, maxval)` |
| `NormalDist` | N(mean, std) | seeded; any rank; the parameterized generalization of `Normal` (`Normal` retained as the N(0, 1) default); `mean`/`std` are `Init` args (`NormalDist.Init([shape], Scalar(m), Scalar(s))`), built as the affine transform of a standard draw; expects `std ≥ 0`; PyTorch `normal_(mean, std)` / Keras `RandomNormal(mean, stddev)` |
| `XavierUniform` | U(−a, a), a = √(6 / (fanIn + fanOut)) | gain 1; seeded; rank ≥ 2 |
| `XavierNormal` | N(0, √(2 / (fanIn + fanOut))) | gain 1; seeded; rank ≥ 2 |
| `KaimingUniform` | U(−b, b), b = √(6 / fanIn) | ReLU gain √2, fan-in mode; seeded; rank ≥ 2; the default weight init of `Linear` and the conv layers |
| `KaimingNormal` | N(0, √(2 / fanIn)) | ReLU gain √2, fan-in mode; seeded; rank ≥ 2 |
| `XavierUniformGain` | U(−a, a), a = gain·√(6 / (fanIn + fanOut)) | seeded; rank ≥ 2; configurable-gain generalization of `XavierUniform` (== it at `gain = 1`; `XavierUniform` retained as the gain-baked default); `gain` is an `Init` arg (`XavierUniformGain.Init([shape], Scalar(g))`); PyTorch `xavier_uniform_(t, gain)` (`gain` = the `calculate_gain` std multiplier — the caller computes it) |
| `XavierNormalGain` | N(0, gain·√(2 / (fanIn + fanOut))) | seeded; rank ≥ 2; configurable-gain generalization of `XavierNormal` (== it at `gain = 1`; `XavierNormal` retained as the gain-baked default); `gain` is an `Init` arg; PyTorch `xavier_normal_(t, gain)` |
| `KaimingUniformGain` | U(−b, b), b = gain·√(3 / fanIn) | seeded; rank ≥ 2; configurable-gain generalization of `KaimingUniform` (== it at `gain = √2`; `KaimingUniform` retained as the gain-baked default). **Base factor is the bare √(3 / fanIn), not √(6 / fanIn)** — the default bakes gain √2, so the user-supplied gain replaces it (no double-bake). `gain` is an `Init` arg; PyTorch `kaiming_uniform_` (gain via `calculate_gain`) |
| `KaimingNormalGain` | N(0, gain·√(1 / fanIn)) | seeded; rank ≥ 2; configurable-gain generalization of `KaimingNormal` (== it at `gain = √2`; `KaimingNormal` retained as the gain-baked default). **Base factor is the bare √(1 / fanIn), not √(2 / fanIn)** — the default bakes gain √2, so the user-supplied gain replaces it (no double-bake). `gain` is an `Init` arg; PyTorch `kaiming_normal_` |
| `TruncatedNormal` | N(0, 1) clamped to [−2, 2] | seeded; clamp approximation (in-graph rejection sampling isn't possible); Keras/JAX-style default |
| `LeCunNormal` | N(0, √(1 / fanIn)) | seeded; rank ≥ 2; JAX/Flax `lecun_normal` (SELU / self-normalizing nets) |
| `Orthogonal` | (semi-)orthogonal matrix (`QᵀQ ≈ I` / `QQᵀ ≈ I`) | seeded; rank ≥ 2; **Björck/Newton–Schulz approximation** (15 cubic iterations `Y ← 1.5·Y − 0.5·Y·(YᵀY)` from a seeded Gaussian — exact QR/SVD-orthogonal isn't expressible in Shorokoo's op set, cf. `TruncatedNormal`); gain 1; Saxe-2013 dynamical isometry (RNN recurrent matrices, deep stacks); PyTorch `orthogonal_` |
| `RecurrentUniform` | U(−1/√H, 1/√H), H = `shape[1]` | seeded; PyTorch's `nn.RNN`/`nn.LSTM`/`nn.GRU` default (`k = 1/hidden_size`); reads the hidden dim from axis 1, so it inits recurrent W `[D, H, in]`, R `[D, H, H]`, and bias `[D, H]` alike; used by the `Recurrent` layers |

- **Seeded determinism**: the random initializers use fixed seeds, so every
  materialization is deterministic — and two parameters of the same shape
  initialized by the same class receive **identical** values.
- **Fan-in/fan-out** are computed in-graph from the shape vector:
  `fanIn = prod(shape) / shape[0]`, `fanOut = prod(shape) / shape[1]` — the
  PyTorch convention for Linear `[out, in]` and Conv `[outC, inC/g, k...]`
  layouts. Hence the rank ≥ 2 requirement: use `Zeros`/`Ones`/`Uniform`/`Normal`
  for biases.

## Layers (`Shorokoo.Modules.Layers`)

Layer hyperparameters are `[Hyper]` graph scalars; pass them as `Scalar(...)`
values. Signatures below are the `Inline` shapes (`Call` takes the same
arguments).

### Linear

```csharp
// y = x @ W^T (+ b); flattens trailing dims: [N, d1, d2, ...] -> [N, d1*d2*...]
Linear.Call(Scalar<int64> outFeatures, Scalar<bit> useBias, Tensor<float32> x)
```

Weight `[outFeatures, inFeatures]` is `KaimingUniform`-initialized; bias
`[outFeatures]` is zero-initialized. The bias parameter is created
unconditionally (both `IfElse` branches are built); `useBias` selects whether it
is added.

### Bilinear

```csharp
// y_k = x1^T A_k x2 + b_k  (PyTorch nn.Bilinear)
Bilinear.Call(Scalar<int64> in1Features, Scalar<int64> in2Features,
              Scalar<int64> outFeatures, Scalar<bit> useBias,
              Tensor<float32> x1, Tensor<float32> x2)
```

A bilinear transformation — the second-order *interaction* sibling of `Linear`.
Per output channel `k`, `y[..., k] = Σ_{i,j} x1[..., i]·A[k,i,j]·x2[..., j] (+ b[k])`.
Weight `A` has shape `[outFeatures, in1Features, in2Features]`; bias `b` is
`[outFeatures]`. Both `A` and `b` are initialized from `U(±1/√in1Features)`
(PyTorch's bound, via `RecurrentUniform`) — note the bias is **not**
zero-initialized, unlike `Linear`. The contraction is over each input's **last**
axis; the two inputs must share their leading (batch) dims (`(*, in1)`, `(*, in2)`
→ `(*, out)`), which are preserved. `useBias = false` omits the bias term.

### Conv2d / Conv1d — dynamic geometry

```csharp
// NCHW; square kernel, symmetric padding
Conv2d.Call(Scalar<int64> outChannels, Scalar<int64> kernelSize, Scalar<int64> stride,
            Scalar<int64> padding, Scalar<int64> dilation, Scalar<int64> groups,
            Scalar<bit> useBias, Tensor<float32> x)

// NCL; same hyperparameters, one spatial dim
Conv1d.Call(outChannels, kernelSize, stride, padding, dilation, groups, useBias, x)

// NCDHW; cubic kernel (one kernelSize covers all three spatial dims)
Conv3d.Call(outChannels, kernelSize, stride, padding, dilation, groups, useBias, x)
```

**Dynamic conv geometry**: all geometry (kernel size, stride, padding, dilation,
groups) is hyperparameter-driven. This works because the layers use the
tensor-geometry `NN.Conv` overload (SHRK_CONV), which carries geometry as int64
tensor inputs and is lowered to a standard ONNX Conv at concretization. Weight
`[outChannels, inChannels/groups, k(, k)]` is `KaimingUniform`-initialized;
`inChannels` is read from the input's shape in-graph. These modules cover the
**square-kernel / symmetric-pad** case; for per-axis geometry, `auto_pad`, or a
non-zeros `padding_mode` use the generalized `Convolution` helper below.

### ConvTranspose2d — default geometry only

```csharp
ConvTranspose2d.Call(Scalar<int64> outChannels, Scalar<int64> kernelSize,
                     Scalar<bit> useBias, Tensor<float32> x)
```

There is no tensorized ConvTranspose lowering (only Conv has SHRK_CONV), so the
geometry attributes stay at the ONNX defaults — stride 1, no padding, dilation 1,
group 1 — with the kernel shape inferred from the (dynamic) weight
`[inChannels, outChannels, k, k]`. For other stride/padding combinations call
`NN.ConvTranspose` directly with static attribute values.

### Convolution — generalized per-axis helpers

The `[Module]` layers above (`Conv1d/2d/3d`, `ConvTranspose2d`) keep their
square/cubic, hyperparameter-driven signature for backward compatibility and the
`Model(...)`/hyperparameter-baking ergonomics — they are the scalar-hyper
convenience. For the **full ONNX attribute surface** (per-axis kernel/stride/
padding/dilation, asymmetric padding, `auto_pad`, `groups`, `padding_mode`, and
transposed-conv `output_padding`/`output_shape`) use the static `Convolution`
helper class. Like `Pooling`, it is a static class of graph-building helpers with
**plain C# array arguments** — not `[Module]`s. Convolution geometry is
shape-determining (it sizes the weight) and is therefore baked at concretization
regardless, so it gains nothing from being a `[Hyper]`; only the `inChannels`
axis of the weight is read in-graph from `x.ShapeTensor()[1]`, so these helpers
stay lazy in the input channel count exactly like the per-dim modules.

```csharp
// Forward conv — per-axis geometry (spatial rank = kernelSize.Length).
//   stride/dilation:  length 1 (broadcast to all axes) or spatialRank
//   padding:          length spatialRank (symmetric) or 2*spatialRank
//                     (ONNX [begin1..beginN, end1..endN] — asymmetric pads)
Convolution.Conv(x, outChannels, long[] kernelSize,
                 stride: null, padding: null, dilation: null,
                 groups: 1, bias: true,
                 autoPad: AutoPad.NotSet, paddingMode: PaddingMode.Zeros);

// Square/cubic convenience: one scalar per knob, broadcast to all spatial axes
// (the spatial rank is taken from x.Rank() - 2, which must be known at build time).
Convolution.Conv(x, outChannels, long kernelSize,
                 stride: 1, padding: 0, dilation: 1, groups: 1, bias: true,
                 autoPad: AutoPad.NotSet, paddingMode: PaddingMode.Zeros);

// Rank-fixing aliases (assert kernelSize.Length == 1/2/3 and forward):
Convolution.Conv1d(x, outChannels, kernelSize /* len 1 */, ...);  // NCL
Convolution.Conv2d(x, outChannels, kernelSize /* len 2 */, ...);  // NCHW
Convolution.Conv3d(x, outChannels, kernelSize /* len 3 */, ...);  // NCDHW

// Transposed conv — per-axis geometry; zeros padding only (no padding_mode).
Convolution.ConvTranspose(x, outChannels, long[] kernelSize,
                          stride: null, padding: null, outputPadding: null,
                          dilation: null, groups: 1, bias: true,
                          outputShape: null, autoPad: AutoPad.NotSet);
// + scalar convenience overload and ConvTranspose1d/2d/3d rank aliases.
```

- **Weight & init.** Forward conv weight is `[outChannels, inChannels/groups, k…]`
  (fan-in `inC/groups·∏k`); transposed conv weight is
  `[inChannels, outChannels/groups, k…]` (in/out axes swapped). Both are
  `KaimingUniform`-initialized. `bias: true` makes a trainable zero-initialized
  bias `[outChannels]`; `bias: false` uses an all-zero constant.
- **`auto_pad`.** `AutoPad.SameUpper` matches TF/PyTorch `"same"` (extra pad on
  the high side); `SameLower`, `Valid`, and `NotSet` (explicit pads) are also
  available. `auto_pad` cannot be combined with an explicit `padding_mode` other
  than `Zeros` (those compose a separate `Pad`) or with `Causal`.
- **`groups`.** `1` is dense; `groups == inChannels` (with `outChannels` a
  multiple of `inChannels`) is depthwise. Weight axis 1 is `inChannels/groups`.
- **`padding_mode`.** `Zeros` uses the conv's own (differentiable) implicit
  padding. `Reflect`/`Replicate`/`Circular` map to `PadMode.Reflect`/`Edge`/`Wrap`
  and are realized by an explicit `Tensor.Pad` over the spatial axes followed by a
  zero-pad conv. **Caveat (loud):** the `Pad` step for these modes is
  **non-differentiable** — reflect/edge/wrap have no autodiff and no QEE values,
  so they **throw in autodiff**. These modes are **forward / inference-grade
  only**; do not expect to train through the pad stage. `Causal` is **1-D only**
  (rejected for higher spatial ranks): it left-pads `(k-1)*dilation` zeros on the
  single spatial axis so `out[t]` never sees future input (WaveNet-style), and is
  itself a constant (differentiable) zero-pad.
- **ConvTranspose `output_padding` / `output_shape`.** `output_padding`
  disambiguates the output size when `stride > 1` maps several input sizes to the
  same output (it only changes the claimed shape; it is not literal output
  zero-padding). PyTorch's `output_padding < max(stride, dilation)` guard is **not**
  re-imposed here — ONNX Runtime validates the geometry. `output_shape` names the
  target spatial size directly and, when given, overrides `output_padding`.
  Transposed conv is **zeros-only** (no `padding_mode`): its "padding" is an
  output-shape crop, not an input border.

### Recurrent layers — `Recurrent.RNN` / `Recurrent.LSTM` / `Recurrent.GRU`

The vanilla (Elman) recurrent layer is the static helper `Recurrent.RNN`, in the
`Recurrent` class alongside `Recurrent.LSTM` and `Recurrent.GRU` (both below). Like `Convolution` and
`Pooling`, `Recurrent` is a **static class of plain-C#-argument graph-building
helpers, not a `[Module]`** — every knob (`hiddenSize`, `nonlinearity`,
`direction`, `numLayers`, `batchFirst`, `bias`) is shape- or topology-determining
and is baked at build time regardless, and the `nonlinearity`/`direction` enums
cannot be expressed as the scalar-only `[Hyper]` type. The owned weights are still
created via the shared `RecurrentUniform` initializer's `Init` (which emits
trainable-parameter nodes into the composed graph exactly as `Convolution.Conv`
owns its weight), so the trainable configuration trains end-to-end.

```csharp
// h_t = act(W·x_t + R·h_{t-1} + b); returns the full output sequence y and the
// final state hN. Defaults match PyTorch nn.RNN.
(Tensor<float32> y, Tensor<float32> hN) = Recurrent.RNN(
    Tensor<float32> x,
    long hiddenSize,                                  // H — required (no default)
    RnnNonlinearity nonlinearity = RnnNonlinearity.Tanh,
    RnnDirection    direction    = RnnDirection.Forward,
    int  numLayers  = 1,
    bool batchFirst = false,
    bool bias       = true);
```

- **Input / output layout.** `x` is `[L, N, inputSize]` (sequence-first) by
  default, or `[N, L, inputSize]` when `batchFirst: true`. `inputSize` is read
  in-graph from the last axis, so the layer is lazy in input size (like
  `Convolution.Conv`'s `inChannels`). `y` is the **full output sequence** (every
  step's hidden state) in PyTorch layout `[L, N, D·H]` (or `[N, L, D·H]` when
  `batchFirst`), where `D = 2` for `Bidirectional`, else `1`. For "last output
  only" (Keras `return_sequences=False`), slice `y[-1]` or read `hN`.
- **Return contract.** `hN` is the final hidden state per direction and layer,
  shaped `[D·numLayers, N, H]` — batch-second regardless of `batchFirst`, matching
  PyTorch. (The `(y, hN)` tuple covers both Keras `return_sequences` and
  `return_state` modes, more flexibly than booleans.)
- **Weights & init.** Per layer, with `D = direction == Bidirectional ? 2 : 1`:
  `W [D, H, in]` (input→hidden), `R [D, H, H]` (hidden→hidden), and `bias [D, H]`,
  all `RecurrentUniform`-initialized (PyTorch's `U(−1/√H, 1/√H)`). The single gate
  means there is no PyTorch↔ONNX gate-reorder.
- **`bias`.** A single owned bias `[D, H]` is fed to the op as
  `B = concat(bias, zeros)` on axis 1 (so the ONNX input-bias `Wb` carries it and
  the recurrent-bias `Rb` is 0 — the two ONNX/PyTorch biases collapse into one, as
  Keras/Flax do; a ported PyTorch RNN folds `b_ih + b_hh` into this). `bias: false`
  passes no bias to the op.
- **`numLayers` stacking.** Builds `numLayers` RNN ops in sequence, feeding each
  layer's output sequence (reshaped `[L, D, N, H] → [L, N, D·H]`) as the next
  layer's input, and concatenating each layer's final state on the leading axis to
  form the `[D·numLayers, N, H]` `hN`. Inter-layer dropout (PyTorch
  `num_layers > 1`) is **not** baked in — compose the `Dropout` module between
  stacked `Recurrent.RNN` calls if wanted.
- **`batchFirst`.** Realized by transposing `[N, L, …] → [L, N, …]` in-graph
  before the stack and transposing `y` back after; the op **always** runs at
  `layout=0` (ORT-CPU rejects `layout=1` and autodiff only supports `layout=0`).
  `hN` stays `[D·numLayers, N, H]`.
- **Initial state.** The baseline ships a zeroed `h_0` (an omitted op input; ONNX
  zero-fills) — no caller-supplied `initial_h` and no stateful carry-across-calls.
- **Autodiff caveat (loud).** Only **single-direction (forward or reverse), tanh,
  `layout=0`** RNNs are **trainable**. `RnnNonlinearity.Relu` and
  `RnnDirection.Bidirectional` **build and run for forward inference but throw
  AD003 in back-propagation through time** — they are intentionally exposed (not
  gated), documented as **inference-grade**. (Forward and reverse tanh are the
  trainable corner; relu and bidirectional are not.)
- **No QEE values.** RNN has no QEE step values, so closed-form / value checks run
  on the **ORT backend**, not the QEE value path.

#### `Recurrent.LSTM`

The Long Short-Term Memory layer is the static helper `Recurrent.LSTM`, sharing the
RNN infrastructure (the same weight-ownership over the ONNX `[num_dir, …]` layout,
the shared `RecurrentUniform` init, the `RnnDirection` enum, and the
`numLayers`/`batchFirst`/`bias`/zeroed-state patterns). The gate recurrence is
fixed (sigmoid gates, tanh cell — there is **no `nonlinearity` knob**):

```
i = σ(W_i·x + R_i·h + b_i)     o = σ(W_o·x + R_o·h + b_o)
f = σ(W_f·x + R_f·h + b_f)     c̃ = tanh(W_c·x + R_c·h + b_c)
C_t = f ⊙ C_{t-1} + i ⊙ c̃      H_t = o ⊙ tanh(C_t)
```

```csharp
// Returns the full output sequence y plus the final hidden AND cell states.
// Defaults match PyTorch nn.LSTM.
(Tensor<float32> y, Tensor<float32> hN, Tensor<float32> cN) = Recurrent.LSTM(
    Tensor<float32> x,
    long hiddenSize,                              // H — required (no default)
    RnnDirection direction  = RnnDirection.Forward,
    int  numLayers  = 1,
    bool batchFirst = false,
    bool bias       = true);
```

- **Return contract.** Returns the **three-tuple** `(y, hN, cN)`: `y` is the full
  output sequence in PyTorch layout `[L, N, D·H]` (or `[N, L, D·H]` when
  `batchFirst`); `hN` is the final hidden state and `cN` the final **cell** state,
  each `[D·numLayers, N, H]` (batch-second regardless of `batchFirst`, matching
  PyTorch). Slice `y[-1]` or read `hN` for "last output only"; ignore `hN`/`cN` for
  "sequence only" — the tuple covers Keras `return_sequences`/`return_state`.
- **Weights & init.** Per layer, with `D = direction == Bidirectional ? 2 : 1`:
  `W [D, 4H, in]` (input→hidden), `R [D, 4H, H]` (hidden→hidden), and a single owned
  bias `[D, 4H]`, all `RecurrentUniform`-initialized with the explicit hidden size
  so the bound is PyTorch's `U(−1/√H, 1/√H)` — **not** `1/√(4H)`.
- **Gate order (port note).** The four gate blocks are packed in the **ONNX-native
  `i, o, f, c` order** — the only layout the underlying op understands — with **no**
  reorder shim. Because the init is uniform across gates, the gate order is
  **unobservable at initialization** (permuting the gate blocks of a uniform-random
  tensor yields a statistically identical tensor), so there is no correctness or
  training-dynamics difference for a from-scratch model. The reorder matters only
  when importing pretrained PyTorch weights (out of scope): PyTorch `nn.LSTM` packs
  `i, f, g(=c), o`, so on import permute the `4H` rows `i,f,g,o → i,o,f,g`, and split
  + sum PyTorch's two `4H` biases (`b_ih`, `b_hh`) into the single `4H` owned bias.
- **`bias`.** A single owned bias `[D, 4H]` is fed to the op as
  `B = concat(bias, zeros)` on axis 1 (`[D, 8H]`, so the ONNX input-bias `Wb`
  carries it and the recurrent-bias `Rb` is 0 — the two ONNX/PyTorch biases collapse
  into one). `bias: false` passes no bias to the op.
- **`numLayers` / `batchFirst`.** Identical to `Recurrent.RNN`: each layer's output
  sequence (reshaped `[L, D, N, H] → [L, N, D·H]`) feeds the next layer's input, and
  each layer's final `hN`/`cN` are concatenated on the leading axis; `batchFirst` is
  realized by an in-graph transpose around a `layout=0` op (`hN`/`cN` stay
  `[D·numLayers, N, H]`).
- **Initial state.** `h_0` **and** `c_0` default to zero (omitted op inputs; ONNX
  zero-fills). Peephole `P` is null. No caller-supplied initial state or stateful
  carry-across-calls.
- **Autodiff caveat (loud).** Single-direction (forward or reverse), `layout=0`,
  default-activation LSTM is **trainable** end-to-end through the `TrainingRig` (the
  rig scheduler fix landed alongside RNN). `RnnDirection.Bidirectional` **builds and
  runs for forward inference / ONNX export but throws AD003 in back-propagation
  through time** — it is intentionally exposed (not gated), documented as
  **inference-grade**. Peephole, `input_forget`, `clip`, custom activations and
  variable-length `sequence_lens` are reachable at the core op but all throw AD003
  in BPTT, so they are **not exposed** by the layer.
- **No QEE values.** Like RNN, LSTM has no QEE step values — closed-form / value
  checks run on the **ORT backend**.

#### `Recurrent.GRU`

The Gated Recurrent Unit layer is the static helper `Recurrent.GRU`, sharing the
RNN/LSTM infrastructure (the same weight-ownership over the ONNX `[num_dir, …]`
layout, the shared `RecurrentUniform` init, the `RnnDirection` enum, and the
`numLayers`/`batchFirst`/`bias`/zeroed-state patterns). It has **two** gates instead
of the LSTM's three and **no separate cell state** — a single hidden vector carries
memory. The gate recurrence is fixed (sigmoid gates, tanh candidate — there is **no
`nonlinearity` knob**):

```
z = σ(W_z·x + R_z·h + b_z)          # update gate
r = σ(W_r·x + R_r·h + b_r)          # reset gate
ĥ = tanh(W_h·x + r ⊙ (R_h·h) + b_h) # candidate (reset-after form; see linearBeforeReset)
H_t = (1 − z) ⊙ ĥ + z ⊙ H_{t-1}     # blend candidate with previous hidden
```

```csharp
// Returns the full output sequence y plus the final hidden state hN (no cell state).
// Defaults match PyTorch nn.GRU (including reset-after via linearBeforeReset: true).
(Tensor<float32> y, Tensor<float32> hN) = Recurrent.GRU(
    Tensor<float32> x,
    long hiddenSize,                              // H — required (no default)
    RnnDirection direction  = RnnDirection.Forward,
    int  numLayers          = 1,
    bool batchFirst         = false,
    bool bias               = true,
    bool linearBeforeReset  = true);              // reset-after (PyTorch / cuDNN); see below
```

- **Return contract.** Returns the **two-tuple** `(y, hN)` — the strict subset of
  LSTM's `(y, hN, cN)` with no cell state. `y` is the full output sequence in PyTorch
  layout `[L, N, D·H]` (or `[N, L, D·H]` when `batchFirst`); `hN` is the final hidden
  state `[D·numLayers, N, H]` (batch-second regardless of `batchFirst`, matching
  PyTorch). Slice `y[-1]` or read `hN` for "last output only"; ignore `hN` for
  "sequence only" — the tuple covers Keras `return_sequences`/`return_state`.
- **`linearBeforeReset` (the reset-before-vs-after knob).** Selects how the reset gate
  enters the candidate. The default **`true`** applies the reset **after** the
  recurrent matmul — `ĥ = tanh(W_h·x + r ⊙ (R_h·h + Rb_h) + Wb_h)` — matching **PyTorch
  `nn.GRU`, Keras `reset_after=True`, Flax, and cuDNN** numerically (the de-facto
  standard, and what a porting user expects). `linearBeforeReset: false` applies the
  reset **before** the matmul — `ĥ = tanh(W_h·x + (r ⊙ h)·R_hᵀ + Rb_h + Wb_h)` — the
  original Cho-et-al. v1 form and the ONNX op's **own** default (the outlier). The two
  forms are numerically distinct with the same weights; **both are trainable** (autodiff
  supports either).
- **Weights & init.** Per layer, with `D = direction == Bidirectional ? 2 : 1`:
  `W [D, 3H, in]` (input→hidden), `R [D, 3H, H]` (hidden→hidden), and a single owned
  bias `[D, 3H]`, all `RecurrentUniform`-initialized with the explicit hidden size so
  the bound is PyTorch's `U(−1/√H, 1/√H)` — **not** `1/√(3H)`.
- **Gate order (port note).** The three gate blocks are packed in the **ONNX-native
  `z, r, h` order** — the only layout the underlying op understands — with **no**
  reorder shim. Because the init is uniform across gates, the gate order is
  **unobservable at initialization** (permuting the gate blocks of a uniform-random
  tensor yields a statistically identical tensor), so there is no correctness or
  training-dynamics difference for a from-scratch model. The reorder matters only when
  importing pretrained PyTorch weights (out of scope): PyTorch `nn.GRU` packs
  `r, z, n(=h)` (Keras `r, z`), so on import swap the first two `3H` gate blocks
  (`r,z,n → z,r,h`; the candidate block stays last), and map PyTorch's two `3H` biases
  (`b_ih`, `b_hh`) onto `Wb`/`Rb` — exact for the default reset-after
  (`linearBeforeReset: true`) form.
- **`bias`.** A single owned bias `[D, 3H]` is fed to the op as
  `B = concat(bias, zeros)` on axis 1 (`[D, 6H]`, so the ONNX input-bias `Wb` carries
  it and the recurrent-bias `Rb` is 0 — the two ONNX/PyTorch biases collapse into one).
  With `linearBeforeReset: true` (the default) the single `Wb` bias is numerically
  equivalent to PyTorch's `b_ih + b_hh` sum. `bias: false` passes no bias to the op.
- **`numLayers` / `batchFirst`.** Identical to `Recurrent.RNN`/`Recurrent.LSTM`: each
  layer's output sequence (reshaped `[L, D, N, H] → [L, N, D·H]`) feeds the next layer's
  input, and each layer's final `hN` is concatenated on the leading axis; `batchFirst`
  is realized by an in-graph transpose around a `layout=0` op (`hN` stays
  `[D·numLayers, N, H]`).
- **Initial state.** `h_0` defaults to zero (an omitted op input; ONNX zero-fills);
  there is no cell state. No caller-supplied initial state or stateful
  carry-across-calls.
- **Autodiff caveat (loud).** Single-direction (forward or reverse), `layout=0`,
  default-activation GRU is **trainable** end-to-end through the `TrainingRig` (the rig
  scheduler fix landed alongside RNN), and **both `linearBeforeReset` forms are
  differentiable**. `RnnDirection.Bidirectional` **builds and runs for forward inference
  / ONNX export but throws AD003 in back-propagation through time** — it is intentionally
  exposed (not gated), documented as **inference-grade**. `clip`, custom activations and
  variable-length `sequence_lens` are reachable at the core op but all throw AD003 in
  BPTT, so they are **not exposed** by the layer.
- **No QEE values.** Like RNN/LSTM, GRU has no QEE step values — closed-form / value
  checks run on the **ORT backend**.

#### Recurrent cells (single-step) — `Recurrent.RNNCell` / `LSTMCell` / `GRUCell`

Single-step siblings of the layers: each computes **one** timestep of the recurrence,
taking the previous hidden state(s) in and returning the new one(s), so a user can
hand-unroll a custom loop (scheduled sampling, attention-augmented decoders, beam
search) that the whole-sequence layer cannot express.

```csharp
// h' = act(W·x + R·h + b). Mirrors PyTorch nn.RNNCell.
Tensor<float32> Recurrent.RNNCell(
    Tensor<float32> x, Tensor<float32> h, long hiddenSize,
    RnnNonlinearity nonlinearity = RnnNonlinearity.Tanh, bool bias = true);   // -> h'

// The four gates over (h, c). Mirrors PyTorch nn.LSTMCell.
(Tensor<float32> h, Tensor<float32> c) Recurrent.LSTMCell(
    Tensor<float32> x, Tensor<float32> h, Tensor<float32> c,
    long hiddenSize, bool bias = true);                                       // -> (h', c')

// reset/update/candidate over h. Mirrors PyTorch nn.GRUCell.
Tensor<float32> Recurrent.GRUCell(
    Tensor<float32> x, Tensor<float32> h, long hiddenSize,
    bool bias = true, bool linearBeforeReset = true);                         // -> h'
```

Each is implemented as the matching ONNX op run at **sequence-length 1** (the previous
state is passed as the op's `initial_h`/`initial_c`; the `num_dir` axis is stripped so
state is a clean `[N, H]`), so the gate math, the `RecurrentUniform` init
(`U(−1/√H, 1/√H)`), the gate packing and the bias collapse are **identical** to the
layers — a cell is definitionally one step of the layer. The initial state is a
**required** tensor input (seed step 0 with an explicit zero tensor — the user owns the
loop). The default (tanh) `RNNCell`, `LSTMCell`, and both `linearBeforeReset` forms of
`GRUCell` are fully **trainable**; `RNNCell` with `RnnNonlinearity.Relu` is
inference-only (BPTT throws AD003), the same limit as `Recurrent.RNN(Relu)`.

### BatchNorm (+ BatchNorm1d / 2d / 3d aliases)

```csharp
// rank-generic: channel is axis 1; reduces over batch + every spatial axis.
BatchNorm.Call(Scalar<float32> momentum, Scalar<float32> epsilon,
               Scalar<bit> training, Scalar<bit> affine,
               Scalar<bit> trackRunningStats, Tensor<float32> x)
```

One rank-generic module covers PyTorch's per-dim `BatchNorm1d/2d/3d` contracts:
**ranks 2–5** are supported — `[N, C]`, `[N, C, L]`, `[N, C, H, W]`,
`[N, C, D, H, W]`. The reduction set `{0} ∪ {2..rank-1}` (batch + spatial,
skipping the channel axis) and the per-channel broadcast shape `[1, C, 1, …, 1]`
are derived from the input's runtime rank in-graph (the same `Range`-from-rank
idiom as `LayerNorm`), so the `[N, C, L]` rank-3 form now works — the old
`BatchNorm1d`'s rank-2-only restriction is lifted.

- `training = true`: normalizes with **batch** statistics (biased variance) and
  EMA-updates the running stats via `Globals.StateUpdate` (ONNX/Keras momentum
  convention: `running = running * momentum + batch * (1 - momentum)`).
- `training = false`: normalizes with the **running** statistics when
  `trackRunningStats = true`, or with the eval **batch** statistics when
  `trackRunningStats = false` (PyTorch `track_running_stats=False` semantics).
  The state update is gated to a no-op, so eval passes always leave the running
  stats untouched regardless of `trackRunningStats`.
- `affine = true`: applies `y = gamma * x̂ + beta`; `affine = false`: returns the
  normalized `x̂` directly. gamma (`Ones`) and beta (`Zeros`) are **always**
  created as trainable params (so the trainable-param struct shape is independent
  of the bits, mirroring `Linear`'s always-present bias); when `affine = false`
  they simply receive zero gradient.
- The running mean/variance are module-owned state which `TrainingRig` threads as
  **model state** (`checkpoint.ModelState`), not trainable params.
- **Defaults**: `momentum = 0.9`, `epsilon = 1e-5`, `affine = true`,
  `trackRunningStats = true`; `training` is the mode switch (no default).
- **Port note**: Shorokoo `momentum` follows the ONNX/Keras sense (it weights the
  *retained* running stat), so to port a PyTorch `BatchNorm(momentum = p)` use
  Shorokoo `momentum = 1 − p` (the default `0.9` ≡ PyTorch `0.1`). The running
  variance EMA uses the **biased** estimator (ONNX/Keras/Flax), a minor numeric
  divergence from PyTorch's Bessel-corrected `running_var`.
- **Rig pipeline required**: graphs containing `StateUpdate` links execute
  through the training pipeline (`TrainingRig`), not the plain inference
  executor — run BatchNorm models via a rig even for eval-mode passes.

```csharp
// Thin aliases over BatchNorm, preserving the 4-arg (momentum, epsilon,
// training, x) shape with affine = trackRunningStats = true:
BatchNorm1d.Call(momentum, epsilon, training, x)  // [N, C] or [N, C, L]
BatchNorm2d.Call(momentum, epsilon, training, x)  // [N, C, H, W]   (NCHW)
BatchNorm3d.Call(momentum, epsilon, training, x)  // [N, C, D, H, W] (NCDHW)
```

The `1d/2d/3d` aliases are named entry points that forward to the generic
`BatchNorm` with `affine` and `trackRunningStats` defaulted on (rank is still
inferred at runtime). Use the generic `BatchNorm` for the full toggle surface.

### LayerNorm / RMSNorm / GroupNorm / InstanceNorm

```csharp
LayerNorm.Call(Scalar<int64> normalizedDims, Scalar<float32> epsilon, x)  // last n dims
RMSNorm.Call(Scalar<int64> normalizedDims, Scalar<float32> epsilon, x)    // last n dims, no mean-subtraction

// Rank-generic feature normalizers over [N, C, *spatial] (channel = axis 1),
// any rank >= 3 ([N,C,L], [N,C,H,W], [N,C,D,H,W], ...) — rank is inferred at runtime:
GroupNorm.Call(Scalar<int64> numGroups, Scalar<bit> affine, Scalar<float32> epsilon, x)
InstanceNorm.Call(Scalar<bit> affine, Scalar<float32> epsilon, x)

// Thin rank-named aliases over InstanceNorm with affine defaulted OFF (PyTorch's
// InstanceNorm default), preserving the 2-arg (epsilon, x) call shape:
InstanceNorm1d.Call(Scalar<float32> epsilon, x)  // [N, C, L]
InstanceNorm2d.Call(Scalar<float32> epsilon, x)  // [N, C, H, W]    (NCHW)
InstanceNorm3d.Call(Scalar<float32> epsilon, x)  // [N, C, D, H, W] (NCDHW)
```

Built in-graph from elementwise/reduce ops (the ONNX normalization ops take
epsilon/numGroups as static attributes, which would forbid `[Hyper]` values).
gamma/beta are trainable (`Ones`/`Zeros`); LayerNorm's are shaped like the
normalized trailing dims, GroupNorm/InstanceNorm's are per-channel (broadcast
`[1, C, 1, …, 1]`, sized to the runtime rank).
`RMSNorm` (`y = x / √(mean(x²) + ε) · gain`) skips the mean-subtraction and the
bias, keeping only a trainable gain — the normalization used by most modern LLMs.

`GroupNorm` and `InstanceNorm` are the same computation differing only in the
number of channel-groups (Wu & He 2018): `GroupNorm(numGroups = 1)` recovers
LayerNorm-over-CHW and `GroupNorm(numGroups = C)` recovers `InstanceNorm`. Both
reduce over each per-(sample, group/channel) region's channels and **every**
spatial axis using the **biased** variance, and both expose an `affine` toggle
(build-both-branches-then-`IfElse`-select, like `Linear`'s `useBias`): gamma/beta
are always created as trainable params but receive zero gradient when
`affine = false`.

- **`GroupNorm`**: `affine` is a required `[Hyper]` bit, conceptually defaulted
  **on** at call sites (PyTorch/Keras/Flax default `affine=True`). **Signature
  change:** `affine` is a new leading bit after `numGroups` — old
  `GroupNorm.Call(numGroups, epsilon, x)` becomes
  `GroupNorm.Call(numGroups, Scalar(true), epsilon, x)`. `C` must be divisible by
  `numGroups`, else the `[N, G, -1]` reshape fails at concretization.
- **`InstanceNorm`**: `affine` defaults **off** in the `1d/2d/3d` aliases,
  matching PyTorch's `affine=False` InstanceNorm default — the *opposite* of
  GroupNorm's on default — because the canonical (style-transfer) use case
  normalizes without a learnable affine. Pass `affine = true` to the generic
  `InstanceNorm` to opt in. **Note:** this changes the previous rank-4-only
  `InstanceNorm2d`, which hardcoded the affine **on**; it is now off by default
  (a deliberate PyTorch-parity fix), while keeping the same 2-arg `(epsilon, x)`
  call shape. InstanceNorm carries **no** running-stats / momentum machinery —
  its statistics are per-instance and identical at train and eval time, so it
  runs on the plain inference pipeline; for batch/running-stat normalization use
  `BatchNorm`.

### LocalResponseNorm

```csharp
LocalResponseNorm.Call(Scalar<float32> alpha, Scalar<float32> beta, Scalar<float32> k, x)  // size baked = 5
LRNHelper.Lrn(x, long size = 5, float alpha = 1e-4f, float beta = 0.75f, float k = 1.0f)    // arbitrary size
```

Cross-channel ("brightness") normalization (Krizhevsky et al. 2012, AlexNet):
`b_c = a_c · (k + (α/size)·Σ_{c'∈window(c)} a_{c'}²)^(−β)` over `[N, C, *spatial]`
(channel = axis 1), same output shape. The module exposes `alpha`/`beta`/`k` as
hyperparameters (`k` = PyTorch's additive constant / ONNX `bias`) and **bakes the
window width `size = 5`** (the ONNX/PyTorch default) because `size` is a compile-time
ONNX attribute; for a different width use the static helper `LRNHelper.Lrn`. Defaults
`α=1e-4, β=0.75, k=1` match `nn.LocalResponseNorm(5)` exactly. **Note:** LRN is largely
**superseded by BatchNorm** — provided for AlexNet-era parity / legacy-model loading,
not as a recommended default. Porting from TensorFlow (`tf.nn.lrn`, half-width
`depth_radius`, bare `α`, different defaults) needs conversion.

### Attention / Transformer

```csharp
// Scaled dot-product attention (no params): q/k/v are [N, H, L, d].
Attention.ScaledDotProductAttention(query, key, value, causal: false, scale: null, additiveMask: null)

// Rotary positional embedding (RoPE; no params): rotates a [N, H, L, d] tensor
// (d EVEN) by an angle proportional to sequence position. Apply to Q and K
// (NOT V) before ScaledDotProductAttention; returns the same shape.
Attention.ApplyRoPE(Tensor<float32> x, long @base = 10000)

// Multi-head attention. query [N, Lq, embedDim]; key/value [N, Lk, embedDim].
// Pass (x, x, x) for self-attention, distinct tensors for cross-attention.
MultiHeadAttention.Call(Scalar<int64> embedDim, Scalar<int64> numHeads,
                        Scalar<bit> useBias, Scalar<bit> causal, query, key, value)

// Pre-LayerNorm encoder layer: h = x + MHA(LN(x)); out = h + FFN(LN(h)).
TransformerEncoderLayer.Call(Scalar<int64> embedDim, Scalar<int64> numHeads,
                             Scalar<int64> ffnDim, Scalar<bit> useBias, x)

// Pre-LayerNorm decoder layer (3 sublayers: masked self-attn + cross-attn + FFN).
// tgt [N, Lt, embedDim]; memory [N, Lm, embedDim] (the encoder output).
TransformerDecoderLayer.Call(Scalar<int64> embedDim, Scalar<int64> numHeads,
                             Scalar<int64> ffnDim, Scalar<bit> useBias, tgt, memory)
```

All built from autodiff-supported primitives (MatMul / Softmax / Transpose /
Where), so they train end-to-end — the fused ONNX `Attention` op has no gradient
rule. `MultiHeadAttention` owns four `XavierUniform` projections (q/k/v/out) with
optional zero biases; the causal mask is a constant gated by the runtime
`causal` flag. `TransformerEncoderLayer` composes `LayerNorm` + `MultiHeadAttention`
and a GELU FFN (the FFN uses explicit token-wise MatMuls, since `Linear` would
flatten the sequence into the feature axis on a `[N, L, E]` input). No PyTorch
backwards-compat surface (`need_weights`/`kdim`/`batch_first`/…) — Shorokoo is
explicit and batch-first.

`Attention.ApplyRoPE` adds **rotary positional embedding** (RoPE; Su et al. 2021):
a parameter-free rotation that encodes *relative* position inside the attention
dot-product. It rotates each per-head query/key vector (a `[N, H, L, d]` tensor
with **`d` even**) by `m·θ_i`, where `m` is the token's sequence position and
`θ_i = base^{-2i/d}` (`base` default `10000`, HF `rope_theta`), using the
GPT-NeoX / HuggingFace **half-split** layout and rotate-half trick:
`RoPE(x) = x·cos(mθ) + rotateHalf(x)·sin(mθ)`, with
`rotateHalf(x) = concat(-x[…, d/2:], x[…, :d/2])`. The cos/sin tables are built
in-graph from the input's own `L` (axis -2) and `d` (axis -1) and broadcast over
the leading `[N, H]`. Apply it to **Q and K only** (never V), *before*
`ScaledDotProductAttention`; the output keeps the same `[N, H, L, d]` shape and,
being an orthogonal rotation, preserves each vector's norm exactly.

`TransformerDecoderLayer` is the pre-LN **cross-attention decoder block** (the
three-sublayer decoder from "Attention Is All You Need"). It mirrors
`TransformerEncoderLayer`'s residual/LN/FFN structure, inserting a cross-attention
sublayer: `h = tgt + MHA_self(LN(tgt), causal: true)` (masked self-attention),
`h2 = h + MHA_cross(LN(h), memory, memory)` (encoder–decoder attention: query =
`LN(h)`, key = value = the **raw** encoder `memory`, non-causal), then
`out = h2 + FFN(LN(h2))` (the same GELU FFN as the encoder layer). It takes two
inputs — `tgt [N, Lt, embedDim]` and `memory [N, Lm, embedDim]` — and because the
query length `Lt` and key/value length `Lm` may differ, it exercises
`MultiHeadAttention`'s distinct-k/v (separate kdim/vdim) cross-attention path. The
self-attention is hard-coded causal; `memory` is fed unnormalized (expected to be
the already-LayerNorm'd encoder-stack output, matching PyTorch).

### PReLU / GLU

```csharp
PReLU.Call(x)                       // y = relu(x) - a*relu(-x); single shared [1] learnable slope (init 0.25)
PReLUChannelwise.Call(x)            // same formula, but a SEPARATE [C] learnable slope per channel (init 0.25)
GatedLinear.GLU(x, axis: -1)        // splits x in two halves [a, b] along axis -> a * sigmoid(b)
GLU.Call(x)                         // param-free module form of GatedLinear.GLU with axis fixed at -1
```

`PReLU` carries one shared trainable slope (PyTorch `num_parameters=1`).
`PReLUChannelwise` is the per-channel variant (PyTorch `num_parameters=C`): it owns
a separate trainable slope for each channel — a `[C]` slope vector read from the
input's channel axis (axis 1) in-graph and broadcast as `[1, C, 1, …, 1]` over the
batch/spatial dims. It is rank-generic (`[N, C]`, `[N, C, L]`, `[N, C, H, W]`, …;
input rank ≥ 2, channels on axis 1) and its slopes init to `0.25` just like the
shared `PReLU` — so a fresh `PReLUChannelwise` equals a fresh `PReLU` numerically
until training pulls the per-channel slopes apart. (Per He et al. 2015, the PReLU
slope should not be weight-decayed; Shorokoo's optimizers apply decay uniformly, so
this is a usage caveat, not enforced.) `GLU` is a param-free static helper (like
`Pooling`) and requires an even size along `axis`. The `GLU` **module** (`GLU.Call(x)`)
is a thin param-free wrapper over `GatedLinear.GLU` with the split axis fixed at `-1`
(PyTorch `nn.GLU(dim=-1)`); use the static helper for an arbitrary split axis.

### Dropout

```csharp
Dropout.Call(Scalar<float32> ratio, Scalar<bit> training, Tensor<float32> x)
```

Training mode zeroes each element with probability `ratio` and scales survivors
by `1/(1-ratio)`; eval mode is the identity. The mask seed is fixed (42) so
builds are deterministic; gradients flow through the forward mask automatically.

### SpatialDropout

```csharp
SpatialDropout.Call(Scalar<float32> ratio, Scalar<bit> training, Tensor<float32> x)
Dropout1d.Call(Scalar<float32> ratio, Scalar<bit> training, Tensor<float32> x)  // [N, C, L]
Dropout2d.Call(Scalar<float32> ratio, Scalar<bit> training, Tensor<float32> x)  // [N, C, H, W]
Dropout3d.Call(Scalar<float32> ratio, Scalar<bit> training, Tensor<float32> x)  // [N, C, D, H, W]
```

Channel-wise (spatial) dropout over `[N, C, D1..Dn]` (channel = axis 1): one
Bernoulli draw per `(sample, channel)` drops or rescales the **entire** feature
map at once (vs `Dropout`'s per-element mask), so whole channels are zeroed or
scaled by `1/(1-ratio)` together — the regularization for strongly-correlated
conv feature maps (Tompson et al. 2015). Eval mode is the identity; the mask seed
is fixed (42). The rank is read in-graph, so `SpatialDropout` is rank-generic; the
`Dropout1d/2d/3d` aliases (PyTorch names) are thin forwarders naming the per-rank
forms. On rank-2 `[N, C]` it degenerates to elementwise `Dropout`.

### AlphaDropout / FeatureAlphaDropout

```csharp
AlphaDropout.Call(Scalar<float32> ratio, Scalar<bit> training, Tensor<float32> x)         // elementwise
FeatureAlphaDropout.Call(Scalar<float32> ratio, Scalar<bit> training, Tensor<float32> x)  // channel-wise [N,C,...]
```

SELU-paired dropout for self-normalizing networks (Klambauer et al. 2017). Instead
of zeroing dropped units, it sets them to SELU's negative saturation `α' = −λα ≈
−1.7581`, then affine-renormalizes the tensor by `out = a·x' + b` with
`a = (q + α'²·q·p)^(−1/2)`, `b = −a·p·α'` (`q = 1−ratio`), which preserves the mean
and variance **in expectation** over the mask — the self-normalizing property plain
`Dropout` (which zeros, shifting the moments) destroys. `FeatureAlphaDropout` is the
channel-wise twin: one Bernoulli draw per `(sample, channel)` drops a whole feature
map to `α'` (reusing the `[N, C, 1, …, 1]` mask shape), rank-generic over 1-D/2-D/3-D.
Both take `(ratio, training)` like `Dropout`; eval mode (`training = false`) is the
**exact** identity (gated explicitly, since the affine isn't the identity); the mask
seed is fixed (42).

### Embedding

```csharp
// indices of any shape [...] -> embeddings [..., embeddingDim]
Embedding.Call(Scalar<int64> numEmbeddings, Scalar<int64> embeddingDim,
               Scalar<int64> paddingIdx, Scalar<float32> maxNorm,
               Scalar<float32> normType, Tensor<int64> indices)
```

Gather over a trainable `[numEmbeddings, embeddingDim]` table, `Normal`-initialized
(N(0,1), PyTorch's `nn.Embedding` default).

**Knobs (all `[Hyper]` scalars; pass the sentinel to disable):**
- **`paddingIdx`** (sentinel `-1` = off): output rows whose index equals `paddingIdx`
  are masked to the zero vector, so they also receive no training gradient.
- **`maxNorm`** (sentinel `0f` = off) / **`normType`** (default `2f`, the *p* of the
  p-norm): gathered output rows whose `normType`-norm exceeds `maxNorm` are scaled
  down to `maxNorm` (shrink-only; under-cap rows untouched). `normType` is inert
  unless `maxNorm` is set.

**Two intentional divergences from PyTorch** (SSA graphs cannot mutate a weight
mid-forward): (1) `maxNorm` is *functional* — Shorokoo clamps the gathered **output**
rows and never mutates the stored weight (PyTorch renormalizes the weight in place, so
across training the stored weights diverge; per-forward/inference outputs match for
binding rows). (2) `paddingIdx` masks the **output** rather than zeroing the stored row
+ freezing its gradient imperatively; the mask routes zero gradient to pad positions,
so under the standard "pad id is reserved" convention the training effect matches (and
is more robust — the output is zero by construction).

**Choosing the initializer.** `[Module] Embedding` is hardcoded to `Normal` (PyTorch
parity). For a different initializer use the static helper
`EmbeddingHelpers.Embed(indices, numEmbeddings, embeddingDim, embeddingInit, paddingIdx,
maxNorm, normType)` — e.g. `EmbeddingHelpers.Embed(idx, V, D, s => XavierUniform.Init(s))`.
The init selector is a compile-time type (not a runtime `[Hyper]`), so it lives on a
plain-C#-argument static helper (the `Recurrent`/`Convolution`/`Pooling` precedent);
`Embed` defaults to `Normal`, strictly generalizing the module.

**Not exposed (deliberate declines).** `scale_grad_by_freq` and `sparse` are
gradient-only knobs with no forward expression and no Shorokoo IR support (absent from
Keras and Flax too).

#### EmbeddingBag

```csharp
EmbeddingBag.Bag(Tensor<int64> indices, long numEmbeddings, long embeddingDim,
                 BagMode mode = BagMode.Mean, Func<...>? embeddingInit = null,
                 long paddingIdx = -1)   // BagMode { Sum, Mean, Max }
```

Looks up a trainable `[V, D]` table (`Normal` by default) for a **2-D batch of
fixed-length bags** `indices [B, L]` and reduces each bag over axis 1 by `mode`,
returning `[B, D]` — i.e. `Embedding(indices).Reduce(mode, axis=1)`. A plain-argument
`static` helper (not a `[Module]`) because `mode` is a build-time structural choice
(the `Recurrent`/`Pooling` precedent); the owned table still trains end-to-end.

**Scope / divergences:** 2-D fixed-length input only — PyTorch's 1-D `input + offsets`
ragged form, `include_last_offset`, and `per_sample_weights` are not supported (the
ragged segment-reduce needs a SegmentSum-style op ONNX/Shorokoo lacks; rectangularize
to `[B, L]` with `paddingIdx` instead). The `[B, L, D]` intermediate is materialized (no
fused gather-reduce kernel) — the result matches PyTorch, the memory profile does not.
`paddingIdx` zeroes pad rows before the reduce: exact for `Sum`, approximate for `Mean`
(divides by full `L`, not the non-pad count), and unfaithful for `Max` with negative
embeddings (use it with `Max` only for non-negative embeddings).

### Activations

Only activations with a hyperparameter get modules:

```csharp
LeakyReLU.Call(Scalar<float32> alpha, x)  // x for x > 0, alpha * x otherwise
ELU.Call(Scalar<float32> alpha, x)        // x for x > 0, alpha * (exp(x) - 1) otherwise
```

(The ONNX LeakyRelu/Elu ops take alpha as a static attribute, so these build the
formula in-graph to allow a true `[Hyper]` alpha.)

Plain activations are tensor one-liners and need no modules: `x.Relu()`,
`x.Gelu()`, `x.Sigmoid()`, `x.Tanh()`, `x.Softmax(axis)`, `x.LogSoftmax(axis)`, …

### Pooling — plain C# helpers

ONNX pooling geometry exists only as static node attributes (no tensorized
variant), so `Pooling` is a static class of graph-building helpers with plain
C# arguments — not `[Module]`s, and the geometry is not hyperparameter-driven.
Like `Convolution`, the windowed pools come in three shapes: a **per-axis**
general form (geometry as `long[]`, spatial rank inferred from
`kernelSize.Length`), a **scalar-square** convenience overload (one scalar per
knob, rank from `x.Rank() - 2`), and per-rank `*1d/2d/3d` aliases that assert the
rank. Unlike `Convolution`, the helpers stay generic in the float element type
(pooling owns no parameters).

```csharp
// Max pooling — per-axis, scalar-square, and per-rank forms.
Pooling.MaxPool(x, kernelSize: [3, 3], stride: [2, 2], padding: [1, 1], dilation: [1, 1],
                ceilMode: false, autoPad: AutoPad.NotSet);   // long[] geometry; padding may be 2*rank ONNX begin..end
Pooling.MaxPool(x, kernelSize: 2);                            // scalar-square (rank from x.Rank()-2)
Pooling.MaxPool1d(x, [2]);  Pooling.MaxPool2d(x, [2, 2]);  Pooling.MaxPool3d(x, [2, 2, 2]);

// Average pooling — same shapes, plus countIncludePad (default false; see below).
Pooling.AvgPool(x, kernelSize: [3, 3], stride: [2, 2], padding: [1, 1],
                ceilMode: false, countIncludePad: false, autoPad: AutoPad.NotSet);
Pooling.AvgPool1d(x, [2]);  Pooling.AvgPool2d(x, [2, 2]);  Pooling.AvgPool3d(x, [2, 2, 2]);

// Lp-norm pooling — each window -> (Σ|x|^p)^(1/p); p is an int, default 2 (L2).
Pooling.LpPool(x, kernelSize: [2, 2], p: 2);                  // L2; p:1 -> Σ|x|
Pooling.LpPool1d(x, [2]);  Pooling.LpPool2d(x, [2, 2]);  Pooling.LpPool3d(x, [2, 2, 2]);

// Global pools — collapse every spatial axis: [N, C, d1..dn] -> [N, C, 1..1].
Pooling.GlobalAvgPool2d(x);
Pooling.GlobalMaxPool2d(x);
Pooling.GlobalLpPool(x, p: 2);

// MaxPool-with-indices -> MaxUnpool round-trip (scatter maxima back to their slots).
var (values, indices) = Pooling.MaxPoolWithIndices(x, [2, 2]);
Pooling.MaxUnpool(values, indices, [2, 2], outputShape: null);   // null -> formula-sized output
Pooling.MaxUnpool1d / MaxUnpool2d / MaxUnpool3d (...);

Pooling.Flatten(x, startAxis: 1);  // [N, d1, d2, ...] -> [N, d1*d2*...]
```

**The historical scalar `MaxPool2d(x, long kernelSize, …)` / `AvgPool2d(x, long
kernelSize, …)` signatures are retained verbatim** and coexist with the new
per-axis `long[]` aliases (C# overload resolution distinguishes `long` from
`long[]`), so existing `Pooling.MaxPool2d(x, 2)` call sites are unaffected.

**Defaults & conventions.** `stride` defaults to `kernelSize` (PyTorch/Keras);
per-axis `stride`/`padding`/`dilation` accept length 1 (broadcast to every axis)
or length spatialRank, and `padding` may also be length `2*spatialRank`
(ONNX `[begin₁…beginₙ, end₁…endₙ]`, allowing asymmetric pads). `LpPool`'s norm
order `p` defaults to **2** (L2) and is an **integer** — ONNX has no fractional
norm, so PyTorch's float `norm_type` is not expressible. `AvgPool`'s
`countIncludePad` defaults to **false** (divide by the count of real cells),
matching the historical 2-D helper and **diverging from PyTorch's
`count_include_pad=True`**; pass `countIncludePad: true` for the PyTorch
denominator.

**Inference-grade backward caveats.** The pools expose their full forward
attribute surface, but some gradients are restricted:

- `AvgPool` with `ceilMode: true` **throws in the backward pass** — forward /
  inference only.
- `LpPool`'s gradient **ignores** `ceilMode`, `dilation`, and `autoPad` — these
  are forward-correct but give an incorrect gradient if trained with non-default
  values.
- `MaxPool` is exact for every attribute (ties route the gradient to the first
  max); `MaxUnpool` and the global pools are fully differentiable.

**Out of scope.** Adaptive pooling (`AdaptiveAvg/MaxPool*`) beyond the
`output_size == 1` case — which **is** `GlobalAvgPool2d`/`GlobalMaxPool2d`/
`GlobalLpPool` — has no general ONNX operator, and fractional (stochastic-window)
max pooling has no core op; both are deferred.

## Losses (`Shorokoo.Modules.Losses`)

The 2-input `Inline(predictions, targets)` form of each loss returns a
`Scalar<float32>` **mean** loss and is the rig-safe default. `predictions`/`targets`
are `Tensor<float32>` unless noted. Most losses also expose **configurable knobs**
(reduction, class weights, `ignore_index`, label smoothing, `pos_weight`, `beta`)
through extra methods — see [Configurable knobs](#loss-configurable-knobs) below.

| Module | Formula (per element, then mean) | Input contract |
|---|---|---|
| `L2Loss` | `(p − t)²` | predictions, targets (MSE; reduces over axis 0 — use rank-1/flattened predictions) |
| `L1Loss` | `\|p − t\|` | predictions, targets (MAE) |
| `HuberLoss` | `0.5·e²` if `\|e\| ≤ δ`, else `δ·(\|e\| − 0.5·δ)` | `(predictions, targets, delta hyper)` — see note below |
| `SmoothL1Loss` | Huber with `δ = 1` | predictions, targets |
| `CrossEntropyLoss` | softmax cross-entropy over logits | predictions `[N, C]` logits; targets `[N]` `Tensor<int64>` class indices |
| `NLLLoss` | `−log p[target]` | predictions `[N, C]` log-probs (e.g. `x.LogSoftmax(1)`); targets `[N]` `Tensor<int64>` |
| `BCELoss` | `−(t·ln p + (1−t)·ln(1−p))` | predictions are probabilities in (0, 1), clamped to `[1e-7, 1−1e-7]` |
| `BCEWithLogitsLoss` | `max(x, 0) − x·t + ln(1 + e^−\|x\|)` | predictions are raw logits (stable sigmoid+BCE) |
| `KLDivLoss` | `(1/N)·Σ p·(log p − log q)` (batchmean) | predictions are **log**-probs (log q), targets are probs (p); `p·log p = 0` at `p = 0` |
| `LogCoshLoss` | `log(cosh(p − t))` (stable `\|d\| + softplus(−2·\|d\|) − log 2`) | predictions, targets (hyperparameter-free Huber: ≈ `d²/2` small, ≈ `\|d\| − log 2` large; overflow-free) |
| `PoissonNLLLoss` | `exp(p) − t·p` (`logInput=true`); else `p − t·log(p + eps)` | predictions are the **log-rate** `log λ` (default) or rate `λ` (`logInput=false`); targets are `Tensor<float32>` counts |
| `HingeLoss` | `max(0, 1 − t·p) = relu(1 − t·p)` | predictions are raw scores; **targets MUST be `±1`** (`+1`/`−1`), **not** auto-converted from `0/1` — map `0/1` upstream with `2·t − 1` |
| `SquaredHingeLoss` | `max(0, 1 − t·p)²` | as `HingeLoss` (**`±1` targets**); penalises margin violations quadratically (smooth at the boundary) |
| `BinaryFocalLoss` | `α_t · (1 − p_t)^γ · ce` (`ce` = stable BCE-with-logits; `p_t = p·t + (1−p)·(1−t)`, `p = σ(x)`) | predictions are raw **logits**; targets are binary `{0, 1}`. `α`/`γ` baked C# floats (defaults `0.25`/`2`, torchvision parity); `α = −1` disables α-weighting |

**HuberLoss vs SmoothL1Loss and the rig**: `HuberLoss`'s `delta` hyperparameter
makes its `ComputationGraph` a 3-input graph, but `TrainingRig`'s loss contract
is exactly `(predictions, targets)` — so `HuberLoss.ComputationGraph` cannot be
handed to a rig directly. Use `SmoothL1Loss` (delta = 1, 2-input) there, or
compose `HuberLoss.Inline(predictions, targets, Scalar(d))` inside your own
2-input loss module.

<a id="loss-configurable-knobs"></a>
### Configurable knobs (`Reduced` / `PerElement`)

The knobs PyTorch exposes on these losses are surfaced as **build-time C#
arguments** (baked into the graph, not `[Hyper]`s) on two extra methods, leaving
the rig-safe `Inline(predictions, targets)` untouched:

- **`Reduced(…, LossReduction reduction = Mean)`** — returns a `Scalar<float32>`
  for `Mean`/`Sum` reduction.
- **`PerElement(…)`** — returns the per-element `Tensor<float32>` (the
  `reduction = None` form; C# can't overload on return type, hence a separate
  method). For `CrossEntropyLoss`/`NLLLoss` the per-element tensor is **zero at
  `ignore_index` positions** (PyTorch/ONNX semantics).

`LossReduction` (in `Shorokoo.Modules.Losses`) is `None | Mean | Sum`, mapping to
the ONNX op's `"none"/"mean"/"sum"` (CE/NLL) or `ReduceKind.Mean/.Sum` (the
regression/BCE losses). `Reduced(..., reduction: None)` throws — use `PerElement`.

| Loss | `Reduced` / `PerElement` extra knobs |
|---|---|
| `CrossEntropyLoss` | `Tensor<float32>? weight` (per-class `[C]`), `long? ignoreIndex`, `float labelSmoothing = 0`, `reduction` |
| `NLLLoss` | `Tensor<float32>? weight`, `long? ignoreIndex`, `reduction` |
| `BCEWithLogitsLoss` | `Tensor<float32>? posWeight`, `reduction` |
| `HuberLoss` | `Scalar<float32> delta` (first arg, live), `reduction` |
| `SmoothL1Loss` | `float beta` (first arg, baked; PyTorch default 1.0), `reduction` |
| `PoissonNLLLoss` | `bool logInput = true`, `bool full = false`, `float eps = 1e-8` (used only when `logInput=false`), `reduction` |
| `BinaryFocalLoss` | `float alpha = 0.25` (`−1` disables α-weighting), `float gamma = 2.0`, `reduction` |
| `L1Loss` / `L2Loss` / `LogCoshLoss` / `HingeLoss` / `SquaredHingeLoss` | `reduction` |

**`label_smoothing` (CE only)** blends the hard target with the uniform
distribution: `loss = (1−α)·NLL + α·(−(1/K)·Σ_k log p_k)`. It is built in-graph
from `LogSoftmax + NegativeLogLikelihoodLoss` primitives (ONNX SoftmaxCrossEntropyLoss
has no such attribute), threading `weight`/`ignoreIndex` through **both** terms;
`α = 0` short-circuits to the exact single-op fast path. Because it adds no graph
input and stays scalar, `labelSmoothing` (like `ignoreIndex` and a `Mean`/`Sum`
`reduction`) is **rig-safe** on a wrapper module (see below).

**SmoothL1 ↔ Huber bridge**: `SmoothL1(e; β) = HuberLoss(δ = β) / β`. So
`SmoothL1Loss.Reduced(beta, p, t)` equals `HuberLoss(delta = beta)` divided by
`beta`; at `β = 1` it reproduces `SmoothL1Loss.Inline` exactly. `HuberLoss`'s
`delta` stays a **live `[Hyper]`** (schedulable); SmoothL1's `beta` is a **baked
C# float**. If you need a *live* transition point, use `HuberLoss`'s `[Hyper]
delta` and divide by `delta` yourself.

**LogCosh stability**: `LogCoshLoss` computes the algebraically-identical but
overflow-free `log(cosh(d)) = |d| + softplus(−2·|d|) − log 2` (the naive
`log((e^d + e^−d)/2)` overflows for `|d| ≳ 89` in float32). It is
hyperparameter-free — the L2→L1 crossover (≈ `d²/2` small, ≈ `|d| − log 2` large)
is built into the function shape, so there is no `δ`/`β` to tune.

**PoissonNLL `logInput`/`full`**: `Inline` hardcodes PyTorch's stable
`logInput=true` form `exp(p) − t·p` (the prediction is `log λ`). The Keras
`Poisson` form `p − t·log(p + ε)` is reachable via `Reduced(p, t, logInput: false,
eps: 1e-7f)` (keep `p > 0`; `eps` only guards exact `p = 0`). `full=true` adds
Stirling's approximation of the dropped `log(t!)` constant
(`t·log t − t + 0.5·log(2π·t)` for `t > 1`, else 0), matching PyTorch; it is
computed with a clamped `max(t, 1)` inside the logs so the `Where`-discarded lane
stays finite (no `0·log 0` NaN at `t = 0`).

#### Which knobs reach the rig

The rig hands the loss graph exactly **two tensor inputs** and wants a
**`Scalar<float32>`** out. So:

- **Rig-safe** (no extra input, stays scalar): `reduction = Mean`/`Sum`,
  `ignoreIndex`, `labelSmoothing` — but only via a **wrapper module** (the
  generated `ComputationGraph` always uses the bare `Inline`). Author a tiny
  2-input `[Module]` whose `Inline` calls `Reduced(...)` with the knobs baked:

  ```csharp
  [Module]
  public partial class CeIgnorePad   // CE that ignores label 0 (padding)
  {
      public static Scalar<float32> Inline(Tensor<float32> logits, Tensor<int64> targets)
          => CrossEntropyLoss.Reduced(logits, targets, ignoreIndex: 0L,
                                      labelSmoothing: 0.1f, reduction: LossReduction.Sum);
  }
  // ...then hand CeIgnorePad.ComputationGraph to the rig.
  ```

- **Direct-only** (`weight`/`posWeight` add a 3rd tensor input; `None` returns a
  tensor): reachable on `Reduced`/`PerElement`, but the **default rig path can't
  bind the extra input**. To train with a class `weight`/`posWeight`, **bake it
  as a graph constant** inside a 2-input wrapper module (the same recipe as the
  HuberLoss `delta` note):

  ```csharp
  [Module]
  public partial class WeightedCe     // CE with class weights [1, 2, 3] baked in
  {
      public static Scalar<float32> Inline(Tensor<float32> logits, Tensor<int64> targets)
          => CrossEntropyLoss.Reduced(logits, targets,
                 weight: Tensor(new long[] { 3L }, 1f, 2f, 3f));   // constant ⇒ no extra input
  }
  // ...then hand WeightedCe.ComputationGraph to the rig.
  ```

  Because the weight tensor is a graph **constant** (not a fed input), the wrapper
  is back to two inputs and satisfies the rig contract. There is no `WithWeight`
  factory — the wrapper module is the supported recipe.

## Optimizers (`Shorokoo.Modules.Optimizers`)

Each operates on one parameter at a time (`(hypers..., currentParam, grad) ->
updatedParam`); the rig applies it per-field across the trainable parameter
struct. State tensors never appear in the signature: each is created inside the
optimizer body by an optimizer-owned state initializer — `OptimizerStateZeros` at
the parameter's shape (so param-shaped state starts zero-filled), `OptimizerScalarZeros`
for a rank-0 scalar seeded at 0 (e.g. Adam's `step`), or `OptimizerScalarOnes` for a
rank-0 scalar seeded at 1 (e.g. NAdam's running momentum product, which needs the
multiplicative identity) — and updated via `Globals.StateUpdate`. Each optimizer gets a generated named hyperparameter set
(`<Name>Hyperparameters`, e.g. `AdamOptimizerHyperparameters`) — see
[training.md](training.md) for schedules, the `HyperValue` kinds, and the
custom-optimizer authoring contract.

| Module | Update rule | Hyper defaults | State per param |
|---|---|---|---|
| `SGDOptimizer` | `p −= lr·g` | `lr 0.01` | — |
| `SGDMomentumOptimizer` | `v = μ·v + g; p −= lr·v` | `lr 0.01, μ 0.9` | velocity |
| `AdamOptimizer` | `m, v` EMAs, **bias-corrected**: `p −= lr·m̂/(√v̂ + ε)` | `lr 0.001, β1 0.9, β2 0.999, ε 1e-8` | m, v, step |
| `AdamWOptimizer` | Adam step (no bias correction) + decoupled decay `p *= 1 − lr·wd` | `lr 0.001, β1 0.9, β2 0.999, ε 1e-8, wd 1e-4` | m, v |
| `RMSpropOptimizer` | `sq = α·sq + (1−α)·g²; buf = μ·buf + g/(√sq + ε); p −= lr·buf` | `lr 0.01, α 0.99, ε 1e-8, μ 0` | squareAvg, momentumBuffer |
| `AdagradOptimizer` | `acc += g²; p −= lr·g/(√acc + ε)` | `lr 0.01, ε 1e-10` | accumulator |
| `AdamaxOptimizer` | Adam with the ∞-norm: `m` EMA; `u = max(β2·u, \|g\|+ε)`; `p −= (lr/(1−β1ᵗ))·m/u` (no bias-correction on `u`) | `lr 0.002, β1 0.9, β2 0.999, ε 1e-8` | m, u, step |
| `NAdamOptimizer` | Nesterov-Adam: `μ_t` schedule + running product `∏μ`; `m̂` blends `μ_{t+1}·m` & `(1−μ_t)·g`, bias-corrected `v`: `p −= lr·m̂/(√v̂ + ε)` | `lr 0.002, β1 0.9, β2 0.999, ε 1e-8, ψ 0.004` | m, v, step, muProduct |
| `RAdamOptimizer` | Rectified Adam: bias-corrected `m̂`; if `ρ_t > 5` rectified adaptive `p −= lr·m̂·r_t·l_t`, else un-adapted `p −= lr·m̂` (runtime `Where`) | `lr 0.001, β1 0.9, β2 0.999, ε 1e-8` | m, v, step |
| `AdadeltaOptimizer` | `sq = ρ·sq + (1−ρ)·g²; Δx = √(accΔ+ε)/√(sq+ε)·g; accΔ = ρ·accΔ + (1−ρ)·Δx²; p −= lr·Δx` (ε **inside** the √s; `lr` is a step multiplier) | `lr 1.0, ρ 0.9, ε 1e-6` | squareAvg, accDelta |
| `LionOptimizer` | Sign-momentum + decoupled decay: `u = sign(β1·m + (1−β1)·g); p −= lr·(u + wd·p); m = β2·m + (1−β2)·g` (note: `m` decayed by **β2**, β1 only in the sign blend) | `lr 1e-4, β1 0.9, β2 0.99, wd 0` | m |
| `AdafactorOptimizer` | **Non-factored** Adafactor: `β̂2ₜ = 1 − tᵗᵃᵘ; ρ = min(lr, 1/√t); α = max(ε₂, RMS(p))·ρ; V = β̂2ₜ·V + (1−β̂2ₜ)·(g²+ε₁); U = g/max(√V, ε₁); Û = U/max(1, RMS(U)/d); p = p·(1−lr·wd) − α·Û` (full param-shaped `V`, **no** row/col factoring) | `lr 0.01, τ −0.8, ε₁ 1e-30, ε₂ 1e-3, d 1.0, wd 0` | v, step |
| `LambOptimizer` | LAMB (You et al. 2019): bias-corrected Adam direction scaled by LARS's **per-tensor trust ratio** — `r = m̂/(√v̂ + ε); u = r + wd·p; trust = (‖p‖>0 ∧ ‖u‖>0) ? ‖p‖/‖u‖ : 1; p −= lr·trust·u` (ε **outside** the √; decoupled `wd` **inside** the trust numerator; ‖·‖ = `√Σx²` over the whole tensor; φ = identity). Per-tensor = layer-wise (Shorokoo runs the optimizer per parameter tensor). | `lr 1e-3, β1 0.9, β2 0.999, ε 1e-6` (LAMB, not 1e-8), `wd 0.01` | m, v, step |

- **Bias correction — Adam vs AdamW**: `AdamOptimizer` applies the textbook
  `m̂ = m/(1−β1^t)`, `v̂ = v/(1−β2^t)` correction, carrying the timestep `t` as a
  third state field — a **scalar** (one float per parameter, created by
  `OptimizerScalarZeros`) that broadcasts against `m̂`/`v̂`, not a param-shaped copy.
  `AdamWOptimizer` **omits** bias correction (no timestep), so its early-step
  behavior differs slightly from reference AdamW.
- `RMSpropOptimizer` with the default `momentum = 0` reduces to plain RMSprop;
  it always carries both state tensors.
- `AdamaxOptimizer` swaps Adam's L2 second moment for an exponentially-weighted
  infinity norm `u` (a running max) and keeps bias correction on `m` only — the
  running max is unbiased from step 1, so `u` needs none. `ε` is placed **inside**
  the max (PyTorch: `u = max(β2·u, |g|+ε)`).
- `NAdamOptimizer` adds Nesterov look-ahead to Adam via Dozat's time-varying
  momentum schedule `μ_t = β1·(1 − ½·0.96^(t·ψ))` and a **running product** `∏μ_i`.
  That product is a second **scalar** state (`muProduct`) alongside the timestep,
  seeded at **1.0** by `OptimizerScalarOnes` (the multiplicative identity — seeding
  at 0 would pin the product at 0 forever). Weight decay (coupled and decoupled) is
  out of scope, matching PyTorch's defaults.
- `RAdamOptimizer` rectifies Adam's adaptive step by `r_t` and, while the adaptive
  variance is not yet tractable (`ρ_t ≤ 5`, the first ~4–5 steps at `β2 = 0.999`),
  falls back to an un-adapted bias-corrected momentum step. The `ρ_t > 5` test is a
  runtime `Scalar<bit>` (it depends on the scalar `step` state), realized as a
  straight-line scalar `Where` selecting between the two param updates — not an `If`
  subgraph. The `m`/`v`/`step` updates are identical in both arms, so each is
  registered once, outside the selection.
- `AdadeltaOptimizer` needs no hand-set learning rate: the ratio of two
  exponentially-decaying accumulators (`√(E[Δx²]) / √(E[g²])`) self-scales the step
  and corrects its units. Two load-bearing details (both matching the paper and
  PyTorch): `ε` is **inside** both square roots (unlike Adagrad/RMSprop/Adam, which
  add `ε` after the √), and `Δx` reads the **previous** step's `accDelta` before that
  accumulator is updated from `Δx`. The leading `lr` is a step **multiplier** (default
  `1.0` ≡ the paper's lr-free method), not the primary learning rate.
- `LionOptimizer` (EvoLved Sign Momentum) takes the **sign** of a β1-blend of momentum and
  gradient, so every coordinate moves by exactly ±`lr` — the step magnitude is decoupled from
  the gradient scale. It stores **only** the momentum buffer `m` (no second moment, no
  timestep), so its optimizer-state footprint is **half** of Adam/AdamW. **Swapped beta roles
  (the easy bug):** the stored `m` is decayed by **β2** (`m = β2·m + (1−β2)·g`), while **β1**
  appears *only* inside the sign blend that forms the update direction — the opposite of Adam's
  convention. Weight decay is decoupled (AdamW-style). **Usage:** because the sign step has unit
  magnitude, a good Lion `lr` is typically **3–10× smaller** than AdamW's and its `wd` **3–10×
  larger** (effective decay ≈ `lr·wd`); the default `wd = 0` matches the reference, so opt into
  the Lion-appropriate decay deliberately. There is no `ε` (the update has no division).
- `AdafactorOptimizer` is the **non-factored** Adafactor: it keeps the algorithm's three
  rank-agnostic ideas — relative step `ρ = min(lr, 1/√t)`, parameter scaling
  `α = max(ε₂, RMS(p))·ρ`, and RMS update clipping `Û = U/max(1, RMS(U)/d)` — over a
  time-increasing second-moment decay `β̂2ₜ = 1 − t^τ`. `RMS(·)` reduces over **all** elements
  to a scalar, so the whole step is rank-agnostic. **Important divergence:** the real Adafactor
  stores only row (`[r]`) + column (`[c]`) accumulators and reconstructs the second moment as
  their rank-1 outer product — its **sublinear-memory** trick, the reason it exists. That
  factoring is **not implemented** (and is not expressible in Shorokoo's single rank-agnostic
  per-parameter optimizer graph: the factored/unfactored choice is a runtime rank branch whose
  arms would thread out differently-shaped state, which ONNX `If` cannot return). So this
  optimizer's state is a **full param-shaped** second moment `v` plus a scalar `step` — the
  **same footprint as Adam**. It reproduces Adafactor's *update dynamics* but **not** its memory
  advantage; a user reaching for Adafactor specifically to save memory gets Adam-sized state.
  `learningRate` (default `0.01`) is the **cap** on `ρ`, not a fixed step.

### TripletMarginLoss (metric / embedding learning)

For an anchor `a`, positive `p`, and negative `n`, `L = max(0, d(a,p) − d(a,n) + margin)`
with p-norm distance `d(x,y) = (Σ|x−y|^p + eps)^(1/p)` over the last axis. Knobs: `margin`
(default 1), `p` (2 ⇒ Euclidean), `eps` (1e-6), and `swap` (Balntas anchor-swap — replaces
`d(a,n)` with `min(d(a,n), d(p,n))`). `Inline` mean-reduces to a scalar; `Reduced(…,
LossReduction)` does mean|sum; `PerElement` returns the unreduced `[N]` vector.

**This is a 3-input loss** — call it `TripletMarginLoss.Call(margin, p, eps, swap, a, p, n)`
(like `MultiHeadAttention`'s q/k/v), **not** a drop-in 2-input `(pred, target)` rig loss. To
rig-train it, make the triplet loss the **tail of your model** (the model emits the three
embeddings and returns the scalar loss). `TripletMarginWithDistance` is the same objective with
a caller-supplied distance `Func<Tensor<float32>, Tensor<float32>, Tensor<float32>>` (e.g.
cosine) replacing the built-in p-norm (static helper; `.Reduced`/`.PerElement`).

### CosineEmbeddingLoss (metric learning)

Cosine-space contrastive loss over two embedding batches `x1`, `x2` (`[N, D]`) and
per-sample labels `y ∈ {+1, −1}`: `L_i = 1 − cos(x1_i, x2_i)` for `y=+1` (pull similar
pairs together), `L_i = max(0, cos(x1_i, x2_i) − margin)` for `y=−1` (push dissimilar
pairs apart). `cos` is over the last axis with PyTorch's denominator floor
`max(‖x1‖·‖x2‖, eps)`. Knobs: `margin` (default 0, affects only the `y=−1` arm) and
`eps` (default 1e-8), both `[Hyper] Scalar<float32>`. Labels must be `±1` (map `2t−1`
upstream, as with `HingeLoss`). Like `TripletMarginLoss` this is a 3-input loss —
`CosineEmbeddingLoss.Call(margin, eps, x1, x2, y)`, with the `Inline`/`Reduced`/
`PerElement` reduction triad. `CosineEmbeddingLoss.CosineSimilarity(x1, x2, eps)` is the
reusable per-row cosine-similarity primitive (PyTorch `nn.CosineSimilarity`).

## End-to-end: tiny conv net + CrossEntropyLoss + Adam

A complete classifier built from library layers and trained with `TrainingRig`
(adapted from the library's coverage tests). Layer hypers are fixed via
`Model(...)` so the model graph is inputs-only, as the rig requires:

```csharp
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using Shorokoo.Modules.Layers;
using Shorokoo.Modules.Losses;
using Shorokoo.Modules.Optimizers;
using Shorokoo.Runtime;
using static Shorokoo.Globals;

/// Conv2d(2 ch, k3, s1, p1) -> ReLU -> GlobalAvgPool -> [N, 2] logits.
[Module]
public partial class TinyConvClassifier
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var x = Conv2d.Model(Scalar(2L), Scalar(3L), Scalar(1L), Scalar(1L),
                             Scalar(1L), Scalar(1L), Scalar(true)).Call(input);
        x = x.Relu();
        x = Pooling.GlobalAvgPool2d(x);
        return x.Reshape([input.DimTensor(0), Scalar(2L)]);
    }
}
```

```csharp
// One 4-sample batch of [1, 4, 4] images, with int64 class-index targets.
var inputData  = TensorData([4L, 1L, 4L, 4L], pixels);                // float[64]
var targetData = TensorData([4L], new long[] { 0L, 1L, 0L, 1L });

var rig = TrainingRig.FromScratch(
    TinyConvClassifier.ComputationGraph,
    CrossEntropyLoss.ComputationGraph,
    AdamOptimizer.ComputationGraph,
    new NamedModelParam[] {
        new TensorDataModelParam("input", ModelParamType.InputParam, inputData) },
    new AdamOptimizerHyperparameters { LearningRate = 0.01f });  // β/ε keep defaults

var ctx      = new ComputeContext();
var compiled = ctx.Compile(rig.TrainingStepPureGraph);

static TensorDataStruct MakeBatch(string field, string structName, TensorData data) =>
    new(new TensorStructDef(
            new[] { new TensorStructFieldDef(field, DataStructure.Tensor,
                                             data.Shape.Dims.Length, data.DType) },
            structName),
        new Dictionary<string, IData> { { field, data } });

var inputBatch  = MakeBatch("input", "ModelInput", inputData);
var targetBatch = MakeBatch("targets", "Target", targetData);

var ckpt = rig.CreateDefaultCheckpoint();
for (int i = 0; i < 15; i++)
{
    var step = rig.TrainStep(ckpt, inputBatch, targetBatch, compiled);
    Console.WriteLine($"step {i}: loss {step.Loss}");
    ckpt = step.Checkpoint;
}
```

(Equivalently, batch the data as `TensorDataStruct[]` and call
`rig.Fit(inputs, targets, numEpochs)` — see
[training.md](training.md).)

## Anti-patterns

- Do not hand `HuberLoss.ComputationGraph` to `TrainingRig` (3-input graph);
  use `SmoothL1Loss` or wrap it in a 2-input module.
- Do not run a BatchNorm-containing graph through the plain inference
  executor; its `StateUpdate` links require the training pipeline.
- Do not expect different stride/padding from `ConvTranspose2d`'s hypers — its
  geometry is fixed at the ONNX defaults; use `NN.ConvTranspose` for the rest.
- Do not use `XavierUniform`/`KaimingUniform` (etc.) for rank-1 biases; they
  require rank ≥ 2 shapes.
