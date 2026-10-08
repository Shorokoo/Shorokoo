# NN library: layers

The layers of `Shorokoo.Modules`, and a small network trained end to end with them. The rest of the
library has pages of its own: [initializers.md](initializers.md) and
[losses-and-optimizers.md](losses-and-optimizers.md).

Related: [defining-models.md](defining-models.md) · [training.md](training.md) · [core-types.md](core-types.md)

## Facts

- `Shorokoo.Modules` is the baseline neural-network library: initializers, layers,
  losses and optimizers, all built from ordinary `[Module]`s /
  `[TrainableParamInitializer]`s.
- Namespaces: `Shorokoo.Modules.Initializers`, `.Layers`, `.Losses`, `.Optimizers`.
- Layers are `[Module]` classes: `Linear.Call(hypers..., x)` inline, or
  `Linear.Model(hypers...).Call(x)` to fix the hyperparameters once (see
  [defining-models.md](defining-models.md)).
- Exceptions: pooling, the generalized convolution helpers and the recurrent layers
  are plain C#-argument static helpers (`Pooling.MaxPool2d(x, 2)`,
  `Convolution.Conv(...)`, `Recurrent.RNN(x, 16)`); plain activations are tensor
  one-liners (`x.Relu()`).
- Attention is the only layer whose activations grow **quadratically** with sequence
  length — see [Sizing an attention run](#attention-memory). Any `[Module]` can be
  marked `[Module(Checkpoint = true)]` to recompute its activations in the backward
  pass — see [Activation checkpointing](#activation-checkpointing).

```bash
dotnet add package Shorokoo.Modules
```

## Layers (`Shorokoo.Modules.Layers`)

Layer hyperparameters are `[Hyper]` graph scalars; pass them as `Scalar(...)`.
Signatures below are the generated `Call` shapes — **hyperparameters first, tensor
inputs last**. `Inline` takes **tensor inputs first**, so
`Linear.Call(outFeatures, useBias, x)` is `Linear.Inline(x, outFeatures, useBias)`.

That reordering applies to `[Module]`s only. An entry spelled `Class.Method(...)` rather
than `Class.Call(...)` is a **plain-C# static helper** with the argument order shown
(tensors first, knobs after, optional parameters): `Pooling`, `Convolution`,
`Recurrent`, `Attention`, `LRNHelper`, `GatedLinear`, `EmbeddingHelpers`,
`EmbeddingBag`, and `TripletMarginWithDistance` under Losses.

<a id="nullable-hypers"></a>
**A declared `[Hyper]` default makes a *nullable* parameter, not an omittable one.** A
`[Hyper(<value>)]` parameter is exposed on `Call`/`Model` as nullable
(`Scalar<float32>?`); `null` means "use the default". A plain `[Hyper]` is non-nullable
and required. A nullable parameter can be omitted only in the **trailing** run of its
signature. In `Call` the tensors come last, so defaulted hypers are **nullable yet
positionally required**: `LocalResponseNorm.Call(null, null, null, x)`. `Model(...)`
takes no tensors, so `LocalResponseNorm.Model()` works — but `BatchNorm.Model`'s
`momentum`/`epsilon` precede three plain bits and stay required. Entries below name only
parameters with a declared default; the rest are required.

<a id="gated-parameters"></a>
**An off toggle costs nothing.** `useBias` (on `Linear`, `Bilinear`, the conv and
attention/transformer layers) and `affine` (on `BatchNorm`, `LayerNorm`, `RMSNorm`,
`GroupNorm`, `InstanceNorm`) gate a block of trainable parameters with
`bit.IfElse(withTheParams, without)`. The bit is fixed before concretization (by
`Call`/`Model`, or from the sample given to `ToConcreteArchitecture`), and the unselected
branch's parameters are **pruned**: no checkpoint field, gradient, optimizer state or
saved bytes. `Linear(useBias: false)` has one parameter; `GroupNorm(affine: false)` has
none. No need to split a model into separate classes to drop a parameter block.

- On the `Foo.ComputationGraph` + `ToConcreteArchitecture` route the bit stays a live
  input of the concrete graph and must be passed again at `Execute` — pass the value you
  concretized with. With the bit **off**, its later value is inert for these layers (their
  gate is pruned with the parameters); a tuple or shared gate would not fold and would
  read a zero stand-in. With the bit **on**, the `IfElse` is live and the opposite value
  silently takes the other branch. To drop the input,
  [`Specialize`](inference.md#hardcoding-hypers-with-specialize) the bit first. Via
  `Linear.Call(...)` the bit is a constant and there is nothing to pass. See
  [What concretization fixes](inference.md#what-concretization-fixes).
- If the **whole model graph** ends with no trainable parameters,
  `TrainingRig.FromScratch` fails with *"No trainable parameters found in the
  computation graph."* A **sub-module** pruned to none is fine.

### Linear

```csharp
// y = x @ W^T (+ b); flattens trailing dims: [N, d1, d2, ...] -> [N, d1*d2*...]
Linear.Call(Scalar<int64> outFeatures, Scalar<bit> useBias, Tensor<float32> x)
```

Weight `[outFeatures, inFeatures]` is `KaimingUniform`-initialized; bias `[outFeatures]`
is zero-initialized. `useBias = false` removes the bias term and parameter.

### Bilinear

```csharp
// y_k = x1^T A_k x2 + b_k  (PyTorch nn.Bilinear)
Bilinear.Call(Scalar<int64> in1Features, Scalar<int64> in2Features,
              Scalar<int64> outFeatures, Scalar<bit> useBias,
              Tensor<float32> x1, Tensor<float32> x2)
```

`y[..., k] = Σ_{i,j} x1[..., i]·A[k,i,j]·x2[..., j] (+ b[k])`. `A` is
`[outFeatures, in1Features, in2Features]`, `b` is `[outFeatures]`; both are initialized
from `U(±1/√in1Features)` (PyTorch's bound) — the bias is **not** zero-initialized. The
contraction is over each input's last axis; leading (batch) dims must match and are
preserved. `useBias = false` removes the bias.

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

All geometry is hyperparameter-driven via the `NN.Conv` overload that takes geometry as
int64 tensors; the exported model contains a standard ONNX Conv. Weight
`[outChannels, inChannels/groups, k(, k)]` is `KaimingUniform`-initialized; `inChannels`
is read from the input shape. `useBias = false` uses an all-zero constant bias (no
parameter). These cover **square-kernel / symmetric-pad** only; for per-axis geometry,
`auto_pad` or a non-zeros `padding_mode` use `Convolution` below.

### ConvTranspose2d — default geometry only

```csharp
ConvTranspose2d.Call(Scalar<int64> outChannels, Scalar<int64> kernelSize,
                     Scalar<bit> useBias, Tensor<float32> x)
```

Stride 1, no padding, dilation 1, group 1; weight `[inChannels, outChannels, k, k]`. For
other geometry use `NN.ConvTranspose` with static attributes, or `Convolution.ConvTranspose`.

### Convolution — generalized per-axis helpers

The static `Convolution` class exposes the **full ONNX attribute surface** (per-axis
kernel/stride/padding/dilation, asymmetric padding, `auto_pad`, `groups`,
`padding_mode`, transposed-conv `output_padding`/`output_shape`) with plain C#
arguments. Geometry is baked at build time; `inChannels` is still read in-graph from
`x.ShapeTensor()[1]`.

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

- **Weights.** Forward: `[outChannels, inChannels/groups, k…]`; transposed:
  `[inChannels, outChannels/groups, k…]`. Both `KaimingUniform`. `bias: true` adds a
  zero-initialized trainable `[outChannels]`; `bias: false` a zero constant.
- **`auto_pad`.** `SameUpper` = TF/PyTorch `"same"`; also `SameLower`, `Valid`, `NotSet`.
  Cannot be combined with a non-`Zeros` `padding_mode` or with `Causal`.
- **`groups`.** `groups == inChannels` (with `outChannels` a multiple) is depthwise.
- **`padding_mode`.** `Zeros` is the conv's own padding. `Reflect`/`Replicate`/`Circular`
  add an explicit `Pad` that is **non-differentiable** (throws in autodiff) — forward /
  inference only. `Causal` is **1-D only**: it left-pads `(k-1)*dilation` zeros so
  `out[t]` never sees future input, and is differentiable.
- **ConvTranspose `output_padding` / `output_shape`.** `output_padding` picks the output
  size when `stride > 1` is ambiguous; PyTorch's `output_padding < max(stride, dilation)`
  guard is not imposed (ONNX Runtime validates). `output_shape` sets the spatial size
  directly and overrides `output_padding`. It may exceed the full extent
  `stride * (in - 1) + output_padding + (kernel - 1) * dilation + 1` by at most one
  element (zero-extending the end; not under `auto_pad: SameUpper`); beyond that,
  concretization fails with **FW054**, naming the output_shape and the full extent — raise
  `output_padding` instead. Transposed conv has no `padding_mode`.

<a id="recurrent-layers"></a>
### Recurrent layers — `Recurrent.RNN` / `Recurrent.LSTM` / `Recurrent.GRU`

`Recurrent` is a static class of plain-C#-argument helpers, not a `[Module]`: every
knob is shape- or topology-determining and baked at build time. Weights are
`RecurrentUniform`-initialized trainable parameters (`U(−1/√H, 1/√H)`, PyTorch's bound),
and the layers train end-to-end within the limits below.

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

Common to RNN, LSTM and GRU (`D = 2` for `Bidirectional`, else `1`):

- **Layout.** `x` is `[L, N, inputSize]`, or `[N, L, inputSize]` with `batchFirst`;
  `inputSize` is read in-graph. `y` is every step's hidden state, `[L, N, D·H]` (or
  `[N, L, D·H]`). `hN` (and LSTM's `cN`) is `[D·numLayers, N, H]`, batch-second
  regardless of `batchFirst`, as in PyTorch. For the last output only, slice `y[-1]` or
  read `hN`.
- **Weights.** Per layer `W [D, G·H, in]`, `R [D, G·H, H]`, one owned bias `[D, G·H]`,
  with `G` = 1 (RNN), 4 (LSTM), 3 (GRU); the init bound stays `1/√H`, not `1/√(G·H)`.
- **`bias`.** The single owned bias is ONNX's input bias `Wb`; the recurrent bias `Rb` is
  0. A ported PyTorch RNN or LSTM sums `b_ih + b_hh` into it; for GRU see *Gate order* below.
  `bias: false` passes no bias.
- **`numLayers`.** Layers are stacked, each consuming the previous `y`; no inter-layer
  dropout — put `Dropout` between separate calls if needed.
- **Initial state** is zero; no caller-supplied initial state, no carry across calls.
- **Trainable** only single-direction (forward or reverse), default activations.
  `RnnDirection.Bidirectional` and `RnnNonlinearity.Relu` build and run forward
  (inference / ONNX export) but **throw AD003 in back-propagation through time**.
- No QEE step values; value checks run on the ORT backend.

#### `Recurrent.LSTM`

Fixed recurrence (sigmoid gates, tanh cell; no `nonlinearity` knob):

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

- Returns `(y, hN, cN)`; `cN` is the final cell state.
- **Gate order (porting).** Gates are packed ONNX-style `i, o, f, c`. PyTorch packs
  `i, f, g, o`: when importing PyTorch weights, permute the `4H` rows
  `i,f,g,o → i,o,f,g` and sum `b_ih + b_hh`. Irrelevant for from-scratch training.
- Peephole, `input_forget`, `clip`, custom activations and `sequence_lens` are not
  exposed (they throw AD003 in BPTT).

#### `Recurrent.GRU`

Two gates, no cell state; fixed recurrence (sigmoid gates, tanh candidate):

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

- **`linearBeforeReset`.** `true` (default) applies the reset **after** the recurrent
  matmul — `ĥ = tanh(W_h·x + r ⊙ (R_h·h + Rb_h) + Wb_h)` — matching PyTorch `nn.GRU`,
  Keras `reset_after=True`, Flax and cuDNN. `false` applies it before —
  `ĥ = tanh(W_h·x + (r ⊙ h)·R_hᵀ + Rb_h + Wb_h)` — the ONNX default. The forms give
  different results; **both are trainable**.
- **Gate order (porting).** Packed ONNX-style `z, r, h`. PyTorch packs `r, z, n`: swap
  the first two `3H` blocks and sum `b_ih + b_hh` into `Wb` for the `z` and `r` blocks. In the
  reset-after form PyTorch's candidate bias `b_hn` sits inside `r ⊙ (…)`, where `Rb` is fixed
  at 0, so it has no slot: a ported GRU is exact only where `b_hn = 0`.
- `clip`, custom activations and `sequence_lens` are not exposed (AD003 in BPTT).

#### Recurrent cells (single-step) — `Recurrent.RNNCell` / `LSTMCell` / `GRUCell`

One timestep each, for hand-unrolled loops (scheduled sampling, custom decoders, beam
search):

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

State is `[N, H]`. Gate math, init, packing and bias match the layers. The previous state
is a **required** input (pass a zero tensor at step 0). All are trainable except
`RNNCell` with `RnnNonlinearity.Relu` (AD003 in BPTT).

### BatchNorm (+ BatchNorm1d / 2d / 3d aliases)

```csharp
// rank-generic: channel is axis 1; reduces over batch + every spatial axis.
BatchNorm.Call(Scalar<float32>? momentum, Scalar<float32>? epsilon,
               Scalar<bit> training, Scalar<bit> affine,
               Scalar<bit> trackRunningStats, Tensor<float32> x)
```

Supports **ranks 2–5** (`[N, C]` to `[N, C, D, H, W]`), rank inferred at runtime.

- `training = true`: normalizes with **batch** statistics (biased variance) and updates
  the running stats: `running = running * momentum + batch * (1 - momentum)`.
- `training = false`: uses the **running** stats when `trackRunningStats = true`, else
  the batch stats. Eval never changes the running stats.
- `affine = true`: `y = gamma * x̂ + beta` (`Ones`/`Zeros`); `false`: no gamma/beta
  parameters.
- The running mean/variance are **model state** (`checkpoint.ModelState`), not
  trainable params.
- **Defaults**: `momentum` `0.9`, `epsilon` `1e-5` ([nullable, positionally
  required](#nullable-hypers)). `training`, `affine`, `trackRunningStats` are required.
  All-defaults call: `BatchNorm.Call(null, null, training, Scalar(true), Scalar(true), x)`.
- **Porting**: `momentum` weights the *retained* stat, so PyTorch `momentum = p` is
  Shorokoo `1 − p` (default `0.9` ≡ PyTorch `0.1`). Running variance uses the biased
  estimator (PyTorch uses Bessel's correction).
- **Run eval passes through the rig** (or `ComputeContext.ExecuteWithState`): the plain
  inference executor does not persist state, so every run sees the initial running stats.

```csharp
// Thin aliases over BatchNorm, preserving the 4-arg (momentum, epsilon,
// training, x) shape with affine = trackRunningStats = true:
BatchNorm1d.Call(momentum, epsilon, training, x)  // [N, C] or [N, C, L]
BatchNorm2d.Call(momentum, epsilon, training, x)  // [N, C, H, W]   (NCHW)
BatchNorm3d.Call(momentum, epsilon, training, x)  // [N, C, D, H, W] (NCDHW)
```

The aliases' `momentum`/`epsilon` have **no** declared defaults, so all their arguments
are [required](#nullable-hypers).

### LayerNorm / RMSNorm / GroupNorm / InstanceNorm

```csharp
LayerNorm.Call(Scalar<int64> normalizedDims, Scalar<bit> affine, Scalar<float32> epsilon, x)  // last n dims
RMSNorm.Call(Scalar<int64> normalizedDims, Scalar<bit> affine, Scalar<float32> epsilon, x)    // last n dims, no mean-subtraction

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

- Affine parameters are `Ones`/`Zeros`-initialized: LayerNorm's shaped like the
  normalized dims, GroupNorm/InstanceNorm's per channel. `RMSNorm`
  (`y = x / √(mean(x²) + ε) · gain`) has a gain and no bias.
- **`affine = false`** removes gamma and beta (for `RMSNorm`, the gain):
  `RMSNorm.Call(Scalar(1L), Scalar(false), Scalar(1e-5f), x)` is gain-free RMSNorm (as in
  nanochat and modded-nanoGPT; Llama, Mistral, Qwen and Gemma keep the gain). A model
  with gain-free norms everywhere still trains as long as some other parameter exists.
- `GroupNorm(numGroups = 1)` is LayerNorm over CHW; `GroupNorm(numGroups = C)` is
  `InstanceNorm`. Both use the biased variance. `C` must be divisible by `numGroups`, else
  concretization fails.
- **No declared defaults**: every argument, including `epsilon`, is
  [required](#nullable-hypers) — e.g. `LayerNorm.Call(Scalar(1L), Scalar(true), Scalar(1e-5f), x)`.
- `InstanceNorm1d/2d/3d` take `(epsilon, x)` with `affine` off (PyTorch's default);
  use the generic `InstanceNorm` for `affine = true`. InstanceNorm has no running stats
  and behaves the same in training and eval.

### LocalResponseNorm

```csharp
LocalResponseNorm.Call(Scalar<float32>? alpha, Scalar<float32>? beta, Scalar<float32>? k, x)  // size baked = 5
LRNHelper.Lrn(x, long size = 5, float alpha = 1e-4f, float beta = 0.75f, float k = 1.0f)    // arbitrary size
```

AlexNet cross-channel normalization:
`b_c = a_c · (k + (α/size)·Σ_{c'∈window(c)} a_{c'}²)^(−β)` over `[N, C, *spatial]`. The
module fixes `size = 5`; use `LRNHelper.Lrn` for another size. Defaults `alpha` `1e-4`,
`beta` `0.75`, `k` `1` (PyTorch `nn.LocalResponseNorm(5)`):
[`LocalResponseNorm.Call(null, null, null, x)`](#nullable-hypers),
`LocalResponseNorm.Model()` or `LRNHelper.Lrn(x)`. TensorFlow's `tf.nn.lrn` uses a
half-width `depth_radius`, a bare `α` and other defaults, so ported values need conversion.

### Attention / Transformer

```csharp
// Scaled dot-product attention (no params): q/k/v must be rank-4 [N, H, L, d].
// `scale` is `float?`: null (the default) means 1/sqrt(d), d = the last query dim.
// `queryChunks` is the memory lever — see "Sizing an attention run" below.
Attention.ScaledDotProductAttention(Tensor<float32> query, Tensor<float32> key,
                                    Tensor<float32> value, bool causal = false,
                                    float? scale = null,
                                    Tensor<float32>? additiveMask = null,
                                    int queryChunks = 1)

// The additive causal mask ScaledDotProductAttention uses when causal: true,
// exposed on its own (no params): shape [Lq, Lk], 0 on/below the diagonal, -1e9 above.
// `queryOffset` shifts the query rows to absolute positions (null = start at 0).
Attention.CausalMask(Scalar<int64> lq, Scalar<int64> lk, Scalar<int64>? queryOffset = null)

// Rotary positional embedding (RoPE; no params): rotates a [N, H, L, d] tensor
// (d EVEN) by an angle proportional to sequence position. Apply to Q and K
// (NOT V) before ScaledDotProductAttention; returns the same shape.
Attention.ApplyRoPE(Tensor<float32> x, long theta = 10000)

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

`ScaledDotProductAttention` computes `softmax(QKᵀ·scale + mask)·V`, returning
`[N, H, Lq, d]`. Inputs must be **rank-4**.

- **`causal`** is a C# `bool` fixed at build time; `true` adds `CausalMask(Lq, Lk)`, so
  position *i* attends only to *j ≤ i*.
- **`scale`**: `null` means `1/sqrt(d)` with `d` read in-graph (follows a dynamic head
  dim); a value is baked in. It is applied to **Q**, not the scores.
- **`queryChunks`**: see [Sizing an attention run](#attention-memory).
- **`additiveMask`**: an optional mask broadcastable to `[…, Lq, Lk]`, added on top of the
  causal one (padding masks, custom patterns, or a mask chosen by a graph bit).

`CausalMask(lq, lk)` is a `[Lq, Lk]` `float32` tensor: `0` where `col ≤ row`, `-1e9`
above. Lengths are graph scalars (e.g. `query.DimTensor(-2)`); the mask has no gradient.
To gate it on a graph bit:
`causal.IfElse(Attention.CausalMask(lq, lk), TensorFill([lq, lk], 0f))`, passed as
`additiveMask`. `queryOffset` makes rows `[offset, offset + Lq)` — for a query slice,
e.g. a decoding step at position `t` against `t + 1` keys.

All attention layers train end-to-end.

- `MultiHeadAttention` has four `XavierUniform` projections (q/k/v/out) and four
  optional zero biases (`useBias`). Its `causal` is a graph bit; baking it (`Call`/`Model`
  or [`Specialize`](inference.md#hardcoding-hypers-with-specialize)) folds the mask
  choice. On the `ComputationGraph` + `ToConcreteArchitecture` route it is not pruned
  (it holds no parameters) and stays a live run-time switch.
- `TransformerEncoderLayer`: `LayerNorm` + `MultiHeadAttention` + a GELU FFN applied
  token-wise. Layouts are batch-first; there is no `need_weights`/`kdim`/`batch_first`.
- `TransformerDecoderLayer` (pre-LN, "Attention Is All You Need"):
  `h = tgt + MHA_self(LN(tgt), causal: true)`,
  `h2 = h + MHA_cross(LN(h), memory, memory)` (non-causal), `out = h2 + FFN(LN(h2))`.
  `Lt` and `Lm` may differ. `memory` is not normalized (pass the already-normalized
  encoder output, as in PyTorch).
- `ApplyRoPE` (Su et al. 2021) rotates each query/key vector by `m·θ_i`,
  `θ_i = theta^{-2i/d}` (`theta` = HF's `rope_theta`, default `10000`), in the GPT-NeoX /
  HuggingFace **half-split** layout: `RoPE(x) = x·cos(mθ) + rotateHalf(x)·sin(mθ)`,
  `rotateHalf(x) = concat(-x[…, d/2:], x[…, :d/2])`. `d` must be even. Apply to **Q and K
  only**, before `ScaledDotProductAttention`; norms are preserved.

<a id="attention-memory"></a>
#### Sizing an attention run

Attention activations are **quadratic in sequence length**. The unit is one **score
block**, the `[N, H, Lq, Lk]` tensor `QKᵀ`:

```
score block = N · H · Lq · Lk · 4 bytes          (float32)
```

| N (batch) | H (heads) | L = Lq = Lk | one score block |
|---|---|---|---|
| 1 | 4 | 256 | 1 MiB |
| 1 | 4 | 1 024 | 16 MiB |
| 8 | 8 | 1 024 | 256 MiB |
| 32 | 6 | 1 024 | 768 MiB |
| 32 | 6 | 2 048 | 3 GiB |

The head dim `d` does not appear. Doubling the sequence quadruples the block; doubling
batch or heads doubles it. The additive mask adds `[Lq, Lk] · 4 bytes` per call site.

A training step also holds q/k/v and their gradients (a **q block** each,
`N · H · L · d · 4` bytes), which dominate at short sequences. A rule of thumb that
over-estimates the training-step peak:

```
peak  ≈  A · (3 · score block  +  6.5 · q block)

  score block = N · H · Lq · Lk · 4 bytes
  q block     = N · H · L  · d  · 4 bytes
  A           = number of ScaledDotProductAttention calls
```

Example: 6 layers, batch 32, 6 heads, `L = 1024`, `d = 64` ≈ 6 × (2.25 + 0.30) GiB ≈
**15 GiB of attention activations**, before parameters, gradients, optimizer state and
other layers. On a card nothing is reserved up front: a session commits memory as each
tensor is allocated, so a step's peak shows in full in the device's figures (see
[Device memory](gpu-backends.md#device-memory-gpu-backends)).

**`queryChunks: c`** splits the query axis into `c` blocks, runs each against the whole
key/value and concatenates. It is exact (outputs, gradients and causal masking match
the dense path to rounding). It shrinks score-sized tensors by `c` at the cost of `c`
MatMul/Softmax launches and a larger graph. It helps most at long sequences and small
head dims, little at short sequences; measure peak and step time. Keep `c` small.

- `Lq` stays dynamic; chunk `i` covers rows `[Lq·i/c, Lq·(i+1)/c)`. Uneven division is
  fine; `c > Lq` gives empty chunks.
- An `additiveMask`'s query axis (axis -2) must be `Lq` or `1`; any other height is
  rejected. A query-broadcasting mask goes to every chunk; an `Lq`-tall one is sliced.
- Only `Attention.ScaledDotProductAttention` takes it — not `MultiHeadAttention` or the
  transformer layers, since a chunk count must be a C# build-time constant.

Other levers, cheapest first: shorter sequences (quadratic), smaller batch (linear),
fewer heads (linear), and [activation checkpointing](#activation-checkpointing).

<a id="activation-checkpointing"></a>
#### Activation checkpointing: `[Module(Checkpoint = true)]`

```csharp
[Module(Checkpoint = true)]
public partial class MlpBlock
{
    public static Tensor<float32> Inline(Tensor<float32> x)
    {
        var h = Linear.Model(Scalar(32L), Scalar(true)).Call(x).Relu();
        return Linear.Model(Scalar(32L), Scalar(true)).Call(h).Relu();
    }
}
```

The equivalent of PyTorch's `torch.utils.checkpoint`: each call of a checkpointed module
is a **segment** whose internal activations are dropped after the forward pass and
recomputed from its inputs during the backward pass. Results are unchanged; parameters,
state, checkpoints and ONNX export are unaffected. Nested checkpointed modules form one
segment (the outer).

The rig applies the hint **unconditionally**, even where its automatic memory-aware pass
would not recompute (small steps, or a trade of much more compute for memory). It is not
always better than the automatic pass, which can recompute at finer grain; use it for a
segment that dominates peak memory, and measure the step before relying on it (see
[limitations.md](limitations.md#gradient-activation-checkpointing)).

### PReLU / GLU

```csharp
PReLU.Call(x)                       // y = relu(x) - a*relu(-x); single shared [1] learnable slope (init 0.25)
PReLUChannelwise.Call(x)            // same formula, but a SEPARATE [C] learnable slope per channel (init 0.25)
GatedLinear.GLU(x, axis: -1)        // splits x in two halves [a, b] along axis -> a * sigmoid(b)
GLU.Call(x)                         // param-free module form of GatedLinear.GLU with axis fixed at -1
```

`PReLUChannelwise` (PyTorch `num_parameters=C`) takes `C` from axis 1; input rank ≥ 2.
Shorokoo's optimizers apply weight decay uniformly, including to PReLU slopes (He et al.
2015 recommend none). `GatedLinear.GLU` requires an even size along `axis`.

### Dropout

```csharp
Dropout.Call(Scalar<float32> ratio, Scalar<bit> training, Tensor<float32> x)
```

Training zeroes each element with probability `ratio` and scales survivors by
`1/(1-ratio)`; eval is the identity. The mask comes from the layer's own stream under the
model's [RNG configuration](rng-configuration.md): reproducible for a config, different
on each training step.

### SpatialDropout

```csharp
SpatialDropout.Call(Scalar<float32> ratio, Scalar<bit> training, Tensor<float32> x)
Dropout1d.Call(Scalar<float32> ratio, Scalar<bit> training, Tensor<float32> x)  // [N, C, L]
Dropout2d.Call(Scalar<float32> ratio, Scalar<bit> training, Tensor<float32> x)  // [N, C, H, W]
Dropout3d.Call(Scalar<float32> ratio, Scalar<bit> training, Tensor<float32> x)  // [N, C, D, H, W]
```

Drops or rescales whole channels: one draw per `(sample, channel)` over
`[N, C, D1..Dn]`. Rank-generic; on `[N, C]` it equals `Dropout`. Eval is the identity.

### AlphaDropout / FeatureAlphaDropout

```csharp
AlphaDropout.Call(Scalar<float32> ratio, Scalar<bit> training, Tensor<float32> x)         // elementwise
FeatureAlphaDropout.Call(Scalar<float32> ratio, Scalar<bit> training, Tensor<float32> x)  // channel-wise [N,C,...]
```

SELU-paired dropout (Klambauer et al. 2017): dropped units are set to
`α' = −λα ≈ −1.7581`, then `out = a·x' + b` with `a = (q + α'²·q·p)^(−1/2)`,
`b = −a·p·α'` (`q = 1−ratio`), preserving mean and variance in expectation.
`FeatureAlphaDropout` drops whole channels. Eval is the exact identity.

### Embedding

```csharp
// indices of any shape [...] -> embeddings [..., embeddingDim]
Embedding.Call(Scalar<int64> numEmbeddings, Scalar<int64> embeddingDim,
               Scalar<int64> paddingIdx, Scalar<float32> maxNorm,
               Scalar<float32> normType, Tensor<int64> indices)
```

A trainable `[numEmbeddings, embeddingDim]` table, `Normal`-initialized. All five
arguments are [required](#nullable-hypers); knobs-off call:
`Embedding.Call(V, D, Scalar(-1L), Scalar(0f), Scalar(2f), indices)`.

- **`paddingIdx`** (`-1` = off): rows for that index are output as zero and get no
  gradient. The stored row is not zeroed (PyTorch zeroes it).
- **`maxNorm`** (`0f` = off) / **`normType`** (usually `2f`): output rows whose
  `normType`-norm exceeds `maxNorm` are scaled down to it. The stored weight is never
  modified (PyTorch renormalizes it in place, so stored weights diverge in training).
- `scale_grad_by_freq` and `sparse` are not supported.
- **Index range**: an index must lie in `[-numEmbeddings, numEmbeddings-1]`; a negative one counts
  from the end, as ONNX's `Gather` does (PyTorch refuses negative indices). An index outside is the
  backend's to handle: ONNX Runtime's CPU kernel raises its own error, and its CUDA kernel does not
  check and can read a zero row ([#501](https://github.com/Shorokoo/Shorokoo/issues/501)).
- For another initializer use `EmbeddingHelpers.Embed(indices, numEmbeddings,
  embeddingDim, embeddingInit, paddingIdx, maxNorm, normType)`, e.g.
  `EmbeddingHelpers.Embed(idx, V, D, s => XavierUniform.Init(s))` (default `Normal`).

#### EmbeddingBag

```csharp
EmbeddingBag.Bag(Tensor<int64> indices, long numEmbeddings, long embeddingDim,
                 BagMode mode = BagMode.Mean, Func<...>? embeddingInit = null,
                 long paddingIdx = -1)   // BagMode { Sum, Mean, Max }
```

Looks up `indices [B, L]` in a trainable `[V, D]` table and reduces each bag over axis 1,
returning `[B, D]`.

- **2-D fixed-length input only**: PyTorch's `offsets` ragged form,
  `include_last_offset` and `per_sample_weights` are unsupported; pad to `[B, L]` with
  `paddingIdx`.
- The `[B, L, D]` intermediate is materialized (more memory than PyTorch).
- `paddingIdx` zeroes pad rows before reducing: exact for `Sum`, approximate for `Mean`
  (divides by `L`), wrong for `Max` unless embeddings are non-negative.

### Activations

Only activations with a hyperparameter have modules:

```csharp
LeakyReLU.Call(Scalar<float32> alpha, x)  // x for x > 0, alpha * x otherwise
ELU.Call(Scalar<float32> alpha, x)        // x for x > 0, alpha * (exp(x) - 1) otherwise
```

Plain activations are tensor methods: `x.Relu()`, `x.Gelu()`, `x.Sigmoid()`,
`x.Tanh()`, `x.Softmax(axis)`, `x.LogSoftmax(axis)`, …

### Pooling — plain C# helpers

`Pooling` is a static class with plain C# arguments; geometry is not
hyperparameter-driven. Windowed pools come as a **per-axis** form (`long[]` geometry,
rank from `kernelSize.Length`), a **scalar-square** overload (rank from `x.Rank() - 2`),
and `*1d/2d/3d` aliases. They are generic in the float element type.

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

- `Pooling.MaxPool2d(x, 2)` and `Pooling.MaxPool2d(x, [2L, 2L])` both work.
- `stride` defaults to `kernelSize`. Per-axis `stride`/`padding`/`dilation` take length 1
  or spatialRank; `padding` may also be `2*spatialRank` (asymmetric).
- `LpPool`'s `p` is an **integer**, default 2.
- `AvgPool`'s `countIncludePad` defaults to **false**, **unlike PyTorch**; pass `true`
  for PyTorch's denominator.
- **Gradients:** `AvgPool` with `ceilMode: true` **throws in the backward pass**.
  `LpPool`'s gradient **ignores** `ceilMode`, `dilation` and `autoPad` (wrong if trained
  with non-defaults). `MaxPool` (ties → first max), `MaxUnpool` and the global pools are
  exact.
- **Unsupported:** adaptive pooling other than `output_size == 1` (use the global pools),
  and fractional max pooling.

## End-to-end: tiny conv net + CrossEntropyLoss + Adam

A classifier trained with `TrainingRig`. Layer hypers are fixed via `Model(...)` so the
model graph is inputs-only, as the rig requires:

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
    [inputData],                                                  // the model's one input
    new AdamOptimizerHyperparameters { LearningRate = 0.01f });  // β/ε keep defaults

static TensorDataStruct MakeBatch(string field, string structName, TensorData data) =>
    new(new TensorStructDef(
            new[] { new TensorStructFieldDef(field, DataStructure.Tensor,
                                             data.Shape.Dims.Length, data.DType) },
            structName),
        new Dictionary<string, IData> { { field, data } });

var inputBatch  = MakeBatch("input", "ModelInput", inputData);
var targetBatch = MakeBatch("targets", "Target", targetData);

var ckpt = rig.CreateInitialCheckpoint();
for (int i = 0; i < 15; i++)
{
    ckpt = rig.TrainStep(ckpt, inputBatch.Shared(), targetBatch.Shared());  // the batch is fed again: read it
    Console.WriteLine($"step {i}: loss {ckpt.Loss}");
}
```

(Or batch the data as `TensorDataStruct[]` and call `rig.Fit(inputs, targets, numEpochs)`
— see [training.md](training.md).)

## Anti-patterns

- Do not hand `HuberLoss.ComputationGraph` to `TrainingRig` (3-input graph); use
  `SmoothL1Loss` or a 2-input wrapper.
- Do not evaluate BatchNorm with the plain inference executor when the running stats
  matter; they stay at their initial values.
- Do not expect stride/padding from `ConvTranspose2d`; use `NN.ConvTranspose`.
- Do not use `XavierUniform`/`KaimingUniform` (etc.) for rank-1 biases; they require
  rank ≥ 2.
- Do not restructure a model to avoid the parameters of a switched-off
  `useBias`/`affine`; they are pruned ([An off toggle costs nothing](#gated-parameters)).
