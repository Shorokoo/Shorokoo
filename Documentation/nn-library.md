# NN library: layers, losses, optimizers, initializers

Related: [defining-models.md](defining-models.md) · [training.md](training.md) · [core-types.md](core-types.md)

## Facts

- `Shorokoo.Modules` is the baseline neural-network library: initializers, layers,
  losses and optimizers, all built from ordinary Shorokoo `[Module]`s /
  `[TrainableParamInitializer]`s.
- Namespaces: `Shorokoo.Modules.Initializers`, `.Layers`, `.Losses`, `.Optimizers`.
- Layers are `[Module]` classes: `Linear.Call(hypers..., x)` inline, or
  `Linear.Model(hypers...).Call(x)` to fix the hyperparameters once (see
  [defining-models.md](defining-models.md)).
- Exceptions: pooling, the generalized convolution helpers and the recurrent layers
  are plain C#-argument static helpers (`Pooling.MaxPool2d(x, 2)`,
  `Convolution.Conv(...)`, `Recurrent.RNN(x, 16)`), and plain activations are tensor
  one-liners (`x.Relu()`).
- Attention is the only layer whose activations grow **quadratically** with
  sequence length: [Sizing an attention run](#attention-memory) gives the arithmetic
  for budgeting a batch size, and `queryChunks` is its knob. Any `[Module]` can be
  marked `[Module(Checkpoint = true)]` to recompute its activations in the backward
  pass instead of keeping them — see [Activation checkpointing](#activation-checkpointing).

```bash
dotnet add package Shorokoo.Modules
```

## Initializers (`Shorokoo.Modules.Initializers`)

All are `[TrainableParamInitializer]`s taking the parameter's shape first:
`Zeros.Init([outFeatures])`, `KaimingUniform.Init([outC, inC, k, k])`. The three
`Scalar*` entries take **no shape** and create a rank-0 parameter (see
[Trainable scalars](#trainable-scalars)).

| Initializer | Fills with | Notes |
|---|---|---|
| `Zeros` | 0.0 | biases, BatchNorm beta |
| `Ones` | 1.0 | BatchNorm/LayerNorm gamma |
| `Constant` | `value` (every element) | deterministic (no RNG); any rank; generalizes `Zeros`/`Ones` (`Constant(0)`/`Constant(1)`); `value` is an `Init` arg (`Constant.Init([shape], Scalar(v))`); PyTorch `constant_` / Keras `Constant` |
| `ScalarZeros` | 0.0, rank 0 | no shape argument (`ScalarZeros.Init()`); a trainable **scalar**, not a `[1]`-shaped tensor; trainable counterpart of `OptimizerScalarZeros` |
| `ScalarOnes` | 1.0, rank 0 | no shape argument (`ScalarOnes.Init()`); a learned gain starts as a no-op; trainable counterpart of `OptimizerScalarOnes` |
| `ScalarConstant` | `value`, rank 0 | no shape argument; `value` is the only `Init` arg (`ScalarConstant.Init(Scalar(v))`); rank-0 analogue of `Constant`, generalizes `ScalarZeros`/`ScalarOnes` |
| `Uniform` | U(0, 1) | seeded; fixed range (use `UniformRange` for a configurable one) |
| `Normal` | N(0, 1) | seeded; PyTorch's `nn.Embedding` default; fixed (use `NormalDist` for configurable mean/std) |
| `UniformRange` | U(low, high) | seeded; any rank; generalizes `Uniform`; `low`/`high` are `Init` args (`UniformRange.Init([shape], Scalar(lo), Scalar(hi))`) passed to the draw itself, so the range is exact at any width — no precision lost near zero, no overflow on a range wider than float32, and `high` is never returned ([uniform-draws.md](uniform-draws.md)); expects `low ≤ high`; PyTorch `uniform_(a, b)` / Keras `RandomUniform(minval, maxval)` |
| `NormalDist` | N(mean, std) | seeded; any rank; generalizes `Normal`; `mean`/`std` are `Init` args (`NormalDist.Init([shape], Scalar(m), Scalar(s))`), applied as an affine transform of a standard draw; expects `std ≥ 0`; PyTorch `normal_(mean, std)` / Keras `RandomNormal(mean, stddev)` |
| `XavierUniform` | U(−a, a), a = √(6 / (fanIn + fanOut)) | gain 1; seeded; rank ≥ 2 |
| `XavierNormal` | N(0, √(2 / (fanIn + fanOut))) | gain 1; seeded; rank ≥ 2 |
| `KaimingUniform` | U(−b, b), b = √(6 / fanIn) | ReLU gain √2, fan-in mode; seeded; rank ≥ 2; default weight init of `Linear` and the conv layers |
| `KaimingNormal` | N(0, √(2 / fanIn)) | ReLU gain √2, fan-in mode; seeded; rank ≥ 2 |
| `XavierUniformGain` | U(−a, a), a = gain·√(6 / (fanIn + fanOut)) | seeded; rank ≥ 2; equals `XavierUniform` at `gain = 1`; `gain` is an `Init` arg (`XavierUniformGain.Init([shape], Scalar(g))`); PyTorch `xavier_uniform_(t, gain)` (`gain` = the `calculate_gain` std multiplier, computed by the caller) |
| `XavierNormalGain` | N(0, gain·√(2 / (fanIn + fanOut))) | seeded; rank ≥ 2; equals `XavierNormal` at `gain = 1`; `gain` is an `Init` arg; PyTorch `xavier_normal_(t, gain)` |
| `KaimingUniformGain` | U(−b, b), b = gain·√(3 / fanIn) | seeded; rank ≥ 2; equals `KaimingUniform` at `gain = √2`. **Base factor is √(3 / fanIn), not √(6 / fanIn)**: `KaimingUniform` bakes in gain √2, and the supplied gain replaces it. `gain` is an `Init` arg; PyTorch `kaiming_uniform_` (gain via `calculate_gain`) |
| `KaimingNormalGain` | N(0, gain·√(1 / fanIn)) | seeded; rank ≥ 2; equals `KaimingNormal` at `gain = √2`. **Base factor is √(1 / fanIn), not √(2 / fanIn)**: `KaimingNormal` bakes in gain √2, and the supplied gain replaces it. `gain` is an `Init` arg; PyTorch `kaiming_normal_` |
| `TruncatedNormal` | N(0, 1) clamped to [−2, 2] | seeded; clamp approximation (in-graph rejection sampling isn't possible); Keras/JAX-style default |
| `LeCunNormal` | N(0, √(1 / fanIn)) | seeded; rank ≥ 2; JAX/Flax `lecun_normal` (SELU / self-normalizing nets) |
| `Orthogonal` | (semi-)orthogonal matrix (`QᵀQ ≈ I` / `QQᵀ ≈ I`) | seeded; rank ≥ 2; **Björck/Newton–Schulz approximation** (15 cubic iterations `Y ← 1.5·Y − 0.5·Y·(YᵀY)` from a seeded Gaussian; exact QR/SVD orthogonalization isn't expressible in Shorokoo's op set); gain 1; for RNN recurrent matrices and deep stacks (Saxe 2013); PyTorch `orthogonal_` |
| `RecurrentUniform` | U(−1/√H, 1/√H) | seeded; PyTorch's `nn.RNN`/`nn.LSTM`/`nn.GRU` default (`k = 1/hidden_size`); `H` is an `Init` arg (`RecurrentUniform.Init([shape], Scalar(hiddenSize))`), not read from the shape — a gated cell stacks its gates along axis 1, so that axis is `4H` for LSTM and `3H` for GRU while the bound stays `1/√H`; inits W, R and bias alike; used by the `Recurrent` layers |

- **Seeded determinism**: each random initializer draws from its own stream, derived
  from the model's [RNG configuration](rng-configuration.md) (the default identity,
  master seed 0, when no config is given) and the parameter's place in the model.
  Materialization is reproducible for a config, and two parameters of the same shape
  initialized by the same class receive **distinct** values. Bind a different master
  seed to re-roll everything coherently; no seed appears in the model definition.
- **The uniform initializers share one draw.** `Uniform`, `UniformRange`,
  `XavierUniform`, `KaimingUniform`, `RecurrentUniform`, `XavierUniformGain` and
  `KaimingUniformGain` pass their bounds to the same U(low, high) draw (no standard
  draw is scaled afterwards), so all fill a `float32` parameter from the half-open
  `[low, high)` and inherit the guarantees in [uniform-draws.md](uniform-draws.md).
- **The normal initializers share one draw.** `Normal`, `NormalDist`, `XavierNormal`,
  `KaimingNormal`, `XavierNormalGain`, `KaimingNormalGain` and `LeCunNormal` draw the
  same standard N(0, 1) and apply the standard deviation afterwards as an ordinary
  `float32` multiply in the graph (`Normal` is the bare draw; `NormalDist` adds its
  `mean` after the multiply). No drawn magnitude exceeds `8`, so none can produce a
  weight past `8·std` (`mean ± 8·std` for `NormalDist`). The bit-exactness guarantee in
  [normal-draws.md](normal-draws.md) covers the draw only; the fan-scaled initializers
  build their standard deviation in-graph with `Sqrt`, whose accuracy ONNX does not
  specify, so their final values can differ in the last ulp between providers.
  `TruncatedNormal` (clamped to the tighter `[−2, 2]`) and `Orthogonal` (Newton–Schulz
  iterations) also draw from it but do not inherit the `8·std` bound.
- **Fan-in/fan-out** are computed in-graph from the shape vector:
  `fanIn = prod(shape) / shape[0]`, `fanOut = prod(shape) / shape[1]` — the PyTorch
  convention for Linear `[out, in]` and Conv `[outC, inC/g, k...]` layouts. Hence
  rank ≥ 2: use `Zeros`/`Ones`/`Uniform`/`Normal` for biases.

<a id="trainable-scalars"></a>
### Trainable scalars (rank 0)

A learned scalar — a per-layer temperature, a residual scale, a gated architecture's
`gamma` — is a rank-0 parameter. `ScalarZeros` / `ScalarOnes` / `ScalarConstant` create
one, taking no shape:

```csharp
var gamma = ScalarOnes.Init();                        // seeded at 1: starts as a no-op
var beta  = ScalarZeros.Init();                       // seeded at 0
var temp  = ScalarConstant.Init(Scalar(0.125f));      // seeded at 1/√d
return x * gamma + beta;                              // broadcasts against any shape
```

All three are deterministic and mirror `Zeros`/`Ones`/`Constant`:
`ScalarZeros == ScalarConstant(0)`, `ScalarOnes == ScalarConstant(1)`. Fan-in/fan-out
do not apply.

Do **not** use `Ones.Init([Scalar(1L)])` instead. It broadcasts the same way, but it is
a rank-1 length-1 tensor and persists in the checkpoint as a `[1]`-shaped parameter. The
`Scalar*` initializers persist as rank 0.

The optimizer side has the same pair: `OptimizerScalarZeros` / `OptimizerScalarOnes`,
both rank 0 and taking no shape (see [Optimizers](#optimizers-shorokoomodulesoptimizers)).

**Writing your own.** An initializer states the shape of its parameter in one of two
places: the shape vector it takes as its **first** `Inline` parameter, or — for a rank-0
one — its `Scalar<T>` return type. A shape baked into the body of a no-argument `Inline`
is rejected by name when the model is lowered.

The body is an ordinary graph body of tensor operations, loops and `IfElse`. The rules
below apply equally to `[StateInitializer]` and `[TrainableParamInitializer]` bodies.

**It can call the shipped initializers.** An `Init(...)` call inside an initializer body
evaluates that initializer's body as a value; it does not define a second parameter.
Only the top-level initializer, the one the `[Module]` calls, defines a parameter, and
the model's parameter inventory has one entry for it however deep the chain of calls.
Either kind may call either kind. To reuse one fixed distribution for every parameter in
a model, wrap it:

```csharp
[TrainableParamInitializer]
public static partial class NormalDist02
{
    public static Tensor<float32> Inline(Vector<int64> shape)
        => NormalDist.Init(shape, Scalar(0f), Scalar(0.02f));
}
```

The wrapper draws what `NormalDist` draws, value for value, keyed on the parameter being
created. Each draw **site** in the body — its own or a called initializer's — gets its
own sub-stream of that parameter's stream, so no two sites repeat each other. A site
inside a `LoopAPI.Iterate` body folds each enclosing loop's iteration index into its key,
so it draws a fresh sample on every trip, as a runtime draw in a loop does.

**It can start from another parameter's value.** An initializer input typed `Tensor<T>`
may be another trainable parameter, passed at the call site. It is not folded to a
constant: materialization runs the initializers in dependency order, so the input is the
value the model starts from. Re-drawing the source inside the dependent's body would draw
from the dependent's stream — the right distribution, but a different matrix.

```csharp
[TrainableParamInitializer]
public static partial class ProductOf
{
    public static Tensor<float32> Inline(Vector<int64> shape, Tensor<float32> a, Tensor<float32> b)
        => a.MatMul(b);
}

Scalar<int64> vocab = Scalar(50257L), d = Scalar(384L);

var emb  = NormalDist02.Init([vocab, d]);
var wv   = NormalDist02.Init([d, d]);
var bank = ProductOf.Init([vocab, d], emb, wv);   // starts as emb · wv, for the emb the model has

return x.MatMul(emb).MatMul(wv) + x.MatMul(bank);   // every one of the three is read by the model
```

The **shape** input must still fold to a constant at the call site: a parameter's shape
is fixed when the architecture is concretized, before anything has a value. (A rank-0
initializer takes no shape input, so its first input may be a parameter.) Chains — one
parameter from another, from a third — work too.

The value passed may also be **computed from parameters** —
`ProductOf.Init([vocab, d], emb * Scalar(2f), wv)`, or the output of a module called on
one. The computation is carried along and evaluated at initialization, after the
parameters it reads. It must be plain tensor arithmetic over parameters created outside
any loop: a computation that runs through a loop or a branch, reads a parameter standing
for a different one on each trip, or draws randomness is refused at concretization,
naming the initializer.

Two sources are refused by name:

- A source the model reads **nowhere else**. A parameter no forward path reads gets no
  gradient, and the stages after concretization drop it, so the concrete model would
  carry fewer parameters than the architecture and its checkpoints declare.
- A source created **inside a loop**, which stands for a different parameter on every
  trip.

For either, create the source outside the loop and use it in the model, or fold what it
computes into the initializer that reads it so no parameter is created for it.

**It may not create or reference a model.** An initializer computes one parameter's
value and owns no parameter space, so its body may contain no `Foo.Model(...)`, no
`Foo.Call(...)` of any `[Module]` (even a parameter-free one, since calling a module
creates a model of it), no `ModelSequence`, no `IModel.GetTrainableParam`, no read of a
model's hyperparameter, and no model-typed input. Such a body is refused with **FW055**,
naming the initializer, when the graph of the module using it is built — or, for an
initializer called from another, when the called one's body is built, however deep. To
use a layer's output, compute it in the `[Module]` that declares the parameter and pass
it in as a `Tensor<T>` input, or write the computation in the initializer from tensor
operations.

## Layers (`Shorokoo.Modules.Layers`)

Layer hyperparameters are `[Hyper]` graph scalars; pass them as `Scalar(...)` values.
Signatures below are the generated `Call` shapes — **hyperparameters first, tensor
inputs last**. `Inline` declares the same parameters **tensor inputs first,
hyperparameters last**, so the two are not interchangeable argument-for-argument:
`Linear.Call(outFeatures, useBias, x)` is `Linear.Inline(x, outFeatures, useBias)`.

**That reordering applies to `[Module]`s only.** An entry spelled `Class.Method(...)`
rather than `Class.Call(...)` is a **plain-C# static helper**: it keeps the ordinary C#
argument order shown at its entry (tensors first, knobs after, with optional
parameters). The helper classes below are `Pooling`, `Convolution`, `Recurrent`,
`Attention`, `LRNHelper`, `GatedLinear`, `EmbeddingHelpers` and `EmbeddingBag`
(`TripletMarginWithDistance`, under Losses, is one too).

<a id="nullable-hypers"></a>
**A declared `[Hyper]` default makes a *nullable* parameter, not an omittable one.** A
hyperparameter written `[Hyper(<value>)]` carries a declared default, and the generated
`Call`/`Model` expose it as **nullable** (`Scalar<float32>?`), with `null` meaning "use
the declared default". A plain `[Hyper]` is non-nullable and always required. A nullable
parameter also gets a `= null` C# default — becoming omittable — only inside the
**trailing** run of its signature. In `Call` that never happens, because the tensor
inputs come last, so a layer's defaulted hypers are **nullable yet positionally
required**: `LocalResponseNorm.Call(null, null, null, x)`. `Model(hypers…)` takes no
tensors, so there a defaulted hyper in the trailing run can be omitted —
`LocalResponseNorm.Model()` works, while `BatchNorm.Model`'s `momentum`/`epsilon`
precede three plain bits and stay required. Each entry below names only its parameters
that carry a declared default; every other parameter is a plain, required `[Hyper]`.

<a id="gated-parameters"></a>
**An off toggle costs nothing.** Several layers gate a block of trainable parameters
behind a `[Hyper]` bit — `useBias` on `Linear`, `Bilinear`, the conv layers and the
attention/transformer layers; `affine` on `BatchNorm`, `LayerNorm`, `RMSNorm`,
`GroupNorm` and `InstanceNorm`. The layer body is `bit.IfElse(withTheParams, without)`,
and the bit is fixed before the graph is concretized (baked by `Call`/`Model`, or taken
from the sample you hand `ToConcreteArchitecture`). The framework **prunes the
unselected branch's trainable parameters**: with the bit off they are never created — no
checkpoint field, no gradient, no optimizer state, no bytes in a saved model.
`Linear(useBias: false)` carries one parameter, not two; `GroupNorm(affine: false)`
carries none of its own. There is no need to split a model into separate `[Module]`
classes to keep an unused parameter block out.

Two edges:

- On the `Foo.ComputationGraph` + `ToConcreteArchitecture` route the bit is baked but
  **not removed**: like every `[Hyper]` there it stays a live input of the concrete graph
  and must be passed again at `Execute`. Pass the value you concretized with. With the
  bit **off**, its later value is inert *for these layers* — each gates a single-output
  `IfElse` that solely owns its parameters, so the gate went with them; a tuple or shared
  gate would not fold, and its branch would read a zero stand-in instead. With the bit
  **on**, nothing was pruned, so the `IfElse` is still live and the opposite value
  silently takes the other branch. To drop the input,
  [`Specialize`](inference.md#hardcoding-hypers-with-specialize) the bit before
  concretizing. Via `Linear.Call(outFeatures, useBias, x)` the bit is a constant in the
  built subgraph and there is nothing to pass. See
  [What concretization fixes](inference.md#what-concretization-fixes).
- If pruning leaves the **whole model graph** with no trainable parameters,
  `TrainingRig.FromScratch` fails with *"No trainable parameters found in the
  computation graph."* This happens for a model that *is* the gated layer, or whose every
  parameter block is switched off. A **sub-module** pruned to none is fine:
  `RMSNorm(affine: false)` beneath a parent with its own parameters builds normally.

### Linear

```csharp
// y = x @ W^T (+ b); flattens trailing dims: [N, d1, d2, ...] -> [N, d1*d2*...]
Linear.Call(Scalar<int64> outFeatures, Scalar<bit> useBias, Tensor<float32> x)
```

Weight `[outFeatures, inFeatures]` is `KaimingUniform`-initialized; bias `[outFeatures]`
is zero-initialized. `useBias = false` drops the bias **term and its parameter**: the
layer is a single-parameter matmul, and no `Zeros` parameter appears in the checkpoint
(see [An off toggle costs nothing](#gated-parameters)).

### Bilinear

```csharp
// y_k = x1^T A_k x2 + b_k  (PyTorch nn.Bilinear)
Bilinear.Call(Scalar<int64> in1Features, Scalar<int64> in2Features,
              Scalar<int64> outFeatures, Scalar<bit> useBias,
              Tensor<float32> x1, Tensor<float32> x2)
```

Per output channel `k`, `y[..., k] = Σ_{i,j} x1[..., i]·A[k,i,j]·x2[..., j] (+ b[k])`.
Weight `A` is `[outFeatures, in1Features, in2Features]`; bias `b` is `[outFeatures]`.
Both are initialized from `U(±1/√in1Features)` (PyTorch's bound, via `RecurrentUniform`)
— the bias is **not** zero-initialized, unlike `Linear`. The contraction is over each
input's **last** axis; the two inputs must share their leading (batch) dims (`(*, in1)`,
`(*, in2)` → `(*, out)`), which are preserved. `useBias = false` omits the bias term and
its parameter ([An off toggle costs nothing](#gated-parameters)).

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

All geometry (kernel size, stride, padding, dilation, groups) is hyperparameter-driven:
the layers use the `NN.Conv` overload that takes geometry as int64 tensor inputs, and the
exported model contains a standard ONNX Conv. Weight `[outChannels, inChannels/groups,
k(, k)]` is `KaimingUniform`-initialized; `inChannels` is read from the input's shape
in-graph. `useBias = false` replaces the trainable zero bias `[outChannels]` with an
all-zero constant, so no bias parameter is created
([An off toggle costs nothing](#gated-parameters)). These modules cover the
**square-kernel / symmetric-pad** case; for per-axis geometry, `auto_pad`, or a
non-zeros `padding_mode`, use the `Convolution` helper below.

### ConvTranspose2d — default geometry only

```csharp
ConvTranspose2d.Call(Scalar<int64> outChannels, Scalar<int64> kernelSize,
                     Scalar<bit> useBias, Tensor<float32> x)
```

Geometry stays at the ONNX defaults — stride 1, no padding, dilation 1, group 1 — with
the kernel shape inferred from the (dynamic) weight `[inChannels, outChannels, k, k]`.
For other stride/padding, call `NN.ConvTranspose` directly with static attribute values.

### Convolution — generalized per-axis helpers

The `[Module]` layers above (`Conv1d/2d/3d`, `ConvTranspose2d`) keep a square/cubic,
hyperparameter-driven signature for `Model(...)` hyperparameter baking. For the **full
ONNX attribute surface** (per-axis kernel/stride/padding/dilation, asymmetric padding,
`auto_pad`, `groups`, `padding_mode`, and transposed-conv `output_padding`/`output_shape`)
use the static `Convolution` class. Like `Pooling`, its helpers take **plain C# array
arguments**. Convolution geometry sizes the weight and is baked at concretization
regardless, so it gains nothing from being a `[Hyper]`. Only the weight's `inChannels`
axis is read in-graph from `x.ShapeTensor()[1]`, so these helpers are lazy in the input
channel count like the modules.

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
  (fan-in `inC/groups·∏k`); transposed conv weight is `[inChannels, outChannels/groups,
  k…]` (in/out axes swapped). Both are `KaimingUniform`-initialized. `bias: true` makes a
  trainable zero-initialized bias `[outChannels]`; `bias: false` uses an all-zero constant.
- **`auto_pad`.** `AutoPad.SameUpper` matches TF/PyTorch `"same"` (extra pad on the high
  side); `SameLower`, `Valid` and `NotSet` (explicit pads) are also available. `auto_pad`
  cannot be combined with a `padding_mode` other than `Zeros` (those compose a separate
  `Pad`) or with `Causal`.
- **`groups`.** `1` is dense; `groups == inChannels` (with `outChannels` a multiple of
  `inChannels`) is depthwise. Weight axis 1 is `inChannels/groups`.
- **`padding_mode`.** `Zeros` uses the conv's own (differentiable) implicit padding.
  `Reflect`/`Replicate`/`Circular` map to `PadMode.Reflect`/`Edge`/`Wrap` and are
  realized by an explicit `Tensor.Pad` over the spatial axes followed by a zero-pad conv.
  **Caveat:** that `Pad` step is **non-differentiable** — reflect/edge/wrap have no
  autodiff and no QEE values, so they **throw in autodiff**. These modes are **forward /
  inference only**. `Causal` is **1-D only** (rejected for higher spatial ranks): it
  left-pads `(k-1)*dilation` zeros on the spatial axis so `out[t]` never sees future
  input (WaveNet-style), and is a constant (differentiable) zero-pad.
- **ConvTranspose `output_padding` / `output_shape`.** `output_padding` disambiguates the
  output size when `stride > 1` maps several input sizes to the same output (it changes
  the claimed shape; it is not literal zero-padding). PyTorch's
  `output_padding < max(stride, dilation)` guard is **not** imposed — ONNX Runtime
  validates the geometry. `output_shape` names the target spatial size directly and, when
  given, overrides `output_padding`. It may exceed the full extent
  `stride * (in - 1) + output_padding + (kernel - 1) * dilation + 1` by one element, which
  zero-extends the end as in ONNX's own `output_shape` example (not under
  `auto_pad: SameUpper`); beyond that ONNX would need negative begin pads, so
  concretization refuses it with **FW054**, naming the output_shape and the full extent.
  Raise `output_padding` instead to reach a larger size. Transposed conv is
  **zeros-only** (no `padding_mode`): its "padding" is an output-shape crop, not an input
  border.

<a id="recurrent-layers"></a>
### Recurrent layers — `Recurrent.RNN` / `Recurrent.LSTM` / `Recurrent.GRU`

The vanilla (Elman) recurrent layer is `Recurrent.RNN`, alongside `Recurrent.LSTM` and
`Recurrent.GRU`. Like `Convolution` and `Pooling`, `Recurrent` is a **static class of
plain-C#-argument helpers, not a `[Module]`**: every knob (`hiddenSize`,
`nonlinearity`, `direction`, `numLayers`, `batchFirst`, `bias`) is shape- or
topology-determining and baked at build time, and the `nonlinearity`/`direction` enums
cannot be `[Hyper]`s, which are scalar-only. The weights are created via
`RecurrentUniform.Init` as trainable parameters in the composed graph (as
`Convolution.Conv` owns its weight), so the layers train end-to-end.

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

- **Input / output layout.** `x` is `[L, N, inputSize]` (sequence-first) by default, or
  `[N, L, inputSize]` when `batchFirst: true`. `inputSize` is read in-graph from the last
  axis, so the layer is lazy in input size. `y` is the **full output sequence** (every
  step's hidden state) in PyTorch layout `[L, N, D·H]` (or `[N, L, D·H]` when
  `batchFirst`), where `D = 2` for `Bidirectional`, else `1`. For the last output only
  (Keras `return_sequences=False`), slice `y[-1]` or read `hN`.
- **Return contract.** `hN` is the final hidden state per direction and layer,
  `[D·numLayers, N, H]` — batch-second regardless of `batchFirst`, as in PyTorch. The
  `(y, hN)` tuple covers Keras's `return_sequences` and `return_state` modes.
- **Weights & init.** Per layer, with `D = direction == Bidirectional ? 2 : 1`:
  `W [D, H, in]` (input→hidden), `R [D, H, H]` (hidden→hidden) and `bias [D, H]`, all
  `RecurrentUniform`-initialized (PyTorch's `U(−1/√H, 1/√H)`). With a single gate there
  is no PyTorch↔ONNX gate reorder.
- **`bias`.** A single owned bias `[D, H]` is fed to the op as `B = concat(bias, zeros)`
  on axis 1: the ONNX input bias `Wb` carries it and the recurrent bias `Rb` is 0, so the
  two ONNX/PyTorch biases collapse into one, as in Keras/Flax. A ported PyTorch RNN folds
  `b_ih + b_hh` into it. `bias: false` passes no bias to the op.
- **`numLayers` stacking.** Builds `numLayers` RNN ops in sequence, feeding each layer's
  output sequence (reshaped `[L, D, N, H] → [L, N, D·H]`) to the next, and concatenating
  each layer's final state on the leading axis to form `hN` `[D·numLayers, N, H]`.
  Inter-layer dropout (PyTorch `num_layers > 1`) is **not** included — put `Dropout`
  between stacked `Recurrent.RNN` calls if wanted.
- **`batchFirst`.** Realized by transposing `[N, L, …] → [L, N, …]` in-graph before the
  stack and transposing `y` back after; the op **always** runs at `layout=0` (ORT-CPU
  rejects `layout=1` and autodiff supports only `layout=0`). `hN` stays
  `[D·numLayers, N, H]`.
- **Initial state.** `h_0` is zero (an omitted op input; ONNX zero-fills). There is no
  caller-supplied `initial_h` and no stateful carry across calls.
- **Autodiff caveat.** Only **single-direction (forward or reverse), tanh, `layout=0`**
  RNNs are **trainable**. `RnnNonlinearity.Relu` and `RnnDirection.Bidirectional`
  **build and run forward but throw AD003 in back-propagation through time** — they are
  **inference only**.
- **No QEE values.** RNN has no QEE step values, so closed-form / value checks run on the
  **ORT backend**, not the QEE value path.

#### `Recurrent.LSTM`

`Recurrent.LSTM` shares the RNN infrastructure (weight ownership over the ONNX
`[num_dir, …]` layout, `RecurrentUniform` init, the `RnnDirection` enum, and the
`numLayers`/`batchFirst`/`bias`/zeroed-state behaviour). The gate recurrence is fixed
(sigmoid gates, tanh cell; **no `nonlinearity` knob**):

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

- **Return contract.** `(y, hN, cN)`: `y` is the full output sequence `[L, N, D·H]` (or
  `[N, L, D·H]` when `batchFirst`); `hN` is the final hidden state and `cN` the final
  **cell** state, each `[D·numLayers, N, H]` (batch-second regardless of `batchFirst`).
  Slice `y[-1]` or read `hN` for the last output only; ignore `hN`/`cN` for the sequence
  only.
- **Weights & init.** Per layer: `W [D, 4H, in]`, `R [D, 4H, H]` and a single owned bias
  `[D, 4H]`, all `RecurrentUniform`-initialized with the explicit hidden size, so the
  bound is PyTorch's `U(−1/√H, 1/√H)`, **not** `1/√(4H)`.
- **Gate order (port note).** The four gate blocks are packed in the **ONNX-native
  `i, o, f, c` order**, the only layout the op understands, with **no** reorder shim.
  Because the init is uniform across gates, the order is unobservable for a from-scratch
  model. It matters only when importing pretrained PyTorch weights (not provided):
  PyTorch `nn.LSTM` packs `i, f, g(=c), o`, so permute the `4H` rows
  `i,f,g,o → i,o,f,g`, and sum PyTorch's two `4H` biases (`b_ih`, `b_hh`) into the single
  owned bias.
- **`bias`.** The owned bias `[D, 4H]` is fed as `B = concat(bias, zeros)` on axis 1
  (`[D, 8H]`; `Wb` carries it, `Rb` is 0). `bias: false` passes no bias to the op.
- **`numLayers` / `batchFirst`.** As for `Recurrent.RNN`; each layer's final `hN`/`cN`
  are concatenated on the leading axis, and `hN`/`cN` stay `[D·numLayers, N, H]`.
- **Initial state.** `h_0` **and** `c_0` are zero (omitted op inputs). Peephole `P` is
  null. No caller-supplied initial state or stateful carry across calls.
- **Autodiff caveat.** Single-direction (forward or reverse), `layout=0`,
  default-activation LSTM is **trainable** end-to-end through the `TrainingRig`.
  `RnnDirection.Bidirectional` **builds and runs for forward inference / ONNX export but
  throws AD003 in back-propagation through time** — it is **inference only**. Peephole,
  `input_forget`, `clip`, custom activations and variable-length `sequence_lens` exist on
  the core op but all throw AD003 in BPTT, so the layer does **not expose** them.
- **No QEE values.** As for RNN — value checks run on the **ORT backend**.

#### `Recurrent.GRU`

`Recurrent.GRU` shares the RNN/LSTM infrastructure. It has **two** gates instead of
three and **no cell state**. The gate recurrence is fixed (sigmoid gates, tanh
candidate; **no `nonlinearity` knob**):

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

- **Return contract.** `(y, hN)`: `y` is the full output sequence `[L, N, D·H]` (or
  `[N, L, D·H]` when `batchFirst`); `hN` is the final hidden state `[D·numLayers, N, H]`
  (batch-second regardless of `batchFirst`).
- **`linearBeforeReset`.** Selects where the reset gate enters the candidate. The default
  **`true`** applies it **after** the recurrent matmul —
  `ĥ = tanh(W_h·x + r ⊙ (R_h·h + Rb_h) + Wb_h)` — matching **PyTorch `nn.GRU`, Keras
  `reset_after=True`, Flax and cuDNN**. `false` applies it **before** the matmul —
  `ĥ = tanh(W_h·x + (r ⊙ h)·R_hᵀ + Rb_h + Wb_h)` — the original Cho et al. form and the
  ONNX op's own default. The two forms give different results with the same weights;
  **both are trainable**.
- **Weights & init.** Per layer: `W [D, 3H, in]`, `R [D, 3H, H]` and a single owned bias
  `[D, 3H]`, all `RecurrentUniform`-initialized with the explicit hidden size, so the
  bound is `U(−1/√H, 1/√H)`, **not** `1/√(3H)`.
- **Gate order (port note).** The three gate blocks are packed in the **ONNX-native
  `z, r, h` order** with **no** reorder shim; unobservable for a from-scratch model (the
  init is uniform across gates). When importing pretrained PyTorch weights (not
  provided): PyTorch `nn.GRU` packs `r, z, n(=h)` (Keras `r, z`), so swap the first two
  `3H` gate blocks (`r,z,n → z,r,h`; the candidate block stays last), and map PyTorch's
  two `3H` biases (`b_ih`, `b_hh`) onto `Wb`/`Rb` — exact for the default reset-after
  form.
- **`bias`.** The owned bias `[D, 3H]` is fed as `B = concat(bias, zeros)` on axis 1
  (`[D, 6H]`; `Wb` carries it, `Rb` is 0). With `linearBeforeReset: true` the single `Wb`
  bias is equivalent to PyTorch's `b_ih + b_hh` sum. `bias: false` passes no bias.
- **`numLayers` / `batchFirst`.** As for `Recurrent.RNN`/`Recurrent.LSTM`; `hN` stays
  `[D·numLayers, N, H]`.
- **Initial state.** `h_0` is zero (an omitted op input). No caller-supplied initial
  state or stateful carry across calls.
- **Autodiff caveat.** Single-direction (forward or reverse), `layout=0`,
  default-activation GRU is **trainable** end-to-end through the `TrainingRig`, in both
  `linearBeforeReset` forms. `RnnDirection.Bidirectional` **builds and runs for forward
  inference / ONNX export but throws AD003 in back-propagation through time** — it is
  **inference only**. `clip`, custom activations and variable-length `sequence_lens`
  exist on the core op but throw AD003 in BPTT, so the layer does **not expose** them.
- **No QEE values.** As for RNN/LSTM — value checks run on the **ORT backend**.

#### Recurrent cells (single-step) — `Recurrent.RNNCell` / `LSTMCell` / `GRUCell`

Each cell computes **one** timestep, taking the previous hidden state(s) and returning
the new one(s), so you can hand-unroll a custom loop (scheduled sampling,
attention-augmented decoders, beam search).

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

Each runs the matching ONNX op at **sequence length 1** (the previous state is the op's
`initial_h`/`initial_c`; the `num_dir` axis is stripped so state is `[N, H]`), so the
gate math, `RecurrentUniform` init (`U(−1/√H, 1/√H)`), gate packing and bias collapse
are **identical** to the layers. The initial state is a **required** tensor input (seed
step 0 with an explicit zero tensor). The default (tanh) `RNNCell`, `LSTMCell`, and both
`linearBeforeReset` forms of `GRUCell` are **trainable**; `RNNCell` with
`RnnNonlinearity.Relu` is inference only (BPTT throws AD003), as for
`Recurrent.RNN(Relu)`.

### BatchNorm (+ BatchNorm1d / 2d / 3d aliases)

```csharp
// rank-generic: channel is axis 1; reduces over batch + every spatial axis.
BatchNorm.Call(Scalar<float32>? momentum, Scalar<float32>? epsilon,
               Scalar<bit> training, Scalar<bit> affine,
               Scalar<bit> trackRunningStats, Tensor<float32> x)
```

One rank-generic module covers PyTorch's `BatchNorm1d/2d/3d`: **ranks 2–5** —
`[N, C]`, `[N, C, L]`, `[N, C, H, W]`, `[N, C, D, H, W]`. The reduction axes
`{0} ∪ {2..rank-1}` and the per-channel broadcast shape `[1, C, 1, …, 1]` are derived
in-graph from the input's runtime rank.

- `training = true`: normalizes with **batch** statistics (biased variance) and
  EMA-updates the running stats via `Globals.StateUpdate` (ONNX/Keras momentum
  convention: `running = running * momentum + batch * (1 - momentum)`).
- `training = false`: normalizes with the **running** statistics when
  `trackRunningStats = true`, or with the eval **batch** statistics when
  `trackRunningStats = false` (PyTorch `track_running_stats=False`). The state update is
  gated off, so eval passes never change the running stats.
- `affine = true` applies `y = gamma * x̂ + beta`; `affine = false` returns `x̂`. gamma
  (`Ones`) and beta (`Zeros`) exist as trainable params **only when `affine = true`**
  (see [An off toggle costs nothing](#gated-parameters)). The running stats are
  unaffected either way.
- The running mean/variance are module-owned state that `TrainingRig` threads as
  **model state** (`checkpoint.ModelState`), not trainable params.
- **Defaults**: only `momentum` (`0.9`) and `epsilon` (`1e-5`) carry a declared default
  ([nullable, positionally required](#nullable-hypers)). `training`, `affine` and
  `trackRunningStats` are plain `[Hyper]` bits, always written out; PyTorch's defaults
  are `affine=True` / `track_running_stats=True`. The all-defaults call is
  `BatchNorm.Call(null, null, training, Scalar(true), Scalar(true), x)`.
- **Port note**: Shorokoo `momentum` weights the *retained* running stat (ONNX/Keras
  sense), so for PyTorch `BatchNorm(momentum = p)` use `momentum = 1 − p` (the default
  `0.9` ≡ PyTorch `0.1`). The running variance uses the **biased** estimator
  (ONNX/Keras/Flax), a minor numeric difference from PyTorch's Bessel-corrected
  `running_var`.
- **Run eval passes through the rig**: the plain inference executor runs a graph with
  `StateUpdate` links, but state does not persist across a one-shot execution — every
  run sees the running stats as the initializer left them, and the update is dropped. An
  eval-mode BatchNorm executed that way normalizes with the initial statistics. Use a
  `TrainingRig`, or `ComputeContext.ExecuteWithState`, whenever the running stats matter.

```csharp
// Thin aliases over BatchNorm, preserving the 4-arg (momentum, epsilon,
// training, x) shape with affine = trackRunningStats = true:
BatchNorm1d.Call(momentum, epsilon, training, x)  // [N, C] or [N, C, L]
BatchNorm2d.Call(momentum, epsilon, training, x)  // [N, C, H, W]   (NCHW)
BatchNorm3d.Call(momentum, epsilon, training, x)  // [N, C, D, H, W] (NCDHW)
```

The `1d/2d/3d` aliases forward to the generic `BatchNorm` with `affine` and
`trackRunningStats` on (rank is still inferred at runtime); use `BatchNorm` for the full
toggle surface. Their `momentum`/`epsilon` are plain `[Hyper]`s — the generic's
`0.9`/`1e-5` defaults are **not** inherited — so all alias arguments are
[non-nullable and required](#nullable-hypers).

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

Built in-graph from elementwise/reduce ops, because the ONNX normalization ops take
epsilon/numGroups as static attributes, which would forbid `[Hyper]` values. The affine
parameters are `Ones`/`Zeros`-initialized: LayerNorm's gamma/beta are shaped like the
normalized trailing dims; GroupNorm/InstanceNorm's are per-channel (broadcast
`[1, C, 1, …, 1]`, sized to the runtime rank). `RMSNorm`
(`y = x / √(mean(x²) + ε) · gain`) skips the mean subtraction and the bias, keeping only
a gain — the normalization used by most modern LLMs.

**All four have the same `affine` toggle** (an `IfElse` gate, like `Linear`'s
`useBias`): the affine parameters exist **only when `affine = true`**; with
`affine = false` the layer has no parameters of its own (see
[An off toggle costs nothing](#gated-parameters)). For `LayerNorm` / `GroupNorm` /
`InstanceNorm` it gates gamma **and** beta; for `RMSNorm`, the gain alone:

```csharp
RMSNorm.Call(Scalar(1L), Scalar(false), Scalar(1e-5f), x)   // x / √(mean(x²) + ε), no gain
```

nanochat and modded-nanoGPT use the gain-free form; Llama, Mistral, Qwen and Gemma keep
the gain. Match your reference.

The gain-free form does **not** trigger the *"No trainable parameters found in the
computation graph."* failure, which checks the whole model graph: `RMSNorm(affine:
false)` — or `GroupNorm(affine: false)` — at every normalization site of a transformer
builds as long as some trainable parameter still reaches the output, as any model with a
projection does. Only a model consisting solely of gain-free normalization fails. See
[An off toggle costs nothing](#gated-parameters).

`GroupNorm` and `InstanceNorm` differ only in the number of channel groups (Wu & He
2018): `GroupNorm(numGroups = 1)` is LayerNorm over CHW and `GroupNorm(numGroups = C)`
is `InstanceNorm`. Both reduce over each per-(sample, group/channel) region's channels
and **every** spatial axis, using the **biased** variance.

- **No declared defaults**: `normalizedDims`, `numGroups`, `affine` and `epsilon` are
  all plain `[Hyper]`s, hence [non-nullable and always passed](#nullable-hypers).
  `epsilon` has no `1e-5` fallback; write it out
  (`LayerNorm.Call(Scalar(1L), Scalar(true), Scalar(1e-5f), x)`). The `InstanceNorm`
  `1d/2d/3d` aliases also require `epsilon`; only `affine` is supplied.
- **`LayerNorm` / `RMSNorm` / `GroupNorm`**: PyTorch/Keras/Flax default `affine` to on
  (`elementwise_affine=True` / `affine=True`). For `GroupNorm`, `C` must be divisible by
  `numGroups`, else the `[N, G, -1]` reshape fails at concretization.
- **`InstanceNorm`**: the `1d/2d/3d` aliases take `(epsilon, x)` with `affine` **off**,
  matching PyTorch's `affine=False` InstanceNorm default (the canonical style-transfer
  use normalizes without a learnable affine). Pass `affine = true` to the generic
  `InstanceNorm` to opt in. InstanceNorm has **no** running stats or momentum — its
  statistics are per-instance and identical at train and eval time, so it runs on the
  plain inference pipeline; for running-stat normalization use `BatchNorm`.

### LocalResponseNorm

```csharp
LocalResponseNorm.Call(Scalar<float32>? alpha, Scalar<float32>? beta, Scalar<float32>? k, x)  // size baked = 5
LRNHelper.Lrn(x, long size = 5, float alpha = 1e-4f, float beta = 0.75f, float k = 1.0f)    // arbitrary size
```

Cross-channel normalization (Krizhevsky et al. 2012, AlexNet):
`b_c = a_c · (k + (α/size)·Σ_{c'∈window(c)} a_{c'}²)^(−β)` over `[N, C, *spatial]`
(channel = axis 1), same output shape. The module exposes `alpha`/`beta`/`k` as
hyperparameters (`k` = PyTorch's additive constant / ONNX `bias`) and **bakes the window
width `size = 5`** (the ONNX/PyTorch default), because `size` is a compile-time ONNX
attribute; for another width use `LRNHelper.Lrn`. LRN is largely **superseded by
BatchNorm** and is provided for AlexNet-era parity. Porting from TensorFlow
(`tf.nn.lrn`: half-width `depth_radius`, bare `α`, different defaults) needs conversion.

- **Defaults**: `alpha` (`1e-4`), `beta` (`0.75`) and `k` (`1`) all carry a declared
  default, matching `nn.LocalResponseNorm(5)`, so
  [`LocalResponseNorm.Call(null, null, null, x)`](#nullable-hypers) is the all-defaults
  call (and `LocalResponseNorm.Model()` the all-defaults model). `LRNHelper.Lrn` is not a
  `[Module]`, so its `size`/`alpha`/`beta`/`k` are ordinary optional C# parameters:
  `LRNHelper.Lrn(x)` is the all-defaults call.

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

`Attention.ScaledDotProductAttention` computes `softmax(QKᵀ·scale + mask)·V` and returns
`[N, H, Lq, d]` (`d` from `value`). `query`/`key`/`value` must be **rank-4** (the
last-two-dims transpose is a static perm `[0, 1, 3, 2]`); `MultiHeadAttention` reshapes
to that layout before calling. The optional arguments:

- **`causal`** is a plain C# `bool` decided at graph build time (not a `Scalar<bit>`):
  `true` adds `CausalMask(Lq, Lk)` to the scores before the softmax, so position *i*
  attends only to *j ≤ i*. `Lq`/`Lk` come in-graph from the query's and key's axis -2.
- **`scale`** is `float?`. Left `null`, the scale is `1/sqrt(d)` with `d` the **last
  query dim, read in-graph**, so it follows a dynamic head dim. A value is baked in as a
  constant multiplier, for models whose scaling differs from `1/sqrt(d)`. It multiplies
  **Q**, not the scores: `Q` is `[N, H, Lq, d]` and the scores are `[N, H, Lq, Lk]`, so
  scaling the smaller operand keeps one score-sized tensor out of the forward pass (and
  its gradient `Mul` out of the backward pass).
- **`queryChunks`** is a plain C# `int` (default `1` = dense) that splits the query axis
  into that many blocks — attention's one memory lever; see
  [Sizing an attention run](#attention-memory).
- **`additiveMask`** is an optional pre-built additive mask, broadcastable to the
  `[…, Lq, Lk]` scores and added **on top of** the causal one (padding masks, custom
  patterns). Use it when the mask is a graph tensor rather than a C# decision.
  `MultiHeadAttention` uses it this way: its `causal` is a graph `Scalar<bit>`, which C#
  cannot branch on, so the mask is selected with an `IfElse`. That bit is a `[Hyper]`,
  so baking it — `Call`/`Model`, or
  [`Specialize`](inference.md#hardcoding-hypers-with-specialize) — folds the `IfElse` to
  one branch. Unlike `useBias`, neither branch holds a trainable parameter, so on the
  `ComputationGraph` + `ToConcreteArchitecture` route nothing is pruned and the `IfElse`
  stays live, selecting at run time (see [An off toggle costs nothing](#gated-parameters)).

`Attention.CausalMask(lq, lk)` is that mask alone: a `[Lq, Lk]` `float32` tensor, `0`
where the key position is on or before the query position (`col ≤ row`) and `-1e9`
above the diagonal. Both lengths are graph scalars (`Scalar<int64>`, e.g.
`query.DimTensor(-2)`), so the mask sizes itself from the actual sequence lengths. It is
built from `Range` + comparison + `Where` on two broadcast `[1]` constants — no `Trilu`,
no gradient, one `[Lq, Lk]` tensor. Use it when assembling attention yourself, or when
the mask must be gated on a graph bit:
`causal.IfElse(Attention.CausalMask(lq, lk), TensorFill([lq, lk], 0f))`, then pass the
result as `additiveMask`.

The optional `queryOffset` shifts the query rows to **absolute** positions
`[offset, offset + Lq)` instead of `[0, Lq)`. This makes a causal mask correct when the
queries are a *slice* of the sequence: a `queryChunks` block (which passes it for you),
or a decoding step whose single query row sits at position `t` against a key history of
length `t + 1`.

All are built from autodiff-supported primitives (MatMul / Softmax / Transpose / Where)
and train end-to-end. `MultiHeadAttention` owns four `XavierUniform` projections
(q/k/v/out) with four optional zero biases; `useBias = false` drops the four bias
parameters ([An off toggle costs nothing](#gated-parameters)). Its causal mask is a
constant gated by the `[Hyper]` `causal` bit through an `IfElse`, folded away once the
bit is baked by `Call`/`Model` or `Specialize`. `TransformerEncoderLayer` composes
`LayerNorm` + `MultiHeadAttention` and a GELU FFN (the FFN uses explicit token-wise
MatMuls, since `Linear` would flatten the sequence into the feature axis on a
`[N, L, E]` input). There is no `need_weights`/`kdim`/`batch_first`/… surface: layouts
are batch-first.

`Attention.ApplyRoPE` applies **rotary positional embedding** (RoPE; Su et al. 2021), a
parameter-free rotation that encodes *relative* position in the attention dot product.
It rotates each per-head query/key vector (a `[N, H, L, d]` tensor with **`d` even**) by
`m·θ_i`, where `m` is the token position and `θ_i = theta^{-2i/d}` (`theta` default
`10000`, HF's `rope_theta`), using the GPT-NeoX / HuggingFace **half-split** layout:
`RoPE(x) = x·cos(mθ) + rotateHalf(x)·sin(mθ)`, with
`rotateHalf(x) = concat(-x[…, d/2:], x[…, :d/2])`. The cos/sin tables are built in-graph
from the input's `L` (axis -2) and `d` (axis -1) and broadcast over `[N, H]`. Apply it
to **Q and K only** (never V), *before* `ScaledDotProductAttention`; the output keeps the
`[N, H, L, d]` shape and, being a rotation, preserves each vector's norm.

`TransformerDecoderLayer` is the pre-LN **cross-attention decoder block** from
"Attention Is All You Need". It follows `TransformerEncoderLayer`'s residual/LN/FFN
structure with an added cross-attention sublayer:
`h = tgt + MHA_self(LN(tgt), causal: true)` (masked self-attention),
`h2 = h + MHA_cross(LN(h), memory, memory)` (query = `LN(h)`, key = value = the **raw**
encoder `memory`, non-causal), then `out = h2 + FFN(LN(h2))` (the encoder's GELU FFN).
Its inputs are `tgt [N, Lt, embedDim]` and `memory [N, Lm, embedDim]`; `Lt` and `Lm` may
differ. The self-attention is always causal; `memory` is fed unnormalized (expected to be
the already-LayerNorm'd encoder output, as in PyTorch).

<a id="attention-memory"></a>
#### Sizing an attention run

Attention's activations are **quadratic in sequence length**, so it is the layer you
budget for by hand. The unit is one **score block** — the `[N, H, Lq, Lk]` tensor `QKᵀ`
and every same-shaped tensor downstream of it:

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

The head dim `d` does not appear (`QKᵀ` contracts it away), nor does the parameter
count. Doubling the sequence length quadruples the block; doubling the batch or head
count doubles it. The additive mask is a separate `[Lq, Lk] · 4 bytes` (4 MiB at
`L = 1024`), shared by every batch element and head but built **per call site**, not
deduplicated, and split into `c` pieces of `[Lq/c, Lk]` under `queryChunks`.

**What a training step holds.** Every figure below is **measured**: the resident
high-water mark of one training step on the CPU backend (`VmHWM` with ONNX Runtime's
arena off, so every activation is a real allocation), for attention with all three
projections trainable and a causal mask, at `N = 2`, `H = 4`, `L = 256` (a 2 MiB score
block), after the training rig's memory pass has run (see
[limitations.md](limitations.md#gradient-activation-checkpointing)).

The peak also holds q/k/v and their gradients, each `N · H · L · d · 4` bytes — a
**q block**. It is `d/L` times a score block: negligible at long sequences, significant
at short ones:

| attention calls | `d` | peak | in score blocks |
|---|---|---|---|
| 1 | 32 | 6.9 MiB | 3.4 |
| 1 | 64 | 8.6 MiB | 4.3 |
| 1 | 128 | 11.8 MiB | 5.9 |
| 2 | 32 | 11.9 MiB | 6.0 |
| 2 | 64 | 14.4 MiB | 7.2 |

Doubling the batch (`N = 4`, a 4 MiB block) gives 14.2 MiB; doubling the sequence
(`L = 512`, an 8 MiB block) gives 26.3 MiB. Precision is about ±0.3 MiB: each figure is
the largest of five readings, and the allocator varies about that much between them.

A rule that bounds every one of those from above:

```
peak  ≈  A · (3 · score block  +  6.5 · q block)

  score block = N · H · Lq · Lk · 4 bytes
  q block     = N · H · L  · d  · 4 bytes
  A           = number of ScaledDotProductAttention calls
```

A 6-layer encoder at batch 32, 6 heads, `L = 1024`, `d = 64` is `A = 6`, a 768 MiB score
block and a 48 MiB q block: ≈ 6 × (2.25 + 0.30) GiB ≈ **15 GiB in attention activations
alone**, before parameters, gradients, optimizer state and other layers. Without the
q-block term you would budget 13.5 GiB, a tenth too low.

Two caveats when checking this against a real run. The measurements are CPU-side; on a
GPU the same tensors are allocated, but below a few hundred MiB a GPU reading shows
nothing, because the ONNX Runtime arena reserves a fixed few hundred MiB up front and the
blocks fit inside it. And the API that produces the figures is internal and unsupported.

**The lever: `queryChunks`.** `queryChunks: c` splits the query axis into `c` blocks,
runs attention on each against the whole key/value, and concatenates. It is exact: the
output matches the dense path to floating-point rounding, gradients and causal masking
included (each chunk gets a `queryOffset` causal mask).

**Measure it, and expect to pay compute.** It shrinks the score-sized tensors by `c`;
the saving depends on how much of the peak they are, and the cost is `c` MatMul and
Softmax launches instead of one. On the model above at `c = 4`, with peak measured as
above and compute taken from the pass's own model of the step (reproducible, unlike
kernel timings):

| `d` | `L` | dense | chunked | peak | modelled compute |
|---|---|---|---|---|---|
| 32 | 256 | 6.9 MiB | 5.0 MiB | **28% better** | +41% |
| 64 | 256 | 8.6 MiB | 7.1 MiB | **18% better** | +37% |
| 32 | 512 | 26.3 MiB | 16.5 MiB | **37% better** | +22% |

It is worth trying when a run is close to fitting, and pays best where the score block
dominates — long sequences, small head dims. At short sequences the retained q blocks
are most of the peak and chunking cannot reduce them. The compute cost is launch
overhead, so it is worst on small steps and shrinks as the sequence grows; check both
numbers afterwards.

An `additiveMask` is handled per chunk. Its query axis (axis -2, once right-aligned to
the scores' rank) must be `Lq` or `1`, as on the dense path; any other height is
rejected. A mask that already broadcasts over queries — rank < 2, or a size-1 axis -2,
as in an `[N, 1, 1, Lk]` padding mask — goes to every chunk whole; one `Lq` rows tall is
sliced to each chunk's rows.

`queryChunks` is fixed when the graph is built; `Lq` stays dynamic, and chunk `i` covers
rows `[Lq·i/c, Lq·(i+1)/c)`. An `Lq` that does not divide evenly gives chunks differing
by one row; a `c` larger than `Lq` gives empty chunks (correct, but wasted launches).
Keep `c` small: the graph grows with it — the built (pre-optimization) one-attention
training step goes from 505 to 1 308 nodes at `c = 4`.

It is available only on `Attention.ScaledDotProductAttention`, not `MultiHeadAttention`,
`TransformerEncoderLayer` or `TransformerDecoderLayer`: a `[Module]`'s parameters are
graph values fixed at concretization, after the module body has run, while a chunk count
must be a C# constant at build time. So only hand-assembled attention can use it.

**The other levers,** cheapest first: shorten the sequence (quadratic), shrink the batch
(linear), cut heads (linear). A block can also be recomputed in the backward pass — see
[Activation checkpointing](#activation-checkpointing).

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

This is PyTorch's `torch.utils.checkpoint`: every call of a checkpointed module is a
**segment** whose forward activations are dropped after the forward pass and recomputed
from the segment's inputs when the backward pass needs them. The segment's inputs and
outputs are kept; everything produced inside the body is recomputed once per segment and
shared by every gradient that reads it. The step's numbers do not change (a checkpointed
stack and its plain twin follow the same loss trajectory); the module's parameters,
state and checkpoints are unaffected; and the hint does not reach an exported ONNX model.
Nested checkpointed modules form one segment, the outer one.

The training rig honours the hint **unconditionally** — independently of the
compute-versus-memory objective of its memory-aware pass, and even on a step too small
for the pass to run. The automatic pass takes only recomputations that pay under its
objective and refuses one that buys memory with a large compute increase; the attribute
is how you request that trade anyway.

It is not always a win over the automatic pass, which can choose finer-grained
recomputations than a whole segment. On a three-block MLP of width 32 behind a linear
head, at a batch the pass leaves alone, the checkpointed twin's modelled peak is 40%
below the plain one's, for about 14% more modelled compute, with an identical loss
trajectory. At a batch large enough for the pass to act, the pass and the hint land
within a few percent of each other on that model; on a wide MLP or a two-layer
transformer encoder the hint came out slightly worse than the pass on both counts. Use
the hint for a segment whose activations dominate the peak and that the pass would leave
alone, or to buy memory at a known compute price. These are the pass's **modelled**
figures; before relying on the hint for a step that must fit, measure the step — see
[limitations.md](limitations.md#gradient-activation-checkpointing) for what the model
does and does not capture.

### PReLU / GLU

```csharp
PReLU.Call(x)                       // y = relu(x) - a*relu(-x); single shared [1] learnable slope (init 0.25)
PReLUChannelwise.Call(x)            // same formula, but a SEPARATE [C] learnable slope per channel (init 0.25)
GatedLinear.GLU(x, axis: -1)        // splits x in two halves [a, b] along axis -> a * sigmoid(b)
GLU.Call(x)                         // param-free module form of GatedLinear.GLU with axis fixed at -1
```

`PReLU` has one shared trainable slope (PyTorch `num_parameters=1`). `PReLUChannelwise`
(PyTorch `num_parameters=C`) has a `[C]` slope vector, sized in-graph from the input's
channel axis (axis 1) and broadcast as `[1, C, 1, …, 1]`. It is rank-generic (`[N, C]`,
`[N, C, L]`, `[N, C, H, W]`, …; rank ≥ 2, channels on axis 1), and its slopes also init
to `0.25`, so a fresh `PReLUChannelwise` equals a fresh `PReLU` numerically. (Per He et
al. 2015 the PReLU slope should not be weight-decayed; Shorokoo's optimizers apply decay
uniformly, so this is not enforced.) `GatedLinear.GLU` is a param-free static helper and
requires an even size along `axis`. The `GLU` **module** (`GLU.Call(x)`) wraps it with
the split axis fixed at `-1` (PyTorch `nn.GLU(dim=-1)`); use the helper for any other
axis.

### Dropout

```csharp
Dropout.Call(Scalar<float32> ratio, Scalar<bit> training, Tensor<float32> x)
```

Training mode zeroes each element with probability `ratio` and scales survivors by
`1/(1-ratio)`; eval mode is the identity. The mask is drawn from the layer's own stream
under the model's [RNG configuration](rng-configuration.md) — reproducible for a config
(the default identity when none is given) and varying per training step via the
generator's execution counter. Gradients flow through the forward mask.

### SpatialDropout

```csharp
SpatialDropout.Call(Scalar<float32> ratio, Scalar<bit> training, Tensor<float32> x)
Dropout1d.Call(Scalar<float32> ratio, Scalar<bit> training, Tensor<float32> x)  // [N, C, L]
Dropout2d.Call(Scalar<float32> ratio, Scalar<bit> training, Tensor<float32> x)  // [N, C, H, W]
Dropout3d.Call(Scalar<float32> ratio, Scalar<bit> training, Tensor<float32> x)  // [N, C, D, H, W]
```

Channel-wise dropout over `[N, C, D1..Dn]` (channel = axis 1): one Bernoulli draw per
`(sample, channel)` zeroes or rescales (by `1/(1-ratio)`) the **entire** feature map,
instead of `Dropout`'s per-element mask — the regularization for strongly correlated
conv feature maps (Tompson et al. 2015). Eval mode is the identity; the mask is drawn as
for `Dropout`. The rank is read in-graph, so `SpatialDropout` is rank-generic; the
`Dropout1d/2d/3d` aliases (PyTorch names) forward to it. On rank-2 `[N, C]` it is
elementwise `Dropout`.

### AlphaDropout / FeatureAlphaDropout

```csharp
AlphaDropout.Call(Scalar<float32> ratio, Scalar<bit> training, Tensor<float32> x)         // elementwise
FeatureAlphaDropout.Call(Scalar<float32> ratio, Scalar<bit> training, Tensor<float32> x)  // channel-wise [N,C,...]
```

SELU-paired dropout for self-normalizing networks (Klambauer et al. 2017). Dropped units
are set to SELU's negative saturation `α' = −λα ≈ −1.7581` instead of zero, then the
tensor is renormalized by `out = a·x' + b` with `a = (q + α'²·q·p)^(−1/2)`,
`b = −a·p·α'` (`q = 1−ratio`), which preserves mean and variance **in expectation** over
the mask — the self-normalizing property that plain `Dropout` destroys.
`FeatureAlphaDropout` is the channel-wise twin: one Bernoulli draw per
`(sample, channel)` drops a whole feature map to `α'` (`[N, C, 1, …, 1]` mask),
rank-generic over 1-D/2-D/3-D. Both take `(ratio, training)` like `Dropout`; eval mode
(`training = false`) is the **exact** identity (gated explicitly, since the affine is
not the identity); the mask is drawn as for `Dropout`.

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
- **`paddingIdx`** (sentinel `-1` = off): output rows whose index equals `paddingIdx` are
  masked to the zero vector, so they receive no training gradient.
- **`maxNorm`** (sentinel `0f` = off) / **`normType`** (conventionally `2f`, the *p* of
  the p-norm): gathered output rows whose `normType`-norm exceeds `maxNorm` are scaled
  down to `maxNorm` (shrink-only). `normType` is inert unless `maxNorm` is set.
- **No declared defaults**: all five are
  [non-nullable and always passed](#nullable-hypers). The sentinels and `2f` are
  call-site conventions you write out:
  `Embedding.Call(V, D, Scalar(-1L), Scalar(0f), Scalar(2f), indices)` is the knobs-off
  call.

**Two differences from PyTorch** (SSA graphs cannot mutate a weight mid-forward):
(1) `maxNorm` is *functional* — Shorokoo clamps the gathered **output** rows and never
changes the stored weight. PyTorch renormalizes the weight in place, so stored weights
diverge across training; per-forward outputs match. (2) `paddingIdx` masks the
**output** rather than zeroing the stored row and freezing its gradient; the mask routes
zero gradient to pad positions, so with a reserved pad id the training effect matches,
and the output is zero by construction.

**Choosing the initializer.** `[Module] Embedding` always uses `Normal`. For another
initializer use the static helper `EmbeddingHelpers.Embed(indices, numEmbeddings,
embeddingDim, embeddingInit, paddingIdx, maxNorm, normType)` — e.g.
`EmbeddingHelpers.Embed(idx, V, D, s => XavierUniform.Init(s))`. The init selector is a
compile-time choice, not a `[Hyper]`, so it lives on a plain-C#-argument helper; `Embed`
defaults to `Normal`.

**Not exposed.** `scale_grad_by_freq` and `sparse` are gradient-only knobs with no
forward expression and no Shorokoo IR support (also absent from Keras and Flax).

#### EmbeddingBag

```csharp
EmbeddingBag.Bag(Tensor<int64> indices, long numEmbeddings, long embeddingDim,
                 BagMode mode = BagMode.Mean, Func<...>? embeddingInit = null,
                 long paddingIdx = -1)   // BagMode { Sum, Mean, Max }
```

Looks up a trainable `[V, D]` table (`Normal` by default) for a **2-D batch of
fixed-length bags** `indices [B, L]` and reduces each bag over axis 1 by `mode`,
returning `[B, D]` — `Embedding(indices).Reduce(mode, axis=1)`. It is a static helper
because `mode` is a build-time structural choice; the table trains end-to-end.

**Limits:** 2-D fixed-length input only. PyTorch's 1-D `input + offsets` ragged form,
`include_last_offset` and `per_sample_weights` are not supported (the ragged reduce needs
a SegmentSum-style op that ONNX lacks); rectangularize to `[B, L]` with `paddingIdx`
instead. The `[B, L, D]` intermediate is materialized (no fused gather-reduce), so the
result matches PyTorch but the memory profile does not. `paddingIdx` zeroes pad rows
before the reduce: exact for `Sum`, approximate for `Mean` (divides by the full `L`, not
the non-pad count), and wrong for `Max` with negative embeddings (use it with `Max` only
for non-negative embeddings).

### Activations

Only activations with a hyperparameter have modules:

```csharp
LeakyReLU.Call(Scalar<float32> alpha, x)  // x for x > 0, alpha * x otherwise
ELU.Call(Scalar<float32> alpha, x)        // x for x > 0, alpha * (exp(x) - 1) otherwise
```

(The ONNX LeakyRelu/Elu ops take alpha as a static attribute, so these build the formula
in-graph to allow a `[Hyper]` alpha.)

Plain activations are tensor one-liners: `x.Relu()`, `x.Gelu()`, `x.Sigmoid()`,
`x.Tanh()`, `x.Softmax(axis)`, `x.LogSoftmax(axis)`, …

### Pooling — plain C# helpers

ONNX pooling geometry exists only as static node attributes, so `Pooling` is a static
class of helpers with plain C# arguments; the geometry is not hyperparameter-driven. Like
`Convolution`, the windowed pools come in three shapes: a **per-axis** form (geometry as
`long[]`, spatial rank from `kernelSize.Length`), a **scalar-square** overload (one
scalar per knob, rank from `x.Rank() - 2`), and per-rank `*1d/2d/3d` aliases that assert
the rank. Unlike `Convolution`, the helpers are generic in the float element type
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

**Both the scalar `MaxPool2d(x, long kernelSize, …)` / `AvgPool2d(x, long kernelSize, …)`
signatures and the per-axis `long[]` aliases exist** (overload resolution distinguishes
`long` from `long[]`), so `Pooling.MaxPool2d(x, 2)` and `Pooling.MaxPool2d(x, [2L, 2L])`
both work.

**Defaults & conventions.** `stride` defaults to `kernelSize` (PyTorch/Keras). Per-axis
`stride`/`padding`/`dilation` accept length 1 (broadcast to every axis) or spatialRank,
and `padding` may also be length `2*spatialRank` (ONNX `[begin₁…beginₙ, end₁…endₙ]`,
for asymmetric pads). `LpPool`'s `p` defaults to **2** (L2) and is an **integer** — ONNX
has no fractional norm, so PyTorch's float `norm_type` is not expressible. `AvgPool`'s
`countIncludePad` defaults to **false** (divide by the count of real cells), **unlike
PyTorch's `count_include_pad=True`**; pass `countIncludePad: true` for PyTorch's
denominator.

**Backward caveats.** The forward attribute surface is complete, but some gradients are
restricted:

- `AvgPool` with `ceilMode: true` **throws in the backward pass** — forward / inference
  only.
- `LpPool`'s gradient **ignores** `ceilMode`, `dilation` and `autoPad` — forward-correct,
  but the gradient is wrong when training with non-default values.
- `MaxPool` is exact for every attribute (ties route the gradient to the first max);
  `MaxUnpool` and the global pools are fully differentiable.

**Unsupported.** Adaptive pooling (`AdaptiveAvg/MaxPool*`) beyond `output_size == 1` —
which **is** `GlobalAvgPool2d`/`GlobalMaxPool2d`/`GlobalLpPool` — has no general ONNX
operator, and fractional (stochastic-window) max pooling has no core op.

## Losses (`Shorokoo.Modules.Losses`)

Sixteen losses: the **fourteen** two-input ones tabulated here, plus the **three-input**
metric-learning losses [`TripletMarginLoss`](#tripletmarginloss-metric--embedding-learning)
and [`CosineEmbeddingLoss`](#cosineembeddingloss-metric-learning) at the end of this
section.

For the fourteen, `Inline(predictions, targets)` returns a `Scalar<float32>` **mean**
loss and is the rig-safe default. `predictions`/`targets` are `Tensor<float32>` unless
noted. (The three-input pair does **not** satisfy the rig's 2-input loss contract; see
those entries.) Most losses also expose **configurable knobs** (reduction, class weights,
`ignore_index`, label smoothing, `pos_weight`, `beta`) through extra methods — see
[Configurable knobs](#loss-configurable-knobs).

| Module | Formula (per element, then mean) | Input contract |
|---|---|---|
| `L2Loss` | `(p − t)²` | predictions, targets (MSE; reduces over axis 0 — use rank-1/flattened predictions) |
| `L1Loss` | `\|p − t\|` | predictions, targets (MAE) |
| `HuberLoss` | `0.5·e²` if `\|e\| ≤ δ`, else `δ·(\|e\| − 0.5·δ)` | `(predictions, targets, delta hyper)` — see note below |
| `SmoothL1Loss` | Huber with `δ = 1` | predictions, targets |
| `CrossEntropyLoss` | softmax cross-entropy over logits | predictions `[N, C]` or `[N, C, d1, …]` logits; targets `[N]` or `[N, d1, …]` `Tensor<int64>` class indices. The class axis is axis 1 — see [Sequence logits](#cross-entropy-sequence-logits) |
| `NLLLoss` | `−log p[target]` | predictions `[N, C]` log-probs (e.g. `x.LogSoftmax(1)`); targets `[N]` `Tensor<int64>` |
| `BCELoss` | `−(t·ln p + (1−t)·ln(1−p))` | predictions are probabilities in (0, 1), clamped to `[1e-7, 1−1e-7]` |
| `BCEWithLogitsLoss` | `max(x, 0) − x·t + ln(1 + e^−\|x\|)` | predictions are raw logits (stable sigmoid+BCE) |
| `KLDivLoss` | `(1/N)·Σ p·(log p − log q)` (batchmean) | predictions are **log**-probs (log q), targets are probs (p); `p·log p = 0` at `p = 0` |
| `LogCoshLoss` | `log(cosh(p − t))` (stable `\|d\| + softplus(−2·\|d\|) − log 2`) | predictions, targets (hyperparameter-free Huber: ≈ `d²/2` small, ≈ `\|d\| − log 2` large; overflow-free) |
| `PoissonNLLLoss` | `exp(p) − t·p` (`logInput=true`); else `p − t·log(p + eps)` | predictions are the **log-rate** `log λ` (default) or rate `λ` (`logInput=false`); targets are `Tensor<float32>` counts |
| `HingeLoss` | `max(0, 1 − t·p) = relu(1 − t·p)` | predictions are raw scores; **targets MUST be `±1`**, **not** converted from `0/1` — map `0/1` upstream with `2·t − 1` |
| `SquaredHingeLoss` | `max(0, 1 − t·p)²` | as `HingeLoss` (**`±1` targets**); penalises margin violations quadratically (smooth at the boundary) |
| `BinaryFocalLoss` | `α_t · (1 − p_t)^γ · ce` (`ce` = stable BCE-with-logits; `p_t = p·t + (1−p)·(1−t)`, `p = σ(x)`) | predictions are raw **logits**; targets are binary `{0, 1}`. `α`/`γ` baked C# floats (defaults `0.25`/`2`, torchvision parity); `α = −1` disables α-weighting |

<a id="cross-entropy-sequence-logits"></a>
**Sequence logits (`CrossEntropyLoss`)**: the class axis must be axis 1. A language model's
`[B, L, V]` logits against `[B, L]` targets are not accepted: the rig builds, and the first step
fails inside ONNX Runtime with `[ShapeInferenceError] Incompatible dimensions`. Transpose the
logits to `[B, V, L]` (`logits.Transpose([0L, 2L, 1L])`), or flatten logits to `[B·L, V]` and
targets to `[B·L]`.

**HuberLoss vs SmoothL1Loss and the rig**: `HuberLoss`'s `delta` hyperparameter makes
its `ComputationGraph` a 3-input graph, but `TrainingRig`'s loss contract is exactly
`(predictions, targets)`, so `HuberLoss.ComputationGraph` cannot be handed to a rig. Use
`SmoothL1Loss` (delta = 1, 2-input), or call `HuberLoss.Inline(predictions, targets,
Scalar(d))` inside your own 2-input loss module.

<a id="loss-configurable-knobs"></a>
### Configurable knobs (`Reduced` / `PerElement`)

PyTorch's loss knobs are **build-time C# arguments** (baked into the graph, not
`[Hyper]`s) on two extra methods; `Inline(predictions, targets)` is unchanged:

- **`Reduced(…, LossReduction reduction = Mean)`** — returns a `Scalar<float32>` for
  `Mean`/`Sum` reduction.
- **`PerElement(…)`** — returns the per-element `Tensor<float32>` (the
  `reduction = None` form; C# can't overload on return type). For
  `CrossEntropyLoss`/`NLLLoss` it is **zero at `ignore_index` positions** (PyTorch/ONNX
  semantics).

`LossReduction` (in `Shorokoo.Modules.Losses`) is `None | Mean | Sum`, mapping to the
ONNX op's `"none"/"mean"/"sum"` (CE/NLL) or `ReduceKind.Mean/.Sum` (the regression/BCE
losses). `Reduced(..., reduction: None)` throws — use `PerElement`.

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

**`label_smoothing` (CE only)** blends the hard target with the uniform distribution:
`loss = (1−α)·NLL + α·(−(1/K)·Σ_k log p_k)`. It is built in-graph from
`LogSoftmax + NegativeLogLikelihoodLoss` (ONNX SoftmaxCrossEntropyLoss has no such
attribute), applying `weight`/`ignoreIndex` to **both** terms; `α = 0` uses the exact
single-op path. It adds no graph input and stays scalar, so `labelSmoothing` (like
`ignoreIndex` and a `Mean`/`Sum` `reduction`) is **rig-safe** on a wrapper module (see
below).

**SmoothL1 ↔ Huber**: `SmoothL1(e; β) = HuberLoss(δ = β) / β`, so
`SmoothL1Loss.Reduced(beta, p, t)` equals `HuberLoss(delta = beta)` divided by `beta`;
at `β = 1` it equals `SmoothL1Loss.Inline`. `HuberLoss`'s `delta` is a **live `[Hyper]`**
(schedulable); SmoothL1's `beta` is a **baked C# float**. For a live transition point,
use `HuberLoss` and divide by `delta` yourself.

**LogCosh stability**: `LogCoshLoss` computes the overflow-free identity
`log(cosh(d)) = |d| + softplus(−2·|d|) − log 2` (the naive `log((e^d + e^−d)/2)`
overflows for `|d| ≳ 89` in float32). It has no hyperparameter: the L2→L1 crossover
(≈ `d²/2` small, ≈ `|d| − log 2` large) is fixed by the function.

**PoissonNLL `logInput`/`full`**: `Inline` uses PyTorch's stable `logInput=true` form
`exp(p) − t·p` (the prediction is `log λ`). The Keras `Poisson` form
`p − t·log(p + ε)` is `Reduced(p, t, logInput: false, eps: 1e-7f)` (keep `p > 0`; `eps`
only guards exact `p = 0`). `full=true` adds Stirling's approximation of the dropped
`log(t!)` constant (`t·log t − t + 0.5·log(2π·t)` for `t > 1`, else 0), as PyTorch does;
it uses a clamped `max(t, 1)` inside the logs so the discarded `Where` lane stays finite
(no `0·log 0` NaN at `t = 0`).

#### Which knobs reach the rig

The rig passes the loss graph exactly **two tensor inputs** and expects a
**`Scalar<float32>`** out. So:

- **Rig-safe** (no extra input, stays scalar): `reduction = Mean`/`Sum`, `ignoreIndex`,
  `labelSmoothing` — via a **wrapper module**, since the generated `ComputationGraph`
  uses the bare `Inline`. Write a 2-input `[Module]` whose `Inline` calls `Reduced(...)`
  with the knobs baked:

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
  tensor): available on `Reduced`/`PerElement`, but the default rig path cannot bind the
  extra input. To train with a class `weight`/`posWeight`, **bake it as a graph
  constant** inside a 2-input wrapper module:

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

  The weight is a graph **constant**, not a fed input, so the wrapper has two inputs and
  satisfies the rig contract. There is no `WithWeight` factory; the wrapper module is the
  supported recipe.

**Or move the loss into the model.** `Inline`, `Reduced` and `PerElement` are plain
static graph builders, callable from **any** `[Module]` body, including your model's. A
model whose `Inline` ends in, say, `CrossEntropyLoss.PerElement(logits, labels,
ignoreIndex: 0L)` and then weights and reduces that tensor itself has a
`Scalar<float32>` output, takes its labels / mask / weights as ordinary model inputs, and
trains against a **pass-through** loss module that forwards its predictions input and
ignores its targets. That lifts every restriction above, at the price of a model whose
output is a loss rather than a prediction. See
[training.md → A loss graph may ignore its `targets`](training.md#loss-ignoring-targets):
the rig sees that the loss never reads its target and derives no target slot, so the
model trains on its inputs alone.

### TripletMarginLoss (metric / embedding learning)

For an anchor `a`, positive `p` and negative `n`, `L = max(0, d(a,p) − d(a,n) + margin)`
with p-norm distance `d(x,y) = (Σ|x−y|^p + eps)^(1/p)` over the last axis. Knobs: `margin`
(default 1), `p` (2 ⇒ Euclidean), `eps` (1e-6), and `swap` (Balntas anchor swap —
replaces `d(a,n)` with `min(d(a,n), d(p,n))`). `Inline` mean-reduces to a scalar;
`Reduced(…, LossReduction)` does mean|sum; `PerElement` returns the unreduced `[N]` vector.

**This is a 3-input loss** — `TripletMarginLoss.Call(margin, p, eps, swap, anchor, positive, negative)` —
**not** a 2-input `(pred, target)` rig loss. To train it with the rig, make it the
**tail of your model** (the model computes the three embeddings and returns the scalar
loss). `TripletMarginWithDistance` is the same objective with a caller-supplied distance
`Func<Tensor<float32>, Tensor<float32>, Tensor<float32>>` (e.g. cosine) replacing the
p-norm (static helper; `.Reduced`/`.PerElement`).

### CosineEmbeddingLoss (metric learning)

Cosine contrastive loss over two embedding batches `x1`, `x2` (`[N, D]`) and per-sample
labels `y ∈ {+1, −1}`: `L_i = 1 − cos(x1_i, x2_i)` for `y=+1` (pull similar pairs
together), `L_i = max(0, cos(x1_i, x2_i) − margin)` for `y=−1` (push dissimilar pairs
apart). `cos` is over the last axis with PyTorch's denominator floor
`max(‖x1‖·‖x2‖, eps)`. Knobs: `margin` (default 0, affects only the `y=−1` arm) and
`eps` (default 1e-8), both `[Hyper] Scalar<float32>`. Labels must be `±1` (map `2t−1`
upstream, as with `HingeLoss`). Like `TripletMarginLoss` it is a 3-input loss —
`CosineEmbeddingLoss.Call(margin, eps, x1, x2, y)` — with `Inline`/`Reduced`/
`PerElement`. `CosineEmbeddingLoss.CosineSimilarity(x1, x2, eps)` is the per-row
cosine-similarity primitive (PyTorch `nn.CosineSimilarity`).

## Optimizers (`Shorokoo.Modules.Optimizers`)

Each operates on one parameter at a time (`(hypers..., currentParam, grad) ->
updatedParam`); the rig applies it per field across the trainable parameter struct.
State tensors are not in the signature: each is created inside the optimizer body by an
optimizer state initializer — `OptimizerStateZeros` at the parameter's shape
(zero-filled param-shaped state), `OptimizerScalarZeros` for a rank-0 scalar seeded at 0
(e.g. Adam's and AdamW's `step`), or `OptimizerScalarOnes` for a rank-0 scalar seeded
at 1 (e.g. NAdam's running momentum product) — and updated via `Globals.StateUpdate`.
Each optimizer gets a generated hyperparameter set (`<Name>Hyperparameters`, e.g.
`AdamOptimizerHyperparameters`) — see [training.md](training.md) for schedules, the
`Hyperparameter` kinds, and writing a custom optimizer.

| Module | Update rule | Hyper defaults | State per param |
|---|---|---|---|
| `SGDOptimizer` | `p −= lr·g` | `lr 0.01` | — |
| `SGDMomentumOptimizer` | `v = μ·v + g; p −= lr·v` | `lr 0.01, μ 0.9` | velocity |
| `AdamOptimizer` | `m, v` EMAs, **bias-corrected**: `p −= lr·m̂/(√v̂ + ε)` | `lr 0.001, β1 0.9, β2 0.999, ε 1e-8` | m, v, step |
| `AdamWOptimizer` | **Bias-corrected** Adam step + decoupled decay `p *= 1 − lr·wd` | `lr 0.001, β1 0.9, β2 0.999, ε 1e-8, wd 1e-4` | m, v, step |
| `RMSpropOptimizer` | `sq = α·sq + (1−α)·g²; buf = μ·buf + g/(√sq + ε); p −= lr·buf` | `lr 0.01, α 0.99, ε 1e-8, μ 0` | squareAvg, momentumBuffer |
| `AdagradOptimizer` | `acc += g²; p −= lr·g/(√acc + ε)` | `lr 0.01, ε 1e-10` | accumulator |
| `AdamaxOptimizer` | Adam with the ∞-norm: `m` EMA; `u = max(β2·u, \|g\|+ε)`; `p −= (lr/(1−β1ᵗ))·m/u` (no bias-correction on `u`) | `lr 0.002, β1 0.9, β2 0.999, ε 1e-8` | m, u, step |
| `NAdamOptimizer` | Nesterov-Adam: `μ_t` schedule + running product `∏μ`; `m̂` blends `μ_{t+1}·m` & `(1−μ_t)·g`, bias-corrected `v`: `p −= lr·m̂/(√v̂ + ε)` | `lr 0.002, β1 0.9, β2 0.999, ε 1e-8, ψ 0.004` | m, v, step, muProduct |
| `RAdamOptimizer` | Rectified Adam: bias-corrected `m̂`; if `ρ_t > 5` rectified adaptive `p −= lr·m̂·r_t·l_t`, else un-adapted `p −= lr·m̂` (runtime `Where`) | `lr 0.001, β1 0.9, β2 0.999, ε 1e-8` | m, v, step |
| `AdadeltaOptimizer` | `sq = ρ·sq + (1−ρ)·g²; Δx = √(accΔ+ε)/√(sq+ε)·g; accΔ = ρ·accΔ + (1−ρ)·Δx²; p −= lr·Δx` (ε **inside** the √s; `lr` is a step multiplier) | `lr 1.0, ρ 0.9, ε 1e-6` | squareAvg, accDelta |
| `LionOptimizer` | Sign-momentum + decoupled decay: `u = sign(β1·m + (1−β1)·g); p −= lr·(u + wd·p); m = β2·m + (1−β2)·g` (`m` decayed by **β2**; β1 only in the sign blend) | `lr 1e-4, β1 0.9, β2 0.99, wd 0` | m |
| `AdafactorOptimizer` | **Non-factored** Adafactor: `β̂2ₜ = 1 − tᵗᵃᵘ; ρ = min(lr, 1/√t); α = max(ε₂, RMS(p))·ρ; V = β̂2ₜ·V + (1−β̂2ₜ)·(g²+ε₁); U = g/max(√V, ε₁); Û = U/max(1, RMS(U)/d); p = p·(1−lr·wd) − α·Û` (full param-shaped `V`, **no** row/col factoring) | `lr 0.01, τ −0.8, ε₁ 1e-30, ε₂ 1e-3, d 1.0, wd 0` | v, step |
| `LambOptimizer` | LAMB (You et al. 2019): bias-corrected Adam direction scaled by a **per-tensor trust ratio** — `r = m̂/(√v̂ + ε); u = r + wd·p; trust = (‖p‖>0 ∧ ‖u‖>0) ? ‖p‖/‖u‖ : 1; p −= lr·trust·u` (ε **outside** the √; decoupled `wd` **inside** the trust numerator; ‖·‖ = `√Σx²` over the whole tensor; φ = identity). Per-tensor = layer-wise, since the optimizer runs per parameter tensor. | `lr 1e-3, β1 0.9, β2 0.999, ε 1e-6` (not 1e-8), `wd 0.01` | m, v, step |

- **Bias correction — Adam and AdamW**: both apply `m̂ = m/(1−β1^t)`,
  `v̂ = v/(1−β2^t)`, carrying the timestep `t` as a third state field — a **scalar** (one
  float per parameter, created by `OptimizerScalarZeros`) that broadcasts against
  `m̂`/`v̂`. Adam's first step is ≈ `lr` whatever the gradient magnitude, and at `wd = 0`
  `AdamWOptimizer` matches `AdamOptimizer` step for step.
- `RMSpropOptimizer` with the default `momentum = 0` is plain RMSprop; it always carries
  both state tensors.
- `AdamaxOptimizer` replaces Adam's second moment with an exponentially weighted
  infinity norm `u` (a running max) and bias-corrects `m` only — the running max needs no
  correction. `ε` is **inside** the max (PyTorch: `u = max(β2·u, |g|+ε)`).
- `NAdamOptimizer` adds Nesterov look-ahead via Dozat's momentum schedule
  `μ_t = β1·(1 − ½·0.96^(t·ψ))` and a **running product** `∏μ_i`, a second **scalar**
  state (`muProduct`) seeded at **1.0** by `OptimizerScalarOnes` (seeding at 0 would pin
  the product at 0). No weight decay, matching PyTorch's defaults.
- `RAdamOptimizer` rectifies Adam's adaptive step by `r_t` and, while the adaptive
  variance is not yet tractable (`ρ_t ≤ 5`, the first ~4–5 steps at `β2 = 0.999`), falls
  back to an un-adapted bias-corrected momentum step. The `ρ_t > 5` test depends on the
  `step` state, so it is a runtime scalar `Where` selecting between the two updates (not
  an `If` subgraph). The `m`/`v`/`step` updates are the same in both arms and registered
  once.
- `AdadeltaOptimizer` needs no hand-set learning rate: the ratio
  `√(E[Δx²]) / √(E[g²])` self-scales the step and corrects its units. As in the paper and
  PyTorch, `ε` is **inside** both square roots (Adagrad/RMSprop/Adam add it after the √),
  and `Δx` reads the **previous** step's `accDelta` before that accumulator is updated.
  `lr` is a step **multiplier** (default `1.0` ≡ the paper's lr-free method).
- `LionOptimizer` takes the **sign** of a β1-blend of momentum and gradient, so every
  coordinate moves by ±`lr` regardless of gradient scale. It stores **only** `m` (no
  second moment, no timestep), so its state is **half** Adam/AdamW's param-shaped state,
  with no scalar. **Beta roles are swapped relative to Adam:** `m` is decayed by **β2**
  (`m = β2·m + (1−β2)·g`), and **β1** appears only in the sign blend. Weight decay is
  decoupled (AdamW-style). A good Lion `lr` is typically **3–10× smaller** than AdamW's
  and its `wd` **3–10× larger** (effective decay ≈ `lr·wd`); the default `wd = 0` matches
  the reference, so set the decay yourself. There is no `ε` (no division).
- `AdafactorOptimizer` is the **non-factored** Adafactor: relative step
  `ρ = min(lr, 1/√t)`, parameter scaling `α = max(ε₂, RMS(p))·ρ`, and RMS update clipping
  `Û = U/max(1, RMS(U)/d)`, over a time-increasing decay `β̂2ₜ = 1 − t^τ`. `RMS(·)` reduces
  over **all** elements, so the step is rank-agnostic. **Divergence:** real Adafactor
  stores only row (`[r]`) + column (`[c]`) accumulators and reconstructs the second
  moment as their outer product — its **sublinear-memory** trick. That factoring is
  **not implemented** (a per-rank branch would thread out differently shaped state, which
  ONNX `If` cannot return). State is a **full param-shaped** `v` plus a scalar `step` —
  the **same footprint as Adam**: Adafactor's update dynamics without its memory saving.
  `learningRate` (default `0.01`) is the **cap** on `ρ`, not a fixed step.

## End-to-end: tiny conv net + CrossEntropyLoss + Adam

A complete classifier built from library layers and trained with `TrainingRig`. Layer
hypers are fixed via `Model(...)` so the model graph is inputs-only, as the rig requires:

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

(Equivalently, batch the data as `TensorDataStruct[]` and call
`rig.Fit(inputs, targets, numEpochs)` — see [training.md](training.md).)

## Anti-patterns

- Do not hand `HuberLoss.ComputationGraph` to `TrainingRig` (3-input graph); use
  `SmoothL1Loss` or wrap it in a 2-input module.
- Do not run a BatchNorm-containing graph through the plain inference executor when the
  running stats matter: on a one-shot run the stats stay as the initializer left them
  and the update is dropped.
- Do not expect different stride/padding from `ConvTranspose2d`'s hypers — its geometry
  is fixed at the ONNX defaults; use `NN.ConvTranspose`.
- Do not use `XavierUniform`/`KaimingUniform` (etc.) for rank-1 biases; they require
  rank ≥ 2.
- Do not restructure a model to avoid the parameters of a switched-off
  `useBias`/`affine`; the unselected branch's params are pruned
  ([An off toggle costs nothing](#gated-parameters)).
