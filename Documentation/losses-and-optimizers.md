# Losses and optimizers

The losses and optimizers of `Shorokoo.Modules`: what each computes, the knobs a loss takes, and
the state each optimizer keeps.

Related: [nn-library.md](nn-library.md) · [training.md](training.md) · [training-hyperparameters.md](training-hyperparameters.md)

## Losses (`Shorokoo.Modules.Losses`)

Sixteen losses: the **fourteen** two-input ones below, plus the **three-input**
[`TripletMarginLoss`](#tripletmarginloss-metric--embedding-learning) and
[`CosineEmbeddingLoss`](#cosineembeddingloss-metric-learning).

For the fourteen, `Inline(predictions, targets)` returns a `Scalar<float32>` **mean**
loss and is the rig-safe form. Inputs are `Tensor<float32>` unless noted. Extra knobs
are on `Reduced` / `PerElement` — see [Configurable knobs](#loss-configurable-knobs).

| Module | Formula (per element, then mean) | Input contract |
|---|---|---|
| `L2Loss` | `(p − t)²` | predictions, targets of any rank (MSE over all elements) |
| `L1Loss` | `\|p − t\|` | predictions, targets (MAE) |
| `HuberLoss` | `0.5·e²` if `\|e\| ≤ δ`, else `δ·(\|e\| − 0.5·δ)` | `(predictions, targets, delta hyper)` — see note below |
| `SmoothL1Loss` | Huber with `δ = 1` | predictions, targets |
| `CrossEntropyLoss` | softmax cross-entropy over logits | predictions `[N, C]` or `[N, C, d1, …]` logits; targets `[N]` or `[N, d1, …]` `Tensor<int64>` class indices. The class axis is axis 1 — see [Sequence logits](#cross-entropy-sequence-logits) |
| `NLLLoss` | `−log p[target]` | predictions `[N, C]` or `[N, C, d1, …]` log-probs (e.g. `x.LogSoftmax(1)`); targets `[N]` or `[N, d1, …]` `Tensor<int64>`. The class axis is axis 1, as for [`CrossEntropyLoss`](#cross-entropy-sequence-logits) |
| `BCELoss` | `−(t·ln p + (1−t)·ln(1−p))` | predictions are probabilities, clamped to `[1e-7, 1−1e-7]` |
| `BCEWithLogitsLoss` | `max(x, 0) − x·t + ln(1 + e^−\|x\|)` | predictions are raw logits |
| `KLDivLoss` | `(1/N)·Σ p·(log p − log q)` (batchmean) | predictions are **log**-probs, targets are probs; `p·log p = 0` at `p = 0` |
| `LogCoshLoss` | `log(cosh(p − t))`, computed overflow-free | predictions, targets (≈ `d²/2` small, ≈ `\|d\| − log 2` large) |
| `PoissonNLLLoss` | `exp(p) − t·p` (`logInput=true`); else `p − t·log(p + eps)` | predictions are `log λ` (default) or `λ`; targets are float counts |
| `HingeLoss` | `max(0, 1 − t·p)` | raw scores; **targets MUST be `±1`** (map `0/1` with `2·t − 1`) |
| `SquaredHingeLoss` | `max(0, 1 − t·p)²` | as `HingeLoss` (**`±1` targets**) |
| `BinaryFocalLoss` | `α_t · (1 − p_t)^γ · ce` (`p = σ(x)`) | raw **logits**; targets `{0, 1}`; `α`/`γ` default `0.25`/`2`; `α = −1` disables α-weighting |

<a id="cross-entropy-sequence-logits"></a>
**Sequence logits (`CrossEntropyLoss`)**: the class axis must be axis 1. A language model's
`[B, L, V]` logits against `[B, L]` targets are not accepted: the rig builds, and the first step
fails inside ONNX Runtime with `[ShapeInferenceError] Incompatible dimensions`. Transpose the
logits to `[B, V, L]` (`logits.Transpose([0L, 2L, 1L])`), or flatten logits to `[B·L, V]` and
targets to `[B·L]`.

**HuberLoss and the rig**: `delta` makes `HuberLoss.ComputationGraph` a 3-input graph, but
the rig's loss contract is exactly `(predictions, targets)`. Use `SmoothL1Loss`, or call
`HuberLoss.Inline(predictions, targets, Scalar(d))` in your own 2-input loss module.

<a id="loss-configurable-knobs"></a>
### Configurable knobs (`Reduced` / `PerElement`)

Knobs are **build-time C# arguments** on two extra methods:

- **`Reduced(…, LossReduction reduction = Mean)`** returns a `Scalar<float32>`
  (`Mean`/`Sum`; `None` throws).
- **`PerElement(…)`** returns the unreduced `Tensor<float32>`. For
  `CrossEntropyLoss`/`NLLLoss` it is zero at `ignoreIndex` positions.

`LossReduction` (`Shorokoo.Modules.Losses`) is `None | Mean | Sum`.

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

- **`labelSmoothing`** (CE): `loss = (1−α)·NLL + α·(−(1/K)·Σ_k log p_k)`, with
  `weight`/`ignoreIndex` applied to both terms.
- **`ignoreIndex`** (CE, NLL): a target equal to it adds nothing to the loss or the gradient, and
  `Mean` divides by the targets that are not ignored, as PyTorch does: by their count, or with
  `weight` by the sum of their classes' weights. The sentinel is any `int64`, a negative one such
  as PyTorch's `-100` included. On the CPU backend, a training step whose batch holds an ignored
  target fails when the sentinel lies outside `[-C, C-1]` for `C` classes — `-100` below 100
  classes, say — because the gradient indexes the class weights with it
  ([#499](https://github.com/Shorokoo/Shorokoo/issues/499)); evaluating the loss does not. Until
  that is fixed, pick a sentinel inside `[-C, C-1]` for a model you train.
- **SmoothL1 ↔ Huber**: `SmoothL1(e; β) = HuberLoss(δ = β) / β`. Huber's `delta` is a
  live, schedulable `[Hyper]`; SmoothL1's `beta` is baked.
- **PoissonNLL**: the Keras `Poisson` form is
  `Reduced(p, t, logInput: false, eps: 1e-7f)` (keep `p > 0`). `full=true` adds
  Stirling's term `t·log t − t + 0.5·log(2π·t)` for `t > 1` (else 0), as PyTorch does,
  NaN-free at `t = 0`.

#### Which knobs reach the rig

The rig gives the loss graph exactly **two tensor inputs** and expects a
**`Scalar<float32>`**.

- **Rig-safe**: `reduction = Mean`/`Sum`, `ignoreIndex` (with a sentinel inside `[-C, C-1]`;
  see [`ignoreIndex`](#loss-configurable-knobs) above), `labelSmoothing` — through a
  **wrapper module**, since the generated `ComputationGraph` uses the bare `Inline`:

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

- **`weight` / `posWeight`** add a third tensor input the rig cannot bind; bake them as
  a graph constant in a wrapper module:

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

**Or move the loss into the model.** `Inline`, `Reduced` and `PerElement` can be called
from any `[Module]` body. A model that ends in, say,
`CrossEntropyLoss.PerElement(logits, labels, ignoreIndex: 0L)`, weights and reduces it,
and takes labels/masks/weights as ordinary inputs, trains against a pass-through loss
that ignores its targets — lifting every restriction above. See
[training.md → A loss graph may ignore its `targets`](training.md#loss-ignoring-targets).

### TripletMarginLoss (metric / embedding learning)

`L = max(0, d(a,p) − d(a,n) + margin)` with `d(x,y) = (Σ|x−y|^p + eps)^(1/p)` over the
last axis. Knobs: `margin` (default 1), `p` (default 2), `eps` (default 1e-6), `swap`
(anchor swap: `d(a,n)` becomes `min(d(a,n), d(p,n))`). `Inline` returns the mean;
`Reduced`/`PerElement` as above (`PerElement` gives `[N]`).

It is a 3-input loss — `TripletMarginLoss.Call(margin, p, eps, swap, anchor, positive, negative)` —
not a rig loss: to train with the rig, compute it at the **end of your model**.
`TripletMarginWithDistance` (static helper, with `Reduced`/`PerElement`) takes a custom distance
`Func<Tensor<float32>, Tensor<float32>, Tensor<float32>>` instead of the p-norm.

### CosineEmbeddingLoss (metric learning)

Over `x1`, `x2` (`[N, D]`) and labels `y ∈ {+1, −1}`: `L_i = 1 − cos(x1_i, x2_i)` for
`y=+1`, `max(0, cos(x1_i, x2_i) − margin)` for `y=−1`. `cos` is over the last axis with
denominator floor `max(‖x1‖·‖x2‖, eps)`. Knobs `margin` (default 0) and `eps` (default
1e-8). Labels must be `±1`. A 3-input loss —
`CosineEmbeddingLoss.Call(margin, eps, x1, x2, y)` — with `Inline`/`Reduced`/`PerElement`.
`CosineEmbeddingLoss.CosineSimilarity(x1, x2, eps)` is the per-row cosine similarity.

## Optimizers (`Shorokoo.Modules.Optimizers`)

Each updates one parameter at a time (`(hypers..., currentParam, grad) -> updatedParam`);
the rig applies it to every trainable parameter. State is created inside the optimizer by
`OptimizerStateZeros` (param-shaped, zero), `OptimizerScalarZeros` (rank 0, seeded 0,
e.g. Adam's `step`) or `OptimizerScalarOnes` (rank 0, seeded 1, e.g. NAdam's momentum
product), and updated via `Globals.StateUpdate`. Each optimizer has a generated
hyperparameter set (`<Name>Hyperparameters`, e.g. `AdamOptimizerHyperparameters`); see
[training-hyperparameters.md](training-hyperparameters.md) for schedules and custom optimizers.

| Module | Update rule | Hyper defaults | State per param |
|---|---|---|---|
| `SGDOptimizer` | `p −= lr·g` | `lr 0.01` | — |
| `SGDMomentumOptimizer` | `v = μ·v + g; p −= lr·v` | `lr 0.01, μ 0.9` | velocity |
| `AdamOptimizer` | `m, v` EMAs, **bias-corrected**: `p −= lr·m̂/(√v̂ + ε)` | `lr 0.001, β1 0.9, β2 0.999, ε 1e-8` | m, v, step |
| `AdamWOptimizer` | **Bias-corrected** Adam step + decoupled decay `p *= 1 − lr·wd` | `lr 0.001, β1 0.9, β2 0.999, ε 1e-8, wd 1e-4` | m, v, step |
| `RMSpropOptimizer` | `sq = α·sq + (1−α)·g²; buf = μ·buf + g/(√sq + ε); p −= lr·buf` | `lr 0.01, α 0.99, ε 1e-8, μ 0` | squareAvg, momentumBuffer |
| `AdagradOptimizer` | `acc += g²; p −= lr·g/(√acc + ε)` | `lr 0.01, ε 1e-10` | accumulator |
| `AdamaxOptimizer` | `m` EMA; `u = max(β2·u, \|g\|+ε)`; `p −= (lr/(1−β1ᵗ))·m/u` | `lr 0.002, β1 0.9, β2 0.999, ε 1e-8` | m, u, step |
| `NAdamOptimizer` | Nesterov-Adam: `μ_t = β1·(1 − ½·0.96^(t·ψ))`, running product `∏μ`; `m̂` blends `μ_{t+1}·m` & `(1−μ_t)·g`; `p −= lr·m̂/(√v̂ + ε)` | `lr 0.002, β1 0.9, β2 0.999, ε 1e-8, ψ 0.004` | m, v, step, muProduct |
| `RAdamOptimizer` | Rectified Adam: if `ρ_t > 5` `p −= lr·m̂·r_t·l_t`, else `p −= lr·m̂` | `lr 0.001, β1 0.9, β2 0.999, ε 1e-8` | m, v, step |
| `AdadeltaOptimizer` | `sq = ρ·sq + (1−ρ)·g²; Δx = √(accΔ+ε)/√(sq+ε)·g; accΔ = ρ·accΔ + (1−ρ)·Δx²; p −= lr·Δx` | `lr 1.0, ρ 0.9, ε 1e-6` | squareAvg, accDelta |
| `LionOptimizer` | `u = sign(β1·m + (1−β1)·g); p −= lr·(u + wd·p); m = β2·m + (1−β2)·g` | `lr 1e-4, β1 0.9, β2 0.99, wd 0` | m |
| `AdafactorOptimizer` | **Non-factored**: `β̂2ₜ = 1 − tᵗᵃᵘ; ρ = min(lr, 1/√t); α = max(ε₂, RMS(p))·ρ; V = β̂2ₜ·V + (1−β̂2ₜ)·(g²+ε₁); U = g/max(√V, ε₁); Û = U/max(1, RMS(U)/d); p = p·(1−lr·wd) − α·Û` | `lr 0.01, τ −0.8, ε₁ 1e-30, ε₂ 1e-3, d 1.0, wd 0` | v, step |
| `LambOptimizer` | `r = m̂/(√v̂ + ε); u = r + wd·p; trust = (‖p‖>0 ∧ ‖u‖>0) ? ‖p‖/‖u‖ : 1; p −= lr·trust·u` (norms over the whole tensor) | `lr 1e-3, β1 0.9, β2 0.999, ε 1e-6, wd 0.01` | m, v, step |

- **Adam / AdamW** bias-correct with a scalar `step` per parameter; the first step is
  ≈ `lr`. At `wd = 0`, AdamW equals Adam. Both compute the step as
  `m/(√v + ε·√(1−β2^t)) · lr·√(1−β2^t)/(1−β1^t)`, with the corrections folded into scalars so the
  update makes as few full passes over the parameter as it can; for a large parameter (an embedding
  table, say) the number of passes, not the arithmetic, sets the optimizer's share of a step. On
  the ONNX Runtime CPU backends a training step runs each Adam or AdamW update as one fused pass,
  which reads the parameter, its gradient and both moments once and writes the parameter and the
  moments once, with the same result to the bit as the operators it is written as; on a CUDA
  backend it runs as those operators, twelve passes over the parameter (thirteen with a weight
  decay). Every element is updated every step, rows the batch did not read included.
- **RMSprop** with `momentum = 0` is plain RMSprop (both state tensors still exist).
- **Adamax** puts `ε` inside the max and bias-corrects only `m`.
- **NAdam** has no weight decay.
- **RAdam** uses the un-adapted step while `ρ_t ≤ 5` (the first ~4–5 steps at
  `β2 = 0.999`).
- **Adadelta**: `ε` is inside both square roots; `lr` is a multiplier (default `1.0`, the
  paper's lr-free method).
- **Lion**: `m` decays with **β2**; β1 is used only in the sign blend. Every coordinate
  moves by ±`lr`. Its state is half of Adam's. Use an `lr` 3–10× smaller and a `wd` 3–10×
  larger than for AdamW; the default `wd = 0` must be set yourself.
- **Adafactor** does **not** factor the second moment into row/column accumulators, so it
  uses **as much memory as Adam** (full param-shaped `v` plus `step`); it keeps
  Adafactor's update rule. `learningRate` caps `ρ`.
- **LAMB** trust ratio is per parameter tensor (= per layer); `ε` default `1e-6`.
