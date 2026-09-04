# Training models

Related: [defining-models.md](defining-models.md) · [nn-library.md](nn-library.md) · [inference.md](inference.md)

## Facts

- Training composes three graphs: a **model**, a **loss**, and an **optimizer**, all
  `[Module]` classes accessed via their `.ComputationGraph` property.
- `TrainingRig` is the entry point. It runs autodiff on the composed graph and
  produces a trainable step.
- Gradients are produced by automatic differentiation; you do not write backward
  passes.
- State (optimizer moments, momentum velocity, BatchNorm running stats) is **created**
  by a `[StateInitializer]` class's `Init(...)` call inside a module's `Inline` (the
  state analog of trainable-parameter initializers) and its per-step update is
  registered via `Globals.StateUpdate(state, newState)`. `StateUpdate` throws
  `InvalidStateUpdateException` if its first argument is not a state variable —
  a runtime input or a trainable parameter is rejected.

## Built-in components

Ready-made losses and optimizers ship in the `Shorokoo.Modules` package
(namespaces `Shorokoo.Modules.Losses` / `Shorokoo.Modules.Optimizers`) — see
[nn-library.md](nn-library.md) for the full catalog (sixteen losses; layers and
initializers too). Each optimizer whose hyperparameters are all tensor-shaped gets a
source-generated, named, defaulted hyperparameter set
(`<Optimizer>Hyperparameters`) implementing `IOptimizerHyperparameters`. The
thirteen optimizers (the positional `params Hyperparameter[]` count for `FromScratch`
equals each set's property count):

| Optimizer | Hyperparameter set (named, init-only `Hyperparameter` properties; defaults from `[Hyper]`) |
|---|---|
| `SGDOptimizer` | `SGDOptimizerHyperparameters { LearningRate = 0.01 }` |
| `SGDMomentumOptimizer` | `SGDMomentumOptimizerHyperparameters { LearningRate = 0.01, MomentumCoeff = 0.9 }` |
| `AdamOptimizer` | `AdamOptimizerHyperparameters { LearningRate = 0.001, Beta1 = 0.9, Beta2 = 0.999, Epsilon = 1e-8 }` |
| `AdamWOptimizer` | `AdamWOptimizerHyperparameters { LearningRate = 0.001, Beta1 = 0.9, Beta2 = 0.999, Epsilon = 1e-8, WeightDecay = 1e-4 }` |
| `RMSpropOptimizer` | `RMSpropOptimizerHyperparameters { LearningRate = 0.01, Alpha = 0.99, Epsilon = 1e-8, Momentum = 0 }` |
| `AdagradOptimizer` | `AdagradOptimizerHyperparameters { LearningRate = 0.01, Epsilon = 1e-10 }` |
| `AdamaxOptimizer` | `AdamaxOptimizerHyperparameters { LearningRate = 0.002, Beta1 = 0.9, Beta2 = 0.999, Epsilon = 1e-8 }` |
| `NAdamOptimizer` | `NAdamOptimizerHyperparameters { LearningRate = 0.002, Beta1 = 0.9, Beta2 = 0.999, Epsilon = 1e-8, MomentumDecay = 0.004 }` |
| `RAdamOptimizer` | `RAdamOptimizerHyperparameters { LearningRate = 0.001, Beta1 = 0.9, Beta2 = 0.999, Epsilon = 1e-8 }` |
| `AdadeltaOptimizer` | `AdadeltaOptimizerHyperparameters { LearningRate = 1.0, Rho = 0.9, Epsilon = 1e-6 }` |
| `LionOptimizer` | `LionOptimizerHyperparameters { LearningRate = 1e-4, Beta1 = 0.9, Beta2 = 0.99, WeightDecay = 0 }` (4 positional) |
| `AdafactorOptimizer` | `AdafactorOptimizerHyperparameters { LearningRate = 0.01, Beta2Decay = -0.8, Epsilon1 = 1e-30, Epsilon2 = 1e-3, ClipThreshold = 1.0, WeightDecay = 0 }` (6 positional; **non-factored** — full param-shaped 2nd moment, no row/col factoring) |
| `LambOptimizer` | `LambOptimizerHyperparameters { LearningRate = 0.001, Beta1 = 0.9, Beta2 = 0.999, Epsilon = 1e-6, WeightDecay = 0.01 }` (5 positional; `Epsilon` is LAMB's `1e-6`, not Adam's `1e-8`) |

A loss module has signature `(predictions, targets) -> Scalar<float32>` with exactly two tensor inputs; targets are typically `Tensor<float32>`, but class-index losses (`CrossEntropyLoss`, `NLLLoss`) take `Tensor<int64>` targets.
The library losses' configurable knobs (`reduction`, `ignore_index`, `label_smoothing`, class `weight`/`pos_weight`, SmoothL1 `beta`) live on extra `Reduced`/`PerElement` methods, *not* on the rig-bound `Inline`. Knobs that stay scalar and add no input (`reduction = Mean`/`Sum`, `ignoreIndex`, `labelSmoothing`) are rig-usable by writing a tiny 2-input wrapper `[Module]` whose `Inline` calls `Reduced(...)` with the knobs baked; a class `weight`/`pos_weight` (an extra tensor input) is rig-usable only when **baked as a graph constant** inside such a wrapper. See the [Losses → Configurable knobs](nn-library.md#loss-configurable-knobs) section for the recipes.

<a id="loss-ignoring-targets"></a>
**A loss graph may ignore its `targets`.** The two-input shape is a *signature* requirement, not a
data-flow one: rig build checks the counts only — exactly two inputs, exactly one output — then wires
the model's output to input 0, creates a fresh runtime input for input 1, and replays the loss body.
Nothing requires input 1 to be read. So a model that computes its **own** loss can be trained with a
pass-through loss module. That is the normal shape when the loss needs more than the one predictions
tensor and one targets tensor the slot can carry — label ids, a padding mask, per-token weights, or the
`Reduced`/`PerElement` knobs: those all arrive as ordinary **model** inputs and are consumed in the
model body (see [Which knobs reach the rig](nn-library.md#loss-configurable-knobs)), and the model's
single output is the scalar loss:

```csharp
[Module]                                  // the model already returns the scalar loss;
public partial class PassThroughLoss      // its labels / mask are ordinary model inputs
{
    public static Scalar<float32> Inline(Scalar<float32> predictions, Tensor<float32> targets)
        => predictions;                   // `targets` unused — legal, and never read
}
```

**You still feed the ignored input.** `rig.TargetDef` is derived from the loss graph's second input
whether or not the body reads it, and that input survives into the compiled trainstep, so every
`TrainStep` still passes a target struct — omitting it fails the input-count check (`CR006`). Only its
**dtype** must match the type the loss declares (a mismatch is rejected by the backend); its shape and
contents are ignored, so a zero-element placeholder is enough and costs next to nothing:

```csharp
var noTargets = rig.TargetDef.FromOrderedData(TensorData(DType.Float32, [0L]));
ckpt = rig.TrainStep(ckpt, inputs, noTargets);   // the real labels ride inside `inputs`
```

Two things this shape does **not** change. The predictions tensor is never an output of the training
step — the step's outputs are the updated parameters, model state, optimizer state and the loss — so
composing the loss into the model does not save the memory of a large logit tensor: the loss body is
inlined into the same graph either way, and the composed training step does the same work (op for op)
whichever side of the seam the loss sits on. And `ExtractInferenceModel` hands back the model **as
authored**, so a loss-computing model yields an inference model that returns a loss and demands the
labels; author the prediction path as its own module if you also need one.

An optimizer module takes its `[Hyper]` hyperparameters, then exactly `(currentParam, grad)`,
and returns the updated parameter. Optimizer state never appears in the signature: it is
created inside the body by an **optimizer-owned state initializer** (e.g.
`OptimizerStateZeros.Init(currentParam.ShapeTensor())`) and updated with one
`Globals.StateUpdate(state, newValue)` call per state — see
[Custom optimizers](#custom-optimizers).

### Hyperparameter kinds (`Hyperparameter`)

Each hyperparameter property is a `Hyperparameter` — a **declared signature bound to exactly one of
three sources** (an explicit closed union with an exhaustive `Kind`, `Baked`/`Scheduled`/`Runtime`).
Its *kind* — not a separate flag — decides the wiring:

| Assign | Kind | Wiring |
|---|---|---|
| a bare value (e.g. `1e-4f`, `5`, `true`), or `Hyperparameter.Baked(v)` | `Baked` | graph `Constant`; change ⇒ rebuild |
| a `Schedule` (e.g. `Schedules.Cosine(3e-4f, total)`), or `Hyperparameter.Scheduled(schedule)` | `Scheduled` | lowered to graph math and computed **in-graph** from the counter input(s) each `TrainStep` — no host evaluation |
| `Hyperparameter.Scheduled(module)` | `Scheduled` (module) | a scheduler module (int64 counter(s) → the declared-dtype value) inlined into the graph, for schedules the built-ins don't cover |
| `Hyperparameter.Runtime()` / `Hyperparameter.Runtime(shape)` | `Runtime` | runtime input with no schedule; supply each step via `MakeHyperparameters` |

`Schedule` factories live on `Schedules` (`Constant`, `Linear`, `Cosine`, `CosineWithWarmup`,
`StepDecay`, `Exponential`, `OneCycle`) with fluent combinators on the result (`WithWarmup`, `Then`,
`Scale`, `Clamp`, `Shift`, `PerEpoch`) — each defined in
[Schedule factories and combinators](#schedule-factories-and-combinators) below.
`Schedule.At(long)` previews a schedule's value at a step.

**Two scheduler construction paths, one runtime representation.** Every schedule the rig accepts is a
graph, from exactly two sources: a built-in `Schedule`, or a scheduler **module** — a Shorokoo module
graph whose inputs are a named subset of the reserved int64 scalar counters `{step, epoch, batchIndex}`
and whose single output is the scheduled scalar at the hyperparameter's declared dtype, passed via
`Hyperparameter.Scheduled(module)`.
Built-in DSL schedules are step-only (`PerEpoch` derives its epoch in-graph from the step); a module
declares which counters it consumes by naming its inputs. Both lower to the **same** artifact — a pure
`counters → value` graph — and rig build enforces that purity (a scheduler graph carrying trainable
params, module state / `StateUpdate`, RNG draws, or an unrecognized input is rejected). There is **no**
API for an arbitrary host lambda (a compiled closure has no durable graph representation and could not
be persisted or resumed).

**Configurable milestones: `[Hyper]` + `Specialize`.** The counter-input rule covers the inputs a
scheduler graph *still has* when the rig sees it, so a module may declare its milestones as `[Hyper]`
parameters and bake them with [`Specialize`](inference.md#hardcoding-hypers-with-specialize) — which
folds each named value in as a constant and **removes it from the graph's input list** — before passing
the graph to `Hyperparameter.Scheduled`. What reaches the rig is then a counters-only graph, and it is
accepted:

```csharp
[Module]
public partial class LinearDecay
{
    public static Scalar<float32> Inline(Scalar<int64> step, [Hyper(10)] Scalar<int32> totalSteps)
        => Scalar(0.1f) * (Scalar(1f) - step.Cast<float32>() / totalSteps.Cast<float32>());
}

var sched = LinearDecay.ComputationGraph;                                     // inputs: totalSteps, step
var decay = sched.Specialize(sched.FromOrderedInputs([TensorData([], 20)]));  // inputs: step

var rig = TrainingRig.FromScratch(model, loss, SGDOptimizer.ComputationGraph, sample,
    new SGDOptimizerHyperparameters { LearningRate = Hyperparameter.Scheduled(decay) });
// learningRate = 0.1, 0.095, 0.09, … at steps 0, 1, 2, …
```

`FromOrderedInputs` pairs values with the *leading* input names, and a module's `[Hyper]` inputs come
first, so the single value here names `totalSteps`. The milestone is a graph constant from then on —
fixed for the rig's life like a `Baked` hyperparameter, so changing it means specializing again and
rebuilding — but it comes from a host value instead of being hardcoded in the module. Bake only the
non-counter inputs: specializing `step` as well is accepted too, and yields a *constant* schedule.

**One value route.** A hyperparameter's value at some counters is always obtained by *evaluating its
canonical graph at those counters*: in-graph every `TrainStep`, and — for optimizer state
initialization — via a build-time evaluation at the initial counters (all 0). There is no second,
host-materialized value that can disagree with the in-graph one. Host preview (`Schedule.At`) is served
by a single interpreter that mirrors the graph lowering.

> **Numeric note.** Because a schedule is evaluated in-graph rather than host-side, its live-training
> value carries the schedule-lowering tolerance: on engines whose `Cos`/`Pow` differ from .NET `MathF`
> (e.g. ONNX Runtime) a schedule using those ops may differ from the host `Schedule.At` value by a few
> ulps (arithmetic/piecewise schedules stay exact). This is the documented `ScheduleLowering` contract.

### Schedule factories and combinators

Write `s` for the 0-based global step counter and `f(s)` for the value of the schedule a combinator is
applied to. Every definition below is the exact arithmetic the rig evaluates (in `float32`), so
`Schedule.At(s)` and the in-graph value agree up to the numeric note above.

**Factories** (`Schedules.…`) — each returns a `Schedule` that starts at step 0:

| Factory | Value at step `s` | Outside its nominal range |
|---|---|---|
| `Constant(float value)` | `value` | unchanging at every step |
| `Linear(float baseValue, float finalValue, int totalSteps)` | `baseValue + (finalValue - baseValue) · p`, with `p = clamp(s / totalSteps, 0, 1)` | clamped, not extrapolated: `baseValue` at and below step 0, `finalValue` from step `totalSteps` on |
| `Cosine(float baseValue, int totalSteps)` | `0.5 · baseValue · (1 + cos(π · p))`, same `p` — `baseValue` at step 0, `baseValue/2` at `totalSteps/2`, `0` at `totalSteps` | clamped the same way: held at `0` from step `totalSteps` on |
| `CosineWithWarmup(float baseValue, int warmupSteps, int totalSteps)` | `Cosine(baseValue, max(1, totalSteps - warmupSteps)).WithWarmup(warmupSteps)`, with both arguments clamped (a negative `warmupSteps` becomes `0`, the cosine's length is never below `1`) — so a linear ramp over the first `warmupSteps` that, per `WithWarmup` below, starts at `baseValue / warmupSteps` (**not** `0`) and reaches `baseValue` at step `warmupSteps - 1`, then a cosine decay reaching `0` at step `totalSteps` | held at `0` afterwards |
| `StepDecay(float baseValue, int stepSize, float gamma)` | `baseValue · gamma^(s / stepSize)`, **integer** division — a staircase that drops every `stepSize` steps | never clamps; keeps decaying (or growing, for `gamma > 1`) indefinitely |
| `Exponential(float baseValue, float gamma)` | `baseValue · gamma^s` | never clamps; unbounded in both directions |
| `OneCycle(float maxValue, int totalSteps, float pctStart = 0.3f, float divFactor = 25f, float finalDivFactor = 1e4f)` | with `initial = maxValue / divFactor`, `final = initial / finalDivFactor`, `up = max(1, round(totalSteps · clamp(pctStart, 0, 1)))` and `down = max(1, totalSteps - up)`: for `s < up`, `initial + (maxValue - initial) · 0.5 · (1 - cos(π · s / up))`; for `s ≥ up`, `final + (maxValue - final) · 0.5 · (1 + cos(π · clamp((s - up) / down, 0, 1)))` — `initial` at step 0, `maxValue` at step `up`, `final` at step `totalSteps` | held at `final` from step `totalSteps` on |

`totalSteps` must be at least 1 — `Linear`, `Cosine`, `CosineWithWarmup` and `OneCycle` throw
otherwise — and so must `StepDecay`'s `stepSize`, which throws on the same rule.

**Combinators** (methods on a `Schedule`, chainable; each returns a new schedule):

| Combinator | Value at step `s` |
|---|---|
| `Scale(float factor)` | `factor · f(s)` |
| `Clamp(float min, float max)` | `clamp(f(s), min, max)`; throws if `min > max` |
| `Shift(int steps)` | `f(s + steps)` — a **positive** `steps` moves the schedule **earlier** (step 0 already sees `f(steps)`); pass a **negative** `steps` to move it later |
| `PerEpoch(int stepsPerEpoch)` | `f(s / stepsPerEpoch)`, integer division — the value is held for each block of `stepsPerEpoch` steps. The epoch index is derived from the step counter, so no epoch input is needed; `stepsPerEpoch` must be at least 1 |
| `WithWarmup(int warmupSteps, float startFactor = 0f)` | with `peak = f(0)` captured when the combinator is called: for `s < warmupSteps`, `peak · (startFactor + (1 - startFactor) · (s + 1) / warmupSteps)`; for `s ≥ warmupSteps`, `f(s - warmupSteps)` — the inner schedule is **re-based** to start after the warmup. `warmupSteps == 0` returns the schedule unchanged |
| `Then(int atStep, Schedule next)` | `f(s)` for `s < atStep`, and `next(s - atStep)` for `s ≥ atStep` — `next` is **re-based**, i.e. evaluated at the step *relative* to `atStep`, never at the absolute step |

Two consequences of `WithWarmup`'s exact form are worth spelling out. `startFactor` multiplies the
**inner schedule's step-0 value** (`peak`), not the optimizer's declared default; and because the ramp
is linear in `s + 1`, step 0 is `peak · (startFactor + (1 - startFactor) / warmupSteps)` rather than
`startFactor · peak`, the ramp first reaches `peak` at step `warmupSteps - 1`, and the inner schedule
then contributes its own step 0 at step `warmupSteps`.

**Worked example: warm up, hold, decay.** Because `Then` re-bases, the second schedule's length is
stated in its *own* steps and the boundary is stated in absolute steps — the two are independent:

```csharp
// Ramp 0 → 1e-3 over steps 0..199, hold 1e-3 to step 3899,
// then decay 1e-3 → 5e-5 over steps 3900..6000 and hold.
Schedule lr = Schedules.Constant(1e-3f)
    .WithWarmup(200)                                    // peak = Constant's step-0 value = 1e-3
    .Then(3900, Schedules.Linear(1e-3f, 5e-5f, 2100));  // Linear's own step 0 is global step 3900
```

| step | `lr.At(step)` | why |
|---|---|---|
| `0` | `5e-6` | ramp: `1e-3 · 1/200` |
| `99` | `5.0e-4` | ramp: `1e-3 · 100/200` |
| `199` | `1e-3` | ramp: `1e-3 · 200/200` — the peak |
| `200` … `3899` | `1e-3` | the `Constant` inner schedule, re-based past the warmup |
| `3900` | `1e-3` | boundary: `Linear` at *its* step 0 |
| `4950` | `5.25e-4` | `Linear` at its step 1050, halfway through 2100 |
| `6000` | `5e-5` | `Linear` at its step 2100 — the final value |
| `7000` | `5e-5` | past `totalSteps`, `Linear` holds its final value |

The decay ends at `3900 + 2100 = 6000`: `Then`'s `atStep` chooses *when* the second schedule starts,
`Linear`'s `totalSteps` chooses how long it takes. Had `next` been evaluated at the absolute step
instead, the `Linear` would already be finished at the boundary and the schedule would jump straight
to `5e-5`.

### Hyperparameter dtypes and shapes

A hyperparameter's dtype and rank are whatever the optimizer **declares** them at — the `Scalar<T>`,
`Vector<T>` or `Tensor<T>` in its `[Hyper(...)]` parameter — and that declaration is the single source
of truth end to end. Most hyperparameters are `float32` scalars (learning rate, weight decay, betas),
but any supported dtype works — an `int32` count, a `bit` (bool) flag, a `float64` coefficient — and so
does any shape, e.g. a per-element learning-rate vector.

```csharp
[Module]
public partial class MyOptimizer
{
    public static Tensor<float32> Inline(
        Tensor<float32> currentParam,
        Tensor<float32> grad,
        [Hyper(0.01f)] Scalar<float32> learningRate,
        [Hyper(2)]     Scalar<int32>   accumSteps,
        [Hyper(true)]  Scalar<bit>     nesterov,
        [Hyper(0.25)]  Scalar<float64> decay,
        [Hyper]        Vector<float32> perGroupScale) => …;
}
```

**Defaults are scalar-only.** `[Hyper(default)]` takes the host literal matching the declared dtype
(`0.01f`, `2`, `true`, `0.25`) and the generated `MyOptimizerHyperparameters` set carries each default
at that dtype. An attribute argument is a compile-time constant, so a non-scalar hyperparameter — and
a dtype with no natural C# literal, e.g. `float16` — takes no default: declare it as a bare `[Hyper]`
(the generated property is then `required`) and bind it explicitly with
`Hyperparameter.Baked(Globals.TensorData(…))`.

**Dtypes.** Host-supplied values — a baked constant, or a per-step `MakeHyperparameters` value — are
fitted to the declared dtype. Between floating-point dtypes that is always allowed: `LearningRate = 0.1`
on a `float32` hyperparameter is the familiar `0.1f`, since rounding a `double` to a `float` is what
float precision means (only an overflow to infinity is rejected). Every other conversion must be
value-preserving: `("accumSteps", 3L)` becomes an `int32` 3, while `("accumSteps", 2.5)` and
`("accumSteps", long.MaxValue)` fail loud rather than silently truncating, as does crossing the bool
boundary in either direction. A **non-scalar** value is not converted element-wise — build it at the
declared dtype (`Globals.TensorData(dtype, shape, …)`) and a mismatch fails loud.
`rig.HyperparameterDTypes` reports the declared dtypes, in the same order as `rig.HyperparameterNames`.

**Shapes.** The declaration pins the *rank* (`Scalar<T>` ⇒ 0, `Vector<T>` ⇒ 1, `Tensor<T>` ⇒ any); the
concrete *shape* comes from the binding, and the rig reports it as `rig.HyperparameterShapes`:

| Kind | Where its shape comes from |
|---|---|
| `Baked` | the constant's own shape — `Hyperparameter.Baked(TensorData([4L], …))` |
| `Scheduled` (module) | the scheduler module's output shape, inferred at rig build |
| `Runtime` | declared by you: `Hyperparameter.Runtime(4L)`; `Runtime()` means a scalar |

A runtime hyperparameter states its shape because the training step is compiled once, so the shape has
to be known at build even though the values are not. That also makes the shape fixed for the rig's
life: a per-step value whose shape differs fails loud rather than silently reshaping.

Built-in `Schedule` math (cosine / linear / decay) is inherently continuous and scalar, so a built-in
schedule drives `float32` **scalar** hyperparameters only; drive any other dtype or shape with a
scheduler **module** producing it. Baked and runtime hyperparameters have no such restriction.

```csharp
var rig = TrainingRig.FromScratch(model, loss, MyOptimizer.ComputationGraph, sample,
    new MyOptimizerHyperparameters
    {
        LearningRate  = Schedules.Cosine(1e-3f, totalSteps),        // float32 scalar, built-in schedule
        AccumSteps    = 4,                                          // int32, baked
        Nesterov      = true,                                       // bool, baked
        PerGroupScale = Hyperparameter.Runtime(3L),                 // float32 vector, host-supplied
    });

rig.TrainStep(ckpt,
    rig.MakeHyperparameters(("perGroupScale", TensorData([3L], 1f, 2f, 3f))),
    inputs, targets);
```

> **Migration (breaking).** The positional-hyperparameter `FromScratch` overloads no longer take a
> `params` array behind the optional `rngConfig` / `mergeContext` / `runtimeContext`. Pass the values
> as an explicit array in the hyperparameter slot and let the optional arguments follow:
> `FromScratch(model, loss, opt, sample, rng, null, null, 0.05f)` →
> `FromScratch(model, loss, opt, sample, [0.05f], rng)`, and a hyperparameter-less
> `FromScratch(model, loss, opt, sample, rng)` (the old empty `params` expansion) →
> `FromScratch(model, loss, opt, sample, [], rng)`. The bare
> `FromScratch(model, loss, opt, sample, 0.05f)` params form is unchanged.

> **Migration (breaking).** `Hyperparameter.BakedValue` is now the `TensorData` the constant was built
> from — carrying its shape as well as its dtype (with `BakedDType` alongside) — not a `float`; `MakeHyperparameters`'s named overload takes
> `(string name, object value)` pairs rather than `(string, float)` — existing call sites such as
> `MakeHyperparameters(("learningRate", 0.1f))` are unaffected. `HyperAttribute.DefaultValue` is
> `object?` (the host literal the constructor took) rather than `float`, and a graph input's
> `HyperDefaultValue` is the default's invariant literal (`string?`) rather than a `float?`, so an
> `int64` / `float64` / `bool` default survives the graph round-trip exactly. In a training `.skpt`, the
> rig block's `bakedHypers` map is gone: each baked binding now records its own `dtype`, `shape` and
> base64 `value`, and a runtime binding records its `shape`, so `rigVersion` stays `1` and older-shaped
> files (none exist in the wild) are not read.

> **Migration (breaking).** `HyperValue` is renamed **`Hyperparameter`** and is now an explicit
> `Baked`/`Scheduled`/`Runtime` union. `HyperValue.Constant(v)` → `Hyperparameter.Baked(v)` (a bare
> `float` still converts implicitly); `HyperValue.Runtime(seed)` → **`Hyperparameter.Runtime()`** (the
> seed is gone — the shape placeholder is internal); the undocumented `InitialValue` is removed. The
> public per-step hyperparameter entry point is renamed `MakeHyperparams` → `MakeHyperparameters`; the
> low-level struct-def / index plumbing behind it (`HyperparameterStructDef`,
> `DynamicHyperparameterIndices`) is now `internal` build machinery — inspect the dynamic hyperparameter
> names via `DynamicHyperparameterNames`. Fresh-checkpoint creation can now **fail loud** (see
> `CreateInitialCheckpoint` below).

## `TrainingRig` API

```csharp
public static TrainingRig FromScratch(
    ComputationGraph modelGraph,      // GraphKind.Module, or a ToConcreteArchitecture result
    ComputationGraph lossGraph,       // kind must be GraphKind.Module
    ComputationGraph optimizerGraph,  // kind must be GraphKind.Module
    NamedModelParam[] sampleInputs,            // names + sample shapes for model inputs
    IOptimizerHyperparameters hyperparameters, // named set, e.g. new AdamWOptimizerHyperparameters { ... }
    RngConfig? rngConfig = null,              // seeds the run — see "Seeding the run" below
    ComputeContext? mergeContext = null,      // build/merge-phase context (rig.MergeContext); null ⇒ Default
    ComputeContext? runtimeContext = null);   // compile/run context (rig.RuntimeContext); null ⇒ Default

// Lower-level: positional values (a float bakes a constant, a Schedule schedules it). The params form
// takes the values and nothing else; supplying an rngConfig or either context selects the array form,
// which takes them in the same order and the same slots as the named-set overload above:
//   FromScratch(model, loss, opt, sampleInputs, params Hyperparameter[] hyperparameters)
//   FromScratch(model, loss, opt, sampleInputs, Hyperparameter[] hyperparameters,
//               RngConfig? rngConfig = null,
//               ComputeContext? mergeContext = null, ComputeContext? runtimeContext = null)
// Each of the three forms above also has a twin taking a ModelParamList (model.FromOrderedInputs([…]))
// for sampleInputs.

// Fresh initial checkpoint. Optimizer state is initialized at each hyperparameter's value at the
// initial counters. Fails loud if the optimizer's state initializer reads a Runtime hyper (its value
// is unknown at build) — supply explicit values with the overload below.
public TrainingCheckpoint CreateInitialCheckpoint();
public TrainingCheckpoint CreateInitialCheckpoint(TensorDataStruct hyperparameters); // from MakeHyperparameters(...)

// Schedule-driven: scheduled hyperparameters are computed in-graph from the checkpoint's
// step (fed as the step counter), then the step advances. Requires no schedule-less runtime hypers.
// Returns the post-step checkpoint directly, with its .Loss set to this step's loss. The rig
// compiles its training-step graph once internally (lazily, cached), so a manual loop is just
// `cp = rig.TrainStep(cp, in, out);` — no caller-side ComputeContext.Compile.
public TrainingCheckpoint TrainStep(
    TrainingCheckpoint checkpoint,
    TensorDataStruct trainingInput,
    TensorDataStruct trainingOutput);

// Explicit override: supply the schedule-less runtime hyperparameter values for this step.
public TrainingCheckpoint TrainStep(
    TrainingCheckpoint checkpoint,
    TensorDataStruct hyperparams,              // from MakeHyperparameters(...)
    TensorDataStruct trainingInput,
    TensorDataStruct trainingOutput);

// Loader-driven single step: draws loader.Next(), sourcing epoch / batch from the loader — the
// single-step form of Fit(loader). The batch's own position drives the scheduler for this step and
// is recorded on the returned checkpoint (the batch USED). Requires no runtime hypers.
public TrainingCheckpoint TrainStep(
    TrainingCheckpoint checkpoint,
    IDataLoader loader);

// Explicit epoch / batch: for a host driving its own iteration (no loader). epoch / batchNumber name
// the batch being trained — fed to the scheduler for this step AND recorded verbatim on the returned
// checkpoint: the same "batch used" convention the loader overload records.
public TrainingCheckpoint TrainStep(
    TrainingCheckpoint checkpoint,
    TensorDataStruct trainingInput,
    TensorDataStruct trainingOutput,
    long epoch,
    long batchNumber);

public TensorDataStruct MakeHyperparameters(float value);                       // exactly one dynamic
//   also: (double), (int), (long), (bool), and (TensorData) for other dtypes / non-scalar shapes
public TensorDataStruct MakeHyperparameters(params (string name, object value)[] values); // named

// Array-driven: one array element per training step (typically a pre-batched batch). The checkpoint
// comes LAST and is optional, so the minimal call is `rig.Fit(inputs, targets, numEpochs: 10)`.
public TrainingResult Fit(
    TensorDataStruct[] trainingInputs,
    TensorDataStruct[] trainingOutputs,
    int numEpochs,
    TrainingCheckpoint? initialCheckpoint = null); // defaults to CreateInitialCheckpoint()
                                                   // compiles/runs via rig.RuntimeContext (one graph per rig)

// Data-loader-driven: the loader owns the batch stream; Fit advances step / epoch / batch for you.
public TrainingResult Fit(
    IDataLoader loader,
    int numEpochs,
    TrainingCheckpoint? initialCheckpoint = null); // defaults to CreateInitialCheckpoint()

// The same array loop, with the checkpoint FIRST and required. `Train` is not an alias for `Fit`:
// the argument orders differ, so the two calls are not interchangeable. The array `Fit` above is
// exactly this call with the checkpoint defaulted.
public TrainingResult Train(
    TrainingCheckpoint initialCheckpoint,
    TensorDataStruct[] trainingInputs,
    TensorDataStruct[] trainingOutputs,
    int numEpochs);
```

### What construction costs

`FromScratch` does real work before any training happens, and a checkpoint/resume workflow
re-pays most of it on every process start. A training `.skpt` carries the constituents and the
state, not the derived build products, so `TrainingRig.Load` rebuilds those — it reads the saved
concrete architecture rather than re-concretizing, but everything after that is redone, including
running every initializer whose values the checkpoint then overwrites.

The build phase, all of it on `MergeContext`, is concretization, composition with the loss,
autograd, optimizer lowering, shape inference and graph optimization, plus two costs that scale
with your parameter count: each trainable parameter's initializer is run, and each optimizer-state
initializer is run per trainable parameter. Both run one backend session per parameter, so they
grow linearly with the number of trainable parameters rather than superlinearly, and both copy
each value onto storage of its own rather than leaving it holding that session's working memory.
Peak host memory during initialization still grows with the model, but far more slowly than it
once did — a few hundred bytes per parameter element rather than a few kilobytes.

Then, on the first `TrainStep`, the rig compiles its training-step graph once and caches it (see
`TrainStep` above) — one fixed cost per rig, independent of how many steps follow.

Neither phase is proportional to your dataset, and neither recurs during the loop: steady-state
`TrainStep` pays neither. If you are timing a run, expect the first step to be markedly slower
than the rest — that is the compile, not a slow optimizer.

### Compute contexts: `MergeContext` and `RuntimeContext`

A rig carries two `ComputeContext` members, both supplied at construction (defaulting to
`ComputeContext.Default`) and both **runtime configuration that is never written to a checkpoint** —
a reloaded run gets fresh contexts by passing them to `FromScratch`, or to
`TrainingRig.Load(path, mergeContext, runtimeContext)` when the rig is rebuilt from a `.skpt`
alone. `MergeContext` runs the
build/merge phase (concretization, shape inference, graph lowering and memory optimization, optimizer
state init); `RuntimeContext` compiles the training-step graph into its executable session and runs it,
so it is the context whose session actually executes training. It is the sole compile/run context for
`TrainStep`, `Train` and `Fit` — none of them takes a per-call context override, so a rig has exactly
one compiled training-step graph that the `Fit`/`Train` loop and a manual `TrainStep` loop all share.
Every `With…` derivation keeps the same two contexts.

**What the two can usefully differ in: nothing, today.** `ComputeContext` has a single parameterless
constructor and carries no per-instance settings — no device, no execution provider, no thread count,
no session options — and every session either context creates is built by the one process-wide backend
factory. Passing two distinct instances therefore selects nothing. In particular you **cannot** merge
on one device and train on another: [only one backend is live per process](inference.md#backend-selection)
and both contexts go through it, so the naming does not offer a CPU-build / GPU-train split. Read the
two members as a division of *phases* — which work is build/merge and which is compile/run — not of
hardware; they would only become a lever if `ComputeContext` gained per-instance configuration.
Leaving both `null`, so each defaults to `ComputeContext.Default`, is the normal choice.

Result types:
- `TrainingCheckpoint` → `.TrainableParams`, `.ModelState`, `.OptimizerState`, `.Step` (global step, `long`; advances each `TrainStep`, so schedules resume from a saved checkpoint), and the host-owned run counters `.Epoch` / `.BatchIndex` (`long?`; the training loop advances them — the counter-agnostic `TrainStep` carries them through unchanged). They are `null` when the position is genuinely **unknown** — an initial checkpoint, or one trained without a data loader / explicit counters — rather than a misleading `0`; the loader-driven and explicit-counter paths set concrete values. A scheduled hyperparameter reading the epoch / batch counter sees `0` for a `null` value. `.Step` is always a concrete `long`; all counters are `int64` end to end. It also carries `.Rig` (the `TrainingRig?` that produced it — set on every rig-produced checkpoint, so `checkpoint.ToInferenceModel()` needs no re-supplied graph) and `.Loss` (`float?`; the loss of the `TrainStep` that produced it, `null` on an initial or bare checkpoint). Both are preserved through the counter derivations (`WithCounters`/`WithStep`/`WithEpoch`/`WithBatchIndex`). `TrainStep` returns this checkpoint directly — read the step's loss off `.Loss`. `.Loss` persists as its own `Loss` component, independent of `Counters` (dropping `Loss`, or an initial checkpoint, reloads with `.Loss == null` — never a sentinel `0`).
- `TrainingResult` → `.FinalCheckpoint`, `.EpochLosses` (the per-epoch mean losses).

`TrainingRig`, `TrainingCheckpoint`, and `TrainingResult` are in
namespace `Shorokoo` (covered by `using Shorokoo;`).

### Seeding the run

`rngConfig` binds the run's [RNG configuration](rng-configuration.md): one master
seed keys parameter initialization and every runtime draw (Dropout masks, in-model
sampling). Omitted (or `null`), the rig keys under the **default identity** (master
seed 0) — training is deterministic and reproducible by default. Dropout masks still
vary per training step (the per-step RNG position is saved in the checkpoint, so a
resumed run continues exactly). Pass
`new RngConfig { MasterSeed = … }` to re-roll all streams coherently, or
`RngConfig.NonDeterministic()` for per-run variation.

## Feeding data: the data loader

The array overloads of `Fit`/`Train` take pre-batched `TensorDataStruct[]` and leave the
checkpoint's epoch / batch counters for you to set. A **data loader** instead owns the batch
stream: it chops your data into batches, tracks its position, and lets `Fit` advance the
checkpoint's step / epoch / batch counters automatically — so a saved checkpoint records exactly
where the run was, and a resumed run continues from the very next batch.

```csharp
// One value per field of the definition, in declaration order; the leading dimension is
// the sample count.
var inputs  = rig.InputDef.FromOrderedData(TensorData([1000L, 64L], features));
var targets = rig.TargetDef.FromOrderedData(TensorData([1000L, 10L], labels));

// Batch into 32s, reshuffling each epoch (deterministically from the seed).
var loader = new InMemoryDataLoader(inputs, targets, batchSize: 32, shuffle: true, seed: 42);

var outcome = rig.Fit(loader, numEpochs: 10);   // step / epoch / batch advance automatically
```

`FromOrderedData` fills the field names in from the definition itself, which is why it is the
form to reach for: the target field is named after the loss module's **second `Inline`
parameter**, so spelling `"targets"` by hand couples your driver code to that module's
implementation and breaks the moment a loss names its parameter `labels`. It pairs values
positionally and throws when their count does not match the field count — so for a many-field
struct whose fields share a shape, the explicit `new TensorDataStruct(def, fields)` form
stays the safer one, since it catches a swapped pair that `FromOrderedData` accepts.

- **`IDataLoader`** is the minimal contract: a current `Position`
  (`DataLoaderPosition`, the epoch + index of the *next* batch it will yield), `Next()` (produces
  the current `DataBatch` — input + target + the position it came from — and advances one batch,
  rolling into the next epoch after the last), and two resume primitives — `RestoreFrom(position)`
  (the next `Next()` yields the batch *at* `position`) and `RestoreAfter(position)` (the next `Next()`
  yields the batch *one step after* `position`, rolling into the next epoch internally).
  `InMemoryDataLoader` also exposes `BatchesPerEpoch`, but that is **not** on the interface — the epoch
  rollover a caller would have used it for now lives inside `RestoreAfter`.
- **One step at a time.** `rig.TrainStep(checkpoint, loader)` is the single-step form of
  `Fit(loader)` — it draws one batch, runs the step (the batch's own position drives any scheduler),
  and returns a checkpoint recording the **batch used** (that same drawn position). `Fit(loader)` is
  just a loop over it, so the two share one source of the loader step-and-counter semantics. For a host
  that owns its own iteration (no loader), `rig.TrainStep(checkpoint, input, target, epoch, batchNumber)`
  records the given `epoch` / `batchNumber` verbatim — the same "batch used" convention (it
  names the batch being trained).
- **`InMemoryDataLoader`** is the bare-minimum implementation over tensors you already hold. Each
  field's leading dimension is the sample count `N`; it slices along that dimension into
  fixed-size batches, optionally reshuffling every epoch.
- **Shuffle is deterministic.** With `shuffle: true`, the permutation for epoch `e` is a pure
  function of `(seed, e)` — a Fisher–Yates shuffle over a SplitMix64 stream, using no ambient
  `Random` and no wall clock. That is what makes resume exact: restoring to `(e, b)` regenerates
  epoch `e`'s order bit-for-bit and skips the first `b` batches, so the continued run sees the
  same batches the original would have.
- **Partial final batch.** `dropLast: true` (the default) drops a trailing partial batch so every
  batch matches the shape the training-step graph was compiled for. Pass `dropLast: false` to keep
  the smaller final batch (only safe if the graph tolerates a variable batch dimension).
- **Resume.** A checkpoint's `.Epoch` / `.BatchIndex` name the batch that was **used** at its last
  step. Save the `FinalCheckpoint` (or any mid-run checkpoint), then in a later process rebuild the rig
  and a loader over the same data/seed and call `rig.Fit(loader, numEpochs, initialCheckpoint: loaded)`:
  `Fit` advances the loader one batch past that recorded position (`RestoreAfter`), so the run picks up
  at exactly the next batch — no re-run and no skip. (A fresh, position-unknown checkpoint instead
  starts at `(0, 0)` via `RestoreFrom`.) `numEpochs` is counted from the loader's resume epoch (a
  checkpoint saved mid-epoch first finishes that partial epoch; one saved at an epoch's last batch
  begins the next). This is Shorokoo owning **its own** loader's position; a host driving an external
  pipeline Shorokoo doesn't own still uses the checkpoint's host user-data bag instead.

## Save and resume a checkpoint (across process restarts)

A `TrainingCheckpoint` holds the full training state — trainable params, model
state, optimizer state, and the host-owned run counters (global step, epoch, batch
index). Save one to disk and resume from it in a later run:

```csharp
// Save mid-training (e.g. every N steps, or at the end of an epoch):
checkpoint.Save("run.safetensors");

// Later — in a fresh process — rebuild the SAME rig, then load:
var rig  = TrainingRig.FromScratch(MyModel.ComputationGraph, L2Loss.ComputationGraph,
                                   AdamOptimizer.ComputationGraph, sampleInputs,
                                   new AdamOptimizerHyperparameters { ... });
var ckpt = rig.LoadCheckpoint("run.safetensors");   // params + optimizer moments + step restored
var more = rig.Fit(inputs, targets, numEpochs: 5, ckpt);  // continues where it left off
```

- The file is a single SafeTensors file (every param/state field plus the run
  counters). The `int64` marker carries `[version, step]` (always present); epoch and batch index
  are each a **presence-gated** `int64` scalar beside it, written only when set — so an unknown
  epoch/batch (a checkpoint trained without a loader / explicit counters) is absent on disk and
  reloads as `null`, never a sentinel `0`. A concrete
  `0` (e.g. a run resting at the start of an epoch) is written and reloads as `0`.
- **The save is atomic**, so overwriting one path every N steps is safe. `checkpoint.Save`
  (and `Persistence.SaveTrainingCheckpoint`, which delegates to it) stages the file under a
  `.tmp-` sibling name in the target's directory, flushes it to disk, then commits it with a
  single rename: a process killed mid-save — an OOM kill, a `Ctrl-C`, a power loss — leaves
  either the previous checkpoint or the new one at that path, never a truncated file — you
  need no stage-and-rename of your own. Two consequences: the target's **directory must
  already exist** (a missing one throws, it is not created), and an interrupted save can
  leave a `.tmp-`-prefixed sibling behind, which the next successful save of the same target
  sweeps. The `.skpt` saves carry the same guarantee — see
  [skpt-checkpoints.md](skpt-checkpoints.md#the-directory-form) for the one window the
  directory form adds when it *replaces* an existing checkpoint.
- For the **native `.skpt` container** instead — the training state with every tensor
  addressed individually through the manifest's `tensorMappings` (the trainable weights and
  model state ride in the concrete inference model's own mapping, so their bytes live once;
  the optimizer state gets a mapping of its own), the bytes themselves in per-kind `data/`
  entries beside the model, with the container's inspectable manifest, per-entry Zstd, and
  provenance metadata — save with
  `Persistence.SaveTrainingCheckpointToSkpt(checkpoint, "run.skpt")` — the checkpoint's
  `.Rig` supplies the self-describing inference model, so no model graph or example input
  is needed (or use the `Persistence.ForTrainingCheckpoint(...)` builder) — and resume with
  `rig.LoadCheckpointFromSkpt("run.skpt")` — or, with no model/loss/optimizer graphs in hand,
  with the static `var (rig, ckpt) = TrainingRig.Load("run.skpt")`, which rebuilds the rig from
  the constituents the file carries and hands it back alongside the resumed checkpoint, so the
  rig need not be rebuilt by you at all. Each on-disk format has its own load entry point:
  `rig.LoadCheckpoint` reads the flat safetensors file only, `rig.LoadCheckpointFromSkpt` and
  `TrainingRig.Load` the `.skpt` container only, and handing any of them the other format fails
  immediately with an error naming the right entry point (nothing sniffs the file's bytes to
  pick a path; to identify an unknown file, use `Persistence.Inspect`).
  See [skpt-checkpoints.md](skpt-checkpoints.md#training-checkpoints).
- `LoadCheckpoint` / `LoadCheckpointFromSkpt` reconstruct the checkpoint against the rig's own
  parameter and state definitions, so the rig must be built from the **same**
  model/loss/optimizer graphs. Loading a checkpoint from a different model or
  optimizer throws.
- Because `.Step` is restored, learning-rate **schedules resume from the right
  step** — not from step 0.
- `rig.LoadCheckpoint(path)` delegates to `TrainingCheckpoint.Load(path, rig)` (and
  `rig.LoadCheckpointFromSkpt(path)` to `TrainingCheckpoint.LoadFromSkpt(path, rig)`), which
  resolves the struct defs from the rig and sets `.Rig` on the result. The lower-level
  `Persistence.LoadTrainingCheckpoint(path, trainableDef, modelStateDef, optimizerStateDef)`
  (flat) / `Persistence.LoadTrainingCheckpointFromSkpt(...)` (`.skpt`) are the def-based forms
  if you hold the struct defs without a rig (their results carry no rig).
- Both save and load take an optional `CheckpointComponents` flags value —
  `InferenceState` (trainable params + model state), `OptimizerState`, `Counters`, `Loss`, and
  `TrainingRig` — combined with `|`. On save, `null` writes every available component; on
  load, `null` reads everything present (a component absent from the file is filled from the
  rig's initial values). `checkpoint.Save(path, CheckpointComponents.InferenceState)` writes
  weights only. `Loss` is its own component, independent of `Counters`; explicitly requesting
  `Loss` on a checkpoint whose loss is `null` is a no-op (it writes nothing and does not throw —
  a null loss is a legitimate value). The `TrainingRig` component — the rig's own constituent
  model/loss/optimizer/scheduler graphs, its hyperparameter bindings and RNG config, enough to
  rebuild the whole rig from the file alone — is **never named explicitly**: every native `.skpt`
  carries it (`Persistence.SaveTrainingCheckpointToSkpt` always writes it) and the static
  `TrainingRig.Load(path)` is the entry point that uses it, returning the rebuilt rig and its
  resumed checkpoint. Requesting the flag (including via `CheckpointComponents.All`, which
  contains it) throws on the paths that cannot honor it: the flat
  safetensors `checkpoint.Save` (that format cannot carry constituent graphs — save to a `.skpt`
  instead), and a rig-supplied load (`rig.LoadCheckpoint` / `rig.LoadCheckpointFromSkpt`), which
  never rebuilds a rig because you already passed one — omit the flag there, or pass `null` to
  load every state component the file contains.
- `rig.AdoptCheckpoint(checkpoint)` returns a new checkpoint identical to the argument but
  bound to that rig (validating the field defs match), so a bare checkpoint — or one loaded
  against a different rig instance — gains a rig for `ToInferenceModel()`.
- To see what a checkpoint file holds (the run counters — step, epoch, batch index —
  and the per-section tensor listing) without loading it — or to identify an unknown
  file — use `Persistence.Inspect(path)`;
  see [onnx-and-weights.md](onnx-and-weights.md#identify-and-summarize-a-file-persistenceinspect).

### Bind trained weights into an inference model

Once trained, turn a checkpoint into a runnable concrete model with one call:

```csharp
var concrete = result.FinalCheckpoint.ToInferenceModel();   // no graph to re-supply
var output   = ComputeContext.Default.Execute(concrete, myInput);
```

`ToInferenceModel()` binds this checkpoint's trainable params and model state, by canonical
identity, into the checkpoint's `.Rig`'s **retained concrete architecture** — the model the rig
concretized once at build time (at **all** its inputs, so multi-input models are supported) and
holds for reuse. No re-concretization and no sample inputs are involved. It requires an attached
rig — every rig-produced checkpoint has one; attach one to a bare checkpoint with
`rig.AdoptCheckpoint(checkpoint)` first.

## Types used by the training API

All of these are in namespace `Shorokoo` (covered by `using Shorokoo;`):

| Type | Role | How to make one |
|---|---|---|
| `NamedModelParam` (abstract) | A named parameter value. | Use the concrete `TensorDataModelParam`. |
| `TensorDataModelParam` | Concrete `NamedModelParam` wrapping one `TensorData`. | `new TensorDataModelParam(name, ModelParamType.InputParam, tensorData)` |
| `ModelParamType` (enum) | Tags a param's role. | `Undefined`, `HyperParam`, `TrainableParam`, `InputParam`, `OutputParam` |
| `ModelParamList` | A set of named params (e.g. loaded weights). | `new ModelParamList(IEnumerable<(string name, TensorData data)>)` |
| `TensorDataStruct` | A struct-shaped bundle of named `TensorData` fields; the form `Train`/`TrainStep` expect for inputs/targets. | Build: `new TensorDataStruct(structDef, fields)` where `structDef` is a `TensorStructDef` and `fields` are `KeyValuePair<string, IData>`. Read: `.Fields` (an `ImmutableDictionary<string, IData>` of name → value), `.Count`, or the `[int]` indexer. |

`sampleInputs` for `FromScratch` is a `NamedModelParam[]` describing each model input
by name and sample shape. `Train`/`TrainStep` take `TensorDataStruct` batches.

## Workflow: train a model

1. Define model, loss, and optimizer as `[Module]` classes (or reuse built-ins).
2. Build the rig with the optimizer's named hyperparameter set (a bare `float` bakes a constant;
   a `Schedule` makes it live):
   ```csharp
   var rig = TrainingRig.FromScratch(
       MyModel.ComputationGraph,
       L2Loss.ComputationGraph,
       SGDMomentumOptimizer.ComputationGraph,
       new NamedModelParam[] {
           new TensorDataModelParam("input", ModelParamType.InputParam,
                                    TensorData([4L, 64L], new float[256])) },
       new SGDMomentumOptimizerHyperparameters {
           LearningRate  = Schedules.CosineWithWarmup(0.5f, warmupSteps: 100, totalSteps: 1000),
           MomentumCoeff = 0.9f,          // baked constant
       });
   ```
3. Initialize parameters: `var ckpt = rig.CreateInitialCheckpoint();`.
4. Run epochs: `var outcome = rig.Fit(inputs, targets, numEpochs: 10);`
   The learning-rate schedule is applied automatically as the global step advances. (Or call
   `rig.TrainStep(...)` per batch; pass `rig.MakeHyperparameters(...)` to override a step explicitly.)
5. Read `outcome.EpochLosses` for the loss curve and
   `outcome.FinalCheckpoint.TrainableParams` for trained weights. `TrainableParams` is a
   `TensorDataStruct`; read its values via `.Fields` (name → `IData`, each a `TensorData`), e.g.:
   ```csharp
   foreach (var (name, value) in outcome.FinalCheckpoint.TrainableParams.Fields)
   {
       var data = (TensorData)value;   // shape via data.Shape.Dims; values via data.As<float32>().AccessMemory()
   }
   ```

## Custom optimizers

A custom optimizer is just a `[Module]` whose `Inline` lists its `[Hyper]` scalars first, then
exactly `(currentParam, grad)`, and returns the updated parameter. Each piece of optimizer
state is created **inside the body** by an optimizer-owned `[StateInitializer]`'s `Init` call —
typically `OptimizerStateZeros.Init(currentParam.ShapeTensor())` from
`Shorokoo.Modules.Optimizers`, which zero-fills at the parameter's shape — and updated with
exactly one `Globals.StateUpdate(state, newValue)` call. Everything else is derived
automatically. For example, a momentum-less RMSprop (the full version ships as
`RMSpropOptimizer` in [Shorokoo.Modules](nn-library.md)):

```csharp
[Module]
public partial class SimpleRMSprop
{
    public static Tensor<float32> Inline(
        Tensor<float32> currentParam,
        Tensor<float32> grad,
        [Hyper(0.001f)] Scalar<float32> learningRate,
        [Hyper(0.99f)]  Scalar<float32> alpha,
        [Hyper(1e-8f)]  Scalar<float32> epsilon)
    {
        var meanSquare = OptimizerStateZeros.Init(currentParam.ShapeTensor()); // one state field per param

        var one = Scalar(1.0f);
        var newMeanSquare = alpha * meanSquare + (one - alpha) * grad * grad;
        Globals.StateUpdate(meanSquare, newMeanSquare);
        return currentParam - learningRate * grad / (newMeanSquare.Sqrt() + epsilon);
    }
}
```

This automatically yields a generated
`SimpleRMSpropHyperparameters { LearningRate = 0.001, Alpha = 0.99, Epsilon = 1e-8 }` with full
schedule support, plus a `meanSquare` state field per trainable parameter — initialized by
running `OptimizerStateZeros` at that parameter's shape — threaded for you:

```csharp
var rig = TrainingRig.FromScratch(model, loss, SimpleRMSprop.ComputationGraph, sample,
    new SimpleRMSpropHyperparameters { LearningRate = Schedules.Cosine(1e-3f, totalSteps) });
```

A custom initial value is just a custom initializer (any `Inline` works; the rig runs it with
the inputs you wired in the body — here the parameter's shape):

```csharp
[StateInitializer(Ownership = StateOwnership.OptimizerOwned)]
public static partial class OptimizerStateOnes
{
    public static Tensor<float32> Inline(Vector<int64> shape) => Globals.TensorFill(shape, 1.0f);
}
```

For state that is logically a single value per parameter — a step counter, a scalar EMA —
use `OptimizerScalarZeros.Init()` (seeded at 0), `OptimizerScalarOnes.Init()` (seeded at the
multiplicative identity 1, for a running product like NAdam's `∏μ_i`), or your own rank-0
initializer. It stores a true scalar that broadcasts against the param-shaped tensors, so it
costs one float per parameter instead of a full copy; Adam's bias-correction timestep works
this way.

Constraints:

- **State must come from an optimizer-owned state initializer.** Declaring state as an
  `Inline` parameter throws at rig-build time, and `Globals.StateUpdate` itself throws
  `InvalidStateUpdateException` if its first argument is not a state variable. Module-owned
  initializers (e.g. BatchNorm's running-stat initializers) are rejected inside optimizer
  graphs, and optimizer-owned ones are rejected inside model graphs.
- **Each state is updated exactly once per step** — combine conditional updates into one
  value (e.g. with `IfElse`) and register it with a single `StateUpdate` call.
- **Hyperparameters must be tensor-shaped** — `Scalar<T>`, `Vector<T>` or `Tensor<T>`, at any supported
  dtype (`float32`, `int32`, `bit`, …); the rig bakes/feeds them at their declared dtype and shape, and a
  set is generated even when the dtypes and shapes are mixed. An `OptionalTensor`, sequence or struct
  hyperparameter yields no generated set. Only a scalar can carry a `[Hyper(default)]` default.
- **Order + `[Hyper]` matter** — hyperparameters must be the leading inputs, and `[Hyper]` is what
  makes the named set generate. Without it the optimizer still works via the positional
  `params Hyperparameter[]` overload, but you lose the named, compile-checked set.
- For non-generated cases you can hand-implement `IOptimizerHyperparameters` yourself.

## Notes / known limitations

- `AdamWOptimizer` omits bias correction (no timestep tracking); early-step behavior
  differs slightly from reference AdamW. Effect is minor after the first few steps.
  `AdamOptimizer` *does* bias-correct (it carries the timestep as a scalar state field —
  one float per parameter, not a param-shaped buffer).
- `LionOptimizer` **swaps the beta roles** versus Adam: the stored momentum `m` is decayed by
  **β2** (`m = β2·m + (1−β2)·g`), while **β1** only appears in the sign blend that forms the
  update direction. The default `(β1 0.9, β2 0.99)` looks Adam-like but means something
  different. Lion's good `lr` is ~3–10× smaller than AdamW's and its `wd` ~3–10× larger
  (default `wd 0`).
- `AdafactorOptimizer` ships the **non-factored** variant: it keeps Adafactor's update dynamics
  (relative step `min(lr, 1/√t)`, parameter scaling, RMS update clipping, increasing decay
  `1 − t^τ`) but **not** its row/column factoring — so its second moment is a full param-shaped
  buffer, the **same memory as Adam**, not the sublinear `R + C` footprint. The factoring is not
  expressible in Shorokoo's single rank-agnostic per-parameter optimizer graph (the state's
  shape would have to depend on each parameter's rank — see the optimizer notes in
  [nn-library.md](nn-library.md)). A user reaching for Adafactor
  specifically to save memory gets Adam-sized state; `learningRate` is the **cap** on the
  relative step, not a fixed lr.
- Prefer the optimizer's generated named set (`<Optimizer>Hyperparameters`); it has the right
  names/defaults and is checked at compile time. The positional `params Hyperparameter[]` overload must
  still match the optimizer's hyperparameter count exactly: SGD=1, SGDMomentum=2, Adam=4,
  RMSprop=4, AdamW=5, Adagrad=2, Adamax=4, NAdam=5, RAdam=4, Adadelta=3, Lion=4, Adafactor=6,
  Lamb=5.
- Optimizer state has one or more fields per trainable parameter (momentum: velocity;
  AdamW: `m`/`v`; Adam: `m`/`v` plus a scalar `step`; RMSprop: `squareAvg`/`momentumBuffer`;
  Adagrad: `accumulator`; Adamax: `m`/`u` plus a scalar `step`; NAdam: `m`/`v` plus two
  scalars — `step` and `muProduct`; RAdam: `m`/`v` plus a scalar `step`; Adadelta:
  `squareAvg`/`accDelta`; Lion: `m` only — half of Adam/AdamW; Lamb: `m`/`v` plus a
  scalar `step` — Adam's footprint, the per-tensor trust ratio being recomputed each
  step and stored nowhere; Adafactor: a **full param-shaped** `v` plus a scalar
  `step` — same footprint as Adam, because the
  sublinear-memory row/column factoring is **not** implemented, see above) — see the table in
  [nn-library.md](nn-library.md). Each field is
  initialized by running its state initializer: `OptimizerStateZeros` zero-fills at the
  parameter's shape, `OptimizerScalarZeros` produces a rank-0 scalar seeded at 0 (e.g. Adam's
  `step`, one float per parameter rather than a param-shaped buffer), and `OptimizerScalarOnes`
  a rank-0 scalar seeded at 1 (e.g. NAdam's running momentum product, which needs the
  multiplicative identity).

## Anti-patterns

- Do not mismatch the positional `params Hyperparameter[]` overload with the optimizer's hyperparameter
  count; prefer the named set so this can't happen.
- Do not call the schedule-driven `TrainStep` on a rig whose dynamic hyperparameter is
  `Hyperparameter.Runtime` (schedule-less); supply it via `MakeHyperparameters` and the override overload.
- Do not implement backward passes manually; rely on autodiff.
- Do not mutate `TrainingCheckpoint` in place across steps; thread the returned
  checkpoint forward.
- Do not declare optimizer state as `Inline` parameters — state is created inside the body
  via an optimizer-owned `[StateInitializer]`'s `Init` and registered with `StateUpdate`.
- Do not call `Globals.StateUpdate` on inputs, trainable parameters, or computed tensors;
  only state variables (a `[StateInitializer]` `Init` result) are accepted.
- Do not call `Globals.StateUpdate` outside a module body — it throws. Inside a
  `LoopAPI.Iterate` body the call is allowed: it registers the post-loop value of the
  updated tensor — the value it holds once the loop finishes (an in-loop call is that
  state's one update for the step). This requires the updated value to be a carried
  loop variable — assigned in the body and read across iterations, so its final value
  surfaces as a loop output (with zero iterations it falls back to its pre-loop value).
  A value that never
  leaves the loop, a scanned result, or an iteration-scoped value (e.g. the iteration
  index) has no well-defined post-loop value and fails the module build.
