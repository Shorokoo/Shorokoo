# Optimizers, hyperparameters and schedules

The hyperparameter set of each built-in optimizer, how a hyperparameter is bound — baked, scheduled
or fed at runtime — the schedule factories, the dtypes and shapes a hyperparameter may take, and how
to write an optimizer of your own.

Related: [training.md](training.md) · [losses-and-optimizers.md](losses-and-optimizers.md)

## Optimizer hyperparameter sets

The thirteen optimizers of `Shorokoo.Modules.Optimizers` and their generated sets (the positional
`params Hyperparameter[]` count for `FromScratch` equals each set's property count):

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
| `AdafactorOptimizer` | `AdafactorOptimizerHyperparameters { LearningRate = 0.01, Beta2Decay = -0.8, Epsilon1 = 1e-30, Epsilon2 = 1e-3, ClipThreshold = 1.0, WeightDecay = 0 }` (6 positional; **non-factored** — full param-shaped 2nd moment) |
| `LambOptimizer` | `LambOptimizerHyperparameters { LearningRate = 0.001, Beta1 = 0.9, Beta2 = 0.999, Epsilon = 1e-6, WeightDecay = 0.01 }` (5 positional; `Epsilon` is LAMB's `1e-6`, not Adam's `1e-8`) |

## Hyperparameter kinds (`Hyperparameter`)

Each hyperparameter property is a `Hyperparameter` bound to exactly one source; its `Kind` decides
the wiring:

| Assign | Kind | Wiring |
|---|---|---|
| a bare value (e.g. `1e-4f`, `5`, `true`), or `Hyperparameter.Baked(v)` | `Baked` | graph `Constant`; change ⇒ rebuild |
| a `Schedule` (e.g. `Schedules.Cosine(3e-4f, total)`), or `Hyperparameter.Scheduled(schedule)` | `Scheduled` | computed **in-graph** from the counter input(s) each step |
| `Hyperparameter.Scheduled(module)` | `Scheduled` (module) | a scheduler module (int64 counter(s) → the declared-dtype value) inlined into the graph |
| `Hyperparameter.Runtime()` / `Hyperparameter.Runtime(shape)` | `Runtime` | runtime input; supply each step via `MakeHyperparameters` |

`Schedules` holds the factories (`Constant`, `Linear`, `Cosine`, `CosineWithWarmup`, `StepDecay`,
`Exponential`, `OneCycle`); a `Schedule` has combinators (`WithWarmup`, `Then`, `Scale`, `Clamp`,
`Shift`, `PerEpoch`) — see [Schedule factories and combinators](#schedule-factories-and-combinators).
`Schedule.At(long)` previews a value. Both types are in `Shorokoo.Core.Training`, which
`using Shorokoo;` does **not** cover; add `using Shorokoo.Core.Training;` — see
[`Shorokoo.Core.*` is not all internal](orientation.md#public-core-namespaces).

**Scheduler modules.** A scheduler module's inputs are a named subset of the int64 scalar counters
`{step, epoch, batchIndex}` and its single output is the value at the hyperparameter's declared
dtype. Built-in schedules are step-only (`PerEpoch` derives the epoch from the step). Rig build
rejects a scheduler graph with trainable params, module state / `StateUpdate`, RNG draws, or an
unrecognized input. There is no API for a host lambda schedule.

**Configurable milestones: `[Hyper]` + `Specialize`.** A module may declare milestones as `[Hyper]`
inputs and bake them with [`Specialize`](inference.md#hardcoding-hypers-with-specialize), which
removes them from the input list, before passing the graph to `Hyperparameter.Scheduled`:

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

`FromOrderedInputs` pairs values with the leading input names, and `[Hyper]` inputs come first, so
the value names `totalSteps`. The milestone is then fixed for the rig's life, like a `Baked` value.
Specializing `step` too yields a constant schedule.

Optimizer state is initialized at each hyperparameter's value at the initial counters (all 0).

**Reading back the applied value.** Every checkpoint a step returns carries
`.AppliedHyperparameters`: each hyperparameter's value in that step (scheduled: as computed
in-graph; baked: the constant; runtime: as fed), keyed by the producing rig's
`HyperparameterNames`, in that order. The step ran at counter `Step - 1`, so for a built-in
schedule the value is `schedule.At(ckpt.Step - 1)`, up to the numeric note below:

```csharp
var ckpt = rig.TrainStep(ckpt, input, target);
float lr = ckpt.AppliedHyperparameters!["learningRate"].ToSingle();
```

Each value is an immutable `AppliedHyperparameter` with `DType`, `Shape`, `ElementCount`,
`ToDouble()` / `ToSingle()` (single-element), `ToArray<T>()` (`T` the dtype's storage type) and
`ToTensorData()`. A resident run exposes its last step's map as `run.AppliedHyperparameters`. The
values are copied to the host as the step runs (for a runtime value kept on a device, a download
each step), so reading the map is free. It is `null` on an initial or loaded checkpoint, survives
`otherRig.AdoptCheckpoint(ckpt)` unchanged, and is not saved on its own — the
[training history](training-data.md#the-training-history) holds every step's values and is saved.

> **Numeric note.** On engines whose `Cos`/`Pow` differ from .NET `MathF` (e.g. ONNX Runtime), an
> in-graph schedule using those ops may differ from `Schedule.At` by a few ulps. Arithmetic and
> piecewise schedules are exact.

## Schedule factories and combinators

`s` is the 0-based global step and `f(s)` the value of the schedule a combinator is applied to.
The rig evaluates these in `float32`.

**Factories** (`Schedules.…`), each starting at step 0:

| Factory | Value at step `s` | Outside its nominal range |
|---|---|---|
| `Constant(float value)` | `value` | unchanging |
| `Linear(float baseValue, float finalValue, int totalSteps)` | `baseValue + (finalValue - baseValue) · p`, with `p = clamp(s / totalSteps, 0, 1)` | clamped: `baseValue` at and below step 0, `finalValue` from `totalSteps` on |
| `Cosine(float baseValue, int totalSteps)` | `0.5 · baseValue · (1 + cos(π · p))`, same `p` | held at `0` from `totalSteps` on |
| `CosineWithWarmup(float baseValue, int warmupSteps, int totalSteps)` | `Cosine(baseValue, max(1, totalSteps - warmupSteps)).WithWarmup(warmupSteps)` (a negative `warmupSteps` becomes `0`): a linear ramp from `0` to `baseValue` at `warmupSteps`, then cosine decay to `0` at `totalSteps` | held at `0` |
| `StepDecay(float baseValue, int stepSize, float gamma)` | `baseValue · gamma^(s / stepSize)`, **integer** division | never clamps |
| `Exponential(float baseValue, float gamma)` | `baseValue · gamma^s` | never clamps |
| `OneCycle(float maxValue, int totalSteps, float pctStart = 0.3f, float divFactor = 25f, float finalDivFactor = 1e4f)` | with `initial = maxValue / divFactor`, `final = initial / finalDivFactor`, `up = max(1, round(totalSteps · clamp(pctStart, 0, 1)))`, `down = max(1, totalSteps - up)`: for `s < up`, `initial + (maxValue - initial) · 0.5 · (1 - cos(π · s / up))`; for `s ≥ up`, `final + (maxValue - final) · 0.5 · (1 + cos(π · clamp((s - up) / down, 0, 1)))` | held at `final` from `totalSteps` on |

`totalSteps` (for `Linear`, `Cosine`, `CosineWithWarmup`, `OneCycle`) and `StepDecay`'s `stepSize`
must be at least 1, or the factory throws.

**Combinators** (chainable; each returns a new schedule):

| Combinator | Value at step `s` |
|---|---|
| `Scale(float factor)` | `factor · f(s)` |
| `Clamp(float min, float max)` | `clamp(f(s), min, max)`; throws if `min > max` |
| `Shift(int steps)` | `f(s + steps)` — positive `steps` moves the schedule **earlier**, negative moves it later |
| `PerEpoch(int stepsPerEpoch)` | `f(s / stepsPerEpoch)`, integer division; needs no epoch input; `stepsPerEpoch ≥ 1` |
| `WithWarmup(int warmupSteps, float startFactor = 0f)` | with `peak = f(0)`: for `s < warmupSteps`, `peak · (startFactor + (1 - startFactor) · s / warmupSteps)`; then `f(s - warmupSteps)` (**re-based**). `warmupSteps == 0` returns the schedule unchanged |
| `Then(int atStep, Schedule next)` | `f(s)` for `s < atStep`, then `next(s - atStep)` (**re-based**) |

`startFactor` multiplies the inner schedule's step-0 value, not the optimizer's default;
`Schedules.Constant(peak).WithWarmup(N)` ramps linearly from `0` to `peak` over steps `0 … N`.

**Worked example: warm up, hold, decay.** `Then` re-bases, so the second schedule's length is in
its own steps and the boundary in absolute steps:

```csharp
using Shorokoo.Core.Training;   // Schedule, Schedules

// Ramp 0 → 1e-3 over steps 0..200, hold 1e-3 to step 3899,
// then decay 1e-3 → 5e-5 over steps 3900..6000 and hold.
Schedule lr = Schedules.Constant(1e-3f)
    .WithWarmup(200)                                    // peak = Constant's step-0 value = 1e-3
    .Then(3900, Schedules.Linear(1e-3f, 5e-5f, 2100));  // Linear's own step 0 is global step 3900
```

| step | `lr.At(step)` |
|---|---|
| `0` | `0` |
| `100` | `5.0e-4` |
| `199` | `9.95e-4` |
| `200` … `3899` | `1e-3` |
| `3900` | `1e-3` (`Linear` at its step 0) |
| `4950` | `5.25e-4` |
| `6000`, `7000` | `5e-5` |

## Hyperparameter dtypes and shapes

A hyperparameter's dtype and rank are what the optimizer **declares** in its `[Hyper(...)]`
parameter (`Scalar<T>`, `Vector<T>` or `Tensor<T>`). Any supported dtype and shape works — an
`int32` count, a `bit` flag, a `float64` coefficient, a per-element rate vector.

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

**Defaults are scalar-only.** `[Hyper(default)]` takes a host literal of the declared dtype. A
non-scalar hyperparameter, or a dtype with no C# literal (e.g. `float16`), takes a bare `[Hyper]`;
its generated property is `required`, bound with `Hyperparameter.Baked(Globals.TensorData(…))`.

**Dtypes.** Host values (baked, or per-step via `MakeHyperparameters`) are fitted to the declared
dtype. Float-to-float is always allowed (only overflow to infinity is rejected). Every other
conversion must be value-preserving: `("accumSteps", 3L)` becomes `int32` 3; `2.5`,
`long.MaxValue`, or crossing the bool boundary fails loud. Non-scalar values are not converted —
build them at the declared dtype (`Globals.TensorData(dtype, shape, …)`). `rig.HyperparameterDTypes`
lists the declared dtypes in `rig.HyperparameterNames` order.

**Shapes.** The declaration fixes the rank; the binding gives the shape, reported by
`rig.HyperparameterShapes`:

| Kind | Where its shape comes from |
|---|---|
| `Baked` | the constant's own shape — `Hyperparameter.Baked(TensorData([4L], …))` |
| `Scheduled` (module) | the scheduler module's output shape, inferred at rig build |
| `Runtime` | declared by you: `Hyperparameter.Runtime(4L)`; `Runtime()` means a scalar |

A runtime shape is fixed for the rig's life; a per-step value of another shape fails loud. A
non-scalar hyperparameter must fit the parameters it updates, or the rig refuses it at build
([Custom optimizers](#custom-optimizers)). A built-in `Schedule` drives only `float32` **scalar**
hyperparameters; use a scheduler module for anything else.

```csharp
var rig = TrainingRig.FromScratch(model, loss, MyOptimizer.ComputationGraph, sample,
    new MyOptimizerHyperparameters
    {
        LearningRate  = Schedules.Cosine(1e-3f, totalSteps),        // float32 scalar, built-in schedule
        AccumSteps    = 4,                                          // int32, baked
        Nesterov      = true,                                       // bool, baked
        PerGroupScale = Hyperparameter.Runtime(3L),                 // float32 vector, host-supplied
    });

ckpt = rig.TrainStep(ckpt,
    rig.MakeHyperparameters(("perGroupScale", TensorData([3L], 1f, 2f, 3f))),
    inputs.Shared(), targets.Shared());
```

`MakeHyperparameters` copies any `TensorData` you give it, so a step never consumes yours. Its named
overload takes `(string name, object value)` pairs; `DynamicHyperparameterNames` lists the runtime
names.

The positional `FromScratch` overloads take the values as an array, followed by the optional
`rngConfig` / `mergeContext` / `runtimeContext`: `(sample, [0.05f], rng)`,
`(sample, [0.05f], rng, merge, runtime)`, `(sample, [], rng)`. The params form
`FromScratch(model, loss, opt, sample, 0.05f)` takes values only. Name a context passed without an
`rngConfig` (`(sample, [0.05f], mergeContext: ctx)`). A literal `null` in the hyperparameter slot
with an optional argument is ambiguous; pass the set or cast.

`Hyperparameter.Baked(v)` exposes `BakedValue` (the `TensorData`) and `BakedDType`.
`HyperAttribute.DefaultValue` is the attribute's literal (`object?`), and a graph input's
`HyperDefaultValue` its invariant string (`string?`), so `int64` / `float64` / `bool` defaults
round-trip exactly. A training `.skpt` records each baked binding's `dtype`, `shape` and base64
`value`, and each runtime binding's `shape`.

## Custom optimizers

A custom optimizer is a `[Module]` whose `Inline` takes exactly `(currentParam, grad)`, then its
`[Hyper]` hyperparameters, and returns the updated parameter. Each piece of state is created
**inside the body** by an optimizer-owned `[StateInitializer]`'s `Init` — typically
`OptimizerStateZeros.Init(currentParam.ShapeTensor())` from `Shorokoo.Modules.Optimizers` — and
updated with exactly one `Globals.StateUpdate(state, newValue)`. For example, a momentum-less
RMSprop (the full one is `RMSpropOptimizer` in [Shorokoo.Modules](losses-and-optimizers.md#optimizers-shorokoomodulesoptimizers)):

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

This generates `SimpleRMSpropHyperparameters { LearningRate = 0.001, Alpha = 0.99, Epsilon = 1e-8 }`
with schedule support, and a `meanSquare` state field per trainable parameter, zero-initialized at
its shape:

```csharp
var rig = TrainingRig.FromScratch(model, loss, SimpleRMSprop.ComputationGraph, sample,
    new SimpleRMSpropHyperparameters { LearningRate = Schedules.Cosine(1e-3f, totalSteps) });
```

For another initial value, write an initializer (any `Inline`; the rig runs it with the inputs you
wire in the body):

```csharp
[StateInitializer(Ownership = StateOwnership.OptimizerOwned)]
public static partial class OptimizerStateOnes
{
    public static Tensor<float32> Inline(Vector<int64> shape) => Globals.TensorFill(shape, 1.0f);
}
```

For one value per parameter (a step counter, a scalar EMA), use `OptimizerScalarZeros.Init()`
(seeded 0), `OptimizerScalarOnes.Init()` (seeded 1, e.g. NAdam's running product `∏μ_i`), or your
own rank-0 initializer; the scalar broadcasts and costs one float per parameter. Adam's and AdamW's
timestep works this way.

Constraints:

- **State must come from an optimizer-owned initializer.** State declared as an `Inline` parameter
  throws at rig build; `Globals.StateUpdate` throws `InvalidStateUpdateException` on a non-state
  argument. Module-owned initializers (e.g. BatchNorm's) are rejected in optimizer graphs, and
  optimizer-owned ones in model graphs.
- **Each state is updated exactly once per step** — merge conditional updates into one value (e.g.
  with `IfElse`).
- **The updated parameter must keep the parameter's shape.** A hyperparameter or state of another
  shape broadcasts; the rig refuses that at build, naming the parameter and both shapes. A
  per-element hyperparameter needs parameters it fits.
- **Hyperparameters must be tensor-shaped** (`Scalar<T>`, `Vector<T>`, `Tensor<T>`, any supported
  dtype); a set is generated even with mixed dtypes and shapes. An `OptionalTensor`, sequence or
  struct hyperparameter yields no generated set. Only scalars can have a `[Hyper(default)]`.
- **Order + `[Hyper]` matter** — `Inline` takes `(currentParam, grad)` first and the hyperparameters after, and `[Hyper]` is what
  makes the named set generate. Without it the optimizer works only through the positional
  `params Hyperparameter[]` overload.
- You can also implement `IOptimizerHyperparameters` by hand.

## Notes / known limitations

- `LionOptimizer` **swaps the beta roles** relative to Adam: momentum `m` decays by **β2**
  (`m = β2·m + (1−β2)·g`); **β1** only blends the sign update. Its good `lr` is ~3–10× smaller than
  AdamW's and its `wd` ~3–10× larger (default `wd 0`).
- `AdafactorOptimizer` is **non-factored**: it keeps Adafactor's update rules (relative step
  `min(lr, 1/√t)`, parameter scaling, RMS update clipping, increasing decay `1 − t^τ`) but its
  second moment is full param-shaped — **Adam's memory**, not `R + C`. The per-parameter optimizer
  graph is rank-agnostic, so it cannot factor (see [losses-and-optimizers.md](losses-and-optimizers.md#optimizers-shorokoomodulesoptimizers)). `learningRate`
  is the **cap** on the relative step.
- Prefer the generated named set. The positional `params Hyperparameter[]` overload must match the
  count exactly: SGD=1, SGDMomentum=2, Adam=4, RMSprop=4, AdamW=5, Adagrad=2, Adamax=4, NAdam=5,
  RAdam=4, Adadelta=3, Lion=4, Adafactor=6, Lamb=5.
- Optimizer state per trainable parameter (table in [losses-and-optimizers.md](losses-and-optimizers.md#optimizers-shorokoomodulesoptimizers)):
  momentum: velocity; Adam/AdamW: `m`/`v` + scalar `step`; RMSprop: `squareAvg`/`momentumBuffer`;
  Adagrad: `accumulator`; Adamax: `m`/`u` + scalar `step`; NAdam: `m`/`v` + scalars `step`,
  `muProduct`; RAdam: `m`/`v` + scalar `step`; Adadelta: `squareAvg`/`accDelta`; Lion: `m` only
  (half of Adam); Lamb: `m`/`v` + scalar `step` (trust ratio not stored); Adafactor: full `v` +
  scalar `step`. Param-shaped fields come from `OptimizerStateZeros`; scalar ones from
  `OptimizerScalarZeros` (seeded 0) or `OptimizerScalarOnes` (seeded 1, NAdam's product).
