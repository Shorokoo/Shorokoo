# Training models

Related: [defining-models.md](defining-models.md) · [nn-library.md](nn-library.md) · [inference.md](inference.md) · [training-backends.md](training-backends.md)

## Facts

- Training composes three `[Module]` graphs — a **model**, a **loss**, and an **optimizer** —
  accessed via their `.ComputationGraph` property.
- `TrainingRig` is the entry point. It runs autodiff on the composed graph and produces a
  trainable step. You never write backward passes. A rig built with
  `trainingBackend: TrainingBackend.Native` leaves the gradient to the backend that runs the step —
  see [training-backends.md](training-backends.md).
- `TrainStep` moves the whole training state through host memory every step, which dominates on a
  GPU; run long loops with `rig.BeginResidentRun()` (or `Fit` / `Train`, which use one) — see
  [Keeping training state on the device](#keeping-training-state-on-the-device).
- A step **consumes** what it is fed as it is (the checkpoint's state and the batch), so
  `cp = rig.TrainStep(cp, x, y)` releases superseded state as it runs. Feed `cp.Shared()` /
  `x.Shared()` to keep something past the step. `CreateInitialCheckpoint()` returns fresh copies
  every call — [What a training step consumes](#what-a-training-step-consumes).
- A step writes updated state over the state it consumed where the graph proves nothing still
  reads the old value — [A step writes its state over the state it consumed](#a-step-writes-its-state-over-the-state-it-consumed).
- State (optimizer moments, momentum velocity, BatchNorm running stats) is **created** by a
  `[StateInitializer]` class's `Init(...)` call inside a module's `Inline`, and its per-step update
  is registered with `Globals.StateUpdate(state, newState)`, which throws
  `InvalidStateUpdateException` if its first argument is not a state variable (e.g. a runtime input
  or a trainable parameter).

## Built-in components

Ready-made losses and optimizers ship in `Shorokoo.Modules` (namespaces
`Shorokoo.Modules.Losses` / `Shorokoo.Modules.Optimizers`) — see [nn-library.md](nn-library.md)
for the catalog (sixteen losses; layers and initializers too). Each optimizer whose
hyperparameters are all tensor-shaped gets a source-generated, named, defaulted set
(`<Optimizer>Hyperparameters`) implementing `IOptimizerHyperparameters`. The thirteen optimizers
(the positional `params Hyperparameter[]` count for `FromScratch` equals each set's property count):

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

A loss module has signature `(predictions, targets) -> Scalar<float32>` with exactly two tensor
inputs; targets are typically `Tensor<float32>`, but class-index losses (`CrossEntropyLoss`,
`NLLLoss`) take `Tensor<int64>`. The library losses' knobs (`reduction`, `ignore_index`,
`label_smoothing`, class `weight`/`pos_weight`, SmoothL1 `beta`) live on extra
`Reduced`/`PerElement` methods, not on the rig-bound `Inline`. To use them in a rig, write a
2-input wrapper `[Module]` whose `Inline` calls `Reduced(...)` with the knobs baked; a class
`weight`/`pos_weight` must be **baked as a graph constant** there. See
[Losses → Configurable knobs](nn-library.md#loss-configurable-knobs).

<a id="loss-ignoring-targets"></a>
**A loss graph may ignore its `targets`.** Two inputs is a signature requirement only: the rig
checks whether the loss output reaches input 1 (through data edges and branch conditions) and
derives the target slot from that. So a model that computes its **own** loss can be trained with a
pass-through loss. Use this when the loss needs more than one predictions and one targets tensor —
label ids, a padding mask, per-token weights, `Reduced`/`PerElement` knobs: pass those as ordinary
**model** inputs (see [Which knobs reach the rig](nn-library.md#loss-configurable-knobs)) and make
the model's single output the scalar loss:

```csharp
[Module]                                  // the model already returns the scalar loss;
public partial class PassThroughLoss      // its labels / mask are ordinary model inputs
{
    public static Scalar<float32> Inline(Scalar<float32> predictions, Tensor<float32> targets)
        => predictions;                   // `targets` unused — legal, and never read
}
```

**You do not feed the ignored input.** When the loss does not read its second input,
`rig.TargetDef` is empty, `rig.HasTargets` is `false`, and the target-free overloads take the
model inputs alone:

```csharp
ckpt = rig.TrainStep(ckpt, inputs);              // the real labels ride inside `inputs`
var result = rig.Fit(batches, numEpochs: 10);
```

The target-free overloads are `TrainStep(checkpoint, inputs)`,
`TrainStep(checkpoint, inputs, epoch, batchNumber)`, `Fit(inputs, numEpochs)`, and the resident
run's `Step(inputs)` / `StepToCheckpoint(inputs)`. Calling one on a rig whose loss reads its target
fails loud. The loader path takes an empty target dataset:

```csharp
var loader = new InMemoryDataLoader(inputs, rig.TargetDef.FromOrderedData(), batchSize: 32);
var result = rig.Fit(loader, numEpochs: 10);
```

`TrainStep(checkpoint, hyperparams, …)` and the resident run's hyperparameter forms still take a
target argument; pass `rig.TargetDef.FromOrderedData()`, an empty struct.

This shape saves no memory: the predictions tensor is never a step output, and the step does the
same work op for op whichever side the loss sits on. `ExtractInferenceModel` returns the model
**as authored**, so a loss-computing model yields an inference model that returns a loss and
demands the labels; author the prediction path as its own module if you need one. For a
*validation loss* from a saved checkpoint, use `Persistence.LoadEvaluationModel(path)` instead,
which composes the model with its loss and needs no rig.

An optimizer module's `Inline` takes exactly `(currentParam, grad)`, then its `[Hyper]`
hyperparameters (which become the graph's leading inputs), and returns the updated parameter.
Optimizer state never appears in the signature: it is created inside the body by an
**optimizer-owned state initializer** (e.g. `OptimizerStateZeros.Init(currentParam.ShapeTensor())`)
and updated with one `Globals.StateUpdate(state, newValue)` call per state — see
[Custom optimizers](#custom-optimizers).

### Hyperparameter kinds (`Hyperparameter`)

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
[training history](#the-training-history) holds every step's values and is saved.

> **Numeric note.** On engines whose `Cos`/`Pow` differ from .NET `MathF` (e.g. ONNX Runtime), an
> in-graph schedule using those ops may differ from `Schedule.At` by a few ulps. Arithmetic and
> piecewise schedules are exact.

### Schedule factories and combinators

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

### Hyperparameter dtypes and shapes

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

## `TrainingRig` API

```csharp
public static TrainingRig FromScratch(
    ComputationGraph modelGraph,      // GraphKind.Module, or a ToConcreteArchitecture result
    ComputationGraph lossGraph,       // kind must be GraphKind.Module
    ComputationGraph optimizerGraph,  // kind must be GraphKind.Module
    IData[] sampleInputs,                      // one sample per model input, in declaration order
    IOptimizerHyperparameters hyperparameters, // named set, e.g. new AdamWOptimizerHyperparameters { ... }
    RngConfig? rngConfig = null,              // seeds the run — see "Seeding the run" below
    ComputeContext? mergeContext = null,      // build/merge-phase context (rig.MergeContext); null ⇒ Default
    ComputeContext? runtimeContext = null,    // compile/run context (rig.RuntimeContext); null ⇒ Default
    IProgress<BuildProgress>? progress = null, // see "Watching a long build"
    TrainingBackend? trainingBackend = null);  // see training-backends.md; null ⇒ TrainingBackend.Shorokoo

// Positional values (a float bakes, a Schedule schedules):
//   FromScratch(model, loss, opt, sampleInputs, params Hyperparameter[] hyperparameters)
//   FromScratch(model, loss, opt, sampleInputs, Hyperparameter[] hyperparameters,
//               RngConfig? rngConfig = null,
//               ComputeContext? mergeContext = null, ComputeContext? runtimeContext = null)
// Each form also takes sampleInputs as NamedModelParam[] or ModelParamList (see "Sample inputs").

// Fresh host copies of the rig's initial values, new every call. Throws if an optimizer state
// initializer reads a Runtime hyper — use the overload taking its values.
public TrainingCheckpoint CreateInitialCheckpoint();
public TrainingCheckpoint CreateInitialCheckpoint(TensorDataStruct hyperparameters); // from MakeHyperparameters(...)

// Scheduled hypers computed in-graph from the checkpoint's Step; requires no Runtime hypers.
// Returns the next checkpoint (.Loss, .AppliedHyperparameters, .History updated). Compiles lazily,
// cached per fed shape. Arguments: a TensorDataStruct, or one via .Shared()/.TryConsume();
// anything else throws ArgumentException.
public TrainingCheckpoint TrainStep(
    TrainingCheckpoint checkpoint,
    IData trainingInput,
    IData trainingOutput);

// Supplies the Runtime hyperparameter values for this step.
public TrainingCheckpoint TrainStep(
    TrainingCheckpoint checkpoint,
    IData hyperparams,                         // from MakeHyperparameters(...)
    IData trainingInput,
    IData trainingOutput);

// Draws loader.Next(); the batch's position drives the scheduler and is recorded on the result.
// Requires no Runtime hypers.
public TrainingCheckpoint TrainStep(
    TrainingCheckpoint checkpoint,
    IDataLoader loader);

// epoch / batchNumber name the batch being trained: fed to the scheduler and recorded verbatim.
public TrainingCheckpoint TrainStep(
    TrainingCheckpoint checkpoint,
    IData trainingInput,
    IData trainingOutput,
    long epoch,
    long batchNumber);

public TensorDataStruct MakeHyperparameters(float value);                       // exactly one dynamic
//   also: (double), (int), (long), (bool), and (TensorData) for other dtypes / non-scalar shapes
public TensorDataStruct MakeHyperparameters(params (string name, object value)[] values); // named

// One array element per step; the arrays are read (not consumed) every epoch.
public TrainingResult Fit(
    TensorDataStruct[] trainingInputs,
    TensorDataStruct[] trainingOutputs,
    int numEpochs,
    TrainingCheckpoint? initialCheckpoint = null); // defaults to CreateInitialCheckpoint()

// The loader owns the batch stream; step / epoch / batch advance for you.
public TrainingResult Fit(
    IDataLoader loader,
    int numEpochs,
    TrainingCheckpoint? initialCheckpoint = null); // defaults to CreateInitialCheckpoint()

// The array Fit with the checkpoint first and required (not interchangeable argument order).
public TrainingResult Train(
    TrainingCheckpoint initialCheckpoint,
    TensorDataStruct[] trainingInputs,
    TensorDataStruct[] trainingOutputs,
    int numEpochs);

// Keeps training state on the device — see "Keeping training state on the device".
// The initial checkpoint is fed to the first step as TrainStep feeds one.
public ResidentTrainingRun BeginResidentRun(TrainingCheckpoint? initialCheckpoint = null);
```

```csharp
public sealed class ResidentTrainingRun : IDisposable
{
    // One step; returns only the loss, so nothing is downloaded.
    public float Step(IData trainingInput, IData trainingTarget);
    public float Step(IData hyperparameters,                            // from MakeHyperparameters(...)
                      IData trainingInput, IData trainingTarget);
    public float Step(IDataLoader loader);                              // draws loader.Next()
    public float Step(DataBatch batch);                                 // a batch you drew yourself

    // One step that also brings the state to the host as a checkpoint.
    public TrainingCheckpoint StepToCheckpoint(IData trainingInput, IData trainingTarget);
    public TrainingCheckpoint StepToCheckpoint(IData hyperparameters,
                                               IData trainingInput, IData trainingTarget);
    public TrainingCheckpoint StepToCheckpoint(IDataLoader loader);
    public TrainingCheckpoint StepToCheckpoint(DataBatch batch);

    public long CurrentStep { get; }   // the Step the run's last checkpoint carries; next is +1
    // Last step's values, host-side (no download).
    public IReadOnlyDictionary<string, AppliedHyperparameter>? AppliedHyperparameters { get; }
    // Starting checkpoint's history plus one entry per step (host-side).
    public TrainingHistory History { get; }
    // Replace or empty the history; later steps append to what is left.
    public void ReplaceHistory(TrainingHistory history);
    public void ClearHistory();

    public void Dispose();             // releases state the run still holds; published checkpoints survive
}
```

### What a training step consumes

A training step feeds its inputs like any run
([inference.md](inference.md#feeding-a-run-consumed-shared-or-tried)): what it is given as it is,
it **consumes** — dead once the step starts, memory returned as the step returns — and what it is
given `.Shared()` it only reads.

```csharp
cp = rig.TrainStep(cp, x.Shared(), y.Shared());    // a batch fed again: read, and kept
cp = rig.TrainStep(cp, x, y);                      // cp's state, x and y are consumed
var next = rig.TrainStep(best.Shared(), x2, y2);   // a checkpoint kept past the step: read
```

**What a step keeps of what it reads.** A tensor the run cannot address in place (any tensor built
from a C# array, and a host tensor on a card) is read through a copy that the tensor keeps for
reuse ([inference.md](inference.md#feeding-a-run-consumed-shared-or-tried)). A training step
releases the copies of its **batch** as it returns, success or failure, so a `.Shared()` dataset is
not held a second time (on a card, not uploaded whole); each step copies its batch afresh. Copies of
the **checkpoint's state** are kept: a checkpoint fed `.Shared()` on a card keeps a copy on the card
while it lives, so drop kept checkpoints when done. In a resident run, a checkpoint it does not own
(the one you began from `.Shared()`, or one `StepToCheckpoint` returned) is read by one step only,
which releases its copies.

- **A checkpoint** is fed as its `FeedMode` says. `null` (every checkpoint a step or load returns)
  is consumed. `cp.Shared()` returns a checkpoint over the same tensors to be read, and
  `cp.TryConsume()` one consumed only where nothing else reads it; derivations (`WithStep`,
  `WithCounters`, `WithTrainableParams`, …) and `rig.AdoptCheckpoint` keep the mode. Reading a
  consumed checkpoint's state throws, naming the step that took it and the section
  ("the checkpoint's trainable parameter …").
- **An initial checkpoint is a copy.** `CreateInitialCheckpoint()`, and a load that falls back on
  the rig for an omitted component, copy the rig's initial values every call; the rig's own values
  are never fed to a run. To start several steps from one initial state, call it once each, or pass
  one checkpoint `.Shared()`.

  ```csharp
  var start = rig.CreateInitialCheckpoint();
  var a = rig.TrainStep(start.Shared(), x.Shared(), y.Shared());   // start is kept
  var b = rig.TrainStep(start, x, y);                              // start is consumed
  ```
- **Batches** are fed as passed. `TrainStep`, a resident run's `Step` and a `DataBatch` take a
  `TensorDataStruct` or one via `.Shared()` / `.TryConsume()`; anything else throws
  `ArgumentException` naming the struct definitions to build from
  (`rig.InputDef.FromOrderedData(...)`).
- **Per-field modes.** A field given `.Shared()` or `.TryConsume()` at construction is fed that way:
  `rig.InputDef.FromOrderedData(tokens, mask.Shared())` keeps the mask and consumes the tokens. A
  field given `.Shared()` is always read; a struct fed `.Shared()` has every field read; otherwise
  each field is fed as given, or as the struct is. `Fields` and the indexer return the tensor
  itself; `To`, `CopyTo`, `ToHost` and `rig.AdoptCheckpoint` keep each field's mode.
- **Runtime hyperparameters** are fed like a batch. `MakeHyperparameters` builds a fresh struct and
  copies the tensors given to it.
- **`Fit` and `Train` over arrays read their batches** and feed the initial checkpoint like
  `TrainStep`. `Fit` over a loader feeds each batch as the loader built it: `InMemoryDataLoader`
  builds a fresh batch per draw, which is consumed; a custom loader that reuses its tensors passes
  them `.Shared()` in its `DataBatch`.
- **A step that fails after starting has still consumed its as-is inputs.** A resident run then
  refuses every later step — see [below](#keeping-training-state-on-the-device).

### Keeping training state on the device

`TrainStep` returns a host copy of every parameter, model state and optimizer state and takes them
back next call. On CPU that is free; on a GPU the whole state crosses the bus twice per step, so
step time tracks parameter count rather than FLOPs. `BeginResidentRun` keeps the state where the
execution provider produced it:

```csharp
using var run = rig.BeginResidentRun();
for (int step = 0; step < 50_000; step++)
{
    if (step % 1_000 == 999)
        run.StepToCheckpoint(loader).Save($"ckpt-{step}.safetensors");  // this step transfers
    else
        run.Step(loader);                                               // no transfer
}
```

- **`Step`** returns only the loss.
- **`StepToCheckpoint`** runs a step (use it *instead of* `Step`) and returns the state as an
  ordinary `TrainingCheckpoint`. Use it on every step you want a checkpoint at, including the last.
- **`Dispose`** discards what the run still holds; checkpoints it already returned stay valid.
- **Each step consumes the state the previous step produced**, writing over it where it can
  ([below](#a-step-writes-its-state-over-the-state-it-consumed)). The starting checkpoint is
  consumed unless passed `.Shared()`; the default is a fresh `CreateInitialCheckpoint()`.
- **A failed step can end the run.** If it fails after consuming the run's own state, every later
  step throws `InvalidOperationException` saying so and what is left to restart from. Restart from
  the last `StepToCheckpoint` result (a failure while reading a published checkpoint leaves it and
  the run intact), or, before any, from the starting checkpoint if you passed it `.Shared()`.
- **Feeding a published checkpoint elsewhere as-is consumes the run's state too**, since they share
  tensors; the run's next step is then refused with that state's error, naming the step that took it.

`Train` and every `Fit` overload use a resident run internally and transfer only on the final step.
A manual `TrainStep` loop transfers every step. On a backend with no device memory, a resident run
gives the same losses and checkpoints, to the bit, as a step loop.

> A checkpoint's tensors are readable only on the host. Reading state a run still holds on the
> device throws; get it through `StepToCheckpoint`.

### A step writes its state over the state it consumed

ONNX Runtime holds every run input until the run ends, so consumed memory cannot be reused
mid-step. Instead the step writes each updated state field (weight, optimizer moment) into the
memory of the field it replaces, wherever the graph proves every reader of the old value runs
before the write. The backend re-checks this on the graph ONNX Runtime actually runs.

It applies:

- **To consumed state** — a checkpoint fed as it is, and a resident run's own state. A `.Shared()`
  checkpoint is left untouched.
- **Where new state lands in the old state's memory** — every step on a CPU backend, and a resident
  run's `Step` on a GPU. `TrainStep` and `StepToCheckpoint` on a GPU return state to the host, so
  they overwrite nothing on the card.
- **Where the graph proves it.** Optimizer state read only by its own update qualifies (e.g. AdamW's
  moments and step counter). A weight the backward pass reads directly to propagate a gradient is
  written anew; for a stack of `Linear` layers under AdamW every weight, bias and moment is written
  over.

Results are bit-identical with or without it, and there is nothing to configure. On a card a
resident run thus holds its state once rather than twice; under a device-memory budget the state
is counted once — see [inference.md](inference.md#a-contexts-device-memory-budget).

### What construction costs

`FromScratch` concretizes, composes the loss, runs autograd, lowers the optimizer, and runs shape
inference and graph optimization, all on `MergeContext`. It also runs each trainable parameter's
initializer and each optimizer-state initializer per parameter. Parameters with the same
initializer and shape share one initialization session, and optimizer-state initializers share one
per parameter dtype and rank, so a model whose layers repeat builds a handful of sessions however
deep it is; what grows with the parameter count is the drawing itself. On a CPU context the draws
run side by side, each on a single thread, as many at once as there are cores and as fits
comfortably in memory — unless one parameter holds more than half of the model's elements, when
they run one after another, each over every core, as they do on a card. The values are the same
either way. A random initializer draws a bounded chunk of values at a time, so peak host memory is
the values produced so far plus one chunk's working memory per draw in progress.

`TrainingRig.Load` repeats all of this except concretization (it reads the saved architecture) and
the model's initializers: it uses zero stand-ins of the declared shape, since the checkpoint
overwrites them. The first call that needs initial *values* — `CreateInitialCheckpoint()`, or a
load that falls back on the rig for an omitted component — runs the initializers then, with the
values an eager build would produce, and re-seeds the optimizer state.

The first `TrainStep` at each input shape compiles the step graph and caches it, so expect the first
step to be much slower. Neither cost recurs during the loop or scales with the dataset.

Steady-state memory is flat: consumed state is freed as each step returns. Checkpoints you drop but
did not let a step consume (fed `.Shared()`, or `.TryConsume()` while shared) hold backend memory
behind small managed handles, so the rig triggers a garbage collection once more than 32 MiB of
such state has accumulated across steps, and backs off (doubling the threshold) while collections
reclaim nothing, e.g. when you keep every checkpoint. You need not collect yourself. Initial
checkpoints count against the same budget; a resident run's `Step` bypasses it, `StepToCheckpoint`
results do not.

On a large model the build can take minutes; see [Watching a long build](#watching-a-long-build).

### Compute contexts: `MergeContext` and `RuntimeContext`

A rig has two `ComputeContext`s, both set at construction (default `ComputeContext.Default`) and
never saved in a checkpoint; pass them again to `FromScratch` or
`TrainingRig.Load(path, mergeContext, runtimeContext)`. `MergeContext` runs the build phase
(concretization, shape inference, lowering and memory optimization, optimizer state init).
`RuntimeContext` compiles and runs the training step for `TrainStep`, `Train` and `Fit`; none takes
a per-call override, so all share one set of compiled sessions (one per input shape). `With…`
derivations keep both. The rig's `TrainingBackend`, also never persisted, decides who computes the
gradient — see [training-backends.md](training-backends.md).

Each context carries its backend, its `DeviceMemory` (a device-memory budget and arena settings)
and `RunSettings` — see [Device memory](inference.md#device-memory-gpu-backends). So a rig can
build on one device and train on another:

```csharp
var rig = TrainingRig.FromScratch(
    model, loss, optimizer,
    sampleInputs, new AdamWOptimizerHyperparameters { LearningRate = 0.001f },
    mergeContext:   new ComputeContext(new LinuxCpuBackend()),
    runtimeContext: new ComputeContext(new LinuxGpuBackend()));
```

Both devices must be reachable from one process, and merge-phase output reaches the runtime
backend by a host copy per feed — see [One model, two devices](inference.md#one-model-two-devices).
Split only when the build does not fit on the card; leaving both `null` is normal.
`rig.MergeContext.Backend` / `rig.RuntimeContext.Backend` name the devices;
`DefaultBackend.RequireDevice(...)` refuses the wrong one — see
[Which device am I on?](inference.md#which-device-am-i-on).

On GPU backends the runtime context's budget covers the state and batches it holds on the card plus
the running step's arena; a step that would exceed it is rebuilt with a smaller arena — see
[A context's device-memory budget](inference.md#a-contexts-device-memory-budget). The default arena
strategy `Auto` uses exact-size extension for a step compiled for one shape. The rig keeps a
compiled step for up to four input shapes; further shapes share one shape-generic step, which `Auto`
gives ORT's doubling. Feed a few stable batch shapes to stay on the tighter arena.

A context's settings are fixed at construction: budget, arena strategy, `ShrinkArenaAfterRun`
(implied by a budget), and the `CancellationToken` that abandons a long `Fit` or `Train` — see
[Stopping a run](inference.md#stopping-a-run). Put them on the `runtimeContext` you pass to
`FromScratch`. To watch a run near the card's limit, read the static `DeviceMemory` class and the
context's `ReadDeviceMemoryUse()` inside your loop.

**Host vs device memory.** The rig's collection governs *host* memory (a `TrainStep` loop's
checkpoints); `DeviceMemory` settings govern only device memory. A resident run keeps state on the
card and frees it deterministically.

Result types:
- `TrainingCheckpoint`:
  - `.TrainableParams`, `.ModelState`, `.OptimizerState`.
  - `.Step` (`long`, global step, advanced by each step; restored on load so schedules resume).
  - `.Epoch` / `.BatchIndex` (`long?`): set by the loader-driven and explicit-counter paths, carried
    through unchanged by the counter-agnostic `TrainStep`, `null` when unknown (a scheduler reading
    them sees `0`). All counters are `int64`.
  - `.Rig` (the producing `TrainingRig?`, so `ToInferenceModel()` needs no graph).
  - `.Loss` (`float?`; `null` on an initial or bare checkpoint; saved as its own `Loss` component).
  - `.AppliedHyperparameters` (see [Hyperparameter kinds](#hyperparameter-kinds-hyperparameter)).
  - `.History` (see [The training history](#the-training-history)).

  `WithCounters`/`WithStep`/`WithEpoch`/`WithBatchIndex` and
  `WithTrainableParams`/`WithModelState`/`WithOptimizerState` return a new checkpoint with one slot
  replaced and everything else carried; the receiver is never mutated.
- `TrainingResult` → `.FinalCheckpoint`, `.EpochLosses` (per-epoch mean losses).

**Constructing one directly.** The three state slots are **required** init properties, so the
compiler rejects a missing one and the call site names each (they share a type and often shapes):

```csharp
var checkpoint = new TrainingCheckpoint
{
    TrainableParams = trainable,
    ModelState      = modelState,
    OptimizerState  = optimizerState,
    Step            = 42,          // Epoch / BatchIndex / Rig / Loss are optional
};
```

To change one slot, prefer a derivation such as `ckpt.WithTrainableParams(next)`.

`TrainingRig`, `TrainingCheckpoint`, and `TrainingResult` are in namespace `Shorokoo`.

### Watching a long build

`FromScratch` can take **minutes** on a large model and reports nothing by default. Hand it a sink:

```csharp
using Shorokoo.Graph;    // SynchronousBuildProgress

var rig = TrainingRig.FromScratch(
    MyModel.ComputationGraph, L2Loss.ComputationGraph, AdamWOptimizer.ComputationGraph,
    sampleInputs, new AdamWOptimizerHyperparameters { LearningRate = 0.001f },
    progress: new SynchronousBuildProgress(p => Console.WriteLine(p)));
```

```
[   0.0s] Concretize: Thaw
[   0.1s] Concretize: Clone
[   0.1s] Concretize: ApplyIdentifierTemplates
…
[  38.4s] Concretize: ExpandAutoGrad
[  44.1s] Concretize: SimplifyAfterAutoGrad
[  44.2s] Concretize: BindRngConfig
[  44.3s] Concretize: WriteRepresentativeInputs
[  44.3s] TrainingStep: NormalizeOptimizerGraph
[  51.9s] TrainingStep: ComposeModelLossAndAutoGrad
[  63.0s] TrainingStep: ReplayOptimizerPerParameter
…
[  91.4s] Initialize: InitializeModelParams
…
[  96.2s] Initialize: OptimizeTrainingStepGraph
[ 118.9s] Initialize: FreezeTrainingStepGraph
[ 121.7s] Initialize: Done
```

(`…` marks elided stages.) Each `BuildProgress` is reported on **entering** a stage, so a quiet
build is inside the last-named stage. Members:

- `Phase` — `BuildPhase.Concretize`, `BuildPhase.TrainingStep`, then `BuildPhase.Initialize`
  (initializers, shape inference, graph optimization); they never interleave.
- `Stage` — the stage name. Diagnostic only; names change, so never branch on one.
- `IsComplete` — true only on the terminal report of a successful build (rendered `Done`); a build
  that throws emits none. Test this, not the string.
- `Elapsed` — time since this build started, across all phases.

`ToString()` renders the lines above. Reports are raised **synchronously on the building thread**:
use `SynchronousBuildProgress`, not `System.Progress<T>` (whose callbacks can arrive out of order or
late). Keep the handler short; an exception from it aborts the build. Give concurrent builds
separate sinks, or their reports interleave.

Only the call you pass a sink to reports. Builds that accept one:
- `FromScratch` (all phases);
- `With…` derivations (from `TrainingStep`), except `WithSeed` (from `Concretize`);
- `TrainingRig.Load` (from `Concretize`, naming its file reads; complete once the checkpoint is
  loaded). It reports `DeferModelParamInitialization` instead of `InitializeModelParams`; the
  later call that runs the initializers reports nothing;
- `ToConcreteArchitecture`, which reports `Concretize` and ends complete:

```csharp
var concrete = MyModel.ComputationGraph.ToConcreteArchitecture(
    inputHints, progress: new SynchronousBuildProgress(p => Console.WriteLine(p)));
```

The `params Hyperparameter[]` shorthands cannot take a sink; pass the values as an array:
`FromScratch(model, loss, opt, sample, [0.01f], progress: sink)`,
`rig.WithOptimizer(opt, [0.01f], sink)`. `ToConcreteModel`, `InitializeTrainableParams`,
`GetRngStreamReport` and the first `TrainStep`'s compile ([above](#what-construction-costs)) report
nothing.

To capture the *graphs* as compilable C#, see [debugging.md](debugging.md).

### Seeding the run

`rngConfig` binds the run's [RNG configuration](rng-configuration.md): one master seed keys
parameter initialization and every runtime draw (Dropout masks, in-model sampling). Omitted (or
`null`), the rig uses the **default identity** (master seed 0), so training is deterministic by
default. Dropout masks vary per step; the RNG position is saved in the checkpoint, so a resumed run
continues exactly.

On the CPU backends, two runs from one seed with the same data are bit-identical at every
step, including across a save and resume, so the CPU backend is the way to verify that a
resume is exact. The GPU backends are not bit-reproducible: two GPU runs from one seed can
differ in the last digits of the loss from early steps, with or without a resume in between. There is no deterministic GPU mode
([#405](https://github.com/Shorokoo/Shorokoo/issues/405)). Pass `new RngConfig { MasterSeed = … }` to re-roll all
streams coherently, or `RngConfig.NonDeterministic()` for per-run variation.

### When a training step runs out of memory

An allocation failure in a step is rethrown as a `ComputeContextException` with code `CR009`
reporting:

- **Which pool**: `HOST memory` (a failed C++ allocation; a bare `bad allocation` is host even on a
  GPU) or `DEVICE memory` (the accelerator's arena). ORT's arena message is the same for the CPU
  arena, so on a CPU-only session it reads as host. Where the backend names no allocator on a
  session with device memory, the report says so.
- **What the step held**: trainable parameters, model state, optimizer state and the batch, each
  with tensor count and size, plus the five largest tensors.
- **The card's figures** (where a CUDA runtime is installed; `DeviceMemory.Read()`): used, free and
  total across processes, and this process's arena cap if set.
- **This process's memory**: working set, commit charge and managed heap against the limit in force
  (cgroup/container, Job Object, or machine RAM).

On Windows/WDDM, device allocations count against system commit, so a process memory limit also
caps device memory and fails with the same message as a full card. The report distinguishes three
cases: the device is full; the device has room but the process is at its limit (raise the limit);
or both have room and the arena could not extend by the block it wanted.

```
[CR009] Compute context operation failed in TrainingRig.TrainStep: allocating memory for the training
step at step 1 failed. The failing allocation was for DEVICE memory — the accelerator's arena (backend
'Shorokoo.WinGPU'). Training state held for this operation: 296 tensor(s), 1.83 GiB in total (...).
Device: 12.59 GiB of 23.99 GiB in use across all processes, 11.4 GiB free. Host process: working set
9.61 GiB, commit 27.4 GiB, managed heap 3.02 GiB; against a configured memory limit of 28 GiB (98%
used). The device has room, yet this process is close to its own memory limit — and on Windows/WDDM a
device allocation is backed by system commit, so a limit meant to bound HOST memory bounds DEVICE
memory too ... This is the limit, not the model: re-run with it raised or removed. Underlying failure:
[ErrorCode:Fail] ...bfc_arena.cc:358 ...
```

The backend's text is kept verbatim at the end and the original exception as `InnerException`.
Other failures keep their type and message. If the host runs out while the **garbage collector**
needs memory, the runtime fails fast (`Fatal error. 0xE0004743`) with no exception to wrap; use the
last CR009 report as the lead.

## Feeding data: the data loader

The array overloads of `Fit`/`Train` take pre-batched `TensorDataStruct[]` and leave the epoch /
batch counters to you. A **data loader** owns the batch stream: it batches your data, tracks its
position, and lets `Fit` advance step / epoch / batch, so a saved checkpoint records where the run
was and a resumed run continues from the next batch.

```csharp
// One value per field of the definition, in declaration order; the leading dimension is
// the sample count.
var inputs  = rig.InputDef.FromOrderedData(TensorData([1000L, 64L], features));
var targets = rig.TargetDef.FromOrderedData(TensorData([1000L, 10L], labels));

// Batch into 32s, reshuffling each epoch (deterministically from the seed).
var loader = new InMemoryDataLoader(inputs, targets, batchSize: 32, shuffle: true, seed: 42);

var outcome = rig.Fit(loader, numEpochs: 10);   // step / epoch / batch advance automatically
```

`FromOrderedData` takes field names from the definition — the target field is named after the loss
module's **second `Inline` parameter**, so do not hard-code `"targets"`. It pairs values
positionally and throws on a count mismatch; for many same-shaped fields,
`new TensorDataStruct(def, fields)` also catches a swapped pair.

- **`IDataLoader`**: `Position` (`DataLoaderPosition`, epoch + index of the *next* batch), `Next()`
  (returns the current `DataBatch` — input, target, and its position — and advances, rolling into
  the next epoch), `RestoreFrom(position)` (next `Next()` yields the batch *at* `position`) and
  `RestoreAfter(position)` (next `Next()` yields the batch after it, rolling over epochs).
  `InMemoryDataLoader.BatchesPerEpoch` is not on the interface.
- **One step at a time.** `rig.TrainStep(checkpoint, loader)` draws one batch, trains on it (its
  position drives any scheduler) and records the **batch used**; `Fit(loader)` loops over it.
  Without a loader, `rig.TrainStep(checkpoint, input, target, epoch, batchNumber)` records the given
  counters verbatim.
- **`InMemoryDataLoader`** slices tensors you hold along the leading (sample) dimension into
  fixed-size batches, optionally reshuffling every epoch.
- **Batch feeding.** `DataBatch.Input` / `.Target` are `IData`: a `TensorDataStruct` (consumed) or
  one via `.Shared()` / `.TryConsume()`. `InMemoryDataLoader` builds a fresh batch per `Next()`; a
  custom loader that reuses its tensors passes them `.Shared()`
  ([What a step keeps of what it reads](#what-a-training-step-consumes)).
- **Shuffle is deterministic.** With `shuffle: true`, epoch `e`'s permutation is a pure function of
  `(seed, e)` (Fisher–Yates over SplitMix64; no ambient `Random` or clock), so resuming at `(e, b)`
  sees the same batches as the original run.
- **Partial final batch.** `dropLast: true` (default) drops it, keeping every batch at the compiled
  shape. `dropLast: false` keeps it (only if the graph tolerates a variable batch dimension); each
  new shape costs one compile, then is cached. Up to four shapes get their own session; further
  shapes share a shape-generic one, so compiles stay bounded.
- **Resume.** A checkpoint's `.Epoch` / `.BatchIndex` name the batch **used** at its last step. In a
  new process, rebuild the rig and a loader over the same data/seed and call
  `rig.Fit(loader, numEpochs, initialCheckpoint: loaded)`: `Fit` calls `RestoreAfter`, so training
  resumes at the next batch. A position-unknown checkpoint starts at `(0, 0)` via `RestoreFrom`.
  `numEpochs` counts from the resume epoch (a mid-epoch checkpoint first finishes that epoch; one saved at an epoch's last batch begins
  the next). For
  an external data pipeline, keep its position in the checkpoint's host user-data bag.

### The training history

Every checkpoint carries `.History`, a `TrainingHistory`: one `TrainingHistoryEntry` per successful
step that led to it, oldest first. It is empty on a checkpoint no step produced. Each step
(`TrainStep`, a resident run's `Step` or `StepToCheckpoint`, and so `Fit` and `Train`) appends one
entry; a failed step appends nothing.

```csharp
public sealed record TrainingHistoryEntry
{
    public long Step { get; init; }          // the counter the step ran at: the produced checkpoint's Step - 1
    public long? Epoch { get; init; }        // the counters it ran at; null where unknown
    public long? BatchIndex { get; init; }
    public float Loss { get; init; }
    // Same map as the produced checkpoint's .AppliedHyperparameters. Immutable.
    public IReadOnlyDictionary<string, AppliedHyperparameter> AppliedHyperparameters { get; init; }
}

public sealed class TrainingHistory : IReadOnlyList<TrainingHistoryEntry>
{
    public static TrainingHistory Empty { get; }
    public static TrainingHistory Of(IEnumerable<TrainingHistoryEntry> entries);  // in that order
    public TrainingHistory Since(long step);             // entries whose Step >= step
    public TrainingHistory TakeLast(int count);          // the last count entries
    // Columns parallel to the entries, built on first read:
    public IReadOnlyList<long> Steps { get; }
    public IReadOnlyList<float> Losses { get; }
    public IReadOnlyList<long?> Epochs { get; }
    public IReadOnlyList<long?> BatchIndices { get; }
    public IReadOnlyList<string> HyperparameterNames { get; }       // every name any entry holds, by first appearance
    public IReadOnlyList<AppliedHyperparameter?> AppliedValues(string name);  // null where an entry lacks it
}
```

```csharp
var result = rig.Fit(loader, numEpochs: 3);
var history = result.FinalCheckpoint.History;
foreach (var (step, loss, lr) in history.Steps.Zip(history.Losses, history.AppliedValues("learningRate")))
    Console.WriteLine($"{step}\t{loss}\t{lr?.ToSingle()}");
```

Entries are `init`-only records, so you can build or merge histories with `TrainingHistory.Of` and
`with`:

```csharp
var merged = TrainingHistory.Of(first.History.Concat(second.History.Select(e => e with { Step = e.Step + offset })));
var ckpt = second.WithHistory(merged);
```

The history is immutable; appending costs `O(log n)` and shares earlier entries, so branches from
one checkpoint do not affect each other. `Step` is not a key: training again from
`ckpt.WithStep(10)` adds a second step-10 entry; entries are in run order.

**Trimming and clearing.** All derivations (`WithCounters`, `WithStep`, `WithTrainableParams`, …,
`Shared()`, `rig.AdoptCheckpoint`) keep the history. `ckpt.WithHistory(ckpt.History.TakeLast(1000))`
or `.Since(5000)` keeps a slice; `WithoutHistory()` clears it. On a resident run use
`run.ReplaceHistory(run.History.TakeLast(1000))` or `run.ClearHistory()`.

**What it costs.** About 200–300 bytes per entry, more with scheduled, runtime or non-scalar
hyperparameters — a few hundred MB for a million steps — and every save writes it all. Bound it on
long runs.

**Saving it.** History is the `CheckpointComponents.History` component, written whenever non-empty
by `checkpoint.Save` and `.skpt` saves. To omit it from a flat save, pass components without it
(`ckpt.Save(path, CheckpointComponents.InferenceState | CheckpointComponents.OptimizerState |
CheckpointComponents.Counters | CheckpointComponents.Loss)`); from a `.skpt` save, save
`ckpt.WithoutHistory()`. A file
without one loads with an empty history; a resumed run continues it. Entries may hold
hyperparameters only some have (e.g. after `otherRig.AdoptCheckpoint(ckpt)`), but if entries give
one hyperparameter different dtypes or shapes the save throws `InvalidOperationException` naming it
and how many trailing entries can be saved; save
`ckpt.WithHistory(ckpt.History.TakeLast(…))`. On-disk layout:
[Save and resume a checkpoint](#save-and-resume-a-checkpoint-across-process-restarts).

## Save and resume a checkpoint (across process restarts)

A `TrainingCheckpoint` holds trainable params, model state, optimizer state, and the run counters
(step, epoch, batch index):

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

- **Flat file layout.** One SafeTensors file with every param/state field. An `int64` marker holds
  `[version, step]`; epoch and batch index are separate `int64` scalars written only when set, so
  unknown counters reload as `null` and a concrete `0` as `0`.
- **History layout.** A non-empty history is the `history/` section, one tensor per column:
  `history/step` (`int64[n]`), `history/loss` (`float32[n]`), `history/epoch` and
  `history/batch_index` (`int64[n]`) with presence columns `history/epoch_present` /
  `history/batch_index_present` (`bool[n]`; `false` reloads as `null`), and per hyperparameter
  `history/hyperparameter/<name>` (its dtype, shape `[n, …valueShape]`) with
  `history/hyperparameter_present/<name>` (`bool[n]`). A missing or unknown column, or one of the
  wrong dtype, rank or length, is refused on load. A `.skpt` stores the same columns in
  `data/history.safetensors`.
- **Saves are atomic.** `checkpoint.Save` (and `Persistence.SaveTrainingCheckpoint`) writes a
  `.tmp-` sibling, flushes it, and renames it into place, so a crash mid-save leaves the old or the
  new file, never a truncated one. The target **directory must exist** (it is not created); a
  leftover `.tmp-` sibling is swept by the next successful save. `.skpt` saves are atomic too — see
  [skpt-checkpoints.md](skpt-checkpoints.md#the-directory-form) for the directory form's one
  exception.
- **`.skpt` container.** `Persistence.SaveTrainingCheckpointToSkpt(checkpoint, "run.skpt")` (or the
  `Persistence.ForTrainingCheckpoint(...)` builder) writes the native container, taking the model
  from the checkpoint's `.Rig`. Resume with `rig.LoadCheckpointFromSkpt("run.skpt")`, or with no
  graphs in hand, `var (rig, ckpt) = TrainingRig.Load("run.skpt")`, which rebuilds the rig from the
  file. `Persistence.Load` and `Persistence.LoadEvaluationModel` read the model from the same file
  without a rig. Layout: [skpt-checkpoints.md](skpt-checkpoints.md#training-checkpoints).
- **One loader per format.** `rig.LoadCheckpoint` reads only flat safetensors;
  `rig.LoadCheckpointFromSkpt` and `TrainingRig.Load` only `.skpt`. The wrong format fails with an
  error naming the right entry point. `Persistence.Inspect` identifies an unknown file.
- **Validation.** `LoadCheckpoint` / `LoadCheckpointFromSkpt` need a rig built from the **same**
  model/loss/optimizer graphs. Field names, dtypes and **dimensions** are checked on load, and each
  value is checked against the model's declared shape when bound, so a hand-built checkpoint of the
  wrong shape is refused too. Provenance is not checked: right-shaped weights load into a different
  model. Values match parameters by name, so names from a local or `.Named(...)` are stable
  ([Parameter names](defining-models.md#parameter-names)); parameters left to class names are
  numbered in creation order, and two same-shaped ones whose order was swapped load into each
  other's places silently.
- Because `.Step` is restored, **schedules resume from the right step**.
- `rig.LoadCheckpoint(path)` delegates to `TrainingCheckpoint.Load(path, rig)` (and
  `rig.LoadCheckpointFromSkpt(path)` to `TrainingCheckpoint.LoadFromSkpt(path, rig)`), which sets
  `.Rig`. Without a rig, `Persistence.LoadTrainingCheckpoint(path)` reads a flat checkpoint (it is
  self-describing) but validates nothing and sets no `.Rig`; pass it to `rig.AdoptCheckpoint(ckpt)`
  to validate it.
- **`CheckpointComponents`.** Save and load take optional flags — `InferenceState` (trainable params
  + model state), `OptimizerState`, `Counters`, `Loss`, `History`, `TrainingRig` — combined with
  `|`. `null` saves every available component and loads everything present (absent components come
  from the rig's initial values; absent history is empty).
  `checkpoint.Save(path, CheckpointComponents.InferenceState)` saves weights only. Requesting `Loss`
  when it is `null`, or `History` when empty, writes nothing and does not throw. `TrainingRig` (the
  rig's graphs, hyperparameter bindings and RNG config) is always written to a `.skpt` and read by
  `TrainingRig.Load`; never name it yourself. Requesting it (including via
  `CheckpointComponents.All`) throws on the flat `checkpoint.Save` and on `rig.LoadCheckpoint` /
  `rig.LoadCheckpointFromSkpt`; omit it, or pass `null`.
- `rig.AdoptCheckpoint(checkpoint)` returns the checkpoint bound to that rig, after validating the
  field defs, enabling `ToInferenceModel()`.
- `Persistence.Inspect(path)` shows a file's counters and per-section tensor listing without
  loading it — see
  [onnx-and-weights.md](onnx-and-weights.md#identify-and-summarize-a-file-persistenceinspect).

### What a save costs

Every single-file checkpoint save returns a `SaveReport`:

```csharp
var save = checkpoint.Save("run.safetensors");
Console.WriteLine(save);
// 200,000,077 bytes in 0.252s (757 MiB/s): write 0.051s, flush 0.194s, commit 0.007s
```

`Write` serializes into the staged file (for `.skpt`, also compressing and hashing), `Flush` is the
fsync, `Commit` the rename plus sweeping stale staged files. They sum to `Elapsed`; `BytesWritten`
is the file size and `BytesPerSecond` the achieved rate. `Persistence.SaveTrainingCheckpoint`,
`Persistence.SaveTrainingCheckpointToSkpt` and the `Persistence.ForTrainingCheckpoint(...)`
builder's `Save` return it too; the directory form (`SaveAsDirectory`) returns `void`. Save time
varies between identical saves (mostly in `Flush`), and multi-GB saves can take tens of seconds, so
exclude it from throughput measurements:

```csharp
steady.Stop();                                   // saving is I/O, not training
var save = checkpoint.Save(path);
steady.Start();
savedBytes += save.BytesWritten;
```

The flat safetensors save streams each tensor from its storage with no extra copy. The `.skpt` save
holds a full serialized copy of the training state in memory before writing. Neither can exceed the
safetensors 2 GB ceiling: a larger checkpoint is written without complaint and cannot be read back
([#48](https://github.com/Shorokoo/Shorokoo/issues/48)).

### Bind trained weights into an inference model

```csharp
var concrete = result.FinalCheckpoint.ToInferenceModel();   // no graph to re-supply
var output   = ComputeContext.Default.Execute(concrete, myInput);
```

A saved training `.skpt` needs no rig: `Persistence.Load(path)` returns the runnable model and
`Persistence.LoadEvaluationModel(path)` the model composed with its loss, for validation. Use
`TrainingRig.Load(path)` only to keep training. A flat safetensors file has no architecture; a
checkpoint from `Persistence.LoadTrainingCheckpoint` needs a rig to bind.

```csharp
var model = Persistence.Load("run.skpt");                  // ConcreteModel, weights bound
var eval  = Persistence.LoadEvaluationModel("run.skpt");   // [model inputs…, targets] → loss
```

`ToInferenceModel()` binds the checkpoint's trainable params and model state into its `.Rig`'s
retained concrete architecture (concretized once at build, at all inputs, so multi-input models
work) — no re-concretization or sample inputs. It requires an attached rig; use
`rig.AdoptCheckpoint(checkpoint)` for a bare checkpoint.

## Types used by the training API

These are in namespace `Shorokoo` (covered by `using Shorokoo;`), except `Schedule` and `Schedules`, which are in `Shorokoo.Core.Training` and need `using Shorokoo.Core.Training;`:

| Type | Role | How to make one |
|---|---|---|
| `NamedModelParam` (abstract) | A named parameter value. | Use the concrete `TensorDataModelParam`. |
| `TensorDataModelParam` | Concrete `NamedModelParam` wrapping one `TensorData`. | `new TensorDataModelParam(name, ModelParamType.InputParam, tensorData)` |
| `ModelParamType` (enum) | Tags a param's role. | `Undefined`, `HyperParam`, `TrainableParam`, `InputParam`, `OutputParam` |
| `ModelParamList` | A set of named params (e.g. loaded weights). | `new ModelParamList(IEnumerable<(string name, TensorData data)>)` |
| `TensorDataStruct` | Named `TensorData` fields; the form `Train`/`TrainStep` take for inputs/targets. | `new TensorDataStruct(structDef, fields)` — `structDef` a `TensorStructDef` (namespace `Shorokoo.Core`), `fields` `KeyValuePair<string, IData>`, one per definition field of the declared kind (a mismatch throws), optionally via `.Shared()` / `.TryConsume()`. Read: `.Fields` (name → value, in definition order), `.Count`, `[int]`. |
| `SharedInput` | A value to be **read** rather than consumed (`Mode` `Shared`), or consumed only if nothing else reads it (`TryConsume`). | `x.Shared()` / `x.TryConsume()` on a `TensorData`, `TensorDataStruct`, `TensorDataSequence` or `OptionalTensorData`. On a checkpoint these return a checkpoint with that `FeedMode`; on a `NamedModelParam`, a copy with its `FeedMode` set. |
| `SaveReport` | A save's `BytesWritten`, `Write` / `Flush` / `Commit`, `Elapsed`, `BytesPerSecond`. | Returned by every single-file checkpoint save — see [What a save costs](#what-a-save-costs). |
| `Schedule` (namespace `Shorokoo.Core.Training`) | A `step → value` schedule; assigning one makes a hyperparameter [`Scheduled`](#hyperparameter-kinds-hyperparameter). | A `Schedules.…` factory plus combinators (`WithWarmup`, `Then`, `Scale`, `Clamp`, `Shift`, `PerEpoch`). Preview with `.At(step)`. |
| `Schedules` (static, namespace `Shorokoo.Core.Training`) | Factories: `Constant`, `Linear`, `Cosine`, `CosineWithWarmup`, `StepDecay`, `Exponential`, `OneCycle`. | `Schedules.Cosine(1e-3f, totalSteps)` — see [Schedule factories and combinators](#schedule-factories-and-combinators). |

### Sample inputs

`sampleInputs` gives `FromScratch` one sample per model data input (its shape and the values
concretization reads), in one of two forms:

- **Positional**: an `IData[]` (`TensorData`, or `OptionalTensorData` for an optional input) in
  input declaration order: `FromScratch(model, loss, opt, [TensorData([4L, 64L], new float[256])], hypers)`.
- **Named**: a `NamedModelParam[]` or `ModelParamList`, in any order:
  `FromScratch(model, loss, opt, [new TensorDataModelParam("input",
  ModelParamType.InputParam, x)], hypers)`.

Wrong count, an unnamed input, an unknown or duplicate name, or a rank contradicting the input's
declared rank is refused with `FW056`, naming the offenders and listing the model's inputs.

`TrainStep` consumes its `TensorDataStruct` batches unless passed `.Shared()`; `Train` and array
`Fit` read theirs ([What a training step consumes](#what-a-training-step-consumes)).

## Workflow: train a model

1. Define model, loss, and optimizer as `[Module]` classes (or reuse built-ins).
   For several sizes of one model, give it `[Hyper]` parameters and `Specialize` the graph per
   variant rather than writing a class each; `FromScratch` takes the specialized graph. See
   [Workflow: one module, many variants](defining-models.md#workflow-one-module-many-variants).
2. Build the rig with the optimizer's named hyperparameter set (a bare `float` bakes a constant;
   a `Schedule` makes it live):
   ```csharp
   var rig = TrainingRig.FromScratch(
       MyModel.ComputationGraph,
       L2Loss.ComputationGraph,
       SGDMomentumOptimizer.ComputationGraph,
       [TensorData([4L, 64L], new float[256])],      // one sample per model input, in order
       new SGDMomentumOptimizerHyperparameters {
           LearningRate  = Schedules.CosineWithWarmup(0.5f, warmupSteps: 100, totalSteps: 1000),
           MomentumCoeff = 0.9f,          // baked constant
       });
   ```
3. Initialize parameters: `var ckpt = rig.CreateInitialCheckpoint();`.
4. Run epochs: `var outcome = rig.Fit(inputs, targets, numEpochs: 10);`
   The schedule advances with the global step. (Or call `rig.TrainStep(...)` per batch, with
   `rig.MakeHyperparameters(...)` for runtime values.)
5. Read `outcome.EpochLosses` for the loss curve and `outcome.FinalCheckpoint.TrainableParams`
   (a `TensorDataStruct`) for the weights, via `.Fields`:
   ```csharp
   foreach (var (name, value) in outcome.FinalCheckpoint.TrainableParams.Fields)
   {
       var data = (TensorData)value;   // shape via data.Shape.Dims; values via data.CopyMemory<float>()
   }
   ```

## Custom optimizers

A custom optimizer is a `[Module]` whose `Inline` takes exactly `(currentParam, grad)`, then its
`[Hyper]` hyperparameters, and returns the updated parameter. Each piece of state is created
**inside the body** by an optimizer-owned `[StateInitializer]`'s `Init` — typically
`OptimizerStateZeros.Init(currentParam.ShapeTensor())` from `Shorokoo.Modules.Optimizers` — and
updated with exactly one `Globals.StateUpdate(state, newValue)`. For example, a momentum-less
RMSprop (the full one is `RMSpropOptimizer` in [Shorokoo.Modules](nn-library.md)):

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
  graph is rank-agnostic, so it cannot factor (see [nn-library.md](nn-library.md)). `learningRate`
  is the **cap** on the relative step.
- Prefer the generated named set. The positional `params Hyperparameter[]` overload must match the
  count exactly: SGD=1, SGDMomentum=2, Adam=4, RMSprop=4, AdamW=5, Adagrad=2, Adamax=4, NAdam=5,
  RAdam=4, Adadelta=3, Lion=4, Adafactor=6, Lamb=5.
- Optimizer state per trainable parameter (table in [nn-library.md](nn-library.md)):
  momentum: velocity; Adam/AdamW: `m`/`v` + scalar `step`; RMSprop: `squareAvg`/`momentumBuffer`;
  Adagrad: `accumulator`; Adamax: `m`/`u` + scalar `step`; NAdam: `m`/`v` + scalars `step`,
  `muProduct`; RAdam: `m`/`v` + scalar `step`; Adadelta: `squareAvg`/`accDelta`; Lion: `m` only
  (half of Adam); Lamb: `m`/`v` + scalar `step` (trust ratio not stored); Adafactor: full `v` +
  scalar `step`. Param-shaped fields come from `OptimizerStateZeros`; scalar ones from
  `OptimizerScalarZeros` (seeded 0) or `OptimizerScalarOnes` (seeded 1, NAdam's product).

## Anti-patterns

- Do not mismatch the positional `params Hyperparameter[]` count; use the named set.
- Do not call the schedule-driven `TrainStep` on a rig with a `Hyperparameter.Runtime`
  hyperparameter; use `MakeHyperparameters` and the override overload.
- Do not implement backward passes manually.
- Do not mutate a `TrainingCheckpoint` across steps; thread the returned one forward.
- Do not feed a batch or checkpoint you will reuse as it is — the step consumes it and the next
  read throws. Pass it `.Shared()`.
- Do not run a long GPU loop on `TrainStep` when you want only the last checkpoint; use
  `rig.BeginResidentRun()`, `Fit` or `Train`.
- Do not expect a resident run's state after disposing it; take it with `StepToCheckpoint` first.
- Do not declare optimizer state as `Inline` parameters.
- Do not call `Globals.StateUpdate` on inputs, trainable parameters, or computed tensors — only on
  a `[StateInitializer]` `Init` result.
- Do not call `Globals.StateUpdate` outside a module body — it throws. Inside a `LoopAPI.Iterate`
  body it registers the updated tensor's post-loop value (that state's one update for the step);
  the value must be a carried loop variable (assigned in the body and read across iterations; with
  zero iterations it is its pre-loop value). A value that never leaves the loop, a scanned result,
  or an iteration-scoped value (e.g. the iteration index) fails the module build.
