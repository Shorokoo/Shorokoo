# Training models

Related: [defining-models.md](defining-models.md) · [nn-library.md](nn-library.md) · [inference.md](inference.md) · [training-backends.md](training-backends.md)

## Facts

- Training composes three graphs: a **model**, a **loss**, and an **optimizer**, all
  `[Module]` classes accessed via their `.ComputationGraph` property.
- `TrainingRig` is the entry point. It runs autodiff on the composed graph and
  produces a trainable step.
- Gradients are produced by automatic differentiation; you do not write backward
  passes. Shorokoo differentiates the step itself by default; a rig built with
  `trainingBackend: TrainingBackend.Native` leaves the gradient to the backend that runs the
  step instead — see [training-backends.md](training-backends.md).
- `TrainStep` moves the whole training state through host memory every step. On a GPU that is what
  sets the pace, so a long run belongs in a `rig.BeginResidentRun()` loop (or in `Fit` / `Train`,
  which already use one) — see [Keeping training state on the device](#keeping-training-state-on-the-device).
- A training step **consumes** what it is fed as it is — the checkpoint's state and the batch — as
  any run does, so `cp = rig.TrainStep(cp, x, y)` releases the state it supersedes as the step runs.
  Feed `cp.Shared()` to keep a checkpoint past the step, and `x.Shared()` for a batch you feed
  again. `CreateInitialCheckpoint()` hands out fresh copies every call, consumed like any other
  checkpoint — [What a training step consumes](#what-a-training-step-consumes).
- A step writes its updated state over the state it consumed wherever the step's graph proves
  nothing still reads the old value, so on a card a resident run holds its state once rather than
  twice — [A step writes its state over the state it consumed](#a-step-writes-its-state-over-the-state-it-consumed).
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
data-flow one: rig build checks the counts — exactly two inputs, exactly one output — then wires
the model's output to input 0, creates a fresh runtime input for input 1, and replays the loss body.
Nothing requires input 1 to be read; the build asks whether it *is* (a reachability walk from the loss
output, which follows branch conditions as well as data edges) and derives the target slot from the
answer. So a model that computes its **own** loss can be trained with a
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

**You do not feed the ignored input.** The rig asks whether the loss body actually reaches its second
input, and when it does not, derives no target field for it: `rig.TargetDef` is empty,
`rig.HasTargets` is `false`, and the target-free `TrainStep` / `Fit` overloads take the model inputs
alone ([#331](https://github.com/Shorokoo/Shorokoo/issues/331)).

```csharp
ckpt = rig.TrainStep(ckpt, inputs);              // the real labels ride inside `inputs`
var result = rig.Fit(batches, numEpochs: 10);
```

The dead input does survive into the compiled trainstep — the step's input layout is positional, so
the slot stays — but it is the rig that supplies its value, not the call site, which is the whole
point: `TrainStep(ckpt, inputs, noTargets)` read as "inputs and targets" while the real targets rode
inside `inputs` and the third argument was a fabricated zero-element tensor. Calling a target-free
overload on a rig whose loss *does* read its target fails loud rather than training against something
you never chose.

The target-free overloads are the counter-agnostic `TrainStep(checkpoint, inputs)`, its
explicit-counter form `TrainStep(checkpoint, inputs, epoch, batchNumber)`, `Fit(inputs, numEpochs)`,
and the resident run's `Step(inputs)` / `StepToCheckpoint(inputs)`. The loader path takes a target
*dataset* rather than a per-step target, and an empty one is what a target-free rig gives it:

```csharp
var loader = new InMemoryDataLoader(inputs, rig.TargetDef.FromOrderedData(), batchSize: 32);
var result = rig.Fit(loader, numEpochs: 10);
```

The only entry points still needing an explicit target argument are those whose signature cannot drop
one unambiguously — `TrainStep(checkpoint, hyperparams, …)` and the resident run's
hyperparameter forms. There too the argument is `rig.TargetDef.FromOrderedData()`: an **empty** struct
contributing no field, not a placeholder tensor you had to invent.

Two things this shape does **not** change. The predictions tensor is never an output of the training
step — the step's outputs are the updated parameters, model state, optimizer state and the loss — so
composing the loss into the model does not save the memory of a large logit tensor: the loss body is
inlined into the same graph either way, and the composed training step does the same work (op for op)
whichever side of the seam the loss sits on. And `ExtractInferenceModel` hands back the model **as
authored**, so a loss-computing model yields an inference model that returns a loss and demands the
labels; author the prediction path as its own module if you also need one.

Folding the loss into the model is not the way to get a *validation loss* out of a checkpoint: `Persistence.LoadEvaluationModel(path)` composes a saved checkpoint's model with its loss
and needs no rig ([#329](https://github.com/Shorokoo/Shorokoo/issues/329)). Fold the loss in when the
loss genuinely needs more inputs than the slot carries — that is what this shape is for.

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

Both types live in namespace `Shorokoo.Core.Training`, which `using Shorokoo;` does **not** cover —
add `using Shorokoo.Core.Training;` to any file that names `Schedule` or `Schedules`. It is supported
public surface, not an assembly-layout artefact — see
[`Shorokoo.Core.*` is not all internal](orientation.md#public-core-namespaces).

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
| `CosineWithWarmup(float baseValue, int warmupSteps, int totalSteps)` | `Cosine(baseValue, max(1, totalSteps - warmupSteps)).WithWarmup(warmupSteps)`, with both arguments clamped (a negative `warmupSteps` becomes `0`, the cosine's length is never below `1`) — so a linear ramp over the first `warmupSteps` that, per `WithWarmup` below, starts at `0` and reaches `baseValue` at step `warmupSteps`, then a cosine decay reaching `0` at step `totalSteps` | held at `0` afterwards |
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
| `WithWarmup(int warmupSteps, float startFactor = 0f)` | with `peak = f(0)` captured when the combinator is called: for `s < warmupSteps`, `peak · (startFactor + (1 - startFactor) · s / warmupSteps)`; for `s ≥ warmupSteps`, `f(s - warmupSteps)` — the inner schedule is **re-based** to start after the warmup. `warmupSteps == 0` returns the schedule unchanged |
| `Then(int atStep, Schedule next)` | `f(s)` for `s < atStep`, and `next(s - atStep)` for `s ≥ atStep` — `next` is **re-based**, i.e. evaluated at the step *relative* to `atStep`, never at the absolute step |

One consequence of `WithWarmup`'s exact form is worth spelling out: `startFactor` multiplies the
**inner schedule's step-0 value** (`peak`), not the optimizer's declared default. The ramp's endpoints
are otherwise the ones a published recipe states — the ramp is denominated in `warmupSteps` and linear
in `s`, so step 0 is exactly `startFactor · peak` (a true `0` at the default `startFactor = 0`), and
`peak` first arrives at step `warmupSteps`, where the re-based inner schedule contributes its own step
0. That value *is* `peak` by definition, so the ramp and the inner schedule meet continuously: writing
`Schedules.Constant(peak).WithWarmup(N)` gives a literal linear warm-up from `0` to `peak` over steps
`0 … N`.

**Worked example: warm up, hold, decay.** Because `Then` re-bases, the second schedule's length is
stated in its *own* steps and the boundary is stated in absolute steps — the two are independent:

```csharp
using Shorokoo.Core.Training;   // Schedule, Schedules

// Ramp 0 → 1e-3 over steps 0..200, hold 1e-3 to step 3899,
// then decay 1e-3 → 5e-5 over steps 3900..6000 and hold.
Schedule lr = Schedules.Constant(1e-3f)
    .WithWarmup(200)                                    // peak = Constant's step-0 value = 1e-3
    .Then(3900, Schedules.Linear(1e-3f, 5e-5f, 2100));  // Linear's own step 0 is global step 3900
```

| step | `lr.At(step)` | why |
|---|---|---|
| `0` | `0` | ramp: `1e-3 · 0/200` |
| `100` | `5.0e-4` | ramp: `1e-3 · 100/200` |
| `199` | `9.95e-4` | ramp: `1e-3 · 199/200` — one step short of the peak |
| `200` … `3899` | `1e-3` | the `Constant` inner schedule, re-based past the warmup — its step 0 *is* the peak, so the ramp arrives there exactly |
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

A runtime hyperparameter states its shape because the training step is compiled ahead of the values, so the shape has
to be known at build even though the values are not. That also makes the shape fixed for the rig's
life: a per-step value whose shape differs fails loud rather than silently reshaping.

A non-scalar hyperparameter also has to fit the parameters it will be applied to: the update is
tensor arithmetic, so a vector rate against a scalar weight broadcasts rather than scales, and the
rig refuses that at build ([Custom optimizers](#custom-optimizers)).

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

ckpt = rig.TrainStep(ckpt,
    rig.MakeHyperparameters(("perGroupScale", TensorData([3L], 1f, 2f, 3f))),
    inputs.Shared(), targets.Shared());
```

`MakeHyperparameters` builds a struct whose tensors are its own: a `TensorData` you give it is
copied, whatever its dtype, so the step that consumes the struct takes nothing of yours.

The positional-hyperparameter `FromScratch` overloads take the hyperparameter values as an array in
the hyperparameter slot, followed by the optional `rngConfig` / `mergeContext` / `runtimeContext`:
`(sample, [0.05f], rng)`, `(sample, [0.05f], rng, merge, runtime)`, `(sample, [], rng)`. The bare
`FromScratch(model, loss, opt, sample, 0.05f)` params form takes the values alone. To pass a context
without an `rngConfig`, name it (`(sample, [0.05f], mergeContext: ctx)`); a literal `null` in the
hyperparameter slot alongside an optional argument is ambiguous between the named-set and array overloads,
so pass the set or cast.

A `Hyperparameter` is a `Baked` / `Scheduled` / `Runtime` union. `Hyperparameter.Baked(v)` bakes a
constant (a bare `float` converts implicitly), and its `BakedValue` is the `TensorData` it was built from,
shape and dtype included (`BakedDType` alongside). `Hyperparameter.Runtime()` is fed per step through
`MakeHyperparameters`, whose named overload takes `(string name, object value)` pairs; the dynamic
hyperparameter names are listed by `DynamicHyperparameterNames`. `HyperAttribute.DefaultValue` is the host
literal the attribute was given (`object?`), and a graph input's `HyperDefaultValue` is that default's
invariant literal (`string?`), so an `int64` / `float64` / `bool` default round-trips exactly. In a training
`.skpt` each baked binding records its own `dtype`, `shape` and base64 `value`, and a runtime binding
records its `shape`. Creating a fresh checkpoint can fail loud (see `CreateInitialCheckpoint` below).

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
    ComputeContext? runtimeContext = null);   // compile/run context (rig.RuntimeContext); null ⇒ Default

// Lower-level: positional values (a float bakes a constant, a Schedule schedules it). The params form
// takes the values and nothing else; supplying an rngConfig or either context selects the array form,
// which takes them in the same order and the same slots as the named-set overload above:
//   FromScratch(model, loss, opt, sampleInputs, params Hyperparameter[] hyperparameters)
//   FromScratch(model, loss, opt, sampleInputs, Hyperparameter[] hyperparameters,
//               RngConfig? rngConfig = null,
//               ComputeContext? mergeContext = null, ComputeContext? runtimeContext = null)
// Each of the three forms above binds its samples by position. Each also has twins binding them by
// name, taking a NamedModelParam[] or a ModelParamList for sampleInputs: each sample goes to the
// model input of its name, in any order (see "Sample inputs" below).

// Fresh initial checkpoint: host copies of the rig's initial values, new every call, so a step
// consumes it like any other checkpoint and the rig keeps its own values for the next one. Optimizer
// state is initialized at each hyperparameter's value at the initial counters. Fails loud if the
// optimizer's state initializer reads a Runtime hyper (its value is unknown at build) — supply
// explicit values with the overload below.
public TrainingCheckpoint CreateInitialCheckpoint();
public TrainingCheckpoint CreateInitialCheckpoint(TensorDataStruct hyperparameters); // from MakeHyperparameters(...)

// Schedule-driven: scheduled hyperparameters are computed in-graph from the checkpoint's
// step (fed as the step counter), then the step advances. Requires no schedule-less runtime hypers.
// Returns the post-step checkpoint directly, with its .Loss set to this step's loss. The rig
// compiles its training-step graph internally (lazily, cached per fed shape), so a manual loop is just
// `cp = rig.TrainStep(cp, in, out);` — no caller-side ComputeContext.Compile.
// Each struct argument is a TensorDataStruct — consumed by the step — or one passed through
// .Shared() or .TryConsume(); the checkpoint is fed as its FeedMode says (see "What a training step
// consumes"). Anything else is refused with an ArgumentException.
public TrainingCheckpoint TrainStep(
    TrainingCheckpoint checkpoint,
    IData trainingInput,
    IData trainingOutput);

// Explicit override: supply the schedule-less runtime hyperparameter values for this step.
public TrainingCheckpoint TrainStep(
    TrainingCheckpoint checkpoint,
    IData hyperparams,                         // from MakeHyperparameters(...)
    IData trainingInput,
    IData trainingOutput);

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
    IData trainingInput,
    IData trainingOutput,
    long epoch,
    long batchNumber);

public TensorDataStruct MakeHyperparameters(float value);                       // exactly one dynamic
//   also: (double), (int), (long), (bool), and (TensorData) for other dtypes / non-scalar shapes
public TensorDataStruct MakeHyperparameters(params (string name, object value)[] values); // named

// Array-driven: one array element per training step (typically a pre-batched batch). The checkpoint
// comes LAST and is optional, so the minimal call is `rig.Fit(inputs, targets, numEpochs: 10)`.
// The arrays are a dataset fed every epoch, so each step reads its batch and they all survive the
// call; the checkpoint is fed to the first step as TrainStep feeds one.
public TrainingResult Fit(
    TensorDataStruct[] trainingInputs,
    TensorDataStruct[] trainingOutputs,
    int numEpochs,
    TrainingCheckpoint? initialCheckpoint = null); // defaults to CreateInitialCheckpoint()
                                                   // compiles/runs via rig.RuntimeContext (cached per fed shape)

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

// A training loop that keeps its state where the execution provider produced it, instead of moving
// the whole of it through host memory on every step — see "Keeping training state on the device".
// The initial checkpoint is fed to the first step as TrainStep feeds one: consumed as it is, read
// when passed .Shared(). The default is a fresh CreateInitialCheckpoint(), which the first step
// consumes.
public ResidentTrainingRun BeginResidentRun(TrainingCheckpoint? initialCheckpoint = null);
```

```csharp
public sealed class ResidentTrainingRun : IDisposable
{
    // Train on one batch; returns that step's loss and nothing else, so nothing is downloaded.
    // The batch is fed as TrainStep feeds one: consumed as it is, read when passed .Shared().
    public float Step(IData trainingInput, IData trainingTarget);
    public float Step(IData hyperparameters,                            // from MakeHyperparameters(...)
                      IData trainingInput, IData trainingTarget);
    public float Step(IDataLoader loader);                              // draws loader.Next()
    public float Step(DataBatch batch);                                 // a batch you drew yourself

    // The same step, with the updated state brought back to the host as an ordinary checkpoint you
    // can read, save and resume from. This is the step that pays for the transfer.
    public TrainingCheckpoint StepToCheckpoint(IData trainingInput, IData trainingTarget);
    public TrainingCheckpoint StepToCheckpoint(IData hyperparameters,
                                               IData trainingInput, IData trainingTarget);
    public TrainingCheckpoint StepToCheckpoint(IDataLoader loader);
    public TrainingCheckpoint StepToCheckpoint(DataBatch batch);

    public long CurrentStep { get; }   // the Step the run's last checkpoint carries; next is +1

    public void Dispose();             // releases state the run still holds; published checkpoints survive
}
```

### What a training step consumes

A training step is a run, and feeds its inputs the way every run does
([inference.md](inference.md#feeding-a-run-consumed-shared-or-tried)): what it is given as it is,
it **consumes** — the tensors are dead once the step starts, and their memory goes back as the
step returns — and what it is given `.Shared()` it only reads.

```csharp
cp = rig.TrainStep(cp, x.Shared(), y.Shared());    // a batch fed again: read, and kept
cp = rig.TrainStep(cp, x, y);                      // cp's state, x and y are consumed
var next = rig.TrainStep(best.Shared(), x2, y2);   // a checkpoint kept past the step: read
```

That is what the ordinary loop wants: the state a step supersedes is released as the step runs,
rather than whenever the old checkpoint is collected.

**What a step keeps of what it reads.** A run reads a tensor it cannot address where it is — every
tensor built from a C# array, on any backend, and a host tensor on a card — through a copy in its
own memory, which the tensor holds and reuses for its next read until it is written or dies
([inference.md](inference.md#feeding-a-run-consumed-shared-or-tried)). A training step lets go of
the copies it made to read its **batch** as it returns, whether it succeeded or failed: a dataset
fed `.Shared()` epoch after epoch is not held a second time in the run's memory — on a card, not
copied onto the card whole — and each step that reads a batch copies it afresh. It keeps the copies
it made to read the **checkpoint's state**, which is read step after step: a checkpoint fed
`.Shared()` to a step on a card keeps a copy of its state on the card for as long as the checkpoint
lives. Drop a kept checkpoint once you are done with it rather than holding it past its use. A
resident run is the exception: a checkpoint it does not own — the one you began it from
`.Shared()`, or one `StepToCheckpoint` handed you — is read by one step only, the step that moves
the run on to state of its own, and that step lets its copies go as it returns.

- **A checkpoint** feeds its trainable parameters, model state and optimizer state as its
  `FeedMode` says. `null` — every checkpoint a step or a load hands you — is as it is, consumed.
  `cp.Shared()` returns a checkpoint over the same tensors to be read instead, and
  `cp.TryConsume()` one to be consumed only where nothing else is reading it; the derivations (`WithStep`, `WithCounters`,
  `WithTrainableParams`, …) and `rig.AdoptCheckpoint` carry the mode through, since they share its
  tensors. Reading a consumed checkpoint's state throws, naming the training step that took it and
  the section it fed ("the checkpoint's trainable parameter …").
- **An initial checkpoint is a copy.** `CreateInitialCheckpoint()` copies the rig's initial values
  into tensors of the checkpoint's own on every call, and so does a load that falls back on the
  rig for a component its file omits. A step consumes one like any other checkpoint, which takes
  nothing of the rig's: the rig's own values are never fed to a run at all, and the next initial
  checkpoint is whole. To start several steps or runs from one initial state, call it once for
  each, or pass one checkpoint `.Shared()`. The step counters the rig builds for a step are its
  own, and consumed.

  ```csharp
  var start = rig.CreateInitialCheckpoint();
  var a = rig.TrainStep(start.Shared(), x.Shared(), y.Shared());   // start is kept
  var b = rig.TrainStep(start, x, y);                              // start is consumed
  ```
- **Batches** are the caller's, fed as passed. `TrainStep`, a resident run's `Step` and a
  `DataBatch` take a `TensorDataStruct` or one passed through `.Shared()` / `.TryConsume()`;
  anything else is refused with an `ArgumentException` naming the struct definitions to build it
  from (`rig.InputDef.FromOrderedData(...)`).
- **A field can be fed its own way.** Build a struct with a field passed `.Shared()` or
  `.TryConsume()` and that field is fed so rather than as the struct is:
  `rig.InputDef.FromOrderedData(tokens, mask.Shared())` keeps the mask while the step consumes the
  tokens. A field given `.Shared()` is read however the struct is fed, a struct fed `.Shared()` has
  every field read, and otherwise each field is fed as it was given, or as the struct is. The struct
  holds the tensor itself — `Fields` and the indexer read `mask`, not a wrapper — and its `To`,
  `CopyTo` and `ToHost` keep each field's mode, as does a checkpoint's state through
  `rig.AdoptCheckpoint`.
- **Runtime hyperparameters** are fed like a batch: the struct is consumed as it is, and read
  through `.Shared()`. `MakeHyperparameters` builds a fresh one per call and copies any tensor it is
  given, so a step never consumes a tensor you passed it.
- **`Fit` and `Train` over arrays read their batches** — the arrays are a dataset, fed every epoch
  — and feed the initial checkpoint to the first step as `TrainStep` would. `Fit` over a loader
  feeds each batch as the loader built it: `InMemoryDataLoader` gathers a fresh batch per draw,
  which the step consumes, and a loader of your own that hands out tensors it keeps passes them
  `.Shared()` in its `DataBatch`. Either way each step lets go of the copies it made to read its
  batch, as above, so a dataset read `.Shared()` costs a copy of each batch per step, not a second
  copy of the whole dataset for as long as it lives.
- **A step that fails after it started has still consumed what it was fed as it is**, as any run
  has. A resident run notes when that took its own state, and then refuses every later step,
  saying so — see [below](#keeping-training-state-on-the-device).

### Keeping training state on the device

`TrainStep` is checkpoint-in / checkpoint-out: it hands back a host-readable copy of every
trainable parameter, all model state and all optimizer state, and takes them all back on the next
call. On a CPU backend that costs nothing — there is one memory. On a GPU it is the whole training
state crossing the bus twice per step, so **step time tracks parameter count rather than
arithmetic**: on one measured pair of transformers, tripling the parameters while slightly
*reducing* the FLOPs more than doubled the time per step.

`BeginResidentRun` is the loop that does not do that. The state stays where the execution provider
produced it, and each step feeds the previous step's values straight back:

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

Read it as a cost model:

- **`Step` returns the loss and nothing else.** The loss is a scalar, so it always comes back; the
  state does not.
- **`StepToCheckpoint` runs a step** — it is not a "fetch the state" call, so it replaces that
  step's `Step` rather than following it — **and brings the state home**, as an ordinary
  `TrainingCheckpoint` — save it, resume from it, extract an inference model from it. Use it on the
  steps you actually want a checkpoint at, including the last step whose state you want to keep.
- **`Dispose` discards whatever the run still holds.** A checkpoint the run already published stays
  valid: the run gives up the right to free that state when it hands it to you, and only reads it
  from then on.
- **Each step consumes the state the run's last step produced**, which is how a resident run
  releases state as it is superseded — and writes the new state over it where it can, see
  [below](#a-step-writes-its-state-over-the-state-it-consumed). The checkpoint you begin from is fed
  to the first step as you passed it — consumed as it is, read and left yours when passed
  `.Shared()` — and the default, a fresh `CreateInitialCheckpoint()`, is consumed by the first
  step like any other.
- **A step that fails can take the run with it.** A step takes the state it trains from when it
  starts, so one that fails after consuming the run's own state leaves nothing to train from: every
  later step throws `InvalidOperationException` saying so, and what is left to begin again from.
  Begin a new run from the last checkpoint you took with `StepToCheckpoint` — a step that fails
  while the run is reading a published one leaves it, and the run, whole. Before the run has handed
  out any, begin from a checkpoint you still hold: the one the run began from survives only if you
  passed it `.Shared()`, since its first step consumes it otherwise.
- **State taken by something else is not the run's loss.** A checkpoint the run published shares
  its tensors with the state the run goes on training from, so feeding it to another step as it is
  consumes that state too. The run's next step, and every one after, is then refused before it
  takes anything, with the error that state's own death gives — naming the step that took it.

`Train` and every `Fit` overload already drive a resident run internally and take their checkpoint
on the final step — they return one checkpoint, so they only ever needed one transfer. A manual
`TrainStep` loop is unchanged and still transfers every step; `BeginResidentRun` is how a manual
loop opts out.

On a backend whose provider has no memory of its own, a resident run is an ordinary step loop —
same losses, same checkpoints, to the bit — that additionally releases each step's state as the
next supersedes it.

> A checkpoint's tensors are readable exactly when they are on the host. Reading one a run is still
> holding on the device throws and says so; that state reaches you through `StepToCheckpoint`.

### A step writes its state over the state it consumed

ONNX Runtime keeps every input of a run until the run ends — measured on a card: a 64 MiB feed
read by a graph's first node alone still held its memory when the last node ran — so memory a step
consumes cannot come back part-way through the step to be used for something else. What a step
does instead is write its new state *into* that memory: the updated weight into the weight it
replaces, each updated optimizer moment into the moment. That is right only where nothing reads the
old value after the new one is written, as in an optimizer's element-wise update
(`W ← W − lr·g`), so it is done only where the step's graph proves it: the rig pairs each updated
state field with the field it replaces, and a pair is used only where every node reading the old
value is one the update waits for. The backend proves each pair again over the graph ONNX Runtime
actually runs, whose rewrites can change which nodes read what.

It applies:

- **To state the step consumed** — a checkpoint fed as it is, and a resident run's own state. A
  checkpoint fed `.Shared()` is only read, and is left exactly as it was.
- **Where the new state is produced in the memory the old is in** — every step on a CPU backend,
  and a resident run's `Step` on a GPU, whose state stays on the card. A `TrainStep` or a
  `StepToCheckpoint` on a GPU brings the new state home to the host, so it writes nothing over the
  state it consumed on the card.
- **To the state the graph proves.** Optimizer state qualifies where nothing but its own update
  reads the old value, as AdamW's moments and step counter do. A weight qualifies where everything
  that reads it is something its update waits for; a weight that the backward pass reads
  directly, to pass a gradient on to an earlier layer, is written anew — a model of two bare
  `MatMul` weights has its first marked and not its second, where a stack of `Linear` layers under
  AdamW, measured on a card, has every weight, bias and moment written over.

Nothing changes in what you see. The step returns new tensors, what it consumed is dead as it always
was, and the results are the same to the bit — measured on a card, with it and without it, and
under a device-memory budget. There is nothing to turn on.

What it saves, measured on an RTX 4090: one 4096×4096 `Linear` layer under AdamW — 192 MiB of
weight and moments — in a resident run with shrinkage on needed 704 MiB of arena at each step's
peak without it and 320 MiB with it, and the card's own peak fell by the same 384 MiB: twice the
state, since neither the state a step consumes nor the state it produces sits in the arena beside
the step's working memory any more. Under a device-memory budget the state is still counted —
once, where it lives — see
[inference.md](inference.md#a-contexts-device-memory-budget).

### What construction costs

`FromScratch` does real work before any training happens, and a checkpoint/resume workflow
re-pays most of it on every process start. A training `.skpt` carries the constituents and the
state, not the derived build products, so `TrainingRig.Load` rebuilds those — it reads the saved
concrete architecture rather than re-concretizing, but composition, autograd, optimizer lowering
and graph optimization are all redone.

One cost it does **not** re-pay is the model's initializers
([#327](https://github.com/Shorokoo/Shorokoo/issues/327)). Their values are about to be overwritten
by the checkpoint the load is reading, so `TrainingRig.Load` skips the run and stands the parameters
in with the dtype and shape the architecture declares for them — enough for shape inference and the
optimization pass, and enough to seed the optimizer's state initializers for their shapes (those do
still run, over the stand-ins, reading zeros wherever a payload was elided). The run is deferred,
not dropped: the first thing
to ask the rig for an initial *value* — `CreateInitialCheckpoint()`, or a load whose file omits a
component and falls back to the rig's initial values — runs the initializers then, to exactly the
values an eager build would have produced, and re-seeds the optimizer state from them.

The build phase, all of it on `MergeContext`, is concretization, composition with the loss,
autograd, optimizer lowering, shape inference and graph optimization, plus two costs that scale
with your parameter count: each trainable parameter's initializer is run, and each optimizer-state
initializer is run per trainable parameter. Both run one backend session per parameter, so they
grow linearly with the number of trainable parameters rather than superlinearly, and both copy
each value onto storage of its own rather than leaving it holding that session's working memory.
Peak host memory during initialization still grows with the model, but far more slowly than it
once did — a few hundred bytes per parameter element rather than a few kilobytes.

Then, on the first `TrainStep`, the rig compiles its training-step graph for the shapes it is fed
and caches it (see `TrainStep` above) — one fixed cost per distinct input shape, independent of how
many steps follow; a run that feeds one shape pays it once.

Neither phase is proportional to your dataset, and neither recurs during the loop: steady-state
`TrainStep` pays neither. If you are timing a run, expect the first step to be markedly slower
than the rest — that is the compile, not a slow optimizer.

Steady-state *memory* is flat. `cp = rig.TrainStep(cp, in, out);` feeds the checkpoint as it is,
so the step consumes the state it supersedes and that state goes back as the step returns —
there is nothing left over for a collection to find. What a step supersedes without consuming it —
a checkpoint fed `.Shared()`, or through `.TryConsume()` while something else was reading it —
is alive after the step, and garbage only once you drop it. A checkpoint a step returns holds the
trainable parameters and every optimizer moment in backend buffers behind managed handles of a few
dozen bytes each, far too little managed garbage to prompt a collection on its own; left to the
runtime, a loop that drops such checkpoints would accumulate them until the process died. The rig
therefore collects for you, once more than 32 MiB of that state has piled up — a running total
across steps, not a per-step test. A model whose whole checkpoint is a few kilobytes only reaches
that after thousands of steps, so it pays essentially nothing; a model producing a few MiB a step
pays one collection every few steps; one producing hundreds of MiB a step pays one per step, which
is what a run of that size has to pay to survive at all. Collecting in your own loop is normally
unnecessary and changes nothing but the timing. An initial checkpoint is state like any other here,
counted against the same budget, though its tensors are of another kind: host copies of the rig's
values in managed arrays, which the collector sees at their full size. They are yours to drop.

The rig backs off when a collection turns out to free nothing — a caller that keeps every checkpoint
buys nothing from one — by watching weakly what it handed back and seeing whether a later collection
took it. What it judges is deliberately two reclamations old: the checkpoint from the last one is
what you feed in as the next step's input, so it is alive at the moment of the collection whatever
you do with it. A resident run's retained steps do not go through any of this — the run releases
that state itself — but its `StepToCheckpoint` steps bring state home for you to keep, so those are
reclaimed like any other.

If you **keep** your checkpoints — every one of them, fed `.Shared()` and held — then nothing is
superseded and a collection would reclaim nothing. (Keeping a few — the best so far, or the one
before the last — leaves the rest to be reclaimed as usual, and changes nothing here.) The rig notices: it watches one checkpoint weakly, and each time one survives the
collection it doubles the budget, backing off until keeping checkpoints costs you no collections at
all. It snaps back the moment a watched checkpoint does not survive.

On a large model the build phase runs for minutes. To watch it stage by stage rather than wait
blind, see [Watching a long build](#watching-a-long-build).

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
one set of compiled training-step sessions (one per fed input shape) that the `Fit`/`Train` loop and a
manual `TrainStep` loop all share.
Every `With…` derivation keeps the same two contexts. A third piece of runtime configuration rides
alongside them, equally never persisted: the rig's `TrainingBackend`, which decides whether Shorokoo
or the runtime context's backend computes the gradient — see [training-backends.md](training-backends.md).

**What the two can usefully differ in: the backend, and how its sessions and runs are
configured.** A `ComputeContext` carries `DeviceMemory` — a budget on what it holds on the card,
and the arena settings of the sessions it compiles — and `RunSettings` for what its runs do, so
the merge phase and the training loop can hold different budgets — see
[Device memory](inference.md#device-memory-gpu-backends). It also carries the
backend: a context constructed with one (`new ComputeContext(new LinuxCpuBackend())`) runs
its work there, so a rig **can** build on one device and train on another:

```csharp
var rig = TrainingRig.FromScratch(
    model, loss, optimizer,
    sampleInputs, new AdamWOptimizerHyperparameters { LearningRate = 0.001f },
    mergeContext:   new ComputeContext(new LinuxCpuBackend()),
    runtimeContext: new ComputeContext(new LinuxGpuBackend()));
```

Both devices then have to be deployed and reachable from one process — see
[One model, two devices](inference.md#one-model-two-devices), which is also where the cost is:
what the merge phase produces is handed to the runtime backend by a host copy per feed. Splitting
this way is worth it when the build phase is what does not fit on the card, and pointless when it
is not. Two default-constructed contexts select nothing between them, and the members are then a
division of *phases*: which work is build/merge and which is compile/run. The device each will use
is readable either way: `rig.MergeContext.Backend` and `rig.RuntimeContext.Backend`, or call
`DefaultBackend.RequireDevice(...)` at startup to refuse to train on the wrong one — see
[Which device am I on?](inference.md#which-device-am-i-on).
Leaving both `null`, so each defaults to `ComputeContext.Default`, is the normal choice.

What *is* configurable — on the GPU backends — is **device** memory, on the context the rig compiles
and runs on: a budget and an arena extend strategy in its `DeviceMemory`, per-step arena shrinkage
in its `RunSettings`. The budget covers what the context holds on the card — the state and batches
the steps read there or copy there to read, and whatever the steps leave there — plus the arena of
the step that is running, whose limit is what the rest leaves; a step that finds the context
holding more than its session left room for is rebuilt with less before it runs — see
[A context's device-memory budget](inference.md#a-contexts-device-memory-budget). The arena
strategy needs no setting for this loop: a step compiled for one batch shape repeats it for the
length of the run, and the default `Auto` leaves it on exact-size extension — the arena tracks what
the step asks for rather than doubling past it, which is what otherwise leaves a long run holding
far more of the card than its steps use.

The one place it departs is worth knowing: the rig keeps a compiled step per input shape **up to a
limit**, and a run that feeds more distinct shapes than that falls back to a single step every
later shape shares. That step really does see growing shapes, so `Auto` gives it ORT's doubling —
the strategy that does not strand a region each time an input outgrows it. Feeding a handful of
stable batch shapes keeps every step on the tighter arena.

All of these are the context's, and a context's settings are fixed when it is built: the budget and
the strategy, `ShrinkArenaAfterRun` — which a budget turns on for every step anyway — and the
`CancellationToken` that stops a step early, which is the way to make a long `Fit` or `Train`
abandon a run: `TrainStep` takes no per-call override, so hand `FromScratch` a `runtimeContext`
carrying what you want before the first step
([Stopping a run](inference.md#stopping-a-run) says what stopping costs and what it does not
promise). On a run close to the card's limit, put a budget on that context and sample the peak
inside your `TrainStep` loop; the readings come from the separate static `DeviceMemory` class, and
what the context holds against its budget from `ReadDeviceMemoryUse()`.
See [Device memory](inference.md#device-memory-gpu-backends).

**Mind which memory is which.** A `TrainStep` loop's checkpoints are fetched to the host, so the
rig's budgeted collection governs *host* memory there, while a context's `DeviceMemory` settings
reach only device memory: a process whose RSS climbs is not helped by a device-memory budget, and a
card that fills up is not helped by the rig's reclamation. A resident run is the case where the two meet — its state
stays on the card, written over itself from step to step where the step's graph allows, and the run
releases what it supersedes deterministically, which is why a retained step does not go through the
rig's collection at all. A `StepToCheckpoint` step hands
state back to you instead, so that one is reclaimed like any other.

Result types:
- `TrainingCheckpoint` → `.TrainableParams`, `.ModelState`, `.OptimizerState`, `.Step` (global step, `long`; advances each `TrainStep`, so schedules resume from a saved checkpoint), and the host-owned run counters `.Epoch` / `.BatchIndex` (`long?`; the training loop advances them — the counter-agnostic `TrainStep` carries them through unchanged). They are `null` when the position is genuinely **unknown** — an initial checkpoint, or one trained without a data loader / explicit counters — rather than a misleading `0`; the loader-driven and explicit-counter paths set concrete values. A scheduled hyperparameter reading the epoch / batch counter sees `0` for a `null` value. `.Step` is always a concrete `long`; all counters are `int64` end to end. It also carries `.Rig` (the `TrainingRig?` that produced it — set on every rig-produced checkpoint, so `checkpoint.ToInferenceModel()` needs no re-supplied graph) and `.Loss` (`float?`; the loss of the `TrainStep` that produced it, `null` on an initial or bare checkpoint). Both are preserved through the counter derivations (`WithCounters`/`WithStep`/`WithEpoch`/`WithBatchIndex`) and through the state derivations (`WithTrainableParams`/`WithModelState`/`WithOptimizerState`), each of which returns a new checkpoint with one slot replaced and every other slot carried through unchanged — the receiver is never mutated. `TrainStep` returns this checkpoint directly — read the step's loss off `.Loss`. `.Loss` persists as its own `Loss` component, independent of `Counters` (dropping `Loss`, or an initial checkpoint, reloads with `.Loss == null` — never a sentinel `0`).
- `TrainingResult` → `.FinalCheckpoint`, `.EpochLosses` (the per-epoch mean losses).

**Constructing one directly.** You are normally *handed* a checkpoint — by `CreateInitialCheckpoint`,
`TrainStep`, `Fit`/`Train`, or a load — and the derivations above cover changing one slot of one you
already have. When you do need to build one outright, the three tensor-state slots are **required
object-initializer properties**, so each is named at the call site:

```csharp
var checkpoint = new TrainingCheckpoint
{
    TrainableParams = trainable,
    ModelState      = modelState,
    OptimizerState  = optimizerState,
    Step            = 42,          // Epoch / BatchIndex / Rig / Loss are optional
};
```

All three are the same type (`TensorDataStruct`), and for a real model they often have the same field
shapes too — so a positional form would let the optimizer's moments be passed as the parameters,
compile, and train to a plausible-looking loss, with the mistake surfacing only much later. Naming
them removes that: the compiler requires all three and rejects an initializer that omits one. Prefer a
derivation over re-writing a whole initializer when you are changing one slot of an existing
checkpoint — `ckpt.WithTrainableParams(next)` cannot drop or transpose the slots it carries through.

`TrainingRig`, `TrainingCheckpoint`, and `TrainingResult` are in
namespace `Shorokoo` (covered by `using Shorokoo;`).

### Watching a long build

`FromScratch` can run for **minutes** on a large model — it concretizes the graph, composes the loss
and autodiff, unrolls loops, and runs shape inference plus the memory-aware graph optimization before
it returns. By default it says nothing while doing so, which is indistinguishable from a hang. Hand it
a sink to see the stage it is in:

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

(`…` marks stages elided here, not gaps in the output — every stage reports. The last line is the
terminal report, the one whose `IsComplete` is true.)

Each `BuildProgress` is reported as the build **enters** the named stage, so a build that has been
quiet for minutes is inside the stage its last report named. It carries four members:

- `Phase` — `BuildPhase.Concretize` (lowering the model to a concrete architecture),
  `BuildPhase.TrainingStep` (composing and lowering the training-step graph), or
  `BuildPhase.Initialize` (running the initializers, shape inference and graph optimization). The
  phases run in that order and do not interleave.
- `Stage` — the work being entered, named for what it does and usually after the pass that runs it
  (`InlineModulesAndFunctions`, `ExpandAutoGrad`, `OptimizeTrainingStepGraph`, …). Stage names are
  diagnostics, not API: they track the pipeline and change with it, so never branch on one.
- `IsComplete` — true for the single terminal report of a build that ran to completion, which names
  no stage being entered and renders its `Stage` as `Done`: a stream sitting on it is finished, not
  stuck (the one report the rule above does not apply to). A build that threw never emits one. It is
  stamped by the build, not re-derived from the stage text, so test it rather than the `Done` string.
  `Stage` is the one member a program should not branch on; the other three are stable.
- `Elapsed` — time since the start of *this* build. One clock spans all three phases.

One caveat for `TrainingRig.Load`, which reports through the same sink: it reports
`DeferModelParamInitialization` where a fresh build reports `InitializeModelParams`, because it does
not run them ([#327](https://github.com/Shorokoo/Shorokoo/issues/327)). That work has not vanished —
it moves to whichever later call first asks the rig for an initial value, and *that* call reports
nothing. A resume never makes such a call, so it simply costs less; a load that then asks for an
initial checkpoint pays the initializers there, silently.

`ToString()` renders the line shown above. Reports are raised **synchronously on the building
thread**, so use `SynchronousBuildProgress` (which calls its handler inline) rather than
`System.Progress<T>`, whose posted callbacks can arrive out of order or after the build returns —
exactly what a liveness signal must not do. Keep the handler short; it runs inside the build, and an
exception it throws propagates out of the build and discards it. Hand the same sink to two concurrent
builds and their reports interleave on their own threads — each build still carries its own clock, so
give each one its own sink unless you want that.

**The sink is an argument to one build, not configuration.** Only the call you hand it to reports to
it: two builds watched at once cannot interleave into one sink by accident, a build handed none is
silent, and there is no global switch to leave on. Every build that can run long takes one —
`FromScratch` (all three phases); each `With…` derivation, which reuses the concrete architecture and
so opens at `TrainingStep`, except `WithSeed`, which clones and rebinds the RNG identity and so opens
with a `Concretize` phase naming that work; `TrainingRig.Load`, which likewise opens at `Concretize`,
naming its file reads rather than lowering passes, and reports complete only once the resumed
checkpoint is in hand; and the lowering step on its own —

```csharp
var concrete = MyModel.ComputationGraph.ToConcreteArchitecture(
    inputHints, progress: new SynchronousBuildProgress(p => Console.WriteLine(p)));
```

— which reports `Concretize`, its own thaw and freeze included, and ends complete. The
positional-hyperparameter shorthands cannot take a sink, since a `params Hyperparameter[]` must come
last; on each, passing the values as an array instead reaches the overload that can —
`FromScratch(model, loss, opt, sample, [0.01f], progress: sink)`,
`rig.WithOptimizer(opt, [0.01f], sink)`.

Reporting covers the **build** and stops there. Calls that are not builds stay silent: `ToConcreteModel`,
`InitializeTrainableParams` and `GetRngStreamReport` take no sink, though all three can be slow. Neither
does the first `TrainStep` at a shape, whose compile ([above](#what-construction-costs)) is not a build —
so expect a quiet stretch there after the build has reported itself complete.

To capture the *graphs* rather than the stage names — after the fact, as compilable C# — see
[debugging.md](debugging.md).

### Seeding the run

`rngConfig` binds the run's [RNG configuration](rng-configuration.md): one master
seed keys parameter initialization and every runtime draw (Dropout masks, in-model
sampling). Omitted (or `null`), the rig keys under the **default identity** (master
seed 0) — training is deterministic and reproducible by default. Dropout masks still
vary per training step (the per-step RNG position is saved in the checkpoint, so a
resumed run continues exactly). Pass
`new RngConfig { MasterSeed = … }` to re-roll all streams coherently, or
`RngConfig.NonDeterministic()` for per-run variation.

### When a training step runs out of memory

An allocation failure inside a step arrives from the backend as bare text — an ONNX Runtime arena
message quoting a build-agent source path and a request size, or the two words `bad allocation` — and
on its own it says neither which memory ran out nor how close the process was to any limit. Shorokoo
wraps it as a `ComputeContextException` with code `CR009` that adds what the step already knows:

- **Which pool.** `HOST memory` for a C++ allocation the process could not commit, `DEVICE memory`
  for the accelerator's arena. A bare `bad allocation` is a *host* failure even on a GPU backend, so
  the two no longer read the same. Where the backend's text names no allocator at all and the session
  does have device memory, the report says so rather than guessing, and leans on the figures below.
  Note ONNX Runtime's arena message is execution-provider-agnostic — the same text comes out of the
  CPU arena — so on a CPU-only session it is read as host memory, never as a device that isn't there.
- **What the step was holding.** Trainable parameters, model state, optimizer state and the batch
  itself, each with its tensor count and total size, plus the five largest tensors by size. For a
  small model on a large batch the batch is the whole of it, and the report says so.
- **The card's own figures**, where a CUDA runtime is installed to ask (`DeviceMemory.Read()`): used,
  free and total across all processes, plus this process's arena cap if one is set. A 2.36 MB request
  refused with gigabytes still free reads as absurd until the free figure is printed beside it.
- **This process's memory position.** Working set, commit charge and managed heap against the memory
  limit in force — a cgroup/container limit, a Job Object limit, or the machine's RAM.

The last two together are the discriminator. On Windows/WDDM a device allocation is backed by system
commit, so a process memory limit meant to bound host RAM silently bounds device memory too, and an
arena expansion past it fails with **the same message a genuinely full accelerator produces**. The two
call for opposite responses — shrink the model, versus raise a limit that has nothing to do with the
model — so the report distinguishes them outright, in three cases: the device is out of memory; the device has
room but this process is at its own limit (the limit, not the model); or the device has room and so
does the process, so the arena simply could not extend by the block it wanted.

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

The backend's own text is preserved verbatim at the end, and the original exception is kept as the
`InnerException`. Only allocation failures are relabelled — everything else a step can raise keeps its
type and message.

One failure mode stays outside this: when the host runs out while the **garbage collector** needs
memory, the runtime fails fast (`Fatal error. 0xE0004743`) before any managed handler runs, so there
is no exception to wrap. The inventory printed by the last step that did fail is the way in.

## Feeding data: the data loader

The array overloads of `Fit`/`Train` take pre-batched `TensorDataStruct[]` — a dataset fed every
epoch, so each step reads its batch and never consumes it — and leave the checkpoint's epoch /
batch counters for you to set. A **data loader** instead owns the batch
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
- **A batch is fed as the loader built it.** `DataBatch.Input` and `.Target` are `IData`: a
  `TensorDataStruct`, which the step that trains on it consumes, or one passed through `.Shared()`
  or `.TryConsume()`. `InMemoryDataLoader` gathers a fresh batch on every `Next()`, so each step
  consumes its own and the dataset stays whole; a loader of your own that hands out tensors it
  keeps, and hands out again, passes them `.Shared()`. The step that reads such a batch lets go of
  the copies it made to read it as it returns
  ([What a step keeps of what it reads](#what-a-training-step-consumes)), so the tensors your loader
  keeps are not also kept in the run's memory — on a card, not all on the card at once.
- **Shuffle is deterministic.** With `shuffle: true`, the permutation for epoch `e` is a pure
  function of `(seed, e)` — a Fisher–Yates shuffle over a SplitMix64 stream, using no ambient
  `Random` and no wall clock. That is what makes resume exact: restoring to `(e, b)` regenerates
  epoch `e`'s order bit-for-bit and skips the first `b` batches, so the continued run sees the
  same batches the original would have.
- **Partial final batch.** `dropLast: true` (the default) drops a trailing partial batch so every
  batch matches the shape the training-step session was compiled for. Pass `dropLast: false` to keep
  the smaller final batch (only safe if the graph tolerates a variable batch dimension). The rig
  compiles its training-step session for the exact input shapes it is fed — that is what lets ONNX
  Runtime fold the step's shape arithmetic away — so a differently-shaped batch costs one extra
  session compile the first time it appears, and is then cached like the full-size shape. Up to four
  distinct shapes get their own session; any further shape runs on a shape-generic session instead,
  so a stream of never-repeating shapes pays a bounded number of compiles and then behaves as before.
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
  the constituents the file carries and hands it back alongside the resumed checkpoint (and, when
  you want the file's model rather than its rig, `Persistence.Load` and
  `Persistence.LoadEvaluationModel` read the same file with no rig at all), so the
  rig need not be rebuilt by you at all. Each on-disk format has its own load entry point:
  `rig.LoadCheckpoint` reads the flat safetensors file only, `rig.LoadCheckpointFromSkpt` and
  `TrainingRig.Load` the `.skpt` container only, and handing any of them the other format fails
  immediately with an error naming the right entry point (nothing sniffs the file's bytes to
  pick a path; to identify an unknown file, use `Persistence.Inspect`).
  See [skpt-checkpoints.md](skpt-checkpoints.md#training-checkpoints).
- `LoadCheckpoint` / `LoadCheckpointFromSkpt` reconstruct the checkpoint against the rig's own
  parameter and state definitions, so the rig must be built from the **same**
  model/loss/optimizer graphs. What is checked, and where: the rig compares field names, dtypes and
  **dimensions** against its own parameters as it adopts the values, and every value is
  checked once more against the shape the model declares for it at the point it is bound into a
  graph — so a value of another shape is refused even when the checkpoint was assembled by hand
  rather than loaded. A parameter whose stored shape differs from the model's is never bound
  silently; unchecked it would broadcast, and the model would run and answer in the wrong shape.
  What is **not** checked is where a value came from: nothing records or compares the model that
  produced a checkpoint, so weights of the right shape deliberately still load into a model that
  computes something else. The limit of that is worth knowing — two parameters of the same shape
  whose roles were swapped agree on every property checked here, and load into each other's places
  without complaint ([#322](https://github.com/Shorokoo/Shorokoo/issues/322)).
- Because `.Step` is restored, learning-rate **schedules resume from the right
  step** — not from step 0.
- `rig.LoadCheckpoint(path)` delegates to `TrainingCheckpoint.Load(path, rig)` (and
  `rig.LoadCheckpointFromSkpt(path)` to `TrainingCheckpoint.LoadFromSkpt(path, rig)`), which
  resolves the struct defs from the rig and sets `.Rig` on the result. Without a rig,
  `Persistence.LoadTrainingCheckpoint(path)` reads a flat checkpoint on its own — the file is
  self-describing, so it needs no struct defs — and `TrainingRig.Load(path)` rebuilds rig and
  checkpoint together from a `.skpt`. A checkpoint read without a rig carries no `.Rig` and has
  been checked against nothing: only a rig knows what parameters to expect, so hand it to
  `rig.AdoptCheckpoint(ckpt)` to have its fields and shapes validated against a model.
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

### What a save costs

Every checkpoint save that writes a **single file** returns a `SaveReport` — the bytes it committed
and where its time went ([#338](https://github.com/Shorokoo/Shorokoo/issues/338)):

```csharp
var save = checkpoint.Save("run.safetensors");
Console.WriteLine(save);
// 200,000,077 bytes in 0.252s (757 MiB/s): write 0.051s, flush 0.194s, commit 0.007s
```

`Write` is producing the content and writing it into the staged file — serializing the state, and
for a `.skpt` also compressing and hashing its entries — `Flush` the fsync that makes it durable,
and `Commit` the rename that publishes it plus the sweep of any staged sibling an earlier
interrupted save left behind. The three are disjoint and add up to `Elapsed`, the wall clock of the call;
`BytesWritten` is the committed file's own size, and `BytesPerSecond` the rate that save achieved.
`Persistence.SaveTrainingCheckpoint`, `Persistence.SaveTrainingCheckpointToSkpt` and the
`Persistence.ForTrainingCheckpoint(...)` builder's `Save` all return the same report. The
**directory** form (`SaveAsDirectory`) does not: it commits a tree of files rather than one, which
is a different measurement, and it still returns `void`.

Two reasons it is reported rather than left to be worked out from the file's size:

- **The cost does not follow the size.** Two identical saves of one identical file differ, and the
  phases are separated because the total alone cannot say why. For a flat save at size it is the
  flush that both dominates and moves — what it costs depends on how much of the file the OS had already written
  back before it ran — while the serialization is steady and the commit is metadata-only. One
  200 MB file saved six times over: write steady at 51 ms, flush between 184 and 243 ms, commit at
  7 ms. The rate a save achieves is a property of that save, not a constant of the machine to
  calibrate once.
- **At a checkpoint cadence it is not negligible to the run.** Saving a multi-GB checkpoint every N
  steps can cost tens of seconds each time; over a long run that is minutes to tens of minutes.

Saving is disk I/O, not training. A loop that reports its own throughput should subtract the
returned `Elapsed` from the window it measures, rather than charging the checkpoint cadence to the
training rate and reporting a step time that silently moves with it:

```csharp
steady.Stop();                                   // saving is I/O, not training
var save = checkpoint.Save(path);
steady.Start();
savedBytes += save.BytesWritten;
```

The **flat safetensors** save writes each tensor's payload straight out of its storage, so it costs
no second copy of the training state in memory. The `.skpt` container does not share that: it
serializes each state kind to a `byte[]`, hashes it and holds every entry in memory until the write
begins, so budget for a full extra copy of the training state there — which is why its `Write` phase
dominates its report. Neither form can yet go past the safetensors layer's 2 GB ceiling — a
checkpoint at or above that size is written without complaint and then cannot be read back
([#48](https://github.com/Shorokoo/Shorokoo/issues/48)).

### Bind trained weights into an inference model

Once trained, turn a checkpoint into a runnable concrete model with one call:

```csharp
var concrete = result.FinalCheckpoint.ToInferenceModel();   // no graph to re-supply
var output   = ComputeContext.Default.Execute(concrete, myInput);
```

From a **saved** checkpoint, no rig is involved at all: a training `.skpt` carries the model, so
`Persistence.Load(path)` returns it runnable and `Persistence.LoadEvaluationModel(path)` returns it
composed with its loss, for scoring a validation set
([#329](https://github.com/Shorokoo/Shorokoo/issues/329)). Reach for `TrainingRig.Load(path)` when
you mean to go on *training*; it is the one of the three that builds a rig. The flat safetensors
format stores state and no architecture, so none of this applies to it — a checkpoint read with
`Persistence.LoadTrainingCheckpoint` has no model in it to bind, and needs a rig that does.

```csharp
var model = Persistence.Load("run.skpt");                  // ConcreteModel, weights bound
var eval  = Persistence.LoadEvaluationModel("run.skpt");   // [model inputs…, targets] → loss
```

`ToInferenceModel()` binds this checkpoint's trainable params and model state, by canonical
identity, into the checkpoint's `.Rig`'s **retained concrete architecture** — the model the rig
concretized once at build time (at **all** its inputs, so multi-input models are supported) and
holds for reuse. No re-concretization and no sample inputs are involved. It requires an attached
rig — every rig-produced checkpoint has one; attach one to a bare checkpoint with
`rig.AdoptCheckpoint(checkpoint)` first.

## Types used by the training API

These are in namespace `Shorokoo` (covered by `using Shorokoo;`), except `Schedule` and `Schedules`, which are in `Shorokoo.Core.Training` and need `using Shorokoo.Core.Training;`:

| Type | Role | How to make one |
|---|---|---|
| `NamedModelParam` (abstract) | A named parameter value. | Use the concrete `TensorDataModelParam`. |
| `TensorDataModelParam` | Concrete `NamedModelParam` wrapping one `TensorData`. | `new TensorDataModelParam(name, ModelParamType.InputParam, tensorData)` |
| `ModelParamType` (enum) | Tags a param's role. | `Undefined`, `HyperParam`, `TrainableParam`, `InputParam`, `OutputParam` |
| `ModelParamList` | A set of named params (e.g. loaded weights). | `new ModelParamList(IEnumerable<(string name, TensorData data)>)` |
| `TensorDataStruct` | A struct-shaped bundle of named `TensorData` fields; the form `Train`/`TrainStep` expect for inputs/targets. | Build: `new TensorDataStruct(structDef, fields)` where `structDef` is a `TensorStructDef` (namespace `Shorokoo.Core`) and `fields` are `KeyValuePair<string, IData>` — one per definition field, each of the kind that field declares (a value contradicting its definition throws), as it is or through `.Shared()` / `.TryConsume()` to be fed that way rather than as the struct is (a struct fed `.Shared()` has every field read). Read: `.Fields` (an `ImmutableDictionary<string, IData>` of name → value), `.Count`, or the `[int]` indexer. |
| `SharedInput` | A value to be **read** by the run it feeds rather than consumed (`Mode` `Shared`), or consumed only if nothing else is reading it (`TryConsume`). | `x.Shared()` / `x.TryConsume()` on a `TensorData`, `TensorDataStruct`, `TensorDataSequence` or `OptionalTensorData`. A checkpoint's own `.Shared()` / `.TryConsume()` return a checkpoint, carrying the mode as its `FeedMode`, and a `NamedModelParam`'s a copy of the parameter with its `FeedMode` set. A struct's field may be given as one when the struct is built. |
| `SaveReport` | What a checkpoint save cost: `BytesWritten`, the disjoint `Write` / `Flush` / `Commit` phases, their sum `Elapsed`, and `BytesPerSecond`. | Returned by every checkpoint save — see [What a save costs](#what-a-save-costs). |
| `Schedule` (namespace `Shorokoo.Core.Training`) | A `step → value` hyperparameter schedule; assign one to a `Hyperparameter` property to make it [`Scheduled`](#hyperparameter-kinds-hyperparameter). | A `Schedules.…` factory, then the combinators on the result (`WithWarmup`, `Then`, `Scale`, `Clamp`, `Shift`, `PerEpoch`). Preview with `.At(step)`. |
| `Schedules` (static, namespace `Shorokoo.Core.Training`) | The factories: `Constant`, `Linear`, `Cosine`, `CosineWithWarmup`, `StepDecay`, `Exponential`, `OneCycle`. | Call one — `Schedules.Cosine(1e-3f, totalSteps)`. See [Schedule factories and combinators](#schedule-factories-and-combinators). |

### Sample inputs

`sampleInputs` for `FromScratch` gives one sample per data input of the model — what its shape
is, and the values concretization reads — in either of two forms:

- **Positional**, an `IData[]` of bare values (`TensorData`, or `OptionalTensorData` for an optional
  input) in the model's input declaration order, each bound to the input at its position:
  `FromScratch(model, loss, opt, [TensorData([4L, 64L], new float[256])], hypers)`. Too few or too many
  is refused with `FW056`.
- **Named**, a `NamedModelParam[]` or a `ModelParamList`, each sample bound to the model input of its
  name, in any order: `FromScratch(model, loss, opt, [new TensorDataModelParam("input",
  ModelParamType.InputParam, x)], hypers)`. An input no sample names, a sample naming no input, and a
  name given twice are refused with `FW056`, the message naming each offender and listing the model's
  inputs.

In either form a sample whose rank contradicts its input's declared rank is refused with `FW056` too.

`Train`/`TrainStep` take `TensorDataStruct` batches — `TrainStep` as they
are, to be consumed, or through `.Shared()` to be read and kept; `Train` and `Fit` over arrays read
every batch, since they feed the arrays again each epoch, and leave them all alive
([What a training step consumes](#what-a-training-step-consumes)).

## Workflow: train a model

1. Define model, loss, and optimizer as `[Module]` classes (or reuse built-ins).
   To train several sizes or configurations of one model, do **not** write a class per
   variant: give the model `[Hyper]` parameters and `Specialize` the graph per variant —
   `FromScratch` takes the specialized graph directly. See
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
costs one float per parameter instead of a full copy; Adam's and AdamW's bias-correction
timestep works this way.

Constraints:

- **State must come from an optimizer-owned state initializer.** Declaring state as an
  `Inline` parameter throws at rig-build time, and `Globals.StateUpdate` itself throws
  `InvalidStateUpdateException` if its first argument is not a state variable. Module-owned
  initializers (e.g. BatchNorm's running-stat initializers) are rejected inside optimizer
  graphs, and optimizer-owned ones are rejected inside model graphs.
- **Each state is updated exactly once per step** — combine conditional updates into one
  value (e.g. with `IfElse`) and register it with a single `StateUpdate` call.
- **The updated parameter must come back at the parameter's own shape.** The update is ordinary
  tensor arithmetic, so a hyperparameter or state of another shape *broadcasts* against the
  parameter instead of scaling it, and the "updated" parameter takes the other shape. A rig whose
  optimizer does that is refused at build, naming the parameter and both shapes — a per-element
  hyperparameter therefore needs parameters it fits (one rate per weight), not one rate vector
  against a scalar weight.
- **Hyperparameters must be tensor-shaped** — `Scalar<T>`, `Vector<T>` or `Tensor<T>`, at any supported
  dtype (`float32`, `int32`, `bit`, …); the rig bakes/feeds them at their declared dtype and shape, and a
  set is generated even when the dtypes and shapes are mixed. An `OptionalTensor`, sequence or struct
  hyperparameter yields no generated set. Only a scalar can carry a `[Hyper(default)]` default.
- **Order + `[Hyper]` matter** — hyperparameters must be the leading inputs, and `[Hyper]` is what
  makes the named set generate. Without it the optimizer still works via the positional
  `params Hyperparameter[]` overload, but you lose the named, compile-checked set.
- For non-generated cases you can hand-implement `IOptimizerHyperparameters` yourself.

## Notes / known limitations

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
  AdamW and Adam: `m`/`v` plus a scalar `step`; RMSprop: `squareAvg`/`momentumBuffer`;
  Adagrad: `accumulator`; Adamax: `m`/`u` plus a scalar `step`; NAdam: `m`/`v` plus two
  scalars — `step` and `muProduct`; RAdam: `m`/`v` plus a scalar `step`; Adadelta:
  `squareAvg`/`accDelta`; Lion: `m` only — half the param-shaped state of Adam/AdamW, and no scalar; Lamb: `m`/`v` plus a
  scalar `step` — Adam's footprint, the per-tensor trust ratio being recomputed each
  step and stored nowhere; Adafactor: a **full param-shaped** `v` plus a scalar
  `step` — same footprint as Adam, because the
  sublinear-memory row/column factoring is **not** implemented, see above) — see the table in
  [nn-library.md](nn-library.md). Each field is
  initialized by running its state initializer: `OptimizerStateZeros` zero-fills at the
  parameter's shape, `OptimizerScalarZeros` produces a rank-0 scalar seeded at 0 (e.g. Adam's and
  AdamW's `step`, one float per parameter rather than a param-shaped buffer), and `OptimizerScalarOnes`
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
- Do not feed a batch or a checkpoint you will use again as it is: the step consumes it, and the
  next read throws. Pass it `.Shared()`.
- Do not run a long GPU training loop on `TrainStep` when you only want the last checkpoint: every
  step then pays a full download and upload of parameters and optimizer state. Use
  `rig.BeginResidentRun()`, or `Fit` / `Train`.
- Do not expect a resident run's state after disposing it — take it with `StepToCheckpoint` on the
  last step you care about, before the run goes away.
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
