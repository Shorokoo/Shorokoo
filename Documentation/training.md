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
  a runtime input or a trainable parameter is rejected with instructions.

## Built-in components

Ready-made losses and optimizers ship in the `Shorokoo.Modules` package
(namespaces `Shorokoo.Modules.Losses` / `Shorokoo.Modules.Optimizers`) — see
[nn-library.md](nn-library.md) for the full catalog (eight losses; layers and
initializers too). Each optimizer with scalar `float32` hyperparameters gets a
source-generated, named, defaulted hyperparameter set
(`<Optimizer>Hyperparameters`) implementing `IOptimizerHyperparameters`. The
twelve optimizers (the positional `params HyperValue[]` count for `FromScratch`
equals each set's property count):

| Optimizer | Hyperparameter set (named, init-only `HyperValue` properties; defaults from `[Hyper]`) |
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

A loss module has signature `(predictions, targets) -> Scalar<float32>` with exactly two tensor inputs; targets are typically `Tensor<float32>`, but class-index losses (`CrossEntropyLoss`, `NLLLoss`) take `Tensor<int64>` targets.
The library losses' configurable knobs (`reduction`, `ignore_index`, `label_smoothing`, class `weight`/`pos_weight`, SmoothL1 `beta`) live on extra `Reduced`/`PerElement` methods, *not* on the rig-bound `Inline`. Knobs that stay scalar and add no input (`reduction = Mean`/`Sum`, `ignoreIndex`, `labelSmoothing`) are rig-usable by writing a tiny 2-input wrapper `[Module]` whose `Inline` calls `Reduced(...)` with the knobs baked; a class `weight`/`pos_weight` (an extra tensor input) is rig-usable only when **baked as a graph constant** inside such a wrapper. See the [Losses → Configurable knobs](nn-library.md#loss-configurable-knobs) section for the recipes.
An optimizer module takes its `[Hyper]` hyperparameters, then exactly `(currentParam, grad)`,
and returns the updated parameter. Optimizer state never appears in the signature: it is
created inside the body by an **optimizer-owned state initializer** (e.g.
`OptimizerStateZeros.Init(currentParam.ShapeTensor())`) and updated with one
`Globals.StateUpdate(state, newValue)` call per state — see
[Custom optimizers](#custom-optimizers).

### Hyperparameter kinds (`HyperValue`)

Each hyperparameter property is a `HyperValue`; its *kind* — not a separate flag — decides the wiring:

| Assign | Kind | Wiring |
|---|---|---|
| a `float` (e.g. `1e-4f`) | baked | graph `Constant`; change ⇒ rebuild |
| a `Schedule` (e.g. `Schedules.Cosine(3e-4f, total)`) | scheduled | runtime input; evaluated at the checkpoint step every `TrainStep` |
| `HyperValue.Runtime(seed)` | manual | runtime input with no schedule; supply each step via `MakeHyperparams` |

`Schedule` factories live on `Schedules` (`Constant`, `Linear`, `Cosine`, `CosineWithWarmup`,
`StepDecay`, `Exponential`, `OneCycle`) with fluent combinators on the result (`WithWarmup`, `Then`,
`Scale`, `Clamp`, `Shift`, `PerEpoch`).

## `TrainingRig` API

```csharp
public static TrainingRig FromScratch(
    FastComputationGraph modelGraph,
    FastComputationGraph lossGraph,
    FastComputationGraph optimizerGraph,
    NamedModelParam[] sampleInputs,            // names + sample shapes for model inputs
    IOptimizerHyperparameters hyperparameters, // named set, e.g. new AdamWOptimizerHyperparameters { ... }
    RngConfig? rngConfig = null);              // seeds the run — see "Seeding the run" below

// Lower-level: positional values (a float bakes a constant, a Schedule schedules it):
//   FromScratch(model, loss, opt, sampleInputs, params HyperValue[] hyperparameters)
//   FromScratch(model, loss, opt, sampleInputs, rngConfig, params HyperValue[] hyperparameters)

public TrainingCheckpoint CreateDefaultCheckpoint();

// Schedule-driven: applies each schedule at checkpoint.Step, then advances the step.
public TrainingStepResult TrainStep(
    TrainingCheckpoint checkpoint,
    TensorDataStruct trainingInput,
    TensorDataStruct trainingOutput,
    CompiledGraph compiled);

// Explicit override: supply dynamic hyperparameter values for this step.
public TrainingStepResult TrainStep(
    TrainingCheckpoint checkpoint,
    TensorDataStruct hyperparams,              // from MakeHyperparams(...)
    TensorDataStruct trainingInput,
    TensorDataStruct trainingOutput,
    CompiledGraph compiled);

public TensorDataStruct MakeHyperparams(float value);                       // exactly one dynamic
public TensorDataStruct MakeHyperparams(params (string name, float value)[] values); // named

public TrainingResult Fit(  // alias: Train(...)
    TrainingCheckpoint initialCheckpoint,
    TensorDataStruct[] trainingInputs,
    TensorDataStruct[] trainingOutputs,
    int numEpochs,
    ComputeContext ctx);
```

Result types:
- `TrainingCheckpoint` → `.TrainableParams`, `.ModelState`, `.OptimizerState`, `.Step` (global step; advances each `TrainStep`, so schedules resume from a saved checkpoint).
- `TrainingStepResult` → `.Checkpoint`, `.Loss`.
- `TrainingResult` → `.FinalCheckpoint`, `.EpochLosses`.

`TrainingRig`, `TrainingCheckpoint`, `TrainingStepResult`, and `TrainingResult` are in
namespace `Shorokoo` (covered by `using Shorokoo;`).

### Seeding the run

`rngConfig` binds the run's [RNG configuration](rng-configuration.md): one master
seed keys parameter initialization and every runtime draw (Dropout masks, in-model
sampling). Omitted (or `null`), the rig keys under the **default identity** (master
seed 0) — training is deterministic and reproducible by default. Dropout masks still
vary per training step (the generator's execution counter advances each step and
rides the checkpoint, so a resumed run continues exactly). Pass
`new RngConfig { MasterSeed = … }` to re-roll all streams coherently, or
`RngConfig.NonDeterministic()` for per-run variation.

## Save and resume a checkpoint (across process restarts)

A `TrainingCheckpoint` holds the full training state — trainable params, model
state, optimizer state, and the global step. Save one to disk and resume from it
in a later run:

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

- The file is a single SafeTensors file (every param/state field plus the step).
- `LoadCheckpoint` reconstructs the checkpoint against the rig's own parameter
  and state definitions, so the rig must be built from the **same**
  model/loss/optimizer graphs. Loading a checkpoint from a different model or
  optimizer throws a clear error rather than silently misbinding.
- Because `.Step` is restored, learning-rate **schedules resume from the right
  step** — not from step 0.
- `TrainingCheckpoint.Load(path, trainableDef, modelStateDef, optimizerStateDef)`
  is the lower-level static form if you hold the struct defs without a rig;
  `rig.LoadCheckpoint(path)` just passes the rig's defs to it.

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
3. Initialize parameters: `var ckpt = rig.CreateDefaultCheckpoint();`.
4. Run epochs: `var outcome = rig.Fit(inputs, targets, numEpochs: 10);`
   The learning-rate schedule is applied automatically as the global step advances. (Or call
   `rig.TrainStep(...)` per batch; pass `rig.MakeHyperparams(...)` to override a step explicitly.)
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
- **Hyperparameters must be scalar `float32`** — the rig bakes/feeds them as float32 scalars and
  schedules are `step → float`. A non-float or *mixed* hyperparameter list yields no generated set.
- **Order + `[Hyper]` matter** — hyperparameters must be the leading inputs, and `[Hyper]` is what
  makes the named set generate. Without it the optimizer still works via the positional
  `params HyperValue[]` overload, but you lose the named, compile-checked set.
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
  expressible in Shorokoo's single rank-agnostic per-parameter optimizer graph (the
  factored/unfactored switch is a runtime rank branch whose arms would thread out
  differently-shaped state, which ONNX `If` cannot return). A user reaching for Adafactor
  specifically to save memory gets Adam-sized state; `learningRate` is the **cap** on the
  relative step, not a fixed lr.
- Prefer the optimizer's generated named set (`<Optimizer>Hyperparameters`); it has the right
  names/defaults and is checked at compile time. The positional `params HyperValue[]` overload must
  still match the optimizer's hyperparameter count exactly: SGD=1, SGDMomentum=2, Adam=4,
  RMSprop=4, AdamW=5, Adagrad=2, Adamax=4, NAdam=5, RAdam=4, Adadelta=3, Lion=4, Adafactor=6.
- Optimizer state has one or more fields per trainable parameter (momentum: velocity;
  AdamW: `m`/`v`; Adam: `m`/`v` plus a scalar `step`; RMSprop: `squareAvg`/`momentumBuffer`;
  Adagrad: `accumulator`; Adamax: `m`/`u` plus a scalar `step`; NAdam: `m`/`v` plus two
  scalars — `step` and `muProduct`; RAdam: `m`/`v` plus a scalar `step`; Adadelta:
  `squareAvg`/`accDelta`; Lion: `m` only — half of Adam/AdamW; Adafactor: a **full
  param-shaped** `v` plus a scalar `step` — same footprint as Adam, because the
  sublinear-memory row/column factoring is **not** implemented, see below) — see the table in
  [nn-library.md](nn-library.md). Each field is
  initialized by running its state initializer: `OptimizerStateZeros` zero-fills at the
  parameter's shape, `OptimizerScalarZeros` produces a rank-0 scalar seeded at 0 (e.g. Adam's
  `step`, one float per parameter rather than a param-shaped buffer), and `OptimizerScalarOnes`
  a rank-0 scalar seeded at 1 (e.g. NAdam's running momentum product, which needs the
  multiplicative identity).

## Anti-patterns

- Do not mismatch the positional `params HyperValue[]` overload with the optimizer's hyperparameter
  count; prefer the named set so this can't happen.
- Do not call the schedule-driven `TrainStep` on a rig whose dynamic hyperparameter is
  `HyperValue.Runtime` (schedule-less); supply it via `MakeHyperparams` and the override overload.
- Do not implement backward passes manually; rely on autodiff.
- Do not mutate `TrainingCheckpoint` in place across steps; thread the returned
  checkpoint forward.
- Do not declare optimizer state as `Inline` parameters — state is created inside the body
  via an optimizer-owned `[StateInitializer]`'s `Init` and registered with `StateUpdate`.
- Do not call `Globals.StateUpdate` on inputs, trainable parameters, or computed tensors;
  only state variables (a `[StateInitializer]` `Init` result) are accepted.
