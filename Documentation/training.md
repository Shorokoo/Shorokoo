# Training models

Compose a model, a loss and an optimizer into a `TrainingRig` and run its training steps. For a
first program end to end, start at [first-training-run.md](first-training-run.md). Four pages go
further:

- [training-hyperparameters.md](training-hyperparameters.md) — each optimizer's hyperparameter set,
  baked, scheduled or runtime hyperparameters, schedules, and custom optimizers.
- [training-data.md](training-data.md) — data loaders, resuming at the next batch, and the training
  history.
- [training-checkpoints.md](training-checkpoints.md) — save and resume a run, what a save costs, and
  bind trained weights into an inference model.
- [training-memory.md](training-memory.md) — out-of-memory reports, what a process's memory holds,
  and sizing a process memory limit.

Related: [defining-models.md](defining-models.md) · [nn-library.md](nn-library.md) · [losses-and-optimizers.md](losses-and-optimizers.md) · [inference.md](inference.md) · [training-backends.md](training-backends.md)

## Facts

- Training composes three `[Module]` graphs — a **model**, a **loss**, and an **optimizer** —
  accessed via their `.ComputationGraph` property.
- `TrainingRig` is the entry point. It runs autodiff on the composed graph and produces a
  trainable step. You never write backward passes. A rig built with
  `trainingBackend: TrainingBackend.Native` leaves the gradient to the backend that runs the step —
  see [training-backends.md](training-backends.md).
- A step leaves the training state where it ran — on a GPU, the card's memory — so nothing
  crosses the bus between steps; reading a state tensor's values copies them to the host and
  leaves it there. `rig.BeginResidentRun()` (which `Fit` / `Train` use) also owns the state between
  steps — see [Keeping training state on the device](#keeping-training-state-on-the-device).
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
`Shorokoo.Modules.Losses` / `Shorokoo.Modules.Optimizers`) — see
[losses-and-optimizers.md](losses-and-optimizers.md) for the catalog (sixteen losses, thirteen
optimizers). Each optimizer whose hyperparameters are all tensor-shaped gets a source-generated,
named, defaulted set (`<Optimizer>Hyperparameters`) implementing `IOptimizerHyperparameters`; the
thirteen sets, and how each hyperparameter is bound, are in
[training-hyperparameters.md](training-hyperparameters.md).

A loss module has signature `(predictions, targets) -> Scalar<float32>` with exactly two tensor
inputs; targets are typically `Tensor<float32>`, but class-index losses (`CrossEntropyLoss`,
`NLLLoss`) take `Tensor<int64>`. The library losses' knobs (`reduction`, `ignoreIndex`,
`labelSmoothing`, class `weight`/`posWeight`, SmoothL1 `beta`) live on extra
`Reduced`/`PerElement` methods, not on the rig-bound `Inline`. To use them in a rig, write a
2-input wrapper `[Module]` whose `Inline` calls `Reduced(...)` with the knobs baked; a class
`weight`/`posWeight` must be **baked as a graph constant** there. See
[Losses → Configurable knobs](losses-and-optimizers.md#loss-configurable-knobs).

<a id="loss-ignoring-targets"></a>
**A loss graph may ignore its `targets`.** Two inputs is a signature requirement only: the rig
checks whether the loss output reaches input 1 (through data edges and branch conditions) and
derives the target slot from that. So a model that computes its **own** loss can be trained with a
pass-through loss. Use this when the loss needs more than one predictions and one targets tensor —
label ids, a padding mask, per-token weights, `Reduced`/`PerElement` knobs: pass those as ordinary
**model** inputs (see [Which knobs reach the rig](losses-and-optimizers.md#loss-configurable-knobs)) and make
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
[Custom optimizers](training-hyperparameters.md#custom-optimizers).

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
    TrainingCheckpoint? initialCheckpoint = null,  // defaults to CreateInitialCheckpoint()
    Action<TrainingStepReport>? onStep = null,     // see "Stopping and watching a run"
    CancellationToken cancellationToken = default);

// The loader owns the batch stream; step / epoch / batch advance for you.
public TrainingResult Fit(
    IDataLoader loader,
    int numEpochs,
    TrainingCheckpoint? initialCheckpoint = null,  // defaults to CreateInitialCheckpoint()
    Action<TrainingStepReport>? onStep = null,
    CancellationToken cancellationToken = default);

// The array Fit with the checkpoint first and required (not interchangeable argument order).
public TrainingResult Train(
    TrainingCheckpoint initialCheckpoint,
    TensorDataStruct[] trainingInputs,
    TensorDataStruct[] trainingOutputs,
    int numEpochs,
    Action<TrainingStepReport>? onStep = null,
    CancellationToken cancellationToken = default);

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

    // One step that also hands out the state as a checkpoint, left where the step put it.
    public TrainingCheckpoint StepToCheckpoint(IData trainingInput, IData trainingTarget);
    public TrainingCheckpoint StepToCheckpoint(IData hyperparameters,
                                               IData trainingInput, IData trainingTarget);
    public TrainingCheckpoint StepToCheckpoint(IDataLoader loader);
    public TrainingCheckpoint StepToCheckpoint(DataBatch batch);

    // The current state as a checkpoint, between steps: no step, no copy.
    public TrainingCheckpoint TakeCheckpoint();

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
([tensors-in-a-run.md](tensors-in-a-run.md#feeding-a-run-consumed-shared-or-tried)): what it is given as it is,
it **consumes** — dead once the step starts, memory returned as the step returns — and what it is
given `.Shared()` it only reads.

```csharp
cp = rig.TrainStep(cp, x.Shared(), y.Shared());    // a batch fed again: read, and kept
cp = rig.TrainStep(cp, x, y);                      // cp's state, x and y are consumed
var next = rig.TrainStep(best.Shared(), x2, y2);   // a checkpoint kept past the step: read
```

**What a step keeps of what it reads.** A tensor the run cannot address in place (any tensor built
from a C# array, and a host tensor on a card) is read through a copy that the tensor keeps for
reuse ([tensors-in-a-run.md](tensors-in-a-run.md#feeding-a-run-consumed-shared-or-tried)). A training step
releases the copies of its **batch** as it returns, success or failure, so a `.Shared()` dataset is
not held a second time (on a card, not uploaded whole); each step copies its batch afresh. Copies of
the **checkpoint's state** are kept: a checkpoint fed `.Shared()` on a card keeps a copy on the card
while it lives, so drop kept checkpoints when done. In a resident run, a checkpoint it does not own
(the one you began from `.Shared()`, or one `StepToCheckpoint` or `TakeCheckpoint` handed out) is
read by one step only, which releases its copies.

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
  itself; `To`, `CopyTo`, `ToHost` and `rig.AdoptCheckpoint` keep each field's mode, and a
  checkpoint's `ToHost()` keeps its `FeedMode`.
- **Runtime hyperparameters** are fed like a batch. `MakeHyperparameters` builds a fresh struct and
  copies the tensors given to it.
- **`Fit` and `Train` over arrays read their batches** and feed the initial checkpoint like
  `TrainStep`. `Fit` over a loader feeds each batch as the loader built it: `InMemoryDataLoader`
  builds a fresh batch per draw, which is consumed; a custom loader that reuses its tensors passes
  them `.Shared()` in its `DataBatch`.
- **A step that fails after starting has still consumed its as-is inputs.** A resident run then
  refuses every later step — see [below](#keeping-training-state-on-the-device).

### Keeping training state on the device

A step's outputs stay where the step ran, so `TrainStep` returns a checkpoint whose state is in
the run's memory — on a GPU, the card's — and the next step reads it there: the state does not
cross the bus between steps. `BeginResidentRun` keeps the state there too, and also owns it between
steps, releasing each step's state as the next one supersedes it; its checkpoints keep the state
where it is:

```csharp
using var run = rig.BeginResidentRun();
for (int step = 0; step < 50_000; step++)
{
    if (step % 1_000 == 999)
        run.StepToCheckpoint(loader).Save($"ckpt-{step}.safetensors");  // saved from the card
    else
        run.Step(loader);                                               // loss only
}
```

- **`Step`** returns only the loss.
- **`StepToCheckpoint`** runs a step (use it *instead of* `Step`) and hands out the state as an
  ordinary `TrainingCheckpoint`, its tensors left where the step put them: on a GPU, the card's
  memory. Nothing is copied to the host.
- **`TakeCheckpoint`** hands out the current state as a checkpoint between steps, without running a
  step and without copying; before the first step it is the checkpoint the run began from. This is
  how a loop that stops (cancelled, or satisfied) keeps what it trained:

  ```csharp
  using var run = rig.BeginResidentRun();
  for (int i = 0; i < steps && !satisfied; i++) run.Step(loader);
  TrainingCheckpoint checkpoint = run.TakeCheckpoint();  // no step, no copy: on a GPU, on the card
  checkpoint.Save("ckpt.safetensors");                    // written straight out of device memory
  TrainingCheckpoint onHost = checkpoint.ToHost();        // a copy of the whole state in host memory
  ```
- **A checkpoint handed out is yours.** The run goes on training from it but only reads it, and
  `Dispose` leaves it alone. A later step therefore writes its new state beside it rather than over
  it: on a card that is a second copy of the state for as long as you hold the checkpoint. A
  checkpoint you drop is freed by the collection the rig triggers once the state dropped since the
  last one exceeds its budget for superseded state: 32 MiB, doubling up to 8 GiB while the rig finds
  you keeping your checkpoints ([What construction costs](#what-construction-costs)). Until then it
  stays beside the run's own state. So a loop that saves and drops a checkpoint every N steps holds,
  besides the run's state, at most that budget of dropped state: a state larger than the budget is
  freed by the step after the checkpoint, so two copies are held only in between, while the dropped
  copies of a smaller state add up to the budget before they are collected.
- **`Dispose`** discards what the run still holds; checkpoints it handed out stay valid.
- **Each step consumes the state the previous step produced**, writing over it where it can
  ([below](#a-step-writes-its-state-over-the-state-it-consumed)). The starting checkpoint is
  consumed unless passed `.Shared()`; the default is a fresh `CreateInitialCheckpoint()`.
- **A failed step can end the run.** If it fails after consuming the run's own state, every later
  step throws `InvalidOperationException` saying so and what is left to restart from. Restart from
  the last checkpoint you took with `StepToCheckpoint` or `TakeCheckpoint` (a failure while reading a
  checkpoint handed out leaves it and the run intact), or, before any, from the starting checkpoint
  if you passed it `.Shared()`.
- **Feeding a published checkpoint elsewhere as-is consumes the run's state too**, since they share
  tensors; the run's next step is then refused with that state's error, naming the step that took it.

`Train` and every `Fit` overload drive a resident run internally and take the result with
`TakeCheckpoint` after the loop, so `TrainingResult.FinalCheckpoint` holds its state where the last
step left it — on a GPU, device memory. A manual `TrainStep` loop leaves its state there too. On a
backend with no device memory, a resident run gives the same losses and checkpoints, to the bit, as
a step loop.

> Saving, resuming, training from and reading a checkpoint work wherever its state is. Reading a
> tensor's values (`CopyMemory`, `ValueAt`, `AccessMemory` and the other accessors) copies them to
> the host and leaves the tensor where it is. `checkpoint.ToHost()` copies each tensor the host
> cannot read into host memory of its own, and is the same checkpoint where every tensor already is
> host-readable (always on a CPU backend).

### Stopping and watching a run

`Fit` and `Train` take a `CancellationToken` and a step callback. Both act **between** steps, never
during one: a step takes the state it trains from when it starts, so one abandoned part-way would
leave nothing whole to return, while the state between two steps always is.

- **`cancellationToken`** is checked before each step (and, for `Fit(loader)`, before the loader
  draws). The step running when it is cancelled finishes; the run then returns the state after it,
  with `StopReason == TrainingStopReason.Cancelled`. Nothing is thrown.
- **`onStep`** is called after every step, in order, on the training thread, with a
  `TrainingStepReport`: `Step`, `Epoch`, `BatchIndex`, `Loss`, `Elapsed` and the full `Entry` (the
  step's [history](training-data.md#the-training-history) entry). All of it is already on the host, so watching
  costs no transfer, and a run given no callback builds no report.
  - `report.RequestStop()` ends the run after this step (`StopReason.StopRequested`, or
    `Completed` when it was the last step anyway), e.g. for early stopping on a validation metric.
  - `report.TakeCheckpoint()` returns the state after this step as a checkpoint, without a copy (see
    `TakeCheckpoint` above), to save it or evaluate it while the run goes on.
  - A report is valid only during its callback; acting on a kept one throws.
  - A callback that throws ends the run with a `TrainingCallbackException`: its `InnerException` is
    what the callback threw, and its `Checkpoint` is the state after that step — whole, since the
    callback runs between steps — so the training done is kept. Passing it back resumes at the next
    batch, as from a run stopped with `RequestStop()`.
- **The result** is the state after the last step taken, whatever ended the run, with its data
  position. Passing it back resumes at the next batch exactly as a completed run's would, for the
  loader and the array forms alike; `numEpochs` counts from the resume epoch (see
  [Feeding data](training-data.md#feeding-data-the-data-loader)). To finish a stopped run at the end it was started
  with, call `rig.FitUntilEpoch(loader, untilEpoch, checkpoint)` instead: it trains until the loader
  reaches `untilEpoch`, counted from the start of training, so a restarted host makes the same call
  every time, and one whose checkpoint already got there trains nothing.
  `EpochLosses` covers the epochs the run trained in, a partial one averaged over its steps.

```csharp
// A BackgroundService, or SIGTERM on a spot instance: stop cleanly and keep the progress.
var result = rig.FitUntilEpoch(loader, untilEpoch: 10, resumeFrom,
    onStep: r =>
    {
        if (r.Step % 1_000 == 999) r.TakeCheckpoint().Save($"ckpt-{r.Step}.safetensors");
        if (float.IsNaN(r.Loss)) r.RequestStop();
    },
    cancellationToken: stoppingToken);
result.FinalCheckpoint.Save("last.safetensors");   // resumable, whether or not it was stopped
stoppingToken.ThrowIfCancellationRequested();
```

The runtime context's own `RunSettings.CancellationToken` is different: it abandons the step
running, which consumes the run's state, so the run ends with nothing newer than the last checkpoint
you saved.

### A step writes its state over the state it consumed

ONNX Runtime holds every run input until the run ends, so consumed memory cannot be reused
mid-step. Instead the step writes each updated state field (weight, optimizer moment) into the
memory of the field it replaces, wherever the graph proves every reader of the old value runs
before the write. The backend re-checks this on the graph ONNX Runtime actually runs.

It applies:

- **To consumed state** — a checkpoint fed as it is, and a resident run's own state. A `.Shared()`
  checkpoint is left untouched.
- **Where the old state is consumed** — every step on either backend: the new state is written
  into the consumed state's memory where it is, on the card on a GPU. A host checkpoint consumed by
  a step on a GPU is copied onto the card first, and the new state is written into that copy. A
  resident step overwrites nothing while you hold a checkpoint the run handed out, since that state
  is only read.
- **Where the graph proves it.** Optimizer state read only by its own update qualifies (e.g. AdamW's
  moments and step counter). A weight the backward pass also reads — to propagate a gradient to the
  layer below, or at each place a tied weight is used — qualifies once its update is ordered after
  those reads. The rig's memory-aware pass orders it when Shorokoo computes the gradient itself
  (`TrainingBackend.Shorokoo`, the default; see [training-backends.md](training-backends.md)) and
  the pass's objective prefers the ordered step. A weight is written anew when something
  reads it after its update, or when the pass declines to order the update. For a stack of `Linear`
  layers under AdamW every weight, bias and moment is written over.

On JAX the consumed state is **donated** to XLA, which writes each updated field over a donated one
of its shape wherever its own analysis allows, whatever the training backend; state the step only
reads is copied in the run's memory first and the copy donated, so a checkpoint you hold is never written
over ([jax-backend.md](jax-backend.md#runs)).

On PyTorch the step goes further: the element-wise arithmetic leading to each new state value —
AdamW's chain to a new moment, say — is written over the consumed state it reads last, so the
optimizer's temporaries take no memory of their own either
([pytorch-backend.md](pytorch-backend.md#runs)).

Results are bit-identical with or without it, and there is nothing to configure. On a card a
resident run thus holds its state once rather than twice; under a device-memory budget the state
is counted once — see [gpu-backends.md](gpu-backends.md#a-contexts-device-memory-budget). A step's peak
falls by what of the new state a step writing it beside the old would hold at its busiest, which
is not always all of it. The update's temporaries take memory of their own on ONNX Runtime either
way, and under AdamW it frees one of them before it writes a weight's new value, so a step over
one large weight peaks lower by the weight's two moments rather than by its whole state.

### What construction costs

`FromScratch` concretizes, composes the loss, runs autograd, lowers the optimizer, and runs shape
inference and graph optimization, all on `MergeContext`. It also runs each trainable parameter's
initializer and each optimizer-state initializer per parameter. Parameters with the same
initializer and shape share one initialization session, and optimizer-state initializers share one
per parameter dtype, so a model whose layers repeat builds a handful of sessions however
deep it is; what grows with the parameter count is the drawing itself. On a CPU context the draws
run side by side, each on a single thread, as many at once as there are cores and as fits
comfortably in memory — unless one parameter holds more than half of the model's elements, when
they run one after another, each over every core, as they do on a card. The values are the same
either way. A random initializer draws a bounded chunk of values at a time, so peak host memory is
the values produced so far plus, per draw in progress, a few copies of that parameter's value and
one chunk's working memory.

`TrainingRig.Load` repeats all of this except concretization (it reads the saved architecture) and
the model's initializers: it uses zero stand-ins of the declared shape, since the checkpoint
overwrites them. The first call that needs initial *values* — `CreateInitialCheckpoint()`, or a
load that falls back on the rig for an omitted component — runs the initializers then, with the
values an eager build would produce, and re-seeds the optimizer state. On the same build of the
same backend, `Load` takes the answers the backend's model of a run gave the saved rig's
memory-aware pass from the checkpoint instead of asking it again, and so chooses as that build did;
anywhere else it asks anew.

The first `TrainStep` at each input shape compiles the step graph and caches it, so expect the first
step to be much slower. Neither cost recurs during the loop or scales with the dataset.

Steady-state memory is flat: consumed state is freed as each step returns. Checkpoints you drop but
did not let a step consume (fed `.Shared()`, or `.TryConsume()` while shared) hold backend memory
behind small managed handles, so the rig triggers a garbage collection once more than 32 MiB of
such state has accumulated across steps, and backs off (doubling the threshold, up to 8 GiB) while collections
reclaim nothing, e.g. when you keep every checkpoint. You need not collect yourself. Initial
checkpoints count against the same budget. A resident run's `Step` bypasses it, but a checkpoint
the run hands out — from `StepToCheckpoint` or `TakeCheckpoint` — counts once a later step has
moved on from it, and the collection it triggers runs after that step, when the run no longer holds
it, so a checkpoint you have dropped by then is freed by it.

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

Each context carries its backend, its `DeviceMemory` (a device-memory budget), its `Precision`
(`float32` in full precision unless it allows TensorFloat-32 on a card — see
[Precision](gpu-backends.md#precision-gpu-backends)) and `RunSettings` — see [Device memory](gpu-backends.md#device-memory-gpu-backends). So a rig can
build on one device and train on another:

```csharp
var rig = TrainingRig.FromScratch(
    model, loss, optimizer,
    sampleInputs, new AdamWOptimizerHyperparameters { LearningRate = 0.001f },
    mergeContext:   new ComputeContext(new LinuxCpuBackend()),
    runtimeContext: new ComputeContext(new LinuxGpuBackend()));
```

Both devices must be reachable from one process, and merge-phase output reaches the runtime
backend by a host copy per feed — see [One model, two devices](backends-and-devices.md#one-model-two-devices).
Split only when the build does not fit on the card; leaving both `null` is normal.
`rig.MergeContext.Backend` / `rig.RuntimeContext.Backend` name the devices;
`DefaultBackend.RequireDevice(...)` refuses the wrong one — see
[Which device am I on?](backends-and-devices.md#which-device-am-i-on).

On GPU backends the runtime context's budget covers the state and batches it holds on the card plus
what the running step's session allocates, which is held to what the state and batches leave it —
see [A context's device-memory budget](gpu-backends.md#a-contexts-device-memory-budget). The rig keeps a
compiled step for up to four input shapes; further shapes share one shape-generic step.

A context's settings are fixed at construction: budget, precision, `ShrinkArenaAfterRun` (implied by a budget), and the `CancellationToken` that abandons the step running — see
[Stopping a run](inference.md#stopping-a-run); to stop a `Fit` or `Train` and keep its progress, pass
its own `cancellationToken` instead ([Stopping and watching a run](#stopping-and-watching-a-run)). Put them on the `runtimeContext` you pass to
`FromScratch`. To watch a run near the card's limit, read the static `DeviceMemory` class and the
context's `ReadDeviceMemoryUse()` inside your loop.

**Host vs device memory.** The rig's collection governs the memory behind a `TrainStep` loop's
dropped checkpoints, on the host or on the card; `DeviceMemory` settings govern only device memory.
A resident run keeps state on the card and frees it deterministically.

Result types:
- `TrainingCheckpoint`:
  - `.TrainableParams`, `.ModelState`, `.OptimizerState`.
  - `.Step` (`long`, global step, advanced by each step; restored on load so schedules resume).
  - `.Epoch` / `.BatchIndex` (`long?`): set by the loader-driven and explicit-counter paths, carried
    through unchanged by the counter-agnostic `TrainStep`, `null` when unknown (a scheduler reading
    them sees `0`). All counters are `int64`.
  - `.Rig` (the producing `TrainingRig?`, so `ToInferenceModel()` needs no graph).
  - `.Loss` (`float?`; `null` on an initial or bare checkpoint; saved as its own `Loss` component).
  - `.AppliedHyperparameters` (see [Hyperparameter kinds](training-hyperparameters.md#hyperparameter-kinds-hyperparameter)).
  - `.History` (see [The training history](training-data.md#the-training-history)).
  - `.ToHost()`: the checkpoint with its state in host memory, copying each tensor the host cannot
    read; the same checkpoint where every tensor already is host-readable. Every other slot carries
    through.

  `WithCounters`/`WithStep`/`WithEpoch`/`WithBatchIndex` and
  `WithTrainableParams`/`WithModelState`/`WithOptimizerState` return a new checkpoint with one slot
  replaced and everything else carried; the receiver is never mutated.
- `TrainingResult` → `.FinalCheckpoint` (its state where the last step left it; on a GPU, device
  memory), `.EpochLosses` (per-epoch mean losses).

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
step, including across a save and resume. On a GPU they are not by default: the card's kernels
add up partial results in whatever order its threads finish, so two runs from one seed can differ
in the last digits of the loss from early steps, with or without a resume in between. To
reproduce a GPU run bit for bit — to verify that a resume on the card is exact, say — train on a
context that asks for deterministic compute, and load the checkpoint onto one too:

```csharp
static ComputeContext Deterministic() =>
    new() { Diagnostics = new DiagnosticSettings { DeterministicCompute = true } };

var rig = TrainingRig.FromScratch(model, loss, optimizer, sampleInputs, hyperparameters,
    runtimeContext: Deterministic());
// ...
var (resumedRig, resumed) = TrainingRig.Load("run.skpt", runtimeContext: Deterministic());
```

ONNX Runtime then runs the deterministic CUDA kernel of every operator that has one, which can
cost the card some speed. An operator with no deterministic kernel runs the one it has, so its
results can still vary; ONNX Runtime names only some of them in its log.
The setting changes nothing on the CPU backends, which are reproducible either way, and the
PyTorch and JAX backends do not apply it. Pass `new RngConfig { MasterSeed = … }` to re-roll all
streams coherently, or `RngConfig.NonDeterministic()` for per-run variation.

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
| `SaveReport` | A save's `BytesWritten`, `Write` / `Flush` / `Commit`, `Elapsed`, `BytesPerSecond`. | Returned by every single-file checkpoint save — see [What a save costs](training-checkpoints.md#what-a-save-costs). |
| `Schedule` (namespace `Shorokoo.Core.Training`) | A `step → value` schedule; assigning one makes a hyperparameter [`Scheduled`](training-hyperparameters.md#hyperparameter-kinds-hyperparameter). | A `Schedules.…` factory plus combinators (`WithWarmup`, `Then`, `Scale`, `Clamp`, `Shift`, `PerEpoch`). Preview with `.At(step)`. |
| `Schedules` (static, namespace `Shorokoo.Core.Training`) | Factories: `Constant`, `Linear`, `Cosine`, `CosineWithWarmup`, `StepDecay`, `Exponential`, `OneCycle`. | `Schedules.Cosine(1e-3f, totalSteps)` — see [Schedule factories and combinators](training-hyperparameters.md#schedule-factories-and-combinators). |

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
- Do not expect a resident run's state after disposing it; take it with `TakeCheckpoint` or
  `StepToCheckpoint` first.
- Do not declare optimizer state as `Inline` parameters.
- Do not call `Globals.StateUpdate` on inputs, trainable parameters, or computed tensors — only on
  a `[StateInitializer]` `Init` result.
- Do not call `Globals.StateUpdate` outside a module body — it throws. Inside a `LoopAPI.Iterate`
  body it registers the updated tensor's post-loop value (that state's one update for the step);
  the value must be a carried loop variable (assigned in the body and read across iterations; with
  zero iterations it is its pre-loop value). A value that never leaves the loop, a scanned result,
  or an iteration-scoped value (e.g. the iteration index) fails the module build.
