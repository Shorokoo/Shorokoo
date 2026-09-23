# Running models (inference)

Related: [core-types.md](core-types.md) · [defining-models.md](defining-models.md) ·
[onnx-and-weights.md](onnx-and-weights.md)

## Facts

- `OnnxEngine.Eval(...)` is the simplest way to get values. It builds an ONNX model
  from the graph, runs it once via OnnxRuntime, and returns `TensorData`.
  - `TensorData Eval(Variable output)`
  - `TensorData[] Eval(Variable[] outputs)`
  - `TensorData[] Eval(Variable output1, Variable output2, params Variable[] outputs)`
  - The parameter type is `Variable`, which every op result converts to implicitly.
    An `IValue`-typed handle does not, and needs `handle.ToVariable()` — see
    [`Variable` and `IValue`](core-types.md#variable-and-ivalue).
  - `Eval` runs a graph of plain ops only. Nothing on its path lowers a
    module-invoke node, so it checks the graph it builds and refuses a `[Module]`
    output up front, naming the lowering that fixes it. Concretize it first — see
    [Running a `[Module]`](#running-a-module).
- `OnnxEngine.Eval` rebuilds and recreates an ORT session on every call. For repeated
  inference, compile once with `ComputeContext` (below).
- Reference one platform backend package and it is normally found for you — no setup
  code. A program that names none gets one, and two of them for one OS is refused rather
  than guessed at. How it is discovered, and how to override the choice:
  [Backend selection](#backend-selection).
- A program that *does* name them can run several at once: give a `ComputeContext` a
  backend and its work goes there, so one process drives the CPU and the card together —
  [One model, two devices](#one-model-two-devices).
- Which device the work will run on is invisible at the call site but answerable:
  `ComputeContext.Backend` and `DefaultBackend.Describe()` name it, and
  `DefaultBackend.RequireDevice(...)` refuses to start on the wrong one —
  [Which device am I on?](#which-device-am-i-on).
- **A tensor fed to a run as it is is consumed by that run**: dead once the call is made, and
  its memory the run's backend's, released before the call returns whatever the run did — or,
  where the graph proves nothing reads it after, written over with one of the run's outputs
  ([A run that writes an output into what it consumed](#a-run-that-writes-an-output-into-what-it-consumed)).
  Pass it `.Shared()` to have the run only read it, so you can use it again, or `.TryConsume()`
  to have it consumed only when nothing else is reading it. Structs, sequences, optionals and
  training checkpoints take both —
  [Feeding a run: consumed, shared or tried](#feeding-a-run-consumed-shared-or-tried).
- A `TensorData` is its memory — one object per allocation, released through the backend that
  made it — and it ends in exactly three ways: deleted, consumed by a run, or moved into an
  attribute. A run holds what it is reading for as long as it runs, so a tensor being read
  cannot be deleted — [A tensor's lifetime](#a-tensors-lifetime-locks-and-deletion).
- A compute context keeps books on tensors and owns none: disposing it releases its sessions and
  leaves every tensor alive. `To(context)` hands a tensor to a context that can read it where it
  is and copies it otherwise, `CopyTo(context)` always copies, and `ToHost()` brings one within
  the host's reach — [Moving data between contexts](#moving-data-between-contexts).
- An input large enough to dominate the step's peak need not exist twice: allocate it on the
  context and fill the runtime's own buffer in place, then feed it as it is — the run consumes it
  where it is, and its memory is released as the run returns —
  [Feeding a large input without a second copy](#feeding-a-large-input-without-a-second-copy).
- On a GPU backend a `ComputeContext`'s `DeviceMemory` is a **budget on what the context holds on
  the card** — the tensors attached to it there, and the arena of whichever of its runs is
  executing — and the arena settings of the sessions it compiles; `RunSettings` says what its runs
  do, and the separate static `DeviceMemory` class reports how much of the card is gone. A transfer
  the budget cannot take is refused, and a run's arena gets what the attached tensors leave:
  [A context's device-memory budget](#a-contexts-device-memory-budget). The arena strategy departs
  from exact-size extension only for a session Shorokoo knows is reused across differing shapes, so
  a long training loop does not end up holding far more of the card than it uses:
  [Device memory](#device-memory-gpu-backends).
- What a *session* holds, what a *run* peaked at, and whether a GPU run quietly did some of its
  work on the host are all answerable, and all off by default:
  `CompiledGraph.ReadArenaStatistics()` reads one session's own allocator and
  `CompiledGraph.ReadPinnedArenaStatistics()` the pinned host memory its crossings went through,
  `ComputeContext.RunStats` folds every run the context makes into exact aggregates plus a bounded
  window of per-run detail, and `CompiledGraph.OutputPlacement` costs nothing while
  `CompiledGraph.ReadNodePlacement()` names the nodes that fell back —
  [What one session's arena did](#what-one-sessions-arena-did) onwards.

## Workflow: one-shot evaluation

`Eval` takes the output values of a graph of plain ops and runs that graph:

```csharp
using Shorokoo;
using static Shorokoo.Globals;
using static Shorokoo.NN;

var input = TensorFill(Vector(1L, 3L, 224L, 224L), 0.1f);
var w     = RandomNormal(Vector(64L, 3L, 7L, 7L));
var b     = VectorFill(64L, 0f);

var features = Conv(input, w, b, AutoPad.NotSet,
                    dilations: [1L, 1L], group: 1L,
                    kernelShape: [7L, 7L], pads: [3L, 3L, 3L, 3L],
                    strides: [2L, 2L]).Relu();

TensorData result = OnnxEngine.Eval(features);

// Read the numbers out (see core-types.md):
ReadOnlySpan<float> values = ((TensorData<float32>)result).AccessMemory();
```

What `Eval` accepts is the trap here:

- **Op results, yes.** Anything implicitly convertible to `Variable` —
  `Tensor<T>`, `Vector<T>`, `Scalar<T>` — which is what every op returns.
- **An `IValue`-typed handle, no.** `Variable` does not implement `IValue`, so
  `Eval(handle)` does not compile for a variable declared `IValue`; write
  `Eval(handle.ToVariable())`. See
  [`Variable` and `IValue`](core-types.md#variable-and-ivalue).
- **A `[Module]`'s output, no.** `Eval` runs the graph as handed to it, so a value
  coming out of `ResNet50.Call(...)` still carries its un-lowered module-invoke
  node. `Eval` detects that before building anything and throws an
  `InvalidOperationException` naming the fix; lower the module's
  `ComputationGraph` first — see [Running a `[Module]`](#running-a-module).
  (`ResNet50` there is from [`samples/RetinaNet`](../samples/RetinaNet) — a
  sample built on Shorokoo, not part of the packages.)

Build the input from a real array (not just a constant fill) with the `params`
overload — the first arg is the shape, the rest are the flat values:

```csharp
var input = TensorData([1L, 3L, 224L, 224L], myPixelFloatArray); // float[] of length 1*3*224*224
```

For multiple outputs:

```csharp
TensorData[] outs = OnnxEngine.Eval(out1, out2, out3);
```

## Running a `[Module]`

`OnnxEngine.Eval` runs a graph of plain ops. A `[Module]`'s output (from
`Foo.Call(...)` or `Foo.Model().Call(...)`) can still carry an un-lowered
module-invoke node, and every eager-evaluation entry point — `OnnxEngine.Eval`,
`ComputeContext.Eval`, `tensor.Eval()`, `inputs.Eval(outputs).With(...)` — scans
the graph it builds and refuses such an output with an
`InvalidOperationException`, whatever the module's parameters are initialized
from — for example:

> `OnnxEngine.Eval requires a concretized graph (a 'concrete-architecture' or
> 'concrete-model'), but this graph is a 'module'. It still carries module machinery
> that lowering removes (ShrkCreateModule, ShrkModelInvoke, ShrkModuleSetHyperparams).
> It comes from module 'ResNet50': lower that module's ComputationGraph the whole way
> — ToConcreteArchitecture(inputHints) then ToConcreteModel() — and execute that,
> passing a value for each of its inputs in order ([Hyper] parameters come first). …`

It names the module your value came from, and the one thing easy to get wrong: the
values go in the graph's input order, `[Hyper]` parameters first. The same refusal
comes from `ComputeContext.Execute`/`Run`/`Compile` when the graph handed to them is
a module — so the mistake reads the same whichever way you make it.

A graph whose module machinery sits inside a function body used to slip past this and
fail later, with OnnxRuntime rejecting the model for an op it has no kernel for
(`No Op registered for ShrkCreateModule`). Those bodies are now lowered on the way out,
so an initializer that calls a module — or a call to a module-typed function — exports
and runs like any other, including when the call sits inside a loop.

A callee that owns a trainable parameter lowers too, with one thing worth knowing: a
function body is not a model, and nothing will ever feed a weight into one, so a
parameter such a callee owns is written as the value its own initializer computes
rather than as a trainable weight of the model. An initializer that calls a layer
therefore initializes from that layer's *initial* weights; the layer contributes no
parameter of its own to the model it is called from. This holds however the callee
was reached — a direct call, or a model taken out of a `ModelSequence`.

Reading a parameter through `IModel.GetTrainableParam` from such a body is the one
shape still left out, and it fails at session creation on an operand nothing
produces ([#318](https://github.com/Shorokoo/Shorokoo/issues/318)). A bare reference
like that names a parameter defined elsewhere instead of carrying its own
initializer, and the emitted body has no way to match the two up; call the module
and use its result instead.

Concretize the module's `ComputationGraph` against the input first, then execute:

```csharp
using Shorokoo;
using Shorokoo.Graph;     // Specialize / ToConcreteArchitecture / FromOrderedInputs / ToConcreteModel
using Shorokoo.Runtime;   // ComputeContext
using static Shorokoo.Globals;

var input    = TensorData([4L], 1f, 2f, 3f, 4f);   // the actual input data
var graph    = MyLayer.ComputationGraph;            // readonly ComputationGraph (kind: Module)
var concrete = graph
    .ToConcreteArchitecture(graph.FromOrderedInputs([input]))
    .ToConcreteModel();

var results = ComputeContext.Default.Execute(concrete, input);   // params IData[]
float[] values = results[0].ToTensorData().As<float32>().AccessMemory<float>().ToArray();
```

When the graph comes from a saved `.srk`/`.zsrk` file, you can catch this mismatch
at load time instead: v2 files record their lowering stage in the header, and
`LoadFastGraphFromFile(path, requiredStage: GraphKind.ConcreteModel)` refuses a
module-stage file with a clear stage-mismatch error — see
[onnx-and-weights.md](onnx-and-weights.md#the-srk-container).

### The lowering pipeline

Turning a `[Module]`'s `ComputationGraph` into a runnable model is a three-step
pipeline, applied in order:

1. **`Specialize(values)`** — *optional.* Bakes a partial set of named inputs
   (typically `[Hyper]` parameters) into constants and folds them through the
   graph, dropping them from the input list. Skip it if you want those inputs to
   stay live. Returns a copy; the original is untouched.
2. **`ToConcreteArchitecture(inputHints)`** — inlines every sub-module and
   function so trainable parameters become visible at the top level, and uses
   `inputHints` to resolve shape-dependent parameters.
3. **`ToConcreteModel(...)`** — binds parameter values (loaded weights, or the
   initializer defaults when called with no argument) into the architecture.

The simple example above has no hypers to bake, so it skips straight to step 2.
The next section shows step 1 in use.

Every `ComputationGraph` carries a reliable **`Kind`** property saying where it
sits in this pipeline — `GraphKind.Module`, `GraphKind.ConcreteArchitecture`, or
`GraphKind.ConcreteModel` — stamped by the step that produced it (and preserved
through copies and `.srk` save/load). The steps check it up front:
`ToConcreteArchitecture` requires a `Module` graph, `ToConcreteModel` a
`ConcreteArchitecture`, and export/weight-query operations name the actual vs
required kind in their error when handed the wrong stage — so a mis-ordered
pipeline fails immediately with a clear message instead of deep inside execution.
Execution (`ComputeContext.Execute`/`Run`/`Compile` and `QuickExecutionEngine`)
likewise refuses a module-kind graph up front with the same lowering hint. Because
`WithKind` and `FromInternal` can stamp a graph the caller's way, `ComputeContext`
does not rest on the stamp alone: it also checks the ops themselves before building
a session. `Eval` takes output values rather than a `ComputationGraph`, so it has no
`Kind` to read at all; it is that op check which refuses a module output handed to
it.
`ComputationGraph`s are **readonly**: operations that used to modify a graph in
place return a new graph instead (e.g. `WithRngConfig`), so a graph's `Kind` can
never be invalidated behind your back.

If a graph arrives with the wrong kind — a foreign import that op-scanning
misjudged, say — re-stamp it with
**`WithKind(kind)`**. The target kind is validated against the graph's content
(a module must not have initialized parameters; a concrete architecture
additionally needs a statically known parameter space; a concrete model needs
every parameter initialized), so a stamp that would lie about the graph is
refused with an error naming the violated requirement.

## Running a `[Module]` with `[Hyper]` parameters

A module's `ComputationGraph` lists its `[Hyper]` parameters as graph inputs
**before** the tensor inputs — the framework keeps the graph's inputs ordered
hyperparameters-first, independent of the inputs-first `Inline` source order — and
they stay inputs in the concretized graph. So both `FromOrderedInputs` and `Execute`
take the hyper values first, then the inputs:

```csharp
// [Module] Dense { Inline(Tensor<float32> x, [Hyper] Scalar<int64> outFeatures) ... }
var hyper = TensorData([], 10L);                  // outFeatures = 10
var input = TensorData([2L, 4L], myFloats);

var graph    = Dense.ComputationGraph;
var concrete = graph
    .ToConcreteArchitecture(graph.FromOrderedInputs([hyper, input]))  // hypers first
    .ToConcreteModel();

var results = ComputeContext.Default.Execute(concrete, hyper, input); // hypers first
```

The hyper value passed to `FromOrderedInputs` is what concretization bakes from.
A hyper that touches the trainable parameters — their shapes (like `outFeatures`),
or which of them exist at all (a `[Hyper]` gating an `IfElse` branch that holds
parameters) — is **parameter-space-determining**, and the value you pass here
fixes that part of the architecture for good; pass the same value at `Execute`
time. Value-only hypers (scale factors, ε's) are read live on every `Execute` and
may vary call to call. See
[defining-models.md](defining-models.md#hyperparameter-baking) for the
distinction, and [What concretization fixes](#what-concretization-fixes) below
for everything else the concretization values pin down.

### What concretization fixes

`ToConcreteArchitecture` produces an architecture whose **parameter space is
static** — every trainable parameter and other id-addressed component is
enumerated at that point, which is exactly what makes the graph
"concrete" and what lets weights bind by name, optimizers allocate their state,
and checkpoints round-trip. Anything derived from the values you hand it is
therefore fixed then and there:

| Fixed at concretization | Derived from |
|---|---|
| Trainable-parameter **shapes** and count | hypers feeding a parameter's shape, and the shapes of the sample inputs |
| **Which** trainable parameters exist | hypers gating an `IfElse` whose branches hold parameters |
| The per-iteration **parameters** realized over a `LoopAPI.Iterate` body (and the whole iteration space, when the count folds to a constant and the loop unrolls) | hypers/inputs that drive the count |

What concretization fixes is the **parameter space**, and it rewrites only as
much control flow as that requires. An `IfElse` whose unselected branch holds
parameters is resolved here and folded away — those parameters do not exist, so
the branch can never be taken, and keeping it would cost the bytes the pruning is
meant to save. An `IfElse` that holds no parameters is left alone, even on the
same hyper: both branches stay, and it still selects on its (still live) input at
run time.

Note which branch drives that. It is the **unselected** one: folding happens
because a branch's parameters were pruned, so an `IfElse` whose *selected* branch
holds the parameters keeps both branches and stays live. For the usual
`bit.IfElse(withParams, without)` shape that means the bit folds the `IfElse`
when baked **off**, and leaves it live when baked **on**. Only an `IfElse` that
*solely* owns the pruned parameters is folded: one sharing them with a second
`IfElse` is left alone, as is a tuple `IfElse` — its slots resolve together, and
the paramless ones must keep switching.

What decides the fold is the value you supply at concretization, not the `[Hyper]`
marker: the parameter space cannot depend on a value that only arrives at
`Execute`, so gating a trainable parameter on a plain runtime input is resolved
from the concretization value just the same. That is a reason to mark such a gate
`[Hyper]` — it makes a baked value look baked at the call site.

So one hyper can be half-resolved and half-live, and that is the intended split:

```csharp
var big = Zeros.Init([outFeatures]).Vec();       // a trainable parameter
var a = flag.IfElse(x * 10f, x * 100f);          // no params -> always stays live
var b = flag.IfElse(x + big, x);                 // holds big -> folded iff flag is baked false
```

Concretized with `flag = false`, `big` does not exist, `b`'s `IfElse` is gone
(so `b` is `x` whatever you pass later), and `a`'s still switches on every
`Execute`. Concretized with `flag = true`, `big` exists, nothing is pruned and so
nothing is folded, and **both** switch at run time.

These values stay **live inputs** of the concrete graph — concretization is not
`Specialize` and removes nothing from the input list — so you supply them again
at every `Execute`. The contract is that you supply **the same values**.
Executing with a value that would have produced a different parameter space is
**invalid use**: the parameters that answer needs were never created, and nothing
re-derives them at run time.

A **parameter gate** is half an exception, and which half depends on whether its
`IfElse` actually folded — so it is not a licence to pass whatever you like:

- **It folded** (single-output gate, exclusive owner, baked off): the value you
  pass later cannot contradict it. The baked branch runs whatever you supply, and
  passing the opposite value is pointless rather than dangerous.
- **It did not** (baked on, or a tuple or shared gate that could not fold): the
  `IfElse` is still live and the opposite value silently takes the other branch —
  skipping parameters that do exist when baked on, or reading a **zero stand-in**
  for parameters that were pruned when baked off.

So the rule is unchanged, and it is the second case that makes it matter: supply
the value you concretized with.

If you would rather make the contradiction impossible than remember the rule,
bake the hyper with [`Specialize`](#hardcoding-hypers-with-specialize) before
concretizing. It drops the input entirely, so passing a value for it at
`Execute` is then an input-count error rather than a silent wrong branch.

### Hardcoding hypers with `Specialize`

If you do not want to re-supply the hyper values on every `Execute` — i.e. you
want them *hardcoded* into the model — run `Specialize` first. It takes a
partial set of named input values, constant-folds them into the graph, and
removes them from the input list. The general process is then **`Specialize`,
then `ToConcreteArchitecture`, then `ToConcreteModel`**:

```csharp
var graph = Dense.ComputationGraph;                 // inputs: outFeatures, x

// 1. Bake the hyper(s). FromOrderedInputs pairs values with the leading input
//    names (hypers come first), so passing just the hyper value names it correctly.
var specialized = graph.Specialize(graph.FromOrderedInputs([hyper]));
//    `specialized` now has a single input: x.

// 2. + 3. Concretize on the remaining (runtime) inputs only.
var concrete = specialized
    .ToConcreteArchitecture(specialized.FromOrderedInputs([input]))
    .ToConcreteModel();

var results = ComputeContext.Default.Execute(concrete, input);   // no hyper needed
```

`Specialize` matches values to inputs **by name** (against the graph's
`InputNames`); names with no matching input are ignored. It returns a copy
and never mutates the original graph, exactly like `ToConcreteArchitecture`.
This works for any input, not just hypers — but baking a runtime input is
usually not what you want.

## Workflow: compile once, run many (repeated inference)

`ComputeContext` builds the ORT session once and reuses it.

```csharp
var ctx      = new ComputeContext();
var compiled = ctx.Compile(graph);                 // graph: a concretized ComputationGraph
var r1 = compiled.Execute(inputData1);             // params IData[] — the data goes here
var r2 = compiled.Execute(inputData2);             // reuses the session
```

Each input fed as it is is consumed by the run it feeds, so `inputData1` is dead once the first
call is made; pass it `.Shared()` there to use it again — see
[Feeding a run: consumed, shared or tried](#feeding-a-run-consumed-shared-or-tried).

`Compile(ComputationGraph graph)` takes the graph and nothing else — the data goes to
the `CompiledGraph` it returns, whose `Execute(params IData[] inputs)` is the call you
repeat. `ComputeContext` also offers `Eval(...)` (the `OnnxEngine.Eval` overloads,
plus `Eval<T>(Tensor<T>)` returning a typed `TensorData<T>`),
`Execute(ComputationGraph graph, params IData[] inputs)`,
`Run(ComputationGraph graph, params NamedModelParam[] inputs)`, and
`ExecuteWithState(...)` (for models that carry state). Wherever `IData` is asked for,
`TensorData` implements it, so pass `TensorData` values directly — as they are, to be consumed,
or through `.Shared()` or `.TryConsume()`. `Execute`, `Run` and
`CompiledGraph.Execute` return `NamedModelParam[]`; read each output with
`namedModelParam.ToTensorData()` then `AccessMemory()`. `ExecuteWithState` returns
`(NamedModelParam[] regularOutputs, ComputationGraph updatedGraph)` — feed the updated
graph to the next call. `Eval` is the exception: it returns `TensorData` (or
`TensorData[]`) directly.

### Stopping a run

A run can be given a `CancellationToken`, on the same `RunSettings` that carries
`ShrinkArenaAfterRun`, and the call then ends in an `OperationCanceledException` rather than
returning outputs:

```csharp
using Shorokoo.Core.Backends;

using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(30));
try
{
    var outputs = compiled.Execute(inputs, new RunSettings { CancellationToken = cts.Token });
}
catch (OperationCanceledException)
{
    // the run was stopped; it produced nothing
}
```

A token already cancelled when the call is made is refused before anything is fed: no input is
built into a runtime value, no feed is locked, and nothing passed as it is is consumed, so the
refused call spends nothing and can be made again with the same inputs. A run stopped after it
started has consumed what it was fed as it is, as a run that fails has — so a caller who means to
retry passes `.Shared()`.

One cancelled while the run is in flight sets ONNX Runtime's terminate flag, which its executor
reads **between nodes** — so what the wait costs is whatever is left of the kernel that was
running, not what is left of the run. Two consequences are worth planning around, and the probe
behind the figures below measures both (`TerminateLatencyProbeTests`, `Purpose=Manual`):

- **The wait tracks one kernel.** On a chain of matmuls the run came back within one kernel's
  duration of the flag being set, and by about as much whether a tenth or nine tenths of the run
  remained: 0.4-0.6 s on a chain whose kernels were 0.7 s, under 0.1 s on one whose kernels were
  0.1 s, single-figure milliseconds on one whose kernels were 2 ms — and below about 10 ms the
  floor is the waiting thread being scheduled again, not ONNX Runtime. So the figure to budget for
  is the model's **longest single operator**, not the step.
- **A graph whose work is one kernel cannot be stopped at all.** There is no boundary to stop at:
  a single 4096x4096 matmul flagged a tenth of the way through still ran the full 0.7 s and
  returned its outputs. The same holds for any graph's last kernel.

The figures are one four-core CPU box's, so read them as the shape rather than as your machine's
numbers; what settles your case is running the probe, or your own model, where you deploy.

Stopping is therefore best effort. A run that reaches the end before the flag is read **succeeds**,
and hands back outputs that are perfectly good — so treat a normal return as a normal return, and
do not take the absence of an `OperationCanceledException` as a sign the token was ignored. A
session is unharmed by having one of its runs stopped: the flag is on that run's options, not on
the session, and the next run of the same compiled graph proceeds normally.

**Where there is no per-call override, set it on the context.** Only `CompiledGraph`'s run entry
points take a `RunSettings` per call. The one-shot entry points that build a session for you —
`ComputeContext.Execute` / `Run` / `Eval`, and a training rig's `TrainStep`, `Train` and `Fit` —
run on the settings their context carries, so the token goes there:

```csharp
using var ctx = new ComputeContext(backend)
{
    RunSettings = new RunSettings { CancellationToken = cts.Token },
};

var outputs = ctx.Execute(graph, inputs);   // stops when cts does
```

`RunSettings` is a record and the property is `init`-only, so this is settled when the context is
built and nothing can change it under a run in flight. Giving a training rig's `runtimeContext`
one is how a long `Fit` is stopped — see
[Compute contexts](training.md#compute-contexts-mergecontext-and-runtimecontext).

## Backend selection

- Add a backend package as a dependency: `Shorokoo.LinuxCPU`, `Shorokoo.LinuxGPU`,
  `Shorokoo.WinCPU`, or `Shorokoo.WinGPU`. Each brings the native ONNX Runtime (CPU- or
  CUDA-flavored) for its platform. One is enough; a program that names its backends may
  reference several, or load them at runtime and reference none — see
  [Loading a backend at runtime](#loading-a-backend-at-runtime).
- With exactly one backend package referenced you normally need no setup at all:
  auto-discovery (below) finds it on the first run. Set the backend
  explicitly to name which one you mean when a deployment holds more than one (which
  discovery otherwise refuses), when you want a startup failure instead of one on the
  first run, or when the backend DLL is not deployed next to `Shorokoo.dll`:

  ```csharp
  using Shorokoo.Core.Backends;
  using Shorokoo.LinuxCPU;                                // the package you referenced

  DefaultBackend.Instance = new LinuxCpuBackend();
  ```

- `DefaultBackend.Instance` is the **default** backend: the one a `ComputeContext` that
  names no backend of its own runs on, and the one a tensor is built for when it is fed
  without a context naming another. The first backend resolved is cached and reused;
  assigning `Instance` afterwards swaps it but does not unload a native ONNX Runtime already
  bound, and does not reach a `ComputeContext.Default` that has already resolved — so assign
  it at startup, before anything runs.
- A `ComputeContext` constructed with a backend runs there instead, and two contexts may
  name different backends — that is how one process uses two devices. See
  [One model, two devices](#one-model-two-devices).
- **Exactly one deployed** is still the rule for *discovery*: a deployment carrying two
  backends for the same OS and naming neither is refused rather than resolved by
  guesswork — see [Auto-discovery](#auto-discovery). It is a rule about silence, not a
  limit on how many can run.
- Which backend you ended up on is a question you can ask — `DefaultBackend.Describe()`,
  or `ComputeContext.Backend` where the work is submitted. See
  [Which device am I on?](#which-device-am-i-on).

### The backend types

Each backend package contains exactly one backend, in a namespace equal to the package
id. **The type name spells the device `Cpu`/`Gpu`, while the package, namespace and
assembly spell it `CPU`/`GPU`** — so `Shorokoo.WinGPU` contains
`WinGpuBackend`, *not* `WinGPUBackend`:

| package (= namespace) | backend type | fully qualified |
|---|---|---|
| `Shorokoo.LinuxCPU` | `LinuxCpuBackend` | `Shorokoo.LinuxCPU.LinuxCpuBackend` |
| `Shorokoo.LinuxGPU` | `LinuxGpuBackend` | `Shorokoo.LinuxGPU.LinuxGpuBackend` |
| `Shorokoo.WinCPU` | `WinCpuBackend` | `Shorokoo.WinCPU.WinCpuBackend` |
| `Shorokoo.WinGPU` | `WinGpuBackend` | `Shorokoo.WinGPU.WinGpuBackend` |

All four implement `IShorokooBackend`, take a parameterless constructor, and
differ only in the execution provider
they configure: the GPU ones append the CUDA provider on device 0, the CPU ones leave
ORT on its default provider.

### Auto-discovery

If you never assign `DefaultBackend.Instance`, the first read of it resolves a backend
once and caches the result:

1. If one of the four backend assemblies is **already loaded** in the process, its
   backend is used — this avoids pulling a second native in alongside one already bound.
   Only assemblies targeting the running OS count, as in step 2, and only those that
   actually expose a backend; anything else falls through to step 2. A backend loaded by
   `IsolatedBackend.Load` is not a candidate at all: it lives in a load context of its own,
   and it is there because the program named it, so it is no answer to which backend a
   program that named none meant.
2. Otherwise the folder next to `Shorokoo.dll` is probed for the known
   `Shorokoo.{Platform}.dll` files, and only those targeting the current OS count as
   candidates. Nothing else is searched: no other directory, no NuGet cache, and no
   assembly whose name is not one of those four.

A single candidate is taken as-is — a lone GPU backend is chosen even when no CUDA
runtime is present. **Two or more are refused**, in either step, with an
`InvalidOperationException` naming them. From the folder probe (step 1 says `already loaded
in this process` in place of `deployed in '<folder>'`, and refuses only backends that
actually expose a backend):

> `Several Shorokoo backends are deployed in '<folder>': Shorokoo.WinCPU (CPU),
> Shorokoo.WinGPU (CUDA). Discovery picks the backend for a program that named none, and
> this deployment gives it no way to choose. Say which you mean: assign
> DefaultBackend.Instance before the first run to make one of them the default.
> To run several at once, give each ComputeContext its own backend -- new ComputeContext(new
> LinuxGpuBackend()) -- and where they need separate native ONNX Runtimes, load
> them with IsolatedBackend.Load.`

Discovery does not resolve that by looking for a CUDA runtime and preferring the GPU. A
deployment holding both packages has already had their native ONNX Runtimes collide —
each ships `libonnxruntime.so` (`onnxruntime.dll`) at the same path, so only one of them is
deployed and which one is NuGet's conflict resolution to decide — and the managed DLL that
discovery would pick says nothing about the native that is actually there. (Separating them
is exactly what [One model, two devices](#one-model-two-devices) does, and why a program
running both deploys each native in a folder of its own.) Backends for
*different* OSes are not ambiguous and are not refused: only those targeting the running one
are candidates, so a Windows backend alongside a Linux one is no ambiguity at all. Carrying
all four, on the other hand, is two for whichever OS you run on — and refused on both.

Mind that step 1 settles it first. If exactly one backend assembly is already loaded when the
first run happens — which naming its backend type anywhere in a method your program
runs is enough to cause — that one wins and the folder is never probed. The refusal is what
happens when the *deployment* is left to make the choice, not a guarantee that an ambiguous
build cannot run.

The usual way to arrive at an ambiguous deployment by accident is a shared library that
references a backend, which flows to everything referencing it; keeping the backend in the
executable is what avoids it — see
[Or keep it to two processes](#or-keep-it-to-two-processes). Arriving there on purpose,
because the program really does want both, is
[Deploying two backends](#deploying-two-backends).

Referencing a backend package is enough for step 2: the package copies its DLL to your
output folder, so discovery finds it whether or not your code mentions the backend type.
On a Linux sandbox that ships only `Shorokoo.LinuxCPU`, discovery picks it with no setup.

If no backend is found, the first run throws `InvalidOperationException`:

> `No Shorokoo backend is set and none was found in '<folder>'. Set one at
> startup -- e.g. DefaultBackend.Instance = new LinuxCpuBackend(); (or the
> backend from whichever Shorokoo.{WinCPU,WinGPU,LinuxCPU,LinuxGPU} package you
> reference) -- or add such a package as a dependency.`

### Which device am I on?

Nothing at a call site says which device the work will go to — `Compile(...)` and
`Execute(...)` look the same on a CPU build and a GPU one. Ask instead:

```csharp
using Shorokoo.Core.Backends;

Console.WriteLine(DefaultBackend.Describe());         // Shorokoo.WinGPU (CUDA device 0)
Console.WriteLine(ComputeContext.Default.Backend);    // the same, at the point work is submitted
```

`BackendDescription` carries the `Name` of the supplying assembly, the `Device`
(`ComputeDevice.Cpu`, `Cuda`, or `Other` for a backend you wrote against a third execution
provider), and the `CudaDeviceId` a CUDA backend allocates on (null on anything else). Record it in a run's log: a training run that cannot say which
device produced its numbers has lost something it cannot reconstruct later.

Two related entry points:

- `DefaultBackend.Current` is the live backend **or null**, and — unlike `Instance` and
  `Describe()` — reading it does not resolve one. Use it to tell "nothing chosen yet" from
  "already bound" without settling the question by asking it.
- `DefaultBackend.RequireDevice(ComputeDevice.Cpu)` throws unless the live backend is on
  that device. Put it at the top of a program whose correctness depends on where it runs —
  a check that must not contend with a training run holding the card — and it fails at
  startup, with the live backend named, instead of quietly sharing the GPU:

  ```csharp
  DefaultBackend.RequireDevice(ComputeDevice.Cpu);   // before anything runs
  ```

### Loading a backend at runtime

A program need not reference a backend at compile time at all. `BackendPackage.TryLoad`
takes a path and hands back a backend, and one that does not fit the machine comes
back as a reason rather than an exception — so one executable can carry backends for
several platforms and pick at startup.

```csharp
using Shorokoo.Core.Backends;

if (BackendPackage.TryLoad("plugins/Shorokoo.WinGPU.dll", out var gpu, out var why))
{
    using var cuda = new ComputeContext(gpu!);
    // ...
}
else
{
    Console.WriteLine($"No GPU backend here: {why.Detail}");   // fall back, warn, or stop
}
```

`BackendPackage.Probe` answers the same question without loading the *backend*: it reads its
declaration out of the file's metadata, so a backend for another operating
system, another architecture, or one whose native libraries are not deployed with it is
refused without the backend or its ONNX Runtime being loaded. `BackendProbe.Reason` says
which it was (`WrongOperatingSystem`, `MissingNative`, `MissingCudaRuntime`, …) and `Detail`
names the file or library that is wrong.

One check is not free of native code: a backend declaring a CUDA requirement is verified by
binding the CUDA runtime and asking it for the device's memory, which initialises this
process's CUDA context on the card if it has none yet. So probing a GPU backend touches the
driver even when the answer turns out to be no. Every other rejection — wrong OS, wrong
architecture, a native that is not deployed — is decided from metadata and file paths alone.

A backend's natives are looked for in both of the places a .NET build puts them: flat
beside the backend assembly, and under `runtimes/<rid>/native/` next to it. Which one a
given deployment has is not the backend's choice. ONNX Runtime's native packages copy
their library to the output root through build props that fire on Windows only, so a
source build of a backend is flat on Windows and under `runtimes/linux-x64/native/` on
Linux; and a program that installs `Shorokoo.LinuxCPU` (or any of the other backend
packages) from NuGet gets the `runtimes/` layout on *every* platform, because those props
live in the ONNX Runtime package's `build/` folder and so do not reach a consumer that
reached ONNX Runtime transitively. The native ONNX Runtime a loaded backend binds is
resolved the same way, so it binds the file that is really there.

Each backend loaded this way gets a load context of its own, so several run side by side
without sharing a native runtime.

### Feeding a run: consumed, shared or tried

A tensor fed to a run **as it is** is given to that run. The run takes it when it starts — the
tensor is dead from that moment — and its memory goes to the run's backend, which releases it
before the call returns, or writes one of the run's outputs into it where the graph allows
([below](#a-run-that-writes-an-output-into-what-it-consumed)). That is what a batch built for one
call wants, and it is what the call does:

```csharp
var result = compiled.Execute(batch)[0].ToTensorData();
// batch is consumed: reading it throws, naming the run that took it
```

To use a tensor after the call, pass it `.Shared()`. The run then only reads it — holding a reader
lock on it while it runs, so nothing can delete it underneath — and it is alive and unchanged
afterwards:

```csharp
var weights = TensorData([4L, 4L], w);
var first  = compiled.Execute(x1, weights.Shared());
var second = compiled.Execute(x2, weights.Shared());   // weights is still there
```

`.TryConsume()` sits between the two: the run consumes the tensor if nothing else is reading it
when the run starts, and reads it otherwise. Fed as it is, a tensor another run is reading is
refused instead — the call throws `InvalidOperationException` naming the run that holds it, and
takes nothing — since consuming it would take its memory from under that run.

| Fed as | The run | Afterwards |
|---|---|---|
| `t` | takes it when it starts; refuses it while another run is reading it | dead |
| `t.Shared()` | reads it, holding a reader lock | alive and unchanged |
| `t.TryConsume()` | takes it if nothing else is reading it, reads it otherwise | dead, or alive if it was read |

- **Consumption is irrevocable.** A run that fails, or is stopped, after it started has still
  consumed what it was fed as it is. A run refused before it starts — a cancelled token, a feed
  that is dead, or one being read that it would have to consume — takes nothing: everything it
  could refuse over is checked before anything is taken.
- **One tensor fed twice in one call** is taken at most once: read if any occurrence is
  `.Shared()`, otherwise consumed if any is bare, otherwise tried.
- **Composites apply the mode to everything they hold.** `TensorDataStruct`, `TensorDataSequence`
  and `OptionalTensorData` have `.Shared()` and `.TryConsume()` too; fed as it is, a struct or a
  sequence gives the run every tensor it holds. So does a training checkpoint — see
  [What a training step consumes](training.md#what-a-training-step-consumes).
- **The error names the call to change.** Reading a consumed tensor throws
  `ObjectDisposedException` naming the run that took it — its graph and its context — and the
  input it fed; the remedy is to pass it `.Shared()` at that call.
- On a tensor, struct, sequence or optional, `.Shared()` and `.TryConsume()` return a
  `SharedInput`: an `IData` carrying the value and its `Mode`, accepted wherever an input is. On a
  training checkpoint they return the checkpoint itself with its `FeedMode` set, which its
  derivations (`WithStep`, …) and `rig.AdoptCheckpoint` keep, since they share its tensors. `Run`,
  which takes `NamedModelParam`s, reads each one's `Sharing` instead — `null` for as it is.

**Memory the run cannot read where it is.** A run reads its inputs in its backend's memory. A
tensor anywhere else — every tensor built from a C# array, whose managed memory no runtime reads
directly, and one another device or another runtime allocated — is fed through a copy in the
run's memory, and the mode decides what becomes of that copy:

- **Consumed**, the contents are copied into the run's memory, the tensor is dead and its own
  memory released at the feed, and the copy is what the run consumes.
- **Read**, the copy is made on the first such read and kept. It is a `TensorData` of its own:
  held by the tensor it was copied from, locked by each run that reads it, attached to the context
  that read it (so it shows in `context.Tensors`), and read again by every later shared read in
  that memory rather than copied afresh. Writing to the tensor (`AccessModifiableMemory` and the
  like) retires the copy — the next read copies what was written — and the copy goes too when the
  tensor is deleted or consumed.

### A run that writes an output into what it consumed

Memory a run consumes is its backend's, and ONNX Runtime keeps every input of a run until the run
ends: measured on a card, a 64 MiB input in the session's own arena, read by the graph's first node
alone and held by nothing but the run, still held its memory when the last node ran — so an arena
with room for the run's own two 64 MiB blocks could not also take the input. That memory cannot come
back part-way through a run. What a run can do is write an output **into** it — output aliasing —
and the same run then fits: the output is produced in the input's memory instead of in a block of
its own.

That is correct only where nothing reads the input after the output is written, so it happens only
for outputs a graph's lowering marks after proving exactly that: every node reading the input is one
the output's writer waits for, and the writer itself reads the input only as an element-wise
update does. Today the one lowering that marks outputs is the training rig's step, which pairs each
updated state field with the field it replaces
([A step writes its state over the state it consumed](training.md#a-step-writes-its-state-over-the-state-it-consumed));
a graph you compile yourself marks none, and its outputs are always memory of their own. ONNX Runtime
rewrites a graph before it runs it, and a rewrite can change which nodes read an input, so the
backend proves each marked pair again over the graph it will actually run and drops any it cannot.

A run then writes a marked output into an input's memory only where:

- **it consumed that input** — memory it was only lent, `.Shared()`, is the caller's, and is read
  and left as it was;
- **it was fed as no other input** — one tensor fed twice is read as both;
- **the output is produced in that memory**, with the input's element type and shape — the shape
  ONNX Runtime settled when it built the session, so a graph compiled for shapes it leaves open
  writes its outputs where it always did. On a GPU backend an output a run fetches back to the host
  is not produced in the card's memory the input is in.

Nothing else changes. The outputs are new `TensorData` objects attached to the running context, the
consumed input is dead as it always was, and the values are the same.

### A tensor's lifetime: locks and deletion

A `TensorData` **is** its memory. There is one object per allocation — no second tensor ever names
the same bytes — and it records the backend that allocated it (`AllocatingBackend`) and where the
memory is: `Space` for the device, and `Location` for the device together with the runtime that
allocated it. That backend is the one that releases the memory, whichever contexts the tensor has
been attached to, or none. A tensor does not know which contexts it is attached to — see
[Moving data between contexts](#moving-data-between-contexts).

**How a tensor ends.** In exactly three ways, and nothing else ends it — in particular, disposing a
context it is attached to does not:

| | |
|---|---|
| **Deleted** | `Delete()`, `Dispose()`, `TryDelete()` or `DeleteAsync(...)` — below. |
| **Consumed** | fed to a run as it is — or through `.TryConsume()` with nothing else reading it — which takes it when the run starts; see [Feeding a run](#feeding-a-run-consumed-shared-or-tried). |
| **Moved into an attribute** | `MoveToAttribute()`, which takes its contents — see [core-types.md](core-types.md#the-two-conversions-and-which-one-spends-its-source). |

A dead tensor records which, and every access to it afterwards — reading its elements, feeding it,
`To`, `CopyTo`, `ToHost`, `Shared()`, `TryConsume()`, `MoveToAttribute` — throws
`ObjectDisposedException` saying so; for a consumed tensor it names the graph and the context whose
run took it, and says to pass it `.Shared()` at that call. `Shape`, `DType`,
`ToString()` and `IsDisposed` keep working, so a dead tensor can still say what it was. Ending a
tensor that is already dead does nothing, so disposing one twice, or at the end of a `using` over a
tensor a run has consumed, is harmless.

A tensor nothing references is reclaimed like any other object, its memory released through its
backend's ordinary path — so deleting is optional. You delete to choose *when* the memory comes
back, not to avoid a leak.

What that release actually frees depends on where the bytes are, and it is worth knowing which
case you are in when you are chasing memory. A tensor whose buffer the runtime allocated — a
run's output, a copy onto a card, an `AllocateUninitialized` on a real context — is holding native
memory, the card's own on a CUDA context, and releasing it hands that back at that moment. A
tensor you built from a C# array is holding a managed array, which stays the collector's to
reclaim: releasing the tensor lets go of the array, and what it frees at once is the copies runs
read it through ([above](#feeding-a-run-consumed-shared-or-tried)), which are native and can
themselves be on a card.

**What a run holds.** A run takes a reader lock on every tensor it reads — fed `.Shared()`, or
through `.TryConsume()` while something else held it — and holds it, and a reference to the
tensor itself, for as long as it runs, giving it up when it returns however it returns. Any number
of runs may read one tensor at once. While any of them holds its lock the tensor cannot be
deleted: `Delete()` and `Dispose()` throw `InvalidOperationException`, and `TryDelete()` declines.
So nothing frees memory a run is in the middle of reading. A tensor a run consumes it holds by
taking it, which no other run can then do.

The lock is taken inside the run, one feed at a time, so it is not held yet while the call is
being set up — and a deletion landing in that window ends the tensor before the run can claim it,
which costs you the run: the lock it then asks for is refused and `Execute` throws
`ObjectDisposedException`. Nothing reads freed memory and no run returns a wrong answer, but
deleting a feed from a second thread is not something to do while a run of it is starting; see
[A feed deleted while a run is starting loses that run](limitations.md#a-feed-deleted-while-a-run-is-starting-loses-that-run).

A context is held the same way — disposing a `ComputeContext` throws, rather than proceeding,
while a run of it is in flight or while it holds a lock on anything.

**Deleting.** Three calls, which differ in what they do when a run is reading the tensor:

| | What it does |
|---|---|
| `Delete()` / `Dispose()` | The same operation: ends the tensor and releases its memory now. Throws if a run is reading it. |
| `bool TryDelete()` | The same if no run is reading the tensor. If one is, it changes **nothing at all** and returns `false`: the tensor stays readable and no run is disturbed. `true` for a tensor already dead. |
| `Task<bool> DeleteAsync(timeout, cancellationToken)` | Ends the tensor at once, asks whatever is reading it to stop, and waits up to `timeout` for the memory to come back. |

Three things about `DeleteAsync` are easy to guess wrong, and all three follow from deletion
being immediate while only *reclamation* waits:

- **The tensor is deleted either way.** A `false` says the memory had not come back inside the
  budget; it never says the deletion did not happen. Retrying waits for something that has
  already happened — the memory comes back when the run ends, with no second call.
- **A timeout never rolls back.** By the time the budget runs out the run has been asked to
  stop and has thrown its work away, and un-asking cannot un-abort it.
- **The `CancellationToken` cancels the wait, not the deletion.**

Neither call is prompt, for the reason [stopping a run](#stopping-a-run) is not: what the wait
costs is the longest single operator in flight, and a whole run where the graph is one
operator. A backend that ignores the request makes `DeleteAsync` slow and never unsafe — the
wait then ends when the run finishes naturally.

**Host memory.** A tensor built from a C# array belongs to no context and no runtime: its
allocating backend is `HostBackend.Instance`, the framework's own managed memory, which every
host backend can read. `ComputeContext.Host` is a name for host memory, a target for `To` and
`CopyTo` like any other context. It holds nothing and runs nothing — `Compile`, `Execute`, `Run`
and `Eval` all refuse, naming a real context — and it cannot be disposed.

A graph's own literals are not tensors at all and have no lifetime — they are
[`TensorAttribute`s](core-types.md#two-kinds-of-concrete-tensor-tensordata-and-tensorattribute),
immutable and attached to nothing.

### Moving data between contexts

A tensor never moves: its memory is where it was allocated for as long as it lives. What the three
operations decide is whether a context is given *this* tensor or a copy of it, and none of them
touches the tensor it is called on:

| | Result |
|---|---|
| `t.To(context)` | `t` itself, if `context`'s backend can read its memory as it stands; otherwise a new copy in `context`'s memory. Either way attached to `context`. |
| `t.CopyTo(context)` | Always a new, independent copy in `context`'s memory, attached to `context`. |
| `t.ToHost()` | `t` itself, if the host can read its memory; otherwise a new copy in the framework's own host memory, attached to nothing. |

Whether a backend can read a tensor's memory as it stands is asked of that backend, and the answer
is **the same device and the same runtime**. The framework's own host memory — every tensor built
from a C# array — is readable by every host backend. A device allocation is meaningful only to the
runtime that made it: two backends over one loaded ONNX Runtime share a card allocation — a CPU
backend and a CUDA backend in one process, or two instances of one — while two isolated runtimes
on one card, which is what `IsolatedBackend` produces, do not, and copy through the host. A run's
outputs come back on the host unless you asked for them to be retained
(`CompiledGraph.Execute(inputs, retainOnDevice)`), so the copy arises for a tensor you put on the
card or kept there deliberately.

```csharp
var onCard = cuda.Compile(model).Execute([input], [true])[0].ToTensorData();  // device memory
var onHost = onCard.To(cpu);        // one copy across the bus; onCard is untouched
var same   = onHost.To(otherCpu);   // no copy: the very same object
var mine   = onHost.CopyTo(cpu);    // always a copy
```

**Attachment is bookkeeping, not ownership.** A context keeps a weak list of the tensors attached
to it, `context.Tensors`: its runs' outputs, what its runs read, and what `To`, `CopyTo` and
`AllocateUninitialized` placed for it. The list never keeps a tensor alive and never ends one's
life; a tensor that dies or is collected drops out of it. `context.Detach(t)` takes a tensor off
the list — refused while a run of that context is reading `t` — and never deletes it. Disposing a
context releases what the context itself holds, its compiled sessions, and leaves every tensor as
it was: a run's outputs outlive the context that ran it. The list is also what a context's
device-memory budget counts, so on a budgeted context `To` and `CopyTo` can be refused — see
[A context's device-memory budget](#a-contexts-device-memory-budget).

`TensorDataStruct` and `TensorDataSequence` take the same three operations, applied to the tensors
they hold. A struct comes back as itself where nothing had to be copied. A sequence owns its
elements, so where any of them has to be copied, the whole sequence is.

### Feeding a large input without a second copy

A feed built the ordinary way exists twice at feed time: you fill a managed array, and the
runtime copies it into a buffer of its own. Where the input dominates the step's peak that is the
largest thing in the run, doubled. Two operations remove one half each.

`ComputeContext.AllocateUninitialized` hands you the runtime's buffer to fill in place:

```csharp
using var cpu = new ComputeContext(new LinuxCpuBackend());

var batch = cpu.AllocateUninitialized<float32>(new Shape(64L, 3L, 224L, 224L));
batch.WriteMemory<float>(ReadImagesInto);   // no managed array in between
```

`Shape` is a class rather than a collection type, so the shape is `new Shape(…)` or a `long[]`,
not a `[…]` collection literal.

**Fill it through `WriteMemory`, not through a bare span.** On a real backend the buffer belongs
to the runtime and the tensor is the only thing keeping it alive. Taking a span is the tensor's
*last read*, so

```csharp
ReadImagesInto(batch.AccessModifiableMemory<float>());   // wrong: nothing roots `batch`
```

leaves no reachable tensor for the whole of `ReadImagesInto`, and the runtime value's finalizer
can hand the block back while you are still writing into it. Being in scope is not being
reachable. `WriteMemory` keeps the tensor alive across the call, the way `CopyMemory` does on the
reading side.

Nothing is written into it — the buffer holds whatever was last there, so fill all of it — and
the tensor is attached to the context exactly as a `CopyTo(context)` result is. The
`(shape, dtype)` overload is the same thing where the element type is only known at runtime. On
a CUDA context the buffer is the card's own memory, which the host cannot write through a span at
all (`TensorData.IsHostResident` is false); getting bytes there is still `CopyTo`.

The other half is feeding it **as it is**, which every feed not passed `.Shared()` already is.
The run takes the buffer where it stands — memory the context's own backend allocated, which its
runs address without a copy — and hands it back to the allocator as it returns:

```csharp
var loss = compiled.Execute(batch)[0].ToTensorData();
// batch is consumed: reading it throws, and its buffer went back to the allocator with the run
```

A consumed tensor's memory is released the moment the run has finished with it, however the run
ends, where a feed you keep — one passed `.Shared()` — is released when *you* let go of it, which
for a batch built per step is at the next collection. See
[Feeding a run](#feeding-a-run-consumed-shared-or-tried) for the rules.

**What consuming does not buy.** ONNX Runtime's memory planner gives every graph input one extra
use count, precisely so that a caller can still read a feed after `Run` returns, so no input's
buffer is ever recycled *inside* the run and no session or run option changes that. Consuming moves
the release from "whenever the caller lets go" to "the instant the run returns"; it does not hand
the input's bytes to the run's own intermediates.

### One model, two devices

One process can run one model on the CPU and on the card. A `ComputeContext` constructed
with a backend compiles and runs there, and two contexts may name different backends:

```csharp
using Shorokoo.Core.Backends;
using Shorokoo.LinuxCPU;
using Shorokoo.LinuxGPU;

var cpu  = new ComputeContext(new LinuxCpuBackend());
var cuda = new ComputeContext(new LinuxGpuBackend());

var onHost = cpu.Execute(graph, input.Shared());   // the host, reading input and leaving it
var onCard = cuda.Execute(graph, input);           // the same graph, the same input, the card
```

The model is compiled once, into one assembly, and both contexts run *that* — so a check on
the CPU tests the model the GPU run is training, not a second compilation of its source.
`ComputeContext.Backend` says which device each one will use, and a `CompiledGraph` carries
the backend it was built on in `CompiledGraph.Backend`.

**Tensors are not tied to a backend.** A `TensorData` you build holds managed bytes and no
backend at all, so building a model and exporting it needs no runtime; a backend enters only
when the tensor is fed to one, and then either context accepts it — a session hands what it
is fed to its own runtime, building it there if it does not have it yet.

How often that costs a copy depends on where the tensor is and how it is fed. A run reads as it
stands only memory its own runtime can address — never a C# array's, and never another
runtime's allocation — and reads anything else through a copy in its own memory
([Feeding a run](#feeding-a-run-consumed-shared-or-tried)). Fed `.Shared()`, the tensor keeps
that copy for the next read, so it is one copy per (tensor, runtime) however many runs follow,
until the tensor is written; fed as it is, the copy is made for that run and consumed with the
tensor. A tensor an execution provider left on the card (`TensorData.IsHostResident` is false,
which a [resident training run](training.md#keeping-training-state-on-the-device) produces)
crosses the same way, its bytes brought through the host by the backend that allocated them.

#### Deploying two backends

The two packages deliver their native ONNX Runtime at the same path
(`runtimes/<rid>/native/libonnxruntime.so`), so referencing both normally is not two
runtimes — it is one of them deployed twice, with the other's provider libraries stranded
beside a core that cannot use them. Which is exactly why
[auto-discovery](#auto-discovery) refuses such a deployment.

How you avoid that depends on whether the two backends need separate native runtimes:

**One runtime, two providers — the simple case, and the usual one.** A native ONNX Runtime
serves every execution provider compiled into it, and the CUDA-flavoured build carries the
CPU provider too. So deploy the GPU package alone and add the CPU package for its backend
type only, with `ExcludeAssets="native"` so it brings no second runtime:

```xml
<PackageReference Include="Shorokoo.LinuxGPU" Version="..." />
<PackageReference Include="Shorokoo.LinuxCPU" Version="..." ExcludeAssets="native" />
```

**Then name the default explicitly, before anything runs.** Both backend assemblies are now
deployed, so [auto-discovery](#auto-discovery) has two candidates and refuses — and it is
the *first* read of `DefaultBackend.Instance` that refuses, which may be some framework
call you did not write. Constructing a backend does not settle the question; assigning does:

```csharp
DefaultBackend.Instance = new LinuxCpuBackend();   // startup, before anything runs

var cpu  = new ComputeContext(DefaultBackend.Instance);
var cuda = new ComputeContext(new LinuxGpuBackend());
```

**Two runtimes.** Two ONNX Runtime *builds*, or two versions, in one process — a vendor
build beside the stock one, say. Give each native a folder of its own and load the second
with `IsolatedBackend`:

```xml
<!-- The glue, referenced directly: it carries the target that reads the items below, and
     it is the assembly every isolated backend loads. Coming in through a platform package
     is the usual route, and this recipe cuts that route on purpose. -->
<PackageReference Include="Shorokoo.OnnxRuntime" Version="..." />

<PackageReference Include="Microsoft.ML.OnnxRuntime" Version="1.26.0"
                  ExcludeAssets="all" GeneratePathProperty="true" />
<PackageReference Include="Microsoft.ML.OnnxRuntime.Gpu.Linux" Version="1.26.0"
                  ExcludeAssets="all" GeneratePathProperty="true" />

<ShorokooBackendNatives BackendId="cpu"
  Include="$(PkgMicrosoft_ML_OnnxRuntime)/runtimes/linux-x64/native/*" />
<ShorokooBackendNatives BackendId="cuda"
  Include="$(PkgMicrosoft_ML_OnnxRuntime_Gpu_Linux)/runtimes/linux-x64/native/*" />
```

`ShorokooBackendNatives` items are read by a target the `Shorokoo.OnnxRuntime` package
imports; each lands in `ort/<BackendId>/` in the output. An execution provider's own library
has to sit beside the core it belongs to, so deploy a package's whole native folder.

The direct reference to `Shorokoo.OnnxRuntime` is what makes the rest of this work, and it
is easy to leave out because every other deployment gets it for free. Reaching the backend
package with `ExcludeAssets="all"`, as the next block does, cuts off the only route the glue
normally takes — so without this line the target never loads, the items above are silently
ignored, `ort/` is empty, and the first `IsolatedBackend.Load` fails with a
`FileNotFoundException` naming a native nothing ever copied. The package is not a backend
and never becomes a discovery candidate.

**The backend's own assembly has to be somewhere too, and not beside `Shorokoo.dll`** — two
backend assemblies there is the ambiguity [auto-discovery](#auto-discovery) refuses. Put it
in the same folder as its native and point `ProbeDirectory` at it:

```xml
<PackageReference Include="Shorokoo.LinuxGPU" Version="..." ExcludeAssets="all"
                  GeneratePathProperty="true" />
<None Include="$(PkgShorokoo_LinuxGPU)/lib/net10.0/Shorokoo.LinuxGPU.dll"
      Link="ort/cuda/Shorokoo.LinuxGPU.dll" CopyToOutputDirectory="PreserveNewest" />
```

```csharp
var cudaDirectory = Path.Combine(AppContext.BaseDirectory, "ort", "cuda");
var cuda = new ComputeContext(IsolatedBackend.Load(new IsolatedBackendSpec
{
    Name = "cuda:0",
    BackendAssembly = "Shorokoo.LinuxGPU",
    NativeRuntimePath = Path.Combine(cudaDirectory, "libonnxruntime.so"),
    ProbeDirectory = cudaDirectory,
}));
```

The ONNX Runtime wrapper and the glue below it are looked for beside the backend first and
beside `Shorokoo.dll` second, so only the backend's own assembly has to move.

`Name` is what `Backend.Name` reports — the backend assembly cannot tell two loads of itself
apart, so give it something a log can act on. It is a label and nothing more: the backend is
identified by its native, so loading one native twice under two names is refused rather than
producing two backends over one runtime. A loaded backend lasts for the life of the
process: its native runtime holds thread pools, arenas and allocators, and nothing unloads
it. Loading the same spec twice returns the same backend.

Only the ONNX Runtime wrapper, the glue over it and the backend assembly are private to an
isolated backend; the core Shorokoo assembly and everything your model is written in stay
shared, which is what lets one context be handed the other's data.

A model with *sequence* outputs runs here like any other, on a card as on the host: ONNX Runtime
materializes a run's sequence output in host memory whichever execution provider produced it. What
a sequence cannot hold is a tensor left in device memory — see
[A sequence's elements live in host memory](limitations.md#a-sequences-elements-live-in-host-memory).

#### Or keep it to two processes

Nothing above is compulsory. Splitting the work across two executables over a shared,
backend-free model library is still a good answer where the two halves are separate jobs —
a long training run and an occasional check — and it costs a process rather than the
coordination of two devices in one:

```xml
<!-- Model.csproj — the model, its modules, its losses. No backend. -->
<ItemGroup>
  <PackageReference Include="Shorokoo" Version="..." />
</ItemGroup>
```

```xml
<!-- Train.csproj — the long run. -->
<ItemGroup>
  <ProjectReference Include="../Model/Model.csproj" />
  <PackageReference Include="Shorokoo.WinGPU" Version="..." />
</ItemGroup>
```

```xml
<!-- Check.csproj — the causality check, the shape probe, the debugging execution. -->
<ItemGroup>
  <ProjectReference Include="../Model/Model.csproj" />
  <PackageReference Include="Shorokoo.WinCPU" Version="..." />
</ItemGroup>
```

What breaks *this* layout is a backend reference in the shared library — a `PackageReference`
or a `ProjectReference` to an executable that carries one. Either flows into the referencing
project's output folder, two backends end up deployed, and, since neither executable named
one, [auto-discovery](#auto-discovery) refuses to guess between them. Keep the backend in the
executable, where the device is decided, and let nothing reference an executable.

### Device memory (GPU backends)

ONNX Runtime allocates device memory out of a BFC arena that extends in blocks and, unless asked to
shrink (below), never gives them back. How large each new block is comes from the *extend strategy*, and the two ORT offers
suit opposite situations:

- **`NextPowerOfTwo`** (ORT's default) makes each extension at least as large as everything the
  arena already holds. The regions are big, splittable and reusable, which is what an
  unpredictable series of allocation sizes needs — but once a run's sizes have settled the
  doubling is pure overshoot, and it is why a long training run ends up holding far more of the
  card than its steps use.
- **`SameAsRequested`** extends by exactly what was asked for, so a settled run's arena tracks it.
  The catch is that an exactly-sized region cannot serve a later, larger request: a session whose
  input shapes keep growing strands every region it outgrows.

**Shorokoo picks between them per session, and does not ship one value for both.** Measured on the
CPU arena — the same allocator with the same two strategies — over four chained matmuls. Both
columns move by a MiB or two between runs, and on the mixed rows a run can put the two within one
MiB of each other, so read every ratio below as approximate and the near-ties as ties:

| shapes fed to the session | `SameAsRequested` | `NextPowerOfTwo` |
|---|---|---|
| one shape, ten runs | **11–12 MiB** | 16 MiB |
| alternating 2048/512, twenty runs | **20–25 MiB** | 28–33 MiB |
| largest first, then settled | 17–18 MiB | 15–16 MiB |
| shuffled from four sizes, twenty runs | 34 MiB | **31 MiB** |
| growing, then settled | 34–35 MiB | **31 MiB** |
| growing 256 to 2048 | 23 MiB | **15 MiB** |
| growing 256 to 2048, 16 MiB arena | does not fit | **15 MiB** |

The measurement is a test in the Shorokoo repository
(`ArenaExtendStrategyProbeTests`, `Purpose=Manual`) rather than something you can run against the
package, and it is taken on the **CPU** arena — the same `BFCArena` with the same strategy enum the
CUDA provider uses. It reads the arena back off glibc's `mallinfo2`, so it runs on Linux only and
says "unavailable" anywhere else.

Both tables here were taken on **one machine**: the host rows under Linux, the card rows below
under Windows on the same box, same CPU and same RTX 4090. So the hardware is common to them and
the operating system is not — enough for the ratios to be compared, not enough to call them a
like-for-like pair. The host rows reproduced on that machine within the ranges shown, and more
steadily than the spread above suggests: across four consecutive runs the settled-series row did
not move at all.

A second probe (`ArenaExtendStrategyCudaProbeTests`, `Purpose=Manual`) answers what the host one
cannot: what each strategy costs **one real training step on a card**. A 49,214,208-parameter
decoder-only transformer — 6 layers, width 384, 6 heads of 64, sequence 1024, vocabulary 50,257,
fp32, `AdamWOptimizer` — at batch 8 on a 24,564 MiB RTX 4090, read off the step's own session arena
through `ComputeContext.RunStats` rather than off the device:

| the training step's own arena | `SameAsRequested` | `NextPowerOfTwo` |
|---|---|---|
| in use at its highest, settled | **8,009 MiB** | 8,043–8,101 MiB |
| taken from the device, step 0 | **8,864 MiB** | 9,233–9,249 MiB |
| taken from the device, step 1 onwards | **15,508 MiB** | 17,425–17,441 MiB |
| blocks held, settled | 278 | 14–15 |

Two runs, and this time the `SameAsRequested` column repeated to the byte while `NextPowerOfTwo`
moved by 16 MiB and one block — the reverse of the host table above, where both columns move by a
MiB or two. Treat both tables as indicative of the shape rather than as your machine's numbers:
what settles your case is `ComputeContext.RunStats`, or `DeviceMemory.Sample()`, around your own
run.

Neither column wins outright, and which one wins is decided by something Shorokoo knows about each
session: whether its allocation sizes settle. A training run feeds one input shape to one compiled
step for its whole length — the first row of the host table — and there exact-size extension holds
about 1.3–1.45x less; on the card, on a real training step, it holds **1.12x** less. Two allocation
sizes still favour exact-size extension (row 2, by 1.2–1.6x depending on the run); it is once
several are in play that ORT's doubling holds less, by about 1.1x, or ties (rows 3 to 5). The last
two rows are the other end: input shapes that grow without settling, where each outgrown region is
stranded, the doubling holds around 1.5x less and — on a card with no room to spare — fits where
exact-size extension does not.

**What the card adds is that the strategy is the smaller half of the story.** The arena roughly
doubles between step 0 and step 1 whichever strategy it is on — 1.75x under exact-size extension,
1.89x under ORT's — and it is a single extension either way: one block of 6,644 MiB where the
region is exactly the request, of 8,192 MiB where it is rounded up to the next power of two. The
step never gives that block back, so a settled training step holds about twice what its steps
actually use under **both** strategies (1.94x and 2.15x here). Choosing the strategy trims that
block; it does not stop it being taken. An arena limit does — see below. A larger model of the same
family (12 layers, width 768, 162,129,408 parameters) at the same batch reached the card's whole
24,564 MiB at step 1 under both strategies and went on training there, which is what having no
headroom looks like rather than a failure.

Which of those a given session is turns on **how its caller feeds it**, and that is not knowable
when the session is built: a compiled graph fed one batch shape for its whole life and one fed a new
shape every call are the same object. So `ArenaExtend` defaults to `Auto`, which is not one of ORT's
values but `SameAsRequested` **except where Shorokoo already knows the shapes differ**.

Today that is one case. A training rig keeps a compiled step per input shape up to a limit; feed it
more distinct shapes than that and it falls back to a single step that every later shape shares.
By the time that step exists the differing shapes have already happened — it is not a guess — and it
is the growing-shape row, the one where exact-size extension strands a region per outgrown input and
cannot fit under a budget at all. That step gets the doubling; everything else keeps exact-size
extension, as it did before `Auto` existed.

**If your own session is fed shapes that keep growing, say so** — Shorokoo cannot know it before the
feeds arrive, and this is the case worth overriding. Naming a strategy applies to the sessions that
context compiles and no others, and `CompiledGraph.DeviceMemory` reports what a graph actually got:

```csharp
using Shorokoo.Core.Backends;
using Shorokoo.Runtime;

var ctx = new ComputeContext
{
    DeviceMemory = new DeviceMemorySettings
    {
        ArenaExtend = ArenaExtendStrategy.NextPowerOfTwo,   // ORT's doubling, back again
        LimitBytes = 16L * 1024 * 1024 * 1024,              // a 16 GiB budget on what ctx holds on the card
    },
};

var compiled = ctx.Compile(graph);
compiled.Execute([batch1]);

Console.WriteLine(compiled.DeviceMemory.ArenaExtend);              // what this session was built with
Console.WriteLine(compiled.DeviceMemory.LimitBytes);               // the arena limit the budget left it
```

| setting | on | ORT option | default | read |
|---|---|---|---|---|
| `LimitBytes` | `DeviceMemorySettings` | each session's `gpu_mem_limit` is the budget less what the context holds on the card | `null` — no budget | on every transfer onto the context and every run of it — see [A context's device-memory budget](#a-contexts-device-memory-budget) |
| `ArenaExtend` | `DeviceMemorySettings` | `arena_extend_strategy` | `Auto` — `SameAsRequested`, except ORT's `NextPowerOfTwo` for a session Shorokoo knows is reused across differing shapes | when a session is built |
| `ShrinkArenaAfterRun` | `RunSettings` | `memory.enable_memory_arena_shrinkage` | `false` — and forced on under a budget | on every run |

The other two are unset by default for their own reasons. `ShrinkArenaAfterRun` costs a
synchronizing device allocation on every step to re-take what it handed back, so outside a budget
it is worth it only when the card is shared with something that needs the room between steps. It
is also the one setting that can fail a run rather than degrade it: ORT rejects the request where
the device it names has no arena allocator registered — an arena disabled through
`ORT_DISABLE_ARENA`, say — so turn it on with a short run before a long one; under a budget it is
on for every run. `LimitBytes` is a budget, not a hint: what would pass it is refused, or fails
with ORT's `BFCArena ... Failed to allocate memory for requested buffer`, rather than eating the rest
of the device, so a figure set too low fails work that would have fitted. An arena limit near what
a step actually uses is also the one lever that reaches the step-1 expansion above: the same
batch-8 transformer, its step's arena capped at 10 GiB, ran every step inside 8,864 MiB (exact-size
extension) or 9,217–9,233 MiB (ORT's), never took the extra block at all, and kept its in-use peak
at 7,305 / 7,469–7,485 MiB — the same run it was, on roughly half the card. That was a cap on the
step's arena alone; a context budget of 10 GiB leaves the arena less than that, by what the context
holds on the card, so size the budget as what the step uses plus what the rig keeps there.

`ArenaExtend` is read **when a session is built** — the first inference call, or a training rig's
first `TrainStep` for a given input shape — so the context has to carry it before the graph is
compiled on it; a graph keeps what it was built with, which is why `CompiledGraph.DeviceMemory`
reports the settled strategy rather than `Auto`. A context's settings are initialize-only, so a
context's budget is fixed with it. `ShrinkArenaAfterRun` ORT reads on every run, so a
`CompiledGraph.Execute` / `Run` call can override it for that call alone — on an unbudgeted
context. The context's own one-shot entry points and a rig's `TrainStep` take no such override and
run on the context's instance, so set it on the context they run on.

The static `DeviceMemory` class — the readings, not the settings — reports what the card is doing:

```csharp
using var run = rig.BeginResidentRun(checkpoint);
for (int step = 0; step < steps; step++)
{
    run.Step(input.Shared(), target.Shared());   // read, so the next step can use them
    DeviceMemory.Sample();
}
Console.WriteLine($"peak {DeviceMemory.PeakUsedBytes / (1024 * 1024)} MiB");
```

`Read()` returns a `DeviceMemoryReading` (`UsedBytes`, `FreeBytes`, `TotalBytes`), `Sample()` does
the same and folds the reading into `PeakUsedBytes`, and `ResetPeak()` starts a fresh peak. Four
things to know about the numbers:

- They are the **device's**, not this process's — every other process on the card, a desktop
  session included, is in `UsedBytes`.
- Nothing samples on its own. `PeakUsedBytes` is exactly the largest figure your own `Sample()`
  calls have seen, which is the point: a step lasting 0.2 s falls between the polls of a
  half-second `nvidia-smi` sampler, and the "peak" such a sampler reports can be half the real one.
  A `Sample()` per step costs about a microsecond and cannot miss the step it follows.
- On a machine with no CUDA runtime installed both return `null` rather than throwing, so the call
  can stay in code that also runs on a CPU backend.
- The reading goes through the CUDA runtime directly, which initializes this process's context on
  the device if it has none — itself a few hundred MiB. Take the first reading after the backend is
  up, not before, or that cost lands inside your baseline.

**Scope: the context, its sessions and its runs — never the process.** A budget is one context's:
two contexts on one card each keep their own, and a tensor on both contexts' books counts on both.
The arena settings follow ONNX Runtime's own shape rather than a convention layered on top: ORT
gives each session its own arena and reads its limit and strategy once while building it, after
which the session keeps them for life — so a context configures the sessions it compiles, two
contexts may differ, and neither reaches the other's sessions. To run something under a different
budget or strategy, compile it on a context that carries one. `RunSettings` ORT reads off the run
instead, so those are settled per call.

The tensors a context places on the card are not in any of those arenas: they come out of one
allocator per card, shared by every context in the process whatever its settings and held for the
life of the process. The budget counts them by what is attached to the context, not by that
allocator.

Where one process must serve both a training loop and a variable-shape inference path, give them a
context each rather than picking one arena strategy for both.

The readings are the exception, and they are readings rather than settings: `Read()` and `Sample()`
go to whichever CUDA device is current for the calling thread — device 0, because that is what the
shipped GPU backends use — and `PeakUsedBytes` is one process's record of its own run.

### A context's device-memory budget

`DeviceMemorySettings.LimitBytes`, on a context whose memory is a card's, is a budget on
**everything that context holds there**: the tensors attached to it in the card's memory and, while
one of its runs executes, the arena that run computes in. It is the context's budget — not one
arena's, and not the process's:

```csharp
using var ctx = new ComputeContext(gpu)
{
    DeviceMemory = new DeviceMemorySettings { LimitBytes = 2L * 1024 * 1024 * 1024 },
};

var onCard = big.CopyTo(ctx);                     // on the card, and on ctx's books
var use = ctx.ReadDeviceMemoryUse();
Console.WriteLine($"{use.AttachedBytes} of {use.LimitBytes} bytes attached, {use.AvailableBytes} left");
```

**What it counts.** A tensor is on a context's books when `To`, `CopyTo` or `AllocateUninitialized`
placed it for the context, when one of the context's runs read it or copied it there to read it,
and when a run left it there as an output (`Execute(inputs, retainOnDevice)`).
`ReadDeviceMemoryUse()` adds up the live ones in the context's memory: `AttachedBytes`,
`AttachedTensors`, and `LimitBytes` — `null` where no budget is in force, because none was set or
because the context's memory is the host's, which a device-memory budget does not govern. A tensor
on two contexts' books counts on both; one that dies, is collected, or is taken off with `Detach`
drops out.

**A transfer it cannot take is refused before it allocates.** `To`, `CopyTo` and
`AllocateUninitialized` onto the context — and the copy a run makes of a tensor it cannot read where
it is, which is how a host tensor fed to a run on the card is read — are refused when what is
attached plus what they would add passes the limit. So is a `To` of a tensor already on the card
that the context's backend reads as it stands: nothing is copied, but attaching it puts its bytes
on the books. The refusal is an `InvalidOperationException` naming the budget, what is attached and
what was asked for:

```
CopyTo(context) of Tensor (8388608,):Float32 asks this compute context for 33554432 bytes of CUDA
device 0 memory, which its device-memory budget cannot give: the budget
(DeviceMemorySettings.LimitBytes) is 67108864 bytes, and 50331648 bytes of it are attached to the
context there, in 1 tensor(s), leaving 16777216. Delete what the context no longer needs, or give
it a larger budget.
```

**A run's arena gets what the context leaves it.** A session's `gpu_mem_limit` is the budget less
the *discount*: what the context holds on the card outside that session's arena for the length of
the run — the tensors attached to it there, and those the run reads there or copies there to read.
A tensor already on the card is read where it is and never enters the arena, so it stays in the
discount for the whole run: measured, a session whose arena was capped at 32 MiB read a 64 MiB
input from the card with its arena never above 256 bytes, while the same bytes fed from host memory
had to be copied into the arena and did not fit. What a run consumed is released as it returns, and
drops out. A run whose discount leaves its arena nothing is refused before it takes anything it was
fed; one whose arena needs more than it was left fails with ORT's `BFCArena` error. On an RTX 4090,
under a 256 MiB budget: with nothing held, a session got a 252 MiB arena and a run filling 160 MiB of
it went through; with a 100 MiB tensor held on the card, the session was rebuilt at 152 MiB and the
same run failed.

**When a session is built again.** ORT fixes `gpu_mem_limit` when a session is built, and building
one costs about as much as the graph is large — measured, around 0.7 ms per node, about a second at
1,600 nodes — so Shorokoo does not build one per run. A session is built with the budget less the
discount rounded up to the next sixty-fourth of the budget, and kept for as long as that limit is
within what the budget allows. A run that finds the discount grown past the room its session left
builds the session again with the lower limit, before it takes anything; a run that finds the
discount fallen keeps the session, and its lower limit. So:

- the limit only ever comes down — at most sixty-four times over a compiled graph's life — and a
  loop that holds the same things on the card from one run to the next never rebuilds;
- a context that lets go of what it held keeps its compiled graphs' smaller arenas: compile the
  graph again to give it the room back;
- `CompiledGraph.DeviceMemory.LimitBytes` is the arena limit the graph's session has now, and
  `ReadArenaStatistics()` and `ReadNodePlacement()` read that session — a rebuilt one starts its
  figures, and its trace, afresh;
- a training rig's step is a compiled graph like any other, so its first steps can rebuild it as the
  rig's state arrives on the card.

An output a session left in its own arena — retained on the device and fed back to the next run of
the same graph — is inside that session's limit already, and is not discounted again.

An output a run [wrote into memory it consumed](#a-run-that-writes-an-output-into-what-it-consumed)
is counted where that memory is, once. The consumed tensor was on the context's books until the run
took it, so the run's discount counted its bytes, and its arena never had to make room for the
output. Afterwards the output is on the books in its place: in the discount of later runs where the
consumed tensor was outside the session's arena — a copy the run made onto the card, or one
`CopyTo` placed there — and inside the arena, and not discounted, where the consumed tensor was an
output of that session's own earlier run. A resident training run that begins from an initial
checkpoint keeps the state it writes over itself outside the arena for the whole run this way, in
the memory the first step copied the checkpoint into.

**One at a time.** Under a budget, the context's runs are serialized: a second waits for the first
to return, and so do a transfer onto the context and a compile on it, since each would be counting
room the running arena may be taking. Every run hands its arena's unused blocks back as it ends,
whatever `RunSettings.ShrinkArenaAfterRun` says, so between runs a session's arena holds what it
keeps — its weights, and outputs left there — rather than its peak. A context with no `LimitBytes`
is none of this.

What the budget does *not* count — spare blocks, the allocator tensors are placed from, a session's
weights between its runs — is in
[Known limitations](limitations.md#a-device-memory-budget-counts-tensors-not-arenas).

### What one session's arena did

`DeviceMemory` reads the card. To read **this session's own allocator** instead — its bytes, nobody
else's, and on a CPU backend as well as a GPU one — ask the compiled graph:

```csharp
using Shorokoo.Core.Backends;
using Shorokoo.Runtime;

var compiled = ctx.Compile(graph);
compiled.Execute(inputs);

if (compiled.ReadArenaStatistics() is { } arena)
    Console.WriteLine($"{arena.InUseBytes} in use, {arena.MaxInUseBytes} at its highest, "
                    + $"{arena.TotalAllocatedBytes} taken from the device");
```

`ArenaStatistics` is nine figures: `InUseBytes`, `MaxInUseBytes`, `MaxAllocSizeBytes`,
`TotalAllocatedBytes`, `LimitBytes` (the session's arena limit — under a device-memory budget, what
the budget left it — and `-1` with no budget),
`AllocationCount`, `ArenaExtensionCount`, `ArenaShrinkageCount` and `ReserveCount`. It is `null`
on a backend that reports none, the way `DeviceMemory.Read()` is `null` with no card.

**`TotalAllocatedBytes` is not a bound on what the card is holding.** On a run that filled a
24,564 MiB card it read 32,462 MiB — more than the device has. The likeliest reading is that an
arena pressed to the card's edge gives regions back and takes others while this counter does not
follow all the way down; either way, read it as the arena's own account of what it has taken, and
`DeviceMemory.Read()` for what the card is actually carrying.

**`MaxInUseBytes` is cumulative over the arena's whole life, not the last run's.** It is a
high-water mark the arena never lowers and there is no reset — asking for arena shrinkage does not
move it — so reading it once tells you the largest this session has ever been. For a figure per
run, let the context collect them.

**A session's weights come out of this arena too**, on the CPU and on a card alike, so a session
holds them before it has run anything: a graph whose only parameter is four mebibytes reads back
`MaxInUseBytes` of exactly 4,194,304 at construction. Two consequences. The first run's peak is
weights plus what that run added — `RunMemoryRecord.PriorPeakBytes` below is what separates them.
And `ReserveCount` and `ArenaExtensionCount` do not compare across devices: the host arena takes a
weight as a reserve, the CUDA arena as a block of its own.

### What crossed the bus

A device session that has to give part of a graph to the host stages the crossings through a
**pinned host arena**, which is a second allocator with figures of its own:

```csharp
if (compiled.ReadPinnedArenaStatistics() is { } pinned)
    Console.WriteLine($"{pinned.MaxInUseBytes} of pinned host memory at its highest");
```

Same nine figures, and `null` on a backend with no such arena — every CPU one, which stages
nothing. They are bytes of *host* memory the provider pinned, so they are deliberately not added
into `ReadArenaStatistics()`: a graph that stays on the card leaves this one at zero, and one the
runtime split pays here for every value that crosses.

### Per-run statistics on the context

Persisted tensors belong to a `ComputeContext`, so what its runs cost is answerable there.
Collection is **off by default** and costs nothing until asked for:

```csharp
using var ctx = new ComputeContext
{
    Diagnostics = new DiagnosticSettings { CollectRunStatistics = true },
};

var rig = TrainingRig.FromScratch(model, loss, optimizer, sample, hypers, runtimeContext: ctx);
for (int step = 0; step < steps; step++)
    checkpoint = rig.TrainStep(checkpoint, inputs.Shared());

var stats = ctx.RunStats;
Console.WriteLine($"{stats.RunCount} runs, peak {stats.PeakBytes / (1024 * 1024)} MiB, "
                + $"{stats.ArenaExtensionCount} arena extensions");

foreach (var run in stats.RecentRuns.TakeLast(5))
    Console.WriteLine($"run {run.RunNumber}: {run.PeakBytes} ({run.PeakKind}), "
                    + $"arena stood at {run.PriorPeakBytes} before it");
```

`RunStats` is a snapshot of every run the context has made, across **all** its sessions — the rig's
compiled steps, any graph you compiled on it, and the one-shot entry points. Two things to know
about the shape:

- **The aggregates are exact over the whole history; the per-run detail is bounded.**
  `PeakBytes`, `RunCount`, `AllocationCount`, `ArenaExtensionCount`, `ArenaShrinkageCount`,
  `LargestAllocationBytes` and `ArenaBytes` are folded in as each run finishes, so a hundred
  thousand steps are all in them. `RecentRuns` keeps the last `DiagnosticSettings.RecentRunCapacity`
  (1000 by default; zero keeps none), because one record per run held for the context's life is a
  leak in any real training loop.
- **A per-run peak says whether it was measured or bounded.** The arena's high-water mark is read
  either side of each run: a run that pushed it up set the record, and its `PeakKind` is
  `MemoryFigureKind.Measured`. A run that stayed under a mark some earlier run set is
  `MemoryFigureKind.UpperBound` — it used no more than that, and how much less is not something the
  arena records. The two are never reported as the same thing.
- **A per-run peak also says what the run found there.** `PriorPeakBytes` is the arena's high-water
  mark as the run found it, so a `Measured` peak is where the arena stood at this run's high point
  rather than the run's own cost: on a card, the **first** run of a four-mebibyte model read
  4,202,496 against a prior mark of 4,194,304, of which 8,192 was the run. Read the difference as
  the run's own only on that first run, where the prior mark is the weights and nothing else. Later
  it is the previous *highest* run's mark, so the difference is how far this run exceeded the
  record — zero for every `UpperBound` run, which in a settled loop is most of them.

`PeakBytes` is the largest mark any one of the context's arenas reached. A context runs its graphs
one session at a time, so that is the peak; where two of its sessions really do run together, read
it as the largest of them rather than their total.

**`ArenaBytes` sits above `PeakBytes` until something shrinks.** It tracks what the arenas have
*taken* from the device, which usually exceeds what is in use by whatever they keep spare — though
it is not a bound on what the device holds, and on a card pressed to its edge it has read above the
card's own capacity —
but `RunSettings.ShrinkArenaAfterRun`, which every run under a device-memory budget has on, hands
blocks back at the end of a run, before these are read, while the peak comes from a mark the
runtime never lowers. Three shrinking runs of a matmul on a
card left `ArenaBytes` below a `PeakBytes` of 3,145,728; on the same graph without shrinkage the
two were equal. How far below depends on how many sessions the context compiled, so read the
ordering rather than a figure. `ArenaExtensionCount` is the same figure's other half and undercounts for the
same reason: it is the blocks the arena is *holding*, so a shrinking run can end below where it
started and the aggregate loses the difference.

### Did part of my GPU graph run on the host?

A CUDA session that meets an operator the provider cannot run leaves that part to the host, and the
results cross the bus to get back. Two signals, one free and one not:

```csharp
switch (compiled.OutputPlacement)
{
    case SessionOutputPlacement.Device: break;               // all of it stayed on the card
    case SessionOutputPlacement.Mixed:                       // some of it did not
    case SessionOutputPlacement.Host: break;                 // none of it did
    case SessionOutputPlacement.Unknown: break;              // this backend does not say
}
```

`OutputPlacement` costs nothing — the session already knows where it puts its outputs — and is
there on every run. A CPU session reports `Host`, which is what it is.

On a card the three answers separate the three cases exactly. A graph the CUDA provider runs whole
reports `Device`. A graph with one operator it has no kernel for — `Det`, say — reports `Mixed`
when an output is left on each side, and `Host` when every output came back. Note that an output
the card computed and the host then consumed is reported as host memory, because that is where the
runtime put it: it lands in the pinned host arena, and `ReadPinnedArenaStatistics()` is what it
cost.

For **which** nodes fell back, and what they moved, ask the context to trace them. This one is not
free: it builds the session with ONNX Runtime's profiler on, which costs every run that session
then makes, so it is off by default and belongs in a diagnosis rather than in a training loop.

```csharp
using var traced = new ComputeContext
{
    Diagnostics = new DiagnosticSettings { TraceNodePlacement = true },
};
var compiled = traced.Compile(graph);
compiled.Execute(inputs);

if (compiled.ReadNodePlacement() is { } placement)
{
    foreach (var share in placement.Providers)
        Console.WriteLine($"{share.Provider}: {share.NodeCount} nodes, {share.OutputBytes} bytes out");

    foreach (var node in placement.NodesOn("CPUExecutionProvider"))
        Console.WriteLine($"  {node.Name} ({node.OpType}) fell back, {node.ActivationBytes} bytes in");
}
```

`Providers` has one entry per execution provider that ran anything, busiest first; more than one
entry on a GPU session *is* the fallback, with the bytes attached. **Reading the trace stops the
recording**: it covers every run up to that call, later runs are not in it, and a second read hands
back the same trace. So run what you are asking about, then read once.

`Nodes` is in the order the runtime ran them, which on a split graph is not the order of
`NodeExecution.NodeIndex`. ONNX Runtime inserts a `MemcpyToHost` / `MemcpyFromHost` node at each
provider boundary and gives it a fresh index above every node of the original graph, while leaving
it where it belongs in the plan — so each inserted copy still appears immediately before the node
it feeds while sorting to the end. On a graph with one crossing, measured, that copy carried the
highest index of all and ran third of four. A graph with several crossings has several such
copies, which sort among themselves and after everything else. Read `Nodes` for what happened, and
`NodeIndex` only as a name.

| what | where | cost | null / none when |
|---|---|---|---|
| `DeviceMemory.Read()` | static, the whole card | a microsecond | no CUDA runtime |
| `CompiledGraph.ReadArenaStatistics()` | one session's allocator | a call into the backend | the backend reports no arena |
| `CompiledGraph.ReadPinnedArenaStatistics()` | one session's pinned host arena | a call into the backend | the backend stages nothing (every CPU one) |
| `ComputeContext.ReadDeviceMemoryUse()` | what the context holds in its memory, against its budget | a walk over its list | never null; `LimitBytes` is null with no budget in force |
| `ComputeContext.RunStats` | every run of the context | two arena reads per run, once switched on | `CollectRunStatistics` is off |
| `CompiledGraph.OutputPlacement` | one session | nothing | the backend does not report it (`Unknown`) |
| `CompiledGraph.ReadNodePlacement()` | one session, per node | a profiler on every run of that session | `TraceNodePlacement` is off |

## Debugging engine (no OnnxRuntime)

```csharp
using Shorokoo.Core.Interpreter;   // QuickExecutionEngine
```

`QuickExecutionEngine` is a CPU-only interpreter used for debugging, shape inference,
and small prototypes. It only materializes values for tensors ≤ `MaxDataElements`
(default 256). Do not use it as a production inference path.

To debug the graph *structure* rather than values — e.g. when `ToConcreteArchitecture`
doesn't produce the graph you expect — snapshot the lowering stages with `DebugRequests`; to see
where a lowering that runs for minutes has got to, pass it a `progress:` sink and watch it stage by
stage. Both are in [debugging.md](debugging.md).

## Anti-patterns

- Do not call `OnnxEngine.Eval` in a tight loop for the same graph; compile once with
  `ComputeContext`.
- Do not expect to switch from CPU to GPU mid-process; the backend is sticky once
  loaded.
- Do not rely on `QuickExecutionEngine` results for large tensors — values above the
  element cap are not materialized.
