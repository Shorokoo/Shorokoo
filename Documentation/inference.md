# Running models (inference)

Related: [core-types.md](core-types.md) · [defining-models.md](defining-models.md) ·
[onnx-and-weights.md](onnx-and-weights.md)

## Facts

- `OnnxEngine.Eval(...)` is the simplest way to get values. It builds an ONNX model
  from the graph, runs it once via OnnxRuntime, and returns `TensorData`.
  - `TensorData Eval(Variable output)`
  - `TensorData[] Eval(Variable[] outputs)`
  - `TensorData[] Eval(Variable output1, Variable output2, params Variable[] outputs)`
  - Every op result converts to `Variable` implicitly. An `IValue`-typed handle does
    not and needs `handle.ToVariable()` — see
    [`Variable` and `IValue`](core-types.md#variable-and-ivalue).
  - `Eval` runs plain ops only. It refuses a `[Module]` output up front, naming the
    lowering that fixes it — see [Running a `[Module]`](#running-a-module).
- `OnnxEngine.Eval` rebuilds the ORT session on every call. For repeated inference,
  compile once with `ComputeContext` (below).
- Reference one platform backend package and it is found with no setup code. Two for one
  OS are refused rather than guessed between: [Backend selection](#backend-selection).
- A program can run several backends at once by giving each `ComputeContext` its own —
  [One model, two devices](#one-model-two-devices).
- `ComputeContext.Backend` and `DefaultBackend.Describe()` name the device work runs on, and
  `DefaultBackend.RequireDevice(...)` refuses to start on the wrong one —
  [Which device am I on?](#which-device-am-i-on).
- **A tensor fed to a run as it is is consumed by that run**: dead once the call is made, its
  memory released by the run's backend before the call returns, or reused for one of the run's
  outputs where the graph proves nothing reads it after
  ([A run that writes an output into what it consumed](#a-run-that-writes-an-output-into-what-it-consumed)).
  Pass `.Shared()` to have the run only read it, or `.TryConsume()` to have it consumed only when
  nothing else is reading it. Structs, sequences, optionals and training checkpoints take both —
  [Feeding a run: consumed, shared or tried](#feeding-a-run-consumed-shared-or-tried).
- A `TensorData` is its memory — one object per allocation, released through the backend that
  made it. It ends when deleted, consumed by a run, or moved into an attribute; a run's copy of a
  tensor, and an element a sequence holds as its own, end with what they belong to. A tensor a run
  is reading cannot be deleted — [A tensor's lifetime](#a-tensors-lifetime-locks-and-deletion).
- A compute context keeps books on tensors and owns none: disposing it releases its sessions and
  leaves every tensor alive. `To(context)` hands a tensor over if the context can read it where it
  is and copies it otherwise, `CopyTo(context)` always copies, and `ToHost()` brings one within the
  host's reach — [Moving data between contexts](#moving-data-between-contexts).
- A large input need not exist twice: allocate it on the context, fill the runtime's buffer in
  place, and feed it as it is; its memory is released as the run returns —
  [Feeding a large input without a second copy](#feeding-a-large-input-without-a-second-copy).
- On a GPU backend, a `ComputeContext`'s `DeviceMemory` is a **budget on what the context holds on
  the card** — its attached tensors there plus the arena of whichever of its runs is executing — and
  the arena settings of the sessions it compiles. `RunSettings` governs its runs; the static
  `DeviceMemory` class reports how much of the card is used. A transfer the budget cannot take is
  refused, and a run's arena gets what the attached tensors leave:
  [A context's device-memory budget](#a-contexts-device-memory-budget). The arena uses exact-size
  extension except for a session Shorokoo knows is reused across differing shapes, so a long
  training loop does not hold far more of the card than it uses:
  [Device memory](#device-memory-gpu-backends).
- Diagnostics, all off by default: `CompiledGraph.ReadArenaStatistics()` reads one session's
  allocator and `CompiledGraph.ReadPinnedArenaStatistics()` the pinned host memory its crossings
  used; `ComputeContext.RunStats` gives exact aggregates over every run plus a bounded window of
  per-run detail; `CompiledGraph.OutputPlacement` (free) and `CompiledGraph.ReadNodePlacement()`
  say whether, and which, nodes of a GPU run fell back to the host —
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
float[] values = result.CopyMemory<float>();
```

What `Eval` accepts:

- **Op results, yes.** Anything implicitly convertible to `Variable` —
  `Tensor<T>`, `Vector<T>`, `Scalar<T>` — which is what every op returns.
- **An `IValue`-typed handle, no.** `Variable` does not implement `IValue`, so
  `Eval(handle)` does not compile for a variable declared `IValue`; write
  `Eval(handle.ToVariable())`. See
  [`Variable` and `IValue`](core-types.md#variable-and-ivalue).
- **A `[Module]`'s output, no.** A value from `ResNet50.Call(...)` still carries its
  un-lowered module-invoke node. `Eval` detects that before building anything and throws
  an `InvalidOperationException` naming the fix; lower the module's `ComputationGraph`
  first — see [Running a `[Module]`](#running-a-module). (`ResNet50` is from
  [`samples/RetinaNet`](../samples/RetinaNet), a sample, not part of the packages.)

Build the input from a real array with the `params` overload — the first arg is the
shape, the rest are the flat values:

```csharp
var input = TensorData([1L, 3L, 224L, 224L], myPixelFloatArray); // float[] of length 1*3*224*224
```

For multiple outputs:

```csharp
TensorData[] outs = OnnxEngine.Eval(out1, out2, out3);
```

## Running a `[Module]`

A `[Module]`'s output (from `Foo.Call(...)` or `Foo.Model().Call(...)`) can carry an
un-lowered module-invoke node. Every eager-evaluation entry point — `OnnxEngine.Eval`,
`ComputeContext.Eval`, `tensor.Eval()`, `inputs.Eval(outputs).With(...)` — scans the graph
it builds and refuses such an output with an `InvalidOperationException`, however the
module's parameters are initialized. For example:

> `OnnxEngine.Eval requires a concretized graph (a 'concrete-architecture' or
> 'concrete-model'), but this graph is a 'module'. It still carries module machinery
> that lowering removes (ShrkCreateModule, ShrkModelInvoke, ShrkModuleSetHyperparams).
> It comes from module 'ResNet50': lower that module's ComputationGraph the whole way
> — ToConcreteArchitecture(inputHints) then ToConcreteModel() — and execute that,
> passing a value for each of its inputs in order ([Hyper] parameters come first). …`

It names the module the value came from and the easy mistake: values go in the graph's
input order, `[Hyper]` parameters first. `ComputeContext.Execute`/`Run`/`Compile` give the
same refusal for a module graph.

An initializer's function body never carries module machinery: an initializer may not
create or reference a model, and building one that does is refused with `FW055` (see
*Writing your own* in [nn-library.md](nn-library.md#initializers-shorokoomodulesinitializers)).

Concretize the module's `ComputationGraph` against the input first, then execute:

```csharp
using Shorokoo;
using Shorokoo.Graph;     // Specialize / ToConcreteArchitecture / FromOrderedInputs / ToConcreteModel
using Shorokoo.Runtime;   // ComputeContext
using static Shorokoo.Globals;

var input    = TensorData([4L], 1f, 2f, 3f, 4f);   // the actual input data
var graph    = MyLayer.ComputationGraph;            // readonly ComputationGraph (kind: Module)
var concrete = graph
    .ToConcreteArchitecture([input])    // one sample per input, in declaration order
    .ToConcreteModel();

var results = ComputeContext.Default.Execute(concrete, input);   // params IData[]
float[] values = results[0].ToTensorData().CopyMemory<float>();
```

For a graph loaded from a `.srk`/`.zsrk` file, catch this at load time instead: the file
header records the lowering stage, and
`LoadFastGraphFromFile(path, requiredStage: GraphKind.ConcreteModel)` refuses a
module-stage file with a stage-mismatch error — see
[onnx-and-weights.md](onnx-and-weights.md#the-srk-container).

### The lowering pipeline

Turning a `[Module]`'s `ComputationGraph` into a runnable model takes three steps, in order:

1. **`Specialize(values)`** — *optional.* Bakes a partial set of named inputs
   (typically `[Hyper]` parameters) into constants, folds them through the graph, and
   drops them from the input list. Skip it to keep those inputs live. Returns a copy.
2. **`ToConcreteArchitecture(inputHints)`** — inlines every sub-module and function so
   trainable parameters become visible at the top level, and uses `inputHints` to
   resolve shape-dependent parameters. It needs **a sample for every input**, `[Hyper]`
   and sequence inputs included (a generic module's type-placeholder slots take none),
   in one of two forms:

   - **Positional** — `ToConcreteArchitecture([hyper, input])`, an `IData[]` of bare
     values (`TensorData`, `OptionalTensorData`, `TensorDataSequence`,
     `TensorDataStruct`), one per input **in declaration order**. Too few is refused with
     **`FW056`**, naming every input missing its sample; too many likewise, stating both
     counts.
   - **Named** — `ToConcreteArchitecture(new ModelParamList([...]))` of
     `NamedModelParam`s (`TensorDataModelParam`, `OptionalTensorDataModelParam`,
     `TensorDataSequenceModelParam`, `TensorStructModelParam`), each bound to the input
     **of its name**, in any order. An input no sample names, a sample naming no input,
     and a name given twice are each refused with `FW056`, naming the offenders and
     listing the graph's inputs.

   ```csharp
   // Dense (below): Inline(Tensor<float32> x, [Hyper] Scalar<int64> outFeatures), graph inputs outFeatures, x
   var arch  = graph.ToConcreteArchitecture([hyper, input]);                 // by position
   var same  = graph.ToConcreteArchitecture(new ModelParamList([
       new TensorDataModelParam("x", ModelParamType.InputParam, input),
       new TensorDataModelParam("outFeatures", ModelParamType.InputParam, hyper)]));  // by name
   ```

   In either form, a sample whose rank differs from its input's declared type (a `Scalar`
   given a vector — positionally, usually two samples swapped) is refused with `FW056`,
   naming the input and the sample's shape. A struct input takes one sample, a
   `TensorDataStruct`, bound by position or by the struct input's name. Once lowered it is
   one input per field, named `<struct>.<field>`, of the field's own kind (tensor, optional
   or sequence); a nested struct field expands in turn to `<struct>.<field>.<subfield>`.

   Each sample's shape (never its values) is recorded on the architecture's input as its
   **representative shape** — the shape the model was concretized at. Every input of a
   concrete architecture, and of every concrete model made from it, carries one: it
   survives `ToConcreteModel`, `Specialize` and a `.srk`/`.skpt` round trip, a training rig
   rebuilds its shape inference from it, and ONNX export reads an input's rank from it
   where the signature states none (see
   [onnx-and-weights.md](onnx-and-weights.md#graph-inputoutput-names-and-shapes)). A
   concrete graph with an input lacking one — a hand-built one, say — is refused wherever
   it is frozen or loaded, with **`FW057`** naming the input; lower it again from its
   module.

   Each **output** likewise records its shape when the graph is evaluated at the samples'
   real values (a value can decide a shape: a flag choosing a branch, the axes a `Squeeze`
   drops), and keeps it through save and load. A struct output is one output per field,
   named `<output>.<field>`; an absent optional output records that it was absent; a
   sequence output records the shape its elements share. The `QuickExecutionEngine`
   computes the shapes. An output it cannot compute (string values, for instance) is taken
   from a run of the graph at the samples on the compute context `ToConcreteArchitecture`
   was given, when the graph has no parameter; where no run is made, or it fails (a locale
   the machine lacks, say), the output records the rank the engine found, with each
   unsettled dimension taken as `1`. Parameters have no values at this step, so an output
   whose shape depends on a parameter's **values** is recorded as *unresolved*;
   `ToConcreteModel` records every output again with the weights it binds, and `Specialize`
   with the values it bakes. A concrete model has no unresolved output: one whose shape
   even its weights cannot settle is refused by `ToConcreteModel` with `FW057`, naming the
   output and carrying the failure as the inner exception. ONNX export reads an output's
   rank from its recorded shape where the signature states none, and refuses a concrete
   graph with an output that records none with `FW057` naming the output.
3. **`ToConcreteModel(...)`** — binds parameter values (loaded weights, or the
   initializer defaults when called with no argument) into the architecture.

The example above has no hypers to bake, so it skips to step 2. The next section uses
step 1.

Every `ComputationGraph` has a **`Kind`** — `GraphKind.Module`,
`GraphKind.ConcreteArchitecture`, or `GraphKind.ConcreteModel` — stamped by the step that
produced it and preserved through copies and `.srk` save/load. `ToConcreteArchitecture`
requires a `Module` graph, `ToConcreteModel` a `ConcreteArchitecture`, and export/weight-query
operations name the actual and required kind when handed the wrong stage, so a mis-ordered
pipeline fails immediately. Execution (`ComputeContext.Execute`/`Run`/`Compile` and
`QuickExecutionEngine`) refuses a module-kind graph up front with the same lowering hint.
Since `WithKind` and `FromInternal` can stamp any kind, `ComputeContext` also checks the ops
themselves before building a session. `Eval` takes output values, not a `ComputationGraph`,
so it has no `Kind`; the op check is what refuses a module output there.
`ComputationGraph`s are **readonly**: operations that change a graph (e.g. `WithRngConfig`)
return a new one, so a graph's `Kind` cannot be invalidated.

If a graph arrives with the wrong kind — a foreign import that op-scanning misjudged, say —
re-stamp it with **`WithKind(kind)`**. The target kind is validated against the graph's
content (a module must not have initialized parameters; a concrete architecture also needs
a statically known parameter space; a concrete model needs every parameter initialized),
and a stamp that would misdescribe the graph is refused with an error naming the violated
requirement.

## Running a `[Module]` with `[Hyper]` parameters

A module's `ComputationGraph` lists its `[Hyper]` parameters as graph inputs **before**
the tensor inputs, whatever the `Inline` source order, and they stay inputs in the
concretized graph. Both `ToConcreteArchitecture`'s positional samples and `Execute` take
the hyper values first:

```csharp
// [Module] Dense { Inline(Tensor<float32> x, [Hyper] Scalar<int64> outFeatures) ... }
var hyper = TensorData([], 10L);                  // outFeatures = 10
var input = TensorData([2L, 4L], myFloats);

var graph    = Dense.ComputationGraph;
var concrete = graph
    .ToConcreteArchitecture([hyper, input])  // hypers first
    .ToConcreteModel();

// Hypers first. Shared, so that both can be passed again: a feed given as it is is consumed.
var results = ComputeContext.Default.Execute(concrete, hyper.Shared(), input.Shared());
```

Concretization bakes from the hyper value passed to `ToConcreteArchitecture`. A hyper that
determines the trainable parameters — their shapes (like `outFeatures`) or which exist at
all (a `[Hyper]` gating an `IfElse` branch that holds parameters) — is
**parameter-space-determining**: the value passed here fixes that part of the architecture
for good, so pass the same value at `Execute`. Value-only hypers (scale factors, ε's) are
read live on every `Execute` and may vary per call. See
[defining-models.md](defining-models.md#hyperparameter-baking) for the distinction, and
[What concretization fixes](#what-concretization-fixes) below.

### What concretization fixes

`ToConcreteArchitecture` produces an architecture whose **parameter space is static**:
every trainable parameter and other id-addressed component is enumerated, which is what
lets weights bind by name, optimizers allocate state, and checkpoints round-trip. Anything
derived from the values you pass is fixed then:

| Fixed at concretization | Derived from |
|---|---|
| Trainable-parameter **shapes** and count | hypers feeding a parameter's shape, and the shapes of the sample inputs |
| **Which** trainable parameters exist | hypers gating an `IfElse` whose branches hold parameters |
| The per-iteration **parameters** realized over a `LoopAPI.Iterate` body (and the whole iteration space, when the count folds to a constant and the loop unrolls) | hypers/inputs that drive the count |

Concretization rewrites only as much control flow as the parameter space requires. An
`IfElse` whose unselected branch holds parameters is resolved and folded away: those
parameters do not exist, so the branch can never be taken. An `IfElse` holding no
parameters is left alone, even on the same hyper: both branches stay, and it selects on its
still-live input at run time.

Folding is driven by the **unselected** branch. An `IfElse` whose *selected* branch holds
the parameters keeps both branches and stays live. For the usual
`bit.IfElse(withParams, without)` shape, the `IfElse` folds when the bit is baked **off**
and stays live when baked **on**. Only an `IfElse` that *solely* owns the pruned parameters
folds: one sharing them with a second `IfElse` is left alone, as is a tuple `IfElse` — its
slots resolve together, and the paramless ones must keep switching.

The fold is decided by the value supplied at concretization, not by the `[Hyper]` marker:
gating a trainable parameter on a plain runtime input is resolved from the concretization
value too. Marking such a gate `[Hyper]` makes a baked value look baked at the call site.

So one hyper can be half-resolved and half-live:

```csharp
var big = Zeros.Init([outFeatures]).Vec();       // a trainable parameter
var a = flag.IfElse(x * 10f, x * 100f);          // no params -> always stays live
var b = flag.IfElse(x + big, x);                 // holds big -> folded iff flag is baked false
```

Concretized with `flag = false`, `big` does not exist, `b`'s `IfElse` is gone (`b` is `x`
whatever you pass later), and `a` still switches on every `Execute`. Concretized with
`flag = true`, `big` exists, nothing is folded, and **both** switch at run time.

These values stay **live inputs** of the concrete graph — concretization, unlike
`Specialize`, removes nothing from the input list — so you supply them at every `Execute`,
and must supply **the same values**. To pass one `TensorData` at every call, feed it
`.Shared()`: fed as it is, the first run consumes it, and the next `Execute` throws naming
the run that took it. Executing with a value that would have produced a different
parameter space is **invalid use**: the parameters that value needs were never created.

For a **parameter gate**, what happens with a different value depends on whether its
`IfElse` folded:

- **It folded** (single-output gate, exclusive owner, baked off): the baked branch runs
  whatever you supply, so the opposite value is harmless.
- **It did not** (baked on, or a tuple or shared gate): the `IfElse` is live and the
  opposite value silently takes the other branch — skipping parameters that exist when
  baked on, or reading a **zero stand-in** for pruned parameters when baked off.

Either way, supply the value you concretized with. To make a contradiction impossible,
bake the hyper with [`Specialize`](#hardcoding-hypers-with-specialize) before concretizing:
it drops the input, so passing a value for it at `Execute` is an input-count error rather
than a silent wrong branch.

### Hardcoding hypers with `Specialize`

To hardcode hyper values into the model instead of re-supplying them on every `Execute`,
run `Specialize` first. It takes a partial set of named input values, constant-folds them
into the graph, and removes them from the input list. The process is then
**`Specialize`, then `ToConcreteArchitecture`, then `ToConcreteModel`**:

```csharp
var graph = Dense.ComputationGraph;                 // inputs: outFeatures, x

// 1. Bake the hyper(s). FromOrderedInputs pairs values with the leading input
//    names (hypers come first), so passing just the hyper value names it correctly.
var specialized = graph.Specialize(graph.FromOrderedInputs([hyper]));
//    `specialized` now has a single input: x.

// 2. + 3. Concretize on the remaining (runtime) inputs only.
var concrete = specialized
    .ToConcreteArchitecture([input])
    .ToConcreteModel();

var results = ComputeContext.Default.Execute(concrete, input);   // no hyper needed
```

`Specialize` matches values to inputs **by name** (against the graph's `InputNames`);
names with no matching input are ignored. It returns a copy and never mutates the
original. It works for any input, not just hypers, though baking a runtime input is
usually not what you want.

## Workflow: compile once, run many (repeated inference)

`ComputeContext` builds the ORT session once and reuses it.

```csharp
var ctx      = new ComputeContext();
var compiled = ctx.Compile(graph);                 // graph: a concretized ComputationGraph
var r1 = compiled.Execute(inputData1);             // params IData[] — the data goes here
var r2 = compiled.Execute(inputData2);             // reuses the session
```

An input fed as it is is consumed by the run, so `inputData1` is dead after the first call;
pass it `.Shared()` to use it again — see
[Feeding a run: consumed, shared or tried](#feeding-a-run-consumed-shared-or-tried).

`Compile(ComputationGraph graph)` takes only the graph; data goes to the returned
`CompiledGraph`'s `Execute(params IData[] inputs)`, the call you repeat. `ComputeContext`
also offers `Eval(...)` (the `OnnxEngine.Eval` overloads, plus `Eval<T>(Tensor<T>)`
returning a typed `TensorData<T>`), `Execute(ComputationGraph graph, params IData[] inputs)`,
`Run(ComputationGraph graph, params NamedModelParam[] inputs)`, and `ExecuteWithState(...)`
(for models that carry state). `TensorData` implements `IData`, so pass `TensorData` values
directly — as they are, to be consumed, or through `.Shared()` or `.TryConsume()`.
`Execute`, `Run` and `CompiledGraph.Execute` return `NamedModelParam[]`; read each output
with `namedModelParam.ToTensorData()` then `CopyMemory<V>()`, or `ValueAt<V>(i)` for one
element, V being the elements' CLR storage type (`float` for `float32`).
`ExecuteWithState` returns `(NamedModelParam[] regularOutputs, ComputationGraph updatedGraph)`
— feed the updated graph to the next call. `Eval` returns `TensorData` (or `TensorData[]`)
directly.

### Stopping a run

Give a run a `CancellationToken` on its `RunSettings` (the record that also carries
`ShrinkArenaAfterRun`), and a cancelled call ends in an `OperationCanceledException` instead
of returning outputs:

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

A token already cancelled when the call is made is refused before anything is fed: no input
is converted to a runtime value, no feed is locked, nothing is consumed, and the call can be
made again with the same inputs. A run stopped after it started has consumed what it was fed
as it is, like a run that fails — so to retry, pass `.Shared()`.

Cancelling mid-run sets ONNX Runtime's terminate flag, which its executor reads **between
nodes**, so the wait is the remainder of the running kernel. The probe behind the figures
below is `TerminateLatencyProbeTests` (`Purpose=Manual`):

- **The wait tracks one kernel.** On a chain of matmuls the run returned within one kernel's
  duration of the flag, whether a tenth or nine tenths of the run remained: 0.4-0.6 s with
  0.7 s kernels, under 0.1 s with 0.1 s kernels, single-digit milliseconds with 2 ms kernels.
  Below about 10 ms the floor is thread rescheduling, not ONNX Runtime. Budget for the model's
  **longest single operator**, not the step.
- **A graph whose work is one kernel cannot be stopped.** A single 4096x4096 matmul flagged a
  tenth of the way through ran the full 0.7 s and returned its outputs. The same holds for any
  graph's last kernel.

The figures come from one four-core CPU machine; treat them as the shape, and measure with the
probe or your own model where you deploy.

Stopping is best effort. A run that finishes before the flag is read **succeeds** with valid
outputs; the absence of an `OperationCanceledException` does not mean the token was ignored.
The flag is on that run's options, not the session, so the next run of the same compiled graph
proceeds normally.

**Where there is no per-call override, set it on the context.** Only `CompiledGraph`'s run entry
points take a `RunSettings` per call. `ComputeContext.Execute` / `Run` / `Eval`, and a training
rig's `TrainStep`, `Train` and `Fit`, run on their context's settings:

```csharp
using var ctx = new ComputeContext(backend)
{
    RunSettings = new RunSettings { CancellationToken = cts.Token },
};

var outputs = ctx.Execute(graph, inputs);   // stops when cts does
```

`RunSettings` is a record with an `init`-only property, fixed when the context is built. Giving
a training rig's `runtimeContext` one is how a long `Fit` is stopped — see
[Compute contexts](training.md#compute-contexts-mergecontext-and-runtimecontext).

## Backend selection

- Add a backend package as a dependency: `Shorokoo.LinuxCPU`, `Shorokoo.LinuxGPU`,
  `Shorokoo.WinCPU`, or `Shorokoo.WinGPU`. Each brings the native ONNX Runtime (CPU- or
  CUDA-flavored) for its platform. One is enough; a program that names its backends may
  reference several, or load them at runtime and reference none — see
  [Loading a backend at runtime](#loading-a-backend-at-runtime).
- With exactly one backend package referenced, auto-discovery (below) finds it on the first
  run. Set the backend explicitly when a deployment holds more than one (which discovery
  refuses), when you want a startup failure instead of one on the first run, or when the
  backend DLL is not deployed next to `Shorokoo.dll`:

  ```csharp
  using Shorokoo.Core.Backends;
  using Shorokoo.LinuxCPU;                                // the package you referenced

  DefaultBackend.Instance = new LinuxCpuBackend();
  ```

- `DefaultBackend.Instance` is the **default** backend: the one a `ComputeContext` naming no
  backend runs on, and the one a tensor is built for when fed without a context naming another.
  The first backend resolved is cached. Assigning `Instance` afterwards swaps it but does not
  unload a native ONNX Runtime already bound, and does not reach a `ComputeContext.Default` that
  has already resolved — so assign it at startup, before anything runs.
- A `ComputeContext` constructed with a backend runs there, and two contexts may name different
  backends — see [One model, two devices](#one-model-two-devices).
- **Exactly one deployed** is the rule for *discovery* only: a deployment with two backends for
  the same OS and naming neither is refused — see [Auto-discovery](#auto-discovery). It does not
  limit how many can run.
- `DefaultBackend.Describe()`, or `ComputeContext.Backend` where work is submitted, names the
  backend in use. See [Which device am I on?](#which-device-am-i-on).
- There are also PyTorch backends, `Shorokoo.PyTorch.Cpu` and `Shorokoo.PyTorch.Cuda`, and JAX
  backends, `Shorokoo.Jax.Cpu` and `Shorokoo.Jax.Cuda`, which run a model with PyTorch or with
  JAX and XLA in an embedded Python. They are never discovered: a program names one for the
  contexts that should use it — `new ComputeContext(new TorchCpuBackend())` — so they can sit
  beside an ONNX Runtime package without making discovery ambiguous. See
  [pytorch-backend.md](pytorch-backend.md) and [jax-backend.md](jax-backend.md).

### The backend types

Each backend package contains one backend, in a namespace equal to the package id. **The type
name spells the device `Cpu`/`Gpu`; the package, namespace and assembly spell it `CPU`/`GPU`** —
so `Shorokoo.WinGPU` contains `WinGpuBackend`, *not* `WinGPUBackend`:

| package (= namespace) | backend type | fully qualified |
|---|---|---|
| `Shorokoo.LinuxCPU` | `LinuxCpuBackend` | `Shorokoo.LinuxCPU.LinuxCpuBackend` |
| `Shorokoo.LinuxGPU` | `LinuxGpuBackend` | `Shorokoo.LinuxGPU.LinuxGpuBackend` |
| `Shorokoo.WinCPU` | `WinCpuBackend` | `Shorokoo.WinCPU.WinCpuBackend` |
| `Shorokoo.WinGPU` | `WinGpuBackend` | `Shorokoo.WinGPU.WinGpuBackend` |

All four implement `IShorokooBackend`, have a parameterless constructor, and differ only in
execution provider: the GPU ones append the CUDA provider on device 0, the CPU ones leave ORT on
its default provider.

The PyTorch and JAX backends follow the same naming, one per package, but are not ONNX Runtime
backends and take no part in discovery:

| package | backend type | fully qualified |
|---|---|---|
| `Shorokoo.PyTorch.Cpu` | `TorchCpuBackend` | `Shorokoo.PyTorch.Cpu.TorchCpuBackend` |
| `Shorokoo.PyTorch.Cuda` | `TorchCudaBackend` | `Shorokoo.PyTorch.Cuda.TorchCudaBackend` |
| `Shorokoo.Jax.Cpu` | `JaxCpuBackend` | `Shorokoo.Jax.Cpu.JaxCpuBackend` |
| `Shorokoo.Jax.Cuda` | `JaxCudaBackend` | `Shorokoo.Jax.Cuda.JaxCudaBackend` |

### Auto-discovery

If you never assign `DefaultBackend.Instance`, its first read resolves a backend once and
caches it:

1. If one of the four backend assemblies is **already loaded** in the process, its backend is
   used, to avoid binding a second native. Only assemblies targeting the running OS that expose
   a backend count; otherwise discovery falls through to step 2. A backend loaded by
   `IsolatedBackend.Load` is never a candidate: it lives in its own load context, loaded because
   the program named it.
2. Otherwise the folder next to `Shorokoo.dll` is probed for the known `Shorokoo.{Platform}.dll`
   files; only those targeting the current OS are candidates. Nothing else is searched: no other
   directory, no NuGet cache, no other assembly name.

A single candidate is taken as-is — a lone GPU backend is chosen even with no CUDA runtime
present. **Two or more are refused**, in either step, with an `InvalidOperationException`
naming them. From the folder probe (step 1 says `already loaded in this process` instead of
`deployed in '<folder>'`, and counts only assemblies that expose a backend):

> `Several Shorokoo backends are deployed in '<folder>': Shorokoo.WinCPU (CPU),
> Shorokoo.WinGPU (CUDA). Discovery picks the backend for a program that named none, and
> this deployment gives it no way to choose. Say which you mean: assign
> DefaultBackend.Instance before the first run to make one of them the default.
> To run several at once, give each ComputeContext its own backend -- new ComputeContext(new
> LinuxGpuBackend()) -- and where they need separate native ONNX Runtimes, load
> them with IsolatedBackend.Load.`

Discovery does not prefer the GPU when a CUDA runtime is present. In a deployment holding both
packages their native ONNX Runtimes have already collided — each ships `libonnxruntime.so`
(`onnxruntime.dll`) at the same path, so only one is deployed, chosen by NuGet's conflict
resolution — and the managed DLL discovery would pick says nothing about which native is there.
([One model, two devices](#one-model-two-devices) separates them, which is why a program running
both deploys each native in its own folder.) Backends for *different* OSes are not ambiguous,
since only those targeting the running OS are candidates. Carrying all four is two for whichever
OS you run on, and refused on both.

Step 1 takes precedence. If exactly one backend assembly is already loaded at the first run —
naming its backend type anywhere in a method your program runs is enough — that one wins and
the folder is never probed. The refusal applies when the *deployment* makes the choice; it does
not guarantee an ambiguous build cannot run.

An ambiguous deployment usually arrives by accident through a shared library that references a
backend, which flows to everything referencing it; keep the backend in the executable — see
[Or keep it to two processes](#or-keep-it-to-two-processes). To deploy both on purpose, see
[Deploying two backends](#deploying-two-backends).

Referencing a backend package is enough for step 2: the package copies its DLL to your output
folder, so discovery finds it whether or not your code mentions the backend type. On a Linux
sandbox that ships only `Shorokoo.LinuxCPU`, discovery picks it with no setup.

If no backend is found, the first run throws `InvalidOperationException`:

> `No Shorokoo backend is set and none was found in '<folder>'. Set one at
> startup -- e.g. DefaultBackend.Instance = new LinuxCpuBackend(); (or the
> backend from whichever Shorokoo.{WinCPU,WinGPU,LinuxCPU,LinuxGPU} package you
> reference) -- or add such a package as a dependency.`

### Which device am I on?

`Compile(...)` and `Execute(...)` look the same on a CPU build and a GPU one. Ask instead:

```csharp
using Shorokoo.Core.Backends;

Console.WriteLine(DefaultBackend.Describe());         // Shorokoo.WinGPU (CUDA device 0)
Console.WriteLine(ComputeContext.Default.Backend);    // the same, at the point work is submitted
```

`BackendDescription` carries the `Name` of the supplying assembly, the `Device`
(`ComputeDevice.Cpu`, `Cuda`, or `Other` for a backend you wrote against a third execution
provider), and the `CudaDeviceId` a CUDA backend allocates on (null otherwise). Record it in a
training run's log; which device produced the numbers cannot be reconstructed later.

Two related entry points:

- `DefaultBackend.Current` is the live backend **or null**; unlike `Instance` and `Describe()`,
  reading it does not resolve one. Use it to tell "nothing chosen yet" from "already bound".
- `DefaultBackend.RequireDevice(ComputeDevice.Cpu)` throws unless the live backend is on that
  device. Put it at the top of a program whose correctness depends on where it runs — a check
  that must not contend with a training run holding the card — so it fails at startup, naming
  the live backend, instead of quietly sharing the GPU:

  ```csharp
  DefaultBackend.RequireDevice(ComputeDevice.Cpu);   // before anything runs
  ```

### Loading a backend at runtime

A program need not reference a backend at compile time. `BackendPackage.TryLoad` takes a path
and returns a backend, or a reason (not an exception) when it does not fit the machine — so one
executable can carry backends for several platforms and pick at startup.

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
declaration from the file's metadata, so a backend for another OS or architecture, or one whose
native libraries are not deployed with it, is refused without loading it or its ONNX Runtime.
`BackendProbe.Reason` says which (`WrongOperatingSystem`, `MissingNative`,
`MissingCudaRuntime`, `MissingCudaDriver`, …) and `Detail` names the offending file or library.

One check runs native code: a backend declaring a CUDA requirement is verified by binding the
CUDA runtime and querying the device's memory, which initializes this process's CUDA context on
the card if it has none — so probing a GPU backend touches the driver even when the answer is
no. A backend that brings its own CUDA libraries and needs only the driver (the
[PyTorch](pytorch-backend.md) and [JAX](jax-backend.md) CUDA backends) declares that instead;
it is verified through the driver API, which initializes the driver but creates no context, and
refused as `MissingCudaDriver` where there is no driver, one too old, or no device. Every other
rejection — wrong OS, wrong architecture, a missing native — is decided from metadata and file
paths alone.

A backend's natives are looked for in both places a .NET build puts them: flat beside the
backend assembly, and under `runtimes/<rid>/native/` next to it. ONNX Runtime's native packages
copy their library to the output root through build props that fire on Windows only, so a
source build of a backend is flat on Windows and under `runtimes/linux-x64/native/` on Linux. A
program that installs `Shorokoo.LinuxCPU` (or any backend package) from NuGet gets the
`runtimes/` layout on *every* platform, because those props live in the ONNX Runtime package's
`build/` folder and do not reach a consumer that gets ONNX Runtime transitively. The native ONNX
Runtime a loaded backend binds is resolved the same way.

Each backend loaded this way gets its own load context, so several run side by side without
sharing a native runtime.

### Feeding a run: consumed, shared or tried

A tensor fed to a run **as it is** is given to that run. The run takes it when it starts — the
tensor is dead from then — and its memory goes to the run's backend, which releases it before
the call returns, or writes one of the run's outputs into it where the graph allows
([below](#a-run-that-writes-an-output-into-what-it-consumed)). That suits a batch built for one
call:

```csharp
var result = compiled.Execute(batch)[0].ToTensorData();
// batch is consumed: reading it throws, naming the run that took it
```

To use a tensor after the call, pass it `.Shared()`. The run only reads it, holding a reader
lock so nothing can delete it meanwhile, and it is alive and unchanged afterwards:

```csharp
var weights = TensorData([4L, 4L], w);
var first  = compiled.Execute(x1, weights.Shared());
var second = compiled.Execute(x2, weights.Shared());   // weights is still there
```

`.TryConsume()` consumes the tensor if nothing else is reading it when the run starts, and reads
it otherwise. Fed as it is, a tensor another run is reading is refused — the call throws
`InvalidOperationException` naming the run that holds it — since consuming it would take its
memory from under that run.

| Fed as | The run | Afterwards |
|---|---|---|
| `t` | takes it when it starts; refuses it while another run is reading it | dead |
| `t.Shared()` | reads it, holding a reader lock | alive and unchanged |
| `t.TryConsume()` | takes it if nothing else is reading it, reads it otherwise | dead, or alive if it was read |

- **Consumption is irrevocable.** A run that fails, or is stopped, after it started has still
  consumed what it was fed as it is. A run refused before it starts — a cancelled token, a dead
  feed, or a feed being read that it would have to consume — takes nothing: everything it could
  refuse over is checked first. The exception is a race: a feed that dies, or that another run
  starts reading, between those checks and this run's taking it refuses the run part-way, and
  what it had taken stays consumed.
- **One tensor fed twice in one call** is taken at most once: read if any occurrence is
  `.Shared()`, otherwise consumed if any is bare, otherwise tried.
- **Composites apply the mode to everything they hold.** `TensorDataStruct`, `TensorDataSequence`
  and `OptionalTensorData` have `.Shared()` and `.TryConsume()` too; fed as it is, a struct or
  sequence gives the run every tensor it holds. So does a training checkpoint — see
  [What a training step consumes](training.md#what-a-training-step-consumes). A sequence that
  holds its elements as its own — the copy a sequence's `To`, `CopyTo` or `ToHost` makes — is
  held element by element: fed as it is, it is refused where another run is reading one of its
  elements; fed `.Shared()`, none of its elements can be deleted while the run reads it; and an
  element fed separately in the same call counts as another occurrence of that element, so
  `Execute(e.Shared(), s)` reads `e` and consumes the rest of `s`, `e` living on without the
  sequence. A struct field can be given its own mode when the struct is built —
  `def.FromOrderedData(tokens, mask.Shared())` — and a `.Shared()` field is read however the
  struct is fed. A struct fed `.Shared()` has every field read; otherwise each field is fed as it
  was given, or as the struct is.
- **The error names the call to change.** Reading a consumed tensor throws
  `ObjectDisposedException` naming the run that took it — its graph and context — and the input
  it fed; the remedy is `.Shared()` at that call.
- On a tensor, struct, sequence or optional, `.Shared()` and `.TryConsume()` return a
  `SharedInput`: an `IData` carrying the value and its `Mode`, accepted wherever an input is. On a
  training checkpoint they return a new checkpoint over the same tensors with its `FeedMode` set —
  the original keeps its own — which its derivations (`WithStep`, …) and `rig.AdoptCheckpoint`
  keep, since they share its tensors. `Run`, which takes `NamedModelParam`s, reads each one's
  `FeedMode` instead — `null` for as it is — and on a parameter they return a copy over the same
  data with its `FeedMode` set: `graph.Run(p.Shared())` reads `p`'s tensor and leaves `p` as it
  was.

**Memory the run cannot read where it is.** A run reads its inputs in its backend's memory. A
tensor anywhere else — every tensor built from a C# array, and one allocated by another device or
runtime — is fed through a copy in the run's memory, and the mode decides what becomes of that
copy:

- **Consumed**: the contents are copied into the run's memory, the tensor is dead and its own
  memory released at the feed, and the run consumes the copy. On a card that copy is ONNX
  Runtime's own, into the session's arena: the run hands the session the contents in host memory
  — the tensor itself where it is already a host value of the session's runtime, such as an output
  an earlier run brought back, whose memory is then released as the run returns. An input an
  output may be [written into](#a-run-that-writes-an-output-into-what-it-consumed) is the
  exception: it is copied onto the card before the run, so the output has card memory to be
  written into.
- **Read**: the copy is made on the first such read and kept. It is a `TensorData` of its own:
  held by the source tensor, locked by each run that reads it, attached to the context that read
  it (so it shows in `context.Tensors`), and reused by every later shared read in that memory.
  Writing to the source tensor (`AccessModifiableMemory` and the like) retires the copy — the next
  read copies what was written — and the copy also goes when the tensor is deleted or consumed.

### A run that writes an output into what it consumed

ONNX Runtime keeps every input of a run until the run ends. Measured on a card: a 64 MiB input in
the session's own arena, read only by the graph's first node and held by nothing but the run, still
held its memory when the last node ran, so an arena with room for the run's own two 64 MiB blocks
could not also take the input. A run can instead write an output **into** a consumed input's memory
— output aliasing — and the same run then fits.

This is correct only where nothing reads the input after the output is written, so it happens only
for outputs a graph's lowering marks after proving that: every node reading the input is one the
output's writer waits for, and the writer reads the input only as an element-wise update does.
The only lowering that marks outputs is the training rig's step, which pairs each updated state
field with the field it replaces
([A step writes its state over the state it consumed](training.md#a-step-writes-its-state-over-the-state-it-consumed));
a graph you compile yourself marks none, and its outputs always get memory of their own. ONNX
Runtime rewrites a graph before running it, which can change which nodes read an input, so the
backend re-proves each marked pair over the graph it will run and drops any it cannot.

A run writes a marked output into an input's memory only where:

- **it consumed that input** — `.Shared()` memory is the caller's, and is read and left as it was;
- **it was fed as no other input** — one tensor fed twice is read as both;
- **the output is produced in that memory**, with the input's element type and shape — the shape
  ONNX Runtime settled when it built the session, so a graph compiled with open shapes writes its
  outputs where it always does. On a GPU backend, an output a run fetches back to the host is not
  produced in the card memory the input is in.

Otherwise nothing differs: the outputs are new `TensorData` objects attached to the running
context, the consumed input is dead, and the values are the same.

### A tensor's lifetime: locks and deletion

A `TensorData` **is** its memory: one object per allocation, no second tensor naming the same
bytes. It records the backend that allocated it (`AllocatingBackend`) and where the memory is:
`Space` for the device, and `Location` for the device plus the allocating runtime. That backend
releases the memory, whichever contexts the tensor has been attached to, or none. A tensor does
not know which contexts it is attached to — see
[Moving data between contexts](#moving-data-between-contexts).

**How a tensor ends.** In one of three ways; disposing a context it is attached to is not one:

| | |
|---|---|
| **Deleted** | `Delete()`, `Dispose()`, `TryDelete()` or `DeleteAsync(...)` — below. |
| **Consumed** | fed to a run as it is — or through `.TryConsume()` with nothing else reading it — which takes it when the run starts; see [Feeding a run](#feeding-a-run-consumed-shared-or-tried). |
| **Moved into an attribute** | `MoveToAttribute()`, which takes its contents — see [core-types.md](core-types.md#the-two-conversions-and-which-one-spends-its-source). |

Two kinds of tensor are also ended by what they belong to. A copy a run made to read a tensor it
could not read where it is ([above](#feeding-a-run-consumed-shared-or-tried)) is retired when that
tensor is written or ends, or lets its copies go — as a training step does with the copies it made
to read its batch. The elements of a sequence that holds them as its own — the copy a sequence's
`To`, `CopyTo` or `ToHost` makes — end when the sequence does, however it ends: an element read
after a run consumed its sequence names that run. The exception is an element a run is reading on
its own account when its sequence ends, which lives on; and a sequence one of whose own elements a
run is reading cannot be disposed.

A dead tensor records why, and every later access — reading its elements, feeding it, `To`,
`CopyTo`, `ToHost`, `Shared()`, `TryConsume()`, `MoveToAttribute` — throws
`ObjectDisposedException` saying so; for a consumed tensor it names the graph and context whose run
took it and says to pass it `.Shared()` at that call. `Shape`, `DType`, `ToString()`, `IsDisposed`
and where its memory was — `AllocatingBackend`, `Space`, `Device`, `Location` — keep working.
Ending a dead tensor does nothing, so disposing one twice, or at the end of a `using` over a tensor
a run has consumed, is harmless.

A tensor nothing references is reclaimed like any other object, its memory released through its
backend's ordinary path. Deleting chooses *when* the memory comes back; it is not needed to avoid a
leak.

What release frees depends on where the bytes are. A tensor whose buffer the runtime allocated — a
run's output, a copy onto a card, an `AllocateUninitialized` on a real context — holds native
memory (the card's own on a CUDA context), and releasing it hands that back immediately. A tensor
built from a C# array holds a managed array, which stays the collector's to reclaim: releasing the
tensor lets go of the array, and what it frees at once is the copies runs read it through
([above](#feeding-a-run-consumed-shared-or-tried)), which are native and may be on a card.

**What a run holds.** A run takes a reader lock on every tensor it reads — fed `.Shared()`, or
through `.TryConsume()` while something else held it — and holds it, and a reference to the tensor,
until it returns, however it returns. Any number of runs may read one tensor at once. While any
holds its lock the tensor cannot be deleted: `Delete()` and `Dispose()` throw
`InvalidOperationException`, and `TryDelete()` declines. A tensor a run consumes it holds by taking
it, which no other run can then do. Copies out of a tensor outside any run take the same lock while
they copy — `ToHost()`, `CopyTo(...)`, `CopyMemory()`, `ValueAt()`, `CopyRawMemory()` and the copy
`MoveToAttribute()` makes — so a run that would consume the tensor meanwhile is refused, and a
delete throws. A span from `AccessMemory()` escapes the call, so no lock covers it: keep the tensor
alive and fed to nothing while you hold one.

The lock — or, for a consumed feed, the take — happens inside the run, one feed at a time, so
nothing is held while the call is being set up. A deletion in that window ends the tensor before the
run claims it: the lock or take is refused, `Execute` throws `ObjectDisposedException`, and what it
had already taken of its other feeds stays consumed. No run reads freed memory or returns a wrong
answer, but do not delete a feed from a second thread while a run of it is starting; see
[A feed deleted while a run is starting loses that run](limitations.md#a-feed-deleted-while-a-run-is-starting-loses-that-run).

Disposing a `ComputeContext` throws while a run of it is in flight or while it holds a lock on
anything, and disposing a compiled graph throws while one of its runs is in flight.

**Deleting.** Three calls, differing in what they do when a run is reading the tensor:

| | What it does |
|---|---|
| `Delete()` / `Dispose()` | The same operation: ends the tensor and releases its memory now. Throws if a run is reading it. |
| `bool TryDelete()` | The same if no run is reading the tensor. If one is, it changes **nothing** and returns `false`: the tensor stays readable and no run is disturbed. `true` for a tensor already dead. |
| `Task<bool> DeleteAsync(timeout, cancellationToken)` | Ends the tensor at once, asks whatever is reading it to stop, and waits up to `timeout` for the memory to come back. |

With `DeleteAsync`, deletion is immediate and only *reclamation* waits:

- **The tensor is deleted either way.** `false` means the memory had not come back within the
  budget, never that the deletion did not happen. Do not retry: the memory comes back when the run
  ends.
- **A timeout never rolls back.** By then the run has been asked to stop and has discarded its
  work.
- **The `CancellationToken` cancels the wait, not the deletion.**

Neither call is prompt, for the reason [stopping a run](#stopping-a-run) is not: the wait is the
longest single operator in flight, or a whole run where the graph is one operator. A backend that
ignores the request makes `DeleteAsync` slow, never unsafe — the wait ends when the run finishes.

**Host memory.** A tensor built from a C# array belongs to no context and no runtime: its
allocating backend is `HostBackend.Instance`, the framework's own managed memory, which every host
backend can read. `ComputeContext.Host` names host memory as a target for `To` and `CopyTo`. It
holds nothing and runs nothing — `Compile`, `Execute`, `Run` and `Eval` all refuse, naming a real
context — and it cannot be disposed.

A graph's own literals are not tensors and have no lifetime — they are
[`TensorAttribute`s](core-types.md#two-kinds-of-concrete-tensor-tensordata-and-tensorattribute),
immutable and attached to nothing.

### Moving data between contexts

A tensor never moves: its memory stays where it was allocated. The three operations decide whether
a context gets *this* tensor or a copy, and none changes the tensor it is called on:

| | Result |
|---|---|
| `t.To(context)` | `t` itself, if `context`'s backend can read its memory as it stands; otherwise a new copy in `context`'s memory. Either way attached to `context`. |
| `t.CopyTo(context)` | Always a new, independent copy in `context`'s memory, attached to `context`. |
| `t.ToHost()` | `t` itself, if the host can read its memory; otherwise a new copy in the framework's own host memory, attached to nothing. |

A backend can read a tensor's memory as it stands when it is **the same device and the same
runtime**. The framework's own host memory — every tensor built from a C# array — counts as every
host backend's, so `To` hands such a tensor to a host context as it stands; a run there still reads
it through a copy its runtime builds, since a session takes runtime values only. A device allocation
is meaningful only to the runtime that made it, on its own device: two backends over one loaded ONNX
Runtime share a card allocation (two instances of the CUDA backend, say); a CPU backend beside them
reads it through a copy; and two isolated runtimes on one card, as `IsolatedBackend` produces, copy
through the host. A run's outputs come back on the host unless retained
(`CompiledGraph.Execute(inputs, retainOnDevice)`), so the copy arises only for a tensor you put on
or kept on the card.

```csharp
var onCard = cuda.Compile(model).Execute([input], [true])[0].ToTensorData();  // device memory
var onHost = onCard.To(cpu);        // one copy across the bus; onCard is untouched
var same   = onHost.To(otherCpu);   // no copy: the very same object
var mine   = onHost.CopyTo(cpu);    // always a copy
```

**Where `To` or `ToHost` needs no copy, it returns `t`**, and what you do to the result you do to
`t`. Fed to a run as it is, it is consumed: `ctx.Execute(graph, t.To(ctx))` consumes `t` itself
where no copy was needed — a tensor built from a C# array, on a CPU context — and only the copy on a
card. Deleted, it is gone: `using var h = t.ToHost();` deletes `t` at the end of the block when `t`
was already host-readable. For an independent tensor use `CopyTo`; to keep `t` past a run, feed it
`.Shared()`.

**Attachment is bookkeeping, not ownership.** A context keeps a weak list of its attached tensors,
`context.Tensors`: its runs' outputs, what its runs read, and what `To`, `CopyTo` and
`AllocateUninitialized` placed for it. The list never keeps a tensor alive or ends one; a tensor
that dies or is collected drops out. `context.Detach(t)` takes a tensor off the list — refused while
a run of that context is reading `t` — and never deletes it. Disposing a context releases its
compiled sessions and leaves every tensor as it was: a run's outputs outlive its context. The list
is also what a context's device-memory budget counts, so on a budgeted context `To` and `CopyTo` can
be refused — see [A context's device-memory budget](#a-contexts-device-memory-budget).

`TensorDataStruct` and `TensorDataSequence` take the same three operations, applied to the tensors
they hold. A struct comes back as itself where nothing had to be copied. A sequence owns its
elements, so where any has to be copied, the whole sequence is.

### Feeding a large input without a second copy

A feed built the ordinary way exists twice at feed time: you fill a managed array, and the runtime
copies it into its own buffer. Where the input dominates the step's peak, that doubles the largest
thing in the run. Two operations remove one half each.

`ComputeContext.AllocateUninitialized` hands you the runtime's buffer to fill in place:

```csharp
using var cpu = new ComputeContext(new LinuxCpuBackend());

var batch = cpu.AllocateUninitialized<float32>(new Shape(64L, 3L, 224L, 224L));
batch.WriteMemory<float>(ReadImagesInto);   // no managed array in between
```

`Shape` is a class, not a collection type, so the shape is `new Shape(…)` or a `long[]`, not a
`[…]` collection literal.

**Fill it through `WriteMemory`, not through a bare span.** On a real backend the buffer belongs to
the runtime and the tensor is the only thing keeping it alive. Taking a span is the tensor's *last
read*, so

```csharp
ReadImagesInto(batch.AccessModifiableMemory<float>());   // wrong: nothing roots `batch`
```

leaves no reachable tensor during `ReadImagesInto`, and the runtime value's finalizer can free the
block while you are writing into it. Being in scope is not being reachable. `WriteMemory` keeps the
tensor alive across the call, as `CopyMemory` does for reading.

The buffer is not initialized — it holds whatever was last there, so fill all of it — and the
tensor is attached to the context as a `CopyTo(context)` result is. The `(shape, dtype)` overload
does the same where the element type is known only at runtime. On a CUDA context the buffer is card
memory, which the host cannot write through a span (`TensorData.IsHostResident` is false); use
`CopyTo` to get bytes there.

The other half is feeding it **as it is**, as every feed not passed `.Shared()` is. The run takes
the buffer where it stands — memory the context's own backend allocated, which its runs address
without a copy — and returns it to the allocator as it returns:

```csharp
var loss = compiled.Execute(batch)[0].ToTensorData();
// batch is consumed: reading it throws, and its buffer went back to the allocator with the run
```

A consumed tensor's memory is released as soon as the run finishes with it, however the run ends;
a feed passed `.Shared()` is released when *you* let go of it, which for a batch built per step is
at the next collection. See [Feeding a run](#feeding-a-run-consumed-shared-or-tried) for the rules.

**What consuming does not buy.** ONNX Runtime's memory planner gives every graph input one extra
use count, so a caller can still read a feed after `Run` returns; the planner therefore never
recycles an input's buffer for the run's intermediates, and no session or run option changes that.
Consuming moves the release from "whenever the caller lets go" to "the instant the run returns".
The only reuse of a consumed input inside a run is an output written into it where the graph proves
nothing reads the input afterwards — [above](#a-run-that-writes-an-output-into-what-it-consumed) —
which only a training step's state gets.

### One model, two devices

One process can run one model on the CPU and on the card. A `ComputeContext` constructed with a
backend compiles and runs there, and two contexts may name different backends:

```csharp
using Shorokoo.Core.Backends;
using Shorokoo.LinuxCPU;
using Shorokoo.LinuxGPU;

var cpu  = new ComputeContext(new LinuxCpuBackend());
var cuda = new ComputeContext(new LinuxGpuBackend());

var onHost = cpu.Execute(graph, input.Shared());   // the host, reading input and leaving it
var onCard = cuda.Execute(graph, input);           // the same graph, the same input, the card
```

The model is compiled once, into one assembly, and both contexts run it — so a check on the CPU
tests the model the GPU run is training. `ComputeContext.Backend` says which device each context
uses, and `CompiledGraph.Backend` the backend a compiled graph was built on.

**Tensors you build are not tied to a runtime.** A `TensorData` you build holds managed bytes; its
allocating backend is `HostBackend.Instance`, the framework's own memory. Building and exporting a
model needs no runtime; a runtime enters only when the tensor is fed, and either context accepts it
— a session hands what it is fed to its own runtime, building it there if needed.

Whether that costs a copy depends on where the tensor is and how it is fed. A run reads as it
stands only memory its own runtime can address — never a C# array's, never another runtime's
allocation — and reads anything else through a copy in its own memory
([Feeding a run](#feeding-a-run-consumed-shared-or-tried)). Fed `.Shared()`, the tensor keeps that
copy, so it is one copy per (tensor, runtime) however many runs follow, until the tensor is
written; fed as it is, the copy is made for that run and consumed with the tensor. A tensor an
execution provider left on the card (`TensorData.IsHostResident` is false, as a
[resident training run](training.md#keeping-training-state-on-the-device) produces) crosses the
same way, its bytes brought through the host by the backend that allocated them.

#### Deploying two backends

The two packages deliver their native ONNX Runtime at the same path
(`runtimes/<rid>/native/libonnxruntime.so`), so referencing both normally deploys one of the two
runtimes, with the other's provider libraries stranded beside a core that cannot use them. That is
why [auto-discovery](#auto-discovery) refuses such a deployment.

How to avoid it depends on whether the two backends need separate native runtimes.

**One runtime, two providers — the usual case.** A native ONNX Runtime serves every execution
provider compiled into it, and the CUDA build includes the CPU provider. Deploy the GPU package, and
add the CPU package for its backend type only, with `ExcludeAssets="native"` so it brings no second
runtime:

```xml
<PackageReference Include="Shorokoo.LinuxGPU" Version="..." />
<PackageReference Include="Shorokoo.LinuxCPU" Version="..." ExcludeAssets="native" />
```

**Then name the default explicitly, before anything runs.** Both backend assemblies are deployed,
so [auto-discovery](#auto-discovery) has two candidates and refuses — on the *first* read of
`DefaultBackend.Instance`, which may be a framework call you did not write. Constructing a backend
does not settle it; assigning does:

```csharp
DefaultBackend.Instance = new LinuxCpuBackend();   // startup, before anything runs

var cpu  = new ComputeContext(DefaultBackend.Instance);
var cuda = new ComputeContext(new LinuxGpuBackend());
```

**Two runtimes.** Two ONNX Runtime *builds*, or two versions, in one process — a vendor build beside
the stock one, say. Give each native its own folder and load the second with `IsolatedBackend`:

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

`ShorokooBackendNatives` items are read by a target the `Shorokoo.OnnxRuntime` package imports;
each lands in `ort/<BackendId>/` in the output. An execution provider's library must sit beside its
core, so deploy a package's whole native folder.

Do not omit the direct reference to `Shorokoo.OnnxRuntime`. Reaching the backend package with
`ExcludeAssets="all"`, as the next block does, cuts the glue's usual route — without this line the
target never loads, the items above are silently ignored, `ort/` is empty, and the first
`IsolatedBackend.Load` fails with a `FileNotFoundException` naming a native that was never copied.
The package is not a backend and never becomes a discovery candidate.

**The backend's own assembly must be deployed too, but not beside `Shorokoo.dll`** — two backend
assemblies there is the ambiguity [auto-discovery](#auto-discovery) refuses. Put it in its native's
folder and point `ProbeDirectory` at it:

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

The ONNX Runtime wrapper and the glue are looked for beside the backend first and beside
`Shorokoo.dll` second, so only the backend's own assembly has to move.

`Name` is what `Backend.Name` reports; the backend assembly cannot tell two loads of itself apart,
so give it something useful in a log. It is only a label: the backend is identified by its native,
so loading one native twice under two names is refused. A loaded backend lasts for the life of the
process — its native runtime holds thread pools, arenas and allocators, and nothing unloads it.
Loading the same spec twice returns the same backend.

Only the ONNX Runtime wrapper, the glue and the backend assembly are private to an isolated
backend; the core Shorokoo assembly and your model's assemblies stay shared, which is what lets one
context be handed the other's data.

A model with *sequence* outputs runs here like any other, on a card or the host: ONNX Runtime
materializes a run's sequence output in host memory whichever execution provider produced it. A
sequence cannot hold a tensor left in device memory — see
[A sequence's elements live in host memory](limitations.md#a-sequences-elements-live-in-host-memory).

#### Or keep it to two processes

Splitting the work across two executables over a shared, backend-free model library remains a good
choice where the halves are separate jobs — a long training run and an occasional check. It costs a
process instead of coordinating two devices in one:

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

This layout breaks if the shared library carries a backend reference — a `PackageReference`, or a
`ProjectReference` to an executable that carries one. Either flows into the referencing project's
output folder, two backends are deployed, and, since neither executable named one,
[auto-discovery](#auto-discovery) refuses to choose. Keep the backend in the executable, and let
nothing reference an executable.

### Device memory (GPU backends)

ONNX Runtime allocates device memory from a BFC arena that extends in blocks and, unless asked to
shrink (below), never returns them. The *extend strategy* sets each new block's size, and ORT's two
strategies suit opposite situations:

- **`NextPowerOfTwo`** (ORT's default) makes each extension at least as large as everything the
  arena already holds. The big, splittable regions suit an unpredictable series of allocation
  sizes, but once sizes settle the doubling is overshoot — why a long training run can hold far
  more of the card than its steps use.
- **`SameAsRequested`** extends by exactly what was asked, so a settled run's arena tracks it. But
  an exactly-sized region cannot serve a later, larger request: a session whose input shapes keep
  growing strands every region it outgrows.

**Shorokoo picks between them per session.** Measured on the CPU arena — the same allocator with the
same two strategies — over four chained matmuls. Both columns vary by a MiB or two between runs, and
on the mixed rows a run can put the two within a MiB of each other, so read every ratio below as
approximate and near-ties as ties:

| shapes fed to the session | `SameAsRequested` | `NextPowerOfTwo` |
|---|---|---|
| one shape, ten runs | **11–12 MiB** | 16 MiB |
| alternating 2048/512, twenty runs | **20–25 MiB** | 28–33 MiB |
| largest first, then settled | 17–18 MiB | 15–16 MiB |
| shuffled from four sizes, twenty runs | 34 MiB | **31 MiB** |
| growing, then settled | 34–35 MiB | **31 MiB** |
| growing 256 to 2048 | 23 MiB | **15 MiB** |
| growing 256 to 2048, 16 MiB arena | does not fit | **15 MiB** |

The measurement is a test in the Shorokoo repository (`ArenaExtendStrategyProbeTests`,
`Purpose=Manual`), not something you can run against the package, taken on the **CPU** arena — the
same `BFCArena` and strategy enum the CUDA provider uses. It reads the arena from glibc's
`mallinfo2`, so it runs on Linux only and reports "unavailable" elsewhere.

Both tables here come from **one machine**: the host rows under Linux, the card rows below under
Windows, same CPU and same RTX 4090 — enough to compare ratios, not a like-for-like pair. Four
consecutive runs reproduced the host rows within the ranges shown, except `NextPowerOfTwo` on rows
4 and 5, which read 31–32 MiB; the settled-series row did not move at all.

A second probe (`ArenaExtendStrategyCudaProbeTests`, `Purpose=Manual`) measures what each strategy
costs **one real training step on a card**: a 49,214,208-parameter decoder-only transformer — 6
layers, width 384, 6 heads of 64, sequence 1024, vocabulary 50,257, fp32, `AdamWOptimizer` — at
batch 8 on a 24,564 MiB RTX 4090, read from the step's own session arena through
`ComputeContext.RunStats`, not from the device:

| the training step's own arena | `SameAsRequested` | `NextPowerOfTwo` |
|---|---|---|
| in use at its highest, step 0 | **7,305 MiB** | 7,441–7,485 MiB |
| in use at its highest, settled | **8,009 MiB** | 8,043–8,101 MiB |
| taken from the device, step 0 | **8,864 MiB** | 9,233–9,249 MiB |
| taken from the device, step 1 onwards | **15,508 MiB** | 17,425–17,441 MiB |
| blocks held, settled | 278 | 14–15 |

Over two runs the `SameAsRequested` column repeated to the byte while `NextPowerOfTwo` moved by
16 MiB and one block. Treat both tables as the shape, not your machine's numbers; measure your case
with `ComputeContext.RunStats`, or `DeviceMemory.Sample()`, around your own run.

Neither strategy wins outright; the winner depends on whether a session's allocation sizes settle.
A training run feeds one input shape to one compiled step — the host table's first row — where
exact-size extension holds about 1.3–1.45x less; on the card, on a real training step, **1.12x**
less. Two allocation sizes still favour exact-size extension (row 2, by 1.2–1.6x); with several,
ORT's doubling holds about 1.1x less, or ties (rows 3 to 5). The last two rows are input shapes
that grow without settling: each outgrown region is stranded, the doubling holds around 1.5x less
and — on a card with no room to spare — fits where exact-size extension does not.

**On the card, the strategy is the smaller effect.** The arena roughly doubles between step 0 and
step 1 under either strategy — 1.75x under exact-size extension, 1.89x under ORT's — in a single
extension: one block of 6,644 MiB where the region is exactly the request, 8,192 MiB where it is
rounded up to a power of two. The step never returns that block, so a settled training step holds
about twice what its steps use under **both** strategies (1.94x and 2.15x here). The strategy trims
that block; an arena limit prevents it — see below. A larger model of the same family (12 layers,
width 768, 162,129,408 parameters) at the same batch reached the card's whole 24,564 MiB at step 1
under both strategies and kept training there: no headroom, not a failure.

Which case a session falls in depends on **how its caller feeds it**, which is unknown when the
session is built. So `ArenaExtend` defaults to `Auto`, which is not an ORT value: it means
`SameAsRequested` **except where Shorokoo already knows the shapes differ**.

That is one case. A training rig keeps a compiled step per input shape up to a limit; fed more
distinct shapes, it falls back to a single step every later shape shares. By the time that step
exists the differing shapes have already occurred, and it matches the growing-shape row, where
exact-size extension strands a region per outgrown input and cannot fit under a budget. That step
gets the doubling; every other session gets exact-size extension.

**If your own session is fed shapes that keep growing, set the strategy** — Shorokoo cannot know it
before the feeds arrive. A strategy set on a context applies to the sessions that context compiles,
and `CompiledGraph.DeviceMemory` reports what a graph got:

```csharp
using Shorokoo.Core.Backends;
using Shorokoo.Runtime;

var ctx = new ComputeContext
{
    DeviceMemory = new DeviceMemorySettings
    {
        ArenaExtend = ArenaExtendStrategy.NextPowerOfTwo,   // ORT's doubling
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

`ShrinkArenaAfterRun` costs a synchronizing device allocation on every step to re-take what it
returned, so outside a budget it is worth it only when the card is shared with something that needs
the room between steps. It can also fail a run: ORT rejects it where the named device has no arena
allocator registered (an arena disabled through `ORT_DISABLE_ARENA`, say), so try it on a short run
before a long one. Under a budget it is on for every run. `LimitBytes` is a hard budget: what would
exceed it is refused, or fails with ORT's `BFCArena ... Failed to allocate memory for requested
buffer`, instead of using the rest of the device, so a figure set too low fails work that would have
fitted. An arena limit near what a step uses is also the one lever that prevents the step-1
expansion above: the same batch-8 transformer, its step's arena capped at 10 GiB, ran four steps
inside 8,864 MiB (exact-size extension) or 9,217–9,233 MiB (ORT's) and never took the extra block.
Its in-use peak across the four steps was 7,305 / 7,469–7,485 MiB, against step 0's 7,305 /
7,441–7,485 in the table, where the uncapped steps went on to settle at 8,009 / 8,043–8,101 — so the
cap clipped nothing step 0 did, and later steps ran in what the arena already held, on roughly half
the card. That was a cap on the step's arena alone; a context budget of 10 GiB leaves the arena less,
by what the context holds on the card, so size the budget as what the step uses plus what the rig
keeps there.

`ArenaExtend` is read **when a session is built** — the first inference call, or a training rig's
first `TrainStep` for a given input shape — so the context must carry it before the graph is
compiled on it. A graph keeps what it was built with, which is why `CompiledGraph.DeviceMemory`
reports the settled strategy rather than `Auto`. A context's settings are initialize-only, so its
budget is fixed with it. ORT reads `ShrinkArenaAfterRun` on every run, so a `CompiledGraph.Execute`
/ `Run` call can override it for that call on an unbudgeted context. The context's one-shot entry
points and a rig's `TrainStep` take no override and use the context's setting.

The static `DeviceMemory` class — readings, not settings — reports what the card is doing:

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
the same and folds the reading into `PeakUsedBytes`, and `ResetPeak()` starts a fresh peak. About
the numbers:

- They are the **device's**, not this process's — every other process on the card, a desktop
  session included, is in `UsedBytes`.
- Nothing samples on its own. `PeakUsedBytes` is the largest figure your own `Sample()` calls have
  seen. A 0.2 s step falls between the polls of a half-second `nvidia-smi` sampler, whose "peak" can
  be half the real one; a `Sample()` per step costs about a microsecond and cannot miss the step it
  follows.
- With no CUDA runtime installed both return `null` rather than throwing, so the call can stay in
  code that also runs on a CPU backend.
- The reading goes through the CUDA runtime directly, which initializes this process's context on
  the device if it has none — itself a few hundred MiB. Take the first reading after the backend is
  up, or that cost lands in your baseline.

**Scope: the context, its sessions and its runs — never the process.** A budget belongs to one
context: two contexts on one card each keep their own, and a tensor on both contexts' books counts
on both. ORT gives each session its own arena and reads its limit and strategy once while building
it, and the session keeps them for life — so a context configures the sessions it compiles, two
contexts may differ, and neither reaches the other's sessions. To run something under a different
budget or strategy, compile it on a context that carries one. ORT reads `RunSettings` per run, so
those are settled per call.

The tensors a context places on the card are not in any of those arenas: they come from one
allocator per card and runtime, shared by every context over that runtime whatever its settings (a
backend loaded in isolation has its own), and held for the life of the process. The budget counts
them by what is attached to the context, not by that allocator.

Where one process serves both a training loop and a variable-shape inference path, give each its
own context rather than one arena strategy for both.

The readings are the exception, and are not settings: `Read()` and `Sample()` go to the CUDA device
current for the calling thread — device 0, which the shipped GPU backends use — and
`PeakUsedBytes` is one process's record of its own run.

### A context's device-memory budget

`DeviceMemorySettings.LimitBytes`, on a context whose memory is a card's, is a budget on
**everything that context holds there**: the tensors attached to it in the card's memory and, while
one of its runs executes, that run's arena. It is the context's budget — not one arena's, and not
the process's:

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
on two contexts' books counts on both; one that dies, is collected, or is removed with `Detach`
drops out.

**A transfer it cannot take is refused before it allocates.** `To`, `CopyTo` and
`AllocateUninitialized` onto the context — and the card copy a run makes of a tensor it cannot read
where it is, such as a host tensor read by a run on the card — are refused when what is attached
plus what they would add exceeds the limit. So is a `To` of a tensor already on the card that the
context's backend reads as it stands: nothing is copied, but attaching it puts its bytes on the
books. A struct's or sequence's `To` and `CopyTo` are checked whole, before any part is placed —
except a sequence a run produced, whose elements are made only as they are read, so each is checked
as it is copied — and a part that fails takes what the rest placed off the books again, releasing
any copies already made. The refusal is an `InvalidOperationException` naming the budget, what is
attached and what was asked for:

```
CopyTo(context) of Tensor (8388608,):Float32 asks this compute context for 33554432 bytes of CUDA
device 0 memory, which its device-memory budget cannot give: the budget
(DeviceMemorySettings.LimitBytes) is 67108864 bytes, and 50331648 bytes of it are attached to the
context there, in 1 tensor(s), leaving 16777216. Delete what the context no longer needs, or give
the context a larger budget.
```

**A run's arena gets what the context leaves it.** A session's `gpu_mem_limit` is the budget less
the *discount*: what the context holds on the card outside that session's arena for the length of
the run — its attached tensors there, and those the run reads there or copies there to read. For a
tensor another runtime holds on the same card, that is both the tensor and its copy: the read
attaches both.

A tensor already on the card is read in place and never enters the arena, so it stays in the
discount for the whole run. Measured: a session whose arena was capped at 32 MiB read a 64 MiB
input from the card with its arena never above 256 bytes, while the same bytes handed to an ONNX
Runtime session directly from host memory had to be copied into its arena and did not fit. Through
Shorokoo, a host tensor fed to a run on the card takes one route or the other depending on how it
is fed. Read (`.Shared()`), it is copied onto the card before the run, outside the arena, kept for
later reads, and counted in the discount. Consumed, it is handed to the session in host memory and
copied into the arena, where it counts against the session's limit rather than reducing it, so a
loop feeding every run a fresh host batch keeps its session; the exception is an input an output
may be written into, which is copied onto the card like a read one. What a run consumed is released
as it returns, and drops out. A run whose discount leaves its arena nothing, or less than what it
would have the runtime copy in, is refused before it takes anything it was fed; one whose arena
needs more than it was left fails with ORT's `BFCArena` error. On an RTX 4090 under a 256 MiB
budget: with nothing held, a session got a 252 MiB arena and a run filling 160 MiB of it succeeded;
with a 100 MiB tensor held on the card, the session was rebuilt at 152 MiB and the same run failed.

**When a session is rebuilt.** ORT fixes `gpu_mem_limit` when a session is built, and building one
costs more the larger the graph — a training step of some 1,500 nodes took 0.4–0.6 s to compile —
so Shorokoo does not build one per run. A session is built with the budget less the discount,
rounded up to the next sixty-fourth of the budget, and kept while that limit is within what the
budget allows. A run that finds the discount grown past the room its session left rebuilds the
session with the lower limit, before it takes anything; a run that finds the discount fallen keeps
the session and its lower limit. So:

- the limit only comes down — at most sixty-four times over a compiled graph's life as the discount
  climbs through the budget, plus once more each time what is left halves in its last sixty-fourth
  — and a loop that holds the same things on the card from run to run never rebuilds;
- a context that releases what it held keeps its compiled graphs' smaller arenas; compile the graph
  again to give it the room back;
- `CompiledGraph.DeviceMemory.LimitBytes` is the current session's arena limit, and
  `ReadArenaStatistics()` and `ReadNodePlacement()` read that session — a rebuilt one starts its
  figures, and its trace, afresh;
- a training rig's step is a compiled graph like any other, so its first steps can rebuild it as the
  rig's state arrives on the card.

An output a session left in its own arena — retained on the device and fed back to the next run of
the same graph — is already inside that session's limit and is not discounted again.

An output a run [wrote into memory it consumed](#a-run-that-writes-an-output-into-what-it-consumed)
is counted once, where that memory is. The consumed tensor was on the context's books until the run
took it, so the run's discount counted its bytes and its arena never made room for the output.
Afterwards the output is on the books in its place: in later runs' discount where the consumed
tensor was outside the session's arena (a copy the run made onto the card, or one `CopyTo` placed
there), and inside the arena, not discounted, where the consumed tensor was an output of that
session's earlier run. A resident training run that begins from an initial checkpoint thereby keeps
the state it overwrites outside the arena for the whole run, in the memory the first step copied the
checkpoint into.

**One at a time.** Under a budget, the context's runs are serialized: a second waits for the first
to return, as do a transfer onto the context and a compile on it, since each would count room the
running arena may be taking. Every run returns its arena's unused blocks as it ends, whatever
`RunSettings.ShrinkArenaAfterRun` says, so between runs a session's arena holds what it keeps — its
weights, and outputs left there — rather than its peak. None of this applies to a context with no
`LimitBytes`.

What the budget does *not* count — spare blocks, the allocator tensors are placed from, a session's
weights between its runs, memory a dead tensor still holds while a run finishes with it — is in
[Known limitations](limitations.md#a-device-memory-budget-counts-tensors-not-arenas).

### What one session's arena did

`DeviceMemory` reads the card. To read **this session's own allocator** — its bytes only, on a CPU
backend as well as a GPU one — ask the compiled graph:

```csharp
using Shorokoo.Core.Backends;
using Shorokoo.Runtime;

var compiled = ctx.Compile(graph);
compiled.Execute(inputs);

if (compiled.ReadArenaStatistics() is { } arena)
    Console.WriteLine($"{arena.InUseBytes} in use, {arena.MaxInUseBytes} at its highest, "
                    + $"{arena.TotalAllocatedBytes} taken from the device");
```

`ArenaStatistics` has nine figures: `InUseBytes`, `MaxInUseBytes`, `MaxAllocSizeBytes`,
`TotalAllocatedBytes`, `LimitBytes` (the session's arena limit — under a device-memory budget, what
the budget left it — and `-1` with no budget), `AllocationCount`, `ArenaExtensionCount`,
`ArenaShrinkageCount` and `ReserveCount`. It is `null` on a backend that reports none, as
`DeviceMemory.Read()` is `null` with no card.

**`TotalAllocatedBytes` is not a bound on what the card holds.** On a run that filled a 24,564 MiB
card it read 32,462 MiB. Likely an arena pressed to the card's edge returns regions and takes others
while this counter does not follow all the way down; read it as the arena's own account of what it
has taken, and `DeviceMemory.Read()` for what the card is carrying.

**`MaxInUseBytes` covers the arena's whole life, not the last run.** It is a high-water mark the
arena never lowers and cannot reset — arena shrinkage does not move it — so it tells you the largest
this session has ever been. For a per-run figure, let the context collect them.

**A session's weights come out of this arena too**, on the CPU and on a card, so a session holds
them before it has run anything: a graph whose only parameter is four mebibytes reads
`MaxInUseBytes` of exactly 4,194,304 at construction. So the first run's peak is weights plus what
that run added — `RunMemoryRecord.PriorPeakBytes` below separates them. And `ReserveCount` and
`ArenaExtensionCount` do not compare across devices: the host arena takes a weight as a reserve, the
CUDA arena as a block of its own.

### What crossed the bus

A device session that gives part of a graph to the host stages the crossings through a **pinned host
arena**, a second allocator with its own figures:

```csharp
if (compiled.ReadPinnedArenaStatistics() is { } pinned)
    Console.WriteLine($"{pinned.MaxInUseBytes} of pinned host memory at its highest");
```

Same nine figures, and `null` on a backend with no such arena — every CPU one, which stages nothing.
They are bytes of pinned *host* memory, so they are not added into `ReadArenaStatistics()`: a graph
that stays on the card leaves this at zero, and one the runtime split pays here for every value that
crosses.

### Per-run statistics on the context

A `ComputeContext` makes the runs, so it can report their cost. Collection is **off by default** and
costs nothing until enabled:

```csharp
using var ctx = new ComputeContext
{
    Diagnostics = new DiagnosticSettings { CollectRunStatistics = true },
};

var rig = TrainingRig.FromScratch(model, loss, optimizer, sample, hypers, runtimeContext: ctx);
var checkpoint = rig.CreateInitialCheckpoint();
for (int step = 0; step < steps; step++)
    checkpoint = rig.TrainStep(checkpoint, inputs.Shared(), targets.Shared());

var stats = ctx.RunStats;
Console.WriteLine($"{stats.RunCount} runs, peak {stats.PeakBytes / (1024 * 1024)} MiB, "
                + $"{stats.ArenaExtensionCount} arena extensions");

foreach (var run in stats.RecentRuns.TakeLast(5))
    Console.WriteLine($"run {run.RunNumber}: {run.PeakBytes} ({run.PeakKind}), "
                    + $"arena stood at {run.PriorPeakBytes} before it");
```

`RunStats` is a snapshot of every run the context has made, across **all** its sessions — the rig's
compiled steps, any graph you compiled on it, and the one-shot entry points.

- **The aggregates are exact over the whole history; per-run detail is bounded.** `PeakBytes`,
  `RunCount`, `AllocationCount`, `ArenaExtensionCount`, `ArenaShrinkageCount`,
  `LargestAllocationBytes` and `ArenaBytes` are folded in as each run finishes, so a hundred
  thousand steps are all in them. `RecentRuns` keeps the last `DiagnosticSettings.RecentRunCapacity`
  (1000 by default; zero keeps none), since one record per run kept for the context's life would
  leak in a training loop.
- **A per-run peak says whether it was measured or bounded.** The arena's high-water mark is read
  before and after each run: a run that raised it set the record, and its `PeakKind` is
  `MemoryFigureKind.Measured`. A run that stayed under a mark an earlier run set is
  `MemoryFigureKind.UpperBound` — it used no more than that, and the arena does not record how much
  less.
- **A per-run peak also records what the run found.** `PriorPeakBytes` is the arena's high-water
  mark as the run found it, so a `Measured` peak is where the arena stood at this run's high point,
  not the run's own cost: on a card, the **first** run of a four-mebibyte model read 4,202,496
  against a prior mark of 4,194,304, of which 8,192 was the run. The difference is the run's own
  only on that first run, where the prior mark is the weights alone. Later it is the previous
  *highest* run's mark, so the difference is how far this run exceeded the record — zero for every
  `UpperBound` run, which in a settled loop is most of them.

`PeakBytes` is the largest mark any one of the context's arenas reached. A context runs its graphs
one session at a time, so that is the peak; where two of its sessions do run together, read it as
the largest of them, not their total.

**`ArenaBytes` sits above `PeakBytes` until something shrinks.** It tracks what the arenas have
*taken* from the device, which usually exceeds what is in use by their spare blocks — though it is
not a bound on what the device holds, and on a card pressed to its edge it has read above the card's
capacity. `RunSettings.ShrinkArenaAfterRun`, on for every run under a device-memory budget, returns
blocks at the end of a run, before these are read, while the peak comes from a mark the runtime
never lowers. Three shrinking runs of a matmul on a card, operands fed `.Shared()`, left
`ArenaBytes` at 0 below a `PeakBytes` of 1,048,576; without shrinkage the two were equal. How far
below depends on how many sessions the context compiled, so rely on the ordering, not a figure.
`ArenaExtensionCount` undercounts for the same reason: it counts the blocks the arena *holds*, so a
shrinking run can end below where it started and the aggregate loses the difference.

### Did part of my GPU graph run on the host?

A CUDA session that meets an operator the provider cannot run leaves that part to the host, and the
results cross the bus back. Two signals, one free and one not:

```csharp
switch (compiled.OutputPlacement)
{
    case SessionOutputPlacement.Device: break;               // all of it stayed on the card
    case SessionOutputPlacement.Mixed:                       // some of it did not
    case SessionOutputPlacement.Host: break;                 // none of it did
    case SessionOutputPlacement.Unknown: break;              // this backend does not say
}
```

`OutputPlacement` costs nothing and is available on every run. A CPU session reports `Host`.

On a card the three answers separate the cases. A graph the CUDA provider runs whole reports
`Device`. A graph with one operator it has no kernel for — `Det`, say — reports `Mixed` when an
output is left on each side, and `Host` when every output came back. An output the card computed and
the host then consumed is reported as host memory, where the runtime put it: it lands in the pinned
host arena, and `ReadPinnedArenaStatistics()` shows its cost.

For **which** nodes fell back and what they moved, have the context trace them. This builds the
session with ONNX Runtime's profiler on, which costs every run that session makes, so it is off by
default and belongs in diagnosis, not a training loop.

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
recording**: it covers every run up to that call, later runs are not in it, and a second read returns
the same trace. Run what you are asking about, then read once.

`Nodes` is in execution order, which on a split graph is not `NodeExecution.NodeIndex` order. ONNX
Runtime inserts a `MemcpyToHost` / `MemcpyFromHost` node at each provider boundary with a fresh index
above every original node, while placing it where it belongs in the plan — so each inserted copy
appears immediately before the node it feeds but sorts to the end. On a graph with one crossing,
that copy carried the highest index and ran third of four. Several crossings give several such
copies, sorting among themselves after everything else. Read `Nodes` for what happened, and
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

`QuickExecutionEngine` is a CPU-only interpreter for debugging, shape inference, and small
prototypes. It materializes values only for tensors ≤ `MaxDataElements` (default 256). Do not use
it as a production inference path.

It consumes nothing: a tensor fed as it is, or a `SharedInput` of any mode, is read. The engine is
a reference evaluator walking the graph in managed code, not a run on a compute context, so every
input is still yours, alive and unchanged, when it returns.

To debug the graph *structure* rather than values — e.g. when `ToConcreteArchitecture` does not
produce the graph you expect — snapshot the lowering stages with `DebugRequests`; to follow a
lowering that runs for minutes, pass it a `progress:` sink. Both are in
[debugging.md](debugging.md).

## Anti-patterns

- Do not call `OnnxEngine.Eval` in a tight loop for the same graph; compile once with
  `ComputeContext`.
- Do not reassign `DefaultBackend.Instance` mid-process to move from CPU to GPU; it does
  not unload the native ONNX Runtime already bound. Give each device its own
  `ComputeContext` instead — see [One model, two devices](#one-model-two-devices).
- Do not rely on `QuickExecutionEngine` results for large tensors — values above the
  element cap are not materialized.
