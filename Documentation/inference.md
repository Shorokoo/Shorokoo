# Running models (inference)

Run a model once or compile it to run many times, with `[Hyper]` parameters fixed or hardcoded,
and stop a run. Three pages go further:

- [backends-and-devices.md](backends-and-devices.md) — pick the backend, load one at runtime, run
  one model on two devices.
- [tensors-in-a-run.md](tensors-in-a-run.md) — what a run consumes or only reads, where its outputs
  are, and when a tensor's memory is freed.
- [gpu-backends.md](gpu-backends.md) — the NVIDIA libraries, precision, device-memory budgets and
  what a run did on the card.

Related: [core-types.md](core-types.md) · [defining-models.md](defining-models.md) ·
[onnx-and-weights.md](onnx-and-weights.md)

## Facts

- `OnnxEngine.Eval(...)` builds an ONNX model from the graph, runs it once via OnnxRuntime,
  and returns `TensorData`:
  - `TensorData Eval(Variable output)`
  - `TensorData[] Eval(Variable[] outputs)`
  - `TensorData[] Eval(Variable output1, Variable output2, params Variable[] outputs)`

  It rebuilds the ORT session on every call; for repeated inference compile once with
  `ComputeContext`. It refuses a `[Module]` output — see
  [Running a `[Module]`](#running-a-module).
- One referenced backend package is found with no setup; two for one OS are refused —
  [Backend selection](backends-and-devices.md#backend-selection). Several can run at once, one
  per `ComputeContext` — [One model, two devices](backends-and-devices.md#one-model-two-devices);
  [Which device am I on?](backends-and-devices.md#which-device-am-i-on).
- **A tensor fed to a run as it is is consumed by that run.** Pass `.Shared()` to have the run
  only read it, or `.TryConsume()` to consume it only when nothing else is reading it —
  [Feeding a run: consumed, shared or tried](tensors-in-a-run.md#feeding-a-run-consumed-shared-or-tried),
  [A tensor's lifetime](tensors-in-a-run.md#a-tensors-lifetime-locks-and-deletion).
- A compute context keeps books on tensors and owns none; disposing it leaves every tensor
  alive — [Moving data between contexts](tensors-in-a-run.md#moving-data-between-contexts).
- **A run's outputs are in the memory of the backend that ran it** — on a GPU backend, the
  card's — and nothing moves them afterwards. Reading one's values copies them to the host and
  leaves it there; fed to a run on another context, that run moves it there —
  [Where a run's inputs and outputs are](tensors-in-a-run.md#where-a-runs-inputs-and-outputs-are).
- **A run may write its values into the memory of the inputs it consumes**, where the graph proves
  it safe and it saves memory; its outputs then stand on that memory, each holding its own range of
  it — [A run that writes into what it consumed](tensors-in-a-run.md#a-run-that-writes-into-what-it-consumed).
- On a GPU backend a context's `DeviceMemory` settings are a budget on what it holds on the card,
  which the allocator its sessions allocate through holds each run to —
  [Device memory](gpu-backends.md#device-memory-gpu-backends). Diagnostics start at
  [What one session's allocator did](gpu-backends.md#what-one-sessions-allocator-did).
- **`float32` is computed in full `float32` precision on every backend and device.** A context
  that sets `Precision = new PrecisionSettings { AllowTensorFloat32 = true }` lets a CUDA card
  compute its `float32` products, convolutions and recurrent layers in TensorFloat-32 instead:
  faster, and hundreds of times less precise — [Precision](gpu-backends.md#precision-gpu-backends).

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

// Read the numbers out (see core-types.md). On a GPU backend the result is on the card, and
// reading it copies its values to the host.
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
  un-lowered module-invoke node; `Eval` throws an `InvalidOperationException` naming the
  fix — see [Running a `[Module]`](#running-a-module). (`ResNet50` is from
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
`ComputeContext.Eval`, `tensor.Eval()`, `inputs.Eval(outputs).With(...)` — refuses such an
output with an `InvalidOperationException`, as do `ComputeContext.Execute`/`Run`/`Compile`
and `QuickExecutionEngine` for a module graph. For example:

> `OnnxEngine.Eval requires a concretized graph (a 'concrete-architecture' or
> 'concrete-model'), but this graph is a 'module'. It still carries module machinery
> that lowering removes (ShrkCreateModule, ShrkModelInvoke, ShrkModuleSetHyperparams).
> It comes from module 'ResNet50': lower that module's ComputationGraph the whole way
> — ToConcreteArchitecture(inputHints) then ToConcreteModel() — and execute that,
> passing a value for each of its inputs in order ([Hyper] parameters come first). …`

Values go in the graph's input order, `[Hyper]` parameters first. An initializer's output
never carries module machinery: an initializer may not create or reference a model, and
building one that does is refused with `FW055` (see *Writing your own* in
[nn-library.md](initializers.md#initializers-shorokoomodulesinitializers)).

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

For a graph loaded from a `.srk`/`.zsrk` file,
`LoadFastGraphFromFile(path, requiredStage: GraphKind.ConcreteModel)` refuses a
module-stage file at load time — see
[onnx-and-weights.md](onnx-and-weights.md#the-srk-container).

### The lowering pipeline

Three steps, in order:

1. **`Specialize(values)`** — *optional.* Bakes a partial set of named inputs
   (typically `[Hyper]` parameters) into constants and drops them from the input list.
   Returns a copy.
2. **`ToConcreteArchitecture(inputHints)`** — inlines every sub-module and function so
   trainable parameters are visible at the top level, resolving shape-dependent
   parameters from the samples. It needs **a sample for every input**, `[Hyper]` and
   sequence inputs included (a generic module's type-placeholder slots take none):

   - **Positional** — `ToConcreteArchitecture([hyper, input])`, an `IData[]` of bare
     values (`TensorData`, `OptionalTensorData`, `TensorDataSequence`,
     `TensorDataStruct`), one per input **in declaration order**. Too few or too many is
     refused with **`FW056`**, naming the missing inputs or stating both counts.
   - **Named** — `ToConcreteArchitecture(new ModelParamList([...]))` of
     `NamedModelParam`s (`TensorDataModelParam`, `OptionalTensorDataModelParam`,
     `TensorDataSequenceModelParam`, `TensorStructModelParam`), bound by name in any
     order. An unnamed input, a sample naming no input, or a duplicate name is refused
     with `FW056`, naming the offenders and listing the graph's inputs.

   ```csharp
   // Dense (below): Inline(Tensor<float32> x, [Hyper] Scalar<int64> outFeatures), graph inputs outFeatures, x
   var arch  = graph.ToConcreteArchitecture([hyper, input]);                 // by position
   var same  = graph.ToConcreteArchitecture(new ModelParamList([
       new TensorDataModelParam("x", ModelParamType.InputParam, input),
       new TensorDataModelParam("outFeatures", ModelParamType.InputParam, hyper)]));  // by name
   ```

   A sample of the wrong rank (a `Scalar` given a vector — usually two positional samples
   swapped) is refused with `FW056`, naming the input and the sample's shape. A struct
   input takes one `TensorDataStruct` sample; once lowered it is one input per field,
   named `<struct>.<field>` (nested: `<struct>.<field>.<subfield>`).

   Each sample's shape is recorded as the input's **representative shape**. It survives
   `ToConcreteModel`, `Specialize` and `.srk`/`.skpt` round trips, a training rig builds
   its shape inference from it, and ONNX export reads an input's rank from it where the
   signature states none (see
   [onnx-and-weights.md](onnx-and-weights.md#graph-inputoutput-names-and-shapes)). A
   concrete graph with an input lacking one (a hand-built one, say) is refused when
   frozen or loaded with **`FW057`**; lower it again from its module.

   Each **output** likewise records its shape at the samples' values (a struct output
   per field as `<output>.<field>`; an absent optional as absent; a sequence as its
   elements' shared shape). Where the shape cannot be computed, the output records its
   rank with unsettled dimensions as `1`. An output whose shape depends on parameter
   **values** is *unresolved* until `ToConcreteModel` (or `Specialize`) records it again;
   one the weights cannot settle either is refused by `ToConcreteModel` with `FW057`,
   carrying the failure as the inner exception. ONNX export refuses a concrete graph with
   an output that records no shape, with `FW057`.
3. **`ToConcreteModel(...)`** — binds parameter values (loaded weights, or the
   initializer defaults when called with no argument).

Every `ComputationGraph` has a **`Kind`** — `GraphKind.Module`,
`GraphKind.ConcreteArchitecture`, or `GraphKind.ConcreteModel` — preserved through copies and
`.srk` save/load. Each step checks it: `ToConcreteArchitecture` requires `Module`,
`ToConcreteModel` requires `ConcreteArchitecture`, and export, weight queries and execution
refuse the wrong stage, naming the actual and required kind. Graphs are **readonly**: changes
(e.g. `WithRngConfig`) return a new graph.

To re-stamp a graph with the wrong kind (a misjudged foreign import, say), use
**`WithKind(kind)`**. It is validated: a module must have no initialized parameters, a
concrete architecture a statically known parameter space, a concrete model every parameter
initialized; a violation is refused with an error naming it. Execution also checks the ops
themselves, whatever the stamp.

## Running a `[Module]` with `[Hyper]` parameters

A module's `ComputationGraph` lists its `[Hyper]` parameters as inputs **before** the
tensor inputs, whatever the `Inline` source order, and they stay inputs in the concretized
graph. Positional samples and `Execute` both take hyper values first:

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

A hyper that determines the trainable parameters — their shapes (like `outFeatures`) or
which exist (a `[Hyper]` gating an `IfElse` branch that holds parameters) — is
**parameter-space-determining**: the value given to `ToConcreteArchitecture` fixes that part
of the architecture, so pass the same value at `Execute`. Value-only hypers (scale factors,
ε's) are read live and may vary per call. See
[defining-models.md](defining-models.md#hyperparameter-baking) and
[What concretization fixes](#what-concretization-fixes).

### What concretization fixes

`ToConcreteArchitecture` makes the **parameter space static** — every trainable parameter
and id-addressed component is enumerated, so weights bind by name, optimizers allocate
state, and checkpoints round-trip. Fixed then:

| Fixed at concretization | Derived from |
|---|---|
| Trainable-parameter **shapes** and count | hypers feeding a parameter's shape, and the shapes of the sample inputs |
| **Which** trainable parameters exist | hypers gating an `IfElse` whose branches hold parameters |
| The per-iteration **parameters** realized over a `LoopAPI.Iterate` body (and the whole iteration space, when the count folds to a constant and the loop unrolls) | hypers/inputs that drive the count |

Control flow is rewritten only as the parameter space requires:

- An `IfElse` whose **unselected** branch holds parameters is folded away.
- An `IfElse` whose selected branch holds the parameters, or that holds none, stays live and
  selects at run time. For `bit.IfElse(withParams, without)`, baking the bit **off** folds it;
  baking it **on** leaves it live.
- Only an `IfElse` that *solely* owns the pruned parameters folds: every use of a pruned
  parameter's value must lead into that one branch. If the value also reaches another
  `IfElse`, an `IfElse` condition or a model output, or the `IfElse` is a tuple, it stays
  live. Ownership follows the parameter's value, not its inputs: a parameter whose
  initializer reads a parameter from outside the branch is still owned by the branch.
- Inside a `LoopAPI.Iterate` body, a gate computed from these values and
  `ctx.IterationIndex` is resolved **per iteration**: each iteration keeps only the
  parameters its own branch holds, so the layers of one stack can differ
  ([Per-layer variants](defining-models.md#per-layer-variants)).
- The concretization value decides, not the `[Hyper]` marker: a plain runtime input gating a
  parameter is resolved the same way. Mark such gates `[Hyper]` so the baking is visible.

```csharp
var big = Zeros.Init([outFeatures]).Vec();       // a trainable parameter
var a = flag.IfElse(x * 10f, x * 100f);          // no params -> always stays live
var b = flag.IfElse(x + big, x);                 // holds big -> folded iff flag is baked false
```

With `flag = false`, `big` does not exist, `b` is `x` whatever you pass later, and `a` still
switches. With `flag = true`, `big` exists and **both** switch at run time.

These values stay **live inputs** — concretization, unlike `Specialize`, removes nothing — so
supply **the same values** at every `Execute` (feed a reused `TensorData` `.Shared()`; fed as
it is, the first run consumes it and the next `Execute` throws naming that run). A value that
would have produced a different parameter space is **invalid use**. For a parameter gate that
folded, the opposite value is harmless; for one that did not, it silently takes the other
branch — skipping existing parameters, or reading a **zero stand-in** for pruned ones. To make
this impossible, bake the hyper with [`Specialize`](#hardcoding-hypers-with-specialize): the
input is dropped, so passing it becomes an input-count error.

### Hardcoding hypers with `Specialize`

`Specialize` constant-folds a partial set of named input values into the graph and removes
them from the input list, so you need not supply them on every `Execute`. The order is
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

`Specialize` matches values to inputs **by name** (`InputNames`); unmatched names are
ignored. It returns a copy. It works for any input, not just hypers.

## Workflow: compile once, run many (repeated inference)

`ComputeContext` builds the ORT session once and reuses it.

```csharp
var ctx      = new ComputeContext();
var compiled = ctx.Compile(graph);                 // graph: a concretized ComputationGraph
var r1 = compiled.Execute(inputData1);             // params IData[] — the data goes here
var r2 = compiled.Execute(inputData2);             // reuses the session
```

`inputData1` is consumed by the first call; pass it `.Shared()` to reuse it — see
[Feeding a run](tensors-in-a-run.md#feeding-a-run-consumed-shared-or-tried).

`ComputeContext` also offers `Eval(...)` (the `OnnxEngine.Eval` overloads, plus
`Eval<T>(Tensor<T>)` returning `TensorData<T>`), `Execute(ComputationGraph graph, params IData[] inputs)`,
`Run(ComputationGraph graph, params NamedModelParam[] inputs)`, and `ExecuteWithState(...)`
(for stateful models). `TensorData` implements `IData`. `Execute`, `Run` and
`CompiledGraph.Execute` return `NamedModelParam[]`, each in the memory of the context's backend;
read each with `ToTensorData()` then `CopyMemory<V>()`, or `ValueAt<V>(i)`, V being the CLR
storage type (`float` for `float32`), wherever it is.
`ExecuteWithState` returns `(NamedModelParam[] regularOutputs, ComputationGraph updatedGraph)`
— feed the updated graph to the next call. `Eval` returns `TensorData` (or `TensorData[]`).

### Loading a saved model onto the device

`ctx.Compile(Persistence.Load(path))` holds the weights in the graph, in host memory, and the
session copies them onto the card as it is built — several host copies of the weights at that
moment. `LoadCompiled` goes from the file to a compiled model instead, reading each weight straight
into the context's memory through one bounded host buffer; the session reads the weights where they
are, and they are never whole in host memory:

```csharp
using var cuda     = new ComputeContext(cudaBackend);
using var compiled = cuda.LoadCompiled("model.skpt");                       // a .skpt checkpoint
using var imported = cuda.LoadCompiled(architecture, "model.safetensors");  // or architecture + weights
using var foreign  = cuda.ImportCompiledOnnx("model.onnx");                 // or an .onnx model
var y = compiled.Execute(x);
```

- The weights belong to the compiled graph and are freed when it is disposed. There is no graph
  with the weights to edit or save; load one with `Persistence.Load` for that.
- A weight another context's run is reading through `.Shared()` when its graph is disposed is
  deleted all the same, and its memory is freed when that run is done with it. While the graph
  lives, `Delete` refuses its weights, `TryDelete` declines, and `DeleteAsync` deletes one and
  counts it on the context's budget until the graph is disposed.
- Weights of at most about a thousand elements stay in the graph on the host, where the compiler
  reads such values.
- The runtime does not fold or fuse over the weights it is handed, so results can round differently
  in the last bits from the same model compiled from its graph.
- On a backend that cannot use weights where they are (PyTorch, JAX), `LoadCompiled` is
  `Compile(Persistence.Load(...))`.
- A graph holds a weight of any size. One of more bytes than one managed array holds
  (`Array.MaxLength`, just under 2 GiB) is kept in host memory of the backend
  `ComputeContext.Default` runs on: `Persistence.Load`, `Persistence.ImportOnnx`,
  `Persistence.ImportSafeTensors` and `CompressedFormatUtils.LoadFastGraphFromFile` bind it where
  they read it, without a copy. Compiling or running
  such a model on an ONNX Runtime context hands its session the weight where it is on the CPU, and
  a copy in the context's memory on CUDA, never through the model's bytes. PyTorch and JAX take
  every weight inside the model's bytes, which protobuf caps at 2 GiB, so compiling or running such
  a model there, `LoadCompiled` included, is refused with `NotSupportedException`. `LoadCompiled`
  and `ImportCompiledOnnx` on an ONNX Runtime context, CPU or CUDA, read it into the context's
  memory.
- `ImportCompiledOnnx` imports the `.onnx` as `Persistence.ImportOnnx` does and refuses it
  alike, with the same `namingScheme` and `inputShapes` overloads. Its weights are read from where
  they lie in the file, inline in the protobuf or in an external-data side file alike, so the
  protobuf is never held whole. A weight ONNX stores as varints (`int32_data`, `int64_data`,
  `uint64_data`) has no bytes to read in place; it is decoded on the host and stays in the graph.

### Stopping a run

Put a `CancellationToken` on a run's `RunSettings`; a cancelled call throws
`OperationCanceledException` instead of returning outputs:

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

- A token already cancelled is refused before anything is fed: nothing is consumed, and the
  call can be repeated with the same inputs. A run stopped after it started has consumed its
  as-is feeds, like a failed run — pass `.Shared()` to retry.
- ONNX Runtime checks for cancellation **between operators**, so the wait is up to the
  model's **longest single operator**. A graph that is one operator cannot be stopped, nor can
  any graph's last operator.
- Stopping is best effort: a run that finishes first **succeeds** with valid outputs. The
  session is unaffected; its next run proceeds normally.

Only `CompiledGraph`'s run entry points take a `RunSettings` per call.
`ComputeContext.Execute` / `Run` / `Eval`, and a training rig's `TrainStep`, `Train` and `Fit`,
use the context's (`init`-only) settings:

```csharp
using var ctx = new ComputeContext(backend)
{
    RunSettings = new RunSettings { CancellationToken = cts.Token },
};

var outputs = ctx.Execute(graph, inputs);   // stops when cts does
```

Giving a training rig's `runtimeContext` one abandons the training step running, which consumes
the run's state. To stop a `Fit` or `Train` between steps and keep what it trained, pass the
`cancellationToken` parameter of `Fit` / `Train` instead — see
[Stopping and watching a run](training.md#stopping-and-watching-a-run).

## Debugging engine (no OnnxRuntime)

```csharp
using Shorokoo.Core.Interpreter;   // QuickExecutionEngine
```

`QuickExecutionEngine` is a CPU-only interpreter for debugging, shape inference, and small
prototypes. It materializes values only for tensors ≤ `MaxDataElements` (default 256). Do not
use it as a production inference path. It consumes nothing: every input, whatever its mode, is
only read.

To debug graph *structure* — e.g. when `ToConcreteArchitecture` does not produce the graph you
expect — snapshot the lowering stages with `DebugRequests`, or pass a `progress:` sink to follow
a long lowering. See [debugging.md](debugging.md).

## Anti-patterns

- Do not call `OnnxEngine.Eval` in a tight loop for the same graph; compile once with
  `ComputeContext`.
- Do not reassign `DefaultBackend.Instance` mid-process to move from CPU to GPU; it does
  not unload the native ONNX Runtime already bound. Give each device its own
  `ComputeContext` instead — see [One model, two devices](backends-and-devices.md#one-model-two-devices).
- Do not rely on `QuickExecutionEngine` results for large tensors — values above the
  element cap are not materialized.
