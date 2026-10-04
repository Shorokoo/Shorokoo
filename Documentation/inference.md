# Running models (inference)

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
  [Backend selection](#backend-selection). Several can run at once, one per `ComputeContext` —
  [One model, two devices](#one-model-two-devices); [Which device am I on?](#which-device-am-i-on).
- **A tensor fed to a run as it is is consumed by that run.** Pass `.Shared()` to have the run
  only read it, or `.TryConsume()` to consume it only when nothing else is reading it —
  [Feeding a run: consumed, shared or tried](#feeding-a-run-consumed-shared-or-tried),
  [A tensor's lifetime](#a-tensors-lifetime-locks-and-deletion).
- A compute context keeps books on tensors and owns none; disposing it leaves every tensor
  alive — [Moving data between contexts](#moving-data-between-contexts).
- **A run's outputs are in the memory of the backend that ran it** — on a GPU backend, the
  card's — and nothing moves them afterwards. Reading one's values copies them to the host and
  leaves it there; fed to a run on another context, that run moves it there —
  [Where a run's inputs and outputs are](#where-a-runs-inputs-and-outputs-are).
- **A run may write its values into the memory of the inputs it consumes**, where the graph proves
  it safe and it saves memory; its outputs then stand on that memory, each holding its own range of
  it — [A run that writes into what it consumed](#a-run-that-writes-into-what-it-consumed).
- On a GPU backend a context's `DeviceMemory` settings are a budget on what it holds on the card,
  which the allocator its sessions allocate through holds each run to —
  [Device memory](#device-memory-gpu-backends). Diagnostics start at
  [What one session's allocator did](#what-one-sessions-allocator-did).
- **`float32` is computed in full `float32` precision on every backend and device.** A context
  that sets `Precision = new PrecisionSettings { AllowTensorFloat32 = true }` lets a CUDA card
  compute its `float32` products, convolutions and recurrent layers in TensorFloat-32 instead:
  faster, and hundreds of times less precise — [Precision](#precision-gpu-backends).

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
[nn-library.md](nn-library.md#initializers-shorokoomodulesinitializers)).

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
- Only an `IfElse` that *solely* owns the pruned parameters folds; one sharing them with
  another `IfElse`, or a tuple `IfElse`, stays live.
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
[Feeding a run](#feeding-a-run-consumed-shared-or-tried).

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

## Backend selection

- Add a backend package: `Shorokoo.LinuxCPU`, `Shorokoo.LinuxGPU`, `Shorokoo.WinCPU`, or
  `Shorokoo.WinGPU`. Each brings the native ONNX Runtime (CPU or CUDA) for its platform. A
  program that names its backends may reference several, or load them at runtime — see
  [Loading a backend at runtime](#loading-a-backend-at-runtime).
- With one backend package referenced, auto-discovery finds it on the first run. Set it
  explicitly when a deployment holds more than one, when you want a startup failure instead
  of a first-run one, or when the backend DLL is not next to `Shorokoo.dll`:

  ```csharp
  using Shorokoo.Core.Backends;
  using Shorokoo.LinuxCPU;                                // the package you referenced

  DefaultBackend.Instance = new LinuxCpuBackend();
  ```

- `DefaultBackend.Instance` is the backend for a `ComputeContext` that names none, and for a
  tensor fed without a context naming another. The first resolved is cached. Reassigning it
  does not unload a native ONNX Runtime already bound, nor reach a `ComputeContext.Default`
  that has already resolved — assign it at startup.
- A `ComputeContext` constructed with a backend runs there — see
  [One model, two devices](#one-model-two-devices).
- PyTorch (`Shorokoo.PyTorch.Cpu`, `Shorokoo.PyTorch.Cuda`) and JAX (`Shorokoo.Jax.Cpu`,
  `Shorokoo.Jax.Cuda`) backends run a model in an embedded Python. They are never
  discovered — name one per context, `new ComputeContext(new TorchCpuBackend())` — so they
  never make discovery ambiguous. See [pytorch-backend.md](pytorch-backend.md) and
  [jax-backend.md](jax-backend.md).

### What a backend provides

`IShorokooBackend` is the interface every backend implements; its comments are the full
contract. In short, a backend provides:

- **Its run memory** — `RunMemoryOf(elementType)` for a tensor and `SequenceRunMemory` for a
  sequence: where its sessions read their inputs and leave their outputs. The defaults are the
  backend's own memory (`MemorySpace`) in its own runtime (`RuntimeIdentity`), and host memory
  for strings and sequences.
- **Sessions** (`CreateSession`) that read every input in the run memory, refuse any value
  outside it with an exception, and leave every output there (`IShorokooSession`). A session
  never moves a value.
- **The moves**, as operations of their own, which Shorokoo calls to place an input before a run
  and to bring a tensor home: into the backend's memory, `CreateTensorInBackendMemory`,
  `CreateUninitializedTensorInBackendMemory` and `TryCopyHostToTensorRange`; out of it,
  `CopyTensorToHost` and `TryCopyTensorRangeToHost`. Their defaults serve a backend whose memory
  the host reads; a backend that computes in memory of its own overrides each of them.
- **Values in its runtime's host memory** — `CreateTensor`, `CreateTensorFromRawBytes`,
  `CreateStringTensor`, `CreateSequence` — and `Release`, the one path its memory goes back by.

### The backend types

One backend per package, in a namespace equal to the package id. **The type name spells
`Cpu`/`Gpu`; the package spells `CPU`/`GPU`** — `WinGpuBackend`, *not* `WinGPUBackend`:

| package (= namespace) | backend type | fully qualified |
|---|---|---|
| `Shorokoo.LinuxCPU` | `LinuxCpuBackend` | `Shorokoo.LinuxCPU.LinuxCpuBackend` |
| `Shorokoo.LinuxGPU` | `LinuxGpuBackend` | `Shorokoo.LinuxGPU.LinuxGpuBackend` |
| `Shorokoo.WinCPU` | `WinCpuBackend` | `Shorokoo.WinCPU.WinCpuBackend` |
| `Shorokoo.WinGPU` | `WinGpuBackend` | `Shorokoo.WinGPU.WinGpuBackend` |

All four implement `IShorokooBackend` with a parameterless constructor; the GPU ones use the
CUDA provider on device 0, the CPU ones ORT's default provider.

| package | backend type | fully qualified |
|---|---|---|
| `Shorokoo.PyTorch.Cpu` | `TorchCpuBackend` | `Shorokoo.PyTorch.Cpu.TorchCpuBackend` |
| `Shorokoo.PyTorch.Cuda` | `TorchCudaBackend` | `Shorokoo.PyTorch.Cuda.TorchCudaBackend` |
| `Shorokoo.Jax.Cpu` | `JaxCpuBackend` | `Shorokoo.Jax.Cpu.JaxCpuBackend` |
| `Shorokoo.Jax.Cuda` | `JaxCudaBackend` | `Shorokoo.Jax.Cuda.JaxCudaBackend` |

The sessions of the four ONNX Runtime backends run their operators on ONNX Runtime's thread pools
of the process, made with its environment when the first of these backends is built, rather than
each on an intra-op pool of its own (`SessionsShareThreadPools`, true unless set otherwise:
`new WinCpuBackend { SessionsShareThreadPools = false }`). A pool's threads spin for a while after a
run, waiting for more work, so sessions with pools of their own run one after another contend: two
compiled graphs run in turn, or a compiled graph whose runs consuming their inputs place values
through a session of their own
([A run that writes into what it consumed](#a-run-that-writes-into-what-it-consumed)). A session
built with an intra-op thread count of its own (`CreateSession`'s `intraOpThreads`) keeps a pool of
its own of that size; where something else made ONNX Runtime's environment first, every session
keeps its own.

### Auto-discovery

If `DefaultBackend.Instance` is never assigned, its first read resolves and caches a backend:

1. If one of the four backend assemblies targeting the running OS is **already loaded** and
   exposes a backend, it is used, and the folder is never probed — so a deployment carrying
   several backends can run without being refused. (Naming a backend type in any method your
   program runs can load it.) Backends loaded by `IsolatedBackend.Load` are never candidates.
2. Otherwise the folder next to `Shorokoo.dll` is probed for the known
   `Shorokoo.{Platform}.dll` files targeting the running OS. Nothing else is searched.
   Referencing a package suffices: it copies its DLL to your output folder.

A single candidate is taken as-is — even a GPU backend with no CUDA runtime present. **Two or
more are refused** with an `InvalidOperationException` (step 1 says `already loaded in this
process` instead of `deployed in '<folder>'`):

> `Several Shorokoo backends are deployed in '<folder>': Shorokoo.WinCPU (CPU),
> Shorokoo.WinGPU (CUDA). Discovery picks the backend for a program that named none, and
> this deployment gives it no way to choose. Say which you mean: assign
> DefaultBackend.Instance before the first run to make one of them the default.
> To run several at once, give each ComputeContext its own backend -- new ComputeContext(new
> LinuxGpuBackend()) -- and where they need separate native ONNX Runtimes, load
> them with IsolatedBackend.Load.`

Discovery never prefers the GPU: both packages ship their native ONNX Runtime at the same
path, so only one native is actually deployed. Backends for other OSes are ignored, so a
deployment with all four is refused on either OS. A shared library referencing a backend is
the usual accidental cause — keep backends in executables (see
[Or keep it to two processes](#or-keep-it-to-two-processes)); to deploy two on purpose, see
[Deploying two backends](#deploying-two-backends).

If no backend is found, the first run throws `InvalidOperationException`:

> `No Shorokoo backend is set and none was found in '<folder>'. Set one at
> startup -- e.g. DefaultBackend.Instance = new LinuxCpuBackend(); (or the
> backend from whichever Shorokoo.{WinCPU,WinGPU,LinuxCPU,LinuxGPU} package you
> reference) -- or add such a package as a dependency.`

### Which device am I on?

```csharp
using Shorokoo.Core.Backends;

Console.WriteLine(DefaultBackend.Describe());         // Shorokoo.WinGPU (CUDA device 0)
Console.WriteLine(ComputeContext.Default.Backend);    // the same, at the point work is submitted
```

`BackendDescription` carries `Name` (the supplying assembly), `Device` (`ComputeDevice.Cpu`,
`Cuda`, or `Other` for a custom execution provider) and `CudaDeviceId` (null off CUDA). Log it
with training runs.

- `DefaultBackend.Current` is the live backend **or null**; unlike `Instance` and
  `Describe()`, reading it does not trigger resolution.
- `DefaultBackend.RequireDevice(ComputeDevice.Cpu)` throws, naming the live backend, unless it
  is on that device — use it at startup in a program that must not share the GPU:

  ```csharp
  DefaultBackend.RequireDevice(ComputeDevice.Cpu);   // before anything runs
  ```

### Loading a backend at runtime

`BackendPackage.TryLoad` loads a backend from a path, returning a reason instead of throwing
when it does not fit the machine — so one executable can carry several and pick at startup:

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

`BackendPackage.Probe` answers without loading the backend, from the file's metadata.
`BackendProbe.Reason` gives the cause (`WrongOperatingSystem`, `MissingNative`,
`MissingCudaRuntime`, `MissingCudaDriver`, …) and `Detail` the offending file or library.
Wrong OS, architecture or missing natives are decided from metadata and paths alone. A backend
requiring the CUDA runtime is checked by binding it, which creates this process's CUDA context
on the card; one needing only the driver (the [PyTorch](pytorch-backend.md) and
[JAX](jax-backend.md) CUDA backends) is checked through the driver API, creating no context,
and refused as `MissingCudaDriver` with no driver, one too old, or no device.

Natives are looked for flat beside the backend assembly and under `runtimes/<rid>/native/`
next to it — a NuGet install uses the `runtimes/` layout on every platform. Each backend
loaded this way gets its own load context, so several run side by side without sharing a
native runtime.

### Feeding a run: consumed, shared or tried

A tensor fed **as it is** is taken by the run when it starts — dead from then — and its memory
is released before the call returns, or reused for the run's values
([below](#a-run-that-writes-into-what-it-consumed)):

```csharp
var result = compiled.Execute(batch)[0].ToTensorData();
// batch is consumed: reading it throws, naming the run that took it
```

Pass `.Shared()` to keep it: the run only reads it, under a reader lock:

```csharp
var weights = TensorData([4L, 4L], w);
var first  = compiled.Execute(x1, weights.Shared());
var second = compiled.Execute(x2, weights.Shared());   // weights is still there
```

| Fed as | The run | Afterwards |
|---|---|---|
| `t` | takes it when it starts; refuses it while another run is reading it | dead |
| `t.Shared()` | reads it, holding a reader lock | alive and unchanged |
| `t.TryConsume()` | takes it if nothing else is reading it, reads it otherwise | dead, or alive if it was read |

A tensor fed as it is while another run reads it makes the call throw
`InvalidOperationException` naming that run.

- **Consumption is irrevocable.** A run that fails or is stopped after starting has consumed
  its as-is feeds. A run refused before starting (cancelled token, dead feed, feed being
  read) takes nothing — except in a race, where a feed that dies or starts being read
  mid-setup refuses the run part-way and what it took stays consumed.
- **One tensor fed twice in one call** is taken once: read if any occurrence is `.Shared()`,
  else consumed if any is bare, else tried.
- **Composites apply the mode to everything they hold.** `TensorDataStruct`,
  `TensorDataSequence` and `OptionalTensorData` have `.Shared()` and `.TryConsume()`, as does
  a training checkpoint ([What a training step consumes](training.md#what-a-training-step-consumes)).
  A sequence that owns its elements (made by a sequence's `To`, `CopyTo` or `ToHost`) is
  handled per element: refused if another run reads any element; fed `.Shared()`, none can be
  deleted during the run; an element fed separately counts as another occurrence, so
  `Execute(e.Shared(), s)` reads `e` and consumes the rest of `s`. A struct field can carry
  its own mode — `def.FromOrderedData(tokens, mask.Shared())` — and a `.Shared()` field is
  always read; a struct fed `.Shared()` has every field read.
- **The error names the call to change.** Reading a consumed tensor throws
  `ObjectDisposedException` naming the run (graph and context) and input that took it.
- `.Shared()` / `.TryConsume()` on a tensor, struct, sequence or optional return a
  `SharedInput` (an `IData` with a `Mode`). On a training checkpoint they return a new
  checkpoint over the same tensors with its `FeedMode` set, kept by its derivations
  (`WithStep`, …) and `rig.AdoptCheckpoint`. `Run` reads each `NamedModelParam`'s `FeedMode`
  (`null` = as it is); on a parameter they return a copy with `FeedMode` set, so
  `graph.Run(p.Shared())` leaves `p` unchanged.

**Memory the run cannot read in place** — every tensor built from a C# array, and one from
another device or runtime — is fed through a copy in the run's memory, which Shorokoo makes
before the run with the backend's own move there
([Where a run's inputs and outputs are](#where-a-runs-inputs-and-outputs-are)):

- **Consumed**: the tensor is dead and its memory released at the feed; the run consumes the
  copy. On a card, the copy is made on the card, outside what the session allocates, and a
  budget counts it ([A context's device-memory budget](#a-contexts-device-memory-budget)).
- **Read**: the copy is made on first read and kept — a `TensorData` held by the source,
  attached to the reading context (visible in `context.Tensors`), and reused by later shared
  reads for as long as the source lives. Deleting or consuming the source retires it, and so
  does a training step letting go of the copies of the batch it read.

### Where a run's inputs and outputs are

Every backend has a **run memory**: the memory its sessions read their inputs in and leave
their outputs in. On a CUDA backend it is that card's memory for every tensor but a string one;
a string tensor and a sequence are in the host memory of the backend's runtime. On a CPU backend
it is host memory. That memory belongs to the backend: two backends on one card share it only
where they share a runtime.

- **Inputs.** Shorokoo puts every input in the run memory before the run: a tensor there
  already is handed over as it is, and any other — a C# array, a tensor of another device or
  runtime, a card's output fed to a run on the host — through a copy made with the backend's own
  moves. A session is handed nothing else; one handed a value outside its run memory refuses it
  with an `InvalidOperationException` rather than copying it.
- **Outputs.** Every output comes back in the run memory, as a new `TensorData` attached to the
  context that ran it — on a GPU backend, on the card, `IsHostResident` false. Nothing moves it
  afterwards. Reading its values copies them to the host and leaves it on the card; `ToHost()`
  makes a copy of it in host memory, `To(context)` puts it on another context, and a run on
  another context that is fed it moves it there as one of its inputs.
- **An output holds nothing of the session's.** On ONNX Runtime a session allocates through an
  allocator of Shorokoo's, and an output is the block its run wrote it into — or a range of the
  memory of an input the run consumed, where the run wrote it there
  ([below](#a-run-that-writes-into-what-it-consumed)). Keeping it keeps that memory and nothing else
  of the session's alive — the session may run on, sit idle or be disposed — and nothing copies it
  after the run. [Device memory](#device-memory-gpu-backends) says how blocks are laid out and what
  the allocator keeps between runs.

```csharp
var compiled = cuda.Compile(graph);
var y = compiled.Execute(x)[0].ToTensorData();     // on the card
float[] values = y.CopyMemory<float>();            // one copy across the bus; y stays on the card
var onCpu = cpu.Execute(graph, y.Shared());        // copied to the host as cpu's input
var next = compiled.Execute(y);                    // read where it is, and consumed: nothing crosses
```

On a CUDA backend an operator the provider runs on the host reads its inputs from the card and
writes its outputs back there ([Shorokoo/Shorokoo#493](https://github.com/Shorokoo/Shorokoo/issues/493)).
On a CPU backend nothing of this is visible: its run memory is the host's.

### A run that writes into what it consumed

ONNX Runtime holds every input until the run ends, so a consumed input's memory cannot be
freed mid-run. Instead a run can write **into** it, so what it writes there needs no memory of its
own. It does so two ways, both only into an input the run consumed — `.Shared()` memory is left as
it was — and fed as no other input. Both inputs and outputs are in the backend's run memory, so
where they are never stands in the way: a host tensor consumed by a card run is copied onto the
card first, and written into as that copy.

**Output aliasing.** An output a lowering marks as safe is written over the whole of an input of
its element type and the shape the session settled at build time. The only such lowering is the
training rig's step, which pairs each updated state field with the one it replaces
([A step writes its state over the state it consumed](training.md#a-step-writes-its-state-over-the-state-it-consumed));
a graph you compile yourself marks no output.

**Placement.** For a compiled graph, a run's values of a mebibyte or more — outputs and
intermediates alike — are written into ranges of the consumed inputs' memory where the graph the
backend runs proves a range free for the value:

- everything that reads what the range held runs before the value is written: by the graph's own
  edges, or on ONNX Runtime in the order the session runs its nodes — one at a time, in the order
  of the graph it writes out, which on a card holds for the nodes the CUDA provider runs;
- nothing reads it after the run: an output, or a view of one, is never written over;
- an operator that reads what it overwrites reads each element where it writes it — an
  element-wise operator in place, a slice at its own offset, a part of a concatenation already in
  its slot, a fused normalization's sum over its own input;
- on a card, the value is computed on the card: a node the CUDA provider has no kernel for runs on
  the host, and what it computes is not placed.

Where the values go is planned the first time a run *signature* — which inputs are consumed, the
shape of every input, the outputs asked for — runs, and kept for it.

| | ONNX Runtime | PyTorch |
|---|---|---|
| **How** | a second session, its placed values bound to their ranges: over the model, where that keeps the first session's order and can bind every placed value, and otherwise over the graph ONNX Runtime runs, its fusions made | a translation writing each placed value with torch's own operator: an `out=` form, the steps of a composed operator (`Gemm`, `Clip`, a normalization) in the range, cuDNN's convolution on a card, a fill, a concatenation part by part, or a copy of what a view reads or of a CPU convolution's result |
| **When it applies** | from the signature's first run, where placing saves — by a model of what the run holds at each node, in the order the second session runs them, less what that session holds of its own — more than the larger of a mebibyte and a sixty-fourth of the plain run. The first placed run is measured, and the runs after it run unplaced where it did not save that much, counting what the second session holds | from the first run: nothing placed allocates |
| **Not used** | where the second session would run other operators than the first; on an execution provider other than ONNX Runtime's CPU and CUDA ones; past 8 signatures; in a run of a graph not compiled (`Execute` or `Run` on the context); for a graph of over 20 000 nodes | in a training step whose gradient torch takes; for a model over 16 MiB; past 8 signatures; in a run of a graph not compiled; for a graph of over 20 000 nodes |

On ONNX Runtime a session keeps what it builds the second session from. A model of 16 MiB or less
it keeps in memory; a larger one on the host, in a temporary file of its own, deleted with the
session. On a card, a session over a model over 16 MiB reads the weights the model carries from
copies in the card's memory, which the second session reads too, and keeps no model: it writes out
the graph it runs, which the second session is built from, into a temporary folder kept for its
life. A session whose outputs are written into its inputs, such as a training step's, also keeps
the graph it runs, which it writes out to prove those outputs, in a temporary folder for its life,
on the host and on a card. The second session reads the weights fed to the session from the context's
memory as they are. Of the weights the model carries it holds a copy of its own for a model of
16 MiB or less, on the host as on a card; for a larger one, on the host, only the packed copies
ONNX Runtime makes of a product's weights, the rest mapped from the files the graph was written
into, and on a card none.

**Outputs on consumed memory.** An output written into an input stands on that input's memory —
its **block** — as a `TensorData` of its own over its range, never overlapping another's. Several
outputs of one run may stand on one block. How the block is freed depends on whose memory it is:

- **On ONNX Runtime**, where the block is one Shorokoo's allocator carved from its reserved memory
  — on the host a tensor of 64 KiB or more, made from your data or by a run; on a card any tensor,
  where the driver offers CUDA's virtual memory management
  ([Device memory](#device-memory-gpu-backends)): each output frees its own range as it ends, and
  what of the block no output stands on is freed as the run ends. The whole pages inside a range go
  back — 4 KiB on the host, 2 MiB on a card (512 bytes for a card block of a mebibyte or less) — to
  the allocator, as a tensor's own memory does.
- **On PyTorch**, and on a card whose driver does not offer virtual memory management, the block is
  freed when the last output on it ends, not before: torch frees a tensor's storage whole, and so
  does `cudaFree` a block. A run places outputs in such a block only where they leave at most a
  mebibyte of it unused.

A device-memory budget counts a block once, for what of it is still held, for as long as any
tensor on it is attached ([A tensor's lifetime](#a-tensors-lifetime-locks-and-deletion)).

Otherwise nothing differs: outputs are new `TensorData` attached to the running context, and
values are the same.

### A tensor's lifetime: locks and deletion

A `TensorData` **is** its memory: one object per allocation, or per range of a block several
tensors stand on — the memory of an input a run consumed and wrote them into
([A run that writes into what it consumed](#a-run-that-writes-into-what-it-consumed)). Ranges
never overlap, and each tensor's range is freed with it, or the block with the last tensor
standing on it ([Outputs on consumed memory](#a-run-that-writes-into-what-it-consumed)). A tensor records the
backend that allocated it (`AllocatingBackend`), which releases it, and where it lives: `Space`
(the device) and `Location` (device plus runtime). It does not know which contexts it is attached
to — see [Moving data between contexts](#moving-data-between-contexts).

**How a tensor ends** (disposing a context is not one of them):

| | |
|---|---|
| **Deleted** | `Delete()`, `Dispose()`, `TryDelete()` or `DeleteAsync(...)` — below. |
| **Consumed** | fed to a run as it is — or through `.TryConsume()` with nothing else reading it — which takes it when the run starts; see [Feeding a run](#feeding-a-run-consumed-shared-or-tried). |
| **Moved into an attribute** | `MoveToAttribute()`, which takes its contents — see [core-types.md](core-types.md#the-two-conversions-and-which-one-spends-its-source). |

A run's read-copy of a tensor ([above](#feeding-a-run-consumed-shared-or-tried)) ends when its
source ends, or releases its copies (as a training step does for its batch).
Elements a sequence owns end with the sequence, except one a run is reading on its own
account; a sequence whose owned element a run is reading cannot be disposed.

Every access to a dead tensor — reading, feeding, `To`, `CopyTo`, `ToHost`, `Shared()`,
`TryConsume()`, `MoveToAttribute` — throws `ObjectDisposedException` saying why (for a
consumed one, which run took it). `Shape`, `DType`, `ToString()`, `IsDisposed`,
`AllocatingBackend`, `Space`, `Device` and `Location` keep working. Ending a dead tensor is a
no-op, so double disposal is harmless.

An unreferenced tensor is reclaimed by the GC through its backend; deleting only chooses
*when*. Runtime-allocated buffers (run outputs, card copies) are native and freed at once — a
tensor standing on a block with others frees the block when it is the last; a tensor built from a
C# array frees its native read-copies at once and leaves the array to the GC.

**What a run holds.** A run holds a reader lock on every tensor it reads until it returns; any
number of runs may read one tensor. While locked, `Delete()` and `Dispose()` throw
`InvalidOperationException` and `TryDelete()` declines. `ToHost()`, `CopyTo(...)`,
`CopyMemory()`, `ValueAt()`, `CopyRawMemory()`, `MoveToAttribute()`'s copy, and the copy
`AccessMemory()` makes of a tensor on a card take the same lock while copying. A span
`AccessMemory()` hands out over host memory is not covered: keep the tensor alive and unfed while
you hold one. One over a card tensor's host copy is valid on its own.

Locks are taken inside the run, one feed at a time. A feed deleted from another thread while a
run is starting makes `Execute` throw `ObjectDisposedException`, with any feeds already taken
staying consumed — see
[A feed deleted while a run is starting loses that run](limitations.md#a-feed-deleted-while-a-run-is-starting-loses-that-run).

Disposing a `ComputeContext` throws while a run of it is in flight or it holds a lock;
disposing a compiled graph throws while one of its runs is in flight.

**Deleting:**

| | What it does |
|---|---|
| `Delete()` / `Dispose()` | The same operation: ends the tensor and releases its memory now. Throws if a run is reading it. |
| `bool TryDelete()` | The same if no run is reading the tensor. If one is, it changes **nothing** and returns `false`. `true` for a tensor already dead. |
| `Task<bool> DeleteAsync(timeout, cancellationToken)` | Ends the tensor at once, asks whatever is reading it to stop, and waits up to `timeout` for the memory to come back. |

With `DeleteAsync` the tensor is deleted either way; `false` means only that the memory had not
come back in time (it will when the run ends — do not retry). A timeout never rolls back the
stop request, and the token cancels the wait, not the deletion. The wait is bounded like
[stopping a run](#stopping-a-run): by the longest operator in flight.

**Host memory.** A tensor built from a C# array belongs to no context; its allocating backend
is `HostBackend.Instance`, readable by every host backend. `ComputeContext.Host` names host
memory as a target for `To` and `CopyTo`; `Compile`, `Execute`, `Run` and `Eval` on it refuse,
and it cannot be disposed.

A graph's literals are
[`TensorAttribute`s](core-types.md#two-kinds-of-concrete-tensor-tensordata-and-tensorattribute),
immutable and without lifetime.

### Moving data between contexts

A tensor's memory never moves. None of these changes the tensor it is called on:

| | Result |
|---|---|
| `t.To(context)` | `t` itself, if `context`'s backend can read its memory as it stands; otherwise a new copy in `context`'s memory. Either way attached to `context`. |
| `t.CopyTo(context)` | Always a new, independent copy in `context`'s memory, attached to `context`. |
| `t.ToHost()` | `t` itself, if the host can read its memory; otherwise a new copy in the framework's own host memory, attached to nothing. |

A backend reads memory in place only on **the same device and the same runtime**. Host memory
from a C# array counts as every host backend's. Two backends over one ONNX Runtime share card
allocations; a CPU backend reads them through a copy; two isolated runtimes on one card copy
through the host. A run's outputs are in its backend's memory
([Where a run's inputs and outputs are](#where-a-runs-inputs-and-outputs-are)).

```csharp
var onCard = cuda.Compile(model).Execute(input)[0].ToTensorData();  // device memory
var onHost = onCard.To(cpu);        // one copy across the bus; onCard is untouched
var same   = onHost.To(otherCpu);   // no copy: the very same object
var mine   = onHost.CopyTo(cpu);    // always a copy
```

**Where `To` or `ToHost` needs no copy, the result is `t`**: feeding it as it is consumes `t`,
and `using var h = t.ToHost();` deletes `t`. Use `CopyTo` for an independent tensor, or
`.Shared()` to keep `t` past a run.

**Attachment is bookkeeping, not ownership.** `context.Tensors` is a weak list of the context's
run outputs, what its runs read, and what `To` and `CopyTo` placed. It never keeps a tensor alive
or ends one. `context.Detach(t)` removes a tensor (refused while a
run of that context reads it) without deleting it. Disposing a context releases its sessions
and leaves every tensor. A budgeted context counts this list — see
[A context's device-memory budget](#a-contexts-device-memory-budget).

`TensorDataStruct` and `TensorDataSequence` take the same three operations. A struct comes back
as itself where nothing was copied; a sequence is copied whole if any element must be.

### One model, two devices

A `ComputeContext` constructed with a backend compiles and runs there, and contexts may name
different backends:

```csharp
using Shorokoo.Core.Backends;
using Shorokoo.LinuxCPU;
using Shorokoo.LinuxGPU;

var cpu  = new ComputeContext(new LinuxCpuBackend());
var cuda = new ComputeContext(new LinuxGpuBackend());

var onHost = cpu.Execute(graph, input.Shared());   // the host, reading input and leaving it
var onCard = cuda.Execute(graph, input);           // the same graph, the same input, the card;
                                                   // its outputs are on the card
```

Both run the same model, one C# build; each context compiles its own session. `ComputeContext.Backend` and `CompiledGraph.Backend` name
the backend. Tensors you build (allocating backend `HostBackend.Instance`) are tied to no
runtime, so building and exporting needs none and either context accepts them. A tensor not
readable in place is copied per run when consumed, or once per (tensor, runtime) when fed
`.Shared()`, for as long as it lives ([Feeding a run](#feeding-a-run-consumed-shared-or-tried)).
A tensor on the card (`TensorData.IsHostResident` false) — every output of a run there, and the
state a [training step](training.md#keeping-training-state-on-the-device) hands back — crosses
via the host when it is fed to the CPU context or moved with `ToHost()`; reading its values
copies them to the host and leaves it on the card.

#### Deploying two backends

Both packages deliver `runtimes/<rid>/native/libonnxruntime.so` at the same path, so
referencing both normally deploys only one native — which is why
[auto-discovery](#auto-discovery) refuses it.

**One runtime, two providers (usual).** The CUDA build of ONNX Runtime includes the CPU
provider. Deploy the GPU package, and the CPU package without its native:

```xml
<PackageReference Include="Shorokoo.LinuxGPU" Version="..." />
<PackageReference Include="Shorokoo.LinuxCPU" Version="..." ExcludeAssets="native" />
```

**Then name the default explicitly, before anything runs** — discovery sees two candidates and
refuses on the first read of `DefaultBackend.Instance`, possibly inside a framework call.
Constructing a backend is not enough; assign it:

```csharp
DefaultBackend.Instance = new LinuxCpuBackend();   // startup, before anything runs

var cpu  = new ComputeContext(DefaultBackend.Instance);
var cuda = new ComputeContext(new LinuxGpuBackend());
```

**Two runtimes** (two ONNX Runtime builds or versions in one process): give each native its own
folder and load the second with `IsolatedBackend`:

```xml
<!-- The glue, referenced directly: it carries the target that reads the items below, and
     it is the assembly every isolated backend loads. Coming in through a platform package
     is the usual route, and this recipe cuts that route on purpose. -->
<PackageReference Include="Shorokoo.OnnxRuntime" Version="..." />

<PackageReference Include="Microsoft.ML.OnnxRuntime" Version="1.30.0"
                  ExcludeAssets="all" GeneratePathProperty="true" />
<PackageReference Include="Microsoft.ML.OnnxRuntime.Gpu.Linux" Version="1.30.0"
                  ExcludeAssets="all" GeneratePathProperty="true" />

<ShorokooBackendNatives BackendId="cpu"
  Include="$(PkgMicrosoft_ML_OnnxRuntime)/runtimes/linux-x64/native/*" />
<ShorokooBackendNatives BackendId="cuda"
  Include="$(PkgMicrosoft_ML_OnnxRuntime_Gpu_Linux)/runtimes/linux-x64/native/*" />
```

Each `ShorokooBackendNatives` item lands in `ort/<BackendId>/`; deploy a package's whole
native folder, since providers must sit beside their core. The direct `Shorokoo.OnnxRuntime`
reference is required: without it the items are silently ignored and the first
`IsolatedBackend.Load` throws `FileNotFoundException` for the missing native. It is not a
backend and never a discovery candidate.

**Deploy the backend assembly beside its native, not beside `Shorokoo.dll`** (two there is
the ambiguity discovery refuses), and point `ProbeDirectory` at it:

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

The ONNX Runtime wrapper and glue are found beside the backend first, then beside
`Shorokoo.dll`. `Name` is the label `Backend.Name` reports; loading one native twice under two
names is refused, and loading the same spec twice returns the same backend. A loaded backend
lives until the process exits. Only the wrapper, glue and backend assembly are private to it;
the core assembly and your model are shared, so contexts can exchange data.

Sequence outputs work on any provider; ONNX Runtime materializes them in host memory — see
[A sequence's elements live in host memory](limitations.md#a-sequences-elements-live-in-host-memory).

#### Two CUDA backends in one process

CUDA backends of different frameworks run side by side in one process, whichever starts first,
because they run on one copy of cuDNN and cuBLAS — see
[The NVIDIA libraries the CUDA backends run on](#the-nvidia-libraries-the-cuda-backends-run-on).

#### Or keep it to two processes

Two executables over a shared, backend-free model library suit separate jobs — a long training
run and an occasional check:

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

A backend reference in the shared library — a `PackageReference`, or a `ProjectReference` to
an executable carrying one — flows into both outputs and makes
[auto-discovery](#auto-discovery) refuse. Keep backends in executables, and reference no
executable.

### The NVIDIA libraries the CUDA backends run on

Every CUDA backend runs on the same release of cuDNN and cuBLAS (with cuBLASLt), CUDA 13 builds,
pinned per platform beside the lock of the CUDA Python environment: the releases that
environment's PyTorch carries.

| | cuDNN | cuBLAS |
|---|---|---|
| Windows x64 | 9.24.0.43 — 393 MiB download, 542 MiB on disk | 13.0.0.19 — 382 MiB download, 504 MiB on disk |
| Linux x64 | 9.24.0.43 — 528 MiB download, 921 MiB on disk | 13.1.1.3 — 404 MiB download, 568 MiB on disk |

They live in a per-user cache, one folder per release: `%LOCALAPPDATA%\shorokoo\cuda\` on
Windows, `$XDG_CACHE_HOME/shorokoo/cuda/` (or `~/.cache/shorokoo/cuda/`) on Linux, beside the
Python environments — `cudnn-9.24.0.43-cu13\`, `cublas-13.0.0.19-cu13\`. A folder is filled once,
the first time a CUDA backend needs it, under a lock file beside it, and marked complete only once
every file is in place and checked; one left half-filled is filled again. It is filled from:

1. **A copy that matches exactly**: every file of the release present, each with the SHA-256 its
   PyPI wheel records. The provisioned CUDA Python environments beside the cache are looked in
   first — `torch\lib` on Windows, the NVIDIA wheels' own folders on Linux — since their PyTorch
   carries the pinned release byte for byte, whichever backend runs first. Then, on Windows, cuDNN
   is looked for on `PATH`, in `%CUDNN_PATH%` and under `%ProgramFiles%\NVIDIA\CUDNN`, and cuBLAS
   on `PATH`, in `%CUDA_PATH%` and in the CUDA 13 toolkits under
   `%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA`; on Linux, on `LD_LIBRARY_PATH`, in
   `$CUDNN_PATH`, `$CUDA_PATH` and `$CUDA_HOME`, in the CUDA 13 toolkits under `/usr/local` and in
   the system's library folders. The copy's files are hard-linked into the cache where the volume
   allows, so nothing is downloaded or stored twice, and copied otherwise. Any other copy — another
   release, a build for another CUDA major, a file that differs — is ignored, never mixed in.
2. **Otherwise the release's wheel from PyPI**, checked against the SHA-256 the pin records. A
   download through which no data arrives for a minute is given up, as is one not done within the
   hour a fill may take; and a download ended with its process leaves no partial wheel behind.

A filled folder is looked at again each time it is used, without reading its files: one whose
size, or whose last write, has changed since the folder was filled — an installed copy it was
linked to, written over in place, say — has the folder checked whole, file by file against the
pin, and filled again where any file is no longer the pinned one. A folder copied from another
machine, its files' times moved, passes that check and is kept.

With neither available — offline, no copy that matches — the first CUDA session fails with an
`InvalidOperationException` naming the library, where it looked and the size of the download. A
cache folder that cannot be written — a full disk — fails it with an `IOException` instead. To
run offline, fill the cache once while online, copy the folders from a machine that has them, or
install exactly the pinned release. `CudaLibraries.Prepare()` (in
`Shorokoo.Core.Backends`) fills the cache and loads the libraries at a moment of your choosing,
at startup rather than on the first CUDA run; it is also what a backend of your own that loads
CUDA libraries calls first.

Within a process every CUDA backend binds to that one copy. The ONNX Runtime CUDA backend loads
the cache's files by full path before its provider loads anything by name, so it never runs on
the cuDNN or cuBLAS that `PATH` happens to offer. A provisioned
[PyTorch CUDA environment](pytorch-backend.md#cpu-and-cuda)'s own copies — `torch\lib` on Windows,
the `nvidia` wheels' folders on Linux — are made hard links to the cache's files when the
environment is provisioned, or first used, so PyTorch loads the very same files. This is what
lets an ONNX Runtime CUDA backend and a PyTorch one share a process in either order: a process
holding two releases of cuDNN cannot run both, since cuDNN's libraries import one another's
internal entry points by name and bind to whichever copy of a name was loaded first.

So whatever else in the process loads cuDNN or cuBLAS has to do so once the pinned copies are
loaded: a CUDA session of your own built on ONNX Runtime directly, say, calls
`CudaLibraries.Prepare()` before appending its provider. A process that already holds another
release when the ONNX Runtime CUDA backend prepares the pinned one — any copy, under the name a
pinned file is loaded by, that is not that file — is refused before anything is loaded: the
backend's first session fails with an `InvalidOperationException` naming the copies held. Loading
the pinned files beside them would not help, as what imports those names binds to the copies
loaded first.

The other NVIDIA libraries both stacks load — the CUDA runtime, cuFFT, nvrtc, nvJitLink — are
each backend's own: the ONNX Runtime backend and Shorokoo's allocator on the card load them by
name, from the machine's CUDA 13 runtime or the copy the process already holds under that name,
and PyTorch loads its environment's. Two copies of these run side by side: they call one another
only through public, versioned entry points, and both work in the card's one context per process,
so memory either one allocates is good to the other.

### Precision (GPU backends)

`float32` is computed in full `float32` precision on every backend and every device: a product, a
convolution or a recurrent layer of `float32` operands is computed in `float32` throughout, so a
model computes the same on a card as on the host, but for the order its sums are added in. Measured
on an RTX 4090, a 1024-square `MatMul`, a 64-channel 3×3 `Conv` and a 256-unit `LSTM` land within
5e-6 of the host's results (the largest difference over the largest value), on every CUDA
backend.

A card from NVIDIA's Ampere generation on can compute those operators faster in **TensorFloat-32**
(TF32): on its tensor cores, with each operand's significand rounded from 24 bits to 11 and the sums
kept in `float32`. A context asks for it with its `Precision`:

```csharp
using Shorokoo.Core.Backends;
using Shorokoo.Runtime;

var fast = new ComputeContext(new LinuxGpuBackend())
{
    Precision = new PrecisionSettings { AllowTensorFloat32 = true },
};
```

The same three operators then land up to 2.5e-3 from the host's, some five hundred times further
than in full precision. What full precision costs against it, measured on an RTX 4090: a 4096-square
`MatMul` takes 1.57 times as long, and a convolutional network's training step 2.6 times as long
on ONNX Runtime and 2.3 times on PyTorch; the training steps of the other model families measured
were within their run-to-run noise.

| Backend | By default | With `AllowTensorFloat32` |
|---|---|---|
| ONNX Runtime CUDA (`WinGpuBackend`, `LinuxGpuBackend`) | the CUDA provider's `use_tf32` is `0` | `use_tf32` is `1`: cuBLAS products and cuDNN convolutions and recurrent layers in TF32 |
| [PyTorch](pytorch-backend.md#runs) CUDA | each run sets `torch.backends.cuda.matmul.allow_tf32` and `torch.backends.cudnn.allow_tf32` off as it starts | each run sets both on: products, cuDNN convolutions and recurrent layers in TF32 |
| [JAX](jax-backend.md#runs) CUDA | every product and convolution is compiled at `Precision.HIGHEST` | compiled at `Precision.HIGH`: products and recurrent layers in TF32; each convolution in TF32 or in full precision, whichever kernel XLA's autotuner finds faster when it compiles the program |
| every CPU backend | full precision | no effect: full precision |

- **It is read when a session is built**, like [`DeviceMemory`](#device-memory-gpu-backends): a
  graph compiled on the context keeps the precision the context carries, and a training rig's steps
  compute in its `runtimeContext`'s. Set it on the context from the start.
- **It allows TF32, and never requires it.** A backend uses it where its kernels choose to: XLA
  puts a small product through a full-precision kernel all the same, and a convolution through
  whichever kernel it timed fastest, which may be either.
- **Only `float32` changes.** Products of `Float16`, `BFloat16` and `Double` operands compute as
  they do either way.
- **PyTorch's switches are the whole process's.** A run on a card sets them from its own session as
  it starts, so a session's precision does not depend on what ran before it, nor on code of your own
  that sets them. Runs of sessions that set them differently do not overlap: while a run that allows
  TF32 is running, runs on the cards that do not wait, and the other way round.
- **`NVIDIA_TF32_OVERRIDE=0`** in a process's environment turns TF32 off in cuBLAS and cuDNN for the
  whole process, a context that allows it included.

### Device memory (GPU backends)

On ONNX Runtime every session allocates through an allocator of Shorokoo's — one per device, the
host or a card, for the whole process, whichever runtime or backend built the session — rather
than through an arena of its own, and the tensors a context places on a card, and every tensor
made in host memory outside a run, come from the same one. A session's blocks are carved from address space reserved for it with no memory behind it:
memory is committed under a block as it is carved, and handed back to the system as the session
lets it go. Each request is rounded up and served by its size:

- **On the host**, a block under 64 KiB is an allocation of its own from the C runtime's heap,
  which may keep its pages for the process's next allocations; a larger one is a whole number of
  4 KiB pages, its memory committed 64 KiB at a time (`VirtualAlloc` on Windows, `mmap` and
  `mprotect` on Linux) and handed back the same way (`VirtualFree`, `madvise`).
- **On a card**, a block of up to a mebibyte is carved from memory every session on the card
  shares, packed into the card's 2 MiB pages as the CUDA runtime packs small allocations; a larger
  one is a whole number of 2 MiB pages of its own, mapped into one contiguous range with CUDA's
  virtual memory management (`cuMemCreate`, `cuMemMap`). Where the driver does not offer it, every
  block is a `cudaMalloc` of its own, a multiple of 512 bytes up to a mebibyte and of an eighth of
  the power of two below it above that.

So a block holds its own memory: on a card a block over a mebibyte has its pages to itself, and
a smaller one shares a page with other small blocks; on the host a block shares at most the
64 KiB at each of its ends with its neighbours.

- **What a session lets go of is kept for its next runs**, so a loop's runs find their blocks
  waiting. A request no kept block serves is carved from the memory the session has committed,
  whichever block it last belonged to, so a session fed one shape after another reuses its warm
  memory as an arena does rather than taking each new size from the system. A session never holds
  more than the most one of its runs has used — what it had in use as the run began, and every
  block the run was handed, each counted once: where a request would take it past that mark, the
  smallest larger block it keeps serves it, or the blocks it kept longest give way to it, and what
  is still over the mark goes back to the system — on the host at once, on a card as the run ends.
  So a loop whose runs repeat finds every block it needs, and a session fed varied shapes holds
  what its busiest run used, not a block of every size it has seen. What the tensors placed on a
  card let go of is kept the same way, for the next tensor placed there, counting what is placed
  between two runs on the card as one run; so is what a tensor made in host memory outside a run
  lets go of under a mebibyte, while from a mebibyte its memory goes back to the system as it goes,
  as the C runtime's heap hands back a block that large: nothing says another of its size follows.
  The small blocks every session on the card shares keep no more memory committed than they had in
  use at their busiest.
- **What is kept goes back to the system**:
  - at the end of a run that asks for it (`RunSettings.ShrinkArenaAfterRun`, always on under a
    budget): what its session keeps, and what the placed tensors left on that device;
  - as a session is disposed, or collected undisposed: everything it keeps, and each block it
    still has out — an output a caller keeps — as that goes;
  - before a session's budget, or the device itself, would refuse a request for want of room;
  - when the program calls `DeviceMemory.ReleaseCached()`: everything kept on every device, for
    every session and every placed tensor, with no run to make. It returns the bytes it handed back.

  Nothing else returns it. A long-lived program doing varied work holds, besides what is in use,
  up to the busiest run of each session it keeps alive and of the tensors it placed on each device,
  and nothing of a session it disposed. ONNX Runtime's own arena, where a session uses one,
  keeps a session's busiest run too, rounded up to the regions it grows by, until the session is
  disposed or a run asks it to shrink.
- **Sessions run without ONNX Runtime's memory pattern.** With it, a session's runs from the second
  on take their planned tensors as one block laid out by the first run; that block came out larger
  than what the runs have in use at once with a block per tensor on every graph measured — 33–50%
  on a scenario of slices, fills and concatenations, 3–24% on training steps, 7–37% on inference —
  for fewer allocations, which cost little: training steps measured without the pattern ran within
  2% of their time with it on the host and on a card, the smallest, a 4×2 linear, within the 8% its
  own runs vary by. So `AllocationCount` counts a run's tensors one by one.
- **A request that cannot be served fails the call that made it**, as ONNX Runtime's own
  allocators fail one: a block the budget leaves no room for, or one the device does not have,
  fails the run — or the placing of the tensor — with an allocation failure, and leaves the
  device as usable as it was for the next one. The text names the memory:

  ```
  [ErrorCode:RuntimeException] Non-zero status code returned while running Expand node. Name:'N3'
  Status Message: Failed to allocate 4503599627370496 bytes on CUDA device 0: the card has no such
  block free (CUDA refused it), with everything Shorokoo's cuda_allocator kept for reuse
  handed back.
  ```

Settings apply to the sessions the context compiles; `CompiledGraph.DeviceMemory` reports what a
graph got:

```csharp
using Shorokoo.Core.Backends;
using Shorokoo.Runtime;

var ctx = new ComputeContext
{
    DeviceMemory = new DeviceMemorySettings
    {
        LimitBytes = 16L * 1024 * 1024 * 1024,   // a 16 GiB budget on what ctx holds on the card
    },
};

var compiled = ctx.Compile(graph);
compiled.Execute([batch1]);

Console.WriteLine(compiled.DeviceMemory.LimitBytes);   // what the budget left the session's last run
```

| setting | on | what it does | default | read |
|---|---|---|---|---|
| `LimitBytes` | `DeviceMemorySettings` | the most the context holds on the card; a run's session may allocate the budget less what the context holds there | `null` — no budget | on every transfer onto the context and every run of it — see [A context's device-memory budget](#a-contexts-device-memory-budget) |
| `ShrinkArenaAfterRun` | `RunSettings` | as the run ends, its session's allocator hands back to the device what it keeps for the session, and what it keeps of tensors that are gone | `false` — and forced on under a budget | on every run |

- `ShrinkArenaAfterRun` makes the next run commit its memory again — on a card mapping each 2 MiB
  page anew, which costs tens of microseconds a block on Windows — so outside a budget use it only
  when the card is shared. On the CPU backend it hands the session's host memory back the same
  way, and the next run's pages are each zeroed by the system as they are first touched.
- `LimitBytes` is hard: a run that needs more fails with an allocation failure naming the limit,
  so a figure set too low fails work that would fit.
- Context settings are `init`-only. `CompiledGraph.Execute` / `Run` can override
  `ShrinkArenaAfterRun` per call on an unbudgeted context; the context's one-shot entry points
  and a rig's `TrainStep` use the context's.
- **Scope is the context, never the process.** Two contexts may have different budgets and never
  affect each other's sessions; to use different settings, compile on another context — e.g. one
  for a training loop, one for variable-shape inference. Every session on a card, and every tensor
  placed there, allocates through that card's allocator, held for the process's life; each
  session's figures and limit are its own, and the budget counts placed tensors by attachment.

The static `DeviceMemory` class reports what the card is doing:

```csharp
using var run = rig.BeginResidentRun(checkpoint);
for (int step = 0; step < steps; step++)
{
    run.Step(input.Shared(), target.Shared());   // read, so the next step can use them
    DeviceMemory.Sample();
}
Console.WriteLine($"this process at its peak {DeviceMemory.PeakProcessBytes / (1024 * 1024)} MiB");
Console.WriteLine($"the card at its peak {DeviceMemory.PeakUsedBytes / (1024 * 1024)} MiB");
```

`Read()` returns a `DeviceMemoryReading` (`UsedBytes`, `FreeBytes`, `TotalBytes`,
`ProcessBytes`); `Sample()` also folds it into `PeakUsedBytes` and `PeakProcessBytes`;
`ResetPeak()` restarts both.

- `UsedBytes`, `FreeBytes` and `TotalBytes` are the **device's**, including other processes.
- `ProcessBytes` is **this process's** share: everything it holds on the card — weights and
  optimizer state, every session's memory, the CUDA context and its libraries' workspaces — and
  nothing another process holds. `PeakProcessBytes` is therefore the one figure for "the most this
  run held on the card". It is the figure Task Manager shows per process on Windows, and
  `nvidia-smi` lists per process where it can.
- `ProcessBytes` is read from DXGI for a Windows card driven by WDDM (a GeForce, or any card
  driving a display), where `nvidia-smi` prints `[N/A]` for it, and from NVML everywhere else. It
  is `null` where neither answers for the card: where DXGI does not, and NVML is not installed,
  gives no per-process figure, or does not see this process's id, as in a container with its own
  process-id namespace. The two figures of one reading are taken one after the other, so while
  this process allocates on another thread its share can momentarily read above the card's.
- The peaks are the largest of your own `Sample()` calls; nothing samples on its own. A
  sample cost about a microsecond on an RTX 4090 under WDDM, and NVML's process list about 120
  microseconds, so one per step is cheap — and, unlike an external poller
  such as `nvidia-smi`, cannot miss the step.
- With no CUDA runtime both return `null`, so the calls can stay in CPU code.
- The first reading initializes this process's CUDA context (a few hundred MiB) if none exists;
  take it after the backend is up. Readings use the thread's current CUDA device (device 0
  for the shipped GPU backends).

### A context's device-memory budget

`DeviceMemorySettings.LimitBytes`, on a context whose memory is a card's, budgets
**everything that context holds there**: its attached tensors in card memory and, during a run,
what that run's session allocates. It is per context, not per session or process:

```csharp
using var ctx = new ComputeContext(gpu)
{
    DeviceMemory = new DeviceMemorySettings { LimitBytes = 2L * 1024 * 1024 * 1024 },
};

var onCard = big.CopyTo(ctx);                     // on the card, and on ctx's books
var use = ctx.ReadDeviceMemoryUse();
Console.WriteLine($"{use.AttachedBytes} of {use.LimitBytes} bytes attached, {use.AvailableBytes} left");
```

**What it counts.** Tensors placed by `To` or `CopyTo`, read or
copied there by the context's runs, or left there as its runs' outputs — every output of a run
on the card, until it is deleted, collected or detached.
`ReadDeviceMemoryUse()` reports `AttachedBytes`, `AttachedTensors` and `LimitBytes` (`null`
with no budget, or for a host context). A tensor on two contexts counts on both; collected or
`Detach`ed tensors drop out, and so do dead ones once their memory is released. A tensor deleted
with `DeleteAsync` while a run, or the session of a graph loaded with `LoadCompiled`, still reads
it keeps counting until that reader stands down, since its memory is still on the card until then.

**A transfer it cannot take is refused before allocating** — `To`, `CopyTo`, a run's card copy
of a tensor it cannot read in place, and a `To` that
merely attaches a tensor already on the card. Structs and sequences are checked whole (a
run-produced sequence per element as copied), and a failure undoes what was placed. The refusal
is an `InvalidOperationException`:

```
CopyTo(context) of Tensor (8388608,):Float32 asks this compute context for 33554432 bytes of CUDA
device 0 memory, which its device-memory budget cannot give: the budget
(DeviceMemorySettings.LimitBytes) is 67108864 bytes, and 50331648 bytes of it are attached to the
context there, in 1 tensor(s), leaving 16777216. Delete what the context no longer needs, or give
the context a larger budget.
```

**A run's session gets what the context leaves it.** What a run's session may allocate on the
card is the budget less the *discount*: what the context holds on the card apart from the session
for the length of the run — attached tensors, and those the run reads or copies there (for a
tensor another runtime holds on the same card, both it and its copy). A tensor already on the card
is read in place and never counts against the session. A host tensor fed to a card run is copied
onto the card before the run, outside the session, and counted in the discount:

- `.Shared()`: the copy is kept for later reads;
- consumed: the copy is the run's, and goes when the run no longer reads it. A tensor fed to
  several inputs is copied once, and counted once.

What the session allocates is its weights, everything the run computes, and the run's outputs
until it returns. From then on every output on the card is counted with the attached tensors, in
the discount of every later run until it goes, so delete each output once you are done with it.
Outputs [written into consumed memory](#a-run-that-writes-into-what-it-consumed) count that
memory once, for what of it is still held, while any of them is attached.

A run whose discount leaves its session nothing is refused before taking anything; one whose
session needs more than it was left fails with an allocation failure naming the limit:

```
[ErrorCode:RuntimeException] Non-zero status code returned while running Expand node. Name:'N3'
Status Message: Failed to allocate 167772160 bytes on CUDA device 0: Shorokoo's cuda_allocator may
hold 163577848 bytes there for this session, what the device-memory budget
(DeviceMemorySettings.LimitBytes) leaves its run, and it holds 512.
```

**The limit follows the discount, run by run.** On ONNX Runtime a session's allocator holds it to
its limit as each block is asked for, so the session takes the limit each run is left without
being built again, and what the context frees is the next run's room.
`CompiledGraph.DeviceMemory.LimitBytes` is the limit its last run was given. A backend whose
sessions take a limit only when they are built (PyTorch) builds one with the budget less the
discount, rounded up to the next sixty-fourth of the budget, and builds it again, before taking
anything, only when the discount grows past that: at most sixty-four times over a compiled
graph's life, plus once each time the remainder halves within its last sixty-fourth. Its limit
only comes down, until the graph is compiled again, and its statistics restart with each build.

**One at a time.** Under a budget the context's runs, transfers onto it and compiles on it are
serialized, and every run hands back what its session's allocator keeps for it as it ends,
whatever `ShrinkArenaAfterRun` says. None of this applies without `LimitBytes`.

What the budget does *not* count is in
[Known limitations](limitations.md#a-device-memory-budget-counts-tensors-not-what-the-allocator-keeps).

### What one session's allocator did

To read **one session's own allocator** (CPU or GPU), ask the compiled graph:

```csharp
using Shorokoo.Core.Backends;
using Shorokoo.Runtime;

var compiled = ctx.Compile(graph);
compiled.Execute(inputs);

if (compiled.ReadArenaStatistics() is { } allocated)
    Console.WriteLine($"{allocated.InUseBytes} in use, {allocated.MaxInUseBytes} at its highest, "
                    + $"{allocated.TotalAllocatedBytes} taken from the device");
```

`ArenaStatistics` has ten figures: `InUseBytes`, `RequestedInUseBytes`, `MaxInUseBytes`,
`MaxAllocSizeBytes`, `TotalAllocatedBytes`, `LimitBytes` (the most the session may allocate; `-1`
with no budget), `AllocationCount`, `ArenaExtensionCount`, `ArenaShrinkageCount` and
`ReserveCount`. It is `null` on a backend that reports none.

- On ONNX Runtime they are the session's own figures in the allocator it allocates through: on the
  card for a CUDA session, in host memory otherwise. `ArenaExtensionCount` is the blocks the
  session holds from the device now, in use or kept, and falls as they go back;
  `ArenaShrinkageCount` is the blocks it has handed back; `ReserveCount` is zero.
- `RequestedInUseBytes` is the part of `InUseBytes` the callers asked for; the rest is what the
  allocator added to round each request up to its size class. A backend whose allocator does not
  report what was requested (JAX) reports `InUseBytes` here, rounding included.
- `TotalAllocatedBytes` is what the session holds from the device, in use or kept for its next
  runs. It is not a bound on card usage; use `DeviceMemory.Read()` for that.
- `MaxInUseBytes` is a lifetime high-water mark; it cannot be reset and handing memory back does
  not lower it. For per-run figures use [`RunStats`](#per-run-statistics-on-the-context).
- A session's weights are among what it allocates, so it is non-zero before the first run, and the
  first run's peak includes them.
- A run's outputs are blocks of their session's: `InUseBytes` after a run is the weights and the
  outputs a caller still holds, and an output's block leaves it when the output goes.

### What crossed the bus

A device session that hands part of a graph to the host stages the crossings through a **pinned
host arena**:

```csharp
if (compiled.ReadPinnedArenaStatistics() is { } pinned)
    Console.WriteLine($"{pinned.MaxInUseBytes} of pinned host memory at its highest");
```

Same ten figures, not included in `ReadArenaStatistics()`; `null` on CPU backends. Zero for a
graph that stays on the card.

### Per-run statistics on the context

Collection is **off by default** and free until enabled:

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
                + $"{stats.ArenaExtensionCount} blocks taken from the device");

foreach (var run in stats.RecentRuns.TakeLast(5))
    Console.WriteLine($"run {run.RunNumber}: {run.PeakBytes} ({run.PeakKind}), "
                    + $"its session stood at {run.PriorPeakBytes} before it");
```

`RunStats` covers every run of the context across **all** its sessions.

- **Aggregates are exact; per-run detail is bounded.** `PeakBytes`, `RunCount`,
  `AllocationCount`, `ArenaExtensionCount`, `ArenaShrinkageCount`, `LargestAllocationBytes` and
  `ArenaBytes` cover the whole history. `RecentRuns` keeps the last
  `DiagnosticSettings.RecentRunCapacity` records (1000 by default; zero keeps none).
- **`PeakKind`**: a run that raised its session's high-water mark is
  `MemoryFigureKind.Measured`; one that stayed below an earlier mark is
  `MemoryFigureKind.UpperBound` (it used at most that).
- **`PriorPeakBytes`** is the mark the run found. On the first run it is the weights, so
  `PeakBytes - PriorPeakBytes` is the run's own use; later it is how far the run exceeded the
  record (zero for `UpperBound` runs).
- `PeakBytes` is the largest mark of any one session, not a sum across sessions.
- `ArenaBytes` (taken from the device) is usually above `PeakBytes`, but shrinking runs —
  every run under a budget — can leave it below; compare ordering, not figures.
  `ArenaExtensionCount` counts blocks held, so it undercounts after shrinkage.

### Did part of my GPU graph run on the host?

A CUDA session leaves operators the provider cannot run to the host. Two signals:

```csharp
switch (compiled.OutputPlacement)
{
    case SessionOutputPlacement.Device: break;               // all of it stayed on the card
    case SessionOutputPlacement.Mixed:                       // some of it did not
    case SessionOutputPlacement.Host: break;                 // none of it did
    case SessionOutputPlacement.Unknown: break;              // this backend does not say
}
```

`OutputPlacement` is free, and says where the outputs are **computed**; every tensor output but
a string one comes back on the card whatever it says. A CPU session reports `Host`. On a card, a graph run wholly by CUDA
reports `Device`; one with an unsupported operator (e.g. `Det`) reports `Mixed` or `Host`
depending on where its outputs are computed. An output the host consumed after the card computed
it counts as host memory, in the pinned arena.

For **which** nodes fell back, trace them. This turns on ONNX Runtime's profiler for every run
of the session, so keep it to diagnosis:

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

`Providers` lists each provider that ran anything, busiest first; more than one on a GPU session
*is* fallback. **Reading the trace stops the recording**: later runs are not included, and a
second read returns the same trace. `Nodes` is in execution order; inserted `MemcpyToHost` /
`MemcpyFromHost` nodes get indices above all original nodes, so treat
`NodeExecution.NodeIndex` as a name, not an order.

| what | where | cost | null / none when |
|---|---|---|---|
| `DeviceMemory.Read()` | static, the whole card, and this process's share of it | about a microsecond through DXGI; about 120 microseconds through NVML | no CUDA runtime; `ProcessBytes` alone is null where the driver attributes no memory to this process |
| `CompiledGraph.ReadArenaStatistics()` | one session's allocator | a call into the backend | the backend reports none |
| `CompiledGraph.ReadPinnedArenaStatistics()` | one session's pinned host arena | a call into the backend | the backend stages nothing (every CPU one) |
| `ComputeContext.ReadDeviceMemoryUse()` | what the context holds in its memory, against its budget | a walk over its list | never null; `LimitBytes` is null with no budget in force |
| `ComputeContext.RunStats` | every run of the context | two allocator reads per run, once switched on | `CollectRunStatistics` is off |
| `CompiledGraph.OutputPlacement` | one session | nothing | the backend does not report it (`Unknown`) |
| `CompiledGraph.ReadNodePlacement()` | one session, per node | a profiler on every run of that session | `TraceNodePlacement` is off |

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
  `ComputeContext` instead — see [One model, two devices](#one-model-two-devices).
- Do not rely on `QuickExecutionEngine` results for large tensors — values above the
  element cap are not materialized.
