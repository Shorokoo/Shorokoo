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
  [A tensor's lifetime](#a-tensors-lifetime-locks-and-deletion),
  [Feeding a large input without a second copy](#feeding-a-large-input-without-a-second-copy).
- A compute context keeps books on tensors and owns none; disposing it leaves every tensor
  alive — [Moving data between contexts](#moving-data-between-contexts).
- On a GPU backend a context's `DeviceMemory` settings are a budget on what it holds on the card
  and the arena settings of the sessions it compiles —
  [Device memory](#device-memory-gpu-backends). Diagnostics start at
  [What one session's arena did](#what-one-sessions-arena-did).

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
`CompiledGraph.Execute` return `NamedModelParam[]`; read each with `ToTensorData()` then
`CopyMemory<V>()`, or `ValueAt<V>(i)`, V being the CLR storage type (`float` for `float32`).
`ExecuteWithState` returns `(NamedModelParam[] regularOutputs, ComputationGraph updatedGraph)`
— feed the updated graph to the next call. `Eval` returns `TensorData` (or `TensorData[]`).

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

Giving a training rig's `runtimeContext` one is how a long `Fit` is stopped — see
[Compute contexts](training.md#compute-contexts-mergecontext-and-runtimecontext).

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
is released before the call returns, or reused for an output
([below](#a-run-that-writes-an-output-into-what-it-consumed)):

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
another device or runtime — is fed through a copy in the run's memory:

- **Consumed**: the tensor is dead and its memory released at the feed; the run consumes the
  copy. On a card, the contents go into the session's arena (except an input an output may be
  written into, which is copied onto the card first).
- **Read**: the copy is made on first read and kept — a `TensorData` held by the source,
  attached to the reading context (visible in `context.Tensors`), and reused by later shared
  reads. Writing the source (`AccessModifiableMemory` and the like) retires it; so does
  deleting or consuming the source.

### A run that writes an output into what it consumed

ONNX Runtime holds every input until the run ends, so a consumed input's memory cannot be
freed mid-run. Instead a run can write an output **into** it (output aliasing), so the output
needs no memory of its own. This happens only for outputs a lowering marks as safe; the only
such lowering is the training rig's step, which pairs each updated state field with the one it
replaces ([A step writes its state over the state it consumed](training.md#a-step-writes-its-state-over-the-state-it-consumed)).
Graphs you compile yourself never alias. A marked output is written into an input only where:

- **the run consumed that input** — `.Shared()` memory is left as it was;
- **it was fed as no other input**;
- **the output is produced in that memory**, with the input's element type and the shape the
  session settled at build time. On a GPU backend, an output fetched back to the host is not.

Otherwise nothing differs: outputs are new `TensorData` attached to the running context, and
values are the same.

### A tensor's lifetime: locks and deletion

A `TensorData` **is** its memory: one object per allocation. It records the backend that
allocated it (`AllocatingBackend`), which releases it, and where it lives: `Space` (the
device) and `Location` (device plus runtime). It does not know which contexts it is attached
to — see [Moving data between contexts](#moving-data-between-contexts).

**How a tensor ends** (disposing a context is not one of them):

| | |
|---|---|
| **Deleted** | `Delete()`, `Dispose()`, `TryDelete()` or `DeleteAsync(...)` — below. |
| **Consumed** | fed to a run as it is — or through `.TryConsume()` with nothing else reading it — which takes it when the run starts; see [Feeding a run](#feeding-a-run-consumed-shared-or-tried). |
| **Moved into an attribute** | `MoveToAttribute()`, which takes its contents — see [core-types.md](core-types.md#the-two-conversions-and-which-one-spends-its-source). |

A run's read-copy of a tensor ([above](#feeding-a-run-consumed-shared-or-tried)) ends when its
source is written or ends, or releases its copies (as a training step does for its batch).
Elements a sequence owns end with the sequence, except one a run is reading on its own
account; a sequence whose owned element a run is reading cannot be disposed.

Every access to a dead tensor — reading, feeding, `To`, `CopyTo`, `ToHost`, `Shared()`,
`TryConsume()`, `MoveToAttribute` — throws `ObjectDisposedException` saying why (for a
consumed one, which run took it). `Shape`, `DType`, `ToString()`, `IsDisposed`,
`AllocatingBackend`, `Space`, `Device` and `Location` keep working. Ending a dead tensor is a
no-op, so double disposal is harmless.

An unreferenced tensor is reclaimed by the GC through its backend; deleting only chooses
*when*. Runtime-allocated buffers (run outputs, card copies, `AllocateUninitialized` on a real
context) are native and freed at once; a tensor built from a C# array frees its native
read-copies at once and leaves the array to the GC.

**What a run holds.** A run holds a reader lock on every tensor it reads until it returns; any
number of runs may read one tensor. While locked, `Delete()` and `Dispose()` throw
`InvalidOperationException` and `TryDelete()` declines. `ToHost()`, `CopyTo(...)`,
`CopyMemory()`, `ValueAt()`, `CopyRawMemory()` and `MoveToAttribute()`'s copy take the same
lock while copying. A span from `AccessMemory()` is not covered: keep the tensor alive and
unfed while you hold one.

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
through the host. Run outputs come back on the host unless retained
(`CompiledGraph.Execute(inputs, retainOnDevice)`).

```csharp
var onCard = cuda.Compile(model).Execute([input], [true])[0].ToTensorData();  // device memory
var onHost = onCard.To(cpu);        // one copy across the bus; onCard is untouched
var same   = onHost.To(otherCpu);   // no copy: the very same object
var mine   = onHost.CopyTo(cpu);    // always a copy
```

**Where `To` or `ToHost` needs no copy, the result is `t`**: feeding it as it is consumes `t`,
and `using var h = t.ToHost();` deletes `t`. Use `CopyTo` for an independent tensor, or
`.Shared()` to keep `t` past a run.

**Attachment is bookkeeping, not ownership.** `context.Tensors` is a weak list of the context's
run outputs, what its runs read, and what `To`, `CopyTo` and `AllocateUninitialized` placed. It
never keeps a tensor alive or ends one. `context.Detach(t)` removes a tensor (refused while a
run of that context reads it) without deleting it. Disposing a context releases its sessions
and leaves every tensor. A budgeted context counts this list — see
[A context's device-memory budget](#a-contexts-device-memory-budget).

`TensorDataStruct` and `TensorDataSequence` take the same three operations. A struct comes back
as itself where nothing was copied; a sequence is copied whole if any element must be.

### Feeding a large input without a second copy

Normally a feed exists twice: your managed array and the runtime's copy.
`ComputeContext.AllocateUninitialized` gives you the runtime's buffer to fill in place:

```csharp
using var cpu = new ComputeContext(new LinuxCpuBackend());

var batch = cpu.AllocateUninitialized<float32>(new Shape(64L, 3L, 224L, 224L));
batch.WriteMemory<float>(ReadImagesInto);   // no managed array in between
```

`Shape` is a class, so write `new Shape(…)` or a `long[]`, not a `[…]` collection literal.

**Fill it through `WriteMemory`, not a bare span.** The tensor alone keeps the runtime buffer
alive, and taking a span is its last read, so

```csharp
ReadImagesInto(batch.AccessModifiableMemory<float>());   // wrong: nothing roots `batch`
```

lets the finalizer free the buffer while you write. `WriteMemory` keeps the tensor alive across
the call.

The buffer is uninitialized — fill all of it. The tensor is attached to the context like a
`CopyTo` result. A `(shape, dtype)` overload takes a runtime element type. On a CUDA context the
buffer is card memory the host cannot write (`TensorData.IsHostResident` is false); use
`CopyTo` there.

Then feed it **as it is**: the run uses the buffer in place and frees it as it returns:

```csharp
var loss = compiled.Execute(batch)[0].ToTensorData();
// batch is consumed: reading it throws, and its buffer went back to the allocator with the run
```

A `.Shared()` feed is instead freed when you let go of it (for a per-step batch, at the next
collection). Consuming does not let the run reuse the input for intermediates — ONNX Runtime
never does that — only for [aliased outputs](#a-run-that-writes-an-output-into-what-it-consumed).

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
var onCard = cuda.Execute(graph, input);           // the same graph, the same input, the card
```

Both run the same model, one C# build; each context compiles its own session. `ComputeContext.Backend` and `CompiledGraph.Backend` name
the backend. Tensors you build (allocating backend `HostBackend.Instance`) are tied to no
runtime, so building and exporting needs none and either context accepts them. A tensor not
readable in place is copied per run when consumed, or once per (tensor, runtime) when fed
`.Shared()`, until written ([Feeding a run](#feeding-a-run-consumed-shared-or-tried)). A
tensor left on the card (`TensorData.IsHostResident` false, e.g. by a
[resident training run](training.md#keeping-training-state-on-the-device)) crosses via the
host.

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

1. **An installed copy, where one matches exactly**: every file of the release present, each with
   the SHA-256 its PyPI wheel records. On Windows cuDNN is looked for on `PATH`, in `%CUDNN_PATH%`
   and under `%ProgramFiles%\NVIDIA\CUDNN`, and cuBLAS on `PATH`, in `%CUDA_PATH%` and in the
   CUDA 13 toolkits under `%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA`; on Linux, on
   `LD_LIBRARY_PATH`, in `$CUDNN_PATH`, `$CUDA_PATH` and `$CUDA_HOME`, in the CUDA 13 toolkits
   under `/usr/local` and in the system's library folders. Its files are hard-linked into the
   cache where the volume allows, and copied otherwise. Any other copy — another release, a build
   for another CUDA major, a file that differs — is ignored, never mixed in.
2. **A provisioned PyTorch CUDA environment's own copies**, which are that same release byte for
   byte (see below), when PyTorch is the backend that gets there first.
3. **Otherwise the release's wheel from PyPI**, checked against the SHA-256 the pin records.

With none of them available — offline, nothing installed that matches — the first CUDA session
fails with an `InvalidOperationException` naming the library, where it looked and the size of
the download. To run offline, fill the cache once while online, copy the folders from a machine
that has them, or install exactly the pinned release. `CudaLibraries.Prepare()` (in
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

The other NVIDIA libraries both stacks load — the CUDA runtime, cuFFT, nvrtc, nvJitLink — are
each backend's own: the ONNX Runtime backend's come from the machine's CUDA 13 runtime, PyTorch's
from its environment. Two copies of these run side by side: they call one another only through
public, versioned entry points.

### Device memory (GPU backends)

ONNX Runtime allocates device memory from an arena that grows in blocks and, unless asked to
shrink, never returns them. The *extend strategy* sets each block's size:

- **`NextPowerOfTwo`** (ORT's default) grows by at least the arena's current size — good for
  unpredictable allocation sizes, wasteful once sizes settle.
- **`SameAsRequested`** grows by exactly the request — tight when sizes settle, but a session
  whose input shapes keep growing strands every region it outgrows.

`ArenaExtend` defaults to `Auto`: `SameAsRequested`, except for a session Shorokoo knows is
fed differing shapes — a training rig's shared fallback step, used once it has seen more
distinct input shapes than its per-shape limit — which gets `NextPowerOfTwo`. **If your own
session is fed growing shapes, set `NextPowerOfTwo`.** Settings apply to the sessions the
context compiles; `CompiledGraph.DeviceMemory` reports what a graph got:

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

- A training step's arena can take one large extra block after step 0 and keep it; on one
  large model it grew 1.75–1.9x between steps 0 and 1 under either strategy. An arena limit
  near the step's real use prevents that block, and the steps still fit in what the arena
  already holds: size a context budget as what the step uses plus what the rig keeps on the
  card.
- `ShrinkArenaAfterRun` costs a synchronizing allocation every step; outside a budget, use it
  only when the card is shared. On the CPU backend it shrinks the host arena the run's
  intermediates live in, and the next run takes those blocks from the host again. ORT rejects it where the device has no arena (e.g.
  `ORT_DISABLE_ARENA`), failing the run, so try it on a short run first.
- `LimitBytes` is hard: exceeding work is refused or fails with ORT's `BFCArena ... Failed to
  allocate memory for requested buffer`, so a figure set too low fails work that would fit.
- `ArenaExtend` is read **when a session is built** (the first call, or a rig's first
  `TrainStep` per input shape); `CompiledGraph.DeviceMemory` reports the resolved strategy,
  not `Auto`. Context settings are `init`-only. `CompiledGraph.Execute` / `Run` can override
  `ShrinkArenaAfterRun` per call on an unbudgeted context; the context's one-shot entry points
  and a rig's `TrainStep` use the context's.
- **Scope is the context, never the process.** Each session keeps its arena settings for life;
  two contexts may differ and never affect each other's sessions. To use different settings,
  compile on another context — e.g. one for a training loop, one for variable-shape inference.
  Tensors a context places on the card come from a separate per-card, per-runtime allocator
  held for the process's life; the budget counts them by attachment.

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
  optimizer state, every session's arena, the CUDA context and its libraries' workspaces — and
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
that run's arena. It is per context, not per arena or process:

```csharp
using var ctx = new ComputeContext(gpu)
{
    DeviceMemory = new DeviceMemorySettings { LimitBytes = 2L * 1024 * 1024 * 1024 },
};

var onCard = big.CopyTo(ctx);                     // on the card, and on ctx's books
var use = ctx.ReadDeviceMemoryUse();
Console.WriteLine($"{use.AttachedBytes} of {use.LimitBytes} bytes attached, {use.AvailableBytes} left");
```

**What it counts.** Tensors placed by `To`, `CopyTo` or `AllocateUninitialized`, read or
copied there by the context's runs, or left there as outputs (`Execute(inputs, retainOnDevice)`).
`ReadDeviceMemoryUse()` reports `AttachedBytes`, `AttachedTensors` and `LimitBytes` (`null`
with no budget, or for a host context). A tensor on two contexts counts on both; dead,
collected or `Detach`ed tensors drop out.

**A transfer it cannot take is refused before allocating** — `To`, `CopyTo`,
`AllocateUninitialized`, a run's card copy of a tensor it cannot read in place, and a `To` that
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

**A run's arena gets what the context leaves it.** A session's `gpu_mem_limit` is the budget
less the *discount*: what the context holds on the card outside the arena during the run —
attached tensors, and those the run reads or copies there (for a tensor another runtime holds
on the same card, both it and its copy). A tensor already on the card is read in place and
never enters the arena. A host tensor fed to a card run:

- `.Shared()`: copied onto the card before the run, kept for later reads, counted in the
  discount;
- consumed: copied into the arena, counting against the session's limit instead, so a loop
  feeding fresh host batches keeps its session (except an input an output may be written into,
  copied like a shared one).

A run whose discount leaves its arena nothing, or less than the consumed inputs it must copy in, is refused before taking anything; one whose arena
needs more than it was left fails with ORT's `BFCArena` error.

**When a session is rebuilt.** The limit is fixed when a session is built, and building is
costly for large graphs, so a session is built with the budget less the discount, rounded up
to the next sixty-fourth of the budget, and rebuilt (before taking anything) only when the
discount grows past that. So:

- the limit only comes down — at most sixty-four times over a compiled graph's life, plus once
  each time the remainder halves within its last sixty-fourth; a loop holding the same things
  never rebuilds;
- releasing what the context held does not restore room; compile the graph again;
- `CompiledGraph.DeviceMemory.LimitBytes` is the current limit, and `ReadArenaStatistics()` and
  `ReadNodePlacement()` restart for a rebuilt session;
- a rig's step can rebuild in its first steps as state arrives on the card.

An output retained in a session's own arena and fed back to the same graph is inside its limit,
not discounted. An output [written into consumed memory](#a-run-that-writes-an-output-into-what-it-consumed)
is counted once, where that memory is: in later discounts if it was outside the arena, inside
the arena if it was that session's own earlier output.

**One at a time.** Under a budget the context's runs, transfers onto it and compiles on it are
serialized, and every run returns its arena's unused blocks as it ends, whatever
`ShrinkArenaAfterRun` says. None of this applies without `LimitBytes`.

What the budget does *not* count is in
[Known limitations](limitations.md#a-device-memory-budget-counts-tensors-not-arenas).

### What one session's arena did

To read **one session's own allocator** (CPU or GPU), ask the compiled graph:

```csharp
using Shorokoo.Core.Backends;
using Shorokoo.Runtime;

var compiled = ctx.Compile(graph);
compiled.Execute(inputs);

if (compiled.ReadArenaStatistics() is { } arena)
    Console.WriteLine($"{arena.InUseBytes} in use, {arena.MaxInUseBytes} at its highest, "
                    + $"{arena.TotalAllocatedBytes} taken from the device");
```

`ArenaStatistics` has ten figures: `InUseBytes`, `RequestedInUseBytes`, `MaxInUseBytes`,
`MaxAllocSizeBytes`, `TotalAllocatedBytes`, `LimitBytes` (the session's arena limit; `-1` with
no budget), `AllocationCount`, `ArenaExtensionCount`, `ArenaShrinkageCount` and
`ReserveCount`. It is `null` on a backend that reports none.

- `RequestedInUseBytes` is the part of `InUseBytes` the callers asked for; the rest is what the
  arena added to round each allocation up to its block sizes. A backend whose allocator does not
  report what was requested (JAX) reports `InUseBytes` here, rounding included.
- `TotalAllocatedBytes` is not a bound on card usage — it can exceed the card's capacity; use
  `DeviceMemory.Read()` for that.
- `MaxInUseBytes` is a lifetime high-water mark; it cannot be reset and shrinkage does not
  lower it. For per-run figures use [`RunStats`](#per-run-statistics-on-the-context).
- A session's weights live in this arena, so it is non-zero before the first run, and the first
  run's peak includes them. `ReserveCount` and `ArenaExtensionCount` do not compare across
  devices.

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
                + $"{stats.ArenaExtensionCount} arena extensions");

foreach (var run in stats.RecentRuns.TakeLast(5))
    Console.WriteLine($"run {run.RunNumber}: {run.PeakBytes} ({run.PeakKind}), "
                    + $"arena stood at {run.PriorPeakBytes} before it");
```

`RunStats` covers every run of the context across **all** its sessions.

- **Aggregates are exact; per-run detail is bounded.** `PeakBytes`, `RunCount`,
  `AllocationCount`, `ArenaExtensionCount`, `ArenaShrinkageCount`, `LargestAllocationBytes` and
  `ArenaBytes` cover the whole history. `RecentRuns` keeps the last
  `DiagnosticSettings.RecentRunCapacity` records (1000 by default; zero keeps none).
- **`PeakKind`**: a run that raised the arena's high-water mark is
  `MemoryFigureKind.Measured`; one that stayed below an earlier mark is
  `MemoryFigureKind.UpperBound` (it used at most that).
- **`PriorPeakBytes`** is the mark the run found. On the first run it is the weights, so
  `PeakBytes - PriorPeakBytes` is the run's own use; later it is how far the run exceeded the
  record (zero for `UpperBound` runs).
- `PeakBytes` is the largest mark of any one arena, not a sum across sessions.
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

`OutputPlacement` is free. A CPU session reports `Host`. On a card, a graph run wholly by CUDA
reports `Device`; one with an unsupported operator (e.g. `Det`) reports `Mixed` or `Host`
depending on where its outputs end up. An output the host consumed after the card computed it
counts as host memory, in the pinned arena.

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
