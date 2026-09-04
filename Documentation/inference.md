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
    module-invoke node, so a `[Module]` output handed to it throws — where, and with
    which message, depends on the module. Concretize it first — see
    [Running a `[Module]`](#running-a-module).
- `OnnxEngine.Eval` rebuilds and recreates an ORT session on every call. For repeated
  inference, compile once with `ComputeContext` (below).
- Reference one platform backend package and it is normally found for you — no setup
  code. Only one backend is live per process. How it is discovered, and how to
  override the choice: [Backend selection](#backend-selection).

## Workflow: one-shot evaluation

`Eval` takes the output values of a graph of plain ops and runs that graph:

```csharp
using Shorokoo;
using static Shorokoo.Globals;
using static Shorokoo.NN;

var input = TensorFill(Vector(1L, 3L, 224L, 224L), TensorData([1], 0.1f));
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
- **A `[Module]`'s output, no.** `Eval` builds and runs the graph exactly as
  handed to it, so a value coming out of `ResNet50.Call(...)` still carries its
  un-lowered module-invoke node and the call throws. `ResNet50` fails while the
  ONNX model is still being built, because its layers initialize from a
  distribution computed in-graph; a module without such an initializer gets as far
  as OnnxRuntime, which rejects `ShrkCreateModule`. Either way, lower the module's
  `ComputationGraph` first — see [Running a `[Module]`](#running-a-module), which
  spells out both errors. (`ResNet50` there is from
  [`samples/RetinaNet`](../samples/RetinaNet) — a sample built on Shorokoo, not
  part of the packages.)

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
module-invoke node, in which case passing it straight to `Eval` throws. *Which*
error you get depends on what the module's parameters are initialized from:

- **An in-graph distribution** — `KaimingUniform`, `XavierUniform` and
  `RecurrentUniform` all derive their bounds from the parameter's shape, so this
  covers `Linear`, the `Conv*` layers, `MultiHeadAttention`, and anything built out
  of them (including `ResNet50`). The failure comes first, while the ONNX model is
  being built, as an `InvalidOperationException`:

  > `FastLowerRandomOps: a shrk_RandomUniform feed with in-graph distribution inputs
  > reached the ONNX fallback ... Call ToConcreteModel first; executing or exporting
  > a non-concrete model is not supported.`

- **Anything else** — no trainable parameters at all, or initializers whose
  distribution is constant (`Zeros`, `KaimingNormal`, `XavierNormal`, …). Those
  build a model, and OnnxRuntime rejects it for the module-invoke node it still
  contains:

  > `[ErrorCode:InvalidGraph] ... Error No Op registered for ShrkCreateModule ...`

The remedy is the same either way. Concretize the module's `ComputationGraph`
against the input first, then execute:

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
likewise refuses a module-kind graph up front with the same lowering hint. `Eval`
is the exception: it takes output values rather than a `ComputationGraph`, so there
is no `Kind` for it to check and a module-invoke node surfaces as one of the two raw
build/run errors above instead of a lowering hint.
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

The hyper value passed to `FromOrderedInputs` is what concretization bakes from:
shape-determining hypers (those that feed trainable-parameter shapes, like
`outFeatures`) fix the parameter shapes then and there, so pass the same value
at `Execute` time. Value-only hypers (scale factors, ε's) are read live on every
`Execute` and may vary call to call. See
[defining-models.md](defining-models.md#hyperparameter-baking) for the
distinction.

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

`Compile(ComputationGraph graph)` takes the graph and nothing else — the data goes to
the `CompiledGraph` it returns, whose `Execute(params IData[] inputs)` is the call you
repeat. `ComputeContext` also offers `Eval(...)` (the `OnnxEngine.Eval` overloads,
plus `Eval<T>(Tensor<T>)` returning a typed `TensorData<T>`),
`Execute(ComputationGraph graph, params IData[] inputs)`,
`Run(ComputationGraph graph, params NamedModelParam[] inputs)`, and
`ExecuteWithState(...)` (for models that carry state). Wherever `IData` is asked for,
`TensorData` implements it, so pass `TensorData` values directly. `Execute`, `Run` and
`CompiledGraph.Execute` return `NamedModelParam[]`; read each output with
`namedModelParam.ToTensorData()` then `AccessMemory()`. `ExecuteWithState` returns
`(NamedModelParam[] regularOutputs, ComputationGraph updatedGraph)` — feed the updated
graph to the next call. `Eval` is the exception: it returns `TensorData` (or
`TensorData[]`) directly.

## Backend selection

- Add exactly one backend package as a dependency: `Shorokoo.LinuxCPU`,
  `Shorokoo.LinuxGPU`, `Shorokoo.WinCPU`, or `Shorokoo.WinGPU`. Each brings the native
  ONNX Runtime (CPU- or CUDA-flavored) for its platform.
- With exactly one backend package referenced you normally need no setup at all:
  auto-discovery (below) finds it on the first inference call. Set the backend
  explicitly when several backends are deployed side by side and you want to override
  the choice, when you want a startup failure instead of one on the first inference
  call, or when the backend DLL is not deployed next to `Shorokoo.dll`:

  ```csharp
  using Shorokoo.Core.Inference.Abstractions;
  using Shorokoo.LinuxCPU;                                // the package you referenced

  InferenceBackend.Factory = new LinuxCpuInferenceFactory();
  ```

- Only one backend is live per process. The first factory resolved is cached and reused
  for every later call; assigning `Factory` afterwards swaps the cached factory but does
  not unload a native ONNX Runtime already bound, so to compare CPU vs GPU use separate
  processes. Every `ComputeContext` in the process shares that one backend — including a
  training rig's two, which for that reason cannot select different devices (see
  [Compute contexts](training.md#compute-contexts-mergecontext-and-runtimecontext) in the
  training guide).

### The factory types

Each backend package contains exactly one factory, in a namespace equal to the package
id. **The type name spells the device `Cpu`/`Gpu`, while the package, namespace and
assembly spell it `CPU`/`GPU`** — so `Shorokoo.WinGPU` contains
`WinGpuInferenceFactory`, *not* `WinGPUInferenceFactory`:

| package (= namespace) | factory type | fully qualified |
|---|---|---|
| `Shorokoo.LinuxCPU` | `LinuxCpuInferenceFactory` | `Shorokoo.LinuxCPU.LinuxCpuInferenceFactory` |
| `Shorokoo.LinuxGPU` | `LinuxGpuInferenceFactory` | `Shorokoo.LinuxGPU.LinuxGpuInferenceFactory` |
| `Shorokoo.WinCPU` | `WinCpuInferenceFactory` | `Shorokoo.WinCPU.WinCpuInferenceFactory` |
| `Shorokoo.WinGPU` | `WinGpuInferenceFactory` | `Shorokoo.WinGPU.WinGpuInferenceFactory` |

All four take a parameterless constructor and differ only in the execution provider
they configure: the GPU ones append the CUDA provider on device 0, the CPU ones leave
ORT on its default provider.

### Auto-discovery

If you never assign `InferenceBackend.Factory`, the first read of it resolves a backend
once and caches the result:

1. If one of the four backend assemblies is **already loaded** in the process, its
   factory is used — this avoids pulling a second native in alongside one already bound.
   The match is on assembly name alone; the OS filter in step 2 does not apply here.
2. Otherwise the folder next to `Shorokoo.dll` is probed for the known
   `Shorokoo.{Platform}.dll` files, and only those targeting the current OS count as
   candidates. Nothing else is searched: no other directory, no NuGet cache, and no
   assembly whose name is not one of those four.

When step 2 finds both the CPU and the GPU backend for the current OS, the GPU one is
used if a CUDA 12.x runtime (`libcudart.so.12` on Linux, `cudart64_12.dll` on Windows)
can be loaded, otherwise the CPU one. A single candidate is taken as-is — a lone GPU
backend is chosen even when no CUDA runtime is present.

Referencing a backend package is enough for step 2: the package copies its DLL to your
output folder, so discovery finds it whether or not your code mentions the factory type.
On a Linux sandbox that ships only `Shorokoo.LinuxCPU`, discovery picks it with no setup.

If no backend is found, the first inference call throws `InvalidOperationException`:

> `No Shorokoo inference backend is set and none was found in '<folder>'. Set one at
> startup -- e.g. InferenceBackend.Factory = new LinuxCpuInferenceFactory(); (or the
> factory from whichever Shorokoo.{WinCPU,WinGPU,LinuxCPU,LinuxGPU} package you
> reference) -- or add such a package as a dependency.`

## Debugging engine (no OnnxRuntime)

`QuickExecutionEngine` is a CPU-only interpreter used for debugging, shape inference,
and small prototypes. It only materializes values for tensors ≤ `MaxDataElements`
(default 256). Do not use it as a production inference path.

To debug the graph *structure* rather than values — e.g. when `ToConcreteArchitecture`
doesn't produce the graph you expect — snapshot the lowering stages with
`DebugRequests`: see [debugging.md](debugging.md).

## Anti-patterns

- Do not call `OnnxEngine.Eval` in a tight loop for the same graph; compile once with
  `ComputeContext`.
- Do not expect to switch from CPU to GPU mid-process; the backend is sticky once
  loaded.
- Do not rely on `QuickExecutionEngine` results for large tensors — values above the
  element cap are not materialized.
