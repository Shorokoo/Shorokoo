# Running models (inference)

Related: [core-types.md](core-types.md) · [defining-models.md](defining-models.md) ·
[onnx-and-weights.md](onnx-and-weights.md)

## Facts

- `OnnxEngine.Eval(...)` is the simplest way to get values. It builds an ONNX model
  from the graph, runs it once via OnnxRuntime, and returns `TensorData`.
  - `TensorData Eval(IValue output)`
  - `TensorData[] Eval(IValue[] outputs)`
  - `TensorData[] Eval(IValue a, IValue b, params IValue[] more)`
  - `Eval` runs a graph of plain ops. A `[Module]` output may still carry an
    un-lowered module-invoke node; if `Eval` throws
    `No Op registered for ShrkCreateModule`, concretize it first — see
    [Running a `[Module]`](#running-a-module).
- `OnnxEngine.Eval` rebuilds and recreates an ORT session on every call. For repeated
  inference, compile once with `ComputeContext` (below).
- Backend selection is implicit: whichever platform backend assembly is loaded
  determines the execution provider (CPU vs CUDA). In a Linux sandbox this is
  `Shorokoo.LinuxCPU`. Only one backend can be loaded per process.

## Workflow: one-shot evaluation

(`ResNet50` here is from [`samples/RetinaNet`](../samples/RetinaNet) — it is a
sample built on Shorokoo, not part of the packages; substitute any `[Module]`.)

```csharp
using Shorokoo;
using static Shorokoo.Globals;

var input  = TensorFill(Vector(1L, 3L, 224L, 224L), TensorData([1], 0.1f));
var logits = ResNet50.Call(
    numClasses: Scalar(1000L), bnMomentum: Scalar(0.9f), bnEps: Scalar(1e-5f),
    includeTop: Scalar(true), applySoftmax: Scalar(true), inputs: input);

TensorData result = OnnxEngine.Eval(logits);

// Read the numbers out (see core-types.md):
ReadOnlySpan<float> values = ((TensorData<float32>)result).AccessMemory();
```

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
module-invoke node, in which case passing it straight to `Eval` throws:

> `[ErrorCode:InvalidGraph] ... Error No Op registered for ShrkCreateModule ...`

Concretize the module's `ComputationGraph` against the input first, then execute:

```csharp
using Shorokoo;
using Shorokoo.Graph;     // Specialize / ToConcreteArchitecture / FromOrderedInputs / ToConcreteModel
using Shorokoo.Runtime;   // ComputeContext
using static Shorokoo.Globals;

var input    = TensorData([4L], 1f, 2f, 3f, 4f);   // the actual input data
var graph    = MyLayer.ComputationGraph;            // generated FastComputationGraph
var concrete = graph
    .ToConcreteArchitecture(graph.FromOrderedInputs([input]))
    .ToConcreteModel();

var results = ComputeContext.Default.Execute(concrete, input);   // params IData[]
float[] values = results[0].ToTensorData().As<float32>().AccessMemory<float>().ToArray();
```

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

`Specialize` matches values to inputs **by name** (against
`InputUniqueNames`); names with no matching input are ignored. It returns a copy
and never mutates the original graph, exactly like `ToConcreteArchitecture`.
This works for any input, not just hypers — but baking a runtime input is
usually not what you want.

## Workflow: compile once, run many (repeated inference)

`ComputeContext` builds the ORT session once and reuses it.

```csharp
var ctx      = new ComputeContext();
var compiled = ctx.Compile(graph);                 // graph: FastComputationGraph
var r1 = compiled.Execute(inputData1);             // params IData[]
var r2 = compiled.Execute(inputData2);             // reuses the session
```

`ComputeContext` also offers `Eval`, `Execute(graph, inputs)`, `Run(graph, params)`,
and `ExecuteWithState(...)` (for models that carry state). `Execute`/`Compile` take
`params IData[]` inputs; `TensorData` implements `IData`, so pass `TensorData` values
directly. They return `NamedModelParam[]`; read each output with
`namedModelParam.ToTensorData()` then `AccessMemory()`.

## Backend selection

- Add exactly one backend package as a dependency: `Shorokoo.LinuxCPU`,
  `Shorokoo.LinuxGPU`, `Shorokoo.WinCPU`, or `Shorokoo.WinGPU`. Each brings the native
  ONNX Runtime (CPU- or CUDA-flavored) for its platform.
- Recommended: set the backend explicitly at startup, before the first inference call:

  ```csharp
  using Shorokoo.Core.Inference.Abstractions;
  using Shorokoo.LinuxCPU;                                // the package you referenced

  InferenceBackend.Factory = new LinuxCpuInferenceFactory();
  ```

- If you don't set one, the first inference call auto-discovers a backend by looking
  **only** in the folder next to `Shorokoo.dll` for the known `Shorokoo.{Platform}`
  DLLs. When both a CPU and a GPU backend for the current OS are deployed there, the
  GPU one is used if a CUDA 12.x runtime is present, otherwise the CPU one. On a Linux
  sandbox that ships only `Shorokoo.LinuxCPU`, discovery picks it with no setup.
- Only one backend is live per process. To compare CPU vs GPU, use separate processes.

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
