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
  code. Only one backend is live per process, and two of them for one OS is refused rather
  than guessed at. How it is discovered, and how to override the choice:
  [Backend selection](#backend-selection).
- Which device the work will run on is invisible at the call site but answerable:
  `ComputeContext.Backend` and `InferenceBackend.Describe()` name it, and
  `InferenceBackend.RequireDevice(...)` refuses to start on the wrong one —
  [Which device am I on?](#which-device-am-i-on). To run part of the work on another
  device, [One model, two devices](#one-model-two-devices).
- On a GPU backend the CUDA arena is configured on the `ComputeContext` — `DeviceMemory` for
  the sessions it compiles, `RunSettings` for what its runs do — while the static `DeviceMemory`
  class reports how much of the card is gone. The arena strategy is chosen per session, so a
  training loop does not end up holding far more of the card than it uses and a variable-shape
  inference path still gets the strategy that suits it:
  [Device memory](#device-memory-gpu-backends).

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
  explicitly to name which one you mean when a deployment holds more than one (which
  discovery otherwise refuses), when you want a startup failure instead of one on the
  first inference call, or when the backend DLL is not deployed next to `Shorokoo.dll`:

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
- **Exactly one** is the rule, not a recommendation: a deployment carrying two backends for
  the same OS is refused rather than resolved by guesswork — see
  [Auto-discovery](#auto-discovery). To run part of the work on another device, see
  [One model, two devices](#one-model-two-devices).
- Which backend you ended up on is a question you can ask — `InferenceBackend.Describe()`,
  or `ComputeContext.Backend` where the work is submitted. See
  [Which device am I on?](#which-device-am-i-on).

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
   Only assemblies targeting the running OS count, as in step 2, and only those that
   actually expose a factory; anything else falls through to step 2.
2. Otherwise the folder next to `Shorokoo.dll` is probed for the known
   `Shorokoo.{Platform}.dll` files, and only those targeting the current OS count as
   candidates. Nothing else is searched: no other directory, no NuGet cache, and no
   assembly whose name is not one of those four.

A single candidate is taken as-is — a lone GPU backend is chosen even when no CUDA
runtime is present. **Two or more are refused**, in either step, with an
`InvalidOperationException` naming them. From the folder probe (step 1 says `already loaded
in this process` in place of `deployed in '<folder>'`, and refuses only backends that
actually expose a factory):

> `Several Shorokoo inference backends are deployed in '<folder>': Shorokoo.WinCPU (CPU),
> Shorokoo.WinGPU (CUDA). Only one can be live in a process, and each package brings its
> own native ONNX Runtime, so a build carrying both is ambiguous. Reference exactly one
> backend package -- keeping model code in a library that references no backend, and one
> executable per device -- or assign InferenceBackend.Factory before the first inference
> call to say which of these you mean.`

Discovery does not resolve that by looking for a CUDA runtime and preferring the GPU. A
deployment holding both packages has already had their native ONNX Runtimes collide —
each ships `libonnxruntime.so` (`onnxruntime.dll`) at the same path, so only one of them is
deployed and which one is NuGet's conflict resolution to decide — and the managed DLL that
discovery would pick says nothing about the native that is actually there. Backends for
*different* OSes are not ambiguous and are not refused: only those targeting the running one
are candidates, so a Windows backend alongside a Linux one is no ambiguity at all. Carrying
all four, on the other hand, is two for whichever OS you run on — and refused on both.

Mind that step 1 settles it first. If exactly one backend assembly is already loaded when the
first inference call happens — which naming its factory type anywhere in a method your program
runs is enough to cause — that one wins and the folder is never probed. The refusal is what
happens when the *deployment* is left to make the choice, not a guarantee that an ambiguous
build cannot run.

The usual way to arrive at an ambiguous deployment is a shared library that references a
backend, which flows to everything referencing it;
[One model, two devices](#one-model-two-devices) is the layout that avoids it.

Referencing a backend package is enough for step 2: the package copies its DLL to your
output folder, so discovery finds it whether or not your code mentions the factory type.
On a Linux sandbox that ships only `Shorokoo.LinuxCPU`, discovery picks it with no setup.

If no backend is found, the first inference call throws `InvalidOperationException`:

> `No Shorokoo inference backend is set and none was found in '<folder>'. Set one at
> startup -- e.g. InferenceBackend.Factory = new LinuxCpuInferenceFactory(); (or the
> factory from whichever Shorokoo.{WinCPU,WinGPU,LinuxCPU,LinuxGPU} package you
> reference) -- or add such a package as a dependency.`

### Which device am I on?

Nothing at a call site says which device the work will go to — `Compile(...)` and
`Execute(...)` look the same on a CPU build and a GPU one. Ask instead:

```csharp
using Shorokoo.Core.Inference.Abstractions;

Console.WriteLine(InferenceBackend.Describe());       // Shorokoo.WinGPU (CUDA device 0)
Console.WriteLine(ComputeContext.Default.Backend);    // the same, at the point work is submitted
```

`BackendDescription` carries the `Name` of the supplying assembly, the `Device`
(`ComputeDevice.Cpu`, `Cuda`, or `Other` for a backend you wrote against a third execution
provider), and the `CudaDeviceId` a CUDA backend allocates on (null on anything else). Record it in a run's log: a training run that cannot say which
device produced its numbers has lost something it cannot reconstruct later.

Two related entry points:

- `InferenceBackend.Current` is the live backend **or null**, and — unlike `Factory` and
  `Describe()` — reading it does not resolve one. Use it to tell "nothing chosen yet" from
  "already bound" without settling the question by asking it.
- `InferenceBackend.RequireDevice(ComputeDevice.Cpu)` throws unless the live backend is on
  that device. Put it at the top of a program whose correctness depends on where it runs —
  a check that must not contend with a training run holding the card — and it fails at
  startup, with the live backend named, instead of quietly sharing the GPU:

  ```csharp
  InferenceBackend.RequireDevice(ComputeDevice.Cpu);   // before any inference call
  ```

### One model, two devices

Only one backend is live per process, so running part of the work on the CPU while the rest
uses the GPU means **two processes**. It does not mean two copies of the model.

Put the model and its `[Module]`s in a class library that references no backend at all —
`Shorokoo` (or `Shorokoo.Core` + `Shorokoo.Modules`) carries no ONNX Runtime dependency, so
such a library compiles and is device-neutral. Then give each device a thin executable that
references that library plus one backend:

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

The model is compiled once, into one assembly, and both hosts run *that* — so a check tests
the model the run is training, not a second compilation of its source. `[Module]` classes the
check needs live beside the model they test, in the library.

What breaks this is a backend reference in the shared library — a `PackageReference` or a
`ProjectReference` to an executable that carries one. Either flows into the referencing
project's output folder, two backends end up deployed, and
[auto-discovery](#auto-discovery) refuses to guess between them. Keep the backend in the
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
package, so treat these as indicative of the shape, not as your machine's numbers — what settles
your case is `DeviceMemory.Sample()` around your own run.

Neither column wins outright, and which one wins is decided by something Shorokoo knows about each
session: whether its allocation sizes settle. A training run feeds one input shape to one compiled
step for its whole length — the first row — and there exact-size extension holds about 1.3–1.45x
less. On the card that prompted this, a step whose first step showed 12,877 MiB ended up with the
arena holding all 24,563 MiB — the whole of a 24 GiB card — and a smaller batch of the same model
settled at roughly 1.8x what its steps used. Two allocation sizes still favour exact-size extension
(row 2, by 1.2–1.6x depending on the run); it is once several are in play that ORT's doubling holds
less, by about 1.1x, or ties (rows 3 to 5). The last two rows are the other end: input shapes that
grow without settling, where each outgrown region is stranded, the doubling holds around 1.5x less
and — on a card with no room to spare — fits where exact-size extension does not.

So `ArenaExtend` defaults to `Auto`, which is not one of ORT's values but the choice between them,
made per session: a **training step** gets `SameAsRequested`, because its shapes are fixed when the
step is compiled and repeat for the life of the run; **every other session** gets ORT's
`NextPowerOfTwo`, because it may be handed a larger input on any call. Name a strategy to decide it
yourself — for the sessions that context compiles, and no others — and read back what a graph
actually got from `CompiledGraph.DeviceMemory`:

```csharp
using Shorokoo.Core.Inference.Abstractions;
using Shorokoo.Runtime;

var ctx = new ComputeContext
{
    DeviceMemory = new DeviceMemorySettings
    {
        ArenaExtend = ArenaExtendStrategy.NextPowerOfTwo,   // ORT's doubling, back again
        LimitBytes = 16L * 1024 * 1024 * 1024,              // cap this session's arena at 16 GiB
    },
    RunSettings = new RunSettings { ShrinkArenaAfterRun = true },  // hand unused blocks back each step
};

var compiled = ctx.Compile(graph);
compiled.Execute(inputs);                                          // the context's run settings
compiled.Execute(inputs, new RunSettings { ShrinkArenaAfterRun = false });   // this run only

Console.WriteLine(compiled.DeviceMemory.ArenaExtend);              // what this session was built with
```

| setting | on | ORT option | default | read |
|---|---|---|---|---|
| `LimitBytes` | `DeviceMemorySettings` | `gpu_mem_limit` | `null` — no cap | when a session is created |
| `ArenaExtend` | `DeviceMemorySettings` | `arena_extend_strategy` | `Auto` — `SameAsRequested` for a training step, ORT's `NextPowerOfTwo` elsewhere | when a session is created |
| `ShrinkArenaAfterRun` | `RunSettings` | `memory.enable_memory_arena_shrinkage` | `false` | on every run |

The other two are unset by default for their own reasons. `ShrinkArenaAfterRun` costs a
synchronizing device allocation on every step to re-take what it handed back, so it is worth it
only when the card is shared with something that needs the room between steps. It is also the one
setting that can fail a run rather than degrade it: ORT rejects the request where the device it
names has no arena allocator registered — an arena disabled through `ORT_DISABLE_ARENA`, say — so
turn it on with a short run before a long one. `LimitBytes` is a
budget, not a hint: a step that needs more than it fails with ORT's `BFCArena ... Failed to
allocate memory for requested buffer` rather than eating the rest of the device, so a figure set
too low fails a run that would have fitted.

Note that the budget caps **each session's** arena, not the process. ORT gives a session its own
CUDA arena, so a process holding a compiled graph and a training rig at once can hold the limit
more than once over; read it as the ceiling on any one session.

The first two are read **when a session is built** — the first inference call, or a training rig's
first `TrainStep` for a given input shape — so the context has to carry them before the graph is
compiled on it; a graph already compiled keeps what it was built with, which is why
`CompiledGraph.DeviceMemory` reports the settled strategy rather than `Auto`. `ShrinkArenaAfterRun`
is read on every run, so it takes effect on sessions already compiled and a single call can
override it.

The same class reports what the card is doing:

```csharp
using var run = rig.BeginResidentRun(checkpoint);
for (int step = 0; step < steps; step++)
{
    run.Step(input, target);
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

**Scope: the session and the run, never the process.** That is ONNX Runtime's own shape, not a
convention layered on top. ORT gives each session its own arena and reads `DeviceMemorySettings`
once while building it, after which the session keeps them for life — so a context configures the
sessions it compiles from then on, two contexts may differ, and a graph already compiled is
untouched by any later change. To run something under a different budget, compile it on a context
that carries one. `RunSettings` ORT reads off the run instead, so those are settled per call and a
compiled graph can shrink its arena on one run and not the next.

Where one process must serve both a training loop and a variable-shape inference path, give them a
context each rather than picking one arena strategy for both.

The readings are the exception, and they are readings rather than settings: `Read()` and `Sample()`
go to whichever CUDA device is current for the calling thread — device 0, because that is what the
shipped GPU backends use — and `PeakUsedBytes` is one process's record of its own run.

## Debugging engine (no OnnxRuntime)

```csharp
using Shorokoo.Core.Inference;   // QuickExecutionEngine
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
