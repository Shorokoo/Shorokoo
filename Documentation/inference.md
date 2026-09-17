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
  `ComputeContext.Backend` and `InferenceBackend.Describe()` name it, and
  `InferenceBackend.RequireDevice(...)` refuses to start on the wrong one —
  [Which device am I on?](#which-device-am-i-on).
- On a GPU backend the CUDA arena is configured on the `ComputeContext` — `DeviceMemory` for
  the sessions it compiles, `RunSettings` for what its runs do — while the separate static
  `DeviceMemory` class reports how much of the card is gone. The arena strategy departs from
  exact-size extension only for a session Shorokoo knows is reused across differing shapes, so a
  long training loop does not end up holding far more of the card than it uses:
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

- Add a backend package as a dependency: `Shorokoo.LinuxCPU`, `Shorokoo.LinuxGPU`,
  `Shorokoo.WinCPU`, or `Shorokoo.WinGPU`. Each brings the native ONNX Runtime (CPU- or
  CUDA-flavored) for its platform. One is enough; a program that names its backends may
  reference several, or load them at runtime and reference none — see
  [Loading a backend at runtime](#loading-a-backend-at-runtime).
- With exactly one backend package referenced you normally need no setup at all:
  auto-discovery (below) finds it on the first inference call. Set the backend
  explicitly to name which one you mean when a deployment holds more than one (which
  discovery otherwise refuses), when you want a startup failure instead of one on the
  first inference call, or when the backend DLL is not deployed next to `Shorokoo.dll`:

  ```csharp
  using Shorokoo.Core.Inference.Abstractions;
  using Shorokoo.LinuxCPU;                                // the package you referenced

  InferenceBackend.Default = new LinuxCpuBackend();
  ```

- `InferenceBackend.Default` is the **default** backend: the one a `ComputeContext` that
  names no backend of its own runs on, and the one a tensor is built for when it is fed
  without a context naming another. The first backend resolved is cached and reused;
  assigning `Default` afterwards swaps it but does not unload a native ONNX Runtime already
  bound, and does not reach a `ComputeContext.Default` that has already resolved — so assign
  it at startup, before anything runs.
- A `ComputeContext` constructed with a backend runs there instead, and two contexts may
  name different backends — that is how one process uses two devices. See
  [One model, two devices](#one-model-two-devices).
- **Exactly one deployed** is still the rule for *discovery*: a deployment carrying two
  backends for the same OS and naming neither is refused rather than resolved by
  guesswork — see [Auto-discovery](#auto-discovery). It is a rule about silence, not a
  limit on how many can run.
- Which backend you ended up on is a question you can ask — `InferenceBackend.Describe()`,
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

All four implement `IShorokooInferenceBackend`, take a parameterless constructor, and
differ only in the execution provider
they configure: the GPU ones append the CUDA provider on device 0, the CPU ones leave
ORT on its default provider.

### Auto-discovery

If you never assign `InferenceBackend.Default`, the first read of it resolves a backend
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

> `Several Shorokoo inference backends are deployed in '<folder>': Shorokoo.WinCPU (CPU),
> Shorokoo.WinGPU (CUDA). Discovery picks the backend for a program that named none, and
> this deployment gives it no way to choose. Say which you mean: assign
> InferenceBackend.Default before the first inference call to make one of them the default.
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
first inference call happens — which naming its backend type anywhere in a method your program
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
output folder, so discovery finds it whether or not your code mentions the factory type.
On a Linux sandbox that ships only `Shorokoo.LinuxCPU`, discovery picks it with no setup.

If no backend is found, the first inference call throws `InvalidOperationException`:

> `No Shorokoo inference backend is set and none was found in '<folder>'. Set one at
> startup -- e.g. InferenceBackend.Default = new LinuxCpuBackend(); (or the
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

### Loading a backend at runtime

A program need not reference a backend at compile time at all. `BackendPackage.TryLoad`
takes a path and hands back a factory, and a backend that does not fit the machine comes
back as a reason rather than an exception — so one executable can carry backends for
several platforms and pick at startup.

```csharp
using Shorokoo.Core.Inference.Abstractions;

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

`BackendPackage.Probe` answers the same question without loading anything: it reads the
backend's own declaration out of the file's metadata, so a backend for another operating
system, another architecture, or one whose native libraries are not deployed with it is
refused before any native code is touched. `BackendProbe.Reason` says which it was
(`WrongOperatingSystem`, `MissingNative`, `MissingCudaRuntime`, …) and `Detail` names the
file or library that is wrong.

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

### Moving data between contexts

A `TensorData` belongs to a compute context — `Context`, null for the framework's own host
memory — and says whether it owns its bytes, in `OwnsMemory`. `Space` says where those bytes
are: host memory, or a particular CUDA device.

Three operations move a tensor between contexts. They differ in what happens to the
ownership rather than to the bytes:

| | Same memory space | Different memory space |
|---|---|---|
| `TransferTo` | nothing is copied; ownership moves to the result | the bytes are copied and the source is spent — owner only |
| `CopyTo` | an independent copy, owned by the result | the same |
| `GiveAccessTo` | a reader that owns nothing; the source keeps what it had | refused — use `CopyTo` |

Whether the bytes move is decided by the space **and** by whether the two contexts share a
native runtime. Host memory is host memory whoever allocated it, so any two host contexts
pass a tensor between them without copying. A device allocation is not: it is meaningful
only to the runtime that made it, so two CUDA contexts share one without copying when they
are the same backend, or two backends over one loaded runtime — and copy through the host
when they are separate runtimes, which is what `IsolatedBackend` produces. A run's outputs
come back on the host unless you asked for them to be retained
(`CompiledGraph.Execute(inputs, retainOnDevice)`), so this arises for a tensor you put on
the card or kept there deliberately.

```csharp
var onCard  = cuda.Execute(model, input, [true])[0].ToTensorData();  // retained: device memory
var onHost  = onCard.TransferTo(cpu);                        // one copy across the bus
var shared  = onHost.TransferTo(otherCpu);                   // no copy: same space
```

Disposing a context releases every tensor it still owns, and reading one afterwards throws
rather than reading freed memory. Bytes that were transferred away are not touched — they
belong to the context that took them. A context constructed with `detachesOutputs: true`
hands its results out belonging to nobody, so they outlive it; `ComputeContext.Default` is
built that way.

`TensorDataStruct` and `TensorDataSequence` carry a context and take the same three
operations, recursing into what they own.

### One model, two devices

One process can run one model on the CPU and on the card. A `ComputeContext` constructed
with a backend compiles and runs there, and two contexts may name different backends:

```csharp
using Shorokoo.Core.Inference.Abstractions;
using Shorokoo.LinuxCPU;
using Shorokoo.LinuxGPU;

var cpu  = new ComputeContext(new LinuxCpuBackend());
var cuda = new ComputeContext(new LinuxGpuBackend());

var onHost = cpu.Execute(graph, input);     // the host
var onCard = cuda.Execute(graph, input);    // the same graph, the same input, the card
```

The model is compiled once, into one assembly, and both contexts run *that* — so a check on
the CPU tests the model the GPU run is training, not a second compilation of its source.
`ComputeContext.Backend` says which device each one will use, and a `CompiledGraph` carries
the backend it was built on in `CompiledGraph.Backend`.

**Tensors are not tied to a backend.** A `TensorData` you build holds managed bytes and no
backend at all, so building a model and exporting it needs no runtime; a backend enters only
when the tensor is fed to one, and then either context accepts it — a session hands what it
is fed to its own runtime, building it there if it does not have it yet. Those built values
are cached per backend, so the cost is one copy per (tensor, backend) pair rather than per
run. It is possible only for data the host can read: a value an execution provider kept in
its own memory (`TensorData.IsHostResident` is false, which a
[resident training run](training.md#keeping-training-state-on-the-device) produces) cannot
cross, and says so rather than being read as a host address.

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
the *first* read of `InferenceBackend.Default` that refuses, which may be some framework
call you did not write. Constructing a backend does not settle the question; assigning does:

```csharp
InferenceBackend.Default = new LinuxCpuBackend();   // startup, before any inference call

var cpu  = new ComputeContext(InferenceBackend.Default);
var cuda = new ComputeContext(new LinuxGpuBackend());
```

**Two runtimes.** Two ONNX Runtime *builds*, or two versions, in one process — a vendor
build beside the stock one, say. Give each native a folder of its own and load the second
with `IsolatedBackend`:

```xml
<PackageReference Include="Microsoft.ML.OnnxRuntime.Managed" Version="1.26.0" />
<PackageReference Include="Microsoft.ML.OnnxRuntime" Version="1.26.0"
                  ExcludeAssets="all" GeneratePathProperty="true" />
<PackageReference Include="Microsoft.ML.OnnxRuntime.Gpu.Linux" Version="1.26.0"
                  ExcludeAssets="all" GeneratePathProperty="true" />

<ShorokooBackendNatives BackendId="cpu"
  Include="$(PkgMicrosoft_ML_OnnxRuntime)\runtimes\linux-x64\native\*" />
<ShorokooBackendNatives BackendId="cuda"
  Include="$(PkgMicrosoft_ML_OnnxRuntime_Gpu_Linux)\runtimes\linux-x64\native\*" />
```

`ShorokooBackendNatives` items are read by a target the `Shorokoo.OnnxRuntime` package
imports; each lands in `ort/<BackendId>/` in the output. An execution provider's own library
has to sit beside the core it belongs to, so deploy a package's whole native folder.

**The backend's own assembly has to be somewhere too, and not beside `Shorokoo.dll`** — two
backend assemblies there is the ambiguity [auto-discovery](#auto-discovery) refuses. Put it
in the same folder as its native and point `ProbeDirectory` at it:

```xml
<PackageReference Include="Shorokoo.LinuxGPU" Version="..." ExcludeAssets="all"
                  GeneratePathProperty="true" />
<None Include="$(PkgShorokoo_LinuxGPU)\lib
et10.0\Shorokoo.LinuxGPU.dll"
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

Only the ONNX Runtime wrapper, the glue over it and the factory assembly are private to an
isolated backend; the core Shorokoo assembly and everything your model is written in stay
shared, which is what lets one context be handed the other's data.

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
CUDA provider uses, so the shape carries over but the numbers do not
([#357](https://github.com/Shorokoo/Shorokoo/issues/357) tracks confirming them on a card). Treat
these as indicative of the shape, not as your machine's numbers — what settles your case is
`DeviceMemory.Sample()` around your own run.

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
| `ArenaExtend` | `DeviceMemorySettings` | `arena_extend_strategy` | `Auto` — `SameAsRequested`, except ORT's `NextPowerOfTwo` for a session Shorokoo knows is reused across differing shapes | when a session is created |
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
ORT reads on every run, so a `CompiledGraph.Execute` / `Run` call can override it for that call
alone. The context's own one-shot entry points and a rig's `TrainStep` take no such override and
run on the context's instance, so set it on the context they run on.

The static `DeviceMemory` class — the readings, not the settings — reports what the card is doing:

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
