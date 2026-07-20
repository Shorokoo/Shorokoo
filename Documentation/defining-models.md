# Defining models with `[Module]`

Related: [core-types.md](core-types.md) · [inference.md](inference.md) ·
[training.md](training.md)

## Facts

- A model/layer is a `partial class` marked `[Module]` containing a `static` method
  named `Inline`. The source generator reads `Inline` and generates `Model(...)`,
  `Call(...)`, and `ComputationGraph` members.
- Attributes (namespace `Shorokoo.Modules`):
  - `[Module]` — on the partial class. Generates the callable graph.
  - `[Hyper]` — on scalar hyperparameter parameters (configured when the module is
    built). All `[Hyper]` parameters must come **after** the input (tensor)
    parameters in the `Inline` signature. An optional default — `[Hyper(0.9f)]` —
    seeds the generated hyperparameter set (see *Generated surface*). For when a
    hyperparameter's value is fixed vs supplied at runtime, see
    [Hyperparameter baking](#hyperparameter-baking) below.
  - `[TrainableParamInitializer]` — on a `static partial class` whose `Inline`
    produces a trainable weight tensor. Generates an `Init(...)` method.
  - `[StateInitializer(Ownership = ...)]` — like `TrainableParamInitializer`, but for
    non-trainable state. `Ownership` declares who updates the state:
    `StateOwnership.ModuleOwned` (the default) for state the module's own forward
    logic updates (e.g. BatchNorm running statistics);
    `StateOwnership.OptimizerOwned` for optimizer state (e.g. Adam moments), which
    the `TrainingRig` replicates per trainable parameter. State **must** be created
    through a state initializer's `Init(...)`: `Globals.StateUpdate(state, newValue)`
    throws `InvalidStateUpdateException` when its first argument is anything else —
    a runtime input, a trainable parameter, or a computed tensor. `StateUpdate` is
    also only valid inside a module body — it throws otherwise. Inside a
    `LoopAPI.Iterate` loop body it registers the post-loop value of the updated
    tensor (sugar for the after-the-loop registration — behaviorally equivalent,
    though the built graph's node order may differ), which requires the updated
    value to be a carried loop variable; each state still gets exactly one update
    per step.
- `Inline` may return a single value or a tuple (multiple outputs).
- The class must be `partial` so the generator can extend it.
- The generator is a convenience, not a requirement — see
  [Without the source generator](#without-the-source-generator) for the
  `ModuleFactory.FromFunc` path.

## Generated surface

For `[Module] class Foo` with `Inline(I x, [Hyper] H h) -> O`:

| Generated member | Meaning |
|---|---|
| `Foo.Model(h)` | Bind hyperparameters; returns a reusable `Model` you can `.Call(...)` many times. |
| `Foo.Model(h).Call(x)` | Build the subgraph for input `x`. |
| `Foo.Call(h, x)` | Shortcut for `Foo.Model(h).Call(x)`. The combined shortcut keeps hyperparameters first (then inputs); only the `Inline` source signature is inputs-first. |
| `Foo.ComputationGraph` | The full `FastComputationGraph` (used for export and training). |
| `FooHyperparameters` | Generated **only when every `[Hyper]` is a scalar `float32`** (the optimizer-shaped case): a named, init-only set implementing `IOptimizerHyperparameters`, with defaults from `[Hyper(default)]`. See [training.md](training.md). |

For `[TrainableParamInitializer] class ConstInit` with `Inline(Vector<int64> shape)`:
`ConstInit.Init(shape)` returns the initialized trainable `Tensor<T>`. (The class
must not itself be named `Init` — the generated `Init` member would collide with
the type name; the generator rejects that with error `MSG003`.)

## Hyperparameter baking

There are two kinds of `[Hyper]` parameters, and they behave differently when a
module is prepared for training or execution:

- **Shape-determining** hyperparameters feed the shape (or number) of trainable
  parameter tensors — e.g. `outFeatures` in a dense layer, used as
  `ConstInit.Init([outFeatures, inFeatures])`. When the graph is concretized
  (`ToConcreteArchitecture` — the step that prepares a model for training,
  weight binding, or export), the value you supply for such a hyperparameter is
  **baked**: the trainable parameters get fixed shapes from it, and supplying a
  different value later does not resize them. These *must* be baked — a model's
  trainable parameters cannot have runtime-dependent shapes.
- **Value-only** hyperparameters (scale factors, momentum coefficients, ε's) do
  not constrain any trainable parameter. They do not have to be baked: in the
  concretized graph they remain live runtime inputs, and in an optimizer they
  may be scheduled per-step (see [training.md](training.md)).

How a hyper value gets supplied depends on the route:

- `Foo.Call(Scalar(k), x)` / `Foo.Model(Scalar(k))` — the hyper is embedded as a
  constant node in the built subgraph.
- `Foo.ComputationGraph` + concretize — **every** hyper stays an input of the
  concrete graph. The framework keeps the concrete graph's inputs ordered
  hyperparameters-first (independent of the inputs-first `Inline` source order), so
  `Execute` must be given the hyper values again — the hypers (in their `Inline`
  relative order) first, then the inputs. The values passed at concretization time bake the
  trainable-parameter shapes (shape-determining hypers) and serve as shape/type
  hints; value-only hypers are read live at every `Execute`. See
  [inference.md](inference.md#running-a-module-with-hyper-parameters) for the
  recipe.
- `Foo.ComputationGraph` + **`Specialize`** + concretize — to hardcode hypers
  (either kind) instead of re-supplying them at every `Execute`, call
  `Specialize` before `ToConcreteArchitecture`. It constant-folds the named hyper
  values into the graph and removes them from the input list, so the concrete
  model runs on the remaining inputs alone. See
  [inference.md](inference.md#hardcoding-hypers-with-specialize).

## Workflow: add a new layer

1. Add `using Shorokoo; using Shorokoo.Modules; using static Shorokoo.Globals; using static Shorokoo.NN;`.
2. Declare `[Module] public partial class MyLayer`.
3. Write `public static <OutputType> Inline(<input tensors...>, <[Hyper] hypers...>)`.
4. Build the output from tensor ops, `NN.*` ops, sub-modules (`Other.Model(...).Call(...)`),
   and weights from a `[TrainableParamInitializer]`.
5. Ensure the project references the generator as an analyzer (see below).
6. Build, then call `MyLayer.Call(...)` or use `MyLayer.ComputationGraph`.

## Control flow inside `Inline`

- Conditional (data-dependent): `condition.IfElse(whenTrue, whenFalse)` where
  `condition` is `Scalar<bit>`. Both branches are built; the value is selected at
  runtime. Tuples are supported.

  ```csharp
  // Apply bias only when useBias is true — both branches are always built.
  var b = ConstInit.Init([outFeatures]).Vec();
  return useBias.IfElse(y + b, y);

  // Tuples are also supported — sort a pair data-dependently:
  var (lo, hi) = (a < b).IfElse((a, b), (b, a));
  ```

- Loops: `foreach (var ctx in LoopAPI.Iterate(count)) { ...; ctx.IterationIndex; }`
  where `count` is `Scalar<int64>`. Use this instead of a plain C# `for` when the
  iteration count is a graph value.

  Simple — add `x` to itself `n` times:
  ```csharp
  var acc = TensorFill(x.TShape, 0f);
  foreach (var ctx in LoopAPI.Iterate(n))
      acc = acc + x;
  ```

  Comprehensive — weighted accumulation using the iteration index:
  ```csharp
  // numSteps is a runtime Scalar<int64> — cannot use a plain C# for.
  var total = TensorFill(x.TShape, 0f);
  foreach (var ctx in LoopAPI.Iterate(numSteps))
  {
      var weight = (ctx.IterationIndex + Scalar(1L)).Cast<float32>();  // 1.0, 2.0, 3.0, …
      total = total + x * weight;
  }
  return total;   // x·1 + x·2 + … + x·numSteps
  ```

## Omittable parameters (defaulted hypers & optional inputs)

`Inline` parameters are always written as ordinary, **non-nullable** types. The source
generator turns two kinds of parameter into **nullable, omittable** parameters on the
generated `Model` / `Call` surface, so callers can leave them out:

| `Inline` parameter | Generated `Model`/`Call` parameter | When omitted / `null` |
|---|---|---|
| `[Hyper(0.9f)] Scalar<float32> momentum` | `Scalar<float32>? momentum = null` | the attribute's default (`0.9f`) is used |
| `OptionalTensor<float32> bias` | `Tensor<float32>? bias = null` | an **absent** optional is passed |

(C#'s "optional parameters last" rule still applies: only the trailing run of
omittable parameters gets a `= null` default — a defaulted hyperparameter that sits
before a required input stays nullable but must be supplied, where `null` still means
"use the default".)

### Defaulted hyperparameters

A `[Hyper(default)]` scalar — e.g. an optimizer's learning rate, an epsilon — can be
omitted entirely:

```csharp
[Module]
public partial class Scaled
{
    public static Tensor<float32> Inline(Tensor<float32> x, [Hyper(2f)] Scalar<float32> factor)
        => x * factor;
}

var m1 = Scaled.Model();            // factor defaults to 2.0
var m2 = Scaled.Model(Scalar(5f));  // factor = 5.0
```

The default is recorded on the module's hyperparameter input, so it is preserved when the
module is serialized — a round-trip through ONNX or C# emission keeps `[Hyper(2f)]`.

### Optional tensor inputs

Declare the parameter as an `OptionalTensor<T>` and branch on its presence with the
optional API. Use the **lazy** (`() => …`) form of `IfElse` so the value is only
unwrapped on the present branch — eagerly unwrapping an absent optional is invalid:

```csharp
[Module]
public partial class DenseWithOptionalBias
{
    public static Tensor<float32> Inline(Tensor<float32> x, OptionalTensor<float32> bias)
    {
        var b = bias.HasValue().IfElse(() => bias.TensorValue(),         // present
                                       () => TensorFill(x.ShapeTensor(), 0f));  // absent default
        return x + b;
    }
}

var y0 = DenseWithOptionalBias.Call(x);          // bias omitted → zeros default
var y1 = DenseWithOptionalBias.Call(x, myBias);  // bias supplied
```

The caller-facing parameter is `Tensor<float32>?`; an `OptionalTensor<T>` is also
implicitly convertible to `Tensor<T>?`, so a present optional can be forwarded
directly.

### Supplying optional values at execution

A `[Module]`'s `ComputationGraph` keeps an optional parameter as an `OptionalTensor`
graph input. Feed it an **`OptionalTensorData`**: `OptionalTensorData.Some(tensor)`
for a value, `OptionalTensorData.None(dtype)` for the absent (default) branch. ONNX
Runtime accepts a plain tensor where a *present* optional is expected, but cannot take
an *absent* optional input — execute graphs that exercise the absent branch through
`new QuickExecutionEngine().Execute(concreteModel, inputs…)`, which is optional-aware
in pure managed code.

## Project wiring (required for codegen)

The source generator (`Shorokoo.CodeGen`) must be referenced from the consuming
`.csproj` as a **Roslyn analyzer**, not as an ordinary assembly/project reference.
If it isn't, no `Model` / `Call` / `ComputationGraph` (or `Init`) members are
generated and the build fails with errors like
`'MyLayer' does not contain a definition for 'Call'`.

**Consuming the NuGet packages** (the normal case) — nothing to wire by hand.
The `Shorokoo` meta package (or an explicit `Shorokoo.CodeGen` package
reference) flows the generator as an analyzer automatically:

```bash
dotnet add package Shorokoo          # generator flows transitively
dotnet add package Shorokoo.LinuxCPU # plus one backend for your platform
```

**Building against the source tree** (the generator project is in your solution) —
add it as a `ProjectReference` marked as an analyzer:

```xml
<ProjectReference Include="..\..\src\Shorokoo.CodeGen\Shorokoo.CodeGen.csproj"
                  OutputItemType="Analyzer" ReferenceOutputAssembly="false" />
```

In both forms the generator runs at compile time only and is kept out of the
runtime closure. If `Call`/`Model` come back "not defined" after a build, the
generator was referenced as a plain `<Reference>`/`<ProjectReference>` instead
of an analyzer.

## Example

```csharp
[TrainableParamInitializer]
public static partial class ConstInit
{
    public static Tensor<float32> Inline(Vector<int64> shape)
        => Tensor<float32>.Fill(shape, Globals.TensorData(1, 1.0f));
}

[Module]
public partial class DenseBasic
{
    public static Tensor<float32> Inline(
        Tensor<float32> x,                    // inputs first
        [Hyper] Scalar<int64> outFeatures,    // hyper params last
        [Hyper] Scalar<bit> useBias)
    {
        var inFeatures = x.TShape[1..^0].T.Reduce(ReduceKind.Prod).Scalar();
        var xFlat = x.Reshape([x.DimTensor(0), inFeatures]);
        var w = ConstInit.Init([outFeatures, inFeatures]);
        var y = xFlat.MatMul(w.Transpose([1L, 0L]));
        var b = ConstInit.Init([outFeatures]).Vec();
        return useBias.IfElse(y + b, y);      // data-dependent branch
    }
}

// Compose and call. Note: the `Inline` signature is inputs-first, but the combined
// `Call` shortcut keeps hyperparameters first (then the input), so call sites are unchanged:
var logits = DenseBasic.Call(Scalar(10L), Scalar(true), features);
```

## Without the source generator

Shorokoo is fully usable without `Shorokoo.CodeGen`. The codegen-free entry point is
`Shorokoo.Modules.ModuleFactory`: write the module body as a **static method** (or a
**non-capturing `static` lambda**) with the same flattened parameter shape an `Inline`
method would have, and the factory gives you everything the generator would have
emitted.

| Generated member | Codegen-free equivalent |
|---|---|
| `Foo.Model(h...)` | `ModuleFactory.FromFuncWithHypers(...).SetHyperparams((h...))` |
| `Foo.Model().Call(x)` | `ModuleFactory.FromFunc(...).SetHyperparams().Call(x)` |
| `Foo.ComputationGraph` | `ModuleFactory.ComputationGraph(body)` |
| `Init.Init(shape)` | `Globals.CallTrainableParamInitializer(body, name, isTrainable, shape)` |

### End-to-end example (define, call, train, export)

```csharp
using Shorokoo;
using Shorokoo.Modules;
using static Shorokoo.Globals;

// 1. Define — a plain static method, same shape as an Inline method.
//    Trainable-param initializers, Globals.StateUpdate, LoopAPI.Iterate, and
//    sub-module calls all work inside the body exactly as in [Module] classes.
static Tensor<float32> InitOnes(Vector<int64> shape) => TensorFill(shape, 1.0f);

static Tensor<float32> ScalarMultiply(Tensor<float32> input)
{
    // Codegen-free spelling of a [TrainableParamInitializer]'s Init(...):
    var weight = (Tensor<float32>)CallTrainableParamInitializer(
        InitOnes, defaultName: "InitOnes", isTrainable: true, Vector(1L));
    return input * weight;
}

// 2. Call — module → model → call, like the generated Model()/Call() pair.
var module = ModuleFactory.FromFunc<Tensor<float32>, Tensor<float32>>(ScalarMultiply);
var model  = module.SetHyperparams();          // generated: ScalarMultiply.Model()
var y      = model.Call(x);                    // generated: ScalarMultiply.Call(x)

// 3. The computation graph — equivalent to the generated ComputationGraph property
//    (cached build, fresh clone per call).
var graph = ModuleFactory.ComputationGraph(
    (Func<Tensor<float32>, Tensor<float32>>)ScalarMultiply);

// 4. Train — TrainingRig consumes graphs, so nothing changes (see training.md).
var rig = TrainingRig.FromScratch(graph, L2Loss.ComputationGraph,
    SGDOptimizer.ComputationGraph, sampleInputs, 0.01f);

// 5. Export — concretize and save/export as usual (see onnx-and-weights.md).
var concrete = graph.ToConcreteArchitecture(graph.FromOrderedInputs([sample]))
                    .ToConcreteModel();
var onnx = FastOnnxModelBuilder.BuildOnnxModel(concrete);
```

### Hyperparameters

Annotate the trailing parameters with `[Hyper]` — on the static method, or on an
explicitly-typed lambda's parameters — and use the `FromFuncWithHypers` overloads
(one runtime input, 1–3 hyperparameters). The graph builder reads the attribute off
the delegate's parameters, so the annotations are required, and the factory rejects
delegates whose annotations don't match the overload's hyper split:

```csharp
static Tensor<float32> Scale(Tensor<float32> x, [Hyper] Scalar<float32> k) => x * k;

var m = ModuleFactory.FromFuncWithHypers<Tensor<float32>, Scalar<float32>, Tensor<float32>>(Scale);
var y = m.SetHyperparams(Scalar(2f)).Call(x);
```

### Multiple inputs

`FromFunc` has overloads for 2–4 runtime inputs; the body keeps flattened parameters
and the module's input type becomes a tuple. Bind a `Model<T1, T2, TOut>` for a
two-argument `Call`:

```csharp
static Tensor<float32> Add(Tensor<float32> a, Tensor<float32> b) => a + b;

var model = ModuleFactory.FromFunc<Tensor<float32>, Tensor<float32>, Tensor<float32>>(Add)
    .SetHyperparams<Model<Tensor<float32>, Tensor<float32>, Tensor<float32>>>();
var y = model.Call(a, b);
```

For hyperparameters combined with *multiple* runtime inputs, construct the
`Module<THypers, TInputs, TOutputs>` base directly with a wrapper lambda (this is
exactly what the generator emits):

```csharp
// Body is inputs-first: Body(Tensor<float32> a, Tensor<float32> b, [Hyper] Scalar<float32> h)
new Module<Scalar<float32>, (Tensor<float32>, Tensor<float32>), Tensor<float32>>(
    (h, ins) => Body(ins.Item1, ins.Item2, h), Body);
```

### Constraints and ergonomics differences

- **Static, non-capturing bodies only.** The body is invoked once to build the graph
  and the result is cached per method, so a capturing lambda (or a delegate bound to
  an object instance) is rejected. Pass varying values as `[Hyper]` parameters or
  runtime inputs instead.
- **Flattened parameters.** Like `Inline` methods, bodies take one parameter per
  tensor — tuple-typed parameters are rejected; use the multi-parameter overloads.
- **No generated typed hyperparameter sets.** The `FooHyperparameters` classes
  (named, defaulted `HyperValue` properties implementing `IOptimizerHyperparameters`)
  are codegen-only. For optimizer-style scheduling, pass `HyperValue` /
  `Schedules.*` values positionally to `TrainingRig.FromScratch(...)` (in the
  optimizer's `[Hyper]` parameter order) — see [training.md](training.md).
- **Naming.** The module name defaults to the body's declaring class; pass the
  optional `name:` argument for lambdas or when you want the codegen-style class
  name in exports.
- Lower-level building blocks are public too if you need them:
  `GraphBuilder.BuildFastComputationGraphFromDelegate(...)` (uncached graph build)
  and the `Module<...>` / `CallbackModule<...>` constructors shown above.

## Anti-patterns

- Do not put `[Hyper]` parameters before input parameters; generation expects inputs
  first and hyperparameters last.
- Do not write nullable `Inline` parameters (`Tensor<T>?`); declare an
  `OptionalTensor<T>` (or a `[Hyper(default)]` scalar) and let the generator expose the
  omittable `Tensor<T>?` / nullable form to callers (see
  [Omittable parameters](#omittable-parameters-defaulted-hypers--optional-inputs)).
- When unwrapping an optional inside a branch, use the lazy `IfElse(() => …, () => …)`
  form so an absent optional is never eagerly unwrapped.
- Do not name a `[TrainableParamInitializer]`/`[StateInitializer]` class `Init`;
  the generated `Init(...)` member would collide with the type name (generator
  error `MSG003`).
- Do not forget `partial` on the class, or the `static` modifier on `Inline`.
- Do not use a plain C# `for`/`if` on graph values (`Scalar<int64>`/`Scalar<bit>`) when
  the count/condition is dynamic; use `LoopAPI.Iterate` / `.IfElse`.
- Do not switch threads inside a module body (`async`/`await`, `Parallel.For`, callbacks
  run elsewhere): the body runs synchronously on a single thread, and calls like
  `Globals.StateUpdate` or `Rng.Pin` made from another thread throw.
- Do not reference the code generator as a normal project reference; it must be an
  analyzer (`OutputItemType="Analyzer"`).
