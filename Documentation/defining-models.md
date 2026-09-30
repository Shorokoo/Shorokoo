# Defining models with `[Module]`

Related: [core-types.md](core-types.md) · [inference.md](inference.md) ·
[training.md](training.md)

## Facts

- A model/layer is a `partial class` marked `[Module]` containing a `static` method
  named `Inline`. The source generator reads `Inline` and generates `Model(...)`,
  `Call(...)`, and `ComputationGraph` members. `Inline` may return a single value or a
  tuple.
- Attributes (namespace `Shorokoo.Modules`):
  - `[Module]` — on the partial class. Generates the callable graph.
  - `[Hyper]` — on hyperparameter parameters, which must come **after** the input
    (tensor) parameters. An optional default (`[Hyper(0.9f)]`) seeds the generated
    hyperparameter set. See [Hyperparameter baking](#hyperparameter-baking).
  - `[TrainableParamInitializer]` — on a `static partial class` whose `Inline`
    produces a trainable weight tensor. Generates an `Init(...)` method.
  - `[StateInitializer(Ownership = ...)]` — the same for non-trainable state.
    `StateOwnership.ModuleOwned` (default): the module's forward logic updates it (e.g.
    BatchNorm running statistics). `StateOwnership.OptimizerOwned`: optimizer state
    (e.g. Adam moments), replicated by the `TrainingRig` per trainable parameter.
- `Globals.StateUpdate(state, newValue)` rules:
  - Its first argument must come from a state initializer's `Init(...)`; anything else
    (a runtime input, a trainable parameter, a computed tensor) throws
    `InvalidStateUpdateException`. It throws outside a module body.
  - Inside a `LoopAPI.Iterate` body it registers the post-loop value, which must be a
    carried loop variable; the body registers one update however many trips it runs.
  - Each **call** of a module registers its update: calling one model twice applies
    both in order, the second reading what the first wrote (a call whose result is
    discarded still updates).
  - A call in a `LoopAPI.Iterate` body is one call site however many trips the loop
    runs: every trip starts from the value the state held entering the loop, the calls
    within a trip compose in order, and the update is the one the last trip that ran
    made. The loop composes with calls before and after it like any other call.
  - In an `IfElse`, only the arm that runs updates; calls within an arm compose with
    each other and with calls before the branch. A call the branch does not choose
    between must come before the ones it does.
- The generator is optional; see [Without the source generator](#without-the-source-generator).

## Generated surface

For `[Module] class Foo` with `Inline(I x, [Hyper] H h) -> O`:

| Generated member | Meaning |
|---|---|
| `Foo.Model(h)` | Bind hyperparameters; returns a reusable `Model` you can `.Call(...)` many times. |
| `Foo.Model(h).Call(x)` | Build the subgraph for input `x`. |
| `Foo.Call(h, x)` | Shortcut for `Foo.Model(h).Call(x)`. Hyperparameters come first here; only the `Inline` source signature is inputs-first. |
| `Foo.ComputationGraph` | The readonly `ComputationGraph` (kind `Module`; used for export and training). |
| `FooHyperparameters` | Generated **only when every `[Hyper]` is tensor-shaped** (`Scalar<T>`, `Vector<T>` or `Tensor<T>`, any dtype): an init-only set implementing `IOptimizerHyperparameters`, with defaults from `[Hyper(default)]` at the declared dtype. Only a scalar can have a default; a non-scalar's property is `required`. See [training.md](training.md). |

For `[TrainableParamInitializer] class ConstInit` with `Inline(Vector<int64> shape)`:
`ConstInit.Init(shape)` returns the initialized trainable `Tensor<T>`. A class named
`Init` is rejected with `MSG003`.

An initializer may take inputs beyond the shape. An `Init(...)` call inside an
initializer body evaluates the called initializer as a value, not a second parameter,
so initializers compose. A `Tensor<T>` input may be **another trainable parameter**,
arriving as its initial value. An initializer body may **not** create or reference a
model (no `Foo.Model(...)`, `Foo.Call(...)`, `ModelSequence`, `GetTrainableParam`, or
model-typed input): that fails with `FW055` when the graph is built. See *Writing your
own* in [nn-library.md](nn-library.md#initializers-shorokoomodulesinitializers).

## Hyperparameter baking

`ToConcreteArchitecture` fixes which trainable parameters exist and their shapes. A
`[Hyper]` is one of two kinds:

- **Parameter-space-determining**: it feeds a parameter's **shape** or count (e.g.
  `outFeatures` in `ConstInit.Init([outFeatures, inFeatures])`), or **gates a branch
  holding parameters** (e.g. `useBias` in `useBias.IfElse(y + b, y)`). It is **baked**
  from the value supplied at concretization. The unselected branch's parameters are not
  created, and the `IfElse` left holding them is folded away. With `useBias = false`
  there is no bias and no branch; with `useBias = true` nothing is pruned and the bit
  still selects at run time. A tuple `IfElse`, or a gate sharing its parameters with
  another gate, does not fold: its pruned branch reads a zero stand-in. Other `IfElse`s
  on the same hyper stay live. See
  [What concretization fixes](inference.md#what-concretization-fixes).
- **Value-only** (scale factors, momentum, ε): read at every `Execute`, may vary per
  call, and may be scheduled per step in an optimizer (see [training.md](training.md)).

How a hyper is supplied depends on the route:

- `Foo.Call(Scalar(k), x)` / `Foo.Model(Scalar(k))` — a constant node in the subgraph.
- `Foo.ComputationGraph` + concretize — **every** hyper stays an input. `Execute` takes
  the hypers first (in `Inline` order), then the inputs. Re-supply a
  parameter-space-determining hyper with **the value you concretized with**; any other
  value either changes nothing or silently runs the other branch. See
  [inference.md](inference.md#running-a-module-with-hyper-parameters).
- `Foo.ComputationGraph` + **`Specialize`** + concretize — `Specialize` constant-folds
  the named hypers and removes them from the inputs. See
  [inference.md](inference.md#hardcoding-hypers-with-specialize).

## Workflow: add a new layer

1. Add `using Shorokoo; using Shorokoo.Modules; using static Shorokoo.Globals; using static Shorokoo.NN;`.
2. Declare `[Module] public partial class MyLayer`.
3. Write `public static <OutputType> Inline(<input tensors...>, <[Hyper] hypers...>)`.
4. Build the output from tensor ops, `NN.*` ops, sub-modules (`Other.Model(...).Call(...)`),
   and weights from a `[TrainableParamInitializer]` (which may not create or call a
   model: `FW055`).
5. Ensure the project references the generator as an analyzer (see below).
6. Build, then call `MyLayer.Call(...)` or use `MyLayer.ComputationGraph`.

## Control flow inside `Inline`

- Conditional: `condition.IfElse(whenTrue, whenFalse)` where `condition` is
  `Scalar<bit>`. Tuples are supported.

  Both branches are *built*, but only the selected one *runs*: work serving only one
  branch (including whole loops and nested `IfElse`s) executes only when that branch is
  taken; work shared with the other branch or read afterwards is computed once. A
  branch may therefore hold an operation invalid on the other path (e.g. unwrapping an
  absent `OptionalTensor`; see [Optional tensor inputs](#optional-tensor-inputs)).

  Hypers gating parameters fold as described in
  [Hyperparameter baking](#hyperparameter-baking). A hyper that is a **constant** before
  lowering (via `Foo.Call(Scalar(k), x)` or
  [`Specialize`](inference.md#hardcoding-hypers-with-specialize)) folds *every*
  `IfElse` on it.

  ```csharp
  // Apply bias only when useBias is true — both branches are built, one runs.
  // (But b exists in the concrete architecture only if useBias was baked true.)
  var b = ConstInit.Init([outFeatures]).Vec();
  return useBias.IfElse(y + b, y);

  // Tuples are also supported — sort a pair data-dependently:
  var (lo, hi) = (a < b).IfElse((a, b), (b, a));
  ```

- Loops: `foreach (var ctx in LoopAPI.Iterate(count)) { ...; ctx.IterationIndex; }`
  where `count` is `Scalar<int64>`. **Prefer this to a plain C# `for` for any
  repetition**, even with a constant count. A plain `for` unrolls at trace time and
  numbers parameters in trace order; `LoopAPI.Iterate` names each iteration's
  parameters by iteration ([Parameter names](#parameter-names)):

  ```
  for (int i = 0; i < 3; i++)          foreach (var ctx in LoopAPI.Iterate(Scalar(3L)))
    TrainableParam#0.NormalDist#0        TrainableParam#0.Loop#0:0.NormalDist#0
    TrainableParam#0.NormalDist#1        TrainableParam#0.Loop#0:1.NormalDist#0
    TrainableParam#0.NormalDist#2        TrainableParam#0.Loop#0:2.NormalDist#0
  ```

  These names key checkpoints and naming schemes
  ([onnx-and-weights.md](onnx-and-weights.md#naming)); under a plain `for`, adding or
  removing one `Init(...)` renumbers every later parameter. Use a plain `for` only when
  the body differs between iterations.

  Simple — add `x` to itself `n` times:
  ```csharp
  var acc = TensorFill(x.TShape, 0f);
  foreach (var ctx in LoopAPI.Iterate(n))
      acc = acc + x;
  ```

  Comprehensive — weighted accumulation using the iteration index:
  ```csharp
  var total = TensorFill(x.TShape, 0f);
  foreach (var ctx in LoopAPI.Iterate(numSteps))
  {
      var weight = (ctx.IterationIndex + Scalar(1L)).Cast<float32>();  // 1.0, 2.0, 3.0, …
      total = total + x * weight;
  }
  return total;   // x·1 + x·2 + … + x·numSteps
  ```

  `ctx.Scan(v)` stacks `v` from every iteration into one tensor with a new leading axis
  of length `count`, available after the loop. `v` can be anything the body reads (a
  carry before or after its update, the index, an outside value):
  ```csharp
  Variable? scanned = null;
  var acc = x;
  foreach (var ctx in LoopAPI.Iterate(n))
  {
      scanned = (Variable)ctx.Scan(acc);   // the state at the top of each iteration
      acc = acc + Scalar(1.0f);
  }
  return (Tensor<float32>)scanned!;   // [x, x+1, …, x+n-1]
  ```

  An **enclosing** loop's `ctx.Scan` called from a nested body records once per
  enclosing iteration (the value at its end). An inner loop's own scan cannot escape
  the enclosing loop; see [limitations.md](limitations.md).

  A local may carry another carry's value from **one iteration ago**:
  ```csharp
  var acc = x, prev = x, sum = Scalar(0.0f);
  foreach (var ctx in LoopAPI.Iterate(trips))
  {
      sum  = sum + prev;          // acc's value from the previous iteration
      prev = acc;
      acc  = acc + Scalar(1.0f);
  }
  ```
  Written bare, a lagged local may be read or scanned inside the body (including a
  nested loop, when created in that loop's enclosing body). Reading it after the loop,
  lagging a local the enclosing loop also carries, trailing a non-carry, chaining two
  lags, or an alias seeded outside the loop is refused, naming the shape; write
  `prev = LoopAPI.Carry(acc)` instead, plus `LoopAPI.Init` for a local the body only
  writes. See [limitations.md](limitations.md).

## Struct values

A struct type is an `IStruct` interface, record or class. Its fields are its public
instance properties, inherited ones included, and each is a `Tensor`, `Vector`, `Scalar`,
`TensorSequence`, `OptionalTensor` or `TensorStruct<T>` (a nested struct). A struct value
is immutable: updating a field means building a new struct.

`Tensor<T>`'s index setter (`t[0..2] = r;`) rebinds a local, so it cannot write through
a field. `s.A[0..2] = r;` does not compile (CS1612), whether `A` is a property or
`GetField<T>("A")`. Copy the field, set the copy, and bind a new struct to `s`:

```csharp
var a = s.A;                                   // s is unchanged
a[0..2] = r;
s = TensorStruct<MyStruct>(a);                 // MyStruct an IStruct interface
p = p with { A = a };                          // p a record: `with` does the same
```

This is the same inside a `LoopAPI.Iterate` body, where a rebound `s` is carried to the
next iteration. `IfElse` chooses between `TensorStruct<T>` handles, so to choose between
structs build them with `TensorStructCreate<MyStruct>(...)` and read the chosen one's
fields with `GetField<T>`.

A struct type with a member that cannot be a field is refused when the type is first
used as a struct (a module input or output, `TensorStruct<T>(...)`): a public C# field
(`public Tensor<float32> A;`), a `ref`-returning property, or a property of any other type.

## Workflow: one module, many variants

A `[Module]` class is a **family** of architectures. Its graph is traced once and
cached, but `[Hyper]` parameters are symbolic inputs, so that graph covers every
configuration: a `Scalar<int64>` hyper sets shapes or trip counts, a `Scalar<bit>`
hyper picks algorithms and which parameters exist. A different scale is a value, not a
new class.

A ViT-shaped classifier parameterised over width, depth, heads, classes, whether blocks
carry biases, and whether the head is linear or an MLP:

```csharp
using Shorokoo;
using Shorokoo.Graph;                // ComputationGraph
using Shorokoo.Modules;
using Shorokoo.Modules.Initializers; // XavierUniform
using Shorokoo.Modules.Layers;       // TransformerEncoderLayer
using Shorokoo.Modules.Optimizers;   // AdamWOptimizerHyperparameters
using static Shorokoo.Globals;

[Module]
public partial class VisionTransformer
{
    public static Tensor<float32> Inline(
        Tensor<float32> patches,                  // [N, L, patchDim]
        [Hyper] Scalar<int64> embedDim,
        [Hyper] Scalar<int64> numHeads,
        [Hyper] Scalar<int64> ffnDim,
        [Hyper] Scalar<int64> numLayers,
        [Hyper] Scalar<int64> numClasses,
        [Hyper] Scalar<bit>   useBias,
        [Hyper] Scalar<bit>   useMlpHead)
    {
        var proj = XavierUniform.Init([patches.DimTensor(2), embedDim]);
        var pos  = XavierUniform.Init([patches.DimTensor(1), embedDim]);
        var x    = patches.MatMul(proj) + pos;

        foreach (var ctx in LoopAPI.Iterate(numLayers))
            x = TransformerEncoderLayer.Call(embedDim, numHeads, ffnDim, useBias, x);

        Vector<int64> seqAxis = [Scalar(1L)];
        var pooled = x.Reduce(ReduceKind.Mean, seqAxis, keepDims: false);

        var wHead   = XavierUniform.Init([embedDim, numClasses]);
        var wHidden = XavierUniform.Init([embedDim, embedDim]);
        var wOut    = XavierUniform.Init([embedDim, numClasses]);
        return useMlpHead.IfElse(
            pooled.MatMul(wHidden).Tanh().MatMul(wOut),   // MLP head: two parameters
            pooled.MatMul(wHead));                        // linear head: one
    }
}
```

Because `numLayers` drives `LoopAPI.Iterate`, even the number of blocks (and so of
parameters) is chosen by value.

### Picking a variant: `Specialize`

(The snippets below assume the `using` block above; `using` is not recursive.)

`Specialize` constant-folds the hyper values **and removes them from the inputs**
([inference.md](inference.md#hardcoding-hypers-with-specialize)):

```csharp
var family = VisionTransformer.ComputationGraph;   // inputs: the 7 hypers, then patches

static ComputationGraph Variant(
    ComputationGraph family,
    long embedDim, long numHeads, long ffnDim, long numLayers, long numClasses,
    bool useBias, bool useMlpHead)
    => family.Specialize(family.FromOrderedInputs([
        TensorData([], embedDim), TensorData([], numHeads), TensorData([], ffnDim),
        TensorData([], numLayers), TensorData([], numClasses),
        TensorData([], useBias), TensorData([], useMlpHead)]));

var tiny  = Variant(family, 32, 4,  64, 2, 10, useBias: false, useMlpHead: false);
var small = Variant(family, 64, 8, 128, 4, 10, useBias: true,  useMlpHead: true);
// tiny.InputNames == small.InputNames == ["patches"] — the hypers are gone.
```

`FromOrderedInputs` pairs values with the leading input names, and hyper inputs come
first, so the hyper values alone name them correctly.

Concretized on the same input, `tiny` has **23** trainable parameters and `small`
**68**: block count, widths, biases and head all differ. Since `Specialize` makes every
condition a constant, *every* `IfElse` on it folds, including `useBias` gates that plain
concretization with `useBias = true` would leave live.

### Training a variant

`TrainingRig.FromScratch` takes the specialized graph directly, with no hypers to pass
to `TrainStep`:

```csharp
var sample = TensorData([batch, numPatches, patchDim], /* … */);

var rig = TrainingRig.FromScratch(
    tiny, Losses.CrossEntropy, Optimizers.AdamW, [sample],   // one sample per input, in order
    new AdamWOptimizerHyperparameters { LearningRate = 3e-4f });
```

Samples bind by position; to bind by name pass `NamedModelParam`s:
`[new TensorDataModelParam("patches", ModelParamType.InputParam, sample)]`.

Swapping `tiny` for `small` is the whole diff. Each variant is still built separately
(see [What construction costs](training.md#what-construction-costs)).

### What must still be a C# argument

- **`Inline`'s return type cannot vary.** A choice that changes the output *type*
  (e.g. a loss reduced to `Scalar<float32>` vs a per-element `Tensor<float32>`) cannot be
  a hyper. `TrainingRig.FromScratch` takes the loss as a **separate graph**, so pick the
  reduction there ([nn-library.md](nn-library.md#loss-configurable-knobs)).
- **There is no enum hyper.** Encode the knob as a `Scalar<int64>` compared in-graph
  (`(mode > Scalar(3L)).IfElse(a, b)`), or make it a plain C# argument of a `static`
  non-`[Module]` helper, as `Recurrent.RNN` and `EmbeddingBag.Bag` do
  ([nn-library.md](nn-library.md#recurrent-layers)).
- **Both `IfElse` branches must agree in *type*.** Shapes may differ: the result takes
  the selected branch's shape, even at run time on a live gate.

## Omittable parameters (defaulted hypers & optional inputs)

`Inline` parameters are always **non-nullable**. The generator makes two kinds
**nullable and omittable** on `Model` / `Call`:

| `Inline` parameter | Generated `Model`/`Call` parameter | When omitted / `null` |
|---|---|---|
| `[Hyper(0.9f)] Scalar<float32> momentum` | `Scalar<float32>? momentum = null` | the attribute's default (`0.9f`) is used |
| `[Hyper(2)] Scalar<int32> accumSteps` | `Scalar<int32>? accumSteps = null` | the default is formatted at the declared dtype (`2`) |
| `[Hyper] Vector<float32> scales` | `Vector<float32> scales` | not omittable — only a scalar hyperparameter can carry a default |
| `OptionalTensor<float32> bias` | `Tensor<float32>? bias = null` | an **absent** optional is passed |

Only the trailing run of omittable parameters gets `= null`; one before a required
input must be passed, with `null` meaning "use the default".

### Defaulted hyperparameters

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

The default survives serialization (ONNX or C# emission keeps `[Hyper(2f)]`).

### Optional tensor inputs

Declare an `OptionalTensor<T>` and branch on its presence. `TensorValue()` runs only
on the present branch (see [Control flow](#control-flow-inside-inline)).

```csharp
[Module]
public partial class DenseWithOptionalBias
{
    public static Tensor<float32> Inline(Tensor<float32> x, OptionalTensor<float32> bias)
    {
        var b = bias.HasValue().IfElse(bias.TensorValue(),                // present
                                       TensorFill(x.ShapeTensor(), 0f));  // absent default
        return x + b;
    }
}

var y0 = DenseWithOptionalBias.Call(x);          // bias omitted → zeros default
var y1 = DenseWithOptionalBias.Call(x, myBias);  // bias supplied
```

An `OptionalTensor<T>` converts implicitly to `Tensor<T>?`, so a present optional can
be forwarded.

### Supplying optional values at execution

On `ComputationGraph` an optional is an `OptionalTensor` input; feed
`OptionalTensorData.Some(tensor)` or `OptionalTensorData.None(dtype)`. ONNX Runtime
accepts a plain tensor for a *present* optional but cannot take an *absent* one; run
the absent branch with `new QuickExecutionEngine().Execute(concreteModel, inputs…)`.

To train, pass the sample to `TrainingRig.FromScratch` as an `OptionalTensorData` (or an
`OptionalTensorDataModelParam` by name) and build batches with
`rig.InputDef.FromOrderedData(...)`. The sample's arrangement only sizes shape
inference, so either arrangement accepts the same batches. Training runs on ONNX
Runtime, so **the absent branch is not trainable**; passing a tensor instead silently
trains the *present* arm. Cover the absent branch at inference through
`QuickExecutionEngine`.

## Parameter names

A parameter's full name is the path of names from the model down to it:

```
TrainableParam#0.encoder#0.proj#0.weight#0
TrainableParam#0.Loop#0:3.table#0
```

The first part is the category; each further part names a sub-model, loop iteration or
parameter within its scope (the module body, or one `LoopAPI.Iterate` body). These names
key every checkpoint, `TrainableParams.Fields`, and naming schemes
([onnx-and-weights.md](onnx-and-weights.md#naming)). A part's name comes from, in order:

1. **`.Named("...")`** on the `Init(...)` or `Model(...)` result (duplicates in one
   scope fail the build):
   ```csharp
   var x = Normal02.Init([vocab, width]).Named("embedding").Gather(tokens);
   ```
2. **The local the call is assigned to**, when the call is the local's whole
   initializer in a `[Module]` class (`var x = Normal02.Init(...).Gather(tokens)` does
   not name it `x`). Same-named locals in one scope get `#0`, `#1` in creation order.
   Requires the generator; see [Project wiring](#project-wiring-required-for-codegen).
   ```csharp
   var proj = Linear.Model(Scalar(128L), Scalar(true));   // proj#0
   var gain = Ones.Init([width]);                         // gain#0
   ```
3. **Otherwise the class name** (`Normal02#0`, `Linear#1`), numbered in creation order
   within its scope.

Built-in layers use PyTorch-style names: a `Linear` in `proj` gives `proj#0.weight#0`
and `proj#0.bias#0`; `BatchNorm` has `running_mean`, `running_var`, `weight`, `bias`;
`MultiHeadAttention` has `q_proj_weight` … `out_proj_bias`. The `Recurrent`,
`Convolution` and `Embedding` helpers keep initializer class names.

Names from `.Named` and from a distinct local survive reordering and moving into a helper
method of the same class; moving code into a sub-module adds the sub-model's name (`w#0` →
`block#0.w#0`). Creation-order numbers (the class-name fallback, same-named locals) shift on
reordering, so name parameters worth keeping. A
checkpoint whose names do not match is refused on load, naming the missing parameter;
bind it through a [naming scheme](onnx-and-weights.md#naming).

`TrainableParams.Fields` (and every `TensorDataStruct.Fields`) enumerates in definition
order (for model parameters, graph order), the same in every process.

## Project wiring (required for codegen)

`Shorokoo.CodeGen` must be referenced as a **Roslyn analyzer**. Otherwise no members
are generated and the build fails with errors like
`'MyLayer' does not contain a definition for 'Call'`.

**NuGet** (the normal case): the `Shorokoo` package (or `Shorokoo.CodeGen`) flows the
analyzer automatically:

```bash
dotnet add package Shorokoo          # generator flows transitively
dotnet add package Shorokoo.LinuxCPU # plus one backend for your platform
```

**Source tree**: reference the generator project as an analyzer:

```xml
<ProjectReference Include="..\..\src\Shorokoo.CodeGen\Shorokoo.CodeGen.csproj"
                  OutputItemType="Analyzer" ReferenceOutputAssembly="false" />
```

Local-based parameter names use C# interceptors in `Shorokoo.Generated.ParamNames`.
The NuGet package enables them. Building against the source tree, `Directory.Build.props`
does it for projects inside the tree, and a project outside the tree adds:

```xml
<PropertyGroup>
  <InterceptorsNamespaces>$(InterceptorsNamespaces);Shorokoo.Generated.ParamNames</InterceptorsNamespaces>
</PropertyGroup>
```

Without it the build warns (`MSG006`) and parameters keep their class names. The
generator is compile-time only.

## Example

```csharp
[TrainableParamInitializer]
public static partial class ConstInit
{
    public static Tensor<float32> Inline(Vector<int64> shape)
        => Tensor<float32>.Fill(shape, Globals.TensorData(1, 1.0f).MoveToAttribute());
}

[Module]
public partial class DenseBasic
{
    public static Tensor<float32> Inline(
        Tensor<float32> x,                    // inputs first
        [Hyper] Scalar<int64> outFeatures,    // hyper params last
        [Hyper] Scalar<bit> useBias)
    {
        var inFeatures = x.TShape[1..^0].Reduce(ReduceKind.Prod).Scalar();
        var xFlat = x.Reshape([x.DimTensor(0), inFeatures]);
        var w = ConstInit.Init([outFeatures, inFeatures]);
        var y = xFlat.MatMul(w.Transpose([1L, 0L]));
        var b = ConstInit.Init([outFeatures]).Vec();
        return useBias.IfElse(y + b, y);      // data-dependent branch
    }
}

// Call takes hyperparameters first, then the input:
var logits = DenseBasic.Call(Scalar(10L), Scalar(true), features);
```

`Tensor<T>.Fill` takes a
[`TensorAttribute`](core-types.md#two-kinds-of-concrete-tensor-tensordata-and-tensorattribute)
(a graph literal), hence `MoveToAttribute()`. The conversion **spends** its tensor, so
build a fresh one per call. `TensorFill(shape, 1.0f)`, `VectorFill(…)` and the
`RandomNormal` / `RandomUniform` families take a primitive and need no conversion.

## Without the source generator

`Shorokoo.Modules.ModuleFactory` provides everything the generator emits. Write the
body as a **static method** (or **non-capturing `static` lambda**) with an `Inline`
method's flattened parameters.

| Generated member | Codegen-free equivalent |
|---|---|
| `Foo.Model(h...)` | `ModuleFactory.FromFuncWithHypers(...).SetHyperparams((h...))` |
| `Foo.Model().Call(x)` | `ModuleFactory.FromFunc(...).SetHyperparams().Call(x)` |
| `Foo.ComputationGraph` | `ModuleFactory.ComputationGraph(body)` |
| `ConstInit.Init(shape)` | `Globals.CallTrainableParamInitializer(body, defaultName, isTrainable, shape)` |

### End-to-end example (define, call, train, export)

```csharp
using Shorokoo;
using Shorokoo.Modules;
using Shorokoo.Core.Factory;          // FastOnnxModelBuilder
using static Shorokoo.Globals;

// 1. Define — initializers, StateUpdate, LoopAPI and sub-modules work as in [Module] classes.
static Tensor<float32> InitOnes(Vector<int64> shape) => TensorFill(shape, 1.0f);

static Tensor<float32> ScalarMultiply(Tensor<float32> input)
{
    // Codegen-free Init(...):
    var weight = (Tensor<float32>)CallTrainableParamInitializer(
        InitOnes, defaultName: "InitOnes", isTrainable: true, Vector(1L));
    return input * weight;
}

// 2. Call.
var module = ModuleFactory.FromFunc<Tensor<float32>, Tensor<float32>>(ScalarMultiply);
var model  = module.SetHyperparams();          // generated: ScalarMultiply.Model()
var y      = model.Call(x);                    // generated: ScalarMultiply.Call(x)

// 3. The computation graph (cached, readonly).
var graph = ModuleFactory.ComputationGraph(
    (Func<Tensor<float32>, Tensor<float32>>)ScalarMultiply);

// 4. Train (see training.md). Losses.* / Optimizers.* live in namespace Shorokoo.
var rig = TrainingRig.FromScratch(graph, Losses.L2Loss,
    Optimizers.SGD, sampleInputs, 0.01f);

// 5. Export (see onnx-and-weights.md).
var concrete = graph.ToConcreteArchitecture([sample])
                    .ToConcreteModel();
var onnx = FastOnnxModelBuilder.BuildOnnxModel(concrete);
```

### Hyperparameters

Mark the trailing parameters `[Hyper]` (on the method, or on an explicitly-typed
lambda's parameters) and use `FromFuncWithHypers` (one runtime input, 1–3
hyperparameters). The annotations are required and must match the overload's split:

```csharp
static Tensor<float32> Scale(Tensor<float32> x, [Hyper] Scalar<float32> k) => x * k;

var m = ModuleFactory.FromFuncWithHypers<Tensor<float32>, Scalar<float32>, Tensor<float32>>(Scale);
var y = m.SetHyperparams(Scalar(2f)).Call(x);
```

### Multiple inputs

`FromFunc` has overloads for 2–4 runtime inputs (the module's input type becomes a
tuple). Bind a `Model<T1, T2, TOut>` for a two-argument `Call`:

```csharp
static Tensor<float32> Add(Tensor<float32> a, Tensor<float32> b) => a + b;

var model = ModuleFactory.FromFunc<Tensor<float32>, Tensor<float32>, Tensor<float32>>(Add)
    .SetHyperparams<Model<Tensor<float32>, Tensor<float32>, Tensor<float32>>>();
var y = model.Call(a, b);
```

For hyperparameters with *multiple* inputs, construct `Module<THypers, TInputs, TOutputs>`
directly, as the generator does:

```csharp
using Shorokoo.Core;   // Module<...> / CallbackModule<...> / GraphBuilder live here

// Body is inputs-first: Body(Tensor<float32> a, Tensor<float32> b, [Hyper] Scalar<float32> h)
new Module<Scalar<float32>, (Tensor<float32>, Tensor<float32>), Tensor<float32>>(
    (h, ins) => Body(ins.Item1, ins.Item2, h), Body);
```

### Constraints and ergonomics differences

- **Static, non-capturing bodies only.** The graph is built once and cached per method,
  so capturing lambdas and instance-bound delegates are rejected; pass varying values as
  `[Hyper]`s or inputs.
- **Flattened parameters.** Tuple-typed parameters are rejected; use the multi-input
  overloads.
- **No `FooHyperparameters` classes.** Pass `Hyperparameter` / `Schedules.*` values
  positionally to `TrainingRig.FromScratch(...)` in the optimizer's `[Hyper]` order (see
  [training.md](training.md)).
- **Naming.** The module name defaults to the declaring class; pass `name:` to override.
- Lower level: `GraphBuilder.BuildComputationGraphFromDelegate(...)` (uncached) and the
  `Module<...>` / `CallbackModule<...>` constructors.

## Anti-patterns

- `[Hyper]` parameters before inputs.
- Nullable `Inline` parameters (`Tensor<T>?`); use `OptionalTensor<T>` or
  `[Hyper(default)]` (see
  [Omittable parameters](#omittable-parameters-defaulted-hypers--optional-inputs)).
- An initializer class named `Init` (`MSG003`).
- Missing `partial` on the class or `static` on `Inline`.
- A plain C# `for`/`if` on graph values; use `LoopAPI.Iterate` / `.IfElse`.
- One class per configuration; use `[Hyper]` values
  ([Workflow: one module, many variants](#workflow-one-module-many-variants)).
- Stacking layers with a plain C# `for`, even with a constant count
  ([Control flow inside `Inline`](#control-flow-inside-inline)).
- Switching threads in a module body (`async`/`await`, `Parallel.For`): the body runs on
  one thread, and `Globals.StateUpdate` or `Rng.Pin` from another thread throws.
- Referencing the generator as a normal project reference instead of an analyzer
  (`OutputItemType="Analyzer"`).
