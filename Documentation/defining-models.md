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
    [Hyperparameter baking](#hyperparameter-baking) below; for writing one module
    that covers a whole family of architectures, see
    [Workflow: one module, many variants](#workflow-one-module-many-variants).
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
    tensor — the value it holds once the loop finishes — which requires the updated
    value to be a carried loop variable, and the body still registers exactly one
    update however many trips it runs. A module body registers one update per
    **call**: calling one model object twice applies both, in call order, with the
    second call reading what the first wrote — including a call whose result the body
    discards, which is how you call for the update alone. Across the arms of an
    `IfElse` only the arm that runs updates the state, and calls within one arm
    compose with each other and with any call made before the branch. A call the
    branch does not choose between has to be made before the ones it does: the
    parameter cannot carry the branch's answer before the branch produces it.
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
| `Foo.ComputationGraph` | The readonly `ComputationGraph` (kind `Module`; used for export and training). |
| `FooHyperparameters` | Generated **only when every `[Hyper]` is tensor-shaped** — `Scalar<T>`, `Vector<T>` or `Tensor<T>`, at any supported dtype (the optimizer-shaped case): a named, init-only set implementing `IOptimizerHyperparameters`, with defaults from `[Hyper(default)]`, each formatted at its declared dtype. Only a scalar can carry a default; a non-scalar hyperparameter's property is `required`. See [training.md](training.md). |

For `[TrainableParamInitializer] class ConstInit` with `Inline(Vector<int64> shape)`:
`ConstInit.Init(shape)` returns the initialized trainable `Tensor<T>`. (The class
must not itself be named `Init` — the generated `Init` member would collide with
the type name; the generator rejects that with error `MSG003`.)

An initializer may take inputs beyond the shape, and two of the shapes those take
are worth knowing: an `Init(...)` call **inside** an initializer body is the called
initializer's body evaluated as a value rather than a second parameter, so the
shipped parameterized initializers compose; and an input typed `Tensor<T>` may be
**another trainable parameter**, which reaches the body as the value that parameter
was initialized to. See *Writing your own* in
[nn-library.md](nn-library.md#initializers-shorokoomodulesinitializers).

## Hyperparameter baking

A concrete architecture has a **static parameter space**: which trainable
parameters exist, and what shape each one has, is fixed once and for all by
`ToConcreteArchitecture` — that is what makes it concrete, and it is what lets
weights bind, optimizers allocate state, and checkpoints round-trip. So the
dividing line between the two kinds of `[Hyper]` parameter is whether the hyper
touches that parameter space:

- **Parameter-space-determining** hyperparameters decide something about the
  trainable parameters themselves. They are **baked** when the graph is
  concretized (`ToConcreteArchitecture` — the step that prepares a model for
  training, weight binding, or export), from the value you supply there. There
  are two ways a hyper lands in this class, and the second is easy to miss:
  - It feeds a parameter's **shape** (or how many there are) — e.g.
    `outFeatures` in a dense layer, used as
    `ConstInit.Init([outFeatures, inFeatures])`. The parameters get fixed shapes
    from the value you supplied; a different value later does not resize them.
  - It **gates a branch that contains parameters** — e.g. `useBias` in
    `useBias.IfElse(y + b, y)`, where `b` is a trainable parameter. Which
    parameters *exist* is part of the parameter space, so the unselected
    branch's parameters are not created, and the `IfElse` that held them is
    resolved and folded away with them. Concretizing this layer with
    `useBias = false` yields an architecture with no bias parameter and no
    branch — passing the bit at `Execute` no longer changes that result.
    Concretizing with `useBias = true` prunes nothing, so nothing is folded and
    the bit still selects at run time. Nor does a gate fold when it does not
    *solely* own the parameters — a tuple `IfElse`, whose slots resolve together,
    or one sharing them with a second gate: there the parameters still go, and
    the branch that held them is left reading a zero stand-in. Note the scope
    either way: only the
    `IfElse` whose *pruned* parameters made it unreachable is folded, so the
    same hyper stays live for any other `IfElse` it gates — see
    [What concretization fixes](inference.md#what-concretization-fixes).
- **Value-only** hyperparameters (scale factors, momentum coefficients, ε's) do
  not touch any trainable parameter — neither its shape nor its existence. They
  are not baked: in the concretized graph they are read live at every `Execute`
  and may vary call to call, and in an optimizer they may be scheduled per-step
  (see [training.md](training.md)).

On the `Foo.ComputationGraph` route both kinds stay **live inputs** of the
concretized graph and must be supplied again at `Execute` (only
[`Specialize`](inference.md#hardcoding-hypers-with-specialize) removes an input;
via `Call`/`Model` the hyper is a constant and was never an input). For a
value-only hyper you may supply anything; for a parameter-space-determining one,
supply **the same value you concretized with**. Where it decided a parameter's
*shape*, a different value later does not resize anything. Where it gated a
parameter's *existence*, a different value either changes nothing (the branch
folded away with the parameters it held), or silently runs the other branch —
because nothing was pruned, or because the gate could not fold and that branch
now reads a zero stand-in for parameters that were. None of those is what you
meant. See
[What concretization fixes](inference.md#what-concretization-fixes) for the full
list of what a concrete architecture pins down.

How a hyper value gets supplied depends on the route:

- `Foo.Call(Scalar(k), x)` / `Foo.Model(Scalar(k))` — the hyper is embedded as a
  constant node in the built subgraph.
- `Foo.ComputationGraph` + concretize — **every** hyper stays an input of the
  concrete graph. The framework keeps the concrete graph's inputs ordered
  hyperparameters-first (independent of the inputs-first `Inline` source order), so
  `Execute` must be given the hyper values again — the hypers (in their `Inline`
  relative order) first, then the inputs. The values passed at concretization time bake the
  parameter space (parameter-space-determining hypers — shapes, count, and which
  parameters exist at all) and serve as shape/type hints; value-only hypers are
  read live at every `Execute`. Re-supply a baked hyper with the value it was
  concretized with. See
  [inference.md](inference.md#running-a-module-with-hyper-parameters) for the
  recipe.
- `Foo.ComputationGraph` + **`Specialize`** + concretize — to hardcode hypers
  (either kind) instead of re-supplying them at every `Execute`, call
  `Specialize` before `ToConcreteArchitecture`. It constant-folds the named hyper
  values into the graph and removes them from the input list, so the concrete
  model runs on the remaining inputs alone. See
  [inference.md](inference.md#hardcoding-hypers-with-specialize).

Together these make a `[Module]` class a *family* of architectures rather than one:
scale, depth and which parameters exist are all chosen by value, so N variants of a
model need one class, not N. See
[Workflow: one module, many variants](#workflow-one-module-many-variants).

## Workflow: add a new layer

1. Add `using Shorokoo; using Shorokoo.Modules; using static Shorokoo.Globals; using static Shorokoo.NN;`.
2. Declare `[Module] public partial class MyLayer`.
3. Write `public static <OutputType> Inline(<input tensors...>, <[Hyper] hypers...>)`.
4. Build the output from tensor ops, `NN.*` ops, sub-modules (`Other.Model(...).Call(...)`),
   and weights from a `[TrainableParamInitializer]` — whose own body may in turn call
   another initializer or start from a parameter already built here.
5. Ensure the project references the generator as an analyzer (see below).
6. Build, then call `MyLayer.Call(...)` or use `MyLayer.ComputationGraph`.

## Control flow inside `Inline`

- Conditional (data-dependent): `condition.IfElse(whenTrue, whenFalse)` where
  `condition` is `Scalar<bit>`. Tuples are supported.

  Both branches are *built*, but only the selected one *runs*. The branch
  expressions are ordinary C# arguments, so they are traced before the `IfElse`
  they belong to; Shorokoo then moves everything that serves one branch, and
  nothing else, inside that branch — so it lowers into the `If` node's own
  subgraph and executes only when the condition picks it. That covers whole
  loops and nested `IfElse`s, not just single ops. Work that feeds *both*
  branches, or that is read after the `IfElse` as well, stays outside and is
  computed once, as it must be.

  So a branch may hold an operation that would be invalid on the other path —
  unwrapping an `OptionalTensor` that is absent there is the usual case (see
  [Optional tensor inputs](#optional-tensor-inputs)) — and an expensive branch
  costs nothing when it is not taken.

  One exception, and it is about parameters rather than control flow: the
  parameter space of a concrete architecture is static, so it cannot depend on a
  value that only arrives at `Execute`. When a `[Hyper]` gates a branch holding
  trainable parameters, `ToConcreteArchitecture` creates only the selected
  branch's parameters — and when that leaves the *unselected* branch holding
  parameters that no longer exist, it folds that `IfElse` away too, so the bit
  no longer switches it at run time. Nothing else is folded: an `IfElse` whose
  selected branch holds the parameters keeps both branches, and so do a paramless
  one on the same hyper, a tuple `IfElse`, and one sharing its parameters with a
  second gate. See
  [Hyperparameter baking](#hyperparameter-baking).

  A hyper folded to a **constant** before lowering — by `Foo.Call(Scalar(k), x)`,
  or by [`Specialize`](inference.md#hardcoding-hypers-with-specialize) — is the
  unconditional case: a constant condition folds *every* `IfElse` on it,
  parameters or not. `Specialize` additionally drops the hyper from the graph's
  input list; on the `Call`/`Model` route it was never an input of the enclosing
  graph to begin with.

  ```csharp
  // Apply bias only when useBias is true — both branches are built, one runs.
  // (But b exists in the concrete architecture only if useBias was baked true.)
  var b = ConstInit.Init([outFeatures]).Vec();
  return useBias.IfElse(y + b, y);

  // Tuples are also supported — sort a pair data-dependently:
  var (lo, hi) = (a < b).IfElse((a, b), (b, a));
  ```

- Loops: `foreach (var ctx in LoopAPI.Iterate(count)) { ...; ctx.IterationIndex; }`
  where `count` is `Scalar<int64>`. **Prefer this to a plain C# `for` for any repetition
  in a model body** — not only when the count is a graph value. A plain `for` runs at
  trace time and leaves nothing of the repetition behind: each iteration's parameters
  become independent parameters numbered in trace order, so their names say how many
  `Init(...)` calls preceded them and nothing about which iteration they belong to. A
  `LoopAPI.Iterate` body carries the iteration index instead, and each iteration numbers
  its own parameters from `#0`:

  ```
  for (int i = 0; i < 3; i++)          foreach (var ctx in LoopAPI.Iterate(Scalar(3L)))
    TrainableParam#0.NormalDist#0        TrainableParam#0.Loop#0:0.NormalDist#0
    TrainableParam#0.NormalDist#1        TrainableParam#0.Loop#0:1.NormalDist#0
    TrainableParam#0.NormalDist#2        TrainableParam#0.Loop#0:2.NormalDist#0
  ```

  Those names are the keys of every checkpoint the model writes and the ids a naming
  scheme maps ([onnx-and-weights.md](onnx-and-weights.md#naming)), so the difference
  outlives the graph: under a plain `for` a parameter's index says how many `Init(...)` calls
  preceded it and nothing about which iteration it belongs to, so adding or removing one
  renumbers every parameter after it and silently re-points any name written against them.

  Fall back to a plain `for` only where `LoopAPI.Iterate` cannot express the stack — a
  body that genuinely differs from iteration to iteration. A uniform stack of layers is
  not that case.

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

  `ctx.Scan(v)` records `v` once per iteration and stacks the recordings into a
  single tensor with a new leading axis of length `count`, available after that loop.
  Scan whatever the body reads at that point — the carry before the body updates it,
  the carry after, the iteration index, a value from outside the loop:
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

  Nested loops scan on either context. An **enclosing** loop's `ctx.Scan`, called from
  inside a nested body, records once per *enclosing* iteration — the value that body ends
  the iteration with. What an inner loop's own scan cannot do is escape the enclosing loop;
  see [limitations.md](limitations.md).

  A local may also carry what another carry held **one iteration ago**. The loop identifies
  it and hands back the trailed carry's value from the start of the previous iteration:
  ```csharp
  var acc = x, prev = x, sum = Scalar(0.0f);
  foreach (var ctx in LoopAPI.Iterate(trips))
  {
      sum  = sum + prev;          // acc's value from the previous iteration
      prev = acc;
      acc  = acc + Scalar(1.0f);
  }
  ```
  Written bare like that, a lagged local works read inside the body or scanned, including
  inside a nested loop when the lagged local is created in that loop's enclosing body.
  Everything else — reading it after the loop, lagging a local the enclosing loop also
  carries, trailing something the loop does not carry, chaining two lag steps, or an alias
  seeded from outside the loop — is refused, naming the shape. Wrapping every such
  assignment as `prev = LoopAPI.Carry(acc)` answers them, and a local the body only writes
  needs `LoopAPI.Init` as well. See [limitations.md](limitations.md).

## Workflow: one module, many variants

A `[Module]` class is not one architecture — it is a **family**. The `Inline` body is
traced once and the graph is cached per method, but `[Hyper]` parameters are *symbolic
graph inputs*, so that one cached graph already covers every configuration its hypers
can take. A `Scalar<int64>` hyper sets a parameter's shape or a `LoopAPI.Iterate` trip
count; a `Scalar<bit>` hyper picks between algorithms and decides which trainable
parameters exist at all ([Hyperparameter baking](#hyperparameter-baking)). So "the same
model at a different scale" is a **value**, not a second class: write the family once and
pick a member by supplying hyper values.

A ViT-shaped classifier, parameterised over its width, depth, head count, class count,
and two structural choices — whether the transformer blocks carry biases, and whether the
classifier head is a single projection or a hidden-layer MLP:

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

Every knob is a hyper, so nothing about a variant is written in source. Note in
particular what a plain C# `for` could not have done: `numLayers` is a graph value
driving `LoopAPI.Iterate`, so the *number* of blocks — and with it the number of
trainable parameters — is chosen by value too.

### Picking a variant: `Specialize`

(`using Shorokoo.Modules;` does not pull in its `.Layers` / `.Initializers` /
`.Optimizers` children — `using` is not recursive — so the three snippets below assume the
block above.)

`Specialize` constant-folds the hyper values into the graph **and removes them from the
input list**, so what comes out is an ordinary single-input graph with nothing left to
re-supply ([inference.md](inference.md#hardcoding-hypers-with-specialize)):

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

`FromOrderedInputs` pairs values with the *leading* input names, and a graph's hyper
inputs come first (in `Inline` order) regardless of the inputs-first source signature —
so passing just the hyper values names them correctly.

The two graphs are different architectures, not two views of one. Concretized on the
same patch input, `tiny` carries **23** trainable parameters and `small` **68**: the
per-block count differs (`useBias = false` prunes the attention and FFN biases), the
number of blocks differs (2 versus 4), the widths differ, and so does the head — `wHead`
exists only when `useMlpHead = false`, `wHidden`/`wOut` only when it is true, and either
way the `IfElse` that held the unselected pair is gone.

Note that these values are **baked with `Specialize`**, not merely supplied as
concretization hints, so each condition is a constant before lowering and *every* `IfElse`
on it folds — including the `useBias` gates, which a plain concretization with
`useBias = true` would have left live
([Control flow inside `Inline`](#control-flow-inside-inline)).

### Training a variant

`TrainingRig.FromScratch` takes the specialized graph directly, so a variant trains with
no hypers to thread through `TrainStep`:

```csharp
var sample = new TensorDataModelParam("patches", ModelParamType.InputParam,
    TensorData([batch, numPatches, patchDim], /* … */));

var rig = TrainingRig.FromScratch(
    tiny, Losses.CrossEntropy, Optimizers.AdamW, [sample],
    new AdamWOptimizerHyperparameters { LearningRate = 3e-4f });
```

Swapping `tiny` for `small` is the whole diff between training the two. (Each variant is
still concretized and lowered separately, so this buys one source of truth, not a cheaper
build — see [What construction costs](training.md#what-construction-costs).)

### What must still be a C# argument

Hypers are graph values, which fixes the boundary:

- **`Inline`'s return type cannot vary.** A hyper is a value *inside* the graph, and a
  graph's output type is fixed when it is built, so a choice that changes the output's
  *type* — a loss reduced to a `Scalar<float32>` versus kept per-element as a
  `Tensor<float32>` — cannot be a hyper. Keep such a choice out of the model: `TrainingRig.FromScratch` takes the
  loss as a **separate graph**, so the reduction is picked there
  ([nn-library.md](nn-library.md#loss-configurable-knobs)), and the model module
  stays one class.
- **A `[Hyper]` is a graph value of a tensor element type — there is no enum hyper.** Where a knob is naturally an
  enum, either encode it as a `Scalar<int64>` and compare in-graph
  (`(mode > Scalar(3L)).IfElse(a, b)`), or make it a plain C# argument on a `static`
  helper that is *not* a `[Module]` — the shape `Recurrent.RNN` and `EmbeddingBag.Bag`
  take, since their knobs are topology-determining and baked at build time either way
  ([nn-library.md](nn-library.md#recurrent-layers)).
- **Both `IfElse` branches are built, and they must agree in *type*** — they are the two
  arguments of one call. Their *shapes* need not match: the selected branch's shape is the
  result's, so gating a branch that adds a token to the sequence is fine, and on a gate
  that is still live at run time the same compiled model returns whichever shape the bit
  selects.

## Omittable parameters (defaulted hypers & optional inputs)

`Inline` parameters are always written as ordinary, **non-nullable** types. The source
generator turns two kinds of parameter into **nullable, omittable** parameters on the
generated `Model` / `Call` surface, so callers can leave them out:

| `Inline` parameter | Generated `Model`/`Call` parameter | When omitted / `null` |
|---|---|---|
| `[Hyper(0.9f)] Scalar<float32> momentum` | `Scalar<float32>? momentum = null` | the attribute's default (`0.9f`) is used |
| `[Hyper(2)] Scalar<int32> accumSteps` | `Scalar<int32>? accumSteps = null` | the default is formatted at the declared dtype (`2`) |
| `[Hyper] Vector<float32> scales` | `Vector<float32> scales` | not omittable — only a scalar hyperparameter can carry a default |
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
optional API. Unwrapping is only valid on the present branch, and that is where it
runs: Shorokoo puts each branch inside the `If` that selects it (see
[Control flow](#control-flow-inside-inline)), so `TensorValue()` is never reached when the optional
is absent.

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

A model with an optional input trains like any other: pass the sample input as an
`OptionalTensorDataModelParam` to `TrainingRig.FromScratch`, and build each batch with
`rig.InputDef.FromOrderedData(...)`, which takes an `OptionalTensorData` in that field's
position. The sample's arrangement — present or absent — only sizes the rig's build-time
shape inference; it is not baked into the training step, so a rig built either way accepts
the same batches.

A training *step* runs on ONNX Runtime and carries the same limit as inference above: it
can feed a present optional, not an absent one. **The absent branch is therefore not
trainable today.** Supplying a tensor instead does not stand in for it — a tensor takes the
*present* arm of the body's `IfElse`, so it trains the wrong arm wherever the two arms
differ, silently. Train the present arrangement, and cover the absent branch on the
inference path through `QuickExecutionEngine`.

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
| `ConstInit.Init(shape)` | `Globals.CallTrainableParamInitializer(body, defaultName, isTrainable, shape)` |

### End-to-end example (define, call, train, export)

```csharp
using Shorokoo;
using Shorokoo.Modules;
using Shorokoo.Core.Factory;          // FastOnnxModelBuilder
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
//    (cached build; the readonly instance is handed out directly, no per-access clone).
var graph = ModuleFactory.ComputationGraph(
    (Func<Tensor<float32>, Tensor<float32>>)ScalarMultiply);

// 4. Train — TrainingRig consumes graphs, so nothing changes (see training.md).
//    Losses.* / Optimizers.* (namespace Shorokoo) are the same graphs as
//    L2Loss.ComputationGraph / SGDOptimizer.ComputationGraph, without the extra usings.
var rig = TrainingRig.FromScratch(graph, Losses.L2Loss,
    Optimizers.SGD, sampleInputs, 0.01f);

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
using Shorokoo.Core;   // Module<...> / CallbackModule<...> / GraphBuilder live here

// Body is inputs-first: Body(Tensor<float32> a, Tensor<float32> b, [Hyper] Scalar<float32> h)
new Module<Scalar<float32>, (Tensor<float32>, Tensor<float32>), Tensor<float32>>(
    (h, ins) => Body(ins.Item1, ins.Item2, h), Body);
```

### Constraints and ergonomics differences

- **Static, non-capturing bodies only.** The body is invoked once to build the graph
  and the result is cached per method, so a capturing lambda (or a delegate bound to
  an object instance) is rejected. Pass varying values as `[Hyper]` parameters or
  runtime inputs instead. Caching per method costs no generality: a `[Hyper]` is a
  symbolic graph input, so the single cached graph still covers every configuration its
  hypers can take — shapes, trip counts and which parameters exist included
  ([Workflow: one module, many variants](#workflow-one-module-many-variants)).
- **Flattened parameters.** Like `Inline` methods, bodies take one parameter per
  tensor — tuple-typed parameters are rejected; use the multi-parameter overloads.
- **No generated typed hyperparameter sets.** The `FooHyperparameters` classes
  (named, defaulted `Hyperparameter` properties implementing `IOptimizerHyperparameters`)
  are codegen-only. For optimizer-style scheduling, pass `Hyperparameter` /
  `Schedules.*` values positionally to `TrainingRig.FromScratch(...)` (in the
  optimizer's `[Hyper]` parameter order) — see [training.md](training.md).
- **Naming.** The module name defaults to the body's declaring class; pass the
  optional `name:` argument for lambdas or when you want the codegen-style class
  name in exports.
- Lower-level building blocks are public too if you need them:
  `GraphBuilder.BuildComputationGraphFromDelegate(...)` (uncached graph build)
  and the `Module<...>` / `CallbackModule<...>` constructors shown above.

## Anti-patterns

- Do not put `[Hyper]` parameters before input parameters; generation expects inputs
  first and hyperparameters last.
- Do not write nullable `Inline` parameters (`Tensor<T>?`); declare an
  `OptionalTensor<T>` (or a `[Hyper(default)]` scalar) and let the generator expose the
  omittable `Tensor<T>?` / nullable form to callers (see
  [Omittable parameters](#omittable-parameters-defaulted-hypers--optional-inputs)).
- Do not name a `[TrainableParamInitializer]`/`[StateInitializer]` class `Init`;
  the generated `Init(...)` member would collide with the type name (generator
  error `MSG003`).
- Do not forget `partial` on the class, or the `static` modifier on `Inline`.
- Do not use a plain C# `for`/`if` on graph values (`Scalar<int64>`/`Scalar<bit>`) when
  the count/condition is dynamic; use `LoopAPI.Iterate` / `.IfElse`.
- Do not write one `[Module]` class per configuration of a model — a different width,
  depth, or a toggled sub-layer is a `[Hyper]` value, not a new type
  ([Workflow: one module, many variants](#workflow-one-module-many-variants)).
- Do not stack layers with a plain C# `for` even when the trip count is a constant: the
  repetition is gone by the time the graph exists, and its parameters are left numbered
  in trace order rather than by iteration
  ([Control flow inside `Inline`](#control-flow-inside-inline)).
- Do not switch threads inside a module body (`async`/`await`, `Parallel.For`, callbacks
  run elsewhere): the body runs synchronously on a single thread, and calls like
  `Globals.StateUpdate` or `Rng.Pin` made from another thread throw.
- Do not reference the code generator as a normal project reference; it must be an
  analyzer (`OutputItemType="Analyzer"`).
