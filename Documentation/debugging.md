# Debugging graph lowering (`DebugRequests`, `BuildProgress`)

Related: [inference.md](inference.md) · [onnx-and-weights.md](onnx-and-weights.md) · [training.md](training.md)

Two facilities watch the lowering pipeline. `DebugRequests` captures the **graph** at chosen
points, and you read it after the call returns; a progress sink reports the **stage name** as the
pipeline enters it, while the call is still running, so you can tell whether a build is alive.
Neither changes the graph the build produces, though a progress handler that throws aborts the
build that called it. What a backend's session runs can differ from that graph by the backend's
own kernel workarounds, which apply after the pipeline ends ([below](#backend-kernel-workarounds)).

When `ToConcreteArchitecture` doesn't produce the graph you expect, `DebugRequests` (namespace
`Shorokoo.Graph`) saves snapshots of the graph at chosen points of the lowering pipeline as
compilable C# (the `SaveToCSharp()` form), so you can diff stages. One graph shape cannot be
written out; see
[limitations.md](limitations.md#c-emission-of-a-runtime-built-tensorstruct). For inspecting
*values* rather than graph structure, see the QuickExecutionEngine debugging engine in
[inference.md](inference.md).

## Basic Usage

```csharp
using Shorokoo.Graph;

// Create debug requests specifying which graphs to save and where
var debugRequests = new DebugRequests(
[
    (GraphCreationPoint.AfterInlineAllModulesAndFunctions, "/tmp/debug/after_inline.cs"),
    (GraphCreationPoint.AfterProcessTrainableParameters, "/tmp/debug/after_trainable.cs"),
    (GraphCreationPoint.FinalGraph, "/tmp/debug/final.cs")
]);

// Call ToConcreteArchitecture with debug requests
var concreteArchitecture = graph.ToConcreteArchitecture(inputHints, computeContext, debugRequests);
```

## Available Debug Points

Every `GraphCreationPoint` names a pass that runs, so every point you request writes its file. The
points are listed in pipeline order. They do not cover every stage; to see that a stage with no
point of its own has been reached, watch the build
([below](#watching-a-build-while-it-runs-buildprogress)).

- `AfterApplyIdentifierTemplates` — After local `ModelId`s are assigned. A graph built from modules
  already carries them, so this snapshot is normally the graph you passed in: the baseline to diff
  the later points against
- `AfterInlineAllModulesAndFunctions` — After every sub-module and function is inlined
- `AfterInjectRngExecutionCounter` — After the model-global RNG execution counter
  (`RngExecutionCounter`, see [rng-configuration.md](rng-configuration.md)) is wired into every
  runtime random feed
- `AfterConvertToIdRefModelParams` — After parameter references become id-refs
- `AfterUnpackModelStruct` — After model structs, model hyperparameters and model ids are unpacked
- `AfterUnpackTensorStructs` — After tensor structs are unpacked into their individual tensors
- `AfterProcessTrainableParameters` — After those id-refs are resolved into the trainable parameters
  themselves
- `AfterFirstSimplify` — After the first simplification pass: constant folding, loop unrolling and
  branch selection
- `AfterLowerAttributeTensorOps` — After Shorokoo's operator variants (e.g. `SHRK_CONV`) are lowered
  to their standard ONNX counterparts
- `AfterExpandAutoGrad` — After autodiff expansion
- `FinalGraph` — The final concrete architecture graph, as returned

The point names do not always match the stage names a progress sink reports:
`AfterFirstSimplify` follows the stage reported as `Simplify`, and `AfterProcessTrainableParameters`
the one reported as `ConvertModelParamIdRefToModelParam`.

## Alternative Construction

`DebugRequests` also takes a dictionary:

```csharp
var debugDict = new Dictionary<GraphCreationPoint, string>
{
    [GraphCreationPoint.AfterInlineAllModulesAndFunctions] = "/tmp/debug/after_inline.cs",
    [GraphCreationPoint.FinalGraph] = "/tmp/debug/final.cs"
};

var debugRequests = new DebugRequests(debugDict);
```

## Notes

- Debug files are C# written by `SaveToCSharp()`
- Missing directories are created
- Passing `null` for `debugRequests` produces no debug output

## Watching a build while it runs (`BuildProgress`)

`DebugRequests` shows the graph only once the call returns. To see whether a call that has been
running for minutes is still making progress, hand it a progress sink: every stage is reported as
the pipeline enters it, so the last report names the stage the build is in.

```csharp
using Shorokoo.Graph;    // SynchronousBuildProgress

var concreteArchitecture = graph.ToConcreteArchitecture(
    inputHints, progress: new SynchronousBuildProgress(p => Console.WriteLine(p)));
```

```
[   0.0s] Concretize: Thaw
[   0.1s] Concretize: Clone
[   0.1s] Concretize: ApplyIdentifierTemplates
…
[   4.8s] Concretize: Simplify
[  12.0s] Concretize: LowerAttributeTensorOps
[  37.9s] Concretize: RejectOversizedConvTransposeOutputShape
[  38.4s] Concretize: ExpandAutoGrad
[  44.1s] Concretize: SimplifyAfterAutoGrad
[  45.2s] Concretize: Freeze
[  46.0s] Concretize: Done
```

(`…` marks stages elided here; every stage reports, and a lowering that finishes ends on a report
whose `IsComplete` is true.)

The same sink passed to `TrainingRig.FromScratch` covers the whole rig build (concretization,
training-step composition and initialization) under one clock. See
[training.md](training.md#watching-a-long-build) for the full report shape (`BuildPhase`, `Stage`,
`Elapsed`, `IsComplete`), the phase order, which calls report, and why to prefer
`SynchronousBuildProgress` over `System.Progress<T>`.

## Backend kernel workarounds

A runtime's kernels sometimes compute an operator otherwise than the ONNX spec says — a wrong value
in one corner, a missing element type, a crash on an empty input. For each such corner a backend can
name a **kernel workaround**: a rewrite of the call into an equivalent one its kernels compute as the
spec does. The workarounds a backend's sessions are built with form one ordered set, which the
backend names through `IShorokooBackend.KernelWorkaroundSet` (the names are in
`KernelWorkaroundSets`). The ONNX Runtime backend names `KernelWorkaroundSets.OnnxRuntime`; the
[PyTorch](pytorch-backend.md) and [JAX](jax-backend.md) backends name none.

The pass runs at one point only: when the model handed to a backend's session is built, after the
lowering pipeline above has produced the graph. It runs on the session's own copy, over the main
graph and every function, loop and branch body, and applies to a graph you built and an imported
model alike. Nothing else sees it — the graph `ToConcreteArchitecture` returns, every `DebugRequests`
snapshot, an ONNX export, generated C# and a saved `.srk` all keep each operator as written, so a
model built on one backend carries nothing of another backend's workarounds. Each workaround fires
only in the scenario its runtime gets wrong; every other call reaches the session unchanged.

No debug point captures the session's model, since it is built after the pipeline ends. To tell
whether a call of yours is rewritten, match it against the table below; the notes in
[operator-support.md](operator-support.md) give each case in full.

The ONNX Runtime set, in the order it applies — a later workaround sees what an earlier one built,
so the `Where`s the pool and `Range` rewrites emit are covered by the last row:

| Operator | Scenario ONNX Runtime gets wrong | Rewrite | Issue |
|---|---|---|---|
| Every `Reduce*` | Negative axes on an empty input | Axes made non-negative, as a constant or in the graph | [#422](https://github.com/Shorokoo/Shorokoo/issues/422) |
| Every `Reduce*` | `noop_with_empty_axes` set with no axes or an empty axes tensor, on an empty input | The same reduction with `keepdims` 0 and the reduced axes put back by `Unsqueeze`, an empty input viewed with a trailing axis of one that alone is reduced, with no branch | [#409](https://github.com/Shorokoo/Shorokoo/issues/409) |
| `ReduceMax`, `ReduceMin` | A reduced axis of extent 0 of a bool input (the kernel throws) | A uint8 `ReduceMax` cast back, over the negated input and negated back for `ReduceMin`, with no branch | [#382](https://github.com/Shorokoo/Shorokoo/issues/382) |
| `MaxPool` | `Indices` read over int8 or uint8, with a window holding only the type's lowest value | Pooled as float32, values cast back | [#420](https://github.com/Shorokoo/Shorokoo/issues/420) |
| `AveragePool`, `LpPool`, `MaxPool` | `SAME_UPPER`/`SAME_LOWER` with a dilation above 1 or a stride above the kernel; explicit pads as large as the kernel | `Pad`, a pool with padding ONNX Runtime accepts, and `Slice` | [#379](https://github.com/Shorokoo/Shorokoo/issues/379), [#408](https://github.com/Shorokoo/Shorokoo/issues/408) |
| `ConvTranspose` | `SAME_UPPER`/`SAME_LOWER` with a stride above the kernel's extent plus `output_padding` (the output is cut to the full transposed convolution) | The call without padding or bias, `Pad` by the negated SAME padding (zeros past the full extent), then the bias | [#444](https://github.com/Shorokoo/Shorokoo/issues/444) |
| `Col2Im` | One spatial axis | `Col2Im` over two axes, the second of extent 1, then `Squeeze` | [#381](https://github.com/Shorokoo/Shorokoo/issues/381) |
| `Resize` | `tf_crop_and_resize` along an axis whose length the resize leaves unchanged | That axis resized to `2L−1` under the same `roi`, every second element kept | [#380](https://github.com/Shorokoo/Shorokoo/issues/380) |
| `Resize` | Cubic rank-4 `tf_crop_and_resize` with scale 1 on axes 0 and 3 and not on axis 1 | The input regrouped as `[N·W, 1, C, H]`, resized over its last two axes (a `not_larger`/`not_smaller` policy kept over the regrouped axes), regrouped back | [#421](https://github.com/Shorokoo/Shorokoo/issues/421) |
| `Resize` | An `axes` attribute: the transpose optimizer misreads its per-axis operands, and the kernel refuses negative axes | Written out over every axis; a `not_larger`/`not_smaller` policy over a subset of the axes keeps its axes, counted from the front, behind an `OptionalGetElement(Optional(x))` the optimizer cannot move a `Transpose` through | [#429](https://github.com/Shorokoo/Shorokoo/issues/429) |
| `Range` | int64 with `start` or `limit` beyond 2^53, or `limit − start` beyond 2^53: the kernel converts both ends to double precision and counts the elements in double precision, which rounds there, so even a short `Range` can get a wrong count (`Range(2^62, 2^62 + 2, 1)` gives none); int32 of three `Constant`s whose `limit − start` is beyond int32 | The count as an exact uint64 ceiling division (0 where `limit` does not lie beyond `start` in the direction of `delta`; a count above 2^62 taken as 2^62, which is refused), then `Range(0, count, 1) · delta + start`, an int32 call computed on its inputs cast to int64 and cast back. Left alone: three `Constant`s within 2^53 of 0 and less than 2^53 apart; a `Constant` `delta` of ±1 with a `Constant` `start` or `limit` within 2^52 of 0, as in `Range(0, n, 1)`, whose count is exact or one no tensor holds; every other int32 call | [#447](https://github.com/Shorokoo/Shorokoo/issues/447), [#450](https://github.com/Shorokoo/Shorokoo/issues/450) |
| `MatMul` | An empty operand, with a contraction dimension of 0, where the product is batched or of a matrix with a vector: the kernel leaves the output unwritten, and gives it the left operand's batch dimension where that is 1 and the right one's is not; a `Transpose` fused into a batched `FusedMatMul` leaves it unwritten too | With every input's dimensions stated, outside a loop body, the call's result goes through an `If` on either operand or the product being empty, the empty branch giving zeros of the product's shape, which ONNX Runtime folds when it builds the session wherever the shapes follow from the stated dimensions; the call runs before the `If`, which reads only the operands' shapes and the product, so a training step still writes an updated parameter over one the call reads; a `Constant` operand with a dimension of 0: those zeros, with no branch. A matrix or vector times a matrix, a vector times a vector, two nonempty `Constant`s, a call in a loop body, and every call of a session built without stated dimensions are left alone (see below) | [#451](https://github.com/Shorokoo/Shorokoo/issues/451) |
| `Where` | int8, int16, uint16, uint32, uint64, bfloat16 or bool values (no kernel) | bool: `Or(And(c, x), And(Not(c), y))`; others: selected through int32, int64 or float32 and cast back | [#423](https://github.com/Shorokoo/Shorokoo/issues/423) |

A call whose input is a scalar or a nonempty `Constant` cannot hit the empty-input rows and is left
alone. The `MatMul` rewrite is the one that branches: its `If` reads the operands' and the
product's shapes, and ONNX Runtime folds it away when it builds the session wherever those shapes
follow from the model's stated input dimensions. An `If` over a shape that depends on the data —
the output of a `NonZero`, of a `TopK` with a computed `k`, of a `Reshape` or `Expand` to a
computed shape — is not folded: it runs on every run, at the cost of the few shape operations of its
condition and the branch it takes. In the body of a `Loop` or `SequenceMap` the shapes are known
only as the body runs, so the `MatMul` is left as written there; an `If` there would run on every
iteration. The other rewrites have no branch: one graph, built from the input's shape and axes, is
right for every input.

The faults below are left alone — wholly, or in the cases a row of the table leaves out — since
telling those cases apart would cost ordinary calls of the operator. They are accepted, and a model
that meets one gets ONNX Runtime's result:

- A `MatMul`, or the `FusedMatMul` ONNX Runtime fuses a `Transpose` into, with an operand of a
  dimension 0, in a session built without every input's dimensions stated or in a loop body: other
  than a matrix or vector times a matrix and a vector times a vector, the result can be unwritten
  memory or of a wrong shape (a left batch dimension of 1 against a right one of 2 gives a batch of
  1). With the dimensions stated, as a training step's session is for the shapes it is fed, the
  `MatMul` row of the table corrects it ([#451](https://github.com/Shorokoo/Shorokoo/issues/451)).
  Whatever the dimensions, the kernel still runs on the empty operands: it refuses an empty batch
  against an operand with no batch dimension or one of 1 (`cannot broadcast`), a `FusedMatMul`
  moving the batch axis of rank-3 operands can stop the process (SIGFPE), and where the product
  of a left batch of 1 is wrongly shaped, a run can fail as ONNX Runtime reuses its memory
  (`Shape mismatch attempting to re-use buffer`).
- A float32, float64 or float16 `Where` gives +0 where it selects −0 from `x`; a −0 it selects from
  `y` keeps its sign. A bfloat16 `Where`, selected through float32, does the same
  ([#439](https://github.com/Shorokoo/Shorokoo/issues/439)).
- A model that returns a `Range` as an output beside a `Gather` the `Range` drives fails: ONNX
  Runtime's graph optimization removes the `Range`, and reading the outputs throws
  `UnsupportedDTypeException` (`OU002`) ([#432](https://github.com/Shorokoo/Shorokoo/issues/432)).
- A float `MaxPool` whose `Indices` output is read, with pads as large as the kernel, is pooled
  over −inf padding written into the input (the `AveragePool`, `LpPool`, `MaxPool` row), so a
  window whose input elements are all −inf takes a padded position before them as its first
  maximum: its index names the padding, such as −1. The values are right
  ([#437](https://github.com/Shorokoo/Shorokoo/issues/437)).
- On ONNX Runtime's CPU provider, an int32 or int64 `ReduceSum`, `ReduceMean`, `ReduceProd`,
  `ReduceL1` or `ReduceSumSquare` accumulates in double precision and clamps the result to the type's range, where Shorokoo's
  folding wraps as two's complement: int32 `ReduceSum([2147483647, 2147483647])` gives 2147483647,
  not −2, and an int64 group is rounded once its accumulation passes 2^53.
