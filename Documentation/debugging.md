# Debugging graph lowering (`DebugRequests`, `BuildProgress`)

Related: [inference.md](inference.md) · [onnx-and-weights.md](onnx-and-weights.md) · [training.md](training.md)

Two facilities watch the same lowering pipeline from opposite ends. `DebugRequests` captures the
**graph** at chosen points, as compilable C#, and you read it after the call returns; a progress sink
reports the **stage name** as the pipeline enters it, while the call is still running — the one that
answers "is this build alive?". Neither changes the graph the build produces
(though a progress handler that throws aborts the build that called it).

When `ToConcreteArchitecture` doesn't produce the graph you expect, the
`DebugRequests` class (namespace `Shorokoo.Graph`) saves snapshots of the
graph at chosen points of the lowering pipeline, as compilable C# (the same
`SaveToCSharp()` form used elsewhere), so you can diff stages and find where
things go wrong. For inspecting *values* rather than graph structure, see the
QuickExecutionEngine debugging engine in [inference.md](inference.md).

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
points are listed here in the order the pipeline reaches them — they do not cover every stage, so to
see that a stage with no point of its own has been reached, watch the build instead
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

The names are historical, so they do not always match the stage names a progress sink reports:
`AfterFirstSimplify` follows the stage reported as `Simplify`, and `AfterProcessTrainableParameters`
the one reported as `ConvertModelParamIdRefToModelParam`.

## Alternative Construction

You can also construct with a dictionary:

```csharp
var debugDict = new Dictionary<GraphCreationPoint, string>
{
    [GraphCreationPoint.AfterInlineAllModulesAndFunctions] = "/tmp/debug/after_inline.cs",
    [GraphCreationPoint.FinalGraph] = "/tmp/debug/final.cs"
};

var debugRequests = new DebugRequests(debugDict);
```

## Notes

- Debug files are saved as C# code using the existing `SaveToCSharp()` functionality
- Directories are automatically created if they don't exist
- Passing `null` for `debugRequests` parameter works normally (no debug output)

## Watching a build while it runs (`BuildProgress`)

`DebugRequests` tells you what the graph looked like at a stage — but only once the call returns,
which is no help when the question is whether a call that has been running for minutes is still
making progress. For that, hand the call a progress sink: every stage is reported as the pipeline
enters it, so the last report names the stage the build is in.

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
[  38.4s] Concretize: ExpandAutoGrad
[  44.1s] Concretize: SimplifyAfterAutoGrad
[  45.2s] Concretize: Freeze
[  46.0s] Concretize: Done
```

(`…` marks stages elided here, not gaps in the output — every stage reports, and a lowering that
finishes ends on a report whose `IsComplete` is true.)

The same sink passed to `TrainingRig.FromScratch` covers the whole rig build — concretization,
training-step composition and initialization — under one clock. See
[training.md](training.md#watching-a-long-build) for the full report shape (`BuildPhase`, `Stage`,
`Elapsed`, `IsComplete`), the phase order, which calls report, and why to prefer
`SynchronousBuildProgress` over `System.Progress<T>`.
