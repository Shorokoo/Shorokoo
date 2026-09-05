# Debugging graph lowering (`DebugRequests`, `ComputeContext.Progress`)

Related: [inference.md](inference.md) · [onnx-and-weights.md](onnx-and-weights.md) · [training.md](training.md)

Two facilities watch the same lowering pipeline from opposite ends. `DebugRequests` captures the
**graph** at chosen points, as compilable C#, and you read it after the call returns.
`ComputeContext.Progress` reports the **stage name** as the pipeline enters it, while the call is
still running — the one that answers "is this build alive?". Neither changes the graph the build
produces (though a progress handler that throws aborts the build that called it).

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

The `GraphCreationPoint` enum declares 13 values, but only these five are actually written today:

- `AfterInlineAllModulesAndFunctions` - After inlining all modules and functions
- `AfterProcessTrainableParameters` - After processing trainable parameters
- `AfterFirstSimplify` - After the first simplification pass
- `AfterExpandAutoGrad` - After autodiff expansion
- `FinalGraph` - The final concrete architecture graph

The remaining eight — `AfterProcessAllModelHyperparamRefs`, `AfterProcessModelSequences`,
`AfterProcessAccessibleModuleSetHyperparams`, `AfterUnrollModuleLoop`, `AfterSimplify`,
`AfterSimplifyTrainableParamInitializers`, `AfterLowerStateUpdateNodes`, `AfterSecondSimplify` —
are **silent no-ops**: requesting one writes no file and reports nothing
([#224](https://github.com/Shorokoo/Shorokoo/issues/224)). To see that a stage of the pipeline the
enum does not cover has been reached, watch the build instead (next section).

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

## Watching a build while it runs (`ComputeContext.Progress`)

`DebugRequests` tells you what the graph looked like at a stage — but only once the call returns,
which is no help when the question is whether a call that has been running for minutes is still
making progress. For that, attach a progress sink to the compute context: every stage is reported as
the pipeline enters it, so the last report names the stage the build is in.

```csharp
using Shorokoo.Graph;    // BuildProgress, SynchronousBuildProgress
using Shorokoo.Runtime;  // ComputeContext

var buildContext = new ComputeContext { Progress = new SynchronousBuildProgress(Console.WriteLine) };

var concreteArchitecture = graph.ToConcreteArchitecture(inputHints, buildContext);
```

```
[   0.0s] Concretize: Clone
[   0.0s] Concretize: ApplyIdentifierTemplates
[   0.4s] Concretize: InlineModulesAndFunctions
…
[   4.8s] Concretize: Simplify
[  12.0s] Concretize: LowerAttributeTensorOps
[  38.4s] Concretize: ExpandAutoGrad
[  44.1s] Concretize: SimplifyAfterAutoGrad
[  46.0s] Concretize: Done
```

(`…` marks stages elided here, not gaps in the output — every stage reports, and a lowering that
finishes ends on `Done`.)

The same context passed to `TrainingRig.FromScratch` as its `mergeContext` covers the whole rig
build — concretization, training-step composition and initialization — under one clock. See
[training.md](training.md#watching-a-long-build) for the full report shape (`BuildPhase`, `Stage`,
`Elapsed`), the phase order, which calls report, and why to prefer `SynchronousBuildProgress` over
`System.Progress<T>`.
