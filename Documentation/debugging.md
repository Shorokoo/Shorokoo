# Debugging graph lowering (`DebugRequests`)

Related: [inference.md](inference.md) · [onnx-and-weights.md](onnx-and-weights.md)

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
var debugRequests = new DebugRequests(new[]
{
    (GraphCreationPoint.AfterInlineAllModulesAndFunctions, "/tmp/debug/after_inline.cs"),
    (GraphCreationPoint.AfterProcessTrainableParameters, "/tmp/debug/after_trainable.cs"),
    (GraphCreationPoint.FinalGraph, "/tmp/debug/final.cs")
});

// Call ToConcreteArchitecture with debug requests
var concreteArchitecture = graph.ToConcreteArchitecture(inputHints, computeContext, debugRequests);
```

## Available Debug Points

The `GraphCreationPoint` enum provides the following options:

- `AfterInlineAllModulesAndFunctions` - After inlining all modules and functions
- `AfterProcessTrainableParameters` - After processing trainable parameters
- `AfterProcessAllModelHyperparamRefs` - After processing model hyperparameter references
- `AfterProcessModelSequences` - After processing model sequences
- `AfterProcessAccessibleModuleSetHyperparams` - After processing accessible module hyperparameters
- `AfterUnrollModuleLoop` - After unrolling each module loop (called multiple times)
- `AfterSimplify` - After simplification (called multiple times)
- `AfterSimplifyTrainableParamInitializers` - After simplifying trainable parameter initializers
- `AfterLowerStateUpdateNodes` - After lowering `StateUpdate` nodes
- `AfterFirstSimplify` - After the first simplification pass
- `AfterExpandAutoGrad` - After autodiff expansion
- `AfterSecondSimplify` - After the second simplification pass
- `FinalGraph` - The final concrete architecture graph

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
- Points like `AfterUnrollModuleLoop` and `AfterSimplify` may be triggered multiple times during processing
