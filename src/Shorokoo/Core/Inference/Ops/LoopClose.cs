using System.Collections.Immutable;
using Shorokoo.Core.Graph;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Onnx;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// Close side of a <c>Loop</c> node pair. <see cref="Execute"/> pulls the paired open node's
/// inputs out of the store and prepends them so the pure <see cref="ComputeWithLoopBack"/>
/// sees a flat layout:
///   inputs[0]                           — maxIterations (may be null / no data)
///   inputs[1]                           — initial continue condition (may be null)
///   inputs[2 .. 1 + N_loop]             — initial loop-variable values
///   inputs[2 + N_loop]                  — continueWhen (close node's own break input)
///   inputs[3 + N_loop .. 2 + 2*N_loop]  — body loop variables (next-iteration values)
///   inputs[3 + 2*N_loop ..]             — body scan variables
///
/// <c>N_loop</c> is passed to <see cref="ComputeWithLoopBack"/> through a thread-local because
/// it cannot be derived from the flat array alone (the body-loopvar vs scan-var split is only
/// knowable by looking at the open node's input count).
///
/// Termination rules (inside <see cref="ComputeWithLoopBack"/>):
///   - If maxIterations is known and its last iteration has been completed → stop.
///   - If the continueWhen value is known false → stop.
///   - If bounds are unknown and we've already done <see cref="MaxIterationsForUnknownBounds"/>
///     iterations → stop (shape-inference heuristic).
///   - Otherwise → loop back.
///
/// On loop-back, emits <c>2 + N_loop</c> tensors matching the open node's outputs; the engine
/// maps them onto those outputs and jumps to the first body node. On terminate, emits
/// <c>N_loop + N_scan</c> tensors matching the close node's declared outputs; loop-var outputs
/// keep the final iteration's concrete data (and the per-iteration <c>History</c>) while also
/// carrying a shape merged across every recorded iteration, and scan outputs get rank+1 with
/// leading dim = iteration count.
/// </summary>
internal sealed class LoopCloseOp : QuickOp
{
    /// <summary>When neither maxIter nor continueWhen is statically known, we iterate this many times.</summary>
    public const int MaxIterationsForUnknownBounds = 4;

    public override string OpCode => OpCodes.LOOP_CLOSE;

    private readonly Dictionary<FastNodeKey, int> _iterationCountByOpenNode = new();

    // Thread-local carrier for per-invocation metadata that the flat Compute signature can't
    // hold. Execute sets this before calling ComputeWithLoopBack and clears it afterwards.
    [ThreadStatic] private static LoopInfo? _currentLoopInfo;

    private sealed class LoopInfo
    {
        public int NLoop;
        public FastNodeKey OpenNodeKey;
    }

    public override (IRuntimeTensor[] results, bool loopBack) Execute(
        FastNode node, FastComputationGraph graph, Dictionary<FastNodeKey, FastNode> nodeByKey,
        Dictionary<FastTensorKey, IRuntimeTensor> store, int maxDataElements)
    {
        FastNode? openNode = null;
        if (node.GraphOpenNodeKey is FastNodeKey openKey && !openKey.IsEmpty)
            nodeByKey.TryGetValue(openKey, out openNode);

        var openInputs = openNode is null
            ? Array.Empty<IRuntimeTensor?>()
            : GatherInputs(openNode.Inputs, store);
        var ownInputs = GatherInputs(node.Inputs, store);

        var merged = new IRuntimeTensor?[openInputs.Length + ownInputs.Length];
        Array.Copy(openInputs, 0, merged, 0, openInputs.Length);
        Array.Copy(ownInputs, 0, merged, openInputs.Length, ownInputs.Length);

        var info = new LoopInfo
        {
            NLoop = Math.Max(0, openInputs.Length - 2),
            OpenNodeKey = openNode?.Key ?? default,
        };
        _currentLoopInfo = info;
        try
        {
            return RunCompute(merged, node, maxDataElements);
        }
        finally { _currentLoopInfo = null; }
    }

    protected override (IRuntimeTensor[] results, bool loopBack) ComputeWithLoopBack(
        IRuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var info = _currentLoopInfo
            ?? throw new InvalidOperationException(
                "LoopCloseOp.ComputeWithLoopBack requires Execute-supplied loop context.");
        var nLoop = info.NLoop;

        // The termination decision only needs to inspect the plain-tensor bookkeeping inputs
        // (maxIter at inputs[0], continueWhen at inputs[2 + nLoop]). Loop/scan variables at
        // the end of the array can be of any IRuntimeTensor variant and flow through
        // untouched via Build{LoopBack,Terminate}Results.
        var maxIterInput = inputs.Length > 0 ? inputs[0] as RuntimeTensor : null;
        var continueWhenInput = inputs.Length > 2 + nLoop
            ? inputs[2 + nLoop] as RuntimeTensor
            : null;

        // Total = openInputs (2 + nLoop) + closeInputs (1 + nLoop + nScan). Therefore:
        var nScan = Math.Max(0, inputs.Length - (3 + 2 * nLoop));

        long? maxIter = null;
        if (maxIterInput?.IntData is { Length: > 0 } mi) maxIter = mi[0];
        bool maxIterKnown = maxIterInput is null || maxIter.HasValue;

        bool? continueWhenValue = null;
        if (continueWhenInput?.BoolData is { Length: > 0 } cw) continueWhenValue = cw[0];
        bool continueWhenKnown = continueWhenInput is null || continueWhenValue.HasValue;

        _iterationCountByOpenNode.TryGetValue(info.OpenNodeKey, out int iter);

        bool knownDone = (maxIter is long m && iter + 1 >= m) || continueWhenValue == false;
        bool anyUnknown = !maxIterKnown || !continueWhenKnown;
        bool capReached = anyUnknown && iter + 1 >= MaxIterationsForUnknownBounds;

        if (!knownDone && !capReached)
        {
            _iterationCountByOpenNode[info.OpenNodeKey] = iter + 1;
            return (BuildLoopBackResults(inputs, nLoop, iter + 1), true);
        }

        _iterationCountByOpenNode.Remove(info.OpenNodeKey);
        return (BuildTerminateResults(inputs, nLoop, nScan, iter + 1), false);
    }


    private static IRuntimeTensor[] BuildLoopBackResults(IRuntimeTensor?[] inputs, int nLoop, int nextIter)
    {
        var results = new IRuntimeTensor[2 + nLoop];

        var iterIdx = RuntimeTensorFactory.Create(DType.Int64, new Shape(Array.Empty<long>()));
        results[0] = iterIdx with { IntData = ImmutableArray.Create((long)nextIter) };

        results[1] = new RuntimeTensor
        {
            DType = DType.Bool,
            Shape = new Shape(Array.Empty<long>()),
            MaxShape = new Shape(Array.Empty<long>()),
            Rank = 0,
            MaxRank = 0,
            BoolData = ImmutableArray.Create(true),
        };

        // Body loop vars live at inputs[3 + nLoop .. 2 + 2*nLoop]. Each is whatever variant the
        // body produced this iteration (plain tensor, sequence, or optional); pass it through so
        // the next iteration observes the same structure.
        for (int i = 0; i < nLoop; i++)
        {
            results[2 + i] = PropagateLoopVar(inputs[3 + nLoop + i]);
        }
        return results;
    }

    private static IRuntimeTensor[] BuildTerminateResults(IRuntimeTensor?[] inputs, int nLoop, int nScan, int totalIterations)
    {
        var results = new IRuntimeTensor[nLoop + nScan];
        for (int i = 0; i < nLoop; i++)
            results[i] = MergeLoopVarAcrossIterations(inputs[3 + nLoop + i]);
        for (int i = 0; i < nScan; i++)
            results[nLoop + i] = BuildScanOutput(inputs[3 + 2 * nLoop + i] as RuntimeTensor, totalIterations);
        return results;
    }

    /// <summary>
    /// Mirror the body's per-iteration loop variable. Plain tensors are rebuilt (with shape and
    /// data, but stripped of iteration metadata that's only meaningful inside the body);
    /// sequences and optionals flow through as-is.
    /// </summary>
    private static IRuntimeTensor PropagateLoopVar(IRuntimeTensor? src) => src switch
    {
        null => RuntimeTensorFactory.Create(DType.Invalid, null),
        RuntimeSequenceTensor seq => seq,
        RuntimeOptionalTensor opt => opt,
        RuntimeTensor t => RuntimeTensorFactory.Create(t.DType, t.Shape) with
        {
            MaxShape = t.MaxShape ?? t.Shape,
            Rank = t.Rank,
            MaxRank = t.MaxRank,
            FloatData = t.FloatData,
            IntData = t.IntData,
            BoolData = t.BoolData,
            StringData = t.StringData,
        },
        _ => RuntimeTensorFactory.Create(DType.Invalid, null),
    };

    private static IRuntimeTensor MergeLoopVarAcrossIterations(IRuntimeTensor? current)
    {
        switch (current)
        {
            case null:
                return RuntimeTensorFactory.Create(DType.Invalid, null);
            case RuntimeSequenceTensor seq:
                // The ONNX loop output equals the body's final-iteration value — a sequence
                // stays a sequence with its existing per-iteration History preserved.
                return seq;
            case RuntimeOptionalTensor opt:
                return opt;
            case RuntimeTensor tensor:
                return MergeAcrossIterations(tensor);
            default:
                return RuntimeTensorFactory.Create(DType.Invalid, null);
        }
    }

    private static RuntimeTensor MergeAcrossIterations(RuntimeTensor current)
    {
        // Shape across every recorded iteration (prior iterations in History + the current one).
        // Even if shapes diverge across iterations, the loop's output is the final iteration's
        // value — so we keep current's concrete data and History and only widen the shape.
        var shapes = new List<Shape?>();
        if (current.History is { } hist)
            foreach (var h in hist)
                if (h is RuntimeTensor rt) shapes.Add(rt.Shape);
        shapes.Add(current.Shape);

        var merged = MergeShapes(current.DType, shapes);
        return merged with
        {
            FloatData = current.FloatData,
            IntData = current.IntData,
            BoolData = current.BoolData,
            StringData = current.StringData,
            History = current.History,
            IterationIndices = current.IterationIndices,
        };
    }

    private static RuntimeTensor BuildScanOutput(RuntimeTensor? current, int numIterations)
    {
        if (current is null) return RuntimeTensorFactory.Create(DType.Invalid, null);

        var shapes = new List<Shape?>();
        if (current.History is { } hist)
            foreach (var h in hist)
                if (h is RuntimeTensor rt) shapes.Add(rt.Shape);
        shapes.Add(current.Shape);

        var innerMerged = MergeShapes(current.DType, shapes);
        var dtype = current.DType;

        if (innerMerged.Shape is not null)
        {
            var dims = new long[innerMerged.Shape.Dims.Length + 1];
            dims[0] = numIterations;
            Array.Copy(innerMerged.Shape.Dims, 0, dims, 1, innerMerged.Shape.Dims.Length);
            var shape = new Shape(dims);
            var rt = RuntimeTensorFactory.Create(dtype, shape);
            return rt with { MaxShape = shape, Rank = dims.Length, MaxRank = dims.Length };
        }

        var fallback = RuntimeTensorFactory.Create(dtype, null);
        Shape? fallbackMax = fallback.MaxShape;
        if (innerMerged.MaxShape is not null)
        {
            var mdims = new long[innerMerged.MaxShape.Dims.Length + 1];
            mdims[0] = numIterations;
            Array.Copy(innerMerged.MaxShape.Dims, 0, mdims, 1, innerMerged.MaxShape.Dims.Length);
            fallbackMax = new Shape(mdims);
        }
        return fallback with
        {
            MaxShape = fallbackMax,
            Rank = innerMerged.Rank is int r ? r + 1 : fallback.Rank,
            MaxRank = innerMerged.Rank is int r2 ? r2 + 1 : fallback.MaxRank,
        };
    }

    private static RuntimeTensor MergeShapes(DType dtype, List<Shape?> shapes)
    {
        var known = shapes.Where(s => s is not null).Cast<Shape>().ToList();
        if (known.Count == 0)
            return RuntimeTensorFactory.Create(dtype, null);

        var rank = known[0].Dims.Length;
        if (!known.All(s => s.Dims.Length == rank))
            return RuntimeTensorFactory.Create(dtype, null);

        var exact = (long[])known[0].Dims.Clone();
        var max = (long[])known[0].Dims.Clone();
        bool allEqual = true;
        foreach (var s in known.Skip(1))
        {
            for (int d = 0; d < rank; d++)
            {
                if (s.Dims[d] != exact[d]) { exact[d] = -1; allEqual = false; }
                if (s.Dims[d] > max[d]) max[d] = s.Dims[d];
            }
        }

        var outShape = allEqual ? new Shape(exact) : null;
        var rt = RuntimeTensorFactory.Create(dtype, outShape);
        return rt with { MaxShape = new Shape(max), Rank = rank, MaxRank = rank };
    }
}
