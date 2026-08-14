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
///   - If the loop is a zero-trip one — maxIterations known and non-positive, or the initial
///     condition known false, on what would be the first iteration → stop.
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
///
/// A zero-trip loop terminates on its outputs alone: the body has already been walked (the
/// engine only discovers the pair here, at the close node), so its tensors stay in the store
/// for the static-discovery passes that read them, but the loop reports its initializers and
/// empty scan outputs — a <c>Loop</c> is a while, not a do-while.
/// </summary>
internal sealed class LoopCloseOp : QuickOp
{
    /// <summary>When neither maxIter nor continueWhen is statically known, we iterate this many times.</summary>
    public const int MaxIterationsForUnknownBounds = 4;

    /// <summary>
    /// When the graph supplies no iteration count at all the loop is unbounded — only the
    /// condition can stop it, and a condition that resolves to a constant <c>true</c> never
    /// will. The engine has to come back with shapes either way, so it walks this many
    /// iterations and then gives up. Larger than
    /// <see cref="MaxIterationsForUnknownBounds"/> because nothing here is unknown: the
    /// prefix is all the engine will ever see of the loop, so it is worth more of it.
    /// </summary>
    public const int MaxIterationsForUnboundedLoops = 100;

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
        FastNode node, InternalComputationGraph graph, Dictionary<FastNodeKey, FastNode> nodeByKey,
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

        var initialCondInput = inputs.Length > 1 ? inputs[1] as RuntimeTensor : null;
        bool? initialCondValue = null;
        if (initialCondInput?.BoolData is { Length: > 0 } ic) initialCondValue = ic[0];

        _iterationCountByOpenNode.TryGetValue(info.OpenNodeKey, out int iter);

        // ONNX Loop is a while, not a do-while: a trip count of 0 — or an initial condition
        // that is already false — means the body contributes nothing. The engine only
        // discovers the pair when it reaches this close node, so the body has already run;
        // its per-node tensors stay in the store (the trainable-param grid is realized from
        // them) but must not reach the loop's own outputs.
        bool zeroTrip = iter == 0
            && ((maxIter is long z && z <= 0) || initialCondValue == false);

        bool knownDone = zeroTrip || (maxIter is long m && iter + 1 >= m) || continueWhenValue == false;
        bool anyUnknown = !maxIterKnown || !continueWhenKnown;
        // No iteration-count input means the graph never bounded this loop, so the walk has
        // to be bounded here instead — otherwise a condition resolving to a constant true
        // (a real one, or one a body forwards from the open node's placeholder) loops the
        // engine forever, accumulating per-iteration history until the process dies.
        bool unbounded = maxIterInput is null;
        int iterationCap = unbounded ? MaxIterationsForUnboundedLoops : MaxIterationsForUnknownBounds;
        bool capReached = (unbounded || anyUnknown) && iter + 1 >= iterationCap;

        if (!knownDone && !capReached)
        {
            _iterationCountByOpenNode[info.OpenNodeKey] = iter + 1;
            return (BuildLoopBackResults(inputs, nLoop, iter + 1), true);
        }

        // Giving up at the cap is not the loop finishing: the values in hand belong to a
        // truncated prefix, so the outputs keep their shapes and lose their data.
        bool cappedOut = capReached && !knownDone;
        _iterationCountByOpenNode.Remove(info.OpenNodeKey);
        return (BuildTerminateResults(inputs, nLoop, nScan, zeroTrip ? 0 : iter + 1, zeroTrip, cappedOut), false);
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

    private static IRuntimeTensor[] BuildTerminateResults(
        IRuntimeTensor?[] inputs, int nLoop, int nScan, int totalIterations, bool zeroTrip, bool cappedOut)
    {
        var results = new IRuntimeTensor[nLoop + nScan];
        // Zero trips: the loop variables come back as their initializers (inputs[2 .. 1 + nLoop])
        // untouched by the body's discovery pass, and every scan output has a leading dim of 0.
        for (int i = 0; i < nLoop; i++)
            results[i] = zeroTrip
                ? PropagateLoopVar(inputs[2 + i])
                : MergeLoopVarAcrossIterations(inputs[3 + nLoop + i], cappedOut);
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

    private static IRuntimeTensor MergeLoopVarAcrossIterations(IRuntimeTensor? current, bool cappedOut)
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
                return MergeAcrossIterations(tensor, cappedOut);
            default:
                return RuntimeTensorFactory.Create(DType.Invalid, null);
        }
    }

    private static RuntimeTensor MergeAcrossIterations(RuntimeTensor current, bool cappedOut)
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
        // MergeShapes builds a data-free tensor, so a capped-out loop simply keeps it that
        // way: the last iteration walked is not the loop's result, and publishing its values
        // would hand downstream folding a confidently wrong constant.
        if (cappedOut)
            return merged with { History = current.History, IterationIndices = current.IterationIndices };

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
