using Shorokoo.Graph;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Utils;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Onnx;
using Shorokoo.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;

namespace Shorokoo.Core.Rng;

/// <summary>
/// Resolves concrete RNG stream keys <b>by executing their in-graph derivation</b> — never by
/// recomputing the key tree host-side (#136). Given derivation specs (a root key plus the
/// ModelId path still to be folded), it builds a throwaway graph that folds the key tree one
/// LEVEL at a time — a single batched split per level over all streams of equal depth — runs it
/// through the ordinary execution path, and reads the resulting key words back.
///
/// <para>The point is that there is <b>no second implementation</b> to drift: the key a caller
/// sees is produced by the same graph op that keys real draws. Note the key tree is currently
/// algorithm-INdependent by construction — <see cref="RngAlgorithms.GetFunction"/> pins
/// <c>split</c> to <see cref="RngAlgorithms.Default"/> so switching the draw algorithm never
/// re-keys a stream — so this resolver emits that same default split. When a custom algorithm
/// may own its <c>split</c> (issue #122), the algorithm name must be threaded through the spec
/// to here; executing the derivation rather than reimplementing it is what makes that a
/// one-line change instead of a second port of the key tree.</para>
///
/// <para>Batching by level is what makes this affordable: graph preparation is super-linear in
/// node count, so emitting one split per fold STEP made resolution scale with the number of
/// streams (thousands of function-call nodes). One call per level instead makes the graph — and
/// the cost — scale with ModelId depth alone, which is a single digit (#138). All inputs are
/// constants, so each level collapses to literals at session build.</para>
///
/// <para>Diagnostic-path only (the RNG stream report / pin skeleton): nothing in model
/// execution consumes these values.</para>
/// </summary>
internal static class RngKeyResolver
{
    /// <summary>
    /// Resolves one key per spec, in order. A spec with an empty fold path (an override, or
    /// SharedKey mode) resolves to its root words without emitting a split.
    /// </summary>
    public static IReadOnlyList<long[]> Resolve(
        IReadOnlyList<((uint k0, uint k1) root, IReadOnlyList<int> foldPath)> specs,
        ComputeContext? computeContext = null)
    {
        ArgumentNullException.ThrowIfNull(specs);
        if (specs.Count == 0) return [];

        // Specs needing no fold are answered directly — their root IS the key, so there is
        // nothing to derive (and no RNG computation involved in reading two stored words).
        var results = new long[specs.Count][];
        var pending = new List<int>();
        for (int i = 0; i < specs.Count; i++)
        {
            if (specs[i].foldPath.Count == 0)
                results[i] = [specs[i].root.k0, specs[i].root.k1];
            else
                pending.Add(i);
        }
        if (pending.Count == 0) return results;

        // One execution per DEPTH group: every stream at a given depth folds together, level
        // by level, so cost scales with the (small) set of distinct ModelId depths rather than
        // with the number of streams.
        foreach (var group in pending.GroupBy(i => specs[i].foldPath.Count))
            ResolveGroup(specs, [.. group], results, computeContext);
        return results;
    }

    /// <summary>
    /// Resolves one group of equal-depth specs in a single graph execution: the group's M roots
    /// enter as one <c>[2, M]</c> key block, and each tree LEVEL is one batched split
    /// (<see cref="RngAlgorithms.KindSplitBatch"/>) folding all M streams at once. So the graph
    /// holds ~2 nodes per level rather than 2 per fold step — the cost stops scaling with the
    /// number of streams and scales only with depth.
    /// </summary>
    private static void ResolveGroup(
        IReadOnlyList<((uint k0, uint k1) root, IReadOnlyList<int> foldPath)> specs,
        IReadOnlyList<int> group,
        long[][] results,
        ComputeContext? computeContext)
    {
        int m = group.Count;
        int depth = specs[group[0]].foldPath.Count;

        // Roots as one [2, M] block: row 0 = k0 words, row 1 = k1 words.
        var rootWords = new long[2 * m];
        for (int j = 0; j < m; j++)
        {
            rootWords[j] = specs[group[j]].root.k0;
            rootWords[m + j] = specs[group[j]].root.k1;
        }
        var nodes = new List<FastNode>();
        var keys = AppendConstant(new OnnxTensorData<int64>(
            new Shape(2, m), OnnxUtils.CreateTensorValue(new Shape(2, m), rootWords)), nodes);

        var batchSplit = RngAlgorithms.GetFunction(RngAlgorithms.Default, RngAlgorithms.KindSplitBatch);
        for (int level = 0; level < depth; level++)
        {
            var counters = new long[m];
            for (int j = 0; j < m; j++) counters[j] = specs[group[j]].foldPath[level];
            var countersKey = AppendConstant(new OnnxTensorData<int64>(
                new Shape(m), OnnxUtils.CreateTensorValue(new Shape(m), counters)), nodes);
            keys = AppendBatchSplit(keys, countersKey, batchSplit, nodes);
        }

        var graph = new InternalComputationGraph
        {
            Nodes = nodes,
            Inputs = [],
            InputUniqueNames = [],
            Outputs = [keys],
            OutputUniqueNames = [null],
        };

        NamedModelParam[] run;
        try
        {
            run = (computeContext ?? ComputeContext.Default).Run(graph);
        }
        catch (Exception ex)
        {
            // Resolving a key executes a graph, so this can fail where the old host-side fold
            // could not (no execution provider, session-build failure). Say which half failed:
            // the stream inventory itself is fine, only the key values are unavailable.
            throw new InvalidOperationException(
                "RngKeyResolver: failed to resolve RNG stream keys by executing their in-graph " +
                "derivation. The stream inventory (paths, kinds, names) is unaffected — only the " +
                "resolved key words require execution. See the inner exception.", ex);
        }

        // The unpacking below is positional, so a size mismatch would silently mis-label keys —
        // the one failure mode that looks plausible and that nothing downstream cross-checks.
        if (run.Length != 1)
            throw new InvalidOperationException(
                $"RngKeyResolver: expected 1 resolved key block, got {run.Length}.");
        var words = run[0].ToTensorData().As<int64>().AccessMemory().ToArray();
        if (words.Length != 2 * m)
            throw new InvalidOperationException(
                $"RngKeyResolver: expected a [2, {m}] key block ({2 * m} words), got {words.Length}.");

        for (int j = 0; j < m; j++)
            results[group[j]] = [words[j], words[m + j]];
    }

    /// <summary>One batched key-tree level: <c>[2, M] keys x [M] counters -> [2, M]</c>, as a call
    /// of the algorithm's non-inlined <c>splitBatch</c> function (never a host computation).</summary>
    private static FastTensorKey AppendBatchSplit(
        FastTensorKey keys, FastTensorKey counters, Function batchSplit, List<FastNode> nodes)
    {
        var attrDefs = Definitions.NodeDefinitions[InternalOpCodes.FUNCTION_INVOKE].AttributeDefs;
        var nodeKey = FastNodeKey.New();
        var outKey = new FastTensorKey(nodeKey, 0);
        nodes.Add(new FastNode
        {
            Key = nodeKey,
            OpCode = InternalOpCodes.FUNCTION_INVOKE,
            Attributes = OnnxCSharpAttributes.FromCSharpVals(
                new Dictionary<string, object?>
                {
                    [ShrkAttrStructure] = (DataStructure[])[DataStructure.Tensor],
                    [ShrkAttrDtype] = (DType[])[DType.Int64],
                    [ShrkAttrRank] = (long[])[2L],
                    [ShrkAttrGenericTypeArgs] = null,
                }, attrDefs),
            TargetFunction = batchSplit,
            FullInputs = { [""] = new List<FastTensorKey?> { keys, counters } },
            FullOutputs = { [""] = new List<FastTensorKey?> { outKey } },
        });
        return outKey;
    }

    private static FastTensorKey AppendConstant(TensorData data, List<FastNode> nodes)
    {
        var attrDefs = Definitions.NodeDefinitions[OpCodes.CONSTANT].AttributeDefs;
        var nodeKey = FastNodeKey.New();
        var outKey = new FastTensorKey(nodeKey, 0);
        nodes.Add(new FastNode
        {
            Key = nodeKey,
            OpCode = OpCodes.CONSTANT,
            Attributes = OnnxCSharpAttributes.FromCSharpVals(
                new Dictionary<string, object?> { [AttrValue] = data }, attrDefs),
            FullInputs = new Dictionary<string, List<FastTensorKey?>>(),
            FullOutputs = { [""] = new List<FastTensorKey?> { outKey } },
        });
        return outKey;
    }
}
