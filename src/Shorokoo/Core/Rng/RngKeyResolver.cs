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
/// ModelId path still to be folded), it builds one throwaway graph of <c>SHRK_RNG_SPLIT</c>
/// chains, runs it through the ordinary execution path (which lowers each split to the
/// registered <c>split</c> function), and reads the resulting key words back.
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
/// <para>All inputs are constants, so the whole batch collapses to literals at session build;
/// one run resolves every requested stream.</para>
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

        var graph = new InternalComputationGraph();
        var nodes = new List<FastNode>();
        var outputs = new List<FastTensorKey>();
        foreach (var i in pending)
        {
            var (root, foldPath) = specs[i];
            var key = AppendConstant(new OnnxTensorData<int64>(
                new Shape(2), OnnxUtils.CreateTensorValue(new Shape(2), (long[])[root.k0, root.k1])), nodes);
            foreach (var v in foldPath)
            {
                var counter = AppendConstant(new OnnxTensorData<int64>(
                    new Shape(Array.Empty<long>()),
                    OnnxUtils.CreateTensorValue(new Shape(Array.Empty<long>()), (long[])[v])), nodes);
                key = AppendSplit(key, counter, nodes);
            }
            outputs.Add(key);
        }

        graph.Nodes = nodes;
        graph.Inputs = [];
        graph.InputUniqueNames = [];
        graph.Outputs = outputs;
        graph.OutputUniqueNames = outputs.Select(_ => (string?)null).ToList();

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

        // The mapping below is positional, so a length mismatch would silently mis-label keys —
        // the one failure mode that looks plausible and that nothing downstream cross-checks.
        if (run.Length != pending.Count)
            throw new InvalidOperationException(
                $"RngKeyResolver: expected {pending.Count} resolved key(s), got {run.Length}.");

        for (int j = 0; j < pending.Count; j++)
        {
            var words = run[j].ToTensorData().As<int64>().AccessMemory().ToArray();
            if (words.Length != 2)
                throw new InvalidOperationException(
                    $"RngKeyResolver: a resolved key must be 2 words, got {words.Length}.");
            results[pending[j]] = [words[0], words[1]];
        }
        return results;
    }

    private static FastTensorKey AppendSplit(
        FastTensorKey key, FastTensorKey counter, List<FastNode> nodes)
    {
        var attrDefs = Definitions.NodeDefinitions[InternalOpCodes.SHRK_RNG_SPLIT].AttributeDefs;
        var nodeKey = FastNodeKey.New();
        var outKey = new FastTensorKey(nodeKey, 0);
        nodes.Add(new FastNode
        {
            Key = nodeKey,
            OpCode = InternalOpCodes.SHRK_RNG_SPLIT,
            Attributes = OnnxCSharpAttributes.FromCSharpVals(
                new Dictionary<string, object?> { [ShrkAttrRngAlgorithm] = RngAlgorithms.Default },
                attrDefs),
            FullInputs = { [""] = new List<FastTensorKey?> { key, counter } },
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
