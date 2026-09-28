using Shorokoo.Core.Nodes.NodeDefinitions;
using System;
using Shorokoo.Core.Graph;
using System.Collections.Generic;
using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Nodes.Processors.AutoGrad;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// Rewrites every <see cref="FastNode.Key"/> in <c>graph</c> to
    /// a fresh sequential <see cref="FastNodeKey"/> (1, 2, 3, …) and rewrites
    /// every <see cref="FastTensorKey"/> reference (inputs, outputs, graph
    /// inputs/outputs, <see cref="FastNode.GraphOpenNodeKey"/>) accordingly.
    /// Each node also gets its <see cref="FastNode.FriendlyName"/> set to
    /// <c>"N{i}"</c> for downstream consumers that read names by string.
    ///
    /// <para>
    /// After this pass, <see cref="FastTensorKey.ToString"/> on any tensor in
    /// the graph produces the canonical <c>"N{i}_T{j}"</c> form, letting the
    /// Fast ONNX builder produce ONNX-friendly names directly from keys
    /// without any string-rewriting step.
    /// </para>
    ///
    /// <para>
    /// Function bodies are not touched — the counter is per-graph. Callers that
    /// want a globally unique counter across main graph + all functions must
    /// invoke this on each graph separately and supply their own external
    /// counter via <see cref="ProcessWithCounter"/>.
    /// </para>
    ///
    /// <para>
    /// A caller may say how to number a graph a pass spliced nodes into (<see cref="Numbering"/>):
    /// the spliced nodes last, and the number of each node the pass dropped left unused where it
    /// stood, so that every other node keeps the number, and every value the name, it has in the
    /// graph built without them.
    /// </para>
    /// </summary>
    internal static class FastUseUniqueNames
    {
        /// <summary>
        /// How to number a graph a pass spliced nodes into: <paramref name="Last"/> after every other
        /// node, and, left unused, <paramref name="LeadingGap"/> numbers before the first node and
        /// <paramref name="GapsAfter"/>[k] numbers after the node keyed k.
        /// </summary>
        internal sealed record Numbering(
            IReadOnlySet<FastNodeKey> Last, IReadOnlyDictionary<FastNodeKey, int> GapsAfter, int LeadingGap);

        /// <summary>
        /// Rewrite the graph using a fresh per-graph counter starting at 1.
        /// </summary>
        public static void Process(InternalComputationGraph graph, Numbering? numbering = null)
        {
            int counter = 0;
            ProcessWithCounterAndReturnMap(graph, ref counter, numbering);
        }

        /// <summary>
        /// Same as <see cref="Process(InternalComputationGraph, Numbering)"/> but returns the
        /// <c>oldKey → newKey</c> map. Useful when the caller has additional
        /// data structures (e.g. tensor info lookups) keyed by old keys that
        /// also need rewriting.
        /// </summary>
        public static Dictionary<FastNodeKey, FastNodeKey> ProcessAndReturnMap(
            InternalComputationGraph graph, Numbering? numbering = null)
        {
            int counter = 0;
            return ProcessWithCounterAndReturnMap(graph, ref counter, numbering);
        }

        /// <summary>
        /// Rewrite the graph using <c>counter</c> — each node consumes
        /// the next id (the counter is incremented by reference). Useful for
        /// keeping a single id space across multiple graphs (e.g. main graph +
        /// functions).
        /// </summary>
        public static void ProcessWithCounter(InternalComputationGraph graph, ref int counter)
        {
            ProcessWithCounterAndReturnMap(graph, ref counter);
        }

        public static Dictionary<FastNodeKey, FastNodeKey> ProcessWithCounterAndReturnMap(
            InternalComputationGraph graph, ref int counter, Numbering? numbering = null)
        {
            // Pass 1: assign fresh sequential keys and FriendlyNames; remember
            // the old → new key mapping for the rewrite pass.
            // The output nodes are numbered last, after the phantoms below: they are the graph's
            // boundary rather than ops, and nothing references them, so numbering them first would
            // only shift every phantom's name.
            var oldToNew = new Dictionary<FastNodeKey, FastNodeKey>();
            List<FastNode> last = [];
            counter += numbering?.LeadingGap ?? 0;
            foreach (var node in graph.Nodes)
            {
                if (InternalOpCodes.IsGraphOutputOp(node.OpCode)) continue;
                if (numbering is not null && numbering.Last.Contains(node.Key))
                {
                    last.Add(node);
                    continue;
                }
                var oldKey = node.Key;
                counter++;
                var newKey = new FastNodeKey((UInt128)(uint)counter);
                oldToNew[node.Key] = newKey;
                node.Key = newKey;
                node.FriendlyName = $"N{counter}";
                if (numbering is not null && numbering.GapsAfter.TryGetValue(oldKey, out var gap)) counter += gap;
            }

            // Pass 1b: scan every reference (FullInputs / FullOutputs slots,
            // GraphOpenNodeKey) for FastNodeKeys
            // that don't correspond to any node in graph.Nodes. These are
            // "phantom" producers — typically zombie nodes (e.g.
            // LOOP_INDEX_VARIABLE) whose Variable outputs survive in
            // structural slots after the node itself is pruned out of
            // ComputationGraph.TopologicalOrderNodes. Each phantom key gets
            // its own fresh sequential id so every reference to it (e.g.
            // LOOP_OPEN's body[i] slot AND the body node that consumes it)
            // is rewritten to the same renamed key. OutputIndex is preserved.
            // Collect every referenced FastNodeKey into a flat list, then
            // register phantoms (those not already in oldToNew) inline.
            var lastKeys = new HashSet<FastNodeKey>();
            foreach (var node in last) lastKeys.Add(node.Key);
            var referenced = new List<FastNodeKey>();
            foreach (var node in graph.Nodes)
            {
                if (node.GraphOpenNodeKey is FastNodeKey openKey) referenced.Add(openKey);
                foreach (var slot in node.FullInputs.Values)
                    foreach (var k in slot)
                        if (k is FastTensorKey tk) referenced.Add(tk.FastNodeKey);
                foreach (var slot in node.FullOutputs.Values)
                    foreach (var k in slot)
                        if (k is FastTensorKey tk) referenced.Add(tk.FastNodeKey);
            }

            foreach (var key in referenced)
            {
                if (key.IsEmpty) continue;
                if (oldToNew.ContainsKey(key) || lastKeys.Contains(key)) continue;
                counter++;
                oldToNew[key] = new FastNodeKey((UInt128)(uint)counter);
            }

            foreach (var node in graph.OutputNodes)
            {
                counter++;
                node.Key = new FastNodeKey((UInt128)(uint)counter);
                node.FriendlyName = $"N{counter}";
            }

            foreach (var node in last)
            {
                counter++;
                var newKey = new FastNodeKey((UInt128)(uint)counter);
                oldToNew[node.Key] = newKey;
                node.Key = newKey;
                node.FriendlyName = $"N{counter}";
            }

            // Pass 2: walk every FastTensorKey/FastNodeKey reference in the graph
            // and substitute via the old → new map. Every key seen above is now
            // in the map, so no reference is left behind.
            foreach (var node in graph.Nodes)
            {
                node.GraphOpenNodeKey = RewriteNodeKey(node.GraphOpenNodeKey, oldToNew);
                RewriteSlots(node.FullInputs, oldToNew);
                RewriteSlots(node.FullOutputs, oldToNew);
            }

            return oldToNew;
        }

        private static void RewriteSlots(
            Dictionary<string, List<FastTensorKey?>> slots,
            Dictionary<FastNodeKey, FastNodeKey> oldToNew)
        {
            foreach (var slot in slots.Values)
                for (int i = 0; i < slot.Count; i++)
                {
                    var k = slot[i];
                    if (k is null) continue;
                    slot[i] = RewriteTensorKey(k.Value, oldToNew);
                }
        }

        private static FastTensorKey RewriteTensorKey(
            FastTensorKey tk, Dictionary<FastNodeKey, FastNodeKey> oldToNew)
        {
            if (tk.IsEmpty) return tk;
            return oldToNew.TryGetValue(tk.FastNodeKey, out var newK)
                ? new FastTensorKey(newK, tk.OutputIndex)
                : tk;
        }

        private static FastNodeKey? RewriteNodeKey(
            FastNodeKey? nk, Dictionary<FastNodeKey, FastNodeKey> oldToNew)
        {
            if (nk is null || nk.Value.IsEmpty) return nk;
            return oldToNew.TryGetValue(nk.Value, out var newK) ? newK : nk;
        }
    }
}
