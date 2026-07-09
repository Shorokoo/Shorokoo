using Shorokoo.Graph;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using System.Collections.Generic;
using System.Linq;
using Shorokoo.Core.Nodes.Processors.AutoGrad;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// Undoes the close-input identity wrapping that <see cref="FastIdentityWrapping.WrapCloseInputs"/>
    /// (a.k.a. PrepForOnnx) inserts before serialization: every <c>IF_CLOSE</c> /
    /// <c>LOOP_CLOSE</c> input that points at a plain <c>IDENTITY</c> (no
    /// <c>InternalAttrRank</c> attribute) is rewritten to skip the identity and
    /// reference its source directly. The orphaned identity nodes are then swept
    /// via <see cref="FastProcessorHelper.RemoveUnreachableNodes"/>.
    ///
    /// <para>Functions referenced (transitively) by <c>graph</c> are
    /// also processed: each is rebuilt as a fresh <see cref="Function"/> instance
    /// over its mutated Fast body, and every <see cref="FastNode.TargetFunction"/>
    /// reference is rewritten to the new instance.</para>
    /// </summary>
    internal static class FastUnPrepFromOnnx
    {
        public static void Process(FastComputationGraph graph)
        {
            // Collect referenced functions in post order (callees first) so that
            // when we rebuild a function, every function its body references has
            // already been rebuilt — letting us point its TargetFunction edges at
            // the new instances before constructing the new Function.
            var functionsPostOrder = CollectFunctionsPostOrder(graph);
            var oldToNew = new Dictionary<Function, Function>(ReferenceEqualityComparer.Instance);

            foreach (var fn in functionsPostOrder)
            {
                var fnFast = fn.OriginalFastGraph.Clone();
                RemapTargetFunctions(fnFast, oldToNew);
                StripCloseInputIdentities(fnFast);
                FastProcessorHelper.RemoveUnreachableNodes(fnFast);

                // Carry the RNG algorithm tags across the rebuild: they gate the function
                // inliner (tagged functions are never inlined) and the export metadata, so
                // dropping them here would silently untag every RNG function on load.
                var newFn = new Function(fnFast, fn.FunctionType, defaultName: fn.DefaultName, friendlyName: fn.FriendlyName)
                {
                    RngAlgorithm = fn.RngAlgorithm,
                    RngFunctionKind = fn.RngFunctionKind,
                };
                oldToNew[fn] = newFn;
            }

            // Process the main graph last so it picks up the rebuilt function
            // references.
            RemapTargetFunctions(graph, oldToNew);
            StripCloseInputIdentities(graph);
            FastProcessorHelper.RemoveUnreachableNodes(graph);
        }

        /// <summary>
        /// Bypasses identity wrappers on every <c>IF_CLOSE</c> / <c>LOOP_CLOSE</c>
        /// input slot, in place. Mutates each close node's <see cref="FastNode.FullInputs"/>
        /// to reference the wrapped input directly when the producer is a plain
        /// (no <c>InternalAttrRank</c>) <c>IDENTITY</c>.
        /// </summary>
        private static void StripCloseInputIdentities(FastComputationGraph graph)
        {
            // Producer lookup: tensor key → the node that emits it. Built once per
            // graph; identity producers are looked up by their output key when we
            // walk the close-node input slots.
            var producers = new Dictionary<FastTensorKey, FastNode>();
            foreach (var node in graph.Nodes)
            {
                foreach (var kvp in node.FullOutputs)
                {
                    foreach (var ok in kvp.Value)
                    {
                        if (ok is null || ok.Value.IsEmpty) continue;
                        producers[ok.Value] = node;
                    }
                }
            }

            foreach (var node in graph.Nodes)
            {
                if (node.OpCode != OpCodes.IF_CLOSE && node.OpCode != OpCodes.LOOP_CLOSE)
                    continue;

                foreach (var groupKey in new List<string>(node.FullInputs.Keys))
                {
                    var slot = node.FullInputs[groupKey];
                    for (int i = 0; i < slot.Count; i++)
                    {
                        var input = slot[i];
                        if (input is null || input.Value.IsEmpty) continue;
                        if (!producers.TryGetValue(input.Value, out var producer)) continue;
                        if (producer.OpCode != OpCodes.IDENTITY) continue;
                        if (producer.Attributes.GetLongVal(OnnxOpAttributeNames.InternalAttrRank) is not null) continue;

                        // Replace with the identity's first (and only) input. CG
                        // path uses node.Inputs[0]; the Fast equivalent is the
                        // first slot under the empty-string graph-attribute key.
                        if (producer.FullInputs.TryGetValue("", out var idInputs) && idInputs.Count > 0)
                            slot[i] = idInputs[0];
                    }
                }
            }
        }

        /// <summary>
        /// Walks every <see cref="FastNode.TargetFunction"/> in <paramref name="graph"/>
        /// and rewrites references that appear in <paramref name="oldToNew"/> to point
        /// at the corresponding new <see cref="Function"/> instance. Used both on
        /// each function-body clone and on the main graph after the rebuild loop.
        /// </summary>
        private static void RemapTargetFunctions(FastComputationGraph graph, Dictionary<Function, Function> oldToNew)
        {
            if (oldToNew.Count == 0) return;
            foreach (var node in graph.Nodes)
            {
                if (node.TargetFunction is null) continue;
                if (oldToNew.TryGetValue(node.TargetFunction, out var newFn))
                    node.TargetFunction = newFn;
            }
        }

        /// <summary>
        /// Returns every <see cref="Function"/> reachable from
        /// <paramref name="graph"/> via <see cref="FastNode.TargetFunction"/>
        /// references, in post order (callees before callers). Mirrors the
        /// equivalent walk in <c>FastOnnxModelBuilder</c>.
        /// </summary>
        private static List<Function> CollectFunctionsPostOrder(FastComputationGraph graph)
        {
            var seen = new HashSet<Function>(ReferenceEqualityComparer.Instance);
            var result = new List<Function>();

            void Visit(FastComputationGraph g)
            {
                foreach (var node in g.Nodes)
                {
                    var fn = node.TargetFunction;
                    if (fn is null) continue;
                    if (!seen.Add(fn)) continue;
                    Visit(fn.OriginalFastGraph);
                    result.Add(fn);
                }
            }

            Visit(graph);
            return result;
        }
    }
}
