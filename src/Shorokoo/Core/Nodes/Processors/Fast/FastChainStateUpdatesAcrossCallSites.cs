using Shorokoo.Graph;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes.NodeDefinitions;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// Makes a module-owned state update apply once per call, rather than once per module.
    ///
    /// <para>Concretization collapses two call sites of one model handle onto one state
    /// parameter — correctly, since they share it — but each call keeps its own
    /// <c>STATE_UPDATE_LINK</c>, and each reads the parameter as the graph gave it. So the second
    /// call computes its update from the initial value rather than from the first call's, and the
    /// two updates do not compose (Shorokoo/Shorokoo#306). This pass points a later call's reads
    /// at the earlier call's link.</para>
    ///
    /// <para><b>Why the link's output rather than the updated value.</b> A link lowers one way
    /// where state persists across executions (<c>Identity(updatedState)</c>, see
    /// <see cref="FastLowerStateUpdateNodes"/>) and the other for one-shot inference
    /// (<c>Identity(originalState)</c>, see <see cref="FastLowerStateUpdateLinksForInference"/>).
    /// Reading through the link keeps both honest with one rewrite: where state persists the
    /// updates compose, and in one-shot inference — where the update computation is dropped — every
    /// call still sees the value it was given.</para>
    ///
    /// <para><b>What bounds a call.</b> <c>WITH_STATE_DEPS</c> is emitted once per call site that
    /// owns state, listing that call's links, so the ones naming a given parameter mark that
    /// parameter's call sites in order. Node order supplies the sequence. Position alone does not:
    /// a body reads its parameter after its own link as often as before it — the link is an
    /// annotation, not an assignment — so a read is attributed to a call by which marker it falls
    /// under, never by whether it follows a link.</para>
    /// </summary>
    internal static class FastChainStateUpdatesAcrossCallSites
    {
        public static void Process(InternalComputationGraph graph)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));

            var nodeByKey = FastProcessorHelper.BuildNodeByKey(graph);
            var positionOf = new Dictionary<FastNodeKey, int>(graph.Nodes.Count);
            for (int i = 0; i < graph.Nodes.Count; i++) positionOf[graph.Nodes[i].Key] = i;

            // Links per state parameter, in node order. A link reaches its parameter through the
            // Identity chain inlining leaves behind, so the walk has to follow it — a direct key
            // match finds nothing and would leave every parameter silently un-chained.
            var linksByParam = new Dictionary<FastNodeKey, List<FastNode>>();
            foreach (var node in graph.Nodes)
            {
                if (node.OpCode != InternalOpCodes.STATE_UPDATE_LINK) continue;
                if (node.Inputs.Count == 0 || node.Inputs[0] is not FastTensorKey originalKey) continue;
                if (ResolveThroughIdentities(originalKey, nodeByKey) is not FastTensorKey paramKey) continue;
                if (!linksByParam.TryGetValue(paramKey.FastNodeKey, out var list))
                    linksByParam[paramKey.FastNodeKey] = list = [];
                list.Add(node);
            }

            if (linksByParam.Count == 0) return;

            var scopeDepth = ComputeScopeDepth(graph);
            var rewrites = new Dictionary<FastNodeKey, List<(int fromPos, int toPos, FastTensorKey replacement)>>();

            foreach (var (paramNodeKey, links) in linksByParam)
            {
                if (links.Count < 2) continue;   // one call site updates nothing anyone else reads

                // Node order is a sequence only where every call actually runs, one after another.
                // Inside control flow it is not: the two arms of an IfElse are alternatives, so
                // composing them would credit an update that never happened, and two calls in a
                // loop body repeat per trip rather than once. Both need reasoning this pass does
                // not do, and dropping one of the updates — which is what happens without it — is
                // the bug being fixed, so refuse instead of guessing (Shorokoo/Shorokoo#308).
                foreach (var link in links)
                    if (scopeDepth[positionOf[link.Key]] != 0)
                        throw new InvalidOperationException(
                            "FastChainStateUpdatesAcrossCallSites: a model owning updateable state is called "
                            + "more than once from inside control flow, so its updates cannot be ordered. Only "
                            + "one arm of an IfElse runs, and a loop body repeats, so neither gives the calls "
                            + "the single running order composing them needs. Call such a model once per scope, "
                            + "or give each call site its own model.");

                // The call-site markers for THIS parameter: a WITH_STATE_DEPS carries the links of
                // the call it closes, so filtering by them keeps a nested call's marker out of the
                // enclosing parameter's sequence.
                var linkKeys = links.Select(l => l.Outputs[0]!.Value).ToHashSet();
                var markers = graph.Nodes
                    .Where(n => n.OpCode == InternalOpCodes.WITH_STATE_DEPS
                                && n.Inputs.Skip(1).Any(k => k is FastTensorKey dep && linkKeys.Contains(dep)))
                    .ToList();
                if (markers.Count != links.Count)
                    throw new InvalidOperationException(
                        "FastChainStateUpdatesAcrossCallSites: a state parameter has " + links.Count
                        + " updates but " + markers.Count + " call sites carrying them, so which call "
                        + "each update belongs to cannot be read off the graph. Leaving them unchained "
                        + "would drop all but one, which is the defect this pass exists to fix.");

                for (int k = 1; k < links.Count; k++)
                {
                    // Everything after the previous call's marker, up to and including this one, is
                    // this call. Its reads of the parameter become the previous call's link.
                    int from = positionOf[markers[k - 1].Key] + 1;
                    int to = positionOf[markers[k].Key];
                    if (!rewrites.TryGetValue(paramNodeKey, out var list))
                        rewrites[paramNodeKey] = list = [];
                    list.Add((from, to, links[k - 1].Outputs[0]!.Value));
                }
            }

            if (rewrites.Count == 0) return;

            foreach (var (paramNodeKey, ranges) in rewrites)
            {
                var paramOutput = new FastTensorKey(paramNodeKey, 0);
                foreach (var (from, to, replacement) in ranges)
                    for (int i = from; i <= to && i < graph.Nodes.Count; i++)
                    {
                        foreach (var kvp in graph.Nodes[i].FullInputs)
                        {
                            var inputs = kvp.Value;
                            for (int j = 0; j < inputs.Count; j++)
                            {
                                // Through the Identity chain, as the link walk above: a read
                                // wrapped in one that was spliced ahead of this call is still a
                                // read of the parameter, and missing it leaves this call on the
                                // stale value with nothing to show for it.
                                if (inputs[j] is not FastTensorKey k) continue;
                                if (ResolveThroughIdentities(k, nodeByKey) is FastTensorKey read
                                    && read.Equals(paramOutput))
                                    inputs[j] = replacement;
                            }
                        }
                    }
            }
        }

        /// <summary>
        /// The value behind an Identity chain. Inlining wraps a parameter in one at every splice,
        /// so a link names its parameter only at the end of the chain.
        /// </summary>
        private static FastTensorKey? ResolveThroughIdentities(
            FastTensorKey key, Dictionary<FastNodeKey, FastNode> nodeByKey)
        {
            var current = key;
            var visited = new HashSet<FastTensorKey>();
            while (visited.Add(current))
            {
                if (!nodeByKey.TryGetValue(current.FastNodeKey, out var node)) return current;
                if (node.OpCode != OpCodes.IDENTITY) return current;
                if (node.Inputs.Count == 0 || node.Inputs[0] is not FastTensorKey inner) return current;
                current = inner;
            }
            return null;
        }

        /// <summary>Nesting depth of every node, counting scope open and close nodes.</summary>
        private static int[] ComputeScopeDepth(InternalComputationGraph graph)
        {
            var depth = new int[graph.Nodes.Count];
            int current = 0;
            for (int i = 0; i < graph.Nodes.Count; i++)
            {
                var node = graph.Nodes[i];
                if (node.IsCloseNode()) current--;
                depth[i] = current;
                if (node.IsOpenNode()) current++;
            }
            return depth;
        }

    }
}
