using Shorokoo.Graph;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes.NodeDefinitions;
using System;
using System.Collections.Generic;

namespace Shorokoo.Core.Nodes.Processors.Training
{
    /// <summary>
    /// Refuses a training graph whose differentiated value comes out of a loop that could not be
    /// unrolled.
    ///
    /// <para>Autograd has no gradient rule for a loop, so the pipeline unrolls every one first —
    /// which needs the trip count as a constant. A count that is not one (read off an input's
    /// shape, or taken from a runtime input) leaves the loop standing, and what followed was not a
    /// backward pass but a broken graph: an invalid ONNX model where the loop only carried
    /// parameters, and a body-scoped tensor named by a module-scope output where it also carried
    /// state (Shorokoo/Shorokoo#309). Both are named here instead.</para>
    ///
    /// <para>The count cannot be recovered later either: the rig compiles one shape-generic
    /// trainstep and specializes only the ONNX session per input shape, so a shape-derived count is
    /// no more constant there than here.</para>
    /// </summary>
    internal static class FastRejectRolledLoopsInTraining
    {
        /// <summary>
        /// Walks back from what the graph differentiates — its <c>AUTO_GRAD</c> nodes, or its
        /// outputs before those exist — and throws on the first loop it reaches.
        /// </summary>
        /// <param name="graph">The graph to check.</param>
        /// <param name="unfoldableOnly">
        /// When true, only a loop whose trip count reads a graph input is refused. Run before the
        /// iteration-count fold, where a count computed from constants alone is still going to
        /// become one; false once the unroll has had its turn and anything left is final.
        /// </param>
        public static void Process(InternalComputationGraph graph, bool unfoldableOnly = false)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));

            // Every training graph is checked, and almost none has a loop; settle that before
            // building anything.
            bool anyLoop = false;
            foreach (var node in graph.Nodes) if (node.OpCode == OpCodes.LOOP_OPEN) { anyLoop = true; break; }
            if (!anyLoop) return;

            var nodesByKey = new Dictionary<FastNodeKey, FastNode>(graph.Nodes.Count);
            foreach (var n in graph.Nodes) nodesByKey[n.Key] = n;

            var producerOf = new Dictionary<FastTensorKey, FastNode>();
            var roots = new List<FastNode>();
            foreach (var node in graph.Nodes)
            {
                if (node.OpCode == InternalOpCodes.AUTO_GRAD) roots.Add(node);
                foreach (var (_, outs) in node.FullOutputs)
                    foreach (var ok in outs)
                        if (ok is not null && !ok.Value.IsEmpty) producerOf[ok.Value] = node;
            }

            if (roots.Count == 0)
                foreach (var outputKey in graph.Outputs)
                    if (producerOf.TryGetValue(outputKey, out var producer)) roots.Add(producer);

            var seen = new HashSet<FastNodeKey>();
            var worklist = new Stack<FastNode>();
            foreach (var root in roots)
                if (seen.Add(root.Key)) worklist.Push(root);

            while (worklist.Count > 0)
            {
                var node = worklist.Pop();
                if (node.OpCode == OpCodes.LOOP_CLOSE
                    && (!unfoldableOnly || TripCountReadsAnInput(node, graph, nodesByKey)))
                    throw new AutoDiffNotSupportedException(
                        ErrorCodes.AD003, OpCodes.LOOP,
                        "this model computes what it trains through a loop whose trip count is not a "
                        + "compile-time constant, and the only backward pass there is for a loop is to "
                        + "unroll it first. A count read off an input's shape is no more constant — the "
                        + "trainstep is compiled once for all input shapes, not once per shape. Give "
                        + "the loop a literal or [Hyper] trip count, or keep the trained computation "
                        + "out of it.");

                foreach (var (_, ins) in node.FullInputs)
                    foreach (var ik in ins)
                        if (ik is FastTensorKey k && !k.IsEmpty
                            && producerOf.TryGetValue(k, out var producer) && seen.Add(producer.Key))
                            worklist.Push(producer);
            }
        }

        /// <summary>Whether this close's loop counts its trips from something the graph is fed.</summary>
        private static bool TripCountReadsAnInput(
            FastNode close, InternalComputationGraph graph, Dictionary<FastNodeKey, FastNode> nodesByKey)
        {
            if (close.GraphOpenNodeKey is not FastNodeKey openKey || openKey.IsEmpty) return false;
            if (!nodesByKey.TryGetValue(openKey, out var open)) return false;
            if (open.Inputs.Count == 0 || open.Inputs[0] is not FastTensorKey tripCount) return false;
            return FastFoldLoopIterationCountsToConstantsProcessor.ReadsAGraphInput(tripCount, graph, nodesByKey);
        }
    }
}
