using Shorokoo.Graph;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Factory;
using Shorokoo.Core.Nodes.NodeDefinitions;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>One arm of one <c>IfElse</c>: the branch node, the condition that selects it, and
    /// which side of it this is.</summary>
    internal readonly record struct IfArm(FastNodeKey IfClose, FastTensorKey Condition, bool IsThen)
    {
        /// <summary>The <c>IF_CLOSE</c> input group this arm's values arrive in.</summary>
        internal string BranchAttribute => FastIfArms.BranchAttribute(IsThen);
    }

    /// <summary>
    /// Which <c>IfElse</c> arms each node belongs to.
    ///
    /// <para>A branch expression is an ordinary C# argument, so its nodes are traced
    /// <em>before</em> the <c>IF_OPEN</c> that selects them and only move inside it at ONNX build
    /// time (see <see cref="FastIfBranchScoper"/>). Membership is therefore read off what each
    /// branch's own <c>IF_CLOSE</c> inputs reach, never off where a node sits.</para>
    ///
    /// <para>A node both arms reach runs whatever the condition says and is nobody's arm; so is a
    /// node something outside the branch reads, and all it is computed from, and so are the inputs
    /// and parameters a branch merely reads. A <c>WITH_STATE_DEPS</c> naming a value only to keep
    /// it does not read it: a call made in an arm is named that way from outside the branch. That exclusion is what makes a value
    /// heading for one a value <em>leaving</em> the arm — the property both callers turn on, one
    /// to thread a state update through the branch that decides it
    /// (<see cref="FastChainStateUpdatesAcrossCallSites"/>), the other to stop a gradient from an
    /// arm that did not run (<see cref="AutoGrad.FastProcessAutoGradProcessor"/>).</para>
    ///
    /// <para>A node reached from nested <c>IfElse</c>s belongs to one arm of each, so the result is
    /// a list; a caller that cannot reason about nesting checks for more than one.</para>
    /// </summary>
    internal static class FastIfArms
    {
        /// <summary>The <c>IF_CLOSE</c> input group one side's values arrive in.</summary>
        public static string BranchAttribute(bool isThen)
            => isThen ? OnnxOpAttributeNames.AttrThenBranch : OnnxOpAttributeNames.AttrElseBranch;

        /// <param name="graph">The graph to classify.</param>
        /// <param name="gradientReadsOnly">Count only the reads a gradient flows through: a
        /// condition, a comparison or an index reads a value without differentiating it, so for
        /// where an arm's gradient has to stop such a read leaves the value in the arm.</param>
        public static Dictionary<FastNodeKey, List<IfArm>> Classify(InternalComputationGraph graph, bool gradientReadsOnly = false)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));

            // Most graphs have no branch at all; settle that before building any lookup.
            var armsOf = new Dictionary<FastNodeKey, List<IfArm>>();
            bool anyIf = false;
            foreach (var node in graph.Nodes) if (node.OpCode == OpCodes.IF_CLOSE) { anyIf = true; break; }
            if (!anyIf) return armsOf;

            var nodeByKey = FastProcessorHelper.BuildNodeByKey(graph);
            var producerOf = new Dictionary<FastTensorKey, FastNode>();
            foreach (var node in graph.Nodes)
                foreach (var (_, outs) in node.FullOutputs)
                    foreach (var ok in outs)
                        if (ok is not null && !ok.Value.IsEmpty) producerOf[ok.Value] = node;

            var readersOf = ReadersOf(graph, producerOf, gradientReadsOnly);
            var closeOf = new Dictionary<FastNodeKey, FastNodeKey>();
            foreach (var node in graph.Nodes)
                if (node.OpCode == OpCodes.IF_CLOSE && node.GraphOpenNodeKey is FastNodeKey openNode)
                    closeOf[openNode] = node.Key;

            foreach (var close in graph.Nodes)
            {
                if (close.OpCode != OpCodes.IF_CLOSE) continue;
                if (close.GraphOpenNodeKey is not FastNodeKey openKey) continue;
                if (!nodeByKey.TryGetValue(openKey, out var open)) continue;
                if (open.Inputs.Count == 0 || open.Inputs[0] is not FastTensorKey condition) continue;

                var reached = new Dictionary<bool, HashSet<FastNodeKey>>();
                foreach (var isThen in (bool[])[true, false])
                {
                    var arm = new IfArm(close.Key, condition, isThen);
                    reached[isThen] = close.FullInputs.TryGetValue(arm.BranchAttribute, out var roots)
                        ? ReachedFrom(roots, producerOf) : [];
                }

                var readOutside = ReadOutsideTheBranch(close.Key, reached, readersOf, closeOf, nodeByKey, producerOf);

                foreach (var isThen in (bool[])[true, false])
                    foreach (var nodeKey in reached[isThen])
                    {
                        if (reached[!isThen].Contains(nodeKey) || readOutside.Contains(nodeKey)) continue;
                        if (nodeByKey.TryGetValue(nodeKey, out var owner) && IsNobodysArm(owner)) continue;
                        if (!armsOf.TryGetValue(nodeKey, out var list)) armsOf[nodeKey] = list = [];
                        list.Add(new IfArm(close.Key, condition, isThen));
                    }
            }
            return armsOf;
        }

        /// <summary>Everything the given tensors are computed from.</summary>
        private static HashSet<FastNodeKey> ReachedFrom(
            IEnumerable<FastTensorKey?> roots, Dictionary<FastTensorKey, FastNode> producerOf)
        {
            var seen = new HashSet<FastNodeKey>();
            var worklist = new Stack<FastNode>();
            foreach (var root in roots)
                if (root is FastTensorKey k && producerOf.TryGetValue(k, out var producer)
                    && seen.Add(producer.Key))
                    worklist.Push(producer);
            while (worklist.Count > 0)
                foreach (var (_, ins) in worklist.Pop().FullInputs)
                    foreach (var ik in ins)
                        if (ik is FastTensorKey k && producerOf.TryGetValue(k, out var producer)
                            && seen.Add(producer.Key))
                            worklist.Push(producer);
            return seen;
        }

        /// <summary>The nodes reading each node's outputs, graph outputs included, leaving out the
        /// state dependencies a <c>WITH_STATE_DEPS</c> names only to keep them — and, where asked,
        /// every read no gradient flows back through: by an op whose output is a boolean, an index
        /// or a shape, by a node nothing the loss depends on reads, or by a graph output.</summary>
        private static Dictionary<FastNodeKey, List<FastNodeKey?>> ReadersOf(
            InternalComputationGraph graph, Dictionary<FastTensorKey, FastNode> producerOf, bool gradientReadsOnly)
        {
            var readersOf = new Dictionary<FastNodeKey, List<FastNodeKey?>>();
            void Read(FastTensorKey? key, FastNodeKey? reader)
            {
                if (key is not FastTensorKey k || !producerOf.TryGetValue(k, out var producer)) return;
                if (!readersOf.TryGetValue(producer.Key, out var list)) readersOf[producer.Key] = list = [];
                list.Add(reader);
            }

            // A node passes a gradient back to what it reads when a gradient reaches it: it is the
            // AUTO_GRAD differentiating the loss, or a node that passes one reads it. The graph's
            // other outputs — the state a step carries out, say — take no gradient, so reading a
            // value for them leaves it in its arm. Readers come after what they read, so one walk
            // from the back settles every node.
            HashSet<FastNodeKey>? passesGradient = null;
            if (gradientReadsOnly)
            {
                passesGradient = [];
                var reachesAGradient = new HashSet<FastTensorKey>();
                for (int n = graph.Nodes.Count - 1; n >= 0; n--)
                {
                    var node = graph.Nodes[n];
                    if (ReadsWithoutGradient.Contains(node.OpCode)) continue;
                    if (node.OpCode != InternalOpCodes.AUTO_GRAD
                        && !node.FullOutputs.Values.Any(outs => outs.Any(o => o is FastTensorKey k && reachesAGradient.Contains(k))))
                        continue;
                    passesGradient.Add(node.Key);
                    foreach (var (_, ins) in node.FullInputs)
                        foreach (var ik in ins)
                            if (ik is FastTensorKey k) reachesAGradient.Add(k);
                }
            }

            foreach (var node in graph.Nodes)
            {
                if (passesGradient is not null && !passesGradient.Contains(node.Key)) continue;
                foreach (var (_, ins) in node.FullInputs)
                    for (int i = 0; i < ins.Count; i++)
                        if (i == 0 || node.OpCode != InternalOpCodes.WITH_STATE_DEPS)
                            Read(ins[i], node.Key);
            }
            if (!gradientReadsOnly)
                foreach (var output in graph.Outputs)
                    Read(output, null);
            return readersOf;
        }

        /// <summary>The nodes of an <c>IfElse</c>'s arms that run whichever arm is taken because
        /// something other than the branch reads them, together with all they are computed from.
        /// The branch's own condition is read before either arm runs, so its <c>IF_OPEN</c> reads
        /// from outside; a nested <c>IfElse</c>'s <c>IF_OPEN</c> reads from wherever its
        /// <c>IF_CLOSE</c> sits.</summary>
        private static HashSet<FastNodeKey> ReadOutsideTheBranch(
            FastNodeKey close,
            Dictionary<bool, HashSet<FastNodeKey>> reached,
            Dictionary<FastNodeKey, List<FastNodeKey?>> readersOf,
            Dictionary<FastNodeKey, FastNodeKey> closeOf,
            Dictionary<FastNodeKey, FastNode> nodeByKey,
            Dictionary<FastTensorKey, FastNode> producerOf)
        {
            bool Reached(FastNodeKey k) => reached[true].Contains(k) || reached[false].Contains(k);
            bool InBranch(FastNodeKey? key)
                => key is FastNodeKey k
                   && (k.Equals(close) || Reached(k)
                       || closeOf.TryGetValue(k, out var nestedClose) && !nestedClose.Equals(close) && Reached(nestedClose));

            var readOutside = new HashSet<FastNodeKey>();
            var worklist = new Stack<FastNodeKey>();
            foreach (var nodeKey in reached[true].Concat(reached[false]))
                if (readersOf.TryGetValue(nodeKey, out var readers) && !readers.All(InBranch)
                    && readOutside.Add(nodeKey))
                    worklist.Push(nodeKey);

            while (worklist.Count > 0)
                if (nodeByKey.TryGetValue(worklist.Pop(), out var node))
                    foreach (var (_, ins) in node.FullInputs)
                        foreach (var ik in ins)
                            if (ik is FastTensorKey k && producerOf.TryGetValue(k, out var producer)
                                && InBranch(producer.Key) && readOutside.Add(producer.Key))
                                worklist.Push(producer.Key);
            return readOutside;
        }

        /// <summary>Ops whose outputs are booleans, indices or shapes, through which no gradient
        /// reaches what they read — an <c>IF_OPEN</c>'s condition among them.</summary>
        private static readonly HashSet<string> ReadsWithoutGradient =
        [
            OpCodes.IF_OPEN, OpCodes.GREATER, OpCodes.GREATER_OR_EQUAL, OpCodes.LESS, OpCodes.LESS_OR_EQUAL,
            OpCodes.EQUAL, OpCodes.AND, OpCodes.OR, OpCodes.XOR, OpCodes.NOT, OpCodes.IS_NAN, OpCodes.IS_INF,
            OpCodes.SHAPE, OpCodes.SIZE, OpCodes.ARG_MAX, OpCodes.ARG_MIN, OpCodes.NON_ZERO,
        ];

        /// <summary>An input or a parameter is read by a branch, never owned by it.</summary>
        private static bool IsNobodysArm(FastNode node)
            => InternalOpCodes.IsModelInputOp(node.OpCode)
            || node.OpCode == InternalOpCodes.MODEL_PARAM
            || node.OpCode == InternalOpCodes.MODEL_PARAM_DATA;
    }
}
