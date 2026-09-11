using Shorokoo.Graph;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Factory;
using Shorokoo.Core.Nodes.NodeDefinitions;
using System;
using System.Collections.Generic;

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
    /// <para>A node both arms reach runs whatever the condition says and is nobody's arm; so are
    /// the inputs and parameters a branch merely reads. That exclusion is what makes a value
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

        public static Dictionary<FastNodeKey, List<IfArm>> Classify(InternalComputationGraph graph)
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

                foreach (var isThen in (bool[])[true, false])
                    foreach (var nodeKey in reached[isThen])
                    {
                        if (reached[!isThen].Contains(nodeKey)) continue;
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

        /// <summary>An input or a parameter is read by a branch, never owned by it.</summary>
        private static bool IsNobodysArm(FastNode node)
            => FastOpsetResolver.IsModelInputOpCode(node.OpCode)
            || node.OpCode == InternalOpCodes.MODEL_PARAM
            || node.OpCode == InternalOpCodes.MODEL_PARAM_DATA;
    }
}
