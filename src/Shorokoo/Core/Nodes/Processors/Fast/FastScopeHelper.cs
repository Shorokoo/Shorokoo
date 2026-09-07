using Shorokoo.Graph;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using System.Collections.Generic;
using System.Linq;
using Shorokoo.Core.Nodes.Processors.AutoGrad;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// Primitives for expanding and shrinking loop scopes in a
    /// <see cref="InternalComputationGraph"/>. Scope membership in the Fast pipeline is
    /// positional: every node whose index in <see cref="InternalComputationGraph.Nodes"/>
    /// falls between a <c>LOOP_OPEN</c> and its paired <c>LOOP_CLOSE</c> is treated as
    /// a body node by <see cref="Shorokoo.Core.Inference.QuickExecutionEngine"/>'s
    /// linear loop-back model. Shrinking a scope moves nodes that positionally fall
    /// inside it but do not actually depend on the loop's body outputs to just before
    /// the <c>LOOP_OPEN</c>, reducing the re-execution range on loop-back and
    /// eliminating spurious per-iteration <c>History</c> accumulation on QEE-tracked
    /// tensors.
    ///
    /// The data-flow notion used here mirrors the legacy CG-side hoisting
    /// primitive. A tensor is
    /// "loop-dependent" if it is produced by a <c>LOOP_OPEN</c> or by any node that
    /// transitively consumes a loop-dependent tensor. A node is "loop-invariant" at
    /// a given scope if none of its inputs is loop-dependent.
    /// </summary>
    internal static class FastScopeHelper
    {
        /// <summary>
        /// Describes a single <c>LOOP_OPEN</c> / <c>LOOP_CLOSE</c> pair and its
        /// positions in <see cref="InternalComputationGraph.Nodes"/>. Returned from
        /// <see cref="TryResolveLoopScope"/> so callers can inspect a scope without
        /// themselves knowing the linear layout convention.
        /// </summary>
        public readonly struct FastLoopScope
        {
            public FastNode OpenNode { get; }
            public FastNode CloseNode { get; }
            public int OpenIdx { get; }
            public int CloseIdx { get; }

            public FastLoopScope(FastNode openNode, FastNode closeNode, int openIdx, int closeIdx)
            {
                OpenNode = openNode;
                CloseNode = closeNode;
                OpenIdx = openIdx;
                CloseIdx = closeIdx;
            }
        }

        /// <summary>
        /// Returns every <see cref="FastTensorKey"/> whose value depends on some
        /// <c>LOOP_OPEN</c>'s body outputs — directly, or transitively through any chain
        /// of nodes. Output keys produced by a <c>LOOP_OPEN</c> are included.
        /// </summary>
        public static HashSet<FastTensorKey> BuildLoopDependentTensors(InternalComputationGraph graph)
        {
            var loopDependent = new HashSet<FastTensorKey>();
            foreach (var node in graph.Nodes)
            {
                if (node.OpCode == OpCodes.LOOP_OPEN)
                {
                    AddAllOutputs(node, loopDependent);
                    continue;
                }

                if (HasLoopDependentInput(node, loopDependent))
                    AddAllOutputs(node, loopDependent);
            }
            return loopDependent;
        }

        /// <summary>
        /// Returns every <see cref="FastTensorKey"/> whose value <em>varies across the iterations</em>
        /// of a loop it sits inside — the keys a consumer reads with a different value on each pass.
        ///
        /// <para>This is narrower than <see cref="BuildLoopDependentTensors"/>, which marks
        /// everything descended from a <c>LOOP_OPEN</c> and so keeps propagating past the
        /// <c>LOOP_CLOSE</c> into the rest of the graph. That is what <see cref="ShrinkAllScopes"/>
        /// wants (it only ever asks about nodes inside a scope), but a loop's <em>result</em> is
        /// computed once and does not vary, so a caller asking "does this differ per iteration?"
        /// about a key anywhere in the graph needs this one instead. A <c>LOOP_CLOSE</c> therefore
        /// sheds the variation of the loop it closes, keeping only an enclosing loop's.</para>
        ///
        /// <para>An enclosing loop's variation reaches a nested loop's body through the inner
        /// <c>LOOP_OPEN</c>'s inputs (its trip count and carry initializers), not through the
        /// <c>LOOP_CLOSE</c>'s, so the open carries that shallower depth onto its outputs; the
        /// close then keeps it while shedding its own. That is what leaves a nested loop's result
        /// varying with the outer loop.</para>
        /// </summary>
        public static HashSet<FastTensorKey> BuildPerIterationTensors(InternalComputationGraph graph)
        {
            // Key → the shallowest loop depth across whose iterations the key varies (scope
            // membership is positional, so depth is just the open/close nesting count).
            var varyDepth = new Dictionary<FastTensorKey, int>();
            int depth = 0;

            foreach (var node in graph.Nodes)
            {
                if (node.OpCode == OpCodes.LOOP_OPEN)
                {
                    // The body sees this loop's own iterations, and on top of that whatever an
                    // enclosing loop varies in the trip count and carry initializers — which reach
                    // the body only through these outputs, so the shallower depth has to ride along
                    // or the paired close would shed it as if it were this loop's.
                    int enclosing = MinInputVaryDepth(node, varyDepth, shallowerThan: int.MaxValue);
                    depth++;
                    RecordDepth(node, enclosing > 0 ? enclosing : depth, varyDepth);
                    continue;
                }

                if (node.OpCode == OpCodes.LOOP_CLOSE && depth > 0)
                {
                    int enclosing = MinInputVaryDepth(node, varyDepth, shallowerThan: depth);
                    depth--;
                    RecordDepth(node, enclosing, varyDepth);
                    continue;
                }

                RecordDepth(node, MinInputVaryDepth(node, varyDepth, shallowerThan: int.MaxValue), varyDepth);
            }

            return [.. varyDepth.Keys];
        }

        /// <summary>Shallowest depth any input of <paramref name="node"/> varies at, counting only
        /// depths below <paramref name="shallowerThan"/>; 0 when no input varies.</summary>
        private static int MinInputVaryDepth(
            FastNode node, Dictionary<FastTensorKey, int> varyDepth, int shallowerThan)
        {
            int min = 0;
            foreach (var kvp in node.FullInputs)
                foreach (var key in kvp.Value)
                    if (key is FastTensorKey tk && !tk.IsEmpty
                        && varyDepth.TryGetValue(tk, out var d) && d < shallowerThan
                        && (min == 0 || d < min))
                        min = d;
            return min;
        }

        /// <summary>Records <paramref name="depth"/> for every output of <paramref name="node"/>,
        /// or nothing when it is 0 (the node's outputs do not vary).</summary>
        private static void RecordDepth(FastNode node, int depth, Dictionary<FastTensorKey, int> varyDepth)
        {
            if (depth == 0) return;
            foreach (var kvp in node.FullOutputs)
                foreach (var key in kvp.Value)
                    if (key is FastTensorKey tk && !tk.IsEmpty)
                        varyDepth[tk] = depth;
        }

        /// <summary>
        /// Resolves a <c>LOOP_CLOSE</c> node to its paired <c>LOOP_OPEN</c> and returns
        /// their positions in <see cref="InternalComputationGraph.Nodes"/>. Returns null if
        /// <paramref name="closeNode"/> is not a <c>LOOP_CLOSE</c>, if its
        /// <see cref="FastNode.GraphOpenNodeKey"/> is missing, or if the paired
        /// <c>LOOP_OPEN</c> is not present in the graph.
        /// </summary>
        public static FastLoopScope? TryResolveLoopScope(InternalComputationGraph graph, FastNode closeNode)
        {
            if (closeNode.OpCode != OpCodes.LOOP_CLOSE) return null;
            if (closeNode.GraphOpenNodeKey is not FastNodeKey openKey || openKey.IsEmpty) return null;

            int openIdx = -1;
            int closeIdx = -1;
            FastNode? openNode = null;
            for (int i = 0; i < graph.Nodes.Count; i++)
            {
                var n = graph.Nodes[i];
                if (n.Key == openKey) { openNode = n; openIdx = i; }
                else if (ReferenceEquals(n, closeNode)) { closeIdx = i; }
                if (openIdx >= 0 && closeIdx >= 0) break;
            }
            if (openNode is null || openIdx < 0 || closeIdx < 0) return null;
            return new FastLoopScope(openNode, closeNode, openIdx, closeIdx);
        }

        /// <summary>
        /// Shrinks every loop scope in <paramref name="graph"/>: any node whose
        /// positional index falls between a <c>LOOP_OPEN</c> and its paired
        /// <c>LOOP_CLOSE</c> but whose inputs include no loop-dependent tensor is moved
        /// to just before the outermost currently-active <c>LOOP_OPEN</c>. The relative
        /// order of nodes is otherwise preserved. Mutates
        /// <see cref="InternalComputationGraph.Nodes"/> in place.
        ///
        /// This is the Fast-pipeline equivalent of the legacy CG-side
        /// hoisting primitive that ran as part of ComputationGraph
        /// construction. QEE consumers benefit because
        /// hoisted nodes no longer appear inside the positional loop body and so:
        ///   - they are not re-executed on loop-back, and
        ///   - their results are not tagged with per-iteration <c>IterationIndices</c> /
        ///     <c>History</c> (which simplifies <c>IntData</c> availability checks for
        ///     downstream consumers).
        /// </summary>
        public static void ShrinkAllScopes(InternalComputationGraph graph)
        {
            var loopDependent = BuildLoopDependentTensors(graph);

            var result = new List<FastNode>(graph.Nodes.Count);
            // FastNodeKey of each active (not-yet-closed) LOOP_OPEN → its position in result.
            var openPositions = new Dictionary<FastNodeKey, int>();

            foreach (var node in graph.Nodes)
            {
                if (node.OpCode == OpCodes.LOOP_OPEN)
                {
                    openPositions[node.Key] = result.Count;
                    result.Add(node);
                    continue;
                }

                bool isCloseNode = node.OpCode == OpCodes.LOOP_CLOSE ||
                                   node.OpCode == OpCodes.IF_CLOSE;

                if (!isCloseNode && openPositions.Count > 0)
                {
                    bool loopDep = HasLoopDependentInput(node, loopDependent);
                    if (!loopDep)
                    {
                        int earliestOpenPos = int.MaxValue;
                        foreach (var p in openPositions.Values)
                            if (p < earliestOpenPos) earliestOpenPos = p;

                        result.Insert(earliestOpenPos, node);
                        foreach (var k in openPositions.Keys.ToList())
                            if (openPositions[k] >= earliestOpenPos)
                                openPositions[k] = openPositions[k] + 1;
                        continue;
                    }
                }

                if (node.OpCode == OpCodes.LOOP_CLOSE &&
                    node.GraphOpenNodeKey is FastNodeKey openKey && !openKey.IsEmpty)
                {
                    openPositions.Remove(openKey);
                }

                result.Add(node);
            }

            graph.Nodes = result;
        }

        private static bool HasLoopDependentInput(FastNode node, HashSet<FastTensorKey> loopDependent)
        {
            foreach (var kvp in node.FullInputs)
                foreach (var key in kvp.Value)
                    if (key is FastTensorKey tk && !tk.IsEmpty && loopDependent.Contains(tk))
                        return true;
            return false;
        }

        private static void AddAllOutputs(FastNode node, HashSet<FastTensorKey> loopDependent)
        {
            foreach (var kvp in node.FullOutputs)
                foreach (var key in kvp.Value)
                    if (key is FastTensorKey tk && !tk.IsEmpty)
                        loopDependent.Add(tk);
        }
    }
}
