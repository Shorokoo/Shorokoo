using Shorokoo.Core.Factory;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Graph;
using System.Collections.Generic;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// Moves the nodes that compute an <c>IF</c> branch's value inside that <c>IF</c>'s scope, so
    /// the branch lowers into the ONNX <c>If</c> subgraph that runs only when the branch is taken.
    ///
    /// <para>The pass exists because a branch expression is an ordinary C# argument: it is
    /// evaluated, and so appends its nodes to the graph, <em>before</em> the <c>IF_OPEN</c> that
    /// selects it. Left there, every branch of every <c>IF</c> is computed on every execution and
    /// the <c>If</c> node only picks a winner — wasted work in the mild case, and a wrong answer
    /// in the sharp one, since an operation may be invalid off-branch (unwrapping an optional that
    /// is absent on that path is the canonical example). Scope membership in the Fast pipeline is
    /// positional, so restoring the intended meaning is a matter of moving those nodes between the
    /// <c>IF_OPEN</c> and the <c>IF_CLOSE</c>.</para>
    ///
    /// <para>A node moves only when it belongs to exactly one branch and to nothing else. The
    /// branch's <em>cone</em> starts as everything the branch's outputs are computed from, minus
    /// everything the other branch's outputs are computed from, and then loses any member with a
    /// consumer outside the cone — a value read after the <c>IF</c>, by the other branch, by an
    /// unrelated scope, or by the <c>IF_OPEN</c> itself (the condition). Model inputs, parameter
    /// data and producers of graph outputs never move. What survives is exactly the set whose
    /// results nothing outside the branch can observe, so moving it changes only whether it
    /// runs.</para>
    ///
    /// <para>Nesting is handled by treating a whole <c>OPEN</c>…<c>CLOSE</c> band as one unit: a
    /// loop used by one branch moves with its body intact, and the pass then recurses into every
    /// scope it has settled, so a branch inside a branch is scoped in turn. Since only siblings
    /// preceding the <c>IF_OPEN</c> are candidates and their relative order is preserved, the
    /// result is still in topological order.</para>
    /// </summary>
    internal static class FastIfBranchScoper
    {
        /// <summary>
        /// Scopes every <c>IF</c> branch in <paramref name="graph"/>, mutating
        /// <see cref="InternalComputationGraph.Nodes"/> in place.
        /// </summary>
        public static void ScopeAllIfBranches(InternalComputationGraph graph)
        {
            if (graph is null) throw new System.ArgumentNullException(nameof(graph));

            System.Diagnostics.Debug.Assert(graph.IsLinearOrderValid(),
                "FastIfBranchScoper.ScopeAllIfBranches: graph must already be in valid linear order " +
                "on entry. This pass moves branch bodies into their scope; it does not fix invalid ones.");

            var ctx = new Context(graph);
            if (!ctx.HasIf) return;

            graph.Nodes = SinkLevel(graph.Nodes, ctx);

            System.Diagnostics.Debug.Assert(graph.IsLinearOrderValid(),
                "FastIfBranchScoper.ScopeAllIfBranches: scoping left the graph in an invalid linear order.");
        }

        /// <summary>Graph-wide lookups the cone analysis reads; built once per pass.</summary>
        private sealed class Context
        {
            public readonly Dictionary<FastNodeKey, FastNode> NodeByKey = new();
            public readonly Dictionary<FastTensorKey, FastNodeKey> ProducerOf = new();
            public readonly Dictionary<FastNodeKey, List<FastNodeKey>> ConsumersOf = new();
            public readonly HashSet<FastNodeKey> Pinned = new();
            public readonly bool HasIf;

            public Context(InternalComputationGraph graph)
            {
                var boundaryKeys = new HashSet<FastTensorKey>(graph.Outputs);
                foreach (var k in graph.Inputs) boundaryKeys.Add(k);

                foreach (var n in graph.Nodes)
                {
                    if (n.OpCode == OpCodes.IF_OPEN) HasIf = true;
                    NodeByKey[n.Key] = n;
                    foreach (var grp in n.FullOutputs)
                        foreach (var key in grp.Value)
                            if (key is FastTensorKey tk && !tk.IsEmpty)
                            {
                                ProducerOf[tk] = n.Key;
                                if (boundaryKeys.Contains(tk)) Pinned.Add(n.Key);
                            }

                    if (FastOpsetResolver.IsModelInputOpCode(n.OpCode) ||
                        n.OpCode == InternalOpCodes.MODEL_PARAM_DATA ||
                        n.OpCode == InternalOpCodes.MODEL_PARAM)
                        Pinned.Add(n.Key);
                }

                foreach (var n in graph.Nodes)
                    foreach (var grp in n.FullInputs)
                        foreach (var key in grp.Value)
                            if (key is FastTensorKey tk && !tk.IsEmpty &&
                                ProducerOf.TryGetValue(tk, out var p))
                            {
                                if (!ConsumersOf.TryGetValue(p, out var list))
                                    ConsumersOf[p] = list = new List<FastNodeKey>();
                                if (!list.Contains(n.Key)) list.Add(n.Key);
                            }
            }
        }

        /// <summary>
        /// One unit of movement at a given nesting level: a single node, or a whole
        /// <c>OPEN</c>…<c>CLOSE</c> band together with everything nested inside it.
        /// </summary>
        private sealed class Block
        {
            public readonly List<FastNode> Nodes;
            public readonly HashSet<FastNodeKey> Keys = new();

            public Block(List<FastNode> nodes)
            {
                Nodes = nodes;
                foreach (var n in nodes) Keys.Add(n.Key);
            }

            public bool IsScope => FastOpsetResolver.IsOpenOpCode(Nodes[0].OpCode);
            public bool IsIf => Nodes[0].OpCode == OpCodes.IF_OPEN;
        }

        /// <summary>
        /// Scopes every <c>IF</c> directly inside <paramref name="level"/> (the whole graph, or one
        /// scope's body), then recurses into each scope the result contains.
        /// </summary>
        private static List<FastNode> SinkLevel(List<FastNode> level, Context ctx)
        {
            var blocks = SplitIntoBlocks(level);

            var blockOfNode = new Dictionary<FastNodeKey, int>();
            for (int b = 0; b < blocks.Count; b++)
                foreach (var k in blocks[b].Keys) blockOfNode[k] = b;

            // Which IF, and which of its branches, has claimed each block. Cones of different IFs
            // are disjoint by construction: a block feeding two of them has, for each, a consumer
            // the other's cone does not contain, so neither keeps it.
            var claim = new Dictionary<int, (int IfIdx, bool IsThen)>();
            for (int b = 0; b < blocks.Count; b++)
            {
                if (!blocks[b].IsIf) continue;
                foreach (var c in Cone(blocks, b, blockOfNode, isThen: true, ctx))
                    if (!claim.ContainsKey(c)) claim[c] = (b, true);
                foreach (var c in Cone(blocks, b, blockOfNode, isThen: false, ctx))
                    if (!claim.ContainsKey(c)) claim[c] = (b, false);
            }

            var moved = new List<FastNode>(level.Count);
            for (int b = 0; b < blocks.Count; b++)
                if (!claim.ContainsKey(b))
                    Assemble(b, blocks, claim, moved);

            return DescendIntoScopes(moved, ctx);
        }

        /// <summary>
        /// Appends block <paramref name="b"/> to <paramref name="into"/>, with anything it has
        /// claimed placed inside it. A claimed block may itself be a scope with claims of its own —
        /// an <c>IF</c> whose result only one branch of an enclosing <c>IF</c> reads travels with
        /// everything that computes it — so this recurses through the claims. It never descends
        /// into a scope's existing body; <see cref="DescendIntoScopes"/> does that afterwards, once
        /// every body is complete.
        /// </summary>
        private static void Assemble(
            int b, List<Block> blocks, Dictionary<int, (int IfIdx, bool IsThen)> claim, List<FastNode> into)
        {
            var blk = blocks[b];
            if (!blk.IsScope) { into.AddRange(blk.Nodes); return; }

            into.Add(blk.Nodes[0]);
            if (blk.IsIf)
            {
                // Then-cone before else-cone, so the branch bifurcation the ONNX builder does by
                // back-walk sees the layout it expects. Each claimed block was positioned before
                // the OPEN and keeps its relative order, so the body stays topologically ordered.
                for (int c = 0; c < blocks.Count; c++)
                    if (claim.TryGetValue(c, out var cl) && cl.IfIdx == b && cl.IsThen)
                        Assemble(c, blocks, claim, into);
                for (int c = 0; c < blocks.Count; c++)
                    if (claim.TryGetValue(c, out var cl) && cl.IfIdx == b && !cl.IsThen)
                        Assemble(c, blocks, claim, into);
            }
            into.AddRange(blk.Nodes.GetRange(1, blk.Nodes.Count - 2));
            into.Add(blk.Nodes[^1]);
        }

        /// <summary>Re-runs the pass one level down, inside every scope of <paramref name="level"/>.</summary>
        private static List<FastNode> DescendIntoScopes(List<FastNode> level, Context ctx)
        {
            var result = new List<FastNode>(level.Count);
            foreach (var blk in SplitIntoBlocks(level))
            {
                if (!blk.IsScope) { result.AddRange(blk.Nodes); continue; }
                result.Add(blk.Nodes[0]);
                result.AddRange(SinkLevel(blk.Nodes.GetRange(1, blk.Nodes.Count - 2), ctx));
                result.Add(blk.Nodes[^1]);
            }
            return result;
        }

        private static List<Block> SplitIntoBlocks(List<FastNode> level)
        {
            var blocks = new List<Block>();
            int i = 0;
            while (i < level.Count)
            {
                if (!FastOpsetResolver.IsOpenOpCode(level[i].OpCode))
                {
                    blocks.Add(new Block(level.GetRange(i, 1)));
                    i++;
                    continue;
                }

                int depth = 0;
                int j = i;
                for (; j < level.Count; j++)
                {
                    if (FastOpsetResolver.IsOpenOpCode(level[j].OpCode)) depth++;
                    else if (FastOpsetResolver.IsCloseOpCode(level[j].OpCode) && --depth == 0) break;
                }
                blocks.Add(new Block(level.GetRange(i, j - i + 1)));
                i = j + 1;
            }
            return blocks;
        }

        /// <summary>
        /// The blocks preceding <paramref name="ifIdx"/> that compute one branch's value and
        /// nothing else, and so may move inside the <c>IF</c>.
        /// </summary>
        private static HashSet<int> Cone(
            List<Block> blocks, int ifIdx, Dictionary<FastNodeKey, int> blockOfNode,
            bool isThen, Context ctx)
        {
            var ifBlock = blocks[ifIdx];
            var ifClose = ifBlock.Nodes[^1];
            var ifOpenKey = ifBlock.Nodes[0].Key;

            var mine = ReachedBlocks(ifClose, BranchAttr(isThen), ifIdx, blockOfNode, ctx);
            var other = ReachedBlocks(ifClose, BranchAttr(!isThen), ifIdx, blockOfNode, ctx);

            var cone = new HashSet<int>();
            foreach (var b in mine)
                if (!other.Contains(b) && IsMovable(blocks[b], ctx)) cone.Add(b);

            bool changed = true;
            while (changed)
            {
                changed = false;
                var doomed = new List<int>();
                foreach (var b in cone)
                    if (HasConsumerOutside(blocks[b], cone, ifBlock, ifOpenKey, blockOfNode, ctx))
                        doomed.Add(b);
                foreach (var b in doomed) { cone.Remove(b); changed = true; }
            }
            return cone;
        }

        private static string BranchAttr(bool isThen) => isThen
            ? OnnxOpAttributeNames.AttrThenBranch
            : OnnxOpAttributeNames.AttrElseBranch;

        /// <summary>
        /// Blocks of this level, positioned before <paramref name="ifIdx"/>, that the branch's
        /// declared outputs are transitively computed from.
        /// </summary>
        private static HashSet<int> ReachedBlocks(
            FastNode ifClose, string branchAttr, int ifIdx,
            Dictionary<FastNodeKey, int> blockOfNode, Context ctx)
        {
            var reached = new HashSet<int>();
            var seen = new HashSet<FastNodeKey>();
            var queue = new Queue<FastNodeKey>();

            if (ifClose.FullInputs.TryGetValue(branchAttr, out var slot))
                foreach (var key in slot)
                    if (key is FastTensorKey tk && !tk.IsEmpty &&
                        ctx.ProducerOf.TryGetValue(tk, out var p) && seen.Add(p))
                        queue.Enqueue(p);

            while (queue.Count > 0)
            {
                var nk = queue.Dequeue();
                if (blockOfNode.TryGetValue(nk, out var b) && b < ifIdx) reached.Add(b);
                if (!ctx.NodeByKey.TryGetValue(nk, out var node)) continue;
                foreach (var grp in node.FullInputs)
                    foreach (var key in grp.Value)
                        if (key is FastTensorKey tk && !tk.IsEmpty &&
                            ctx.ProducerOf.TryGetValue(tk, out var p) && seen.Add(p))
                            queue.Enqueue(p);
            }
            return reached;
        }

        /// <summary>
        /// Whether any value <paramref name="block"/> produces is read from somewhere the move
        /// would put out of reach: outside this level, outside the cone, or by the <c>IF_OPEN</c>,
        /// whose condition is evaluated before either branch.
        /// </summary>
        private static bool HasConsumerOutside(
            Block block, HashSet<int> cone, Block ifBlock, FastNodeKey ifOpenKey,
            Dictionary<FastNodeKey, int> blockOfNode, Context ctx)
        {
            foreach (var nk in block.Keys)
            {
                if (!ctx.ConsumersOf.TryGetValue(nk, out var consumers)) continue;
                foreach (var c in consumers)
                {
                    if (c == ifOpenKey) return true;
                    if (ifBlock.Keys.Contains(c)) continue;
                    if (blockOfNode.TryGetValue(c, out var cb) && cone.Contains(cb)) continue;
                    return true;
                }
            }
            return false;
        }

        private static bool IsMovable(Block block, Context ctx)
        {
            foreach (var nk in block.Keys)
                if (ctx.Pinned.Contains(nk)) return false;
            return true;
        }
    }
}
