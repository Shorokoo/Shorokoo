using Shorokoo.Core.Factory;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Graph;
using System.Collections.Generic;
using System.Linq;

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
    /// <para>One reader outside the branch still leaves a value to it: another <c>IF</c> on the
    /// same condition that reads it on the same side, and so runs only when the branch does. The
    /// value is then handed out of its <c>IF</c> as a further output, which that reader reads
    /// instead, against a placeholder from the other branch that no reader sees (see
    /// <see cref="CollectExports"/>).</para>
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

            var inputs = graph.Inputs;
            List<FastNode> scoped;
            while (true)
            {
                var exports = new List<Export>();
                scoped = SinkLevel(graph.Nodes, ctx, exports);
                if (exports.Count == 0 || !ApplyExports(graph, exports)) break;
                ctx = new Context(graph);
            }
            graph.Nodes = scoped;
            graph.SetInputs(inputs);
            graph.MoveOutputsToEnd();

            System.Diagnostics.Debug.Assert(graph.IsLinearOrderValid(),
                "FastIfBranchScoper.ScopeAllIfBranches: scoping left the graph in an invalid linear order.");
        }

        /// <summary>
        /// Puts every <c>IF</c> branch back where it was traced — immediately before the
        /// <c>IF_OPEN</c> that selects it — undoing <see cref="ScopeAllIfBranches"/>.
        ///
        /// <para>A backward pass reads the forward's intermediates, and cannot read one the
        /// forward computes only on a branch it may not take. Autograd therefore flattens the
        /// branches first and emits at module scope; the simplify that follows it scopes what is
        /// left, now with the gradient nodes among the consumers (Shorokoo/Shorokoo#312). A value
        /// the backward reads only from the branch of a gate on the same condition goes back in
        /// and is handed out of the <c>IF</c> to it; one the backward reads anywhere else stays
        /// out. Scoping before the backward exists would move those values inside and then read
        /// them from outside, which is not a graph.</para>
        /// </summary>
        public static void UnscopeAllIfBranches(InternalComputationGraph graph)
        {
            if (graph is null) throw new System.ArgumentNullException(nameof(graph));

            // One band at a time, restarting: moving an outer branch out carries any scope nested
            // in it along whole, still scoped, for the next pass to reach.
            while (TryUnscopeOneIf(graph.Nodes)) { }

            System.Diagnostics.Debug.Assert(graph.TryValidateLinearOrder(out var orderError),
                "FastIfBranchScoper.UnscopeAllIfBranches: " + orderError);
        }

        private static bool TryUnscopeOneIf(List<FastNode> nodes)
        {
            for (int open = 0; open < nodes.Count; open++)
            {
                if (nodes[open].OpCode != OpCodes.IF_OPEN) continue;

                int depth = 0, close = -1;
                for (int i = open; i < nodes.Count; i++)
                {
                    if (FastOpsetResolver.IsOpenOpCode(nodes[i].OpCode)) depth++;
                    else if (FastOpsetResolver.IsCloseOpCode(nodes[i].OpCode) && --depth == 0) { close = i; break; }
                }
                if (close < 0 || close == open + 1) continue;   // unmatched, or already empty

                var body = nodes.GetRange(open + 1, close - open - 1);
                nodes.RemoveRange(open + 1, close - open - 1);
                nodes.InsertRange(open, body);
                return true;
            }
            return false;
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

                    if (InternalOpCodes.IsModelInputOp(n.OpCode) ||
                        InternalOpCodes.IsGraphOutputOp(n.OpCode) ||
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
        private static List<FastNode> SinkLevel(List<FastNode> level, Context ctx, List<Export> exports)
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

            for (int b = 0; b < blocks.Count; b++)
            {
                if (!blocks[b].IsIf) continue;
                CollectExports(blocks, b, blockOfNode, claim, isThen: true, ctx, exports);
                CollectExports(blocks, b, blockOfNode, claim, isThen: false, ctx, exports);
            }

            var moved = new List<FastNode>(level.Count);
            for (int b = 0; b < blocks.Count; b++)
                if (!claim.ContainsKey(b))
                    Assemble(b, blocks, claim, moved);

            return DescendIntoScopes(moved, ctx, exports);
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
        private static List<FastNode> DescendIntoScopes(List<FastNode> level, Context ctx, List<Export> exports)
        {
            var result = new List<FastNode>(level.Count);
            foreach (var blk in SplitIntoBlocks(level))
            {
                if (!blk.IsScope) { result.AddRange(blk.Nodes); continue; }
                result.Add(blk.Nodes[0]);
                result.AddRange(SinkLevel(blk.Nodes.GetRange(1, blk.Nodes.Count - 2), ctx, exports));
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

        /// <summary>A branch value to hand out of its <c>IF</c> as one more output, and the
        /// readers to hand it to.</summary>
        private sealed record Export(FastNode IfOpen, FastNode IfClose, bool IsThen, FastTensorKey Value,
                                    List<FastNodeKey> Readers);

        /// <summary>
        /// The values of one branch that only another <c>IF</c> on the same condition, and on the
        /// same side of it, reads from outside the branch.
        ///
        /// <para>Such a reader runs only when this branch does, yet it keeps the value, and
        /// everything the value is computed from, out of the branch: the cone above admits a value
        /// only when nothing outside it reads it. So they would run on every execution, and fail
        /// on what is valid only on the branch's own path. The backward of an <c>IfElse</c> arm is
        /// the case in point: it reads the arm's intermediates, and sits in the branch of the
        /// gate that hands its gradient out, on the arm's own condition.</para>
        ///
        /// <para>Handing the value out of its branch as a further output of the <c>IF</c> lets the
        /// branch keep it, since its reader then reads the <c>IF</c>. The other branch hands out a
        /// placeholder in its place, which no reader ever sees: each one sits in a branch taken
        /// only when this one is.</para>
        /// </summary>
        private static void CollectExports(
            List<Block> blocks, int ifIdx, Dictionary<FastNodeKey, int> blockOfNode,
            Dictionary<int, (int IfIdx, bool IsThen)> claim, bool isThen, Context ctx, List<Export> exports)
        {
            var ifBlock = blocks[ifIdx];
            var ifOpen = ifBlock.Nodes[0];
            var ifClose = ifBlock.Nodes[^1];
            if (ConditionOf(ifOpen) is not FastTensorKey condition) return;

            var mine = ReachedBlocks(ifClose, BranchAttr(isThen), ifIdx, blockOfNode, ctx);
            var other = ReachedBlocks(ifClose, BranchAttr(!isThen), ifIdx, blockOfNode, ctx);

            var kept = new HashSet<int>();
            foreach (var b in mine)
                if (!other.Contains(b) && IsMovable(blocks[b], ctx)
                    && (!claim.TryGetValue(b, out var cl) || cl == (ifIdx, isThen)))
                    kept.Add(b);

            bool SameSide(int b) => blocks[b].IsIf && b > ifIdx && ConditionOf(blocks[b].Nodes[0]) == condition;

            bool ReadOnlyWhenTaken(FastNode producer, FastNodeKey reader, int readerBlock)
            {
                if (readerBlock <= ifIdx) return false;
                if (SameSide(readerBlock) && blocks[readerBlock].Nodes[^1].Key == reader
                    && ReadOnlyOnSide(blocks[readerBlock].Nodes[^1], producer, isThen))
                    return true;
                for (int b = readerBlock; claim.TryGetValue(b, out var cl); b = cl.IfIdx)
                    if (cl.IsThen == isThen && SameSide(cl.IfIdx))
                        return true;
                return false;
            }

            // Each value either stays with the branch or is read from where the branch is taken.
            bool changed = true;
            while (changed)
            {
                changed = false;
                foreach (var b in kept.ToList())
                {
                    foreach (var producer in blocks[b].Nodes)
                    {
                        if (!ctx.ConsumersOf.TryGetValue(producer.Key, out var consumers)) continue;
                        foreach (var c in consumers)
                        {
                            if (c == ifOpen.Key) goto drop;
                            if (ifBlock.Keys.Contains(c)) continue;
                            if (!blockOfNode.TryGetValue(c, out var cb)) goto drop;
                            if (kept.Contains(cb) || ReadOnlyWhenTaken(producer, c, cb)) continue;
                            goto drop;
                        }
                    }
                    continue;
                drop:
                    kept.Remove(b);
                    changed = true;
                }
            }

            foreach (var b in kept)
                foreach (var node in blocks[b].Nodes)
                {
                    if (!ctx.ConsumersOf.TryGetValue(node.Key, out var consumers)) continue;
                    foreach (var grp in node.FullOutputs)
                        foreach (var key in grp.Value)
                        {
                            if (key is not FastTensorKey value || value.IsEmpty) continue;
                            var readers = new List<FastNodeKey>();
                            foreach (var c in consumers)
                                if (!ifBlock.Keys.Contains(c) && blockOfNode.TryGetValue(c, out var cb)
                                    && !kept.Contains(cb) && Reads(ctx.NodeByKey[c], value))
                                    readers.Add(c);
                            if (readers.Count > 0)
                                exports.Add(new Export(ifOpen, ifClose, isThen, value, readers));
                        }
                }
        }

        private static FastTensorKey? ConditionOf(FastNode ifOpen)
            => ifOpen.Inputs.Count > 0 ? ifOpen.Inputs[0] : null;

        /// <summary>Whether <paramref name="close"/> reads what <paramref name="producer"/> makes
        /// only as a value of the given branch.</summary>
        private static bool ReadOnlyOnSide(FastNode close, FastNode producer, bool isThen)
        {
            foreach (var grp in close.FullInputs)
            {
                if (grp.Key == BranchAttr(isThen)) continue;
                foreach (var k in grp.Value)
                    if (k is FastTensorKey tk && tk.FastNodeKey == producer.Key) return false;
            }
            return true;
        }

        private static bool Reads(FastNode node, FastTensorKey key)
        {
            foreach (var grp in node.FullInputs)
                foreach (var k in grp.Value)
                    if (k == key) return true;
            return false;
        }

        /// <summary>
        /// Hands each export's value out of its <c>IF</c> and rewires its readers to that output,
        /// against a placeholder of the value's dtype from the other branch: an empty tensor of the
        /// value's rank (of rank one where the rank is not known; an <c>If</c> may hand out
        /// values of different shapes from its two branches), or a zero where the value is a
        /// scalar. A value that is not a tensor of a known dtype stays where it is. Whether any
        /// export was applied.
        /// </summary>
        private static bool ApplyExports(InternalComputationGraph graph, List<Export> exports)
        {
            var tensorInfo = FastTensorInfoProcessor.BuildTensorInfoLookup(graph);
            var nodeByKey = new Dictionary<FastNodeKey, FastNode>();
            foreach (var n in graph.Nodes) nodeByKey[n.Key] = n;

            bool applied = false;
            foreach (var export in exports)
            {
                if (Placeholder(tensorInfo, export.Value) is not TensorAttribute placeholder) continue;

                var constant = FastInternalOp.Constant(placeholder);
                graph.Nodes.Insert(graph.Nodes.IndexOf(export.IfOpen), constant);
                var placeholderKey = new FastTensorKey(constant.Key, 0);

                var close = export.IfClose;
                var outputs = close.FullOutputs[""];
                var handedOut = new FastTensorKey(close.Key, outputs.Count);
                outputs.Add(handedOut);
                close.FullInputs[BranchAttr(export.IsThen)].Add(export.Value);
                close.FullInputs[BranchAttr(!export.IsThen)].Add(placeholderKey);

                foreach (var reader in export.Readers)
                    foreach (var grp in nodeByKey[reader].FullInputs)
                        for (int i = 0; i < grp.Value.Count; i++)
                            if (grp.Value[i] == export.Value) grp.Value[i] = handedOut;
                applied = true;
            }
            return applied;
        }

        private static TensorAttribute? Placeholder(Dictionary<FastTensorKey, FastTensorInfo> tensorInfo, FastTensorKey key)
        {
            if (!tensorInfo.TryGetValue(key, out var info) || info.Structure != DataStructure.Tensor
                || info.DType == DType.Invalid)
                return null;
            var dtype = info.DType.ToNonGenericType();
            var rank = info.Rank ?? 1;
            if (rank > 0) return TensorAttribute.Create(new Shape(new long[rank]), dtype, []);
            int bits = TensorData.StorageBits(dtype);
            return bits == 0 ? null : TensorAttribute.Create(new Shape(), dtype, new byte[(bits + 7) / 8]);
        }
    }
}
