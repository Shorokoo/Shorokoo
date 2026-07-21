using Shorokoo.Core.Factory.IR;
using Shorokoo.Core.Graph;
using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.OnnxNodes;
using System;
using System.Collections.Generic;
using System.Linq;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using Shorokoo.Onnx;

namespace Shorokoo.Core.Factory
{
    /// <summary>
    /// Helpers for the Fast ONNX builder that turn the implicit subgraph membership of a
    /// <see cref="InternalComputationGraph"/> (where IF/LOOP bodies are just the positional
    /// band between an open node and its matching close node) into explicit per-branch
    /// node lists.
    ///
    /// <para>
    /// The IF body is split into <em>then</em> and <em>else</em> via a back-walk
    /// described by the user: starting at <c>IF_CLOSE - 1</c>, walk backwards
    /// accumulating into the <em>else</em> bucket until we encounter the first node
    /// whose output appears in the IF_CLOSE's <c>then_tensors</c> input group; from
    /// there on, the remaining backward sweep accumulates into the <em>then</em>
    /// bucket. Nested scopes are skipped past atomically — when we encounter a
    /// <c>*_CLOSE</c> node during the back-walk, we jump to its matching open and
    /// treat the whole inner block as one unit (still routed by whichever bucket the
    /// outer back-walk is currently filling).
    /// </para>
    ///
    /// <para>
    /// The <c>LOOP_CLOSE</c> body is just the entire band — no bifurcation needed.
    /// </para>
    /// </summary>
    internal static class FastSubgraphExtractor
    {
        /// <summary>
        /// Index pairing for every open node in <c>graph</c>: each
        /// <c>OpenIndex → CloseIndex</c> and the reverse mapping. Built once per
        /// build pass; <see cref="BifurcateIfBody"/> needs both directions to
        /// jump past nested scopes during the back-walk.
        /// </summary>
        public sealed class ScopeIndex
        {
            public Dictionary<int, int> OpenIdxToCloseIdx { get; } = new();
            public Dictionary<int, int> CloseIdxToOpenIdx { get; } = new();
        }

        /// <summary>
        /// Builds a <see cref="ScopeIndex"/> for <paramref name="graph"/>. Throws on
        /// unmatched open nodes or close nodes whose <c>GraphOpenNodeKey</c> doesn't
        /// resolve to a node in the graph.
        /// </summary>
        public static ScopeIndex BuildScopeIndex(InternalComputationGraph graph)
        {
            var keyToIndex = new Dictionary<FastNodeKey, int>();
            for (int i = 0; i < graph.Nodes.Count; i++)
                keyToIndex[graph.Nodes[i].Key] = i;

            var index = new ScopeIndex();
            for (int i = 0; i < graph.Nodes.Count; i++)
            {
                var n = graph.Nodes[i];
                if (!FastOpsetResolver.IsCloseOpCode(n.OpCode)) continue;
                if (n.GraphOpenNodeKey is not FastNodeKey openKey || openKey.IsEmpty)
                    throw new InvalidOperationException(
                        $"FastSubgraphExtractor: close node at index {i} ({n.OpCode}) has no GraphOpenNodeKey.");
                if (!keyToIndex.TryGetValue(openKey, out var openIdx))
                    throw new InvalidOperationException(
                        $"FastSubgraphExtractor: close node at index {i} references open key {openKey} which is not in the graph.");
                index.OpenIdxToCloseIdx[openIdx] = i;
                index.CloseIdxToOpenIdx[i] = openIdx;
            }
            return index;
        }

        /// <summary>
        /// Direct-child node indices strictly between <paramref name="openIdx"/>
        /// and <paramref name="closeIdx"/>: every index in <c>(openIdx,closeIdx)</c>
        /// except those that fall <em>inside</em> a nested scope. A nested
        /// scope's CLOSE index is kept (its NodeProto carries the inner subgraph
        /// as an attribute), but the nested OPEN and the nested body are
        /// excluded — those nodes belong to the inner scope's own subgraph and
        /// must not be re-emitted at the outer level.
        /// </summary>
        public static List<int> BodyBand(ScopeIndex scopeIndex, int openIdx, int closeIdx)
        {
            var list = new List<int>();
            int i = openIdx + 1;
            while (i < closeIdx)
            {
                if (scopeIndex.OpenIdxToCloseIdx.TryGetValue(i, out var nestedClose) &&
                    nestedClose < closeIdx)
                {
                    // Skip the nested OPEN and its body, keep only the nested CLOSE.
                    list.Add(nestedClose);
                    i = nestedClose + 1;
                }
                else
                {
                    list.Add(i);
                    i++;
                }
            }
            return list;
        }

        /// <summary>
        /// Splits the body band of an <c>IF_OPEN</c>/<c>IF_CLOSE</c> pair into
        /// <em>then</em> and <em>else</em> node-index lists in topological order.
        /// Implements the back-walk described in the class summary.
        /// </summary>
        /// <exception cref="InvalidOperationException">
        /// Thrown when <paramref name="ifCloseIdx"/> isn't actually an IF_CLOSE, or
        /// when the close node has no <c>then_tensors</c> input group (which would
        /// mean the graph is malformed).
        /// </exception>
        public static (List<int> thenIdxs, List<int> elseIdxs) BifurcateIfBody(
            InternalComputationGraph graph,
            ScopeIndex scopeIndex,
            int ifOpenIdx,
            int ifCloseIdx)
        {
            var ifClose = graph.Nodes[ifCloseIdx];
            if (ifClose.OpCode != OpCodes.IF_CLOSE)
                throw new InvalidOperationException(
                    $"FastSubgraphExtractor.BifurcateIfBody: index {ifCloseIdx} is not an IF_CLOSE (op={ifClose.OpCode}).");

            // The IF_CLOSE has its branch outputs split across two named input groups
            // — "else_tensors" (group 0) and "then_tensors" (group 1) — see
            // Definitions.HL.cs's PairGraphClose. We only need the then-tensor key
            // set for the back-walk; everything trailing the last then-producer is
            // implicitly else.
            var thenKeys = new HashSet<FastTensorKey>();
            if (ifClose.FullInputs.TryGetValue(OnnxOpAttributeNames.AttrThenBranch, out var thenInputs))
            {
                foreach (var k in thenInputs)
                    if (k is FastTensorKey tk && !tk.IsEmpty)
                        thenKeys.Add(tk);
            }
            else
            {
                // Fall back: positional second half of input slots when the close
                // hasn't been keyed by branch attribute name. Defensive — shouldn't
                // happen in practice for IF_CLOSE built via the standard path.
                var allInputs = ifClose.Inputs;
                int half = allInputs.Count / 2;
                for (int slot = half; slot < allInputs.Count; slot++)
                    if (allInputs[slot] is FastTensorKey tk && !tk.IsEmpty)
                        thenKeys.Add(tk);
            }

            var thenIdxs = new List<int>();
            var elseIdxs = new List<int>();
            bool inElsePhase = true;

            int i = ifCloseIdx - 1;
            while (i > ifOpenIdx)
            {
                int blockStart, blockEnd;
                if (scopeIndex.CloseIdxToOpenIdx.TryGetValue(i, out var nestedOpen))
                {
                    // i is a nested CLOSE — black-box the whole nested scope.
                    blockEnd = i;
                    blockStart = nestedOpen;
                }
                else
                {
                    blockStart = blockEnd = i;
                }

                // While still in the else phase, check whether this block contributes
                // a then-input. For nested-scope blocks we only inspect the close
                // node's outputs, since by construction nothing inside a nested scope
                // is visible to outer consumers except via the close node.
                if (inElsePhase)
                {
                    bool blockProducesThenInput;
                    if (blockStart == blockEnd)
                    {
                        blockProducesThenInput = NodeProducesAnyKey(graph.Nodes[blockEnd], thenKeys);
                    }
                    else
                    {
                        // Nested scope — only the close node's outputs count.
                        blockProducesThenInput = NodeProducesAnyKey(graph.Nodes[blockEnd], thenKeys);
                    }
                    if (blockProducesThenInput) inElsePhase = false;
                }

                var bucket = inElsePhase ? elseIdxs : thenIdxs;
                if (blockStart == blockEnd)
                {
                    // Plain body node — add directly.
                    bucket.Add(blockStart);
                }
                else
                {
                    // Nested scope — only the close (with its attached subgraph)
                    // belongs in the outer bucket. The nested body lives in the
                    // close's graph attribute and must not be re-emitted here.
                    bucket.Add(blockEnd);
                }

                i = blockStart - 1;
            }

            // We accumulated both lists in reverse topological order; restore topo order.
            thenIdxs.Reverse();
            elseIdxs.Reverse();
            return (thenIdxs, elseIdxs);
        }

        private static bool NodeProducesAnyKey(FastNode node, HashSet<FastTensorKey> keys)
        {
            foreach (var slot in node.FullOutputs.Values)
                foreach (var k in slot)
                    if (k is FastTensorKey tk && !tk.IsEmpty && keys.Contains(tk))
                        return true;
            return false;
        }

        /// <summary>
        /// Outputs declared on the IF_CLOSE for a particular branch attribute
        /// (<c>then_branch</c> or <c>else_branch</c>). These become the GraphProto
        /// outputs for that branch's subgraph. They are read off the close node's
        /// <see cref="FastNode.FullInputs"/> bucket keyed by the same attribute
        /// name (since branch outputs are CG-side close-node <em>inputs</em>).
        /// </summary>
        public static IReadOnlyList<FastTensorKey?> BranchOutputs(FastNode ifClose, string branchAttr)
        {
            if (ifClose.FullInputs.TryGetValue(branchAttr, out var slot))
                return slot;
            return Array.Empty<FastTensorKey?>();
        }

        /// <summary>
        /// LOOP body outputs: the close node's <c>FullInputs[AttrBody]</c> slot.
        /// </summary>
        public static IReadOnlyList<FastTensorKey?> LoopBodyOutputs(FastNode loopClose)
            => BranchOutputs(loopClose, OnnxOpAttributeNames.AttrBody);
    }
}
