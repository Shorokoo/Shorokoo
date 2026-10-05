using System;
using System.Collections.Generic;
using System.Linq;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Graph;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// ONNX-export pre-pass that rewires a RESHAPE whose target shape is a fully literal
    /// constant (1-D int64, no <c>0</c> entries, at most one <c>-1</c>) to consume the ROOT of
    /// any chain of RESHAPE/SQUEEZE/UNSQUEEZE producers feeding its data input. Such a reshape
    /// only depends on the element sequence — which every metadata-only step in the chain
    /// preserves — so bypassing the chain is semantics-preserving (a <c>0</c> entry would copy a
    /// dim from the direct producer, so those are excluded; <c>-1</c> derives from the invariant
    /// element count and composes fine). Bypassed producers keep serving any other consumers.
    ///
    /// <para>The static reshape then reads the chain's root directly, so it does not wait on
    /// reshapes whose shapes are computed at run time — GroupNorm's shape-restoring reshape, which
    /// feeds from <c>Shape(x)</c>, say — and a chain step nothing else reads any more is removed
    /// here. Left in the graph, such a step inside an <c>IF</c> branch reads a branch value and
    /// feeds nothing, so scoping finds nothing that holds it in the branch and hoists it, with what
    /// it reads, to the enclosing scope, where it runs whether or not the branch is taken and fails
    /// on what is valid only on the branch's own path.</para>
    ///
    /// <para>The walk deliberately stops at anything but RESHAPE/SQUEEZE/UNSQUEEZE — in
    /// particular at IDENTITY, which <see cref="FastAddIdentityForOuterScopeValues"/> has already
    /// inserted on cross-scope references by the time this runs, so composition never crosses a
    /// scope boundary.</para>
    /// </summary>
    internal static class FastComposeContiguousReshapes
    {
        public static void Process(InternalComputationGraph graph)
        {
            var nodeByKey = graph.Nodes.ToDictionary(n => n.Key);

            // Positional scope of every node: the innermost enclosing OPEN node's key (null =
            // top level). Prep-stage graphs keep OPEN…CLOSE spans contiguous and properly
            // nested, so a linear sweep suffices.
            var scopeOf = new Dictionary<FastNodeKey, FastNodeKey?>();
            var openStack = new Stack<FastNodeKey>();
            foreach (var n in graph.Nodes)
            {
                if (n.OpCode is OpCodes.LOOP_CLOSE or OpCodes.IF_CLOSE) openStack.Pop();
                scopeOf[n.Key] = openStack.Count > 0 ? openStack.Peek() : null;
                if (n.OpCode is OpCodes.LOOP_OPEN or OpCodes.IF_OPEN) openStack.Push(n.Key);
            }

            var bypassed = new HashSet<FastNodeKey>();
            foreach (var node in graph.Nodes)
            {
                if (node.OpCode != OpCodes.RESHAPE) continue;
                if (!node.FullInputs.TryGetValue("", out var inputs) || inputs.Count < 2) continue;
                if (inputs[0] is not { } dataKey || inputs[1] is not { } shapeKey) continue;
                if (!IsLiteralComposableShape(nodeByKey, shapeKey)) continue;

                // Walk up through metadata-only producers; every step preserves the element
                // sequence this reshape reinterprets. A SAME-SCOPE IDENTITY (e.g. a lowered
                // state-update link) is transparent too: it passes the element sequence through
                // unchanged. Cross-scope identities — the outer-scope wrapping
                // FastAddIdentityForOuterScopeValues inserts — remain barriers, so composition
                // never crosses a scope boundary.
                var nodeScope = scopeOf[node.Key];
                var root = dataKey;
                while (nodeByKey.TryGetValue(root.FastNodeKey, out var producer)
                       && root.OutputIndex == 0)
                {
                    bool composable = producer.OpCode == OpCodes.RESHAPE
                        || producer.OpCode == OpCodes.SQUEEZE
                        || producer.OpCode == OpCodes.UNSQUEEZE;
                    bool transparentIdentity = producer.OpCode == OpCodes.IDENTITY
                        && scopeOf.TryGetValue(producer.Key, out var producerScope)
                        && Nullable.Equals(producerScope, nodeScope);
                    if (!composable && !transparentIdentity) break;
                    if (!producer.FullInputs.TryGetValue("", out var producerInputs)
                        || producerInputs.Count < 1
                        || producerInputs[0] is not { } upstream) break;
                    bypassed.Add(producer.Key);
                    root = upstream;
                }

                if (root != dataKey) inputs[0] = root;
            }

            RemoveUnread(graph, bypassed);
        }

        /// <summary>Removes the bypassed chain steps nothing reads any more, and in turn the steps
        /// only they read.</summary>
        private static void RemoveUnread(InternalComputationGraph graph, HashSet<FastNodeKey> bypassed)
        {
            bool removed = true;
            while (removed && bypassed.Count > 0)
            {
                var read = new HashSet<FastNodeKey>();
                foreach (var node in graph.Nodes)
                    foreach (var slot in node.FullInputs.Values)
                        foreach (var key in slot)
                            if (key is { IsEmpty: false } k) read.Add(k.FastNodeKey);
                foreach (var key in graph.Inputs)
                    if (!key.IsEmpty) read.Add(key.FastNodeKey);

                var unread = bypassed.Where(k => !read.Contains(k)).ToHashSet();
                removed = graph.Nodes.RemoveAll(n => unread.Contains(n.Key)) > 0;
                bypassed.ExceptWith(unread);
            }
        }

        /// <summary>Whether the shape input is a CONSTANT whose value is a 1-D int64 tensor with
        /// no <c>0</c> entries and at most one <c>-1</c> — the target-shape form whose meaning is
        /// independent of the data producer's dims.</summary>
        private static bool IsLiteralComposableShape(
            Dictionary<FastNodeKey, FastNode> nodeByKey, FastTensorKey shapeKey)
        {
            if (!nodeByKey.TryGetValue(shapeKey.FastNodeKey, out var shapeNode)) return false;
            if (shapeNode.OpCode != OpCodes.CONSTANT) return false;
            var data = shapeNode.Attributes.GetAttributeVal(OnnxOpAttributeNames.AttrValue);
            if (data is null || data.DType != DType.Int64 || data.Shape.Dims.Length != 1) return false;
            var dims = data.Elements<long>().ToArray();
            int negOnes = 0;
            foreach (var d in dims)
            {
                if (d == 0) return false;
                if (d == -1 && ++negOnes > 1) return false;
            }
            return true;
        }
    }
}
