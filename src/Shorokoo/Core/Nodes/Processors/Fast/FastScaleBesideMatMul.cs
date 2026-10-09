using System.Collections.Generic;
using System.Linq;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Graph;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// Moves a product by a scalar across the <c>Reshape</c> that stands between it and a
    /// <c>MatMul</c>, so that the scale sits directly on the <c>MatMul</c>'s output or input.
    ///
    /// <para>A scale by a constant is free beside a matrix product where the backend folds it in:
    /// ONNX Runtime makes a <c>Mul</c> by a scalar constant that reads or feeds a <c>MatMul</c> the
    /// <c>alpha</c> of the <c>FusedMatMul</c> it rewrites the pair into. A <c>Reshape</c> in between
    /// hides the pair, and the scale costs a pass over the whole tensor. Both shapes occur at a
    /// language model's head: <c>MatMul</c>, <c>Reshape</c> to <c>[tokens, vocabulary]</c>, then
    /// <c>Mul</c> by a soft cap's <c>1/c</c> going forward; and coming back, the logits' gradient
    /// scaled once (<see cref="FastFoldScalarFactors"/>) and reshaped back to each of the two
    /// <c>MatMul</c>s that differentiate the projection. A scale commutes with a reshape, so:</para>
    /// <list type="bullet">
    ///   <item><c>Mul(Reshape(MatMul(…)), s)</c> becomes <c>Reshape(Mul(MatMul(…), s))</c> where
    ///         the <c>Reshape</c> reads nothing else's data and the <c>Mul</c> is its only
    ///         reader;</item>
    ///   <item><c>Mul(x, s)</c> read only by <c>Reshape</c>s, through which it flows into
    ///         <c>MatMul</c>s alone, becomes <c>Mul(Reshape(x), s)</c> for each of them. Across more
    ///         than one reshape the scale is multiplied once per reshape, which costs nothing only
    ///         where each is folded into its product; so that happens only for a static scale
    ///         (<see cref="FastScalarValues.FindWithStatic"/>), which a session built at concrete
    ///         dimensions holds as the constant such a fold needs.</item>
    /// </list>
    /// <para>The training rig runs it only for a backend that folds scales into matrix products
    /// (<c>IShorokooBackend.FoldsScalesIntoMatMul</c>): elsewhere a scale beside a product is no
    /// cheaper than one a reshape away, and one moved across several reshapes costs a pass per
    /// reshape.</para>
    /// <para>Only a top-level <c>Mul</c> by a rank-0 factor moves, and nothing a graph output
    /// reads. Each element is still multiplied by the same scalar once; a backend that then folds
    /// the scale into the product rounds it there.</para>
    /// </summary>
    internal static class FastScaleBesideMatMul
    {
        public static void Process(InternalComputationGraph graph)
        {
            // Each round moves a scale across one reshape; a scale behind a chain of them moves
            // once per round.
            bool any = false;
            while (Round(graph)) any = true;
            if (any) FastProcessorHelper.RemoveUnreachableNodes(graph);
        }

        private static bool Round(InternalComputationGraph graph)
        {
            var producer = new Dictionary<FastTensorKey, FastNode>();
            var readers = new Dictionary<FastTensorKey, List<FastNode>>();
            var topLevel = new HashSet<FastNode>(ReferenceEqualityComparer.Instance);
            int depth = 0;
            foreach (var node in graph.Nodes)
            {
                if (node.IsCloseNode()) depth--;
                if (depth == 0) topLevel.Add(node);
                foreach (var output in node.Outputs)
                    if (output is FastTensorKey key && !key.IsEmpty) producer[key] = node;
                foreach (var input in node.Inputs)
                    if (input is FastTensorKey key)
                    {
                        if (!readers.TryGetValue(key, out var list)) readers[key] = list = [];
                        list.Add(node);
                    }
                if (node.IsOpenNode()) depth++;
            }
            var outputs = graph.OutputNodes.SelectMany(n => n.Inputs).OfType<FastTensorKey>().ToHashSet();

            var candidates = graph.Nodes.Where(n => n.OpCode == OpCodes.MUL && topLevel.Contains(n)
                && n.Inputs is [FastTensorKey, FastTensorKey] && n.Outputs is [FastTensorKey]).ToList();
            if (candidates.Count == 0) return false;
            var (scalarValues, staticValues) = FastScalarValues.FindWithStatic(graph);
            bool IsScalar(FastTensorKey key) => scalarValues.Contains(key);

            List<FastNode> ReadersOf(FastTensorKey key) => readers.TryGetValue(key, out var list) ? list : [];
            // What reads a value's data: a Shape or Size reads only its shape, which no move changes.
            IEnumerable<FastNode> DataReadersOf(FastTensorKey key) => ReadersOf(key).Where(r => r.OpCode is not (OpCodes.SHAPE or OpCodes.SIZE));
            bool IsMatMul(FastNode node) => node.OpCode is OpCodes.MATMUL or OpCodes.GEMM;
            bool IsReshapeOf(FastNode node, FastTensorKey data) => node.OpCode == OpCodes.RESHAPE && topLevel.Contains(node)
                && node.Inputs.Count == 2 && node.Inputs[0] == data && node.Inputs[1] != data;
            // Whether everything a value flows into, through reshapes, is a matrix product.
            bool EndsInMatMuls(FastTensorKey key) => !outputs.Contains(key) && ReadersOf(key) is { Count: > 0 } all
                && all.All(r => IsMatMul(r) || (IsReshapeOf(r, key) && EndsInMatMuls((FastTensorKey)r.Outputs[0]!)));
            // A node rewritten here has stale entries in the maps above, so nothing that reads or
            // is read by one is looked at again in this round.
            var touched = new HashSet<FastNode>(ReferenceEqualityComparer.Instance);

            bool changed = false;
            foreach (var mul in candidates)
            {
                var (a, b) = ((FastTensorKey)mul.Inputs[0]!, (FastTensorKey)mul.Inputs[1]!);
                if (!IsScalar(b) && IsScalar(a)) (a, b) = (b, a);
                if (!IsScalar(b) || IsScalar(a) || touched.Contains(mul)) continue;
                var product = (FastTensorKey)mul.Outputs[0]!;

                // Going forward: Mul(Reshape(MatMul), s) -> Reshape(Mul(MatMul, s)).
                if (producer.TryGetValue(a, out var reshape) && reshape.Inputs is [FastTensorKey matmulOut, FastTensorKey]
                    && IsReshapeOf(reshape, matmulOut) && !touched.Contains(reshape) && !outputs.Contains(a)
                    && ReadersOf(a) is [var onlyMul] && ReferenceEquals(onlyMul, mul)
                    && producer.TryGetValue(matmulOut, out var matmul) && IsMatMul(matmul) && !outputs.Contains(matmulOut)
                    && DataReadersOf(matmulOut).All(r => ReferenceEquals(r, reshape)))
                {
                    var shape = reshape.Inputs[1];
                    var (reshapeAttributes, mulAttributes) = (reshape.Attributes, mul.Attributes);
                    // Each node keeps its key, which its readers know it by, and takes the other's
                    // operator: its name and call stack go with the operator.
                    (reshape.FriendlyName, mul.FriendlyName) = (mul.FriendlyName, reshape.FriendlyName);
                    (reshape.CallStack, mul.CallStack) = (mul.CallStack, reshape.CallStack);
                    reshape.OpCode = OpCodes.MUL;
                    reshape.Attributes = mulAttributes;
                    reshape.FullInputs.Clear();
                    reshape.FullInputs[""] = new List<FastTensorKey?> { matmulOut, b };
                    mul.OpCode = OpCodes.RESHAPE;
                    mul.Attributes = reshapeAttributes;
                    mul.FullInputs.Clear();
                    mul.FullInputs[""] = new List<FastTensorKey?> { a, shape };
                    graph.Nodes.Remove(reshape);
                    graph.Nodes.Insert(graph.Nodes.IndexOf(mul), reshape);
                    touched.UnionWith([mul, reshape, matmul]);
                    changed = true;
                    continue;
                }

                // Coming back: Mul(x, s) flowing through reshapes into matrix products only -> each
                // reshape reads x, and the scale follows it.
                // A scale on a MatMul's own output is already beside it: moving it on would undo
                // the move above and, at the next round, have it made again.
                if (producer.TryGetValue(a, out var source) && IsMatMul(source)) continue;
                var reshapes = ReadersOf(product);
                if (reshapes.Count == 0 || !EndsInMatMuls(product) || reshapes.Any(touched.Contains)) continue;
                var isStatic = staticValues.Contains(b);
                // A reshape of a reshape of the product, its shape taken literally (allowzero),
                // reshapes the product itself: rewired to read it, it is one more reshape the
                // scale can follow, rather than a reader the scale has to pass two reshapes for.
                var chained = isStatic && !reshapes.Any(IsMatMul) ? reshapes.SelectMany(r => ReadersOf((FastTensorKey)r.Outputs[0]!))
                    .Where(r2 => r2.OpCode == OpCodes.RESHAPE && topLevel.Contains(r2) && !touched.Contains(r2)
                        && r2.Attributes.GetBoolVal(OnnxOpAttributeNames.AttrAllowzero) == true).ToList() : [];
                if (chained.Count > 0)
                {
                    foreach (var r2 in chained)
                    {
                        var viewed = r2.Inputs[0];
                        foreach (var group in r2.FullInputs.Values)
                            for (int i = 0; i < group.Count; i++)
                                if (group[i] == viewed) group[i] = product;
                        touched.Add(r2);
                    }
                    touched.Add(mul);
                    changed = true;
                    continue;
                }
                if (reshapes.Any(IsMatMul) || (reshapes.Count > 1 && !isStatic)) continue;
                foreach (var r in reshapes)
                {
                    var key = FastNodeKey.New();
                    var view = new FastNode
                    {
                        Key = key,
                        OpCode = OpCodes.RESHAPE,
                        Attributes = r.Attributes,
                        FullInputs = { [""] = new List<FastTensorKey?> { a, r.Inputs[1] } },
                        FullOutputs = { [""] = new List<FastTensorKey?> { new FastTensorKey(key, 0) } },
                        FriendlyName = r.FriendlyName,
                        CallStack = r.CallStack,
                    };
                    (r.FriendlyName, r.CallStack) = (mul.FriendlyName, mul.CallStack);
                    r.OpCode = OpCodes.MUL;
                    r.Attributes = mul.Attributes;
                    r.FullInputs.Clear();
                    r.FullInputs[""] = new List<FastTensorKey?> { new FastTensorKey(key, 0), b };
                    graph.Nodes.Insert(graph.Nodes.IndexOf(r), view);
                    touched.Add(r);
                    touched.UnionWith(ReadersOf((FastTensorKey)r.Outputs[0]!));
                }
                touched.Add(mul);
                changed = true;
            }
            return changed;
        }
    }
}
