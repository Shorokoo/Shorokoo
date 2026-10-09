using System.Collections.Generic;
using System.Linq;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Graph;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// Multiplies the scalar factors of a product of several tensors together before they meet the
    /// tensors, so that a product scaled by several scalars along the way is scaled once, at its end.
    ///
    /// <para>A lowered training step's backward pass chains elementwise products: the gradient of
    /// <c>tanh(x / c) · c</c> under a mean loss is the loss gradient times the mean's <c>1/N</c>,
    /// times <c>c</c>, times <c>1 − y²</c>, times <c>1/c</c>, each product a pass over the whole
    /// tensor. The scalars among those factors commute with everything else in the product, so
    /// the product is rebuilt as the tensors' product times the scalars' product: the scalars
    /// multiply each other where they cost nothing, the rank-0 constants among them fold to one
    /// constant, and a constant product of one drops out, which leaves one pass over the tensor
    /// for all of them, or none.</para>
    ///
    /// <para>A product is a tree of top-level <c>Mul</c>s: each <c>Mul</c> it reads that is read by
    /// nothing else (and is no graph output) belongs to it, and whatever else it reads is one of
    /// its factors. A factor is a scalar where it is one by construction
    /// (<see cref="FastScalarValues"/>), so it can never
    /// change the product's shape. Only a tree with at least two scalar factors is rebuilt — one
    /// scalar is already one pass — and the tree's root keeps its node, so whatever reads the
    /// product reads it where it did. Reassociating a floating-point product moves its rounding,
    /// within a unit or two in the last place of each element.</para>
    /// </summary>
    internal static class FastFoldScalarFactors
    {
        public static void Process(InternalComputationGraph graph)
        {
            var producer = new Dictionary<FastTensorKey, FastNode>();
            var readers = new Dictionary<FastTensorKey, int>();
            var reader = new Dictionary<FastTensorKey, FastNode>();
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
                        readers[key] = readers.GetValueOrDefault(key) + 1;
                        reader[key] = node;
                    }
                if (node.IsOpenNode()) depth++;
            }
            var outputs = graph.OutputNodes.SelectMany(n => n.Inputs).OfType<FastTensorKey>().ToHashSet();

            bool IsProduct(FastNode node) => node.OpCode == OpCodes.MUL && topLevel.Contains(node)
                && node.Inputs is [FastTensorKey, FastTensorKey] && node.Outputs is [FastTensorKey];

            // A Mul read only by another Mul of the product is inside the product, not its root.
            bool Absorbed(FastTensorKey key) => producer.TryGetValue(key, out var p) && IsProduct(p)
                && readers.GetValueOrDefault(key) == 1 && !outputs.Contains(key) && IsProduct(reader[key]);

            var roots = new List<(FastNode Root, List<FastNode> Inner, List<FastTensorKey> Factors)>();
            foreach (var node in graph.Nodes)
            {
                if (!IsProduct(node)) continue;
                var product = (FastTensorKey)node.Outputs[0]!;
                if (Absorbed(product)) continue;
                var inner = new List<FastNode>();
                var factors = new List<FastTensorKey>();
                void Collect(FastNode mul)
                {
                    foreach (var input in mul.Inputs)
                    {
                        var key = (FastTensorKey)input!;
                        if (Absorbed(key)) { inner.Add(producer[key]); Collect(producer[key]); }
                        else factors.Add(key);
                    }
                }
                Collect(node);
                if (factors.Count >= 3) roots.Add((node, inner, factors));
            }
            if (roots.Count == 0) return;

            var scalarValues = FastScalarValues.Find(graph);
            bool IsScalar(FastTensorKey key) => scalarValues.Contains(key);

            bool changed = false;
            foreach (var (root, inner, factors) in roots)
            {
                var scalars = factors.Where(IsScalar).ToList();
                var tensors = factors.Where(f => !IsScalar(f)).ToList();
                if (scalars.Count < 2 || tensors.Count == 0) continue;

                var (constant, runtime) = Fold(scalars, producer);
                var scale = new List<FastTensorKey>(runtime);
                var built = new List<FastNode>();
                if (constant is not null)
                {
                    built.Add(constant);
                    scale.Insert(0, (FastTensorKey)constant.Outputs[0]!);
                }
                if (scale.Count == 0 && tensors.Count == 1) continue;

                FastTensorKey Multiply(FastTensorKey a, FastTensorKey b)
                {
                    var mul = NewMul(a, b);
                    built.Add(mul);
                    return (FastTensorKey)mul.Outputs[0]!;
                }

                var tensorProduct = tensors[0];
                var lastTensor = scale.Count == 0 ? tensors.Count - 1 : tensors.Count;
                for (int t = 1; t < lastTensor; t++) tensorProduct = Multiply(tensorProduct, tensors[t]);
                FastTensorKey rootLeft, rootRight;
                if (scale.Count == 0)
                {
                    (rootLeft, rootRight) = (tensorProduct, tensors[^1]);
                }
                else
                {
                    var scaleProduct = scale[0];
                    for (int s = 1; s < scale.Count; s++) scaleProduct = Multiply(scaleProduct, scale[s]);
                    (rootLeft, rootRight) = (tensorProduct, scaleProduct);
                }

                root.FullInputs.Clear();
                root.FullInputs[""] = new List<FastTensorKey?> { rootLeft, rootRight };
                graph.Nodes.InsertRange(graph.Nodes.IndexOf(root), built);
                foreach (var gone in inner) graph.Nodes.Remove(gone);
                changed = true;
            }
            if (changed) FastProcessorHelper.RemoveUnreachableNodes(graph);
        }

        // The rank-0 float constants among the scalars multiplied into one constant (none where
        // there are none, or where they multiply to one), and the scalars that are not constants.
        private static (FastNode? Constant, List<FastTensorKey> Runtime) Fold(
            List<FastTensorKey> scalars, Dictionary<FastTensorKey, FastNode> producer)
        {
            var runtime = new List<FastTensorKey>();
            float? single = null;
            double? wide = null;
            foreach (var key in scalars)
            {
                var value = producer.TryGetValue(key, out var p) && p.OpCode == OpCodes.CONSTANT
                    ? p.Attributes.GetAttributeVal(OnnxOpAttributeNames.AttrValue) : null;
                if (value is { DType: var d } && d == DType.Float32 && value.Shape.Dims.Length == 0)
                    single = (single ?? 1f) * value.Elements<float>()[0];
                else if (value is { DType: var e } && e == DType.Float64 && value.Shape.Dims.Length == 0)
                    wide = (wide ?? 1d) * value.Elements<double>()[0];
                else
                    runtime.Add(key);
            }
            if (single is float f && f != 1f)
                return (FastInternalOp.Constant(TensorAttribute.Create(new Shape(), f)), runtime);
            if (wide is double w && w != 1d)
                return (FastInternalOp.Constant(TensorAttribute.Create(new Shape(), w)), runtime);
            return (null, runtime);
        }

        private static FastNode NewMul(FastTensorKey a, FastTensorKey b)
        {
            var key = FastNodeKey.New();
            return new FastNode
            {
                Key = key,
                OpCode = OpCodes.MUL,
                Attributes = OnnxCSharpAttributes.FromCSharpVals(new Dictionary<string, object?>(),
                    Definitions.NodeDefinitions[OpCodes.MUL].AttributeDefs),
                FullInputs = { [""] = new List<FastTensorKey?> { a, b } },
                FullOutputs = { [""] = new List<FastTensorKey?> { new FastTensorKey(key, 0) } },
            };
        }
    }
}
