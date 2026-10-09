using System.Collections.Generic;
using System.Linq;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Graph;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// Multiplies the scalar factors of a product of several tensors together before they meet the
    /// tensors, so that a product scaled by several scalars along the way is scaled once.
    ///
    /// <para>A lowered training step's backward pass chains elementwise products: the gradient of
    /// <c>tanh(x / c) · c</c> under a mean loss is the loss gradient times the mean's <c>1/N</c>,
    /// times <c>c</c>, times <c>1 − y²</c>, times <c>1/c</c>, each product a pass over the whole
    /// tensor. The scalars among those factors commute with everything else in the product, so the
    /// product is rebuilt with the scalars multiplied together where they cost nothing, the rank-0
    /// constants among them folded to one constant, and a constant product of one dropped (a tensor
    /// left multiplied by one alone is then the tensor, see <see cref="FastDropMultiplyByOne"/>):
    /// one pass over the tensor for all of them, or none.</para>
    ///
    /// <para>The scale goes where the outermost scalar of the original went — onto the product of
    /// the tensor factors that scalar met, ahead of the others — so it never meets a larger product
    /// than one a scalar met before: scales a weight met before it broadcast over a batch stay on the
    /// weight. Where the scalars met products none of which holds all the others, the product is
    /// left as it is.</para>
    ///
    /// <para>A product is a tree of top-level <c>Mul</c>s: each <c>Mul</c> it reads that is read by
    /// nothing else (and is no graph output) belongs to it, and whatever else it reads is one of
    /// its factors. A factor is a scalar where it is one by construction
    /// (<see cref="FastScalarValues"/>), so it never changes the product's shape. Only a tree with
    /// at least two scalar factors is rebuilt — one scalar is already one pass — and the tree's root
    /// keeps its node, so whatever reads the product reads it where it did. Reassociating a
    /// floating-point product moves its rounding, and, for scalars at the ends of the range, where
    /// it overflows or underflows.</para>
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

            var scalarValues = FastScalarValues.Find(graph);
            bool IsScalar(FastTensorKey key) => scalarValues.Contains(key);

            bool changed = false;
            foreach (var root in graph.Nodes.Where(n => IsProduct(n) && !Absorbed((FastTensorKey)n.Outputs[0]!)).ToList())
            {
                // The factors in order, the Muls inside the product, and for each scalar factor the
                // tensor factors of the operand it scales: what it was applied to.
                var inner = new List<FastNode>();
                var factors = new List<FastTensorKey>();
                var appliedTo = new List<HashSet<FastTensorKey>>();
                HashSet<FastTensorKey> Collect(FastNode mul)
                {
                    var sides = new HashSet<FastTensorKey>[2];
                    var scalarSide = new bool[2];
                    for (int i = 0; i < 2; i++)
                    {
                        var key = (FastTensorKey)mul.Inputs[i]!;
                        // A side made of scalars alone, multiplied together first, is one scalar.
                        if (Absorbed(key)) { inner.Add(producer[key]); sides[i] = Collect(producer[key]); scalarSide[i] = sides[i].Count == 0; }
                        else
                        {
                            factors.Add(key);
                            scalarSide[i] = IsScalar(key);
                            sides[i] = scalarSide[i] ? [] : [key];
                        }
                    }
                    for (int i = 0; i < 2; i++)
                        if (scalarSide[i]) appliedTo.Add(sides[1 - i]);
                    return [.. sides[0], .. sides[1]];
                }
                Collect(root);
                var scalars = factors.Where(IsScalar).ToList();
                var tensors = factors.Where(f => !IsScalar(f)).ToList();
                if (scalars.Count < 2 || tensors.Count == 0) continue;

                // The scale goes where the outermost scalar went, which holds where every other one
                // went: never onto a larger product than a scalar of the original met. Scalars that
                // met products none of which holds the others stay where they are.
                var target = appliedTo.MaxBy(a => a.Count)!;
                if (appliedTo.Any(a => !a.IsSubsetOf(target))) continue;
                var scaled = tensors.Where(target.Contains).ToList();
                var rest = tensors.Where(t => !target.Contains(t)).ToList();

                var (constant, runtime) = Fold(scalars, producer);
                var scale = new List<FastTensorKey>(runtime);
                var built = new List<FastNode>();
                if (constant is not null)
                {
                    built.Add(constant);
                    scale.Insert(0, (FastTensorKey)constant.Outputs[0]!);
                }
                // A scale that folds to one leaves a single tensor multiplied by one, which the drop of
                // products by one that follows reads as the tensor itself.
                if (scale.Count == 0 && tensors.Count == 1)
                {
                    var one = One(scalars, producer);
                    built.Add(one);
                    scale.Add((FastTensorKey)one.Outputs[0]!);
                }

                FastTensorKey Multiply(FastTensorKey a, FastTensorKey b)
                {
                    var mul = NewMul(a, b);
                    built.Add(mul);
                    return (FastTensorKey)mul.Outputs[0]!;
                }

                // The terms, multiplied from the left: the scaled tensors, the scale, then the others.
                List<FastTensorKey> terms = [.. scaled];
                if (scale.Count > 0)
                {
                    var scaleProduct = scale[0];
                    for (int t = 1; t < scale.Count; t++) scaleProduct = Multiply(scaleProduct, scale[t]);
                    terms.Add(scaleProduct);
                }
                terms.AddRange(rest);
                var left = terms[0];
                for (int t = 1; t < terms.Count - 1; t++) left = Multiply(left, terms[t]);

                root.FullInputs.Clear();
                root.FullInputs[""] = new List<FastTensorKey?> { left, terms[^1] };
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

        // A rank-0 one of the type of the constants among the scalars.
        private static FastNode One(List<FastTensorKey> scalars, Dictionary<FastTensorKey, FastNode> producer)
        {
            var wide = scalars.Any(k => producer.TryGetValue(k, out var p) && p.OpCode == OpCodes.CONSTANT
                && p.Attributes.GetAttributeVal(OnnxOpAttributeNames.AttrValue) is { DType: var d } && d == DType.Float64);
            return FastInternalOp.Constant(wide ? TensorAttribute.Create(new Shape(), 1d) : TensorAttribute.Create(new Shape(), 1f));
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
