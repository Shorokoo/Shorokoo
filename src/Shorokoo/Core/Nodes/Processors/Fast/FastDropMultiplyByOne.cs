using System.Collections.Generic;
using System.Linq;
using Shorokoo.Core.Backends;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Graph;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// Reads <c>x</c> where a lowered training step reads <c>Mul(x, 1)</c> by a rank-0 floating-point
    /// constant one of any width, and drops the Mul. Such a product is <c>x</c> to the bit, in shape and dtype, so
    /// nothing changes but a pass over <c>x</c>: an optimizer's decay factor <c>1 − lr·wd</c> folds to
    /// that constant when its weight decay is a baked zero, and the product it scales is then the
    /// whole parameter, every step. A factor fed at run time is not a constant and is left alone.
    ///
    /// <para>Only a Mul at the top level of the graph is dropped, and only its body consumers are
    /// rewired; one a graph output reads directly keeps the Mul, so an output never becomes an
    /// alias of another tensor.</para>
    /// </summary>
    internal static class FastDropMultiplyByOne
    {
        public static void Process(InternalComputationGraph graph)
        {
            var nodeByKey = graph.Nodes.ToDictionary(n => n.Key);
            var replacement = new Dictionary<FastTensorKey, FastTensorKey>();
            int depth = 0;
            foreach (var node in graph.Nodes)
            {
                if (node.IsCloseNode()) depth--;
                if (depth == 0 && node.OpCode == OpCodes.MUL
                    && node.Inputs is [FastTensorKey left, FastTensorKey right]
                    && node.Outputs is [FastTensorKey product])
                {
                    if (IsScalarOne(nodeByKey, right)) replacement[product] = left;
                    else if (IsScalarOne(nodeByKey, left)) replacement[product] = right;
                }
                if (node.IsOpenNode()) depth++;
            }
            if (replacement.Count == 0) return;

            var outputKeys = graph.OutputNodes.Select(n => n.Key).ToHashSet();
            foreach (var node in graph.Nodes)
            {
                if (outputKeys.Contains(node.Key)) continue;
                foreach (var slots in node.FullInputs.Values)
                    for (int i = 0; i < slots.Count; i++)
                        if (slots[i] is FastTensorKey k && replacement.TryGetValue(k, out var x))
                            slots[i] = x;
            }
            FastProcessorHelper.RemoveUnreachableNodes(graph);
        }

        private static bool IsScalarOne(Dictionary<FastNodeKey, FastNode> nodeByKey, FastTensorKey key)
        {
            if (!nodeByKey.TryGetValue(key.FastNodeKey, out var producer) || producer.OpCode != OpCodes.CONSTANT)
                return false;
            var value = producer.Attributes.GetAttributeVal(OnnxOpAttributeNames.AttrValue);
            if (value is null || value.Shape.Dims.Length != 0) return false;
            if (value.DType == DType.Float32) return value.Elements<float>()[0] == 1f;
            if (value.DType == DType.Float64) return value.Elements<double>()[0] == 1d;
            if (value.DType == DType.Float16) return value.Elements<Float16>()[0].Bits == Float16.One.Bits;
            if (value.DType == DType.BFloat16) return value.Elements<BFloat16>()[0].Bits == BFloat16.One.Bits;
            return false;
        }
    }
}
