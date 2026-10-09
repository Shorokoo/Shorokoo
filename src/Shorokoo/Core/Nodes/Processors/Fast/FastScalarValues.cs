using System.Collections.Generic;
using System.Linq;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Graph;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// The values of a graph that are scalars by construction — rank 0 whatever the graph is fed —
    /// worked out from the nodes alone, in one walk: a <c>Constant</c> of rank 0; a <c>Size</c>; a
    /// reduction over every axis that keeps none; a <c>Reshape</c> to a constant empty shape; and an
    /// elementwise operator of scalars. A value the walk cannot settle is not a scalar here.
    /// </summary>
    internal static class FastScalarValues
    {
        private static readonly HashSet<string> Elementwise = new(System.StringComparer.Ordinal)
        {
            OpCodes.ADD, OpCodes.SUB, OpCodes.MUL, OpCodes.DIV, OpCodes.POW, OpCodes.CAST, OpCodes.NEG,
            OpCodes.RECIPROCAL, OpCodes.SQRT, OpCodes.EXP, OpCodes.LOG, OpCodes.ABS, OpCodes.IDENTITY,
        };

        private static readonly HashSet<string> Reductions = new(System.StringComparer.Ordinal)
        {
            OpCodes.REDUCE_SUM, OpCodes.REDUCE_MEAN, OpCodes.REDUCE_MAX, OpCodes.REDUCE_MIN, OpCodes.REDUCE_PROD,
        };

        public static HashSet<FastTensorKey> Find(InternalComputationGraph graph) => FindWithStatic(graph).Scalars;

        /// <summary>
        /// The scalars by construction, and those of them that are static: computed from constants
        /// and the sizes of values alone, so a session built at concrete dimensions holds each as a
        /// constant.
        /// </summary>
        public static (HashSet<FastTensorKey> Scalars, HashSet<FastTensorKey> Static) FindWithStatic(InternalComputationGraph graph)
        {
            var scalars = new HashSet<FastTensorKey>();
            var isStatic = new HashSet<FastTensorKey>();
            var constants = new Dictionary<FastTensorKey, FastNode>();
            foreach (var node in graph.Nodes)
            {
                if (node.Outputs is not [FastTensorKey output]) continue;
                if (node.OpCode == OpCodes.CONSTANT) constants[output] = node;
                if (!IsScalar(node, scalars, constants)) continue;
                scalars.Add(output);
                if (node.OpCode is OpCodes.CONSTANT or OpCodes.SIZE
                    || (Elementwise.Contains(node.OpCode) && node.Inputs.All(i => i is FastTensorKey k && isStatic.Contains(k))))
                    isStatic.Add(output);
            }
            return (scalars, isStatic);
        }

        private static bool IsScalar(FastNode node, HashSet<FastTensorKey> scalars, Dictionary<FastTensorKey, FastNode> constants)
        {
            var inputs = node.Inputs;
            switch (node.OpCode)
            {
                case OpCodes.CONSTANT:
                    return ValueOf(node) is { } value && value.Shape.Dims.Length == 0;
                case OpCodes.SIZE:
                    return true;
                case OpCodes.RESHAPE:
                    return inputs.Count >= 2 && inputs[1] is FastTensorKey shape
                        && constants.TryGetValue(shape, out var c) && ValueOf(c) is { } dims && dims.Shape.Dims is [0];
                default:
                    if (Reductions.Contains(node.OpCode))
                        return node.Attributes.GetBoolVal(OnnxOpAttributeNames.AttrKeepdims) == false
                            && (inputs.Count < 2 || inputs[1] is null)
                            && node.Attributes.GetBoolVal(OnnxOpAttributeNames.AttrNoopWithEmptyAxes) != true;
                    return Elementwise.Contains(node.OpCode) && inputs.Count > 0
                        && inputs.All(i => i is FastTensorKey k && scalars.Contains(k));
            }
        }

        private static TensorAttribute? ValueOf(FastNode constant)
            => constant.Attributes.GetAttributeVal(OnnxOpAttributeNames.AttrValue);
    }
}
