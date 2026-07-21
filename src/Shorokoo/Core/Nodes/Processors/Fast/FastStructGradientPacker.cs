using Shorokoo.Graph;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using System;
using Shorokoo.Core.Nodes.Processors.AutoGrad;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// Packs an array of gradient <see cref="FastTensorKey"/>s into a
    /// <c>TENSOR_STRUCT_CREATE</c> node matching a <see cref="TensorStructDef"/>,
    /// appending the new node to the host graph.
    /// </summary>
    internal static class FastStructGradientPacker
    {
        /// <summary>
        /// Emits a <c>TENSOR_STRUCT_CREATE</c> FastNode into <paramref name="graph"/> with the
        /// supplied <paramref name="gradientKeys"/> as field values, in the field order of
        /// <paramref name="structDefinition"/>. Returns the struct's output <see cref="FastTensorKey"/>.
        /// </summary>
        public static FastTensorKey PackGradients(
            InternalComputationGraph graph,
            TensorStructDef structDefinition,
            FastTensorKey[] gradientKeys)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));
            if (structDefinition is null) throw new ArgumentNullException(nameof(structDefinition));
            if (gradientKeys is null) throw new ArgumentNullException(nameof(gradientKeys));

            if (gradientKeys.Length != structDefinition.Fields.Length)
                throw new ArgumentException(
                    $"Gradient array length ({gradientKeys.Length}) must match struct field count " +
                    $"({structDefinition.Fields.Length}).",
                    nameof(gradientKeys));

            var structDType = DType.GetOrCreateForTensorStruct(structDefinition);
            var node = FastInternalOp.TensorStructCreate(structDType, gradientKeys);
            graph.Nodes.Add(node);
            return new FastTensorKey(node.Key, 0);
        }
    }
}
