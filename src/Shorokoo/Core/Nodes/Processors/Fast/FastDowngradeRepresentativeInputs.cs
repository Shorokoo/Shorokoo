using Shorokoo.Graph;
using Shorokoo.Core.Nodes.NodeDefinitions;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// Normalizes each <c>MODEL_TENSOR_INPUT</c> node's representative-input attribute for a size-limited
    /// serialization target: if the node carries an inline
    /// <see cref="OnnxOpAttributeNames.ShrkAttrRepresentativeInput"/> tensor with more than
    /// <c>maxInlineElements</c> elements, it is replaced with the shape-only
    /// <see cref="OnnxOpAttributeNames.ShrkAttrRepresentativeInputShape"/> attribute (dims only);
    /// a tensor at or below the limit, and an already shape-only attribute, are left as-is.
    ///
    /// <para>Run once, in place, on the builder's cloned graph before emission, so the "downgrade above N
    /// elements" decision is a graph transform on the same graph that is then serialized — never a
    /// re-decision at emission time. The native <c>.srk</c> passthrough target does not run this pass
    /// (equivalently N = ∞); vanilla ONNX export runs it with <see cref="VanillaMaxInlineElements"/> so the
    /// per-input ONNX metadata stays compact.</para>
    /// </summary>
    internal static class FastDowngradeRepresentativeInputs
    {
        /// <summary>The element-count limit vanilla ONNX export uses: an inline representative tensor over
        /// this many elements is downgraded to shape-only in the exported <c>.onnx</c> metadata.</summary>
        internal const int VanillaMaxInlineElements = 16;

        public static void Process(InternalComputationGraph graph, int maxInlineElements)
        {
            foreach (var node in graph.Nodes)
            {
                if (node.OpCode != InternalOpCodes.MODEL_TENSOR_INPUT) continue;
                if (!node.Attributes.IsAttributeDefined(OnnxOpAttributeNames.ShrkAttrRepresentativeInput))
                    continue;
                var inline = node.Attributes.GetTensorVal(OnnxOpAttributeNames.ShrkAttrRepresentativeInput);
                if (inline is null || inline.Shape.Count <= maxInlineElements) continue;

                node.Attributes = node.Attributes.SetAttributes(
                    (OnnxOpAttributeNames.ShrkAttrRepresentativeInputShape, (object?)inline.Shape.Dims),
                    (OnnxOpAttributeNames.ShrkAttrRepresentativeInput, (object?)null));
            }
        }
    }
}
