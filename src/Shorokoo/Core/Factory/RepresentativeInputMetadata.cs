using System;
using System.Linq;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes.NodeDefinitions;

namespace Shorokoo.Core.Factory
{
    /// <summary>
    /// The compact, symmetric string codec for a <c>MODEL_TENSOR_INPUT</c> node's representative-input
    /// attribute, carried in vanilla ONNX per-input <see cref="IR.ValueInfoProto"/> metadata (a graph
    /// input has no attribute bag). Both sides pair a node with its own ValueInfoProto intrinsically —
    /// the writer (<see cref="FastOnnxProtoFactory.CreateGraphInputInfo"/>) has the producing node and the
    /// ValueInfoProto in hand; the reader (<c>OnnxModelReader.CreateFastInputTensors</c>) has the
    /// ValueInfoProto and the FastNode it is building in hand — so there is no cross-graph index pairing.
    ///
    /// <para>The representative shape is always dims-only, so the codec has a single form:
    /// <c>"shape|{protoDtype}|{d0,d1,…}"</c>. The metadata is not redundant with the ValueInfoProto's
    /// own dims: those are unnamed (rank-only) placeholders, and stamping the concrete representative
    /// shape into them would falsely freeze the batch dimension for every external consumer.</para>
    /// </summary>
    internal static class RepresentativeInputMetadata
    {
        /// <summary>Per-input <see cref="IR.ValueInfoProto"/> metadata key for the representative-input info.</summary>
        public const string Key = "shrk_repr_input";

        /// <summary>
        /// Encodes the representative-input shape attribute currently set on <paramref name="node"/> (a
        /// <c>MODEL_TENSOR_INPUT</c>), or <c>null</c> when the attribute is not set.
        /// </summary>
        public static string? Encode(FastNode node)
        {
            var attrs = node.Attributes;

            if (attrs.IsAttributeDefined(OnnxOpAttributeNames.ShrkAttrRepresentativeInputShape)
                && attrs.GetLongsVal(OnnxOpAttributeNames.ShrkAttrRepresentativeInputShape) is { } dims
                && attrs.GetDTypeVal(OnnxOpAttributeNames.AttrDtype) is { } dtype)
            {
                return $"shape|{dtype.ProtoTypeNum}|{DimsToString(dims)}";
            }

            return null;
        }

        /// <summary>
        /// Decodes <paramref name="encoded"/> and sets the representative-shape attribute on
        /// <paramref name="node"/>. Fail-safe: a malformed value — wrong field count, an unknown form
        /// tag, or unparseable dims — is skipped, leaving the node untouched.
        /// </summary>
        public static void Apply(FastNode node, string encoded)
        {
            var parts = encoded.Split('|');
            if (parts.Length < 3 || parts[0] != "shape") return;

            long[] dims;
            try { dims = ParseDims(parts[2]); }
            catch (FormatException) { return; }

            node.Attributes = node.Attributes.SetAttributes(
                (OnnxOpAttributeNames.ShrkAttrRepresentativeInputShape, (object?)dims));
        }

        private static string DimsToString(long[] dims) => string.Join(",", dims);

        private static long[] ParseDims(string s)
            => s.Length == 0 ? [] : s.Split(',').Select(long.Parse).ToArray();
    }
}
