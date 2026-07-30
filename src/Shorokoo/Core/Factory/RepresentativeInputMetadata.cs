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
    /// <para>Encoding is passthrough: whichever of the two mutually-exclusive attributes is set on the
    /// node is serialized verbatim. The vanilla &gt;N-element downgrade is a separate graph pre-pass
    /// (<see cref="Shorokoo.Core.Nodes.Processors.Fast.FastDowngradeRepresentativeInputs"/>) applied
    /// before emission, so this codec never re-thresholds.</para>
    /// <list type="bullet">
    ///   <item><c>"tensor|{protoDtype}|{d0,d1,…}|{base64 raw bytes}"</c> — an inline (zero-filled) tensor.</item>
    ///   <item><c>"shape|{protoDtype}|{d0,d1,…}"</c> — dims only.</item>
    /// </list>
    /// </summary>
    internal static class RepresentativeInputMetadata
    {
        /// <summary>Per-input <see cref="IR.ValueInfoProto"/> metadata key for the representative-input info.</summary>
        public const string Key = "shrk_repr_input";

        /// <summary>
        /// Encodes the representative-input attribute currently set on <paramref name="node"/> (a
        /// <c>MODEL_TENSOR_INPUT</c>), or <c>null</c> when neither attribute is set. Passthrough — no
        /// downgrade.
        /// </summary>
        public static string? Encode(FastNode node)
        {
            var attrs = node.Attributes;

            if (attrs.IsAttributeDefined(OnnxOpAttributeNames.ShrkAttrRepresentativeInput)
                && attrs.GetTensorVal(OnnxOpAttributeNames.ShrkAttrRepresentativeInput) is { } inline)
            {
                var b64 = Convert.ToBase64String(inline.AccessRawMemory().ToArray());
                return $"tensor|{inline.DType.ProtoTypeNum}|{DimsToString(inline.Shape.Dims)}|{b64}";
            }

            if (attrs.IsAttributeDefined(OnnxOpAttributeNames.ShrkAttrRepresentativeInputShape)
                && attrs.GetLongsVal(OnnxOpAttributeNames.ShrkAttrRepresentativeInputShape) is { } dims
                && attrs.GetDTypeVal(OnnxOpAttributeNames.AttrDtype) is { } dtype)
            {
                return $"shape|{dtype.ProtoTypeNum}|{DimsToString(dims)}";
            }

            return null;
        }

        /// <summary>
        /// Decodes <paramref name="encoded"/> and sets the matching representative attribute on
        /// <paramref name="node"/> (clearing the other). Fail-safe: a malformed value — wrong field count,
        /// unparseable dtype/dims, non-base64 payload, or a byte count that does not match the shape/dtype
        /// — is skipped, leaving the node untouched. The dtype is parsed only in the tensor branch, where
        /// it is needed.
        /// </summary>
        public static void Apply(FastNode node, string encoded)
        {
            var parts = encoded.Split('|');
            if (parts.Length < 3) return;

            long[] dims;
            try { dims = ParseDims(parts[2]); }
            catch (FormatException) { return; }

            if (parts[0] == "tensor" && parts.Length >= 4)
            {
                if (!int.TryParse(parts[1], out var protoDtype)) return;
                byte[] bytes;
                try { bytes = Convert.FromBase64String(parts[3]); }
                catch (FormatException) { return; }

                var dtype = (DType)protoDtype;
                var shape = new Shape(dims);
                // Guard against a byte-count/shape mismatch: CreateFromRawBytes throws (not a
                // FormatException) on a bad length, and this codec's contract is to skip malformed values.
                var bytesPerElement = dtype.EncodingBitCount / 8;
                if (shape.Count < 0 || bytesPerElement <= 0
                    || bytes.LongLength != shape.Count * bytesPerElement)
                    return;

                var tensor = TensorData.CreateFromRawBytes(shape, dtype, bytes);
                node.Attributes = node.Attributes.SetAttributes(
                    (OnnxOpAttributeNames.ShrkAttrRepresentativeInput, (object?)tensor),
                    (OnnxOpAttributeNames.ShrkAttrRepresentativeInputShape, (object?)null));
            }
            else if (parts[0] == "shape")
            {
                node.Attributes = node.Attributes.SetAttributes(
                    (OnnxOpAttributeNames.ShrkAttrRepresentativeInputShape, (object?)dims),
                    (OnnxOpAttributeNames.ShrkAttrRepresentativeInput, (object?)null));
            }
        }

        private static string DimsToString(long[] dims) => string.Join(",", dims);

        private static long[] ParseDims(string s)
            => s.Length == 0 ? [] : s.Split(',').Select(long.Parse).ToArray();
    }
}
