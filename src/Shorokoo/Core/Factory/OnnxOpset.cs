using System.Collections.Generic;
using System.Linq;
using Shorokoo.Core.Factory.IR;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes.NodeDefinitions;

namespace Shorokoo.Core.Factory
{
    /// <summary>
    /// The one ONNX opset Shorokoo reads and writes. Every model it imports — a foreign
    /// <c>.onnx</c> and the payload of a <c>.srk</c> alike — must declare default-domain
    /// (<c>ai.onnx</c>) opset <see cref="Version"/>, and every model and function it writes is
    /// stamped with it.
    ///
    /// <para>Shorokoo also defines a few operators and optional attributes ONNX introduced after
    /// that opset (<see cref="OperatorsIntroducedLater"/>, <see cref="AttributesIntroducedLater"/>).
    /// None of them is part of opset 21, so a node carrying one is malformed in an opset-21 model:
    /// import refuses it. The node definitions declare none of the attributes, and
    /// <c>NodeBuilder</c> refuses a node carrying one. The operators have definitions, and the
    /// model builder refuses one that reaches emission. The authoring surface never gets it that
    /// far — <c>Swish</c> and <c>RMSNormalization</c> lower to opset-21 primitives at their
    /// <c>OnnxOp</c> entry points, <c>TensorScatter</c> is decomposed by the builder's lowering
    /// pre-pass, and <c>Attention</c>, <c>RotaryEmbedding</c>, <c>BitCast</c> and <c>CumProd</c>
    /// throw at theirs — so what the emission check catches is a node stamped through the raw
    /// <c>NodeBuilder</c> surface.</para>
    /// </summary>
    internal static class OnnxOpset
    {
        /// <summary>The default-domain opset Shorokoo reads and writes.</summary>
        internal const OpSetVersion Version = OpSetVersion.OPS_21;

        /// <summary>
        /// Standard operators ONNX introduced after <see cref="Version"/>, keyed by op code, with
        /// the opset that introduced each.
        /// </summary>
        internal static readonly IReadOnlyDictionary<string, OpSetVersion> OperatorsIntroducedLater =
            new Dictionary<string, OpSetVersion>
            {
                [OpCodes.ATTENTION] = OpSetVersion.OPS_23,
                [OpCodes.RMS_NORMALIZATION] = OpSetVersion.OPS_23,
                [OpCodes.ROTARY_EMBEDDING] = OpSetVersion.OPS_23,
                [OpCodes.TENSOR_SCATTER] = OpSetVersion.OPS_24,
                [OpCodes.SWISH] = OpSetVersion.OPS_24,
                [OpCodes.BIT_CAST] = OpSetVersion.OPS_26,
                [OpCodes.CUM_PROD] = OpSetVersion.OPS_26,
            };

        /// <summary>
        /// Optional attributes ONNX added after <see cref="Version"/> to operators opset 21 already
        /// defines, keyed by (op code, attribute name), with the opset that added each.
        /// </summary>
        internal static readonly IReadOnlyDictionary<(string Op, string Attr), OpSetVersion> AttributesIntroducedLater =
            new Dictionary<(string Op, string Attr), OpSetVersion>
            {
                [(OpCodes.DEQUANTIZE_LINEAR, OnnxOpAttributeNames.AttrOutputDtype)] = OpSetVersion.OPS_23,
                [(OpCodes.QUANTIZE_LINEAR, OnnxOpAttributeNames.AttrPrecision)] = OpSetVersion.OPS_23,
                [(OpCodes.CAST, OnnxOpAttributeNames.AttrRoundMode)] = OpSetVersion.OPS_24,
                [(OpCodes.CAST_LIKE, OnnxOpAttributeNames.AttrRoundMode)] = OpSetVersion.OPS_24,
            };

        private const string ImportContext = "ONNX import";
        private const string BuildContext = "ONNX model build";
        private const string NodeBuildContext = "node build";

        internal static bool IsDefaultDomain(string? domain)
            => string.IsNullOrEmpty(domain) || domain == "ai.onnx";

        /// <summary>
        /// Refuses <paramref name="model"/> with <see cref="ErrorCodes.FW060"/> unless it — and each
        /// of its functions that declares one — imports the default domain at <see cref="Version"/>.
        /// The entry is found by domain, wherever it sits in the list.
        /// </summary>
        internal static void ThrowIfNotAtVersion(ModelProto model)
        {
            var declared = model.OpsetImports.Where(o => o is not null && IsDefaultDomain(o.Domain)).ToList();
            if (declared.Count == 0)
                throw new ModelException(ErrorCodes.FW060, ImportContext,
                    "the model declares no default-domain (ai.onnx) opset import. Shorokoo reads ONNX " +
                    $"opset {(int)Version} models only, and such a model declares opset {(int)Version}.");
            foreach (var entry in declared)
                ThrowIfOtherVersion(entry.Version, "the model");

            foreach (var function in model.Functions)
                foreach (var entry in function.OpsetImports.Where(o => o is not null && IsDefaultDomain(o.Domain)))
                    ThrowIfOtherVersion(entry.Version, $"function '{function.Name}'");
        }

        private static void ThrowIfOtherVersion(long found, string where)
        {
            if (found != (int)Version)
                throw new ModelException(ErrorCodes.FW060, ImportContext,
                    $"{where} declares default-domain (ai.onnx) opset {found}, but Shorokoo reads ONNX " +
                    $"opset {(int)Version} models only. Convert the model to opset {(int)Version} (for " +
                    "example with the ONNX version converter) and import that.");
        }

        /// <summary>
        /// Refuses an imported default-domain <paramref name="node"/> with
        /// <see cref="ErrorCodes.FW060"/> when it is an operator, or carries an attribute, that opset
        /// <see cref="Version"/> does not define. A node in another domain is not an opset-21
        /// operator, and is left to the reader.
        /// </summary>
        internal static void ThrowIfOutsideVersion(NodeProto node)
        {
            if (!IsDefaultDomain(node.Domain)) return;
            var reason = DescribeOperator(node.OpType, node.Name);
            foreach (var attribute in node.Attributes)
                reason ??= DescribeAttribute(node.OpType, node.Name, attribute.Name);
            if (reason is not null)
                throw new ModelException(ErrorCodes.FW060, ImportContext,
                    $"{reason} The model is malformed: Shorokoo reads ONNX opset {(int)Version} models only.");
        }

        /// <summary>
        /// Refuses <paramref name="node"/> with <see cref="ErrorCodes.FW060"/> when it reaches ONNX
        /// emission as an operator opset <see cref="Version"/> does not define: Shorokoo writes that
        /// opset only.
        /// </summary>
        internal static void ThrowIfOutsideVersion(FastNode node)
        {
            if (DescribeOperator(node.OpCode, node.FriendlyName) is { } reason)
                throw new ModelException(ErrorCodes.FW060, BuildContext,
                    $"{reason} Shorokoo writes ONNX opset {(int)Version} only, so the node has no form it " +
                    "can emit.");
        }

        /// <summary>
        /// Refuses building a <paramref name="opCode"/> node carrying <paramref name="attributeName"/>
        /// with <see cref="ErrorCodes.FW060"/> when that is an attribute ONNX added after opset
        /// <see cref="Version"/>, which the node definitions therefore do not declare.
        /// </summary>
        internal static void ThrowIfAttributeOutsideVersion(string opCode, string attributeName)
        {
            if (DescribeAttribute(opCode, null, attributeName) is { } reason)
                throw new ModelException(ErrorCodes.FW060, NodeBuildContext,
                    $"{reason} Shorokoo builds ONNX opset {(int)Version} nodes only.");
        }

        private static string Label(string opCode, string? nodeName)
            => string.IsNullOrEmpty(nodeName) ? $"'{opCode}'" : $"node '{nodeName}' ('{opCode}')";

        private static string? DescribeOperator(string opCode, string? nodeName)
            => OperatorsIntroducedLater.TryGetValue(opCode, out var since)
                ? $"{Label(opCode, nodeName)} is an operator ONNX introduced at opset {(int)since}, " +
                    $"which opset {(int)Version} does not define."
                : null;

        private static string? DescribeAttribute(string opCode, string? nodeName, string attribute)
            => AttributesIntroducedLater.TryGetValue((opCode, attribute), out var since)
                ? $"{Label(opCode, nodeName)} carries the '{attribute}' attribute, which ONNX added at " +
                    $"opset {(int)since} and opset {(int)Version} does not define."
                : null;
    }
}
