using Shorokoo.Core.Factory.IR;
using Shorokoo.Core.Graph;
using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.OnnxNodes;
using System;
using System.Collections.Generic;
using System.Linq;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using Shorokoo.Onnx;

namespace Shorokoo.Core.Factory
{
    /// <summary>
    /// Fast-CG counterpart of the legacy <c>Node.ForOpsetMany</c>. Decides
    /// the opcode/domain/attributes used when emitting an ONNX <c>NodeProto</c> for a
    /// given <see cref="FastNode"/>. Pure: no <see cref="Variable"/> access,
    /// no tensor-info dictionary, no per-node side state — purely a function of the
    /// FastNode and its graph.
    ///
    /// <para>
    /// Mirrors the four cases handled by the CG-side method:
    /// <list type="number">
    ///   <item>Open node → returns <c>null</c> (open nodes are not emitted as ONNX nodes).</item>
    ///   <item>Close node → opcode is the def's full name; the close node's <em>own</em>
    ///     inputs become subgraph outputs, so the NodeProto's input list is the
    ///     <em>open</em> node's flat inputs (e.g. the IF condition or the LOOP control
    ///     tensors).</item>
    ///   <item>Function call (<c>FUNCTION_INVOKE</c> or a model-param initializer fn) →
    ///     opcode is the target function's <c>DefaultName</c>, domain is
    ///     <c>"Functions"</c>, and the <c>shrk_function_name</c>/<c>shrk_domain_name</c>
    ///     attributes get re-stamped to match.</item>
    ///   <item>Anything else → straight pass-through of opcode/attributes.</item>
    /// </list>
    /// </para>
    ///
    /// <para>
    /// Exports are opset 21 in practice: the builder requests
    /// <see cref="OpSetVersion.OPS_21"/> as the baseline and <see cref="RaiseToRequired"/>
    /// only bumps it for ops/attributes actually present in the graph. The post-21 ops in
    /// <see cref="MinimumOpsetByOpCode"/> never reach emission today — Swish and
    /// RMSNormalization lower inline to opset-21 primitives in their <c>OnnxOp</c> entry
    /// points, and Attention / RotaryEmbedding / TensorScatter / BitCast / CumProd throw
    /// <c>NotImplementedException</c> there — so those floors are presently unreachable and
    /// kept only as the documented restore point for when a runtime registers the ops at a
    /// usable opset. The attribute floors in <see cref="MinimumOpsetByAttribute"/> remain
    /// live (a user can still author e.g. a Cast <c>round_mode</c>).
    /// </para>
    /// </summary>
    internal static class FastOpsetResolver
    {
        /// <summary>
        /// Standard ops introduced after opset 21, keyed by op code, with the minimum
        /// ai.onnx opset that defines them. The exporter raises a model's opset_import
        /// just enough to cover every op present in the graph. The baseline stays at
        /// <see cref="OpSetVersion.OPS_21"/> because blanket-stamping a higher opset
        /// selects newer kernel versions in ONNX Runtime, and ORT's CPU provider has
        /// gaps there (e.g. GlobalLpPool/RandomNormalLike have no registered opset-22
        /// kernels even in ORT 1.26, where the opset-22 change was bfloat16-only).
        /// </summary>
        private static readonly Dictionary<string, OpSetVersion> MinimumOpsetByOpCode = new()
        {
            // Attention is defined since opset 23, but ORT 1.26's CPU provider only
            // registers the kernel for opset 24+ (and the def models the opset-24
            // input list with nonpad_kv_seqlen), so stamp 24.
            ["Attention"] = OpSetVersion.OPS_24,
            ["RMSNormalization"] = OpSetVersion.OPS_23,
            ["RotaryEmbedding"] = OpSetVersion.OPS_23,
            ["TensorScatter"] = OpSetVersion.OPS_24,
            ["Swish"] = OpSetVersion.OPS_24,
            ["BitCast"] = OpSetVersion.OPS_26,
            ["CumProd"] = OpSetVersion.OPS_26,
        };

        /// <summary>
        /// Post-21 OPTIONAL ATTRIBUTES on pre-existing ops: when such an attribute is
        /// actually present on a node, the model must be stamped at the opset that
        /// introduced it (e.g. ORT rejects <c>round_mode</c> on a Cast in an opset-21
        /// model as an unrecognized attribute). Keyed by (op code, attribute name).
        /// </summary>
        private static readonly Dictionary<(string Op, string Attr), OpSetVersion> MinimumOpsetByAttribute = new()
        {
            [("DequantizeLinear", OnnxOpAttributeNames.AttrOutputDtype)] = OpSetVersion.OPS_23,
            [("QuantizeLinear", OnnxOpAttributeNames.AttrPrecision)] = OpSetVersion.OPS_23,
            [("Cast", OnnxOpAttributeNames.AttrRoundMode)] = OpSetVersion.OPS_24,
            [("CastLike", OnnxOpAttributeNames.AttrRoundMode)] = OpSetVersion.OPS_24,
        };

        /// <summary>
        /// Returns <paramref name="requested"/> raised to the smallest opset that
        /// covers every node in <paramref name="nodes"/> (main graph and function
        /// bodies included by the caller): post-21 ops per
        /// <see cref="MinimumOpsetByOpCode"/>, and post-21 optional attributes per
        /// <see cref="MinimumOpsetByAttribute"/>.
        /// </summary>
        internal static OpSetVersion RaiseToRequired(IEnumerable<FastNode> nodes, OpSetVersion requested)
        {
            var result = requested;
            foreach (var node in nodes)
            {
                if (MinimumOpsetByOpCode.TryGetValue(node.OpCode, out var min) && min > result)
                    result = min;
                foreach (var ((op, attr), attrMin) in MinimumOpsetByAttribute)
                    if (attrMin > result && node.OpCode == op
                        && node.Attributes.IsAttributeDefined(attr)
                        && !node.Attributes.IsDefaultValue(attr))
                        result = attrMin;
            }
            return result;
        }

        internal readonly struct OpsetInfo
        {
            public readonly string OpCode;
            public readonly string Domain;
            public readonly OpSetVersion Version;
            public readonly OnnxProtoAttributes Attributes;
            /// <summary>Flat list of input keys whose <see cref="FastTensorKey.ToString"/>
            /// becomes the NodeProto's input names. Null entries become empty strings.</summary>
            public readonly IReadOnlyList<FastTensorKey?> InputKeys;
            /// <summary>Flat list of output keys, same convention as <see cref="InputKeys"/>.</summary>
            public readonly IReadOnlyList<FastTensorKey?> OutputKeys;
            public readonly string? IdentifierTemplateString;
            public readonly string? StackTrace;

            public OpsetInfo(string opCode, string domain, OpSetVersion version,
                OnnxProtoAttributes attributes,
                IReadOnlyList<FastTensorKey?> inputKeys,
                IReadOnlyList<FastTensorKey?> outputKeys,
                string? identifierTemplateString, string? stackTrace)
            {
                OpCode = opCode;
                Domain = domain;
                Version = version;
                Attributes = attributes;
                InputKeys = inputKeys;
                OutputKeys = outputKeys;
                IdentifierTemplateString = identifierTemplateString;
                StackTrace = stackTrace;
            }
        }

        /// <summary>
        /// Resolve the ONNX-emit info for <paramref name="node"/>. Returns <c>null</c>
        /// when the node should not be emitted (open nodes, model inputs, model param
        /// data — those are emitted via <c>ValueInfoProto</c>/<c>TensorProto</c>
        /// instead).
        /// </summary>
        public static OpsetInfo? Resolve(
            FastNode node,
            FastNode? graphOpenNode,
            OpSetVersion opset)
        {
            var nodeDef = Definitions.NodeDefinitions[node.OpCode].Resolve(node.Attributes.ToProto());
            if (nodeDef.IsGraphNode && IsOpenOpCode(node.OpCode))
                return null;

            // Close-node form: the NodeProto's inputs come from the matching OPEN node;
            // the close's own inputs are subgraph outputs, not NodeProto inputs.
            if (nodeDef.IsGraphNode && IsCloseOpCode(node.OpCode))
            {
                if (graphOpenNode is null)
                    throw new InvalidOperationException(
                        $"FastOpsetResolver: close node {node.OpCode} (Key={node.Key}) has no resolved open node.");
                return new OpsetInfo(
                    opCode: nodeDef.FullNodeOpName,
                    domain: "",
                    version: opset,
                    attributes: node.Attributes.ToProto(),
                    inputKeys: graphOpenNode.Inputs,
                    outputKeys: node.Outputs,
                    identifierTemplateString: null,
                    stackTrace: NormalizeStackTrace(node.StackTrace));
            }

            // Function-call form: rewrite opcode and attributes to point at the target
            // function. Also handles model-param-initializer fns (treated as functions
            // for emission purposes).
            var attributes = node.Attributes;
            var opCode = node.OpCode;
            var domain = "";

            bool isFunctionCall = node.OpCode == InternalOpCodes.FUNCTION_INVOKE;
            bool isParamInitializerFn = node.TargetFunction is not null
                && (node.TargetFunction.FunctionType == FunctionType.TrainableParamInitializer
                 || node.TargetFunction.FunctionType == FunctionType.StateParamInitializer)
                && node.OpCode != InternalOpCodes.TRAINABLE_PARAM_REF
                && node.OpCode != InternalOpCodes.TRAINABLE_PARAM_MODEL_REF
                && node.OpCode != InternalOpCodes.TRAINABLE_PARAM_ID_REF;

            // Write the function-name attribute for any node that carries a TargetFunction
            // whose schema declares the attribute. The previous form restricted this to
            // isFunctionCall || isParamInitializerFn, which left ops like SEQUENCE_CONSTRUCT
            // / SEQUENCE_EMPTY (whose ModuleFn is carried by TargetFunction per Node.cs:285)
            // without a serialized function reference. The reload path then couldn't restore
            // TargetFunction → ModuleFn on the SequenceConstruct's output → the consuming
            // MODEL_INVOKE tripped Inputs[0].ModuleFn-is-null on reload.
            // The emitted ONNX op_type / function-name must dodge built-in ONNX op names
            // (e.g. a Constant initializer collides with the ONNX Constant op, which ORT
            // would dispatch to regardless of domain). OnnxFunctionName.Encode prefixes
            // only colliding names; OnnxModelReader.Decode reverses it on load.
            if (node.TargetFunction is not null
                && attributes.IsAttributeDefined(OnnxOpAttributeNames.ShrkAttrFunctionName))
            {
                attributes = attributes.SetAttributes(
                    (OnnxOpAttributeNames.ShrkAttrFunctionName, OnnxFunctionName.Encode(node.TargetFunction.DefaultName)),
                    (OnnxOpAttributeNames.ShrkAttrDomainName, "Functions"));
            }

            if (isFunctionCall || isParamInitializerFn)
            {
                opCode = OnnxFunctionName.Encode(node.TargetFunction!.DefaultName);
                domain = "Functions";
            }

            // The CG-side resolver only sets identifierTemplateString when the node has
            // exactly one output. We follow the same shape here.
            string? identifierTemplate = null;
            int outputCount = node.Outputs.Count(k => k is not null && !k.Value.IsEmpty);
            if (outputCount == 1) identifierTemplate = node.IdentifierTemplate;

            return new OpsetInfo(
                opCode: opCode,
                domain: domain,
                version: opset,
                attributes: attributes.ToProto(),
                inputKeys: node.Inputs,
                outputKeys: node.Outputs,
                identifierTemplateString: identifierTemplate,
                stackTrace: NormalizeStackTrace(node.StackTrace));
        }

        /// <summary>
        /// True for nodes that are part of the model boundary rather than ops
        /// emitted as <c>NodeProto</c>: graph inputs (variants), parameter data,
        /// and the open side of an open/close pair.
        /// </summary>
        public static bool IsBoundaryOrOpen(FastNode node)
            => IsOpenOpCode(node.OpCode)
            || IsModelInputOpCode(node.OpCode)
            || node.OpCode == InternalOpCodes.MODEL_PARAM_DATA;

        public static bool IsOpenOpCode(string opCode)
            => opCode == OpCodes.IF_OPEN
            || opCode == OpCodes.LOOP_OPEN
            || opCode == OpCodes.SCAN_OPEN
            || opCode == OpCodes.SEQUENCE_MAP_OPEN;

        public static bool IsCloseOpCode(string opCode)
            => opCode == OpCodes.IF_CLOSE
            || opCode == OpCodes.LOOP_CLOSE
            || opCode == OpCodes.SCAN_CLOSE
            || opCode == OpCodes.SEQUENCE_MAP_CLOSE;

        public static bool IsModelInputOpCode(string opCode)
            => opCode == InternalOpCodes.MODEL_TENSOR_INPUT
            || opCode == InternalOpCodes.MODEL_OPTIONAL_INPUT
            || opCode == InternalOpCodes.MODEL_SEQUENCE_INPUT
            || opCode == InternalOpCodes.MODEL_TENSORSTRUCT_INPUT
            || opCode == InternalOpCodes.GENERIC_TYPE_INPUT;

        private static string? NormalizeStackTrace(string? trace)
            => string.IsNullOrEmpty(trace) ? null : trace;
    }
}
