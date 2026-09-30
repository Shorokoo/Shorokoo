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
    /// Decides the opcode/domain/attributes used when emitting an ONNX <c>NodeProto</c> for a
    /// given <see cref="FastNode"/>. Pure: no <see cref="Variable"/> access,
    /// no tensor-info dictionary, no per-node side state — purely a function of the
    /// FastNode and its graph.
    ///
    /// <para>
    /// Four cases:
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
    /// Every node is emitted at <see cref="OnnxOpset.Version"/>, the one opset Shorokoo writes;
    /// a node that opset does not define — an operator or optional attribute ONNX introduced
    /// later — is refused (<see cref="OnnxOpset.ThrowIfOutsideVersion(FastNode)"/>) rather than
    /// emitted.
    /// </para>
    /// </summary>
    internal static class FastOpsetResolver
    {
        internal readonly struct OpsetInfo
        {
            public readonly string OpCode;
            public readonly string Domain;
            public readonly OnnxProtoAttributes Attributes;
            /// <summary>Flat list of input keys whose <see cref="FastTensorKey.ToString"/>
            /// becomes the NodeProto's input names. Null entries become empty strings.</summary>
            public readonly IReadOnlyList<FastTensorKey?> InputKeys;
            /// <summary>Flat list of output keys, same convention as <see cref="InputKeys"/>.</summary>
            public readonly IReadOnlyList<FastTensorKey?> OutputKeys;
            public readonly string? IdentifierTemplateString;
            public readonly string? StackTrace;

            public OpsetInfo(string opCode, string domain,
                OnnxProtoAttributes attributes,
                IReadOnlyList<FastTensorKey?> inputKeys,
                IReadOnlyList<FastTensorKey?> outputKeys,
                string? identifierTemplateString, string? stackTrace)
            {
                OpCode = opCode;
                Domain = domain;
                Attributes = attributes;
                InputKeys = inputKeys;
                OutputKeys = outputKeys;
                IdentifierTemplateString = identifierTemplateString;
                StackTrace = stackTrace;
            }
        }

        private static FastNode StripCheckpointStamp(FastNode node)
        {
            var stripped = Shorokoo.Core.AutoDiffCheckpointing.CheckpointSegment.Strip(node.Attributes);
            if (ReferenceEquals(stripped, node.Attributes)) return node;
            return new FastNode
            {
                Key = node.Key,
                OpCode = node.OpCode,
                Attributes = stripped,
                FullInputs = node.FullInputs,
                FullOutputs = node.FullOutputs,
                FriendlyName = node.FriendlyName,
                StackTrace = node.StackTrace,
                GraphOpenNodeKey = node.GraphOpenNodeKey,
                IdentifierTemplate = node.IdentifierTemplate,
                TargetFunction = node.TargetFunction,
            };
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
            bool stripCheckpointStamp = true)
        {
            OnnxOpset.ThrowIfOutsideVersion(node);
            var nodeDef = Definitions.NodeDefinitions[node.OpCode].Resolve(node.Attributes.ToProto());
            if (nodeDef.IsGraphNode && IsOpenOpCode(node.OpCode))
                return null;

            // The activation-checkpoint stamp inlining puts on a segment's nodes is consumed by
            // the memory-aware pass and is not part of any op's schema, so it never reaches a
            // NodeProto ORT will run or a vanilla export — ORT rejects an attribute its kernel
            // does not declare, and an export must carry nothing Shorokoo-private. Shorokoo's own
            // .srk dialect keeps it (stripCheckpointStamp false): a rig reloaded from a checkpoint
            // must still honour the [Module(Checkpoint = true)] its architecture was built with.
            // (A MODEL_INVOKE's own hint is in its definition and is not emitted here.)
            if (stripCheckpointStamp && node.OpCode != InternalOpCodes.MODEL_INVOKE)
                node = StripCheckpointStamp(node);

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
                && node.OpCode != InternalOpCodes.MODEL_PARAM_REF
                && node.OpCode != InternalOpCodes.MODEL_PARAM_MODEL_REF
                && node.OpCode != InternalOpCodes.MODEL_PARAM_ID_REF;

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
                attributes: attributes.ToProto(),
                inputKeys: node.Inputs,
                outputKeys: node.Outputs,
                identifierTemplateString: identifierTemplate,
                stackTrace: NormalizeStackTrace(node.StackTrace));
        }

        /// <summary>
        /// True for nodes that are part of the model boundary rather than ops
        /// emitted as <c>NodeProto</c>: graph inputs (variants), graph outputs, parameter data,
        /// and the open side of an open/close pair.
        /// </summary>
        public static bool IsBoundaryOrOpen(FastNode node)
            => IsOpenOpCode(node.OpCode)
            || InternalOpCodes.IsModelInputOp(node.OpCode)
            || InternalOpCodes.IsGraphOutputOp(node.OpCode)
            || node.OpCode == InternalOpCodes.MODEL_PARAM_DATA;

        public static bool IsOpenOpCode(string opCode)
            => opCode == OpCodes.IF_OPEN
            || opCode == OpCodes.LOOP_OPEN
            || opCode == OpCodes.SEQUENCE_MAP_OPEN;

        public static bool IsCloseOpCode(string opCode)
            => opCode == OpCodes.IF_CLOSE
            || opCode == OpCodes.LOOP_CLOSE
            || opCode == OpCodes.SEQUENCE_MAP_CLOSE;

        private static string? NormalizeStackTrace(string? trace)
            => string.IsNullOrEmpty(trace) ? null : trace;
    }
}
