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
    /// Builds the leaf-level ONNX protos (<see cref="ValueInfoProto"/>,
    /// <see cref="TensorProto"/>, <see cref="NodeProto"/>) directly from
    /// <see cref="FastNode"/>/<see cref="FastNodeKey"/>/<see cref="FastTensorKey"/>
    /// without ever consulting an <see cref="Variable"/> or a tensor-info dictionary.
    /// All ONNX names are derived from the Fast keys via
    /// <see cref="FastTensorKey.ToString"/> / <see cref="FastNodeKey.ToString"/>, which
    /// produce <c>"N{i}"</c> / <c>"N{i}_T{j}"</c> when the source CG was run through
    /// <see cref="Shorokoo.Core.Nodes.Processors.Fast.FastUseUniqueNames"/> first.
    ///
    /// <para>
    /// As discussed when designing the Fast builder, ONNX itself only carries metadata
    /// (dtype/shape/structure) at the model boundary — graph inputs/outputs and
    /// initializer tensors. We read that metadata directly off the producing
    /// <see cref="FastNode"/>'s attributes (e.g. <c>AttrDtype</c>, <c>ShrkAttrRank</c>
    /// on input nodes; <c>ShrkAttrTensorData</c> on parameter-data nodes) and never
    /// stash it in a side dictionary that would have to be kept in sync with the
    /// graph.
    /// </para>
    /// </summary>
    internal static class FastOnnxProtoFactory
    {
        // ------------------------- name helpers -------------------------

        /// <summary>The ONNX tensor name for a <see cref="FastTensorKey"/>.</summary>
        public static string TensorName(FastTensorKey key) => key.ToString();

        /// <summary>The ONNX tensor name for a nullable key; null/empty → empty string.</summary>
        public static string TensorName(FastTensorKey? key)
            => (key is null || key.Value.IsEmpty) ? string.Empty : key.Value.ToString();

        /// <summary>The ONNX node name for a <see cref="FastNode"/> — its key's string form.</summary>
        public static string NodeName(FastNode node) => node.Key.ToString();

        // ------------------------- boundary protos -------------------------

        /// <summary>
        /// Builds the <see cref="ValueInfoProto"/> for a graph input. Reads
        /// dtype/rank/structure/input-type from the producing input node's
        /// attributes; the structure (Tensor/Optional/Sequence/TensorStruct) is
        /// implied by the input op code.
        /// </summary>
        public static ValueInfoProto CreateGraphInputInfo(FastNode inputNode, FastTensorKey key)
        {
            (DType dtype, int? rank, DataStructure structure) = ReadInputMetadata(inputNode);
            string? inputTypeName = ReadInputTypeName(inputNode);
            // The default-value attribute is optional; nodes rebuilt without it (any non-defaulted
            // input) simply carry no default, so read it tolerantly rather than indexing.
            float? defaultValue = inputNode.Attributes.GetAttributeVals().TryGetValue(OnnxOpAttributeNames.ShrkAttrDefaultValue, out var dvObj)
                ? (float?)dvObj
                : null;

            var dims = rank is int r ? OnnxIRFactory.CreateDims(MakeUnnamedDims(r), key.ToString()) : null;
            return OnnxIRFactory.CreateTensorInfo(
                dims: dims,
                name: key.ToString(),
                type: dtype,
                structure: structure,
                targetFunctionName: null,
                inputTypeName: inputTypeName,
                defaultValue: defaultValue);
        }

        /// <summary>
        /// Builds the <see cref="ValueInfoProto"/> for a graph output. We can't
        /// infer dtype/shape from a producing input node here (the output is
        /// produced by an arbitrary op), so we emit a name-only ValueInfoProto and
        /// leave ONNX type-inference to fill in the rest at load time. Optional:
        /// callers that have stronger info (e.g. via a Fast tensor-info pass) can
        /// override.
        /// </summary>
        public static ValueInfoProto CreateGraphOutputInfo(FastTensorKey key)
        {
            var info = new ValueInfoProto();
            info.Name = key.ToString();
            info.Type = new TypeProto();
            return info;
        }

        /// <summary>
        /// Builds the <see cref="TensorProto"/> for an initializer
        /// (<c>MODEL_PARAM_DATA</c> node). All metadata (dtype, dims, raw data,
        /// trainability, identifier template) lives on the node's attributes.
        /// </summary>
        public static TensorProto CreateInitializer(FastNode paramDataNode, FastTensorKey outputKey)
        {
            if (paramDataNode.OpCode != InternalOpCodes.MODEL_PARAM_DATA)
                throw new InvalidOperationException(
                    $"FastOnnxProtoFactory.CreateInitializer: node {paramDataNode.OpCode} is not MODEL_PARAM_DATA.");

            var attrs = paramDataNode.Attributes;
            var data = attrs.GetTensorVal(OnnxOpAttributeNames.ShrkAttrTensorData)
                ?? throw new InvalidOperationException(
                    $"FastOnnxProtoFactory.CreateInitializer: MODEL_PARAM_DATA node has no {OnnxOpAttributeNames.ShrkAttrTensorData} attribute.");
            bool isTrainable = attrs.GetBoolVal(OnnxOpAttributeNames.ShrkAttrIsTrainable)
                ?? throw new InvalidOperationException(
                    $"FastOnnxProtoFactory.CreateInitializer: MODEL_PARAM_DATA node has no {OnnxOpAttributeNames.ShrkAttrIsTrainable} attribute.");

            return OnnxIRFactory.CreateTensor(
                dims: data.Shape.Dims,
                name: outputKey.ToString(),
                type: data.DType,
                identifierTemplate: paramDataNode.IdentifierTemplate,
                isTrainable: isTrainable,
                data: data.AccessRawMemory().ToArray());
        }

        // ------------------------- node proto -------------------------

        /// <summary>
        /// Builds the ONNX <see cref="NodeProto"/> for an emitted FastNode using
        /// the resolver-provided <see cref="FastOpsetResolver.OpsetInfo"/> and any
        /// graph-attribute subgraphs the caller has already built (e.g. then/else
        /// branches for an IF_CLOSE, body for a LOOP_CLOSE).
        /// </summary>
        public static NodeProto CreateNodeProto(
            FastNode node,
            FastOpsetResolver.OpsetInfo info,
            Dictionary<string, GraphProto>? graphAttributes)
        {
            var inputNames = info.InputKeys.Select(TensorName).ToArray();
            // Trailing omitted optionals are emitted by ONNX convention as absent inputs, not
            // empty-name placeholders — some ORT kernels (e.g. MaxUnpool) reject the latter
            // with "input count mismatch". Mid-list empties (e.g. Resize's roi) must stay.
            var inputCount = inputNames.Length;
            while (inputCount > 0 && inputNames[inputCount - 1].Length == 0) inputCount--;
            if (inputCount < inputNames.Length) inputNames = inputNames[..inputCount];
            var outputNames = info.OutputKeys.Select(TensorName).ToArray();
            return OnnxIRFactory.CreateNode(
                name: NodeName(node),
                opCode: info.OpCode,
                domain: info.Domain,
                version: info.Version,
                inputTensors: inputNames,
                outputTensors: outputNames,
                attributes: info.Attributes,
                graphAttributes: graphAttributes ?? new Dictionary<string, GraphProto>(),
                identifierTemplate: info.IdentifierTemplateString,
                stackTrace: info.StackTrace);
        }

        // ------------------------- subgraph value-info -------------------------

        /// <summary>
        /// <see cref="ValueInfoProto"/> for a subgraph output (e.g. a then-branch
        /// output that the IF_CLOSE consumes). The output's dtype/shape can't be
        /// recovered here without a tensor-info pass, so this emits a name-only
        /// info and lets ONNX type inference fill in the rest. ONNX permits this
        /// for subgraph outputs.
        /// </summary>
        public static ValueInfoProto CreateSubgraphOutputInfo(FastTensorKey key)
        {
            var info = new ValueInfoProto();
            info.Name = key.ToString();
            info.Type = new TypeProto();
            return info;
        }

        // ------------------------- private helpers -------------------------

        private static (DType dtype, int? rank, DataStructure structure) ReadInputMetadata(FastNode node)
        {
            var attrs = node.Attributes;
            var dtype = attrs.GetDTypeVal(OnnxOpAttributeNames.AttrDtype);
            if (dtype is null)
                throw new InvalidOperationException(
                    $"FastOnnxProtoFactory.ReadInputMetadata: input node {node.OpCode} (Key={node.Key}) has no {OnnxOpAttributeNames.AttrDtype} attribute.");

            int? rank = null;
            // Only the tensor-shaped variants carry a rank attribute.
            if (node.OpCode == InternalOpCodes.MODEL_TENSOR_INPUT
             || node.OpCode == InternalOpCodes.GENERIC_TYPE_INPUT)
            {
                var rl = attrs.GetLongVal(OnnxOpAttributeNames.ShrkAttrRank);
                rank = rl is long l ? (int?)l : null;
            }

            DataStructure structure = node.OpCode switch
            {
                InternalOpCodes.MODEL_TENSOR_INPUT => DataStructure.Tensor,
                InternalOpCodes.GENERIC_TYPE_INPUT => DataStructure.Tensor,
                InternalOpCodes.MODEL_OPTIONAL_INPUT => DataStructure.Optional,
                InternalOpCodes.MODEL_SEQUENCE_INPUT => DataStructure.Sequence,
                InternalOpCodes.MODEL_TENSORSTRUCT_INPUT => DataStructure.TensorStruct,
                _ => throw new InvalidOperationException(
                    $"FastOnnxProtoFactory.ReadInputMetadata: unrecognised input op {node.OpCode}.")
            };

            return (dtype, rank, structure);
        }

        private static string? ReadInputTypeName(FastNode node)
        {
            var attrs = node.Attributes;
            var inputType = attrs.GetEnumVal<InputType>(OnnxOpAttributeNames.ShrkAttrInputType);
            return inputType is null ? null : Function.ToInputTypeName(inputType.Value);
        }

        /// <summary>
        /// Produces an array of <see cref="TensorDim"/> entries with no concrete
        /// values — used for graph input shapes when only rank is known. The
        /// resulting <see cref="TensorShapeProto"/> has the right rank but every
        /// dim is symbolic/unset.
        /// </summary>
        private static TensorDim[] MakeUnnamedDims(int rank)
        {
            var dims = new TensorDim[rank];
            for (int i = 0; i < rank; i++) dims[i] = new TensorDim();
            return dims;
        }
    }
}
