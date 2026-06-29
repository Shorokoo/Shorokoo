using Shorokoo.Graph;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using System.Collections.Generic;
using Shorokoo.Core.Nodes.Processors.AutoGrad;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// Fast-side counterparts to the Variable factory helpers in <see cref="InternalOp"/>.
    /// Each method produces a single <see cref="FastNode"/> with its required attributes
    /// pre-populated and a fresh <see cref="FastNodeKey"/>; callers append the returned node
    /// to their graph's node list and use the node's output <see cref="FastTensorKey"/>(s)
    /// to wire downstream consumers.
    /// </summary>
    internal static class FastInternalOp
    {
        /// <summary>Mirrors <see cref="InternalOp.RuntimeInput"/>.</summary>
        public static FastNode RuntimeInput(DType dtype, int? rank, string? defaultName = null)
        {
            var nodeKey = FastNodeKey.New();
            var attrDefs = Definitions.NodeDefinitions[InternalOpCodes.MODEL_TENSOR_INPUT].AttributeDefs;
            var attrs = OnnxCSharpAttributes.FromCSharpVals(
                new Dictionary<string, object?>
                {
                    [OnnxOpAttributeNames.AttrDtype] = dtype,
                    [OnnxOpAttributeNames.ShrkAttrRank] = (long?)rank,
                },
                attrDefs);

            return new FastNode
            {
                Key = nodeKey,
                OpCode = InternalOpCodes.MODEL_TENSOR_INPUT,
                Attributes = attrs,
                FriendlyName = defaultName,
                FullOutputs = { [""] = new List<FastTensorKey?> { new FastTensorKey(nodeKey, 0) } },
            };
        }

        /// <summary>Mirrors <see cref="InternalOp.TensorStructInput"/>.</summary>
        public static FastNode TensorStructInput(DType structDType, string? defaultName = null)
        {
            var nodeKey = FastNodeKey.New();
            var attrDefs = Definitions.NodeDefinitions[InternalOpCodes.MODEL_TENSORSTRUCT_INPUT].AttributeDefs;
            var attrs = OnnxCSharpAttributes.FromCSharpVals(
                new Dictionary<string, object?>
                {
                    [OnnxOpAttributeNames.AttrDtype] = structDType,
                    [OnnxOpAttributeNames.ShrkAttrInputType] = (InputType?)null,
                },
                attrDefs);

            return new FastNode
            {
                Key = nodeKey,
                OpCode = InternalOpCodes.MODEL_TENSORSTRUCT_INPUT,
                Attributes = attrs,
                FriendlyName = defaultName,
                FullOutputs = { [""] = new List<FastTensorKey?> { new FastTensorKey(nodeKey, 0) } },
            };
        }

        /// <summary>Mirrors <see cref="InternalOp.TensorStructGetField"/>.</summary>
        public static FastNode TensorStructGetField(
            FastTensorKey structInputKey, string fieldName, DType fieldDType, int? fieldRank, DataStructure fieldStructure)
        {
            var nodeKey = FastNodeKey.New();
            var attrDefs = Definitions.NodeDefinitions[InternalOpCodes.TENSOR_STRUCT_GETFIELD].AttributeDefs;
            var attrs = OnnxCSharpAttributes.FromCSharpVals(
                new Dictionary<string, object?>
                {
                    [OnnxOpAttributeNames.ShrkAttrFieldName] = fieldName,
                    [OnnxOpAttributeNames.ShrkAttrDtype] = fieldDType,
                    [OnnxOpAttributeNames.ShrkAttrRank] = (long?)fieldRank,
                    [OnnxOpAttributeNames.ShrkAttrStructure] = fieldStructure,
                },
                attrDefs);

            return new FastNode
            {
                Key = nodeKey,
                OpCode = InternalOpCodes.TENSOR_STRUCT_GETFIELD,
                Attributes = attrs,
                FullInputs = { [""] = new List<FastTensorKey?> { structInputKey } },
                FullOutputs = { [""] = new List<FastTensorKey?> { new FastTensorKey(nodeKey, 0) } },
            };
        }

        /// <summary>Mirrors <see cref="InternalOp.TensorStructCreate"/>.</summary>
        public static FastNode TensorStructCreate(DType structDType, IEnumerable<FastTensorKey> fieldKeys)
        {
            var nodeKey = FastNodeKey.New();
            var attrDefs = Definitions.NodeDefinitions[InternalOpCodes.TENSOR_STRUCT_CREATE].AttributeDefs;
            var attrs = OnnxCSharpAttributes.FromCSharpVals(
                new Dictionary<string, object?>
                {
                    [OnnxOpAttributeNames.AttrDtype] = structDType,
                },
                attrDefs);

            var inputs = new List<FastTensorKey?>();
            foreach (var k in fieldKeys) inputs.Add(k);

            return new FastNode
            {
                Key = nodeKey,
                OpCode = InternalOpCodes.TENSOR_STRUCT_CREATE,
                Attributes = attrs,
                FullInputs = { [""] = inputs },
                FullOutputs = { [""] = new List<FastTensorKey?> { new FastTensorKey(nodeKey, 0) } },
            };
        }

        /// <summary>
        /// Mirrors <see cref="InternalOp.AutoGrad"/>. Produces an AUTO_GRAD node whose inputs
        /// are <c>[loss, ...inputs]</c> and which yields one output per element of
        /// <paramref name="inputKeys"/> (parallel to the inputs, matching the Variable variant
        /// which returns <c>Variable?[]</c> of the same length).
        /// </summary>
        public static FastNode AutoGrad(FastTensorKey lossKey, IReadOnlyList<FastTensorKey> inputKeys)
        {
            var nodeKey = FastNodeKey.New();
            var attrDefs = Definitions.NodeDefinitions[InternalOpCodes.AUTO_GRAD].AttributeDefs;
            var attrs = OnnxCSharpAttributes.FromCSharpVals(new Dictionary<string, object?>(), attrDefs);

            var inputs = new List<FastTensorKey?>(inputKeys.Count + 1) { lossKey };
            foreach (var k in inputKeys) inputs.Add(k);

            var outputs = new List<FastTensorKey?>(inputKeys.Count);
            for (int i = 0; i < inputKeys.Count; i++)
                outputs.Add(new FastTensorKey(nodeKey, i));

            return new FastNode
            {
                Key = nodeKey,
                OpCode = InternalOpCodes.AUTO_GRAD,
                Attributes = attrs,
                FullInputs = { [""] = inputs },
                FullOutputs = { [""] = outputs },
            };
        }

        /// <summary>
        /// Mirrors <c>OnnxOp.Constant(TensorData)</c>: emits a CONSTANT node with the supplied
        /// <see cref="TensorData"/> attached as the <c>value</c> attribute.
        /// </summary>
        public static FastNode Constant(TensorData value)
        {
            var nodeKey = FastNodeKey.New();
            var attrDefs = Definitions.NodeDefinitions[OpCodes.CONSTANT].AttributeDefs;
            var attrs = OnnxCSharpAttributes.FromCSharpVals(
                new Dictionary<string, object?>
                {
                    [OnnxOpAttributeNames.AttrValue] = value,
                },
                attrDefs);

            return new FastNode
            {
                Key = nodeKey,
                OpCode = OpCodes.CONSTANT,
                Attributes = attrs,
                FullOutputs = { [""] = new List<FastTensorKey?> { new FastTensorKey(nodeKey, 0) } },
            };
        }
    }
}
