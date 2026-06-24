using Shorokoo.Core;
using Shorokoo.Core.Graph;
using Shorokoo.Graph;
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
    /// Reusable FastNode constructors for common op codes. Each factory builds a single-output
    /// <see cref="FastNode"/> with the op's required attributes pre-populated. Callers are
    /// responsible for supplying a fresh <see cref="FastNodeKey"/> and for appending the returned
    /// node to their graph's node list.
    /// </summary>
    internal static class FastNodeConstructionUtils
    {
        public static FastNode CreateSequenceEmpty(FastNodeKey nodeKey, DType elementDType)
        {
            var attrDefs = Definitions.NodeDefinitions[OpCodes.SEQUENCE_EMPTY].AttributeDefs;
            var attrs = OnnxCSharpAttributes.FromCSharpVals(
                new Dictionary<string, object?> { [OnnxOpAttributeNames.AttrDtype] = elementDType },
                attrDefs);

            return new FastNode
            {
                Key = nodeKey,
                OpCode = OpCodes.SEQUENCE_EMPTY,
                Attributes = attrs,
                FullOutputs = { [""] = new List<FastTensorKey?> { new FastTensorKey(nodeKey, 0) } },
            };
        }

        public static FastNode CreateSequenceConstruct(FastNodeKey nodeKey, IEnumerable<FastTensorKey?> elements)
        {
            var attrDefs = Definitions.NodeDefinitions[OpCodes.SEQUENCE_CONSTRUCT].AttributeDefs;
            var attrs = OnnxCSharpAttributes.FromCSharpVals(new Dictionary<string, object?>(), attrDefs);

            return new FastNode
            {
                Key = nodeKey,
                OpCode = OpCodes.SEQUENCE_CONSTRUCT,
                Attributes = attrs,
                FullInputs = { [""] = new List<FastTensorKey?>(elements) },
                FullOutputs = { [""] = new List<FastTensorKey?> { new FastTensorKey(nodeKey, 0) } },
            };
        }

        public static FastNode CreateSequenceAt(FastNodeKey nodeKey, FastTensorKey sequence, FastTensorKey position)
        {
            var attrDefs = Definitions.NodeDefinitions[OpCodes.SEQUENCE_AT].AttributeDefs;
            var attrs = OnnxCSharpAttributes.FromCSharpVals(new Dictionary<string, object?>(), attrDefs);

            return new FastNode
            {
                Key = nodeKey,
                OpCode = OpCodes.SEQUENCE_AT,
                Attributes = attrs,
                FullInputs = { [""] = new List<FastTensorKey?> { sequence, position } },
                FullOutputs = { [""] = new List<FastTensorKey?> { new FastTensorKey(nodeKey, 0) } },
            };
        }

        public static FastNode CreateSequenceErase(FastNodeKey nodeKey, FastTensorKey sequence, FastTensorKey position)
        {
            var attrDefs = Definitions.NodeDefinitions[OpCodes.SEQUENCE_ERASE].AttributeDefs;
            var attrs = OnnxCSharpAttributes.FromCSharpVals(new Dictionary<string, object?>(), attrDefs);

            return new FastNode
            {
                Key = nodeKey,
                OpCode = OpCodes.SEQUENCE_ERASE,
                Attributes = attrs,
                FullInputs = { [""] = new List<FastTensorKey?> { sequence, position } },
                FullOutputs = { [""] = new List<FastTensorKey?> { new FastTensorKey(nodeKey, 0) } },
            };
        }

        /// <summary><paramref name="position"/> may be null to append at the end of the sequence.</summary>
        public static FastNode CreateSequenceInsert(
            FastNodeKey nodeKey, FastTensorKey sequence, FastTensorKey element, FastTensorKey? position)
        {
            var attrDefs = Definitions.NodeDefinitions[OpCodes.SEQUENCE_INSERT].AttributeDefs;
            var attrs = OnnxCSharpAttributes.FromCSharpVals(new Dictionary<string, object?>(), attrDefs);

            return new FastNode
            {
                Key = nodeKey,
                OpCode = OpCodes.SEQUENCE_INSERT,
                Attributes = attrs,
                FullInputs = { [""] = new List<FastTensorKey?> { sequence, element, position } },
                FullOutputs = { [""] = new List<FastTensorKey?> { new FastTensorKey(nodeKey, 0) } },
            };
        }

        public static FastNode CreateSequenceLength(FastNodeKey nodeKey, FastTensorKey sequence)
        {
            var attrDefs = Definitions.NodeDefinitions[OpCodes.SEQUENCE_LENGTH].AttributeDefs;
            var attrs = OnnxCSharpAttributes.FromCSharpVals(new Dictionary<string, object?>(), attrDefs);

            return new FastNode
            {
                Key = nodeKey,
                OpCode = OpCodes.SEQUENCE_LENGTH,
                Attributes = attrs,
                FullInputs = { [""] = new List<FastTensorKey?> { sequence } },
                FullOutputs = { [""] = new List<FastTensorKey?> { new FastTensorKey(nodeKey, 0) } },
            };
        }
    }
}
