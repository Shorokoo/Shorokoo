using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using System.Linq;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Graph;

namespace Shorokoo.Core.Graph
{
    /// <summary>
    /// Convenience classification helpers mirroring the analogous predicates on
    /// <see cref="Shorokoo.Core.Nodes.Node"/>. Kept as extension methods (rather than properties on
    /// <see cref="FastNode"/>) so the FastNode type itself stays a dumb data record.
    /// </summary>
    internal static class FastNodeClassification
    {
        public static bool IsModelInput(this FastNode node) =>
            node.OpCode == InternalOpCodes.MODEL_TENSOR_INPUT ||
            node.OpCode == InternalOpCodes.MODEL_OPTIONAL_INPUT ||
            node.OpCode == InternalOpCodes.MODEL_SEQUENCE_INPUT ||
            node.OpCode == InternalOpCodes.MODEL_TENSORSTRUCT_INPUT ||
            node.OpCode == InternalOpCodes.GENERIC_TYPE_INPUT;

        public static bool IsModelParamData(this FastNode node) =>
            node.OpCode == InternalOpCodes.MODEL_PARAM_DATA;

        public static bool IsFunction(this FastNode node) =>
            node.OpCode == InternalOpCodes.FUNCTION_INVOKE;

        /// <summary>
        /// Open node (LOOP_OPEN / IF_OPEN). Resolved via <see cref="Definitions.NodeDefinitions"/>.
        /// </summary>
        public static bool IsOpenNode(this FastNode node)
            => Definitions.NodeDefinitions.TryGetValue(node.OpCode, out var def) && def.IsOpenNode;

        /// <summary>
        /// Close node (LOOP_CLOSE / IF_CLOSE). Resolved via <see cref="Definitions.NodeDefinitions"/>.
        /// </summary>
        public static bool IsCloseNode(this FastNode node)
            => Definitions.NodeDefinitions.TryGetValue(node.OpCode, out var def) && def.IsCloseNode;

        /// <summary>
        /// Returns true if any attribute on this node is a graph-typed attribute (e.g. If/Loop bodies).
        /// </summary>
        public static bool HasGraphAttribute(this FastNode node)
            => node.Attributes.AttributeDefs.Any(d => d.Type == AttributeType.Graph);

        /// <summary>
        /// Returns the producer node for each <see cref="FastTensorKey"/> in <paramref name="graph"/>'s
        /// nodes (keyed by output tensor key).
        /// </summary>
        public static System.Collections.Generic.Dictionary<FastTensorKey, FastNode> BuildProducerByOutputMap(this InternalComputationGraph graph)
        {
            var map = new System.Collections.Generic.Dictionary<FastTensorKey, FastNode>();
            foreach (var node in graph.Nodes)
            {
                foreach (var slot in node.FullOutputs.Values)
                {
                    foreach (var k in slot)
                    {
                        if (k is FastTensorKey key && !key.IsEmpty)
                            map[key] = node;
                    }
                }
            }
            return map;
        }
    }
}
