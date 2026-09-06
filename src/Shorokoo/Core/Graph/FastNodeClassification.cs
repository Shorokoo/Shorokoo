using Shorokoo.Core.Nodes.OnnxNodes;
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
        /// True when <paramref name="node"/> is module-stage machinery: an op from the canonical
        /// <see cref="InternalOpCodes.ModuleStageOps"/> inventory, or a
        /// <see cref="InternalOpCodes.FUNCTION_INVOKE"/> whose target is module-typed (or
        /// unresolved). A FUNCTION_INVOKE calling a plain <see cref="FunctionType.Function"/> is
        /// executable content, not machinery — a reimported concrete model legitimately carries
        /// such calls (e.g. the RNG draw functions emitted at export), so the coarser
        /// <see cref="InternalOpCodes.IsModuleStageOp"/> would misclassify it.
        /// </summary>
        public static bool IsModuleStageMachinery(this FastNode node)
            => node.OpCode == InternalOpCodes.FUNCTION_INVOKE
                ? node.TargetFunction is not { FunctionType: FunctionType.Function }
                : InternalOpCodes.IsModuleStageOp(node.OpCode);

        /// <summary>
        /// True when <paramref name="node"/> is an op the execution pipeline cannot lower away, so a
        /// graph carrying it can only fail at session creation — ONNX Runtime rejects the model
        /// naming an op that appears nowhere in the public API
        /// (<c>No Op registered for ShrkCreateModule</c>).
        ///
        /// <para>Every <see cref="InternalOpCodes.FUNCTION_INVOKE"/> is runnable, whatever its
        /// target: an invoke is inlined or emitted as a call on a <c>FunctionProto</c>, which is how
        /// <c>ToConcreteModel</c> materializes parameters (an initializer-typed call), how a
        /// reimported concrete model carries its RNG draw functions (a plain
        /// <see cref="FunctionType.Function"/>), and how a module-typed call with a machinery-free
        /// body executes. Whether an invoke <em>can</em> run is a property of the callee's body, not
        /// of its <see cref="FunctionType"/>, and this scan deliberately does not descend into
        /// bodies: a live initializer body legitimately carries dead <c>ShrkCreateModule</c>
        /// metadata (see <c>FastInitKeyedDraws</c>), so recursing would refuse the very graphs
        /// <c>ToConcreteModel</c> executes. An invoke of a body that does carry live machinery is
        /// left to fail downstream — the conservative half of the trade, taken because the
        /// alternative refuses graphs that run today.</para>
        ///
        /// <para>This is the executability question, and <see cref="IsModuleStageMachinery"/> the
        /// classification one: an initializer-typed invoke marks a graph as pre-lowering while
        /// running perfectly well.</para>
        /// </summary>
        public static bool IsUnrunnableModuleOp(this FastNode node)
            => node.OpCode != InternalOpCodes.FUNCTION_INVOKE
                && InternalOpCodes.IsModuleStageOp(node.OpCode);

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
