using Shorokoo.Graph;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using Shorokoo.Core.Utils;
using Shorokoo.Onnx;
using System.Collections.Generic;
using System.Collections.Immutable;
using Shorokoo.Core.Nodes.Processors.AutoGrad;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// Fast-native port of <c>ApplyModelParamValues</c>.
    /// Replaces every <c>TRAINABLE_PARAM</c> node in <c>graph</c> with either a
    /// <c>CONSTANT</c> (for trainable params) or a <c>MODEL_PARAM_DATA</c> (for state params)
    /// holding the corresponding value from <c>paramValues</c>. Each replacement
    /// preserves the original output <see cref="FastTensorKey"/> so downstream consumers stay
    /// valid; the disconnected initializer-param producers are then swept by
    /// <see cref="FastProcessorHelper.RemoveUnreachableNodes"/>.
    /// </summary>
    internal static class FastApplyModelParamValues
    {
        public static FastComputationGraph Process(
            FastComputationGraph graph,
            ImmutableDictionary<ModelId, TensorData> paramValues)
        {
            var workGraph = graph.Clone();

            var constantAttrDefs = Definitions.NodeDefinitions[OpCodes.CONSTANT].AttributeDefs;
            var modelParamDataAttrDefs = Definitions.NodeDefinitions[InternalOpCodes.MODEL_PARAM_DATA].AttributeDefs;

            foreach (var node in workGraph.Nodes)
            {
                if (node.OpCode != InternalOpCodes.TRAINABLE_PARAM) continue;

                var modelIdVals = node.Attributes.GetIntsVal(OnnxOpAttributeNames.ShrkAttrLocalModelId).AssertNotNull();
                var modelId = new ModelId(modelIdVals);
                var paramValue = paramValues[modelId];
                var isTrainable = node.Attributes.GetBoolVal(OnnxOpAttributeNames.ShrkAttrIsTrainable) ?? false;

                if (isTrainable)
                {
                    node.OpCode = OpCodes.CONSTANT;
                    node.Attributes = OnnxCSharpAttributes.FromCSharpVals(
                        new Dictionary<string, object?> { [OnnxOpAttributeNames.AttrValue] = paramValue },
                        constantAttrDefs);
                    node.IdentifierTemplate = null;
                }
                else
                {
                    node.OpCode = InternalOpCodes.MODEL_PARAM_DATA;
                    node.Attributes = OnnxCSharpAttributes.FromCSharpVals(
                        new Dictionary<string, object?>
                        {
                            [OnnxOpAttributeNames.ShrkAttrTensorData] = paramValue,
                            [OnnxOpAttributeNames.ShrkAttrIsTrainable] = false,
                        },
                        modelParamDataAttrDefs);
                    // IdentifierTemplate carries through unchanged so the round-trip
                    // back to ComputationGraph keeps the original parameter name.
                }

                node.FullInputs = new Dictionary<string, List<FastTensorKey?>>();
                node.TargetFunction = null;
            }

            FastProcessorHelper.RemoveUnreachableNodes(workGraph);
            // Run FastSimplify so the constant-folding / sequence-folding / unrolling
            // pipeline reshapes the graph in the way the CG-side
            // ApplyModelParamValues + RebuildGraph + TopologicalOrder + RepairScopeNesting
            // pipeline does — without it, the in-place TRAINABLE_PARAM rewrites can leave
            // the rebuilt CG with an empty IF/LOOP body when body-side dependencies cross
            // the new constant, breaking ONNX serialization.
            FastSimplify.Process(workGraph);
            return workGraph;
        }
    }
}
