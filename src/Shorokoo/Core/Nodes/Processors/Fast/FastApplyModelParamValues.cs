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
    /// Replaces every <c>MODEL_PARAM</c> node in <c>graph</c> with a
    /// <c>MODEL_PARAM_DATA</c> node holding the corresponding value from
    /// <c>paramValues</c>, carrying the original trainability flag. Trainable and
    /// state params alike therefore serialize as ONNX <c>graph.initializer</c>
    /// tensors (never as baked <c>Constant</c> op-nodes), matching ONNX convention
    /// and keeping the params discoverable/retrainable on a loaded model. Each
    /// replacement preserves the original output <see cref="FastTensorKey"/> so
    /// downstream consumers stay valid; the disconnected initializer-param
    /// producers are then swept by
    /// <see cref="FastProcessorHelper.RemoveUnreachableNodes"/>.
    /// </summary>
    internal static class FastApplyModelParamValues
    {
        public static InternalComputationGraph Process(
            InternalComputationGraph graph,
            ImmutableDictionary<ModelId, TensorData> paramValues)
        {
            var workGraph = graph.Clone();

            var modelParamDataAttrDefs = Definitions.NodeDefinitions[InternalOpCodes.MODEL_PARAM_DATA].AttributeDefs;

            foreach (var node in workGraph.Nodes)
            {
                if (node.OpCode != InternalOpCodes.MODEL_PARAM) continue;

                // The RngSeed parameter at reserved ModelId [0] is not a weight — it is
                // filled by ApplyRngConfig (the caller binds the default identity when no
                // config was given), never from the trainable-parameter value set.
                if (node.Attributes.GetIntsVal(OnnxOpAttributeNames.ShrkAttrLocalModelId) is [0])
                    continue;

                var modelIdVals = node.Attributes.GetIntsVal(OnnxOpAttributeNames.ShrkAttrLocalModelId).AssertNotNull();
                var modelId = new ModelId(modelIdVals);

                // The RngExecutionCounter is framework bookkeeping that a materialization source
                // may omit — notably a .safetensors interchange file, which deliberately excludes
                // it. When no value is supplied for it, fall back to its initializer default (0)
                // rather than requiring it like a weight. A native .skpt always writes the counter
                // (CollectWeightNodes keeps it), so on resume its checkpointed value is present and
                // used; in practice this fallback only fires for an interchange import that omits
                // it. The cheap dictionary lookup is tested first so the identifier parse runs only
                // for a param actually absent from the supplied set (normally just the counter).
                var paramValue = !paramValues.ContainsKey(modelId)
                                 && FastInjectRngDrawCounter.IsExecutionCounter(node.IdentifierTemplate)
                    ? FastInjectRngDrawCounter.ExecutionCounterInitialValue()
                    : paramValues[modelId];
                var isTrainable = node.Attributes.GetBoolVal(OnnxOpAttributeNames.ShrkAttrIsTrainable) ?? false;

                node.OpCode = InternalOpCodes.MODEL_PARAM_DATA;
                node.Attributes = OnnxCSharpAttributes.FromCSharpVals(
                    new Dictionary<string, object?>
                    {
                        [OnnxOpAttributeNames.ShrkAttrTensorData] = paramValue,
                        [OnnxOpAttributeNames.ShrkAttrIsTrainable] = isTrainable,
                    },
                    modelParamDataAttrDefs);
                // IdentifierTemplate carries through unchanged so the round-trip
                // back to ComputationGraph keeps the original parameter name.

                node.FullInputs = new Dictionary<string, List<FastTensorKey?>>();
                node.TargetFunction = null;
            }

            FastProcessorHelper.RemoveUnreachableNodes(workGraph);
            // Run FastSimplify so the constant-folding / sequence-folding / unrolling
            // pipeline reshapes the graph in the way the CG-side
            // ApplyModelParamValues + RebuildGraph + TopologicalOrder + RepairScopeNesting
            // pipeline does — without it, the in-place MODEL_PARAM rewrites can leave
            // the rebuilt CG with an empty IF/LOOP body when body-side dependencies cross
            // the new param-data node, breaking ONNX serialization.
            FastSimplify.Process(workGraph);
            return workGraph;
        }
    }
}
