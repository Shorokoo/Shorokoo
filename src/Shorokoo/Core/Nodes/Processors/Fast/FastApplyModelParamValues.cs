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
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
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
            IReadOnlyDictionary<ModelId, TensorData> paramValues)
            // Copied first: the values belong to the caller, who goes on holding them -- a training
            // rig keeps the very same tensors as its initial checkpoint -- so the graph takes a
            // literal of its own rather than spending theirs. Per node, so a value whose parameter is
            // not in this graph is never copied.
            => Process(graph, paramValues.ContainsKey,
                id => paramValues[id].CopyTo(Shorokoo.Runtime.ComputeContext.Host).MoveToAttribute());

        /// <summary>
        /// The same against parameter <b>descriptions</b> rather than values: what a build binds when
        /// it has no values yet (Shorokoo/Shorokoo#327), each parameter's slot carrying its declared
        /// shape and dtype and — above the size a shape pass would read — no elements at all.
        /// </summary>
        public static InternalComputationGraph Process(
            InternalComputationGraph graph,
            IReadOnlyDictionary<ModelId, TensorAttribute> paramSlots)
            => Process(graph, paramSlots.ContainsKey, id => paramSlots[id]);

        private static InternalComputationGraph Process(
            InternalComputationGraph graph,
            Func<ModelId, bool> isSupplied,
            Func<ModelId, TensorAttribute> attributeFor)
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
                var paramValue = !isSupplied(modelId)
                                 && FastInjectRngDrawCounter.IsExecutionCounter(node.IdentifierTemplate)
                    ? FastInjectRngDrawCounter.ExecutionCounterInitialValue().MoveToAttribute()
                    : attributeFor(modelId);
                var isTrainable = node.Attributes.GetBoolVal(OnnxOpAttributeNames.ShrkAttrIsTrainable) ?? false;

                // Binding is the last point at which a value and the model that is to use it are
                // both in hand, and the node states the shape the model declared for this
                // parameter — so it is the one place every load path can be held to it. A value of
                // another shape does not fail on its own: matmuls and elementwise ops broadcast, so
                // the graph builds, runs, and returns a differently shaped answer. A declared
                // dimension of -1 is symbolic and constrains nothing.
                var declaredDims = node.Attributes.GetLongsVal(OnnxOpAttributeNames.ShrkAttrShape);
                if (declaredDims is not null && !declaredDims.Contains(-1L)
                    && !paramValue.Shape.Dims.SequenceEqual(declaredDims))
                    throw new InvalidOperationException(
                        $"Parameter '{node.IdentifierTemplate ?? modelId.ToString()}' is declared "
                        + $"[{string.Join(",", declaredDims)}] by this model, but the value being bound "
                        + $"is [{string.Join(",", paramValue.Shape.Dims)}]. A value is bound at the shape "
                        + "the model declares for it, and nothing adapts one to the other — the values "
                        + "may come from another model, or from an initializer that returned a shape "
                        + "other than the one it was called with.");

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
