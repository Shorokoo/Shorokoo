using Shorokoo.Graph;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using System;
using System.Collections.Generic;
using System.Linq;
using Shorokoo.Core.Nodes.Processors.AutoGrad;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// Fast version of the state-update lowering pass. Operates directly on a
    /// <see cref="FastComputationGraph"/>:
    /// <list type="number">
    ///   <item>Each <c>STATE_UPDATE_LINK(originalState, updatedState)</c> is rewritten to
    ///         <c>IDENTITY(updatedState)</c>. Its output tensor (the linked updated state) is
    ///         appended to <see cref="FastComputationGraph.Outputs"/> so the executor can
    ///         retrieve the new state value.</item>
    ///   <item>Each <c>WITH_STATE_DEPS(mainOutput, ...stateDeps)</c> is rewritten to
    ///         <c>IDENTITY(mainOutput)</c>. The state-dep inputs are dropped.</item>
    /// </list>
    /// Tensor keys are preserved by mutating nodes in place, so consumers of the lowered
    /// nodes' outputs do not need any rewriting.
    /// </summary>
    internal static class FastLowerStateUpdateNodes
    {
        public static void Process(FastComputationGraph graph)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));

            var hasStateNodes = graph.Nodes.Any(n =>
                n.OpCode == InternalOpCodes.WITH_STATE_DEPS ||
                n.OpCode == InternalOpCodes.STATE_UPDATE_LINK);

            if (!hasStateNodes) return;

            var identityAttrDefs = Definitions.NodeDefinitions[OpCodes.IDENTITY].AttributeDefs;
            var emptyIdentityAttrs = OnnxCSharpAttributes.FromCSharpVals(
                new Dictionary<string, object?>(), identityAttrDefs);

            var newStateOutputs = new List<FastTensorKey>();

            foreach (var node in graph.Nodes)
            {
                if (node.OpCode == InternalOpCodes.STATE_UPDATE_LINK)
                {
                    // STATE_UPDATE_LINK(originalState, updatedState) → IDENTITY(updatedState)
                    var inputs = node.Inputs;
                    var updatedState = inputs[1]
                        ?? throw new InvalidOperationException(
                            "STATE_UPDATE_LINK has null updatedState input.");

                    node.OpCode = OpCodes.IDENTITY;
                    node.Attributes = emptyIdentityAttrs;
                    node.FullInputs = new Dictionary<string, List<FastTensorKey?>>
                    {
                        [""] = new List<FastTensorKey?> { updatedState }
                    };

                    var outputs = node.Outputs;
                    if (outputs.Count == 0 || outputs[0] is null)
                        throw new InvalidOperationException(
                            "STATE_UPDATE_LINK has no output.");
                    newStateOutputs.Add(outputs[0]!.Value);
                }
                else if (node.OpCode == InternalOpCodes.WITH_STATE_DEPS)
                {
                    // WITH_STATE_DEPS(mainOutput, ...stateDeps) → IDENTITY(mainOutput)
                    var inputs = node.Inputs;
                    var mainOutput = inputs[0]
                        ?? throw new InvalidOperationException(
                            "WITH_STATE_DEPS has null mainOutput input.");

                    node.OpCode = OpCodes.IDENTITY;
                    node.Attributes = emptyIdentityAttrs;
                    node.FullInputs = new Dictionary<string, List<FastTensorKey?>>
                    {
                        [""] = new List<FastTensorKey?> { mainOutput }
                    };
                }
            }

            foreach (var key in newStateOutputs)
            {
                graph.Outputs.Add(key);
                graph.OutputUniqueNames.Add(null);
            }
        }
    }
}
