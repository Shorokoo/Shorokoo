using Shorokoo.Graph;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using System;
using System.Collections.Generic;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// Lowers any <c>STATE_UPDATE_LINK</c> node surviving to ONNX export into an IDENTITY of
    /// the ORIGINAL state value. The training pipeline lowers state updates into explicit
    /// state outputs long before export, so a link reaching this point is in a one-shot
    /// inference graph, where state does not persist across executions: consumers observe
    /// the initial state value and the update computation is dropped. This makes graphs with
    /// module-owned state (Dropout's draw counter, BatchNorm running stats) executable on the
    /// plain inference path without behavioral change.
    /// </summary>
    internal static class FastLowerStateUpdateLinksForInference
    {
        public static void Process(FastComputationGraph graph)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));

            var identityAttrDefs = Definitions.NodeDefinitions[OpCodes.IDENTITY].AttributeDefs;
            foreach (var node in graph.Nodes)
            {
                // STATE_UPDATE_LINK(original, updated) -> Identity(original): consumers see the
                // initial state value; the update computation is dead in one-shot inference.
                // WITH_STATE_DEPS(main, deps...) -> Identity(main): the dependency edges only
                // exist to keep state updates alive for the training pipeline.
                bool isLink = node.OpCode == InternalOpCodes.STATE_UPDATE_LINK;
                bool isDeps = node.OpCode == InternalOpCodes.WITH_STATE_DEPS;
                if (!isLink && !isDeps) continue;

                var forwarded = node.Inputs[0]
                    ?? throw new InvalidOperationException(node.OpCode + " has null primary input.");
                node.OpCode = OpCodes.IDENTITY;
                node.Attributes = OnnxCSharpAttributes.FromCSharpVals(
                    new Dictionary<string, object?>(), identityAttrDefs);
                node.FullInputs = new Dictionary<string, List<FastTensorKey?>>
                {
                    [""] = new List<FastTensorKey?> { forwarded }
                };
            }
        }
    }
}
