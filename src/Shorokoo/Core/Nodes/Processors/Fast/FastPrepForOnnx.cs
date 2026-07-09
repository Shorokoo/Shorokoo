using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Nodes.Processors.AutoGrad;
using System.Collections.Generic;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// ONNX-export-only preparation: composes contiguous reshape chains
    /// (<see cref="FastComposeContiguousReshapes"/> — works around an ONNX Runtime
    /// ReshapeFusion crash, see Shorokoo/Shorokoo#10) and performs the same close-input
    /// identity wrapping as <see cref="FastAddIdentityForOuterScopeValues"/>. Mutates
    /// <c>graph</c> in place.
    /// </summary>
    internal static class FastPrepForOnnx
    {
        public static void Process(FastComputationGraph graph)
        {
            FastComposeContiguousReshapes.Process(graph);
            FastIdentityWrapping.WrapCloseInputs(graph);

            // The RNG carriers — the compact key vector (SHRK_RNG_KEY_VECTOR) and each feed
            // site's key entity (SHRK_RNG_KEY) — keep their data but become plain CONSTANTs,
            // so the ONNX runtime treats keys as ordinary tensor data. Prep-only, exactly
            // once: a non-prepped save keeps the entities intact (metadata and all), which is
            // what makes a loaded model re-bindable — ApplyRngConfig re-materializes the
            // entities' values in place and the lowered draws Gather from them. (Key
            // derivation already ran — FastLowerRandomOps precedes this pass. The key
            // vector's identity survives save/load as the reserved-name initializer the
            // serializer mirrors it into; see AttachRngKeyVector.)
            foreach (var node in graph.Nodes)
            {
                if (node.OpCode != InternalOpCodes.SHRK_RNG_KEY_VECTOR &&
                    node.OpCode != InternalOpCodes.SHRK_RNG_KEY)
                    continue;
                var data = node.Attributes.GetTensorVal(OnnxOpAttributeNames.AttrValue);
                var constDefs = Definitions.NodeDefinitions[OpCodes.CONSTANT].AttributeDefs;
                node.OpCode = OpCodes.CONSTANT;
                node.Attributes = OnnxCSharpAttributes.FromCSharpVals(
                    new Dictionary<string, object?>
                    { [OnnxOpAttributeNames.AttrValue] = data }, constDefs);
            }
        }
    }
}
