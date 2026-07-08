using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Nodes.Processors.AutoGrad;

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

            // The model's compact RNG key vector: for ONNX execution/export, keep the data but
            // become a plain CONSTANT so the runtime treats it as ordinary (unused) tensor
            // data. (Key derivation already ran — FastLowerRandomOps precedes this pass. The
            // carrier's identity survives save/load as the reserved-name initializer the
            // serializer mirrors it into; see AttachRngKeyVector.)
            foreach (var node in graph.Nodes)
            {
                if (node.OpCode != Shorokoo.Core.Nodes.NodeDefinitions.InternalOpCodes.SHRK_RNG_KEY_VECTOR) continue;
                var data = node.Attributes.GetTensorVal(
                    Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames.AttrValue);
                var constDefs = Shorokoo.Core.Nodes.NodeDefinitions.Definitions
                    .NodeDefinitions[Shorokoo.Core.Nodes.NodeDefinitions.OpCodes.CONSTANT].AttributeDefs;
                node.OpCode = Shorokoo.Core.Nodes.NodeDefinitions.OpCodes.CONSTANT;
                node.Attributes = Shorokoo.Core.Nodes.NodeDefinitions.OnnxCSharpAttributes.FromCSharpVals(
                    new System.Collections.Generic.Dictionary<string, object?>
                    { [Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames.AttrValue] = data }, constDefs);
            }
        }
    }
}
