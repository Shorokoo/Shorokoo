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
        public static void Process(InternalComputationGraph graph)
        {
            FastComposeContiguousReshapes.Process(graph);
            FastIdentityWrapping.WrapCloseInputs(graph);
        }
    }
}
