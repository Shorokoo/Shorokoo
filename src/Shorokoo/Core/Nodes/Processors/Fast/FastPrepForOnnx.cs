using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Nodes.Processors.AutoGrad;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// Performs the same close-input identity wrapping as
    /// <see cref="FastAddIdentityForOuterScopeValues"/>. Mutates
    /// <c>graph</c> in place.
    /// </summary>
    internal static class FastPrepForOnnx
    {
        public static void Process(FastComputationGraph graph)
        {
            FastIdentityWrapping.WrapCloseInputs(graph);
        }
    }
}
