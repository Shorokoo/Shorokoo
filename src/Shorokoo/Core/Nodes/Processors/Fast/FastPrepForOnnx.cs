using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Nodes.Processors.AutoGrad;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// ONNX-export-only preparation: composes contiguous reshape chains
    /// (<see cref="FastComposeContiguousReshapes"/>) and performs the same close-input
    /// identity wrapping as <see cref="FastAddIdentityForOuterScopeValues"/>, then removes every
    /// node nothing reads any more. Mutates <c>graph</c> in place.
    ///
    /// <para>The sweep is what keeps a bypassed chain step from running. Left in the graph, a step
    /// inside an <c>IF</c> branch reads a branch value and feeds nothing: scoping, which places a
    /// node by what it reads and what reads it, then finds nothing that holds it in the branch and
    /// hoists it, with what it reads, to the enclosing scope, where it runs whether or not the
    /// branch is taken and fails on what is valid only on the branch's own path.</para>
    /// </summary>
    internal static class FastPrepForOnnx
    {
        public static void Process(InternalComputationGraph graph)
        {
            FastComposeContiguousReshapes.Process(graph);
            FastIdentityWrapping.WrapCloseInputs(graph);
            FastProcessorHelper.RemoveUnreachableNodes(graph);
        }
    }
}
