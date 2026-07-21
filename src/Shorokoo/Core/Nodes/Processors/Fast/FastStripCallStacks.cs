using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Nodes.Processors.AutoGrad;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// Clears the captured stack trace from every node in the graph in place. Apply
    /// before serialization to keep architecture binaries deterministic and compact
    /// — without this, every <c>BuildOnnxModel</c> call would embed a freshly
    /// captured stack trace per node, breaking byte-for-byte reproducibility.
    /// </summary>
    internal static class FastStripCallStacks
    {
        public static void Process(InternalComputationGraph graph)
        {
            foreach (var node in graph.Nodes)
                node.StackTrace = null;
        }
    }
}
