using System.Collections.Generic;
using Shorokoo.Core.Graph;
using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using Shorokoo.Core.Nodes.Processors.AutoGrad;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// <para>
    /// ONNX requires that every output of a subgraph (the per-branch outputs an
    /// IF_CLOSE consumes; the body outputs a LOOP_CLOSE consumes) be produced
    /// by a node <em>inside</em> that subgraph — outer-scope values may be
    /// <em>referenced</em> from a subgraph but cannot be subgraph outputs
    /// directly. To guarantee that, we wrap every close-node input in an
    /// Identity node positioned just before the close, and rewire the close
    /// node's input slot to point at the new Identity output.
    /// </para>
    ///
    /// <para>
    /// This processor mutates <c>graph</c> in place — new Identity
    /// FastNodes are inserted into <see cref="FastComputationGraph.Nodes"/> just
    /// before each close node, and the close node's <see cref="FastNode.FullInputs"/>
    /// slots are rewritten in place.
    /// </para>
    /// </summary>
    internal static class FastAddIdentityForOuterScopeValues
    {
        public static void Process(FastComputationGraph graph)
        {
            FastIdentityWrapping.WrapCloseInputs(graph);
        }
    }
}
