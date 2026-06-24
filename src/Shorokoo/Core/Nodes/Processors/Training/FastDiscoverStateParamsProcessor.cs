using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using System;
using System.Collections.Immutable;

namespace Shorokoo.Core.Nodes.Processors.Training
{
    /// <summary>
    /// Discovers all state parameter producer nodes (non-trainable model parameters such as
    /// BatchNorm running stats) in a <see cref="FastComputationGraph"/>. Scans for
    /// <c>TRAINABLE_PARAM</c>, <c>MODEL_PARAM_DATA</c>, and <c>TRAINABLE_PARAM_ID_REF</c>
    /// nodes whose <c>shrk_is_trainable</c> attribute is false.
    /// </summary>
    internal static class FastDiscoverStateParamsProcessor
    {
        public static ImmutableArray<FastDiscoveredParamInfo> Process(FastComputationGraph graph)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));
            return FastDiscoverParamsHelpers.Discover(graph, wantTrainable: false);
        }
    }
}
