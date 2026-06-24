using Shorokoo.Graph;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Nodes.Processors.Training
{
    /// <summary>
    /// Fast-side info about a single trainable or state parameter discovered in a
    /// <see cref="FastComputationGraph"/> by <see cref="FastDiscoverTrainableParamsProcessor"/>
    /// or <see cref="FastDiscoverStateParamsProcessor"/>. All metadata is read directly off
    /// <see cref="FastNode.Attributes"/> — no round-trip to <c>ComputationGraph</c>
    /// is performed.
    /// </summary>
    internal sealed class FastDiscoveredParamInfo
    {
        /// <summary>The resolved parameter name (sanitized identifier-template / friendly-name fallback).</summary>
        public string Name { get; }

        /// <summary>Output tensor key produced by <see cref="Node"/>.</summary>
        public FastTensorKey OutputKey { get; }

        /// <summary>Whether this parameter is marked as trainable.</summary>
        public bool IsTrainable { get; }

        /// <summary>Element data type of the parameter tensor.</summary>
        public DType DType { get; }

        /// <summary>Tensor rank (number of dimensions). Null when not statically known.</summary>
        public int? Rank { get; }

        /// <summary>Top-level data-structure category — always <see cref="DataStructure.Tensor"/>
        /// for the three handled op codes after the Fast lowering pipeline.</summary>
        public DataStructure Structure { get; }

        /// <summary>The Fast node producing this parameter.</summary>
        public FastNode Node { get; }

        internal FastDiscoveredParamInfo(
            string name, FastTensorKey outputKey, bool isTrainable,
            DType dtype, int? rank, DataStructure structure, FastNode node)
        {
            Name = name;
            OutputKey = outputKey;
            IsTrainable = isTrainable;
            DType = dtype;
            Rank = rank;
            Structure = structure;
            Node = node;
        }
    }
}
