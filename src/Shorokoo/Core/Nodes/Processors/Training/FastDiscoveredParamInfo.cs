using Shorokoo.Graph;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using System.Collections.Immutable;

namespace Shorokoo.Core.Nodes.Processors.Training
{
    /// <summary>
    /// Fast-side info about a single trainable or state parameter discovered in a
    /// <see cref="InternalComputationGraph"/> by <see cref="FastDiscoverTrainableParamsProcessor"/>
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

        /// <summary>
        /// The FURTHER sites that resolve to this same parameter, as the key each produces and the
        /// node producing it: bare references to it (<c>IModel.GetTrainableParam</c>, which reads
        /// the parameter the model already owns rather than declaring one — Shorokoo/Shorokoo#263)
        /// and repeat definitions of it (one model handle called twice leaves one definition site
        /// per call — Shorokoo/Shorokoo#284). Neither contributes a field of its own: its consumers
        /// are rewired to this parameter's field and its node is removed alongside
        /// <see cref="Node"/>.
        /// </summary>
        public ImmutableArray<(FastTensorKey OutputKey, FastNode Node)> Aliases { get; }

        internal FastDiscoveredParamInfo(
            string name, FastTensorKey outputKey, bool isTrainable,
            DType dtype, int? rank, DataStructure structure, FastNode node,
            ImmutableArray<(FastTensorKey OutputKey, FastNode Node)> aliases)
        {
            Name = name;
            OutputKey = outputKey;
            IsTrainable = isTrainable;
            DType = dtype;
            Rank = rank;
            Structure = structure;
            Node = node;
            Aliases = aliases.IsDefault ? [] : aliases;
        }
    }
}
