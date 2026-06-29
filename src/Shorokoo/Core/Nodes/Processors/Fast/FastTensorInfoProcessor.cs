using Shorokoo.Graph;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using System;
using System.Collections.Generic;
using Shorokoo.Core.Nodes.Processors.AutoGrad;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// Builds <see cref="FastTensorInfo"/> metadata (dtype, structure, rank, unique name,
    /// owning module function) for tensors in a <see cref="FastComputationGraph"/>.
    ///
    /// <para>
    /// <see cref="FastNode"/> stores op codes, attributes, and tensor-key references but not
    /// per-tensor metadata. To produce that metadata this processor rebuilds the equivalent
    /// <see cref="Variable"/>s via
    /// <see cref="FastComputationGraphConverter.BuildTensorMapping"/> — the Variable
    /// factories are the canonical source of dtype/structure/rank propagation rules, and
    /// <see cref="Variable.UniqueName"/> / <see cref="Variable.ModuleFn"/> are intrinsic
    /// to that side of the model. No <c>ComputationGraph</c> is constructed; we
    /// only need the FastTensorKey → Variable map.
    /// </para>
    /// </summary>
    internal static class FastTensorInfoProcessor
    {
        /// <summary>
        /// Returns a dictionary mapping every <see cref="FastTensorKey"/> in
        /// <paramref name="graph"/> to its <see cref="FastTensorInfo"/>.
        /// </summary>
        public static Dictionary<FastTensorKey, FastTensorInfo> BuildTensorInfoLookup(FastComputationGraph graph)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));

            var tensorMapping = FastComputationGraphConverter.BuildTensorMapping(graph);

            var lookup = new Dictionary<FastTensorKey, FastTensorInfo>(tensorMapping.Count);
            foreach (var (key, variable) in tensorMapping)
            {
                if (key.IsEmpty || variable is null) continue;
                lookup[key] = BuildTensorInfo(key, variable);
            }
            return lookup;
        }

        /// <summary>
        /// Retrieves the <see cref="FastTensorInfo"/> for a single <paramref name="key"/>.
        /// Throws if the key is not produced by any node (including close-node connecting
        /// tensors) in the graph.
        /// </summary>
        public static FastTensorInfo GetTensorInfo(FastComputationGraph graph, FastTensorKey key)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));

            if (!TryGetTensorInfo(graph, key, out var info))
                throw new KeyNotFoundException(
                    $"FastTensorInfoProcessor: tensor {key} is not produced by any node in the graph.");

            return info!;
        }

        /// <summary>
        /// Try-variant of <see cref="GetTensorInfo"/>. Returns false and sets
        /// <paramref name="info"/> to null when the key cannot be resolved in the graph.
        /// </summary>
        public static bool TryGetTensorInfo(FastComputationGraph graph, FastTensorKey key, out FastTensorInfo? info)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));

            if (key.IsEmpty)
            {
                info = null;
                return false;
            }

            var tensorMapping = FastComputationGraphConverter.BuildTensorMapping(graph);
            if (tensorMapping.TryGetValue(key, out var variable) && variable is not null)
            {
                info = BuildTensorInfo(key, variable);
                return true;
            }

            info = null;
            return false;
        }

        private static FastTensorInfo BuildTensorInfo(FastTensorKey key, Variable variable) =>
            new FastTensorInfo
            {
                Key = key,
                DType = variable.Type,
                Structure = variable.Structure(),
                Rank = variable.Rank,
                UniqueName = variable.UniqueName,
                ModuleFn = variable.ModuleFn,
            };
    }
}
