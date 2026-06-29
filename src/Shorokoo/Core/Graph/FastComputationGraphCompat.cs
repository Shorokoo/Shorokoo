using Shorokoo.Graph;
using Shorokoo.Core.Graph;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Modules;
using Shorokoo.Core.Utils;
using Shorokoo.Onnx;
using System;
using System.Collections.Immutable;
using System.Linq;
using Shorokoo.Core.Nodes.Processors.Helpers;

namespace Shorokoo.Graph
{
    /// <summary>
    /// Compatibility shims for legacy <c>ComputationGraph</c> property accessors.
    /// New code should prefer the direct FastCG API (<see cref="FastComputationGraph.Inputs"/>,
    /// <see cref="FastComputationGraph.Outputs"/>, <see cref="FastComputationGraphConverter.BuildNodes"/>,
    /// <see cref="FastComputationGraphConverter.FunctionsPostOrder"/>). Each property re-builds
    /// the Variable / Node view on every access.
    /// </summary>
    public partial class FastComputationGraph
    {
        public ImmutableArray<Node> TopologicalOrderNodes
            => FastComputationGraphConverter.BuildNodes(this).nodesInTopoOrder;

        public ImmutableArray<Variable> InputTensors
            => FastComputationGraphConverter.BuildNodes(this).inputs;

        public ImmutableArray<Variable> OutputTensors
            => FastComputationGraphConverter.BuildNodes(this).outputs;

        public ImmutableArray<Function> FunctionsPostOrlder
            => FastComputationGraphConverter.FunctionsPostOrder(this);

        // Note: `LocalFunctions` shadows the static helper as an instance property.
        // Prefer `FastComputationGraphConverter.LocalFunctions(graph)` in new code.
        public ImmutableArray<Function> LocalFunctions
            => FastComputationGraphConverter.LocalFunctions(this);
    }

    /// <summary>
    /// Identity-passthrough extensions for callers that used to round-trip via
    /// <c>FastComputationGraphConverter.ToFastGraph</c> / <c>ToComputationGraph</c>.
    /// </summary>
    public static class FastComputationGraphCompat
    {
        public static FastComputationGraph ToFastGraph(this FastComputationGraph graph) => graph;

        public static FastComputationGraph ToComputationGraph(this FastComputationGraph graph) => graph;

        /// <summary>
        /// .Length alias on a <c>List&lt;FastTensorKey&gt;</c> for legacy callers that
        /// used to read .Length on the Variable[] form of Inputs/Outputs.
        /// </summary>
        public static int Length(this System.Collections.Generic.List<FastTensorKey> list) => list.Count;

        public static ImmutableArray<Variable> GetHyperparamInputs(this FastComputationGraph graph)
            => graph.InputTensors
                .Where(x => x.OwningNode.Attributes.GetEnumVal<InputType>(OnnxOpAttributeNames.ShrkAttrInputType) == InputType.Hyperparam)
                .ToImmutableArray();

        public static ImmutableArray<Variable> GetNonHyperparamInputs(this FastComputationGraph graph)
            => graph.InputTensors
                .Where(x => x.OwningNode.Attributes.GetEnumVal<InputType>(OnnxOpAttributeNames.ShrkAttrInputType) != InputType.Hyperparam)
                .ToImmutableArray();

        /// <summary>
        /// Tensors emitted by <see cref="FastComputationGraph.Nodes"/> in topological order
        /// (one Variable per output slot). Mirrors the deleted CG property.
        /// </summary>
        public static ImmutableArray<Variable> TopologicalOrderTensors(this FastComputationGraph graph)
            => FastComputationGraphConverter.BuildNodes(graph).nodesInTopoOrder
                .SelectMany(n => n.Outputs)
                .NotNulls()
                .ToImmutableArray();
    }

    public static partial class FastComputationGraphConverter
    {
        /// <summary>
        /// Identity overload of <c>ToFastGraph</c> for callers that already hold a
        /// <see cref="FastComputationGraph"/>. The CG → FastCG bridge is gone.
        /// </summary>
        public static FastComputationGraph ToFastGraph(FastComputationGraph graph, bool useSequentialIds = false)
            => graph;

        /// <summary>
        /// Identity overload of <c>ToComputationGraph</c> for callers that already
        /// hold a <see cref="FastComputationGraph"/>. The FastCG → CG bridge is gone.
        /// </summary>
        public static FastComputationGraph ToComputationGraph(FastComputationGraph graph)
            => graph;
    }
}
