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
    /// New code should prefer the direct FastCG API (<see cref="InternalComputationGraph.Inputs"/>,
    /// <see cref="InternalComputationGraph.Outputs"/>, <see cref="InternalComputationGraphConverter.BuildNodes"/>,
    /// <see cref="InternalComputationGraphConverter.FunctionsPostOrder"/>). Each property re-builds
    /// the Variable / Node view on every access.
    /// </summary>
    public partial class InternalComputationGraph
    {
        public ImmutableArray<Node> TopologicalOrderNodes
            => InternalComputationGraphConverter.BuildNodes(this).nodesInTopoOrder;

        public ImmutableArray<Variable> InputTensors
            => InternalComputationGraphConverter.BuildNodes(this).inputs;

        public ImmutableArray<Variable> OutputTensors
            => InternalComputationGraphConverter.BuildNodes(this).outputs;

        public ImmutableArray<Function> FunctionsPostOrlder
            => InternalComputationGraphConverter.FunctionsPostOrder(this);

        // Note: `LocalFunctions` shadows the static helper as an instance property.
        // Prefer `InternalComputationGraphConverter.LocalFunctions(graph)` in new code.
        public ImmutableArray<Function> LocalFunctions
            => InternalComputationGraphConverter.LocalFunctions(this);
    }

    /// <summary>
    /// Identity-passthrough extensions for callers that used to round-trip via
    /// <c>InternalComputationGraphConverter.ToFastGraph</c> / <c>ToComputationGraph</c>.
    /// </summary>
    public static class InternalComputationGraphCompat
    {
        public static InternalComputationGraph ToFastGraph(this InternalComputationGraph graph) => graph;

        public static InternalComputationGraph ToComputationGraph(this InternalComputationGraph graph) => graph;

        /// <summary>
        /// .Length alias on a <c>List&lt;FastTensorKey&gt;</c> for legacy callers that
        /// used to read .Length on the Variable[] form of Inputs/Outputs.
        /// </summary>
        public static int Length(this System.Collections.Generic.List<FastTensorKey> list) => list.Count;

        public static ImmutableArray<Variable> GetHyperparamInputs(this InternalComputationGraph graph)
            => graph.InputTensors
                .Where(x => x.OwningNode.Attributes.GetEnumVal<InputType>(OnnxOpAttributeNames.ShrkAttrInputType) == InputType.Hyperparam)
                .ToImmutableArray();

        public static ImmutableArray<Variable> GetNonHyperparamInputs(this InternalComputationGraph graph)
            => graph.InputTensors
                .Where(x => x.OwningNode.Attributes.GetEnumVal<InputType>(OnnxOpAttributeNames.ShrkAttrInputType) != InputType.Hyperparam)
                .ToImmutableArray();

        /// <summary>
        /// Tensors emitted by <see cref="InternalComputationGraph.Nodes"/> in topological order
        /// (one Variable per output slot). Mirrors the deleted CG property.
        /// </summary>
        public static ImmutableArray<Variable> TopologicalOrderTensors(this InternalComputationGraph graph)
            => InternalComputationGraphConverter.BuildNodes(graph).nodesInTopoOrder
                .SelectMany(n => n.Outputs)
                .NotNulls()
                .ToImmutableArray();
    }

    public static partial class InternalComputationGraphConverter
    {
        /// <summary>
        /// Identity overload of <c>ToFastGraph</c> for callers that already hold a
        /// <see cref="InternalComputationGraph"/>. The CG → FastCG bridge is gone.
        /// </summary>
        public static InternalComputationGraph ToFastGraph(InternalComputationGraph graph, bool useSequentialIds = false)
            => graph;

        /// <summary>
        /// Identity overload of <c>ToComputationGraph</c> for callers that already
        /// hold a <see cref="InternalComputationGraph"/>. The FastCG → CG bridge is gone.
        /// </summary>
        public static InternalComputationGraph ToComputationGraph(InternalComputationGraph graph)
            => graph;
    }
}
