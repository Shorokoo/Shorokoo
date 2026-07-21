using Shorokoo.Core;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Utils;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;

namespace Shorokoo.Graph
{
    /// <summary>
    /// The readonly, user-facing computation graph: what module build
    /// (<c>MyModule.ComputationGraph</c>), <c>ToConcreteArchitecture</c>,
    /// <c>ToConcreteModel</c>, and the importers hand out. It carries a reliable
    /// <see cref="Kind"/> describing what the graph <em>is</em> (see
    /// <see cref="GraphKind"/>), stamped where the graph was produced.
    ///
    /// <para>The graph is stored in a private <b>frozen representation</b> — immutable
    /// arrays of immutable node records; there is no
    /// <see cref="InternalComputationGraph"/> reference anywhere in this type, so the
    /// kind (and the invariants behind it) cannot be invalidated by mutation, by
    /// construction rather than by convention. Construction freezes a copy of the
    /// source graph; <see cref="ToInternal"/> thaws a fresh mutable copy — the two
    /// conversions are the only bridges, and both copy.</para>
    /// </summary>
    public sealed partial class ComputationGraph
    {
        /// <summary>
        /// Immutable mirror of one <see cref="FastNode"/>: value/immutable members are
        /// held directly (<see cref="OnnxCSharpAttributes"/> and <see cref="Function"/>
        /// are immutable by design), and the input/output key groups are frozen into
        /// immutable arrays. Field-for-field faithful to what
        /// <see cref="InternalComputationGraph.Clone"/> preserves, so a thawed graph is
        /// indistinguishable from a clone of the source.
        /// </summary>
        private sealed class FrozenNode
        {
            public required FastNodeKey Key { get; init; }
            public required string OpCode { get; init; }
            public required OnnxCSharpAttributes Attributes { get; init; }
            public string? FriendlyName { get; init; }
            public string? StackTrace { get; init; }
            public FastNodeKey? GraphOpenNodeKey { get; init; }
            public string? IdentifierTemplate { get; init; }
            public Function? TargetFunction { get; init; }
            public required ImmutableArray<KeyValuePair<string, ImmutableArray<FastTensorKey?>>> Inputs { get; init; }
            public required ImmutableArray<KeyValuePair<string, ImmutableArray<FastTensorKey?>>> Outputs { get; init; }
        }

        private readonly ImmutableArray<FrozenNode> _nodes;
        private readonly ImmutableArray<FastTensorKey> _inputs;
        private readonly ImmutableArray<FastTensorKey> _outputs;
        private readonly ImmutableArray<string?> _inputNames;
        private readonly ImmutableArray<string?> _outputNames;
        private readonly ImmutableArray<int?>? _outputRankOverrides;

        /// <summary>
        /// What this graph is — a <see cref="GraphKind.Module"/>, a
        /// <see cref="GraphKind.ConcreteArchitecture"/>, or a
        /// <see cref="GraphKind.ConcreteModel"/>. Reliable by construction: stamped by
        /// the producing path, carried through copies and (as the .srk v2 header
        /// <c>stage</c> and the ONNX <c>shrk_graph_kind</c> metadata) through
        /// serialization, never re-derived by op-scanning.
        /// </summary>
        public GraphKind Kind { get; }

        /// <summary>
        /// Freezes a copy of <paramref name="graph"/> into the immutable representation
        /// stamped with <paramref name="kind"/>. The source graph is only read — the
        /// caller keeps ownership of it, and no reference to it (or to any of its
        /// mutable parts) survives in this instance.
        /// </summary>
        internal ComputationGraph(InternalComputationGraph graph, GraphKind kind)
        {
            if (graph is null) throw new System.ArgumentNullException(nameof(graph));
            Kind = kind;
            _nodes = graph.Nodes.Select(Freeze).ToImmutableArray();
            _inputs = [.. graph.Inputs];
            _outputs = [.. graph.Outputs];
            _inputNames = [.. graph.InputUniqueNames];
            _outputNames = [.. graph.OutputUniqueNames];
            _outputRankOverrides = graph.OutputRankOverrides is null
                ? null
                : [.. graph.OutputRankOverrides];
        }

        /// <summary>Re-stamping constructor: shares the frozen (immutable) data.</summary>
        private ComputationGraph(ComputationGraph source, GraphKind kind)
        {
            Kind = kind;
            _nodes = source._nodes;
            _inputs = source._inputs;
            _outputs = source._outputs;
            _inputNames = source._inputNames;
            _outputNames = source._outputNames;
            _outputRankOverrides = source._outputRankOverrides;
        }

        private static FrozenNode Freeze(FastNode node) => new()
        {
            Key = node.Key,
            OpCode = node.OpCode,
            Attributes = node.Attributes,
            FriendlyName = node.FriendlyName,
            StackTrace = node.StackTrace,
            GraphOpenNodeKey = node.GraphOpenNodeKey,
            IdentifierTemplate = node.IdentifierTemplate,
            TargetFunction = node.TargetFunction,
            Inputs = FreezeKeyGroups(node.FullInputs),
            Outputs = FreezeKeyGroups(node.FullOutputs),
        };

        private static ImmutableArray<KeyValuePair<string, ImmutableArray<FastTensorKey?>>> FreezeKeyGroups(
            Dictionary<string, List<FastTensorKey?>> groups)
            => groups
                .Select(kvp => new KeyValuePair<string, ImmutableArray<FastTensorKey?>>(
                    kvp.Key, [.. kvp.Value]))
                .ToImmutableArray();

        private static FastNode Thaw(FrozenNode node)
        {
            var thawed = new FastNode
            {
                Key = node.Key,
                OpCode = node.OpCode,
                Attributes = node.Attributes,
                FriendlyName = node.FriendlyName,
                StackTrace = node.StackTrace,
                GraphOpenNodeKey = node.GraphOpenNodeKey,
                IdentifierTemplate = node.IdentifierTemplate,
                TargetFunction = node.TargetFunction,
            };
            foreach (var kvp in node.Inputs)
                thawed.FullInputs[kvp.Key] = [.. kvp.Value];
            foreach (var kvp in node.Outputs)
                thawed.FullOutputs[kvp.Key] = [.. kvp.Value];
            return thawed;
        }

        /// <summary>
        /// Original <c>UniqueName</c> of each graph input, in declaration order —
        /// the names <c>FromOrderedInputs</c> pairs values with.
        /// </summary>
        public IReadOnlyList<string?> InputNames => _inputNames;

        /// <summary>Original <c>UniqueName</c> of each graph output, in declaration order.</summary>
        public IReadOnlyList<string?> OutputNames => _outputNames;

        /// <summary>
        /// Thaws the frozen representation into a fresh, fully mutable
        /// <see cref="InternalComputationGraph"/> — yours to mutate freely; this
        /// <see cref="ComputationGraph"/> (and its <see cref="Kind"/>) stays valid no
        /// matter what is done to the copy. This is the only way graph structure
        /// leaves the wrapper.
        /// </summary>
        public InternalComputationGraph ToInternal()
        {
            var graph = new InternalComputationGraph
            {
                Inputs = [.. _inputs],
                Outputs = [.. _outputs],
                InputUniqueNames = [.. _inputNames],
                OutputUniqueNames = [.. _outputNames],
                OutputRankOverrides = _outputRankOverrides?.ToArray(),
            };
            foreach (var node in _nodes)
                graph.Nodes.Add(Thaw(node));
            System.Diagnostics.Debug.Assert(graph.IsLinearOrderValid(), "thawed.IsLinearOrderValid()");
            return graph;
        }

        /// <summary>
        /// Freezes <paramref name="graph"/> into a readonly
        /// <see cref="ComputationGraph"/> stamped with <paramref name="kind"/> — or,
        /// when no kind is given, with the kind detected by op-scanning
        /// (<see cref="SrkFileFormat.DetectStage(InternalComputationGraph)"/>, the fallback classification for
        /// graphs that arrive without a stamp). The source is only read; the caller
        /// keeps ownership.
        /// </summary>
        public static ComputationGraph FromInternal(InternalComputationGraph graph, GraphKind? kind = null)
        {
            if (graph is null) throw new System.ArgumentNullException(nameof(graph));
            return new ComputationGraph(graph, kind ?? SrkFileFormat.DetectStage(graph));
        }

        /// <summary>
        /// Returns this graph re-stamped as <paramref name="kind"/> — the escape hatch for
        /// graphs whose stamp is missing or wrong (a legacy file whose header predates the
        /// stamp, a foreign import classified by op-scanning). The target kind must be
        /// valid for the graph's content, checked structurally
        /// (<see cref="SrkFileFormat.DescribeKindViolation"/>): a module must not have
        /// initialized model parameters; a concrete architecture additionally needs a
        /// statically known parameter space and no initialized parameters (the RngSeed
        /// identity excepted); a concrete model needs every parameter statically known and
        /// initialized. Throws naming the violated requirement otherwise. Both wrappers
        /// share the same frozen (immutable) data; no copy is made.
        /// </summary>
        public ComputationGraph WithKind(GraphKind kind)
        {
            if (kind == Kind) return this;
            if (SrkFileFormat.DescribeKindViolation(ToInternal(), kind) is { } violation)
                throw new System.InvalidOperationException(
                    $"Cannot re-stamp this '{SrkFileFormat.StageName(Kind)}' graph as " +
                    $"'{SrkFileFormat.StageName(kind)}': {violation}");
            return new ComputationGraph(this, kind);
        }

        /// <summary>
        /// Fail-fast kind check for user-facing operations: throws when <see cref="Kind"/>
        /// is not <paramref name="required"/>, naming the operation and both kinds via the
        /// shared mismatch format (<see cref="SrkFileFormat.KindMismatchMessage"/>).
        /// </summary>
        internal void RequireKind(GraphKind required, string operation, string? hint = null)
        {
            if (Kind == required) return;
            throw new System.InvalidOperationException(SrkFileFormat.KindMismatchMessage(
                operation, $"a '{SrkFileFormat.StageName(required)}' graph", Kind, hint));
        }

        /// <summary>
        /// Fail-fast check for operations that work on any concretized graph (a concrete
        /// architecture, a concrete model, or a lowered step graph) but not on a module
        /// graph — execution and RNG binding, whose machinery is wired at concretization.
        /// </summary>
        internal void RequireConcretized(string operation, string? hint = null)
        {
            if (Kind != GraphKind.Module) return;
            throw new System.InvalidOperationException(SrkFileFormat.KindMismatchMessage(
                operation, "a concretized graph (a 'concrete-architecture' or 'concrete-model')", Kind,
                hint ?? "Lower the graph with ToConcreteArchitecture(inputHints, ...) " +
                        "(and ToConcreteModel(...) for a runnable model) first."));
        }
    }
}
