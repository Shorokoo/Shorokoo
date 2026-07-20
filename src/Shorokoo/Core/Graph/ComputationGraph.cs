using Shorokoo.Core.Utils;
using System.Collections.Generic;

namespace Shorokoo.Graph
{
    /// <summary>
    /// The readonly, user-facing computation graph: what module build
    /// (<c>MyModule.ComputationGraph</c>), <c>ToConcreteArchitecture</c>,
    /// <c>ToConcreteModel</c>, and the importers hand out. It carries a reliable
    /// <see cref="Kind"/> describing what the graph <em>is</em> (see
    /// <see cref="GraphKind"/>), stamped where the graph was produced — because this
    /// type exposes no mutating surface and hands out no reference to its underlying
    /// mutable representation, the kind (and the invariants behind it) cannot be
    /// invalidated after the fact.
    ///
    /// <para>The modifiable counterpart is <see cref="InternalComputationGraph"/>, the
    /// type the lowering passes and processors work on. The wrapped graph is owned
    /// exclusively by this instance and never escapes: every conversion copies —
    /// <see cref="ToInternal"/> returns a mutable deep copy, and
    /// <see cref="FromInternal"/> freezes a deep copy of a mutable graph.</para>
    /// </summary>
    public sealed partial class ComputationGraph
    {
        private readonly InternalComputationGraph _graph;

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
        /// Wraps <paramref name="graph"/> without copying. The caller relinquishes the
        /// reference — from here on this instance owns the graph exclusively, and no
        /// member ever hands it back out (only <see cref="ToInternal"/> copies of it).
        /// Producing paths hand in freshly built graphs; everyone else goes through
        /// <see cref="FromInternal"/>, which copies.
        /// </summary>
        internal ComputationGraph(InternalComputationGraph graph, GraphKind kind)
        {
            _graph = graph ?? throw new System.ArgumentNullException(nameof(graph));
            Kind = kind;
        }

        /// <summary>
        /// Original <c>UniqueName</c> of each graph input, in declaration order —
        /// the names <c>FromOrderedInputs</c> pairs values with.
        /// </summary>
        public IReadOnlyList<string?> InputNames => _graph.InputUniqueNames;

        /// <summary>Original <c>UniqueName</c> of each graph output, in declaration order.</summary>
        public IReadOnlyList<string?> OutputNames => _graph.OutputUniqueNames;

        /// <summary>
        /// Converts to the modifiable representation by <b>deep copy</b>: the returned
        /// <see cref="InternalComputationGraph"/> is yours to mutate freely — this
        /// <see cref="ComputationGraph"/> (and its <see cref="Kind"/>) stays valid no
        /// matter what is done to the copy. This is the only way graph structure
        /// leaves the wrapper.
        /// </summary>
        public InternalComputationGraph ToInternal() => _graph.Clone();

        /// <summary>
        /// Freezes a <b>deep copy</b> of <paramref name="graph"/> into a readonly
        /// <see cref="ComputationGraph"/> stamped with <paramref name="kind"/> — or,
        /// when no kind is given, with the kind detected by op-scanning
        /// (<see cref="SrkFileFormat.DetectStage(InternalComputationGraph)"/>, the fallback classification for
        /// graphs that arrive without a stamp).
        /// </summary>
        public static ComputationGraph FromInternal(InternalComputationGraph graph, GraphKind? kind = null)
        {
            if (graph is null) throw new System.ArgumentNullException(nameof(graph));
            return new ComputationGraph(graph.Clone(), kind ?? SrkFileFormat.DetectStage(graph));
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
        /// share the same exclusively-held graph; no copy is made.
        /// </summary>
        public ComputationGraph WithKind(GraphKind kind)
        {
            if (kind == Kind) return this;
            if (SrkFileFormat.DescribeKindViolation(_graph, kind) is { } violation)
                throw new System.InvalidOperationException(
                    $"Cannot re-stamp this '{SrkFileFormat.StageName(Kind)}' graph as " +
                    $"'{SrkFileFormat.StageName(kind)}': {violation}");
            return new ComputationGraph(_graph, kind);
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
