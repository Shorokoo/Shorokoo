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
    /// type exposes no mutating surface, the kind (and the invariants behind it)
    /// cannot be invalidated after the fact.
    ///
    /// <para>The modifiable counterpart is <see cref="InternalComputationGraph"/>, the
    /// type the lowering passes and processors work on. Conversions between the two
    /// copy in both directions — <see cref="ToInternal"/> returns a mutable deep copy
    /// (a caller-held <see cref="ComputationGraph"/> never observes mutation), and
    /// <see cref="FromInternal"/> freezes a deep copy of a mutable graph.</para>
    /// </summary>
    public sealed class ComputationGraph
    {
        private readonly InternalComputationGraph _graph;

        /// <summary>
        /// What this graph is — a <see cref="GraphKind.Module"/>, a
        /// <see cref="GraphKind.ConcreteArchitecture"/>, or a
        /// <see cref="GraphKind.ConcreteModel"/>. Reliable by construction: stamped by
        /// the producing path, carried through copies and (as the .srk v2 header
        /// <c>stage</c>) through serialization, never re-derived by op-scanning.
        /// </summary>
        public GraphKind Kind { get; }

        /// <summary>
        /// Wraps <paramref name="graph"/> without copying. The caller relinquishes the
        /// reference: retaining and mutating it afterwards would invalidate
        /// <see cref="Kind"/>. Producing paths hand in freshly built graphs; everyone
        /// else goes through <see cref="FromInternal"/>, which copies.
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
        /// The wrapped graph, borrowed without a copy for <b>read-only</b> use
        /// (execution, export, inspection). Never mutate it — that would invalidate
        /// <see cref="Kind"/> behind every holder of this <see cref="ComputationGraph"/>.
        /// Passes take <see cref="ToInternal"/> instead.
        /// </summary>
        internal InternalComputationGraph Internal => _graph;

        /// <summary>
        /// Converts to the modifiable representation by <b>deep copy</b>: the returned
        /// <see cref="InternalComputationGraph"/> shares no mutable state with this
        /// graph, so this <see cref="ComputationGraph"/> (and its <see cref="Kind"/>)
        /// stays valid no matter what is done to the copy.
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
        /// Fail-fast kind check for user-facing operations: throws when <see cref="Kind"/>
        /// is not <paramref name="required"/>, naming the operation and both kinds;
        /// returns the borrowed internal graph (read-only — see <see cref="Internal"/>)
        /// otherwise.
        /// </summary>
        internal InternalComputationGraph RequireKind(GraphKind required, string operation, string? hint = null)
        {
            if (Kind == required) return _graph;
            throw new System.InvalidOperationException(
                $"{operation} requires a '{SrkFileFormat.StageName(required)}' graph, but this graph " +
                $"is a '{SrkFileFormat.StageName(Kind)}'." +
                (hint is null ? string.Empty : " " + hint));
        }
    }
}
