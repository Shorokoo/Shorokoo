namespace Shorokoo.Graph
{
    /// <summary>
    /// What a computation graph <em>is</em> — its position in the lowering lifecycle.
    /// Stamped on <see cref="ComputationGraph.Kind"/> by every producing path (module
    /// build, <c>ToConcreteArchitecture</c>, <c>ToConcreteModel</c>, the importers),
    /// carried through copies and recorded in the .srk v2 header (as the
    /// <c>stage</c> field) so a loader can route or refuse a file at load time
    /// instead of failing deep inside execution (e.g.
    /// <c>No Op registered for ShrkCreateModule</c>).
    /// </summary>
    public enum GraphKind
    {
        /// <summary>A pre-lowering module graph: high-level Shorokoo operators
        /// (<c>ShrkModelInvoke</c> etc.) are present; the shape <em>and number</em> of
        /// model parameters is dynamic.</summary>
        Module,

        /// <summary>A lowered architecture from <c>ToConcreteArchitecture</c>: the
        /// number and shapes of every trainable parameter are statically known and
        /// visible at top level, but values are not yet materialized (parameters
        /// still carry their initializer functions).</summary>
        ConcreteArchitecture,

        /// <summary>A weight-filled, runnable graph from <c>ToConcreteModel</c>: all
        /// model parameters' number, shapes, and values are statically known.</summary>
        ConcreteModel,
    }
}
