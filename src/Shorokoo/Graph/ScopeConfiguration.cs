using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
namespace Shorokoo.Graph
{
    /// <summary>
    /// How aggressively a scope should claim free (optional) nodes when
    /// <see cref="FastComputationGraph.ConfigureScopes"/> reorders the graph.
    /// </summary>
    public enum ScopeSize
    {
        /// <summary>Pull every free node positionally inside the scope.</summary>
        Maximal,

        /// <summary>Push every free node positionally outside the scope.</summary>
        Minimal,
    }

    /// <summary>
    /// Tie-breaker for nodes that are free in two nested scopes whose preferred
    /// <see cref="ScopeSize"/>s disagree. A "conflict" arises only at cross-kind
    /// nesting (a loop nested in an if or vice versa) when
    /// <c>loopSize != ifSize</c>; same-kind nesting always agrees because the
    /// size is uniform per kind.
    /// </summary>
    public enum ScopePriority
    {
        /// <summary>The loop scope's size preference wins.</summary>
        Loop,

        /// <summary>The if scope's size preference wins.</summary>
        If,

        /// <summary>
        /// No preference. Only valid when no nested-scope conflict mixes kinds —
        /// i.e. when every nested-tension pair shares its kind.
        /// <see cref="FastComputationGraph.ConfigureScopes"/> throws if this
        /// assumption is violated.
        /// </summary>
        None,
    }
}
