using Shorokoo.Core.Nodes.Processors.AutoGrad;
using Shorokoo.Core.Nodes.Processors.Fast;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Graph;

namespace Shorokoo.Graph
{
    public partial class InternalComputationGraph
    {
        /// <summary>
        /// Reorders <see cref="Nodes"/> so that, for every (LOOP_OPEN, LOOP_CLOSE)
        /// and (IF_OPEN, IF_CLOSE) pair, the choice of which optional ("free")
        /// nodes lie positionally inside the scope is governed by
        /// <paramref name="loopSize"/> and <paramref name="ifSize"/>. Nested-scope
        /// size conflicts (cross-kind pairs with disagreeing sizes) are resolved
        /// by <paramref name="priority"/>. Mutates this graph in place.
        ///
        /// <para>
        /// Per-scope classification of every non-boundary node N:
        /// <list type="bullet">
        ///   <item>Must-in S: the scope's open node is in N's ancestors.</item>
        ///   <item>Must-out S: the scope's close node is not in N's descendants,
        ///   or N has another scope's close as a descendant where that other
        ///   scope is not nested inside S (non-overlap rule).</item>
        ///   <item>Free: neither must-in nor must-out — placement is decided
        ///   by <paramref name="loopSize"/>/<paramref name="ifSize"/> and
        ///   <paramref name="priority"/>.</item>
        /// </list>
        /// </para>
        ///
        /// <para>
        /// Before any of that,
        /// <see cref="Shorokoo.Core.Nodes.Processors.Fast.FastIfBranchScoper"/> moves each
        /// <c>IF</c> branch inside the <c>IF</c> that selects it. That is a correctness step
        /// rather than a size preference — a branch left outside runs whatever the condition
        /// says — so it is not governed by <paramref name="ifSize"/>, which decides only where
        /// the genuinely free nodes go.
        /// </para>
        /// </summary>
        /// <exception cref="System.InvalidOperationException">
        /// Thrown if open/close pairs overlap rather than nest, an open or close
        /// is unmatched, a node is forced both in and out of the same scope
        /// (leaked body value), or <paramref name="priority"/> is
        /// <see cref="ScopePriority.None"/> while the graph contains a
        /// cross-kind nested pair and <paramref name="loopSize"/> differs from
        /// <paramref name="ifSize"/>.
        /// </exception>
        public void ConfigureScopes(ScopeSize loopSize, ScopeSize ifSize, ScopePriority priority)
        {
            FastIfBranchScoper.ScopeAllIfBranches(this);
            FastScopeConfigurator.Configure(this, loopSize, ifSize, priority);
        }
    }
}
