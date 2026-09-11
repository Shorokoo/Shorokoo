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
        /// Reorders <see cref="Nodes"/> so that every scope — each (LOOP_OPEN, LOOP_CLOSE)
        /// and (IF_OPEN, IF_CLOSE) pair — positionally contains exactly the nodes that
        /// belong to it. Mutates this graph in place.
        ///
        /// <para>
        /// Two things decide that, in order. First,
        /// <see cref="Shorokoo.Core.Nodes.Processors.Fast.FastIfBranchScoper"/> moves each
        /// <c>IF</c> branch inside the <c>IF</c> that selects it, so a branch runs only when
        /// its condition picks it. Then every remaining node is placed inside each scope its
        /// data flow does not force it out of:
        /// </para>
        /// <list type="bullet">
        ///   <item>Must-in S: the scope's open node is in N's ancestors.</item>
        ///   <item>Must-out S: the scope's close node is not in N's descendants, N has
        ///   another scope's close as a descendant where that other scope is not nested
        ///   inside S (non-overlap rule), or some scope enclosing S forces N out.</item>
        ///   <item>Otherwise N goes inside S. A node left outside a scope it could have been
        ///   in is a node that runs when that scope does not, so the widest legal placement
        ///   is the only correct one — there is nothing here for a caller to tune.</item>
        /// </list>
        /// </summary>
        /// <exception cref="System.InvalidOperationException">
        /// Thrown if open/close pairs overlap rather than nest, an open or close is
        /// unmatched, or a node is required inside a scope that an enclosing scope forces
        /// it out of (a leaked body value).
        /// </exception>
        public void ConfigureScopes()
        {
            FastIfBranchScoper.ScopeAllIfBranches(this);
            FastScopeConfigurator.Configure(this);
        }
    }
}
