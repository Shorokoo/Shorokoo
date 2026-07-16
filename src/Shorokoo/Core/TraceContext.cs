using System.Diagnostics;

namespace Shorokoo.Core
{
    /// <summary>
    /// The ambient state of one graph trace — one execution of body code that creates graph
    /// nodes. Carries the state every kind of trace needs (<see cref="Loopers"/>); the
    /// scoping mechanics live in <see cref="AmbientScope{TScope}"/>, and the registries that
    /// only a module build harvests live on the derived <see cref="ModuleBuildContext"/>.
    ///
    /// <para>A plain <see cref="TraceContext"/> (<see cref="EnterIsolated"/>) backs traces
    /// with no harvesting build: standalone <c>LoopAPI.Iterate</c> loops (hand-built graphs)
    /// and internal node rebuilds that must not leak into an enclosing trace's loop-pass
    /// tracking.</para>
    /// </summary>
    internal class TraceContext : AmbientScope<TraceContext>
    {
        /// <summary>The loop-tracing state of this trace (see <see cref="LooperStack"/>).</summary>
        internal LooperStack Loopers { get; } = new LooperStack();

        /// <summary>Enters an isolated trace: same loop tracking as any trace, but nothing
        /// records into it and nothing harvests it.</summary>
        internal static TraceContext EnterIsolated() => EnterScope(new TraceContext());

        protected override void OnExiting()
            => Debug.Assert(Loopers.Count == 0,
                "TraceContext: a LoopAPI looper is still active — an Iterate trace was left " +
                "open across the context boundary.");
    }
}
