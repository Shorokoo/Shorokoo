using System;
using System.Diagnostics;

namespace Shorokoo.Core
{
    /// <summary>
    /// Gatekeeper for the ambient graph-trace state. This class does no work of its own: it
    /// manages <b>access</b> to the per-trace states and the <b>validity</b> of that access,
    /// and nothing else. Consumers enter/exit traces through the disposable
    /// <see cref="Scope"/>s and reach the states through the static properties — whose
    /// accessors are the single point that validates the context requirements and throws
    /// when they don't hold. What is then done with a state (recording pins, registering
    /// state updates, driving loop passes, harvesting) lives entirely at the caller.
    ///
    /// <para><see cref="AmbientScope{TScope}"/> owns the scoping mechanics,
    /// <see cref="TraceContext"/> is a plain data bag only this class touches, and each
    /// state's behavior lives in its feature's file.</para>
    /// </summary>
    internal static class GraphTrace
    {
        // ───────────────────────── entering and exiting ─────────────────────────

        /// <summary>
        /// Disposable handle to an entered trace: <c>using var scope = GraphTrace.Enter…()</c>
        /// exits the trace at the end of the block, even on exceptions. May wrap "nothing
        /// entered" (<see cref="EnterIsolatedIfNone"/> when a trace already exists), in which
        /// case disposing is a no-op.
        /// </summary>
        internal readonly struct Scope : IDisposable
        {
            private readonly TraceContext? _entered;

            internal Scope(TraceContext? entered) => _entered = entered;

            public void Dispose()
            {
                if (_entered is not null)
                    TraceContext.Exit(_entered);
            }
        }

        /// <summary>Enters the trace for one graph-builder body trace; the builder harvests
        /// <see cref="StateUpdates"/> and <see cref="Pins"/> after the body returns.</summary>
        internal static Scope EnterModuleBuild() => new Scope(TraceContext.Enter(isModuleBuild: true));

        /// <summary>Enters an isolated trace — same loop tracking as any trace, but nothing
        /// records into it and nothing harvests it. Shields internal node rebuilds from an
        /// enclosing trace's loop-pass tracking.</summary>
        internal static Scope EnterIsolated() => new Scope(TraceContext.Enter(isModuleBuild: false));

        /// <summary>Enters an isolated trace only if no trace is current (a standalone
        /// LoopAPI.Iterate outside any build); otherwise the returned scope wraps nothing
        /// and disposing it is a no-op.</summary>
        internal static Scope EnterIsolatedIfNone()
            => new Scope(TraceContext.Current is null ? TraceContext.Enter(isModuleBuild: false) : null);

        // ─────────────────────── validated state access ───────────────────────

        /// <summary>Whether any trace is in progress on the current thread.</summary>
        internal static bool IsTracing => TraceContext.Current is not null;

        /// <summary>
        /// The loop-tracing state of the current trace. Requires a trace in progress on the
        /// current thread — callers on paths where none may exist (node interception) check
        /// <see cref="IsTracing"/> first.
        /// </summary>
        internal static LooperStack Loopers
            => (TraceContext.Current ?? throw new InvalidOperationException(
                    "No graph trace is in progress on the current thread.")).Loopers;

        /// <summary>
        /// The state-update registrations of the current module build. The accessor is the
        /// validity gate for <c>Globals.StateUpdate</c>: it requires a module build on the
        /// current thread, and a call site outside any <c>LoopAPI.Iterate</c> body — a loop
        /// body is traced once per construction pass (up to four times), so an in-loop
        /// registration would fire repeatedly, mostly against throwaway nodes, and what a
        /// per-iteration state update should even mean is undefined.
        /// </summary>
        internal static StateUpdateRegistry StateUpdates
        {
            get
            {
                var build = RequireModuleBuild("Globals.StateUpdate");
                if (build.Loopers.InLoopBody)
                    throw new InvalidOperationException(
                        "Globals.StateUpdate is not supported inside a LoopAPI.Iterate body. " +
                        "Compute the new value inside the loop, then register the update " +
                        "once after the loop, from the loop's final value.");
                return build.StateUpdates;
            }
        }

        /// <summary>
        /// The pin recordings of the current module build. The accessor is the validity
        /// gate for <c>Rng.Pin</c>: it requires a module build on the current thread —
        /// nothing else could ever apply a pin.
        /// </summary>
        internal static RngPinRegistry Pins => RequireModuleBuild("Rng.Pin").Pins;

        // ────────────────────────────── internals ──────────────────────────────

        private static TraceContext RequireModuleBuild(string api)
        {
            var trace = TraceContext.Current;
            if (trace is null)
                throw new InvalidOperationException(
                    $"{api} may only be called inside a module body — a [Module] Inline " +
                    "method or a codegen-free delegate body — and no module body is " +
                    "executing on the current thread. If the call is written inside a " +
                    "module body, make sure the body does not switch threads: a body runs " +
                    "synchronously on a single thread, and code on async continuations, " +
                    "Parallel.For workers, or other callbacks runs outside the body.");
            if (!trace.IsModuleBuild)
                throw new InvalidOperationException(
                    $"{api} may only be called inside a module body — a [Module] Inline " +
                    "method or a codegen-free delegate body. This call is inside a " +
                    "standalone LoopAPI.Iterate loop with no enclosing module body, so " +
                    "there is no module it could apply to.");
            return trace;
        }
    }

    /// <summary>
    /// The ambient state of one graph trace — pure data, no logic: the loop-tracing state
    /// every trace carries, the registries a module build harvests, and the kind marker.
    /// Only <see cref="GraphTrace"/> touches this class. The scoping mechanics live in
    /// <see cref="AmbientScope{TScope}"/>; each state's behavior lives in its feature's
    /// file.
    /// </summary>
    internal sealed class TraceContext : AmbientScope<TraceContext>
    {
        private TraceContext(bool isModuleBuild) => IsModuleBuild = isModuleBuild;

        internal static TraceContext Enter(bool isModuleBuild)
            => EnterScope(new TraceContext(isModuleBuild));

        /// <summary>Whether this trace is a graph-builder body trace, whose entry point
        /// harvests the registries at build exit (as opposed to an isolated trace).</summary>
        internal bool IsModuleBuild { get; }

        /// <summary>The loop-tracing state of this trace (see <see cref="LooperStack"/>).</summary>
        internal LooperStack Loopers { get; } = new LooperStack();

        /// <summary>The Globals.StateUpdate registrations of this trace.</summary>
        internal StateUpdateRegistry StateUpdates { get; } = new StateUpdateRegistry();

        /// <summary>The Rng.Pin recordings of this trace.</summary>
        internal RngPinRegistry Pins { get; } = new RngPinRegistry();

        protected override void OnExiting()
            => Debug.Assert(Loopers.Count == 0,
                "TraceContext: a LoopAPI looper is still active — an Iterate trace was left " +
                "open across the context boundary.");
    }
}
