using System;
using System.Diagnostics;

namespace Shorokoo.Core
{
    /// <summary>
    /// The single entry point through which features use the ambient graph-trace state.
    /// Consumers call the static methods here and nothing else: entering and exiting traces
    /// (the graph builder, Function's node-rebuild shield, standalone LoopAPI.Iterate),
    /// recording (Rng.Pin, Globals.StateUpdate), harvesting (the graph builder at build
    /// exit), and reaching the loop-tracing state (LoopAPI).
    ///
    /// <para>This is deliberately the one place where the trace state meets its consumers:
    /// the per-feature usage policies — which calls need a module build, which loop pass
    /// records, what each misuse throws — are declared here, while the pieces they
    /// coordinate stay ignorant of each other. <see cref="AmbientScope{TScope}"/> owns the
    /// scoping mechanics, <see cref="TraceContext"/> is a plain data bag only this class
    /// touches, and each registry's behavior lives in its feature's file.</para>
    /// </summary>
    internal static class GraphTrace
    {
        // ───────────────────────── entering and exiting ─────────────────────────

        /// <summary>Enters the trace for one graph-builder body trace. Harvest with
        /// <see cref="HarvestStateUpdates"/> / <see cref="HarvestPins"/> after the body
        /// returns, and pass the handle back to <see cref="Exit"/> in a finally.</summary>
        internal static TraceContext EnterModuleBuild() => TraceContext.Enter(isModuleBuild: true);

        /// <summary>Enters an isolated trace — same loop tracking as any trace, but nothing
        /// records into it and nothing harvests it. Shields internal node rebuilds from an
        /// enclosing trace's loop-pass tracking.</summary>
        internal static TraceContext EnterIsolated() => TraceContext.Enter(isModuleBuild: false);

        /// <summary>Enters an isolated trace only if no trace is current (a standalone
        /// LoopAPI.Iterate outside any build); returns null — nothing to exit — when a
        /// trace is already in progress.</summary>
        internal static TraceContext? EnterIsolatedIfNone()
            => TraceContext.Current is null ? EnterIsolated() : null;

        /// <summary>Exits a trace entered by one of the Enter methods above.</summary>
        internal static void Exit(TraceContext trace) => TraceContext.Exit(trace);

        // ──────────────── recording (Rng.Pin, Globals.StateUpdate) ────────────────

        /// <summary>Records positional Rng pins. Requires a module build (throws in
        /// user-facing terms otherwise); silently skips non-canonical loop passes.</summary>
        internal static void RecordPins(object[] items)
        {
            var build = RequireModuleBuild("Rng.Pin");
            // A loop body is traced multiple times during graph construction; only the
            // canonical pass builds the surviving nodes. Recording outside it would pin
            // throwaway nodes.
            if (!build.Loopers.InCanonicalRecordingScope) return;
            build.Pins.AddPositional(items);
        }

        /// <summary>Records sparse (slot-addressed) Rng pins; same contract as
        /// <see cref="RecordPins"/>.</summary>
        internal static void RecordSparsePins((int[] path, object item)[] items)
        {
            var build = RequireModuleBuild("Rng.Pin");
            if (!build.Loopers.InCanonicalRecordingScope) return;
            build.Pins.AddSparse(items);
        }

        /// <summary>
        /// Throws unless a state update may be recorded right now: requires a module build,
        /// and rejects call sites inside a LoopAPI.Iterate body — a loop body is traced once
        /// per construction pass (up to four times), so an in-loop registration would fire
        /// repeatedly, mostly against throwaway nodes, and what a per-iteration state update
        /// should even mean is undefined.
        /// </summary>
        internal static void EnsureStateUpdateRecordable()
        {
            var build = RequireModuleBuild("Globals.StateUpdate");
            if (build.Loopers.InLoopBody)
                throw new InvalidOperationException(
                    "Globals.StateUpdate is not supported inside a LoopAPI.Iterate body. " +
                    "Compute the new value inside the loop, then register the update once " +
                    "after the loop, from the loop's final value.");
        }

        /// <summary>Records one state-update pair; <paramref name="linkedUpdated"/> is the
        /// STATE_UPDATE_LINK output wrapping the user's updated value.</summary>
        internal static void RecordStateUpdate(Variable original, Variable linkedUpdated)
            => RequireModuleBuild("Globals.StateUpdate").StateUpdates.Add(original, linkedUpdated);

        // ───────────────── harvest (graph builder, at build exit) ─────────────────

        /// <summary>Harvests (and clears) the linked updated-state tensors registered
        /// during <paramref name="build"/>, in registration order.</summary>
        internal static Variable[] HarvestStateUpdates(TraceContext build)
        {
            Debug.Assert(build.IsModuleBuild, "Harvesting a trace that is not a module build.");
            return build.StateUpdates.Take();
        }

        /// <summary>Harvests (and clears) the pins recorded during <paramref name="build"/>.</summary>
        internal static (object[] positional, (int[] path, object item)[] sparse) HarvestPins(TraceContext build)
        {
            Debug.Assert(build.IsModuleBuild, "Harvesting a trace that is not a module build.");
            return build.Pins.Take();
        }

        // ───────────────────────────── loop tracing ─────────────────────────────

        /// <summary>The current trace's loop-tracing state, or null when no trace is in
        /// progress on the current thread.</summary>
        internal static LooperStack? CurrentLoopers => TraceContext.Current?.Loopers;

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
    /// Only <see cref="GraphTrace"/> touches this class; every other consumer holds
    /// instances purely as opaque handles between Enter and Exit. The scoping mechanics
    /// live in <see cref="AmbientScope{TScope}"/>; each registry's behavior lives in its
    /// feature's file.
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
