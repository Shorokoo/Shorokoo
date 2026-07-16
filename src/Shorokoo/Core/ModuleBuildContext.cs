using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace Shorokoo.Core
{
    /// <summary>
    /// The single ambient container for the state a module-body trace accumulates: the
    /// <see cref="LoopAPI"/> looper stack, the <see cref="Globals.StateUpdate{T}(T, T)"/>
    /// registrations, and the <see cref="Shorokoo.Rng.Pin(object[])"/> recordings. Exactly one
    /// thread-static slot holds "the current context, or none"; the graph builder enters a
    /// fresh context per body trace and restores the enclosing one on exit.
    ///
    /// <para><b>Re-entrancy.</b> The graph builder re-enters mid-trace whenever a body
    /// first-uses a sub-module or initializer whose <c>Function</c> is not yet cached. The
    /// save/restore discipline for that re-entrancy lives here, once — <see cref="Exit"/>
    /// restores the enclosing context — instead of one hand-coordinated push/pop pair per
    /// ambient registry (the arrangement this class replaced).</para>
    ///
    /// <para><b>Kinds.</b> A <em>module build</em> context (<see cref="EnterModuleBuild"/>) is
    /// entered by the graph builder, which harvests the recorded state at build exit
    /// (<see cref="TakeStateUpdates"/> / <see cref="TakePins"/>). Recorders whose records only
    /// that harvest can apply — <c>Rng.Pin</c>, <c>Globals.StateUpdate</c> — demand one via
    /// <see cref="RequireModuleBuild"/> and throw otherwise, instead of recording into a list
    /// nobody will ever read. An <em>isolated</em> context (<see cref="EnterIsolated"/>) carries
    /// the same state but is never harvested: it backs standalone <c>LoopAPI.Iterate</c> traces
    /// (hand-built graphs) and shields internal node rebuilds from an enclosing trace.</para>
    ///
    /// <para><b>Thread affinity.</b> The ambient slot is <see cref="ThreadStaticAttribute"/>, so
    /// a context is visible only on the thread that entered it. A module body that hops threads
    /// (async continuations, <c>Parallel.For</c>, callbacks run elsewhere) finds no context on
    /// the other thread, so <see cref="RequireModuleBuild"/> fails loudly at the call site
    /// rather than recording into the wrong thread's build.</para>
    /// </summary>
    internal sealed class ModuleBuildContext
    {
        [ThreadStatic]
        private static ModuleBuildContext? _current;

        private readonly ModuleBuildContext? _enclosing;

        private ModuleBuildContext(bool isModuleBuild, ModuleBuildContext? enclosing)
        {
            IsModuleBuild = isModuleBuild;
            _enclosing = enclosing;
        }

        /// <summary>The context of the trace in progress on the current thread, or null.</summary>
        internal static ModuleBuildContext? Current => _current;

        /// <summary>Whether this context is a graph-builder body trace, whose entry point
        /// harvests the recorded state at build exit (as opposed to an isolated trace).</summary>
        internal bool IsModuleBuild { get; }

        /// <summary>Enters the context for one graph-builder body trace.</summary>
        internal static ModuleBuildContext EnterModuleBuild() => Enter(isModuleBuild: true);

        /// <summary>Enters a context that carries trace state but is never harvested: standalone
        /// <c>LoopAPI.Iterate</c> traces, and internal node rebuilds that must not leak into an
        /// enclosing trace's loop-pass tracking.</summary>
        internal static ModuleBuildContext EnterIsolated() => Enter(isModuleBuild: false);

        private static ModuleBuildContext Enter(bool isModuleBuild)
        {
            var context = new ModuleBuildContext(isModuleBuild, _current);
            _current = context;
            return context;
        }

        /// <summary>
        /// Exits <paramref name="context"/>, restoring the enclosing context. Must be called on
        /// the entering thread with the innermost context — anything else means an enter/exit
        /// bracket was broken, and fails loudly.
        /// </summary>
        internal static void Exit(ModuleBuildContext context)
        {
            if (!ReferenceEquals(_current, context))
                throw new InvalidOperationException(
                    "ModuleBuildContext.Exit: the given context is not the innermost one on the " +
                    "current thread — an enter/exit bracket was broken (mismatched nesting or a " +
                    "cross-thread exit).");
            Debug.Assert(context.LooperStack.Count == 0,
                "ModuleBuildContext.Exit: a LoopAPI looper is still active — an Iterate trace " +
                "was left open across the context boundary.");
            _current = context._enclosing;
        }

        /// <summary>
        /// The current module-build context, for recorders whose records only a module build can
        /// harvest. This is the uniform "no build in progress → throw" contract shared by
        /// <c>Rng.Pin</c> and <c>Globals.StateUpdate</c>: without it a record would sit in an
        /// ambient list forever — neither applied nor rejected.
        /// </summary>
        /// <param name="api">The user-facing API name, for the error message.</param>
        internal static ModuleBuildContext RequireModuleBuild(string api)
        {
            var context = _current;
            if (context is null)
                throw new InvalidOperationException(
                    $"{api} requires a module build in progress, and none is active on the " +
                    "current thread. Call it inside a module body being traced — a [Module] " +
                    "Inline method or a codegen-free delegate body. (Builds are thread-affine: " +
                    "a body that hops threads — async continuations, Parallel.For, callbacks — " +
                    "cannot record into its build.)");
            if (!context.IsModuleBuild)
                throw new InvalidOperationException(
                    $"{api} requires a module build in progress, but the current trace is not " +
                    "one (e.g. a standalone LoopAPI.Iterate outside any module build). The " +
                    "record could never be applied to a module, so the call fails loudly " +
                    "instead of being silently orphaned.");
            return context;
        }

        // ───────────────────────── LoopAPI looper state ─────────────────────────

        /// <summary>The in-progress <see cref="Looper"/>s of this trace, outermost first.</summary>
        internal List<Looper> LooperStack { get; } = new();

        /// <summary>Whether the trace is currently inside a <c>LoopAPI.Iterate</c> body.</summary>
        internal bool InLoopBody => LooperStack.Count > 0;

        /// <summary>
        /// The looper whose pass currently drives node interception, and its stack index: the
        /// innermost looper that has started tracing (inner loops sit on the stack at pass 0
        /// while an outer loop is mid-pass). Null when no loop is active.
        /// </summary>
        internal (Looper looper, int index)? ActiveLooper
        {
            get
            {
                if (LooperStack.Count == 0) return null;
                var activeIndex = LooperStack.FindIndex(x => x.CurrentPass == 0) - 1;
                if (activeIndex < 0) activeIndex = LooperStack.Count - 1;
                return (LooperStack[activeIndex], activeIndex);
            }
        }

        /// <summary>
        /// Whether trace-time actions that must fire <b>exactly once per source occurrence</b>
        /// (e.g. <see cref="Shorokoo.Rng.Pin(object[])"/>) should record right now. A loop body
        /// is executed once per construction pass — first (track), second (identify loop vars),
        /// third (build the real body), fourth (expose outputs) — so a naive record inside a
        /// loop body would fire up to four times, three of them against throwaway nodes. Only
        /// the pass that builds the surviving body nodes (the active looper's third pass) is
        /// canonical; at module level (no active loop) recording is always canonical. This is
        /// THE canonical-pass query: the loop tracer publishes its pass state to the context it
        /// lives in and every recorder asks here, rather than each re-deriving the
        /// active-looper selection from <c>LoopAPI</c> internals.
        /// </summary>
        internal bool InCanonicalRecordingScope
            => ActiveLooper is not { } active || active.looper.CurrentPass == 3;

        // ─────────────────── Globals.StateUpdate registrations ───────────────────

        private List<(Variable original, Variable updated)>? _stateUpdatePairs;

        /// <summary>Records one state-update pair; <paramref name="linkedUpdated"/> is the
        /// STATE_UPDATE_LINK output wrapping the user's updated value.</summary>
        internal void AddStateUpdate(Variable original, Variable linkedUpdated)
            => (_stateUpdatePairs ??= new List<(Variable, Variable)>()).Add((original, linkedUpdated));

        /// <summary>Harvests (and clears) the linked updated-state tensors registered during
        /// this build, in registration order.</summary>
        internal Variable[] TakeStateUpdates()
        {
            if (_stateUpdatePairs is null || _stateUpdatePairs.Count == 0)
                return Array.Empty<Variable>();
            var updates = _stateUpdatePairs.Select(p => p.updated).ToArray();
            _stateUpdatePairs = null;
            return updates;
        }

        // ───────────────────────── Rng.Pin recordings ─────────────────────────

        private List<object>? _pins;
        private List<(int[] path, object item)>? _slotPins;

        /// <summary>Records positional pins (see <see cref="Shorokoo.Rng.Pin(object[])"/>).</summary>
        internal void AddPins(object[] items)
            => (_pins ??= new List<object>()).AddRange(items);

        /// <summary>Records sparse (slot-addressed) pins (see
        /// <see cref="Shorokoo.Rng.Pin(System.ValueTuple{int[], object}[])"/>).</summary>
        internal void AddSlotPins((int[] path, object item)[] items)
            => (_slotPins ??= new List<(int[], object)>()).AddRange(items);

        /// <summary>Harvests (and clears) the pins recorded during this build.</summary>
        internal (object[] positional, (int[] path, object item)[] sparse) TakePins()
        {
            var pins = _pins;
            var slotPins = _slotPins;
            _pins = null;
            _slotPins = null;
            return (pins?.ToArray() ?? Array.Empty<object>(),
                    slotPins?.ToArray() ?? Array.Empty<(int[], object)>());
        }
    }
}
