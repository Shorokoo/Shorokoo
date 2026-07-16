using System;

namespace Shorokoo.Core
{
    /// <summary>
    /// The trace of one module build: a <see cref="TraceContext"/> plus the registries the
    /// graph builder harvests when the body returns — <see cref="StateUpdates"/>
    /// (Globals.StateUpdate) and <see cref="Pins"/> (Rng.Pin). This class contains no logic:
    /// the scoping mechanics live in <see cref="AmbientScope{TScope}"/>, and each registry's
    /// behavior lives in its feature's file.
    ///
    /// <para>The graph builder enters one per body trace; because entering stacks onto the
    /// enclosing trace (see <see cref="AmbientScope{TScope}"/>), the builder may re-enter
    /// mid-trace — a body first-using a sub-module or initializer whose <c>Function</c> is
    /// not yet cached — without disturbing the outer build's records. Recorders whose
    /// records only the builder's harvest can apply — <c>Rng.Pin</c>,
    /// <c>Globals.StateUpdate</c> — obtain the current instance via
    /// <see cref="RequireModuleBuild"/>, which throws (in user-facing, module-body terms)
    /// when the current trace is not a module build.</para>
    /// </summary>
    internal sealed class ModuleBuildContext : TraceContext
    {
        /// <summary>The Globals.StateUpdate registrations of this build.</summary>
        internal StateUpdateRegistry StateUpdates { get; } = new StateUpdateRegistry();

        /// <summary>The Rng.Pin recordings of this build.</summary>
        internal RngPinRegistry Pins { get; } = new RngPinRegistry();

        /// <summary>Enters the context for one graph-builder body trace.</summary>
        internal static ModuleBuildContext Enter()
            => (ModuleBuildContext)EnterScope(new ModuleBuildContext());

        /// <summary>
        /// The current module-build context, for recorders whose records only a module
        /// build can harvest. Throws when the current trace is not one: without a
        /// harvesting build, a record would sit in an ambient list forever — neither
        /// applied nor rejected.
        /// </summary>
        /// <param name="api">The user-facing API name, for the error message.</param>
        internal static ModuleBuildContext RequireModuleBuild(string api)
        {
            var context = Current;
            if (context is null)
                throw new InvalidOperationException(
                    $"{api} may only be called inside a module body — a [Module] Inline " +
                    "method or a codegen-free delegate body — and no module body is " +
                    "executing on the current thread. If the call is written inside a " +
                    "module body, make sure the body does not switch threads: a body runs " +
                    "synchronously on a single thread, and code on async continuations, " +
                    "Parallel.For workers, or other callbacks runs outside the body.");
            if (context is not ModuleBuildContext build)
                throw new InvalidOperationException(
                    $"{api} may only be called inside a module body — a [Module] Inline " +
                    "method or a codegen-free delegate body. This call is inside a " +
                    "standalone LoopAPI.Iterate loop with no enclosing module body, so " +
                    "there is no module it could apply to.");
            return build;
        }
    }
}
