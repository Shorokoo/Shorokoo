using System;

namespace Shorokoo.Core
{
    /// <summary>
    /// The mechanics of a thread-affine, re-entrant ambient scope — and nothing else. This
    /// class knows nothing about what a scope carries or what entering one means; derived
    /// types add the payload.
    ///
    /// <para><b>The problem it solves.</b> Code deep inside a call tree needs to find "the
    /// current X" without threading it through every signature; the same operation can start
    /// again before the previous one finishes (re-entrancy); and concurrent threads must not
    /// see each other's X. The classic failure modes are a destructive reset on entry (which
    /// wipes the outer operation's state) and state leaking across threads or between
    /// operations.</para>
    ///
    /// <para><b>The shape of the solution.</b> One <see cref="ThreadStaticAttribute"/> slot
    /// per scope type holds the innermost scope of the current thread; each scope remembers
    /// the scope it displaced, so nested scopes form a chain. <see cref="EnterScope"/> pushes
    /// onto the chain and <see cref="Exit"/> pops it, verifying strict bracketing: exiting
    /// anything other than the innermost scope on the entering thread throws. Callers pair
    /// the two in try/finally.</para>
    /// </summary>
    /// <typeparam name="TScope">The concrete scope type (curiously recurring), which gives
    /// each scope type its own thread-static slot.</typeparam>
    internal abstract class AmbientScope<TScope> where TScope : AmbientScope<TScope>
    {
        [ThreadStatic]
        private static TScope? _current;

        private TScope? _enclosing;

        /// <summary>The innermost scope on the current thread, or null.</summary>
        internal static TScope? Current => _current;

        /// <summary>Makes <paramref name="scope"/> the innermost scope on the current
        /// thread, remembering the scope it displaces. Pair with <see cref="Exit"/> in a
        /// try/finally.</summary>
        protected static TScope EnterScope(TScope scope)
        {
            scope._enclosing = _current;
            _current = scope;
            return scope;
        }

        /// <summary>
        /// Exits <paramref name="scope"/>, restoring the scope it displaced. Must be called
        /// on the entering thread with the innermost scope — anything else means an
        /// enter/exit bracket was broken, and throws.
        /// </summary>
        internal static void Exit(TScope scope)
        {
            if (!ReferenceEquals(_current, scope))
                throw new InvalidOperationException(
                    $"{typeof(TScope).Name}: attempted to exit a scope that is not the " +
                    "innermost one on the current thread — an enter/exit bracket was broken " +
                    "(mismatched nesting or a cross-thread exit).");
            scope.OnExiting();
            _current = scope._enclosing;
        }

        /// <summary>Called just before this scope is exited; derived types verify their own
        /// invariants here.</summary>
        protected virtual void OnExiting() { }
    }
}
