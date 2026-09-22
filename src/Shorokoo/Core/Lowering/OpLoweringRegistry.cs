using System.Collections.Immutable;
using System.Diagnostics.CodeAnalysis;
using System.Reflection;

namespace Shorokoo.Core.Lowering;

/// <summary>
/// The operators that have a lowering, keyed by op code.
///
/// <para>The scan covers <see cref="OpLowerings"/> and nothing else, deliberately — not the whole
/// assembly like <c>OpRegistry</c>'s. Which operators an engine may fall back to computing out of
/// primitives is a fact about the framework that should be readable in one place, and that class
/// is the place: nothing becomes lowerable by merely existing in the assembly.</para>
/// </summary>
internal static class OpLoweringRegistry
{
    private static readonly ImmutableDictionary<string, OpLowering> ByOpCode =
        typeof(OpLowerings).GetMethods(BindingFlags.Public | BindingFlags.Static)
            .Select(m => (Method: m, Attribute: m.GetCustomAttribute<OpLoweringAttribute>()))
            .Where(x => x.Attribute is not null)
            .ToImmutableDictionary(
                x => x.Attribute!.OpName,
                x => new OpLowering(x.Attribute!.OpName, x.Method),
                StringComparer.Ordinal);

    /// <summary>
    /// Thread-scoped lowerings installed by <see cref="Override"/>. Consulted ahead of the
    /// process-wide table so a caller can add or swap a decomposition without other threads
    /// observing it.
    /// </summary>
    [ThreadStatic]
    private static ImmutableDictionary<string, OpLowering>? overrides;

    /// <summary>
    /// Finds the lowering for <paramref name="opCode"/>, or returns false when the operator has
    /// none — which is the ordinary case and not an error: a domain that names an operator it
    /// cannot handle either has a lowering to fall back on or gives up on the node.
    /// </summary>
    public static bool TryGet(string opCode, [MaybeNullWhen(false)] out OpLowering lowering)
    {
        if (overrides is { } o && o.TryGetValue(opCode, out lowering)) return true;
        return ByOpCode.TryGetValue(opCode, out lowering);
    }

    /// <summary>
    /// Adds or replaces <paramref name="lowerings"/> on the calling thread only, until the
    /// returned scope is disposed. Overrides are invisible to other threads, so concurrent
    /// callers keep seeing the decompositions the framework ships.
    /// </summary>
    public static IDisposable Override(params OpLowering[] lowerings) => new OverrideScope(lowerings);

    private sealed class OverrideScope : IDisposable
    {
        private readonly ImmutableDictionary<string, OpLowering>? previous;

        internal OverrideScope(OpLowering[] lowerings)
        {
            this.previous = overrides;
            var next = this.previous ?? ImmutableDictionary.Create<string, OpLowering>(StringComparer.Ordinal);
            foreach (var lowering in lowerings) next = next.SetItem(lowering.OpCode, lowering);
            overrides = next;
        }

        public void Dispose() => overrides = this.previous;
    }
}
