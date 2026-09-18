using System.Collections.Immutable;
using System.Diagnostics.CodeAnalysis;

namespace Shorokoo.Core.Lowering;

/// <summary>
/// The operators that have an <see cref="OpLowering"/>, keyed by op code.
///
/// <para>Registration is this explicit table, deliberately — not a reflection scan like
/// <c>OpRegistry</c>'s. Which operators an engine may fall back to computing out of primitives
/// is a fact about the framework that should be readable in one place, and a table is that
/// place: nothing becomes lowerable by merely existing in the assembly.</para>
/// </summary>
internal static class OpLoweringRegistry
{
    private static readonly OpLowering[] Table = [new SoftsignLowering()];

    private static readonly ImmutableDictionary<string, OpLowering> ByOpCode =
        Table.ToImmutableDictionary(x => x.OpCode, StringComparer.Ordinal);

    /// <summary>
    /// Finds the lowering for <paramref name="opCode"/>, or returns false when the operator has
    /// none — which is the ordinary case and not an error: an engine that cannot compute the
    /// operator directly either has a lowering to fall back on or gives up on the node.
    /// </summary>
    public static bool TryGet(string opCode, [MaybeNullWhen(false)] out OpLowering lowering)
        => ByOpCode.TryGetValue(opCode, out lowering);
}
