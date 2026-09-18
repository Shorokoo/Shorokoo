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
    /// Finds the lowering for <paramref name="opCode"/>, or returns false when the operator has
    /// none — which is the ordinary case and not an error: an engine that cannot compute the
    /// operator directly either has a lowering to fall back on or gives up on the node.
    /// </summary>
    public static bool TryGet(string opCode, [MaybeNullWhen(false)] out OpLowering lowering)
        => ByOpCode.TryGetValue(opCode, out lowering);
}
