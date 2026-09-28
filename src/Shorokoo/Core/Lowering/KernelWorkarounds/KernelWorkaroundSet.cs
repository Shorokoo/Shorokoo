using System.Collections.Immutable;

namespace Shorokoo.Core.Lowering.KernelWorkarounds;

/// <summary>
/// An ordered, immutable list of <see cref="KernelWorkaround"/>s applied together to the model
/// built for one backend's sessions. The pass applies them in order, one walk over the graph
/// each, so a later workaround sees what an earlier one spliced in.
/// </summary>
internal sealed class KernelWorkaroundSet
{
    /// <summary>The set that rewrites nothing.</summary>
    public static KernelWorkaroundSet Empty { get; } = new("none", []);

    public KernelWorkaroundSet(string name, ImmutableArray<KernelWorkaround> workarounds)
    {
        Name = name;
        Workarounds = workarounds;
        OpCodes = workarounds.SelectMany(w => w.OpCodes).ToImmutableHashSet(StringComparer.Ordinal);
    }

    /// <summary>The set's name: one of <see cref="Shorokoo.Core.Backends.KernelWorkaroundSets"/>
    /// for a registered set.</summary>
    public string Name { get; }

    /// <summary>The workarounds, in the order they are applied.</summary>
    public ImmutableArray<KernelWorkaround> Workarounds { get; }

    /// <summary>Every op code some workaround of the set looks at.</summary>
    public IReadOnlySet<string> OpCodes { get; }

    /// <summary>Whether the set holds no workaround.</summary>
    public bool IsEmpty => Workarounds.IsEmpty;

    public override string ToString() => Name;
}
