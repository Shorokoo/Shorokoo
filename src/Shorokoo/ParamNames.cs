using System;
using System.Collections.Generic;
using System.ComponentModel;
using Shorokoo.Core;

namespace Shorokoo;

/// <summary>
/// Names a module's trainable parameters and sub-models by what they are.
///
/// <para>A parameter's name is the path of names from the model down to it —
/// <c>TrainableParam#0.encoder#0.proj#0.weight#0</c> — and is the key it is saved under in
/// every checkpoint. Each part of the path names one parameter or sub-model inside its
/// module (and inside its loop, for one created in a <c>LoopAPI.Iterate</c> body). Where a
/// part's name comes from, most specific first:</para>
/// <list type="number">
/// <item><see cref="Named{T}"/>: <c>Normal02.Init([v, d]).Named("embedding")</c>.</item>
/// <item>The local variable the call is assigned to, when the whole initializer of a
/// local declared in a <c>[Module]</c> body is the <c>Init(...)</c> or <c>Model(...)</c>
/// call: <c>var proj = Linear.Model(...)</c> is named <c>proj</c>. The source generator
/// supplies these.</item>
/// <item>Otherwise the initializer's or module's class name — <c>Normal02</c>,
/// <c>Linear</c> — numbered <c>#0</c>, <c>#1</c>, ... in creation order among the
/// same-named parts of its scope.</item>
/// </list>
/// <para>A name survives moving the code that creates it, which a creation-order number
/// does not. Renaming it renames the parameter: a checkpoint holding the parameter under
/// another name is refused.</para>
/// </summary>
public static class ParamNames
{
    /// <summary>
    /// Names the parameter an <c>Init(...)</c> call created, or the sub-model a
    /// <c>Model(...)</c> call created, and returns it unchanged. Call it inside the module
    /// body that creates <paramref name="item"/>. Two items given the same name in one
    /// scope of a module fail the module build.
    /// </summary>
    public static T Named<T>(this T item, string name) where T : IModuleParam
    {
        ArgumentNullException.ThrowIfNull(item);
        if (string.IsNullOrWhiteSpace(name) || name != name.Trim())
            throw new ArgumentException(
                $"A parameter or sub-model name must be non-empty and neither start nor end with whitespace; got \"{name}\".",
                nameof(name));
        var names = GraphTrace.ParamNames("Named");
        if (GraphTrace.Loopers.InCanonicalRecordingScope)
            names.Add(item, name, inferred: false);
        return item;
    }

    /// <summary>
    /// The name of the local variable a call was assigned to, recorded by generated code.
    /// Not for direct use: <see cref="Named{T}"/> is the explicit form.
    /// </summary>
    [EditorBrowsable(EditorBrowsableState.Never)]
    public static T NamedByLocal<T>(T item, string name) where T : IModuleParam
    {
        ArgumentNullException.ThrowIfNull(item);
        if (GraphTrace.ParamNamesIfBuilding is { } names && GraphTrace.Loopers.InCanonicalRecordingScope)
            names.Add(item, name, inferred: true);
        return item;
    }
}

/// <summary>
/// The names recorded during one module-body trace, harvested by the graph builder after
/// the body returns, as <see cref="RngPinRegistry"/> is.
/// </summary>
internal sealed class ParamNameRegistry
{
    private List<(object item, string name, bool inferred)>? _names;

    internal void Add(object item, string name, bool inferred)
        => (_names ??= new List<(object, string, bool)>()).Add((item, name, inferred));

    /// <summary>Harvests (and clears) the recorded names.</summary>
    internal (object item, string name, bool inferred)[] Take()
    {
        var result = _names?.ToArray() ?? Array.Empty<(object, string, bool)>();
        _names = null;
        return result;
    }
}
