using System;
using System.Collections.Generic;

namespace Shorokoo;

/// <summary>
/// User-facing RNG helpers for module bodies.
///
/// <para><see cref="Pin(object[])"/> freezes the RNG stream identities of a module's random
/// consumers against code refactoring. Called at the end of a module's <c>Inline</c> body, it
/// records the listed items (model objects from <c>X.Model(...)</c>, trainable-param tensors
/// from initializer <c>Init(...)</c> calls) into a trace-time static list that the module
/// compiler reads when assigning ModelId slots: <b>listed items take the module-local id slots
/// in list order; unlisted items follow in node order</b>. Since RNG stream keys fold along
/// ModelIds, a pinned item's streams no longer depend on its creation position — reorder or
/// move code freely, only the Pin list defines the mapping. To freeze a module's current
/// streams before refactoring, list ALL its random consumers in current creation order.</para>
///
/// <para>Nothing here is baked into the computation graph: the pin list only reshapes how the
/// compiler numbers the graph it was already building.</para>
/// </summary>
public static class Rng
{
    [ThreadStatic]
    private static List<object>? _pins;

    /// <summary>
    /// Pins the RNG stream identities of the listed items to their list positions (first item
    /// = first module-local id slot, ...). Call once, at the end of the module's Inline body.
    /// </summary>
    public static void Pin(params object[] items)
    {
        ArgumentNullException.ThrowIfNull(items);
        (_pins ??= new List<object>()).AddRange(items);
    }

    /// <summary>Collects and clears the pins recorded during the current body trace (module-compiler side).</summary>
    internal static object[] GetAndClearPins()
    {
        var pins = _pins;
        _pins = null;
        return pins?.ToArray() ?? Array.Empty<object>();
    }
}
