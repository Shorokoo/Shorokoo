using System;
using System.Collections.Generic;
using System.Linq;

namespace Shorokoo;

/// <summary>
/// User-facing RNG helpers for module bodies.
///
/// <para><see cref="Pin(object[])"/> freezes the RNG stream identities of a module's random
/// consumers against code refactoring. Called at the end of a module's <c>Inline</c> body, it
/// records the listed items (model objects from <c>X.Model(...)</c>, trainable-param tensors
/// from initializer <c>Init(...)</c> calls, runtime feed tensors from <c>Globals.Random*</c>)
/// into a trace-time static list that the module compiler reads when assigning ModelId slots:
/// <b>listed items take the module-local id slots in list order; unlisted items follow in node
/// order</b>. Since RNG stream keys fold along ModelIds, a pinned item's streams no longer
/// depend on its creation position — reorder or move code freely, only the Pin list defines
/// the mapping. To freeze a module's current streams before refactoring, list ALL its random
/// consumers in current creation order — or use the sparse form
/// <see cref="Pin(System.ValueTuple{int[], object}[])"/>, which pins items to explicit slots
/// and leaves unlisted items at theirs.</para>
///
/// <para>Nothing here is baked into the computation graph: the pin list only reshapes how the
/// compiler numbers the graph it was already building. A pin that cannot be resolved to a
/// node of the module's graph fails the module build — an inactive pin the author believes
/// is active is exactly the silent re-keying pinning exists to prevent.</para>
/// </summary>
public static class Rng
{
    [ThreadStatic]
    private static List<object>? _pins;

    [ThreadStatic]
    private static List<(int[] path, object item)>? _slotPins;

    /// <summary>
    /// Pins the RNG stream identities of the listed items to their list positions (first item
    /// = first module-local id slot, ...). Call once, at the end of the module's Inline body.
    /// Unlisted consumers follow in node order, so a PARTIAL positional pin re-keys them; to
    /// pin some items while leaving the rest untouched, use the sparse form instead.
    /// </summary>
    public static void Pin(params object[] items)
    {
        ArgumentNullException.ThrowIfNull(items);
        (_pins ??= new List<object>()).AddRange(items);
    }

    /// <summary>
    /// Sparse pin: each item takes exactly the module-local id slot named by its path
    /// (1-based; e.g. <c>Rng.Pin(([3], proj))</c> pins <c>proj</c> to slot 3). Unlisted
    /// consumers fill the remaining slots in node order — so unlike the positional form, a
    /// partial sparse pin leaves every unlisted consumer's slot (hence stream) unchanged.
    /// This is the form bind-time stream reports emit skeletons for. Currently top-level
    /// slots only (path length 1); nested/loop paths are a specified follow-up.
    /// </summary>
    public static void Pin(params (int[] path, object item)[] items)
    {
        ArgumentNullException.ThrowIfNull(items);
        foreach (var (path, item) in items)
        {
            if (path is null || path.Length == 0)
                throw new ArgumentException(
                    "Rng.Pin (sparse): each pin needs a non-empty 1-based id path, e.g. ([3], item).",
                    nameof(items));
            if (path.Length > 1)
                throw new NotSupportedException(
                    "Rng.Pin (sparse): only top-level slots (path of length 1) are supported for " +
                    $"now; got [{string.Join(", ", path)}]. Nested and loop-body paths are a " +
                    "specified follow-up.");
            if (path[0] < 1)
                throw new ArgumentException(
                    $"Rng.Pin (sparse): id slots are 1-based; got {path[0]}.", nameof(items));
            if (item is null)
                throw new ArgumentException("Rng.Pin (sparse): pinned item is null.", nameof(items));
        }
        (_slotPins ??= new List<(int[], object)>()).AddRange(items);
    }

    /// <summary>Collects and clears the pins recorded during the current body trace (module-compiler side).</summary>
    internal static (object[] positional, (int[] path, object item)[] sparse) GetAndClearPins()
    {
        var pins = _pins;
        var slotPins = _slotPins;
        _pins = null;
        _slotPins = null;
        return (pins?.ToArray() ?? Array.Empty<object>(),
                slotPins?.ToArray() ?? Array.Empty<(int[], object)>());
    }
}
