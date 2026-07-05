using System;
using System.Collections.Generic;
using System.Linq;

namespace Shorokoo;

/// <summary>
/// User-facing RNG helpers for module bodies.
///
/// <para><see cref="Pin(object[])"/> freezes the RNG stream identities of a module's random
/// consumers against code refactoring. It records the listed items (model objects from
/// <c>X.Model(...)</c>, trainable-param tensors from initializer <c>Init(...)</c> calls,
/// runtime feed tensors from <c>Globals.Random*</c>) into a trace-time static list that the
/// module compiler reads when assigning ModelId slots: <b>listed items take the id slots in
/// list order; unlisted items follow in node order</b>. Since RNG stream keys fold along
/// ModelIds, a pinned item's streams no longer depend on its creation position — reorder or
/// move code freely, only the Pin list defines the mapping.</para>
///
/// <para><b>A pin reshapes only its own scope.</b> A <c>Rng.Pin(...)</c> written at the end of
/// <c>Inline</c> reshapes the module-level slots; a <c>Rng.Pin(...)</c> written at the end of a
/// <c>LoopAPI.Iterate(...)</c> loop body reshapes that loop's local slots (its consumers are
/// scoped to the loop and are named from inside it) without disturbing the loop's own slot. So
/// to freeze a module with loops, place one pin per scope — the codegen <c>Rng.Pin</c>
/// suggestion (an Info diagnostic) writes them for you, at any nesting depth. A loop body is
/// traced several times during construction; a pin records only in the canonical pass, so it
/// resolves to the surviving nodes exactly once.</para>
///
/// <para>To freeze a scope's current streams, list ALL of its random consumers in creation
/// order — or use the sparse form <see cref="Pin(System.ValueTuple{int[], object}[])"/>, which
/// pins items to explicit local slots and leaves unlisted items at theirs.</para>
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
    /// Pins the RNG stream identities of the listed items to their list positions within the
    /// current scope (first item = first local slot, ...). Call once per scope, at the end of
    /// the module's Inline body or at the end of a loop body (to pin that loop's consumers).
    /// Unlisted consumers follow in node order, so a PARTIAL positional pin re-keys them; to
    /// pin some items while leaving the rest untouched, use the sparse form instead.
    /// </summary>
    public static void Pin(params object[] items)
    {
        ArgumentNullException.ThrowIfNull(items);
        // A loop body is traced multiple times during graph construction; only the canonical
        // pass builds the surviving nodes. Recording outside it would pin throwaway nodes.
        if (!LoopAPI.InCanonicalRecordingScope) return;
        (_pins ??= new List<object>()).AddRange(items);
    }

    /// <summary>
    /// Sparse pin: each item takes exactly the local id slot named by its path within the
    /// current scope (1-based; e.g. <c>Rng.Pin(([3], proj))</c> pins <c>proj</c> to slot 3 of
    /// the scope it is written in — the module body, or the enclosing loop body). Unlisted
    /// consumers fill the remaining slots in node order — so unlike the positional form, a
    /// partial sparse pin leaves every unlisted consumer's slot (hence stream) unchanged. This
    /// is the form bind-time stream reports emit skeletons for. The path is a single local
    /// slot; the scope is given by where the pin is written, not by the path.
    /// </summary>
    public static void Pin(params (int[] path, object item)[] items)
    {
        ArgumentNullException.ThrowIfNull(items);
        foreach (var (path, item) in items)
        {
            if (path is null || path.Length == 0)
                throw new ArgumentException(
                    "Rng.Pin (sparse): each pin needs a non-empty 1-based slot, e.g. ([3], item).",
                    nameof(items));
            if (path.Length > 1)
                throw new NotSupportedException(
                    "Rng.Pin (sparse): a pin's path is a single local slot in the scope it is " +
                    $"written in; got [{string.Join(", ", path)}]. To pin a loop's consumer, write " +
                    "the pin inside that loop body (with a length-1 local slot), not a multi-level path.");
            if (path[0] < 1)
                throw new ArgumentException(
                    $"Rng.Pin (sparse): id slots are 1-based; got {path[0]}.", nameof(items));
            if (item is null)
                throw new ArgumentException("Rng.Pin (sparse): pinned item is null.", nameof(items));
        }
        if (!LoopAPI.InCanonicalRecordingScope) return;
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
