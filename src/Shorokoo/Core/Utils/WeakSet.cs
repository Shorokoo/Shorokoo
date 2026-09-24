using System.Collections.Generic;
using System.Runtime.CompilerServices;

namespace Shorokoo.Core.Utils;

/// <summary>
/// A set of objects held weakly and compared by reference — the shape every one of the layering
/// registries needs: a memory device's backends, a backend's contexts, a context's tensors.
///
/// <para>Weak because a registry must not be the reason something stays alive. A backend the
/// program has dropped, a context it has let go of and a tensor it no longer names are all
/// garbage the moment nothing else holds them, and a list that kept them would turn every
/// registration into a leak. This is the discipline <c>ComputeContext</c>'s attached tensors and
/// compiled graphs already follow.</para>
///
/// <para>By reference because two backends are the same backend exactly when they are the same
/// object, and two tensors the same tensor exactly when they are one allocation.
/// <see cref="ConditionalWeakTable{TKey,TValue}"/> gives both properties at once: its keys are
/// weak and its comparison is identity, with no comparer to pass and no way to override it.</para>
/// </summary>
internal sealed class WeakSet<T>
    where T : class
{
    private readonly ConditionalWeakTable<T, object> _entries = new();

    private static readonly object Present = new();

    /// <summary>
    /// Records <paramref name="item"/>. Idempotent — and cheap when it is already recorded, which
    /// is the common case for a context's list: every run re-attaches what it reads, and the lookup
    /// takes no lock where an add would.
    /// </summary>
    internal void Add(T item)
    {
        if (!_entries.TryGetValue(item, out _)) _entries.TryAdd(item, Present);
    }

    /// <summary>Forgets <paramref name="item"/>, if it was ever recorded.</summary>
    internal void Remove(T item) => _entries.Remove(item);

    /// <summary>Whether <paramref name="item"/> is recorded.</summary>
    internal bool Contains(T item) => _entries.TryGetValue(item, out _);

    /// <summary>Forgets everything recorded.</summary>
    internal void Clear() => _entries.Clear();

    /// <summary>
    /// The members still alive, as a list of the caller's own. A snapshot rather than a live view,
    /// because the underlying table's enumerator holds its lock for as long as it is open and a
    /// caller iterating one while constructing tensors would block every other thread doing the
    /// same.
    /// </summary>
    internal IReadOnlyList<T> Snapshot()
    {
        List<T> live = [];
        foreach (var (item, _) in _entries) live.Add(item);
        return live;
    }
}
