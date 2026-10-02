namespace Shorokoo.Core.Backends;

/// <summary>
/// A block of a backend's memory that several tensor values stand on, each over a byte range of its
/// own that no other value's range overlaps: the memory of an input a run consumed, with the run's
/// outputs placed in it. Each value standing on it holds a <b>lease</b>; the block is let go of —
/// <see cref="Release"/>'s action runs — when the last lease is released, and not before, so a value
/// standing on it never outlives its memory, and deleting one value of several frees nothing until
/// the others are gone too.
///
/// <para>A block is counted whole by a device-memory budget, once, for as long as a tensor attached
/// to the context stands on it: its memory is held for that long, whichever of its ranges are still
/// in use.</para>
/// </summary>
internal sealed class SharedBlock
{
    private readonly Action _letGo;
    private int _leases;
    private int _released;

    /// <summary>
    /// A block of <paramref name="bytes"/> bytes with no lease yet, which <paramref name="letGo"/>
    /// lets go of once the last lease taken on it is released. A block on which no lease is ever
    /// taken is never let go of here: whoever made it still owns its memory.
    /// </summary>
    internal SharedBlock(long bytes, Action letGo)
    {
        Bytes = bytes;
        _letGo = letGo;
    }

    /// <summary>The size of the block in bytes: what a budget counts while any tensor on it is
    /// attached.</summary>
    internal long Bytes { get; }

    /// <summary>How many leases are held on the block now.</summary>
    internal int Leases => Volatile.Read(ref _leases);

    /// <summary>Whether the block has been let go of.</summary>
    internal bool IsReleased => Volatile.Read(ref _released) != 0;

    /// <summary>Takes a lease on the block for a value standing on it.</summary>
    /// <exception cref="ObjectDisposedException">The block was let go of already.</exception>
    internal void Lease()
    {
        if (IsReleased) throw new ObjectDisposedException(nameof(SharedBlock), "The block was let go of when its last lease was released.");
        Interlocked.Increment(ref _leases);
    }

    /// <summary>Releases a lease, letting the block go when it was the last one.</summary>
    internal void Release()
    {
        if (Interlocked.Decrement(ref _leases) != 0) return;
        if (Interlocked.Exchange(ref _released, 1) != 0) return;
        _letGo();
    }
}

/// <summary>Where a value stands on a <see cref="SharedBlock"/>: <see cref="Bytes"/> bytes from
/// byte <see cref="Offset"/> of <see cref="Block"/>, which the value holds a lease on.</summary>
internal readonly record struct BlockRange(SharedBlock Block, long Offset, long Bytes);
