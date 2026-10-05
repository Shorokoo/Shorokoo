namespace Shorokoo.Core.Backends;

/// <summary>
/// A block of a backend's memory that several tensor values stand on, each over a byte range of its
/// own that no other value's range overlaps: the memory of an input a run consumed, with the run's
/// outputs placed in it. Each value standing on it holds a <b>lease</b> on its range; the block is
/// let go of — the action it was made with runs — when the last lease is released, and not before, so
/// a value standing on it never outlives its memory.
///
/// <para>A part of the block no lease covers is memory nothing reads any more: what lay between the
/// ranges the run's outputs were written into, and a range whose value has ended while others live.
/// Where the block was made with a way to hand such a part back, it goes back as no lease covers it
/// any more — as the run that placed its values ends (<see cref="Settle"/>), and as each lease is
/// released — so a value that ends frees what of the block it alone stood on, as a value in memory
/// of its own does. Only whole units of the memory's allocator go back: the 4 KiB pages of a block on
/// the host, the 2 MiB granules of one on a card — all but the block's first, by whose address the
/// allocator knows the block until it is let go of.</para>
///
/// <para>A block is counted by a device-memory budget once, for what of it is still held
/// (<see cref="HeldBytes"/>), for as long as a tensor attached to the context stands on it.</para>
///
/// <para><b>A value nothing references releases its lease as it is collected</b>
/// (<see cref="HeldLease"/>), as a tensor in memory of its own frees it then. While any lease is
/// held the block is kept reachable from here, and with it whatever its letting go releases: the
/// memory it stands for is let go of by the block alone, with its last lease, never by a finalizer of
/// its own that could free it before the leases still handing parts of it back.</para>
/// </summary>
internal sealed class SharedBlock
{
    private readonly Action _letGo;
    private readonly Func<long, long, bool, long>? _giveBack;
    private readonly List<(long Offset, long Bytes)> _leases = [];
    private bool _released;
    private long _givenBack;
    // Keeps the block reachable while a lease is held.
    private System.Runtime.InteropServices.GCHandle _leased;

    /// <summary>
    /// A block of <paramref name="bytes"/> bytes with no lease yet, which <paramref name="letGo"/>
    /// lets go of once the last lease taken on it is released, and <paramref name="giveBack"/> — where
    /// there is one — hands back a part of while others are still leased: the part's offset and
    /// length, whether it runs on to the end of the block's memory, answering the bytes it handed
    /// back. A block on which no lease is ever taken is never let go of here: whoever made it still
    /// owns its memory.
    /// </summary>
    internal SharedBlock(long bytes, Action letGo, Func<long, long, bool, long>? giveBack = null)
    {
        Bytes = bytes;
        _letGo = letGo;
        _giveBack = giveBack;
    }

    /// <summary>The size of the block in bytes.</summary>
    internal long Bytes { get; }

    /// <summary>Whether a part of the block no lease covers goes back while other parts are
    /// leased.</summary>
    internal bool GivesBack => _giveBack is not null;

    /// <summary>The bytes of the block still held: all of them but those handed back as no lease
    /// covered them any more. What a budget counts while any tensor on it is attached.</summary>
    internal long HeldBytes
    {
        get { lock (_leases) return Bytes - _givenBack; }
    }

    /// <summary>How many leases are held on the block now.</summary>
    internal int Leases
    {
        get { lock (_leases) return _leases.Count; }
    }

    /// <summary>Whether the block has been let go of.</summary>
    internal bool IsReleased
    {
        get { lock (_leases) return _released; }
    }

    /// <summary>Takes a lease on <paramref name="bytes"/> bytes of the block from byte
    /// <paramref name="offset"/>, for a value standing there.</summary>
    /// <exception cref="ObjectDisposedException">The block was let go of already.</exception>
    internal void Lease(long offset, long bytes)
    {
        lock (_leases)
        {
            if (_released) throw new ObjectDisposedException(nameof(SharedBlock), "The block was let go of when its last lease was released.");
            if (!_leased.IsAllocated) _leased = System.Runtime.InteropServices.GCHandle.Alloc(this);
            _leases.Add((offset, bytes));
        }
    }

    /// <summary>Releases a lease <see cref="Lease"/> took: the block is let go of where it was the
    /// last one, and otherwise what no lease covers any more is handed back.</summary>
    internal void Release(long offset, long bytes)
    {
        lock (_leases)
        {
            var at = _leases.IndexOf((offset, bytes));
            if (at < 0) return;
            _leases.RemoveAt(at);
            if (_leases.Count > 0)
            {
                GiveBackUnleased();
                return;
            }
            if (_released) return;
            _released = true;
            if (_leased.IsAllocated) _leased.Free();
        }
        _letGo();
    }

    /// <summary>Hands back what of the block no lease covers: once the run that placed values in it
    /// has ended, what lay between the ranges of the values it handed back.</summary>
    internal void Settle()
    {
        lock (_leases)
            if (!_released && _leases.Count > 0) GiveBackUnleased();
    }

    /// <summary>Hands back every stretch no lease covers, the last on to the end of the block's
    /// memory. Under the lock; a stretch handed back already hands back nothing more.</summary>
    private void GiveBackUnleased()
    {
        if (_giveBack is null) return;
        long at = 0;
        foreach (var (offset, bytes) in _leases.OrderBy(l => l.Offset))
        {
            if (offset > at) _givenBack += _giveBack(at, offset - at, false);
            at = Math.Max(at, offset + bytes);
        }
        _givenBack += _giveBack(at, Math.Max(0, Bytes - at), true);
    }
}

/// <summary>Where a value stands on a <see cref="SharedBlock"/>: <see cref="Bytes"/> bytes from
/// byte <see cref="Offset"/> of <see cref="Block"/>, which the value holds a lease on.</summary>
internal readonly record struct BlockRange(SharedBlock Block, long Offset, long Bytes)
{
    /// <summary>Takes the lease a value standing here holds.</summary>
    internal void Lease() => Block.Lease(Offset, Bytes);

    /// <summary>Releases the lease <see cref="Lease"/> took.</summary>
    internal void Release() => Block.Release(Offset, Bytes);
}

/// <summary>
/// The lease a value standing on a <see cref="SharedBlock"/> holds on its range, taken already,
/// released once: by the value as it is released, or — where nothing references the value any more
/// and it is collected without — as this is finalized, so the range goes back, and the block with
/// its last lease, as a tensor in memory of its own is freed then.
/// </summary>
internal sealed class HeldLease(BlockRange range)
{
    private int _released;

    /// <summary>Where the value stands.</summary>
    internal BlockRange Range => range;

    /// <summary>Releases the lease, the first time only.</summary>
    internal void Release()
    {
        if (Interlocked.Exchange(ref _released, 1) != 0) return;
        GC.SuppressFinalize(this);
        range.Release();
    }

    ~HeldLease()
    {
        if (Interlocked.Exchange(ref _released, 1) != 0) return;
        // The block is still reachable — it keeps itself so while leased — and so is what its
        // letting go releases. Nothing may escape a finalizer: a failure to hand a range back leaves
        // it held, as a value never collected would.
        try { range.Release(); }
        catch (Exception) { }
    }
}
