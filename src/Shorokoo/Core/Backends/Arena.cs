using System.Runtime.InteropServices;

namespace Shorokoo.Core.Backends;

/// <summary>
/// Address space carved into blocks for <see cref="CachingAllocator"/>, and backed by memory a
/// granule at a time: the address space is reserved in chunks with no memory behind it, a granule is
/// committed under a block as the block is carved over it (<see cref="ArenaBacking.Commit"/>), and
/// stays committed when the block goes back — so the next block carved there, of whatever size,
/// finds its memory warm — until it is shed (<see cref="Decommit"/>), which hands it back to the
/// system with no block over it.
///
/// <para>A block is whole units, carved best fit from the free stretches of a chunk, which are
/// coalesced as blocks go back. Where the arena's unit is its granule, as in an account's arena on a
/// card, no two blocks share a granule: a block kept alive holds its own granules and no other
/// memory of the arena's. Where the unit is smaller — the host's 4 KiB pages in its 64 KiB
/// granules, a card's arena of small blocks — blocks share the granules their ends lie in, and a
/// granule is shed only once no block is over it.</para>
///
/// <para>Every member runs under the lock of the allocator that owns the arena.</para>
/// </summary>
internal sealed class Arena
{
    private readonly ArenaBacking _backing;
    private readonly long _unit;
    private readonly long _granule;
    private readonly long _unitsPerGranule;
    private readonly long _chunkBytes;

    /// <summary>An arena of <paramref name="unit"/>-byte units over <paramref name="backing"/>'s
    /// granules, reserving address space <paramref name="chunkBytes"/> at a time at least.</summary>
    internal Arena(ArenaBacking backing, long unit, long chunkBytes)
    {
        _backing = backing;
        _unit = unit;
        _granule = backing.Granule;
        _unitsPerGranule = _granule / unit;
        _chunkBytes = chunkBytes;
    }

    /// <summary>One reservation of address space and what is carved and committed in it.</summary>
    private sealed class Chunk
    {
        internal int Index;
        internal IntPtr Base;
        internal long Bytes;
        internal long Units;
        internal object? State;
        internal long CarvedUnits;
        // The free stretches, by their first unit and by the unit past their last.
        internal readonly Dictionary<long, long> FreeByStart = [];
        internal readonly Dictionary<long, long> FreeByEnd = [];
        // Per granule: whether it is committed, how many carved units are over it, and its place in
        // the arena's list of idle granules.
        internal bool[] Committed = [];
        internal long[] Live = [];
        internal IdleLink[] Idling = [];
    }

    /// <summary>A granule's neighbours in the list of idle granules, each a chunk and a granule of
    /// it; a chunk of -1 for none.</summary>
    private struct IdleLink
    {
        internal bool Listed;
        internal int PreviousChunk;
        internal long PreviousGranule;
        internal int NextChunk;
        internal long NextGranule;
    }

    private readonly List<Chunk?> _chunks = [];

    // Every free stretch, smallest first, for best fit: a sorted list searched by halving, which finds
    // and changes a stretch without allocating.
    private readonly List<(long Units, int Chunk, long Start)> _free = [];

    // Every committed granule no carved unit is over, linked in the order they went idle, so one joins
    // at the end and leaves from anywhere at once: the first is the one idle longest.
    private (int Chunk, long Granule) _idleFirst = (-1, 0);
    private (int Chunk, long Granule) _idleLast = (-1, 0);
    private long _idleCount;

    /// <summary>The bytes of the granules committed now.</summary>
    internal long CommittedBytes { get; private set; }

    /// <summary>The bytes of the committed granules no block is over.</summary>
    internal long IdleBytes => _idleCount * _granule;

    /// <summary>The most bytes of committed granules blocks have been over at once: the arena's
    /// busiest moment.</summary>
    internal long MaxBusyBytes { get; private set; }

    private void NoteBusy()
    {
        var busy = CommittedBytes - IdleBytes;
        if (busy > MaxBusyBytes) MaxBusyBytes = busy;
    }

    /// <summary>The bytes carved into blocks now.</summary>
    internal long CarvedBytes { get; private set; }

    /// <summary>The bytes of address space the arena holds reserved now.</summary>
    internal long ReservedBytes { get; private set; }

    /// <summary>The granule the arena's memory is committed in.</summary>
    internal long Granule => _granule;

    /// <summary>How many times the arena has committed granules, and how many granules that was over
    /// its life: what a block from the device costs it.</summary>
    internal long Commits { get; private set; }

    /// <summary>
    /// A block of <paramref name="bytes"/> (a multiple of the unit), carved where its memory is
    /// committed already if any free stretch allows — wherever in the stretch the committed memory
    /// lies; failing that, where <paramref name="mayCommit"/> allows, over granules committed for it —
    /// in the free stretch that fits it best, or in a chunk reserved for it. Zero where nothing fits
    /// without committing and that is not allowed, or where the system has no address space or memory
    /// for it.
    /// </summary>
    internal IntPtr Carve(long bytes, bool mayCommit)
    {
        var units = bytes / _unit;
        if (units <= 0) return IntPtr.Zero;
        (long Units, int Chunk, long Start)? fallback = null;
        var first = _free.BinarySearch((units, int.MinValue, long.MinValue));
        if (first < 0) first = ~first;
        // Past a few stretches, the best fit that needs committing is as good as any.
        for (var i = first; i < _free.Count && i < first + 16; i++)
        {
            var span = _free[i];
            if (CommittedWithin(_chunks[span.Chunk]!, span.Start, span.Units, units) is { } at) return Take(span, at, units);
            fallback ??= span;
        }
        if (!mayCommit) return IntPtr.Zero;
        if (fallback is { } fit)
            return Commit(_chunks[fit.Chunk]!, fit.Start, units) ? Take(fit, fit.Start, units) : IntPtr.Zero;
        if (Reserve(Math.Max(_chunkBytes, RoundUp(bytes, _granule))) is not { } chunk) return IntPtr.Zero;
        var whole = (chunk.Units, chunk.Index, 0L);
        return Commit(chunk, 0, units) ? Take(whole, 0, units) : IntPtr.Zero;
    }

    /// <summary>Takes the block of <paramref name="bytes"/> at <paramref name="address"/> back into its
    /// chunk's free stretches, its granules staying committed.</summary>
    internal void Uncarve(IntPtr address, long bytes)
    {
        var chunk = ChunkOf(address);
        var start = ((long)address - (long)chunk.Base) / _unit;
        var units = bytes / _unit;
        Carved(chunk, start, units, -1);
        chunk.CarvedUnits -= units;
        CarvedBytes -= units * _unit;
        var end = start + units;
        if (chunk.FreeByEnd.Remove(start, out var before))
        {
            var length = start - before;
            RemoveFree((length, chunk.Index, before));
            chunk.FreeByStart.Remove(before);
            start = before;
        }
        if (chunk.FreeByStart.Remove(end, out var after))
        {
            RemoveFree((after, chunk.Index, end));
            chunk.FreeByEnd.Remove(end + after);
            end += after;
        }
        Free(chunk, start, end - start);
    }

    /// <summary>
    /// Hands back to the system the committed granules no block is over, idle longest first, until
    /// they come to <paramref name="bytes"/> or none is left: a run of adjacent ones at a time.
    /// Answers the bytes handed back, and in <paramref name="runs"/> how many runs that was.
    /// </summary>
    internal long Decommit(long bytes, out int runs)
    {
        runs = 0;
        long released = 0;
        while (released < bytes && _idleCount > 0)
        {
            var (index, granule) = _idleFirst;
            var chunk = _chunks[index]!;
            var first = granule;
            var last = granule;
            while (first > 0 && IsIdle(chunk, first - 1)) first--;
            while (last + 1 < chunk.Committed.Length && IsIdle(chunk, last + 1)) last++;
            var count = last - first + 1;
            for (var g = first; g <= last; g++)
            {
                Unidle(chunk, g);
                chunk.Committed[g] = false;
            }
            _backing.Decommit(chunk.Base, chunk.State, first, count);
            CommittedBytes -= count * _granule;
            released += count * _granule;
            runs++;
        }
        return released;
    }

    /// <summary>Hands back to the system every committed granule no block is over.</summary>
    internal long DecommitAll(out int runs) => Decommit(long.MaxValue, out runs);

    /// <summary>
    /// Hands back to the system the memory of the whole granules at <paramref name="address"/> for
    /// <paramref name="bytes"/>, which stay carved: a part of a block nothing reads any more whose
    /// address must not be carved again while the block is out. In an arena whose unit is its
    /// granule. Answers the bytes handed back: none where the system would not take them.
    /// </summary>
    internal long DecommitCarved(IntPtr address, long bytes)
    {
        var chunk = ChunkOf(address);
        var first = ((long)address - (long)chunk.Base) / _granule;
        var count = bytes / _granule;
        if (!_backing.Decommit(chunk.Base, chunk.State, first, count)) return 0;
        for (var g = first; g < first + count; g++) chunk.Committed[g] = false;
        CommittedBytes -= count * _granule;
        return count * _granule;
    }

    /// <summary>Whether no block is carved in the arena.</summary>
    internal bool IsEmpty => CarvedBytes == 0;

    /// <summary>
    /// Releases the address space of every chunk no block is carved in, after handing back what is
    /// committed there: what an arena whose account has closed does once its last block goes.
    /// </summary>
    internal void ReleaseEmptyChunks()
    {
        DecommitAll(out _);
        for (var i = 0; i < _chunks.Count; i++)
        {
            if (_chunks[i] is not { CarvedUnits: 0 } chunk) continue;
            RemoveFree((chunk.Units, chunk.Index, 0));
            _backing.Release(chunk.Base, chunk.Bytes, chunk.State);
            ReservedBytes -= chunk.Bytes;
            _chunks[i] = null;
        }
    }

    private bool IsIdle(Chunk chunk, long granule) => chunk.Committed[granule] && chunk.Live[granule] == 0;

    private Chunk ChunkOf(IntPtr address)
    {
        foreach (var chunk in _chunks)
            if (chunk is not null && (long)address >= (long)chunk.Base && (long)address < (long)chunk.Base + chunk.Bytes)
                return chunk;
        throw new InvalidOperationException("The block is not in this arena.");
    }

    /// <summary>The first unit of the free stretch of <paramref name="length"/> units from
    /// <paramref name="start"/> from which <paramref name="units"/> units lie over committed granules
    /// alone, or null where none does: the stretch's runs of committed granules, found a run at a
    /// time.</summary>
    private long? CommittedWithin(Chunk chunk, long start, long length, long units)
    {
        var end = start + length;
        var committed = chunk.Committed.AsSpan();
        var g = start / _unitsPerGranule;
        var last = (end - 1) / _unitsPerGranule;
        while (g <= last)
        {
            var found = committed.Slice((int)g, (int)(last - g + 1)).IndexOf(true);
            if (found < 0) return null;
            var runFirst = g + found;
            var gap = committed.Slice((int)runFirst, (int)(last - runFirst + 1)).IndexOf(false);
            var runEnd = gap < 0 ? last + 1 : runFirst + gap;
            var from = Math.Max(start, runFirst * _unitsPerGranule);
            if (Math.Min(end, runEnd * _unitsPerGranule) - from >= units) return from;
            g = runEnd;
        }
        return null;
    }

    /// <summary>Commits every granule under <paramref name="units"/> units from <paramref name="start"/>
    /// not committed yet, a run of adjacent ones at a time; on a failure, hands back what this
    /// committed and answers false.</summary>
    private bool Commit(Chunk chunk, long start, long units)
    {
        List<(long First, long Count)> done = [];
        var g = start / _unitsPerGranule;
        var last = (start + units - 1) / _unitsPerGranule;
        while (g <= last)
        {
            if (chunk.Committed[g]) { g++; continue; }
            var first = g;
            while (g <= last && !chunk.Committed[g]) g++;
            if (!_backing.Commit(chunk.Base, chunk.State, first, g - first))
            {
                foreach (var (f, c) in done) _backing.Decommit(chunk.Base, chunk.State, f, c);
                return false;
            }
            done.Add((first, g - first));
        }
        foreach (var (f, c) in done)
        {
            for (var i = f; i < f + c; i++) chunk.Committed[i] = true;
            CommittedBytes += c * _granule;
            Commits++;
        }
        return true;
    }

    /// <summary>Takes <paramref name="units"/> units from unit <paramref name="at"/> of free stretch
    /// <paramref name="span"/>, leaving what lies before and after them free.</summary>
    private IntPtr Take((long Units, int Chunk, long Start) span, long at, long units)
    {
        var chunk = _chunks[span.Chunk]!;
        RemoveFree(span);
        chunk.FreeByStart.Remove(span.Start);
        chunk.FreeByEnd.Remove(span.Start + span.Units);
        if (at > span.Start) Free(chunk, span.Start, at - span.Start);
        if (span.Start + span.Units > at + units) Free(chunk, at + units, span.Start + span.Units - at - units);
        Carved(chunk, at, units, +1);
        chunk.CarvedUnits += units;
        CarvedBytes += units * _unit;
        NoteBusy();
        return chunk.Base + (nint)(at * _unit);
    }

    private void Free(Chunk chunk, long start, long units)
    {
        var span = (units, chunk.Index, start);
        var at = _free.BinarySearch(span);
        _free.Insert(at < 0 ? ~at : at, span);
        chunk.FreeByStart[start] = units;
        chunk.FreeByEnd[start + units] = start;
    }

    private void RemoveFree((long Units, int Chunk, long Start) span)
    {
        var at = _free.BinarySearch(span);
        if (at >= 0) _free.RemoveAt(at);
    }

    /// <summary>Puts granule <paramref name="granule"/> of <paramref name="chunk"/> last in the list of
    /// idle granules.</summary>
    private void Idled(Chunk chunk, long granule)
    {
        ref var link = ref chunk.Idling[granule];
        if (link.Listed) return;
        link = new IdleLink { Listed = true, PreviousChunk = _idleLast.Chunk, PreviousGranule = _idleLast.Granule, NextChunk = -1 };
        if (_idleLast.Chunk >= 0)
        {
            ref var last = ref _chunks[_idleLast.Chunk]!.Idling[_idleLast.Granule];
            last.NextChunk = chunk.Index;
            last.NextGranule = granule;
        }
        else
            _idleFirst = (chunk.Index, granule);
        _idleLast = (chunk.Index, granule);
        _idleCount++;
    }

    /// <summary>Takes granule <paramref name="granule"/> of <paramref name="chunk"/> out of the list
    /// of idle granules, where it is in it.</summary>
    private void Unidle(Chunk chunk, long granule)
    {
        ref var link = ref chunk.Idling[granule];
        if (!link.Listed) return;
        if (link.PreviousChunk >= 0)
        {
            ref var previous = ref _chunks[link.PreviousChunk]!.Idling[link.PreviousGranule];
            previous.NextChunk = link.NextChunk;
            previous.NextGranule = link.NextGranule;
        }
        else
            _idleFirst = (link.NextChunk, link.NextGranule);
        if (link.NextChunk >= 0)
        {
            ref var next = ref _chunks[link.NextChunk]!.Idling[link.NextGranule];
            next.PreviousChunk = link.PreviousChunk;
            next.PreviousGranule = link.PreviousGranule;
        }
        else
            _idleLast = (link.PreviousChunk, link.PreviousGranule);
        link = default;
        _idleCount--;
    }

    /// <summary>Counts <paramref name="units"/> units from <paramref name="start"/> carved
    /// (<paramref name="sign"/> +1) or no longer (-1) over the granules they lie in, keeping the idle
    /// granules' order.</summary>
    private void Carved(Chunk chunk, long start, long units, int sign)
    {
        var end = start + units;
        for (var g = start / _unitsPerGranule; g <= (end - 1) / _unitsPerGranule; g++)
        {
            var from = Math.Max(start, g * _unitsPerGranule);
            var to = Math.Min(end, (g + 1) * _unitsPerGranule);
            var was = chunk.Live[g];
            chunk.Live[g] += sign * (to - from);
            if (!chunk.Committed[g]) continue;
            // A granule committed for this very block was never idle, which Unidle allows.
            if (was == 0 && chunk.Live[g] > 0) Unidle(chunk, g);
            else if (was > 0 && chunk.Live[g] == 0) Idled(chunk, g);
        }
    }

    /// <summary>A chunk of <paramref name="bytes"/> of address space, all of it one free stretch, or
    /// null where the system has none to give.</summary>
    private Chunk? Reserve(long bytes)
    {
        bytes = RoundUp(bytes, _granule);
        var @base = _backing.Reserve(bytes, out var state);
        if (@base == IntPtr.Zero) return null;
        ReservedBytes += bytes;
        var index = _chunks.IndexOf(null);
        if (index < 0)
        {
            index = _chunks.Count;
            _chunks.Add(null);
        }
        var granules = bytes / _granule;
        var chunk = new Chunk
        {
            Index = index, Base = @base, Bytes = bytes, Units = bytes / _unit, State = state,
            Committed = new bool[granules], Live = new long[granules], Idling = new IdleLink[granules],
        };
        _chunks[index] = chunk;
        Free(chunk, 0, chunk.Units);
        return chunk;
    }

    private static long RoundUp(long bytes, long to) => (bytes + to - 1) / to * to;
}

/// <summary>
/// Where an <see cref="Arena"/>'s memory comes from: address space reserved with nothing behind it,
/// and memory committed under a range of it a granule at a time, each step undone on its own.
/// </summary>
internal abstract class ArenaBacking
{
    /// <summary>The bytes memory is committed in.</summary>
    internal abstract long Granule { get; }

    /// <summary><paramref name="bytes"/> of address space with nothing committed, or zero where the
    /// system has none; <paramref name="state"/> is the backing's own record of it.</summary>
    internal abstract IntPtr Reserve(long bytes, out object? state);

    /// <summary>Releases address space <see cref="Reserve"/> answered, nothing committed in it.</summary>
    internal abstract void Release(IntPtr @base, long bytes, object? state);

    /// <summary>Commits <paramref name="count"/> granules from granule <paramref name="first"/> of
    /// the reservation at <paramref name="base"/>; false where the system has not that much.</summary>
    internal abstract bool Commit(IntPtr @base, object? state, long first, long count);

    /// <summary>Hands back the <paramref name="count"/> granules from granule <paramref name="first"/>,
    /// which <see cref="Commit"/> committed; false where the system would not take them.</summary>
    internal abstract bool Decommit(IntPtr @base, object? state, long first, long count);

    /// <summary>Waits for the work the device has in hand, as handing back memory some of that work
    /// may still read must; false where it cannot be waited for. Nothing to wait for by
    /// default.</summary>
    internal virtual bool AwaitDevice() => true;
}

/// <summary>
/// Host memory for an <see cref="Arena"/>, in granules of 64 KiB: on Windows address space reserved
/// with <c>VirtualAlloc</c> and committed and decommitted in place; on Linux a private anonymous
/// mapping with no access, made readable and writable to commit and dropped with
/// <c>madvise(MADV_DONTNEED)</c> to decommit. Committed pages are zeroed by the system as they are
/// first touched, and are the process's from then until decommitted.
/// </summary>
internal sealed partial class HostMemory : ArenaBacking
{
    internal static readonly HostMemory Instance = new();

    /// <summary>Whether this system's host memory can back an arena.</summary>
    internal static bool Supported => OperatingSystem.IsWindows() || OperatingSystem.IsLinux();

    internal override long Granule => 64L << 10;

    internal override IntPtr Reserve(long bytes, out object? state)
    {
        state = null;
        if (OperatingSystem.IsWindows()) return VirtualAlloc(IntPtr.Zero, (nuint)bytes, MemReserve, PageNoAccess);
        var pages = mmap(IntPtr.Zero, (nuint)bytes, ProtNone, MapPrivate | MapAnonymous | MapNoReserve, -1, 0);
        return pages == MapFailed ? IntPtr.Zero : pages;
    }

    internal override void Release(IntPtr @base, long bytes, object? state)
    {
        if (OperatingSystem.IsWindows()) VirtualFree(@base, 0, MemRelease);
        else munmap(@base, (nuint)bytes);
    }

    internal override bool Commit(IntPtr @base, object? state, long first, long count)
    {
        var at = @base + (nint)(first * Granule);
        var bytes = (nuint)(count * Granule);
        if (OperatingSystem.IsWindows()) return VirtualAlloc(at, bytes, MemCommit, PageReadWrite) != IntPtr.Zero;
        return mprotect(at, bytes, ProtRead | ProtWrite) == 0;
    }

    internal override bool Decommit(IntPtr @base, object? state, long first, long count)
    {
        var at = @base + (nint)(first * Granule);
        var bytes = (nuint)(count * Granule);
        if (OperatingSystem.IsWindows())
        {
            VirtualFree(at, bytes, MemDecommit);
            return true;
        }
        madvise(at, bytes, MadvDontNeed);
        mprotect(at, bytes, ProtNone);
        return true;
    }

    private const uint MemCommit = 0x1000;
    private const uint MemReserve = 0x2000;
    private const uint MemDecommit = 0x4000;
    private const uint MemRelease = 0x8000;
    private const uint PageNoAccess = 0x01;
    private const uint PageReadWrite = 0x04;
    private const int ProtNone = 0x0;
    private const int ProtRead = 0x1;
    private const int ProtWrite = 0x2;
    private const int MapPrivate = 0x02;
    private const int MapAnonymous = 0x20;
    private const int MapNoReserve = 0x4000;
    private const int MadvDontNeed = 4;
    private static readonly IntPtr MapFailed = -1;

    [LibraryImport("kernel32")]
    private static partial IntPtr VirtualAlloc(IntPtr address, nuint size, uint allocationType, uint protection);

    [LibraryImport("kernel32")]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static partial bool VirtualFree(IntPtr address, nuint size, uint freeType);

    [LibraryImport("libc")]
    private static partial IntPtr mmap(IntPtr address, nuint length, int protection, int flags, int descriptor, nint offset);

    [LibraryImport("libc")]
    private static partial int munmap(IntPtr address, nuint length);

    [LibraryImport("libc")]
    private static partial int mprotect(IntPtr address, nuint length, int protection);

    [LibraryImport("libc")]
    private static partial int madvise(IntPtr address, nuint length, int advice);
}

/// <summary>
/// A card's memory for an <see cref="Arena"/>, through the CUDA driver's virtual memory management
/// (<see cref="CudaVirtualMemory"/>): address space reserved on the device, and each granule its
/// own physical allocation, mapped into place and opened to the device as it is committed, unmapped
/// and released as it is decommitted. One allocation per granule is what lets a granule go back on
/// its own whatever is committed beside it — a block's tail, or one of two ranges of a block.
///
/// <para>Unmapping does not wait for the card: nothing here unmaps a granule that work the card has
/// in hand may still read. The allocator sheds a card's arenas only as a call ends, by when its
/// session's stream is done, or after waiting for the card.</para>
/// </summary>
internal sealed unsafe class CardMemory : ArenaBacking
{
    private readonly int _device;
    private readonly CudaVirtualMemory.Api _api;
    private readonly long _granule;

    private CardMemory(int device, CudaVirtualMemory.Api api, long granule)
    {
        _device = device;
        _api = api;
        _granule = granule;
    }

    /// <summary>The backing for CUDA device <paramref name="device"/>, or null where the driver or the
    /// device does not offer virtual memory management.</summary>
    internal static CardMemory? For(int device)
        => CudaRuntime.OnDevice(device, () => CudaVirtualMemory.For(device)) is { } found
            ? new CardMemory(device, found.Api, found.Granule)
            : null;

    internal override long Granule => _granule;

    internal override IntPtr Reserve(long bytes, out object? state)
    {
        state = null;
        var api = _api;
        var granule = _granule;
        var reserved = CudaRuntime.OnDevice(_device, () =>
        {
            ulong address;
            return api.AddressReserve(&address, (nuint)bytes, (nuint)granule, 0, 0) == 0 ? (IntPtr)(long)address : IntPtr.Zero;
        });
        // The handles' record only once the device has given the range: a request no card could hold
        // is refused by the driver, not by an array the size of its granules.
        if (reserved != IntPtr.Zero) state = new ulong[bytes / _granule];
        return reserved;
    }

    internal override void Release(IntPtr @base, long bytes, object? state)
    {
        var api = _api;
        CudaRuntime.OnDevice(_device, () => api.AddressFree((ulong)(long)@base, (nuint)bytes));
    }

    internal override bool Commit(IntPtr @base, object? state, long first, long count)
    {
        var handles = (ulong[])state!;
        var api = _api;
        var granule = _granule;
        var device = _device;
        return CudaRuntime.OnDevice(_device, () =>
        {
            var properties = CudaVirtualMemory.Properties(device);
            var mapped = 0L;
            for (; mapped < count; mapped++)
            {
                ulong handle;
                if (api.Create(&handle, (nuint)granule, &properties, 0) != 0) break;
                if (api.Map((ulong)(long)@base + (ulong)((first + mapped) * granule), (nuint)granule, 0, handle, 0) != 0)
                {
                    api.ReleaseHandle(handle);
                    break;
                }
                handles[first + mapped] = handle;
            }
            var access = CudaVirtualMemory.ReadWrite(device);
            if (mapped == count
                && api.SetAccess((ulong)(long)@base + (ulong)(first * granule), (nuint)(count * granule), &access, 1) == 0)
                return true;
            Unmap(api, @base, handles, first, mapped, granule);
            return false;
        });
    }

    internal override bool Decommit(IntPtr @base, object? state, long first, long count)
    {
        var handles = (ulong[])state!;
        var api = _api;
        var granule = _granule;
        return CudaRuntime.OnDevice(_device, () => Unmap(api, @base, handles, first, count, granule));
    }

    internal override bool AwaitDevice() => CudaRuntime.Synchronize(_device);

    private static bool Unmap(CudaVirtualMemory.Api api, IntPtr @base, ulong[] handles, long first, long count, long granule)
    {
        if (count == 0) return true;
        api.Unmap((ulong)(long)@base + (ulong)(first * granule), (nuint)(count * granule));
        for (var g = first; g < first + count; g++)
        {
            api.ReleaseHandle(handles[g]);
            handles[g] = 0;
        }
        return true;
    }
}
