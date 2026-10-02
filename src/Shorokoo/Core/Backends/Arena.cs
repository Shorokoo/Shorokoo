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
/// coalesced as blocks go back. Where the arena's unit is its granule, as in an account's own arena,
/// no two blocks share a granule: a block kept alive holds its own granules and no other memory of
/// the arena's. Where the unit is smaller, as in a card's arena of small blocks, blocks share
/// granules, and a granule is shed only once no block is over it.</para>
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
        // Per granule: whether it is committed, how many carved units are over it, and when the last
        // of them went (the arena's clock), which is what shedding goes by.
        internal bool[] Committed = [];
        internal long[] Live = [];
    }

    private readonly List<Chunk?> _chunks = [];

    // Every free stretch, smallest first, for best fit.
    private readonly SortedSet<(long Units, int Chunk, long Start)> _free = [];

    // Every committed granule no carved unit is over, the one idle longest first.
    private readonly SortedSet<(long Since, int Chunk, long Granule)> _idle = [];
    private readonly Dictionary<(int Chunk, long Granule), long> _idleSince = [];
    private long _clock;

    /// <summary>The bytes of the granules committed now.</summary>
    internal long CommittedBytes { get; private set; }

    /// <summary>The bytes of the committed granules no block is over.</summary>
    internal long IdleBytes => _idle.Count * _granule;

    /// <summary>The bytes carved into blocks now.</summary>
    internal long CarvedBytes { get; private set; }

    /// <summary>The granule the arena's memory is committed in.</summary>
    internal long Granule => _granule;

    /// <summary>How many times the arena has committed granules, and how many granules that was over
    /// its life: what a block from the device costs it.</summary>
    internal long Commits { get; private set; }

    /// <summary>
    /// A block of <paramref name="bytes"/> (a multiple of the unit), carved where its memory is
    /// committed already if any free stretch allows; failing that, where <paramref name="mayCommit"/>
    /// allows, over granules committed for it — in the free stretch that fits it best, or in a chunk
    /// reserved for it. Zero where nothing fits without committing and that is not allowed, or where
    /// the system has no address space or memory for it.
    /// </summary>
    internal IntPtr Carve(long bytes, bool mayCommit)
    {
        var units = bytes / _unit;
        (long Units, int Chunk, long Start)? fallback = null;
        var tried = 0;
        foreach (var span in _free.GetViewBetween((units, int.MinValue, long.MinValue), (long.MaxValue, int.MaxValue, long.MaxValue)))
        {
            if (Committed(_chunks[span.Chunk]!, span.Start, units)) return Take(span, units);
            fallback ??= span;
            // Past a few stretches, the best fit that needs committing is as good as any.
            if (++tried == 16) break;
        }
        if (!mayCommit) return IntPtr.Zero;
        if (fallback is { } fit)
            return Commit(_chunks[fit.Chunk]!, fit.Start, units) ? Take(fit, units) : IntPtr.Zero;
        if (Reserve(Math.Max(_chunkBytes, RoundUp(bytes, _granule))) is not { } chunk) return IntPtr.Zero;
        var whole = (chunk.Units, chunk.Index, 0L);
        return Commit(chunk, 0, units) ? Take(whole, units) : IntPtr.Zero;
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
            _free.Remove((length, chunk.Index, before));
            chunk.FreeByStart.Remove(before);
            start = before;
        }
        if (chunk.FreeByStart.Remove(end, out var after))
        {
            _free.Remove((after, chunk.Index, end));
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
        while (released < bytes && _idle.Count > 0)
        {
            var (_, index, granule) = _idle.Min;
            var chunk = _chunks[index]!;
            var first = granule;
            var last = granule;
            while (first > 0 && IsIdle(chunk, first - 1)) first--;
            while (last + 1 < chunk.Committed.Length && IsIdle(chunk, last + 1)) last++;
            var count = last - first + 1;
            for (var g = first; g <= last; g++)
            {
                _idle.Remove((_idleSince[(index, g)], index, g));
                _idleSince.Remove((index, g));
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
            _free.Remove((chunk.Units, chunk.Index, 0));
            _backing.Release(chunk.Base, chunk.Bytes, chunk.State);
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

    /// <summary>Whether every granule under <paramref name="units"/> units from
    /// <paramref name="start"/> is committed.</summary>
    private bool Committed(Chunk chunk, long start, long units)
    {
        for (var g = start / _unitsPerGranule; g <= (start + units - 1) / _unitsPerGranule; g++)
            if (!chunk.Committed[g]) return false;
        return true;
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

    /// <summary>Takes <paramref name="units"/> units from the start of free stretch
    /// <paramref name="span"/>, leaving the rest free.</summary>
    private IntPtr Take((long Units, int Chunk, long Start) span, long units)
    {
        var chunk = _chunks[span.Chunk]!;
        _free.Remove(span);
        chunk.FreeByStart.Remove(span.Start);
        chunk.FreeByEnd.Remove(span.Start + span.Units);
        if (span.Units > units) Free(chunk, span.Start + units, span.Units - units);
        Carved(chunk, span.Start, units, +1);
        chunk.CarvedUnits += units;
        CarvedBytes += units * _unit;
        return chunk.Base + (nint)(span.Start * _unit);
    }

    private void Free(Chunk chunk, long start, long units)
    {
        _free.Add((units, chunk.Index, start));
        chunk.FreeByStart[start] = units;
        chunk.FreeByEnd[start + units] = start;
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
            if (was == 0 && chunk.Live[g] > 0)
            {
                // A granule committed for this very block was never idle.
                if (_idleSince.Remove((chunk.Index, g), out var since)) _idle.Remove((since, chunk.Index, g));
            }
            else if (was > 0 && chunk.Live[g] == 0)
            {
                var since = ++_clock;
                _idle.Add((since, chunk.Index, g));
                _idleSince[(chunk.Index, g)] = since;
            }
        }
    }

    /// <summary>A chunk of <paramref name="bytes"/> of address space, all of it one free stretch, or
    /// null where the system has none to give.</summary>
    private Chunk? Reserve(long bytes)
    {
        bytes = RoundUp(bytes, _granule);
        var @base = _backing.Reserve(bytes, out var state);
        if (@base == IntPtr.Zero) return null;
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
            Committed = new bool[granules], Live = new long[granules],
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
    /// which <see cref="Commit"/> committed.</summary>
    internal abstract void Decommit(IntPtr @base, object? state, long first, long count);
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

    internal override void Decommit(IntPtr @base, object? state, long first, long count)
    {
        var at = @base + (nint)(first * Granule);
        var bytes = (nuint)(count * Granule);
        if (OperatingSystem.IsWindows())
        {
            VirtualFree(at, bytes, MemDecommit);
            return;
        }
        madvise(at, bytes, MadvDontNeed);
        mprotect(at, bytes, ProtNone);
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

    internal override void Decommit(IntPtr @base, object? state, long first, long count)
    {
        var handles = (ulong[])state!;
        var api = _api;
        var granule = _granule;
        CudaRuntime.OnDevice(_device, () => Unmap(api, @base, handles, first, count, granule));
    }

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
