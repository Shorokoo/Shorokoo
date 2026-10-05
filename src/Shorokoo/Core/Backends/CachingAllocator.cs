using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Shorokoo.Core.Backends;

/// <summary>
/// Shorokoo's allocator for one device — the host, or one CUDA card — for the whole process: what
/// every ONNX Runtime session allocates through on that device, whichever runtime and whichever
/// copy of the backend built it, and what every tensor a backend places on a card comes from. A
/// backend hands it to its runtime through a native allocator of Shorokoo's, which forwards each
/// request to <see cref="AllocateEntry"/> and <see cref="FreeEntry"/>.
///
/// <para><b>One per device, not per runtime.</b> Two backends can bind one native runtime — the
/// program's own and one loaded in isolation over the same file — and a runtime keeps one allocator
/// per device for every session it builds, whichever backend asked. So the allocator, its accounts
/// and the charging that picks an account are this assembly's, which every copy of a backend
/// shares, and a runtime is handed it once (<see cref="HandTo"/>): the sessions of either
/// backend then charge their own accounts, and are held to their own limits.</para>
///
/// <para><b>Why the session's allocator is Shorokoo's.</b> ONNX Runtime writes a run's every output
/// into memory its session's allocator gives it — an output whose shape only the run learns
/// included — and has no hook for one output's memory apart from the rest. An arena whose regions go
/// back only whole lets a kept output hold its region, and everything carved from it, alive. Here a
/// kept output holds its own pages and no other memory: an output is never copied after its run, and
/// nothing it holds stops the rest from going back.</para>
///
/// <para><b>Where a block's memory comes from.</b> A request is rounded up to a size class
/// (<see cref="ClassOf"/>), and the class decides:</para>
/// <list type="bullet">
/// <item>on the host, a block under 64 KiB is an allocation of its own from the C runtime's heap; a
/// larger one, a whole number of 4 KiB pages, is carved from its account's <see cref="Arena"/> of
/// host pages, committed 64 KiB at a time (<see cref="HostMemory"/>);</item>
/// <item>on a card whose driver offers virtual memory management, a block of up to a mebibyte is
/// carved from the card's arena of small blocks, which every account shares, packed into the card's
/// 2 MiB granules as the CUDA runtime packs small blocks into its pages; a larger one, a whole number
/// of granules, from its account's arena of granules (<see cref="CardMemory"/>);</item>
/// <item>on a card whose driver does not, every block is a <c>cudaMalloc</c> of its own.</item>
/// </list>
/// <para>An account's arena reserves address space with no memory behind it and commits a granule as
/// a block is carved over it. A block let go of whole stays carved and is kept for the next request
/// of its class, so a loop's runs find their blocks waiting with no work at all — one that handed
/// parts of itself back while in use (<see cref="ReleaseRange"/>) goes back to its arena instead,
/// and so does a large one of the host's own account (<see cref="Account.KeepsLargeBlocks"/>); a request of a class no
/// kept block serves is carved from what the arena keeps committed, whatever block it was carved for
/// before, so a session fed one shape after another reuses warm memory as an arena does, rather than
/// taking every new size from the system. On a card no two blocks of an account's arena share a
/// granule, so a block kept alive holds its own granules and nothing else of the arena's; on the host
/// a block kept alive holds its own pages, and at most the rest of the two granules its ends lie
/// in.</para>
///
/// <para><b>What an account keeps.</b> An account never holds from the device more than the most
/// one of its calls has used: the blocks it had in use as the call began and every block the call
/// was handed, each counted once (<see cref="Account.MaxUsed"/>) — for the device's own account, the
/// tensors placed between one session call's end on the device and the next. Where a request would
/// take it past that mark and nothing committed fits it, the smallest larger block the account keeps
/// in its arena gives way to it, its memory counted once in the call; or else kept blocks give way,
/// longest kept first, their memory staying committed for the request to be carved from — until the
/// request fits where it would take the account over by an eighth of itself or more, and otherwise
/// only as much as it would, the request then committing fresh memory. What is still over the mark
/// is shed — kept blocks of their own and small ones first, then the granules idle longest, then
/// kept blocks of the arena — at once on the host, and on a card as the call that asked ends, since a
/// granule must not go back while work the card has in hand may read it. The host's own account keeps
/// no block of <see cref="LargeBlock"/> or more (<see cref="Account.KeepsLargeBlocks"/>): a large
/// tensor made outside a run goes back to the system as it goes. What is kept also goes back
/// when a run that hands back its memory ends (<see cref="ReleaseCached(Account)"/>), when a session
/// closes, before an account would refuse an allocation for want of room, before the device would
/// refuse one, and when the program asks (<see cref="ReleaseEverywhere"/>).</para>
///
/// <para><b>Accounts.</b> What a session allocates is charged to its <see cref="Account"/>: the
/// thread building or running it charges that account for the length of the call
/// (<see cref="Charge"/>), and ONNX Runtime allocates on the thread that called it — measured, every
/// allocation of a session's construction and runs, on the host with four intra-op threads and on
/// the card alike. Each block remembers its account, so a block let go of on any thread — a
/// collected output, say — is credited to the account that took it. An account carries a session's
/// statistics (<see cref="Statistics"/>) and, on a card, the limit a device-memory budget leaves its
/// runs (<see cref="Account.Limit"/>), which it enforces as each block is asked for. Anything not
/// charged to a session — what the framework places on a card, and the tensors a backend makes in
/// host memory outside a run — goes to the device's own account (<see cref="Placements"/>).</para>
///
/// <para><b>The card's streams.</b> A session's kernels run on a stream of its own, so a block a run
/// lets go of may still be read by work queued before it. Such a block is reused by that run alone
/// until it ends — on the same stream, so in order — and only then by anything else.</para>
///
/// <para><b>Refusing.</b> A request this cannot serve — one an account's limit has no room for, or
/// one the device has no memory for — is answered with null and the reason, and the native side
/// throws that reason as ONNX Runtime's own allocators throw theirs: the call that asked fails as an
/// allocation failure, and the device is left as usable as before it.</para>
///
/// <para><b>Held for the life of the process</b>, as every tensor that came from it frees itself
/// through it.</para>
/// </summary>
internal sealed unsafe class CachingAllocator
{
    /// <summary>The key the host's allocator is held under, beside the CUDA devices'.</summary>
    private const int Host = -1;

    /// <summary>On the host, the smallest class carved from an account's arena: its granule.</summary>
    private const long HostArenaFrom = 64L << 10;

    /// <summary>On the host, what a block carved from an account's arena is a whole number of: a
    /// page, so a block holds its own pages and less than one more.</summary>
    private const long HostPage = 4L << 10;

    /// <summary>On a card, the largest class packed into the card's arena of small blocks rather than
    /// carved from an account's arena of whole granules: a mebibyte, the most the CUDA runtime packs
    /// two to its 2 MiB page.</summary>
    private const long CardSmallTo = 1L << 20;

    /// <summary>What the card's arena of small blocks carves in.</summary>
    private const long SmallUnit = 512;

    /// <summary>The most a request may ask for: more than any device holds, and far enough below the
    /// largest size there is that rounding a request up to its class cannot run past it.</summary>
    private const long Largest = 1L << 62;

    private static readonly object _registryGate = new();
    private static readonly Dictionary<int, CachingAllocator> _byDevice = [];

    /// <summary>The allocator for the host, made on first use.</summary>
    internal static CachingAllocator ForHost() => For(Host);

    /// <summary>The allocator for CUDA device <paramref name="deviceId"/>, made on first use.</summary>
    internal static CachingAllocator ForCard(int deviceId) => For(deviceId);

    private static CachingAllocator For(int device)
    {
        lock (_registryGate)
        {
            if (_byDevice.TryGetValue(device, out var existing)) return existing;
            return _byDevice[device] = new CachingAllocator(device);
        }
    }

    private readonly int _device;
    private Gate _gate;
    private readonly BlockTable _blocks = new();
    private readonly HashSet<IntPtr> _runtimes = [];

    // What the device's arenas are backed by -- host pages, or the card's granules -- or null on a
    // card whose driver offers no virtual memory management; and on such a card, the arena of small
    // blocks every account carves from.
    private readonly ArenaBacking? _backing;
    private readonly Arena? _small;

    /// <summary>Where a block's memory came from, which is where it goes back to.</summary>
    internal enum Source : byte
    {
        /// <summary>An allocation of its own: the C runtime's heap, or <c>cudaMalloc</c>.</summary>
        Own,

        /// <summary>Carved from its account's arena.</summary>
        Arena,

        /// <summary>Carved from the card's arena of small blocks.</summary>
        Small,
    }

    /// <summary>What a block is: its size class, what was asked for, the account it is charged to,
    /// where its memory came from, the account's call it was last counted in, whether that account
    /// has handed it over (<see cref="HandOver"/>), and the parts of it handed back while the rest
    /// was still in use (<see cref="ReleaseRange"/>) — by offset, in order, none where none was;
    /// whether the memory of its first granule went back to the card with the block still carved
    /// over it — with their bytes, and those of them that had been asked for.</summary>
    private struct Block
    {
        internal long Size;
        internal long Requested;
        internal Account Account;
        internal long Call;
        internal Source Source;
        internal bool HandedOver;
        internal List<(long Start, long End)>? Released;
        internal bool FirstGone;
        internal long ReleasedBytes;
        internal long ReleasedRequested;
    }

    /// <summary>
    /// The blocks out, by address: open addressing with linear probing and backward-shift deletion,
    /// which hands a block in and out with a hash and a probe or two and no allocation, each address
    /// beside its block so that a probe reads one entry.
    /// </summary>
    private sealed class BlockTable
    {
        private struct Entry
        {
            internal IntPtr Key;
            internal Block Value;
        }

        private Entry[] _entries = new Entry[256];
        private int _count;

        private int Slot(IntPtr key) => (int)(((ulong)key >> 6) * 0x9E3779B97F4A7C15UL >> 40) & (_entries.Length - 1);

        /// <summary>Adds <paramref name="value"/> under <paramref name="key"/>; false, adding nothing,
        /// where a block is out under that address already.</summary>
        internal bool Add(IntPtr key, in Block value)
        {
            if ((_count + 1) * 2 > _entries.Length) Grow();
            var entries = _entries;
            var mask = entries.Length - 1;
            var i = Slot(key);
            while (entries[i].Key != IntPtr.Zero)
            {
                if (entries[i].Key == key) return false;
                i = (i + 1) & mask;
            }
            _count++;
            entries[i].Key = key;
            entries[i].Value = value;
            return true;
        }

        internal ref Block Find(IntPtr key)
        {
            var entries = _entries;
            var mask = entries.Length - 1;
            for (var i = Slot(key); entries[i].Key != IntPtr.Zero; i = (i + 1) & mask)
                if (entries[i].Key == key) return ref entries[i].Value;
            return ref Unsafe.NullRef<Block>();
        }

        internal bool Remove(IntPtr key, out Block value)
        {
            var entries = _entries;
            var mask = entries.Length - 1;
            var i = Slot(key);
            while (entries[i].Key != key)
            {
                if (entries[i].Key == IntPtr.Zero)
                {
                    value = default;
                    return false;
                }
                i = (i + 1) & mask;
            }
            value = entries[i].Value;
            // Each later entry of the run that could no longer be found moves into the gap.
            for (var j = (i + 1) & mask; entries[j].Key != IntPtr.Zero; j = (j + 1) & mask)
            {
                var home = Slot(entries[j].Key);
                var reachable = i <= j ? home > i && home <= j : home > i || home <= j;
                if (reachable) continue;
                entries[i] = entries[j];
                i = j;
            }
            entries[i] = default;
            _count--;
            return true;
        }

        private void Grow()
        {
            var entries = _entries;
            _entries = new Entry[entries.Length * 2];
            _count = 0;
            foreach (ref var entry in entries.AsSpan())
                if (entry.Key != IntPtr.Zero) Add(entry.Key, entry.Value);
        }
    }

    private CachingAllocator(int device)
        : this(device, device == Host ? HostMemory.Supported ? HostMemory.Instance : null : CardMemory.For(device))
    {
    }

    /// <summary>An allocator for <paramref name="device"/> — the host for -1, a card otherwise — whose
    /// arenas are backed by <paramref name="backing"/>, or which takes every block of its own from the
    /// device where that is null.</summary>
    internal CachingAllocator(int device, ArenaBacking? backing)
    {
        _device = device;
        _backing = backing;
        if (device != Host && backing is not null) _small = new Arena(backing, unit: SmallUnit, chunkBytes: 256L << 20);
        Placements = new Account(this, "placements", keepsLargeBlocks: device != Host);
        lock (_registryGate)
        {
            _states.Add(this);
            State = _states.Count;
        }
    }

    /// <summary>The handle the native allocator passes back to the entry points, naming this
    /// allocator.</summary>
    internal IntPtr State { get; }

    /// <summary>Where the native allocator asks for a block: <c>(state, size, reason, capacity)</c>,
    /// answering the block or null with the reason written down.</summary>
    internal static IntPtr AllocateEntry => (IntPtr)(delegate* unmanaged<IntPtr, nuint, byte*, int, IntPtr>)&AllocCallback;

    /// <summary>Where the native allocator hands a block back: <c>(state, block)</c>.</summary>
    internal static IntPtr FreeEntry => (IntPtr)(delegate* unmanaged<IntPtr, IntPtr, void>)&FreeCallback;

    /// <summary>
    /// Hands this allocator to the runtime whose environment is <paramref name="environment"/>
    /// through <paramref name="register"/>, once: the first time that runtime is named, and not again
    /// however many backends bind it. A runtime keeps one allocator per device, so a second
    /// registration would take the device over from the first. A registration the runtime refuses —
    /// <paramref name="register"/> throwing — leaves the runtime without it, so the next backend to
    /// name that runtime registers it again and is refused again, as loudly.
    /// </summary>
    internal void HandTo(IntPtr environment, Action register)
    {
        lock (_runtimes)
        {
            if (_runtimes.Contains(environment)) return;
            register();
            _runtimes.Add(environment);
        }
    }

    /// <summary>Whether this is a card's allocator.</summary>
    internal bool OnCard => _device != Host;

    /// <summary>The account of what the framework places on the device itself.</summary>
    internal Account Placements { get; }

    /// <summary>A new account, for a session about to be built.</summary>
    internal Account Open(string name, bool keepsLargeBlocks = true) => new(this, name, keepsLargeBlocks);

    /// <summary>The least a host block is that an account keeping no large block
    /// (<see cref="Account.KeepsLargeBlocks"/>) hands back as it is let go of.</summary>
    internal const long LargeBlock = 1L << 20;

    // ---- the entry points the native allocator forwards to ----

    // Every allocator made, the one a native allocator names by its state at index state - 1: an
    // index rather than a handle, so the entry points find it with one read.
    private static readonly List<CachingAllocator> _states = [];
    private static CachingAllocator[] _byState = [];

    private static CachingAllocator Of(IntPtr state)
    {
        var all = Volatile.Read(ref _byState);
        var index = (int)state - 1;
        if ((uint)index < (uint)all.Length) return all[index];
        lock (_registryGate) Volatile.Write(ref _byState, [.. _states]);
        return _byState[index];
    }

    /// <summary>
    /// A block of <paramref name="size"/> bytes, or null with the reason written into
    /// <paramref name="reason"/> (UTF-8, NUL-terminated, at most <paramref name="capacity"/> bytes),
    /// which the native side throws. Nothing unwinds out of here: whatever goes wrong is a reason.
    /// </summary>
    [UnmanagedCallersOnly]
    private static IntPtr AllocCallback(IntPtr state, nuint size, byte* reason, int capacity)
        => AllocateOrRefuse(state, size, reason is null || capacity <= 0 ? default : new Span<byte>(reason, capacity));

    /// <summary>What <see cref="AllocCallback"/> answers: a block of <paramref name="size"/> bytes from
    /// the allocator <paramref name="state"/> names, or null with the reason written into
    /// <paramref name="reason"/>. Nothing escapes it: wording a failure can fail too — the memory a
    /// message takes may be the very memory there is none of, or the failure may not say what it
    /// is — and then a reason fixed in advance is written, with nothing to allocate.</summary>
    internal static IntPtr AllocateOrRefuse(IntPtr state, nuint size, Span<byte> reason)
    {
        try
        {
            try
            {
                var block = Of(state).Allocate(size > long.MaxValue ? long.MaxValue : (long)size, out var refusal);
                if (block != IntPtr.Zero) return block;
                Write(refusal ?? $"Failed to allocate {size} bytes.", reason);
            }
            catch (Exception failure)
            {
                Write($"Failed to allocate {size} bytes: Shorokoo's allocator failed ({failure.Message}).", reason);
            }
        }
        catch
        {
            Write("Failed to allocate: Shorokoo's allocator failed."u8, reason);
        }
        return IntPtr.Zero;
    }

    /// <summary><paramref name="text"/> into <paramref name="buffer"/> as UTF-8, cut short where it
    /// does not fit, and NUL-terminated.</summary>
    private static void Write(string text, Span<byte> buffer)
    {
        if (buffer.IsEmpty) return;
        System.Text.Unicode.Utf8.FromUtf16(text, buffer[..^1], out _, out var written, replaceInvalidSequences: true, isFinalBlock: true);
        buffer[written] = 0;
    }

    /// <summary><paramref name="text"/>, UTF-8 already, into <paramref name="buffer"/>, cut short where
    /// it does not fit, and NUL-terminated: nothing allocated.</summary>
    private static void Write(ReadOnlySpan<byte> text, Span<byte> buffer)
    {
        if (buffer.IsEmpty) return;
        var written = Math.Min(text.Length, buffer.Length - 1);
        text[..written].CopyTo(buffer);
        buffer[written] = 0;
    }

    [UnmanagedCallersOnly]
    private static void FreeCallback(IntPtr state, IntPtr pointer)
    {
        try
        {
            Of(state).Free(pointer);
        }
        catch
        {
            // Nothing may cross into native code: a block whose bookkeeping failed is lost rather
            // than the process.
        }
    }

    // ---- sizes ----

    /// <summary>The size class <paramref name="bytes"/> is served from on the host: a multiple of
    /// 512 bytes under 64 KiB, and of a 4 KiB page from there.</summary>
    internal static long SizeClass(long bytes)
        => bytes < HostArenaFrom ? (bytes + 511) & ~511L : (bytes + HostPage - 1) & ~(HostPage - 1);

    /// <summary>
    /// The size class <paramref name="bytes"/> is served from on this device: on the host
    /// <see cref="SizeClass"/>; on a card with arenas a multiple of 512 bytes up to a mebibyte, and a
    /// whole number of granules above it; on a card without, a multiple of 512 bytes up to a
    /// mebibyte, and above it of an eighth of the power of two below.
    /// </summary>
    internal long ClassOf(long bytes)
    {
        if (!OnCard) return SizeClass(bytes);
        if (_backing is { } card)
            return bytes <= CardSmallTo ? (bytes + 511) & ~511L : (bytes + card.Granule - 1) / card.Granule * card.Granule;
        if (bytes <= 1L << 20) return (bytes + 511) & ~511L;
        var step = 1L << (63 - BitOperations.LeadingZeroCount((ulong)bytes) - 3);
        return (bytes + step - 1) & ~(step - 1);
    }

    /// <summary>Where a block of class <paramref name="size"/> comes from on this device.</summary>
    private Source SourceOf(long size)
    {
        if (_backing is null) return Source.Own;
        if (!OnCard) return size < HostArenaFrom ? Source.Own : Source.Arena;
        return size <= CardSmallTo ? Source.Small : Source.Arena;
    }

    // ---- allocation ----

    /// <summary>
    /// A block of at least <paramref name="bytes"/>, charged to the account this thread charges for
    /// this device (<see cref="Charge"/>) or to <see cref="Placements"/>; or null, with
    /// <paramref name="refusal"/> saying why, for a block the account's limit has no room for or
    /// the device has no memory for. What is kept for reuse goes back first: the account's own
    /// before its limit refuses, as much as makes room, everything the device keeps before the device
    /// does — and on a card what the asking call let go of, once the card is done with it, waited for
    /// outside the lock and only where nothing else makes room.
    ///
    /// <para>The reason is worded for <c>AllocationFailureReport</c>, which reads it out of the
    /// failure ONNX Runtime raises: it says the allocation failed and names the memory — the CUDA
    /// allocator on a card, a bad allocation on the host.</para>
    /// </summary>
    internal IntPtr Allocate(long bytes, out string? refusal)
    {
        refusal = null;
        if (bytes <= 0) return IntPtr.Zero;
        var scope = t_scope;
        var account = scope?.AccountOn(this) ?? Placements;
        if (bytes > Largest)
        {
            using (_gate.Hold()) account.Refusals++;
            refusal = OnCard
                ? $"Failed to allocate {bytes} bytes {Where}: {Allocator} serves nothing that large, more than any card holds."
                : $"Failed to allocate {bytes} bytes {Where}: more than any process could commit (a bad allocation).";
            return IntPtr.Zero;
        }
        var size = ClassOf(bytes);
        var source = SourceOf(size);
        List<(IntPtr, long)>? release = null;
        var carved = IntPtr.Zero;
        // On a card what this call let go of goes back only once the card is done with the work the
        // call queued: where the request needs it, the card is waited for -- outside the lock, which
        // every allocation and free on the device takes -- and the request asked afresh.
        for (var waited = false; ; waited = true)
        {
            if (waited)
            {
                Release(release);
                release = null;
                AwaitCard();
            }
            using (_gate.Hold())
            {
                if (account == Placements) BeginPlacementsCall();
                if (scope is not null && scope.TakeHeld(account, size, out var held, out var call))
                    return Hand(held, size, bytes, account, source, call);
                if (account.TakeKept(size, out held, out call))
                    return Hand(held, size, bytes, account, source, call);
                if (account.LimitUnderLock is { } limit && account.Charged + account.KeptBytes + size > limit)
                {
                    // Under a budget what the account keeps goes back first, as much as makes room;
                    // then what this call let go of on a card, once the card is done with it; and
                    // then only what it has out counts. Where what it has out leaves no room already,
                    // nothing it keeps or holds could make any: none of it goes, and the card is not
                    // waited for.
                    if (account.Charged + size <= limit)
                    {
                        release = Shed(account, account.Charged + account.KeptBytes + size - limit);
                        if (account.Charged + account.KeptBytes + size > limit && scope is not null && scope.Holds(account))
                        {
                            if (!waited) continue;
                            foreach (var (block, blockSize, from, last) in scope.TakeAllHeld(account)) account.Keep(block, blockSize, from, last);
                            release.AddRange(Shed(account, account.Charged + account.KeptBytes + size - limit));
                        }
                    }
                    if (account.Charged + size > limit)
                    {
                        account.Refusals++;
                        refusal = $"Failed to allocate {bytes} bytes {Where}: {Allocator} may hold {limit} "
                            + "bytes there for this session, what the device-memory budget "
                            + $"(DeviceMemorySettings.LimitBytes) leaves its run, and it holds {account.Charged}.";
                    }
                }
                // What the account would hold from the device with one more block of this class.
                var excess = account.HeldBytes + size - Math.Max(account.MaxUsed, account.Used + size);
                if (refusal is null && source != Source.Own)
                {
                    carved = CarveFor(account, size, source, excess, out var counted);
                    // The device is full: on a card, once the card is done with what this call let
                    // go of, ...
                    if (carved == IntPtr.Zero && OnCard && !waited) continue;
                    if (carved == IntPtr.Zero)
                    {
                        // ... what is kept anywhere on it goes back, and that too, and the request is
                        // tried once more.
                        ReleaseEverythingCachedUnderLock(release ??= [], scope);
                        carved = CarveFor(account, size, source, excess: 0, out counted);
                    }
                    if (carved == IntPtr.Zero)
                    {
                        account.Refusals++;
                        refusal = Refusal(bytes);
                    }
                    else
                    {
                        account.Blocks++;
                        Hand(carved, size, bytes, account, source, counted);
                        // On the host what is over the account's mark goes back at once; on a card as
                        // the call charging the account ends (Scope.End), where one does.
                        if (!OnCard || scope is null || !scope.Charges(account))
                            (release ??= []).AddRange(Shed(account, account.HeldBytes - account.Bound));
                    }
                }
                else if (refusal is null && (!OnCard || scope is null || !scope.Charges(account)))
                    (release ??= []).AddRange(Shed(account, excess));
            }
            break;
        }
        Release(release);
        if (carved != IntPtr.Zero) return carved;
        if (refusal is not null) return IntPtr.Zero;
        var own = Fresh(size);
        if (own == IntPtr.Zero)
        {
            // The device is full: what is kept anywhere on it goes back, and what this call let go
            // of -- the card waiting for the work it has in hand as each block goes back to it --
            // and the request is tried once more.
            ReleaseEverythingCached(scope);
            own = Fresh(size);
            if (own == IntPtr.Zero)
            {
                using (_gate.Hold()) account.Refusals++;
                refusal = Refusal(bytes);
                return IntPtr.Zero;
            }
        }
        using (_gate.Hold())
        {
            account.Blocks++;
            return Hand(own, size, bytes, account, Source.Own, call: -1);
        }
    }

    /// <summary>
    /// A block of class <paramref name="size"/> carved from where <paramref name="source"/> says: the
    /// card's arena of small blocks, or the account's own arena — from what it keeps committed where
    /// that fits; where nothing does and a new granule would take the account past its mark
    /// (<paramref name="excess"/> over zero), from the memory of the smallest larger block it keeps,
    /// or else of what its kept blocks of that arena give way to, longest kept first; and only then
    /// over granules committed for it, what is over the mark going back as the caller sheds it. Kept
    /// blocks give way until the request fits where the excess is an eighth of the request or more,
    /// its memory having to come mostly from what is kept anyway; where it is less, only as much as
    /// the excess, since committing the request and shedding that little costs less than giving way
    /// blocks the call may yet ask for again. Zero where the device has no memory for it.
    /// <paramref name="counted"/> is the call the block's memory was last counted in: that of the
    /// larger kept block it takes the place of, so a call that lets a block go and asks for a smaller
    /// one counts the memory once.
    /// </summary>
    private IntPtr CarveFor(Account account, long size, Source source, long excess, out long counted)
    {
        counted = -1;
        if (source == Source.Small) return _small!.Carve(size, mayCommit: true);
        var arena = account.Arena ??= new Arena(_backing!, unit: OnCard ? _backing!.Granule : HostPage,
            chunkBytes: OnCard ? 1L << 30 : 256L << 20);
        var block = arena.Carve(size, mayCommit: false);
        if (block == IntPtr.Zero && excess > 0 && account.TakeKeptLarger(size, out var larger, out var largerSize, out var call))
        {
            arena.Uncarve(larger, largerSize);
            block = arena.Carve(size, mayCommit: false);
            // Counted where the block is carved from memory the account had: fresh memory counts.
            if (block != IntPtr.Zero) counted = call;
        }
        var giveWay = excess * 8 >= size ? long.MaxValue : excess;
        for (long givenWay = 0; block == IntPtr.Zero && givenWay < giveWay && account.GiveWayOldest(fromArena: true, out var kept, out var keptSize, out _); givenWay += keptSize)
        {
            arena.Uncarve(kept, keptSize);
            block = arena.Carve(size, mayCommit: false);
        }
        return block != IntPtr.Zero ? block : arena.Carve(size, mayCommit: true);
    }

    /// <summary>The words a request the device has no memory for is refused in.</summary>
    private string Refusal(long bytes) => OnCard
        ? $"Failed to allocate {bytes} bytes {Where}: the card has no such block free "
          + $"(CUDA refused it), with everything {Allocator} kept for reuse handed back."
        : $"Failed to allocate {bytes} bytes {Where}: the process could not commit that much "
          + "more (a bad allocation).";

    // How many session calls on this device have ended, which divides what is placed on it into
    // calls of its own (BeginPlacementsCall).
    private long _callsEnded;

    /// <summary>
    /// Starts a call of <see cref="Placements"/> where a session call on this device has ended since
    /// it was last handed a block: what is placed between two session calls is one call of its own.
    /// Under the lock.
    /// </summary>
    private void BeginPlacementsCall()
    {
        if (Placements.CallsSeen == _callsEnded) return;
        Placements.CallsSeen = _callsEnded;
        Placements.BeginCall();
    }

    /// <summary>One block handed out, or one taken back, as <see cref="Observer"/> is told of it:
    /// whether it is handed out, on which device, at what address, of how many bytes asked for and
    /// served, and whether the device gave it fresh rather than out of what was kept.</summary>
    internal readonly record struct Event(bool Allocation, bool OnCard, IntPtr Address, long Requested, long Size, bool Fresh);

    /// <summary>Told of every block every allocator hands out and takes back — and of each part of
    /// a block taken back while the rest is in use (<see cref="ReleaseRange"/>), at the part's own
    /// address, the block's own taking back then telling only what was left — under the allocator's
    /// lock and on the thread making the call. Null tells nothing; the memory-reuse measurements
    /// set it for the length of a run.</summary>
    internal static Action<Event>? Observer;

    /// <summary>Where this allocator's memory is, as a refusal says it.</summary>
    private string Where => OnCard ? $"on CUDA device {_device}" : "of host memory";

    /// <summary>What a refusal calls this allocator: on a card, by the name
    /// <c>AllocationFailureReport</c> reads as the card's.</summary>
    private string Allocator => OnCard ? "Shorokoo's cuda_allocator" : "Shorokoo's host allocator";

    /// <summary>Records <paramref name="block"/> as handed to <paramref name="account"/>, and as used
    /// by the account's call where it was not counted in it already (<paramref name="call"/> is the
    /// call it was last counted in).</summary>
    /// <exception cref="InvalidOperationException">A block is out at that address already: the two
    /// could not be told apart as they are let go of.</exception>
    private IntPtr Hand(IntPtr block, long size, long requested, Account account, Source source, long call)
    {
        if (!_blocks.Add(block, new Block { Size = size, Requested = requested, Account = account, Source = source, Call = account.CallNumber }))
            throw new InvalidOperationException($"Shorokoo's allocator was about to hand out the block at 0x{block:x} {Where} while a block it handed out there is still in use.");
        Observer?.Invoke(new Event(true, OnCard, block, requested, size, Fresh: call < 0));
        if (call != account.CallNumber)
        {
            account.Used += size;
            if (account.Used > account.MaxUsed) account.MaxUsed = account.Used;
        }
        account.InUse += size;
        account.Requested += requested;
        account.Allocations++;
        if (account.InUse > account.MaxInUse) account.MaxInUse = account.InUse;
        if (account.InUse > account.SpanPeak) account.SpanPeak = account.InUse;
        if (requested > account.MaxAllocSize) account.MaxAllocSize = requested;
        return block;
    }

    /// <summary>
    /// Takes <paramref name="pointer"/> back from whoever had it: kept for its account's next request
    /// of its class, or, on a card, held for the call this thread is making for that account, whose
    /// stream may still read it, until the call ends. A block of an account that has closed goes back:
    /// nothing is left to reuse it.
    /// </summary>
    internal void Free(IntPtr pointer)
    {
        if (pointer == IntPtr.Zero) return;
        var scope = t_scope;
        List<(IntPtr, long)>? release = null;
        Block block;
        using (_gate.Hold())
        {
            if (!_blocks.Remove(pointer, out block)) return;
            Observer?.Invoke(new Event(false, OnCard, pointer, block.Requested - block.ReleasedRequested, block.Size - block.ReleasedBytes, Fresh: false));
            var account = block.Account;
            account.InUse -= block.Size - block.ReleasedBytes;
            account.Requested -= block.Requested - block.ReleasedRequested;
            if (block.HandedOver) account.HandedOver -= block.Size - block.ReleasedBytes;
            if (block.Released is not null || block.FirstGone)
            {
                // Parts of it went back while the rest was in use, so what is left is not a block to
                // keep whole: it goes back to the arena as those parts did -- on a card where this
                // thread's call charges its account, once the card is done with the work the call
                // queued, waited for outside the lock.
                if (OnCard && scope is not null && scope.Charges(account)) goto waitForTheCard;
                UncarveRest(pointer, block);
                return;
            }
            if (!account.Closed)
            {
                if (OnCard && scope is not null && scope.Charges(account))
                    scope.Hold(account, pointer, block.Size, block.Source, block.Call);
                else if (!OnCard && !account.KeepsLargeBlocks && block.Source == Source.Arena && block.Size >= LargeBlock)
                {
                    // A large host tensor made outside a run -- a checkpoint's, a batch's -- goes back
                    // to the system as it goes, as one from the C runtime's heap would: nothing says
                    // another of its size follows, and carving one again costs what the heap's does.
                    account.Blocks--;
                    account.Arena!.Uncarve(pointer, block.Size);
                    account.Arena.Decommit(block.Size, out var runs);
                    account.Shrinkages += runs;
                }
                else
                    account.Keep(pointer, block.Size, block.Source, block.Call);
                return;
            }
            account.Blocks--;
            release = GoneFromClosed(account, pointer, block.Size, block.Source);
        }
        Release(release);
        return;

    waitForTheCard:
        AwaitCard();
        using (_gate.Hold()) UncarveRest(pointer, block);
    }

    /// <summary>What is left of <paramref name="block"/>, at <paramref name="pointer"/>, once parts of
    /// it went back while the rest was in use, back to the arena those parts went back to. Under the
    /// lock.</summary>
    private void UncarveRest(IntPtr pointer, in Block block)
    {
        var account = block.Account;
        var arena = ArenaOf(block.Source, account);
        foreach (var (from, to) in Outside(block.Released, 0, block.Size)) arena.Uncarve(pointer + (nint)from, to - from);
        account.Blocks--;
        if (account.Closed) Emptied(account, block.Source);
    }

    /// <summary>A block of closed <paramref name="account"/> let go of: back where it came from, and
    /// the account's arena released once it holds nothing. Under the lock; what is of its own is
    /// answered for the caller to release outside it.</summary>
    private List<(IntPtr, long)>? GoneFromClosed(Account account, IntPtr block, long size, Source source)
    {
        switch (source)
        {
            case Source.Own:
                account.Shrinkages++;
                return [(block, size)];
            case Source.Small:
                _small!.Uncarve(block, size);
                account.Shrinkages++;
                ShedSmall();
                return null;
            default:
                account.Arena!.Uncarve(block, size);
                account.Shrinkages++;
                if (account.Arena.IsEmpty) account.Arena.ReleaseEmptyChunks();
                else account.Arena.DecommitAll(out _);
                return null;
        }
    }

    /// <summary>
    /// Marks <paramref name="pointer"/>, a block <paramref name="account"/> took, as handed over: an
    /// output its session's run handed to the caller, which a device-memory budget counts with what
    /// the context holds from then on rather than against the session's runs. Its statistics still
    /// count it until it is let go of. Answers whether it was such a block.
    /// </summary>
    internal bool HandOver(Account account, IntPtr pointer)
    {
        using (_gate.Hold())
        {
            ref var block = ref _blocks.Find(pointer);
            if (Unsafe.IsNullRef(ref block) || block.Account != account || block.HandedOver) return false;
            block.HandedOver = true;
            account.HandedOver += block.Size - block.ReleasedBytes;
            return true;
        }
    }

    /// <summary>Whether <paramref name="pointer"/> is a block in use that hands back parts of itself
    /// (<see cref="ReleaseRange"/>): one carved from an arena, its account's or the card's arena of
    /// small blocks.</summary>
    internal bool ReleasesRanges(IntPtr pointer)
    {
        using (_gate.Hold())
        {
            ref var block = ref _blocks.Find(pointer);
            return !Unsafe.IsNullRef(ref block) && block.Source != Source.Own;
        }
    }

    /// <summary>
    /// Hands back the part of <paramref name="pointer"/>, a block in use, from byte
    /// <paramref name="offset"/> for <paramref name="length"/> bytes — on to the block's end where
    /// <paramref name="toTheEnd"/> — which nothing reads any more while the rest of the block is
    /// still in use: the memory of an input a run consumed that none of the outputs standing on it
    /// covers. The whole units of the block's arena lying inside it — 4 KiB pages on the host,
    /// granules on a card, 512 bytes in the card's arena of small blocks — go back to the arena:
    /// still committed, for the account's next request to be carved from, and shed as what the
    /// account keeps is. All but the block's first unit, which stays carved for as long as the block
    /// is out: the block is known by its address until it is let go of, and a block carved there
    /// meanwhile could not be told from it. Where that unit is a granule of its own, as in an
    /// account's arena on a card, its memory goes back to the card all the same, its address staying
    /// the block's. On a card,
    /// where this thread's call charges the block's account, the card is waited for first — outside
    /// the lock, which every allocation and free on the device takes: work the call queued may still
    /// read the range. Answers the bytes handed back that lie in the part —
    /// up to what was asked for of the block, where it runs on to the end, not the rest of the last
    /// unit — none for a block of its own from the device, and none for a part handed back already.
    /// </summary>
    internal long ReleaseRange(IntPtr pointer, long offset, long length, bool toTheEnd)
    {
        if (pointer == IntPtr.Zero || offset < 0 || (length <= 0 && !toTheEnd)) return 0;
        var scope = t_scope;
        for (var waited = false; ; waited = true)
        {
            if (waited) AwaitCard();
            using var gate = _gate.Hold();
            ref var block = ref _blocks.Find(pointer);
            if (Unsafe.IsNullRef(ref block) || block.Source == Source.Own) return 0;
            var account = block.Account;
            var arena = ArenaOf(block.Source, account);
            var unit = block.Source == Source.Small ? SmallUnit : OnCard ? _backing!.Granule : HostPage;
            var start = (offset + unit - 1) / unit * unit;
            var end = toTheEnd ? block.Size : Math.Min(block.Size, (offset + length) / unit * unit);
            if (end <= start) return 0;
            var pieces = Outside(block.Released, Math.Max(start, unit), end);
            // The first unit's memory goes back only where it is a granule of its own.
            var first = start == 0 && !block.FirstGone && OnCard && block.Source == Source.Arena;
            if (pieces.Count == 0 && !first) return 0;
            if (!waited && OnCard && scope is not null && scope.Charges(account)) continue;
            if (first && arena.DecommitCarved(pointer, unit) == unit)
            {
                block.FirstGone = true;
                account.Shrinkages++;
                pieces.Insert(0, (0, unit));
            }
            else if (pieces.Count == 0)
                return 0;
            long released = 0, inside = 0;
            var partEnd = toTheEnd ? block.Requested : offset + length;
            foreach (var (from, to) in pieces)
            {
                if (from >= unit) arena.Uncarve(pointer + (nint)from, to - from);
                released += to - from;
                inside += Math.Max(0, Math.Min(to, partEnd) - Math.Max(from, offset));
                var asked = Math.Max(0, Math.Min(to, block.Requested) - from);
                Observer?.Invoke(new Event(false, OnCard, pointer + (nint)from, asked, to - from, Fresh: false));
                block.ReleasedRequested += asked;
                account.Requested -= asked;
            }
            if (block.FirstGone) pieces.RemoveAll(piece => piece.Start == 0);
            if (pieces.Count > 0) block.Released = Merged(block.Released, pieces);
            block.ReleasedBytes += released;
            account.InUse -= released;
            if (block.HandedOver) account.HandedOver -= released;
            // Nothing of a closed account's is kept: what went back to its arena goes back to the
            // device.
            if (account.Closed) Emptied(account, block.Source);
            return inside;
        }
    }

    /// <summary>The arena a block from <paramref name="source"/> of <paramref name="account"/>'s
    /// was carved from: the card's arena of small blocks, or the account's own.</summary>
    private Arena ArenaOf(Source source, Account account) => source == Source.Small ? _small! : account.Arena!;

    /// <summary>What of closed <paramref name="account"/>'s went back to the arena of
    /// <paramref name="source"/> goes back to the device. Under the lock.</summary>
    private void Emptied(Account account, Source source)
    {
        account.Shrinkages++;
        if (source == Source.Small) ShedSmall();
        else if (account.Arena!.IsEmpty) account.Arena.ReleaseEmptyChunks();
        else account.Arena.DecommitAll(out _);
    }

    /// <summary>The stretches of [<paramref name="start"/>, <paramref name="end"/>) outside every
    /// stretch of <paramref name="taken"/>, which are in order and do not overlap.</summary>
    private static List<(long Start, long End)> Outside(List<(long Start, long End)>? taken, long start, long end)
    {
        List<(long, long)> outside = [];
        var at = start;
        foreach (var (from, to) in taken ?? [])
        {
            if (to <= at) continue;
            if (from >= end) break;
            if (from > at) outside.Add((at, from));
            at = Math.Max(at, to);
        }
        if (at < end) outside.Add((at, end));
        return outside;
    }

    /// <summary><paramref name="taken"/> and <paramref name="more"/>, in order, adjacent stretches
    /// joined.</summary>
    private static List<(long Start, long End)> Merged(List<(long Start, long End)>? taken, List<(long Start, long End)> more)
    {
        List<(long Start, long End)> merged = [];
        foreach (var (from, to) in (taken ?? []).Concat(more).OrderBy(r => r.Start))
        {
            if (merged.Count > 0 && merged[^1].End >= from) merged[^1] = (merged[^1].Start, Math.Max(merged[^1].End, to));
            else merged.Add((from, to));
        }
        return merged;
    }

    /// <summary>A block of its own from the device itself, or null where it has not that much
    /// free.</summary>
    private IntPtr Fresh(long size)
    {
        if (OnCard) return CudaRuntime.Allocate(_device, size);
        try
        {
            // Sixty-four bytes, as ONNX Runtime aligns its own host blocks for its kernels.
            return (IntPtr)NativeMemory.AlignedAlloc((nuint)size, 64);
        }
        catch (OutOfMemoryException)
        {
            return IntPtr.Zero;
        }
    }

    /// <summary>Waits for the work the card has in hand, as handing back memory some of that work may
    /// still read must.</summary>
    /// <exception cref="InvalidOperationException">The card could not be waited for: nothing is handed
    /// back that its work may still read.</exception>
    private void AwaitCard()
    {
        if (_backing is { } backing ? backing.AwaitDevice() : CudaRuntime.Synchronize(_device)) return;
        throw new InvalidOperationException(
            $"Shorokoo's allocator could not wait for CUDA device {_device} to finish the work it has in hand, "
            + "so it hands back none of the memory that work may still read.");
    }

    /// <summary>Whether this allocator's lock is taken, by any thread.</summary>
    internal bool Locked => _gate.Taken;

    /// <summary>Hands <paramref name="blocks"/> of their own, each with its size class, back to the
    /// device. Outside the lock: on a card each waits for the work the card has in hand.</summary>
    private void Release(List<(IntPtr Block, long Size)>? blocks)
    {
        if (blocks is null) return;
        foreach (var (block, _) in blocks)
        {
            if (OnCard) CudaRuntime.Release(_device, block);
            else NativeMemory.AlignedFree((void*)block);
        }
    }

    // ---- shedding ----

    /// <summary>
    /// Brings what <paramref name="account"/> holds from the device down by <paramref name="excess"/>
    /// where it can: the granules of its arena no block is over, idle longest first; then its kept
    /// blocks, longest kept first, their granules handed back as they come free. Under the lock;
    /// blocks of their own are answered for the caller to release outside it.
    /// </summary>
    private List<(IntPtr, long)> Shed(Account account, long excess)
    {
        List<(IntPtr, long)> release = [];
        if (excess <= 0) return release;
        // What is cheapest to have again goes first: kept blocks of their own, and kept small blocks,
        // which go back to the card's arena of small blocks still committed -- longest kept first.
        while (excess > 0 && account.GiveWayOldest(fromArena: false, out var block, out var size, out var from))
        {
            if (from == Source.Own) release.Add((block, size));
            else Uncarved(account, block, size, from);
            account.Shrinkages++;
            excess -= size;
        }
        // Then the account's arena: its granules no block is over, idle longest first, and then its
        // kept blocks, longest kept first, each one's granules handed back as it gives way.
        if (account.Arena is { } arena)
        {
            excess -= arena.Decommit(excess, out var runs);
            account.Shrinkages += runs;
            while (excess > 0 && account.GiveWayOldest(fromArena: true, out var block, out var size, out _))
            {
                arena.Uncarve(block, size);
                excess -= arena.Decommit(excess, out runs);
                account.Shrinkages += runs;
            }
        }
        ShedSmall();
        return release;
    }


    /// <summary>Everything <paramref name="account"/> keeps, and on a card what the call
    /// <paramref name="scope"/> holds for it, back to the device. Under the lock.</summary>
    private List<(IntPtr, long)> ShedAll(Account account, Scope? scope)
    {
        if (scope is not null)
            foreach (var (block, size, source, call) in scope.TakeAllHeld(account))
                account.Keep(block, size, source, call);
        var release = Shed(account, long.MaxValue);
        if (account.Arena is { } arena)
        {
            arena.DecommitAll(out var runs);
            account.Shrinkages += runs;
        }
        return release;
    }

    /// <summary>A kept block of <paramref name="account"/>'s carved back into the arena it came from,
    /// its memory staying committed there.</summary>
    private void Uncarved(Account account, IntPtr block, long size, Source source)
    {
        if (source == Source.Small) _small!.Uncarve(block, size);
        else account.Arena!.Uncarve(block, size);
    }

    /// <summary>The card's arena of small blocks keeps no more committed than at its busiest.</summary>
    private void ShedSmall()
    {
        if (_small is null) return;
        var spare = _small.CommittedBytes - _small.MaxBusyBytes;
        if (spare > 0) _small.Decommit(spare, out _);
    }

    /// <summary>
    /// Hands back to the device what <paramref name="account"/> and <see cref="Placements"/> keep:
    /// what a run that hands its memory back does as it ends.
    /// </summary>
    internal void ReleaseCached(Account account)
    {
        List<(IntPtr, long)> release;
        using (_gate.Hold())
        {
            release = ShedAll(account, scope: null);
            release.AddRange(ShedAll(Placements, scope: null));
            _small?.DecommitAll(out _);
        }
        Release(release);
    }

    /// <summary>Hands back everything kept on the device, in every account still open — and what the
    /// call <paramref name="scope"/> holds on a card, the card being done with it — and answers how
    /// many bytes that was.</summary>
    internal long ReleaseEverythingCached(Scope? scope = null)
    {
        List<(IntPtr Block, long Size)> release = [];
        long bytes;
        using (_gate.Hold()) bytes = ReleaseEverythingCachedUnderLock(release, scope);
        Release(release);
        return bytes;
    }

    /// <summary><see cref="ReleaseEverythingCached"/>, under the lock, blocks of their own added to
    /// <paramref name="release"/>.</summary>
    private long ReleaseEverythingCachedUnderLock(List<(IntPtr, long)> release, Scope? scope)
    {
        // What goes back is what the device stops holding: the memory the arenas hand back, and the
        // blocks of their own released -- not what the accounts stop keeping, since a small block
        // kept goes back into the card's arena of small blocks and a host block into its
        // account's, and leaves the device only with its granule.
        var from = release.Count;
        var before = CommittedBytes;
        foreach (var account in _accounts) release.AddRange(ShedAll(account, scope));
        _small?.DecommitAll(out _);
        var own = 0L;
        for (var i = from; i < release.Count; i++) own += release[i].Item2;
        return before - CommittedBytes + own;
    }

    /// <summary>What the device's arenas hold committed: every open account's, and the card's arena
    /// of small blocks. Under the lock.</summary>
    private long CommittedBytes => _accounts.Sum(account => account.Arena?.CommittedBytes ?? 0) + (_small?.CommittedBytes ?? 0);

    /// <summary>
    /// Hands back to every device — the host and each card this process has allocated on —
    /// everything every account keeps there for reuse, and answers how many bytes that was. Nothing
    /// in use is touched, nor what a call on a card is still holding for its own reuse.
    /// </summary>
    internal static long ReleaseEverywhere()
    {
        CachingAllocator[] allocators;
        lock (_registryGate) allocators = [.. _byDevice.Values];
        return allocators.Sum(allocator => allocator.ReleaseEverythingCached());
    }

    // Every account opened and not yet closed, for a device that has run out.
    private readonly HashSet<Account> _accounts = [];

    /// <summary>
    /// Closes <paramref name="account"/>, whose session is gone: what it kept goes back to the
    /// device, and each block it still has out — an output its caller keeps — goes back as it is let
    /// go of.
    /// </summary>
    internal void Close(Account account)
    {
        List<(IntPtr, long)> release;
        using (_gate.Hold())
        {
            if (account.Closed) return;
            account.Closed = true;
            _accounts.Remove(account);
            release = ShedAll(account, scope: null);
            if (account.Arena is { IsEmpty: true } arena) arena.ReleaseEmptyChunks();
        }
        Release(release);
    }

    /// <summary>The figures of <paramref name="account"/>, in the form a session reports its
    /// allocator's.</summary>
    internal ArenaStatistics Statistics(Account account)
    {
        using (_gate.Hold())
        {
            return new ArenaStatistics(
                InUseBytes: account.InUse,
                LimitBytes: account.LimitUnderLock ?? -1,
                MaxAllocSizeBytes: account.MaxAllocSize,
                MaxInUseBytes: account.MaxInUse,
                AllocationCount: account.Allocations,
                ArenaExtensionCount: account.Blocks,
                ArenaShrinkageCount: account.Shrinkages,
                ReserveCount: 0,
                TotalAllocatedBytes: account.HeldBytes,
                RequestedInUseBytes: account.Requested);
        }
    }

    /// <summary>
    /// What a session's allocations on one device come to, and what bounds them. Every figure is
    /// changed under the allocator's lock.
    /// </summary>
    internal sealed class Account
    {
        internal Account(CachingAllocator allocator, string name, bool keepsLargeBlocks = true)
        {
            Allocator = allocator;
            Name = name;
            KeepsLargeBlocks = keepsLargeBlocks;
            using (allocator._gate.Hold()) allocator._accounts.Add(this);
        }

        internal CachingAllocator Allocator { get; }

        /// <summary>
        /// Whether a block of <see cref="LargeBlock"/> or more carved from the account's arena on the
        /// host is kept for the next request of its class as it is let go of; where not, its memory
        /// goes back to the system at once.
        /// </summary>
        internal bool KeepsLargeBlocks { get; }

        internal string Name { get; }

        /// <summary>
        /// The most this account may have out and kept, not counting what it has handed over; null
        /// for no limit. Set before a run by the device-memory budget the run is under.
        /// </summary>
        internal long? Limit
        {
            get { using (Allocator._gate.Hold()) return _limit; }
            set { using (Allocator._gate.Hold()) _limit = value; }
        }

        private long? _limit;

        /// <summary><see cref="Limit"/>, for a caller holding the allocator's lock.</summary>
        internal long? LimitUnderLock => _limit;

        internal long InUse;
        internal long SpanPeak;
        internal long SpanStart;
        internal long Requested;
        internal long HandedOver;
        internal long Cached;
        internal long Held;
        internal long MaxInUse;
        internal long MaxAllocSize;
        internal long Allocations;
        internal long Blocks;
        internal long Shrinkages;
        internal long Refusals;
        internal bool Closed;

        /// <summary>A store of held blocks a call of the account's has finished with, for the next
        /// one to hold its blocks in.</summary>
        internal HeldBlocks? SpareHeld;

        /// <summary>The account's own arena, made as its first block of an arena's class is asked
        /// for.</summary>
        internal Arena? Arena;

        /// <summary>What counts against <see cref="Limit"/> of what is out: all of it but what was
        /// handed over.</summary>
        internal long Charged => InUse - HandedOver;

        /// <summary>What the account keeps beyond what it has out: kept blocks, blocks a call holds,
        /// and its arena's committed granules no block is over.</summary>
        internal long KeptBytes => Cached + Held + (Arena?.IdleBytes ?? 0);

        /// <summary>What the account holds from the device: what it has out, and what it keeps.</summary>
        internal long HeldBytes => InUse + KeptBytes;

        /// <summary>The most it may hold from the device: the most one of its calls has used.</summary>
        internal long Bound => MaxUsed;

        /// <summary>What the account's current call has used: what it had in use as the call
        /// began, and every block the call was handed, each counted once.</summary>
        internal long Used;

        /// <summary>
        /// The most <see cref="Used"/> has reached over the account's calls: what it holds from the
        /// device is kept at or under it, so a call like the busiest one finds every block it needs.
        /// </summary>
        internal long MaxUsed;

        /// <summary>The number of the account's current call: a block handed to it carries the
        /// call it was counted in, so it is counted in a call once.</summary>
        internal long CallNumber;

        // For the device's own account, how many session calls had ended on the device when its call
        // began.
        internal long CallsSeen;

        /// <summary>A call begins: what is in use now is what it starts out using.</summary>
        internal void BeginCall()
        {
            CallNumber++;
            Used = InUse;
        }

        // The blocks kept for reuse, by size class -- those up to 64 KiB by the class's index, the rest
        // by size; each class's oldest first, each block with when it was kept (_clock), where its
        // memory came from, and the call it was last counted in.
        private readonly Kept?[] _small = new Kept?[129];
        private readonly Dictionary<long, Kept> _kept = [];
        private long _clock;
        private long _keptCount;

        /// <summary>Starts measuring the most this account has out beyond what it has out now
        /// (<see cref="Peak"/>): what a run asks of it, its outputs included.</summary>
        internal void BeginPeak()
        {
            using (Allocator._gate.Hold()) SpanStart = SpanPeak = InUse;
        }

        /// <summary>The most this account has had out since <see cref="BeginPeak"/>, beyond what it
        /// had out then.</summary>
        internal long Peak
        {
            get { using (Allocator._gate.Hold()) return SpanPeak - SpanStart; }
        }

        /// <summary>The blocks of one size class kept for reuse, in the order they were kept.</summary>
        private sealed class Kept
        {
            internal (IntPtr Block, long Since, long Call, Source Source)[] Items = new (IntPtr, long, long, Source)[4];
            internal int Head;
            internal int Count;

            internal void Push((IntPtr, long, long, Source) item)
            {
                if (Count == Items.Length)
                {
                    var grown = new (IntPtr, long, long, Source)[Items.Length * 2];
                    for (var i = 0; i < Count; i++) grown[i] = Items[(Head + i) & (Items.Length - 1)];
                    Items = grown;
                    Head = 0;
                }
                Items[(Head + Count) & (Items.Length - 1)] = item;
                Count++;
            }

            internal (IntPtr Block, long Since, long Call, Source Source) PopNewest()
            {
                Count--;
                return Items[(Head + Count) & (Items.Length - 1)];
            }

            internal (IntPtr Block, long Since, long Call, Source Source) PopOldest()
            {
                var item = Items[Head];
                Head = (Head + 1) & (Items.Length - 1);
                Count--;
                return item;
            }

            internal long OldestSince => Items[Head].Since;
        }

        /// <summary>The kept blocks of class <paramref name="size"/>, made where
        /// <paramref name="make"/> asks and there are none yet.</summary>
        private Kept? ClassOf(long size, bool make)
        {
            if (size <= 64L << 10)
            {
                var index = (int)(size >> 9);
                return _small[index] ?? (make ? _small[index] = new Kept() : null);
            }
            if (_kept.TryGetValue(size, out var kept)) return kept;
            return make ? _kept[size] = new Kept() : null;
        }

        internal void Keep(IntPtr block, long size, Source source, long call)
        {
            var kept = ClassOf(size, make: true)!;
            kept.Push((block, ++_clock, call, source));
            Cached += size;
            _keptCount++;
        }

        /// <summary>The block of <paramref name="size"/> kept last, if one is kept, and the call it was
        /// last counted in.</summary>
        internal bool TakeKept(long size, out IntPtr block, out long call)
        {
            if (ClassOf(size, make: false) is { Count: > 0 } kept)
            {
                (block, _, call, _) = kept.PopNewest();
                Cached -= size;
                _keptCount--;
                return true;
            }
            block = IntPtr.Zero;
            call = -1;
            return false;
        }

        /// <summary>The block of the smallest class larger than <paramref name="size"/> carved from the
        /// account's arena, the one of it kept last, taken out to give way, with its class and the
        /// call it was last counted in.</summary>
        internal bool TakeKeptLarger(long size, out IntPtr block, out long larger, out long call)
        {
            block = IntPtr.Zero;
            call = -1;
            Kept? best = null;
            long bestSize = 0;
            void Consider(long classSize, Kept? kept)
            {
                if (kept is null || kept.Count == 0 || classSize <= size || kept.Items[kept.Head].Source != Source.Arena) return;
                if (best is null || classSize < bestSize)
                {
                    best = kept;
                    bestSize = classSize;
                }
            }
            Consider(64L << 10, _small[^1]);
            foreach (var (classSize, kept) in _kept) Consider(classSize, kept);
            larger = bestSize;
            if (best is null) return false;
            (block, _, call, _) = best.PopNewest();
            Cached -= larger;
            _keptCount--;
            Blocks--;
            return true;
        }

        /// <summary>The block kept longest — carved from the account's arena where
        /// <paramref name="fromArena"/>, of any other source where not — taken out to give way, with
        /// its class and where its memory came from.</summary>
        internal bool GiveWayOldest(bool fromArena, out IntPtr block, out long size, out Source from)
        {
            block = IntPtr.Zero;
            size = 0;
            from = Source.Own;
            if (_keptCount == 0) return false;
            Kept? oldest = null;
            long oldestSize = 0;
            void Consider(long classSize, Kept? kept)
            {
                if (kept is null || kept.Count == 0 || (kept.Items[kept.Head].Source == Source.Arena) != fromArena) return;
                if (oldest is null || kept.OldestSince < oldest.OldestSince)
                {
                    oldest = kept;
                    oldestSize = classSize;
                }
            }
            for (var index = 1; index < _small.Length; index++) Consider((long)index << 9, _small[index]);
            foreach (var (classSize, kept) in _kept) Consider(classSize, kept);
            size = oldestSize;
            if (oldest is null) return false;
            (block, _, _, from) = oldest.PopOldest();
            Cached -= size;
            _keptCount--;
            Blocks--;
            return true;
        }

    }

    // ---- what a thread is charging ----

    [ThreadStatic]
    private static Scope? t_scope;

    /// <summary>
    /// Charges what this thread allocates through <paramref name="host"/>'s and
    /// <paramref name="card"/>'s allocators to those accounts until the answer is disposed — the
    /// length of one call into ONNX Runtime that builds or runs a session, or makes a tensor. What
    /// the call lets go of on a card is held for its own reuse until then. Either may be null, for
    /// <see cref="Placements"/>.
    /// </summary>
    internal static ChargeScope Charge(Account? host, Account? card)
    {
        var scope = new Scope(host, card, t_scope);
        if (host is not null) Begin(host);
        if (card is not null) Begin(card);
        t_scope = scope;
        return new ChargeScope(scope);
    }

    /// <summary>A call charging <paramref name="account"/> begins, and with it what the account's
    /// call uses — whatever other call is under way, so that calls overlapping one another without
    /// end still count what they use afresh, rather than everything any of them ever used.</summary>
    private static void Begin(Account account)
    {
        using (account.Allocator._gate.Hold()) account.BeginCall();
    }

    /// <summary>The disposable <see cref="Charge"/> answers.</summary>
    internal readonly struct ChargeScope(Scope scope) : IDisposable
    {
        public void Dispose()
        {
            t_scope = scope.Outer;
            scope.End();
        }
    }

    /// <summary>One call's charging: its accounts, and the blocks it let go of on a card.</summary>
    internal sealed class Scope(Account? host, Account? card, Scope? outer)
    {
        internal Scope? Outer { get; } = outer;

        private HeldBlocks? _held;

        // The bytes of the blocks it holds.
        private long _holding;

        internal Account? AccountOn(CachingAllocator allocator)
            => card is not null && card.Allocator == allocator ? card
                : host is not null && host.Allocator == allocator ? host
                : null;

        internal bool Charges(Account account) => account == card || account == host;

        /// <summary>Whether this call holds blocks it let go of for <paramref name="account"/>. Under
        /// the allocator's lock.</summary>
        internal bool Holds(Account account) => account == card && _holding > 0;

        /// <summary>Holds <paramref name="block"/>, which this call let go of, for its own reuse.
        /// Under the allocator's lock.</summary>
        internal void Hold(Account account, IntPtr block, long size, Source source, long call)
        {
            if (_held is null)
            {
                _held = account.SpareHeld ?? new HeldBlocks();
                account.SpareHeld = null;
            }
            _held.Push(size, block, source, call);
            account.Held += size;
            _holding += size;
        }

        /// <summary>Every block this call let go of for <paramref name="account"/>, taken out of its
        /// hold. Under the allocator's lock.</summary>
        internal List<(IntPtr Block, long Size, Source Source, long Call)> TakeAllHeld(Account account)
        {
            List<(IntPtr, long, Source, long)> blocks = [];
            if (account != card || _held is null) return blocks;
            while (_held.TakeAny(out var block, out var size, out var source, out var call))
            {
                blocks.Add((block, size, source, call));
                account.Held -= size;
                _holding -= size;
            }
            return blocks;
        }

        /// <summary>A block of <paramref name="size"/> this call let go of, if it holds one. Under the
        /// allocator's lock.</summary>
        internal bool TakeHeld(Account account, long size, out IntPtr block, out long call)
        {
            if (account == card && _held is not null && _held.Pop(size, out block, out call))
            {
                account.Held -= size;
                _holding -= size;
                return true;
            }
            block = IntPtr.Zero;
            call = -1;
            return false;
        }

        /// <summary>The call is over, its stream done with what it held: kept, or back to the device
        /// where the account has closed meanwhile; and each account sheds what it holds beyond the
        /// most one of its calls has used.</summary>
        internal void End()
        {
            if (card is not null) EndOn(card);
            if (host is not null) EndOn(host);
        }

        private void EndOn(Account account)
        {
            var allocator = account.Allocator;
            List<(IntPtr, long)> release = [];
            using (allocator._gate.Hold())
            {
                allocator._callsEnded++;
                if (account == card && _held is not null)
                {
                    while (_held.TakeAny(out var block, out var size, out var source, out var call))
                    {
                        account.Held -= size;
                        _holding -= size;
                        if (!account.Closed)
                        {
                            account.Keep(block, size, source, call);
                            continue;
                        }
                        account.Blocks--;
                        if (allocator.GoneFromClosed(account, block, size, source) is { } own) release.AddRange(own);
                    }
                    account.SpareHeld ??= _held;
                    _held = null;
                }
                if (!account.Closed)
                    release.AddRange(allocator.Shed(account, account.HeldBytes - account.Bound));
            }
            allocator.Release(release);
        }
    }

    /// <summary>
    /// The blocks a call on a card let go of, held for its own reuse until it ends: those of up to a
    /// mebibyte by their class's index, larger ones by size, each class's newest on top; and the
    /// classes holding any, so the call's end visits those alone. An account keeps one for its next
    /// call, so a call holds blocks without allocating.
    /// </summary>
    internal sealed class HeldBlocks
    {
        private sealed class Pile(long size)
        {
            internal readonly long Size = size;
            internal (IntPtr Block, Source Source, long Call)[] Items = new (IntPtr, Source, long)[4];
            internal int Count;
            internal bool Listed;
        }

        private readonly Pile?[] _small = new Pile?[(int)(CardSmallTo >> 9) + 1];
        private readonly Dictionary<long, Pile> _large = [];
        private readonly List<Pile> _listed = [];

        private Pile? PileOf(long size, bool make)
        {
            if (size <= CardSmallTo)
            {
                var index = (int)(size >> 9);
                return _small[index] ?? (make ? _small[index] = new Pile(size) : null);
            }
            if (_large.TryGetValue(size, out var pile)) return pile;
            return make ? _large[size] = new Pile(size) : null;
        }

        internal void Push(long size, IntPtr block, Source source, long call)
        {
            var pile = PileOf(size, make: true)!;
            if (!pile.Listed)
            {
                pile.Listed = true;
                _listed.Add(pile);
            }
            if (pile.Count == pile.Items.Length) Array.Resize(ref pile.Items, pile.Count * 2);
            pile.Items[pile.Count++] = (block, source, call);
        }

        internal bool Pop(long size, out IntPtr block, out long call)
        {
            if (PileOf(size, make: false) is { Count: > 0 } pile)
            {
                (block, _, call) = pile.Items[--pile.Count];
                return true;
            }
            block = IntPtr.Zero;
            call = -1;
            return false;
        }

        /// <summary>Any block held, taken out, with its class.</summary>
        internal bool TakeAny(out IntPtr block, out long size, out Source source, out long call)
        {
            while (_listed.Count > 0)
            {
                var pile = _listed[^1];
                if (pile.Count > 0)
                {
                    (block, source, call) = pile.Items[--pile.Count];
                    size = pile.Size;
                    return true;
                }
                pile.Listed = false;
                _listed.RemoveAt(_listed.Count - 1);
            }
            block = IntPtr.Zero;
            size = 0;
            source = Source.Own;
            call = -1;
            return false;
        }
    }

    /// <summary>
    /// The allocator's lock, taken by every allocation and every free: one compare-and-swap to take
    /// it and a store to let it go, where a monitor costs two such swaps. Taken as
    /// <c>using (_gate.Hold())</c>; a struct, so that <c>lock</c> cannot be written over it by
    /// mistake. Not reentrant — nothing done under it takes it again — and a thread finding it taken
    /// spins, then yields, until it is let go of.
    /// </summary>
    internal struct Gate
    {
        private int _taken;

        [UnscopedRef]
        internal Held Hold()
        {
            if (Interlocked.CompareExchange(ref _taken, 1, 0) != 0) Wait();
            return new Held(ref _taken);
        }

        /// <summary>Whether the lock is taken now.</summary>
        internal bool Taken => Volatile.Read(ref _taken) != 0;

        [MethodImpl(MethodImplOptions.NoInlining)]
        private void Wait()
        {
            var spin = new SpinWait();
            do spin.SpinOnce();
            while (Volatile.Read(ref _taken) != 0 || Interlocked.CompareExchange(ref _taken, 1, 0) != 0);
        }

        /// <summary>The lock taken, let go of as this is disposed.</summary>
        internal readonly ref struct Held
        {
            private readonly ref int _taken;

            internal Held(ref int taken) => _taken = ref taken;

            public void Dispose() => Volatile.Write(ref _taken, 0);
        }
    }
}
