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
/// shares, and a runtime is handed it once (<see cref="ClaimRuntime"/>): the sessions of either
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
/// a block is carved over it. A block let go of stays carved and is kept for the next request of its
/// class, so a loop's runs find their blocks waiting with no work at all; a request of a class no
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
/// take it past that mark, the kept blocks give way first, longest kept first, their memory staying
/// committed for the request to be carved from; what is still over the mark is shed, the granules
/// idle longest handed back first — at once on the host, and on a card as the call that asked ends,
/// since a granule must not go back while work the card has in hand may read it. What is kept also
/// goes back when a run that hands back its memory ends (<see cref="ReleaseCached(Account)"/>), when
/// a session closes, before an account would refuse an allocation for want of room, before the
/// device would refuse one, and when the program asks (<see cref="ReleaseEverywhere"/>).</para>
///
/// <para><b>Accounts.</b> What a session allocates is charged to its <see cref="Account"/>: the
/// thread building or running it charges that account for the length of the call
/// (<see cref="Charge"/>), and ONNX Runtime allocates on the thread that called it — measured, every
/// allocation of a session's construction and runs, on the host with four intra-op threads and on
/// the card alike. Each block remembers its account, so a block let go of on any thread — a
/// collected output, say — is credited to the account that took it. An account carries a session's
/// statistics (<see cref="Statistics"/>) and, on a card, the limit a device-memory budget leaves its
/// runs (<see cref="Account.Limit"/>), which it enforces as each block is asked for. Anything not
/// charged to a session — what the framework places on a card — goes to the device's own
/// account (<see cref="Placements"/>).</para>
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
    private readonly object _gate = new();
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
    /// where its memory came from, the account's call it was last counted in, and whether that
    /// account has handed it over (<see cref="HandOver"/>).</summary>
    private struct Block
    {
        internal long Size;
        internal long Requested;
        internal Account Account;
        internal long Call;
        internal Source Source;
        internal bool HandedOver;
    }

    /// <summary>
    /// The blocks out, by address: open addressing with linear probing and backward-shift deletion,
    /// which hands a block in and out with a hash and a probe or two and no allocation.
    /// </summary>
    private sealed class BlockTable
    {
        private IntPtr[] _keys = new IntPtr[256];
        private Block[] _values = new Block[256];
        private int _count;

        private int Slot(IntPtr key) => (int)(((ulong)key >> 6) * 0x9E3779B97F4A7C15UL >> 40) & (_keys.Length - 1);

        internal void Add(IntPtr key, in Block value)
        {
            if ((_count + 1) * 2 > _keys.Length) Grow();
            var mask = _keys.Length - 1;
            var i = Slot(key);
            while (_keys[i] != IntPtr.Zero && _keys[i] != key) i = (i + 1) & mask;
            if (_keys[i] == IntPtr.Zero) _count++;
            _keys[i] = key;
            _values[i] = value;
        }

        internal ref Block Find(IntPtr key)
        {
            var mask = _keys.Length - 1;
            for (var i = Slot(key); _keys[i] != IntPtr.Zero; i = (i + 1) & mask)
                if (_keys[i] == key) return ref _values[i];
            return ref Unsafe.NullRef<Block>();
        }

        internal bool Remove(IntPtr key, out Block value)
        {
            var mask = _keys.Length - 1;
            var i = Slot(key);
            while (_keys[i] != key)
            {
                if (_keys[i] == IntPtr.Zero)
                {
                    value = default;
                    return false;
                }
                i = (i + 1) & mask;
            }
            value = _values[i];
            // Each later entry of the run that could no longer be found moves into the gap.
            for (var j = (i + 1) & mask; _keys[j] != IntPtr.Zero; j = (j + 1) & mask)
            {
                var home = Slot(_keys[j]);
                var reachable = i <= j ? home > i && home <= j : home > i || home <= j;
                if (reachable) continue;
                _keys[i] = _keys[j];
                _values[i] = _values[j];
                i = j;
            }
            _keys[i] = IntPtr.Zero;
            _values[i] = default;
            _count--;
            return true;
        }

        private void Grow()
        {
            var keys = _keys;
            var values = _values;
            _keys = new IntPtr[keys.Length * 2];
            _values = new Block[keys.Length * 2];
            _count = 0;
            for (var i = 0; i < keys.Length; i++)
                if (keys[i] != IntPtr.Zero) Add(keys[i], values[i]);
        }
    }

    private CachingAllocator(int device)
    {
        _device = device;
        if (device == Host)
            _backing = HostMemory.Supported ? HostMemory.Instance : null;
        else if (CardMemory.For(device) is { } card)
        {
            _backing = card;
            _small = new Arena(card, unit: 512, chunkBytes: 256L << 20);
        }
        Placements = new Account(this, "placements");
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
    /// Whether the runtime whose environment is <paramref name="environment"/> still has to be handed
    /// this allocator: true the first time a runtime is named, for the caller to register it there,
    /// and false after, however many backends bind that runtime. A runtime keeps one allocator per
    /// device, so a second registration would take the device over from the first.
    /// </summary>
    internal bool ClaimRuntime(IntPtr environment)
    {
        lock (_gate) return _runtimes.Add(environment);
    }

    /// <summary>Whether this is a card's allocator.</summary>
    internal bool OnCard => _device != Host;

    /// <summary>The account of what the framework places on the device itself.</summary>
    internal Account Placements { get; }

    /// <summary>A new account, for a session about to be built.</summary>
    internal Account Open(string name) => new(this, name);

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
    {
        string? refusal;
        try
        {
            var block = Of(state).Allocate(size > long.MaxValue ? long.MaxValue : (long)size, out refusal);
            if (block != IntPtr.Zero) return block;
        }
        catch (Exception failure)
        {
            refusal = $"Failed to allocate {size} bytes: Shorokoo's allocator failed ({failure.Message}).";
        }
        Write(refusal ?? $"Failed to allocate {size} bytes.", reason, capacity);
        return IntPtr.Zero;
    }

    /// <summary><paramref name="text"/> into <paramref name="buffer"/> as UTF-8, cut short where it
    /// does not fit, and NUL-terminated.</summary>
    private static void Write(string text, byte* buffer, int capacity)
    {
        if (buffer is null || capacity <= 0) return;
        var span = new Span<byte>(buffer, capacity - 1);
        System.Text.Unicode.Utf8.FromUtf16(text, span, out _, out var written, replaceInvalidSequences: true, isFinalBlock: true);
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
    /// before its limit refuses, everything the device keeps before the device does.
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
        var size = ClassOf(bytes);
        var source = SourceOf(size);
        List<(IntPtr, long)>? release = null;
        var carved = IntPtr.Zero;
        lock (_gate)
        {
            if (account == Placements) BeginPlacementsCall();
            if (scope is not null && scope.TakeHeld(account, size, out var held, out var call))
                return Hand(held, size, bytes, account, source, call);
            if (account.TakeKept(size, out held, out call))
                return Hand(held, size, bytes, account, source, call);
            // What the account would hold from the device with one more block of this class.
            var excess = account.HeldBytes + size - Math.Max(account.MaxUsed, account.Used + size);
            if (account.LimitUnderLock is { } limit && account.Charged + account.KeptBytes + size > limit)
            {
                // Under a budget what it keeps goes back first -- kept, and on a card what this call
                // let go of, which handing back waits for the card to be done with -- and then only
                // what it has out counts.
                if (OnCard) CudaRuntime.Synchronize(_device);
                release = ShedAll(account, scope);
                if (account.Charged + size > limit)
                {
                    account.Refusals++;
                    refusal = $"Failed to allocate {bytes} bytes {Where}: {Allocator} may hold {limit} "
                        + "bytes there for this session, what the device-memory budget "
                        + $"(DeviceMemorySettings.LimitBytes) leaves its run, and it holds {account.Charged}.";
                }
            }
            if (refusal is null && source != Source.Own)
            {
                carved = CarveFor(account, size, source, excess, out var counted);
                if (carved == IntPtr.Zero)
                {
                    // The device is full: what is kept anywhere on it goes back -- on a card once the
                    // card is done with it -- and the request is tried once more.
                    if (OnCard) CudaRuntime.Synchronize(_device);
                    ReleaseEverythingCachedUnderLock(release ??= []);
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
                    // On the host what is over the account's mark goes back at once; on a card as the
                    // call charging the account ends (Scope.End), where one does.
                    if (!OnCard || scope is null || !scope.Charges(account))
                        (release ??= []).AddRange(Shed(account, account.HeldBytes - account.Bound));
                }
            }
            else if (refusal is null && (!OnCard || scope is null || !scope.Charges(account)))
                (release ??= []).AddRange(Shed(account, excess));
        }
        Release(release);
        if (carved != IntPtr.Zero) return carved;
        if (refusal is not null) return IntPtr.Zero;
        var own = Fresh(size);
        if (own == IntPtr.Zero)
        {
            // The device is full: what is kept anywhere on it goes back, and the request is tried
            // once more.
            ReleaseEverythingCached();
            own = Fresh(size);
            if (own == IntPtr.Zero)
            {
                lock (_gate) account.Refusals++;
                refusal = Refusal(bytes);
                return IntPtr.Zero;
            }
        }
        lock (_gate)
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
    /// or else of what its kept blocks of that arena give way to, longest kept first, until as much as
    /// the excess has; and only then over granules committed for it, what is over the mark going back
    /// as the caller sheds it. Zero where the device has no memory for it.
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
            counted = call;
        }
        for (long givenWay = 0; block == IntPtr.Zero && givenWay < excess && account.GiveWayOldest(fromArena: true, out var kept, out var keptSize, out _); givenWay += keptSize)
        {
            arena.Uncarve(kept, keptSize);
            block = arena.Carve(size, mayCommit: false);
        }
        return block != IntPtr.Zero ? block : arena.Carve(size, mayCommit: true);
    }

    /// <summary>The words a request the device has no memory for is refused in.</summary>
    private string Refusal(long bytes) => OnCard
        ? $"Failed to allocate {bytes} bytes {Where}: the card has no such block free "
          + $"(cudaMalloc refused it), with everything {Allocator} kept for reuse handed back."
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

    /// <summary>Where this allocator's memory is, as a refusal says it.</summary>
    private string Where => OnCard ? $"on CUDA device {_device}" : "of host memory";

    /// <summary>What a refusal calls this allocator: on a card, by the name
    /// <c>AllocationFailureReport</c> reads as the card's.</summary>
    private string Allocator => OnCard ? "Shorokoo's cuda_allocator" : "Shorokoo's host allocator";

    /// <summary>Records <paramref name="block"/> as handed to <paramref name="account"/>, and as used
    /// by the account's call where it was not counted in it already (<paramref name="call"/> is the
    /// call it was last counted in).</summary>
    private IntPtr Hand(IntPtr block, long size, long requested, Account account, Source source, long call)
    {
        if (call != account.CallNumber)
        {
            account.Used += size;
            if (account.Used > account.MaxUsed) account.MaxUsed = account.Used;
        }
        _blocks.Add(block, new Block { Size = size, Requested = requested, Account = account, Source = source, Call = account.CallNumber });
        account.InUse += size;
        account.Requested += requested;
        account.Allocations++;
        if (account.InUse > account.MaxInUse) account.MaxInUse = account.InUse;
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
        lock (_gate)
        {
            if (!_blocks.Remove(pointer, out var block)) return;
            var account = block.Account;
            account.InUse -= block.Size;
            account.Requested -= block.Requested;
            if (block.HandedOver) account.HandedOver -= block.Size;
            if (!account.Closed)
            {
                if (OnCard && scope is not null && scope.Charges(account))
                    scope.Hold(account, pointer, block.Size, block.Source, block.Call);
                else
                    account.Keep(pointer, block.Size, block.Source, block.Call);
                return;
            }
            account.Blocks--;
            release = GoneFromClosed(account, pointer, block.Size, block.Source);
        }
        Release(release);
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
        lock (_gate)
        {
            ref var block = ref _blocks.Find(pointer);
            if (Unsafe.IsNullRef(ref block) || block.Account != account || block.HandedOver) return false;
            block.HandedOver = true;
            account.HandedOver += block.Size;
            return true;
        }
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
        lock (_gate)
        {
            release = ShedAll(account, scope: null);
            release.AddRange(ShedAll(Placements, scope: null));
            _small?.DecommitAll(out _);
        }
        Release(release);
    }

    /// <summary>Hands back everything kept on the device, in every account still open, and answers
    /// how many bytes that was.</summary>
    private long ReleaseEverythingCached()
    {
        List<(IntPtr Block, long Size)> release = [];
        long bytes;
        lock (_gate) bytes = ReleaseEverythingCachedUnderLock(release);
        Release(release);
        return bytes;
    }

    /// <summary><see cref="ReleaseEverythingCached"/>, under the lock, blocks of their own added to
    /// <paramref name="release"/>.</summary>
    private long ReleaseEverythingCachedUnderLock(List<(IntPtr, long)> release)
    {
        long before = 0, after = 0;
        foreach (var account in _accounts)
        {
            before += account.HeldBytes - account.InUse;
            release.AddRange(ShedAll(account, scope: null));
            after += account.HeldBytes - account.InUse;
        }
        var smallBefore = _small?.IdleBytes ?? 0;
        _small?.DecommitAll(out _);
        return before - after + smallBefore;
    }

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
        lock (_gate)
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
        lock (_gate)
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
        internal Account(CachingAllocator allocator, string name)
        {
            Allocator = allocator;
            Name = name;
            lock (allocator._gate) allocator._accounts.Add(this);
        }

        internal CachingAllocator Allocator { get; }

        internal string Name { get; }

        /// <summary>
        /// The most this account may have out and kept, not counting what it has handed over; null
        /// for no limit. Set before a run by the device-memory budget the run is under.
        /// </summary>
        internal long? Limit
        {
            get { lock (Allocator._gate) return _limit; }
            set { lock (Allocator._gate) _limit = value; }
        }

        private long? _limit;

        /// <summary><see cref="Limit"/>, for a caller holding the allocator's lock.</summary>
        internal long? LimitUnderLock => _limit;

        internal long InUse;
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

        // The calls charging the account now, and for the device's own account how many session calls
        // had ended on the device when its call began.
        internal int Calls;
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

    /// <summary>A call charging <paramref name="account"/> begins: its first, where no other is
    /// under way, begins what the account's call uses.</summary>
    private static void Begin(Account account)
    {
        lock (account.Allocator._gate)
            if (account.Calls++ == 0) account.BeginCall();
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

        private Dictionary<long, Stack<(IntPtr Block, Source Source, long Call)>>? _held;

        internal Account? AccountOn(CachingAllocator allocator)
            => card is not null && card.Allocator == allocator ? card
                : host is not null && host.Allocator == allocator ? host
                : null;

        internal bool Charges(Account account) => account == card || account == host;

        /// <summary>Holds <paramref name="block"/>, which this call let go of, for its own reuse.
        /// Under the allocator's lock.</summary>
        internal void Hold(Account account, IntPtr block, long size, Source source, long call)
        {
            _held ??= [];
            if (!_held.TryGetValue(size, out var blocks)) _held[size] = blocks = new();
            blocks.Push((block, source, call));
            account.Held += size;
        }

        /// <summary>Every block this call let go of for <paramref name="account"/>, taken out of its
        /// hold. Under the allocator's lock.</summary>
        internal List<(IntPtr Block, long Size, Source Source, long Call)> TakeAllHeld(Account account)
        {
            List<(IntPtr, long, Source, long)> blocks = [];
            if (account != card || _held is null) return blocks;
            foreach (var (size, held) in _held)
            {
                foreach (var (block, source, call) in held)
                {
                    blocks.Add((block, size, source, call));
                    account.Held -= size;
                }
                held.Clear();
            }
            return blocks;
        }

        /// <summary>A block of <paramref name="size"/> this call let go of, if it holds one. Under the
        /// allocator's lock.</summary>
        internal bool TakeHeld(Account account, long size, out IntPtr block, out long call)
        {
            if (account == card && _held is not null && _held.TryGetValue(size, out var blocks) && blocks.TryPop(out var held))
            {
                account.Held -= size;
                block = held.Block;
                call = held.Call;
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
            lock (allocator._gate)
            {
                account.Calls--;
                allocator._callsEnded++;
                if (account == card && _held is not null)
                {
                    foreach (var (block, size, source, call) in TakeAllHeld(account))
                    {
                        if (!account.Closed)
                        {
                            account.Keep(block, size, source, call);
                            continue;
                        }
                        account.Blocks--;
                        if (allocator.GoneFromClosed(account, block, size, source) is { } own) release.AddRange(own);
                    }
                    _held = null;
                }
                if (!account.Closed)
                    release.AddRange(allocator.Shed(account, account.HeldBytes - account.Bound));
            }
            allocator.Release(release);
        }
    }
}
