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
/// included — and has no hook for one output's memory apart from the rest. An arena's tensor keeps
/// the arena alive, so an output kept alive would keep the run's every block alive. Here every
/// block is one allocation of its own, never a piece of a larger one: a kept output holds its own
/// block and nothing else, and is never copied after its run.</para>
///
/// <para><b>Caching.</b> A block that is let go of is kept for the next request of its size class
/// rather than handed back, so a loop's runs find their blocks waiting, as they would in an arena.
/// Sizes are rounded up to a class — to 512 bytes up to a mebibyte, to an eighth of the power of
/// two above — so a block is reused across requests that differ a little. An account never holds
/// from the device more than the most it has ever had in use at once: a request no kept block
/// serves hands back the blocks kept longest, as many as the new block takes it past that mark —
/// on a card as the call that asked ends, since handing a block back there waits for the card — so
/// an account fed one shape after another holds what its busiest moment needed rather than a block
/// of every class it ever used. The price is a block from the device for each request of a shape
/// it no longer keeps blocks for, where an arena would carve one out of a larger free region. What is kept also goes back when a run that
/// hands back its memory ends (<see cref="ReleaseCached(Account)"/>), when a session closes,
/// before an account would refuse an allocation for want of room, before the device would refuse
/// one, and when the program asks (<see cref="ReleaseEverywhere"/>).</para>
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
    private readonly Dictionary<IntPtr, Block> _blocks = [];
    private readonly HashSet<IntPtr> _runtimes = [];

    /// <summary>What a block is: its size class, what was asked for, the account it is charged to,
    /// and whether that account has handed it over (<see cref="HandOver"/>).</summary>
    private struct Block
    {
        internal long Size;
        internal long Requested;
        internal Account Account;
        internal bool HandedOver;
    }

    private CachingAllocator(int device)
    {
        _device = device;
        Placements = new Account(this, "placements");
        State = GCHandle.ToIntPtr(GCHandle.Alloc(this));
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

    private static CachingAllocator Of(IntPtr state) => (CachingAllocator)GCHandle.FromIntPtr(state).Target!;

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

    /// <summary>The size class <paramref name="bytes"/> is served from: a multiple of 512 bytes up
    /// to a mebibyte, and above it a multiple of an eighth of the power of two below.</summary>
    internal static long SizeClass(long bytes)
    {
        if (bytes <= 1L << 20) return (bytes + 511) & ~511L;
        var step = 1L << (63 - BitOperations.LeadingZeroCount((ulong)bytes) - 3);
        return (bytes + step - 1) & ~(step - 1);
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
        var size = SizeClass(bytes);
        List<(IntPtr, long)>? release = null;
        lock (_gate)
        {
            if ((scope is not null && scope.TakeHeld(account, size, out var held)) || account.TakeCached(size, out held))
                return Hand(held, size, bytes, account);
            // A fresh block: what the account kept longest goes back, as much as the block takes
            // what it holds from the device past the most it has had in use at once. On a card, as
            // the call charging the account ends (Scope.End), where one does: handing a block back
            // waits for the card, which mid-run would stall the run's queued work.
            release = OnCard && scope is not null && scope.Charges(account)
                ? []
                : account.Shed(account.InUse + account.Cached + account.Held + size
                               - Math.Max(account.MaxInUse, account.InUse + size));
            if (account.LimitUnderLock is { } limit && account.Charged + account.Cached + account.Held + size > limit)
            {
                // What it keeps goes back first -- cached, and on a card what this call let go of,
                // which handing back waits for the stream to be done with -- and then only what it
                // has out counts.
                release.AddRange(account.EmptyCache());
                if (scope is not null) release.AddRange(scope.EmptyHeld(account));
                if (account.Charged + size > limit)
                {
                    account.Refusals++;
                    refusal = $"Failed to allocate {bytes} bytes {Where}: {Allocator} may hold {limit} "
                        + "bytes there for this session, what the device-memory budget "
                        + $"(DeviceMemorySettings.LimitBytes) leaves its run, and it holds {account.Charged}.";
                }
            }
        }
        Release(release);
        if (refusal is not null) return IntPtr.Zero;
        var block = Fresh(size);
        if (block == IntPtr.Zero)
        {
            // The device is full: what is kept anywhere on it goes back, and the request is tried
            // once more.
            ReleaseEverythingCached();
            block = Fresh(size);
            if (block == IntPtr.Zero)
            {
                lock (_gate) account.Refusals++;
                refusal = OnCard
                    ? $"Failed to allocate {bytes} bytes {Where}: the card has no such block free "
                      + $"(cudaMalloc refused it), with everything {Allocator} kept for reuse handed back."
                    : $"Failed to allocate {bytes} bytes {Where}: the process could not commit that much "
                      + "more (a bad allocation).";
                return IntPtr.Zero;
            }
        }
        lock (_gate)
        {
            account.Blocks++;
            return Hand(block, size, bytes, account);
        }
    }

    /// <summary>Where this allocator's memory is, as a refusal says it.</summary>
    private string Where => OnCard ? $"on CUDA device {_device}" : "of host memory";

    /// <summary>What a refusal calls this allocator: on a card, by the name
    /// <c>AllocationFailureReport</c> reads as the card's.</summary>
    private string Allocator => OnCard ? "Shorokoo's cuda_allocator" : "Shorokoo's host allocator";

    /// <summary>Records <paramref name="block"/> as handed to <paramref name="account"/>.</summary>
    private IntPtr Hand(IntPtr block, long size, long requested, Account account)
    {
        _blocks[block] = new Block { Size = size, Requested = requested, Account = account };
        account.InUse += size;
        account.Requested += requested;
        account.Allocations++;
        if (account.InUse > account.MaxInUse) account.MaxInUse = account.InUse;
        if (requested > account.MaxAllocSize) account.MaxAllocSize = requested;
        return block;
    }

    /// <summary>
    /// Takes <paramref name="pointer"/> back from whoever had it: into the cache of the account it
    /// was charged to, or, on a card, held for the call this thread is making for that account,
    /// whose stream may still read it, until the call ends. A block of an account that has closed
    /// goes back to the device: nothing is left to reuse it.
    /// </summary>
    internal void Free(IntPtr pointer)
    {
        if (pointer == IntPtr.Zero) return;
        var scope = t_scope;
        Block block;
        lock (_gate)
        {
            if (!_blocks.Remove(pointer, out block)) return;
            var account = block.Account;
            account.InUse -= block.Size;
            account.Requested -= block.Requested;
            if (block.HandedOver) account.HandedOver -= block.Size;
            if (!account.Closed)
            {
                if (OnCard && scope is not null && scope.Charges(account))
                    scope.Hold(account, pointer, block.Size);
                else
                    account.Cache(pointer, block.Size);
                return;
            }
            account.Blocks--;
            account.Shrinkages++;
        }
        Release([(pointer, block.Size)]);
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
            ref var block = ref CollectionsMarshal.GetValueRefOrNullRef(_blocks, pointer);
            if (Unsafe.IsNullRef(ref block) || block.Account != account || block.HandedOver) return false;
            block.HandedOver = true;
            account.HandedOver += block.Size;
            return true;
        }
    }

    /// <summary>
    /// A block from the device itself, or null where it has not that much free. On the host, a
    /// block of <see cref="HostPages.From"/> or more is pages of its own from the operating system,
    /// which go back to it whole when the block is handed back: a C runtime's heap serves such a
    /// block out of memory it keeps for itself once the block is freed, so the process would go on
    /// holding what this allocator had handed back.
    /// </summary>
    private IntPtr Fresh(long size)
    {
        if (OnCard) return CudaRuntime.Allocate(_device, size);
        if (HostPages.Serve(size)) return HostPages.Allocate(size);
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

    /// <summary>Hands <paramref name="blocks"/>, each with its size class, back to the device.
    /// Outside the lock: on a card each waits for the work the card has in hand.</summary>
    private void Release(List<(IntPtr Block, long Size)>? blocks)
    {
        if (blocks is null) return;
        foreach (var (block, size) in blocks)
        {
            if (OnCard) CudaRuntime.Release(_device, block);
            else if (HostPages.Serve(size)) HostPages.Release(block, size);
            else NativeMemory.AlignedFree((void*)block);
        }
    }

    /// <summary>
    /// Hands back to the device what <paramref name="account"/> and <see cref="Placements"/> hold
    /// cached: what a run that hands its memory back does as it ends.
    /// </summary>
    internal void ReleaseCached(Account account)
    {
        List<(IntPtr, long)> release;
        lock (_gate)
        {
            release = account.EmptyCache();
            release.AddRange(Placements.EmptyCache());
        }
        Release(release);
    }

    /// <summary>Hands back everything cached on the device, in every account still open, and
    /// answers how many bytes that was.</summary>
    private long ReleaseEverythingCached()
    {
        List<(IntPtr Block, long Size)> release = [];
        lock (_gate)
        {
            foreach (var account in _accounts)
                release.AddRange(account.EmptyCache());
        }
        Release(release);
        return release.Sum(block => block.Size);
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
            release = account.EmptyCache();
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
                TotalAllocatedBytes: account.InUse + account.Cached + account.Held,
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
        /// The most this account may have out and cached, not counting what it has handed over; null
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

        /// <summary>What counts against <see cref="Limit"/> of what is out: all of it but what was
        /// handed over.</summary>
        internal long Charged => InUse - HandedOver;

        // The blocks kept for reuse, by size class, each list oldest first; a class with none is
        // not here. Each block carries when it was kept, by _clock.
        private readonly Dictionary<long, LinkedList<(IntPtr Block, long Kept)>> _kept = [];
        private long _clock;

        internal void Cache(IntPtr block, long size)
        {
            if (!_kept.TryGetValue(size, out var blocks)) _kept[size] = blocks = new();
            blocks.AddLast((block, ++_clock));
            Cached += size;
        }

        /// <summary>The block of <paramref name="size"/> kept last, if one is kept.</summary>
        internal bool TakeCached(long size, out IntPtr block)
        {
            if (_kept.TryGetValue(size, out var blocks))
            {
                block = blocks.Last!.Value.Block;
                blocks.RemoveLast();
                if (blocks.Count == 0) _kept.Remove(size);
                Cached -= size;
                return true;
            }
            block = IntPtr.Zero;
            return false;
        }

        /// <summary>The blocks kept longest, taken out of the cache until they come to at least
        /// <paramref name="bytes"/> or none is left, for the caller to hand back.</summary>
        internal List<(IntPtr, long)> Shed(long bytes)
        {
            List<(IntPtr, long)> blocks = [];
            while (bytes > 0 && _kept.Count > 0)
            {
                var (size, oldest) = _kept.MinBy(kept => kept.Value.First!.Value.Kept);
                blocks.Add((oldest.First!.Value.Block, size));
                oldest.RemoveFirst();
                if (oldest.Count == 0) _kept.Remove(size);
                Cached -= size;
                bytes -= size;
            }
            Shrinkages += blocks.Count;
            Blocks -= blocks.Count;
            return blocks;
        }

        /// <summary>Every cached block, taken out of the cache, for the caller to hand back.</summary>
        internal List<(IntPtr, long)> EmptyCache()
        {
            List<(IntPtr, long)> blocks = [];
            foreach (var (size, kept) in _kept)
                foreach (var (block, _) in kept)
                    blocks.Add((block, size));
            _kept.Clear();
            Shrinkages += blocks.Count;
            Blocks -= blocks.Count;
            Cached = 0;
            return blocks;
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
        t_scope = scope;
        return new ChargeScope(scope);
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

        private Dictionary<long, Stack<IntPtr>>? _held;

        internal Account? AccountOn(CachingAllocator allocator)
            => card is not null && card.Allocator == allocator ? card
                : host is not null && host.Allocator == allocator ? host
                : null;

        internal bool Charges(Account account) => account == card || account == host;

        /// <summary>Holds <paramref name="block"/>, which this call let go of, for its own reuse.
        /// Under the allocator's lock.</summary>
        internal void Hold(Account account, IntPtr block, long size)
        {
            _held ??= [];
            if (!_held.TryGetValue(size, out var blocks)) _held[size] = blocks = new Stack<IntPtr>();
            blocks.Push(block);
            account.Held += size;
        }

        /// <summary>Every block this call let go of for <paramref name="account"/>, taken out of its
        /// hold for the caller to hand back. Under the allocator's lock.</summary>
        internal List<(IntPtr, long)> EmptyHeld(Account account)
        {
            List<(IntPtr, long)> blocks = [];
            if (account != card || _held is null) return blocks;
            foreach (var (size, held) in _held)
            {
                foreach (var block in held)
                {
                    blocks.Add((block, size));
                    account.Held -= size;
                }
                held.Clear();
            }
            account.Shrinkages += blocks.Count;
            account.Blocks -= blocks.Count;
            return blocks;
        }

        /// <summary>A block of <paramref name="size"/> this call let go of, if it holds one. Under the
        /// allocator's lock.</summary>
        internal bool TakeHeld(Account account, long size, out IntPtr block)
        {
            if (account == card && _held is not null && _held.TryGetValue(size, out var blocks) && blocks.TryPop(out block))
            {
                account.Held -= size;
                return true;
            }
            block = IntPtr.Zero;
            return false;
        }

        /// <summary>The call is over, its stream done with what it held: into the cache, or back to
        /// the device where the account has closed meanwhile; and each account sheds what it keeps
        /// beyond the most it has had in use at once.</summary>
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
                if (account == card && _held is not null)
                {
                    foreach (var (size, blocks) in _held)
                        foreach (var block in blocks)
                        {
                            account.Held -= size;
                            if (!account.Closed)
                            {
                                account.Cache(block, size);
                                continue;
                            }
                            account.Blocks--;
                            account.Shrinkages++;
                            release.Add((block, size));
                        }
                    _held = null;
                }
                if (!account.Closed)
                    release.AddRange(account.Shed(account.InUse + account.Cached + account.Held - account.MaxInUse));
            }
            allocator.Release(release);
        }
    }
}

/// <summary>
/// Host memory as pages of its own from the operating system, for the blocks of the host's
/// <see cref="CachingAllocator"/> of <see cref="From"/> bytes or more: <c>VirtualAlloc</c> on
/// Windows, <c>mmap</c> on Linux, each handed back whole. The pages are zeroed on their first touch.
/// </summary>
internal static partial class HostPages
{
    /// <summary>The smallest block served this way: the granularity Windows reserves address space
    /// in, below which a C runtime's heap serves a block out of its own pages anyway.</summary>
    internal const long From = 64L << 10;

    /// <summary>Whether a block of <paramref name="size"/> is pages of its own.</summary>
    internal static bool Serve(long size) => size >= From && (OperatingSystem.IsWindows() || OperatingSystem.IsLinux());

    /// <summary><paramref name="size"/> bytes of fresh pages, or null where the system has not that
    /// much to commit.</summary>
    internal static IntPtr Allocate(long size)
    {
        if (OperatingSystem.IsWindows()) return VirtualAlloc(IntPtr.Zero, (nuint)size, MemCommit | MemReserve, PageReadWrite);
        var pages = mmap(IntPtr.Zero, (nuint)size, ProtRead | ProtWrite, MapPrivate | MapAnonymous, -1, 0);
        return pages == MapFailed ? IntPtr.Zero : pages;
    }

    /// <summary>Hands the <paramref name="size"/> bytes at <paramref name="block"/>, which
    /// <see cref="Allocate"/> answered, back to the system.</summary>
    internal static void Release(IntPtr block, long size)
    {
        if (OperatingSystem.IsWindows()) VirtualFree(block, 0, MemRelease);
        else munmap(block, (nuint)size);
    }

    private const uint MemCommit = 0x1000;
    private const uint MemReserve = 0x2000;
    private const uint MemRelease = 0x8000;
    private const uint PageReadWrite = 0x04;
    private const int ProtRead = 0x1;
    private const int ProtWrite = 0x2;
    private const int MapPrivate = 0x02;
    private const int MapAnonymous = 0x20;
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
}
