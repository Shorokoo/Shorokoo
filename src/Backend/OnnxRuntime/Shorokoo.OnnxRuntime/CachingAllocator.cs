using System.Numerics;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Shorokoo.Core.Backends;

namespace Shorokoo.OnnxRuntime;

/// <summary>
/// The allocator every ONNX Runtime session of this backend allocates through on one device — the
/// host, or one CUDA card — and that every tensor the framework places on a card comes from: a
/// caching allocator of Shorokoo's own, registered with the runtime's environment, which each session
/// is built to use in place of the arena it would otherwise make.
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
/// What is kept is handed back to the device when a run that hands back its memory ends
/// (<see cref="ReleaseCached"/>), when a session closes with nothing more to ask, and before an
/// account would refuse an allocation for want of room. Sizes are rounded up to a class — to 512
/// bytes up to a mebibyte, to an eighth of the power of two above — so a block is reused across
/// requests that differ a little.</para>
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
/// <para><b>Refusing.</b> ONNX Runtime holds the allocator through a native library of Shorokoo's
/// (<see cref="NativeAllocator"/>), which forwards each request here. A request this cannot serve —
/// one an account's limit has no room for, or one the device has no memory for — is answered with
/// null and the reason, and the native side throws that reason as ONNX Runtime's own allocators
/// throw theirs: the call that asked fails as an allocation failure, and the device is left as
/// usable as before it.</para>
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

    /// <summary>The allocator for the host, registered on first use.</summary>
    internal static CachingAllocator ForHost() => For(Host);

    /// <summary>The allocator for CUDA device <paramref name="deviceId"/>, registered on first
    /// use.</summary>
    internal static CachingAllocator ForCard(int deviceId) => For(deviceId);

    private static CachingAllocator For(int device)
    {
        lock (_registryGate)
        {
            if (_byDevice.TryGetValue(device, out var existing)) return existing;
            var made = new CachingAllocator(device);
            OrtEnvironment.Register(made._native);
            _byDevice[device] = made;
            return made;
        }
    }

    private readonly int _device;
    private readonly IntPtr _native;
    private readonly OrtMemoryInfo _info;
    private readonly IntPtr _infoPointer;
    private readonly object _gate = new();
    private readonly Dictionary<IntPtr, Block> _blocks = [];

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
        _info = InfoFor(device);
        _infoPointer = OrtEnvironment.PointerOf(_info);
        Placements = new Account(this, "placements");

        _native = NativeAllocator.Create(GCHandle.ToIntPtr(GCHandle.Alloc(this)), &AllocCallback, &FreeCallback, _infoPointer);
        Managed = OrtEnvironment.Wrap(_native);
    }

    /// <summary>The memory info ONNX Runtime matches this allocator to a device by: the host's, or
    /// CUDA device <paramref name="device"/>'s.</summary>
    private static OrtMemoryInfo InfoFor(int device)
    {
        if (device == Host)
            return new OrtMemoryInfo(OrtMemoryInfo.allocatorCPU, OrtAllocatorType.DeviceAllocator, 0, OrtMemType.Default);
        return new OrtMemoryInfo(OrtMemoryInfo.allocatorCUDA, OrtAllocatorType.DeviceAllocator, device, OrtMemType.Default);
    }

    /// <summary>Whether this is a card's allocator.</summary>
    internal bool OnCard => _device != Host;

    /// <summary>This allocator as the managed surface takes one, to make a tensor from.</summary>
    internal OrtAllocator Managed { get; }

    /// <summary>The account of what the framework places on the device itself, and of every block a
    /// closed account still had cached.</summary>
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
        List<IntPtr>? release = null;
        lock (_gate)
        {
            if ((scope is not null && scope.TakeHeld(account, size, out var held)) || account.TakeCached(size, out held))
                return Hand(held, size, bytes, account);
            if (account.LimitUnderLock is { } limit && account.Charged + account.Cached + account.Held + size > limit)
            {
                // What it keeps goes back first -- cached, and on a card what this call let go of,
                // which handing back waits for the stream to be done with -- and then only what it
                // has out counts.
                release = account.EmptyCache();
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
    /// was charged to — of <see cref="Placements"/> where that account has closed — or, on a card,
    /// held for the call this thread is making for that account, whose stream may still read it,
    /// until the call ends.
    /// </summary>
    internal void Free(IntPtr pointer)
    {
        if (pointer == IntPtr.Zero) return;
        var scope = t_scope;
        lock (_gate)
        {
            if (!_blocks.Remove(pointer, out var block)) return;
            var account = block.Account;
            account.InUse -= block.Size;
            account.Requested -= block.Requested;
            if (block.HandedOver) account.HandedOver -= block.Size;
            if (account.Closed)
                Placements.TakeOver(account, pointer, block.Size);
            else if (OnCard && scope is not null && scope.Charges(account))
                scope.Hold(account, pointer, block.Size);
            else
                account.Cache(pointer, block.Size);
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
            ref var block = ref CollectionsMarshal.GetValueRefOrNullRef(_blocks, pointer);
            if (Unsafe.IsNullRef(ref block) || block.Account != account || block.HandedOver) return false;
            block.HandedOver = true;
            account.HandedOver += block.Size;
            return true;
        }
    }

    /// <summary>A block from the device itself, or null where it has not that much free.</summary>
    private IntPtr Fresh(long size)
    {
        if (OnCard) return CudaInterop.Allocate(_device, size);
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

    /// <summary>Hands <paramref name="blocks"/> back to the device. Outside the lock: on a card
    /// each waits for the work the card has in hand.</summary>
    private void Release(List<IntPtr>? blocks)
    {
        if (blocks is null) return;
        foreach (var block in blocks)
        {
            if (OnCard) CudaInterop.Release(_device, block);
            else NativeMemory.AlignedFree((void*)block);
        }
    }

    /// <summary>
    /// Hands back to the device what <paramref name="account"/> and <see cref="Placements"/> hold
    /// cached: what a run that hands its memory back does as it ends.
    /// </summary>
    internal void ReleaseCached(Account account)
    {
        List<IntPtr> release;
        lock (_gate)
        {
            release = account.EmptyCache();
            release.AddRange(Placements.EmptyCache());
        }
        Release(release);
    }

    /// <summary>Hands back everything cached on the device, in every account still open.</summary>
    private void ReleaseEverythingCached()
    {
        List<IntPtr> release = [];
        lock (_gate)
        {
            foreach (var account in _accounts)
                release.AddRange(account.EmptyCache());
        }
        Release(release);
    }

    // Every account opened and not yet closed, for a device that has run out.
    private readonly HashSet<Account> _accounts = [];

    /// <summary>
    /// Closes <paramref name="account"/>, whose session is gone: what it held cached goes to
    /// <see cref="Placements"/>, and each block it still has out goes there as it is let go of.
    /// </summary>
    internal void Close(Account account)
    {
        lock (_gate)
        {
            if (account.Closed) return;
            account.Closed = true;
            _accounts.Remove(account);
            foreach (var (size, blocks) in account.Free)
                foreach (var block in blocks) Placements.TakeOver(account, block, size);
            account.Free.Clear();
            account.Cached = 0;
        }
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

        internal readonly Dictionary<long, Stack<IntPtr>> Free = [];

        internal void Cache(IntPtr block, long size)
        {
            if (!Free.TryGetValue(size, out var blocks)) Free[size] = blocks = new Stack<IntPtr>();
            blocks.Push(block);
            Cached += size;
        }

        /// <summary>Caches <paramref name="block"/>, a block of <paramref name="from"/>, which is
        /// closed, as this account's own.</summary>
        internal void TakeOver(Account from, IntPtr block, long size)
        {
            from.Blocks--;
            Blocks++;
            Cache(block, size);
        }

        internal bool TakeCached(long size, out IntPtr block)
        {
            if (Free.TryGetValue(size, out var blocks) && blocks.TryPop(out block))
            {
                Cached -= size;
                return true;
            }
            block = IntPtr.Zero;
            return false;
        }

        /// <summary>Every cached block, taken out of the cache, for the caller to hand back.</summary>
        internal List<IntPtr> EmptyCache()
        {
            List<IntPtr> blocks = [];
            foreach (var cached in Free.Values)
            {
                blocks.AddRange(cached);
                cached.Clear();
            }
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
        internal List<IntPtr> EmptyHeld(Account account)
        {
            List<IntPtr> blocks = [];
            if (account != card || _held is null) return blocks;
            foreach (var (size, held) in _held)
            {
                foreach (var block in held)
                {
                    blocks.Add(block);
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

        /// <summary>The call is over, its stream done with what it held: into the cache.</summary>
        internal void End()
        {
            if (_held is null || card is null) return;
            var allocator = card.Allocator;
            lock (allocator._gate)
            {
                foreach (var (size, blocks) in _held)
                    foreach (var block in blocks)
                    {
                        card.Held -= size;
                        if (card.Closed) allocator.Placements.TakeOver(card, block, size);
                        else card.Cache(block, size);
                    }
            }
            _held = null;
        }
    }
}
