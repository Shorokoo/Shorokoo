using System.IO.MemoryMappedFiles;
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
/// <para><b>Held for the life of the process</b>, as every tensor that came from it frees itself
/// through it.</para>
/// </summary>
internal sealed unsafe class CachingAllocator
{
    /// <summary>The key the host's allocator is held under, beside the CUDA devices'.</summary>
    private const int Host = -1;

    /// <summary>The <c>OrtAllocator</c> version this fills in: the fields up to <c>Shrink</c>, ONNX
    /// Runtime 1.25's, none of which past <c>Info</c> it implements.</summary>
    private const uint OrtAllocatorVersion = 25;

    // The native OrtAllocator: its version and seven entry points, then a slot of Shorokoo's own --
    // past every field the version announces, so ONNX Runtime never reads it -- holding the handle
    // of the managed allocator the entry points answer for.
    private const int NativeSize = 9 * 8;
    private const int SelfSlot = 8 * 8;

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
    /// whether that account has handed it over (<see cref="HandOver"/>), and whether it is an
    /// overdraft (<see cref="Overdraft"/>), which counts against nothing.</summary>
    private struct Block
    {
        internal long Size;
        internal long Requested;
        internal Account Account;
        internal bool HandedOver;
        internal bool Overdraft;
    }

    private CachingAllocator(int device)
    {
        _device = device;
        _info = InfoFor(device);
        _infoPointer = OrtEnvironment.PointerOf(_info);
        Placements = new Account(this, "placements");

        _native = (IntPtr)NativeMemory.AllocZeroed(NativeSize);
        *(uint*)_native = OrtAllocatorVersion;
        ((IntPtr*)_native)[1] = (IntPtr)(delegate* unmanaged<IntPtr, nuint, IntPtr>)&AllocCallback;
        ((IntPtr*)_native)[2] = (IntPtr)(delegate* unmanaged<IntPtr, IntPtr, void>)&FreeCallback;
        ((IntPtr*)_native)[3] = (IntPtr)(delegate* unmanaged<IntPtr, IntPtr>)&InfoCallback;
        *(IntPtr*)(_native + SelfSlot) = GCHandle.ToIntPtr(GCHandle.Alloc(this));
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

    // ---- the entry points ONNX Runtime calls ----

    private static CachingAllocator Of(IntPtr self)
        => (CachingAllocator)GCHandle.FromIntPtr(*(IntPtr*)(self + SelfSlot)).Target!;

    [UnmanagedCallersOnly]
    private static IntPtr AllocCallback(IntPtr self, nuint size)
    {
        try
        {
            return Of(self).Allocate(checked((long)size));
        }
        catch
        {
            // Nothing may cross into native code; a null is how an allocator says it could not.
            return IntPtr.Zero;
        }
    }

    [UnmanagedCallersOnly]
    private static void FreeCallback(IntPtr self, IntPtr pointer)
    {
        try
        {
            Of(self).Free(pointer);
        }
        catch
        {
            // As above: a block whose bookkeeping failed is lost rather than the process.
        }
    }

    [UnmanagedCallersOnly]
    private static IntPtr InfoCallback(IntPtr self) => Of(self)._infoPointer;

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
    /// this device (<see cref="Charge"/>) or to <see cref="Placements"/>.
    ///
    /// <para>Never null for a request it cannot serve, since ONNX Runtime does not check: it hands a
    /// null block to the kernel that asked, whose first write is then an illegal access — on a card
    /// one that leaves the device unusable for the rest of the process, measured. So a block the
    /// account's limit has no room for, or the device has none for, is refused another way
    /// (<see cref="Overdraft"/>): the request is served from memory that is not the device's, and the
    /// call this thread is making is stopped at its next node and fails as an allocation
    /// failure.</para>
    /// </summary>
    internal IntPtr Allocate(long bytes)
    {
        if (bytes <= 0) return IntPtr.Zero;
        if (Redirect?.Invoke(OnCard, bytes) is { } redirected && redirected != IntPtr.Zero)
        {
            lock (_gate) _redirected[redirected] = _redirected.GetValueOrDefault(redirected) + 1;
            Observer?.Invoke(new Event(true, OnCard, redirected, bytes, bytes, Fresh: false, Redirected: true));
            return redirected;
        }
        var scope = t_scope;
        var account = scope?.AccountOn(this) ?? Placements;
        var size = SizeClass(bytes);
        List<IntPtr>? release = null;
        lock (_gate)
        {
            if ((scope is not null && scope.TakeHeld(account, size, out var held)) || account.TakeCached(size, out held))
                return Observed(Hand(held, size, bytes, account), bytes, size, fresh: false);
            if (account.LimitUnderLock is { } limit && account.Charged + account.Cached + account.Held + size > limit)
            {
                // What it keeps goes back first -- cached, and on a card what this call let go of,
                // which handing back waits for the stream to be done with -- and then only what it
                // has out counts.
                release = account.EmptyCache();
                if (scope is not null) release.AddRange(scope.EmptyHeld(account));
                if (account.Charged + size > limit)
                {
                    var refusal = new Refusal(bytes, limit, account.Charged, OnCard ? _device : null, DeviceFull: false);
                    Release(release);
                    return Overdraft(scope, account, bytes, refusal);
                }
            }
        }
        Release(release);
        var block = Fresh(size);
        if (block == IntPtr.Zero)
        {
            // The device is full: what is cached anywhere on it goes back, and the request is tried
            // once more.
            ReleaseEverythingCached();
            block = Fresh(size);
            if (block == IntPtr.Zero)
            {
                long charged;
                lock (_gate) charged = account.Charged;
                return Overdraft(scope, account, bytes, new Refusal(bytes, account.Limit, charged, OnCard ? _device : null, DeviceFull: true));
            }
        }
        lock (_gate)
        {
            account.Blocks++;
            return Observed(Hand(block, size, bytes, account), bytes, size, fresh: true);
        }
    }

    // ---- what an investigation watches and steers ----

    /// <summary>One request served, or one block taken back, as <see cref="Observer"/> is told of
    /// it: whether it is a request, on which device, at what address, of how many bytes asked for
    /// and served, whether the device gave the block for it, and whether <see cref="Redirect"/>
    /// answered it.</summary>
    internal readonly record struct Event(
        bool Allocation, bool OnCard, IntPtr Address, long Requested, long Size, bool Fresh, bool Redirected);

    /// <summary>Told of every request every allocator serves and every block it takes back, on the
    /// thread making the call. Null tells nothing.</summary>
    internal static Action<Event>? Observer;

    /// <summary>
    /// Asked first of every request, on the thread making it, with whether it is a card's and how
    /// many bytes it asks for: an address it answers is what the request is served, memory the
    /// answerer owns, and the free of it is let go of here without anything taken back. Zero, or a
    /// null hook, serves the request as usual.
    /// </summary>
    internal static Func<bool, long, IntPtr>? Redirect;

    // How many requests each address Redirect answered still has out.
    private readonly Dictionary<IntPtr, int> _redirected = [];

    private IntPtr Observed(IntPtr block, long requested, long size, bool fresh)
    {
        Observer?.Invoke(new Event(true, OnCard, block, requested, size, fresh, Redirected: false));
        return block;
    }

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
    /// A request <paramref name="refusal"/> says this device will not serve, served all the same,
    /// from memory that is not the device's: on a card, pinned host memory mapped into the card's
    /// address space, which a kernel reads and writes over the bus; on the host, a mapping of a
    /// temporary file of its own, which the operating system pages to that file rather than charging
    /// to the commit limit that refused the block. The call this thread is making is told
    /// (<see cref="Scope.Refuse"/>) and stopped at its next node, so the block lives for one node at
    /// most, never counts against the device, and goes back as soon as it is let go of. Null only
    /// where even that cannot be had.
    /// </summary>
    private IntPtr Overdraft(Scope? scope, Account account, long bytes, Refusal refusal)
    {
        scope?.Refuse(refusal);
        lock (_gate) account.Refusals++;
        var block = OnCard ? CudaInterop.AllocateMappedHost(_device, bytes) : MappedFile(bytes);
        if (block == IntPtr.Zero) return IntPtr.Zero;
        lock (_gate) _blocks[block] = new Block { Size = bytes, Requested = bytes, Account = account, Overdraft = true };
        return block;
    }

    // The host's overdrafts, by address: each a view of a mapped temporary file, deleted as it closes.
    private readonly Dictionary<IntPtr, (MemoryMappedFile File, MemoryMappedViewAccessor View)> _mapped = [];

    /// <summary><paramref name="bytes"/> of a mapping of a temporary file of its own, or null where
    /// there is no room for the file either.</summary>
    private IntPtr MappedFile(long bytes)
    {
        FileStream? backing = null;
        MemoryMappedFile? file = null;
        MemoryMappedViewAccessor? view = null;
        try
        {
            backing = new FileStream(
                Path.Combine(Path.GetTempPath(), "shorokoo-overdraft-" + Guid.NewGuid().ToString("N")),
                FileMode.CreateNew, FileAccess.ReadWrite, FileShare.None, 4096, FileOptions.DeleteOnClose);
            backing.SetLength(bytes);
            file = MemoryMappedFile.CreateFromFile(
                backing, null, bytes, MemoryMappedFileAccess.ReadWrite, HandleInheritability.None, leaveOpen: false);
            backing = null;
            view = file.CreateViewAccessor(0, bytes);
            byte* pointer = null;
            view.SafeMemoryMappedViewHandle.AcquirePointer(ref pointer);
            var block = (IntPtr)(pointer + view.PointerOffset);
            lock (_gate) _mapped[block] = (file, view);
            return block;
        }
        catch (Exception)
        {
            view?.Dispose();
            file?.Dispose();
            backing?.Dispose();
            return IntPtr.Zero;
        }
    }

    /// <summary>Hands an overdraft <paramref name="block"/> back.</summary>
    private void ReleaseOverdraft(IntPtr block)
    {
        if (OnCard)
        {
            CudaInterop.ReleaseMappedHost(_device, block);
            return;
        }
        (MemoryMappedFile File, MemoryMappedViewAccessor View) mapped;
        lock (_gate)
        {
            if (!_mapped.Remove(block, out mapped)) return;
        }
        mapped.View.SafeMemoryMappedViewHandle.ReleasePointer();
        mapped.View.Dispose();
        mapped.File.Dispose();
    }

    /// <summary>
    /// Takes <paramref name="pointer"/> back from whoever had it: into the cache of the account it
    /// was charged to — of <see cref="Placements"/> where that account has closed — or, on a card,
    /// held for the call this thread is making for that account, whose stream may still read it,
    /// until the call ends. An overdraft goes straight back.
    /// </summary>
    internal void Free(IntPtr pointer)
    {
        if (pointer == IntPtr.Zero) return;
        var scope = t_scope;
        lock (_gate)
        {
            if (_redirected.TryGetValue(pointer, out var outstanding))
            {
                if (outstanding == 1) _redirected.Remove(pointer);
                else _redirected[pointer] = outstanding - 1;
                Observer?.Invoke(new Event(false, OnCard, pointer, 0, 0, Fresh: false, Redirected: true));
                return;
            }
            if (!_blocks.Remove(pointer, out var block)) return;
            Observer?.Invoke(new Event(false, OnCard, pointer, block.Requested, block.Size, Fresh: false, Redirected: false));
            if (!block.Overdraft)
            {
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
                return;
            }
        }
        ReleaseOverdraft(pointer);
    }

    /// <summary>
    /// A request a device would not serve: <see cref="Requested"/> bytes, against an account whose
    /// limit (<see cref="Limit"/>) left no room beside the <see cref="Charged"/> bytes it had out, or
    /// on a device that had no such block free (<see cref="DeviceFull"/>); on CUDA device
    /// <see cref="CudaDevice"/>, or the host where it is null.
    /// </summary>
    internal sealed record Refusal(long Requested, long? Limit, long Charged, int? CudaDevice, bool DeviceFull)
    {
        /// <summary>
        /// The failure a call that met this refusal throws, worded so that
        /// <c>AllocationFailureReport</c> recognizes it and names the memory that ran out.
        /// </summary>
        internal InvalidOperationException ToException(string what, Exception? inner)
        {
            var where = CudaDevice is { } device ? $"CUDA device {device}" : "host memory";
            var why = (DeviceFull, CudaDevice) switch
            {
                (false, _) =>
                    $"what it may allocate there is {Limit} bytes, what the device-memory budget "
                    + $"(DeviceMemorySettings.LimitBytes) leaves it, and it had {Charged} out",
                (true, not null) => "the card had no such block free (cudaMalloc refused it)",
                (true, null) => "the process could not commit that much more (a bad allocation)",
            };
            var allocator = CudaDevice is null ? "Shorokoo's host allocator" : "Shorokoo's cuda_allocator";
            return new InvalidOperationException(
                $"Failed to allocate {Requested} bytes on {where} for {what}: {why}. {allocator} served "
                + "the block from memory that is not the device's and stopped the call at its next "
                + "step, so it produced nothing.", inner);
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
            if (Unsafe.IsNullRef(ref block) || block.Account != account || block.HandedOver || block.Overdraft) return false;
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
    /// <see cref="Placements"/>. A request refused on the way (<see cref="Overdraft"/>) stops the run
    /// <paramref name="run"/> belongs to, where there is one, and is what
    /// <see cref="ChargeScope.Refusal"/> answers once the call is over.
    /// </summary>
    internal static ChargeScope Charge(Account? host, Account? card, RunOptions? run = null)
    {
        var scope = new Scope(host, card, run, t_scope);
        t_scope = scope;
        return new ChargeScope(scope);
    }

    /// <summary>The disposable <see cref="Charge"/> answers.</summary>
    internal readonly struct ChargeScope(Scope scope) : IDisposable
    {
        /// <summary>The first request refused during the call, or null.</summary>
        internal Refusal? Refusal => scope.Refusal;

        public void Dispose()
        {
            t_scope = scope.Outer;
            scope.End();
        }
    }

    /// <summary>One call's charging: its accounts, and the blocks it let go of on a card.</summary>
    internal sealed class Scope(Account? host, Account? card, RunOptions? run, Scope? outer)
    {
        internal Scope? Outer { get; } = outer;

        /// <summary>The first request refused during the call.</summary>
        internal Refusal? Refusal { get; private set; }

        /// <summary>Records <paramref name="refusal"/>, and stops the run this call belongs to at its
        /// next node.</summary>
        internal void Refuse(Refusal refusal)
        {
            Refusal ??= refusal;
            if (run is not null) run.Terminate = true;
        }

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
