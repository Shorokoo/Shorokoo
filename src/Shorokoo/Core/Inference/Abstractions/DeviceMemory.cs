namespace Shorokoo.Core.Inference.Abstractions;

/// <summary>
/// How ONNX Runtime's device arena grows when it needs a block it does not already hold.
/// </summary>
public enum ArenaExtendStrategy
{
    /// <summary>
    /// ORT's own default: each extension is at least as large as everything the arena already
    /// holds. Its regions are large, and get split and reused, which is what an unpredictable
    /// series of allocation sizes needs — but on a loop whose sizes have settled the doubling is
    /// pure overshoot, and that is what a long training run ends up holding.
    /// </summary>
    NextPowerOfTwo,

    /// <summary>
    /// Extend by exactly the requested size — Shorokoo's default. A loop whose allocation sizes
    /// have settled then tracks them instead of doubling past them. The cost is that an
    /// exactly-sized region cannot serve a later, larger request: a run whose input shapes keep
    /// growing strands each region it outgrows and can need <b>more</b> memory this way.
    /// </summary>
    SameAsRequested,
}

/// <summary>
/// A snapshot of the CUDA device's memory, in bytes. <see cref="UsedBytes"/> is
/// <see cref="TotalBytes"/> minus <see cref="FreeBytes"/> and so counts <b>every</b>
/// process on the device — a desktop session included — not this process alone.
/// </summary>
/// <param name="UsedBytes">Device memory in use, by all processes.</param>
/// <param name="FreeBytes">Device memory still allocatable.</param>
/// <param name="TotalBytes">The device's usable memory.</param>
public readonly record struct DeviceMemoryReading(long UsedBytes, long FreeBytes, long TotalBytes);

/// <summary>
/// Process-wide device-memory settings and readings for the CUDA backends
/// (<c>Shorokoo.LinuxGPU</c>, <c>Shorokoo.WinGPU</c>). It is the only place in the public
/// API that can influence what the ORT arena does with the card, and the only place that
/// reports how much of the card is gone.
///
/// <para><b>Settings.</b> <see cref="LimitBytes"/> and <see cref="ArenaExtend"/> are read
/// when a session is <i>created</i>, so set them at startup, before the first inference or
/// training call; changing them later leaves already-compiled sessions as they were.
/// <see cref="ShrinkArenaAfterRun"/> is read on every run and takes effect immediately.
/// All three are ignored by the CPU backends. <see cref="ArenaExtend"/> is the one whose
/// default is not ORT's own; the reason is on the property.</para>
///
/// <para><b>Readings.</b> <see cref="Read"/> and <see cref="Sample"/> call the CUDA
/// runtime's <c>cudaMemGetInfo</c> directly and return <c>null</c> when there is no CUDA
/// runtime to call. They read the <i>device</i>, not this process, and cost about a
/// microsecond, so calling one per training step is the way to catch a peak that a
/// half-second <c>nvidia-smi</c> poll steps straight over.</para>
///
/// <code>
/// using Shorokoo.Core.Inference.Abstractions;
///
/// DeviceMemory.LimitBytes = 16L * 1024 * 1024 * 1024;        // 16 GiB budget
///
/// for (int step = 0; step &lt; steps; step++)
/// {
///     checkpoint = rig.TrainStep(checkpoint, inputs);
///     DeviceMemory.Sample();
/// }
/// Console.WriteLine($"peak {DeviceMemory.PeakUsedBytes / (1024 * 1024)} MiB");
/// </code>
///
/// <para><b>This is a process-wide stopgap, not the final shape.</b> One setting for every
/// session in the process, mutable at any time and read at the moment a session is built or
/// run, and readings taken from whichever CUDA device is current for the calling thread —
/// which is device 0 only because that is the device the shipped GPU backends use. A per-<c>ComputeContext</c> settings
/// object is what this should become; until then, treat these as startup configuration
/// and do not expect two contexts to differ.</para>
/// </summary>
public static class DeviceMemory
{
    private static long _peakUsedBytes;

    // A long, not a long?: Nullable<long> is two fields, so neither the write nor the read is
    // atomic, and a session built on another thread mid-assignment could see HasValue with the
    // old value -- a gpu_mem_limit of 0, which the setter exists to refuse. Zero is not a legal
    // limit, so it doubles as "unset" and one Interlocked pair covers both accessors.
    private static long _limitBytes;

    /// <summary>
    /// The upper bound, in bytes, on what the CUDA arena may allocate — ORT's
    /// <c>gpu_mem_limit</c>. <c>null</c> (the default) leaves ORT free to take the whole
    /// card. A step that needs more than this fails with an ORT <c>BFCArena</c> allocation
    /// error rather than eating into what is left of the device, which is what makes a
    /// too-large configuration fail early and visibly instead of starving everything else
    /// on the machine.
    ///
    /// <para>It caps <b>each session's</b> arena, not the process: ORT gives a session its own
    /// CUDA arena, so a process holding several live sessions — a compiled graph plus a rig, say —
    /// can hold this much more than once. Read it as the ceiling on any one session, and halve it
    /// accordingly when two must coexist.</para>
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">A limit of zero or less.</exception>
    public static long? LimitBytes
    {
        get => Interlocked.Read(ref _limitBytes) is var limit && limit != 0 ? limit : null;
        set
        {
            if (value is <= 0)
                throw new ArgumentOutOfRangeException(nameof(value), value, "The device-memory limit must be positive.");
            Interlocked.Exchange(ref _limitBytes, value ?? 0);
        }
    }

    /// <summary>
    /// How the arena extends itself — ORT's <c>arena_extend_strategy</c>. Defaults to
    /// <see cref="ArenaExtendStrategy.SameAsRequested"/>, <b>not</b> to ORT's own
    /// <see cref="ArenaExtendStrategy.NextPowerOfTwo"/>.
    ///
    /// <para>Neither strategy is better in general; the default is a bet on the workload
    /// Shorokoo exists for. A training run feeds one input shape to one compiled step for its
    /// whole length, and on that shape exact-size extension holds about 1.45x less than ORT's
    /// doubling (measured small, and 1.8x was reported on a 24 GiB card). Where several
    /// allocation sizes are in play it is the doubling that holds less, but by 1.06-1.13x — an
    /// order of magnitude less at stake. The one case it loses badly is input shapes that keep
    /// growing without settling, where each outgrown region is stranded: set
    /// <see cref="ArenaExtendStrategy.NextPowerOfTwo"/> if that is your workload and device
    /// memory is tight.</para>
    ///
    /// <para>Whichever you set applies to every session in the process, and a session keeps the
    /// value it was built with, so this is startup configuration.</para>
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">Not one of the strategies.</exception>
    public static ArenaExtendStrategy ArenaExtend
    {
        get => _arenaExtend;
        set
        {
            if (!Enum.IsDefined(value))
                throw new ArgumentOutOfRangeException(nameof(value), value, "Not an arena-extend strategy.");
            _arenaExtend = value;
        }
    }

    private static ArenaExtendStrategy _arenaExtend = ArenaExtendStrategy.SameAsRequested;

    /// <summary>
    /// Whether to hand the arena's unused blocks back to the device after every run —
    /// ORT's <c>memory.enable_memory_arena_shrinkage</c> run option. Off by default, since
    /// the blocks then have to be re-allocated on the next run, which costs a synchronizing
    /// <c>cudaMalloc</c> per step. On, the arena stops being a ratchet: what a run does not
    /// need stays available to the rest of the machine.
    /// </summary>
    public static bool ShrinkArenaAfterRun { get; set; }

    /// <summary>
    /// The device's memory right now, or <c>null</c> when no CUDA runtime is installed (so
    /// on a CPU-only machine every reading is <c>null</c> rather than an error). Does not
    /// affect <see cref="PeakUsedBytes"/>.
    /// </summary>
    public static DeviceMemoryReading? Read()
        => CudaRuntime.TryMemGetInfo(out var free, out var total)
            ? new DeviceMemoryReading(total - free, free, total)
            : null;

    /// <summary>
    /// <see cref="Read"/>, and folds the reading into <see cref="PeakUsedBytes"/>. A
    /// <c>null</c> reading leaves the peak alone.
    /// </summary>
    public static DeviceMemoryReading? Sample() => SampleFrom(Read());

    /// <summary>What <see cref="Sample"/> does with a reading once it has one, split out so the
    /// fold can be driven from a machine that has no card to read.</summary>
    internal static DeviceMemoryReading? SampleFrom(DeviceMemoryReading? reading)
    {
        if (reading is { } taken) ObservePeak(taken.UsedBytes);
        return reading;
    }

    /// <summary>
    /// The largest <see cref="DeviceMemoryReading.UsedBytes"/> any <see cref="Sample"/> has
    /// returned since the process started or <see cref="ResetPeak"/> was last called; zero
    /// if nothing has been sampled. Nothing samples on its own — this is exactly what your
    /// own <see cref="Sample"/> calls have seen.
    /// </summary>
    public static long PeakUsedBytes => Interlocked.Read(ref _peakUsedBytes);

    /// <summary>Forgets the peak, so the next <see cref="Sample"/> starts a fresh one.</summary>
    public static void ResetPeak() => Interlocked.Exchange(ref _peakUsedBytes, 0);

    /// <summary>Raises the peak to <paramref name="usedBytes"/> if it is higher, and returns it.</summary>
    internal static long ObservePeak(long usedBytes)
    {
        long peak;
        while (usedBytes > (peak = Interlocked.Read(ref _peakUsedBytes)))
            if (Interlocked.CompareExchange(ref _peakUsedBytes, usedBytes, peak) == peak)
                return usedBytes;
        return peak;
    }
}
