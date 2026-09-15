namespace Shorokoo.Core.Inference.Abstractions;

/// <summary>
/// How ONNX Runtime's device arena grows when it needs a block it does not already hold.
/// </summary>
public enum ArenaExtendStrategy
{
    /// <summary>
    /// Extend by the next power of two at or above the requested size — ORT's default, and
    /// what makes the arena settle at roughly 1.8x the memory a training step actually needs.
    /// Fewer, larger extensions: fastest, and the reason a run that fits can still end up
    /// holding the whole card.
    /// </summary>
    NextPowerOfTwo = 0,

    /// <summary>
    /// Extend by exactly the requested size. The arena then tracks what the step asks for
    /// instead of rounding up to a power of two, at the cost of more (and more frequent)
    /// device allocations. Use it when a configuration is close to the card's limit.
    /// </summary>
    SameAsRequested = 1,
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
/// All three are ignored by the CPU backends.</para>
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
/// DeviceMemory.ArenaExtend = ArenaExtendStrategy.SameAsRequested;
///
/// for (int step = 0; step &lt; steps; step++)
/// {
///     checkpoint = rig.TrainStep(checkpoint, inputs);
///     DeviceMemory.Sample();
/// }
/// Console.WriteLine($"peak {DeviceMemory.PeakUsedBytes / (1024 * 1024)} MiB");
/// </code>
///
/// <para><b>This is a process-wide stopgap, not the final shape.</b> One CUDA device
/// (device 0) and one setting for every session in the process, mutable at any time and
/// read at the moment a session is built or run. A per-<c>ComputeContext</c> settings
/// object is what this should become; until then, treat these as startup configuration
/// and do not expect two contexts to differ.</para>
/// </summary>
public static class DeviceMemory
{
    private static long _peakUsedBytes;
    private static long? _limitBytes;

    /// <summary>
    /// The upper bound, in bytes, on what the CUDA arena may allocate — ORT's
    /// <c>gpu_mem_limit</c>. <c>null</c> (the default) leaves ORT free to take the whole
    /// card. A step that needs more than this fails with an ORT <c>BFCArena</c> allocation
    /// error rather than eating into what is left of the device, which is what makes a
    /// too-large configuration fail early and visibly instead of starving everything else
    /// on the machine.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">A limit of zero or less.</exception>
    public static long? LimitBytes
    {
        get => _limitBytes;
        set
        {
            if (value is <= 0)
                throw new ArgumentOutOfRangeException(nameof(value), value, "The device-memory limit must be positive.");
            _limitBytes = value;
        }
    }

    /// <summary>
    /// How the arena extends itself — ORT's <c>arena_extend_strategy</c>. Defaults to
    /// <see cref="ArenaExtendStrategy.NextPowerOfTwo"/>, which is ORT's own default and the
    /// faster of the two.
    /// </summary>
    public static ArenaExtendStrategy ArenaExtend { get; set; } = ArenaExtendStrategy.NextPowerOfTwo;

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
    public static DeviceMemoryReading? Sample()
    {
        var reading = Read();
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
