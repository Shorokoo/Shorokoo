namespace Shorokoo.Core.Backends;

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
/// Readings of the CUDA device's memory, for the GPU backends (<c>Shorokoo.LinuxGPU</c>,
/// <c>Shorokoo.WinGPU</c>). It reports what is gone; it configures nothing.
///
/// <para><see cref="Read"/> and <see cref="Sample"/> call the CUDA runtime's
/// <c>cudaMemGetInfo</c> directly and return <c>null</c> when there is no CUDA runtime to
/// call. They read the <i>device</i>, not this process, and cost about a microsecond, so
/// calling one per training step is the way to catch a peak that a half-second
/// <c>nvidia-smi</c> poll steps straight over. The device read is whichever is current for
/// the calling thread — device 0, because that is the device the shipped GPU backends
/// use.</para>
///
/// <para>The peak is process-wide because it is an observation of one process's run, and it
/// moves only when you call <see cref="Sample"/>. For the settings that <i>configure</i>
/// device memory, which are not process-wide, see <see cref="DeviceMemorySettings"/> (per
/// session) and <see cref="RunSettings"/> (per run).</para>
///
/// <code>
/// using Shorokoo.Core.Backends;
/// using Shorokoo.Runtime;
///
/// // The budget belongs to the sessions the rig compiles, so it goes on the rig's runtime context.
/// var ctx = new ComputeContext
/// {
///     DeviceMemory = new DeviceMemorySettings { LimitBytes = 16L * 1024 * 1024 * 1024 },
/// };
/// var rig = TrainingRig.FromScratch(model, loss, optimizer, sample, hypers, runtimeContext: ctx);
///
/// for (int step = 0; step &lt; steps; step++)
/// {
///     checkpoint = rig.TrainStep(checkpoint, inputs);
///     DeviceMemory.Sample();
/// }
/// Console.WriteLine($"peak {DeviceMemory.PeakUsedBytes / (1024 * 1024)} MiB");
/// </code>
/// </summary>
public static class DeviceMemory
{
    private static long _peakUsedBytes;

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
