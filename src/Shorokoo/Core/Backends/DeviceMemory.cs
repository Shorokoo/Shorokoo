namespace Shorokoo.Core.Backends;

/// <summary>
/// A snapshot of the CUDA device's memory, in bytes. <see cref="UsedBytes"/> is
/// <see cref="TotalBytes"/> minus <see cref="FreeBytes"/> and so counts <b>every</b>
/// process on the device — a desktop session included. <see cref="ProcessBytes"/> is this
/// process's share of it.
/// </summary>
/// <param name="UsedBytes">Device memory in use, by all processes.</param>
/// <param name="FreeBytes">Device memory still allocatable.</param>
/// <param name="TotalBytes">The device's usable memory.</param>
/// <param name="ProcessBytes">Device memory this process holds: everything it has on the card —
/// weights and optimizer state, every session's arena, the CUDA context and its libraries'
/// workspaces — and nothing any other process has. It is the figure Task Manager shows per process
/// on Windows, and <c>nvidia-smi</c> lists per process wherever it can. <c>null</c> where the driver will not
/// attribute device memory to a process — see <see cref="DeviceMemory"/>.</param>
public readonly record struct DeviceMemoryReading(long UsedBytes, long FreeBytes, long TotalBytes, long? ProcessBytes);

/// <summary>
/// Readings of the CUDA device's memory, for the GPU backends (<c>Shorokoo.LinuxGPU</c>,
/// <c>Shorokoo.WinGPU</c>), and the one call that hands back what Shorokoo's allocators keep for
/// reuse (<see cref="ReleaseCached"/>). It configures nothing.
///
/// <para><see cref="Read"/> and <see cref="Sample"/> call the CUDA runtime's
/// <c>cudaMemGetInfo</c> directly and return <c>null</c> when there is no CUDA runtime to
/// call. That reads the <i>device</i> — every process on it. Beside it they read what <i>this
/// process</i> holds there, <see cref="DeviceMemoryReading.ProcessBytes"/>, from whatever the
/// driver answers through: DXGI for a Windows card driven by WDDM, and NVML for any other. Measured
/// on an RTX 4090 under WDDM, a whole reading costs about a microsecond; NVML's list of the
/// processes on a card took about 120 microseconds there. Calling one per training step is the way
/// to catch a peak that a half-second <c>nvidia-smi</c> poll steps straight over. The
/// device read is whichever is current for the calling thread — device 0, because that is the
/// device the shipped GPU backends use.</para>
///
/// <para>The process figure is <c>null</c> where neither answers for the card: where DXGI does
/// not, and NVML is not installed, gives no figure for a process — as it does for a WDDM card —
/// or does not list this process under its own id, as in a container with its own process-id
/// namespace. The two figures of a reading are read one after the other, so while this process
/// allocates or frees on another thread, a reading can put its share a little above or below
/// what the card held at the same instant.</para>
///
/// <para>The peak is process-wide because it is an observation of one process's run, and it
/// moves only when you call <see cref="Sample"/>. For the settings that <i>configure</i>
/// device memory, which are not process-wide, see <see cref="DeviceMemorySettings"/> (per
/// context, and per session) and <see cref="RunSettings"/> (per run); for what a context holds on
/// its card against its budget, <see cref="Shorokoo.Runtime.ComputeContext.ReadDeviceMemoryUse"/>.</para>
///
/// <code>
/// using Shorokoo.Core.Backends;
/// using Shorokoo.Runtime;
///
/// // The budget is the context's, and the rig's steps run on its runtime context.
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
/// Console.WriteLine($"this process {DeviceMemory.PeakProcessBytes / (1024 * 1024)} MiB at its peak");
/// Console.WriteLine($"the card {DeviceMemory.PeakUsedBytes / (1024 * 1024)} MiB at its peak");
/// </code>
/// </summary>
public static class DeviceMemory
{
    private static long _peakUsedBytes;
    private static long _peakProcessBytes;

    /// <summary>
    /// The device's memory right now, or <c>null</c> when no CUDA runtime is installed (so
    /// on a CPU-only machine every reading is <c>null</c> rather than an error). Does not
    /// affect <see cref="PeakUsedBytes"/> or <see cref="PeakProcessBytes"/>.
    /// </summary>
    public static DeviceMemoryReading? Read()
        => CudaRuntime.TryMemGetInfo(out var free, out var total)
            ? new DeviceMemoryReading(total - free, free, total, ProcessDeviceMemory.Read())
            : null;

    /// <summary>
    /// <see cref="Read"/>, and folds the reading into <see cref="PeakUsedBytes"/> and
    /// <see cref="PeakProcessBytes"/>. A <c>null</c> reading leaves both peaks alone, and a
    /// reading without a process figure leaves <see cref="PeakProcessBytes"/> alone.
    /// </summary>
    public static DeviceMemoryReading? Sample() => SampleFrom(Read());

    /// <summary>What <see cref="Sample"/> does with a reading once it has one, split out so the
    /// fold can be driven from a machine that has no card to read.</summary>
    internal static DeviceMemoryReading? SampleFrom(DeviceMemoryReading? reading)
    {
        if (reading is { } taken)
        {
            ObservePeak(ref _peakUsedBytes, taken.UsedBytes);
            if (taken.ProcessBytes is { } own) ObservePeak(ref _peakProcessBytes, own);
        }
        return reading;
    }

    /// <summary>
    /// The largest <see cref="DeviceMemoryReading.UsedBytes"/> any <see cref="Sample"/> has
    /// returned since the process started or <see cref="ResetPeak"/> was last called; zero
    /// if nothing has been sampled. Nothing samples on its own — this is exactly what your
    /// own <see cref="Sample"/> calls have seen.
    /// </summary>
    public static long PeakUsedBytes => Interlocked.Read(ref _peakUsedBytes);

    /// <summary>
    /// The largest <see cref="DeviceMemoryReading.ProcessBytes"/> any <see cref="Sample"/> has
    /// returned since the process started or <see cref="ResetPeak"/> was last called; zero if
    /// nothing with a process figure has been sampled. This is the one figure for "the most this
    /// run held on the card": resident state and every arena together, and no other process.
    /// </summary>
    public static long PeakProcessBytes => Interlocked.Read(ref _peakProcessBytes);

    /// <summary>
    /// Hands back to the system everything Shorokoo's allocators keep for reuse — on the host and on
    /// every card, for every session and for the tensors placed on a card — and answers how many
    /// bytes that was. Nothing in use is touched: every tensor stays where it is, and every session
    /// runs on, taking its blocks from the device again as its next runs ask for them.
    ///
    /// <para>The allocators keep a block that is let go of for the next request of its size, so a
    /// loop's runs find their blocks waiting. Each session keeps no more than the most one of its runs
    /// has used, and gives what it keeps back as it is disposed, or at the end of a run with
    /// <see cref="RunSettings.ShrinkArenaAfterRun"/>; what the tensors placed on a card left kept
    /// goes back at the end of such a run on that card. This is the way to give it all back with
    /// no run to make: between phases of a long-lived program, say, or before handing the card to
    /// another process. It covers the ONNX Runtime backends' memory; a PyTorch or JAX backend keeps
    /// a cache of its own.</para>
    /// </summary>
    public static long ReleaseCached() => CachingAllocator.ReleaseEverywhere();

    /// <summary>Forgets both peaks, so the next <see cref="Sample"/> starts fresh ones.</summary>
    public static void ResetPeak()
    {
        Interlocked.Exchange(ref _peakUsedBytes, 0);
        Interlocked.Exchange(ref _peakProcessBytes, 0);
    }

    /// <summary>Raises <paramref name="peak"/> to <paramref name="bytes"/> if it is higher.</summary>
    private static void ObservePeak(ref long peak, long bytes)
    {
        long seen;
        while (bytes > (seen = Interlocked.Read(ref peak)))
            if (Interlocked.CompareExchange(ref peak, bytes, seen) == seen)
                return;
    }
}
