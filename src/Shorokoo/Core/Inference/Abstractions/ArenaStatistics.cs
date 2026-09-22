namespace Shorokoo.Core.Inference.Abstractions;

/// <summary>
/// Whether a byte figure is the quantity itself or only a ceiling on it. A bound and a
/// measurement are not interchangeable — a run credited with a peak it never reached would read
/// as the run to shrink — so every figure that can be either says which it is.
/// </summary>
public enum MemoryFigureKind
{
    /// <summary>The figure is what was used.</summary>
    Measured,

    /// <summary>Whatever was used was no more than this, and may have been far less.</summary>
    UpperBound,
}

/// <summary>
/// One inference session's memory arena, as its runtime reports it. Unlike
/// <see cref="DeviceMemoryReading"/>, which is the whole device across every process on it, this
/// is <b>one session's own allocator</b> — ONNX Runtime gives each session an arena per device,
/// so these figures are that session's and nobody else's.
///
/// <para>Which allocator that is follows the session's backend: a CUDA session's device arena on
/// a GPU backend, the session's own CPU arena on a CPU one. A session that answers at all answers
/// every field.</para>
///
/// <para><b><see cref="MaxInUseBytes"/> is cumulative over the arena's whole life</b>, not the
/// last run's: it is a high-water mark the arena never lowers, and the runtime exposes no reset
/// (asking for arena shrinkage does not move it). So reading it once tells you the largest the
/// session has ever been; attributing a peak to a <i>run</i> takes a read either side of that run,
/// which is what <see cref="RunMemoryRecord"/> is and why its peak carries a
/// <see cref="MemoryFigureKind"/>.</para>
/// </summary>
/// <param name="InUseBytes">Bytes the arena has handed out and not taken back.</param>
/// <param name="LimitBytes">The arena's cap — <see cref="DeviceMemorySettings.LimitBytes"/> —
/// or <c>-1</c> when it has none.</param>
/// <param name="MaxAllocSizeBytes">The largest single allocation the arena has served.</param>
/// <param name="MaxInUseBytes">The most that was ever in use at once, over the arena's whole
/// life.</param>
/// <param name="AllocationCount">Allocations the arena has served.</param>
/// <param name="ArenaExtensionCount">Times the arena took a fresh block from the device.</param>
/// <param name="ArenaShrinkageCount">Times the arena handed blocks back to the device.</param>
/// <param name="ReserveCount">Allocations made outside the arena's own blocks.</param>
/// <param name="TotalAllocatedBytes">Bytes the arena holds from the device, in use or not.</param>
public readonly record struct ArenaStatistics(
    long InUseBytes,
    long LimitBytes,
    long MaxAllocSizeBytes,
    long MaxInUseBytes,
    long AllocationCount,
    long ArenaExtensionCount,
    long ArenaShrinkageCount,
    long ReserveCount,
    long TotalAllocatedBytes);
