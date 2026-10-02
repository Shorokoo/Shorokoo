namespace Shorokoo.Core.Backends;

/// <summary>
/// Whether a byte figure is the quantity itself or only a ceiling on it. A bound and a
/// measurement are not interchangeable — a run credited with a peak it never reached would read
/// as the run to shrink — so every figure that can be either says which it is.
/// </summary>
public enum MemoryFigureKind
{
    /// <summary>The figure was read at the high point it describes: where the memory stood then,
    /// everything already held there included — not what one run alone used.</summary>
    Measured,

    /// <summary>Whatever was used was no more than this, and may have been far less.</summary>
    UpperBound,
}

/// <summary>
/// One session's memory, as its backend's allocator reports it. Unlike
/// <see cref="DeviceMemoryReading"/>, which is the whole device across every process on it, this
/// is <b>one session's own figures</b> and nobody else's.
///
/// <para>Which memory that is follows the session's backend. On ONNX Runtime it is what the
/// session allocates through Shorokoo's allocator — on the card for a CUDA session, in host memory
/// for a CPU one: its weights, everything its runs compute, and the outputs they make, each block
/// one of its own. A backend with an allocator of its own reports that allocator's. A session that
/// answers at all answers every field.</para>
///
/// <para><b><see cref="MaxInUseBytes"/> is cumulative over the session's whole life</b>, not the
/// last run's: it is a high-water mark that is never lowered, and handing memory back does not move
/// it. So reading it once tells you the largest the session has ever been; attributing a peak to a
/// <i>run</i> takes a read either side of that run, which is what <see cref="RunMemoryRecord"/> is
/// and why its peak carries a <see cref="MemoryFigureKind"/>.</para>
///
/// <para><b>A session's initializers are among what it allocates</b>, so a session is already
/// holding its weights before it has run anything: a graph whose only weight is four mebibytes
/// reads back 4,194,304 bytes of <see cref="MaxInUseBytes"/> at construction, in one allocation of
/// one block.</para>
/// </summary>
/// <param name="InUseBytes">Bytes the session has been handed and not given back — an output a
/// caller keeps included, since it is the block the run wrote it into.</param>
/// <param name="LimitBytes">The most the session may allocate — on a context under a device-memory
/// budget, what <see cref="DeviceMemorySettings.LimitBytes"/> left its last run — or <c>-1</c>
/// when it has none.</param>
/// <param name="MaxAllocSizeBytes">The largest single allocation the session has been
/// served.</param>
/// <param name="MaxInUseBytes">The most that was ever in use at once, over the session's whole
/// life.</param>
/// <param name="AllocationCount">Allocations the session has been served.</param>
/// <param name="ArenaExtensionCount">Blocks the session holds from the device, in use or kept for
/// its next runs. It rises as fresh ones are taken and <b>falls when they are handed back</b>, so it
/// is the standing count and not a tally of every block ever taken — on a run asking for
/// <see cref="RunSettings.ShrinkArenaAfterRun"/> it can end below where it started.</param>
/// <param name="ArenaShrinkageCount">Blocks handed back to the device. This one only ever
/// rises.</param>
/// <param name="ReserveCount">Allocations served apart from the allocator's own blocks; none on ONNX
/// Runtime.</param>
/// <param name="TotalAllocatedBytes">Bytes the session holds from the device, in use or kept for its
/// next runs. For what the card is carrying across everything on it, read
/// <see cref="DeviceMemory"/>.</param>
/// <param name="RequestedInUseBytes">Bytes the callers asked for and have not given back: the part of
/// <paramref name="InUseBytes"/> they requested, without what the allocator added to round each
/// request up to the size it serves. A backend whose allocator does not report what was requested
/// reports <paramref name="InUseBytes"/> here, rounding included.</param>
public readonly record struct ArenaStatistics(
    long InUseBytes,
    long LimitBytes,
    long MaxAllocSizeBytes,
    long MaxInUseBytes,
    long AllocationCount,
    long ArenaExtensionCount,
    long ArenaShrinkageCount,
    long ReserveCount,
    long TotalAllocatedBytes,
    long RequestedInUseBytes);
