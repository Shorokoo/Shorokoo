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
/// One session's memory arena, as its runtime reports it. Unlike
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
///
/// <para><b>A session's initializers come out of this arena</b>, on a CPU backend and on a CUDA
/// one alike, so a session is already holding its weights before it has run anything: a graph whose
/// only weight is four mebibytes reads back 4,194,304 bytes of <see cref="MaxInUseBytes"/> at
/// construction. How they are held differs, though — the CPU arena takes the weight as a reserve
/// (<see cref="ReserveCount"/>) and the CUDA arena as a block of its own
/// (<see cref="ArenaExtensionCount"/>) — so those two counts are not comparable between the
/// devices.</para>
/// </summary>
/// <param name="InUseBytes">Bytes the arena has handed out and not taken back.</param>
/// <param name="LimitBytes">The arena's cap — the session's <c>gpu_mem_limit</c>, which on a
/// context under a device-memory budget is what <see cref="DeviceMemorySettings.LimitBytes"/> left
/// it — or <c>-1</c> when it has none.</param>
/// <param name="MaxAllocSizeBytes">The largest single allocation the arena has served.</param>
/// <param name="MaxInUseBytes">The most that was ever in use at once, over the arena's whole
/// life.</param>
/// <param name="AllocationCount">Allocations the arena has served.</param>
/// <param name="ArenaExtensionCount">Blocks the arena holds from the device. It rises as the arena
/// takes fresh ones and <b>falls when it hands them back</b>, so it is the standing count and not a
/// tally of every extension ever made — on a run asking for
/// <see cref="RunSettings.ShrinkArenaAfterRun"/> it can end below where it started.</param>
/// <param name="ArenaShrinkageCount">Times the arena handed blocks back to the device. This one
/// only ever rises.</param>
/// <param name="ReserveCount">Allocations made outside the arena's own blocks.</param>
/// <param name="TotalAllocatedBytes">Bytes the arena has taken from the device, in use or not —
/// and <b>not</b> a bound on what the device is holding. A CUDA run that filled a 24,564 MiB card
/// reported 32,462 MiB here; the likeliest reading is that an arena pressed to the card's edge
/// gives regions back and takes others while this counter does not follow all the way down. For
/// what the card is carrying, read <see cref="DeviceMemory"/>.</param>
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
