namespace Shorokoo.Core.Backends;

/// <summary>
/// What one run did to the arena it ran in.
/// </summary>
/// <param name="RunNumber">Which run of the context this was, counting from one.</param>
/// <param name="PeakBytes">The most the arena held at once, as far as a read either side of this
/// run can tell — see <paramref name="PeakKind"/>.</param>
/// <param name="PeakKind">
/// <see cref="MemoryFigureKind.Measured"/> when this run pushed the arena's high-water mark up,
/// so the mark is this run's own peak and exact. <see cref="MemoryFigureKind.UpperBound"/> when it
/// did not: some earlier run holds the mark, this run stayed under it, and how far under is not
/// something the arena records.
/// </param>
/// <param name="Arena">The arena's figures as the run left them.</param>
public readonly record struct RunMemoryRecord(
    long RunNumber,
    long PeakBytes,
    MemoryFigureKind PeakKind,
    ArenaStatistics Arena);

/// <summary>
/// What every run a <see cref="Shorokoo.Runtime.ComputeContext"/> has made did to the memory of
/// the sessions it ran in — read from
/// <see cref="Shorokoo.Runtime.ComputeContext.RunStats"/>, and empty unless the context was given
/// <see cref="DiagnosticSettings.CollectRunStatistics"/>.
///
/// <para>The aggregates are folded in as each run finishes, so they cover <b>every</b> run the
/// context has made — not the <see cref="RecentRuns"/> window, and not only the sessions still
/// alive. That matters because a context holds its compiled graphs weakly: one the program has
/// dropped is collected, and figures gathered by walking the live ones would quietly lose
/// everything it did.</para>
///
/// <para>A context runs its graphs one session at a time, so <see cref="PeakBytes"/> is the
/// largest high-water mark any one of its arenas reached rather than a sum over several at once.
/// Where two of a context's sessions really do run together, read it as the largest of them and
/// not as the total.</para>
/// </summary>
public sealed record RunStatistics
{
    /// <summary>Nothing recorded: what a context that is not collecting answers with.</summary>
    public static RunStatistics Empty { get; } = new();

    /// <summary>Runs whose arena figures were read. A run that failed counts, because the memory
    /// it took is exactly what a failure is usually about; a run on a backend that reports no
    /// figures does not, because there was nothing to record.</summary>
    public long RunCount { get; init; }

    /// <summary>The largest an arena of this context ever held at once, over every run. Exact,
    /// and not bounded by <see cref="RecentRuns"/>.</summary>
    public long PeakBytes { get; init; }

    /// <summary>The largest single allocation any of those runs asked for.</summary>
    public long LargestAllocationBytes { get; init; }

    /// <summary>
    /// The most an arena of this context ever held from its device, in use or not — usually above
    /// <see cref="PeakBytes"/> by whatever the arena keeps spare.
    ///
    /// <para>Not an invariant, though. This is the high-water mark of a figure that falls when the
    /// arena hands blocks back, which <see cref="RunSettings.ShrinkArenaAfterRun"/> asks it to do
    /// at the end of a run — before this is read — while <see cref="PeakBytes"/> comes from a mark
    /// the runtime never lowers. A shrinking run can therefore leave this below the peak it
    /// reached, so treat the difference as spare capacity only where nothing is shrinking.</para>
    /// </summary>
    public long ArenaBytes { get; init; }

    /// <summary>Allocations those runs made between them.</summary>
    public long AllocationCount { get; init; }

    /// <summary>Times those runs made an arena take a fresh block from its device.</summary>
    public long ArenaExtensionCount { get; init; }

    /// <summary>Times those runs made an arena hand blocks back —
    /// <see cref="RunSettings.ShrinkArenaAfterRun"/>, read back.</summary>
    public long ArenaShrinkageCount { get; init; }

    /// <summary>The most recent runs, oldest first, up to
    /// <see cref="DiagnosticSettings.RecentRunCapacity"/> of them.</summary>
    public IReadOnlyList<RunMemoryRecord> RecentRuns { get; init; } = [];
}

/// <summary>
/// Folds each run's arena figures into a <see cref="RunStatistics"/> as the run finishes, and
/// keeps the last N of them in a ring.
///
/// <para>Folding rather than polling is the whole point. The aggregates have to survive the
/// session that produced them: a context tracks its compiled graphs weakly, so a graph the program
/// has dropped is collected and anything computed by walking the live ones would lose its
/// contribution without saying so.</para>
///
/// <para>Locked rather than interlocked: a run's record touches seven aggregates and a ring slot,
/// and they have to move together for a snapshot taken from another thread to be one moment
/// rather than a mixture of several.</para>
/// </summary>
internal sealed class RunStatisticsCollector
{
    private readonly object _gate = new();
    private readonly RunMemoryRecord[] _recent;
    private int _written;
    private long _runCount;
    private long _peakBytes;
    private long _largestAllocationBytes;
    private long _arenaBytes;
    private long _allocationCount;
    private long _arenaExtensionCount;
    private long _arenaShrinkageCount;

    internal RunStatisticsCollector(int recentRunCapacity)
    {
        _recent = new RunMemoryRecord[recentRunCapacity];
    }

    /// <summary>
    /// Records one run from the arena figures taken either side of it.
    ///
    /// <para>The peak is the honest one. The arena's high-water mark never falls and cannot be
    /// reset, so a mark that rose over the run belongs to the run and is exact; one that did not
    /// belongs to some earlier run and bounds this one from above without measuring it. The record
    /// carries which, so the two are never read as the same thing.</para>
    ///
    /// <para>The counts are differences across the same arena, so they are this run's own.</para>
    /// </summary>
    internal void Record(in ArenaStatistics before, in ArenaStatistics after)
    {
        var record = new RunMemoryRecord(
            0,
            after.MaxInUseBytes,
            after.MaxInUseBytes > before.MaxInUseBytes
                ? MemoryFigureKind.Measured
                : MemoryFigureKind.UpperBound,
            after);

        lock (_gate)
        {
            record = record with { RunNumber = ++_runCount };
            _peakBytes = Math.Max(_peakBytes, after.MaxInUseBytes);
            _largestAllocationBytes = Math.Max(_largestAllocationBytes, after.MaxAllocSizeBytes);
            _arenaBytes = Math.Max(_arenaBytes, after.TotalAllocatedBytes);
            // Differences, and clamped at zero. Two runs of one session overlapping read each
            // other's allocations into both their differences, which overcounts; only a session
            // reporting a counter that went backwards could make one negative, and a negative
            // count is worse than a lost one.
            _allocationCount += Math.Max(0, after.AllocationCount - before.AllocationCount);
            _arenaExtensionCount += Math.Max(0, after.ArenaExtensionCount - before.ArenaExtensionCount);
            _arenaShrinkageCount += Math.Max(0, after.ArenaShrinkageCount - before.ArenaShrinkageCount);
            if (_recent.Length != 0)
            {
                _recent[(int)((_runCount - 1) % _recent.Length)] = record;
                _written = Math.Min(_written + 1, _recent.Length);
            }
        }
    }

    /// <summary>The aggregates and the retained window as they stand, oldest run first.</summary>
    internal RunStatistics Snapshot()
    {
        lock (_gate)
        {
            var recent = new RunMemoryRecord[_written];
            for (int i = 0; i < _written; i++)
                recent[i] = _recent[(int)((_runCount - _written + i) % _recent.Length)];
            return new RunStatistics
            {
                RunCount = _runCount,
                PeakBytes = _peakBytes,
                LargestAllocationBytes = _largestAllocationBytes,
                ArenaBytes = _arenaBytes,
                AllocationCount = _allocationCount,
                ArenaExtensionCount = _arenaExtensionCount,
                ArenaShrinkageCount = _arenaShrinkageCount,
                RecentRuns = recent,
            };
        }
    }
}
