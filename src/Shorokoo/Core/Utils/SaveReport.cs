using System;
using System.Globalization;

namespace Shorokoo
{
    /// <summary>
    /// What a save cost: the bytes it committed to disk and where its time went. Returned by the
    /// save entry points a training loop calls at a checkpoint cadence
    /// (<see cref="TrainingCheckpoint.Save(string, CheckpointComponents?)"/>,
    /// <see cref="Persistence.SaveTrainingCheckpoint(TrainingCheckpoint, string)"/>,
    /// <see cref="Persistence.SaveTrainingCheckpointToSkpt"/> and the builder behind it), so the
    /// cost of saving is a value the caller has rather than a gap it has to infer from a wall clock
    /// it kept itself (Shorokoo/Shorokoo#338).
    ///
    /// <para>The three phases are disjoint and cover the whole call, so
    /// <see cref="Write"/> + <see cref="Flush"/> + <see cref="Commit"/> is exactly
    /// <see cref="Elapsed"/>, and <see cref="Elapsed"/> is what a caller's own stopwatch around the
    /// call would read. Producing the content is inside the measurement, not before it — a save that
    /// serialized its state and then handed the bytes to a measured write would report a fraction of
    /// what it cost, which is the miscount this type exists to end.</para>
    ///
    /// <para>The phases are separated because at multi-GB sizes they do not scale together and the
    /// total alone cannot say which one moved: <see cref="Write"/> is CPU and page cache (producing
    /// the content — serializing the state, and for a <c>.skpt</c> compressing and hashing its
    /// entries — and writing it into the staged file), <see cref="Flush"/> is the fsync that puts
    /// those pages on the device, and <see cref="Commit"/> is the rename that publishes the file
    /// plus the post-commit housekeeping, which is metadata-only and stays flat as the file grows.
    /// Which phase dominates follows from the shape: for a flat safetensors save at size it is the
    /// flush, which is also the one that varies between identical saves of one identical file, since
    /// what it costs depends on how much dirty data the OS had already written back before it ran
    /// (measured on one 200 MB file saved six times over: the write steady at 51 ms, the flush
    /// between 184 and 243 ms, the commit at 7 ms); for a <c>.skpt</c> it is the write, which
    /// carries the container's serialization and hashing.</para>
    ///
    /// <para>A save is disk I/O, not training: a run measuring its own throughput subtracts
    /// <see cref="Elapsed"/> from the window it measures rather than reporting a step rate that
    /// silently carries the checkpoint cadence in it.</para>
    /// </summary>
    /// <param name="BytesWritten">Size of the committed file, in bytes — what the save actually put
    /// on disk, measured on the staged file after it was flushed, not a figure derived from the
    /// state's element counts.</param>
    /// <param name="Write">Producing the content and writing it into the staged file.</param>
    /// <param name="Flush">Flushing the staged file to the device (fsync), which is what makes the
    /// atomic commit meaningful — the file is durable before the rename publishes it.</param>
    /// <param name="Commit">Closing the staged file, the rename that publishes it onto the target,
    /// and the post-commit housekeeping: the sweep of staged siblings an earlier failed save left
    /// behind, and the pruning of older members where the save runs under a retain policy.</param>
    public readonly record struct SaveReport(
        long BytesWritten, TimeSpan Write, TimeSpan Flush, TimeSpan Commit)
    {
        /// <summary>The whole save: <see cref="Write"/> + <see cref="Flush"/> + <see cref="Commit"/>.</summary>
        public TimeSpan Elapsed => Write + Flush + Commit;

        /// <summary>Bytes committed per second over <see cref="Elapsed"/> — the rate the save
        /// actually achieved, which for a given file is not a constant of the machine. Zero for a
        /// save too fast for the clock to resolve.</summary>
        public double BytesPerSecond =>
            Elapsed.Ticks == 0 ? 0.0 : BytesWritten / Elapsed.TotalSeconds;

        /// <summary>A one-line rendering —
        /// <c>200,000,077 bytes in 0.252s (757 MiB/s): write 0.051s, flush 0.194s, commit 0.007s</c>.
        /// Culture-invariant, so a log line reads the same on every machine.</summary>
        public override string ToString() => string.Format(
            CultureInfo.InvariantCulture,
            "{0:N0} bytes in {1:F3}s ({2:F0} MiB/s): write {3:F3}s, flush {4:F3}s, commit {5:F3}s",
            BytesWritten, Elapsed.TotalSeconds, BytesPerSecond / (1024 * 1024),
            Write.TotalSeconds, Flush.TotalSeconds, Commit.TotalSeconds);
    }
}
