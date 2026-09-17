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
    /// <see cref="Elapsed"/>. They are separated because at multi-GB sizes they do not scale
    /// together and the total alone cannot say which one moved: <see cref="Write"/> is CPU and page
    /// cache (serializing the state and writing it into the staged file), <see cref="Flush"/> is the
    /// fsync that puts those pages on the device — the phase that makes two identical saves of one
    /// identical file differ by a factor of several, since what it costs depends on how much dirty
    /// data the OS had already written back before it ran — and <see cref="Commit"/> is the rename
    /// that publishes the file plus the sweep of stale staged siblings, which is metadata-only and
    /// stays flat as the file grows.</para>
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
    /// and the sweep of staged siblings an earlier failed save left behind.</param>
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
        /// <c>201,327,183 bytes in 0.743s (258 MiB/s): write 0.281s, flush 0.459s, commit 0.003s</c>.
        /// Culture-invariant, so a log line reads the same on every machine.</summary>
        public override string ToString() => string.Format(
            CultureInfo.InvariantCulture,
            "{0:N0} bytes in {1:F3}s ({2:F0} MiB/s): write {3:F3}s, flush {4:F3}s, commit {5:F3}s",
            BytesWritten, Elapsed.TotalSeconds, BytesPerSecond / (1024 * 1024),
            Write.TotalSeconds, Flush.TotalSeconds, Commit.TotalSeconds);
    }
}
