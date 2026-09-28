using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;

namespace Shorokoo
{
    /// <summary>
    /// What one training step did: the counters it ran at, its loss, and the value every optimizer
    /// hyperparameter had in it. One entry of a <see cref="TrainingHistory"/>.
    ///
    /// <para>Two entries are equal when their counters and loss are, and their
    /// <see cref="Hyperparameters"/> hold the same names with equal values — so an entry read back
    /// from a saved checkpoint equals the one that was saved.</para>
    /// </summary>
    public sealed record TrainingHistoryEntry
    {
        private static readonly AppliedHyperparameterMap NoHyperparameters =
            AppliedHyperparameterMap.Of(new Dictionary<string, AppliedHyperparameter>());

        private readonly IReadOnlyDictionary<string, AppliedHyperparameter> _hyperparameters = NoHyperparameters;

        /// <summary>
        /// The global step counter the step <b>ran at</b> — the value its schedules saw — which is
        /// the <see cref="TrainingCheckpoint.Step"/> of the checkpoint it trained from, one less than
        /// that of the checkpoint it produced.
        /// </summary>
        public long Step { get; init; }

        /// <summary>The epoch counter the step ran at, or <c>null</c> where it was unknown
        /// (see <see cref="TrainingCheckpoint.Epoch"/>).</summary>
        public long? Epoch { get; init; }

        /// <summary>The batch index the step ran at, or <c>null</c> where it was unknown
        /// (see <see cref="TrainingCheckpoint.BatchIndex"/>).</summary>
        public long? BatchIndex { get; init; }

        /// <summary>The step's loss.</summary>
        public float Loss { get; init; }

        /// <summary>
        /// The value every optimizer hyperparameter had in the step, keyed by the producing rig's
        /// <see cref="TrainingRig.HyperparameterNames"/> — the same map as the produced checkpoint's
        /// <see cref="TrainingCheckpoint.AppliedHyperparameters"/>. Immutable: a map given here is
        /// kept as it is where it is already immutable, and copied otherwise.
        /// </summary>
        public IReadOnlyDictionary<string, AppliedHyperparameter> Hyperparameters
        {
            get => _hyperparameters;
            init => _hyperparameters = AppliedHyperparameterMap.Of(
                value ?? throw new ArgumentNullException(nameof(Hyperparameters)));
        }

        /// <inheritdoc />
        public bool Equals(TrainingHistoryEntry? other)
            => other is not null
               && Step == other.Step
               && Epoch == other.Epoch
               && BatchIndex == other.BatchIndex
               && Loss.Equals(other.Loss)
               && AppliedHyperparameterMap.ValueEquals(_hyperparameters, other._hyperparameters);

        /// <inheritdoc />
        public override int GetHashCode() => HashCode.Combine(Step, Epoch, BatchIndex, Loss, _hyperparameters.Count);
    }

    /// <summary>
    /// The steps a training run took, oldest first: one <see cref="TrainingHistoryEntry"/> per
    /// successful training step, carried by every <see cref="TrainingCheckpoint"/> as
    /// <see cref="TrainingCheckpoint.History"/>.
    ///
    /// <para>Immutable. Each step's checkpoint holds its parent's history with one entry appended,
    /// sharing everything before it, so appending costs O(log n) and two runs branched from one
    /// checkpoint each extend the common history independently, leaving the parent's unchanged.</para>
    ///
    /// <para><see cref="TrainingHistoryEntry.Step"/> is not a key: training from a checkpoint whose
    /// counter was set back (<see cref="TrainingCheckpoint.WithStep"/>) appends a second entry with
    /// a step already in the history, and both stay. Entries are in the order they were run.</para>
    ///
    /// <para>To keep only part of it, derive a checkpoint with
    /// <see cref="TrainingCheckpoint.WithHistory"/> over a slice (<see cref="Since"/>,
    /// <see cref="TakeLast"/>); <see cref="TrainingCheckpoint.WithoutHistory"/> clears it.</para>
    /// </summary>
    public sealed class TrainingHistory : IReadOnlyList<TrainingHistoryEntry>
    {
        private readonly ImmutableList<TrainingHistoryEntry> _entries;

        private TrainingHistory(ImmutableList<TrainingHistoryEntry> entries) => _entries = entries;

        /// <summary>The history with no entries: that of a checkpoint no step produced.</summary>
        public static TrainingHistory Empty { get; } = new(ImmutableList<TrainingHistoryEntry>.Empty);

        /// <summary>A history over <paramref name="entries"/>, in that order.</summary>
        internal static TrainingHistory Of(IEnumerable<TrainingHistoryEntry> entries)
            => new(ImmutableList.CreateRange(entries));

        /// <summary>This history with <paramref name="entry"/> after its last entry.</summary>
        internal TrainingHistory Append(TrainingHistoryEntry entry)
            => new(_entries.Add(entry ?? throw new ArgumentNullException(nameof(entry))));

        /// <summary>The number of entries.</summary>
        public int Count => _entries.Count;

        /// <summary>The entry at <paramref name="index"/>, 0 being the oldest.</summary>
        public TrainingHistoryEntry this[int index] => _entries[index];

        /// <summary>The entries whose <see cref="TrainingHistoryEntry.Step"/> is at least
        /// <paramref name="step"/>, in their order.</summary>
        public TrainingHistory Since(long step)
        {
            var kept = _entries.RemoveAll(e => e.Step < step);
            return kept.Count == _entries.Count ? this : new TrainingHistory(kept);
        }

        /// <summary>The last <paramref name="count"/> entries — all of them where there are fewer.</summary>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="count"/> is negative.</exception>
        public TrainingHistory TakeLast(int count)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(count);
            return count >= _entries.Count
                ? this
                : new TrainingHistory(_entries.GetRange(_entries.Count - count, count));
        }

        /// <summary>Every entry's <see cref="TrainingHistoryEntry.Step"/>, in a new list.</summary>
        public IReadOnlyList<long> Steps => [.. _entries.Select(e => e.Step)];

        /// <summary>Every entry's <see cref="TrainingHistoryEntry.Loss"/>, in a new list.</summary>
        public IReadOnlyList<float> Losses => [.. _entries.Select(e => e.Loss)];

        /// <summary>Every entry's <see cref="TrainingHistoryEntry.Epoch"/>, in a new list.</summary>
        public IReadOnlyList<long?> Epochs => [.. _entries.Select(e => e.Epoch)];

        /// <summary>Every entry's <see cref="TrainingHistoryEntry.BatchIndex"/>, in a new list.</summary>
        public IReadOnlyList<long?> BatchIndices => [.. _entries.Select(e => e.BatchIndex)];

        /// <summary>
        /// Every hyperparameter name any entry holds, in the order they first appear. Entries of one
        /// rig all hold the same names; a history continued under another rig can hold names only
        /// some entries have.
        /// </summary>
        public IReadOnlyList<string> HyperparameterNames
        {
            get
            {
                var seen = new HashSet<string>(StringComparer.Ordinal);
                var names = new List<string>();
                foreach (var entry in _entries)
                    foreach (var name in entry.Hyperparameters.Keys)
                        if (seen.Add(name)) names.Add(name);
                return names.AsReadOnly();
            }
        }

        /// <summary>
        /// The value hyperparameter <paramref name="name"/> had in each entry, in a new list parallel
        /// to the entries: <c>null</c> for an entry that holds no such hyperparameter.
        /// </summary>
        public IReadOnlyList<AppliedHyperparameter?> Hyperparameter(string name)
        {
            ArgumentNullException.ThrowIfNull(name);
            return [.. _entries.Select(e => e.Hyperparameters.TryGetValue(name, out var v) ? v : null)];
        }

        /// <inheritdoc />
        public IEnumerator<TrainingHistoryEntry> GetEnumerator() => _entries.GetEnumerator();

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
    }
}
