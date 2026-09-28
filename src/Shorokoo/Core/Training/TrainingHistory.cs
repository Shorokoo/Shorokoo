using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading;

namespace Shorokoo
{
    /// <summary>
    /// What one training step did: the counters it ran at, its loss, and the value every optimizer
    /// hyperparameter had in it. One entry of a <see cref="TrainingHistory"/>.
    ///
    /// <para>Two entries are equal when their counters and loss are, and their
    /// <see cref="AppliedHyperparameters"/> hold the same names with equal values — so an entry read
    /// back from a saved checkpoint equals the one that was saved.</para>
    ///
    /// <para>Every property is <c>init</c>, so an entry can be built, or rewritten with
    /// <c>with</c>, to assemble a history with <see cref="TrainingHistory.Of"/> — merging two runs'
    /// histories, say, or relabelling their steps.</para>
    /// </summary>
    public sealed record TrainingHistoryEntry
    {
        private static readonly AppliedHyperparameterMap NoHyperparameters =
            AppliedHyperparameterMap.Of(new Dictionary<string, AppliedHyperparameter>());

        private readonly IReadOnlyDictionary<string, AppliedHyperparameter> _appliedHyperparameters = NoHyperparameters;

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
        /// The value every optimizer hyperparameter had in the step, keyed by the names of the rig
        /// that ran it (its <see cref="TrainingRig.HyperparameterNames"/>) — the same map as the
        /// produced checkpoint's <see cref="TrainingCheckpoint.AppliedHyperparameters"/>. Immutable:
        /// a map given here is kept as it is where it is already immutable, and copied otherwise, in
        /// its enumeration order.
        /// </summary>
        public IReadOnlyDictionary<string, AppliedHyperparameter> AppliedHyperparameters
        {
            get => _appliedHyperparameters;
            init => _appliedHyperparameters = AppliedHyperparameterMap.Of(
                value ?? throw new ArgumentNullException(nameof(AppliedHyperparameters)));
        }

        /// <inheritdoc />
        public bool Equals(TrainingHistoryEntry? other)
            => other is not null
               && Step == other.Step
               && Epoch == other.Epoch
               && BatchIndex == other.BatchIndex
               && Loss.Equals(other.Loss)
               && AppliedHyperparameterMap.ValueEquals(_appliedHyperparameters, other._appliedHyperparameters);

        /// <inheritdoc />
        public override int GetHashCode() => HashCode.Combine(Step, Epoch, BatchIndex, Loss, _appliedHyperparameters.Count);

        private bool PrintMembers(StringBuilder builder)
        {
            var culture = System.Globalization.CultureInfo.InvariantCulture;
            builder.Append(culture, $"Step = {Step}, Epoch = {Epoch}, BatchIndex = {BatchIndex}, Loss = ");
            builder.Append(Loss.ToString(culture)).Append(", AppliedHyperparameters = {");
            var first = true;
            foreach (var (name, value) in _appliedHyperparameters)
            {
                builder.Append(first ? " " : ", ").Append(name).Append(" = ").Append(value);
                first = false;
            }
            builder.Append(first ? "}" : " }");
            return true;
        }
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
    /// <see cref="TakeLast"/>) or over entries assembled with <see cref="Of"/>;
    /// <see cref="TrainingCheckpoint.WithoutHistory"/> clears it, and a
    /// <see cref="ResidentTrainingRun"/> does the same with
    /// <see cref="ResidentTrainingRun.ReplaceHistory"/> and
    /// <see cref="ResidentTrainingRun.ClearHistory"/>.</para>
    /// </summary>
    public sealed class TrainingHistory : IReadOnlyList<TrainingHistoryEntry>
    {
        private readonly ImmutableList<TrainingHistoryEntry> _entries;

        private TrainingHistory(ImmutableList<TrainingHistoryEntry> entries) => _entries = entries;

        /// <summary>The history with no entries: that of a checkpoint no step produced.</summary>
        public static TrainingHistory Empty { get; } = new(ImmutableList<TrainingHistoryEntry>.Empty);

        /// <summary>
        /// A history over <paramref name="entries"/>, in that order — for one assembled by hand, such
        /// as two runs' histories merged, or entries rewritten with <c>with</c>.
        /// </summary>
        /// <exception cref="ArgumentNullException"><paramref name="entries"/>, or one of them, is
        /// <c>null</c>.</exception>
        public static TrainingHistory Of(IEnumerable<TrainingHistoryEntry> entries)
        {
            ArgumentNullException.ThrowIfNull(entries);
            var list = ImmutableList.CreateRange(entries);
            foreach (var entry in list)
                if (entry is null) throw new ArgumentNullException(nameof(entries), "A training history entry is null.");
            return list.IsEmpty ? Empty : new TrainingHistory(list);
        }

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

        private IReadOnlyList<long>? _steps;
        private IReadOnlyList<float>? _losses;
        private IReadOnlyList<long?>? _epochs;
        private IReadOnlyList<long?>? _batchIndices;
        private IReadOnlyList<string>? _hyperparameterNames;

        private IReadOnlyList<T> Column<T>(ref IReadOnlyList<T>? cache, Func<TrainingHistoryEntry, T> select)
            => cache ?? LazyInitializer.EnsureInitialized(ref cache, () => _entries.Select(select).ToImmutableArray());

        /// <summary>Every entry's <see cref="TrainingHistoryEntry.Step"/>, parallel to the entries.
        /// Read-only, built on first read and kept.</summary>
        public IReadOnlyList<long> Steps => Column(ref _steps, e => e.Step);

        /// <summary>Every entry's <see cref="TrainingHistoryEntry.Loss"/>, parallel to the entries.
        /// Read-only, built on first read and kept.</summary>
        public IReadOnlyList<float> Losses => Column(ref _losses, e => e.Loss);

        /// <summary>Every entry's <see cref="TrainingHistoryEntry.Epoch"/>, parallel to the entries.
        /// Read-only, built on first read and kept.</summary>
        public IReadOnlyList<long?> Epochs => Column(ref _epochs, e => e.Epoch);

        /// <summary>Every entry's <see cref="TrainingHistoryEntry.BatchIndex"/>, parallel to the
        /// entries. Read-only, built on first read and kept.</summary>
        public IReadOnlyList<long?> BatchIndices => Column(ref _batchIndices, e => e.BatchIndex);

        /// <summary>
        /// Every hyperparameter name any entry holds, in the order they first appear. Entries of one
        /// rig all hold the same names; a history continued under another rig can hold names only
        /// some entries have. Read-only, built on first read and kept.
        /// </summary>
        public IReadOnlyList<string> HyperparameterNames
            => _hyperparameterNames ?? LazyInitializer.EnsureInitialized(ref _hyperparameterNames, () =>
            {
                var seen = new HashSet<string>(StringComparer.Ordinal);
                var names = ImmutableArray.CreateBuilder<string>();
                foreach (var entry in _entries)
                    foreach (var name in entry.AppliedHyperparameters.Keys)
                        if (seen.Add(name)) names.Add(name);
                return names.ToImmutable();
            });

        /// <summary>
        /// The value hyperparameter <paramref name="name"/> had in each entry, in a new list parallel
        /// to the entries: <c>null</c> for an entry that holds no such hyperparameter.
        /// </summary>
        public IReadOnlyList<AppliedHyperparameter?> AppliedValues(string name)
        {
            ArgumentNullException.ThrowIfNull(name);
            return [.. _entries.Select(e => e.AppliedHyperparameters.TryGetValue(name, out var v) ? v : null)];
        }

        /// <inheritdoc />
        public IEnumerator<TrainingHistoryEntry> GetEnumerator() => _entries.GetEnumerator();

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
    }
}
