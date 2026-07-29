using System;
using System.Collections.Generic;
using System.Linq;
using Shorokoo.Graph;

namespace Shorokoo
{
    /// <summary>
    /// A loader's position in its batch stream: the (0-based) <see cref="Epoch"/> and the
    /// <see cref="BatchIndex"/> of the <b>next</b> batch that <see cref="IDataLoader.Next"/> will
    /// yield within that epoch — the loader's live position. A run's resume point is derived from
    /// the checkpoint's recorded counters (which name the batch that was <b>used</b>, see
    /// <see cref="TrainingCheckpoint.Epoch"/> / <see cref="TrainingCheckpoint.BatchIndex"/>) and fed
    /// back with <see cref="IDataLoader.RestoreFrom"/> (position at) or
    /// <see cref="IDataLoader.RestoreAfter"/> (position one batch after) so the resumed run continues
    /// exactly where it left off.
    /// </summary>
    public readonly struct DataLoaderPosition : IEquatable<DataLoaderPosition>
    {
        /// <summary>The 0-based epoch (complete passes over the data already made).</summary>
        public long Epoch { get; }

        /// <summary>The 0-based index of the next batch to yield within <see cref="Epoch"/>
        /// (equivalently, the number of batches already consumed in this epoch).</summary>
        public long BatchIndex { get; }

        /// <summary>Creates a position at <paramref name="epoch"/> / <paramref name="batchIndex"/>.</summary>
        public DataLoaderPosition(long epoch, long batchIndex)
        {
            if (epoch < 0) throw new ArgumentOutOfRangeException(nameof(epoch), epoch, "Epoch must be non-negative.");
            if (batchIndex < 0) throw new ArgumentOutOfRangeException(nameof(batchIndex), batchIndex, "Batch index must be non-negative.");
            Epoch = epoch;
            BatchIndex = batchIndex;
        }

        /// <inheritdoc/>
        public bool Equals(DataLoaderPosition other) => Epoch == other.Epoch && BatchIndex == other.BatchIndex;
        /// <inheritdoc/>
        public override bool Equals(object? obj) => obj is DataLoaderPosition p && Equals(p);
        /// <inheritdoc/>
        public override int GetHashCode() => HashCode.Combine(Epoch, BatchIndex);
        /// <summary>Structural equality.</summary>
        public static bool operator ==(DataLoaderPosition a, DataLoaderPosition b) => a.Equals(b);
        /// <summary>Structural inequality.</summary>
        public static bool operator !=(DataLoaderPosition a, DataLoaderPosition b) => !a.Equals(b);
        /// <inheritdoc/>
        public override string ToString() => $"(epoch {Epoch}, batch {BatchIndex})";
    }

    /// <summary>
    /// One batch produced by an <see cref="IDataLoader"/>: the model <see cref="Input"/> and the
    /// training <see cref="Target"/> (each a <see cref="TensorDataStruct"/> shaped exactly as the
    /// rig's <c>TrainStep</c> expects), tagged with the <see cref="Position"/> it was drawn from.
    /// </summary>
    public readonly struct DataBatch
    {
        /// <summary>Model input fields for this batch.</summary>
        public TensorDataStruct Input { get; }
        /// <summary>Training target fields for this batch.</summary>
        public TensorDataStruct Target { get; }
        /// <summary>The stream position this batch was drawn from (the epoch + batch index it belongs to).</summary>
        public DataLoaderPosition Position { get; }

        /// <summary>Packages an input/target pair with the position it came from.</summary>
        public DataBatch(TensorDataStruct input, TensorDataStruct target, DataLoaderPosition position)
        {
            Input = input ?? throw new ArgumentNullException(nameof(input));
            Target = target ?? throw new ArgumentNullException(nameof(target));
            Position = position;
        }
    }

    /// <summary>
    /// Owns a training data stream: it produces (input, target) batches one at a time, tracks its
    /// <see cref="Position"/> in the stream, and can be repositioned — <see cref="RestoreFrom"/> to a
    /// position, or <see cref="RestoreAfter"/> to one batch past it — so a resumed run continues
    /// exactly where it left off. <see cref="TrainingRig"/>'s loader-taking
    /// <c>Fit</c> overload drives a loader and advances the checkpoint's step / epoch / batch counters
    /// for you.
    ///
    /// <para>This is the bare-minimum contract: a finite dataset chopped into a fixed number of
    /// batches per epoch, cycling epoch after epoch. Streaming / distributed / infinite sources are
    /// out of scope. The loader owns <b>Shorokoo's own</b> position; a host driving an external
    /// pipeline Shorokoo doesn't own still uses the checkpoint's host user-data bag instead.</para>
    /// <para>Drive one with <see cref="TrainingRig"/>'s loader-taking <c>Fit</c> overload, which
    /// advances the checkpoint's step / epoch / batch counters for you.</para>
    /// </summary>
    public interface IDataLoader
    {
        /// <summary>The current position — the epoch and within-epoch index of the batch the next
        /// <see cref="Next"/> will return.</summary>
        DataLoaderPosition Position { get; }

        /// <summary>Produces the batch at the current <see cref="Position"/> and advances the loader by
        /// one batch (rolling into the next epoch after the last batch of the current one).</summary>
        DataBatch Next();

        /// <summary>Repositions the loader so the next <see cref="Next"/> yields the batch <b>at</b>
        /// <paramref name="position"/> — the resume-from-a-known-position primitive.
        /// <paramref name="position"/>'s batch index must be a valid in-epoch index.</summary>
        void RestoreFrom(DataLoaderPosition position);

        /// <summary>Repositions the loader so the next <see cref="Next"/> yields the batch <b>one step
        /// after</b> <paramref name="position"/> — <paramref name="position"/> advanced by one batch,
        /// rolling into the next epoch when it names the last batch of an epoch (the loader performs the
        /// rollover internally). This is the resume primitive for the unified checkpoint convention: a
        /// checkpoint records the batch that was <b>used</b>, so resuming continues at the batch after it.
        /// <paramref name="position"/>'s batch index must be a valid in-epoch index.</summary>
        void RestoreAfter(DataLoaderPosition position);
    }

    /// <summary>
    /// The bare-minimum <see cref="IDataLoader"/>: batches tensors already held in memory. Given the
    /// full input and target datasets (each a <see cref="TensorDataStruct"/> whose fields share a
    /// leading sample dimension of length <c>N</c>), it slices them into fixed-size batches along
    /// that dimension, optionally reshuffling the sample order every epoch.
    ///
    /// <para><b>Shuffle determinism.</b> With <c>shuffle: true</c>, the permutation for epoch <c>e</c>
    /// is a pure function of <c>(seed, e)</c> — a Fisher–Yates shuffle driven by a SplitMix64 stream
    /// seeded from the two mixed together. It uses no ambient <see cref="System.Random"/> and no wall
    /// clock, so it is identical across processes and runtimes. That is what makes resume exact:
    /// <see cref="RestoreFrom"/>ing to <c>(e, b)</c> regenerates epoch <c>e</c>'s order bit-for-bit and
    /// skips the first <c>b</c> batches, so the continued run sees the very batches the original
    /// would have.</para>
    ///
    /// <para><b>Partial final batch.</b> With <c>dropLast: true</c> (the default) a trailing partial
    /// batch (when <c>N</c> is not a multiple of <c>batchSize</c>) is dropped, so every batch has the
    /// same leading dimension — the shape the rig's training-step graph was compiled for. Pass
    /// <c>dropLast: false</c> to keep the smaller final batch (only safe if the graph tolerates a
    /// variable batch dimension).</para>
    /// </summary>
    public sealed class InMemoryDataLoader : IDataLoader
    {
        private readonly TensorDataStruct _inputs;
        private readonly TensorDataStruct _targets;
        private readonly int _batchSize;
        private readonly bool _shuffle;
        private readonly long _seed;
        private readonly long _sampleCount;

        // Cached per-epoch sample order and the epoch it was generated for (-1 = none yet).
        private int[]? _order;
        private long _orderEpoch = -1;

        private long _epoch;
        private long _batchIndex;

        /// <summary>
        /// Builds an in-memory loader over <paramref name="inputs"/> / <paramref name="targets"/>.
        /// </summary>
        /// <param name="inputs">Full input dataset; every field is a tensor whose leading dimension is the sample count.</param>
        /// <param name="targets">Full target dataset; same sample count as <paramref name="inputs"/>.</param>
        /// <param name="batchSize">Samples per batch (must be positive and no larger than the sample count).</param>
        /// <param name="shuffle">Reshuffle the sample order each epoch (deterministically from <paramref name="seed"/>).</param>
        /// <param name="seed">Seed for the per-epoch shuffle. Ignored when <paramref name="shuffle"/> is false.</param>
        /// <param name="dropLast">Drop a trailing partial batch so all batches share one shape (default true).</param>
        public InMemoryDataLoader(
            TensorDataStruct inputs,
            TensorDataStruct targets,
            int batchSize,
            bool shuffle = false,
            long seed = 0,
            bool dropLast = true)
        {
            _inputs = inputs ?? throw new ArgumentNullException(nameof(inputs));
            _targets = targets ?? throw new ArgumentNullException(nameof(targets));
            if (batchSize <= 0) throw new ArgumentOutOfRangeException(nameof(batchSize), batchSize, "Batch size must be positive.");

            if (_inputs.Definition.Fields.Length == 0)
                throw new ArgumentException("Input dataset has no fields.", nameof(inputs));
            if (_targets.Definition.Fields.Length == 0)
                throw new ArgumentException("Target dataset has no fields.", nameof(targets));

            _sampleCount = LeadingDimAcrossFields(_inputs, nameof(inputs));
            long targetCount = LeadingDimAcrossFields(_targets, nameof(targets));
            if (targetCount != _sampleCount)
                throw new ArgumentException(
                    $"Input and target sample counts disagree: inputs have {_sampleCount} samples, targets have {targetCount}.",
                    nameof(targets));

            if (batchSize > _sampleCount)
                throw new ArgumentOutOfRangeException(nameof(batchSize), batchSize,
                    $"Batch size exceeds the sample count ({_sampleCount}).");

            _batchSize = batchSize;
            _shuffle = shuffle;
            _seed = seed;

            long full = _sampleCount / batchSize;
            long withPartial = (_sampleCount + batchSize - 1) / batchSize;
            long batches = dropLast ? full : withPartial;
            if (batches <= 0)
                throw new ArgumentException(
                    $"No complete batch fits: {_sampleCount} samples with batch size {batchSize} and dropLast=true yields zero batches.",
                    nameof(batchSize));
            BatchesPerEpoch = checked((int)batches);
        }

        /// <summary>The number of batches in one epoch (a full pass over the data). Fixed for the
        /// loader's lifetime, so <c>step = epoch * BatchesPerEpoch + batchIndex</c> holds for a run
        /// driven from a single loader. Concrete to <see cref="InMemoryDataLoader"/> — not on
        /// <see cref="IDataLoader"/>, since the interface's resume primitives
        /// (<see cref="RestoreFrom"/> / <see cref="RestoreAfter"/>) do the epoch rollover internally,
        /// so a caller never needs the count to compute an "after" position by hand.</summary>
        public int BatchesPerEpoch { get; }

        /// <summary>The number of samples in the dataset (the shared leading dimension of every field).</summary>
        public long SampleCount => _sampleCount;

        /// <inheritdoc/>
        public DataLoaderPosition Position => new DataLoaderPosition(_epoch, _batchIndex);

        /// <inheritdoc/>
        public void RestoreFrom(DataLoaderPosition position)
        {
            ValidateInEpoch(position);
            _epoch = position.Epoch;
            _batchIndex = position.BatchIndex;
        }

        /// <inheritdoc/>
        public void RestoreAfter(DataLoaderPosition position)
        {
            ValidateInEpoch(position);
            // Advance one batch past the given (used) position, rolling into the next epoch after the
            // last batch of an epoch — the same rollover Next() performs.
            long epoch = position.Epoch;
            long batchIndex = position.BatchIndex + 1;
            if (batchIndex >= BatchesPerEpoch)
            {
                batchIndex = 0;
                epoch++;
            }
            _epoch = epoch;
            _batchIndex = batchIndex;
        }

        /// <summary>Rejects a batch index outside a single epoch (guards against a mismatched
        /// checkpoint/loader — e.g. a different batch size or dataset).</summary>
        private void ValidateInEpoch(DataLoaderPosition position)
        {
            if (position.BatchIndex >= BatchesPerEpoch)
                throw new ArgumentOutOfRangeException(nameof(position), position.BatchIndex,
                    $"Batch index {position.BatchIndex} is out of range for a loader with {BatchesPerEpoch} batches per epoch. " +
                    "Was this checkpoint produced with a different batch size / dataset?");
        }

        /// <inheritdoc/>
        public DataBatch Next()
        {
            EnsureOrderForCurrentEpoch();

            int start = checked((int)(_batchIndex * _batchSize));
            int count = (int)Math.Min(_batchSize, _sampleCount - start);
            int[] batchIndices = new int[count];
            Array.Copy(_order!, start, batchIndices, 0, count);

            var input = GatherRows(_inputs, batchIndices);
            var target = GatherRows(_targets, batchIndices);
            var drawn = new DataLoaderPosition(_epoch, _batchIndex);

            // Advance one batch, rolling into the next epoch after the last batch.
            _batchIndex++;
            if (_batchIndex >= BatchesPerEpoch)
            {
                _batchIndex = 0;
                _epoch++;
            }

            return new DataBatch(input, target, drawn);
        }

        // ---- sample order ----

        private void EnsureOrderForCurrentEpoch()
        {
            if (_order is not null && (!_shuffle ? _orderEpoch == 0 : _orderEpoch == _epoch))
                return;

            int n = checked((int)_sampleCount);
            int[] order = new int[n];
            for (int i = 0; i < n; i++) order[i] = i;

            if (_shuffle)
            {
                // Fisher–Yates driven by a SplitMix64 stream seeded from (seed, epoch). Pure function
                // of the epoch, so regenerating epoch e always yields the identical permutation.
                ulong state = unchecked((ulong)_seed ^ ((ulong)_epoch * 0x9E3779B97F4A7C15UL));
                for (int i = n - 1; i > 0; i--)
                {
                    ulong r = SplitMix64(ref state);
                    int j = (int)(r % (ulong)(i + 1));
                    (order[i], order[j]) = (order[j], order[i]);
                }
                _orderEpoch = _epoch;
            }
            else
            {
                _orderEpoch = 0; // identity order is epoch-independent
            }

            _order = order;
        }

        private static ulong SplitMix64(ref ulong state)
        {
            unchecked
            {
                state += 0x9E3779B97F4A7C15UL;
                ulong z = state;
                z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
                z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
                return z ^ (z >> 31);
            }
        }

        // ---- batching ----

        /// <summary>Validates every field shares one positive leading dimension and returns it.</summary>
        private static long LeadingDimAcrossFields(TensorDataStruct data, string paramName)
        {
            long? shared = null;
            foreach (var fieldDef in data.Definition.Fields)
            {
                if (data.Fields[fieldDef.Name] is not TensorData td)
                    throw new ArgumentException(
                        $"Field '{fieldDef.Name}' is not a plain tensor; the in-memory loader batches plain tensors only.",
                        paramName);
                long[] dims = td.Shape.Dims;
                if (dims.Length == 0)
                    throw new ArgumentException(
                        $"Field '{fieldDef.Name}' is a rank-0 scalar; a dataset field needs a leading sample dimension.",
                        paramName);
                long lead = dims[0];
                if (shared is null) shared = lead;
                else if (shared != lead)
                    throw new ArgumentException(
                        $"Dataset fields disagree on the sample dimension: found both {shared} and {lead}.",
                        paramName);
            }
            if (shared is not > 0)
                throw new ArgumentException("Dataset has no samples (leading dimension is 0).", paramName);
            return shared.Value;
        }

        /// <summary>Gathers the given sample rows from every field into a new batch struct (dtype-agnostic, via raw bytes).</summary>
        private static TensorDataStruct GatherRows(TensorDataStruct data, int[] indices)
        {
            var fields = new Dictionary<string, IData>(data.Definition.Fields.Length);
            foreach (var fieldDef in data.Definition.Fields)
            {
                var src = (TensorData)data.Fields[fieldDef.Name];
                fields[fieldDef.Name] = GatherRows(src, indices);
            }
            return new TensorDataStruct(data.Definition, fields);
        }

        private static TensorData GatherRows(TensorData src, int[] indices)
        {
            long[] dims = src.Shape.Dims;
            long n = dims[0];

            long rowElems = 1;
            for (int d = 1; d < dims.Length; d++) rowElems *= dims[d];

            var raw = src.AccessRawMemory();
            long totalElems = n * rowElems;
            int elemBytes = totalElems == 0 ? 0 : raw.Length / checked((int)totalElems);
            int rowBytes = checked((int)rowElems) * elemBytes;

            byte[] outBytes = new byte[indices.Length * rowBytes];
            for (int k = 0; k < indices.Length; k++)
            {
                int srcOffset = indices[k] * rowBytes;
                raw.Slice(srcOffset, rowBytes).CopyTo(outBytes.AsSpan(k * rowBytes, rowBytes));
            }

            long[] outDims = new long[dims.Length];
            outDims[0] = indices.Length;
            for (int d = 1; d < dims.Length; d++) outDims[d] = dims[d];

            return TensorData.CreateFromRawBytes(new Shape(outDims), src.DType, outBytes);
        }
    }
}
