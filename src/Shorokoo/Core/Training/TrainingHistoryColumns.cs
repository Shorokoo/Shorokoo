using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using System.Linq;

namespace Shorokoo
{
    /// <summary>
    /// The on-disk form of a <see cref="TrainingHistory"/>, shared by the flat safetensors checkpoint
    /// (each column under the <c>history/</c> section) and the <c>.skpt</c> container (each column a
    /// tensor of its <c>history</c> data entry). Columnar: one tensor per column, one row per entry.
    ///
    /// <list type="bullet">
    /// <item><c>step</c>: int64[n].</item>
    /// <item><c>loss</c>: float32[n].</item>
    /// <item><c>epoch</c>: int64[n] and <c>epoch_present</c>: bool[n] — a row's epoch is the value
    /// where its presence is true and absent (<c>null</c>) where it is false, whatever the value
    /// column holds there.</item>
    /// <item><c>batch_index</c>: int64[n] and <c>batch_index_present</c>: bool[n], likewise.</item>
    /// <item>Per hyperparameter name: <c>hyperparameter/&lt;name&gt;</c>, of the value's dtype and
    /// shape [n, …valueShape], and <c>hyperparameter_present/&lt;name&gt;</c>: bool[n], saying which
    /// rows hold a value for it.</item>
    /// </list>
    ///
    /// <para>A column holds one dtype and shape, so a history whose entries give one hyperparameter
    /// values of different dtypes or shapes cannot be written. Reading is strict: every required
    /// column, all of length n, at its dtype and rank, and nothing else.</para>
    /// </summary>
    internal static class TrainingHistoryColumns
    {
        internal const string StepColumn = "step";
        internal const string LossColumn = "loss";
        internal const string EpochColumn = "epoch";
        internal const string EpochPresentColumn = "epoch_present";
        internal const string BatchIndexColumn = "batch_index";
        internal const string BatchIndexPresentColumn = "batch_index_present";
        internal const string HyperparameterPrefix = "hyperparameter/";
        internal const string HyperparameterPresentPrefix = "hyperparameter_present/";

        /// <summary>The columns of <paramref name="history"/>, in the order they are written.</summary>
        /// <exception cref="InvalidOperationException">One hyperparameter has values of different
        /// dtypes or shapes across the entries.</exception>
        internal static List<(string Name, TensorData Data)> Write(TrainingHistory history)
        {
            int n = history.Count;
            var steps = new long[n];
            var losses = new float[n];
            var epochs = new long[n];
            var batches = new long[n];
            var epochPresent = new byte[n];
            var batchPresent = new byte[n];
            for (int i = 0; i < n; i++)
            {
                var e = history[i];
                steps[i] = e.Step;
                losses[i] = e.Loss;
                if (e.Epoch is long epoch) { epochs[i] = epoch; epochPresent[i] = 1; }
                if (e.BatchIndex is long batch) { batches[i] = batch; batchPresent[i] = 1; }
            }

            var columns = new List<(string, TensorData)>
            {
                (StepColumn, Globals.TensorData([n], steps)),
                (LossColumn, Globals.TensorData([n], losses)),
                (EpochColumn, Globals.TensorData([n], epochs)),
                (EpochPresentColumn, BoolColumn(epochPresent)),
                (BatchIndexColumn, Globals.TensorData([n], batches)),
                (BatchIndexPresentColumn, BoolColumn(batchPresent)),
            };

            foreach (var name in history.HyperparameterNames)
            {
                var values = history.Hyperparameter(name);
                int last = n - 1;
                while (values[last] is null) last--;
                var reference = values[last]!;
                for (int i = last - 1; i >= 0; i--)
                {
                    if (values[i] is not { } v || SameLayout(v, reference)) continue;
                    throw new InvalidOperationException(
                        $"The training history cannot be saved: hyperparameter '{name}' is " +
                        $"{Describe(v)} at step {history[i].Step} but {Describe(reference)} at step " +
                        $"{history[last].Step}, and a saved history holds one dtype and shape per " +
                        "hyperparameter. Save a history in which it has one of them, e.g. " +
                        $"checkpoint.WithHistory(checkpoint.History.Since({history[i + 1].Step})), or " +
                        "none with checkpoint.WithoutHistory().");
                }

                int rowBytes = reference.RawBytes.Length;
                var bytes = new byte[(long)rowBytes * n];
                var present = new byte[n];
                for (int i = 0; i < n; i++)
                {
                    if (values[i] is not { } v) continue;
                    v.RawBytes.CopyTo(bytes.AsSpan(i * rowBytes, rowBytes));
                    present[i] = 1;
                }
                columns.Add((HyperparameterPrefix + name,
                    TensorData.CreateFromRawBytes(new Shape([n, .. reference.Dims]), reference.DType, bytes)));
                columns.Add((HyperparameterPresentPrefix + name, BoolColumn(present)));
            }
            return columns;
        }

        /// <summary>
        /// The history <paramref name="columns"/> hold, validated strictly: <paramref name="origin"/>
        /// names the file in what a malformed history is refused with.
        /// </summary>
        /// <exception cref="InvalidDataException">The columns do not form a history: one is missing,
        /// unknown, of the wrong dtype or rank, or of another length than the rest.</exception>
        internal static TrainingHistory Read(IEnumerable<(string Name, TensorData Data)> columns, string origin)
        {
            var byName = new Dictionary<string, TensorData>(StringComparer.Ordinal);
            var hyperNames = new List<string>();
            foreach (var (name, data) in columns)
            {
                if (!byName.TryAdd(name, data))
                    throw Malformed(origin, $"column '{name}' appears twice");
                if (name.StartsWith(HyperparameterPrefix, StringComparison.Ordinal))
                    hyperNames.Add(name.Substring(HyperparameterPrefix.Length));
                else if (!name.StartsWith(HyperparameterPresentPrefix, StringComparison.Ordinal)
                         && name is not (StepColumn or LossColumn or EpochColumn or EpochPresentColumn
                             or BatchIndexColumn or BatchIndexPresentColumn))
                    throw Malformed(origin, $"it has an unknown column '{name}'");
            }

            var steps = Column(byName, StepColumn, DType.Int64, origin);
            long n = steps.Shape.Dims[0];
            if (n > int.MaxValue) throw Malformed(origin, $"it declares {n} entries");
            long[] Longs(string name) => Sized(Column(byName, name, DType.Int64, origin), n, name, origin).As<int64>().CopyMemory<long>();
            bool[] Flags(string name) => Sized(Column(byName, name, DType.Bool, origin), n, name, origin).CopyRawMemory().Select(b => b != 0).ToArray();

            var stepValues = steps.As<int64>().CopyMemory<long>();
            var losses = Sized(Column(byName, LossColumn, DType.Float32, origin), n, LossColumn, origin).As<float32>().CopyMemory<float>();
            var epochs = Longs(EpochColumn);
            var epochPresent = Flags(EpochPresentColumn);
            var batches = Longs(BatchIndexColumn);
            var batchPresent = Flags(BatchIndexPresentColumn);

            var hypers = new List<(string Name, DType DType, long[] Dims, byte[] Bytes, bool[] Present)>(hyperNames.Count);
            foreach (var name in hyperNames)
            {
                var valueColumn = byName[HyperparameterPrefix + name];
                var dims = valueColumn.Shape.Dims;
                if (dims.Length < 1 || dims[0] != n)
                    throw Malformed(origin, $"column '{HyperparameterPrefix + name}' has shape [{string.Join(", ", dims)}], " +
                        $"not [{n}, …]");
                var presentName = HyperparameterPresentPrefix + name;
                if (!byName.ContainsKey(presentName))
                    throw Malformed(origin, $"column '{HyperparameterPrefix + name}' has no '{presentName}' column");
                hypers.Add((name, valueColumn.DType, dims[1..], valueColumn.CopyRawMemory(), Flags(presentName)));
            }
            foreach (var name in byName.Keys)
                if (name.StartsWith(HyperparameterPresentPrefix, StringComparison.Ordinal)
                    && !byName.ContainsKey(HyperparameterPrefix + name.Substring(HyperparameterPresentPrefix.Length)))
                    throw Malformed(origin, $"column '{name}' has no value column");

            var layouts = new Dictionary<string, AppliedHyperparameterMap.Layout>(StringComparer.Ordinal);
            var entries = new List<TrainingHistoryEntry>((int)n);
            for (int i = 0; i < n; i++)
            {
                var names = new List<string>(hypers.Count);
                var values = ImmutableArray.CreateBuilder<AppliedHyperparameter>(hypers.Count);
                foreach (var h in hypers)
                {
                    if (!h.Present[i]) continue;
                    int rowBytes = (int)(h.Bytes.LongLength / n);
                    names.Add(h.Name);
                    values.Add(AppliedHyperparameter.FromRawBytes(
                        h.DType, [.. h.Dims], h.Bytes.AsSpan(i * rowBytes, rowBytes).ToArray()));
                }
                var key = string.Join("\0", names);
                if (!layouts.TryGetValue(key, out var layout))
                    layouts[key] = layout = AppliedHyperparameterMap.Layout.Of(names);
                entries.Add(new TrainingHistoryEntry
                {
                    Step = stepValues[i],
                    Loss = losses[i],
                    Epoch = epochPresent[i] ? epochs[i] : null,
                    BatchIndex = batchPresent[i] ? batches[i] : null,
                    Hyperparameters = new AppliedHyperparameterMap(layout, values.ToImmutable()),
                });
            }
            return TrainingHistory.Of(entries);
        }

        private static TensorData BoolColumn(byte[] flags)
            => TensorData.CreateFromRawBytes(new Shape([flags.Length]), DType.Bool, flags);

        private static bool SameLayout(AppliedHyperparameter a, AppliedHyperparameter b)
            => a.DType == b.DType && a.Dims.SequenceEqual(b.Dims);

        private static string Describe(AppliedHyperparameter v) => $"{v.DType}[{string.Join(", ", v.Dims.ToArray())}]";

        private static TensorData Column(Dictionary<string, TensorData> byName, string name, DType dtype, string origin)
        {
            if (!byName.TryGetValue(name, out var column))
                throw Malformed(origin, $"it has no '{name}' column");
            if (column.DType != dtype || column.Shape.Dims.Length != 1)
                throw Malformed(origin, $"column '{name}' is {column.DType}[{string.Join(", ", column.Shape.Dims)}], " +
                    $"not a one-dimensional {dtype} column");
            return column;
        }

        private static TensorData Sized(TensorData column, long n, string name, string origin)
            => column.Shape.Dims[0] == n
                ? column
                : throw Malformed(origin, $"column '{name}' has {column.Shape.Dims[0]} entries where '{StepColumn}' has {n}");

        private static InvalidDataException Malformed(string origin, string why)
            => new($"'{origin}' holds a malformed training history: {why}.");
    }
}
