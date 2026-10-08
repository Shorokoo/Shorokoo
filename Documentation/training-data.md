# Feeding training data

Feed a rig its batches through a data loader, resume a run at the batch after its checkpoint, and
read the history a run records.

Related: [training.md](training.md) · [training-checkpoints.md](training-checkpoints.md)

## Feeding data: the data loader

The array overloads of `Fit`/`Train` take pre-batched `TensorDataStruct[]`, each element one
batch: element `i` is batch index `i` of every epoch. A **data loader** owns the batch stream: it
batches your data, tracks its position, and lets `Fit` advance step / epoch / batch, so a saved checkpoint records where the run
was and a resumed run continues from the next batch.

```csharp
// One value per field of the definition, in declaration order; the leading dimension is
// the sample count.
var inputs  = rig.InputDef.FromOrderedData(TensorData([1000L, 64L], features));
var targets = rig.TargetDef.FromOrderedData(TensorData([1000L, 10L], labels));

// Batch into 32s, reshuffling each epoch (deterministically from the seed).
var loader = new InMemoryDataLoader(inputs, targets, batchSize: 32, shuffle: true, seed: 42);

var outcome = rig.Fit(loader, numEpochs: 10);   // step / epoch / batch advance automatically
```

`FromOrderedData` takes field names from the definition — the target field is named after the loss
module's **second `Inline` parameter**, so do not hard-code `"targets"`. It pairs values
positionally and throws on a count mismatch; for many same-shaped fields,
`new TensorDataStruct(def, fields)` also catches a swapped pair.

- **`IDataLoader`**: `Position` (`DataLoaderPosition`, epoch + index of the *next* batch), `Next()`
  (returns the current `DataBatch` — input, target, and its position — and advances, rolling into
  the next epoch), `RestoreFrom(position)` (next `Next()` yields the batch *at* `position`) and
  `RestoreAfter(position)` (next `Next()` yields the batch after it, rolling over epochs).
  `InMemoryDataLoader.BatchesPerEpoch` is not on the interface.
- **One step at a time.** `rig.TrainStep(checkpoint, loader)` draws one batch, trains on it (its
  position drives any scheduler) and records the **batch used**; `Fit(loader)` loops over it.
  Without a loader, `rig.TrainStep(checkpoint, input, target, epoch, batchNumber)` records the given
  counters verbatim.
- **`InMemoryDataLoader`** slices tensors you hold along the leading (sample) dimension into
  fixed-size batches, optionally reshuffling every epoch.
- **Batch feeding.** `DataBatch.Input` / `.Target` are `IData`: a `TensorDataStruct` (consumed) or
  one via `.Shared()` / `.TryConsume()`. `InMemoryDataLoader` builds a fresh batch per `Next()`; a
  custom loader that reuses its tensors passes them `.Shared()`
  ([What a step keeps of what it reads](training.md#what-a-training-step-consumes)).
- **Shuffle is deterministic.** With `shuffle: true`, epoch `e`'s permutation is a pure function of
  `(seed, e)` (Fisher–Yates over SplitMix64; no ambient `Random` or clock), so resuming at `(e, b)`
  sees the same batches as the original run.
- **Partial final batch.** `dropLast: true` (default) drops it, keeping every batch at the compiled
  shape. `dropLast: false` keeps it (only if the graph tolerates a variable batch dimension); each
  new shape costs one compile, then is cached. Up to four shapes get their own session; further
  shapes share a shape-generic one, so compiles stay bounded.
- **Resume.** A checkpoint's `.Epoch` / `.BatchIndex` name the batch **used** at its last step. In a
  new process, rebuild the rig and a loader over the same data/seed and call
  `rig.Fit(loader, numEpochs, initialCheckpoint: loaded)`: `Fit` calls `RestoreAfter`, so training
  resumes at the next batch. A position-unknown checkpoint starts at `(0, 0)` via `RestoreFrom`.
  `numEpochs` counts from the resume epoch (a mid-epoch checkpoint first finishes that epoch; one
  saved at an epoch's last batch begins the next); `rig.FitUntilEpoch(loader, untilEpoch, loaded)`
  trains to an epoch counted from the start instead. The array forms of `Fit` / `Train` stamp and
  resume the same way, batch `i` of the array being batch index `i`. For an external data
  pipeline, keep its position in the checkpoint's host user-data bag.

### The training history

Every checkpoint carries `.History`, a `TrainingHistory`: one `TrainingHistoryEntry` per successful
step that led to it, oldest first. It is empty on a checkpoint no step produced. Each step
(`TrainStep`, a resident run's `Step` or `StepToCheckpoint`, and so `Fit` and `Train`) appends one
entry; a failed step appends nothing.

```csharp
public sealed record TrainingHistoryEntry
{
    public long Step { get; init; }          // the counter the step ran at: the produced checkpoint's Step - 1
    public long? Epoch { get; init; }        // the counters it ran at; null where unknown
    public long? BatchIndex { get; init; }
    public float Loss { get; init; }
    // Same map as the produced checkpoint's .AppliedHyperparameters. Immutable.
    public IReadOnlyDictionary<string, AppliedHyperparameter> AppliedHyperparameters { get; init; }
}

public sealed class TrainingHistory : IReadOnlyList<TrainingHistoryEntry>
{
    public static TrainingHistory Empty { get; }
    public static TrainingHistory Of(IEnumerable<TrainingHistoryEntry> entries);  // in that order
    public TrainingHistory Since(long step);             // entries whose Step >= step
    public TrainingHistory TakeLast(int count);          // the last count entries
    // Columns parallel to the entries, built on first read:
    public IReadOnlyList<long> Steps { get; }
    public IReadOnlyList<float> Losses { get; }
    public IReadOnlyList<long?> Epochs { get; }
    public IReadOnlyList<long?> BatchIndices { get; }
    public IReadOnlyList<string> HyperparameterNames { get; }       // every name any entry holds, by first appearance
    public IReadOnlyList<AppliedHyperparameter?> AppliedValues(string name);  // null where an entry lacks it
}
```

```csharp
var result = rig.Fit(loader, numEpochs: 3);
var history = result.FinalCheckpoint.History;
foreach (var (step, loss, lr) in history.Steps.Zip(history.Losses, history.AppliedValues("learningRate")))
    Console.WriteLine($"{step}\t{loss}\t{lr?.ToSingle()}");
```

Entries are `init`-only records, so you can build or merge histories with `TrainingHistory.Of` and
`with`:

```csharp
var merged = TrainingHistory.Of(first.History.Concat(second.History.Select(e => e with { Step = e.Step + offset })));
var ckpt = second.WithHistory(merged);
```

The history is immutable; appending costs `O(log n)` and shares earlier entries, so branches from
one checkpoint do not affect each other. `Step` is not a key: training again from
`ckpt.WithStep(10)` adds a second step-10 entry; entries are in run order.

**Trimming and clearing.** All derivations (`WithCounters`, `WithStep`, `WithTrainableParams`, …,
`Shared()`, `rig.AdoptCheckpoint`) keep the history. `ckpt.WithHistory(ckpt.History.TakeLast(1000))`
or `.Since(5000)` keeps a slice; `WithoutHistory()` clears it. On a resident run use
`run.ReplaceHistory(run.History.TakeLast(1000))` or `run.ClearHistory()`.

**What it costs.** About 200–300 bytes per entry, more with scheduled, runtime or non-scalar
hyperparameters — a few hundred MB for a million steps — and every save writes it all. Bound it on
long runs.

**Saving it.** History is the `CheckpointComponents.History` component, written whenever non-empty
by `checkpoint.Save` and `.skpt` saves. To omit it from a flat save, pass components without it
(`ckpt.Save(path, CheckpointComponents.InferenceState | CheckpointComponents.OptimizerState |
CheckpointComponents.Counters | CheckpointComponents.Loss)`); from a `.skpt` save, save
`ckpt.WithoutHistory()`. A file
without one loads with an empty history; a resumed run continues it. Entries may hold
hyperparameters only some have (e.g. after `otherRig.AdoptCheckpoint(ckpt)`), but if entries give
one hyperparameter different dtypes or shapes the save throws `InvalidOperationException` naming it
and how many trailing entries can be saved; save
`ckpt.WithHistory(ckpt.History.TakeLast(…))`. On-disk layout:
[Save and resume a checkpoint](training-checkpoints.md#save-and-resume-a-checkpoint-across-process-restarts).
