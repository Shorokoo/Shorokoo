# Saving, resuming and deploying a training run

Save a training run, resume it in a new process, what a save costs, and bind the trained weights
into an inference model.

Related: [training.md](training.md) · [skpt-checkpoints.md](skpt-checkpoints.md) · [training-data.md](training-data.md)

## Save and resume a checkpoint (across process restarts)

A `TrainingCheckpoint` holds trainable params, model state, optimizer state, and the run counters
(step, epoch, batch index):

```csharp
// Save mid-training (e.g. every N steps, or at the end of an epoch):
checkpoint.Save("run.safetensors");

// Later — in a fresh process — rebuild the SAME rig, then load:
var rig  = TrainingRig.FromScratch(MyModel.ComputationGraph, L2Loss.ComputationGraph,
                                   AdamOptimizer.ComputationGraph, sampleInputs,
                                   new AdamOptimizerHyperparameters { ... });
var ckpt = rig.LoadCheckpoint("run.safetensors");   // params + optimizer moments + step restored
var more = rig.Fit(inputs, targets, numEpochs: 5, ckpt);  // continues where it left off
```

- **Flat file layout.** One SafeTensors file with every param/state field. An `int64` marker holds
  `[version, step]`; epoch and batch index are separate `int64` scalars written only when set, so
  unknown counters reload as `null` and a concrete `0` as `0`.
- **History layout.** A non-empty history is the `history/` section, one tensor per column:
  `history/step` (`int64[n]`), `history/loss` (`float32[n]`), `history/epoch` and
  `history/batch_index` (`int64[n]`) with presence columns `history/epoch_present` /
  `history/batch_index_present` (`bool[n]`; `false` reloads as `null`), and per hyperparameter
  `history/hyperparameter/<name>` (its dtype, shape `[n, …valueShape]`) with
  `history/hyperparameter_present/<name>` (`bool[n]`). A missing or unknown column, or one of the
  wrong dtype, rank or length, is refused on load. A `.skpt` stores the same columns in
  `data/history.safetensors`.
- **Saves are atomic.** `checkpoint.Save` (and `Persistence.SaveTrainingCheckpoint`) writes a
  `.tmp-` sibling, flushes it, and renames it into place, so a crash mid-save leaves the old or the
  new file, never a truncated one. The target **directory must exist** (it is not created); a
  leftover `.tmp-` sibling is swept by the next successful save. A file held for a moment by
  another process (an antivirus scanner, the search indexer) does not fail the save; see
  [onnx-and-weights.md](onnx-and-weights.md#facts). `.skpt` saves are atomic too — see
  [skpt-checkpoints.md](skpt-checkpoints.md#the-directory-form) for the directory form's one
  exception.
- **`.skpt` container.** `Persistence.SaveTrainingCheckpointToSkpt(checkpoint, "run.skpt")` (or the
  `Persistence.ForTrainingCheckpoint(...)` builder) writes the native container, taking the model
  from the checkpoint's `.Rig`. Resume with `rig.LoadCheckpointFromSkpt("run.skpt")`, or with no
  graphs in hand, `var (rig, ckpt) = TrainingRig.Load("run.skpt")`, which rebuilds the rig from the
  file. `Persistence.Load` and `Persistence.LoadEvaluationModel` read the model from the same file
  without a rig. Layout: [skpt-checkpoints.md](skpt-checkpoints.md#training-checkpoints).
- **One loader per format.** `rig.LoadCheckpoint` reads only flat safetensors;
  `rig.LoadCheckpointFromSkpt` and `TrainingRig.Load` only `.skpt`. The wrong format fails with an
  error naming the right entry point. `Persistence.Inspect` identifies an unknown file.
- **Validation.** `LoadCheckpoint` / `LoadCheckpointFromSkpt` need a rig built from the **same**
  model/loss/optimizer graphs. Field names, dtypes and **dimensions** are checked on load, and each
  value is checked against the model's declared shape when bound, so a hand-built checkpoint of the
  wrong shape is refused too. Provenance is not checked: right-shaped weights load into a different
  model. Values match parameters by name, so names from a local or `.Named(...)` are stable
  ([Parameter names](defining-models.md#parameter-names)); parameters left to class names are
  numbered in creation order, and two same-shaped ones whose order was swapped load into each
  other's places silently.
- Because `.Step` is restored, **schedules resume from the right step**.
- `rig.LoadCheckpoint(path)` delegates to `TrainingCheckpoint.Load(path, rig)` (and
  `rig.LoadCheckpointFromSkpt(path)` to `TrainingCheckpoint.LoadFromSkpt(path, rig)`), which sets
  `.Rig`. Without a rig, `Persistence.LoadTrainingCheckpoint(path)` reads a flat checkpoint (it is
  self-describing) but validates nothing and sets no `.Rig`; pass it to `rig.AdoptCheckpoint(ckpt)`
  to validate it.
- **`CheckpointComponents`.** Save and load take optional flags — `InferenceState` (trainable params
  + model state), `OptimizerState`, `Counters`, `Loss`, `History`, `TrainingRig` — combined with
  `|`. `null` saves every available component and loads everything present (absent components come
  from the rig's initial values; absent history is empty).
  `checkpoint.Save(path, CheckpointComponents.InferenceState)` saves weights only. Requesting `Loss`
  when it is `null`, or `History` when empty, writes nothing and does not throw. `TrainingRig` (the
  rig's graphs, hyperparameter bindings and RNG config) is always written to a `.skpt` and read by
  `TrainingRig.Load`; never name it yourself. Requesting it (including via
  `CheckpointComponents.All`) throws on the flat `checkpoint.Save` and on `rig.LoadCheckpoint` /
  `rig.LoadCheckpointFromSkpt`; omit it, or pass `null`.
- `rig.AdoptCheckpoint(checkpoint)` returns the checkpoint bound to that rig, after validating the
  field defs, enabling `ToInferenceModel()`.
- `Persistence.Inspect(path)` shows a file's counters and per-section tensor listing without
  loading it — see
  [onnx-and-weights.md](onnx-and-weights.md#identify-and-summarize-a-file-persistenceinspect).

### What a save costs

Every single-file checkpoint save returns a `SaveReport`:

```csharp
var save = checkpoint.Save("run.safetensors");
Console.WriteLine(save);
// 200,000,077 bytes in 0.252s (757 MiB/s): write 0.051s, flush 0.194s, commit 0.007s
```

`Write` serializes into the staged file (for `.skpt`, also hashing, and compressing Zstd entries), `Flush` is the
fsync, `Commit` the rename plus sweeping stale staged files. They sum to `Elapsed`; `BytesWritten`
is the file size and `BytesPerSecond` the achieved rate. `Persistence.SaveTrainingCheckpoint`,
`Persistence.SaveTrainingCheckpointToSkpt` and the `Persistence.ForTrainingCheckpoint(...)`
builder's `Save` return it too; the directory form (`SaveAsDirectory`) returns `void`.

The save is streamed through an ordinary buffered file, so `Write` only hands the bytes to the OS's
page cache; `Flush` is where they reach the disk. What is still unwritten when it starts depends on
how much the OS has written back in the background while the bytes were produced (on Windows, the
cache manager's lazy writer). On a fast disk the flush takes well under a second; on a slow disk
`Flush` approaches the file size divided by the disk's write bandwidth. A report such as `write 0.6s, flush 5.8s` therefore says the disk is the cost, not the
fsync call. Save time varies between identical saves (mostly in `Flush`), and multi-GB saves can
take tens of seconds, so exclude it from throughput measurements:

```csharp
steady.Stop();                                   // saving is I/O, not training
var save = checkpoint.Save(path);
steady.Start();
savedBytes += save.BytesWritten;
```

The flat safetensors save streams each tensor from its storage with no extra copy, and a load reads
the file forward a tensor at a time, so a file of any size, holding tensors of any size, is read back
whole. Into host memory, a tensor one managed array holds (`Array.MaxLength` bytes, just under
2 GiB) is read into the framework's own host memory; a larger one is read through one bounded host
buffer (8 MiB) into host memory of the backend `ComputeContext.Default` runs on, so it is never whole
in a managed array, and a run on that context reads it where it is.

The `.skpt` save also streams each entry straight from the tensors' storage, with no managed copy of
the training state. An entry compressed with `WithZstdCompressedData` is compressed as it streams,
afresh on each pass the writer makes over it, so it is never held whole either; its length is known
only once it is compressed, so its tensors are read three times rather than two. An entry and an
archive may be of any size: the single file writes Zip64 records where a size or offset is
4,294,967,295 bytes (4 GiB − 1) or more, or where there are 65,535 entries or more. A load reads each data entry forward, a tensor at a
time, exactly as the flat load does.

A checkpoint in device memory is saved from there: the flat and `.skpt` saves (file and directory
form) write each tensor through one bounded host staging buffer (8 MiB), piece by piece, so the
state is never whole in host memory. On ONNX Runtime CUDA the CUDA runtime copies the pieces, and
PyTorch and JAX copy them with their own operations; JAX brings a tensor no larger than the buffer
home whole. The `.skpt` save reads each device tensor twice, once to hash it for the manifest
and once to write it, and binds the model it writes from the weights' shapes and dtypes, copying
only the smallest weights to the host. There is no direct device-to-disk path.

Loading is the same in reverse. A checkpoint loaded for a rig that trains on a device
(`rig.LoadCheckpoint`, `rig.LoadCheckpointFromSkpt`) reads its state from the file straight into
the rig's device memory, through the same bounded buffer, and comes back device-resident, as a
trained one does; the counters and the history are read on the host. A compressed `.skpt` entry is
decoded as it streams.

### Bind trained weights into an inference model

```csharp
var concrete = result.FinalCheckpoint.ToInferenceModel();   // no graph to re-supply
var output   = ComputeContext.Default.Execute(concrete, myInput);
```

A saved training `.skpt` needs no rig: `Persistence.Load(path)` returns the runnable model and
`Persistence.LoadEvaluationModel(path)` the model composed with its loss, for validation. Use
`TrainingRig.Load(path)` only to keep training. A flat safetensors file has no architecture; a
checkpoint from `Persistence.LoadTrainingCheckpoint` needs a rig to bind.

```csharp
var model = Persistence.Load("run.skpt");                  // ConcreteModel, weights bound
var eval  = Persistence.LoadEvaluationModel("run.skpt");   // [model inputs…, targets] → loss
```

Each run of the evaluation model gives one batch's loss. Under a `Mean` loss with `ignoreIndex`,
that figure is a mean over the batch's own targets that are not ignored (with a class `weight`,
over their total weight), so batches are not interchangeable: weight each batch's figure by that
count (or total weight) before combining, or give the loss `reduction: LossReduction.Sum` and
divide the batches' total once, at the end, by the whole set's count of targets that are not
ignored (with a class `weight`, by their total weight).

`ToInferenceModel()` binds the checkpoint's trainable params and model state into its `.Rig`'s
retained concrete architecture (concretized once at build, at all inputs, so multi-input models
work) — no re-concretization or sample inputs. It requires an attached rig; use
`rig.AdoptCheckpoint(checkpoint)` for a bare checkpoint. It copies each value into the model, so
the checkpoint stays usable, and binds a parameter of any size: one past 2 GiB is copied 8 MiB at
a time into host memory of the backend `ComputeContext.Default` runs on, where the model holds it.
