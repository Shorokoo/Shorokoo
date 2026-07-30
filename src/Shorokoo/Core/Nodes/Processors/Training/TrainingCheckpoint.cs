using Shorokoo.Runtime;
using Shorokoo.Graph;
using Shorokoo.Onnx;
using Shorokoo.Core;
using Shorokoo.Core.Utils;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Shorokoo
{
    /// <summary>
    /// Selects which parts of a <see cref="TrainingCheckpoint"/> a save writes or a load reads. Combine
    /// with <c>|</c>; pass <c>null</c> to <see cref="TrainingCheckpoint.Save(string, CheckpointComponents?)"/>
    /// / <see cref="TrainingCheckpoint.Load"/> for "every available component" on save and "everything
    /// present" on load.
    /// </summary>
    [Flags]
    public enum CheckpointComponents
    {
        /// <summary>No component.</summary>
        None = 0,
        /// <summary>The rig's constituent model/loss/optimizer graphs, hyperparameters and RNG config —
        /// enough to rebuild the whole rig from the file alone. Serialization not yet implemented
        /// (Shorokoo/Shorokoo#115); requesting it throws.</summary>
        TrainingRig = 1 << 0,
        /// <summary>Trainable parameters plus model state — everything the inference model binds.</summary>
        InferenceState = 1 << 1,
        /// <summary>Optimizer state (moment buffers, scalar timesteps, …).</summary>
        OptimizerState = 1 << 2,
        /// <summary>The host-owned run counters: step, epoch, batch index.</summary>
        Counters = 1 << 3,
        /// <summary>The host-owned run-progress loss scalar of the step that produced the checkpoint
        /// (a nullable value — <c>null</c> on an initial/bare checkpoint contributes nothing to a
        /// save). Independent of <see cref="Counters"/>.</summary>
        Loss = 1 << 4,
        /// <summary>Every component.</summary>
        All = TrainingRig | InferenceState | OptimizerState | Counters | Loss,
    }

    /// <summary>
    /// Holds the full training state between training steps.
    /// </summary>
    public class TrainingCheckpoint
    {
        /// <summary>Current trainable parameter values (fields per <see cref="TrainingRig.TrainableParamStructDef"/>).</summary>
        public TensorDataStruct TrainableParams { get; }

        /// <summary>Current model state values (empty struct for stateless models).</summary>
        public TensorDataStruct ModelState { get; }

        /// <summary>Current optimizer state values, e.g. moment buffers (empty for basic SGD).</summary>
        public TensorDataStruct OptimizerState { get; }

        /// <summary>
        /// The 0-based global training step this checkpoint sits at. Each
        /// <see cref="TrainingRig.TrainStep(TrainingCheckpoint, TensorDataStruct, TensorDataStruct)"/>
        /// advances it by one and the rig evaluates scheduled hyperparameters at this step, so a
        /// schedule resumes correctly from a saved checkpoint.
        /// </summary>
        public long Step { get; }

        /// <summary>
        /// The 0-based epoch counter this checkpoint sits at — a host-owned run counter the
        /// training loop advances (the graph never does), persisted so a resumed run restores
        /// its position in the data schedule — or <c>null</c> when it is genuinely <b>unknown</b>: a
        /// checkpoint produced without a data loader or an explicit epoch (an initial checkpoint, or one
        /// trained through <see cref="TrainingRig.Train"/> / <see cref="TrainingRig.Fit(TensorDataStruct[], TensorDataStruct[], int, TrainingCheckpoint?)"/>
        /// / the counter-agnostic <c>TrainStep</c>) carries <c>null</c> rather than a misleading <c>0</c>.
        /// The loader-driven and explicit-counter paths (<see cref="TrainingRig.Fit(IDataLoader, int, TrainingCheckpoint?)"/>,
        /// <see cref="TrainingRig.TrainStep(TrainingCheckpoint, IDataLoader)"/>,
        /// <see cref="TrainingRig.TrainStep(TrainingCheckpoint, TensorDataStruct, TensorDataStruct, long, long)"/>)
        /// set a concrete value. Persisted as its own presence-gated part of the
        /// <see cref="CheckpointComponents.Counters"/> component (absent on disk ⇒ <c>null</c>, never a
        /// sentinel 0). A scheduled hyperparameter reading the epoch counter sees <c>0</c> for a null epoch.
        /// </summary>
        public long? Epoch { get; }

        /// <summary>
        /// The 0-based batch index within the current epoch — a host-owned run counter the
        /// training loop advances (the graph never does), persisted for exact resume — or <c>null</c> when
        /// genuinely <b>unknown</b>, on the same terms as <see cref="Epoch"/> (no loader / no explicit
        /// counter ⇒ <c>null</c>, never a sentinel 0). The loader-driven and explicit-counter paths set a
        /// concrete value; a scheduled hyperparameter reading the batch counter sees <c>0</c> for a null value.
        /// </summary>
        public long? BatchIndex { get; }

        /// <summary>
        /// The <see cref="TrainingRig"/> this checkpoint belongs to, or <c>null</c> for a bare
        /// checkpoint constructed without one. Every rig-produced checkpoint carries its rig
        /// (<see cref="TrainingRig.CreateInitialCheckpoint()"/>, <see cref="TrainingRig.TrainStep(TrainingCheckpoint, TensorDataStruct, TensorDataStruct)"/>,
        /// <see cref="TrainingRig.Train"/>/<see cref="TrainingRig.Fit(TensorDataStruct[], TensorDataStruct[], int, TrainingCheckpoint?)"/>, load, and
        /// <see cref="TrainingRig.AdoptCheckpoint"/> all set it), so <see cref="ToInferenceModel()"/>
        /// can extract the inference model with no re-supplied graph. The rig does not store
        /// checkpoints, so there is no reference cycle. Attach one to a bare checkpoint via
        /// <see cref="TrainingRig.AdoptCheckpoint"/>.
        /// </summary>
        public TrainingRig? Rig { get; }

        /// <summary>
        /// The loss computed for the training step that produced this checkpoint, or <c>null</c> on an
        /// initial or bare checkpoint that no step produced. Set by
        /// <see cref="TrainingRig.TrainStep(TrainingCheckpoint, TensorDataStruct, TensorDataStruct)"/>
        /// (which now returns the post-step checkpoint directly) to that step's loss. Carried
        /// unchanged through the counter derivations, and persisted as its own
        /// <see cref="CheckpointComponents.Loss"/> component, independent of the counters (absent, or a
        /// null loss, ⇒ reads back <c>null</c>).
        /// </summary>
        public float? Loss { get; }

        /// <summary>Packages trainable params, model state and optimizer state at
        /// <paramref name="step"/> / <paramref name="epoch"/> / <paramref name="batchIndex"/>,
        /// optionally attaching the producing <paramref name="rig"/> and the <paramref name="loss"/>
        /// of the step that produced it. <paramref name="epoch"/> and <paramref name="batchIndex"/>
        /// default to <c>null</c> — "unknown", the right value for a checkpoint with no data-loader or
        /// explicit position (see <see cref="Epoch"/>); pass concrete values only when the position is known.</summary>
        public TrainingCheckpoint(
            TensorDataStruct trainableParams,
            TensorDataStruct modelState,
            TensorDataStruct optimizerState,
            long step = 0,
            long? epoch = null,
            long? batchIndex = null,
            TrainingRig? rig = null,
            float? loss = null)
        {
            TrainableParams = trainableParams ?? throw new ArgumentNullException(nameof(trainableParams));
            ModelState = modelState ?? throw new ArgumentNullException(nameof(modelState));
            OptimizerState = optimizerState ?? throw new ArgumentNullException(nameof(optimizerState));
            Step = step;
            Epoch = epoch;
            BatchIndex = batchIndex;
            Rig = rig;
            Loss = loss;
        }

        // ---- Counter derivations (§5.8.5): step/epoch/batch are host-owned scalars, not rig
        // state, so resetting one yields a NEW checkpoint value carrying the same trainable
        // params / model state / optimizer state (shared by reference — nothing is re-derived).
        // The receiver is never mutated. ----

        /// <summary>
        /// Returns a new checkpoint identical to this one but with the given host-owned run
        /// counter(s) set (each defaulting to this checkpoint's current value when omitted). The
        /// tensor state — trainable params, model state, optimizer state — is shared by reference,
        /// since counters are not graph state (§5.8.1). The receiver is unchanged.
        ///
        /// <para>An omitted (<c>null</c>) argument keeps the current value — which, for
        /// <paramref name="epoch"/> / <paramref name="batchIndex"/>, may itself be <c>null</c> (unknown).
        /// This "null means keep" convention means the method sets a concrete value or leaves the current
        /// one; it does not reset a set counter back to <c>null</c> (nothing in the run needs to — an
        /// unknown counter only ever arises at construction, then a loader / explicit step gives it a
        /// concrete value that is carried forward).</para>
        /// </summary>
        public TrainingCheckpoint WithCounters(long? step = null, long? epoch = null, long? batchIndex = null)
            => new(TrainableParams, ModelState, OptimizerState,
                step ?? Step, epoch ?? Epoch, batchIndex ?? BatchIndex, Rig, Loss);

        /// <summary>A new checkpoint with <see cref="Step"/> set (epoch/batch carried through).</summary>
        public TrainingCheckpoint WithStep(long step) => WithCounters(step: step);

        /// <summary>A new checkpoint with <see cref="Epoch"/> set (step/batch carried through).</summary>
        public TrainingCheckpoint WithEpoch(long epoch) => WithCounters(epoch: epoch);

        /// <summary>A new checkpoint with <see cref="BatchIndex"/> set (step/epoch carried through).</summary>
        public TrainingCheckpoint WithBatchIndex(long batchIndex) => WithCounters(batchIndex: batchIndex);

        // ---- Inference: bind trained weights into a concrete model for execution ----

        /// <summary>
        /// Builds a concrete inference model from this checkpoint's trained weights in one call.
        /// Requires an attached <see cref="Rig"/>: the checkpoint's trainable params and model state
        /// are bound by canonical identity into the rig's <b>retained concrete architecture</b> — the
        /// one the rig concretized once at build time (at all its inputs, so a multi-input model is
        /// supported), held on the rig and reused, never re-concretized and needing no sample inputs.
        /// Attach a rig first via <see cref="TrainingRig.AdoptCheckpoint"/> — or load the checkpoint
        /// against a rig — if this one has none.
        /// </summary>
        public ComputationGraph ToInferenceModel()
        {
            if (Rig is null)
                throw new InvalidOperationException(
                    "ToInferenceModel() requires a training rig, but this checkpoint has none attached. " +
                    "Adopt one via rig.AdoptCheckpoint(checkpoint), or load the checkpoint against a rig " +
                    "(rig.LoadCheckpoint(path)), then call ToInferenceModel().");
            return Rig.ExtractInferenceModel(this);
        }

        // ---- Persistence: save a checkpoint to disk and resume across process restarts ----

        // The three sections share one SafeTensors file; each field is namespaced as
        // "<section>/<fieldName>". A Shorokoo field name never contains '/', so the split
        // is unambiguous and the '/'-free marker tensor below can't be mistaken for a field.
        // Internal (not private) so Persistence.Inspect recognizes checkpoints by the same
        // marker/section names the writer uses — one definition, no drift.
        internal const string TrainableSection = "trainable";
        internal const string ModelStateSection = "model_state";
        internal const string OptimizerStateSection = "opt_state";
        internal const string CheckpointMarkerName = "__shorokoo_checkpoint__"; // int64[2] = [version, step]
        // The loss (a host-owned run-progress scalar, its OWN savable component) is written as a
        // presence-gated float32 scalar tensor, only when Loss.HasValue and the Loss component is
        // included — never overloading the int64 marker, so a null loss is genuinely absent (not a
        // sentinel 0.0). A '/'-free name, so it can't be mistaken for a namespaced section field.
        internal const string CheckpointLossName = "__shorokoo_loss__";
        // Epoch and batch index are host-owned run counters, each its own presence-gated int64 scalar
        // beside the marker (part of the Counters component), written only when non-null — so an unknown
        // epoch/batch is genuinely absent on disk (⇒ reads back null), never a sentinel 0. Same
        // '/'-free-name discipline as the loss/marker, so they can't be mistaken for section fields.
        internal const string CheckpointEpochName = "__shorokoo_epoch__";
        internal const string CheckpointBatchName = "__shorokoo_batch__";
        // Version 3: the int64 marker carries only [version, step] (always present); epoch and batch
        // moved out into their own presence-gated int64 scalars (v3), so an unknown epoch/batch reads
        // back null instead of a misleading 0. Version 2 added the presence-gated loss tensor beside the
        // then-int64[4] marker. No released users, so no back-compat shim — older files are neither
        // produced nor read.
        internal const long CheckpointFormatVersion = 3;

        /// <summary>
        /// Saves this checkpoint to a single SafeTensors file so training can resume across process
        /// restarts. Every trainable-parameter, model-state, and optimizer-state field is written as
        /// a namespaced tensor, alongside the host-owned run counters <see cref="Step"/>,
        /// <see cref="Epoch"/>, and <see cref="BatchIndex"/> (so schedules resume from the right step
        /// and the run resumes at the right point in its data schedule). Reload with
        /// <see cref="TrainingRig.LoadCheckpoint(string, CheckpointComponents?)"/> — or with
        /// <see cref="Load"/> against the same rig. A <c>.safetensors</c> extension is
        /// conventional. Fields must be plain tensors (nested-struct fields are unsupported); rank-0
        /// scalars are fine — they serialize as the SafeTensors empty-shape encoding (e.g. an
        /// optimizer's scalar timestep).
        ///
        /// <para>
        /// <paramref name="components"/> selects which parts to write. <c>null</c> writes every
        /// <b>available</b> component: <see cref="CheckpointComponents.InferenceState"/> (trainable
        /// params + model state) and <see cref="CheckpointComponents.Counters"/> always,
        /// <see cref="CheckpointComponents.OptimizerState"/> when this checkpoint carries any, and
        /// <see cref="CheckpointComponents.Loss"/> when it carries a (non-null) loss. Explicitly
        /// requesting <see cref="CheckpointComponents.Loss"/> on a checkpoint whose loss is
        /// <c>null</c> is a no-op — it writes nothing, and does not throw (a null loss is a
        /// legitimate value). The
        /// <see cref="CheckpointComponents.TrainingRig"/> component is never written automatically and
        /// throws when requested explicitly — serializing the rig's constituent graphs is not yet
        /// implemented (Shorokoo/Shorokoo#115).
        /// </para>
        ///
        /// <para>
        /// The write is atomic: the checkpoint is staged to a temp file in the target directory and
        /// committed by rename, so a crash or power loss mid-save never corrupts an existing
        /// checkpoint at <paramref name="filePath"/> — either the old or the new content survives.
        /// The directory must already exist.
        /// </para>
        /// </summary>
        public void Save(string filePath, CheckpointComponents? components = null)
        {
            if (string.IsNullOrWhiteSpace(filePath))
                throw new ArgumentException("Checkpoint path cannot be null or empty.", nameof(filePath));

            var comps = ResolveSaveComponents(components);
            AtomicFileWriter.WriteFile(
                filePath, stream => SafeTensorLoader.SaveSafeTensorsToStream(stream, BuildCheckpointTensors(comps)));
        }

        /// <summary>
        /// Resolves the effective component set for a save. <c>null</c> ⇒ every available component
        /// (never the <see cref="CheckpointComponents.TrainingRig"/> one, whose serialization is
        /// unimplemented — #115). Explicitly requesting <see cref="CheckpointComponents.TrainingRig"/>
        /// throws: <see cref="InvalidOperationException"/> when no rig is attached, otherwise a
        /// <see cref="NotSupportedException"/> naming #115.
        /// </summary>
        private CheckpointComponents ResolveSaveComponents(CheckpointComponents? requested)
        {
            if (requested is CheckpointComponents c)
            {
                if ((c & CheckpointComponents.TrainingRig) != 0)
                {
                    if (Rig is null)
                        throw new InvalidOperationException(
                            "Cannot save the TrainingRig component: this checkpoint has no rig attached " +
                            "(see TrainingCheckpoint.Rig / TrainingRig.AdoptCheckpoint).");
                    throw new NotSupportedException(
                        "Saving the TrainingRig component — the rig's constituent model/loss/optimizer " +
                        "graphs, hyperparameters and RNG config — is not yet implemented (Shorokoo/Shorokoo#115). " +
                        "Save InferenceState / OptimizerState / Counters (the default) and rebuild the rig " +
                        "from its source graphs to resume.");
                }
                return c;
            }

            // null ⇒ all AVAILABLE components. TrainingRig is never auto-included (its serialization is
            // unimplemented, #115); request it explicitly to get the clear #115 error. Loss is included
            // only when this checkpoint actually carries one (a null loss contributes nothing).
            var comps = CheckpointComponents.InferenceState | CheckpointComponents.Counters;
            if (OptimizerState.Definition.Fields.Length > 0) comps |= CheckpointComponents.OptimizerState;
            if (Loss.HasValue) comps |= CheckpointComponents.Loss;
            return comps;
        }

        /// <summary>Serializes the requested namespaced sections plus the checkpoint marker.</summary>
        private List<SafeTensor> BuildCheckpointTensors(CheckpointComponents comps)
        {
            var tensors = new List<SafeTensor>();
            if ((comps & CheckpointComponents.InferenceState) != 0)
            {
                AppendSection(tensors, TrainableSection, TrainableParams);
                AppendSection(tensors, ModelStateSection, ModelState);
            }
            if ((comps & CheckpointComponents.OptimizerState) != 0)
                AppendSection(tensors, OptimizerStateSection, OptimizerState);

            // The marker always identifies the file as a Shorokoo checkpoint and carries the format
            // version plus the step (a graph-advanced counter that is always concrete). Its VALUE for
            // step is written only when the Counters component is included (else 0, so a counters-less
            // save reloads at step 0).
            bool counters = (comps & CheckpointComponents.Counters) != 0;
            var marker = Globals.TensorData(
                [2L], CheckpointFormatVersion, counters ? Step : 0L);
            tensors.Add(new SafeTensor(CheckpointMarkerName, marker, "I64", [2L]));

            // Epoch and batch index are host-owned run counters that may be genuinely unknown (null).
            // Each is written as its own presence-gated int64 scalar beside the marker — only when the
            // Counters component is included AND the value is non-null — so an unknown epoch/batch is
            // absent on disk and reloads as null (never a sentinel 0), mirroring the loss treatment.
            if (counters && Epoch is long epochValue)
            {
                var epochTensor = Globals.TensorData(Array.Empty<long>(), epochValue);
                tensors.Add(new SafeTensor(
                    CheckpointEpochName, epochTensor,
                    SafeTensorLoader.DTypeToSafeTensorDType(epochTensor.DType), epochTensor.Shape.Dims));
            }
            if (counters && BatchIndex is long batchValue)
            {
                var batchTensor = Globals.TensorData(Array.Empty<long>(), batchValue);
                tensors.Add(new SafeTensor(
                    CheckpointBatchName, batchTensor,
                    SafeTensorLoader.DTypeToSafeTensorDType(batchTensor.DType), batchTensor.Shape.Dims));
            }

            // Loss is its own savable component (a host-owned run-progress scalar), independent of the
            // counters. Written only when this checkpoint actually carries a loss AND the Loss
            // component is included — so a Loss-less save, or an initial/bare checkpoint with no loss,
            // writes no loss tensor and reloads with Loss == null (never a sentinel 0.0). Explicitly
            // requesting Loss on a null-loss checkpoint is a no-op (writes nothing), not an error — a
            // null loss is a legitimate value, unlike the unavailable TrainingRig case.
            if ((comps & CheckpointComponents.Loss) != 0 && Loss is float lossValue)
            {
                var lossTensor = Globals.TensorData(Array.Empty<long>(), lossValue);
                tensors.Add(new SafeTensor(
                    CheckpointLossName, lossTensor,
                    SafeTensorLoader.DTypeToSafeTensorDType(lossTensor.DType), lossTensor.Shape.Dims));
            }
            return tensors;
        }

        private static void AppendSection(List<SafeTensor> tensors, string section, TensorDataStruct data)
        {
            foreach (var fieldDef in data.Definition.Fields)
            {
                if (data.Fields[fieldDef.Name] is not TensorData td)
                    throw new NotSupportedException(
                        $"Checkpoint field '{section}/{fieldDef.Name}' is not a plain tensor; nested-struct " +
                        "fields are not supported by checkpoint serialization.");
                tensors.Add(new SafeTensor(
                    $"{section}/{fieldDef.Name}", td,
                    SafeTensorLoader.DTypeToSafeTensorDType(td.DType), td.Shape.Dims));
            }
        }

        /// <summary>
        /// Loads a checkpoint against <paramref name="rig"/>, whose struct definitions the sections are
        /// reconstructed against and whose model/loss/optimizer this checkpoint is attached to
        /// (<see cref="Rig"/> is set on the result). Reads either on-disk shape — the legacy flat
        /// sectioned-safetensors file (<see cref="Save(string, CheckpointComponents?)"/>) or the native
        /// <c>.skpt</c> container — detected automatically. <paramref name="components"/> selects which
        /// parts to load; <c>null</c> loads everything present. A component not present in the file is
        /// filled from the rig's initial values (counters default to 0). Explicitly requesting the
        /// <see cref="CheckpointComponents.TrainingRig"/> component (including via
        /// <see cref="CheckpointComponents.All"/>) throws a <see cref="NotSupportedException"/> naming
        /// #115 — no file stores the rig's constituent graphs yet, so the request cannot be satisfied;
        /// pass the rig and omit the flag. Throws if the file is not a
        /// Shorokoo checkpoint, was written by a newer format, or its fields don't match the rig (e.g.
        /// a checkpoint from a different model or optimizer). Prefer
        /// <see cref="TrainingRig.LoadCheckpoint(string, CheckpointComponents?)"/>.
        ///
        /// <para>A <paramref name="rig"/> is required: the struct definitions come from it. Rebuilding
        /// the rig from the checkpoint file alone is not yet implemented (Shorokoo/Shorokoo#115), so a
        /// <c>null</c> rig throws.</para>
        /// </summary>
        public static TrainingCheckpoint Load(
            string filePath,
            TrainingRig? rig = null,
            CheckpointComponents? components = null)
        {
            if (string.IsNullOrWhiteSpace(filePath))
                throw new ArgumentException("Checkpoint path cannot be null or empty.", nameof(filePath));
            if (rig is null)
                throw new InvalidOperationException(
                    "TrainingCheckpoint.Load requires a rig to resolve the checkpoint's struct definitions. " +
                    "Pass the rig you are resuming (or use rig.LoadCheckpoint(path)). Reconstructing the rig " +
                    "from the checkpoint file itself is not yet implemented (Shorokoo/Shorokoo#115).");

            // Explicitly requesting the TrainingRig component (including via CheckpointComponents.All)
            // throws, symmetric with Save (ResolveSaveComponents): the file never stores the rig's
            // constituent graphs — that serialization is unimplemented (#115) — so the request cannot
            // be satisfied. null ⇒ "everything present" is the way to load without naming the flag.
            if (components is CheckpointComponents cReq && (cReq & CheckpointComponents.TrainingRig) != 0)
                throw new NotSupportedException(
                    "Cannot load the TrainingRig component — the rig's constituent model/loss/optimizer " +
                    "graphs, hyperparameters and RNG config — because that serialization is not yet " +
                    "implemented (Shorokoo/Shorokoo#115) and no checkpoint file stores it. Pass the rig to " +
                    "Load (or use rig.LoadCheckpoint(path)) and omit the TrainingRig flag; null components " +
                    "loads every component the file contains.");

            var raw = Persistence.IsSkptFile(filePath)
                ? Persistence.LoadTrainingCheckpointFromSkpt(
                    filePath, rig.TrainableParamStructDef, rig.ModelStateDef, rig.OptimizerStateDef,
                    components, rig)
                : LoadFlat(
                    filePath, rig.TrainableParamStructDef, rig.ModelStateDef, rig.OptimizerStateDef,
                    components, rig);
            // Attach the rig (sets Rig, preserves counters); the raw checkpoint was read against the
            // rig's own defs, so the compatibility check inside AdoptCheckpoint always passes.
            return rig.AdoptCheckpoint(raw);
        }

        /// <summary>
        /// Reads the legacy flat sectioned-safetensors checkpoint against the given defs, honoring
        /// <paramref name="components"/> (null ⇒ everything present). A component not present in the
        /// file (or not requested) is filled from <paramref name="rigForDefaults"/>'s initial values
        /// when a rig is supplied; without a rig, an absent-but-expected section fails loud.
        /// </summary>
        internal static TrainingCheckpoint LoadFlat(
            string filePath,
            TensorStructDef trainableParamDef,
            TensorStructDef modelStateDef,
            TensorStructDef optimizerStateDef,
            CheckpointComponents? components,
            TrainingRig? rigForDefaults)
        {
            if (trainableParamDef is null) throw new ArgumentNullException(nameof(trainableParamDef));
            if (modelStateDef is null) throw new ArgumentNullException(nameof(modelStateDef));
            if (optimizerStateDef is null) throw new ArgumentNullException(nameof(optimizerStateDef));

            var byName = SafeTensorLoader.LoadSafeTensors(filePath).ToDictionary(t => t.Name, t => t.Data);

            if (!byName.TryGetValue(CheckpointMarkerName, out var markerData))
                throw new InvalidOperationException(
                    $"'{filePath}' is not a Shorokoo training checkpoint (missing '{CheckpointMarkerName}' marker).");

            var marker = markerData.As<int64>().AccessMemory<long>();
            // The marker is a fixed int64[2] = [version, step]. Exactly one format version exists (3);
            // a wrong shape or version is unreadable by this build. (v3 moves epoch/batch out of the
            // marker into presence-gated int64 scalars; there are no released v1/v2 files.)
            if (marker.Length != 2)
                throw new InvalidOperationException(
                    $"'{filePath}' has a malformed checkpoint marker: expected 2 int64 elements " +
                    $"[version, step], found {marker.Length}.");
            if (marker[0] != CheckpointFormatVersion)
                throw new InvalidOperationException(
                    $"Unsupported checkpoint format version {marker[0]}; this build reads version " +
                    $"{CheckpointFormatVersion} only.");

            bool Want(CheckpointComponents c) => components is null || (components.Value & c) != 0;
            bool SectionPresent(string section)
            {
                var prefix = section + "/";
                foreach (var k in byName.Keys)
                    if (k.StartsWith(prefix, StringComparison.Ordinal)) return true;
                return false;
            }

            long step = Want(CheckpointComponents.Counters) ? marker[1] : 0L;
            // Epoch and batch index ride with the Counters component and are each their own
            // presence-gated int64 scalar: read only when Counters is wanted and the scalar is actually
            // present; absence ⇒ null (an unknown position — no loader / no explicit counter was set),
            // never a sentinel 0.
            long? epoch = Want(CheckpointComponents.Counters)
                          && byName.TryGetValue(CheckpointEpochName, out var epochData)
                ? epochData.As<int64>().AccessMemory<long>()[0]
                : (long?)null;
            long? batchIndex = Want(CheckpointComponents.Counters)
                               && byName.TryGetValue(CheckpointBatchName, out var batchData)
                ? batchData.As<int64>().AccessMemory<long>()[0]
                : (long?)null;

            // Loss is its own component (independent of the counters): read it only when Loss is
            // wanted and the presence-gated loss tensor is actually present; absence ⇒ null (an
            // initial/bare or Loss-less checkpoint), never a sentinel 0.
            float? loss = Want(CheckpointComponents.Loss)
                          && byName.TryGetValue(CheckpointLossName, out var lossData)
                ? lossData.As<float32>().AccessMemory<float>()[0]
                : (float?)null;

            TensorDataStruct trainable, modelState, optState;

            if (Want(CheckpointComponents.InferenceState)
                && (SectionPresent(TrainableSection) || trainableParamDef.Fields.Length == 0))
            {
                trainable = ReadSection(byName, TrainableSection, trainableParamDef, filePath);
                modelState = ReadSection(byName, ModelStateSection, modelStateDef, filePath);
            }
            else if (rigForDefaults is not null)
            {
                trainable = rigForDefaults.InitialTrainableStruct;
                modelState = rigForDefaults.InitialModelStateStruct;
            }
            else
            {
                trainable = ReadSection(byName, TrainableSection, trainableParamDef, filePath);
                modelState = ReadSection(byName, ModelStateSection, modelStateDef, filePath);
            }

            if (Want(CheckpointComponents.OptimizerState)
                && (SectionPresent(OptimizerStateSection) || optimizerStateDef.Fields.Length == 0))
            {
                optState = ReadSection(byName, OptimizerStateSection, optimizerStateDef, filePath);
            }
            else if (rigForDefaults is not null)
            {
                optState = rigForDefaults.InitialOptimizerStateStruct;
            }
            else
            {
                optState = ReadSection(byName, OptimizerStateSection, optimizerStateDef, filePath);
            }

            return new TrainingCheckpoint(trainable, modelState, optState, step, epoch, batchIndex, rig: null, loss: loss);
        }

        private static TensorDataStruct ReadSection(
            IReadOnlyDictionary<string, TensorData> byName, string section, TensorStructDef def, string filePath)
        {
            var fields = new List<KeyValuePair<string, IData>>(def.Fields.Length);
            foreach (var fieldDef in def.Fields)
            {
                var key = $"{section}/{fieldDef.Name}";
                if (!byName.TryGetValue(key, out var td))
                    throw new InvalidOperationException(
                        $"Checkpoint '{filePath}' is missing field '{key}'. Does it match this model/optimizer?");
                if (fieldDef.Rank is int rank && td.Shape.Dims.Length != rank)
                    throw new InvalidOperationException(
                        $"Checkpoint field '{key}' has rank {td.Shape.Dims.Length}, expected {rank}.");
                fields.Add(new KeyValuePair<string, IData>(fieldDef.Name, td));
            }

            // Reject stray tensors namespaced into this section — a sign of a mismatched checkpoint.
            var prefix = section + "/";
            foreach (var name in byName.Keys)
            {
                if (!name.StartsWith(prefix, StringComparison.Ordinal)) continue;
                var fieldName = name.Substring(prefix.Length);
                if (def.GetField(fieldName) is null)
                    throw new InvalidOperationException(
                        $"Checkpoint '{filePath}' has unexpected field '{name}' not in this model/optimizer's '{section}' definition.");
            }

            return new TensorDataStruct(def, fields);
        }
    }
}
