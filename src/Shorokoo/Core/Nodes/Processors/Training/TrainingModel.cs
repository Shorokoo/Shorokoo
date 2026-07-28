using Shorokoo;
using Shorokoo.Core.Graph;
using Shorokoo.Runtime;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Onnx;
using Shorokoo.Core.AutoDiffCheckpointing;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Training;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Nodes.Processors.Training;
using Shorokoo.Core.Utils;
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
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
        /// <see cref="TrainingRig.TrainStep(TrainingCheckpoint, TensorDataStruct, TensorDataStruct, CompiledGraph)"/>
        /// advances it by one and the rig evaluates scheduled hyperparameters at this step, so a
        /// schedule resumes correctly from a saved checkpoint.
        /// </summary>
        public long Step { get; }

        /// <summary>
        /// The 0-based epoch counter this checkpoint sits at — a host-owned run counter the
        /// training loop advances (the graph never does), persisted so a resumed run restores
        /// its position in the data schedule — or <c>null</c> when it is genuinely <b>unknown</b>: a
        /// checkpoint produced without a data loader or an explicit epoch (an initial checkpoint, or one
        /// trained through <see cref="TrainingRig.Train"/> / <see cref="TrainingRig.Fit(TensorDataStruct[], TensorDataStruct[], int, TrainingCheckpoint?, ComputeContext?)"/>
        /// / the counter-agnostic <c>TrainStep</c>) carries <c>null</c> rather than a misleading <c>0</c>.
        /// The loader-driven and explicit-counter paths (<see cref="TrainingRig.Fit(IDataLoader, int, TrainingCheckpoint?, ComputeContext?)"/>,
        /// <see cref="TrainingRig.TrainStep(TrainingCheckpoint, IDataLoader, CompiledGraph)"/>,
        /// <see cref="TrainingRig.TrainStep(TrainingCheckpoint, TensorDataStruct, TensorDataStruct, long, long, CompiledGraph)"/>)
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
        /// (<see cref="TrainingRig.CreateInitialCheckpoint()"/>, <see cref="TrainingRig.TrainStep(TrainingCheckpoint, TensorDataStruct, TensorDataStruct, CompiledGraph)"/>,
        /// <see cref="TrainingRig.Train"/>/<see cref="TrainingRig.Fit(TensorDataStruct[], TensorDataStruct[], int, TrainingCheckpoint?, ComputeContext?)"/>, load, and
        /// <see cref="TrainingRig.AdoptCheckpoint"/> all set it), so <see cref="ToInferenceModel()"/>
        /// can extract the inference model with no re-supplied graph. The rig does not store
        /// checkpoints, so there is no reference cycle. Attach one to a bare checkpoint via
        /// <see cref="TrainingRig.AdoptCheckpoint"/>.
        /// </summary>
        public TrainingRig? Rig { get; }

        /// <summary>
        /// The loss computed for the training step that produced this checkpoint, or <c>null</c> on an
        /// initial or bare checkpoint that no step produced. Set by
        /// <see cref="TrainingRig.TrainStep(TrainingCheckpoint, TensorDataStruct, TensorDataStruct, CompiledGraph)"/>
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

    /// <summary>
    /// Result of a full training run (multiple epochs).
    /// </summary>
    public class TrainingResult
    {
        /// <summary>The checkpoint after the last epoch.</summary>
        public TrainingCheckpoint FinalCheckpoint { get; }
        /// <summary>Mean loss per epoch, in epoch order.</summary>
        public float[] EpochLosses { get; }

        /// <summary>Packages the final checkpoint and the per-epoch losses.</summary>
        public TrainingResult(TrainingCheckpoint finalCheckpoint, float[] epochLosses)
        {
            FinalCheckpoint = finalCheckpoint;
            EpochLosses = epochLosses;
        }
    }

    /// <summary>
    /// Builds and manages a training pipeline by composing model, loss, autograd, and optimizer
    /// into a single TrainingStepPureGraph — a stateless computation graph that performs one
    /// training step.
    /// 
    /// The TrainingStepPureGraph contains no embedded state (no trainable parameters, no model
    /// state). All state flows through inputs and outputs as TensorStructs:
    /// 
    /// Inputs:  trainable_params, model_state, optimizer_state, [hyperparams], [step], training_inputs, training_targets
    /// Outputs: updated_trainable_params, updated_model_state, updated_optimizer_state, loss
    ///
    /// Optimizer hyperparameters are baked in as constants by default. A scheduled hyperparameter
    /// (a built-in <see cref="Schedule"/> or a scheduler module) is instead computed in-graph from the
    /// int64 "step" counter input each step — no recompilation and no host evaluation. A schedule-less
    /// <see cref="Hyperparameter.Runtime"/> hyperparameter is routed as a runtime "hyperparams" input (see
    /// <see cref="HyperparameterStructDef"/>) and supplied explicitly per step.
    ///
    /// The training loop calls TrainStep repeatedly, passing updated state from one step to the next.
    /// </summary>
    public class TrainingRig
    {
        /// <summary>
        /// The lowered, executable computation graph for one training step
        /// (stamped <see cref="GraphKind.ConcreteModel"/> — fully lowered and runnable).
        /// Contains no embedded state — all state flows through inputs/outputs.
        /// </summary>
        public ComputationGraph TrainingStepPureGraph { get; private set; } = null!;

        /// <summary>
        /// Construction-time working graph: built by <c>BuildTrainingStepPureGraph</c>,
        /// optimized by <c>InitializeAndOptimize</c>, then relinquished into the readonly
        /// <see cref="TrainingStepPureGraph"/> wrapper (and nulled — the wrapper owns it).
        /// </summary>
        private InternalComputationGraph? _trainingStepWorkGraph;

        /// <summary>
        /// The rig's <b>constituent</b> layer (§5.8): the swappable source-of-truth models — the
        /// inference model, the loss graph, the optimizer graph (as authored), and the scheduler
        /// (carried in the <see cref="Hyperparameters"/> until #106 folds it into its own persisted
        /// constituent) — plus the RNG config the trainstep is derived from. This
        /// is the source of truth; the <see cref="TrainingStepPureGraph"/> (<c>trainstep</c>) is the
        /// purely in-memory derived executable, composed from these and never persisted. A rig is an
        /// <b>immutable value</b>: the <c>With…</c> derivations return a NEW rig sharing the
        /// unchanged constituents by reference (via <c>record with</c>) and re-deriving only what
        /// changed — the receiver is never mutated.
        /// </summary>
        private RigConstituents _constituents = null!;

        /// <summary>
        /// The model constituent's <b>concrete architecture</b> — derived state, computed once when
        /// the rig is first built (§5.8): the model graph run through <c>ToConcreteArchitecture</c> at
        /// its inputs and bound to the RNG config, shape-specialized and with every trainable parameter
        /// visible at the top level. Like <see cref="TrainingStepPureGraph"/> it is environment-
        /// independent and NEVER persisted; unlike the trainstep it does not change when loss /
        /// optimizer / scheduler are swapped, so every <c>With…</c> derivation reuses this same graph by
        /// reference instead of re-concretizing (only <see cref="WithSeed"/> rebinds it, on a clone).
        /// It is the single substrate both the trainstep build and inference extraction read from, so
        /// weight-binding for <see cref="TrainingCheckpoint.ToInferenceModel()"/> and the <c>.skpt</c>
        /// container's self-describing model can never diverge. Consumed read-only (its consumers clone
        /// before mutating), so sharing it across derived rigs preserves immutability.
        ///
        /// <para>It is also <b>self-describing for shape inference</b>: each <c>MODEL_TENSOR_INPUT</c>
        /// node carries a zero-filled representative input on its
        /// <see cref="OnnxOpAttributeNames.ShrkAttrRepresentativeInput"/> attribute (shape+dtype-only
        /// placeholder for large inputs), recording the shape the model was concretized at. The two
        /// training-graph shape-inference sites reconstruct their <c>sampleInputs[]</c> off these
        /// attributes (<see cref="ReadRepresentativeInputs"/>), so no separate sample-input field is
        /// stored on the rig. The attribute is inert for export (a boundary node is not emitted as a
        /// NodeProto) and rides along on <c>Clone()</c>, so it survives re-seeding.</para>
        /// </summary>
        private InternalComputationGraph _concreteArch = null!;

        /// <summary>The inference-model constituent (a module graph or concrete architecture), as authored.</summary>
        public ComputationGraph ModelConstituent => _constituents.Model;

        /// <summary>The loss-graph constituent (a module graph), as authored.</summary>
        public ComputationGraph LossConstituent => _constituents.Loss;

        /// <summary>The optimizer-graph constituent (a module graph), as authored (pre-normalization).</summary>
        public ComputationGraph OptimizerConstituent => _constituents.Optimizer;

        /// <summary>
        /// The optimizer hyperparameters this rig was derived with, in the optimizer's declared order —
        /// the schedule-carrying constituent (a baked constant, a built-in <see cref="Schedule"/>, or a
        /// scheduler module per field) until #106 promotes the scheduler to its own persisted model entry.
        /// </summary>
        public IReadOnlyList<Hyperparameter> Hyperparameters => _constituents.Hyperparameters;

        /// <summary>The RNG configuration bound into the derived trainstep (never null; defaults to
        /// <see cref="RngConfig.Default"/>). Re-seed with <see cref="WithSeed"/>.</summary>
        public RngConfig RngConfig => _constituents.RngConfig;

        /// <summary>Struct definition for trainable parameters. Internal build/persistence machinery —
        /// persistence sources the defs from the rig directly, and callers drive training through the
        /// checkpoint's <see cref="TrainingCheckpoint.TrainableParams"/> rather than the def.</summary>
        internal TensorStructDef TrainableParamStructDef { get; private set; } = null!;

        /// <summary>
        /// Result of the <see cref="MemoryAwareGraphOptimizer"/> pass applied to
        /// <see cref="TrainingStepPureGraph"/>: which strategy won, the per-strategy
        /// evaluations, and the unoptimized baseline graph used as the starting point.
        /// Exposed for diagnostics — lets callers measure how much the optimizer actually
        /// improved the compute / peak-memory metric over the unoptimized graph.
        /// </summary>
        internal GraphOptimizationResult OptimizationResult { get; private set; } = null!;

        /// <summary>
        /// The unoptimized training-step graph, before <see cref="MemoryAwareGraphOptimizer"/>
        /// ran. Held alongside <see cref="OptimizationResult"/> so the per-strategy
        /// improvement is measurable.
        /// </summary>
        internal ComputationGraph PreOptimizationGraph { get; private set; } = null!;

        /// <summary>
        /// Compute time + peak memory the <see cref="GraphEvaluator"/> projected for the
        /// unoptimized <see cref="PreOptimizationGraph"/>, under the same shape inference
        /// the optimizer used. Compare with <see cref="OptimizationResult"/>'s evaluation
        /// to quantify the optimizer's improvement.
        /// </summary>
        internal GraphEvaluationResult PreOptimizationEval { get; private set; } = null!;

        /// <summary>Struct definition for model state (empty for stateless models). Internal
        /// build/persistence machinery — see <see cref="TrainableParamStructDef"/>.</summary>
        internal TensorStructDef ModelStateDef { get; private set; } = null!;

        /// <summary>Struct definition for optimizer state (empty for basic SGD). Internal
        /// build/persistence machinery — see <see cref="TrainableParamStructDef"/>.</summary>
        internal TensorStructDef OptimizerStateDef { get; private set; } = null!;

        /// <summary>
        /// Struct definition for the <b>schedule-less runtime</b> optimizer hyperparameters — the ones
        /// built with <see cref="Hyperparameter.Runtime"/> that the caller supplies explicitly each step
        /// (one scalar <c>float32</c> field each). Empty when every hyperparameter is either baked as a
        /// constant or scheduled in-graph. Scheduled hyperparameters (a built-in <see cref="Schedule"/>
        /// or a scheduler module) are <b>not</b> here — they are computed in-graph from the step counter
        /// and need no per-step value. When non-empty, supply values via
        /// <see cref="TrainStep(TrainingCheckpoint, TensorDataStruct, TensorDataStruct, TensorDataStruct, CompiledGraph)"/>.
        /// Internal build machinery — build the per-step values with <see cref="MakeHyperparameters(float)"/>
        /// (which reads this def internally); inspect the dynamic names via <see cref="DynamicHyperparameterNames"/>.
        /// </summary>
        internal TensorStructDef HyperparameterStructDef { get; private set; } = null!;

        /// <summary>
        /// Struct definition for the model's runtime inputs — one field per model input tensor,
        /// in declaration order. Use <see cref="TensorStructDef.FromOrderedData"/> to construct
        /// a <see cref="TensorDataStruct"/> for each training batch without building the definition
        /// manually: <c>rig.InputDef.FromOrderedData(TensorData([4L, 8L], myArray))</c>.
        /// </summary>
        public TensorStructDef InputDef { get; private set; } = null!;

        /// <summary>
        /// Struct definition for the loss function's target inputs — one field per non-prediction
        /// input of the loss graph, in declaration order. Use <see cref="TensorStructDef.FromOrderedData"/>
        /// to construct target batches without building the definition manually:
        /// <c>rig.TargetDef.FromOrderedData(TensorData([4L, 8L], myTargets))</c>.
        /// </summary>
        public TensorStructDef TargetDef { get; private set; } = null!;

        /// <summary>
        /// Indices into the optimizer's hyperparameter order that were routed as runtime inputs, in
        /// <see cref="HyperparameterStructDef"/> field order. Used by <see cref="MakeHyperparameters(float)"/>
        /// to map caller-supplied values to the right fields. Internal build machinery.
        /// </summary>
        internal IReadOnlyList<int> DynamicHyperparameterIndices { get; private set; } = Array.Empty<int>();

        /// <summary>
        /// The optimizer's hyperparameter names, in declaration order (e.g. <c>learningRate, beta1,
        /// …</c>). Derived from the strongly-typed hyperparameter set when one is supplied, else the
        /// fallback <c>hyperparam_{i}</c> names.
        /// </summary>
        public IReadOnlyList<string> HyperparameterNames { get; private set; } = Array.Empty<string>();

        /// <summary>
        /// The names of the dynamic (runtime-input) hyperparameters, in <see cref="HyperparameterStructDef"/>
        /// field order — the names accepted by <see cref="MakeHyperparameters(ValueTuple{string, float}[])"/>.
        /// </summary>
        public IReadOnlyList<string> DynamicHyperparameterNames { get; private set; } = Array.Empty<string>();

        /// <summary>
        /// The int64 scalar counter inputs on the training-step graph, in input order — a subset of
        /// <see cref="CounterInputNames"/> (<c>step</c>, <c>epoch</c>, <c>batchIndex</c>), the union of
        /// what the rig's scheduled hyperparameters consume (D1). Empty when no hyperparameter is
        /// scheduled. Each is fed the checkpoint's corresponding counter every <c>TrainStep</c>; the
        /// scheduler math computes the hyperparameter values from them in-graph (no host evaluation).
        /// Built-in DSL schedules consume only <c>step</c>; a scheduler module declares its subset by
        /// naming its inputs.
        /// </summary>
        private string[] _counterInputNames = Array.Empty<string>();

        /// <summary>Number of trainable parameter fields in graph outputs. Internal output-layout machinery.</summary>
        internal int UpdatedParamFieldCount { get; private set; }

        /// <summary>Number of model state fields in graph outputs. Internal output-layout machinery.</summary>
        internal int UpdatedStateFieldCount { get; private set; }

        /// <summary>Number of optimizer state fields in graph outputs. Internal output-layout machinery.</summary>
        internal int UpdatedOptimizerStateFieldCount { get; private set; }

        /// <summary>Initial trainable parameter values, computed at FromScratch time.</summary>
        private Dictionary<string, IData> _initialParamFields = null!;

        /// <summary>Initial model state values, computed at FromScratch time.</summary>
        private Dictionary<string, IData> _initialStateFields = null!;

        /// <summary>Initial optimizer state values, computed by the optimizer's state initializers.</summary>
        private Dictionary<string, IData> _initialOptStateFields = null!;

        /// <summary>
        /// Graph computing the optimizer's initial state values (one output per state field, inputs
        /// = the optimizer's [hyperparams..., currentParam, grad]); produced by
        /// <see cref="Shorokoo.Core.Nodes.Processors.Fast.FastNormalizeOptimizerGraph"/> from the
        /// optimizer's [StateInitializer] Init calls. Null for stateless optimizers.
        /// </summary>
        private InternalComputationGraph? _optimizerStateInitGraph;

        /// <summary>
        /// The value each hyperparameter contributes to optimizer state init, evaluated at the
        /// <b>initial counters</b> (step/epoch/batchIndex = 0) through the single value route (§2.5):
        /// a baked hyper's constant, a scheduled hyper's canonical graph evaluated via QEE at build
        /// (built-in schedule <i>and</i> user module alike), and <c>null</c> for a runtime hyper
        /// (its value is host-supplied — see D5). Indexed in optimizer order. Replaces the old
        /// hardcoded-<c>0f</c> state-init seed that silently fed <c>0</c> for scheduler modules.
        /// </summary>
        private float?[] _hyperparamInitialCounterValues = Array.Empty<float?>();

        /// <summary>
        /// Optimizer-order indices of the hyperparameters the optimizer's state-init graph actually
        /// <b>consumes</b> (reachable from its outputs) — the D5 dependency analysis. Empty for every
        /// built-in optimizer (their state inits are shape-only zeros/ones).
        /// </summary>
        private HashSet<int> _stateInitConsumedHyperIndices = new();

        /// <summary>Runtime-hyper optimizer-order index → its <see cref="HyperparameterStructDef"/> field name.</summary>
        private Dictionary<int, string> _runtimeHyperNameByOptIndex = new();

        /// <summary>
        /// True when the optimizer's state-init graph reads a <see cref="HyperparameterKind.Runtime"/>
        /// hyper: its initial value is unknowable at build, so <see cref="CreateInitialCheckpoint()"/>
        /// fails loud (D5) until <see cref="CreateInitialCheckpoint(TensorDataStruct)"/> supplies it.
        /// </summary>
        private bool _stateInitNeedsRuntimeHypers;

        /// <summary>The names of the runtime hyperparameters the state-init graph consumes (for the D5 error).</summary>
        private string[] _stateInitConsumedRuntimeHyperNames = Array.Empty<string>();

        /// <summary>Default values for the dynamic hyperparameter fields (their initial values from
        /// FromScratch), used to seed shape inference / optimization. Empty when no hyperparameter is dynamic.</summary>
        private Dictionary<string, IData> _initialHyperparamFields = null!;

        /// <summary>
        /// Creates a TrainingRig from scratch by composing the model, loss, and optimizer
        /// computation graphs into a single TrainingStepPureGraph. Sample inputs are required:
        /// they drive trainable-parameter initialization (for models whose param shapes depend
        /// on input shapes), input-aware pruning of trainable params whose reachability is
        /// killed by the sample input shape (e.g. inside a folded-out IfElse branch), and
        /// shape inference of the lowered training-step graph.
        /// </summary>
        /// <param name="modelGraph">The model's InternalComputationGraph (typically a source-generated module's static graph property)</param>
        /// <param name="lossGraph">The loss function's computation graph (2 inputs: predictions, targets; 1 output: loss)</param>
        /// <param name="optimizerGraph">The optimizer's computation graph (inputs: hyperparams + param + grad; outputs: updated_param). Optimizer state is created inside the module body via optimizer-owned [StateInitializer] Init calls and updated via Globals.StateUpdate — never declared in the signature.</param>
        /// <param name="sampleInputs">Sample model inputs (one per model graph input) used to resolve parameter shapes and seed shape inference. Only the shapes matter, not the values.</param>
        /// <param name="hyperparameters">
        /// The optimizer's named hyperparameters — typically the source-generated set, e.g.
        /// <c>new AdamWOptimizerHyperparameters { LearningRate = Schedules.Cosine(3e-4f, total), WeightDecay = 1e-4f }</c>.
        /// Each value's kind decides its wiring: a bare <see cref="float"/> is baked as a constant; a
        /// <see cref="Schedule"/> is applied per step; <see cref="Hyperparameter.Runtime"/> is supplied manually.
        /// </param>
        /// <param name="rngConfig">
        /// Optional RNG configuration. Trainable parameters initialize from per-parameter keyed
        /// streams and the config is bound to the training-step graph (keying every runtime
        /// random feed, e.g. Dropout masks), making the whole run's randomness deterministic
        /// and reproducible from the config's master seed. When <c>null</c>,
        /// <see cref="RngConfig.Default"/> (master seed 0) is used — "no config" means the
        /// default deterministic identity, never non-reproducible backend randomness.
        /// </param>
        /// <returns>A configured TrainingRig ready for training</returns>
        public static TrainingRig FromScratch(
            ComputationGraph modelGraph,
            ComputationGraph lossGraph,
            ComputationGraph optimizerGraph,
            NamedModelParam[] sampleInputs,
            IOptimizerHyperparameters hyperparameters,
            RngConfig? rngConfig = null)
        {
            if (hyperparameters is null) throw new ArgumentNullException(nameof(hyperparameters));
            return FromScratchCore(modelGraph, lossGraph, optimizerGraph, sampleInputs,
                hyperparameters.InOptimizerOrder(), hyperparameters.HyperparameterNames, rngConfig);
        }

        /// <summary>
        /// Lower-level overload that takes the hyperparameter values positionally (in the optimizer's
        /// declared order) rather than as a named set. Each <see cref="Hyperparameter"/>'s kind still
        /// decides baked-vs-runtime; a bare <c>float</c> implicitly converts to a baked constant, so
        /// <c>FromScratch(model, loss, opt, sample, 0.01f)</c> bakes a single learning rate. Generated
        /// graph fields fall back to <c>hyperparam_{i}</c> names since no names are supplied.
        /// </summary>
        public static TrainingRig FromScratch(
            ComputationGraph modelGraph,
            ComputationGraph lossGraph,
            ComputationGraph optimizerGraph,
            NamedModelParam[] sampleInputs,
            params Hyperparameter[] hyperparameters)
            => FromScratchCore(modelGraph, lossGraph, optimizerGraph, sampleInputs, hyperparameters,
                names: null, rngConfig: null);

        /// <summary>
        /// Positional-hyperparameter overload with an RNG configuration. The config precedes
        /// the hyperparameter values because a <c>params</c> array must come last.
        /// </summary>
        public static TrainingRig FromScratch(
            ComputationGraph modelGraph,
            ComputationGraph lossGraph,
            ComputationGraph optimizerGraph,
            NamedModelParam[] sampleInputs,
            RngConfig? rngConfig,
            params Hyperparameter[] hyperparameters)
            => FromScratchCore(modelGraph, lossGraph, optimizerGraph, sampleInputs, hyperparameters,
                names: null, rngConfig: rngConfig);

        /// <summary>
        /// Convenience overload that accepts a <see cref="ModelParamList"/> for sample inputs,
        /// as returned by <c>model.FromOrderedInputs([…])</c>, so you can write
        /// <c>FromScratch(model, Losses.L2Loss, Optimizers.Adam, model.FromOrderedInputs([…]), hypers)</c>
        /// without constructing <see cref="TensorDataModelParam"/> objects by hand.
        /// </summary>
        public static TrainingRig FromScratch(
            ComputationGraph modelGraph,
            ComputationGraph lossGraph,
            ComputationGraph optimizerGraph,
            ModelParamList sampleInputs,
            IOptimizerHyperparameters hyperparameters,
            RngConfig? rngConfig = null)
        {
            if (sampleInputs is null) throw new ArgumentNullException(nameof(sampleInputs));
            return FromScratch(modelGraph, lossGraph, optimizerGraph,
                sampleInputs.ModelParams.ToArray(), hyperparameters, rngConfig);
        }

        /// <summary>
        /// Convenience overload that accepts a <see cref="ModelParamList"/> for sample inputs
        /// with positional hyperparameter values.
        /// </summary>
        public static TrainingRig FromScratch(
            ComputationGraph modelGraph,
            ComputationGraph lossGraph,
            ComputationGraph optimizerGraph,
            ModelParamList sampleInputs,
            params Hyperparameter[] hyperparameters)
        {
            if (sampleInputs is null) throw new ArgumentNullException(nameof(sampleInputs));
            return FromScratch(modelGraph, lossGraph, optimizerGraph,
                sampleInputs.ModelParams.ToArray(), hyperparameters);
        }

        /// <summary>
        /// <see cref="ModelParamList"/> convenience overload with an RNG configuration and
        /// positional hyperparameter values (the config precedes the <c>params</c> array).
        /// </summary>
        public static TrainingRig FromScratch(
            ComputationGraph modelGraph,
            ComputationGraph lossGraph,
            ComputationGraph optimizerGraph,
            ModelParamList sampleInputs,
            RngConfig? rngConfig,
            params Hyperparameter[] hyperparameters)
        {
            if (sampleInputs is null) throw new ArgumentNullException(nameof(sampleInputs));
            return FromScratch(modelGraph, lossGraph, optimizerGraph,
                sampleInputs.ModelParams.ToArray(), rngConfig, hyperparameters);
        }

        private static TrainingRig FromScratchCore(
            ComputationGraph modelGraph,
            ComputationGraph lossGraph,
            ComputationGraph optimizerGraph,
            NamedModelParam[] sampleInputs,
            Hyperparameter[] hyperparameters,
            IReadOnlyList<string>? names,
            RngConfig? rngConfig)
        {
            if (modelGraph is null) throw new ArgumentNullException(nameof(modelGraph));
            if (lossGraph is null) throw new ArgumentNullException(nameof(lossGraph));
            if (optimizerGraph is null) throw new ArgumentNullException(nameof(optimizerGraph));
            if (sampleInputs is null) throw new ArgumentNullException(nameof(sampleInputs));
            if (hyperparameters is null) throw new ArgumentNullException(nameof(hyperparameters));

            // Capture the constituents (the swappable source-of-truth layer, §5.8) and take the
            // initial build path, which concretizes the model from the sample inputs. The sample
            // inputs are a construction-time argument only — consumed here to produce the retained
            // concrete arch and its shape exemplars, and never stored on the rig. "No config" means
            // the default deterministic identity.
            return BuildInitialRig(
                new RigConstituents(
                    modelGraph, lossGraph, optimizerGraph, hyperparameters, names,
                    rngConfig ?? RngConfig.Default),
                sampleInputs);
        }

        /// <summary>
        /// The model-graph precondition shared by <see cref="FromScratch(ComputationGraph,
        /// ComputationGraph, ComputationGraph, NamedModelParam[], IOptimizerHyperparameters, RngConfig?)"/>
        /// and <see cref="TrainingCheckpoint.ToInferenceModel"/>: a module graph or an
        /// already-lowered concrete architecture (both feed the idempotent
        /// <c>ToConcreteArchitecture</c> pipeline). A weight-filled concrete model is
        /// refused — its parameters are already materialized as values, so there is
        /// nothing left to discover or initialize.
        /// </summary>
        internal static InternalComputationGraph RequireModelGraphKind(ComputationGraph modelGraph, string operation)
        {
            if (modelGraph.Kind is GraphKind.Module or GraphKind.ConcreteArchitecture)
                return modelGraph.ToInternal();
            throw new InvalidOperationException(Shorokoo.Core.Utils.SrkFileFormat.KindMismatchMessage(
                operation, "a 'module' or 'concrete-architecture' model graph", modelGraph.Kind,
                "Its parameters are already materialized as values; pass the module graph " +
                "(e.g. MyModel.ComputationGraph) or its ToConcreteArchitecture result instead."));
        }

        /// <summary>
        /// Shared shape check for both derive paths: hyperparameter names, when supplied, must match
        /// the hyperparameter value count (a swapped optimizer/scheduler is checked just as at build).
        /// </summary>
        private static void ValidateConstituents(RigConstituents c)
        {
            if (c.Names is not null && c.Names.Count != c.Hyperparameters.Length)
                throw new ArgumentException(
                    $"hyperparameter names ({c.Names.Count}) must match hyperparameter values " +
                    $"({c.Hyperparameters.Length}).", nameof(c));
        }

        /// <summary>
        /// The <b>initial build path</b> (<see cref="FromScratch(ComputationGraph, ComputationGraph, ComputationGraph, NamedModelParam[], IOptimizerHyperparameters, RngConfig?)"/>
        /// only): concretizes the model from the <paramref name="sampleInputs"/> once, binds the RNG
        /// config, and derives the retained concrete arch + its shape exemplars — then hands off to
        /// <see cref="DeriveFromConcreteArch"/> to compose and optimize the trainstep. The sample
        /// inputs are consumed here (parameter shape resolution, liveness pruning, concretization value
        /// fallbacks) and NOT stored: everything the derivation path later needs is captured on the
        /// concrete arch itself (structure + RNG identity + the representative-input attributes written
        /// onto its <c>MODEL_TENSOR_INPUT</c> nodes below).
        /// </summary>
        private static TrainingRig BuildInitialRig(RigConstituents constituents, NamedModelParam[] sampleInputs)
        {
            var c = constituents;
            ValidateConstituents(c);
            if (sampleInputs.Length == 0)
                throw new ArgumentException(
                    "A training rig requires at least one sample input. Sample inputs " +
                    "drive parameter shape resolution and training-graph shape inference.",
                    nameof(sampleInputs));

            // The model position takes a module graph or an already-lowered concrete architecture
            // (the concretization pipeline is idempotent on the latter); a weight-filled concrete
            // model is refused up front.
            var model = RequireModelGraphKind(c.Model, "TrainingRig (model constituent)");

            var ctx = ComputeContext.Default;

            // Single ToConcreteArchitecture pass — the ONE concretization for this rig and all its
            // future derivations. The resulting concrete arch is the shared substrate: the trainstep
            // build composes it with loss + autograd + optimizer; initialization reads its MODEL_PARAM
            // nodes for initial values and prediction shape; inference extraction binds weights into
            // it. The pass also runs the QEE-backed liveness filter that prunes trainable params whose
            // reachability is killed by the sample input shape. Sample input VALUES matter only here
            // (concretization's QEE/ORT resolution fallbacks); the derivation path needs only shapes.
            var concreteArch = model.ToConcreteArchitecture(new ModelParamList(sampleInputs), ctx);

            // Bind the RNG config at the shared concretization point: binding writes the
            // config's runtime identity into the RngSeed parameter, which — with the feeds'
            // key derivation chains — rides unchanged through loss composition and autograd
            // into the training-step graph, where the ONNX-prep lowering emits the keyed draws.
            concreteArch.ApplyRngConfig(c.RngConfig);

            // Make the concrete arch self-describing: record a representative input on each model-input
            // node (zero-filled, shape+dtype only — never the user's values), so every re-derivation's
            // shape inference reconstructs its sampleInputs off the arch and no separate exemplar field
            // is stored. Done once here; the attributes ride along on Clone() and survive re-seeding.
            WriteRepresentativeInputs(concreteArch, sampleInputs);

            return DeriveFromConcreteArch(c, concreteArch);
        }

        /// <summary>
        /// The <b>derivation path</b> — shared by the initial build (once its concrete arch exists) and
        /// every <c>With…</c> derivation. Reuses the already-<paramref name="concreteArch"/> (NO
        /// <c>ToConcreteArchitecture</c>, NO sample inputs — the model constituent never changes under a
        /// derivation; the shape metadata shape inference needs is read off the arch's own
        /// representative-input attributes), re-validates the swappable loss / optimizer constituent
        /// kinds, composes the in-memory <c>trainstep</c>, and initializes / optimizes it. The receiver
        /// is never mutated: the concrete arch is shared by reference (its consumers clone before
        /// mutating); only the trainstep is derived anew.
        /// </summary>
        private static TrainingRig DeriveFromConcreteArch(
            RigConstituents constituents,
            InternalComputationGraph concreteArch)
        {
            var c = constituents;
            ValidateConstituents(c);

            // Loss and optimizer are composed as module bodies and must be module graphs; re-validated
            // on every derivation so a swapped constituent is checked. The model is not re-checked — it
            // is already the concrete arch and never changes on a derivation.
            c.Loss.RequireKind(GraphKind.Module, "TrainingRig (loss constituent)",
                "Pass the loss module graph (e.g. Losses.L2Loss).");
            c.Optimizer.RequireKind(GraphKind.Module, "TrainingRig (optimizer constituent)",
                "Pass the optimizer module graph (e.g. Optimizers.Adam).");

            var ctx = ComputeContext.Default;

            var rig = new TrainingRig
            {
                _constituents = c,
                _concreteArch = concreteArch,
            };
            rig.BuildTrainingStepPureGraph(
                concreteArch, c.Loss.ToInternal(), c.Optimizer.ToInternal(), c.Hyperparameters, c.Names);
            rig.InitializeAndOptimize(concreteArch, ctx, c.RngConfig);
            return rig;
        }

        /// <summary>
        /// Writes a representative input onto each of the concrete arch's <c>MODEL_TENSOR_INPUT</c> nodes
        /// (in graph-input order, one per <paramref name="sampleInputs"/>), making the arch self-describing
        /// for training-graph shape inference. The value is <b>zero-filled</b> (never the user's sample
        /// values) and <b>shape-limited</b> by the shape-inference engine's small-tensor threshold: inputs
        /// with at most
        /// <see cref="Shorokoo.Core.AutoDiffCheckpointing.ShapeInferenceInterpreter.MaxSmallTensorElements"/>
        /// elements get a real zero payload; larger ones get a shape+dtype-only placeholder (no payload),
        /// which is all the engine keeps for a tensor over that threshold anyway. Only concretization
        /// (already done) needed input values; from here on only shapes matter, so this records exactly
        /// the shape metadata.
        /// </summary>
        private static void WriteRepresentativeInputs(InternalComputationGraph concreteArch, NamedModelParam[] sampleInputs)
        {
            if (concreteArch.Inputs.Count != sampleInputs.Length)
                throw new InvalidOperationException(
                    $"Concrete arch has {concreteArch.Inputs.Count} input(s) but {sampleInputs.Length} " +
                    "sample input(s) were supplied; they must correspond one-to-one in declaration order.");
            var producerByOutput = BuildProducerByOutputMap(concreteArch);
            for (int i = 0; i < concreteArch.Inputs.Count; i++)
            {
                if (!producerByOutput.TryGetValue(concreteArch.Inputs[i], out var node))
                    throw new InvalidOperationException(
                        $"Concrete arch input {concreteArch.Inputs[i]} has no producing node.");
                if (node.OpCode != InternalOpCodes.MODEL_TENSOR_INPUT) continue;
                var sample = sampleInputs[i].ToTensorData();
                node.Attributes = node.Attributes.SetAttributes(
                    (OnnxOpAttributeNames.ShrkAttrRepresentativeInput,
                     (object?)RepresentativeInputFor(sample.Shape, sample.DType)));
            }
        }

        /// <summary>
        /// Reconstructs the <c>sampleInputs[]</c> array for shape inference off the concrete arch's
        /// representative-input attributes (in graph-input order) — the derivation-path counterpart of
        /// <see cref="WriteRepresentativeInputs"/>. Each stored tensor is fed straight to
        /// <see cref="ShapeInferenceInterpreter"/>: a small one carries real zeros; a large one is a
        /// shape+dtype-only placeholder that QEE reads shape-only (it never touches the tensor's memory
        /// for a tensor above its threshold), so no large buffer is ever materialized.
        /// </summary>
        private static TensorData[] ReadRepresentativeInputs(InternalComputationGraph concreteArch)
        {
            var producerByOutput = BuildProducerByOutputMap(concreteArch);
            var inputs = new TensorData[concreteArch.Inputs.Count];
            for (int i = 0; i < concreteArch.Inputs.Count; i++)
            {
                if (!producerByOutput.TryGetValue(concreteArch.Inputs[i], out var node)
                    || node.OpCode != InternalOpCodes.MODEL_TENSOR_INPUT)
                    throw new InvalidOperationException(
                        $"Concrete arch input {concreteArch.Inputs[i]} is not a MODEL_TENSOR_INPUT node; " +
                        "cannot read its representative input.");
                inputs[i] = node.Attributes.GetTensorVal(OnnxOpAttributeNames.ShrkAttrRepresentativeInput)
                    ?? throw new InvalidOperationException(
                        "Concrete arch input node carries no representative-input attribute; the rig was " +
                        "not built through BuildInitialRig (which records it on every model input).");
            }
            return inputs;
        }

        /// <summary>
        /// A zero-filled representative tensor for shape inference: a real zero payload when the element
        /// count is within the consuming shape-inference engine's small-tensor threshold
        /// (<see cref="Shorokoo.Core.AutoDiffCheckpointing.ShapeInferenceInterpreter.MaxSmallTensorElements"/>),
        /// else a shape+dtype-only placeholder (no allocation). Holds no sample-input values.
        ///
        /// <para>The threshold MUST match the one <see cref="ShapeInferenceInterpreter"/> hands its
        /// <see cref="Shorokoo.Core.Inference.QuickExecutionEngine"/> (<c>MaxSmallTensorElements</c>),
        /// not QEE's own <c>DefaultMaxDataElements</c>: a placeholder is legal input to QEE only when
        /// its element count is <b>strictly above</b> the threshold QEE reads payloads at — otherwise
        /// <see cref="Shorokoo.Core.Inference.Helpers.TensorDataConverter.ToRuntimeTensor"/> tries to read
        /// the placeholder's elided memory and throws, defeating the whole QEE shape-inference pass. Below
        /// the threshold we must therefore carry a real (zero) payload, exactly as the retired
        /// <c>ZeroExemplar</c> did for every size.</para>
        /// </summary>
        private static TensorData RepresentativeInputFor(Shape shape, DType dtype)
        {
            if (shape.Count > Shorokoo.Core.AutoDiffCheckpointing.ShapeInferenceInterpreter.MaxSmallTensorElements)
                return new WeightPlaceholderTensorData(shape, dtype);
            var bytesPerElement = dtype.EncodingBitCount / 8;
            return TensorData.CreateFromRawBytes(shape, dtype, new byte[shape.Count * bytesPerElement]);
        }

        // ───────────────────── Two-layer rig: immutable derivations (§5.8.5) ─────────────────────
        // A TrainingRig is an immutable value (consistent with the frozen ComputationGraph). None of
        // the operations below mutate the receiver; each returns a NEW rig that shares the unchanged
        // constituents (and their graphs) BY REFERENCE — via `record with` on RigConstituents — and
        // re-derives only its own trainstep from the swapped constituent. Deriving is therefore free
        // of aliasing surprises: the original rig is untouched.

        /// <summary>
        /// A new rig with the loss constituent replaced; its <c>trainstep</c> is re-derived, and the
        /// model (its retained concrete arch), optimizer, hyperparameters and RNG config are shared by
        /// reference.
        /// </summary>
        public TrainingRig WithLoss(ComputationGraph loss)
        {
            if (loss is null) throw new ArgumentNullException(nameof(loss));
            return DeriveFromConcreteArch(_constituents with { Loss = loss }, _concreteArch);
        }

        /// <summary>
        /// A new rig with the optimizer constituent (and its hyperparameters) replaced; optimizer
        /// state is re-initialized as part of re-deriving the <c>trainstep</c>, everything else shared.
        /// </summary>
        public TrainingRig WithOptimizer(ComputationGraph optimizer, IOptimizerHyperparameters hyperparameters)
        {
            if (optimizer is null) throw new ArgumentNullException(nameof(optimizer));
            if (hyperparameters is null) throw new ArgumentNullException(nameof(hyperparameters));
            return DeriveFromConcreteArch(_constituents with
            {
                Optimizer = optimizer,
                Hyperparameters = hyperparameters.InOptimizerOrder(),
                Names = hyperparameters.HyperparameterNames,
            }, _concreteArch);
        }

        /// <summary>Positional-hyperparameter overload of <see cref="WithOptimizer(ComputationGraph, IOptimizerHyperparameters)"/>.</summary>
        public TrainingRig WithOptimizer(ComputationGraph optimizer, params Hyperparameter[] hyperparameters)
        {
            if (optimizer is null) throw new ArgumentNullException(nameof(optimizer));
            if (hyperparameters is null) throw new ArgumentNullException(nameof(hyperparameters));
            return DeriveFromConcreteArch(
                _constituents with { Optimizer = optimizer, Hyperparameters = hyperparameters, Names = null },
                _concreteArch);
        }

        /// <summary>
        /// A new rig with the scheduler swapped, keeping the optimizer graph — the schedule rides in
        /// the hyperparameters (a baked constant, a built-in <see cref="Schedule"/>, or a scheduler
        /// module per field) until #106 folds the scheduler into its own persisted constituent. Only
        /// the <c>trainstep</c> is re-derived; the model, loss and optimizer graphs are shared.
        /// </summary>
        public TrainingRig WithScheduler(IOptimizerHyperparameters hyperparameters)
        {
            if (hyperparameters is null) throw new ArgumentNullException(nameof(hyperparameters));
            return DeriveFromConcreteArch(_constituents with
            {
                Hyperparameters = hyperparameters.InOptimizerOrder(),
                Names = hyperparameters.HyperparameterNames,
            }, _concreteArch);
        }

        /// <summary>Positional-hyperparameter overload of <see cref="WithScheduler(IOptimizerHyperparameters)"/>.</summary>
        public TrainingRig WithScheduler(params Hyperparameter[] hyperparameters)
        {
            if (hyperparameters is null) throw new ArgumentNullException(nameof(hyperparameters));
            return DeriveFromConcreteArch(
                _constituents with { Hyperparameters = hyperparameters, Names = null },
                _concreteArch);
        }

        /// <summary>
        /// A new rig re-seeded with <paramref name="rngConfig"/> — the model's (and other drawing
        /// constituents') RNG identity is re-initialized, everything else shared by reference. Unlike
        /// the other derivations, WithSeed does change the concrete arch: the RNG identity is bound into
        /// its RngSeed parameter, so re-seeding rebinds it. That binding mutates, so this rig's retained
        /// arch is left untouched — a <b>clone</b> is re-keyed and the new rig derives from that. It is
        /// still only a rebind, not a re-concretization: the model constituent's structure is unchanged,
        /// so no <c>ToConcreteArchitecture</c> and no sample inputs are needed; re-initialization then
        /// re-draws every trainable parameter on the new seed's keyed streams (§2.5). The design's
        /// cheaper path (share even the trainstep, re-derive only the compiled session, since the seed
        /// rides as an aliased param value) rests on the #22 param-identity substrate; until that lands
        /// the re-seed re-derives the trainstep, which is correct and equally immutable.
        /// </summary>
        public TrainingRig WithSeed(RngConfig rngConfig)
        {
            if (rngConfig is null) throw new ArgumentNullException(nameof(rngConfig));
            // Rebind the new RNG identity on a clone (ApplyRngConfig mutates), keeping this rig's
            // retained arch pristine. Clone() copies node attributes by reference, so the clone's
            // MODEL_TENSOR_INPUT nodes still carry the representative-input attributes (same model inputs).
            var reArch = _concreteArch.Clone();
            reArch.ApplyRngConfig(rngConfig);
            return DeriveFromConcreteArch(_constituents with { RngConfig = rngConfig }, reArch);
        }

        /// <summary>
        /// Extracts the inference model for a checkpoint — a <b>pure read off the model constituent's
        /// mapping</b> (§5.8.2): bind the checkpoint's model-owned params (trainable weights + module
        /// state) by their canonical identifiers into the rig's retained concrete arch. No
        /// re-concretization and no sample inputs — the arch was concretized once at build (at all
        /// inputs, so a multi-input model extracts correctly) and is reused. No copy step, no
        /// which-tensors-belong-to-inference heuristic; because model-owned parameter identity is
        /// preserved across composition, the tensors the <c>trainstep</c> updated bind straight back
        /// into the inference model by the same name.
        /// </summary>
        public ComputationGraph ExtractInferenceModel(TrainingCheckpoint checkpoint)
            => new(BindInferenceWeights(checkpoint), GraphKind.ConcreteModel);

        /// <summary>
        /// The single weight-binding step shared by <see cref="ExtractInferenceModel"/> (and thus
        /// <see cref="TrainingCheckpoint.ToInferenceModel()"/>) and the <c>.skpt</c> container's
        /// self-describing model, so the two can never disagree on how a checkpoint concretizes: bind
        /// the checkpoint's model-owned params — its trainable weights AND its module-owned state
        /// (BatchNorm running stats, …), each by its canonical identifier — into the rig's retained
        /// concrete arch. The arch is consumed read-only (<c>ToConcreteModel</c> clones before
        /// binding), so this never disturbs the retained graph. For a stateless model
        /// <see cref="TrainingCheckpoint.ModelState"/> is empty, so this is a trainable-only bind.
        /// </summary>
        internal InternalComputationGraph BindInferenceWeights(TrainingCheckpoint checkpoint)
        {
            if (checkpoint is null) throw new ArgumentNullException(nameof(checkpoint));
            var weights = new ModelParamList(
                checkpoint.TrainableParams.Fields
                    .Where(f => f.Value is TensorData)
                    .Concat(checkpoint.ModelState.Fields.Where(f => f.Value is TensorData))
                    .Select(f => new KeyValuePair<string, TensorData>(f.Key, (TensorData)f.Value)),
                ModelParamType.TrainableParam);
            return _concreteArch.ToConcreteModel(weights, _concreteArch.GetShorokooIdNamingScheme());
        }

        /// <summary>
        /// Returns a NEW checkpoint identical to <paramref name="checkpoint"/> — same trainable params,
        /// model state, optimizer state, counters and loss — but with its <see cref="TrainingCheckpoint.Rig"/>
        /// set to this rig, so <see cref="TrainingCheckpoint.ToInferenceModel()"/> and rig-based load/save
        /// work against it. Validates that the checkpoint's field definitions are compatible with this rig
        /// (trainable-param, model-state and optimizer-state field names and shapes must match); throws a
        /// clear <see cref="ArgumentException"/> otherwise. The argument is not mutated.
        /// </summary>
        public TrainingCheckpoint AdoptCheckpoint(TrainingCheckpoint checkpoint)
        {
            if (checkpoint is null) throw new ArgumentNullException(nameof(checkpoint));
            AssertStructDefCompatible(checkpoint.TrainableParams.Definition, TrainableParamStructDef, "trainable-parameter");
            AssertStructDefCompatible(checkpoint.ModelState.Definition, ModelStateDef, "model-state");
            AssertStructDefCompatible(checkpoint.OptimizerState.Definition, OptimizerStateDef, "optimizer-state");
            return new TrainingCheckpoint(
                checkpoint.TrainableParams, checkpoint.ModelState, checkpoint.OptimizerState,
                checkpoint.Step, checkpoint.Epoch, checkpoint.BatchIndex, this, checkpoint.Loss);
        }

        /// <summary>Fails loud when a checkpoint's struct def does not match this rig's, by field
        /// names and per-field rank/dtype/structure (the compatibility contract for adoption/load).</summary>
        private static void AssertStructDefCompatible(TensorStructDef actual, TensorStructDef expected, string kind)
        {
            if (actual.Fields.Length != expected.Fields.Length)
                throw new ArgumentException(
                    $"Checkpoint's {kind} definition has {actual.Fields.Length} field(s), but this rig " +
                    $"expects {expected.Fields.Length}. The checkpoint was produced by a different model/optimizer.");
            for (int i = 0; i < expected.Fields.Length; i++)
            {
                var e = expected.Fields[i];
                var a = actual.GetField(e.Name)
                    ?? throw new ArgumentException(
                        $"Checkpoint's {kind} definition is missing field '{e.Name}' this rig expects. " +
                        "The checkpoint was produced by a different model/optimizer.");
                if (a.Rank != e.Rank || a.ElementType != e.ElementType || a.Structure != e.Structure)
                    throw new ArgumentException(
                        $"Checkpoint's {kind} field '{e.Name}' (rank {a.Rank?.ToString() ?? "?"}, {a.ElementType}) " +
                        $"does not match this rig's (rank {e.Rank?.ToString() ?? "?"}, {e.ElementType}). " +
                        "The checkpoint was produced by a different model/optimizer.");
            }
        }

        /// <summary>The rig's initial trainable-parameter values, as a struct (for load-time defaults).</summary>
        internal TensorDataStruct InitialTrainableStruct => new(TrainableParamStructDef, _initialParamFields);

        /// <summary>The rig's initial model-state values, as a struct (for load-time defaults).</summary>
        internal TensorDataStruct InitialModelStateStruct => new(ModelStateDef, _initialStateFields);

        /// <summary>The rig's initial optimizer-state values, as a struct (for load-time defaults).</summary>
        internal TensorDataStruct InitialOptimizerStateStruct => new(OptimizerStateDef, _initialOptStateFields);

        /// <summary>
        /// Builds the TrainingStepPureGraph by composing model + loss + autograd + optimizer.
        /// 
        /// Pipeline:
        /// 1. Use TrainingGraphBuilder to compose model + loss + autograd
        /// 2. Extract param struct, gradient struct, and state struct from the composed graph
        /// 3. For each param field, replay the optimizer graph to compute updated params
        /// 4. Build the complete training step graph
        /// 5. Lower to an executable graph (expand structs, process autograd, simplify)
        /// </summary>
        private void BuildTrainingStepPureGraph(
            InternalComputationGraph concreteArch,
            InternalComputationGraph lossGraph,
            InternalComputationGraph optimizerGraph,
            Hyperparameter[] hyperparameters,
            IReadOnlyList<string>? hyperparamNames)
        {
            // Normalize the optimizer graph in place. State variables created by the optimizer's
            // [StateInitializer] Init calls are rewritten into explicit graph inputs appended
            // after grad, and the StateUpdate-pattern nodes (STATE_UPDATE_LINK + WITH_STATE_DEPS)
            // into the explicit multi-output convention [updated_param, updated_state_0, ...]
            // expected by the replay loop. The split-off state-init graph computes the initial
            // state values (per trainable parameter) in InitializeAndOptimize.
            // FromScratchInternal hands in an owned thawed copy — normalize it in place.
            var optimizerFastGraph = optimizerGraph;
            var optimizerInfo = Shorokoo.Core.Nodes.Processors.Fast.FastNormalizeOptimizerGraph.Process(optimizerFastGraph);
            _optimizerStateInitGraph = optimizerInfo.StateInitGraph;

            // Value route (§2.5): the value each hyper contributes to optimizer state init, at the
            // initial counters. Baked → its constant; scheduled → its graph evaluated via QEE below;
            // runtime → null (host-supplied, D5). Filled per kind as the hypers are wired.
            _hyperparamInitialCounterValues = new float?[hyperparameters.Length];

            // Step 1: Compose model + loss + autograd via TrainingGraphBuilder. The model
            // graph is already through ToConcreteArchitecture (done once at FromScratch),
            // so the input-aware liveness filter has already pruned dead-branch trainable
            // params — FastReplaceTrainableParamsWithInputProcessor inside PrepareForTraining
            // builds the trainable param struct from only the live MODEL_PARAM nodes.
            var fastTraining = TrainingGraphBuilder.PrepareForTrainingAsFast(concreteArch, lossGraph);

            // PrepareForTraining's output layout:
            //   Inputs:  [model_inputs_struct, targets, trainable_param_struct, state_struct?]
            //   Outputs: [loss, gradient_struct, state_struct]
            var producerByOutput = BuildProducerByOutputMap(fastTraining);

            // Step 1b: Extract model-input struct def from training graph input[0].
            InputDef = ReadStructDefFromInput(fastTraining, producerByOutput, fastTraining.Inputs[0])
                ?? throw new InvalidOperationException(
                    "Input[0] of training graph is not a TensorStruct. Expected model inputs struct.");

            // Build TargetDef from the loss graph's second input (the target tensor).
            // The target is a plain tensor in the training graph (not a TensorStruct), so we
            // synthesize a single-field struct so Fit can accept TensorDataStructs uniformly.
            {
                var lossProdMap = BuildProducerByOutputMap(lossGraph);
                if (!lossProdMap.TryGetValue(lossGraph.Inputs[1], out var targetProducer))
                    throw new InvalidOperationException("Loss graph target input (index 1) has no producer node.");
                var targetDtype = targetProducer.Attributes.GetDTypeVal(OnnxOpAttributeNames.AttrDtype)
                    ?? throw new InvalidOperationException("Loss graph target input has no AttrDtype.");
                var targetRank = (int?)targetProducer.Attributes.GetLongVal(OnnxOpAttributeNames.ShrkAttrRank);
                var targetFieldName = lossGraph.InputUniqueNames.Count > 1
                    ? lossGraph.InputUniqueNames[1] ?? "targets"
                    : "targets";
                TargetDef = new TensorStructDef(
                    [new TensorStructFieldDef(targetFieldName, DataStructure.Tensor, targetRank, targetDtype)],
                    "Targets");
            }

            // Step 2: Extract param struct definition from input[2].
            var trainableParamStructInputKey = fastTraining.Inputs[2];
            TrainableParamStructDef = ReadStructDefFromInput(fastTraining, producerByOutput, trainableParamStructInputKey)
                ?? throw new InvalidOperationException(
                    "Input[2] of training graph is not a TensorStruct. Expected param struct input.");

            // Step 3: Extract state struct definition from input[3] (if present).
            FastTensorKey? stateStructInputKey = null;
            if (fastTraining.Inputs.Count > 3)
            {
                stateStructInputKey = fastTraining.Inputs[3];
                ModelStateDef = ReadStructDefFromInput(fastTraining, producerByOutput, stateStructInputKey.Value)
                    ?? throw new InvalidOperationException(
                        "Input[3] of training graph is not a TensorStruct. Expected state struct input.");
            }
            else
            {
                ModelStateDef = new TensorStructDef(Array.Empty<TensorStructFieldDef>(), "ModelState");
            }

            // Step 4: Extract gradient struct definition from the second output (a
            // TENSOR_STRUCT_CREATE node; AttrDtype carries the struct dtype).
            var gradStructOutputKey = fastTraining.Outputs[1];
            var gradStructDef = ReadStructDefFromProducer(producerByOutput[gradStructOutputKey])
                ?? throw new InvalidOperationException(
                    "Second output of training graph is not a TensorStruct. Expected gradient struct output.");

            // Track input-style nodes so we can move them to the front of
            // fastTraining.Nodes at the end (in creation order). Param-field
            // GETFIELDs, hyperparam CONSTANTs, optimizer-state INPUT and its
            // GETFIELDs are all body-independent and belong before the body in
            // topological order. Grad-field GETFIELDs depend on a body-produced
            // tensor and stay where they are (after the body, before the
            // replays that consume them).
            var headNodesInOrder = new List<FastNode>();

            // Step 5: Build per-field GETFIELD nodes for params and gradients.
            var paramFieldKeys = new FastTensorKey[TrainableParamStructDef.Fields.Length];
            for (int i = 0; i < TrainableParamStructDef.Fields.Length; i++)
            {
                var f = TrainableParamStructDef.Fields[i];
                var node = Shorokoo.Core.Nodes.Processors.Fast.FastInternalOp.TensorStructGetField(
                    trainableParamStructInputKey, f.Name, f.ElementType, f.Rank, f.Structure);
                fastTraining.Nodes.Add(node);
                headNodesInOrder.Add(node);
                paramFieldKeys[i] = new FastTensorKey(node.Key, 0);
            }

            var gradFieldKeys = new FastTensorKey[gradStructDef.Fields.Length];
            for (int i = 0; i < gradStructDef.Fields.Length; i++)
            {
                var f = gradStructDef.Fields[i];
                var node = Shorokoo.Core.Nodes.Processors.Fast.FastInternalOp.TensorStructGetField(
                    gradStructOutputKey, f.Name, f.ElementType, f.Rank, f.Structure);
                fastTraining.Nodes.Add(node);
                gradFieldKeys[i] = new FastTensorKey(node.Key, 0);
            }

            // Step 6: Optimizer state structure, as discovered by FastNormalizeOptimizerGraph
            // from the optimizer's [StateInitializer] Init calls. The normalized graph follows:
            //   optimizer outputs = [updated_param, updated_state_0, ...]
            //   optimizer inputs  = [hyperparam_0, ..., param, grad, state_0, ...]
            // where the state inputs were appended by the normalization pass (the authored
            // Inline signature contains only hyperparams + param + grad).
            int numOptimizerStateFieldsPerParam = optimizerInfo.StateCount;
            int numHyperparams = optimizerInfo.HyperparamCount;

            if (hyperparameters.Length != numHyperparams)
                throw new ArgumentException(
                    $"Optimizer expects {numHyperparams} hyperparameter(s), but {hyperparameters.Length} were provided.");

            // Each hyperparameter's kind decides how it is wired into the training-step graph:
            //  • baked (a bare float)          → a graph CONSTANT (its value also seeds shape inference).
            //  • scheduled (built-in Schedule  → lowered to graph math (ScheduleLowering); scheduler
            //    or a scheduler module)          module → inlined — both computed in-graph from the
            //                                     int64 step-counter input, with no per-step host
            //                                     evaluation (#99).
            //  • schedule-less runtime         → a "hyperparams" TensorStruct runtime input (one scalar
            //    (Hyperparameter.Runtime)             float32 field each), supplied explicitly every step.
            string NameOf(int h) => hyperparamNames is not null && h < hyperparamNames.Count
                ? hyperparamNames[h] : $"hyperparam_{h}";
            float SeedOf(int h) => SeedValue(hyperparameters[h]);

            HyperparameterNames = Enumerable.Range(0, numHyperparams).Select(NameOf).ToArray();

            // Classify the dynamic hyperparameters into in-graph scheduled vs schedule-less runtime.
            var scheduledIndices = new List<int>();
            var runtimeIndices = new List<int>();
            for (int h = 0; h < numHyperparams; h++)
            {
                var hv = hyperparameters[h];
                switch (hv.Kind)
                {
                    case HyperparameterKind.Baked: continue;
                    case HyperparameterKind.Scheduled: scheduledIndices.Add(h); break;
                    case HyperparameterKind.Runtime: runtimeIndices.Add(h); break;
                }
            }

            // The "hyperparams" struct carries only the schedule-less runtime hyperparameters — the
            // ones the caller supplies via MakeHyperparameters. Scheduled hyperparameters are computed
            // in-graph and never appear here.
            DynamicHyperparameterIndices = runtimeIndices;
            var hyperFields = runtimeIndices
                .Select(h => new TensorStructFieldDef(NameOf(h), DataStructure.Tensor, 0, DType.Float32))
                .ToArray();
            HyperparameterStructDef = new TensorStructDef(hyperFields, "Hyperparameters");
            DynamicHyperparameterNames = hyperFields.Select(f => f.Name).ToArray();

            // Runtime-hyper optimizer-index → field name, for the D5 CreateInitialCheckpoint override.
            _runtimeHyperNameByOptIndex = new Dictionary<int, string>();
            for (int i = 0; i < runtimeIndices.Count; i++)
                _runtimeHyperNameByOptIndex[runtimeIndices[i]] = hyperFields[i].Name;

            // The key feeding each optimizer replay slot: a runtime GETFIELD, an in-graph scheduler
            // output, or a baked CONSTANT. Shared across all per-parameter optimizer replays.
            var hyperparamKeys = new FastTensorKey[numHyperparams];

            // --- Schedule-less runtime hyperparameters: a "hyperparams" TensorStruct input. ---
            _initialHyperparamFields = new Dictionary<string, IData>();
            FastTensorKey? hyperparamsInputKey = null;
            if (hyperFields.Length > 0)
            {
                var hyperDType = DType.GetOrCreateForTensorStruct(HyperparameterStructDef);
                var hyperInputNode = Shorokoo.Core.Nodes.Processors.Fast.FastInternalOp.TensorStructInput(
                    hyperDType, "hyperparams");
                fastTraining.Nodes.Add(hyperInputNode);
                headNodesInOrder.Add(hyperInputNode);
                hyperparamsInputKey = new FastTensorKey(hyperInputNode.Key, 0);

                for (int i = 0; i < hyperFields.Length; i++)
                {
                    var f = hyperFields[i];
                    var node = Shorokoo.Core.Nodes.Processors.Fast.FastInternalOp.TensorStructGetField(
                        hyperparamsInputKey.Value, f.Name, f.ElementType, f.Rank, f.Structure);
                    fastTraining.Nodes.Add(node);
                    headNodesInOrder.Add(node);
                    hyperparamKeys[runtimeIndices[i]] = new FastTensorKey(node.Key, 0);
                    _initialHyperparamFields[f.Name] =
                        Shorokoo.Globals.TensorData(Array.Empty<long>(), SeedOf(runtimeIndices[i]));
                }
            }

            // --- Scheduled hyperparameters: emitted in-graph from the named int64 counter inputs. ---
            // The counter inputs {step, epoch, batchIndex} are shared graph inputs; each scheduler
            // (built-in lowering or user module) consumes a named subset (D1) and is inlined against
            // exactly those inputs via FastReplay. Built-in DSL schedules are step-only (PerEpoch
            // derives its epoch in-graph from step, #39); a module declares its subset by input name.
            var counterInputsInOrder = new List<(FastTensorKey Key, string Name)>();
            if (scheduledIndices.Count > 0)
            {
                // Build every scheduler graph first, so the union of counter inputs they consume is
                // known before the shared counter input nodes are created.
                var builtByIndex = new Dictionary<int, SchedulerGraph>(scheduledIndices.Count);
                var needed = new HashSet<string>(StringComparer.Ordinal);
                foreach (var h in scheduledIndices)
                {
                    var built = BuildSchedulerModule(hyperparameters[h], NameOf(h));
                    builtByIndex[h] = built;
                    foreach (var c in built.CounterNames) needed.Add(c);
                }

                // Create one shared int64 scalar input per needed counter, in canonical order.
                var counterKeyByName = new Dictionary<string, FastTensorKey>(StringComparer.Ordinal);
                foreach (var cn in CounterInputNames)
                {
                    if (!needed.Contains(cn)) continue;
                    var counterNode = Shorokoo.Core.Nodes.Processors.Fast.FastInternalOp.RuntimeInput(
                        DType.Int64, rank: 0, cn);
                    fastTraining.Nodes.Add(counterNode);
                    headNodesInOrder.Add(counterNode);
                    var key = new FastTensorKey(counterNode.Key, 0);
                    counterKeyByName[cn] = key;
                    counterInputsInOrder.Add((key, cn));
                }
                _counterInputNames = counterInputsInOrder.Select(c => c.Name).ToArray();

                foreach (var h in scheduledIndices)
                {
                    var built = builtByIndex[h];
                    // Value route (§2.5): the scheduler graph is the single truth, so its value at the
                    // initial counters — what optimizer state init needs — comes from evaluating that
                    // very graph via QEE, not a hardcoded 0f (the old scheduler-module state-init hole).
                    _hyperparamInitialCounterValues[h] = EvaluateSchedulerAtInitialCounters(built.Graph);
                    // Map the scheduler's inputs (in its own input order) to the shared counter keys.
                    var mappedCounters = built.CounterNames.Select(c => counterKeyByName[c]).ToArray();
                    var replayed = Shorokoo.Core.Nodes.Processors.Fast.FastReplay.ReplayInto(
                        fastTraining, built.Graph, mappedCounters);
                    hyperparamKeys[h] = replayed[0];
                }
            }

            // --- Baked hyperparameters: graph CONSTANTs. ---
            for (int h = 0; h < numHyperparams; h++)
            {
                if (hyperparameters[h].Kind != HyperparameterKind.Baked) continue;
                _hyperparamInitialCounterValues[h] = hyperparameters[h].BakedValue;
                var node = Shorokoo.Core.Nodes.Processors.Fast.FastInternalOp.Constant(
                    Shorokoo.Globals.TensorData(Array.Empty<long>(), SeedOf(h)));
                fastTraining.Nodes.Add(node);
                headNodesInOrder.Add(node);
                hyperparamKeys[h] = new FastTensorKey(node.Key, 0);
            }

            // Build optimizer state struct definition. Element type comes from each state's
            // initializer; the rank falls back to the parameter's rank when the initializer's
            // output rank is dynamic (the common shape-driven case, where the state is created
            // at the parameter's shape).
            if (numOptimizerStateFieldsPerParam > 0)
            {
                var optStateFields = new List<TensorStructFieldDef>();
                for (int i = 0; i < TrainableParamStructDef.Fields.Length; i++)
                {
                    var pf = TrainableParamStructDef.Fields[i];
                    for (int s = 0; s < numOptimizerStateFieldsPerParam; s++)
                    {
                        optStateFields.Add(new TensorStructFieldDef(
                            $"{pf.Name}_opt_{s}", pf.Structure,
                            optimizerInfo.StateRanks[s] ?? pf.Rank,
                            optimizerInfo.StateDTypes[s]));
                    }
                }
                OptimizerStateDef = new TensorStructDef(optStateFields.ToArray(), "OptimizerState");
            }
            else
            {
                OptimizerStateDef = new TensorStructDef(Array.Empty<TensorStructFieldDef>(), "OptimizerState");
            }

            // Optimizer-state struct input + per-field GETFIELDs (if non-empty).
            FastTensorKey? optStateInputKey = null;
            var optStateFieldKeys = new FastTensorKey[OptimizerStateDef.Fields.Length];
            if (OptimizerStateDef.Fields.Length > 0)
            {
                var optStateDType = DType.GetOrCreateForTensorStruct(OptimizerStateDef);
                var optStateInputNode = Shorokoo.Core.Nodes.Processors.Fast.FastInternalOp.TensorStructInput(
                    optStateDType, "optimizer_state");
                fastTraining.Nodes.Add(optStateInputNode);
                headNodesInOrder.Add(optStateInputNode);
                optStateInputKey = new FastTensorKey(optStateInputNode.Key, 0);

                for (int i = 0; i < OptimizerStateDef.Fields.Length; i++)
                {
                    var f = OptimizerStateDef.Fields[i];
                    var node = Shorokoo.Core.Nodes.Processors.Fast.FastInternalOp.TensorStructGetField(
                        optStateInputKey.Value, f.Name, f.ElementType, f.Rank, f.Structure);
                    fastTraining.Nodes.Add(node);
                    headNodesInOrder.Add(node);
                    optStateFieldKeys[i] = new FastTensorKey(node.Key, 0);
                }
            }

            // Step 7: Apply optimizer per field by replaying the optimizer graph.
            var updatedParamKeys = new FastTensorKey[paramFieldKeys.Length];
            var updatedOptStateFieldKeys = new FastTensorKey[OptimizerStateDef.Fields.Length];
            for (int i = 0; i < paramFieldKeys.Length; i++)
            {
                var mappedInputs = new List<FastTensorKey>(numHyperparams + 2 + numOptimizerStateFieldsPerParam);
                mappedInputs.AddRange(hyperparamKeys);
                mappedInputs.Add(paramFieldKeys[i]);
                mappedInputs.Add(gradFieldKeys[i]);
                for (int s = 0; s < numOptimizerStateFieldsPerParam; s++)
                    mappedInputs.Add(optStateFieldKeys[i * numOptimizerStateFieldsPerParam + s]);

                var replayedOutputs = Shorokoo.Core.Nodes.Processors.Fast.FastReplay.ReplayInto(
                    fastTraining, optimizerFastGraph, mappedInputs.ToArray());

                updatedParamKeys[i] = replayedOutputs[0];
                for (int s = 0; s < numOptimizerStateFieldsPerParam; s++)
                    updatedOptStateFieldKeys[i * numOptimizerStateFieldsPerParam + s] = replayedOutputs[1 + s];
            }

            // Step 8: pack updated params into a struct.
            var paramDType = DType.GetOrCreateForTensorStruct(TrainableParamStructDef);
            var updatedParamStructNode = Shorokoo.Core.Nodes.Processors.Fast.FastInternalOp.TensorStructCreate(
                paramDType, updatedParamKeys);
            fastTraining.Nodes.Add(updatedParamStructNode);
            var updatedParamStructKey = new FastTensorKey(updatedParamStructNode.Key, 0);

            // Pack updated optimizer state into struct (if non-empty).
            FastTensorKey? updatedOptStateStructKey = null;
            if (OptimizerStateDef.Fields.Length > 0)
            {
                var optStateDType = DType.GetOrCreateForTensorStruct(OptimizerStateDef);
                var optStateOutputNode = Shorokoo.Core.Nodes.Processors.Fast.FastInternalOp.TensorStructCreate(
                    optStateDType, updatedOptStateFieldKeys);
                fastTraining.Nodes.Add(optStateOutputNode);
                updatedOptStateStructKey = new FastTensorKey(optStateOutputNode.Key, 0);
            }

            // Step 9: reorder fastTraining.Inputs and Outputs to the TrainStep convention.
            // Original order: [model_inputs_struct, targets, param_struct, state_struct?]
            // Target order:   [param_struct, state_struct?, optimizer_state_struct?, hyperparams_struct?, step_counter?, model_inputs_struct, targets]
            var modelInputsStructKey = fastTraining.Inputs[0];
            var targetsKey = fastTraining.Inputs[1];
            var modelInputsName = fastTraining.InputUniqueNames.Count > 0 ? fastTraining.InputUniqueNames[0] : null;
            var targetsName = fastTraining.InputUniqueNames.Count > 1 ? fastTraining.InputUniqueNames[1] : null;
            var paramStructName = fastTraining.InputUniqueNames.Count > 2 ? fastTraining.InputUniqueNames[2] : null;
            var stateStructName = stateStructInputKey is not null && fastTraining.InputUniqueNames.Count > 3
                ? fastTraining.InputUniqueNames[3] : null;

            var newInputs = new List<FastTensorKey>();
            var newInputNames = new List<string?>();
            newInputs.Add(trainableParamStructInputKey); newInputNames.Add(paramStructName);
            if (stateStructInputKey is FastTensorKey ssk) { newInputs.Add(ssk); newInputNames.Add(stateStructName); }
            if (optStateInputKey is FastTensorKey osk) { newInputs.Add(osk); newInputNames.Add("optimizer_state"); }
            if (hyperparamsInputKey is FastTensorKey hpk) { newInputs.Add(hpk); newInputNames.Add("hyperparams"); }
            foreach (var (key, cn) in counterInputsInOrder) { newInputs.Add(key); newInputNames.Add(cn); }
            newInputs.Add(modelInputsStructKey); newInputNames.Add(modelInputsName);
            newInputs.Add(targetsKey); newInputNames.Add(targetsName);

            // Original outputs: [loss, gradient_struct, state_struct]
            // Target outputs:   [updated_param_struct, state_struct, updated_optimizer_state?, loss]
            var lossOutputKey = fastTraining.Outputs[0];
            var stateStructOutputKey = fastTraining.Outputs[2];

            var newOutputs = new List<FastTensorKey>();
            newOutputs.Add(updatedParamStructKey);
            newOutputs.Add(stateStructOutputKey);
            if (updatedOptStateStructKey is FastTensorKey uosk) newOutputs.Add(uosk);
            newOutputs.Add(lossOutputKey);

            fastTraining.Inputs = newInputs;
            fastTraining.InputUniqueNames = newInputNames;
            fastTraining.Outputs = newOutputs;
            fastTraining.OutputUniqueNames = new List<string?>(new string?[newOutputs.Count]);
            fastTraining.OutputRankOverrides = null;

            Shorokoo.Core.Nodes.Processors.Fast.FastProcessorHelper.RemoveUnreachableNodes(fastTraining);

            // Move tracked head nodes (param-field GETFIELDs, hyperparam CONSTANTs,
            // optimizer-state INPUT and GETFIELDs) to the front in creation order.
            // They have no body dependencies and the body is already nested by
            // construction, so no Kahn re-sort is needed.
            var headKeys = new HashSet<FastNodeKey>(headNodesInOrder.Select(n => n.Key));
            var rebuiltTraining = new List<FastNode>(fastTraining.Nodes.Count);
            rebuiltTraining.AddRange(headNodesInOrder);
            foreach (var n in fastTraining.Nodes)
                if (!headKeys.Contains(n.Key)) rebuiltTraining.Add(n);
            fastTraining.Nodes = rebuiltTraining;
            System.Diagnostics.Debug.Assert(fastTraining.IsLinearOrderValid(), "fastTraining.IsLinearOrderValid()");

            // Step 10: lower to an executable form. LowerGraph runs its Fast pipeline
            // in place on fastTraining and returns the same graph for the public-facing
            // TrainingStepPureGraph property.
            _trainingStepWorkGraph = LowerGraph(fastTraining);

            UpdatedParamFieldCount = TrainableParamStructDef.Fields.Length;
            UpdatedStateFieldCount = ModelStateDef.Fields.Length;
            UpdatedOptimizerStateFieldCount = OptimizerStateDef.Fields.Length;
        }

        /// <summary>
        /// The scalar value used to seed shape inference (and, for a baked hyper, its graph constant):
        /// a baked hyper's constant, a built-in schedule's step-0 value, else 0 (a scheduler module —
        /// whose value comes from evaluating its graph, see <see cref="EvaluateSchedulerAtInitialCounters"/>
        /// — or a seedless runtime hyper).
        /// </summary>
        private static float SeedValue(Hyperparameter h) => h.Kind switch
        {
            HyperparameterKind.Baked => h.BakedValue,
            HyperparameterKind.Scheduled => h.AsSchedule is Schedule s && s.CanLower() ? s.At(0) : 0f,
            _ => 0f,
        };

        /// <summary>
        /// Evaluates a scheduler graph (built-in lowering or user module) at the <b>initial counters</b>
        /// — every counter input bound to 0 — via the pure managed <see cref="Shorokoo.Core.Inference.QuickExecutionEngine"/>,
        /// returning the float32 value. This is the single value route (§2.5) for optimizer state init:
        /// the scheduler graph is normative, so its build-time value comes from evaluating it, not from
        /// a host closure or a hardcoded placeholder. The graph is pure (enforced, D4), so all-zero
        /// counters fully determine the value.
        /// </summary>
        private static float EvaluateSchedulerAtInitialCounters(InternalComputationGraph schedulerGraph)
        {
            var inputs = new IData[schedulerGraph.Inputs.Count];
            for (int i = 0; i < inputs.Length; i++)
                inputs[i] = Shorokoo.Globals.TensorData(Array.Empty<long>(), 0L);
            var result = new Shorokoo.Core.Inference.QuickExecutionEngine().Execute(schedulerGraph, inputs);
            return ((TensorData)result[0]).As<float32>().AccessMemory<float>()[0];
        }

        /// <summary>
        /// Builds the graph a scheduled hyperparameter is emitted from: a module taking the int64
        /// scalar counter input(s) and producing the float32 scalar scheduled value. A built-in
        /// <see cref="Schedule"/> is lowered via <see cref="ScheduleLowering"/>; a user scheduler
        /// module is validated, purity-checked, and inlined. The returned graph is spliced into the
        /// training-step graph by <see cref="Shorokoo.Core.Nodes.Processors.Fast.FastReplay.ReplayInto"/>
        /// against the shared step-counter input.
        /// </summary>
        /// <summary>The reserved counter inputs a scheduler graph may consume, in canonical order (D1).</summary>
        internal static readonly string[] CounterInputNames = ["step", "epoch", "batchIndex"];

        /// <summary>A built scheduler graph and the counter inputs it consumes, in the graph's input order.</summary>
        private readonly record struct SchedulerGraph(InternalComputationGraph Graph, string[] CounterNames);

        private static SchedulerGraph BuildSchedulerModule(Hyperparameter hv, string name)
        {
            if (hv.AsSchedule is Schedule schedule)
            {
                if (!schedule.CanLower())
                    throw new ArgumentException(
                        $"Scheduled hyperparameter '{name}' wraps an opaque host function and cannot be " +
                        "lowered to graph math. Build the schedule from the Schedules factories and " +
                        "Schedule combinators, or supply a scheduler module.", nameof(hv));
                // Built-in DSL schedules are step-only (PerEpoch derives epoch in-graph from step, #39).
                var step = Shorokoo.Globals.InputScalar<int64>("step");
                var value = schedule.LowerToGraph(step);
                return new SchedulerGraph(new InternalComputationGraph([step], [value]), ["step"]);
            }

            var module = hv.AsSchedulerModule
                ?? throw new InvalidOperationException(
                    $"Scheduled hyperparameter '{name}' has neither a built-in schedule nor a scheduler module.");
            return ValidateAndInlineSchedulerModule(module, name);
        }

        /// <summary>
        /// Validates a user scheduler module's signature — its inputs a subset of the reserved int64
        /// scalar counters <c>{step, epoch, batchIndex}</c> (D1; each named, rank-0, no duplicates) and
        /// a single float32 scalar output — enforces purity (D4), and returns its inlined graph together
        /// with the counter names it consumes (in input order, for wiring). Fails loud at rig build with
        /// a clear message on any signature/purity mismatch.
        /// </summary>
        private static SchedulerGraph ValidateAndInlineSchedulerModule(ComputationGraph module, string name)
        {
            if (module.Kind is not (GraphKind.Module or GraphKind.ConcreteArchitecture or GraphKind.ConcreteModel))
                throw new ArgumentException(
                    $"Scheduler module for hyperparameter '{name}' must be a module graph (e.g. " +
                    $"MyScheduler.ComputationGraph); got graph kind '{module.Kind}'.", nameof(module));

            var g = module.ToInternal().Clone();

            // Inline any sub-modules/functions so no MODEL_INVOKE / FUNCTION_INVOKE survives into the
            // training-step graph (the training-graph lowering does not inline modules).
            if (HasHighLevelForms(g))
            {
                Shorokoo.Core.Nodes.Processors.Fast.FastApplyIdentifierTemplates.Process(g);
                Shorokoo.Core.Nodes.Processors.Fast.FastInlineModulesAndFunctions.Process(g);
                Shorokoo.Core.Nodes.Processors.Fast.FastProcessorHelper.RemoveUnreachableNodes(g);
            }

            // Purity contract (D4): a scheduler graph is a pure function of its counter inputs. After
            // inlining, reject any trainable param, module state / StateUpdate, or RNG draw — impure
            // constructs would be inlined into the trainstep with an undefined failure mode.
            AssertSchedulerGraphPure(g, name);

            // Each input must be a named reserved counter (int64 scalar), with no duplicates (D1).
            var producerByOutput = BuildProducerByOutputMap(g);
            var counterNames = new string[g.Inputs.Count];
            var seen = new HashSet<string>(StringComparer.Ordinal);
            for (int i = 0; i < g.Inputs.Count; i++)
            {
                var inName = i < g.InputUniqueNames.Count ? g.InputUniqueNames[i] : null;
                if (inName is null || Array.IndexOf(CounterInputNames, inName) < 0)
                    throw new ArgumentException(
                        $"Scheduler module for hyperparameter '{name}' has input '{inName ?? "(unnamed)"}', " +
                        $"which is not a reserved counter. Its inputs must be named from " +
                        $"[{string.Join(", ", CounterInputNames)}].", nameof(module));
                if (!seen.Add(inName))
                    throw new ArgumentException(
                        $"Scheduler module for hyperparameter '{name}' takes the counter '{inName}' more than once.",
                        nameof(module));

                var inProducer = producerByOutput[g.Inputs[i]];
                var inDType = inProducer.Attributes.GetDTypeVal(OnnxOpAttributeNames.AttrDtype);
                var inRank = (int?)inProducer.Attributes.GetLongVal(OnnxOpAttributeNames.ShrkAttrRank);
                if (inDType != DType.Int64 || (inRank is int ir && ir != 0))
                    throw new ArgumentException(
                        $"Scheduler module for hyperparameter '{name}' counter '{inName}' must be an int64 " +
                        $"scalar (rank-0); got {inDType?.ToString() ?? "unknown"} rank {inRank?.ToString() ?? "?"}.",
                        nameof(module));
                counterNames[i] = inName;
            }

            if (g.Outputs.Count != 1)
                throw new ArgumentException(
                    $"Scheduler module for hyperparameter '{name}' must produce exactly one output " +
                    $"(the scheduled value), but produces {g.Outputs.Count}.", nameof(module));

            // Validate the output dtype/shape via shape inference at the initial counters (all 0), which
            // also smoke-checks that the module executes at all.
            var zeros = new TensorData[g.Inputs.Count];
            for (int i = 0; i < zeros.Length; i++)
                zeros[i] = (TensorData)Shorokoo.Globals.TensorData(Array.Empty<long>(), 0L);
            var outInfo = new ShapeInferenceInterpreter(ComputeContext.Default)
                .Infer(g, zeros)
                .GetTensorInfo(g.Outputs[0])
                ?? throw new ArgumentException(
                    $"Scheduler module for hyperparameter '{name}': could not infer its output shape.",
                    nameof(module));
            if (outInfo.DType != DType.Float32)
                throw new ArgumentException(
                    $"Scheduler module for hyperparameter '{name}' must produce a float32 value; " +
                    $"got {outInfo.DType}.", nameof(module));
            if (outInfo.Shape.Dims.Length != 0)
                throw new ArgumentException(
                    $"Scheduler module for hyperparameter '{name}' must produce a scalar (rank-0) value; " +
                    $"got rank {outInfo.Shape.Dims.Length}.", nameof(module));

            return new SchedulerGraph(g, counterNames);
        }

        /// <summary>
        /// Enforces the scheduler-graph purity contract (D4): fails loud at rig build if the inlined
        /// scheduler graph carries a trainable parameter, module state (a <c>StateUpdate</c> link /
        /// state-deps marker), or an RNG draw. Such a graph would be inlined straight into the
        /// training-step graph, where trainable-param discovery and state threading would misbehave.
        /// (A future learnable/stateful scheduler would relax this into its own constituent kind.)
        /// </summary>
        private static void AssertSchedulerGraphPure(InternalComputationGraph graph, string name)
        {
            foreach (var node in graph.Nodes)
            {
                string? violation = node.OpCode switch
                {
                    InternalOpCodes.MODEL_PARAM or InternalOpCodes.MODEL_PARAM_DATA
                        or InternalOpCodes.MODEL_PARAM_REF or InternalOpCodes.MODEL_PARAM_ID_REF
                        or InternalOpCodes.MODEL_PARAM_MODEL_REF
                        => "a trainable/model parameter",
                    InternalOpCodes.STATE_UPDATE_LINK or InternalOpCodes.WITH_STATE_DEPS
                        => "module state (a StateUpdate)",
                    InternalOpCodes.SHRK_RANDOM_UNIFORM or InternalOpCodes.SHRK_RANDOM_NORMAL
                        or InternalOpCodes.SHRK_RNG_UNIFORM or InternalOpCodes.SHRK_RNG_NORMAL
                        or InternalOpCodes.SHRK_RNG_SPLIT
                        => "an RNG draw",
                    _ => null,
                };
                if (violation is not null)
                    throw new ArgumentException(
                        $"Scheduler module for hyperparameter '{name}' must be a pure function of its " +
                        $"counter input(s), but carries {violation}. Scheduler graphs may use only " +
                        "arithmetic over the counter inputs — no trainable params, module state/RNG.",
                        nameof(name));
            }
        }

        /// <summary>True if <paramref name="graph"/> still carries un-inlined module/function forms.</summary>
        private static bool HasHighLevelForms(InternalComputationGraph graph)
        {
            foreach (var node in graph.Nodes)
            {
                if (node.OpCode == InternalOpCodes.MODEL_INVOKE
                    || node.OpCode == InternalOpCodes.FUNCTION_INVOKE
                    || node.OpCode == InternalOpCodes.MODEL_PARAM_REF
                    || node.OpCode == InternalOpCodes.MODEL_PARAM_MODEL_REF
                    || node.OpCode == InternalOpCodes.MODEL_PARAM_ID_REF)
                    return true;
            }
            return false;
        }

        private static Dictionary<FastTensorKey, FastNode> BuildProducerByOutputMap(InternalComputationGraph graph)
        {
            var map = new Dictionary<FastTensorKey, FastNode>();
            foreach (var node in graph.Nodes)
            {
                foreach (var (_, outs) in node.FullOutputs)
                {
                    foreach (var ok in outs)
                    {
                        if (ok is FastTensorKey k && !k.IsEmpty)
                            map[k] = node;
                    }
                }
            }
            return map;
        }

        private static TensorStructDef? ReadStructDefFromInput(
            InternalComputationGraph graph,
            Dictionary<FastTensorKey, FastNode> producerByOutput,
            FastTensorKey inputKey)
        {
            return producerByOutput.TryGetValue(inputKey, out var producer)
                ? ReadStructDefFromProducer(producer)
                : null;
        }

        private static TensorStructDef? ReadStructDefFromProducer(FastNode producer)
        {
            DType? dtype = producer.OpCode == InternalOpCodes.TENSOR_STRUCT_GETFIELD
                ? producer.Attributes.GetDTypeVal(OnnxOpAttributeNames.ShrkAttrDtype)
                : producer.Attributes.GetDTypeVal(OnnxOpAttributeNames.AttrDtype);
            return dtype?.TensorStructDef;
        }

        /// <summary>
        /// Lowers a high-level training graph to an executable graph in place.
        /// Pipeline: expand struct outputs → unpack TensorStructs → simplify → unroll loops → process autograd → simplify.
        /// The unroll step is required because autograd has no gradient for Loop nodes, so any
        /// loop over trainable parameters (e.g. ResNet residual stacks) must be flattened before
        /// the autograd pass runs.
        /// </summary>
        private static InternalComputationGraph LowerGraph(InternalComputationGraph fast)
        {
            // Expand TensorStruct outputs into individual field outputs.
            Shorokoo.Core.Nodes.Processors.Fast.FastExpandStructOutputs.Process(fast);

            // Unpack TensorStruct inputs (struct → individual fields).
            Shorokoo.Core.Nodes.Processors.Fast.FastUnpackTensorStructs.Process(fast);

            // Simplify before loop unrolling. Any loop whose iteration count is already a direct
            // Constant node will be unrolled here via FastFoldConstantIterationLoops inside
            // FastSimplify.
            Shorokoo.Core.Nodes.Processors.Fast.FastSimplify.Process(fast);

            // Resolve any remaining LOOP_OPEN iteration counts that are computed from constants
            // (e.g. Sub(Constant(2), Constant(1))) into literal Constant nodes. Autograd has no
            // gradient implementation for Loop, so every loop reaching autograd must be flattened.
            FastFoldLoopIterationCountsToConstantsProcessor.Process(fast, ComputeContext.Default);

            // Simplify after iteration-count resolution; the FastFoldConstantIterationLoops pass
            // inside FastSimplify performs the actual unroll, then folds remaining constants.
            Shorokoo.Core.Nodes.Processors.Fast.FastSimplify.Process(fast);

            // Lower attribute-tensorized variant ops (e.g. SHRK_CONV) to standard ONNX ops before
            // autograd — they have no gradient rule. Loops are unrolled by this point, so their
            // geometry inputs are constant-foldable.
            Shorokoo.Core.Nodes.Processors.Fast.FastLowerAttributeTensorOps.Process(fast, compute: ComputeContext.Default);

            // Lower AUTO_GRAD nodes natively on the Fast graph — no CG round-trip needed.
            Shorokoo.Core.Nodes.Processors.AutoGrad.FastProcessAutoGradProcessor.Process(fast);

            Shorokoo.Core.Nodes.Processors.Fast.FastSimplify.Process(fast);
            return fast;
        }

        /// <summary>
        /// Executes a single training step and advances the checkpoint's
        /// <see cref="TrainingCheckpoint.Step"/>. Scheduled hyperparameters (a built-in
        /// <see cref="Schedule"/> or a scheduler module) are computed <b>in-graph</b> from the
        /// checkpoint's current step — fed as the step-counter input — so nothing is host-evaluated
        /// here: compile once, then just loop. This overload requires the rig to have <b>no</b>
        /// schedule-less runtime hyperparameter (<see cref="Hyperparameter.Runtime"/>), which has no value
        /// to apply automatically; use the explicit-override overload for those.
        /// </summary>
        /// <param name="checkpoint">Current training state (params, model state, optimizer state, step)</param>
        /// <param name="trainingInput">Training input data as TensorDataStruct</param>
        /// <param name="trainingOutput">Training target data as TensorDataStruct</param>
        /// <param name="compiled">Compiled training step graph (from <see cref="ComputeContext.Compile(ComputationGraph)"/>)</param>
        /// <returns>The post-step checkpoint (advanced step, updated params/state) with its
        /// <see cref="TrainingCheckpoint.Loss"/> set to this step's loss.</returns>
        public TrainingCheckpoint TrainStep(
            TrainingCheckpoint checkpoint,
            TensorDataStruct trainingInput,
            TensorDataStruct trainingOutput,
            CompiledGraph compiled)
        {
            if (checkpoint is null) throw new ArgumentNullException(nameof(checkpoint));
            if (HyperparameterStructDef.Fields.Length > 0)
                throw new InvalidOperationException(
                    $"This rig has schedule-less runtime hyperparameter(s) " +
                    $"[{string.Join(", ", DynamicHyperparameterNames)}] with no schedule to apply " +
                    "automatically; supply their values via MakeHyperparameters and the " +
                    "TrainStep(checkpoint, hyperparams, …) overload.");
            return RunStep(checkpoint, hyperparams: null, trainingInput, trainingOutput, compiled);
        }

        /// <summary>
        /// Executes a single training step with explicit hyperparameter values, overriding any
        /// schedules for this step (build the values with <see cref="MakeHyperparameters(float)"/> or
        /// <see cref="MakeHyperparameters(ValueTuple{string, float}[])"/>). Use this for manual control, or
        /// for rigs whose dynamic hyperparameters are schedule-less (<see cref="Hyperparameter.Runtime"/>).
        /// In-graph scheduled hyperparameters (built-in schedules / scheduler modules) are unaffected
        /// by this overload — they are always computed from the step counter — so <paramref name="hyperparams"/>
        /// carries only the schedule-less runtime values.
        /// </summary>
        /// <param name="checkpoint">Current training state (params, model state, optimizer state, step)</param>
        /// <param name="hyperparams">Values for the schedule-less runtime hyperparameters (<see cref="HyperparameterStructDef"/> order).</param>
        /// <param name="trainingInput">Training input data as TensorDataStruct</param>
        /// <param name="trainingOutput">Training target data as TensorDataStruct</param>
        /// <param name="compiled">Compiled training step graph (from <see cref="ComputeContext.Compile(ComputationGraph)"/>)</param>
        /// <returns>The post-step checkpoint (advanced step, updated params/state) with its
        /// <see cref="TrainingCheckpoint.Loss"/> set to this step's loss.</returns>
        public TrainingCheckpoint TrainStep(
            TrainingCheckpoint checkpoint,
            TensorDataStruct hyperparams,
            TensorDataStruct trainingInput,
            TensorDataStruct trainingOutput,
            CompiledGraph compiled)
        {
            if (hyperparams is null) throw new ArgumentNullException(nameof(hyperparams));
            return RunStep(checkpoint, hyperparams, trainingInput, trainingOutput, compiled);
        }

        /// <summary>
        /// Executes a single training step on the next batch drawn from <paramref name="loader"/>,
        /// sourcing the epoch / batch counters from the loader — the single-step analogue of
        /// <see cref="Fit(IDataLoader, int, TrainingCheckpoint?, ComputeContext?)"/>, and the one place
        /// the loader-step-and-counter semantics live (<c>Fit(loader)</c> loops over this).
        ///
        /// <para>The batch's <b>own</b> position (<see cref="DataBatch.Position"/>) drives the scheduler
        /// counters for this step, so a scheduled hyperparameter reading the epoch / batch counters sees
        /// the batch being trained. The returned checkpoint then records the loader's <b>new</b> position
        /// — the next batch it will yield (<see cref="IDataLoader.Position"/> after
        /// <see cref="IDataLoader.Next"/>) — as its <see cref="TrainingCheckpoint.Epoch"/> /
        /// <see cref="TrainingCheckpoint.BatchIndex"/>, matching the checkpoint-wide invariant that those
        /// counters are the <b>resume</b> position: feeding the returned checkpoint back to
        /// <c>Fit(loader)</c> (or <see cref="IDataLoader.Restore"/>) continues from exactly the batch this
        /// step stopped before. <see cref="TrainingCheckpoint.Step"/> is advanced by one and the attached
        /// <see cref="TrainingCheckpoint.Rig"/> and this step's <see cref="TrainingCheckpoint.Loss"/> are
        /// preserved (via <see cref="TrainingCheckpoint.WithCounters"/>).</para>
        ///
        /// <para>Like the counter-agnostic <see cref="TrainStep(TrainingCheckpoint, TensorDataStruct, TensorDataStruct, CompiledGraph)"/>
        /// it drives, this schedule-driven form requires the rig to have no schedule-less runtime
        /// hyperparameter (<see cref="Hyperparameter.Runtime"/>); supply those via
        /// <see cref="MakeHyperparameters(float)"/> and a manual explicit-data loop instead.</para>
        /// </summary>
        /// <param name="checkpoint">Current training state; its counters are replaced from the loader.</param>
        /// <param name="loader">The data loader; <see cref="IDataLoader.Next"/> is called once.</param>
        /// <param name="compiled">Compiled training step graph.</param>
        /// <returns>The post-step checkpoint: step advanced, epoch / batch set to the loader's next
        /// (resume) position, with this step's loss.</returns>
        public TrainingCheckpoint TrainStep(
            TrainingCheckpoint checkpoint,
            IDataLoader loader,
            CompiledGraph compiled)
        {
            if (checkpoint is null) throw new ArgumentNullException(nameof(checkpoint));
            if (loader is null) throw new ArgumentNullException(nameof(loader));
            if (compiled is null) throw new ArgumentNullException(nameof(compiled));

            var batch = loader.Next();
            // The batch's own position drives the scheduler counters for THIS step (a scheduler reading
            // epoch / batchIndex sees the batch being trained). TrainStep advances Step and preserves the
            // attached rig and this step's loss.
            var stepInput = checkpoint.WithCounters(
                epoch: batch.Position.Epoch, batchIndex: batch.Position.BatchIndex);
            var stepped = TrainStep(stepInput, batch.Input, batch.Target, compiled);

            // Record the loader's NEW position (the next batch to yield) as the checkpoint's resume
            // position, so a later Fit(loader) / TrainStep(loader) that Restores from it continues exactly.
            var next = loader.Position;
            return stepped.WithCounters(epoch: next.Epoch, batchIndex: next.BatchIndex);
        }

        /// <summary>
        /// Executes a single training step on caller-supplied data with an explicit epoch and batch
        /// number — the counter-sourcing analogue of
        /// <see cref="TrainStep(TrainingCheckpoint, IDataLoader, CompiledGraph)"/> for a host driving its
        /// own data iteration (no <see cref="IDataLoader"/>).
        ///
        /// <para><b>What the recorded counters mean.</b> <paramref name="epoch"/> and
        /// <paramref name="batchNumber"/> name the position of the batch you are training now. They are
        /// fed to any scheduled hyperparameter reading the epoch / batch counters during this step, and
        /// are recorded <b>verbatim</b> on the returned checkpoint's <see cref="TrainingCheckpoint.Epoch"/>
        /// / <see cref="TrainingCheckpoint.BatchIndex"/>. Unlike the loader overload this one does
        /// <b>not</b> advance to a "next batch" position — without a loader it has no batches-per-epoch to
        /// know when a batch index rolls into the next epoch, so verbatim recording is the only
        /// well-defined contract. Consequently, since a loader-driven resume restores from the recorded
        /// counters, resuming <c>Fit(loader)</c> from a checkpoint produced here would re-run this batch;
        /// a manual loop that owns its own iteration simply passes each batch's position and manages
        /// resume itself. <see cref="TrainingCheckpoint.Step"/> is advanced by one; the attached
        /// <see cref="TrainingCheckpoint.Rig"/> and this step's <see cref="TrainingCheckpoint.Loss"/> are
        /// preserved.</para>
        ///
        /// <para>Like <see cref="TrainStep(TrainingCheckpoint, TensorDataStruct, TensorDataStruct, CompiledGraph)"/>,
        /// this schedule-driven form requires the rig to have no schedule-less runtime hyperparameter
        /// (<see cref="Hyperparameter.Runtime"/>); use the explicit-hyperparameters overload and set the
        /// counters via <see cref="TrainingCheckpoint.WithCounters"/> for those.</para>
        /// </summary>
        /// <param name="checkpoint">Current training state; its epoch / batch counters are replaced by the arguments.</param>
        /// <param name="trainingInput">Training input data as TensorDataStruct.</param>
        /// <param name="trainingOutput">Training target data as TensorDataStruct.</param>
        /// <param name="epoch">The 0-based epoch of the batch being trained; recorded verbatim.</param>
        /// <param name="batchNumber">The 0-based batch index of the batch being trained; recorded verbatim.</param>
        /// <param name="compiled">Compiled training step graph.</param>
        /// <returns>The post-step checkpoint: step advanced, epoch / batch set to the given values, with this step's loss.</returns>
        public TrainingCheckpoint TrainStep(
            TrainingCheckpoint checkpoint,
            TensorDataStruct trainingInput,
            TensorDataStruct trainingOutput,
            long epoch,
            long batchNumber,
            CompiledGraph compiled)
        {
            if (checkpoint is null) throw new ArgumentNullException(nameof(checkpoint));
            if (epoch < 0)
                throw new ArgumentOutOfRangeException(nameof(epoch), epoch, "Epoch must be non-negative.");
            if (batchNumber < 0)
                throw new ArgumentOutOfRangeException(nameof(batchNumber), batchNumber, "Batch number must be non-negative.");
            var stepInput = checkpoint.WithCounters(epoch: epoch, batchIndex: batchNumber);
            return TrainStep(stepInput, trainingInput, trainingOutput, compiled);
        }

        /// <summary>The checkpoint's value for one reserved counter input ({step, epoch, batchIndex}).
        /// A null (unknown) epoch / batch feeds the scheduler <c>0</c> — the schedule sees the run's
        /// start until a loader / explicit counter gives the position a concrete value.</summary>
        private static long CounterValue(TrainingCheckpoint ckpt, string counter) => counter switch
        {
            "step" => ckpt.Step,
            "epoch" => ckpt.Epoch ?? 0L,
            "batchIndex" => ckpt.BatchIndex ?? 0L,
            _ => throw new InvalidOperationException($"Unknown scheduler counter input '{counter}'."),
        };

        private TrainingCheckpoint RunStep(
            TrainingCheckpoint checkpoint,
            TensorDataStruct? hyperparams,
            TensorDataStruct trainingInput,
            TensorDataStruct trainingOutput,
            CompiledGraph compiled)
        {
            if (checkpoint is null) throw new ArgumentNullException(nameof(checkpoint));
            if (trainingInput is null) throw new ArgumentNullException(nameof(trainingInput));
            if (trainingOutput is null) throw new ArgumentNullException(nameof(trainingOutput));
            if (compiled is null) throw new ArgumentNullException(nameof(compiled));
            if (HyperparameterStructDef.Fields.Length > 0 && hyperparams is null)
                throw new ArgumentNullException(nameof(hyperparams),
                    "This rig was built with dynamic hyperparameters; supply their values each step " +
                    "(see TrainingRig.MakeHyperparameters).");

            // Execute the training step graph.
            // Graph inputs (after lowering): [param_fields..., state_fields..., opt_state_fields..., hyperparam_fields..., counter_inputs..., model_input_fields..., target_fields...]
            // CompiledGraph.Execute expands TensorDataStruct inputs into individual fields; an empty
            // struct contributes no fields. The hyperparams input slot exists only when the rig has
            // schedule-less runtime hyperparameters; the int64 counter inputs {step, epoch, batchIndex}
            // exist only for those a scheduled hyperparameter consumes, and are fed the checkpoint's
            // current counters so the scheduler math resumes correctly from a saved checkpoint.
            var execInputs = new List<IData>(9)
            {
                checkpoint.TrainableParams,
                checkpoint.ModelState,
                checkpoint.OptimizerState,
            };
            if (HyperparameterStructDef.Fields.Length > 0) execInputs.Add(hyperparams!);
            foreach (var counter in _counterInputNames)
                execInputs.Add(Shorokoo.Globals.TensorData(Array.Empty<long>(), CounterValue(checkpoint, counter)));
            execInputs.Add(trainingInput);
            execInputs.Add(trainingOutput);
            var results = compiled.Execute(execInputs.ToArray());

            // Graph outputs (after lowering): [updated_param_field_0, ..., updated_state_field_0, ..., updated_opt_state_field_0, ..., loss]
            // Repack updated param fields into a TensorDataStruct
            var updatedParamFields = new Dictionary<string, IData>();
            for (int i = 0; i < UpdatedParamFieldCount; i++)
            {
                updatedParamFields[TrainableParamStructDef.Fields[i].Name] = results[i].ToTensorData();
            }
            var updatedParams = new TensorDataStruct(TrainableParamStructDef, updatedParamFields);

            // Repack updated state fields into a TensorDataStruct
            var updatedStateFields = new Dictionary<string, IData>();
            for (int i = 0; i < UpdatedStateFieldCount; i++)
            {
                updatedStateFields[ModelStateDef.Fields[i].Name] = results[UpdatedParamFieldCount + i].ToTensorData();
            }
            var updatedModelState = new TensorDataStruct(ModelStateDef, updatedStateFields);

            // Repack updated optimizer state fields into a TensorDataStruct
            var updatedOptStateFields = new Dictionary<string, IData>();
            for (int i = 0; i < UpdatedOptimizerStateFieldCount; i++)
            {
                updatedOptStateFields[OptimizerStateDef.Fields[i].Name] =
                    results[UpdatedParamFieldCount + UpdatedStateFieldCount + i].ToTensorData();
            }
            var updatedOptimizerState = new TensorDataStruct(OptimizerStateDef, updatedOptStateFields);

            // Loss is the last output
            var lossIndex = UpdatedParamFieldCount + UpdatedStateFieldCount + UpdatedOptimizerStateFieldCount;
            var lossValue = results[lossIndex].ToTensorData<float32>().AccessMemory()[0];

            // Step is the graph-advanced counter (one training step per call). Epoch and batch
            // index are host-owned — the training loop advances them — so they carry through
            // unchanged here.
            var newCheckpoint = new TrainingCheckpoint(
                updatedParams,
                updatedModelState,
                updatedOptimizerState,
                checkpoint.Step + 1,
                checkpoint.Epoch,
                checkpoint.BatchIndex,
                this,
                lossValue);

            return newCheckpoint;
        }

        /// <summary>
        /// Runs a full training loop over the training data for the specified number of epochs.
        /// Each element in the input/output arrays represents one training step (typically a pre-batched batch).
        /// </summary>
        /// <param name="initialCheckpoint">Initial training state (with initial parameter values)</param>
        /// <param name="trainingInputs">Array of training input batches (each as TensorDataStruct)</param>
        /// <param name="trainingOutputs">Array of training target batches (each as TensorDataStruct)</param>
        /// <param name="numEpochs">Number of passes over the training data</param>
        /// <param name="ctx">Compute context for execution</param>
        /// <returns>Training result with final checkpoint and per-epoch average losses</returns>
        public TrainingResult Train(
            TrainingCheckpoint initialCheckpoint,
            TensorDataStruct[] trainingInputs,
            TensorDataStruct[] trainingOutputs,
            int numEpochs,
            ComputeContext ctx)
        {
            if (initialCheckpoint is null) throw new ArgumentNullException(nameof(initialCheckpoint));
            if (trainingInputs is null) throw new ArgumentNullException(nameof(trainingInputs));
            if (trainingOutputs is null) throw new ArgumentNullException(nameof(trainingOutputs));
            if (ctx is null) throw new ArgumentNullException(nameof(ctx));
            if (trainingInputs.Length != trainingOutputs.Length)
                throw new ArgumentException("Training inputs and outputs must have the same length.");
            if (numEpochs < 1) throw new ArgumentException("Number of epochs must be at least 1.", nameof(numEpochs));

            var compiled = ctx.Compile(TrainingStepPureGraph);
            var checkpoint = initialCheckpoint;
            var epochLosses = new float[numEpochs];

            for (int epoch = 0; epoch < numEpochs; epoch++)
            {
                float epochLoss = 0;

                for (int i = 0; i < trainingInputs.Length; i++)
                {
                    checkpoint = TrainStep(checkpoint, trainingInputs[i], trainingOutputs[i], compiled);
                    // TrainStep sets the post-step checkpoint's Loss to this step's loss.
                    epochLoss += checkpoint.Loss!.Value;
                }

                epochLosses[epoch] = epochLoss / trainingInputs.Length;
            }

            return new TrainingResult(checkpoint, epochLosses);
        }

        /// <summary>
        /// Fits the model to the data for <paramref name="numEpochs"/> epochs — a one-liner over
        /// <see cref="TrainingRig.TrainStep(TrainingCheckpoint, TensorDataStruct, TensorDataStruct, CompiledGraph)"/>.
        /// Scheduled hyperparameters are applied automatically (the global step advances across epochs
        /// via the checkpoint), so the schedule sees a monotonically increasing step. Alias for
        /// <see cref="Train"/>. <paramref name="initialCheckpoint"/> defaults to
        /// <see cref="CreateInitialCheckpoint()"/> and <paramref name="ctx"/> defaults to
        /// <see cref="ComputeContext.Default"/>, so a minimal call is
        /// <c>rig.Fit(inputs, targets, numEpochs: 10)</c>.
        /// </summary>
        public TrainingResult Fit(
            TensorDataStruct[] trainingInputs,
            TensorDataStruct[] trainingOutputs,
            int numEpochs,
            TrainingCheckpoint? initialCheckpoint = null,
            ComputeContext? ctx = null)
            => Train(initialCheckpoint ?? CreateInitialCheckpoint(), trainingInputs, trainingOutputs, numEpochs, ctx ?? ComputeContext.Default);

        /// <summary>
        /// Fits the model by draining an <see cref="IDataLoader"/> for <paramref name="numEpochs"/>
        /// epochs, advancing the checkpoint's <see cref="TrainingCheckpoint.Step"/>,
        /// <see cref="TrainingCheckpoint.Epoch"/> and <see cref="TrainingCheckpoint.BatchIndex"/> at
        /// the right points — so the host no longer hand-sets epoch / batch: the loader owns the data
        /// stream and the rig reads the counters off it. Each produced checkpoint therefore carries a
        /// correct position, and the returned <see cref="TrainingResult.FinalCheckpoint"/> can be saved
        /// and later resumed by passing it back as <paramref name="initialCheckpoint"/>: the loader is
        /// <see cref="IDataLoader.Restore"/>d to the checkpoint's position, so the run continues from
        /// exactly the next batch it had reached.
        ///
        /// <para>"Epochs" are counted from the checkpoint's current epoch: the loop trains until the
        /// loader reaches <c>startEpoch + numEpochs</c>. Resuming a checkpoint saved mid-epoch first
        /// finishes that partial epoch. Scheduled hyperparameters are applied automatically (the global
        /// step advances across the run); this schedule-driven form requires the rig to have no
        /// schedule-less runtime hyperparameter — supply those via <see cref="MakeHyperparameters(float)"/>
        /// and a manual <see cref="TrainStep(TrainingCheckpoint, TensorDataStruct, TensorDataStruct, TensorDataStruct, CompiledGraph)"/>
        /// loop instead.</para>
        /// </summary>
        /// <param name="loader">The data loader owning the (input, target) batch stream and its position.</param>
        /// <param name="numEpochs">Number of additional epochs to train, counted from the checkpoint's epoch.</param>
        /// <param name="initialCheckpoint">State to resume from; defaults to <see cref="CreateInitialCheckpoint()"/>.</param>
        /// <param name="ctx">Compute context; defaults to <see cref="ComputeContext.Default"/>.</param>
        /// <returns>Final checkpoint (with advanced step / epoch / batch) and the per-epoch mean losses.</returns>
        public TrainingResult Fit(
            IDataLoader loader,
            int numEpochs,
            TrainingCheckpoint? initialCheckpoint = null,
            ComputeContext? ctx = null)
        {
            if (loader is null) throw new ArgumentNullException(nameof(loader));
            if (numEpochs < 1) throw new ArgumentException("Number of epochs must be at least 1.", nameof(numEpochs));
            ctx ??= ComputeContext.Default;

            var checkpoint = initialCheckpoint ?? CreateInitialCheckpoint();

            // Resume: point the loader at the checkpoint's position so training continues from the very
            // next batch. A fresh checkpoint — or one whose epoch / batch is unknown (null), e.g. trained
            // without a loader — starts at (epoch 0, batch 0).
            long startEpoch = checkpoint.Epoch ?? 0L;
            long startBatch = checkpoint.BatchIndex ?? 0L;
            loader.Restore(new DataLoaderPosition(startEpoch, startBatch));

            var compiled = ctx.Compile(TrainingStepPureGraph);
            long targetEpoch = startEpoch + numEpochs;

            var epochLosses = new List<float>();
            long runningEpoch = startEpoch;
            float epochLossSum = 0f;
            int epochBatchCount = 0;

            // Drive off the loader's live position (always concrete) and route each step through the
            // single-step TrainStep(loader) overload — the one source of the loader-step-and-counter
            // semantics. TrainStep(loader) feeds the batch's own position to the scheduler and stamps
            // the loader's next (resume) position onto the returned checkpoint.
            while (loader.Position.Epoch < targetEpoch)
            {
                long batchEpoch = loader.Position.Epoch;   // the epoch of the batch TrainStep(loader) will draw
                checkpoint = TrainStep(checkpoint, loader, compiled);

                // Group per-epoch mean loss by the epoch the batch belonged to.
                if (batchEpoch != runningEpoch)
                {
                    epochLosses.Add(epochBatchCount > 0 ? epochLossSum / epochBatchCount : 0f);
                    runningEpoch = batchEpoch;
                    epochLossSum = 0f;
                    epochBatchCount = 0;
                }
                // TrainStep(loader) sets the post-step checkpoint's Loss to this step's loss.
                epochLossSum += checkpoint.Loss!.Value;
                epochBatchCount++;
            }
            if (epochBatchCount > 0)
                epochLosses.Add(epochLossSum / epochBatchCount);

            return new TrainingResult(checkpoint, epochLosses.ToArray());
        }

        /// <summary>
        /// Returns the default initial checkpoint produced at <see cref="FromScratch(ComputationGraph, ComputationGraph, ComputationGraph, NamedModelParam[], IOptimizerHyperparameters, RngConfig?)"/> time.
        /// Trainable parameters and model state were initialized from the model's built-in
        /// initializers, and optimizer state from the optimizer's [StateInitializer]s (run once per
        /// trainable parameter, at each hyperparameter's value at the initial counters). This is pure
        /// packaging — no computation happens here.
        ///
        /// <para><b>Fails loud (D5)</b> when the optimizer's state initializer actually reads a
        /// <see cref="HyperparameterKind.Runtime"/> hyperparameter, whose value is unknown at build:
        /// supply explicit initial values via <see cref="CreateInitialCheckpoint(TensorDataStruct)"/>
        /// (build them with <see cref="MakeHyperparameters(float)"/>). No silent placeholder is ever
        /// fed to an initializer that reads it.</para>
        /// </summary>
        public TrainingCheckpoint CreateInitialCheckpoint()
        {
            if (_stateInitNeedsRuntimeHypers)
                throw new InvalidOperationException(
                    "This optimizer's state initializer reads runtime hyperparameter(s) " +
                    $"[{string.Join(", ", _stateInitConsumedRuntimeHyperNames)}], whose value is not " +
                    "known when the checkpoint is created. Supply explicit initial values via " +
                    "CreateInitialCheckpoint(MakeHyperparameters(...)).");
            return new TrainingCheckpoint(
                new TensorDataStruct(TrainableParamStructDef, _initialParamFields),
                new TensorDataStruct(ModelStateDef, _initialStateFields),
                new TensorDataStruct(OptimizerStateDef, _initialOptStateFields),
                rig: this);
        }

        /// <summary>
        /// Like <see cref="CreateInitialCheckpoint()"/>, but with explicit initial values for the
        /// <see cref="HyperparameterKind.Runtime"/> hyperparameters (build the struct with
        /// <see cref="MakeHyperparameters(float)"/> / <see cref="MakeHyperparameters(ValueTuple{string, float}[])"/>
        /// — the same struct the per-step override <c>TrainStep</c> takes). Required (D5) when the
        /// optimizer's state initializer reads a runtime hyperparameter; harmless otherwise. Baked and
        /// scheduled hyperparameters still contribute their build-time value at the initial counters.
        /// </summary>
        public TrainingCheckpoint CreateInitialCheckpoint(TensorDataStruct hyperparameters)
        {
            if (hyperparameters is null) throw new ArgumentNullException(nameof(hyperparameters));
            var optState = OptimizerStateDef.Fields.Length > 0
                ? ComputeInitialOptStateFields(
                    ResolveStateInitHyperValues(hyperparameters, throwOnMissingConsumed: true), ComputeContext.Default)
                : _initialOptStateFields;
            return new TrainingCheckpoint(
                new TensorDataStruct(TrainableParamStructDef, _initialParamFields),
                new TensorDataStruct(ModelStateDef, _initialStateFields),
                new TensorDataStruct(OptimizerStateDef, optState),
                rig: this);
        }

        /// <summary>
        /// The value each hyperparameter contributes to optimizer state init, in optimizer order:
        /// baked/scheduled hypers use their build-time value at the initial counters
        /// (<see cref="_hyperparamInitialCounterValues"/>); a runtime hyper takes its value from
        /// <paramref name="runtimeHypers"/> when supplied. A runtime hyper the state-init graph
        /// actually <b>consumes</b> must be present (D5): its absence fails loud rather than defaulting
        /// to a placeholder. An unconsumed runtime hyper is irrelevant to state init, so it defaults to 0.
        /// </summary>
        private float[] ResolveStateInitHyperValues(TensorDataStruct? runtimeHypers, bool throwOnMissingConsumed)
        {
            var values = new float[_hyperparamInitialCounterValues.Length];
            for (int i = 0; i < values.Length; i++)
            {
                if (_hyperparamInitialCounterValues[i] is float known) { values[i] = known; continue; }

                // Runtime hyper: use the supplied value; a value the state-init graph actually consumes
                // must be present when the caller means it (throwOnMissingConsumed) — else it is an
                // internal 0 placeholder used only to seed shape inference at build.
                var name = _runtimeHyperNameByOptIndex[i];
                if (runtimeHypers is not null
                    && runtimeHypers.Fields.TryGetValue(name, out var d) && d is TensorData td)
                {
                    values[i] = td.As<float32>().AccessMemory<float>()[0];
                }
                else if (throwOnMissingConsumed && _stateInitConsumedHyperIndices.Contains(i))
                {
                    throw new ArgumentException(
                        $"The optimizer's state initializer reads runtime hyperparameter '{name}', but no " +
                        "value for it was supplied. Pass it via CreateInitialCheckpoint(MakeHyperparameters(...)).",
                        nameof(runtimeHypers));
                }
                else
                {
                    values[i] = 0f;   // unconsumed runtime hyper, or a build-time shape-inference placeholder
                }
            }
            return values;
        }

        /// <summary>
        /// Runs the optimizer's split-off state-init graph once per trainable parameter, binding its
        /// hyperparameter inputs to <paramref name="hyperValuesInOptOrder"/>, the parameter's initial
        /// value, and a zero gradient; returns the initial optimizer-state field values.
        /// </summary>
        private Dictionary<string, IData> ComputeInitialOptStateFields(float[] hyperValuesInOptOrder, ComputeContext ctx)
        {
            var fields = new Dictionary<string, IData>();
            var stateInitGraph = _optimizerStateInitGraph
                ?? throw new InvalidOperationException("Optimizer state fields exist but no state-init graph was produced.");
            var statesPerParam = OptimizerStateDef.Fields.Length / TrainableParamStructDef.Fields.Length;
            var hyperSeeds = hyperValuesInOptOrder
                .Select(v => (TensorData)Shorokoo.Globals.TensorData(Array.Empty<long>(), v))
                .ToArray();

            for (var paramIdx = 0; paramIdx < TrainableParamStructDef.Fields.Length; paramIdx++)
            {
                var paramData = (TensorData)_initialParamFields[TrainableParamStructDef.Fields[paramIdx].Name];
                var bytesPerElement = paramData.DType.EncodingBitCount / 8;
                var zeroGrad = TensorData.CreateFromRawBytes(
                    paramData.Shape, paramData.DType, new byte[paramData.Shape.Count * bytesPerElement]);

                var stateValues = Shorokoo.Core.Nodes.Processors.Fast.FastNormalizeOptimizerGraph
                    .RunStateInitGraph(stateInitGraph, ctx, [.. hyperSeeds, paramData, zeroGrad]);

                for (var s = 0; s < statesPerParam; s++)
                    fields[OptimizerStateDef.Fields[paramIdx * statesPerParam + s].Name] = stateValues[s];
            }
            return fields;
        }

        /// <summary>
        /// The subset of the first <paramref name="count"/> input indices of <paramref name="graph"/>
        /// that are actually reachable from its outputs — the D5 dependency analysis over the optimizer
        /// state-init graph, whose leading inputs are the hyperparameters (then param, grad).
        /// </summary>
        private static HashSet<int> ConsumedInputIndices(InternalComputationGraph graph, int count)
        {
            var producerByOutput = BuildProducerByOutputMap(graph);
            var reached = new HashSet<FastTensorKey>();
            var queue = new Queue<FastTensorKey>(graph.Outputs);
            while (queue.Count > 0)
            {
                var key = queue.Dequeue();
                if (key.IsEmpty || !reached.Add(key)) continue;
                if (producerByOutput.TryGetValue(key, out var node))
                    foreach (var (_, slots) in node.FullInputs)
                        foreach (var s in slots)
                            if (s is FastTensorKey ik && !ik.IsEmpty) queue.Enqueue(ik);
            }
            var consumed = new HashSet<int>();
            for (int i = 0; i < count && i < graph.Inputs.Count; i++)
                if (reached.Contains(graph.Inputs[i])) consumed.Add(i);
            return consumed;
        }

        /// <summary>
        /// Loads a checkpoint previously written by <see cref="TrainingCheckpoint.Save(string, CheckpointComponents?)"/>
        /// (legacy flat safetensors) or <see cref="Persistence.SaveTrainingCheckpointToSkpt"/> (the
        /// native .skpt container) — the on-disk shape is detected automatically — reconstructing it
        /// against this rig's parameter/state struct definitions so training resumes exactly where it
        /// left off: trainable params, optimizer moments, model state, and the host-owned run counters
        /// (global step, epoch, batch index) are all restored (schedules resume from that step; older
        /// checkpoints lacking epoch/batch restore them as 0). Throws if the file's fields don't match this
        /// rig — e.g. a checkpoint produced by a different model or optimizer. The rig must be built
        /// from the same model/loss/optimizer graphs as the one that saved the checkpoint.
        /// </summary>
        public TrainingCheckpoint LoadCheckpoint(string filePath, CheckpointComponents? components = null)
            => TrainingCheckpoint.Load(filePath, this, components);

        /// <summary>
        /// Packs a single dynamic hyperparameter value into a <see cref="TensorDataStruct"/> for the
        /// explicit <see cref="TrainStep(TrainingCheckpoint, TensorDataStruct, TensorDataStruct, TensorDataStruct, CompiledGraph)"/>
        /// overload. Convenience for the common case of exactly one dynamic hyperparameter (e.g. the
        /// learning rate); throws if the rig has a different number. For multiple, use the named overload.
        /// </summary>
        public TensorDataStruct MakeHyperparameters(float value)
        {
            if (HyperparameterStructDef.Fields.Length != 1)
                throw new InvalidOperationException(
                    $"MakeHyperparameters(float) requires exactly one dynamic hyperparameter; this rig has " +
                    $"{HyperparameterStructDef.Fields.Length} ([{string.Join(", ", DynamicHyperparameterNames)}]). " +
                    "Use MakeHyperparameters((name, value), …).");
            return PackHyperparams([value]);
        }

        /// <summary>
        /// Packs named dynamic hyperparameter values into a <see cref="TensorDataStruct"/> for the
        /// explicit <see cref="TrainStep(TrainingCheckpoint, TensorDataStruct, TensorDataStruct, TensorDataStruct, CompiledGraph)"/>
        /// overload. Every dynamic hyperparameter must be named exactly once (case-insensitive); names
        /// are those in <see cref="DynamicHyperparameterNames"/>, e.g.
        /// <c>MakeHyperparameters(("learningRate", lr), ("weightDecay", wd))</c>.
        /// </summary>
        public TensorDataStruct MakeHyperparameters(params (string name, float value)[] values)
        {
            if (values is null) throw new ArgumentNullException(nameof(values));

            var byName = new Dictionary<string, float>(StringComparer.OrdinalIgnoreCase);
            foreach (var (name, value) in values)
            {
                if (name is null) throw new ArgumentException("Hyperparameter name cannot be null.", nameof(values));
                if (!byName.TryAdd(name, value))
                    throw new ArgumentException($"Hyperparameter '{name}' was supplied more than once.", nameof(values));
            }

            var ordered = new float[HyperparameterStructDef.Fields.Length];
            for (int i = 0; i < HyperparameterStructDef.Fields.Length; i++)
            {
                var fieldName = HyperparameterStructDef.Fields[i].Name;
                if (!byName.Remove(fieldName, out var v))
                    throw new ArgumentException(
                        $"Missing value for dynamic hyperparameter '{fieldName}'. Expected exactly: " +
                        $"[{string.Join(", ", DynamicHyperparameterNames)}].", nameof(values));
                ordered[i] = v;
            }
            if (byName.Count > 0)
                throw new ArgumentException(
                    $"Unknown dynamic hyperparameter(s): [{string.Join(", ", byName.Keys)}]. Expected exactly: " +
                    $"[{string.Join(", ", DynamicHyperparameterNames)}].", nameof(values));

            return PackHyperparams(ordered);
        }

        /// <summary>Packs values (in <see cref="HyperparameterStructDef"/> field order) into a scalar-field struct.</summary>
        private TensorDataStruct PackHyperparams(float[] orderedValues)
        {
            var fields = new KeyValuePair<string, IData>[orderedValues.Length];
            for (int i = 0; i < orderedValues.Length; i++)
                fields[i] = new KeyValuePair<string, IData>(
                    HyperparameterStructDef.Fields[i].Name,
                    Shorokoo.Globals.TensorData(Array.Empty<long>(), orderedValues[i]));
            return new TensorDataStruct(HyperparameterStructDef, fields);
        }

        /// <summary>
        /// Phase 2: read the concrete-architecture graph for initial trainable / state
        /// parameter values, run the optimizer's state initializers per trainable parameter,
        /// derive the target tensor shape by shape-inferring the concrete model, and run shape
        /// inference + <see cref="MemoryAwareGraphOptimizer"/> on the lowered training-step
        /// graph.
        /// </summary>
        private void InitializeAndOptimize(
            InternalComputationGraph concreteArch,
            ComputeContext ctx,
            RngConfig? rngConfig = null)
        {
            // The model inputs for shape inference are read off the concrete arch's own
            // representative-input attributes (recorded once at BuildInitialRig) — no separate
            // sample-input field. Zero-filled shapes for small inputs; shape+dtype-only placeholders
            // for large ones (QEE keeps those shape-only anyway), so no big buffer is materialized.
            var modelInputExemplars = ReadRepresentativeInputs(concreteArch);

            // Step 1: walk concreteArch's MODEL_PARAM nodes in linear order to capture
            // each one's (ModelId, isTrainable). The same linear order is what Phase 1's
            // FastReplaceTrainableParamsWithInputProcessor used to build the param /
            // state struct defs, so this ordering aligns Phase 2 values with Phase 1 fields.
            // FastInitializeModelParams runs the initializer functions and returns
            // ModelId → TensorData; reindex by our captured order for alignment.
            var trainableModelIds = new List<ModelId>();
            var stateModelIds = new List<ModelId>();
            foreach (var node in concreteArch.Nodes)
            {
                if (node.OpCode != InternalOpCodes.MODEL_PARAM) continue;
                var modelIdVals = node.Attributes.GetIntsVal(OnnxOpAttributeNames.ShrkAttrLocalModelId).AssertNotNull();
                var modelId = new ModelId(modelIdVals);
                var isTrainable = node.Attributes.GetBoolVal(OnnxOpAttributeNames.ShrkAttrIsTrainable) ?? true;
                (isTrainable ? trainableModelIds : stateModelIds).Add(modelId);
            }

            // Pass the concrete param infos so keyed per-parameter init actually engages:
            // FastInitializeModelParams keys init noise only when BOTH rngConfig and paramInfos
            // are non-null. Without the infos the rig would silently fall back to the legacy
            // seeded init, ignoring the config's master seed / algorithm for the weights.
            var paramInfos = rngConfig is null ? null : concreteArch.GetConcreteModelParamInfos();
            var paramValuesById = Shorokoo.Core.Nodes.Processors.Fast.FastInitializeModelParams.Process(
                concreteArch, ctx, rngConfig, paramInfos);

            if (trainableModelIds.Count != TrainableParamStructDef.Fields.Length)
                throw new InvalidOperationException(
                    $"Initialized {trainableModelIds.Count} trainable params but expected " +
                    $"{TrainableParamStructDef.Fields.Length}. State: {stateModelIds.Count} vs " +
                    $"expected {ModelStateDef.Fields.Length}.");

            _initialParamFields = new Dictionary<string, IData>();
            for (var i = 0; i < TrainableParamStructDef.Fields.Length; i++)
                _initialParamFields[TrainableParamStructDef.Fields[i].Name] = paramValuesById[trainableModelIds[i]];

            _initialStateFields = new Dictionary<string, IData>();
            for (var i = 0; i < ModelStateDef.Fields.Length; i++)
                _initialStateFields[ModelStateDef.Fields[i].Name] = paramValuesById[stateModelIds[i]];

            // Initial optimizer state: run the optimizer's state initializers once per trainable
            // parameter, binding the optimizer's hyperparameter inputs to their value at the initial
            // counters (§2.5's single value route — baked constant, or scheduler graph evaluated via
            // QEE at build; no more hardcoded 0f for scheduler modules), the parameter's initial
            // value, and a zero gradient. The state-init graph carries the [StateInitializer]
            // functions split out of the optimizer graph by FastNormalizeOptimizerGraph.
            _initialOptStateFields = new Dictionary<string, IData>();
            if (OptimizerStateDef.Fields.Length > 0)
            {
                var stateInitGraph = _optimizerStateInitGraph
                    ?? throw new InvalidOperationException(
                        "Optimizer state fields exist but no state-init graph was produced.");

                // D5: which hyperparameters does the state-init graph actually consume? A runtime hyper
                // it reads has no build-time value, so defer to CreateInitialCheckpoint(hyperparameters)
                // and fail loud on the no-arg path — no silent placeholder ever reaches an initializer.
                _stateInitConsumedHyperIndices =
                    ConsumedInputIndices(stateInitGraph, _hyperparamInitialCounterValues.Length);
                var consumedRuntime = _runtimeHyperNameByOptIndex.Keys
                    .Where(_stateInitConsumedHyperIndices.Contains).OrderBy(i => i).ToList();
                _stateInitNeedsRuntimeHypers = consumedRuntime.Count > 0;
                _stateInitConsumedRuntimeHyperNames =
                    consumedRuntime.Select(i => _runtimeHyperNameByOptIndex[i]).ToArray();

                // Always compute state seed values so shape inference / optimization below has
                // shape-correct optimizer-state tensors. A consumed runtime hyper contributes an
                // internal 0 placeholder here (shape only — the state init is shape-driven); the real
                // value is required (and recomputed) in CreateInitialCheckpoint(hyperparameters), and
                // the no-arg CreateInitialCheckpoint fails loud on the _stateInitNeedsRuntimeHypers flag.
                _initialOptStateFields = ComputeInitialOptStateFields(
                    ResolveStateInitHyperValues(null, throwOnMissingConsumed: false), ctx);
            }

            // Step 2: derive the target tensor's shape from the model's prediction. Reuse
            // the already-computed paramValuesById via FastApplyModelParamValues — this
            // rewrites MODEL_PARAM → MODEL_PARAM_DATA in place without a
            // second initializer-execution pass.
            var shapeInferencer = new ShapeInferenceInterpreter(ctx);
            var concreteModel = Shorokoo.Core.Nodes.Processors.Fast.FastApplyModelParamValues.Process(concreteArch, paramValuesById);
            var modelShapeInfo = shapeInferencer.Infer(concreteModel, modelInputExemplars);
            var modelOutputInfo = modelShapeInfo.GetTensorInfo(concreteModel.Outputs[0])
                ?? throw new InvalidOperationException(
                    "Shape inference of concrete model graph failed to produce an output shape.");
            var targetShape = modelOutputInfo.Shape;
            var targetDType = modelOutputInfo.DType;

            // Step 3: Assemble inputs in TrainingStepPureGraph order.
            // Layout: [param_fields, state_fields, opt_state_fields, hyperparam_fields, counter_inputs..., model_input_fields, target_fields].
            // Current losses (L2, CE) use a single Tensor target, so target_field_count is 1.
            var graph = _trainingStepWorkGraph!;
            const int targetFieldCount = 1;
            var counterFieldCount = _counterInputNames.Length;
            var expectedModelInputFields =
                graph.Inputs.Count
                - TrainableParamStructDef.Fields.Length
                - ModelStateDef.Fields.Length
                - OptimizerStateDef.Fields.Length
                - HyperparameterStructDef.Fields.Length
                - counterFieldCount
                - targetFieldCount;
            if (modelInputExemplars.Length != expectedModelInputFields)
                throw new ArgumentException(
                    $"Expected {expectedModelInputFields} model-input shape exemplars (one per model " +
                    $"input field), got {modelInputExemplars.Length}.",
                    nameof(modelInputExemplars));

            var allInputs = new TensorData[graph.Inputs.Count];
            var idx = 0;

            foreach (var f in TrainableParamStructDef.Fields)
                allInputs[idx++] = (TensorData)_initialParamFields[f.Name];
            foreach (var f in ModelStateDef.Fields)
                allInputs[idx++] = (TensorData)_initialStateFields[f.Name];
            foreach (var f in OptimizerStateDef.Fields)
                allInputs[idx++] = (TensorData)_initialOptStateFields[f.Name];

            // Schedule-less runtime hyperparameter fields: seed shape inference / optimization with
            // their default (initial) scalar values. At run time these are supplied per step.
            foreach (var f in HyperparameterStructDef.Fields)
                allInputs[idx++] = (TensorData)_initialHyperparamFields[f.Name];

            // Counter-input fields (int64 scalars): seed shape inference at the initial counters (0);
            // the scheduler math downstream computes the hyperparameter values from them. At run time
            // each is fed the checkpoint's corresponding counter.
            foreach (var _ in _counterInputNames)
                allInputs[idx++] = (TensorData)Shorokoo.Globals.TensorData(Array.Empty<long>(), 0L);

            // Model-input fields: one zero-filled shape exemplar per model input, in the model
            // graph's input order — shape inference reads only their shapes.
            foreach (var exemplar in modelInputExemplars)
                allInputs[idx++] = exemplar;

            // Remaining inputs are target fields (typically one Tensor target for L2/CE losses).
            // Synthesize zero tensors with the predicted output shape.
            while (idx < graph.Inputs.Count)
            {
                var bytesPerElement = targetDType.EncodingBitCount / 8;
                var zeroBytes = new byte[targetShape.Count * bytesPerElement];
                allInputs[idx++] = TensorData.CreateFromRawBytes(targetShape, targetDType, zeroBytes);
            }

            // Step 4: Shape inference + memory-aware graph optimization. The optimizer
            // alternates Rematerializer and MemoryAwareScheduler under a combined
            // compute+memory metric, only committing transforms that strictly improve it.
            var shapeInfo = shapeInferencer.Infer(graph, allInputs);
            var baselineEval = new Shorokoo.Core.AutoDiffCheckpointing.GraphEvaluator().Evaluate(graph, shapeInfo);
            var optimizer = new MemoryAwareGraphOptimizer(shapeInference: shapeInferencer);
            var optResult = optimizer.OptimizeWithShapeInfo(graph, shapeInfo);
            PreOptimizationEval = baselineEval;
            OptimizationResult = optResult;

            // Freeze the public views: the working graphs are relinquished into the
            // readonly wrappers, which own them exclusively from here on (the rig
            // compiles through the wrappers, which copy).
            PreOptimizationGraph = new ComputationGraph(graph, GraphKind.ConcreteModel);
            TrainingStepPureGraph = new ComputationGraph(optResult.OptimizedGraph, GraphKind.ConcreteModel);
            _trainingStepWorkGraph = null;
        }
    }

    /// <summary>
    /// The immutable <b>constituent</b> layer of a <see cref="TrainingRig"/> (§5.8): the swappable
    /// source-of-truth models plus the hyperparameters and RNG config needed to (re-)derive the
    /// in-memory <c>trainstep</c>. A <c>With…</c> derivation produces a new value with <c>record
    /// with</c>, sharing every unchanged constituent (and its graph) by reference and re-deriving only
    /// what changed. Sample inputs are deliberately NOT here: they are a construction-time argument,
    /// consumed once to produce the rig's retained concrete arch (and its shape exemplars) and never
    /// stored — the derivation path reuses that arch, so it needs no sample inputs. Held as an
    /// implementation value on the rig; the rig exposes the individual constituents through its own
    /// public accessors.
    /// </summary>
    internal sealed record RigConstituents(
        ComputationGraph Model,
        ComputationGraph Loss,
        ComputationGraph Optimizer,
        Hyperparameter[] Hyperparameters,
        IReadOnlyList<string>? Names,
        RngConfig RngConfig);
}
