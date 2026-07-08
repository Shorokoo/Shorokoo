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
        public int Step { get; }

        /// <summary>Packages trainable params, model state and optimizer state at <paramref name="step"/>.</summary>
        public TrainingCheckpoint(
            TensorDataStruct trainableParams,
            TensorDataStruct modelState,
            TensorDataStruct optimizerState,
            int step = 0)
        {
            TrainableParams = trainableParams ?? throw new ArgumentNullException(nameof(trainableParams));
            ModelState = modelState ?? throw new ArgumentNullException(nameof(modelState));
            OptimizerState = optimizerState ?? throw new ArgumentNullException(nameof(optimizerState));
            Step = step;
        }

        // ---- Inference: bind trained weights into a concrete model for execution ----

        /// <summary>
        /// Builds a concrete inference model from this checkpoint's trained weights in one call,
        /// encapsulating the full <c>ToConcreteArchitecture → weights → ToConcreteModel</c> pipeline.
        /// </summary>
        public FastComputationGraph ToInferenceModel(FastComputationGraph modelGraph, TensorData exampleInput)
        {
            if (modelGraph is null) throw new ArgumentNullException(nameof(modelGraph));
            if (exampleInput is null) throw new ArgumentNullException(nameof(exampleInput));
            var arch    = modelGraph.ToConcreteArchitecture(modelGraph.FromOrderedInputs([exampleInput]));
            var weights = new ModelParamList(
                TrainableParams.Fields
                    .Where(f => f.Value is TensorData)
                    .Select(f => new KeyValuePair<string, TensorData>(f.Key, (TensorData)f.Value)),
                ModelParamType.TrainableParam);
            return arch.ToConcreteModel(weights, arch.GetShorokooIdNamingScheme());
        }

        // ---- Persistence: save a checkpoint to disk and resume across process restarts ----

        // The three sections share one SafeTensors file; each field is namespaced as
        // "<section>/<fieldName>". A Shorokoo field name never contains '/', so the split
        // is unambiguous and the '/'-free marker tensor below can't be mistaken for a field.
        private const string TrainableSection = "trainable";
        private const string ModelStateSection = "model_state";
        private const string OptimizerStateSection = "opt_state";
        private const string CheckpointMarkerName = "__shorokoo_checkpoint__"; // int64[2] = [version, step]
        private const long CheckpointFormatVersion = 1;

        /// <summary>
        /// Saves this checkpoint to a single SafeTensors file so training can resume across process
        /// restarts. Every trainable-parameter, model-state, and optimizer-state field is written as
        /// a namespaced tensor, alongside the global <see cref="Step"/> (so schedules resume from the
        /// right step). Reload with <see cref="TrainingRig.LoadCheckpoint(string)"/> — or with
        /// <see cref="Load"/> if you hold the struct defs directly. A <c>.safetensors</c> extension is
        /// conventional. Fields must be plain tensors (nested-struct fields are unsupported); rank-0
        /// scalars are fine — they serialize as the SafeTensors empty-shape encoding (e.g. an
        /// optimizer's scalar timestep).
        /// </summary>
        public void Save(string filePath)
        {
            if (string.IsNullOrWhiteSpace(filePath))
                throw new ArgumentException("Checkpoint path cannot be null or empty.", nameof(filePath));

            var tensors = new List<SafeTensor>();
            AppendSection(tensors, TrainableSection, TrainableParams);
            AppendSection(tensors, ModelStateSection, ModelState);
            AppendSection(tensors, OptimizerStateSection, OptimizerState);

            var marker = Globals.TensorData(new long[] { 2L }, CheckpointFormatVersion, (long)Step);
            tensors.Add(new SafeTensor(CheckpointMarkerName, marker, "I64", new long[] { 2L }));

            SafeTensorLoader.SaveSafeTensors(filePath, tensors);
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
        /// Loads a checkpoint written by <see cref="Save"/>, reconstructing each section against the
        /// supplied struct defs (normally the ones on the <see cref="TrainingRig"/> the checkpoint was
        /// produced by — prefer <see cref="TrainingRig.LoadCheckpoint(string)"/>, which passes them for
        /// you). Throws if the file is not a Shorokoo checkpoint, was written by a newer format, or its
        /// fields don't match the given defs (e.g. a checkpoint from a different model or optimizer).
        /// </summary>
        public static TrainingCheckpoint Load(
            string filePath,
            TensorStructDef trainableParamDef,
            TensorStructDef modelStateDef,
            TensorStructDef optimizerStateDef)
        {
            if (trainableParamDef is null) throw new ArgumentNullException(nameof(trainableParamDef));
            if (modelStateDef is null) throw new ArgumentNullException(nameof(modelStateDef));
            if (optimizerStateDef is null) throw new ArgumentNullException(nameof(optimizerStateDef));

            var byName = SafeTensorLoader.LoadSafeTensors(filePath).ToDictionary(t => t.Name, t => t.Data);

            if (!byName.TryGetValue(CheckpointMarkerName, out var markerData))
                throw new InvalidOperationException(
                    $"'{filePath}' is not a Shorokoo training checkpoint (missing '{CheckpointMarkerName}' marker).");

            var marker = markerData.As<int64>().AccessMemory<long>();
            if (marker.Length < 2 || marker[0] != CheckpointFormatVersion)
                throw new InvalidOperationException(
                    $"Unsupported checkpoint format version {(marker.Length > 0 ? marker[0] : -1)}; " +
                    $"this build reads version {CheckpointFormatVersion}.");
            int step = checked((int)marker[1]);

            var trainable = ReadSection(byName, TrainableSection, trainableParamDef, filePath);
            var modelState = ReadSection(byName, ModelStateSection, modelStateDef, filePath);
            var optState = ReadSection(byName, OptimizerStateSection, optimizerStateDef, filePath);

            return new TrainingCheckpoint(trainable, modelState, optState, step);
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
    /// Result of a single training step.
    /// </summary>
    public class TrainingStepResult
    {
        /// <summary>The post-step checkpoint (updated params/state, step advanced by one).</summary>
        public TrainingCheckpoint Checkpoint { get; }
        /// <summary>The loss computed for this step.</summary>
        public float Loss { get; }

        /// <summary>Packages a post-step checkpoint and its loss.</summary>
        public TrainingStepResult(TrainingCheckpoint checkpoint, float loss)
        {
            Checkpoint = checkpoint;
            Loss = loss;
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
    /// Inputs:  trainable_params, model_state, optimizer_state, [hyperparams], training_inputs, training_targets
    /// Outputs: updated_trainable_params, updated_model_state, updated_optimizer_state, loss
    ///
    /// Optimizer hyperparameters are baked in as constants by default; any subset can instead be
    /// routed as a runtime "hyperparams" input (see <see cref="HyperparamStructDef"/>) so a schedule
    /// can vary them per step without recompiling.
    ///
    /// The training loop calls TrainStep repeatedly, passing updated state from one step to the next.
    /// </summary>
    public class TrainingRig
    {
        /// <summary>
        /// The lowered, executable computation graph for one training step.
        /// Contains no embedded state — all state flows through inputs/outputs.
        /// </summary>
        public FastComputationGraph TrainingStepPureGraph { get; private set; } = null!;

        /// <summary>Struct definition for trainable parameters.</summary>
        public TensorStructDef TrainableParamStructDef { get; private set; } = null!;

        /// <summary>
        /// Result of the <see cref="MemoryAwareGraphOptimizer"/> pass applied to
        /// <see cref="TrainingStepPureGraph"/>: which strategy won, the per-strategy
        /// evaluations, and the unoptimized baseline graph used as the starting point.
        /// Exposed for diagnostics — lets callers measure how much the optimizer actually
        /// improved the compute / peak-memory metric over the unoptimized graph.
        /// </summary>
        public GraphOptimizationResult OptimizationResult { get; private set; } = null!;

        /// <summary>
        /// The unoptimized training-step graph, before <see cref="MemoryAwareGraphOptimizer"/>
        /// ran. Held alongside <see cref="OptimizationResult"/> so the per-strategy
        /// improvement is measurable.
        /// </summary>
        public FastComputationGraph PreOptimizationGraph { get; private set; } = null!;

        /// <summary>
        /// Compute time + peak memory the <see cref="GraphEvaluator"/> projected for the
        /// unoptimized <see cref="PreOptimizationGraph"/>, under the same shape inference
        /// the optimizer used. Compare with <see cref="OptimizationResult"/>'s evaluation
        /// to quantify the optimizer's improvement.
        /// </summary>
        public GraphEvaluationResult PreOptimizationEval { get; private set; } = null!;

        /// <summary>Struct definition for model state (empty for stateless models).</summary>
        public TensorStructDef ModelStateDef { get; private set; } = null!;

        /// <summary>Struct definition for optimizer state (empty for basic SGD).</summary>
        public TensorStructDef OptimizerStateDef { get; private set; } = null!;

        /// <summary>
        /// Struct definition for the optimizer hyperparameters that flow as <b>runtime inputs</b>
        /// (one scalar <c>float32</c> field per dynamic hyperparameter). Empty when every
        /// hyperparameter is baked into the graph as a constant (the default). When non-empty, the
        /// rig compiles once and the caller supplies fresh values each step via
        /// <see cref="TrainStep(TrainingCheckpoint, TensorDataStruct, TensorDataStruct, TensorDataStruct, CompiledGraph)"/>
        /// — enabling learning-rate / weight-decay schedules without rebuilding the rig.
        /// </summary>
        public TensorStructDef HyperparamStructDef { get; private set; } = null!;

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
        /// <see cref="HyperparamStructDef"/> field order. Used by <see cref="MakeHyperparams(float)"/>
        /// to map caller-supplied values to the right fields.
        /// </summary>
        public IReadOnlyList<int> DynamicHyperparamIndices { get; private set; } = Array.Empty<int>();

        /// <summary>
        /// The optimizer's hyperparameter names, in declaration order (e.g. <c>learningRate, beta1,
        /// …</c>). Derived from the strongly-typed hyperparameter set when one is supplied, else the
        /// fallback <c>hyperparam_{i}</c> names.
        /// </summary>
        public IReadOnlyList<string> HyperparameterNames { get; private set; } = Array.Empty<string>();

        /// <summary>
        /// The names of the dynamic (runtime-input) hyperparameters, in <see cref="HyperparamStructDef"/>
        /// field order — the names accepted by <see cref="MakeHyperparams(ValueTuple{string, float}[])"/>.
        /// </summary>
        public IReadOnlyList<string> DynamicHyperparameterNames { get; private set; } = Array.Empty<string>();

        /// <summary>
        /// Per dynamic hyperparameter (in <see cref="HyperparamStructDef"/> field order), the schedule
        /// the rig evaluates each step, or <c>null</c> for a schedule-less runtime hyperparameter whose
        /// value must be supplied explicitly. Empty when no hyperparameter is dynamic.
        /// </summary>
        private Schedule?[] _dynamicSchedules = Array.Empty<Schedule?>();

        /// <summary>Number of trainable parameter fields in graph outputs.</summary>
        public int UpdatedParamFieldCount { get; private set; }

        /// <summary>Number of model state fields in graph outputs.</summary>
        public int UpdatedStateFieldCount { get; private set; }

        /// <summary>Number of optimizer state fields in graph outputs.</summary>
        public int UpdatedOptimizerStateFieldCount { get; private set; }

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
        private FastComputationGraph? _optimizerStateInitGraph;

        /// <summary>
        /// Hyperparameter seed values (a schedule's step-0 value, or the baked constant), in
        /// optimizer order. Bound as the hyperparameter inputs when the optimizer's state
        /// initializers are executed at FromScratch time.
        /// </summary>
        private float[] _hyperparamSeedValues = Array.Empty<float>();

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
        /// <param name="modelGraph">The model's FastComputationGraph (typically a source-generated module's static graph property)</param>
        /// <param name="lossGraph">The loss function's computation graph (2 inputs: predictions, targets; 1 output: loss)</param>
        /// <param name="optimizerGraph">The optimizer's computation graph (inputs: hyperparams + param + grad; outputs: updated_param). Optimizer state is created inside the module body via optimizer-owned [StateInitializer] Init calls and updated via Globals.StateUpdate — never declared in the signature.</param>
        /// <param name="sampleInputs">Sample model inputs (one per model graph input) used to resolve parameter shapes and seed shape inference. Only the shapes matter, not the values.</param>
        /// <param name="hyperparameters">
        /// The optimizer's named hyperparameters — typically the source-generated set, e.g.
        /// <c>new AdamWOptimizerHyperparameters { LearningRate = Schedules.Cosine(3e-4f, total), WeightDecay = 1e-4f }</c>.
        /// Each value's kind decides its wiring: a bare <see cref="float"/> is baked as a constant; a
        /// <see cref="Schedule"/> is applied per step; <see cref="HyperValue.Runtime"/> is supplied manually.
        /// </param>
        /// <param name="rngConfig">
        /// Optional RNG configuration. When supplied, trainable parameters initialize from
        /// per-parameter keyed streams and the config is bound to the training-step graph
        /// (its key-vector carrier keys every runtime random feed, e.g. Dropout masks),
        /// making the whole run's randomness deterministic and reproducible from the config's
        /// master seed. When <c>null</c>, the legacy in-graph seeded init and the ONNX
        /// random-op fallback are used.
        /// </param>
        /// <returns>A configured TrainingRig ready for training</returns>
        public static TrainingRig FromScratch(
            FastComputationGraph modelGraph,
            FastComputationGraph lossGraph,
            FastComputationGraph optimizerGraph,
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
        /// declared order) rather than as a named set. Each <see cref="HyperValue"/>'s kind still
        /// decides baked-vs-runtime; a bare <c>float</c> implicitly converts to a baked constant, so
        /// <c>FromScratch(model, loss, opt, sample, 0.01f)</c> bakes a single learning rate. Generated
        /// graph fields fall back to <c>hyperparam_{i}</c> names since no names are supplied.
        /// </summary>
        public static TrainingRig FromScratch(
            FastComputationGraph modelGraph,
            FastComputationGraph lossGraph,
            FastComputationGraph optimizerGraph,
            NamedModelParam[] sampleInputs,
            params HyperValue[] hyperparameters)
            => FromScratchCore(modelGraph, lossGraph, optimizerGraph, sampleInputs, hyperparameters,
                names: null, rngConfig: null);

        /// <summary>
        /// Positional-hyperparameter overload with an RNG configuration. The config precedes
        /// the hyperparameter values because a <c>params</c> array must come last.
        /// </summary>
        public static TrainingRig FromScratch(
            FastComputationGraph modelGraph,
            FastComputationGraph lossGraph,
            FastComputationGraph optimizerGraph,
            NamedModelParam[] sampleInputs,
            RngConfig? rngConfig,
            params HyperValue[] hyperparameters)
            => FromScratchCore(modelGraph, lossGraph, optimizerGraph, sampleInputs, hyperparameters,
                names: null, rngConfig: rngConfig);

        /// <summary>
        /// Convenience overload that accepts a <see cref="ModelParamList"/> for sample inputs,
        /// as returned by <c>model.FromOrderedInputs([…])</c>, so you can write
        /// <c>FromScratch(model, Losses.L2Loss, Optimizers.Adam, model.FromOrderedInputs([…]), hypers)</c>
        /// without constructing <see cref="TensorDataModelParam"/> objects by hand.
        /// </summary>
        public static TrainingRig FromScratch(
            FastComputationGraph modelGraph,
            FastComputationGraph lossGraph,
            FastComputationGraph optimizerGraph,
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
            FastComputationGraph modelGraph,
            FastComputationGraph lossGraph,
            FastComputationGraph optimizerGraph,
            ModelParamList sampleInputs,
            params HyperValue[] hyperparameters)
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
            FastComputationGraph modelGraph,
            FastComputationGraph lossGraph,
            FastComputationGraph optimizerGraph,
            ModelParamList sampleInputs,
            RngConfig? rngConfig,
            params HyperValue[] hyperparameters)
        {
            if (sampleInputs is null) throw new ArgumentNullException(nameof(sampleInputs));
            return FromScratch(modelGraph, lossGraph, optimizerGraph,
                sampleInputs.ModelParams.ToArray(), rngConfig, hyperparameters);
        }

        private static TrainingRig FromScratchCore(
            FastComputationGraph modelGraph,
            FastComputationGraph lossGraph,
            FastComputationGraph optimizerGraph,
            NamedModelParam[] sampleInputs,
            HyperValue[] hyperparameters,
            IReadOnlyList<string>? names,
            RngConfig? rngConfig)
        {
            if (modelGraph is null) throw new ArgumentNullException(nameof(modelGraph));
            if (lossGraph is null) throw new ArgumentNullException(nameof(lossGraph));
            if (optimizerGraph is null) throw new ArgumentNullException(nameof(optimizerGraph));
            if (sampleInputs is null) throw new ArgumentNullException(nameof(sampleInputs));
            if (hyperparameters is null) throw new ArgumentNullException(nameof(hyperparameters));
            if (names is not null && names.Count != hyperparameters.Length)
                throw new ArgumentException(
                    $"hyperparameter names ({names.Count}) must match hyperparameter values " +
                    $"({hyperparameters.Length}).", nameof(names));
            if (sampleInputs.Length == 0)
                throw new ArgumentException(
                    "TrainingRig.FromScratch requires at least one sample input. Sample inputs " +
                    "drive parameter shape resolution and training-graph shape inference.",
                    nameof(sampleInputs));

            var ctx = ComputeContext.Default;

            // Single ToConcreteArchitecture pass. The resulting concrete arch graph is the
            // shared substrate for both phases below: Phase 1 composes it with loss +
            // autograd + optimizer to build the training-step graph; Phase 2 reads its
            // TRAINABLE_PARAM nodes to initialize values and to determine prediction shape.
            // The concrete pass also runs the QEE-backed liveness filter that prunes
            // trainable params whose reachability is killed by the sample input shape.
            var concreteArch = modelGraph.ToConcreteArchitecture(new ModelParamList(sampleInputs), ctx);

            // Bind the RNG config at the shared concretization point: binding writes the
            // config's key-vector carrier (one node, no other graph change), which rides
            // unchanged through loss composition and autograd into the training-step graph,
            // where the ONNX-prep lowering derives every feed's keys from it.
            if (rngConfig is not null)
                concreteArch.ApplyRngConfig(rngConfig);

            var rig = new TrainingRig();
            rig.BuildTrainingStepPureGraph(concreteArch, lossGraph, optimizerGraph, hyperparameters, names);
            rig.InitializeAndOptimize(concreteArch, sampleInputs, ctx, rngConfig);
            return rig;
        }

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
            FastComputationGraph concreteArch,
            FastComputationGraph lossGraph,
            FastComputationGraph optimizerGraph,
            HyperValue[] hyperparameters,
            IReadOnlyList<string>? hyperparamNames)
        {
            // Normalize the optimizer graph in place. State variables created by the optimizer's
            // [StateInitializer] Init calls are rewritten into explicit graph inputs appended
            // after grad, and the StateUpdate-pattern nodes (STATE_UPDATE_LINK + WITH_STATE_DEPS)
            // into the explicit multi-output convention [updated_param, updated_state_0, ...]
            // expected by the replay loop. The split-off state-init graph computes the initial
            // state values (per trainable parameter) in InitializeAndOptimize.
            var optimizerFastGraph = optimizerGraph.Clone();
            var optimizerInfo = Shorokoo.Core.Nodes.Processors.Fast.FastNormalizeOptimizerGraph.Process(optimizerFastGraph);
            _optimizerStateInitGraph = optimizerInfo.StateInitGraph;
            _hyperparamSeedValues = hyperparameters.Select(h => h.InitialValue).ToArray();

            // Step 1: Compose model + loss + autograd via TrainingGraphBuilder. The model
            // graph is already through ToConcreteArchitecture (done once at FromScratch),
            // so the input-aware liveness filter has already pruned dead-branch trainable
            // params — FastReplaceTrainableParamsWithInputProcessor inside PrepareForTraining
            // builds the trainable param struct from only the live TRAINABLE_PARAM nodes.
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

            // Each hyperparameter's kind decides its wiring: a dynamic HyperValue (scheduled or
            // schedule-less runtime) flows as a runtime input via a "hyperparams" TensorStruct (one
            // scalar float32 field each); a baked HyperValue stays a graph Constant. The seed value
            // (a schedule's step-0 value, or the constant) is baked into the Constant and seeds shape
            // inference / optimization for dynamic fields.
            string NameOf(int h) => hyperparamNames is not null && h < hyperparamNames.Count
                ? hyperparamNames[h] : $"hyperparam_{h}";
            float SeedOf(int h) => hyperparameters[h].InitialValue;

            HyperparameterNames = Enumerable.Range(0, numHyperparams).Select(NameOf).ToArray();

            var dynamicIndices = new List<int>();
            for (int h = 0; h < numHyperparams; h++)
                if (hyperparameters[h].IsDynamic) dynamicIndices.Add(h);
            DynamicHyperparamIndices = dynamicIndices;

            var hyperFields = dynamicIndices
                .Select(h => new TensorStructFieldDef(NameOf(h), DataStructure.Tensor, 0, DType.Float32))
                .ToArray();
            HyperparamStructDef = new TensorStructDef(hyperFields, "Hyperparams");
            DynamicHyperparameterNames = hyperFields.Select(f => f.Name).ToArray();
            _dynamicSchedules = dynamicIndices.Select(h => hyperparameters[h].AsSchedule).ToArray();

            _initialHyperparamFields = new Dictionary<string, IData>();
            FastTensorKey? hyperparamsInputKey = null;
            var dynamicHyperparamKeyByIndex = new Dictionary<int, FastTensorKey>();
            if (hyperFields.Length > 0)
            {
                var hyperDType = DType.GetOrCreateForTensorStruct(HyperparamStructDef);
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
                    dynamicHyperparamKeyByIndex[dynamicIndices[i]] = new FastTensorKey(node.Key, 0);
                    _initialHyperparamFields[f.Name] =
                        Shorokoo.Globals.TensorData(Array.Empty<long>(), SeedOf(dynamicIndices[i]));
                }
            }

            // Per-hyperparam key: a runtime GETFIELD when dynamic, else a baked CONSTANT.
            // Shared across all per-param optimizer replays.
            var hyperparamKeys = new FastTensorKey[numHyperparams];
            for (int h = 0; h < numHyperparams; h++)
            {
                if (dynamicHyperparamKeyByIndex.TryGetValue(h, out var dynKey))
                {
                    hyperparamKeys[h] = dynKey;
                }
                else
                {
                    var node = Shorokoo.Core.Nodes.Processors.Fast.FastInternalOp.Constant(
                        Shorokoo.Globals.TensorData(Array.Empty<long>(), SeedOf(h)));
                    fastTraining.Nodes.Add(node);
                    headNodesInOrder.Add(node);
                    hyperparamKeys[h] = new FastTensorKey(node.Key, 0);
                }
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
            // Target order:   [param_struct, state_struct?, optimizer_state_struct?, hyperparams_struct?, model_inputs_struct, targets]
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
            TrainingStepPureGraph = LowerGraph(fastTraining);

            UpdatedParamFieldCount = TrainableParamStructDef.Fields.Length;
            UpdatedStateFieldCount = ModelStateDef.Fields.Length;
            UpdatedOptimizerStateFieldCount = OptimizerStateDef.Fields.Length;
        }

        private static Dictionary<FastTensorKey, FastNode> BuildProducerByOutputMap(FastComputationGraph graph)
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
            FastComputationGraph graph,
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
        private static FastComputationGraph LowerGraph(FastComputationGraph fast)
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
        /// <see cref="TrainingCheckpoint.Step"/>. When the rig has scheduled hyperparameters, each
        /// schedule is evaluated at the checkpoint's current step and applied automatically — compile
        /// once, then just loop. A schedule-less runtime hyperparameter (<see cref="HyperValue.Runtime"/>)
        /// has no value to apply here; use the explicit-override overload for those.
        /// </summary>
        /// <param name="checkpoint">Current training state (params, model state, optimizer state, step)</param>
        /// <param name="trainingInput">Training input data as TensorDataStruct</param>
        /// <param name="trainingOutput">Training target data as TensorDataStruct</param>
        /// <param name="compiled">Compiled training step graph (from <see cref="ComputeContext.Compile"/>)</param>
        /// <returns>Result containing the advanced checkpoint and loss value</returns>
        public TrainingStepResult TrainStep(
            TrainingCheckpoint checkpoint,
            TensorDataStruct trainingInput,
            TensorDataStruct trainingOutput,
            CompiledGraph compiled)
        {
            if (checkpoint is null) throw new ArgumentNullException(nameof(checkpoint));
            var hyperparams = HyperparamStructDef.Fields.Length > 0
                ? MakeScheduledHyperparams(checkpoint.Step)
                : null;
            return RunStep(checkpoint, hyperparams, trainingInput, trainingOutput, compiled);
        }

        /// <summary>
        /// Executes a single training step with explicit hyperparameter values, overriding any
        /// schedules for this step (build the values with <see cref="MakeHyperparams(float)"/> or
        /// <see cref="MakeHyperparams(ValueTuple{string, float}[])"/>). Use this for manual control, or
        /// for rigs whose dynamic hyperparameters are schedule-less (<see cref="HyperValue.Runtime"/>).
        /// </summary>
        /// <param name="checkpoint">Current training state (params, model state, optimizer state, step)</param>
        /// <param name="hyperparams">Values for all dynamic hyperparameters (<see cref="HyperparamStructDef"/> order).</param>
        /// <param name="trainingInput">Training input data as TensorDataStruct</param>
        /// <param name="trainingOutput">Training target data as TensorDataStruct</param>
        /// <param name="compiled">Compiled training step graph (from <see cref="ComputeContext.Compile"/>)</param>
        /// <returns>Result containing the advanced checkpoint and loss value</returns>
        public TrainingStepResult TrainStep(
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
        /// Builds the per-step hyperparameter struct by evaluating each dynamic hyperparameter's
        /// schedule at <paramref name="step"/>. Throws when a dynamic hyperparameter is schedule-less
        /// (built with <see cref="HyperValue.Runtime"/>) — supply those via the explicit TrainStep overload.
        /// </summary>
        private TensorDataStruct MakeScheduledHyperparams(int step)
        {
            var values = new float[_dynamicSchedules.Length];
            for (int i = 0; i < _dynamicSchedules.Length; i++)
            {
                var schedule = _dynamicSchedules[i];
                if (schedule is null)
                    throw new InvalidOperationException(
                        $"Dynamic hyperparameter '{DynamicHyperparameterNames[i]}' has no schedule " +
                        "(built with HyperValue.Runtime); supply it explicitly via the " +
                        "TrainStep(checkpoint, hyperparams, …) overload / MakeHyperparams.");
                values[i] = schedule.At(step);
            }
            return PackHyperparams(values);
        }

        private TrainingStepResult RunStep(
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
            if (HyperparamStructDef.Fields.Length > 0 && hyperparams is null)
                throw new ArgumentNullException(nameof(hyperparams),
                    "This rig was built with dynamic hyperparameters; supply their values each step " +
                    "(see TrainingRig.MakeHyperparams).");

            // Execute the training step graph.
            // Graph inputs (after lowering): [param_fields..., state_fields..., opt_state_fields..., hyperparam_fields..., model_input_fields..., target_fields...]
            // CompiledGraph.Execute expands TensorDataStruct inputs into individual fields; an empty
            // struct contributes no fields. The hyperparams input slot exists only when the rig has
            // dynamic hyperparameters, so it is included only then.
            var execInputs = new List<IData>(6)
            {
                checkpoint.TrainableParams,
                checkpoint.ModelState,
                checkpoint.OptimizerState,
            };
            if (HyperparamStructDef.Fields.Length > 0) execInputs.Add(hyperparams!);
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

            var newCheckpoint = new TrainingCheckpoint(
                updatedParams,
                updatedModelState,
                updatedOptimizerState,
                checkpoint.Step + 1);

            return new TrainingStepResult(newCheckpoint, lossValue);
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
                    var result = TrainStep(checkpoint, trainingInputs[i], trainingOutputs[i], compiled);
                    checkpoint = result.Checkpoint;
                    epochLoss += result.Loss;
                }

                epochLosses[epoch] = epochLoss / trainingInputs.Length;
            }

            return new TrainingResult(checkpoint, epochLosses);
        }

        /// <summary>
        /// Fits the model to the data for <paramref name="numEpochs"/> epochs — a one-liner over
        /// <see cref="TrainStep(TrainingCheckpoint, TensorDataStruct, TensorDataStruct, CompiledGraph)"/>.
        /// Scheduled hyperparameters are applied automatically (the global step advances across epochs
        /// via the checkpoint), so the schedule sees a monotonically increasing step. Alias for
        /// <see cref="Train"/>. <paramref name="initialCheckpoint"/> defaults to
        /// <see cref="CreateDefaultCheckpoint"/> and <paramref name="ctx"/> defaults to
        /// <see cref="ComputeContext.Default"/>, so a minimal call is
        /// <c>rig.Fit(inputs, targets, numEpochs: 10)</c>.
        /// </summary>
        public TrainingResult Fit(
            TensorDataStruct[] trainingInputs,
            TensorDataStruct[] trainingOutputs,
            int numEpochs,
            TrainingCheckpoint? initialCheckpoint = null,
            ComputeContext? ctx = null)
            => Train(initialCheckpoint ?? CreateDefaultCheckpoint(), trainingInputs, trainingOutputs, numEpochs, ctx ?? ComputeContext.Default);

        /// <summary>
        /// Returns the default initial checkpoint produced at <see cref="FromScratch(FastComputationGraph, FastComputationGraph, FastComputationGraph, NamedModelParam[], IOptimizerHyperparameters, RngConfig?)"/> time.
        /// Trainable parameters and model state were initialized from the model's built-in
        /// initializers, and optimizer state from the optimizer's [StateInitializer]s (run once
        /// per trainable parameter). This is pure packaging — no computation happens here.
        /// </summary>
        public TrainingCheckpoint CreateDefaultCheckpoint()
        {
            return new TrainingCheckpoint(
                new TensorDataStruct(TrainableParamStructDef, _initialParamFields),
                new TensorDataStruct(ModelStateDef, _initialStateFields),
                new TensorDataStruct(OptimizerStateDef, _initialOptStateFields));
        }

        /// <summary>
        /// Loads a checkpoint previously written by <see cref="TrainingCheckpoint.Save(string)"/>,
        /// reconstructing it against this rig's parameter/state struct definitions so training resumes
        /// exactly where it left off: trainable params, optimizer moments, model state, and the global
        /// step are all restored (schedules resume from that step). Throws if the file's fields don't
        /// match this rig — e.g. a checkpoint produced by a different model or optimizer. The rig must
        /// be built from the same model/loss/optimizer graphs as the one that saved the checkpoint.
        /// </summary>
        public TrainingCheckpoint LoadCheckpoint(string filePath)
            => TrainingCheckpoint.Load(filePath, TrainableParamStructDef, ModelStateDef, OptimizerStateDef);

        /// <summary>
        /// Packs a single dynamic hyperparameter value into a <see cref="TensorDataStruct"/> for the
        /// explicit <see cref="TrainStep(TrainingCheckpoint, TensorDataStruct, TensorDataStruct, TensorDataStruct, CompiledGraph)"/>
        /// overload. Convenience for the common case of exactly one dynamic hyperparameter (e.g. the
        /// learning rate); throws if the rig has a different number. For multiple, use the named overload.
        /// </summary>
        public TensorDataStruct MakeHyperparams(float value)
        {
            if (HyperparamStructDef.Fields.Length != 1)
                throw new InvalidOperationException(
                    $"MakeHyperparams(float) requires exactly one dynamic hyperparameter; this rig has " +
                    $"{HyperparamStructDef.Fields.Length} ([{string.Join(", ", DynamicHyperparameterNames)}]). " +
                    "Use MakeHyperparams((name, value), …).");
            return PackHyperparams(new[] { value });
        }

        /// <summary>
        /// Packs named dynamic hyperparameter values into a <see cref="TensorDataStruct"/> for the
        /// explicit <see cref="TrainStep(TrainingCheckpoint, TensorDataStruct, TensorDataStruct, TensorDataStruct, CompiledGraph)"/>
        /// overload. Every dynamic hyperparameter must be named exactly once (case-insensitive); names
        /// are those in <see cref="DynamicHyperparameterNames"/>, e.g.
        /// <c>MakeHyperparams(("learningRate", lr), ("weightDecay", wd))</c>.
        /// </summary>
        public TensorDataStruct MakeHyperparams(params (string name, float value)[] values)
        {
            if (values is null) throw new ArgumentNullException(nameof(values));

            var byName = new Dictionary<string, float>(StringComparer.OrdinalIgnoreCase);
            foreach (var (name, value) in values)
            {
                if (name is null) throw new ArgumentException("Hyperparameter name cannot be null.", nameof(values));
                if (!byName.TryAdd(name, value))
                    throw new ArgumentException($"Hyperparameter '{name}' was supplied more than once.", nameof(values));
            }

            var ordered = new float[HyperparamStructDef.Fields.Length];
            for (int i = 0; i < HyperparamStructDef.Fields.Length; i++)
            {
                var fieldName = HyperparamStructDef.Fields[i].Name;
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

        /// <summary>Packs values (in <see cref="HyperparamStructDef"/> field order) into a scalar-field struct.</summary>
        private TensorDataStruct PackHyperparams(float[] orderedValues)
        {
            var fields = new KeyValuePair<string, IData>[orderedValues.Length];
            for (int i = 0; i < orderedValues.Length; i++)
                fields[i] = new KeyValuePair<string, IData>(
                    HyperparamStructDef.Fields[i].Name,
                    Shorokoo.Globals.TensorData(Array.Empty<long>(), orderedValues[i]));
            return new TensorDataStruct(HyperparamStructDef, fields);
        }

        /// <summary>
        /// Phase 2: read the concrete-architecture graph for initial trainable / state
        /// parameter values, run the optimizer's state initializers per trainable parameter,
        /// derive the target tensor shape by shape-inferring the concrete model, and run shape
        /// inference + <see cref="MemoryAwareGraphOptimizer"/> on the lowered training-step
        /// graph.
        /// </summary>
        private void InitializeAndOptimize(
            FastComputationGraph concreteArch,
            NamedModelParam[] sampleInputs,
            ComputeContext ctx,
            RngConfig? rngConfig = null)
        {
            // Step 1: walk concreteArch's TRAINABLE_PARAM nodes in linear order to capture
            // each one's (ModelId, isTrainable). The same linear order is what Phase 1's
            // FastReplaceTrainableParamsWithInputProcessor used to build the param /
            // state struct defs, so this ordering aligns Phase 2 values with Phase 1 fields.
            // FastInitializeModelParams runs the initializer functions and returns
            // ModelId → TensorData; reindex by our captured order for alignment.
            var trainableModelIds = new List<ModelId>();
            var stateModelIds = new List<ModelId>();
            foreach (var node in concreteArch.Nodes)
            {
                if (node.OpCode != InternalOpCodes.TRAINABLE_PARAM) continue;
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
            // parameter, binding the optimizer's inputs to (hyperparameter seed values, the
            // parameter's initial value, a zero gradient). This is the same mechanism that
            // initializes trainable params — the state-init graph carries the [StateInitializer]
            // functions split out of the optimizer graph by FastNormalizeOptimizerGraph.
            _initialOptStateFields = new Dictionary<string, IData>();
            if (OptimizerStateDef.Fields.Length > 0)
            {
                var stateInitGraph = _optimizerStateInitGraph
                    ?? throw new InvalidOperationException(
                        "Optimizer state fields exist but no state-init graph was produced.");
                var statesPerParam = OptimizerStateDef.Fields.Length / TrainableParamStructDef.Fields.Length;
                var hyperSeeds = _hyperparamSeedValues
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
                        _initialOptStateFields[OptimizerStateDef.Fields[paramIdx * statesPerParam + s].Name] =
                            stateValues[s];
                }
            }

            // Step 2: derive the target tensor's shape from the model's prediction. Reuse
            // the already-computed paramValuesById via FastApplyModelParamValues — this
            // rewrites TRAINABLE_PARAM → CONSTANT/MODEL_PARAM_DATA in place without a
            // second initializer-execution pass.
            var shapeInferencer = new ShapeInferenceInterpreter(ctx);
            var concreteModel = Shorokoo.Core.Nodes.Processors.Fast.FastApplyModelParamValues.Process(concreteArch, paramValuesById);
            var modelInputTensors = sampleInputs.Select(p => p.ToTensorData()).ToArray();
            var modelShapeInfo = shapeInferencer.Infer(concreteModel, modelInputTensors);
            var modelOutputInfo = modelShapeInfo.GetTensorInfo(concreteModel.Outputs[0])
                ?? throw new InvalidOperationException(
                    "Shape inference of concrete model graph failed to produce an output shape.");
            var targetShape = modelOutputInfo.Shape;
            var targetDType = modelOutputInfo.DType;

            // Step 3: Assemble inputs in TrainingStepPureGraph order.
            // Layout: [param_fields, state_fields, opt_state_fields, hyperparam_fields, model_input_fields, target_fields].
            // Current losses (L2, CE) use a single Tensor target, so target_field_count is 1.
            var graph = TrainingStepPureGraph;
            const int targetFieldCount = 1;
            var expectedModelInputFields =
                graph.Inputs.Count
                - TrainableParamStructDef.Fields.Length
                - ModelStateDef.Fields.Length
                - OptimizerStateDef.Fields.Length
                - HyperparamStructDef.Fields.Length
                - targetFieldCount;
            if (sampleInputs.Length != expectedModelInputFields)
                throw new ArgumentException(
                    $"Expected {expectedModelInputFields} sample inputs (one per model input field), " +
                    $"got {sampleInputs.Length}.",
                    nameof(sampleInputs));

            var allInputs = new TensorData[graph.Inputs.Count];
            var idx = 0;

            foreach (var f in TrainableParamStructDef.Fields)
                allInputs[idx++] = (TensorData)_initialParamFields[f.Name];
            foreach (var f in ModelStateDef.Fields)
                allInputs[idx++] = (TensorData)_initialStateFields[f.Name];
            foreach (var f in OptimizerStateDef.Fields)
                allInputs[idx++] = (TensorData)_initialOptStateFields[f.Name];

            // Dynamic-hyperparameter fields: seed shape inference / optimization with their
            // default (initial) scalar values. At run time these are supplied per step.
            foreach (var f in HyperparamStructDef.Fields)
                allInputs[idx++] = (TensorData)_initialHyperparamFields[f.Name];

            // Model-input fields: one per sample input (in the order the user provided them,
            // which matches the model graph's input order).
            foreach (var sample in sampleInputs)
                allInputs[idx++] = sample.ToTensorData();

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
            PreOptimizationGraph = graph;
            PreOptimizationEval = baselineEval;
            OptimizationResult = optResult;
            TrainingStepPureGraph = optResult.OptimizedGraph;
        }
    }
}
