using Shorokoo.Core.Graph;
using Shorokoo.Runtime;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Graph;
using Shorokoo.Core.AutoDiffCheckpointing;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Training;
using Shorokoo.Core.Nodes.Processors.Training;
using Shorokoo.Core.Utils;
using Shorokoo.Core.Interpreter;
using Shorokoo.Core.Backends;
using Shorokoo.Core.Interpreter.Helpers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;

namespace Shorokoo
{
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
    /// <see cref="Hyperparameter.Runtime()"/> hyperparameter is routed as a runtime "hyperparams" input (see
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
        ///
        /// <para>This is the graph the memory-aware pass has already rewritten, so what it
        /// duplicates it duplicates on purpose. Compiling it through the ordinary
        /// <see cref="ComputeContext.Compile(ComputationGraph)"/> runs ONNX Runtime's
        /// common-subexpression pass over it, which merges every recomputation back into the
        /// tensor it exists to free — the rig's own sessions therefore use
        /// <see cref="Shorokoo.Core.Backends.ShorokooGraphOptimization.TrainingStep"/>. A caller compiling this graph
        /// to observe what the rig runs needs that profile, which <see cref="ComputeContext"/> does
        /// not expose: build the model with <c>FastOnnxModelBuilder</c> and hand it to
        /// <c>DefaultBackend.Instance.CreateSession</c> with that level.</para>
        /// </summary>
        public ComputationGraph TrainingStepPureGraph { get; private set; } = null!;

        /// <summary>
        /// Construction-time working graph: built by <c>BuildTrainingStepPureGraph</c>,
        /// optimized by <c>InitializeAndOptimize</c>, then relinquished into the readonly
        /// <see cref="TrainingStepPureGraph"/> wrapper (and nulled — the wrapper owns it).
        /// </summary>
        private InternalComputationGraph? _trainingStepWorkGraph;

        /// <summary>
        /// Lazily-compiled, cached executables for <see cref="TrainingStepPureGraph"/>, compiled via
        /// <see cref="RuntimeContext"/> on the first <c>TrainStep</c> and reused by every subsequent
        /// step — so a manual <c>for (…) cp = rig.TrainStep(cp, in, out);</c> loop compiles nothing
        /// caller-side. Each rig instance owns its own, since its trainstep is distinct. This is a pure
        /// in-memory memo of the already-derived (and never-persisted) trainstep graph; it does not
        /// participate in the rig's observable value, so it leaves the rig's immutability intact.
        ///
        /// <para>Keyed by the <b>shapes actually fed</b> (every expanded graph input: params, state,
        /// optimizer state, hyperparameters, counters, model inputs, targets). A session is compiled
        /// with those concrete dimensions stamped on its graph inputs, which lets ONNX Runtime resolve
        /// every intermediate shape at session build and constant-fold the shape arithmetic
        /// (<c>Shape</c> and the broadcast-reduction chains autograd emits over it — most of a
        /// step's kernels, and outputs that pinned large activations alive) out of the executed graph.
        /// A step at another shape — a partial final batch from a <c>dropLast: false</c> loader, say —
        /// simply compiles its own entry; the trainstep graph itself is shape-generic, so nothing but
        /// the ORT session is specialized. Bounded: past
        /// <see cref="MaxShapeSpecializedTrainSteps"/> distinct shapes every further shape runs on the
        /// one shape-generic (symbolic-dims) session, so a run that never repeats a shape pays at most
        /// that many extra session builds and then behaves exactly as before.</para>
        /// </summary>
        private readonly Dictionary<string, CompiledGraph> _compiledTrainSteps = new();

        /// <summary>The shape-generic fallback trainstep session; see <see cref="_compiledTrainSteps"/>.</summary>
        private CompiledGraph? _compiledTrainStepGeneric;

        /// <summary>How many distinct input-shape signatures get their own shape-specialized session.</summary>
        internal const int MaxShapeSpecializedTrainSteps = 4;

        /// <summary>The input-shape signatures with a shape-specialized session so far (test hook).</summary>
        internal IReadOnlyCollection<string> CompiledTrainStepShapeKeys
        {
            get { lock (_compiledTrainSteps) return _compiledTrainSteps.Keys.ToArray(); }
        }

        /// <summary>Whether the shape-generic fallback session has been compiled (test hook).</summary>
        internal bool HasGenericTrainStepSession
        {
            get { lock (_compiledTrainSteps) return _compiledTrainStepGeneric is not null; }
        }

        /// <summary>The state pairs the compiled training step for <paramref name="shapeKey"/> was
        /// marked to write each updated field into the field it replaces, by position — the same
        /// position on both sides (test hook).</summary>
        internal IReadOnlyList<(int Output, int Input)> MarkedStatePairs(string shapeKey)
        {
            lock (_compiledTrainSteps) return _compiledTrainSteps[shapeKey].MarkedPairs();
        }

        /// <summary>The rig's own initial values, for a test to see that nothing it did fed them to
        /// a run (test hook).</summary>
        internal IEnumerable<TensorData> OwnInitialValues
        {
            get
            {
                Dictionary<string, IData>[] families = [InitialParamFields, InitialStateFields, InitialOptStateFields];
                return families.SelectMany(family => family.Values.OfType<TensorData>());
            }
        }

        /// <summary>
        /// The compiled trainstep for the given (struct-expanded, graph-input-ordered) inputs, compiled
        /// on first use for their shapes via <see cref="RuntimeContext"/>; see <see cref="_compiledTrainSteps"/>.
        /// </summary>
        private CompiledGraph CompiledTrainStepFor(IData[] expandedInputs)
        {
            var dims = new long[]?[expandedInputs.Length];
            for (int i = 0; i < expandedInputs.Length; i++)
                dims[i] = FedTensor(expandedInputs[i]) is TensorData t ? t.Shape.Dims : null;
            var key = string.Join(";", dims.Select(d => d is null ? "?" : string.Join(",", d)));

            lock (_compiledTrainSteps)
            {
                if (_compiledTrainSteps.TryGetValue(key, out var compiled)) return compiled;
                if (_compiledTrainSteps.Count < MaxShapeSpecializedTrainSteps)
                    return _compiledTrainSteps[key] = RuntimeContext.Compile(
                        TrainingStepPureGraph.ToInternal(), dims, trainingStep: true,
                        description: TrainStepDescription, aliasCandidates: StateAliasCandidates());
                // Reached only once more distinct shapes have been fed than there are specialized
                // slots, and shared by every shape after that -- so this session's sizes are known
                // not to settle, which is the one case the arena strategy departs on.
                return _compiledTrainStepGeneric ??= RuntimeContext.Compile(
                    TrainingStepPureGraph.ToInternal(), inputDims: null, trainingStep: true,
                    reusedAcrossShapes: true, description: TrainStepDescription,
                    aliasCandidates: StateAliasCandidates());
            }
        }

        /// <summary>
        /// The outputs of a training step that could be written into the memory of the inputs they
        /// replace: each updated parameter, model-state and optimizer-state field with the field it
        /// updates. The step's inputs and outputs lead with those fields in one order, so the pairs
        /// are positional. They are candidates, not promises: the compile keeps a pair only where the
        /// lowered step proves nothing reads the input after the output is written — an optimizer's
        /// element-wise update, typically, and never a weight a later node still reads — and the
        /// backend binds it only on a step that consumed that state.
        /// </summary>
        private (int Output, int Input)[] StateAliasCandidates()
        {
            var state = UpdatedParamFieldCount + UpdatedStateFieldCount + UpdatedOptimizerStateFieldCount;
            var pairs = new (int Output, int Input)[state];
            for (int i = 0; i < state; i++) pairs[i] = (i, i);
            return pairs;
        }

        /// <summary>What a message about a run of the training step calls it: its inputs are one per
        /// parameter, which names nothing a caller would recognise.</summary>
        private const string TrainStepDescription = "a TrainingRig's training step";

        /// <summary>How a message names a run of the training step: what it calls the run -- the
        /// description a tensor the run consumed names -- and, for an allocation the run could not
        /// make, the call the caller made and the step it took.</summary>
        private sealed record StepCall(string Description, string Operation, string Step);

        /// <summary>A run through <c>TrainStep</c>, over a loader or not.</summary>
        private static readonly StepCall TrainStepCall =
            new(TrainStepDescription, "TrainingRig.TrainStep", "the training step");

        /// <summary>A run of the same step taken by a <see cref="ResidentTrainingRun"/>, which the
        /// caller drove through the run rather than through <c>TrainStep</c>.</summary>
        private static readonly StepCall ResidentStepCall =
            new("a TrainingRig's resident run step", "ResidentTrainingRun.Step", "the resident run step");

        /// <summary>What a message calls each input of a step fed structs named as the rig's own
        /// definitions name them, worked out by the first step; see <see cref="StepLabels"/>. The
        /// definitions are fixed once the rig is built, and two steps racing to fill this build the
        /// same labels.</summary>
        private string[]? _stepLabels;

        /// <summary>The tensor a feed is, or wraps, or null for anything else.</summary>
        private static TensorData? FedTensor(IData feed)
            => feed is SharedInput shared ? shared.Value as TensorData : feed as TensorData;

        /// <summary>
        /// The rig's <b>constituent</b> layer: the swappable source-of-truth models — the
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
        /// the rig is first built: the model graph run through <c>ToConcreteArchitecture</c> at
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
        /// node carries the shape the model was concretized at, as dims-only
        /// <see cref="OnnxOpAttributeNames.ShrkAttrRepresentativeInputShape"/> (see
        /// <see cref="WriteRepresentativeInputs(InternalComputationGraph, long[][])"/>). The two
        /// training-graph shape-inference sites reconstruct their <c>sampleInputs[]</c> off that
        /// attribute plus the node's dtype (<see cref="ReadRepresentativeInputs"/>), so no separate
        /// sample-input field is stored on the rig. In the native <c>.srk</c> dialect a
        /// <c>MODEL_TENSOR_INPUT</c> serializes as a NodeProto, so the attribute round-trips on disk
        /// (making the saved arch self-describing); it also rides along on <c>Clone()</c>, so it
        /// survives re-seeding.</para>
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

        /// <summary>
        /// The compute context used for the rig's <b>build/merge phase</b>: concretizing the model and
        /// the hyperparameters, building scheduler modules, shape-inferring, lowering, memory-optimizing
        /// and initializing the training-step graph and the optimizer state. Which backend it merges on
        /// is its own — see <see cref="RuntimeContext"/>. Supplied at construction (defaults to <see cref="ComputeContext.Default"/>);
        /// every <c>With…</c> derivation carries it forward by reference. It is <b>runtime configuration,
        /// never persisted</b> — no checkpoint (flat or <c>.skpt</c>) or manifest records it, so a
        /// reloaded rig receives a fresh one via <see cref="FromScratch(ComputationGraph, ComputationGraph,
        /// ComputationGraph, NamedModelParam[], IOptimizerHyperparameters, RngConfig?, ComputeContext?, ComputeContext?, IProgress{BuildProgress}, TrainingBackend?)"/>.
        ///
        /// <para>The default is taken on the first READ, not at construction. Reading
        /// <see cref="ComputeContext.Default"/> resolves a backend, so a field
        /// initializer here made merely constructing a rig require one deployed — including when the
        /// caller supplied both contexts explicitly and the default was overwritten unread.</para>
        /// </summary>
        public ComputeContext MergeContext
        {
            get => _mergeContext ??= ComputeContext.Default;
            private set => _mergeContext = value;
        }

        private ComputeContext? _mergeContext;

        /// <summary>
        /// The compute context used to <b>compile the merged <see cref="TrainingStepPureGraph"/> into an
        /// executable and run it</b>: the lazily-cached trainstep sessions (see
        /// <see cref="_compiledTrainSteps"/>) that <see cref="Train"/>, every <c>Fit</c> overload and the
        /// manual <c>TrainStep</c> all share — the context whose session actually executes the training
        /// step. It is the rig's sole compile/run context — <c>Train</c>/<c>Fit</c> take no per-call
        /// context override, so every compiled graph of a rig comes from it. Supplied at construction
        /// (defaults to <see cref="ComputeContext.Default"/>); every <c>With…</c> derivation carries it
        /// forward by reference and, like <see cref="MergeContext"/>, it is runtime configuration that is
        /// <b>never persisted</b>.
        ///
        /// <para>Left unset, this context and <see cref="MergeContext"/> are both
        /// <see cref="ComputeContext.Default"/> and divide <i>phases</i> rather than hardware: which
        /// work is build/merge and which is compile/run. They may differ, though. A context carries
        /// the backend it runs on (see <see cref="ComputeContext"/>), so a rig <b>can</b> merge on one
        /// device and train on another, at the cost of a host copy per feed — both backends then have
        /// to be deployed and reachable from the one process. They may also differ in their device
        /// memory: this context's <see cref="ComputeContext.DeviceMemory"/> is the budget on what the
        /// training steps hold on the card and configures the arena of every training-step session,
        /// and its <see cref="ComputeContext.RunSettings"/> what each step's run does. Which device
        /// each will use is readable either way, off <see cref="ComputeContext.Backend"/>.</para>
        ///
        /// <para>Resolved on first read rather than at construction, for the reason given on
        /// <see cref="MergeContext"/>.</para>
        /// </summary>
        public ComputeContext RuntimeContext
        {
            get => _runtimeContext ??= ComputeContext.Default;
            private set => _runtimeContext = value;
        }

        private ComputeContext? _runtimeContext;

        /// <summary>
        /// Who computes the gradient of the training step (never null; defaults to
        /// <see cref="TrainingBackend.Shorokoo"/>, Shorokoo's own automatic differentiation).
        /// <see cref="TrainingBackend.Native"/> leaves it to the execution backend of
        /// <see cref="RuntimeContext"/>, which must accept <see cref="TrainingFormats.OnnxAutoGrad"/>;
        /// the rig refuses at build otherwise. Supplied at construction or through
        /// <see cref="WithTrainingBackend"/>; every <c>With…</c> derivation carries it forward. Like the
        /// two compute contexts it is <b>runtime configuration, never persisted</b>: a checkpoint
        /// written by a rig on either backend loads into a rig on either.
        /// </summary>
        public TrainingBackend TrainingBackend { get; private set; } = TrainingBackend.Shorokoo;

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

        /// <summary>
        /// Shape and dtype of every <see cref="TrainingStepPureGraph"/> input, in input order, as the
        /// shape inference behind <see cref="PreOptimizationEval"/> and <see cref="OptimizationResult"/>
        /// saw them: parameter / state / optimizer-state fields, hyperparameter and counter seeds, the
        /// representative model inputs, and the target at the shape and dtype the loss declares for
        /// it (<see cref="DeriveTargetExemplar"/>). Shared by the pre- and post-optimization graphs,
        /// so a diagnostic can synthesize a feed and run either against a real session on exactly the
        /// shapes the pass was judged on. Shapes only — the exemplars behind them may carry no values
        /// at all.
        ///
        /// <para>An input that is not a tensor — a model's <c>OptionalTensor</c> input, say — has no
        /// shape to report, so this view refuses such a rig rather than inventing one; read
        /// <see cref="OptimizationInputs"/>, which carries the exemplars themselves.</para>
        /// </summary>
        internal (Shape Shape, DType DType)[] OptimizationInputShapes =>
            [.. OptimizationInputs.Select(d => d is RuntimeTensor { Shape: { } shape } t
                ? (shape, t.DType)
                : throw new InvalidOperationException(
                    $"Training-step input of structure '{d.GetType().Name}' has no shape; read " +
                    $"{nameof(OptimizationInputs)} for the exemplars themselves."))];

        /// <summary>
        /// The exemplars behind <see cref="OptimizationInputShapes"/>, in input order: one per
        /// <see cref="TrainingStepPureGraph"/> input, as the shape inference behind
        /// <see cref="PreOptimizationEval"/> and <see cref="OptimizationResult"/> saw them.
        /// </summary>
        internal IRuntimeTensor[] OptimizationInputs { get; private set; } = [];

        /// <summary>Struct definition for model state (empty for stateless models). Internal
        /// build/persistence machinery — see <see cref="TrainableParamStructDef"/>.</summary>
        internal TensorStructDef ModelStateDef { get; private set; } = null!;

        /// <summary>Struct definition for optimizer state (empty for basic SGD). Internal
        /// build/persistence machinery — see <see cref="TrainableParamStructDef"/>.</summary>
        internal TensorStructDef OptimizerStateDef { get; private set; } = null!;

        /// <summary>
        /// The <see cref="OptimizerStateDef"/> field name of one (trainable parameter × state slot)
        /// instance. The single naming rule shared by the rig build (which generates the def,
        /// param-major: every slot of parameter 0, then of parameter 1, …) and .skpt persistence
        /// (which addresses each instance per tensor, issue #184) — change it in one place or not
        /// at all.
        /// </summary>
        internal static string OptimizerStateFieldName(string paramName, int slot)
            => $"{paramName}_opt_{slot}";

        /// <summary>
        /// Struct definition for the <b>schedule-less runtime</b> optimizer hyperparameters — the ones
        /// built with <see cref="Hyperparameter.Runtime()"/> that the caller supplies explicitly each step
        /// (one field each, at the hyperparameter's declared dtype and built shape). Empty when every hyperparameter is either baked as a
        /// constant or scheduled in-graph. Scheduled hyperparameters (a built-in <see cref="Schedule"/>
        /// or a scheduler module) are <b>not</b> here — they are computed in-graph from the step counter
        /// and need no per-step value. When non-empty, supply values via
        /// <see cref="TrainStep(TrainingCheckpoint, IData, IData, IData)"/>.
        /// Internal build machinery — build the per-step values with <see cref="MakeHyperparameters(float)"/>
        /// (which reads this def internally); inspect the dynamic names via <see cref="DynamicHyperparameterNames"/>.
        /// </summary>
        internal TensorStructDef HyperparameterStructDef { get; private set; } = null!;

        /// <summary>
        /// Struct definition for the model's runtime inputs — one field per model input tensor,
        /// in declaration order. Use <see cref="TensorStructDef.FromOrderedData(TensorData[])"/> to construct
        /// a <see cref="TensorDataStruct"/> for each training batch without building the definition
        /// manually: <c>rig.InputDef.FromOrderedData(TensorData([4L, 8L], myArray))</c>.
        /// </summary>
        public TensorStructDef InputDef { get; private set; } = null!;

        /// <summary>
        /// Struct definition for the loss function's target inputs — one field per non-prediction
        /// input of the loss graph, in declaration order. Use <see cref="TensorStructDef.FromOrderedData(TensorData[])"/>
        /// to construct target batches without building the definition manually:
        /// <c>rig.TargetDef.FromOrderedData(TensorData([4L, 8L], myTargets))</c>.
        ///
        /// <para><b>Empty when the loss never reads its target</b> (Shorokoo/Shorokoo#331). A model that
        /// computes its own loss takes a forwarding loss module whose body ignores the second input; the
        /// rig sees that the input is unreachable from the loss output and derives no target field, so
        /// there is nothing for a caller to construct. Use <see cref="HasTargets"/> to ask, and the
        /// target-free <see cref="TrainStep(TrainingCheckpoint, IData)"/> /
        /// <see cref="Fit(TensorDataStruct[], int, TrainingCheckpoint?)"/> to step such a rig.</para>
        /// </summary>
        public TensorStructDef TargetDef { get; private set; } = null!;

        /// <summary>
        /// Whether the loss reads a target at all — <c>false</c> for a rig whose loss body ignores its
        /// second input, whose <see cref="TargetDef"/> is therefore empty (Shorokoo/Shorokoo#331).
        /// </summary>
        public bool HasTargets => TargetDef.Fields.Length > 0;

        /// <summary>
        /// The value fed to the composed trainstep's target input when the loss ignores it — a
        /// zero-element tensor at the loss's declared target dtype and rank, built once at derivation.
        /// <c>null</c> when the loss does read its target, which is then supplied by the caller.
        ///
        /// <para>The input survives composition even though nothing reads it: the trainstep's input
        /// layout is positional, so the slot stays and the runtime still requires a value for it. What
        /// Shorokoo/Shorokoo#331 is about is who constructs that value — the rig knows the input is dead,
        /// so it holds the placeholder itself rather than making every call site pass one.</para>
        /// </summary>
        private TensorData? _ignoredTargetPlaceholder;

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
        /// The dtype each hyperparameter is <b>declared</b> at by the optimizer's
        /// <c>[Hyper(...)] Scalar&lt;T&gt;</c> signature, in the same order as
        /// <see cref="HyperparameterNames"/>. This is the single source of truth for the pipeline: a
        /// baked value is converted to it, a runtime field is typed by it, and a scheduler module must
        /// produce it. A hyperparameter is any supported scalar dtype, not just <c>float32</c>.
        /// </summary>
        public IReadOnlyList<DType> HyperparameterDTypes { get; private set; } = Array.Empty<DType>();

        /// <summary>
        /// The shape each hyperparameter was <b>built</b> at, in the same order as
        /// <see cref="HyperparameterNames"/>: empty for a scalar, else the dims fixed by whatever the
        /// hyperparameter is bound to — a baked constant's own shape, a scheduler graph's output shape,
        /// or the shape declared by <see cref="Hyperparameter.Runtime(long[])"/>. The training step is
        /// compiled once, so these are fixed for the rig's life; per-step values must match.
        /// </summary>
        public IReadOnlyList<Shape> HyperparameterShapes { get; private set; } = Array.Empty<Shape>();

        /// <summary>
        /// The names of the dynamic (runtime-input) hyperparameters, in <see cref="HyperparameterStructDef"/>
        /// field order — the names accepted by <see cref="MakeHyperparameters(ValueTuple{string, object}[])"/>.
        /// </summary>
        public IReadOnlyList<string> DynamicHyperparameterNames { get; private set; } = Array.Empty<string>();

        /// <summary>
        /// The int64 scalar counter inputs on the training-step graph, in input order — a subset of
        /// <see cref="CounterInputNames"/> (<c>step</c>, <c>epoch</c>, <c>batchIndex</c>), the union of
        /// what the rig's scheduled hyperparameters consume. Empty when no hyperparameter is
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

        /// <summary>
        /// Initial trainable parameter values — <b>empty</b> on a deferred build, which has none yet
        /// (Shorokoo/Shorokoo#327) and describes its fields through
        /// <see cref="DeferredInitialization.ParamSlots"/> instead. Read
        /// <see cref="InitialParamFields"/> to get values, which runs the initializers if a deferred
        /// build has not yet; read the field directly only where an empty one is the right answer.
        /// </summary>
        private Dictionary<string, IData> _initialParamFields = null!;

        /// <summary>Initial model state values, on the same deferred terms as <see cref="_initialParamFields"/>.</summary>
        private Dictionary<string, IData> _initialStateFields = null!;

        /// <summary>Initial optimizer state values, on the same deferred terms as <see cref="_initialParamFields"/>.</summary>
        private Dictionary<string, IData> _initialOptStateFields = null!;

        /// <summary>
        /// What a deferred build needs to run the initializers it skipped: the concrete arch, the
        /// compute context and RNG config the build would have used, and the parameter order the
        /// struct defs were built in. <c>null</c> once the values are in hand — on an eager build from
        /// the start, and on a deferred one from the moment something asks for a value
        /// (Shorokoo/Shorokoo#327).
        /// </summary>
        private DeferredInitialization? _deferredInit;

        /// <summary>
        /// Guards the one-time materialization in <see cref="EnsureInitialValues"/>. A rig is
        /// otherwise an immutable value that several threads may read at once — the coverage suite
        /// runs four in parallel — and deferral is the one piece of state that changes after
        /// construction, so it is the one piece that needs a lock.
        /// </summary>
        private readonly object _initialValuesGate = new();

        /// <summary>
        /// The initializer run a deferred build skipped, held until something actually wants the
        /// values it would have produced.
        /// </summary>
        private sealed record DeferredInitialization(
            InternalComputationGraph ConcreteArch,
            ComputeContext Context,
            RngConfig? RngConfig,
            ModelId[] TrainableModelIds,
            ModelId[] StateModelIds,
            IReadOnlyDictionary<string, TensorAttribute> ParamSlots,
            IReadOnlyDictionary<string, TensorAttribute> StateSlots);

        /// <summary>
        /// The rig's initial trainable-parameter <b>values</b>, running the deferred initializers on
        /// first use. Everything that hands initial values to a caller goes through here; everything
        /// that only needs their shape and dtype reads the field descriptions instead, and so never
        /// triggers the run (Shorokoo/Shorokoo#327).
        /// </summary>
        private Dictionary<string, IData> InitialParamFields
        {
            get { EnsureInitialValues(); return _initialParamFields; }
        }

        /// <summary>The materialized counterpart of <see cref="_initialStateFields"/>.</summary>
        private Dictionary<string, IData> InitialStateFields
        {
            get { EnsureInitialValues(); return _initialStateFields; }
        }

        /// <summary>The materialized counterpart of <see cref="_initialOptStateFields"/>.</summary>
        private Dictionary<string, IData> InitialOptStateFields
        {
            get { EnsureInitialValues(); return _initialOptStateFields; }
        }

        /// <summary>
        /// Runs the initializers a deferred build skipped — the model's, then the optimizer's
        /// state initializers over the resulting parameter values — and replaces the stand-ins with
        /// what they produce. A no-op on an eager build, and after the first call on a deferred one.
        ///
        /// <para>The stand-ins and the values agree on shape and dtype by construction (both come from
        /// the arch's own <c>MODEL_PARAM</c> declarations), so nothing <i>shape-driven</i> derived from
        /// the stand-ins — the struct defs, a checkpoint's shape check, the optimized trainstep — is
        /// invalidated by this running late. The build's shape inference does read small payloads, and
        /// a stand-in's are zeros, so a graph whose shapes depended on a trainable parameter's
        /// <b>value</b> could be inferred differently under deferral. That is the assumption the pass
        /// already makes of every model input, which it feeds zero exemplars; deferral extends it from
        /// inputs to parameters rather than introducing it.</para>
        /// </summary>
        private void EnsureInitialValues()
        {
            // Read once, with acquire semantics: a caller that sees null must also see the three
            // dictionaries the materializing thread published before clearing it.
            if (Volatile.Read(ref _deferredInit) is null) return;
            lock (_initialValuesGate)
            {
                if (_deferredInit is not { } deferred) return;

                var paramInfos = deferred.RngConfig is null
                    ? null
                    : deferred.ConcreteArch.GetConcreteModelParamInfos();
                var paramValuesById = Shorokoo.Core.Nodes.Processors.Fast.FastInitializeModelParams.Process(
                    deferred.ConcreteArch, deferred.Context, deferred.RngConfig, paramInfos);

                var paramFields = new Dictionary<string, IData>();
                for (var i = 0; i < TrainableParamStructDef.Fields.Length; i++)
                    paramFields[TrainableParamStructDef.Fields[i].Name] =
                        paramValuesById[deferred.TrainableModelIds[i]];

                var stateFields = new Dictionary<string, IData>();
                for (var i = 0; i < ModelStateDef.Fields.Length; i++)
                    stateFields[ModelStateDef.Fields[i].Name] =
                        paramValuesById[deferred.StateModelIds[i]];

                // The optimizer's state seeds were computed against the stand-ins, so they are
                // recomputed here against the real parameter values — an optimizer whose state
                // initializer reads a parameter's value (rather than only its shape) would otherwise
                // be seeded from zeros.
                var optStateFields = OptimizerStateDef.Fields.Length > 0
                    ? ComputeInitialOptStateFields(
                        ResolveStateInitHyperValues(null, throwOnMissingConsumed: false),
                        deferred.Context, NameValueOf(paramFields))
                    : _initialOptStateFields;

                // Everything is built into locals and published before the flag is cleared, because
                // the flag is what every other thread reads to decide the values are ready. Clearing
                // it first — with the dictionaries still empty — handed a concurrent caller a value
                // family with nothing in it.
                _initialParamFields = paramFields;
                _initialStateFields = stateFields;
                _initialOptStateFields = optStateFields;
                Volatile.Write(ref _deferredInit, null);
                // Publishing last means the flag is still set for the whole run above, so nothing this
                // method calls may read the materializing properties — it would re-enter, find the flag
                // set, and recurse (the lock is re-entrant and would not stop it). Nothing does today:
                // the initializer pass, ResolveStateInitHyperValues and ComputeInitialOptStateFields
                // all take what they need as arguments or read build-time state.
            }
        }

        /// <summary>
        /// Graph computing the optimizer's initial state values (one output per state field, inputs
        /// = the optimizer's [hyperparams..., currentParam, grad]); produced by
        /// <see cref="Shorokoo.Core.Nodes.Processors.Fast.FastNormalizeOptimizerGraph"/> from the
        /// optimizer's [StateInitializer] Init calls. Null for stateless optimizers.
        /// </summary>
        private InternalComputationGraph? _optimizerStateInitGraph;

        /// <summary>
        /// The value each hyperparameter contributes to optimizer state init, evaluated at the
        /// <b>initial counters</b> (step/epoch/batchIndex = 0) through the single value route:
        /// a baked hyper's constant, a scheduled hyper's canonical graph evaluated via QEE at build
        /// (built-in schedule <i>and</i> user module alike), and <c>null</c> for a runtime hyper
        /// (its value is host-supplied). Indexed in optimizer order. Replaces the old
        /// hardcoded-<c>0f</c> state-init seed that silently fed <c>0</c> for scheduler modules. Each
        /// value carries the hyperparameter's declared dtype and its built shape.
        /// </summary>
        private TensorData?[] _hyperparamInitialCounterValues = Array.Empty<TensorData?>();

        /// <summary>
        /// Optimizer-order indices of the hyperparameters the optimizer's state-init graph actually
        /// <b>consumes</b> (reachable from its outputs) — the dependency analysis. Empty for every
        /// built-in optimizer (their state inits are shape-only zeros/ones).
        /// </summary>
        private HashSet<int> _stateInitConsumedHyperIndices = new();

        /// <summary>Runtime-hyper optimizer-order index → its <see cref="HyperparameterStructDef"/> field name.</summary>
        private Dictionary<int, string> _runtimeHyperNameByOptIndex = new();

        /// <summary>
        /// True when the optimizer's state-init graph reads a <see cref="HyperparameterKind.Runtime"/>
        /// hyper: its initial value is unknowable at build, so <see cref="CreateInitialCheckpoint()"/>
        /// fails loud until <see cref="CreateInitialCheckpoint(TensorDataStruct)"/> supplies it.
        /// </summary>
        private bool _stateInitNeedsRuntimeHypers;

        /// <summary>The names of the runtime hyperparameters the state-init graph consumes (for the error).</summary>
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
        /// <see cref="Schedule"/> is applied per step; <see cref="Hyperparameter.Runtime()"/> is supplied manually.
        /// </param>
        /// <param name="rngConfig">
        /// Optional RNG configuration. Trainable parameters initialize from per-parameter keyed
        /// streams and the config is bound to the training-step graph (keying every runtime
        /// random feed, e.g. Dropout masks), making the whole run's randomness deterministic
        /// and reproducible from the config's master seed. When <c>null</c>,
        /// <see cref="RngConfig.Default"/> (master seed 0) is used — "no config" means the
        /// default deterministic identity, never non-reproducible backend randomness.
        /// </param>
        /// <param name="mergeContext">
        /// Optional build/merge-phase compute context (see <see cref="MergeContext"/>); <c>null</c> ⇒
        /// <see cref="ComputeContext.Default"/>. Never persisted — a reloaded rig gets a fresh one here.
        /// </param>
        /// <param name="runtimeContext">
        /// Optional compile/run compute context (see <see cref="RuntimeContext"/>); <c>null</c> ⇒
        /// <see cref="ComputeContext.Default"/>. Never persisted — a reloaded rig gets a fresh one here.
        /// </param>
        /// <param name="progress">
        /// Optional sink this build reports each stage to as it enters it (see
        /// <see cref="BuildProgress"/>). A build can run for minutes on a large graph; with a sink it is
        /// visibly alive rather than indistinguishable from a hang. Watches this call alone — every
        /// other build that wants watching, each <c>With…</c> derivation and <see cref="Load"/>
        /// included, takes a sink of its own.
        /// </param>
        /// <param name="trainingBackend">
        /// Optional: who computes the step's gradient (see <see cref="TrainingBackend"/>); <c>null</c> ⇒
        /// <see cref="TrainingBackend.Shorokoo"/>. Never persisted, like the compute contexts.
        /// </param>
        /// <returns>A configured TrainingRig ready for training</returns>
        public static TrainingRig FromScratch(
            ComputationGraph modelGraph,
            ComputationGraph lossGraph,
            ComputationGraph optimizerGraph,
            NamedModelParam[] sampleInputs,
            IOptimizerHyperparameters hyperparameters,
            RngConfig? rngConfig = null,
            ComputeContext? mergeContext = null,
            ComputeContext? runtimeContext = null,
            IProgress<BuildProgress>? progress = null,
            TrainingBackend? trainingBackend = null)
        {
            if (hyperparameters is null) throw new ArgumentNullException(nameof(hyperparameters));
            return FromScratchCore(modelGraph, lossGraph, optimizerGraph, sampleInputs,
                hyperparameters.InOptimizerOrder(), hyperparameters.HyperparameterNames, rngConfig,
                mergeContext, runtimeContext, progress, trainingBackend);
        }

        /// <summary>
        /// Lower-level overload that takes the hyperparameter values positionally (in the optimizer's
        /// declared order) rather than as a named set. Each <see cref="Hyperparameter"/>'s kind still
        /// decides baked-vs-runtime; a bare <c>float</c> implicitly converts to a baked constant, so
        /// <c>FromScratch(model, loss, opt, sample, 0.01f)</c> bakes a single learning rate. Generated
        /// graph fields fall back to <c>hyperparam_{i}</c> names since no names are supplied. A <c>params</c>
        /// array must come last, so this shape takes no <see cref="RngConfig"/>, compute context or
        /// progress sink; hand the values to the overload below as an array for those:
        /// <c>FromScratch(model, loss, opt, sample, [0.01f], rngConfig)</c>, or
        /// <c>…, [0.01f], progress: sink)</c>.
        /// </summary>
        public static TrainingRig FromScratch(
            ComputationGraph modelGraph,
            ComputationGraph lossGraph,
            ComputationGraph optimizerGraph,
            NamedModelParam[] sampleInputs,
            params Hyperparameter[] hyperparameters)
            => FromScratchCore(modelGraph, lossGraph, optimizerGraph, sampleInputs, hyperparameters,
                names: null, rngConfig: null, mergeContext: null, runtimeContext: null, progress: null,
                trainingBackend: null);

        /// <summary>
        /// Positional-hyperparameter overload with an RNG configuration and the optional build/merge and
        /// compile/run compute contexts (see <see cref="MergeContext"/> / <see cref="RuntimeContext"/>).
        /// The values are an explicit array in the slot the named set occupies, so the optional arguments
        /// follow in the same order as on the named-set overload rather than being pushed in front of a
        /// trailing <c>params</c> array; each defaults to its neutral value
        /// (<see cref="RngConfig.Default"/> / <see cref="ComputeContext.Default"/>, and no progress sink).
        /// Supplying any of them selects this overload — an array alone still binds the <c>params</c>
        /// overload above, to the same effect. Name an argument you reach past
        /// <paramref name="rngConfig"/> (<c>FromScratch(model, loss, opt, sample, [0.01f], progress: sink)</c>),
        /// and pass an empty array for an optimizer that takes no hyperparameters at all.
        /// </summary>
        public static TrainingRig FromScratch(
            ComputationGraph modelGraph,
            ComputationGraph lossGraph,
            ComputationGraph optimizerGraph,
            NamedModelParam[] sampleInputs,
            Hyperparameter[] hyperparameters,
            RngConfig? rngConfig = null,
            ComputeContext? mergeContext = null,
            ComputeContext? runtimeContext = null,
            IProgress<BuildProgress>? progress = null,
            TrainingBackend? trainingBackend = null)
            => FromScratchCore(modelGraph, lossGraph, optimizerGraph, sampleInputs, hyperparameters,
                names: null, rngConfig: rngConfig, mergeContext: mergeContext,
                runtimeContext: runtimeContext, progress: progress, trainingBackend: trainingBackend);

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
            RngConfig? rngConfig = null,
            ComputeContext? mergeContext = null,
            ComputeContext? runtimeContext = null,
            IProgress<BuildProgress>? progress = null,
            TrainingBackend? trainingBackend = null)
        {
            if (sampleInputs is null) throw new ArgumentNullException(nameof(sampleInputs));
            return FromScratch(modelGraph, lossGraph, optimizerGraph,
                sampleInputs.ModelParams.ToArray(), hyperparameters, rngConfig, mergeContext,
                runtimeContext, progress, trainingBackend);
        }

        /// <summary>
        /// Convenience overload that accepts a <see cref="ModelParamList"/> for sample inputs
        /// with positional hyperparameter values. As on the <see cref="NamedModelParam"/> pair, an
        /// <see cref="RngConfig"/>, a compute context or a progress sink means handing the values to the
        /// overload below as an array instead.
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
        /// <see cref="ModelParamList"/> convenience overload with an RNG configuration and the optional
        /// build/merge and compile/run compute contexts (see <see cref="MergeContext"/> /
        /// <see cref="RuntimeContext"/>) and the progress sink, which follow the hyperparameter array as they
        /// do on the named-set overload. Selected by supplying any of them — the same resolution as the
        /// <see cref="NamedModelParam"/> array overload above, and the same need to name an argument
        /// reached past <paramref name="rngConfig"/>.
        /// </summary>
        public static TrainingRig FromScratch(
            ComputationGraph modelGraph,
            ComputationGraph lossGraph,
            ComputationGraph optimizerGraph,
            ModelParamList sampleInputs,
            Hyperparameter[] hyperparameters,
            RngConfig? rngConfig = null,
            ComputeContext? mergeContext = null,
            ComputeContext? runtimeContext = null,
            IProgress<BuildProgress>? progress = null,
            TrainingBackend? trainingBackend = null)
        {
            if (sampleInputs is null) throw new ArgumentNullException(nameof(sampleInputs));
            return FromScratch(modelGraph, lossGraph, optimizerGraph,
                sampleInputs.ModelParams.ToArray(), hyperparameters, rngConfig, mergeContext,
                runtimeContext, progress, trainingBackend);
        }

        private static TrainingRig FromScratchCore(
            ComputationGraph modelGraph,
            ComputationGraph lossGraph,
            ComputationGraph optimizerGraph,
            NamedModelParam[] sampleInputs,
            Hyperparameter[] hyperparameters,
            IReadOnlyList<string>? names,
            RngConfig? rngConfig,
            ComputeContext? mergeContext,
            ComputeContext? runtimeContext,
            IProgress<BuildProgress>? progress,
            TrainingBackend? trainingBackend)
        {
            if (modelGraph is null) throw new ArgumentNullException(nameof(modelGraph));
            if (lossGraph is null) throw new ArgumentNullException(nameof(lossGraph));
            if (optimizerGraph is null) throw new ArgumentNullException(nameof(optimizerGraph));
            if (sampleInputs is null) throw new ArgumentNullException(nameof(sampleInputs));
            if (hyperparameters is null) throw new ArgumentNullException(nameof(hyperparameters));

            // Capture the constituents (the swappable source-of-truth layer) and take the
            // initial build path, which concretizes the model from the sample inputs. The sample
            // inputs are a construction-time argument only — consumed here to produce the retained
            // concrete arch and its shape exemplars, and never stored on the rig. "No config" means
            // the default deterministic identity. The two compute contexts are runtime config, held
            // directly on the rig (never inside the persisted constituents); "no context" ⇒ Default.
            return BuildInitialRig(
                new RigConstituents(
                    modelGraph, lossGraph, optimizerGraph, hyperparameters, names,
                    rngConfig ?? RngConfig.Default),
                sampleInputs,
                mergeContext ?? ComputeContext.Default,
                runtimeContext ?? ComputeContext.Default,
                trainingBackend ?? TrainingBackend.Shorokoo,
                progress);
        }

        /// <summary>
        /// The model-graph precondition shared by <see cref="FromScratch(ComputationGraph,
        /// ComputationGraph, ComputationGraph, NamedModelParam[], IOptimizerHyperparameters, RngConfig?, ComputeContext?, ComputeContext?, IProgress{BuildProgress}, TrainingBackend?)"/>
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
        /// Refuses a model that takes a sequence, which a rig cannot feed: every stage of a rig past
        /// concretization -- the representative inputs its shape inference is seeded with among them
        /// -- knows a model input only as a tensor or an optional. Refused here, before anything is
        /// built, rather than where the first of those stages fails, which it does in terms of its
        /// own internals.
        /// </summary>
        /// <exception cref="NotSupportedException">An input of <paramref name="model"/> is a
        /// sequence.</exception>
        private static void RequireNoSequenceInput(InternalComputationGraph model)
        {
            var producers = BuildProducerByOutputMap(model);
            for (int i = 0; i < model.Inputs.Count; i++)
            {
                if (!producers.TryGetValue(model.Inputs[i], out var node)
                    || node.OpCode != InternalOpCodes.MODEL_SEQUENCE_INPUT) continue;
                var name = i < model.InputUniqueNames.Count && !string.IsNullOrEmpty(model.InputUniqueNames[i])
                    ? $"'{model.InputUniqueNames[i]}' (#{i})"
                    : $"#{i}";
                throw new NotSupportedException(
                    $"A training rig feeds its model tensors and optional tensors, and the model's input {name} "
                    + "is a sequence. Train a model that takes the sequence's tensors as inputs of their own, "
                    + "or that builds the sequence from them itself.");
            }
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
        /// The <b>initial build path</b> (<see cref="FromScratch(ComputationGraph, ComputationGraph, ComputationGraph, NamedModelParam[], IOptimizerHyperparameters, RngConfig?, ComputeContext?, ComputeContext?, IProgress{BuildProgress}, TrainingBackend?)"/>
        /// only): concretizes the model from the <paramref name="sampleInputs"/> once, binds the RNG
        /// config, and derives the retained concrete arch + its shape exemplars — then hands off to
        /// <see cref="DeriveFromConcreteArch"/> to compose and optimize the trainstep. The sample
        /// inputs are consumed here (parameter shape resolution, liveness pruning, concretization value
        /// fallbacks) and NOT stored: everything the derivation path later needs is captured on the
        /// concrete arch itself (structure + RNG identity + the representative-input attributes written
        /// onto its <c>MODEL_TENSOR_INPUT</c> nodes below).
        /// </summary>
        private static TrainingRig BuildInitialRig(
            RigConstituents constituents,
            NamedModelParam[] sampleInputs,
            ComputeContext mergeContext,
            ComputeContext runtimeContext,
            TrainingBackend trainingBackend,
            IProgress<BuildProgress>? progress)
        {
            var c = constituents;
            ValidateConstituents(c);
            if (sampleInputs.Length == 0)
                throw new ArgumentException(
                    "A training rig requires at least one sample input. Sample inputs " +
                    "drive parameter shape resolution and training-graph shape inference.",
                    nameof(sampleInputs));

            // Concretization is a build/merge-phase step, so it runs on the merge context. The caller's
            // progress sink is independent of that: one reporter is built from it here and threaded
            // through the whole build so every phase reports against one clock — built first so the
            // thaw below, a full walk of the caller's graph, is inside that clock.
            var ctx = mergeContext;
            var reporter = BuildProgressReporter.For(progress);
            void Concretizing(string stage) => reporter?.Report(BuildPhase.Concretize, stage);

            // The model position takes a module graph or an already-lowered concrete architecture
            // (the concretization pipeline is idempotent on the latter); a weight-filled concrete
            // model is refused up front.
            Concretizing("Thaw");
            var model = RequireModelGraphKind(c.Model, "TrainingRig (model constituent)");
            RequireNoSequenceInput(model);

            // Single ToConcreteArchitecture pass — the ONE concretization for this rig and all its
            // future derivations. The resulting concrete arch is the shared substrate: the trainstep
            // build composes it with loss + autograd + optimizer; initialization reads its MODEL_PARAM
            // nodes for initial values and prediction shape; inference extraction binds weights into
            // it. The pass also runs the QEE-backed liveness filter that prunes trainable params whose
            // reachability is killed by the sample input shape. Sample input VALUES matter only here
            // (concretization's QEE/ORT resolution fallbacks); the derivation path needs only shapes.
            var concreteArch = model.ToConcreteArchitecture(
                new ModelParamList(sampleInputs), ctx, debugRequests: null, reporter);

            // Bind the RNG config at the shared concretization point: binding writes the
            // config's runtime identity into the RngSeed parameter, which — with the feeds'
            // key derivation chains — rides unchanged through loss composition and autograd
            // into the training-step graph, where the ONNX-prep lowering emits the keyed draws.
            // Reported: a config carrying runtime overrides re-runs the whole key-derivation wiring
            // and a reachability sweep here, which is graph-sized work, not bookkeeping.
            Concretizing("BindRngConfig");
            concreteArch.ApplyRngConfig(c.RngConfig);

            // Make the concrete arch self-describing: record the representative shape (dims only —
            // never the user's values) on each model-input node, so every re-derivation's shape
            // inference reconstructs its sampleInputs off the arch and no separate exemplar field
            // is stored. Done once here; the attribute rides along on Clone() and survives re-seeding.
            Concretizing("WriteRepresentativeInputs");
            WriteRepresentativeInputs(concreteArch, sampleInputs);

            return DeriveFromConcreteArch(
                c, concreteArch, mergeContext, runtimeContext, trainingBackend, reporter);
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
            InternalComputationGraph concreteArch,
            ComputeContext mergeContext,
            ComputeContext runtimeContext,
            TrainingBackend trainingBackend,
            BuildProgressReporter? progress,
            bool completesBuild = true,
            bool deferInitialization = false)
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

            // A step whose gradient is left to the execution backend is one only such a backend
            // runs, so a runtime context that cannot is refused before anything is composed. Asked
            // only off the default path, which never resolves the runtime context this early.
            if (!trainingBackend.LowersAutoGrad
                && !runtimeContext.ResolvedBackend.AcceptsTrainingFormat(trainingBackend.Format))
                throw new NotSupportedException(
                    $"The training backend {trainingBackend} leaves the gradient to the execution backend, "
                    + $"and the runtime context's backend {runtimeContext.Backend} does not run steps in "
                    + $"format '{trainingBackend.Format}'. Train on a context whose backend accepts it (a "
                    + "PyTorch backend, for one), or use TrainingBackend.Shorokoo, whose steps every "
                    + "backend runs.");

            // The two runtime contexts ride on the rig itself (never in the persisted constituents);
            // a derivation keeps the same two contexts. All the composition/lowering/optimization below
            // is build/merge-phase work and therefore runs on MergeContext.
            var rig = new TrainingRig
            {
                _constituents = c,
                _concreteArch = concreteArch,
                MergeContext = mergeContext,
                RuntimeContext = runtimeContext,
                TrainingBackend = trainingBackend,
            };
            // One thaw of the loss, read by both halves of the build: composition splices a clone of
            // it into the training graph and leaves it as it found it, and the initialization half
            // reads its target declaration back off it (see DeriveTargetExemplar).
            var lossGraph = c.Loss.ToInternal();
            rig.BuildTrainingStepPureGraph(
                concreteArch, lossGraph, c.Optimizer.ToInternal(), c.Hyperparameters, c.Names,
                progress);
            rig.InitializeAndOptimize(
                concreteArch, lossGraph, mergeContext, c.RngConfig, progress, deferInitialization);
            if (completesBuild) progress?.ReportComplete(BuildPhase.Initialize);
            return rig;
        }

        /// <summary>
        /// Writes a representative shape onto each of the concrete arch's <c>MODEL_TENSOR_INPUT</c> nodes
        /// (in graph-input order, one per <paramref name="sampleInputs"/>), making the arch self-describing
        /// for training-graph shape inference. Never records the user's sample values. Only concretization
        /// (already done) needed input values; from here on only shapes matter, so this records exactly
        /// the shape.
        /// </summary>
        private static void WriteRepresentativeInputs(InternalComputationGraph concreteArch, NamedModelParam[] sampleInputs)
            => WriteRepresentativeInputs(concreteArch, sampleInputs.Select(RepresentativeShapeOf).ToArray());

        /// <summary>
        /// The single negative dim recorded for an optional input the rig was built with ABSENT. A
        /// concretized shape never holds one, so it cannot be confused with a real shape — and
        /// recording it rather than recording nothing keeps a MISSING attribute meaning what it
        /// means on a tensor input: an arch that was not built self-describing, which
        /// <see cref="ReadRepresentativeInputs"/> refuses rather than reading as some default.
        /// </summary>
        private static readonly long[] AbsentOptionalShape = [-1L];

        /// <summary>
        /// The dims to record for one sample, or <c>null</c> when the sample's input node carries no
        /// representative-shape attribute to record them on. An absent optional records
        /// <see cref="AbsentOptionalShape"/>, not nothing. Reading the shape off the sample rather
        /// than converting the sample to a tensor is what lets an absent optional through at all: it
        /// has no tensor value, and asking for one threw before the shape was ever needed
        /// (Shorokoo/Shorokoo#314).
        /// </summary>
        private static long[]? RepresentativeShapeOf(NamedModelParam sample) => sample switch
        {
            OptionalTensorDataModelParam optional =>
                optional.Data is { HasValue: true, Value: { } value } ? value.Shape.Dims : AbsentOptionalShape,
            { Structure: DataStructure.Tensor } => sample.ToTensorData().Shape.Dims,
            _ => null,
        };

        /// <summary>
        /// Dims-shaped counterpart of
        /// <see cref="WriteRepresentativeInputs(InternalComputationGraph, NamedModelParam[])"/>:
        /// records <see cref="OnnxOpAttributeNames.ShrkAttrRepresentativeInputShape"/> — the dims,
        /// whatever the input's size — on each input node that carries the attribute (the node's own
        /// dtype attribute completes the pair), and records nothing for a <c>null</c> entry. An
        /// optional input records <see cref="AbsentOptionalShape"/> when the optional was supplied
        /// absent, which is how <see cref="ReadRepresentativeInputs"/> reads its presence back — a
        /// value, so that a MISSING attribute stays the loud error it already is on a tensor input.
        /// The concrete arch's input ops serialize as NodeProtos in the native <c>.srk</c>
        /// dialect, so the attribute round-trips on disk verbatim and the saved arch is
        /// self-describing — no separate manifest input-shape field is needed.
        /// </summary>
        private static void WriteRepresentativeInputs(InternalComputationGraph concreteArch, long[]?[] inputShapes)
        {
            if (concreteArch.Inputs.Count != inputShapes.Length)
                throw new InvalidOperationException(
                    $"Concrete arch has {concreteArch.Inputs.Count} input(s) but {inputShapes.Length} " +
                    "input shape(s) were supplied; they must correspond one-to-one in declaration order.");
            var producerByOutput = BuildProducerByOutputMap(concreteArch);
            for (int i = 0; i < concreteArch.Inputs.Count; i++)
            {
                if (!producerByOutput.TryGetValue(concreteArch.Inputs[i], out var node))
                    throw new InvalidOperationException(
                        $"Concrete arch input {concreteArch.Inputs[i]} has no producing node.");
                if (node.OpCode is not (InternalOpCodes.MODEL_TENSOR_INPUT or InternalOpCodes.MODEL_OPTIONAL_INPUT))
                    continue;
                if (inputShapes[i] is not { } dims) continue;
                node.Attributes = node.Attributes.SetAttributes(
                    (OnnxOpAttributeNames.ShrkAttrRepresentativeInputShape, (object?)dims));
            }
        }

        /// <summary>
        /// Reconstructs the <c>sampleInputs[]</c> array for shape inference off the concrete arch's
        /// representative-input attributes (in graph-input order) — the derivation-path counterpart of
        /// <see cref="WriteRepresentativeInputs(InternalComputationGraph, NamedModelParam[])"/>. Reads
        /// each node's <see cref="OnnxOpAttributeNames.ShrkAttrRepresentativeInputShape"/> dims plus its
        /// dtype and describes each input via <see cref="RepresentativeInputFor"/> (real zeros while the
        /// values are small enough to be read, shape and dtype alone above that, so no large buffer is
        /// materialized); a node without the attribute fails loud (the arch was not built
        /// self-describing). Each description becomes the runtime tensor
        /// <see cref="ShapeInferenceInterpreter"/> is fed directly — a values-less one simply arrives
        /// with its data null, which is what a shape-driven pass wants anyway.
        /// </summary>
        private static IRuntimeTensor[] ReadRepresentativeInputs(InternalComputationGraph concreteArch)
        {
            var producerByOutput = BuildProducerByOutputMap(concreteArch);
            var inputs = new IRuntimeTensor[concreteArch.Inputs.Count];
            for (int i = 0; i < concreteArch.Inputs.Count; i++)
            {
                if (!producerByOutput.TryGetValue(concreteArch.Inputs[i], out var node)
                    || node.OpCode is not (InternalOpCodes.MODEL_TENSOR_INPUT or InternalOpCodes.MODEL_OPTIONAL_INPUT))
                    throw new InvalidOperationException(
                        $"Concrete arch input {concreteArch.Inputs[i]} is not a MODEL_TENSOR_INPUT or " +
                        "MODEL_OPTIONAL_INPUT node; cannot read its representative input.");

                var dtype = node.Attributes.GetDTypeVal(OnnxOpAttributeNames.AttrDtype)
                    ?? throw new InvalidOperationException(
                        "Concrete arch input node records a representative-input shape but no dtype; " +
                        "cannot re-materialize its representative input.");
                var dims = node.Attributes.GetLongsVal(OnnxOpAttributeNames.ShrkAttrRepresentativeInputShape)
                    ?? throw new InvalidOperationException(
                        "Concrete arch input node carries no representative-input shape attribute: the rig " +
                        "was not built through BuildInitialRig (which records one on every model input), or " +
                        "the arch was saved by an older Shorokoo that recorded small inputs as an inline " +
                        "representative tensor instead of dims (there is no legacy read path; rebuild the " +
                        "rig from its source graphs and re-save).");

                inputs[i] = node.OpCode == InternalOpCodes.MODEL_OPTIONAL_INPUT
                    ? (dims.AsSpan().SequenceEqual(AbsentOptionalShape)
                        ? new RuntimeOptionalTensor { DType = dtype, HasValue = false }
                        : new RuntimeOptionalTensor
                        {
                            DType = dtype,
                            HasValue = true,
                            ValueTensor = RepresentativeRuntimeInputFor(new Shape(dims), dtype),
                        })
                    : RepresentativeRuntimeInputFor(new Shape(dims), dtype);
            }
            return inputs;
        }

        /// <summary>
        /// What a shape-driven pass is handed in place of a value it will not read: shape and dtype
        /// always, plus a real zero payload while the element count is within the consuming engine's
        /// small-tensor threshold
        /// (<see cref="Shorokoo.Core.AutoDiffCheckpointing.ShapeInferenceInterpreter.MaxSmallTensorElements"/>),
        /// above which the description carries no elements at all and so costs no allocation. Holds no
        /// sample-input values either way — the zeros are zeros.
        ///
        /// <para>The threshold decides only how faithful the description is, never whether it is legal:
        /// a values-elided attribute is readable as "shape and dtype, no values" at any threshold, so
        /// this one and the engines' read thresholds no longer have to agree.</para>
        /// </summary>
        internal static TensorAttribute RepresentativeInputFor(Shape shape, DType dtype)
        {
            if (shape.Count > Shorokoo.Core.AutoDiffCheckpointing.ShapeInferenceInterpreter.MaxSmallTensorElements)
                return TensorAttribute.WithoutValues(shape, dtype);
            var bytesPerElement = dtype.EncodingBitCount / 8;
            return TensorAttribute.OverBytes(shape, dtype, new byte[shape.Count * bytesPerElement]);
        }

        /// <summary>
        /// <see cref="RepresentativeInputFor"/> as the shape-inference engines take it: a runtime
        /// tensor stating the shape and dtype, carrying the zeros where there are any and leaving its
        /// data null where there are not.
        /// </summary>
        internal static RuntimeTensor RepresentativeRuntimeInputFor(Shape shape, DType dtype)
            => TensorDataConverter.ToRuntimeTensor(
                RepresentativeInputFor(shape, dtype),
                Shorokoo.Core.AutoDiffCheckpointing.ShapeInferenceInterpreter.MaxSmallTensorElements);

        /// <summary>
        /// The zero-element stand-in for a target input the loss never reads (Shorokoo/Shorokoo#331),
        /// at the loss's declared target dtype and — where it declares one — rank, so the dead slot is
        /// fed something of the shape the graph expects to see rather than merely something typed.
        /// Nothing reads its contents, so it carries no elements and costs no allocation worth naming.
        /// It is fed to a real run, not to a shape pass, so it is an actual tensor.
        /// </summary>
        private static TensorData IgnoredTargetPlaceholder(DType dtype, int? rank)
        {
            long[] dims = new long[rank is int r && r > 1 ? r : 1];
            for (var i = 1; i < dims.Length; i++) dims[i] = 1L;
            var shape = new Shape(dims);
            return TensorData.CreateFromRawBytes(
                shape, dtype, new byte[shape.Count * (dtype.EncodingBitCount / 8)]);
        }

        /// <summary>
        /// The shape and dtype the composed step's target input is seeded with for shape inference
        /// and the memory-aware pass — the <b>loss's own target</b>, rather than the model's
        /// prediction standing in for it.
        ///
        /// <para>The two coincide for a distance loss: L2, L1, Huber, BCE and the rest score a
        /// prediction against a target of exactly its shape and dtype, so the prediction is the right
        /// answer and is what this falls back to wherever the loss declares nothing more specific.
        /// They do not coincide for a class-index loss. ONNX's <c>SoftmaxCrossEntropyLoss</c> and
        /// <c>NegativeLogLikelihoodLoss</c> score <c>[N, C, d…]</c> float scores against
        /// <c>[N, d…]</c> int64 class indices, so a prediction standing in for that target makes the
        /// one-hot the gradient builds <c>[N, C, C]</c> where it should be <c>[N, C]</c> — a factor of
        /// C too large. At ten classes that is a rounding error, which is why every cross-entropy rig
        /// in the suite was judged on the wrong target without anyone noticing; at a language model's
        /// 50,257 it is ten billion elements for a batch of four, and the build dies inferring shapes
        /// over it.</para>
        ///
        /// <para>The scores are the prediction only where the loss hands it straight over, so they
        /// are inferred rather than assumed: a loss that folds <c>[N, T, C]</c> into <c>[N*T, C]</c>
        /// before scoring takes a target of <c>[N*T]</c>, which no dimension of the prediction names.
        /// That inference is seeded with the prediction-shaped target this derivation exists to
        /// replace, which is the pair the whole composed graph was inferred with before, so it can
        /// choke on nothing the build did not already choke on.</para>
        /// </summary>
        private static (Shape Shape, DType DType) DeriveTargetExemplar(
            InternalComputationGraph lossGraph,
            ShapeInferenceInterpreter shapeInferencer,
            Shape predictionShape,
            DType predictionDType)
        {
            var targetKey = lossGraph.Inputs[1];
            var declared = BuildProducerByOutputMap(lossGraph).TryGetValue(targetKey, out var producer)
                ? producer.Attributes.GetDTypeVal(OnnxOpAttributeNames.AttrDtype)
                : null;
            // An unspecialized dtype names no width, so it is nothing to build an exemplar out of. A
            // generic loss does not reach a rig today — its type-placeholder slot makes a third graph
            // input, which the two-input requirement refuses — so this is the declaration being
            // missing rather than a case with an answer of its own.
            var dtype = declared is { IsGenericType: false } d ? d : predictionDType;

            if (ClassIndexScoresOf(lossGraph, targetKey) is not { } scoresKey)
                return (predictionShape, dtype);

            var lossShapes = shapeInferencer.Infer(lossGraph, [scoresKey],
                RepresentativeRuntimeInputFor(predictionShape, predictionDType),
                RepresentativeRuntimeInputFor(predictionShape, dtype));
            if (lossShapes.GetTensorInfo(scoresKey) is not { Shape.Dims.Length: >= 2 } scores)
                return (predictionShape, dtype);

            // Scores [N, C, d…] against indices [N, d…]: the class axis is axis 1, and dropping it is
            // the whole of the difference between the two shapes.
            return (new Shape([.. scores.Shape.Dims[..1], .. scores.Shape.Dims[2..]]), dtype);
        }

        /// <summary>
        /// The scores input of the class-index loss op that reads <paramref name="targetKey"/> as its
        /// class indices, or <c>null</c> where none does. ONNX names exactly two such ops and both
        /// take the indices second. Every other way a loss reads a target broadcasts it against the
        /// prediction, where the prediction's own shape is the answer and always was.
        ///
        /// <para>The target is followed through the shape and type ops a loss puts between its own
        /// input and the labels slot — flattening <c>[N, T]</c> to <c>[N·T]</c> is the ordinary way
        /// to write a sequence model's loss, and casting an integer target is the ordinary way to
        /// write one whose caller feeds int32. Matching only a direct edge would fall back to the
        /// prediction's shape for those, which is the one-hot blow-up this derivation exists to
        /// avoid, and it would do it silently.</para>
        /// </summary>
        private static FastTensorKey? ClassIndexScoresOf(
            InternalComputationGraph lossGraph, FastTensorKey targetKey)
        {
            var reaches = TargetReaches(lossGraph, targetKey);
            foreach (var node in lossGraph.Nodes)
            {
                if (node.OpCode is not (OpCodes.SOFTMAX_CROSS_ENTROPY_LOSS
                                        or OpCodes.NEGATIVE_LOG_LIKELIHOOD_LOSS)) continue;
                var inputs = node.Inputs;
                if (inputs.Count < 2) continue;
                if (inputs[1] is not { IsEmpty: false } labels || !reaches.Contains(labels)) continue;
                if (inputs[0] is { IsEmpty: false } scores) return scores;
            }
            return null;
        }

        /// <summary>
        /// <paramref name="targetKey"/> and everything a chain of element-preserving shape or type
        /// ops turns it into. Only the data edge is followed — <c>Reshape</c>'s second input is a
        /// shape, not a target — and only ops that carry every element through, so the class axis
        /// cannot have been introduced or removed along the way.
        /// </summary>
        private static HashSet<FastTensorKey> TargetReaches(
            InternalComputationGraph lossGraph, FastTensorKey targetKey)
        {
            var reaches = new HashSet<FastTensorKey> { targetKey };
            // A node's inputs are produced before it, so one pass in graph order closes the chain.
            foreach (var node in lossGraph.Nodes)
            {
                if (node.OpCode is not (OpCodes.CAST or OpCodes.RESHAPE or OpCodes.SQUEEZE
                                        or OpCodes.UNSQUEEZE or OpCodes.FLATTEN)) continue;
                if (node.Inputs.Count == 0) continue;
                if (node.Inputs[0] is not { IsEmpty: false } from || !reaches.Contains(from)) continue;
                foreach (var output in node.Outputs)
                    if (output is { IsEmpty: false } produced) reaches.Add(produced);
            }
            return reaches;
        }

        // ───────────────────── Two-layer rig: immutable derivations ─────────────────────
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
        public TrainingRig WithLoss(ComputationGraph loss, IProgress<BuildProgress>? progress = null)
        {
            if (loss is null) throw new ArgumentNullException(nameof(loss));
            return DeriveFromConcreteArch(
                _constituents with { Loss = loss }, _concreteArch, MergeContext, RuntimeContext,
                TrainingBackend, BuildProgressReporter.For(progress));
        }

        /// <summary>
        /// A new rig with the optimizer constituent (and its hyperparameters) replaced; optimizer
        /// state is re-initialized as part of re-deriving the <c>trainstep</c>, everything else shared.
        /// </summary>
        public TrainingRig WithOptimizer(
            ComputationGraph optimizer,
            IOptimizerHyperparameters hyperparameters,
            IProgress<BuildProgress>? progress = null)
        {
            if (optimizer is null) throw new ArgumentNullException(nameof(optimizer));
            if (hyperparameters is null) throw new ArgumentNullException(nameof(hyperparameters));
            return DeriveFromConcreteArch(_constituents with
            {
                Optimizer = optimizer,
                Hyperparameters = hyperparameters.InOptimizerOrder(),
                Names = hyperparameters.HyperparameterNames,
            }, _concreteArch, MergeContext, RuntimeContext, TrainingBackend,
                BuildProgressReporter.For(progress));
        }

        /// <summary>Positional-hyperparameter overload of <see cref="WithOptimizer(ComputationGraph, IOptimizerHyperparameters, IProgress{BuildProgress})"/>.
        /// A <c>params</c> array must come last, so pass the values as an array to reach the sink —
        /// the same shape <c>FromScratch</c> takes.</summary>
        public TrainingRig WithOptimizer(ComputationGraph optimizer, params Hyperparameter[] hyperparameters)
            => WithOptimizer(optimizer, hyperparameters, progress: null);

        /// <summary>Array-form counterpart of
        /// <see cref="WithOptimizer(ComputationGraph, Hyperparameter[])"/> that can also take a
        /// progress sink; an array alone still binds the <c>params</c> overload, to the same effect.</summary>
        public TrainingRig WithOptimizer(
            ComputationGraph optimizer,
            Hyperparameter[] hyperparameters,
            IProgress<BuildProgress>? progress = null)
        {
            if (optimizer is null) throw new ArgumentNullException(nameof(optimizer));
            if (hyperparameters is null) throw new ArgumentNullException(nameof(hyperparameters));
            return DeriveFromConcreteArch(
                _constituents with { Optimizer = optimizer, Hyperparameters = hyperparameters, Names = null },
                _concreteArch, MergeContext, RuntimeContext, TrainingBackend,
                BuildProgressReporter.For(progress));
        }

        /// <summary>
        /// A new rig with the scheduler swapped, keeping the optimizer graph — the schedule rides in
        /// the hyperparameters (a baked constant, a built-in <see cref="Schedule"/>, or a scheduler
        /// module per field) until #106 folds the scheduler into its own persisted constituent. Only
        /// the <c>trainstep</c> is re-derived; the model, loss and optimizer graphs are shared.
        /// </summary>
        public TrainingRig WithScheduler(
            IOptimizerHyperparameters hyperparameters, IProgress<BuildProgress>? progress = null)
        {
            if (hyperparameters is null) throw new ArgumentNullException(nameof(hyperparameters));
            return DeriveFromConcreteArch(_constituents with
            {
                Hyperparameters = hyperparameters.InOptimizerOrder(),
                Names = hyperparameters.HyperparameterNames,
            }, _concreteArch, MergeContext, RuntimeContext, TrainingBackend,
                BuildProgressReporter.For(progress));
        }

        /// <summary>Positional-hyperparameter overload of <see cref="WithScheduler(IOptimizerHyperparameters, IProgress{BuildProgress})"/>.
        /// A <c>params</c> array must come last, so pass the values as an array to reach the sink —
        /// the same shape <c>FromScratch</c> takes.</summary>
        public TrainingRig WithScheduler(params Hyperparameter[] hyperparameters)
            => WithScheduler(hyperparameters, progress: null);

        /// <summary>Array-form counterpart of <see cref="WithScheduler(Hyperparameter[])"/> that can
        /// also take a progress sink; an array alone still binds the <c>params</c> overload, to the
        /// same effect.</summary>
        public TrainingRig WithScheduler(
            Hyperparameter[] hyperparameters, IProgress<BuildProgress>? progress = null)
        {
            if (hyperparameters is null) throw new ArgumentNullException(nameof(hyperparameters));
            return DeriveFromConcreteArch(
                _constituents with { Hyperparameters = hyperparameters, Names = null },
                _concreteArch, MergeContext, RuntimeContext, TrainingBackend,
                BuildProgressReporter.For(progress));
        }

        /// <summary>
        /// A new rig re-seeded with <paramref name="rngConfig"/> — the model's (and other drawing
        /// constituents') RNG identity is re-initialized, everything else shared by reference. Unlike
        /// the other derivations, WithSeed does change the concrete arch: the RNG identity is bound into
        /// its RngSeed parameter, so re-seeding rebinds it. That binding mutates, so this rig's retained
        /// arch is left untouched — a <b>clone</b> is re-keyed and the new rig derives from that. It is
        /// still only a rebind, not a re-concretization: the model constituent's structure is unchanged,
        /// so no <c>ToConcreteArchitecture</c> and no sample inputs are needed; re-initialization then
        /// re-draws every trainable parameter on the new seed's keyed streams. The
        /// cheaper path (share even the trainstep, re-derive only the compiled session, since the seed
        /// rides as an aliased param value) rests on the #22 param-identity substrate; until that lands
        /// the re-seed re-derives the trainstep, which is correct and equally immutable.
        /// </summary>
        public TrainingRig WithSeed(RngConfig rngConfig, IProgress<BuildProgress>? progress = null)
        {
            if (rngConfig is null) throw new ArgumentNullException(nameof(rngConfig));
            // Rebind the new RNG identity on a clone (ApplyRngConfig mutates), keeping this rig's
            // retained arch pristine. Clone() copies node attributes by reference, so the clone's
            // MODEL_TENSOR_INPUT nodes still carry the representative-input attributes (same model inputs).
            var reporter = BuildProgressReporter.For(progress);
            reporter?.Report(BuildPhase.Concretize, "CloneArchitecture");
            var reArch = _concreteArch.Clone();
            reporter?.Report(BuildPhase.Concretize, "BindRngConfig");
            reArch.ApplyRngConfig(rngConfig);
            return DeriveFromConcreteArch(
                _constituents with { RngConfig = rngConfig }, reArch, MergeContext, RuntimeContext, TrainingBackend,
                reporter);
        }

        /// <summary>
        /// A new rig whose training step's gradient is computed by <paramref name="backend"/> —
        /// <see cref="TrainingBackend.Shorokoo"/>, or <see cref="TrainingBackend.Native"/> to leave it
        /// to the execution backend of <see cref="RuntimeContext"/>. The model, loss, optimizer,
        /// hyperparameters, RNG config and both compute contexts are shared; only the
        /// <c>trainstep</c> is re-derived. A checkpoint of this rig resumes on the new one: the
        /// parameters, model state and optimizer state are laid out identically on both backends.
        /// </summary>
        /// <exception cref="NotSupportedException"><paramref name="backend"/> leaves the gradient to
        /// the execution backend, and <see cref="RuntimeContext"/>'s backend does not accept its
        /// <see cref="TrainingBackend.Format"/>.</exception>
        public TrainingRig WithTrainingBackend(TrainingBackend backend, IProgress<BuildProgress>? progress = null)
        {
            if (backend is null) throw new ArgumentNullException(nameof(backend));
            return DeriveFromConcreteArch(
                _constituents, _concreteArch, MergeContext, RuntimeContext, backend,
                BuildProgressReporter.For(progress));
        }

        /// <summary>
        /// Extracts the inference model for a checkpoint — a <b>pure read off the model constituent's
        /// mapping</b>: bind the checkpoint's model-owned params (trainable weights + module
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
            // Read through the definitions, not the field dictionary: a definition field is what the
            // model expects to be given, and a struct holds every one of them as the kind it declares.
            // Filtering the dictionary for tensors instead used to drop a field that was not one and
            // leave the bind to fail on a lookup for a parameter nothing had supplied. Both defs
            // declare only tensors and a struct is held to its definition, so the refusal below is
            // unreachable today — it is here so that stops being true loudly rather than silently.
            static IEnumerable<KeyValuePair<string, TensorData>> Declared(TensorDataStruct s) =>
                s.Definition.Fields.Select(f => s.Fields[f.Name] is TensorData t
                    ? new KeyValuePair<string, TensorData>(f.Name, t)
                    : throw new NotSupportedException(
                        $"Field '{f.Name}' is a {s.Fields[f.Name].GetType().Name}; only plain tensors can "
                        + "be bound into a model as weights."));
            var weights = new ModelParamList(
                Declared(checkpoint.TrainableParams).Concat(Declared(checkpoint.ModelState)),
                ModelParamType.TrainableParam);
            return _concreteArch.ToConcreteModel(weights, _concreteArch.GetShorokooIdNamingScheme());
        }

        /// <summary>
        /// Returns a NEW checkpoint carrying <paramref name="checkpoint"/>'s values, counters and loss,
        /// with its <see cref="TrainingCheckpoint.Rig"/> set to this rig, so
        /// <see cref="TrainingCheckpoint.ToInferenceModel()"/> and rig-based load/save work against it.
        /// Validates that the checkpoint fits this rig — trainable-param, model-state and
        /// optimizer-state field names, element types and dimensions — and throws a clear
        /// <see cref="ArgumentException"/> otherwise. The argument is not mutated.
        ///
        /// <para>The values are re-labelled with THIS rig's struct definitions rather than keeping the
        /// argument's, so the result's <c>Definition</c> is the rig's and its fields are in the rig's
        /// order. That matters to anything reading a struct positionally (the <c>[int]</c> indexer,
        /// <c>FlattenedFieldsOfType</c>): a checkpoint read from a file orders its fields the way the
        /// file does.</para>
        /// </summary>
        public TrainingCheckpoint AdoptCheckpoint(TrainingCheckpoint checkpoint)
        {
            if (checkpoint is null) throw new ArgumentNullException(nameof(checkpoint));
            AssertStructDefCompatible(checkpoint.TrainableParams.Definition, TrainableParamStructDef, "trainable-parameter");
            AssertStructDefCompatible(checkpoint.ModelState.Definition, ModelStateDef, "model-state");
            AssertStructDefCompatible(checkpoint.OptimizerState.Definition, OptimizerStateDef, "optimizer-state");
            // Read once: a deferred build describes its parameter and state fields through the
            // stand-ins it was built with, and a materialized one through the values themselves.
            var deferred = Volatile.Read(ref _deferredInit);
            AssertValuesCompatible(
                checkpoint.TrainableParams,
                deferred is null ? Described(_initialParamFields) : Described(deferred.ParamSlots),
                "trainable-parameter");
            AssertValuesCompatible(
                checkpoint.ModelState,
                deferred is null ? Described(_initialStateFields) : Described(deferred.StateSlots),
                "model-state");
            AssertValuesCompatible(checkpoint.OptimizerState, Described(_initialOptStateFields), "optimizer-state");
            // Rebuilt against THIS rig's defs, not carried over: the checks above establish the two
            // agree field for field, but a checkpoint read straight from a file carries a def
            // reconstructed from that file, whose field ORDER is the file's. Everything that indexes
            // a struct positionally (TensorDataStruct's indexer, FlattenedFieldsOfType) would then
            // read the rig's order against the file's. Same values, rig's definition -- and each
            // field fed as it was given, as the checkpoint's own FeedMode is carried below.
            return new TrainingCheckpoint
            {
                TrainableParams = checkpoint.TrainableParams.WithDefinition(TrainableParamStructDef),
                ModelState = checkpoint.ModelState.WithDefinition(ModelStateDef),
                OptimizerState = checkpoint.OptimizerState.WithDefinition(OptimizerStateDef),
                Step = checkpoint.Step,
                Epoch = checkpoint.Epoch,
                BatchIndex = checkpoint.BatchIndex,
                Rig = this,
                Loss = checkpoint.Loss,
                FeedMode = checkpoint.FeedMode,
            };
        }

        /// <summary>Fails loud when a checkpoint's struct def does not match this rig's, by field
        /// names and per-field dtype/structure. The dimensions are checked separately, against the
        /// rig's own values; see <see cref="AssertValuesCompatible"/>.</summary>
        private static void AssertStructDefCompatible(TensorStructDef actual, TensorStructDef expected, string kind)
        {
            if (actual.Fields.Length != expected.Fields.Length)
                throw new ArgumentException(
                    $"Checkpoint's {kind} definition has {actual.Fields.Length} field(s), but this rig " +
                    $"expects {expected.Fields.Length}. It may come from a different model or optimizer" +
                    // A file with no trainable section is refused as it is read, so only the other two
                    // kinds can reach here by having been saved without their component.
                    (kind == "trainable-parameter" ? "." :
                        ", or be a checkpoint saved without this component — read without a rig, a " +
                        "component the file omits comes back empty."));
            for (int i = 0; i < expected.Fields.Length; i++)
            {
                var e = expected.Fields[i];
                var a = actual.GetField(e.Name)
                    ?? throw new ArgumentException(
                        $"Checkpoint's {kind} definition is missing field '{e.Name}' this rig expects.");
                // Not compared: the field def's Rank. It describes the graph's struct TYPE, read off
                // the dtype the training graph carries, and a trainable parameter's rank is not
                // stated there at all (it is null) — the parameter's shape lives on its MODEL_PARAM
                // node. Rank was standing in for "does this value fit this slot", which
                // AssertValuesCompatible below answers exactly, against the rig's own parameters.
                if (a.ElementType != e.ElementType || a.Structure != e.Structure)
                    throw new ArgumentException(
                        $"Checkpoint's {kind} field '{e.Name}' ({a.ElementType}) does not match this " +
                        $"rig's ({e.ElementType}).");
            }
        }

        /// <summary>Fails loud when a checkpoint field's dimensions or element type differ from this
        /// rig's own initial value for that field. The struct defs cannot make either check: a field def
        /// carries a rank and no shape, and on a rig-supplied load the checkpoint's def <i>is</i> this
        /// rig's, so comparing the two defs' dtypes compares a def with itself. Left unchecked, a
        /// checkpoint from a model of another width matches def-for-def and is adopted, to surface later
        /// as a shape-inference error inside the runtime.</summary>
        private static void AssertValuesCompatible(
            TensorDataStruct actual, IEnumerable<(string Name, Shape Shape, DType DType)> expected, string kind)
        {
            foreach (var (name, shape, dtype) in expected)
            {
                if (!actual.Fields.TryGetValue(name, out var actualField) || actualField is not TensorData a) continue;
                if (a.DType != dtype)
                    throw new ArgumentException(
                        $"Checkpoint's {kind} '{name}' is {a.DType}, but this rig's is {dtype}.");
                if (a.Shape.Dims.SequenceEqual(shape.Dims)) continue;
                throw new ArgumentException(
                    $"Checkpoint's {kind} '{name}' is shaped [{string.Join(",", a.Shape.Dims)}], but this rig's "
                    + $"is [{string.Join(",", shape.Dims)}].");
            }
        }

        /// <summary>Shape and dtype per field of a value family that has its values.</summary>
        private static IEnumerable<(string Name, Shape Shape, DType DType)> Described(
            Dictionary<string, IData> fields)
            => fields.Where(kv => kv.Value is TensorData)
                     .Select(kv => (kv.Key, ((TensorData)kv.Value).Shape, ((TensorData)kv.Value).DType));

        /// <summary>Shape and dtype per field of a family a deferred build has only stand-ins for.</summary>
        private static IEnumerable<(string Name, Shape Shape, DType DType)> Described(
            IReadOnlyDictionary<string, TensorAttribute> slots)
            => slots.Select(kv => (kv.Key, kv.Value.Shape, kv.Value.DType));

        /// <summary>Copies of the rig's initial trainable-parameter values, as a struct (for load-time
        /// defaults) — copies for the reason <see cref="CopiesOf"/> gives.</summary>
        internal TensorDataStruct InitialTrainableStruct => new(TrainableParamStructDef, CopiesOf(InitialParamFields));

        /// <summary>Copies of the rig's initial model-state values, as a struct (for load-time
        /// defaults).</summary>
        internal TensorDataStruct InitialModelStateStruct => new(ModelStateDef, CopiesOf(InitialStateFields));

        /// <summary>Copies of the rig's initial optimizer-state values, as a struct (for load-time
        /// defaults).</summary>
        internal TensorDataStruct InitialOptimizerStateStruct => new(OptimizerStateDef, CopiesOf(InitialOptStateFields));

        /// <summary>
        /// Fresh copies of one family of the rig's initial values, in the framework's own host
        /// memory: what every checkpoint the rig hands out over its initial values is built from.
        ///
        /// <para>Copies, because a step consumes the checkpoint it is fed as it is, and the rig keeps
        /// its initial values for the next such checkpoint. A step fed copies takes nothing of the
        /// rig's, and its outputs can be written into the memory of the copies it consumed. A step
        /// fed the rig's own values could only read them — and a run on a card reads host memory
        /// through a copy it makes there and keeps on the tensor it read, for as long as that tensor
        /// lives: for the rig's values, the rig's whole life, a copy of the initial state held on the
        /// card and counted against its runtime context's budget for good. So the rig's own values
        /// are never fed to a run at all.</para>
        /// </summary>
        private static Dictionary<string, IData> CopiesOf(Dictionary<string, IData> family)
        {
            var copies = new Dictionary<string, IData>(family.Count);
            foreach (var (name, value) in family)
                copies[name] = value is TensorData tensor ? tensor.CopyTo(ComputeContext.Host) : value;
            return copies;
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
            InternalComputationGraph concreteArch,
            InternalComputationGraph lossGraph,
            InternalComputationGraph optimizerGraph,
            Hyperparameter[] hyperparameters,
            IReadOnlyList<string>? hyperparamNames,
            BuildProgressReporter? progress = null)
        {
            void Stage(string stage) => progress?.Report(BuildPhase.TrainingStep, stage);

            // Normalize the optimizer graph in place. State variables created by the optimizer's
            // [StateInitializer] Init calls are rewritten into explicit graph inputs appended
            // after grad, and the StateUpdate-pattern nodes (STATE_UPDATE_LINK + WITH_STATE_DEPS)
            // into the explicit multi-output convention [updated_param, updated_state_0, ...]
            // expected by the replay loop. The split-off state-init graph computes the initial
            // state values (per trainable parameter) in InitializeAndOptimize.
            // FromScratchInternal hands in an owned thawed copy — normalize it in place.
            var optimizerFastGraph = optimizerGraph;
            Stage("NormalizeOptimizerGraph");
            var optimizerInfo = Shorokoo.Core.Nodes.Processors.Fast.FastNormalizeOptimizerGraph.Process(optimizerFastGraph);
            _optimizerStateInitGraph = optimizerInfo.StateInitGraph;

            // Value route: the value each hyper contributes to optimizer state init, at the
            // initial counters. Baked → its constant; scheduled → its graph evaluated via QEE below;
            // runtime → null (host-supplied). Filled per kind as the hypers are wired.
            _hyperparamInitialCounterValues = new TensorData?[hyperparameters.Length];

            // Step 1: Compose model + loss + autograd via TrainingGraphBuilder. The model
            // graph is already through ToConcreteArchitecture (done once at FromScratch),
            // so the input-aware liveness filter has already pruned dead-branch trainable
            // params — FastReplaceTrainableParamsWithInputProcessor inside PrepareForTraining
            // builds the trainable param struct from only the live MODEL_PARAM nodes.
            Stage("ComposeModelLossAndAutoGrad");
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
            //
            // A loss whose body never reads that input (a forwarder over a model that computes its own
            // loss) gets no field at all and the rig keeps a placeholder for the dead graph input
            // instead (Shorokoo/Shorokoo#331). The question is the same reachability one asked of the
            // optimizer's state-init graph, so it is asked the same way.
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
                if (Shorokoo.Core.Training.TrainingGraphBuilder.LossReadsTarget(lossGraph))
                {
                    TargetDef = new TensorStructDef(
                        [new TensorStructFieldDef(targetFieldName, DataStructure.Tensor, targetRank, targetDtype)],
                        "Targets");
                    _ignoredTargetPlaceholder = null;
                }
                else
                {
                    TargetDef = new TensorStructDef([], "Targets");
                    _ignoredTargetPlaceholder = IgnoredTargetPlaceholder(targetDtype, targetRank);
                }
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
            //    (Hyperparameter.Runtime)             field each, at the declared dtype), supplied every step.
            // The optimizer's declared Scalar<T> signature is the dtype source of truth throughout.
            string NameOf(int h) => hyperparamNames is not null && h < hyperparamNames.Count
                ? hyperparamNames[h] : $"hyperparam_{h}";

            HyperparameterNames = Enumerable.Range(0, numHyperparams).Select(NameOf).ToArray();
            HyperparameterDTypes = optimizerInfo.HyperparamDTypes;
            for (int h = 0; h < numHyperparams; h++)
                HyperparameterValues.AssertSupported(optimizerInfo.HyperparamDTypes[h], NameOf(h));

            // A hyperparameter's shape is fixed by whatever it is bound to — a baked constant's own
            // shape, a scheduler graph's output shape, or a runtime binding's declared shape — and is
            // checked against the declared rank (when the signature pins one) as each is wired below.
            var hyperShapes = new Shape[numHyperparams];
            HyperparameterShapes = hyperShapes;

            TensorData SeedOf(int h) => SeedValue(
                hyperparameters[h], optimizerInfo.HyperparamDTypes[h], optimizerInfo.HyperparamRanks[h], NameOf(h));

            void PinShape(int h, Shape shape)
            {
                var declaredRank = optimizerInfo.HyperparamRanks[h];
                if (declaredRank is int dr && dr != shape.Dims.Length)
                    throw new ArgumentException(
                        $"Hyperparameter '{NameOf(h)}' is declared with rank {dr}, but it is bound to a " +
                        $"rank-{shape.Dims.Length} value (shape [{string.Join(", ", shape.Dims)}]).",
                        nameof(hyperparameters));
                hyperShapes[h] = shape;
            }

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
            foreach (var h in runtimeIndices) PinShape(h, new Shape([.. hyperparameters[h].RuntimeShape]));
            var hyperFields = runtimeIndices
                .Select(h => new TensorStructFieldDef(
                    NameOf(h), DataStructure.Tensor, hyperShapes[h].Dims.Length,
                    optimizerInfo.HyperparamDTypes[h]))
                .ToArray();
            HyperparameterStructDef = new TensorStructDef(hyperFields, "Hyperparameters");
            DynamicHyperparameterNames = hyperFields.Select(f => f.Name).ToArray();

            // Runtime-hyper optimizer-index → field name, for the CreateInitialCheckpoint override.
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
                    _initialHyperparamFields[f.Name] = SeedOf(runtimeIndices[i]);
                }
            }

            // --- Scheduled hyperparameters: emitted in-graph from the named int64 counter inputs. ---
            // The counter inputs {step, epoch, batchIndex} are shared graph inputs; each scheduler
            // (built-in lowering or user module) consumes a named subset and is inlined against
            // exactly those inputs via FastReplay. Built-in DSL schedules are step-only (PerEpoch
            // derives its epoch in-graph from step, #39); a module declares its subset by input name.
            var counterInputsInOrder = new List<(FastTensorKey Key, string Name)>();
            if (scheduledIndices.Count > 0)
            {
                // Build every scheduler graph first, so the union of counter inputs they consume is
                // known before the shared counter input nodes are created.
                Stage("BuildSchedulers");
                var builtByIndex = new Dictionary<int, SchedulerGraph>(scheduledIndices.Count);
                var needed = new HashSet<string>(StringComparer.Ordinal);
                foreach (var h in scheduledIndices)
                {
                    var built = BuildSchedulerModule(
                        hyperparameters[h], NameOf(h), optimizerInfo.HyperparamDTypes[h],
                        optimizerInfo.HyperparamRanks[h], MergeContext);
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
                    // Value route: the scheduler graph is the single truth, so its value at the
                    // initial counters — what optimizer state init needs — comes from evaluating that
                    // very graph via QEE, not a hardcoded 0f (the old scheduler-module state-init hole).
                    _hyperparamInitialCounterValues[h] = EvaluateSchedulerAtInitialCounters(built.Graph);
                    PinShape(h, _hyperparamInitialCounterValues[h]!.Shape);
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
                _hyperparamInitialCounterValues[h] = SeedOf(h);
                PinShape(h, _hyperparamInitialCounterValues[h]!.Shape);
                // A copy: the rig keeps the seed value as the hyperparameter's own, and the graph
                // keeps the literal.
                var node = Shorokoo.Core.Nodes.Processors.Fast.FastInternalOp.Constant(
                    _hyperparamInitialCounterValues[h]!.CopyTo(ComputeContext.Host).MoveToAttribute());
                fastTraining.Nodes.Add(node);
                headNodesInOrder.Add(node);
                hyperparamKeys[h] = new FastTensorKey(node.Key, 0);
            }

            // Record the bindings the rig actually built with: a baked value is normalized to its
            // declared dtype here, so rig.Hyperparameters[h].BakedDType is always
            // HyperparameterDTypes[h] and persistence writes the constant the graph carries rather
            // than whichever host literal the caller happened to type.
            var normalizedHypers = (Hyperparameter[])hyperparameters.Clone();
            for (int h = 0; h < numHyperparams; h++)
                if (hyperparameters[h].Kind == HyperparameterKind.Baked)
                    normalizedHypers[h] = Hyperparameter.Baked(_hyperparamInitialCounterValues[h]!);
            _constituents = _constituents with { Hyperparameters = normalizedHypers };

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
                            OptimizerStateFieldName(pf.Name, s), pf.Structure,
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

            // Step 7: Apply optimizer per field by replaying the optimizer graph. One replay per
            // trainable parameter into a graph that grows with each — the build's longest unreported
            // stretch before this reported it, so a large model looked stuck in the previous stage.
            Stage("ReplayOptimizerPerParameter");
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

            Stage("PruneAndOrderTrainingStep");
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
            _trainingStepWorkGraph = LowerGraph(
                fastTraining, MergeContext, progress, lowerAutoGrad: TrainingBackend.LowersAutoGrad);

            UpdatedParamFieldCount = TrainableParamStructDef.Fields.Length;
            UpdatedStateFieldCount = ModelStateDef.Fields.Length;
            UpdatedOptimizerStateFieldCount = OptimizerStateDef.Fields.Length;
        }

        /// <summary>
        /// The tensor used to seed shape inference (and, for a baked hyper, its graph constant), at the
        /// hyperparameter's <paramref name="declared"/> dtype: a baked hyper's constant fitted to that
        /// dtype (keeping its own shape), a built-in schedule's step-0 scalar, else a zero of the shape
        /// the binding declares (a runtime hyper, or a scheduler module — whose value comes from
        /// evaluating its graph, see <see cref="EvaluateSchedulerAtInitialCounters"/>).
        /// </summary>
        private static TensorData SeedValue(Hyperparameter h, DType declared, int? declaredRank, string name)
            => h.Kind switch
            {
                HyperparameterKind.Baked => HyperparameterValues.ConvertTo(h.BakedValue, declared, name),
                HyperparameterKind.Scheduled when h.AsSchedule is Schedule s && s.CanLower()
                    => HyperparameterValues.ConvertTo(HyperparameterValues.Of(s.At(0)), declared, name),
                HyperparameterKind.Runtime => HyperparameterValues.Zero(declared, h.RuntimeShape),
                _ => HyperparameterValues.Zero(declared, new long[declaredRank ?? 0]),
            };

        /// <summary>
        /// Evaluates a scheduler graph (built-in lowering or user module) at the <b>initial counters</b>
        /// — every counter input bound to 0 — via the pure managed <see cref="Shorokoo.Core.Interpreter.QuickExecutionEngine"/>,
        /// returning the scalar value at the scheduler's own (declared) dtype. This is the single value route for optimizer state init:
        /// the scheduler graph is normative, so its build-time value comes from evaluating it, not from
        /// a host closure or a hardcoded placeholder. The graph is pure (enforced), so all-zero
        /// counters fully determine the value.
        /// </summary>
        private static TensorData EvaluateSchedulerAtInitialCounters(InternalComputationGraph schedulerGraph)
        {
            var inputs = new IData[schedulerGraph.Inputs.Count];
            for (int i = 0; i < inputs.Length; i++)
                inputs[i] = Shorokoo.Globals.TensorData(Array.Empty<long>(), 0L);
            var result = new Shorokoo.Core.Interpreter.QuickExecutionEngine().Execute(schedulerGraph, inputs);
            return (TensorData)result[0];
        }

        /// <summary>
        /// Builds the graph a scheduled hyperparameter is emitted from: a module taking the int64
        /// scalar counter input(s) and producing the scheduled value at the hyperparameter's declared
        /// dtype and shape. A built-in <see cref="Schedule"/> is lowered via <see cref="ScheduleLowering"/>
        /// — its math is continuous and scalar, so it drives <c>float32</c> scalar hyperparameters only; a
        /// user scheduler module is validated, purity-checked, and inlined, and may produce any declared
        /// dtype at any shape. The returned graph is spliced into the
        /// training-step graph by <see cref="Shorokoo.Core.Nodes.Processors.Fast.FastReplay.ReplayInto"/>
        /// against the shared step-counter input.
        /// </summary>
        /// <summary>The reserved counter inputs a scheduler graph may consume, in canonical order.</summary>
        internal static readonly string[] CounterInputNames = ["step", "epoch", "batchIndex"];

        /// <summary>A built scheduler graph and the counter inputs it consumes, in the graph's input order.</summary>
        private readonly record struct SchedulerGraph(InternalComputationGraph Graph, string[] CounterNames);

        private static SchedulerGraph BuildSchedulerModule(
            Hyperparameter hv, string name, DType declared, int? declaredRank, ComputeContext mergeContext)
        {
            if (hv.AsSchedule is Schedule schedule)
            {
                if (!schedule.CanLower())
                    throw new ArgumentException(
                        $"Scheduled hyperparameter '{name}' wraps an opaque host function and cannot be " +
                        "lowered to graph math. Build the schedule from the Schedules factories and " +
                        "Schedule combinators, or supply a scheduler module.", nameof(hv));
                // Built-in Schedule math (cosine / linear / decay) is inherently continuous float32, so a
                // hyperparameter of any other dtype needs a scheduler module rather than a built-in.
                if (declared != DType.Float32 || declaredRank is int r && r != 0)
                    throw new ArgumentException(
                        $"Hyperparameter '{name}' is declared '{declared}' rank " +
                        $"{declaredRank?.ToString() ?? "(any)"}, but a built-in Schedule produces a float32 " +
                        "scalar. Drive a non-float32 or non-scalar hyperparameter with a scheduler module " +
                        "(Hyperparameter.Scheduled(module)) that produces its declared dtype and shape.",
                        nameof(hv));
                // Built-in DSL schedules are step-only (PerEpoch derives epoch in-graph from step, #39).
                var step = Shorokoo.Globals.InputScalar<int64>("step");
                var value = schedule.LowerToGraph(step);
                return new SchedulerGraph(new InternalComputationGraph([step], [value]), ["step"]);
            }

            var module = hv.AsSchedulerModule
                ?? throw new InvalidOperationException(
                    $"Scheduled hyperparameter '{name}' has neither a built-in schedule nor a scheduler module.");
            return ValidateAndInlineSchedulerModule(module, name, declared, declaredRank, mergeContext);
        }

        /// <summary>
        /// Validates a user scheduler module's signature — its inputs a subset of the reserved int64
        /// scalar counters <c>{step, epoch, batchIndex}</c> (each named, rank-0, no duplicates) and
        /// a single output at the hyperparameter's declared dtype and rank (any shape the module produces
        /// is allowed when the declaration is rank-agnostic) — enforces purity, and returns its
        /// inlined graph together
        /// with the counter names it consumes (in input order, for wiring). Fails loud at rig build with
        /// a clear message on any signature/purity mismatch.
        /// </summary>
        private static SchedulerGraph ValidateAndInlineSchedulerModule(
            ComputationGraph module, string name, DType declared, int? declaredRank, ComputeContext mergeContext)
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

            // Purity contract: a scheduler graph is a pure function of its counter inputs. After
            // inlining, reject any trainable param, module state / StateUpdate, or RNG draw — impure
            // constructs would be inlined into the trainstep with an undefined failure mode.
            AssertSchedulerGraphPure(g, name);

            // Each input must be a named reserved counter (int64 scalar), with no duplicates.
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
            var outInfo = new ShapeInferenceInterpreter(mergeContext)
                .Infer(g, zeros)
                .GetTensorInfo(g.Outputs[0])
                ?? throw new ArgumentException(
                    $"Scheduler module for hyperparameter '{name}': could not infer its output shape.",
                    nameof(module));
            if (outInfo.DType != declared)
                throw new ArgumentException(
                    $"Scheduler module for hyperparameter '{name}' must produce a '{declared}' value " +
                    $"(the dtype the optimizer declares it at); got {outInfo.DType}.", nameof(module));
            if (declaredRank is int wantRank && outInfo.Shape.Dims.Length != wantRank)
                throw new ArgumentException(
                    $"Scheduler module for hyperparameter '{name}' must produce a rank-{wantRank} value " +
                    $"(the rank the optimizer declares it at); got rank {outInfo.Shape.Dims.Length}.",
                    nameof(module));

            return new SchedulerGraph(g, counterNames);
        }

        /// <summary>
        /// Enforces the scheduler-graph purity contract: fails loud at rig build if the inlined
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
                        or InternalOpCodes.SHRK_RANDOM_BITS
                        or InternalOpCodes.SHRK_RNG_UNIFORM or InternalOpCodes.SHRK_RNG_NORMAL
                        or InternalOpCodes.SHRK_RNG_BITS or InternalOpCodes.SHRK_RNG_SPLIT
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
        /// <param name="fast">The composed training graph, lowered in place and returned.</param>
        /// <param name="mergeContext">The build/merge context the folds evaluate on.</param>
        /// <param name="progress">The build's progress reporter, if any.</param>
        /// <param name="lowerAutoGrad">False for a step whose gradient is left to the execution
        /// backend (<see cref="TrainingBackend.Native"/>): the pipeline stops before the autograd
        /// expansion, and the step keeps its one <c>AUTO_GRAD</c> node for the backend to run.</param>
        private static InternalComputationGraph LowerGraph(
            InternalComputationGraph fast, ComputeContext mergeContext, BuildProgressReporter? progress = null,
            bool lowerAutoGrad = true)
        {
            void Stage(string stage) => progress?.Report(BuildPhase.TrainingStep, stage);

            // Expand TensorStruct outputs into individual field outputs.
            Stage("ExpandStructOutputs");
            Shorokoo.Core.Nodes.Processors.Fast.FastExpandStructOutputs.Process(fast);

            // Unpack TensorStruct inputs (struct → individual fields).
            Stage("UnpackTensorStructs");
            Shorokoo.Core.Nodes.Processors.Fast.FastUnpackTensorStructs.Process(fast);

            // Simplify before loop unrolling. Any loop whose iteration count is already a direct
            // Constant node will be unrolled here via FastFoldConstantIterationLoops inside
            // FastSimplify.
            Stage("Simplify");
            Shorokoo.Core.Nodes.Processors.Fast.FastSimplify.Process(fast);

            // Resolve any remaining LOOP_OPEN iteration counts that are computed from constants
            // (e.g. Sub(Constant(2), Constant(1))) into literal Constant nodes. Autograd has no
            // gradient implementation for Loop, so every loop reaching autograd must be flattened.
            Stage("FoldLoopIterationCounts");
            FastFoldLoopIterationCountsToConstantsProcessor.Process(fast, mergeContext);

            // Simplify after iteration-count resolution; the FastFoldConstantIterationLoops pass
            // inside FastSimplify performs the actual unroll, then folds remaining constants.
            Stage("UnrollLoops");
            Shorokoo.Core.Nodes.Processors.Fast.FastSimplify.Process(fast);

            // Every loop that feeds the loss had to be unrolled by now; one that could not be is a
            // limitation to name, not a backward pass to attempt (Shorokoo/Shorokoo#309).
            FastRejectRolledLoopsInTraining.Process(fast);

            // Lower attribute-tensorized variant ops (e.g. SHRK_CONV) to standard ONNX ops before
            // autograd — they have no gradient rule. Loops are unrolled by this point, so their
            // geometry inputs are constant-foldable.
            Stage("LowerAttributeTensorOps");
            Shorokoo.Core.Nodes.Processors.Fast.FastLowerAttributeTensorOps.Process(fast, compute: mergeContext);

            // The cut for a step whose gradient the execution backend computes: everything above is
            // shared with the default path, and nothing below runs. The registered-op lowering and
            // the If unscoping inside the autograd pass exist for Shorokoo's own backward walk, so
            // the backend receives those ops, and the scopes, as the user wrote them.
            if (!lowerAutoGrad)
            {
                Stage("DeferAutoGradToExecutionBackend");
                RequireSingleTopLevelAutoGrad(fast);
                return fast;
            }

            // Lower AUTO_GRAD nodes natively on the Fast graph — no CG round-trip needed.
            Stage("ExpandAutoGrad");
            Shorokoo.Core.Nodes.Processors.AutoGrad.FastProcessAutoGradProcessor.Process(fast);

            Stage("SimplifyAfterAutoGrad");
            Shorokoo.Core.Nodes.Processors.Fast.FastSimplify.Process(fast);
            return fast;
        }

        /// <summary>
        /// The shape a step handed over in <see cref="TrainingFormats.OnnxAutoGrad"/> is held to: one
        /// <c>AUTO_GRAD</c> node — the rig's own, since concretization lowers any a module authored —
        /// at the top level of the graph, where a backend can differentiate the loss's whole ancestry
        /// with respect to its inputs rather than one arm of a branch or one trip of a loop.
        /// </summary>
        private static void RequireSingleTopLevelAutoGrad(InternalComputationGraph fast)
        {
            int depth = 0, topLevel = 0, total = 0;
            foreach (var node in fast.Nodes)
            {
                if (node.IsCloseNode()) depth--;
                if (node.OpCode == InternalOpCodes.AUTO_GRAD)
                {
                    total++;
                    if (depth == 0) topLevel++;
                }
                if (node.IsOpenNode()) depth++;
            }
            if (total != 1 || topLevel != 1)
                throw new InvalidOperationException(
                    $"A training step whose gradient is left to the execution backend must carry exactly one "
                    + $"top-level AUTO_GRAD node; this one carries {total}, {topLevel} of them at the top level.");
        }

        /// <summary>
        /// Executes a single training step and advances the checkpoint's
        /// <see cref="TrainingCheckpoint.Step"/>. Scheduled hyperparameters (a built-in
        /// <see cref="Schedule"/> or a scheduler module) are computed <b>in-graph</b> from the
        /// checkpoint's current step — fed as the step-counter input — so nothing is host-evaluated
        /// here: the rig compiles its trainstep once (internally, lazily, and cached on the rig) and
        /// every step reuses it, so a manual loop is just <c>cp = rig.TrainStep(cp, in, out);</c> with
        /// no caller-side compile. This overload requires the rig to have <b>no</b>
        /// schedule-less runtime hyperparameter (<see cref="Hyperparameter.Runtime()"/>), which has no value
        /// to apply automatically; use the explicit-override overload for those.
        ///
        /// <para>Each call hands back a host-readable copy of the whole training state and takes it
        /// all back on the next one — free on a CPU backend, and on a GPU the thing that sets the
        /// pace of a long run. A loop that does not need every step's checkpoint should run through
        /// <see cref="BeginResidentRun(TrainingCheckpoint?)"/> instead (Shorokoo/Shorokoo#325).</para>
        ///
        /// <para><b>What the step consumes.</b> Its arguments are fed the way any run's inputs are:
        /// as they are, they are <b>consumed</b> — the checkpoint's state, the input and the target
        /// are dead once the step has started, but for a field built into one of them with a mode of
        /// its own, and their memory is released with the step rather than whenever they are
        /// collected. That is what <c>cp = rig.TrainStep(cp, x, y)</c> with a
        /// batch built per step wants. To use one again, pass it <c>.Shared()</c> — a checkpoint you
        /// keep, a batch you feed every step — or <c>.TryConsume()</c> to have it consumed only when
        /// nothing else is reading it. A checkpoint from <see cref="CreateInitialCheckpoint()"/> is
        /// no exception: its tensors are copies of the rig's initial values, made for it, so
        /// consuming it takes nothing from the rig or from the next initial checkpoint.</para>
        ///
        /// <para>A step that reads what it cannot address where it is — a host tensor on a card, a
        /// managed array on any backend — reads it through a copy the tensor holds for its next read.
        /// It lets go of the copies of its batch as it returns, however it ends, so a batch fed every
        /// step is copied afresh each time rather than held twice; the copies of a checkpoint it read
        /// stay with the checkpoint for as long as it lives.</para>
        /// </summary>
        /// <param name="checkpoint">Current training state (params, model state, optimizer state,
        /// step) — consumed by the step unless passed <c>.Shared()</c> or <c>.TryConsume()</c>.</param>
        /// <param name="trainingInput">Training input data: a <see cref="TensorDataStruct"/>, or one
        /// passed through <c>.Shared()</c> or <c>.TryConsume()</c>.</param>
        /// <param name="trainingOutput">Training target data, in the same forms.</param>
        /// <returns>The post-step checkpoint (advanced step, updated params/state) with its
        /// <see cref="TrainingCheckpoint.Loss"/> set to this step's loss. Its tensors are new, and
        /// consumed in turn when it is fed to the next step as it is.</returns>
        /// <exception cref="ArgumentException"><paramref name="trainingInput"/> or
        /// <paramref name="trainingOutput"/> is not a struct, nor one passed through
        /// <c>.Shared()</c> or <c>.TryConsume()</c>; or it does not hold what
        /// <see cref="InputDef"/> or <see cref="TargetDef"/> declares — as many fields, each of the
        /// declared kind, element type and stated rank, in order. Refused before the step takes
        /// anything.</exception>
        public TrainingCheckpoint TrainStep(
            TrainingCheckpoint checkpoint,
            IData trainingInput,
            IData trainingOutput)
            => TrainStepWith(checkpoint, trainingInput, trainingOutput);

        /// <summary>
        /// Executes a single training step on a rig whose loss reads no target
        /// (<see cref="HasTargets"/> is <c>false</c>) — a model that computes its own loss under a
        /// forwarding loss module (Shorokoo/Shorokoo#331). Identical to
        /// <see cref="TrainStep(TrainingCheckpoint, IData, IData)"/> in every other respect —
        /// what it consumes included; the target the composed graph still carries is the rig's to
        /// supply, not the caller's. Throws when the rig <i>does</i> read a target, since dropping it
        /// would silently train against something the caller never chose.
        /// </summary>
        /// <param name="checkpoint">Current training state (params, model state, optimizer state, step)</param>
        /// <param name="trainingInput">Training input data: a <see cref="TensorDataStruct"/>, or one
        /// passed through <c>.Shared()</c> or <c>.TryConsume()</c>.</param>
        /// <returns>The post-step checkpoint (advanced step, updated params/state) with its
        /// <see cref="TrainingCheckpoint.Loss"/> set to this step's loss.</returns>
        public TrainingCheckpoint TrainStep(
            TrainingCheckpoint checkpoint,
            IData trainingInput)
        {
            RequireTargetless(nameof(TrainStep));
            return TrainStepWith(checkpoint, trainingInput, TargetDef.FromOrderedData());
        }

        /// <summary>
        /// The explicit-counter form of <see cref="TrainStep(TrainingCheckpoint, IData)"/>,
        /// for a host driving its own data iteration over a rig whose loss reads no target
        /// (Shorokoo/Shorokoo#331). <paramref name="epoch"/> and <paramref name="batchNumber"/> mean
        /// what they mean on
        /// <see cref="TrainStep(TrainingCheckpoint, IData, IData, long, long)"/>.
        /// </summary>
        public TrainingCheckpoint TrainStep(
            TrainingCheckpoint checkpoint,
            IData trainingInput,
            long epoch,
            long batchNumber)
        {
            RequireTargetless(nameof(TrainStep));
            return TrainStep(checkpoint, trainingInput, TargetDef.FromOrderedData(), epoch, batchNumber);
        }

        /// <summary>
        /// Fails loud when <paramref name="operation"/> — a target-free entry point — is called on a rig
        /// whose loss does read a target, naming the overload that takes one.
        /// </summary>
        internal void RequireTargetless(string operation)
        {
            if (HasTargets)
                throw new InvalidOperationException(
                    $"{operation} without targets requires a loss that ignores its target input, but this "
                    + $"rig's loss reads one (TargetDef: [{string.Join(", ", TargetDef.Fields.Select(f => f.Name))}]). "
                    + $"Use the {operation} overload that takes targets.");
        }

        /// <summary>
        /// Executes a single training step with explicit hyperparameter values, overriding any
        /// schedules for this step (build the values with <see cref="MakeHyperparameters(float)"/> or
        /// <see cref="MakeHyperparameters(ValueTuple{string, object}[])"/>). Use this for manual control, or
        /// for rigs whose dynamic hyperparameters are schedule-less (<see cref="Hyperparameter.Runtime()"/>).
        /// In-graph scheduled hyperparameters (built-in schedules / scheduler modules) are unaffected
        /// by this overload — they are always computed from the step counter — so <paramref name="hyperparams"/>
        /// carries only the schedule-less runtime values.
        /// </summary>
        /// <param name="checkpoint">Current training state (params, model state, optimizer state, step)</param>
        /// <param name="hyperparams">Values for the schedule-less runtime hyperparameters
        /// (<see cref="HyperparameterStructDef"/> order): a struct, consumed like every other feed
        /// — <see cref="MakeHyperparameters(float)"/> builds a fresh one per call, copying any tensor
        /// it is given — or one passed through <c>.Shared()</c> to be read every step.</param>
        /// <param name="trainingInput">Training input data: a <see cref="TensorDataStruct"/>, or one
        /// passed through <c>.Shared()</c> or <c>.TryConsume()</c>.</param>
        /// <param name="trainingOutput">Training target data, in the same forms.</param>
        /// <returns>The post-step checkpoint (advanced step, updated params/state) with its
        /// <see cref="TrainingCheckpoint.Loss"/> set to this step's loss.</returns>
        public TrainingCheckpoint TrainStep(
            TrainingCheckpoint checkpoint,
            IData hyperparams,
            IData trainingInput,
            IData trainingOutput)
        {
            if (hyperparams is null) throw new ArgumentNullException(nameof(hyperparams));
            return RunStep(checkpoint, hyperparams, trainingInput, trainingOutput);
        }

        /// <summary>
        /// Executes a single training step on the next batch drawn from <paramref name="loader"/>,
        /// sourcing the epoch / batch counters from the loader — the single-step analogue of
        /// <see cref="Fit(IDataLoader, int, TrainingCheckpoint?)"/>, sharing its step body so the two
        /// agree on the loader-step-and-counter semantics below.
        ///
        /// <para>The batch's <b>own</b> position (<see cref="DataBatch.Position"/>) both drives the
        /// scheduler counters for this step (so a scheduled hyperparameter reading the epoch / batch
        /// counters sees the batch being trained) and is recorded on the returned checkpoint's
        /// <see cref="TrainingCheckpoint.Epoch"/> / <see cref="TrainingCheckpoint.BatchIndex"/> — the
        /// unified convention that the checkpoint stores the batch that was <b>used</b> (recorded ==
        /// what-drove-the-step). Resuming advances past it: feeding the returned checkpoint back to
        /// <c>Fit(loader)</c> restores the loader with <see cref="IDataLoader.RestoreAfter"/>, continuing
        /// at exactly the batch after this one. <see cref="TrainingCheckpoint.Step"/> is advanced by one
        /// and the attached <see cref="TrainingCheckpoint.Rig"/> and this step's
        /// <see cref="TrainingCheckpoint.Loss"/> are preserved (via
        /// <see cref="TrainingCheckpoint.WithCounters"/>).</para>
        ///
        /// <para>Like the counter-agnostic <see cref="TrainStep(TrainingCheckpoint, IData, IData)"/>
        /// it drives, this schedule-driven form requires the rig to have no schedule-less runtime
        /// hyperparameter (<see cref="Hyperparameter.Runtime()"/>); supply those via
        /// <see cref="MakeHyperparameters(float)"/> and a manual explicit-data loop instead. It
        /// consumes what that one does: the checkpoint as passed, and the batch as the loader hands
        /// it over (<see cref="DataBatch"/>).</para>
        /// </summary>
        /// <param name="checkpoint">Current training state; its counters are replaced from the loader.</param>
        /// <param name="loader">The data loader; <see cref="IDataLoader.Next"/> is called once.</param>
        /// <returns>The post-step checkpoint: step advanced, epoch / batch set to the position of the
        /// batch used, with this step's loss.</returns>
        public TrainingCheckpoint TrainStep(
            TrainingCheckpoint checkpoint,
            IDataLoader loader)
            => TrainStepWith(checkpoint, loader);

        /// <summary>
        /// Executes a single training step on caller-supplied data with an explicit epoch and batch
        /// number — the counter-sourcing analogue of
        /// <see cref="TrainStep(TrainingCheckpoint, IDataLoader)"/> for a host driving its
        /// own data iteration (no <see cref="IDataLoader"/>).
        ///
        /// <para><b>What the recorded counters mean.</b> <paramref name="epoch"/> and
        /// <paramref name="batchNumber"/> name the position of the batch you are training now. They are
        /// fed to any scheduled hyperparameter reading the epoch / batch counters during this step, and
        /// are recorded <b>verbatim</b> on the returned checkpoint's <see cref="TrainingCheckpoint.Epoch"/>
        /// / <see cref="TrainingCheckpoint.BatchIndex"/> — the same "batch used" convention the loader
        /// overload records (there it is the drawn batch's own position). Both overloads therefore agree:
        /// the checkpoint stores the batch that was <b>used</b>, and a loader-driven resume advances one
        /// past it via <see cref="IDataLoader.RestoreAfter"/> (the loader owns the epoch rollover). So
        /// resuming <c>Fit(loader)</c> from a checkpoint produced here continues at the batch <b>after</b>
        /// this one — no re-run. A host owning its own iteration passes each batch's position and manages
        /// resume itself. <see cref="TrainingCheckpoint.Step"/> is advanced by one; the attached
        /// <see cref="TrainingCheckpoint.Rig"/> and this step's <see cref="TrainingCheckpoint.Loss"/> are
        /// preserved.</para>
        ///
        /// <para>Like <see cref="TrainStep(TrainingCheckpoint, IData, IData)"/>,
        /// this schedule-driven form requires the rig to have no schedule-less runtime hyperparameter
        /// (<see cref="Hyperparameter.Runtime()"/>); use the explicit-hyperparameters overload and set the
        /// counters via <see cref="TrainingCheckpoint.WithCounters"/> for those. It consumes what that
        /// one does.</para>
        /// </summary>
        /// <param name="checkpoint">Current training state; its epoch / batch counters are replaced by the arguments.</param>
        /// <param name="trainingInput">Training input data: a <see cref="TensorDataStruct"/>, or one
        /// passed through <c>.Shared()</c> or <c>.TryConsume()</c>.</param>
        /// <param name="trainingOutput">Training target data, in the same forms.</param>
        /// <param name="epoch">The 0-based epoch of the batch being trained; recorded verbatim.</param>
        /// <param name="batchNumber">The 0-based batch index of the batch being trained; recorded verbatim.</param>
        /// <returns>The post-step checkpoint: step advanced, epoch / batch set to the given values, with this step's loss.</returns>
        public TrainingCheckpoint TrainStep(
            TrainingCheckpoint checkpoint,
            IData trainingInput,
            IData trainingOutput,
            long epoch,
            long batchNumber)
        {
            if (checkpoint is null) throw new ArgumentNullException(nameof(checkpoint));
            if (epoch < 0)
                throw new ArgumentOutOfRangeException(nameof(epoch), epoch, "Epoch must be non-negative.");
            if (batchNumber < 0)
                throw new ArgumentOutOfRangeException(nameof(batchNumber), batchNumber, "Batch number must be non-negative.");
            var stepInput = checkpoint.WithCounters(epoch: epoch, batchIndex: batchNumber);
            return TrainStep(stepInput, trainingInput, trainingOutput);
        }

        /// <summary>
        /// The shared body of the counter-agnostic data <c>TrainStep</c>: applies the
        /// no-runtime-hyperparameter guard, then runs the step against the rig's cached compiled
        /// trainstep (<see cref="_compiledTrainSteps"/>, compiled via <see cref="RuntimeContext"/>). Both
        /// the public <c>TrainStep</c> overload and <see cref="Train"/> route through here, so they share
        /// the rig's compiled graphs.
        /// </summary>
        private TrainingCheckpoint TrainStepWith(
            TrainingCheckpoint checkpoint,
            IData trainingInput,
            IData trainingOutput)
        {
            if (checkpoint is null) throw new ArgumentNullException(nameof(checkpoint));
            RequireNoRuntimeHyperparameters();
            return RunStep(checkpoint, hyperparams: null, trainingInput, trainingOutput);
        }

        /// <summary>
        /// The shared body of the loader <c>TrainStep</c>, run against the rig's cached compiled
        /// trainstep (<see cref="_compiledTrainSteps"/>). It draws the batch and hands it to
        /// <see cref="BatchStep"/>, the one place the loader-step-and-counter semantics live
        /// — the same one <see cref="Fit(IDataLoader, int, TrainingCheckpoint?)"/> and
        /// <see cref="ResidentTrainingRun"/> step through, so every loader-driven form agrees. It
        /// retains nothing: the checkpoint this returns is the caller's to read.
        /// </summary>
        private TrainingCheckpoint TrainStepWith(
            TrainingCheckpoint checkpoint,
            IDataLoader loader)
        {
            if (checkpoint is null) throw new ArgumentNullException(nameof(checkpoint));
            if (loader is null) throw new ArgumentNullException(nameof(loader));
            return BatchStep(checkpoint, loader.Next(), retain: false, TrainStepCall);
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

        /// <summary>
        /// Backend bytes of superseded checkpoint state produced since the last reclamation.
        /// </summary>
        private long _supersededStateBytes;

        /// <summary>
        /// The budget currently in force. Starts at <see cref="SupersededStateBudgetBytes"/> and
        /// doubles, up to <see cref="MaxReclaimBudgetBytes"/>, for as long as reclaiming turns out
        /// not to reclaim anything — see <see cref="ReclaimSupersededState"/>.
        /// </summary>
        private long _reclaimBudgetBytes = SupersededStateBudgetBytes;

        /// <summary>The budget the backoff snaps back to. Constant in production; a test lowers it
        /// so a handful of steps exercises the backoff that a real run reaches after thousands.</summary>
        private long _baseReclaimBudgetBytes = SupersededStateBudgetBytes;

        /// <summary>The budget currently in force (test hook).</summary>
        internal long ReclaimBudgetBytes => System.Threading.Interlocked.Read(ref _reclaimBudgetBytes);

        /// <summary>Superseded state counted towards the next reclamation and not yet reclaimed
        /// (test hook).</summary>
        internal long SupersededStateBytesPending => System.Threading.Interlocked.Read(ref _supersededStateBytes);

        /// <summary>Lowers both the base budget and the one in force (test hook).</summary>
        internal void SetReclaimBudgetForTests(long bytes)
        {
            System.Threading.Interlocked.Exchange(ref _baseReclaimBudgetBytes, bytes);
            System.Threading.Interlocked.Exchange(ref _reclaimBudgetBytes, bytes);
        }

        /// <summary>The checkpoints produced at the last few reclamations, watched weakly, newest
        /// first. Whether they all survived is what says the caller is keeping its checkpoints —
        /// see <see cref="ReclaimSupersededState"/> for why all of them, and why not the newest.</summary>
        private readonly WeakReference<TrainingCheckpoint>?[] _watches = new WeakReference<TrainingCheckpoint>?[4];

        /// <summary>Guards a reclamation end to end — the threshold, the collection and the verdict
        /// are one decision, and two threads making it at once would judge each other's watches.</summary>
        private readonly object _reclaimGate = new();

        /// <summary>
        /// How much superseded state may pile up before <see cref="ReclaimSupersededState"/> does
        /// something about it. Small enough that a step's state is reclaimed within a few steps of
        /// being superseded; large enough that a model whose whole checkpoint is a few kilobytes
        /// reaches it only after thousands of steps, and so pays essentially nothing. It is a
        /// running total, not a per-step test.
        /// </summary>
        private const long SupersededStateBudgetBytes = 32L * 1024 * 1024;

        /// <summary>The ceiling the backoff in <see cref="ReclaimSupersededState"/> climbs to.</summary>
        private const long MaxReclaimBudgetBytes = 8L * 1024 * 1024 * 1024;

        /// <summary>
        /// Collects, once more than the current budget of superseded checkpoint state has piled up.
        ///
        /// <para>A step's checkpoint holds the trainable parameters and every optimizer moment in
        /// runtime-owned buffers, behind managed wrappers of a few dozen bytes each. In
        /// <c>cp = rig.TrainStep(cp, …)</c> — the loop the training guide documents — the previous
        /// checkpoint becomes garbage on every step, but garbage so small that nothing prompts a
        /// collection: the wrappers are never finalized, their buffers are never released, and the
        /// process grows by the whole parameter set plus both moments every step until it dies
        /// (Shorokoo/Shorokoo#321). At 49 M parameters that is ~565 MiB a step, and the run dies
        /// around step 12 of 48,000.</para>
        ///
        /// <para>A step fed its checkpoint as it is consumes it, and that state is released as the
        /// step runs, so for the ordinary loop there is nothing left to collect and this does
        /// nothing: it counts only state the step superseded <b>without</b> consuming — a checkpoint
        /// fed <c>.Shared()</c>, or through <c>.TryConsume()</c> while something else was reading
        /// it — which is alive after the step and garbage the moment its holder drops it.</para>
        ///
        /// <para>For that state the rig knows what the runtime cannot infer from the managed heap:
        /// the step just taken superseded a known quantity of runtime memory. Collecting on that
        /// budget is the same thing a caller would otherwise have to write by hand after every
        /// step. A model superseding a few MiB a step pays one collection every few steps; one
        /// superseding hundreds of MiB a step pays one per step, which is what the by-hand version
        /// cost and what a run of that size has to pay to survive at all. The alternative,
        /// <see cref="GC.AddMemoryPressure"/>, is too slack at this ratio: it collects roughly
        /// every ten steps, and the runtime's arena ratchets upward between collections instead of
        /// settling.</para>
        ///
        /// <para><b>Superseded state is only garbage if the caller drops it.</b> A caller that
        /// shares its checkpoints may legitimately keep them — comparing a step against the one
        /// before it, or holding the best so far — and then there is nothing to reclaim and a
        /// forced collection is pure cost, repeated forever. So the rig watches weakly the
        /// checkpoints it produced, and asks whether one of them survived a later collection: if it
        /// did, the caller is keeping them, and the budget doubles, up to
        /// <see cref="MaxReclaimBudgetBytes"/>, snapping back the moment a watched checkpoint does
        /// not survive.</para>
        ///
        /// <para><b>Which watches are judged matters, and judging one recent checkpoint does not
        /// work.</b> The checkpoint handed back at the last reclamation is exactly what the caller
        /// feeds in as the next step's input, so while reclamations fall on consecutive steps it is
        /// rooted as a live argument of the very call doing the collecting, and survives whether or
        /// not the caller keeps anything — measured, the budget doubled identically for a caller
        /// that kept every checkpoint and one that kept none (Shorokoo/Shorokoo#348). So the newest
        /// watch is never judged. Nor is one older watch enough: a caller holding a single
        /// checkpoint — the best so far — supersedes all the rest, so collecting is still worth
        /// doing, but that one checkpoint surfacing as the watch would read as keeping them all.
        /// The question is therefore asked of a population: the budget doubles only when <b>every</b>
        /// older watch survived, which one retained checkpoint among several cannot produce.</para>
        ///
        /// <para>A tensor owns its storage and releases it when disposed, so a caller who wants
        /// determinism has it; what the rig cannot do is dispose the checkpoint it was handed,
        /// because the caller may still be holding it. Nothing here disposes anything the caller
        /// can still see — it schedules a collection, and only unreachable state is affected.</para>
        /// </summary>
        private void ReclaimSupersededState(long stepBytes, TrainingCheckpoint produced)
        {
            if (System.Threading.Interlocked.Add(ref _supersededStateBytes, stepBytes)
                < System.Threading.Interlocked.Read(ref _reclaimBudgetBytes)) return;

            // The whole sequence, not just the update: two threads crossing the threshold together
            // would otherwise both collect, and the second would judge watches the first had already
            // rotated -- a caller that keeps nothing seeing the budget double, which is the shape
            // #348 was. Collecting under the gate is safe because nothing finalizable takes it.
            lock (_reclaimGate)
            {
                long budget = System.Threading.Interlocked.Read(ref _reclaimBudgetBytes);
                if (System.Threading.Interlocked.Read(ref _supersededStateBytes) < budget) return;
                // Subtract the budget rather than zeroing: another step's bytes may have landed in
                // between, and zeroing would drop them.
                System.Threading.Interlocked.Add(ref _supersededStateBytes, -budget);

                GC.Collect(2, GCCollectionMode.Forced, blocking: true);
                GC.WaitForPendingFinalizers();

                // Did every checkpoint we handed back, bar the newest, survive this collection? If
                // so the caller is holding its checkpoints, this collection freed nothing, and the
                // next one would not either. The newest is skipped because it is this step's own
                // input; two survivors are the fewest that distinguish keeping them all from
                // keeping one.
                int judged = 0, survived = 0;
                for (int i = 1; i < _watches.Length; i++)
                {
                    if (_watches[i] is not { } watch) continue;
                    judged++;
                    if (watch.TryGetTarget(out _)) survived++;
                }
                bool retained = judged >= 2 && survived == judged;

                for (int i = _watches.Length - 1; i > 0; i--) _watches[i] = _watches[i - 1];
                _watches[0] = new WeakReference<TrainingCheckpoint>(produced);

                System.Threading.Interlocked.Exchange(ref _reclaimBudgetBytes,
                    retained ? Math.Min(budget * 2, MaxReclaimBudgetBytes)
                             : System.Threading.Interlocked.Read(ref _baseReclaimBudgetBytes));
            }
        }

        /// <summary>Backend bytes the tensor fields of <paramref name="structs"/> that are still
        /// alive hold. A field whose size is not derivable — a dtype with no fixed byte stride, a
        /// shape with no element count, or a sequence or optional rather than a tensor — contributes
        /// nothing, which only makes the budget conservative.</summary>
        private static long SupersededBytes(params IEnumerable<IData>[] structs)
        {
            long total = 0;
            foreach (var s in structs)
                foreach (var field in s)
                    total += field switch
                    {
                        TensorDataStruct nested => SupersededBytes(nested),
                        TensorData { IsDisposed: false } tensor => tensor.ByteCount,
                        _ => 0,
                    };
            return total;
        }

        /// <summary>
        /// Test hook: invoked on the step's execution path, in the scope the allocation-failure
        /// wrap covers, so a test can raise the backend's own failure text without the memory
        /// conditions that produce it. Thread-scoped, so a hook installed by one parallel test is
        /// invisible to every other thread; still reset it in a <c>finally</c>.
        /// </summary>
        [ThreadStatic]
        internal static Action? StepFaultInjection;

        /// <summary>
        /// The tensor state a training step holds resident, by section — what the step path already
        /// knows and an allocation failure never said (Shorokoo/Shorokoo#330). Best-effort
        /// throughout: a field whose size is not derivable contributes an unknown size rather than
        /// failing the report, since this runs only while a failure is already being raised.
        /// </summary>
        internal static List<TensorInventorySection> StepTensorInventory(
            TrainingCheckpoint checkpoint,
            TensorDataStruct? trainingInput = null,
            TensorDataStruct? trainingOutput = null)
        {
            var sections = new List<TensorInventorySection>(5);
            Add("trainable parameters", checkpoint.TrainableParams);
            Add("model state", checkpoint.ModelState);
            Add("optimizer state", checkpoint.OptimizerState);
            // The batch is state the step holds too, and for a small model on a large batch it is
            // the whole of it — a report naming only the checkpoint would point at the wrong thing.
            if (trainingInput is not null) Add("training input", trainingInput);
            if (trainingOutput is not null) Add("training target", trainingOutput);
            return sections;

            void Add(string name, TensorDataStruct? data)
            {
                var entries = new List<TensorInventoryEntry>();
                try
                {
                    foreach (var fieldDef in data?.Definition.Fields ?? [])
                    {
                        if (data!.Fields[fieldDef.Name] is not TensorData td) continue;
                        // A dtype with no storage width is reported as unknown rather than as zero
                        // bytes, which would read as an empty tensor.
                        entries.Add(new TensorInventoryEntry(
                            fieldDef.Name, td.DType.ToString(), td.Shape.Dims,
                            TensorData.StorageBits(td.DType) == 0 ? -1 : td.ByteCount));
                    }
                }
                catch { /* a partially built struct still contributes what it managed to describe */ }
                sections.Add(new TensorInventorySection(name, entries));
            }
        }

        /// <summary>
        /// One training step. <paramref name="retainStateOnDevice"/> leaves the three state outputs
        /// (params, model state, optimizer state) in the execution provider's own memory instead of
        /// fetching them back to the host, so the next step feeds them without crossing the bus —
        /// the returned checkpoint's tensors are then <b>not</b> host-readable, which is why only
        /// <see cref="ResidentTrainingRun"/>, which owns their lifetime, ever passes <c>true</c>.
        /// The loss is never retained: it is a scalar the host reads every step either way.
        /// <paramref name="call"/> is how a message about the run names it, <c>TrainStep</c> where
        /// none is given.
        /// </summary>
        private TrainingCheckpoint RunStep(
            TrainingCheckpoint checkpoint,
            IData? hyperparams,
            IData trainingInput,
            IData trainingOutput,
            bool retainStateOnDevice = false,
            StepCall? call = null)
        {
            call ??= TrainStepCall;
            if (checkpoint is null) throw new ArgumentNullException(nameof(checkpoint));
            if (trainingInput is null) throw new ArgumentNullException(nameof(trainingInput));
            if (trainingOutput is null) throw new ArgumentNullException(nameof(trainingOutput));
            if (HyperparameterStructDef.Fields.Length > 0 && hyperparams is null)
                throw new ArgumentNullException(nameof(hyperparams),
                    "This rig was built with dynamic hyperparameters; supply their values each step " +
                    "(see TrainingRig.MakeHyperparameters).");
            var inputStruct = TrainingFeeds.StructOf(trainingInput, nameof(trainingInput));
            var targetStruct = TrainingFeeds.StructOf(trainingOutput, nameof(trainingOutput));
            var hyperStruct = hyperparams is null ? null : TrainingFeeds.StructOf(hyperparams, nameof(hyperparams));
            // Before anything is fed: the runtime refuses what does not fit only once the step has
            // taken what it was fed as it is, the checkpoint among it.
            RequireBatchFits(inputStruct, targetStruct, nameof(trainingInput), nameof(trainingOutput));
            if (HyperparameterStructDef.Fields.Length > 0)
                TrainingFeeds.RequireFits(
                    hyperStruct!, HyperparameterStructDef, nameof(hyperparams), "rig.MakeHyperparameters(...)");

            // Execute the training step graph.
            // Graph inputs (after lowering): [param_fields..., state_fields..., opt_state_fields..., hyperparam_fields..., counter_inputs..., model_input_fields..., target_fields...]
            // Each struct contributes its fields in order (AddStruct); an empty struct contributes
            // none. The hyperparams input slot exists only when the rig has
            // schedule-less runtime hyperparameters; the int64 counter inputs {step, epoch, batchIndex}
            // exist only for those a scheduled hyperparameter consumes, and are fed the checkpoint's
            // current counters so the scheduler math resumes correctly from a saved checkpoint.
            //
            // How each is fed is the caller's, as for any run: the checkpoint's state as its
            // FeedMode says, the batch and the hyperparameters as they were passed -- an initial
            // checkpoint included, whose tensors are copies of the rig's own -- and a field built
            // into its struct with a mode of its own as that says too. The counters are the rig's
            // business, built here for this step alone (consumed).
            //
            // Each feed is labelled with what the caller passed it as, which is what a message about
            // it -- the one a tensor this step consumed throws, say -- has to name: the graph's own
            // names for the state inputs are internal identifiers.
            var hyperparameters = HyperparameterStructDef.Fields.Length > 0 ? hyperStruct! : null;
            var labels = StepLabels(checkpoint, hyperparameters, inputStruct, targetStruct);
            var execInputs = new List<IData>(labels.Count);
            AddStruct(execInputs, checkpoint.TrainableParams, checkpoint.FeedMode);
            AddStruct(execInputs, checkpoint.ModelState, checkpoint.FeedMode);
            AddStruct(execInputs, checkpoint.OptimizerState, checkpoint.FeedMode);
            if (hyperparameters is not null) AddStruct(execInputs, hyperparameters, TrainingFeeds.ModeOf(hyperparams!));
            foreach (var counter in _counterInputNames)
                execInputs.Add(Shorokoo.Globals.TensorData(Array.Empty<long>(), CounterValue(checkpoint, counter)));
            AddStruct(execInputs, inputStruct, TrainingFeeds.ModeOf(trainingInput));
            AddStruct(execInputs, targetStruct, TrainingFeeds.ModeOf(trainingOutput));
            // A loss that ignores its target leaves TargetDef empty, so the struct above contributes
            // no field — the rig supplies the dead input's value itself (Shorokoo/Shorokoo#331). Target
            // fields come last in the layout, so it goes here. It is the rig's, and fed every step:
            // read, never consumed.
            if (_ignoredTargetPlaceholder is not null) execInputs.Add(_ignoredTargetPlaceholder.Shared());
            var expandedInputs = execInputs.ToArray();
            var compiled = CompiledTrainStepFor(expandedInputs);
            var stateOutputCount =
                UpdatedParamFieldCount + UpdatedStateFieldCount + UpdatedOptimizerStateFieldCount;
            NamedModelParam[] results;
            try
            {
                StepFaultInjection?.Invoke();
                if (retainStateOnDevice && compiled.HasDeviceMemory)
                {
                    // Every output but the trailing loss is state the next step feeds straight back.
                    // Sized from the session's own outputs, not from the field counts: the release
                    // loop below already allows more outputs than state plus loss, and Execute refuses
                    // a retention array of any other length -- so deriving it twice would fail the GPU
                    // path on a graph the CPU path runs fine.
                    var retain = new bool[compiled.OutputCount];
                    for (int i = 0; i < stateOutputCount; i++) retain[i] = true;
                    results = compiled.Execute(expandedInputs, labels, retain, call.Description);
                }
                else
                {
                    results = compiled.Execute(expandedInputs, labels, retainOnDevice: null, call.Description);
                }
            }
            catch (Exception ex) when (AllocationFailureReport.IsAllocationFailure(ex))
            {
                // The backend reports an allocation abort as bare text — an arena source path and
                // a request size, or the two words "bad allocation" — with no pool, no host/device
                // indication, and the same managed stack either way (Shorokoo/Shorokoo#330), so
                // nothing separates a full accelerator from a process that may not commit any more
                // host memory (Shorokoo/Shorokoo#332). Everything needed to tell them apart is
                // already here: the step's own resident state, whether this session even has device
                // memory to exhaust, and this process's commit against any limit in force.
                string report;
                try
                {
                    // DeviceMemory.Read() asks the card itself where a CUDA runtime is installed and
                    // answers null otherwise, so "the accelerator is full" stops being an inference
                    // (Shorokoo/Shorokoo#332, Shorokoo/Shorokoo#347). HasDeviceMemory is the
                    // session's own answer to whether there is device memory to exhaust, and one
                    // value feeds both the classification and the wording built on it.
                    // The arena cap is the one this session was BUILT with: under a device-memory
                    // budget, what the budget left it once the tensors the context holds on the card
                    // were counted. Reading LimitBytes off the context would report the budget,
                    // which is a cap the failing session never had.
                    var device = new DeviceFacts(
                        compiled.HasDeviceMemory, DeviceMemory.Read(), compiled.DeviceMemory.LimitBytes,
                        AllocationFailureReport.BackendAssemblyName());
                    report = AllocationFailureReport.Render(
                        $"{call.Step} at step {checkpoint.Step}",
                        AllocationFailureReport.Classify(ex, device.HasDeviceMemory),
                        device,
                        StepTensorInventory(checkpoint, inputStruct, targetStruct),
                        AllocationFailureReport.ReadProcessMemory(),
                        ex.Message);
                }
                catch
                {
                    // Building the report allocates, under exactly the condition that just refused
                    // an allocation. If that fails, the backend's own diagnosis is still worth more
                    // than anything this could add, so it leaves unchanged rather than replaced.
                    throw ex;
                }
                throw new ComputeContextException(ErrorCodes.CR009, call.Operation, report, ex);
            }
            finally
            {
                // However the step ended. A batch it only read stays the caller's, but the copies
                // made to read it -- on a card, the batch copied onto the card -- need not: see
                // ReleaseReadCopies.
                ReleaseReadCopies(inputStruct);
                ReleaseReadCopies(targetStruct);
            }

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
            // Read the loss through the rooted accessor, not a bare span, and release everything
            // past the state outputs here: the state is the resident run's to own or the
            // checkpoint's to carry, the rest is a step's worth of outputs nobody keeps.
            var lossTensor = results[stateOutputCount].ToTensorData<float32>();
            var lossValue = lossTensor.ValueAt<float>(0);
            for (int i = stateOutputCount; i < results.Length; i++)
                results[i].ToTensorData().Dispose();

            // Step is the graph-advanced counter (one training step per call). Epoch and batch
            // index are host-owned — the training loop advances them — so they carry through
            // unchanged here.
            var newCheckpoint = new TrainingCheckpoint
            {
                TrainableParams = updatedParams,
                ModelState = updatedModelState,
                OptimizerState = updatedOptimizerState,
                Step = checkpoint.Step + 1,
                Epoch = checkpoint.Epoch,
                BatchIndex = checkpoint.BatchIndex,
                Rig = this,
                Loss = lossValue,
            };

            // Only state this step superseded and left alive: what it consumed is released already.
            // A retained step's state belongs to the ResidentTrainingRun, which consumes it with the
            // next step or releases it itself, so counting it here would buy a forced blocking gen-2
            // collection per step -- on a model whose state crosses the budget every step, that is a
            // full-heap collection with nothing to collect, in the loop whose whole point is that
            // per-step overhead dominates.
            if (!retainStateOnDevice)
                ReclaimSupersededState(
                    SupersededBytes(checkpoint.TrainableParams, checkpoint.ModelState, checkpoint.OptimizerState),
                    newCheckpoint);

            return newCheckpoint;
        }

        /// <summary>
        /// Adds <paramref name="fed"/>'s fields to a step's inputs in its definition's order, each fed
        /// as <paramref name="mode"/> — the mode the struct was passed in, or the checkpoint's
        /// <see cref="TrainingCheckpoint.FeedMode"/> for its state; null is as it is, and consumed —
        /// combines with any mode the field was given when the struct was built
        /// (<see cref="TensorDataStruct.FieldFeedMode"/>). No checkpoint holds the rig's own initial
        /// values (<see cref="CopiesOf"/>), so every field of state is the caller's to spend.
        /// </summary>
        private static void AddStruct(List<IData> inputs, TensorDataStruct fed, SharedInputMode? mode)
        {
            foreach (var field in fed.Definition.Fields)
            {
                var value = fed.Fields[field.Name];
                inputs.Add(fed.FieldFeedMode(field.Name, mode) is { } fieldMode ? new SharedInput(value, fieldMode) : value);
            }
        }

        /// <summary>
        /// What a message about a training step calls each of its inputs, in the order the step feeds
        /// them: what the caller passed each as — "the checkpoint's trainable parameter 'w'", "the
        /// training input 'x'" — since the step graph's own names for them are internal identifiers.
        ///
        /// <para>Only a message reads them. So they are worked out once per rig for structs whose
        /// fields are named as the rig's own definitions name them — every struct the rig builds, and
        /// every batch built from <see cref="InputDef"/> or <see cref="TargetDef"/> — and afresh, from
        /// the names the caller's structs give their fields, for any other. The hyperparameters are
        /// null where the rig has none to feed.</para>
        /// </summary>
        private IReadOnlyList<string> StepLabels(
            TrainingCheckpoint checkpoint, TensorDataStruct? hyperparameters, TensorDataStruct input, TensorDataStruct target)
        {
            if (NamedAs(checkpoint.TrainableParams, TrainableParamStructDef)
                && NamedAs(checkpoint.ModelState, ModelStateDef)
                && NamedAs(checkpoint.OptimizerState, OptimizerStateDef)
                && (hyperparameters is null || NamedAs(hyperparameters, HyperparameterStructDef))
                && NamedAs(input, InputDef)
                && NamedAs(target, TargetDef))
                return _stepLabels ??= LabelsFor(TrainableParamStructDef, ModelStateDef, OptimizerStateDef,
                    hyperparameters is null ? null : HyperparameterStructDef, InputDef, TargetDef);
            return LabelsFor(checkpoint.TrainableParams.Definition, checkpoint.ModelState.Definition,
                checkpoint.OptimizerState.Definition, hyperparameters?.Definition, input.Definition, target.Definition);
        }

        /// <summary><see cref="StepLabels"/> for structs of these definitions.</summary>
        private string[] LabelsFor(
            TensorStructDef parameters, TensorStructDef modelState, TensorStructDef optimizerState,
            TensorStructDef? hyperparameters, TensorStructDef input, TensorStructDef target)
        {
            var labels = new List<string>();
            void Section(TensorStructDef def, string section)
            {
                foreach (var field in def.Fields) labels.Add($"{section} '{field.Name}'");
            }

            Section(parameters, "the checkpoint's trainable parameter");
            Section(modelState, "the checkpoint's model state");
            Section(optimizerState, "the checkpoint's optimizer state");
            if (hyperparameters is not null) Section(hyperparameters, "the hyperparameter");
            foreach (var counter in _counterInputNames) labels.Add($"the '{counter}' counter");
            Section(input, "the training input");
            Section(target, "the training target");
            if (_ignoredTargetPlaceholder is not null) labels.Add("the rig's stand-in for the target its loss ignores");
            return [.. labels];
        }

        /// <summary>Whether <paramref name="fed"/>'s fields are named, in order, as
        /// <paramref name="own"/>'s are — which they are by reference for every struct the rig
        /// builds.</summary>
        private static bool NamedAs(TensorDataStruct fed, TensorStructDef own)
        {
            if (ReferenceEquals(fed.Definition, own)) return true;
            var fields = fed.Definition.Fields;
            if (fields.Length != own.Fields.Length) return false;
            for (int i = 0; i < fields.Length; i++)
                if (fields[i].Name != own.Fields[i].Name) return false;
            return true;
        }

        /// <summary>Refuses a batch that does not fit <see cref="InputDef"/> and <see cref="TargetDef"/>,
        /// before anything is fed; see <see cref="TrainingFeeds.RequireFits"/>.</summary>
        private void RequireBatchFits(TensorDataStruct input, TensorDataStruct target, string inputName, string targetName)
        {
            TrainingFeeds.RequireFits(input, InputDef, inputName, "rig.InputDef.FromOrderedData(...)");
            TrainingFeeds.RequireFits(target, TargetDef, targetName, "rig.TargetDef.FromOrderedData(...)");
        }

        /// <summary>
        /// Retires the copies runs made of <paramref name="batch"/>'s tensors and sequences to read
        /// them — every field, nested structs and present optionals included — once the step that
        /// read them has returned.
        ///
        /// <para>A run reads memory it cannot address — a host tensor on a card, and every managed
        /// array on any backend — through a copy the tensor holds and reuses until it is written or
        /// dies, which is right for a tensor read again and again. A batch is read once a step: a
        /// dataset fed <c>.Shared()</c> epoch after epoch would otherwise keep a copy of every batch
        /// in the run's memory for as long as the dataset lives — on a card, through an allocator
        /// that never shrinks, every batch of it on the card at once. So a step lets go of them as it
        /// returns, and the next read of the batch copies it again. What the step consumed has no
        /// copies left to retire, and a copy another run is still reading goes when that run
        /// returns. The checkpoint's state is not a batch: a checkpoint fed <c>.Shared()</c> to
        /// <c>TrainStep</c> may be read step after step, and keeps its copies -- except by a
        /// <see cref="ResidentTrainingRun"/>, which reads a checkpoint it does not own for one step
        /// only and lets its copies go after it (<see cref="ReleaseStateReadCopies"/>).</para>
        /// </summary>
        private static void ReleaseReadCopies(TensorDataStruct batch)
        {
            foreach (var field in batch)
                switch (field)
                {
                    case TensorData tensor: tensor.ReleaseRunCopies(); break;
                    case TensorDataSequence sequence: sequence.ReleaseRunCopies(); break;
                    case OptionalTensorData { Value: { } present }: present.ReleaseRunCopies(); break;
                    case TensorDataStruct nested: ReleaseReadCopies(nested); break;
                }
        }

        // ---- Device-resident training (Shorokoo/Shorokoo#325) ----

        /// <summary>
        /// Starts a <see cref="ResidentTrainingRun"/>: a training loop that leaves its state —
        /// parameters, model state, optimizer state — where the execution provider produced it
        /// instead of moving the whole of it through host memory on every step. Use it wherever a
        /// manual <c>TrainStep</c> loop would go; on a GPU it is the difference between a step that
        /// costs the model's arithmetic and one that costs its parameter count.
        ///
        /// <para><paramref name="initialCheckpoint"/> defaults to
        /// <see cref="CreateInitialCheckpoint()"/>. It is fed to the run's first step the way
        /// <c>TrainStep</c> feeds a checkpoint: as it is, that step consumes its state; passed
        /// <c>.Shared()</c>, the run reads it and it stays the caller's, whole. (An initial
        /// checkpoint's tensors are copies of the rig's values, so a run begun from
        /// <see cref="CreateInitialCheckpoint()"/> consumes them and takes nothing of the rig's.)
        /// Dispose the run when the loop ends — anything it still holds goes with it, so take the
        /// checkpoint you want to keep with
        /// <see cref="ResidentTrainingRun.StepToCheckpoint(IData, IData)"/> first.</para>
        ///
        /// <para><b>Steps write the new state over the old.</b> A step's updated state is written
        /// into the memory of the state it consumed wherever the step's graph proves nothing reads
        /// the old value after the new one is written, as an optimizer's element-wise update allows,
        /// so on a card such state is held once rather than twice. A run begun from a fresh initial
        /// checkpoint does this from its first step.</para>
        /// </summary>
        public ResidentTrainingRun BeginResidentRun(TrainingCheckpoint? initialCheckpoint = null)
            => new(this, initialCheckpoint ?? CreateInitialCheckpoint());

        /// <summary>
        /// One step of a <see cref="ResidentTrainingRun"/> on caller-supplied data. Applies the same
        /// no-runtime-hyperparameter guard the schedule-driven <c>TrainStep</c> does when
        /// <paramref name="hyperparams"/> is absent, so a rig that needs values still says so.
        /// </summary>
        internal TrainingCheckpoint ResidentStep(
            TrainingCheckpoint checkpoint,
            IData? hyperparams,
            IData trainingInput,
            IData trainingOutput,
            bool retain)
        {
            if (hyperparams is null) RequireNoRuntimeHyperparameters();
            return RunStep(checkpoint, hyperparams, trainingInput, trainingOutput, retain, ResidentStepCall);
        }

        /// <summary>One step of a <see cref="ResidentTrainingRun"/> on an already-drawn batch; see
        /// <see cref="BatchStep"/>.</summary>
        internal TrainingCheckpoint ResidentBatchStep(
            TrainingCheckpoint checkpoint, DataBatch batch, bool retain)
            => BatchStep(checkpoint, batch, retain, ResidentStepCall);

        /// <summary>
        /// One step on an already-drawn batch, and the one place the loader-step-and-counter
        /// semantics live: every loader-driven form — <c>TrainStep(loader)</c>,
        /// <see cref="Fit(IDataLoader, int, TrainingCheckpoint?)"/> and
        /// <see cref="ResidentTrainingRun"/> — steps through here. The batch is drawn by the caller
        /// because <c>Fit</c> must see the loader's position after the draw to know whether this is
        /// the step to bring the state home on.
        /// </summary>
        private TrainingCheckpoint BatchStep(
            TrainingCheckpoint checkpoint, DataBatch batch, bool retain, StepCall call)
        {
            if (checkpoint is null) throw new ArgumentNullException(nameof(checkpoint));
            RequireNoRuntimeHyperparameters();
            // The batch's own position drives the scheduler counters for THIS step (a scheduler
            // reading epoch / batchIndex sees the batch being trained) AND is recorded on the
            // returned checkpoint (the unified "batch used" convention). RunStep carries those
            // counters through unchanged and advances Step, preserving the attached rig and this
            // step's loss — so a later Fit(loader) resumes past this batch via RestoreAfter.
            var stepInput = checkpoint.WithCounters(
                epoch: batch.Position.Epoch, batchIndex: batch.Position.BatchIndex);
            return RunStep(stepInput, hyperparams: null, batch.Input, batch.Target, retain, call);
        }

        /// <summary>The guard the schedule-driven step paths share: a schedule-less runtime
        /// hyperparameter has no value to apply automatically, so it must be supplied.</summary>
        private void RequireNoRuntimeHyperparameters()
        {
            if (HyperparameterStructDef.Fields.Length > 0)
                throw new InvalidOperationException(
                    $"This rig has schedule-less runtime hyperparameter(s) " +
                    $"[{string.Join(", ", DynamicHyperparameterNames)}] with no schedule to apply " +
                    "automatically; supply their values via MakeHyperparameters and the " +
                    "TrainStep(checkpoint, hyperparams, …) overload.");
        }

        /// <summary>
        /// Frees the backend tensors behind a checkpoint's state. Only a
        /// <see cref="ResidentTrainingRun"/> calls this, and only for state it produced and nothing
        /// else can still be holding that no step went on to consume — what it still holds when it
        /// is disposed — since a device allocation left behind would otherwise sit on the card until
        /// a finalizer ran, which on a training loop that allocates almost nothing managed is far too
        /// late. A tensor a step already consumed is dead, and ending it again does nothing.
        /// </summary>
        internal static void ReleaseCheckpointState(TrainingCheckpoint checkpoint)
        {
            ReleaseStructFields(checkpoint.TrainableParams);
            ReleaseStructFields(checkpoint.ModelState);
            ReleaseStructFields(checkpoint.OptimizerState);
        }

        /// <summary>
        /// Retires the copies runs made of <paramref name="checkpoint"/>'s state to read it, as
        /// <see cref="ReleaseReadCopies"/> does a batch's. Only a <see cref="ResidentTrainingRun"/>
        /// calls this, for a checkpoint it read and does not own -- the one it began from, or one
        /// it handed out -- once the step that read it has moved the run on to state of its own.
        /// </summary>
        internal static void ReleaseStateReadCopies(TrainingCheckpoint checkpoint)
        {
            ReleaseReadCopies(checkpoint.TrainableParams);
            ReleaseReadCopies(checkpoint.ModelState);
            ReleaseReadCopies(checkpoint.OptimizerState);
        }

        private static void ReleaseStructFields(TensorDataStruct fields)
        {
            // Release through the tensor, not the value behind it. Disposing the backing value
            // directly leaves the tensor's IsDisposed false over a freed buffer, so every later
            // read sails past ThrowIfDisposed and hands out a span over released memory instead
            // of throwing -- which turns any ownership slip here into a wild read rather than an
            // ObjectDisposedException. A nested struct is state too, and would otherwise be the
            // one thing a resident run never frees.
            foreach (var field in fields.Fields.Values)
                switch (field)
                {
                    case TensorData tensor: tensor.Dispose(); break;
                    case TensorDataStruct nested: ReleaseStructFields(nested); break;
                    // Conservative is harmless where the budget counts and wrong here: this is the
                    // one place that frees, so a kind it does not know is state nobody releases.
                    default: throw new NotSupportedException(
                        $"Cannot release checkpoint state field of type {field.GetType().Name}.");
                }
        }

        /// <summary>
        /// Runs a full training loop over the training data for the specified number of epochs.
        /// Each element in the input/output arrays represents one training step (typically a pre-batched batch).
        ///
        /// <para>The arrays are a dataset, fed once per epoch, so every step <b>reads</b> its batch —
        /// as if it were passed <c>.Shared()</c> — and the batches are all alive and unchanged when
        /// this returns. Each step lets go of the copies it made to read its batch, so the dataset is
        /// not held a second time in the run's memory — on a card, not all on the card at once. The
        /// checkpoint is fed to the first step as <c>TrainStep</c> feeds one: consumed as it is, read
        /// when passed <c>.Shared()</c>.</para>
        /// </summary>
        /// <param name="initialCheckpoint">Initial training state (with initial parameter values)</param>
        /// <param name="trainingInputs">Array of training input batches (each as TensorDataStruct)</param>
        /// <param name="trainingOutputs">Array of training target batches (each as TensorDataStruct)</param>
        /// <param name="numEpochs">Number of passes over the training data</param>
        /// <returns>Training result with final checkpoint and per-epoch average losses</returns>
        public TrainingResult Train(
            TrainingCheckpoint initialCheckpoint,
            TensorDataStruct[] trainingInputs,
            TensorDataStruct[] trainingOutputs,
            int numEpochs)
        {
            if (initialCheckpoint is null) throw new ArgumentNullException(nameof(initialCheckpoint));
            if (trainingInputs is null) throw new ArgumentNullException(nameof(trainingInputs));
            if (trainingOutputs is null) throw new ArgumentNullException(nameof(trainingOutputs));
            if (trainingInputs.Length != trainingOutputs.Length)
                throw new ArgumentException("Training inputs and outputs must have the same length.");
            if (numEpochs < 1) throw new ArgumentException("Number of epochs must be at least 1.", nameof(numEpochs));

            // The step body runs against the rig's lazily-compiled, cached trainstep (compiled via
            // RuntimeContext, per fed input shape), so a Fit()/Train() loop and a manual TrainStep loop
            // share the rig's compiled graphs.
            RequireNoRuntimeHyperparameters();

            // Every batch before the first step takes anything. Each step refuses one that does not
            // fit anyway, but by then the steps before it have consumed the checkpoint the run began
            // from.
            for (int i = 0; i < trainingInputs.Length; i++)
                RequireBatchFits(trainingInputs[i], trainingOutputs[i],
                    $"{nameof(trainingInputs)}[{i}]", $"{nameof(trainingOutputs)}[{i}]");

            // The loop owns every intermediate state and returns only the last, so it trains through a
            // resident run: the state stays where the provider produced it and crosses to the host on
            // the final step alone, which is the only one whose checkpoint anybody sees
            // (Shorokoo/Shorokoo#325).
            using var run = BeginResidentRun(initialCheckpoint);
            var checkpoint = initialCheckpoint;
            var epochLosses = new float[numEpochs];

            for (int epoch = 0; epoch < numEpochs; epoch++)
            {
                float epochLoss = 0;

                for (int i = 0; i < trainingInputs.Length; i++)
                {
                    // Read, not consumed: the same batch is fed again next epoch.
                    var (input, target) = (trainingInputs[i].Shared(), trainingOutputs[i].Shared());
                    bool last = epoch == numEpochs - 1 && i == trainingInputs.Length - 1;
                    if (last)
                    {
                        checkpoint = run.StepToCheckpoint(input, target);
                        epochLoss += checkpoint.Loss!.Value;
                    }
                    else
                    {
                        epochLoss += run.Step(input, target);
                    }
                }

                epochLosses[epoch] = epochLoss / trainingInputs.Length;
            }

            return new TrainingResult(checkpoint, epochLosses);
        }

        /// <summary>
        /// Fits the model to the data for <paramref name="numEpochs"/> epochs, over a resident run
        /// that takes its one checkpoint on the final step.
        /// Scheduled hyperparameters are applied automatically (the global step advances across epochs
        /// via the checkpoint), so the schedule sees a monotonically increasing step. Delegates to
        /// <see cref="Train"/>, but the argument orders are not interchangeable: <see cref="Train"/>
        /// takes the checkpoint first and requires it, this takes it last and optional.
        /// <paramref name="initialCheckpoint"/> defaults to
        /// <see cref="CreateInitialCheckpoint()"/>, so a minimal call is
        /// <c>rig.Fit(inputs, targets, numEpochs: 10)</c>. The trainstep is compiled and run through the
        /// rig's <see cref="RuntimeContext"/> (set at construction), the single compiled graph per rig.
        ///
        /// <para>The arrays are fed as <see cref="Train"/> feeds them: they are a dataset, fed once per
        /// epoch, so every step <b>reads</b> its batch, never consumes it, and the batches are all
        /// alive and unchanged when this returns. <paramref name="initialCheckpoint"/> is fed to the
        /// first step as <c>TrainStep</c> feeds one — consumed as it is, read when passed
        /// <c>.Shared()</c> — and the default, a fresh <see cref="CreateInitialCheckpoint()"/>, is
        /// consumed by that step.</para>
        /// </summary>
        public TrainingResult Fit(
            TensorDataStruct[] trainingInputs,
            TensorDataStruct[] trainingOutputs,
            int numEpochs,
            TrainingCheckpoint? initialCheckpoint = null)
            => Train(initialCheckpoint ?? CreateInitialCheckpoint(), trainingInputs, trainingOutputs, numEpochs);

        /// <summary>
        /// Fits a rig whose loss reads no target (<see cref="HasTargets"/> is <c>false</c>) over
        /// <paramref name="trainingInputs"/> — the target-free counterpart of
        /// <see cref="Fit(TensorDataStruct[], TensorDataStruct[], int, TrainingCheckpoint?)"/>
        /// (Shorokoo/Shorokoo#331), for a model that computes its own loss. Throws when the rig's loss
        /// does read a target. Its batches are read, never consumed, and the checkpoint fed as that
        /// overload's are.
        /// </summary>
        public TrainingResult Fit(
            TensorDataStruct[] trainingInputs,
            int numEpochs,
            TrainingCheckpoint? initialCheckpoint = null)
        {
            if (trainingInputs is null) throw new ArgumentNullException(nameof(trainingInputs));
            RequireTargetless(nameof(Fit));
            var noTargets = new TensorDataStruct[trainingInputs.Length];
            for (var i = 0; i < noTargets.Length; i++) noTargets[i] = TargetDef.FromOrderedData();
            return Train(initialCheckpoint ?? CreateInitialCheckpoint(), trainingInputs, noTargets, numEpochs);
        }

        /// <summary>
        /// Fits the model by draining an <see cref="IDataLoader"/> for <paramref name="numEpochs"/>
        /// epochs, advancing the checkpoint's <see cref="TrainingCheckpoint.Step"/>,
        /// <see cref="TrainingCheckpoint.Epoch"/> and <see cref="TrainingCheckpoint.BatchIndex"/> at
        /// the right points — so the host no longer hand-sets epoch / batch: the loader owns the data
        /// stream and the rig reads the counters off it. Each produced checkpoint therefore carries a
        /// correct position — the batch that was USED — and the returned
        /// <see cref="TrainingResult.FinalCheckpoint"/> can be saved and later resumed by passing it back
        /// as <paramref name="initialCheckpoint"/>: the loader is advanced one past the checkpoint's
        /// position with <see cref="IDataLoader.RestoreAfter"/>, so the run continues from exactly the
        /// batch after the last one it trained. A fresh (or position-unknown) checkpoint instead starts
        /// at <c>(0, 0)</c> via <see cref="IDataLoader.RestoreFrom"/>.
        ///
        /// <para>"Epochs" are counted from the loader's resume epoch: the loop trains until the
        /// loader reaches <c>resumeEpoch + numEpochs</c>. Resuming a checkpoint saved mid-epoch first
        /// finishes that partial epoch (the resume position is still within it); resuming one saved at an
        /// epoch's last batch begins the next epoch. Scheduled hyperparameters are applied automatically (the global
        /// step advances across the run); this schedule-driven form requires the rig to have no
        /// schedule-less runtime hyperparameter — supply those via <see cref="MakeHyperparameters(float)"/>
        /// and a manual <see cref="TrainStep(TrainingCheckpoint, IData, IData, IData)"/>
        /// loop instead.</para>
        /// </summary>
        /// <param name="loader">The data loader owning the (input, target) batch stream and its position.</param>
        /// <param name="numEpochs">Number of additional epochs to train, counted from the loader's resume epoch.</param>
        /// <param name="initialCheckpoint">State to resume from; defaults to <see cref="CreateInitialCheckpoint()"/>.
        /// Fed to the first step as <c>TrainStep</c> feeds one: consumed as it is, read when passed
        /// <c>.Shared()</c>.</param>
        /// <returns>Final checkpoint (with advanced step / epoch / batch) and the per-epoch mean losses.</returns>
        public TrainingResult Fit(
            IDataLoader loader,
            int numEpochs,
            TrainingCheckpoint? initialCheckpoint = null)
        {
            if (loader is null) throw new ArgumentNullException(nameof(loader));
            if (numEpochs < 1) throw new ArgumentException("Number of epochs must be at least 1.", nameof(numEpochs));

            var checkpoint = initialCheckpoint ?? CreateInitialCheckpoint();

            // Resume: point the loader at the next batch to train. A checkpoint's epoch / batch now names
            // the batch that was USED, so a resuming run advances one past it via RestoreAfter (the loader
            // does the epoch rollover). A fresh checkpoint — or one whose epoch / batch is unknown (null),
            // e.g. trained without a loader — starts at (epoch 0, batch 0) via RestoreFrom.
            if (checkpoint.Epoch is long ckptEpoch && checkpoint.BatchIndex is long ckptBatch)
                loader.RestoreAfter(new DataLoaderPosition(ckptEpoch, ckptBatch));
            else
                loader.RestoreFrom(new DataLoaderPosition(0, 0));

            // The step body runs against the rig's cached trainstep (one compiled graph per rig,
            // compiled once via RuntimeContext).
            // Count epochs from the loader's live resume position (always concrete), not the checkpoint's
            // recorded "batch used" — resuming a full epoch's last batch lands the loader at the next
            // epoch's start, and numEpochs is added to THAT.
            long startEpoch = loader.Position.Epoch;
            long targetEpoch = startEpoch + numEpochs;

            var epochLosses = new List<float>();
            long runningEpoch = startEpoch;
            float epochLossSum = 0f;
            int epochBatchCount = 0;

            // Drive off the loader's live position (always concrete), through a resident run so the
            // state stays where the provider produced it across the whole fit (Shorokoo/Shorokoo#325).
            // Each step keeps the loader-step-and-counter semantics of TrainStep(loader): the batch's
            // own position feeds the scheduler and is stamped on the checkpoint the step produces.
            // The batch is drawn here rather than inside the step because only the loader's position
            // AFTER the draw says whether this is the last step — the one to bring the state home on,
            // since its checkpoint is the run's result.
            using var run = BeginResidentRun(checkpoint);
            while (loader.Position.Epoch < targetEpoch)
            {
                long batchEpoch = loader.Position.Epoch;   // the epoch of the batch about to be drawn
                var batch = loader.Next();
                bool last = loader.Position.Epoch >= targetEpoch;

                float loss = last
                    ? (checkpoint = run.StepToCheckpoint(batch)).Loss!.Value
                    : run.Step(batch);

                // Group per-epoch mean loss by the epoch the batch belonged to.
                if (batchEpoch != runningEpoch)
                {
                    epochLosses.Add(epochBatchCount > 0 ? epochLossSum / epochBatchCount : 0f);
                    runningEpoch = batchEpoch;
                    epochLossSum = 0f;
                    epochBatchCount = 0;
                }
                epochLossSum += loss;
                epochBatchCount++;
            }
            if (epochBatchCount > 0)
                epochLosses.Add(epochLossSum / epochBatchCount);

            return new TrainingResult(checkpoint, epochLosses.ToArray());
        }

        /// <summary>
        /// Returns the default initial checkpoint produced at <see cref="FromScratch(ComputationGraph, ComputationGraph, ComputationGraph, NamedModelParam[], IOptimizerHyperparameters, RngConfig?, ComputeContext?, ComputeContext?, IProgress{BuildProgress}, TrainingBackend?)"/> time.
        /// Trainable parameters and model state were initialized from the model's built-in
        /// initializers, and optimizer state from the optimizer's [StateInitializer]s (run once per
        /// trainable parameter, at each hyperparameter's value at the initial counters). Nothing is
        /// computed here; the values are copied.
        ///
        /// <para>Each call returns tensors of its own: host copies of the rig's initial values, which
        /// the rig keeps. So a checkpoint made here is fed like any other — a training step given it
        /// as it is consumes it, and its memory goes with that step (on a card, the step's outputs can
        /// be written into it) — and the rig's values, and the next initial checkpoint, are untouched.
        /// To start several runs from the same initial state, call this once per run, or pass one
        /// checkpoint <c>.Shared()</c>.</para>
        ///
        /// <para><b>Fails loud</b> when the optimizer's state initializer actually reads a
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
            return new TrainingCheckpoint
            {
                TrainableParams = new TensorDataStruct(TrainableParamStructDef, CopiesOf(InitialParamFields)),
                ModelState = new TensorDataStruct(ModelStateDef, CopiesOf(InitialStateFields)),
                OptimizerState = new TensorDataStruct(OptimizerStateDef, CopiesOf(InitialOptStateFields)),
                Rig = this,
            };
        }

        /// <summary>
        /// Like <see cref="CreateInitialCheckpoint()"/>, but with explicit initial values for the
        /// <see cref="HyperparameterKind.Runtime"/> hyperparameters (build the struct with
        /// <see cref="MakeHyperparameters(float)"/> / <see cref="MakeHyperparameters(ValueTuple{string, object}[])"/>
        /// — the same struct the per-step override <c>TrainStep</c> takes). Required when the
        /// optimizer's state initializer reads a runtime hyperparameter; harmless otherwise. Baked and
        /// scheduled hyperparameters still contribute their build-time value at the initial counters.
        /// Its tensors are its own, as that one's are.
        /// </summary>
        public TrainingCheckpoint CreateInitialCheckpoint(TensorDataStruct hyperparameters)
        {
            if (hyperparameters is null) throw new ArgumentNullException(nameof(hyperparameters));
            // Computed afresh for this checkpoint where there is optimizer state, so already its own.
            var optState = OptimizerStateDef.Fields.Length > 0
                ? ComputeInitialOptStateFields(
                    ResolveStateInitHyperValues(hyperparameters, throwOnMissingConsumed: true),
                    MergeContext, NameValueOf(InitialParamFields))
                : CopiesOf(InitialOptStateFields);
            return new TrainingCheckpoint
            {
                TrainableParams = new TensorDataStruct(TrainableParamStructDef, CopiesOf(InitialParamFields)),
                ModelState = new TensorDataStruct(ModelStateDef, CopiesOf(InitialStateFields)),
                OptimizerState = new TensorDataStruct(OptimizerStateDef, optState),
                Rig = this,
            };
        }

        /// <summary>
        /// The value each hyperparameter contributes to optimizer state init, in optimizer order:
        /// baked/scheduled hypers use their build-time value at the initial counters
        /// (<see cref="_hyperparamInitialCounterValues"/>); a runtime hyper takes its value from
        /// <paramref name="runtimeHypers"/> when supplied. A runtime hyper the state-init graph
        /// actually <b>consumes</b> must be present: its absence fails loud rather than defaulting
        /// to a placeholder. An unconsumed runtime hyper is irrelevant to state init, so it defaults to
        /// its declared dtype's zero.
        /// </summary>
        private TensorData[] ResolveStateInitHyperValues(TensorDataStruct? runtimeHypers, bool throwOnMissingConsumed)
        {
            var values = new TensorData[_hyperparamInitialCounterValues.Length];
            for (int i = 0; i < values.Length; i++)
            {
                if (_hyperparamInitialCounterValues[i] is TensorData known) { values[i] = known; continue; }

                // Runtime hyper: use the supplied value; a value the state-init graph actually consumes
                // must be present when the caller means it (throwOnMissingConsumed) — else it is an
                // internal zero placeholder used only to seed shape inference at build.
                var name = _runtimeHyperNameByOptIndex[i];
                if (runtimeHypers is not null
                    && runtimeHypers.Fields.TryGetValue(name, out var d) && d is TensorData td)
                {
                    values[i] = HyperparameterValues.ConvertTo(td, HyperparameterDTypes[i], name);
                    HyperparameterValues.AssertShape(values[i], HyperparameterShapes[i], name);
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
                    // Unconsumed runtime hyper, or a build-time shape-inference placeholder.
                    values[i] = HyperparameterValues.Zero(HyperparameterDTypes[i], HyperparameterShapes[i].Dims);
                }
            }
            return values;
        }

        /// <summary>The lookup <see cref="ComputeInitialOptStateFields"/> takes over a value family
        /// whose values are in hand.</summary>
        private static Func<string, TensorData> NameValueOf(Dictionary<string, IData> fields)
            => name => (TensorData)fields[name];

        /// <summary>
        /// The same lookup over a family a deferred build has only descriptions for: zeros of the
        /// field's declared shape, which is what a shape-seeding run of the state initializers needs.
        /// One parameter's worth at a time, and dropped as soon as the run returns.
        /// </summary>
        private static Func<string, TensorData> ZerosOf(Dictionary<string, TensorAttribute> slots)
            => name =>
            {
                var slot = slots[name];
                return TensorData.CreateFromRawBytes(
                    slot.Shape, slot.DType, new byte[slot.Shape.Count * (slot.DType.EncodingBitCount / 8)]);
            };

        /// <summary>
        /// Runs the optimizer's split-off state-init graph once per trainable parameter, binding its
        /// hyperparameter inputs to <paramref name="hyperSeeds"/> (in optimizer order), the parameter's
        /// value from <paramref name="paramValueFor"/>, and a zero gradient; returns the initial
        /// optimizer-state field values.
        ///
        /// <para>The parameter values come from a lookup rather than off the rig because a deferred
        /// build has none yet (Shorokoo/Shorokoo#327) and must seed shapes without running the
        /// initializers: it answers with zeros of the field's declared shape, and
        /// <see cref="EnsureInitialValues"/> recomputes these seeds from the real values later.</para>
        /// </summary>
        private Dictionary<string, IData> ComputeInitialOptStateFields(
            TensorData[] hyperSeeds, ComputeContext ctx, Func<string, TensorData> paramValueFor)
        {
            var fields = new Dictionary<string, IData>();
            var stateInitGraph = _optimizerStateInitGraph
                ?? throw new InvalidOperationException("Optimizer state fields exist but no state-init graph was produced.");
            var statesPerParam = OptimizerStateDef.Fields.Length / TrainableParamStructDef.Fields.Length;

            for (var paramIdx = 0; paramIdx < TrainableParamStructDef.Fields.Length; paramIdx++)
            {
                var paramData = paramValueFor(TrainableParamStructDef.Fields[paramIdx].Name);
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
        /// that are actually reachable from its outputs — the dependency analysis over the optimizer
        /// state-init graph, whose leading inputs are the hyperparameters (then param, grad). Shares
        /// <see cref="Shorokoo.Core.Training.TrainingGraphBuilder"/>'s scope-aware walk with the
        /// target-reachability question, so the two cannot disagree about what "reaches" means: a
        /// hyperparameter read only as a branch condition is consumed, and must be supplied.
        /// </summary>
        private static HashSet<int> ConsumedInputIndices(InternalComputationGraph graph, int count)
            => Shorokoo.Core.Training.TrainingGraphBuilder.ConsumedInputIndices(graph, count);

        /// <summary>
        /// Loads a checkpoint previously written by <see cref="TrainingCheckpoint.Save(string, CheckpointComponents?)"/>
        /// (the flat safetensors file), reconstructing it
        /// against this rig's parameter/state struct definitions so training resumes exactly where it
        /// left off: trainable params, optimizer moments, model state, and the host-owned run counters
        /// (global step, epoch, batch index) are all restored (schedules resume from that step; older
        /// checkpoints lacking epoch/batch restore them as null, an unknown position). Throws if the
        /// file's fields don't match this
        /// rig — e.g. a checkpoint produced by a different model or optimizer. The rig must be built
        /// from the same model/loss/optimizer graphs as the one that saved the checkpoint. This entry
        /// point reads the flat shape only: handed a native <c>.skpt</c> container it fails
        /// immediately, naming <see cref="LoadCheckpointFromSkpt"/> as the entry point for that shape.
        /// </summary>
        public TrainingCheckpoint LoadCheckpoint(string filePath, CheckpointComponents? components = null)
            => TrainingCheckpoint.Load(filePath, this, components);

        /// <summary>
        /// Loads a checkpoint previously written by <see cref="Persistence.SaveTrainingCheckpointToSkpt"/>
        /// (the native <c>.skpt</c> container) — the <c>.skpt</c> counterpart of
        /// <see cref="LoadCheckpoint"/>, with the same resume semantics and fail-loud contract.
        /// This entry point reads the container shape only: handed a flat safetensors checkpoint it
        /// fails immediately, naming <see cref="LoadCheckpoint"/> as the entry point for that shape.
        /// To rebuild the whole rig from a <c>.skpt</c> alone (no pre-existing rig), use the static
        /// <see cref="Load(string, ComputeContext?, ComputeContext?, IProgress{BuildProgress}, TrainingBackend?)"/> instead.
        /// </summary>
        public TrainingCheckpoint LoadCheckpointFromSkpt(string filePath, CheckpointComponents? components = null)
            => TrainingCheckpoint.LoadFromSkpt(filePath, this, components);

        // ───────── Constituent persistence & from-file reconstruction (#115/#106) ─────────
        // A training .skpt stores the rig's constituents as ordinary models/ entries so a fresh process
        // rebuilds the whole rig — trainstep and all — from the file alone. Save reads the graphs and
        // recipe off these members; Load (static, below) reads them back and re-derives via the same
        // DeriveFromConcreteArch path a fresh build uses.

        /// <summary>The rig's concrete-architecture constituent (value-less), the substrate a from-file
        /// reconstruction re-derives the trainstep from. Environment-independent; serialized as
        /// the checkpoint's <c>model-arch</c> constituent entry.</summary>
        internal ComputationGraph ConcreteArchConstituent => new(_concreteArch, GraphKind.ConcreteArchitecture);

        /// <summary>
        /// Composes the per-hyperparameter scheduler graphs (one pure <c>counters → value</c> graph per
        /// scheduled hyperparameter) into ONE scheduler model — the union of the counter inputs they
        /// consume, one named output per scheduled hyperparameter (named by the hyperparameter) — for
        /// persistence as the checkpoint's <c>scheduler</c> constituent entry (#106). Returns a null
        /// graph when no hyperparameter is scheduled. Split back to per-hyperparameter bindings on load
        /// by <see cref="SplitSchedulerOutput"/>.
        /// </summary>
        internal (ComputationGraph? Graph, IReadOnlyList<string> ScheduledNames) BuildComposedSchedulerModel()
        {
            var hyperparameters = _constituents.Hyperparameters;
            var names = _constituents.Names;
            string NameOf(int h) => names is not null && h < names.Count ? names[h] : $"hyperparam_{h}";

            var scheduledIndices = new List<int>();
            for (int h = 0; h < hyperparameters.Length; h++)
                if (hyperparameters[h].Kind == HyperparameterKind.Scheduled) scheduledIndices.Add(h);
            if (scheduledIndices.Count == 0)
                return (null, Array.Empty<string>());

            // Build each scheduler graph and collect the union of the counter inputs they consume.
            var builtByIndex = new Dictionary<int, SchedulerGraph>(scheduledIndices.Count);
            var needed = new HashSet<string>(StringComparer.Ordinal);
            foreach (var h in scheduledIndices)
            {
                var built = BuildSchedulerModule(
                    hyperparameters[h], NameOf(h), HyperparameterDTypes[h],
                    HyperparameterShapes[h].Dims.Length, MergeContext);
                builtByIndex[h] = built;
                foreach (var c in built.CounterNames) needed.Add(c);
            }

            // One shared int64 scalar input per needed counter, in canonical order.
            var composed = new InternalComputationGraph();
            var counterKeyByName = new Dictionary<string, FastTensorKey>(StringComparer.Ordinal);
            foreach (var cn in CounterInputNames)
            {
                if (!needed.Contains(cn)) continue;
                var node = Shorokoo.Core.Nodes.Processors.Fast.FastInternalOp.RuntimeInput(DType.Int64, rank: 0, cn);
                composed.Nodes.Add(node);
                var key = new FastTensorKey(node.Key, 0);
                counterKeyByName[cn] = key;
                composed.Inputs.Add(key);
                composed.InputUniqueNames.Add(cn);
            }

            var scheduledNames = new List<string>(scheduledIndices.Count);
            foreach (var h in scheduledIndices)
            {
                var built = builtByIndex[h];
                var mapped = built.CounterNames.Select(c => counterKeyByName[c]).ToArray();
                var replayed = Shorokoo.Core.Nodes.Processors.Fast.FastReplay.ReplayInto(composed, built.Graph, mapped);
                composed.Outputs.Add(replayed[0]);
                composed.OutputUniqueNames.Add(NameOf(h));
                scheduledNames.Add(NameOf(h));
            }

            return (new ComputationGraph(composed, GraphKind.ConcreteModel), scheduledNames);
        }

        /// <summary>
        /// Splits one hyperparameter's <c>counters → value</c> graph back out of the composed scheduler
        /// model (#106) by its output <paramref name="outputName"/>: the sub-graph reachable from that
        /// output, keeping only the counter inputs it actually consumes — a single-output scheduler
        /// module a <see cref="Hyperparameter.Scheduled(ComputationGraph)"/> binding re-inlines.
        /// </summary>
        internal static ComputationGraph SplitSchedulerOutput(ComputationGraph composedScheduler, string outputName)
        {
            var composed = composedScheduler.ToInternal().Clone();
            int oi = composed.OutputUniqueNames.IndexOf(outputName);
            if (oi < 0)
                throw new System.IO.InvalidDataException(
                    $"The composed scheduler model has no output named '{outputName}'; the checkpoint's " +
                    "scheduler constituent does not match its hyperparameter bindings.");
            var outKey = composed.Outputs[oi];

            var producerByOutput = BuildProducerByOutputMap(composed);
            var reachedKeys = new HashSet<FastTensorKey>();
            var reachedNodes = new HashSet<FastNodeKey>();
            var queue = new Queue<FastTensorKey>();
            queue.Enqueue(outKey);
            while (queue.Count > 0)
            {
                var k = queue.Dequeue();
                if (k.IsEmpty || !reachedKeys.Add(k)) continue;
                if (producerByOutput.TryGetValue(k, out var node))
                {
                    reachedNodes.Add(node.Key);
                    foreach (var (_, slots) in node.FullInputs)
                        foreach (var s in slots)
                            if (s is FastTensorKey ik && !ik.IsEmpty) queue.Enqueue(ik);
                }
            }

            var g = new InternalComputationGraph();
            foreach (var n in composed.Nodes)
                if (reachedNodes.Contains(n.Key)) g.Nodes.Add(n);
            for (int i = 0; i < composed.Inputs.Count; i++)
                if (reachedKeys.Contains(composed.Inputs[i]))
                {
                    g.Inputs.Add(composed.Inputs[i]);
                    g.InputUniqueNames.Add(i < composed.InputUniqueNames.Count ? composed.InputUniqueNames[i] : null);
                }
            g.Outputs.Add(outKey);
            g.OutputUniqueNames.Add(outputName);
            return new ComputationGraph(g, GraphKind.ConcreteModel);
        }

        /// <summary>
        /// Rebuilds a rig from its persisted constituents (#115/#106) — the concrete architecture, the
        /// loss and optimizer module graphs, the hyperparameter bindings and RNG config — with NO
        /// host-supplied source graphs. The deserialized <paramref name="concreteArch"/> is already
        /// self-describing: its <c>MODEL_TENSOR_INPUT</c> nodes carry the representative-input attribute
        /// (round-tripped as NodeProtos in the native <c>.srk</c> dialect), so the shape metadata the
        /// re-derivation's shape inference needs is read straight off the arch — no separate input-shape
        /// field is re-attached. The trainstep is re-derived exactly as a fresh build's derivation path
        /// does. The two compute contexts seed the rebuilt rig (rev 22; never persisted).
        /// </summary>
        internal static TrainingRig ReconstructFromConstituents(
            ComputationGraph concreteArch,
            ComputationGraph loss,
            ComputationGraph optimizer,
            Hyperparameter[] hyperparameters,
            IReadOnlyList<string>? hyperparameterNames,
            RngConfig rngConfig,
            ComputeContext mergeContext,
            ComputeContext runtimeContext,
            TrainingBackend trainingBackend,
            BuildProgressReporter? progress = null,
            bool deferInitialization = false)
        {
            if (concreteArch is null) throw new ArgumentNullException(nameof(concreteArch));
            if (loss is null) throw new ArgumentNullException(nameof(loss));
            if (optimizer is null) throw new ArgumentNullException(nameof(optimizer));
            if (hyperparameters is null) throw new ArgumentNullException(nameof(hyperparameters));
            if (rngConfig is null) throw new ArgumentNullException(nameof(rngConfig));

            // Own a copy so the rig's retained arch is independent of the caller's deserialized graph.
            // The arch is already self-describing (its MODEL_TENSOR_INPUT nodes carry the
            // representative-input attribute), so nothing is re-attached here. The RNG config was baked
            // into the arch's RngSeed param at the original build, so it is NOT re-applied here; it rides
            // as a constituent so the reconstructed rig re-derives identical initial values (load-time
            // defaults, optimizer-state seeding).
            // Two full walks of the largest graph in the file, so named rather than left as silence
            // before the derivation's own first report.
            progress?.Report(BuildPhase.Concretize, "ThawConcreteArchitecture");
            var archInternal = concreteArch.ToInternal().Clone();

            var constituents = new RigConstituents(
                new ComputationGraph(archInternal, GraphKind.ConcreteArchitecture),
                loss, optimizer, hyperparameters, hyperparameterNames, rngConfig);

            // Not the end of the build: Load still has the checkpoint payload to read, and owns the
            // terminal report accordingly.
            return DeriveFromConcreteArch(
                constituents, archInternal, mergeContext, runtimeContext, trainingBackend, progress,
                completesBuild: false, deferInitialization);
        }

        /// <summary>
        /// Rebuilds a whole training rig — and its resumed checkpoint — from a native <c>.skpt</c>
        /// checkpoint file ALONE (#115), with NO host-supplied model/loss/optimizer graphs: the rig's
        /// serialized constituents (concrete architecture, loss, optimizer, and the composed scheduler
        /// when present), hyperparameter bindings, and RNG config are read from the file and the
        /// in-memory <c>trainstep</c> re-derived, then the checkpoint's state is loaded against the
        /// reconstructed rig. This is the from-file-alone counterpart of
        /// <see cref="LoadCheckpoint"/> (which requires a pre-existing rig). The two compute contexts
        /// seed the rebuilt rig (rev 22; never persisted — a reloaded run gets fresh ones), each
        /// defaulting to <see cref="ComputeContext.Default"/>. Re-deriving the trainstep is most of a
        /// build — everything but the concretization, which the file's saved architecture replaces — so
        /// <paramref name="progress"/> reports this too: the file read and the checkpoint payload read
        /// included, ending complete only once the resumed checkpoint is in hand.
        /// The file must be a training <c>.skpt</c>
        /// written with the rig constituents (every training <c>.skpt</c> carries them); a flat
        /// checkpoint has no constituents to rebuild from and fails loudly — pass the rig and use
        /// <see cref="LoadCheckpoint"/> for that shape.
        /// </summary>
        /// <returns>The reconstructed rig and the checkpoint resumed against it (its
        /// <see cref="TrainingCheckpoint.Rig"/> set to the rig).</returns>
        public static (TrainingRig Rig, TrainingCheckpoint Checkpoint) Load(
            string filePath,
            ComputeContext? mergeContext = null,
            ComputeContext? runtimeContext = null,
            IProgress<BuildProgress>? progress = null,
            TrainingBackend? trainingBackend = null)
        {
            if (string.IsNullOrWhiteSpace(filePath))
                throw new ArgumentException("Checkpoint path cannot be null or empty.", nameof(filePath));

            // Load owns the whole span, so it owns the terminal report: the payload read below is the
            // build's largest I/O, and a "finished" report ahead of it would be worse than none.
            var reporter = BuildProgressReporter.For(progress);
            reporter?.Report(BuildPhase.Concretize, "ReadCheckpointFile");
            // The checkpoint below overwrites every value the model's initializers would produce, so
            // the rebuild does not run them (Shorokoo/Shorokoo#327). They are not lost: a component the
            // file does not carry still falls back to the rig's initial values, and asking for one runs
            // the initializers then.
            var rig = Persistence.ReconstructRigFromSkpt(
                filePath, mergeContext ?? ComputeContext.Default,
                runtimeContext ?? ComputeContext.Default, trainingBackend ?? TrainingBackend.Shorokoo,
                reporter, deferInitialization: true);

            reporter?.Report(BuildPhase.Initialize, "LoadCheckpointState");
            var checkpoint = rig.LoadCheckpointFromSkpt(filePath);
            reporter?.ReportComplete(BuildPhase.Initialize);
            return (rig, checkpoint);
        }

        /// <summary>
        /// Packs a single dynamic hyperparameter value into a <see cref="TensorDataStruct"/> for the
        /// explicit <see cref="TrainStep(TrainingCheckpoint, IData, IData, IData)"/>
        /// overload. Convenience for the common case of exactly one dynamic hyperparameter (e.g. the
        /// learning rate); throws if the rig has a different number. For multiple, use the named overload.
        /// The value is converted to the hyperparameter's declared dtype, failing loud if it would not
        /// survive the conversion. The struct's tensors are its own — built from the value, or copied
        /// from a tensor given — so a step that consumes the struct takes nothing of the caller's.
        /// </summary>
        public TensorDataStruct MakeHyperparameters(float value) => MakeSingleHyperparameter(value);

        /// <summary>Double-precision form of <see cref="MakeHyperparameters(float)"/>.</summary>
        public TensorDataStruct MakeHyperparameters(double value) => MakeSingleHyperparameter(value);

        /// <summary>Integer form of <see cref="MakeHyperparameters(float)"/>.</summary>
        public TensorDataStruct MakeHyperparameters(int value) => MakeSingleHyperparameter(value);

        /// <summary>64-bit integer form of <see cref="MakeHyperparameters(float)"/>.</summary>
        public TensorDataStruct MakeHyperparameters(long value) => MakeSingleHyperparameter(value);

        /// <summary>Boolean form of <see cref="MakeHyperparameters(float)"/>.</summary>
        public TensorDataStruct MakeHyperparameters(bool value) => MakeSingleHyperparameter(value);

        /// <summary>Explicitly typed form of <see cref="MakeHyperparameters(float)"/>, for a dtype with
        /// no natural C# literal (e.g. <c>float16</c>) and for a non-scalar hyperparameter; its shape
        /// must match the shape the rig was built at. The struct holds a copy of
        /// <paramref name="value"/> whatever its dtype, so <paramref name="value"/> stays yours when a
        /// step consumes the struct.</summary>
        public TensorDataStruct MakeHyperparameters(TensorData value)
            => MakeSingleHyperparameter(value ?? throw new ArgumentNullException(nameof(value)));

        private TensorDataStruct MakeSingleHyperparameter(object value)
        {
            if (HyperparameterStructDef.Fields.Length != 1)
                throw new InvalidOperationException(
                    $"MakeHyperparameters(value) requires exactly one dynamic hyperparameter; this rig has " +
                    $"{HyperparameterStructDef.Fields.Length} ([{string.Join(", ", DynamicHyperparameterNames)}]). " +
                    "Use MakeHyperparameters((name, value), …).");
            return PackHyperparams([value]);
        }

        /// <summary>
        /// Packs named dynamic hyperparameter values into a <see cref="TensorDataStruct"/> for the
        /// explicit <see cref="TrainStep(TrainingCheckpoint, IData, IData, IData)"/>
        /// overload. Every dynamic hyperparameter must be named exactly once (case-insensitive); names
        /// are those in <see cref="DynamicHyperparameterNames"/>, e.g.
        /// <c>MakeHyperparameters(("learningRate", lr), ("weightDecay", wd))</c>. Each value is a host
        /// value — a numeric or <c>bool</c> scalar, or a <see cref="TensorData"/> — fitted to that
        /// hyperparameter's declared dtype and checked against its built shape, so a rig may mix dtypes
        /// and shapes: <c>MakeHyperparameters(("learningRate", 0.1f), ("useNesterov", true),
        /// ("perGroupScale", TensorData([3L], 1f, 2f, 3f)))</c>. A <see cref="TensorData"/> given is
        /// copied into the struct whatever its dtype, so it stays yours when a step consumes the
        /// struct.
        /// </summary>
        public TensorDataStruct MakeHyperparameters(params (string name, object value)[] values)
        {
            if (values is null) throw new ArgumentNullException(nameof(values));

            var byName = new Dictionary<string, object>(StringComparer.OrdinalIgnoreCase);
            foreach (var (name, value) in values)
            {
                if (name is null) throw new ArgumentException("Hyperparameter name cannot be null.", nameof(values));
                if (!byName.TryAdd(name, value))
                    throw new ArgumentException($"Hyperparameter '{name}' was supplied more than once.", nameof(values));
            }

            var ordered = new object[HyperparameterStructDef.Fields.Length];
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

        /// <summary>
        /// Packs host values (in <see cref="HyperparameterStructDef"/> field order) into the runtime
        /// hyperparameter struct, fitting each to its field's declared dtype and checking it against the
        /// shape the rig was built at.
        ///
        /// <para>The struct owns every tensor in it, whatever it was built from: a step fed it as it
        /// is consumes them, and a tensor the caller passed in stays the caller's. Fitting a value
        /// already of the declared dtype hands back the caller's very tensor, which is copied, as a
        /// value of another dtype is by its conversion.</para>
        /// </summary>
        private TensorDataStruct PackHyperparams(object[] orderedValues)
        {
            var fields = new KeyValuePair<string, IData>[orderedValues.Length];
            for (int i = 0; i < orderedValues.Length; i++)
            {
                var field = HyperparameterStructDef.Fields[i];
                var value = HyperparameterValues.ConvertTo(
                    HyperparameterValues.Of(orderedValues[i]), field.ElementType, field.Name);
                HyperparameterValues.AssertShape(
                    value, ((TensorData)_initialHyperparamFields[field.Name]).Shape, field.Name);
                if (ReferenceEquals(value, orderedValues[i])) value = value.CopyTo(ComputeContext.Host);
                fields[i] = new KeyValuePair<string, IData>(field.Name, value);
            }
            return new TensorDataStruct(HyperparameterStructDef, fields);
        }

        /// <summary>
        /// Phase 2: read the concrete-architecture graph for initial trainable / state
        /// parameter values, run the optimizer's state initializers per trainable parameter,
        /// derive the target exemplar from <paramref name="lossGraph"/>'s own target input
        /// (<see cref="DeriveTargetExemplar"/>) against the shape-inferred prediction, and run
        /// shape inference + <see cref="MemoryAwareGraphOptimizer"/> on the lowered training-step
        /// graph.
        /// </summary>
        private void InitializeAndOptimize(
            InternalComputationGraph concreteArch,
            InternalComputationGraph lossGraph,
            ComputeContext ctx,
            RngConfig? rngConfig = null,
            BuildProgressReporter? progress = null,
            bool deferInitialization = false)
        {
            void Stage(string stage) => progress?.Report(BuildPhase.Initialize, stage);

            // Reported before the reads below, not after: until this fires the last report a caller
            // has seen names the previous phase, so a slow read here would be attributed to it.
            Stage("ReadModelParams");

            // The model inputs for shape inference are read off the concrete arch's own
            // representative-input attributes (recorded once at BuildInitialRig) — no separate
            // sample-input field. Zero-filled shapes for small inputs; shape and dtype alone for
            // large ones (QEE keeps those shape-only anyway), so no big buffer is materialized.
            var modelInputExemplars = ReadRepresentativeInputs(concreteArch);

            // Step 1: walk concreteArch's MODEL_PARAM nodes in linear order to capture
            // each one's (ModelId, isTrainable). The same linear order is what Phase 1's
            // FastReplaceTrainableParamsWithInputProcessor used to build the param /
            // state struct defs, so this ordering aligns Phase 2 values with Phase 1 fields.
            // FastInitializeModelParams runs the initializer functions and returns
            // ModelId → TensorData; reindex by our captured order for alignment.
            var trainableModelIds = new List<ModelId>();
            var stateModelIds = new List<ModelId>();
            var standInById = new Dictionary<ModelId, TensorAttribute>();
            var canDefer = deferInitialization;
            foreach (var node in concreteArch.Nodes)
            {
                if (node.OpCode != InternalOpCodes.MODEL_PARAM) continue;
                var modelIdVals = node.Attributes.GetIntsVal(OnnxOpAttributeNames.ShrkAttrLocalModelId).AssertNotNull();
                var modelId = new ModelId(modelIdVals);
                var isTrainable = node.Attributes.GetBoolVal(OnnxOpAttributeNames.ShrkAttrIsTrainable) ?? true;
                (isTrainable ? trainableModelIds : stateModelIds).Add(modelId);
                // A concrete arch declares every parameter's dtype and shape on the node itself, so a
                // deferred build has everything the pass below needs without running one initializer
                // (Shorokoo/Shorokoo#327). The RngSeed parameter at reserved ModelId [0] carries no
                // weight and FastInitializeModelParams skips it, so it is skipped here too.
                if (deferInitialization && modelIdVals is not [0])
                {
                    // Deferral needs a concrete dtype and shape to stand the parameter in with. A
                    // parameter that declares neither — no shape attribute, or one with a symbolic
                    // dimension, which names no element count — cannot be stood in for, so the build
                    // runs the initializers instead of failing. The eager path never needed the
                    // attribute, and deferral is an optimization: it may decline, but it must not
                    // narrow what a rig can be built from.
                    var dims = node.Attributes.GetLongsVal(OnnxOpAttributeNames.ShrkAttrShape);
                    if (dims is null || Array.IndexOf(dims, -1L) >= 0)
                    {
                        canDefer = false;
                        continue;
                    }
                    var dtype = node.Attributes.GetDTypeVal(OnnxOpAttributeNames.ShrkAttrDtype).AssertNotNull();
                    standInById[modelId] = RepresentativeInputFor(new Shape(dims), dtype);
                }
            }

            if (trainableModelIds.Count != TrainableParamStructDef.Fields.Length)
                throw new InvalidOperationException(
                    $"Initialized {trainableModelIds.Count} trainable params but expected " +
                    $"{TrainableParamStructDef.Fields.Length}. State: {stateModelIds.Count} vs " +
                    $"expected {ModelStateDef.Fields.Length}.");

            // Pass the concrete param infos so keyed per-parameter init actually engages:
            // FastInitializeModelParams keys init noise only when BOTH rngConfig and paramInfos
            // are non-null. Without the infos the rig would silently fall back to unkeyed
            // seeded init, ignoring the config's master seed / algorithm for the weights.
            IReadOnlyDictionary<ModelId, TensorData>? paramValuesById = null;
            if (canDefer)
            {
                // The run itself is what is deferred, not merely its bookkeeping: nothing below reads
                // a parameter's value, so the whole per-parameter initializer pass — the largest
                // single cost in a rig build, and pure waste when a checkpoint is about to overwrite
                // every value it produces — is left for EnsureInitialValues to run if anything ever
                // asks (Shorokoo/Shorokoo#327).
                Stage("DeferModelParamInitialization");
            }
            else
            {
                var paramInfos = rngConfig is null ? null : concreteArch.GetConcreteModelParamInfos();
                Stage("InitializeModelParams");
                paramValuesById = Shorokoo.Core.Nodes.Processors.Fast.FastInitializeModelParams.Process(
                    concreteArch, ctx, rngConfig, paramInfos);
                _deferredInit = null;
            }

            // Shape and dtype of every trainable-parameter and model-state field — read off the value
            // where the initializers ran, and taken from the field's stand-in description where they
            // did not. Everything below that wants only a field's description reads these, so a
            // deferred build answers with no value in hand and no buffer per parameter.
            var paramSlots = new Dictionary<string, TensorAttribute>(StringComparer.Ordinal);
            var stateSlots = new Dictionary<string, TensorAttribute>(StringComparer.Ordinal);
            _initialParamFields = new Dictionary<string, IData>();
            _initialStateFields = new Dictionary<string, IData>();
            FileFields(TrainableParamStructDef, trainableModelIds, _initialParamFields, paramSlots);
            FileFields(ModelStateDef, stateModelIds, _initialStateFields, stateSlots);

            void FileFields(
                TensorStructDef def, List<ModelId> ids,
                Dictionary<string, IData> fields, Dictionary<string, TensorAttribute> slots)
            {
                for (var i = 0; i < def.Fields.Length; i++)
                {
                    var name = def.Fields[i].Name;
                    if (paramValuesById is { } values)
                    {
                        var value = values[ids[i]];
                        fields[name] = value;
                        slots[name] = TensorAttribute.WithoutValues(value.Shape, value.DType);
                    }
                    else
                    {
                        // No value to file: the stand-in IS the field until EnsureInitialValues runs.
                        slots[name] = standInById[ids[i]];
                    }
                }
            }

            if (paramValuesById is null)
                _deferredInit = new DeferredInitialization(
                    concreteArch, ctx, rngConfig, [.. trainableModelIds], [.. stateModelIds],
                    paramSlots, stateSlots);

            // Initial optimizer state: run the optimizer's state initializers once per trainable
            // parameter, binding the optimizer's hyperparameter inputs to their value at the initial
            // counters (the single value route — baked constant, or scheduler graph evaluated via
            // QEE at build; no more hardcoded 0f for scheduler modules), the parameter's initial
            // value, and a zero gradient. The state-init graph carries the [StateInitializer]
            // functions split out of the optimizer graph by FastNormalizeOptimizerGraph.
            _initialOptStateFields = new Dictionary<string, IData>();
            if (OptimizerStateDef.Fields.Length > 0)
            {
                var stateInitGraph = _optimizerStateInitGraph
                    ?? throw new InvalidOperationException(
                        "Optimizer state fields exist but no state-init graph was produced.");

                // Which hyperparameters does the state-init graph actually consume? A runtime hyper
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
                Stage("InitializeOptimizerState");
                _initialOptStateFields = ComputeInitialOptStateFields(
                    ResolveStateInitHyperValues(null, throwOnMissingConsumed: false), ctx,
                    paramValuesById is not null ? NameValueOf(_initialParamFields) : ZerosOf(paramSlots));
            }

            // Step 2: shape-infer the model to get its prediction, then derive the target exemplar
            // from the loss's own target input (see DeriveTargetExemplar — the prediction answers
            // for it only where the loss declares nothing else). Reuse the already-computed
            // paramValuesById via FastApplyModelParamValues — this rewrites MODEL_PARAM →
            // MODEL_PARAM_DATA in place without a second initializer-execution pass.
            Stage("InferModelShapes");
            var shapeInferencer = new ShapeInferenceInterpreter(ctx);
            var concreteModel = paramValuesById is not null
                ? Shorokoo.Core.Nodes.Processors.Fast.FastApplyModelParamValues.Process(concreteArch, paramValuesById)
                : Shorokoo.Core.Nodes.Processors.Fast.FastApplyModelParamValues.Process(concreteArch, standInById);
            var modelShapeInfo = shapeInferencer.Infer(concreteModel, modelInputExemplars);
            var modelOutputInfo = modelShapeInfo.GetTensorInfo(concreteModel.Outputs[0])
                ?? throw new InvalidOperationException(
                    "Shape inference of concrete model graph failed to produce an output shape.");
            var (targetShape, targetDType) = DeriveTargetExemplar(
                lossGraph, shapeInferencer, modelOutputInfo.Shape, modelOutputInfo.DType);

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

            const int readThreshold = ShapeInferenceInterpreter.MaxSmallTensorElements;
            var allInputs = new IRuntimeTensor[graph.Inputs.Count];
            var idx = 0;

            // Parameter and model-state fields: their values where the initializers ran, and the
            // field's description where they did not — which is all a shape-driven pass reads of a
            // parameter too large to be worth materializing, either way.
            foreach (var f in TrainableParamStructDef.Fields)
                allInputs[idx++] = FieldExemplar(_initialParamFields, paramSlots, f.Name);
            foreach (var f in ModelStateDef.Fields)
                allInputs[idx++] = FieldExemplar(_initialStateFields, stateSlots, f.Name);
            foreach (var f in OptimizerStateDef.Fields)
                allInputs[idx++] = TensorDataConverter.ToRuntimeInput(_initialOptStateFields[f.Name], readThreshold);

            // Schedule-less runtime hyperparameter fields: seed shape inference / optimization with
            // their default (initial) scalar values. At run time these are supplied per step.
            foreach (var f in HyperparameterStructDef.Fields)
                allInputs[idx++] = TensorDataConverter.ToRuntimeInput(_initialHyperparamFields[f.Name], readThreshold);

            // Counter-input fields (int64 scalars): seed shape inference at the initial counters (0);
            // the scheduler math downstream computes the hyperparameter values from them. At run time
            // each is fed the checkpoint's corresponding counter.
            foreach (var _ in _counterInputNames)
                allInputs[idx++] = TensorDataConverter.ToRuntimeTensor(
                    (TensorData)Shorokoo.Globals.TensorData(Array.Empty<long>(), 0L), readThreshold);

            // Model-input fields: one shape exemplar per model input, in the model graph's input
            // order — shape inference reads only their shapes.
            foreach (var exemplar in modelInputExemplars)
                allInputs[idx++] = exemplar;

            // Remaining inputs are target fields (typically one Tensor target for L2/CE losses).
            // Through RepresentativeRuntimeInputFor, so a large target states its shape and dtype
            // rather than carrying a real zero buffer — the same threshold the model-input exemplars
            // use. These exemplars outlive the pass on the rig (OptimizationInputs), so a
            // multi-megabyte target would otherwise stay allocated for as long as the rig does.
            while (idx < graph.Inputs.Count)
                allInputs[idx++] = RepresentativeRuntimeInputFor(targetShape, targetDType);

            // The field's value where one is in hand, else its description as the engine reads it:
            // shape and dtype, and data only where the description carries any.
            static IRuntimeTensor FieldExemplar(
                Dictionary<string, IData> values, Dictionary<string, TensorAttribute> slots, string name)
                => values.TryGetValue(name, out var value)
                    ? TensorDataConverter.ToRuntimeInput(value, readThreshold)
                    : TensorDataConverter.ToRuntimeTensor(slots[name], readThreshold);

            // Step 4: Shape inference + memory-aware graph optimization. The optimizer
            // alternates Rematerializer and MemoryAwareScheduler under a combined
            // compute+memory metric, only committing transforms that strictly improve it.
            Stage("InferTrainingStepShapes");
            ShapeInferenceResult shapeInfo;
            if (TrainingBackend.LowersAutoGrad)
                shapeInfo = shapeInferencer.Infer(graph, allInputs);
            else
                // The gradient node the step keeps has a shape rule for this inference only; see
                // AutoGradShapeOp for why it is never registered.
                using (Shorokoo.Core.Interpreter.OpRegistry.Override(Shorokoo.Core.Interpreter.Ops.AutoGradShapeOp.Instance))
                    shapeInfo = shapeInferencer.Infer(graph, allInputs);

            // A parameter's shape is declared by its initializer and baked into the arch, and every
            // other part of the framework holds to it — binding a value of another shape into the
            // model is refused. The optimizer is the one place that could change it: its update is
            // ordinary tensor arithmetic, so a hyperparameter of another shape broadcasts against the
            // parameter and the "updated" parameter comes back that shape instead. Nothing downstream
            // would catch it — the update is packed by field name — and the rig would train happily
            // while being unable to checkpoint or serve what it produced.
            //
            // The step's updated-parameter outputs lead this graph's outputs, and their shapes have
            // just been inferred for the optimizer pass, so holding each to its parameter costs
            // nothing and fails here, at build, rather than after a step. A dimension inference
            // leaves symbolic (-1) constrains nothing.
            for (int p = 0; p < TrainableParamStructDef.Fields.Length && p < graph.Outputs.Count; p++)
            {
                var field = TrainableParamStructDef.Fields[p];
                if (shapeInfo.GetTensorInfo(graph.Outputs[p]) is not { } updatedInfo) continue;
                var updatedDims = updatedInfo.Shape.Dims;
                var declaredDims = paramSlots[field.Name].Shape.Dims;
                if (updatedDims.Contains(-1L) || updatedDims.SequenceEqual(declaredDims)) continue;
                throw new ArgumentException(
                    $"The optimizer returns trainable parameter '{field.Name}' shaped "
                    + $"[{string.Join(", ", updatedDims)}], but the model declares it "
                    + $"[{string.Join(", ", declaredDims)}]. An optimizer must return each parameter "
                    + "at that parameter's own shape; a hyperparameter or optimizer state of another "
                    + "shape broadcasts against it instead of scaling it.");
            }
            var baselineEval = new Shorokoo.Core.AutoDiffCheckpointing.GraphEvaluator().Evaluate(graph, shapeInfo);
            GraphOptimizationResult optResult;
            if (TrainingBackend.LowersAutoGrad)
            {
                Stage("OptimizeTrainingStepGraph");
                var optimizer = new MemoryAwareGraphOptimizer(shapeInference: shapeInferencer);
                optResult = optimizer.OptimizeWithShapeInfo(graph, shapeInfo);
            }
            else
            {
                // The memory-aware pass rewrites a backward pass it can see, and this step has none:
                // the gradient is the execution backend's, and so is what it keeps alive for it. The
                // step goes out as composed -- and a [Module(Checkpoint = true)] segment with it,
                // unhonoured.
                optResult = new GraphOptimizationResult
                {
                    StrategyName = "Baseline",
                    OptimizedGraph = graph,
                    ShapeInfo = shapeInfo,
                    Evaluation = baselineEval,
                    AllStrategies = [("Baseline", baselineEval, graph)],
                };
            }
            PreOptimizationEval = baselineEval;
            OptimizationResult = optResult;
            OptimizationInputs = allInputs;

            // Freeze the public views: the working graphs are relinquished into the
            // readonly wrappers, which own them exclusively from here on (the rig
            // compiles through the wrappers, which copy). Two full walks of the lowered
            // training graph, so named rather than left inside the optimizer's report.
            Stage("FreezeTrainingStepGraph");
            PreOptimizationGraph = new ComputationGraph(graph, GraphKind.ConcreteModel);
            TrainingStepPureGraph = new ComputationGraph(optResult.OptimizedGraph, GraphKind.ConcreteModel);
            _trainingStepWorkGraph = null;
        }
    }

    /// <summary>
    /// The immutable <b>constituent</b> layer of a <see cref="TrainingRig"/>: the swappable
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
