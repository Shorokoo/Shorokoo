using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Runtime;
using Shorokoo.Modules.Losses;
using Shorokoo.Modules.Optimizers;
using Shorokoo.Core.Nodes.Processors.Training;

namespace Shorokoo.Tests;

// ---------------------------------------------------------------------------
// Coverage-only training modules. Each model wraps a single shape-manipulation
// op around `input * trainable_weight` so that the autograd-lowered training
// graph carries the wrapped op, which forces
// <see cref="Shorokoo.Core.AutoDiffCheckpointing.OpsPerf.TensorManipulationPerf"/>
// (consulted by <see cref="GraphEvaluator"/> during optimization scoring) to
// hit the matching switch branch. The combos in
// <see cref="TrainingRigCoverageTests.TestShapeManipulationOpsCoverage"/> drive
// each module through the training rig so all the previously-uncovered ops
// (SLICE, TILE, CLIP, EXPAND no-op fast path, SCATTER_ELEMENTS, SPLIT) get
// estimated at least once per coverage run.
// ---------------------------------------------------------------------------

/// <summary>Slice op: y = (x * w)[1:5]. Output element-count differs from input.</summary>
[Module]
public partial class ScalarMultiplyAndSliceModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var weight = InitScalarWeight.Init(Vector(1L));
        var scaled = input * weight;
        return (Tensor<float32>)OnnxOp.Slice(scaled, Vector(1L), Vector(5L));
    }
}

/// <summary>Tile op: y = tile(x * w, [2]). Output is 2x larger than input.</summary>
[Module]
public partial class ScalarMultiplyAndTileModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var weight = InitScalarWeight.Init(Vector(1L));
        var scaled = input * weight;
        return (Tensor<float32>)OnnxOp.Tile(scaled, Vector(2L));
    }
}

/// <summary>Clip op: y = clip(x * w, -1, 1).</summary>
[Module]
public partial class ScalarMultiplyAndClipModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var weight = InitScalarWeight.Init(Vector(1L));
        var scaled = input * weight;
        return (Tensor<float32>)OnnxOp.Clip(scaled, Scalar(-1f), Scalar(1f));
    }
}

/// <summary>
/// Expand op exercising the zero-cost fast path: Expand([8] → [8]). Input and
/// target shape have the same element count, so
/// <c>TensorManipulationPerf.EXPAND</c> hits the "no actual expansion needed"
/// branch (zero compute, output aliases input buffer).
/// </summary>
[Module]
public partial class ScalarMultiplyAndExpandNoOpModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var weight = InitScalarWeight.Init(Vector(1L));
        var scaled = input * weight;
        return (Tensor<float32>)OnnxOp.Expand(scaled, Vector(8L));
    }
}

/// <summary>
/// ScatterElements op: writes a slice of (x * w) back into (x * w) at index 1.
/// Both `data` and `updates` depend on the trainable weight, so autograd has
/// well-defined gradient paths through both scatter inputs.
/// </summary>
[Module]
public partial class ScalarMultiplyAndScatterModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var weight = InitScalarWeight.Init(Vector(1L));
        var scaled = input * weight;
        var updates = (Tensor<float32>)OnnxOp.Slice(scaled, Vector(0L), Vector(1L));
        var indices = Vector(1L);
        return (Tensor<float32>)OnnxOp.ScatterElements(
            scaled, indices, updates,
            axis: 0, reduction: null);
    }
}

/// <summary>
/// Split op: split (x * w) into two halves and return the first half. Split is
/// multi-output; the gradient path concats per-output grads back together.
/// </summary>
[Module]
public partial class ScalarMultiplyAndSplitModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var weight = InitScalarWeight.Init(Vector(1L));
        var scaled = input * weight;
        var halves = scaled.Split(numOutputs: 2, axis: 0);
        return halves[0];
    }
}

/// <summary>
/// Loop whose iteration count is an <c>Add</c> of two constants. FastFoldConstants
/// excludes LOOP_OPEN's inputs from its required-constant set, so this expression
/// reaches <see cref="Shorokoo.Core.Nodes.Processors.Training.FastFoldLoopIterationCountsToConstantsProcessor"/>
/// with a non-CONSTANT producer. <c>Add</c> on int64 scalars is QEE-modelled, so
/// the processor's QEE-first path resolves it without spinning up an ORT session.
/// </summary>
[Module]
public partial class ScalarMultiplyWithQeeFoldableLoopIterCountModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var weight = InitScalarWeight.Init(Vector(1L));
        var scaled = input * weight;
        foreach (var ctx in LoopAPI.Iterate(Scalar(2L) + Scalar(1L)))
        {
            scaled = scaled * Scalar(1.0f);
        }
        return scaled;
    }
}

/// <summary>
/// Loop whose iteration count goes through <c>Det</c> — an op absent from QEE's
/// registry. The QEE-first attempt produces only a shape-only output for the
/// Det node (and therefore for the downstream <c>Cast</c>), so the iter-count
/// resolver falls back to <see cref="ComputeContext.Execute"/> (ORT) to recover
/// the value (1L, the determinant of the 2×2 identity matrix).
/// </summary>
[Module]
public partial class ScalarMultiplyWithOrtOnlyLoopIterCountModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var weight = InitScalarWeight.Init(Vector(1L));
        var scaled = input * weight;
        var identity = Tensor(new long[] { 2L, 2L }, 1f, 0f, 0f, 1f);
        var det = (Scalar<float32>)OnnxOp.Det(identity);
        var iter = det.Cast<int64>();
        foreach (var ctx in LoopAPI.Iterate(iter))
        {
            scaled = scaled * Scalar(1.0f);
        }
        return scaled;
    }
}

/// <summary>
/// Self-attention-shaped model whose training graph carries batched (3-D) matmuls
/// whose operands are computed intermediates (null static Rank): a linear
/// projection, a <c>q @ qᵀ</c> score matmul, a softmax, an <c>attn @ q</c> context
/// matmul, mean-pooling, then a 2-D classifier head. The batched-matmul backward
/// goes through the MatMul gradient's rank-agnostic last-two-dims transpose; before
/// that path existed, <see cref="TrainingRig.TrainStep"/> threw an OnnxRuntime
/// "operand cannot broadcast on dim 0".
/// </summary>
[Module]
public partial class BatchedMatmulModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var embed = Scalar(8L);    // E
        var classes = Scalar(4L);  // C

        var q = input.MatMul(InitXavier.Init([embed, embed]));        // (B,S,E)@(E,E) -> (B,S,E)
        var scores = q.MatMul(q.Transpose(0, 2, 1));                  // (B,S,E)@(B,E,S) -> (B,S,S) batched
        var attn = (Tensor<float32>)OnnxOp.Softmax(scores, axis: 2);
        var ctx = attn.MatMul(q);                                     // (B,S,S)@(B,S,E) -> (B,S,E) batched
        var pooled = ctx.Reduce(ReduceKind.Mean, Vector(1L), keepDims: false); // (B,E)
        return (Tensor<float32>)OnnxOp.Softmax(pooled.MatMul(InitXavier.Init([embed, classes])), axis: 1);
    }
}

/// <summary>
/// Two-input model: y = a·wa + b·wb, with two trainable scalar weights. Used to prove the
/// multi-input inference-extraction path (<see cref="TrainingRig.ExtractInferenceModel"/> /
/// <see cref="TrainingCheckpoint.ToInferenceModel()"/> bind weights into a retained concrete arch that
/// was concretized at ALL the rig's inputs, not just the first).
/// </summary>
[Module]
public partial class TwoInputSumModel
{
    public static Tensor<float32> Inline(Tensor<float32> a, Tensor<float32> b)
    {
        var wa = InitScalarWeight.Init(Vector(1L));
        var wb = InitScalarWeight.Init(Vector(1L));
        return a * wa + b * wb;
    }
}

/// <summary>
/// Coverage-purpose training-rig pipeline tests. Each [Fact] drives the full
/// model + loss + optimizer composition through <see cref="TrainingRig.FromScratch"/>
/// and <c>CreateInitialCheckpoint</c> for a curated combination of modules.
///
/// <para>
/// This file is structured like
/// <see cref="Shorokoo.Tests.Modules.CoverageTests.ModulesCoverageTests"/>:
/// a single helper (<see cref="CoverFromScratch"/>) drives one
/// (model, loss, optimizer, input-shape, hyperparams) combo, so each [Fact]
/// reduces to a series of one-liners covering different combinations. This
/// lets the Coverage suite reach every optimizer module
/// (<see cref="SGDOptimizer"/>, <see cref="SGDMomentumOptimizer"/>,
/// <see cref="AdamWOptimizer"/>), the state-update path
/// (<see cref="ScalarMultiplyWithBatchNormModel"/>), and additional loss
/// modules (<see cref="SoftmaxL2Loss"/>) without the per-test boilerplate of
/// constructing sample inputs by hand.
/// </para>
/// </summary>
[Trait("Domain", "Training")]
[Trait("Purpose", "Coverage")]
public class TrainingRigCoverageTests
{
    /// <summary>
    /// Drives a single training-rig configuration through the FromScratch
    /// pipeline (which routes through <c>BuildTrainingStepPureGraph</c> →
    /// autograd → optimizer replay → <c>MemoryAwareGraphOptimizer</c>) and
    /// then constructs the default checkpoint. The default checkpoint path
    /// in turn exercises QEE-store-based trainable-param discovery
    /// (<see cref="Shorokoo.Core.Nodes.Processors.Fast.FastConvertModelParamIdRefToModelParam.DiscoverTrainableParamInfos"/>).
    /// </summary>
    private static void CoverFromScratch(
        ComputationGraph modelGraph,
        ComputationGraph lossGraph,
        ComputationGraph optimizerGraph,
        long[] inputShape,
        params Hyperparameter[] hyperparams)
    {
        long totalElements = 1;
        foreach (var d in inputShape) totalElements *= d;
        var sampleInput = new TensorDataModelParam(
            "input", ModelParamType.InputParam,
            TensorData(inputShape, new float[totalElements]));

        var rig = TrainingRig.FromScratch(modelGraph, lossGraph, optimizerGraph,
            new NamedModelParam[] { sampleInput }, hyperparams);

        var checkpoint = rig.CreateInitialCheckpoint();
        Assert.NotEmpty(rig.TrainableParamStructDef.Fields);
        Assert.NotNull(checkpoint.TrainableParams);
    }

    /// <summary>
    /// Like <see cref="CoverFromScratch"/>, but additionally asserts the trained
    /// checkpoint round-trips straight back into a concrete inference model: every
    /// <c>TrainableParams</c> field name must resolve to a graph ModelId via the Shorokoo
    /// naming scheme (no silent drops), and the by-name <c>ToConcreteModel</c> must succeed
    /// and compile. Guards the contract that training preserves the inference model's
    /// canonical dotted param names (not a sanitized '.'→'_' form, which made
    /// <c>ToConcreteModel</c> throw <c>KeyNotFoundException</c>).
    /// </summary>
    private static void CoverCheckpointRebind(
        ComputationGraph modelGraph,
        ComputationGraph lossGraph,
        ComputationGraph optimizerGraph,
        long[] inputShape,
        params Hyperparameter[] hyperparams)
    {
        long totalElements = 1;
        foreach (var d in inputShape) totalElements *= d;
        var sampleInput = new TensorDataModelParam(
            "input", ModelParamType.InputParam,
            TensorData(inputShape, new float[totalElements]));

        var rig = TrainingRig.FromScratch(modelGraph, lossGraph, optimizerGraph,
            new NamedModelParam[] { sampleInput }, hyperparams);
        var checkpoint = rig.CreateInitialCheckpoint();

        // Concretize the inference model + Shorokoo naming scheme (the documented binding flow).
        var hints = new ModelParamList(
            new[] { new KeyValuePair<string, TensorData>(modelGraph.ToInternal().Inputs[0].ToString(), TensorData(inputShape, new float[totalElements])) },
            ModelParamType.InputParam);
        var ctx = new ComputeContext();
        var concrete = modelGraph.ToConcreteArchitecture(hints, ctx, null);
        var scheme = ModuleParamSetNamingScheme.FromModelIdFormats(concrete.GetShorokooIdNamingScheme(), "Shorokoo");
        var modelIds = concrete.GetConcreteModelParamInfos().ModelIds;

        var checkpointParams = checkpoint.TrainableParams.Fields
            .Where(f => f.Value is TensorData)
            .Select(f => new KeyValuePair<string, TensorData>(f.Key, (TensorData)f.Value))
            .ToList();
        Assert.NotEmpty(checkpointParams);
        foreach (var p in checkpointParams)
            Assert.True(scheme.ToModelId(p.Key, modelIds) is not null,
                $"checkpoint param '{p.Key}' did not resolve to a ModelId (name preservation regressed)");

        var bound = concrete.ToConcreteModel(
            new ModelParamList(checkpointParams, ModelParamType.TrainableParam), scheme);
        Assert.NotNull(bound);
        Assert.NotNull(ctx.Compile(bound));
    }

    /// <summary>
    /// Coverage for the minimal pipeline: scalar-multiply model + L2 loss +
    /// plain SGD. Asserts initial weight equals 1.0 (the model's default
    /// trainable-param initializer) — a smoke check that
    /// <c>CreateInitialCheckpoint</c> wired initializers correctly.
    /// </summary>
    [Fact]
    public void TestScalarMultiplySgdCoverage()
    {
        var modelGraph = ScalarMultiplyModel.ComputationGraph;
        var lossGraph = L2Loss.ComputationGraph;
        var optimizerGraph = SGDOptimizer.ComputationGraph;

        var sampleInput = new TensorDataModelParam(
            "input", ModelParamType.InputParam,
            TensorData([4L], new float[] { 1f, 2f, 3f, 4f }));

        var rig = TrainingRig.FromScratch(modelGraph, lossGraph, optimizerGraph,
            new NamedModelParam[] { sampleInput }, 0.01f);

        var checkpoint = rig.CreateInitialCheckpoint();

        Assert.Single(rig.TrainableParamStructDef.Fields);
        var weightField = rig.TrainableParamStructDef.Fields[0].Name;
        var weight = ((TensorData<float32>)checkpoint.TrainableParams.Fields[weightField]).AccessMemory()[0];
        Assert.Equal(1.0f, weight);
    }

    /// <summary>
    /// Coverage for non-default optimizers (<see cref="SGDMomentumOptimizer"/>
    /// and <see cref="AdamWOptimizer"/>), which the original Coverage suite
    /// missed entirely. Both optimizers carry per-parameter state, so this
    /// also exercises the optimizer-state initialization branch of
    /// <c>BuildTrainingStepPureGraph</c>.
    /// </summary>
    [Fact]
    public void TestNonDefaultOptimizersCoverage()
    {
        // SGD with momentum: lr=0.5, momentum=0.9 — adds 1 optimizer-state field per param.
        CoverFromScratch(ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            SGDMomentumOptimizer.ComputationGraph, [4L], 0.5f, 0.9f);
        // AdamW: lr, beta1, beta2, epsilon, weight_decay — adds 2 optimizer-state fields (m, v) per param.
        CoverFromScratch(ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            AdamWOptimizer.ComputationGraph, [4L], 0.001f, 0.9f, 0.999f, 1e-8f, 0.01f);
    }

    /// <summary>
    /// Coverage for the BatchNorm-bearing model path. <c>StateUpdate</c> calls
    /// in <see cref="ScalarMultiplyWithBatchNormModel"/> produce running-mean
    /// / running-var state fields that flow through training as
    /// <see cref="TrainingRig.ModelStateDef"/> — a different code path from
    /// the "no model state" combo above.
    /// </summary>
    [Fact]
    public void TestStatefulModelCoverage()
    {
        CoverFromScratch(ScalarMultiplyWithBatchNormModel.ComputationGraph, L2Loss.ComputationGraph,
            SGDOptimizer.ComputationGraph, [8L], 0.5f);
        CoverFromScratch(ScalarMultiplyWithBatchNormModel.ComputationGraph, L2Loss.ComputationGraph,
            SGDMomentumOptimizer.ComputationGraph, [8L], 0.5f, 0.9f);
    }

    /// <summary>
    /// Coverage for the multi-layer classifier path with the softmax loss
    /// (<see cref="SoftmaxL2Loss"/>). DigitClassifier has multiple trainable
    /// param fields and a 2-D input, exercising a much broader slice of the
    /// optimizer's per-field replay loop and OpsPerf shape models than the
    /// 1-D scalar combos above.
    /// </summary>
    [Fact]
    public void TestDigitClassifierCoverage()
    {
        CoverFromScratch(DigitClassifier.ComputationGraph, SoftmaxL2Loss.ComputationGraph,
            SGDMomentumOptimizer.ComputationGraph, [4L, 64L], 0.5f, 0.9f);
        CoverFromScratch(DigitClassifier.ComputationGraph, SoftmaxL2Loss.ComputationGraph,
            AdamWOptimizer.ComputationGraph, [4L, 64L], 0.001f, 0.9f, 0.999f, 1e-8f, 0.01f);
    }

    /// <summary>
    /// Coverage for the concrete arch's representative-input attribute across the
    /// shape-inference size threshold. The rig records a zero-filled representative input
    /// on each <c>MODEL_TENSOR_INPUT</c> node; inputs at or below
    /// <see cref="Shorokoo.Core.AutoDiffCheckpointing.ShapeInferenceInterpreter.MaxSmallTensorElements"/>
    /// carry a real zero payload, larger ones a shape+dtype-only placeholder.
    ///
    /// <para>The <c>512</c> case pins a regression: the placeholder threshold must be the
    /// engine's small-tensor limit (1024), not QEE's <c>DefaultMaxDataElements</c> (256).
    /// With the wrong (256) threshold a 512-element input got a values-elided placeholder that
    /// the shape-inference engine — which reads payloads up to 1024 — then tried to read,
    /// throwing and collapsing the whole QEE pass so no output shape could be inferred. The
    /// <c>256</c> (boundary), <c>1024</c> (boundary, real payload) and <c>2048</c>
    /// (over-threshold, genuine placeholder — read shape-only, never materialized) cases fence
    /// the boundary and exercise the placeholder path.</para>
    /// </summary>
    [Fact]
    public void TestRepresentativeInputThresholdCoverage()
    {
        // Boundary at/below the small-tensor threshold: real zero payload.
        CoverFromScratch(ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            SGDOptimizer.ComputationGraph, [256L], 0.01f);
        // Regression window (256, 1024]: must be a real payload, not an elided placeholder.
        CoverFromScratch(ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            SGDOptimizer.ComputationGraph, [512L], 0.01f);
        CoverFromScratch(ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            SGDOptimizer.ComputationGraph, [1024L], 0.01f);
        // Over the threshold: genuine shape+dtype-only placeholder, read shape-only by QEE.
        CoverFromScratch(ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            SGDOptimizer.ComputationGraph, [2048L], 0.01f);
    }

    /// <summary>
    /// The representative inputs live on the shared concrete arch and are re-read on every
    /// derivation (<c>ReadRepresentativeInputs</c>). This drives an over-threshold input (so the
    /// derivation path re-reads a shape-only placeholder and feeds it to shape inference) through
    /// <see cref="TrainingRig.WithLoss"/>, <see cref="TrainingRig.WithOptimizer(ComputationGraph, Hyperparameter[])"/>
    /// and <see cref="TrainingRig.WithScheduler(Hyperparameter[])"/>, asserting each derived rig
    /// still builds a valid checkpoint — proving the attribute survives derivation and the
    /// placeholder is never materialized on the re-read.
    /// </summary>
    [Fact]
    public void TestRepresentativeInputSurvivesDerivationCoverage()
    {
        NamedModelParam[] sample =
        [
            new TensorDataModelParam("input", ModelParamType.InputParam,
                TensorData([2048L], new float[2048])),
        ];
        var rig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            SGDOptimizer.ComputationGraph, sample, 0.01f);

        var reLoss = rig.WithLoss(L2Loss.ComputationGraph);
        Assert.NotNull(reLoss.CreateInitialCheckpoint().TrainableParams);

        var reOpt = rig.WithOptimizer(SGDMomentumOptimizer.ComputationGraph, 0.5f, 0.9f);
        Assert.NotEmpty(reOpt.CreateInitialCheckpoint().OptimizerState.Fields);

        var reSched = reOpt.WithScheduler(0.25f, 0.9f);
        Assert.NotNull(reSched.CreateInitialCheckpoint().TrainableParams);
    }

    /// <summary>
    /// <see cref="TrainingRig.WithSeed"/> rebinds the RNG identity on a <c>Clone()</c> of the
    /// concrete arch; <c>Clone</c> copies node attributes by reference, so the reseeded arch's
    /// <c>MODEL_TENSOR_INPUT</c> nodes must still carry their representative-input attributes.
    /// A checkpoint built from the reseeded rig (whose derivation re-reads those attributes for
    /// shape inference) therefore has to succeed. Uses an over-threshold input so the carried
    /// attribute is a shape-only placeholder.
    /// </summary>
    [Fact]
    public void TestRepresentativeInputSurvivesReSeedCoverage()
    {
        NamedModelParam[] sample =
        [
            new TensorDataModelParam("input", ModelParamType.InputParam,
                TensorData([2048L], new float[2048])),
        ];
        var rig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            SGDOptimizer.ComputationGraph, sample, 0.01f);

        var reseeded = rig.WithSeed(new RngConfig { MasterSeed = 7 });
        Assert.NotNull(reseeded.CreateInitialCheckpoint().TrainableParams);
    }

    /// <summary>
    /// The two runtime <see cref="ComputeContext"/> members — <see cref="TrainingRig.MergeContext"/>
    /// (build/merge phase) and <see cref="TrainingRig.RuntimeContext"/> (compile/run) — default to
    /// <see cref="ComputeContext.Default"/> under a plain construction, are stored verbatim when
    /// supplied non-default, and are carried forward BY REFERENCE (never re-defaulted) through every
    /// <c>With…</c> derivation, including the distinct <see cref="TrainingRig.WithSeed"/> re-key path.
    /// </summary>
    [Fact]
    public void TestComputeContextsStoredAndPropagatedThroughDerivationCoverage()
    {
        NamedModelParam[] sample =
        [
            new TensorDataModelParam("input", ModelParamType.InputParam,
                TensorData([4L], new float[4])),
        ];

        // Default construction: both contexts are the process-wide Default singleton.
        var defaultRig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            SGDOptimizer.ComputationGraph, sample, 0.01f);
        Assert.Same(ComputeContext.Default, defaultRig.MergeContext);
        Assert.Same(ComputeContext.Default, defaultRig.RuntimeContext);

        // Non-default contexts are stored verbatim. The rng overload places the two contexts before
        // the params hyperparameters, matching the RngConfig-before-params convention.
        var merge = new ComputeContext();
        var runtime = new ComputeContext();
        var rig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            SGDOptimizer.ComputationGraph, sample, null, merge, runtime, 0.01f);
        Assert.Same(merge, rig.MergeContext);
        Assert.Same(runtime, rig.RuntimeContext);

        // Propagated by reference through a constituent-swap derivation…
        var derived = rig.WithLoss(L2Loss.ComputationGraph);
        Assert.Same(merge, derived.MergeContext);
        Assert.Same(runtime, derived.RuntimeContext);

        // …and through the re-seed derivation (a separate code path that re-keys a cloned arch).
        var reseeded = rig.WithSeed(new RngConfig { MasterSeed = 3 });
        Assert.Same(merge, reseeded.MergeContext);
        Assert.Same(runtime, reseeded.RuntimeContext);
    }

    /// <summary>
    /// The compute contexts are runtime config, never serialized to any checkpoint. A rig built with
    /// distinctive custom contexts saves a checkpoint; a <b>fresh</b> rig carrying its own, different
    /// contexts loads it. Because nothing context-shaped is written to (or read back from) the file,
    /// the reloaded checkpoint's rig carries the LOADER rig's contexts — the supplied-rig precedence —
    /// and never a trace of the saver's. This pins both "not persisted" and the precedence rule.
    /// </summary>
    [Fact]
    public void TestComputeContextsNeverPersistedCoverage()
    {
        NamedModelParam[] sample =
        [
            new TensorDataModelParam("input", ModelParamType.InputParam,
                TensorData([4L], new float[4])),
        ];

        var saverMerge = new ComputeContext();
        var saverRuntime = new ComputeContext();
        var saverRig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            SGDOptimizer.ComputationGraph, sample, null, saverMerge, saverRuntime, 0.01f);
        var ckpt = saverRig.CreateInitialCheckpoint();

        var path = Path.Combine(Path.GetTempPath(), $"shrk_ctx_notpersisted_{Guid.NewGuid():N}.safetensors");
        try
        {
            ckpt.Save(path);

            var loaderMerge = new ComputeContext();
            var loaderRuntime = new ComputeContext();
            var loaderRig = TrainingRig.FromScratch(
                ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
                SGDOptimizer.ComputationGraph, sample, null, loaderMerge, loaderRuntime, 0.01f);
            var loaded = loaderRig.LoadCheckpoint(path);

            Assert.Same(loaderMerge, loaded.Rig!.MergeContext);
            Assert.Same(loaderRuntime, loaded.Rig!.RuntimeContext);
            Assert.NotSame(saverMerge, loaded.Rig!.MergeContext);
            Assert.NotSame(saverRuntime, loaded.Rig!.RuntimeContext);
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    /// <summary>
    /// Inertness: the representative-input attribute must never reach a serialized artifact. A
    /// boundary/input node is emitted as a <c>ValueInfoProto</c>, never a <c>NodeProto</c>, so its
    /// attributes are not serialized. This saves an initial checkpoint (whose model lineage carries
    /// the attribute on its input node) to a native <c>.skpt</c> and asserts the raw container bytes
    /// contain no trace of the attribute name, then confirms the artifact still loads as a concrete
    /// inference model. Uses an over-threshold input so the attribute value is a placeholder.
    /// </summary>
    [Fact]
    public void TestRepresentativeInputIsInertInSkptCoverage()
    {
        NamedModelParam[] sample =
        [
            new TensorDataModelParam("input", ModelParamType.InputParam,
                TensorData([2048L], new float[2048])),
        ];
        var rig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            SGDOptimizer.ComputationGraph, sample, 0.01f);
        var ckpt = rig.CreateInitialCheckpoint();

        var path = Path.Combine(Path.GetTempPath(), $"shrk_repin_inert_{Guid.NewGuid():N}.skpt");
        try
        {
            Persistence.SaveTrainingCheckpointToSkpt(ckpt, path);

            var bytes = File.ReadAllBytes(path);
            var needle = System.Text.Encoding.ASCII.GetBytes(
                Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames.ShrkAttrRepresentativeInput);
            Assert.False(ContainsSubsequence(bytes, needle),
                "the representative-input attribute leaked into the serialized .skpt artifact");

            var inferenceModel = Persistence.Load(path);
            Assert.Equal(GraphKind.ConcreteModel, inferenceModel.Kind);
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    /// <summary>Byte-subsequence search for the inertness assertion (no dependency on the STORED-zip
    /// entry layout — the attribute name must be absent from the whole container).</summary>
    private static bool ContainsSubsequence(byte[] haystack, byte[] needle)
    {
        if (needle.Length == 0 || haystack.Length < needle.Length) return false;
        for (int i = 0; i <= haystack.Length - needle.Length; i++)
        {
            int j = 0;
            while (j < needle.Length && haystack[i + j] == needle[j]) j++;
            if (j == needle.Length) return true;
        }
        return false;
    }

    /// <summary>
    /// Coverage for the shape-manipulation arms of
    /// <see cref="Shorokoo.Core.AutoDiffCheckpointing.OpsPerf.TensorManipulationPerf"/>
    /// that no mainstream model in the Coverage suite exercises. Each model
    /// wraps a single op (SLICE, TILE, CLIP, EXPAND no-op, SCATTER_ELEMENTS,
    /// SPLIT) around the trainable forward path so the optimizer's
    /// per-strategy evaluation has to score that op for real.
    /// </summary>
    [Fact]
    public void TestShapeManipulationOpsCoverage()
    {
        CoverFromScratch(ScalarMultiplyAndSliceModel.ComputationGraph, L2Loss.ComputationGraph,
            SGDOptimizer.ComputationGraph, [8L], 0.01f);
        CoverFromScratch(ScalarMultiplyAndTileModel.ComputationGraph, L2Loss.ComputationGraph,
            SGDOptimizer.ComputationGraph, [4L], 0.01f);
        CoverFromScratch(ScalarMultiplyAndClipModel.ComputationGraph, L2Loss.ComputationGraph,
            SGDOptimizer.ComputationGraph, [4L], 0.01f);
        CoverFromScratch(ScalarMultiplyAndExpandNoOpModel.ComputationGraph, L2Loss.ComputationGraph,
            SGDOptimizer.ComputationGraph, [8L], 0.01f);
        CoverFromScratch(ScalarMultiplyAndScatterModel.ComputationGraph, L2Loss.ComputationGraph,
            SGDOptimizer.ComputationGraph, [4L], 0.01f);
        CoverFromScratch(ScalarMultiplyAndSplitModel.ComputationGraph, L2Loss.ComputationGraph,
            SGDOptimizer.ComputationGraph, [4L], 0.01f);
    }

    /// <summary>
    /// Coverage for the loop iter-count folding processor's two evaluation
    /// strategies. <see cref="ScalarMultiplyWithQeeFoldableLoopIterCountModel"/>
    /// has an iter-count expression QEE can resolve (Add of two int64 scalar
    /// constants), driving the QEE-first happy path inside
    /// <see cref="Shorokoo.Core.Nodes.Processors.Training.FastFoldLoopIterationCountsToConstantsProcessor"/>.
    /// <see cref="ScalarMultiplyWithOrtOnlyLoopIterCountModel"/> routes the iter
    /// count through <c>Det</c> — absent from QEE's op registry — so the
    /// processor's per-key extractor misses the value and the ORT fallback
    /// runs the resolver subgraph.
    /// </summary>
    [Fact]
    public void TestLoopIterCountFoldingCoverage()
    {
        CoverFromScratch(ScalarMultiplyWithQeeFoldableLoopIterCountModel.ComputationGraph,
            L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph, [4L], 0.01f);
        CoverFromScratch(ScalarMultiplyWithOrtOnlyLoopIterCountModel.ComputationGraph,
            L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph, [4L], 0.01f);
    }

    /// <summary>
    /// Drives <see cref="TrainingRig.TrainStep"/> and <see cref="TrainingRig.Train"/>
    /// through one minimal step. These methods previously had 0% coverage in the
    /// Coverage suite — <c>CoverFromScratch</c> only exercises the rig-construction
    /// and default-checkpoint paths, never an actual training-step execution.
    /// One forward+backward+update pass is enough to hit every line in
    /// <c>TrainStep</c> (output-repacking loops for params / model state /
    /// optimizer state plus loss extraction) and <c>Train</c> (epoch+batch
    /// loop and per-batch <c>TrainStep</c> dispatch).
    /// </summary>
    [Fact]
    public void TestTrainStepAndTrainCoverage()
    {
        var rig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph,
            L2Loss.ComputationGraph,
            SGDOptimizer.ComputationGraph,
            new NamedModelParam[]
            {
                new TensorDataModelParam("input", ModelParamType.InputParam,
                    TensorData([4L], new float[] { 1f, 2f, 3f, 4f })),
            },
            0.1f);

        var initial = rig.CreateInitialCheckpoint();

        var modelInputDef = new TensorStructDef(
            new[] { new TensorStructFieldDef("input", DataStructure.Tensor, 1, DType.Float32) },
            "ModelInput");
        var targetDef = new TensorStructDef(
            new[] { new TensorStructFieldDef("targets", DataStructure.Tensor, 1, DType.Float32) },
            "Target");

        var inputBatch = new TensorDataStruct(modelInputDef,
            new Dictionary<string, IData> { { "input", TensorData([4L], new float[] { 1f, 2f, 3f, 4f }) } });
        var targetBatch = new TensorDataStruct(targetDef,
            new Dictionary<string, IData> { { "targets", TensorData([4L], new float[] { 0f, 0f, 0f, 0f }) } });

        // Drive Train: covers the per-epoch / per-batch loop and the
        // TrainingResult constructor + EpochLosses / FinalCheckpoint getters.
        var trainResult = rig.Train(initial, new[] { inputBatch }, new[] { targetBatch }, numEpochs: 1);
        Assert.Single(trainResult.EpochLosses);
        Assert.NotNull(trainResult.FinalCheckpoint);

        // Drive TrainStep directly so its output-repacking branches all execute
        // outside the Train wrapper; TrainStep now returns the post-step checkpoint directly.
        var stepResult = rig.TrainStep(initial, inputBatch, targetBatch);
        Assert.NotNull(stepResult);
        Assert.NotNull(stepResult.TrainableParams);
        // ModelState / OptimizerState are empty for this combo, but their getters
        // still need to be exercised for full coverage.
        Assert.NotNull(stepResult.ModelState);
        Assert.NotNull(stepResult.OptimizerState);
        Assert.True(float.IsFinite(stepResult.Loss!.Value));

        // FastTrainingGraphs is a plain container (internal, reachable via
        // InternalsVisibleTo); its constructor and three getters are otherwise
        // unreachable. It is typed on the mutable internal graph, so hand it
        // safe deep copies of the shared cached module graphs.
        var graphs = new FastTrainingGraphs(
            ScalarMultiplyModel.ComputationGraph.ToInternal(),
            L2Loss.ComputationGraph.ToInternal(),
            SGDOptimizer.ComputationGraph.ToInternal());
        Assert.NotNull(graphs.ModelGraph);
        Assert.NotNull(graphs.LossGraph);
        Assert.NotNull(graphs.OptimizerGraph);
    }

    /// <summary>
    /// Covers initializer-driven optimizer state: state variables are created inside the
    /// optimizer body by an optimizer-owned [StateInitializer] (never in the Inline signature),
    /// the rig runs that initializer per trainable parameter for the default checkpoint (here a
    /// ones-fill, so a blanket zero-init would fail the assert), and the updated state
    /// round-trips through <see cref="TrainingRig.TrainStep"/>. Also covers the two
    /// ownership-misuse rejections: a module-owned state initializer inside an optimizer graph,
    /// and an optimizer-owned state initializer inside a model graph.
    /// </summary>
    [Fact]
    public void TestOptimizerStateInitializerCoverage()
    {
        var sample = new NamedModelParam[]
        {
            new TensorDataModelParam("input", ModelParamType.InputParam,
                TensorData([4L], new float[] { 1f, 2f, 3f, 4f })),
        };

        var rig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            StepCountingSgdOptimizer.ComputationGraph, sample, 0.1f);

        // One state field per (single) trainable param, initialized by InitOptStateOnes to 1.
        var initial = rig.CreateInitialCheckpoint();
        Assert.Single(rig.OptimizerStateDef.Fields);
        Assert.All(FlattenStruct(initial.OptimizerState), v => Assert.Equal(1f, v));

        // After one step the counter state must have advanced to 2 (round-trip through outputs).
        var modelInputDef = new TensorStructDef(
            new[] { new TensorStructFieldDef("input", DataStructure.Tensor, 1, DType.Float32) }, "ModelInput");
        var targetDef = new TensorStructDef(
            new[] { new TensorStructFieldDef("targets", DataStructure.Tensor, 1, DType.Float32) }, "Target");
        var inputBatch = new TensorDataStruct(modelInputDef,
            new Dictionary<string, IData> { { "input", TensorData([4L], new float[] { 1f, 2f, 3f, 4f }) } });
        var targetBatch = new TensorDataStruct(targetDef,
            new Dictionary<string, IData> { { "targets", TensorData([4L], new float[] { 0f, 0f, 0f, 0f }) } });

        var step = rig.TrainStep(initial, inputBatch, targetBatch);
        Assert.All(FlattenStruct(step.OptimizerState), v => Assert.Equal(2f, v));

        // Ownership misuse is rejected in both directions, with guidance in the message.
        var optEx = Assert.Throws<InvalidOperationException>(() => TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            ModuleOwnedStateOptimizer.ComputationGraph, sample, 0.1f));
        Assert.Contains("OptimizerOwned", optEx.Message);

        var modelEx = Assert.Throws<ArgumentException>(() => TrainingRig.FromScratch(
            OptimizerOwnedStateModel.ComputationGraph, L2Loss.ComputationGraph,
            SGDOptimizer.ComputationGraph, sample, 0.1f));
        Assert.Contains("ModuleOwned", modelEx.Message);
    }

    /// <summary>
    /// Adam carries its timestep as a true rank-0 scalar (one float per parameter) rather than a
    /// param-shaped buffer: the <c>_opt_2</c> (step) optimizer-state field must be rank 0, and a
    /// trained checkpoint — scalar step included — must survive a save → fresh-rig → load
    /// round-trip exactly. Guards both the scalar-state pipeline and the SafeTensors rank-0
    /// serialization fix.
    /// </summary>
    [Fact]
    public void TestAdamScalarStepCheckpointRoundtrip()
    {
        var sample = new NamedModelParam[]
        {
            new TensorDataModelParam("input", ModelParamType.InputParam,
                TensorData([4L], new float[] { 1f, 2f, 3f, 4f })),
        };
        TrainingRig AdamRig() => TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            AdamOptimizer.ComputationGraph, sample,
            new AdamOptimizerHyperparameters { LearningRate = 0.1f });

        var rig = AdamRig();
        // m, v (param-shaped) + step (scalar) per the single trainable param.
        Assert.Equal(3, rig.OptimizerStateDef.Fields.Length);
        var stepField = rig.OptimizerStateDef.Fields[2];
        Assert.Equal(0, stepField.Rank);   // the timestep is a rank-0 scalar, not param-shaped

        var modelInputDef = new TensorStructDef(
            new[] { new TensorStructFieldDef("input", DataStructure.Tensor, 1, DType.Float32) }, "ModelInput");
        var targetDef = new TensorStructDef(
            new[] { new TensorStructFieldDef("targets", DataStructure.Tensor, 1, DType.Float32) }, "Target");
        var inputBatch = new TensorDataStruct(modelInputDef,
            new Dictionary<string, IData> { { "input", TensorData([4L], new float[] { 1f, 2f, 3f, 4f }) } });
        var targetBatch = new TensorDataStruct(targetDef,
            new Dictionary<string, IData> { { "targets", TensorData([4L], new float[] { 2f, 4f, 6f, 8f }) } });

        var ckpt = rig.CreateInitialCheckpoint();
        for (int i = 0; i < 2; i++)
            ckpt = rig.TrainStep(ckpt, inputBatch, targetBatch);

        // After two steps the scalar step state holds 2 (and is genuinely rank-0 in the data).
        var stepData = (TensorData)ckpt.OptimizerState.Fields[stepField.Name];
        Assert.Empty(stepData.Shape.Dims);
        Assert.Equal(2f, stepData.As<float32>().AccessMemory()[0]);

        var path = System.IO.Path.Combine(System.IO.Path.GetTempPath(),
            $"adam_scalar_{System.Guid.NewGuid():N}.safetensors");
        try
        {
            ckpt.Save(path);   // exercises the SafeTensors rank-0 save path

            var rigB = AdamRig();
            var loaded = rigB.LoadCheckpoint(path);
            Assert.Equal(2, loaded.Step);
            Assert.Equal(FlattenStruct(ckpt.OptimizerState), FlattenStruct(loaded.OptimizerState));
            Assert.Equal(FlattenStruct(ckpt.TrainableParams), FlattenStruct(loaded.TrainableParams));
            var loadedStep = (TensorData)loaded.OptimizerState.Fields[stepField.Name];
            Assert.Empty(loadedStep.Shape.Dims);   // rank-0 survives the round-trip
        }
        finally
        {
            if (System.IO.File.Exists(path)) System.IO.File.Delete(path);
        }
    }

    private static float[] FlattenStruct(TensorDataStruct s) =>
        s.Definition.Fields
            .SelectMany(f => ((TensorData)s.Fields[f.Name]).As<float32>().AccessMemory<float>().ToArray())
            .ToArray();

    /// <summary>
    /// Covers <see cref="TrainingCheckpoint.Save(string)"/> / <see cref="TrainingRig.LoadCheckpoint"/>
    /// (and the static <see cref="TrainingCheckpoint.Load"/> they delegate to): a checkpoint must
    /// survive a save → "fresh process" (a brand-new rig + compiled graph from the same graphs) →
    /// load, with the global step, trainable params, model state, and optimizer state all restored
    /// so training resumes exactly. Drives three sections: AdamW (non-empty optimizer state m/v) for
    /// the trainable + optimizer-state path with real TrainSteps; a BatchNorm model for the non-empty
    /// model-state path; and the mismatch error path (loading a checkpoint into a rig whose
    /// definitions don't match).
    /// </summary>
    [Fact]
    public void TestCheckpointSaveLoadResumeCoverage()
    {
        var sample = new NamedModelParam[]
        {
            new TensorDataModelParam("input", ModelParamType.InputParam,
                TensorData([4L], new float[] { 1f, 2f, 3f, 4f })),
        };
        TrainingRig AdamRig() => TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            AdamWOptimizer.ComputationGraph, sample,
            new AdamWOptimizerHyperparameters { LearningRate = 0.1f });

        var modelInputDef = new TensorStructDef(
            new[] { new TensorStructFieldDef("input", DataStructure.Tensor, 1, DType.Float32) }, "ModelInput");
        var targetDef = new TensorStructDef(
            new[] { new TensorStructFieldDef("targets", DataStructure.Tensor, 1, DType.Float32) }, "Target");
        var inputBatch = new TensorDataStruct(modelInputDef,
            new Dictionary<string, IData> { { "input", TensorData([4L], new float[] { 1f, 2f, 3f, 4f }) } });
        var targetBatch = new TensorDataStruct(targetDef,
            new Dictionary<string, IData> { { "targets", TensorData([4L], new float[] { 2f, 4f, 6f, 8f }) } });

        var path = Path.Combine(Path.GetTempPath(), $"shrk_ckpt_{Guid.NewGuid():N}.safetensors");
        try
        {
            // Train two steps, then save mid-training.
            var rigA = AdamRig();
            var ckpt = rigA.CreateInitialCheckpoint();
            for (int i = 0; i < 2; i++)
                ckpt = rigA.TrainStep(ckpt, inputBatch, targetBatch);
            Assert.Equal(2, ckpt.Step);
            ckpt.Save(path);
            Assert.True(File.Exists(path));

            // "Fresh process": a brand-new rig + compiled graph loads the checkpoint.
            var rigB = AdamRig();
            var loaded = rigB.LoadCheckpoint(path);

            // Step, trainable params, and optimizer state (m/v) must all round-trip exactly.
            Assert.Equal(2, loaded.Step);
            Assert.Equal(FlattenStruct(ckpt.TrainableParams), FlattenStruct(loaded.TrainableParams));
            Assert.Equal(FlattenStruct(ckpt.OptimizerState), FlattenStruct(loaded.OptimizerState));
            Assert.NotEmpty(loaded.OptimizerState.Fields); // AdamW carries m/v per param

            // Resuming from the loaded checkpoint advances the step and yields a finite loss.
            var resumed = rigB.TrainStep(loaded, inputBatch, targetBatch);
            Assert.Equal(3, resumed.Step);
            Assert.True(float.IsFinite(resumed.Loss!.Value));

            // Non-empty model-state path: a BatchNorm model's default checkpoint round-trips its
            // running-stat state fields (no TrainStep needed — exercises the model_state section).
            var bnRig = TrainingRig.FromScratch(
                ScalarMultiplyWithBatchNormModel.ComputationGraph, L2Loss.ComputationGraph,
                SGDMomentumOptimizer.ComputationGraph,
                new NamedModelParam[]
                {
                    new TensorDataModelParam("input", ModelParamType.InputParam,
                        TensorData([8L], new float[8])),
                },
                0.5f, 0.9f);
            var bnPath = Path.Combine(Path.GetTempPath(), $"shrk_ckpt_bn_{Guid.NewGuid():N}.safetensors");
            try
            {
                var bnCkpt = bnRig.CreateInitialCheckpoint();
                Assert.NotEmpty(bnCkpt.ModelState.Fields);
                bnCkpt.Save(bnPath);
                var bnLoaded = bnRig.LoadCheckpoint(bnPath);
                Assert.Equal(FlattenStruct(bnCkpt.ModelState), FlattenStruct(bnLoaded.ModelState));
                Assert.Equal(FlattenStruct(bnCkpt.OptimizerState), FlattenStruct(bnLoaded.OptimizerState));

                // Mismatch: the AdamW (ScalarMultiply) checkpoint must not load into the BatchNorm rig.
                Assert.Throws<InvalidOperationException>(() => bnRig.LoadCheckpoint(path));
            }
            finally { if (File.Exists(bnPath)) File.Delete(bnPath); }
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    /// <summary>
    /// Covers the truncated-checkpoint diagnostic inherited from the SafeTensors loader:
    /// a checkpoint file cut short (interrupted download/copy, disk full) is refused by
    /// <see cref="TrainingRig.LoadCheckpoint"/> with an error naming truncation, the declared
    /// vs. actual byte counts, and the checkpoint path — not an incidental parse failure.
    /// </summary>
    [Fact]
    public void TestCheckpointLoadTruncatedFailsLoudlyCoverage()
    {
        var rig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            SGDOptimizer.ComputationGraph,
            [
                new TensorDataModelParam("input", ModelParamType.InputParam,
                    TensorData([4L], new float[] { 1f, 2f, 3f, 4f })),
            ],
            0.1f);
        var path = Path.Combine(Path.GetTempPath(), $"shrk_ckpt_trunc_{Guid.NewGuid():N}.safetensors");
        try
        {
            rig.CreateInitialCheckpoint().Save(path);
            var full = File.ReadAllBytes(path);
            File.WriteAllBytes(path, full[..^8]);   // cut mid-tensor-data, as an interrupted copy would

            var ex = Assert.Throws<ModelException>(() => rig.LoadCheckpoint(path));
            Assert.Equal(ErrorCodes.ST003, ex.ErrorCode);
            Assert.Contains("truncated", ex.Message);
            Assert.Contains(path, ex.Message);                       // the checkpoint path surfaces
            Assert.Contains($"{full.Length} bytes", ex.Message);     // declared (required) size
            Assert.Contains($"{full.Length - 8} bytes", ex.Message); // actual size
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    /// <summary>
    /// Covers the atomic commit in <see cref="TrainingCheckpoint.Save(string)"/> (via
    /// <see cref="AtomicFileWriter"/>): a save that dies between writing the staged temp file
    /// and the rename leaves the previous checkpoint intact and loadable at the target path
    /// (never a truncated file); the stale temp from the failed save is swept by the next
    /// successful save; and saving into a directory that does not exist fails up front with a
    /// clear error instead of writing anywhere.
    /// </summary>
    [Fact]
    public void TestCheckpointSaveAtomicCoverage()
    {
        var rig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            SGDOptimizer.ComputationGraph,
            new NamedModelParam[]
            {
                new TensorDataModelParam("input", ModelParamType.InputParam,
                    TensorData([4L], new float[] { 1f, 2f, 3f, 4f })),
            },
            0.1f);
        var ckptV1 = rig.CreateInitialCheckpoint();                       // Step 0
        var ckptV2 = new TrainingCheckpoint(                              // same tensors, Step 7
            ckptV1.TrainableParams, ckptV1.ModelState, ckptV1.OptimizerState, step: 7);

        var dir = Path.Combine(Path.GetTempPath(), $"shrk_ckpt_atomic_{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        try
        {
            var path = Path.Combine(dir, "ckpt.safetensors");
            ckptV1.Save(path);
            Assert.Equal(0, rig.LoadCheckpoint(path).Step);

            // Simulated crash between write and rename: Save throws, but the good v1
            // checkpoint still loads from the target path. The hook filters on this test's
            // directory so parallel tests saving elsewhere are unaffected.
            AtomicFileWriter.CommitFaultInjection = p =>
            {
                if (p.StartsWith(dir, StringComparison.Ordinal)) throw new IOException("injected crash");
            };
            try
            {
                Assert.Throws<IOException>(() => ckptV2.Save(path));
            }
            finally { AtomicFileWriter.CommitFaultInjection = null; }
            Assert.Equal(0, rig.LoadCheckpoint(path).Step);

            // A stale temp left by a hard-killed save (planted by hand) is swept by the next
            // successful save, which lands v2.
            var stale = Path.Combine(dir, $".tmp-ckpt.safetensors-{Guid.NewGuid():N}");
            File.WriteAllText(stale, "partial");
            ckptV2.Save(path);
            Assert.Equal(7, rig.LoadCheckpoint(path).Step);
            Assert.False(File.Exists(stale));
            Assert.Empty(Directory.GetFileSystemEntries(dir, ".tmp-*"));

            // A target in a nonexistent directory is rejected before anything is written.
            Assert.Throws<DirectoryNotFoundException>(
                () => ckptV1.Save(Path.Combine(dir, "missing", "ckpt.safetensors")));
            Assert.False(Directory.Exists(Path.Combine(dir, "missing")));
        }
        finally { Directory.Delete(dir, recursive: true); }
    }

    /// <summary>
    /// <see cref="Persistence.Inspect"/> (issue #57) recognizes <see cref="TrainingCheckpoint.Save(string)"/>
    /// output via the marker tensor and reports the checkpoint format version, the global step, and
    /// the per-section tensor listing — all matching what was written, from the SafeTensors header
    /// plus the marker's 16 bytes only (tensor payloads are never loaded). A SafeTensors file
    /// without the marker inspects as plain <see cref="ArtifactKind.SafeTensors"/>, not as a
    /// checkpoint.
    /// </summary>
    [Fact]
    public void TestCheckpointInspectRecognizesSavedCheckpoint()
    {
        var rig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            SGDMomentumOptimizer.ComputationGraph,
            new NamedModelParam[]
            {
                new TensorDataModelParam("input", ModelParamType.InputParam,
                    TensorData([4L], new float[] { 1f, 2f, 3f, 4f })),
            },
            0.5f, 0.9f);
        var ckpt0 = rig.CreateInitialCheckpoint();
        var ckpt = new TrainingCheckpoint(                       // same tensors, at step 5
            ckpt0.TrainableParams, ckpt0.ModelState, ckpt0.OptimizerState, step: 5);

        var path = Path.Combine(Path.GetTempPath(), $"shrk_inspect_{Guid.NewGuid():N}.safetensors");
        try
        {
            ckpt.Save(path);
            var result = Persistence.Inspect(path);

            Assert.Equal(ArtifactKind.TrainingCheckpoint, result.Kind);
            Assert.Empty(result.Observations);
            Assert.NotNull(result.SafeTensors);   // a checkpoint is a SafeTensors file
            Assert.Null(result.Srk);

            var info = result.TrainingCheckpoint!;
            Assert.Equal(3, info.FormatVersion);   // v3 marker: int64 [version, step] + presence-gated epoch/batch/loss
            Assert.Equal(5, info.Step);
            Assert.Null(info.Epoch);               // not set on this checkpoint → unknown (null), never a sentinel 0
            Assert.Null(info.BatchIndex);

            // Per-section listing matches the rig's struct defs field-for-field (names with
            // the section prefix stripped; SGD-momentum carries per-param velocity state,
            // ScalarMultiply has no model state).
            string[] sectionNames = ["trainable", "model_state", "opt_state"];
            Assert.Equal(sectionNames.Length, info.Sections.Count);
            foreach (var section in sectionNames)
                Assert.Contains(section, info.Sections.Keys);
            Assert.Equal(
                rig.TrainableParamStructDef.Fields.Select(f => f.Name),
                info.Sections["trainable"].Select(t => t.Name));
            Assert.Equal(
                rig.ModelStateDef.Fields.Select(f => f.Name),
                info.Sections["model_state"].Select(t => t.Name));
            Assert.Equal(
                rig.OptimizerStateDef.Fields.Select(f => f.Name),
                info.Sections["opt_state"].Select(t => t.Name));
            Assert.NotEmpty(info.Sections["opt_state"]);

            // The listed metadata matches the written tensors.
            var trainableField = rig.TrainableParamStructDef.Fields[0];
            var written = (TensorData)ckpt.TrainableParams.Fields[trainableField.Name];
            var listed = info.Sections["trainable"].Single(t => t.Name == trainableField.Name);
            Assert.Equal(written.Shape.Dims, listed.Shape);
            Assert.Equal("F32", listed.DType);

            var text = result.ToString();
            Assert.Contains("training checkpoint", text);
            Assert.Contains("global step: 5", text);

            // Without the marker, the same tensors inspect as plain SafeTensors weights.
            var plainPath = Path.Combine(Path.GetTempPath(), $"shrk_inspect_plain_{Guid.NewGuid():N}.safetensors");
            try
            {
                var plainTensors = new List<SafeTensor>
                {
                    new SafeTensor(trainableField.Name, written,
                        SafeTensorLoader.DTypeToSafeTensorDType(written.DType), written.Shape.Dims),
                };
                SafeTensorLoader.SaveSafeTensors(plainPath, plainTensors);
                Assert.Equal(ArtifactKind.SafeTensors, Persistence.Inspect(plainPath).Kind);
                Assert.Null(Persistence.Inspect(plainPath).TrainingCheckpoint);
            }
            finally { if (File.Exists(plainPath)) File.Delete(plainPath); }
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    /// <summary>
    /// Dynamic optimizer hyperparameters, in both flavours. A <see cref="Hyperparameter.Runtime"/> hyper
    /// routes the learning rate as a schedule-less runtime input (<see cref="TrainingRig.HyperparameterStructDef"/>)
    /// supplied each step; a built-in <see cref="Schedule"/> is instead lowered and evaluated entirely
    /// in-graph from the step counter (#99), so it is not a runtime field. Drives the wiring in
    /// <c>BuildTrainingStepPureGraph</c> (hyperparam struct input + GETFIELDs, in-graph scheduler splice,
    /// step-counter input, input reorder, real names), <c>InitializeAndOptimize</c> (seed values), the
    /// named/single <see cref="TrainingRig.MakeHyperparameters(float)"/>, and both the auto (no-override) and
    /// explicit-override <c>TrainStep</c> overloads.
    ///
    /// Correctness check: from one starting state the SGD update is <c>w − lr·grad</c>, so two
    /// steps that differ only in LR must move the weight by exactly the LR ratio. That linear
    /// response — measured on a single compiled graph — is what proves the LR is genuinely live
    /// and not baked.
    /// </summary>
    [Fact]
    public void TestDynamicHyperparamScheduleCoverage()
    {
        var sample = new NamedModelParam[]
        {
            new TensorDataModelParam("input", ModelParamType.InputParam,
                TensorData([4L], new float[] { 1f, 2f, 3f, 4f })),
        };
        // SGD's learning rate is its sole hyperparameter. Hyperparameter.Runtime marks it as a
        // schedule-less runtime input so we can inject explicit values and prove the LR is live.
        var rig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            sample, new SGDOptimizerHyperparameters { LearningRate = Hyperparameter.Runtime() });

        Assert.Single(rig.HyperparameterStructDef.Fields);
        Assert.Equal(new[] { 0 }, rig.DynamicHyperparameterIndices);
        // Real hyperparameter names now flow end-to-end (not "hyperparam_0").
        Assert.Equal(new[] { "learningRate" }, rig.DynamicHyperparameterNames);
        Assert.Equal("learningRate", rig.HyperparameterStructDef.Fields[0].Name);

        var modelInputDef = new TensorStructDef(
            new[] { new TensorStructFieldDef("input", DataStructure.Tensor, 1, DType.Float32) }, "ModelInput");
        var targetDef = new TensorStructDef(
            new[] { new TensorStructFieldDef("targets", DataStructure.Tensor, 1, DType.Float32) }, "Target");
        var inputBatch = new TensorDataStruct(modelInputDef,
            new Dictionary<string, IData> { { "input", TensorData([4L], new float[] { 1f, 2f, 3f, 4f }) } });
        var targetBatch = new TensorDataStruct(targetDef,
            new Dictionary<string, IData> { { "targets", TensorData([4L], new float[] { 0f, 0f, 0f, 0f }) } });

        var initial = rig.CreateInitialCheckpoint();
        Assert.Equal(0, initial.Step);
        string wName = rig.TrainableParamStructDef.Fields[0].Name;
        float w0 = ((TensorData<float32>)initial.TrainableParams.Fields[wName]).AccessMemory()[0];

        // Same start state, two different runtime learning rates, supplied via the explicit-override
        // TrainStep, once by the single-value helper and once by name.
        var stepA = rig.TrainStep(initial, rig.MakeHyperparameters(0.1f), inputBatch, targetBatch);
        var stepB = rig.TrainStep(initial, rig.MakeHyperparameters(("learningRate", 0.3f)), inputBatch, targetBatch);
        Assert.Equal(1, stepA.Step);   // the global step counter advanced
        float wA = ((TensorData<float32>)stepA.TrainableParams.Fields[wName]).AccessMemory()[0];
        float wB = ((TensorData<float32>)stepB.TrainableParams.Fields[wName]).AccessMemory()[0];

        float deltaA = w0 - wA;   // = 0.1 · grad
        float deltaB = w0 - wB;   // = 0.3 · grad
        Assert.True(MathF.Abs(deltaA) > 1e-4f, "LR must actually move the weight (grad·lr ≠ 0).");
        Assert.True(MathF.Abs(stepA.Loss!.Value - stepB.Loss!.Value) < 1e-4f);       // identical starting state
        Assert.True(MathF.Abs(deltaB - 3f * deltaA) < 1e-4f,           // 3× LR ⇒ 3× step
            $"expected ΔB ≈ 3·ΔA; got ΔA={deltaA}, ΔB={deltaB}");

        // A schedule-less runtime hyperparameter cannot be auto-driven: the no-override TrainStep throws.
        Assert.Throws<InvalidOperationException>(() => rig.TrainStep(initial, inputBatch, targetBatch));
        // Named MakeHyperparameters rejects unknown / missing names.
        Assert.Throws<ArgumentException>(() => rig.MakeHyperparameters(("bogus", 0.1f)));

        // A built-in Schedule is lowered and applied entirely in-graph (#99): it is NOT a runtime
        // "hyperparams" field, so the schedule rig has an empty HyperparameterStructDef and the no-override
        // TrainStep drives it from the checkpoint's step. The in-graph value at step 0 of Linear(0.2→0, 4)
        // is 0.2, so the auto step must match a Runtime reference rig fed 0.2 explicitly.
        var schedRig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            sample, new SGDOptimizerHyperparameters { LearningRate = Schedules.Linear(0.2f, 0.0f, 4) });
        Assert.Empty(schedRig.HyperparameterStructDef.Fields);              // schedule is in-graph, not a runtime field
        Assert.Empty(schedRig.DynamicHyperparameterIndices);
        var sc = schedRig.CreateInitialCheckpoint();
        string swName = schedRig.TrainableParamStructDef.Fields[0].Name;
        var autoStep = schedRig.TrainStep(sc, inputBatch, targetBatch);   // no override; step 0 ⇒ LR 0.2
        float swAuto = ((TensorData<float32>)autoStep.TrainableParams.Fields[swName]).AccessMemory()[0];

        var refRig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            sample, new SGDOptimizerHyperparameters { LearningRate = Hyperparameter.Runtime() });
        var refStep = refRig.TrainStep(refRig.CreateInitialCheckpoint(), refRig.MakeHyperparameters(0.2f),
            inputBatch, targetBatch);
        float swRef = ((TensorData<float32>)refStep.TrainableParams.Fields[swName]).AccessMemory()[0];
        Assert.True(MathF.Abs(swAuto - swRef) < 1e-5f, "in-graph scheduled step must equal explicit LR = schedule(step).");

        // Scheduled LR also works for a stateful optimizer (AdamW: 5 hyperparams, m/v state), built with
        // the named set — LR scheduled in-graph, everything else left at its [Hyper] default (baked). No
        // hyperparameter is a runtime field, so DynamicHyperparameterIndices is empty.
        var adamRig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, AdamWOptimizer.ComputationGraph,
            sample, new AdamWOptimizerHyperparameters { LearningRate = Schedules.Constant(0.01f) });
        Assert.Empty(adamRig.DynamicHyperparameterIndices);                // LR is scheduled in-graph; betas baked
        Assert.Empty(adamRig.HyperparameterStructDef.Fields);
        var adamStep = adamRig.TrainStep(adamRig.CreateInitialCheckpoint(), inputBatch, targetBatch);
        Assert.True(float.IsFinite(adamStep.Loss!.Value));
        Assert.NotEmpty(adamStep.OptimizerState.Fields);    // m/v state still flows
    }

    /// <summary>
    /// Coverage for the <see cref="Schedule"/> factories and fluent combinators: each is a pure
    /// <c>step → value</c> function, so this checks their numerics directly (no rig needed). Mirrors
    /// the one-liner style of the rig coverage tests above.
    /// </summary>
    [Fact]
    public void TestScheduleCombinatorsCoverage()
    {
        static void Eq(float expected, float actual) => Assert.True(MathF.Abs(expected - actual) < 1e-4f,
            $"expected {expected}, got {actual}");

        // Factories.
        Eq(0.5f, Schedules.Constant(0.5f).At(123));
        Eq(1.0f, Schedules.Linear(1.0f, 0.0f, 10).At(0));
        Eq(0.5f, Schedules.Linear(1.0f, 0.0f, 10).At(5));
        Eq(0.0f, Schedules.Linear(1.0f, 0.0f, 10).At(10));
        Eq(1.0f, Schedules.Cosine(1.0f, 8).At(0));          // starts at base
        Eq(0.0f, Schedules.Cosine(1.0f, 8).At(8));          // decays to ~0
        Eq(0.25f, Schedules.StepDecay(1.0f, 2, 0.5f).At(4)); // 1·0.5^(4/2)
        Eq(0.25f, Schedules.Exponential(1.0f, 0.5f).At(2));  // 1·0.5^2

        // CosineWithWarmup: linear ramp up to base over warmup, then cosine decay; peak hit at warmup end.
        var cw = Schedules.CosineWithWarmup(1.0f, warmupSteps: 4, totalSteps: 12);
        Eq(0.25f, cw.At(0));   // base·(0+1)/4
        Eq(1.0f, cw.At(3));    // base·(3+1)/4  → peak
        Assert.True(cw.At(11) < 0.05f);     // decayed toward 0

        // WithWarmup composed onto a bare cosine matches CosineWithWarmup.
        var composed = Schedules.Cosine(1.0f, 8).WithWarmup(4);
        Eq(cw.At(0), composed.At(0));
        Eq(cw.At(7), composed.At(7));

        // Scale / Clamp / Shift / PerEpoch / Then.
        Eq(2.0f, Schedules.Constant(1.0f).Scale(2.0f).At(0));
        Eq(1.0f, Schedules.Linear(0f, 5f, 5).Clamp(0f, 1f).At(4));
        Eq(Schedules.Linear(0f, 5f, 5).At(3), Schedules.Linear(0f, 5f, 5).Shift(1).At(2));
        var perEpoch = Schedules.Linear(0f, 4f, 4).PerEpoch(stepsPerEpoch: 3);
        Eq(perEpoch.At(0), perEpoch.At(2));                  // constant within an epoch
        Assert.True(MathF.Abs(perEpoch.At(2) - perEpoch.At(3)) > 1e-6f);  // changes at the epoch boundary
        var joined = Schedules.Constant(1.0f).Then(atStep: 3, Schedules.Constant(2.0f));
        Eq(1.0f, joined.At(2));
        Eq(2.0f, joined.At(3));

        // OneCycle: anneals up from max/divFactor to max, then down below the start.
        var oc = Schedules.OneCycle(maxValue: 1.0f, totalSteps: 100, pctStart: 0.3f, divFactor: 25f);
        Eq(1.0f / 25f, oc.At(0));
        Assert.True(oc.At(30) > oc.At(0));      // climbed toward the peak
        Assert.True(oc.At(99) < oc.At(0));      // ended below the start
    }

    /// <summary>
    /// Coverage that a schedule genuinely drives the <see cref="TrainingRig.Fit"/>/<see cref="TrainingRig.Train"/>
    /// loop: the global step advances across the loop (so the schedule sees increasing steps), and two
    /// rigs differing only in their learning-rate schedule reach different final weights — the loop-level
    /// analogue of the per-step liveness proof. Also covers multi-dynamic named MakeHyperparameters via
    /// SGD-with-momentum.
    /// </summary>
    [Fact]
    public void TestSchedulesDriveTrainingLoopCoverage()
    {
        var sample = new NamedModelParam[]
        {
            new TensorDataModelParam("input", ModelParamType.InputParam,
                TensorData([4L], new float[] { 1f, 2f, 3f, 4f })),
        };
        var modelInputDef = new TensorStructDef(
            new[] { new TensorStructFieldDef("input", DataStructure.Tensor, 1, DType.Float32) }, "ModelInput");
        var targetDef = new TensorStructDef(
            new[] { new TensorStructFieldDef("targets", DataStructure.Tensor, 1, DType.Float32) }, "Target");
        var inputBatch = new TensorDataStruct(modelInputDef,
            new Dictionary<string, IData> { { "input", TensorData([4L], new float[] { 1f, 2f, 3f, 4f }) } });
        var targetBatch = new TensorDataStruct(targetDef,
            new Dictionary<string, IData> { { "targets", TensorData([4L], new float[] { 0f, 0f, 0f, 0f }) } });

        float FinalWeight(Schedule lr)
        {
            var rig = TrainingRig.FromScratch(
                ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
                sample, new SGDOptimizerHyperparameters { LearningRate = lr });
            var ckpt = rig.CreateInitialCheckpoint();
            // 4 batches → Fit advances the global step each batch, so the schedule is sampled at 0..3.
            var result = rig.Fit(
                [inputBatch, inputBatch, inputBatch, inputBatch],
                [targetBatch, targetBatch, targetBatch, targetBatch],
                numEpochs: 1, ckpt);
            Assert.Single(result.EpochLosses);
            var wName = rig.TrainableParamStructDef.Fields[0].Name;
            return ((TensorData<float32>)result.FinalCheckpoint.TrainableParams.Fields[wName]).AccessMemory()[0];
        }

        // A decaying schedule and a constant-at-the-initial-value schedule take different total steps,
        // because the decaying one shrinks the LR over the four batches.
        float wDecay = FinalWeight(Schedules.Linear(0.2f, 0.0f, 4));
        float wConst = FinalWeight(Schedules.Constant(0.2f));
        Assert.True(MathF.Abs(wDecay - wConst) > 1e-4f,
            $"a live schedule must change the trajectory vs constant LR; got {wDecay} vs {wConst}");

        // Multi-dynamic: SGD-with-momentum, both hyperparameters scheduled; named MakeHyperparameters must
        // accept both names (order-independent) and reject a wrong set.
        var momRig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, SGDMomentumOptimizer.ComputationGraph,
            sample, new SGDMomentumOptimizerHyperparameters
            {
                LearningRate = Hyperparameter.Runtime(),
                MomentumCoeff = Hyperparameter.Runtime(),
            });
        Assert.Equal(new[] { "learningRate", "momentumCoeff" }, momRig.DynamicHyperparameterNames.ToArray());
        var momStep = momRig.TrainStep(momRig.CreateInitialCheckpoint(),
            momRig.MakeHyperparameters(("momentumCoeff", 0.9f), ("learningRate", 0.1f)),  // order-independent
            inputBatch, targetBatch);
        Assert.True(float.IsFinite(momStep.Loss!.Value));
        Assert.NotEmpty(momStep.OptimizerState.Fields);     // velocity state flows
        Assert.Throws<ArgumentException>(() => momRig.MakeHyperparameters(("learningRate", 0.1f))); // missing momentumCoeff
    }

    // ───────────────────────── in-graph scheduler (#99) ─────────────────────────

    /// <summary>Sample input + a fixed input/target batch for the <see cref="ScalarMultiplyModel"/> rigs.</summary>
    private static (NamedModelParam[] sample, TensorDataStruct input, TensorDataStruct target) ScalarMultiplyBatches()
    {
        var sample = new NamedModelParam[]
        {
            new TensorDataModelParam("input", ModelParamType.InputParam,
                TensorData([4L], new float[] { 1f, 2f, 3f, 4f })),
        };
        var modelInputDef = new TensorStructDef(
            new[] { new TensorStructFieldDef("input", DataStructure.Tensor, 1, DType.Float32) }, "ModelInput");
        var targetDef = new TensorStructDef(
            new[] { new TensorStructFieldDef("targets", DataStructure.Tensor, 1, DType.Float32) }, "Target");
        var input = new TensorDataStruct(modelInputDef,
            new Dictionary<string, IData> { { "input", TensorData([4L], new float[] { 1f, 2f, 3f, 4f }) } });
        var target = new TensorDataStruct(targetDef,
            new Dictionary<string, IData> { { "targets", TensorData([4L], new float[] { 0f, 0f, 0f, 0f }) } });
        return (sample, input, target);
    }

    /// <summary>Wraps a <c>int64 step → float32 value</c> body as a scheduler module graph.</summary>
    private static ComputationGraph SchedulerModule(Func<Scalar<int64>, Scalar<float32>> body)
    {
        var step = InputScalar<int64>("step");
        var value = body(step);
        return new ComputationGraph(new InternalComputationGraph([step], [value]), GraphKind.Module);
    }

    /// <summary>Wraps an arbitrary (possibly ill-formed) input/output set as a scheduler module graph.</summary>
    private static ComputationGraph SchedulerModuleRaw(Variable[] inputs, Variable[] outputs)
        => new(new InternalComputationGraph([.. inputs], [.. outputs]), GraphKind.Module);

    /// <summary>
    /// A built-in <see cref="Schedule"/> lowered and evaluated <b>in-graph</b> from the step counter (#99)
    /// must, step for step, match the previous host-evaluated path — modelled here by a reference rig fed
    /// <see cref="Schedule.At"/> explicitly. The schedule includes <c>Cos</c>, so the in-graph value carries
    /// #39's few-ulps ORT transcendental tolerance; the weight it drives stays within that of the host-fed
    /// weight. Same model / loss / optimizer / default seed ⇒ both rigs share the initial weight and, given
    /// equal LR each step, stay in lockstep — the only difference is where the LR comes from.
    /// </summary>
    [Fact]
    public void TestLoweredBuiltinScheduleParityWithHostValues()
    {
        var (sample, inputBatch, targetBatch) = ScalarMultiplyBatches();
        var schedule = Schedules.Cosine(0.05f, 6).WithWarmup(2);   // Cos ⇒ exercises the #39 tolerance

        var schedRig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            sample, new SGDOptimizerHyperparameters { LearningRate = schedule });
        Assert.Empty(schedRig.HyperparameterStructDef.Fields);        // schedule is in-graph, not a runtime field

        var refRig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            sample, new SGDOptimizerHyperparameters { LearningRate = Hyperparameter.Runtime() });

        var schedCkpt = schedRig.CreateInitialCheckpoint();
        var refCkpt = refRig.CreateInitialCheckpoint();
        string wName = schedRig.TrainableParamStructDef.Fields[0].Name;

        for (int s = 0; s < 6; s++)
        {
            schedCkpt = schedRig.TrainStep(schedCkpt, inputBatch, targetBatch); // in-graph LR at step s
            refCkpt = refRig.TrainStep(refCkpt, refRig.MakeHyperparameters(schedule.At(s)),                   // host LR at step s
                inputBatch, targetBatch);
            float wSched = ((TensorData<float32>)schedCkpt.TrainableParams.Fields[wName]).AccessMemory()[0];
            float wRef = ((TensorData<float32>)refCkpt.TrainableParams.Fields[wName]).AccessMemory()[0];
            Assert.True(MathF.Abs(wSched - wRef) < 1e-5f,
                $"step {s}: in-graph scheduled weight {wSched} vs host-fed {wRef} beyond #39 tolerance.");
        }
        Assert.Equal(6, schedCkpt.Step);   // step counter advanced across the loop
    }

    /// <summary>
    /// A user-supplied scheduler <b>module</b> (int64 step → float32 value, #99) inlines into the rig and
    /// drives training exactly as feeding its value explicitly does: over several steps the module-scheduled
    /// weight tracks a reference rig fed the module's <c>lr(step)</c> via <see cref="Hyperparameter.Runtime"/>.
    /// The module here is pure arithmetic, so the match is exact.
    /// </summary>
    [Fact]
    public void TestUserSchedulerModuleRoundTrips()
    {
        var (sample, inputBatch, targetBatch) = ScalarMultiplyBatches();

        // lr(step) = 0.3 − 0.05·step, authored as a module over the int64 step counter.
        var schedulerModule = SchedulerModule(step => Scalar(0.3f) - step.Cast<float32>() * Scalar(0.05f));
        float Lr(int s) => 0.3f - 0.05f * s;

        var modRig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            sample, new SGDOptimizerHyperparameters { LearningRate = Hyperparameter.Scheduled(schedulerModule) });
        Assert.Empty(modRig.HyperparameterStructDef.Fields);          // module is in-graph, not a runtime field

        var refRig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            sample, new SGDOptimizerHyperparameters { LearningRate = Hyperparameter.Runtime() });

        var modCkpt = modRig.CreateInitialCheckpoint();
        var refCkpt = refRig.CreateInitialCheckpoint();
        string wName = modRig.TrainableParamStructDef.Fields[0].Name;

        for (int s = 0; s < 4; s++)
        {
            modCkpt = modRig.TrainStep(modCkpt, inputBatch, targetBatch);
            refCkpt = refRig.TrainStep(refCkpt, refRig.MakeHyperparameters(Lr(s)),
                inputBatch, targetBatch);
            float wMod = ((TensorData<float32>)modCkpt.TrainableParams.Fields[wName]).AccessMemory()[0];
            float wRef = ((TensorData<float32>)refCkpt.TrainableParams.Fields[wName]).AccessMemory()[0];
            Assert.True(MathF.Abs(wMod - wRef) < 1e-5f,
                $"step {s}: user-module scheduled weight {wMod} vs host-fed lr={Lr(s)} weight {wRef} differ.");
        }
    }

    /// <summary>
    /// The two-source contract (#99) fails loud at rig build for everything outside it: a scheduler module
    /// with the wrong signature (bad input dtype, wrong input count, non-float32 output) and an opaque,
    /// non-lowerable schedule are each rejected by <see cref="TrainingRig.FromScratch"/>. (There is no
    /// host-lambda schedule API to reject — it was removed at compile time.)
    /// </summary>
    [Fact]
    public void TestBadSchedulerFailsLoudAtRigBuild()
    {
        var (sample, _, _) = ScalarMultiplyBatches();
        TrainingRig Build(Hyperparameter lr) => TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            sample, new SGDOptimizerHyperparameters { LearningRate = lr });

        // Wrong input dtype: a float32 step counter instead of int64.
        var floatStep = InputScalar<float32>("step");
        var floatInputModule = SchedulerModuleRaw([floatStep], [floatStep]);
        Assert.Throws<ArgumentException>(() => Build(Hyperparameter.Scheduled(floatInputModule)));

        // Wrong input count: two inputs.
        var a = InputScalar<int64>("a");
        var b = InputScalar<int64>("b");
        var twoInputModule = SchedulerModuleRaw([a, b], [a.Cast<float32>() + b.Cast<float32>()]);
        Assert.Throws<ArgumentException>(() => Build(Hyperparameter.Scheduled(twoInputModule)));

        // Wrong output dtype: an int64 (not float32) value.
        var intStep = InputScalar<int64>("step");
        var intOutputModule = SchedulerModuleRaw([intStep], [intStep]);
        Assert.Throws<ArgumentException>(() => Build(Hyperparameter.Scheduled(intOutputModule)));

        // An opaque, non-lowerable schedule (built internally) is likewise rejected at build.
        Assert.Throws<ArgumentException>(() => Build(Hyperparameter.Scheduled(new Schedule((ScheduleExpr?)null))));
    }

    // ───────────────────────── P3: one value route (#105) ─────────────────────────

    /// <summary>The single scalar the <see cref="InitFromHyperOptimizer"/> writes into its optimizer
    /// state at fresh-checkpoint creation — which equals the learning-rate hyper's value at the initial
    /// counters, so it directly reads back the value the §2.5 route fed to state init.</summary>
    private static float FreshOptStateValue(TrainingRig rig, TrainingCheckpoint ckpt)
    {
        var field = rig.OptimizerStateDef.Fields[0].Name;
        return ((TensorData<float32>)ckpt.OptimizerState.Fields[field]).AccessMemory()[0];
    }

    /// <summary>
    /// The §2.5 value route: optimizer state init sees each scheduled hyper's real value at the initial
    /// counters — a built-in <see cref="Schedule"/> via its lowered graph, and a scheduler <b>module</b>
    /// via QEE of its graph — not a placeholder. The module case explicitly pins that the old hardcoded
    /// <c>0f</c> state-init hole (a module fed 0 to state init while feeding its true value to the
    /// trainstep) is gone. <see cref="InitFromHyperOptimizer"/> initializes its state to the LR hyper,
    /// so the fresh optimizer-state value <em>is</em> that fed value.
    /// </summary>
    [Fact]
    public void TestScheduledHyperStateInitUsesGraphValueCoverage()
    {
        var (sample, _, _) = ScalarMultiplyBatches();
        TrainingRig Build(Hyperparameter lr) => TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, InitFromHyperOptimizer.ComputationGraph,
            sample, new InitFromHyperOptimizerHyperparameters { LearningRate = lr });

        // Built-in schedule: state init sees Constant(0.5).At(0) = 0.5.
        var dslRig = Build(Schedules.Constant(0.5f));
        Assert.True(MathF.Abs(0.5f - FreshOptStateValue(dslRig, dslRig.CreateInitialCheckpoint())) < 1e-4f);

        // A decaying schedule: state init sees the step-0 value (0.2), not the final value.
        var decayRig = Build(Schedules.Linear(0.2f, 0f, 10));
        Assert.True(MathF.Abs(0.2f - FreshOptStateValue(decayRig, decayRig.CreateInitialCheckpoint())) < 1e-4f);

        // Scheduler module returning 0.7 at step 0: state init must see 0.7 — NOT the old 0f.
        var moduleRig = Build(Hyperparameter.Scheduled(
            SchedulerModule(step => Scalar(0.7f) + step.Cast<float32>() * Scalar(0f))));
        float moduleState = FreshOptStateValue(moduleRig, moduleRig.CreateInitialCheckpoint());
        Assert.True(MathF.Abs(0.7f - moduleState) < 1e-4f, $"expected 0.7, got {moduleState}");
        Assert.True(MathF.Abs(moduleState) > 1e-4f, "the old hardcoded-0f scheduler-module state-init hole must be gone.");
    }

    /// <summary>
    /// D5: when the optimizer's state initializer actually reads a <see cref="HyperparameterKind.Runtime"/>
    /// hyper (dependency-analyzed at build), fresh-checkpoint creation fails loud unless explicit initial
    /// values are supplied via <see cref="TrainingRig.CreateInitialCheckpoint(TensorDataStruct)"/>. A
    /// baked hyper never trips this, since its value is known at build.
    /// </summary>
    [Fact]
    public void TestRuntimeHyperStateInitFailsLoudUnlessSuppliedCoverage()
    {
        var (sample, _, _) = ScalarMultiplyBatches();
        var runtimeRig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, InitFromHyperOptimizer.ComputationGraph,
            sample, new InitFromHyperOptimizerHyperparameters { LearningRate = Hyperparameter.Runtime() });

        // No value known at build → fail loud (D5), naming the hyper.
        var ex = Assert.Throws<InvalidOperationException>(() => runtimeRig.CreateInitialCheckpoint());
        Assert.Contains("learningRate", ex.Message);

        // Supplied explicitly → state init uses the given value.
        var ckpt = runtimeRig.CreateInitialCheckpoint(runtimeRig.MakeHyperparameters(0.3f));
        Assert.True(MathF.Abs(0.3f - FreshOptStateValue(runtimeRig, ckpt)) < 1e-4f);

        // A baked LR needs no override — its value is known at build.
        var bakedRig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, InitFromHyperOptimizer.ComputationGraph,
            sample, new InitFromHyperOptimizerHyperparameters { LearningRate = 0.05f });
        Assert.True(MathF.Abs(0.05f - FreshOptStateValue(bakedRig, bakedRig.CreateInitialCheckpoint())) < 1e-4f);
    }

    /// <summary>
    /// D4 purity enforcement: a scheduler module that carries a trainable parameter, module state (a
    /// StateUpdate), or an RNG draw is rejected at rig build with a message naming the purity contract —
    /// it would otherwise be inlined into the trainstep with an undefined failure mode.
    /// </summary>
    [Fact]
    public void TestImpureSchedulerModuleRejectedAtRigBuildCoverage()
    {
        var (sample, _, _) = ScalarMultiplyBatches();
        TrainingRig Build(ComputationGraph schedulerModule) => TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            sample, new SGDOptimizerHyperparameters { LearningRate = Hyperparameter.Scheduled(schedulerModule) });

        foreach (var impure in new[] { ParamScheduler.ComputationGraph, StateScheduler.ComputationGraph, RngScheduler.ComputationGraph })
        {
            var ex = Assert.Throws<ArgumentException>(() => Build(impure));
            Assert.Contains("pure", ex.Message);
        }
    }

    // ───────────────────────── P4: int64 counters (#105) ─────────────────────────

    /// <summary>
    /// int64 counter round-trip: a checkpoint whose step/epoch/batchIndex exceed int32 survives Save
    /// and Load in both formats — the flat safetensors marker (v3) and the native .skpt manifest
    /// — with no truncation. Pins the widening of <see cref="TrainingCheckpoint.Step"/> et al. to int64.
    /// </summary>
    [Fact]
    public void TestInt64CounterRoundTripCoverage()
    {
        var (_, trained, _, _) = BuildTrainedAdamRig(steps: 1);
        long bigStep = 5_000_000_000L;              // > int.MaxValue
        long bigEpoch = 3_000_000_000L;             // > int.MaxValue
        long bigBatch = (long)int.MaxValue + 7L;    // just past int32
        var ckpt = new TrainingCheckpoint(
            trained.TrainableParams, trained.ModelState, trained.OptimizerState,
            step: bigStep, epoch: bigEpoch, batchIndex: bigBatch, rig: trained.Rig);

        var legacyPath = Path.Combine(Path.GetTempPath(), $"shrk_i64_{Guid.NewGuid():N}.safetensors");
        var skptPath = Path.Combine(Path.GetTempPath(), $"shrk_i64_{Guid.NewGuid():N}.skpt");
        try
        {
            // Legacy flat safetensors (v3 marker).
            ckpt.Save(legacyPath);
            var legacy = BuildTrainedAdamRig(steps: 0).Rig.LoadCheckpoint(legacyPath);
            Assert.Equal(bigStep, legacy.Step);
            Assert.Equal(bigEpoch, legacy.Epoch);
            Assert.Equal(bigBatch, legacy.BatchIndex);
            Assert.Equal(3, Persistence.Inspect(legacyPath).TrainingCheckpoint!.FormatVersion);

            // Native .skpt manifest.
            Persistence.SaveTrainingCheckpointToSkpt(ckpt, skptPath);
            var skpt = BuildTrainedAdamRig(steps: 0).Rig.LoadCheckpoint(skptPath);
            Assert.Equal(bigStep, skpt.Step);
            Assert.Equal(bigEpoch, skpt.Epoch);
            Assert.Equal(bigBatch, skpt.BatchIndex);
        }
        finally
        {
            if (File.Exists(legacyPath)) File.Delete(legacyPath);
            if (File.Exists(skptPath)) File.Delete(skptPath);
        }
    }

    /// <summary>
    /// D1 multi-counter scheduler module: a module consuming both the <c>step</c> and <c>epoch</c>
    /// reserved counters is fed each from the checkpoint, so its value tracks a reference rig fed the
    /// same <c>lr(step, epoch)</c> explicitly. Proves the rig creates and wires the named counter subset
    /// (not just step) and feeds epoch (a host-owned counter not derivable from step).
    /// </summary>
    [Fact]
    public void TestMultiCounterSchedulerModuleConsumesEpochCoverage()
    {
        var (sample, inputBatch, targetBatch) = ScalarMultiplyBatches();
        var ctx = new ComputeContext();
        float Lr(long s, long e) => 0.5f - 0.01f * s - 0.1f * e;

        var modRig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            sample, new SGDOptimizerHyperparameters { LearningRate = Hyperparameter.Scheduled(StepEpochScheduler.ComputationGraph) });
        var refRig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            sample, new SGDOptimizerHyperparameters { LearningRate = Hyperparameter.Runtime() });

        string wName = modRig.TrainableParamStructDef.Fields[0].Name;

        // Several (step, epoch) points — epoch varies independently of step (host-owned).
        foreach (var (s, e) in new[] { (0L, 0L), (3L, 1L), (7L, 4L) })
        {
            var seed = modRig.CreateInitialCheckpoint();
            var atCkpt = new TrainingCheckpoint(seed.TrainableParams, seed.ModelState, seed.OptimizerState, step: s, epoch: e);
            var modStep = modRig.TrainStep(atCkpt, inputBatch, targetBatch);

            var refSeed = refRig.CreateInitialCheckpoint();
            var refAt = new TrainingCheckpoint(refSeed.TrainableParams, refSeed.ModelState, refSeed.OptimizerState, step: s, epoch: e);
            var refStep = refRig.TrainStep(refAt, refRig.MakeHyperparameters(Lr(s, e)), inputBatch, targetBatch);

            float wMod = ((TensorData<float32>)modStep.TrainableParams.Fields[wName]).AccessMemory()[0];
            float wRef = ((TensorData<float32>)refStep.TrainableParams.Fields[wName]).AccessMemory()[0];
            Assert.True(MathF.Abs(wMod - wRef) < 1e-5f,
                $"(step {s}, epoch {e}): module-scheduled weight {wMod} vs host lr={Lr(s, e)} weight {wRef} differ.");
        }
    }

    /// <summary>
    /// Coverage for a model whose training graph carries batched (3-D) matmuls
    /// (<see cref="BatchedMatmulModel"/>): drives the MatMul gradient's rank-agnostic
    /// last-two-dims transpose through the FromScratch autograd pipeline. Execution and
    /// gradient-value correctness of that batched backward are checked by the self-checking
    /// <c>AutoGradMatMulUnknownRankBatchedCheck</c> coverage module.
    /// </summary>
    [Fact]
    public void TestBatchedMatmulCoverage()
    {
        CoverFromScratch(BatchedMatmulModel.ComputationGraph, SoftmaxL2Loss.ComputationGraph,
            SGDOptimizer.ComputationGraph, [4L, 5L, 8L], 0.01f);
    }

    /// <summary>
    /// A trained checkpoint must preserve the inference model's canonical (dotted) param
    /// names so it round-trips straight back through <c>graph.ToConcreteModel(...)</c> by
    /// name (previously the training side sanitized '.'→'_' and <c>ToConcreteModel</c> threw
    /// <c>KeyNotFoundException</c>).
    /// </summary>
    [Fact]
    public void TestTrainedCheckpointRebindsByName()
    {
        CoverCheckpointRebind(DigitClassifier.ComputationGraph, SoftmaxL2Loss.ComputationGraph,
            SGDOptimizer.ComputationGraph, [4L, 64L], 0.01f);
    }

    /// <summary>
    /// Trainable-parameter discovery (<c>GetConcreteModelParamInfos</c> /
    /// <c>InitializeTrainableParams</c>) only scans top-level nodes. On a raw module graph
    /// whose sub-modules are still un-inlined (here <see cref="CallsSimplestModule"/> wraps
    /// <c>SimplestLayer</c>) the trainable param is nested inside a sub-function, so the guard
    /// rejects the graph instead of silently returning an empty set. After
    /// <c>ToConcreteArchitecture</c> the param is top-level and discovery succeeds.
    /// </summary>
    [Fact]
    public void TestParamDiscoveryRequiresConcreteArchitecture()
    {
        // Use a mutable deep copy so the internal (op-scan-guarded) extensions are
        // exercised directly — the public ComputationGraph extensions would reject the
        // module-kind graph on its Kind stamp before the discovery guard ever ran.
        var moduleGraph = CallsSimplestModule.ComputationGraph.ToInternal();
        Assert.Contains(moduleGraph.Nodes, n =>
            n.OpCode == InternalOpCodes.MODEL_INVOKE || n.OpCode == InternalOpCodes.FUNCTION_INVOKE);

        Assert.Throws<System.InvalidOperationException>(() => moduleGraph.GetConcreteModelParamInfos());
        Assert.Throws<System.InvalidOperationException>(() => moduleGraph.InitializeTrainableParams());

        var sample = TensorData([4L], new float[] { 1f, 2f, 3f, 4f });
        var arch = moduleGraph.ToConcreteArchitecture(moduleGraph.FromOrderedInputs([sample]));
        Assert.NotEmpty(arch.GetConcreteModelParamInfos().ParamInfos);
        Assert.NotEmpty(arch.InitializeTrainableParams().ModelParams);
    }

    /// <summary>
    /// Covers <see cref="TrainingLoop.LowerTrainingGraph"/> — the minimal
    /// autograd-flatten pipeline also exercised by the
    /// AutoDiffCheckpointing tests.
    /// </summary>
    [Fact]
    public void TestTrainingLoopCoverage()
    {
        var trainingGraph = TrainingGraphBuilder.PrepareForTrainingAsFast(
            ScalarMultiplyModel.ComputationGraph.ToInternal(),
            L2Loss.ComputationGraph.ToInternal());

        var lowered = TrainingLoop.LowerTrainingGraph(trainingGraph);
        Assert.NotNull(lowered);
        Assert.NotEmpty(lowered.Nodes);
    }

    /// <summary>
    /// Covers the <c>Func</c>-loss overload of
    /// <see cref="TrainingGraphBuilder.PrepareForTrainingAsFast{TOut,TLoss}(InternalComputationGraph, Func{TOut,TOut,TLoss})"/>
    /// and its companion reflection helper
    /// <see cref="TrainingGraphBuilder.ExtractFastGraphFromDelegate"/>, plus
    /// the three argument-validation error paths. These are otherwise only
    /// hit by <c>TrainingGraphBuilderQuickTests</c>.
    /// </summary>
    [Fact]
    public void TestTrainingGraphBuilderFuncOverloadCoverage()
    {
        // PrepareForTrainingAsFast is typed on the mutable internal graph.
        var modelGraph = ScalarMultiplyModel.ComputationGraph.ToInternal();

        // Happy path: Func referencing a [Module]'s Inline method.
        Func<Tensor<float32>, Tensor<float32>, Scalar<float32>> lossFunc = L2Loss.Inline;
        var trainingGraph = TrainingGraphBuilder.PrepareForTrainingAsFast(modelGraph, lossFunc);
        Assert.True(trainingGraph.Inputs.Count >= 3);
        Assert.True(trainingGraph.Outputs.Count >= 2);

        // Argument-validation error paths.
        Assert.Throws<ArgumentNullException>(() =>
            TrainingGraphBuilder.PrepareForTrainingAsFast<Tensor<float32>, Scalar<float32>>(modelGraph, null!));
        Assert.Throws<ArgumentNullException>(() =>
            TrainingGraphBuilder.PrepareForTrainingAsFast<Tensor<float32>, Scalar<float32>>(null!, lossFunc));

        // ExtractFastGraphFromDelegate rejects non-module delegates (lambda's
        // Method.Name is not "Inline").
        Func<Tensor<float32>, Tensor<float32>, Scalar<float32>> notAModule =
            (pred, targ) => ((Tensor<float32>)OnnxOp.ReduceSum(pred - targ, keepdims: false)).Scalar();
        Assert.Throws<ArgumentException>(() =>
            TrainingGraphBuilder.PrepareForTrainingAsFast(modelGraph, notAModule));
    }

    /// <summary>
    /// Verifies the <see cref="Losses"/> hub returns non-null graphs identical to
    /// the underlying XxxLoss.ComputationGraph properties.
    /// </summary>
    [Fact]
    public void TestLossesHubCoverage()
    {
        Assert.NotNull(Losses.L2Loss);
        Assert.NotNull(Losses.L1Loss);
        Assert.NotNull(Losses.CrossEntropy);
        Assert.NotNull(Losses.BCE);
        Assert.NotNull(Losses.BCEWithLogits);
        Assert.NotNull(Losses.SmoothL1);
        Assert.NotNull(Losses.Huber);
        Assert.NotNull(Losses.Hinge);
        Assert.NotNull(Losses.SquaredHinge);
        Assert.NotNull(Losses.KLDiv);
        Assert.NotNull(Losses.NLL);
        Assert.NotNull(Losses.PoissonNLL);
        Assert.NotNull(Losses.LogCosh);
        Assert.NotNull(Losses.CosineEmbedding);
        Assert.NotNull(Losses.TripletMargin);
        Assert.NotNull(Losses.BinaryFocal);
        // Rig-safe losses always have exactly 2 inputs: (predictions, targets).
        // (.ToInternal() is a read-only borrow — do not mutate the shared cached graph.)
        Assert.Equal(2, Losses.L2Loss.ToInternal().Inputs.Count);
        Assert.Equal(2, Losses.L1Loss.ToInternal().Inputs.Count);
    }

    /// <summary>
    /// Verifies the <see cref="Optimizers"/> hub returns non-null graphs identical to
    /// the underlying XxxOptimizer.ComputationGraph properties.
    /// </summary>
    [Fact]
    public void TestOptimizersHubCoverage()
    {
        Assert.NotNull(Optimizers.SGD);
        Assert.NotNull(Optimizers.SGDMomentum);
        Assert.NotNull(Optimizers.Adam);
        Assert.NotNull(Optimizers.AdamW);
        Assert.NotNull(Optimizers.Adamax);
        Assert.NotNull(Optimizers.NAdam);
        Assert.NotNull(Optimizers.Adagrad);
        Assert.NotNull(Optimizers.Adadelta);
        Assert.NotNull(Optimizers.RMSprop);
        Assert.NotNull(Optimizers.RAdam);
        Assert.NotNull(Optimizers.Lamb);
        Assert.NotNull(Optimizers.Lion);
        Assert.NotNull(Optimizers.Adafactor);
    }

    /// <summary>
    /// Verifies the <see cref="TrainingRig.FromScratch(ComputationGraph,ComputationGraph,ComputationGraph,ModelParamList,Hyperparameter[])"/>
    /// overload, the <see cref="TrainingRig.InputDef"/> and <see cref="TrainingRig.TargetDef"/> properties,
    /// and <see cref="TensorStructDef.FromOrderedData"/> — covering all the convenience APIs
    /// added to clean up training call sites.
    /// </summary>
    [Fact]
    public void TestFromScratchModelParamListAndStructDefsCoverage()
    {
        var modelGraph   = ScalarMultiplyModel.ComputationGraph;
        var exampleInput = TensorData([4L], new float[] { 1f, 2f, 3f, 4f });

        var rig = TrainingRig.FromScratch(
            modelGraph, Losses.L2Loss, Optimizers.SGD,
            modelGraph.FromOrderedInputs([exampleInput]),
            0.01f);

        // InputDef should have one field ("input" from the model's parameter name).
        Assert.NotNull(rig.InputDef);
        Assert.Equal(1, rig.InputDef.Fields.Length);
        Assert.Equal("input", rig.InputDef.Fields[0].Name);

        // TargetDef should have one field ("targets" from L2Loss's second parameter).
        Assert.NotNull(rig.TargetDef);
        Assert.Equal(1, rig.TargetDef.Fields.Length);
        Assert.Equal("targets", rig.TargetDef.Fields[0].Name);

        // FromOrderedData should produce a TensorDataStruct matching the field count.
        var inputBatch  = rig.InputDef.FromOrderedData(exampleInput);
        var targetBatch = rig.TargetDef.FromOrderedData(TensorData([4L], new float[4]));
        Assert.NotNull(inputBatch);
        Assert.NotNull(targetBatch);
        Assert.Same(rig.InputDef,  inputBatch.Definition);
        Assert.Same(rig.TargetDef, targetBatch.Definition);

        // Fit with defaults (no checkpoint, no ctx) should complete and produce a finite loss.
        var result = rig.Fit([inputBatch, inputBatch], [targetBatch, targetBatch], numEpochs: 1);
        Assert.Single(result.EpochLosses);
        Assert.True(float.IsFinite(result.EpochLosses[0]));
    }

    // ---------------------------------------------------------------------------
    // Data loader (issue #111): interface + in-memory implementation + rig integration.
    // These use a dataset where sample i carries the value i in every feature, so the
    // floats read back out of a gathered batch reveal exactly which sample indices the
    // loader drew — letting the shuffle order and resume position be checked exactly.
    // ---------------------------------------------------------------------------

    /// <summary>Builds a rig over ScalarMultiply + L2Loss whose batch shape is [batchSize, features].</summary>
    private static TrainingRig LoaderRig(int batchSize, int features) =>
        TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            [
                new TensorDataModelParam("input", ModelParamType.InputParam,
                    TensorData([batchSize, features], new float[batchSize * features])),
            ],
            0.1f);

    /// <summary>Dataset of <paramref name="n"/> samples where sample i holds the value i in each of its
    /// <paramref name="features"/> features, plus a matching (constant-zero) target set.</summary>
    private static (TensorDataStruct inputs, TensorDataStruct targets) IndexDataset(
        TrainingRig rig, int n, int features)
    {
        float[] inVals = new float[n * features];
        for (int i = 0; i < n; i++)
            for (int f = 0; f < features; f++)
                inVals[i * features + f] = i;
        var inputs = new TensorDataStruct(rig.InputDef,
            new Dictionary<string, IData> { { "input", TensorData([n, (long)features], inVals) } });
        var targets = new TensorDataStruct(rig.TargetDef,
            new Dictionary<string, IData> { { "targets", TensorData([n, (long)features], new float[n * features]) } });
        return (inputs, targets);
    }

    /// <summary>Drains one full epoch and returns the sample index each batch row was gathered from.</summary>
    private static int[] EpochIndexSequence(InMemoryDataLoader loader, int features)
    {
        var seq = new List<int>();
        for (int b = 0; b < loader.BatchesPerEpoch; b++)
        {
            var batch = loader.Next();
            var vals = ((TensorData)batch.Input.Fields["input"]).As<float32>().AccessMemory().ToArray();
            for (int r = 0; r < vals.Length; r += features)
                seq.Add((int)vals[r]);
        }
        return seq.ToArray();
    }

    /// <summary>
    /// Covers <see cref="InMemoryDataLoader"/> batching, position tracking / epoch rollover, and
    /// deterministic per-epoch shuffle. dropLast drops the trailing partial batch; the shuffle is a
    /// genuine, seed-determined permutation (identical across loaders sharing a seed, regenerated
    /// bit-for-bit when an epoch is revisited) and differs from the identity order.
    /// </summary>
    [Fact]
    public void TestInMemoryDataLoaderBatchingAndShuffleCoverage()
    {
        const int features = 1;
        var rig = LoaderRig(batchSize: 2, features);
        var (inputs, targets) = IndexDataset(rig, n: 8, features);

        // dropLast=true: 8 samples / batch 2 = 4 batches; identity order in-order.
        var plain = new InMemoryDataLoader(inputs, targets, batchSize: 2);
        Assert.Equal(4, plain.BatchesPerEpoch);
        Assert.Equal(8, plain.SampleCount);
        Assert.Equal(new DataLoaderPosition(0, 0), plain.Position);
        int[] identity = [0, 1, 2, 3, 4, 5, 6, 7];
        Assert.Equal(identity, EpochIndexSequence(plain, features));
        // After a full epoch the position has rolled to the start of epoch 1.
        Assert.Equal(new DataLoaderPosition(1, 0), plain.Position);

        // dropLast=false keeps a trailing partial batch: 9 samples / batch 2 = 5 batches.
        var (inputs9, targets9) = IndexDataset(rig, n: 9, features);
        var keepPartial = new InMemoryDataLoader(inputs9, targets9, batchSize: 2, dropLast: false);
        Assert.Equal(5, keepPartial.BatchesPerEpoch);
        var partialSeq = EpochIndexSequence(keepPartial, features);
        Assert.Equal(9, partialSeq.Length);   // the lone final sample is included
        Assert.Equal(Enumerable.Range(0, 9), partialSeq);

        // Position stepping within an epoch.
        var stepper = new InMemoryDataLoader(inputs, targets, batchSize: 2);
        stepper.Next();
        Assert.Equal(new DataLoaderPosition(0, 1), stepper.Position);
        stepper.Next();
        Assert.Equal(new DataLoaderPosition(0, 2), stepper.Position);

        // Shuffle: same seed → identical order; a revisited epoch regenerates the same order;
        // the order is a real permutation and (for this seed) not the identity.
        var s1 = new InMemoryDataLoader(inputs, targets, batchSize: 2, shuffle: true, seed: 12345);
        var s2 = new InMemoryDataLoader(inputs, targets, batchSize: 2, shuffle: true, seed: 12345);
        int[] order1 = EpochIndexSequence(s1, features);
        int[] order2 = EpochIndexSequence(s2, features);
        Assert.Equal(order1, order2);                        // deterministic across loaders
        Assert.Equal(identity, order1.OrderBy(i => i));      // a genuine permutation of 0..7
        Assert.NotEqual(identity, order1);                   // actually reordered

        // Different epoch → different draw order (epoch is mixed into the seed).
        int[] epoch1Order = EpochIndexSequence(s1, features);
        Assert.Equal(identity, epoch1Order.OrderBy(i => i));
        Assert.NotEqual(order1, epoch1Order);

        // Revisiting epoch 0 (via RestoreFrom) regenerates epoch 0's order exactly.
        var s3 = new InMemoryDataLoader(inputs, targets, batchSize: 2, shuffle: true, seed: 12345);
        s3.RestoreFrom(new DataLoaderPosition(0, 0));
        Assert.Equal(order1, EpochIndexSequence(s3, features));

        // Out-of-range batch index is rejected (guards against a mismatched checkpoint/loader).
        Assert.Throws<ArgumentOutOfRangeException>(() => plain.RestoreFrom(new DataLoaderPosition(0, 4)));
        Assert.Throws<ArgumentOutOfRangeException>(() => plain.RestoreAfter(new DataLoaderPosition(0, 4)));

        // RestoreAfter advances one batch past the given (used) position, rolling into the next epoch
        // after an epoch's last batch — the resume primitive for the "checkpoint stores the batch used"
        // convention. From the last batch of epoch 0 (index 3) it lands at (1, 0); mid-epoch it just +1s.
        var afterStepper = new InMemoryDataLoader(inputs, targets, batchSize: 2);
        afterStepper.RestoreAfter(new DataLoaderPosition(0, 1));
        Assert.Equal(new DataLoaderPosition(0, 2), afterStepper.Position);
        afterStepper.RestoreAfter(new DataLoaderPosition(0, 3));   // last batch of the epoch
        Assert.Equal(new DataLoaderPosition(1, 0), afterStepper.Position);
    }

    /// <summary>
    /// Covers <see cref="TrainingRig.Fit(IDataLoader, int, TrainingCheckpoint?)"/>:
    /// driving a loader advances the checkpoint's step, epoch, and batch counters automatically — with
    /// no host hand-setting — so a run of E epochs over B batches/epoch lands at step = E*B and records
    /// the <b>batch used</b> at the last step (epoch = E-1, batchIndex = B-1). Reports one mean loss per
    /// epoch. The invariant step == epoch*B + batchIndex + 1 (batches used) holds for a single-loader run.
    /// </summary>
    [Fact]
    public void TestFitWithDataLoaderAdvancesCountersCoverage()
    {
        const int features = 4;
        var rig = LoaderRig(batchSize: 2, features);
        var (inputs, targets) = IndexDataset(rig, n: 6, features);
        var loader = new InMemoryDataLoader(inputs, targets, batchSize: 2);   // 3 batches/epoch
        Assert.Equal(3, loader.BatchesPerEpoch);

        var result = rig.Fit(loader, numEpochs: 2);

        Assert.Equal(2, result.EpochLosses.Length);
        Assert.All(result.EpochLosses, l => Assert.True(float.IsFinite(l)));

        var final = result.FinalCheckpoint;
        Assert.Equal(6, final.Step);        // 2 epochs * 3 batches
        Assert.Equal(1, final.Epoch);       // the batch USED at the last step: epoch 1 (0-based)...
        Assert.Equal(2, final.BatchIndex);  // ...batch index 2 (the epoch's last batch)
        Assert.Equal(final.Epoch * loader.BatchesPerEpoch + final.BatchIndex + 1, final.Step);

        // The loader itself is left rolled past that last batch, at the next epoch's start.
        Assert.Equal(new DataLoaderPosition(2, 0), loader.Position);
    }

    /// <summary>
    /// Covers loader-position resume end to end. A run split by a save → fresh-rig-and-loader reload →
    /// continue must reach the exact same trained weights and counters as one uninterrupted run — the
    /// deterministic shuffle plus <see cref="IDataLoader.RestoreAfter"/> (resume advances one past the
    /// batch the checkpoint recorded as used) make the resumed batch stream identical. Separately, a
    /// mid-epoch batch-used position round-trips through a saved checkpoint and the reloaded loader,
    /// RestoreAfter'd from it, yields the very next batch the original would have.
    /// </summary>
    [Fact]
    public void TestDataLoaderResumeRoundTripCoverage()
    {
        const int features = 4;
        const long seed = 777;

        // --- End-to-end: uninterrupted 2-epoch run vs a 1+1 split with save/reload between. ---
        var rigRef = LoaderRig(batchSize: 2, features);
        var (inRef, tgtRef) = IndexDataset(rigRef, n: 6, features);
        var loaderRef = new InMemoryDataLoader(inRef, tgtRef, batchSize: 2, shuffle: true, seed: seed);
        var refResult = rigRef.Fit(loaderRef, numEpochs: 2);
        float[] refWeights = FlattenStruct(refResult.FinalCheckpoint.TrainableParams);

        var path = Path.Combine(Path.GetTempPath(), $"shrk_loader_resume_{Guid.NewGuid():N}.safetensors");
        try
        {
            // First half: one epoch, then save mid-run.
            var rigA = LoaderRig(batchSize: 2, features);
            var (inA, tgtA) = IndexDataset(rigA, n: 6, features);
            var loaderA = new InMemoryDataLoader(inA, tgtA, batchSize: 2, shuffle: true, seed: seed);
            var half = rigA.Fit(loaderA, numEpochs: 1);
            Assert.Equal(3, half.FinalCheckpoint.Step);
            Assert.Equal(0, half.FinalCheckpoint.Epoch);       // batch used at the last step: epoch 0...
            Assert.Equal(2, half.FinalCheckpoint.BatchIndex);  // ...its last batch (index 2)
            half.FinalCheckpoint.Save(path);

            // Second half: fresh rig + fresh loader, load the checkpoint, continue one more epoch.
            var rigB = LoaderRig(batchSize: 2, features);
            var (inB, tgtB) = IndexDataset(rigB, n: 6, features);
            var loaderB = new InMemoryDataLoader(inB, tgtB, batchSize: 2, shuffle: true, seed: seed);
            var loaded = rigB.LoadCheckpoint(path);
            Assert.Equal(0, loaded.Epoch);       // the batch-used position round-trips
            Assert.Equal(2, loaded.BatchIndex);
            var resumed = rigB.Fit(loaderB, numEpochs: 1, loaded);  // RestoreAfter((0,2)) → (1,0)

            // Same counters and — because the batch stream was reproduced exactly — the same weights.
            Assert.Equal(refResult.FinalCheckpoint.Step, resumed.FinalCheckpoint.Step);
            Assert.Equal(refResult.FinalCheckpoint.Epoch, resumed.FinalCheckpoint.Epoch);
            Assert.Equal(refWeights, FlattenStruct(resumed.FinalCheckpoint.TrainableParams));
        }
        finally { if (File.Exists(path)) File.Delete(path); }

        // --- Mid-epoch batch-used position round-trips through a saved checkpoint. ---
        var rigM = LoaderRig(batchSize: 2, features: 1);
        var (inM, tgtM) = IndexDataset(rigM, n: 8, features: 1);

        // Reference loader that has USED two batches (indices 0 and 1); the batches it would yield next.
        var refLoader = new InMemoryDataLoader(inM, tgtM, batchSize: 2, shuffle: true, seed: seed);
        refLoader.Next(); refLoader.Next();                       // use 2 batches → position now (0, 2)
        var lastUsed = new DataLoaderPosition(0, 1);              // the batch USED at the last step
        var midPos = refLoader.Position;
        Assert.Equal(new DataLoaderPosition(0, 2), midPos);
        int[] refTail = TailIndices(refLoader, features: 1);

        // Persist the batch-used position via a checkpoint, reload it, and RestoreAfter a fresh loader.
        var midPath = Path.Combine(Path.GetTempPath(), $"shrk_loader_midpos_{Guid.NewGuid():N}.safetensors");
        try
        {
            var ckpt0 = rigM.CreateInitialCheckpoint();
            var midCkpt = new TrainingCheckpoint(
                ckpt0.TrainableParams, ckpt0.ModelState, ckpt0.OptimizerState,
                step: 2, epoch: lastUsed.Epoch, batchIndex: lastUsed.BatchIndex);
            midCkpt.Save(midPath);
            var reloaded = rigM.LoadCheckpoint(midPath);
            Assert.Equal(lastUsed.Epoch, reloaded.Epoch);
            Assert.Equal(lastUsed.BatchIndex, reloaded.BatchIndex);

            var restored = new InMemoryDataLoader(inM, tgtM, batchSize: 2, shuffle: true, seed: seed);
            restored.RestoreAfter(new DataLoaderPosition(reloaded.Epoch!.Value, reloaded.BatchIndex!.Value));
            Assert.Equal(midPos, restored.Position);                     // RestoreAfter((0,1)) → (0,2)
            Assert.Equal(refTail, TailIndices(restored, features: 1));   // continues from the exact next batch
        }
        finally { if (File.Exists(midPath)) File.Delete(midPath); }
    }

    /// <summary>Reads the remaining sample indices in the current epoch without rolling past its end.</summary>
    private static int[] TailIndices(InMemoryDataLoader loader, int features)
    {
        var seq = new List<int>();
        long remaining = loader.BatchesPerEpoch - loader.Position.BatchIndex;
        for (long b = 0; b < remaining; b++)
        {
            var batch = loader.Next();
            var vals = ((TensorData)batch.Input.Fields["input"]).As<float32>().AccessMemory().ToArray();
            for (int r = 0; r < vals.Length; r += features)
                seq.Add((int)vals[r]);
        }
        return seq.ToArray();
    }

    /// <summary>
    /// Covers <see cref="TrainingRig.TrainStep(TrainingCheckpoint, IDataLoader)"/>: a
    /// single loader-driven step advances <see cref="TrainingCheckpoint.Step"/> by one, records the
    /// position of the batch it <b>used</b> (the drawn batch's own position) on the checkpoint's epoch /
    /// batch, and preserves the rig + loss — while the loader itself advances one batch (rolling into the
    /// next epoch after the last). Looping this overload by hand reproduces
    /// <see cref="TrainingRig.Fit(IDataLoader, int, TrainingCheckpoint?)"/>
    /// exactly (same counters and weights), pinning that Fit routes through this one source of truth.
    /// </summary>
    [Fact]
    public void TestTrainStepWithLoaderAdvancesCountersCoverage()
    {
        const int features = 4;
        var rig = LoaderRig(batchSize: 2, features);
        var (inputs, targets) = IndexDataset(rig, n: 6, features);      // 3 batches/epoch
        var loader = new InMemoryDataLoader(inputs, targets, batchSize: 2);
        Assert.Equal(3, loader.BatchesPerEpoch);

        var ckpt = rig.CreateInitialCheckpoint();

        // Step 1: uses batch (0,0); records that used position. The loader advances to (0,1).
        var s1 = rig.TrainStep(ckpt, loader);
        Assert.Equal(1, s1.Step);
        Assert.Equal(0, s1.Epoch);
        Assert.Equal(0, s1.BatchIndex);
        Assert.True(float.IsFinite(s1.Loss!.Value));
        Assert.Same(rig, s1.Rig);                               // rig preserved via WithCounters
        Assert.Equal(new DataLoaderPosition(0, 1), loader.Position);

        // Step 2: uses batch (0,1); records (0,1). Loader advances to (0,2).
        var s2 = rig.TrainStep(s1, loader);
        Assert.Equal(2, s2.Step);
        Assert.Equal(0, s2.Epoch);
        Assert.Equal(1, s2.BatchIndex);

        // Step 3: uses the last batch of epoch 0 (0,2); records it. The loader rolls to (1,0).
        var s3 = rig.TrainStep(s2, loader);
        Assert.Equal(3, s3.Step);
        Assert.Equal(0, s3.Epoch);
        Assert.Equal(2, s3.BatchIndex);
        Assert.Equal(new DataLoaderPosition(1, 0), loader.Position);

        // Hand-looping TrainStep(loader) for one epoch equals Fit(loader, numEpochs: 1): the same
        // final counters and — since the batch stream is identical — the same trained weights.
        var fitRig = LoaderRig(batchSize: 2, features);
        var (fin, ftg) = IndexDataset(fitRig, n: 6, features);
        var fitLoader = new InMemoryDataLoader(fin, ftg, batchSize: 2);
        var fitResult = fitRig.Fit(fitLoader, numEpochs: 1);
        Assert.Equal(fitResult.FinalCheckpoint.Step, s3.Step);
        Assert.Equal(fitResult.FinalCheckpoint.Epoch, s3.Epoch);
        Assert.Equal(fitResult.FinalCheckpoint.BatchIndex, s3.BatchIndex);
        Assert.Equal(FlattenStruct(fitResult.FinalCheckpoint.TrainableParams), FlattenStruct(s3.TrainableParams));
    }

    /// <summary>
    /// Covers <see cref="TrainingRig.TrainStep(TrainingCheckpoint, TensorDataStruct, TensorDataStruct, long, long)"/>:
    /// the explicit epoch / batch overload records the given <c>epoch</c> / <c>batchNumber</c> verbatim on
    /// the returned checkpoint (with <see cref="TrainingCheckpoint.Step"/> advanced), and feeds them to a
    /// scheduler as the counters in effect for the step — matching the same rig stepped with a checkpoint
    /// carrying those counters directly. Negative counters are rejected.
    /// </summary>
    [Fact]
    public void TestTrainStepWithExplicitEpochBatchRecordsCountersCoverage()
    {
        var (sample, inputBatch, targetBatch) = ScalarMultiplyBatches();

        // A rig whose learning rate is a scheduler module reading (step, epoch), so the epoch counter
        // supplied to the explicit overload demonstrably drives the step (not just gets recorded).
        var rig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            sample, new SGDOptimizerHyperparameters { LearningRate = Hyperparameter.Scheduled(StepEpochScheduler.ComputationGraph) });
        string wName = rig.TrainableParamStructDef.Fields[0].Name;

        // Explicit overload at (epoch 4, batch 7): the returned checkpoint records exactly those.
        var initial = rig.CreateInitialCheckpoint();
        var stepped = rig.TrainStep(initial, inputBatch, targetBatch, epoch: 4, batchNumber: 7);
        Assert.Equal(initial.Step + 1, stepped.Step);
        Assert.Equal(4, stepped.Epoch);
        Assert.Equal(7, stepped.BatchIndex);
        Assert.True(float.IsFinite(stepped.Loss!.Value));
        Assert.Same(rig, stepped.Rig);

        // The supplied epoch is the counter in effect for the step: a reference stepped from a
        // checkpoint that already carries (step 0, epoch 4) produces the identical weight.
        var refAt = new TrainingCheckpoint(
            initial.TrainableParams, initial.ModelState, initial.OptimizerState, step: 0, epoch: 4);
        var refStep = rig.TrainStep(refAt, inputBatch, targetBatch);
        float wExplicit = ((TensorData<float32>)stepped.TrainableParams.Fields[wName]).AccessMemory()[0];
        float wRef = ((TensorData<float32>)refStep.TrainableParams.Fields[wName]).AccessMemory()[0];
        Assert.True(MathF.Abs(wExplicit - wRef) < 1e-6f,
            $"explicit-epoch weight {wExplicit} vs reference {wRef} differ — the supplied epoch did not drive the step.");

        Assert.Throws<ArgumentOutOfRangeException>(
            () => rig.TrainStep(initial, inputBatch, targetBatch, epoch: -1, batchNumber: 0));
        Assert.Throws<ArgumentOutOfRangeException>(
            () => rig.TrainStep(initial, inputBatch, targetBatch, epoch: 0, batchNumber: -1));
    }

    /// <summary>
    /// Verifies the parameterless <see cref="TrainingCheckpoint.ToInferenceModel()"/> produces a
    /// concrete model that executes successfully and returns the expected output shape — reading the
    /// model graph and sample-input shape off the checkpoint's rig (no re-supplied graph).
    /// </summary>
    [Fact]
    [Trait("Purpose", "Coverage")]
    [Trait("Domain", "Training")]
    public void TestToInferenceModelCoverage()
    {
        var modelGraph   = ScalarMultiplyModel.ComputationGraph;
        var exampleInput = TensorData([4L], new float[] { 1f, 2f, 3f, 4f });

        var rig    = TrainingRig.FromScratch(
            modelGraph, Losses.L2Loss, Optimizers.SGD,
            modelGraph.FromOrderedInputs([exampleInput]),
            0.01f);
        var result = rig.Fit(
            [rig.InputDef.FromOrderedData(exampleInput)],
            [rig.TargetDef.FromOrderedData(TensorData([4L], new float[4]))],
            numEpochs: 1);

        // Parameterless: the rig on the checkpoint supplies the model graph + sample-input shape [4].
        var concrete       = result.FinalCheckpoint.ToInferenceModel();
        Assert.NotNull(concrete);

        var inferenceInput = TensorData([4L], new float[] { 5f, 6f, 7f, 8f });
        var outputs = ComputeContext.Default.Execute(concrete, inferenceInput);
        Assert.Single(outputs);
        var output = outputs[0].ToTensorData<float32>();
        Assert.Equal(1, output.Shape.Dims.Length);
        Assert.Equal(4L, output.Shape.Dims[0]);
    }

    /// <summary>
    /// Issue #54: FromScratch takes the model as a module graph or an
    /// already-lowered concrete architecture (the lowering pipeline is idempotent
    /// on the latter), refuses a weight-filled concrete model up front naming the
    /// actual vs required kinds, and still requires module graphs in the loss and
    /// optimizer positions — instead of failing deep inside lowering.
    /// </summary>
    [Fact]
    public void TestFromScratchModelGraphKinds()
    {
        var sample = TensorData([4L], 1f, 2f, 3f, 4f);
        var modelGraph = ScalarMultiplyModel.ComputationGraph;
        var arch = modelGraph.ToConcreteArchitecture(modelGraph.FromOrderedInputs([sample]));

        var sampleInput = new TensorDataModelParam("input", ModelParamType.InputParam, sample);

        // A pre-concretized architecture is accepted in the model position and
        // produces a working rig.
        var rig = TrainingRig.FromScratch(
            arch, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            [sampleInput], 0.5f);
        Assert.NotEmpty(rig.TrainableParamStructDef.Fields);
        Assert.Equal(GraphKind.ConcreteModel, rig.TrainingStepPureGraph.Kind);
        Assert.NotNull(rig.CreateInitialCheckpoint().TrainableParams);

        // A weight-filled concrete model is refused with the kinds named.
        var model = arch.ToConcreteModel();
        var exModel = Assert.Throws<InvalidOperationException>(() => TrainingRig.FromScratch(
            model, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            [sampleInput], 0.5f));
        Assert.Contains("'concrete-model'", exModel.Message);
        Assert.Contains("'concrete-architecture'", exModel.Message);

        // The loss (and optimizer) positions still require module graphs.
        var exLoss = Assert.Throws<InvalidOperationException>(() => TrainingRig.FromScratch(
            modelGraph, arch, SGDOptimizer.ComputationGraph,
            [sampleInput], 0.5f));
        Assert.Contains("'module'", exLoss.Message);
    }

    // ──────────────────────────────────────────────────────────────────────
    // Native .skpt training checkpoints (issue #95): a training checkpoint
    // persists into the .skpt container with training state split into per-kind
    // data entries plus the concrete inference model, and reloads bit-identically.
    // ──────────────────────────────────────────────────────────────────────

    private static readonly long[] ScalarInputShape = [4L];

    private static (TrainingRig Rig, TrainingCheckpoint Ckpt, TensorDataStruct In, TensorDataStruct Out)
        BuildTrainedAdamRig(int steps)
    {
        var sample = new NamedModelParam[]
        {
            new TensorDataModelParam("input", ModelParamType.InputParam,
                TensorData(ScalarInputShape, new float[] { 1f, 2f, 3f, 4f })),
        };
        var rig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            AdamWOptimizer.ComputationGraph, sample,
            new AdamWOptimizerHyperparameters { LearningRate = 0.1f });

        var inDef = new TensorStructDef(
            [new TensorStructFieldDef("input", DataStructure.Tensor, 1, DType.Float32)], "ModelInput");
        var outDef = new TensorStructDef(
            [new TensorStructFieldDef("targets", DataStructure.Tensor, 1, DType.Float32)], "Target");
        var inBatch = new TensorDataStruct(inDef,
            new Dictionary<string, IData> { { "input", TensorData(ScalarInputShape, new float[] { 1f, 2f, 3f, 4f }) } });
        var outBatch = new TensorDataStruct(outDef,
            new Dictionary<string, IData> { { "targets", TensorData(ScalarInputShape, new float[] { 2f, 4f, 6f, 8f }) } });

        var ckpt = rig.CreateInitialCheckpoint();
        for (int i = 0; i < steps; i++)
            ckpt = rig.TrainStep(ckpt, inBatch, outBatch);
        return (rig, ckpt, inBatch, outBatch);
    }

    /// <summary>
    /// The .skpt training-checkpoint acceptance round-trip (issue #95): a mid-training checkpoint
    /// saves to a native .skpt whose training state is split into per-kind data entries recorded in
    /// config.json (trainable weights, optimizer state; no model-state entry for a stateless model),
    /// and reloads — through a fresh rig, as a fresh process would — with the step, trainable params
    /// and optimizer state bit-identical and a resumed TrainStep matching the pre-save trajectory
    /// exactly. The file is a standard STORED zip, and it also loads as a self-describing inference
    /// model via Persistence.Load.
    /// </summary>
    [Fact]
    public void TestSkptTrainingCheckpointRoundTripResumeCoverage()
    {
        var (rigA, ckpt, inBatch, outBatch) = BuildTrainedAdamRig(steps: 2);
        Assert.Equal(2, ckpt.Step);
        Assert.NotEmpty(ckpt.OptimizerState.Fields);   // AdamW carries m/v/step per param
        Assert.Empty(ckpt.ModelState.Fields);          // ScalarMultiply is stateless

        // The reference trajectory: one more step from the in-memory checkpoint.
        var reference = rigA.TrainStep(ckpt, inBatch, outBatch);

        var path = Path.Combine(Path.GetTempPath(), $"shrk_skpt_ckpt_{Guid.NewGuid():N}.skpt");
        try
        {
            Persistence.SaveTrainingCheckpointToSkpt(ckpt, path);
            Assert.True(File.Exists(path));

            // Standard STORED zip with exactly the expected per-kind entries (read via the BCL).
            using (var zip = System.IO.Compression.ZipFile.OpenRead(path))
            {
                var names = zip.Entries.Select(e => e.FullName).OrderBy(n => n, StringComparer.Ordinal).ToArray();
                Assert.Equal(
                    new[]
                    {
                        SkptFileFormat.ConfigEntryName,
                        SkptFileFormat.OptimizerStateEntryPath,
                        SkptFileFormat.TrainableEntryPath,
                        SkptFileFormat.ModelEntryPath,
                        // Rig constituents (#115): the concrete architecture, loss, and optimizer graphs
                        // (no scheduler entry — this rig has no scheduled hyperparameter).
                        SkptFileFormat.ArchEntryPath,
                        SkptFileFormat.LossEntryPath,
                        SkptFileFormat.OptimizerEntryPath,
                    }.OrderBy(n => n, StringComparer.Ordinal),
                    names);
                Assert.All(zip.Entries, e => Assert.Equal(e.Length, e.CompressedLength));   // all STORED
                Assert.DoesNotContain(SkptFileFormat.ModelStateEntryPath, names);           // stateless → no entry
                Assert.DoesNotContain(SkptFileFormat.SchedulerEntryPath, names);            // no scheduled hyper
            }

            // The manifest records the per-kind data entries and the step in config.json.
            var manifest = SkptFileFormat.ParseManifest(
                ReadEntryBytesViaBcl(path, SkptFileFormat.ConfigEntryName), path);
            Assert.NotNull(manifest.Training);
            Assert.Equal(SkptFileFormat.TrainingCheckpointVersion, manifest.Training!.CheckpointVersion);
            Assert.Equal(2, manifest.Training.Step);
            Assert.Contains(SkptFileFormat.TrainingKindTrainableParams, manifest.Training.Kinds!.Keys);
            Assert.Contains(SkptFileFormat.TrainingKindOptimizerState, manifest.Training.Kinds.Keys);
            Assert.DoesNotContain(SkptFileFormat.TrainingKindModelState, manifest.Training.Kinds.Keys);

            // Fresh rig, as a fresh process: state + step round-trip bit-identically.
            var rigB = BuildTrainedAdamRig(steps: 0).Rig;
            var loaded = rigB.LoadCheckpoint(path);
            Assert.Equal(2, loaded.Step);
            Assert.Equal(FlattenStruct(ckpt.TrainableParams), FlattenStruct(loaded.TrainableParams));
            Assert.Equal(FlattenStruct(ckpt.OptimizerState), FlattenStruct(loaded.OptimizerState));
            Assert.Empty(loaded.ModelState.Fields);

            // Resuming from the loaded checkpoint reproduces the pre-save trajectory exactly.
            var resumed = rigB.TrainStep(loaded, inBatch, outBatch);
            Assert.Equal(3, resumed.Step);
            Assert.Equal(reference.Loss!.Value, resumed.Loss!.Value);
            Assert.Equal(
                FlattenStruct(reference.TrainableParams),
                FlattenStruct(resumed.TrainableParams));
            Assert.Equal(
                FlattenStruct(reference.OptimizerState),
                FlattenStruct(resumed.OptimizerState));

            // The .skpt is self-describing for inference: Persistence.Load rebinds a concrete model
            // that executes identically to one built straight from the checkpoint's trained weights.
            var inferenceModel = Persistence.Load(path);
            Assert.Equal(GraphKind.ConcreteModel, inferenceModel.Kind);
            var probe = TensorData(ScalarInputShape, new float[] { 5f, 6f, 7f, 8f });
            var fromCkpt = ckpt.ToInferenceModel();
            var loadedOut = ComputeContext.Default.Execute(inferenceModel, probe)[0].ToTensorData().As<float32>().AccessMemory().ToArray();
            var ckptOut = ComputeContext.Default.Execute(fromCkpt, probe)[0].ToTensorData().As<float32>().AccessMemory().ToArray();
            Assert.Equal(ckptOut, loadedOut);
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    /// <summary>
    /// From-file-alone rig reconstruction (#115): a mid-training checkpoint saves to a native .skpt
    /// carrying the rig's constituents (concrete architecture, loss, optimizer), and
    /// <see cref="TrainingRig.Load(string, ComputeContext?, ComputeContext?)"/> rebuilds the WHOLE
    /// rig — trainstep and all — from the file alone, with NO host-supplied source graphs. The
    /// reconstructed rig's resumed checkpoint restores the step, trainable params and optimizer state
    /// bit-identically; a resumed TrainStep reproduces the pre-save trajectory exactly; and the
    /// reconstructed rig extracts the same inference model.
    /// </summary>
    [Fact]
    [Trait("Purpose", "Coverage")]
    [Trait("Domain", "Training")]
    public void TestTrainingRigLoadFromFileAloneCoverage()
    {
        var (rigA, ckpt, inBatch, outBatch) = BuildTrainedAdamRig(steps: 3);
        var reference = rigA.TrainStep(ckpt, inBatch, outBatch);

        var path = Path.Combine(Path.GetTempPath(), $"shrk_rigload_{Guid.NewGuid():N}.skpt");
        try
        {
            Persistence.SaveTrainingCheckpointToSkpt(ckpt, path);

            // Rebuild the whole rig from the file alone — no ScalarMultiplyModel / L2Loss / AdamWOptimizer
            // graphs are supplied here, as a fresh process resuming a run would not have them.
            var (rig2, loaded) = TrainingRig.Load(path);
            Assert.Same(rig2, loaded.Rig);
            Assert.Equal(ckpt.Step, loaded.Step);
            Assert.Equal(FlattenStruct(ckpt.TrainableParams), FlattenStruct(loaded.TrainableParams));
            Assert.Equal(FlattenStruct(ckpt.OptimizerState), FlattenStruct(loaded.OptimizerState));

            // The reconstructed rig resumes exactly (same trainstep math, same restored step).
            var resumed = rig2.TrainStep(loaded, inBatch, outBatch);
            Assert.Equal(reference.Step, resumed.Step);
            Assert.Equal(reference.Loss!.Value, resumed.Loss!.Value);
            Assert.Equal(FlattenStruct(reference.TrainableParams), FlattenStruct(resumed.TrainableParams));
            Assert.Equal(FlattenStruct(reference.OptimizerState), FlattenStruct(resumed.OptimizerState));

            // The reconstructed rig extracts the same inference model as the original checkpoint.
            var probe = TensorData(ScalarInputShape, new float[] { 5f, 6f, 7f, 8f });
            var fromReconstructed = loaded.ToInferenceModel();
            var fromOriginal = ckpt.ToInferenceModel();
            var a = ComputeContext.Default.Execute(fromReconstructed, probe)[0].ToTensorData().As<float32>().AccessMemory().ToArray();
            var b = ComputeContext.Default.Execute(fromOriginal, probe)[0].ToTensorData().As<float32>().AccessMemory().ToArray();
            Assert.Equal(b, a);
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    /// <summary>
    /// From-file rig reconstruction folds in the scheduler constituent (#106): a rig with a scheduled
    /// learning rate composes its per-hyperparameter scheduler graph into the persisted
    /// <c>scheduler</c> model entry, and <see cref="TrainingRig.Load(string, ComputeContext?, ComputeContext?)"/>
    /// splits it back to a scheduled hyperparameter binding. The reconstructed rig resumes at the saved
    /// step and reproduces the pre-save trajectory exactly — the scheduler math (which reads the step
    /// counter) survives the round-trip.
    /// </summary>
    [Fact]
    [Trait("Purpose", "Coverage")]
    [Trait("Domain", "Training")]
    public void TestTrainingRigLoadReconstructsSchedulerCoverage()
    {
        var sample = new NamedModelParam[]
        {
            new TensorDataModelParam("input", ModelParamType.InputParam,
                TensorData(ScalarInputShape, new float[] { 1f, 2f, 3f, 4f })),
        };
        TrainingRig SchedRig() => TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            AdamWOptimizer.ComputationGraph, sample,
            new AdamWOptimizerHyperparameters { LearningRate = Shorokoo.Core.Training.Schedules.Cosine(0.1f, 50) });

        var inDef = new TensorStructDef(
            [new TensorStructFieldDef("input", DataStructure.Tensor, 1, DType.Float32)], "ModelInput");
        var outDef = new TensorStructDef(
            [new TensorStructFieldDef("targets", DataStructure.Tensor, 1, DType.Float32)], "Target");
        var inBatch = new TensorDataStruct(inDef,
            new Dictionary<string, IData> { { "input", TensorData(ScalarInputShape, new float[] { 1f, 2f, 3f, 4f }) } });
        var outBatch = new TensorDataStruct(outDef,
            new Dictionary<string, IData> { { "targets", TensorData(ScalarInputShape, new float[] { 2f, 4f, 6f, 8f }) } });

        var rigA = SchedRig();
        Assert.Equal(HyperparameterKind.Scheduled, rigA.Hyperparameters[0].Kind);
        var ckpt = rigA.CreateInitialCheckpoint();
        for (int i = 0; i < 5; i++)
            ckpt = rigA.TrainStep(ckpt, inBatch, outBatch);
        var reference = rigA.TrainStep(ckpt, inBatch, outBatch);

        var path = Path.Combine(Path.GetTempPath(), $"shrk_rigload_sched_{Guid.NewGuid():N}.skpt");
        try
        {
            Persistence.SaveTrainingCheckpointToSkpt(ckpt, path);

            // The composed scheduler is persisted as its own constituent model entry (#106).
            using (var zip = System.IO.Compression.ZipFile.OpenRead(path))
                Assert.Contains(SkptFileFormat.SchedulerEntryPath, zip.Entries.Select(e => e.FullName));

            var (rig2, loaded) = TrainingRig.Load(path);
            // The built-in schedule persisted and split back as a scheduler MODULE binding.
            Assert.Equal(HyperparameterKind.Scheduled, rig2.Hyperparameters[0].Kind);
            Assert.NotNull(rig2.Hyperparameters[0].AsSchedulerModule);
            Assert.Equal(ckpt.Step, loaded.Step);

            var resumed = rig2.TrainStep(loaded, inBatch, outBatch);
            Assert.Equal(reference.Step, resumed.Step);
            Assert.Equal(reference.Loss!.Value, resumed.Loss!.Value);
            Assert.Equal(FlattenStruct(reference.TrainableParams), FlattenStruct(resumed.TrainableParams));
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    /// <summary>
    /// <see cref="TrainingRig.Load(string, ComputeContext?, ComputeContext?)"/> fails loudly when the
    /// file carries no rig constituents to rebuild from: a legacy flat safetensors checkpoint (not a
    /// .skpt container at all), and an inference-only .skpt (no training/rig block). The from-file path
    /// is only for a training .skpt written with the rig; otherwise the host rebuilds the rig and uses
    /// <see cref="TrainingRig.LoadCheckpoint"/>.
    /// </summary>
    [Fact]
    [Trait("Purpose", "Coverage")]
    [Trait("Domain", "Training")]
    public void TestTrainingRigLoadFailsWithoutConstituentsCoverage()
    {
        var (_, ckpt, _, _) = BuildTrainedAdamRig(steps: 1);

        var flatPath = Path.Combine(Path.GetTempPath(), $"shrk_rigload_flat_{Guid.NewGuid():N}.safetensors");
        var infPath = Path.Combine(Path.GetTempPath(), $"shrk_rigload_inf_{Guid.NewGuid():N}.skpt");
        try
        {
            // Legacy flat checkpoint: not a .skpt container at all.
            ckpt.Save(flatPath);
            Assert.ThrowsAny<Exception>(() => TrainingRig.Load(flatPath));

            // Inference-only .skpt: a container with a model but no training/rig block.
            Persistence.From(ckpt.ToInferenceModel()).WithModel().WithWeights().Save(infPath);
            Assert.Throws<System.IO.InvalidDataException>(() => TrainingRig.Load(infPath));
        }
        finally
        {
            if (File.Exists(flatPath)) File.Delete(flatPath);
            if (File.Exists(infPath)) File.Delete(infPath);
        }
    }

    /// <summary>Reads one archive entry's bytes through the BCL zip reader (independent of the
    /// .skpt writer), for asserting on the raw config.json manifest.</summary>
    private static byte[] ReadEntryBytesViaBcl(string path, string entryName)
    {
        using var zip = System.IO.Compression.ZipFile.OpenRead(path);
        using var s = zip.GetEntry(entryName)!.Open();
        using var buf = new MemoryStream();
        s.CopyTo(buf);
        return buf.ToArray();
    }

    /// <summary>
    /// A .skpt training checkpoint carrying non-empty model state (a BatchNorm model) writes a
    /// model-state data entry too, and round-trips its running-stat state alongside trainable and
    /// optimizer state — exercising the model-state per-kind path.
    /// </summary>
    [Fact]
    public void TestSkptTrainingCheckpointModelStateCoverage()
    {
        var sample = new NamedModelParam[]
        {
            new TensorDataModelParam("input", ModelParamType.InputParam, TensorData([8L], new float[8])),
        };
        TrainingRig BnRig() => TrainingRig.FromScratch(
            ScalarMultiplyWithBatchNormModel.ComputationGraph, L2Loss.ComputationGraph,
            SGDMomentumOptimizer.ComputationGraph, sample, 0.5f, 0.9f);

        var rig = BnRig();
        var ckpt = new TrainingCheckpoint(
            rig.CreateInitialCheckpoint().TrainableParams,
            rig.CreateInitialCheckpoint().ModelState,
            rig.CreateInitialCheckpoint().OptimizerState, step: 11, rig: rig);
        Assert.NotEmpty(ckpt.ModelState.Fields);

        var path = Path.Combine(Path.GetTempPath(), $"shrk_skpt_bn_{Guid.NewGuid():N}.skpt");
        try
        {
            Persistence.SaveTrainingCheckpointToSkpt(ckpt, path);

            using (var zip = System.IO.Compression.ZipFile.OpenRead(path))
                Assert.Contains(SkptFileFormat.ModelStateEntryPath, zip.Entries.Select(e => e.FullName));

            var loaded = BnRig().LoadCheckpoint(path);
            Assert.Equal(11, loaded.Step);
            Assert.Equal(FlattenStruct(ckpt.ModelState), FlattenStruct(loaded.ModelState));
            Assert.Equal(FlattenStruct(ckpt.TrainableParams), FlattenStruct(loaded.TrainableParams));
            Assert.Equal(FlattenStruct(ckpt.OptimizerState), FlattenStruct(loaded.OptimizerState));
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    /// <summary>
    /// Persistence.Inspect recognizes a .skpt training checkpoint and reports the training block —
    /// checkpoint version, global step, and the per-kind data entries — from the manifest alone,
    /// without loading any tensor payload (Observations stays empty).
    /// </summary>
    [Fact]
    public void TestSkptTrainingCheckpointInspectCoverage()
    {
        var (_, ckpt, _, _) = BuildTrainedAdamRig(steps: 3);
        var path = Path.Combine(Path.GetTempPath(), $"shrk_skpt_inspect_{Guid.NewGuid():N}.skpt");
        try
        {
            Persistence.SaveTrainingCheckpointToSkpt(ckpt, path);

            var result = Persistence.Inspect(path);
            Assert.Equal(ArtifactKind.SkptCheckpoint, result.Kind);
            Assert.Empty(result.Observations);
            Assert.NotNull(result.Skpt);

            var training = result.Skpt!.Training;
            Assert.NotNull(training);
            Assert.Equal(SkptFileFormat.TrainingCheckpointVersion, training!.CheckpointVersion);
            Assert.Equal(3, training.Step);
            var kindNames = training.Kinds.Select(k => k.Key).ToArray();
            Assert.Contains(SkptFileFormat.TrainingKindTrainableParams, kindNames);
            Assert.Contains(SkptFileFormat.TrainingKindOptimizerState, kindNames);

            var text = result.ToString();
            Assert.Contains("training checkpoint: version 1", text);
            Assert.Contains("global step 3", text);
            Assert.Contains(SkptFileFormat.TrainingKindTrainableParams, text);
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    /// <summary>
    /// Fail-loud contract for .skpt training-checkpoint load: a checkpoint reconstructed against
    /// mismatched struct defs (a different model/optimizer) throws; a tampered data entry fails its
    /// SHA-256 check; and loading an ordinary inference .skpt (no training block) as a training
    /// checkpoint is refused. Composes with per-entry Zstd + provenance metadata on the happy path.
    /// </summary>
    [Fact]
    public void TestSkptTrainingCheckpointFailsLoudAndComposesCoverage()
    {
        var (_, ckpt, _, _) = BuildTrainedAdamRig(steps: 1);

        var path = Path.Combine(Path.GetTempPath(), $"shrk_skpt_fail_{Guid.NewGuid():N}.skpt");
        var tampered = Path.Combine(Path.GetTempPath(), $"shrk_skpt_tamper_{Guid.NewGuid():N}.skpt");
        try
        {
            // Happy path composes Zstd + metadata; the data entries declare zstd compression and
            // the checkpoint still round-trips through a fresh rig.
            Persistence.ForTrainingCheckpoint(ckpt)
                .WithZstdCompressedData()
                .WithMetadata(runName: "skpt-95-run", gitCommit: "abc123")
                .Save(path);

            var inspect = Persistence.Inspect(path);
            Assert.Empty(inspect.Observations);   // zstd is a recognized compression, not flagged
            Assert.Equal("skpt-95-run", inspect.Skpt!.UserMetadata!["runName"]);
            Assert.Contains(inspect.Skpt.DataEntries,
                d => d.Key == SkptFileFormat.TrainableDataKey && d.Compression == SkptFileFormat.CompressionZstd);
            Assert.NotNull(inspect.Skpt.Training);
            Assert.Equal(1, inspect.Skpt.Training!.Step);

            var rig = BuildTrainedAdamRig(steps: 0).Rig;
            var loaded = rig.LoadCheckpoint(path);
            Assert.Equal(1, loaded.Step);
            Assert.Equal(FlattenStruct(ckpt.TrainableParams), FlattenStruct(loaded.TrainableParams));

            // Mismatch: a checkpoint from the Adam/ScalarMultiply rig must not load into a
            // BatchNorm+SGD-momentum rig (its struct defs differ).
            var bnRig = TrainingRig.FromScratch(
                ScalarMultiplyWithBatchNormModel.ComputationGraph, L2Loss.ComputationGraph,
                SGDMomentumOptimizer.ComputationGraph,
                [new TensorDataModelParam("input", ModelParamType.InputParam, TensorData([8L], new float[8]))],
                0.5f, 0.9f);
            Assert.ThrowsAny<Exception>(() => bnRig.LoadCheckpoint(path));

            // Tamper: flip a byte inside a data entry → SHA-256 mismatch on load.
            var bytes = File.ReadAllBytes(path);
            int idx = bytes.Length / 2;
            bytes[idx] ^= 0xFF;
            File.WriteAllBytes(tampered, bytes);
            var rig2 = BuildTrainedAdamRig(steps: 0).Rig;
            Assert.ThrowsAny<Exception>(() => rig2.LoadCheckpoint(tampered));

            // An inference-only .skpt (no training block) is refused as a training checkpoint.
            var infPath = Path.Combine(Path.GetTempPath(), $"shrk_skpt_inf_{Guid.NewGuid():N}.skpt");
            try
            {
                var infModel = ckpt.ToInferenceModel();
                Persistence.From(infModel).WithModel().WithWeights().Save(infPath);
                var ex = Assert.Throws<System.IO.InvalidDataException>(() =>
                    Persistence.LoadTrainingCheckpoint(infPath,
                        rig.TrainableParamStructDef, rig.ModelStateDef, rig.OptimizerStateDef));
                Assert.Contains("training", ex.Message);
            }
            finally { if (File.Exists(infPath)) File.Delete(infPath); }
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
            if (File.Exists(tampered)) File.Delete(tampered);
        }
    }

    /// <summary>
    /// Back-compat: <see cref="TrainingRig.LoadCheckpoint"/> and
    /// <see cref="Persistence.LoadTrainingCheckpoint"/> still read a legacy flat safetensors
    /// checkpoint (written by <see cref="TrainingCheckpoint.Save"/>) — the shape is detected from the
    /// file, so old and new checkpoints load through one entry point.
    /// </summary>
    [Fact]
    public void TestLegacyFlatCheckpointStillLoadsCoverage()
    {
        var (rig, ckpt, _, _) = BuildTrainedAdamRig(steps: 2);
        var path = Path.Combine(Path.GetTempPath(), $"shrk_legacy_{Guid.NewGuid():N}.safetensors");
        try
        {
            Persistence.SaveTrainingCheckpoint(ckpt, path);   // legacy flat format
            Assert.Equal(ArtifactKind.TrainingCheckpoint, Persistence.Inspect(path).Kind);

            var rigB = BuildTrainedAdamRig(steps: 0).Rig;
            var loaded = rigB.LoadCheckpoint(path);            // routes to the legacy reader
            Assert.Equal(2, loaded.Step);
            Assert.Equal(FlattenStruct(ckpt.TrainableParams), FlattenStruct(loaded.TrainableParams));
            Assert.Equal(FlattenStruct(ckpt.OptimizerState), FlattenStruct(loaded.OptimizerState));
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    /// <summary>
    /// Host-owned epoch/batch counters (issue #100) persist and restore in both checkpoint formats.
    /// A checkpoint saved at (step, epoch, batchIndex) reloads with all three through a fresh rig, the
    /// .skpt and legacy flat formats agree, and <see cref="Persistence.Inspect"/> surfaces epoch and
    /// batch from the manifest / marker alone (no tensor payload load — Observations stays empty).
    /// </summary>
    [Fact]
    public void TestCheckpointEpochBatchCountersRoundTripCoverage()
    {
        // Step comes from training; epoch and batch index are host-owned, set here by the "host".
        var (_, trained, _, _) = BuildTrainedAdamRig(steps: 4);
        var ckpt = new TrainingCheckpoint(
            trained.TrainableParams, trained.ModelState, trained.OptimizerState,
            step: trained.Step, epoch: 7, batchIndex: 340, rig: trained.Rig);
        Assert.Equal(4, ckpt.Step);
        Assert.Equal(7, ckpt.Epoch);
        Assert.Equal(340, ckpt.BatchIndex);

        var skptPath = Path.Combine(Path.GetTempPath(), $"shrk_ctr_skpt_{Guid.NewGuid():N}.skpt");
        var legacyPath = Path.Combine(Path.GetTempPath(), $"shrk_ctr_legacy_{Guid.NewGuid():N}.safetensors");
        try
        {
            // --- .skpt format: manifest records the counters; a fresh rig restores them. ---
            Persistence.SaveTrainingCheckpointToSkpt(ckpt, skptPath);
            var manifest = SkptFileFormat.ParseManifest(
                ReadEntryBytesViaBcl(skptPath, SkptFileFormat.ConfigEntryName), skptPath);
            Assert.Equal(7, manifest.Training!.Epoch);
            Assert.Equal(340, manifest.Training.BatchIndex);

            var skptLoaded = BuildTrainedAdamRig(steps: 0).Rig.LoadCheckpoint(skptPath);
            Assert.Equal(4, skptLoaded.Step);
            Assert.Equal(7, skptLoaded.Epoch);
            Assert.Equal(340, skptLoaded.BatchIndex);

            // --- Legacy flat safetensors format: same counters round-trip via the marker. ---
            ckpt.Save(legacyPath);
            var legacyLoaded = BuildTrainedAdamRig(steps: 0).Rig.LoadCheckpoint(legacyPath);
            Assert.Equal(4, legacyLoaded.Step);
            Assert.Equal(7, legacyLoaded.Epoch);
            Assert.Equal(340, legacyLoaded.BatchIndex);

            // --- The two formats agree. ---
            Assert.Equal(skptLoaded.Step, legacyLoaded.Step);
            Assert.Equal(skptLoaded.Epoch, legacyLoaded.Epoch);
            Assert.Equal(skptLoaded.BatchIndex, legacyLoaded.BatchIndex);

            // --- Inspect reports them without loading tensor data. ---
            var skptInspect = Persistence.Inspect(skptPath);
            Assert.Empty(skptInspect.Observations);
            Assert.Equal(7, skptInspect.Skpt!.Training!.Epoch);
            Assert.Equal(340, skptInspect.Skpt.Training.BatchIndex);
            var skptText = skptInspect.ToString();
            Assert.Contains("epoch 7", skptText);
            Assert.Contains("batch index 340", skptText);

            var legacyInspect = Persistence.Inspect(legacyPath);
            Assert.Empty(legacyInspect.Observations);
            Assert.Equal(3, legacyInspect.TrainingCheckpoint!.FormatVersion);
            Assert.Equal(7, legacyInspect.TrainingCheckpoint.Epoch);
            Assert.Equal(340, legacyInspect.TrainingCheckpoint.BatchIndex);
            var legacyText = legacyInspect.ToString();
            Assert.Contains("epoch: 7", legacyText);
            Assert.Contains("batch index: 340", legacyText);
        }
        finally
        {
            if (File.Exists(skptPath)) File.Delete(skptPath);
            if (File.Exists(legacyPath)) File.Delete(legacyPath);
        }
    }

    /// <summary>
    /// Epoch and batch index are host-owned run counters, so a single <see cref="TrainingRig.TrainStep(TrainingCheckpoint, TensorDataStruct, TensorDataStruct)"/>
    /// (a graph execution) advances only the step and carries epoch/batch through unchanged — the
    /// training loop, not the graph, moves them.
    /// </summary>
    [Fact]
    public void TestTrainStepCarriesEpochBatchCountersCoverage()
    {
        var (rig, trained, inBatch, outBatch) = BuildTrainedAdamRig(steps: 1);
        var ckpt = new TrainingCheckpoint(
            trained.TrainableParams, trained.ModelState, trained.OptimizerState,
            step: trained.Step, epoch: 3, batchIndex: 12);

        var next = rig.TrainStep(ckpt, inBatch, outBatch);
        Assert.Equal(ckpt.Step + 1, next.Step);   // step advances (one graph execution)
        Assert.Equal(3, next.Epoch);              // host-owned: unchanged by TrainStep
        Assert.Equal(12, next.BatchIndex);
    }

    /// <summary>
    /// A .skpt manifest whose training block lacks the add-only <c>epoch</c>/<c>batchIndex</c>
    /// keys (a #95-era manifest, written before those counters existed) loads with those
    /// counters read back as <c>null</c> (unknown) — the add-only-field leniency of the .skpt manifest
    /// combined with the nullable counters (issue #111): an absent key is genuinely unknown, never a
    /// sentinel 0.
    /// </summary>
    [Fact]
    public void TestSkptWithoutEpochBatchKeysLoadsDefaultsCoverage()
    {
        var (_, trained, _, _) = BuildTrainedAdamRig(steps: 2);
        var ckpt = new TrainingCheckpoint(
            trained.TrainableParams, trained.ModelState, trained.OptimizerState, step: 2, rig: trained.Rig);

        var skptPath = Path.Combine(Path.GetTempPath(), $"shrk_old_{Guid.NewGuid():N}.skpt");
        try
        {
            // Strip the epoch/batchIndex keys to mimic a #95-era manifest.
            Persistence.SaveTrainingCheckpointToSkpt(ckpt, skptPath);
            StripSkptTrainingCounterKeys(skptPath);

            var manifest = SkptFileFormat.ParseManifest(
                ReadEntryBytesViaBcl(skptPath, SkptFileFormat.ConfigEntryName), skptPath);
            Assert.Null(manifest.Training!.Epoch);       // absent keys ⇒ null (unknown)
            Assert.Null(manifest.Training.BatchIndex);

            var skptLoaded = BuildTrainedAdamRig(steps: 0).Rig.LoadCheckpoint(skptPath);
            Assert.Equal(2, skptLoaded.Step);
            Assert.Null(skptLoaded.Epoch);
            Assert.Null(skptLoaded.BatchIndex);

            var skptInspect = Persistence.Inspect(skptPath);
            Assert.Empty(skptInspect.Observations);
            Assert.Null(skptInspect.Skpt!.Training!.Epoch);
            Assert.Null(skptInspect.Skpt.Training.BatchIndex);
        }
        finally
        {
            if (File.Exists(skptPath)) File.Delete(skptPath);
        }
    }

    /// <summary>
    /// Issue #111: a checkpoint with an unknown position — no data loader, no explicit counter — has
    /// <see cref="TrainingCheckpoint.Epoch"/> / <see cref="TrainingCheckpoint.BatchIndex"/> = <c>null</c>,
    /// and that null survives a save/load round-trip as null (never a sentinel 0) through <b>both</b> the
    /// flat safetensors marker (its presence-gated epoch/batch scalars are simply absent) and the .skpt
    /// manifest (the keys are omitted). <see cref="Persistence.Inspect"/> reports the null counters too.
    /// </summary>
    [Fact]
    public void TestNullEpochBatchCountersRoundTripAsNullCoverage()
    {
        // Step comes from training (always concrete); epoch/batch are left unset ⇒ null (unknown).
        var (_, trained, _, _) = BuildTrainedAdamRig(steps: 3);
        var ckpt = new TrainingCheckpoint(
            trained.TrainableParams, trained.ModelState, trained.OptimizerState,
            step: trained.Step, rig: trained.Rig);
        Assert.Null(ckpt.Epoch);
        Assert.Null(ckpt.BatchIndex);

        var legacyPath = Path.Combine(Path.GetTempPath(), $"shrk_nullctr_{Guid.NewGuid():N}.safetensors");
        var skptPath = Path.Combine(Path.GetTempPath(), $"shrk_nullctr_{Guid.NewGuid():N}.skpt");
        try
        {
            // Flat safetensors: the presence-gated epoch/batch scalars are absent, so they reload null.
            ckpt.Save(legacyPath);
            var legacy = BuildTrainedAdamRig(steps: 0).Rig.LoadCheckpoint(legacyPath);
            Assert.Equal(trained.Step, legacy.Step);   // step stays concrete
            Assert.Null(legacy.Epoch);
            Assert.Null(legacy.BatchIndex);

            var flatInspect = Persistence.Inspect(legacyPath);
            Assert.Empty(flatInspect.Observations);
            Assert.Equal(3, flatInspect.TrainingCheckpoint!.FormatVersion);
            Assert.Null(flatInspect.TrainingCheckpoint.Epoch);
            Assert.Null(flatInspect.TrainingCheckpoint.BatchIndex);
            Assert.Contains("epoch: unset", flatInspect.ToString());

            // Native .skpt: the manifest omits epoch/batch, so they reload null too.
            Persistence.SaveTrainingCheckpointToSkpt(ckpt, skptPath);
            var manifest = SkptFileFormat.ParseManifest(
                ReadEntryBytesViaBcl(skptPath, SkptFileFormat.ConfigEntryName), skptPath);
            Assert.Null(manifest.Training!.Epoch);
            Assert.Null(manifest.Training.BatchIndex);

            var skptLoaded = BuildTrainedAdamRig(steps: 0).Rig.LoadCheckpoint(skptPath);
            Assert.Equal(trained.Step, skptLoaded.Step);
            Assert.Null(skptLoaded.Epoch);
            Assert.Null(skptLoaded.BatchIndex);
        }
        finally
        {
            if (File.Exists(legacyPath)) File.Delete(legacyPath);
            if (File.Exists(skptPath)) File.Delete(skptPath);
        }
    }

    /// <summary>
    /// Issue #111: a loader-driven run records <b>concrete</b> epoch/batch counters on its checkpoint
    /// (the batch used, sourced from the loader), and those survive save/load as concrete values —
    /// including an epoch of exactly <c>0</c>, which persists as a present scalar valued 0 (NOT omitted,
    /// so it reloads as 0, never null). This is the concrete-vs-unknown distinction the presence-gating
    /// draws.
    /// </summary>
    [Fact]
    public void TestLoaderDrivenRunPersistsConcreteCountersCoverage()
    {
        const int features = 4;
        var rig = LoaderRig(batchSize: 2, features);
        var (inputs, targets) = IndexDataset(rig, n: 6, features);   // 3 batches/epoch
        var loader = new InMemoryDataLoader(inputs, targets, batchSize: 2);

        var final = rig.Fit(loader, numEpochs: 1).FinalCheckpoint;
        Assert.Equal(0, final.Epoch);       // concrete 0 (batch used was in epoch 0), not null
        Assert.Equal(2, final.BatchIndex);  // the last batch used (epoch 0's last batch)

        var path = Path.Combine(Path.GetTempPath(), $"shrk_concctr_{Guid.NewGuid():N}.safetensors");
        try
        {
            final.Save(path);
            var loaded = LoaderRig(batchSize: 2, features).LoadCheckpoint(path);
            Assert.Equal(0, loaded.Epoch);       // a concrete 0 reloads as 0, not null
            Assert.Equal(2, loaded.BatchIndex);  // concrete value round-trips

            var inspect = Persistence.Inspect(path);
            Assert.Empty(inspect.Observations);
            Assert.Equal(0, inspect.TrainingCheckpoint!.Epoch);
            Assert.Equal(2, inspect.TrainingCheckpoint.BatchIndex);
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Two-layer rig (issue #110): swappable constituents + a derived in-memory
    // trainstep, as an immutable value with a With… derivation surface, plus
    // extraction as a pure read off the model constituent.
    // ──────────────────────────────────────────────────────────────────────

    /// <summary>
    /// The rig exposes its constituent layer, and every <c>With…</c> derivation returns a NEW rig
    /// that shares the unchanged constituents by reference and re-derives only its own trainstep —
    /// never mutating the receiver (a rig is an immutable value, §5.8.5). WithOptimizer re-initializes
    /// optimizer state (plain SGD → SGD-momentum grows a state field per param); WithScheduler keeps
    /// the optimizer graph and swaps the schedule-carrying hyperparameters; WithSeed re-seeds. Each
    /// derived rig is a working rig that trains a step to a finite loss on its own trainstep.
    /// </summary>
    [Fact]
    public void TestRigDerivationsShareConstituentsAndAreImmutableCoverage()
    {
        var (sample, input, target) = ScalarMultiplyBatches();
        var model = ScalarMultiplyModel.ComputationGraph;
        var loss = L2Loss.ComputationGraph;
        var opt = SGDOptimizer.ComputationGraph;

        var rig = TrainingRig.FromScratch(model, loss, opt, sample, 0.1f);

        // The constituent layer is exactly what was authored (shared by reference).
        Assert.Same(model, rig.ModelConstituent);
        Assert.Same(loss, rig.LossConstituent);
        Assert.Same(opt, rig.OptimizerConstituent);
        Assert.Equal(0UL, rig.RngConfig.MasterSeed);
        Assert.Empty(rig.OptimizerStateDef.Fields);   // plain SGD carries no optimizer state

        // WithLoss: loss swapped; model/optimizer shared; the original rig is untouched.
        var newLoss = Losses.L1Loss;
        var lossRig = rig.WithLoss(newLoss);
        Assert.NotSame(rig, lossRig);
        Assert.Same(newLoss, lossRig.LossConstituent);
        Assert.Same(model, lossRig.ModelConstituent);      // shared by reference
        Assert.Same(opt, lossRig.OptimizerConstituent);    // shared by reference
        Assert.Same(loss, rig.LossConstituent);            // receiver unchanged (immutable)

        // WithOptimizer: optimizer + hyperparameters swapped; optimizer state re-initialized.
        var momRig = rig.WithOptimizer(SGDMomentumOptimizer.ComputationGraph,
            new SGDMomentumOptimizerHyperparameters { LearningRate = 0.5f, MomentumCoeff = 0.9f });
        Assert.Same(model, momRig.ModelConstituent);
        Assert.Same(loss, momRig.LossConstituent);
        Assert.NotSame(opt, momRig.OptimizerConstituent);
        Assert.NotEmpty(momRig.OptimizerStateDef.Fields);  // opt-state re-derived on the swap
        Assert.Empty(rig.OptimizerStateDef.Fields);        // receiver unchanged

        // WithScheduler: same optimizer graph, schedule-carrying hyperparameters swapped.
        var schedRig = rig.WithScheduler(
            new SGDOptimizerHyperparameters { LearningRate = Schedules.Linear(0.2f, 0f, 4) });
        Assert.Same(opt, schedRig.OptimizerConstituent);
        Assert.Empty(schedRig.HyperparameterStructDef.Fields);   // schedule is in-graph, not a runtime field

        // WithSeed: re-seeded; all constituents shared; the original keeps the default seed.
        var reseeded = rig.WithSeed(new RngConfig { MasterSeed = 42 });
        Assert.Same(model, reseeded.ModelConstituent);
        Assert.Same(opt, reseeded.OptimizerConstituent);
        Assert.Equal(42UL, reseeded.RngConfig.MasterSeed);
        Assert.Equal(0UL, rig.RngConfig.MasterSeed);

        // Every derived rig trains a step to a finite loss on its own re-derived trainstep.
        foreach (var derived in new[] { lossRig, momRig, schedRig, reseeded })
        {
            var stepped = derived.TrainStep(derived.CreateInitialCheckpoint(), input, target);
            Assert.True(float.IsFinite(stepped.Loss!.Value));
        }
    }

    /// <summary>
    /// The extraction contract (§5.8.2), pinned end to end: build a rig, train a real step, then
    /// <see cref="TrainingRig.ExtractInferenceModel"/> — a pure read off the model constituent's
    /// mapping, with no re-supplied model graph. The extracted model's params ARE the trainstep's
    /// updated tensors, matched by identifier: every updated trainable-param name resolves to a
    /// ModelId in the extracted model (identity preserved across composition), and the extracted
    /// model's forward (ScalarMultiply: <c>input · weight</c>) equals <c>probe · wUpdated</c> — i.e.
    /// it computes with exactly the tensor the trainstep produced for that identifier, not the init.
    /// </summary>
    [Fact]
    public void TestExtractInferenceModelIdentityCoverage()
    {
        var (sample, input, target) = ScalarMultiplyBatches();
        var rig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            sample, 0.1f);

        var stepped = rig.TrainStep(rig.CreateInitialCheckpoint(), input, target);
        string wName = rig.TrainableParamStructDef.Fields[0].Name;
        float wUpdated = ((TensorData<float32>)stepped.TrainableParams.Fields[wName]).AccessMemory()[0];
        Assert.NotEqual(1.0f, wUpdated);   // the step genuinely moved the weight off its init

        // Extraction: pure read off the model constituent — no model graph re-supplied by the caller.
        var inference = rig.ExtractInferenceModel(stepped);
        Assert.Equal(GraphKind.ConcreteModel, inference.Kind);

        // Identity preserved: each updated-tensor identifier resolves to a ModelId in the extracted model.
        var concrete = rig.ModelConstituent.ToConcreteArchitecture(
            rig.ModelConstituent.FromOrderedInputs([sample[0].ToTensorData()]));
        var scheme = ModuleParamSetNamingScheme.FromModelIdFormats(concrete.GetShorokooIdNamingScheme(), "Shorokoo");
        var modelIds = concrete.GetConcreteModelParamInfos().ModelIds;
        foreach (var f in stepped.TrainableParams.Fields.Where(f => f.Value is TensorData))
            Assert.True(scheme.ToModelId(f.Key, modelIds) is not null,
                $"extracted param '{f.Key}' did not resolve to a ModelId (composition identity regressed)");

        // The extracted model computes with exactly the trainstep's updated tensor for wName.
        var probe = TensorData([4L], new float[] { 2f, 3f, 4f, 5f });
        var outputs = ComputeContext.Default.Execute(inference, probe)[0]
            .ToTensorData<float32>().AccessMemory().ToArray();
        float[] expected = [2f * wUpdated, 3f * wUpdated, 4f * wUpdated, 5f * wUpdated];
        for (int i = 0; i < expected.Length; i++)
            Assert.True(MathF.Abs(expected[i] - outputs[i]) < 1e-5f,
                $"extracted forward [{i}] = {outputs[i]}, expected probe·wUpdated = {expected[i]}");
    }

    /// <summary>
    /// Counter derivations (§5.8.5): step/epoch/batch are host-owned scalars, so resetting one yields
    /// a NEW <see cref="TrainingCheckpoint"/> carrying the same tensor state by reference (nothing is
    /// re-derived), leaving the receiver untouched. The single-counter helpers carry the others through.
    /// </summary>
    [Fact]
    public void TestCheckpointCounterDerivationsCoverage()
    {
        var (sample, _, _) = ScalarMultiplyBatches();
        var rig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            sample, 0.1f);
        var ck = rig.CreateInitialCheckpoint();   // step = 0; epoch/batch = null (no loader / explicit counter)
        Assert.Equal(0, ck.Step);

        var moved = ck.WithCounters(step: 5, epoch: 2, batchIndex: 3);
        Assert.NotSame(ck, moved);
        Assert.Equal(5, moved.Step);
        Assert.Equal(2, moved.Epoch);
        Assert.Equal(3, moved.BatchIndex);
        // Tensor state shared by reference — counters are not rig state.
        Assert.Same(ck.TrainableParams, moved.TrainableParams);
        Assert.Same(ck.ModelState, moved.ModelState);
        Assert.Same(ck.OptimizerState, moved.OptimizerState);
        // Receiver untouched (immutable value); its epoch/batch stay unknown (null).
        Assert.Equal(0, ck.Step);
        Assert.Null(ck.Epoch);
        Assert.Null(ck.BatchIndex);

        // Single-counter helpers carry the others through unchanged (a null epoch stays null).
        Assert.Equal(9, ck.WithStep(9).Step);
        Assert.Null(ck.WithStep(9).Epoch);
        Assert.Equal(7, ck.WithEpoch(7).Epoch);
        Assert.Equal(0, ck.WithEpoch(7).Step);
        Assert.Equal(4, ck.WithBatchIndex(4).BatchIndex);
        Assert.Equal(0, ck.WithBatchIndex(4).Step);
    }

    // ──────────────────────────────────────────────────────────────────────
    // Checkpoint/rig API redesign (#110 follow-up, towards #115): a checkpoint
    // carries its rig; parameterless ToInferenceModel(); AdoptCheckpoint; the
    // step's loss on the checkpoint; component-subset save/load; the flags API.
    // ──────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Every rig-produced checkpoint carries its <see cref="TrainingCheckpoint.Rig"/> — the initial
    /// checkpoint, each <see cref="TrainingRig.TrainStep(TrainingCheckpoint, TensorDataStruct, TensorDataStruct)"/>
    /// result, and the counter derivations (which also preserve <see cref="TrainingCheckpoint.Loss"/>).
    /// A bare checkpoint built by the public constructor has none.
    /// </summary>
    [Fact]
    public void TestCheckpointCarriesRigCoverage()
    {
        var (sample, input, target) = ScalarMultiplyBatches();
        var rig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            sample, 0.1f);

        var initial = rig.CreateInitialCheckpoint();
        Assert.Same(rig, initial.Rig);
        Assert.Null(initial.Loss);   // no step produced it

        var stepped = rig.TrainStep(initial, input, target);
        Assert.Same(rig, stepped.Rig);

        // Counter derivations preserve both Rig and Loss.
        var moved = stepped.WithStep(42);
        Assert.Same(rig, moved.Rig);
        Assert.Equal(stepped.Loss, moved.Loss);

        // A bare checkpoint carries no rig.
        var bare = new TrainingCheckpoint(initial.TrainableParams, initial.ModelState, initial.OptimizerState);
        Assert.Null(bare.Rig);
    }

    /// <summary>
    /// <see cref="TrainingCheckpoint.Loss"/> is <c>null</c> on an initial checkpoint and set by
    /// <see cref="TrainingRig.TrainStep(TrainingCheckpoint, TensorDataStruct, TensorDataStruct)"/>
    /// (which now returns the post-step checkpoint directly) to that step's loss, carried through
    /// the counter derivations.
    /// </summary>
    [Fact]
    public void TestCheckpointLossSetByStepCoverage()
    {
        var (sample, input, target) = ScalarMultiplyBatches();
        var rig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            sample, 0.1f);

        var initial = rig.CreateInitialCheckpoint();
        Assert.Null(initial.Loss);

        var result = rig.TrainStep(initial, input, target);
        Assert.NotNull(result.Loss);
        Assert.True(float.IsFinite(result.Loss!.Value));
        Assert.Equal(result.Loss, result.WithEpoch(3).Loss);   // carried through the counter derivation
    }

    /// <summary>
    /// The parameterless <see cref="TrainingCheckpoint.ToInferenceModel()"/> works for a
    /// <b>multi-input</b> model: the rig's retained concrete arch was concretized at ALL its inputs (not
    /// just the first), so a two-input model round-trips. y = a·wa + b·wb with wa = wb = 1 ⇒ output = a + b.
    /// This is the path the old single-sample-input extraction could not build.
    /// </summary>
    [Fact]
    public void TestToInferenceModelMultiInputCoverage()
    {
        var sample = new NamedModelParam[]
        {
            new TensorDataModelParam("a", ModelParamType.InputParam, TensorData([4L], new float[] { 1f, 2f, 3f, 4f })),
            new TensorDataModelParam("b", ModelParamType.InputParam, TensorData([4L], new float[] { 5f, 6f, 7f, 8f })),
        };
        var rig = TrainingRig.FromScratch(
            TwoInputSumModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            sample, 0.1f);
        Assert.Equal(2, rig.InputDef.Fields.Length);          // two model inputs

        var ckpt = rig.CreateInitialCheckpoint();             // wa = wb = 1
        var inference = ckpt.ToInferenceModel();              // binds weights into the retained arch (concretized at BOTH [4] inputs)
        Assert.Equal(GraphKind.ConcreteModel, inference.Kind);

        var a = TensorData([4L], new float[] { 1f, 2f, 3f, 4f });
        var b = TensorData([4L], new float[] { 10f, 20f, 30f, 40f });
        var outputs = ComputeContext.Default.Execute(inference, a, b)[0]
            .ToTensorData<float32>().AccessMemory().ToArray();
        float[] expected = [11f, 22f, 33f, 44f];              // a + b
        for (int i = 0; i < expected.Length; i++)
            Assert.True(MathF.Abs(expected[i] - outputs[i]) < 1e-5f,
                $"multi-input extract [{i}] = {outputs[i]}, expected a+b = {expected[i]}");
    }

    /// <summary>
    /// <see cref="TrainingRig.AdoptCheckpoint"/> attaches a rig to a bare checkpoint (so
    /// <see cref="TrainingCheckpoint.ToInferenceModel()"/> works), preserving counters and loss and
    /// leaving the argument untouched; a checkpoint whose field defs are incompatible with the rig is
    /// rejected. A rig-less checkpoint's <see cref="TrainingCheckpoint.ToInferenceModel()"/> throws.
    /// </summary>
    [Fact]
    public void TestAdoptCheckpointCoverage()
    {
        var (sample, _, _) = ScalarMultiplyBatches();
        var rig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            sample, 0.1f);
        var seed = rig.CreateInitialCheckpoint();

        // A bare checkpoint (no rig) at some counters; ToInferenceModel throws.
        var bare = new TrainingCheckpoint(
            seed.TrainableParams, seed.ModelState, seed.OptimizerState, step: 5, epoch: 2, batchIndex: 1);
        Assert.Null(bare.Rig);
        Assert.Throws<InvalidOperationException>(() => bare.ToInferenceModel());

        // Adopt: new checkpoint with the rig attached; counters preserved; argument untouched.
        var adopted = rig.AdoptCheckpoint(bare);
        Assert.NotSame(bare, adopted);
        Assert.Same(rig, adopted.Rig);
        Assert.Equal(5, adopted.Step);
        Assert.Equal(2, adopted.Epoch);
        Assert.Equal(1, adopted.BatchIndex);
        Assert.Null(bare.Rig);                                // argument untouched
        Assert.NotNull(adopted.ToInferenceModel());           // now works

        // Incompatible defs are rejected: a BatchNorm (+ momentum) rig cannot adopt a
        // ScalarMultiply/SGD checkpoint (different trainable/state/opt defs).
        var bnRig = TrainingRig.FromScratch(
            ScalarMultiplyWithBatchNormModel.ComputationGraph, L2Loss.ComputationGraph,
            SGDMomentumOptimizer.ComputationGraph,
            [new TensorDataModelParam("input", ModelParamType.InputParam, TensorData([8L], new float[8]))],
            0.5f, 0.9f);
        Assert.Throws<ArgumentException>(() => bnRig.AdoptCheckpoint(bare));
    }

    /// <summary>
    /// The component flags on save/load: saving <see cref="CheckpointComponents.InferenceState"/> only
    /// writes the trainable params (and model state), dropping optimizer state and counters. Reloading
    /// against a fresh rig restores the trainable params, fills the dropped optimizer state from the
    /// rig's initial values (so it is the init, not the trained state), and reads counters as 0. The
    /// reloaded checkpoint carries the rig. Requesting the (unimplemented) TrainingRig component throws
    /// a clear <see cref="NotSupportedException"/> naming #115.
    /// </summary>
    [Fact]
    public void TestSaveLoadComponentsSubsetCoverage()
    {
        var (rigA, trained, _, _) = BuildTrainedAdamRig(steps: 3);
        Assert.NotEmpty(trained.OptimizerState.Fields);       // AdamW carries m/v/step
        var trainedOpt = FlattenStruct(trained.OptimizerState);
        var initialOpt = FlattenStruct(rigA.CreateInitialCheckpoint().OptimizerState);
        Assert.NotEqual(trainedOpt, initialOpt);              // training moved the optimizer state

        var path = Path.Combine(Path.GetTempPath(), $"shrk_subset_{Guid.NewGuid():N}.safetensors");
        try
        {
            // Save inference state only — no optimizer state, no counters.
            trained.Save(path, CheckpointComponents.InferenceState);

            var rigB = BuildTrainedAdamRig(steps: 0).Rig;
            var loaded = rigB.LoadCheckpoint(path);
            Assert.Same(rigB, loaded.Rig);                    // load attaches the rig
            Assert.Equal(FlattenStruct(trained.TrainableParams), FlattenStruct(loaded.TrainableParams));
            Assert.Equal(0, loaded.Step);                     // counters were dropped ⇒ 0
            // Optimizer state was not saved ⇒ filled from the rig's initial values (not the trained ones).
            Assert.Equal(initialOpt, FlattenStruct(loaded.OptimizerState));

            // Requesting the TrainingRig component is a clear NotSupportedException (#115).
            var ex = Assert.Throws<NotSupportedException>(
                () => trained.Save(path, CheckpointComponents.All));
            Assert.Contains("#115", ex.Message);

            // Symmetric on load: explicitly asking for the TrainingRig component (directly or via All)
            // throws the same #115 NotSupportedException — the file never stores the rig's constituents.
            var loadRigEx = Assert.Throws<NotSupportedException>(
                () => rigB.LoadCheckpoint(path, CheckpointComponents.TrainingRig));
            Assert.Contains("#115", loadRigEx.Message);
            Assert.Throws<NotSupportedException>(() => rigB.LoadCheckpoint(path, CheckpointComponents.All));
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    /// <summary>
    /// The component filter is honored on a partial load from a native <c>.skpt</c> exactly as it is
    /// for the flat path (Fix #3): a full <c>.skpt</c> is saved, then reloaded with only
    /// <see cref="CheckpointComponents.InferenceState"/> — the trainable params restore from the file,
    /// but the dropped optimizer state fills from the rig's initial values (not the trained ones), the
    /// counters read as 0, and the loss reads back <c>null</c> (its own Loss component was dropped too).
    /// Requesting the (unimplemented) TrainingRig component throws a #115 <see cref="NotSupportedException"/>.
    /// </summary>
    [Fact]
    public void TestSkptSaveLoadComponentsSubsetCoverage()
    {
        var (rigA, trained, _, _) = BuildTrainedAdamRig(steps: 3);
        Assert.NotEmpty(trained.OptimizerState.Fields);       // AdamW carries m/v/step
        Assert.NotNull(trained.Loss);                         // a step produced the loss
        var trainedOpt = FlattenStruct(trained.OptimizerState);
        var initialOpt = FlattenStruct(rigA.CreateInitialCheckpoint().OptimizerState);
        Assert.NotEqual(trainedOpt, initialOpt);              // training moved the optimizer state

        var path = Path.Combine(Path.GetTempPath(), $"shrk_skpt_subset_{Guid.NewGuid():N}.skpt");
        try
        {
            // The .skpt save writes every kind; the FILTER is exercised on load.
            Persistence.SaveTrainingCheckpointToSkpt(trained, path);

            var rigB = BuildTrainedAdamRig(steps: 0).Rig;
            var loaded = rigB.LoadCheckpoint(path, CheckpointComponents.InferenceState);
            Assert.Same(rigB, loaded.Rig);                    // load attaches the rig
            Assert.Equal(FlattenStruct(trained.TrainableParams), FlattenStruct(loaded.TrainableParams));
            Assert.Equal(0, loaded.Step);                     // counters dropped ⇒ 0
            Assert.Null(loaded.Loss);                         // Loss component dropped ⇒ null
            // Optimizer state was filtered out ⇒ filled from the rig's initial values, not the trained ones.
            Assert.Equal(initialOpt, FlattenStruct(loaded.OptimizerState));

            // Symmetric with the flat path: requesting the TrainingRig component throws a #115 error.
            var ex = Assert.Throws<NotSupportedException>(
                () => rigB.LoadCheckpoint(path, CheckpointComponents.TrainingRig));
            Assert.Contains("#115", ex.Message);
            Assert.Throws<NotSupportedException>(() => rigB.LoadCheckpoint(path, CheckpointComponents.All));
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    /// <summary>
    /// <see cref="TrainingCheckpoint.Loss"/> persists as its <b>own</b>
    /// <see cref="CheckpointComponents.Loss"/> component — independent of
    /// <see cref="CheckpointComponents.Counters"/> — in BOTH the flat safetensors format and the
    /// <c>.skpt</c> manifest: a checkpoint with a known non-null loss round-trips its loss; an initial
    /// checkpoint (null loss) reads back <c>null</c> (never a sentinel 0.0). The two axes are exercised
    /// independently: a <c>Counters</c>-only save carries the counters but NOT the loss, and a
    /// <c>Loss</c>-only save carries the loss but drops the counters. Explicitly requesting <c>Loss</c>
    /// on a null-loss checkpoint is a no-op (writes nothing, no throw).
    /// </summary>
    [Fact]
    public void TestCheckpointLossPersistsCoverage()
    {
        var (rigA, trained, _, _) = BuildTrainedAdamRig(steps: 3);
        Assert.NotNull(trained.Loss);
        float loss = trained.Loss!.Value;
        Assert.True(trained.Step > 0);
        var initial = rigA.CreateInitialCheckpoint();
        Assert.Null(initial.Loss);

        var flatPath = Path.Combine(Path.GetTempPath(), $"shrk_loss_flat_{Guid.NewGuid():N}.safetensors");
        var flatInitPath = Path.Combine(Path.GetTempPath(), $"shrk_loss_flatinit_{Guid.NewGuid():N}.safetensors");
        var flatCountersOnlyPath = Path.Combine(Path.GetTempPath(), $"shrk_loss_flatco_{Guid.NewGuid():N}.safetensors");
        var flatLossOnlyPath = Path.Combine(Path.GetTempPath(), $"shrk_loss_flatlo_{Guid.NewGuid():N}.safetensors");
        var flatNullLossReqPath = Path.Combine(Path.GetTempPath(), $"shrk_loss_flatnull_{Guid.NewGuid():N}.safetensors");
        var skptPath = Path.Combine(Path.GetTempPath(), $"shrk_loss_skpt_{Guid.NewGuid():N}.skpt");
        var skptInitPath = Path.Combine(Path.GetTempPath(), $"shrk_loss_skptinit_{Guid.NewGuid():N}.skpt");
        try
        {
            // ---- Flat format: non-null loss round-trips; null loss reads back null. ----
            trained.Save(flatPath);
            var full = BuildTrainedAdamRig(steps: 0).Rig.LoadCheckpoint(flatPath);
            Assert.Equal(loss, full.Loss!.Value);
            Assert.Equal(trained.Step, full.Step);            // counters ride too, independently

            initial.Save(flatInitPath);
            Assert.Null(BuildTrainedAdamRig(steps: 0).Rig.LoadCheckpoint(flatInitPath).Loss);

            // Axis 1 — Counters WITHOUT Loss: the counters persist but the loss is dropped (reads null).
            trained.Save(flatCountersOnlyPath,
                CheckpointComponents.InferenceState | CheckpointComponents.Counters);
            var countersOnly = BuildTrainedAdamRig(steps: 0).Rig.LoadCheckpoint(flatCountersOnlyPath);
            Assert.Equal(trained.Step, countersOnly.Step);    // counters carried
            Assert.Null(countersOnly.Loss);                   // loss NOT carried (its own component was dropped)

            // Axis 2 — Loss WITHOUT Counters: the loss persists but the counters drop to 0.
            trained.Save(flatLossOnlyPath,
                CheckpointComponents.InferenceState | CheckpointComponents.Loss);
            var lossOnly = BuildTrainedAdamRig(steps: 0).Rig.LoadCheckpoint(flatLossOnlyPath);
            Assert.Equal(loss, lossOnly.Loss!.Value);         // loss carried
            Assert.Equal(0, lossOnly.Step);                   // counters dropped ⇒ 0

            // Requesting Loss on a null-loss checkpoint is a no-op (writes nothing, no throw).
            initial.Save(flatNullLossReqPath, CheckpointComponents.InferenceState | CheckpointComponents.Loss);
            Assert.Null(BuildTrainedAdamRig(steps: 0).Rig.LoadCheckpoint(flatNullLossReqPath).Loss);

            // ---- .skpt manifest: non-null loss round-trips; null loss reads back null. ----
            Persistence.SaveTrainingCheckpointToSkpt(trained, skptPath);
            Assert.Equal(loss, BuildTrainedAdamRig(steps: 0).Rig.LoadCheckpoint(skptPath).Loss!.Value);
            // .skpt load honors the Loss component independently: drop Loss ⇒ null even though counters load.
            var skptNoLoss = BuildTrainedAdamRig(steps: 0).Rig.LoadCheckpoint(
                skptPath, CheckpointComponents.InferenceState | CheckpointComponents.OptimizerState | CheckpointComponents.Counters);
            Assert.Equal(trained.Step, skptNoLoss.Step);      // counters carried
            Assert.Null(skptNoLoss.Loss);                     // Loss component dropped ⇒ null

            Persistence.SaveTrainingCheckpointToSkpt(initial, skptInitPath);
            Assert.Null(BuildTrainedAdamRig(steps: 0).Rig.LoadCheckpoint(skptInitPath).Loss);
        }
        finally
        {
            string[] paths =
                [flatPath, flatInitPath, flatCountersOnlyPath, flatLossOnlyPath, flatNullLossReqPath, skptPath, skptInitPath];
            foreach (var p in paths)
                if (File.Exists(p)) File.Delete(p);
        }
    }

    /// <summary>
    /// Rename smoke: the initial-checkpoint factory is <c>CreateInitialCheckpoint</c> (was
    /// <c>CreateDefaultCheckpoint</c>), in both the no-arg and hyperparameter-struct overloads.
    /// </summary>
    [Fact]
    public void TestCreateInitialCheckpointRenameSmokeCoverage()
    {
        var (sample, _, _) = ScalarMultiplyBatches();
        var rig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            sample, 0.1f);
        Assert.NotNull(rig.CreateInitialCheckpoint().TrainableParams);

        var runtimeRig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, InitFromHyperOptimizer.ComputationGraph,
            sample, new InitFromHyperOptimizerHyperparameters { LearningRate = Hyperparameter.Runtime() });
        Assert.NotNull(runtimeRig.CreateInitialCheckpoint(runtimeRig.MakeHyperparameters(0.3f)).OptimizerState);
    }

    /// <summary>Rewrites a .skpt in place, deleting the <c>epoch</c>/<c>batchIndex</c> keys from its
    /// manifest training block to mimic a checkpoint written before those add-only fields existed.
    /// Every other entry (and each data entry's bytes, hence its recorded sha256) is preserved.</summary>
    private static void StripSkptTrainingCounterKeys(string path)
    {
        var entries = new List<SkptFileFormat.ZipEntrySpec>();
        using (var zip = System.IO.Compression.ZipFile.OpenRead(path))
            foreach (var e in zip.Entries)
            {
                using var s = e.Open();
                using var buf = new MemoryStream();
                s.CopyTo(buf);
                var data = buf.ToArray();
                if (e.FullName == SkptFileFormat.ConfigEntryName)
                {
                    var node = System.Text.Json.Nodes.JsonNode.Parse(data)!;
                    var training = node["training"]!.AsObject();
                    training.Remove("epoch");
                    training.Remove("batchIndex");
                    data = System.Text.Encoding.UTF8.GetBytes(node.ToJsonString());
                }
                entries.Add(new SkptFileFormat.ZipEntrySpec(e.FullName, data, Align: false));
            }
        using var outStream = File.Create(path);
        SkptFileFormat.WriteStoredZip(outStream, entries, DateTime.UtcNow);
    }

}
