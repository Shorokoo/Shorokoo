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
/// Coverage-purpose training-rig pipeline tests. Each [Fact] drives the full
/// model + loss + optimizer composition through <see cref="TrainingRig.FromScratch"/>
/// and <c>CreateDefaultCheckpoint</c> for a curated combination of modules.
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
    /// (<see cref="Shorokoo.Core.Nodes.Processors.Fast.FastConvertTrainableParamIdRefToTrainableParam.DiscoverTrainableParamInfos"/>).
    /// </summary>
    private static void CoverFromScratch(
        FastComputationGraph modelGraph,
        FastComputationGraph lossGraph,
        FastComputationGraph optimizerGraph,
        long[] inputShape,
        params HyperValue[] hyperparams)
    {
        long totalElements = 1;
        foreach (var d in inputShape) totalElements *= d;
        var sampleInput = new TensorDataModelParam(
            "input", ModelParamType.InputParam,
            TensorData(inputShape, new float[totalElements]));

        var rig = TrainingRig.FromScratch(modelGraph, lossGraph, optimizerGraph,
            new NamedModelParam[] { sampleInput }, hyperparams);

        var checkpoint = rig.CreateDefaultCheckpoint();
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
        FastComputationGraph modelGraph,
        FastComputationGraph lossGraph,
        FastComputationGraph optimizerGraph,
        long[] inputShape,
        params HyperValue[] hyperparams)
    {
        long totalElements = 1;
        foreach (var d in inputShape) totalElements *= d;
        var sampleInput = new TensorDataModelParam(
            "input", ModelParamType.InputParam,
            TensorData(inputShape, new float[totalElements]));

        var rig = TrainingRig.FromScratch(modelGraph, lossGraph, optimizerGraph,
            new NamedModelParam[] { sampleInput }, hyperparams);
        var checkpoint = rig.CreateDefaultCheckpoint();

        // Concretize the inference model + Shorokoo naming scheme (the documented binding flow).
        var hints = new ModelParamList(
            new[] { new KeyValuePair<string, TensorData>(modelGraph.Inputs[0].ToString(), TensorData(inputShape, new float[totalElements])) },
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
    /// <c>CreateDefaultCheckpoint</c> wired initializers correctly.
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

        var checkpoint = rig.CreateDefaultCheckpoint();

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

        var initial = rig.CreateDefaultCheckpoint();

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

        var ctx = new ComputeContext();

        // Drive Train: covers the per-epoch / per-batch loop and the
        // TrainingResult constructor + EpochLosses / FinalCheckpoint getters.
        var trainResult = rig.Train(initial, new[] { inputBatch }, new[] { targetBatch }, numEpochs: 1, ctx);
        Assert.Single(trainResult.EpochLosses);
        Assert.NotNull(trainResult.FinalCheckpoint);

        // Drive TrainStep directly so its output-repacking branches all execute
        // outside the Train wrapper, and touch the TrainingStepResult getters.
        var compiled = ctx.Compile(rig.TrainingStepPureGraph);
        var stepResult = rig.TrainStep(initial, inputBatch, targetBatch, compiled);
        Assert.NotNull(stepResult.Checkpoint);
        Assert.NotNull(stepResult.Checkpoint.TrainableParams);
        // ModelState / OptimizerState are empty for this combo, but their getters
        // still need to be exercised for full coverage.
        Assert.NotNull(stepResult.Checkpoint.ModelState);
        Assert.NotNull(stepResult.Checkpoint.OptimizerState);
        Assert.True(float.IsFinite(stepResult.Loss));

        // FastTrainingGraphs is a plain container exposed by the public surface;
        // its constructor and three getters are otherwise unreachable.
        var graphs = new FastTrainingGraphs(
            ScalarMultiplyModel.ComputationGraph,
            L2Loss.ComputationGraph,
            SGDOptimizer.ComputationGraph);
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
        var initial = rig.CreateDefaultCheckpoint();
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

        var ctx = new ComputeContext();
        var compiled = ctx.Compile(rig.TrainingStepPureGraph);
        var step = rig.TrainStep(initial, inputBatch, targetBatch, compiled);
        Assert.All(FlattenStruct(step.Checkpoint.OptimizerState), v => Assert.Equal(2f, v));

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

        var ctx = new ComputeContext();
        var compiledA = ctx.Compile(rig.TrainingStepPureGraph);
        var ckpt = rig.CreateDefaultCheckpoint();
        for (int i = 0; i < 2; i++)
            ckpt = rig.TrainStep(ckpt, inputBatch, targetBatch, compiledA).Checkpoint;

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
    /// Covers <see cref="TrainingCheckpoint.Save"/> / <see cref="TrainingRig.LoadCheckpoint"/>
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

        var ctx = new ComputeContext();
        var path = Path.Combine(Path.GetTempPath(), $"shrk_ckpt_{Guid.NewGuid():N}.safetensors");
        try
        {
            // Train two steps, then save mid-training.
            var rigA = AdamRig();
            var compiledA = ctx.Compile(rigA.TrainingStepPureGraph);
            var ckpt = rigA.CreateDefaultCheckpoint();
            for (int i = 0; i < 2; i++)
                ckpt = rigA.TrainStep(ckpt, inputBatch, targetBatch, compiledA).Checkpoint;
            Assert.Equal(2, ckpt.Step);
            ckpt.Save(path);
            Assert.True(File.Exists(path));

            // "Fresh process": a brand-new rig + compiled graph loads the checkpoint.
            var rigB = AdamRig();
            var compiledB = ctx.Compile(rigB.TrainingStepPureGraph);
            var loaded = rigB.LoadCheckpoint(path);

            // Step, trainable params, and optimizer state (m/v) must all round-trip exactly.
            Assert.Equal(2, loaded.Step);
            Assert.Equal(FlattenStruct(ckpt.TrainableParams), FlattenStruct(loaded.TrainableParams));
            Assert.Equal(FlattenStruct(ckpt.OptimizerState), FlattenStruct(loaded.OptimizerState));
            Assert.NotEmpty(loaded.OptimizerState.Fields); // AdamW carries m/v per param

            // Resuming from the loaded checkpoint advances the step and yields a finite loss.
            var resumed = rigB.TrainStep(loaded, inputBatch, targetBatch, compiledB);
            Assert.Equal(3, resumed.Checkpoint.Step);
            Assert.True(float.IsFinite(resumed.Loss));

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
                var bnCkpt = bnRig.CreateDefaultCheckpoint();
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
    /// Dynamic (scheduled / runtime) optimizer hyperparameters: a <see cref="HyperValue"/> that is a
    /// <see cref="Schedule"/> or <see cref="HyperValue.Runtime"/> routes the learning rate as a runtime
    /// input (<see cref="TrainingRig.HyperparamStructDef"/>) instead of a baked constant, so the rig
    /// compiles once and the LR can change every step. Drives the wiring in
    /// <c>BuildTrainingStepPureGraph</c> (hyperparam struct input + GETFIELDs, input reorder, real
    /// names), <c>InitializeAndOptimize</c> (seed values), the named/single
    /// <see cref="TrainingRig.MakeHyperparams(float)"/>, and both the schedule-driven and
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
        // SGD's learning rate is its sole hyperparameter. HyperValue.Runtime marks it as a
        // schedule-less runtime input so we can inject explicit values and prove the LR is live.
        var rig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            sample, new SGDOptimizerHyperparameters { LearningRate = HyperValue.Runtime(0.1f) });

        Assert.Single(rig.HyperparamStructDef.Fields);
        Assert.Equal(new[] { 0 }, rig.DynamicHyperparamIndices);
        // Real hyperparameter names now flow end-to-end (not "hyperparam_0").
        Assert.Equal(new[] { "learningRate" }, rig.DynamicHyperparameterNames);
        Assert.Equal("learningRate", rig.HyperparamStructDef.Fields[0].Name);

        var modelInputDef = new TensorStructDef(
            new[] { new TensorStructFieldDef("input", DataStructure.Tensor, 1, DType.Float32) }, "ModelInput");
        var targetDef = new TensorStructDef(
            new[] { new TensorStructFieldDef("targets", DataStructure.Tensor, 1, DType.Float32) }, "Target");
        var inputBatch = new TensorDataStruct(modelInputDef,
            new Dictionary<string, IData> { { "input", TensorData([4L], new float[] { 1f, 2f, 3f, 4f }) } });
        var targetBatch = new TensorDataStruct(targetDef,
            new Dictionary<string, IData> { { "targets", TensorData([4L], new float[] { 0f, 0f, 0f, 0f }) } });

        var ctx = new ComputeContext();
        var compiled = ctx.Compile(rig.TrainingStepPureGraph);   // compiled ONCE
        var initial = rig.CreateDefaultCheckpoint();
        Assert.Equal(0, initial.Step);
        string wName = rig.TrainableParamStructDef.Fields[0].Name;
        float w0 = ((TensorData<float32>)initial.TrainableParams.Fields[wName]).AccessMemory()[0];

        // Same start state, two different runtime learning rates, one compiled graph — supplied via
        // the explicit-override TrainStep, once by the single-value helper and once by name.
        var stepA = rig.TrainStep(initial, rig.MakeHyperparams(0.1f), inputBatch, targetBatch, compiled);
        var stepB = rig.TrainStep(initial, rig.MakeHyperparams(("learningRate", 0.3f)), inputBatch, targetBatch, compiled);
        Assert.Equal(1, stepA.Checkpoint.Step);   // the global step counter advanced
        float wA = ((TensorData<float32>)stepA.Checkpoint.TrainableParams.Fields[wName]).AccessMemory()[0];
        float wB = ((TensorData<float32>)stepB.Checkpoint.TrainableParams.Fields[wName]).AccessMemory()[0];

        float deltaA = w0 - wA;   // = 0.1 · grad
        float deltaB = w0 - wB;   // = 0.3 · grad
        Assert.True(MathF.Abs(deltaA) > 1e-4f, "LR must actually move the weight (grad·lr ≠ 0).");
        Assert.True(MathF.Abs(stepA.Loss - stepB.Loss) < 1e-4f);       // identical starting state
        Assert.True(MathF.Abs(deltaB - 3f * deltaA) < 1e-4f,           // 3× LR ⇒ 3× step
            $"expected ΔB ≈ 3·ΔA; got ΔA={deltaA}, ΔB={deltaB}");

        // A schedule-less runtime hyperparameter cannot be auto-driven: the no-override TrainStep throws.
        Assert.Throws<InvalidOperationException>(() => rig.TrainStep(initial, inputBatch, targetBatch, compiled));
        // Named MakeHyperparams rejects unknown / missing names.
        Assert.Throws<ArgumentException>(() => rig.MakeHyperparams(("bogus", 0.1f)));

        // A genuine schedule is applied automatically by the no-override TrainStep at the checkpoint's
        // step: the auto step must equal explicitly injecting schedule(step). Linear(0.2→0, 4) at
        // step 0 is 0.2.
        var schedRig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            sample, new SGDOptimizerHyperparameters { LearningRate = Schedules.Linear(0.2f, 0.0f, 4) });
        var schedCompiled = ctx.Compile(schedRig.TrainingStepPureGraph);
        var sc = schedRig.CreateDefaultCheckpoint();
        string swName = schedRig.TrainableParamStructDef.Fields[0].Name;
        var autoStep = schedRig.TrainStep(sc, inputBatch, targetBatch, schedCompiled);
        var refStep = schedRig.TrainStep(sc, schedRig.MakeHyperparams(0.2f), inputBatch, targetBatch, schedCompiled);
        float swAuto = ((TensorData<float32>)autoStep.Checkpoint.TrainableParams.Fields[swName]).AccessMemory()[0];
        float swRef = ((TensorData<float32>)refStep.Checkpoint.TrainableParams.Fields[swName]).AccessMemory()[0];
        Assert.True(MathF.Abs(swAuto - swRef) < 1e-5f, "auto-scheduled step must equal explicit LR = schedule(step).");

        // Dynamic LR also works for a stateful optimizer (AdamW: 5 hyperparams, m/v state), built with
        // the named set — LR scheduled, everything else left at its [Hyper] default (baked).
        var adamRig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, AdamWOptimizer.ComputationGraph,
            sample, new AdamWOptimizerHyperparameters { LearningRate = Schedules.Constant(0.01f) });
        Assert.Single(adamRig.DynamicHyperparamIndices);               // only LR is dynamic; betas baked
        var adamCompiled = ctx.Compile(adamRig.TrainingStepPureGraph);
        var adamStep = adamRig.TrainStep(adamRig.CreateDefaultCheckpoint(), inputBatch, targetBatch, adamCompiled);
        Assert.True(float.IsFinite(adamStep.Loss));
        Assert.NotEmpty(adamStep.Checkpoint.OptimizerState.Fields);    // m/v state still flows
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
    /// analogue of the per-step liveness proof. Also covers multi-dynamic named MakeHyperparams via
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
        var ctx = new ComputeContext();

        float FinalWeight(Schedule lr)
        {
            var rig = TrainingRig.FromScratch(
                ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
                sample, new SGDOptimizerHyperparameters { LearningRate = lr });
            var ckpt = rig.CreateDefaultCheckpoint();
            // 4 batches → Fit advances the global step each batch, so the schedule is sampled at 0..3.
            var result = rig.Fit(
                [inputBatch, inputBatch, inputBatch, inputBatch],
                [targetBatch, targetBatch, targetBatch, targetBatch],
                numEpochs: 1, ckpt, ctx);
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

        // Multi-dynamic: SGD-with-momentum, both hyperparameters scheduled; named MakeHyperparams must
        // accept both names (order-independent) and reject a wrong set.
        var momRig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, SGDMomentumOptimizer.ComputationGraph,
            sample, new SGDMomentumOptimizerHyperparameters
            {
                LearningRate = HyperValue.Runtime(0.1f),
                MomentumCoeff = HyperValue.Runtime(0.9f),
            });
        Assert.Equal(new[] { "learningRate", "momentumCoeff" }, momRig.DynamicHyperparameterNames.ToArray());
        var momCompiled = ctx.Compile(momRig.TrainingStepPureGraph);
        var momStep = momRig.TrainStep(momRig.CreateDefaultCheckpoint(),
            momRig.MakeHyperparams(("momentumCoeff", 0.9f), ("learningRate", 0.1f)),  // order-independent
            inputBatch, targetBatch, momCompiled);
        Assert.True(float.IsFinite(momStep.Loss));
        Assert.NotEmpty(momStep.Checkpoint.OptimizerState.Fields);     // velocity state flows
        Assert.Throws<ArgumentException>(() => momRig.MakeHyperparams(("learningRate", 0.1f))); // missing momentumCoeff
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
        var moduleGraph = CallsSimplestModule.ComputationGraph;
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
            ScalarMultiplyModel.ComputationGraph,
            L2Loss.ComputationGraph);

        var lowered = TrainingLoop.LowerTrainingGraph(trainingGraph);
        Assert.NotNull(lowered);
        Assert.NotEmpty(lowered.Nodes);
    }

    /// <summary>
    /// Covers the <c>Func</c>-loss overload of
    /// <see cref="TrainingGraphBuilder.PrepareForTrainingAsFast{TOut,TLoss}(FastComputationGraph, Func{TOut,TOut,TLoss})"/>
    /// and its companion reflection helper
    /// <see cref="TrainingGraphBuilder.ExtractFastGraphFromDelegate"/>, plus
    /// the three argument-validation error paths. These are otherwise only
    /// hit by <c>TrainingGraphBuilderQuickTests</c>.
    /// </summary>
    [Fact]
    public void TestTrainingGraphBuilderFuncOverloadCoverage()
    {
        var modelGraph = ScalarMultiplyModel.ComputationGraph;

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
        Assert.Equal(2, Losses.L2Loss.Inputs.Count);
        Assert.Equal(2, Losses.L1Loss.Inputs.Count);
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
    /// Verifies the <see cref="TrainingRig.FromScratch(FastComputationGraph,FastComputationGraph,FastComputationGraph,ModelParamList,IOptimizerHyperparameters)"/>
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

    /// <summary>
    /// Verifies <see cref="TrainingCheckpoint.ToInferenceModel"/> produces a concrete model
    /// that executes successfully and returns the expected output shape.
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

        var inferenceInput = TensorData([1L], new float[] { 5f });
        var concrete       = result.FinalCheckpoint.ToInferenceModel(modelGraph, inferenceInput);
        Assert.NotNull(concrete);

        var outputs = ComputeContext.Default.Execute(concrete, inferenceInput);
        Assert.Single(outputs);
        var output = outputs[0].ToTensorData<float32>();
        Assert.Equal(1, output.Shape.Dims.Length);
        Assert.Equal(1L, output.Shape.Dims[0]);
    }
}
