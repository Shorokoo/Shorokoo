using System.Globalization;
using Microsoft.ML.OnnxRuntime;
using Shorokoo.Core.Factory;
using Shorokoo.Core.Interpreter;
using Shorokoo.Core.Factory.IR;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Modules.Initializers;
using Shorokoo.Runtime;
using Shorokoo.Modules.Losses;
using Shorokoo.Modules.Optimizers;
using Shorokoo.Core.Nodes.Processors.Training;
using static Shorokoo.Tests.TrainingRigHelpers;

namespace Shorokoo.Tests;

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

[Module]
public partial class ScalarMultiplyWithOrtOnlyLoopIterCountModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var weight = InitScalarWeight.Init(Vector(1L));
        var scaled = input * weight;
        var identity = Tensor([2L, 2L], 1f, 0f, 0f, 1f);
        var det = (Scalar<float32>)OnnxOp.Det(identity);
        var iter = det.Cast<int64>();
        foreach (var ctx in LoopAPI.Iterate(iter))
        {
            scaled = scaled * Scalar(1.0f);
        }
        return scaled;
    }
}

[Module]
public partial class BatchedMatmulModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var embed = Scalar(8L);
        var classes = Scalar(4L);

        var q = input.MatMul(InitXavier.Init([embed, embed]));
        var scores = q.MatMul(q.Transpose(0, 2, 1));
        var attn = (Tensor<float32>)OnnxOp.Softmax(scores, axis: 2);
        var ctx = attn.MatMul(q);
        var pooled = ctx.Reduce(ReduceKind.Mean, Vector(1L), keepDims: false);
        return (Tensor<float32>)OnnxOp.Softmax(pooled.MatMul(InitXavier.Init([embed, classes])), axis: 1);
    }
}

[Module]
public partial class ParamTooLargeToAllocateModel
{
    public static Tensor<float32> Inline(Tensor<float32> x)
        => Zeros.Init([Scalar(1L << 25), Scalar(1L << 25)]);
}

/// <summary>Two trainable parameters, the FIRST too large to allocate and the SECOND larger
/// still. Initialization runs them one session apiece in order, so the one that fails is not the
/// one a "report the largest" message would name.</summary>
[Module]
public partial class TwoParamsFirstTooLargeModel
{
    public static Tensor<float32> Inline(Tensor<float32> x)
        => Zeros.Init([Scalar(1L << 25), Scalar(1L << 25)])
             * Zeros.Init([Scalar(1L << 26), Scalar(1L << 25)])
                 .Reduce(ReduceKind.Mean, null, keepDims: false).Scalar();
}

[Module]
public partial class ParamSizeOverflowingModel
{
    public static Tensor<float32> Inline(Tensor<float32> x)
        => Zeros.Init([Scalar(1L << 32), Scalar(1L << 32)]);
}

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

[Module]
public partial class ThreeInputMixedModel
{
    public static Tensor<float32> Inline(Tensor<float32> beta, Tensor<int64> alpha, Tensor<float32> gamma)
    {
        var w = InitScalarWeight.Init(Vector(1L));
        var sBeta = beta.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        var sAlpha = alpha.Reduce(ReduceKind.Sum, keepDims: false).Scalar().Cast<float32>();
        var sGamma = gamma.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return ((sBeta + sAlpha + sGamma) * w.Scalar()).Reshape([Scalar(1L)]);
    }
}

/// <summary>Two same-shaped parameters drawn from their own streams, the first scaling the input
/// and the second offsetting it.</summary>
[Module]
public partial class ParamOrderAModel
{
    public static Tensor<float32> Inline(Tensor<float32> x)
    {
        var scale = NormalDist.Init(Vector(1L), Scalar(0f), Scalar(1f));
        var offset = NormalDist.Init(Vector(1L), Scalar(0f), Scalar(1f));
        return x * scale.Scalar() + offset.Scalar();
    }
}

/// <summary>One <c>[4, 2]</c> weight applied to the input.</summary>
[Module]
public partial class ParamShapeNarrowModel
{
    public static Tensor<float32> Inline(Tensor<float32> x)
    {
        var w = NormalDist.Init(Vector(4L, 2L), Scalar(0f), Scalar(1f));
        return x.MatMul(w);
    }
}

/// <summary><see cref="ParamShapeNarrowModel"/> at a wider hidden size: the same parameter name
/// and rank, shaped <c>[4, 8]</c>.</summary>
[Module]
public partial class ParamShapeWideModel
{
    public static Tensor<float32> Inline(Tensor<float32> x)
    {
        var w = NormalDist.Init(Vector(4L, 8L), Scalar(0f), Scalar(1f));
        return x.MatMul(w);
    }
}

/// <summary>One weight over the representative-input threshold and one under it, so a rig built
/// from this describes the first and materializes the second.</summary>
[Module]
public partial class WideAndNarrowWeightsModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var wide = InitXavier.Init([Scalar(32L), Scalar(64L)]);
        var narrow = InitZeroBias.Init([Scalar(32L)]).Vec();
        return input.MatMul(wide.Transpose(1, 0)) + narrow;
    }
}

internal static class TrainingRigHelpers
{
    // A fresh array per call: a static readonly long[] is still mutable, and this suite
    // hands it to product code across four parallel workers.
    internal static long[] ScalarInputShape => [4L];

    internal static readonly TensorStructDef ScalarInputDef = new(
        [new TensorStructFieldDef("input", DataStructure.Tensor, 1, DType.Float32)], "ModelInput");

    internal static readonly TensorStructDef ScalarTargetDef = new(
        [new TensorStructFieldDef("targets", DataStructure.Tensor, 1, DType.Float32)], "Target");

    internal static TensorDataStruct InBatch(params float[] values) => new(ScalarInputDef,
        new Dictionary<string, IData> { { "input", TensorData([(long)values.Length], values) } });

    internal static TensorDataStruct TargetBatch(params float[] values) => new(ScalarTargetDef,
        new Dictionary<string, IData> { { "targets", TensorData([(long)values.Length], values) } });

    internal static long ProductOf(long[] shape)
    {
        long p = 1;
        foreach (var d in shape) p *= d;
        return p;
    }

    internal static float[] FlattenStruct(TensorDataStruct s) =>
        s.Definition.Fields
            .SelectMany(f => ((TensorData)s.Fields[f.Name]).As<float32>().AccessMemory<float>().ToArray())
            .ToArray();

    internal static string TempPath(string tag) =>
        Path.Combine(Path.GetTempPath(), $"shrk_{tag}_{Guid.NewGuid():N}");

    internal static (TrainingRig Rig, TrainingCheckpoint Ckpt) CoverFromScratch(
        ComputationGraph modelGraph,
        ComputationGraph lossGraph,
        ComputationGraph optimizerGraph,
        long[] inputShape,
        params Hyperparameter[] hyperparams)
    {
        var sampleInput = new TensorDataModelParam(
            "input", ModelParamType.InputParam,
            TensorData(inputShape, new float[ProductOf(inputShape)]));

        var rig = TrainingRig.FromScratch(modelGraph, lossGraph, optimizerGraph,
            [sampleInput], hyperparams);

        var checkpoint = rig.CreateInitialCheckpoint();
        Assert.NotEmpty(rig.TrainableParamStructDef.Fields);
        Assert.NotNull(checkpoint.TrainableParams);
        return (rig, checkpoint);
    }

    internal static (NamedModelParam[] sample, TensorDataStruct input, TensorDataStruct target) ScalarMultiplyBatches()
    {
        NamedModelParam[] sample =
        [
            new TensorDataModelParam("input", ModelParamType.InputParam,
                TensorData([4L], [1f, 2f, 3f, 4f])),
        ];
        return (sample, InBatch(1f, 2f, 3f, 4f), TargetBatch(0f, 0f, 0f, 0f));
    }

    internal static float Weight(TrainingRig rig, TrainingCheckpoint ckpt) =>
        ((TensorData<float32>)ckpt.TrainableParams.Fields[rig.TrainableParamStructDef.Fields[0].Name])
            .AccessMemory()[0];

    internal static TrainingRig LoaderRig(int batchSize, int features) =>
        TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            [
                new TensorDataModelParam("input", ModelParamType.InputParam,
                    TensorData([batchSize, features], new float[batchSize * features])),
            ],
            0.1f);

    internal static (TensorDataStruct inputs, TensorDataStruct targets) IndexDataset(
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

    internal static (TrainingRig Rig, TrainingCheckpoint Ckpt, TensorDataStruct In, TensorDataStruct Out)
        BuildTrainedAdamWRig(int steps)
    {
        NamedModelParam[] sample =
        [
            new TensorDataModelParam("input", ModelParamType.InputParam,
                TensorData(ScalarInputShape, [1f, 2f, 3f, 4f])),
        ];
        var rig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            AdamWOptimizer.ComputationGraph, sample,
            new AdamWOptimizerHyperparameters { LearningRate = 0.1f });

        var inBatch = InBatch(1f, 2f, 3f, 4f);
        var outBatch = TargetBatch(2f, 4f, 6f, 8f);

        var ckpt = rig.CreateInitialCheckpoint();
        for (int i = 0; i < steps; i++)
            ckpt = rig.TrainStep(ckpt, inBatch.Shared(), outBatch.Shared());
        return (rig, ckpt, inBatch, outBatch);
    }

    internal static float EvalLoss(ComputationGraph evaluationModel, float[] inputs, float[] targets)
        => ComputeContext.Default
            .Execute(evaluationModel, TensorData([(long)inputs.Length], inputs), TensorData([(long)targets.Length], targets))[0]
            .ToTensorData<float32>().ValueAt<float>(0);

    internal static byte[] ReadEntryBytesViaBcl(string path, string entryName)
    {
        using var zip = System.IO.Compression.ZipFile.OpenRead(path);
        using var s = zip.GetEntry(entryName)!.Open();
        using var buf = new MemoryStream();
        s.CopyTo(buf);
        return buf.ToArray();
    }
}

[Trait("Domain", "Training")]
[Trait("Purpose", "Coverage")]
public class TrainingRigFromScratchCoverageTests
{
    private static void CoverCheckpointRebind(
        ComputationGraph modelGraph,
        ComputationGraph lossGraph,
        ComputationGraph optimizerGraph,
        long[] inputShape,
        params Hyperparameter[] hyperparams)
    {
        long totalElements = ProductOf(inputShape);
        var sampleInput = new TensorDataModelParam(
            "input", ModelParamType.InputParam,
            TensorData(inputShape, new float[totalElements]));

        var rig = TrainingRig.FromScratch(modelGraph, lossGraph, optimizerGraph,
            [sampleInput], hyperparams);
        var checkpoint = rig.CreateInitialCheckpoint();

        var hints = new ModelParamList(
            [new KeyValuePair<string, TensorData>(modelGraph.ToInternal().Inputs[0].ToString(), TensorData(inputShape, new float[totalElements]))],
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
            Assert.True(scheme.ToModelId(p.Key, modelIds) is not null);

        var bound = concrete.ToConcreteModel(
            new ModelParamList(checkpointParams, ModelParamType.TrainableParam), scheme);
        Assert.NotNull(bound);
        Assert.NotNull(ctx.Compile(bound));
    }

    [Fact]
    public void TestASpecializedModuleGraphTrainsWithItsHypersBakedOutOfTheInputList()
    {
        var family = FCLayer.ComputationGraph;
        var specialized = family.Specialize(family.FromOrderedInputs([TensorData([], 3L)]));
        Assert.Equal(["input"], specialized.InputNames);

        NamedModelParam[] sample =
        [
            new TensorDataModelParam("input", ModelParamType.InputParam, TensorData([2L, 4L], new float[8])),
        ];
        var rig = TrainingRig.FromScratch(
            specialized, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph, sample,
            new SGDOptimizerHyperparameters { LearningRate = 0.05f });

        Assert.Equal(["learningRate"], rig.HyperparameterNames);
        Assert.Equal(2, rig.TrainableParamStructDef.Fields.Count());

        var step = rig.TrainStep(rig.CreateInitialCheckpoint(),
            NNLibraryTrainingFixtures.MakeBatch("input", "ModelInput", TensorData([2L, 4L], new float[8])),
            NNLibraryTrainingFixtures.MakeBatch("targets", "Target", TensorData([2L, 3L], new float[6])));
        Assert.NotNull(step.Loss);
    }

    /// <summary>
    /// A class-index loss takes a target of another shape and dtype than the model's output —
    /// <c>[N]</c> int64 against <c>[N, C]</c> float32 — and the rig stands the model's output in
    /// for it when it seeds shape inference and the memory-aware pass. The pass is then judged on
    /// a one-hot of <c>[N, C, C]</c>, which is harmless while C is a handful and is why every
    /// existing cross-entropy rig passes; at a language model's vocabulary it is 10 T elements,
    /// and building the rig fails outright.
    /// </summary>
    [Fact]
    public void TestAClassIndexLossIsOptimizedAgainstItsOwnTargetRatherThanThePrediction()
    {
        NamedModelParam[] narrow =
        [
            new TensorDataModelParam("input", ModelParamType.InputParam, TensorData([4L, 64L], new float[256])),
        ];
        var rig = TrainingRig.FromScratch(
            DigitClassifier.ComputationGraph, CrossEntropyLoss.ComputationGraph,
            SGDOptimizer.ComputationGraph, narrow, 0.01f);
        var target = rig.OptimizationInputShapes[^1];
        Assert.Equal(DType.Int64, target.DType);
        Assert.Equal([4L], target.Shape.Dims);

        NamedModelParam[] wide =
        [
            new TensorDataModelParam("input", ModelParamType.InputParam, TensorData([4L, 8L], new float[32])),
        ];
        var wideRig = TrainingRig.FromScratch(
            WideLogitClassifier.ComputationGraph, CrossEntropyLoss.ComputationGraph,
            SGDOptimizer.ComputationGraph, wide, 0.01f);
        var step = wideRig.TrainStep(wideRig.CreateInitialCheckpoint(),
            NNLibraryTrainingFixtures.MakeBatch("input", "ModelInput", TensorData([4L, 8L], new float[32])),
            NNLibraryTrainingFixtures.MakeBatch("targets", "Target", TensorData([4L], [0L, 1L, 2L, 3L])));
        Assert.True(float.IsFinite(step.Loss!.Value));

        var nll = TrainingRig.FromScratch(
            DigitClassifier.ComputationGraph, NLLLoss.ComputationGraph,
            SGDOptimizer.ComputationGraph, narrow, 0.01f).OptimizationInputShapes[^1];
        Assert.Equal(DType.Int64, nll.DType);
        Assert.Equal([4L], nll.Shape.Dims);
    }

    [Fact]
    public void TestPositionalHyperparametersPrecedeTheRngConfigAndContextsCoverage()
    {
        NamedModelParam[] sample =
        [
            new TensorDataModelParam("input", ModelParamType.InputParam, TensorData([4L], new float[4])),
        ];
        var cfg = new RngConfig { MasterSeed = 7 };
        var merge = new ComputeContext();
        var runtime = new ComputeContext();

        var rig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            SGDOptimizer.ComputationGraph, sample, [0.05f], cfg, merge, runtime);
        Assert.Equal(7UL, rig.RngConfig.MasterSeed);
        Assert.Same(merge, rig.MergeContext);
        Assert.Same(runtime, rig.RuntimeContext);
        Assert.Single(rig.TrainableParamStructDef.Fields);

        // Six arguments reach only the array overload, so the omitted contexts are its own defaults.
        var listRig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            ScalarMultiplyModel.ComputationGraph.FromOrderedInputs([TensorData([4L], new float[4])]),
            [0.05f], cfg);
        Assert.Equal(7UL, listRig.RngConfig.MasterSeed);
        Assert.Same(ComputeContext.Default, listRig.MergeContext);
        Assert.Same(ComputeContext.Default, listRig.RuntimeContext);
        Assert.Single(listRig.TrainableParamStructDef.Fields);

        Assert.Throws<ArgumentNullException>(() => TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            (ModelParamList)null!, [0.05f], cfg));
    }

    [Fact]
    public void TestFromScratchAcrossModelsLossesAndOptimizersCoverage()
    {
        var (sgdRig, sgdCkpt) = CoverFromScratch(ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            SGDOptimizer.ComputationGraph, [4L], 0.01f);
        Assert.Single(sgdRig.TrainableParamStructDef.Fields);
        Assert.Equal(1.0f, ((TensorData<float32>)sgdCkpt.TrainableParams
            .Fields[sgdRig.TrainableParamStructDef.Fields[0].Name]).AccessMemory()[0]);

        var (pfpRig, pfpCkpt) = CoverFromScratch(ScalarMultiplyParamFromParamModel.ComputationGraph,
            L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph, [4L], 0.01f);
        Assert.Equal(2, pfpRig.TrainableParamStructDef.Fields.Length);
        float[] PfpValues(TrainingCheckpoint c) => [.. pfpRig.TrainableParamStructDef.Fields
            .Select(f => ((TensorData<float32>)c.TrainableParams.Fields[f.Name]).AccessMemory()[0]).Order()];
        Assert.Equal([1.0f, 3.0f], PfpValues(pfpCkpt));
        var pfpStepped = PfpValues(pfpRig.TrainStep(pfpCkpt.Shared(), InBatch(1f, 2f, 3f, 4f), TargetBatch(2f, 4f, 6f, 8f)));
        Assert.All(PfpValues(pfpCkpt).Zip(pfpStepped), p => Assert.NotEqual(p.First, p.Second));
        CoverFromScratch(ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            SGDMomentumOptimizer.ComputationGraph, [4L], 0.5f, 0.9f);
        CoverFromScratch(ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            AdamWOptimizer.ComputationGraph, [4L], 0.001f, 0.9f, 0.999f, 1e-8f, 0.01f);
        CoverFromScratch(ScalarMultiplyWithBatchNormModel.ComputationGraph, L2Loss.ComputationGraph,
            SGDOptimizer.ComputationGraph, [8L], 0.5f);
        CoverFromScratch(ScalarMultiplyWithBatchNormModel.ComputationGraph, L2Loss.ComputationGraph,
            SGDMomentumOptimizer.ComputationGraph, [8L], 0.5f, 0.9f);
        CoverFromScratch(DigitClassifier.ComputationGraph, SoftmaxL2Loss.ComputationGraph,
            SGDMomentumOptimizer.ComputationGraph, [4L, 64L], 0.5f, 0.9f);
        CoverFromScratch(DigitClassifier.ComputationGraph, SoftmaxL2Loss.ComputationGraph,
            AdamWOptimizer.ComputationGraph, [4L, 64L], 0.001f, 0.9f, 0.999f, 1e-8f, 0.01f);
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
        CoverFromScratch(ScalarMultiplyWithQeeFoldableLoopIterCountModel.ComputationGraph,
            L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph, [4L], 0.01f);
        CoverFromScratch(ScalarMultiplyWithOrtOnlyLoopIterCountModel.ComputationGraph,
            L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph, [4L], 0.01f);
        CoverFromScratch(BatchedMatmulModel.ComputationGraph, SoftmaxL2Loss.ComputationGraph,
            SGDOptimizer.ComputationGraph, [4L, 5L, 8L], 0.01f);
        CoverCheckpointRebind(DigitClassifier.ComputationGraph, SoftmaxL2Loss.ComputationGraph,
            SGDOptimizer.ComputationGraph, [4L, 64L], 0.01f);
    }
}

[Trait("Domain", "Training")]
[Trait("Purpose", "Coverage")]
public class TrainingRigRepresentativeInputCoverageTests
{
    private const string ReprShapeAttr =
        Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames.ShrkAttrRepresentativeInputShape;

    private const string DtypeAttr =
        Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames.AttrDtype;

    private static TrainingRig RigWithInputShape(long[] shape)
        => TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            [
                new TensorDataModelParam("input", ModelParamType.InputParam,
                    TensorData(shape, new float[ProductOf(shape)])),
            ],
            0.01f);

    private static Shorokoo.Core.Graph.FastNode TensorInputNode(ComputationGraph graph)
        => graph.ToInternal().Nodes.First(
            n => n.OpCode == Shorokoo.Core.Nodes.NodeDefinitions.InternalOpCodes.MODEL_TENSOR_INPUT);

    private static Shorokoo.Core.Graph.FastNode InputNodeFor(
        Shorokoo.Graph.InternalComputationGraph g, Shorokoo.Core.Graph.FastTensorKey key)
        => g.Nodes.First(n => n.Outputs.Any(o => o.HasValue && o.Value.Equals(key)));

    private static TrainingRig OptionalRig(OptionalTensorData bias)
        => TrainingRig.FromScratch(
            NullableTrainableBiasLayer.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            [new TensorDataModelParam("x", ModelParamType.InputParam, TensorData([3L], 1f, 2f, 3f)),
             new OptionalTensorDataModelParam("bias", ModelParamType.InputParam, bias)], 0.1f);

    private static long[]? OptionalInputShape(ComputationGraph graph)
        => graph.ToInternal().Nodes
            .First(n => n.OpCode == Shorokoo.Core.Nodes.NodeDefinitions.InternalOpCodes.MODEL_OPTIONAL_INPUT)
            .Attributes.GetLongsVal(ReprShapeAttr);

    // An optional input records the arrangement it was concretized at as a value, never as a
    // missing attribute, and keeps it through both serialization dialects — so a reloaded arch
    // cannot read a present optional back as an absent one.
    [Fact]
    public void TestAnOptionalInputRecordsItsArrangementAndKeepsItThroughSerializationCoverage()
    {
        Assert.Equal((long[])[3L], OptionalInputShape(
            OptionalRig(OptionalTensorData.Some(TensorData([3L], 0f, 0f, 0f))).ConcreteArchConstituent));
        Assert.Equal((long[])[-1L], OptionalInputShape(
            OptionalRig(OptionalTensorData.None<float32>()).ConcreteArchConstituent));

        foreach (var bias in (OptionalTensorData[])[
            OptionalTensorData.Some(TensorData([3L], 0f, 0f, 0f)), OptionalTensorData.None<float32>()])
        {
            var arch = OptionalRig(bias).ConcreteArchConstituent;
            var expected = OptionalInputShape(arch);

            var srk = Shorokoo.Core.Utils.CompressedFormatUtils.LoadFastGraphFromBinary(
                Shorokoo.Core.Utils.CompressedFormatUtils.SaveFastGraphToBinary(arch));
            Assert.Equal(expected, OptionalInputShape(srk));

            var onnxPath = TempPath("rep_opt_onnx") + ".onnx";
            try
            {
                Persistence.ExportOnnx(OptionalRig(bias).CreateInitialCheckpoint().ToInferenceModel(), onnxPath);
                Assert.Equal(expected, OptionalInputShape(Persistence.ImportOnnx(onnxPath)));
            }
            finally { if (File.Exists(onnxPath)) File.Delete(onnxPath); }
        }
    }

    [Fact]
    public void TestRepresentativeInputShapeIsAlwaysDimsOnlyCoverage()
    {
        foreach (long n in (long[])[256L, 512L, 1024L, 2048L])
            CoverFromScratch(ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
                SGDOptimizer.ComputationGraph, [n], 0.01f);

        Assert.Equal((long[])[4L], TensorInputNode(
            RigWithInputShape([4L]).ConcreteArchConstituent).Attributes.GetLongsVal(ReprShapeAttr));
        Assert.Equal((long[])[2048L], TensorInputNode(
            RigWithInputShape([2048L]).ConcreteArchConstituent).Attributes.GetLongsVal(ReprShapeAttr));
    }

    [Fact]
    public void TestRepresentativeInputSurvivesDerivationReSeedAndSerializationCoverage()
    {
        NamedModelParam[] sample =
        [
            new TensorDataModelParam("input", ModelParamType.InputParam,
                TensorData([2048L], new float[2048])),
        ];
        var rig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            SGDOptimizer.ComputationGraph, sample, 0.01f);

        Assert.NotNull(rig.WithLoss(L2Loss.ComputationGraph).CreateInitialCheckpoint().TrainableParams);
        var reOpt = rig.WithOptimizer(SGDMomentumOptimizer.ComputationGraph, 0.5f, 0.9f);
        Assert.NotEmpty(reOpt.CreateInitialCheckpoint().OptimizerState.Fields);
        Assert.NotNull(reOpt.WithScheduler(0.25f, 0.9f).CreateInitialCheckpoint().TrainableParams);
        Assert.NotNull(rig.WithSeed(new RngConfig { MasterSeed = 7 }).CreateInitialCheckpoint().TrainableParams);

        var path = TempPath("repin_selfdesc") + ".skpt";
        try
        {
            Persistence.SaveTrainingCheckpointToSkpt(rig.CreateInitialCheckpoint(), path);

            var inferenceModel = Persistence.Load(path);
            Assert.Equal(GraphKind.ConcreteModel, inferenceModel.Kind);

            var (rig2, loaded) = TrainingRig.Load(path);
            Assert.NotNull(rig2);
            Assert.NotNull(loaded.TrainableParams);
            Assert.Equal((long[])[2048L],
                TensorInputNode(rig2.ConcreteArchConstituent).Attributes.GetLongsVal(ReprShapeAttr));
        }
        finally { if (File.Exists(path)) File.Delete(path); }

        AssertArchSrkRoundTrip([]);
        AssertArchSrkRoundTrip([4L]);
        AssertArchSrkRoundTrip([2048L]);

        AssertOnnxRepRoundTrip([]);
        AssertOnnxRepRoundTrip([4L]);
        AssertOnnxRepRoundTrip([2048L]);
    }

    private static void AssertArchSrkRoundTrip(long[] shape)
    {
        var arch = RigWithInputShape(shape).ConcreteArchConstituent;
        var bytes = Shorokoo.Core.Utils.CompressedFormatUtils.SaveFastGraphToBinary(arch);
        var reloaded = Shorokoo.Core.Utils.CompressedFormatUtils.LoadFastGraphFromBinary(bytes);
        var internalReloaded = reloaded.ToInternal();

        Assert.Single(internalReloaded.Inputs);

        var node = internalReloaded.Nodes.First(
            n => n.OpCode == Shorokoo.Core.Nodes.NodeDefinitions.InternalOpCodes.MODEL_TENSOR_INPUT);
        Assert.Equal(shape, node.Attributes.GetLongsVal(ReprShapeAttr));
    }

    private static void AssertOnnxRepRoundTrip(long[] setShape)
    {
        var baseModel = RigWithInputShape([4L]).CreateInitialCheckpoint().ToInferenceModel();
        var internalModel = baseModel.ToInternal();
        var inputNode = internalModel.Nodes.First(
            n => n.OpCode == Shorokoo.Core.Nodes.NodeDefinitions.InternalOpCodes.MODEL_TENSOR_INPUT);
        inputNode.Attributes = inputNode.Attributes.SetAttributes(
            (ReprShapeAttr, (object?)setShape));
        var modelWithRep = new ComputationGraph(internalModel, GraphKind.ConcreteModel);

        var onnxPath = TempPath("rep_onnx") + ".onnx";
        try
        {
            Persistence.ExportOnnx(modelWithRep, onnxPath);

            using (var ms = new MemoryStream(File.ReadAllBytes(onnxPath)))
            {
                var proto = ProtoBuf.Serializer.Deserialize<Shorokoo.Core.Factory.IR.ModelProto>(ms);
                Assert.Single(proto.Graph.Inputs);
                var repr = proto.Graph.Inputs[0].MetadataProps.Single(
                    p => p.Key == Shorokoo.Core.Factory.RepresentativeInputMetadata.Key);
                Assert.StartsWith("shape|", repr.Value);
                Assert.DoesNotContain(proto.MetadataProps,
                    p => p.Key == Shorokoo.Core.Factory.RepresentativeInputMetadata.Key
                      || p.Key.StartsWith("shrk_repr_input", StringComparison.Ordinal));
            }

            var imported = Persistence.ImportOnnx(onnxPath);
            var node = imported.ToInternal().Nodes.First(
                n => n.OpCode == Shorokoo.Core.Nodes.NodeDefinitions.InternalOpCodes.MODEL_TENSOR_INPUT);
            Assert.Equal(setShape, node.Attributes.GetLongsVal(ReprShapeAttr));
        }
        finally { if (File.Exists(onnxPath)) File.Delete(onnxPath); }
    }

    [Fact]
    public void TestAShapeExemplarCarriesValuesOnlyWhileAnEngineWouldReadThemCoverage()
    {
        (long[] Dims, bool Values)[] cases =
        [
            ([], true), ([1L], true), ([32L, 32L], true), ([1024L], true),
            ([1025L], false), ([2048L], false), ([64L, 64L], false), ([1L << 20], false),
        ];
        foreach (var (dims, values) in cases)
        {
            var shape = new Shape(dims);
            var slot = TrainingRig.RepresentativeInputFor(shape, DType.Float32);
            Assert.Equal(values, slot.HasValues);
            Assert.Equal(dims, slot.Shape.Dims);
            Assert.Equal(DType.Float32, slot.DType);

            var exemplar = TrainingRig.RepresentativeRuntimeInputFor(shape, DType.Float32);
            Assert.Equal(values, exemplar.HasAnyData);
            Assert.Equal(dims, exemplar.Shape!.Dims);
            Assert.Equal(DType.Float32, exemplar.DType);
            Assert.Equal(values ? new float[ProductOf(dims)] : null, exemplar.FloatData?.ToArray());
        }
    }

    [Fact]
    public void TestALargeModelInputReachesTheOptimizationPassAsShapeAndDTypeOnlyCoverage()
    {
        (long N, bool Values)[] cases = [(4L, true), (1024L, true), (1025L, false), (2048L, false)];
        foreach (var (n, values) in cases)
        {
            var exemplars = RigWithInputShape([n]).OptimizationInputs
                .OfType<RuntimeTensor>().Where(t => t.Shape!.Dims.SequenceEqual((long[])[n])).ToList();
            Assert.Equal(2, exemplars.Count);
            Assert.All(exemplars, t => Assert.Equal(values, t.HasAnyData));
            Assert.All(exemplars, t => Assert.Equal(values ? new float[n] : null, t.FloatData?.ToArray()));
        }
    }

    [Fact]
    public void TestADeferredBuildDescribesItsLargeParametersRatherThanMaterializingThemCoverage()
    {
        var rig = TrainingRig.FromScratch(
            WideAndNarrowWeightsModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            [new TensorDataModelParam("input", ModelParamType.InputParam, TensorData([4L, 64L], new float[256]))],
            0.01f);
        var path = TempPath("repin_defer") + ".skpt";
        try
        {
            Persistence.SaveTrainingCheckpointToSkpt(rig.CreateInitialCheckpoint(), path);
            var (deferred, _) = TrainingRig.Load(path);

            foreach (var inputs in (IRuntimeTensor[][])[rig.OptimizationInputs, deferred.OptimizationInputs])
            {
                var byShape = inputs.OfType<RuntimeTensor>()
                    .ToDictionary(t => string.Join(",", t.Shape!.Dims), t => t.HasAnyData);
                Assert.False(byShape["32,64"]);
                Assert.True(byShape["32"]);
                Assert.True(byShape["4,64"]);
                Assert.True(byShape["4,32"]);
            }

            Assert.Equal(
                FlattenStruct(rig.CreateInitialCheckpoint().TrainableParams),
                FlattenStruct(deferred.CreateInitialCheckpoint().TrainableParams));
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    [Fact]
    public void TestMultiInputArchSrkNodeRoundTripPreservesInputListAndReprAttrsCoverage()
    {
        NamedModelParam[] sample =
        [
            new TensorDataModelParam("beta", ModelParamType.InputParam, TensorData([4L], new float[4])),
            new TensorDataModelParam("alpha", ModelParamType.InputParam, TensorData([2048L], new long[2048])),
            new TensorDataModelParam("gamma", ModelParamType.InputParam, TensorData([2L, 3L], new float[6])),
        ];
        var arch = TrainingRig.FromScratch(
            ThreeInputMixedModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            sample, 0.01f).ConcreteArchConstituent;
        Assert.Equal(GraphKind.ConcreteArchitecture, arch.Kind);

        var original = arch.ToInternal();
        var reloaded = Shorokoo.Core.Utils.CompressedFormatUtils.LoadFastGraphFromBinary(
            Shorokoo.Core.Utils.CompressedFormatUtils.SaveFastGraphToBinary(arch)).ToInternal();

        Assert.Equal(3, original.Inputs.Count);
        Assert.Equal(original.Inputs.Count, reloaded.Inputs.Count);
        Assert.Equal(original.InputUniqueNames, reloaded.InputUniqueNames);
        Assert.Equal(3, reloaded.InputUniqueNames.Distinct().Count());

        long[][] expectDims = [[4L], [2048L], [2L, 3L]];
        DType[] expectDtype = [DType.Float32, DType.Int64, DType.Float32];

        for (int i = 0; i < 3; i++)
        {
            var before = InputNodeFor(original, original.Inputs[i]);
            var after = InputNodeFor(reloaded, reloaded.Inputs[i]);
            Assert.Equal(Shorokoo.Core.Nodes.NodeDefinitions.InternalOpCodes.MODEL_TENSOR_INPUT, after.OpCode);

            Assert.Equal(expectDtype[i], before.Attributes.GetDTypeVal(DtypeAttr));
            Assert.Equal(expectDtype[i], after.Attributes.GetDTypeVal(DtypeAttr));
            Assert.Equal(expectDims[i], before.Attributes.GetLongsVal(ReprShapeAttr));
            Assert.Equal(expectDims[i], after.Attributes.GetLongsVal(ReprShapeAttr));
        }
    }
}

[Trait("Domain", "Training")]
[Trait("Purpose", "Coverage")]
public class TrainingRigCompositionCoverageTests
{
    private static TrainingRig SelfScoringRig() => TrainingRig.FromScratch(
        SelfScoringModel.ComputationGraph, ForwardingLoss.ComputationGraph, SGDOptimizer.ComputationGraph,
        [new TensorDataModelParam("input", ModelParamType.InputParam, TensorData([4L], [1f, 2f, 3f, 4f]))],
        0.1f);

    [Fact]
    public void TestALossThatIgnoresItsTargetDerivesNoTargetFieldAndStepsWithoutOneCoverage()
    {
        var rig = SelfScoringRig();
        Assert.False(rig.HasTargets);
        Assert.Empty(rig.TargetDef.Fields);

        var inputs = rig.InputDef.FromOrderedData(TensorData([4L], [1f, 2f, 3f, 4f]));
        var start = rig.CreateInitialCheckpoint().Shared();
        var stepped = rig.TrainStep(start, inputs.Shared());
        Assert.Equal(1, stepped.Step);
        Assert.True(stepped.Loss > 0f);
        Assert.True(rig.TrainStep(stepped, inputs.Shared()).Loss < stepped.Loss);

        Assert.Equal(
            FlattenStruct(rig.TrainStep(start, inputs.Shared()).TrainableParams),
            FlattenStruct(rig.TrainStep(start, inputs.Shared(), rig.TargetDef.FromOrderedData()).TrainableParams));

        var fitted = rig.Fit([inputs, inputs], numEpochs: 3);
        Assert.Equal(3, fitted.EpochLosses.Length);
        Assert.True(fitted.EpochLosses[^1] < fitted.EpochLosses[0]);
        Assert.Equal(6, fitted.FinalCheckpoint.Step);

        var targeted = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            ScalarMultiplyBatches().sample, 0.1f);
        Assert.True(targeted.HasTargets);
        Assert.Throws<InvalidOperationException>(
            () => targeted.TrainStep(targeted.CreateInitialCheckpoint(), InBatch(1f, 2f, 3f, 4f)));
        Assert.Throws<InvalidOperationException>(() => targeted.Fit([InBatch(1f, 2f, 3f, 4f)], numEpochs: 1));
    }

    [Fact]
    public void TestAScalarIgnoredTargetAndAModelInputNamedTargetsBothComposeCoverage()
    {
        var scalarRig = TrainingRig.FromScratch(
            SelfScoringModel.ComputationGraph, ScalarTargetForwardingLoss.ComputationGraph,
            SGDOptimizer.ComputationGraph,
            [new TensorDataModelParam("input", ModelParamType.InputParam, TensorData([4L], [1f, 2f, 3f, 4f]))],
            0.1f);
        Assert.False(scalarRig.HasTargets);
        var scalarIn = scalarRig.InputDef.FromOrderedData(TensorData([4L], [1f, 2f, 3f, 4f]));
        var scalarCkpt = scalarRig.TrainStep(scalarRig.CreateInitialCheckpoint(), scalarIn.Shared());
        Assert.True(scalarCkpt.Loss > 0f);

        var namedRig = TrainingRig.FromScratch(
            TargetsNamedInputModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            [new TensorDataModelParam("targets", ModelParamType.InputParam, TensorData([4L], [1f, 2f, 3f, 4f]))],
            0.1f);
        var namedIn = namedRig.InputDef.FromOrderedData(TensorData([4L], [1f, 2f, 3f, 4f]));
        var namedTg = namedRig.TargetDef.FromOrderedData(TensorData([4L], [2f, 4f, 6f, 8f]));
        var namedCkpt = namedRig.TrainStep(namedRig.CreateInitialCheckpoint(), namedIn.Shared(), namedTg.Shared());

        var scalarPath = TempPath("skpt_scalar_target") + ".skpt";
        var namedPath = TempPath("skpt_named_target") + ".skpt";
        try
        {
            Persistence.SaveTrainingCheckpointToSkpt(scalarCkpt, scalarPath);
            Assert.False(Persistence.EvaluationModelTakesTarget(scalarPath));
            Assert.Equal((double)scalarRig.TrainStep(scalarCkpt.Shared(), scalarIn).Loss!.Value,
                ComputeContext.Default.Execute(Persistence.LoadEvaluationModel(scalarPath), TensorData([4L], [1f, 2f, 3f, 4f]))[0]
                    .ToTensorData<float32>().ValueAt<float>(0), 5);

            Persistence.SaveTrainingCheckpointToSkpt(namedCkpt, namedPath);
            var namedEval = Persistence.LoadEvaluationModel(namedPath);
            Assert.Equal(namedEval.InputNames.Count, namedEval.InputNames.Distinct().Count());
            Assert.Equal((double)namedRig.TrainStep(namedCkpt.Shared(), namedIn, namedTg).Loss!.Value,
                EvalLoss(namedEval, [1f, 2f, 3f, 4f], [2f, 4f, 6f, 8f]), 5);
        }
        finally
        {
            if (File.Exists(scalarPath)) File.Delete(scalarPath);
            if (File.Exists(namedPath)) File.Delete(namedPath);
        }
    }

    [Fact]
    public void TestATargetReadOnlyThroughABranchConditionStillDerivesATargetFieldCoverage()
    {
        var rig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, TargetGatedLoss.ComputationGraph,
            SGDOptimizer.ComputationGraph, ScalarMultiplyBatches().sample, 0.1f);

        Assert.True(rig.HasTargets);
        Assert.Equal(1, rig.TargetDef.Fields.Length);

        var ckpt = rig.CreateInitialCheckpoint();
        var inputs = InBatch(1f, 2f, 3f, 4f);
        var gated = rig.TrainStep(ckpt.Shared(), inputs.Shared(), TargetBatch(1f, 1f, 1f, 1f)).Loss!.Value;
        var ungated = rig.TrainStep(ckpt.Shared(), inputs.Shared(), TargetBatch(-1f, -1f, -1f, -1f)).Loss!.Value;
        Assert.Equal((double)(gated * 2f), ungated, 4);
        Assert.Throws<InvalidOperationException>(() => rig.TrainStep(ckpt.Shared(), inputs.Shared()));

        var path = TempPath("skpt_gated") + ".skpt";
        try
        {
            Persistence.SaveTrainingCheckpointToSkpt(ckpt, path);
            Assert.True(Persistence.EvaluationModelTakesTarget(path));
            var eval = Persistence.LoadEvaluationModel(path);
            Assert.Equal((double)gated, EvalLoss(eval, [1f, 2f, 3f, 4f], [1f, 1f, 1f, 1f]), 4);
            Assert.Equal((double)ungated, EvalLoss(eval, [1f, 2f, 3f, 4f], [-1f, -1f, -1f, -1f]), 4);
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    [Fact]
    public void TestATargetlessRigDrivesALoaderAndAResidentRunWithoutATargetCoverage()
    {
        var rig = SelfScoringRig();
        var rows = new float[8 * 4];
        for (var i = 0; i < rows.Length; i++) rows[i] = (i % 7) / 7f;
        var dataset = new TensorDataStruct(rig.InputDef,
            new Dictionary<string, IData> { { "input", TensorData([8L, 4L], rows) } });

        var loader = new InMemoryDataLoader(dataset, rig.TargetDef.FromOrderedData(), batchSize: 4);
        Assert.Equal(2, loader.BatchesPerEpoch);

        var start = rig.CreateInitialCheckpoint().Shared();
        var stepped = rig.TrainStep(start, loader);
        Assert.Equal(1, stepped.Step);
        Assert.Equal(0, stepped.Epoch);
        Assert.True(stepped.Loss > 0f);

        var fitted = rig.Fit(loader, numEpochs: 1, stepped);
        Assert.True(fitted.FinalCheckpoint.Step > stepped.Step);

        var inputs = rig.InputDef.FromOrderedData(TensorData([4L, 4L], rows[..16]));
        using var run = rig.BeginResidentRun(start);
        var residentLoss = run.Step(inputs.Shared());
        Assert.True(residentLoss > 0f);
        var published = run.StepToCheckpoint(inputs.Shared());
        Assert.Equal(2, published.Step);

        var targeted = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            ScalarMultiplyBatches().sample, 0.1f);
        using var targetedRun = targeted.BeginResidentRun(targeted.CreateInitialCheckpoint());
        Assert.Throws<InvalidOperationException>(() => targetedRun.Step(InBatch(1f, 2f, 3f, 4f)));
        Assert.Throws<InvalidOperationException>(() => targetedRun.StepToCheckpoint(InBatch(1f, 2f, 3f, 4f)));
    }

    [Fact]
    public void TestAnIgnoredTargetIsAbsentFromTheCheckpointsEvaluationModelTooCoverage()
    {
        var rig = SelfScoringRig();
        var inputs = rig.InputDef.FromOrderedData(TensorData([4L], [1f, 2f, 3f, 4f]));
        var ckpt = rig.TrainStep(rig.CreateInitialCheckpoint(), inputs.Shared());
        var expected = rig.TrainStep(ckpt.Shared(), inputs.Shared()).Loss!.Value;
        var path = TempPath("skpt_notarget") + ".skpt";
        try
        {
            Persistence.SaveTrainingCheckpointToSkpt(ckpt, path);
            Assert.False(Persistence.EvaluationModelTakesTarget(path));
            var eval = Persistence.LoadEvaluationModel(path);
            Assert.Equal(GraphKind.ConcreteModel, eval.Kind);
            var loss = ComputeContext.Default
                .Execute(eval, TensorData([4L], [1f, 2f, 3f, 4f]))[0].ToTensorData<float32>().ValueAt<float>(0);
            Assert.Equal((double)expected, loss, 5);
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    [Fact]
    public void TestFromScratchGraphKindsAndConvenienceApisCoverage()
    {
        var modelGraph = ScalarMultiplyModel.ComputationGraph;
        var exampleInput = TensorData([4L], [1f, 2f, 3f, 4f]);

        var rig = TrainingRig.FromScratch(
            modelGraph, Losses.L2Loss, Optimizers.SGD,
            modelGraph.FromOrderedInputs([exampleInput]),
            0.01f);

        var namedHyperRig = TrainingRig.FromScratch(
            modelGraph, Losses.L2Loss, Optimizers.SGD,
            modelGraph.FromOrderedInputs([exampleInput]),
            new SGDOptimizerHyperparameters { LearningRate = 0.01f });
        Assert.NotEmpty(namedHyperRig.TrainableParamStructDef.Fields);
        Assert.Throws<ArgumentNullException>(() => TrainingRig.FromScratch(
            modelGraph, Losses.L2Loss, Optimizers.SGD,
            (ModelParamList)null!, new SGDOptimizerHyperparameters { LearningRate = 0.01f }));
        Assert.Throws<ArgumentNullException>(() => TrainingRig.FromScratch(
            modelGraph, Losses.L2Loss, Optimizers.SGD, (ModelParamList)null!, 0.01f));

        Assert.NotNull(rig.InputDef);
        Assert.Equal(1, rig.InputDef.Fields.Length);
        Assert.Equal("input", rig.InputDef.Fields[0].Name);
        Assert.NotNull(rig.TargetDef);
        Assert.Equal(1, rig.TargetDef.Fields.Length);
        Assert.Equal("targets", rig.TargetDef.Fields[0].Name);

        var inputBatch = rig.InputDef.FromOrderedData(exampleInput);
        var targetBatch = rig.TargetDef.FromOrderedData(TensorData([4L], new float[4]));
        Assert.NotNull(inputBatch);
        Assert.NotNull(targetBatch);
        Assert.Same(rig.InputDef, inputBatch.Definition);
        Assert.Same(rig.TargetDef, targetBatch.Definition);

        var result = rig.Fit([inputBatch, inputBatch], [targetBatch, targetBatch], numEpochs: 1);
        Assert.Single(result.EpochLosses);
        Assert.True(float.IsFinite(result.EpochLosses[0]));

        var arch = modelGraph.ToConcreteArchitecture(modelGraph.FromOrderedInputs([exampleInput]));
        var sampleInput = new TensorDataModelParam("input", ModelParamType.InputParam, exampleInput);

        var archRig = TrainingRig.FromScratch(
            arch, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            [sampleInput], 0.5f);
        Assert.NotEmpty(archRig.TrainableParamStructDef.Fields);
        Assert.Equal(GraphKind.ConcreteModel, archRig.TrainingStepPureGraph.Kind);
        Assert.NotNull(archRig.CreateInitialCheckpoint().TrainableParams);

        var exModel = Assert.Throws<InvalidOperationException>(() => TrainingRig.FromScratch(
            arch.ToConcreteModel(), L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            [sampleInput], 0.5f));
        Assert.Contains("'concrete-model'", exModel.Message);
        Assert.Contains("'concrete-architecture'", exModel.Message);

        var exLoss = Assert.Throws<InvalidOperationException>(() => TrainingRig.FromScratch(
            modelGraph, arch, SGDOptimizer.ComputationGraph,
            [sampleInput], 0.5f));
        Assert.Contains("'module'", exLoss.Message);
    }

    [Fact]
    public void TestTrainingGraphLoweringBuilderOverloadsAndParamDiscoveryCoverage()
    {
        var scalarMultiply = ScalarMultiplyModel.ComputationGraph;
        InternalComputationGraph ConcreteScalarMultiply() => scalarMultiply.ToConcreteArchitecture(
            scalarMultiply.FromOrderedInputs([TensorData([4L], [1f, 2f, 3f, 4f])])).ToInternal();

        var trainingGraph = TrainingGraphBuilder.PrepareForTrainingAsFast(
            ConcreteScalarMultiply(), L2Loss.ComputationGraph.ToInternal());
        var lowered = TrainingLoop.LowerTrainingGraph(trainingGraph);
        Assert.NotNull(lowered);
        Assert.NotEmpty(lowered.Nodes);

        // A model graph that was never lowered is refused: training needs the parameter shapes and
        // initial values only ToConcreteArchitecture resolves.
        Assert.Throws<ArgumentException>(() => TrainingGraphBuilder.PrepareForTrainingAsFast(
            ScalarMultiplyModel.ComputationGraph.ToInternal(), L2Loss.ComputationGraph.ToInternal()));

        var modelGraph = ConcreteScalarMultiply();
        Func<Tensor<float32>, Tensor<float32>, Scalar<float32>> lossFunc = L2Loss.Inline;
        var funcTrainingGraph = TrainingGraphBuilder.PrepareForTrainingAsFast(modelGraph, lossFunc);
        Assert.True(funcTrainingGraph.Inputs.Count >= 3);
        Assert.True(funcTrainingGraph.Outputs.Count >= 2);

        Assert.Throws<ArgumentNullException>(() =>
            TrainingGraphBuilder.PrepareForTrainingAsFast<Tensor<float32>, Scalar<float32>>(modelGraph, null!));
        Assert.Throws<ArgumentNullException>(() =>
            TrainingGraphBuilder.PrepareForTrainingAsFast<Tensor<float32>, Scalar<float32>>(null!, lossFunc));

        Func<Tensor<float32>, Tensor<float32>, Scalar<float32>> notAModule =
            (pred, targ) => ((Tensor<float32>)OnnxOp.ReduceSum(pred - targ, keepdims: false)).Scalar();
        Assert.Throws<ArgumentException>(() =>
            TrainingGraphBuilder.PrepareForTrainingAsFast(modelGraph, notAModule));

        var moduleGraph = CallsSimplestModule.ComputationGraph.ToInternal();
        Assert.Contains(moduleGraph.Nodes, n =>
            n.OpCode == InternalOpCodes.MODEL_INVOKE || n.OpCode == InternalOpCodes.FUNCTION_INVOKE);
        Assert.Throws<System.InvalidOperationException>(() => moduleGraph.GetConcreteModelParamInfos());
        Assert.Throws<System.InvalidOperationException>(() => moduleGraph.InitializeTrainableParams());

        var arch = moduleGraph.ToConcreteArchitecture(
            moduleGraph.FromOrderedInputs([TensorData([4L], [1f, 2f, 3f, 4f])]));
        Assert.NotEmpty(arch.GetConcreteModelParamInfos().ParamInfos);
        Assert.NotEmpty(arch.InitializeTrainableParams().ModelParams);
    }

    [Fact]
    public void TestOnlyAnAllocationFailureIsReportedAndItNamesTheParametersShapesAndSizes()
    {
        var graph = ParamTooLargeToAllocateModel.ComputationGraph.ToInternal();
        var arch = graph.ToConcreteArchitecture(
            graph.FromOrderedInputs([TensorData([1L, 4L], [1f, 2f, 3f, 4f])]));

        var ex = Assert.Throws<ComputeContextException>(() => arch.InitializeTrainableParams());
        Assert.Contains("Zeros", ex.Message);
        Assert.Contains("[33554432, 33554432]", ex.Message);
        Assert.Contains("4.00 PiB", ex.Message);
        Assert.NotNull(ex.InnerException);

        var other = ParamSizeOverflowingModel.ComputationGraph.ToInternal();
        var otherArch = other.ToConcreteArchitecture(
            other.FromOrderedInputs([TensorData([1L, 4L], [1f, 2f, 3f, 4f])]));
        Assert.IsNotType<ComputeContextException>(
            Record.Exception(() => otherArch.InitializeTrainableParams()));

        // Each parameter initializes in its own session, so the failure names the one that
        // actually failed — here the smaller of the two, which is initialized first — and lists
        // the larger only as context.
        var two = TwoParamsFirstTooLargeModel.ComputationGraph.ToInternal();
        var twoArch = two.ToConcreteArchitecture(
            two.FromOrderedInputs([TensorData([1L, 4L], [1f, 2f, 3f, 4f])]));
        var twoEx = Assert.Throws<ComputeContextException>(() => twoArch.InitializeTrainableParams());
        Assert.Contains("[33554432, 33554432] = 4.00 PiB failed", twoEx.Message);
        Assert.Contains("[67108864, 33554432] = 8.00 PiB", twoEx.Message);
        Assert.Contains("1 of 2", twoEx.Message);
    }

    [Fact]
    public void TestLossAndOptimizerHubsCoverage()
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
        Assert.Equal(2, Losses.L2Loss.ToInternal().Inputs.Count);
        Assert.Equal(2, Losses.L1Loss.ToInternal().Inputs.Count);

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

    [Fact]
    public void TestRigDerivationsShareConstituentsAndAreImmutableCoverage()
    {
        var (sample, input, target) = ScalarMultiplyBatches();
        var model = ScalarMultiplyModel.ComputationGraph;
        var loss = L2Loss.ComputationGraph;
        var opt = SGDOptimizer.ComputationGraph;

        var rig = TrainingRig.FromScratch(model, loss, opt, sample, 0.1f);

        Assert.Same(model, rig.ModelConstituent);
        Assert.Same(loss, rig.LossConstituent);
        Assert.Same(opt, rig.OptimizerConstituent);
        Assert.Equal(0UL, rig.RngConfig.MasterSeed);
        Assert.Empty(rig.OptimizerStateDef.Fields);

        var newLoss = Losses.L1Loss;
        var lossRig = rig.WithLoss(newLoss);
        Assert.NotSame(rig, lossRig);
        Assert.Same(newLoss, lossRig.LossConstituent);
        Assert.Same(model, lossRig.ModelConstituent);
        Assert.Same(opt, lossRig.OptimizerConstituent);
        Assert.Same(loss, rig.LossConstituent);

        var momRig = rig.WithOptimizer(SGDMomentumOptimizer.ComputationGraph,
            new SGDMomentumOptimizerHyperparameters { LearningRate = 0.5f, MomentumCoeff = 0.9f });
        Assert.Same(model, momRig.ModelConstituent);
        Assert.Same(loss, momRig.LossConstituent);
        Assert.NotSame(opt, momRig.OptimizerConstituent);
        Assert.NotEmpty(momRig.OptimizerStateDef.Fields);
        Assert.Empty(rig.OptimizerStateDef.Fields);

        var schedRig = rig.WithScheduler(
            new SGDOptimizerHyperparameters { LearningRate = Schedules.Linear(0.2f, 0f, 4) });
        Assert.Same(opt, schedRig.OptimizerConstituent);
        Assert.Empty(schedRig.HyperparameterStructDef.Fields);

        var reseeded = rig.WithSeed(new RngConfig { MasterSeed = 42 });
        Assert.Same(model, reseeded.ModelConstituent);
        Assert.Same(opt, reseeded.OptimizerConstituent);
        Assert.Equal(42UL, reseeded.RngConfig.MasterSeed);
        Assert.Equal(0UL, rig.RngConfig.MasterSeed);

        foreach (var derived in (TrainingRig[])[lossRig, momRig, schedRig, reseeded])
        {
            var stepped = derived.TrainStep(derived.CreateInitialCheckpoint(), input.Shared(), target.Shared());
            Assert.True(float.IsFinite(stepped.Loss!.Value));
        }
    }

    [Fact]
    public void TestComputeContextsStoredPropagatedAndNeverPersistedCoverage()
    {
        NamedModelParam[] sample =
        [
            new TensorDataModelParam("input", ModelParamType.InputParam,
                TensorData([4L], new float[4])),
        ];

        var defaultRig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            SGDOptimizer.ComputationGraph, sample, 0.01f);
        Assert.Same(ComputeContext.Default, defaultRig.MergeContext);
        Assert.Same(ComputeContext.Default, defaultRig.RuntimeContext);

        var merge = new ComputeContext();
        var runtime = new ComputeContext();
        var rig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            SGDOptimizer.ComputationGraph, sample, [0.01f], null, merge, runtime);
        Assert.Same(merge, rig.MergeContext);
        Assert.Same(runtime, rig.RuntimeContext);

        var derived = rig.WithLoss(L2Loss.ComputationGraph);
        Assert.Same(merge, derived.MergeContext);
        Assert.Same(runtime, derived.RuntimeContext);

        var reseeded = rig.WithSeed(new RngConfig { MasterSeed = 3 });
        Assert.Same(merge, reseeded.MergeContext);
        Assert.Same(runtime, reseeded.RuntimeContext);

        var path = TempPath("ctx_notpersisted") + ".safetensors";
        try
        {
            rig.CreateInitialCheckpoint().Save(path);

            var loaderMerge = new ComputeContext();
            var loaderRuntime = new ComputeContext();
            var loaderRig = TrainingRig.FromScratch(
                ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
                SGDOptimizer.ComputationGraph, sample, [0.01f], null, loaderMerge, loaderRuntime);
            var loaded = loaderRig.LoadCheckpoint(path);

            Assert.Same(loaderMerge, loaded.Rig!.MergeContext);
            Assert.Same(loaderRuntime, loaded.Rig!.RuntimeContext);
            Assert.NotSame(merge, loaded.Rig!.MergeContext);
            Assert.NotSame(runtime, loaded.Rig!.RuntimeContext);
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }
}

[Trait("Domain", "Training")]
[Trait("Purpose", "Coverage")]
public class TrainingRigScheduleCoverageTests
{
    private static ComputationGraph SchedulerModule(Func<Scalar<int64>, Scalar<float32>> body)
    {
        var step = InputScalar<int64>("step");
        var value = body(step);
        return new ComputationGraph(new InternalComputationGraph([step], [value]), GraphKind.Module);
    }

    private static ComputationGraph SchedulerModuleRaw(Variable[] inputs, Variable[] outputs)
        => new(new InternalComputationGraph([.. inputs], [.. outputs]), GraphKind.Module);

    [Fact]
    public void TestWarmupRampSpansStartFactorToPeakOverExactlyWarmupSteps()
    {
        static void Ramp(float peak, int warmup, float startFactor)
        {
            var s = Schedules.Constant(peak).WithWarmup(warmup, startFactor);
            Assert.Equal(startFactor * peak, s.At(0));
            Assert.Equal(peak, s.At(warmup));
            Assert.Equal(peak, s.At(warmup + 1));
            for (long k = 0; k < warmup; k++)
                Assert.Equal(peak * (startFactor + (1f - startFactor) * ((float)k / warmup)), s.At(k));
        }

        Ramp(5e-4f, 800, 0f);
        Ramp(5e-4f, 1600, 0f);
        Ramp(1f, 1, 0f);
        Ramp(3e-4f, 100, 0.25f);
        Ramp(0.05f, 7, 0.5f);

        var decaying = Schedules.Cosine(2e-3f, 500).WithWarmup(50);
        Assert.Equal(0f, decaying.At(0));
        Assert.Equal(2e-3f, decaying.At(50));
        Assert.True(decaying.At(49) < decaying.At(50));
        Assert.True(decaying.At(51) < decaying.At(50));

        var recipe = Schedules.Constant(5e-4f).WithWarmup(800)
            .Then(15600, Schedules.Linear(5e-4f, 0.05f * 5e-4f, 24000 - 15600));
        Assert.Equal(0f, recipe.At(0));
        Assert.Equal(5e-4f, recipe.At(800));
        Assert.Equal(5e-4f, recipe.At(15599));
        Assert.True(MathF.Abs(recipe.At(24000) - 0.05f * 5e-4f) < 1e-9f);

        var documented = Schedules.Constant(1e-3f).WithWarmup(200)
            .Then(3900, Schedules.Linear(1e-3f, 5e-5f, 2100));
        Assert.Equal(0f, documented.At(0));
        Assert.Equal(5.0e-4f, documented.At(100));
        Assert.Equal(9.95e-4f, documented.At(199));
        Assert.Equal(1e-3f, documented.At(200));
        Assert.Equal(1e-3f, documented.At(3899));
    }

    [Fact]
    public void TestScheduleCombinatorsCoverage()
    {
        static void Eq(float expected, float actual) => Assert.True(MathF.Abs(expected - actual) < 1e-4f);

        Eq(0.5f, Schedules.Constant(0.5f).At(123));
        Eq(1.0f, Schedules.Linear(1.0f, 0.0f, 10).At(0));
        Eq(0.5f, Schedules.Linear(1.0f, 0.0f, 10).At(5));
        Eq(0.0f, Schedules.Linear(1.0f, 0.0f, 10).At(10));
        Eq(1.0f, Schedules.Cosine(1.0f, 8).At(0));
        Eq(0.0f, Schedules.Cosine(1.0f, 8).At(8));
        Eq(0.25f, Schedules.StepDecay(1.0f, 2, 0.5f).At(4));
        Eq(0.25f, Schedules.Exponential(1.0f, 0.5f).At(2));

        var cw = Schedules.CosineWithWarmup(1.0f, warmupSteps: 4, totalSteps: 12);
        Eq(0.0f, cw.At(0));
        Eq(0.5f, cw.At(2));
        Eq(1.0f, cw.At(4));
        Assert.True(cw.At(11) < 0.05f);

        var composed = Schedules.Cosine(1.0f, 8).WithWarmup(4);
        Eq(cw.At(0), composed.At(0));
        Eq(cw.At(7), composed.At(7));

        Eq(2.0f, Schedules.Constant(1.0f).Scale(2.0f).At(0));
        Eq(1.0f, Schedules.Linear(0f, 5f, 5).Clamp(0f, 1f).At(4));
        Eq(Schedules.Linear(0f, 5f, 5).At(3), Schedules.Linear(0f, 5f, 5).Shift(1).At(2));
        var perEpoch = Schedules.Linear(0f, 4f, 4).PerEpoch(stepsPerEpoch: 3);
        Eq(perEpoch.At(0), perEpoch.At(2));
        Assert.True(MathF.Abs(perEpoch.At(2) - perEpoch.At(3)) > 1e-6f);
        var joined = Schedules.Constant(1.0f).Then(atStep: 3, Schedules.Constant(2.0f));
        Eq(1.0f, joined.At(2));
        Eq(2.0f, joined.At(3));

        var oc = Schedules.OneCycle(maxValue: 1.0f, totalSteps: 100, pctStart: 0.3f, divFactor: 25f);
        Eq(1.0f / 25f, oc.At(0));
        Assert.True(oc.At(30) > oc.At(0));
        Assert.True(oc.At(99) < oc.At(0));
    }

    [Fact]
    public void TestRuntimeScheduleAndSchedulerModuleHyperparametersDriveTrainingCoverage()
    {
        var (sample, inputBatch, targetBatch) = ScalarMultiplyBatches();

        var runtimeRig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            sample, new SGDOptimizerHyperparameters { LearningRate = Hyperparameter.Runtime() });

        Assert.Single(runtimeRig.HyperparameterStructDef.Fields);
        Assert.Equal((int[])[0], runtimeRig.DynamicHyperparameterIndices);
        Assert.Equal((string[])["learningRate"], runtimeRig.DynamicHyperparameterNames);
        Assert.Equal("learningRate", runtimeRig.HyperparameterStructDef.Fields[0].Name);

        var initial = runtimeRig.CreateInitialCheckpoint().Shared();
        Assert.Equal(0, initial.Step);
        float w0 = Weight(runtimeRig, initial);

        var stepA = runtimeRig.TrainStep(initial, runtimeRig.MakeHyperparameters(0.1f), inputBatch.Shared(), targetBatch.Shared());
        var stepB = runtimeRig.TrainStep(initial, runtimeRig.MakeHyperparameters(("learningRate", 0.3f)), inputBatch.Shared(), targetBatch.Shared());
        Assert.Equal(1, stepA.Step);
        float deltaA = w0 - Weight(runtimeRig, stepA);
        float deltaB = w0 - Weight(runtimeRig, stepB);
        Assert.True(MathF.Abs(deltaA) > 1e-4f);
        Assert.True(MathF.Abs(stepA.Loss!.Value - stepB.Loss!.Value) < 1e-4f);
        Assert.True(MathF.Abs(deltaB - 3f * deltaA) < 1e-4f);

        Assert.Throws<InvalidOperationException>(() => runtimeRig.TrainStep(initial, inputBatch.Shared(), targetBatch.Shared()));
        Assert.Throws<ArgumentException>(() => runtimeRig.MakeHyperparameters(("bogus", 0.1f)));

        var schedRig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            sample, new SGDOptimizerHyperparameters { LearningRate = Schedules.Linear(0.2f, 0.0f, 4) });
        Assert.Empty(schedRig.HyperparameterStructDef.Fields);
        Assert.Empty(schedRig.DynamicHyperparameterIndices);
        float swAuto = Weight(schedRig, schedRig.TrainStep(schedRig.CreateInitialCheckpoint(), inputBatch.Shared(), targetBatch.Shared()));
        float swRef = Weight(runtimeRig, runtimeRig.TrainStep(
            runtimeRig.CreateInitialCheckpoint(), runtimeRig.MakeHyperparameters(0.2f), inputBatch.Shared(), targetBatch.Shared()));
        Assert.True(MathF.Abs(swAuto - swRef) < 1e-5f);

        var adamRig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, AdamWOptimizer.ComputationGraph,
            sample, new AdamWOptimizerHyperparameters { LearningRate = Schedules.Constant(0.01f) });
        Assert.Empty(adamRig.DynamicHyperparameterIndices);
        Assert.Empty(adamRig.HyperparameterStructDef.Fields);
        var adamStep = adamRig.TrainStep(adamRig.CreateInitialCheckpoint(), inputBatch.Shared(), targetBatch.Shared());
        Assert.True(float.IsFinite(adamStep.Loss!.Value));
        Assert.NotEmpty(adamStep.OptimizerState.Fields);

        float FinalWeight(Schedule lr)
        {
            var fitRig = TrainingRig.FromScratch(
                ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
                sample, new SGDOptimizerHyperparameters { LearningRate = lr });
            var result = fitRig.Fit(
                [inputBatch, inputBatch, inputBatch, inputBatch],
                [targetBatch, targetBatch, targetBatch, targetBatch],
                numEpochs: 1, fitRig.CreateInitialCheckpoint());
            Assert.Single(result.EpochLosses);
            return Weight(fitRig, result.FinalCheckpoint);
        }
        Assert.True(MathF.Abs(FinalWeight(Schedules.Linear(0.2f, 0.0f, 4)) - FinalWeight(Schedules.Constant(0.2f))) > 1e-4f);

        var momRig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, SGDMomentumOptimizer.ComputationGraph,
            sample, new SGDMomentumOptimizerHyperparameters
            {
                LearningRate = Hyperparameter.Runtime(),
                MomentumCoeff = Hyperparameter.Runtime(),
            });
        Assert.Equal((string[])["learningRate", "momentumCoeff"], momRig.DynamicHyperparameterNames.ToArray());
        var momStep = momRig.TrainStep(momRig.CreateInitialCheckpoint(),
            momRig.MakeHyperparameters(("momentumCoeff", 0.9f), ("learningRate", 0.1f)),
            inputBatch.Shared(), targetBatch.Shared());
        Assert.True(float.IsFinite(momStep.Loss!.Value));
        Assert.NotEmpty(momStep.OptimizerState.Fields);
        Assert.Throws<ArgumentException>(() => momRig.MakeHyperparameters(("learningRate", 0.1f)));

        var cosine = Schedules.Cosine(0.05f, 6).WithWarmup(2);
        var cosineRig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            sample, new SGDOptimizerHyperparameters { LearningRate = cosine });
        Assert.Empty(cosineRig.HyperparameterStructDef.Fields);
        var cosineCkpt = cosineRig.CreateInitialCheckpoint();
        var hostCkpt = runtimeRig.CreateInitialCheckpoint();
        for (int s = 0; s < 6; s++)
        {
            cosineCkpt = cosineRig.TrainStep(cosineCkpt, inputBatch.Shared(), targetBatch.Shared());
            hostCkpt = runtimeRig.TrainStep(hostCkpt, runtimeRig.MakeHyperparameters(cosine.At(s)),
                inputBatch.Shared(), targetBatch.Shared());
            Assert.True(MathF.Abs(Weight(cosineRig, cosineCkpt) - Weight(runtimeRig, hostCkpt)) < 1e-5f);
        }
        Assert.Equal(6, cosineCkpt.Step);

        var moduleRig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            sample, new SGDOptimizerHyperparameters
            {
                LearningRate = Hyperparameter.Scheduled(
                    SchedulerModule(step => Scalar(0.3f) - step.Cast<float32>() * Scalar(0.05f))),
            });
        Assert.Empty(moduleRig.HyperparameterStructDef.Fields);
        var moduleCkpt = moduleRig.CreateInitialCheckpoint();
        var moduleRefCkpt = runtimeRig.CreateInitialCheckpoint();
        for (int s = 0; s < 4; s++)
        {
            moduleCkpt = moduleRig.TrainStep(moduleCkpt, inputBatch.Shared(), targetBatch.Shared());
            moduleRefCkpt = runtimeRig.TrainStep(moduleRefCkpt, runtimeRig.MakeHyperparameters(0.3f - 0.05f * s),
                inputBatch.Shared(), targetBatch.Shared());
            Assert.True(MathF.Abs(Weight(moduleRig, moduleCkpt) - Weight(runtimeRig, moduleRefCkpt)) < 1e-5f);
        }
    }

    private static float FreshOptStateValue(TrainingRig rig, TrainingCheckpoint ckpt)
    {
        var field = rig.OptimizerStateDef.Fields[0].Name;
        return ((TensorData<float32>)ckpt.OptimizerState.Fields[field]).AccessMemory()[0];
    }

    [Fact]
    public void TestOptimizerStateInitAndSchedulerContractRejectionsCoverage()
    {
        var (sample, inputBatch, targetBatch) = ScalarMultiplyBatches();

        var stepCountingRig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            StepCountingSgdOptimizer.ComputationGraph, sample, 0.1f);
        var initial = stepCountingRig.CreateInitialCheckpoint();
        Assert.Single(stepCountingRig.OptimizerStateDef.Fields);
        Assert.All(FlattenStruct(initial.OptimizerState), v => Assert.Equal(1f, v));
        Assert.All(FlattenStruct(stepCountingRig.TrainStep(initial, inputBatch.Shared(), targetBatch.Shared()).OptimizerState),
            v => Assert.Equal(2f, v));

        var optEx = Assert.Throws<InvalidOperationException>(() => TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            ModuleOwnedStateOptimizer.ComputationGraph, sample, 0.1f));
        Assert.Contains("OptimizerOwned", optEx.Message);

        var modelEx = Assert.Throws<ArgumentException>(() => TrainingRig.FromScratch(
            OptimizerOwnedStateModel.ComputationGraph, L2Loss.ComputationGraph,
            SGDOptimizer.ComputationGraph, sample, 0.1f));
        Assert.Contains("ModuleOwned", modelEx.Message);

        TrainingRig HyperRig(Hyperparameter lr) => TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, InitFromHyperOptimizer.ComputationGraph,
            sample, new InitFromHyperOptimizerHyperparameters { LearningRate = lr });

        var dslRig = HyperRig(Schedules.Constant(0.5f));
        Assert.True(MathF.Abs(0.5f - FreshOptStateValue(dslRig, dslRig.CreateInitialCheckpoint())) < 1e-4f);

        var decayRig = HyperRig(Schedules.Linear(0.2f, 0f, 10));
        Assert.True(MathF.Abs(0.2f - FreshOptStateValue(decayRig, decayRig.CreateInitialCheckpoint())) < 1e-4f);

        var moduleRig = HyperRig(Hyperparameter.Scheduled(
            SchedulerModule(step => Scalar(0.7f) + step.Cast<float32>() * Scalar(0f))));
        float moduleState = FreshOptStateValue(moduleRig, moduleRig.CreateInitialCheckpoint());
        Assert.True(MathF.Abs(0.7f - moduleState) < 1e-4f);
        Assert.True(MathF.Abs(moduleState) > 1e-4f);

        var runtimeRig = HyperRig(Hyperparameter.Runtime());
        var ex = Assert.Throws<InvalidOperationException>(() => runtimeRig.CreateInitialCheckpoint());
        Assert.Contains("learningRate", ex.Message);
        Assert.True(MathF.Abs(0.3f - FreshOptStateValue(
            runtimeRig, runtimeRig.CreateInitialCheckpoint(runtimeRig.MakeHyperparameters(0.3f)))) < 1e-4f);

        var bakedRig = HyperRig(0.05f);
        Assert.True(MathF.Abs(0.05f - FreshOptStateValue(bakedRig, bakedRig.CreateInitialCheckpoint())) < 1e-4f);

        TrainingRig SgdSchedRig(Hyperparameter lr) => TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            sample, new SGDOptimizerHyperparameters { LearningRate = lr });

        var floatStep = InputScalar<float32>("step");
        Assert.Throws<ArgumentException>(
            () => SgdSchedRig(Hyperparameter.Scheduled(SchedulerModuleRaw([floatStep], [floatStep]))));

        var a = InputScalar<int64>("a");
        var b = InputScalar<int64>("b");
        Assert.Throws<ArgumentException>(() => SgdSchedRig(Hyperparameter.Scheduled(
            SchedulerModuleRaw([a, b], [a.Cast<float32>() + b.Cast<float32>()]))));

        var intStep = InputScalar<int64>("step");
        Assert.Throws<ArgumentException>(
            () => SgdSchedRig(Hyperparameter.Scheduled(SchedulerModuleRaw([intStep], [intStep]))));

        Assert.Throws<ArgumentException>(
            () => SgdSchedRig(Hyperparameter.Scheduled(new Schedule((ScheduleExpr?)null))));

        foreach (var impure in (ComputationGraph[])[ParamScheduler.ComputationGraph, StateScheduler.ComputationGraph, RngScheduler.ComputationGraph])
        {
            var impureEx = Assert.Throws<ArgumentException>(
                () => SgdSchedRig(Hyperparameter.Scheduled(impure)));
            Assert.Contains("pure", impureEx.Message);
        }
    }
}

[Trait("Domain", "Training")]
[Trait("Purpose", "Coverage")]
public class TrainingRigTrainingLoopCoverageTests
{
    private static float StateAfterOneStep(ComputationGraph modelGraph)
    {
        var x = TensorData([2L], 1f, 2f);
        var rig = TrainingRig.FromScratch(modelGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            [new TensorDataModelParam("input", ModelParamType.InputParam, x)], 0.1f);
        var step = rig.TrainStep(rig.CreateInitialCheckpoint(),
            NNLibraryTrainingFixtures.MakeBatch("input", "ModelInput", x),
            NNLibraryTrainingFixtures.MakeBatch("targets", "Target", TensorData([2L], 0f, 0f)));
        return NNLibraryTrainingFixtures.Floats(step.ModelState.Fields[rig.ModelStateDef.Fields.Single().Name])[0];
    }

    private static float LossAfterOneStep(ComputationGraph modelGraph)
    {
        var x = TensorData([2L], 1f, 2f);
        var rig = TrainingRig.FromScratch(modelGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            [new TensorDataModelParam("input", ModelParamType.InputParam, x)], 0.1f);
        return rig.TrainStep(rig.CreateInitialCheckpoint(),
            NNLibraryTrainingFixtures.MakeBatch("input", "ModelInput", x),
            NNLibraryTrainingFixtures.MakeBatch("targets", "Target", TensorData([2L], 0f, 0f))).Loss!.Value;
    }

    // A read placed after Globals.StateUpdate must still see the value fed in for this step: the
    // link is a graph-level registration, not an assignment. Guards the semantics an attempt at
    // Shorokoo/Shorokoo#306 broke while the whole suite stayed green.
    [Fact]
    public void TestAStateReadAfterItsUpdateStillSeesTheValueFedInForThisStep()
        => Assert.Equal(2.5f, LossAfterOneStep(StatefulGainNoRefModel.ComputationGraph), 1e-4f);

    // Both calls update, in call order, so the second sees the first's result.
    [Fact]
    public void TestAStatefulModelCalledTwiceAppliesBothItsStateUpdates()
        => Assert.Equal(2f * StateAfterOneStep(StatefulGainNoRefModel.ComputationGraph),
                        StateAfterOneStep(StatefulGainCalledTwiceModel.ComputationGraph));

    private static float[] StateFieldsAfterOneStep(ComputationGraph modelGraph)
    {
        var x = TensorData([2L], 1f, 2f);
        var rig = TrainingRig.FromScratch(modelGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            [new TensorDataModelParam("input", ModelParamType.InputParam, x)], 0.1f);
        var step = rig.TrainStep(rig.CreateInitialCheckpoint(),
            NNLibraryTrainingFixtures.MakeBatch("input", "ModelInput", x),
            NNLibraryTrainingFixtures.MakeBatch("targets", "Target", TensorData([2L], 0f, 0f)));
        float[] values = [.. rig.ModelStateDef.Fields.Select(f =>
            NNLibraryTrainingFixtures.Floats(step.ModelState.Fields[f.Name])[0])];
        Array.Sort(values);
        return values;
    }

    // A call closes as many state scopes as the modules nested at it own, and one scope can close
    // several fields at once; each field's own sequence of calls has to compose separately.
    [Fact]
    public void TestNestedAndMultiFieldStateBothComposeAcrossTwoCalls()
    {
        Assert.Equal([2f, 20f], StateFieldsAfterOneStep(NestedStatefulCalledTwiceModel.ComputationGraph));
        Assert.Equal([2f, 200f], StateFieldsAfterOneStep(TwoStateFieldsCalledTwiceModel.ComputationGraph));
    }

    // A loop body is one call site however many trips it runs, so there is nothing to chain and the
    // in-loop update still registers once for the step.
    [Fact]
    public void TestAStatefulModelCalledOnceInALoopKeepsItsSingleUpdate()
        => Assert.Equal([1f], StateFieldsAfterOneStep(StatefulCalledOnceInALoopModel.ComputationGraph));

    // Calling for the state update alone is what module-owned state is for, so the call must reach
    // the graph through more than its output.
    [Fact]
    public void TestAStatefulCallWhoseOutputIsDiscardedStillUpdatesItsState()
        => Assert.Equal([2f], StateFieldsAfterOneStep(StatefulCallDiscardedModel.ComputationGraph));

    /// <summary>The ops an inference model computes inside its <c>If</c>, rather than before it.</summary>
    private static string[] IfBodyOps(ComputationGraph modelGraph)
    {
        var f = modelGraph.ToConcreteArchitecture(
            modelGraph.FromOrderedInputs([TensorData([2L], 1f, 2f)])).ToConcreteModel().ToInternal();
        int open = f.Nodes.FindIndex(n => n.OpCode == OpCodes.IF_OPEN);
        int close = f.Nodes.FindIndex(n => n.OpCode == OpCodes.IF_CLOSE);
        return [.. f.Nodes.GetRange(open + 1, close - open - 1).Select(n => n.OpCode)];
    }

    // A backward pass reads the forward's intermediates, so a branch that computes one cannot keep
    // it to itself; the rest of the branch stays inside it, and inference keeps all of it. Both
    // shapes train: each arm owning its parameter, and both sharing one.
    [Fact]
    public void TestAParameterSharedByBothIfElseArmsTrains()
    {
        Assert.Equal(2.5f, LossAfterOneStep(GainInBothIfArmsOnARuntimeConditionModel.ComputationGraph), 1e-4f);
        Assert.Equal(2.5f, LossAfterOneStep(SharedGainInBothIfArmsModel.ComputationGraph), 1e-4f);
        Assert.NotEmpty(IfBodyOps(SharedGainInBothIfArmsModel.ComputationGraph));
    }

    private static float[] TrainedParams(ComputationGraph modelGraph, bool cond, params float[] xs)
    {
        object[] values = [.. xs.Select(v => (object)v)];
        var x = TensorData(DType.Float32, [(long)xs.Length], values);
        var c = TensorData(DType.Bool, [], cond);
        var rig = TrainingRig.FromScratch(modelGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            [new TensorDataModelParam("t", ModelParamType.InputParam, x),
             new TensorDataModelParam("cond", ModelParamType.InputParam, c)], 0.1f);
        var step = rig.TrainStep(rig.CreateInitialCheckpoint(), rig.InputDef.FromOrderedData(x, c),
            rig.TargetDef.FromOrderedData(TensorData([(long)xs.Length], new float[xs.Length])));
        return NNLibraryTrainingFixtures.Floats(
            step.TrainableParams.Fields[rig.TrainableParamStructDef.Fields[0].Name]);
    }

    // The arm that did not run contributes exactly zero, whether or not its own derivative is a
    // number: the same input trains to the same weights with the other arm finite and non-finite.
    [Fact]
    public void TestTheIfElseArmThatDidNotRunLeavesTheGradientAlone()
    {
        Assert.Equal<float>([0.9f, -0.6f], TrainedParams(SqrtInOneIfArmModel.ComputationGraph, false, -1f, -4f));
        Assert.Equal<float>([0.9f, -0.6f], TrainedParams(SqrtInOneIfArmModel.ComputationGraph, false, 1f, 4f));
        Assert.Equal<float>([0.95f, 0.8f], TrainedParams(SqrtInOneIfArmModel.ComputationGraph, true, 1f, 4f));
    }

    private static TrainingRig OptionalBiasRig(OptionalTensorData bias, TensorData x)
        => TrainingRig.FromScratch(NullableTrainableBiasLayer.ComputationGraph,
            L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            [new TensorDataModelParam("x", ModelParamType.InputParam, x),
             new OptionalTensorDataModelParam("bias", ModelParamType.InputParam, bias)], 0.1f);

    // An OptionalTensor input is a supported model input; a rig builds over one whether it is
    // supplied present or absent, and the present arrangement takes a step. The absent one is not
    // stepped here: ONNX Runtime has no absent optional to feed, the same limit inference carries.
    [Fact]
    public void TestAModelWithAnOptionalInputTrains()
    {
        var x = TensorData([3L], 1f, 2f, 3f);
        var present = OptionalTensorData.Some(TensorData([3L], 1f, 1f, 1f));
        Assert.NotEmpty(OptionalBiasRig(OptionalTensorData.None<float32>(), x).TrainableParamStructDef.Fields);

        var rig = OptionalBiasRig(present, x);
        Assert.NotEmpty(rig.TrainableParamStructDef.Fields);
        Assert.Equal(29f / 3f, StepLoss(rig, x, present), 1e-3f);

        // A plain tensor for the optional field is the present arm, and is what a caller holding the
        // tensor writes; the runtime takes one where an optional is expected.
        Assert.Equal(29f / 3f, rig.TrainStep(rig.CreateInitialCheckpoint(),
            rig.InputDef.FromOrderedData(x, TensorData([3L], 1f, 1f, 1f)),
            rig.TargetDef.FromOrderedData(TensorData([3L], 0f, 0f, 0f))).Loss!.Value, 1e-3f);
    }

    private static float StepLoss(TrainingRig rig, TensorData x, OptionalTensorData bias)
        => rig.TrainStep(rig.CreateInitialCheckpoint(),
            rig.InputDef.FromOrderedData(x, bias).Shared(),
            rig.TargetDef.FromOrderedData(TensorData([3L], 0f, 0f, 0f))).Loss!.Value;

    // Two reads of one optional input give the backward pass two gradients to accumulate into a
    // single optional-structured slot.
    [Fact]
    public void TestAModelReadingItsOptionalInputTwiceTrains()
    {
        var x = TensorData([3L], 1f, 2f, 3f);
        var bias = OptionalTensorData.Some(TensorData([3L], 1f, 1f, 1f));
        var rig = TrainingRig.FromScratch(NullableBiasReadTwiceLayer.ComputationGraph,
            L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            [new TensorDataModelParam("x", ModelParamType.InputParam, x),
             new OptionalTensorDataModelParam("bias", ModelParamType.InputParam, bias)], 0.1f);
        Assert.True(float.IsFinite(StepLoss(rig, x, bias)));
    }

    // Training differentiates a loop by unrolling it, so one whose trip count is not a constant
    // has no backward pass. A constant count is unrolled and trains, which is why every other
    // in-loop training test passes.
    [Fact]
    public void TestATrainableParameterInsideARolledLoopIsRefused()
    {
        Assert.Equal(2.5f, LossAfterOneStep(GainInConstantTripLoopModel.ComputationGraph), 1e-4f);
        Assert.Contains("unroll it first", Assert.Throws<AutoDiffNotSupportedException>(
            () => LossAfterOneStep(GainInRolledLoopModel.ComputationGraph)).Message);
        Assert.Contains("unroll it first", Assert.Throws<AutoDiffNotSupportedException>(
            () => LossAfterOneStep(StatefulGainInRolledLoopModel.ComputationGraph)).Message);
    }

    [Fact]
    public void TestTrainStepAndTrainLoopCoverage()
    {
        var rig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph,
            L2Loss.ComputationGraph,
            SGDOptimizer.ComputationGraph,
            [
                new TensorDataModelParam("input", ModelParamType.InputParam,
                    TensorData([4L], [1f, 2f, 3f, 4f])),
            ],
            0.1f);

        var initial = rig.CreateInitialCheckpoint();
        var inputBatch = InBatch(1f, 2f, 3f, 4f);
        var targetBatch = TargetBatch(0f, 0f, 0f, 0f);

        var trainResult = rig.Train(initial.Shared(), [inputBatch], [targetBatch], numEpochs: 1);
        Assert.Single(trainResult.EpochLosses);
        Assert.NotNull(trainResult.FinalCheckpoint);

        var stepResult = rig.TrainStep(initial, inputBatch.Shared(), targetBatch.Shared());
        Assert.NotNull(stepResult);
        Assert.NotNull(stepResult.TrainableParams);
        Assert.NotNull(stepResult.ModelState);
        Assert.NotNull(stepResult.OptimizerState);
        Assert.True(float.IsFinite(stepResult.Loss!.Value));

        var graphs = new FastTrainingGraphs(
            ScalarMultiplyModel.ComputationGraph.ToInternal(),
            L2Loss.ComputationGraph.ToInternal(),
            SGDOptimizer.ComputationGraph.ToInternal());
        Assert.NotNull(graphs.ModelGraph);
        Assert.NotNull(graphs.LossGraph);
        Assert.NotNull(graphs.OptimizerGraph);
    }

    [Fact]
    public void TestTrainStepSessionCarriesConcreteDimsFoldsShapeArithmeticAndRecompilesPerShapeCoverage()
    {
        var (sample, input, target) = ScalarMultiplyBatches();
        var rig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph, sample, 0.1f);
        var ckpt = rig.CreateInitialCheckpoint();
        var graph = rig.TrainingStepPureGraph.ToInternal();

        var fed = ComputeContext.ExpandStructInputs([ckpt.TrainableParams, ckpt.ModelState, ckpt.OptimizerState, input, target]);
        long[]?[] dims = fed.Select(d => ((TensorData)d).Shape.Dims).ToArray()!;
        var concrete = FastOnnxModelBuilder.BuildInternalOnnxModel(graph, prepForOnnx: true, inputDims: dims);
        var rankOnly = FastOnnxModelBuilder.BuildInternalOnnxModel(graph, prepForOnnx: true);

        Assert.Equal(dims, concrete.Graph.Inputs.Select(vi => vi.Type.TensorType.Shape.Dims.Select(d => d.DimValue).ToArray()));
        Assert.All(rankOnly.Graph.Inputs, vi => Assert.Null(vi.Type.TensorType.Shape));
        Assert.Equal(0, OrtOptimizedNodeCount(concrete, "Shape"));
        Assert.NotEqual(0, OrtOptimizedNodeCount(rankOnly, "Shape"));

        var generic = ComputeContext.Default.Compile(rig.TrainingStepPureGraph.ToInternal(), inputDims: null, trainingStep: true);
        float GenericLoss(TensorDataStruct i, TensorDataStruct t) =>
            generic.Execute(ComputeContext.ExpandStructInputs(
                [ckpt.TrainableParams.Shared(), ckpt.ModelState.Shared(), ckpt.OptimizerState.Shared(), i.Shared(), t.Shared()]))[^1]
                .ToTensorData<float32>().AccessMemory()[0];
        var halfInput = rig.InputDef.FromOrderedData(TensorData([2L], [1f, 2f]));
        var halfTarget = rig.TargetDef.FromOrderedData(TensorData([2L], [0f, 0f]));
        Assert.Equal(GenericLoss(input, target), rig.TrainStep(ckpt.Shared(), input.Shared(), target.Shared()).Loss);
        Assert.Equal(GenericLoss(halfInput, halfTarget), rig.TrainStep(ckpt.Shared(), halfInput.Shared(), halfTarget.Shared()).Loss);
        Assert.Equal(GenericLoss(input, target), rig.TrainStep(ckpt.Shared(), input.Shared(), target.Shared()).Loss);
        Assert.Equal(2, rig.CompiledTrainStepShapeKeys.Count);
        Assert.Equal(2, rig.CompiledTrainStepShapeKeys.Distinct().Count());
        Assert.False(rig.HasGenericTrainStepSession);

        foreach (var n in (int[])[1, 3, 5])
        {
            var i = rig.InputDef.FromOrderedData(TensorData([(long)n], Enumerable.Range(1, n).Select(v => (float)v).ToArray()));
            var t = rig.TargetDef.FromOrderedData(TensorData([(long)n], new float[n]));
            Assert.Equal(GenericLoss(i, t), rig.TrainStep(ckpt.Shared(), i, t).Loss);
        }
        Assert.Equal(TrainingRig.MaxShapeSpecializedTrainSteps, rig.CompiledTrainStepShapeKeys.Count);
        Assert.True(rig.HasGenericTrainStepSession);
    }

    private static int OrtOptimizedNodeCount(ModelProto model, string opType)
    {
        var bytes = new MemoryStream();
        ProtoBuf.Serializer.Serialize(bytes, model);
        var optimizedPath = Path.Combine(Path.GetTempPath(), $"shrk-opt-{Guid.NewGuid():N}.onnx");
        using var options = new SessionOptions();
        options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL;
        options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        options.OptimizedModelFilePath = optimizedPath;
        using (new InferenceSession(bytes.ToArray(), options)) { }
        ModelProto optimized;
        using (var fs = File.OpenRead(optimizedPath))
            optimized = ProtoBuf.Serializer.Deserialize<ModelProto>(fs);
        File.Delete(optimizedPath);
        return optimized.Graph.Nodes.Count(n => n.OpType == opType);
    }

    [Fact]
    public void TestTrainStepAndCheckpointCounterSemanticsCoverage()
    {
        var (sample, inputBatch, targetBatch) = ScalarMultiplyBatches();

        var (adamRig, trained, adamIn, adamOut) = BuildTrainedAdamWRig(steps: 1);
        var carried = adamRig.TrainStep(
            new TrainingCheckpoint
            {
                TrainableParams = trained.TrainableParams,
                ModelState = trained.ModelState,
                OptimizerState = trained.OptimizerState,
                Step = trained.Step, Epoch = 3, BatchIndex = 12,
            },
            adamIn, adamOut);
        Assert.Equal(trained.Step + 1, carried.Step);
        Assert.Equal(3, carried.Epoch);
        Assert.Equal(12, carried.BatchIndex);

        var schedRig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            sample, new SGDOptimizerHyperparameters { LearningRate = Hyperparameter.Scheduled(StepEpochScheduler.ComputationGraph) });
        var refRig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            sample, new SGDOptimizerHyperparameters { LearningRate = Hyperparameter.Runtime() });
        float Lr(long s, long e) => 0.5f - 0.01f * s - 0.1f * e;

        var seed = schedRig.CreateInitialCheckpoint();
        var refSeed = refRig.CreateInitialCheckpoint();
        foreach (var (s, e) in ((long, long)[])[(0L, 0L), (3L, 1L), (7L, 4L)])
        {
            var modStep = schedRig.TrainStep(
                new TrainingCheckpoint
                {
                    TrainableParams = seed.TrainableParams,
                    ModelState = seed.ModelState,
                    OptimizerState = seed.OptimizerState,
                    Step = s, Epoch = e, FeedMode = SharedInputMode.Shared,
                },
                inputBatch.Shared(), targetBatch.Shared());
            var refStep = refRig.TrainStep(
                new TrainingCheckpoint
                {
                    TrainableParams = refSeed.TrainableParams,
                    ModelState = refSeed.ModelState,
                    OptimizerState = refSeed.OptimizerState,
                    Step = s, Epoch = e, FeedMode = SharedInputMode.Shared,
                },
                refRig.MakeHyperparameters(Lr(s, e)), inputBatch.Shared(), targetBatch.Shared());
            Assert.True(MathF.Abs(Weight(schedRig, modStep) - Weight(refRig, refStep)) < 1e-5f);
        }

        var stepped = schedRig.TrainStep(seed.Shared(), inputBatch.Shared(), targetBatch.Shared(), epoch: 4, batchNumber: 7);
        Assert.Equal(seed.Step + 1, stepped.Step);
        Assert.Equal(4, stepped.Epoch);
        Assert.Equal(7, stepped.BatchIndex);
        Assert.True(float.IsFinite(stepped.Loss!.Value));
        Assert.Same(schedRig, stepped.Rig);
        var explicitRef = schedRig.TrainStep(
            new TrainingCheckpoint
            {
                TrainableParams = seed.TrainableParams,
                ModelState = seed.ModelState,
                OptimizerState = seed.OptimizerState,
                Step = 0, Epoch = 4, FeedMode = SharedInputMode.Shared,
            },
            inputBatch.Shared(), targetBatch.Shared());
        Assert.True(MathF.Abs(Weight(schedRig, stepped) - Weight(schedRig, explicitRef)) < 1e-6f);

        Assert.Throws<ArgumentOutOfRangeException>(
            () => schedRig.TrainStep(seed.Shared(), inputBatch.Shared(), targetBatch.Shared(), epoch: -1, batchNumber: 0));
        Assert.Throws<ArgumentOutOfRangeException>(
            () => schedRig.TrainStep(seed.Shared(), inputBatch.Shared(), targetBatch.Shared(), epoch: 0, batchNumber: -1));

        var ck = refSeed;
        Assert.Equal(0, ck.Step);
        var moved = ck.WithCounters(step: 5, epoch: 2, batchIndex: 3);
        Assert.NotSame(ck, moved);
        Assert.Equal(5, moved.Step);
        Assert.Equal(2, moved.Epoch);
        Assert.Equal(3, moved.BatchIndex);
        Assert.Same(ck.TrainableParams, moved.TrainableParams);
        Assert.Same(ck.ModelState, moved.ModelState);
        Assert.Same(ck.OptimizerState, moved.OptimizerState);
        Assert.Equal(0, ck.Step);
        Assert.Null(ck.Epoch);
        Assert.Null(ck.BatchIndex);
        Assert.Equal(9, ck.WithStep(9).Step);
        Assert.Null(ck.WithStep(9).Epoch);
        Assert.Equal(7, ck.WithEpoch(7).Epoch);
        Assert.Equal(0, ck.WithEpoch(7).Step);
        Assert.Equal(4, ck.WithBatchIndex(4).BatchIndex);
        Assert.Equal(0, ck.WithBatchIndex(4).Step);
    }

    private static int[] EpochIndexSequence(InMemoryDataLoader loader, int features)
    {
        var seq = new List<int>();
        for (int b = 0; b < loader.BatchesPerEpoch; b++)
        {
            var batch = loader.Next();
            var vals = ((TensorData)((TensorDataStruct)batch.Input).Fields["input"]).As<float32>().AccessMemory().ToArray();
            for (int r = 0; r < vals.Length; r += features)
                seq.Add((int)vals[r]);
        }
        return seq.ToArray();
    }

    private static int[] TailIndices(InMemoryDataLoader loader, int features)
    {
        var seq = new List<int>();
        long remaining = loader.BatchesPerEpoch - loader.Position.BatchIndex;
        for (long b = 0; b < remaining; b++)
        {
            var batch = loader.Next();
            var vals = ((TensorData)((TensorDataStruct)batch.Input).Fields["input"]).As<float32>().AccessMemory().ToArray();
            for (int r = 0; r < vals.Length; r += features)
                seq.Add((int)vals[r]);
        }
        return seq.ToArray();
    }

    [Fact]
    public void TestInMemoryDataLoaderBatchingAndShuffleCoverage()
    {
        const int features = 1;
        var rig = LoaderRig(batchSize: 2, features);
        var (inputs, targets) = IndexDataset(rig, n: 8, features);

        var plain = new InMemoryDataLoader(inputs, targets, batchSize: 2);
        Assert.Equal(4, plain.BatchesPerEpoch);
        Assert.Equal(8, plain.SampleCount);
        Assert.Equal(new DataLoaderPosition(0, 0), plain.Position);
        int[] identity = [0, 1, 2, 3, 4, 5, 6, 7];
        Assert.Equal(identity, EpochIndexSequence(plain, features));
        Assert.Equal(new DataLoaderPosition(1, 0), plain.Position);

        var (inputs9, targets9) = IndexDataset(rig, n: 9, features);
        var keepPartial = new InMemoryDataLoader(inputs9, targets9, batchSize: 2, dropLast: false);
        Assert.Equal(5, keepPartial.BatchesPerEpoch);
        var partialSeq = EpochIndexSequence(keepPartial, features);
        Assert.Equal(9, partialSeq.Length);
        Assert.Equal(Enumerable.Range(0, 9), partialSeq);

        var stepper = new InMemoryDataLoader(inputs, targets, batchSize: 2);
        stepper.Next();
        Assert.Equal(new DataLoaderPosition(0, 1), stepper.Position);
        stepper.Next();
        Assert.Equal(new DataLoaderPosition(0, 2), stepper.Position);

        var s1 = new InMemoryDataLoader(inputs, targets, batchSize: 2, shuffle: true, seed: 12345);
        var s2 = new InMemoryDataLoader(inputs, targets, batchSize: 2, shuffle: true, seed: 12345);
        int[] order1 = EpochIndexSequence(s1, features);
        int[] order2 = EpochIndexSequence(s2, features);
        Assert.Equal(order1, order2);
        Assert.Equal(identity, order1.OrderBy(i => i));
        Assert.NotEqual(identity, order1);

        int[] epoch1Order = EpochIndexSequence(s1, features);
        Assert.Equal(identity, epoch1Order.OrderBy(i => i));
        Assert.NotEqual(order1, epoch1Order);

        var s3 = new InMemoryDataLoader(inputs, targets, batchSize: 2, shuffle: true, seed: 12345);
        s3.RestoreFrom(new DataLoaderPosition(0, 0));
        Assert.Equal(order1, EpochIndexSequence(s3, features));

        Assert.Throws<ArgumentOutOfRangeException>(() => plain.RestoreFrom(new DataLoaderPosition(0, 4)));
        Assert.Throws<ArgumentOutOfRangeException>(() => plain.RestoreAfter(new DataLoaderPosition(0, 4)));

        var afterStepper = new InMemoryDataLoader(inputs, targets, batchSize: 2);
        afterStepper.RestoreAfter(new DataLoaderPosition(0, 1));
        Assert.Equal(new DataLoaderPosition(0, 2), afterStepper.Position);
        afterStepper.RestoreAfter(new DataLoaderPosition(0, 3));
        Assert.Equal(new DataLoaderPosition(1, 0), afterStepper.Position);
    }

    [Fact]
    public void TestLoaderDrivenFitAndTrainStepAdvanceCountersCoverage()
    {
        const int features = 4;
        var rig = LoaderRig(batchSize: 2, features);
        var (inputs, targets) = IndexDataset(rig, n: 6, features);

        var fitLoader = new InMemoryDataLoader(inputs, targets, batchSize: 2);
        Assert.Equal(3, fitLoader.BatchesPerEpoch);
        var result = rig.Fit(fitLoader, numEpochs: 2);
        Assert.Equal(2, result.EpochLosses.Length);
        Assert.All(result.EpochLosses, l => Assert.True(float.IsFinite(l)));
        var final = result.FinalCheckpoint;
        Assert.Equal(6, final.Step);
        Assert.Equal(1, final.Epoch);
        Assert.Equal(2, final.BatchIndex);
        Assert.Equal(final.Epoch * fitLoader.BatchesPerEpoch + final.BatchIndex + 1, final.Step);
        Assert.Equal(new DataLoaderPosition(2, 0), fitLoader.Position);

        var stepLoader = new InMemoryDataLoader(inputs, targets, batchSize: 2);
        var s1 = rig.TrainStep(rig.CreateInitialCheckpoint(), stepLoader);
        Assert.Equal(1, s1.Step);
        Assert.Equal(0, s1.Epoch);
        Assert.Equal(0, s1.BatchIndex);
        Assert.True(float.IsFinite(s1.Loss!.Value));
        Assert.Same(rig, s1.Rig);
        Assert.Equal(new DataLoaderPosition(0, 1), stepLoader.Position);

        var s2 = rig.TrainStep(s1, stepLoader);
        Assert.Equal(2, s2.Step);
        Assert.Equal(0, s2.Epoch);
        Assert.Equal(1, s2.BatchIndex);

        var s3 = rig.TrainStep(s2, stepLoader);
        Assert.Equal(3, s3.Step);
        Assert.Equal(0, s3.Epoch);
        Assert.Equal(2, s3.BatchIndex);
        Assert.Equal(new DataLoaderPosition(1, 0), stepLoader.Position);

        var oneEpoch = rig.Fit(new InMemoryDataLoader(inputs, targets, batchSize: 2), numEpochs: 1);
        Assert.Equal(oneEpoch.FinalCheckpoint.Step, s3.Step);
        Assert.Equal(oneEpoch.FinalCheckpoint.Epoch, s3.Epoch);
        Assert.Equal(oneEpoch.FinalCheckpoint.BatchIndex, s3.BatchIndex);
        Assert.Equal(FlattenStruct(oneEpoch.FinalCheckpoint.TrainableParams), FlattenStruct(s3.TrainableParams));
    }

    [Fact]
    public void TestDataLoaderResumeRoundTripCoverage()
    {
        const int features = 4;
        const long seed = 777;

        var rigRef = LoaderRig(batchSize: 2, features);
        var (inRef, tgtRef) = IndexDataset(rigRef, n: 6, features);
        var loaderRef = new InMemoryDataLoader(inRef, tgtRef, batchSize: 2, shuffle: true, seed: seed);
        var refResult = rigRef.Fit(loaderRef, numEpochs: 2);
        float[] refWeights = FlattenStruct(refResult.FinalCheckpoint.TrainableParams);

        var path = TempPath("loader_resume") + ".safetensors";
        try
        {
            var rigA = LoaderRig(batchSize: 2, features);
            var (inA, tgtA) = IndexDataset(rigA, n: 6, features);
            var loaderA = new InMemoryDataLoader(inA, tgtA, batchSize: 2, shuffle: true, seed: seed);
            var half = rigA.Fit(loaderA, numEpochs: 1);
            Assert.Equal(3, half.FinalCheckpoint.Step);
            Assert.Equal(0, half.FinalCheckpoint.Epoch);
            Assert.Equal(2, half.FinalCheckpoint.BatchIndex);
            half.FinalCheckpoint.Save(path);

            var rigB = LoaderRig(batchSize: 2, features);
            var (inB, tgtB) = IndexDataset(rigB, n: 6, features);
            var loaderB = new InMemoryDataLoader(inB, tgtB, batchSize: 2, shuffle: true, seed: seed);
            var loaded = rigB.LoadCheckpoint(path);
            Assert.Equal(0, loaded.Epoch);
            Assert.Equal(2, loaded.BatchIndex);
            var resumed = rigB.Fit(loaderB, numEpochs: 1, loaded);

            Assert.Equal(refResult.FinalCheckpoint.Step, resumed.FinalCheckpoint.Step);
            Assert.Equal(refResult.FinalCheckpoint.Epoch, resumed.FinalCheckpoint.Epoch);
            Assert.Equal(refWeights, FlattenStruct(resumed.FinalCheckpoint.TrainableParams));
        }
        finally { if (File.Exists(path)) File.Delete(path); }

        var rigM = LoaderRig(batchSize: 2, features: 1);
        var (inM, tgtM) = IndexDataset(rigM, n: 8, features: 1);

        var refLoader = new InMemoryDataLoader(inM, tgtM, batchSize: 2, shuffle: true, seed: seed);
        refLoader.Next(); refLoader.Next();
        var lastUsed = new DataLoaderPosition(0, 1);
        var midPos = refLoader.Position;
        Assert.Equal(new DataLoaderPosition(0, 2), midPos);
        int[] refTail = TailIndices(refLoader, features: 1);

        var midPath = TempPath("loader_midpos") + ".safetensors";
        try
        {
            var ckpt0 = rigM.CreateInitialCheckpoint();
            var midCkpt = new TrainingCheckpoint
            {
                TrainableParams = ckpt0.TrainableParams,
                ModelState = ckpt0.ModelState,
                OptimizerState = ckpt0.OptimizerState,
                Step = 2, Epoch = lastUsed.Epoch, BatchIndex = lastUsed.BatchIndex,
            };
            midCkpt.Save(midPath);
            var reloaded = rigM.LoadCheckpoint(midPath);
            Assert.Equal(lastUsed.Epoch, reloaded.Epoch);
            Assert.Equal(lastUsed.BatchIndex, reloaded.BatchIndex);

            var restored = new InMemoryDataLoader(inM, tgtM, batchSize: 2, shuffle: true, seed: seed);
            restored.RestoreAfter(new DataLoaderPosition(reloaded.Epoch!.Value, reloaded.BatchIndex!.Value));
            Assert.Equal(midPos, restored.Position);
            Assert.Equal(refTail, TailIndices(restored, features: 1));
        }
        finally { if (File.Exists(midPath)) File.Delete(midPath); }
    }

    [Fact]
    public void TestInferenceModelExtractionSingleAndMultiInputCoverage()
    {
        var (sample, input, target) = ScalarMultiplyBatches();
        var rig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            sample, 0.1f);

        var stepped = rig.TrainStep(rig.CreateInitialCheckpoint(), input.Shared(), target.Shared());
        float wUpdated = Weight(rig, stepped);
        Assert.NotEqual(1.0f, wUpdated);

        var inference = rig.ExtractInferenceModel(stepped);
        Assert.Equal(GraphKind.ConcreteModel, inference.Kind);

        var concrete = rig.ModelConstituent.ToConcreteArchitecture(
            rig.ModelConstituent.FromOrderedInputs([sample[0].ToTensorData()]));
        var scheme = ModuleParamSetNamingScheme.FromModelIdFormats(concrete.GetShorokooIdNamingScheme(), "Shorokoo");
        var modelIds = concrete.GetConcreteModelParamInfos().ModelIds;
        foreach (var f in stepped.TrainableParams.Fields.Where(f => f.Value is TensorData))
            Assert.True(scheme.ToModelId(f.Key, modelIds) is not null);

        var probe = TensorData([4L], [2f, 3f, 4f, 5f]);
        var outputs = ComputeContext.Default.Execute(inference, probe)[0]
            .ToTensorData<float32>().AccessMemory().ToArray();
        float[] expected = [2f * wUpdated, 3f * wUpdated, 4f * wUpdated, 5f * wUpdated];
        for (int i = 0; i < expected.Length; i++)
            Assert.True(MathF.Abs(expected[i] - outputs[i]) < 1e-5f);

        var fitResult = rig.Fit([input], [target], numEpochs: 1);
        var fromCkpt = fitResult.FinalCheckpoint.ToInferenceModel();
        Assert.NotNull(fromCkpt);
        var fitOutputs = ComputeContext.Default.Execute(fromCkpt, TensorData([4L], [5f, 6f, 7f, 8f]));
        Assert.Single(fitOutputs);
        var output = fitOutputs[0].ToTensorData<float32>();
        Assert.Equal(1, output.Shape.Dims.Length);
        Assert.Equal(4L, output.Shape.Dims[0]);

        NamedModelParam[] twoInputs =
        [
            new TensorDataModelParam("a", ModelParamType.InputParam, TensorData([4L], [1f, 2f, 3f, 4f])),
            new TensorDataModelParam("b", ModelParamType.InputParam, TensorData([4L], [5f, 6f, 7f, 8f])),
        ];
        var twoRig = TrainingRig.FromScratch(
            TwoInputSumModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            twoInputs, 0.1f);
        Assert.Equal(2, twoRig.InputDef.Fields.Length);

        var twoInference = twoRig.CreateInitialCheckpoint().ToInferenceModel();
        Assert.Equal(GraphKind.ConcreteModel, twoInference.Kind);
        var twoOut = ComputeContext.Default.Execute(twoInference,
            TensorData([4L], [1f, 2f, 3f, 4f]), TensorData([4L], [10f, 20f, 30f, 40f]))[0]
            .ToTensorData<float32>().AccessMemory().ToArray();
        float[] twoExpected = [11f, 22f, 33f, 44f];
        for (int i = 0; i < twoExpected.Length; i++)
            Assert.True(MathF.Abs(twoExpected[i] - twoOut[i]) < 1e-5f);
    }

    // ---- Resident training runs (Shorokoo/Shorokoo#325) ----

    private static TrainingRig AdamWScalarRig(ComputeContext? runtimeContext = null) => TrainingRig.FromScratch(
        ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, AdamWOptimizer.ComputationGraph,
        [new TensorDataModelParam("input", ModelParamType.InputParam, TensorData([4L], [1f, 2f, 3f, 4f]))],
        new AdamWOptimizerHyperparameters { LearningRate = 0.1f }, runtimeContext: runtimeContext);

    /// <summary>The losses and final checkpoint of <paramref name="steps"/> TrainStep calls.</summary>
    private static (float[] Losses, TrainingCheckpoint Final) StepLoopRun(TrainingRig rig, int steps)
    {
        var (input, target) = (InBatch(1f, 2f, 3f, 4f), TargetBatch(2f, 4f, 6f, 8f));
        var ckpt = rig.CreateInitialCheckpoint();
        var losses = new float[steps];
        for (int i = 0; i < steps; i++)
        {
            ckpt = rig.TrainStep(ckpt, input.Shared(), target.Shared());
            losses[i] = ckpt.Loss!.Value;
        }
        return (losses, ckpt);
    }

    /// <summary>The other half of the ownership rule, and the one only a benchmark watched: the
    /// release a resident run performs on the state each step supersedes. The run owns that state
    /// internally, so the primitive it calls is what is assertable here — and a released tensor
    /// says so rather than reading freed memory, which is the invariant that makes it safe.</summary>
    [Fact]
    public void TestReleasingSupersededStateDisposesItsTensorsRatherThanLeavingThemReadable()
    {
        var rig = AdamWScalarRig();
        var (input, target) = (InBatch(1f, 2f, 3f, 4f), TargetBatch(2f, 4f, 6f, 8f));
        var superseded = rig.TrainStep(rig.CreateInitialCheckpoint(), input.Shared(), target.Shared());
        var tensors = Tensors(superseded);
        Assert.NotEmpty(tensors);
        Assert.All(tensors, t => Assert.False(t.IsDisposed));

        TrainingRig.ReleaseCheckpointState(superseded);

        Assert.All(tensors, t => Assert.True(t.IsDisposed));
        Assert.All(tensors, t => Assert.Throws<ObjectDisposedException>(() => t.CopyRawMemory()));
        TrainingRig.ReleaseCheckpointState(superseded);   // idempotent
    }

    /// <summary>A checkpoint the run published, and the one it was handed, outlive it: handing one
    /// over gives up the right to free it.</summary>
    [Fact]
    public void TestAResidentRunLeavesEveryCheckpointItPublishedReadable()
    {
        var rig = AdamWScalarRig();
        var (input, target) = (InBatch(1f, 2f, 3f, 4f), TargetBatch(2f, 4f, 6f, 8f));
        var initial = rig.CreateInitialCheckpoint();

        var run = rig.BeginResidentRun(initial.Shared());
        var published = run.StepToCheckpoint(input.Shared(), target.Shared());
        run.Step(input.Shared(), target.Shared());
        run.Dispose();

        Assert.All(Tensors(published), t => Assert.False(t.IsDisposed));
        Assert.All(Tensors(initial), t => Assert.False(t.IsDisposed));
    }

    private static TensorData[] Tensors(TrainingCheckpoint checkpoint) =>
        [.. checkpoint.TrainableParams.Fields.Values.OfType<TensorData>()];

    [Fact]
    public void TestAStepWritesItsStateIntoTheStateItConsumedAndTrainsExactlyAsOneThatDoesNot()
    {
        (float[] Values, long Aliased) Trained(bool aliasing)
        {
            using var context = new ComputeContext { OutputAliasing = aliasing };
            var rig = AdamWScalarRig(context);
            var (input, target) = (InBatch(1f, 2f, 3f, 4f), TargetBatch(2f, 4f, 6f, 8f));
            var ckpt = rig.CreateInitialCheckpoint();
            for (int i = 0; i < 3; i++) ckpt = rig.TrainStep(ckpt, input.Shared(), target.Shared());
            using var run = rig.BeginResidentRun(ckpt);
            run.Step(input.Shared(), target.Shared());
            var final = run.StepToCheckpoint(input.Shared(), target.Shared());
            return ([.. FlattenStruct(final.TrainableParams), .. FlattenStruct(final.OptimizerState)], context.AliasedOutputs);
        }

        var (aliased, written) = Trained(aliasing: true);
        var (plain, none) = Trained(aliasing: false);
        Assert.Equal(plain, aliased);
        Assert.Equal(20L, written);
        Assert.Equal(0L, none);
    }

    [Fact]
    public void TestAStepIsMarkedToWriteOverTheStateNothingReadsAfterItsUpdateAndNoOther()
    {
        var adamW = AdamWScalarRig();
        adamW.TrainStep(adamW.CreateInitialCheckpoint(), InBatch(1f, 2f, 3f, 4f), TargetBatch(2f, 4f, 6f, 8f));
        Assert.Equal([(0, 0), (1, 1), (2, 2), (3, 3)], adamW.MarkedStatePairs(Assert.Single(adamW.CompiledTrainStepShapeKeys)));

        var (matmul, ckpt) = CoverFromScratch(BatchedMatmulModel.ComputationGraph, SoftmaxL2Loss.ComputationGraph,
            SGDOptimizer.ComputationGraph, [4L, 5L, 8L], 0.01f);
        matmul.TrainStep(ckpt, matmul.InputDef.FromOrderedData(TensorData([4L, 5L, 8L], new float[160])),
            matmul.TargetDef.FromOrderedData(TensorData([4L, 4L], new float[16])));
        Assert.Equal([(0, 0)], matmul.MarkedStatePairs(Assert.Single(matmul.CompiledTrainStepShapeKeys)));
    }

    [Fact]
    public void TestEveryInitialCheckpointIsACopyAStepConsumesAndTheRigsOwnValuesAreNeverFed()
    {
        var rig = AdamWScalarRig();
        var (input, target) = (InBatch(1f, 2f, 3f, 4f), TargetBatch(2f, 4f, 6f, 8f));
        TensorData[] All(TrainingCheckpoint c) =>
            [.. ((TensorDataStruct[])[c.TrainableParams, c.ModelState, c.OptimizerState]).SelectMany(s => s.Fields.Values.OfType<TensorData>())];
        var (first, second) = (rig.CreateInitialCheckpoint(), rig.CreateInitialCheckpoint());

        Assert.Equal(FlattenStruct(second.OptimizerState), FlattenStruct(first.OptimizerState));
        Assert.Empty(All(first).Intersect(All(second)));
        Assert.Empty(All(first).Intersect(rig.OwnInitialValues));
        rig.TrainStep(first, input.Shared(), target.Shared());
        Assert.All(All(first), t => Assert.True(t.IsDisposed));
        using (var run = rig.BeginResidentRun()) run.Step(input.Shared(), target.Shared());
        rig.Fit([input], [target], numEpochs: 1);

        Assert.All(All(second), t => Assert.False(t.IsDisposed));
        Assert.All(rig.OwnInitialValues, t => Assert.True(!t.IsDisposed && t.CopiesAreEmpty));
        Assert.Equal(FlattenStruct(second.TrainableParams), FlattenStruct(rig.CreateInitialCheckpoint().TrainableParams));
    }

    [Fact]
    public void TestAnInitialCheckpointGivenHyperparametersAndTheStateALoadFillsInAreFreshCopiesOfTheRigsValuesToo()
    {
        var x = TensorData([2L], 1f, 2f);
        var rig = TrainingRig.FromScratch(StatefulGainNoRefModel.ComputationGraph, L2Loss.ComputationGraph,
            AdamWOptimizer.ComputationGraph, [new TensorDataModelParam("input", ModelParamType.InputParam, x)],
            new AdamWOptimizerHyperparameters { LearningRate = 0.1f });
        TensorDataStruct[] Families(TrainingCheckpoint c) => [c.TrainableParams, c.ModelState, c.OptimizerState];
        var (flat, skpt) = (TempPath("defaults") + ".safetensors", TempPath("defaults") + ".skpt");
        try
        {
            rig.CreateInitialCheckpoint().Save(flat, CheckpointComponents.Counters);
            Persistence.SaveTrainingCheckpointToSkpt(rig.CreateInitialCheckpoint(), skpt);
            TrainingCheckpoint[] handedOut =
            [
                rig.CreateInitialCheckpoint(rig.MakeHyperparameters()),
                rig.CreateInitialCheckpoint(rig.MakeHyperparameters()),
                rig.LoadCheckpoint(flat, CheckpointComponents.Counters),
                rig.LoadCheckpoint(flat, CheckpointComponents.Counters),
                rig.LoadCheckpointFromSkpt(skpt, CheckpointComponents.Counters),
            ];

            Assert.All(handedOut, c => Assert.All(Families(c), s => Assert.NotEmpty(s.Fields)));
            TensorData[] tensors = [.. handedOut.SelectMany(Families).SelectMany(s => s.Fields.Values.OfType<TensorData>())];
            Assert.Equal(tensors.Length, tensors.Distinct().Count());
            Assert.Empty(tensors.Intersect(rig.OwnInitialValues));
        }
        finally
        {
            string[] written = [flat, skpt];
            foreach (var p in written) if (File.Exists(p)) File.Delete(p);
        }
    }

    /// <summary>The same run through a resident run, checkpointing on the last step only.</summary>
    private static (float[] Losses, TrainingCheckpoint Final) ResidentRun(TrainingRig rig, int steps)
    {
        var (input, target) = (InBatch(1f, 2f, 3f, 4f), TargetBatch(2f, 4f, 6f, 8f));
        using var run = rig.BeginResidentRun();
        var losses = new float[steps];
        TrainingCheckpoint final = null!;
        for (int i = 0; i < steps; i++)
        {
            if (i == steps - 1) losses[i] = (final = run.StepToCheckpoint(input.Shared(), target.Shared())).Loss!.Value;
            else losses[i] = run.Step(input.Shared(), target.Shared());
        }
        return (losses, final);
    }

    [Fact]
    public void TestAResidentRunTrainsTheSameTrajectoryAsATrainStepLoop()
    {
        var (stepLosses, stepFinal) = StepLoopRun(AdamWScalarRig(), 5);
        var (residentLosses, residentFinal) = ResidentRun(AdamWScalarRig(), 5);

        Assert.Equal(stepLosses, residentLosses);
        Assert.Equal(FlattenStruct(stepFinal.TrainableParams), FlattenStruct(residentFinal.TrainableParams));
        Assert.Equal(FlattenStruct(stepFinal.OptimizerState), FlattenStruct(residentFinal.OptimizerState));
        Assert.Equal(stepFinal.Step, residentFinal.Step);
        Assert.Equal(stepFinal.Loss, residentFinal.Loss);
    }

    [Fact]
    public void TestAResidentRunOverALoaderMatchesFitAndCarriesTheSameCounters()
    {
        const int features = 4;
        var fitRig = LoaderRig(batchSize: 2, features);
        var (fitIn, fitTgt) = IndexDataset(fitRig, n: 6, features);
        var fit = fitRig.Fit(new InMemoryDataLoader(fitIn, fitTgt, batchSize: 2), numEpochs: 1);

        var runRig = LoaderRig(batchSize: 2, features);
        var (runIn, runTgt) = IndexDataset(runRig, n: 6, features);
        var loader = new InMemoryDataLoader(runIn, runTgt, batchSize: 2);
        using var run = runRig.BeginResidentRun();
        run.Step(loader);
        run.Step(loader);
        var final = run.StepToCheckpoint(loader);

        Assert.Equal(fit.FinalCheckpoint.Step, final.Step);
        Assert.Equal(fit.FinalCheckpoint.Epoch, final.Epoch);
        Assert.Equal(fit.FinalCheckpoint.BatchIndex, final.BatchIndex);
        Assert.Equal(FlattenStruct(fit.FinalCheckpoint.TrainableParams), FlattenStruct(final.TrainableParams));
        Assert.Same(runRig, final.Rig);
        Assert.Equal(3, run.CurrentStep);
    }

    [Fact]
    public void TestAResidentRunFreesOnlyTheStateNobodyElseHolds()
    {
        var rig = AdamWScalarRig();
        var (input, target) = (InBatch(1f, 2f, 3f, 4f), TargetBatch(2f, 4f, 6f, 8f));
        var initial = rig.CreateInitialCheckpoint();

        var initialBefore = FlattenStruct(initial.TrainableParams);

        var run = rig.BeginResidentRun(initial.Shared());
        run.Step(input.Shared(), target.Shared());
        var published = run.StepToCheckpoint(input.Shared(), target.Shared());
        var publishedBefore = FlattenStruct(published.TrainableParams);
        run.Step(input.Shared(), target.Shared());
        run.Dispose();

        // Values, not emptiness: a released tensor still reports its element count, so an array of
        // the right length says nothing. These two have to still hold what they held.
        Assert.Equal(initialBefore, FlattenStruct(initial.TrainableParams));
        Assert.Equal(publishedBefore, FlattenStruct(published.TrainableParams));
        Assert.NotEmpty(initialBefore);
        Assert.Equal(2, published.Step);
        Assert.Throws<ObjectDisposedException>(() => run.Step(input.Shared(), target.Shared()));
        run.Dispose();
    }

    [Fact]
    public void TestACheckpointIsConsumedByTheStepItFeedsUnlessSharedAndSaysWhichStepTookIt()
    {
        var rig = AdamWScalarRig();
        var (input, target) = (InBatch(1f, 2f, 3f, 4f), TargetBatch(2f, 4f, 6f, 8f));
        var first = rig.TrainStep(rig.CreateInitialCheckpoint(), input.Shared(), target.Shared());

        Assert.Null(first.FeedMode);
        Assert.Equal(SharedInputMode.Shared, first.Shared().WithStep(7).FeedMode);
        Assert.Equal(SharedInputMode.TryConsume, first.TryConsume().FeedMode);
        var read = rig.TrainStep(first.Shared(), input.Shared(), target.Shared());
        Assert.Null(read.FeedMode);
        Assert.DoesNotContain(Tensors(first), t => t.IsDisposed);
        rig.TrainStep(read.TryConsume(), input.Shared(), target.Shared());
        Assert.All(Tensors(read), t => Assert.True(t.IsDisposed));

        rig.TrainStep(first, input.Shared(), target.Shared());
        var refused = Assert.Throws<ObjectDisposedException>(() => FlattenStruct(first.TrainableParams)).Message;
        Assert.Contains("consumed by a run of a TrainingRig's training step", refused);
        Assert.Contains("the checkpoint's trainable parameter", refused);
    }

    [Fact]
    public void TestAResidentRunWhoseStepFailsAfterTakingItsOwnStateIsLostAndOneThatOnlyReadAPublishedCheckpointGoesOn()
    {
        var rig = AdamWScalarRig();
        var (input, target) = (InBatch(1f, 2f, 3f, 4f), TargetBatch(2f, 4f, 6f, 8f));
        TensorDataStruct Mistyped() => new(
            new TensorStructDef([new TensorStructFieldDef("targets", DataStructure.Tensor, 1, DType.Float64)], "Target"),
            new Dictionary<string, IData> { { "targets", TensorData([4L], 2.0, 4.0, 6.0, 8.0) } });
        using var run = rig.BeginResidentRun();

        run.Step(input.Shared(), target.Shared());
        var published = run.StepToCheckpoint(input.Shared(), target.Shared());
        Assert.Throws<OnnxRuntimeException>(() => run.Step(input.Shared(), Mistyped()));
        Assert.True(float.IsFinite(run.Step(input.Shared(), target.Shared())));
        Assert.Throws<OnnxRuntimeException>(() => run.Step(input.Shared(), Mistyped()));
        Assert.Contains("StepToCheckpoint", Assert.Throws<InvalidOperationException>(
            () => run.Step(input.Shared(), target.Shared())).Message);
        Assert.DoesNotContain(Tensors(published), t => t.IsDisposed);
    }

    [Fact]
    public void TestABatchIsConsumedByTheStepItFeedsUnlessSharedAndAnythingButAStructIsRefused()
    {
        var rig = AdamWScalarRig();
        var (input, target) = (InBatch(1f, 2f, 3f, 4f), TargetBatch(2f, 4f, 6f, 8f));
        var at = new DataLoaderPosition(0, 0);
        var loose = TensorData([4L], 1f, 2f, 3f, 4f);
        static TensorData Field(TensorDataStruct batch) => (TensorData)batch.Fields.Values.Single();
        using var run = rig.BeginResidentRun();

        run.Step(new DataBatch(input.Shared(), target.Shared(), at));
        Assert.False(Field(input).IsDisposed || Field(target).IsDisposed);
        run.Step(new DataBatch(input, target, at));
        Assert.True(Field(input).IsDisposed && Field(target).IsDisposed);

        Assert.Contains("rig.InputDef.FromOrderedData",
            Assert.Throws<ArgumentException>(() => new DataBatch(loose, TargetBatch(1f), at)).Message);
        Assert.Contains("a shared TensorData", Assert.Throws<ArgumentException>(() => rig.TrainStep(
            rig.CreateInitialCheckpoint(), loose.Shared(), TargetBatch(2f, 4f, 6f, 8f))).Message);
        Assert.False(loose.IsDisposed);
    }

    [Fact]
    public void TestAResidentRunAppliesRuntimeHyperparametersAndRefusesThemMissing()
    {
        var rig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            [new TensorDataModelParam("input", ModelParamType.InputParam, TensorData([4L], [1f, 2f, 3f, 4f]))],
            Hyperparameter.Runtime());
        var (input, target) = (InBatch(1f, 2f, 3f, 4f), TargetBatch(2f, 4f, 6f, 8f));
        var hypers = rig.MakeHyperparameters(0.1f);

        var stepped = rig.TrainStep(rig.CreateInitialCheckpoint(), hypers.Shared(), input.Shared(), target.Shared());
        using var run = rig.BeginResidentRun();
        var resident = run.StepToCheckpoint(hypers.Shared(), input.Shared(), target.Shared());

        Assert.Equal(FlattenStruct(stepped.TrainableParams), FlattenStruct(resident.TrainableParams));
        Assert.Contains("MakeHyperparameters", Assert.Throws<InvalidOperationException>(
            () => run.Step(input.Shared(), target.Shared())).Message);
    }

    // Retention is a backend capability, and this one has no memory but the host's. A provider
    // wrongly reported as having its own would leave every checkpoint tensor unreadable, so the
    // discovery must not misfire — and asking to retain must stay a no-op when there is nowhere to
    // retain to.
    [Fact]
    public void TestOnAHostOnlyBackendNothingIsRetainedAndEveryOutputStaysReadable()
    {
        var rig = AdamWScalarRig();
        var (input, target) = (InBatch(1f, 2f, 3f, 4f), TargetBatch(2f, 4f, 6f, 8f));
        var ckpt = rig.CreateInitialCheckpoint();
        var compiled = rig.RuntimeContext.Compile(rig.TrainingStepPureGraph);
        Assert.False(compiled.HasDeviceMemory);

        // A retention array of any other length is refused rather than half-applied.
        Assert.Throws<InvalidTensorOperationException>(() => compiled.Execute(
            ComputeContext.ExpandStructInputs([ckpt.TrainableParams.Shared(), ckpt.ModelState.Shared(), ckpt.OptimizerState.Shared(), input.Shared(), target.Shared()]),
            [.. Enumerable.Repeat(true, compiled.OutputCount + 1)]));

        IData[] inputs = [ckpt.TrainableParams.Shared(), ckpt.ModelState.Shared(), ckpt.OptimizerState.Shared(), input.Shared(), target.Shared()];
        var outputs = compiled.Execute(
            ComputeContext.ExpandStructInputs(inputs),
            [.. Enumerable.Repeat(true, compiled.OutputCount)]);
        Assert.All(outputs, o => Assert.True(o.ToTensorData().IsHostResident));

        using var run = rig.BeginResidentRun(ckpt);
        run.Step(input.Shared(), target.Shared());
        var stepped = run.StepToCheckpoint(input.Shared(), target.Shared());
        Assert.All(stepped.TrainableParams.Fields.Values, f => Assert.True(((TensorData)f).IsHostResident));
        Assert.All(stepped.OptimizerState.Fields.Values, f => Assert.True(((TensorData)f).IsHostResident));
        Assert.NotEmpty(FlattenStruct(stepped.TrainableParams));
    }
}

[Trait("Domain", "Training")]
[Trait("Purpose", "Coverage")]
public class TrainingRigCheckpointCoverageTests
{
    private static TrainingRig AdamRig() => TrainingRig.FromScratch(
        ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, AdamOptimizer.ComputationGraph,
        [new TensorDataModelParam("input", ModelParamType.InputParam, TensorData([4L], [1f, 2f, 3f, 4f]))],
        new AdamOptimizerHyperparameters { LearningRate = 0.1f });

    private static TrainingCheckpoint SteppedOnce(TrainingRig rig) => rig.TrainStep(
        rig.CreateInitialCheckpoint(), InBatch(1f, 2f, 3f, 4f), TargetBatch(2f, 4f, 6f, 8f));

    [Fact]
    public void TestStepTensorInventoryNamesEverySectionOfTheResidentState()
    {
        var ckpt = SteppedOnce(AdamRig());

        var inventory = TrainingRig.StepTensorInventory(ckpt);
        Assert.Equal(["trainable parameters", "model state", "optimizer state"],
            inventory.Select(s => s.Name));
        var withBatch = TrainingRig.StepTensorInventory(
            ckpt, InBatch(1f, 2f, 3f, 4f), TargetBatch(2f, 4f, 6f, 8f));
        Assert.Equal(
            ["trainable parameters", "model state", "optimizer state", "training input", "training target"],
            withBatch.Select(s => s.Name));
        Assert.Equal(4 * sizeof(float), withBatch[3].TotalBytes);
        Assert.Equal(ckpt.TrainableParams.Definition.Fields.Length, inventory[0].Tensors.Count);
        Assert.Equal(ckpt.OptimizerState.Definition.Fields.Length, inventory[2].Tensors.Count);
        Assert.All(inventory.SelectMany(s => s.Tensors), t => Assert.True(t.Bytes > 0));
        Assert.All(inventory, s => Assert.True(s.TotalBytes >= 0));
        Assert.True(inventory.Sum(s => s.TotalBytes) > 0);

        var report = AllocationFailureReport.Render(
            $"the training step at step {ckpt.Step}", AllocationPool.Device,
            new DeviceFacts(true, null, null, "Shorokoo.LinuxGPU"),
            inventory, AllocationFailureReport.ReadProcessMemory(), "bad allocation");
        Assert.Contains("trainable parameters", report);
        Assert.Contains("optimizer state", report);
        Assert.Contains("the training step at step 1", report);
    }

    [Fact]
    public void TestATrainStepAllocationFailureIsWrappedWithTheStepAndItsOriginalCause()
    {
        var rig = AdamRig();
        var ckpt = rig.CreateInitialCheckpoint().WithStep(41);
        const string arena = "[ErrorCode:Fail] /onnxruntime/core/framework/bfc_arena.cc:358 "
            + "onnxruntime::BFCArena::AllocateRawInternal Failed to allocate memory for "
            + "requested buffer of size 2359296";

        TrainingRig.StepFaultInjection = () => throw new InvalidOperationException(arena);
        try
        {
            var thrown = Assert.Throws<ComputeContextException>(
                () => rig.TrainStep(ckpt.Shared(), InBatch(1f, 2f, 3f, 4f), TargetBatch(2f, 4f, 6f, 8f)));
            Assert.Equal(ErrorCodes.CR009, thrown.ErrorCode);
            Assert.Contains("the training step at step 41", thrown.Message);
            Assert.Contains("trainable parameters", thrown.Message);
            Assert.Contains(arena, thrown.Message);
            Assert.Equal(arena, thrown.InnerException?.Message);
            Assert.Contains("HOST memory", thrown.Message);

            TrainingRig.StepFaultInjection = () => throw new InvalidOperationException("Node (Foo) is not supported");
            Assert.IsNotType<ComputeContextException>(Assert.Throws<InvalidOperationException>(
                () => rig.TrainStep(ckpt.Shared(), InBatch(1f, 2f, 3f, 4f), TargetBatch(2f, 4f, 6f, 8f))));
        }
        finally { TrainingRig.StepFaultInjection = null; }
    }

    [Fact]
    public void TestCheckpointDerivationsCarryEverySlotThrough()
    {
        var rig = AdamRig();
        var ckpt = SteppedOnce(rig).WithCounters(step: 5, epoch: 2, batchIndex: 7);

        static void Same(TrainingCheckpoint a, TrainingCheckpoint b)
        {
            Assert.Same(a.TrainableParams, b.TrainableParams);
            Assert.Same(a.ModelState, b.ModelState);
            Assert.Same(a.OptimizerState, b.OptimizerState);
            Assert.Equal(a.Step, b.Step);
            Assert.Equal(a.Epoch, b.Epoch);
            Assert.Equal(a.BatchIndex, b.BatchIndex);
            Assert.Same(a.Rig, b.Rig);
            Assert.Equal(a.Loss, b.Loss);
        }

        Same(ckpt, ckpt.WithCounters());
        Same(ckpt, ckpt.WithTrainableParams(ckpt.TrainableParams));
        Same(ckpt, ckpt.WithModelState(ckpt.ModelState));
        Same(ckpt, ckpt.WithOptimizerState(ckpt.OptimizerState));

        var swapped = ckpt.WithTrainableParams(ckpt.OptimizerState);
        Assert.Same(ckpt.OptimizerState, swapped.TrainableParams);
        Assert.Same(ckpt.ModelState, swapped.ModelState);
        Assert.Same(ckpt.OptimizerState, swapped.OptimizerState);
        Assert.Equal(5, swapped.Step);
        Assert.Equal(2, swapped.Epoch);
        Assert.Equal(7, swapped.BatchIndex);
        Assert.Same(rig, swapped.Rig);
        Assert.Equal(ckpt.Loss, swapped.Loss);

        Assert.Equal(9, ckpt.WithStep(9).Step);
        Assert.Equal(3, ckpt.WithEpoch(3).Epoch);
        Assert.Equal(4, ckpt.WithBatchIndex(4).BatchIndex);
        Assert.NotSame(ckpt, ckpt.WithStep(9));
        Assert.Equal(5, ckpt.Step);

        Assert.Throws<ArgumentNullException>(() => ckpt.WithTrainableParams(null!));
        Assert.Throws<ArgumentNullException>(() => ckpt.WithModelState(null!));
        Assert.Throws<ArgumentNullException>(() => ckpt.WithOptimizerState(null!));
        Assert.Throws<ArgumentNullException>(() => new TrainingCheckpoint
        {
            TrainableParams = null!, ModelState = ckpt.ModelState, OptimizerState = ckpt.OptimizerState,
        });
    }

    private static TrainingRig ShapeRig(ComputationGraph model) => TrainingRig.FromScratch(
        model, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
        [new TensorDataModelParam("x", ModelParamType.InputParam, TensorData([4L, 4L],
            [1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f]))],
        0.1f);

    private static long[] ParamDims(TrainingCheckpoint c) =>
        [.. ((TensorData)c.TrainableParams.Fields[c.TrainableParams.Definition.Fields[0].Name]).Shape.Dims];

    /// <summary>The rig collects on a budget of checkpoint state a step superseded without
    /// consuming — fed <c>.Shared()</c> — which is only garbage if the caller drops it. A caller that
    /// keeps its checkpoints would otherwise buy a blocking collection that frees nothing, on every
    /// step, forever.</summary>
    [Fact]
    public void TestReclamationBacksOffWhileTheCallerKeepsItsCheckpointsAndResumesWhenItStops()
    {
        var rig = ShapeRig(ParamOrderAModel.ComputationGraph);
        var input = rig.InputDef.FromOrderedData(TensorData([4L], [1f, 2f, 3f, 4f]));
        var target = rig.TargetDef.FromOrderedData(TensorData([4L], [1f, 2f, 3f, 4f]));
        rig.SetReclaimBudgetForTests(1);

        var kept = new List<TrainingCheckpoint>();
        var cp = rig.CreateInitialCheckpoint();
        for (int i = 0; i < 5; i++) { cp = rig.TrainStep(cp.Shared(), input.Shared(), target.Shared()); kept.Add(cp); }
        Assert.True(rig.ReclaimBudgetBytes > 1);

        kept.Clear();
        for (int i = 0; i < 4; i++) cp = rig.TrainStep(cp.Shared(), input.Shared(), target.Shared());
        Assert.Equal(1, rig.ReclaimBudgetBytes);
        Assert.NotNull(cp);
    }

    /// <summary>A caller that keeps no checkpoint, and one that keeps a single older checkpoint,
    /// both supersede everything else — so every collection reclaims and the budget must stay at its
    /// base. Neither is distinguishable by watching one recent checkpoint: the one handed back at a
    /// reclamation is the next step's own input, alive at the collection whatever the caller does,
    /// and a single older survivor is what keeping one looks like. The rule is therefore all the
    /// older watches or none.</summary>
    [Fact]
    public void TestReclamationDoesNotBackOffForACallerThatKeepsNoCheckpointAtAll()
    {
        var rig = ShapeRig(ParamOrderAModel.ComputationGraph);
        var input = rig.InputDef.FromOrderedData(TensorData([4L], [1f, 2f, 3f, 4f]));
        var target = rig.TargetDef.FromOrderedData(TensorData([4L], [1f, 2f, 3f, 4f]));
        rig.SetReclaimBudgetForTests(1);

        var consumed = rig.CreateInitialCheckpoint();
        for (int i = 0; i < 4; i++) consumed = rig.TrainStep(consumed, input.Shared(), target.Shared());
        Assert.Equal(0, rig.SupersededStateBytesPending);

        var cp = rig.CreateInitialCheckpoint();
        long worst = 0;
        for (int i = 0; i < 12; i++)
        {
            cp = rig.TrainStep(cp.Shared(), input.Shared(), target.Shared());
            worst = Math.Max(worst, rig.ReclaimBudgetBytes);
        }
        Assert.NotNull(cp);
        Assert.Equal(1, worst);

        // Keeping one checkpoint — the best so far, the step before — supersedes every other, so
        // the collections are still worth making and the budget still must not climb.
        var keepsOne = ShapeRig(ParamOrderAModel.ComputationGraph);
        keepsOne.SetReclaimBudgetForTests(1);
        var current = keepsOne.CreateInitialCheckpoint();
        TrainingCheckpoint? best = null;
        long worstKeepingOne = 0;
        for (int i = 0; i < 12; i++)
        {
            var previous = current;
            current = keepsOne.TrainStep(previous.Shared(), input.Shared(), target.Shared());
            // One checkpoint held that is not the one being fed back in, which is what "the best so
            // far" is: every third step improves, and the rest are superseded and freed.
            if (i % 3 == 0) best = previous;
            worstKeepingOne = Math.Max(worstKeepingOne, keepsOne.ReclaimBudgetBytes);
        }
        Assert.NotNull(best);
        Assert.Equal(1, worstKeepingOne);
    }

    /// <summary>A checkpoint whose parameters are shaped differently is refused on every route
    /// into a rig: dimensions are compared, not just ranks.</summary>
    [Fact]
    public void TestACheckpointIsRefusedByAModelWhoseParametersAreShapedDifferently()
    {
        var narrow = ShapeRig(ParamShapeNarrowModel.ComputationGraph);
        var wide = ShapeRig(ParamShapeWideModel.ComputationGraph);
        var flat = TempPath("ckpt_narrow") + ".safetensors";
        var skpt = TempPath("ckpt_narrow") + ".skpt";
        try
        {
            var narrowCkpt = narrow.CreateInitialCheckpoint();
            narrowCkpt.Save(flat);
            Persistence.SaveTrainingCheckpointToSkpt(narrowCkpt, skpt);

            Assert.Equal([4L, 2L], ParamDims(narrowCkpt));
            Assert.Equal([4L, 8L], ParamDims(wide.CreateInitialCheckpoint()));
            Assert.Equal([4L, 2L], ParamDims(narrow.LoadCheckpoint(flat)));
            Assert.Equal([4L, 2L], ParamDims(narrow.LoadCheckpointFromSkpt(skpt)));

            var defLess = Persistence.LoadTrainingCheckpoint(flat);
            Assert.Equal([4L, 2L], ParamDims(defLess));

            string Refusal(Action load) => Assert.Throws<ArgumentException>(load).Message;
            Assert.All(
                (string[])
                [
                    Refusal(() => wide.LoadCheckpoint(flat)),
                    Refusal(() => wide.LoadCheckpointFromSkpt(skpt)),
                    Refusal(() => wide.AdoptCheckpoint(defLess)),
                    Refusal(() => wide.AdoptCheckpoint(narrowCkpt)),
                ],
                m => Assert.Contains("[4,2]", m.Replace(", ", ",")));
        }
        finally
        {
            if (File.Exists(flat)) File.Delete(flat);
            if (File.Exists(skpt)) File.Delete(skpt);
        }
    }

    /// <summary>A rig refuses a checkpoint whose values do not fit it in any respect it can see: a
    /// file saved without its inference state, which read on its own would claim a model with no
    /// parameters, and a value whose element type is not the parameter's.</summary>
    [Fact]
    public void TestAPartialCheckpointAndAMismatchedElementTypeAreRefused()
    {
        var rig = ShapeRig(ParamShapeNarrowModel.ComputationGraph);
        var paramName = rig.TrainableParamStructDef.Fields[0].Name;
        var path = TempPath("ckpt_partial") + ".safetensors";
        try
        {
            rig.CreateInitialCheckpoint().Save(
                path, CheckpointComponents.OptimizerState | CheckpointComponents.Counters);

            Assert.Contains("saved without its inference state",
                Assert.Throws<InvalidOperationException>(() => Persistence.LoadTrainingCheckpoint(path)).Message);
            Assert.Equal([4L, 2L], ParamDims(rig.LoadCheckpoint(path)));

            var foreign = TempPath("not_a_checkpoint") + ".safetensors";
            try
            {
                Shorokoo.Onnx.SafeTensorLoader.SaveSafeTensors(foreign,
                    [new Shorokoo.Onnx.SafeTensor("w", TensorData([2L], [1f, 2f]), "F32", [2L])]);
                Assert.Contains("is not a Shorokoo training checkpoint",
                    Assert.Throws<InvalidOperationException>(() => Persistence.LoadTrainingCheckpoint(foreign)).Message);
            }
            finally { if (File.Exists(foreign)) File.Delete(foreign); }

            var wrongType = new TrainingCheckpoint
            {
                TrainableParams = new TensorDataStruct(rig.TrainableParamStructDef,
                    [new(paramName, TensorData([4L, 2L], [1d, 2d, 3d, 4d, 5d, 6d, 7d, 8d]))]),
                ModelState = rig.CreateInitialCheckpoint().ModelState,
                OptimizerState = rig.CreateInitialCheckpoint().OptimizerState,
            };
            Assert.Contains("Float64", Assert.Throws<ArgumentException>(() => rig.AdoptCheckpoint(wrongType)).Message);
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    /// <summary>Binding is the last point where a value and the model that will use it are both in
    /// hand, so it is where a value of the wrong shape is refused — a checkpoint assembled by hand
    /// reaches the model without passing a load. Left unchecked the graph builds and runs, returning
    /// a result of the bound value's shape instead of the model's. Part of Shorokoo/Shorokoo#322.</summary>
    [Fact]
    public void TestAValueOfAnotherShapeIsRefusedWhereItIsBoundToTheModel()
    {
        var narrow = ShapeRig(ParamShapeNarrowModel.ComputationGraph);
        var wide = ShapeRig(ParamShapeWideModel.ComputationGraph);
        var wideCkpt = wide.CreateInitialCheckpoint();
        var narrowValues = narrow.CreateInitialCheckpoint().TrainableParams.Fields;

        Assert.NotNull(wideCkpt.ToInferenceModel());
        var handAssembled = new TrainingCheckpoint
        {
            TrainableParams = new TensorDataStruct(wide.TrainableParamStructDef, narrowValues),
            ModelState = wideCkpt.ModelState,
            OptimizerState = wideCkpt.OptimizerState,
            Rig = wide,
        };
        Assert.Throws<InvalidOperationException>(() => handAssembled.ToInferenceModel());
    }

    /// <summary>A flat checkpoint is self-describing, so it loads with no struct defs supplied: the
    /// section prefixes give the kinds and the safetensors header gives each field's name, rank and
    /// element type, in the order the file lays them out. The rig is what judges such a checkpoint —
    /// adopting one re-labels it with the rig's own defs and refuses it if it does not match.</summary>
    [Fact]
    public void TestAFlatCheckpointLoadsWithNoStructDefsAndKeepsItsFieldOrder()
    {
        NamedModelParam[] sample =
        [
            new TensorDataModelParam("input", ModelParamType.InputParam,
                TensorData([4L], [1f, 2f, 3f, 4f])),
        ];
        var rig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            AdamWOptimizer.ComputationGraph, sample,
            new AdamWOptimizerHyperparameters { LearningRate = 0.1f });
        static string[] Names(TensorStructDef d) => [.. d.Fields.Select(f => f.Name)];
        static int?[] Ranks(TensorStructDef d) => [.. d.Fields.Select(f => f.Rank)];
        static DType[] Types(TensorStructDef d) => [.. d.Fields.Select(f => f.ElementType)];

        var saved = rig.TrainStep(rig.CreateInitialCheckpoint(),
            InBatch(1f, 2f, 3f, 4f), TargetBatch(2f, 4f, 6f, 8f));
        var path = TempPath("ckpt_nodefs") + ".safetensors";
        try
        {
            saved.Save(path);
            var loaded = Persistence.LoadTrainingCheckpoint(path);

            Assert.Equal(Names(rig.TrainableParamStructDef), Names(loaded.TrainableParams.Definition));
            Assert.Equal(Types(rig.TrainableParamStructDef), Types(loaded.TrainableParams.Definition));
            Assert.Equal(Names(rig.OptimizerStateDef), Names(loaded.OptimizerState.Definition));
            Assert.Equal((int?[])[1], Ranks(loaded.TrainableParams.Definition));
            Assert.Equal((int?[])[1, 1, 0], Ranks(loaded.OptimizerState.Definition));
            Assert.Equal(1, loaded.Step);
            Assert.Null(loaded.Rig);

            var adopted = rig.AdoptCheckpoint(loaded);
            Assert.Same(rig.TrainableParamStructDef, adopted.TrainableParams.Definition);
            Assert.Same(rig.OptimizerStateDef, adopted.OptimizerState.Definition);
            Assert.Equal(
                FlattenStruct(rig.LoadCheckpoint(path).TrainableParams), FlattenStruct(adopted.TrainableParams));
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    [Fact]
    public void TestCheckpointSaveLoadResumeAndAdamScalarStepCoverage()
    {
        NamedModelParam[] sample =
        [
            new TensorDataModelParam("input", ModelParamType.InputParam,
                TensorData([4L], [1f, 2f, 3f, 4f])),
        ];
        TrainingRig AdamWRig() => TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            AdamWOptimizer.ComputationGraph, sample,
            new AdamWOptimizerHyperparameters { LearningRate = 0.1f });
        TrainingRig AdamRig() => TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            AdamOptimizer.ComputationGraph, sample,
            new AdamOptimizerHyperparameters { LearningRate = 0.1f });

        var inputBatch = InBatch(1f, 2f, 3f, 4f);
        var targetBatch = TargetBatch(2f, 4f, 6f, 8f);

        var path = TempPath("ckpt") + ".safetensors";
        try
        {
            var rigA = AdamWRig();
            var ckpt = rigA.CreateInitialCheckpoint();
            for (int i = 0; i < 2; i++)
                ckpt = rigA.TrainStep(ckpt, inputBatch.Shared(), targetBatch.Shared());
            Assert.Equal(2, ckpt.Step);
            Assert.Equal(3, rigA.OptimizerStateDef.Fields.Length);
            var adamWStep = (TensorData)ckpt.OptimizerState.Fields[rigA.OptimizerStateDef.Fields[2].Name];
            Assert.Empty(adamWStep.Shape.Dims);
            Assert.Equal(2f, adamWStep.As<float32>().AccessMemory()[0]);
            ckpt.Save(path);
            Assert.True(File.Exists(path));

            var rigB = AdamWRig();
            var loaded = rigB.LoadCheckpoint(path);

            Assert.Equal(2, loaded.Step);
            Assert.Equal(FlattenStruct(ckpt.TrainableParams), FlattenStruct(loaded.TrainableParams));
            Assert.Equal(FlattenStruct(ckpt.OptimizerState), FlattenStruct(loaded.OptimizerState));
            Assert.NotEmpty(loaded.OptimizerState.Fields);

            var resumed = rigB.TrainStep(loaded.Shared(), inputBatch.Shared(), targetBatch.Shared());
            Assert.Equal(3, resumed.Step);
            Assert.True(float.IsFinite(resumed.Loss!.Value));

            var bnRig = TrainingRig.FromScratch(
                ScalarMultiplyWithBatchNormModel.ComputationGraph, L2Loss.ComputationGraph,
                SGDMomentumOptimizer.ComputationGraph,
                [
                    new TensorDataModelParam("input", ModelParamType.InputParam,
                        TensorData([8L], new float[8])),
                ],
                0.5f, 0.9f);
            var bnPath = TempPath("ckpt_bn") + ".safetensors";
            try
            {
                var bnCkpt = bnRig.CreateInitialCheckpoint();
                Assert.NotEmpty(bnCkpt.ModelState.Fields);
                bnCkpt.Save(bnPath);
                var bnLoaded = bnRig.LoadCheckpoint(bnPath);
                Assert.Equal(FlattenStruct(bnCkpt.ModelState), FlattenStruct(bnLoaded.ModelState));
                Assert.Equal(FlattenStruct(bnCkpt.OptimizerState), FlattenStruct(bnLoaded.OptimizerState));

                Assert.Throws<InvalidOperationException>(() => bnRig.LoadCheckpoint(path));
            }
            finally { if (File.Exists(bnPath)) File.Delete(bnPath); }
        }
        finally { if (File.Exists(path)) File.Delete(path); }

        var adamRig = AdamRig();
        Assert.Equal(3, adamRig.OptimizerStateDef.Fields.Length);
        var stepField = adamRig.OptimizerStateDef.Fields[2];
        Assert.Equal(0, stepField.Rank);

        var adamCkpt = adamRig.CreateInitialCheckpoint();
        for (int i = 0; i < 2; i++)
            adamCkpt = adamRig.TrainStep(adamCkpt, inputBatch.Shared(), targetBatch.Shared());

        var stepData = (TensorData)adamCkpt.OptimizerState.Fields[stepField.Name];
        Assert.Empty(stepData.Shape.Dims);
        Assert.Equal(2f, stepData.As<float32>().AccessMemory()[0]);

        var adamPath = TempPath("adam_scalar") + ".safetensors";
        try
        {
            adamCkpt.Save(adamPath);

            var loaded = AdamRig().LoadCheckpoint(adamPath);
            Assert.Equal(2, loaded.Step);
            Assert.Equal(FlattenStruct(adamCkpt.OptimizerState), FlattenStruct(loaded.OptimizerState));
            Assert.Equal(FlattenStruct(adamCkpt.TrainableParams), FlattenStruct(loaded.TrainableParams));
            Assert.Empty(((TensorData)loaded.OptimizerState.Fields[stepField.Name]).Shape.Dims);
        }
        finally { if (File.Exists(adamPath)) File.Delete(adamPath); }
    }

    /// <summary>Runs one save and holds it to the bytes and the phases. That the measurement covers
    /// the content production is pinned by
    /// <see cref="CoreUtilsCoverageTests.TestAtomicFileWriterReportsWhatEachWriteCost"/> against an
    /// injected delay, which is the direction contention cannot break.</summary>
    private static SaveReport Saved(Func<SaveReport> save, string path)
    {
        var clock = System.Diagnostics.Stopwatch.StartNew();
        var r = save();
        var outer = clock.Elapsed;
        Assert.Equal(new FileInfo(path).Length, r.BytesWritten);
        Assert.True(r.Write > TimeSpan.Zero && r.Flush > TimeSpan.Zero && r.Commit > TimeSpan.Zero);
        Assert.True(r.Elapsed <= outer);
        return r;
    }

    [Fact]
    public void TestEveryCheckpointSaveReportsItsBytesAndWhereItsTimeWent()
    {
        var rig = AdamRig();
        var ckpt = SteppedOnce(rig);

        var flat = TempPath("save_report") + ".safetensors";
        var narrow = TempPath("save_report_narrow") + ".safetensors";
        var skpt = TempPath("save_report") + ".skpt";
        try
        {
            ckpt.Save(flat);
            Persistence.SaveTrainingCheckpointToSkpt(ckpt, skpt);

            var full = Saved(() => ckpt.Save(flat), flat);
            var weightsOnly = Saved(() => ckpt.Save(narrow, CheckpointComponents.InferenceState), narrow);
            Assert.True(weightsOnly.BytesWritten < full.BytesWritten);

            Saved(() => Persistence.SaveTrainingCheckpoint(ckpt, flat), flat);
            Saved(() => Persistence.SaveTrainingCheckpointToSkpt(ckpt, skpt), skpt);
            Saved(() => Persistence.ForTrainingCheckpoint(ckpt).Save(skpt), skpt);

            Assert.Equal(FlattenStruct(ckpt.OptimizerState),
                FlattenStruct(rig.LoadCheckpoint(flat).OptimizerState));
            Assert.Equal(FlattenStruct(ckpt.OptimizerState),
                FlattenStruct(rig.LoadCheckpointFromSkpt(skpt).OptimizerState));
        }
        finally
        {
            string[] written = [flat, narrow, skpt];
            foreach (var p in written) if (File.Exists(p)) File.Delete(p);
        }
    }

    [Fact]
    public void TestSavingAParameterSetWritesItsStorageWithoutCopyingIt()
    {
        var big = TensorData([2L << 20], new float[2 << 20]);
        var scalar = TensorData(Array.Empty<long>(), 7f);
        List<SafeTensor> tensors =
        [
            new SafeTensor("w", big, SafeTensorLoader.DTypeToSafeTensorDType(big.DType), big.Shape.Dims),
            new SafeTensor("s", scalar, SafeTensorLoader.DTypeToSafeTensorDType(scalar.DType), scalar.Shape.Dims),
        ];

        SafeTensorLoader.SaveSafeTensorsToStream(Stream.Null, tensors);
        var before = GC.GetAllocatedBytesForCurrentThread();
        SafeTensorLoader.SaveSafeTensorsToStream(Stream.Null, tensors);
        Assert.True(GC.GetAllocatedBytesForCurrentThread() - before < big.AccessRawMemory().Length / 128);

        using var buffer = new MemoryStream();
        SafeTensorLoader.SaveSafeTensorsToStream(buffer, tensors);
        var read = SafeTensorLoader.ParseSafeTensorBytes(buffer.ToArray());
        Assert.Equal(["w", "s"], read.Select(t => t.Name));
        Assert.Equal(big.CopyRawMemory(), read[0].Data.CopyRawMemory());
        Assert.Equal(scalar.CopyRawMemory(), read[1].Data.CopyRawMemory());
    }

    [Fact]
    public void TestCheckpointSaveAtomicityAndTruncatedLoadFailsLoudlyCoverage()
    {
        var rig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            SGDOptimizer.ComputationGraph,
            [
                new TensorDataModelParam("input", ModelParamType.InputParam,
                    TensorData([4L], [1f, 2f, 3f, 4f])),
            ],
            0.1f);

        var truncPath = TempPath("ckpt_trunc") + ".safetensors";
        try
        {
            rig.CreateInitialCheckpoint().Save(truncPath);
            var full = File.ReadAllBytes(truncPath);
            File.WriteAllBytes(truncPath, full[..^8]);

            var ex = Assert.Throws<ModelException>(() => rig.LoadCheckpoint(truncPath));
            Assert.Equal(ErrorCodes.ST003, ex.ErrorCode);
            Assert.Contains("truncated", ex.Message);
            Assert.Contains(truncPath, ex.Message);
            Assert.Contains($"{full.Length} bytes", ex.Message);
            Assert.Contains($"{full.Length - 8} bytes", ex.Message);
        }
        finally { if (File.Exists(truncPath)) File.Delete(truncPath); }

        var ckptV1 = rig.CreateInitialCheckpoint();
        var ckptV2 = new TrainingCheckpoint
        {
            TrainableParams = ckptV1.TrainableParams,
            ModelState = ckptV1.ModelState,
            OptimizerState = ckptV1.OptimizerState,
            Step = 7,
        };

        var dir = TempPath("ckpt_atomic");
        Directory.CreateDirectory(dir);
        try
        {
            var path = Path.Combine(dir, "ckpt.safetensors");
            ckptV1.Save(path);
            Assert.Equal(0, rig.LoadCheckpoint(path).Step);

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

            var stale = Path.Combine(dir, $".tmp-ckpt.safetensors-{Guid.NewGuid():N}");
            File.WriteAllText(stale, "partial");
            ckptV2.Save(path);
            Assert.Equal(7, rig.LoadCheckpoint(path).Step);
            Assert.False(File.Exists(stale));
            Assert.Empty(Directory.GetFileSystemEntries(dir, ".tmp-*"));

            Assert.Throws<DirectoryNotFoundException>(
                () => ckptV1.Save(Path.Combine(dir, "missing", "ckpt.safetensors")));
            Assert.False(Directory.Exists(Path.Combine(dir, "missing")));
        }
        finally { Directory.Delete(dir, recursive: true); }
    }

    [Fact]
    public void TestCheckpointInspectRecognizesSavedCheckpoint()
    {
        var rig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            SGDMomentumOptimizer.ComputationGraph,
            [
                new TensorDataModelParam("input", ModelParamType.InputParam,
                    TensorData([4L], [1f, 2f, 3f, 4f])),
            ],
            0.5f, 0.9f);
        var ckpt0 = rig.CreateInitialCheckpoint();
        var ckpt = new TrainingCheckpoint
        {
            TrainableParams = ckpt0.TrainableParams,
            ModelState = ckpt0.ModelState,
            OptimizerState = ckpt0.OptimizerState,
            Step = 5,
        };

        var path = TempPath("inspect") + ".safetensors";
        try
        {
            ckpt.Save(path);
            var result = Persistence.Inspect(path);

            Assert.Equal(ArtifactKind.TrainingCheckpoint, result.Kind);
            Assert.Empty(result.Observations);
            Assert.NotNull(result.SafeTensors);
            Assert.Null(result.Srk);

            var info = result.TrainingCheckpoint!;
            Assert.Equal(1, info.FormatVersion);
            Assert.Equal(5, info.Step);
            Assert.Null(info.Epoch);
            Assert.Null(info.BatchIndex);

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

            var trainableField = rig.TrainableParamStructDef.Fields[0];
            var written = (TensorData)ckpt.TrainableParams.Fields[trainableField.Name];
            var listed = info.Sections["trainable"].Single(t => t.Name == trainableField.Name);
            Assert.Equal(written.Shape.Dims, listed.Shape);
            Assert.Equal("F32", listed.DType);

            var text = result.ToString();
            Assert.Contains("training checkpoint", text);
            Assert.Contains("global step: 5", text);

            var plainPath = TempPath("inspect_plain") + ".safetensors";
            try
            {
                List<SafeTensor> plainTensors =
                [
                    new SafeTensor(trainableField.Name, written,
                        SafeTensorLoader.DTypeToSafeTensorDType(written.DType), written.Shape.Dims),
                ];
                SafeTensorLoader.SaveSafeTensors(plainPath, plainTensors);
                Assert.Equal(ArtifactKind.SafeTensors, Persistence.Inspect(plainPath).Kind);
                Assert.Null(Persistence.Inspect(plainPath).TrainingCheckpoint);
            }
            finally { if (File.Exists(plainPath)) File.Delete(plainPath); }
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    [Fact]
    public void TestCheckpointCountersPersistAcrossFormatsCoverage()
    {
        var (_, trained, _, _) = BuildTrainedAdamWRig(steps: 4);
        var loaderRig = BuildTrainedAdamWRig(steps: 0).Rig;

        long bigStep = 5_000_000_000L;
        long bigEpoch = 3_000_000_000L;
        long bigBatch = (long)int.MaxValue + 7L;
        var big = new TrainingCheckpoint
        {
            TrainableParams = trained.TrainableParams,
            ModelState = trained.ModelState,
            OptimizerState = trained.OptimizerState,
            Step = bigStep, Epoch = bigEpoch, BatchIndex = bigBatch, Rig = trained.Rig,
        };

        var bigFlat = TempPath("i64") + ".safetensors";
        var bigSkpt = TempPath("i64") + ".skpt";
        try
        {
            big.Save(bigFlat);
            var flat = loaderRig.LoadCheckpoint(bigFlat);
            Assert.Equal(bigStep, flat.Step);
            Assert.Equal(bigEpoch, flat.Epoch);
            Assert.Equal(bigBatch, flat.BatchIndex);
            Assert.Equal(1, Persistence.Inspect(bigFlat).TrainingCheckpoint!.FormatVersion);

            Persistence.SaveTrainingCheckpointToSkpt(big, bigSkpt);
            var skpt = loaderRig.LoadCheckpointFromSkpt(bigSkpt);
            Assert.Equal(bigStep, skpt.Step);
            Assert.Equal(bigEpoch, skpt.Epoch);
            Assert.Equal(bigBatch, skpt.BatchIndex);
        }
        finally
        {
            if (File.Exists(bigFlat)) File.Delete(bigFlat);
            if (File.Exists(bigSkpt)) File.Delete(bigSkpt);
        }

        var ckpt = new TrainingCheckpoint
        {
            TrainableParams = trained.TrainableParams,
            ModelState = trained.ModelState,
            OptimizerState = trained.OptimizerState,
            Step = trained.Step, Epoch = 7, BatchIndex = 340, Rig = trained.Rig,
        };
        Assert.Equal(4, ckpt.Step);
        Assert.Equal(7, ckpt.Epoch);
        Assert.Equal(340, ckpt.BatchIndex);

        var skptPath = TempPath("ctr_skpt") + ".skpt";
        var flatPath = TempPath("ctr_flat") + ".safetensors";
        try
        {
            Persistence.SaveTrainingCheckpointToSkpt(ckpt, skptPath);
            var manifest = SkptFileFormat.ParseManifest(
                ReadEntryBytesViaBcl(skptPath, SkptFileFormat.ConfigEntryName), skptPath);
            Assert.Equal(7, manifest.Training!.Epoch);
            Assert.Equal(340, manifest.Training.BatchIndex);

            var skptLoaded = loaderRig.LoadCheckpointFromSkpt(skptPath);
            Assert.Equal(4, skptLoaded.Step);
            Assert.Equal(7, skptLoaded.Epoch);
            Assert.Equal(340, skptLoaded.BatchIndex);

            ckpt.Save(flatPath);
            var flatLoaded = loaderRig.LoadCheckpoint(flatPath);
            Assert.Equal(4, flatLoaded.Step);
            Assert.Equal(7, flatLoaded.Epoch);
            Assert.Equal(340, flatLoaded.BatchIndex);

            Assert.Equal(skptLoaded.Step, flatLoaded.Step);
            Assert.Equal(skptLoaded.Epoch, flatLoaded.Epoch);
            Assert.Equal(skptLoaded.BatchIndex, flatLoaded.BatchIndex);

            var skptInspect = Persistence.Inspect(skptPath);
            Assert.Empty(skptInspect.Observations);
            Assert.Equal(7, skptInspect.Skpt!.Training!.Epoch);
            Assert.Equal(340, skptInspect.Skpt.Training.BatchIndex);
            Assert.Contains("epoch 7", skptInspect.ToString());
            Assert.Contains("batch index 340", skptInspect.ToString());

            var flatInspect = Persistence.Inspect(flatPath);
            Assert.Empty(flatInspect.Observations);
            Assert.Equal(1, flatInspect.TrainingCheckpoint!.FormatVersion);
            Assert.Equal(7, flatInspect.TrainingCheckpoint.Epoch);
            Assert.Equal(340, flatInspect.TrainingCheckpoint.BatchIndex);
            Assert.Contains("epoch: 7", flatInspect.ToString());
            Assert.Contains("batch index: 340", flatInspect.ToString());
        }
        finally
        {
            if (File.Exists(skptPath)) File.Delete(skptPath);
            if (File.Exists(flatPath)) File.Delete(flatPath);
        }

        var unset = new TrainingCheckpoint
        {
            TrainableParams = trained.TrainableParams,
            ModelState = trained.ModelState,
            OptimizerState = trained.OptimizerState,
            Step = trained.Step, Rig = trained.Rig,
        };
        Assert.Null(unset.Epoch);
        Assert.Null(unset.BatchIndex);

        var nullFlat = TempPath("nullctr") + ".safetensors";
        var nullSkpt = TempPath("nullctr") + ".skpt";
        try
        {
            unset.Save(nullFlat);
            var flat = loaderRig.LoadCheckpoint(nullFlat);
            Assert.Equal(trained.Step, flat.Step);
            Assert.Null(flat.Epoch);
            Assert.Null(flat.BatchIndex);

            var flatInspect = Persistence.Inspect(nullFlat);
            Assert.Empty(flatInspect.Observations);
            Assert.Equal(1, flatInspect.TrainingCheckpoint!.FormatVersion);
            Assert.Null(flatInspect.TrainingCheckpoint.Epoch);
            Assert.Null(flatInspect.TrainingCheckpoint.BatchIndex);
            Assert.Contains("epoch: unset", flatInspect.ToString());

            Persistence.SaveTrainingCheckpointToSkpt(unset, nullSkpt);
            var manifest = SkptFileFormat.ParseManifest(
                ReadEntryBytesViaBcl(nullSkpt, SkptFileFormat.ConfigEntryName), nullSkpt);
            Assert.Null(manifest.Training!.Epoch);
            Assert.Null(manifest.Training.BatchIndex);

            var skptLoaded = loaderRig.LoadCheckpointFromSkpt(nullSkpt);
            Assert.Equal(trained.Step, skptLoaded.Step);
            Assert.Null(skptLoaded.Epoch);
            Assert.Null(skptLoaded.BatchIndex);
        }
        finally
        {
            if (File.Exists(nullFlat)) File.Delete(nullFlat);
            if (File.Exists(nullSkpt)) File.Delete(nullSkpt);
        }

        const int features = 4;
        var rig = LoaderRig(batchSize: 2, features);
        var (inputs, targets) = IndexDataset(rig, n: 6, features);
        var final = rig.Fit(new InMemoryDataLoader(inputs, targets, batchSize: 2), numEpochs: 1).FinalCheckpoint;
        Assert.Equal(0, final.Epoch);
        Assert.Equal(2, final.BatchIndex);

        var concretePath = TempPath("concctr") + ".safetensors";
        try
        {
            final.Save(concretePath);
            var loaded = LoaderRig(batchSize: 2, features).LoadCheckpoint(concretePath);
            Assert.Equal(0, loaded.Epoch);
            Assert.Equal(2, loaded.BatchIndex);

            var inspect = Persistence.Inspect(concretePath);
            Assert.Empty(inspect.Observations);
            Assert.Equal(0, inspect.TrainingCheckpoint!.Epoch);
            Assert.Equal(2, inspect.TrainingCheckpoint.BatchIndex);
        }
        finally { if (File.Exists(concretePath)) File.Delete(concretePath); }
    }

    [Fact]
    public void TestFlatCheckpointLoadsCoverage()
    {
        var (_, ckpt, _, _) = BuildTrainedAdamWRig(steps: 2);
        var path = TempPath("flat") + ".safetensors";
        try
        {
            Persistence.SaveTrainingCheckpoint(ckpt, path);
            Assert.Equal(ArtifactKind.TrainingCheckpoint, Persistence.Inspect(path).Kind);

            var loaded = BuildTrainedAdamWRig(steps: 0).Rig.LoadCheckpoint(path);
            Assert.Equal(2, loaded.Step);
            Assert.Equal(FlattenStruct(ckpt.TrainableParams), FlattenStruct(loaded.TrainableParams));
            Assert.Equal(FlattenStruct(ckpt.OptimizerState), FlattenStruct(loaded.OptimizerState));
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    [Fact]
    public void TestCheckpointCarriesRigLossAndInitialFactoryCoverage()
    {
        var (sample, input, target) = ScalarMultiplyBatches();
        var rig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            sample, 0.1f);

        var initial = rig.CreateInitialCheckpoint();
        Assert.NotNull(initial.TrainableParams);
        Assert.Same(rig, initial.Rig);
        Assert.Null(initial.Loss);

        var stepped = rig.TrainStep(initial, input.Shared(), target.Shared());
        Assert.Same(rig, stepped.Rig);
        Assert.NotNull(stepped.Loss);
        Assert.True(float.IsFinite(stepped.Loss!.Value));

        var moved = stepped.WithStep(42);
        Assert.Same(rig, moved.Rig);
        Assert.Equal(stepped.Loss, moved.Loss);
        Assert.Equal(stepped.Loss, stepped.WithEpoch(3).Loss);

        var bare = new TrainingCheckpoint
        {
            TrainableParams = initial.TrainableParams,
            ModelState = initial.ModelState,
            OptimizerState = initial.OptimizerState,
        };
        Assert.Null(bare.Rig);

        var runtimeRig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, InitFromHyperOptimizer.ComputationGraph,
            sample, new InitFromHyperOptimizerHyperparameters { LearningRate = Hyperparameter.Runtime() });
        Assert.NotNull(runtimeRig.CreateInitialCheckpoint(runtimeRig.MakeHyperparameters(0.3f)).OptimizerState);
    }

    [Fact]
    public void TestAdoptCheckpointCoverage()
    {
        var (sample, _, _) = ScalarMultiplyBatches();
        var rig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            sample, 0.1f);
        var seed = rig.CreateInitialCheckpoint();

        var bare = new TrainingCheckpoint
        {
            TrainableParams = seed.TrainableParams,
            ModelState = seed.ModelState,
            OptimizerState = seed.OptimizerState,
            Step = 5, Epoch = 2, BatchIndex = 1,
        };
        Assert.Null(bare.Rig);
        Assert.Throws<InvalidOperationException>(() => bare.ToInferenceModel());

        var adopted = rig.AdoptCheckpoint(bare);
        Assert.NotSame(bare, adopted);
        Assert.Same(rig, adopted.Rig);
        Assert.Equal(5, adopted.Step);
        Assert.Equal(2, adopted.Epoch);
        Assert.Equal(1, adopted.BatchIndex);
        Assert.Null(bare.Rig);
        Assert.NotNull(adopted.ToInferenceModel());

        var bnRig = TrainingRig.FromScratch(
            ScalarMultiplyWithBatchNormModel.ComputationGraph, L2Loss.ComputationGraph,
            SGDMomentumOptimizer.ComputationGraph,
            [new TensorDataModelParam("input", ModelParamType.InputParam, TensorData([8L], new float[8]))],
            0.5f, 0.9f);
        Assert.Throws<ArgumentException>(() => bnRig.AdoptCheckpoint(bare));
    }

    [Fact]
    public void TestSaveLoadComponentsSubsetCoverage()
    {
        var (rigA, trained, _, _) = BuildTrainedAdamWRig(steps: 3);
        Assert.NotEmpty(trained.OptimizerState.Fields);
        var initialOpt = FlattenStruct(rigA.CreateInitialCheckpoint().OptimizerState);
        Assert.NotEqual(FlattenStruct(trained.OptimizerState), initialOpt);

        var path = TempPath("subset") + ".safetensors";
        try
        {
            trained.Save(path, CheckpointComponents.InferenceState);

            var rigB = BuildTrainedAdamWRig(steps: 0).Rig;
            var loaded = rigB.LoadCheckpoint(path);
            Assert.Same(rigB, loaded.Rig);
            Assert.Equal(FlattenStruct(trained.TrainableParams), FlattenStruct(loaded.TrainableParams));
            Assert.Equal(0, loaded.Step);
            Assert.Equal(initialOpt, FlattenStruct(loaded.OptimizerState));

            var ex = Assert.Throws<NotSupportedException>(
                () => trained.Save(path, CheckpointComponents.All));
            Assert.Contains("#115", ex.Message);

            var loadRigEx = Assert.Throws<NotSupportedException>(
                () => rigB.LoadCheckpoint(path, CheckpointComponents.TrainingRig));
            Assert.Contains("#115", loadRigEx.Message);
            Assert.Throws<NotSupportedException>(() => rigB.LoadCheckpoint(path, CheckpointComponents.All));
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    [Fact]
    public void TestCheckpointLossPersistsCoverage()
    {
        var (rigA, trained, _, _) = BuildTrainedAdamWRig(steps: 3);
        Assert.NotNull(trained.Loss);
        float loss = trained.Loss!.Value;
        Assert.True(trained.Step > 0);
        var initial = rigA.CreateInitialCheckpoint();
        Assert.Null(initial.Loss);
        var reader = BuildTrainedAdamWRig(steps: 0).Rig;

        var flatPath = TempPath("loss_flat") + ".safetensors";
        var flatInitPath = TempPath("loss_flatinit") + ".safetensors";
        var flatCountersOnlyPath = TempPath("loss_flatco") + ".safetensors";
        var flatLossOnlyPath = TempPath("loss_flatlo") + ".safetensors";
        var flatNullLossReqPath = TempPath("loss_flatnull") + ".safetensors";
        var skptPath = TempPath("loss_skpt") + ".skpt";
        var skptInitPath = TempPath("loss_skptinit") + ".skpt";
        try
        {
            trained.Save(flatPath);
            var full = reader.LoadCheckpoint(flatPath);
            Assert.Equal(loss, full.Loss!.Value);
            Assert.Equal(trained.Step, full.Step);

            initial.Save(flatInitPath);
            Assert.Null(reader.LoadCheckpoint(flatInitPath).Loss);

            trained.Save(flatCountersOnlyPath,
                CheckpointComponents.InferenceState | CheckpointComponents.Counters);
            var countersOnly = reader.LoadCheckpoint(flatCountersOnlyPath);
            Assert.Equal(trained.Step, countersOnly.Step);
            Assert.Null(countersOnly.Loss);

            trained.Save(flatLossOnlyPath,
                CheckpointComponents.InferenceState | CheckpointComponents.Loss);
            var lossOnly = reader.LoadCheckpoint(flatLossOnlyPath);
            Assert.Equal(loss, lossOnly.Loss!.Value);
            Assert.Equal(0, lossOnly.Step);

            initial.Save(flatNullLossReqPath, CheckpointComponents.InferenceState | CheckpointComponents.Loss);
            Assert.Null(reader.LoadCheckpoint(flatNullLossReqPath).Loss);

            Persistence.SaveTrainingCheckpointToSkpt(trained, skptPath);
            Assert.Equal(loss, reader.LoadCheckpointFromSkpt(skptPath).Loss!.Value);
            var skptNoLoss = reader.LoadCheckpointFromSkpt(
                skptPath, CheckpointComponents.InferenceState | CheckpointComponents.OptimizerState | CheckpointComponents.Counters);
            Assert.Equal(trained.Step, skptNoLoss.Step);
            Assert.Null(skptNoLoss.Loss);

            Persistence.SaveTrainingCheckpointToSkpt(initial, skptInitPath);
            Assert.Null(reader.LoadCheckpointFromSkpt(skptInitPath).Loss);
        }
        finally
        {
            string[] paths =
                [flatPath, flatInitPath, flatCountersOnlyPath, flatLossOnlyPath, flatNullLossReqPath, skptPath, skptInitPath];
            foreach (var p in paths)
                if (File.Exists(p)) File.Delete(p);
        }
    }
}

[Trait("Domain", "Training")]
[Trait("Purpose", "Coverage")]
public class TrainingRigSkptCheckpointCoverageTests
{
    private static int IndexOfSubsequence(byte[] haystack, byte[] needle)
    {
        if (needle.Length == 0 || haystack.Length < needle.Length) return -1;
        for (int i = 0; i <= haystack.Length - needle.Length; i++)
        {
            int j = 0;
            while (j < needle.Length && haystack[i + j] == needle[j]) j++;
            if (j == needle.Length) return i;
        }
        return -1;
    }

    private static void RewriteSkptManifest(string path, Action<System.Text.Json.Nodes.JsonNode> edit)
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
                    edit(node);
                    data = System.Text.Encoding.UTF8.GetBytes(node.ToJsonString());
                }
                entries.Add(new SkptFileFormat.ZipEntrySpec(e.FullName, data, Align: false));
            }
        using var outStream = File.Create(path);
        SkptFileFormat.WriteStoredZip(outStream, entries, DateTime.UtcNow);
    }

    [Fact]
    public void TestSkptCheckpointRoundTripResumeModelStateAndInspectCoverage()
    {
        var (rigA, ckpt, inBatch, outBatch) = BuildTrainedAdamWRig(steps: 2);
        Assert.Equal(2, ckpt.Step);
        Assert.NotEmpty(ckpt.OptimizerState.Fields);
        Assert.Empty(ckpt.ModelState.Fields);

        var reference = rigA.TrainStep(ckpt.Shared(), inBatch.Shared(), outBatch.Shared());

        var path = TempPath("skpt_ckpt") + ".skpt";
        try
        {
            Persistence.SaveTrainingCheckpointToSkpt(ckpt, path);
            Assert.True(File.Exists(path));

            using (var zip = System.IO.Compression.ZipFile.OpenRead(path))
            {
                var names = zip.Entries.Select(e => e.FullName).OrderBy(n => n, StringComparer.Ordinal).ToArray();
                string[] expected =
                [
                    SkptFileFormat.ConfigEntryName,
                    SkptFileFormat.OptimizerStateEntryPath,
                    SkptFileFormat.TrainableEntryPath,
                    SkptFileFormat.ModelEntryPath,
                    SkptFileFormat.ArchEntryPath,
                    SkptFileFormat.LossEntryPath,
                    SkptFileFormat.OptimizerEntryPath,
                ];
                Assert.Equal(expected.OrderBy(n => n, StringComparer.Ordinal), names);
                Assert.All(zip.Entries, e => Assert.Equal(e.Length, e.CompressedLength));
                Assert.DoesNotContain(SkptFileFormat.ModelStateEntryPath, names);
                Assert.DoesNotContain(SkptFileFormat.SchedulerEntryPath, names);
            }

            var manifest = SkptFileFormat.ParseManifest(
                ReadEntryBytesViaBcl(path, SkptFileFormat.ConfigEntryName), path);
            Assert.NotNull(manifest.Training);
            Assert.Equal(SkptFileFormat.TrainingCheckpointVersion, manifest.Training!.CheckpointVersion);
            Assert.Equal(2, manifest.Training.Step);
            Assert.True(manifest.Training.AdditionalFields is null
                || !manifest.Training.AdditionalFields.ContainsKey("kinds"));
            var modelTensors = manifest.TensorMappings!["model"]["default"].Tensors!;
            var optTensors = manifest.TensorMappings["optimizer"]["default"].Tensors!;
            Assert.All(modelTensors.Values, r => Assert.Equal("trainable", r.Data));
            Assert.Equal(3 * modelTensors.Count, optTensors.Count);
            Assert.All(optTensors, kv =>
            {
                Assert.True(SkptFileFormat.TryParseOptimizerStateId(kv.Key, out var paramId, out _));
                Assert.Contains(paramId, modelTensors.Keys);
                Assert.Equal("optimizer_state", kv.Value.Data);
            });

            var rigB = BuildTrainedAdamWRig(steps: 0).Rig;
            var loaded = rigB.LoadCheckpointFromSkpt(path);
            Assert.Equal(2, loaded.Step);
            Assert.Equal(FlattenStruct(ckpt.TrainableParams), FlattenStruct(loaded.TrainableParams));
            Assert.Equal(FlattenStruct(ckpt.OptimizerState), FlattenStruct(loaded.OptimizerState));
            Assert.Empty(loaded.ModelState.Fields);

            var resumed = rigB.TrainStep(loaded.Shared(), inBatch.Shared(), outBatch.Shared());
            Assert.Equal(3, resumed.Step);
            Assert.Equal(reference.Loss!.Value, resumed.Loss!.Value);
            Assert.Equal(FlattenStruct(reference.TrainableParams), FlattenStruct(resumed.TrainableParams));
            Assert.Equal(FlattenStruct(reference.OptimizerState), FlattenStruct(resumed.OptimizerState));

            var inferenceModel = Persistence.Load(path);
            Assert.Equal(GraphKind.ConcreteModel, inferenceModel.Kind);
            var probe = TensorData(ScalarInputShape, [5f, 6f, 7f, 8f]);
            var loadedOut = ComputeContext.Default.Execute(inferenceModel, probe.Shared())[0].ToTensorData().As<float32>().AccessMemory().ToArray();
            var ckptOut = ComputeContext.Default.Execute(ckpt.ToInferenceModel(), probe)[0].ToTensorData().As<float32>().AccessMemory().ToArray();
            Assert.Equal(ckptOut, loadedOut);

            var inspect = Persistence.Inspect(path);
            Assert.Equal(ArtifactKind.SkptCheckpoint, inspect.Kind);
            Assert.Empty(inspect.Observations);
            Assert.NotNull(inspect.Skpt);
            var training = inspect.Skpt!.Training;
            Assert.NotNull(training);
            Assert.Equal(SkptFileFormat.TrainingCheckpointVersion, training!.CheckpointVersion);
            Assert.Equal(2, training.Step);
            var text = inspect.ToString();
            Assert.Contains("training checkpoint: version 1", text);
            Assert.Contains("global step 2", text);
        }
        finally { if (File.Exists(path)) File.Delete(path); }

        NamedModelParam[] bnSample =
        [
            new TensorDataModelParam("input", ModelParamType.InputParam, TensorData([8L], new float[8])),
        ];
        TrainingRig BnRig() => TrainingRig.FromScratch(
            ScalarMultiplyWithBatchNormModel.ComputationGraph, L2Loss.ComputationGraph,
            SGDMomentumOptimizer.ComputationGraph, bnSample, 0.5f, 0.9f);

        var bnRig = BnRig();
        var bnSeed = bnRig.CreateInitialCheckpoint();
        var bnCkpt = new TrainingCheckpoint
        {
            TrainableParams = bnSeed.TrainableParams,
            ModelState = bnSeed.ModelState,
            OptimizerState = bnSeed.OptimizerState,
            Step = 11, Rig = bnRig,
        };
        Assert.NotEmpty(bnCkpt.ModelState.Fields);

        var bnPath = TempPath("skpt_bn") + ".skpt";
        try
        {
            Persistence.SaveTrainingCheckpointToSkpt(bnCkpt, bnPath);

            using (var zip = System.IO.Compression.ZipFile.OpenRead(bnPath))
                Assert.Contains(SkptFileFormat.ModelStateEntryPath, zip.Entries.Select(e => e.FullName));

            var bnLoaded = BnRig().LoadCheckpointFromSkpt(bnPath);
            Assert.Equal(11, bnLoaded.Step);
            Assert.Equal(FlattenStruct(bnCkpt.ModelState), FlattenStruct(bnLoaded.ModelState));
            Assert.Equal(FlattenStruct(bnCkpt.TrainableParams), FlattenStruct(bnLoaded.TrainableParams));
            Assert.Equal(FlattenStruct(bnCkpt.OptimizerState), FlattenStruct(bnLoaded.OptimizerState));
        }
        finally { if (File.Exists(bnPath)) File.Delete(bnPath); }
    }

    [Fact]
    public void TestSkptCheckpointFailsLoudLenientManifestAndComponentSubsetCoverage()
    {
        var (rigA, trained, _, _) = BuildTrainedAdamWRig(steps: 3);
        var rig = BuildTrainedAdamWRig(steps: 0).Rig;
        var one = new TrainingCheckpoint
        {
            TrainableParams = trained.TrainableParams,
            ModelState = trained.ModelState,
            OptimizerState = trained.OptimizerState,
            Step = 1, Rig = trained.Rig,
        };

        var path = TempPath("skpt_fail") + ".skpt";
        var tampered = TempPath("skpt_tamper") + ".skpt";
        try
        {
            Persistence.ForTrainingCheckpoint(one)
                .WithZstdCompressedData()
                .WithMetadata(runName: "skpt-95-run", gitCommit: "abc123")
                .Save(path);

            var inspect = Persistence.Inspect(path);
            Assert.Empty(inspect.Observations);
            Assert.Equal("skpt-95-run", inspect.Skpt!.UserMetadata!["runName"]);
            Assert.Contains(inspect.Skpt.DataEntries,
                d => d.Key == SkptFileFormat.TrainableDataKey && d.Compression == SkptFileFormat.CompressionZstd);
            Assert.NotNull(inspect.Skpt.Training);
            Assert.Equal(1, inspect.Skpt.Training!.Step);

            var loaded = rig.LoadCheckpointFromSkpt(path);
            Assert.Equal(1, loaded.Step);
            Assert.Equal(FlattenStruct(one.TrainableParams), FlattenStruct(loaded.TrainableParams));

            var bnRig = TrainingRig.FromScratch(
                ScalarMultiplyWithBatchNormModel.ComputationGraph, L2Loss.ComputationGraph,
                SGDMomentumOptimizer.ComputationGraph,
                [new TensorDataModelParam("input", ModelParamType.InputParam, TensorData([8L], new float[8]))],
                0.5f, 0.9f);
            Assert.ThrowsAny<Exception>(() => bnRig.LoadCheckpointFromSkpt(path));

            TrainingRig SgdmRig() => TrainingRig.FromScratch(
                ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
                SGDMomentumOptimizer.ComputationGraph,
                [new TensorDataModelParam("input", ModelParamType.InputParam,
                    TensorData(ScalarInputShape, [1f, 2f, 3f, 4f]))],
                0.5f, 0.9f);
            var strayOpt = Assert.Throws<System.IO.InvalidDataException>(() => SgdmRig().LoadCheckpointFromSkpt(path));
            Assert.Contains("optimizer", strayOpt.Message);

            var sgdmPath = TempPath("skpt_sgdm") + ".skpt";
            try
            {
                Persistence.SaveTrainingCheckpointToSkpt(SgdmRig().CreateInitialCheckpoint(), sgdmPath);
                var missingOpt = Assert.Throws<System.IO.InvalidDataException>(() => rig.LoadCheckpointFromSkpt(sgdmPath));
                Assert.Contains("optimizer-state", missingOpt.Message);
            }
            finally { if (File.Exists(sgdmPath)) File.Delete(sgdmPath); }

            var nullRefPath = TempPath("skpt_nullref") + ".skpt";
            try
            {
                Persistence.SaveTrainingCheckpointToSkpt(one, nullRefPath);
                RewriteSkptManifest(nullRefPath, n =>
                {
                    var tensors = n["tensorMappings"]!["model"]!["default"]!["tensors"]!.AsObject();
                    tensors[tensors.First().Key] = null;
                });
                var nullEx = Assert.Throws<System.IO.InvalidDataException>(() => rig.LoadCheckpointFromSkpt(nullRefPath));
                Assert.Contains("null reference", nullEx.Message);
            }
            finally { if (File.Exists(nullRefPath)) File.Delete(nullRefPath); }

            var bytes = File.ReadAllBytes(path);
            var entryBytes = ReadEntryBytesViaBcl(path, SkptFileFormat.TrainableEntryPath);
            int window = Math.Min(24, entryBytes.Length);
            var needle = entryBytes.Skip((entryBytes.Length - window) / 2).Take(window).ToArray();
            int at = IndexOfSubsequence(bytes, needle);
            Assert.True(at >= 0);
            bytes[at] ^= 0xFF;
            File.WriteAllBytes(tampered, bytes);
            Assert.ThrowsAny<Exception>(() => rig.LoadCheckpointFromSkpt(tampered));

            var infPath = TempPath("skpt_inf") + ".skpt";
            try
            {
                Persistence.From(one.ToInferenceModel()).WithModel().WithWeights().Save(infPath);
                var ex = Assert.Throws<System.IO.InvalidDataException>(() => TrainingRig.Load(infPath));
                Assert.Contains("training", ex.Message);
            }
            finally { if (File.Exists(infPath)) File.Delete(infPath); }
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
            if (File.Exists(tampered)) File.Delete(tampered);
        }

        var strippedPath = TempPath("skpt_old") + ".skpt";
        try
        {
            var two = new TrainingCheckpoint
            {
                TrainableParams = trained.TrainableParams,
                ModelState = trained.ModelState,
                OptimizerState = trained.OptimizerState,
                Step = 2, Rig = trained.Rig,
            };
            Persistence.SaveTrainingCheckpointToSkpt(two, strippedPath);
            RewriteSkptManifest(strippedPath, n =>
            {
                var training = n["training"]!.AsObject();
                training.Remove("epoch");
                training.Remove("batchIndex");
            });

            var manifest = SkptFileFormat.ParseManifest(
                ReadEntryBytesViaBcl(strippedPath, SkptFileFormat.ConfigEntryName), strippedPath);
            Assert.Null(manifest.Training!.Epoch);
            Assert.Null(manifest.Training.BatchIndex);

            var strippedLoaded = rig.LoadCheckpointFromSkpt(strippedPath);
            Assert.Equal(2, strippedLoaded.Step);
            Assert.Null(strippedLoaded.Epoch);
            Assert.Null(strippedLoaded.BatchIndex);

            var strippedInspect = Persistence.Inspect(strippedPath);
            Assert.Empty(strippedInspect.Observations);
            Assert.Null(strippedInspect.Skpt!.Training!.Epoch);
            Assert.Null(strippedInspect.Skpt.Training.BatchIndex);
        }
        finally { if (File.Exists(strippedPath)) File.Delete(strippedPath); }

        Assert.NotEmpty(trained.OptimizerState.Fields);
        Assert.NotNull(trained.Loss);
        var initialOpt = FlattenStruct(rigA.CreateInitialCheckpoint().OptimizerState);
        Assert.NotEqual(FlattenStruct(trained.OptimizerState), initialOpt);

        var subsetPath = TempPath("skpt_subset") + ".skpt";
        try
        {
            Persistence.SaveTrainingCheckpointToSkpt(trained, subsetPath);

            var loaded = rig.LoadCheckpointFromSkpt(subsetPath, CheckpointComponents.InferenceState);
            Assert.Same(rig, loaded.Rig);
            Assert.Equal(FlattenStruct(trained.TrainableParams), FlattenStruct(loaded.TrainableParams));
            Assert.Equal(0, loaded.Step);
            Assert.Null(loaded.Loss);
            Assert.Equal(initialOpt, FlattenStruct(loaded.OptimizerState));

            var ex = Assert.Throws<NotSupportedException>(
                () => rig.LoadCheckpointFromSkpt(subsetPath, CheckpointComponents.TrainingRig));
            Assert.Contains("#115", ex.Message);
            Assert.Throws<NotSupportedException>(() => rig.LoadCheckpointFromSkpt(subsetPath, CheckpointComponents.All));
        }
        finally { if (File.Exists(subsetPath)) File.Delete(subsetPath); }
    }

    [Fact]
    public void TestCheckpointLoadEntryPointsAreFormatExplicitCoverage()
    {
        var (_, trained, _, _) = BuildTrainedAdamWRig(steps: 1);
        var reader = BuildTrainedAdamWRig(steps: 0).Rig;
        var flatPath = TempPath("fmt_flat") + ".safetensors";
        var skptPath = TempPath("fmt_skpt") + ".skpt";
        var junkPath = TempPath("fmt_junk") + ".bin";
        try
        {
            trained.Save(flatPath);
            Persistence.SaveTrainingCheckpointToSkpt(trained, skptPath);
            File.WriteAllBytes(junkPath, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

            Assert.Equal(1, reader.LoadCheckpoint(flatPath).Step);
            Assert.Equal(1, reader.LoadCheckpointFromSkpt(skptPath).Step);
            Assert.Equal(1, Persistence.LoadTrainingCheckpoint(flatPath).Step);
            Assert.Equal(1, TrainingRig.Load(skptPath).Checkpoint.Step);

            var flatGotSkpt = Assert.Throws<System.IO.InvalidDataException>(
                () => reader.LoadCheckpoint(skptPath));
            Assert.Contains(".skpt", flatGotSkpt.Message);
            Assert.Contains("LoadCheckpointFromSkpt", flatGotSkpt.Message);
            var skptGotFlat = Assert.Throws<System.IO.InvalidDataException>(
                () => reader.LoadCheckpointFromSkpt(flatPath));
            Assert.Contains("safetensors", skptGotFlat.Message);
            Assert.Contains("LoadCheckpoint(path)", skptGotFlat.Message);
            Assert.Contains("TrainingRig.Load", Assert.Throws<System.IO.InvalidDataException>(
                () => Persistence.LoadTrainingCheckpoint(skptPath)).Message);
            Assert.Contains("rig.LoadCheckpoint", Assert.Throws<System.IO.InvalidDataException>(
                () => TrainingRig.Load(flatPath)).Message);
            Assert.Contains("neither", Assert.Throws<System.IO.InvalidDataException>(
                () => reader.LoadCheckpointFromSkpt(junkPath)).Message);
            Assert.Contains("neither", Assert.Throws<System.IO.InvalidDataException>(
                () => reader.LoadCheckpoint(junkPath)).Message);
        }
        finally
        {
            string[] paths = [flatPath, skptPath, junkPath];
            foreach (var p in paths)
                if (File.Exists(p)) File.Delete(p);
        }
    }

    [Fact]
    public void TestTrainingRigLoadFromFileAloneWithSchedulerCoverage()
    {
        var (rigA, ckpt, inBatch, outBatch) = BuildTrainedAdamWRig(steps: 3);
        var reference = rigA.TrainStep(ckpt.Shared(), inBatch.Shared(), outBatch.Shared());

        var path = TempPath("rigload") + ".skpt";
        try
        {
            Persistence.SaveTrainingCheckpointToSkpt(ckpt, path);

            var (rig2, loaded) = TrainingRig.Load(path);
            Assert.Same(rig2, loaded.Rig);
            Assert.Equal(ckpt.Step, loaded.Step);
            Assert.Equal(FlattenStruct(ckpt.TrainableParams), FlattenStruct(loaded.TrainableParams));
            Assert.Equal(FlattenStruct(ckpt.OptimizerState), FlattenStruct(loaded.OptimizerState));

            var resumed = rig2.TrainStep(loaded.Shared(), inBatch.Shared(), outBatch.Shared());
            Assert.Equal(reference.Step, resumed.Step);
            Assert.Equal(reference.Loss!.Value, resumed.Loss!.Value);
            Assert.Equal(FlattenStruct(reference.TrainableParams), FlattenStruct(resumed.TrainableParams));
            Assert.Equal(FlattenStruct(reference.OptimizerState), FlattenStruct(resumed.OptimizerState));

            var probe = TensorData(ScalarInputShape, [5f, 6f, 7f, 8f]);
            var a = ComputeContext.Default.Execute(loaded.ToInferenceModel(), probe.Shared())[0].ToTensorData().As<float32>().AccessMemory().ToArray();
            var b = ComputeContext.Default.Execute(ckpt.ToInferenceModel(), probe)[0].ToTensorData().As<float32>().AccessMemory().ToArray();
            Assert.Equal(b, a);
        }
        finally { if (File.Exists(path)) File.Delete(path); }

        var flatPath = TempPath("rigload_flat") + ".safetensors";
        var infPath = TempPath("rigload_inf") + ".skpt";
        try
        {
            ckpt.Save(flatPath);
            Assert.ThrowsAny<Exception>(() => TrainingRig.Load(flatPath));

            Persistence.From(ckpt.ToInferenceModel()).WithModel().WithWeights().Save(infPath);
            Assert.Throws<System.IO.InvalidDataException>(() => TrainingRig.Load(infPath));
        }
        finally
        {
            if (File.Exists(flatPath)) File.Delete(flatPath);
            if (File.Exists(infPath)) File.Delete(infPath);
        }

        NamedModelParam[] sample =
        [
            new TensorDataModelParam("input", ModelParamType.InputParam,
                TensorData(ScalarInputShape, [1f, 2f, 3f, 4f])),
        ];
        var cosineRig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            AdamWOptimizer.ComputationGraph, sample,
            new AdamWOptimizerHyperparameters { LearningRate = Shorokoo.Core.Training.Schedules.Cosine(0.1f, 50) });
        Assert.Equal(HyperparameterKind.Scheduled, cosineRig.Hyperparameters[0].Kind);
        var cosineCkpt = cosineRig.CreateInitialCheckpoint();
        for (int i = 0; i < 5; i++)
            cosineCkpt = cosineRig.TrainStep(cosineCkpt, inBatch.Shared(), outBatch.Shared());
        var cosineReference = cosineRig.TrainStep(cosineCkpt.Shared(), inBatch.Shared(), outBatch.Shared());

        var cosinePath = TempPath("rigload_sched") + ".skpt";
        try
        {
            Persistence.SaveTrainingCheckpointToSkpt(cosineCkpt, cosinePath);

            using (var zip = System.IO.Compression.ZipFile.OpenRead(cosinePath))
                Assert.Contains(SkptFileFormat.SchedulerEntryPath, zip.Entries.Select(e => e.FullName));

            var (rig2, loaded) = TrainingRig.Load(cosinePath);
            Assert.Equal(HyperparameterKind.Scheduled, rig2.Hyperparameters[0].Kind);
            Assert.NotNull(rig2.Hyperparameters[0].AsSchedulerModule);
            Assert.Equal(cosineCkpt.Step, loaded.Step);

            var resumed = rig2.TrainStep(loaded.Shared(), inBatch.Shared(), outBatch.Shared());
            Assert.Equal(cosineReference.Step, resumed.Step);
            Assert.Equal(cosineReference.Loss!.Value, resumed.Loss!.Value);
            Assert.Equal(FlattenStruct(cosineReference.TrainableParams), FlattenStruct(resumed.TrainableParams));
        }
        finally { if (File.Exists(cosinePath)) File.Delete(cosinePath); }

        var (stepEpochSample, stepEpochIn, stepEpochTarget) = ScalarMultiplyBatches();
        var stepEpochRig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            stepEpochSample,
            new SGDOptimizerHyperparameters
            {
                LearningRate = Hyperparameter.Scheduled(StepEpochScheduler.ComputationGraph),
            });
        var stepEpochInitial = stepEpochRig.CreateInitialCheckpoint();

        var stepEpochPath = TempPath("sched_rt") + ".skpt";
        try
        {
            Persistence.SaveTrainingCheckpointToSkpt(stepEpochInitial, stepEpochPath);

            using (var archive = System.IO.Compression.ZipFile.OpenRead(stepEpochPath))
            {
                Assert.NotNull(archive.GetEntry("models/scheduler.srk"));

                using var configStream = archive.GetEntry("config.json")!.Open();
                using var manifest = System.Text.Json.JsonDocument.Parse(configStream);
                var rigBlock = manifest.RootElement.GetProperty("training").GetProperty("rig");
                Assert.Equal("scheduler", rigBlock.GetProperty("schedulerModel").GetString());
                Assert.Contains(
                    rigBlock.GetProperty("hyperparameters").EnumerateArray(),
                    h => h.GetProperty("kind").GetString() == "scheduled");
            }

            var (reloaded, _) = TrainingRig.Load(stepEpochPath);
            Assert.NotNull(reloaded);

            foreach (var (s, e) in ((long, long)[])[(0L, 0L), (3L, 1L), (7L, 4L)])
            {
                var at = new TrainingCheckpoint
                {
                    TrainableParams = stepEpochInitial.TrainableParams,
                    ModelState = stepEpochInitial.ModelState,
                    OptimizerState = stepEpochInitial.OptimizerState,
                    Step = s, Epoch = e,
                };
                float wOriginal = Weight(stepEpochRig, stepEpochRig.TrainStep(at.Shared(), stepEpochIn.Shared(), stepEpochTarget.Shared()));
                float wReloaded = Weight(reloaded, reloaded.TrainStep(at.Shared(), stepEpochIn.Shared(), stepEpochTarget.Shared()));
                Assert.True(MathF.Abs(wOriginal - wReloaded) < 1e-6f);
            }
        }
        finally { if (File.Exists(stepEpochPath)) File.Delete(stepEpochPath); }
    }

    [Fact]
    public void TestSkptDirectoryFormTrainingCheckpointRoundTripCoverage()
    {
        var (rig, ckpt, inBatch, outBatch) = BuildTrainedAdamWRig(steps: 2);
        var reference = rig.TrainStep(ckpt.Shared(), inBatch.Shared(), outBatch.Shared());
        var dirPath = TempPath("skpt_ckpt_dir") + ".skpt";
        var packedPath = TempPath("skpt_ckpt_packed") + ".skpt";
        try
        {
            Persistence.ForTrainingCheckpoint(ckpt).SaveAsDirectory(dirPath);
            Assert.True(File.Exists(Path.Combine(dirPath, SkptFileFormat.ConfigEntryName)));
            Assert.True(File.Exists(Path.Combine(dirPath, SkptFileFormat.TrainableEntryPath)));

            var (rig2, loaded) = TrainingRig.Load(dirPath);
            Assert.Equal(2, loaded.Step);
            Assert.Equal(FlattenStruct(ckpt.TrainableParams), FlattenStruct(loaded.TrainableParams));
            Assert.Equal(FlattenStruct(ckpt.OptimizerState), FlattenStruct(loaded.OptimizerState));
            var resumed = rig2.TrainStep(loaded.Shared(), inBatch.Shared(), outBatch.Shared());
            Assert.Equal(FlattenStruct(reference.TrainableParams), FlattenStruct(resumed.TrainableParams));

            Assert.Equal(2, rig.LoadCheckpointFromSkpt(dirPath).Step);
            Assert.Equal(GraphKind.ConcreteModel, Persistence.Load(dirPath).Kind);

            Persistence.PackSkpt(dirPath, packedPath);
            var (_, packedLoaded) = TrainingRig.Load(packedPath);
            Assert.Equal(FlattenStruct(loaded.TrainableParams), FlattenStruct(packedLoaded.TrainableParams));
            Assert.Equal(2, packedLoaded.Step);
        }
        finally
        {
            if (Directory.Exists(dirPath)) Directory.Delete(dirPath, recursive: true);
            if (File.Exists(packedPath)) File.Delete(packedPath);
        }
    }

    [Fact]
    public void TestEvaluationModelLoadsFromASkptWithNoRigAndScoresWhatTheRigWouldCoverage()
    {
        var (rig, ckpt, inBatch, outBatch) = BuildTrainedAdamWRig(steps: 2);
        var expected = rig.TrainStep(ckpt.Shared(), inBatch.Shared(), outBatch.Shared()).Loss!.Value;
        var path = TempPath("skpt_eval") + ".skpt";
        var dirPath = TempPath("skpt_eval_dir") + ".skpt";
        try
        {
            Persistence.SaveTrainingCheckpointToSkpt(ckpt, path);
            Persistence.ForTrainingCheckpoint(ckpt).SaveAsDirectory(dirPath);

            Assert.True(Persistence.EvaluationModelTakesTarget(path));
            var eval = Persistence.LoadEvaluationModel(path);
            Assert.Equal(GraphKind.ConcreteModel, eval.Kind);
            Assert.Equal((double)expected, EvalLoss(eval, [1f, 2f, 3f, 4f], [2f, 4f, 6f, 8f]), 5);
            Assert.Equal((double)expected, EvalLoss(Persistence.LoadEvaluationModel(dirPath), [1f, 2f, 3f, 4f], [2f, 4f, 6f, 8f]), 5);

            Assert.Equal(GraphKind.ConcreteModel, Persistence.Load(path).Kind);
            Assert.Throws<ArgumentException>(() => Persistence.LoadEvaluationModel(""));
            Assert.Throws<ArgumentException>(() => Persistence.EvaluationModelTakesTarget(""));

            var flat = TempPath("flat_eval") + ".safetensors";
            try
            {
                ckpt.Save(flat);
                Assert.Throws<InvalidDataException>(() => Persistence.LoadEvaluationModel(flat));
                Assert.Throws<InvalidDataException>(() => Persistence.EvaluationModelTakesTarget(flat));
                Assert.Contains("LoadEvaluationModel",
                    Assert.Throws<InvalidOperationException>(
                        () => Persistence.LoadTrainingCheckpoint(flat).ToInferenceModel()).Message);
            }
            finally { if (File.Exists(flat)) File.Delete(flat); }

            var inferenceOnly = TempPath("inf_eval") + ".skpt";
            try
            {
                Persistence.From(ckpt.ToInferenceModel()).WithModel().WithWeights().Save(inferenceOnly);
                Assert.Throws<InvalidDataException>(() => Persistence.LoadEvaluationModel(inferenceOnly));
            }
            finally { if (File.Exists(inferenceOnly)) File.Delete(inferenceOnly); }
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
            if (Directory.Exists(dirPath)) Directory.Delete(dirPath, recursive: true);
        }
    }

    [Fact]
    public void TestLoadDefersTheInitializersItsCheckpointOverwritesAndStillProducesThemOnDemandCoverage()
    {
        var (rig, ckpt, inBatch, outBatch) = BuildTrainedAdamWRig(steps: 2);
        var reference = rig.TrainStep(ckpt.Shared(), inBatch.Shared(), outBatch.Shared());
        var eagerInitial = rig.CreateInitialCheckpoint();
        var path = TempPath("skpt_defer") + ".skpt";
        try
        {
            Persistence.SaveTrainingCheckpointToSkpt(ckpt, path);
            var (loadedRig, loaded) = TrainingRig.Load(path);

            Assert.Equal(2, loaded.Step);
            Assert.Equal(FlattenStruct(ckpt.TrainableParams), FlattenStruct(loaded.TrainableParams));
            Assert.Equal(FlattenStruct(ckpt.OptimizerState), FlattenStruct(loaded.OptimizerState));

            var resumed = loadedRig.TrainStep(loaded.Shared(), inBatch.Shared(), outBatch.Shared());
            Assert.Equal(FlattenStruct(reference.TrainableParams), FlattenStruct(resumed.TrainableParams));
            Assert.Equal(FlattenStruct(reference.OptimizerState), FlattenStruct(resumed.OptimizerState));
            Assert.Equal(reference.Loss!.Value, resumed.Loss!.Value);

            var deferredInitial = loadedRig.CreateInitialCheckpoint();
            Assert.Equal(FlattenStruct(eagerInitial.TrainableParams), FlattenStruct(deferredInitial.TrainableParams));
            Assert.Equal(FlattenStruct(eagerInitial.ModelState), FlattenStruct(deferredInitial.ModelState));
            Assert.Equal(FlattenStruct(eagerInitial.OptimizerState), FlattenStruct(deferredInitial.OptimizerState));
            Assert.Equal(FlattenStruct(eagerInitial.TrainableParams), FlattenStruct(loadedRig.CreateInitialCheckpoint().TrainableParams));

            var weightsOnly = loadedRig.LoadCheckpointFromSkpt(path, CheckpointComponents.InferenceState);
            Assert.Equal(FlattenStruct(ckpt.TrainableParams), FlattenStruct(weightsOnly.TrainableParams));
            Assert.Equal(FlattenStruct(eagerInitial.OptimizerState), FlattenStruct(weightsOnly.OptimizerState));
            Assert.Equal(0, weightsOnly.Step);

            Assert.NotNull(loadedRig.AdoptCheckpoint(ckpt).Rig);
            Assert.NotNull(loaded.ToInferenceModel());
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    [Fact]
    public void TestADeferredRigReseedsOptimizerStateThatReadsAParameterValueCoverage()
    {
        NamedModelParam[] sample =
            [new TensorDataModelParam("input", ModelParamType.InputParam, TensorData(ScalarInputShape, [1f, 2f, 3f, 4f]))];
        var rig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            ParamValueSeededOptimizer.ComputationGraph, sample, 0.1f);
        var eager = rig.CreateInitialCheckpoint();
        var ckpt = rig.TrainStep(eager.Shared(), InBatch(1f, 2f, 3f, 4f), TargetBatch(2f, 4f, 6f, 8f));

        Assert.Equal(
            FlattenStruct(eager.TrainableParams).Select(v => v * 3f).ToArray(),
            FlattenStruct(eager.OptimizerState));

        var path = TempPath("skpt_paramseed") + ".skpt";
        try
        {
            Persistence.SaveTrainingCheckpointToSkpt(ckpt, path);
            var (loadedRig, loaded) = TrainingRig.Load(path);
            Assert.Equal(FlattenStruct(ckpt.OptimizerState), FlattenStruct(loaded.OptimizerState));
            Assert.Equal(FlattenStruct(eager.OptimizerState),
                FlattenStruct(loadedRig.CreateInitialCheckpoint().OptimizerState));
            Assert.Equal(FlattenStruct(eager.OptimizerState),
                FlattenStruct(loadedRig.LoadCheckpointFromSkpt(path, CheckpointComponents.InferenceState).OptimizerState));
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    [Fact]
    public void TestADeferredRigHandsTheSameInitialValuesToEveryConcurrentCallerCoverage()
    {
        const int width = 2048;
        var sample = new float[width];
        for (var i = 0; i < width; i++) sample[i] = (i % 13) / 13f;

        var rig = TrainingRig.FromScratch(
            WideWeightModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            [new TensorDataModelParam("input", ModelParamType.InputParam, TensorData([(long)width], sample))],
            0.1f);
        var inputs = rig.InputDef.FromOrderedData(TensorData([(long)width], sample));
        var targets = rig.TargetDef.FromOrderedData(TensorData([(long)width], new float[width]));
        var expected = FlattenStruct(rig.CreateInitialCheckpoint().TrainableParams);
        var ckpt = rig.TrainStep(rig.CreateInitialCheckpoint(), inputs.Shared(), targets.Shared());

        var path = TempPath("skpt_concurrent") + ".skpt";
        try
        {
            Persistence.SaveTrainingCheckpointToSkpt(ckpt, path);
            for (var attempt = 0; attempt < 4; attempt++)
            {
                var (loadedRig, _) = TrainingRig.Load(path);
                var seen = new float[8][];
                Parallel.For(0, seen.Length, i =>
                    seen[i] = FlattenStruct(loadedRig.CreateInitialCheckpoint().TrainableParams));
                Assert.All(seen, values => Assert.Equal(expected, values));
            }
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    [Fact]
    public void TestEvaluationModelCoversModelStateMultiInputAndInt64TargetShapesCoverage()
    {
        float[] batch = [1f, 2f, 3f, 4f];
        float[] labels = [2f, 4f, 6f, 8f];

        var bnRig = TrainingRig.FromScratch(
            StateReadingModel.ComputationGraph, L2Loss.ComputationGraph,
            SGDOptimizer.ComputationGraph,
            [new TensorDataModelParam("input", ModelParamType.InputParam, TensorData(ScalarInputShape, batch))],
            0.1f);
        var bnInitial = bnRig.CreateInitialCheckpoint();
        var bnCkpt = bnRig.TrainStep(bnInitial.Shared(), InBatch(batch), TargetBatch(labels));
        Assert.NotEmpty(bnCkpt.ModelState.Fields);
        Assert.NotEqual(FlattenStruct(bnInitial.ModelState), FlattenStruct(bnCkpt.ModelState));

        var twoRig = TrainingRig.FromScratch(
            TwoInputSumModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            [
                new TensorDataModelParam("a", ModelParamType.InputParam, TensorData(ScalarInputShape, batch)),
                new TensorDataModelParam("b", ModelParamType.InputParam, TensorData(ScalarInputShape, batch)),
            ],
            0.1f);
        var twoInputs = twoRig.InputDef.FromOrderedData(
            TensorData(ScalarInputShape, batch), TensorData(ScalarInputShape, batch));
        var twoCkpt = twoRig.TrainStep(twoRig.CreateInitialCheckpoint(), twoInputs.Shared(),
            twoRig.TargetDef.FromOrderedData(TensorData(ScalarInputShape, labels)));

        var bnPath = TempPath("skpt_eval_bn") + ".skpt";
        var twoPath = TempPath("skpt_eval_two") + ".skpt";
        try
        {
            Persistence.SaveTrainingCheckpointToSkpt(bnCkpt, bnPath);
            Persistence.SaveTrainingCheckpointToSkpt(twoCkpt, twoPath);

            Assert.Equal((double)bnRig.TrainStep(bnCkpt.Shared(), InBatch(batch), TargetBatch(labels)).Loss!.Value,
                EvalLoss(Persistence.LoadEvaluationModel(bnPath), batch, labels), 5);

            var initialPath = TempPath("skpt_eval_bn0") + ".skpt";
            try
            {
                Persistence.SaveTrainingCheckpointToSkpt(bnInitial, initialPath);
                Assert.NotEqual(
                    EvalLoss(Persistence.LoadEvaluationModel(initialPath), batch, labels),
                    EvalLoss(Persistence.LoadEvaluationModel(bnPath), batch, labels));
            }
            finally { if (File.Exists(initialPath)) File.Delete(initialPath); }

            var (bnLoadedRig, bnLoaded) = TrainingRig.Load(bnPath);
            Assert.Equal(FlattenStruct(bnCkpt.ModelState), FlattenStruct(bnLoaded.ModelState));
            Assert.Equal(FlattenStruct(bnRig.CreateInitialCheckpoint().ModelState),
                FlattenStruct(bnLoadedRig.CreateInitialCheckpoint().ModelState));

            var (ceInput, ceTarget) = NNLibraryTrainingFixtures.MakeTinyConvBatch();
            var ceRig = TrainingRig.FromScratch(
                NNTinyConvClassifier.ComputationGraph, CrossEntropyLoss.ComputationGraph,
                SGDOptimizer.ComputationGraph,
                [new TensorDataModelParam("input", ModelParamType.InputParam, ceInput)],
                0.1f);
            var ceIn = ceRig.InputDef.FromOrderedData(ceInput);
            var ceTg = ceRig.TargetDef.FromOrderedData(ceTarget);
            var ceCkpt = ceRig.TrainStep(ceRig.CreateInitialCheckpoint(), ceIn.Shared(), ceTg.Shared());
            Assert.Equal(DType.Int64, ceRig.TargetDef.Fields[0].ElementType);

            var cePath = TempPath("skpt_eval_ce") + ".skpt";
            try
            {
                Persistence.SaveTrainingCheckpointToSkpt(ceCkpt, cePath);
                var ceLoss = ComputeContext.Default
                    .Execute(Persistence.LoadEvaluationModel(cePath), ceInput.Shared(), ceTarget.Shared())[0]
                    .ToTensorData<float32>().ValueAt<float>(0);
                Assert.Equal((double)ceRig.TrainStep(ceCkpt.Shared(), ceIn.Shared(), ceTg.Shared()).Loss!.Value, ceLoss, 5);
            }
            finally { if (File.Exists(cePath)) File.Delete(cePath); }

            var twoEval = Persistence.LoadEvaluationModel(twoPath);
            var twoLoss = ComputeContext.Default.Execute(twoEval,
                TensorData(ScalarInputShape, batch), TensorData(ScalarInputShape, batch),
                TensorData(ScalarInputShape, labels))[0].ToTensorData<float32>().ValueAt<float>(0);
            Assert.Equal((double)twoRig.TrainStep(twoCkpt.Shared(), twoInputs.Shared(),
                twoRig.TargetDef.FromOrderedData(TensorData(ScalarInputShape, labels))).Loss!.Value, twoLoss, 5);
        }
        finally
        {
            if (File.Exists(bnPath)) File.Delete(bnPath);
            if (File.Exists(twoPath)) File.Delete(twoPath);
        }
    }

}

[Trait("Domain", "Training")]
[Trait("Purpose", "Coverage")]
public class TrainingRigHyperparameterDTypeCoverageTests
{
    private static ComputationGraph SchedulerModuleRaw(Variable[] inputs, Variable[] outputs)
        => new(new InternalComputationGraph([.. inputs], [.. outputs]), GraphKind.Module);

    private static TrainingRig MixedRig(MixedDTypeHyperOptimizerHyperparameters hypers)
        => TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            MixedDTypeHyperOptimizer.ComputationGraph, ScalarMultiplyBatches().sample, hypers);

    [Fact]
    public void TestDeclaredDTypesDriveBakedRuntimeAndScheduledHyperparametersCoverage()
    {
        var (_, inputBatch, targetBatch) = ScalarMultiplyBatches();

        var bakedRig = MixedRig(new MixedDTypeHyperOptimizerHyperparameters());
        Assert.Equal((string[])["learningRate", "gradScale", "descend", "decay"],
            bakedRig.HyperparameterNames.ToArray());
        Assert.Equal((DType[])[DType.Float32, DType.Int32, DType.Bool, DType.Float64],
            bakedRig.HyperparameterDTypes.ToArray());
        Assert.Equal((DType[])[DType.Float32, DType.Int32, DType.Bool, DType.Float64],
            bakedRig.Hyperparameters.Select(h => h.BakedDType).ToArray());
        Assert.Equal(2, ((TensorData<int32>)bakedRig.Hyperparameters[1].BakedValue).AccessMemory()[0]);
        Assert.True(((TensorData<bit>)bakedRig.Hyperparameters[2].BakedValue).AccessMemory()[0]);
        Assert.Equal(0.25, ((TensorData<float64>)bakedRig.Hyperparameters[3].BakedValue).AccessMemory()[0]);
        Assert.Empty(bakedRig.HyperparameterStructDef.Fields);

        float Step(TrainingRig rig, TensorDataStruct? hypers = null)
        {
            var ckpt = rig.CreateInitialCheckpoint();
            return Weight(rig, hypers is null
                ? rig.TrainStep(ckpt.Shared(), inputBatch.Shared(), targetBatch.Shared())
                : rig.TrainStep(ckpt.Shared(), hypers.Shared(), inputBatch.Shared(), targetBatch.Shared()));
        }

        var runtimeRig = MixedRig(new MixedDTypeHyperOptimizerHyperparameters
        {
            LearningRate = Hyperparameter.Runtime(),
            GradScale = Hyperparameter.Runtime(),
            Descend = Hyperparameter.Runtime(),
            Decay = Hyperparameter.Runtime(),
        });
        Assert.Equal((DType[])[DType.Float32, DType.Int32, DType.Bool, DType.Float64],
            runtimeRig.HyperparameterStructDef.Fields.Select(f => f.ElementType).ToArray());

        var matching = runtimeRig.MakeHyperparameters(
            ("learningRate", 0.1f), ("gradScale", 2), ("descend", true), ("decay", 0.25));
        Assert.Equal(DType.Int32, ((TensorData)matching.Fields["gradScale"]).DType);
        Assert.Equal(DType.Bool, ((TensorData)matching.Fields["descend"]).DType);
        Assert.True(MathF.Abs(Step(bakedRig) - Step(runtimeRig, matching)) < 1e-5f);

        var ascend = runtimeRig.MakeHyperparameters(
            ("learningRate", 0.1f), ("gradScale", 2), ("descend", false), ("decay", 0.25));
        Assert.True(MathF.Abs(Step(runtimeRig, ascend) - Step(runtimeRig, matching)) > 1e-4f);

        var scheduledRig = MixedRig(new MixedDTypeHyperOptimizerHyperparameters
        {
            LearningRate = Schedules.Constant(0.1f),
            GradScale = Hyperparameter.Scheduled(IntStepScheduler.ComputationGraph),
            Descend = true,
            Decay = 0.25,
        });
        Assert.Empty(scheduledRig.HyperparameterStructDef.Fields);
        var scheduledStep = Step(scheduledRig);
        var scaleTwoAtStepZero = runtimeRig.MakeHyperparameters(
            ("learningRate", 0.1f), ("gradScale", 2), ("descend", true), ("decay", 0.25));
        Assert.True(MathF.Abs(scheduledStep - Step(runtimeRig, scaleTwoAtStepZero)) < 1e-5f);

        var intStateRig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            InitFromIntHyperOptimizer.ComputationGraph, ScalarMultiplyBatches().sample,
            new InitFromIntHyperOptimizerHyperparameters { StateSeed = 7 });
        Assert.All(FlattenStruct(intStateRig.CreateInitialCheckpoint().OptimizerState),
            v => Assert.Equal(7f, v));
    }

    [Fact]
    public void TestHyperparameterDTypeConversionsAndRejectionsCoverage()
    {
        var runtimeRig = MixedRig(new MixedDTypeHyperOptimizerHyperparameters
        {
            LearningRate = Hyperparameter.Runtime(),
            GradScale = Hyperparameter.Runtime(),
            Descend = Hyperparameter.Runtime(),
            Decay = Hyperparameter.Runtime(),
        });

        TensorDataStruct Make(object lr, object scale, object descend, object decay)
            => runtimeRig.MakeHyperparameters(
                ("learningRate", lr), ("gradScale", scale), ("descend", descend), ("decay", decay));

        // Value-preserving widening/narrowing is converted to the declared dtype.
        Assert.Equal(3, ((TensorData<int32>)Make(0.1f, 3L, true, 0.25).Fields["gradScale"]).AccessMemory()[0]);
        Assert.Equal(0.5f, ((TensorData<float32>)Make(0.5, 3, true, 0.25).Fields["learningRate"]).AccessMemory()[0]);
        Assert.Equal(2.0, ((TensorData<float64>)Make(0.1f, 3, true, 2).Fields["decay"]).AccessMemory()[0]);

        // Rounding between float dtypes is ordinary precision, not a lost value: a plain `0.1` double
        // literal is the familiar 0.1f for a float32 hyperparameter, and only overflow is rejected.
        Assert.Equal(0.1f, ((TensorData<float32>)Make(0.1, 3, true, 0.25).Fields["learningRate"]).AccessMemory()[0]);
        Assert.Equal(1e-8f, ((TensorData<float32>)Make(1e-8, 3, true, 0.25).Fields["learningRate"]).AccessMemory()[0]);
        Assert.Throws<ArgumentException>(() => Make(1e300, 3, true, 0.25));
        Assert.Equal(0.1f, ((TensorData<float32>)MixedRig(new MixedDTypeHyperOptimizerHyperparameters
            { LearningRate = 0.1 }).Hyperparameters[0].BakedValue).AccessMemory()[0]);

        // A value that would not survive the conversion, or crosses the bool boundary, fails loud.
        Assert.Throws<ArgumentException>(() => Make(0.1f, 2.5, true, 0.25));
        Assert.Throws<ArgumentException>(() => Make(0.1f, long.MaxValue, true, 0.25));
        Assert.Throws<ArgumentException>(() => Make(0.1f, true, true, 0.25));
        Assert.Throws<ArgumentException>(() => Make(0.1f, 2, 1, 0.25));
        Assert.Throws<ArgumentException>(() => Make(0.1f, 2, true, "0.25"));

        // A baked value is converted at rig build under the same rule.
        Assert.Equal(5, ((TensorData<int32>)MixedRig(new MixedDTypeHyperOptimizerHyperparameters
            { GradScale = 5L }).Hyperparameters[1].BakedValue).AccessMemory()[0]);
        Assert.Throws<ArgumentException>(() => MixedRig(new MixedDTypeHyperOptimizerHyperparameters
            { GradScale = 2.5 }));
        Assert.Throws<ArgumentException>(() => MixedRig(new MixedDTypeHyperOptimizerHyperparameters
            { Descend = 1 }));

        // Built-in Schedule math is float32; a non-float32 hyperparameter needs a scheduler module.
        var schedEx = Assert.Throws<ArgumentException>(() => MixedRig(
            new MixedDTypeHyperOptimizerHyperparameters { GradScale = Schedules.Constant(2f) }));
        Assert.Contains("scheduler module", schedEx.Message);

        // A scheduler module must produce the declared dtype, not merely float32.
        var step = InputScalar<int64>("step");
        var moduleEx = Assert.Throws<ArgumentException>(() => MixedRig(
            new MixedDTypeHyperOptimizerHyperparameters
                { GradScale = Hyperparameter.Scheduled(SchedulerModuleRaw([step], [step.Cast<float32>()])) }));
        Assert.Contains("Int32", moduleEx.Message);

        // A non-scalar constant bound to a Scalar<T>-declared hyperparameter is rejected at rig build.
        var rankEx = Assert.Throws<ArgumentException>(() => MixedRig(
            new MixedDTypeHyperOptimizerHyperparameters
                { GradScale = Hyperparameter.Baked((TensorData)TensorData([2L], [1, 2])) }));
        Assert.Contains("declared with rank 0", rankEx.Message);
    }

    [Fact]
    public void TestNonFloatHyperparametersRoundTripThroughSkptCoverage()
    {
        var (_, inputBatch, targetBatch) = ScalarMultiplyBatches();
        var rig = MixedRig(new MixedDTypeHyperOptimizerHyperparameters
        {
            LearningRate = Schedules.Constant(0.1f),
            GradScale = 3,
            Descend = false,
            Decay = 0.125,
        });
        var ckpt = rig.TrainStep(rig.CreateInitialCheckpoint(), inputBatch.Shared(), targetBatch.Shared());
        var reference = rig.TrainStep(ckpt.Shared(), inputBatch.Shared(), targetBatch.Shared());

        var path = TempPath("hyperdtype") + ".skpt";
        try
        {
            Persistence.SaveTrainingCheckpointToSkpt(ckpt, path);

            var manifest = System.Text.Encoding.UTF8.GetString(ReadEntryBytesViaBcl(path, "config.json"));
            Assert.Contains("\"dtype\": \"Int32\"", manifest);
            Assert.Contains("\"dtype\": \"Bool\"", manifest);
            Assert.Contains("\"dtype\": \"Float64\"", manifest);
            Assert.DoesNotContain("bakedHypers", manifest);

            var (rig2, loaded) = TrainingRig.Load(path);
            Assert.Equal((DType[])[DType.Float32, DType.Int32, DType.Bool, DType.Float64],
                rig2.HyperparameterDTypes.ToArray());
            Assert.Equal(3, ((TensorData<int32>)rig2.Hyperparameters[1].BakedValue).AccessMemory()[0]);
            Assert.False(((TensorData<bit>)rig2.Hyperparameters[2].BakedValue).AccessMemory()[0]);
            Assert.Equal(0.125, ((TensorData<float64>)rig2.Hyperparameters[3].BakedValue).AccessMemory()[0]);

            var resumed = rig2.TrainStep(loaded.Shared(), inputBatch.Shared(), targetBatch.Shared());
            Assert.Equal(FlattenStruct(reference.TrainableParams), FlattenStruct(resumed.TrainableParams));
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }
}

[Trait("Domain", "Training")]
[Trait("Purpose", "Coverage")]
public class TrainingRigHyperparameterShapeCoverageTests
{
    private static TrainingRig VectorRig(VectorRateOptimizerHyperparameters hypers)
        => TrainingRig.FromScratch(
            VectorMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            VectorRateOptimizer.ComputationGraph, ScalarMultiplyBatches().sample, hypers);

    private static TensorData Rate(params float[] v) => (TensorData)TensorData([(long)v.Length], v);

    [Fact]
    public void TestNonScalarHyperparametersDriveTrainingThroughEveryKindCoverage()
    {
        var (_, inputBatch, targetBatch) = ScalarMultiplyBatches();

        var bakedRig = VectorRig(new VectorRateOptimizerHyperparameters
            { PerElementRate = Hyperparameter.Baked(Rate(0.1f, 0.2f, 0.4f, 0.8f)) });
        Assert.Equal((long[])[4L], bakedRig.HyperparameterShapes[0].Dims);
        Assert.Equal((long[])[], bakedRig.HyperparameterShapes[1].Dims);
        Assert.Equal(DType.Float32, bakedRig.HyperparameterDTypes[0]);
        Assert.Empty(bakedRig.HyperparameterStructDef.Fields);

        float[] Step(TrainingRig rig, TensorDataStruct? hypers = null)
        {
            var ckpt = rig.CreateInitialCheckpoint();
            return FlattenStruct((hypers is null
                ? rig.TrainStep(ckpt.Shared(), inputBatch.Shared(), targetBatch.Shared())
                : rig.TrainStep(ckpt.Shared(), hypers.Shared(), inputBatch.Shared(), targetBatch.Shared())).TrainableParams);
        }
        static bool Close(float[] a, float[] b) =>
            a.Length == b.Length && a.Zip(b).All(p => MathF.Abs(p.First - p.Second) < 1e-5f);

        var runtimeRig = VectorRig(new VectorRateOptimizerHyperparameters
        {
            PerElementRate = Hyperparameter.Runtime(4L),
            Gain = Hyperparameter.Runtime(),
        });
        Assert.Equal((string[])["perElementRate", "gain"], runtimeRig.DynamicHyperparameterNames.ToArray());
        Assert.Equal((int?[])[1, 0], runtimeRig.HyperparameterStructDef.Fields.Select(f => f.Rank).ToArray());
        Assert.Equal((long[])[4L], runtimeRig.HyperparameterShapes[0].Dims);

        var matching = runtimeRig.MakeHyperparameters(
            ("perElementRate", Rate(0.1f, 0.2f, 0.4f, 0.8f)), ("gain", 1f));
        Assert.Equal((long[])[4L], ((TensorData)matching.Fields["perElementRate"]).Shape.Dims);
        Assert.True(Close(Step(bakedRig), Step(runtimeRig, matching)));

        var doubled = runtimeRig.MakeHyperparameters(
            ("perElementRate", Rate(0.1f, 0.2f, 0.4f, 0.8f)), ("gain", 2f));
        Assert.False(Close(Step(runtimeRig, doubled), Step(runtimeRig, matching)));

        var scheduledRig = VectorRig(new VectorRateOptimizerHyperparameters
            { PerElementRate = Hyperparameter.Scheduled(VectorRateScheduler.ComputationGraph) });
        Assert.Equal((long[])[4L], scheduledRig.HyperparameterShapes[0].Dims);
        Assert.Empty(scheduledRig.HyperparameterStructDef.Fields);
        Assert.True(Close(Step(scheduledRig), Step(bakedRig)));

        var scheduledCkpt = scheduledRig.CreateInitialCheckpoint();
        var hostCkpt = runtimeRig.CreateInitialCheckpoint();
        for (int s = 0; s < 3; s++)
        {
            scheduledCkpt = scheduledRig.TrainStep(scheduledCkpt, inputBatch.Shared(), targetBatch.Shared());
            hostCkpt = runtimeRig.TrainStep(hostCkpt, runtimeRig.MakeHyperparameters(
                ("perElementRate", Rate(0.1f - 0.01f * s, 0.2f - 0.01f * s, 0.4f - 0.01f * s, 0.8f - 0.01f * s)),
                ("gain", 1f)), inputBatch.Shared(), targetBatch.Shared());
            Assert.True(Close(FlattenStruct(scheduledCkpt.TrainableParams),
                               FlattenStruct(hostCkpt.TrainableParams)));
        }

        var stateRig = TrainingRig.FromScratch(
            VectorMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            InitFromVectorHyperOptimizer.ComputationGraph, ScalarMultiplyBatches().sample,
            new InitFromVectorHyperOptimizerHyperparameters
                { PerElementRate = Hyperparameter.Baked(Rate(0.5f, 1.5f, 2f, 3f)) });
        Assert.All(FlattenStruct(stateRig.CreateInitialCheckpoint().OptimizerState),
            v => Assert.True(MathF.Abs(v - 7f) < 1e-4f));
    }

    [Fact]
    public void TestHyperparameterShapeMismatchesAndScalarOnlySourcesAreRejectedCoverage()
    {
        // A binding whose rank contradicts the declared Vector<T> / Scalar<T> fails at rig build.
        static void RankMismatch(Func<TrainingRig> build) => Assert.Contains(
            "declared with rank", Assert.Throws<ArgumentException>(() => build()).Message);

        RankMismatch(() => VectorRig(new VectorRateOptimizerHyperparameters
            { PerElementRate = 0.1f }));
        RankMismatch(() => VectorRig(new VectorRateOptimizerHyperparameters
            { PerElementRate = Hyperparameter.Runtime(2L, 2L) }));
        RankMismatch(() => VectorRig(new VectorRateOptimizerHyperparameters
            { PerElementRate = Hyperparameter.Baked(Rate(0.1f)), Gain = Hyperparameter.Runtime(3L) }));

        // Built-in Schedule math is a float32 scalar, so it cannot drive a vector hyperparameter.
        var schedEx = Assert.Throws<ArgumentException>(() => VectorRig(new VectorRateOptimizerHyperparameters
            { PerElementRate = Schedules.Constant(0.1f) }));
        Assert.Contains("scheduler module", schedEx.Message);

        // A scheduler module must produce the declared rank.
        var moduleEx = Assert.Throws<ArgumentException>(() => VectorRig(new VectorRateOptimizerHyperparameters
            { PerElementRate = Hyperparameter.Scheduled(SchedulerModuleRaw()) }));
        Assert.Contains("rank-1", moduleEx.Message);

        // A per-step value must match the shape the rig was built at, and keeps its declared dtype.
        var runtimeRig = VectorRig(new VectorRateOptimizerHyperparameters
            { PerElementRate = Hyperparameter.Runtime(4L) });
        var shapeEx = Assert.Throws<ArgumentException>(
            () => runtimeRig.MakeHyperparameters(Rate(0.1f, 0.2f)));
        Assert.Contains("shape is fixed", shapeEx.Message);
        Assert.Throws<ArgumentException>(() => runtimeRig.MakeHyperparameters(0.1f));
        Assert.Throws<ArgumentException>(() => runtimeRig.MakeHyperparameters(
            (TensorData)TensorData([4L], [1, 2, 3, 4])));
    }

    private static ComputationGraph SchedulerModuleRaw()
    {
        var step = InputScalar<int64>("step");
        return new ComputationGraph(
            new InternalComputationGraph([step], [step.Cast<float32>()]), GraphKind.Module);
    }

    /// <summary>An optimizer whose update returns a parameter at another shape — a <c>[4]</c>
    /// per-element rate broadcasting against a <c>[1]</c> weight — is refused when the rig is
    /// built, before any step can reshape the parameter.</summary>
    [Fact]
    public void TestARigWhoseOptimizerWouldReshapeAParameterIsRefused()
    {
        var refusal = Assert.Throws<ArgumentException>(() => TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            VectorRateOptimizer.ComputationGraph, ScalarMultiplyBatches().sample,
            new VectorRateOptimizerHyperparameters
            {
                PerElementRate = Hyperparameter.Baked(Rate(0.1f, 0.2f, 0.4f, 0.8f)),
                Gain = Hyperparameter.Runtime(),
            })).Message;
        Assert.Contains("InitScalarWeight#0", refusal);
        Assert.Contains("at that parameter's own shape", refusal);
    }

    [Fact]
    public void TestNonScalarHyperparametersRoundTripThroughSkptCoverage()
    {
        var (_, inputBatch, targetBatch) = ScalarMultiplyBatches();
        var rig = VectorRig(new VectorRateOptimizerHyperparameters
        {
            PerElementRate = Hyperparameter.Baked(Rate(0.1f, 0.2f, 0.4f, 0.8f)),
            Gain = Hyperparameter.Runtime(),
        });
        var ckpt = rig.TrainStep(rig.CreateInitialCheckpoint(), rig.MakeHyperparameters(1f),
            inputBatch.Shared(), targetBatch.Shared());
        var reference = rig.TrainStep(ckpt.Shared(), rig.MakeHyperparameters(1f), inputBatch.Shared(), targetBatch.Shared());

        var path = TempPath("hypershape") + ".skpt";
        try
        {
            Persistence.SaveTrainingCheckpointToSkpt(ckpt, path);

            var (rig2, loaded) = TrainingRig.Load(path);
            Assert.Equal((long[])[4L], rig2.HyperparameterShapes[0].Dims);
            Assert.Equal((long[])[], rig2.HyperparameterShapes[1].Dims);
            Assert.Equal((float[])[0.1f, 0.2f, 0.4f, 0.8f],
                ((TensorData<float32>)rig2.Hyperparameters[0].BakedValue).AccessMemory().ToArray());
            Assert.Equal((long[])[], rig2.Hyperparameters[1].RuntimeShape.ToArray());

            var resumed = rig2.TrainStep(loaded.Shared(), rig2.MakeHyperparameters(1f), inputBatch.Shared(), targetBatch.Shared());
            Assert.Equal(FlattenStruct(reference.TrainableParams), FlattenStruct(resumed.TrainableParams));
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }
}

[Trait("Domain", "Training")]
[Trait("Purpose", "Coverage")]
public class BuildProgressCoverageTests
{
    private static (List<BuildProgress> Reports, SynchronousBuildProgress Sink) Watched()
    {
        var reports = new List<BuildProgress>();
        return (reports, new SynchronousBuildProgress(reports.Add));
    }

    private static BuildPhase[] PhaseRuns(List<BuildProgress> reports)
    {
        var runs = new List<BuildPhase>();
        foreach (var r in reports)
            if (runs.Count == 0 || runs[^1] != r.Phase) runs.Add(r.Phase);
        return [.. runs];
    }

    private static string[] StagesOf(List<BuildProgress> reports, BuildPhase phase)
        => [.. reports.Where(r => r.Phase == phase).Select(r => r.Stage)];

    private static readonly string[] ConcretizePasses =
    [
        "Clone", "ApplyIdentifierTemplates", "InlineModulesAndFunctions", "InjectRngDrawCounter",
        "ExtractIdentifierTemplates", "ConvertToIdRefModelParams", "UnpackModelStruct",
        "UnpackTensorStructs", "ConvertModelParamIdRefToModelParam", "Simplify",
        "LowerAttributeTensorOps", "ExpandAutoGrad", "SimplifyAfterAutoGrad",
    ];

    [Fact]
    public void TestLoadReportsTheDeferredInitializationStageInsteadOfRunningInitializersCoverage()
    {
        var (rig, ckpt, inBatch, outBatch) = BuildTrainedAdamWRig(steps: 1);
        var path = TempPath("skpt_defer_stage") + ".skpt";
        try
        {
            Persistence.SaveTrainingCheckpointToSkpt(ckpt, path);

            var (reports, sink) = Watched();
            var (loadedRig, loaded) = TrainingRig.Load(path, progress: sink);
            var stages = StagesOf(reports, BuildPhase.Initialize);

            Assert.Contains("DeferModelParamInitialization", stages);
            Assert.DoesNotContain("InitializeModelParams", stages);
            Assert.Equal(1, loaded.Step);

            var (eagerReports, eagerSink) = Watched();
            _ = TrainingRig.FromScratch(
                ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
                AdamWOptimizer.ComputationGraph, ScalarMultiplyBatches().sample,
                new AdamWOptimizerHyperparameters { LearningRate = 0.1f }, progress: eagerSink);
            var eagerStages = StagesOf(eagerReports, BuildPhase.Initialize);
            Assert.Contains("InitializeModelParams", eagerStages);
            Assert.DoesNotContain("DeferModelParamInitialization", eagerStages);

            Assert.Equal(
                FlattenStruct(rig.CreateInitialCheckpoint().TrainableParams),
                FlattenStruct(loadedRig.CreateInitialCheckpoint().TrainableParams));
            Assert.Equal(
                FlattenStruct(rig.TrainStep(ckpt.Shared(), inBatch.Shared(), outBatch.Shared()).TrainableParams),
                FlattenStruct(loadedRig.TrainStep(loaded.Shared(), inBatch.Shared(), outBatch.Shared()).TrainableParams));
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    [Fact]
    public void TestFromScratchReportsEveryStageOfEveryPhaseInOrderCoverage()
    {
        var reports = new List<BuildProgress>();
        var reportThreads = new List<int>();
        var sink = new SynchronousBuildProgress(r =>
        {
            reports.Add(r);
            reportThreads.Add(Environment.CurrentManagedThreadId);
        });
        var (sample, _, _) = ScalarMultiplyBatches();
        var buildThread = Environment.CurrentManagedThreadId;

        var rig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            AdamWOptimizer.ComputationGraph, sample,
            new AdamWOptimizerHyperparameters { LearningRate = 0.1f }, progress: sink);

        BuildPhase[] phases = [BuildPhase.Concretize, BuildPhase.TrainingStep, BuildPhase.Initialize];
        string[] concretize = ["Thaw", .. ConcretizePasses, "BindRngConfig", "WriteRepresentativeInputs"];
        string[] trainingStep =
        [
            "NormalizeOptimizerGraph", "ComposeModelLossAndAutoGrad", "ReplayOptimizerPerParameter",
            "PruneAndOrderTrainingStep", "ExpandStructOutputs", "UnpackTensorStructs", "Simplify",
            "FoldLoopIterationCounts", "UnrollLoops", "LowerAttributeTensorOps", "ExpandAutoGrad",
            "SimplifyAfterAutoGrad",
        ];
        string[] initialize =
        [
            "ReadModelParams", "InitializeModelParams", "InitializeOptimizerState", "InferModelShapes",
            "InferTrainingStepShapes", "OptimizeTrainingStepGraph", "FreezeTrainingStepGraph", "Done",
        ];

        Assert.Equal(GraphKind.ConcreteModel, rig.TrainingStepPureGraph.Kind);
        Assert.Equal(phases, PhaseRuns(reports));
        Assert.Equal(concretize, StagesOf(reports, BuildPhase.Concretize));
        Assert.Equal(trainingStep, StagesOf(reports, BuildPhase.TrainingStep));
        Assert.Equal(initialize, StagesOf(reports, BuildPhase.Initialize));
        Assert.True(reports[^1].IsComplete);
        Assert.DoesNotContain(reports[..^1], r => r.IsComplete);
        Assert.Equal(reports.Select(r => r.Elapsed).OrderBy(e => e), reports.Select(r => r.Elapsed));
        Assert.All(reportThreads, t => Assert.Equal(buildThread, t));
    }

    [Fact]
    public void TestOnlyTheWatchedBuildReportsToItsOwnSinkCoverage()
    {
        var (reports, sink) = Watched();
        var model = ScalarMultiplyModel.ComputationGraph;
        var hints = new ModelParamList(
            [new KeyValuePair<string, TensorData>(model.ToInternal().Inputs[0].ToString(), TensorData([4L], new float[4]))],
            ModelParamType.InputParam);
        string[] stages = ["Thaw", .. ConcretizePasses, "Freeze", "Done"];
        BuildPhase[] concretizeOnly = [BuildPhase.Concretize];

        Assert.Equal(GraphKind.ConcreteArchitecture, model.ToConcreteArchitecture(hints, progress: sink).Kind);
        Assert.Equal(concretizeOnly, PhaseRuns(reports));
        Assert.Equal(stages, StagesOf(reports, BuildPhase.Concretize));
        Assert.True(reports[^1].IsComplete);

        var culture = CultureInfo.CurrentCulture;
        try
        {
            CultureInfo.CurrentCulture = new CultureInfo("fr-FR");
            Assert.Equal("[   1.5s] Concretize: Clone",
                new BuildProgress(BuildPhase.Concretize, "Clone", TimeSpan.FromSeconds(1.5)).ToString());
        }
        finally { CultureInfo.CurrentCulture = culture; }

        Assert.Equal(GraphKind.ConcreteArchitecture, model.ToConcreteArchitecture(hints).Kind);
        Assert.Equal(stages.Length, reports.Count);

        var second = new List<BuildProgress>();
        Assert.Equal(GraphKind.ConcreteArchitecture,
            model.ToConcreteArchitecture(hints, progress: new SynchronousBuildProgress(second.Add)).Kind);
        Assert.Equal(stages, StagesOf(second, BuildPhase.Concretize));
    }

    [Fact]
    public void TestASchedulerBuildIsReportedCoverage()
    {
        var (reports, sink) = Watched();
        var (sample, _, _) = ScalarMultiplyBatches();
        var step = InputScalar<int64>("step");
        var scheduler = new ComputationGraph(
            new InternalComputationGraph([step], [Scalar(0.3f) - step.Cast<float32>() * Scalar(0.05f)]),
            GraphKind.Module);

        var rig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            sample, new SGDOptimizerHyperparameters { LearningRate = Hyperparameter.Scheduled(scheduler) },
            progress: sink);

        string[] trainingStep =
        [
            "NormalizeOptimizerGraph", "ComposeModelLossAndAutoGrad", "BuildSchedulers",
            "ReplayOptimizerPerParameter", "PruneAndOrderTrainingStep", "ExpandStructOutputs",
            "UnpackTensorStructs", "Simplify", "FoldLoopIterationCounts", "UnrollLoops",
            "LowerAttributeTensorOps", "ExpandAutoGrad", "SimplifyAfterAutoGrad",
        ];

        Assert.Equal(GraphKind.ConcreteModel, rig.TrainingStepPureGraph.Kind);
        Assert.Equal(trainingStep, StagesOf(reports, BuildPhase.TrainingStep));
        Assert.True(reports[^1].IsComplete);
    }

    [Fact]
    public void TestLoadReportsItsFileReadsAndCompletesOnlyAtTheEndCoverage()
    {
        var (sample, _, _) = ScalarMultiplyBatches();
        var path = TempPath("progress_load") + ".skpt";
        try
        {
            var rig = TrainingRig.FromScratch(
                ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
                AdamWOptimizer.ComputationGraph, sample,
                new AdamWOptimizerHyperparameters { LearningRate = 0.1f });
            Persistence.SaveTrainingCheckpointToSkpt(rig.CreateInitialCheckpoint(), path);

            var (reports, sink) = Watched();
            var (loaded, checkpoint) = TrainingRig.Load(path, progress: sink);

            Assert.NotNull(loaded);
            Assert.NotNull(checkpoint.TrainableParams);
            Assert.Equal("ReadCheckpointFile", reports[0].Stage);
            Assert.Equal("ThawConcreteArchitecture", reports[1].Stage);
            Assert.Equal("LoadCheckpointState", reports[^2].Stage);
            Assert.True(reports[^1].IsComplete);
            Assert.DoesNotContain(reports[..^1], r => r.IsComplete);
            Assert.DoesNotContain(reports, r => r.Stage == "Clone");
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    [Fact]
    public void TestDerivationsReportFromTheirOwnFirstPhaseCoverage()
    {
        var (reports, sink) = Watched();
        var (sample, _, _) = ScalarMultiplyBatches();
        BuildPhase[] derivation = [BuildPhase.TrainingStep, BuildPhase.Initialize];
        BuildPhase[] reseed = [BuildPhase.Concretize, BuildPhase.TrainingStep, BuildPhase.Initialize];

        var rig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            AdamWOptimizer.ComputationGraph, sample,
            new AdamWOptimizerHyperparameters { LearningRate = 0.1f });

        Assert.NotSame(rig, rig.WithLoss(L2Loss.ComputationGraph, sink));
        Assert.Equal(derivation, PhaseRuns(reports));
        Assert.True(reports[^1].IsComplete);
        Assert.DoesNotContain(reports[..^1], r => r.IsComplete);

        reports.Clear();
        Assert.NotSame(rig, rig.WithOptimizer(
            AdamWOptimizer.ComputationGraph, new AdamWOptimizerHyperparameters { LearningRate = 0.2f }, sink));
        Assert.Equal(derivation, PhaseRuns(reports));
        Assert.True(reports[^1].IsComplete);

        reports.Clear();
        Assert.NotSame(rig, rig.WithOptimizer(SGDOptimizer.ComputationGraph, [0.2f], sink));
        Assert.Equal(derivation, PhaseRuns(reports));

        reports.Clear();
        Assert.NotSame(rig, rig.WithScheduler(
            new AdamWOptimizerHyperparameters { LearningRate = 0.3f }, sink));
        Assert.Equal(derivation, PhaseRuns(reports));

        reports.Clear();
        Assert.NotSame(rig, rig.WithSeed(new RngConfig { MasterSeed = 7 }, sink));
        Assert.Equal(reseed, PhaseRuns(reports));
        Assert.Equal((string[])["CloneArchitecture", "BindRngConfig"],
            StagesOf(reports, BuildPhase.Concretize));
        Assert.True(reports[^1].IsComplete);
        Assert.DoesNotContain(reports[..^1], r => r.IsComplete);
    }
}
