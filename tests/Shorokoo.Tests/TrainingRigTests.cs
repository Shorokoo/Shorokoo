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
            ckpt = rig.TrainStep(ckpt, inBatch, outBatch);
        return (rig, ckpt, inBatch, outBatch);
    }

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
        var trainingGraph = TrainingGraphBuilder.PrepareForTrainingAsFast(
            ScalarMultiplyModel.ComputationGraph.ToInternal(),
            L2Loss.ComputationGraph.ToInternal());
        var lowered = TrainingLoop.LowerTrainingGraph(trainingGraph);
        Assert.NotNull(lowered);
        Assert.NotEmpty(lowered.Nodes);

        var modelGraph = ScalarMultiplyModel.ComputationGraph.ToInternal();
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
            var stepped = derived.TrainStep(derived.CreateInitialCheckpoint(), input, target);
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
        Eq(0.25f, cw.At(0));
        Eq(1.0f, cw.At(3));
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

        var initial = runtimeRig.CreateInitialCheckpoint();
        Assert.Equal(0, initial.Step);
        float w0 = Weight(runtimeRig, initial);

        var stepA = runtimeRig.TrainStep(initial, runtimeRig.MakeHyperparameters(0.1f), inputBatch, targetBatch);
        var stepB = runtimeRig.TrainStep(initial, runtimeRig.MakeHyperparameters(("learningRate", 0.3f)), inputBatch, targetBatch);
        Assert.Equal(1, stepA.Step);
        float deltaA = w0 - Weight(runtimeRig, stepA);
        float deltaB = w0 - Weight(runtimeRig, stepB);
        Assert.True(MathF.Abs(deltaA) > 1e-4f);
        Assert.True(MathF.Abs(stepA.Loss!.Value - stepB.Loss!.Value) < 1e-4f);
        Assert.True(MathF.Abs(deltaB - 3f * deltaA) < 1e-4f);

        Assert.Throws<InvalidOperationException>(() => runtimeRig.TrainStep(initial, inputBatch, targetBatch));
        Assert.Throws<ArgumentException>(() => runtimeRig.MakeHyperparameters(("bogus", 0.1f)));

        var schedRig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            sample, new SGDOptimizerHyperparameters { LearningRate = Schedules.Linear(0.2f, 0.0f, 4) });
        Assert.Empty(schedRig.HyperparameterStructDef.Fields);
        Assert.Empty(schedRig.DynamicHyperparameterIndices);
        float swAuto = Weight(schedRig, schedRig.TrainStep(schedRig.CreateInitialCheckpoint(), inputBatch, targetBatch));
        float swRef = Weight(runtimeRig, runtimeRig.TrainStep(
            runtimeRig.CreateInitialCheckpoint(), runtimeRig.MakeHyperparameters(0.2f), inputBatch, targetBatch));
        Assert.True(MathF.Abs(swAuto - swRef) < 1e-5f);

        var adamRig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, AdamWOptimizer.ComputationGraph,
            sample, new AdamWOptimizerHyperparameters { LearningRate = Schedules.Constant(0.01f) });
        Assert.Empty(adamRig.DynamicHyperparameterIndices);
        Assert.Empty(adamRig.HyperparameterStructDef.Fields);
        var adamStep = adamRig.TrainStep(adamRig.CreateInitialCheckpoint(), inputBatch, targetBatch);
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
            inputBatch, targetBatch);
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
            cosineCkpt = cosineRig.TrainStep(cosineCkpt, inputBatch, targetBatch);
            hostCkpt = runtimeRig.TrainStep(hostCkpt, runtimeRig.MakeHyperparameters(cosine.At(s)),
                inputBatch, targetBatch);
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
            moduleCkpt = moduleRig.TrainStep(moduleCkpt, inputBatch, targetBatch);
            moduleRefCkpt = runtimeRig.TrainStep(moduleRefCkpt, runtimeRig.MakeHyperparameters(0.3f - 0.05f * s),
                inputBatch, targetBatch);
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
        Assert.All(FlattenStruct(stepCountingRig.TrainStep(initial, inputBatch, targetBatch).OptimizerState),
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

        var trainResult = rig.Train(initial, [inputBatch], [targetBatch], numEpochs: 1);
        Assert.Single(trainResult.EpochLosses);
        Assert.NotNull(trainResult.FinalCheckpoint);

        var stepResult = rig.TrainStep(initial, inputBatch, targetBatch);
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
    public void TestTrainStepAndCheckpointCounterSemanticsCoverage()
    {
        var (sample, inputBatch, targetBatch) = ScalarMultiplyBatches();

        var (adamRig, trained, adamIn, adamOut) = BuildTrainedAdamWRig(steps: 1);
        var carried = adamRig.TrainStep(
            new TrainingCheckpoint(trained.TrainableParams, trained.ModelState, trained.OptimizerState,
                step: trained.Step, epoch: 3, batchIndex: 12),
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
                new TrainingCheckpoint(seed.TrainableParams, seed.ModelState, seed.OptimizerState, step: s, epoch: e),
                inputBatch, targetBatch);
            var refStep = refRig.TrainStep(
                new TrainingCheckpoint(refSeed.TrainableParams, refSeed.ModelState, refSeed.OptimizerState, step: s, epoch: e),
                refRig.MakeHyperparameters(Lr(s, e)), inputBatch, targetBatch);
            Assert.True(MathF.Abs(Weight(schedRig, modStep) - Weight(refRig, refStep)) < 1e-5f);
        }

        var stepped = schedRig.TrainStep(seed, inputBatch, targetBatch, epoch: 4, batchNumber: 7);
        Assert.Equal(seed.Step + 1, stepped.Step);
        Assert.Equal(4, stepped.Epoch);
        Assert.Equal(7, stepped.BatchIndex);
        Assert.True(float.IsFinite(stepped.Loss!.Value));
        Assert.Same(schedRig, stepped.Rig);
        var explicitRef = schedRig.TrainStep(
            new TrainingCheckpoint(seed.TrainableParams, seed.ModelState, seed.OptimizerState, step: 0, epoch: 4),
            inputBatch, targetBatch);
        Assert.True(MathF.Abs(Weight(schedRig, stepped) - Weight(schedRig, explicitRef)) < 1e-6f);

        Assert.Throws<ArgumentOutOfRangeException>(
            () => schedRig.TrainStep(seed, inputBatch, targetBatch, epoch: -1, batchNumber: 0));
        Assert.Throws<ArgumentOutOfRangeException>(
            () => schedRig.TrainStep(seed, inputBatch, targetBatch, epoch: 0, batchNumber: -1));

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
            var vals = ((TensorData)batch.Input.Fields["input"]).As<float32>().AccessMemory().ToArray();
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
            var vals = ((TensorData)batch.Input.Fields["input"]).As<float32>().AccessMemory().ToArray();
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
            var midCkpt = new TrainingCheckpoint(
                ckpt0.TrainableParams, ckpt0.ModelState, ckpt0.OptimizerState,
                step: 2, epoch: lastUsed.Epoch, batchIndex: lastUsed.BatchIndex);
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

        var stepped = rig.TrainStep(rig.CreateInitialCheckpoint(), input, target);
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
}

[Trait("Domain", "Training")]
[Trait("Purpose", "Coverage")]
public class TrainingRigCheckpointCoverageTests
{
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
                ckpt = rigA.TrainStep(ckpt, inputBatch, targetBatch);
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

            var resumed = rigB.TrainStep(loaded, inputBatch, targetBatch);
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
            adamCkpt = adamRig.TrainStep(adamCkpt, inputBatch, targetBatch);

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
        var ckptV2 = new TrainingCheckpoint(
            ckptV1.TrainableParams, ckptV1.ModelState, ckptV1.OptimizerState, step: 7);

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
        var ckpt = new TrainingCheckpoint(
            ckpt0.TrainableParams, ckpt0.ModelState, ckpt0.OptimizerState, step: 5);

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
        var big = new TrainingCheckpoint(
            trained.TrainableParams, trained.ModelState, trained.OptimizerState,
            step: bigStep, epoch: bigEpoch, batchIndex: bigBatch, rig: trained.Rig);

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

        var ckpt = new TrainingCheckpoint(
            trained.TrainableParams, trained.ModelState, trained.OptimizerState,
            step: trained.Step, epoch: 7, batchIndex: 340, rig: trained.Rig);
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

        var unset = new TrainingCheckpoint(
            trained.TrainableParams, trained.ModelState, trained.OptimizerState,
            step: trained.Step, rig: trained.Rig);
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

        var stepped = rig.TrainStep(initial, input, target);
        Assert.Same(rig, stepped.Rig);
        Assert.NotNull(stepped.Loss);
        Assert.True(float.IsFinite(stepped.Loss!.Value));

        var moved = stepped.WithStep(42);
        Assert.Same(rig, moved.Rig);
        Assert.Equal(stepped.Loss, moved.Loss);
        Assert.Equal(stepped.Loss, stepped.WithEpoch(3).Loss);

        var bare = new TrainingCheckpoint(initial.TrainableParams, initial.ModelState, initial.OptimizerState);
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

        var bare = new TrainingCheckpoint(
            seed.TrainableParams, seed.ModelState, seed.OptimizerState, step: 5, epoch: 2, batchIndex: 1);
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

        var reference = rigA.TrainStep(ckpt, inBatch, outBatch);

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

            var resumed = rigB.TrainStep(loaded, inBatch, outBatch);
            Assert.Equal(3, resumed.Step);
            Assert.Equal(reference.Loss!.Value, resumed.Loss!.Value);
            Assert.Equal(FlattenStruct(reference.TrainableParams), FlattenStruct(resumed.TrainableParams));
            Assert.Equal(FlattenStruct(reference.OptimizerState), FlattenStruct(resumed.OptimizerState));

            var inferenceModel = Persistence.Load(path);
            Assert.Equal(GraphKind.ConcreteModel, inferenceModel.Kind);
            var probe = TensorData(ScalarInputShape, [5f, 6f, 7f, 8f]);
            var loadedOut = ComputeContext.Default.Execute(inferenceModel, probe)[0].ToTensorData().As<float32>().AccessMemory().ToArray();
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
        var bnCkpt = new TrainingCheckpoint(
            bnSeed.TrainableParams, bnSeed.ModelState, bnSeed.OptimizerState, step: 11, rig: bnRig);
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
        var one = new TrainingCheckpoint(
            trained.TrainableParams, trained.ModelState, trained.OptimizerState, step: 1, rig: trained.Rig);

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
                var ex = Assert.Throws<System.IO.InvalidDataException>(() =>
                    Persistence.LoadTrainingCheckpointFromSkpt(infPath,
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

        var strippedPath = TempPath("skpt_old") + ".skpt";
        try
        {
            var two = new TrainingCheckpoint(
                trained.TrainableParams, trained.ModelState, trained.OptimizerState, step: 2, rig: trained.Rig);
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
            Assert.Equal(1, Persistence.LoadTrainingCheckpoint(
                flatPath, reader.TrainableParamStructDef, reader.ModelStateDef, reader.OptimizerStateDef).Step);
            Assert.Equal(1, Persistence.LoadTrainingCheckpointFromSkpt(
                skptPath, reader.TrainableParamStructDef, reader.ModelStateDef, reader.OptimizerStateDef).Step);

            var flatGotSkpt = Assert.Throws<System.IO.InvalidDataException>(
                () => reader.LoadCheckpoint(skptPath));
            Assert.Contains(".skpt", flatGotSkpt.Message);
            Assert.Contains("LoadCheckpointFromSkpt", flatGotSkpt.Message);
            var skptGotFlat = Assert.Throws<System.IO.InvalidDataException>(
                () => reader.LoadCheckpointFromSkpt(flatPath));
            Assert.Contains("safetensors", skptGotFlat.Message);
            Assert.Contains("LoadCheckpoint(path)", skptGotFlat.Message);
            Assert.Contains("LoadTrainingCheckpointFromSkpt", Assert.Throws<System.IO.InvalidDataException>(
                () => Persistence.LoadTrainingCheckpoint(
                    skptPath, reader.TrainableParamStructDef, reader.ModelStateDef, reader.OptimizerStateDef)).Message);
            Assert.Contains("Persistence.LoadTrainingCheckpoint", Assert.Throws<System.IO.InvalidDataException>(
                () => Persistence.LoadTrainingCheckpointFromSkpt(
                    flatPath, reader.TrainableParamStructDef, reader.ModelStateDef, reader.OptimizerStateDef)).Message);
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
        var reference = rigA.TrainStep(ckpt, inBatch, outBatch);

        var path = TempPath("rigload") + ".skpt";
        try
        {
            Persistence.SaveTrainingCheckpointToSkpt(ckpt, path);

            var (rig2, loaded) = TrainingRig.Load(path);
            Assert.Same(rig2, loaded.Rig);
            Assert.Equal(ckpt.Step, loaded.Step);
            Assert.Equal(FlattenStruct(ckpt.TrainableParams), FlattenStruct(loaded.TrainableParams));
            Assert.Equal(FlattenStruct(ckpt.OptimizerState), FlattenStruct(loaded.OptimizerState));

            var resumed = rig2.TrainStep(loaded, inBatch, outBatch);
            Assert.Equal(reference.Step, resumed.Step);
            Assert.Equal(reference.Loss!.Value, resumed.Loss!.Value);
            Assert.Equal(FlattenStruct(reference.TrainableParams), FlattenStruct(resumed.TrainableParams));
            Assert.Equal(FlattenStruct(reference.OptimizerState), FlattenStruct(resumed.OptimizerState));

            var probe = TensorData(ScalarInputShape, [5f, 6f, 7f, 8f]);
            var a = ComputeContext.Default.Execute(loaded.ToInferenceModel(), probe)[0].ToTensorData().As<float32>().AccessMemory().ToArray();
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
            cosineCkpt = cosineRig.TrainStep(cosineCkpt, inBatch, outBatch);
        var cosineReference = cosineRig.TrainStep(cosineCkpt, inBatch, outBatch);

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

            var resumed = rig2.TrainStep(loaded, inBatch, outBatch);
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
                var at = new TrainingCheckpoint(
                    stepEpochInitial.TrainableParams, stepEpochInitial.ModelState, stepEpochInitial.OptimizerState,
                    step: s, epoch: e);
                float wOriginal = Weight(stepEpochRig, stepEpochRig.TrainStep(at, stepEpochIn, stepEpochTarget));
                float wReloaded = Weight(reloaded, reloaded.TrainStep(at, stepEpochIn, stepEpochTarget));
                Assert.True(MathF.Abs(wOriginal - wReloaded) < 1e-6f);
            }
        }
        finally { if (File.Exists(stepEpochPath)) File.Delete(stepEpochPath); }
    }

    [Fact]
    public void TestSkptDirectoryFormTrainingCheckpointRoundTripCoverage()
    {
        var (rig, ckpt, inBatch, outBatch) = BuildTrainedAdamWRig(steps: 2);
        var reference = rig.TrainStep(ckpt, inBatch, outBatch);
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
            var resumed = rig2.TrainStep(loaded, inBatch, outBatch);
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
                ? rig.TrainStep(ckpt, inputBatch, targetBatch)
                : rig.TrainStep(ckpt, hypers, inputBatch, targetBatch));
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
        var ckpt = rig.TrainStep(rig.CreateInitialCheckpoint(), inputBatch, targetBatch);
        var reference = rig.TrainStep(ckpt, inputBatch, targetBatch);

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

            var resumed = rig2.TrainStep(loaded, inputBatch, targetBatch);
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
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
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

        float Step(TrainingRig rig, TensorDataStruct? hypers = null)
        {
            var ckpt = rig.CreateInitialCheckpoint();
            return Weight(rig, hypers is null
                ? rig.TrainStep(ckpt, inputBatch, targetBatch)
                : rig.TrainStep(ckpt, hypers, inputBatch, targetBatch));
        }

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
        Assert.True(MathF.Abs(Step(bakedRig) - Step(runtimeRig, matching)) < 1e-5f);

        var doubled = runtimeRig.MakeHyperparameters(
            ("perElementRate", Rate(0.1f, 0.2f, 0.4f, 0.8f)), ("gain", 2f));
        Assert.True(MathF.Abs(Step(runtimeRig, doubled) - Step(runtimeRig, matching)) > 1e-4f);

        var scheduledRig = VectorRig(new VectorRateOptimizerHyperparameters
            { PerElementRate = Hyperparameter.Scheduled(VectorRateScheduler.ComputationGraph) });
        Assert.Equal((long[])[4L], scheduledRig.HyperparameterShapes[0].Dims);
        Assert.Empty(scheduledRig.HyperparameterStructDef.Fields);
        Assert.True(MathF.Abs(Step(scheduledRig) - Step(bakedRig)) < 1e-5f);

        var scheduledCkpt = scheduledRig.CreateInitialCheckpoint();
        var hostCkpt = runtimeRig.CreateInitialCheckpoint();
        for (int s = 0; s < 3; s++)
        {
            scheduledCkpt = scheduledRig.TrainStep(scheduledCkpt, inputBatch, targetBatch);
            hostCkpt = runtimeRig.TrainStep(hostCkpt, runtimeRig.MakeHyperparameters(
                ("perElementRate", Rate(0.1f - 0.01f * s, 0.2f - 0.01f * s, 0.4f - 0.01f * s, 0.8f - 0.01f * s)),
                ("gain", 1f)), inputBatch, targetBatch);
            Assert.True(MathF.Abs(Weight(scheduledRig, scheduledCkpt) - Weight(runtimeRig, hostCkpt)) < 1e-5f);
        }

        var stateRig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
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
        Assert.Throws<ArgumentException>(() => VectorRig(new VectorRateOptimizerHyperparameters
            { PerElementRate = 0.1f }));
        Assert.Throws<ArgumentException>(() => VectorRig(new VectorRateOptimizerHyperparameters
            { PerElementRate = Hyperparameter.Runtime(2L, 2L) }));
        Assert.Throws<ArgumentException>(() => VectorRig(new VectorRateOptimizerHyperparameters
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
            inputBatch, targetBatch);
        var reference = rig.TrainStep(ckpt, rig.MakeHyperparameters(1f), inputBatch, targetBatch);

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

            var resumed = rig2.TrainStep(loaded, rig2.MakeHyperparameters(1f), inputBatch, targetBatch);
            Assert.Equal(FlattenStruct(reference.TrainableParams), FlattenStruct(resumed.TrainableParams));
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }
}

[Trait("Domain", "Training")]
[Trait("Purpose", "Coverage")]
public class BuildProgressCoverageTests
{
    private static (List<BuildProgress> Reports, ComputeContext Context) Watched()
    {
        var reports = new List<BuildProgress>();
        return (reports, new ComputeContext { Progress = new BuildProgressHandler(reports.Add) });
    }

    private static List<BuildPhase> PhaseRuns(List<BuildProgress> reports)
    {
        var runs = new List<BuildPhase>();
        foreach (var r in reports)
            if (runs.Count == 0 || runs[^1] != r.Phase) runs.Add(r.Phase);
        return runs;
    }

    private static bool Reported(List<BuildProgress> reports, BuildPhase phase, string stage)
        => reports.Any(r => r.Phase == phase && r.Stage == stage);

    [Fact]
    public void TestFromScratchReportsEveryBuildPhaseInOrderCoverage()
    {
        var reports = new List<BuildProgress>();
        var reportThreads = new List<int>();
        var ctx = new ComputeContext
        {
            Progress = new BuildProgressHandler(r =>
            {
                reports.Add(r);
                reportThreads.Add(Environment.CurrentManagedThreadId);
            }),
        };
        var (sample, _, _) = ScalarMultiplyBatches();
        var buildThread = Environment.CurrentManagedThreadId;

        var rig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            AdamWOptimizer.ComputationGraph, sample,
            new AdamWOptimizerHyperparameters { LearningRate = 0.1f }, mergeContext: ctx);

        Assert.NotNull(rig.TrainingStepPureGraph);
        Assert.Equal((BuildPhase[])[BuildPhase.Concretize, BuildPhase.TrainingStep, BuildPhase.Initialize],
            PhaseRuns(reports).ToArray());
        Assert.True(Reported(reports, BuildPhase.Concretize, "InlineModulesAndFunctions"));
        Assert.True(Reported(reports, BuildPhase.Concretize, "ExpandAutoGrad"));
        Assert.True(Reported(reports, BuildPhase.TrainingStep, "NormalizeOptimizerGraph"));
        Assert.True(Reported(reports, BuildPhase.TrainingStep, "ComposeModelLossAndAutoGrad"));
        Assert.True(Reported(reports, BuildPhase.TrainingStep, "ExpandAutoGrad"));
        Assert.True(Reported(reports, BuildPhase.Initialize, "InitializeModelParams"));
        Assert.True(Reported(reports, BuildPhase.Initialize, "InitializeOptimizerState"));
        Assert.True(Reported(reports, BuildPhase.Initialize, "OptimizeTrainingStepGraph"));
        Assert.Equal(BuildPhase.Initialize, reports[^1].Phase);
        Assert.Equal("Done", reports[^1].Stage);
        Assert.Equal(reports.Select(r => r.Elapsed).OrderBy(e => e), reports.Select(r => r.Elapsed));
        Assert.Equal(reports.Count, reportThreads.Count);
        Assert.All(reportThreads, t => Assert.Equal(buildThread, t));
    }

    [Fact]
    public void TestToConcreteArchitectureReportsLoweringStagesAndIsSilentWithoutASinkCoverage()
    {
        var (reports, ctx) = Watched();
        var model = ScalarMultiplyModel.ComputationGraph;
        var hints = new ModelParamList(
            [new KeyValuePair<string, TensorData>(model.ToInternal().Inputs[0].ToString(), TensorData([4L], new float[4]))],
            ModelParamType.InputParam);

        Assert.NotNull(model.ToConcreteArchitecture(hints, ctx));
        Assert.All(reports, r => Assert.Equal(BuildPhase.Concretize, r.Phase));
        Assert.Equal("Clone", reports[0].Stage);
        Assert.Equal("SimplifyAfterAutoGrad", reports[^1].Stage);
        Assert.Equal("[   1.5s] Concretize: Clone",
            new BuildProgress(BuildPhase.Concretize, "Clone", TimeSpan.FromSeconds(1.5)).ToString());

        var silent = new ComputeContext();
        Assert.Null(silent.Progress);
        Assert.NotNull(model.ToConcreteArchitecture(hints, silent));

        var second = new List<BuildProgress>();
        ctx.Progress = new BuildProgressHandler(second.Add);
        Assert.NotNull(model.ToConcreteArchitecture(hints, ctx));
        Assert.Equal(reports.Select(r => r.Stage), second.Select(r => r.Stage));
    }
}
