using System.Collections.Immutable;
using Shorokoo.Runtime;
using Shorokoo.Core.AutoDiffCheckpointing;
using Shorokoo.Core.Inference;
using Shorokoo.Core.Nodes.Processors.Helpers;

namespace Shorokoo.Tests;

/// <summary>
/// Coverage for the AutoDiffCheckpointing chain — <see cref="ShapeInferenceInterpreter"/>,
/// <see cref="GraphEvaluator"/> (and the <c>OpsPerf</c> estimators behind it),
/// <see cref="MemoryAwareScheduler"/>, <see cref="Rematerializer"/>,
/// <see cref="SimpleBackpropOptimizer"/> and <see cref="MemoryAwareGraphOptimizer"/>.
///
/// <para><see cref="TestComputeMemoryObjectiveIsScaleFreeCoverage"/> guards the property the
/// whole pass rests on: the objective must weigh the same proportional trade identically at
/// any model size. A byte-scaled coefficient — what this replaced — passes every other test
/// in the suite while silently reducing the pass to a no-op on real models.</para>
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class AutoDiffCheckpointingCoverageTests
{
    private static ComputeContext CpuContext => new ComputeContext();

    [Fact]
    public void TestAutoDiffCheckpointingChainAndOpsPerfEstimatorBranchesCoverage()
    {
        var input = InputTensor<float32>("input", rank: 2);
        var weights = InputTensor<float32>("weights", rank: 2);
        var bias1 = InputVector<float32>("bias1");
        var bias2 = InputVector<float32>("bias2");

        var mat = OnnxOp.MatMul(input, weights);
        var activation = OnnxOp.Relu(mat);
        var branch1 = OnnxOp.Add(activation, bias1);
        var branch2 = OnnxOp.Add(activation, bias2);
        var final = OnnxOp.Add(branch1, branch2);

        var graph = new InternalComputationGraph(
            ImmutableArray.Create<Variable>(input, weights, bias1, bias2),
            ImmutableArray.Create((Variable)final));

        var inputData = Globals.TensorDataWithSmallVals(DType.Float32, [256, 256]);
        var weightsData = Globals.TensorDataWithSmallVals(DType.Float32, [256, 256]);
        var biasData = Globals.TensorDataWithSmallVals(DType.Float32, [256]);

        var shapeInterpreter = new ShapeInferenceInterpreter(CpuContext);
        var shapeInfo = shapeInterpreter.Infer(graph, inputData, weightsData, biasData, biasData);
        Assert.True(shapeInfo.TensorCount > 0);

        var evaluator = new GraphEvaluator();
        var eval = evaluator.Evaluate(graph, shapeInfo);
        Assert.True(eval.TotalComputeTime > 0);
        Assert.True(eval.PeakMemoryBytes > 0);

        var scheduler = new MemoryAwareScheduler();
        var reordered = scheduler.Reorder(graph, shapeInfo);
        Assert.Equal(graph.Nodes.Count, reordered.Nodes.Count);

        var rematerializer = new Rematerializer(
            new ComputeMemoryObjective(1.0, 1.0, eval), maxIterations: 20);
        var rematGraph = rematerializer.Apply(graph, shapeInfo);
        Assert.True(rematGraph.Nodes.Count >= graph.Nodes.Count);

        var backprop = new SimpleBackpropOptimizer(computeFactor: 1.0, memoryFactor: 1.0, maxIterations: 20);
        var backpropResult = backprop.Optimize(graph, shapeInfo);
        Assert.NotNull(backpropResult.OptimizedGraph);
        Assert.True(backpropResult.Evaluation.TotalComputeTime > 0);

        var fullOptimizer = new MemoryAwareGraphOptimizer(
            computeFactor: 1.0,
            memoryFactor: 1.0,
            maxRematerializationIterations: 20,
            shapeInference: new ShapeInferenceInterpreter(CpuContext));
        var fullResult = fullOptimizer.Optimize(graph, inputData, weightsData, biasData, biasData);
        Assert.NotNull(fullResult.OptimizedGraph);
        Assert.NotEmpty(fullResult.StrategyName);
        Assert.True(fullResult.AllStrategies.Count > 0);

        var directEval = fullOptimizer.EvaluateGraph(graph, inputData, weightsData, biasData, biasData);
        Assert.True(directEval.PeakMemoryBytes > 0);
        Assert.True(fullOptimizer.ComputeCombinedMetric(directEval, directEval) > 0);

        var x = InputTensor<float32>("x", rank: 4);
        var w = InputTensor<float32>("w", rank: 4);
        var convBias = InputVector<float32>("convBias");
        var deconvBias = InputVector<float32>("deconvBias");
        var a = InputTensor<float32>("a", rank: 2);
        var b = InputTensor<float32>("b", rank: 2);
        var scale = InputVector<float32>("scale");
        var bias = InputVector<float32>("bias");
        var mean = InputVector<float32>("mean");
        var variance = InputVector<float32>("variance");
        var v = InputVector<float32>("v");

        var conv = OnnxOp.Conv(x, w, convBias, AutoPad.NotSet,
            dilations: [1L, 1L], group: 1, kernelShape: [3L, 3L], pads: [1L, 1L, 1L, 1L], strides: [1L, 1L]);
        var deconv = OnnxOp.ConvTranspose(conv, w, deconvBias, AutoPad.NotSet,
            dilations: [1L, 1L], group: 1, kernelShape: [3L, 3L],
            outputPadding: null, outputShape: null, pads: [1L, 1L, 1L, 1L], strides: [1L, 1L]);
        var gemm = OnnxOp.Gemm(a, b, c: null, alpha: 1f, beta: 1f, transA: 1, transB: 1);
        var einsum = OnnxOp.Einsum([a, b], "ij,jk->ik");
        var maxPool = OnnxOp.MaxPool(x, kernelShape: [2L, 2L], strides: [2L, 2L]);
        var avgPool = OnnxOp.AveragePool(x, null, null, null, null, [2L, 2L], null, [2L, 2L]);
        var globalLp = OnnxOp.GlobalLpPool(x);
        var globalMax = OnnxOp.GlobalMaxPool(x);
        var globalAvg = OnnxOp.GlobalAveragePool(x);
        var bn = OnnxOp.BatchNormalization(x, scale, bias, mean, variance,
            epsilon: 1e-5f, momentum: null, trainingMode: null);
        var lrn = OnnxOp.Lrn(x, size: 3);
        var det = OnnxOp.Det(a);
        var (topVals, topIdx) = OnnxOp.TopK(v, OnnxOp.Constant((long[])[2L]), axis: -1, largest: true, sorted: true);
        var resized = OnnxOp.Resize(x, null, OnnxOp.Constant((float[])[1f, 1f, 2f, 2f]), null,
            antialias: null, axes: null, coordinateTransformationMode: null, cubicCoeffA: null,
            excludeOutside: null, extrapolationValue: null, keepAspectRatioPolicy: null,
            mode: null, nearestMode: null);
        var randomLike = OnnxOp.RandomNormalLike(x, seed: 11f);

        var perfGraph = new InternalComputationGraph(
            [x, w, convBias, deconvBias, a, b, scale, bias, mean, variance, v],
            [conv, deconv, gemm, einsum, maxPool, avgPool, globalLp, globalMax, globalAvg,
             bn, lrn, det, topVals, topIdx, resized, randomLike]);

        var perfShapeInfo = new ShapeInferenceInterpreter(CpuContext).Infer(perfGraph,
            Globals.TensorDataWithSmallVals(DType.Float32, [1, 2, 8, 8]),
            Globals.TensorDataWithSmallVals(DType.Float32, [3, 2, 3, 3]),
            Globals.TensorDataWithSmallVals(DType.Float32, [3]),
            Globals.TensorDataWithSmallVals(DType.Float32, [2]),
            Globals.TensorDataWithSmallVals(DType.Float32, [4, 4]),
            Globals.TensorDataWithSmallVals(DType.Float32, [4, 4]),
            Globals.TensorDataWithSmallVals(DType.Float32, [2]),
            Globals.TensorDataWithSmallVals(DType.Float32, [2]),
            Globals.TensorDataWithSmallVals(DType.Float32, [2]),
            Globals.TensorDataWithSmallVals(DType.Float32, [2]),
            Globals.TensorDataWithSmallVals(DType.Float32, [6]));
        Assert.True(perfShapeInfo.TensorCount > 0);

        var perfEval = new GraphEvaluator().Evaluate(perfGraph, perfShapeInfo);
        Assert.True(perfEval.TotalComputeTime > 0);
        Assert.True(perfEval.PeakMemoryBytes > 0);
    }

    /// <summary>QuickOp stub whose Compute always throws, forcing QEE to write Invalid
    /// placeholders so <see cref="ShapeInferenceInterpreter"/> falls back to ORT.</summary>
    private sealed class QeeFailStub : QuickOp
    {
        private readonly string _opCode;
        public QeeFailStub(string opCode) { _opCode = opCode; }
        public override string OpCode => _opCode;
        protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attributes, int maxDataElements)
            => throw new InvalidOperationException("forced QEE failure for ORT-fallback coverage");
    }

    private static GraphEvaluationResult Eval(double computeTime, long peakBytes)
        => new() { TotalComputeTime = computeTime, PeakMemoryBytes = peakBytes, NodeDetails = [] };

    [Fact]
    public void TestComputeMemoryObjectiveIsScaleFreeCoverage()
    {
        var small = new ComputeMemoryObjective(1.0, 1.0, Eval(10, 1_000));
        var large = new ComputeMemoryObjective(1.0, 1.0, Eval(10_000_000, 1_000_000_000));

        Assert.Equal(small.Score(Eval(11, 500)), large.Score(Eval(11_000_000, 500_000_000)), 9);
        Assert.Equal(2.0, small.Score(Eval(10, 1_000)), 9);
        Assert.Equal(2.0, large.Score(Eval(10_000_000, 1_000_000_000)), 9);
        Assert.Equal(1.5, small.Score(Eval(10, 500)), 9);
        Assert.Equal(1.5, new ComputeMemoryObjective(2.0, 1.0, Eval(10, 1_000)).Score(Eval(5, 500)), 9);

        Assert.False(double.IsNaN(new ComputeMemoryObjective(1.0, 1.0, Eval(0, 0)).Score(Eval(1, 1))));

        Assert.True(small.TradeDelta(extraComputeTime: 1, savedPeakBytes: 500) < 0);
        Assert.True(small.TradeDelta(extraComputeTime: 5, savedPeakBytes: 100) > 0);
        Assert.Equal(small.TradeDelta(1, 500), large.TradeDelta(1_000_000, 500_000_000), 9);
    }

    [Fact]
    public void TestShapeInferenceOrtFallbackResolvesDetTopKAndConstantCoverage()
    {
        var x = InputTensor<float32>("x", rank: 2);
        var v = InputVector<float32>("v");

        var det = OnnxOp.Det(x);
        var (topVals, topIdx) = OnnxOp.TopK(v, OnnxOp.Constant((long[])[2L]), axis: -1, largest: true, sorted: true);
        var constTensor = OnnxOp.Constant(Globals.TensorData(DType.Float32, [2L], 5f, 6f));
        var constInt = OnnxOp.Constant(7L);
        var constFloat = OnnxOp.Constant(2.5f);

        var graph = new InternalComputationGraph(
            [x, v],
            [det, topVals, topIdx, constTensor, constInt, constFloat]);

        ShapeInferenceResult shapeInfo;
        using (OpRegistry.Override(
            new QeeFailStub(OpCodes.DET),
            new QeeFailStub(OpCodes.TOPK),
            new QeeFailStub(OpCodes.CONSTANT)))
        {
            shapeInfo = new ShapeInferenceInterpreter(CpuContext).Infer(graph,
                Globals.TensorDataWithSmallVals(DType.Float32, [3, 3]),
                Globals.TensorDataWithSmallVals(DType.Float32, [6]));
        }

        var detInfo = shapeInfo.GetTensorInfo(graph.Outputs[0]);
        Assert.NotNull(detInfo);
        Assert.Empty(detInfo!.Shape.Dims);

        var valsInfo = shapeInfo.GetTensorInfo(graph.Outputs[1]);
        var idxInfo = shapeInfo.GetTensorInfo(graph.Outputs[2]);
        Assert.NotNull(valsInfo);
        Assert.NotNull(idxInfo);
        Assert.Equal((long[])[2], valsInfo!.Shape.Dims);
        Assert.Equal((long[])[2], idxInfo!.Shape.Dims);
        Assert.Equal(DType.Int64, idxInfo.DType);

        var tensorConstInfo = shapeInfo.GetTensorInfo(graph.Outputs[3]);
        Assert.NotNull(tensorConstInfo);
        Assert.Equal((long[])[2], tensorConstInfo!.Shape.Dims);
        var intConstInfo = shapeInfo.GetTensorInfo(graph.Outputs[4]);
        Assert.NotNull(intConstInfo);
        Assert.Equal(DType.Int64, intConstInfo!.DType);
        var floatConstInfo = shapeInfo.GetTensorInfo(graph.Outputs[5]);
        Assert.NotNull(floatConstInfo);
        Assert.Equal(DType.Float32, floatConstInfo!.DType);
    }
}
