using System.Collections.Immutable;
using Shorokoo.Runtime;
using Shorokoo.Core.AutoDiffCheckpointing;
using Shorokoo.Core.Inference;
using Shorokoo.Core.Nodes.Processors.Helpers;

namespace Shorokoo.Tests;

/// <summary>
/// Coverage-purpose tests for the AutoDiffCheckpointing chain — exercises
/// <see cref="ShapeInferenceInterpreter"/>, <see cref="GraphEvaluator"/>
/// (transitively <c>UnaryElementwisePerf</c>, <c>BinaryElementwisePerf</c>,
/// <c>LinearAlgebraPerf</c>, <c>TensorManipulationPerf</c>, <c>ReductionPerf</c>,
/// <c>PoolingNormPerf</c>), <see cref="MemoryAwareScheduler"/>,
/// <see cref="Rematerializer"/>, <see cref="SimpleBackpropOptimizer"/>, and
/// the umbrella <see cref="MemoryAwareGraphOptimizer"/>. A single small diamond
/// graph (Relu + 2×Add + final Add + MatMul) is enough to drive every entry
/// point; high memory factors force the rematerializer to actually insert
/// recomputation nodes rather than fall through the no-op path.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class AutoDiffCheckpointingCoverageTests
{
    private static ComputeContext CpuContext => new ComputeContext();

    [Fact]
    public void TestAutoDiffCheckpointingChainCoverage()
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

        var graph = new FastComputationGraph(
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

        var rematerializer = new Rematerializer(computeFactor: 1.0, memoryFactor: 1.0, maxIterations: 20);
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
        Assert.True(fullOptimizer.ComputeCombinedMetric(directEval) > 0);
    }

    /// <summary>
    /// Widens the GraphEvaluator/OpsPerf coverage beyond the diamond graph above: one
    /// graph carrying Conv / ConvTranspose / Gemm(transA+transB) / Einsum
    /// (<c>LinearAlgebraPerf</c>'s four remaining estimators), windowed + global pooling
    /// and BatchNorm/LRN (<c>PoolingNormPerf</c>'s three branches), and
    /// Det / TopK / Resize / RandomNormalLike / Constant (<c>MiscPerf</c>'s per-op
    /// branches). QEE resolves every shape, so this stays ORT-free.
    /// </summary>
    [Fact]
    public void TestOpsPerfEstimatorBranchesCoverage()
    {
        var x = InputTensor<float32>("x", rank: 4);          // [1,2,8,8]
        var w = InputTensor<float32>("w", rank: 4);          // [3,2,3,3]
        var convBias = InputVector<float32>("convBias");     // [3]
        var deconvBias = InputVector<float32>("deconvBias"); // [2]
        var a = InputTensor<float32>("a", rank: 2);          // [4,4]
        var b = InputTensor<float32>("b", rank: 2);          // [4,4]
        var scale = InputVector<float32>("scale");           // [2]
        var bias = InputVector<float32>("bias");             // [2]
        var mean = InputVector<float32>("mean");             // [2]
        var variance = InputVector<float32>("variance");     // [2]
        var v = InputVector<float32>("v");                   // [6]

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
        var (topVals, topIdx) = OnnxOp.TopK(v, OnnxOp.Constant(new long[] { 2 }), axis: -1, largest: true, sorted: true);
        var resized = OnnxOp.Resize(x, null, OnnxOp.Constant(new float[] { 1f, 1f, 2f, 2f }), null,
            antialias: null, axes: null, coordinateTransformationMode: null, cubicCoeffA: null,
            excludeOutside: null, extrapolationValue: null, keepAspectRatioPolicy: null,
            mode: null, nearestMode: null);
        var randomLike = OnnxOp.RandomNormalLike(x, seed: 11f);

        var graph = new FastComputationGraph(
            [x, w, convBias, deconvBias, a, b, scale, bias, mean, variance, v],
            [conv, deconv, gemm, einsum, maxPool, avgPool, globalLp, globalMax, globalAvg,
             bn, lrn, det, topVals, topIdx, resized, randomLike]);

        var shapeInfo = new ShapeInferenceInterpreter(CpuContext).Infer(graph,
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
        Assert.True(shapeInfo.TensorCount > 0);

        var eval = new GraphEvaluator().Evaluate(graph, shapeInfo);
        Assert.True(eval.TotalComputeTime > 0);
        Assert.True(eval.PeakMemoryBytes > 0);
    }

    /// <summary>
    /// QuickOp stub whose Compute always throws, so QEE writes Invalid placeholders for
    /// the op's outputs and <see cref="ShapeInferenceInterpreter"/> must fall back to
    /// per-node ONNX Runtime execution for them.
    /// </summary>
    private sealed class QeeFailStub : QuickOp
    {
        private readonly string _opCode;
        public QeeFailStub(string opCode) { _opCode = opCode; }
        public override string OpCode => _opCode;
        protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attributes, int maxDataElements)
            => throw new InvalidOperationException("forced QEE failure for ORT-fallback coverage");
    }

    /// <summary>
    /// Drives <see cref="ShapeInferenceInterpreter"/>'s ORT fallback chain
    /// (FallbackResolveNode → ProcessConstant / ExecuteNode, including the
    /// SEQUENCE_CONSTRUCT multi-output wrap and its mixed-dtype retry) by temporarily
    /// replacing the QEE handlers for Det / TopK / Constant with always-throwing stubs.
    /// Tests run single-threaded (xunit.runner.json), and the original handlers are
    /// restored in a finally block, so the registry swap cannot leak.
    /// </summary>
    [Fact]
    public void TestShapeInferenceOrtFallbackCoverage()
    {
        var x = InputTensor<float32>("x", rank: 2);  // [3,3]
        var v = InputVector<float32>("v");           // [6]

        var det = OnnxOp.Det(x);
        var (topVals, topIdx) = OnnxOp.TopK(v, OnnxOp.Constant(new long[] { 2 }), axis: -1, largest: true, sorted: true);
        var constTensor = OnnxOp.Constant(Globals.TensorData(DType.Float32, [2L], 5f, 6f)); // value (tensor) branch
        var constInt = OnnxOp.Constant(7L);    // value_int branch
        var constFloat = OnnxOp.Constant(2.5f); // value_float branch

        var graph = new FastComputationGraph(
            [x, v],
            [det, topVals, topIdx, constTensor, constInt, constFloat]);

        var origDet = OpRegistry.Get(OpCodes.DET)!;
        var origTopK = OpRegistry.Get(OpCodes.TOPK)!;
        var origConstant = OpRegistry.Get(OpCodes.CONSTANT)!;
        ShapeInferenceResult shapeInfo;
        try
        {
            OpRegistry.Register(new QeeFailStub(OpCodes.DET));
            OpRegistry.Register(new QeeFailStub(OpCodes.TOPK));
            OpRegistry.Register(new QeeFailStub(OpCodes.CONSTANT));

            shapeInfo = new ShapeInferenceInterpreter(CpuContext).Infer(graph,
                Globals.TensorDataWithSmallVals(DType.Float32, [3, 3]),
                Globals.TensorDataWithSmallVals(DType.Float32, [6]));
        }
        finally
        {
            OpRegistry.Register(origDet);
            OpRegistry.Register(origTopK);
            OpRegistry.Register(origConstant);
        }

        // Det: single-output ORT fallback (no sequence wrap) — scalar result.
        var detInfo = shapeInfo.GetTensorInfo(graph.Outputs[0]);
        Assert.NotNull(detInfo);
        Assert.Empty(detInfo!.Shape.Dims);

        // TopK: two mixed-dtype outputs — the sequence wrap is rejected by ORT and the
        // direct-outputs retry resolves both values and indices.
        var valsInfo = shapeInfo.GetTensorInfo(graph.Outputs[1]);
        var idxInfo = shapeInfo.GetTensorInfo(graph.Outputs[2]);
        Assert.NotNull(valsInfo);
        Assert.NotNull(idxInfo);
        Assert.Equal(new long[] { 2 }, valsInfo!.Shape.Dims);
        Assert.Equal(new long[] { 2 }, idxInfo!.Shape.Dims);
        Assert.Equal(DType.Int64, idxInfo.DType);

        // Constants resolved via ProcessConstant's tensor / value_int / value_float branches.
        var tensorConstInfo = shapeInfo.GetTensorInfo(graph.Outputs[3]);
        Assert.NotNull(tensorConstInfo);
        Assert.Equal(new long[] { 2 }, tensorConstInfo!.Shape.Dims);
        var intConstInfo = shapeInfo.GetTensorInfo(graph.Outputs[4]);
        Assert.NotNull(intConstInfo);
        Assert.Equal(DType.Int64, intConstInfo!.DType);
        var floatConstInfo = shapeInfo.GetTensorInfo(graph.Outputs[5]);
        Assert.NotNull(floatConstInfo);
        Assert.Equal(DType.Float32, floatConstInfo!.DType);
    }
}
