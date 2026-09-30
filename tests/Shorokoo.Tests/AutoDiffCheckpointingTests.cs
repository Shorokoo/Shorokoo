using Shorokoo.Core.Factory;
using Shorokoo.Core.Factory.IR;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Graph;
using Shorokoo.Modules.Losses;
using Shorokoo.Modules.Optimizers;
using System.Collections.Immutable;
using Shorokoo.Runtime;
using Shorokoo.Core.AutoDiffCheckpointing;
using Shorokoo.Core.Utils;
using Shorokoo.Tests.Benchmarks;
using Shorokoo.Core.Graph;
using Shorokoo.Core.AutoDiffCheckpointing.OpsPerf;
using Shorokoo.Core.Interpreter;
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

        var inputData = Globals.TensorDataWithSmallVals(DType.Float32, [512, 512]);
        var weightsData = Globals.TensorDataWithSmallVals(DType.Float32, [512, 512]);
        var biasData = Globals.TensorDataWithSmallVals(DType.Float32, [512]);

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
        var (rematGraph, rematShapeInfo) = rematerializer.Apply(graph, shapeInfo);
        Assert.True(rematGraph.Nodes.Count >= graph.Nodes.Count);
        foreach (var node in rematGraph.Nodes)
            foreach (var output in node.Outputs)
                if (output is not null)
                    Assert.NotNull(rematShapeInfo.GetTensorInfo(output.Value));

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

    private static long ConvWorkspace(long batch, long group, long kernel)
    {
        var x = new TensorShapeInfo(new Shape(batch, 4 * group, 16, 16), DType.Float32, null);
        var w = new TensorShapeInfo(new Shape(8 * group, 4, kernel, kernel), DType.Float32, null);
        var y = new TensorShapeInfo(new Shape(batch, 8 * group, 16, 16), DType.Float32, null);
        return new OpPerfRegistry().Estimate(new OpPerfInput
        {
            OpCode = "Conv",
            InputShapes = [x, w],
            OutputShapes = [y],
            InputMustRemainIntact = [true, true],
            Attributes = new Dictionary<string, object?> { ["group"] = group },
        }).ExtraMemoryBytes;
    }

    [Fact]
    public void TestConvWorkspaceIsPerImagePerGroupCoverage()
    {
        Assert.Equal(4 * 9 * 256 * 4L, ConvWorkspace(1, 1, 3));
        Assert.Equal(ConvWorkspace(1, 1, 3), ConvWorkspace(8, 1, 3));
        Assert.Equal(ConvWorkspace(1, 1, 3), ConvWorkspace(1, 4, 3));
        Assert.Equal(0L, ConvWorkspace(8, 1, 1));
    }

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
    public void TestShapeInferenceSizesSequencesAsTheSumOfTheirElementsCoverage()
    {
        var x = InputTensor<float32>("x", rank: 2);
        var y = InputTensor<float32>("y", rank: 2);
        var constructed = OnnxOp.SequenceConstruct(x, y);
        var split = OnnxOp.SplitToSequence(x, axis: 0);
        var graph = new InternalComputationGraph([x, y], [constructed, split]);

        var shapeInfo = new ShapeInferenceInterpreter(CpuContext).Infer(graph,
            Globals.TensorDataWithSmallVals(DType.Float32, [4, 8]),
            Globals.TensorDataWithSmallVals(DType.Float32, [2, 8]));

        Assert.Equal(48 * 4L, shapeInfo.GetTensorInfo(graph.Outputs[0])!.MemoryBytes);
        Assert.Equal(32 * 4L, shapeInfo.GetTensorInfo(graph.Outputs[1])!.MemoryBytes);
        Assert.Equal(DType.Float32, shapeInfo.GetTensorInfo(graph.Outputs[1])!.DType);
    }

    [Fact]
    public void TestShapeInferenceOrtFallbackResolvesDetTopKAndConstantCoverage()
    {
        var x = InputTensor<float32>("x", rank: 2);
        var v = InputVector<float32>("v");

        var det = OnnxOp.Det(x);
        var (topVals, topIdx) = OnnxOp.TopK(v, OnnxOp.Constant((long[])[2L]), axis: -1, largest: true, sorted: true);
        var constTensor = OnnxOp.Constant(Globals.TensorData(DType.Float32, [2L], 5f, 6f).MoveToAttribute());
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

    private const long Mb = 512 * 512 * 4L;

    private static ShapeInferenceResult Infer(InternalComputationGraph graph, params long[][] shapes)
        => new ShapeInferenceInterpreter(CpuContext).Infer(graph,
            shapes.Select(shape => Globals.TensorDataWithSmallVals(DType.Float32, shape)).ToArray());

    private static InternalComputationGraph InOrder(InternalComputationGraph graph, IEnumerable<int> order)
    {
        var copy = graph.Clone();
        copy.Nodes = [.. order.Select(i => graph.Nodes[i]), .. copy.OutputNodes];
        return copy;
    }

    private static string[] Ops(InternalComputationGraph graph, IEnumerable<int> order)
        => order.Select(i => graph.Nodes[i].OpCode).ToArray();

    private static string[] OrtOps(InternalComputationGraph graph)
        => Ops(graph, OrtExecutionOrder.Compute(graph.Nodes));

    private static InternalComputationGraph Realized(InternalComputationGraph graph, IEnumerable<int> preferred)
    {
        var copy = graph.Clone();
        copy.Nodes = OrtExecutionOrder.Realize(graph.Nodes, preferred.Select(i => graph.Nodes[i]).ToList());
        return copy;
    }

    private static (InternalComputationGraph Graph, ShapeInferenceResult ShapeInfo) SiblingBranchGraph()
    {
        var x = InputTensor<float32>("x", rank: 2);
        var negated = OnnxOp.Neg(x);
        var summed = OnnxOp.ReduceSum(OnnxOp.Concat([x, x], axis: 0));
        var graph = new InternalComputationGraph([x], [OnnxOp.Add(negated, summed)]);
        return (graph, Infer(graph, [512, 512]));
    }

    [Fact]
    public void TestOrtOrderEvaluationWalksOrtsDfsRatherThanTheProtoOrderCoverage()
    {
        var (graph, shapeInfo) = SiblingBranchGraph();
        var evaluator = new GraphEvaluator();
        var dfs = OrtExecutionOrder.Compute(graph.Nodes);
        var proto = evaluator.Evaluate(graph, shapeInfo, EvaluationOrder.ProtoOrder);
        var ort = evaluator.Evaluate(graph, shapeInfo);

        Assert.Equal((string[])[InternalOpCodes.MODEL_TENSOR_INPUT, "Neg", "Concat", "ReduceSum", "Add"], Ops(graph, Enumerable.Range(0, 5)));
        Assert.Equal((string[])[InternalOpCodes.MODEL_TENSOR_INPUT, "Concat", "ReduceSum", "Neg", "Add"], Ops(graph, dfs));
        Assert.Equal(4 * Mb, proto.PeakMemoryBytes);
        Assert.Equal(3 * Mb + 4, ort.PeakMemoryBytes);
        Assert.Equal(evaluator.Evaluate(InOrder(graph, dfs), shapeInfo, EvaluationOrder.ProtoOrder).PeakMemoryBytes, ort.PeakMemoryBytes);
        Assert.Equal(2.0 / 3, ort.OrderFidelity, 9);
        Assert.Equal(ort.OrderFidelity, proto.OrderFidelity, 9);
        Assert.Equal(dfs, ort.NodeDetails.Select(d => d.NodeIndex));
        Assert.Equal(Enumerable.Range(0, 5), proto.NodeDetails.Select(d => d.NodeIndex));

        Assert.Equal(Ops(graph, dfs), OrtOps(Realized(graph, dfs)));
        Assert.Equal(Ops(graph, Enumerable.Range(0, 5)), OrtOps(Realized(graph, Enumerable.Range(0, 5))));
        Assert.Equal(2.0 / 3, evaluator.Evaluate(Realized(graph, Enumerable.Range(0, 5)), shapeInfo).OrderFidelity, 9);
        Assert.True(Realized(graph, Enumerable.Range(0, 5)).IsLinearOrderValid());
    }

    [Fact]
    public void TestOptimizerClaimsOnlyWhatOrtOrderDeliversCoverage()
    {
        var (graph, shapeInfo) = SiblingBranchGraph();
        var evaluator = new GraphEvaluator();
        var baseline = evaluator.Evaluate(graph, shapeInfo);

        var result = new MemoryAwareGraphOptimizer(shapeInference: new ShapeInferenceInterpreter(CpuContext))
            .OptimizeWithShapeInfo(graph, shapeInfo);
        Assert.Equal(result.Evaluation.PeakMemoryBytes, evaluator.Evaluate(result.OptimizedGraph, result.ShapeInfo).PeakMemoryBytes);
        Assert.True(result.Evaluation.PeakMemoryBytes <= baseline.PeakMemoryBytes);
        Assert.True(result.OptimizedGraph.IsLinearOrderValid());

        var reordered = new MemoryAwareScheduler().Reorder(graph, shapeInfo);
        Assert.True(reordered.IsLinearOrderValid());
        Assert.Equal(
            evaluator.Evaluate(InOrder(reordered, OrtExecutionOrder.Compute(reordered.Nodes)), shapeInfo, EvaluationOrder.ProtoOrder).PeakMemoryBytes,
            evaluator.Evaluate(reordered, shapeInfo).PeakMemoryBytes);
    }

    [Fact]
    public void TestEvaluatorSharesAliasedBuffersAndFreesUnconsumedOutputsCoverage()
    {
        var x = InputTensor<float32>("x", rank: 2);
        var reshaped = OnnxOp.Reshape(x, Vector(512L * 512L), allowZero: false);
        var direct = OnnxOp.ReduceSum(x);
        var viaView = OnnxOp.ReduceSum(reshaped);
        var aliasGraph = new InternalComputationGraph([x], [OnnxOp.Add(viaView, direct)]);

        var y = InputTensor<float32>("y", rank: 2);
        var halves = OnnxOp.Split(y, Vector(256L, 256L), axis: 0, numOutputs: null, variadicOutputCount: 2);
        var grown = OnnxOp.Exp(halves[0]);
        var danglingGraph = new InternalComputationGraph([y], [OnnxOp.Concat([grown, grown, grown, grown], axis: 0)]);

        var evaluator = new GraphEvaluator();
        foreach (var order in (EvaluationOrder[])[EvaluationOrder.OrtOrder, EvaluationOrder.ProtoOrder])
        {
            Assert.Equal(Mb + 8, evaluator.Evaluate(aliasGraph, Infer(aliasGraph, [512, 512]), order).PeakMemoryBytes);
            Assert.Equal(5 * Mb / 2, evaluator.Evaluate(danglingGraph, Infer(danglingGraph, [512, 512]), order).PeakMemoryBytes);
        }
    }

    private static readonly StepState StateInPlace = new([(0, 0)], WrittenInPlace: true);
    private static readonly StepState StateHeld = new([(0, 0)], WrittenInPlace: false);

    private static (InternalComputationGraph Graph, ShapeInferenceResult ShapeInfo) StateGraph(
        Func<Tensor<float32>, Tensor<float32>, Variable[]> outputs, long[]? stateShape = null)
    {
        var p = InputTensor<float32>("p", rank: 2);
        var g = InputTensor<float32>("g", rank: 2);
        var graph = new InternalComputationGraph([p, g], [.. outputs(p, g)]);
        return (graph, Infer(graph, stateShape ?? [512, 512], [512, 512]));
    }

    private static GraphEvaluationResult EvalState(
        StepState? state, Func<Tensor<float32>, Tensor<float32>, Variable[]> outputs, long[]? stateShape = null)
    {
        var (graph, shapeInfo) = StateGraph(outputs, stateShape);
        return new GraphEvaluator(state: state).Evaluate(graph, shapeInfo);
    }

    private static Variable Looped(Tensor<float32> p, Tensor<float32> g)
    {
        Variable carried = g;
        foreach (var _ in LoopAPI.Iterate(Scalar(2L)))
            carried = OnnxOp.Add(carried, p);
        return carried;
    }

    [Fact]
    public void TestAStateOutputWrittenInPlaceIsChargedInItsInputsBufferCoverage()
    {
        Assert.Equal(2 * Mb, EvalState(StateInPlace, (p, g) => [OnnxOp.Sub(p, g)]).PeakMemoryBytes);
        Assert.Equal(3 * Mb, EvalState(StateHeld, (p, g) => [OnnxOp.Sub(p, g)]).PeakMemoryBytes);
        Assert.Equal(3 * Mb, EvalState(null, (p, g) => [OnnxOp.Sub(p, g)]).PeakMemoryBytes);
        Assert.Equal(Mb + 4, EvalState(StateInPlace, (p, g) => [OnnxOp.Sub(p, OnnxOp.ReduceSum(p))]).PeakMemoryBytes);
        Assert.Equal(2 * Mb + 4, EvalState(StateHeld, (p, g) => [OnnxOp.Sub(p, OnnxOp.ReduceSum(p))]).PeakMemoryBytes);
        Assert.Equal(2 * Mb + 4, EvalState(StateHeld, (p, g) => [OnnxOp.ReduceSum(g)]).PeakMemoryBytes);
        Assert.Equal(Mb + 4, EvalState(null, (p, g) => [OnnxOp.ReduceSum(g)]).PeakMemoryBytes);
        Assert.Equal([(0, 0)], EvalState(StateInPlace, (p, g) => [OnnxOp.Sub(p, g)]).StateWrittenInPlace);
        Assert.Empty(EvalState(StateHeld, (p, g) => [OnnxOp.Sub(p, g)]).StateWrittenInPlace);
    }

    [Fact]
    public void TestAStateOutputItsGraphCannotProveIsChargedBesideItsInputCoverage()
    {
        Assert.Equal(4 * Mb, EvalState(StateInPlace, (p, g) => [OnnxOp.Sub(p, g), OnnxOp.Neg(p)]).PeakMemoryBytes);
        Assert.Equal(EvalState(StateHeld, (p, g) => [OnnxOp.Sub(p, g), OnnxOp.Neg(p)]).PeakMemoryBytes,
            EvalState(StateInPlace, (p, g) => [OnnxOp.Sub(p, g), OnnxOp.Neg(p)]).PeakMemoryBytes);
        Assert.Empty(EvalState(StateInPlace, (p, g) => [OnnxOp.Sub(p, g), OnnxOp.Neg(p)]).StateWrittenInPlace);
        Assert.Empty(EvalState(StateInPlace, (p, g) => [OnnxOp.Sub(g, p)]).StateWrittenInPlace);
        Assert.Empty(EvalState(StateInPlace, (p, g) => [OnnxOp.Relu(p)]).StateWrittenInPlace);
        Assert.Empty(EvalState(StateInPlace, (p, g) => [OnnxOp.Sub(p, g), OnnxOp.Reshape(p, Vector(512L * 512L), allowZero: false)]).StateWrittenInPlace);
        Assert.Empty(EvalState(StateInPlace, (p, g) => [OnnxOp.Sub(p, Looped(p, g))]).StateWrittenInPlace);
        Assert.Empty(EvalState(StateInPlace, (p, g) => [OnnxOp.Sub(p, g)], stateShape: [1, 512]).StateWrittenInPlace);
        Assert.Empty(EvalState(new StepState([(1, 0), (0, 2)], WrittenInPlace: true), (p, g) => [OnnxOp.Sub(p, g)]).StateWrittenInPlace);
    }

    private static Variable Twice(Variable p) => OnnxOp.Transpose(OnnxOp.Transpose(p));

    private static Variable Update(Variable p, Variable g, Variable lr) => OnnxOp.Sub(p, OnnxOp.Mul(g, lr));

    private static Variable[] Linear(Variable p, Variable g, Variable lr) => [Update(p, g, lr), OnnxOp.MatMul(g, Twice(p))];

    private static Variable[] Tied(Variable p, Variable g, Variable lr) => [Update(p, g, lr), OnnxOp.MatMul(g, Twice(p)), OnnxOp.MatMul(OnnxOp.Transpose(p), g)];

    private static Variable[] Direct(Variable p, Variable g, Variable lr) => [Update(p, g, lr), OnnxOp.MatMul(g, p)];

    private static Variable[] EmptyReader(Variable p, Variable g, Variable lr) => [Update(p, g, lr), OnnxOp.Slice(Twice(p), Vector(0L), Vector(0L))];

    private static Variable[] ReadAfterUpdate(Variable p, Variable g, Variable lr) => ReadAfter(Update(p, g, lr), p);

    private static Variable[] ReadAfter(Variable updated, Variable p) => [updated, OnnxOp.MatMul(updated, Twice(p))];

    private static Variable[] Unanchored(Variable p, Variable g, Variable lr) => [OnnxOp.Sub(p, g), OnnxOp.MatMul(g, Twice(p))];

    private static Variable[] BoolReader(Variable p, Variable g, Variable lr) => [Update(p, g, lr), OnnxOp.Greater(Twice(p), g)];

    private static Variable[] DoubleReader(Variable p, Variable g, Variable lr) => [Update(p, g, lr), OnnxOp.Cast(Twice(p), null, DType.Float64)];

    private static Variable[] ScalarReader(Variable p, Variable g, Variable lr) => [Update(p, g, lr), OnnxOp.ReduceSum(Twice(p), keepdims: false)];

    private static Variable[] DoubleAnchor(Variable p, Variable g, Variable lr) =>
        [OnnxOp.Sub(p, OnnxOp.Cast(OnnxOp.Mul(OnnxOp.Cast(g, null, DType.Float64), OnnxOp.Cast(lr, null, DType.Float64)), null, DType.Float32)), OnnxOp.MatMul(g, Twice(p))];

    private static (InternalComputationGraph Graph, ShapeInferenceResult ShapeInfo) StepGraph(
        Func<Variable, Variable, Variable, Variable[]> outputs, bool ordered, long side = 512, long rows = 512, bool withReaders = false)
    {
        var (p, g, lr) = (InputTensor<float32>("p", rank: 2), InputTensor<float32>("g", rank: 2), InputScalar<float32>("lr"));
        var graph = new InternalComputationGraph([p, g, lr], [.. outputs(p, g, lr)]);
        var shapeInfo = Infer(graph, [side, side], [Math.Min(rows, side), side], []);
        return ordered ? StateReadOrdering.Apply(graph, shapeInfo, [(0, 0)], withReaders) : (graph, shapeInfo);
    }

    private static readonly StepState TwoStatesInPlace = new([(0, 0), (1, 1)], WrittenInPlace: true);

    private static (InternalComputationGraph Graph, ShapeInferenceResult ShapeInfo) TwoStatesSharingAnUpdate(bool ordered, long side = 512)
    {
        var (p, q, g, lr) = (InputTensor<float32>("p", rank: 2), InputTensor<float32>("q", rank: 2), InputTensor<float32>("g", rank: 2), InputScalar<float32>("lr"));
        var step = OnnxOp.Mul(g, lr);
        var graph = new InternalComputationGraph([p, q, g, lr], [OnnxOp.Sub(p, step), OnnxOp.Sub(q, step), OnnxOp.MatMul(g, Twice(p)), OnnxOp.MatMul(g, Twice(q))]);
        var shapeInfo = Infer(graph, [side, side], [side, side], [side, side], []);
        return ordered ? StateReadOrdering.Apply(graph, shapeInfo, TwoStatesInPlace.Pairs) : (graph, shapeInfo);
    }

    private static Variable[] ReadEarly(Variable p, Variable g, Variable lr)
        => [Update(p, g, lr), Then(OnnxOp.Mul(g, OnnxOp.ReduceSum(OnnxOp.MatMul(g, Twice(p)))), g)];

    private static Variable Then(Variable t, Variable g) => OnnxOp.ReduceSum(OnnxOp.Add(OnnxOp.MatMul(t, g), t));

    private static GraphEvaluationResult Evaluated(StepState state, Func<Variable, Variable, Variable, Variable[]> outputs, bool ordered, bool withReaders = false)
    {
        var (graph, shapeInfo) = StepGraph(outputs, ordered, withReaders: withReaders);
        return new GraphEvaluator(state: state).Evaluate(graph, shapeInfo);
    }

    [Fact]
    public void TestAnUpdateRunWithItsReadersHoldsNoReadersOutputUntilTheUpdateCoverage()
    {
        var withReaders = Evaluated(StateInPlace, ReadEarly, ordered: true, withReaders: true);
        Assert.True(withReaders.PeakMemoryBytes < Evaluated(StateInPlace, ReadEarly, ordered: true).PeakMemoryBytes);
        Assert.Equal([(0, 0)], withReaders.StateWrittenInPlace);
        var (graph, shapeInfo) = StepGraph(ReadEarly, ordered: false);
        Assert.True(new MemoryAwareGraphOptimizer(evaluator: new GraphEvaluator(state: StateInPlace), shapeInference: new ShapeInferenceInterpreter(CpuContext))
            .OptimizeWithShapeInfo(graph, shapeInfo).Evaluation.PeakMemoryBytes <= withReaders.PeakMemoryBytes);
        Assert.Equal((0L, 1L, true), Ordered(ReadEarly, withReaders: true, 0.5f));
    }

    private static IReadOnlyList<(int Output, int Input)> Modelled(Func<Variable, Variable, Variable, Variable[]> outputs, bool ordered)
    {
        var (graph, shapeInfo) = StepGraph(outputs, ordered);
        return new GraphEvaluator(state: StateInPlace).Evaluate(graph, shapeInfo).StateWrittenInPlace;
    }

    private static GraphEvaluationResult Optimized(Func<Variable, Variable, Variable, Variable[]> outputs, StepState state)
    {
        var (graph, shapeInfo) = StepGraph(outputs, ordered: false, rows: 1);
        return new MemoryAwareGraphOptimizer(evaluator: new GraphEvaluator(state: state), shapeInference: new ShapeInferenceInterpreter(CpuContext))
            .OptimizeWithShapeInfo(graph, shapeInfo).Evaluation;
    }

    private static (float[] Values, long Aliased) Execute(InternalComputationGraph graph, IReadOnlyList<(int, int)> pairs, float[] lrs)
    {
        using var context = new ComputeContext();
        var squares = graph.Inputs.Count - 1;
        var compiled = context.Compile(graph, [.. Enumerable.Repeat<long[]>([16L, 16L], squares), []], trainingStep: true, aliasCandidates: pairs);
        float[] Square(int seed) => [.. Enumerable.Range(0, 256).Select(i => (i * seed % 17) - 8f)];
        var values = new List<float>();
        foreach (var lr in lrs)
        {
            TensorData[] inputs = [.. Enumerable.Range(0, squares).Select(i => TensorData([16L, 16L], Square(2 * i + 3))), TensorData([], lr)];
            foreach (var result in compiled.Execute(inputs))
                values.AddRange(result.ToTensorData() is var data && data.DType == DType.Float32 ? data.As<float32>().CopyMemory<float>()
                    : data.DType == DType.Float64 ? [.. data.As<float64>().CopyMemory<double>().Select(v => (float)v)]
                    : [.. data.As<bit>().CopyMemory<bool>().Select(v => v ? 1f : 0f)]);
        }
        return ([.. values], context.AliasedOutputs);
    }

    private static (long Plain, long Ordered, bool Same) Ordered(Func<Variable, Variable, Variable, Variable[]> outputs, params float[] lrs)
        => Ordered(outputs, withReaders: false, lrs);

    private static (long Plain, long Ordered, bool Same) Ordered(Func<Variable, Variable, Variable, Variable[]> outputs, bool withReaders, params float[] lrs)
    {
        var plain = Execute(StepGraph(outputs, ordered: false, side: 16).Graph, [(0, 0)], lrs);
        var ordered = Execute(StepGraph(outputs, ordered: true, side: 16, withReaders: withReaders).Graph, [(0, 0)], lrs);
        return (plain.Aliased, ordered.Aliased, plain.Values.SequenceEqual(ordered.Values));
    }

    private static bool OrderingLeaves(Func<Variable, Variable, Variable, Variable[]> outputs)
    {
        var (graph, shapeInfo) = StepGraph(outputs, ordered: false);
        return ReferenceEquals(graph, StateReadOrdering.Apply(graph, shapeInfo, [(0, 0)]).Graph);
    }

    private static CompiledGraph CompiledOrdered(Func<Variable, Variable, Variable, Variable[]> outputs)
    {
        using var context = new ComputeContext();
        var (graph, _) = StepGraph(outputs, ordered: true, side: 16);
        return context.Compile(graph, [[16L, 16L], [16L, 16L], []], trainingStep: true, aliasCandidates: [(0, 0)]);
    }

    [Fact]
    public void TestAnUpdateIsNeverOrderedThroughAnOperandThatDeterminesAShapeCoverage()
    {
        Assert.True(OrderingLeaves((p, g, lr) => [OnnxOp.Sub(p, OnnxOp.Mul(g, OnnxOp.Range(Scalar(0f), Scalar(512f), Scalar(1f)))), OnnxOp.MatMul(g, Twice(p))]));
        Assert.True(OrderingLeaves((p, g, lr) => [OnnxOp.Sub(p, OnnxOp.Resize(g, null, Vector(1f, 1f), null, null, null, null, null, null, null, null, null, null)), OnnxOp.MatMul(g, Twice(p))]));
    }

    [Fact]
    public void TestAnUpdateWhoseStateIsReadIntoASequenceCompilesCoverage()
        => Assert.NotNull(CompiledOrdered((p, g, lr) => [Update(p, g, lr), OnnxOp.SplitToSequence(Twice(p))]));

    [Fact]
    public void TestTheModelCountsAReaderOfWhatTheStateAloneComputesAsAReaderOfTheStateCoverage()
    {
        Assert.Empty(Modelled(Linear, ordered: false));
        Assert.Empty(Modelled(Tied, ordered: false));
        Assert.Empty(Modelled(EmptyReader, ordered: false));
        Assert.Equal([(0, 0)], Modelled((p, g, lr) => [OnnxOp.Sub(p, OnnxOp.MatMul(g, Twice(p)))], ordered: false));
        Assert.Equal([(0, 0)], Modelled((p, g, lr) => [OnnxOp.Sub(p, g), OnnxOp.MatMul(g, OnnxOp.Transpose(g))], ordered: false));
        Assert.Empty(Modelled((p, g, lr) => [OnnxOp.Sub(p, g), OnnxOp.Reshape(Twice(p), OnnxOp.Shape(g), allowZero: false)], ordered: false));
    }

    [Fact]
    public void TestOrderingAnUpdateAfterEveryReaderOfItsStateLetsTheModelWriteItInPlaceCoverage()
    {
        Assert.Equal([(0, 0)], Modelled(Linear, ordered: true));
        Assert.Equal([(0, 0)], Modelled(Tied, ordered: true));
        Assert.Equal([(0, 0)], Modelled(Direct, ordered: true));
        Assert.Equal([(0, 0)], Modelled(EmptyReader, ordered: true));
        Assert.Equal([(0, 0)], Modelled(BoolReader, ordered: true));
        Assert.Equal([(0, 0)], Modelled(DoubleReader, ordered: true));
        Assert.Equal([(0, 0)], Modelled(ScalarReader, ordered: true));
        Assert.Equal([(0, 0)], Modelled(DoubleAnchor, ordered: true));
        var (graph, shapeInfo) = TwoStatesSharingAnUpdate(ordered: true);
        Assert.Equal([(0, 0), (1, 1)], new GraphEvaluator(state: TwoStatesInPlace).Evaluate(graph, shapeInfo).StateWrittenInPlace);
    }

    [Fact]
    public void TestAnUpdateThatCannotBeOrderedIsLeftAsItCameCoverage()
    {
        Assert.Empty(Modelled(ReadAfterUpdate, ordered: true));
        Assert.Empty(Modelled(Unanchored, ordered: true));
        Assert.True(OrderingLeaves(ReadAfterUpdate));
        Assert.True(OrderingLeaves(Unanchored));
    }

    [Fact]
    public void TestTheMemoryPassScoresEveryCandidateWithTheStateItWritesInPlaceCoverage()
    {
        var (graph, shapeInfo) = StateGraph((p, g) => [OnnxOp.Sub(p, g)]);
        var optimized = new MemoryAwareGraphOptimizer(evaluator: new GraphEvaluator(state: StateInPlace), shapeInference: new ShapeInferenceInterpreter(CpuContext))
            .OptimizeWithShapeInfo(graph, shapeInfo);
        Assert.Equal([(0, 0)], optimized.Evaluation.StateWrittenInPlace);
        Assert.Equal(2 * Mb, optimized.Evaluation.PeakMemoryBytes);
        Assert.Equal([(0, 0)], Optimized(Direct, StateInPlace).StateWrittenInPlace);
        Assert.True(Optimized(Direct, StateInPlace).PeakMemoryBytes < Optimized(Direct, StateHeld).PeakMemoryBytes);
    }

    [Fact]
    public void TestAnUpdateOrderedAfterEveryReaderOfItsStateIsWrittenInPlaceByTheRuntimeAndComputesTheSameCoverage()
    {
        Assert.Equal((0L, 1L, true), Ordered(Linear, 0.5f));
        Assert.Equal((0L, 3L, true), Ordered(Tied, 0.5f, float.NaN, float.NegativeInfinity));
        Assert.Equal((0L, 1L, true), Ordered(Direct, 0.5f));
        Assert.Equal((0L, 2L, true), Ordered(EmptyReader, 0.5f, float.PositiveInfinity));
        Assert.Equal((0L, 0L, true), Ordered(ReadAfterUpdate, 0.5f));
    }

    [Fact]
    public void TestAnUpdateOrderedAfterReadersOfEveryKindIsWrittenInPlaceByTheRuntimeAndComputesTheSameCoverage()
    {
        Assert.Equal((0L, 1L, true), Ordered(BoolReader, 0.5f));
        Assert.Equal((0L, 1L, true), Ordered(DoubleReader, 0.5f));
        Assert.Equal((0L, 1L, true), Ordered(ScalarReader, 0.5f));
        Assert.Equal((0L, 1L, true), Ordered(DoubleAnchor, 0.5f));
        var plain = Execute(TwoStatesSharingAnUpdate(ordered: false, side: 16).Graph, TwoStatesInPlace.Pairs, [0.5f]);
        var ordered = Execute(TwoStatesSharingAnUpdate(ordered: true, side: 16).Graph, TwoStatesInPlace.Pairs, [0.5f]);
        Assert.Equal((0L, 2L, true), (plain.Aliased, ordered.Aliased, plain.Values.SequenceEqual(ordered.Values)));
    }

    private static bool Recomputable(Variable input, Variable output)
    {
        var g = new InternalComputationGraph([input], [output]);
        return Rematerializer.IsRecomputable(g.Nodes.Single(n => n.Outputs.Any(o => o is not null && g.Outputs.Contains(o.Value))));
    }

    [Fact]
    public void TestAnUnknownDimIsNeverPricedCoverage()
    {
        Assert.True(new Shape(-1L, 1024L).Count < 0);
        Assert.Equal(128L, new Shape(-1L, -1L, 128L).Count);
        Assert.All((Shape[])[new Shape(-1L, 1024L), new Shape(-1L, -1L, 128L)],
            s => Assert.Null(ShapeInferenceInterpreter.ShapeInfoForTest(s, DType.Float32)));
        Assert.NotNull(ShapeInferenceInterpreter.ShapeInfoForTest(new Shape(2L, 4L), DType.Float32));
    }

    [Fact]
    public void TestLivenessPeakMatchesTheEvaluatorsPeakCoverage()
    {
        var x = InputTensor<float32>("x", rank: 2);
        var wide = OnnxOp.Concat([OnnxOp.Exp(x), x], axis: 0);
        var graph = new InternalComputationGraph([x], [OnnxOp.Add(OnnxOp.ReduceSum(wide), OnnxOp.ReduceSum(x))]);
        var shapeInfo = Infer(graph, [512, 512]);
        var eval = new GraphEvaluator().Evaluate(graph, shapeInfo);

        Assert.Equal(eval.PeakMemoryBytes, Rematerializer.LivenessPeakFor(graph, eval, shapeInfo));
    }

    [Fact]
    public void TestDrawsAreNeverRecomputedAndOtherOpsAreCoverage()
    {
        string[] notRecomputable =
        [
            OpCodes.RANDOM_UNIFORM, OpCodes.RANDOM_NORMAL, OpCodes.RANDOM_UNIFORM_LIKE, OpCodes.RANDOM_NORMAL_LIKE,
            OpCodes.BERNOULLI, OpCodes.MULTINOMIAL, OpCodes.DROPOUT,
            InternalOpCodes.SHRK_RANDOM_UNIFORM, InternalOpCodes.SHRK_RANDOM_NORMAL, InternalOpCodes.SHRK_RANDOM_BITS,
        ];
        string[] recomputable =
        [
            OpCodes.RELU, OpCodes.MATMUL, OpCodes.EXP, OpCodes.CONV,
            OpCodes.TOPK, OpCodes.ARG_MAX, OpCodes.UNIQUE,
            InternalOpCodes.SHRK_RNG_UNIFORM, InternalOpCodes.SHRK_RNG_NORMAL, InternalOpCodes.SHRK_RNG_BITS,
        ];
        Assert.All(notRecomputable, op => Assert.False(Rematerializer.IsDeterministicOpCode(op)));
        Assert.All(recomputable, op => Assert.True(Rematerializer.IsDeterministicOpCode(op)));

        var x = InputTensor<float32>("x", rank: 2);
        Assert.True(Recomputable(x, OnnxOp.Relu(x)));
        Assert.False(Recomputable(x, OnnxOp.RandomUniformLike(x, seed: 3f)));
        Assert.False(Recomputable(x, OnnxOp.RandomNormalLike(x, seed: 3f)));
        Assert.False(Recomputable(x, OnnxOp.Bernoulli(x, dtype: null, seed: 3f)));
        Assert.False(Recomputable(x, OnnxOp.Dropout(x, null, null, seed: 3L).output));
    }

    [Fact]
    public void TestLoopGraphIsRescheduledAndOrderedCoverage()
    {
        var x = InputTensor<float32>("x", rank: 2);
        Variable carried = x;
        foreach (var _ in LoopAPI.Iterate(Scalar(3L)))
            carried = OnnxOp.Add(OnnxOp.Exp(carried), x);
        var negated = OnnxOp.Neg(x);
        var rectified = OnnxOp.Relu(carried);
        var graph = new InternalComputationGraph([x], [OnnxOp.Add(rectified, negated)]);
        var shapeInfo = Infer(graph, [512, 512]);
        var scheduler = new MemoryAwareScheduler();

        Assert.Contains(graph.Nodes, n => n.IsOpenNode());
        Assert.NotNull(scheduler.MemoryAwareTopologicalSort(graph.Nodes, shapeInfo));
        var reordered = scheduler.Reorder(graph, shapeInfo);
        Assert.NotSame(graph, reordered);
        Assert.True(reordered.IsLinearOrderValid());

        var walk = OrtExecutionOrder.Compute(graph.Nodes);
        Assert.Equal(graph.BodyEnd, walk.Distinct().Count());
        Assert.True(InOrder(graph, walk).IsLinearOrderValid());

        var untouched = new MemoryAwareGraphOptimizer().OptimizeWithShapeInfo(graph, shapeInfo);
        Assert.Equal("Baseline", untouched.StrategyName);
        Assert.Same(graph, untouched.OptimizedGraph);
        Assert.True(new GraphEvaluator().Evaluate(graph, shapeInfo).PeakMemoryBytes >= Mb);
    }

    // ----- [Module(Checkpoint = true)] and the rematerializer's invariants -----

    private static TensorData Pattern(long[] dims, float scale)
    {
        var n = dims.Aggregate(1L, (a, b) => a * b);
        var v = new float[n];
        for (var i = 0; i < n; i++) v[i] = (((i * 37) % 101) * 0.01f - 0.5f) * scale;
        return TensorData(dims, v);
    }

    private static TensorData Synthesize(Shape shape, DType dtype)
        => dtype == DType.Float32 ? Pattern(shape.Dims, 1f) : TensorData(shape.Dims, new long[shape.Count]);

    private static (TrainingRig Rig, TensorDataStruct Input, TensorDataStruct Target) MlpStackRig(ComputationGraph model, long[] inShape)
    {
        var x = Pattern(inShape, 1f);
        var rig = TrainingRig.FromScratch(model, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            [new TensorDataModelParam(model.InputNames[0]!, ModelParamType.InputParam, x)], 0.01f);
        var input = new TensorDataStruct(rig.InputDef, new Dictionary<string, IData> { [rig.InputDef.Fields[0].Name] = x });
        var target = new TensorDataStruct(rig.TargetDef, new Dictionary<string, IData> { [rig.TargetDef.Fields[0].Name] = Pattern([inShape[0], 16L], 0.5f) });
        return (rig, input, target);
    }

    private static float[] Losses(TrainingRig rig, TensorDataStruct input, TensorDataStruct target, int steps)
    {
        var ckpt = rig.CreateInitialCheckpoint();
        var losses = new float[steps];
        for (var i = 0; i < steps; i++)
        {
            ckpt = rig.TrainStep(ckpt, input.Shared(), target.Shared());
            losses[i] = ckpt.Loss!.Value;
        }
        return losses;
    }

    private static int NodeCount(TrainingRig rig) => rig.TrainingStepPureGraph.ToInternal().Nodes.Count;

    private static bool Stamped(TrainingRig rig)
        => rig.TrainingStepPureGraph.ToInternal().Nodes.Any(n => CheckpointSegment.IdOf(n) is not null);

    /// <summary>Every segment stamp in the rig's architecture, signed as the attribute encodes it.</summary>
    private static long[] Stamps(TrainingRig rig)
        => [.. rig.ConcreteArchConstituent.ToInternal().Nodes
            .Select(n => CheckpointSegment.IdOf(n) is { } id ? (CheckpointSegment.ProducesSegmentOutput(n) ? -id : id) : 0L)
            .Where(v => v != 0).OrderBy(v => v)];

    private static bool CarriesCheckpointStamp(ModelProto proto)
        => proto.Graph.Nodes.Concat(proto.Functions.SelectMany(f => f.Nodes))
            .Any(n => n.Attributes.Any(a => a.Name == OnnxOpAttributeNames.ShrkAttrCheckpoint));

    private static TensorData[] Run(TrainingRig rig, ComputationGraph graph)
        => new ComputeContext()
            .Execute(graph, rig.OptimizationInputShapes.Select(s => (IData)Synthesize(s.Shape, s.DType)).ToArray())
            .Select(o => o.ToTensorData()).ToArray();

    private static bool Close(TensorData expected, TensorData actual)
    {
        if (expected.DType != DType.Float32)
            return expected.As<int64>().AccessMemory().SequenceEqual(actual.As<int64>().AccessMemory());
        var e = expected.As<float32>().AccessMemory();
        var a = actual.As<float32>().AccessMemory();
        if (e.Length != a.Length) return false;
        for (var i = 0; i < e.Length; i++)
            if (Math.Abs(e[i] - a[i]) > 1e-5f * Math.Max(1f, Math.Abs(e[i]))) return false;
        return true;
    }

    [Fact]
    public void TestCheckpointedModuleTrainsLikeItsTwinWithLowerPeakAndARealRecomputeCoverage()
    {
        var (plain, input, target) = MlpStackRig(Modules.PlainNarrowMlpStack.ComputationGraph, [256L, 32L]);
        var (checkpointed, _, _) = MlpStackRig(Modules.CheckpointedNarrowMlpStack.ComputationGraph, [256L, 32L]);

        var expected = Losses(plain, input, target, 3);
        var actual = Losses(checkpointed, input, target, 3);
        for (var i = 0; i < 3; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-6f * Math.Max(1f, Math.Abs(expected[i])));

        Assert.True(checkpointed.OptimizationResult.Evaluation.PeakMemoryBytes < plain.OptimizationResult.Evaluation.PeakMemoryBytes);
        Assert.True(NodeCount(checkpointed) > NodeCount(plain));
        Assert.True(Stamped(checkpointed));
        Assert.False(Stamped(plain));
        Assert.False(CarriesCheckpointStamp(FastOnnxModelBuilder.BuildInternalOnnxModel(checkpointed.TrainingStepPureGraph.ToInternal(), prepForOnnx: true)));
        Assert.False(CarriesCheckpointStamp(FastOnnxModelBuilder.BuildOnnxModel(checkpointed.CreateInitialCheckpoint().ToInferenceModel())));

        var path = Path.Combine(Path.GetTempPath(), $"shrk_ckpt_stamp_{Guid.NewGuid():N}.skpt");
        try
        {
            Persistence.SaveTrainingCheckpointToSkpt(checkpointed.CreateInitialCheckpoint(), path);
            var (reloaded, _) = TrainingRig.Load(path);
            Assert.Equal(Stamps(checkpointed), Stamps(reloaded));
            Assert.NotEmpty(Stamps(reloaded));
            Assert.Equal(NodeCount(checkpointed), NodeCount(reloaded));
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    [Fact]
    public void TestCheckpointHintIsHonouredWhereTheMemoryPassWouldSkipCoverage()
    {
        var (plain, _, _) = MlpStackRig(Modules.PlainTinyMlpStack.ComputationGraph, [2L, 8L]);
        var (checkpointed, _, _) = MlpStackRig(Modules.CheckpointedTinyMlpStack.ComputationGraph, [2L, 8L]);

        Assert.True(plain.PreOptimizationEval.PeakMemoryBytes < MemoryAwareGraphOptimizer.MinimumPeakBytesToOptimize);
        Assert.Contains(plain.OptimizationResult.StrategyName, ["Baseline", MemoryAwareGraphOptimizer.OrderedStateReads]);
        Assert.True(NodeCount(checkpointed) > NodeCount(plain));
        Assert.True(checkpointed.OptimizationResult.Evaluation.PeakMemoryBytes <= plain.OptimizationResult.Evaluation.PeakMemoryBytes);
    }

    [Fact]
    public void TestRematerializedTrainingStepsMatchTheirOriginalsThroughOrtCoverage()
    {
        foreach (var model in (ComputationGraph[])[Modules.PlainNarrowMlpStack.ComputationGraph, Modules.CheckpointedNarrowMlpStack.ComputationGraph])
        {
            var (rig, _, _) = MlpStackRig(model, [1024L, 32L]);
            var expected = Run(rig, rig.PreOptimizationGraph);
            var actual = Run(rig, rig.TrainingStepPureGraph);
            Assert.Equal(expected.Length, actual.Length);
            for (var o = 0; o < expected.Length; o++)
                Assert.True(Close(expected[o], actual[o]));
        }
    }

    [Fact]
    public void TestRematerializerClonesEachChainOnceWithinBudgetAndNeverRaisesThePeakCoverage()
    {
        var rig = TrainingRig.FromScratch(SdpaMeanPoolModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            [new TensorDataModelParam("input", ModelParamType.InputParam, Pattern([2L, 4L, 256L, 32L], 1f))], 0.01f);
        var graph = rig.PreOptimizationGraph.ToInternal();
        var shapeInfo = new ShapeInferenceInterpreter(CpuContext).Infer(graph,
            rig.OptimizationInputShapes.Select(s => Synthesize(s.Shape, s.DType)).ToArray());
        var evaluator = new GraphEvaluator();
        var before = evaluator.Evaluate(graph, shapeInfo);
        var remat = new Rematerializer(new ComputeMemoryObjective(1.0, 2.0, before), evaluator: evaluator);
        var (after, afterInfo) = remat.Apply(graph, shapeInfo);

        Assert.NotEmpty(remat.CommitLog);
        Assert.True(remat.EvaluationsUsed <= Rematerializer.MaxEvaluationsPerCall);
        Assert.Equal(graph.Nodes.Count + remat.CommitLog.Sum(c => c.ChainLength), after.Nodes.Count);
        var originals = graph.Nodes.Select(n => n.Key).ToHashSet();
        var read = after.Nodes.SelectMany(n => n.Inputs).OfType<FastTensorKey>().ToHashSet();
        Assert.All(after.Nodes.Where(n => !originals.Contains(n.Key)), c => Assert.True(c.Outputs.OfType<FastTensorKey>().All(read.Contains)));
        Assert.All(remat.CommitLog, c => Assert.True(c.RewiredConsumers >= 1 && c.PeakAfter <= c.PeakBefore));
        Assert.True(evaluator.Evaluate(after, afterInfo).PeakMemoryBytes <= before.PeakMemoryBytes);
        foreach (var node in after.Nodes)
            foreach (var output in node.Outputs)
                if (output is not null)
                    Assert.NotNull(afterInfo.GetTensorInfo(output.Value));
    }

    private static bool FirstCommittedBatchIsMinimal(InternalComputationGraph graph, ShapeInferenceResult shapeInfo, GraphEvaluator evaluator)
    {
        var baseline = evaluator.Evaluate(graph, shapeInfo);
        var objective = new ComputeMemoryObjective(1.0, 2.0, baseline);
        var remat = new Rematerializer(objective, evaluator: evaluator);
        remat.Apply(graph, shapeInfo);
        var log = remat.CommitLog.TakeWhile(c => c.PeakBefore == baseline.PeakMemoryBytes).ToList();
        var candidates = remat.FindCandidates(Rematerializer.Liveness.Build(graph, baseline, shapeInfo), shapeInfo, []);
        List<Rematerializer.RematCandidate> batch = [.. log.Select(c => candidates.First(k => k.Rewires[0].Target.Equals(c.Target)
            && k.Variant == c.Variant && k.Chain.Count == c.ChainLength && k.Rewires.Sum(r => r.Consumers.Count) == c.RewiredConsumers))];

        (double Score, long Peak) Measure(IEnumerable<Rematerializer.RematCandidate> members)
        {
            var (g, mapping) = Rematerializer.ApplyCandidates(graph, [.. members], log[0].Placement);
            var eval = evaluator.Evaluate(g, Rematerializer.AugmentShapeInfo(shapeInfo, mapping));
            return (objective.Score(eval), eval.PeakMemoryBytes);
        }

        var committed = Measure(batch);
        return batch.All(dropped => Measure(batch.Where(c => c != dropped)) is var r && !(r.Score < committed.Score && r.Peak <= committed.Peak));
    }

    [Fact]
    public void TestNoMemberOfACommittedRecomputeBatchCanBeDroppedForABetterScoreCoverage()
    {
        var rig = TrainingRig.FromScratch(SdpaMeanPoolModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            [new TensorDataModelParam("input", ModelParamType.InputParam, Pattern([2L, 4L, 256L, 64L], 1f))], 0.01f);
        var graph = rig.PreOptimizationGraph.ToInternal();
        var shapeInfo = new ShapeInferenceInterpreter(CpuContext).Infer(graph,
            rig.OptimizationInputShapes.Select(s => Synthesize(s.Shape, s.DType)).ToArray());

        Assert.True(FirstCommittedBatchIsMinimal(graph, shapeInfo, new GraphEvaluator(state: new StepState([(0, 0), (1, 1), (2, 2)], WrittenInPlace: true))));
    }

    [Fact]
    public void TestShapeReadersDoNotHoldTheirInputCoverage()
    {
        static (InternalComputationGraph, ShapeInferenceResult) Build(bool withShapeReader)
        {
            var x = InputTensor<float32>("x", rank: 2);
            var a = OnnxOp.Exp(x);
            var c = OnnxOp.Concat([a, a], axis: 0);
            var d = OnnxOp.Concat([c, c], axis: 0);
            Variable outv = OnnxOp.ReduceSum(d);
            if (withShapeReader)
                outv = OnnxOp.Add(outv, OnnxOp.Cast(OnnxOp.ReduceProd(OnnxOp.Shape(a)), saturate: null, to: DType.Float32));
            var g = new InternalComputationGraph([x], [outv]);
            return (g, Infer(g, [512, 512]));
        }
        var (plain, plainInfo) = Build(false);
        var (reading, readingInfo) = Build(true);
        var evaluator = new GraphEvaluator();
        foreach (var order in (EvaluationOrder[])[EvaluationOrder.OrtOrder, EvaluationOrder.ProtoOrder])
        {
            var without = evaluator.Evaluate(plain, plainInfo, order).PeakMemoryBytes;
            var with = evaluator.Evaluate(reading, readingInfo, order).PeakMemoryBytes;
            Assert.Equal(6 * Mb, without);
            Assert.InRange(with, without, without + 64);
        }
    }

    [Fact]
    public void TestDeadBufferStaysOccupiedUntilASameShapeSuccessorTakesItCoverage()
    {
        var x = InputTensor<float32>("x", rank: 2);
        var a = OnnxOp.Exp(x);
        var s1 = OnnxOp.ReduceSum(a);
        var c = OnnxOp.Concat([x, x], axis: 0);
        var s2 = OnnxOp.ReduceSum(c);
        var b = OnnxOp.Neg(x);
        var graph = new InternalComputationGraph([x], [OnnxOp.Add(OnnxOp.Add(b, s1), s2)]);
        var shapeInfo = Infer(graph, [512, 512]);

        Assert.Equal(4 * Mb + 8, new GraphEvaluator().Evaluate(graph, shapeInfo, EvaluationOrder.ProtoOrder).PeakMemoryBytes);
        Assert.Equal(3 * Mb + 8, new GraphEvaluator(modelOrtBufferReuse: false).Evaluate(graph, shapeInfo, EvaluationOrder.ProtoOrder).PeakMemoryBytes);
        Assert.True(new GraphEvaluator().Evaluate(graph, shapeInfo).PeakMemoryBytes >= new GraphEvaluator(modelOrtBufferReuse: false).Evaluate(graph, shapeInfo).PeakMemoryBytes);

        var o = InputTensor<float32>("o", rank: 2);
        var wide = OnnxOp.Concat([o, o], axis: 0);
        var outputInPlace = new InternalComputationGraph([o], [OnnxOp.Relu(wide)]);
        var outputOverwritten = new InternalComputationGraph([o], [wide, OnnxOp.ReduceSum(OnnxOp.Relu(wide))]);
        var neither = new InternalComputationGraph([o], [OnnxOp.ReduceSum(OnnxOp.Relu(wide))]);
        Assert.Equal(4 * Mb, new GraphEvaluator().Evaluate(outputInPlace, Infer(outputInPlace, [512, 512])).PeakMemoryBytes);
        Assert.Equal(4 * Mb + 4, new GraphEvaluator().Evaluate(outputOverwritten, Infer(outputOverwritten, [512, 512])).PeakMemoryBytes);
        Assert.Equal(3 * Mb, new GraphEvaluator().Evaluate(neither, Infer(neither, [512, 512])).PeakMemoryBytes);

        var y = InputTensor<float32>("y", rank: 2);
        var a2 = OnnxOp.Exp(y);
        var t1 = OnnxOp.ReduceSum(a2);
        var c2 = OnnxOp.Concat([y, y], axis: 0);
        var t2 = OnnxOp.ReduceSum(c2);
        var b2 = OnnxOp.Neg(OnnxOp.Reshape(y, OnnxOp.Constant((long[])[256L, 1024L]), allowZero: false));
        var sameBytesOtherShape = new InternalComputationGraph([y], [OnnxOp.Add(OnnxOp.Add(OnnxOp.ReduceSum(b2), t1), t2)]);
        var otherInfo = Infer(sameBytesOtherShape, [512, 512]);
        Assert.Equal(3 * Mb + 8, new GraphEvaluator().Evaluate(sameBytesOtherShape, otherInfo, EvaluationOrder.ProtoOrder).PeakMemoryBytes);
    }

    [Fact]
    public void TestMemoryPassBenchmarkMeasuresTheRigsOwnModelCoverage()
    {
        var rig = TrainingRig.FromScratch(MemoryPassMlp.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            [new TensorDataModelParam("x", ModelParamType.InputParam, Pattern([64L, 256L], 1f))], 0.01f);
        var inputShapes = rig.OptimizationInputShapes;
        var model = ProtoBuf.Serializer.Deserialize<Shorokoo.Core.Factory.IR.ModelProto>(
            new MemoryStream(Benchmarks.MemoryPassBenchmarkTests.RigModelBytes(rig.TrainingStepPureGraph, inputShapes)));
        Assert.Equal(inputShapes.Length, model.Graph.Inputs.Count);
        for (var i = 0; i < model.Graph.Inputs.Count; i++)
            Assert.Equal(inputShapes[i].Shape.Dims.Select(d => (long)d), model.Graph.Inputs[i].Type.TensorType.Shape.Dims.Select(d => d.DimValue));
    }
}
