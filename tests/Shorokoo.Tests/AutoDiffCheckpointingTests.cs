using Shorokoo.Core.Factory;
using Shorokoo.Core.Factory.IR;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Graph;
using Shorokoo.Modules.Losses;
using Shorokoo.Modules.Optimizers;
using System.Collections.Immutable;
using Shorokoo.Runtime;
using Shorokoo.Core.AutoDiffCheckpointing;
using Shorokoo.Tests.Benchmarks;
using Shorokoo.Core.Graph;
using Shorokoo.Core.AutoDiffCheckpointing.OpsPerf;
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

    private const long Mb = 512 * 512 * 4L;

    private static ShapeInferenceResult Infer(InternalComputationGraph graph, params long[][] shapes)
        => new ShapeInferenceInterpreter(CpuContext).Infer(graph,
            shapes.Select(shape => Globals.TensorDataWithSmallVals(DType.Float32, shape)).ToArray());

    private static InternalComputationGraph InOrder(InternalComputationGraph graph, IEnumerable<int> order)
    {
        var copy = graph.Clone();
        copy.Nodes = order.Select(i => graph.Nodes[i]).ToList();
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
        foreach (var order in new[] { EvaluationOrder.OrtOrder, EvaluationOrder.ProtoOrder })
        {
            Assert.Equal(Mb + 8, evaluator.Evaluate(aliasGraph, Infer(aliasGraph, [512, 512]), order).PeakMemoryBytes);
            Assert.Equal(5 * Mb / 2, evaluator.Evaluate(danglingGraph, Infer(danglingGraph, [512, 512]), order).PeakMemoryBytes);
        }
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
        Assert.Equal(graph.Nodes.Count, walk.Distinct().Count());
        Assert.True(InOrder(graph, walk).IsLinearOrderValid());
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
            [new TensorDataModelParam("input", ModelParamType.InputParam, x)], 0.01f);
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
            ckpt = rig.TrainStep(ckpt, input, target);
            losses[i] = ckpt.Loss!.Value;
        }
        return losses;
    }

    private static int NodeCount(TrainingRig rig) => rig.TrainingStepPureGraph.ToInternal().Nodes.Count;

    private static bool Stamped(TrainingRig rig)
        => rig.TrainingStepPureGraph.ToInternal().Nodes.Any(n => CheckpointSegment.IdOf(n) is not null);

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
    }

    [Fact]
    public void TestCheckpointHintIsHonouredWhereTheMemoryPassWouldSkipCoverage()
    {
        var (plain, _, _) = MlpStackRig(Modules.PlainTinyMlpStack.ComputationGraph, [2L, 8L]);
        var (checkpointed, _, _) = MlpStackRig(Modules.CheckpointedTinyMlpStack.ComputationGraph, [2L, 8L]);

        Assert.True(plain.PreOptimizationEval.PeakMemoryBytes < MemoryAwareGraphOptimizer.MinimumPeakBytesToOptimize);
        Assert.Equal("Baseline", plain.OptimizationResult.StrategyName);
        Assert.True(NodeCount(checkpointed) > NodeCount(plain));
        Assert.True(checkpointed.OptimizationResult.Evaluation.PeakMemoryBytes <= plain.OptimizationResult.Evaluation.PeakMemoryBytes);
    }

    [Fact]
    public void TestRematerializedTrainingStepsMatchTheirOriginalsThroughOrtCoverage()
    {
        foreach (var model in new[] { Modules.PlainNarrowMlpStack.ComputationGraph, Modules.CheckpointedNarrowMlpStack.ComputationGraph })
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
        foreach (var order in new[] { EvaluationOrder.OrtOrder, EvaluationOrder.ProtoOrder })
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
    }

    [Fact]
    public void TestMemoryPassBenchmarkMeasuresTheRigsOwnModelCoverage()
    {
        var rig = TrainingRig.FromScratch(MemoryPassMlp.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            [new TensorDataModelParam("input", ModelParamType.InputParam, Pattern([64L, 256L], 1f))], 0.01f);
        var model = ProtoBuf.Serializer.Deserialize<Shorokoo.Core.Factory.IR.ModelProto>(
            new MemoryStream(Benchmarks.MemoryPassBenchmarkTests.RigModelBytes(rig.TrainingStepPureGraph, rig.OptimizationInputShapes)));
        Assert.Equal(rig.OptimizationInputShapes.Length, model.Graph.Inputs.Count);
        for (var i = 0; i < model.Graph.Inputs.Count; i++)
            Assert.Equal(rig.OptimizationInputShapes[i].Shape.Dims.Select(d => (long)d), model.Graph.Inputs[i].Type.TensorType.Shape.Dims.Select(d => d.DimValue));
    }
}
