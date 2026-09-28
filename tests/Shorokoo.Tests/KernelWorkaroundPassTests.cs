using System.Collections.Immutable;
using System.Reflection;
using Shorokoo.Core.Backends;
using Shorokoo.Core.Factory;
using Shorokoo.Core.Factory.IR;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Interpreter;
using Shorokoo.Core.Lowering;
using Shorokoo.Core.Lowering.KernelWorkarounds;
using Shorokoo.Core.Nodes.Processors.Fast;
using Shorokoo.Jax.Cpu;
using Shorokoo.Modules.Losses;
using Shorokoo.Modules.Optimizers;
using Shorokoo.PyTorch.Cpu;
using Shorokoo.Runtime;
using Shorokoo.Tests.Utils;
using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;
using static Shorokoo.Tests.OnnxProtoBuilders;

namespace Shorokoo.Tests;

[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class KernelWorkaroundPassTests
{
    private static readonly KernelWorkaroundSet NegSet = new("neg", [new NegAsSubtraction()]);
    private static readonly KernelWorkaroundSet AbsSet = new("abs", [new AbsAsIf()]);
    private static readonly KernelWorkaroundSet TopKSet = new("topk", [new TopKThroughIdentities()]);
    private static readonly KernelWorkaroundSet UnarySet = new("unary", [new UnaryThroughIdentity()]);
    private static readonly KernelWorkaroundSet NegThenSubSet = new("neg-sub", [new NegAsSubtraction(), new SubAsAddOfNeg()]);
    private static readonly KernelWorkaroundSet NegAbsSet = new("neg-abs", [new NegAsSubtraction(), new AbsAsIf()]);
    private static readonly KernelWorkaroundSet NegIfSet = new("neg-if", [new NegAsIf()]);
    private static readonly KernelWorkaroundSet AddConstantSet = new("add-constant", [new AddOfConstantAsSubtraction()]);
    private static readonly KernelWorkaroundSet AddShapeSet = new("add-shape", [new AddOfConstantReshapedToItsShape()]);

    [Fact]
    public void TestAGraphNoWorkaroundAppliesToBuildsTheSameModel()
    {
        var x = InputTensor<float32>("x", rank: 1);
        var d = InputTensor<float64>("d", rank: 1);
        Assert.Equal(Bytes(Session(Graph(x, OnnxOp.Mul(x, x)), null)), Bytes(Session(Graph(x, OnnxOp.Mul(x, x)), NegSet)));
        Assert.Equal(Bytes(Session(Graph(d, OnnxOp.Neg(d)), null)), Bytes(Session(Graph(d, OnnxOp.Neg(d)), NegSet)));
    }

    [Fact]
    public void TestTheOnnxRuntimeReductionWorkaroundsLeaveAScalarInputAsWritten()
    {
        var i = InputTensor<int64>("i", rank: 0);
        var h = InputTensor<float16>("h", rank: 0);
        var f = InputTensor<float32>("f", rank: 0);
        var axes = InputTensor<int64>("axes", rank: 1);
        Assert.True(AsWritten(Graph(i, i.Reduce(ReduceKind.Max))));
        Assert.True(AsWritten(Graph(i, i.Reduce(ReduceKind.Min, keepDims: true))));
        Assert.True(AsWritten(Graph(h, h.Reduce(ReduceKind.SumSquare))));
        Assert.True(AsWritten(new([f, axes], [OnnxOp.ReduceSum(f, axes, true, null)])));
    }

    [Fact]
    public void TestTheUnitAxisANoopReductionIsViewedWithIsNotNormalisedAgain()
    {
        var x = InputTensor<float32>("x", rank: 2);
        Assert.DoesNotContain(AllNodes(Session(Graph(x, NN.Reduce(ReduceKind.Sum, x, null, keepDims: true, noOp: true)), KernelWorkaroundRegistry.OnnxRuntime)), n => n.OpType == WHERE);
        Assert.DoesNotContain(AllNodes(Session(Graph(x, NN.Reduce(ReduceKind.Max, x, null, keepDims: false, noOp: true)), KernelWorkaroundRegistry.OnnxRuntime)), n => n.OpType == WHERE);
    }

    [Fact]
    public void TestAWorkaroundRewritesTheSessionModelInsideIfLoopAndFunctionBodies()
    {
        var x = InputTensor<float32>("x", rank: 1);
        var two = TensorData(DType.Float32, [2L], 2f, -3f);
        AssertRewritten(NegSet, MUL, SUB, true, Graph(x, OnnxOp.Mul(OnnxOp.Neg(x), x)), two);
        AssertRewritten(NegSet, "If", SUB, true, Graph(x, Shorokoo.Core.Nodes.Ops.IfElse((Scalar<bit>)OnnxOp.Less(OnnxOp.ReduceMin(x, keepdims: false), Scalar(0f)), OnnxOp.Neg(x), x)), two);
        AssertRewritten(NegSet, "Loop", SUB, true, Concrete(KernelWorkaroundNegInLoop.ComputationGraph, TensorData(DType.Float32, [2L], 2f, 3f)), TensorData(DType.Float32, [2L], 2f, 3f));
        AssertRewritten(NegSet, "Functions", SUB, true, Import(NegInAFunction()), TensorData(DType.Float32, [3L], 1f, -2f, 4f));
        AssertRewritten(AbsSet, ADD, "If", true, Graph(x, OnnxOp.Add(OnnxOp.Abs(x), x)), two);
        AssertRewritten(TopKSet, ADD, IDENTITY, false, TopKGraph(), TensorData(DType.Float32, [4L], 3f, 1f, 4f, 2f));
        AssertRewritten(UnarySet, ADD, IDENTITY, false, Graph(x, OnnxOp.Add(OnnxOp.Neg(x), OnnxOp.Abs(x))), two);
        AssertRewritten(NegThenSubSet, MUL, ADD, false, Graph(x, OnnxOp.Mul(OnnxOp.Neg(x), x)), two);
        Assert.DoesNotContain(AllNodes(Session(Graph(x, OnnxOp.Mul(OnnxOp.Neg(x), x)), NegThenSubSet)), n => n.OpType == SUB);
        AssertRewritten(TopKSet, "If", IDENTITY, false, TopKInIf(), TensorData(DType.Float32, [4L], 3f, 1f, 4f, 2f));
        AssertRewritten(TopKSet, "Loop", IDENTITY, false, Concrete(KernelWorkaroundTopKInLoop.ComputationGraph, TensorData(DType.Float32, [4L], 3f, 2f, 4f, 2f)), TensorData(DType.Float32, [4L], 3f, 2f, 4f, 2f));
    }

    [Fact]
    public void TestAPlanIsReusedOnlyForCallsWhoseConstantsItReadAgree()
    {
        var x = InputTensor<float32>("x", rank: 1);
        var y = InputTensor<float32>("y", rank: 2);
        var three = TensorData(DType.Float32, [3L], 1f, 2f, 3f);
        AssertRewritten(AddConstantSet, MUL, SUB, true, Graph(x, OnnxOp.Mul(OnnxOp.Add(x, Scalar(1f)), OnnxOp.Add(x, Scalar(2f)))), three);
        AssertRewritten(AddShapeSet, MUL, RESHAPE, false, Graph(y, OnnxOp.Mul(OnnxOp.Add(y, Constant([1L, 3L], 1f, 2f, 3f)), OnnxOp.Add(y, Constant([2L, 1L], 4f, 5f)))),
            TensorData(DType.Float32, [2L, 3L], 1f, 2f, 3f, 4f, 5f, 6f));
    }

    [Fact]
    public void TestAFunctionBodyIsTypedWhenAWorkaroundSplicesAnIfIntoWhatAnExportLoweringBuilt()
    {
        using var lowering = OpLoweringRegistry.Override(new OpLowering(ABS, Method(nameof(AbsAsMaxOfNeg))));
        using var listed = FastOnnxModelBuilder.OverrideExportLoweredOpCodes(ABS);
        AssertRewritten(NegIfSet, "Functions", IF, false, Import(UnaryInAFunction("Abs")), TensorData(DType.Float32, [3L], 1f, -2f, 4f));
    }

    [Fact]
    public void TestALoweringEndingInAnIfMakesItsHostTheIfClosePairedWithTheSplicedOpen()
    {
        var x = InputTensor<float32>("x", rank: 1);
        var g = Graph(x, OnnxOp.Mul(OnnxOp.Abs(x), x));
        var host = g.Nodes.Single(n => n.OpCode == ABS).Key;
        using var lowering = OpLoweringRegistry.Override(new OpLowering(ABS, Method(nameof(AbsLoweredToAnIf))));
        FastLowerRegisteredOps.Process(g, new HashSet<string>([ABS]));
        var close = g.Nodes.Single(n => n.Key == host);
        Assert.Equal(IF_CLOSE, close.OpCode);
        Assert.Equal(g.Nodes.Single(n => n.OpCode == IF_OPEN).Key, close.GraphOpenNodeKey);
        Assert.True(g.IsLinearOrderValid());
        Assert.Equal<float>([-4f, 9f], ((RuntimeTensor)new QuickExecutionEngine().Run(g, TensorData(DType.Float32, [2L], -2f, 3f))[g.Outputs[0]]).FloatData!.Value);
    }

    [Fact]
    public void TestEveryValueOfTheModelBuiltWithoutWorkaroundsKeepsItsName()
    {
        var x = InputTensor<float32>("x", rank: 1);
        var g = Graph(x, OnnxOp.Add(OnnxOp.Mul(OnnxOp.Neg(x), x), OnnxOp.Neg(OnnxOp.Abs(x))));
        Assert.Empty(Untouched(g, NegSet).Except(Signatures(Session(g, NegSet))));
        Assert.Empty(Untouched(g, AbsSet).Except(Signatures(Session(g, AbsSet))));
        Assert.Empty(UntouchedValues(TopKGraph(), TopKSet).Except(Values(Session(TopKGraph(), TopKSet))));
        Assert.Empty(UntouchedValues(TopKInIf(), TopKSet).Except(Values(Session(TopKInIf(), TopKSet))));
        var loop = Concrete(KernelWorkaroundTopKInLoop.ComputationGraph, TensorData(DType.Float32, [4L], 3f, 2f, 4f, 2f));
        Assert.Empty(UntouchedValues(loop, TopKSet).Except(Values(Session(loop, TopKSet))));
    }

    [Fact]
    public void TestTheWorkaroundPassTypesWhatItSplicesWithoutRebuildingTheLookup()
    {
        var x = InputTensor<float32>("x", rank: 1);
        Assert.Same(FastApplyKernelWorkarounds.Splices.None, Applied(Graph(x, OnnxOp.Mul(x, x)), NegAbsSet));
        Assert.True(TypesAsARebuildWould(Graph(x, OnnxOp.Mul(OnnxOp.Neg(x), x)), NegThenSubSet));
        Assert.True(TypesAsARebuildWould(Graph(x, OnnxOp.Add(OnnxOp.Neg(x), OnnxOp.Abs(x))), NegAbsSet));
        Assert.True(TypesAsARebuildWould(TopKGraph(), TopKSet));
        Assert.True(TypesAsARebuildWould(TopKInIf(), TopKSet));
        Assert.True(TypesAsARebuildWould(TrainingStep(), KernelWorkaroundRegistry.OnnxRuntime));
    }

    [Fact]
    public void TestAWorkaroundThatFailsNamesItselfAndTheCall()
    {
        var x = InputTensor<float32>("x", rank: 1);
        var g = Graph(x, OnnxOp.Neg(x));
        var call = g.Nodes.Single(n => n.OpCode == NEG);
        var name = call.FriendlyName ?? call.Key.ToString();
        foreach (var workaround in (KernelWorkaround[])[new Failing(FailingAt.Applies), new Failing(FailingAt.Rewrite), new Failing(FailingAt.OutputCount)])
        {
            var message = Assert.Throws<InvalidOperationException>(() => Session(g, new("failing", [workaround]))).Message;
            Assert.Contains(workaround.Name, message);
            Assert.Contains(name, message);
        }
    }

    [Fact]
    public void TestTheBackendAuditComparesTheOutputsOfACallAWorkaroundDropped()
    {
        var x = TensorData(DType.Float32, [4L], 3f, 1f, 4f, 2f);
        Assert.Equal(["TopK"], QeeAuditOnBackend.Torch.ConvictedOperators(TopKGraph(), [x], SmallestFirst, TopKSet));
    }

    [Fact]
    public void TestOnlyTheSessionModelCarriesTheOnnxRuntimeWorkaround()
    {
        var x = InputTensor<float32>("x", rank: 3);
        var g = Graph(x, OnnxOp.AveragePool(x, null, null, true, null, [2L], [2L, 0L], null));
        string[] asWritten = [.. g.Nodes.Select(n => n.OpCode)];
        Assert.Equal([0f, 0.5f, 1.5f, 2.5f, 3.5f, 4.5f],
            Pooled(g));
        Assert.Equal(asWritten, g.Nodes.Select(n => n.OpCode));
        Assert.Contains(AllNodes(Session(g, KernelWorkaroundRegistry.OnnxRuntime)), n => n.OpType == PAD);
        Assert.DoesNotContain(AllNodes(FastOnnxModelBuilder.BuildOnnxModel(g)), n => n.OpType == PAD);
        Assert.DoesNotContain(AllNodes(FastOnnxModelBuilder.BuildInternalOnnxModel(g, applyExecutionLowerings: false,
            emitInputsAsNodes: true, workarounds: KernelWorkaroundRegistry.OnnxRuntime)), n => n.OpType == PAD);
        Assert.DoesNotContain(AllNodes(FastOnnxModelBuilder.BuildInternalOnnxModel(g,
            workarounds: KernelWorkaroundRegistry.OnnxRuntime)), n => n.OpType == PAD);
    }

    [Fact]
    public void TestAnImportedModelGetsTheOnnxRuntimeWorkaroundWhenItsSessionIsBuilt()
    {
        var g = Import(PaddedAsFarAsTheKernel());
        Assert.Equal([0f, 0.5f, 1.5f, 2.5f, 3.5f, 4.5f],
            Pooled(g));
        Assert.DoesNotContain(g.Nodes, n => n.OpCode == PAD);
    }

    [Fact]
    public void TestOnlyTheOnnxRuntimeBackendsNameAWorkaroundSet()
    {
        Assert.Same(KernelWorkaroundRegistry.OnnxRuntime, KernelWorkaroundRegistry.For(DefaultBackend.Instance.KernelWorkaroundSet));
        Assert.True(KernelWorkaroundRegistry.For(((IShorokooBackend)new TorchCpuBackend()).KernelWorkaroundSet).IsEmpty);
        Assert.True(KernelWorkaroundRegistry.For(((IShorokooBackend)new JaxCpuBackend()).KernelWorkaroundSet).IsEmpty);
        Assert.True(KernelWorkaroundRegistry.For(((IShorokooBackend)HostBackend.Instance).KernelWorkaroundSet).IsEmpty);
        Assert.True(KernelWorkaroundRegistry.For("none of them").IsEmpty);
    }

    [Fact]
    public void TestBuildersBuildThePlainOperator()
    {
        var x = InputTensor<float32>("x", rank: 3);
        var y = InputTensor<float32>("y", rank: 4);
        var i = InputTensor<int64>("i", rank: 2);
        Assert.True(Plain(OnnxOp.AveragePool(x, null, null, true, null, [2L], [2L, 0L], null), AVERAGE_POOL, x));
        Assert.True(Plain(OnnxOp.MaxPool(x, AutoPad.SameUpper, null, [2L], [2L], null, null, null), MAX_POOL, x));
        Assert.True(Plain(OnnxOp.MaxPoolWithIndices(x, null, null, null, [2L], [2L, 2L], null, null).indices, MAX_POOL, x));
        Assert.True(Plain(OnnxOp.LpPool(x, AutoPad.SameLower, null, null, [2L], 2L, null, [3L]), LP_POOL, x));
        Assert.True(Plain(OnnxOp.Col2Im(x, Vector(5L), Vector(2L), [1L], [1L, 1L], [1L]), COL2IM, x));
        Assert.True(Plain(OnnxOp.Resize(y, Vector(0f, 0f, 0f, 0.5f, 1f, 1f, 1f, 1.5f), Vector(1f, 1f, 1f, 1f), null, null, null,
            CoordinateTransformationMode.Tf_crop_and_resize, null, null, -1f, null, ResizeMode.Linear, null), RESIZE, y));
        Assert.True(Plain((Variable)i.Reduce(ReduceKind.Max, Vector(1L)), REDUCE_MAX, i));
        Assert.True(Plain((Variable)i.Reduce(ReduceKind.Min), REDUCE_MIN, i));
    }

    [Fact]
    public void TestATrainingStepRunsNoIfOfTheOnnxRuntimeWorkaroundsWithOrWithoutConcreteShapes()
    {
        var step = TrainingStep();
        List<long[]?> dims = [.. step.InputNodes.Select(RepresentativeInputShapes.Get)];
        Assert.Equal(Ifs(Session(step, null)), Ifs(Session(step, KernelWorkaroundRegistry.OnnxRuntime)));
        Assert.True(Ifs(FastOnnxModelBuilder.BuildInternalOnnxModel(step, prepForOnnx: true, inputDims: dims, workarounds: KernelWorkaroundRegistry.OnnxRuntime)) > 0);
        Assert.Equal(0, Ifs(Optimized(FastOnnxModelBuilder.BuildInternalOnnxModel(step, prepForOnnx: true, inputDims: dims, workarounds: KernelWorkaroundRegistry.OnnxRuntime))));
    }

    [Fact]
    public void TestOnnxRuntimeFoldsEveryIfOfTheReductionWorkaroundsWithConcreteShapes()
    {
        var x = InputTensor<float32>("x", rank: 2);
        var axes = InputTensor<int64>("axes", rank: 1);
        var g = new InternalComputationGraph([x, axes], [x.Cast<float16>().Reduce(ReduceKind.SumSquare), x.Cast<int64>().Reduce(ReduceKind.Max, Vector(1L)),
            x.Cast<bit>().Reduce(ReduceKind.Min, Vector(0L)), NN.Reduce(ReduceKind.Sum, x, axes, keepDims: true, noOp: true)]);
        List<long[]?> dims = [[2L, 3L], [1L]];
        var built = FastOnnxModelBuilder.BuildInternalOnnxModel(g, prepForOnnx: true, inputDims: dims, workarounds: KernelWorkaroundRegistry.OnnxRuntime);
        Assert.True(Ifs(built) >= 3);
        Assert.Equal(0, Ifs(Optimized(built)));
    }

    [Fact]
    public void TestANoopReductionComputesItsGroupsOnOnnxRuntimeWithConcreteShapes()
    {
        var x = TensorData(DType.Float32, [2L, 3L], 1f, 2f, 3f, 4f, 5f, 6f);
        Assert.True(AllTrueWithConcreteShapes(NoopReduceOfEachElementCheck.ComputationGraph, x));
        Assert.True(AllTrueWithConcreteShapes(NoopReduceOfEachElementByShapeCheck.ComputationGraph, x));
        Assert.True(AllTrueWithConcreteShapes(NoopReduceAxesFormsCheck.ComputationGraph, x));
    }

    private static InternalComputationGraph Graph(Variable input, Variable output) => new([input], [output]);

    private static bool AsWritten(InternalComputationGraph g)
        => Bytes(Session(g, null)).SequenceEqual(Bytes(Session(g, KernelWorkaroundRegistry.OnnxRuntime)));

    private static Variable Constant(long[] dims, params float[] values) => OnnxOp.Constant(TensorAttribute.Create(new Shape(dims), values));

    private static MethodInfo Method(string name) => typeof(KernelWorkaroundPassTests).GetMethod(name, BindingFlags.NonPublic | BindingFlags.Static)!;

    private static Variable?[] AbsAsMaxOfNeg<T>(Tensor<T> x) where T : IVarType => [OnnxOp.Max(x, OnnxOp.Neg(x))];

    private static Variable?[] AbsLoweredToAnIf<T>(Tensor<T> x) where T : IVarType
        => [Shorokoo.Core.Nodes.Ops.IfElse((Scalar<bit>)OnnxOp.Less(OnnxOp.ReduceMin(x, keepdims: false), OnnxOp.CastLike(Scalar(0f), x, null)),
            OnnxOp.Mul(x, OnnxOp.Sign(x)), OnnxOp.Identity(x, null))];

    private static FastApplyKernelWorkarounds.Splices Applied(InternalComputationGraph g, KernelWorkaroundSet set)
        => FastApplyKernelWorkarounds.Process(g.Clone(), set);

    private static bool TypesAsARebuildWould(InternalComputationGraph g, KernelWorkaroundSet set)
    {
        var applied = g.Clone();
        var splices = FastApplyKernelWorkarounds.Process(applied, set);
        FastAddIdentityForOuterScopeValues.Process(applied);
        FastPrepForOnnx.Process(applied);
        var shared = splices.TensorInfo(applied);
        var rebuilt = FastTensorInfoProcessor.BuildTensorInfoLookup(applied);
        return shared is not null && rebuilt.Count > 0 && rebuilt.All(p => shared.TryGetValue(p.Key, out var info)
            && info.DType == p.Value.DType && info.Structure == p.Value.Structure && (info.Rank == p.Value.Rank || p.Value.Rank is null));
    }

    private static InternalComputationGraph TrainingStep()
        => TrainingRig.FromScratch(ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, AdamWOptimizer.ComputationGraph,
            [new TensorDataModelParam("input", ModelParamType.InputParam, TensorData([4L], [1f, 2f, 3f, 4f]))],
            new AdamWOptimizerHyperparameters { LearningRate = 0.1f }).TrainingStepPureGraph.ToInternal();

    private static ModelProto SmallestFirst(ModelProto model)
    {
        var topK = model.Graph.Nodes.Single(n => n.OpType == TOPK);
        topK.Attributes.RemoveAll(a => a.Name == "largest");
        topK.Attributes.Add(new AttributeProto { Name = "largest", Type = AttributeProto.AttributeType.Int, I = 0 });
        return model;
    }

    private static int Ifs(ModelProto model) => AllNodes(model).Count(n => n.OpType == IF);

    private static ModelProto Optimized(ModelProto model)
    {
        var path = Path.Combine(Path.GetTempPath(), $"shrk_kw_{Guid.NewGuid():N}.onnx");
        try
        {
            using (var options = new Microsoft.ML.OnnxRuntime.SessionOptions { OptimizedModelFilePath = path })
            using (new Microsoft.ML.OnnxRuntime.InferenceSession(Bytes(model), options)) { }
            using var file = File.OpenRead(path);
            return ProtoBuf.Serializer.Deserialize<ModelProto>(file);
        }
        finally { File.Delete(path); }
    }

    private static bool AllTrueWithConcreteShapes(ComputationGraph module, TensorData x)
    {
        using var context = new ComputeContext();
        using var compiled = context.Compile(Concrete(module, x), [x.Shape.Dims], trainingStep: false);
        var bits = compiled.Execute(x.Shared())[0].ToTensorData().AccessRawMemory().ToArray();
        return bits.Length > 0 && bits.All(b => b != 0);
    }

    private static InternalComputationGraph Concrete(ComputationGraph module, TensorData x)
        => module.ToInternal().ToConcreteArchitecture([x]).ToConcreteModel();

    private static InternalComputationGraph TopKGraph()
    {
        var x = InputTensor<float32>("x", rank: 1);
        var (values, indices) = OnnxOp.TopK(x, Vector(2L));
        return new([x], [OnnxOp.Add(values, values), indices]);
    }

    private static InternalComputationGraph TopKInIf()
    {
        var x = InputTensor<float32>("x", rank: 1);
        var (largest, at) = OnnxOp.TopK(OnnxOp.Neg(x), Vector(2L));
        var (smallest, _) = OnnxOp.TopK(OnnxOp.Abs(x), Vector(2L), largest: false);
        return new([x], [Shorokoo.Core.Nodes.Ops.IfElse((Scalar<bit>)OnnxOp.Less(OnnxOp.ReduceMin(x, keepdims: false), Scalar(2f)),
            OnnxOp.Add(largest, OnnxOp.CastLike(at, largest, null)), OnnxOp.Mul(smallest, smallest))]);
    }

    private static ModelProto Session(InternalComputationGraph g, KernelWorkaroundSet? set)
        => FastOnnxModelBuilder.BuildInternalOnnxModel(g, prepForOnnx: true, workarounds: set);

    private static void AssertRewritten(KernelWorkaroundSet set, string within, string introduced, bool removes, InternalComputationGraph g, params TensorData[] x)
    {
        string[] asWritten = [.. g.Nodes.Select(n => n.OpCode)];
        var plain = Session(g, null);
        var rewritten = Session(g, set);
        Assert.Contains(AllNodes(plain), n => n.OpType == within || n.Domain == within);
        Assert.True(AllNodes(rewritten).Count(n => n.OpType == introduced) > AllNodes(plain).Count(n => n.OpType == introduced));
        Assert.Equal(removes, !AllNodes(rewritten).Any(n => set.OpCodes.Contains(n.OpType)));
        Assert.Equal(Run(g, plain, x), Run(g, rewritten, x));
        Assert.Equal(asWritten, g.Nodes.Select(n => n.OpCode));
    }

    private static byte[][] Run(InternalComputationGraph g, ModelProto model, TensorData[] x)
        => [.. ComputeContext.Default.ExecuteModel(g, model, [.. x.Select(t => (IData)t.Shared())])
            .Select(p => p.ToTensorData().AccessRawMemory().ToArray())];

    private static float[] Pooled(InternalComputationGraph g)
    {
        using var context = new ComputeContext();
        return System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(context
            .Execute(g, TensorData(DType.Float32, [1L, 1L, 5L], 1f, 2f, 3f, 4f, 5f).Shared())[0].ToTensorData().AccessRawMemory()).ToArray();
    }

    private static byte[] Bytes(ModelProto model)
    {
        using var stream = new MemoryStream();
        ProtoBuf.Serializer.Serialize(stream, model);
        return stream.ToArray();
    }

    private static IEnumerable<NodeProto> AllNodes(ModelProto model)
        => model.Graph.Nodes.Concat(model.Functions.SelectMany(f => f.Nodes)).SelectMany(Within);

    private static IEnumerable<NodeProto> Within(NodeProto node)
        => node.Attributes.Where(a => a.G is not null).SelectMany(a => a.G!.Nodes.SelectMany(Within)).Prepend(node);

    private static IEnumerable<string> Signatures(ModelProto model)
        => model.Graph.Nodes.Select(n => $"{n.OpType}({string.Join(",", n.Inputs)})->{string.Join(",", n.Outputs)}")
            .Concat(model.Graph.Outputs.Select(o => o.Name));

    private static IEnumerable<string> Untouched(InternalComputationGraph g, KernelWorkaroundSet set)
        => Signatures(Session(g, null)).Where(s => !set.OpCodes.Any(op => s.StartsWith(op + "(", StringComparison.Ordinal)));

    private static IEnumerable<string> Values(ModelProto model) => AllNodes(model).SelectMany(n => n.Outputs);

    private static IEnumerable<string> UntouchedValues(InternalComputationGraph g, KernelWorkaroundSet set)
        => AllNodes(Session(g, null)).Where(n => !set.OpCodes.Contains(n.OpType)).SelectMany(n => n.Outputs);

    private static bool Plain(Variable output, string opCode, Variable input)
        => output.OwningNode is { } node && node.OpCode == opCode && node.Inputs[0] == input;

    private static AttributeProto Ints(string name, params long[] values)
        => new() { Name = name, Type = AttributeProto.AttributeType.Ints, Ints = values };

    private static NodeProto Node(string opType, string[] inputs, string[] outputs, params AttributeProto[] attributes)
    {
        var node = new NodeProto { OpType = opType, Name = outputs[0] + "_node" };
        node.Inputs.AddRange(inputs);
        node.Outputs.AddRange(outputs);
        node.Attributes.AddRange(attributes);
        return node;
    }

    private static ModelProto PaddedAsFarAsTheKernel()
    {
        var graph = new GraphProto { Name = "padded_pool" };
        graph.Inputs.Add(TensorInfo("x", 1, 1, 1, 5));
        graph.Nodes.Add(Node("AveragePool", ["x"], ["y"], Ints("kernel_shape", 2), Ints("pads", 2, 0),
            new AttributeProto { Name = "count_include_pad", Type = AttributeProto.AttributeType.Int, I = 1 }));
        graph.Outputs.Add(TensorInfo("y", 1, 1, 1, 6));
        return WrapModel(graph);
    }

    private static ModelProto NegInAFunction() => UnaryInAFunction("Neg");

    private static ModelProto UnaryInAFunction(string op)
    {
        var fn = new FunctionProto { Name = op + "Fn", Domain = "Functions" };
        fn.OpsetImports.Add(new OperatorSetIdProto { Domain = "", Version = 21 });
        fn.Inputs.Add("fn_x");
        fn.Outputs.Add("fn_y");
        fn.ValueInfoes.Add(TensorInfo("fn_x", 1, 3));
        fn.ValueInfoes.Add(TensorInfo("fn_y", 1, 3));
        fn.Nodes.Add(Node(op, ["fn_x"], ["negated"]));
        fn.Nodes.Add(Node("Mul", ["negated", "fn_x"], ["fn_y"]));

        var graph = new GraphProto { Name = "neg_fn_graph" };
        graph.Inputs.Add(TensorInfo("x", 1, 3));
        var call = Node(op + "Fn", ["x"], ["y"]);
        call.Domain = "Functions";
        graph.Nodes.Add(call);
        graph.Outputs.Add(TensorInfo("y", 1, 3));

        var model = WrapModel(graph);
        model.OpsetImports.Add(new OperatorSetIdProto { Domain = "Functions", Version = 1 });
        model.Functions.Add(fn);
        return model;
    }

    private sealed class NegAsSubtraction : KernelWorkaround
    {
        public override IReadOnlySet<string> OpCodes { get; } = new HashSet<string>([NEG]);

        public override bool Applies(WorkaroundSite site) => site.DTypeOf(0) == DType.Float32;

        public override Variable?[] Rewrite(WorkaroundSite site, Variable?[] inputs)
            => [OnnxOp.Sub(OnnxOp.CastLike(Scalar(0f), inputs[0]!, null), inputs[0]!)];
    }

    private sealed class AbsAsIf : KernelWorkaround
    {
        public override IReadOnlySet<string> OpCodes { get; } = new HashSet<string>([ABS]);

        public override bool Applies(WorkaroundSite site) => true;

        public override Variable?[] Rewrite(WorkaroundSite site, Variable?[] inputs)
        {
            var x = inputs[0]!;
            return [Shorokoo.Core.Nodes.Ops.IfElse((Scalar<bit>)OnnxOp.Less(OnnxOp.ReduceMin(x, keepdims: false), Scalar(0f)),
                OnnxOp.Mul(x, OnnxOp.Sign(x)), OnnxOp.Identity(x, null))];
        }
    }

    private sealed class UnaryThroughIdentity : KernelWorkaround
    {
        public override IReadOnlySet<string> OpCodes { get; } = new HashSet<string>([NEG, ABS]);

        public override bool Applies(WorkaroundSite site) => true;

        public override Variable?[] Rewrite(WorkaroundSite site, Variable?[] inputs)
            => [OnnxOp.Identity(site.OpCode == NEG ? OnnxOp.Neg(inputs[0]!) : OnnxOp.Abs(inputs[0]!), null)];
    }

    private sealed class SubAsAddOfNeg : KernelWorkaround
    {
        public override IReadOnlySet<string> OpCodes { get; } = new HashSet<string>([SUB]);

        public override bool Applies(WorkaroundSite site) => true;

        public override Variable?[] Rewrite(WorkaroundSite site, Variable?[] inputs) => [OnnxOp.Add(inputs[0]!, OnnxOp.Neg(inputs[1]!))];
    }

    private sealed class NegAsIf : KernelWorkaround
    {
        public override IReadOnlySet<string> OpCodes { get; } = new HashSet<string>([NEG]);

        public override bool Applies(WorkaroundSite site) => true;

        public override Variable?[] Rewrite(WorkaroundSite site, Variable?[] inputs)
        {
            var x = inputs[0]!;
            return [Shorokoo.Core.Nodes.Ops.IfElse((Scalar<bit>)OnnxOp.Less(OnnxOp.ReduceMin(x, keepdims: false), OnnxOp.CastLike(Scalar(0f), x, null)),
                OnnxOp.Sub(OnnxOp.CastLike(Scalar(0f), x, null), x), OnnxOp.Neg(x))];
        }
    }

    private sealed class AddOfConstantAsSubtraction : KernelWorkaround
    {
        public override IReadOnlySet<string> OpCodes { get; } = new HashSet<string>([ADD]);

        public override bool Applies(WorkaroundSite site) => site.ConstantOf(1) is { DType: var t } && t == DType.Float32;

        public override Variable?[] Rewrite(WorkaroundSite site, Variable?[] inputs)
            => [OnnxOp.Sub(inputs[0]!, Scalar(-site.ConstantOf(1)!.Elements<float>()[0]))];
    }

    private sealed class AddOfConstantReshapedToItsShape : KernelWorkaround
    {
        public override IReadOnlySet<string> OpCodes { get; } = new HashSet<string>([ADD]);

        public override bool Applies(WorkaroundSite site) => site.ConstantShapeOf(1) is not null;

        public override Variable?[] Rewrite(WorkaroundSite site, Variable?[] inputs)
            => [OnnxOp.Add(inputs[0]!, OnnxOp.Reshape(inputs[1]!, Vector(site.ConstantShapeOf(1)!.Dims), false))];
    }

    private enum FailingAt { Applies, Rewrite, OutputCount }

    private sealed class Failing(FailingAt at) : KernelWorkaround
    {
        public override IReadOnlySet<string> OpCodes { get; } = new HashSet<string>([NEG]);

        public override string Name => $"Failing{at}";

        public override bool Applies(WorkaroundSite site) => at != FailingAt.Applies ? true : throw new ArgumentException("applies");

        public override Variable?[] Rewrite(WorkaroundSite site, Variable?[] inputs)
            => at == FailingAt.Rewrite ? throw new ArgumentException("rewrite") : [inputs[0]!, inputs[0]!];
    }

    private sealed class TopKThroughIdentities : KernelWorkaround
    {
        public override IReadOnlySet<string> OpCodes { get; } = new HashSet<string>([TOPK]);

        public override bool Applies(WorkaroundSite site) => true;

        public override Variable?[] Rewrite(WorkaroundSite site, Variable?[] inputs)
        {
            var (values, indices) = OnnxOp.TopK(inputs[0]!, inputs[1]!, site.Attributes.GetLongVal(OnnxOpAttributeNames.AttrAxis),
                site.Attributes.GetBoolVal(OnnxOpAttributeNames.AttrLargest), site.Attributes.GetBoolVal(OnnxOpAttributeNames.AttrSorted));
            return [OnnxOp.Identity(values, null), OnnxOp.Identity(indices, null)];
        }
    }
}

[Module]
public partial class KernelWorkaroundNegInLoop
{
    public static Tensor<float32> Inline(Tensor<float32> x)
    {
        var a = x;
        var trips = x.Reduce(ReduceKind.Min, keepDims: false).Scalar().Cast<int64>();
        foreach (var ctx in LoopAPI.Iterate(trips))
            a = (Tensor<float32>)OnnxOp.Neg(a) * Scalar(2f);
        return a;
    }
}

[Module]
public partial class KernelWorkaroundTopKInLoop
{
    public static Tensor<float32> Inline(Tensor<float32> x)
    {
        var a = x;
        var trips = x.Reduce(ReduceKind.Min, keepDims: false).Scalar().Cast<int64>();
        foreach (var ctx in LoopAPI.Iterate(trips))
        {
            var (values, indices) = OnnxOp.TopK(a, Vector(4L));
            a = (Tensor<float32>)OnnxOp.Add(values, OnnxOp.CastLike(indices, values, null));
        }
        return a;
    }
}
