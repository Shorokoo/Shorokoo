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
    private static readonly KernelWorkaroundSet MulIfSet = new("mul-if", [new MulAsIfOnItsRightShape()]);
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
    public void TestTheIntegerRangeCountWorkaroundFiresOnlyWhereOnnxRuntimeCanMiscount()
    {
        var i = InputTensor<int64>("i", rank: 0);
        var n = InputTensor<int32>("n", rank: 0);
        var h = InputTensor<int16>("h", rank: 0);
        Assert.False(AsWritten(Graph(i, OnnxOp.Range(Scalar(0L), Scalar(1L << 62), i))));
        Assert.False(AsWritten(new([], [OnnxOp.Range(Scalar(0L), Scalar((1L << 62) + 1L), Scalar(1L << 61))])));
        Assert.False(AsWritten(new([], [OnnxOp.Range(Scalar(1L << 54), Scalar(0L), Scalar(-2L))])));
        Assert.True(AsWritten(new([], [OnnxOp.Range(Scalar(1L - (1L << 53)), Scalar(0L), Scalar(1L << 50))])));
        Assert.True(AsWritten(Graph(n, OnnxOp.Range(Scalar(0), n, Scalar(1)))));
        Assert.True(AsWritten(Graph(n, OnnxOp.Range(Scalar(0), n, Scalar(-1)))));
        Assert.True(AsWritten(new([], [OnnxOp.Range(Scalar(int.MinValue), Scalar(-1), Scalar(1 << 30))])));
        Assert.True(AsWritten(Graph(n, OnnxOp.Range(Scalar(2), n, Scalar(1)))));
        Assert.True(AsWritten(Graph(n, OnnxOp.Range(Scalar(0), n, Scalar(2)))));
        Assert.False(AsWritten(new([], [OnnxOp.Range(Scalar(int.MinValue), Scalar(int.MaxValue), Scalar(1 << 30))])));
        Assert.True(AsWritten(Graph(h, OnnxOp.Range(Scalar((short)0), h, Scalar((short)3)))));
        Assert.True(AsWritten(Graph(i, OnnxOp.Range(Scalar(0L), i, Scalar(1L)))));
        Assert.True(AsWritten(Graph(i, OnnxOp.Range(Scalar(2L), i, Scalar(1L)))));
        Assert.True(AsWritten(Graph(i, OnnxOp.Range(Scalar(1L << 52), i, Scalar(-1L)))));
        Assert.True(AsWritten(Graph(i, OnnxOp.Range(i, Scalar(-(1L << 52)), Scalar(-1L)))));
        Assert.False(AsWritten(new([], [OnnxOp.Range(Scalar(1L << 53), Scalar(1L - (1L << 53)), Scalar(-1L))])));
        Assert.False(AsWritten(Graph(i, OnnxOp.Range(Scalar(1L - (1L << 62)), i, Scalar(-1L)))));
        Assert.False(AsWritten(Graph(i, OnnxOp.Range(Scalar(1L << 62), i, Scalar(1L)))));
        Assert.False(AsWritten(Graph(i, OnnxOp.Range(i, Scalar(long.MinValue), Scalar(-1L)))));
        Assert.False(AsWritten(new([], [OnnxOp.Range(Scalar(long.MaxValue - 2L), Scalar(long.MinValue + 2L), Scalar(1L))])));
        Assert.False(AsWritten(new([], [OnnxOp.Range(Scalar(1L << 53), Scalar((1L << 53) + 2L), Scalar(1L))])));
        Assert.False(AsWritten(new([i, n], [OnnxOp.Range(i, OnnxOp.Add(i, OnnxOp.Cast(n, null, DType.Int64)), Scalar(1L))])));
        Assert.False(AsWritten(Graph(i, OnnxOp.Range(Scalar(0L), i, Scalar(2L)))));
    }

    [Fact]
    public void TestAUnitStepRangeWhoseSpanWrapsItsTypeIsEmpty()
    {
        Assert.Empty(UnitRange(long.MaxValue - 2L, long.MinValue + 2L, 1L));
        Assert.Empty(UnitRange(long.MinValue, long.MaxValue, -1L));
        Assert.Empty(UnitRange((1L << 62) - 1L, long.MinValue, 1L));
        Assert.Empty(UnitRange32(int.MaxValue - 2, int.MinValue + 2, 1));
        Assert.Empty(UnitRange32(int.MinValue, int.MaxValue, -1));
    }

    private static long[] UnitRange(long start, long limit, long delta)
    {
        var (s, l) = (InputScalar<int64>("s"), InputScalar<int64>("l"));
        return ComputeContext.Default.Execute(new InternalComputationGraph([s, l], [OnnxOp.Range(s, l, Scalar(delta))]),
            TensorData(DType.Int64, [], start).Shared(), TensorData(DType.Int64, [], limit).Shared())[0].ToTensorData().As<int64>().CopyMemory<long>();
    }

    private static int[] UnitRange32(int start, int limit, int delta)
    {
        var (s, l) = (InputScalar<int32>("s"), InputScalar<int32>("l"));
        return ComputeContext.Default.Execute(new InternalComputationGraph([s, l], [OnnxOp.Range(s, l, Scalar(delta))]),
            TensorData(DType.Int32, [], start).Shared(), TensorData(DType.Int32, [], limit).Shared())[0].ToTensorData().As<int32>().CopyMemory<int>();
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
        Assert.Same(DefaultBackend.Instance.Description.Device == ComputeDevice.Cuda ? KernelWorkaroundRegistry.OnnxRuntimeCuda : KernelWorkaroundRegistry.OnnxRuntime,
            KernelWorkaroundRegistry.For(DefaultBackend.Instance.KernelWorkaroundSet));
        Assert.Same(KernelWorkaroundRegistry.OnnxRuntimeCuda, KernelWorkaroundRegistry.For(KernelWorkaroundSets.OnnxRuntimeCuda));
        Assert.Equal(KernelWorkaroundRegistry.OnnxRuntime.Workarounds.Select(w => w.Name),
            KernelWorkaroundRegistry.OnnxRuntimeCuda.Workarounds.SkipLast(1).Select(w => w.Name));
        Assert.True(KernelWorkaroundRegistry.For(((IShorokooBackend)new TorchCpuBackend()).KernelWorkaroundSet).IsEmpty);
        Assert.True(KernelWorkaroundRegistry.For(((IShorokooBackend)new JaxCpuBackend()).KernelWorkaroundSet).IsEmpty);
        Assert.True(KernelWorkaroundRegistry.For(((IShorokooBackend)HostBackend.Instance).KernelWorkaroundSet).IsEmpty);
        Assert.True(KernelWorkaroundRegistry.For("none of them").IsEmpty);
    }

    [Fact]
    public void TestTheCudaSetTakesWhatABodyDidNotComputeThroughMaxBeforeACallTheProviderRunsOnTheCpu()
    {
        var u = InputTensor<uint32>("u", rank: 1);
        var w = InputTensor<uint64>("w", rank: 1);
        var i = InputTensor<int64>("i", rank: 1);
        Assert.True(CudaMaxes(InBody(u, (outer, carried) => OnnxOp.BitwiseXor(outer, Scalar(5u)))) > 0);
        Assert.True(CudaMaxes(InBody(u, (outer, carried) => OnnxOp.BitShift(carried, Scalar(1u), BitShiftDirection.Right))) > 0);
        Assert.True(CudaMaxes(InBody(w, (outer, carried) => OnnxOp.BitwiseOr(outer, carried + Scalar(1UL)))) > 0);
        Assert.True(CudaMaxes(InBody(i, (outer, carried) => OnnxOp.BitwiseAnd(outer, carried))) > 0);
        Assert.True(CudaMaxes(InBody(u, (outer, carried) => OnnxOp.BitwiseXor(OnnxOp.Identity(outer, null), Scalar(5u)))) > 0);
        Assert.True(CudaMaxes(InBody(u, (outer, carried) => OnnxOp.BitwiseXor(OnnxOp.Cast(outer, null, DType.UInt32), Scalar(5u)))) > 0);
        Assert.True(CudaMaxes(Concrete(KernelWorkaroundRandomNormalInLoop.ComputationGraph, TensorData(DType.Float32, [2L], 2f, 3f))) > 0);
        Assert.Equal(0, CudaMaxes(InBody(u, (outer, carried) => OnnxOp.BitwiseXor(carried + Scalar(1u), carried * Scalar(3u)))));
        Assert.Equal(0, CudaMaxes(InBody(w, (outer, carried) => OnnxOp.Equal(outer, carried))));
        Assert.Single(AllNodes(Session(InBody(u, (outer, carried) => (Tensor<uint32>)OnnxOp.BitwiseXor(outer, carried + Scalar(1u)) + (Tensor<uint32>)OnnxOp.BitwiseXor(carried + Scalar(2u), outer)),
            KernelWorkaroundRegistry.OnnxRuntimeCuda)).Where(n => n.OpType == MAX).Select(n => n.Inputs[0]).Distinct());
        Assert.True(CudaMaxes(InBranch(u, outer => OnnxOp.BitwiseXor(outer, Scalar(5u)))) > 0);
        Assert.True(CudaMaxes(InNestedBody(u, outer => OnnxOp.BitwiseXor(outer, Scalar(5u)))) > 0);
        Assert.Contains(AllNodes(Optimized(Session(InBody(u, (outer, carried) => OnnxOp.BitwiseXor(outer, Scalar(5u))), KernelWorkaroundRegistry.OnnxRuntimeCuda))), n => n.OpType == MAX);
        Assert.Equal(0, CudaMaxes(Graph(u, OnnxOp.BitwiseXor(u, u))));
        Assert.Equal(0, AllNodes(Session(InBody(u, (outer, carried) => OnnxOp.BitwiseXor(outer, carried)), KernelWorkaroundRegistry.OnnxRuntime)).Count(n => n.OpType == MAX));
    }

    [Fact]
    public void TestTheCudaSetComputesWhatTheCallsItRewritesCompute()
    {
        var u = InputTensor<uint32>("u", rank: 1);
        var w = InputTensor<uint64>("w", rank: 1);
        var us = TensorData([3L], [1u, 6u, 0xFFFFFFFFu]);
        var ws = TensorData([3L], [1UL, 7UL, 0xFFFFFFFFFFFFFFFFUL]);
        Assert.True(AsWrittenOnCuda(InBody(u, (outer, carried) => OnnxOp.BitShift(outer, Scalar(1u), BitShiftDirection.Right)), us));
        Assert.True(AsWrittenOnCuda(InBody(u, (outer, carried) => OnnxOp.BitShift(carried, Scalar(3u), BitShiftDirection.Left)), us));
        Assert.True(AsWrittenOnCuda(InBody(u, (outer, carried) => OnnxOp.BitwiseXor(outer, carried)), us));
        Assert.True(AsWrittenOnCuda(InBody(u, (outer, carried) => OnnxOp.BitwiseNot(outer)), us));
        Assert.True(AsWrittenOnCuda(InBody(w, (outer, carried) => OnnxOp.BitwiseOr(carried, Scalar(8UL))), ws));
        Assert.True(AsWrittenOnCuda(InBody(w, (outer, carried) => OnnxOp.Equal(outer, carried)), ws));
    }

    [Fact]
    public void TestAGraphCompiledForAOneShotProfileIsBuiltWithTheBackendsWorkarounds()
    {
        var c = InputTensor<bit>("c", rank: 1);
        var x = InputTensor<uint64>("x", rank: 1);
        var g = new InternalComputationGraph([c, x], [OnnxOp.Where(c, x, OnnxOp.Add(x, x))]);
        using var context = new ComputeContext();
        using var compiled = context.Compile(g, ShorokooGraphOptimization.DisableAll);
        var selected = compiled.Execute(TensorData([2L], [true, false]).Shared(), TensorData([2L], [3UL, 5UL]).Shared())[0].ToTensorData();
        Assert.Equal([3UL, 10UL], selected.As<uint64>().CopyMemory<ulong>());
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
        Assert.Equal(Ifs(Session(step, null)), Ifs(FastOnnxModelBuilder.BuildInternalOnnxModel(step, prepForOnnx: true, inputDims: dims, workarounds: KernelWorkaroundRegistry.OnnxRuntime)));
        Assert.Equal(0, Ifs(Optimized(FastOnnxModelBuilder.BuildInternalOnnxModel(step, prepForOnnx: true, inputDims: dims, workarounds: KernelWorkaroundRegistry.OnnxRuntime))));
    }

    [Fact]
    public void TestASessionModelHoldsNoConstantNothingReads()
    {
        var x = InputTensor<float32>("x", rank: 1);
        var x23 = TensorData(DType.Float32, [2L, 3L], 1f, 2f, 3f, 4f, 5f, 6f);
        IEnumerable<string> Noop(ComputationGraph module)
            => UnreadConstants(FastOnnxModelBuilder.BuildInternalOnnxModel(Concrete(module, x23), prepForOnnx: true, inputDims: [[2L, 3L]], workarounds: KernelWorkaroundRegistry.OnnxRuntime));
        Assert.Equal<string>([], [
            .. Noop(NoopReduceOfEachElementCheck.ComputationGraph),
            .. Noop(NoopReduceOfEachElementByShapeCheck.ComputationGraph),
            .. Noop(NoopReduceAxesFormsCheck.ComputationGraph),
            .. UnreadConstants(Session(Graph(x, OnnxOp.Mul(OnnxOp.Add(x, Scalar(1f)), OnnxOp.Add(x, Scalar(2f)))), AddConstantSet))]);
    }

    [Fact]
    public void TestANoopReductionComputesItsGroupsOnOnnxRuntimeWithConcreteShapes()
    {
        var x = TensorData(DType.Float32, [2L, 3L], 1f, 2f, 3f, 4f, 5f, 6f);
        Assert.True(AutoTest.AllTrueWithConcreteShapes(NoopReduceOfEachElementCheck.ComputationGraph, x));
        Assert.True(AutoTest.AllTrueWithConcreteShapes(NoopReduceOfEachElementByShapeCheck.ComputationGraph, x));
        Assert.True(AutoTest.AllTrueWithConcreteShapes(NoopReduceAxesFormsCheck.ComputationGraph, x));
    }

    [Fact]
    public void TestASamePoolStridedPastItsKernelPoolsAtItsStrideAndCropsOnlyWhereTheSpecDoes()
    {
        var x = InputTensor<float32>("x", rank: 4);
        var g = Graph(x, OnnxOp.MaxPool(x, AutoPad.SameUpper, null, null, [1L, 1L], null, null, [2L, 2L]));
        Assert.All(AllNodes(Session(g, KernelWorkaroundRegistry.OnnxRuntime)).Where(n => n.OpType == MAX_POOL),
            n => Assert.Equal([2L, 2L], n.Attributes.Single(a => a.Name == "strides").Ints.ToArray()));
        Assert.DoesNotContain(AllNodes(Optimized(FastOnnxModelBuilder.BuildInternalOnnxModel(g, prepForOnnx: true,
            inputDims: [[1L, 1L, 5L, 7L]], workarounds: KernelWorkaroundRegistry.OnnxRuntime))), n => n.OpType == SLICE);
    }

    [Fact]
    public void TestACropAndResizeItsConstantsDecideTakesNoIf()
    {
        var y = InputTensor<float32>("y", rank: 4);
        Variable Crop(Vector<float32> scales, ResizeMode mode) => OnnxOp.Resize(y, Vector(0f, 0f, 0.1f, 0.2f, 1f, 1f, 0.9f, 0.8f), scales,
            null, null, null, CoordinateTransformationMode.Tf_crop_and_resize, null, null, -1f, null, mode, null);
        Assert.Equal(0, Ifs(Session(Graph(y, Crop(Vector(1f, 1f, 2f, 0.5f), ResizeMode.Linear)), KernelWorkaroundRegistry.OnnxRuntime)));
        Assert.Equal(0, Ifs(Session(Graph(y, Crop(Vector(1f, 1f, 1f, 2f), ResizeMode.Linear)), KernelWorkaroundRegistry.OnnxRuntime)));
    }

    [Fact]
    public void TestACubicCropAndResizeRegroupsItsInputWithinItsIf()
    {
        var y = InputTensor<float32>("y", rank: 4);
        var cubic = Session(Graph(y, OnnxOp.Resize(y, Vector(0f, 0.1f, 0f, 0f, 1f, 0.9f, 1f, 1f),
            OnnxOp.Add(Vector(1f, 2f, 1f, 1f), OnnxOp.Mul(OnnxOp.Cast(OnnxOp.Shape(y), null, DType.Float32), Scalar(0f))),
            null, null, null, CoordinateTransformationMode.Tf_crop_and_resize, null, null, -1f, null, ResizeMode.Cubic, null)),
            KernelWorkaroundRegistry.OnnxRuntime);
        Assert.DoesNotContain(cubic.Graph.Nodes, n => n.OpType == TRANSPOSE);
        Assert.Contains(AllNodes(cubic), n => n.OpType == TRANSPOSE);
    }

    [Fact]
    public void TestACubicCropAndResizeOfAnUnrankedInputRunsAtRankTwo()
    {
        var x = InputTensor<float32>("x");
        var zeros = OnnxOp.Mul(OnnxOp.Cast(OnnxOp.Shape(x), null, DType.Float32), Scalar(0f));
        var roi = OnnxOp.Add(Vector(0.1f, 0.2f, 0.9f, 0.8f), OnnxOp.Concat([zeros, zeros], 0L));
        var g = Graph(x, OnnxOp.Resize(x, roi, OnnxOp.Add(Vector(1.5f, 1.5f), zeros), null, null, null,
            CoordinateTransformationMode.Tf_crop_and_resize, null, null, -1f, null, ResizeMode.Cubic, null));
        var data = TensorData(DType.Float32, [4L, 5L], [.. Enumerable.Range(0, 20).Select(i => (float)(i % 7))]);
        Assert.Equal(Run(g, Session(g, null), [data]), Run(g, Session(g, KernelWorkaroundRegistry.OnnxRuntime), [data]));
    }

    [Fact]
    public void TestTheMatMulWorkaroundFiresWithConcreteShapesOnlyWhereABatchedOrVectorProductsOperandMayBeEmpty()
    {
        var v = InputTensor<float32>("v", rank: 1);
        var m = InputTensor<float32>("m", rank: 2);
        var b = InputTensor<float32>("b", rank: 3);
        var u = InputTensor<float32>("u");
        var i = InputTensor<int64>("i", rank: 3);
        var s = InputTensor<int64>("s", rank: 1);
        Assert.True(AsWritten(new([m], [OnnxOp.MatMul(m, m)]), [2L, 2L]));
        Assert.True(AsWritten(new([v, m], [OnnxOp.MatMul(v, m)]), [2L], [2L, 3L]));
        Assert.True(AsWritten(new([v], [OnnxOp.MatMul(v, v)]), [2L]));
        Assert.True(AsWritten(new([m, v], [OnnxOp.MatMul(m, v)]), [3L, 2L], [2L]));
        Assert.True(AsWritten(new([b], [OnnxOp.MatMul(b, b)]), [2L, 2L, 2L]));
        Assert.True(AsWritten(new([m, b], [OnnxOp.MatMul(m, b)]), [3L, 2L], [4L, 2L, 5L]));
        Assert.True(AsWritten(new([u, m], [OnnxOp.MatMul(m, u)]), [2L, 2L], [2L, 2L]));
        Assert.True(AsWritten(new([i], [OnnxOp.MatMul(i, i)]), [2L, 2L, 2L]));
        Assert.False(AsWritten(new([b, m], [OnnxOp.MatMul(b, m)]), [2L, 3L, 0L], [0L, 4L]));
        Assert.False(AsWritten(new([b, s], [OnnxOp.MatMul(b, OnnxOp.Reshape(b, s, allowZero: false))]), [2L, 2L, 2L], [3L]));
        Assert.False(AsWritten(MatMulOfABranchesOperand(), [2L, 2L, 2L], [1L], [1L]));
        Assert.False(AsWritten(MatMulOfAnExpandedSlice(), [1L, 1L, 4L], [1L]));
        Assert.False(AsWritten(Concrete(KernelWorkaroundMatMulOfALoopsCarry.ComputationGraph, Eights, TensorData(DType.Float32, [1L], 0f)), [2L, 2L, 2L], [1L]));
    }

    [Fact]
    public void TestAMatMulOfAnOperandEmptyOnlyAtRunTimeGivesZerosOnOnnxRuntimeWithConcreteShapes()
    {
        TensorData[] branch = [Eights, TensorData(DType.Float32, [1L], -1f), TensorData(DType.Int64, [1L], 0L)];
        TensorData[] loop = [Eights, TensorData(DType.Float32, [1L], 0f)];
        var looped = Concrete(KernelWorkaroundMatMulOfALoopsCarry.ComputationGraph, loop);
        Assert.Equal(new byte[32], Run(MatMulOfABranchesOperand(), AtStatedDims(MatMulOfABranchesOperand(), branch), branch).Single());
        Assert.Equal(new byte[32], Run(looped, AtStatedDims(looped, loop), loop).Single());
        TensorData[] expanded = [TensorData(DType.Float32, [1L, 1L, 4L], 1f, 2f, 3f, 4f), TensorData(DType.Int64, [1L], 0L)];
        Assert.Equal(new byte[32], Run(MatMulOfAnExpandedSlice(), AtStatedDims(MatMulOfAnExpandedSlice(), expanded), expanded).Single());
    }

    private static readonly TensorData Eights = TensorData(DType.Float32, [2L, 2L, 2L], 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f);

    private static InternalComputationGraph MatMulOfABranchesOperand()
    {
        var (b, x, e) = (InputTensor<float32>("b", rank: 3), InputTensor<float32>("x", rank: 1), InputTensor<int64>("e", rank: 1));
        var m = Shorokoo.Core.Nodes.Ops.IfElse((Scalar<bit>)OnnxOp.Less(OnnxOp.ReduceMin(x, keepdims: false), Scalar(0f)),
            OnnxOp.Slice(b, Vector(0L), e, Vector(2L)), OnnxOp.Identity(b, null));
        return new([b, x, e], [OnnxOp.MatMul(m, OnnxOp.Transpose(m, [0L, 2L, 1L]))]);
    }

    private static InternalComputationGraph MatMulOfAnExpandedSlice()
    {
        var (b, e) = (InputTensor<float32>("b", rank: 3), InputTensor<int64>("e", rank: 1));
        var m = OnnxOp.Expand(OnnxOp.Slice(b, Vector(0L), e, Vector(2L)), Vector(2L, 2L, 1L));
        return new([b, e], [OnnxOp.MatMul(m, OnnxOp.Transpose(m, [0L, 2L, 1L]))]);
    }

    private static ModelProto AtStatedDims(InternalComputationGraph g, TensorData[] x)
        => FastOnnxModelBuilder.BuildInternalOnnxModel(g, prepForOnnx: true, inputDims: [.. x.Select(t => t.Shape.Dims)], workarounds: KernelWorkaroundRegistry.OnnxRuntime);

    [Fact]
    public void TestTheMatMulWorkaroundFiresWithShapesKnownOnlyAtRunTimeOnlyForAnEmptyConstantOperand()
    {
        var m = InputTensor<float32>("m", rank: 2);
        var b = InputTensor<float32>("b", rank: 3);
        var u = InputTensor<float32>("u");
        Assert.True(AsWritten(new([b], [OnnxOp.MatMul(b, b)])));
        Assert.True(AsWritten(new([b, m], [OnnxOp.MatMul(b, m)])));
        Assert.True(AsWritten(new([u, m], [OnnxOp.MatMul(m, u)])));
        Assert.True(AsWritten(new([b], [OnnxOp.MatMul(b, Constant([2L, 1L], 5f, 6f))])));
        Assert.True(AsWritten(new([], [OnnxOp.MatMul(Constant([2L, 1L, 2L], 1f, 2f, 3f, 4f), Constant([2L, 1L], 5f, 6f))]), []));
        Assert.False(AsWritten(new([b], [OnnxOp.MatMul(b, Constant([2L, 0L, 4L]))])));
        Assert.False(AsWritten(new([u], [OnnxOp.MatMul(Constant([0L, 3L]), u)])));
        Assert.Equal(0, Ifs(Session(new([b], [OnnxOp.MatMul(b, Constant([2L, 0L, 4L]))]), KernelWorkaroundRegistry.OnnxRuntime)));
    }

    [Fact]
    public void TestAMatMulWithAnEmptyOperandGivesZerosOfTheProductsShapeOrAnEmptyProductOnOnnxRuntimeWithConcreteShapes()
    {
        var a = TensorData(DType.Float32, [], 2f);
        Assert.True(AutoTest.AllTrueWithConcreteShapes(EmptyMatMulTransposedCheck.ComputationGraph, a));
        Assert.True(AutoTest.AllTrueWithConcreteShapes(EmptyMatMulUntransposedCheck.ComputationGraph, a));
        Assert.True(AutoTest.AllTrueWithConcreteShapes(EmptyMatMulOfUnknownRankCheck.ComputationGraph, a));
        Assert.True(AutoTest.AdvancedTestGraph<EmptyMatMulOfAnEmptyConstantCheck>([], [a]));
    }

    [Fact]
    public void TestOnnxRuntimesMatMulIsAcceptedForAnEmptyBatchAgainstNoneOrOneAndForALeftBatchOfOneWithoutStatedShapes()
    {
        Assert.Equal([1L, 3L, 4L], ProductDims([1L, 3L, 0L], [2L, 0L, 4L], concrete: false));
        Assert.Equal([2L, 3L, 4L], ProductDims([1L, 3L, 0L], [2L, 0L, 4L], concrete: true));
        Assert.Throws<Microsoft.ML.OnnxRuntime.OnnxRuntimeException>(() => ProductDims([4L, 2L], [0L, 2L, 2L], concrete: true));
        Assert.Throws<Microsoft.ML.OnnxRuntime.OnnxRuntimeException>(() => ProductDims([2L], [0L, 2L, 3L], concrete: true));
        Assert.Throws<Microsoft.ML.OnnxRuntime.OnnxRuntimeException>(() => ProductDims([0L, 2L, 3L], [3L], concrete: true));
        Assert.Throws<Microsoft.ML.OnnxRuntime.OnnxRuntimeException>(() => ProductDims([0L, 2L, 3L], [3L], concrete: false));
        Assert.True(AutoTest.AllTrueWithConcreteShapes(EmptyMatMulOfALeftBatchOfOneCheck.ComputationGraph, TensorData(DType.Float32, [], 2f)));
    }

    private static long[] ProductDims(long[] left, long[] right, bool concrete)
    {
        var (a, b) = (InputTensor<float32>("a", rank: left.Length), InputTensor<float32>("b", rank: right.Length));
        TensorData Ones(long[] dims) => TensorData(DType.Float32, dims, [.. Enumerable.Repeat((object)1f, (int)dims.Aggregate(1L, (x, y) => x * y))]);
        using var context = new ComputeContext();
        using var compiled = context.Compile(new InternalComputationGraph([a, b], [OnnxOp.MatMul(a, b)]), concrete ? [left, right] : [null, null], trainingStep: false);
        return compiled.Execute(Ones(left).Shared(), Ones(right).Shared())[0].ToTensorData().Shape.Dims;
    }

    [Fact]
    public void TestWorkaroundsOverValuesAShapeAlsoReadsBuildAndComputeOnOnnxRuntimeWithConcreteShapes()
    {
        Assert.True(AutoTest.AllTrueWithConcreteShapes(ChainedWorkaroundsReadByAShapeCheck.ComputationGraph,
            TensorData(DType.Float32, [1L, 1L, 5L, 6L], [.. Enumerable.Range(0, 30).Select(i => (object)(float)(i % 5 - 1))])));
        var a = InputTensor<float32>("a", rank: 2);
        var r = OnnxOp.Relu(a);
        Variable By(Variable v) => OnnxOp.Reshape(a, OnnxOp.Concat([Vector(-1L), OnnxOp.Shape(v, start: 1L)], 0L), true);
        var (ints, halves) = (OnnxOp.Cast(r, null, DType.Int64), OnnxOp.Cast(r, null, DType.Float16));
        var data = TensorData(DType.Float32, [3L, 4L], [.. Enumerable.Range(0, 12).Select(i => (object)(float)(i % 5 - 1))]);
        Assert.True(AsComputedWithoutConcreteShapes(Graph(a, OnnxOp.Add(By(ints), OnnxOp.Cast(OnnxOp.ReduceMax(ints, Vector(1L), true, null), null, DType.Float32))), data));
        Assert.True(AsComputedWithoutConcreteShapes(Graph(a, OnnxOp.Add(By(halves), OnnxOp.Cast(OnnxOp.ReduceSumSquare(halves, null, true, null), null, DType.Float32))), data));
    }

    [Fact]
    public void TestANoopReductionOverATensorOfOnesOfAnInputsShapeBuildsItsSessionWithConcreteShapes()
    {
        var a = InputTensor<float32>("a", rank: 2);
        var ones = (Tensor<float32>)OnnxOp.Expand(Scalar(1f), OnnxOp.Shape(a));
        var data = TensorData(DType.Float32, [40L, 3L], [.. Enumerable.Range(0, 120).Select(i => (object)(float)i)]);
        Assert.True(AsComputedWithoutConcreteShapes(Graph(a, OnnxOp.Add(a, NN.Reduce(ReduceKind.Sum, ones, null, keepDims: true, noOp: true))), data));
    }

    [Fact]
    public void TestAMatMulWhoseProductsShapeALaterBranchReadsBuildsItsSessionOnOnnxRuntimeWithConcreteShapes()
    {
        var (a, w, g) = (InputTensor<float32>("a", rank: 3), InputTensor<float32>("w", rank: 3), InputTensor<float32>("g", rank: 3));
        var (k, c) = (InputTensor<int64>("k", rank: 1), InputScalar<bit>("c"));
        var t = OnnxOp.Neg(w);
        var gy = OnnxOp.Reshape(g, OnnxOp.Shape(OnnxOp.MatMul(a, t)), allowZero: true);
        var at = OnnxOp.Transpose(OnnxOp.Reshape(a, OnnxOp.Concat([k, OnnxOp.Shape(t, start: 0L, end: 1L)], 0), allowZero: true), [0L, 2L, 1L]);
        var graph = new InternalComputationGraph([a, w, g, k, c], [Shorokoo.Core.Nodes.Ops.IfElse(c, OnnxOp.Neg(gy), OnnxOp.MatMul(at, gy))]);
        using var context = new ComputeContext();
        using var compiled = context.Compile(graph, [[2L, 2L, 8L], [2L, 8L, 8L], [2L, 2L, 8L], [2L], []], trainingStep: false);
        Assert.Equal(1, compiled.OutputCount);
    }

    [Fact]
    public void TestAnAttentionTrainingStepKeepsEveryMatMulAsWrittenWithOrWithoutConcreteShapes()
    {
        var step = AttentionTrainingStep();
        List<long[]?> dims = [.. step.InputNodes.Select(RepresentativeInputShapes.Get)];
        ModelProto Built(KernelWorkaroundSet set) => FastOnnxModelBuilder.BuildInternalOnnxModel(step, prepForOnnx: true, inputDims: dims, workarounds: set);
        Assert.Equal(Bytes(Built(WithoutTheMatMulWorkaround)), Bytes(Built(KernelWorkaroundRegistry.OnnxRuntime)));
        var shapes = FastApplyKernelWorkarounds.ConcreteShapes(step, KernelWorkaroundRegistry.OnnxRuntime, dims);
        Assert.NotNull(shapes);
        Assert.True(step.Nodes.Where(n => n.OpCode == MATMUL).SelectMany(n => n.Inputs).All(k => shapes.ContainsKey(k!.Value)));
    }

    [Fact]
    public void TestOnnxRuntimeFoldsAwayTheIfOfTheMatMulWorkaroundWhereTheShapesFollowFromTheStatedDimensions()
    {
        var (b, m) = (InputTensor<float32>("b", rank: 3), InputTensor<float32>("m", rank: 2));
        var built = FastOnnxModelBuilder.BuildInternalOnnxModel(new([b, m], [OnnxOp.MatMul(b, m)]), prepForOnnx: true,
            inputDims: [[2L, 3L, 0L], [0L, 4L]], workarounds: KernelWorkaroundRegistry.OnnxRuntime);
        Assert.Equal(1, Ifs(built));
        Assert.Equal(0, Ifs(Optimized(built)));
    }

    [Fact]
    public void TestAParameterAWorkaroundReadsIsWrittenOverWhereItsIfFolds()
    {
        var x = InputTensor<float32>("x", rank: 3);
        var w = InputTensor<float32>("w", rank: 3);
        var m = InputTensor<float32>("m", rank: 2);
        var s = InputTensor<int64>("s", rank: 1);
        var batch = TensorData(DType.Float32, [2L, 3L, 3L], [.. Enumerable.Range(0, 18).Select(i => (object)(i / 8f))]);
        var matrix = TensorData(DType.Float32, [3L, 3L], [.. Enumerable.Range(0, 9).Select(i => (object)(i / 4f))]);
        var dims = TensorData(DType.Int64, [3L], 2L, 3L, 3L);
        Assert.Equal(1, AliasedOverTheFirstInput(new([w, x], [OnnxOp.Sub(w, OnnxOp.ReduceMean(OnnxOp.MatMul(x, w), keepdims: false))]), batch, batch));
        Assert.Equal(1, AliasedOverTheFirstInput(new([m, x], [OnnxOp.Sub(m, OnnxOp.ReduceMean(OnnxOp.MatMul(x, m), keepdims: false))]), matrix, batch));
        Assert.Equal(1, AliasedOverTheFirstInput(new([w, x, s], [OnnxOp.Sub(w, OnnxOp.ReduceMean(OnnxOp.MatMul(OnnxOp.Reshape(x, s, allowZero: false), w), keepdims: false))]), batch, batch, dims));
    }

    private static long AliasedOverTheFirstInput(InternalComputationGraph g, TensorData first, params TensorData[] rest)
    {
        byte[] Result(ComputeContext context, TensorData consumed)
        {
            using var compiled = context.Compile(g, [first.Shape.Dims, .. rest.Select(t => t.Shape.Dims)], trainingStep: false, aliasCandidates: [(0, 0)]);
            return compiled.Execute([consumed, .. rest.Select(t => (IData)t.Shared())])[0].ToTensorData().AccessRawMemory().ToArray();
        }
        using var unaliased = new ComputeContext { OutputAliasing = false };
        using var context = new ComputeContext();
        Assert.Equal(Result(unaliased, first.CopyTo(ComputeContext.Host)), Result(context, first.CopyTo(ComputeContext.Host)));
        return context.AliasedOutputs;
    }

    [Fact]
    public void TestAnIfOnnxRuntimeFoldsToABranchHoldingAConstantOfAHundredAndTwentyEightBytesBuildsItsSession()
    {
        var x = InputTensor<float32>("x", rank: 1);
        var g = Graph(x, Shorokoo.Core.Nodes.Ops.IfElse((Scalar<bit>)OnnxOp.Equal(OnnxOp.ReduceProd(OnnxOp.Shape(x), keepdims: false), Scalar(32L)),
            OnnxOp.Expand(Scalar(0f), Vector(32L)), x));
        var ones = TensorData(DType.Float32, [32L], [.. Enumerable.Repeat((object)1f, 32)]);
        using var context = new ComputeContext();
        using var concrete = context.Compile(g, [[32L]], trainingStep: false);
        Assert.Equal(new byte[128], concrete.Execute(ones.Shared())[0].ToTensorData().AccessRawMemory().ToArray());
        using var symbolic = context.Compile(g, [null], trainingStep: false);
        Assert.Equal(new byte[128], symbolic.Execute(ones.Shared())[0].ToTensorData().AccessRawMemory().ToArray());
    }

    [Fact]
    public void TestAWorkaroundIfOnnxRuntimeFoldsToItsEmptySideBuildsItsSession()
    {
        var i = InputTensor<int32>("i", rank: 2);
        var b = InputTensor<float32>("b", rank: 3);
        byte[] Run(InternalComputationGraph g, TensorData data)
        {
            using var context = new ComputeContext();
            using var compiled = context.Compile(g, [data.Shape.Dims], trainingStep: false);
            return compiled.Execute(data.Shared())[0].ToTensorData().AccessRawMemory().ToArray();
        }
        Assert.Equal([.. Enumerable.Repeat((byte[])[0, 0, 0, 0x80], 40).SelectMany(e => e)], Run(Graph(i, i.Reduce(ReduceKind.Max, Vector(0L))), TensorData(DType.Int32, [0L, 40L], Array.Empty<object>())));
        Assert.Equal(new byte[512], Run(Graph(b, OnnxOp.MatMul(b, OnnxOp.Transpose(b, [0L, 2L, 1L]))), TensorData(DType.Float32, [2L, 8L, 0L], Array.Empty<object>())));
    }

    [Fact]
    public void TestAnIfSplicedIntoALoopBodyKeepsTheLoopInvariantShapeItsConditionAndBranchShareBeforeBoth()
    {
        var x = TensorData(DType.Float32, [3L], 2f, 3f, 4f);
        AssertRewritten(MulIfSet, "Loop", IF, false, Concrete(KernelWorkaroundMulByOuterInLoop.ComputationGraph, x), x);
    }

    [Fact]
    public void TestAWorkaroundThatDecidesByShapeBuildsNoIfInALoopBodyWithConcreteShapes()
    {
        var x = TensorData(DType.Float32, [2L, 2L, 2L], 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f);
        var g = Concrete(KernelWorkaroundMatMulAndNoopReduceInLoop.ComputationGraph, x);
        ModelProto Built(KernelWorkaroundSet? set) => FastOnnxModelBuilder.BuildInternalOnnxModel(g, prepForOnnx: true, inputDims: [x.Shape.Dims], workarounds: set);
        var (plain, rewritten) = (Built(null), Built(KernelWorkaroundRegistry.OnnxRuntime));
        Assert.Equal(0, Ifs(rewritten));
        Assert.NotEqual(Bytes(plain), Bytes(rewritten));
        Assert.Null(FastApplyKernelWorkarounds.ConcreteShapes(g, KernelWorkaroundRegistry.OnnxRuntime, [x.Shape.Dims]));
        Assert.Equal(Run(g, plain, [x]), Run(g, rewritten, [x]));
    }

    [Fact]
    public void TestAnArithmeticOpWithAnEmptyConstantOperandBroadcastsToAnEmptyResultOnOnnxRuntime()
    {
        var x = InputTensor<float32>("x", rank: 3);
        var data = TensorData(DType.Float32, [2L, 1L, 3L], 1f, 2f, 3f, 4f, 5f, 6f);
        long[] Broadcast(Func<Variable, Variable, Variable> op, long[] dims)
            => ComputeContext.Default.Execute(Graph(x, OnnxOp.Relu(op(OnnxOp.Relu(x), Constant(dims)))), data.Shared())[0].ToTensorData().Shape.Dims;
        Assert.Equal([2L, 0L, 3L], Broadcast(OnnxOp.Add, [2L, 0L, 1L]));
        Assert.Equal([2L, 0L, 3L], Broadcast(OnnxOp.Sub, [2L, 0L, 3L]));
        Assert.Equal([2L, 0L, 3L], Broadcast(OnnxOp.Mul, [0L, 1L]));
        Assert.Equal([2L, 0L, 3L], Broadcast(OnnxOp.Div, [1L, 0L, 1L]));
    }

    [Fact]
    public void TestAnArithmeticOpWithAnEmptyConstantOperandIsEmptyOnOnnxRuntime()
    {
        var x = InputTensor<float32>("x", rank: 3);
        var data = TensorData(DType.Float32, [2L, 1L, 3L], 1f, 2f, 3f, 4f, 5f, 6f);
        var empty = Constant([2L, 0L, 1L]);
        long[] Dims(Variable output) => ComputeContext.Default.Execute(Graph(x, output), data.Shared())[0].ToTensorData().Shape.Dims;
        Assert.Equal([2L, 0L, 3L], Dims(OnnxOp.Add(x, empty)));
        Assert.Equal([2L, 0L, 3L], Dims(OnnxOp.Relu(OnnxOp.Add(empty, OnnxOp.Relu(x)))));
        Assert.Equal([2L, 0L, 3L], Dims(OnnxOp.Relu(OnnxOp.Mul(empty, OnnxOp.Relu(x)))));
        Assert.Equal([2L, 0L, 3L], Dims(OnnxOp.Sub(x, empty)));
        Assert.Equal([2L, 0L, 3L], Dims(OnnxOp.Div(x, empty)));
        Assert.Equal([2L, 0L, 3L], Dims(OnnxOp.Cast(OnnxOp.Add(OnnxOp.Cast(x, null, DType.Int64), OnnxOp.Constant(TensorAttribute.Create(new Shape(2L, 0L, 1L), Array.Empty<long>()))), null, DType.Float32)));
        Assert.Equal([2L, 0L, 3L], Dims(OnnxOp.Relu(OnnxOp.Add(OnnxOp.Relu(x), OnnxOp.Add(Constant([1L, 1L, 3L], 1f, 2f, 3f), empty)))));
    }

    private static InternalComputationGraph Graph(Variable input, Variable output) => new([input], [output]);

    private static InternalComputationGraph InBody<T>(Tensor<T> x, Func<Tensor<T>, Tensor<T>, Variable> call) where T : IVarType
    {
        var carried = x;
        foreach (var ctx in LoopAPI.Iterate(Scalar(3L)))
            carried = (Tensor<T>)OnnxOp.Add(carried, OnnxOp.CastLike(call(x, carried), carried, null));
        return new([x], [carried]);
    }

    private static InternalComputationGraph InBranch(Tensor<uint32> x, Func<Tensor<uint32>, Variable> call)
    {
        Tensor<uint32> outer = x * Scalar(3u);
        return new([x], [Shorokoo.Core.Nodes.Ops.IfElse((Scalar<bit>)OnnxOp.Less(OnnxOp.ReduceMin(x, keepdims: false), Scalar(3u)),
            (Tensor<uint32>)call(outer), outer + Scalar(1u))]);
    }

    private static InternalComputationGraph InNestedBody(Tensor<uint32> x, Func<Tensor<uint32>, Variable> call)
    {
        var carried = x;
        foreach (var ctx in LoopAPI.Iterate(Scalar(2L)))
        {
            Tensor<uint32> outer = carried * Scalar(3u);
            var inner = carried;
            foreach (var innerCtx in LoopAPI.Iterate(Scalar(2L)))
                inner = inner + (Tensor<uint32>)call(outer);
            carried = inner;
        }
        return new([x], [carried]);
    }

    private static int CudaMaxes(InternalComputationGraph g)
        => AllNodes(Session(g, KernelWorkaroundRegistry.OnnxRuntimeCuda)).Count(n => n.OpType == MAX)
         - AllNodes(Session(g, null)).Count(n => n.OpType == MAX);

    private static bool AsWrittenOnCuda(InternalComputationGraph g, TensorData x)
        => Run(g, Session(g, null), [x]).Zip(Run(g, Session(g, KernelWorkaroundRegistry.OnnxRuntimeCuda), [x])).All(p => p.First.SequenceEqual(p.Second));

    private static bool AsWritten(InternalComputationGraph g)
        => Bytes(Session(g, null)).SequenceEqual(Bytes(Session(g, KernelWorkaroundRegistry.OnnxRuntime)));

    private static bool AsWritten(InternalComputationGraph g, params long[][] dims)
    {
        ModelProto Built(KernelWorkaroundSet? set) => FastOnnxModelBuilder.BuildInternalOnnxModel(g, prepForOnnx: true, inputDims: dims, workarounds: set);
        return Bytes(Built(null)).SequenceEqual(Bytes(Built(KernelWorkaroundRegistry.OnnxRuntime)));
    }

    private static readonly KernelWorkaroundSet WithoutTheMatMulWorkaround = new("without-matmul",
        [.. KernelWorkaroundRegistry.OnnxRuntime.Workarounds.Where(w => w is not Shorokoo.Core.Lowering.KernelWorkarounds.OnnxRuntime.MatMulEmptyOperandWorkaround)]);

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

    private static InternalComputationGraph AttentionTrainingStep()
        => TrainingRig.FromScratch(SdpaMeanPoolModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            [new TensorDataModelParam("input", ModelParamType.InputParam, TensorData([1L, 2L, 8L, 4L], TransformerTrainingFixtures.Floats(64, seed: 0.05f)))],
            0.01f).TrainingStepPureGraph.ToInternal();

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

    private static InternalComputationGraph Concrete(ComputationGraph module, params TensorData[] x)
        => module.ToInternal().ToConcreteArchitecture(x).ToConcreteModel();

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

    private static bool AsComputedWithoutConcreteShapes(InternalComputationGraph g, TensorData x)
        => Run(g, Session(g, KernelWorkaroundRegistry.OnnxRuntime), [x]).Single().SequenceEqual(Run(g,
            FastOnnxModelBuilder.BuildInternalOnnxModel(g, prepForOnnx: true, inputDims: [x.Shape.Dims], workarounds: KernelWorkaroundRegistry.OnnxRuntime), [x]).Single());

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

    private static IEnumerable<string> UnreadConstants(ModelProto model)
    {
        var read = AllNodes(model).SelectMany(n => n.Inputs)
            .Concat(AllNodes(model).SelectMany(n => n.Attributes).Where(a => a.G is not null).SelectMany(a => a.G!.Outputs.Select(o => o.Name)))
            .Concat(model.Graph.Outputs.Select(o => o.Name)).Concat(model.Functions.SelectMany(f => f.Outputs)).ToHashSet();
        return AllNodes(model).Where(n => n.OpType == CONSTANT && !read.Contains(n.Outputs[0])).Select(n => n.Outputs[0]);
    }

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

    private sealed class MulAsIfOnItsRightShape : KernelWorkaround
    {
        public override IReadOnlySet<string> OpCodes { get; } = new HashSet<string>([MUL]);

        public override bool Applies(WorkaroundSite site) => site.DTypeOf(0) == DType.Float32;

        public override Variable?[] Rewrite(WorkaroundSite site, Variable?[] inputs)
        {
            var shape = OnnxOp.Shape(inputs[1]!);
            return [Shorokoo.Core.Nodes.Ops.IfElse((Scalar<bit>)OnnxOp.Equal(OnnxOp.ReduceProd(shape, keepdims: false), Scalar(0L)),
                OnnxOp.Expand(Scalar(0f), OnnxOp.Slice(shape, Vector(0L), Vector(1L))), OnnxOp.Mul(inputs[0]!, inputs[1]!))];
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
public partial class KernelWorkaroundMulByOuterInLoop
{
    public static Tensor<float32> Inline(Tensor<float32> x)
    {
        var a = x;
        foreach (var ctx in LoopAPI.Iterate(x.Reduce(ReduceKind.Min, keepDims: false).Scalar().Cast<int64>()))
            a = (Tensor<float32>)OnnxOp.Mul(a, x);
        return a;
    }
}

[Module]
public partial class KernelWorkaroundMatMulOfALoopsCarry
{
    public static Tensor<float32> Inline(Tensor<float32> b, Tensor<float32> x)
    {
        var m = (Tensor<float32>)OnnxOp.Slice(b, Vector(0L), Vector(0L), Vector(2L));
        foreach (var ctx in LoopAPI.Iterate(x.Reduce(ReduceKind.Min, keepDims: false).Scalar().Cast<int64>()))
            m = b + m.Reduce(ReduceKind.Sum, keepDims: false);
        return (Tensor<float32>)OnnxOp.MatMul(m, OnnxOp.Transpose(m, [0L, 2L, 1L]));
    }
}

[Module]
public partial class KernelWorkaroundMatMulAndNoopReduceInLoop
{
    public static Tensor<float32> Inline(Tensor<float32> x)
    {
        var a = x;
        foreach (var ctx in LoopAPI.Iterate(x.Reduce(ReduceKind.Min, keepDims: false).Scalar().Cast<int64>()))
            a = (Tensor<float32>)OnnxOp.ReduceSum(OnnxOp.MatMul(a, a), null, true, true);
        return a;
    }
}

[Module]
public partial class KernelWorkaroundRandomNormalInLoop
{
    public static Tensor<float32> Inline(Tensor<float32> x)
    {
        var a = x;
        foreach (var ctx in LoopAPI.Iterate(Scalar(3L)))
            a = a + RandomNormal((Vector<int64>)OnnxOp.Shape(a), Scalar(0f), Scalar(1f));
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

internal static class EmptyMatMul
{
    internal static Tensor<float32> Of(Scalar<float32> a, long[] dims, int? rank = null)
    {
        var t = (Tensor<float32>)OnnxOp.Expand(a, Vector(dims));
        return rank is { } r ? (Tensor<float32>)OnnxOp.Identity(t, rank: r) : t;
    }

    internal static Scalar<bit> Is(Variable product, params long[] dims)
    {
        var p = (Tensor<float32>)product;
        var shapeGap = ((Tensor<int64>)OnnxOp.Shape(p) - Vector(dims)).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return (shapeGap == Scalar(0L)) & (p.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar() == Scalar(0f));
    }
}

[Module]
public partial class EmptyMatMulTransposedCheck
{
    public static Scalar<bit> Inline(Scalar<float32> a)
    {
        Tensor<float32> T(long[] dims) => EmptyMatMul.Of(a, dims, dims.Length);
        return EmptyMatMul.Is(OnnxOp.MatMul(T([2L, 0L, 3L]).Transpose(0L, 2L, 1L), T([2L, 0L, 4L])), 2L, 3L, 4L)
            & EmptyMatMul.Is(OnnxOp.MatMul(T([2L, 3L, 0L]), T([2L, 4L, 0L]).Transpose(0L, 2L, 1L)), 2L, 3L, 4L)
            & EmptyMatMul.Is(OnnxOp.MatMul(T([2L, 0L, 3L]).Transpose(0L, 2L, 1L), T([2L, 4L, 0L]).Transpose(0L, 2L, 1L)), 2L, 3L, 4L)
            & EmptyMatMul.Is(OnnxOp.MatMul(T([1L, 2L, 0L, 3L]).Transpose(0L, 1L, 3L, 2L), T([2L, 1L, 0L, 4L])), 2L, 2L, 3L, 4L)
            & EmptyMatMul.Is(OnnxOp.MatMul(T([0L, 3L]).Transpose(1L, 0L), T([2L, 0L, 4L])), 2L, 3L, 4L)
            & EmptyMatMul.Is(OnnxOp.MatMul(T([2L, 0L, 3L]).Transpose(0L, 2L, 1L), T([0L, 4L])), 2L, 3L, 4L);
    }
}

[Module]
public partial class EmptyMatMulUntransposedCheck
{
    public static Scalar<bit> Inline(Scalar<float32> a)
    {
        Tensor<float32> T(long[] dims) => EmptyMatMul.Of(a, dims, dims.Length);
        return EmptyMatMul.Is(OnnxOp.MatMul(T([3L, 0L]), T([2L, 0L, 4L])), 2L, 3L, 4L)
            & EmptyMatMul.Is(OnnxOp.MatMul(T([2L, 3L, 0L]), T([0L])), 2L, 3L)
            & EmptyMatMul.Is(OnnxOp.MatMul(T([3L, 0L]), T([0L])), 3L)
            & EmptyMatMul.Is(OnnxOp.Cast(OnnxOp.MatMul(OnnxOp.Cast(T([3L, 0L]), null, DType.Int64), OnnxOp.Cast(T([2L, 0L, 4L]), null, DType.Int64)), null, DType.Float32), 2L, 3L, 4L);
    }
}

[Module]
public partial class EmptyMatMulOfALeftBatchOfOneCheck
{
    public static Scalar<bit> Inline(Scalar<float32> a)
        => EmptyMatMul.Is(OnnxOp.MatMul(EmptyMatMul.Of(a, [1L, 3L, 0L], 3), EmptyMatMul.Of(a, [2L, 0L, 4L], 3)), 2L, 3L, 4L);
}

[Module]
public partial class EmptyMatMulOfAnEmptyConstantCheck
{
    public static Scalar<bit> Inline(Scalar<float32> a)
    {
        Variable Empty(params long[] dims) => OnnxOp.Constant(TensorAttribute.Create(new Shape(dims), Array.Empty<float>()));
        return EmptyMatMul.Is(OnnxOp.MatMul(EmptyMatMul.Of(a, [2L, 3L, 0L], 3), Empty(0L, 4L)), 2L, 3L, 4L)
            & EmptyMatMul.Is(OnnxOp.MatMul(Empty(3L, 0L), EmptyMatMul.Of(a, [2L, 0L, 4L])), 2L, 3L, 4L)
            & EmptyMatMul.Is(OnnxOp.MatMul(EmptyMatMul.Of(a, [2L, 3L, 5L], 3), Empty(2L, 5L, 0L)), 2L, 3L, 0L);
    }
}

[Module]
public partial class EmptyMatMulOfUnknownRankCheck
{
    public static Scalar<bit> Inline(Scalar<float32> a)
    {
        Tensor<float32> T(long[] dims) => EmptyMatMul.Of(a, dims);
        return EmptyMatMul.Is(OnnxOp.MatMul(T([2L, 0L, 3L]).Transpose(0L, 2L, 1L), T([2L, 0L, 4L])), 2L, 3L, 4L)
            & EmptyMatMul.Is(OnnxOp.MatMul(T([2L, 3L, 0L]), T([2L, 4L, 0L]).Transpose(0L, 2L, 1L)), 2L, 3L, 4L)
            & EmptyMatMul.Is(OnnxOp.MatMul(T([1L, 2L, 0L, 3L]).Transpose(0L, 1L, 3L, 2L), T([2L, 1L, 0L, 4L])), 2L, 2L, 3L, 4L)
            & EmptyMatMul.Is(OnnxOp.MatMul(T([0L]), T([2L, 4L, 0L]).Transpose(0L, 2L, 1L)), 2L, 4L)
            & EmptyMatMul.Is(OnnxOp.MatMul(T([2L, 0L, 3L]).Transpose(0L, 2L, 1L), T([0L])), 2L, 3L)
            & EmptyMatMul.Is(OnnxOp.MatMul(T([0L]), T([0L])))
            & EmptyMatMul.Is(OnnxOp.MatMul(T([3L, 0L]), T([0L])), 3L);
    }
}

[Module]
public partial class ChainedWorkaroundsReadByAShapeCheck
{
    public static Tensor<bit> Inline(Tensor<float32> x)
    {
        var r = (Tensor<float32>)OnnxOp.Relu(x);
        var pooled = (Tensor<float32>)OnnxOp.MaxPool(r, AutoPad.SameUpper, null, null, [1L, 1L], null, null, [2L, 2L]);
        var strided = (Tensor<float32>)OnnxOp.Relu(OnnxOp.Slice(x, Vector(0L, 1L), Vector(5L, 6L), Vector(2L, 3L), Vector(2L, 2L)));
        var counts = pooled.Cast<int64>();
        var halves = pooled.Cast<float16>();
        var kept = (Tensor<float32>)OnnxOp.Neg(pooled);
        Tensor<bit> Regrouped(Variable v, long last)
            => ((Tensor<float32>)OnnxOp.Reshape(x, OnnxOp.Concat([Vector(-1L), OnnxOp.Shape(v, start: 3L)], 0L), true) == x.Reshape(Vector(-1L, last))).Reshape(Vector(-1L));
        return OnnxOp.Concat(
        [
            (pooled == strided).Reshape(Vector(-1L)),
            (counts.Reduce(ReduceKind.Max, Vector(2L, 3L)) == strided.Reduce(ReduceKind.Max, Vector(2L, 3L)).Cast<int64>()).Reshape(Vector(-1L)),
            (((Tensor<float16>)OnnxOp.ReduceSumSquare(halves, null, true, null)).Cast<float32>() == strided.Reduce(ReduceKind.SumSquare)).Reshape(Vector(-1L)),
            (NN.Reduce(ReduceKind.Sum, kept, null, keepDims: true, noOp: true) == -strided).Reshape(Vector(-1L)),
            Regrouped(r, 6L),
            Regrouped(counts, 3L),
            Regrouped(halves, 3L),
            Regrouped(kept, 3L),
        ], 0);
    }
}
