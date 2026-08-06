using Shorokoo.Core.Factory.CSharpFactory;
using Shorokoo.Core.Nodes.Processors.Helpers;
using System.Collections.Immutable;
using FrameworkOps = Shorokoo.Core.Nodes.Ops;

namespace Shorokoo.Tests;

/// <summary>
/// Drives <see cref="CSharpModelBuilder"/> over graph shapes that reach the
/// per-op / per-attribute / per-DType / per-keyword branches of the codegen dispatch.
/// </summary>
[Trait("Domain", "Factory")]
[Trait("Purpose", "Coverage")]
public class CSharpModelBuilderCoverageTests
{
    [Fact]
    public void TestBuildFullGraphModuleTensorStructSequenceAndConstantCodegen()
    {
        AssertCodegens(CallsHypersLayer.ComputationGraph.ToInternal(), "HypersLayer");
        AssertCodegens(TensorStructLoopCarry.ComputationGraph.ToInternal(),
            "InternalOp.TensorStructCreate", "InternalOp.TensorStructGetField");
        AssertCodegens(SequenceOpsOnStructs.ComputationGraph.ToInternal());
        AssertCodegens(BuildConstantBranchesGraph(),
            "1.5d", "6UL", "true", "(short[])", "(ushort[])", "(uint[])", "EmptyVector<int32>");
        AssertCodegens(BatchNormWithStateUpdate.ComputationGraph.ToInternal(),
            "[StateInitializer]", "isTrainable: false");
        AssertCodegens(BuildLowOpInlinedGraph(), "(", ")");
        AssertCodegens(BuildBigConstantGraph(), "MakeTensor<");
        AssertCodegens(BuildSequenceRankInferGraph());
        AssertCodegens(BuildScanLoopGraph(), ".Scan(", ".ContinueWhile(");
    }

    [Fact]
    public void TestCodegenConstantFallbackIfElseArrayPoolRanksAndDeepSequenceRankInference()
    {
        AssertCodegens(BuildUnsupportedConstantDtypeGraph(), "MakeTensor<");
        AssertCodegens(BuildIfElseManyOutputsGraph(), "IfElse(", "[0]", "[8]");
        AssertCodegens(BuildAveragePoolGraph(), "AveragePool(", ".float32().Tensor()");
        AssertCodegens(BuildLpPoolGraph(), "LpPool(", ".float32().Vec()");
        AssertCodegens(BuildConstantOfShapeGraph(), ".Fill(");
        AssertCodegens(BuildConstantValueFloatsLongsGraph(), "Vector(");
        AssertCodegens(BuildDeepSequenceRankChainGraph());
    }

    [Fact]
    public void TestLoopCodegenInlineInitRankMismatchAndHoisting()
    {
        AssertCodegens(BuildLoopInlineAndInitGraph(), "LoopAPI.Iterate(");
        AssertCodegens(BuildLoopCarryRankMismatchGraph(), "LoopAPI.Iterate");
        AssertCodegens(BuildLoopBodyHoistingGraph(), "foreach(var");
    }

    [Fact]
    public void TestBuildLambdaArityOverloads()
    {
        var x1 = InputScalar<float32>("x");
        var oneArg = new CSharpModelBuilder().BuildLambda<Scalar<float32>, Scalar<float32>>(
            new InternalComputationGraph([x1], [x1 + x1]), "OneArgModel");
        Assert.NotNull(oneArg(x1));

        var x2 = InputScalar<float32>("x");
        var y2 = InputScalar<float32>("y");
        var twoArg = new CSharpModelBuilder().BuildLambda<Scalar<float32>, Scalar<float32>, Scalar<float32>>(
            new InternalComputationGraph([x2, y2], [x2 + y2]), "TwoArgModel");
        Assert.NotNull(twoArg(x2, y2));
    }

    [Fact]
    public void TestGetTypeDefStringTensorStructAndFallback()
    {
        TensorStructFieldDef[] simpleFields =
            [new TensorStructFieldDef("CovField_Simple_A", DataStructure.Tensor, rank: 2, DType.Float32)];
        var simpleStruct = InternalOp.TensorStructInput(
            DType.GetOrCreateForTensorStruct(new TensorStructDef(simpleFields, "CovSimpleStruct")),
            InputType.ModelInput, targetFunction: null, defaultName: "simpleStruct");
        Assert.Equal("TensorStruct<CovSimpleStruct>", CSharpModelBuilder.GetTypeDefString(simpleStruct));

        TensorStructFieldDef[] dottedFields =
            [new TensorStructFieldDef("CovField_Dotted_A", DataStructure.Tensor, rank: 1, DType.Int32)];
        var dottedStruct = InternalOp.TensorStructInput(
            DType.GetOrCreateForTensorStruct(new TensorStructDef(dottedFields, "Some.Namespace.CovDottedStruct")),
            InputType.ModelInput, targetFunction: null, defaultName: "dottedStruct");
        Assert.Equal("TensorStruct<DTypeStruct>", CSharpModelBuilder.GetTypeDefString(dottedStruct, null));

        Assert.Contains("float32",
            CSharpModelBuilder.GetTypeDefString(OnnxOp.SequenceEmpty(DType.Float32), null));
    }

    // ---- helpers ----

    private static void AssertCodegens(InternalComputationGraph graph, params string[] containsAll)
    {
        var code = new CSharpModelBuilder().BuildFullGraph(graph, "CovTest");
        Assert.NotNull(code);
        foreach (var s in containsAll)
            Assert.Contains(s, code);
    }

    private static InternalComputationGraph BuildConstantBranchesGraph()
    {
        short[] shorts = [1, 2, 3, 4];
        ushort[] ushorts = [1, 2, 3, 4];
        uint[] uints = [1u, 2u, 3u, 4u];
        Variable[] outputs =
        [
            Scalar(1.5),
            Scalar((short)2),
            Scalar(3),
            Scalar((ushort)4),
            Scalar(5u),
            Scalar(6UL),
            Scalar(true),
            Vector(shorts),
            Vector(ushorts),
            Vector(uints),
            Vector(1.0, 2.0),
            Vector(7L, 8L),
            EmptyVector<int32>(),
        ];
        return new InternalComputationGraph([], ImmutableArray.Create(outputs));
    }

    private static InternalComputationGraph BuildLowOpInlinedGraph()
    {
        var x = InputScalar<float32>("x");
        var y = InputScalar<float32>("y");
        var z = InputScalar<float32>("z");
        return new InternalComputationGraph([x, y, z], [(x * y) + (y * z)]);
    }

    private static InternalComputationGraph BuildBigConstantGraph()
    {
        var bigVec = Enumerable.Range(0, 200).Select(i => (float)i).ToArray();
        return new InternalComputationGraph([], [Vector(bigVec)]);
    }

    private static InternalComputationGraph BuildSequenceRankInferGraph()
    {
        var rankedElem = InputTensor<float32>("ranked", rank: 2);
        var unrankedElem = InputTensor<float32>("unranked");

        var seq1 = OnnxOp.SequenceInsert(OnnxOp.SequenceEmpty(DType.Float32), rankedElem, null);
        seq1 = OnnxOp.SequenceInsert(seq1, unrankedElem, null);
        var atInsert = OnnxOp.SequenceAt(seq1, Scalar(0L));

        var seq2 = OnnxOp.SequenceConstruct(rankedElem, unrankedElem);
        seq2 = OnnxOp.SequenceErase(seq2, Scalar(0L));
        var atErase = OnnxOp.SequenceAt(seq2, Scalar(0L));

        var seq3 = OnnxOp.SequenceConstruct(rankedElem);
        seq3 = OnnxOp.Identity(seq3, rank: null);
        var atIdentity = OnnxOp.SequenceAt(seq3, Scalar(0L));

        return new InternalComputationGraph(
            [rankedElem, unrankedElem],
            ImmutableArray.Create<Variable>(atInsert, atErase, atIdentity));
    }

    private static InternalComputationGraph BuildScanLoopGraph()
    {
        Vector<float32>? scannedScalar = null;
        Tensor<float32>? scannedTensor = null;
        foreach (var ctx in LoopAPI.Iterate(Scalar(3L)))
        {
            scannedScalar = ctx.Scan(ctx.IterationIndex.Cast<float32>());
            scannedTensor = ctx.Scan(Vector(1.0f, 2.0f));
            ctx.Break(ctx.IterationIndex >= Scalar(10L));
        }
        return new InternalComputationGraph([], [scannedScalar!, scannedTensor!]);
    }

    private static InternalComputationGraph BuildUnsupportedConstantDtypeGraph()
        => new InternalComputationGraph([], [Scalar((sbyte)1), Scalar((byte)2)]);

    private static InternalComputationGraph BuildIfElseManyOutputsGraph()
    {
        var cond = InputScalar<bit>("cond");
        IValue[] t =
        [
            Scalar(1.0f), Scalar(2.0f), Scalar(3.0f),
            Scalar(4.0f), Scalar(5.0f), Scalar(6.0f),
            Scalar(7.0f), Scalar(8.0f), Scalar(9.0f),
        ];
        IValue[] f =
        [
            Scalar(10.0f), Scalar(20.0f), Scalar(30.0f),
            Scalar(40.0f), Scalar(50.0f), Scalar(60.0f),
            Scalar(70.0f), Scalar(80.0f), Scalar(90.0f),
        ];
        return new InternalComputationGraph([cond], ImmutableArray.Create(FrameworkOps.IfElse(cond, t, f)));
    }

    private static InternalComputationGraph BuildAveragePoolGraph()
    {
        var x = InputTensor<float32>("x2", rank: 4);
        long[] kernelShape = [2, 2];
        var pooled = OnnxOp.AveragePool(x, autoPad: null, ceilMode: null, countIncludePad: null,
            dilations: null, kernelShape: kernelShape, pads: null, strides: null);
        return new InternalComputationGraph([x], [pooled]);
    }

    private static InternalComputationGraph BuildLpPoolGraph()
    {
        var x = InputTensor<float32>("x1", rank: 1);
        long[] kernelShape = [1];
        var pooled = OnnxOp.LpPool(x, autoPad: null, ceilMode: null, kernelShape: kernelShape,
            p: null, pads: null, strides: null, dilations: null);
        return new InternalComputationGraph([x], [pooled]);
    }

    private static InternalComputationGraph BuildConstantOfShapeGraph()
    {
        long[] dims = [1];
        var filled = OnnxOp.ConstantOfShape(Vector(2L, 3L), TensorData(DType.Float32, dims, 7.0f));
        return new InternalComputationGraph([], [filled]);
    }

    private static InternalComputationGraph BuildConstantValueFloatsLongsGraph()
    {
        float[] floatVals = [1.0f, 2.0f, 3.0f];
        long[] intVals = [10L, 20L, 30L];
        object?[] floatAttrs = [OnnxOpAttributeNames.AttrValueFloats, floatVals];
        object?[] intAttrs = [OnnxOpAttributeNames.AttrValueInts, intVals];
        var floats = NodeBuilder.CallCustomOperator<Vector<float32>>(OpCodes.CONSTANT, [], floatAttrs);
        var ints = NodeBuilder.CallCustomOperator<Vector<int64>>(OpCodes.CONSTANT, [], intAttrs);
        return new InternalComputationGraph([], ImmutableArray.Create<Variable>(floats, ints));
    }

    private static InternalComputationGraph BuildDeepSequenceRankChainGraph()
    {
        var unrankedElem = (Tensor<float32>)OnnxOp.Identity(
            InputTensor<float32>("eltIn", rank: 2), rank: null);

        var s0 = OnnxOp.SequenceConstruct(unrankedElem);
        var s1 = OnnxOp.Identity(s0, rank: null);
        var s2 = OnnxOp.SequenceErase(s1, Scalar(0L));
        var s3 = OnnxOp.Identity(s2, rank: null);
        var deepAt = OnnxOp.SequenceAt(s3, Scalar(0L));

        return new InternalComputationGraph([InputTensor<float32>("eltIn", rank: 2)], [deepAt]);
    }

    private static InternalComputationGraph BuildLoopInlineAndInitGraph()
    {
        Scalar<int64> counter = Scalar(0L);
        foreach (var ctx in LoopAPI.Iterate(Scalar(3L)))
        {
            LoopAPI.Init(counter);
            counter = counter + Scalar(1L);
        }
        return new InternalComputationGraph([], [counter]);
    }

    private static InternalComputationGraph BuildLoopCarryRankMismatchGraph()
    {
        Scalar<float32> accum = (Scalar<float32>)OnnxOp.Identity(Scalar(1.0f), rank: null);
        foreach (var ctx in LoopAPI.Iterate(Scalar(2L)))
        {
            LoopAPI.Init(accum);
            accum = accum + Scalar(1.0f);
        }
        return new InternalComputationGraph([], [accum]);
    }

    private static InternalComputationGraph BuildLoopBodyHoistingGraph()
    {
        Tensor<float32>? scanned = null;
        Scalar<int64>? finalIdx = null;
        foreach (var ctx in LoopAPI.Iterate(Scalar(2L)))
        {
            scanned = ctx.Scan(Vector(1.0f, 2.0f));
            finalIdx = ctx.IterationIndex;
            ctx.Break(ctx.IterationIndex >= Scalar(0L));
        }
        return new InternalComputationGraph([], ImmutableArray.Create<Variable>(scanned!, finalIdx!));
    }
}
