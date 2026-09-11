using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Shorokoo.Core.Factory.CSharpFactory;
using Shorokoo.Core.Nodes.Processors.Helpers;
using System.Collections.Immutable;
using FrameworkOps = Shorokoo.Core.Nodes.Ops;

namespace Shorokoo.Tests;

/// <summary>
/// Drives <see cref="CSharpModelBuilder"/> over graph shapes that reach the
/// per-op / per-attribute / per-DType / per-keyword branches of the codegen dispatch.
/// Every emitted source is compiled, not just substring-matched.
/// </summary>
[Trait("Domain", "Factory")]
[Trait("Purpose", "Coverage")]
public class CSharpModelBuilderCoverageTests
{
    [Fact]
    public void TestBuildFullGraphModuleTensorStructSequenceAndConstantCodegen()
    {
        AssertCodegens(CallsHypersLayer.ComputationGraph.ToInternal(), "HypersLayer");
        // Not the struct's name: DType.GetOrCreateForTensorStruct keys on structure alone, so which
        // of several structurally identical IStructs is named depends on registration order.
        AssertCodegens(TensorStructLoopCarry.ComputationGraph.ToInternal(),
            "Globals.TensorStructCreate<", "Globals.TensorStructGetField");
        AssertCodegens(SequenceOpsOnStructs.ComputationGraph.ToInternal());
        AssertCodegens(BuildConstantBranchesGraph(),
            "1.5d", "6UL", "true", "(short[])", "(ushort[])", "(uint[])", "EmptyVector<int32>");
        AssertCodegens(BatchNormWithStateUpdate.ComputationGraph.ToInternal(), "isTrainable: false");
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
    public void TestScanCodegenScansTheCarryBeforeTheBodyUpdatesIt()
    {
        var g = ScanCarryBeforeUpdate.ComputationGraph.ToInternal();
        var arch = g.ToConcreteArchitecture(g.FromOrderedInputs(
            [TensorData(DType.Float32, [], 10f), TensorData(DType.Int64, [], 3L)]));
        string[] body = [.. new CSharpModelBuilder().BuildFullGraph(arch, "CovTest")
            .Split('\n').Select(x => x.Trim())];

        var scan = Assert.Single(body, x => x.Contains(" = ctx.Scan("));
        string[] names = [.. scan.TrimEnd(';', ')').Split(" = ctx.Scan(")];
        Assert.True(Array.IndexOf(body, scan) < Array.FindIndex(body, x => x.StartsWith(names[1] + " = ")));
        Assert.Contains(body, x => x.EndsWith($"> {names[0]} = default;"));
    }

    [Fact]
    public void TestADTypeValuedAttributeCodegensItsValue()
        => AssertCodegens(ScanZeroInputOpInLoopBody.ComputationGraph.ToInternal(),
            "\"dtype\", DType.Float32", "\"shape\", new long[] { 2L }");

    /// <summary>A concrete dtype carrying a generic-parameter tag writes as the plain dtype static;
    /// the tag is metadata the literal does not carry. Fails: DTypeLiteral compares the tagged dtype
    /// to DType.FromName, whose result is untagged, so every tagged dtype is rejected as unwritable.</summary>
    [Fact]
    public void TestATaggedConcreteDTypeCodegensItsBaseDType()
        => AssertCodegens(new InternalComputationGraph([],
                [OnnxOp.RandomNormal([2L], dtype: DType.CreateWithGenericParam(DType.Float32, "T"))]),
            "\"dtype\", DType.Float32");

    /// <summary>Fails: StructTypeName writes TensorStructDef.TypeName, which is Type.FullName, so a
    /// generic IStruct emits a backtick-arity assembly-qualified name that is not C#.</summary>
    [Fact]
    public void TestAGenericIStructCodegensAWritableTypeName()
        => AssertCodegens(BuildGenericStructGraph());

    /// <summary>Fails: an absent optional input emits "null, " whatever the placeholder asked for,
    /// so a custom-operator call — whose template supplies its own separators — gets a doubled one
    /// and an empty argument, "null, , null".</summary>
    [Fact]
    public void TestAnAbsentOptionalInputCodegensOneArgument()
        => AssertCodegens(BuildCustomOpNullInputGraph(), "null, null");

    /// <summary>Fails: a state initializer's StateOwnership is not emitted at all, so an
    /// optimizer-owned one codegens as the ModuleOwned default the 3-argument
    /// CallTrainableParamInitializer overload hardcodes.</summary>
    [Fact]
    public void TestAStateInitializerCodegensItsOwnership()
        => AssertCodegens(OptimizerOwnedStateModel.ComputationGraph.ToInternal(),
            "stateOwnership: StateOwnership.OptimizerOwned");

    /// <summary>Fails: a string-valued Constant emits Vector(["a", "b"]), and Globals.Vector has an
    /// overload for every element type but string.</summary>
    [Fact]
    public void TestAStringConstantCodegensAVectorItCanBind()
        => AssertCodegens(new InternalComputationGraph([],
                [OnnxOp.Constant((string[])["cova", "covb"]), OnnxOp.Constant("covc")]),
            "Vector(", "Scalar(");

    [Fact]
    public void TestCodegenedSourceRebuildsTheGraphItCameFrom()
    {
        TensorData[] two = [TensorData([], 1f), TensorData([], 2f)];
        AssertRoundTrips(TensorStructLoopCarry.ComputationGraph.ToInternal(), two);
        AssertRoundTrips(SequenceOpsOnStructs.ComputationGraph.ToInternal(),
            [.. two, TensorData([], 3f), TensorData([], 4f)]);
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

        TensorStructFieldDef[] namelessFields =
            [new TensorStructFieldDef("CovField_Nameless_A", DataStructure.Tensor, rank: 1, DType.Int32)];
        var namelessStruct = InternalOp.TensorStructCreate(
            DType.GetOrCreateForTensorStruct(new TensorStructDef(namelessFields)), [Vector(1, 2)]);
        Assert.Throws<UnsupportedDTypeException>(() => new CSharpModelBuilder()
            .BuildFullGraph(new InternalComputationGraph([], [namelessStruct]), "CovTest"));
    }

    // ---- helpers ----

    private static void AssertCodegens(InternalComputationGraph graph, params string[] containsAll)
    {
        var code = new CSharpModelBuilder().BuildFullGraph(graph, "CovTest");
        Assert.NotNull(code);
        foreach (var s in containsAll)
            Assert.Contains(s, code);

        var compilation = CSharpCompilation.Create("CovTest", [CSharpSyntaxTree.ParseText(code)],
            CSharpModelBuilder.CompilationReferences(),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));

        using var ms = new MemoryStream();
        Assert.Empty(compilation.Emit(ms).Diagnostics.Where(x => x.Severity == DiagnosticSeverity.Error));
    }

    /// <summary><see cref="AutoTest"/> executes generated source only for a graph with no inputs,
    /// so this is the one place the emitted code for a graph that takes them runs at all.</summary>
    private static void AssertRoundTrips(InternalComputationGraph graph, TensorData[] inputs)
    {
        var method = new CSharpModelBuilder().BuildMethod(graph, "CovTest");
        var graphInputs = InternalComputationGraphConverter.BuildNodes(graph).inputs;
        object?[] args = [.. method.GetParameters().Zip(graphInputs).Select(x =>
            x.First.ParameterType.GetMethod("op_Implicit", [typeof(Variable)])!.Invoke(null, [x.Second]))];
        var rebuilt = new InternalComputationGraph(
            graphInputs, [((IValue)method.Invoke(null, args)!).ToVariable()]);

        Assert.Equal(Run(graph, inputs), Run(rebuilt, inputs));
    }

    private static byte[][] Run(InternalComputationGraph graph, TensorData[] inputs)
    {
        var model = graph.ToConcreteArchitecture(graph.FromOrderedInputs([.. inputs])).ToConcreteModel();
        return [.. Shorokoo.Runtime.ComputeContext.Default.Execute(model, [.. inputs.Cast<IData>()])
            .Select(x => x.ToTensorData().AccessRawMemory().ToArray())];
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
        var eltIn = InputTensor<float32>("eltIn", rank: 2);
        var unrankedElem = (Tensor<float32>)OnnxOp.Identity(eltIn, rank: null);

        var s0 = OnnxOp.SequenceConstruct(unrankedElem);
        var s1 = OnnxOp.Identity(s0, rank: null);
        var s2 = OnnxOp.SequenceErase(s1, Scalar(0L));
        var s3 = OnnxOp.Identity(s2, rank: null);
        var deepAt = OnnxOp.SequenceAt(s3, Scalar(0L));

        return new InternalComputationGraph([eltIn], [deepAt]);
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

    private static InternalComputationGraph BuildCustomOpNullInputGraph()
    {
        var x = InputTensor<float32>("lx", rank: 3);
        var w = InputTensor<float32>("lw", rank: 3);
        var r = InputTensor<float32>("lr", rank: 3);
        var (y, _, _) = OnnxOp.Lstm(x, w, r, null, null, null, null, null,
            null, null, null, null, LSTMDirection.Forward, 4L, null, null);
        return new InternalComputationGraph([x, w, r], [y]);
    }

    private static InternalComputationGraph BuildGenericStructGraph()
    {
        var pair = TensorStruct<CovGenericPair<float32>>(Scalar(1.0f), Scalar(2.0f));
        return new InternalComputationGraph([], [pair.CovGenericPairFieldA + pair.CovGenericPairFieldB]);
    }

    private static InternalComputationGraph BuildLoopBodyHoistingGraph()
    {
        Tensor<float32>? scanned = null;
        var finalIdx = Scalar(-1L);
        foreach (var ctx in LoopAPI.Iterate(Scalar(2L)))
        {
            LoopAPI.Init(finalIdx);
            scanned = ctx.Scan(Vector(1.0f, 2.0f));
            finalIdx = ctx.IterationIndex + Scalar(0L);
            ctx.Break(ctx.IterationIndex >= Scalar(0L));
        }
        return new InternalComputationGraph([], ImmutableArray.Create<Variable>(scanned!, finalIdx));
    }
}

/// <summary>A generic IStruct, whose field names are unique to this file so its structural
/// registration cannot collide with another fixture's.</summary>
public interface CovGenericPair<T> : IStruct where T : IVarType
{
    Scalar<T> CovGenericPairFieldA { get; }
    Scalar<T> CovGenericPairFieldB { get; }
}
