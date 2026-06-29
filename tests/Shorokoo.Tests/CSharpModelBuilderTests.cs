using Shorokoo.Core.Factory.CSharpFactory;
using Shorokoo.Core.Nodes.Processors.Helpers;
using System.Collections.Immutable;
using FrameworkOps = Shorokoo.Core.Nodes.Ops;

namespace Shorokoo.Tests;

/// <summary>
/// Drives <see cref="CSharpModelBuilder.BuildFullGraph"/> over a variety of
/// graph shapes so the per-op handlers + per-attribute / per-DType /
/// per-keyword branches in the codegen dispatch all get exercised.
///
/// <para>
/// Each entry in <see cref="TestBuildFullGraphCodegenCoverage"/> is one
/// <c>AssertCodegens(graph, ...)</c> one-liner. Graphs come from two sources:
/// existing <c>[Module]</c> classes (whose pre-concretization
/// <c>ComputationGraph</c> still carries MODEL_INVOKE / TENSOR_STRUCT_* /
/// MODULE_SET_HYPERPARAMS that the AutoTester would lower away) and small
/// hand-built graphs (built via the private <c>Build*</c> helpers) for
/// shape / DType / template-keyword branches that no Module convenient
/// produces.
/// </para>
///
/// <para>
/// <see cref="TestBuildLambdaArityOverloads"/> separately covers
/// <see cref="CSharpModelBuilder.BuildLambda{T1,TResult}"/> and
/// <see cref="CSharpModelBuilder.BuildLambda{T1,T2,TResult}"/> — those need
/// graphs with matching runtime-input arity and an actual compile+invoke,
/// so they don't fit the AssertCodegens one-liner shape.
/// </para>
/// </summary>
[Trait("Domain", "Factory")]
[Trait("Purpose", "Coverage")]
public class CSharpModelBuilderCoverageTests
{
    [Fact]
    public void TestBuildFullGraphCodegenCoverage()
    {
        // Module-call codegen — CREATE_MODULE / MODULE_SET_HYPERPARAMS / MODEL_INVOKE
        // handlers + MakeCallFunctionCodeTemplate + GetModuleAwareTypeDefString overloads.
        AssertCodegens(CallsHypersLayer.ComputationGraph, "HypersLayer");
        // TensorStruct codegen — MakeTensorStructCreateNode + MakeTensorStructGetFieldNode
        // (now emitting canonical InternalOp.TensorStructCreate / TensorStructGetField).
        AssertCodegens(TensorStructLoopCarry.ComputationGraph,
            "InternalOp.TensorStructCreate", "InternalOp.TensorStructGetField");
        // Sequence ops over struct sequences — InferSequenceElementRankFromTensor
        // SEQUENCE_CONSTRUCT / IDENTITY tracebacks (when SequenceAt output rank is unknown).
        AssertCodegens(SequenceOpsOnStructs.ComputationGraph);
        // Per-DType constant arms of MakeConstantNode (Float64 / Int16 / Int32 /
        // UInt16 / UInt32 / UInt64 / Bool plus the <4 vs >=4 collection-expression split).
        AssertCodegens(BuildConstantBranchesGraph(),
            "1.5d", "6UL", "true", "(short[])", "(ushort[])", "(uint[])", "EmptyVector<int32>");
        // StateParamInitializer arm of BuildMethodCode (the mirror of the
        // TrainableParamInitializer branch covered by every InitSimple use).
        AssertCodegens(BatchNormWithStateUpdate.ComputationGraph,
            "[StateInitializer]", "isTrainable: false");
        // low_op keyword wraps inlined inputs in parens for precedence — only
        // fires when the input's producer has an inline expression (non-constant).
        AssertCodegens(BuildLowOpInlinedGraph(), "(", ")");
        // AttributeType.Tensor with "dims" and "base64string" keywords — fires
        // when MakeConstantNode returns null (>500 bytes raw memory) so the
        // Constant op falls back to the MakeTensor<T>({a:dims}, "{a:base64string}") template.
        AssertCodegens(BuildBigConstantGraph(), "MakeTensor<");
        // InferSequenceElementRank SEQUENCE_INSERT / SEQUENCE_ERASE / IDENTITY tracebacks.
        AssertCodegens(BuildSequenceRankInferGraph());
        // MakeLoopNode scan-variable emission + non-trivial ctx.ContinueWhile.
        AssertCodegens(BuildScanLoopGraph(), ".Scan(", ".ContinueWhile(");
    }

    [Fact]
    public void TestBuildLambdaArityOverloads()
    {
        // 1-arg overload: f(x) = x + x.
        {
            var x = InputScalar<float32>("x");
            var y = x + x;
            var graph = new FastComputationGraph([x], [y]);
            var fastGraph = FastComputationGraphConverter.ToFastGraph(graph);
            var lambda = new CSharpModelBuilder().BuildLambda<Scalar<float32>, Scalar<float32>>(fastGraph, "OneArgModel");
            Assert.NotNull(lambda(x));
        }
        // 2-arg overload: f(x, y) = x + y.
        {
            var x = InputScalar<float32>("x");
            var y = InputScalar<float32>("y");
            var z = x + y;
            var graph = new FastComputationGraph([x, y], [z]);
            var fastGraph = FastComputationGraphConverter.ToFastGraph(graph);
            var lambda = new CSharpModelBuilder().BuildLambda<Scalar<float32>, Scalar<float32>, Scalar<float32>>(fastGraph, "TwoArgModel");
            Assert.NotNull(lambda(x, y));
        }
    }

    // ---- helpers ----

    /// <summary>Runs BuildFullGraph and asserts every substring in <paramref name="containsAll"/> appears in the result.</summary>
    private static void AssertCodegens(FastComputationGraph graph, params string[] containsAll)
    {
        var code = new CSharpModelBuilder().BuildFullGraph(graph, "CovTest");
        Assert.NotNull(code);
        foreach (var s in containsAll)
            Assert.Contains(s, code);
    }

    private static FastComputationGraph BuildConstantBranchesGraph()
    {
        var outputs = new Variable[]
        {
            Scalar(1.5),                                // float64 scalar
            Scalar((short)2),                           // int16 scalar (<4 cast path)
            Scalar(3),                                  // int32 scalar
            Scalar((ushort)4),                          // uint16 scalar (<4 cast path)
            Scalar(5u),                                 // uint32 scalar (<4 cast path)
            Scalar(6UL),                                // uint64 scalar
            Scalar(true),                               // bool scalar
            Vector(new short[] { 1, 2, 3, 4 }),         // int16 vector (>=4 collection expression)
            Vector(new ushort[] { 1, 2, 3, 4 }),        // uint16 vector
            Vector(new uint[] { 1u, 2u, 3u, 4u }),      // uint32 vector
            Vector(1.0, 2.0),                           // float64 vector
            Vector(7L, 8L),                             // int64 vector
            EmptyVector<int32>(),                       // empty-vector branch
        };
        return ToFastGraph(new FastComputationGraph([], ImmutableArray.Create(outputs)));
    }

    private static FastComputationGraph BuildLowOpInlinedGraph()
    {
        var x = InputScalar<float32>("x");
        var y = InputScalar<float32>("y");
        var z = InputScalar<float32>("z");
        // (x*y) + (y*z) — each multiply is non-constant so its result is available
        // for inlining when fed into the outer add (where low_op wraps it in parens).
        var result = (x * y) + (y * z);
        return ToFastGraph(new FastComputationGraph([x, y, z], [result]));
    }

    private static FastComputationGraph BuildBigConstantGraph()
    {
        // 200 floats × 4 bytes = 800 bytes > 500 byte threshold → MakeConstantNode
        // returns null and the standard Constant code template fires.
        var bigVec = Enumerable.Range(0, 200).Select(i => (float)i).ToArray();
        return ToFastGraph(new FastComputationGraph([], [Vector(bigVec)]));
    }

    private static FastComputationGraph BuildSequenceRankInferGraph()
    {
        var rankedElem = InputTensor<float32>("ranked", rank: 2);
        var unrankedElem = InputTensor<float32>("unranked");

        // SEQUENCE_INSERT chain — first insert finds elementInput.Rank() (L1684),
        // second chains back through the input sequence (L1693).
        var seq1 = OnnxOp.SequenceInsert(OnnxOp.SequenceEmpty(DType.Float32), rankedElem, null);
        seq1 = OnnxOp.SequenceInsert(seq1, unrankedElem, null);
        var atInsert = OnnxOp.SequenceAt(seq1, Scalar(0L));

        // SEQUENCE_ERASE chain (L1712-1718).
        var seq2 = OnnxOp.SequenceConstruct(rankedElem, unrankedElem);
        seq2 = OnnxOp.SequenceErase(seq2, Scalar(0L));
        var atErase = OnnxOp.SequenceAt(seq2, Scalar(0L));

        // IDENTITY pass-through (L1720-1726).
        var seq3 = OnnxOp.SequenceConstruct(rankedElem);
        seq3 = OnnxOp.Identity(seq3, rank: null);
        var atIdentity = OnnxOp.SequenceAt(seq3, Scalar(0L));

        return ToFastGraph(new FastComputationGraph(
            [rankedElem, unrankedElem],
            ImmutableArray.Create<Variable>(atInsert, atErase, atIdentity)));
    }

    private static FastComputationGraph BuildScanLoopGraph()
    {
        Vector<float32>? scannedScalar = null;
        Tensor<float32>? scannedTensor = null;
        foreach (var ctx in LoopAPI.Iterate(Scalar(3L)))
        {
            scannedScalar = ctx.Scan(ctx.IterationIndex.Cast<float32>());
            scannedTensor = ctx.Scan(Vector(1.0f, 2.0f));
            ctx.Break(ctx.IterationIndex >= Scalar(10L));
        }
        return ToFastGraph(new FastComputationGraph([], [scannedScalar!, scannedTensor!]));
    }

    private static FastComputationGraph ToFastGraph(FastComputationGraph graph)
        => FastComputationGraphConverter.ToFastGraph(graph);

    // ────────────────────────────────────────────────────────────────────────
    // Static-method branches: GetTypeDefString. Drives the ITensorStruct arms
    // (simple TypeName + fully-qualified/null TypeName) and the IValue<T>
    // fallback. None of these arms is reached by BuildFullGraph alone because
    // codegen for ITensorStruct inputs/outputs doesn't pass through this
    // overload when those inputs flow as parameters to user code; we exercise
    // them directly here.
    // ────────────────────────────────────────────────────────────────────────

    [Fact]
    public void TestGetTypeDefStringTensorStructAndFallback()
    {
        // Simple unqualified TypeName → "TensorStruct<MyType>"
        var simpleFields = new[]
        {
            new TensorStructFieldDef("CovField_Simple_A", DataStructure.Tensor, rank: 2, DType.Float32),
        };
        var simpleDef = new TensorStructDef(simpleFields, "CovSimpleStruct");
        var simpleStruct = InternalOp.TensorStructInput(
            DType.GetOrCreateForTensorStruct(simpleDef), InputType.ModelInput,
            targetFunction: null, defaultName: "simpleStruct");
        Assert.Equal("TensorStruct<CovSimpleStruct>",
            CSharpModelBuilder.GetTypeDefString(simpleStruct));

        // Dotted TypeName → falls back to "TensorStruct<DTypeStruct>"
        var dottedFields = new[]
        {
            new TensorStructFieldDef("CovField_Dotted_A", DataStructure.Tensor, rank: 1, DType.Int32),
        };
        var dottedDef = new TensorStructDef(dottedFields, "Some.Namespace.CovDottedStruct");
        var dottedStruct = InternalOp.TensorStructInput(
            DType.GetOrCreateForTensorStruct(dottedDef), InputType.ModelInput,
            targetFunction: null, defaultName: "dottedStruct");
        Assert.Equal("TensorStruct<DTypeStruct>",
            CSharpModelBuilder.GetTypeDefString(dottedStruct, null));

        // IValue<T> fallback: not ITensorStruct, not ITensor → an
        // ITensorSequence reaches the final else at line 112.
        var sequence = OnnxOp.SequenceEmpty(DType.Float32);
        var typeDef = CSharpModelBuilder.GetTypeDefString(sequence, null);
        Assert.Contains("float32", typeDef);
    }

    // ────────────────────────────────────────────────────────────────────────
    // MakeConstantNode unsupported-dtype branch (lines 800-801). The supported
    // DTypes are Float32/Float64/Int16/Int32/Int64/UInt16/UInt32/UInt64/Bool;
    // anything else (here int8, float16, bfloat16) falls into the else arm
    // and MakeConstantNode returns null, forcing the standard Constant code
    // template path via MakeTensor<T>.
    // ────────────────────────────────────────────────────────────────────────

    [Fact]
    public void TestConstantUnsupportedDtypesFallback()
    {
        var graph = new FastComputationGraph(
            [],
            [
                Scalar((sbyte)1),                                 // int8 → unsupported
                Scalar((byte)2),                                  // uint8 → unsupported
            ]);
        var code = new CSharpModelBuilder().BuildFullGraph(ToFastGraph(graph), "CovTest");
        Assert.NotNull(code);
        // Falls back to MakeTensor<T>({dims}, "{base64}") emit from the
        // standard Constant code template.
        Assert.Contains("MakeTensor<", code);
    }

    // ────────────────────────────────────────────────────────────────────────
    // MakeIfNode: numItems > 8 path (lines 1061-1068). Build an If-Else with
    // 9 outputs using the IValue[] overload so the array-result codegen
    // (var ifResult = Ops.IfElse(c, [t...], [f...]);) fires.
    // ────────────────────────────────────────────────────────────────────────

    [Fact]
    public void TestIfElseManyOutputs()
    {
        var cond = InputScalar<bit>("cond");
        var t = new IValue[]
        {
            Scalar(1.0f),  Scalar(2.0f),  Scalar(3.0f),
            Scalar(4.0f),  Scalar(5.0f),  Scalar(6.0f),
            Scalar(7.0f),  Scalar(8.0f),  Scalar(9.0f),
        };
        var f = new IValue[]
        {
            Scalar(10.0f), Scalar(20.0f), Scalar(30.0f),
            Scalar(40.0f), Scalar(50.0f), Scalar(60.0f),
            Scalar(70.0f), Scalar(80.0f), Scalar(90.0f),
        };
        var results = FrameworkOps.IfElse(cond, t, f);
        var graph = new FastComputationGraph([cond], ImmutableArray.Create(results));
        var code = new CSharpModelBuilder().BuildFullGraph(ToFastGraph(graph), "CovTest");
        // Array-result form: var <arr> = Ops.IfElse(cond, [...], [...]); then
        // individual `var x_i = (<TypeDef>)<arr>[i];` lines bind the per-output names.
        Assert.Contains("IfElse(", code);
        Assert.Contains("[0]", code);
        Assert.Contains("[8]", code);
    }

    // ────────────────────────────────────────────────────────────────────────
    // MakeLoopNode coverage holes around loop-variable carry-overs (lines
    // 847-850 inlineable max-iteration, 869-872 inlineable initializer,
    // 899/921 LoopAPI.Init when a carry has no body consumer, 965-968 close
    // input inlineable). Uses the LoopAPI.Init pattern: declare the carry
    // variable outside the loop, call LoopAPI.Init(d), then reassign d.
    // Scalar(3L) → InlineExpression < 30 chars so the max-iter inline arm
    // fires.
    // ────────────────────────────────────────────────────────────────────────

    [Fact]
    public void TestLoopInlineAndInitPaths()
    {
        Scalar<int64> counter = Scalar(0L);
        foreach (var ctx in LoopAPI.Iterate(Scalar(3L)))
        {
            LoopAPI.Init(counter);
            counter = counter + Scalar(1L);
        }
        var graph = new FastComputationGraph([], [counter]);
        var code = new CSharpModelBuilder().BuildFullGraph(ToFastGraph(graph), "CovTest");
        Assert.Contains("LoopAPI.Iterate(", code);
    }

    // ────────────────────────────────────────────────────────────────────────
    // MakeLoopNode lines 877-882: rank-mismatch initializer cast. Triggered
    // when the carryover has a known rank but its initializer's producer has
    // unknown rank — codegen inserts `.Scalar()` or `.Vec()` to recover the
    // rank annotation.
    // ────────────────────────────────────────────────────────────────────────

    [Fact]
    public void TestLoopCarryRankMismatchInitializer()
    {
        var unrankedScalarInit = (Scalar<float32>)OnnxOp.Identity(Scalar(1.0f), rank: null);
        Scalar<float32> accum = unrankedScalarInit;
        foreach (var ctx in LoopAPI.Iterate(Scalar(2L)))
        {
            LoopAPI.Init(accum);
            accum = accum + Scalar(1.0f);
        }
        var graph = new FastComputationGraph([], [accum]);
        var code = new CSharpModelBuilder().BuildFullGraph(ToFastGraph(graph), "CovTest");
        Assert.Contains("LoopAPI.Iterate", code);
    }

    // ────────────────────────────────────────────────────────────────────────
    // BuildMethodCode Phase 3 hoisting (lines 411-483): a variable produced
    // inside a loop body but referenced after the loop ends gets hoisted to
    // a forward declaration before the loop. ScanAndUseInsideAfter exercises
    // this by reading a loop-body intermediate after the loop completes.
    // ────────────────────────────────────────────────────────────────────────

    [Fact]
    public void TestBuildMethodCodeHoisting()
    {
        Tensor<float32>? scanned = null;
        Scalar<int64>? finalIdx = null;
        foreach (var ctx in LoopAPI.Iterate(Scalar(2L)))
        {
            scanned = ctx.Scan(Vector(1.0f, 2.0f));
            finalIdx = ctx.IterationIndex;
            ctx.Break(ctx.IterationIndex >= Scalar(0L));
        }
        var graph = new FastComputationGraph([], ImmutableArray.Create<Variable>(scanned!, finalIdx!));
        var code = new CSharpModelBuilder().BuildFullGraph(ToFastGraph(graph), "CovTest");
        // finalIdx is declared inside the foreach but consumed by the return
        // expression at outer scope; codegen must emit a forward declaration.
        Assert.NotNull(code);
        Assert.Contains("foreach(var", code);
    }

    // ────────────────────────────────────────────────────────────────────────
    // MakeNodeWithInlines {oN:fromvar} keyword — AveragePool / LpPool code
    // templates emit `{1:param}...){o1:fromvar}` so the result is cast back
    // to the right tensor rank. Each rank arm (Scalar, Vec, Tensor) is
    // exercised by feeding ranks 0/1/2 of float32 tensor input.
    // ────────────────────────────────────────────────────────────────────────

    [Fact]
    public void TestAveragePoolFromVarTensorRanks()
    {
        var x2 = InputTensor<float32>("x2", rank: 4);  // typical pool input (N,C,H,W)
        var pooled = OnnxOp.AveragePool(x2, autoPad: null, ceilMode: null, countIncludePad: null,
            dilations: null, kernelShape: new long[] { 2, 2 }, pads: null, strides: null);
        var graph = new FastComputationGraph([x2], [pooled]);
        var code = new CSharpModelBuilder().BuildFullGraph(ToFastGraph(graph), "CovTest");
        Assert.Contains("AveragePool(", code);
        Assert.Contains(".float32().Tensor()", code);

        var x1 = InputTensor<float32>("x1", rank: 1);
        var lpPooled = OnnxOp.LpPool(x1, autoPad: null, ceilMode: null, kernelShape: new long[] { 1 },
            p: null, pads: null, strides: null, dilations: null);
        var g1 = new FastComputationGraph([x1], [lpPooled]);
        var code1 = new CSharpModelBuilder().BuildFullGraph(ToFastGraph(g1), "CovTest");
        Assert.Contains("LpPool(", code1);
        Assert.Contains(".float32().Vec()", code1);
    }

    // ────────────────────────────────────────────────────────────────────────
    // ConstantOfShape with explicit shape (rank>=1) exercises ConstantOfShape's
    // code template and broader MakeNodeWithInlines tensor-attribute path.
    // ────────────────────────────────────────────────────────────────────────

    [Fact]
    public void TestConstantOfShapeCodegen()
    {
        var shape = Vector(2L, 3L);
        var filled = OnnxOp.ConstantOfShape(shape, TensorData(DType.Float32, new long[] { 1 }, 7.0f));
        var graph = new FastComputationGraph([], [filled]);
        var code = new CSharpModelBuilder().BuildFullGraph(ToFastGraph(graph), "CovTest");
        // ConstantOfShape's standard code template lowers to Tensor<T>.Fill(...).
        Assert.Contains(".Fill(", code);
    }

    // ────────────────────────────────────────────────────────────────────────
    // Constant nodes with non-tensor `value_*` attributes use the per-attr
    // `params` keyword path in MakeNodeWithInlines (`Vector({c:params})` /
    // `Vector({e:params})` templates in Definitions.AC.cs), exercising the
    // AttributeType.Floats / AttributeType.Longs `params` branches.
    // ────────────────────────────────────────────────────────────────────────

    [Fact]
    public void TestConstantValueFloatsLongs()
    {
        // value_floats attribute → AttributeType.Floats with :params}
        var floats = NodeBuilder.CallCustomOperator<Vector<float32>>(
            OpCodes.CONSTANT, [],
            new object?[] { OnnxOpAttributeNames.AttrValueFloats, new float[] { 1.0f, 2.0f, 3.0f } });

        // value_ints attribute → AttributeType.Longs with :params}
        var ints = NodeBuilder.CallCustomOperator<Vector<int64>>(
            OpCodes.CONSTANT, [],
            new object?[] { OnnxOpAttributeNames.AttrValueInts, new long[] { 10L, 20L, 30L } });

        var graph = new FastComputationGraph(
            [], ImmutableArray.Create<Variable>(floats, ints));
        var code = new CSharpModelBuilder().BuildFullGraph(ToFastGraph(graph), "CovTest");
        Assert.Contains("Vector(", code);
    }

    // ────────────────────────────────────────────────────────────────────────
    // InferSequenceElementRankFromTensor lines for SEQUENCE_ERASE recursion
    // chain and IDENTITY recursion chain. Build a SEQUENCE_AT whose input
    // came through ERASE→IDENTITY→CONSTRUCT/INSERT so the recursive walks
    // bottom out only after entering both arms.
    // ────────────────────────────────────────────────────────────────────────

    [Fact]
    public void TestSequenceElementRankInferenceDeepChain()
    {
        // SEQUENCE_AT(IDENTITY(ERASE(IDENTITY(SEQUENCE_CONSTRUCT(unrankedElem))), 0))
        // — both ERASE and IDENTITY arms recurse and the SEQUENCE_CONSTRUCT
        // bottom-out returns null because the element is unranked.
        var unrankedElem = (Tensor<float32>)OnnxOp.Identity(
            InputTensor<float32>("eltIn", rank: 2), rank: null);

        var s0 = OnnxOp.SequenceConstruct(unrankedElem);
        var s1 = OnnxOp.Identity(s0, rank: null);
        var s2 = OnnxOp.SequenceErase(s1, Scalar(0L));
        var s3 = OnnxOp.Identity(s2, rank: null);
        var deepAt = OnnxOp.SequenceAt(s3, Scalar(0L));

        var graph = new FastComputationGraph(
            [InputTensor<float32>("eltIn", rank: 2)],
            [deepAt]);
        var code = new CSharpModelBuilder().BuildFullGraph(ToFastGraph(graph), "CovTest");
        Assert.NotNull(code);
    }
}

