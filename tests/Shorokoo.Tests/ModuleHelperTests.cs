using Shorokoo.Core.Nodes.Processors.Helpers;
using System.Reflection;

namespace Shorokoo.Tests;

/// <summary>
/// Direct unit-test coverage for <see cref="ModuleHelper"/> internals. The existing
/// <c>ModulesCoverageTests</c> drives <see cref="ModuleHelper"/> indirectly through
/// the AutoTester roundtrip, which exercises the common Module construction path
/// but doesn't reach the per-type branches in <c>Format</c>, <c>Reformat</c>,
/// <c>DefaultVariable</c>, <c>ToSignatureStringWithOverride</c>, <c>InfosFromTouts</c>,
/// or the cache-hit replay of <c>CreateTargetFunction</c> /
/// <c>CreateFunctionSignature</c>.
/// </summary>
[Trait("Domain", "Modules")]
[Trait("Purpose", "Coverage")]
public class ModuleHelperCoverageTests
{
    [Fact]
    public void TestModuleHelperCoverage()
    {
        // ──────────────────────────────────────────────────────────────────
        // IsValueTuple — generic vs non-generic vs generic-but-not-ValueTuple
        // ──────────────────────────────────────────────────────────────────
        Assert.True(ModuleHelper.IsValueTuple(typeof((int, string))));
        Assert.True(ModuleHelper.IsValueTuple<(float, double, int)>());
        Assert.False(ModuleHelper.IsValueTuple(typeof(int)));
        Assert.False(ModuleHelper.IsValueTuple(typeof(string)));
        Assert.False(ModuleHelper.IsValueTuple(typeof(List<int>)));
        Assert.False(ModuleHelper.IsValueTuple(typeof(Dictionary<int, string>)));

        // ──────────────────────────────────────────────────────────────────
        // DefaultVariable — one branch per supported variable shape + unsupported throw
        // ──────────────────────────────────────────────────────────────────
        Assert.IsAssignableFrom<ITensor>(ModuleHelper.DefaultVariable(typeof(Tensor<float32>)));
        Assert.IsAssignableFrom<IOptionalTensor>(ModuleHelper.DefaultVariable(typeof(OptionalTensor<float32>)));
        Assert.IsAssignableFrom<ITensorSequence>(ModuleHelper.DefaultVariable(typeof(TensorSequence<float32>)));
        // ITensorStruct branch (lines 170-181): pass a concrete TensorStruct type.
        Assert.IsAssignableFrom<ITensorStruct>(
            ModuleHelper.DefaultVariable(typeof(TensorStruct<GenericPairStruct>)));
        Assert.Throws<UnsupportedDTypeException>(() => ModuleHelper.DefaultVariable(typeof(int)));

        // ──────────────────────────────────────────────────────────────────
        // ToSignatureStringWithOverride — each variable-type arm
        // ──────────────────────────────────────────────────────────────────
        var tensor = InputTensor<float32>("t", rank: 2);
        Assert.Equal("float32#2", ModuleHelper.ToSignatureStringWithOverride(tensor, 2));

        var unrankedTensor = (Tensor<float32>)OnnxOp.Identity(Scalar(1.0f), rank: null);
        Assert.Equal("float32", ModuleHelper.ToSignatureStringWithOverride(unrankedTensor, -1));

        var opt = OptionalTensor<float32>();
        Assert.Equal("float32?", ModuleHelper.ToSignatureStringWithOverride(opt, -1));

        var seq = OnnxOp.SequenceEmpty(DType.Float32);
        Assert.Contains("float32/seq", ModuleHelper.ToSignatureStringWithOverride((ITensorSequence)seq, -1));

        // ITensorStruct arm (lines 128-132).
        var structFields = new[]
        {
            new TensorStructFieldDef("CovHelperA", DataStructure.Tensor, rank: 1, DType.Float32),
        };
        var structDef = new TensorStructDef(structFields, "CovHelperStruct");
        var structDType = DType.GetOrCreateForTensorStruct(structDef);
        var tensorStruct = InternalOp.TensorStructInput(
            structDType, InputType.ModelInput, targetFunction: null, defaultName: "ts");
        Assert.Contains("struct:CovHelperStruct",
            ModuleHelper.ToSignatureStringWithOverride(tensorStruct, null));

        // IModel arm (line 111) + Scalar<IModelVarType> arm (line 117).
        var hypersModel = HypersLayer.Model(Scalar(1.0f), Scalar(0.0f));
        Assert.StartsWith("[", ModuleHelper.ToSignatureStringWithOverride(hypersModel, null));

        // IModule arm (line 114) + Scalar<IModuleVarType> arm (line 120).
        var hypersModule = new HypersLayerModule();
        Assert.StartsWith("[", ModuleHelper.ToSignatureStringWithOverride(hypersModule, null));

        Assert.Throws<InvalidTensorOperationException>(
            () => ModuleHelper.ToSignatureStringWithOverride(new NonVariableModuleParam(), null));

        // ──────────────────────────────────────────────────────────────────
        // Format — every return-value-type arm
        // ──────────────────────────────────────────────────────────────────
        var v = InputScalar<float32>("v");
        Assert.Throws<InvalidTensorOperationException>(() => ModuleHelper.Format(null));
        Assert.Single(ModuleHelper.Format(new IVariable[] { v }));
        Assert.Single(ModuleHelper.Format(v));
        // IModuleParam[] arm (line 408-409).
        Assert.Single(ModuleHelper.Format(new IModuleParam[] { v }));
        Assert.Equal(2, ModuleHelper.Format((v, v)).Length);

        // IStruct record arm (lines 423-440).
        var record = new GenericPairRecord<float32, float32>(Scalar(1.0f), Scalar(2.0f));
        var recordVars = ModuleHelper.Format(record);
        Assert.Single(recordVars);

        // ITensorStructProxy arm (line 417-418).
        var proxy = TensorStruct<GenericPairStruct>(Scalar(3.0f), Scalar(4.0f));
        Assert.Single(ModuleHelper.Format(proxy));

        // IEnumerable arm (line 442-443).
        Assert.Equal(2, ModuleHelper.Format(new List<IModuleParam> { v, v }).Length);

        Assert.Throws<InvalidTensorOperationException>(() => ModuleHelper.Format(42));

        // ──────────────────────────────────────────────────────────────────
        // Reformat<T> — array, ValueTuple, IVariable, throw branches
        // ──────────────────────────────────────────────────────────────────
        var a = InputScalar<float32>("a");
        var b = InputScalar<float32>("b");
        Assert.NotNull(ModuleHelper.Reformat<Scalar<float32>>(new IVariable[] { a }));

        var tuple = ModuleHelper.Reformat<(Scalar<float32>, Scalar<float32>)>(new IVariable[] { a, b });
        Assert.NotNull(tuple.Item1);
        Assert.NotNull(tuple.Item2);

        var array = ModuleHelper.Reformat<Scalar<float32>[]>(new IVariable[] { a, b });
        Assert.Equal(2, array.Length);

        Assert.Throws<UnsupportedDTypeException>(
            () => ModuleHelper.Reformat<List<IVariable>>(new IVariable[] { a }));
        Assert.Throws<InvalidTensorOperationException>(
            () => ModuleHelper.Reformat<(Scalar<float32>, Scalar<float32>)>(new IVariable[] { a }));

        // ──────────────────────────────────────────────────────────────────
        // InfosFromTouts — each per-element-type arm + tuple / non-tuple split
        // ──────────────────────────────────────────────────────────────────
        var (s1, d1, r1) = ModuleHelper.InfosFromTouts<Tensor<float32>>();
        Assert.Single(s1);
        Assert.Equal(DataStructure.Tensor, s1[0]);
        Assert.Equal(DType.Float32, d1[0]);
        Assert.Equal(-1, r1[0]);

        var (_, _, r2) = ModuleHelper.InfosFromTouts<Vector<float32>>();
        Assert.Equal(1, r2[0]);

        var (_, _, r3) = ModuleHelper.InfosFromTouts<Scalar<float32>>();
        Assert.Equal(0, r3[0]);

        var (s4, _, _) = ModuleHelper.InfosFromTouts<OptionalTensor<float32>>();
        Assert.Equal(DataStructure.Optional, s4[0]);

        var (s5, _, _) = ModuleHelper.InfosFromTouts<TensorSequence<float32>>();
        Assert.Equal(DataStructure.Sequence, s5[0]);

        // ITensorStruct arm (lines 512-525).
        var (s7, _, _) = ModuleHelper.InfosFromTouts<TensorStruct<GenericPairStruct>>();
        Assert.Equal(DataStructure.TensorStruct, s7[0]);

        var (s6, d6, _) = ModuleHelper.InfosFromTouts<(Scalar<float32>, Vector<int64>, OptionalTensor<float32>)>();
        Assert.Equal(3, s6.Length);
        Assert.Equal(DType.Int64, d6[1]);

        Assert.Throws<UnsupportedDTypeException>(
            () => ModuleHelper.InfosFromTouts<((int, int), int)>());
        Assert.Throws<UnsupportedDTypeException>(
            () => ModuleHelper.InfosFromTouts<int>());

        // ──────────────────────────────────────────────────────────────────
        // CreateFunctionSignatureString / CreateFunctionSignature / CreateModule
        // ──────────────────────────────────────────────────────────────────
        var sig1 = ModuleHelper.CreateFunctionSignatureString(
            new[] { typeof(Scalar<float32>) },
            new[] { typeof(Tensor<float32>) },
            new[] { typeof(Tensor<float32>) });
        Assert.Contains(">", sig1.moduleSignature);

        var fn1 = ModuleHelper.CreateFunctionSignature(
            new[] { typeof(Scalar<float32>) },
            new[] { typeof(Tensor<float32>) },
            new[] { typeof(Tensor<float32>) });
        Assert.NotNull(fn1);
        Assert.Equal(FunctionType.ModuleSignature, fn1.FunctionType);

        // Cache-hit replay path.
        var fn2 = ModuleHelper.CreateFunctionSignature(
            new[] { typeof(Scalar<float32>) },
            new[] { typeof(Tensor<float32>) },
            new[] { typeof(Tensor<float32>) });
        Assert.Same(fn1, fn2);

        Assert.NotNull(ModuleHelper.CreateModule(fn1));

        // ──────────────────────────────────────────────────────────────────
        // CreateTargetFunction — cache-hit replay + mutually-exclusive guard
        // ──────────────────────────────────────────────────────────────────
        Func<Tensor<float32>, Tensor<float32>> impl = DoubleTensor;
        var tfn1 = ModuleHelper.CreateTargetFunction(impl);
        var tfn2 = ModuleHelper.CreateTargetFunction(impl);
        Assert.Same(tfn1, tfn2);

        // Mutual-exclusivity throw (line 231-232).
        Assert.Throws<InvalidOperationException>(
            () => ModuleHelper.CreateTargetFunction(impl,
                isTrainableParamInitializer: true,
                isStateParamInitializer: true));

        // ──────────────────────────────────────────────────────────────────
        // ConstructStructFromTensorStruct (private) — drive via Format-IStruct
        // and via GenericRecordSum's ComputationGraph build which calls
        // InvokeAndFormat with an IStruct record parameter (lines 354-388 +
        // InvokeAndFormat IStruct-record arm at line 336-338).
        // ──────────────────────────────────────────────────────────────────
        var graph = GenericRecordSum.ComputationGraph;
        Assert.NotNull(graph);
        Assert.NotEmpty(graph.Nodes);

        // ──────────────────────────────────────────────────────────────────
        // ExtractGenericTypeArgsFromType — non-generic null + generic non-null
        // + nested-generic recursion through TryExtractDTypeFromType.
        // ──────────────────────────────────────────────────────────────────
        var extractMI = typeof(ModuleHelper)
            .GetMethod("ExtractGenericTypeArgsFromType",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static)!;
        Assert.Null(extractMI.Invoke(null, new object[] { typeof(int) }));
        Assert.NotNull(extractMI.Invoke(null, new object[] { typeof(Tensor<float32>) }));
        // Nested generic: List<Tensor<float32>> — typeArgs = [Tensor<float32>],
        // which is itself generic, driving the TryExtractDTypeFromType
        // recursion arm (line 616-624).
        Assert.NotNull(extractMI.Invoke(null, new object[] { typeof(List<Tensor<float32>>) }));

        // ──────────────────────────────────────────────────────────────────
        // GetInputType extension — drive each attribute-check arm
        // ──────────────────────────────────────────────────────────────────
        var harness = typeof(ModuleHelperCoverageTests).GetMethod(
            nameof(InputTypeHarness),
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static)!;
        var parameters = harness.GetParameters();
        var getInputType = typeof(ModuleHelper)
            .GetMethod("GetInputType",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static)!;
        Assert.Equal(InputType.Hyperparam, (InputType)getInputType.Invoke(null, new object[] { parameters[0] })!);
        Assert.Equal(InputType.ReadyInput, (InputType)getInputType.Invoke(null, new object[] { parameters[1] })!);
    }

    private class NonVariableModuleParam : IModuleParam
    {
        public IVariable ToVariable() => throw new NotImplementedException();
    }

    private static Tensor<float32> DoubleTensor(Tensor<float32> x) => x + x;

    private static void InputTypeHarness(
        [Hyper] Scalar<float32> hyperParam,
        Tensor<float32> plainParam)
    { }
}
