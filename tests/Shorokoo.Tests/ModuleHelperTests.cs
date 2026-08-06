using Shorokoo.Core.Nodes.Processors.Helpers;
using System.Reflection;

namespace Shorokoo.Tests;

/// <summary>
/// Direct coverage for <see cref="ModuleHelper"/> internals — the per-type branches in
/// <c>Format</c>, <c>Reformat</c>, <c>DefaultVariable</c>,
/// <c>ToSignatureStringWithOverride</c>, <c>InfosFromTouts</c> and the cache-hit replay
/// of <c>CreateTargetFunction</c> / <c>CreateFunctionSignature</c> that the AutoTester
/// roundtrip in <c>ModulesCoverageTests</c> does not reach.
/// </summary>
[Trait("Domain", "Modules")]
[Trait("Purpose", "Coverage")]
public class ModuleHelperCoverageTests
{
    [Fact]
    public void TestIsValueTupleDefaultVariableAndVariableRejectionAtEveryChokepoint()
    {
        Assert.True(ModuleHelper.IsValueTuple(typeof((int, string))));
        Assert.True(ModuleHelper.IsValueTuple<(float, double, int)>());
        Assert.False(ModuleHelper.IsValueTuple(typeof(int)));
        Assert.False(ModuleHelper.IsValueTuple(typeof(string)));
        Assert.False(ModuleHelper.IsValueTuple(typeof(List<int>)));
        Assert.False(ModuleHelper.IsValueTuple(typeof(Dictionary<int, string>)));

        Assert.Equal(DataStructure.Tensor, InternalGlobals.DefaultVariable(typeof(Tensor<float32>)).Structure());
        Assert.Equal(DataStructure.Optional, InternalGlobals.DefaultVariable(typeof(OptionalTensor<float32>)).Structure());
        Assert.Equal(DataStructure.Sequence, InternalGlobals.DefaultVariable(typeof(TensorSequence<float32>)).Structure());
        Assert.Equal(DataStructure.TensorStruct,
            InternalGlobals.DefaultVariable(typeof(TensorStruct<GenericPairStruct>)).Structure());
        Assert.Throws<UnsupportedDTypeException>(() => InternalGlobals.DefaultVariable(typeof(int)));

        ModuleHelper.RejectVariableParam(typeof(Scalar<float32>));
        ModuleHelper.RejectVariableParam(typeof((Scalar<float32>, Tensor<float32>)));
        Assert.Throws<InvalidTensorOperationException>(() => ModuleHelper.RejectVariableParam(typeof(Variable)));
        Assert.Throws<InvalidTensorOperationException>(() => ModuleHelper.RejectVariableParam(typeof((Variable, Scalar<float32>))));
        Assert.Throws<InvalidTensorOperationException>(() => ModuleHelper.RejectVariableParam(typeof(Variable[])));
        Assert.Throws<InvalidTensorOperationException>(() => InternalGlobals.DefaultVariable(typeof(Variable)));
        Assert.Throws<InvalidTensorOperationException>(() => ModuleHelper.ModuleParamInputBasedOn(typeof(Variable), InputType.ReadyInput, "x"));
        Assert.Throws<InvalidTensorOperationException>(() => ModuleHelper.CreateFunctionSignature([], [typeof(Variable)], [typeof(Scalar<float32>)]));
        Assert.Throws<InvalidTensorOperationException>(() => ModuleHelper.CreateFunctionSignature([], [typeof(Scalar<float32>)], [typeof(Variable)]));
    }

    [Fact]
    public void TestToSignatureStringWithOverrideTensorOptionalSequenceStructModelAndModuleArms()
    {
        Assert.Equal("float32#2",
            ModuleHelper.ToSignatureStringWithOverride(InputTensor<float32>("t", rank: 2), 2));
        Assert.Equal("float32",
            ModuleHelper.ToSignatureStringWithOverride((Tensor<float32>)OnnxOp.Identity(Scalar(1.0f), rank: null), -1));
        Assert.Equal("float32?",
            ModuleHelper.ToSignatureStringWithOverride(OptionalTensor<float32>(), -1));
        Assert.Contains("float32/seq",
            ModuleHelper.ToSignatureStringWithOverride(OnnxOp.SequenceEmpty(DType.Float32), -1));

        TensorStructFieldDef[] structFields =
            [new TensorStructFieldDef("CovHelperA", DataStructure.Tensor, rank: 1, DType.Float32)];
        var tensorStruct = InternalOp.TensorStructInput(
            DType.GetOrCreateForTensorStruct(new TensorStructDef(structFields, "CovHelperStruct")),
            InputType.ModelInput, targetFunction: null, defaultName: "ts");
        Assert.Contains("struct:CovHelperStruct",
            ModuleHelper.ToSignatureStringWithOverride(tensorStruct, null));

        Assert.StartsWith("[", ModuleHelper.ToSignatureStringWithOverride(
            ((IModuleParam)HypersLayer.Model(Scalar(1.0f), Scalar(0.0f))).ToVariable(), null));
        Assert.StartsWith("[", ModuleHelper.ToSignatureStringWithOverride(
            ((IModuleParam)new HypersLayerModule()).ToVariable(), null));
        Assert.Throws<InvalidTensorOperationException>(
            () => ((IModuleParam)new NonVariableModuleParam()).ToVariable());
    }

    [Fact]
    public void TestFormatAndReformatPerReturnTypeArms()
    {
        var v = InputScalar<float32>("v");
        Assert.Throws<InvalidTensorOperationException>(() => ModuleHelper.Format(null));
        Assert.Single(ModuleHelper.Format((IValue[])[v]));
        Assert.Single(ModuleHelper.Format(v));
        Assert.Single(ModuleHelper.Format((IModuleParam[])[v]));
        Assert.Equal(2, ModuleHelper.Format((v, v)).Length);
        Assert.Single(ModuleHelper.Format(new GenericPairRecord<float32, float32>(Scalar(1.0f), Scalar(2.0f))));
        Assert.Single(ModuleHelper.Format(TensorStruct<GenericPairStruct>(Scalar(3.0f), Scalar(4.0f))));
        Assert.Equal(2, ModuleHelper.Format(new List<IModuleParam> { v, v }).Length);
        Assert.Throws<InvalidTensorOperationException>(() => ModuleHelper.Format(42));

        var a = InputScalar<float32>("a");
        var b = InputScalar<float32>("b");
        Assert.NotNull(ModuleHelper.Reformat<Scalar<float32>>((Variable[])[a]));
        var tuple = ModuleHelper.Reformat<(Scalar<float32>, Scalar<float32>)>((Variable[])[a, b]);
        Assert.NotNull(tuple.Item1);
        Assert.NotNull(tuple.Item2);
        Assert.Equal(2, ModuleHelper.Reformat<Scalar<float32>[]>((Variable[])[a, b]).Length);
        Assert.Throws<UnsupportedDTypeException>(
            () => ModuleHelper.Reformat<List<IValue>>((Variable[])[a]));
        Assert.Throws<InvalidTensorOperationException>(
            () => ModuleHelper.Reformat<(Scalar<float32>, Scalar<float32>)>((Variable[])[a]));
    }

    [Fact]
    public void TestInfosFromToutsPerElementTypeArmsAndTupleSplit()
    {
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

        var (s7, _, _) = ModuleHelper.InfosFromTouts<TensorStruct<GenericPairStruct>>();
        Assert.Equal(DataStructure.TensorStruct, s7[0]);

        var (s6, d6, _) = ModuleHelper.InfosFromTouts<(Scalar<float32>, Vector<int64>, OptionalTensor<float32>)>();
        Assert.Equal(3, s6.Length);
        Assert.Equal(DType.Int64, d6[1]);

        Assert.Throws<UnsupportedDTypeException>(() => ModuleHelper.InfosFromTouts<((int, int), int)>());
        Assert.Throws<UnsupportedDTypeException>(() => ModuleHelper.InfosFromTouts<int>());
    }

    [Fact]
    public void TestFunctionSignatureCacheTargetFunctionGuardStructRecordBuildAndInputTypeArms()
    {
        Type[] hyperTypes = [typeof(Scalar<float32>)];
        Type[] inputTypes = [typeof(Tensor<float32>)];
        Type[] outputTypes = [typeof(Tensor<float32>)];

        Assert.Contains(">",
            ModuleHelper.CreateFunctionSignatureString(hyperTypes, inputTypes, outputTypes).moduleSignature);

        var fn1 = ModuleHelper.CreateFunctionSignature(hyperTypes, inputTypes, outputTypes);
        Assert.NotNull(fn1);
        Assert.Equal(FunctionType.ModuleSignature, fn1.FunctionType);
        var fn2 = ModuleHelper.CreateFunctionSignature(hyperTypes, inputTypes, outputTypes);
        Assert.Same(fn1, fn2);
        Assert.NotNull(ModuleHelper.CreateModule(fn1));

        Func<Tensor<float32>, Tensor<float32>> impl = DoubleTensor;
        Assert.Same(ModuleHelper.CreateTargetFunction(impl), ModuleHelper.CreateTargetFunction(impl));
        Assert.Throws<InvalidOperationException>(
            () => ModuleHelper.CreateTargetFunction(impl,
                isTrainableParamInitializer: true,
                isStateParamInitializer: true));

        Assert.NotEmpty(GenericRecordSum.ComputationGraph.ToInternal().Nodes);

        var extract = typeof(ModuleHelper)
            .GetMethod("ExtractGenericTypeArgsFromType", BindingFlags.NonPublic | BindingFlags.Static)!;
        object[] nonGenericArg = [typeof(int)];
        object[] genericArg = [typeof(Tensor<float32>)];
        object[] nestedGenericArg = [typeof(List<Tensor<float32>>)];
        Assert.Null(extract.Invoke(null, nonGenericArg));
        Assert.NotNull(extract.Invoke(null, genericArg));
        Assert.NotNull(extract.Invoke(null, nestedGenericArg));

        var parameters = typeof(ModuleHelperCoverageTests)
            .GetMethod(nameof(InputTypeHarness), BindingFlags.NonPublic | BindingFlags.Static)!
            .GetParameters();
        var getInputType = typeof(ModuleHelper)
            .GetMethod("GetInputType", BindingFlags.NonPublic | BindingFlags.Static)!;
        object[] hyperParamArg = [parameters[0]];
        object[] plainParamArg = [parameters[1]];
        Assert.Equal(InputType.Hyperparam, (InputType)getInputType.Invoke(null, hyperParamArg)!);
        Assert.Equal(InputType.ReadyInput, (InputType)getInputType.Invoke(null, plainParamArg)!);
    }

    private class NonVariableModuleParam : IModuleParam
    {
    }

    private static Tensor<float32> DoubleTensor(Tensor<float32> x) => x + x;

    private static void InputTypeHarness(
        [Hyper] Scalar<float32> hyperParam,
        Tensor<float32> plainParam)
    { }
}
