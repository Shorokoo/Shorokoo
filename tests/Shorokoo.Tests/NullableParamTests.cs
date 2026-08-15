using System.Linq;
using Shorokoo.Runtime;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Inference;
using Shorokoo.Core.Factory.CSharpFactory;

namespace Shorokoo.Tests;

/// <summary>
/// Coverage for the source-generated nullable surface: an <c>OptionalTensor&lt;T&gt;</c> parameter
/// exposed as <c>Tensor&lt;T&gt;?</c> (omit / null = absent), and a <c>[Hyper(default)]</c> scalar
/// exposed as a nullable, omittable parameter falling back to its attribute default.
/// </summary>
[Trait("Domain", "Modules")]
[Trait("Purpose", "Coverage")]
public class NullableParamTests
{
    private static System.Collections.Immutable.ImmutableArray<Variable> InputsOf(ComputationGraph graph)
        => InternalComputationGraphConverter.BuildNodes(graph.ToInternal()).inputs;

    private static byte[] Bytes(params float[] values) => TensorData([(long)values.Length], values).AccessRawMemory().ToArray();

    private static byte[] RunWithOptionals(ComputationGraph graph, TensorData[] shapeHints, params IData[] runtimeInputs)
    {
        var concrete = graph
            .ToConcreteArchitecture(graph.FromOrderedInputs([.. shapeHints]))
            .ToConcreteModel();
        var outputs = new QuickExecutionEngine().Execute(concrete.ToInternal(), runtimeInputs);
        return ((TensorData)outputs[0]).AccessRawMemory().ToArray();
    }

    private static readonly TensorData SampleX = TensorData([3L], 1f, 2f, 3f);

    [Fact]
    public void TestOptionalTensorDataAndHandleSurface()
    {
        var v = TensorData([2L], 1f, 2f);
        var some = OptionalTensorData.Some(v);
        Assert.True(some.HasValue);
        Assert.Same(v, some.Value);
        Assert.Equal(DType.Float32, some.DType);

        var none = OptionalTensorData.None<float32>();
        Assert.False(none.HasValue);
        Assert.Null(none.Value);
        Assert.Equal(DType.Float32, none.DType);

        Assert.True(typeof(OptionalTensor<float32>).IsValueType);

        // Unwrapping the handle to its immutable and re-wrapping keeps the same graph value.
        OptionalTensor<float32> handle = OptionalTensor<float32>(Vector(1f, 2f, 3f));
        Variable imm = handle;
        OptionalTensor<float32> rewrapped = imm;
        Assert.Equal(imm.Key, ((IValue)rewrapped).Key);
        Assert.Equal(imm.Key, ((IValue)handle).Key);

        // A defaulted handle lazily materialises an absent optional on first member access.
        OptionalTensor<float32> defaulted = default;
        var asVar = (IValue)defaulted;
        Assert.Equal(DataStructure.Optional, asVar.Structure());
        Assert.NotNull(asVar.OwningNode);
    }

    [Fact]
    public void TestGeneratedSurfaceExposesOmittableNullableParameters()
    {
        var biasParam = typeof(NullableBiasLayer).GetMethod("Call")!.GetParameters().Single(p => p.Name == "bias");
        Assert.Equal(typeof(Tensor<float32>?), biasParam.ParameterType);
        Assert.True(biasParam.HasDefaultValue);

        var factorParam = typeof(DefaultedHyperLayer).GetMethod("Model")!.GetParameters().Single(p => p.Name == "factor");
        Assert.Equal(typeof(Scalar<float32>?), factorParam.ParameterType);
        Assert.True(factorParam.HasDefaultValue);
    }

    [Fact]
    public void TestOptionalInputResolution()
    {
        var inputs = InputsOf(NullableBiasLayer.ComputationGraph);
        Assert.Equal(2, inputs.Length);
        Assert.Equal(1, inputs.Count(v => v.Structure() == DataStructure.Optional));
        Assert.Equal(1, inputs.Count(v => v.Structure() == DataStructure.Tensor));

        var x = TensorData([3L], 1f, 2f, 3f);
        var bias = TensorData([3L], 10f, 20f, 30f);
        Assert.Equal(Bytes(11f, 22f, 33f),
            RunWithOptionals(NullableBiasLayer.ComputationGraph, [x, bias], x, OptionalTensorData.Some(bias)));
        Assert.Equal(Bytes(1f, 2f, 3f),
            RunWithOptionals(NullableBiasLayer.ComputationGraph, [x, x], x, OptionalTensorData.None(DType.Float32)));

        // Two optionals resolve independently: both absent → bias=zeros, scale=ones → x.
        var x2 = TensorData([3L], 5f, 6f, 7f);
        var ones = TensorData([3L], 1f, 1f, 1f);
        Assert.Equal(Bytes(5f, 6f, 7f),
            RunWithOptionals(TwoNullableLayer.ComputationGraph, [x2, x2, x2],
                x2, OptionalTensorData.None(DType.Float32), OptionalTensorData.None(DType.Float32)));
        Assert.Equal(Bytes(6f, 7f, 8f),
            RunWithOptionals(TwoNullableLayer.ComputationGraph, [x2, x2, x2],
                x2, OptionalTensorData.Some(ones), OptionalTensorData.None(DType.Float32)));

        // A present optional roundtrips through ONNX, C# emission and QEE — fed directly and
        // through the generated sub-module Call.
        Assert.True(AutoTest.AdvancedTestGraph<NullableBiasLayer>(hyperparamInputs: [],
            runtimeInputs: [TensorData([2L, 3L], 1f, 2f, 3f, 4f, 5f, 6f),
                            TensorData([2L, 3L], 10f, 20f, 30f, 40f, 50f, 60f)],
                            expected: [11.0, 22.0, 33.0, 44.0, 55.0, 66.0]));
        Assert.True(AutoTest.AdvancedTestGraph<NullableBiasPresentCheck>(hyperparamInputs: [],
            runtimeInputs: [TensorData([3L], 1f, 2f, 3f), TensorData([3L], 10f, 20f, 30f)]));
    }

    [Fact]
    public void TestOptionalTensorImplicitlyCastsToNullableTensor()
    {
        OptionalTensor<float32> present = OptionalTensor<float32>(Vector(1f, 2f, 3f));
        Tensor<float32>? asTensor = present;
        Assert.NotNull(asTensor);
        var value = ComputeContext.Default.Eval([asTensor!])[0];
        Assert.Equal(Bytes(1f, 2f, 3f), value.AccessRawMemory().ToArray());

        OptionalTensor<float32>? nullOptional = null;
        Tensor<float32>? fromNull = nullOptional;
        Assert.Null(fromNull);
    }

    // Each caller below is a self-checking [Module]: its Inline calls a defaulted-hyper module —
    // omitting some or all defaults — and returns a Scalar<bit> that is false unless the expected
    // value came out. AdvancedTestGraph fails on a false bit and roundtrips through ONNX/C#/QEE.
    [Fact]
    public void TestDefaultedHypers()
    {
        Assert.True(AutoTest.AdvancedTestGraph<DefaultedHyperOmitCheck>(hyperparamInputs: [], runtimeInputs: [SampleX]));
        Assert.True(AutoTest.AdvancedTestGraph<DefaultedIntHyperOmitCheck>(hyperparamInputs: [], runtimeInputs: [SampleX]));
        Assert.True(AutoTest.AdvancedTestGraph<DefaultedHyperSupplyCheck>(hyperparamInputs: [], runtimeInputs: [SampleX]));
        Assert.True(AutoTest.AdvancedTestGraph<TwoDefaultedHyperOmitAllCheck>(hyperparamInputs: [], runtimeInputs: [SampleX]));
        Assert.True(AutoTest.AdvancedTestGraph<TwoDefaultedHyperOmitBiasCheck>(hyperparamInputs: [], runtimeInputs: [SampleX]));
        Assert.True(AutoTest.AdvancedTestGraph<TwoDefaultedHyperOmitScaleCheck>(hyperparamInputs: [], runtimeInputs: [SampleX]));
    }

    [Fact]
    public void TestDefaultedHyperSerialization()
    {
        // The [Hyper(3f)] default is declarative metadata on the hyperparameter input node only.
        var inputs = InputsOf(DefaultedHyperLayer.ComputationGraph);
        Assert.Equal("3", inputs.Single(v => v.InputType == InputType.Hyperparam).HyperDefaultValue);
        Assert.Null(inputs.Single(v => v.InputType != InputType.Hyperparam).HyperDefaultValue);

        var bytes = CompressedFormatUtils.SaveFastGraphToBinary(DefaultedHyperLayer.ComputationGraph, compressed: true);
        var roundtripped = CompressedFormatUtils.LoadFastGraphFromBinary(bytes);
        Assert.Equal("3", InputsOf(roundtripped).Single(v => v.InputType == InputType.Hyperparam).HyperDefaultValue);

        // Re-emitted as [Hyper(3f)] rather than a bare [Hyper] when written as a sub-module function.
        var code = new CSharpModelBuilder().BuildFullGraph(
            DefaultedHyperSupplyCheck.ComputationGraph.ToInternal(), "DefaultedHyperRoundtrip");
        Assert.Contains("[Hyper(3f)]", code);

        // A non-float32 default is recorded and re-emitted at its declared dtype, not through a float.
        var intInputs = InputsOf(DefaultedIntHyperLayer.ComputationGraph);
        var intHyper = intInputs.Single(v => v.InputType == InputType.Hyperparam);
        Assert.Equal("3", intHyper.HyperDefaultValue);
        Assert.Equal(DType.Int32, intHyper.Type);
        Assert.Contains("[Hyper(3)]", new CSharpModelBuilder().BuildFullGraph(
            DefaultedIntHyperOmitCheck.ComputationGraph.ToInternal(), "DefaultedIntHyperRoundtrip"));
    }

    [Fact]
    public void TestOptionalInputWithTrainableParamPresentRoundtripsAndAbsentUsesDefault()
    {
        var x = TensorData([2L, 2L], 1f, 2f, 3f, 4f);
        var bias = TensorData([2L, 2L], 5f, 6f, 7f, 8f);
        Assert.True(AutoTest.AdvancedTestGraph<NullableTrainableBiasLayer>(
            hyperparamInputs: [], runtimeInputs: [x, bias],
            expected: [6.0, 8.0, 10.0, 12.0]));
        Assert.Equal(Bytes(1f, 2f, 3f, 4f),
            RunWithOptionals(NullableTrainableBiasLayer.ComputationGraph, [x, x], x, OptionalTensorData.None(DType.Float32)));
    }

    [Fact]
    public void TestAbsentOptionalDataToOnnxRuntimeThrowsWithGuidance()
    {
        var x = TensorData([3L], 1f, 2f, 3f);
        var concrete = NullableBiasLayer.ComputationGraph
            .ToConcreteArchitecture(NullableBiasLayer.ComputationGraph.FromOrderedInputs([x, x]))
            .ToConcreteModel();
        var ex = Assert.Throws<InvalidTensorOperationException>(() =>
            ComputeContext.Default.Execute(concrete, x, OptionalTensorData.None(DType.Float32)));
        Assert.Contains("QuickExecutionEngine", ex.Message);
    }
}
