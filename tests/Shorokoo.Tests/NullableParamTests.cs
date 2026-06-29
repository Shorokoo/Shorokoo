using System.Linq;
using Shorokoo.Runtime;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Inference;
using Shorokoo.Core.Utils;
using Shorokoo.Core.Factory.CSharpFactory;
using Shorokoo.Core.Nodes.NodeDefinitions;

namespace Shorokoo.Tests;

/// <summary>
/// Coverage for the source-generated nullable surface. An <c>OptionalTensor&lt;T&gt;</c> parameter
/// is exposed to callers as <c>Tensor&lt;T&gt;?</c> (omit / null = absent), and a
/// <c>[Hyper(default)]</c> scalar as a nullable, omittable parameter that falls back to its
/// attribute default. Execution drives present and absent cases via <see cref="OptionalTensorData"/>.
/// </summary>
[Trait("Domain", "Modules")]
[Trait("Purpose", "Coverage")]
public class NullableParamTests
{
    private static System.Collections.Immutable.ImmutableArray<Variable> InputsOf(FastComputationGraph graph)
        => FastComputationGraphConverter.BuildNodes(graph).inputs;

    private static byte[] Bytes(params float[] values) => TensorData([(long)values.Length], values).AccessRawMemory().ToArray();

    /// <summary>Concretizes (tensor shape hints stand in for optional inputs) and executes via the optional-aware QEE.</summary>
    private static byte[] RunWithOptionals(FastComputationGraph graph, TensorData[] shapeHints, params IData[] runtimeInputs)
    {
        var concrete = graph
            .ToConcreteArchitecture(graph.FromOrderedInputs([.. shapeHints]))
            .ToConcreteModel();
        var outputs = new QuickExecutionEngine().Execute(concrete, runtimeInputs);
        return ((TensorData)outputs[0]).AccessRawMemory().ToArray();
    }

    // ───────────────────────────── OptionalTensorData ─────────────────────────────

    [Fact]
    public void OptionalTensorData_SomeAndNone_CarryValueAndDType()
    {
        var v = TensorData([2L], new float[] { 1f, 2f });
        var some = OptionalTensorData.Some(v);
        Assert.True(some.HasValue);
        Assert.Same(v, some.Value);
        Assert.Equal(DType.Float32, some.DType);

        var none = OptionalTensorData.None<float32>();
        Assert.False(none.HasValue);
        Assert.Null(none.Value);
        Assert.Equal(DType.Float32, none.DType);
    }

    // ───────────────────────────── optional inputs ─────────────────────────────

    [Fact]
    public void OptionalInput_BecomesOptionalGraphInput()
    {
        var inputs = InputsOf(NullableBiasLayer.ComputationGraph);
        Assert.Equal(2, inputs.Length);
        Assert.Equal(1, inputs.Count(v => v.Structure() == DataStructure.Optional));
        Assert.Equal(1, inputs.Count(v => v.Structure() == DataStructure.Tensor));
    }

    [Fact]
    public void OptionalInput_PresentUsesValue_AbsentUsesDefault()
    {
        var x = TensorData([3L], new float[] { 1f, 2f, 3f });
        var bias = TensorData([3L], new float[] { 10f, 20f, 30f });
        // present → x + bias
        Assert.Equal(Bytes(11f, 22f, 33f),
            RunWithOptionals(NullableBiasLayer.ComputationGraph, [x, bias], x, OptionalTensorData.Some(bias)));
        // absent → x + zeros = x
        Assert.Equal(Bytes(1f, 2f, 3f),
            RunWithOptionals(NullableBiasLayer.ComputationGraph, [x, x], x, OptionalTensorData.None(DType.Float32)));
    }

    [Fact]
    public void TwoOptionalInputs_ResolveIndependently()
    {
        var x = TensorData([3L], new float[] { 5f, 6f, 7f });
        var bias = TensorData([3L], new float[] { 1f, 1f, 1f });
        // both absent → bias=zeros, scale=ones → x.
        Assert.Equal(Bytes(5f, 6f, 7f),
            RunWithOptionals(TwoNullableLayer.ComputationGraph, [x, x, x],
                x, OptionalTensorData.None(DType.Float32), OptionalTensorData.None(DType.Float32)));
        // bias present, scale absent (→ ones) → x*1 + bias.
        Assert.Equal(Bytes(6f, 7f, 8f),
            RunWithOptionals(TwoNullableLayer.ComputationGraph, [x, x, x],
                x, OptionalTensorData.Some(bias), OptionalTensorData.None(DType.Float32)));
    }

    [Fact]
    public void GeneratedSurface_ExposesOmittableNullableParameters()
    {
        // An OptionalTensor<T> input is exposed to callers as Tensor<T>? with a `= null` default,
        // so it can be omitted or passed null. Tensor<T> is now a value-struct handle, so Tensor<T>?
        // is Nullable<Tensor<T>>.
        var biasParam = typeof(NullableBiasLayer).GetMethod("Call")!.GetParameters().Single(p => p.Name == "bias");
        Assert.Equal(typeof(Tensor<float32>?), biasParam.ParameterType);
        Assert.True(biasParam.HasDefaultValue);

        // A [Hyper(default)] scalar is exposed on Model() as a nullable, omittable parameter.
        var factorParam = typeof(DefaultedHyperLayer).GetMethod("Model")!.GetParameters().Single(p => p.Name == "factor");
        Assert.Equal(typeof(Scalar<float32>?), factorParam.ParameterType);
        Assert.True(factorParam.HasDefaultValue);
    }

    // ───────────────────────────── defaulted hyperparameters ─────────────────────────────
    //
    // Each caller is a self-checking [Module]: its Inline calls a defaulted-hyper module — omitting
    // some or all defaults — and returns a Scalar<bit> that is false unless the expected value came
    // out. AdvancedTestGraph fails the test on a false bit and also roundtrips the graph through
    // ONNX, C# emission and the QuickExecutionEngine.

    private static readonly TensorData SampleX = TensorData([3L], new float[] { 1f, 2f, 3f });

    [Fact]
    public void DefaultedHyper_OmittedInCall_UsesAttributeDefault()
        => Assert.True(AutoTest.AdvancedTestGraph<DefaultedHyperOmitCheck>(
            hyperparamInputs: [], runtimeInputs: [SampleX]));

    [Fact]
    public void DefaultedHyper_SuppliedExplicitly_OverridesDefault()
        => Assert.True(AutoTest.AdvancedTestGraph<DefaultedHyperSupplyCheck>(
            hyperparamInputs: [], runtimeInputs: [SampleX]));

    [Fact]
    public void TwoDefaultedHypers_OmitAll_UsesBothDefaults()
        => Assert.True(AutoTest.AdvancedTestGraph<TwoDefaultedHyperOmitAllCheck>(
            hyperparamInputs: [], runtimeInputs: [SampleX]));

    [Fact]
    public void TwoDefaultedHypers_OmitSome_SupplyScale_DefaultBias()
        => Assert.True(AutoTest.AdvancedTestGraph<TwoDefaultedHyperOmitBiasCheck>(
            hyperparamInputs: [], runtimeInputs: [SampleX]));

    [Fact]
    public void TwoDefaultedHypers_OmitSome_NamedBias_DefaultScale()
        => Assert.True(AutoTest.AdvancedTestGraph<TwoDefaultedHyperOmitScaleCheck>(
            hyperparamInputs: [], runtimeInputs: [SampleX]));

    // ─────────────────── defaulted-hyper default value serialization ───────────────────

    [Fact]
    public void DefaultedHyper_RecordsDefaultValueOnHyperparamInputOnly()
    {
        // The [Hyper(3f)] default is recorded as declarative metadata on the hyperparameter input
        // node; ordinary inputs carry no default.
        var inputs = InputsOf(DefaultedHyperLayer.ComputationGraph);
        var hyper = inputs.Single(v => v.InputType == InputType.Hyperparam);
        var tensorInput = inputs.Single(v => v.InputType != InputType.Hyperparam);
        Assert.Equal(3f, hyper.HyperDefaultValue);
        Assert.Null(tensorInput.HyperDefaultValue);
    }

    [Fact]
    public void DefaultedHyper_DefaultValue_SurvivesOnnxBinaryRoundtrip()
    {
        var bytes = CompressedFormatUtils.SaveFastGraphToBinary(DefaultedHyperLayer.ComputationGraph, compressed: true);
        var roundtripped = CompressedFormatUtils.LoadFastGraphFromBinary(bytes, isCompressed: true);
        var hyper = InputsOf(roundtripped).Single(v => v.InputType == InputType.Hyperparam);
        Assert.Equal(3f, hyper.HyperDefaultValue);
    }

    [Fact]
    public void DefaultedHyper_DefaultValue_ReEmittedInCSharp()
    {
        // The defaulted hyper surfaces as a [Hyper] attribute when emitted as a sub-module function
        // (a caller's graph references it). The recorded default is re-written as [Hyper(3f)] rather
        // than a bare [Hyper].
        var code = new CSharpModelBuilder().BuildFullGraph(DefaultedHyperSupplyCheck.ComputationGraph, "DefaultedHyperRoundtrip");
        Assert.Contains("[Hyper(3f)]", code);
    }

    // ───────────────────────────── implicit cast ─────────────────────────────

    [Fact]
    public void OptionalTensor_ImplicitlyCastsToNullableTensor()
    {
        OptionalTensor<float32> present = OptionalTensor<float32>(Vector(1f, 2f, 3f));
        Tensor<float32>? asTensor = present;     // implicit OptionalTensor<T> -> Tensor<T>?
        Assert.NotNull(asTensor);
        var value = ComputeContext.Default.Eval([asTensor!])[0];
        Assert.Equal(Bytes(1f, 2f, 3f), value.AccessRawMemory().ToArray());

        OptionalTensor<float32>? nullOptional = null;
        Tensor<float32>? fromNull = nullOptional; // a C#-null optional maps to null
        Assert.Null(fromNull);
    }

    // ───────────────────────── ONNX / C# / QEE roundtrips ─────────────────────────

    /// <summary>A present optional (fed as a plain tensor) survives ONNX export, C# emission and QEE
    /// execution identically — the OptionalTensor input + If lower and re-run consistently.</summary>
    [Fact]
    public void OptionalInput_PresentValue_RoundtripsThroughOnnxCsAndQee()
        => Assert.True(AutoTest.AdvancedTestGraph<NullableBiasLayer>(hyperparamInputs: [],
            runtimeInputs: [TensorData([2L, 3L], new float[] { 1f, 2f, 3f, 4f, 5f, 6f }),
                            TensorData([2L, 3L], new float[] { 10f, 20f, 30f, 40f, 50f, 60f })]));

    /// <summary>A present bias passed through the generated sub-module <c>Call</c> roundtrips through
    /// ONNX, C# and QEE (NullableBiasPresentCheck self-checks the value is x + bias).</summary>
    [Fact]
    public void OptionalSubModuleCall_PresentValue_Roundtrips()
        => Assert.True(AutoTest.AdvancedTestGraph<NullableBiasPresentCheck>(hyperparamInputs: [],
            runtimeInputs: [TensorData([3L], new float[] { 1f, 2f, 3f }),
                            TensorData([3L], new float[] { 10f, 20f, 30f })]));

    [Fact]
    public void OptionalInputWithTrainableParam_PresentRoundtrips_AbsentUsesDefault()
    {
        var x = TensorData([2L, 2L], new float[] { 1f, 2f, 3f, 4f });
        var bias = TensorData([2L, 2L], new float[] { 5f, 6f, 7f, 8f });
        // present (plain tensor): x*w + bias with w=ones → x + bias.
        Assert.True(AutoTest.AdvancedTestGraph<NullableTrainableBiasLayer>(
            hyperparamInputs: [], runtimeInputs: [x, bias]));
        // absent: x*w + 0 = x.
        Assert.Equal(Bytes(1f, 2f, 3f, 4f),
            RunWithOptionals(NullableTrainableBiasLayer.ComputationGraph, [x, x], x, OptionalTensorData.None(DType.Float32)));
    }

    // ───────────────────────── ONNX-Runtime absent-optional guidance ─────────────────────────

    [Fact]
    public void AbsentOptionalData_ToOnnxRuntime_ThrowsWithGuidance()
    {
        var x = TensorData([3L], new float[] { 1f, 2f, 3f });
        var concrete = NullableBiasLayer.ComputationGraph
            .ToConcreteArchitecture(NullableBiasLayer.ComputationGraph.FromOrderedInputs([x, x]))
            .ToConcreteModel();
        var ex = Assert.Throws<InvalidTensorOperationException>(() =>
            ComputeContext.Default.Execute(concrete, x, OptionalTensorData.None(DType.Float32)));
        Assert.Contains("QuickExecutionEngine", ex.Message);
    }

    // ───────────────────────── value-struct handle (OptionalTensor) ─────────────────────────
    // OptionalTensor<T> is now a value-type handle wrapping a Variable; these
    // pin the conversion web, graph-identity preservation across conversion, and default materialisation.

    [Fact]
    public void OptionalTensorHandle_IsValueType()
        => Assert.True(typeof(OptionalTensor<float32>).IsValueType);

    /// <summary>Unwrapping the handle to its immutable and re-wrapping returns the same graph value
    /// (same <see cref="IValue.Key"/>); boxing the handle as <see cref="IValue"/> keeps that identity.</summary>
    [Fact]
    public void OptionalTensorHandle_WrapUnwrap_PreservesGraphIdentity()
    {
        OptionalTensor<float32> handle = OptionalTensor<float32>(Vector(1f, 2f, 3f));
        Variable imm = handle;   // unwrap (implicit)
        OptionalTensor<float32> rewrapped = imm;                       // wrap (implicit)
        Assert.Equal(imm.Key, ((IValue)rewrapped).Key);
        Assert.Equal(imm.Key, ((IValue)handle).Key);                // boxing keeps Key
    }

    /// <summary>A defaulted handle (<c>inner == null</c>) reports as an optional and lazily
    /// materialises an absent optional graph value on first member access.</summary>
    [Fact]
    public void OptionalTensorHandle_Default_MaterialisesAbsentOptional()
    {
        OptionalTensor<float32> defaulted = default;
        var asVar = (IValue)defaulted;
        Assert.Equal(DataStructure.Optional, asVar.Structure());
        Assert.NotNull(asVar.OwningNode);                              // forces default materialisation
    }
}
