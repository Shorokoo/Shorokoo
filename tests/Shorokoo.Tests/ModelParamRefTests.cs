using Shorokoo.Modules.Initializers;
using Shorokoo.Modules.Layers;

namespace Shorokoo.Tests;

/// <summary>
/// Linear forward-correctness WITHOUT relying on tied init: the reference matmul uses the
/// model's ACTUAL weight (referenced by ModelId via <see cref="Shorokoo.Core.IModel.GetTrainableParam{T}"/>),
/// so it matches the layer regardless of how initialization is keyed.
/// </summary>
[Module]
public partial class LinearParamRefMatchesManualMatMul
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var model = Linear.Model(Scalar(4L), Scalar(true));
        var y = model.Call(x);

        var inFeatures = x.TShape[1..^0].Reduce(ReduceKind.Prod).Scalar();
        // Linear creates its weight first (relative model id [1]; parameters are numbered
        // from 1 within a model), then its bias ([2]).
        var w = model.GetTrainableParam<float32>([1], rank: 2);   // the layer's OWN weight [4, in]
        var yRef = x.Reshape([x.DimTensor(0), inFeatures]).MatMul(w.Transpose(1L, 0L));

        var diff = (y - yRef).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return diff < Scalar(1e-3f) * (Scalar(1f) + yRef.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar());
    }
}

/// <summary>Leaf (ModelB): a weight-only linear, y = x @ Wᵀ.</summary>
[Module]
public partial class NestedLeafLinear
{
    public static Tensor<float32> Inline(Tensor<float32> x, [Hyper] Scalar<int64> outF)
    {
        var inF = x.TShape[1..^0].Reduce(ReduceKind.Prod).Scalar();
        var w = KaimingUniform.Init([outF, inF]);
        return x.Reshape([x.DimTensor(0), inF]).MatMul(w.Transpose(1L, 0L));
    }
}

/// <summary>Wrapper (ModelA): just calls the leaf (ModelB); has no parameters of its own.</summary>
[Module]
public partial class NestedWrapper
{
    public static Tensor<float32> Inline(Tensor<float32> x, [Hyper] Scalar<int64> outF)
        => NestedLeafLinear.Model(outF).Call(x);
}

/// <summary>
/// Two levels deep: the test module builds ModelA (<see cref="NestedWrapper"/>), which builds
/// ModelB (<see cref="NestedLeafLinear"/>). The reference reaches ModelB's weight *through
/// ModelA's handle* by the nested path [1, 1] — ModelB is the 1st model created inside ModelA,
/// and its weight is the 1st parameter inside ModelB.
/// </summary>
[Module]
public partial class NestedParamRefMatchesManualMatMul
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var modelA = NestedWrapper.Model(Scalar(4L));
        var y = modelA.Call(x);

        var inF = x.TShape[1..^0].Reduce(ReduceKind.Prod).Scalar();
        var w = modelA.GetTrainableParam<float32>([1, 1], rank: 2);   // ModelB [1] → its weight [1]
        var yRef = x.Reshape([x.DimTensor(0), inF]).MatMul(w.Transpose(1L, 0L));

        var diff = (y - yRef).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return diff < Scalar(1e-3f) * (Scalar(1f) + yRef.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar());
    }
}

/// <summary>
/// A <b>rank-0</b> parameter reached by reference. A rank-0 initializer takes no shape, so the
/// reference node and the definition carry the same (empty) initializer inputs and nothing about
/// their inputs tells them apart. The reference is built BEFORE the call that defines the
/// parameter, and names the SECOND parameter of <see cref="Rank0BiasThenGainModel"/> — so a
/// resolution that let the reference stand in for the definition would fall back to the module's
/// first initializer and read the bias's 0 instead of the gain's 1.
/// </summary>
[Module]
public partial class Rank0ParamRefMatchesTheGainItScales
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var model = Rank0BiasThenGainModel.Model();
        var g = model.GetTrainableParam<float32>([2], rank: 0);   // reference precedes the definition
        var y = model.Call(x);

        var diff = (y - x * g).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        var seed = (g - Scalar(1f)).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return diff + seed < Scalar(1e-6f);
    }
}

[Trait("Domain", "Modules")]
[Trait("Purpose", "Coverage")]
public class ModelParamRefTests
{
    // RngConfig.Default forces per-parameter init (NOT the shared-key fixture): the reference
    // uses the model's actual weight, so it must match without same-shape params being tied.
    [Fact]
    public void TestParamRefFlatAndTwoLevelsDeepUnderPerParameterInit()
    {
        var x = TensorData(DType.Float32, [2L, 3L], 0.5f, -1f, 2f, 0.3f, -0.5f, 1.5f);
        Assert.True(AutoTest.AdvancedTestGraph<LinearParamRefMatchesManualMatMul>(
            hyperparamInputs: [], runtimeInputs: [x], rngConfig: RngConfig.Default));
        Assert.True(AutoTest.AdvancedTestGraph<NestedParamRefMatchesManualMatMul>(
            hyperparamInputs: [], runtimeInputs: [x], rngConfig: RngConfig.Default));
        Assert.True(AutoTest.AdvancedTestGraph<Rank0ParamRefMatchesTheGainItScales>(
            hyperparamInputs: [], runtimeInputs: [x], rngConfig: RngConfig.Default));
    }

    private static string ParamIdOf(ComputationGraph g)
    {
        var arch = g.ToConcreteArchitecture(g.FromOrderedInputs([TensorData([2L], 1f, 2f)]));
        return Assert.Single(arch.GetConcreteModelParamInfos().ParamInfos).ToShorokooIdString();
    }

    // Pins Shorokoo/Shorokoo#238: adding a read-only GetTrainableParam reference renames the
    // parameter it references (ParamRef_1 wins over the initializer-derived name), so a checkpoint
    // written by a model carrying the reference cannot be loaded by the same model without it.
    [Fact(Skip = "Shorokoo/Shorokoo#238: GetTrainableParam renames the parameter it references")]
    public void TestAParamRefDoesNotRenameTheParameterItReferences()
        => Assert.Equal(ParamIdOf(Rank0GainNoRefModel.ComputationGraph),
                        ParamIdOf(Rank0GainWithRefModel.ComputationGraph));
}
