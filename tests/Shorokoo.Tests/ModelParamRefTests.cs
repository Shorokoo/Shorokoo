using Shorokoo.Modules.Initializers;
using Shorokoo.Modules.Layers;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes.Processors.Training;

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

    private static string[] ParamIdsOf(ComputationGraph g)
    {
        var arch = g.ToConcreteArchitecture(g.FromOrderedInputs([TensorData([2L], 1f, 2f)]));
        string[] ids = [.. arch.GetConcreteModelParamInfos().ParamInfos.Select(x => x.ToShorokooIdString())];
        Assert.NotEmpty(ids);
        return ids;
    }

    private static void SameIds(ComputationGraph noRef, ComputationGraph withRef)
        => Assert.Equal(ParamIdsOf(noRef), ParamIdsOf(withRef));

    [Fact]
    public void TestAParamRefDoesNotRenameTheParameterItReferences()
    {
        SameIds(Rank0GainNoRefModel.ComputationGraph, Rank0GainWithRefModel.ComputationGraph);
        SameIds(Rank1GainNoRefModel.ComputationGraph, Rank1GainWithRefModel.ComputationGraph);
        SameIds(MixedDepthGainNoRefModel.ComputationGraph, MixedDepthGainWithRefsModel.ComputationGraph);
        SameIds(Rank1GainInLoopNoRefModel.ComputationGraph, Rank1GainRefInLoopModel.ComputationGraph);
    }

    // Goes through the public entry point rather than re-listing the passes it runs, so the
    // test cannot quietly stop guarding when that list changes.
    private static string[] TrainingStructNamesOf(ComputationGraph g, string inputName)
    {
        var arch = g.ToConcreteArchitecture(g.FromOrderedInputs([TensorData([2L], 1f, 2f)]));
        var training = TrainingGraphBuilder.PrepareForTrainingAsFast(
            arch.ToInternal(), SimpleSumSquaredLoss.ComputationGraph.ToInternal());
        var input = training.Inputs[training.InputUniqueNames.IndexOf(inputName)];
        var producer = training.Nodes.Single(n => n.FullOutputs.Values
            .Any(slot => slot.Any(k => k is FastTensorKey key && key.Equals(input))));
        var structDef = (TensorStructDef)producer.Attributes
            .GetDTypeVal(OnnxOpAttributeNames.AttrDtype)!.TensorStructDef!;
        return [.. structDef.Fields.Select(x => x.Name)];
    }

    private static string[] TrainingParamNamesOf(ComputationGraph g)
        => TrainingStructNamesOf(g, "trainable_params");

    private static void SameTrainingNames(ComputationGraph noRef, ComputationGraph withRef)
        => Assert.Equal(TrainingParamNamesOf(noRef), TrainingParamNamesOf(withRef));

    [Fact]
    public void TestAParamRefAddsNoParameterOnTheTrainingPath()
    {
        SameTrainingNames(Rank1GainNoRefModel.ComputationGraph, Rank1GainWithRefModel.ComputationGraph);
        SameTrainingNames(MixedDepthGainNoRefModel.ComputationGraph, MixedDepthGainWithRefsModel.ComputationGraph);
        SameTrainingNames(Rank1GainInLoopNoRefModel.ComputationGraph, Rank1GainRefInLoopModel.ComputationGraph);
    }

    [Fact]
    public void TestAParamRefWithNoDefinitionIsRejected()
        => Assert.Throws<InvalidOperationException>(() => ParamIdsOf(RefWithoutDefinitionModel.ComputationGraph));

    [Fact]
    public void TestAModelCalledTwiceIsOneTrainableParameterOnTheTrainingPath()
        => Assert.Equal(TrainingParamNamesOf(Rank1GainNoRefModel.ComputationGraph),
                        TrainingParamNamesOf(SharedModelCalledTwiceModel.ComputationGraph));

    [Fact]
    public void TestAStatefulModelCalledTwiceIsOneStateParameterOnTheTrainingPath()
    {
        Assert.Equal(TrainingStructNamesOf(StatefulGainNoRefModel.ComputationGraph, "model_state"),
                     TrainingStructNamesOf(StatefulGainCalledTwiceModel.ComputationGraph, "model_state"));
        Assert.Equal(TrainingParamNamesOf(StatefulGainNoRefModel.ComputationGraph),
                     TrainingParamNamesOf(StatefulGainCalledTwiceModel.ComputationGraph));
    }

    // A loop body's per-iteration parameters are separate parameters that share one generalized
    // identifier template, so each must keep its own struct field.
    [Fact]
    public void TestALoopsPerIterationParamsStayDistinctOnTheTrainingPath()
    {
        var names = TrainingParamNamesOf(Rank0ParamsInLoopModel.ComputationGraph);
        Assert.Equal(6, names.Length);
        Assert.Equal(6, names.Distinct().Count());
    }

    private static FastDiscoveredParamInfo NamedParam(string name)
    {
        var key = FastNodeKey.New();
        var node = new FastNode
        {
            Key = key,
            OpCode = InternalOpCodes.MODEL_PARAM,
            FullOutputs = { [""] = [new FastTensorKey(key, 0)] },
        };
        return new FastDiscoveredParamInfo(
            name, new FastTensorKey(key, 0), true, DType.Float32, 1, DataStructure.Tensor, node);
    }

    [Fact]
    public void TestTwoParametersCannotShareAStructFieldName()
    {
        Assert.Throws<InvalidOperationException>(() => FastBuildTrainableParamStructDefProcessor.Process(
            [NamedParam("Gain"), NamedParam("Gain")]));
        Assert.Equal(2, FastBuildTrainableParamStructDefProcessor.Process(
            [NamedParam("Gain"), NamedParam("Bias")]).Fields.Length);
    }

    // Pins Shorokoo/Shorokoo#301: a ModelSequence takes its module function from element 0, so
    // indexing any other element inlines element 0's body while naming its parameters after the
    // element that was indexed — one parameter here instead of two, and the wrong forward.
    [Fact(Skip = "Shorokoo/Shorokoo#301: a heterogeneous ModelSequence calls element 0's body whichever element is indexed")]
    public void TestAHeterogeneousModelSequenceCallsTheElementItIndexed()
        => SameIds(HyperScaledGainNoRefModel.ComputationGraph,
                   HeterogeneousHyperSequenceAtOneModel.ComputationGraph);

    [Fact]
    public void TestAModelPassedAsAHyperparameterKeepsItsTrainableParams()
        => SameIds(Rank1GainNoRefModel.ComputationGraph, HyperModelGainModel.ComputationGraph);

    [Fact]
    public void TestAModelPassedAsAHyperparameterSurvivesItsHostComingOutOfASequence()
        => SameIds(Rank1GainNoRefModel.ComputationGraph,
                   HyperModelGainFromSequenceModel.ComputationGraph);

    [Fact]
    public void TestAModelPassedAsAHyperparameterSurvivesItsHostComingOutOfADynamicallyIndexedSequence()
        => SameIds(GainFromDynamicSequenceModel.ComputationGraph,
                   HyperModelGainFromDynamicSequenceModel.ComputationGraph);

    [Fact]
    public void TestAModelPassedAsAHyperparameterSurvivesItsHostBeingAppendedToAnEmptySequence()
        => SameIds(GainFromDynamicSequenceModel.ComputationGraph,
                   HyperModelGainFromAppendedSequenceModel.ComputationGraph);

    // A ModelSequence names element 0's module whichever element is indexed, so the element
    // indexed and the module named disagree; the body spliced must be the one indexed.
    [Fact]
    public void TestAHeterogeneousModelSequenceCallsTheElementItIndexed()
        => SameIds(TwoParamGainNoRefModel.ComputationGraph,
                   HeterogeneousSequenceAtOneModel.ComputationGraph);

    [Fact]
    public void TestAHeterogeneousModelSequenceCallsTheElementItIndexedWhenThatElementHasAHyperparameter()
        => SameIds(HyperScaledGainNoRefModel.ComputationGraph,
                   HeterogeneousHyperSequenceAtOneModel.ComputationGraph);

    [Fact]
    public void TestAModelPassedAsAHyperparameterSurvivesItsHostOutlivingAnErasedSibling()
        => SameIds(GainFromErasedSequenceModel.ComputationGraph,
                   HyperModelGainFromErasedSequenceModel.ComputationGraph);
}
