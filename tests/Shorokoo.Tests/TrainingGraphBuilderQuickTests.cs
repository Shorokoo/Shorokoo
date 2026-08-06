using Shorokoo.Core.Nodes.Processors.Helpers;
using System.Collections.Immutable;

namespace Shorokoo.Tests;

/// <summary>Sum of squared differences: sum((pred - target)^2). Namespace level for the generator.</summary>
[Module]
public partial class SimpleSumSquaredLoss
{
    public static Scalar<float32> Inline(Tensor<float32> predictions, Tensor<float32> targets)
    {
        var diff = predictions - targets;
        var squared = diff * diff;
        var reduced = (Tensor<float32>)OnnxOp.ReduceSum(squared, keepdims: false);
        return reduced.Scalar();
    }
}

/// <summary>
/// <c>TrainingGraphBuilder.PrepareForTrainingAsFast</c>: a model graph composed with a loss and
/// automatic differentiation yields a high-level training graph whose inputs cover model input,
/// targets and the trainable param struct, whose outputs cover loss and gradient struct, and
/// whose AUTO_GRAD nodes are not yet lowered.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class TrainingGraphBuilderQuickTests
{
    private static void AssertTrainingGraphStructure(InternalComputationGraph trainingGraph)
    {
        Assert.True(trainingGraph.Inputs.Count >= 3);
        Assert.True(trainingGraph.Outputs.Count >= 2);
        Assert.Contains(trainingGraph.Nodes, n => n.OpCode == InternalOpCodes.AUTO_GRAD);
    }

    [Fact]
    public void PrepareForTraining_ProducesCorrectStructure()
    {
        // PrepareForTrainingAsFast is typed on the mutable internal graph; hand it deep copies
        // of the shared cached module graphs.
        AssertTrainingGraphStructure(TrainingGraphBuilder.PrepareForTrainingAsFast(
            SimplestLayer.ComputationGraph.ToInternal(),
            SimpleSumSquaredLoss.ComputationGraph.ToInternal()));

        Func<Tensor<float32>, Tensor<float32>, Scalar<float32>> lossFunc = SimpleSumSquaredLoss.Inline;
        AssertTrainingGraphStructure(TrainingGraphBuilder.PrepareForTrainingAsFast(
            SimplestLayer.ComputationGraph.ToInternal(), lossFunc));
    }

    [Fact]
    public void PrepareForTraining_InvalidInputs_Throw()
    {
        var input = Globals.InputTensor<float32>("input", rank: 1);
        var noParamsGraph = new InternalComputationGraph(
            ImmutableArray.Create<Variable>(input),
            ImmutableArray.Create(OnnxOp.Identity(input, null)));
        Assert.Throws<InvalidOperationException>(() =>
            TrainingGraphBuilder.PrepareForTrainingAsFast(
                noParamsGraph, SimpleSumSquaredLoss.ComputationGraph.ToInternal()));

        // A lambda is not a module Inline method — its method name won't be "Inline".
        Func<Tensor<float32>, Tensor<float32>, Scalar<float32>> notAModule =
            (pred, targ) => ((Tensor<float32>)OnnxOp.ReduceSum(pred - targ, keepdims: false)).Scalar();
        Assert.Throws<ArgumentException>(() =>
            TrainingGraphBuilder.PrepareForTrainingAsFast(SimplestLayer.ComputationGraph.ToInternal(), notAModule));
    }

    [Fact]
    public void PrepareForTraining_NullArgs_Throw()
    {
        Assert.Throws<ArgumentNullException>(() => TrainingGraphBuilder.PrepareForTrainingAsFast(
            null!, SimpleSumSquaredLoss.ComputationGraph.ToInternal()));
        Assert.Throws<ArgumentNullException>(() => TrainingGraphBuilder.PrepareForTrainingAsFast(
            SimplestLayer.ComputationGraph.ToInternal(), (InternalComputationGraph)null!));
        Assert.Throws<ArgumentNullException>(() =>
            TrainingGraphBuilder.PrepareForTrainingAsFast<Tensor<float32>, Scalar<float32>>(
                SimplestLayer.ComputationGraph.ToInternal(), null!));
    }
}
