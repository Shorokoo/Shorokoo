using Shorokoo.Core.Nodes.Processors.Helpers;
using System.Collections.Immutable;

namespace Shorokoo.Tests;

/// <summary>
/// Simple sum-of-squared-differences loss: sum((pred - target)^2).
/// Takes two tensor inputs and returns a scalar loss.
/// Must be at namespace level for the source generator to work.
/// </summary>
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
/// Quick tier tests for TrainingGraphBuilder.PrepareForTrainingAsFast.
/// Verifies that a model computation graph can be composed with a loss function
/// and automatic differentiation to produce a high-level training graph with correct
/// inputs (model input, targets, trainable param struct) and outputs (loss, gradient struct).
/// The returned graph contains AUTO_GRAD nodes that have not been lowered.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class TrainingGraphBuilderQuickTests
{
    #region PrepareForTrainingAsFast with FastComputationGraph Loss Overload

    /// <summary>
    /// Verifies that PrepareForTrainingAsFast produces a high-level graph with correct structure:
    /// inputs include model input, targets, and trainable param struct;
    /// outputs include loss and gradient struct;
    /// graph contains AUTO_GRAD nodes (not lowered).
    /// </summary>
    [Fact]
    public void PrepareForTraining_GraphOverload_ProducesCorrectStructure()
    {
        var modelGraph = SimplestLayer.ComputationGraph;
        var lossGraph = SimpleSumSquaredLoss.ComputationGraph;

        var trainingGraph = TrainingGraphBuilder.PrepareForTrainingAsFast(modelGraph, lossGraph);

        // Should have 3 inputs: model input, targets, param struct
        Assert.True(trainingGraph.Inputs.Count >= 3,
            $"Expected at least 3 inputs, got {trainingGraph.Inputs.Count}");

        // Should have 2 outputs: loss and gradient struct
        Assert.True(trainingGraph.Outputs.Count >= 2,
            $"Expected at least 2 outputs, got {trainingGraph.Outputs.Count}");

        // The high-level graph should still contain AUTO_GRAD nodes (not lowered)
        var hasAutoGrad = trainingGraph.Nodes
            .Any(n => n.OpCode == InternalOpCodes.AUTO_GRAD);
        Assert.True(hasAutoGrad,
            "Expected the training graph to contain AUTO_GRAD nodes (high-level, not lowered)");
    }

    /// <summary>
    /// Verifies that PrepareForTrainingAsFast throws when the model graph has no trainable parameters.
    /// </summary>
    [Fact]
    public void PrepareForTraining_NoTrainableParams_Throws()
    {
        // A simple identity-like model with no trainable params
        var input = Globals.InputTensor<float32>("input", rank: 1);
        var output = OnnxOp.Identity(input, null);
        var modelGraph = new FastComputationGraph(
            ImmutableArray.Create<Variable>(input),
            ImmutableArray.Create(output));

        var lossGraph = SimpleSumSquaredLoss.ComputationGraph;

        Assert.Throws<InvalidOperationException>(() =>
            TrainingGraphBuilder.PrepareForTrainingAsFast(modelGraph, lossGraph));
    }

    #endregion

    #region PrepareForTrainingAsFast with Func Loss Overload

    /// <summary>
    /// Verifies that PrepareForTrainingAsFast with a Func delegate correctly extracts the
    /// FastComputationGraph from the module and produces a valid high-level training graph.
    /// </summary>
    [Fact]
    public void PrepareForTraining_FuncOverload_ProducesCorrectStructure()
    {
        var modelGraph = SimplestLayer.ComputationGraph;

        Func<Tensor<float32>, Tensor<float32>, Scalar<float32>> lossFunc = SimpleSumSquaredLoss.Inline;

        var trainingGraph = TrainingGraphBuilder.PrepareForTrainingAsFast(modelGraph, lossFunc);

        // Same structural checks as graph overload
        Assert.True(trainingGraph.Inputs.Count >= 3,
            $"Expected at least 3 inputs, got {trainingGraph.Inputs.Count}");
        Assert.True(trainingGraph.Outputs.Count >= 2,
            $"Expected at least 2 outputs, got {trainingGraph.Outputs.Count}");

        // The high-level graph should still contain AUTO_GRAD nodes (not lowered)
        var hasAutoGrad = trainingGraph.Nodes
            .Any(n => n.OpCode == InternalOpCodes.AUTO_GRAD);
        Assert.True(hasAutoGrad,
            "Expected the training graph to contain AUTO_GRAD nodes (high-level, not lowered)");
    }

    /// <summary>
    /// Verifies that the Func overload throws when the delegate doesn't reference
    /// a module's Inline method.
    /// </summary>
    [Fact]
    public void PrepareForTraining_NonModuleFunc_Throws()
    {
        var modelGraph = SimplestLayer.ComputationGraph;

        // A lambda is not a module Inline method — its method name won't be "Inline"
        Func<Tensor<float32>, Tensor<float32>, Scalar<float32>> notAModule =
            (pred, targ) => ((Tensor<float32>)OnnxOp.ReduceSum(pred - targ, keepdims: false)).Scalar();

        Assert.Throws<ArgumentException>(() =>
            TrainingGraphBuilder.PrepareForTrainingAsFast(modelGraph, notAModule));
    }

    #endregion

    #region Null Argument Validation

    [Fact]
    public void PrepareForTraining_NullModelGraph_Throws()
    {
        var lossGraph = SimpleSumSquaredLoss.ComputationGraph;
        Assert.Throws<ArgumentNullException>(() =>
            TrainingGraphBuilder.PrepareForTrainingAsFast(null!, lossGraph));
    }

    [Fact]
    public void PrepareForTraining_NullLossGraph_Throws()
    {
        var modelGraph = SimplestLayer.ComputationGraph;
        Assert.Throws<ArgumentNullException>(() =>
            TrainingGraphBuilder.PrepareForTrainingAsFast(modelGraph, (FastComputationGraph)null!));
    }

    [Fact]
    public void PrepareForTraining_NullLossFunc_Throws()
    {
        var modelGraph = SimplestLayer.ComputationGraph;
        Assert.Throws<ArgumentNullException>(() =>
            TrainingGraphBuilder.PrepareForTrainingAsFast<Tensor<float32>, Scalar<float32>>(modelGraph, null!));
    }

    #endregion
}
