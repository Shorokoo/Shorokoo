using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Onnx;
using static Shorokoo.Globals;

namespace Shorokoo.Modules.Losses;

/// <summary>
/// Negative log-likelihood loss over log-probabilities with int64 class-index
/// targets, via the ONNX NegativeLogLikelihoodLoss op with mean reduction.
/// predictions: <c>[N, C]</c> log-probabilities (e.g. from
/// <c>x.LogSoftmax(1)</c>); targets: <c>[N]</c> class indices → scalar loss.
/// <para>
/// The 2-input <see cref="Inline"/> is the rig-safe default. <see cref="Reduced"/>
/// and <see cref="PerElement"/> expose the configurable knobs (class
/// <c>weight</c>, <c>ignoreIndex</c>, <c>reduction</c>); a <c>weight</c> tensor
/// adds a third graph input, so those overloads leave the default rig path.
/// </para>
/// </summary>
[Module]
public partial class NLLLoss
{
    public static Scalar<float32> Inline(Tensor<float32> predictions, Tensor<int64> targets)
        => NN.NegativeLogLikelihoodLoss(predictions, targets,
            weight: null, ignoreIndex: null, reduction: "mean").Scalar();

    /// <summary>
    /// Negative log-likelihood with the full set of build-time knobs, reduced to a scalar.
    /// </summary>
    /// <param name="predictions"><c>[N, C, ...]</c> log-probabilities.</param>
    /// <param name="targets"><c>[N, ...]</c> int64 class indices.</param>
    /// <param name="weight">Optional per-class <c>[C]</c> rescaling vector. Supplying
    /// it adds a third graph input, so the loss leaves the default rig path.</param>
    /// <param name="ignoreIndex">Optional sentinel target value whose samples
    /// contribute zero loss and zero gradient and are excluded from the mean
    /// denominator.</param>
    /// <param name="reduction"><see cref="LossReduction.Mean"/> (default) or
    /// <see cref="LossReduction.Sum"/>; use <see cref="PerElement"/> for
    /// <see cref="LossReduction.None"/>.</param>
    public static Scalar<float32> Reduced(
        Tensor<float32> predictions,
        Tensor<int64> targets,
        Tensor<float32>? weight = null,
        long? ignoreIndex = null,
        LossReduction reduction = LossReduction.Mean)
    {
        if (reduction == LossReduction.None)
            throw new System.ArgumentException(
                "LossReduction.None returns a per-element tensor; call PerElement instead.",
                nameof(reduction));
        return NN.NegativeLogLikelihoodLoss(predictions, targets,
            weight, ignoreIndex, reduction.ToOnnxReduction()).Scalar();
    }

    /// <summary>
    /// Per-element negative log-likelihood (<see cref="LossReduction.None"/>
    /// semantics): returns the <c>[N, ...]</c> loss tensor, <b>zero at ignored
    /// positions</b>. Same knobs as <see cref="Reduced"/> minus the reduction.
    /// </summary>
    public static Tensor<float32> PerElement(
        Tensor<float32> predictions,
        Tensor<int64> targets,
        Tensor<float32>? weight = null,
        long? ignoreIndex = null)
        => NN.NegativeLogLikelihoodLoss(predictions, targets, weight, ignoreIndex, reduction: "none");
}
