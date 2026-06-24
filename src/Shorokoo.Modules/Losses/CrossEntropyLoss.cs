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
/// Cross-entropy classification loss over raw logits with int64 class-index
/// targets, via the ONNX SoftmaxCrossEntropyLoss op with mean reduction.
/// predictions: <c>[N, C]</c> (or <c>[N, C, d1, ...]</c>) logits;
/// targets: <c>[N]</c> (or <c>[N, d1, ...]</c>) class indices → scalar loss.
/// <para>
/// The 2-input <see cref="Inline"/> is the rig-safe default (mean reduction, no
/// extra inputs). <see cref="Reduced"/> and <see cref="PerElement"/> expose the
/// configurable knobs (class <c>weight</c>, <c>ignoreIndex</c>,
/// <c>labelSmoothing</c>, <c>reduction</c>); passing a <c>weight</c> tensor adds a
/// third graph input, so those overloads leave the default rig path — see the
/// Losses section of <c>Documentation/nn-library.md</c> for the rig-via-baked-weight
/// recipe.
/// </para>
/// </summary>
[Module]
public partial class CrossEntropyLoss
{
    public static Scalar<float32> Inline(Tensor<float32> predictions, Tensor<int64> targets)
    {
        var (loss, _) = NN.SoftmaxCrossEntropyLoss(predictions, targets,
            weights: null, ignoreIndex: null, reduction: "mean");
        return loss.Scalar();
    }

    /// <summary>
    /// Cross-entropy with the full set of build-time knobs, reduced to a scalar.
    /// </summary>
    /// <param name="predictions"><c>[N, C, ...]</c> logits.</param>
    /// <param name="targets"><c>[N, ...]</c> int64 class indices.</param>
    /// <param name="weight">Optional per-class <c>[C]</c> rescaling vector. Supplying
    /// it adds a third graph input, so the loss leaves the default rig path.</param>
    /// <param name="ignoreIndex">Optional sentinel target value whose samples
    /// contribute zero loss and zero gradient and are excluded from the mean
    /// denominator (matching PyTorch's "averaged over non-ignored targets").</param>
    /// <param name="labelSmoothing">Smoothing factor <c>α ∈ [0, 1]</c>: blends the
    /// hard target with the uniform distribution as
    /// <c>(1−α)·NLL + α·(−(1/K)·Σ_k log p_k)</c>. <c>α = 0</c> (the default) takes
    /// the single-op fast path.</param>
    /// <param name="reduction"><see cref="LossReduction.Mean"/> (default) or
    /// <see cref="LossReduction.Sum"/>; use <see cref="PerElement"/> for
    /// <see cref="LossReduction.None"/>.</param>
    public static Scalar<float32> Reduced(
        Tensor<float32> predictions,
        Tensor<int64> targets,
        Tensor<float32>? weight = null,
        long? ignoreIndex = null,
        float labelSmoothing = 0f,
        LossReduction reduction = LossReduction.Mean)
    {
        if (reduction == LossReduction.None)
            throw new System.ArgumentException(
                "LossReduction.None returns a per-element tensor; call PerElement instead.",
                nameof(reduction));

        if (labelSmoothing == 0f)
        {
            // Fast path: exact single-op cross-entropy (no in-graph decomposition).
            var (loss, _) = NN.SoftmaxCrossEntropyLoss(predictions, targets,
                weights: weight, ignoreIndex: ignoreIndex, reduction: reduction.ToOnnxReduction());
            return loss.Scalar();
        }

        return BuildLabelSmoothed(predictions, targets, weight, ignoreIndex, labelSmoothing, reduction)
            .Scalar();
    }

    /// <summary>
    /// Per-element cross-entropy (<see cref="LossReduction.None"/> semantics):
    /// returns the <c>[N, ...]</c> loss tensor, <b>zero at ignored positions</b>
    /// (matching PyTorch/ONNX). Same knobs as <see cref="Reduced"/> minus the
    /// reduction.
    /// </summary>
    public static Tensor<float32> PerElement(
        Tensor<float32> predictions,
        Tensor<int64> targets,
        Tensor<float32>? weight = null,
        long? ignoreIndex = null,
        float labelSmoothing = 0f)
    {
        if (labelSmoothing == 0f)
        {
            var (loss, _) = NN.SoftmaxCrossEntropyLoss(predictions, targets,
                weights: weight, ignoreIndex: ignoreIndex, reduction: "none");
            return loss;
        }

        return BuildLabelSmoothed(predictions, targets, weight, ignoreIndex, labelSmoothing, LossReduction.None);
    }

    /// <summary>
    /// Builds the label-smoothed cross-entropy in-graph from LogSoftmax + NLL
    /// primitives (ONNX SoftmaxCrossEntropyLoss has no label_smoothing attribute):
    /// <c>loss = (1−α)·NLL(logp, target) + α·(−(1/K)·Σ_k logp_k)</c>, threading the
    /// same <paramref name="weight"/> / <paramref name="ignoreIndex"/> masking through
    /// <b>both</b> terms so the weighted/ignored denominators match PyTorch. The
    /// returned tensor is already reduced under <paramref name="reduction"/>.
    /// </summary>
    private static Tensor<float32> BuildLabelSmoothed(
        Tensor<float32> predictions,
        Tensor<int64> targets,
        Tensor<float32>? weight,
        long? ignoreIndex,
        float labelSmoothing,
        LossReduction reduction)
    {
        var alpha = labelSmoothing;
        var reductionStr = reduction.ToOnnxReduction();

        // log-probabilities over the class axis (axis 1, as PyTorch / ONNX SCEL).
        var logp = predictions.LogSoftmax(1);

        // Standard NLL term, with weight + ignore_index masking and the requested
        // reduction (this gives exactly the SCEL behaviour for the hard target).
        var nll = NN.NegativeLogLikelihoodLoss(logp, targets, weight, ignoreIndex, reductionStr);

        // Uniform-smoothing term: the per-sample mean negative log-prob across all
        // K classes, i.e. −(1/K)·Σ_k logp_k. We build it as another NLL so that the
        // weight[target] factor and ignore_index masking (and the matching
        // weighted/ignored mean denominator) are applied identically to the NLL
        // term. ReduceMean over the class axis collapses [N, C, ...] → [N, 1, ...];
        // adding a zero-scaled `logp` broadcasts that per-sample value back across
        // all C class slots ([N, 1, ...] + [N, C, ...] → [N, C, ...]) so the
        // subsequent NLL gathers the same uniform value at every target.
        var meanLogpPerSample = logp.Reduce(ReduceKind.Mean, Vector(1L), keepDims: true);
        var uniformLogp = meanLogpPerSample + logp * 0f;
        var smooth = NN.NegativeLogLikelihoodLoss(uniformLogp, targets, weight, ignoreIndex, reductionStr);

        return smooth * Scalar(alpha) + nll * Scalar(1f - alpha);
    }
}
