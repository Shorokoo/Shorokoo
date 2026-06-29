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
/// Binary cross-entropy over raw logits, using the numerically stable form
/// <c>loss = mean(max(x, 0) - x * t + ln(1 + exp(-|x|)))</c>
/// (equivalent to sigmoid followed by <see cref="BCELoss"/>, without the
/// overflow of computing the sigmoid explicitly).
/// <para>
/// The 2-input <see cref="Inline"/> is the rig-safe default. <see cref="Reduced"/>
/// and <see cref="PerElement"/> add a <c>posWeight</c> (per-positive-term
/// rescaling) and a <c>reduction</c> knob; a <c>posWeight</c> tensor adds a third
/// graph input, so those overloads leave the default rig path.
/// </para>
/// </summary>
[Module]
public partial class BCEWithLogitsLoss
{
    public static Scalar<float32> Inline(Tensor<float32> predictions, Tensor<float32> targets)
    {
        var x = predictions;
        var perElement = x.Relu() - x * targets
            + (1f + (x.Abs() * -1f).Exp()).Ln();
        return perElement.Reduce(ReduceKind.Mean, keepDims: false).Scalar();
    }

    /// <summary>
    /// Binary cross-entropy over logits with optional <paramref name="posWeight"/>
    /// and a configurable reduction, reduced to a scalar.
    /// </summary>
    /// <param name="predictions">Raw logits.</param>
    /// <param name="targets">Binary targets (same shape as <paramref name="predictions"/>).</param>
    /// <param name="posWeight">Optional weight on the positive-label term
    /// (PyTorch's <c>pos_weight</c>, rule of thumb <c>#neg/#pos</c>); broadcastable
    /// against the logits. Supplying it adds a third graph input, so the loss
    /// leaves the default rig path.</param>
    /// <param name="reduction"><see cref="LossReduction.Mean"/> (default) or
    /// <see cref="LossReduction.Sum"/>; use <see cref="PerElement"/> for
    /// <see cref="LossReduction.None"/>.</param>
    public static Scalar<float32> Reduced(
        Tensor<float32> predictions,
        Tensor<float32> targets,
        Tensor<float32>? posWeight = null,
        LossReduction reduction = LossReduction.Mean)
    {
        if (reduction == LossReduction.None)
            throw new System.ArgumentException(
                "LossReduction.None returns a per-element tensor; call PerElement instead.",
                nameof(reduction));
        return reduction.ApplyToPerElement(StablePerElement(predictions, targets, posWeight)).Scalar();
    }

    /// <summary>
    /// Per-element binary cross-entropy over logits (<see cref="LossReduction.None"/>
    /// semantics): returns the loss tensor (same shape as the logits) with the
    /// optional <paramref name="posWeight"/> applied.
    /// </summary>
    public static Tensor<float32> PerElement(
        Tensor<float32> predictions,
        Tensor<float32> targets,
        Tensor<float32>? posWeight = null)
        => StablePerElement(predictions, targets, posWeight);

    /// <summary>
    /// The numerically stable per-element BCE-with-logits with optional
    /// <paramref name="posWeight"/> <c>p_c</c>:
    /// <c>max(x, 0) − x·t + (1 + (p_c − 1)·t)·ln(1 + e^{−|x|})</c>
    /// (the stable rearrangement of <c>−[p_c·t·ln σ(x) + (1−t)·ln(1−σ(x))]</c>).
    /// With <c>p_c = 1</c> (or null) this reduces exactly to the unweighted form.
    /// </summary>
    private static Tensor<float32> StablePerElement(
        Tensor<float32> predictions, Tensor<float32> targets, Tensor<float32>? posWeight)
    {
        var x = predictions;
        var softplus = (1f + (x.Abs() * -1f).Exp()).Ln();
        if (posWeight is null)
            return x.Relu() - x * targets + softplus;

        // logWeight = 1 + (p_c − 1)·t  scales the softplus (log-sum-exp) term only.
        var logWeight = Scalar(1f) + (posWeight.Value - 1f) * targets;
        return x.Relu() - x * targets + logWeight * softplus;
    }
}
