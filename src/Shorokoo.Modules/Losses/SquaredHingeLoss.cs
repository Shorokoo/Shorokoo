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
/// Squared-hinge (max-margin / SVM) classification loss: per element
/// <c>max(0, 1 − t·p)² = relu(1 − targets·predictions)²</c>, mean over all
/// elements. Identical to <see cref="HingeLoss"/> but penalises margin
/// violations <b>quadratically</b> rather than linearly (smooth at the margin
/// boundary, where the plain hinge has a kink).
/// <para>
/// <b>Targets MUST be encoded as ±1</b> (<c>+1</c> positive class, <c>−1</c>
/// negative class). Shorokoo does <b>not</b> auto-convert <c>0/1</c> labels
/// (Keras does); pass <c>±1</c> directly, mapping <c>0/1</c> labels upstream
/// with <c>2·t − 1</c> if needed. <c>predictions</c> are raw real-valued
/// scores; the margin is fixed at 1; there is no margin hyperparameter and no
/// class weighting.
/// </para>
/// <para>
/// Two tensor inputs (predictions, targets) → scalar loss: rig-safe default.
/// The 2-input <see cref="Inline"/> is the mean-reduction default;
/// <see cref="Reduced"/> (scalar) and <see cref="PerElement"/> (tensor) add the
/// <c>reduction</c> knob.
/// </para>
/// </summary>
[Module]
public partial class SquaredHingeLoss
{
    public static Scalar<float32> Inline(Tensor<float32> predictions, Tensor<float32> targets)
        => SquaredHingePerElement(predictions, targets)
            .Reduce(ReduceKind.Mean, keepDims: false).Scalar();

    /// <summary>
    /// Squared-hinge loss with a configurable reduction, reduced to a scalar.
    /// </summary>
    /// <param name="predictions">Raw real-valued scores.</param>
    /// <param name="targets">Class labels encoded as <c>±1</c> (see the type
    /// summary — no <c>0/1</c> auto-conversion).</param>
    /// <param name="reduction"><see cref="LossReduction.Mean"/> (default) or
    /// <see cref="LossReduction.Sum"/>; use <see cref="PerElement"/> for
    /// <see cref="LossReduction.None"/>.</param>
    public static Scalar<float32> Reduced(
        Tensor<float32> predictions,
        Tensor<float32> targets,
        LossReduction reduction = LossReduction.Mean)
    {
        if (reduction == LossReduction.None)
            throw new System.ArgumentException(
                "LossReduction.None returns a per-element tensor; call PerElement instead.",
                nameof(reduction));
        return reduction.ApplyToPerElement(SquaredHingePerElement(predictions, targets)).Scalar();
    }

    /// <summary>
    /// Per-element squared-hinge value <c>max(0, 1 − targets·predictions)²</c>
    /// (<see cref="LossReduction.None"/> semantics): returns the loss tensor
    /// (same shape as the inputs) without reduction. Targets are <c>±1</c>.
    /// </summary>
    public static Tensor<float32> PerElement(Tensor<float32> predictions, Tensor<float32> targets)
        => SquaredHingePerElement(predictions, targets);

    /// <summary>
    /// Squares the shared hinge core <c>relu(1 − t·p)²</c>, reusing
    /// <see cref="HingeLoss.HingePerElement"/> so the <c>max(0, 1 − t·p)</c> core
    /// lives in exactly one place (the squared variant is genuinely "square the
    /// hinge"). Self-multiply rather than <c>Pow(2)</c> — cheapest, no
    /// <c>Pow</c>-at-0 subtlety.
    /// </summary>
    private static Tensor<float32> SquaredHingePerElement(
        Tensor<float32> predictions, Tensor<float32> targets)
    {
        var h = HingeLoss.HingePerElement(predictions, targets);   // relu(1 − t·p)
        return h * h;                                              // (·)²
    }
}
