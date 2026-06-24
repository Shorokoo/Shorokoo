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
/// Hinge (max-margin / SVM) classification loss: per element
/// <c>max(0, 1 − t·p) = relu(1 − targets·predictions)</c>, mean over all
/// elements.
/// <para>
/// <b>Targets MUST be encoded as ±1</b> (<c>+1</c> for the positive class,
/// <c>−1</c> for the negative class). Shorokoo does <b>not</b> auto-convert
/// <c>0/1</c> labels (Keras silently remaps them); pass <c>±1</c> directly. A
/// <c>0/1</c> target yields a wrong-but-finite loss with no useful gradient
/// (<c>t = 0</c> zeroes the margin term), so this contract is load-bearing. If
/// you have <c>0/1</c> labels, map them upstream with <c>2·t − 1</c>.
/// </para>
/// <para>
/// <c>predictions</c> are raw real-valued scores. The margin is fixed at 1
/// (Keras <c>Hinge</c> parity); there is no margin hyperparameter and no class
/// weighting. Two tensor inputs (predictions, targets) → scalar loss: rig-safe
/// default. The 2-input <see cref="Inline"/> is the mean-reduction default;
/// <see cref="Reduced"/> (scalar) and <see cref="PerElement"/> (tensor) add the
/// <c>reduction</c> knob.
/// </para>
/// </summary>
[Module]
public partial class HingeLoss
{
    public static Scalar<float32> Inline(Tensor<float32> predictions, Tensor<float32> targets)
        => HingePerElement(predictions, targets)
            .Reduce(ReduceKind.Mean, keepDims: false).Scalar();

    /// <summary>
    /// Hinge loss with a configurable reduction, reduced to a scalar.
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
        return reduction.ApplyToPerElement(HingePerElement(predictions, targets)).Scalar();
    }

    /// <summary>
    /// Per-element hinge value <c>max(0, 1 − targets·predictions)</c>
    /// (<see cref="LossReduction.None"/> semantics): returns the loss tensor
    /// (same shape as the inputs) without reduction. Targets are <c>±1</c>.
    /// </summary>
    public static Tensor<float32> PerElement(Tensor<float32> predictions, Tensor<float32> targets)
        => HingePerElement(predictions, targets);

    /// <summary>
    /// The shared hinge core <c>relu(1 − t·p) = max(0, 1 − targets·predictions)</c>.
    /// Marked <c>internal</c> so <see cref="SquaredHingeLoss"/> reuses the exact
    /// same per-element hinge and merely squares it.
    /// </summary>
    internal static Tensor<float32> HingePerElement(
        Tensor<float32> predictions, Tensor<float32> targets)
        => (1f - targets * predictions).Relu();
}
