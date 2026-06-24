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
/// L1 (Mean Absolute Error) loss:
/// <c>loss = mean(|predictions - targets|)</c> over all elements.
/// Two tensor inputs (predictions, targets) → scalar loss, like <see cref="L2Loss"/>.
/// <para>
/// The 2-input <see cref="Inline"/> is the rig-safe default. <see cref="Reduced"/>
/// (scalar) and <see cref="PerElement"/> (tensor) add the <c>reduction</c> knob.
/// </para>
/// </summary>
[Module]
public partial class L1Loss
{
    public static Scalar<float32> Inline(Tensor<float32> predictions, Tensor<float32> targets)
        => (predictions - targets).Abs().Reduce(ReduceKind.Mean, keepDims: false).Scalar();

    /// <summary>
    /// L1 loss with a configurable reduction, reduced to a scalar.
    /// </summary>
    /// <param name="predictions">Predicted values.</param>
    /// <param name="targets">Target values.</param>
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
        return reduction.ApplyToPerElement((predictions - targets).Abs()).Scalar();
    }

    /// <summary>
    /// Per-element absolute error <c>|predictions − targets|</c>
    /// (<see cref="LossReduction.None"/> semantics).
    /// </summary>
    public static Tensor<float32> PerElement(Tensor<float32> predictions, Tensor<float32> targets)
        => (predictions - targets).Abs();
}
