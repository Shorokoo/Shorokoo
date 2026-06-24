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
/// Log-cosh regression loss: per element <c>log(cosh(p − t))</c>, mean over all
/// elements. A smooth, hyperparameter-free Huber alternative — ≈ <c>d²/2</c>
/// (MSE-like) for small errors and ≈ <c>|d| − log 2</c> (MAE-like) for large
/// errors, and twice-differentiable everywhere (its derivative is <c>tanh(d)</c>).
/// <para>
/// Computed via the numerically stable identity
/// <c>log(cosh(d)) = |d| + softplus(−2·|d|) − log 2</c> so it does <b>not</b>
/// overflow/NaN in the tails (the naive <c>log(cosh(d)) = log((e^d + e^−d)/2)</c>
/// overflows for <c>|d| ≳ 89</c> in float32). The softplus argument is
/// <c>−2·|d|</c> (note the <c>|·|</c>, not <c>−2·d</c>): it is <c>≤ 0</c> for
/// <b>both</b> signs of <c>d</c>, so <c>e^{−2|d|} ∈ (0, 1]</c> never overflows.
/// </para>
/// <para>
/// Two tensor inputs (predictions, targets) → scalar loss: rig-safe default. The
/// 2-input <see cref="Inline"/> is the mean-reduction default; <see cref="Reduced"/>
/// (scalar) and <see cref="PerElement"/> (tensor) add the <c>reduction</c> knob.
/// </para>
/// </summary>
[Module]
public partial class LogCoshLoss
{
    public static Scalar<float32> Inline(Tensor<float32> predictions, Tensor<float32> targets)
        => LogCoshPerElement(predictions, targets)
            .Reduce(ReduceKind.Mean, keepDims: false).Scalar();

    /// <summary>
    /// Log-cosh loss with a configurable reduction, reduced to a scalar.
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
        return reduction.ApplyToPerElement(LogCoshPerElement(predictions, targets)).Scalar();
    }

    /// <summary>
    /// Per-element log-cosh value <c>log(cosh(predictions − targets))</c>
    /// (<see cref="LossReduction.None"/> semantics): returns the loss tensor (same
    /// shape as the inputs) without reduction.
    /// </summary>
    public static Tensor<float32> PerElement(Tensor<float32> predictions, Tensor<float32> targets)
        => LogCoshPerElement(predictions, targets);

    /// <summary>
    /// The numerically stable per-element log-cosh:
    /// <c>log(cosh(d)) = |d| + softplus(−2·|d|) − log 2</c>, where
    /// <c>d = predictions − targets</c> and <c>log 2 ≈ 0.6931472</c>. The
    /// <c>−2·|d|</c> argument keeps softplus's input <c>≤ 0</c> for both signs of
    /// <c>d</c>, so the form is overflow-free everywhere.
    /// </summary>
    private static Tensor<float32> LogCoshPerElement(
        Tensor<float32> predictions, Tensor<float32> targets)
    {
        var absD = (predictions - targets).Abs();
        const float Log2 = 0.6931472f;
        return absD + (-2f * absD).Softplus() - Scalar(Log2);
    }
}
