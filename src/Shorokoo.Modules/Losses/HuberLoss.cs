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
/// Huber loss with hyperparameter transition point <c>delta</c>:
/// per element, <c>0.5 * e^2</c> when <c>|e| &lt;= delta</c>, else
/// <c>delta * (|e| - 0.5 * delta)</c>; mean over all elements.
/// Note: the delta hyperparameter makes this module's ComputationGraph a
/// 3-input graph, so it cannot be handed to TrainingRig directly (the rig's
/// loss contract is exactly (predictions, targets)) — use
/// <see cref="SmoothL1Loss"/> (delta = 1) there, or compose
/// <c>HuberLoss.Inline(predictions, targets, Scalar(d))</c> inside your own 2-input loss module.
/// </summary>
[Module]
public partial class HuberLoss
{
    public static Scalar<float32> Inline(
        Tensor<float32> predictions,
        Tensor<float32> targets,
        [Hyper] Scalar<float32> delta)
        => HuberPerElement(delta, predictions, targets)
            .Reduce(ReduceKind.Mean, keepDims: false).Scalar();

    /// <summary>
    /// Huber loss with a live <c>[Hyper]</c> <paramref name="delta"/> and a
    /// configurable reduction, reduced to a scalar.
    /// </summary>
    /// <param name="delta">Transition point hyperparameter (live / schedulable).</param>
    /// <param name="predictions">Predicted values.</param>
    /// <param name="targets">Target values.</param>
    /// <param name="reduction"><see cref="LossReduction.Mean"/> (default) or
    /// <see cref="LossReduction.Sum"/>; use <see cref="PerElement"/> for
    /// <see cref="LossReduction.None"/>.</param>
    public static Scalar<float32> Reduced(
        Scalar<float32> delta,
        Tensor<float32> predictions,
        Tensor<float32> targets,
        LossReduction reduction = LossReduction.Mean)
    {
        if (reduction == LossReduction.None)
            throw new System.ArgumentException(
                "LossReduction.None returns a per-element tensor; call PerElement instead.",
                nameof(reduction));
        return reduction.ApplyToPerElement(HuberPerElement(delta, predictions, targets)).Scalar();
    }

    /// <summary>
    /// Per-element Huber loss (<see cref="LossReduction.None"/> semantics): returns
    /// the loss tensor (same shape as the inputs) without reduction.
    /// </summary>
    public static Tensor<float32> PerElement(
        Scalar<float32> delta,
        Tensor<float32> predictions,
        Tensor<float32> targets)
        => HuberPerElement(delta, predictions, targets);

    /// <summary>
    /// The piecewise Huber value per element: <c>0.5·e²</c> when <c>|e| ≤ δ</c>,
    /// else <c>δ·(|e| − 0.5·δ)</c>.
    /// </summary>
    private static Tensor<float32> HuberPerElement(
        Scalar<float32> delta, Tensor<float32> predictions, Tensor<float32> targets)
    {
        var err = predictions - targets;
        var absErr = err.Abs();
        var quadratic = Scalar(0.5f) * err * err;
        var linear = delta * absErr - 0.5f * delta * delta;
        var cond = (Tensor<bit>)OnnxOp.LessOrEqual(absErr, delta);
        return cond.Where(quadratic, linear);
    }
}

/// <summary>
/// Smooth-L1 loss: <see cref="HuberLoss"/> with delta fixed at 1, exposed as a
/// 2-input (predictions, targets) module so it satisfies the TrainingRig loss
/// contract.
/// <para>
/// <see cref="Reduced"/> bakes a build-time <c>beta</c> transition point via the
/// bridge <c>SmoothL1(e; β) = Huber(e; δ = β) / β</c> (so <c>β = 1</c> reproduces
/// the 2-input form exactly) and adds a reduction knob. <c>beta</c> is a baked C#
/// <c>float</c>, not a <c>[Hyper]</c>: for a live transition point use
/// <see cref="HuberLoss"/>'s <c>[Hyper] delta</c> and divide by <c>delta</c>.
/// </para>
/// </summary>
[Module]
public partial class SmoothL1Loss
{
    public static Scalar<float32> Inline(Tensor<float32> predictions, Tensor<float32> targets)
        => HuberLoss.Inline(predictions, targets, 1.0f);

    /// <summary>
    /// Smooth-L1 loss with a baked-at-build <paramref name="beta"/> transition
    /// point and a configurable reduction, reduced to a scalar. Computes
    /// <c>Huber(δ = β) / β</c> per element before reducing, so all of
    /// <c>mean</c>/<c>sum</c> stay correct (and <c>β = 1</c> equals
    /// <see cref="Inline"/>).
    /// </summary>
    /// <param name="beta">Transition point (baked C# float; PyTorch default 1.0).</param>
    /// <param name="predictions">Predicted values.</param>
    /// <param name="targets">Target values.</param>
    /// <param name="reduction"><see cref="LossReduction.Mean"/> (default) or
    /// <see cref="LossReduction.Sum"/>; use <see cref="PerElement"/> for
    /// <see cref="LossReduction.None"/>.</param>
    public static Scalar<float32> Reduced(
        float beta,
        Tensor<float32> predictions,
        Tensor<float32> targets,
        LossReduction reduction = LossReduction.Mean)
    {
        if (reduction == LossReduction.None)
            throw new System.ArgumentException(
                "LossReduction.None returns a per-element tensor; call PerElement instead.",
                nameof(reduction));
        return reduction.ApplyToPerElement(SmoothL1PerElement(beta, predictions, targets)).Scalar();
    }

    /// <summary>
    /// Per-element Smooth-L1 loss (<see cref="LossReduction.None"/> semantics) at a
    /// baked <paramref name="beta"/>: <c>Huber(δ = β) / β</c>, i.e.
    /// <c>0.5·e²/β</c> when <c>|e| &lt; β</c>, else <c>|e| − 0.5·β</c>.
    /// </summary>
    public static Tensor<float32> PerElement(
        float beta,
        Tensor<float32> predictions,
        Tensor<float32> targets)
        => SmoothL1PerElement(beta, predictions, targets);

    /// <summary>
    /// The per-element Smooth-L1 value via the Huber bridge
    /// <c>SmoothL1(e; β) = Huber(e; δ = β) / β</c>.
    /// </summary>
    private static Tensor<float32> SmoothL1PerElement(
        float beta, Tensor<float32> predictions, Tensor<float32> targets)
        => HuberLoss.PerElement(Scalar(beta), predictions, targets) * Scalar(1f / beta);
}
