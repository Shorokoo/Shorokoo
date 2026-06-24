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
/// Negative log-likelihood of a Poisson-distributed target whose (log-)rate is
/// predicted by the model.
/// <para>
/// The 2-input <see cref="Inline"/> is the rig-safe default: PyTorch's stable
/// <c>log_input = True</c> form, per element <c>exp(pred) − target·pred</c>, mean
/// over all elements. Predicting the log-rate (<c>λ = exp(pred)</c>) avoids any
/// logarithm of a model output, so there is no <c>log(0)</c>/<c>log(negative)</c>
/// hazard and the rate's non-negativity is automatic.
/// </para>
/// <para>
/// <see cref="Reduced"/> (scalar) and <see cref="PerElement"/> (tensor) expose the
/// build-time knobs <c>logInput</c>, <c>full</c>, <c>eps</c> (and <c>reduction</c>),
/// all baked at build time (no extra graph inputs). With <c>logInput = false</c>
/// the per-element loss is <c>pred − target·log(pred + eps)</c> (the Keras
/// <c>Poisson</c> form, reachable as <c>Reduced(p, t, logInput: false, eps: 1e-7f)</c>);
/// the caller must then keep <c>pred &gt; 0</c> (<c>eps</c> only guards exact
/// <c>pred = 0</c>).
/// </para>
/// <para>
/// <c>full = true</c> adds Stirling's approximation of the dropped constant
/// <c>log(target!)</c>: <c>target·log target − target + 0.5·log(2π·target)</c> for
/// <c>target &gt; 1</c>, else <c>0</c>. Because <c>Where</c> evaluates <b>both</b>
/// lanes, the Stirling expression is computed with a clamped
/// <c>tSafe = max(target, 1)</c> inside its <c>log</c> terms so the discarded lane
/// stays finite (otherwise <c>0·log 0 = NaN</c> at <c>target = 0</c> would leak
/// through the select); the live value is then gated by <c>target &gt; 1</c>.
/// </para>
/// </summary>
[Module]
public partial class PoissonNLLLoss
{
    public static Scalar<float32> Inline(Tensor<float32> predictions, Tensor<float32> targets)
        => PerElement(predictions, targets, logInput: true, full: false, eps: 1e-8f)
            .Reduce(ReduceKind.Mean, keepDims: false).Scalar();

    /// <summary>
    /// Poisson NLL with the full set of build-time knobs, reduced to a scalar.
    /// </summary>
    /// <param name="predictions">Predicted (log-)rate. With
    /// <paramref name="logInput"/> = <c>true</c> this is <c>log λ</c>; with
    /// <c>false</c> it is the rate <c>λ</c> directly (keep it <c>&gt; 0</c>).</param>
    /// <param name="targets">Poisson-distributed target counts/rates
    /// (real-valued <c>Tensor&lt;float32&gt;</c>).</param>
    /// <param name="logInput">If <c>true</c> (PyTorch default, the stable form) the
    /// per-element loss is <c>exp(pred) − target·pred</c>; if <c>false</c> it is
    /// <c>pred − target·log(pred + eps)</c> (the Keras form).</param>
    /// <param name="full">If <c>true</c>, adds Stirling's approximation of
    /// <c>log(target!)</c> for <c>target &gt; 1</c> (zero otherwise), matching
    /// PyTorch.</param>
    /// <param name="eps">Small value added inside the log to guard <c>log(0)</c>,
    /// used <b>only</b> when <paramref name="logInput"/> = <c>false</c> (PyTorch
    /// default <c>1e-8</c>; pass <c>1e-7f</c> for exact Keras parity).</param>
    /// <param name="reduction"><see cref="LossReduction.Mean"/> (default) or
    /// <see cref="LossReduction.Sum"/>; use <see cref="PerElement"/> for
    /// <see cref="LossReduction.None"/>.</param>
    public static Scalar<float32> Reduced(
        Tensor<float32> predictions,
        Tensor<float32> targets,
        bool logInput = true,
        bool full = false,
        float eps = 1e-8f,
        LossReduction reduction = LossReduction.Mean)
    {
        if (reduction == LossReduction.None)
            throw new System.ArgumentException(
                "LossReduction.None returns a per-element tensor; call PerElement instead.",
                nameof(reduction));
        return reduction
            .ApplyToPerElement(PerElement(predictions, targets, logInput, full, eps))
            .Scalar();
    }

    /// <summary>
    /// Per-element Poisson NLL (<see cref="LossReduction.None"/> semantics):
    /// <c>exp(pred) − target·pred</c> when <paramref name="logInput"/>, else
    /// <c>pred − target·log(pred + eps)</c>, plus the Stirling
    /// <paramref name="full"/> term. Returns the loss tensor (same shape as the
    /// inputs) without reduction.
    /// </summary>
    public static Tensor<float32> PerElement(
        Tensor<float32> predictions,
        Tensor<float32> targets,
        bool logInput = true,
        bool full = false,
        float eps = 1e-8f)
    {
        var pred = predictions;
        var loss = logInput
            ? pred.Exp() - targets * pred                       // exp(pred) − t·pred
            : pred - targets * (pred + Scalar(eps)).Ln();       // pred − t·log(pred+eps)

        if (full)
            loss = loss + StirlingTerm(targets);
        return loss;
    }

    /// <summary>
    /// PyTorch's Stirling approximation of <c>log(target!)</c>:
    /// <c>target·log target − target + 0.5·log(2π·target)</c> for
    /// <c>target &gt; 1</c>, else <c>0</c>.
    /// <para>
    /// <c>Where</c> evaluates both lanes, so the <c>log</c> terms are formed with a
    /// clamped <c>tSafe = max(target, 1)</c> to keep the discarded lane finite
    /// (avoiding <c>0·log 0 = NaN</c> at <c>target = 0</c>); the live value is then
    /// gated by <c>target &gt; 1</c> against a same-shape zero tensor.
    /// </para>
    /// </summary>
    private static Tensor<float32> StirlingTerm(Tensor<float32> targets)
    {
        const float TwoPi = 6.2831855f; // 2π
        // Clamp inside the log terms so the masked-out lane is finite (no 0·log 0).
        var tSafe = targets.Max(Scalar(1f));
        var stirling = tSafe * tSafe.Ln() - targets
            + 0.5f * (Scalar(TwoPi) * tSafe).Ln();
        var gate = (Tensor<bit>)OnnxOp.Greater(targets, Scalar(1f)); // target > 1
        return gate.Where(stirling, targets * 0f);           // else zeros (shape-matched)
    }
}
