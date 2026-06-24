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
/// Sigmoid focal loss of Lin et al. (RetinaNet): binary cross-entropy over
/// logits reshaped by a <c>(1 − p_t)^γ</c> <i>modulating factor</i> that
/// down-weights well-classified ("easy") examples, plus an <c>α</c>
/// class-balancing factor. The per-element value is
/// <c>α_t · (1 − p_t)^γ · ce</c>, where <c>ce</c> is the numerically stable
/// BCE-with-logits, <c>p = σ(x)</c>, <c>p_t = p·t + (1−p)·(1−t)</c> and
/// <c>α_t = α·t + (1−α)·(1−t)</c>.
/// <para>
/// <b>Inputs:</b> <c>predictions</c> are raw <b>logits</b>; <c>targets</c> are
/// binary labels in <c>{0, 1}</c> of the same shape (the <c>p_t</c>/<c>α_t</c>
/// expressions assume exact <c>{0, 1}</c> targets — soft labels are not
/// supported). This is the from-logits form (numerically stable); the <c>ce</c>
/// term reuses <see cref="BCEWithLogitsLoss.PerElement"/> directly rather than
/// recomputing <c>−log(p_t)</c> from <c>p</c> (which would overflow at large
/// <c>|x|</c>).
/// </para>
/// <para>
/// <b>Knobs.</b> <c>γ</c> (focusing, default <c>2</c>) controls how aggressively
/// easy examples are down-weighted (<c>γ = 0</c> recovers α-weighted BCE).
/// <c>α</c> (class balancing, default <c>0.25</c>) statically weights the
/// positive class by <c>α</c> and the negative class by <c>1−α</c>; the
/// sentinel <c>α = −1</c> disables α-weighting entirely (torchvision's "ignore"
/// convention / Keras <c>apply_class_balancing=False</c>). The defaults
/// (<c>α = 0.25</c>, <c>γ = 2</c>) match torchvision's
/// <c>sigmoid_focal_loss</c> exactly. <c>α</c>/<c>γ</c> are build-time C# floats
/// (architectural — baked into the graph), not <c>[Hyper]</c>s, so no overload
/// adds a third graph input.
/// </para>
/// <para>
/// Two tensor inputs (predictions, targets) → scalar loss: rig-safe default.
/// The 2-input <see cref="Inline"/> is the mean-reduction default;
/// <see cref="Reduced"/> (scalar) and <see cref="PerElement"/> (tensor) add the
/// <c>alpha</c>/<c>gamma</c>/<c>reduction</c> knobs. (RetinaNet's
/// normalize-by-#positives denominator is a caller concern: use
/// <c>Reduced(..., reduction: Sum)</c> and divide.)
/// </para>
/// </summary>
[Module]
public partial class BinaryFocalLoss
{
    public static Scalar<float32> Inline(Tensor<float32> predictions, Tensor<float32> targets)
        => FocalPerElement(predictions, targets, alpha: 0.25f, gamma: 2.0f)
            .Reduce(ReduceKind.Mean, keepDims: false).Scalar();

    /// <summary>
    /// Sigmoid focal loss with build-time <paramref name="alpha"/>/<paramref name="gamma"/>
    /// knobs and a configurable reduction, reduced to a scalar.
    /// </summary>
    /// <param name="predictions">Raw logits.</param>
    /// <param name="targets">Binary targets in <c>{0, 1}</c> (same shape as
    /// <paramref name="predictions"/>).</param>
    /// <param name="alpha">Class-balancing weight on the positive class
    /// (<c>1−α</c> on the negative class); torchvision default <c>0.25</c>. The
    /// sentinel <c>α = −1</c> disables α-weighting (Keras
    /// <c>apply_class_balancing=False</c>).</param>
    /// <param name="gamma">Focusing exponent on the <c>(1 − p_t)^γ</c> modulating
    /// factor; torchvision default <c>2</c>. <c>γ = 0</c> recovers α-weighted
    /// BCE.</param>
    /// <param name="reduction"><see cref="LossReduction.Mean"/> (default) or
    /// <see cref="LossReduction.Sum"/>; use <see cref="PerElement"/> for
    /// <see cref="LossReduction.None"/>.</param>
    public static Scalar<float32> Reduced(
        Tensor<float32> predictions,
        Tensor<float32> targets,
        float alpha = 0.25f,
        float gamma = 2.0f,
        LossReduction reduction = LossReduction.Mean)
    {
        if (reduction == LossReduction.None)
            throw new System.ArgumentException(
                "LossReduction.None returns a per-element tensor; call PerElement instead.",
                nameof(reduction));
        return reduction.ApplyToPerElement(FocalPerElement(predictions, targets, alpha, gamma)).Scalar();
    }

    /// <summary>
    /// Per-element sigmoid focal loss <c>α_t · (1 − p_t)^γ · ce</c>
    /// (<see cref="LossReduction.None"/> semantics — torchvision's default
    /// form): returns the loss tensor (same shape as the logits) without
    /// reduction.
    /// </summary>
    /// <param name="predictions">Raw logits.</param>
    /// <param name="targets">Binary targets in <c>{0, 1}</c>.</param>
    /// <param name="alpha">Class-balancing weight (default <c>0.25</c>;
    /// <c>−1</c> disables α-weighting).</param>
    /// <param name="gamma">Focusing exponent (default <c>2</c>).</param>
    public static Tensor<float32> PerElement(
        Tensor<float32> predictions,
        Tensor<float32> targets,
        float alpha = 0.25f,
        float gamma = 2.0f)
        => FocalPerElement(predictions, targets, alpha, gamma);

    /// <summary>
    /// The in-graph focal formula, mirroring torchvision's
    /// <c>sigmoid_focal_loss</c> line-for-line: <c>loss = ce · (1 − p_t)^γ</c>,
    /// then <c>· α_t</c> when <c>α ≥ 0</c>. The <c>ce</c> term reuses the stable
    /// <see cref="BCEWithLogitsLoss.PerElement"/>; <c>p_t</c> and <c>α_t</c> are
    /// the branch-free <c>p·t + (1−p)·(1−t)</c> / <c>α·t + (1−α)·(1−t)</c>
    /// forms. The <c>α &gt;= 0f</c> test is a build-time C# branch
    /// (<paramref name="alpha"/> is a plain C# float), so the <c>α = −1</c>
    /// sentinel removes the <c>α_t</c> multiply from the graph entirely.
    /// </summary>
    private static Tensor<float32> FocalPerElement(
        Tensor<float32> predictions, Tensor<float32> targets, float alpha, float gamma)
    {
        var ce = BCEWithLogitsLoss.PerElement(predictions, targets);      // stable BCE-with-logits
        var p = predictions.Sigmoid();
        var pT = p * targets + (1f - p) * (1f - targets); // p_t
        var modulating = (1f - pT).Pow(Scalar(gamma));            // (1 − p_t)^γ
        var loss = ce * modulating;
        if (alpha >= 0f)                                                  // α < 0 ⇒ α-weighting off
        {
            var aT = Scalar(alpha) * targets + Scalar(1f - alpha) * (1f - targets); // α_t
            loss = loss * aT;
        }
        return loss;
    }
}
