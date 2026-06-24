using System;
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
/// Triplet margin loss (FaceNet / metric-embedding learning): for an anchor
/// <c>a</c>, a positive <c>p</c> (same identity/class) and a negative <c>n</c>
/// (different), the per-triplet loss is
/// <c>L(a, p, n) = max(0, d(a, p) − d(a, n) + margin)</c>, with the p-norm
/// distance <c>d(x, y) = (Σ_feat |x − y|^p + eps)^(1/p)</c> over the <b>last</b>
/// axis. Training pushes the anchor closer to the positive than to the negative
/// by at least <c>margin</c>; a triplet already past the margin contributes zero
/// gradient, so optimisation concentrates on the violating ("hard"/"semi-hard")
/// triplets. This is the canonical metric / embedding-learning objective
/// (Schroff et al., <i>FaceNet</i>, CVPR 2015).
/// <para>
/// <b>Knobs.</b> <c>margin</c> (default <c>1</c>), <c>p</c> (the norm degree,
/// default <c>2</c> ⇒ Euclidean), <c>eps</c> (norm numerical-stability term,
/// default <c>1e-6</c>, added inside the outer root as <c>(Σ + eps)^(1/p)</c> —
/// optax's gradient-clean placement) and <c>swap</c> (default <c>false</c>; the
/// Balntas et al. BMVC 2016 "anchor swap" — when on, the anchor–negative
/// distance is replaced by <c>min(d(a, n), d(p, n))</c>, which never relaxes the
/// constraint). <c>margin</c>/<c>p</c>/<c>eps</c> are <c>[Hyper] Scalar&lt;float32&gt;</c>
/// (schedulable); <c>swap</c> is a runtime <c>[Hyper] Scalar&lt;bit&gt;</c> gated
/// with <c>.IfElse(...)</c> (the <see cref="Shorokoo.Modules.Layers.MultiHeadAttention"/>
/// <c>useBias</c>/<c>causal</c> idiom — C# can't branch on a runtime bit).
/// </para>
/// <para>
/// <b>This is a 3-INPUT loss</b> (anchor, positive, negative — each <c>[N, D]</c>),
/// used via <c>.Call(margin, p, eps, swap, a, p, n)</c> /
/// <c>.Model(margin, p, eps, swap).Call(a, p, n)</c> exactly like
/// <see cref="Shorokoo.Modules.Layers.MultiHeadAttention"/>'s q/k/v. It is <b>not</b> a drop-in 2-input
/// rig loss: the <c>TrainingRig</c> loss slot wires exactly
/// <c>(predictions, targets) → Scalar&lt;float32&gt;</c>, and a 3-tensor loss has
/// no slot. To rig-train it, make the triplet loss the <b>tail of your model</b>
/// (the model emits the three embeddings and returns the scalar triplet loss),
/// paired with a pass-through identity loss in the rig's loss slot — the standard
/// route for any objective that isn't a literal <c>loss(pred, label)</c>.
/// </para>
/// <para>
/// <see cref="Inline"/> is the mean-reduction default (returns a scalar);
/// <see cref="Reduced"/> adds the <c>reduction</c> knob (scalar, mean|sum) and
/// <see cref="PerElement"/> returns the unreduced per-triplet <c>[N]</c> tensor.
/// Higher-rank inputs norm over the last axis only (optax's <c>axis=-1</c>) and
/// the remaining axes are flattened by the reduction.
/// </para>
/// </summary>
[Module]
public partial class TripletMarginLoss
{
    public static Scalar<float32> Inline(
        Tensor<float32> anchor,
        Tensor<float32> positive,
        Tensor<float32> negative,
        [Hyper(1.0f)] Scalar<float32> margin,
        [Hyper(2.0f)] Scalar<float32> p,
        [Hyper(1e-6f)] Scalar<float32> eps,
        [Hyper] Scalar<bit> swap)
        => TripletPerElement(margin, p, eps, swap, anchor, positive, negative)
            .Reduce(ReduceKind.Mean, keepDims: false).Scalar();

    /// <summary>
    /// Triplet margin loss with a configurable reduction, reduced to a scalar.
    /// </summary>
    /// <param name="margin">Margin by which the anchor–positive distance must beat
    /// the anchor–negative distance (default <c>1</c>).</param>
    /// <param name="p">The norm degree of the pairwise distance (default <c>2</c>
    /// ⇒ Euclidean).</param>
    /// <param name="eps">Norm numerical-stability term (default <c>1e-6</c>),
    /// added inside the outer root as <c>(Σ + eps)^(1/p)</c>.</param>
    /// <param name="swap">Balntas "anchor swap"; when <c>true</c> the
    /// anchor–negative distance is replaced by <c>min(d(a, n), d(p, n))</c>.</param>
    /// <param name="anchor">Anchor embeddings <c>[N, D]</c>.</param>
    /// <param name="positive">Positive embeddings <c>[N, D]</c> (same identity as
    /// the anchor).</param>
    /// <param name="negative">Negative embeddings <c>[N, D]</c> (different
    /// identity).</param>
    /// <param name="reduction"><see cref="LossReduction.Mean"/> (default) or
    /// <see cref="LossReduction.Sum"/>; use <see cref="PerElement"/> for
    /// <see cref="LossReduction.None"/>.</param>
    public static Scalar<float32> Reduced(
        Scalar<float32> margin,
        Scalar<float32> p,
        Scalar<float32> eps,
        Scalar<bit> swap,
        Tensor<float32> anchor,
        Tensor<float32> positive,
        Tensor<float32> negative,
        LossReduction reduction = LossReduction.Mean)
    {
        if (reduction == LossReduction.None)
            throw new System.ArgumentException(
                "LossReduction.None returns a per-triplet tensor; call PerElement instead.",
                nameof(reduction));
        return reduction.ApplyToPerElement(
            TripletPerElement(margin, p, eps, swap, anchor, positive, negative)).Scalar();
    }

    /// <summary>
    /// Per-triplet loss <c>max(0, d(a, p) − d(a, n) + margin)</c>
    /// (<see cref="LossReduction.None"/> semantics): returns the length-<c>N</c>
    /// loss tensor (one scalar per triplet) without reduction.
    /// </summary>
    /// <param name="margin">Margin (default <c>1</c>).</param>
    /// <param name="p">The norm degree (default <c>2</c>).</param>
    /// <param name="eps">Norm numerical-stability term (default <c>1e-6</c>).</param>
    /// <param name="swap">Balntas "anchor swap" (default <c>false</c>).</param>
    /// <param name="anchor">Anchor embeddings <c>[N, D]</c>.</param>
    /// <param name="positive">Positive embeddings <c>[N, D]</c>.</param>
    /// <param name="negative">Negative embeddings <c>[N, D]</c>.</param>
    public static Tensor<float32> PerElement(
        Scalar<float32> margin,
        Scalar<float32> p,
        Scalar<float32> eps,
        Scalar<bit> swap,
        Tensor<float32> anchor,
        Tensor<float32> positive,
        Tensor<float32> negative)
        => TripletPerElement(margin, p, eps, swap, anchor, positive, negative);

    /// <summary>
    /// The shared per-triplet core. Computes the three p-norm distances, applies
    /// the runtime-gated Balntas swap, and returns
    /// <c>relu(margin + d(a, p) − dNeg)</c> as the <c>[N]</c> loss vector
    /// (<c>dNeg = swap ? min(d(a, n), d(p, n)) : d(a, n)</c>).
    /// </summary>
    internal static Tensor<float32> TripletPerElement(
        Scalar<float32> margin,
        Scalar<float32> p,
        Scalar<float32> eps,
        Scalar<bit> swap,
        Tensor<float32> anchor,
        Tensor<float32> positive,
        Tensor<float32> negative)
    {
        var dAp = PNormDistance(anchor, positive, p, eps);   // [N]
        var dAn = PNormDistance(anchor, negative, p, eps);   // [N]
        var dPn = PNormDistance(positive, negative, p, eps); // [N] (always built; swap gates its use)
        var dNeg = swap.IfElse(NN.Min(dAn, dPn), dAn);       // Balntas swap, runtime-gated
        return (margin + dAp - dNeg).Relu();                 // max(0, margin + d_ap − d_neg)
    }

    /// <summary>
    /// p-norm pairwise distance over the LAST axis:
    /// <c>d(x, y) = (Σ_feat |x − y|^p + eps)^(1/p)</c>. The <c>eps</c> sits inside
    /// the outer root (optax's placement) so the gradient stays finite at
    /// coincident points (where <c>Σ|·|^p = 0</c>).
    /// </summary>
    internal static Tensor<float32> PNormDistance(
        Tensor<float32> x, Tensor<float32> y, Scalar<float32> p, Scalar<float32> eps)
    {
        var summed = (x - y).Abs().Pow(p)                            // |x − y|^p   [N, D]
            .Reduce(ReduceKind.Sum, [Scalar(-1L)], keepDims: false); // Σ over feat [N]
        return (summed + eps).Pow(1f / p);                   // (Σ + eps)^(1/p) [N]
    }
}

/// <summary>
/// Generalised triplet margin loss with a <b>caller-supplied distance</b>
/// (PyTorch's <c>TripletMarginWithDistanceLoss</c>): identical
/// <c>max(0, distFn(a, p) − distFn(a, n) + margin)</c> skeleton, but the built-in
/// p-norm is replaced by an arbitrary
/// <c>Func&lt;Tensor&lt;float32&gt;, Tensor&lt;float32&gt;, Tensor&lt;float32&gt;&gt;</c>
/// mapping a pair of <c>[N, D]</c> embeddings to a per-row distance <c>[N]</c>
/// (e.g. cosine distance, squared-L2, a learned metric).
/// <para>
/// Because a C# <c>Func</c> can't be a graph input (no <c>[Hyper]</c>/<c>.Call</c>
/// surface), this is a <b>static helper</b>, not a <c>[Module]</c> — there is no
/// <c>p</c>/<c>eps</c> (those belong to whatever distance the caller passes). It
/// is compose-only (never a rig loss), exactly like the plain
/// <see cref="TripletMarginLoss"/>. <c>swap</c> is the Balntas anchor swap
/// applied through the supplied distance: <c>dNeg = swap ? min(distFn(a, n),
/// distFn(p, n)) : distFn(a, n)</c>.
/// </para>
/// </summary>
public static class TripletMarginWithDistance
{
    /// <summary>
    /// Caller-supplied-distance triplet margin loss with a configurable reduction,
    /// reduced to a scalar.
    /// </summary>
    /// <param name="distance">Pairwise distance <c>(x, y) → [N]</c> (nonnegative);
    /// e.g. <c>TripletMarginLoss.PNormDistance</c> reproduces the built-in
    /// p-norm.</param>
    /// <param name="margin">Margin (default <c>1</c>).</param>
    /// <param name="swap">Balntas "anchor swap" (default <c>false</c>).</param>
    /// <param name="anchor">Anchor embeddings <c>[N, D]</c>.</param>
    /// <param name="positive">Positive embeddings <c>[N, D]</c>.</param>
    /// <param name="negative">Negative embeddings <c>[N, D]</c>.</param>
    /// <param name="reduction"><see cref="LossReduction.Mean"/> (default) or
    /// <see cref="LossReduction.Sum"/>; use <see cref="PerElement"/> for
    /// <see cref="LossReduction.None"/>.</param>
    public static Scalar<float32> Reduced(
        Func<Tensor<float32>, Tensor<float32>, Tensor<float32>> distance,
        float margin,
        bool swap,
        Tensor<float32> anchor,
        Tensor<float32> positive,
        Tensor<float32> negative,
        LossReduction reduction = LossReduction.Mean)
    {
        if (reduction == LossReduction.None)
            throw new System.ArgumentException(
                "LossReduction.None returns a per-triplet tensor; call PerElement instead.",
                nameof(reduction));
        return reduction.ApplyToPerElement(
            DistancePerElement(distance, margin, swap, anchor, positive, negative)).Scalar();
    }

    /// <summary>
    /// Per-triplet caller-supplied-distance loss
    /// <c>max(0, distFn(a, p) − distFn(a, n) + margin)</c>
    /// (<see cref="LossReduction.None"/> semantics): returns the <c>[N]</c> loss
    /// vector without reduction.
    /// </summary>
    /// <param name="distance">Pairwise distance <c>(x, y) → [N]</c>.</param>
    /// <param name="margin">Margin (default <c>1</c>).</param>
    /// <param name="swap">Balntas "anchor swap" (default <c>false</c>).</param>
    /// <param name="anchor">Anchor embeddings <c>[N, D]</c>.</param>
    /// <param name="positive">Positive embeddings <c>[N, D]</c>.</param>
    /// <param name="negative">Negative embeddings <c>[N, D]</c>.</param>
    public static Tensor<float32> PerElement(
        Func<Tensor<float32>, Tensor<float32>, Tensor<float32>> distance,
        float margin,
        bool swap,
        Tensor<float32> anchor,
        Tensor<float32> positive,
        Tensor<float32> negative)
        => DistancePerElement(distance, margin, swap, anchor, positive, negative);

    private static Tensor<float32> DistancePerElement(
        Func<Tensor<float32>, Tensor<float32>, Tensor<float32>> distance,
        float margin,
        bool swap,
        Tensor<float32> anchor,
        Tensor<float32> positive,
        Tensor<float32> negative)
    {
        var dAp = distance(anchor, positive);                       // [N]
        var dAn = distance(anchor, negative);                       // [N]
        var dNeg = swap ? NN.Min(dAn, distance(positive, negative)) // Balntas swap (build-time bool)
                        : dAn;
        return (Scalar(margin) + dAp - dNeg).Relu();                // max(0, margin + d_ap − d_neg)
    }
}
