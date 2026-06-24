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
/// Cosine embedding loss (the cosine-space contrastive / metric-learning loss):
/// given two batches of embedding vectors <c>x1</c>, <c>x2</c> (each
/// <c>[N, D]</c>) and a per-sample label <c>y_i ∈ {+1, −1}</c>, with
/// <c>cos(a, b) = (a·b) / max(‖a‖₂·‖b‖₂, eps)</c> the cosine similarity over the
/// <b>last</b> axis, the per-sample loss is
/// <c>L_i = 1 − cos(x1_i, x2_i)</c> when <c>y_i = +1</c> (the pair <i>should</i>
/// be similar — pull them together; minimised at <c>cos = 1</c>) and
/// <c>L_i = max(0, cos(x1_i, x2_i) − margin)</c> when <c>y_i = −1</c> (the pair
/// <i>should</i> be dissimilar — push them apart, but only until
/// <c>cos ≤ margin</c>; the hinge means an already-dissimilar pair contributes
/// zero gradient). Branch-free as
/// <c>where(y == 1, 1 − cos, relu(cos − margin))</c>.
/// <para>
/// Cosine measures the <i>angle</i> between two vectors and is invariant to their
/// magnitudes (<c>cos(a, b) = cos(λa, μb)</c> for positive <c>λ, μ</c>), which is
/// exactly what you want when an embedding's <i>direction</i> carries the
/// semantics and its norm does not — sentence/document embeddings, word vectors,
/// face/speaker embeddings, and any siamese / two-tower retrieval model trained
/// on angular similarity. This is the cosine-space sibling of the (Euclidean)
/// contrastive/triplet losses (the canonical objective for sentence-pair
/// similarity / paraphrase learning); the <c>margin</c> is the cosine-space
/// analog of triplet's additive margin (it lets "far enough apart",
/// <c>cos ≤ margin</c>, stop contributing gradient).
/// </para>
/// <para>
/// <b>Labels MUST be encoded as ±1</b> (<c>+1</c> = the "similar" pull arm,
/// <c>−1</c> = the "dissimilar" hinged-push arm), <i>per sample</i> — so <c>y</c>
/// is a <c>Tensor&lt;float32&gt;</c> input (data), not a <c>[Hyper]</c>. Shorokoo
/// does <b>not</b> auto-convert <c>0/1</c> labels; pass <c>±1</c> directly (the
/// same explicit contract as <see cref="HingeLoss"/>). If you have <c>0/1</c>
/// labels, map them upstream with <c>2·t − 1</c>.
/// </para>
/// <para>
/// <b>Knobs.</b> <c>margin</c> (default <c>0</c>, range <c>[−1, 1]</c>, "0 to 0.5
/// suggested"; affects <b>only</b> the <c>y = −1</c> arm) and <c>eps</c>
/// (denominator floor, default <c>1e-8</c> — PyTorch's <c>nn.CosineSimilarity</c>
/// default; clamps the <i>product</i> of norms via <c>max(‖x1‖·‖x2‖, eps)</c> so
/// a zero vector does not divide by zero). Both are
/// <c>[Hyper] Scalar&lt;float32&gt;</c> (schedulable — <c>margin</c> annealing is a
/// real metric-learning technique).
/// </para>
/// <para>
/// <b>This is a 3-INPUT loss</b> (<c>x1</c>, <c>x2</c>, <c>y</c>), used via
/// <c>.Call(margin, eps, x1, x2, y)</c> /
/// <c>.Model(margin, eps).Call(x1, x2, y)</c> exactly like
/// <see cref="Shorokoo.Modules.Layers.MultiHeadAttention"/>'s q/k/v and
/// <see cref="TripletMarginLoss"/>'s a/p/n. It is <b>not</b> a drop-in 2-input rig
/// loss: the <c>TrainingRig</c> loss slot wires exactly
/// <c>(predictions, targets) → Scalar&lt;float32&gt;</c>, and a 3-tensor loss has no
/// slot. Because <c>y</c> is a genuine per-sample label, the clean rig recipe is
/// to have the model emit <c>concat(e1, e2)</c> as its prediction and feed
/// <c>y</c> as the rig target, with a thin 2-input adapter that splits the
/// prediction back into <c>(e1, e2)</c> and calls this loss — so unlike triplet,
/// cosine drops into the rig's loss slot directly with that adapter (no
/// pass-through identity loss needed). The general fallback (shared with triplet)
/// is to make the loss the tail of the model and pair it with a pass-through
/// identity loss.
/// </para>
/// <para>
/// <see cref="Inline"/> is the mean-reduction default (returns a scalar);
/// <see cref="Reduced"/> adds the <c>reduction</c> knob (scalar, mean|sum) and
/// <see cref="PerElement"/> returns the unreduced per-sample <c>[N]</c> tensor.
/// Higher-rank inputs take cosine over the last axis only and the remaining axes
/// are flattened by the reduction.
/// </para>
/// </summary>
[Module]
public partial class CosineEmbeddingLoss
{
    public static Scalar<float32> Inline(
        Tensor<float32> x1,
        Tensor<float32> x2,
        Tensor<float32> y,
        [Hyper(0f)] Scalar<float32> margin,
        [Hyper(1e-8f)] Scalar<float32> eps)
        => CosinePerElement(margin, eps, x1, x2, y)
            .Reduce(ReduceKind.Mean, keepDims: false).Scalar();

    /// <summary>
    /// Cosine embedding loss with a configurable reduction, reduced to a scalar.
    /// </summary>
    /// <param name="margin">Margin for the <c>y = −1</c> (dissimilar) arm, default
    /// <c>0</c> (range <c>[−1, 1]</c>); affects only that arm.</param>
    /// <param name="eps">Denominator floor (default <c>1e-8</c>), clamping the
    /// product of norms via <c>max(‖x1‖·‖x2‖, eps)</c>.</param>
    /// <param name="x1">First embedding batch <c>[N, D]</c>.</param>
    /// <param name="x2">Second embedding batch <c>[N, D]</c>.</param>
    /// <param name="y">Per-sample labels encoded as <c>±1</c> (<c>+1</c> = similar
    /// pull arm, <c>−1</c> = dissimilar hinged-push arm; see the type summary — no
    /// <c>0/1</c> auto-conversion).</param>
    /// <param name="reduction"><see cref="LossReduction.Mean"/> (default) or
    /// <see cref="LossReduction.Sum"/>; use <see cref="PerElement"/> for
    /// <see cref="LossReduction.None"/>.</param>
    public static Scalar<float32> Reduced(
        Scalar<float32> margin,
        Scalar<float32> eps,
        Tensor<float32> x1,
        Tensor<float32> x2,
        Tensor<float32> y,
        LossReduction reduction = LossReduction.Mean)
    {
        if (reduction == LossReduction.None)
            throw new System.ArgumentException(
                "LossReduction.None returns a per-sample tensor; call PerElement instead.",
                nameof(reduction));
        return reduction.ApplyToPerElement(
            CosinePerElement(margin, eps, x1, x2, y)).Scalar();
    }

    /// <summary>
    /// Per-sample cosine embedding loss
    /// <c>where(y == 1, 1 − cos, relu(cos − margin))</c>
    /// (<see cref="LossReduction.None"/> semantics): returns the length-<c>N</c>
    /// loss tensor (one scalar per pair) without reduction.
    /// </summary>
    /// <param name="margin">Margin for the <c>y = −1</c> arm (default <c>0</c>).</param>
    /// <param name="eps">Denominator floor (default <c>1e-8</c>).</param>
    /// <param name="x1">First embedding batch <c>[N, D]</c>.</param>
    /// <param name="x2">Second embedding batch <c>[N, D]</c>.</param>
    /// <param name="y">Per-sample labels encoded as <c>±1</c>.</param>
    public static Tensor<float32> PerElement(
        Scalar<float32> margin,
        Scalar<float32> eps,
        Tensor<float32> x1,
        Tensor<float32> x2,
        Tensor<float32> y)
        => CosinePerElement(margin, eps, x1, x2, y);

    /// <summary>
    /// The shared per-sample core. Computes <c>cos(x1, x2)</c> over the last axis
    /// (<see cref="CosineSimilarity"/>), then selects branch-free between the
    /// pull arm <c>1 − cos</c> (where <c>y = +1</c>) and the hinged push arm
    /// <c>relu(cos − margin)</c> (where <c>y = −1</c>) via <c>where(y == 1, …)</c>,
    /// returning the <c>[N]</c> loss vector.
    /// </summary>
    internal static Tensor<float32> CosinePerElement(
        Scalar<float32> margin,
        Scalar<float32> eps,
        Tensor<float32> x1,
        Tensor<float32> x2,
        Tensor<float32> y)
    {
        var cos = CosineSimilarity(x1, x2, eps);     // [N]
        var pull = Scalar(1f) - cos;                 // y = +1 arm: 1 − cos        [N]
        var push = (cos - margin).Relu();            // y = −1 arm: max(0, cos − m) [N]
        return (y == Scalar(1f)).Where(pull, push);  // branch-free per-sample      [N]
    }

    /// <summary>
    /// Cosine similarity over the <b>last</b> axis (the reusable primitive — exactly
    /// PyTorch's <c>nn.CosineSimilarity</c> and Keras's <c>CosineSimilarity</c>
    /// metric): <c>cos_i = (Σ_feat x1·x2) / max(‖x1_i‖₂·‖x2_i‖₂, eps)</c>, returning
    /// the per-row similarity <c>[N]</c>. The <c>eps</c> floor clamps the
    /// <i>product</i> of norms (PyTorch's placement) so a zero vector yields a
    /// finite <c>0</c> rather than a NaN.
    /// </summary>
    /// <param name="x1">First batch <c>[N, D]</c>.</param>
    /// <param name="x2">Second batch <c>[N, D]</c>.</param>
    /// <param name="eps">Denominator floor (PyTorch default <c>1e-8</c>).</param>
    public static Tensor<float32> CosineSimilarity(
        Tensor<float32> x1, Tensor<float32> x2, Scalar<float32> eps)
    {
        var dot = (x1 * x2).Reduce(ReduceKind.Sum, [Scalar(-1L)], keepDims: false); // x1·x2  [N]
        var n1 = x1.Reduce(ReduceKind.L2, [Scalar(-1L)], keepDims: false);          // ‖x1‖₂  [N]
        var n2 = x2.Reduce(ReduceKind.L2, [Scalar(-1L)], keepDims: false);          // ‖x2‖₂  [N]
        var denom = (n1 * n2).Clip(eps, Scalar(float.PositiveInfinity));            // max(‖x1‖‖x2‖, eps) [N]
        return dot / denom;                                                          // cos    [N]
    }
}
