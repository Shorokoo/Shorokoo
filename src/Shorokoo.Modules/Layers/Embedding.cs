using System;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Modules.Initializers;
using Shorokoo.Onnx;
using static Shorokoo.Globals;

namespace Shorokoo.Modules.Layers;

/// <summary>
/// Embedding lookup table: maps int64 indices of any shape <c>[...]</c> to
/// embeddings of shape <c>[..., embeddingDim]</c> via Gather over a trainable
/// weight <c>[numEmbeddings, embeddingDim]</c>, initialized N(0, 1)
/// (<see cref="Normal"/>, the PyTorch nn.Embedding default).
///
/// <para>
/// <c>paddingIdx</c> (sentinel <c>-1</c> = off): rows whose index equals
/// <c>paddingIdx</c> are masked to the zero vector in the FORWARD OUTPUT (and so
/// receive no training gradient). <b>Divergence:</b> PyTorch zeroes the stored
/// weight row and freezes its gradient in the backward kernel; Shorokoo masks the
/// OUTPUT (the stored row keeps its init value, which is never observed). Under
/// the standard "pad id is reserved" convention the training effect matches.
/// </para>
///
/// <para>
/// <c>maxNorm</c> (sentinel <c>0f</c> = off) / <c>normType</c> (default <c>2f</c>,
/// the p of the p-norm): the GATHERED output rows whose <c>normType</c>-norm
/// exceeds <c>maxNorm</c> are scaled down to <c>maxNorm</c> (shrink-only).
/// <b>Divergence:</b> PyTorch renormalizes the STORED weight IN PLACE at forward
/// time (a persistent side effect that compounds across training); Shorokoo
/// clamps the OUTPUT functionally and never mutates the weight, recomputing the
/// clamp fresh each forward. Per-step / inference outputs match for binding rows;
/// the stored weight diverges across training.
/// </para>
///
/// <para>
/// <c>scale_grad_by_freq</c> and <c>sparse</c> are intentionally NOT exposed:
/// both are gradient-only knobs (frequency reweighting / sparse-gradient storage)
/// with no SSA-graph forward expression and no Shorokoo IR support.
/// </para>
///
/// <para>
/// To choose a different weight initializer, use
/// <see cref="EmbeddingHelpers.Embed"/> (an initializer is a compile-time type,
/// not a runtime <c>[Hyper]</c> scalar).
/// </para>
/// </summary>
[Module]
public partial class Embedding
{
    public static Tensor<float32> Inline(
        Tensor<int64> indices,
        [Hyper] Scalar<int64> numEmbeddings,
        [Hyper] Scalar<int64> embeddingDim,
        [Hyper] Scalar<int64> paddingIdx,           // -1 = off (default supplied at the call site)
        [Hyper] Scalar<float32> maxNorm,            // 0f = off (default supplied at the call site)
        [Hyper] Scalar<float32> normType)           // 2f (only used when maxNorm > 0; default at call site)
    {
        var weight = Normal.Init([numEmbeddings, embeddingDim]);
        return EmbeddingHelpers.EmbeddingCore(weight, indices, paddingIdx, maxNorm, normType);
    }
}

/// <summary>
/// Embedding graph-building helpers, hosting the init-choosing overload (Keras
/// <c>embeddings_initializer</c> / Flax <c>embedding_init</c> parity). Like
/// <see cref="Recurrent"/> / <see cref="Convolution"/> / <see cref="Pooling"/>,
/// the initializer is a compile-time structural choice (a type, not a runtime
/// scalar — it cannot be a <c>[Hyper]</c>), so it lives on this plain-C#-argument
/// static helper rather than on the rig-trainable <c>[Module] Embedding</c>.
/// </summary>
public static class EmbeddingHelpers
{
    /// <summary>
    /// Embedding lookup with a caller-chosen weight initializer. Defaults to
    /// <see cref="Normal"/> (the <c>[Module] Embedding</c>'s hardcoded default), so
    /// this strictly generalizes it. <paramref name="paddingIdx"/> /
    /// <paramref name="maxNorm"/> / <paramref name="normType"/> carry the same
    /// semantics and sentinels as <see cref="Embedding"/>.
    /// </summary>
    /// <param name="indices">Int64 index tensor of any shape <c>[...]</c>.</param>
    /// <param name="numEmbeddings">Vocabulary size <c>V</c> (weight rows).</param>
    /// <param name="embeddingDim">Embedding width <c>D</c> (weight columns).</param>
    /// <param name="embeddingInit">
    /// Weight initializer selector, e.g. <c>shape =&gt; XavierUniform.Init(shape)</c>.
    /// Null (default) uses <see cref="Normal"/>.
    /// </param>
    /// <param name="paddingIdx">Pad index whose output rows are masked to zero; <c>-1</c> (default) disables.</param>
    /// <param name="maxNorm">Max-norm cap on the gathered output rows; <c>0f</c> (default) disables.</param>
    /// <param name="normType">The p of the max-norm p-norm; <c>2f</c> (default, Euclidean). Only used when <paramref name="maxNorm"/> &gt; 0.</param>
    /// <returns>Embeddings of shape <c>[..., embeddingDim]</c>.</returns>
    public static Tensor<float32> Embed(
        Tensor<int64> indices,
        long numEmbeddings,
        long embeddingDim,
        Func<Vector<int64>, Tensor<float32>>? embeddingInit = null,
        long paddingIdx = -1,
        float maxNorm = 0f,
        float normType = 2f)
    {
        var init = embeddingInit ?? (shape => Normal.Init(shape));
        var weight = init([Scalar(numEmbeddings), Scalar(embeddingDim)]);
        return EmbeddingCore(weight, indices,
            Scalar(paddingIdx), Scalar(maxNorm), Scalar(normType));
    }

    /// <summary>
    /// The shared in-graph embedding wiring used by both the <c>[Module] Embedding</c>
    /// and <see cref="Embed"/>: Gather, then the <c>paddingIdx</c> output-mask and the
    /// <c>maxNorm</c>/<c>normType</c> output-renorm, each gated on its disable sentinel
    /// via <c>IfElse</c> so the default path is byte-identical to a plain Gather.
    /// </summary>
    internal static Tensor<float32> EmbeddingCore(
        Tensor<float32> weight,
        Tensor<int64> indices,
        Scalar<int64> paddingIdx,
        Scalar<float32> maxNorm,
        Scalar<float32> normType)
    {
        var gathered = weight.Gather(indices, axis: 0);              // [..., D]

        // ---- paddingIdx: mask the gathered pad rows to zero ----
        // isPad : [...] bit (the Scalar hyper broadcasts against the index tensor).
        var isPad = indices == paddingIdx;                          // Tensor<bit> [...]
        var padMask = isPad.Unsqueeze(-1);                          // [..., 1] -> broadcasts over D
        var zeros = gathered * 0f;                         // [..., D] zeros, same shape
        var padded = padMask.Where(zeros, gathered);                // pad rows -> 0, else gathered
        // Gate on the -1 sentinel so "off" is a true no-op:
        var afterPad = (paddingIdx < 0L).IfElse(gathered, padded);

        // ---- maxNorm / normType: shrink-only renorm of the OUTPUT rows ----
        // per-row p-norm over the last axis (keepDims), general p via Abs/Pow:
        var rowNorm = afterPad.Abs().Pow(normType)
            .Reduce(ReduceKind.Sum, [Scalar(-1L)], keepDims: true)  // [..., 1]
            .Pow(1f / normType);                            // (Σ|x|^p)^(1/p)
        var scale = (maxNorm / rowNorm).Clip(0f, 1f); // min(1, maxNorm/norm)
        var renormed = afterPad * scale;                            // over-cap rows -> norm == maxNorm
        // Gate on the 0f sentinel so "off" is a true no-op:
        var afterNorm = (maxNorm <= Scalar(0f)).IfElse(afterPad, renormed);

        return afterNorm;
    }
}

/// <summary>The per-bag reduction of an <see cref="EmbeddingBag"/> (PyTorch's <c>mode</c>).</summary>
public enum BagMode
{
    /// <summary>Sum the bag's embeddings (PyTorch <c>mode='sum'</c>; the bag-of-words sum).</summary>
    Sum,
    /// <summary>Mean of the bag's embeddings (PyTorch <c>mode='mean'</c>; PyTorch's default).</summary>
    Mean,
    /// <summary>Per-feature max over the bag (PyTorch <c>mode='max'</c>).</summary>
    Max,
}

/// <summary>
/// EmbeddingBag: for a 2-D batch of fixed-length bags <c>indices [B, L]</c>, looks up
/// a trainable table <c>W [V, D]</c> (Normal N(0, 1), like <see cref="Embedding"/>)
/// and reduces each bag (axis 1) by <c>mode</c> → <c>[B, D]</c>. Equivalent to
/// <c>Embedding(input).Reduce(mode, axis=1)</c>.
///
/// <para>
/// Like <see cref="Recurrent"/> / <see cref="Pooling"/> / <see cref="Convolution"/>,
/// this is a plain-C#-argument <c>static</c> helper rather than a <c>[Module]</c>:
/// <c>mode</c> is a structural enum that selects the reduce op (it cannot be a
/// scalar-only <c>[Hyper]</c>), so it is baked at build time. The owned
/// <c>Normal.Init([V, D])</c> table is still a trainable parameter that flows
/// through the rig via the owning model's graph, so an EmbeddingBag trains
/// end-to-end despite not being an independently <c>.Call</c>/<c>.Model</c>-able
/// <c>[Module]</c>.
/// </para>
///
/// <para>
/// <b>SCOPE / DIVERGENCE (loud):</b> this accepts ONLY the 2-D fixed-length form.
/// PyTorch's 1-D <c>input + offsets</c> ragged (variable-length-bag) form,
/// <c>include_last_offset</c>, and <c>per_sample_weights</c> are NOT supported —
/// the ragged segment-reduce needs a SegmentSum-style op that ONNX/Shorokoo lacks.
/// Users with ragged bags must rectangularize to <c>[B, L]</c> (pad to the max bag
/// length and use <c>paddingIdx</c> to zero the pad rows).
/// </para>
///
/// <para>
/// <b>No fused kernel — the <c>[B, L, D]</c> intermediate IS materialized.</b> PyTorch's
/// whole reason to exist is to avoid instantiating the intermediate embeddings;
/// Shorokoo's EmbeddingBag is literally <c>Gather</c> (which produces <c>[B, L, D]</c>)
/// then <c>Reduce</c>. The numerical result is identical; the memory/perf
/// characteristic is not — there is no fused gather-reduce op in the ONNX-shaped op
/// set, and none is required for correctness.
/// </para>
///
/// <para>
/// <b><c>paddingIdx</c> semantics &amp; the Max caveat (documented divergence).</b>
/// Pad rows (whose index equals <c>paddingIdx</c>) are masked to the
/// zero vector BEFORE the reduce. This is exact for <see cref="BagMode.Sum"/> (a
/// zero summand doesn't move the sum); approximate for <see cref="BagMode.Mean"/>
/// (Shorokoo divides by the full <c>L</c>, whereas PyTorch's <c>mode='mean'</c>
/// divides by the NON-pad bag length); and not faithful for <see cref="BagMode.Max"/>
/// when embeddings can be negative (the injected <c>0</c> would win, whereas PyTorch
/// EXCLUDES the pad row from the max) — use <c>paddingIdx</c> with
/// <see cref="BagMode.Max"/> only when embeddings are known non-negative, or leave
/// it off.
/// </para>
/// </summary>
public static class EmbeddingBag
{
    /// <param name="indices">int64 bag indices, shape <c>[B, L]</c> (<c>B</c> fixed-length bags of length <c>L</c>).</param>
    /// <param name="numEmbeddings">Vocabulary size <c>V</c> (weight rows).</param>
    /// <param name="embeddingDim">Embedding width <c>D</c> (weight columns).</param>
    /// <param name="mode">Per-bag reduction; default <see cref="BagMode.Mean"/> (PyTorch's default).</param>
    /// <param name="embeddingInit">
    /// Weight initializer selector, e.g. <c>shape =&gt; XavierUniform.Init(shape)</c>.
    /// Null (default) uses <see cref="Normal"/>, à la <see cref="EmbeddingHelpers.Embed"/>.
    /// </param>
    /// <param name="paddingIdx">
    /// Pad index whose rows are masked to zero BEFORE the reduce (so they don't
    /// contribute to Sum/Mean; see the type summary for the Mean denominator and Max
    /// caveats). <c>-1</c> (default) disables.
    /// </param>
    /// <returns>Reduced embeddings, shape <c>[B, D]</c>.</returns>
    public static Tensor<float32> Bag(
        Tensor<int64> indices,
        long numEmbeddings,
        long embeddingDim,
        BagMode mode = BagMode.Mean,
        Func<Vector<int64>, Tensor<float32>>? embeddingInit = null,
        long paddingIdx = -1)
    {
        var init = embeddingInit ?? (shape => Normal.Init(shape));
        var weight = init([Scalar(numEmbeddings), Scalar(embeddingDim)]);    // [V, D]
        var gathered = weight.Gather(indices, axis: 0);                      // [B, L, D]

        // padding_idx: mask pad rows to zero before reducing (see summary caveat for Max).
        if (paddingIdx >= 0)
        {
            var isPad = indices == Scalar(paddingIdx);                       // [B, L] bit
            var padMask = isPad.Unsqueeze(-1);                               // [B, L, 1] broadcasts over D
            gathered = padMask.Where(gathered * 0f, gathered);       // pad rows -> 0
        }

        var kind = mode switch
        {
            BagMode.Sum => ReduceKind.Sum,
            BagMode.Max => ReduceKind.Max,
            _ => ReduceKind.Mean,
        };
        return gathered.Reduce(kind, axes: Vector(1L), keepDims: false);     // [B, D]
    }
}
