using System;
using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;
using static Shorokoo.Globals;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Nodes.AutoDiff
{
    /// <summary>
    /// Multi-output / multi-input gradient implementations that need the variadic
    /// <c>(inputs, outputGrads, attributes) =&gt; Variable?[]</c> shape because their
    /// natural method signature does not fit the [AutoDiff] reflection model
    /// (more than one differentiable output, optional inputs whose absence changes
    /// the gradient layout, or sequence outputs).
    /// </summary>
    internal static partial class AutoDiffs
    {
        // ===== LayerNormalization =====
        //
        // Forward (3 outputs): y = scale * (x - mean) * invStd + bias (where bias is optional).
        //   Normalization is over the suffix axes starting at `axis` (default -1).
        //
        // Gradient w.r.t. x uses the standard batch-norm-style closed form:
        //   dx = invStd * (dY_scaled - mean(dY_scaled, axes, keep) -
        //                  xCentered * mean(dY_scaled * xCentered, axes, keep) * invStd^2)
        // where dY_scaled = grad * scale (broadcast).
        //
        // Gradient w.r.t. scale = sum(grad * xHat) over the non-feature axes.
        // Gradient w.r.t. bias  = sum(grad) over the non-feature axes.
        //
        // Only outputGrads[0] (the y-grad) is consulted; gradients flowing back into the
        // mean/invStd statistics outputs would be unusual in practice — accumulating them
        // here would just shift the same closed form so we treat them as zero.

        internal static Variable?[] LayerNormalizationGradient(
            Variable?[] inputs, Variable?[] outputGrads, OnnxCSharpAttributes attributes)
        {
            var x = inputs[0]!;
            var scale = inputs[1]!;
            var bias = inputs.Length > 2 ? inputs[2] : null;
            var grad = outputGrads[0]!;

            var axisAttr = (attributes.GetAttributeObj(AttrAxis) as long?) ?? -1;
            var epsilon = (attributes.GetAttributeObj(AttrEpsilon) as float?) ?? 1e-5f;

            var floatType = x.Type;
            var epsConst = OnnxOp.Cast(Scalar(epsilon), saturate: null, to: floatType);

            // Runtime axis arithmetic (don't require a static rank — InstanceNormalization
            // and GroupNormalization use the same pattern).
            //   xRankScalar = rank(x)
            //   effectiveAxisScalar = axisAttr if >= 0 else rank + axisAttr
            //   reduceAxes = Range(effectiveAxisScalar, xRankScalar, 1)
            //   nonFeatureAxes = Range(0, effectiveAxisScalar, 1)
            var xShape = OnnxOp.Shape(x);
            var xRankShape = OnnxOp.Shape(xShape);                                     // [1] containing rank
            var xRankScalar = OnnxOp.Squeeze(xRankShape, Vector(0L));
            var effectiveAxisScalar = axisAttr >= 0
                ? (Variable)Scalar(axisAttr)
                : OnnxOp.Add(xRankScalar, Scalar(axisAttr));
            var reduceAxes = OnnxOp.Range(effectiveAxisScalar, xRankScalar, Scalar(1L));
            var nonFeatureAxes = OnnxOp.Range(Scalar(0L), effectiveAxisScalar, Scalar(1L));

            // Per-feature stats
            var mean = OnnxOp.ReduceMean(x, reduceAxes, keepdims: true);
            var xCentered = OnnxOp.Sub(x, mean);
            var variance = OnnxOp.ReduceMean(OnnxOp.Mul(xCentered, xCentered), reduceAxes, keepdims: true);
            var invStd = OnnxOp.Reciprocal(OnnxOp.Sqrt(OnnxOp.Add(variance, epsConst)));
            var xHat = OnnxOp.Mul(xCentered, invStd);

            // dx (standard batch-norm-style closed form, dscale = scale broadcast).
            var gradScaled = OnnxOp.Mul(grad, scale);
            var meanGrad = OnnxOp.ReduceMean(gradScaled, reduceAxes, keepdims: true);
            var meanGradXc = OnnxOp.ReduceMean(OnnxOp.Mul(gradScaled, xCentered), reduceAxes, keepdims: true);
            var dx = OnnxOp.Mul(invStd,
                OnnxOp.Sub(
                    OnnxOp.Sub(gradScaled, meanGrad),
                    OnnxOp.Mul(OnnxOp.Mul(OnnxOp.Mul(xCentered, meanGradXc), invStd), invStd)));

            // dscale / dbias: reduce over the non-feature axes. When axis == 0 the axes set
            // is empty and ReduceSum becomes identity (noopWithEmptyAxes=true).
            var dScale = OnnxOp.ReduceSum(OnnxOp.Mul(grad, xHat), nonFeatureAxes,
                keepdims: false, noopWithEmptyAxes: true);
            var dBias = OnnxOp.ReduceSum(grad, nonFeatureAxes,
                keepdims: false, noopWithEmptyAxes: true);

            return inputs.Length > 2 ? [dx, dScale, bias is null ? null : dBias] : [dx, dScale];
        }

        // ===== NegativeLogLikelihoodLoss =====
        //
        // Forward:
        //   loss[i, d1, ...] = -weight[t] * input[i, t, d1, ...]   where t = target[i, d1, ...]
        //   reduction in {"none", "sum", "mean"} aggregates over [i, d1, ...]; ignore_index
        //   contributions are masked out.
        //
        // Gradient w.r.t. input:
        //   dinput[i, c, d1, ...] = -(weight[c] if weight else 1) * onehot(target=c) * upstream
        //   For "sum"  : upstream = grad (scalar broadcast)
        //   For "mean" : upstream = grad / Σ effective_weights
        //   For "none" : upstream = grad[i, d1, ...]
        // ignore_index masks both the loss contribution and the gradient.

        internal static Variable?[] NegativeLogLikelihoodLossGradient(
            Variable?[] inputs, Variable?[] outputGrads, OnnxCSharpAttributes attributes)
        {
            var input = inputs[0]!;
            var target = inputs[1]!;
            var weight = inputs.Length > 2 ? inputs[2] : null;
            var grad = outputGrads[0]!;

            var ignoreIndex = attributes.GetAttributeObj(AttrIgnoreIndex) as long?;
            var reduction = (attributes.GetAttributeObj(AttrReduction) as string) ?? "mean";

            var floatType = input.Type;

            // Build a [C]-shaped per-class weight vector (defaulting to 1.0 when absent).
            var cVec = OnnxOp.Slice(OnnxOp.Shape(input), Vector(1L), Vector(2L));
            Variable wVec;
            if (weight is null)
            {
                wVec = OnnxOp.Expand(OnnxOp.Cast(Scalar(1.0f), saturate: null, to: floatType), cVec);
            }
            else
            {
                wVec = weight;
            }

            // weightPerSample[i, d1, ...] = weight[target[i, d1, ...]]  (gather along axis 0)
            var weightPerSample = OnnxOp.Gather(wVec, target, axis: 0);

            // ignore_index masking: zero out the per-class weight at samples whose target
            // equals ignore_index so they contribute neither to dInput nor to totalWeight.
            // Cast target to int64 explicitly so the stand-in dtype (which the variadic
            // gradient dispatcher defaults to float32) doesn't break the Equal-op type check.
            if (ignoreIndex is long ig)
            {
                var targetInt = OnnxOp.Cast(target, saturate: null, to: DType.Int64);
                var notIgnored = OnnxOp.Not(OnnxOp.Equal(targetInt, Scalar(ig)));
                var activeMask = OnnxOp.Cast(notIgnored, saturate: null, to: floatType);
                weightPerSample = OnnxOp.Mul(weightPerSample, activeMask);
            }

            // upstream gradient shaped [N, d1, ...]
            Variable upstream;
            if (reduction == "none")
            {
                upstream = grad;
            }
            else if (reduction == "sum")
            {
                upstream = OnnxOp.Expand(grad, OnnxOp.Shape(target));
            }
            else // "mean"
            {
                var totalWeight = OnnxOp.ReduceSum(weightPerSample, axes: null, keepdims: false, noopWithEmptyAxes: false);
                var scaled = OnnxOp.Div(grad, totalWeight);
                upstream = OnnxOp.Expand(scaled, OnnxOp.Shape(target));
            }

            // Per-sample scalar gradient contribution: -weight[target] * upstream
            var negOne = OnnxOp.Cast(Scalar(-1.0f), saturate: null, to: floatType);
            var perSample = OnnxOp.Mul(negOne, OnnxOp.Mul(weightPerSample, upstream));

            // OneHot the target along the channel axis (axis=1 in input). values = [0, 1] in input dtype.
            var depthScalar = OnnxOp.Gather(OnnxOp.Shape(input), Scalar(1L), axis: 0);
            var values = OnnxOp.Cast(Vector(0.0f, 1.0f), saturate: null, to: floatType);
            var onehot = OnnxOp.OneHot(target, depthScalar, values, axis: 1);

            // Broadcast perSample to [N, 1, d1, ...] and multiply by onehot to get dinput shape [N, C, d1, ...].
            var perSampleExp = OnnxOp.Unsqueeze(perSample, Vector(1L));
            var dInput = OnnxOp.Mul(perSampleExp, onehot);

            var result = new Variable?[inputs.Length];
            result[0] = dInput;
            // target gradient = null (integer indices, non-diff)
            // weight gradient = null (treated as constant)
            return result;
        }

        // ===== SoftmaxCrossEntropyLoss =====
        //
        // Forward (2 outputs):
        //   log_prob = LogSoftmax(scores, axis=1)
        //   loss     = NegativeLogLikelihoodLoss(log_prob, labels, weight, ignore_index, reduction)
        //
        // Gradient w.r.t. scores closed-form (standard CE backward):
        //   dscores[i, c, ...] = (softmax(scores)[i, c, ...] - onehot(labels)[i, c, ...]) *
        //                        weight[labels] * upstream * active_mask
        //   With "mean" reduction the upstream divides by Σ effective weights; "sum"
        //   broadcasts the scalar grad; "none" uses the per-element grad directly.
        // Gradients into the log_prob output (outputGrads[1]) flow back through LogSoftmax.

        internal static Variable?[] SoftmaxCrossEntropyLossGradient(
            Variable?[] inputs, Variable?[] outputGrads, OnnxCSharpAttributes attributes)
        {
            var scores = inputs[0]!;
            var labels = inputs[1]!;
            var weight = inputs.Length > 2 ? inputs[2] : null;

            var lossGrad = outputGrads.Length > 0 ? outputGrads[0] : null;
            var logProbGrad = outputGrads.Length > 1 ? outputGrads[1] : null;

            var ignoreIndex = attributes.GetAttributeObj(AttrIgnoreIndex) as long?;
            var reduction = (attributes.GetAttributeObj(AttrReduction) as string) ?? "mean";

            var floatType = scores.Type;

            // softmax(scores, axis=1) — the channel axis is always 1 in ONNX SCEL.
            var softmax = OnnxOp.Softmax(scores, axis: 1);

            // weight vector and ignore mask (same shape as labels)
            var cScalar = OnnxOp.Gather(OnnxOp.Shape(scores), Scalar(1L), axis: 0);
            Variable wVec;
            if (weight is null)
            {
                wVec = OnnxOp.Expand(OnnxOp.Cast(Scalar(1.0f), saturate: null, to: floatType), OnnxOp.Reshape(cScalar, Vector(1L), allowZero: false));
            }
            else
            {
                wVec = weight;
            }
            var weightPerSample = OnnxOp.Gather(wVec, labels, axis: 0);

            if (ignoreIndex is long ig)
            {
                var labelsInt = OnnxOp.Cast(labels, saturate: null, to: DType.Int64);
                var notIgnored = OnnxOp.Not(OnnxOp.Equal(labelsInt, Scalar(ig)));
                var activeMaskFloat = OnnxOp.Cast(notIgnored, saturate: null, to: floatType);
                weightPerSample = OnnxOp.Mul(weightPerSample, activeMaskFloat);
            }

            // Upstream gradient (shape of labels) for the loss path
            Variable upstream;
            if (lossGrad is null)
            {
                // Loss output unused; zero-grad path for this term.
                upstream = OnnxOp.Expand(OnnxOp.Cast(Scalar(0.0f), saturate: null, to: floatType), OnnxOp.Shape(labels));
            }
            else if (reduction == "none")
            {
                upstream = lossGrad;
            }
            else if (reduction == "sum")
            {
                upstream = OnnxOp.Expand(lossGrad, OnnxOp.Shape(labels));
            }
            else // mean
            {
                var totalWeight = OnnxOp.ReduceSum(weightPerSample, axes: null, keepdims: false, noopWithEmptyAxes: false);
                var scaled = OnnxOp.Div(lossGrad, totalWeight);
                upstream = OnnxOp.Expand(scaled, OnnxOp.Shape(labels));
            }

            // OneHot the labels along axis 1 in scores' shape.
            var depthScalar = OnnxOp.Gather(OnnxOp.Shape(scores), Scalar(1L), axis: 0);
            var values = OnnxOp.Cast(Vector(0.0f, 1.0f), saturate: null, to: floatType);
            var onehot = OnnxOp.OneHot(labels, depthScalar, values, axis: 1);

            // dscores from the loss term: (softmax - onehot) * weight[labels] * upstream
            var scale1d = OnnxOp.Mul(weightPerSample, upstream);
            var scaleBroadcast = OnnxOp.Unsqueeze(scale1d, Vector(1L));
            var dScoresLoss = OnnxOp.Mul(OnnxOp.Sub(softmax, onehot), scaleBroadcast);

            // dscores from the log_prob output (if used downstream): LogSoftmax gradient
            // applied to logProbGrad. Closed form: grad - softmax * sum(grad, axis=1, keepdims).
            Variable dScores = dScoresLoss;
            if (logProbGrad is not null)
            {
                var sumLp = OnnxOp.ReduceSum(logProbGrad, axes: Vector(1L), keepdims: true, noopWithEmptyAxes: false);
                var dFromLp = OnnxOp.Sub(logProbGrad, OnnxOp.Mul(softmax, sumLp));
                dScores = OnnxOp.Add(dScores, dFromLp);
            }

            var result = new Variable?[inputs.Length];
            result[0] = dScores;
            // labels grad = null; weight grad = null
            return result;
        }

        // ===== STFT =====
        //
        // Forward: y[batch, frame, k, 2] = STFT(signal[batch, T, 1|2], frame_step,
        //                                       window?, frame_length?, onesided?)
        //   with L = frame_length (or window length), S = frame_step,
        //   k = 0..K-1 where K = floor(L/2)+1 (onesided, the DEFAULT) or L (two-sided):
        //     y[b, m, k] = Σ_n signal[b, m·S + n] · window[n] · e^{-i2πkn/L}
        //
        // STFT is linear in the signal, so the adjoint maps gradients back through the
        // transpose of the same operator:
        //   1. zero-pad the onesided gradient bins back to L along the frequency axis
        //      (truncated bins are not outputs → zero adjoint, like the DFT gradient);
        //   2. per-frame DFT adjoint: g = L · IDFT(dY_padded) along axis 2 (complex);
        //   3. slice the complex channel dim down to the signal's own channel count
        //      (1 = real signal keeps the real part; 2 = complex keeps both);
        //   4. dWindow[n] = Σ_{b,m,c} g[b,m,n,c] · signal[b, m·S+n, c]   (window is real);
        //   5. dSignal = overlap-add of g·window: ScatterElements(Add) of the flattened
        //      frames onto a zeros[B, T, C] base at indices m·S + n along the time axis.
        //
        // (This replaces the AD-B1 ZERO-STUB that silently returned null gradients —
        // a silently frozen parameter is the worst failure mode for training.)

        internal static Variable?[] STFTGradient(
            Variable?[] inputs, Variable?[] outputGrads, OnnxCSharpAttributes attributes)
        {
            var signal = inputs[0]!;                                     // [B, T] or [B, T, 1|2]
            var frameStep = inputs[1]!;                                  // int64 scalar
            var window = inputs.Length > 2 ? inputs[2] : null;           // [L] (optional)
            var frameLength = inputs.Length > 3 ? inputs[3] : null;      // int64 scalar (optional)
            var dY = outputGrads[0]!;                                    // [B, F, K, 2]

            // onesided DEFAULTS TO 1 in the ONNX spec.
            var onesidedRaw = attributes.GetAttributeObj(AttrOnesided);
            var onesided = onesidedRaw is null || onesidedRaw is true
                || (onesidedRaw is long lo && lo != 0);

            var floatType = dY.Type;
            var zeroF = OnnxOp.Cast(Scalar(0.0f), saturate: null, to: floatType);

            // The variadic gradient dispatcher synthesizes float32 stand-ins for every
            // input slot, so the int64 frame_step / frame_length scalars must be cast
            // back explicitly before they feed integer ops (Range/Mul/Sub).
            var stepInt = OnnxOp.Cast(frameStep, saturate: null, to: DType.Int64);

            // Effective frame length L (scalar): frame_length input, else window length.
            var lScalar = frameLength is not null
                ? OnnxOp.Cast(frameLength, saturate: null, to: DType.Int64)
                : OnnxOp.Gather(OnnxOp.Shape(window!), Scalar(0L), axis: 0);

            // Normalize the signal to its 3-D [B, T, C] form. The stand-in carries the
            // host tensor's rank; rank 2 means an implicit single (real) channel. When the
            // rank is unknown, assume the canonical 3-D layout.
            var signalRank = signal.Rank;
            var sigShape = OnnxOp.Shape(signal);
            var bVec = OnnxOp.Slice(sigShape, Vector(0L), Vector(1L));   // [B]
            var tVec = OnnxOp.Slice(sigShape, Vector(1L), Vector(2L));   // [T]
            var cVec = signalRank == 2
                ? (Variable)Vector(1L)
                : OnnxOp.Slice(sigShape, Vector(2L), Vector(3L));        // [C]
            var sig3Shape = OnnxOp.Concat([bVec, tVec, cVec], axis: 0);  // [B, T, C]
            var sig3 = signalRank == 2
                ? OnnxOp.Reshape(signal, sig3Shape, allowZero: false)
                : signal;

            // 1. onesided: zero-pad the K bins back to L along the frequency axis (2).
            var effectiveGrad = dY;
            if (onesided)
            {
                var kScalar = OnnxOp.Gather(OnnxOp.Shape(dY), Scalar(2L), axis: 0);
                var padAmount = OnnxOp.Reshape(OnnxOp.Sub(lScalar, kScalar), Vector(1L), allowZero: false);
                var pads = OnnxOp.Concat([Vector(0L), padAmount], axis: 0);
                effectiveGrad = OnnxOp.Pad(dY, pads, constantValue: null,
                    axes: OnnxOp.Reshape(Scalar(2L), Vector(1L), allowZero: false));
            }

            // 2. DFT adjoint per frame: g = L · IDFT(dY_padded) along axis 2 → [B, F, L, 2]
            var lFloat = OnnxOp.Cast(lScalar, saturate: null, to: floatType);
            var gAdj = OnnxOp.Mul(
                OnnxOp.Dft(effectiveGrad, null, Scalar(2L), inverse: true, onesided: false),
                lFloat);

            // 3. Slice the complex channel down to the signal's channel count C → [B, F, L, C]
            var cEnd = signalRank == 2 ? (Variable)Vector(1L) : OnnxOp.Reshape(cVec, Vector(1L), allowZero: false);
            var gAdjC = OnnxOp.Slice(gAdj, Vector(0L), cEnd, Vector(-1L), null);

            // Frame gather indices idx[m, n] = m·S + n  → [F, L]
            var fScalar = OnnxOp.Gather(OnnxOp.Shape(dY), Scalar(1L), axis: 0);
            var frameStarts = OnnxOp.Mul(OnnxOp.Range(Scalar(0L), fScalar, Scalar(1L)), stepInt);   // [F]
            var inFrame = OnnxOp.Range(Scalar(0L), lScalar, Scalar(1L));                            // [L]
            var idx2d = OnnxOp.Add(
                OnnxOp.Unsqueeze(frameStarts, Vector(1L)),
                OnnxOp.Unsqueeze(inFrame, Vector(0L)));                                             // [F, L]

            // 4. dWindow[n] = Σ_{b,m,c} gAdjC[b,m,n,c] · signal[b, m·S+n, c]
            //    (BEFORE the window scaling — the adjoint of z = w·x w.r.t. real w is
            //    Σ_c Re-pair products, which the channel-sliced gAdjC already encodes.)
            Variable? dWindow = null;
            if (window is not null)
            {
                var frames = OnnxOp.Gather(sig3, idx2d, axis: 1);        // [B, F, L, C]
                dWindow = OnnxOp.ReduceSum(OnnxOp.Mul(gAdjC, frames),
                    Vector(0L, 1L, 3L), keepdims: false, noopWithEmptyAxes: null);  // [L]
            }

            // 5. dSignal: scale by the window (adjoint of the forward's window multiply),
            //    then overlap-add the frames onto the time axis.
            var gWindowed = window is not null
                ? OnnxOp.Mul(gAdjC, OnnxOp.Reshape(window, Vector(1L, 1L, -1L, 1L), allowZero: false))
                : gAdjC;                                                  // [B, F, L, C]

            var flatShape = OnnxOp.Concat([bVec, Vector(-1L), cVec], axis: 0);
            var gFlat = OnnxOp.Reshape(gWindowed, flatShape, allowZero: false);      // [B, F·L, C]
            var idxFlat = OnnxOp.Reshape(idx2d, Vector(1L, -1L, 1L), allowZero: false); // [1, F·L, 1]
            var idxExpanded = OnnxOp.Expand(idxFlat, OnnxOp.Shape(gFlat));            // [B, F·L, C]

            var zeros = OnnxOp.Expand(zeroF, sig3Shape);                              // [B, T, C]
            var dSig3 = OnnxOp.ScatterElements(zeros, idxExpanded, gFlat,
                axis: 1, reduction: ScatterNDReduction.Add);

            var dSignal = signalRank == 2
                ? OnnxOp.Reshape(dSig3, sigShape, allowZero: false)
                : dSig3;

            var result = new Variable?[inputs.Length];
            result[0] = dSignal;
            // frame_step (int64) non-differentiable
            if (inputs.Length > 2) result[2] = dWindow;
            // frame_length (int64) non-differentiable
            return result;
        }

        // ===== DeformConv =====
        //
        // Forward inputs: X, W, offset, [B, mask].
        // The true gradient requires the deformable-convolution adjoint, which mixes
        // bilinear-sampling Jacobians across the offset and mask channels (dX needs a
        // scatter of W-weighted grads onto the 4 bilinear corners per sample position,
        // dOffset needs the spatial derivative of the bilinear kernel, and dW/dB need the
        // sampled-patch images). That adjoint is not implemented; throw AD003 instead of
        // the previous ZERO-STUB (null gradients), which silently froze every trainable
        // parameter behind a DeformConv. Use a standard Conv when training end-to-end is
        // required.

        internal static Variable?[] DeformConvGradient(
            Variable?[] inputs, Variable?[] outputGrads, OnnxCSharpAttributes attributes)
        {
            _ = inputs;
            _ = outputGrads;
            _ = attributes;
            throw new AutoDiffNotSupportedException(ErrorCodes.AD003, DEFORM_CONV,
                "the DeformConv gradient (bilinear-sampling adjoint for dX/dW/dOffset/dMask) "
                + "is not implemented — training through it would silently freeze the "
                + "parameters behind it. This is an implementation limitation, not a "
                + "mathematical one. Use a standard Conv when the architecture must be "
                + "trained end-to-end, or detach DeformConv from the loss path.");
        }

        // ===== SplitToSequence =====
        //
        // Forward: sequence = SplitToSequence(input, [split], axis, keepdims).
        // Gradient: concatenate the sequence's gradient tensors along the split axis,
        // restoring keepdims=false by unsqueezing each element first.

        internal static Variable?[] SplitToSequenceGradient(
            Variable?[] inputs, Variable?[] outputGrads, OnnxCSharpAttributes attributes)
        {
            // Single-output op: the dispatcher skips invocation when outputGrads[0] is null,
            // so dSeq is guaranteed non-null here.
            var dSeq = outputGrads[0]!;

            var axisAttr = attributes.GetAttributeObj(AttrAxis) as long?;
            var keepdimsObj = attributes.GetAttributeObj(AttrKeepdims);
            var keepdims = keepdimsObj is bool kb ? kb : keepdimsObj is long kl ? kl != 0 : true;
            var axis = axisAttr ?? 0L;

            // Build a tensor sequence of grads-with-axis (insert axis if it was squeezed).
            var len = ((Tensor<int64>)OnnxOp.SequenceLength(dSeq)).Scalar();
            var pieces = OnnxOp.SequenceEmpty(dSeq.Type);
            foreach (var ctx in LoopAPI.Iterate(len))
            {
                var piece = OnnxOp.SequenceAt(dSeq, ctx.IterationIndex);
                if (!keepdims) piece = OnnxOp.Unsqueeze(piece, Vector(axis));
                pieces = OnnxOp.SequenceInsert(pieces, piece, null);
            }

            var dInput = OnnxOp.ConcatFromSequence(pieces, axis: axis, newAxis: false);

            var result = new Variable?[inputs.Length];
            result[0] = dInput;
            // split argument is non-differentiable (integer)
            return result;
        }

        // ===== Sequence/Optional/Internal sentinel ops =====
        //
        // These ops either produce non-differentiable outputs (length, has-element),
        // are pure structural plumbing in the Shorokoo graph (StateUpdateLink,
        // WithStateDeps, TrainableParamIdRef, SequenceConcat, SequenceSlice), or
        // wrap string/non-float values that have no float gradient. Each gradient
        // method returns an array of nulls of the right length so the autograd
        // dispatcher does not throw NotImplementedException.

        internal static Variable?[] NullInputGradient(
            Variable?[] inputs, Variable?[] outputGrads, OnnxCSharpAttributes attributes)
        {
            _ = outputGrads;
            _ = attributes;
            return new Variable?[inputs.Length];
        }
    }
}
