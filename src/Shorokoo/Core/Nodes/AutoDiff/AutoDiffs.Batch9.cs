using System;
using System.Diagnostics;
using System.Linq;
using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.OnnxNodes;
using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;
using static Shorokoo.Globals;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Nodes.AutoDiff
{
    internal static partial class AutoDiffs
    {
        // ===== GRU (Gated Recurrent Unit) =====
        //
        // Forward pass (default activations f=sigmoid, g=tanh, forward direction, layout=0):
        //   For each timestep t = 0..T-1:
        //     z_t = f(X_t @ Wz^T + H_{t-1} @ Rz^T + Wbz + Rbz)
        //     r_t = f(X_t @ Wr^T + H_{t-1} @ Rr^T + Wbr + Rbr)
        //     linear_before_reset=0: ht = g(X_t @ Wh^T + (r_t*H_{t-1}) @ Rh^T + Wbh + Rbh)
        //     linear_before_reset=1: ht = g(X_t @ Wh^T + r_t*(H_{t-1} @ Rh^T + Rbh) + Wbh)
        //     H_t = (1 - z_t) * ht + z_t * H_{t-1}
        //
        // Gradient via Backpropagation Through Time (BPTT):
        //   Recomputes forward intermediate values into per-timestep sequences, then
        //   propagates gradients backward through timesteps using the chain rule. The
        //   forward and backward sweeps are emitted as graph-level Loop nodes via
        //   LoopAPI.Iterate so the sequence length is read at runtime — the gradient
        //   builder does not need a concrete value of T at construction time.
        //
        // Supports: forward AND reverse direction (reverse reduces to the forward BPTT
        //           on a time-flipped x/dY with the resulting dX flipped back — see
        //           RnnGradient), default activations (sigmoid/tanh), layout=0,
        //           both linear_before_reset values, optional bias and initial hidden state.
        // Guarded (AD003): bidirectional, custom activations, clip, layout=1,
        //           wired sequence_lens.

        internal static Variable?[] GruGradient(Variable?[] inputs, Variable?[] outputGrads, OnnxCSharpAttributes attributes)
        {
            var x = inputs[0]!;          // [T, B, I]
            var w = inputs[1]!;          // [D, 3H, I]  (D = num_directions = 1)
            var r = inputs[2]!;          // [D, 3H, H]
            var b = inputs[3];           // [D, 6H] (optional)
            var sequenceLens = inputs[4]; // (not differentiable; unsupported when wired)
            var initialH = inputs[5];    // [D, B, H] (optional)

            var dY = outputGrads[0];     // [T, D, B, H] (optional)
            var dYh = outputGrads[1];    // [D, B, H] (optional)

            // At least one of dY/dYh is non-null: FastProcessAutoGrad/AutoDiffEngine
            // skip multi-output gradient nodes whose entire outputGrads array is null
            // before invoking the gradient method.
            Debug.Assert(dY is not null || dYh is not null);

            var hiddenSize = (long)attributes.GetAttributeObj("hidden_size")!;
            var lbrObj = attributes.GetAttributeObj("linear_before_reset");
            var linearBeforeReset = lbrObj is true || (lbrObj is long lbrLong && lbrLong != 0);

            var direction = ResolveRecurrentDirection(attributes.GetAttributeObj("direction"));
            GuardRecurrentAttributeEnvelope(GRU, attributes, direction, sequenceLens, ["Sigmoid", "Tanh"]);

            // direction='reverse': flip x and dY along time, run the forward-direction
            // BPTT, and flip the resulting dX back (see RnnGradient).
            var isReverse = direction == "reverse";
            if (isReverse)
            {
                x = ReverseTimeAxis(x);
                if (dY is not null) dY = ReverseTimeAxis(dY);
            }

            // Sequence length as a runtime Variable (no host-side execution).
            var seqLen = ((Tensor<int64>)OnnxOp.Gather(OnnxOp.Shape(x), Scalar(0L), axis: 0)).Scalar();

            var H = hiddenSize;
            var one = OnnxOp.Cast(Scalar(1.0f), saturate: null, to: x.Type);

            // Squeeze num_directions dimension (=1) from weights
            var wSq = OnnxOp.Squeeze(w, Vector(0L));  // [3H, I]
            var rSq = OnnxOp.Squeeze(r, Vector(0L));  // [3H, H]

            // Split input weights: W = [Wz; Wr; Wh], each [H, I]
            var splitH = Vector(H, H, H);
            var wParts = OnnxOp.Split(wSq, splitH, axis: 0, numOutputs: null, variadicOutputCount: 3);
            var Wz = wParts[0];
            var Wr = wParts[1];
            var Wh = wParts[2];

            // Split recurrent weights: R = [Rz; Rr; Rh], each [H, H]
            var rParts = OnnxOp.Split(rSq, splitH, axis: 0, numOutputs: null, variadicOutputCount: 3);
            var Rz = rParts[0];
            var Rr = rParts[1];
            var Rh = rParts[2];

            // Split biases if present: B = [Wbz, Wbr, Wbh, Rbz, Rbr, Rbh], each [H]
            Variable? Wbz = null, Wbr = null, Wbh = null, Rbz = null, Rbr = null, Rbh = null;
            Variable? bSq = null;
            if (b is not null)
            {
                bSq = OnnxOp.Squeeze(b, Vector(0L));  // [6H]
                var bParts = OnnxOp.Split(bSq, Vector(H, H, H, H, H, H), axis: 0,
                    numOutputs: null, variadicOutputCount: 6);
                Wbz = bParts[0]; Wbr = bParts[1]; Wbh = bParts[2];
                Rbz = bParts[3]; Rbr = bParts[4]; Rbh = bParts[5];
            }

            // Initial hidden state: [B, H]
            Variable h0;
            if (initialH is not null)
            {
                h0 = OnnxOp.Squeeze(initialH, Vector(0L));  // [B, H]
            }
            else
            {
                var batchDim = OnnxOp.Gather(OnnxOp.Shape(x), Scalar(1L), axis: 0);
                var h0Shape = OnnxOp.Concat([OnnxOp.Unsqueeze(batchDim, Vector(0L)), Vector(H)], axis: 0);
                h0 = OnnxOp.Expand(OnnxOp.Cast(Scalar(0.0f), saturate: null, to: x.Type), h0Shape);
            }

            // Pre-transpose weight matrices for matmul: x @ W^T
            var WzT = OnnxOp.Transpose(Wz);  // [I, H]
            var WrT = OnnxOp.Transpose(Wr);
            var WhT = OnnxOp.Transpose(Wh);
            var RzT = OnnxOp.Transpose(Rz);  // [H, H]
            var RrT = OnnxOp.Transpose(Rr);
            var RhT = OnnxOp.Transpose(Rh);

            // ===== Forward pass: stash per-timestep intermediates into sequences =====
            // hPrev/z/r/htilde (and recH if linearBeforeReset) are needed by the backward
            // sweep; xt is re-Gathered from x there to avoid an extra sequence.
            var hPrevSeq = OnnxOp.SequenceEmpty(x.Type);
            var zSeq = OnnxOp.SequenceEmpty(x.Type);
            var rSeq = OnnxOp.SequenceEmpty(x.Type);
            var htSeq = OnnxOp.SequenceEmpty(x.Type);
            var recHSeq = linearBeforeReset ? OnnxOp.SequenceEmpty(x.Type) : null;
            var h = h0;

            foreach (var ctx in LoopAPI.Iterate(seqLen))
            {
                var xt = OnnxOp.Gather(x, ctx.IterationIndex, axis: 0);  // [B, I]
                var hPrev = h;

                var preZ = OnnxOp.Add(OnnxOp.MatMul(xt, WzT), OnnxOp.MatMul(hPrev, RzT));
                if (Wbz is not null) preZ = OnnxOp.Add(preZ, Wbz);
                if (Rbz is not null) preZ = OnnxOp.Add(preZ, Rbz);
                var zt = OnnxOp.Sigmoid(preZ);

                var preR = OnnxOp.Add(OnnxOp.MatMul(xt, WrT), OnnxOp.MatMul(hPrev, RrT));
                if (Wbr is not null) preR = OnnxOp.Add(preR, Wbr);
                if (Rbr is not null) preR = OnnxOp.Add(preR, Rbr);
                var rt = OnnxOp.Sigmoid(preR);

                Variable preH;
                Variable? recH = null;
                if (linearBeforeReset)
                {
                    recH = OnnxOp.MatMul(hPrev, RhT);
                    if (Rbh is not null) recH = OnnxOp.Add(recH, Rbh);
                    preH = OnnxOp.Add(OnnxOp.MatMul(xt, WhT), OnnxOp.Mul(rt, recH));
                }
                else
                {
                    preH = OnnxOp.Add(
                        OnnxOp.MatMul(xt, WhT),
                        OnnxOp.MatMul(OnnxOp.Mul(rt, hPrev), RhT));
                    if (Rbh is not null) preH = OnnxOp.Add(preH, Rbh);
                }
                if (Wbh is not null) preH = OnnxOp.Add(preH, Wbh);
                var ht = OnnxOp.Tanh(preH);

                hPrevSeq = OnnxOp.SequenceInsert(hPrevSeq, hPrev, null);
                zSeq = OnnxOp.SequenceInsert(zSeq, zt, null);
                rSeq = OnnxOp.SequenceInsert(rSeq, rt, null);
                htSeq = OnnxOp.SequenceInsert(htSeq, ht, null);
                if (linearBeforeReset)
                    recHSeq = OnnxOp.SequenceInsert(recHSeq!, recH!, null);

                // H_t = (1 - z_t) * ht_t + z_t * H_{t-1}
                h = OnnxOp.Add(
                    OnnxOp.Mul(OnnxOp.Sub(one, zt), ht),
                    OnnxOp.Mul(zt, hPrev));
            }

            // ===== Backward pass (BPTT) =====
            // Iterate forward i in [0, T) and read reversed timestep t = T-1-i.
            // Pre-loop seeds: dHNext absorbs dYh at t=T-1; weight/bias accumulators start
            // at zeros so each iteration's update reduces to a plain Add.
            var dHNext = dYh is not null
                ? OnnxOp.Squeeze(dYh, Vector(0L))   // [B, H]
                : OnnxOp.Sub(h0, h0);               // [B, H] zeros (matches h0 dtype/shape)
            var dWacc = OnnxOp.Sub(wSq, wSq);       // [3H, I] zeros
            var dRacc = OnnxOp.Sub(rSq, rSq);       // [3H, H] zeros
            Variable? dBacc = b is not null ? OnnxOp.Sub(bSq!, bSq!) : null;  // [6H] zeros or null
            // dXSeq is built by SequenceInsert at position 0 each iteration so that the
            // final order is forward time (iteration i contributes to t = T-1-i).
            var dXSeq = OnnxOp.SequenceEmpty(x.Type);

            foreach (var ctx in LoopAPI.Iterate(seqLen))
            {
                var tRev = OnnxOp.Sub(OnnxOp.Sub(seqLen, Scalar(1L)), ctx.IterationIndex);

                var xt = OnnxOp.Gather(x, tRev, axis: 0);       // [B, I]
                var hPrev = OnnxOp.SequenceAt(hPrevSeq, tRev);  // [B, H]
                var zt = OnnxOp.SequenceAt(zSeq, tRev);
                var rt = OnnxOp.SequenceAt(rSeq, tRev);
                var ht = OnnxOp.SequenceAt(htSeq, tRev);

                // dHt = dY[t] (if any) + dHNext (always defined; seeded with dYhSq or zeros)
                Variable dHt;
                if (dY is not null)
                {
                    var dYt = OnnxOp.Squeeze(OnnxOp.Gather(dY, tRev, axis: 0), Vector(0L));  // [B, H]
                    dHt = OnnxOp.Add(dYt, dHNext);
                }
                else
                {
                    dHt = dHNext;
                }

                // From H_t = (1-z_t)*ht_t + z_t*H_{t-1}
                var dHtilde = OnnxOp.Mul(dHt, OnnxOp.Sub(one, zt));
                var dZt = OnnxOp.Mul(dHt, OnnxOp.Sub(hPrev, ht));
                var dHPrevOut = OnnxOp.Mul(dHt, zt);

                // From ht_t = tanh(preH): d_tanh = 1 - tanh²
                var dPreH = OnnxOp.Mul(dHtilde, OnnxOp.Sub(one, OnnxOp.Mul(ht, ht)));

                // Gate gradients through ht computation
                Variable drFromH;
                Variable dHPrevFromH;
                if (linearBeforeReset)
                {
                    var recHCurrent = OnnxOp.SequenceAt(recHSeq!, tRev);
                    drFromH = OnnxOp.Mul(dPreH, recHCurrent);
                    var dRecH = OnnxOp.Mul(dPreH, rt);
                    dHPrevFromH = OnnxOp.MatMul(dRecH, Rh);
                }
                else
                {
                    var dRhInput = OnnxOp.MatMul(dPreH, Rh);
                    drFromH = OnnxOp.Mul(dRhInput, hPrev);
                    dHPrevFromH = OnnxOp.Mul(dRhInput, rt);
                }

                // Sigmoid gradients: σ'(x) = σ(x)(1-σ(x))
                var dPreZ = OnnxOp.Mul(dZt, OnnxOp.Mul(zt, OnnxOp.Sub(one, zt)));
                var dPreR = OnnxOp.Mul(drFromH, OnnxOp.Mul(rt, OnnxOp.Sub(one, rt)));

                // Input gradient: dX_t = dPreZ@Wz + dPreR@Wr + dPreH@Wh
                var dXt = OnnxOp.Add(
                    OnnxOp.Add(OnnxOp.MatMul(dPreZ, Wz), OnnxOp.MatMul(dPreR, Wr)),
                    OnnxOp.MatMul(dPreH, Wh));
                // Insert at position 0 so the sequence ends up in forward-time order.
                dXSeq = OnnxOp.SequenceInsert(dXSeq, dXt, Scalar(0L));

                // Hidden state gradient for previous timestep
                dHNext = OnnxOp.Add(
                    OnnxOp.Add(
                        OnnxOp.Add(OnnxOp.MatMul(dPreZ, Rz), OnnxOp.MatMul(dPreR, Rr)),
                        dHPrevFromH),
                    dHPrevOut);

                // Weight gradients: dW_t = [dPreZ^T@x_t; dPreR^T@x_t; dPreH^T@x_t]
                var dW_t = OnnxOp.Concat([
                    OnnxOp.MatMul(OnnxOp.Transpose(dPreZ), xt),
                    OnnxOp.MatMul(OnnxOp.Transpose(dPreR), xt),
                    OnnxOp.MatMul(OnnxOp.Transpose(dPreH), xt)
                ], axis: 0);  // [3H, I]
                dWacc = OnnxOp.Add(dWacc, dW_t);

                // Recurrent weight gradients
                Variable dRh_t;
                if (linearBeforeReset)
                {
                    var dRecH = OnnxOp.Mul(dPreH, rt);
                    dRh_t = OnnxOp.MatMul(OnnxOp.Transpose(dRecH), hPrev);
                }
                else
                {
                    dRh_t = OnnxOp.MatMul(OnnxOp.Transpose(dPreH), OnnxOp.Mul(rt, hPrev));
                }
                var dR_t = OnnxOp.Concat([
                    OnnxOp.MatMul(OnnxOp.Transpose(dPreZ), hPrev),
                    OnnxOp.MatMul(OnnxOp.Transpose(dPreR), hPrev),
                    dRh_t
                ], axis: 0);  // [3H, H]
                dRacc = OnnxOp.Add(dRacc, dR_t);

                // Bias gradients
                if (b is not null)
                {
                    var dPreZSum = OnnxOp.ReduceSum(dPreZ, Vector(0L), keepdims: false, noopWithEmptyAxes: null);
                    var dPreRSum = OnnxOp.ReduceSum(dPreR, Vector(0L), keepdims: false, noopWithEmptyAxes: null);
                    var dPreHSum = OnnxOp.ReduceSum(dPreH, Vector(0L), keepdims: false, noopWithEmptyAxes: null);

                    Variable dRbhSum;
                    if (linearBeforeReset)
                    {
                        var dRecH = OnnxOp.Mul(dPreH, rt);
                        dRbhSum = OnnxOp.ReduceSum(dRecH, Vector(0L), keepdims: false, noopWithEmptyAxes: null);
                    }
                    else
                    {
                        dRbhSum = dPreHSum;
                    }

                    // [Wbz, Wbr, Wbh, Rbz, Rbr, Rbh]
                    var dB_t = OnnxOp.Concat([
                        dPreZSum, dPreRSum, dPreHSum,
                        dPreZSum, dPreRSum, dRbhSum
                    ], axis: 0);  // [6H]
                    dBacc = OnnxOp.Add(dBacc!, dB_t);
                }
            }

            // ===== Assemble final outputs =====
            // dXSeq: T elements of [B, I] in forward-time order; stack along new axis 0 -> [T, B, I]
            var dX = OnnxOp.ConcatFromSequence(dXSeq, axis: 0, newAxis: true);
            // direction='reverse': dX was computed against the time-flipped x; flip back.
            if (isReverse) dX = ReverseTimeAxis(dX);

            // Add back num_directions dimension
            var dW = OnnxOp.Unsqueeze(dWacc, Vector(0L));   // [1, 3H, I]
            var dR = OnnxOp.Unsqueeze(dRacc, Vector(0L));   // [1, 3H, H]
            var dB = dBacc is not null ? OnnxOp.Unsqueeze(dBacc, Vector(0L)) : null;
            var dInitialH = initialH is not null ? OnnxOp.Unsqueeze(dHNext, Vector(0L)) : null;

            // Return: [dX, dW, dR, dB, dSequenceLens, dInitialH]
            return [dX, dW, dR, dB, null, dInitialH];
        }
    }
}
