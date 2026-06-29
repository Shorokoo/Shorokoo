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
        // ===== LSTM (Long Short-Term Memory) =====
        //
        // Forward pass (default activations f=sigmoid, g=tanh, h=tanh, forward direction, layout=0):
        //   For each timestep t = 0..T-1:
        //     i_t = f(X_t @ Wi^T + H_{t-1} @ Ri^T + Wbi + Rbi)
        //     f_t = f(X_t @ Wf^T + H_{t-1} @ Rf^T + Wbf + Rbf)
        //     c_t = g(X_t @ Wc^T + H_{t-1} @ Rc^T + Wbc + Rbc)
        //     o_t = f(X_t @ Wo^T + H_{t-1} @ Ro^T + Wbo + Rbo)
        //     C_t = f_t * C_{t-1} + i_t * c_t
        //     H_t = o_t * h(C_t)
        //
        // Gradient via Backpropagation Through Time (BPTT):
        //   Forward and backward sweeps run as graph-level Loop nodes via LoopAPI.Iterate
        //   so the sequence length is read at runtime.
        //
        // Supports: forward AND reverse direction (reverse reduces to the forward BPTT
        //           on a time-flipped x/dY with the resulting dX flipped back — see
        //           RnnGradient; dYh/dYc are scan-end quantities and map unchanged),
        //           default activations (sigmoid/tanh/tanh), layout=0,
        //           optional bias, initial hidden state, initial cell state.
        // Guarded (AD003): bidirectional, custom activations, clip, layout=1,
        //           wired sequence_lens, peephole weights, input_forget=1.

        internal static Variable?[] LstmGradient(Variable?[] inputs, Variable?[] outputGrads, OnnxCSharpAttributes attributes)
        {
            var x = inputs[0]!;          // [T, B, I]
            var w = inputs[1]!;          // [D, 4H, I]  (D = num_directions = 1)
            var r = inputs[2]!;          // [D, 4H, H]
            var b = inputs[3];           // [D, 8H] (optional)
            var sequenceLens = inputs[4]; // (not differentiable; unsupported when wired)
            var initialH = inputs[5];    // [D, B, H] (optional)
            var initialC = inputs[6];    // [D, B, H] (optional)
            var peephole = inputs[7];    // [D, 3H] (optional, not supported for gradient)

            var dY = outputGrads[0];     // [T, D, B, H] (optional)
            var dYh = outputGrads[1];    // [D, B, H] (optional)
            var dYc = outputGrads[2];    // [D, B, H] (optional)

            // At least one of dY/dYh/dYc is non-null: FastProcessAutoGrad/AutoDiffEngine
            // skip multi-output gradient nodes whose entire outputGrads array is null
            // before invoking the gradient method.
            Debug.Assert(dY is not null || dYh is not null || dYc is not null);

            var hiddenSize = (long)attributes.GetAttributeObj("hidden_size")!;

            var direction = ResolveRecurrentDirection(attributes.GetAttributeObj("direction"));
            GuardRecurrentAttributeEnvelope(LSTM, attributes, direction, sequenceLens,
                ["Sigmoid", "Tanh", "Tanh"]);

            // Peephole weights not supported for gradient: the i/f/o pre-activations gain
            // P⊙C terms whose backprop (incl. dP) is not modeled by the BPTT below.
            if (peephole is not null)
                throw new AutoDiffNotSupportedException(ErrorCodes.AD003, LSTM,
                    "the gradient does not support peephole weights (input P): the "
                    + "recomputed forward omits the P⊙C gate terms, which would produce a "
                    + "silently wrong gradient. This is an implementation limitation, not "
                    + "a mathematical one.");

            // input_forget=1 couples the input and forget gates (f = 1 - i), which the
            // BPTT below does not model.
            var inputForgetObj = attributes.GetAttributeObj("input_forget");
            if (inputForgetObj is true || (inputForgetObj is long ifLong && ifLong != 0))
                throw new AutoDiffNotSupportedException(ErrorCodes.AD003, LSTM,
                    "the gradient does not support input_forget=1 (coupled input/forget "
                    + "gates): the recomputed forward treats the gates as independent, "
                    + "which would produce a silently wrong gradient. This is an "
                    + "implementation limitation, not a mathematical one.");

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
            var wSq = OnnxOp.Squeeze(w, Vector(0L));  // [4H, I]
            var rSq = OnnxOp.Squeeze(r, Vector(0L));  // [4H, H]

            // Split input weights: W = [Wi; Wo; Wf; Wc], each [H, I] (ONNX LSTM gate order: i, o, f, c)
            var splitH4 = Vector(H, H, H, H);
            var wParts = OnnxOp.Split(wSq, splitH4, axis: 0, numOutputs: null, variadicOutputCount: 4);
            var Wi = wParts[0];
            var Wo = wParts[1];
            var Wf = wParts[2];
            var Wc = wParts[3];

            // Split recurrent weights: R = [Ri; Ro; Rf; Rc], each [H, H]
            var rParts = OnnxOp.Split(rSq, splitH4, axis: 0, numOutputs: null, variadicOutputCount: 4);
            var Ri = rParts[0];
            var Ro = rParts[1];
            var Rf = rParts[2];
            var Rc = rParts[3];

            // Split biases if present: B = [Wbi, Wbo, Wbf, Wbc, Rbi, Rbo, Rbf, Rbc], each [H]
            Variable? Wbi = null, Wbo = null, Wbf = null, Wbc = null;
            Variable? Rbi = null, Rbo = null, Rbf = null, Rbc = null;
            Variable? bSq = null;
            if (b is not null)
            {
                bSq = OnnxOp.Squeeze(b, Vector(0L));  // [8H]
                var bParts = OnnxOp.Split(bSq, Vector(H, H, H, H, H, H, H, H), axis: 0,
                    numOutputs: null, variadicOutputCount: 8);
                Wbi = bParts[0]; Wbo = bParts[1]; Wbf = bParts[2]; Wbc = bParts[3];
                Rbi = bParts[4]; Rbo = bParts[5]; Rbf = bParts[6]; Rbc = bParts[7];
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

            // Initial cell state: [B, H]
            Variable c0;
            if (initialC is not null)
            {
                c0 = OnnxOp.Squeeze(initialC, Vector(0L));  // [B, H]
            }
            else
            {
                var batchDim = OnnxOp.Gather(OnnxOp.Shape(x), Scalar(1L), axis: 0);
                var c0Shape = OnnxOp.Concat([OnnxOp.Unsqueeze(batchDim, Vector(0L)), Vector(H)], axis: 0);
                c0 = OnnxOp.Expand(OnnxOp.Cast(Scalar(0.0f), saturate: null, to: x.Type), c0Shape);
            }

            // Pre-transpose weight matrices for matmul: x @ W^T
            var WiT = OnnxOp.Transpose(Wi);  // [I, H]
            var WoT = OnnxOp.Transpose(Wo);
            var WfT = OnnxOp.Transpose(Wf);
            var WcT = OnnxOp.Transpose(Wc);
            var RiT = OnnxOp.Transpose(Ri);  // [H, H]
            var RoT = OnnxOp.Transpose(Ro);
            var RfT = OnnxOp.Transpose(Rf);
            var RcT = OnnxOp.Transpose(Rc);

            // ===== Forward pass: stash per-timestep intermediates into sequences =====
            // hPrev/cPrev (= H_{t-1}, C_{t-1}), the four gate activations, and tanh(C_t)
            // are needed by the backward sweep; xt is re-Gathered from x there.
            var hPrevSeq = OnnxOp.SequenceEmpty(x.Type);
            var cPrevSeq = OnnxOp.SequenceEmpty(x.Type);
            var iSeq = OnnxOp.SequenceEmpty(x.Type);
            var fSeq = OnnxOp.SequenceEmpty(x.Type);
            var oSeq = OnnxOp.SequenceEmpty(x.Type);
            var cTildeSeq = OnnxOp.SequenceEmpty(x.Type);
            var tCtSeq = OnnxOp.SequenceEmpty(x.Type);
            var hLoop = h0;
            var cLoop = c0;

            foreach (var ctx in LoopAPI.Iterate(seqLen))
            {
                var xt = OnnxOp.Gather(x, ctx.IterationIndex, axis: 0);  // [B, I]
                var hPrev = hLoop;
                var cPrev = cLoop;

                var preI = OnnxOp.Add(OnnxOp.MatMul(xt, WiT), OnnxOp.MatMul(hPrev, RiT));
                if (Wbi is not null) preI = OnnxOp.Add(preI, Wbi);
                if (Rbi is not null) preI = OnnxOp.Add(preI, Rbi);
                var it = OnnxOp.Sigmoid(preI);

                var preF = OnnxOp.Add(OnnxOp.MatMul(xt, WfT), OnnxOp.MatMul(hPrev, RfT));
                if (Wbf is not null) preF = OnnxOp.Add(preF, Wbf);
                if (Rbf is not null) preF = OnnxOp.Add(preF, Rbf);
                var ft = OnnxOp.Sigmoid(preF);

                var preC = OnnxOp.Add(OnnxOp.MatMul(xt, WcT), OnnxOp.MatMul(hPrev, RcT));
                if (Wbc is not null) preC = OnnxOp.Add(preC, Wbc);
                if (Rbc is not null) preC = OnnxOp.Add(preC, Rbc);
                var ct = OnnxOp.Tanh(preC);

                var preO = OnnxOp.Add(OnnxOp.MatMul(xt, WoT), OnnxOp.MatMul(hPrev, RoT));
                if (Wbo is not null) preO = OnnxOp.Add(preO, Wbo);
                if (Rbo is not null) preO = OnnxOp.Add(preO, Rbo);
                var ot = OnnxOp.Sigmoid(preO);

                // C_t = f_t * C_{t-1} + i_t * c_t
                var cNew = OnnxOp.Add(OnnxOp.Mul(ft, cPrev), OnnxOp.Mul(it, ct));
                // H_t = o_t * tanh(C_t)
                var tCt = OnnxOp.Tanh(cNew);
                var hNew = OnnxOp.Mul(ot, tCt);

                hPrevSeq = OnnxOp.SequenceInsert(hPrevSeq, hPrev, null);
                cPrevSeq = OnnxOp.SequenceInsert(cPrevSeq, cPrev, null);
                iSeq = OnnxOp.SequenceInsert(iSeq, it, null);
                fSeq = OnnxOp.SequenceInsert(fSeq, ft, null);
                oSeq = OnnxOp.SequenceInsert(oSeq, ot, null);
                cTildeSeq = OnnxOp.SequenceInsert(cTildeSeq, ct, null);
                tCtSeq = OnnxOp.SequenceInsert(tCtSeq, tCt, null);

                hLoop = hNew;
                cLoop = cNew;
            }

            // ===== Backward pass (BPTT) =====
            // Pre-loop seeds: dHNext absorbs dYh at t=T-1, dCNext absorbs dYc at t=T-1,
            // accumulators start at zeros so each iteration's update is a plain Add.
            var dHNext = dYh is not null
                ? OnnxOp.Squeeze(dYh, Vector(0L))   // [B, H]
                : OnnxOp.Sub(h0, h0);               // [B, H] zeros
            var dCNext = dYc is not null
                ? OnnxOp.Squeeze(dYc, Vector(0L))   // [B, H]
                : OnnxOp.Sub(c0, c0);               // [B, H] zeros
            var dWacc = OnnxOp.Sub(wSq, wSq);       // [4H, I] zeros
            var dRacc = OnnxOp.Sub(rSq, rSq);       // [4H, H] zeros
            Variable? dBacc = b is not null ? OnnxOp.Sub(bSq!, bSq!) : null;  // [8H] zeros or null
            var dXSeq = OnnxOp.SequenceEmpty(x.Type);

            foreach (var ctx in LoopAPI.Iterate(seqLen))
            {
                var tRev = OnnxOp.Sub(OnnxOp.Sub(seqLen, Scalar(1L)), ctx.IterationIndex);

                var xt = OnnxOp.Gather(x, tRev, axis: 0);            // [B, I]
                var hPrev = OnnxOp.SequenceAt(hPrevSeq, tRev);
                var cPrev = OnnxOp.SequenceAt(cPrevSeq, tRev);
                var it = OnnxOp.SequenceAt(iSeq, tRev);
                var ft = OnnxOp.SequenceAt(fSeq, tRev);
                var ot = OnnxOp.SequenceAt(oSeq, tRev);
                var ct = OnnxOp.SequenceAt(cTildeSeq, tRev);
                var tCt = OnnxOp.SequenceAt(tCtSeq, tRev);

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

                // From H_t = o_t * tanh(C_t)
                var dOt = OnnxOp.Mul(dHt, tCt);
                var dTanhCt = OnnxOp.Mul(dHt, ot);

                // tanh derivative: dC_t' = dTanhCt * (1 - tanh(C_t)^2)
                // dCt totals: tanh-derived term + dCNext (which already absorbed dYc at t=T-1)
                var dCtFromH = OnnxOp.Mul(dTanhCt, OnnxOp.Sub(one, OnnxOp.Mul(tCt, tCt)));
                var dCt = OnnxOp.Add(dCtFromH, dCNext);

                // From C_t = f_t * C_{t-1} + i_t * c_t
                var dFt = OnnxOp.Mul(dCt, cPrev);
                var dIt = OnnxOp.Mul(dCt, ct);
                var dCtilde = OnnxOp.Mul(dCt, it);
                dCNext = OnnxOp.Mul(dCt, ft);  // gradient flowing to C_{t-1}

                // Sigmoid gradients: σ'(x) = σ(x)(1-σ(x))
                var dPreI = OnnxOp.Mul(dIt, OnnxOp.Mul(it, OnnxOp.Sub(one, it)));
                var dPreF = OnnxOp.Mul(dFt, OnnxOp.Mul(ft, OnnxOp.Sub(one, ft)));
                var dPreO = OnnxOp.Mul(dOt, OnnxOp.Mul(ot, OnnxOp.Sub(one, ot)));
                // tanh'(x) = 1 - tanh(x)^2
                var dPreC = OnnxOp.Mul(dCtilde, OnnxOp.Sub(one, OnnxOp.Mul(ct, ct)));

                // Input gradient: dX_t = dPreI@Wi + dPreF@Wf + dPreC@Wc + dPreO@Wo
                var dXt = OnnxOp.Add(
                    OnnxOp.Add(
                        OnnxOp.Add(OnnxOp.MatMul(dPreI, Wi), OnnxOp.MatMul(dPreF, Wf)),
                        OnnxOp.MatMul(dPreC, Wc)),
                    OnnxOp.MatMul(dPreO, Wo));
                dXSeq = OnnxOp.SequenceInsert(dXSeq, dXt, Scalar(0L));

                // Hidden state gradient for previous timestep
                dHNext = OnnxOp.Add(
                    OnnxOp.Add(
                        OnnxOp.Add(OnnxOp.MatMul(dPreI, Ri), OnnxOp.MatMul(dPreF, Rf)),
                        OnnxOp.MatMul(dPreC, Rc)),
                    OnnxOp.MatMul(dPreO, Ro));

                // Weight gradients: dW_t = [dPreI^T@x_t; dPreO^T@x_t; dPreF^T@x_t; dPreC^T@x_t]
                var dWi_t = OnnxOp.MatMul(OnnxOp.Transpose(dPreI), xt);
                var dWo_t = OnnxOp.MatMul(OnnxOp.Transpose(dPreO), xt);
                var dWf_t = OnnxOp.MatMul(OnnxOp.Transpose(dPreF), xt);
                var dWc_t = OnnxOp.MatMul(OnnxOp.Transpose(dPreC), xt);
                var dW_t = OnnxOp.Concat([dWi_t, dWo_t, dWf_t, dWc_t], axis: 0);  // [4H, I]
                dWacc = OnnxOp.Add(dWacc, dW_t);

                // Recurrent weight gradients
                var dRi_t = OnnxOp.MatMul(OnnxOp.Transpose(dPreI), hPrev);
                var dRo_t = OnnxOp.MatMul(OnnxOp.Transpose(dPreO), hPrev);
                var dRf_t = OnnxOp.MatMul(OnnxOp.Transpose(dPreF), hPrev);
                var dRc_t = OnnxOp.MatMul(OnnxOp.Transpose(dPreC), hPrev);
                var dR_t = OnnxOp.Concat([dRi_t, dRo_t, dRf_t, dRc_t], axis: 0);  // [4H, H]
                dRacc = OnnxOp.Add(dRacc, dR_t);

                // Bias gradients
                if (b is not null)
                {
                    var dPreISum = OnnxOp.ReduceSum(dPreI, Vector(0L), keepdims: false, noopWithEmptyAxes: null);
                    var dPreOSum = OnnxOp.ReduceSum(dPreO, Vector(0L), keepdims: false, noopWithEmptyAxes: null);
                    var dPreFSum = OnnxOp.ReduceSum(dPreF, Vector(0L), keepdims: false, noopWithEmptyAxes: null);
                    var dPreCSum = OnnxOp.ReduceSum(dPreC, Vector(0L), keepdims: false, noopWithEmptyAxes: null);

                    // [Wbi, Wbo, Wbf, Wbc, Rbi, Rbo, Rbf, Rbc]
                    var dB_t = OnnxOp.Concat([
                        dPreISum, dPreOSum, dPreFSum, dPreCSum,
                        dPreISum, dPreOSum, dPreFSum, dPreCSum
                    ], axis: 0);  // [8H]
                    dBacc = OnnxOp.Add(dBacc!, dB_t);
                }
            }

            // ===== Assemble final outputs =====
            // dXSeq: T elements of [B, I] in forward-time order; stack along new axis 0 -> [T, B, I]
            var dX = OnnxOp.ConcatFromSequence(dXSeq, axis: 0, newAxis: true);
            // direction='reverse': dX was computed against the time-flipped x; flip back.
            if (isReverse) dX = ReverseTimeAxis(dX);

            // Add back num_directions dimension
            var dW = OnnxOp.Unsqueeze(dWacc, Vector(0L));   // [1, 4H, I]
            var dR = OnnxOp.Unsqueeze(dRacc, Vector(0L));   // [1, 4H, H]
            var dB = dBacc is not null ? OnnxOp.Unsqueeze(dBacc, Vector(0L)) : null;
            var dInitialH = initialH is not null ? OnnxOp.Unsqueeze(dHNext, Vector(0L)) : null;
            var dInitialC = initialC is not null ? OnnxOp.Unsqueeze(dCNext, Vector(0L)) : null;

            // Return: [dX, dW, dR, dB, dSequenceLens, dInitialH, dInitialC, dP]
            return [dX, dW, dR, dB, null, dInitialH, dInitialC, null];
        }
    }
}
