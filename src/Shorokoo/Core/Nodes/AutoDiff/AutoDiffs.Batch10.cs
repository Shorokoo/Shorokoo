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
        // ------------------------------------------------------------------------------
        // Shared attribute-envelope helpers for the recurrent (RNN / GRU / LSTM)
        // gradients. The BPTT implementations assume the DEFAULT activations, no
        // clipping, layout=0 and no per-batch sequence_lens; any other combination
        // would make the recomputed forward (and therefore the gradient) silently wrong,
        // so they throw AD003 instead (AD-B2/AD-B3 pattern: no silent wrongness).
        // ------------------------------------------------------------------------------

        /// <summary>
        /// Maps the recurrent <c>direction</c> attribute object (typed enum, string, or
        /// null) onto "forward" / "reverse" / "bidirectional".
        /// </summary>
        private static string ResolveRecurrentDirection(object? dirObj) => dirObj switch
        {
            null => "forward",
            RNNDirection d => d.ToString().ToLowerInvariant(),
            GRUDirection d => d.ToString().ToLowerInvariant(),
            LSTMDirection d => d.ToString().ToLowerInvariant(),
            string s => s.ToLowerInvariant(),
            _ => dirObj.ToString()!.ToLowerInvariant(),
        };

        /// <summary>
        /// Throws AD003 for attribute combinations the recurrent BPTT gradients do not
        /// model: bidirectional scans, non-default activations, the clip attribute,
        /// layout=1, and a wired per-batch sequence_lens input.
        /// <paramref name="defaultActivations"/> lists the op's spec defaults for ONE
        /// direction (e.g. ["Sigmoid", "Tanh"] for GRU).
        /// </summary>
        private static void GuardRecurrentAttributeEnvelope(
            string opCode, OnnxCSharpAttributes attributes, string direction,
            Variable? sequenceLens, string[] defaultActivations)
        {
            if (direction == "bidirectional")
                throw new AutoDiffNotSupportedException(ErrorCodes.AD003, opCode,
                    "the gradient is only implemented for direction='forward'/'reverse': "
                    + "bidirectional BPTT (two weight slabs along num_directions) is not "
                    + "implemented. This is an implementation limitation, not a mathematical "
                    + "one — split the layer into a forward and a reverse pass and Concat "
                    + "the outputs.");

            var layoutObj = attributes.GetAttributeObj("layout");
            if (layoutObj is true || (layoutObj is long layoutLong && layoutLong != 0))
                throw new AutoDiffNotSupportedException(ErrorCodes.AD003, opCode,
                    "the gradient is only implemented for layout=0 ([seq, batch, ...]); "
                    + "layout=1 (batch-major) is not implemented. This is an implementation "
                    + "limitation, not a mathematical one — Transpose around the op instead.");

            if (attributes.GetAttributeObj("clip") is float)
                throw new AutoDiffNotSupportedException(ErrorCodes.AD003, opCode,
                    "the gradient does not model the `clip` attribute: the recomputed "
                    + "forward would ignore the cell clipping and produce a silently wrong "
                    + "gradient. This is an implementation limitation, not a mathematical "
                    + "one.");

            if (attributes.GetAttributeObj("activations") is string[] { Length: > 0 } acts)
            {
                var matchesDefault = acts.Length == defaultActivations.Length;
                for (int i = 0; matchesDefault && i < acts.Length; i++)
                    matchesDefault = string.Equals(acts[i], defaultActivations[i],
                        StringComparison.OrdinalIgnoreCase);
                if (!matchesDefault)
                    throw new AutoDiffNotSupportedException(ErrorCodes.AD003, opCode,
                        "the gradient is only implemented for the default activations ("
                        + string.Join("/", defaultActivations) + "); got ["
                        + string.Join(", ", acts) + "]. Custom activation backprop is not "
                        + "implemented. This is an implementation limitation, not a "
                        + "mathematical one.");
            }

            if (sequenceLens is not null)
                throw new AutoDiffNotSupportedException(ErrorCodes.AD003, opCode,
                    "the gradient does not model a wired `sequence_lens` input: per-batch "
                    + "sequence masking is ignored by the recomputed forward, which would "
                    + "produce a silently wrong gradient for any batch entry shorter than "
                    + "the padded length. This is an implementation limitation, not a "
                    + "mathematical one — drop the input when every sequence has the full "
                    + "length.");
        }

        /// <summary>
        /// Reverses a tensor along axis 0 (the time axis for layout=0 recurrent tensors)
        /// with a runtime-length Gather: indices = T-1 .. 0. Used to reduce the
        /// direction='reverse' gradients to the forward-direction BPTT (a reverse scan
        /// over x is a forward scan over time-flipped x).
        /// </summary>
        private static Variable ReverseTimeAxis(Variable t)
        {
            var len = OnnxOp.Gather(OnnxOp.Shape(t), Scalar(0L), axis: 0);
            var indices = OnnxOp.Sub(
                OnnxOp.Sub(len, Scalar(1L)),
                OnnxOp.Range(Scalar(0L), len, Scalar(1L)));
            return OnnxOp.Gather(t, indices, axis: 0);
        }

        // ===== RNN (Simple/Elman Recurrent Neural Network) =====
        //
        // Forward pass (default activation f=tanh, forward direction, layout=0):
        //   For each timestep t = 0..T-1:
        //     H_t = f(X_t @ W^T + H_{t-1} @ R^T + Wb + Rb)
        //
        // Gradient via Backpropagation Through Time (BPTT):
        //   Forward and backward sweeps run as graph-level Loop nodes via LoopAPI.Iterate
        //   so the sequence length is read at runtime.
        //
        // Supports: forward AND reverse direction (reverse reduces to the forward BPTT
        //           on a time-flipped x/dY with the resulting dX flipped back: the
        //           reverse scan's Y[t] equals the forward-on-flipped-x scan's Y[T-1-t],
        //           and Y_h is the scan-end state in both), default activation (tanh),
        //           layout=0, optional bias and initial hidden state.
        // Guarded (AD003): bidirectional, custom activations, clip, layout=1,
        //           wired sequence_lens.

        internal static Variable?[] RnnGradient(Variable?[] inputs, Variable?[] outputGrads, OnnxCSharpAttributes attributes)
        {
            var x = inputs[0]!;          // [T, B, I]
            var w = inputs[1]!;          // [D, H, I]  (D = num_directions = 1)
            var r = inputs[2]!;          // [D, H, H]
            var b = inputs[3];           // [D, 2H] (optional)
            var sequenceLens = inputs[4]; // (not differentiable; unsupported when wired)
            var initialH = inputs[5];    // [D, B, H] (optional)

            var dY = outputGrads[0];     // [T, D, B, H] (optional)
            var dYh = outputGrads[1];    // [D, B, H] (optional)

            // At least one of dY/dYh is non-null: FastProcessAutoGrad/AutoDiffEngine
            // skip multi-output gradient nodes whose entire outputGrads array is null
            // before invoking the gradient method.
            Debug.Assert(dY is not null || dYh is not null);

            var hiddenSize = (long)attributes.GetAttributeObj("hidden_size")!;

            var direction = ResolveRecurrentDirection(attributes.GetAttributeObj("direction"));
            GuardRecurrentAttributeEnvelope(RNN, attributes, direction, sequenceLens, ["Tanh"]);

            // direction='reverse': flip x and dY along time, run the forward-direction
            // BPTT, and flip the resulting dX back. dYh / dW / dR / dB / dInitialH are
            // time-sum or scan-end quantities and map through unchanged.
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
            var wSq = OnnxOp.Squeeze(w, Vector(0L));  // [H, I]
            var rSq = OnnxOp.Squeeze(r, Vector(0L));  // [H, H]

            // Split biases if present: B = [Wb, Rb], each [H]
            Variable? Wb = null, Rb = null;
            Variable? bSq = null;
            if (b is not null)
            {
                bSq = OnnxOp.Squeeze(b, Vector(0L));  // [2H]
                var bParts = OnnxOp.Split(bSq, Vector(H, H), axis: 0,
                    numOutputs: null, variadicOutputCount: 2);
                Wb = bParts[0]; Rb = bParts[1];
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
            var WT = OnnxOp.Transpose(wSq);  // [I, H]
            var RT = OnnxOp.Transpose(rSq);  // [H, H]

            // ===== Forward pass: stash per-timestep hidden states into sequences =====
            var hPrevSeq = OnnxOp.SequenceEmpty(x.Type);
            var htSeq = OnnxOp.SequenceEmpty(x.Type);
            var h = h0;

            foreach (var ctx in LoopAPI.Iterate(seqLen))
            {
                var xt = OnnxOp.Gather(x, ctx.IterationIndex, axis: 0);  // [B, I]
                var hPrev = h;

                // H_t = tanh(X_t @ W^T + H_{t-1} @ R^T + Wb + Rb)
                var preH = OnnxOp.Add(OnnxOp.MatMul(xt, WT), OnnxOp.MatMul(hPrev, RT));
                if (Wb is not null) preH = OnnxOp.Add(preH, Wb);
                if (Rb is not null) preH = OnnxOp.Add(preH, Rb);
                var ht = OnnxOp.Tanh(preH);

                hPrevSeq = OnnxOp.SequenceInsert(hPrevSeq, hPrev, null);
                htSeq = OnnxOp.SequenceInsert(htSeq, ht, null);

                h = ht;
            }

            // ===== Backward pass (BPTT) =====
            // dHNext absorbs dYh at t=T-1 and is updated each iteration; weight/bias
            // accumulators start at zeros so each iteration's update is a plain Add.
            var dHNext = dYh is not null
                ? OnnxOp.Squeeze(dYh, Vector(0L))   // [B, H]
                : OnnxOp.Sub(h0, h0);               // [B, H] zeros
            var dWacc = OnnxOp.Sub(wSq, wSq);       // [H, I] zeros
            var dRacc = OnnxOp.Sub(rSq, rSq);       // [H, H] zeros
            Variable? dBacc = b is not null ? OnnxOp.Sub(bSq!, bSq!) : null;  // [2H] zeros or null
            // dXSeq is built by SequenceInsert at position 0 each iteration so the final
            // order is forward time (iteration i contributes to t = T-1-i).
            var dXSeq = OnnxOp.SequenceEmpty(x.Type);

            foreach (var ctx in LoopAPI.Iterate(seqLen))
            {
                var tRev = OnnxOp.Sub(OnnxOp.Sub(seqLen, Scalar(1L)), ctx.IterationIndex);

                var xt = OnnxOp.Gather(x, tRev, axis: 0);            // [B, I]
                var hPrev = OnnxOp.SequenceAt(hPrevSeq, tRev);       // [B, H]
                var ht = OnnxOp.SequenceAt(htSeq, tRev);             // [B, H]

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

                // From H_t = tanh(preH): d_tanh = 1 - tanh^2
                var dPreH = OnnxOp.Mul(dHt, OnnxOp.Sub(one, OnnxOp.Mul(ht, ht)));

                // Input gradient: dX_t = dPreH @ W
                var dXt = OnnxOp.MatMul(dPreH, wSq);
                dXSeq = OnnxOp.SequenceInsert(dXSeq, dXt, Scalar(0L));

                // Hidden state gradient for previous timestep: dH_{t-1} = dPreH @ R
                dHNext = OnnxOp.MatMul(dPreH, rSq);

                // Weight gradient: dW_t = dPreH^T @ X_t
                var dW_t = OnnxOp.MatMul(OnnxOp.Transpose(dPreH), xt);  // [H, I]
                dWacc = OnnxOp.Add(dWacc, dW_t);

                // Recurrent weight gradient: dR_t = dPreH^T @ H_{t-1}
                var dR_t = OnnxOp.MatMul(OnnxOp.Transpose(dPreH), hPrev);  // [H, H]
                dRacc = OnnxOp.Add(dRacc, dR_t);

                // Bias gradients: both Wb and Rb get the same gradient
                if (b is not null)
                {
                    var dPreHSum = OnnxOp.ReduceSum(dPreH, Vector(0L), keepdims: false, noopWithEmptyAxes: null);
                    var dB_t = OnnxOp.Concat([dPreHSum, dPreHSum], axis: 0);  // [2H]
                    dBacc = OnnxOp.Add(dBacc!, dB_t);
                }
            }

            // ===== Assemble final outputs =====
            // dXSeq: T elements of [B, I] in forward-time order; stack along new axis 0 -> [T, B, I]
            var dX = OnnxOp.ConcatFromSequence(dXSeq, axis: 0, newAxis: true);
            // direction='reverse': dX was computed against the time-flipped x; flip back.
            if (isReverse) dX = ReverseTimeAxis(dX);

            // Add back num_directions dimension
            var dW = OnnxOp.Unsqueeze(dWacc, Vector(0L));   // [1, H, I]
            var dR = OnnxOp.Unsqueeze(dRacc, Vector(0L));   // [1, H, H]
            var dB = dBacc is not null ? OnnxOp.Unsqueeze(dBacc, Vector(0L)) : null;
            var dInitialH = initialH is not null ? OnnxOp.Unsqueeze(dHNext, Vector(0L)) : null;

            // Return: [dX, dW, dR, dB, dSequenceLens, dInitialH]
            return [dX, dW, dR, dB, null, dInitialH];
        }
    }
}
