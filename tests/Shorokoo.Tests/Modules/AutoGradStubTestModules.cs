namespace Shorokoo.Tests.Modules
{
    // ===================================================================
    //  Self-checking modules for the non-differentiable AutoDiff stubs in
    //  AutoDiffs.Batch28.cs. Each module routes a runtime input through a
    //  stub op such that its analytical gradient is zero (either because
    //  the op's output is locally constant, because the op outputs a
    //  non-differentiable dtype, or because the op's inputs don't appear
    //  in its mathematical signature). The module then asks AutoGrad for
    //  the input's gradient and verifies it's zero — exercising each
    //  Batch28 registration via the FastProcessAutoGrad pipeline.
    // ===================================================================

    // Round / Hardmax / OneHot / IsInf / IsNaN / Multinomial / Size are wired into the
    // AutoDiff system only as Batch28 [AutoDiff] registrations — they have no OnnxOp
    // factory or Definitions schema entry, so they're constructed exclusively by
    // ONNX import in production. Their stub registrations are still useful (the
    // gradient engine no longer crashes when an imported graph hits one of these),
    // but they're not directly exercisable through a Module-driven coverage test.

    /// <summary>Bitwise integer ops: not differentiable — gradient is zero at the integer input.
    /// Operates on rank-1 tensors so the BitwiseAnd/Or/Xor/Not results stay as Tensor (the
    /// OnnxOp factories return rank-unspecified IValue, so Scalar-target casts fail).</summary>
    [Module]
    public partial class AutoGradBitwiseStubCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var asInt = x.Cast<int64>();
            var via = (Tensor<int64>)OnnxOp.BitwiseAnd(asInt, Vector(0xFL));
            var via2 = (Tensor<int64>)OnnxOp.BitwiseOr(via, Vector(0L));
            var via3 = (Tensor<int64>)OnnxOp.BitwiseXor(via2, Vector(0L));
            var via4 = (Tensor<int64>)OnnxOp.BitwiseNot(via3);
            var via5 = (Tensor<int64>)OnnxOp.BitwiseNot(via4);
            var loss = via5.Cast<float32>().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            var maxAbs = grad.Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
            return maxAbs < Scalar(1e-4f);
        }
    }

    /// <summary>BitShift on unsigned int: not differentiable.</summary>
    [Module]
    public partial class AutoGradBitShiftStubCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var asInt = x.Cast<uint64>();
            var shifted = (Tensor<uint64>)OnnxOp.BitShift(asInt, Vector((ulong)1), direction: BitShiftDirection.Left);
            var loss = shifted.Cast<float32>().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            var maxAbs = grad.Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
            return maxAbs < Scalar(1e-4f);
        }
    }

    /// <summary>Boolean ops (AND/OR/XOR/NOT): bit-typed inputs and outputs. The Inline graph
    /// uses these ops on a side branch whose result is multiplied into the loss by a constant
    /// zero, so the analytical gradient of <c>x</c> through this path is exactly zero —
    /// exercising the And/Or/Xor/Not [AutoDiff] stubs without inviting type-pinning issues
    /// from gradients flowing back through Cast(bit → float32).</summary>
    [Module]
    public partial class AutoGradBooleanStubCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var asBit = x > Scalar(0f);
            var bAnd = (Tensor<bit>)OnnxOp.And(asBit, asBit);
            var bOr = (Tensor<bit>)OnnxOp.Or(bAnd, asBit);
            var bXor = (Tensor<bit>)OnnxOp.Xor(bOr, (Tensor<bit>)OnnxOp.Expand(Scalar(false), x.DShape));
            var bNot = (Tensor<bit>)OnnxOp.Not(bXor);
            // Mask the bool side-branch out of the loss via Cast→multiply-by-zero so the
            // mathematical gradient of x is zero (we're only exercising the Batch28 stubs).
            var maskFloat = bNot.Cast<float32>() * Scalar(0f);
            var loss = (x + maskFloat).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            // ∂L/∂x = 1 (from the unmasked path) — bool stubs contribute zero.
            var diff = (grad - ((Tensor<float32>)OnnxOp.Expand(Scalar(1f), x.DShape))).Abs();
            return diff.Reduce(ReduceKind.Max, keepDims: false).Scalar() < Scalar(1e-4f);
        }
    }

    /// <summary>Comparison ops (Equal/Greater/Less etc.): bool output. Mask-out pattern as in
    /// <see cref="AutoGradBooleanStubCheck"/>: comparisons feed into the loss only through a
    /// multiply-by-zero so x's analytical gradient stays well-defined.</summary>
    [Module]
    public partial class AutoGradComparisonStubCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x, Tensor<float32> y)
        {
            var eq = (Tensor<bit>)OnnxOp.Equal(x, y);
            var gt = (Tensor<bit>)OnnxOp.Greater(x, y);
            var ge = (Tensor<bit>)OnnxOp.GreaterOrEqual(x, y);
            var lt = (Tensor<bit>)OnnxOp.Less(x, y);
            var le = (Tensor<bit>)OnnxOp.LessOrEqual(x, y);
            var combined = (Tensor<bit>)OnnxOp.Or(eq,
                (Tensor<bit>)OnnxOp.Or(gt,
                    (Tensor<bit>)OnnxOp.Or(ge,
                        (Tensor<bit>)OnnxOp.Or(lt, le))));
            var maskFloat = combined.Cast<float32>() * Scalar(0f);
            var loss = (x + y + maskFloat).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var (gx, gy) = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, y, loss);
            var diffX = ((Tensor<float32>)gx! - ((Tensor<float32>)OnnxOp.Expand(Scalar(1f), x.DShape))).Abs();
            var diffY = ((Tensor<float32>)gy! - ((Tensor<float32>)OnnxOp.Expand(Scalar(1f), y.DShape))).Abs();
            var maxAbs = OnnxOp.Max(
                diffX.Reduce(ReduceKind.Max, keepDims: false).Scalar(),
                diffY.Reduce(ReduceKind.Max, keepDims: false).Scalar());
            return ((Scalar<float32>)maxAbs) < Scalar(1e-4f);
        }
    }

    /// <summary>EyeLike: output is purely structural — gradient at the template input is zero.</summary>
    [Module]
    public partial class AutoGradEyeLikeStubCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var eye = (Tensor<float32>)OnnxOp.EyeLike(x, dtype: null, k: 0);
            var loss = eye.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            var maxAbs = grad.Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
            return maxAbs < Scalar(1e-4f);
        }
    }

    /// <summary>Shape: int64 output that doesn't depend on the input values — gradient null.</summary>
    [Module]
    public partial class AutoGradShapeStubCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var shape = (Tensor<int64>)OnnxOp.Shape(x, end: null, start: null);
            var loss = shape.Cast<float32>().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            var maxAbs = grad.Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
            return maxAbs < Scalar(1e-4f);
        }
    }

    /// <summary>ArgMax + ArgMin: integer index outputs — gradient null.</summary>
    [Module]
    public partial class AutoGradArgMaxArgMinStubCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var amax = (Tensor<int64>)OnnxOp.ArgMax(x, axis: 0, keepdims: false, selectLastIndex: false);
            var amin = (Tensor<int64>)OnnxOp.ArgMin(x, axis: 0, keepdims: false, selectLastIndex: false);
            var loss = (amax + amin).Cast<float32>().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            var maxAbs = grad.Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
            return maxAbs < Scalar(1e-4f);
        }
    }

    /// <summary>NonZero: integer index output — diff input can't influence the result values.</summary>
    [Module]
    public partial class AutoGradNonZeroStubCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var nz = (Tensor<int64>)OnnxOp.NonZero(x);
            var loss = nz.Cast<float32>().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            var maxAbs = grad.Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
            return maxAbs < Scalar(1e-4f);
        }
    }

    /// <summary>Range: integer index range — gradient null at the (non-diff) bounds.</summary>
    [Module]
    public partial class AutoGradRangeStubCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            var rng = (Vector<int64>)OnnxOp.Range(x.Cast<int64>(), Scalar(5L), Scalar(1L));
            var loss = rng.Cast<float32>().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            return grad.Abs() < Scalar(1e-4f);
        }
    }

    /// <summary>Bernoulli: stochastic sample — autograd stub returns null, runtime input
    /// falls back to the SpliceZerosLike path so the reported gradient is zero.</summary>
    [Module]
    public partial class AutoGradBernoulliStubCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var bern = (Tensor<float32>)OnnxOp.Bernoulli(x, dtype: null, seed: 1f);
            var loss = bern.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            var maxAbs = grad.Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
            return maxAbs < Scalar(1e-4f);
        }
    }

    /// <summary>RandomNormalLike + RandomUniformLike: sampling — gradient null.</summary>
    [Module]
    public partial class AutoGradRandomLikeStubCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var rnLike = (Tensor<float32>)OnnxOp.RandomNormalLike(x, mean: 0f, scale: 1f, dtype: null, seed: 1f);
            var ruLike = (Tensor<float32>)OnnxOp.RandomUniformLike(x, high: 1f, low: 0f, dtype: null, seed: 1f);
            var loss = (rnLike + ruLike).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            var maxAbs = grad.Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
            return maxAbs < Scalar(1e-4f);
        }
    }

    /// <summary>BlackmanWindow inside a multiply-by-zero masking path: the window is built
    /// from a length parameter (non-diff) and added to the loss through a zero scale,
    /// exercising the Batch28 BlackmanWindow stub.</summary>
    [Module]
    public partial class AutoGradBlackmanWindowStubCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var win = (Tensor<float32>)OnnxOp.BlackmanWindow(Scalar(4L), outputDatatype: DType.Float32, periodic: false);
            var maskScaled = win.Reduce(ReduceKind.Sum, keepDims: false).Scalar() * Scalar(0f);
            var loss = (x.Reduce(ReduceKind.Sum, keepDims: false).Scalar() + maskScaled);
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            var expected = (Tensor<float32>)OnnxOp.Expand(Scalar(1f), x.DShape);
            var diff = (grad - expected).Abs();
            return diff.Reduce(ReduceKind.Max, keepDims: false).Scalar() < Scalar(1e-4f);
        }
    }

    // ===================================================================
    //  Self-checking modules for the AutoDiff null-stubs added alongside
    //  the Batch30 dispatcher registrations: quantization, windows, the
    //  string family, NonMaxSuppression, OptionalHasElement, SequenceLength.
    //  Each module routes a float runtime input through the stub op so its
    //  analytical gradient is zero (or a constant from an unmasked path),
    //  exercising the new RegisterVariadicGradientOps entries via the
    //  FastProcessAutoGrad pipeline.
    // ===================================================================

    /// <summary>QuantizeLinear → DequantizeLinear roundtrip. The float input round-trips
    /// through uint8 (non-diff), so the analytical gradient of <c>x</c> through this path
    /// is zero. Masked via multiply-by-zero so we can still pin <c>dL/dx = 1</c> on the
    /// unmasked sum-of-x path. Uses the typed <c>OnnxOps.QuantizeLinear</c> /
    /// <c>OnnxOps.DequantizeLinear</c> wrappers so generic-type inference for the
    /// output dtype is handled at the C# level rather than via runtime DType plumbing.</summary>
    [Module]
    public partial class AutoGradQuantizeLinearStubCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var scaleVec = (Tensor<float32>)OnnxOp.Expand(Scalar(0.5f), Vector(1L));
            var q = NN.QuantizeLinear<float32, uint8>(x, scaleVec);
            var dq = (Tensor<float32>)OnnxOp.DequantizeLinear(
                q, scaleVec, xZeroPoint: null, axis: null, blockSize: null);
            var maskScaled = dq.Reduce(ReduceKind.Sum, keepDims: false).Scalar() * Scalar(0f);
            var loss = x.Reduce(ReduceKind.Sum, keepDims: false).Scalar() + maskScaled;
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            var expected = (Tensor<float32>)OnnxOp.Expand(Scalar(1f), x.DShape);
            return (grad - expected).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar() < Scalar(1e-4f);
        }
    }

    /// <summary>DynamicQuantizeLinear on a 2D float tensor: produces uint8 output + float scale +
    /// uint8 zero-point. Side-branch is masked to zero; the unmasked Σ(x) path is still 1.</summary>
    [Module]
    public partial class AutoGradDynamicQuantizeLinearStubCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var y = (Tensor<uint8>)OnnxOp.DynamicQuantizeLinear(x).y;
            var sideFloat = y.Cast<float32>().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var maskScaled = sideFloat * Scalar(0f);
            var loss = x.Reduce(ReduceKind.Sum, keepDims: false).Scalar() + maskScaled;
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            var expected = (Tensor<float32>)OnnxOp.Expand(Scalar(1f), x.DShape);
            return (grad - expected).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar() < Scalar(1e-4f);
        }
    }

/// <summary>HammingWindow and HannWindow built from a length scalar (non-diff). The
    /// window contributes through a zero-mask so dL/dx on the unmasked path stays 1.</summary>
    [Module]
    public partial class AutoGradWindowsStubCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var ham = (Tensor<float32>)OnnxOp.HammingWindow(Scalar(4L), outputDatatype: DType.Float32, periodic: true);
            var han = (Tensor<float32>)OnnxOp.HannWindow(Scalar(4L), outputDatatype: DType.Float32, periodic: true);
            var sideSum = ham.Reduce(ReduceKind.Sum, keepDims: false).Scalar()
                        + han.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var maskScaled = sideSum * Scalar(0f);
            var loss = x.Reduce(ReduceKind.Sum, keepDims: false).Scalar() + maskScaled;
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            var expected = (Tensor<float32>)OnnxOp.Expand(Scalar(1f), x.DShape);
            return (grad - expected).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar() < Scalar(1e-4f);
        }
    }

    /// <summary>NonMaxSuppression: integer index output — gradient null at the float
    /// box/score inputs. Masked via multiply-by-zero.</summary>
    [Module]
    public partial class AutoGradNonMaxSuppressionStubCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            // boxes: [1, N, 4] from x; scores: [1, 1, N] also from x — both feed the stub op.
            var boxes = (Tensor<float32>)OnnxOp.Reshape(
                (Tensor<float32>)OnnxOp.Concat([x, x, x, x], axis: 0),
                Vector(1L, -1L, 4L), allowZero: false);
            var scores = (Tensor<float32>)OnnxOp.Reshape(x, Vector(1L, 1L, -1L), allowZero: false);
            var nms = (Tensor<int64>)OnnxOp.NonMaxSuppression(boxes, scores, Scalar(1L));
            var maskScaled = nms.Cast<float32>().Reduce(ReduceKind.Sum, keepDims: false).Scalar() * Scalar(0f);
            var loss = x.Reduce(ReduceKind.Sum, keepDims: false).Scalar() + maskScaled;
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            var expected = (Tensor<float32>)OnnxOp.Expand(Scalar(1f), x.DShape);
            return (grad - expected).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar() < Scalar(1e-4f);
        }
    }

    /// <summary>SequenceLength on a sequence built from a float tensor. The integer length
    /// output contributes through a zero-mask path; dL/dx on the unmasked Σ(x) path is 1.</summary>
    [Module]
    public partial class AutoGradSequenceLengthStubCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var seq = OnnxOp.SequenceConstruct([x, x]);
            var len = (Scalar<int64>)OnnxOp.SequenceLength(seq);
            var maskScaled = len.Cast<float32>() * Scalar(0f);
            var loss = x.Reduce(ReduceKind.Sum, keepDims: false).Scalar() + maskScaled;
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            // Both branches of SequenceConstruct(x, x) feed gradient back, so dL/dx = 0 from
            // the masked path + 1 from the unmasked path. The SequenceConstruct gradient
            // itself routes [grad_seq_at_0, grad_seq_at_1] back to its inputs as zeros under
            // the mask, leaving the loss-direct path unaffected.
            var expected = (Tensor<float32>)OnnxOp.Expand(Scalar(1f), x.DShape);
            return (grad - expected).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar() < Scalar(1e-4f);
        }
    }

    /// <summary>OptionalHasElement on an Optional wrapping a float tensor: bool output, no
    /// float gradient back to the wrapped tensor.</summary>
    [Module]
    public partial class AutoGradOptionalHasElementStubCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var opt = OnnxOp.Optional(x, DataStructure.Tensor, x.Type);
            var has = (Scalar<bit>)OnnxOp.OptionalHasElement(opt);
            var maskScaled = has.Cast<float32>() * Scalar(0f);
            var loss = x.Reduce(ReduceKind.Sum, keepDims: false).Scalar() + maskScaled;
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            var expected = (Tensor<float32>)OnnxOp.Expand(Scalar(1f), x.DShape);
            return (grad - expected).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar() < Scalar(1e-4f);
        }
    }

    /// <summary>STFT on a signal built from x: the spectrogram side-branch is masked
    /// to zero via multiply-by-zero, so dL/dx on the unmasked Σ(x) path stays 1. Since
    /// AD-B3 this exercises the REAL overlap-add adjoint (no longer a null-stub) with a
    /// zero-valued upstream gradient — its contribution must be exactly zero, leaving
    /// dL/dx = 1. The signal shape is <c>[batch=1, signal_length=N, channels=1]</c>.</summary>
    [Module]
    public partial class AutoGradSTFTStubCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            // Reshape x [N] to a signal [1, N, 1]
            var signal = (Tensor<float32>)OnnxOp.Reshape(x, Vector(1L, -1L, 1L), allowZero: false);
            var spec = (Tensor<float32>)OnnxOp.STFT(signal, Scalar(2L), window: null,
                frameLength: Scalar(2L), onesided: true);
            var maskScaled = spec.Reduce(ReduceKind.Sum, keepDims: false).Scalar() * Scalar(0f);
            var loss = x.Reduce(ReduceKind.Sum, keepDims: false).Scalar() + maskScaled;
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            var expected = (Tensor<float32>)OnnxOp.Expand(Scalar(1f), x.DShape);
            return (grad - expected).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar() < Scalar(1e-4f);
        }
    }

    /// <summary>DeformConv on a 1x1x4x4 input and 1x1x3x3 weight, with the input built
    /// from x. Side-branch masked to zero (×0), but the masked branch still routes a
    /// real (zero-valued) gradient into DeformConv — and since AD-B3 replaced the
    /// silent ZERO-STUB with an AD003 guard, the AUTO_GRAD lowering now throws
    /// (asserted via Assert.Throws in TestAutoGradDetectionStubsCoverage). The bit
    /// check below is never reached.</summary>
    [Module]
    public partial class AutoGradDeformConvStubCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            // Build minimal-shape Conv inputs from x. Spatial output for stride=1, pad=0
            // on a 4x4 input with a 3x3 kernel is 2x2; offset shape must be
            // [N, 2 * kH * kW * offset_group, H_out, W_out] = [1, 18, 2, 2].
            var input4d = (Tensor<float32>)OnnxOp.Reshape(x, Vector(1L, 1L, 4L, 4L), allowZero: false);
            var w = (Tensor<float32>)OnnxOp.Expand(Scalar(1f), Vector(1L, 1L, 3L, 3L));
            var offset = (Tensor<float32>)OnnxOp.Expand(Scalar(0f), Vector(1L, 18L, 2L, 2L));
            var dc = (Tensor<float32>)OnnxOp.DeformConv(input4d, w, offset, b: null, mask: null,
                dilations: null, group: null, kernelShape: null, offsetGroup: null,
                pads: null, strides: null);
            var maskScaled = dc.Reduce(ReduceKind.Sum, keepDims: false).Scalar() * Scalar(0f);
            var loss = x.Reduce(ReduceKind.Sum, keepDims: false).Scalar() + maskScaled;
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            var expected = (Tensor<float32>)OnnxOp.Expand(Scalar(1f), x.DShape);
            return (grad - expected).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar() < Scalar(1e-4f);
        }
    }
}
