namespace Shorokoo.Tests.Modules
{
    /// <summary>
    /// Phase 4 AD-B3 modules for the recurrent / signal / misc gradient completion batch
    /// (reverse-direction RNN/GRU/LSTM, DFT onesided + inverse, the STFT overlap-add
    /// adjoint, the generic [2-D and 3-D] AffineGrid gradient, training-mode
    /// BatchNormalization, and the tanh-approximation Gelu derivative). Same
    /// self-checking <c>Scalar&lt;bit&gt;</c> pattern as
    /// <c>AutoGradStructuralModules.cs</c>: most modules verify the analytical gradient
    /// against a two-sided directional derivative computed in-graph (all the ops under
    /// test are linear or smooth at the chosen inputs). The AD003 attribute-envelope
    /// scenarios are driven through <c>Assert.Throws</c> in
    /// <c>AutoGradRecurrentSignalTests</c> (the exception surfaces from the AUTO_GRAD
    /// lowering during concretization, before any execution backend runs).
    /// </summary>

    // ===================================================================
    //  Gelu approximate="tanh": the gradient must use the tanh-approximation
    //  derivative, matching the (QEE-A2-fixed) forward.
    // ===================================================================

    /// <summary>loss = gelu(x, approximate="tanh"). Self-checking at any smooth x —
    /// the FD probes the tanh-approx FORWARD, so an exact-erf gradient (the old
    /// behavior, which ignored the attribute) fails the directional check.</summary>
    [Module]
    public partial class AutoGradGeluTanhCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            Func<Scalar<float32>, Scalar<float32>> f = z => z.Gelu(GeluApproximate.Tanh);
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, f(x));
            return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(x, grad, f);
        }
    }

    // ===================================================================
    //  Reverse-direction RNN / GRU / LSTM. The losses consume BOTH the full
    //  Y sequence and the final state(s), so the dY time-flip AND the
    //  scan-end dYh/dYc mapping of the reverse reduction are exercised.
    //  Weights/inputs come from the shared RecurrentTestData constants.
    // ===================================================================

    internal static class RecurrentReverseHelpers
    {
        public static Scalar<float32> RnnRevLoss(Variable x)
        {
            var (y, yh) = OnnxOp.Rnn(x,
                RecurrentTestData.RnnWConst(),
                RecurrentTestData.RnnRConst(),
                RecurrentTestData.RnnBConst(), null, null,
                null, null, null, null, RNNDirection.Reverse, 2L, false);
            return ((Tensor<float32>)y).Reduce(ReduceKind.Sum, keepDims: false).Scalar()
                 + ((Tensor<float32>)yh).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        }

        public static Scalar<float32> GruRevLoss(Variable x)
        {
            var (y, yh) = OnnxOp.Gru(x,
                RecurrentTestData.GruWConst(),
                RecurrentTestData.GruRConst(),
                RecurrentTestData.GruBConst(), null, null,
                null, null, null, null, GRUDirection.Reverse, 2L, false, null);
            return ((Tensor<float32>)y).Reduce(ReduceKind.Sum, keepDims: false).Scalar()
                 + ((Tensor<float32>)yh).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        }

        public static Scalar<float32> LstmRevLoss(Variable x)
        {
            var (y, yh, yc) = OnnxOp.Lstm(x,
                RecurrentTestData.LstmWConst(),
                RecurrentTestData.LstmRConst(),
                RecurrentTestData.LstmBConst(), null, null, null, null,
                null, null, null, null, LSTMDirection.Reverse, 2L, null, false);
            return ((Tensor<float32>)y).Reduce(ReduceKind.Sum, keepDims: false).Scalar()
                 + ((Tensor<float32>)yh).Reduce(ReduceKind.Sum, keepDims: false).Scalar()
                 + ((Tensor<float32>)yc).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        }
    }

    /// <summary>Reverse-direction RNN, loss = ΣY + ΣY_h over a length-3 sequence whose
    /// x[0,0,0] is the probed scalar. Self-checking via two-sided directional
    /// derivative against ORT's own reverse forward.</summary>
    [Module]
    public partial class AutoGradRnnReverseCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> xv)
        {
            Func<Scalar<float32>, Scalar<float32>> f =
                v => RecurrentReverseHelpers.RnnRevLoss(RecurrentTestData.BuildX(v, 3));
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(xv, f(xv));
            return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(xv, grad, f);
        }
    }

    /// <summary>Reverse-direction GRU (with bias), loss = ΣY + ΣY_h, FD-checked.</summary>
    [Module]
    public partial class AutoGradGruReverseCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> xv)
        {
            Func<Scalar<float32>, Scalar<float32>> f =
                v => RecurrentReverseHelpers.GruRevLoss(RecurrentTestData.BuildX(v, 3));
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(xv, f(xv));
            return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(xv, grad, f);
        }
    }

    /// <summary>Reverse-direction LSTM (with bias), loss = ΣY + ΣY_h + ΣY_c, FD-checked.</summary>
    [Module]
    public partial class AutoGradLstmReverseCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> xv)
        {
            Func<Scalar<float32>, Scalar<float32>> f =
                v => RecurrentReverseHelpers.LstmRevLoss(RecurrentTestData.BuildX(v, 3));
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(xv, f(xv));
            return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(xv, grad, f);
        }
    }

    // ===================================================================
    //  AD003 guards: unsupported recurrent attribute combinations and the
    //  DeformConv adjoint must throw loudly at lowering (asserted via
    //  Assert.Throws in AutoGradRecurrentSignalTests).
    // ===================================================================

    /// <summary>RNN with direction='bidirectional' (correctly-shaped [2,H,*] weights)
    /// on the loss path → AD003 at lowering (bidirectional BPTT unimplemented).</summary>
    [Module]
    public partial class AutoGradRnnBidirectionalThrowCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> xv)
        {
            var w = (Tensor<float32>)OnnxOp.Concat(
                [RecurrentTestData.RnnWConst(),
                 RecurrentTestData.RnnWConst()], axis: 0);
            var r = (Tensor<float32>)OnnxOp.Concat(
                [RecurrentTestData.RnnRConst(),
                 RecurrentTestData.RnnRConst()], axis: 0);
            var (_, yh) = OnnxOp.Rnn(RecurrentTestData.BuildX(xv, 2), w, r, null, null, null,
                null, null, null, null, RNNDirection.Bidirectional, 2L, false);
            var loss = ((Tensor<float32>)yh).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(xv, loss);
            // Never reached: AUTO_GRAD lowering throws AD003 (asserted by the test).
            return grad.Abs() < Scalar(1e9f);
        }
    }

    /// <summary>GRU with the clip attribute on the loss path → AD003 at lowering
    /// (the recomputed forward would ignore the clipping).</summary>
    [Module]
    public partial class AutoGradGruClipThrowCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> xv)
        {
            var (_, yh) = OnnxOp.Gru(RecurrentTestData.BuildX(xv, 2),
                RecurrentTestData.GruWConst(),
                RecurrentTestData.GruRConst(), null, null, null,
                null, null, null, 1.0f, GRUDirection.Forward, 2L, false, null);
            var loss = ((Tensor<float32>)yh).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(xv, loss);
            // Never reached: AUTO_GRAD lowering throws AD003 (asserted by the test).
            return grad.Abs() < Scalar(1e9f);
        }
    }

    /// <summary>LSTM with wired peephole weights (P input) on the loss path → AD003 at
    /// lowering (the BPTT omits the P⊙C gate terms).</summary>
    [Module]
    public partial class AutoGradLstmPeepholeThrowCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> xv)
        {
            var p = OnnxOp.Reshape(Vector(0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f),
                Vector(1L, 6L), allowZero: false);  // [D, 3H] with H=2
            var (_, yh, _) = OnnxOp.Lstm(RecurrentTestData.BuildX(xv, 2),
                RecurrentTestData.LstmWConst(),
                RecurrentTestData.LstmRConst(), null, null,
                null, null, p, null, null, null, null, LSTMDirection.Forward, 2L, null, false);
            var loss = ((Tensor<float32>)yh).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(xv, loss);
            // Never reached: AUTO_GRAD lowering throws AD003 (asserted by the test).
            return grad.Abs() < Scalar(1e9f);
        }
    }

    /// <summary>DeformConv on the loss path → AD003 at lowering (the bilinear-sampling
    /// adjoint is unimplemented; the previous ZERO-STUB silently froze the params).</summary>
    [Module]
    public partial class AutoGradDeformConvThrowCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var x = (Tensor<float32>)OnnxOp.Expand(a, Vector(1L, 1L, 4L, 4L));
            var w = (Tensor<float32>)OnnxOp.Expand(Scalar(0.5f), Vector(1L, 1L, 2L, 2L));
            var offset = (Tensor<float32>)OnnxOp.Expand(Scalar(0.25f), Vector(1L, 8L, 3L, 3L));
            var conved = (Tensor<float32>)OnnxOp.DeformConv(x, w, offset, null, null,
                dilations: null, group: null, kernelShape: [2L, 2L],
                offsetGroup: null, pads: null, strides: null);
            var loss = conved.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            // Never reached: AUTO_GRAD lowering throws AD003 (asserted by the test).
            return grad.Abs() < Scalar(1e9f);
        }
    }

    // ===================================================================
    //  DFT: onesided=1 (RFFT adjoint = zero-pad + full-DFT adjoint) and the
    //  inverse=1 path. Both losses weight every output element differently
    //  so a mis-placed gradient bin cannot cancel out. DFT is linear in x,
    //  so the two-sided FD is exact up to float noise.
    // ===================================================================

    /// <summary>Forward DFT with onesided=1 on a real [1,4,1] signal (3 kept bins).
    /// loss = Σ w·DFT(x); FD-checked on the whole signal tensor.</summary>
    [Module]
    public partial class AutoGradDftOnesidedCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, DftLoss(x));

            var h = Scalar(1e-2f);
            var pert = h * grad;
            var deriv = (DftLoss(x + pert) - DftLoss(x - pert)) / (Scalar(2f) * h);
            var gradNormSq = (grad * grad).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            return (deriv - gradNormSq).Abs() < Scalar(1e-3f) * (gradNormSq.Abs() + Scalar(1f));
        }

        private static Scalar<float32> DftLoss(Tensor<float32> x)
        {
            var dft = (Tensor<float32>)OnnxOp.Dft(x, null, Scalar(1L), inverse: false, onesided: true);
            var w = (Tensor<float32>)OnnxOp.Reshape(
                OnnxOp.Cast(OnnxOp.Range(Scalar(1L), Scalar(7L), Scalar(1L)), saturate: null, to: DType.Float32),
                Vector(1L, 3L, 2L), allowZero: false);
            return (dft * w).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        }
    }

    /// <summary>Inverse DFT (inverse=1, two-sided) on a real [1,4,1] signal.
    /// loss = Σ w·IDFT(x); FD-checked on the whole signal tensor.</summary>
    [Module]
    public partial class AutoGradDftInverseCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, IdftLoss(x));

            var h = Scalar(1e-2f);
            var pert = h * grad;
            var deriv = (IdftLoss(x + pert) - IdftLoss(x - pert)) / (Scalar(2f) * h);
            var gradNormSq = (grad * grad).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            return (deriv - gradNormSq).Abs() < Scalar(1e-3f) * (gradNormSq.Abs() + Scalar(1f));
        }

        private static Scalar<float32> IdftLoss(Tensor<float32> x)
        {
            var idft = (Tensor<float32>)OnnxOp.Dft(x, null, Scalar(1L), inverse: true, onesided: false);
            var w = (Tensor<float32>)OnnxOp.Reshape(
                OnnxOp.Cast(OnnxOp.Range(Scalar(1L), Scalar(9L), Scalar(1L)), saturate: null, to: DType.Float32),
                Vector(1L, 4L, 2L), allowZero: false);
            return (idft * w).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        }
    }

    // ===================================================================
    //  STFT overlap-add adjoint (replaces the AD-B1 ZERO-STUB). Signal
    //  [1,8,1], frame_step 2, window length 4 → 3 OVERLAPPING frames, so
    //  the ScatterElements(Add) overlap-add and the onesided zero-padding
    //  (K = 3 of L = 4 bins, the spec default) are both exercised. STFT is
    //  linear in both signal and window → FD exact up to float noise.
    // ===================================================================

    /// <summary>STFT signal gradient: loss = Σ w·STFT(x, S=2, window), FD-checked on
    /// the whole [1,8,1] signal tensor.</summary>
    [Module]
    public partial class AutoGradStftSignalCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, StftLoss(x));

            var h = Scalar(1e-2f);
            var pert = h * grad;
            var deriv = (StftLoss(x + pert) - StftLoss(x - pert)) / (Scalar(2f) * h);
            var gradNormSq = (grad * grad).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            return (deriv - gradNormSq).Abs() < Scalar(1e-3f) * (gradNormSq.Abs() + Scalar(1f));
        }

        private static Scalar<float32> StftLoss(Tensor<float32> x)
        {
            var window = Vector(0.5f, 1.0f, 0.75f, 0.25f);
            var stft = (Tensor<float32>)OnnxOp.STFT(x, Scalar(2L), window, null);  // [1,3,3,2]
            var w = (Tensor<float32>)OnnxOp.Reshape(
                OnnxOp.Cast(OnnxOp.Range(Scalar(1L), Scalar(19L), Scalar(1L)), saturate: null, to: DType.Float32),
                Vector(1L, 3L, 3L, 2L), allowZero: false);
            return (stft * w).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        }
    }

    /// <summary>STFT window gradient: window[0] is the probed scalar, signal is a fixed
    /// ramp. loss = Σ w·STFT(x, S=2, window(wv)); FD-checked.</summary>
    [Module]
    public partial class AutoGradStftWindowCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> wv)
        {
            Func<Scalar<float32>, Scalar<float32>> f = StftWindowLoss;
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(wv, f(wv));
            return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(wv, grad, f);
        }

        private static Scalar<float32> StftWindowLoss(Scalar<float32> wv)
        {
            var wVec = (Tensor<float32>)OnnxOp.Unsqueeze(wv, Vector(0L));
            var window = (Tensor<float32>)OnnxOp.Concat([wVec, Vector(1.0f, 0.75f, 0.25f)], axis: 0);
            var x = (Tensor<float32>)OnnxOp.Reshape(
                OnnxOp.Cast(OnnxOp.Range(Scalar(1L), Scalar(9L), Scalar(1L)), saturate: null, to: DType.Float32),
                Vector(1L, 8L, 1L), allowZero: false);
            var stft = (Tensor<float32>)OnnxOp.STFT(x, Scalar(2L), window, null);  // [1,3,3,2]
            var w = (Tensor<float32>)OnnxOp.Reshape(
                OnnxOp.Cast(OnnxOp.Range(Scalar(1L), Scalar(19L), Scalar(1L)), saturate: null, to: DType.Float32),
                Vector(1L, 3L, 3L, 2L), allowZero: false);
            return (stft * w).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        }
    }

    /// <summary>STFT with NO window and a wired frame_length input (rectangular window
    /// path: dWindow slot absent, frame-length-driven L). Signal [1,10,1], S=3, L=4 →
    /// 3 non-overlapping-with-gap frames. FD-checked on the signal tensor.</summary>
    [Module]
    public partial class AutoGradStftNoWindowCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, StftLoss(x));

            var h = Scalar(1e-2f);
            var pert = h * grad;
            var deriv = (StftLoss(x + pert) - StftLoss(x - pert)) / (Scalar(2f) * h);
            var gradNormSq = (grad * grad).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            return (deriv - gradNormSq).Abs() < Scalar(1e-3f) * (gradNormSq.Abs() + Scalar(1f));
        }

        private static Scalar<float32> StftLoss(Tensor<float32> x)
        {
            var stft = (Tensor<float32>)OnnxOp.STFT(x, Scalar(3L), null, Scalar(4L));  // [1,3,3,2]
            var w = (Tensor<float32>)OnnxOp.Reshape(
                OnnxOp.Cast(OnnxOp.Range(Scalar(1L), Scalar(19L), Scalar(1L)), saturate: null, to: DType.Float32),
                Vector(1L, 3L, 3L, 2L), allowZero: false);
            return (stft * w).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        }
    }

    // ===================================================================
    //  AffineGrid 3-D (the gradient is now built from the op's own base grid
    //  via an identity theta, so 2-D and 3-D share one code path; the 2-D
    //  case stays covered by AutoGradAffineGridMultiBatchCheck).
    // ===================================================================

    /// <summary>AffineGrid with a 3-D size ([1,1,2,2,2], theta [1,3,4]) and
    /// align_corners=false. The grid is linear in theta = base·a, so the FD on the
    /// position-weighted loss is exact.</summary>
    [Module]
    public partial class AutoGradAffineGrid3DCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            Func<Scalar<float32>, Scalar<float32>> f = GridLoss;
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, f(a));
            return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(a, grad, f);
        }

        private static Scalar<float32> GridLoss(Scalar<float32> a)
        {
            var thetaBase = (Tensor<float32>)OnnxOp.Reshape(
                OnnxOp.Cast(OnnxOp.Range(Scalar(1L), Scalar(13L), Scalar(1L)), saturate: null, to: DType.Float32),
                Vector(1L, 3L, 4L), allowZero: false);
            var theta = thetaBase * a * Scalar(0.1f);
            var size = (Tensor<int64>)OnnxOp.Constant(TensorData(5, 1L, 1L, 2L, 2L, 2L));
            var grid = (Tensor<float32>)OnnxOp.AffineGrid(theta, size, alignCorners: false); // [1,2,2,2,3]
            var w = (Tensor<float32>)OnnxOp.Reshape(
                OnnxOp.Cast(OnnxOp.Range(Scalar(1L), Scalar(25L), Scalar(1L)), saturate: null, to: DType.Float32),
                Vector(1L, 2L, 2L, 2L, 3L), allowZero: false);
            return (grid * w).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        }
    }

    // ===================================================================
    //  BatchNormalization training_mode=1 (implemented this batch — was an
    //  AD-B2 AD003 guard): dx must carry the batch-statistics backprop
    //  terms, dscale/dbias must use the batch-stat x̂.
    // ===================================================================

    /// <summary>Training-mode BN dx. x = c·a + c² ([1,2,2,2], c = 1..8) so the loss
    /// depends on `a` through the batch statistics non-trivially (a pure x = c·a would
    /// be scale-invariant under training BN and the gradient would degenerate to ~0).
    /// Self-checking via a FIXED-step two-sided FD (h = 0.05): the gradient here is
    /// ~0.03, so the usual pert = h·grad probe moves the float32 loss (≈3.05) by less
    /// than its rounding noise; a fixed step keeps Δloss ≫ noise while the curvature
    /// stays negligible (truncation ~5e-6 at this h).</summary>
    [Module]
    public partial class AutoGradBatchNormTrainingInputCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, BnLoss(a));
            var h = Scalar(0.05f);
            var deriv = (BnLoss(a + h) - BnLoss(a - h)) / (Scalar(2f) * h);
            return (deriv - grad).Abs() < Scalar(2e-3f) * (grad.Abs() + Scalar(1f));
        }

        private static Scalar<float32> BnLoss(Scalar<float32> a)
        {
            var c = (Tensor<float32>)OnnxOp.Reshape(
                OnnxOp.Cast(OnnxOp.Range(Scalar(1L), Scalar(9L), Scalar(1L)), saturate: null, to: DType.Float32),
                Vector(1L, 2L, 2L, 2L), allowZero: false);
            var x = c * a + c * c;
            var scale = Vector(1.5f, 0.5f).Tensor();
            var bias = Vector(0.2f, -0.3f).Tensor();
            var mean = Vector(0.0f, 0.0f).Tensor();
            var variance = Vector(1.0f, 1.0f).Tensor();
            var y = (Tensor<float32>)OnnxOp.BatchNormalization(x, scale, bias, mean, variance,
                epsilon: 1e-5f, momentum: null, trainingMode: true);
            return (y * c).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        }
    }

    /// <summary>Training-mode BN dscale/dbias against in-graph closed forms: with
    /// x = grid 1..8 ([1,2,2,2]) and loss = Σ y·w (w = grid), dL/ds = Σ w·x̂ (x̂ from
    /// the BATCH statistics, computed in-graph) and dL/dt = Σ w = 36.</summary>
    [Module]
    public partial class AutoGradBatchNormTrainingScaleBiasCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> s, Scalar<float32> t)
        {
            var grid = (Tensor<float32>)OnnxOp.Reshape(
                OnnxOp.Cast(OnnxOp.Range(Scalar(1L), Scalar(9L), Scalar(1L)), saturate: null, to: DType.Float32),
                Vector(1L, 2L, 2L, 2L), allowZero: false);
            var scale = (Tensor<float32>)OnnxOp.Expand(s, Vector(2L));
            var bias = (Tensor<float32>)OnnxOp.Expand(t, Vector(2L));
            var mean = Vector(0.0f, 0.0f).Tensor();
            var variance = Vector(1.0f, 1.0f).Tensor();
            var y = (Tensor<float32>)OnnxOp.BatchNormalization(grid, scale, bias, mean, variance,
                epsilon: 1e-5f, momentum: null, trainingMode: true);
            var loss = (y * grid).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var (gradS, gradT) = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(s, t, loss);

            // In-graph expected dscale: Σ w·x̂ with batch stats over axes [0,2,3].
            var axes = Vector(0L, 2L, 3L);
            var batchMean = (Tensor<float32>)OnnxOp.ReduceMean(grid, axes, keepdims: true, noopWithEmptyAxes: null);
            var xc = grid - batchMean;
            var batchVar = (Tensor<float32>)OnnxOp.ReduceMean(xc * xc, axes, keepdims: true, noopWithEmptyAxes: null);
            var xHat = xc / (batchVar + Scalar(1e-5f)).Sqrt();
            var expectedS = (grid * xHat).Reduce(ReduceKind.Sum, keepDims: false).Scalar();

            var okS = (gradS! - expectedS).Abs() < Scalar(1e-3f);
            var okT = (gradT! - Scalar(36f)).Abs() < Scalar(1e-3f);
            return okS & okT;
        }
    }
}
