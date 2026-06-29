namespace Shorokoo.Tests.Modules
{
    /// <summary>
    /// Phase 4 AD-B1 gradient-correctness modules for the structural op family
    /// (Conv / ConvTranspose / MaxPool / AveragePool / GlobalAveragePool /
    /// BatchNorm / LayerNorm / GroupNorm / InstanceNorm / Concat / Split /
    /// Sum / Min / Max / Mean / Dropout). Same self-checking pattern as
    /// <c>AutoGradTestModules.cs</c>: each <c>Inline</c> computes a small loss
    /// through the op, calls <c>Ops.AutoGrad</c>, compares against a closed-form
    /// or two-sided-directional-derivative expectation computed in-graph, and
    /// returns <c>Scalar&lt;bit&gt;</c>. Where pooling argmax routing matters,
    /// the inputs are strictly increasing grids (built from Range) so the
    /// expected routing is deterministic and the closed-form coefficient exact.
    /// </summary>

    // ===================================================================
    //  Conv: stride / asymmetric pads / dilations / group attribute coverage
    // ===================================================================

    /// <summary>
    /// Conv dx with stride 2 and asymmetric pads [1,0,1,0]. Self-checking via
    /// two-sided directional derivative. Exercises the output_padding=stride−1 +
    /// slice-to-input-shape path in <c>ConvGradient</c> with non-uniform padding.
    /// </summary>
    [Module]
    public partial class AutoGradStructConvStridePadCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var w = InitSimple.Init([Scalar(2L), Scalar(2L), Scalar(3L), Scalar(3L)]);
            var b = InitSimple.Init([Scalar(2L)]).Vec();
            var loss = ConvLoss(x, w, b);
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);

            var h = Scalar(1e-3f);
            var pert = h * grad;
            var deriv = (ConvLoss(x + pert, w, b) - ConvLoss(x - pert, w, b)) / (Scalar(2f) * h);
            var gradNormSq = (grad * grad).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            return (deriv - gradNormSq).Abs() < Scalar(1e-3f) * (gradNormSq.Abs() + Scalar(1f));
        }

        private static Scalar<float32> ConvLoss(Tensor<float32> x, Tensor<float32> w, Vector<float32> b)
        {
            var conved = NN.Conv(x, w, b, AutoPad.NotSet,
                dilations: [1L, 1L], group: 1L, kernelShape: [3L, 3L],
                pads: [1L, 0L, 1L, 0L], strides: [2L, 2L]);
            return conved.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        }
    }

    /// <summary>
    /// Conv dx with dilations [2,2]. Self-checking via two-sided directional
    /// derivative. Exercises the dilation pass-through into the ConvTranspose
    /// dx subgraph of <c>ConvGradient</c>.
    /// </summary>
    [Module]
    public partial class AutoGradStructConvDilationCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var w = InitSimple.Init([Scalar(2L), Scalar(2L), Scalar(3L), Scalar(3L)]);
            var b = InitSimple.Init([Scalar(2L)]).Vec();
            var loss = ConvLoss(x, w, b);
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);

            var h = Scalar(1e-3f);
            var pert = h * grad;
            var deriv = (ConvLoss(x + pert, w, b) - ConvLoss(x - pert, w, b)) / (Scalar(2f) * h);
            var gradNormSq = (grad * grad).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            return (deriv - gradNormSq).Abs() < Scalar(1e-3f) * (gradNormSq.Abs() + Scalar(1f));
        }

        private static Scalar<float32> ConvLoss(Tensor<float32> x, Tensor<float32> w, Vector<float32> b)
        {
            var conved = NN.Conv(x, w, b, AutoPad.NotSet,
                dilations: [2L, 2L], group: 1L, kernelShape: [3L, 3L],
                pads: [0L, 0L, 0L, 0L], strides: [1L, 1L]);
            return conved.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        }
    }

    /// <summary>
    /// Conv dw with stride 2 (kernel [2,2] on a 5×5 input, output 2×2). The
    /// weight-gradient conv (strides/dilations swapped) overruns the kernel
    /// spatial size by stride−1 here, so this drives the slice-to-w-shape path
    /// in <c>ConvGradient</c>. Self-checking via two-sided directional
    /// derivative on <c>w</c> (loss is linear in w, so the FD is exact).
    /// </summary>
    [Module]
    public partial class AutoGradStructConvWeightStride2Check
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var w = InitSimple.Init([Scalar(2L), Scalar(2L), Scalar(2L), Scalar(2L)]);
            var b = InitSimple.Init([Scalar(2L)]).Vec();
            var loss = ConvLoss(x, w, b);
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(w, loss);

            var h = Scalar(1e-3f);
            var pert = h * grad;
            var deriv = (ConvLoss(x, w + pert, b) - ConvLoss(x, w - pert, b)) / (Scalar(2f) * h);
            var gradNormSq = (grad * grad).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            return (deriv - gradNormSq).Abs() < Scalar(1e-3f) * (gradNormSq.Abs() + Scalar(1f));
        }

        private static Scalar<float32> ConvLoss(Tensor<float32> x, Tensor<float32> w, Vector<float32> b)
        {
            var conved = NN.Conv(x, w, b, AutoPad.NotSet,
                dilations: [1L, 1L], group: 1L, kernelShape: [2L, 2L],
                pads: [0L, 0L, 0L, 0L], strides: [2L, 2L]);
            return conved.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        }
    }

    /// <summary>
    /// Grouped Conv (group=2) dx. Self-checking via two-sided directional
    /// derivative. <c>ConvGradient</c> passes <c>group</c> through to the
    /// ConvTranspose dx subgraph, which this verifies numerically.
    /// </summary>
    [Module]
    public partial class AutoGradStructConvGroupedInputCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var w = InitSimple.Init([Scalar(4L), Scalar(2L), Scalar(3L), Scalar(3L)]);
            var b = InitSimple.Init([Scalar(4L)]).Vec();
            var loss = ConvLoss(x, w, b);
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);

            var h = Scalar(1e-3f);
            var pert = h * grad;
            var deriv = (ConvLoss(x + pert, w, b) - ConvLoss(x - pert, w, b)) / (Scalar(2f) * h);
            var gradNormSq = (grad * grad).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            return (deriv - gradNormSq).Abs() < Scalar(1e-3f) * (gradNormSq.Abs() + Scalar(1f));
        }

        private static Scalar<float32> ConvLoss(Tensor<float32> x, Tensor<float32> w, Vector<float32> b)
        {
            var conved = NN.Conv(x, w, b, AutoPad.NotSet,
                dilations: [1L, 1L], group: 2L, kernelShape: [3L, 3L],
                pads: [0L, 0L, 0L, 0L], strides: [1L, 1L]);
            return conved.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        }
    }

    /// <summary>
    /// Grouped Conv (group=2) weight gradient, closed form. <c>ConvGradient</c>
    /// historically returned a null dw whenever group != 1 (silently freezing
    /// every grouped/depthwise Conv kernel during training); the per-group
    /// swapped-roles Conv + Concat in <c>GroupedConvWeightGradient</c> now
    /// computes it. With <c>x ≡ 0.1</c> ([1,4,5,5]), kernel 3×3, stride 1, no
    /// pads, every weight element's gradient is OH·OW·0.1 = 9·0.1 = 0.9.
    /// </summary>
    [Module]
    public partial class AutoGradStructConvGroupedWeightCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var w = InitSimple.Init([Scalar(4L), Scalar(2L), Scalar(3L), Scalar(3L)]);
            var b = InitSimple.Init([Scalar(4L)]).Vec();
            var loss = ConvLoss(x, w, b);
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(w, loss);

            // Closed form: every weight element's gradient is OH*OW * x_val = 9 * 0.1 = 0.9.
            var diff = (grad - Scalar(0.9f)).Abs();
            return diff.Reduce(ReduceKind.Max, keepDims: false).Scalar() < Scalar(1e-3f);
        }

        private static Scalar<float32> ConvLoss(Tensor<float32> x, Tensor<float32> w, Vector<float32> b)
        {
            var conved = NN.Conv(x, w, b, AutoPad.NotSet,
                dilations: [1L, 1L], group: 2L, kernelShape: [3L, 3L],
                pads: [0L, 0L, 0L, 0L], strides: [1L, 1L]);
            return conved.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        }
    }

    /// <summary>
    /// Grouped Conv (group=2) weight gradient with stride 2 AND asymmetric pads
    /// [1,0,1,0] — drives the per-group swapped-roles Conv with non-trivial
    /// stride/pad attributes plus the slice-to-w-shape path. Self-checking via
    /// two-sided directional derivative on <c>w</c> (loss is linear in w, so the
    /// FD is exact).
    /// </summary>
    [Module]
    public partial class AutoGradStructConvGroupedWeightStridePadCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var w = InitSimple.Init([Scalar(4L), Scalar(2L), Scalar(2L), Scalar(2L)]);
            var b = InitSimple.Init([Scalar(4L)]).Vec();
            var loss = ConvLoss(x, w, b);
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(w, loss);

            var h = Scalar(1e-3f);
            var pert = h * grad;
            var deriv = (ConvLoss(x, w + pert, b) - ConvLoss(x, w - pert, b)) / (Scalar(2f) * h);
            var gradNormSq = (grad * grad).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            return (deriv - gradNormSq).Abs() < Scalar(1e-3f) * (gradNormSq.Abs() + Scalar(1f));
        }

        private static Scalar<float32> ConvLoss(Tensor<float32> x, Tensor<float32> w, Vector<float32> b)
        {
            var conved = NN.Conv(x, w, b, AutoPad.NotSet,
                dilations: [1L, 1L], group: 2L, kernelShape: [2L, 2L],
                pads: [1L, 0L, 1L, 0L], strides: [2L, 2L]);
            return conved.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        }
    }

    /// <summary>
    /// Grouped ConvTranspose (group=2) weight gradient — the same per-group
    /// machinery as grouped Conv dw with the image/kernel roles swapped. For
    /// <c>loss = Σ ConvTranspose(x, w, b)</c> with stride 1 / no pads, every
    /// (input-position, kernel-position) pair maps to a valid output, so with
    /// <c>x ≡ 0.1</c> ([1,4,5,5]) every weight element's gradient is
    /// H·W·0.1 = 25·0.1 = 2.5.
    /// </summary>
    [Module]
    public partial class AutoGradStructConvTransposeGroupedWeightCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var w = InitSimple.Init([Scalar(4L), Scalar(2L), Scalar(3L), Scalar(3L)]);
            var b = InitSimple.Init([Scalar(4L)]).Vec();
            var loss = ConvTLoss(x, w, b);
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(w, loss);

            var diff = (grad - Scalar(2.5f)).Abs();
            return diff.Reduce(ReduceKind.Max, keepDims: false).Scalar() < Scalar(1e-3f);
        }

        private static Scalar<float32> ConvTLoss(Tensor<float32> x, Tensor<float32> w, Vector<float32> b)
        {
            var convT = (Tensor<float32>)NN.ConvTranspose(x, w, b, AutoPad.NotSet,
                dilations: [1L, 1L], group: 2L, kernelShape: [3L, 3L],
                outputPadding: [0L, 0L], outputShape: [],
                pads: [0L, 0L, 0L, 0L], strides: [1L, 1L]);
            return convT.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        }
    }

    // ===================================================================
    //  ConvTranspose: weight gradient (historically null → silently zero)
    // ===================================================================

    /// <summary>
    /// ConvTranspose dw — the path <c>ConvTransposeGradient</c> historically
    /// dropped (returned a null weight gradient, silently freezing every
    /// ConvTranspose kernel at its initial value throughout training). For
    /// <c>loss = Σ ConvTranspose(x, w, b)</c> with stride 1 / no pads, every
    /// (input-position, kernel-position) pair maps to a valid output, so
    /// <c>dL/dw[ci,co,kh,kw] = Σ_{n,ih,iw} x[n,ci,ih,iw]</c>. With
    /// <c>x ≡ 0.1</c> ([1,3,5,5]) that is 25·0.1 = 2.5 for every weight element.
    /// </summary>
    [Module]
    public partial class AutoGradStructConvTransposeWeightCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var w = InitSimple.Init([Scalar(3L), Scalar(2L), Scalar(3L), Scalar(3L)]);
            var b = InitSimple.Init([Scalar(2L)]).Vec();
            var loss = ConvTLoss(x, w, b);
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(w, loss);

            // Closed form: every weight element's gradient is H*W * x_val = 25 * 0.1 = 2.5.
            var diff = (grad - Scalar(2.5f)).Abs();
            return diff.Reduce(ReduceKind.Max, keepDims: false).Scalar() < Scalar(1e-3f);
        }

        private static Scalar<float32> ConvTLoss(Tensor<float32> x, Tensor<float32> w, Vector<float32> b)
        {
            var convT = (Tensor<float32>)NN.ConvTranspose(x, w, b, AutoPad.NotSet,
                dilations: [1L, 1L], group: 1L, kernelShape: [3L, 3L],
                outputPadding: [0L, 0L], outputShape: [],
                pads: [0L, 0L, 0L, 0L], strides: [1L, 1L]);
            return convT.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        }
    }

    /// <summary>
    /// ConvTranspose dx with stride 2 and output_padding [1,1]. Self-checking
    /// via two-sided directional derivative. The dx Conv subgraph must absorb
    /// the output_padding rows implicitly (output_padding &lt; stride keeps the
    /// dx spatial size exact).
    /// </summary>
    [Module]
    public partial class AutoGradStructConvTransposeStride2Check
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var w = InitSimple.Init([Scalar(2L), Scalar(2L), Scalar(3L), Scalar(3L)]);
            var b = InitSimple.Init([Scalar(2L)]).Vec();
            var loss = ConvTLoss(x, w, b);
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);

            var h = Scalar(1e-3f);
            var pert = h * grad;
            var deriv = (ConvTLoss(x + pert, w, b) - ConvTLoss(x - pert, w, b)) / (Scalar(2f) * h);
            var gradNormSq = (grad * grad).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            return (deriv - gradNormSq).Abs() < Scalar(1e-3f) * (gradNormSq.Abs() + Scalar(1f));
        }

        private static Scalar<float32> ConvTLoss(Tensor<float32> x, Tensor<float32> w, Vector<float32> b)
        {
            var convT = (Tensor<float32>)NN.ConvTranspose(x, w, b, AutoPad.NotSet,
                dilations: [1L, 1L], group: 1L, kernelShape: [3L, 3L],
                outputPadding: [1L, 1L], outputShape: [],
                pads: [0L, 0L, 0L, 0L], strides: [2L, 2L]);
            return convT.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        }
    }

    /// <summary>
    /// ConvTranspose dw with stride 2 + output_padding [1,1]. The weight-gradient
    /// conv (roles of x/grad swapped, strides/dilations swapped) overruns the
    /// kernel size by floor(output_padding/dilation) here, driving the
    /// slice-to-w-shape path. Self-checking via two-sided directional derivative
    /// on <c>w</c> (loss is linear in w, so the FD is exact).
    /// </summary>
    [Module]
    public partial class AutoGradStructConvTransposeWeightStride2Check
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var w = InitSimple.Init([Scalar(2L), Scalar(2L), Scalar(3L), Scalar(3L)]);
            var b = InitSimple.Init([Scalar(2L)]).Vec();
            var loss = ConvTLoss(x, w, b);
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(w, loss);

            var h = Scalar(1e-3f);
            var pert = h * grad;
            var deriv = (ConvTLoss(x, w + pert, b) - ConvTLoss(x, w - pert, b)) / (Scalar(2f) * h);
            var gradNormSq = (grad * grad).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            return (deriv - gradNormSq).Abs() < Scalar(1e-3f) * (gradNormSq.Abs() + Scalar(1f));
        }

        private static Scalar<float32> ConvTLoss(Tensor<float32> x, Tensor<float32> w, Vector<float32> b)
        {
            var convT = (Tensor<float32>)NN.ConvTranspose(x, w, b, AutoPad.NotSet,
                dilations: [1L, 1L], group: 1L, kernelShape: [3L, 3L],
                outputPadding: [1L, 1L], outputShape: [],
                pads: [0L, 0L, 0L, 0L], strides: [2L, 2L]);
            return convT.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        }
    }

    // ===================================================================
    //  MaxPool: overlapping windows / pads / ceil_mode / dilations.
    //  Each module pools a strictly increasing 1..N grid scaled by `a`, so
    //  every window's argmax is its highest-flat-index covered cell and the
    //  closed-form dL/da is the exact sum of the winning grid values.
    // ===================================================================

    /// <summary>
    /// Overlapping MaxPool (kernel 2, stride 1) on a 4×4 grid of 1..16 times a.
    /// Each 2×2 window's max is its bottom-right cell, so
    /// loss = a·(6+7+8+10+11+12+14+15+16) = 99a → dL/da = 99. The previous
    /// Resize-based gradient mis-routed overlapping windows; the indices-based
    /// scatter handles them exactly.
    /// </summary>
    [Module]
    public partial class AutoGradStructMaxPoolOverlapCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var grid = (Tensor<float32>)OnnxOp.Reshape(
                OnnxOp.Cast(OnnxOp.Range(Scalar(1L), Scalar(17L), Scalar(1L)), saturate: null, to: DType.Float32),
                Vector(1L, 1L, 4L, 4L), allowZero: false);
            var x = grid * a;
            var pooled = (Tensor<float32>)OnnxOp.MaxPool(x,
                autoPad: AutoPad.NotSet, ceilMode: false, dilations: [1L, 1L],
                kernelShape: [2L, 2L], pads: [0L, 0L, 0L, 0L], storageOrder: 0L, strides: [1L, 1L]);
            var loss = pooled.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(99f)).Abs() < Scalar(1e-2f);
        }
    }

    /// <summary>
    /// MaxPool with pads [1,1,1,1] (kernel 2, stride 2) on a 4×4 grid of 1..16
    /// times a. Window starts at −1/1/3 along each axis; per-window argmaxes sit
    /// at rows {0,2,3} × cols {0,2,3}, so loss = a·(1+3+4+9+11+12+13+15+16) =
    /// 84a → dL/da = 84.
    /// </summary>
    [Module]
    public partial class AutoGradStructMaxPoolPadCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var grid = (Tensor<float32>)OnnxOp.Reshape(
                OnnxOp.Cast(OnnxOp.Range(Scalar(1L), Scalar(17L), Scalar(1L)), saturate: null, to: DType.Float32),
                Vector(1L, 1L, 4L, 4L), allowZero: false);
            var x = grid * a;
            var pooled = (Tensor<float32>)OnnxOp.MaxPool(x,
                autoPad: AutoPad.NotSet, ceilMode: false, dilations: [1L, 1L],
                kernelShape: [2L, 2L], pads: [1L, 1L, 1L, 1L], storageOrder: 0L, strides: [2L, 2L]);
            var loss = pooled.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(84f)).Abs() < Scalar(1e-2f);
        }
    }

    /// <summary>
    /// MaxPool with ceil_mode=true (kernel 2, stride 2) on a 5×5 grid of 1..25
    /// times a. The trailing partial windows contribute; argmaxes sit at rows
    /// {1,3,4} × cols {1,3,4}, so loss = a·(7+9+10+17+19+20+22+24+25) = 153a →
    /// dL/da = 153.
    /// </summary>
    [Module]
    public partial class AutoGradStructMaxPoolCeilCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var grid = (Tensor<float32>)OnnxOp.Reshape(
                OnnxOp.Cast(OnnxOp.Range(Scalar(1L), Scalar(26L), Scalar(1L)), saturate: null, to: DType.Float32),
                Vector(1L, 1L, 5L, 5L), allowZero: false);
            var x = grid * a;
            var pooled = (Tensor<float32>)OnnxOp.MaxPool(x,
                autoPad: AutoPad.NotSet, ceilMode: true, dilations: [1L, 1L],
                kernelShape: [2L, 2L], pads: [0L, 0L, 0L, 0L], storageOrder: 0L, strides: [2L, 2L]);
            var loss = pooled.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(153f)).Abs() < Scalar(1e-2f);
        }
    }

    /// <summary>
    /// MaxPool with dilations [2,2] (kernel 2, stride 1) on a 4×4 grid of 1..16
    /// times a. Each dilated window covers {(r,c),(r,c+2),(r+2,c),(r+2,c+2)};
    /// argmax at (r+2,c+2), so loss = a·(11+12+15+16) = 54a → dL/da = 54.
    /// </summary>
    [Module]
    public partial class AutoGradStructMaxPoolDilationCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var grid = (Tensor<float32>)OnnxOp.Reshape(
                OnnxOp.Cast(OnnxOp.Range(Scalar(1L), Scalar(17L), Scalar(1L)), saturate: null, to: DType.Float32),
                Vector(1L, 1L, 4L, 4L), allowZero: false);
            var x = grid * a;
            var pooled = (Tensor<float32>)OnnxOp.MaxPool(x,
                autoPad: AutoPad.NotSet, ceilMode: false, dilations: [2L, 2L],
                kernelShape: [2L, 2L], pads: [0L, 0L, 0L, 0L], storageOrder: 0L, strides: [1L, 1L]);
            var loss = pooled.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(54f)).Abs() < Scalar(1e-2f);
        }
    }

    // ===================================================================
    //  AveragePool dilations + GlobalAveragePool 5-D
    // ===================================================================

    /// <summary>
    /// AveragePool with dilations [2,2] (kernel 2, stride 1) on a 4×4 grid of
    /// 1..16 times a. The four dilated windows tile the input exactly once, so
    /// loss = (1/4)·a·Σ(1..16) = 34a → dL/da = 34. Drives the dilation
    /// pass-through into the Col2Im fold of <c>AveragePoolGradient</c>.
    /// </summary>
    [Module]
    public partial class AutoGradStructAvgPoolDilationCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var grid = (Tensor<float32>)OnnxOp.Reshape(
                OnnxOp.Cast(OnnxOp.Range(Scalar(1L), Scalar(17L), Scalar(1L)), saturate: null, to: DType.Float32),
                Vector(1L, 1L, 4L, 4L), allowZero: false);
            var x = grid * a;
            var pooled = (Tensor<float32>)OnnxOp.AveragePool(
                x, autoPad: AutoPad.NotSet, ceilMode: false, countIncludePad: false,
                dilations: [2L, 2L], kernelShape: [2L, 2L],
                pads: [0L, 0L, 0L, 0L], strides: [1L, 1L]);
            var loss = pooled.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(34f)).Abs() < Scalar(1e-2f);
        }
    }

    /// <summary>
    /// GlobalAveragePool on a 5-D input [1,2,2,2,2]: loss = Σ_c mean_spatial = 2a
    /// → dL/da = 2. Exercises the rank-agnostic spatial-size arithmetic in the
    /// <c>GlobalAveragePool</c> [AutoDiff] entry beyond the usual 4-D case.
    /// </summary>
    [Module]
    public partial class AutoGradStructGlobalAvgPool5DCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var x = (Tensor<float32>)OnnxOp.Expand(a, Vector(1L, 2L, 2L, 2L, 2L));
            var pooled = (Tensor<float32>)OnnxOp.GlobalAveragePool(x);
            var loss = pooled.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(2f)).Abs() < Scalar(1e-4f);
        }
    }

    // ===================================================================
    //  Normalization scale/bias gradients (dscale / dbias closed forms)
    // ===================================================================

    /// <summary>
    /// BatchNormalization dscale/dbias. x = [[1,2]] with mean 0 / var 1 / eps
    /// 1e-8 → x̂ ≈ x, so for loss = Σ y: dL/ds = Σ_c x̂_c = 3 and dL/dt = 2
    /// (channel count). Verifies the dscale/dbias reductions of
    /// <c>BatchNormalizationGradient</c> (existing tests only cover dx).
    /// </summary>
    [Module]
    public partial class AutoGradStructBatchNormScaleBiasCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> s, Scalar<float32> t)
        {
            var x = (Tensor<float32>)OnnxOp.Reshape(Vector(1f, 2f), Vector(1L, 2L), allowZero: false);
            var scale = (Tensor<float32>)OnnxOp.Expand(s, Vector(2L));
            var bias = (Tensor<float32>)OnnxOp.Expand(t, Vector(2L));
            var mean = (Tensor<float32>)OnnxOp.Expand(Scalar(0f), Vector(2L));
            var variance = (Tensor<float32>)OnnxOp.Expand(Scalar(1f), Vector(2L));
            var bn = (Tensor<float32>)OnnxOp.BatchNormalization(x, scale, bias, mean, variance,
                epsilon: 1e-8f, momentum: null, trainingMode: false);
            var loss = bn.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var (gradS, gradT) = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(s, t, loss);
            var okS = (gradS! - Scalar(3f)).Abs() < Scalar(1e-4f);
            var okT = (gradT! - Scalar(2f)).Abs() < Scalar(1e-4f);
            return okS & okT;
        }
    }

    /// <summary>
    /// LayerNormalization dscale/dbias over the last axis. x = [1,2,3,4]
    /// (mean 2.5, var 1.25), loss = Σ y·w with w = [1,2,3,4]:
    /// dL/ds = Σ w·x̂ = 5/√(1.25+ε), dL/dt = Σ w = 10. The expected dscale is
    /// computed in-graph so the epsilon term stays exact.
    /// </summary>
    [Module]
    public partial class AutoGradStructLayerNormScaleBiasCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> s, Scalar<float32> t)
        {
            var x = Vector(1f, 2f, 3f, 4f).Tensor();
            var weights = Vector(1f, 2f, 3f, 4f).Tensor();
            var scale = (Tensor<float32>)OnnxOp.Expand(s, Vector(4L));
            var bias = (Tensor<float32>)OnnxOp.Expand(t, Vector(4L));
            var y = (Tensor<float32>)OnnxOp.LayerNormalization(x, scale, bias, axis: -1, epsilon: 1e-5f).y;
            var loss = (y * weights).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var (gradS, gradT) = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(s, t, loss);

            var expectedS = Scalar(5f) / (Scalar(1.25f) + Scalar(1e-5f)).Sqrt();
            var okS = (gradS! - expectedS).Abs() < Scalar(1e-3f);
            var okT = (gradT! - Scalar(10f)).Abs() < Scalar(1e-3f);
            return okS & okT;
        }
    }

    /// <summary>
    /// LayerNormalization dx with a positive (non-default) axis on a 2-D input.
    /// x = c·a + c² with c = 1..8 reshaped [2,4], normalized over axis 1, with a
    /// weighted loss so the gradient is non-trivial. Self-checking via two-sided
    /// directional derivative on <c>a</c>. Exercises the axisAttr ≥ 0 branch of
    /// <c>LayerNormalizationGradient</c>.
    /// </summary>
    [Module]
    public partial class AutoGradStructLayerNormAxis1Check
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var loss = LnLoss(a);
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);

            var h = Scalar(1e-3f);
            var pert = h * grad;
            var deriv = (LnLoss(a + pert) - LnLoss(a - pert)) / (Scalar(2f) * h);
            var gradSq = grad * grad;
            return (deriv - gradSq).Abs() < Scalar(1e-3f) * (gradSq.Abs() + Scalar(1f));
        }

        private static Scalar<float32> LnLoss(Scalar<float32> a)
        {
            var c = (Tensor<float32>)OnnxOp.Reshape(
                OnnxOp.Cast(OnnxOp.Range(Scalar(1L), Scalar(9L), Scalar(1L)), saturate: null, to: DType.Float32),
                Vector(2L, 4L), allowZero: false);
            var x = c * a + c * c;
            var scale = (Tensor<float32>)OnnxOp.Expand(Scalar(1f), Vector(4L));
            var bias = (Tensor<float32>)OnnxOp.Expand(Scalar(0f), Vector(4L));
            var y = (Tensor<float32>)OnnxOp.LayerNormalization(x, scale, bias, axis: 1, epsilon: 1e-5f).y;
            return (y * c).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        }
    }

    /// <summary>
    /// GroupNormalization dscale/dbias with 2 groups. Input [1,4,1,2] of 1..8,
    /// weighted loss with w = 1..8: each group contributes Σ w·(v−mean) = 5, so
    /// dL/ds = 10/√(1.25+ε) and dL/dt = Σ w = 36 (expected dscale computed
    /// in-graph). Verifies <c>GroupNormalization</c>'s dscale/dbias reductions.
    /// </summary>
    [Module]
    public partial class AutoGradStructGroupNormScaleBiasCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> s, Scalar<float32> t)
        {
            var grid = (Tensor<float32>)OnnxOp.Reshape(
                OnnxOp.Cast(OnnxOp.Range(Scalar(1L), Scalar(9L), Scalar(1L)), saturate: null, to: DType.Float32),
                Vector(1L, 4L, 1L, 2L), allowZero: false);
            var scale = (Tensor<float32>)OnnxOp.Expand(s, Vector(4L));
            var bias = (Tensor<float32>)OnnxOp.Expand(t, Vector(4L));
            var y = (Tensor<float32>)OnnxOp.GroupNormalization(grid, scale, bias,
                epsilon: 1e-5f, numGroups: 2);
            var loss = (y * grid).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var (gradS, gradT) = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(s, t, loss);

            var expectedS = Scalar(10f) / (Scalar(1.25f) + Scalar(1e-5f)).Sqrt();
            var okS = (gradS! - expectedS).Abs() < Scalar(1e-3f);
            var okT = (gradT! - Scalar(36f)).Abs() < Scalar(1e-3f);
            return okS & okT;
        }
    }

    /// <summary>
    /// InstanceNormalization dscale/dbias. Input [1,2,2,2] of 1..8, weighted
    /// loss with w = 1..8: each (n,c) instance contributes Σ w·(v−mean) = 5, so
    /// dL/ds = 10/√(1.25+ε) and dL/dt = Σ w = 36 (expected dscale computed
    /// in-graph). Verifies <c>InstanceNormalization</c>'s dscale/dbias reductions.
    /// </summary>
    [Module]
    public partial class AutoGradStructInstanceNormScaleBiasCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> s, Scalar<float32> t)
        {
            var grid = (Tensor<float32>)OnnxOp.Reshape(
                OnnxOp.Cast(OnnxOp.Range(Scalar(1L), Scalar(9L), Scalar(1L)), saturate: null, to: DType.Float32),
                Vector(1L, 2L, 2L, 2L), allowZero: false);
            var scale = (Tensor<float32>)OnnxOp.Expand(s, Vector(2L));
            var bias = (Tensor<float32>)OnnxOp.Expand(t, Vector(2L));
            var y = (Tensor<float32>)OnnxOp.InstanceNormalization(grid, scale, bias, epsilon: 1e-5f);
            var loss = (y * grid).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var (gradS, gradT) = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(s, t, loss);

            var expectedS = Scalar(10f) / (Scalar(1.25f) + Scalar(1e-5f)).Sqrt();
            var okS = (gradS! - expectedS).Abs() < Scalar(1e-3f);
            var okT = (gradT! - Scalar(36f)).Abs() < Scalar(1e-3f);
            return okS & okT;
        }
    }

    // ===================================================================
    //  Concat / Split: negative axis + partially-consumed Split outputs
    // ===================================================================

    /// <summary>
    /// Concat along a negative axis (-1) of differently-sized operands
    /// ([2,1] + [2,2] → [2,3]) with a position-weighted loss
    /// w = [[1,2,3],[4,5,6]]: dL/da = w[:,0] sum = 5, dL/db = 2+3+5+6 = 16.
    /// Exercises <c>ConcatGradient</c>'s Gather-on-shape split-size computation
    /// with a negative axis.
    /// </summary>
    [Module]
    public partial class AutoGradStructConcatNegAxisCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b)
        {
            var aMat = (Tensor<float32>)OnnxOp.Expand(a, Vector(2L, 1L));
            var bMat = (Tensor<float32>)OnnxOp.Expand(b, Vector(2L, 2L));
            var cat = (Tensor<float32>)OnnxOp.Concat([aMat, bMat], axis: -1);
            var w = (Tensor<float32>)OnnxOp.Reshape(
                OnnxOp.Cast(OnnxOp.Range(Scalar(1L), Scalar(7L), Scalar(1L)), saturate: null, to: DType.Float32),
                Vector(2L, 3L), allowZero: false);
            var loss = (cat * w).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var (gradA, gradB) = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, b, loss);
            var okA = (gradA! - Scalar(5f)).Abs() < Scalar(1e-3f);
            var okB = (gradB! - Scalar(16f)).Abs() < Scalar(1e-3f);
            return okA & okB;
        }
    }

    /// <summary>
    /// Split along a negative axis (-1). data = grid(1..8, [2,4])·a split into
    /// two [2,2] halves: loss = Σ s0 + 3·Σ s1 = (1+2+5+6)a + 3·(3+4+7+8)a = 80a
    /// → dL/da = 80. Exercises <c>SplitGradient</c>'s Concat with a negative axis.
    /// </summary>
    [Module]
    public partial class AutoGradStructSplitNegAxisCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var grid = (Tensor<float32>)OnnxOp.Reshape(
                OnnxOp.Cast(OnnxOp.Range(Scalar(1L), Scalar(9L), Scalar(1L)), saturate: null, to: DType.Float32),
                Vector(2L, 4L), allowZero: false);
            var data = grid * a;
            var splits = OnnxOp.Split(data, Vector(2L, 2L), axis: -1, numOutputs: null, variadicOutputCount: 2);
            var s0 = ((Tensor<float32>)splits[0]).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var s1 = ((Tensor<float32>)splits[1]).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var loss = s0 + Scalar(3f) * s1;
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(80f)).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>
    /// Split where only ONE of the two outputs reaches the loss. The gradient
    /// for the unused split must be a correctly-sized zero block — dropping it
    /// (the pre-fix behavior of <c>SplitGradient</c>) mis-shapes and mis-places
    /// the input gradient. x = [x0..x3], loss = 3·(x2+x3) → dL/dx = [0,0,3,3].
    /// </summary>
    [Module]
    public partial class AutoGradStructSplitPartialUseCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var splits = OnnxOp.Split(x, Vector(2L, 2L), axis: 0, numOutputs: null, variadicOutputCount: 2);
            var loss = ((Tensor<float32>)splits[1]).Reduce(ReduceKind.Sum, keepDims: false).Scalar() * Scalar(3f);
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);

            var zeros = (Tensor<float32>)OnnxOp.Expand(Scalar(0f), Vector(2L));
            var threes = (Tensor<float32>)OnnxOp.Expand(Scalar(3f), Vector(2L));
            var expected = (Tensor<float32>)OnnxOp.Concat([zeros, threes], axis: 0);
            var diff = (grad - expected).Abs();
            return diff.Reduce(ReduceKind.Max, keepDims: false).Scalar() < Scalar(1e-4f);
        }
    }

    // ===================================================================
    //  Variadic Sum/Mean/Max/Min with broadcasting + Dropout with ratio input
    // ===================================================================

    /// <summary>
    /// Variadic Sum with broadcasting ([2,2] + [2]): loss = Σ Sum(aMat, bVec) →
    /// dL/da = 4 and dL/db = 4 (each b element feeds 2 output cells). Exercises
    /// <c>SumGradient</c>'s ReverseBroadcast on mismatched operand shapes.
    /// </summary>
    [Module]
    public partial class AutoGradStructSumBroadcastCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b)
        {
            var aMat = (Tensor<float32>)OnnxOp.Expand(a, Vector(2L, 2L));
            var bVec = (Tensor<float32>)OnnxOp.Expand(b, Vector(2L));
            var summed = (Tensor<float32>)OnnxOp.Sum(aMat, bVec);
            var loss = summed.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var (gradA, gradB) = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, b, loss);
            return (gradA! - Scalar(4f)).Abs() < Scalar(1e-3f)
                & (gradB! - Scalar(4f)).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>
    /// Variadic Mean with broadcasting ([2,2], [2]): loss = Σ Mean → each
    /// gradient halves the Sum case: dL/da = 2, dL/db = 2.
    /// </summary>
    [Module]
    public partial class AutoGradStructMeanBroadcastCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b)
        {
            var aMat = (Tensor<float32>)OnnxOp.Expand(a, Vector(2L, 2L));
            var bVec = (Tensor<float32>)OnnxOp.Expand(b, Vector(2L));
            var meaned = (Tensor<float32>)OnnxOp.Mean(aMat, bVec);
            var loss = meaned.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var (gradA, gradB) = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, b, loss);
            return (gradA! - Scalar(2f)).Abs() < Scalar(1e-3f)
                & (gradB! - Scalar(2f)).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>
    /// Variadic Max with broadcasting (a=2 [2,2], b=5 [2]): b wins every cell,
    /// so dL/da = 0 and dL/db = 4. Exercises the mask + ReverseBroadcast path
    /// of <c>MaxGradient</c> on mismatched operand shapes.
    /// </summary>
    [Module]
    public partial class AutoGradStructMaxBroadcastCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b)
        {
            var aMat = (Tensor<float32>)OnnxOp.Expand(a, Vector(2L, 2L));
            var bVec = (Tensor<float32>)OnnxOp.Expand(b, Vector(2L));
            var maxed = (Tensor<float32>)OnnxOp.Max(aMat, bVec);
            var loss = maxed.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var (gradA, gradB) = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, b, loss);
            return gradA!.Abs() < Scalar(1e-3f)
                & (gradB! - Scalar(4f)).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>
    /// Variadic Min with broadcasting (a=7 [2,2], b=2 [2]): b wins every cell,
    /// so dL/da = 0 and dL/db = 4.
    /// </summary>
    [Module]
    public partial class AutoGradStructMinBroadcastCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b)
        {
            var aMat = (Tensor<float32>)OnnxOp.Expand(a, Vector(2L, 2L));
            var bVec = (Tensor<float32>)OnnxOp.Expand(b, Vector(2L));
            var minned = (Tensor<float32>)OnnxOp.Min(aMat, bVec);
            var loss = minned.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var (gradA, gradB) = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, b, loss);
            return gradA!.Abs() < Scalar(1e-3f)
                & (gradB! - Scalar(4f)).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>
    /// Dropout with explicit ratio and training_mode=false inputs: still an
    /// identity in inference mode, so dL/da = 3. Exercises the 3-input form of
    /// <c>DropoutGradient</c> (ratio/training_mode slots get null gradients).
    /// </summary>
    [Module]
    public partial class AutoGradStructDropoutRatioInputCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var aVec = (Tensor<float32>)OnnxOp.Expand(a, Vector(3L));
            var (output, _) = OnnxOp.Dropout(aVec, Scalar(0.5f), Scalar(false));
            var loss = ((Tensor<float32>)output).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(3f)).Abs() < Scalar(1e-3f);
        }
    }
}
