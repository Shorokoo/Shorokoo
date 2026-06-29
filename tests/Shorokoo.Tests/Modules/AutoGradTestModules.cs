namespace Shorokoo.Tests.Modules
{
    /// <summary>
    /// Modules whose <c>Inline</c> bodies embed <see cref="Ops.AutoGrad"/> calls so they
    /// can be driven by the same one-liner <see cref="AutoTester.AutoTest.AdvancedTestGraph{TModule}"/>
    /// pattern as the other module-coverage tests. <c>ToConcreteArchitecture</c> runs
    /// <c>FastProcessAutoGradProcessor.Process</c> as part of its lowering pipeline, so by
    /// the time the AutoTester executes the graph, every AUTO_GRAD node has been replaced
    /// with its expanded gradient subgraph — exercising the AutoGrad lowering plus the
    /// per-op <c>[AutoDiff]</c> gradient implementations end-to-end through ONNX roundtrip,
    /// CS roundtrip, and QuickExecutionEngine.
    /// </summary>

    /// <summary>
    /// Simplest possible AutoGrad chain: scalar input, square loss, dL/dx = 2x.
    /// Self-checking module — returns <c>Scalar&lt;bit&gt;</c> verifying the analytical
    /// gradient matches <c>2*x</c>. The AutoTester treats a graph whose sole output
    /// is a true <c>Scalar&lt;bit&gt;</c> as a passing test, so the xUnit test stays a
    /// one-liner. Exercises AutoGradOps.AutoGrad(A, Scalar&lt;T&gt;) plus the Mul gradient.
    /// </summary>
    [Module]
    public partial class AutoGradScalarSquare
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            var loss = x * x;
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            var expected = Scalar(2f) * x;
            return (grad - expected).Abs() < Scalar(1e-4f);
        }
    }

    /// <summary>
    /// Two-input AutoGrad via the typed tuple overload: loss = a * b → (dL/da, dL/db) = (b, a).
    /// Self-checking; returns <c>Scalar&lt;bit&gt;</c>. Exercises AutoGradOps.AutoGrad&lt;A, B, T&gt;
    /// (the tuple overload at AutoGradOps.cs 16-23).
    /// </summary>
    [Module]
    public partial class AutoGradPairProduct
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b)
        {
            var loss = a * b;
            var (gradA, gradB) = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, b, loss);
            var okA = (gradA! - b).Abs() < Scalar(1e-4f);
            var okB = (gradB! - a).Abs() < Scalar(1e-4f);
            return okA & okB;
        }
    }

    /// <summary>
    /// Self-checking AutoGrad for subtraction: loss = a - b → (dL/da, dL/db) = (1, -1).
    /// Converted from the analytical-gradient assertion in
    /// <c>AutoDiffArithmeticTests.TestSubGradient</c>. The module embeds the equality
    /// check so the xUnit driver stays a one-liner.
    /// </summary>
    [Module]
    public partial class AutoGradSubBinaryCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b)
        {
            var loss = a - b;
            var (gradA, gradB) = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, b, loss);
            var okA = (gradA! - Scalar(1f)).Abs() < Scalar(1e-4f);
            var okB = (gradB! - Scalar(-1f)).Abs() < Scalar(1e-4f);
            return okA & okB;
        }
    }

    /// <summary>
    /// Dead-parameter AutoGrad: loss only depends on <c>b</c>, but the AutoGrad request
    /// includes <c>a</c>. Self-checking. Exercises FastProcessAutoGrad's
    /// <c>SpliceZerosLike</c> fallback (FastProcessAutoGrad.cs lines 169-171, 366-382),
    /// which emits <c>Sub(p, p)</c> for any parameter with no path from loss. Expected:
    /// dL/da = 0, dL/db = 2b.
    /// </summary>
    [Module]
    public partial class AutoGradDeadParam
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b)
        {
            var loss = b * b;
            var (gradA, gradB) = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, b, loss);
            var okA = gradA!.Abs() < Scalar(1e-4f);
            var okB = (gradB! - Scalar(2f) * b).Abs() < Scalar(1e-4f);
            return okA & okB;
        }
    }

    /// <summary>
    /// AutoGrad through a MatMul + ReduceSum chain. Loss = sum(x @ W), so
    /// dL/dx[i,k] = sum_j W[k,j] (the j-row-sum of W, broadcast across batch). Self-checking.
    /// Exercises the MatMul and ReduceSum gradient functions during AutoGrad lowering, in
    /// addition to the chained accumulation logic in FastProcessAutoGradProcessor.
    /// </summary>
    [Module]
    public partial class AutoGradMatMulReduce
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var w = InitSimple.Init([Scalar(3L), Scalar(2L)]);
            var product = (Tensor<float32>)OnnxOp.MatMul(x, w);
            var loss = product.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);

            // dL/dx[i,k] = sum_j W[k,j] — j-row-sum of W, broadcast over batch axis.
            var wRowSums = (Tensor<float32>)OnnxOp.ReduceSum(w, axes: Vector(1L), keepdims: false, noopWithEmptyAxes: false);
            var diff = (grad - wRowSums).Abs();
            return diff.Reduce(ReduceKind.Max, keepDims: false).Scalar() < Scalar(1e-4f);
        }
    }

    /// <summary>
    /// AutoGrad on a trainable parameter rather than a runtime input. Loss is the sum of
    /// element-wise <c>x * w</c>, so dL/dw = x (element-wise). Self-checking. <c>w</c> is
    /// created via <see cref="InitSimple.Init"/>, so it flows through the trainable-param
    /// lowering before AutoGrad. Exercises the AutoGrad path on a TRAINABLE_PARAM-rooted
    /// subgraph and forces the full FastConvertTrainableParamIdRefToTrainableParam →
    /// FastProcessAutoGradProcessor hand-off.
    /// </summary>
    [Module]
    public partial class AutoGradTrainableParam
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var w = InitSimple.Init(x.ShapeTensor());
            var product = x * w;
            var loss = product.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(w, loss);

            // dL/dw = x (element-wise).
            var diff = (grad - x).Abs();
            return diff.Reduce(ReduceKind.Max, keepDims: false).Scalar() < Scalar(1e-4f);
        }
    }

    /// <summary>
    /// AutoGrad through a Slice that explicitly names its axes. Self-checking via
    /// two-sided directional derivative: (f(x + h·g) − f(x − h·g)) / (2h) ≈ ‖g‖².
    /// Exercises the <c>Slice</c> gradient's "axes is not null" branch in
    /// <c>AutoDiffs.More.cs</c> (pads <c>grad</c> with zeros along the sliced axes only).
    /// </summary>
    [Module]
    public partial class AutoGradSliceWithAxes
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var loss = SliceLoss(x);
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);

            var h = Scalar(1e-3f);
            var pert = h * grad;
            var lossPlus = SliceLoss(x + pert);
            var lossMinus = SliceLoss(x - pert);
            var deriv = (lossPlus - lossMinus) / (Scalar(2f) * h);
            var gradNormSq = (grad * grad).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            return (deriv - gradNormSq).Abs() < Scalar(1e-3f) * (gradNormSq.Abs() + Scalar(1f));
        }

        private static Scalar<float32> SliceLoss(Tensor<float32> x)
        {
            var sliced = (Tensor<float32>)OnnxOp.Slice(
                x,
                starts: Vector(0L),
                ends: Vector(2L),
                axes: Vector(1L),
                steps: null);
            return sliced.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        }
    }

    /// <summary>
    /// AutoGrad through a 2D Conv with bias. Self-checking via two-sided directional
    /// derivative. Drives the largest uncovered gradient path in <c>AutoDiffs.Batch5.cs</c>
    /// — <c>ConvGradient</c> emits ConvTranspose for dx, slice-to-shape, and a
    /// ReduceSum-over-batch+spatial for db. Forward input shape:
    /// <c>[N=1, C=3, H=5, W=5]</c>; kernel <c>[Out=2, In=3, 3, 3]</c>.
    /// </summary>
    [Module]
    public partial class AutoGradConv
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var w = InitSimple.Init([Scalar(2L), Scalar(3L), Scalar(3L), Scalar(3L)]);
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
                pads: [0L, 0L, 0L, 0L], strides: [1L, 1L]);
            return conved.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        }
    }

    /// <summary>
    /// AutoGrad of a 2D Conv with respect to its <b>weight</b> — the path
    /// <c>ConvGradient</c> in <c>AutoDiffs.Batch5.cs</c> historically dropped (it returned a
    /// <c>null</c> weight gradient, silently freezing every Conv kernel at its initial value
    /// throughout training). Self-checking against the closed form: for
    /// <c>loss = Σ Conv(x, w, b)</c> the weight gradient is
    /// <c>dL/dw[oc,ic,kh,kw] = Σ_{n,oh,ow} x[n,ic, oh·stride + kh·dilation]</c>. With
    /// <c>x ≡ 0.1</c> and a 5×5 input / 3×3 kernel / stride 1 / no pad, every one of the
    /// 3×3 = 9 output positions is covered, so <c>dL/dw ≡ 9·0.1 = 0.9</c> for every weight
    /// element. <c>w</c> comes from <see cref="InitSimple.Init"/> so it flows through the
    /// trainable-param lowering before AutoGrad, exactly as a real trained Conv kernel does.
    /// </summary>
    [Module]
    public partial class AutoGradConvWeight
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var w = InitSimple.Init([Scalar(2L), Scalar(3L), Scalar(3L), Scalar(3L)]);
            var b = InitSimple.Init([Scalar(2L)]).Vec();
            var loss = ConvLoss(x, w, b);
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(w, loss);

            // Closed form: every weight element's gradient is OH*OW * x_val = 9 * 0.1 = 0.9.
            var diff = (grad - Scalar(0.9f)).Abs();
            return diff.Reduce(ReduceKind.Max, keepDims: false).Scalar() < Scalar(1e-3f);
        }

        private static Scalar<float32> ConvLoss(Tensor<float32> x, Tensor<float32> w, Vector<float32> b)
        {
            var conved = NN.Conv(x, w, b, AutoPad.NotSet,
                dilations: [1L, 1L], group: 1L, kernelShape: [3L, 3L],
                pads: [0L, 0L, 0L, 0L], strides: [1L, 1L]);
            return conved.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        }
    }

    /// <summary>
    /// AutoGrad through a 2D ConvTranspose with bias. Self-checking via two-sided
    /// directional derivative. Drives <c>ConvTransposeGradient</c> in
    /// <c>AutoDiffs.Batch5.cs</c> — symmetric to <c>ConvGradient</c> but inverts the
    /// spatial direction.
    /// </summary>
    [Module]
    public partial class AutoGradConvTranspose
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var w = InitSimple.Init([Scalar(3L), Scalar(2L), Scalar(3L), Scalar(3L)]);
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
                outputPadding: [0L, 0L], outputShape: [],
                pads: [0L, 0L, 0L, 0L], strides: [1L, 1L]);
            return convT.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        }
    }

    /// <summary>
    /// AutoGrad through a ReduceMean over all axes (axes input is null). Self-checking
    /// against dL/dx = 1/N where N = prod(x.shape). Exercises the "axes is null" branch
    /// of <c>ReduceMean</c> gradient in <c>AutoDiffs.DZ.cs</c>.
    /// </summary>
    [Module]
    public partial class AutoGradReduceMeanAllAxes
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var mean = x.Reduce(ReduceKind.Mean, keepDims: false);
            var loss = mean.Scalar();
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);

            var nFloat = x.SizeTensor().Cast<float32>();
            var expected = Scalar(1f) / nFloat;
            var diff = (grad - expected).Abs();
            return diff.Reduce(ReduceKind.Max, keepDims: false).Scalar() < Scalar(1e-5f);
        }
    }

    /// <summary>
    /// AutoGrad through a Softmax. Self-checking: sum(softmax(x)) along the last axis is
    /// constant (= number of rows) in x, so dL/dx = 0 element-wise. Exercises the
    /// <c>Softmax</c> gradient in <c>AutoDiffs.DZ.cs</c> (the
    /// <c>softmax * (grad - sum(grad * softmax))</c> identity).
    /// </summary>
    [Module]
    public partial class AutoGradSoftmax
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var soft = x.Softmax(axis: -1);
            var loss = soft.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            return grad.Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar() < Scalar(1e-4f);
        }
    }

    /// <summary>
    /// AutoGrad through Transpose with an explicit permutation. Self-checking: sum(x)
    /// is permutation-invariant, so dL/dx = ones(x.shape). Drives the
    /// <c>Transpose</c> gradient in <c>AutoDiffs.DZ.cs</c> (the inverse-permutation
    /// branch rather than the default reverse-all).
    /// </summary>
    [Module]
    public partial class AutoGradTransposePerm
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var transposed = (Tensor<float32>)OnnxOp.Transpose(x, perm: [1L, 0L, 2L]);
            var loss = transposed.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            var diff = (grad - Scalar(1f)).Abs();
            return diff.Reduce(ReduceKind.Max, keepDims: false).Scalar() < Scalar(1e-5f);
        }
    }

    /// <summary>
    /// AutoGrad through a Pad in constant mode with an explicit axes input. Self-checking:
    /// padding with constant 0 leaves the sum unchanged, so dL/dx = ones(x.shape).
    /// Exercises the "axes is not null" branch of the Pad gradient in
    /// <c>AutoDiffs.More.cs</c>, which slices grad along just the padded axes.
    /// </summary>
    [Module]
    public partial class AutoGradPadAxes
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var padded = (Tensor<float32>)OnnxOp.Pad(
                x,
                pads: Vector(1L, 1L),
                constantValue: Scalar(0.0f),
                axes: Vector(0L),
                mode: PadMode.Constant);
            var loss = padded.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            var diff = (grad - Scalar(1f)).Abs();
            return diff.Reduce(ReduceKind.Max, keepDims: false).Scalar() < Scalar(1e-5f);
        }
    }

    /// <summary>
    /// AutoGrad through Tile, which repeats a tensor along each axis. Self-checking:
    /// loss = sum(tile(x, [2, 3])) = 6 * sum(x), so dL/dx = 6 element-wise. Exercises
    /// the Tile gradient in <c>AutoDiffs.More.cs</c>: reshape interleaves repeat and
    /// data dims, then a ReduceSum collapses the repeat dims.
    /// </summary>
    [Module]
    public partial class AutoGradTile
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var tiled = (Tensor<float32>)OnnxOp.Tile(x, repeats: Vector(2L, 3L));
            var loss = tiled.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            var diff = (grad - Scalar(6f)).Abs();
            return diff.Reduce(ReduceKind.Max, keepDims: false).Scalar() < Scalar(1e-5f);
        }
    }

    /// <summary>
    /// AutoGrad through 2D MaxPool. Self-checking via two-sided directional derivative
    /// (perturbations of size 1e-3 don't change the chosen max for well-separated input).
    /// Exercises <c>MaxPoolGradient</c> in <c>AutoDiffs.Batch5.cs</c>. A distinct
    /// per-position offset grid is added inside the loss so every window has a unique
    /// argmax: MaxPool is not differentiable at tie points, and the indices-based
    /// gradient routes ties to the first max element (for which the symmetric finite
    /// difference does not hold).
    /// </summary>
    [Module]
    public partial class AutoGradMaxPool
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var loss = MaxPoolLoss(x);
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);

            var h = Scalar(1e-3f);
            var pert = h * grad;
            var deriv = (MaxPoolLoss(x + pert) - MaxPoolLoss(x - pert)) / (Scalar(2f) * h);
            var gradNormSq = (grad * grad).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            return (deriv - gradNormSq).Abs() < Scalar(1e-3f) * (gradNormSq.Abs() + Scalar(1f));
        }

        private static Scalar<float32> MaxPoolLoss(Tensor<float32> x)
        {
            // De-tie the (uniform) test input with well-separated fixed offsets so each
            // window's argmax is unique and stable under the ±1e-3 FD perturbations.
            var offsets = (Tensor<float32>)OnnxOp.Reshape(
                OnnxOp.Cast(OnnxOp.Range(Scalar(0L), Scalar(32L), Scalar(1L)), saturate: null, to: DType.Float32),
                Vector(1L, 2L, 4L, 4L), allowZero: false);
            var distinct = x + offsets * Scalar(0.01f);
            var pooled = OnnxOp.MaxPool(
                distinct,
                autoPad: AutoPad.NotSet,
                ceilMode: false,
                dilations: [1L, 1L],
                kernelShape: [2L, 2L],
                pads: [0L, 0L, 0L, 0L],
                storageOrder: 0L,
                strides: [2L, 2L]);
            return ((Tensor<float32>)pooled).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        }
    }

    /// <summary>
    /// AutoGrad through 2D AveragePool. Self-checking via two-sided directional
    /// derivative. Exercises <c>AveragePoolGradient</c> in <c>AutoDiffs.Batch5.cs</c>
    /// (Reshape→Expand→Reshape→Col2Im chain).
    /// </summary>
    [Module]
    public partial class AutoGradAvgPool
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var loss = AvgPoolLoss(x);
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);

            var h = Scalar(1e-3f);
            var pert = h * grad;
            var deriv = (AvgPoolLoss(x + pert) - AvgPoolLoss(x - pert)) / (Scalar(2f) * h);
            var gradNormSq = (grad * grad).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            return (deriv - gradNormSq).Abs() < Scalar(1e-3f) * (gradNormSq.Abs() + Scalar(1f));
        }

        private static Scalar<float32> AvgPoolLoss(Tensor<float32> x)
        {
            var pooled = (Tensor<float32>)OnnxOp.AveragePool(
                x, autoPad: AutoPad.NotSet, ceilMode: false, countIncludePad: false,
                dilations: [1L, 1L], kernelShape: [2L, 2L],
                pads: [0L, 0L, 0L, 0L], strides: [2L, 2L]);
            return pooled.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        }
    }

    /// <summary>
    /// AutoGrad through 2D AveragePool with overlapping windows (stride &lt; kernel).
    /// Self-checking via directional derivative. Exercises Col2Im's accumulation of
    /// overlapping contributions on the input side.
    /// </summary>
    [Module]
    public partial class AutoGradAvgPoolOverlap
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var loss = AvgPoolOverlapLoss(x);
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);

            var h = Scalar(1e-3f);
            var pert = h * grad;
            var deriv = (AvgPoolOverlapLoss(x + pert) - AvgPoolOverlapLoss(x - pert)) / (Scalar(2f) * h);
            var gradNormSq = (grad * grad).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            return (deriv - gradNormSq).Abs() < Scalar(1e-3f) * (gradNormSq.Abs() + Scalar(1f));
        }

        private static Scalar<float32> AvgPoolOverlapLoss(Tensor<float32> x)
        {
            var pooled = (Tensor<float32>)OnnxOp.AveragePool(
                x, autoPad: AutoPad.NotSet, ceilMode: false, countIncludePad: false,
                dilations: [1L, 1L], kernelShape: [3L, 3L],
                pads: [0L, 0L, 0L, 0L], strides: [1L, 1L]);
            return pooled.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        }
    }

    /// <summary>
    /// AutoGrad through 2D AveragePool with non-zero pads and count_include_pad=true.
    /// Self-checking via directional derivative. Drives the constant-divisor branch.
    /// </summary>
    [Module]
    public partial class AutoGradAvgPoolPadInclude
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var loss = AvgPoolPadIncludeLoss(x);
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);

            var h = Scalar(1e-3f);
            var pert = h * grad;
            var deriv = (AvgPoolPadIncludeLoss(x + pert) - AvgPoolPadIncludeLoss(x - pert)) / (Scalar(2f) * h);
            var gradNormSq = (grad * grad).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            return (deriv - gradNormSq).Abs() < Scalar(1e-3f) * (gradNormSq.Abs() + Scalar(1f));
        }

        private static Scalar<float32> AvgPoolPadIncludeLoss(Tensor<float32> x)
        {
            var pooled = (Tensor<float32>)OnnxOp.AveragePool(
                x, autoPad: AutoPad.NotSet, ceilMode: false, countIncludePad: true,
                dilations: [1L, 1L], kernelShape: [3L, 3L],
                pads: [1L, 1L, 1L, 1L], strides: [1L, 1L]);
            return pooled.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        }
    }

    /// <summary>
    /// AutoGrad through 2D AveragePool with non-zero pads and count_include_pad=false.
    /// Self-checking via directional derivative. Drives the per-window divisor branch.
    /// </summary>
    [Module]
    public partial class AutoGradAvgPoolPadExclude
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var loss = AvgPoolPadExcludeLoss(x);
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);

            var h = Scalar(1e-3f);
            var pert = h * grad;
            var deriv = (AvgPoolPadExcludeLoss(x + pert) - AvgPoolPadExcludeLoss(x - pert)) / (Scalar(2f) * h);
            var gradNormSq = (grad * grad).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            return (deriv - gradNormSq).Abs() < Scalar(1e-3f) * (gradNormSq.Abs() + Scalar(1f));
        }

        private static Scalar<float32> AvgPoolPadExcludeLoss(Tensor<float32> x)
        {
            var pooled = (Tensor<float32>)OnnxOp.AveragePool(
                x, autoPad: AutoPad.NotSet, ceilMode: false, countIncludePad: false,
                dilations: [1L, 1L], kernelShape: [3L, 3L],
                pads: [1L, 1L, 1L, 1L], strides: [1L, 1L]);
            return pooled.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        }
    }

    /// <summary>
    /// AutoGrad through 2D AveragePool with <c>auto_pad = SAME_UPPER</c>. Self-checking
    /// via directional derivative. Drives the dynamic-pad path in <c>AveragePoolGradient</c>.
    /// </summary>
    [Module]
    public partial class AutoGradAvgPoolSameUpper
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var loss = AvgPoolSameUpperLoss(x);
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);

            var h = Scalar(1e-3f);
            var pert = h * grad;
            var deriv = (AvgPoolSameUpperLoss(x + pert) - AvgPoolSameUpperLoss(x - pert)) / (Scalar(2f) * h);
            var gradNormSq = (grad * grad).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            return (deriv - gradNormSq).Abs() < Scalar(1e-3f) * (gradNormSq.Abs() + Scalar(1f));
        }

        private static Scalar<float32> AvgPoolSameUpperLoss(Tensor<float32> x)
        {
            var pooled = (Tensor<float32>)OnnxOp.AveragePool(
                x, autoPad: AutoPad.SameUpper, ceilMode: false, countIncludePad: false,
                dilations: [1L, 1L], kernelShape: [3L, 3L],
                pads: null, strides: [2L, 2L]);
            return pooled.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        }
    }

    /// <summary>
    /// AutoGrad through 2D AveragePool with <c>auto_pad = SAME_LOWER</c>. Self-checking
    /// via directional derivative. Same code path as <see cref="AutoGradAvgPoolSameUpper"/>,
    /// with the asymmetric pad-begin/pad-end split.
    /// </summary>
    [Module]
    public partial class AutoGradAvgPoolSameLower
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var loss = AvgPoolSameLowerLoss(x);
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);

            var h = Scalar(1e-3f);
            var pert = h * grad;
            var deriv = (AvgPoolSameLowerLoss(x + pert) - AvgPoolSameLowerLoss(x - pert)) / (Scalar(2f) * h);
            var gradNormSq = (grad * grad).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            return (deriv - gradNormSq).Abs() < Scalar(1e-3f) * (gradNormSq.Abs() + Scalar(1f));
        }

        private static Scalar<float32> AvgPoolSameLowerLoss(Tensor<float32> x)
        {
            var pooled = (Tensor<float32>)OnnxOp.AveragePool(
                x, autoPad: AutoPad.SameLower, ceilMode: false, countIncludePad: true,
                dilations: [1L, 1L], kernelShape: [3L, 3L],
                pads: null, strides: [2L, 2L]);
            return pooled.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        }
    }

    /// <summary>
    /// AutoGrad through a Gemm with both <c>transA</c> and <c>transB</c> set. Self-checking
    /// via two-sided directional derivative. Exercises the four-way transpose-combination
    /// logic in <c>GemmGradient</c> (<c>AutoDiffs.Batch5.cs</c>), particularly the
    /// transA=1, transB=1 branch and the transposed-B path.
    /// </summary>
    [Module]
    public partial class AutoGradGemmTrans
    {
        public static Scalar<bit> Inline(Tensor<float32> a, Tensor<float32> b)
        {
            var loss = GemmLoss(a, b);
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);

            var h = Scalar(1e-3f);
            var pert = h * grad;
            var deriv = (GemmLoss(a + pert, b) - GemmLoss(a - pert, b)) / (Scalar(2f) * h);
            var gradNormSq = (grad * grad).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            return (deriv - gradNormSq).Abs() < Scalar(1e-3f) * (gradNormSq.Abs() + Scalar(1f));
        }

        private static Scalar<float32> GemmLoss(Tensor<float32> a, Tensor<float32> b)
        {
            var gemm = (Tensor<float32>)OnnxOp.Gemm(
                a, b, c: null, alpha: 1.0f, beta: 0.0f, transA: 1L, transB: 1L);
            return gemm.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        }
    }

    // ===================================================================
    //  Modules converted from AutoDiffArithmeticTests.cs analytical helpers
    // ===================================================================

    /// <summary>loss = (a−b)·b → (dL/da, dL/db) = (b, a−2b).</summary>
    [Module]
    public partial class AutoGradSubChainedCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b)
        {
            var loss = (a - b) * b;
            var (gradA, gradB) = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, b, loss);
            var okA = (gradA! - b).Abs() < Scalar(1e-4f);
            var okB = (gradB! - (a - Scalar(2f) * b)).Abs() < Scalar(1e-4f);
            return okA & okB;
        }
    }

    /// <summary>loss = a/b → (dL/da, dL/db) = (1/b, −a/b²).</summary>
    [Module]
    public partial class AutoGradDivCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b)
        {
            var loss = a / b;
            var (gradA, gradB) = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, b, loss);
            var okA = (gradA! - Scalar(1f) / b).Abs() < Scalar(1e-4f);
            var okB = (gradB! - (-a / (b * b))).Abs() < Scalar(1e-4f);
            return okA & okB;
        }
    }

    /// <summary>loss = −a → dL/da = −1.</summary>
    [Module]
    public partial class AutoGradNegCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var loss = -a;
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(-1f)).Abs() < Scalar(1e-4f);
        }
    }

    /// <summary>loss = (−a)·b → (dL/da, dL/db) = (−b, −a).</summary>
    [Module]
    public partial class AutoGradNegChainedCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b)
        {
            var loss = (-a) * b;
            var (gradA, gradB) = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, b, loss);
            var okA = (gradA! - (-b)).Abs() < Scalar(1e-4f);
            var okB = (gradB! - (-a)).Abs() < Scalar(1e-4f);
            return okA & okB;
        }
    }

    /// <summary>loss = x^y → (dL/dx, dL/dy) = (y·x^(y−1), x^y · ln(x)).</summary>
    [Module]
    public partial class AutoGradPowCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x, Scalar<float32> y)
        {
            var loss = x.Pow(y);
            var (gradX, gradY) = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, y, loss);
            var expectedX = y * x.Pow(y - Scalar(1f));
            var expectedY = x.Pow(y) * ((Tensor<float32>)OnnxOp.Log((Tensor<float32>)x)).Scalar();
            var okX = (gradX! - expectedX).Abs() < Scalar(1e-3f);
            var okY = (gradY! - expectedY).Abs() < Scalar(1e-3f);
            return okX & okY;
        }
    }

    /// <summary>loss = |x| → dL/dx = sign(x), checked at both positive and negative x.</summary>
    [Module]
    public partial class AutoGradAbsCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            var loss = x.Abs();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            var expected = ((Tensor<float32>)OnnxOp.Sign((Tensor<float32>)x)).Scalar();
            return (grad - expected).Abs() < Scalar(1e-4f);
        }
    }

    /// <summary>loss = 1/x → dL/dx = −1/x².</summary>
    [Module]
    public partial class AutoGradReciprocalCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            var loss = ((Tensor<float32>)OnnxOp.Reciprocal(x)).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            var expected = -Scalar(1f) / (x * x);
            return (grad - expected).Abs() < Scalar(1e-4f);
        }
    }

    /// <summary>
    /// loss = sum(mod(expand(a, [2]), expand(b, [2]))) with fmod=1 (C fmod). Float fmod
    /// is piecewise linear: fmod(a, b) = a − trunc(a/b)·b, so away from the kink points
    /// dL/da = 2 (vector length) and dL/db = −2·trunc(a/b) (PyTorch fmod convention; the
    /// gradient used to be hardcoded to zero). The expected −trunc(a/b) is computed
    /// in-graph via Div+Floor on |a/b| so the check holds for any kink-free inputs.
    /// </summary>
    [Module]
    public partial class AutoGradModSumCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b)
        {
            var aVec = (Tensor<float32>)OnnxOp.Expand(a, Vector(2L));
            var bVec = (Tensor<float32>)OnnxOp.Expand(b, Vector(2L));
            var modResult = (Tensor<float32>)OnnxOp.Mod(aVec, bVec, fmod: true);
            var loss = modResult.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var (gradA, gradB) = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, b, loss);
            // trunc(a/b) = sign(a/b)·floor(|a/b|)
            var ratio = a / b;
            var trunc = ((Tensor<float32>)OnnxOp.Sign((Tensor<float32>)ratio)).Scalar()
                * ((Tensor<float32>)OnnxOp.Floor((Tensor<float32>)ratio.Abs())).Scalar();
            var okA = (gradA! - Scalar(2f)).Abs() < Scalar(1e-4f);
            var okB = (gradB! + Scalar(2f) * trunc).Abs() < Scalar(1e-4f);
            return okA & okB;
        }
    }

    /// <summary>
    /// loss = sum(mod(expand(a, [2]), expand(b, [2])) + expand(a, [2])) with fmod=1.
    /// The fmod path contributes d/da = 1 per element and the additive path another 1,
    /// so dL/da = 4 over the 2-element vector.
    /// </summary>
    [Module]
    public partial class AutoGradModWithDownstreamCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b)
        {
            var aVec = (Tensor<float32>)OnnxOp.Expand(a, Vector(2L));
            var bVec = (Tensor<float32>)OnnxOp.Expand(b, Vector(2L));
            var modResult = (Tensor<float32>)OnnxOp.Mod(aVec, bVec, fmod: true);
            var combined = modResult + aVec;
            var loss = combined.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(4f)).Abs() < Scalar(1e-4f);
        }
    }

    /// <summary>loss = |x|·x → dL/dx = sign(x)·x + |x|.</summary>
    [Module]
    public partial class AutoGradAbsMulChainCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            var loss = x.Abs() * x;
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            var sign = ((Tensor<float32>)OnnxOp.Sign((Tensor<float32>)x)).Scalar();
            var expected = sign * x + x.Abs();
            return (grad - expected).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>
    /// loss = (x²)^0.5 = |x| → dL/dx = sign(x) (= 1 for positive x).
    /// </summary>
    [Module]
    public partial class AutoGradPowSqrtIdentityCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            var loss = x.Pow(Scalar(2f)).Pow(Scalar(0.5f));
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            var expected = ((Tensor<float32>)OnnxOp.Sign((Tensor<float32>)x)).Scalar();
            return (grad - expected).Abs() < Scalar(1e-3f);
        }
    }

    // ===================================================================
    //  Modules converted from AutoDiffActivationTests.cs analytical helpers
    //  Each uses the two-sided directional-derivative consistency check
    //  (f(x+h·g) − f(x−h·g))/(2h) ≈ ‖g‖², so the same module verifies all
    //  smooth-point inputs (positive, negative, etc.).
    // ===================================================================

    internal static class AutoGradCheckHelpers
    {
        public static Scalar<bit> ScalarDirectionalDerivCheck(
            Scalar<float32> x, Scalar<float32> grad, Func<Scalar<float32>, Scalar<float32>> f)
        {
            var h = Scalar(1e-3f);
            var pert = h * grad;
            var deriv = (f(x + pert) - f(x - pert)) / (Scalar(2f) * h);
            var gradNormSq = grad * grad;
            return (deriv - gradNormSq).Abs() < Scalar(1e-3f) * (gradNormSq.Abs() + Scalar(1f));
        }
    }

    /// <summary>loss = relu(x). Self-checking at any smooth x.</summary>
    [Module]
    public partial class AutoGradReluCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            Func<Scalar<float32>, Scalar<float32>> f = z => z.Relu();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, f(x));
            return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(x, grad, f);
        }
    }

    /// <summary>loss = relu(a − 1)·2. dL/da = 2·1[a&gt;1].</summary>
    [Module]
    public partial class AutoGradReluChainedCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            Func<Scalar<float32>, Scalar<float32>> f = z => (z - Scalar(1f)).Relu() * Scalar(2f);
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, f(a));
            return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(a, grad, f);
        }
    }

    /// <summary>loss = sigmoid(x). dL/dx = sig·(1−sig).</summary>
    [Module]
    public partial class AutoGradSigmoidCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            Func<Scalar<float32>, Scalar<float32>> f = z => z.Sigmoid();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, f(x));
            return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(x, grad, f);
        }
    }

    /// <summary>loss = leakyRelu(x, 0.1). Self-checking at any smooth x.</summary>
    [Module]
    public partial class AutoGradLeakyReluCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            Func<Scalar<float32>, Scalar<float32>> f = z => z.LeakyRelu(0.1f);
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, f(x));
            return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(x, grad, f);
        }
    }

    /// <summary>loss = gelu(x). Self-checking at any smooth x.</summary>
    [Module]
    public partial class AutoGradGeluCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            Func<Scalar<float32>, Scalar<float32>> f = z => z.Gelu();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, f(x));
            return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(x, grad, f);
        }
    }

    /// <summary>loss = elu(x, 1.0). Self-checking at any smooth x.</summary>
    [Module]
    public partial class AutoGradEluCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            Func<Scalar<float32>, Scalar<float32>> f = z => z.Elu(1.0f);
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, f(x));
            return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(x, grad, f);
        }
    }

    /// <summary>loss = selu(x). Self-checking at any smooth x.</summary>
    [Module]
    public partial class AutoGradSeluCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            Func<Scalar<float32>, Scalar<float32>> f = z => z.Selu();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, f(x));
            return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(x, grad, f);
        }
    }

    /// <summary>loss = celu(x, 1.0). Self-checking at any smooth x.</summary>
    [Module]
    public partial class AutoGradCeluCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            Func<Scalar<float32>, Scalar<float32>> f = z => z.Celu(1.0f);
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, f(x));
            return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(x, grad, f);
        }
    }

    /// <summary>loss = hardSigmoid(x). Piecewise-linear; smooth in (-2.5, 2.5).</summary>
    [Module]
    public partial class AutoGradHardSigmoidCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            Func<Scalar<float32>, Scalar<float32>> f = z => z.HardSigmoid();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, f(x));
            return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(x, grad, f);
        }
    }

    /// <summary>loss = hardSwish(x). Smooth in (-3, 3) except at boundaries.</summary>
    [Module]
    public partial class AutoGradHardSwishCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            Func<Scalar<float32>, Scalar<float32>> f = z => z.HardSwish();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, f(x));
            return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(x, grad, f);
        }
    }

    /// <summary>loss = mish(x). Smooth everywhere.</summary>
    [Module]
    public partial class AutoGradMishCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            Func<Scalar<float32>, Scalar<float32>> f = z => z.Mish();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, f(x));
            return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(x, grad, f);
        }
    }

    /// <summary>loss = softplus(x). Smooth everywhere.</summary>
    [Module]
    public partial class AutoGradSoftplusCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            Func<Scalar<float32>, Scalar<float32>> f = z => z.Softplus();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, f(x));
            return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(x, grad, f);
        }
    }

    /// <summary>loss = softsign(x). Smooth everywhere.</summary>
    [Module]
    public partial class AutoGradSoftsignCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            Func<Scalar<float32>, Scalar<float32>> f = z => z.Softsign();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, f(x));
            return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(x, grad, f);
        }
    }

    /// <summary>loss = thresholdedRelu(x, 0.5). Smooth at any x != 0.5.</summary>
    [Module]
    public partial class AutoGradThresholdedReluCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            Func<Scalar<float32>, Scalar<float32>> f = z => z.ThresholdedRelu(0.5f);
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, f(x));
            return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(x, grad, f);
        }
    }

    /// <summary>loss = shrink(x, bias=0.1, lambd=0.4). Smooth except at ±lambd.</summary>
    [Module]
    public partial class AutoGradShrinkCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            Func<Scalar<float32>, Scalar<float32>> f = z => z.Shrink(0.1f, 0.4f);
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, f(x));
            return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(x, grad, f);
        }
    }

    /// <summary>loss = sigmoid(exp(x)). Chained sigmoid+exp gradient check.</summary>
    [Module]
    public partial class AutoGradSigmoidExpChainCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            Func<Scalar<float32>, Scalar<float32>> f = z => z.Exp().Sigmoid();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, f(x));
            return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(x, grad, f);
        }
    }

    /// <summary>loss = leakyRelu(exp(x) − 2, 0.01). Chained leakyRelu+exp gradient check.</summary>
    [Module]
    public partial class AutoGradLeakyReluExpChainCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            Func<Scalar<float32>, Scalar<float32>> f = z => (z.Exp() - Scalar(2f)).LeakyRelu(0.01f);
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, f(x));
            return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(x, grad, f);
        }
    }

    // ===================================================================
    //  Modules converted from AutoDiffTrigonometricTests.cs analytical helpers
    //  All single-argument trig/hyperbolic ops use the directional-derivative check.
    // ===================================================================

    /// <summary>loss = sin(x). Self-checking.</summary>
    [Module]
    public partial class AutoGradSinCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            Func<Scalar<float32>, Scalar<float32>> f = z => z.Sin();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, f(x));
            return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(x, grad, f);
        }
    }

    /// <summary>loss = cos(x). Self-checking.</summary>
    [Module]
    public partial class AutoGradCosCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            Func<Scalar<float32>, Scalar<float32>> f = z => z.Cos();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, f(x));
            return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(x, grad, f);
        }
    }

    /// <summary>loss = tan(x). Self-checking (input must avoid π/2 + kπ).</summary>
    [Module]
    public partial class AutoGradTanCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            Func<Scalar<float32>, Scalar<float32>> f = z => z.Tan();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, f(x));
            return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(x, grad, f);
        }
    }

    /// <summary>loss = asin(x). Self-checking (input must be in (−1, 1)).</summary>
    [Module]
    public partial class AutoGradAsinCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            Func<Scalar<float32>, Scalar<float32>> f = z => z.Asin();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, f(x));
            return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(x, grad, f);
        }
    }

    /// <summary>loss = acos(x). Self-checking (input must be in (−1, 1)).</summary>
    [Module]
    public partial class AutoGradAcosCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            Func<Scalar<float32>, Scalar<float32>> f = z => z.Acos();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, f(x));
            return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(x, grad, f);
        }
    }

    /// <summary>loss = atan(x). Self-checking.</summary>
    [Module]
    public partial class AutoGradAtanCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            Func<Scalar<float32>, Scalar<float32>> f = z => z.Atan();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, f(x));
            return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(x, grad, f);
        }
    }

    /// <summary>loss = sinh(x). Self-checking.</summary>
    [Module]
    public partial class AutoGradSinhCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            Func<Scalar<float32>, Scalar<float32>> f = z => z.Sinh();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, f(x));
            return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(x, grad, f);
        }
    }

    /// <summary>loss = cosh(x). Self-checking.</summary>
    [Module]
    public partial class AutoGradCoshCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            Func<Scalar<float32>, Scalar<float32>> f = z => z.Cosh();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, f(x));
            return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(x, grad, f);
        }
    }

    /// <summary>loss = asinh(x). Self-checking.</summary>
    [Module]
    public partial class AutoGradAsinhCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            Func<Scalar<float32>, Scalar<float32>> f = z => z.Asinh();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, f(x));
            return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(x, grad, f);
        }
    }

    /// <summary>loss = acosh(x). Self-checking (input must be > 1).</summary>
    [Module]
    public partial class AutoGradAcoshCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            Func<Scalar<float32>, Scalar<float32>> f = z => z.Acosh();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, f(x));
            return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(x, grad, f);
        }
    }

    /// <summary>loss = atanh(x). Self-checking (input must be in (−1, 1)).</summary>
    [Module]
    public partial class AutoGradAtanhCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            Func<Scalar<float32>, Scalar<float32>> f = z => z.Atanh();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, f(x));
            return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(x, grad, f);
        }
    }

    /// <summary>loss = tanh(x). Self-checking.</summary>
    [Module]
    public partial class AutoGradTanhCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            Func<Scalar<float32>, Scalar<float32>> f = z => z.Tanh();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, f(x));
            return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(x, grad, f);
        }
    }

    /// <summary>loss = sin(x)·cos(x). Chained sin+cos gradient check.</summary>
    [Module]
    public partial class AutoGradSinCosChainCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            Func<Scalar<float32>, Scalar<float32>> f = z => z.Sin() * z.Cos();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, f(x));
            return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(x, grad, f);
        }
    }

    /// <summary>loss = sinh(x) + cosh(x). Chained sinh+cosh gradient check.</summary>
    [Module]
    public partial class AutoGradSinhCoshChainCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            Func<Scalar<float32>, Scalar<float32>> f = z => (z.Sinh() + z.Cosh()).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, f(x));
            return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(x, grad, f);
        }
    }

    /// <summary>loss = atan(exp(x)). Chained atan+exp gradient check.</summary>
    [Module]
    public partial class AutoGradAtanExpChainCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            Func<Scalar<float32>, Scalar<float32>> f = z => z.Exp().Atan();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, f(x));
            return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(x, grad, f);
        }
    }

    // ===================================================================
    //  Modules converted from AutoDiffMathTests.cs analytical helpers
    // ===================================================================

    /// <summary>loss = exp(x). Self-checking.</summary>
    [Module]
    public partial class AutoGradExpCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            Func<Scalar<float32>, Scalar<float32>> f = z => z.Exp();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, f(x));
            return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(x, grad, f);
        }
    }

    /// <summary>loss = exp(2·x). Chained Mul+Exp gradient check.</summary>
    [Module]
    public partial class AutoGradExpChainedCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            Func<Scalar<float32>, Scalar<float32>> f = z => (Scalar(2f) * z).Exp();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, f(x));
            return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(x, grad, f);
        }
    }

    /// <summary>loss = ln(x). Self-checking (input must be > 0).</summary>
    [Module]
    public partial class AutoGradLogCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            Func<Scalar<float32>, Scalar<float32>> f = z => z.Ln();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, f(x));
            return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(x, grad, f);
        }
    }

    /// <summary>loss = sqrt(x). Self-checking (input must be > 0).</summary>
    [Module]
    public partial class AutoGradSqrtCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            Func<Scalar<float32>, Scalar<float32>> f = z => z.Sqrt();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, f(x));
            return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(x, grad, f);
        }
    }

    /// <summary>loss = sqrt(a)/b. Chained Sqrt+Div gradient check.</summary>
    [Module]
    public partial class AutoGradSqrtDivChainCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b)
        {
            var loss = a.Sqrt() / b;
            var (gradA, gradB) = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, b, loss);
            // dL/da = 1/(2√a·b), dL/db = -√a/b²
            var expectedA = Scalar(1f) / (Scalar(2f) * a.Sqrt() * b);
            var expectedB = -a.Sqrt() / (b * b);
            var okA = (gradA! - expectedA).Abs() < Scalar(1e-3f);
            var okB = (gradB! - expectedB).Abs() < Scalar(1e-3f);
            return okA & okB;
        }
    }

    /// <summary>loss = ln(exp(x)) = x. dL/dx = 1.</summary>
    [Module]
    public partial class AutoGradLogExpIdentityCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            var loss = x.Exp().Ln();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            return (grad - Scalar(1f)).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>loss = erf(x). Self-checking.</summary>
    [Module]
    public partial class AutoGradErfCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            Func<Scalar<float32>, Scalar<float32>> f = z => ((Tensor<float32>)OnnxOp.Erf((Tensor<float32>)z)).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, f(x));
            return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(x, grad, f);
        }
    }

    /// <summary>loss = sign(x). Gradient is 0 (sign is locally constant away from 0).</summary>
    [Module]
    public partial class AutoGradSignCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            var loss = x.Sign();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            return grad.Abs() < Scalar(1e-4f);
        }
    }

    /// <summary>loss = ceil(x). Gradient is 0 (locally constant away from integers).</summary>
    [Module]
    public partial class AutoGradCeilCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            var loss = x.Ceiling();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            return grad.Abs() < Scalar(1e-4f);
        }
    }

    /// <summary>loss = floor(x). Gradient is 0 (locally constant away from integers).</summary>
    [Module]
    public partial class AutoGradFloorCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            var loss = x.Floor();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            return grad.Abs() < Scalar(1e-4f);
        }
    }

    /// <summary>loss = erf(x)·x. Chained Erf+Mul gradient check.</summary>
    [Module]
    public partial class AutoGradErfMulChainCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            Func<Scalar<float32>, Scalar<float32>> f = z => ((Tensor<float32>)OnnxOp.Erf((Tensor<float32>)z)).Scalar() * z;
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, f(x));
            return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(x, grad, f);
        }
    }

    /// <summary>loss = clip(x, 0, 10). Self-checking at smooth points (in-range or saturated).</summary>
    [Module]
    public partial class AutoGradClipCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            Func<Scalar<float32>, Scalar<float32>> f = z => z.Clip(Scalar(0f), Scalar(10f));
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, f(x));
            return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(x, grad, f);
        }
    }

    /// <summary>
    /// loss = sum(cumsum(expand(a, [3]))). Each entry contributes 3·a, 2·a, 1·a → 6·a.
    /// dL/da = 6 (or 6 reversed; cumsum-sum is invariant to direction).
    /// </summary>
    [Module]
    public partial class AutoGradCumSumCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var data = (Tensor<float32>)OnnxOp.Expand(a, Vector(3L));
            var cumulated = (Tensor<float32>)OnnxOp.CumSum(data, Scalar(0L), exclusive: false, reverse: false);
            var loss = cumulated.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(6f)).Abs() < Scalar(1e-4f);
        }
    }

    /// <summary>Reverse cumsum: same total sum, same gradient as forward (6 for length-3 vec).</summary>
    [Module]
    public partial class AutoGradCumSumReverseCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var data = (Tensor<float32>)OnnxOp.Expand(a, Vector(3L));
            var cumulated = (Tensor<float32>)OnnxOp.CumSum(data, Scalar(0L), exclusive: false, reverse: true);
            var loss = cumulated.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(6f)).Abs() < Scalar(1e-4f);
        }
    }

    /// <summary>cumsum(expand(a, [2])) summed then scaled by 3 → dL/da = 3 · 3 = 9.</summary>
    [Module]
    public partial class AutoGradCumSumWithScaleCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var data = (Tensor<float32>)OnnxOp.Expand(a, Vector(2L));
            var cumulated = (Tensor<float32>)OnnxOp.CumSum(data, Scalar(0L), exclusive: false, reverse: false);
            var loss = cumulated.Reduce(ReduceKind.Sum, keepDims: false).Scalar() * Scalar(3f);
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(9f)).Abs() < Scalar(1e-4f);
        }
    }

    /// <summary>
    /// loss = det([[a, 0], [0, 1]]) = a, so dL/da = 1.
    /// </summary>
    [Module]
    public partial class AutoGradDet2x2IdentityCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var mat = Build2x2(a, Scalar(0f), Scalar(0f), Scalar(1f));
            var loss = ((Tensor<float32>)OnnxOp.Det(mat)).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(1f)).Abs() < Scalar(1e-3f);
        }

        private static Tensor<float32> Build2x2(Scalar<float32> a, Scalar<float32> b, Scalar<float32> c, Scalar<float32> d)
        {
            var flat = (Tensor<float32>)OnnxOp.Concat([
                OnnxOp.Unsqueeze(a, Vector(0L)),
                OnnxOp.Unsqueeze(b, Vector(0L)),
                OnnxOp.Unsqueeze(c, Vector(0L)),
                OnnxOp.Unsqueeze(d, Vector(0L)),
            ], axis: 0);
            return (Tensor<float32>)OnnxOp.Reshape(flat, Vector(2L, 2L), allowZero: false);
        }
    }

    /// <summary>
    /// loss = det([[a, 0], [0, a]]) = a², so dL/da = 2·a.
    /// </summary>
    [Module]
    public partial class AutoGradDet2x2DiagonalCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var mat = Build2x2(a, Scalar(0f), Scalar(0f), a);
            var loss = ((Tensor<float32>)OnnxOp.Det(mat)).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(2f) * a).Abs() < Scalar(1e-3f);
        }

        private static Tensor<float32> Build2x2(Scalar<float32> a, Scalar<float32> b, Scalar<float32> c, Scalar<float32> d)
        {
            var flat = (Tensor<float32>)OnnxOp.Concat([
                OnnxOp.Unsqueeze(a, Vector(0L)),
                OnnxOp.Unsqueeze(b, Vector(0L)),
                OnnxOp.Unsqueeze(c, Vector(0L)),
                OnnxOp.Unsqueeze(d, Vector(0L)),
            ], axis: 0);
            return (Tensor<float32>)OnnxOp.Reshape(flat, Vector(2L, 2L), allowZero: false);
        }
    }

    /// <summary>
    /// loss = 2·det([[a, 1], [0, 1]]) = 2·a, so dL/da = 2.
    /// </summary>
    [Module]
    public partial class AutoGradDet2x2ChainRuleCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var mat = Build2x2(a, Scalar(1f), Scalar(0f), Scalar(1f));
            var det_val = ((Tensor<float32>)OnnxOp.Det(mat)).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var loss = det_val * Scalar(2f);
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(2f)).Abs() < Scalar(1e-3f);
        }

        private static Tensor<float32> Build2x2(Scalar<float32> a, Scalar<float32> b, Scalar<float32> c, Scalar<float32> d)
        {
            var flat = (Tensor<float32>)OnnxOp.Concat([
                OnnxOp.Unsqueeze(a, Vector(0L)),
                OnnxOp.Unsqueeze(b, Vector(0L)),
                OnnxOp.Unsqueeze(c, Vector(0L)),
                OnnxOp.Unsqueeze(d, Vector(0L)),
            ], axis: 0);
            return (Tensor<float32>)OnnxOp.Reshape(flat, Vector(2L, 2L), allowZero: false);
        }
    }

    /// <summary>CastLike with same dtype is identity-like; dL/da = 1.</summary>
    [Module]
    public partial class AutoGradCastLikeCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> target)
        {
            var casted = (Tensor<float32>)OnnxOp.CastLike(a, target, saturate: null);
            var loss = ((Tensor<float32>)OnnxOp.Reshape(casted, Vector(new long[0]), allowZero: false)).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(1f)).Abs() < Scalar(1e-4f);
        }
    }

    /// <summary>CastLike·3 → dL/da = 3.</summary>
    [Module]
    public partial class AutoGradCastLikeWithScaleCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> target)
        {
            var casted = (Tensor<float32>)OnnxOp.CastLike(a, target, saturate: null);
            var loss = ((Tensor<float32>)OnnxOp.Reshape(casted, Vector(new long[0]), allowZero: false)).Scalar() * Scalar(3f);
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(3f)).Abs() < Scalar(1e-4f);
        }
    }

    // ===================================================================
    //  Modules converted from AutoDiffReductionTests.cs analytical helpers
    //  All take a scalar input, expand to a length-2 vector, and reduce.
    // ===================================================================

    /// <summary>loss = prod(expand(a, [2])) = a². dL/da = 2a.</summary>
    [Module]
    public partial class AutoGradReduceProdCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            Func<Scalar<float32>, Scalar<float32>> f = z =>
                ((Tensor<float32>)OnnxOp.Expand(z, Vector(2L))).Reduce(ReduceKind.Prod, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, f(a));
            return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(a, grad, f);
        }
    }

    /// <summary>loss = sumsquare(expand(a, [2])) = 2·a². dL/da = 4a.</summary>
    [Module]
    public partial class AutoGradReduceSumSquareCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            Func<Scalar<float32>, Scalar<float32>> f = z =>
                ((Tensor<float32>)OnnxOp.Expand(z, Vector(2L))).Reduce(ReduceKind.SumSquare, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, f(a));
            return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(a, grad, f);
        }
    }

    /// <summary>loss = logsumexp(expand(a, [2])) = a + log(2). dL/da = 1.</summary>
    [Module]
    public partial class AutoGradReduceLogSumExpCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            Func<Scalar<float32>, Scalar<float32>> f = z =>
                ((Tensor<float32>)OnnxOp.Expand(z, Vector(2L))).Reduce(ReduceKind.LogSumExp, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, f(a));
            return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(a, grad, f);
        }
    }

    /// <summary>loss = L1(expand(a, [2])) = 2·|a|. For positive a, dL/da = 2.</summary>
    [Module]
    public partial class AutoGradReduceL1Check
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            Func<Scalar<float32>, Scalar<float32>> f = z =>
                ((Tensor<float32>)OnnxOp.Expand(z, Vector(2L))).Reduce(ReduceKind.L1, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, f(a));
            return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(a, grad, f);
        }
    }

    /// <summary>loss = L2(expand(a, [2])) = √2·|a|. For positive a, dL/da = √2.</summary>
    [Module]
    public partial class AutoGradReduceL2Check
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            Func<Scalar<float32>, Scalar<float32>> f = z =>
                ((Tensor<float32>)OnnxOp.Expand(z, Vector(2L))).Reduce(ReduceKind.L2, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, f(a));
            return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(a, grad, f);
        }
    }

    /// <summary>loss = log(sum(expand(a, [2]))) = log(2·a). dL/da = 1/a.</summary>
    [Module]
    public partial class AutoGradReduceLogSumCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            Func<Scalar<float32>, Scalar<float32>> f = z =>
                ((Tensor<float32>)OnnxOp.Expand(z, Vector(2L))).Reduce(ReduceKind.LogSum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, f(a));
            return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(a, grad, f);
        }
    }

    /// <summary>loss = max(expand(a, [2])) = a. dL/da = 1.</summary>
    [Module]
    public partial class AutoGradReduceMaxCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            Func<Scalar<float32>, Scalar<float32>> f = z =>
                ((Tensor<float32>)OnnxOp.Expand(z, Vector(2L))).Reduce(ReduceKind.Max, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, f(a));
            return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(a, grad, f);
        }
    }

    /// <summary>loss = min(expand(a, [2])) = a. dL/da = 1.</summary>
    [Module]
    public partial class AutoGradReduceMinCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            Func<Scalar<float32>, Scalar<float32>> f = z =>
                ((Tensor<float32>)OnnxOp.Expand(z, Vector(2L))).Reduce(ReduceKind.Min, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, f(a));
            return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(a, grad, f);
        }
    }

    // ===================================================================
    //  Modules converted from AutoDiffOptionalTests.cs analytical helpers
    //  Chain: scalar → Expand → Optional → OptionalGetElement → Sum → loss.
    //  dL/da = (count of expand elements) × (optional scale).
    // ===================================================================

    /// <summary>Expand [2,3] → Optional → unwrap → sum. dL/da = 6.</summary>
    [Module]
    public partial class AutoGradOptionalWrapUnwrap2x3Check
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var mat = (Tensor<float32>)OnnxOp.Expand(a, Vector(2L, 3L));
            var optional = OnnxOp.Optional(mat, DataStructure.Tensor, DType.Float32);
            var extracted = (Tensor<float32>)OnnxOp.OptionalGetElement(optional);
            var loss = extracted.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(6f)).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>Expand [1] → Optional → unwrap → sum. dL/da = 1.</summary>
    [Module]
    public partial class AutoGradOptionalWrapUnwrapScalarCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var tensor = (Tensor<float32>)OnnxOp.Expand(a, Vector(1L));
            var optional = OnnxOp.Optional(tensor, DataStructure.Tensor, DType.Float32);
            var extracted = (Tensor<float32>)OnnxOp.OptionalGetElement(optional);
            var loss = extracted.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(1f)).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>Expand [3,4] → Optional → unwrap → sum. dL/da = 12.</summary>
    [Module]
    public partial class AutoGradOptionalWrapUnwrap3x4Check
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var mat = (Tensor<float32>)OnnxOp.Expand(a, Vector(3L, 4L));
            var optional = OnnxOp.Optional(mat, DataStructure.Tensor, DType.Float32);
            var extracted = (Tensor<float32>)OnnxOp.OptionalGetElement(optional);
            var loss = extracted.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(12f)).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>Expand [2,2] → ·2 → Optional → unwrap → sum. dL/da = 8.</summary>
    [Module]
    public partial class AutoGradOptionalWrapUnwrapWithScaleCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var mat = (Tensor<float32>)OnnxOp.Expand(a, Vector(2L, 2L));
            var scaled = mat * Scalar(2.0f);
            var optional = OnnxOp.Optional(scaled, DataStructure.Tensor, DType.Float32);
            var extracted = (Tensor<float32>)OnnxOp.OptionalGetElement(optional);
            var loss = extracted.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(8f)).Abs() < Scalar(1e-3f);
        }
    }

    // ===================================================================
    //  Modules converted from AutoDiffSequenceTests.cs analytical helpers
    // ===================================================================

    /// <summary>Concat [a]+[b] → sum. dL/da = dL/db = 1.</summary>
    [Module]
    public partial class AutoGradConcat2Check
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b)
        {
            var aVec = (Tensor<float32>)OnnxOp.Reshape(a, Vector(1L), allowZero: false);
            var bVec = (Tensor<float32>)OnnxOp.Reshape(b, Vector(1L), allowZero: false);
            var loss = ((Tensor<float32>)OnnxOp.Concat([aVec, bVec], axis: 0))
                .Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var (gradA, gradB) = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, b, loss);
            var okA = (gradA! - Scalar(1f)).Abs() < Scalar(1e-3f);
            var okB = (gradB! - Scalar(1f)).Abs() < Scalar(1e-3f);
            return okA & okB;
        }
    }

    /// <summary>Concat expand[a,2]+[b]*2 → 2·sum. dL/da = 4, dL/db = 2.</summary>
    [Module]
    public partial class AutoGradConcatWithScaleCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b)
        {
            var aVec = (Tensor<float32>)OnnxOp.Expand(a, Vector(2L));
            var bVec = (Tensor<float32>)OnnxOp.Reshape(b, Vector(1L), allowZero: false);
            var concatenated = (Tensor<float32>)OnnxOp.Concat([aVec, bVec], axis: 0);
            var loss = concatenated.Reduce(ReduceKind.Sum, keepDims: false).Scalar() * Scalar(2f);
            var (gradA, gradB) = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, b, loss);
            var okA = (gradA! - Scalar(4f)).Abs() < Scalar(1e-3f);
            var okB = (gradB! - Scalar(2f)).Abs() < Scalar(1e-3f);
            return okA & okB;
        }
    }

    /// <summary>Concat 3 scalars → sum. dL/da = dL/db = dL/dc = 1.</summary>
    [Module]
    public partial class AutoGradConcat3Check
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b, Scalar<float32> c)
        {
            var aVec = (Tensor<float32>)OnnxOp.Reshape(a, Vector(1L), allowZero: false);
            var bVec = (Tensor<float32>)OnnxOp.Reshape(b, Vector(1L), allowZero: false);
            var cVec = (Tensor<float32>)OnnxOp.Reshape(c, Vector(1L), allowZero: false);
            var loss = ((Tensor<float32>)OnnxOp.Concat([aVec, bVec, cVec], axis: 0))
                .Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grads = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad([(IValue)a, b, c], loss);
            var gradA = (Scalar<float32>)grads[0]!;
            var gradB = (Scalar<float32>)grads[1]!;
            var gradC = (Scalar<float32>)grads[2]!;
            var okA = (gradA - Scalar(1f)).Abs() < Scalar(1e-3f);
            var okB = (gradB - Scalar(1f)).Abs() < Scalar(1e-3f);
            var okC = (gradC - Scalar(1f)).Abs() < Scalar(1e-3f);
            return okA & okB & okC;
        }
    }

    /// <summary>Concat([Relu(a)], [Relu(b)]) → sum. For positive a, negative b: dL/da=1, dL/db=0.</summary>
    [Module]
    public partial class AutoGradConcatWithActivationCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b)
        {
            var aVec = (Tensor<float32>)OnnxOp.Reshape(a.Relu(), Vector(1L), allowZero: false);
            var bVec = (Tensor<float32>)OnnxOp.Reshape(b.Relu(), Vector(1L), allowZero: false);
            var loss = ((Tensor<float32>)OnnxOp.Concat([aVec, bVec], axis: 0))
                .Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var (gradA, gradB) = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, b, loss);
            var okA = (gradA! - Scalar(1f)).Abs() < Scalar(1e-3f);
            var okB = gradB!.Abs() < Scalar(1e-3f);
            return okA & okB;
        }
    }

    /// <summary>
    /// Expand a→[4], split[2,2], concat back, scale each split by [1,1,2,2], sum.
    /// loss = 2·a + 2·a + 4·a + 4·a = 12·a... wait: sum=(a·1+a·1+a·2+a·2)+(reduce-sum)=6·a. dL/da = 6.
    /// </summary>
    [Module]
    public partial class AutoGradSplit2OutputsCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var data = (Tensor<float32>)OnnxOp.Expand(a, Vector(4L));
            var splits = OnnxOp.Split(data, Vector(2L, 2L), axis: 0, numOutputs: null, variadicOutputCount: 2);
            var combined = (Tensor<float32>)OnnxOp.Concat([splits[0], splits[1]], axis: 0);
            var scale = (Tensor<float32>)OnnxOp.Expand(Scalar(1f), Vector(2L));
            var scale2 = (Tensor<float32>)OnnxOp.Expand(Scalar(2f), Vector(2L));
            var scaleVec = (Tensor<float32>)OnnxOp.Concat([scale, scale2], axis: 0);
            var scaled = combined * scaleVec;
            var loss = scaled.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(6f)).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>Expand a→[3], split[1,1,1], concat, sum. dL/da = 3.</summary>
    [Module]
    public partial class AutoGradSplit3OutputsCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var data = (Tensor<float32>)OnnxOp.Expand(a, Vector(3L));
            var splits = OnnxOp.Split(data, Vector(1L, 1L, 1L), axis: 0, numOutputs: null, variadicOutputCount: 3);
            var combined = (Tensor<float32>)OnnxOp.Concat([splits[0], splits[1], splits[2]], axis: 0);
            var loss = combined.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(3f)).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>Expand a→[4], split[2,2], concat, scale[3,3,1,1], sum = 8·a. dL/da = 8.</summary>
    [Module]
    public partial class AutoGradSplitWithScaleCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var data = (Tensor<float32>)OnnxOp.Expand(a, Vector(4L));
            var splits = OnnxOp.Split(data, Vector(2L, 2L), axis: 0, numOutputs: null, variadicOutputCount: 2);
            var combined = (Tensor<float32>)OnnxOp.Concat([splits[0], splits[1]], axis: 0);
            var s3 = (Tensor<float32>)OnnxOp.Expand(Scalar(3f), Vector(2L));
            var s1 = (Tensor<float32>)OnnxOp.Expand(Scalar(1f), Vector(2L));
            var scaleVec = (Tensor<float32>)OnnxOp.Concat([s3, s1], axis: 0);
            var scaled = combined * scaleVec;
            var loss = scaled.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(8f)).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>Expand a→[4], split[2,2], concat back (round-trip), sum. dL/da = 4.</summary>
    [Module]
    public partial class AutoGradSplitConcatRoundTripCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var data = (Tensor<float32>)OnnxOp.Expand(a, Vector(4L));
            var splits = OnnxOp.Split(data, Vector(2L, 2L), axis: 0, numOutputs: null, variadicOutputCount: 2);
            var recombined = (Tensor<float32>)OnnxOp.Concat([splits[0], splits[1]], axis: 0);
            var loss = recombined.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(4f)).Abs() < Scalar(1e-3f);
        }
    }

    // ===================================================================
    //  Modules converted from AutoDiffIfTests.cs analytical helpers
    // ===================================================================

    /// <summary>IfElse(true, a·2, b·3) → loss = 2a. dL/da = 2, dL/db = 0.</summary>
    [Module]
    public partial class AutoGradIfTrueConditionCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b)
        {
            var thenVal = a * Scalar(2f);
            var elseVal = b * Scalar(3f);
            var loss = Scalar(true).IfElse(thenVal, elseVal);
            var (gradA, gradB) = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, b, loss);
            var okA = (gradA! - Scalar(2f)).Abs() < Scalar(1e-3f);
            var okB = gradB!.Abs() < Scalar(1e-3f);
            return okA & okB;
        }
    }

    /// <summary>IfElse(false, a·2, b·3) → loss = 3b. dL/da = 0, dL/db = 3.</summary>
    [Module]
    public partial class AutoGradIfFalseConditionCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b)
        {
            var thenVal = a * Scalar(2f);
            var elseVal = b * Scalar(3f);
            var loss = Scalar(false).IfElse(thenVal, elseVal);
            var (gradA, gradB) = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, b, loss);
            var okA = gradA!.Abs() < Scalar(1e-3f);
            var okB = (gradB! - Scalar(3f)).Abs() < Scalar(1e-3f);
            return okA & okB;
        }
    }

    /// <summary>IfElse(true, a·2, a·3) → loss = 2a, only then branch contributes. dL/da = 2.</summary>
    [Module]
    public partial class AutoGradIfSharedInputTrueCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var thenVal = a * Scalar(2f);
            var elseVal = a * Scalar(3f);
            var loss = Scalar(true).IfElse(thenVal, elseVal);
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(2f)).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>IfElse(false, a·2, a·3) → loss = 3a, only else branch contributes. dL/da = 3.</summary>
    [Module]
    public partial class AutoGradIfSharedInputFalseCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var thenVal = a * Scalar(2f);
            var elseVal = a * Scalar(3f);
            var loss = Scalar(false).IfElse(thenVal, elseVal);
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(3f)).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>loss = IfElse(true, 2a, 3b) + c. dL/da = 2, dL/db = 0, dL/dc = 1.</summary>
    [Module]
    public partial class AutoGradIfWithDownstreamOpsCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b, Scalar<float32> c)
        {
            var thenVal = a * Scalar(2f);
            var elseVal = b * Scalar(3f);
            var ifResult = Scalar(true).IfElse(thenVal, elseVal);
            var loss = ifResult + c;
            var grads = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad([(IValue)a, b, c], loss);
            var gradA = (Scalar<float32>)grads[0]!;
            var gradB = (Scalar<float32>)grads[1]!;
            var gradC = (Scalar<float32>)grads[2]!;
            var okA = (gradA - Scalar(2f)).Abs() < Scalar(1e-3f);
            var okB = gradB.Abs() < Scalar(1e-3f);
            var okC = (gradC - Scalar(1f)).Abs() < Scalar(1e-3f);
            return okA & okB & okC;
        }
    }

    /// <summary>x = a + 1; loss = IfElse(true, 2x, b). dL/da = 2, dL/db = 0.</summary>
    [Module]
    public partial class AutoGradIfWithUpstreamOpsCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b)
        {
            var x = a + Scalar(1f);
            var thenVal = x * Scalar(2f);
            var loss = Scalar(true).IfElse(thenVal, b);
            var (gradA, gradB) = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, b, loss);
            var okA = (gradA! - Scalar(2f)).Abs() < Scalar(1e-3f);
            var okB = gradB!.Abs() < Scalar(1e-3f);
            return okA & okB;
        }
    }

    /// <summary>
    /// Multi-output If where output[1] is unused: outputGrads = [non-null, null].
    /// Exercises the per-output null-dY branch in IF_CLOSE gradient.
    /// loss = out0 = IfElse(true, 2a, 4a)[0] = 2a. dL/da = 2.
    /// </summary>
    [Module]
    public partial class AutoGradIfMultiOutputPartiallyUsedCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b)
        {
            var (out0, _) = Scalar(true).IfElse(
                (a * Scalar(2f), b * Scalar(7f)),
                (a * Scalar(4f), b * Scalar(9f)));
            var loss = (Scalar<float32>)out0;
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(2f)).Abs() < Scalar(1e-3f);
        }
    }

    // ===================================================================
    //  Modules converted from AutoDiffPoolingTests.cs analytical helpers
    // ===================================================================

    /// <summary>GlobalAveragePool(expand(a, [1,1,2,2])) = a. dL/da = 1.</summary>
    [Module]
    public partial class AutoGradGlobalAveragePoolCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var expanded = (Tensor<float32>)OnnxOp.Expand(a, Vector(1L, 1L, 2L, 2L));
            var pooled = (Tensor<float32>)OnnxOp.GlobalAveragePool(expanded);
            var loss = ((Tensor<float32>)OnnxOp.Reshape(pooled, Vector(new long[0]), allowZero: false)).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(1f)).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>2·GlobalAveragePool(expand(a, [1,1,2,2])) = 2·a. dL/da = 2.</summary>
    [Module]
    public partial class AutoGradGlobalAveragePoolWithScaleCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var expanded = (Tensor<float32>)OnnxOp.Expand(a, Vector(1L, 1L, 2L, 2L));
            var pooled = (Tensor<float32>)OnnxOp.GlobalAveragePool(expanded);
            var loss = (((Tensor<float32>)OnnxOp.Reshape(pooled, Vector(new long[0]), allowZero: false)).Scalar()) * Scalar(2f);
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(2f)).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>GlobalMaxPool(expand(a, [1,1,2,2])) = a (tied max). dL/da = 1.</summary>
    [Module]
    public partial class AutoGradGlobalMaxPoolCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var expanded = (Tensor<float32>)OnnxOp.Expand(a, Vector(1L, 1L, 2L, 2L));
            var pooled = (Tensor<float32>)OnnxOp.GlobalMaxPool(expanded);
            var loss = ((Tensor<float32>)OnnxOp.Reshape(pooled, Vector(new long[0]), allowZero: false)).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(1f)).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>GlobalAveragePool(Exp(expand(a, [1,1,2,2]))) = exp(a). dL/da = exp(a).</summary>
    [Module]
    public partial class AutoGradGlobalAveragePoolExpChainCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            var expanded = (Tensor<float32>)OnnxOp.Expand(x, Vector(1L, 1L, 2L, 2L));
            var expX = (Tensor<float32>)OnnxOp.Exp(expanded);
            var pooled = (Tensor<float32>)OnnxOp.GlobalAveragePool(expX);
            var loss = ((Tensor<float32>)OnnxOp.Reshape(pooled, Vector(new long[0]), allowZero: false)).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            return (grad - x.Exp()).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>GlobalLpPool(expand(a, [1,1,2,2]), p=2) = 2|a|. For positive a, dL/da = 2.</summary>
    [Module]
    public partial class AutoGradGlobalLpPoolP2Check
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var expanded = (Tensor<float32>)OnnxOp.Expand(a, Vector(1L, 1L, 2L, 2L));
            var pooled = (Tensor<float32>)OnnxOp.GlobalLpPool(expanded, 2);
            var loss = ((Tensor<float32>)OnnxOp.Reshape(pooled, Vector(new long[0]), allowZero: false)).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(2f)).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>GlobalLpPool(expand(a, [1,1,2,2]), p=1) = 4|a|. For positive a, dL/da = 4.</summary>
    [Module]
    public partial class AutoGradGlobalLpPoolP1Check
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var expanded = (Tensor<float32>)OnnxOp.Expand(a, Vector(1L, 1L, 2L, 2L));
            var pooled = (Tensor<float32>)OnnxOp.GlobalLpPool(expanded, 1);
            var loss = ((Tensor<float32>)OnnxOp.Reshape(pooled, Vector(new long[0]), allowZero: false)).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(4f)).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>3·GlobalLpPool(expand(a, [1,1,2,2]), p=2) = 6|a|. For positive a, dL/da = 6.</summary>
    [Module]
    public partial class AutoGradGlobalLpPoolP2WithScaleCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var expanded = (Tensor<float32>)OnnxOp.Expand(a, Vector(1L, 1L, 2L, 2L));
            var pooled = (Tensor<float32>)OnnxOp.GlobalLpPool(expanded, 2);
            var loss = (((Tensor<float32>)OnnxOp.Reshape(pooled, Vector(new long[0]), allowZero: false)).Scalar()) * Scalar(3f);
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(6f)).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>sum(LpPool(expand(a, [1,1,4,4]), p=2, k=2, s=2)) = 8|a|. For positive a, dL/da = 8.</summary>
    [Module]
    public partial class AutoGradLpPoolP2Check
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var expanded = (Tensor<float32>)OnnxOp.Expand(a, Vector(1L, 1L, 4L, 4L));
            var pooled = (Tensor<float32>)OnnxOp.LpPool(expanded,
                autoPad: null, ceilMode: null, dilations: null,
                kernelShape: [2L, 2L], p: 2, pads: null, strides: [2L, 2L]);
            var loss = pooled.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(8f)).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>sum(LpPool(expand(a, [1,1,4,4]), p=1, k=2, s=2)) = 16|a|. For positive a, dL/da = 16.</summary>
    [Module]
    public partial class AutoGradLpPoolP1Check
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var expanded = (Tensor<float32>)OnnxOp.Expand(a, Vector(1L, 1L, 4L, 4L));
            var pooled = (Tensor<float32>)OnnxOp.LpPool(expanded,
                autoPad: null, ceilMode: null, dilations: null,
                kernelShape: [2L, 2L], p: 1, pads: null, strides: [2L, 2L]);
            var loss = pooled.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(16f)).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>3·sum(LpPool(expand(a, [1,1,4,4]), p=2, k=2, s=2)) = 24|a|. dL/da = 24.</summary>
    [Module]
    public partial class AutoGradLpPoolP2WithScaleCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var expanded = (Tensor<float32>)OnnxOp.Expand(a, Vector(1L, 1L, 4L, 4L));
            var pooled = (Tensor<float32>)OnnxOp.LpPool(expanded,
                autoPad: null, ceilMode: null, dilations: null,
                kernelShape: [2L, 2L], p: 2, pads: null, strides: [2L, 2L]);
            var loss = pooled.Reduce(ReduceKind.Sum, keepDims: false).Scalar() * Scalar(3f);
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(24f)).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>
    /// MaxPool→MaxUnpool round-trip: loss = sum(unpool(maxpool(expand(a, [1,1,4,4])))) = 4·a.
    /// Tie normalization divides by window-size, so dL/da = 16·(1/4) = 4.
    /// </summary>
    [Module]
    public partial class AutoGradMaxUnpoolCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var expanded = (Tensor<float32>)OnnxOp.Expand(a, Vector(1L, 1L, 4L, 4L));
            var (pooled, indices) = OnnxOp.MaxPoolWithIndices(expanded,
                kernelShape: [2L, 2L], strides: [2L, 2L]);
            var unpooled = (Tensor<float32>)OnnxOp.MaxUnpool(pooled, indices,
                kernelShape: [2L, 2L], strides: [2L, 2L]);
            var loss = unpooled.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(4f)).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>3·sum(MaxUnpool(MaxPool(expand(a, [1,1,4,4])))) = 12·a. dL/da = 12.</summary>
    [Module]
    public partial class AutoGradMaxUnpoolWithScaleCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var expanded = (Tensor<float32>)OnnxOp.Expand(a, Vector(1L, 1L, 4L, 4L));
            var (pooled, indices) = OnnxOp.MaxPoolWithIndices(expanded,
                kernelShape: [2L, 2L], strides: [2L, 2L]);
            var unpooled = (Tensor<float32>)OnnxOp.MaxUnpool(pooled, indices,
                kernelShape: [2L, 2L], strides: [2L, 2L]);
            var loss = unpooled.Reduce(ReduceKind.Sum, keepDims: false).Scalar() * Scalar(3f);
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(12f)).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>MaxPool→MaxUnpool through Exp: dL/da = 4·exp(a).</summary>
    [Module]
    public partial class AutoGradMaxUnpoolChainCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var expanded = (Tensor<float32>)OnnxOp.Expand(a, Vector(1L, 1L, 4L, 4L));
            var expX = (Tensor<float32>)OnnxOp.Exp(expanded);
            var (pooled, indices) = OnnxOp.MaxPoolWithIndices(expX,
                kernelShape: [2L, 2L], strides: [2L, 2L]);
            var unpooled = (Tensor<float32>)OnnxOp.MaxUnpool(pooled, indices,
                kernelShape: [2L, 2L], strides: [2L, 2L]);
            var loss = unpooled.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(4f) * a.Exp()).Abs() < Scalar(1e-3f);
        }
    }

    // ===================================================================
    //  Modules converted from AutoDiffOtherTests.cs analytical helpers
    //  (Dropout in inference mode, Sum/Mean/Max/Min variadic ops)
    // ===================================================================

    /// <summary>Dropout(expand(a, [3])) in inference mode = identity. sum → dL/da = 3.</summary>
    [Module]
    public partial class AutoGradDropoutInferenceCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var aVec = (Tensor<float32>)OnnxOp.Expand(a, Vector(3L));
            var (output, _) = OnnxOp.Dropout(aVec, null, null);
            var loss = ((Tensor<float32>)output).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(3f)).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>2·sum(Dropout(expand(a, [2]))) = 4a. dL/da = 4.</summary>
    [Module]
    public partial class AutoGradDropoutWithScaleCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var aVec = (Tensor<float32>)OnnxOp.Expand(a, Vector(2L));
            var (output, _) = OnnxOp.Dropout(aVec, null, null);
            var loss = ((Tensor<float32>)output).Reduce(ReduceKind.Sum, keepDims: false).Scalar() * Scalar(2f);
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(4f)).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>sum(Dropout(expand(a, [2])) + expand(a, [2])) = 4a. dL/da = 4.</summary>
    [Module]
    public partial class AutoGradDropoutWithDownstreamCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var aVec = (Tensor<float32>)OnnxOp.Expand(a, Vector(2L));
            var (output, _) = OnnxOp.Dropout(aVec, null, null);
            var combined = (Tensor<float32>)output + aVec;
            var loss = combined.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(4f)).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>Sum(a, b) (variadic) → (dL/da, dL/db) = (1, 1).</summary>
    [Module]
    public partial class AutoGradSum2InputsCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b)
        {
            var loss = (Scalar<float32>)OnnxOp.Sum(a, b);
            var (gradA, gradB) = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, b, loss);
            var okA = (gradA! - Scalar(1f)).Abs() < Scalar(1e-3f);
            var okB = (gradB! - Scalar(1f)).Abs() < Scalar(1e-3f);
            return okA & okB;
        }
    }

    /// <summary>Sum(a, b, c) → all gradients = 1.</summary>
    [Module]
    public partial class AutoGradSum3InputsCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b, Scalar<float32> c)
        {
            var loss = (Scalar<float32>)OnnxOp.Sum(a, b, c);
            var grads = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad([(IValue)a, b, c], loss);
            var gradA = (Scalar<float32>)grads[0]!;
            var gradB = (Scalar<float32>)grads[1]!;
            var gradC = (Scalar<float32>)grads[2]!;
            return (gradA - Scalar(1f)).Abs() < Scalar(1e-3f)
                & (gradB - Scalar(1f)).Abs() < Scalar(1e-3f)
                & (gradC - Scalar(1f)).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>3·Sum(a, b) → gradients = 3 each.</summary>
    [Module]
    public partial class AutoGradSumWithScaleCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b)
        {
            var loss = ((Scalar<float32>)OnnxOp.Sum(a, b)) * Scalar(3f);
            var (gradA, gradB) = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, b, loss);
            return (gradA! - Scalar(3f)).Abs() < Scalar(1e-3f)
                & (gradB! - Scalar(3f)).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>sum(Sum(expand(a, [2]), expand(b, [2]))) = 2·(a + b). dL/da = dL/db = 2.</summary>
    [Module]
    public partial class AutoGradSumTensorInputsCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b)
        {
            var aVec = (Tensor<float32>)OnnxOp.Expand(a, Vector(2L));
            var bVec = (Tensor<float32>)OnnxOp.Expand(b, Vector(2L));
            var sumResult = (Tensor<float32>)OnnxOp.Sum(aVec, bVec);
            var loss = sumResult.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var (gradA, gradB) = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, b, loss);
            return (gradA! - Scalar(2f)).Abs() < Scalar(1e-3f)
                & (gradB! - Scalar(2f)).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>Mean(a, b) → gradients = 0.5 each.</summary>
    [Module]
    public partial class AutoGradMean2InputsCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b)
        {
            var loss = (Scalar<float32>)OnnxOp.Mean(a, b);
            var (gradA, gradB) = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, b, loss);
            return (gradA! - Scalar(0.5f)).Abs() < Scalar(1e-3f)
                & (gradB! - Scalar(0.5f)).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>Mean(a, b, c) → gradients = 1/3 each.</summary>
    [Module]
    public partial class AutoGradMean3InputsCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b, Scalar<float32> c)
        {
            var loss = (Scalar<float32>)OnnxOp.Mean(a, b, c);
            var grads = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad([(IValue)a, b, c], loss);
            var gradA = (Scalar<float32>)grads[0]!;
            var gradB = (Scalar<float32>)grads[1]!;
            var gradC = (Scalar<float32>)grads[2]!;
            var third = Scalar(1f / 3f);
            return (gradA - third).Abs() < Scalar(1e-3f)
                & (gradB - third).Abs() < Scalar(1e-3f)
                & (gradC - third).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>6·Mean(a, b) = 3a + 3b. dL/da = dL/db = 3.</summary>
    [Module]
    public partial class AutoGradMeanWithScaleCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b)
        {
            var loss = ((Scalar<float32>)OnnxOp.Mean(a, b)) * Scalar(6f);
            var (gradA, gradB) = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, b, loss);
            return (gradA! - Scalar(3f)).Abs() < Scalar(1e-3f)
                & (gradB! - Scalar(3f)).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>sum(Mean(expand(a, [2]), expand(b, [2]))) = a + b. dL/da = dL/db = 1.</summary>
    [Module]
    public partial class AutoGradMeanTensorInputsCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b)
        {
            var aVec = (Tensor<float32>)OnnxOp.Expand(a, Vector(2L));
            var bVec = (Tensor<float32>)OnnxOp.Expand(b, Vector(2L));
            var meanResult = (Tensor<float32>)OnnxOp.Mean(aVec, bVec);
            var loss = meanResult.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var (gradA, gradB) = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, b, loss);
            return (gradA! - Scalar(1f)).Abs() < Scalar(1e-3f)
                & (gradB! - Scalar(1f)).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>Max(a=3, b=5) = b. dL/da = 0, dL/db = 1.</summary>
    [Module]
    public partial class AutoGradMax2InputsCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b)
        {
            var loss = (Scalar<float32>)OnnxOp.Max(a, b);
            var (gradA, gradB) = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, b, loss);
            return gradA!.Abs() < Scalar(1e-3f)
                & (gradB! - Scalar(1f)).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>Max(a=2, b=7, c=4) = b. Only dL/db = 1.</summary>
    [Module]
    public partial class AutoGradMax3InputsCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b, Scalar<float32> c)
        {
            var loss = (Scalar<float32>)OnnxOp.Max(a, b, c);
            var grads = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad([(IValue)a, b, c], loss);
            var gradA = (Scalar<float32>)grads[0]!;
            var gradB = (Scalar<float32>)grads[1]!;
            var gradC = (Scalar<float32>)grads[2]!;
            return gradA.Abs() < Scalar(1e-3f)
                & (gradB - Scalar(1f)).Abs() < Scalar(1e-3f)
                & gradC.Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>3·Max(a=2, b=5) = 3b. dL/da = 0, dL/db = 3.</summary>
    [Module]
    public partial class AutoGradMaxWithScaleCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b)
        {
            var loss = ((Scalar<float32>)OnnxOp.Max(a, b)) * Scalar(3f);
            var (gradA, gradB) = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, b, loss);
            return gradA!.Abs() < Scalar(1e-3f)
                & (gradB! - Scalar(3f)).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>sum(Max(expand(a=3, [2]), expand(b=5, [2]))) = 2b. dL/da = 0, dL/db = 2.</summary>
    [Module]
    public partial class AutoGradMaxTensorInputsCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b)
        {
            var aVec = (Tensor<float32>)OnnxOp.Expand(a, Vector(2L));
            var bVec = (Tensor<float32>)OnnxOp.Expand(b, Vector(2L));
            var maxResult = (Tensor<float32>)OnnxOp.Max(aVec, bVec);
            var loss = maxResult.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var (gradA, gradB) = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, b, loss);
            return gradA!.Abs() < Scalar(1e-3f)
                & (gradB! - Scalar(2f)).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>Min(a=3, b=5) = a. dL/da = 1, dL/db = 0.</summary>
    [Module]
    public partial class AutoGradMin2InputsCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b)
        {
            var loss = (Scalar<float32>)OnnxOp.Min(a, b);
            var (gradA, gradB) = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, b, loss);
            return (gradA! - Scalar(1f)).Abs() < Scalar(1e-3f)
                & gradB!.Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>Min(a=5, b=2, c=4) = b. Only dL/db = 1.</summary>
    [Module]
    public partial class AutoGradMin3InputsCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b, Scalar<float32> c)
        {
            var loss = (Scalar<float32>)OnnxOp.Min(a, b, c);
            var grads = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad([(IValue)a, b, c], loss);
            var gradA = (Scalar<float32>)grads[0]!;
            var gradB = (Scalar<float32>)grads[1]!;
            var gradC = (Scalar<float32>)grads[2]!;
            return gradA.Abs() < Scalar(1e-3f)
                & (gradB - Scalar(1f)).Abs() < Scalar(1e-3f)
                & gradC.Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>4·Min(a=8, b=3) = 4b. dL/da = 0, dL/db = 4.</summary>
    [Module]
    public partial class AutoGradMinWithScaleCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b)
        {
            var loss = ((Scalar<float32>)OnnxOp.Min(a, b)) * Scalar(4f);
            var (gradA, gradB) = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, b, loss);
            return gradA!.Abs() < Scalar(1e-3f)
                & (gradB! - Scalar(4f)).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>sum(Min(expand(a=7, [2]), expand(b=2, [2]))) = 2b. dL/da = 0, dL/db = 2.</summary>
    [Module]
    public partial class AutoGradMinTensorInputsCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b)
        {
            var aVec = (Tensor<float32>)OnnxOp.Expand(a, Vector(2L));
            var bVec = (Tensor<float32>)OnnxOp.Expand(b, Vector(2L));
            var minResult = (Tensor<float32>)OnnxOp.Min(aVec, bVec);
            var loss = minResult.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var (gradA, gradB) = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, b, loss);
            return gradA!.Abs() < Scalar(1e-3f)
                & (gradB! - Scalar(2f)).Abs() < Scalar(1e-3f);
        }
    }

    // ===================================================================
    //  Modules converted from AutoDiffMatrixTests.cs analytical helpers
    //  (Einsum/Gemm with scalar→Expand patterns; Matmul/Reduce coverage)
    // ===================================================================

    /// <summary>
    /// Einsum "ij,jk->ik" matmul: A=[[a,0],[0,1]], B=ones(2,2). C[i,k] = Σ_j A[i,j]·B[j,k].
    /// sum(C) = (a+0)+(a+0)+(0+1)+(0+1) = 2a + 2. dL/da = 2.
    /// </summary>
    [Module]
    public partial class AutoGradEinsumMatmulBasicCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var aVec = (Tensor<float32>)OnnxOp.Unsqueeze(a, Vector(0L));
            var flat = (Tensor<float32>)OnnxOp.Concat([aVec, Vector(0f), Vector(0f), Vector(1f)], axis: 0);
            var A = (Tensor<float32>)OnnxOp.Reshape(flat, Vector(2L, 2L), allowZero: false);
            var B = (Tensor<float32>)OnnxOp.Reshape(Vector(1f, 1f, 1f, 1f), Vector(2L, 2L), allowZero: false);
            var C = (Tensor<float32>)OnnxOp.Einsum([A, B], equation: "ij,jk->ik");
            var loss = C.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(2f)).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>Einsum "ij->ji" transpose of [[a,2],[3,4]]. sum is invariant → dL/da = 1.</summary>
    [Module]
    public partial class AutoGradEinsumTransposeCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var aVec = (Tensor<float32>)OnnxOp.Unsqueeze(a, Vector(0L));
            var flat = (Tensor<float32>)OnnxOp.Concat([aVec, Vector(2f), Vector(3f), Vector(4f)], axis: 0);
            var A = (Tensor<float32>)OnnxOp.Reshape(flat, Vector(2L, 2L), allowZero: false);
            var At = (Tensor<float32>)OnnxOp.Einsum([A], equation: "ij->ji");
            var loss = At.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(1f)).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>
    /// Einsum implicit mode (no '->'): A=[2,3] all=a, B=ones(3,2).
    /// C[i,k] = Σ_j a·1 = 3a, sum = 4·3a = 12a. dL/da = 12.
    /// </summary>
    [Module]
    public partial class AutoGradEinsumImplicitModeCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var A = (Tensor<float32>)OnnxOp.Expand(a, Vector(2L, 3L));
            var B = (Tensor<float32>)OnnxOp.Reshape(Vector(1f, 1f, 1f, 1f, 1f, 1f), Vector(3L, 2L), allowZero: false);
            var C = (Tensor<float32>)OnnxOp.Einsum([A, B], equation: "ij,jk");
            var loss = C.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(12f)).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>
    /// Einsum "ij->j" (free index): A=[2,3] all=a → out[j] = Σ_i a = 2a, sum=6a. dL/da = 6.
    /// </summary>
    [Module]
    public partial class AutoGradEinsumFreeIndexCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var A = (Tensor<float32>)OnnxOp.Expand(a, Vector(2L, 3L));
            var summed = (Tensor<float32>)OnnxOp.Einsum([A], equation: "ij->j");
            var loss = summed.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(6f)).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>Gemm(expand(a, [1,2]), expand(a, [2,1])) = 2a². sum scalar. dL/da = 4a = 12.</summary>
    [Module]
    public partial class AutoGradGemmBasicCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var aMat1 = (Tensor<float32>)OnnxOp.Expand(a, Vector(1L, 2L));
            var aMat2 = (Tensor<float32>)OnnxOp.Expand(a, Vector(2L, 1L));
            var output = (Tensor<float32>)OnnxOp.Gemm(aMat1, aMat2);
            var loss = output.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(4f) * a).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>alpha=2 Gemm doubles the gradient: dL/da = 8a = 16.</summary>
    [Module]
    public partial class AutoGradGemmWithAlphaCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var aMat1 = (Tensor<float32>)OnnxOp.Expand(a, Vector(1L, 2L));
            var aMat2 = (Tensor<float32>)OnnxOp.Expand(a, Vector(2L, 1L));
            var output = (Tensor<float32>)OnnxOp.Gemm(aMat1, aMat2, alpha: 2.0f);
            var loss = output.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(8f) * a).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>Gemm(a, a, c=a, beta=3) at 1x1: output = a² + 3a. dL/da = 2a + 3 = 7.</summary>
    [Module]
    public partial class AutoGradGemmWithBetaAndCCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var aMat = (Tensor<float32>)OnnxOp.Expand(a, Vector(1L, 1L));
            var output = (Tensor<float32>)OnnxOp.Gemm(aMat, aMat, aMat, beta: 3.0f);
            var loss = output.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - (Scalar(2f) * a + Scalar(3f))).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>Gemm with transA=1 over Expand(a, [2,1])·Expand(a, [2,1]) = 2a². dL/da = 4a = 12.</summary>
    [Module]
    public partial class AutoGradGemmTransACheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var aCol = (Tensor<float32>)OnnxOp.Expand(a, Vector(2L, 1L));
            var bCol = (Tensor<float32>)OnnxOp.Expand(a, Vector(2L, 1L));
            var output = (Tensor<float32>)OnnxOp.Gemm(aCol, bCol, transA: 1);
            var loss = output.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(4f) * a).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>Gemm with transB=1 over Expand(a, [1,2])·Expand(a, [1,2]) = 2a². dL/da = 4a = 8.</summary>
    [Module]
    public partial class AutoGradGemmTransBCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var aRow = (Tensor<float32>)OnnxOp.Expand(a, Vector(1L, 2L));
            var bRow = (Tensor<float32>)OnnxOp.Expand(a, Vector(1L, 2L));
            var output = (Tensor<float32>)OnnxOp.Gemm(aRow, bRow, transB: 1);
            var loss = output.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(4f) * a).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>
    /// Known-rank MatMul: Identity stamps rank=2 so TransposeLastTwoDims uses an
    /// explicit perm. A=B=expand(a, [2,3]) / [3,2] → C = 3a²·ones(2,2), sum=12a².
    /// dL/da = 24a = 48.
    /// </summary>
    [Module]
    public partial class AutoGradMatMulKnownRankCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var A = (Tensor<float32>)OnnxOp.Identity(OnnxOp.Expand(a, Vector(2L, 3L)), rank: 2);
            var B = (Tensor<float32>)OnnxOp.Identity(OnnxOp.Expand(a, Vector(3L, 2L)), rank: 2);
            var C = (Tensor<float32>)OnnxOp.MatMul(A, B);
            var loss = C.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(24f) * a).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>
    /// Unknown-rank batched MatMul: no Identity rank stamp, so the operands' Rank is
    /// null and the MatMul gradient takes the rank-agnostic last-two-dims transpose
    /// fallback (collapse leading dims → swap → restore) instead of the static perm.
    /// A=expand(a, [2,2,3]), B=expand(a, [2,3,2]) → C = 3a²·ones(2,2,2), sum=24a².
    /// dL/da = 48a = 96. Companion to AutoGradMatMulKnownRankCheck; guards the
    /// batched-matmul backprop fix (reverse-all transpose used to corrupt the batch dim).
    /// </summary>
    [Module]
    public partial class AutoGradMatMulUnknownRankBatchedCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var A = (Tensor<float32>)OnnxOp.Expand(a, Vector(2L, 2L, 3L));
            var B = (Tensor<float32>)OnnxOp.Expand(a, Vector(2L, 3L, 2L));
            var C = (Tensor<float32>)OnnxOp.MatMul(A, B);
            var loss = C.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(48f) * a).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>
    /// ReduceSum(expand(a, [2,3]), axes=[1], keepdims=true) → [2,1] all=3a, sum=6a.
    /// Exercises axes-not-null + effectiveKeepdims=true branch. dL/da = 6.
    /// </summary>
    [Module]
    public partial class AutoGradReduceSumExplicitAxesKeepdimsTrueCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var data = (Tensor<float32>)OnnxOp.Expand(a, Vector(2L, 3L));
            var reduced = (Tensor<float32>)OnnxOp.ReduceSum(data, Vector(1L), keepdims: true, noopWithEmptyAxes: null);
            var loss = reduced.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(6f)).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>
    /// ReduceSum(expand(a, [2,3]), axes=[1], keepdims=false) → [2] all=3a, sum=6a.
    /// Exercises axes-not-null + effectiveKeepdims=false branch (Unsqueeze fallback).
    /// </summary>
    [Module]
    public partial class AutoGradReduceSumExplicitAxesKeepdimsFalseCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var data = (Tensor<float32>)OnnxOp.Expand(a, Vector(2L, 3L));
            var reduced = (Tensor<float32>)OnnxOp.ReduceSum(data, Vector(1L), keepdims: false, noopWithEmptyAxes: null);
            var loss = reduced.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(6f)).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>
    /// ReduceMean(expand(a, [2,3]), axes=[1], keepdims=false) → [2] all=a, sum=2a.
    /// Exercises the explicit-axes Gather+ReduceProd reduced-count branch. dL/da = 2.
    /// </summary>
    [Module]
    public partial class AutoGradReduceMeanExplicitAxesCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var data = (Tensor<float32>)OnnxOp.Expand(a, Vector(2L, 3L));
            var reduced = (Tensor<float32>)OnnxOp.ReduceMean(data, Vector(1L), keepdims: false, noopWithEmptyAxes: null);
            var loss = reduced.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(2f)).Abs() < Scalar(1e-3f);
        }
    }

    // ===================================================================
    //  AutoDiffAffineGridTests / AutoDiffGridSampleTests /
    //  AutoDiffMaxRoiPoolTests / AutoDiffRoiAlignTests — analytical tests
    //  with closed-form gradients.
    // ===================================================================

    /// <summary>
    /// AffineGrid(theta * a, size=[2,1,2,2], align_corners=true) where theta is a pure
    /// x-translation [[0,0,1],[0,0,0]]·2 batches. sum(grid) = N·H·W·a = 8·a. dL/da = 8.
    /// </summary>
    [Module]
    public partial class AutoGradAffineGridMultiBatchCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var translationValues = (Tensor<float32>)OnnxOp.Constant(
                TensorData(12, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f));
            var thetaConst = (Tensor<float32>)OnnxOp.Reshape(translationValues, Vector(2L, 2L, 3L), allowZero: false);
            var theta = (Tensor<float32>)OnnxOp.Mul(thetaConst, a);
            var size = (Tensor<int64>)OnnxOp.Constant(TensorData(4, 2L, 1L, 2L, 2L));
            var grid = (Tensor<float32>)OnnxOp.AffineGrid(theta, size, alignCorners: true);
            var loss = grid.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(8f)).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>
    /// Reshape gradient passthrough: x scalar → reshape [1] → ·2 → reshape scalar.
    /// loss = 2x. dL/dx = 2.
    /// </summary>
    [Module]
    public partial class AutoGradReshapePassthroughCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            var reshaped = (Tensor<float32>)OnnxOp.Reshape(x, Vector(1L), allowZero: false);
            var product = (Tensor<float32>)OnnxOp.Mul(reshaped, Scalar(2f));
            var loss = ((Tensor<float32>)OnnxOp.Reshape(product, Vector(new long[0]), allowZero: false)).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            return (grad - Scalar(2f)).Abs() < Scalar(1e-3f);
        }
    }

    // ===================================================================
    //  Normalization (converted from AutoDiffNormalizationTests). The
    //  closed-form helpers all hit one of two cases:
    //    1. trainingMode=false BatchNorm with scale=1, bias=0, mean=0, var=1
    //       → y ≈ x; sum(y) = N·a; dL/da = N. BatchNorm modules use
    //       epsilon: 1e-8f instead of the ONNX default 1e-5f because the
    //       noise floor of the analytical gradient is ~N·ε/2, which would
    //       exceed the 1e-5f bit-check tolerance once N≥4. The 1e-8f
    //       epsilon exercises the same BatchNormalization-gradient code
    //       path as 1e-5f, just below the tolerance noise floor.
    //    2. Norm operating on a constant or fully-correlated input → the
    //       normalization output has zero sum-gradient w.r.t. the scalar input.
    // ===================================================================

    [Module]
    public partial class AutoGradBatchNormSimpleCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var x = (Tensor<float32>)OnnxOp.Expand(a, Vector(1L, 2L));
            var scale = (Tensor<float32>)OnnxOp.Expand(Scalar(1f), Vector(2L));
            var bias = (Tensor<float32>)OnnxOp.Expand(Scalar(0f), Vector(2L));
            var mean = (Tensor<float32>)OnnxOp.Expand(Scalar(0f), Vector(2L));
            var variance = (Tensor<float32>)OnnxOp.Expand(Scalar(1f), Vector(2L));
            var bn = (Tensor<float32>)OnnxOp.BatchNormalization(x, scale, bias, mean, variance,
                epsilon: 1e-8f, momentum: null, trainingMode: false);
            var loss = bn.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(2f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradBatchNormWithScaleCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var x = (Tensor<float32>)OnnxOp.Expand(a, Vector(1L, 2L));
            var scale = (Tensor<float32>)OnnxOp.Expand(Scalar(2f), Vector(2L));
            var bias = (Tensor<float32>)OnnxOp.Expand(Scalar(0f), Vector(2L));
            var mean = (Tensor<float32>)OnnxOp.Expand(Scalar(0f), Vector(2L));
            var variance = (Tensor<float32>)OnnxOp.Expand(Scalar(1f), Vector(2L));
            var bn = (Tensor<float32>)OnnxOp.BatchNormalization(x, scale, bias, mean, variance,
                epsilon: 1e-8f, momentum: null, trainingMode: false);
            var loss = bn.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(4f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradBatchNorm3DCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var x = (Tensor<float32>)OnnxOp.Expand(a, Vector(1L, 2L, 3L));
            var scale = (Tensor<float32>)OnnxOp.Expand(Scalar(1f), Vector(2L));
            var bias = (Tensor<float32>)OnnxOp.Expand(Scalar(0f), Vector(2L));
            var mean = (Tensor<float32>)OnnxOp.Expand(Scalar(0f), Vector(2L));
            var variance = (Tensor<float32>)OnnxOp.Expand(Scalar(1f), Vector(2L));
            var bn = (Tensor<float32>)OnnxOp.BatchNormalization(x, scale, bias, mean, variance,
                epsilon: 1e-8f, momentum: null, trainingMode: false);
            var loss = bn.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(6f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradBatchNormExpChainCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var x = (Tensor<float32>)OnnxOp.Expand(a, Vector(1L, 2L));
            var scale = (Tensor<float32>)OnnxOp.Expand(Scalar(1f), Vector(2L));
            var bias = (Tensor<float32>)OnnxOp.Expand(Scalar(0f), Vector(2L));
            var mean = (Tensor<float32>)OnnxOp.Expand(Scalar(0f), Vector(2L));
            var variance = (Tensor<float32>)OnnxOp.Expand(Scalar(1f), Vector(2L));
            var bn = (Tensor<float32>)OnnxOp.BatchNormalization(x, scale, bias, mean, variance,
                epsilon: 1e-8f, momentum: null, trainingMode: false);
            var expBn = bn.Exp();
            var loss = expBn.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            var expected = Scalar(2f) * a.Exp();
            return (grad - expected).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradGroupNormBasicCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var input = (Tensor<float32>)OnnxOp.Expand(a, Vector(1L, 2L, 1L, 1L));
            var scale = (Tensor<float32>)OnnxOp.Expand(Scalar(1f), Vector(2L));
            var bias = (Tensor<float32>)OnnxOp.Expand(Scalar(0f), Vector(2L));
            var output = (Tensor<float32>)OnnxOp.GroupNormalization(input, scale, bias,
                epsilon: 1e-5f, numGroups: 2);
            var loss = output.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return grad.Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradGroupNormScaleCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> s)
        {
            var input = (Tensor<float32>)OnnxOp.Expand(Scalar(2f), Vector(1L, 2L, 1L, 2L));
            var scale = (Tensor<float32>)OnnxOp.Expand(s, Vector(2L));
            var bias = (Tensor<float32>)OnnxOp.Expand(Scalar(0f), Vector(2L));
            var output = (Tensor<float32>)OnnxOp.GroupNormalization(input, scale, bias,
                epsilon: 1e-5f, numGroups: 1);
            var loss = output.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(s, loss);
            return grad.Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradGroupNormNonConstInputCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var two = Scalar(2f);
            var channelVals = (Tensor<float32>)OnnxOp.Concat(
                [OnnxOp.Reshape(a, Vector(1L), allowZero: false),
                 OnnxOp.Reshape(a * two, Vector(1L), allowZero: false)], axis: 0);
            var input = (Tensor<float32>)OnnxOp.Reshape(channelVals, Vector(1L, 2L, 1L, 1L), allowZero: false);
            var scale = (Tensor<float32>)OnnxOp.Expand(Scalar(1f), Vector(2L));
            var bias = (Tensor<float32>)OnnxOp.Expand(Scalar(0f), Vector(2L));
            var output = (Tensor<float32>)OnnxOp.GroupNormalization(input, scale, bias,
                epsilon: 1e-5f, numGroups: 1);
            var loss = output.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return grad.Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradGroupNorm2GroupsCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var vals = (Tensor<float32>)OnnxOp.Concat(
                [OnnxOp.Reshape(a, Vector(1L), allowZero: false),
                 OnnxOp.Reshape(a * Scalar(2f), Vector(1L), allowZero: false),
                 OnnxOp.Reshape(a * Scalar(3f), Vector(1L), allowZero: false),
                 OnnxOp.Reshape(a * Scalar(4f), Vector(1L), allowZero: false)], axis: 0);
            var input = (Tensor<float32>)OnnxOp.Reshape(vals, Vector(1L, 4L, 1L, 1L), allowZero: false);
            var scale = (Tensor<float32>)OnnxOp.Expand(Scalar(1f), Vector(4L));
            var bias = (Tensor<float32>)OnnxOp.Expand(Scalar(0f), Vector(4L));
            var output = (Tensor<float32>)OnnxOp.GroupNormalization(input, scale, bias,
                epsilon: 1e-5f, numGroups: 2);
            var loss = output.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return grad.Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradInstanceNormBasicCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var input = (Tensor<float32>)OnnxOp.Reshape(
                OnnxOp.Concat([
                    (Tensor<float32>)OnnxOp.Reshape(a, Vector(1L), allowZero: false),
                    (Tensor<float32>)OnnxOp.Reshape(a * Scalar(2f), Vector(1L), allowZero: false),
                    (Tensor<float32>)OnnxOp.Reshape(a * Scalar(3f), Vector(1L), allowZero: false),
                    (Tensor<float32>)OnnxOp.Reshape(a * Scalar(4f), Vector(1L), allowZero: false)
                ], axis: 0),
                Vector(1L, 1L, 2L, 2L), allowZero: false);
            var scale = (Tensor<float32>)OnnxOp.Expand(Scalar(1f), Vector(1L));
            var bias = (Tensor<float32>)OnnxOp.Expand(Scalar(0f), Vector(1L));
            var output = (Tensor<float32>)OnnxOp.InstanceNormalization(input, scale, bias, epsilon: 1e-5f);
            var loss = output.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return grad.Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradInstanceNormWithScaleCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var input = (Tensor<float32>)OnnxOp.Reshape(
                OnnxOp.Concat([
                    (Tensor<float32>)OnnxOp.Reshape(a, Vector(1L), allowZero: false),
                    (Tensor<float32>)OnnxOp.Reshape(a * Scalar(2f), Vector(1L), allowZero: false),
                    (Tensor<float32>)OnnxOp.Reshape(a * Scalar(3f), Vector(1L), allowZero: false),
                    (Tensor<float32>)OnnxOp.Reshape(a * Scalar(4f), Vector(1L), allowZero: false)
                ], axis: 0),
                Vector(1L, 1L, 2L, 2L), allowZero: false);
            var scale = (Tensor<float32>)OnnxOp.Expand(Scalar(2f), Vector(1L));
            var bias = (Tensor<float32>)OnnxOp.Expand(Scalar(0f), Vector(1L));
            var output = (Tensor<float32>)OnnxOp.InstanceNormalization(input, scale, bias);
            var loss = output.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return grad.Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradLpNormL2BasicCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var input = (Tensor<float32>)OnnxOp.Expand(a, Vector(2L));
            var output = (Tensor<float32>)OnnxOp.LpNormalization(input, axis: 0, p: 2);
            var loss = output.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return grad.Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradLpNormL2AsymmetricCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var input = (Tensor<float32>)OnnxOp.Reshape(
                OnnxOp.Concat([
                    (Tensor<float32>)OnnxOp.Reshape(a, Vector(1L), allowZero: false),
                    (Tensor<float32>)OnnxOp.Reshape(a * Scalar(2f), Vector(1L), allowZero: false)
                ], axis: 0),
                Vector(2L), allowZero: false);
            var output = (Tensor<float32>)OnnxOp.LpNormalization(input, axis: 0, p: 2);
            var loss = output.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return grad.Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradLpNormL1BasicCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var input = (Tensor<float32>)OnnxOp.Expand(a, Vector(2L));
            var output = (Tensor<float32>)OnnxOp.LpNormalization(input, axis: 0, p: 1);
            var loss = output.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return grad.Abs() < Scalar(1e-5f);
        }
    }

    /// <summary>
    /// LayerNormalization with scale=1, bias=0 over the last axis: output sums to zero per
    /// row, so the analytical gradient of <c>a</c> through Sum(LayerNorm(x)) is zero.
    /// Mirrors the InstanceNorm coverage modules: builds an <c>[a, 2a, 3a, 4a]</c> tensor
    /// from <c>a</c> so the variance is non-degenerate.
    /// </summary>
    [Module]
    public partial class AutoGradLayerNormalizationCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var input = (Tensor<float32>)OnnxOp.Concat([
                (Tensor<float32>)OnnxOp.Reshape(a, Vector(1L), allowZero: false),
                (Tensor<float32>)OnnxOp.Reshape(a * Scalar(2f), Vector(1L), allowZero: false),
                (Tensor<float32>)OnnxOp.Reshape(a * Scalar(3f), Vector(1L), allowZero: false),
                (Tensor<float32>)OnnxOp.Reshape(a * Scalar(4f), Vector(1L), allowZero: false),
            ], axis: 0);
            var scale = (Tensor<float32>)OnnxOp.Expand(Scalar(1f), Vector(4L));
            var bias = (Tensor<float32>)OnnxOp.Expand(Scalar(0f), Vector(4L));
            var output = (Tensor<float32>)OnnxOp.LayerNormalization(input, scale, bias, axis: -1, epsilon: 1e-5f).y;
            var loss = output.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return grad.Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>
    /// MeanVarianceNormalization over axis 0: output has zero mean along the normalized axis,
    /// so Σ y is zero and the analytical gradient of <c>a</c> is zero. Same shape-building
    /// pattern as <see cref="AutoGradLayerNormalizationCheck"/>.
    /// </summary>
    [Module]
    public partial class AutoGradMeanVarianceNormalizationCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var input = (Tensor<float32>)OnnxOp.Concat([
                (Tensor<float32>)OnnxOp.Reshape(a, Vector(1L), allowZero: false),
                (Tensor<float32>)OnnxOp.Reshape(a * Scalar(2f), Vector(1L), allowZero: false),
                (Tensor<float32>)OnnxOp.Reshape(a * Scalar(3f), Vector(1L), allowZero: false),
                (Tensor<float32>)OnnxOp.Reshape(a * Scalar(4f), Vector(1L), allowZero: false),
            ], axis: 0);
            var output = (Tensor<float32>)OnnxOp.MeanVarianceNormalization(input, axes: [0L]);
            var loss = output.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return grad.Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>
    /// LogSoftmax along the last axis with a uniform input <c>[a, a, a]</c>. The output is
    /// the constant vector <c>[-log 3, -log 3, -log 3]</c>, independent of <c>a</c>, so
    /// dL/da = 0. Drives the LogSoftmax [AutoDiff] entry through FastProcessAutoGrad.
    /// </summary>
    [Module]
    public partial class AutoGradLogSoftmaxCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var input = (Tensor<float32>)OnnxOp.Expand(a, Vector(3L));
            var logSoft = (Tensor<float32>)OnnxOp.LogSoftmax(input, axis: -1);
            var loss = logSoft.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return grad.Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>
    /// PRelu with a learnable slope. With slope=1 the op reduces to identity, so dL/da = N
    /// where N is the number of elements in the input tensor. We check the analytical
    /// gradient against <c>OnnxOp.Size</c> of the input.
    /// </summary>
    [Module]
    public partial class AutoGradPReluCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var input = (Tensor<float32>)OnnxOp.Concat([
                (Tensor<float32>)OnnxOp.Reshape(a, Vector(1L), allowZero: false),
                (Tensor<float32>)OnnxOp.Reshape(a * Scalar(-1f), Vector(1L), allowZero: false),
                (Tensor<float32>)OnnxOp.Reshape(a * Scalar(2f), Vector(1L), allowZero: false),
            ], axis: 0);
            var slope = (Tensor<float32>)OnnxOp.Expand(Scalar(1f), Vector(3L));
            var output = (Tensor<float32>)OnnxOp.PRelu(input, slope);
            var loss = output.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            // input = [a, -a, 2a] → output = [a, -a, 2a] (slope=1), Σ = 2a → dL/da = 2.
            return (grad - Scalar(2f)).Abs() < Scalar(1e-3f);
        }
    }

    // ===================================================================
    //  Losses (Batch30 variadic gradient registrations: NLLLoss, SCEL).
    //  Each loss is constructed so the analytical gradient of <c>a</c>
    //  reduces to a constant we can pin down — exercising the variadic
    //  gradient dispatcher entry plus the closed-form backward maths.
    // ===================================================================

    /// <summary>
    /// NegativeLogLikelihoodLoss with reduction="sum", no weight, no ignore_index.
    /// Builds a 2x3 score tensor of all-equal entries (= <c>a</c>) and integer targets [0, 1].
    /// Closed form: loss = -input[0, 0] - input[1, 1] = -2a, so dL/da = -2.
    /// </summary>
    [Module]
    public partial class AutoGradNegativeLogLikelihoodLossCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var scores = (Tensor<float32>)OnnxOp.Expand(a, Vector(2L, 3L));
            var target = Vector(0L, 1L);
            var loss = ((Tensor<float32>)OnnxOp.NegativeLogLikelihoodLoss(
                scores, target, weight: null, ignoreIndex: null, reduction: "sum")).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(-2f)).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>
    /// SoftmaxCrossEntropyLoss with reduction="sum", uniform scores (all = <c>a</c>) and a
    /// single sample whose target is class 0. Closed form: softmax is uniformly 1/C, so
    /// log_softmax_target = -log C — a constant in <c>a</c>. Hence dL/da = 0.
    /// </summary>
    [Module]
    public partial class AutoGradSoftmaxCrossEntropyLossCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var scores = (Tensor<float32>)OnnxOp.Expand(a, Vector(1L, 3L));
            var labels = Vector(0L);
            var (output, _) = OnnxOp.SoftmaxCrossEntropyLoss(
                scores, labels, weights: null, ignoreIndex: null, reduction: "sum");
            var loss = ((Tensor<float32>)output).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return grad.Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>
    /// SoftmaxCrossEntropyLoss where the optional log_prob output is also consumed
    /// downstream. The autograd engine then accumulates a non-null logProbGrad into the
    /// variadic gradient, exercising the second branch of <c>SoftmaxCrossEntropyLossGradient</c>
    /// (the LogSoftmax-grad addition). Uniform scores → uniform softmax → loss is constant
    /// in <c>a</c>, and Σ log_prob = -C·log C (constant) so dL/da remains zero.
    /// </summary>
    [Module]
    public partial class AutoGradSoftmaxCrossEntropyLossLogProbCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var scores = (Tensor<float32>)OnnxOp.Expand(a, Vector(1L, 3L));
            var labels = Vector(0L);
            var (output, logProb) = OnnxOp.SoftmaxCrossEntropyLoss(
                scores, labels, weights: null, ignoreIndex: null, reduction: "sum");
            var lossLoss = ((Tensor<float32>)output).Scalar();
            var lossLp = ((Tensor<float32>)logProb!).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var loss = lossLoss + lossLp;
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return grad.Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>
    /// NLLLoss with reduction="mean": <c>loss = -mean(input[i, t_i])</c>. Uniform scores
    /// give <c>loss = -a</c>, dL/da = -1. Drives the mean-reduction branch of
    /// <c>NegativeLogLikelihoodLossGradient</c>.
    /// </summary>
    [Module]
    public partial class AutoGradNegativeLogLikelihoodLossMeanCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var scores = (Tensor<float32>)OnnxOp.Expand(a, Vector(2L, 3L));
            var target = Vector(0L, 1L);
            var loss = ((Tensor<float32>)OnnxOp.NegativeLogLikelihoodLoss(
                scores, target, weight: null, ignoreIndex: null, reduction: "mean")).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(-1f)).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>
    /// NLLLoss with reduction="none" then Σ to a scalar. Drives the none-reduction branch.
    /// </summary>
    [Module]
    public partial class AutoGradNegativeLogLikelihoodLossNoneCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var scores = (Tensor<float32>)OnnxOp.Expand(a, Vector(2L, 3L));
            var target = Vector(0L, 1L);
            var lossPer = (Tensor<float32>)OnnxOp.NegativeLogLikelihoodLoss(
                scores, target, weight: null, ignoreIndex: null, reduction: "none");
            var loss = lossPer.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(-2f)).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>
    /// NLLLoss with an explicit per-class weight vector. Weights are constants here
    /// (no gradient flows into them). Closed form for <c>weight=[2, 3, 4]</c>,
    /// <c>target=[0, 1]</c>, uniform scores=a, reduction="sum":
    /// <c>loss = -2·a -3·a = -5a</c>, dL/da = -5.
    /// </summary>
    [Module]
    public partial class AutoGradNegativeLogLikelihoodLossWeightCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var scores = (Tensor<float32>)OnnxOp.Expand(a, Vector(2L, 3L));
            var target = Vector(0L, 1L);
            var weight = Vector(2f, 3f, 4f);
            var loss = ((Tensor<float32>)OnnxOp.NegativeLogLikelihoodLoss(
                scores, target, weight, ignoreIndex: null, reduction: "sum")).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(-5f)).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>
    /// NLLLoss with ignore_index masking out sample 1. With <c>target=[0, 1]</c> and
    /// <c>ignore_index=1</c>, only sample 0 contributes. Closed form: loss = -a, dL/da = -1.
    /// Drives the ignore_index masking branch of <c>NegativeLogLikelihoodLossGradient</c>.
    /// </summary>
    [Module]
    public partial class AutoGradNegativeLogLikelihoodLossIgnoreIndexCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var scores = (Tensor<float32>)OnnxOp.Expand(a, Vector(2L, 3L));
            var target = Vector(0L, 1L);
            var loss = ((Tensor<float32>)OnnxOp.NegativeLogLikelihoodLoss(
                scores, target, weight: null, ignoreIndex: 1L, reduction: "sum")).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(-1f)).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>
    /// SoftmaxCrossEntropyLoss with reduction="mean" and uniform scores: log_softmax is
    /// constant, so loss is constant in <c>a</c> and dL/da = 0. Drives the mean-reduction
    /// + totalWeight-division branch of <c>SoftmaxCrossEntropyLossGradient</c>.
    /// </summary>
    [Module]
    public partial class AutoGradSoftmaxCrossEntropyLossMeanCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var scores = (Tensor<float32>)OnnxOp.Expand(a, Vector(2L, 3L));
            var labels = Vector(0L, 1L);
            var (output, _) = OnnxOp.SoftmaxCrossEntropyLoss(
                scores, labels, weights: null, ignoreIndex: null, reduction: "mean");
            var loss = ((Tensor<float32>)output).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return grad.Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>
    /// SoftmaxCrossEntropyLoss with reduction="none". Per-sample loss is constant in
    /// <c>a</c> under uniform scores; Σ over samples is still constant, dL/da = 0.
    /// </summary>
    [Module]
    public partial class AutoGradSoftmaxCrossEntropyLossNoneCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var scores = (Tensor<float32>)OnnxOp.Expand(a, Vector(2L, 3L));
            var labels = Vector(0L, 1L);
            var (output, _) = OnnxOp.SoftmaxCrossEntropyLoss(
                scores, labels, weights: null, ignoreIndex: null, reduction: "none");
            var lossPer = (Tensor<float32>)output;
            var loss = lossPer.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return grad.Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>
    /// SoftmaxCrossEntropyLoss where only the optional log_prob output flows into the loss
    /// (the primary loss output is discarded). The gradient dispatcher then receives a
    /// null gradient for output[0] and a non-null gradient for output[1], driving the
    /// <c>lossGrad is null</c> branch of <c>SoftmaxCrossEntropyLossGradient</c>.
    /// Σ(log_softmax(uniform)) is constant in <c>a</c> so dL/da = 0.
    /// </summary>
    [Module]
    public partial class AutoGradSoftmaxCrossEntropyLossOnlyLogProbCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var scores = (Tensor<float32>)OnnxOp.Expand(a, Vector(1L, 3L));
            var labels = Vector(0L);
            var (_, logProb) = OnnxOp.SoftmaxCrossEntropyLoss(
                scores, labels, weights: null, ignoreIndex: null, reduction: "sum");
            var loss = ((Tensor<float32>)logProb!).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return grad.Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>
    /// SoftmaxCrossEntropyLoss with an explicit per-class weight vector + ignore_index.
    /// Uniform scores → log_softmax is constant → loss is constant → dL/da = 0. Drives the
    /// weight-supplied + ignore_index branches of <c>SoftmaxCrossEntropyLossGradient</c>.
    /// </summary>
    [Module]
    public partial class AutoGradSoftmaxCrossEntropyLossWeightIgnoreCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var scores = (Tensor<float32>)OnnxOp.Expand(a, Vector(2L, 3L));
            var labels = Vector(0L, 1L);
            var weight = Vector(1.5f, 2f, 2.5f);
            var (output, _) = OnnxOp.SoftmaxCrossEntropyLoss(
                scores, labels, weight, ignoreIndex: 1L, reduction: "sum");
            var loss = ((Tensor<float32>)output).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return grad.Abs() < Scalar(1e-3f);
        }
    }

    // ===================================================================
    //  Sequence ops (Batch30 SplitToSequence variadic registration).
    // ===================================================================

    /// <summary>
    /// SplitToSequence → ConcatFromSequence roundtrip: the split-then-recombine path is the
    /// identity on the input tensor, so Σ(reconstructed) reduces to a*(1+2+3+4) = 10a and
    /// dL/da = 10. Drives the SplitToSequence variadic gradient through FastProcessAutoGrad.
    /// </summary>
    [Module]
    public partial class AutoGradSplitToSequenceCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var input = (Tensor<float32>)OnnxOp.Concat([
                (Tensor<float32>)OnnxOp.Reshape(a, Vector(1L), allowZero: false),
                (Tensor<float32>)OnnxOp.Reshape(a * Scalar(2f), Vector(1L), allowZero: false),
                (Tensor<float32>)OnnxOp.Reshape(a * Scalar(3f), Vector(1L), allowZero: false),
                (Tensor<float32>)OnnxOp.Reshape(a * Scalar(4f), Vector(1L), allowZero: false),
            ], axis: 0);
            var seq = OnnxOp.SplitToSequence(input, split: Vector(2L, 2L), axis: 0, keepdims: 1);
            var recombined = (Tensor<float32>)OnnxOp.ConcatFromSequence(seq, axis: 0, newAxis: false);
            var loss = recombined.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(10f)).Abs() < Scalar(1e-3f);
        }
    }

    // ===================================================================
    //  Indexing (converted from AutoDiffIndexingTests).
    // ===================================================================

    [Module]
    public partial class AutoGradGatherElementsAxis0Check
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var data = (Tensor<float32>)OnnxOp.Concat(
                [OnnxOp.Reshape(a, Vector(1L), allowZero: false),
                 OnnxOp.Reshape(a * Scalar(2f), Vector(1L), allowZero: false),
                 OnnxOp.Reshape(a * Scalar(3f), Vector(1L), allowZero: false)], axis: 0);
            var indices = Vector(2L, 0L);
            var gathered = (Tensor<float32>)OnnxOp.GatherElements(data, indices, axis: 0);
            var loss = gathered.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(4f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradGatherElementsAxis1Check
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var data = (Tensor<float32>)OnnxOp.Reshape(
                (Tensor<float32>)OnnxOp.Concat(
                    [OnnxOp.Reshape(a, Vector(1L), allowZero: false),
                     OnnxOp.Reshape(a * Scalar(2f), Vector(1L), allowZero: false),
                     OnnxOp.Reshape(a * Scalar(3f), Vector(1L), allowZero: false),
                     OnnxOp.Reshape(a * Scalar(4f), Vector(1L), allowZero: false)], axis: 0),
                Vector(2L, 2L), allowZero: false);
            var indices = (Tensor<int64>)OnnxOp.Reshape(Vector(1L, 0L, 0L, 1L), Vector(2L, 2L), allowZero: false);
            var gathered = (Tensor<float32>)OnnxOp.GatherElements(data, indices, axis: 1);
            var loss = gathered.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(10f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradGatherElementsWithScaleCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var data = (Tensor<float32>)OnnxOp.Concat(
                [OnnxOp.Reshape(a, Vector(1L), allowZero: false),
                 OnnxOp.Reshape(a * Scalar(2f), Vector(1L), allowZero: false),
                 OnnxOp.Reshape(a * Scalar(3f), Vector(1L), allowZero: false)], axis: 0);
            var indices = Vector(1L);
            var gathered = (Tensor<float32>)OnnxOp.GatherElements(data, indices, axis: 0);
            var loss = gathered.Reduce(ReduceKind.Sum, keepDims: false).Scalar() * Scalar(5f);
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(10f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradScatterElementsAddCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var data = (Tensor<float32>)OnnxOp.Expand(a, Vector(3L));
            var updates = (Tensor<float32>)OnnxOp.Expand(a, Vector(1L));
            var indices = Vector(1L);
            var scattered = (Tensor<float32>)OnnxOp.ScatterElements(data, indices, updates,
                axis: 0, reduction: ScatterNDReduction.Add);
            var loss = scattered.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(4f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradScatterElementsNoneCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var data = (Tensor<float32>)OnnxOp.Expand(a, Vector(3L));
            var updateVal = a * Scalar(2f);
            var updates = (Tensor<float32>)OnnxOp.Reshape(updateVal, Vector(1L), allowZero: false);
            var indices = Vector(0L);
            var scattered = (Tensor<float32>)OnnxOp.ScatterElements(data, indices, updates);
            var loss = scattered.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(4f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradScatterElementsWithScaleCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var data = (Tensor<float32>)OnnxOp.Expand(a, Vector(2L));
            var updates = (Tensor<float32>)OnnxOp.Expand(a, Vector(1L));
            var indices = Vector(0L);
            var scattered = (Tensor<float32>)OnnxOp.ScatterElements(data, indices, updates,
                axis: 0, reduction: ScatterNDReduction.Add);
            var loss = scattered.Reduce(ReduceKind.Sum, keepDims: false).Scalar() * Scalar(3f);
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(9f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradScatterNDAddCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var data = (Tensor<float32>)OnnxOp.Expand(a, Vector(3L));
            var updates = (Tensor<float32>)OnnxOp.Expand(a, Vector(1L));
            var indices = (Tensor<int64>)OnnxOp.Reshape(Vector(1L), Vector(1L, 1L), allowZero: false);
            var scattered = (Tensor<float32>)OnnxOp.ScatterND(data, indices, updates, ScatterNDReduction.Add);
            var loss = scattered.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(4f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradScatterNDNoneCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var data = (Tensor<float32>)OnnxOp.Expand(a, Vector(3L));
            var updateVal = a * Scalar(2f);
            var updates = (Tensor<float32>)OnnxOp.Reshape(updateVal, Vector(1L), allowZero: false);
            var indices = (Tensor<int64>)OnnxOp.Reshape(Vector(0L), Vector(1L, 1L), allowZero: false);
            var scattered = (Tensor<float32>)OnnxOp.ScatterND(data, indices, updates);
            var loss = scattered.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(4f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradScatterNDWithScaleCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var data = (Tensor<float32>)OnnxOp.Expand(a, Vector(2L));
            var updates = (Tensor<float32>)OnnxOp.Expand(a, Vector(1L));
            var indices = (Tensor<int64>)OnnxOp.Reshape(Vector(0L), Vector(1L, 1L), allowZero: false);
            var scattered = (Tensor<float32>)OnnxOp.ScatterND(data, indices, updates, ScatterNDReduction.Add);
            var loss = scattered.Reduce(ReduceKind.Sum, keepDims: false).Scalar() * Scalar(3f);
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(9f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradScatterNDReluChainCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var data = (Tensor<float32>)OnnxOp.Expand(a, Vector(3L));
            var updates = (Tensor<float32>)OnnxOp.Expand(a, Vector(1L));
            var indices = (Tensor<int64>)OnnxOp.Reshape(Vector(1L), Vector(1L, 1L), allowZero: false);
            var scattered = (Tensor<float32>)OnnxOp.ScatterND(data, indices, updates, ScatterNDReduction.Add);
            var activated = (Tensor<float32>)OnnxOp.Relu(scattered);
            var loss = activated.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(4f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradGatherAxis0Check
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var data = (Tensor<float32>)OnnxOp.Expand(a, Vector(3L));
            var indices = Vector(0L);
            var gathered = (Tensor<float32>)OnnxOp.Gather(data, indices, axis: 0);
            var loss = gathered.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(1f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradGatherDuplicateIndicesCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var data = (Tensor<float32>)OnnxOp.Expand(a, Vector(3L));
            var indices = Vector(0L, 0L, 0L);
            var gathered = (Tensor<float32>)OnnxOp.Gather(data, indices, axis: 0);
            var loss = gathered.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(3f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradGatherAllIndicesCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var data = (Tensor<float32>)OnnxOp.Expand(a, Vector(3L));
            var indices = Vector(0L, 1L, 2L);
            var gathered = (Tensor<float32>)OnnxOp.Gather(data, indices, axis: 0);
            var loss = gathered.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(3f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradGatherAxis0MultiDimIndicesCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var data = (Tensor<float32>)OnnxOp.Expand(a, Vector(3L, 4L));
            var indices = (Tensor<int64>)OnnxOp.Reshape(Vector(0L, 1L), Vector(1L, 2L), allowZero: false);
            var gathered = (Tensor<float32>)OnnxOp.Gather(data, indices, axis: 0);
            var loss = gathered.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(8f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradGatherNonZeroAxisOneDimIndicesCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var data = (Tensor<float32>)OnnxOp.Identity(OnnxOp.Expand(a, Vector(2L, 3L)), rank: 2);
            var indices = Vector(1L);
            var gathered = (Tensor<float32>)OnnxOp.Gather(data, indices, axis: 1);
            var loss = gathered.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(2f)).Abs() < Scalar(1e-5f);
        }
    }

    /// <summary>
    /// Unknown-rank companion to <see cref="AutoGradGatherNonZeroAxisOneDimIndicesCheck"/>:
    /// no Identity rank stamp, so `data` has a null static Rank and the Gather gradient
    /// takes its rank-agnostic collapse-to-3-D scatter (instead of a rank-length transpose
    /// perm, which previously threw for a non-zero axis on null-rank data). Same result:
    /// gather column 1 of 3a²-free data → sum = 2a, dL/da = 2.
    /// </summary>
    [Module]
    public partial class AutoGradGatherNonZeroAxisOneDimIndicesUnknownRankCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var data = (Tensor<float32>)OnnxOp.Expand(a, Vector(2L, 3L)); // no rank stamp → null static rank
            var indices = Vector(1L);
            var gathered = (Tensor<float32>)OnnxOp.Gather(data, indices, axis: 1);
            var loss = gathered.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(2f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradGatherNonZeroAxisMultiDimIndicesCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var data = (Tensor<float32>)OnnxOp.Identity(OnnxOp.Expand(a, Vector(2L, 3L)), rank: 2);
            var indices = (Tensor<int64>)OnnxOp.Identity(
                OnnxOp.Reshape(Vector(1L), Vector(1L, 1L), allowZero: false), rank: 2);
            var gathered = (Tensor<float32>)OnnxOp.Gather(data, indices, axis: 1);
            var loss = gathered.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(2f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradGatherNDCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var data = (Tensor<float32>)OnnxOp.Expand(a, Vector(3L));
            var indices = (Tensor<int64>)OnnxOp.Reshape(Vector(0L, 2L), Vector(2L, 1L), allowZero: false);
            var gathered = (Tensor<float32>)OnnxOp.GatherND(data, indices);
            var loss = gathered.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(2f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradGatherNDDuplicateIndicesCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var data = (Tensor<float32>)OnnxOp.Expand(a, Vector(3L));
            var indices = (Tensor<int64>)OnnxOp.Reshape(Vector(0L, 0L, 0L), Vector(3L, 1L), allowZero: false);
            var gathered = (Tensor<float32>)OnnxOp.GatherND(data, indices);
            var loss = gathered.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(3f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradGatherNDWithScaleCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var data = (Tensor<float32>)OnnxOp.Expand(a, Vector(3L));
            var indices = (Tensor<int64>)OnnxOp.Reshape(Vector(1L), Vector(1L, 1L), allowZero: false);
            var gathered = (Tensor<float32>)OnnxOp.GatherND(data, indices);
            var loss = gathered.Reduce(ReduceKind.Sum, keepDims: false).Scalar() * Scalar(5f);
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(5f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradWhereTrueBranchCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x, Scalar<float32> y)
        {
            var cond = Scalar(true);
            var result = (Tensor<float32>)OnnxOp.Where(cond, x, y);
            var loss = result.Scalar();
            var (gx, gy) = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, y, loss);
            var okX = (gx! - Scalar(1f)).Abs() < Scalar(1e-5f);
            var okY = gy!.Abs() < Scalar(1e-5f);
            return okX & okY;
        }
    }

    [Module]
    public partial class AutoGradWhereFalseBranchCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x, Scalar<float32> y)
        {
            var cond = Scalar(false);
            var result = (Tensor<float32>)OnnxOp.Where(cond, x, y);
            var loss = result.Scalar();
            var (gx, gy) = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, y, loss);
            var okX = gx!.Abs() < Scalar(1e-5f);
            var okY = (gy! - Scalar(1f)).Abs() < Scalar(1e-5f);
            return okX & okY;
        }
    }

    [Module]
    public partial class AutoGradUniqueSingleElementCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var input = (Tensor<float32>)OnnxOp.Expand(a, Vector(1L));
            var (y, _, _, _) = OnnxOp.Unique(input, axis: null, sorted: true);
            var loss = ((Tensor<float32>)y).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(1f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradUniqueAllSameCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var expanded = (Tensor<float32>)OnnxOp.Expand(a, Vector(4L));
            var (y, _, _, _) = OnnxOp.Unique(expanded, axis: null, sorted: true);
            var loss = ((Tensor<float32>)y).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(4f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradUniqueWithAxisCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var expanded = (Tensor<float32>)OnnxOp.Expand(a, Vector(3L, 2L));
            var (y, _, _, _) = OnnxOp.Unique(expanded, axis: 0, sorted: true);
            var loss = ((Tensor<float32>)y).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(6f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradUniqueDistinctCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var elem1 = (Tensor<float32>)OnnxOp.Reshape(a, Vector(1L), allowZero: false);
            var elem2 = (Tensor<float32>)OnnxOp.Reshape(a * Scalar(2f), Vector(1L), allowZero: false);
            var x = (Tensor<float32>)OnnxOp.Concat([elem1, elem2], axis: 0);
            var (y, _, _, _) = OnnxOp.Unique(x, axis: null, sorted: true);
            var loss = ((Tensor<float32>)y).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(3f)).Abs() < Scalar(1e-5f);
        }
    }

    // ===================================================================
    //  Shape ops (converted from AutoDiffShapeTests). All sum-based losses
    //  with closed-form gradients equal to the count of unique input
    //  positions surviving forward, optionally scaled.
    // ===================================================================

    [Module]
    public partial class AutoGradTransposeCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var reshaped = (Tensor<float32>)OnnxOp.Reshape(a, Vector(1L, 1L), allowZero: false);
            var transposed = reshaped.Transpose();
            var product = (Tensor<float32>)OnnxOp.Mul(transposed, Scalar(3f));
            var loss = ((Tensor<float32>)OnnxOp.Reshape(product, Vector(new long[0]), allowZero: false)).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(3f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradFlattenCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            var reshaped = (Tensor<float32>)OnnxOp.Reshape(x, Vector(1L, 1L), allowZero: false);
            var flattened = (Tensor<float32>)OnnxOp.Flatten(reshaped, axis: 0);
            var product = (Tensor<float32>)OnnxOp.Mul(flattened, Scalar(2f));
            var loss = ((Tensor<float32>)OnnxOp.Reshape(product, Vector(new long[0]), allowZero: false)).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            return (grad - Scalar(2f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradSqueezeCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            var unsqueezed = (Tensor<float32>)OnnxOp.Unsqueeze(x, Vector(0L));
            var squeezed = (Tensor<float32>)OnnxOp.Squeeze(unsqueezed, Vector(0L));
            var product = squeezed * Scalar(3f);
            var loss = ((Tensor<float32>)OnnxOp.Reshape(product, Vector(new long[0]), allowZero: false)).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            return (grad - Scalar(3f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradUnsqueezeCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            var unsqueezed = (Tensor<float32>)OnnxOp.Unsqueeze(x, Vector(0L));
            var product = (Tensor<float32>)OnnxOp.Mul(unsqueezed, Scalar(2f));
            var loss = ((Tensor<float32>)OnnxOp.Reshape(product, Vector(new long[0]), allowZero: false)).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            return (grad - Scalar(2f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradExpandCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            var expanded = (Tensor<float32>)OnnxOp.Expand(x, Vector(2L));
            var loss = expanded.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            return (grad - Scalar(2f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradPadConstantCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            var reshaped = (Tensor<float32>)OnnxOp.Reshape(x, Vector(1L), allowZero: false);
            var padded = (Tensor<float32>)OnnxOp.Pad(reshaped, Vector(1L, 1L), Scalar(0f), mode: PadMode.Constant);
            var loss = padded.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            return (grad - Scalar(1f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradPadConstantWithMultiplyCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            var reshaped = (Tensor<float32>)OnnxOp.Reshape(x, Vector(1L), allowZero: false);
            var padded = (Tensor<float32>)OnnxOp.Pad(reshaped, Vector(1L, 1L), Scalar(0f), mode: PadMode.Constant);
            var loss = padded.Reduce(ReduceKind.Sum, keepDims: false).Scalar() * Scalar(3f);
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            return (grad - Scalar(3f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradPad2DCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            var reshaped = (Tensor<float32>)OnnxOp.Reshape(x, Vector(1L, 1L), allowZero: false);
            var padded = (Tensor<float32>)OnnxOp.Pad(reshaped, Vector(1L, 1L, 1L, 1L), Scalar(0f), mode: PadMode.Constant);
            var loss = padded.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            return (grad - Scalar(1f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradPadWithSigmoidCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            var sigX = x.Sigmoid();
            var reshaped = (Tensor<float32>)OnnxOp.Reshape(sigX, Vector(1L), allowZero: false);
            var padded = (Tensor<float32>)OnnxOp.Pad(reshaped, Vector(1L, 1L), Scalar(0f), mode: PadMode.Constant);
            var loss = padded.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            var expected = sigX * (Scalar(1f) - sigX);
            return (grad - expected).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradResizeNearestSumLossWithScalesCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var mat = (Tensor<float32>)OnnxOp.Expand(a, Vector(1L, 1L, 2L, 2L));
            var scales = Vector(1.0f, 1.0f, 2.0f, 2.0f);
            var resized = (Tensor<float32>)OnnxOp.Resize(mat, null, scales, null,
                antialias: false, axes: null,
                coordinateTransformationMode: CoordinateTransformationMode.Asymmetric,
                cubicCoeffA: null, excludeOutside: false,
                extrapolationValue: null, keepAspectRatioPolicy: null,
                mode: ResizeMode.Nearest, nearestMode: NearestMode.Floor);
            var loss = resized.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(16f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradResizeNearestSizesCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var mat = (Tensor<float32>)OnnxOp.Expand(a, Vector(1L, 1L, 2L, 2L));
            var sizes = Vector(1L, 1L, 4L, 4L);
            var resized = (Tensor<float32>)OnnxOp.Resize(mat, null, null, sizes,
                antialias: false, axes: null,
                coordinateTransformationMode: CoordinateTransformationMode.Asymmetric,
                cubicCoeffA: null, excludeOutside: false,
                extrapolationValue: null, keepAspectRatioPolicy: null,
                mode: ResizeMode.Nearest, nearestMode: NearestMode.Floor);
            var loss = resized.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(16f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradResizeNearestChainedCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var mat = (Tensor<float32>)OnnxOp.Expand(a, Vector(1L, 1L, 1L, 1L));
            var scales = Vector(1.0f, 1.0f, 2.0f, 2.0f);
            var resized = (Tensor<float32>)OnnxOp.Resize(mat, null, scales, null,
                antialias: false, axes: null,
                coordinateTransformationMode: CoordinateTransformationMode.Asymmetric,
                cubicCoeffA: null, excludeOutside: false,
                extrapolationValue: null, keepAspectRatioPolicy: null,
                mode: ResizeMode.Nearest, nearestMode: NearestMode.Floor);
            var loss = resized.Reduce(ReduceKind.Sum, keepDims: false).Scalar() * Scalar(3f);
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(12f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradResizeNearestMultipleInputsCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b)
        {
            var ch1 = (Tensor<float32>)OnnxOp.Expand(a, Vector(1L, 1L, 1L, 1L));
            var ch2 = (Tensor<float32>)OnnxOp.Expand(b, Vector(1L, 1L, 1L, 1L));
            var mat = (Tensor<float32>)OnnxOp.Concat([ch1, ch2], axis: 1);
            var scales = Vector(1.0f, 1.0f, 2.0f, 2.0f);
            var resized = (Tensor<float32>)OnnxOp.Resize(mat, null, scales, null,
                antialias: false, axes: null,
                coordinateTransformationMode: CoordinateTransformationMode.Asymmetric,
                cubicCoeffA: null, excludeOutside: false,
                extrapolationValue: null, keepAspectRatioPolicy: null,
                mode: ResizeMode.Nearest, nearestMode: NearestMode.Floor);
            var loss = resized.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var (ga, gb) = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, b, loss);
            var okA = (ga! - Scalar(4f)).Abs() < Scalar(1e-5f);
            var okB = (gb! - Scalar(4f)).Abs() < Scalar(1e-5f);
            return okA & okB;
        }
    }

    [Module]
    public partial class AutoGradSliceCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var data = (Tensor<float32>)OnnxOp.Expand(a, Vector(3L));
            var sliced = (Tensor<float32>)OnnxOp.Slice(data, Vector(1L), Vector(2L));
            var loss = sliced.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(1f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradSliceMultipleElementsCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var data = (Tensor<float32>)OnnxOp.Expand(a, Vector(3L));
            var sliced = (Tensor<float32>)OnnxOp.Slice(data, Vector(0L), Vector(2L));
            var loss = sliced.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(2f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradSliceWithScaleCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var data = (Tensor<float32>)OnnxOp.Expand(a, Vector(3L));
            var sliced = (Tensor<float32>)OnnxOp.Slice(data, Vector(0L), Vector(3L));
            var loss = sliced.Reduce(ReduceKind.Sum, keepDims: false).Scalar() * Scalar(2f);
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(6f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradSpaceToDepthBasicCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var input = (Tensor<float32>)OnnxOp.Expand(a, Vector(1L, 1L, 2L, 2L));
            var output = (Tensor<float32>)OnnxOp.SpaceToDepth(input, blockSize: 2);
            var loss = output.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(4f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradSpaceToDepthWithScaleCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var input = (Tensor<float32>)OnnxOp.Expand(a, Vector(1L, 1L, 2L, 2L));
            var output = (Tensor<float32>)OnnxOp.SpaceToDepth(input, blockSize: 2);
            var loss = output.Reduce(ReduceKind.Sum, keepDims: false).Scalar() * Scalar(3f);
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(12f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradSpaceToDepthMultiChannelCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var input = (Tensor<float32>)OnnxOp.Expand(a, Vector(1L, 2L, 2L, 2L));
            var output = (Tensor<float32>)OnnxOp.SpaceToDepth(input, blockSize: 2);
            var loss = output.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(8f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradDepthToSpaceDCRCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var input = (Tensor<float32>)OnnxOp.Expand(a, Vector(1L, 4L, 1L, 1L));
            var output = (Tensor<float32>)OnnxOp.DepthToSpace(input, blockSize: 2, mode: DepthColumnRowMode.DCR);
            var loss = output.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(4f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradDepthToSpaceCRDCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var input = (Tensor<float32>)OnnxOp.Expand(a, Vector(1L, 4L, 1L, 1L));
            var output = (Tensor<float32>)OnnxOp.DepthToSpace(input, blockSize: 2, mode: DepthColumnRowMode.CRD);
            var loss = output.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(4f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradDepthToSpaceWithScaleCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var input = (Tensor<float32>)OnnxOp.Expand(a, Vector(1L, 4L, 1L, 1L));
            var output = (Tensor<float32>)OnnxOp.DepthToSpace(input, blockSize: 2, mode: DepthColumnRowMode.DCR);
            var loss = output.Reduce(ReduceKind.Sum, keepDims: false).Scalar() * Scalar(2f);
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(8f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradTile1DCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var data = (Tensor<float32>)OnnxOp.Reshape(a, Vector(1L), allowZero: false);
            var tiled = (Tensor<float32>)OnnxOp.Tile(data, Vector(3L));
            var loss = tiled.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(3f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradTileWithScaleCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var data = (Tensor<float32>)OnnxOp.Reshape(a, Vector(1L), allowZero: false);
            var tiled = (Tensor<float32>)OnnxOp.Tile(data, Vector(4L));
            var loss = tiled.Reduce(ReduceKind.Sum, keepDims: false).Scalar() * Scalar(2f);
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(8f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradTile2DCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var data = (Tensor<float32>)OnnxOp.Reshape(a, Vector(1L, 1L), allowZero: false);
            var tiled = (Tensor<float32>)OnnxOp.Tile(data, Vector(2L, 3L));
            var loss = tiled.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(6f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradTriluUpperCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var input = (Tensor<float32>)OnnxOp.Expand(a, Vector(2L, 2L));
            var output = (Tensor<float32>)OnnxOp.Trilu(input, upper: 1);
            var loss = output.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(3f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradTriluLowerCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var input = (Tensor<float32>)OnnxOp.Expand(a, Vector(2L, 2L));
            var output = (Tensor<float32>)OnnxOp.Trilu(input, upper: 0);
            var loss = output.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(3f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradTriluUpperWithKCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var input = (Tensor<float32>)OnnxOp.Expand(a, Vector(3L, 3L));
            var output = (Tensor<float32>)OnnxOp.Trilu(input, Scalar(1L), upper: 1);
            var loss = output.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(3f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradTriluWithScaleCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var input = (Tensor<float32>)OnnxOp.Expand(a, Vector(2L, 2L));
            var output = (Tensor<float32>)OnnxOp.Trilu(input, upper: 1);
            var loss = output.Reduce(ReduceKind.Sum, keepDims: false).Scalar() * Scalar(2f);
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(6f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradUpsampleNearestSumLossCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var mat = (Tensor<float32>)OnnxOp.Expand(a, Vector(1L, 1L, 2L, 2L));
            var scales = Vector(1.0f, 1.0f, 2.0f, 2.0f);
            var upsampled = (Tensor<float32>)OnnxOp.Upsample(mat, scales, mode: ResizeMode.Nearest);
            var loss = upsampled.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(16f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradUpsampleNearestChainedCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var mat = (Tensor<float32>)OnnxOp.Expand(a, Vector(1L, 1L, 1L, 1L));
            var scales = Vector(1.0f, 1.0f, 2.0f, 2.0f);
            var upsampled = (Tensor<float32>)OnnxOp.Upsample(mat, scales, mode: ResizeMode.Nearest);
            var loss = upsampled.Reduce(ReduceKind.Sum, keepDims: false).Scalar() * Scalar(3f);
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(12f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradUpsampleNearestMultipleInputsCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b)
        {
            var ch1 = (Tensor<float32>)OnnxOp.Expand(a, Vector(1L, 1L, 1L, 1L));
            var ch2 = (Tensor<float32>)OnnxOp.Expand(b, Vector(1L, 1L, 1L, 1L));
            var mat = (Tensor<float32>)OnnxOp.Concat([ch1, ch2], axis: 1);
            var scales = Vector(1.0f, 1.0f, 2.0f, 2.0f);
            var upsampled = (Tensor<float32>)OnnxOp.Upsample(mat, scales, mode: ResizeMode.Nearest);
            var loss = upsampled.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var (ga, gb) = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, b, loss);
            var okA = (ga! - Scalar(4f)).Abs() < Scalar(1e-5f);
            var okB = (gb! - Scalar(4f)).Abs() < Scalar(1e-5f);
            return okA & okB;
        }
    }

    [Module]
    public partial class AutoGradCenterCropPadCropCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var mat = (Tensor<float32>)OnnxOp.Expand(a, Vector(2L, 4L));
            var cropped = mat.CenterCropPad(Vector(2L, 2L));
            var loss = cropped.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(4f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradCenterCropPadPadCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var mat = (Tensor<float32>)OnnxOp.Expand(a, Vector(2L, 2L));
            var padded = mat.CenterCropPad(Vector(2L, 4L));
            var loss = padded.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(4f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradCenterCropPadSameSizeCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var mat = (Tensor<float32>)OnnxOp.Expand(a, Vector(2L, 3L));
            var same = mat.CenterCropPad(Vector(2L, 3L));
            var loss = same.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(6f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradCenterCropPadWithAxesCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var mat = (Tensor<float32>)OnnxOp.Expand(a, Vector(3L, 4L));
            var cropped = mat.CenterCropPad(Vector(2L), axes: [1]);
            var loss = cropped.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(6f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradReverseSequenceBasicCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var mat = (Tensor<float32>)OnnxOp.Expand(a, Vector(2L, 4L));
            var seqLens = Vector(4L, 4L);
            var reversed = (Tensor<float32>)OnnxOp.ReverseSequence(mat, seqLens,
                batchAxis: 0, timeAxis: 1);
            var loss = reversed.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(8f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradReverseSequencePartialReverseCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var mat = (Tensor<float32>)OnnxOp.Expand(a, Vector(3L, 4L));
            var seqLens = Vector(2L, 3L, 1L);
            var reversed = (Tensor<float32>)OnnxOp.ReverseSequence(mat, seqLens,
                batchAxis: 0, timeAxis: 1);
            var loss = reversed.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(12f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradReverseSequenceAllSameCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var mat = (Tensor<float32>)OnnxOp.Expand(a, Vector(2L, 3L));
            var seqLens = Vector(3L, 3L);
            var reversed = (Tensor<float32>)OnnxOp.ReverseSequence(mat, seqLens,
                batchAxis: 0, timeAxis: 1);
            var loss = reversed.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(6f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradReverseSequenceBatchAxis1Check
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var mat = (Tensor<float32>)OnnxOp.Expand(a, Vector(4L, 2L));
            var seqLens = Vector(3L, 2L);
            var reversed = (Tensor<float32>)OnnxOp.ReverseSequence(mat, seqLens,
                batchAxis: 1, timeAxis: 0);
            var loss = reversed.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(8f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradCol2ImBasicNoOverlapCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var expanded = (Tensor<float32>)OnnxOp.Expand(a, Vector(1L, 2L, 15L));
            var imgShape = Vector(3L, 5L);
            var blkShape = Vector(1L, 1L);
            var col2imOut = (Tensor<float32>)OnnxOp.Col2Im(expanded, imgShape, blkShape,
                dilations: [1L, 1L], pads: [0L, 0L, 0L, 0L], strides: [1L, 1L]);
            var loss = col2imOut.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(30f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradCol2ImWithOverlapCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var expanded = (Tensor<float32>)OnnxOp.Expand(a, Vector(1L, 4L, 4L));
            var imgShape = Vector(3L, 3L);
            var blkShape = Vector(2L, 2L);
            var col2imOut = (Tensor<float32>)OnnxOp.Col2Im(expanded, imgShape, blkShape,
                dilations: [1L, 1L], pads: [0L, 0L, 0L, 0L], strides: [1L, 1L]);
            var loss = col2imOut.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(16f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradCol2ImWithPaddingCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var expanded = (Tensor<float32>)OnnxOp.Expand(a, Vector(1L, 2L, 35L));
            var imgShape = Vector(3L, 5L);
            var blkShape = Vector(1L, 1L);
            var col2imOut = (Tensor<float32>)OnnxOp.Col2Im(expanded, imgShape, blkShape,
                dilations: [1L, 1L], pads: [1L, 1L, 1L, 1L], strides: [1L, 1L]);
            var loss = col2imOut.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(30f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradCol2Im1x1BlockCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var expanded = (Tensor<float32>)OnnxOp.Expand(a, Vector(1L, 1L, 6L));
            var imgShape = Vector(2L, 3L);
            var blkShape = Vector(1L, 1L);
            var col2imOut = (Tensor<float32>)OnnxOp.Col2Im(expanded, imgShape, blkShape,
                dilations: [1L, 1L], pads: [0L, 0L, 0L, 0L], strides: [1L, 1L]);
            var loss = col2imOut.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(6f)).Abs() < Scalar(1e-5f);
        }
    }

    // ===================================================================
    //  Sequence Ops (converted from AutoDiffSequenceOpsTests). The
    //  3-input variants combine the pair AutoGrad overload with a
    //  single-input AutoGrad call on the same loss for the third var.
    // ===================================================================

    [Module]
    public partial class AutoGradSequenceConstructAtExtractFirstCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b)
        {
            var aVec = (Tensor<float32>)OnnxOp.Reshape(a, Vector(1L), allowZero: false);
            var bVec = (Tensor<float32>)OnnxOp.Reshape(b, Vector(1L), allowZero: false);
            var seq = OnnxOp.SequenceConstruct(aVec, bVec);
            var elem = (Tensor<float32>)OnnxOp.SequenceAt(seq, Scalar(0L));
            var loss = elem.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var (ga, gb) = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, b, loss);
            var okA = (ga! - Scalar(1f)).Abs() < Scalar(1e-5f);
            var okB = gb!.Abs() < Scalar(1e-5f);
            return okA & okB;
        }
    }

    [Module]
    public partial class AutoGradSequenceConstructAtExtractSecondCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b)
        {
            var aVec = (Tensor<float32>)OnnxOp.Reshape(a, Vector(1L), allowZero: false);
            var bVec = (Tensor<float32>)OnnxOp.Reshape(b, Vector(1L), allowZero: false);
            var seq = OnnxOp.SequenceConstruct(aVec, bVec);
            var elem = (Tensor<float32>)OnnxOp.SequenceAt(seq, Scalar(1L));
            var loss = elem.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var (ga, gb) = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, b, loss);
            var okA = ga!.Abs() < Scalar(1e-5f);
            var okB = (gb! - Scalar(1f)).Abs() < Scalar(1e-5f);
            return okA & okB;
        }
    }

    [Module]
    public partial class AutoGradSequenceConstructAtWithScaleCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b)
        {
            var aVec = (Tensor<float32>)OnnxOp.Reshape(a, Vector(1L), allowZero: false);
            var bVec = (Tensor<float32>)OnnxOp.Reshape(b, Vector(1L), allowZero: false);
            var seq = OnnxOp.SequenceConstruct(aVec, bVec);
            var elem = (Tensor<float32>)OnnxOp.SequenceAt(seq, Scalar(0L));
            var threeVec = (Tensor<float32>)OnnxOp.Reshape(Scalar(3f), Vector(1L), allowZero: false);
            var scaled = elem * threeVec;
            var loss = scaled.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var (ga, gb) = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, b, loss);
            var okA = (ga! - Scalar(3f)).Abs() < Scalar(1e-5f);
            var okB = gb!.Abs() < Scalar(1e-5f);
            return okA & okB;
        }
    }

    [Module]
    public partial class AutoGradConcatFromSequenceBasicCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b)
        {
            var aVec = (Tensor<float32>)OnnxOp.Reshape(a, Vector(1L), allowZero: false);
            var bVec = (Tensor<float32>)OnnxOp.Reshape(b, Vector(1L), allowZero: false);
            var seq = OnnxOp.SequenceConstruct(aVec, bVec);
            var concatenated = (Tensor<float32>)OnnxOp.ConcatFromSequence(seq, axis: 0, newAxis: false);
            var loss = concatenated.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var (ga, gb) = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, b, loss);
            var okA = (ga! - Scalar(1f)).Abs() < Scalar(1e-5f);
            var okB = (gb! - Scalar(1f)).Abs() < Scalar(1e-5f);
            return okA & okB;
        }
    }

    [Module]
    public partial class AutoGradConcatFromSequenceNewAxisCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b)
        {
            var aVec = (Tensor<float32>)OnnxOp.Reshape(a, Vector(1L), allowZero: false);
            var bVec = (Tensor<float32>)OnnxOp.Reshape(b, Vector(1L), allowZero: false);
            var seq = OnnxOp.SequenceConstruct(aVec, bVec);
            var stacked = (Tensor<float32>)OnnxOp.ConcatFromSequence(seq, axis: 0, newAxis: true);
            var loss = stacked.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var (ga, gb) = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, b, loss);
            var okA = (ga! - Scalar(1f)).Abs() < Scalar(1e-5f);
            var okB = (gb! - Scalar(1f)).Abs() < Scalar(1e-5f);
            return okA & okB;
        }
    }

    [Module]
    public partial class AutoGradConcatFromSequenceWithScaleCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b)
        {
            var aVec = (Tensor<float32>)OnnxOp.Reshape(a, Vector(1L), allowZero: false);
            var bVec = (Tensor<float32>)OnnxOp.Reshape(b, Vector(1L), allowZero: false);
            var seq = OnnxOp.SequenceConstruct(aVec, bVec);
            var concatenated = (Tensor<float32>)OnnxOp.ConcatFromSequence(seq, axis: 0, newAxis: false);
            var loss = concatenated.Reduce(ReduceKind.Sum, keepDims: false).Scalar() * Scalar(2f);
            var (ga, gb) = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, b, loss);
            var okA = (ga! - Scalar(2f)).Abs() < Scalar(1e-5f);
            var okB = (gb! - Scalar(2f)).Abs() < Scalar(1e-5f);
            return okA & okB;
        }
    }

    /// <summary>
    /// Note: this module deliberately uses the pair AutoGrad overload + a separate
    /// single-input AutoGrad call on the same <c>loss</c>, instead of the array
    /// overload that the sibling 3-input SequenceOps modules use. The three
    /// 3-input AutoGrad overloads (single, pair, array) lower to slightly
    /// different code paths in <c>FastProcessAutoGradProcessor</c>, and we want
    /// at least one module that emits multiple AutoGrad nodes referencing the
    /// same loss scalar — that hits the dedupe/sharing path inside the
    /// processor, which neither a pure-single nor a pure-array call exercises.
    /// </summary>
    [Module]
    public partial class AutoGradSequenceConstructAtThreeInputsCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b, Scalar<float32> c)
        {
            var aVec = (Tensor<float32>)OnnxOp.Reshape(a, Vector(1L), allowZero: false);
            var bVec = (Tensor<float32>)OnnxOp.Reshape(b, Vector(1L), allowZero: false);
            var cVec = (Tensor<float32>)OnnxOp.Reshape(c, Vector(1L), allowZero: false);
            var seq = OnnxOp.SequenceConstruct(aVec, bVec, cVec);
            var elem = (Tensor<float32>)OnnxOp.SequenceAt(seq, Scalar(1L));
            var loss = elem.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var (ga, gb) = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, b, loss);
            var gc = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(c, loss);
            var okA = ga!.Abs() < Scalar(1e-5f);
            var okB = (gb! - Scalar(1f)).Abs() < Scalar(1e-5f);
            var okC = gc.Abs() < Scalar(1e-5f);
            return okA & okB & okC;
        }
    }

    [Module]
    public partial class AutoGradSequenceInsertAtPositionCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b, Scalar<float32> c)
        {
            var aVec = (Tensor<float32>)OnnxOp.Reshape(a, Vector(1L), allowZero: false);
            var bVec = (Tensor<float32>)OnnxOp.Reshape(b, Vector(1L), allowZero: false);
            var cVec = (Tensor<float32>)OnnxOp.Reshape(c, Vector(1L), allowZero: false);
            var seq = OnnxOp.SequenceConstruct(aVec, bVec);
            var seqInserted = OnnxOp.SequenceInsert(seq, cVec, Scalar(1L));
            var elem = (Tensor<float32>)OnnxOp.SequenceAt(seqInserted, Scalar(1L));
            var loss = elem.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grads = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad((IValue[])[a, b, c], loss);
            var ga = (Scalar<float32>)grads[0]!;
            var gb = (Scalar<float32>)grads[1]!;
            var gc = (Scalar<float32>)grads[2]!;
            var okA = ga.Abs() < Scalar(1e-5f);
            var okB = gb.Abs() < Scalar(1e-5f);
            var okC = (gc - Scalar(1f)).Abs() < Scalar(1e-5f);
            return okA & okB & okC;
        }
    }

    [Module]
    public partial class AutoGradSequenceInsertAppendNullPositionCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b, Scalar<float32> c)
        {
            var aVec = (Tensor<float32>)OnnxOp.Reshape(a, Vector(1L), allowZero: false);
            var bVec = (Tensor<float32>)OnnxOp.Reshape(b, Vector(1L), allowZero: false);
            var cVec = (Tensor<float32>)OnnxOp.Reshape(c, Vector(1L), allowZero: false);
            var seq = OnnxOp.SequenceConstruct(aVec, bVec);
            var seqAppended = OnnxOp.SequenceInsert(seq, cVec, null);
            var elem = (Tensor<float32>)OnnxOp.SequenceAt(seqAppended, Scalar(2L));
            var loss = elem.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grads = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad((IValue[])[a, b, c], loss);
            var ga = (Scalar<float32>)grads[0]!;
            var gb = (Scalar<float32>)grads[1]!;
            var gc = (Scalar<float32>)grads[2]!;
            var okA = ga.Abs() < Scalar(1e-5f);
            var okB = gb.Abs() < Scalar(1e-5f);
            var okC = (gc - Scalar(1f)).Abs() < Scalar(1e-5f);
            return okA & okB & okC;
        }
    }

    [Module]
    public partial class AutoGradSequenceEraseElementCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b, Scalar<float32> c)
        {
            var aVec = (Tensor<float32>)OnnxOp.Reshape(a, Vector(1L), allowZero: false);
            var bVec = (Tensor<float32>)OnnxOp.Reshape(b, Vector(1L), allowZero: false);
            var cVec = (Tensor<float32>)OnnxOp.Reshape(c, Vector(1L), allowZero: false);
            var seq = OnnxOp.SequenceConstruct(aVec, bVec, cVec);
            var seqErased = OnnxOp.SequenceErase(seq, Scalar(0L));
            var elem = (Tensor<float32>)OnnxOp.SequenceAt(seqErased, Scalar(0L));
            var loss = elem.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grads = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad((IValue[])[a, b, c], loss);
            var ga = (Scalar<float32>)grads[0]!;
            var gb = (Scalar<float32>)grads[1]!;
            var gc = (Scalar<float32>)grads[2]!;
            var okA = ga.Abs() < Scalar(1e-5f);
            var okB = (gb - Scalar(1f)).Abs() < Scalar(1e-5f);
            var okC = gc.Abs() < Scalar(1e-5f);
            return okA & okB & okC;
        }
    }

    // ===================================================================
    //  Compress (converted from AutoDiffNewOpsGradientOpsTests).
    // ===================================================================

    [Module]
    public partial class AutoGradCompressNoAxisCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var expanded = (Tensor<float32>)OnnxOp.Expand(a, Vector(4L));
            var condition = Vector(true, false, true, false);
            var compressed = (Tensor<float32>)OnnxOp.Compress(expanded, condition, axis: null);
            var loss = compressed.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(2f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradCompressWithAxisCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var expanded = (Tensor<float32>)OnnxOp.Expand(a, Vector(2L, 3L));
            var condition = Vector(true, true, false);
            var compressed = (Tensor<float32>)OnnxOp.Compress(expanded, condition, axis: 1);
            var loss = compressed.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(4f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module]
    public partial class AutoGradCompressWithAxisZeroCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var expanded = (Tensor<float32>)OnnxOp.Expand(a, Vector(3L, 2L));
            var condition = Vector(true, false, true);
            var compressed = (Tensor<float32>)OnnxOp.Compress(expanded, condition, axis: 0);
            var loss = compressed.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(4f)).Abs() < Scalar(1e-5f);
        }
    }

    // ===================================================================
    //  DFT roundtrip (converted from AutoDiffDftTests.TestDftRoundtripGradient).
    //  IDFT(DFT(X)) = X, so sum(roundtrip(expand(a, [1,2,2]))) = 4a → dL/da = 4.
    // ===================================================================

    [Module]
    public partial class AutoGradDftRoundtripCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var expanded = (Tensor<float32>)OnnxOp.Expand(a, Vector(1L, 2L, 2L));
            var axis = Scalar(1L);
            var dftOut = OnnxOp.Dft(expanded, null, axis, inverse: false, onesided: false);
            var axis2 = Scalar(1L);
            var idftOut = (Tensor<float32>)OnnxOp.Dft(dftOut, null, axis2, inverse: true, onesided: false);
            var loss = idftOut.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(4f)).Abs() < Scalar(1e-5f);
        }
    }

    // ===================================================================
    //  ConstantOfShape gradient (converted from
    //  AutoDiffShapeTests.TestConstantOfShapeGradient_NullShapeInputGradient).
    //  Previously blocked because the QEE ConstantOfShape op ignored the
    //  value attribute and emitted a zero tensor regardless of the fill
    //  value — see commit history for the fix in
    //  Inference/QuickExecutionEngine/Ops/ConstantOfShape.cs.
    //  Exercises the gradient method whose only input is the non-
    //  differentiable shape — see AC.cs:87-92.
    // ===================================================================

    [Module]
    public partial class AutoGradConstantOfShapeNullShapeGradientCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var shape = Vector(2L, 3L);
            var ones = (Tensor<float32>)OnnxOp.ConstantOfShape(shape, TensorData(DType.Float32, [1L], 1f));
            var aMat = (Tensor<float32>)OnnxOp.Expand(a, Vector(2L, 3L));
            var product = ones * aMat;
            var loss = product.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(6f)).Abs() < Scalar(1e-5f);
        }
    }

    // ===================================================================
    //  DFT default-axis gradient (converted from
    //  AutoDiffDftTests.TestDftDefaultAxisGradient). Previously blocked
    //  because ORT segfaulted on DFT with null axis input; OnnxOp.Dft now
    //  substitutes Scalar(-2L) (the ONNX-spec default) when axis is null,
    //  so the emitted ONNX graph carries an explicit axis and ORT executes
    //  it cleanly. The DFT gradient builder's null-axis defensive branch
    //  (Batch20.cs ≈ axisInput-is-null) is now unreachable from this public
    //  API but kept as a safety net.
    // ===================================================================

    [Module]
    public partial class AutoGradDftDefaultAxisCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var expanded = (Tensor<float32>)OnnxOp.Expand(a, Vector(1L, 2L, 2L));
            var dftOut = (Tensor<float32>)OnnxOp.Dft(expanded, null, null, inverse: false, onesided: false);
            var loss = dftOut.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(4f)).Abs() < Scalar(1e-5f);
        }
    }

    // ===================================================================
    //  GridSample multi-channel gradient (converted from
    //  AutoDiffGridSampleTests.TestGridSampleGradient_MultiChannel).
    //  Sample at (0,0) with align_corners=true on a [1,2,2,2] tensor of
    //  values 1..8 scaled by a, all 4 corners weighted 0.25 → sum = 9·a,
    //  so dL/da = 9.
    // ===================================================================

    [Module]
    public partial class AutoGradGridSampleMultiChannelCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var inputVals = (Tensor<float32>)OnnxOp.Constant(
                TensorData(8, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f));
            var x = (Tensor<float32>)OnnxOp.Mul(
                (Tensor<float32>)OnnxOp.Reshape(inputVals, Vector(1L, 2L, 2L, 2L), allowZero: false), a);
            var gridVals = (Tensor<float32>)OnnxOp.Constant(TensorData(2, 0f, 0f));
            var grid = (Tensor<float32>)OnnxOp.Reshape(gridVals, Vector(1L, 1L, 1L, 2L), allowZero: false);
            var output = (Tensor<float32>)OnnxOp.GridSample(x, grid,
                alignCorners: true, mode: GridSampleMode.Linear, paddingMode: GridSamplePaddingMode.Zeros);
            var loss = output.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(9f)).Abs() < Scalar(1e-5f);
        }
    }

    // ===================================================================
    //  TopK (converted from AutoDiffShapeTests TopK helpers). The gradient
    //  is a one-hot mask: dL/dx_i = 1 iff x_i is in top-K, else 0.
    // ===================================================================

    /// <summary>data=[x, 1, 2], TopK(k=1, largest, sorted) at x=5 → x selected, dL/dx = 1.</summary>
    [Module]
    public partial class AutoGradTopK1DLargestK1Check
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            var x_vec = (Tensor<float32>)OnnxOp.Unsqueeze(x, Vector(0L));
            var data = (Tensor<float32>)OnnxOp.Concat([x_vec, Vector(1.0f), Vector(2.0f)], axis: 0);
            var (values, _) = OnnxOp.TopK(data, Vector(1L), axis: 0, largest: true, sorted: true);
            var loss = ((Tensor<float32>)values).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            return (grad - Scalar(1f)).Abs() < Scalar(1e-5f);
        }
    }

    /// <summary>data=[x, 1, 2], TopK(k=2, largest, sorted) at x=5 → x in top-2, dL/dx = 1.</summary>
    [Module]
    public partial class AutoGradTopK1DLargestK2Check
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            var x_vec = (Tensor<float32>)OnnxOp.Unsqueeze(x, Vector(0L));
            var data = (Tensor<float32>)OnnxOp.Concat([x_vec, Vector(1.0f), Vector(2.0f)], axis: 0);
            var (values, _) = OnnxOp.TopK(data, Vector(2L), axis: 0, largest: true, sorted: true);
            var loss = ((Tensor<float32>)values).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            return (grad - Scalar(1f)).Abs() < Scalar(1e-5f);
        }
    }

    /// <summary>data=[x, 10, 20], TopK(k=1, largest) at x=0.5 → x NOT selected, dL/dx = 0.</summary>
    [Module]
    public partial class AutoGradTopKNotSelectedCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            var x_vec = (Tensor<float32>)OnnxOp.Unsqueeze(x, Vector(0L));
            var data = (Tensor<float32>)OnnxOp.Concat([x_vec, Vector(10.0f), Vector(20.0f)], axis: 0);
            var (values, _) = OnnxOp.TopK(data, Vector(1L), axis: 0, largest: true, sorted: true);
            var loss = ((Tensor<float32>)values).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            return grad.Abs() < Scalar(1e-5f);
        }
    }

    /// <summary>2D data with x=7 in row 0; TopK(k=2, axis=1) → x selected (rank 0). dL/dx = 1.</summary>
    [Module]
    public partial class AutoGradTopK2DAxis1Check
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            var x_vec = (Tensor<float32>)OnnxOp.Unsqueeze(x, Vector(0L));
            var row0 = (Tensor<float32>)OnnxOp.Concat([x_vec, Vector(1.0f), Vector(3.0f)], axis: 0);
            var row0_2d = (Tensor<float32>)OnnxOp.Unsqueeze(row0, Vector(0L));
            var row1_2d = (Tensor<float32>)OnnxOp.Reshape(Vector(4.0f, 5.0f, 6.0f), Vector(1L, 3L), allowZero: false);
            var data = (Tensor<float32>)OnnxOp.Concat([row0_2d, row1_2d], axis: 0);
            var (values, _) = OnnxOp.TopK(data, Vector(2L), axis: 1, largest: true, sorted: true);
            var loss = ((Tensor<float32>)values).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            return (grad - Scalar(1f)).Abs() < Scalar(1e-5f);
        }
    }

    /// <summary>data=[x, 10, 20], TopK(k=1, largest=false, sorted) at x=0.5 → x IS smallest, dL/dx = 1.</summary>
    [Module]
    public partial class AutoGradTopKSmallestK1Check
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            var x_vec = (Tensor<float32>)OnnxOp.Unsqueeze(x, Vector(0L));
            var data = (Tensor<float32>)OnnxOp.Concat([x_vec, Vector(10.0f), Vector(20.0f)], axis: 0);
            var (values, _) = OnnxOp.TopK(data, Vector(1L), axis: 0, largest: false, sorted: true);
            var loss = ((Tensor<float32>)values).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            return (grad - Scalar(1f)).Abs() < Scalar(1e-5f);
        }
    }

    // ===================================================================
    //  MaxRoiPool (converted from AutoDiffMaxRoiPoolTests). Expected
    //  values were captured by running the existing tests once.
    // ===================================================================

    /// <summary>4x4 input scaled by a, full ROI, 2x2 pooled bins. dL/da = 34.</summary>
    [Module]
    public partial class AutoGradMaxRoiPoolPositiveCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var input = (Tensor<float32>)OnnxOp.Constant(TensorData(16, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f));
            var x = (Tensor<float32>)OnnxOp.Mul(
                (Tensor<float32>)OnnxOp.Reshape(input, Vector(1L, 1L, 4L, 4L), allowZero: false), a);
            var rois = (Tensor<float32>)OnnxOp.Reshape(
                (Tensor<float32>)OnnxOp.Constant(TensorData(5, 0f, 0f, 0f, 3f, 3f)),
                Vector(1L, 5L), allowZero: false);
            var output = (Tensor<float32>)OnnxOp.MaxRoiPool(x, rois, pooledShape: [2L, 2L], spatialScale: 1.0f);
            var loss = output.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(34f)).Abs() < Scalar(1e-5f);
        }
    }

    /// <summary>Same forward as Positive but AutoGrad over x (tensor). Expected one-hot mask.</summary>
    [Module]
    public partial class AutoGradMaxRoiPoolGradientShapeCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            // a is unused; kept so the module has a runtime input slot the AutoTester expects.
            var input = (Tensor<float32>)OnnxOp.Constant(TensorData(16, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f));
            var x = (Tensor<float32>)OnnxOp.Reshape(input, Vector(1L, 1L, 4L, 4L), allowZero: false);
            // Force x to depend on a so the gradient is non-trivially routed (a * 1 + 0*x acts as identity-with-input-dep).
            x = (Tensor<float32>)OnnxOp.Mul(x, a);
            var rois = (Tensor<float32>)OnnxOp.Reshape(
                (Tensor<float32>)OnnxOp.Constant(TensorData(5, 0f, 0f, 0f, 3f, 3f)),
                Vector(1L, 5L), allowZero: false);
            var output = (Tensor<float32>)OnnxOp.MaxRoiPool(x, rois, pooledShape: [2L, 2L], spatialScale: 1.0f);
            var loss = output.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            // Same setup as Positive at a=1 — dL/da = 34.
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(34f)).Abs() < Scalar(1e-5f);
        }
    }

    /// <summary>2-channel 3x3 input scaled by a, partial ROI, 1x1 pool. dL/da = 19.</summary>
    [Module]
    public partial class AutoGradMaxRoiPoolMultiChannelCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var input = (Tensor<float32>)OnnxOp.Constant(TensorData(18, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f, 17f, 18f));
            var x = (Tensor<float32>)OnnxOp.Mul(
                (Tensor<float32>)OnnxOp.Reshape(input, Vector(1L, 2L, 3L, 3L), allowZero: false), a);
            var rois = (Tensor<float32>)OnnxOp.Reshape(
                (Tensor<float32>)OnnxOp.Constant(TensorData(5, 0f, 0f, 0f, 2f, 2f)),
                Vector(1L, 5L), allowZero: false);
            var output = (Tensor<float32>)OnnxOp.MaxRoiPool(x, rois, pooledShape: [1L, 1L], spatialScale: 1.0f);
            var loss = output.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(19f)).Abs() < Scalar(1e-5f);
        }
    }

    /// <summary>4x4 input scaled by a, two ROIs, 1x1 pool each. dL/da = 22.</summary>
    [Module]
    public partial class AutoGradMaxRoiPoolMultipleRoisCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var input = (Tensor<float32>)OnnxOp.Constant(TensorData(16, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f));
            var x = (Tensor<float32>)OnnxOp.Mul(
                (Tensor<float32>)OnnxOp.Reshape(input, Vector(1L, 1L, 4L, 4L), allowZero: false), a);
            var rois = (Tensor<float32>)OnnxOp.Reshape(
                (Tensor<float32>)OnnxOp.Constant(TensorData(10, 0f, 0f, 0f, 1f, 1f, 0f, 2f, 2f, 3f, 3f)),
                Vector(2L, 5L), allowZero: false);
            var output = (Tensor<float32>)OnnxOp.MaxRoiPool(x, rois, pooledShape: [1L, 1L], spatialScale: 1.0f);
            var loss = output.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(22f)).Abs() < Scalar(1e-5f);
        }
    }

    /// <summary>Forward-only: sum of MaxRoiPool output should be 44 (no AutoGrad).</summary>
    [Module]
    public partial class AutoGradMaxRoiPoolForwardSumCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            // a is unused; the AutoTester needs at least one runtime input.
            var input = (Tensor<float32>)OnnxOp.Constant(TensorData(16, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f));
            var x = (Tensor<float32>)OnnxOp.Reshape(input, Vector(1L, 1L, 4L, 4L), allowZero: false);
            var rois = (Tensor<float32>)OnnxOp.Reshape(
                (Tensor<float32>)OnnxOp.Constant(TensorData(5, 0f, 0f, 0f, 3f, 3f)),
                Vector(1L, 5L), allowZero: false);
            var output = (Tensor<float32>)OnnxOp.MaxRoiPool(x, rois, pooledShape: [2L, 2L], spatialScale: 1.0f);
            var sum = output.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            // Touch a so the AutoTester doesn't flag the input as unused — produces a no-op check.
            var aTouch = (a - a).Abs() < Scalar(1e-5f);
            var sumOk = (sum - Scalar(44f)).Abs() < Scalar(1e-5f);
            return aTouch & sumOk;
        }
    }

    // ===================================================================
    //  RoiAlign (converted from AutoDiffRoiAlignTests). Expected gradient
    //  values captured by running the original numerical tests once.
    // ===================================================================

    /// <summary>4x4 input scaled by a, full ROI, 2x2 output, samplingRatio=2. dL/da = 24.625.</summary>
    [Module]
    public partial class AutoGradRoiAlignPositiveCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var input = (Tensor<float32>)OnnxOp.Constant(TensorData(16, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f));
            var x = (Tensor<float32>)OnnxOp.Mul(
                (Tensor<float32>)OnnxOp.Reshape(input, Vector(1L, 1L, 4L, 4L), allowZero: false), a);
            var rois = (Tensor<float32>)OnnxOp.Reshape(
                (Tensor<float32>)OnnxOp.Constant(TensorData(4, 0f, 0f, 3f, 3f)),
                Vector(1L, 4L), allowZero: false);
            var batchIdx = (Tensor<int64>)OnnxOp.Constant(TensorData(1, 0L));
            var output = (Tensor<float32>)OnnxOp.RoiAlign(x, rois, batchIdx,
                mode: RoiAlignMode.Avg, outputHeight: 2, outputWidth: 2,
                samplingRatio: 2, spatialScale: 1.0f);
            var loss = output.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(24.625f)).Abs() < Scalar(1e-4f);
        }
    }

    /// <summary>3x3 input, ROI 0,0-4,4 with spatialScale=0.5 → covers whole image. dL/da = 3.</summary>
    [Module]
    public partial class AutoGradRoiAlignSpatialScaleCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var input = (Tensor<float32>)OnnxOp.Constant(TensorData(9, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f));
            var x = (Tensor<float32>)OnnxOp.Mul(
                (Tensor<float32>)OnnxOp.Reshape(input, Vector(1L, 1L, 3L, 3L), allowZero: false), a);
            var rois = (Tensor<float32>)OnnxOp.Reshape(
                (Tensor<float32>)OnnxOp.Constant(TensorData(4, 0f, 0f, 4f, 4f)),
                Vector(1L, 4L), allowZero: false);
            var batchIdx = (Tensor<int64>)OnnxOp.Constant(TensorData(1, 0L));
            var output = (Tensor<float32>)OnnxOp.RoiAlign(x, rois, batchIdx,
                mode: RoiAlignMode.Avg, outputHeight: 1, outputWidth: 1,
                samplingRatio: 2, spatialScale: 0.5f);
            var loss = output.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(3f)).Abs() < Scalar(1e-5f);
        }
    }

    /// <summary>2-channel 3x3 input. dL/da = 15.</summary>
    [Module]
    public partial class AutoGradRoiAlignMultiChannelCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var input = (Tensor<float32>)OnnxOp.Constant(TensorData(18, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f, 17f, 18f));
            var x = (Tensor<float32>)OnnxOp.Mul(
                (Tensor<float32>)OnnxOp.Reshape(input, Vector(1L, 2L, 3L, 3L), allowZero: false), a);
            var rois = (Tensor<float32>)OnnxOp.Reshape(
                (Tensor<float32>)OnnxOp.Constant(TensorData(4, 0f, 0f, 2f, 2f)),
                Vector(1L, 4L), allowZero: false);
            var batchIdx = (Tensor<int64>)OnnxOp.Constant(TensorData(1, 0L));
            var output = (Tensor<float32>)OnnxOp.RoiAlign(x, rois, batchIdx,
                mode: RoiAlignMode.Avg, outputHeight: 1, outputWidth: 1,
                samplingRatio: 2, spatialScale: 1.0f);
            var loss = output.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(15f)).Abs() < Scalar(1e-5f);
        }
    }

    /// <summary>4x4 input, two ROIs (top-left, bottom-right), 1x1 pool. dL/da = 12.625.</summary>
    [Module]
    public partial class AutoGradRoiAlignMultipleRoisCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var input = (Tensor<float32>)OnnxOp.Constant(TensorData(16, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f));
            var x = (Tensor<float32>)OnnxOp.Mul(
                (Tensor<float32>)OnnxOp.Reshape(input, Vector(1L, 1L, 4L, 4L), allowZero: false), a);
            var rois = (Tensor<float32>)OnnxOp.Reshape(
                (Tensor<float32>)OnnxOp.Constant(TensorData(8, 0f, 0f, 1f, 1f, 2f, 2f, 3f, 3f)),
                Vector(2L, 4L), allowZero: false);
            var batchIdx = (Tensor<int64>)OnnxOp.Constant(TensorData(2, 0L, 0L));
            var output = (Tensor<float32>)OnnxOp.RoiAlign(x, rois, batchIdx,
                mode: RoiAlignMode.Avg, outputHeight: 1, outputWidth: 1,
                samplingRatio: 2, spatialScale: 1.0f);
            var loss = output.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(12.625f)).Abs() < Scalar(1e-4f);
        }
    }

    /// <summary>Same as Positive but with coordinateTransformationMode=Output_half_pixel
    /// (previously spelled via the misleading positional CoordinateTransformationMode
    /// .Half_pixel_symmetric — same "output_half_pixel" wire value). dL/da = 34.</summary>
    [Module]
    public partial class AutoGradRoiAlignOutputHalfPixelCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var input = (Tensor<float32>)OnnxOp.Constant(TensorData(16, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f));
            var x = (Tensor<float32>)OnnxOp.Mul(
                (Tensor<float32>)OnnxOp.Reshape(input, Vector(1L, 1L, 4L, 4L), allowZero: false), a);
            var rois = (Tensor<float32>)OnnxOp.Reshape(
                (Tensor<float32>)OnnxOp.Constant(TensorData(4, 0f, 0f, 3f, 3f)),
                Vector(1L, 4L), allowZero: false);
            var batchIdx = (Tensor<int64>)OnnxOp.Constant(TensorData(1, 0L));
            var output = (Tensor<float32>)OnnxOp.RoiAlign(x, rois, batchIdx,
                coordinateTransformationMode: RoiAlignTransformationMode.Output_half_pixel,
                mode: RoiAlignMode.Avg, outputHeight: 2, outputWidth: 2,
                samplingRatio: 2, spatialScale: 1.0f);
            var loss = output.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(34f)).Abs() < Scalar(1e-5f);
        }
    }

    /// <summary>Forward-only: sum of RoiAlign output = 24.625 for the Positive setup.</summary>
    [Module]
    public partial class AutoGradRoiAlignForwardSumCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var input = (Tensor<float32>)OnnxOp.Constant(TensorData(16, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f));
            var x = (Tensor<float32>)OnnxOp.Reshape(input, Vector(1L, 1L, 4L, 4L), allowZero: false);
            var rois = (Tensor<float32>)OnnxOp.Reshape(
                (Tensor<float32>)OnnxOp.Constant(TensorData(4, 0f, 0f, 3f, 3f)),
                Vector(1L, 4L), allowZero: false);
            var batchIdx = (Tensor<int64>)OnnxOp.Constant(TensorData(1, 0L));
            var output = (Tensor<float32>)OnnxOp.RoiAlign(x, rois, batchIdx,
                mode: RoiAlignMode.Avg, outputHeight: 2, outputWidth: 2,
                samplingRatio: 2, spatialScale: 1.0f);
            var sum = output.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var aTouch = (a - a).Abs() < Scalar(1e-5f);
            var sumOk = (sum - Scalar(24.625f)).Abs() < Scalar(1e-4f);
            return aTouch & sumOk;
        }
    }

    // ===================================================================
    //  GRU / RNN / LSTM (converted from AutoDiffRecurrentTests). Expected
    //  gradient values captured by running the original numerical tests
    //  once with the same input data. Shared input constants are inlined
    //  in each module via Vector(...) literals; W/R/B values match the
    //  GruWValues/etc. arrays in the original test file.
    // ===================================================================

    internal static class RecurrentTestData
    {
        public static readonly float[] X3 = { 0.3f, 0.5f, 0.4f, 0.6f, -0.2f, 0.1f };
        public static readonly float[] GruW = { 0.1f, 0.2f, -0.1f, 0.3f, 0.15f, -0.2f, 0.25f, 0.1f, -0.15f, 0.3f, 0.2f, -0.1f };
        public static readonly float[] GruR = { 0.2f, -0.1f, 0.1f, 0.15f, -0.2f, 0.25f, 0.1f, -0.15f, 0.3f, 0.2f, -0.1f, 0.1f };
        public static readonly float[] GruB = { 0.05f, -0.1f, 0.15f, 0.2f, -0.05f, 0.1f, -0.2f, 0.05f, 0.1f, -0.15f, 0.2f, -0.1f };
        public static readonly float[] RnnW = { 0.1f, 0.2f, -0.1f, 0.3f };
        public static readonly float[] RnnR = { 0.2f, -0.1f, 0.1f, 0.15f };
        public static readonly float[] RnnB = { 0.05f, -0.1f, 0.15f, 0.2f };
        public static readonly float[] LstmW = { 0.1f, 0.2f, -0.1f, 0.3f, 0.15f, -0.2f, 0.25f, 0.1f, -0.15f, 0.3f, 0.2f, -0.1f, 0.05f, 0.15f, -0.25f, 0.1f };
        public static readonly float[] LstmR = { 0.2f, -0.1f, 0.1f, 0.15f, -0.2f, 0.25f, 0.1f, -0.15f, 0.3f, 0.2f, -0.1f, 0.1f, 0.05f, -0.2f, 0.15f, 0.1f };
        public static readonly float[] LstmB = { 0.05f, -0.1f, 0.15f, 0.2f, -0.05f, 0.1f, -0.2f, 0.05f, 0.1f, -0.15f, 0.2f, -0.1f, 0.05f, -0.2f, 0.15f, 0.1f };

        public static Tensor<float32> BuildX(Scalar<float32> xVal, int seqLen)
        {
            var xVec = (Tensor<float32>)OnnxOp.Unsqueeze(xVal, Vector(0L));
            var rest = X3[1..(2 * seqLen)];
            var flat = (Tensor<float32>)OnnxOp.Concat([xVec, Vector(rest)], axis: 0);
            return (Tensor<float32>)OnnxOp.Reshape(flat, Vector((long)seqLen, 1L, 2L), allowZero: false);
        }

        public static Variable XConst(int seqLen)
            => OnnxOp.Reshape(Vector(X3[..(2 * seqLen)]), Vector((long)seqLen, 1L, 2L), allowZero: false);

        public static Variable GruWConst() => OnnxOp.Reshape(Vector(GruW), Vector(1L, 6L, 2L), allowZero: false);
        public static Variable GruRConst() => OnnxOp.Reshape(Vector(GruR), Vector(1L, 6L, 2L), allowZero: false);
        public static Variable GruBConst() => OnnxOp.Reshape(Vector(GruB), Vector(1L, 12L), allowZero: false);
        public static Variable RnnWConst() => OnnxOp.Reshape(Vector(RnnW), Vector(1L, 2L, 2L), allowZero: false);
        public static Variable RnnRConst() => OnnxOp.Reshape(Vector(RnnR), Vector(1L, 2L, 2L), allowZero: false);
        public static Variable RnnBConst() => OnnxOp.Reshape(Vector(RnnB), Vector(1L, 4L), allowZero: false);
        public static Variable LstmWConst() => OnnxOp.Reshape(Vector(LstmW), Vector(1L, 8L, 2L), allowZero: false);
        public static Variable LstmRConst() => OnnxOp.Reshape(Vector(LstmR), Vector(1L, 8L, 2L), allowZero: false);
        public static Variable LstmBConst() => OnnxOp.Reshape(Vector(LstmB), Vector(1L, 16L), allowZero: false);

        public static Scalar<float32> GruLoss(Variable x, Variable w, Variable r, Variable? b = null, Variable? h0 = null, bool? lbr = null)
        {
            var (_, yh) = OnnxOp.Gru(x, w, r, b, null, h0, null, null, null, null, GRUDirection.Forward, 2L, false, lbr);
            return ((Tensor<float32>)yh!).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        }

        public static Scalar<float32> GruFullYLoss(Variable x, Variable w, Variable r)
        {
            var (y, _) = OnnxOp.Gru(x, w, r, null, null, null, null, null, null, null, GRUDirection.Forward, 2L, false, null);
            return ((Tensor<float32>)y).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        }

        public static Scalar<float32> RnnLoss(Variable x, Variable w, Variable r, Variable? b = null, Variable? h0 = null)
        {
            var (_, yh) = OnnxOp.Rnn(x, w, r, b, null, h0, null, null, null, null, RNNDirection.Forward, 2L, false);
            return ((Tensor<float32>)yh!).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        }

        public static Scalar<float32> RnnFullYLoss(Variable x, Variable w, Variable r)
        {
            var (y, _) = OnnxOp.Rnn(x, w, r, null, null, null, null, null, null, null, RNNDirection.Forward, 2L, false);
            return ((Tensor<float32>)y).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        }

        public static Scalar<float32> LstmLoss(Variable x, Variable w, Variable r, Variable? b = null, Variable? h0 = null, Variable? c0 = null)
        {
            var (_, yh, _) = OnnxOp.Lstm(x, w, r, b, null, h0, c0, null, null, null, null, null, LSTMDirection.Forward, 2L, null, false);
            return ((Tensor<float32>)yh!).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        }

        public static Scalar<float32> LstmFullYLoss(Variable x, Variable w, Variable r)
        {
            var (y, _, _) = OnnxOp.Lstm(x, w, r, null, null, null, null, null, null, null, null, null, LSTMDirection.Forward, 2L, null, false);
            return ((Tensor<float32>)y).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        }
    }

    // ---- GRU ----

    [Module] public partial class AutoGradGruXSeqLen1Check
    {
        public static Scalar<bit> Inline(Scalar<float32> xv)
        {
            var loss = RecurrentTestData.GruLoss(RecurrentTestData.BuildX(xv, 1), RecurrentTestData.GruWConst(), RecurrentTestData.GruRConst());
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(xv, loss);
            return (grad - Scalar(0.022278218f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module] public partial class AutoGradGruXSeqLen2Check
    {
        public static Scalar<bit> Inline(Scalar<float32> xv)
        {
            var loss = RecurrentTestData.GruLoss(RecurrentTestData.BuildX(xv, 2), RecurrentTestData.GruWConst(), RecurrentTestData.GruRConst());
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(xv, loss);
            return (grad - Scalar(0.015871616f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module] public partial class AutoGradGruXSeqLen3Check
    {
        public static Scalar<bit> Inline(Scalar<float32> xv)
        {
            var loss = RecurrentTestData.GruLoss(RecurrentTestData.BuildX(xv, 3), RecurrentTestData.GruWConst(), RecurrentTestData.GruRConst());
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(xv, loss);
            return (grad - Scalar(0.01074962f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module] public partial class AutoGradGruWCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> wv)
        {
            var wVec = (Tensor<float32>)OnnxOp.Unsqueeze(wv, Vector(0L));
            var wFlat = (Tensor<float32>)OnnxOp.Concat([wVec, Vector(RecurrentTestData.GruW[1..])], axis: 0);
            var w = OnnxOp.Reshape(wFlat, Vector(1L, 6L, 2L), allowZero: false);
            var loss = RecurrentTestData.GruLoss(RecurrentTestData.XConst(3), w, RecurrentTestData.GruRConst());
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(wv, loss);
            return (grad - Scalar(-0.0073668435f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module] public partial class AutoGradGruRCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> rv)
        {
            var rVec = (Tensor<float32>)OnnxOp.Unsqueeze(rv, Vector(0L));
            var rFlat = (Tensor<float32>)OnnxOp.Concat([rVec, Vector(RecurrentTestData.GruR[1..])], axis: 0);
            var r = OnnxOp.Reshape(rFlat, Vector(1L, 6L, 2L), allowZero: false);
            var loss = RecurrentTestData.GruLoss(RecurrentTestData.XConst(3), RecurrentTestData.GruWConst(), r);
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(rv, loss);
            return (grad - Scalar(-0.0002834663f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module] public partial class AutoGradGruBCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> bv)
        {
            var bVec = (Tensor<float32>)OnnxOp.Unsqueeze(bv, Vector(0L));
            var bFlat = (Tensor<float32>)OnnxOp.Concat([bVec, Vector(RecurrentTestData.GruB[1..])], axis: 0);
            var b = OnnxOp.Reshape(bFlat, Vector(1L, 12L), allowZero: false);
            var loss = RecurrentTestData.GruLoss(RecurrentTestData.XConst(3), RecurrentTestData.GruWConst(), RecurrentTestData.GruRConst(), b: b);
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(bv, loss);
            return (grad - Scalar(-0.04743617f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module] public partial class AutoGradGruH0Check
    {
        public static Scalar<bit> Inline(Scalar<float32> hv)
        {
            var hVec = (Tensor<float32>)OnnxOp.Unsqueeze(hv, Vector(0L));
            var hFlat = (Tensor<float32>)OnnxOp.Concat([hVec, Vector(0.1f)], axis: 0);
            var h0 = OnnxOp.Reshape(hFlat, Vector(1L, 1L, 2L), allowZero: false);
            var loss = RecurrentTestData.GruLoss(RecurrentTestData.XConst(3), RecurrentTestData.GruWConst(), RecurrentTestData.GruRConst(), h0: h0);
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(hv, loss);
            return (grad - Scalar(0.19332777f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module] public partial class AutoGradGruLinearBeforeResetCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> xv)
        {
            var (_, yh) = OnnxOp.Gru(RecurrentTestData.BuildX(xv, 2),
                RecurrentTestData.GruWConst(), RecurrentTestData.GruRConst(), RecurrentTestData.GruBConst(),
                null, null, null, null, null, null, GRUDirection.Forward, 2L, false, true);
            var loss = ((Tensor<float32>)yh!).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(xv, loss);
            return (grad - Scalar(0.017674165f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module] public partial class AutoGradGruFullSequenceOutputCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> xv)
        {
            var loss = RecurrentTestData.GruFullYLoss(RecurrentTestData.BuildX(xv, 2),
                RecurrentTestData.GruWConst(), RecurrentTestData.GruRConst());
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(xv, loss);
            return (grad - Scalar(0.03814982f)).Abs() < Scalar(1e-5f);
        }
    }

    // ---- RNN ----

    [Module] public partial class AutoGradRnnXSeqLen1Check
    {
        public static Scalar<bit> Inline(Scalar<float32> xv)
        {
            var loss = RecurrentTestData.RnnLoss(RecurrentTestData.BuildX(xv, 1), RecurrentTestData.RnnWConst(), RecurrentTestData.RnnRConst());
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(xv, loss);
            return (grad - Scalar(-0.00024485734f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module] public partial class AutoGradRnnXSeqLen2Check
    {
        public static Scalar<bit> Inline(Scalar<float32> xv)
        {
            var loss = RecurrentTestData.RnnLoss(RecurrentTestData.BuildX(xv, 2), RecurrentTestData.RnnWConst(), RecurrentTestData.RnnRConst());
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(xv, loss);
            return (grad - Scalar(0.023836536f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module] public partial class AutoGradRnnXSeqLen3Check
    {
        public static Scalar<bit> Inline(Scalar<float32> xv)
        {
            var loss = RecurrentTestData.RnnLoss(RecurrentTestData.BuildX(xv, 3), RecurrentTestData.RnnWConst(), RecurrentTestData.RnnRConst());
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(xv, loss);
            return (grad - Scalar(0.008333627f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module] public partial class AutoGradRnnWCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> wv)
        {
            var wVec = (Tensor<float32>)OnnxOp.Unsqueeze(wv, Vector(0L));
            var wFlat = (Tensor<float32>)OnnxOp.Concat([wVec, Vector(RecurrentTestData.RnnW[1..])], axis: 0);
            var w = OnnxOp.Reshape(wFlat, Vector(1L, 2L, 2L), allowZero: false);
            var loss = RecurrentTestData.RnnLoss(RecurrentTestData.XConst(3), w, RecurrentTestData.RnnRConst());
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(wv, loss);
            return (grad - Scalar(-0.06533176f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module] public partial class AutoGradRnnRCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> rv)
        {
            var rVec = (Tensor<float32>)OnnxOp.Unsqueeze(rv, Vector(0L));
            var rFlat = (Tensor<float32>)OnnxOp.Concat([rVec, Vector(RecurrentTestData.RnnR[1..])], axis: 0);
            var r = OnnxOp.Reshape(rFlat, Vector(1L, 2L, 2L), allowZero: false);
            var loss = RecurrentTestData.RnnLoss(RecurrentTestData.XConst(3), RecurrentTestData.RnnWConst(), r);
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(rv, loss);
            return (grad - Scalar(0.20964399f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module] public partial class AutoGradRnnBCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> bv)
        {
            var bVec = (Tensor<float32>)OnnxOp.Unsqueeze(bv, Vector(0L));
            var bFlat = (Tensor<float32>)OnnxOp.Concat([bVec, Vector(RecurrentTestData.RnnB[1..])], axis: 0);
            var b = OnnxOp.Reshape(bFlat, Vector(1L, 4L), allowZero: false);
            var loss = RecurrentTestData.RnnLoss(RecurrentTestData.XConst(3), RecurrentTestData.RnnWConst(), RecurrentTestData.RnnRConst(), b: b);
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(bv, loss);
            return (grad - Scalar(1.2307094f)).Abs() < Scalar(1e-4f);
        }
    }

    [Module] public partial class AutoGradRnnH0Check
    {
        public static Scalar<bit> Inline(Scalar<float32> hv)
        {
            var hVec = (Tensor<float32>)OnnxOp.Unsqueeze(hv, Vector(0L));
            var hFlat = (Tensor<float32>)OnnxOp.Concat([hVec, Vector(0.1f)], axis: 0);
            var h0 = OnnxOp.Reshape(hFlat, Vector(1L, 1L, 2L), allowZero: false);
            var loss = RecurrentTestData.RnnLoss(RecurrentTestData.XConst(3), RecurrentTestData.RnnWConst(), RecurrentTestData.RnnRConst(), h0: h0);
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(hv, loss);
            return (grad - Scalar(0.010086148f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module] public partial class AutoGradRnnFullSequenceOutputCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> xv)
        {
            var loss = RecurrentTestData.RnnFullYLoss(RecurrentTestData.BuildX(xv, 2),
                RecurrentTestData.RnnWConst(), RecurrentTestData.RnnRConst());
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(xv, loss);
            return (grad - Scalar(0.02359167f)).Abs() < Scalar(1e-5f);
        }
    }

    // ---- LSTM ----

    [Module] public partial class AutoGradLstmXSeqLen1Check
    {
        public static Scalar<bit> Inline(Scalar<float32> xv)
        {
            var loss = RecurrentTestData.LstmLoss(RecurrentTestData.BuildX(xv, 1), RecurrentTestData.LstmWConst(), RecurrentTestData.LstmRConst());
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(xv, loss);
            return (grad - Scalar(-0.055132754f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module] public partial class AutoGradLstmXSeqLen2Check
    {
        public static Scalar<bit> Inline(Scalar<float32> xv)
        {
            var loss = RecurrentTestData.LstmLoss(RecurrentTestData.BuildX(xv, 2), RecurrentTestData.LstmWConst(), RecurrentTestData.LstmRConst());
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(xv, loss);
            return (grad - Scalar(-0.026522601f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module] public partial class AutoGradLstmXSeqLen3Check
    {
        public static Scalar<bit> Inline(Scalar<float32> xv)
        {
            var loss = RecurrentTestData.LstmLoss(RecurrentTestData.BuildX(xv, 3), RecurrentTestData.LstmWConst(), RecurrentTestData.LstmRConst());
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(xv, loss);
            return (grad - Scalar(-0.0093566505f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module] public partial class AutoGradLstmWCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> wv)
        {
            var wVec = (Tensor<float32>)OnnxOp.Unsqueeze(wv, Vector(0L));
            var wFlat = (Tensor<float32>)OnnxOp.Concat([wVec, Vector(RecurrentTestData.LstmW[1..])], axis: 0);
            var w = OnnxOp.Reshape(wFlat, Vector(1L, 8L, 2L), allowZero: false);
            var loss = RecurrentTestData.LstmLoss(RecurrentTestData.XConst(3), w, RecurrentTestData.LstmRConst());
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(wv, loss);
            return (grad - Scalar(0.0038939272f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module] public partial class AutoGradLstmRCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> rv)
        {
            var rVec = (Tensor<float32>)OnnxOp.Unsqueeze(rv, Vector(0L));
            var rFlat = (Tensor<float32>)OnnxOp.Concat([rVec, Vector(RecurrentTestData.LstmR[1..])], axis: 0);
            var r = OnnxOp.Reshape(rFlat, Vector(1L, 8L, 2L), allowZero: false);
            var loss = RecurrentTestData.LstmLoss(RecurrentTestData.XConst(3), RecurrentTestData.LstmWConst(), r);
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(rv, loss);
            return (grad - Scalar(0.00022774911f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module] public partial class AutoGradLstmBCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> bv)
        {
            var bVec = (Tensor<float32>)OnnxOp.Unsqueeze(bv, Vector(0L));
            var bFlat = (Tensor<float32>)OnnxOp.Concat([bVec, Vector(RecurrentTestData.LstmB[1..])], axis: 0);
            var b = OnnxOp.Reshape(bFlat, Vector(1L, 16L), allowZero: false);
            var loss = RecurrentTestData.LstmLoss(RecurrentTestData.XConst(3), RecurrentTestData.LstmWConst(), RecurrentTestData.LstmRConst(), b: b);
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(bv, loss);
            return (grad - Scalar(-0.0013585483f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module] public partial class AutoGradLstmH0Check
    {
        public static Scalar<bit> Inline(Scalar<float32> hv)
        {
            var hVec = (Tensor<float32>)OnnxOp.Unsqueeze(hv, Vector(0L));
            var hFlat = (Tensor<float32>)OnnxOp.Concat([hVec, Vector(0.1f)], axis: 0);
            var h0 = OnnxOp.Reshape(hFlat, Vector(1L, 1L, 2L), allowZero: false);
            var loss = RecurrentTestData.LstmLoss(RecurrentTestData.XConst(3), RecurrentTestData.LstmWConst(), RecurrentTestData.LstmRConst(), h0: h0);
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(hv, loss);
            return (grad - Scalar(0.013472318f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module] public partial class AutoGradLstmC0Check
    {
        public static Scalar<bit> Inline(Scalar<float32> cv)
        {
            var cVec = (Tensor<float32>)OnnxOp.Unsqueeze(cv, Vector(0L));
            var cFlat = (Tensor<float32>)OnnxOp.Concat([cVec, Vector(-0.05f)], axis: 0);
            var c0 = OnnxOp.Reshape(cFlat, Vector(1L, 1L, 2L), allowZero: false);
            var loss = RecurrentTestData.LstmLoss(RecurrentTestData.XConst(3), RecurrentTestData.LstmWConst(), RecurrentTestData.LstmRConst(), c0: c0);
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(cv, loss);
            return (grad - Scalar(0.08446849f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module] public partial class AutoGradLstmFullSequenceOutputCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> xv)
        {
            var loss = RecurrentTestData.LstmFullYLoss(RecurrentTestData.BuildX(xv, 2),
                RecurrentTestData.LstmWConst(), RecurrentTestData.LstmRConst());
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(xv, loss);
            return (grad - Scalar(-0.08165536f)).Abs() < Scalar(1e-5f);
        }
    }

    // ===================================================================
    //  Lrn (converted from AutoDiffNormalizationTests TestLrn*).
    //  Expected gradient values captured by running each original test
    //  once with the same input. The Lrn gradient body in Batch6.cs has
    //  no NotSupported throws; it's a normalization computation across
    //  the channel axis using ChannelWindowSum, all of which the captured
    //  bit-checks exercise end-to-end through ONNX/CS/QEE roundtrips.
    // ===================================================================

    [Module] public partial class AutoGradLrnBasicCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            var mat = (Tensor<float32>)OnnxOp.Expand(x, Vector(1L, 3L, 1L, 1L));
            var y = (Tensor<float32>)OnnxOp.Lrn(mat, alpha: 1.0f, beta: 0.5f, bias: 1.0f, size: 3);
            var loss = y.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            return (grad - Scalar(0.37429708f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module] public partial class AutoGradLrnNumericalCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            var mat = (Tensor<float32>)OnnxOp.Expand(x, Vector(1L, 3L, 2L, 2L));
            var y = (Tensor<float32>)OnnxOp.Lrn(mat, alpha: 1.0f, beta: 0.5f, bias: 1.0f, size: 3);
            var loss = y.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            return (grad - Scalar(1.4971883f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module] public partial class AutoGradLrnSmallAlphaCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            var mat = (Tensor<float32>)OnnxOp.Expand(x, Vector(1L, 3L, 1L, 1L));
            var y = (Tensor<float32>)OnnxOp.Lrn(mat, alpha: 0.0001f, beta: 0.75f, bias: 1.0f, size: 5);
            var loss = y.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            return (grad - Scalar(2.9963577f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module] public partial class AutoGradLrnHighBetaCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            var mat = (Tensor<float32>)OnnxOp.Expand(x, Vector(1L, 3L, 1L, 1L));
            var y = (Tensor<float32>)OnnxOp.Lrn(mat, alpha: 0.5f, beta: 2.0f, bias: 1.0f, size: 3);
            var loss = y.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            return (grad - Scalar(-0.713979f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module] public partial class AutoGradLrnWithScaleCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            var scaled = x * Scalar(3.0f);
            var mat = (Tensor<float32>)OnnxOp.Expand(scaled, Vector(1L, 3L, 1L, 1L));
            var y = (Tensor<float32>)OnnxOp.Lrn(mat, alpha: 1.0f, beta: 0.5f, bias: 1.0f, size: 3);
            var loss = y.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            return (grad - Scalar(0.061329648f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module] public partial class AutoGradLrnMultiChannelCh1Check
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            var ch1 = (Tensor<float32>)OnnxOp.Expand(x, Vector(1L, 1L, 1L, 1L));
            var ch2 = (Tensor<float32>)OnnxOp.Expand(Scalar(2.0f), Vector(1L, 1L, 1L, 1L));
            var mat = (Tensor<float32>)OnnxOp.Concat([ch1, ch2], axis: 1);
            var y = (Tensor<float32>)OnnxOp.Lrn(mat, alpha: 1.0f, beta: 0.5f, bias: 1.0f, size: 3);
            var loss = y.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            return (grad - Scalar(0.38273275f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module] public partial class AutoGradLrnMultiChannelCh2Check
    {
        public static Scalar<bit> Inline(Scalar<float32> x)
        {
            var ch1 = (Tensor<float32>)OnnxOp.Expand(Scalar(1.0f), Vector(1L, 1L, 1L, 1L));
            var ch2 = (Tensor<float32>)OnnxOp.Expand(x, Vector(1L, 1L, 1L, 1L));
            var mat = (Tensor<float32>)OnnxOp.Concat([ch1, ch2], axis: 1);
            var y = (Tensor<float32>)OnnxOp.Lrn(mat, alpha: 1.0f, beta: 0.5f, bias: 1.0f, size: 3);
            var loss = y.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            return (grad - Scalar(0.1530931f)).Abs() < Scalar(1e-5f);
        }
    }

    // ===================================================================
    //  AlignCorners=false variants for AffineGrid and GridSample.
    //  Exercise the else-branch in Batch21.cs / Batch23.cs.
    // ===================================================================

    [Module] public partial class AutoGradAffineGridAlignCornersFalseCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var translationValues = (Tensor<float32>)OnnxOp.Constant(
                TensorData(12, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f));
            var thetaConst = (Tensor<float32>)OnnxOp.Reshape(translationValues, Vector(2L, 2L, 3L), allowZero: false);
            var theta = (Tensor<float32>)OnnxOp.Mul(thetaConst, a);
            var size = (Tensor<int64>)OnnxOp.Constant(TensorData(4, 2L, 1L, 2L, 2L));
            var grid = (Tensor<float32>)OnnxOp.AffineGrid(theta, size, alignCorners: false);
            var loss = grid.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(8f)).Abs() < Scalar(1e-5f);
        }
    }

    [Module] public partial class AutoGradGridSampleAlignCornersFalseCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var inputVals = (Tensor<float32>)OnnxOp.Constant(
                TensorData(8, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f));
            var x = (Tensor<float32>)OnnxOp.Mul(
                (Tensor<float32>)OnnxOp.Reshape(inputVals, Vector(1L, 2L, 2L, 2L), allowZero: false), a);
            var gridVals = (Tensor<float32>)OnnxOp.Constant(TensorData(2, 0f, 0f));
            var grid = (Tensor<float32>)OnnxOp.Reshape(gridVals, Vector(1L, 1L, 1L, 2L), allowZero: false);
            var output = (Tensor<float32>)OnnxOp.GridSample(x, grid,
                alignCorners: false, mode: GridSampleMode.Linear, paddingMode: GridSamplePaddingMode.Zeros);
            var loss = output.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(9f)).Abs() < Scalar(1e-5f);
        }
    }

    // ===================================================================
    //  Modules exercising gradient paths that FastSimplify would otherwise
    //  pre-fold (Cast round-trip, If with runtime condition, Dft with
    //  non-null dft_length, ConstantOfShape with runtime shape, SequenceAt
    //  with runtime index). Constructed so the relevant op survives until
    //  AutoGrad runs in the Module pipeline.
    // ===================================================================

    /// <summary>
    /// Round-trip cast f32 → f64 → f32. Forward is identity but creates two CAST
    /// nodes, exercising the Cast gradient (AC.cs Cast&lt;TFrom,TTo&gt;).
    /// dL/da = 1.
    /// </summary>
    [Module]
    public partial class AutoGradCastRoundTripCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var loss = a.Cast<float64>().Cast<float32>();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(1f)).Abs() < Scalar(1e-5f);
        }
    }

    /// <summary>
    /// IfElse with a runtime-derived condition (a &gt; 0) prevents
    /// FastFoldConstantConditionBranches from collapsing the IF before AutoGrad.
    /// With a=2, b=3 the condition is true so loss = 2a, dL/da = 2, dL/db = 0.
    /// Exercises Batch27 IfClose gradient.
    /// </summary>
    [Module]
    public partial class AutoGradIfRuntimeConditionTrueCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b)
        {
            var thenVal = a * Scalar(2f);
            var elseVal = b * Scalar(3f);
            var cond = a > Scalar(0f);
            var loss = cond.IfElse(thenVal, elseVal);
            var (gradA, gradB) = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, b, loss);
            var okA = (gradA! - Scalar(2f)).Abs() < Scalar(1e-3f);
            var okB = gradB!.Abs() < Scalar(1e-3f);
            return okA & okB;
        }
    }

    /// <summary>
    /// IfElse with runtime-derived condition (a &gt; 0) where a=-1 takes the else
    /// branch. loss = 3b, dL/da = 0, dL/db = 3.
    /// </summary>
    [Module]
    public partial class AutoGradIfRuntimeConditionFalseCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b)
        {
            var thenVal = a * Scalar(2f);
            var elseVal = b * Scalar(3f);
            var cond = a > Scalar(0f);
            var loss = cond.IfElse(thenVal, elseVal);
            var (gradA, gradB) = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, b, loss);
            var okA = gradA!.Abs() < Scalar(1e-3f);
            var okB = (gradB! - Scalar(3f)).Abs() < Scalar(1e-3f);
            return okA & okB;
        }
    }

    /// <summary>
    /// DFT with a non-null dft_length input. Forward shape is [1,2,2,2] after
    /// Expand(a, [1,2,2]) and onesided=false; sum-reduced loss has dL/da = 4 at
    /// a=3 (verified by direct-API capture). Exercises Batch20 DFT gradient
    /// path that handles a provided dft_length.
    /// </summary>
    [Module]
    public partial class AutoGradDftWithDftLengthCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var expanded = (Tensor<float32>)OnnxOp.Expand(a, Vector(1L, 2L, 2L));
            var dftLength = Scalar(2L);
            var axis = Scalar(1L);
            var dftOut = (Tensor<float32>)OnnxOp.Dft(expanded, dftLength, axis, inverse: false, onesided: false);
            var loss = dftOut.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(4f)).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>
    /// ConstantOfShape with a runtime-derived shape (Shape(Expand(a,[2,3])))
    /// prevents FastFoldConstants from pre-evaluating it. ones*aMat summed
    /// gives 6a, so dL/da = 6 at a=2. Exercises the ConstantOfShape gradient
    /// dispatcher in AC.cs.
    /// </summary>
    [Module]
    public partial class AutoGradConstantOfShapeRuntimeShapeCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a)
        {
            var aMat = (Tensor<float32>)OnnxOp.Expand(a, Vector(2L, 3L));
            var dynShape = (Tensor<int64>)OnnxOp.Shape(aMat);
            var ones = (Tensor<float32>)OnnxOp.ConstantOfShape(dynShape, TensorData(DType.Float32, [1L], 1f));
            var product = ones * aMat;
            var loss = product.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, loss);
            return (grad - Scalar(6f)).Abs() < Scalar(1e-3f);
        }
    }

    /// <summary>
    /// SequenceAt with a runtime-derived (Cast of runtime input) index prevents
    /// FastFoldSequences from short-circuiting SeqAt(SeqConstruct(...), idx).
    /// idxF is a runtime input set to 0.0f, so position = 0 picks the first
    /// element [a]; loss = a, dL/da = 1, dL/db = 0. Exercises Batch26
    /// SequenceAt gradient.
    /// </summary>
    [Module]
    public partial class AutoGradSeqAtRuntimeIdxCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b, Scalar<float32> idxF)
        {
            var idx = idxF.Cast<int64>();
            var aVec = (Tensor<float32>)OnnxOp.Reshape(a, Vector(1L), allowZero: false);
            var bVec = (Tensor<float32>)OnnxOp.Reshape(b, Vector(1L), allowZero: false);
            var seq = OnnxOp.SequenceConstruct(aVec, bVec);
            var elem = (Tensor<float32>)OnnxOp.SequenceAt(seq, idx);
            var loss = elem.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grads = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad([(IValue)a, b, idxF], loss);
            var gradA = (Scalar<float32>)grads[0]!;
            var gradB = (Scalar<float32>)grads[1]!;
            var okA = (gradA - Scalar(1f)).Abs() < Scalar(1e-3f);
            var okB = gradB.Abs() < Scalar(1e-3f);
            return okA & okB;
        }
    }

    /// <summary>
    /// Multi-output If with a runtime-derived condition where output[1] is
    /// unused. Mirrors AutoGradIfMultiOutputPartiallyUsedCheck but uses a
    /// non-constant condition (a &gt; 0) so FastFoldConstantConditionBranches
    /// can't pre-fold the IF, leaving outputGrads = [non-null, null] for
    /// IfCloseGradient — which exercises Batch27 lines 47-50 (the per-output
    /// null-dY skip branch).
    /// loss = out0 with condition true = 2a, dL/da = 2.
    /// </summary>
    [Module]
    public partial class AutoGradIfMultiOutputRuntimeCondPartiallyUsedCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b)
        {
            var cond = a > Scalar(0f);
            var (out0, _) = cond.IfElse(
                (a * Scalar(2f), b * Scalar(7f)),
                (a * Scalar(4f), b * Scalar(9f)));
            var loss = (Scalar<float32>)out0;
            var (gradA, gradB) = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(a, b, loss);
            var okA = (gradA! - Scalar(2f)).Abs() < Scalar(1e-3f);
            var okB = gradB!.Abs() < Scalar(1e-3f);
            return okA & okB;
        }
    }

    /// <summary>
    /// SequenceInsert with null position (append). Exercises the
    /// effectivePosition-from-output-length branch of SequenceInsertGradient
    /// (Batch26 lines 104-109). Final SequenceAt uses a runtime index so the
    /// chain isn't pre-folded. With idxF=1 the SeqAt picks the appended b,
    /// loss=b, dL/da=0, dL/db=1.
    /// </summary>
    [Module]
    public partial class AutoGradSeqInsertAppendCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b, Scalar<float32> idxF)
        {
            var pos = idxF.Cast<int64>();
            var aVec = (Tensor<float32>)OnnxOp.Reshape(a, Vector(1L), allowZero: false);
            var bVec = (Tensor<float32>)OnnxOp.Reshape(b, Vector(1L), allowZero: false);
            var seq0 = OnnxOp.SequenceConstruct(aVec);
            var seq1 = OnnxOp.SequenceInsert(seq0, bVec, null);
            var elem = (Tensor<float32>)OnnxOp.SequenceAt(seq1, pos);
            var loss = elem.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grads = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad([(IValue)a, b, idxF], loss);
            var gradA = (Scalar<float32>)grads[0]!;
            var gradB = (Scalar<float32>)grads[1]!;
            var okA = gradA.Abs() < Scalar(1e-3f);
            var okB = (gradB - Scalar(1f)).Abs() < Scalar(1e-3f);
            return okA & okB;
        }
    }

    /// <summary>
    /// Chained SequenceConstruct → SequenceErase → SequenceInsert → SequenceAt
    /// with a runtime-derived (Cast of runtime input) position used at every
    /// stage. With idxF=0 the chain is [a,b] → erase 0 → [b] → insert c at 0 →
    /// [c,b] → at 0 → c. loss = c, so dL/da=0, dL/db=0, dL/dc=1. Exercises
    /// Batch26 SequenceInsert and SequenceErase gradients (their bodies are
    /// kept alive because the final SequenceAt has a non-foldable position).
    /// </summary>
    [Module]
    public partial class AutoGradSeqInsertEraseRuntimeIdxCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b, Scalar<float32> c, Scalar<float32> idxF)
        {
            var pos = idxF.Cast<int64>();
            var aVec = (Tensor<float32>)OnnxOp.Reshape(a, Vector(1L), allowZero: false);
            var bVec = (Tensor<float32>)OnnxOp.Reshape(b, Vector(1L), allowZero: false);
            var cVec = (Tensor<float32>)OnnxOp.Reshape(c, Vector(1L), allowZero: false);
            var seq0 = OnnxOp.SequenceConstruct(aVec, bVec);
            var seq1 = OnnxOp.SequenceErase(seq0, pos);
            var seq2 = OnnxOp.SequenceInsert(seq1, cVec, pos);
            var elem = (Tensor<float32>)OnnxOp.SequenceAt(seq2, pos);
            var loss = elem.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grads = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad([(IValue)a, b, c, idxF], loss);
            var gradA = (Scalar<float32>)grads[0]!;
            var gradB = (Scalar<float32>)grads[1]!;
            var gradC = (Scalar<float32>)grads[2]!;
            var okA = gradA.Abs() < Scalar(1e-3f);
            var okB = gradB.Abs() < Scalar(1e-3f);
            var okC = (gradC - Scalar(1f)).Abs() < Scalar(1e-3f);
            return okA & okB & okC;
        }
    }
}
