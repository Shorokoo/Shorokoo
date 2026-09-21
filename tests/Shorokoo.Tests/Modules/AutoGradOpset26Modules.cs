namespace Shorokoo.Tests.Modules
{
    /// <summary>
    /// Gradient-correctness modules for the decomposable members of the opset 22-26
    /// op batch. Under Shorokoo's single-opset-21 export, Swish (@24) and
    /// RMSNormalization (@23) lower inline to opset-21 primitives, so their gradients
    /// flow through those primitives and are checked here by closed-form /
    /// two-sided-directional-derivative self-checks (same pattern as
    /// <c>AutoGradStructuralModules.cs</c>). TensorScatter (@24) is decomposed by its
    /// registered lowering instead of at its entry point, so the reverse walk sees the
    /// mask and gather it is made of; its two checks below take the expected gradients
    /// from outside the graph, which is what keeps them a check against numbers rather
    /// than against another expression of the same rule. The non-decomposable ops
    /// (Attention, RotaryEmbedding, BitCast, CumProd) throw
    /// <c>NotImplementedException</c> from their <see cref="OnnxOp"/> entry point, so
    /// there is no graph to differentiate — <c>AutoGradOpset26Tests</c> asserts that
    /// authoring throw directly rather than through a module.
    /// </summary>

    // ===================================================================
    //  Swish: default alpha + explicit alpha (FD self-check)
    // ===================================================================

    /// <summary>
    /// loss = Σ Swish(x) + Σ Swish(x, alpha=2) — covers both the default-alpha and
    /// explicit-alpha paths of the Swish gradient (s + a·x·s·(1−s)) in one
    /// two-sided directional-derivative check.
    /// </summary>
    [Module]
    public partial class AutoGradSwishCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var loss = SwishLoss(x);
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);

            var h = Scalar(1e-3f);
            var pert = h * grad;
            var deriv = (SwishLoss(x + pert) - SwishLoss(x - pert)) / (Scalar(2f) * h);
            var gradNormSq = (grad * grad).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            return (deriv - gradNormSq).Abs() < Scalar(1e-3f) * (gradNormSq.Abs() + Scalar(1f));
        }

        private static Scalar<float32> SwishLoss(Tensor<float32> x)
            => (NN.Swish(x) + NN.Swish(x, alpha: 2.0f))
                .Reduce(ReduceKind.Sum, keepDims: false).Scalar();
    }

    // ===================================================================
    //  RMSNormalization: dX (default axis −1 + explicit epsilon) and dScale
    //  (explicit positive axis) via FD self-checks
    // ===================================================================

    /// <summary>
    /// RMSNormalization input gradient, default axis −1 with an explicit epsilon.
    /// loss = Σ (Y ⊙ W) with constant per-feature weights so the gradient mixes
    /// the direct invRms·g term and the −x·invRms³·mean(g·x) coupling term.
    /// Self-checking via two-sided directional derivative.
    /// </summary>
    [Module]
    public partial class AutoGradRmsNormInputCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var scale = Vector(0.5f, 1f, 2f).Tensor();
            var loss = RmsLoss(x, scale);
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);

            var h = Scalar(1e-3f);
            var pert = h * grad;
            var deriv = (RmsLoss(x + pert, scale) - RmsLoss(x - pert, scale)) / (Scalar(2f) * h);
            var gradNormSq = (grad * grad).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            return (deriv - gradNormSq).Abs() < Scalar(1e-3f) * (gradNormSq.Abs() + Scalar(1f));
        }

        private static Scalar<float32> RmsLoss(Tensor<float32> x, Tensor<float32> scale)
        {
            var y = NN.RMSNormalization(x, scale, epsilon: 1e-3f);
            var weights = Vector(1f, 2f, 3f).Tensor();
            return (y * weights).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        }
    }

    /// <summary>
    /// RMSNormalization scale gradient with an explicit POSITIVE axis (1) —
    /// dScale = ReverseBroadcast(grad ⊙ xHat, scale shape). The scale is a
    /// trainable param (InitSimple) so AutoGrad targets it directly; FD is exact up
    /// to float noise because the loss is linear in scale.
    /// </summary>
    [Module]
    public partial class AutoGradRmsNormScaleCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var scale = InitSimple.Init([Scalar(3L)]);
            var loss = RmsLoss(x, scale);
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(scale, loss);

            var h = Scalar(1e-3f);
            var pert = h * grad;
            var deriv = (RmsLoss(x, scale + pert) - RmsLoss(x, scale - pert)) / (Scalar(2f) * h);
            var gradNormSq = (grad * grad).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            return (deriv - gradNormSq).Abs() < Scalar(1e-3f) * (gradNormSq.Abs() + Scalar(1f));
        }

        private static Scalar<float32> RmsLoss(Tensor<float32> x, Tensor<float32> scale)
        {
            var y = NN.RMSNormalization(x, scale, axis: 1L);
            var weights = Vector(1f, 2f, 3f).Tensor();
            return (y * weights).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        }
    }

    // ===================================================================
    //  TensorScatter: the written window routes the output gradient to
    //  `update`, everything else to `past_cache`. past = [[1,2,3],[4,5,6]],
    //  loss = Σ present ⊙ [[1,2,3],[4,5,6]], sequence axis −1.
    // ===================================================================

    /// <summary>
    /// Linear mode, write_indices [0,2], one-wide window: `update` claims (0,0) and (1,2), so
    /// dUpdate = [[1],[6]] and dPast is the weights with those two positions zeroed.
    /// </summary>
    [Module]
    public partial class AutoGradTensorScatterLinearGradientCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> past, Tensor<float32> update,
            Tensor<float32> expectedPast, Tensor<float32> expectedUpdate)
            => TensorScatterGradientCheck.Verdict(past, update, expectedPast, expectedUpdate,
                Vector(0L, 2L), null);
    }

    /// <summary>
    /// Circular mode, write_indices [2,1], two-wide window: batch 0's window wraps off the end
    /// onto (0,2) and (0,0), batch 1's stays at (1,1) and (1,2), so dUpdate = [[3,1],[5,6]] and
    /// dPast keeps only (0,1) and (1,0).
    /// </summary>
    [Module]
    public partial class AutoGradTensorScatterCircularGradientCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> past, Tensor<float32> update,
            Tensor<float32> expectedPast, Tensor<float32> expectedUpdate)
            => TensorScatterGradientCheck.Verdict(past, update, expectedPast, expectedUpdate,
                Vector(2L, 1L), TensorScatterMode.Circular);
    }

    internal static class TensorScatterGradientCheck
    {
        internal static Scalar<bit> Verdict(Tensor<float32> past, Tensor<float32> update,
            Tensor<float32> expectedPast, Tensor<float32> expectedUpdate,
            Vector<int64> writeIndices, TensorScatterMode? mode)
        {
            var weights = Vector(1f, 2f, 3f, 4f, 5f, 6f).Reshape(Vector(2L, 3L));
            var present = (Tensor<float32>)OnnxOp.TensorScatter(past, update, writeIndices, -1L, mode);
            var loss = (present * weights).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var (gradPast, gradUpdate) =
                Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad<Tensor<float32>, Tensor<float32>, float32>(
                    past, update, loss);
            var off = Worst(gradPast, expectedPast) + Worst(gradUpdate, expectedUpdate);
            return off < Scalar(1e-6f);
        }

        private static Scalar<float32> Worst(Tensor<float32>? actual, Tensor<float32> expected)
            => (actual!.Value - expected).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
    }
}
