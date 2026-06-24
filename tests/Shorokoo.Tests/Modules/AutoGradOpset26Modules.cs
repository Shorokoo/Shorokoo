namespace Shorokoo.Tests.Modules
{
    /// <summary>
    /// Gradient-correctness modules for the decomposable members of the opset 22-26
    /// op batch. Under Shorokoo's single-opset-21 export, Swish (@24) and
    /// RMSNormalization (@23) lower inline to opset-21 primitives, so their gradients
    /// flow through those primitives and are checked here by closed-form /
    /// two-sided-directional-derivative self-checks (same pattern as
    /// <c>AutoGradStructuralModules.cs</c>). The non-decomposable ops (Attention,
    /// RotaryEmbedding, TensorScatter, BitCast, CumProd) throw
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
}
