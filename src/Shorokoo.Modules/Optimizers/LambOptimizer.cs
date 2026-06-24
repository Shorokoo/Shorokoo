using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Onnx;
using static Shorokoo.Globals;

namespace Shorokoo.Modules.Optimizers;

/// <summary>
/// LAMB optimizer (You et al. 2019, arXiv:1904.00962, "Large Batch Optimization for Deep
/// Learning: Training BERT in 76 minutes"): Adam's per-coordinate adaptive direction scaled by
/// LARS's layer-wise <i>trust ratio</i>, enabling stable very-large-batch training.
/// Update rules (paper Algorithm 2; t is the timestep):
///   t_new     = t + 1
///   m_new     = beta1 * m + (1 - beta1) * grad
///   v_new     = beta2 * v + (1 - beta2) * grad^2
///   m_hat     = m_new / (1 - beta1^t_new)                 // bias correction
///   v_hat     = v_new / (1 - beta2^t_new)                 // bias correction
///   r         = m_hat / (sqrt(v_hat) + epsilon)           // Adam direction; eps OUTSIDE the sqrt
///   u         = r + weightDecay * param                   // decoupled WD added INSIDE the update
///   trust     = (‖param‖ &gt; 0 &amp;&amp; ‖u‖ &gt; 0) ? ‖param‖ / ‖u‖ : 1   // layer-wise trust ratio
///   param_new = param - learningRate * trust * u
/// where ‖x‖ = sqrt(sum(x^2)) is the true L2 norm reduced over ALL elements to a scalar, and
/// phi = identity (no trust-ratio clamp). Epsilon defaults to 1e-6 (the LAMB convention), not
/// Adam's 1e-8.
///
/// <b>Layer-wise = per-tensor trust ratio.</b> LAMB is defined "layer-wise": the trust ratio is
/// the L2 norm of a whole parameter tensor divided by the L2 norm of that tensor's update. Because
/// Shorokoo applies the optimizer graph once per trainable parameter tensor, the "layer" <i>is</i>
/// the parameter tensor, so the reduce-all-to-scalar ‖param‖ / ‖u‖ computed inside the graph is
/// already the per-tensor norm — matching the per-named-parameter granularity of PyTorch/timm/APEX
/// with no special machinery. The scalar trust ratio broadcasts across every element of the
/// parameter when scaling the update. The zero-guard falls back to a plain Adam step (trust = 1)
/// when either norm is 0, avoiding the 0/0 NaN.
///
/// The timestep t is carried as a scalar optimizer-state tensor alongside the param-shaped m and v,
/// created by <see cref="OptimizerScalarZeros"/>; it broadcasts against the param-shaped tensors in
/// the bias correction — one float per parameter.
///
/// Computation graph has 5 hyperparameters and 2 mandatory inputs:
///   Inputs:  [learningRate, beta1, beta2, epsilon, weightDecay, currentParam, grad]
///   Output:  [updatedParam]
/// The m and v moment states are created at the parameter's shape via the optimizer-owned
/// <see cref="OptimizerStateZeros"/> initializer and the scalar step via
/// <see cref="OptimizerScalarZeros"/> (never in the signature); their per-step updates are
/// registered via <see cref="Globals.StateUpdate"/>.
///
/// This module operates on individual tensors (one parameter at a time).
/// The TrainingRig applies it per-field across the trainable parameter struct and
/// threads m/v/step through the optimizer state struct between steps.
/// </summary>
[Module]
public partial class LambOptimizer
{
    public static Tensor<float32> Inline(
        Tensor<float32> currentParam,
        Tensor<float32> grad,
        [Hyper(0.001f)] Scalar<float32> learningRate,   // LAMB/APEX/timm default 1e-3
        [Hyper(0.9f)] Scalar<float32> beta1,            // first-moment EMA decay
        [Hyper(0.999f)] Scalar<float32> beta2,          // second-moment EMA decay
        [Hyper(1e-6f)] Scalar<float32> epsilon,         // NOTE: 1e-6 (LAMB convention), not Adam's 1e-8
        [Hyper(0.01f)] Scalar<float32> weightDecay)     // decoupled WD, added INSIDE the trust-ratio numerator
    {
        // State: Adam footprint — param-shaped m, v + a scalar step for bias correction.
        var m = OptimizerStateZeros.Init(currentParam.ShapeTensor());
        var v = OptimizerStateZeros.Init(currentParam.ShapeTensor());
        var step = OptimizerScalarZeros.Init();

        var one = Scalar(1.0f);
        var zero = Scalar(0.0f);

        // Advance the timestep (a scalar shared across the whole parameter).
        var newStep = step + one;

        // Adam moments + bias correction (eps OUTSIDE the sqrt, like AdamOptimizer).
        var newM = beta1 * m + (one - beta1) * grad;
        var newV = beta2 * v + (one - beta2) * grad * grad;
        var mHat = newM / (one - (Tensor<float32>)OnnxOp.Pow(beta1, newStep));
        var vHat = newV / (one - (Tensor<float32>)OnnxOp.Pow(beta2, newStep));
        var r = mHat / (vHat.Sqrt() + epsilon);             // Adam direction

        // Decoupled weight decay added INSIDE the update (the LAMB-defining placement).
        var u = r + weightDecay * currentParam;

        // Layer-wise (= per-tensor) trust ratio: ‖param‖ / ‖u‖, both whole-tensor L2 norms -> scalars.
        var wNorm = (currentParam * currentParam)
                        .Reduce(ReduceKind.Sum, axes: null, keepDims: false).Scalar().Sqrt();
        var uNorm = (u * u)
                        .Reduce(ReduceKind.Sum, axes: null, keepDims: false).Scalar().Sqrt();

        // Zero-guard: trust = (‖param‖>0 && ‖u‖>0) ? ‖param‖/‖u‖ : 1  (nested Where, exactly timm).
        // Falls back to a plain Adam step when either norm is 0 — and avoids 0/0 NaN.
        var ratio = wNorm / uNorm;
        var trust = (wNorm > zero).Where(
                        (uNorm > zero).Where(ratio, one),
                        one);

        // The LAMB step: param_new = param - lr * trust * u   (scalar trust broadcasts over the param).
        var updatedParam = currentParam - learningRate * trust * u;

        // Register state updates (same order as the state declarations: m, v, step).
        Globals.StateUpdate(m, newM);
        Globals.StateUpdate(v, newV);
        Globals.StateUpdate(step, newStep);

        return updatedParam;
    }
}
