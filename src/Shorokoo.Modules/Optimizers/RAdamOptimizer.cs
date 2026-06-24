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
/// RAdam — Rectified Adam (Liu et al. 2019, "On the Variance of the Adaptive
/// Learning Rate and Beyond"). Adam with a variance-rectification term on the
/// adaptive learning rate, plus a fallback to an un-adapted momentum step while
/// the adaptive variance is not yet tractable.
/// Update rules:
///   t_new = t + 1
///   m_new = beta1 * m + (1 - beta1) * grad
///   v_new = beta2 * v + (1 - beta2) * grad^2
///   mHat  = m_new / (1 - beta1^t_new)
///   rhoInf = 2/(1 - beta2) - 1
///   rho_t  = rhoInf - 2*t_new*beta2^t_new / (1 - beta2^t_new)
///   if rho_t > 5:                              // variance tractable -> rectified adaptive step
///     l_t = sqrt((1 - beta2^t_new) / (v_new + eps))
///     r_t = sqrt(((rho_t-4)(rho_t-2)*rhoInf) / ((rhoInf-4)(rhoInf-2)*rho_t))
///     param_new = param - learningRate * mHat * r_t * l_t
///   else:                                      // un-adapted momentum step
///     param_new = param - learningRate * mHat
/// The rho_t > 5 test is a runtime Scalar&lt;bit&gt; (rho_t depends on the scalar
/// step STATE). It is realized with a scalar <c>Where</c>: both the rectified and the
/// un-adapted param updates are built into the graph as a single straight-line
/// expression and the scalar condition selects between them at runtime (broadcasting
/// against the param-shaped arms) — no If subgraph. The m/v/step state updates are
/// identical in both arms, so each is registered with a single StateUpdate outside the
/// selection.
///
/// Computation graph has 4 hyperparameters and 2 mandatory inputs:
///   Inputs:  [learningRate, beta1, beta2, epsilon, currentParam, grad]
///   Output:  [updatedParam]
/// The m and v states are created at the parameter's shape via the optimizer-owned
/// <see cref="OptimizerStateZeros"/> initializer and the scalar step via
/// <see cref="OptimizerScalarZeros"/> (never in the signature); their per-step updates
/// are registered via <see cref="Globals.StateUpdate"/>.
///
/// This module operates on individual tensors (one parameter at a time).
/// The TrainingRig applies it per-field across the trainable parameter struct and
/// threads m/v/step through the optimizer state struct between steps.
/// </summary>
[Module]
public partial class RAdamOptimizer
{
    public static Tensor<float32> Inline(
        Tensor<float32> currentParam,
        Tensor<float32> grad,
        [Hyper(0.001f)] Scalar<float32> learningRate,
        [Hyper(0.9f)] Scalar<float32> beta1,
        [Hyper(0.999f)] Scalar<float32> beta2,
        [Hyper(1e-8f)] Scalar<float32> epsilon)
    {
        var m = OptimizerStateZeros.Init(currentParam.ShapeTensor());
        var v = OptimizerStateZeros.Init(currentParam.ShapeTensor());
        var step = OptimizerScalarZeros.Init();

        var one = Scalar(1.0f);
        var two = Scalar(2.0f);
        var four = Scalar(4.0f);
        var threshold = Scalar(5.0f);

        // Advance the timestep (a scalar shared across the whole parameter).
        var newStep = step + one;

        // Biased moments (as in Adam).
        var newM = beta1 * m + (one - beta1) * grad;
        var newV = beta2 * v + (one - beta2) * grad * grad;

        // beta2^t and the bias-correction factors (scalars).
        var beta2T = (Scalar<float32>)OnnxOp.Pow(beta2, newStep);
        var oneMinusBeta1T = one - (Scalar<float32>)OnnxOp.Pow(beta1, newStep);
        var oneMinusBeta2T = one - beta2T;

        // Bias-corrected first moment (param-shaped via scalar broadcast).
        var mHat = newM / oneMinusBeta1T;

        // SMA lengths.
        var rhoInf = two / (one - beta2) - one;                          // scalar constant
        var rhoT = rhoInf - two * newStep * beta2T / oneMinusBeta2T;     // scalar, grows with t

        // --- rectified ADAPTIVE branch ---
        var lT = (oneMinusBeta2T / (newV + epsilon)).Sqrt();            // = 1/sqrt(v̂), param-shaped
        var rT = (((rhoT - four) * (rhoT - two) * rhoInf)
                 / ((rhoInf - four) * (rhoInf - two) * rhoT)).Sqrt();   // scalar rectifier
        var rectifiedParam = currentParam - learningRate * mHat * rT * lT;

        // --- un-adapted MOMENTUM branch ---
        var unadaptedParam = currentParam - learningRate * mHat;

        // Runtime selection on rho_t > threshold (rho_t depends on the step STATE).
        // The scalar Where builds both arms as one straight-line expression and selects
        // at runtime, broadcasting the scalar condition against the param-shaped arms.
        var updatedParam = (rhoT > threshold).Where(rectifiedParam, unadaptedParam);

        // State updates are identical across branches; one StateUpdate each (m, v, step).
        Globals.StateUpdate(m, newM);
        Globals.StateUpdate(v, newV);
        Globals.StateUpdate(step, newStep);

        return updatedParam;
    }
}
