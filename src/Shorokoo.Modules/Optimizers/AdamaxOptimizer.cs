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
/// Adamax optimizer (Kingma &amp; Ba §7.1): Adam with the exponentially weighted
/// infinity norm in place of the L2 second moment.
/// Update rules:
///   t_new = t + 1
///   m_new = beta1 * m + (1 - beta1) * grad
///   u_new = max(beta2 * u, |grad| + epsilon)        // eps inside the max (PyTorch)
///   param_new = param - (learningRate / (1 - beta1^t_new)) * m_new / u_new
/// No bias correction on u (the running max is unbiased from step 1).
///
/// The timestep t is carried as a scalar optimizer-state tensor alongside the
/// param-shaped m and u, created by <see cref="OptimizerScalarZeros"/>; it broadcasts
/// against the param-shaped tensors in the bias correction — one float per parameter.
///
/// Computation graph has 4 hyperparameters and 2 mandatory inputs:
///   Inputs:  [learningRate, beta1, beta2, epsilon, currentParam, grad]
///   Output:  [updatedParam]
/// The m and u states are created at the parameter's shape via the optimizer-owned
/// <see cref="OptimizerStateZeros"/> initializer and the scalar step via
/// <see cref="OptimizerScalarZeros"/> (never in the signature); their per-step updates
/// are registered via <see cref="Globals.StateUpdate"/>.
///
/// This module operates on individual tensors (one parameter at a time).
/// The TrainingRig applies it per-field across the trainable parameter struct and
/// threads m/u/step through the optimizer state struct between steps.
/// </summary>
[Module]
public partial class AdamaxOptimizer
{
    public static Tensor<float32> Inline(
        Tensor<float32> currentParam,
        Tensor<float32> grad,
        [Hyper(0.002f)] Scalar<float32> learningRate,
        [Hyper(0.9f)] Scalar<float32> beta1,
        [Hyper(0.999f)] Scalar<float32> beta2,
        [Hyper(1e-8f)] Scalar<float32> epsilon)
    {
        var m = OptimizerStateZeros.Init(currentParam.ShapeTensor());
        var u = OptimizerStateZeros.Init(currentParam.ShapeTensor());
        var step = OptimizerScalarZeros.Init();

        var one = Scalar(1.0f);

        // Advance the timestep (a scalar shared across the whole parameter).
        var newStep = step + one;

        // First moment (as in Adam) and the exponentially weighted infinity norm.
        var newM = beta1 * m + (one - beta1) * grad;
        var newU = (beta2 * u).Max(grad.Abs() + epsilon);   // eps INSIDE the max

        // Bias correction lives only on m; fold 1/(1 - beta1^t) into the step size.
        var biasCorrection = one - (Tensor<float32>)OnnxOp.Pow(beta1, newStep);
        var updatedParam = currentParam - (learningRate / biasCorrection) * newM / newU;

        // Register state updates (same order as the state declarations: m, u, step).
        Globals.StateUpdate(m, newM);
        Globals.StateUpdate(u, newU);
        Globals.StateUpdate(step, newStep);

        return updatedParam;
    }
}
