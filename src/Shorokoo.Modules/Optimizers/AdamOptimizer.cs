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
/// Adam optimizer (Kingma &amp; Ba) WITH bias correction.
/// Update rules:
///   t_new = t + 1
///   m_new = beta1 * m + (1 - beta1) * grad
///   v_new = beta2 * v + (1 - beta2) * grad^2
///   m_hat = m_new / (1 - beta1^t_new)
///   v_hat = v_new / (1 - beta2^t_new)
///   param_new = param - learningRate * m_hat / (sqrt(v_hat) + epsilon)
///
/// The timestep t is carried as a scalar optimizer-state tensor alongside the
/// param-shaped m and v. It is a single value (created by
/// <see cref="OptimizerScalarZeros"/>) that broadcasts against m̂ / v̂ during the
/// bias correction — one float per parameter rather than a redundant param-shaped copy.
///
/// Computation graph has 4 hyperparameters and 2 mandatory inputs:
///   Inputs:  [learningRate, beta1, beta2, epsilon, currentParam, grad]
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
public partial class AdamOptimizer
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

        // Advance the timestep (a scalar shared across the whole parameter).
        var newStep = step + one;

        // Update biased first and second moment estimates.
        var newM = beta1 * m + (one - beta1) * grad;
        var newV = beta2 * v + (one - beta2) * grad * grad;

        // Bias correction.
        var mHat = newM / (one - (Tensor<float32>)OnnxOp.Pow(beta1, newStep));
        var vHat = newV / (one - (Tensor<float32>)OnnxOp.Pow(beta2, newStep));

        var updatedParam = currentParam - learningRate * mHat / (vHat.Sqrt() + epsilon);

        // Register state updates (same order as the state declarations: m, v, step).
        Globals.StateUpdate(m, newM);
        Globals.StateUpdate(v, newV);
        Globals.StateUpdate(step, newStep);

        return updatedParam;
    }
}
