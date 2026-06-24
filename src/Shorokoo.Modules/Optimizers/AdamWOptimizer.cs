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
/// AdamW optimizer (Adam with decoupled weight decay).
/// Update rules:
///   m_new = beta1 * m + (1 - beta1) * grad
///   v_new = beta2 * v + (1 - beta2) * grad^2
///   param_new = param * (1 - learningRate * weightDecay) - learningRate * m_new / (sqrt(v_new) + epsilon)
///
/// Note: Bias correction for m and v is omitted here because it requires tracking
/// the timestep t across iterations. Without bias correction, the optimizer still
/// converges but may need slightly different hyperparameters in early iterations.
/// In practice, the effect is minimal after the first few steps.
///
/// Computation graph has 5 hyperparameters and 2 mandatory inputs:
///   Inputs:  [learningRate, beta1, beta2, epsilon, weightDecay, currentParam, grad]
///   Output:  [updatedParam]
/// The m and v moment states are created inside the body at the parameter's shape via the
/// optimizer-owned <see cref="OptimizerStateZeros"/> initializer (never in the signature)
/// and their per-step updates are registered via <see cref="Globals.StateUpdate"/>.
///
/// This module operates on individual tensors (one parameter at a time).
/// The TrainingRig applies it per-field across the trainable parameter struct and
/// threads m/v through the optimizer state struct between steps.
/// </summary>
[Module]
public partial class AdamWOptimizer
{
    public static Tensor<float32> Inline(
        Tensor<float32> currentParam,
        Tensor<float32> grad,
        [Hyper(0.001f)] Scalar<float32> learningRate,
        [Hyper(0.9f)] Scalar<float32> beta1,
        [Hyper(0.999f)] Scalar<float32> beta2,
        [Hyper(1e-8f)] Scalar<float32> epsilon,
        [Hyper(0.0001f)] Scalar<float32> weightDecay)
    {
        var m = OptimizerStateZeros.Init(currentParam.ShapeTensor());
        var v = OptimizerStateZeros.Init(currentParam.ShapeTensor());

        var one = Scalar(1.0f);

        // Update first moment estimate (mean of gradients)
        var newM = beta1 * m + (one - beta1) * grad;

        // Update second moment estimate (mean of squared gradients)
        var newV = beta2 * v + (one - beta2) * grad * grad;

        // Decoupled weight decay: applied directly to parameters
        var decayedParam = currentParam * (one - learningRate * weightDecay);

        // Parameter update: Adam step
        var updatedParam = decayedParam - learningRate * newM / (newV.Sqrt() + epsilon);

        // Register state updates
        Globals.StateUpdate(m, newM);
        Globals.StateUpdate(v, newV);

        return updatedParam;
    }
}
