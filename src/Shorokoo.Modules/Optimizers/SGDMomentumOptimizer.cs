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
/// SGD with Momentum optimizer.
/// Update rules:
///   v_new = momentum * v_old + grad
///   param_new = param - learningRate * v_new
///
/// Computation graph has 2 hyperparameters and 2 mandatory inputs:
///   Inputs:  [learningRate, momentumCoeff, currentParam, grad]
///   Output:  [updatedParam]
/// The velocity state is created inside the body at the parameter's shape via the
/// optimizer-owned <see cref="OptimizerStateZeros"/> initializer (never in the signature)
/// and its per-step update is registered via <see cref="Globals.StateUpdate"/>.
///
/// This module operates on individual tensors (one parameter at a time).
/// The TrainingRig applies it per-field across the trainable parameter struct,
/// initializes the velocity by running the state initializer per parameter, and
/// threads it through the optimizer state struct between steps.
/// </summary>
[Module]
public partial class SGDMomentumOptimizer
{
    public static Tensor<float32> Inline(
        Tensor<float32> currentParam,
        Tensor<float32> grad,
        [Hyper(0.01f)] Scalar<float32> learningRate,
        [Hyper(0.9f)] Scalar<float32> momentumCoeff)
    {
        var velocity = OptimizerStateZeros.Init(currentParam.ShapeTensor());

        var newVelocity = momentumCoeff * velocity + grad;
        Globals.StateUpdate(velocity, newVelocity);
        return currentParam - learningRate * newVelocity;
    }
}
