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
/// RMSprop optimizer (Tieleman &amp; Hinton), with optional momentum (PyTorch semantics).
/// Update rules:
///   sq_new  = alpha * sq + (1 - alpha) * grad^2
///   buf_new = momentum * buf + grad / (sqrt(sq_new) + epsilon)
///   param_new = param - learningRate * buf_new
///
/// With momentum = 0 (the default) buf_new reduces to the plain RMSprop step
/// grad / (sqrt(sq_new) + epsilon), so a single module covers both variants.
///
/// Computation graph has 4 hyperparameters and 2 mandatory inputs:
///   Inputs:  [learningRate, alpha, epsilon, momentum, currentParam, grad]
///   Output:  [updatedParam]
/// The squareAvg and momentumBuffer states are created inside the body at the parameter's
/// shape via the optimizer-owned <see cref="OptimizerStateZeros"/> initializer (never in
/// the signature) and their per-step updates are registered via <see cref="Globals.StateUpdate"/>.
///
/// This module operates on individual tensors (one parameter at a time).
/// The TrainingRig applies it per-field across the trainable parameter struct.
/// </summary>
[Module]
public partial class RMSpropOptimizer
{
    public static Tensor<float32> Inline(
        Tensor<float32> currentParam,
        Tensor<float32> grad,
        [Hyper(0.01f)] Scalar<float32> learningRate,
        [Hyper(0.99f)] Scalar<float32> alpha,
        [Hyper(1e-8f)] Scalar<float32> epsilon,
        [Hyper(0.0f)] Scalar<float32> momentum)
    {
        var squareAvg = OptimizerStateZeros.Init(currentParam.ShapeTensor());
        var momentumBuffer = OptimizerStateZeros.Init(currentParam.ShapeTensor());

        var one = Scalar(1.0f);

        var newSquareAvg = alpha * squareAvg + (one - alpha) * grad * grad;
        var newBuffer = momentum * momentumBuffer + grad / (newSquareAvg.Sqrt() + epsilon);
        var updatedParam = currentParam - learningRate * newBuffer;

        Globals.StateUpdate(squareAvg, newSquareAvg);
        Globals.StateUpdate(momentumBuffer, newBuffer);

        return updatedParam;
    }
}
