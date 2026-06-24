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
/// Adagrad optimizer (Duchi et al.).
/// Update rules:
///   acc_new = acc + grad^2
///   param_new = param - learningRate * grad / (sqrt(acc_new) + epsilon)
///
/// Computation graph has 2 hyperparameters and 2 mandatory inputs:
///   Inputs:  [learningRate, epsilon, currentParam, grad]
///   Output:  [updatedParam]
/// The accumulator state is created inside the body at the parameter's shape via the
/// optimizer-owned <see cref="OptimizerStateZeros"/> initializer (never in the signature)
/// and its per-step update is registered via <see cref="Globals.StateUpdate"/>.
///
/// This module operates on individual tensors (one parameter at a time).
/// The TrainingRig applies it per-field across the trainable parameter struct and
/// threads the accumulator through the optimizer state struct between steps.
/// </summary>
[Module]
public partial class AdagradOptimizer
{
    public static Tensor<float32> Inline(
        Tensor<float32> currentParam,
        Tensor<float32> grad,
        [Hyper(0.01f)] Scalar<float32> learningRate,
        [Hyper(1e-10f)] Scalar<float32> epsilon)
    {
        var accumulator = OptimizerStateZeros.Init(currentParam.ShapeTensor());

        var newAccumulator = accumulator + grad * grad;
        var updatedParam = currentParam - learningRate * grad / (newAccumulator.Sqrt() + epsilon);

        Globals.StateUpdate(accumulator, newAccumulator);

        return updatedParam;
    }
}
