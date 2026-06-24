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
/// Adadelta optimizer (Zeiler 2012): an adaptive per-dimension method that needs
/// no manually-set learning rate, using two exponentially-decaying accumulators
/// and a units-correcting update.
/// Update rules (PyTorch torch.optim.Adadelta):
///   squareAvg_new = rho * squareAvg + (1 - rho) * grad^2          // E[g^2]_t
///   delta         = sqrt(accDelta + eps) / sqrt(squareAvg_new + eps) * grad
///   accDelta_new  = rho * accDelta + (1 - rho) * delta^2          // E[Δx^2]_t
///   param_new     = param - learningRate * delta
///
/// Two load-bearing details (both matching the paper and PyTorch):
///   * eps is INSIDE both square roots (sqrt(acc + eps)), not added after the sqrt
///     like Adagrad/RMSprop/Adam — at step 1 the numerator is sqrt(0 + eps) = sqrt(eps),
///     which sets the bootstrap step scale.
///   * the delta numerator uses the OLD accDelta (E[Δx^2]_{t-1}, before this step's
///     update), and accDelta_new is then computed FROM that delta. The graph is pure
///     SSA, so accDelta and accDelta_new are distinct values and the wiring is correct.
///
/// learningRate is PyTorch's overall step multiplier (default 1.0 = paper-faithful,
/// where the self-computed RMS[Δx]/RMS[g] ratio plays the role of the learning rate);
/// it is NOT the primary learning rate. Adadelta has no timestep, so there is no scalar
/// step state (contrast Adam).
///
/// Computation graph has 3 hyperparameters and 2 mandatory inputs:
///   Inputs:  [learningRate, rho, epsilon, currentParam, grad]
///   Output:  [updatedParam]
/// The squareAvg and accDelta accumulators are created inside the body at the parameter's
/// shape via the optimizer-owned <see cref="OptimizerStateZeros"/> initializer (never in
/// the signature) and their per-step updates are registered via <see cref="Globals.StateUpdate"/>.
///
/// This module operates on individual tensors (one parameter at a time).
/// The TrainingRig applies it per-field across the trainable parameter struct and
/// threads squareAvg/accDelta through the optimizer state struct between steps.
/// </summary>
[Module]
public partial class AdadeltaOptimizer
{
    public static Tensor<float32> Inline(
        Tensor<float32> currentParam,
        Tensor<float32> grad,
        [Hyper(1.0f)] Scalar<float32> learningRate,
        [Hyper(0.9f)] Scalar<float32> rho,
        [Hyper(1e-6f)] Scalar<float32> epsilon)
    {
        // Two param-shaped accumulators, both zero-filled (E[g^2]_0 = E[Δx^2]_0 = 0).
        var squareAvg = OptimizerStateZeros.Init(currentParam.ShapeTensor()); // E[g^2]
        var accDelta = OptimizerStateZeros.Init(currentParam.ShapeTensor());  // E[Δx^2]

        var one = Scalar(1.0f);

        // 1) Accumulate squared GRADIENT (this step's E[g^2]).
        var newSquareAvg = rho * squareAvg + (one - rho) * grad * grad;

        // 2) Compute Δx using the PREVIOUS step's E[Δx^2] (accDelta, BEFORE updating it),
        //    eps INSIDE both square roots: Δx = sqrt(E[Δx^2]_{t-1}+ε)/sqrt(E[g^2]_t+ε) · g.
        var delta = (accDelta + epsilon).Sqrt()
                    / (newSquareAvg + epsilon).Sqrt()
                    * grad;

        // 3) Accumulate squared UPDATE using the Δx just computed (this step's E[Δx^2]).
        var newAccDelta = rho * accDelta + (one - rho) * delta * delta;

        // 4) Apply, with PyTorch's lr multiplier (default 1.0): x -= lr · Δx.
        var updatedParam = currentParam - learningRate * delta;

        // Register state updates (same order as the state declarations: squareAvg, accDelta).
        Globals.StateUpdate(squareAvg, newSquareAvg);
        Globals.StateUpdate(accDelta, newAccDelta);

        return updatedParam;
    }
}
