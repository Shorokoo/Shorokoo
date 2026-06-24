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
/// NAdam optimizer (Dozat 2016): Adam with Nesterov-accelerated momentum.
/// The first moment is blended with a look-ahead using a time-varying momentum
/// schedule mu_t and a running product of those coefficients.
/// Update rules (PyTorch torch.optim.NAdam):
///   t_new       = t + 1
///   mu_t        = beta1 * (1 - 0.5 * 0.96^(t_new * momentumDecay))
///   mu_next     = beta1 * (1 - 0.5 * 0.96^((t_new + 1) * momentumDecay))
///   muProduct_new  = muProduct * mu_t            // running product ∏_{i=1}^{t} mu_i
///   muProductNext  = muProduct_new * mu_next     // ∏_{i=1}^{t+1} mu_i (derived, not stored)
///   m_new = beta1 * m + (1 - beta1) * grad
///   v_new = beta2 * v + (1 - beta2) * grad^2
///   v_hat = v_new / (1 - beta2^t_new)
///   denom = sqrt(v_hat) + epsilon
///   param_new = param - learningRate
///               * ((1 - mu_t) / (1 - muProduct_new) * grad
///                  + mu_next / (1 - muProductNext) * m_new) / denom
///
/// Two scalar states are carried per parameter: the timestep t (<see cref="OptimizerScalarZeros"/>)
/// and the running momentum product muProduct (<see cref="OptimizerScalarOnes"/>, seeded at 1 so
/// the first step yields mu_1 rather than 0). Both broadcast against the param-shaped m / v.
///
/// Computation graph has 5 hyperparameters and 2 mandatory inputs:
///   Inputs:  [learningRate, beta1, beta2, epsilon, momentumDecay, currentParam, grad]
///   Output:  [updatedParam]
/// The m and v states are created at the parameter's shape via the optimizer-owned
/// <see cref="OptimizerStateZeros"/> initializer, the scalar step via
/// <see cref="OptimizerScalarZeros"/>, and the scalar muProduct via
/// <see cref="OptimizerScalarOnes"/> (never in the signature); their per-step updates are
/// registered via <see cref="Globals.StateUpdate"/>.
///
/// This module operates on individual tensors (one parameter at a time).
/// The TrainingRig applies it per-field across the trainable parameter struct and
/// threads m/v/step/muProduct through the optimizer state struct between steps.
/// Weight decay (coupled and decoupled) is out of scope, matching PyTorch's defaults
/// (weight_decay = 0, decoupled_weight_decay = False).
/// </summary>
[Module]
public partial class NAdamOptimizer
{
    public static Tensor<float32> Inline(
        Tensor<float32> currentParam,
        Tensor<float32> grad,
        [Hyper(0.002f)] Scalar<float32> learningRate,
        [Hyper(0.9f)] Scalar<float32> beta1,
        [Hyper(0.999f)] Scalar<float32> beta2,
        [Hyper(1e-8f)] Scalar<float32> epsilon,
        [Hyper(0.004f)] Scalar<float32> momentumDecay)
    {
        var m = OptimizerStateZeros.Init(currentParam.ShapeTensor());
        var v = OptimizerStateZeros.Init(currentParam.ShapeTensor());
        var step = OptimizerScalarZeros.Init();
        var muProduct = OptimizerScalarOnes.Init();

        var one = Scalar(1.0f);
        var half = Scalar(0.5f);
        var base96 = Scalar(0.96f);

        // Advance the timestep (a scalar shared across the whole parameter).
        var newStep = step + one;                          // t

        // Momentum schedule: mu_t (current step) and mu_next (look-ahead).
        var muT = beta1 * (one - half * (Scalar<float32>)OnnxOp.Pow(base96, newStep * momentumDecay));
        var muNext = beta1 * (one - half * (Scalar<float32>)OnnxOp.Pow(base96, (newStep + one) * momentumDecay));

        // Running product ∏ mu_i (current) and the look-ahead product (derived, not stored).
        var newMuProduct = muProduct * muT;                // ∏_{i=1}^{t} mu_i
        var muProductNext = newMuProduct * muNext;         // ∏_{i=1}^{t+1} mu_i

        // Biased first and second moment estimates (as in Adam).
        var newM = beta1 * m + (one - beta1) * grad;
        var newV = beta2 * v + (one - beta2) * grad * grad;

        // v bias correction: sqrt(v / (1 - beta2^t)) + eps (divide BEFORE sqrt, per PyTorch).
        var vHat = newV / (one - (Tensor<float32>)OnnxOp.Pow(beta2, newStep));
        var denom = vHat.Sqrt() + epsilon;

        // Nesterov-blended numerator over the shared denominator.
        var gradTerm = (one - muT) / (one - newMuProduct) * grad;      // (1 - mu_t)·g / (1 - ∏_t)
        var momTerm = muNext / (one - muProductNext) * newM;          // mu_next·m / (1 - ∏_{t+1})
        var updatedParam = currentParam - learningRate * (gradTerm + momTerm) / denom;

        // Register state updates (declaration order: m, v, step, muProduct).
        Globals.StateUpdate(m, newM);
        Globals.StateUpdate(v, newV);
        Globals.StateUpdate(step, newStep);
        Globals.StateUpdate(muProduct, newMuProduct);

        return updatedParam;
    }
}
