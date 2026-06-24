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
/// Lion optimizer (EvoLved Sign Momentum; Chen et al. 2023, arXiv:2302.06675):
/// a sign-momentum method with decoupled weight decay (AdamW-style), discovered by
/// symbolic program search.
/// Update rules:
///   update    = sign(beta1 * m + (1 - beta1) * grad)              // direction: +/-1 per element
///   param_new = param - learningRate * (update + weightDecay * param)   // decoupled WD, folded in
///   m_new     = beta2 * m + (1 - beta2) * grad                    // momentum EMA, decayed by beta2
///
/// Two defining properties:
///   * Sign-based step: every coordinate moves by exactly +/-learningRate, independent of the
///     gradient magnitude (the update magnitude is decoupled from the gradient scale). ONNX
///     Sign(0) = 0, so a coordinate whose blend is exactly zero contributes no update.
///   * Half the optimizer state of Adam/AdamW: ONLY the momentum EMA buffer m is stored — no
///     second moment v, no bias correction, and (deliberately) no timestep scalar.
///
/// SWAPPED BETA ROLES (read carefully — the easiest way to get Lion subtly wrong):
///   The roles of beta1 and beta2 are SWAPPED versus Adam. In Adam beta1 decays the first-moment
///   (momentum) EMA and beta2 decays the second-moment EMA. In Lion the stored momentum EMA m is
///   decayed by beta2 (m_new = beta2 * m + (1 - beta2) * grad), while beta1 appears ONLY inside the
///   sign interpolation that forms the per-step update DIRECTION (sign(beta1 * m + (1 - beta1) * grad))
///   and never updates a stored buffer. The default (beta1 = 0.9, beta2 = 0.99) looks Adam-like but
///   means something different: the carried buffer uses beta2; the per-step look-ahead blend uses beta1.
///
/// Usability note (lr / wd scale vs Adam): because the sign step has unit magnitude, the effective
/// per-step move is learningRate itself (not learningRate * ||grad||). A suitable Lion learning rate
/// is typically 3-10x SMALLER than AdamW's, and the weight decay 3-10x LARGER, to match Adam's
/// effective decay (= learningRate * weightDecay). Lion has no epsilon (its update has no division).
///
/// Computation graph has 4 hyperparameters and 2 mandatory inputs:
///   Inputs:  [learningRate, beta1, beta2, weightDecay, currentParam, grad]
///   Output:  [updatedParam]
/// The single momentum state m is created inside the body at the parameter's shape via the
/// optimizer-owned <see cref="OptimizerStateZeros"/> initializer (never in the signature) and its
/// per-step update is registered via <see cref="Globals.StateUpdate"/>. There is no scalar step state.
///
/// This module operates on individual tensors (one parameter at a time).
/// The TrainingRig applies it per-field across the trainable parameter struct and
/// threads m through the optimizer state struct between steps.
/// </summary>
[Module]
public partial class LionOptimizer
{
    public static Tensor<float32> Inline(
        Tensor<float32> currentParam,
        Tensor<float32> grad,
        [Hyper(0.0001f)] Scalar<float32> learningRate,   // ~10x smaller than Adam's 1e-3
        [Hyper(0.9f)] Scalar<float32> beta1,             // update-blend coeff (sign interpolation)
        [Hyper(0.99f)] Scalar<float32> beta2,            // momentum EMA decay (note: NOT 0.999)
        [Hyper(0.0f)] Scalar<float32> weightDecay)       // decoupled; users set ~10x Adam's wd
    {
        // The ONLY state: the momentum EMA buffer (no v, no step scalar).
        var m = OptimizerStateZeros.Init(currentParam.ShapeTensor());

        var one = Scalar(1.0f);

        // Update direction: sign of the beta1-blend of momentum and gradient (+/-1 per element).
        var update = (beta1 * m + (one - beta1) * grad).Sign();

        // Decoupled weight decay folded into the step (== param * (1 - lr * wd) - lr * update).
        var updatedParam = currentParam - learningRate * (update + weightDecay * currentParam);

        // Momentum EMA, decayed by beta2 (the SWAPPED role); registered as the single state update.
        var newM = beta2 * m + (one - beta2) * grad;
        Globals.StateUpdate(m, newM);

        return updatedParam;
    }
}
