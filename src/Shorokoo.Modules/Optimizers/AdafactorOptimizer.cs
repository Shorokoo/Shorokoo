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
/// Adafactor optimizer (Shazeer &amp; Stern 2018, arXiv:1804.04235), the
/// <b>NON-FACTORED</b> variant. This ships Adafactor's three rank-agnostic innovations
/// — relative step size, parameter scaling, and RMS update clipping, with a time-increasing
/// second-moment decay — over a <b>full param-shaped</b> second moment. It is exactly
/// PyTorch's <c>torch.optim.Adafactor</c> unfactored (<c>dim == 1</c>) branch, generalized to
/// every rank.
/// Update rules (PyTorch parity; t is the timestep):
///   beta2t    = 1 - t^beta2Decay                                   // increasing decay (beta2Decay = tau, default -0.8)
///   rho       = min(learningRate, 1/sqrt(t))                       // relative step (lr is the cap)
///   alpha     = max(epsilon2, RMS(param)) * rho                    // parameter scaling
///   param     = param * (1 - learningRate * weightDecay)           // decoupled weight decay
///   V_new     = beta2t * V + (1 - beta2t) * (grad*grad + epsilon1) // full second-moment EMA
///   U         = grad / max(sqrt(V_new), epsilon1)                  // raw update
///   U_clipped = U / max(1, RMS(U) / clipThreshold)                 // RMS update clipping (d = clipThreshold)
///   param_new = param - alpha * U_clipped
/// where <c>RMS(x) = sqrt(mean(x^2))</c> reduced over ALL elements to a scalar (rank-agnostic).
///
/// <b>NON-FACTORED DIVERGENCE (loud, deliberate, and the whole point of the scope).</b>
/// True Adafactor keeps only a row accumulator R (shape [r]) and a column accumulator C (shape
/// [c]) and reconstructs the per-element second moment as the rank-1 outer product
/// V = (R . C^T) / sum(R), dropping the optimizer state from r*c floats to r + c — its
/// SUBLINEAR-MEMORY trick and entire reason for existing. That factoring is <b>not
/// implemented here</b>, and is <b>not expressible</b> in Shorokoo's optimizer model: an
/// optimizer is ONE rank-agnostic graph applied per parameter, the factored vs unfactored
/// choice is a runtime rank branch (PyTorch's <c>dim &gt; 1</c>), and the two arms would have to
/// thread out differently-shaped state (R:[r], C:[c] vs V:[shape]) — which ONNX <c>If</c>
/// (shape-matched arms) cannot return. A fixed-rank-2-only factored variant would silently
/// corrupt every bias/embedding/conv weight, so it is rejected as unsafe. Consequently THIS
/// optimizer's state is a FULL param-shaped second moment <c>v</c> plus a scalar step — the
/// SAME footprint as Adam (one param-shaped buffer + one scalar). We keep Adafactor's
/// <i>update dynamics</i> but <b>NOT</b> its memory advantage. A user reaching for Adafactor
/// specifically for memory gets Adam-sized state.
///
/// Other parity notes:
///   * Relative step is always on; <c>learningRate</c> plays PyTorch's role as the cap on rho
///     (<c>min(lr, 1/sqrt(t))</c>), default 0.01 (= the paper's epsilon_lr = 1e-2). There is no
///     external-lr-only mode (booleans cannot be hyperparameters).
///   * epsilon1 is applied in the denominator floor and inside the g^2 accumulation; epsilon2
///     floors the parameter RMS in the scaling term. We follow PyTorch's epsilon placement.
///   * beta1 first-moment momentum is OFF (default everywhere); adding it would add another
///     param-shaped buffer, defeating the (already-compromised) memory story.
///   * At t = 1, beta2Decay = -0.8 gives 1 - 1^(-0.8) = 0, so V_new = grad^2 + epsilon1 (the EMA
///     is fully replaced on the first step).
///
/// Computation graph has 6 hyperparameters and 2 mandatory inputs:
///   Inputs:  [learningRate, beta2Decay, epsilon1, epsilon2, clipThreshold, weightDecay,
///            currentParam, grad]
///   Output:  [updatedParam]
/// The full second-moment state v is created inside the body at the parameter's shape via the
/// optimizer-owned <see cref="OptimizerStateZeros"/> initializer and the scalar step via
/// <see cref="OptimizerScalarZeros"/> (never in the signature); their per-step updates are
/// registered via <see cref="Globals.StateUpdate"/>.
///
/// This module operates on individual tensors (one parameter at a time).
/// The TrainingRig applies it per-field across the trainable parameter struct and
/// threads v/step through the optimizer state struct between steps.
/// </summary>
[Module]
public partial class AdafactorOptimizer
{
    public static Tensor<float32> Inline(
        Tensor<float32> currentParam,
        Tensor<float32> grad,
        [Hyper(0.01f)] Scalar<float32> learningRate,    // caps rho = min(lr, 1/sqrt(t)); = paper's epsilon_lr = 1e-2
        [Hyper(-0.8f)] Scalar<float32> beta2Decay,      // tau in beta2t = 1 - t^tau (PyTorch beta2_decay)
        [Hyper(1e-30f)] Scalar<float32> epsilon1,       // denominator / accumulation floor
        [Hyper(1e-3f)] Scalar<float32> epsilon2,        // floor in alpha = max(epsilon2, RMS(param)) * rho
        [Hyper(1.0f)] Scalar<float32> clipThreshold,    // d in RMS update clipping
        [Hyper(0.0f)] Scalar<float32> weightDecay)      // decoupled weight decay, default off
    {
        // State: a FULL param-shaped second moment + a scalar step.
        // (No row/col factoring — not expressible in Shorokoo's per-param rank-agnostic graph; see summary.)
        var v = OptimizerStateZeros.Init(currentParam.ShapeTensor());
        var step = OptimizerScalarZeros.Init();

        var one = Scalar(1.0f);

        // Advance the timestep (a scalar shared across the whole parameter).
        var newStep = step + one;

        // Increasing second-moment decay: beta2t = 1 - t^tau (0 at t = 1, -> 1 as t grows).
        var beta2t = one - (Tensor<float32>)OnnxOp.Pow(newStep, beta2Decay);

        // Relative step size rho = min(lr, 1/sqrt(t)) and parameter-scaled effective lr.
        var rho = learningRate.Min(one / newStep.Sqrt());
        // RMS(param) = sqrt(mean(param^2)) reduced over ALL axes -> scalar (rank-agnostic).
        var rmsTheta = (currentParam * currentParam).Reduce(ReduceKind.Mean, keepDims: false).Scalar().Sqrt();
        var alpha = epsilon2.Max(rmsTheta) * rho;       // max(epsilon2, RMS(param)) * rho

        // Decoupled weight decay (no-op when weightDecay == 0).
        var decayed = currentParam * (one - learningRate * weightDecay);

        // Full (unfactored) second-moment EMA: V <- beta2t * V + (1 - beta2t) * (g^2 + epsilon1).
        var newV = beta2t * v + (one - beta2t) * (grad * grad + epsilon1);

        // Raw update and RMS clipping: U = g / max(sqrt(V), epsilon1);  Uhat = U / max(1, RMS(U)/d).
        var u = grad / newV.Sqrt().Max(epsilon1);
        var rmsU = (u * u).Reduce(ReduceKind.Mean, keepDims: false).Scalar().Sqrt();   // RMS(U), reduce-all -> scalar
        var uClipped = u / one.Max(rmsU / clipThreshold);

        var updatedParam = decayed - alpha * uClipped;

        // Register state updates (same order as the state declarations: v, step).
        Globals.StateUpdate(v, newV);
        Globals.StateUpdate(step, newStep);

        return updatedParam;
    }
}
