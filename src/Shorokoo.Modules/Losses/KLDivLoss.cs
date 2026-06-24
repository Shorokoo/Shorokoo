using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Onnx;
using static Shorokoo.Globals;

namespace Shorokoo.Modules.Losses;

/// <summary>
/// Kullback-Leibler divergence loss, "batchmean" reduction.
/// Convention (matching PyTorch's <c>nn.KLDivLoss(input=log-probs, target=probs)</c>):
/// <c>predictions</c> are LOG-probabilities (log q) and
/// <c>targets</c> are probabilities (p). Computes
/// <c>KL = (1/N) * Σ p * (log p - log q)</c> over all elements, where N is the
/// batch size (<c>predictions.DimTensor(0)</c>) — the mathematically correct
/// "batchmean" reduction (PyTorch's plain "mean" divides by the total element
/// count and does not equal the true KL divergence). The <c>p * log p = 0 when
/// p = 0</c> convention is enforced with a Where guard, so the <c>log p</c> term
/// is never evaluated for a zero target. It is a 2-input
/// (predictions, targets) → Scalar&lt;float32&gt; module, so it satisfies the
/// TrainingRig loss contract.
/// </summary>
[Module]
public partial class KLDivLoss
{
    public static Scalar<float32> Inline(Tensor<float32> predictions, Tensor<float32> targets)
    {
        var zeros = targets * 0f;
        var term = (targets > zeros).Where(targets * (targets.Ln() - predictions), zeros);
        var n = predictions.DimTensor(0).Cast<float32>();
        return term.Reduce(ReduceKind.Sum, keepDims: false).Scalar() / n;
    }
}
