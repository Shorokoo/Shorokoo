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
/// Binary cross-entropy over probabilities (predictions in (0, 1)):
/// <c>loss = -mean(t * ln(p) + (1 - t) * ln(1 - p))</c>.
/// Predictions are clamped to <c>[1e-7, 1 - 1e-7]</c> for numerical stability.
/// For raw logits use <see cref="BCEWithLogitsLoss"/> instead.
/// </summary>
[Module]
public partial class BCELoss
{
    public static Scalar<float32> Inline(Tensor<float32> predictions, Tensor<float32> targets)
    {
        var one = Scalar(1f);
        var p = predictions.Clip(Scalar(1e-7f), Scalar(1f - 1e-7f));
        var perElement = (targets * p.Ln() + (one - targets) * (one - p).Ln()) * -1f;
        return perElement.Reduce(ReduceKind.Mean, keepDims: false).Scalar();
    }
}
