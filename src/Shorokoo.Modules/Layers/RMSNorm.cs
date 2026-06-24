using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Modules.Initializers;
using Shorokoo.Onnx;
using static Shorokoo.Globals;

namespace Shorokoo.Modules.Layers;

/// <summary>
/// Root-mean-square layer normalization over the last <c>normalizedDims</c>
/// dimensions: <c>y = x / sqrt(mean(x², over those axes) + epsilon) * gain</c>.
/// Unlike <see cref="LayerNorm"/> there is no mean-subtraction and no bias —
/// only a trainable per-element gain shaped like the trailing normalized dims
/// (<see cref="Ones"/>, broadcast over the leading dims). Built in-graph from
/// reduce/sqrt/div/mul primitives so <c>epsilon</c> can be a hyperparameter.
/// </summary>
[Module]
public partial class RMSNorm
{
    public static Tensor<float32> Inline(
        Tensor<float32> x,
        [Hyper] Scalar<int64> normalizedDims,
        [Hyper] Scalar<float32> epsilon)
    {
        var shape = x.ShapeTensor();
        Scalar<int64> rank = shape.ShapeTensor()[0];
        var start = rank - normalizedDims;

        // axes = [rank - normalizedDims, ..., rank - 1]
        var axes = ((Tensor<int64>)OnnxOp.Range(start, rank, Scalar(1L))).Vec();

        var ms = (x * x).Reduce(ReduceKind.Mean, axes, keepDims: true);
        var xHat = x / (ms + epsilon).Sqrt();

        var paramShape = shape.Slice(start, rank);
        var gain = Ones.Init(paramShape);

        return xHat * gain;
    }
}
