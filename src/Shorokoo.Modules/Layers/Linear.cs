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
/// Fully-connected (dense) layer: <c>y = x @ W^T (+ b)</c>.
/// Flattens all trailing input dimensions into the feature axis, so an input
/// of shape <c>[N, d1, d2, ...]</c> is treated as <c>[N, d1*d2*...]</c>.
/// Weight <c>[outFeatures, inFeatures]</c> is <see cref="KaimingUniform"/>-initialized;
/// bias <c>[outFeatures]</c> is zero-initialized. The bias parameter is created
/// unconditionally (both <c>IfElse</c> branches are built); <c>useBias</c>
/// selects whether it is added.
/// </summary>
[Module]
public partial class Linear
{
    public static Tensor<float32> Inline(
        Tensor<float32> x,
        [Hyper] Scalar<int64> outFeatures,
        [Hyper] Scalar<bit> useBias)
    {
        var batchSize = x.DimTensor(0);
        var inFeatures = x.TShape[1..^0].Reduce(ReduceKind.Prod).Scalar();
        var xFlat = x.Reshape([batchSize, inFeatures]);

        var w = KaimingUniform.Init([outFeatures, inFeatures]);
        var y = xFlat.MatMul(w.Transpose(1L, 0L));

        var b = Zeros.Init([outFeatures]).Vec();
        return useBias.IfElse(y + b, y);
    }
}
