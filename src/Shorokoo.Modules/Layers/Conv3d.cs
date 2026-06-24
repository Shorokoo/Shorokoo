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
/// 3-D convolution over NCDHW input, with fully dynamic
/// (hyperparameter-driven) geometry via the tensor-geometry <c>NN.Conv</c>
/// overload (SHRK_CONV, lowered to standard ONNX Conv at concretization).
/// Same dynamic-geometry path as <see cref="Conv2d"/> (SHRK_CONV's shape
/// arithmetic is spatial-rank-agnostic, so rank-5 inputs lower to a standard
/// 3-spatial-dim ONNX Conv). Cubic kernels: one <c>kernelSize</c> hyper covers
/// all three spatial dims; padding is symmetric (<c>padding</c> on every side).
/// Weight <c>[outChannels, inChannels/groups, k, k, k]</c> is
/// <see cref="KaimingUniform"/>-initialized; bias <c>[outChannels]</c> is
/// zero-initialized and created unconditionally — <c>useBias</c>
/// selects the trainable bias vs an all-zero constant vector.
/// </summary>
[Module]
public partial class Conv3d
{
    public static Tensor<float32> Inline(
        Tensor<float32> x,
        [Hyper] Scalar<int64> outChannels,
        [Hyper] Scalar<int64> kernelSize,
        [Hyper] Scalar<int64> stride,
        [Hyper] Scalar<int64> padding,
        [Hyper] Scalar<int64> dilation,
        [Hyper] Scalar<int64> groups,
        [Hyper] Scalar<bit> useBias)
    {
        Scalar<int64> inChannels = x.ShapeTensor()[1];
        var w = KaimingUniform.Init([outChannels, inChannels / groups, kernelSize, kernelSize, kernelSize]);

        var bTrainable = Zeros.Init([outChannels]).Vec();
        var b = useBias.IfElse(bTrainable, VectorFill(outChannels, 0f));

        return NN.Conv(x, w, b, AutoPad.NotSet,
            pads: [padding, padding, padding, padding, padding, padding],
            strides: [stride, stride, stride],
            dilations: [dilation, dilation, dilation],
            kernelShape: [kernelSize, kernelSize, kernelSize],
            group: groups);
    }
}
