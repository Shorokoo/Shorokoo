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
/// 2-D convolution over NCHW input, with fully dynamic (hyperparameter-driven)
/// geometry via the tensor-geometry <c>NN.Conv</c> overload (SHRK_CONV, lowered
/// to standard ONNX Conv at concretization). Square kernels: one
/// <c>kernelSize</c> hyper covers both spatial dims; padding is
/// symmetric (<c>padding</c> on every side).
/// Weight <c>[outChannels, inChannels/groups, k, k]</c> is
/// <see cref="KaimingUniform"/>-initialized; bias <c>[outChannels]</c> is
/// zero-initialized and created unconditionally — <c>useBias</c>
/// selects the trainable bias vs an all-zero constant vector.
/// </summary>
[Module]
public partial class Conv2d
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
        var w = KaimingUniform.Init([outChannels, inChannels / groups, kernelSize, kernelSize]);

        var bTrainable = Zeros.Init([outChannels]).Vec();
        var b = useBias.IfElse(bTrainable, VectorFill(outChannels, 0f));

        return NN.Conv(x, w, b, AutoPad.NotSet,
            pads: [padding, padding, padding, padding],
            strides: [stride, stride],
            dilations: [dilation, dilation],
            kernelShape: [kernelSize, kernelSize],
            group: groups);
    }
}

/// <summary>
/// 1-D convolution over NCL input. Same dynamic-geometry path as
/// <see cref="Conv2d"/> (SHRK_CONV's shape arithmetic is spatial-rank-agnostic,
/// so rank-3 inputs lower to a standard 1-spatial-dim ONNX Conv).
/// Weight <c>[outChannels, inChannels/groups, k]</c>.
/// </summary>
[Module]
public partial class Conv1d
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
        var w = KaimingUniform.Init([outChannels, inChannels / groups, kernelSize]);

        var bTrainable = Zeros.Init([outChannels]).Vec();
        var b = useBias.IfElse(bTrainable, VectorFill(outChannels, 0f));

        return NN.Conv(x, w, b, AutoPad.NotSet,
            pads: [padding, padding],
            strides: [stride],
            dilations: [dilation],
            kernelShape: [kernelSize],
            group: groups);
    }
}

/// <summary>
/// 2-D transposed convolution over NCHW input.
/// There is no attribute-tensorized ConvTranspose variant (only Conv has a
/// SHRK_CONV lowering), so geometry cannot be hyperparameter-driven: this
/// module uses the standard ONNX ConvTranspose with all geometry attributes
/// left at their spec defaults — stride 1, no padding, dilation 1, group 1 —
/// and the kernel shape inferred from the weight tensor (which IS dynamic:
/// <c>[inChannels, outChannels, k, k]</c> from the hypers). For other
/// stride/padding combinations call <c>NN.ConvTranspose</c> directly with
/// static attribute values.
/// </summary>
[Module]
public partial class ConvTranspose2d
{
    public static Tensor<float32> Inline(
        Tensor<float32> x,
        [Hyper] Scalar<int64> outChannels,
        [Hyper] Scalar<int64> kernelSize,
        [Hyper] Scalar<bit> useBias)
    {
        Scalar<int64> inChannels = x.ShapeTensor()[1];
        var w = KaimingUniform.Init([inChannels, outChannels, kernelSize, kernelSize]);

        var bTrainable = Zeros.Init([outChannels]).Vec();
        var b = useBias.IfElse(bTrainable, VectorFill(outChannels, 0f));

        return NN.ConvTranspose(x, w, b, AutoPad.NotSet,
            dilations: null, group: 1L, kernelShape: null,
            outputPadding: null, outputShape: null, pads: null, strides: null);
    }
}
