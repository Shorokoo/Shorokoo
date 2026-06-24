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

/// <summary>Initializes the PReLU slope to 0.25 (PyTorch default).</summary>
[TrainableParamInitializer]
public static partial class PReLUAlphaInit
{
    public static Tensor<float32> Inline(Vector<int64> shape)
        => Globals.TensorFill(shape, 0.25f);
}

/// <summary>
/// Parametric ReLU with a single SHARED learnable slope (PyTorch's
/// <c>num_parameters=1</c> default, init 0.25): <c>y = relu(x) - alpha * relu(-x)</c>
/// (= <c>x</c> for <c>x &gt; 0</c>, <c>alpha * x</c> otherwise). The slope is a
/// trainable <c>[1]</c> parameter that broadcasts over the whole input. For a
/// per-channel variant, reshape an alpha of the channel count and broadcast it
/// manually against the channel axis.
/// </summary>
[Module]
public partial class PReLU
{
    public static Tensor<float32> Inline(Tensor<float32> x)
    {
        var alpha = PReLUAlphaInit.Init(Vector(1L)).Vec();
        return x.Relu() - alpha * (x * -1f).Relu();
    }
}

/// <summary>
/// Per-channel Parametric ReLU (PyTorch's <c>nn.PReLU(num_parameters=C)</c>):
/// <c>y = relu(x) - a_c * relu(-x)</c>, with a SEPARATE learnable slope per channel
/// (axis 1), each init 0.25. This differs from the shared <see cref="PReLU"/>
/// (a single <c>[1]</c> slope) only in the slope's shape: here the slope is a
/// trainable <c>[C]</c> parameter, broadcast as <c>[1, C, 1, …, 1]</c> over the
/// batch/spatial dims. The channel count <c>C</c> is read in-graph from the input,
/// so the module is rank-generic (<c>[N, C]</c>, <c>[N, C, L]</c>,
/// <c>[N, C, H, W]</c>, …, rank ≥ 2 with channels on axis 1). For a single shared
/// slope, use <see cref="PReLU"/>.
/// </summary>
[Module]
public partial class PReLUChannelwise
{
    public static Tensor<float32> Inline(Tensor<float32> x)
    {
        var shape = x.ShapeTensor();
        Scalar<int64> rank = shape.ShapeTensor()[0];          // rank as a graph scalar
        Scalar<int64> numChannels = shape[1];                 // C = axis 1, read in-graph

        // [C] trainable slope, all entries 0.25 at init (reuses PReLUAlphaInit).
        var alpha = PReLUAlphaInit.Init([numChannels]).Vec();

        // Broadcast shape [1, C, 1, …, 1] sized to the runtime rank (the BatchNorm
        // idiom, inlined here to keep activations decoupled from the norm helpers):
        // a leading 1, the channel count, then (rank - 2) trailing ones.
        Vector<int64> bcast = [Scalar(1L), numChannels];
        bcast = bcast.Concat(VectorFill(rank - 2L, 1L));
        var a = alpha.Reshape(bcast);

        // Same differentiable relu-built form as the shared PReLU, with a [C] slope.
        return x.Relu() - a * (x * -1f).Relu();
    }
}
