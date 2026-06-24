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
/// Gated linear unit helper. Like <see cref="Pooling"/>, this is a plain
/// graph-building helper (no hyperparameters), so it is a static method rather
/// than a <c>[Module]</c>.
/// </summary>
public static class GatedLinear
{
    /// <summary>
    /// Gated Linear Unit: splits <paramref name="x"/> into two equal halves
    /// <c>[a, b]</c> along <paramref name="axis"/> and returns
    /// <c>a * sigmoid(b)</c>. The input's size along <paramref name="axis"/>
    /// must be even.
    /// </summary>
    public static Tensor<T> GLU<T>(Tensor<T> x, long axis = -1) where T : FloatLike
    {
        var halves = x.Split(numOutputs: 2, axis);
        return halves[0] * halves[1].Sigmoid();
    }
}

/// <summary>
/// Gated Linear Unit (Dauphin et al. 2017) as a param-free module: splits the
/// input in half along the LAST axis into <c>[a, b]</c> and returns
/// <c>a * sigmoid(b)</c>. The input's size along the last axis must be even (it
/// is halved). This is PyTorch's <c>nn.GLU(dim=-1)</c> / Keras
/// <c>activations.glu</c> / <c>jax.nn.glu</c> default. The split axis is fixed at
/// <c>-1</c> (the universal default); for an arbitrary split axis use the static
/// helper <see cref="GatedLinear.GLU{T}"/>.
/// </summary>
[Module]
public partial class GLU
{
    public static Tensor<float32> Inline(Tensor<float32> x)
        => GatedLinear.GLU(x, axis: -1);
}
