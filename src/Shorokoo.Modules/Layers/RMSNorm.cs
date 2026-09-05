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
/// dimensions: <c>y = x / sqrt(mean(x², over those axes) + epsilon)</c>, times a
/// gain when <c>affine = true</c>. Unlike <see cref="LayerNorm"/> there is no
/// mean-subtraction and no bias — the affine here is gain-only, a per-element
/// <see cref="Ones"/> shaped like the trailing normalized dims and broadcast over the
/// leading dims. Built in-graph from reduce/sqrt/div/mul primitives so
/// <c>epsilon</c> can be a hyperparameter.
/// <para>
/// With <c>affine = false</c> the normalized <c>x̂</c> is returned directly — the
/// gain-free form nanochat and modded-nanoGPT use (Llama, Mistral, Qwen and Gemma all
/// keep the gain). <c>affine</c> is a <c>[Hyper]</c> bit fixed before concretization,
/// so <c>affine = false</c> folds the <c>IfElse</c> away and prunes the gain: no
/// checkpoint field, no gradient, no optimizer state, leaving the module with no
/// trainable parameters of its own (the same gate as
/// <see cref="GroupNorm"/>/<see cref="InstanceNorm"/> and <see cref="Linear"/>'s
/// <c>useBias</c>).
/// </para>
/// </summary>
[Module]
public partial class RMSNorm
{
    public static Tensor<float32> Inline(
        Tensor<float32> x,
        [Hyper] Scalar<int64> normalizedDims,
        [Hyper] Scalar<bit> affine,          // true = learnable gain; false = gain-free
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

        return affine.IfElse(xHat * gain, xHat);
    }
}
