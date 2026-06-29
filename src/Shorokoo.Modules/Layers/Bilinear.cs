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
/// Bilinear transformation: <c>y_k = x1ᵀ A_k x2 + b_k</c> (PyTorch's
/// <c>nn.Bilinear</c>). Computes, per output channel <c>k</c>, the bilinear form
/// of the two inputs' LAST axes:
/// <c>y[..., k] = Σ_{i,j} x1[..., i]·A[k,i,j]·x2[..., j] (+ b[k])</c>.
/// Weight <c>A</c> has shape <c>[outFeatures, in1Features, in2Features]</c>,
/// bias <c>b</c> is <c>[outFeatures]</c>. The two inputs must share their leading
/// (batch) dims; the contraction is over each input's last axis, with the batch
/// dims preserved. Weight and bias are both U(±1/√in1Features)-initialized
/// (PyTorch's bound, via <see cref="RecurrentUniform"/>) — note the bias is NOT
/// zero-initialized, unlike <see cref="Linear"/>. The bias parameter is created
/// unconditionally (both <c>IfElse</c> branches are built); <c>useBias</c>
/// selects whether it is added.
/// </summary>
[Module]
public partial class Bilinear
{
    public static Tensor<float32> Inline(
        Tensor<float32> x1,
        Tensor<float32> x2,
        [Hyper] Scalar<int64> in1Features,
        [Hyper] Scalar<int64> in2Features,
        [Hyper] Scalar<int64> outFeatures,
        [Hyper] Scalar<bit> useBias)
    {
        // Flatten leading (batch) dims to a single axis, like Linear, so the
        // einsum can use an explicit-label (no-ellipsis) equation and keep a
        // resolvable static shape.
        Vector<int64> lead = x1.TShape[..^1];                 // leading dims of x1, e.g. [N] or [B, T]
        var n = lead.Reduce(ReduceKind.Prod).Scalar();        // product of leading dims
        var x1f = x1.Reshape([n, in1Features]);               // [n, in1]
        var x2f = x2.Reshape([n, in2Features]);               // [n, in2]

        // Weight A [out, in1, in2] and bias b [out], both U(±1/√in1) (PyTorch bound).
        var a = RecurrentUniform.Init([outFeatures, in1Features, in2Features], in1Features);

        // Bilinear form per batch row n, per output channel k.
        var yf = (Tensor<float32>)OnnxOp.Einsum([x1f, a, x2f], "ni,kij,nj->nk");   // [n, out]

        var b = RecurrentUniform.Init([outFeatures], in1Features).Vec();           // [out] — NOT Zeros
        yf = useBias.IfElse(yf + b, yf);

        // Restore the original leading dims with out as the last axis.
        return yf.Reshape((Vector<int64>)[.. lead, outFeatures]);                  // [..., out]
    }
}
