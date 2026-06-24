using Shorokoo;
using Shorokoo.Modules;
using Shorokoo.Modules.Initializers;
using static Shorokoo.Globals;

[Module]
public partial class StackedLinear
{
    // Weight-tied feedforward: the same (w, b) applied `depth` times.
    // Intermediate layers get ReLU; the output layer does not.
    public static Tensor<float32> Inline(
        Tensor<float32> x,
        [Hyper] Scalar<int64> depth)
    {
        var n = x.DimTensor(1);
        var w = XavierUniform.Init([n, n]);    // trainable weight matrix
        var b = Zeros.Init([n]).Vec();          // trainable bias

        var h = x;
        foreach (var ctx in LoopAPI.Iterate(depth))
        {
            var z      = h.MatMul(w.Transpose([1L, 0L])) + b;
            var isLast = ctx.IterationIndex + Scalar(1L) == depth;
            h = isLast.IfElse(z, z.Relu());
        }
        return h;
    }
}
