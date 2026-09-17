using Shorokoo.Modules.Initializers;

namespace Shorokoo.Tests.Modules;

/// <summary>
/// A two-layer perceptron, written the way a Shorokoo model is written rather than assembled
/// out of graph nodes: a <c>[Module]</c> whose <c>Inline</c> is the forward pass, from which the
/// source generator derives the <c>ComputationGraph</c> that the side-by-side backend tests run.
///
/// <para>Trainable weights on both layers with a ReLU between them, so running it is
/// matmul-shaped work over parameters the framework concretized — the path a model actually
/// takes to a backend — and not an expression over its inputs alone. The biases are ones rather
/// than zeros so that the bias add is visible in the result if it is dropped.</para>
/// </summary>
[Module]
public partial class SideBySideMlp
{
    public static Tensor<float32> Inline(Tensor<float32> x)
    {
        var n = x.DimTensor(-1);
        var w1 = XavierUniform.Init([n, n]);
        var b1 = Ones.Init([n]).Vec();
        var w2 = XavierUniform.Init([n, n]);
        var b2 = Ones.Init([n]).Vec();
        return (x.MatMul(w1) + b1).Relu().MatMul(w2) + b2;
    }
}
