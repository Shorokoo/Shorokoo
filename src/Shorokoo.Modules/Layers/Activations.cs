using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Onnx;
using static Shorokoo.Globals;

namespace Shorokoo.Modules.Layers;

// Only activations that carry hyperparameters get modules. Plain activations
// are one-liner tensor methods and need no wrapper:
//   x.Relu(), x.Gelu(), x.Sigmoid(), x.Tanh(), x.Softmax(axis), x.HardSwish(),
//   x.Selu(), x.Softplus(), x.Mish(), x.LogSoftmax(axis), ...
// The ONNX LeakyRelu/Elu ops take alpha as a *static attribute*, so the
// modules below build the elementwise formula in-graph to allow a true
// [Hyper] alpha.

/// <summary>
/// LeakyReLU with hyperparameter slope: <c>y = relu(x) - alpha * relu(-x)</c>
/// (= <c>x</c> for <c>x &gt; 0</c>, <c>alpha * x</c> otherwise).
/// </summary>
[Module]
public partial class LeakyReLU
{
    public static Tensor<float32> Inline(
        Tensor<float32> x,
        [Hyper] Scalar<float32> alpha)
        => x.Relu() - alpha * (x * -1f).Relu();
}

/// <summary>
/// ELU with hyperparameter alpha:
/// <c>y = relu(x) + alpha * (exp(min(x, 0)) - 1)</c>
/// (= <c>x</c> for <c>x &gt; 0</c>, <c>alpha * (exp(x) - 1)</c> otherwise).
/// </summary>
[Module]
public partial class ELU
{
    public static Tensor<float32> Inline(
        Tensor<float32> x,
        [Hyper] Scalar<float32> alpha)
        => x.Relu() + alpha * (x.Min(Scalar(0f)).Exp() - Scalar(1f));
}
