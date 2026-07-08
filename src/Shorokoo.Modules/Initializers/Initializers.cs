using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Onnx;
using static Shorokoo.Globals;

namespace Shorokoo.Modules.Initializers;

// ---------------------------------------------------------------------------
// Baseline trainable-parameter initializers.
//
// All initializers are shape-only ([TrainableParamInitializer] Inline(shape)):
// the codegen does support extra Inline parameters, but keeping the baseline
// set shape-only keeps every call site a one-liner (`Zeros.Init([c])`) and
// keeps the initializer graphs trivially resolvable at materialization time.
// Gains are therefore fixed, documented constants:
//   - Xavier initializers use gain = 1.
//   - Kaiming initializers use gain = sqrt(2) (the ReLU gain, fan-in mode).
//   - Random initializers draw from the parameter's OWN RNG stream — keyed by
//     its ModelId under the model's RNG identity (RngConfig; the default
//     deterministic identity when none is bound). Materialization is
//     deterministic and reproducible for a config, and two same-shape
//     parameters receive distinct values (SharedKey mode ties them for
//     reference tests). There is no per-initializer seed.
// Fan-in/fan-out are computed in-graph from the shape vector:
//   fanIn  = prod(shape) / shape[0]   (= shape[1] * receptive-field size)
//   fanOut = prod(shape) / shape[1]   (= shape[0] * receptive-field size)
// which matches the PyTorch convention for Linear ([out, in]) and Conv
// ([outC, inC/g, k...]) weight layouts. Xavier/Kaiming initializers require
// rank >= 2 shapes (use Zeros/Ones/Uniform/Normal for biases).
// ---------------------------------------------------------------------------

/// <summary>Shared in-graph fan-in / fan-out arithmetic for the variance-scaling initializers.</summary>
internal static class InitializerMath
{
    /// <summary>fanIn = prod(shape) / shape[0], as float32.</summary>
    public static Scalar<float32> FanIn(Vector<int64> shape)
    {
        var total = shape.Reduce(ReduceKind.Prod);
        Scalar<int64> dim0 = shape[0];
        return (total / dim0).Cast<float32>();
    }

    /// <summary>fanOut = prod(shape) / shape[1], as float32.</summary>
    public static Scalar<float32> FanOut(Vector<int64> shape)
    {
        var total = shape.Reduce(ReduceKind.Prod);
        Scalar<int64> dim1 = shape[1];
        return (total / dim1).Cast<float32>();
    }
}

/// <summary>Fills the parameter with 0.0 (biases, BatchNorm beta).</summary>
[TrainableParamInitializer]
public static partial class Zeros
{
    public static Tensor<float32> Inline(Vector<int64> shape)
        => Globals.TensorFill(shape, 0.0f);
}

/// <summary>Fills the parameter with 1.0 (BatchNorm/LayerNorm gamma).</summary>
[TrainableParamInitializer]
public static partial class Ones
{
    public static Tensor<float32> Inline(Vector<int64> shape)
        => Globals.TensorFill(shape, 1.0f);
}

/// <summary>
/// Constant initializer: fills the parameter with a caller-supplied scalar
/// <c>value</c> (every element == value). Deterministic (no RNG, no
/// seed) and shape-agnostic (works for biases and weights alike — no fan-in/out,
/// no rank requirement). The parameterized generalization of <see cref="Zeros"/>
/// (== Constant(0)) and <see cref="Ones"/> (== Constant(1)); mirrors PyTorch's
/// <c>nn.init.constant_(tensor, val)</c>, Keras's <c>Constant(value)</c>, and
/// JAX's <c>jax.nn.initializers.constant(value)</c>. Use it for non-zero bias
/// priors (e.g. a +1 LSTM forget-gate bias, a focal-loss detector head prior).
/// The value is taken as an extra Inline parameter (the <see cref="RecurrentUniform"/>
/// extra-param precedent), generating <c>Constant.Init(shape, value)</c>.
/// <see cref="Globals.TensorFill(Vector{int64}, float)"/> bakes a literal fill into
/// the ConstantOfShape value attribute, so it cannot take a runtime graph scalar;
/// the runtime <c>value</c> is therefore applied by scaling an all-ones
/// fill (<c>TensorFill(shape, 1.0f) * value</c>), the same fill-times-scalar shape the
/// variance-scaling initializers use.
/// </summary>
[TrainableParamInitializer]
public static partial class Constant
{
    public static Tensor<float32> Inline(Vector<int64> shape, Scalar<float32> value)
        => Globals.TensorFill(shape, 1.0f) * value;   // TensorFill takes a LITERAL; apply the runtime value via *
}

/// <summary>Uniform U(0, 1) initializer, stream-keyed (deterministic).</summary>
[TrainableParamInitializer]
public static partial class Uniform
{
    public static Tensor<float32> Inline(Vector<int64> shape)
        => Globals.RandomUniform(shape, low: 0.0f, high: 1.0f);
}

/// <summary>Standard normal N(0, 1) initializer, stream-keyed (deterministic). PyTorch's nn.Embedding default.</summary>
[TrainableParamInitializer]
public static partial class Normal
{
    public static Tensor<float32> Inline(Vector<int64> shape)
        => Globals.RandomNormal(shape, mean: 0.0f, scale: 1.0f);
}

/// <summary>
/// Xavier/Glorot uniform: U(-a, a) with a = sqrt(6 / (fanIn + fanOut)), gain = 1. Stream-keyed.
/// Requires a rank >= 2 shape.
/// </summary>
[TrainableParamInitializer]
public static partial class XavierUniform
{
    public static Tensor<float32> Inline(Vector<int64> shape)
    {
        var bound = (6.0f / (InitializerMath.FanIn(shape) + InitializerMath.FanOut(shape))).Sqrt();
        return Globals.RandomUniform(shape, low: -1.0f, high: 1.0f) * bound;
    }
}

/// <summary>
/// Xavier/Glorot normal: N(0, std) with std = sqrt(2 / (fanIn + fanOut)), gain = 1. Stream-keyed.
/// Requires a rank >= 2 shape.
/// </summary>
[TrainableParamInitializer]
public static partial class XavierNormal
{
    public static Tensor<float32> Inline(Vector<int64> shape)
    {
        var std = (2.0f / (InitializerMath.FanIn(shape) + InitializerMath.FanOut(shape))).Sqrt();
        return Globals.RandomNormal(shape, mean: 0.0f, scale: 1.0f) * std;
    }
}

/// <summary>
/// Kaiming/He uniform (fan-in mode, ReLU gain sqrt(2)): U(-b, b) with b = sqrt(6 / fanIn). Stream-keyed.
/// Requires a rank >= 2 shape. The default weight initializer for <see cref="Layers.Linear"/> and the Conv layers.
/// </summary>
[TrainableParamInitializer]
public static partial class KaimingUniform
{
    public static Tensor<float32> Inline(Vector<int64> shape)
    {
        var bound = (6.0f / InitializerMath.FanIn(shape)).Sqrt();
        return Globals.RandomUniform(shape, low: -1.0f, high: 1.0f) * bound;
    }
}

/// <summary>
/// Kaiming/He normal (fan-in mode, ReLU gain sqrt(2)): N(0, std) with std = sqrt(2 / fanIn). Stream-keyed.
/// Requires a rank >= 2 shape.
/// </summary>
[TrainableParamInitializer]
public static partial class KaimingNormal
{
    public static Tensor<float32> Inline(Vector<int64> shape)
    {
        var std = (2.0f / InitializerMath.FanIn(shape)).Sqrt();
        return Globals.RandomNormal(shape, mean: 0.0f, scale: 1.0f) * std;
    }
}

/// <summary>
/// Recurrent-weight uniform init: U(−1/√H, 1/√H) with the hidden dimension <c>H</c>
/// passed explicitly via <c>hiddenSize</c>, stream-keyed. This is PyTorch's
/// <c>nn.RNN</c>/<c>nn.LSTM</c>/<c>nn.GRU</c> default (<c>k = 1/hidden_size</c>,
/// every weight and bias drawn from <c>U(−√k, √k)</c>). The bound is keyed off the
/// caller-supplied hidden size rather than <c>shape[1]</c> because for the gated
/// cells <c>shape[1] = gates·H</c> (<c>4H</c> for LSTM, <c>3H</c> for GRU), so
/// <c>1/√shape[1]</c> would be <c>1/√(4H)</c> ≠ PyTorch's <c>1/√H</c>. For the
/// single-gate RNN where <c>shape[1] = H</c>, passing <c>Scalar(hiddenSize)</c>
/// reproduces the former <c>shape[1]</c>-derived bound exactly, so RNN init is
/// unchanged. Shared by the recurrent layers (<see cref="Layers.Recurrent"/>) so
/// RNN/LSTM/GRU init identically (one <c>1/√hidden_size</c> bound across all gates).
/// </summary>
[TrainableParamInitializer]
public static partial class RecurrentUniform
{
    public static Tensor<float32> Inline(Vector<int64> shape, Scalar<int64> hiddenSize)
    {
        var bound = Scalar(1.0f) / hiddenSize.Cast<float32>().Sqrt();
        return Globals.RandomUniform(shape, low: -1.0f, high: 1.0f) * bound;
    }
}
