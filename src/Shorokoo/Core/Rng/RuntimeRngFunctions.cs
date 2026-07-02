using System;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Core.Nodes.OnnxNodes;
using static Shorokoo.Globals;

namespace Shorokoo.Core.Rng;

/// <summary>
/// Captures <see cref="RuntimeRng"/>'s uniform / normal draws as reusable <see cref="Function"/>s
/// so the random-op lowering (<see cref="Shorokoo.Core.Nodes.Processors.Fast.FastLowerRandomOps"/>)
/// can splice an in-graph counter-based draw at each runtime <c>SHRK_RANDOM_*</c> site via
/// <c>FUNCTION_INVOKE</c> + the standard function inliner. Each function takes
/// <c>(shape, k0, k1, drawBase, a, b)</c> — the draw shape, the 32-bit Threefry key words, the
/// per-execution counter high word, and the distribution parameters (low/high or mean/scale).
/// </summary>
internal static class RuntimeRngFunctions
{
    private static Function? _uniform;
    private static Function? _normal;

    public static Function Uniform => _uniform ??= Build(
        (Func<Tensor<int64>, Scalar<int64>, Scalar<int64>, Scalar<int64>, Scalar<float32>, Scalar<float32>, Tensor<float32>>)UniformDraw,
        "RuntimeRngUniform");

    public static Function Normal => _normal ??= Build(
        (Func<Tensor<int64>, Scalar<int64>, Scalar<int64>, Scalar<int64>, Scalar<float32>, Scalar<float32>, Tensor<float32>>)NormalDraw,
        "RuntimeRngNormal");

    private static Tensor<float32> UniformDraw(
        Tensor<int64> shape, Scalar<int64> k0, Scalar<int64> k1, Scalar<int64> drawBase,
        Scalar<float32> low, Scalar<float32> high)
        => RuntimeRng.Uniform(shape.Vec(), k0, k1, drawBase, low, high);

    private static Tensor<float32> NormalDraw(
        Tensor<int64> shape, Scalar<int64> k0, Scalar<int64> k1, Scalar<int64> drawBase,
        Scalar<float32> mean, Scalar<float32> scale)
        => RuntimeRng.Normal(shape.Vec(), k0, k1, drawBase, mean, scale);

    private static Function Build(Delegate body, string name)
    {
        var graph = GraphBuilder.BuildFastComputationGraphFromDelegate(body);
        return new Function(graph, FunctionType.Function, name, name);
    }
}
