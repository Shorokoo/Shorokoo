using System;
using System.Linq;
using Shorokoo.Core.Rng;
using Shorokoo.Runtime;

namespace Shorokoo.Tests;

/// <summary>Emits the in-graph runtime uniform draw at the input's shape (fixed key/drawBase).</summary>
[Module]
public partial class RtUniformDraw
{
    public static Tensor<float32> Inline(Tensor<float32> x)
        => RuntimeRng.StandardUniform(x.ShapeTensor(), Scalar(123L), Scalar(456L), Scalar(0L));
}

/// <summary>Emits the in-graph runtime normal draw at the input's shape (fixed key/drawBase).</summary>
[Module]
public partial class RtNormalDraw
{
    public static Tensor<float32> Inline(Tensor<float32> x)
        => RuntimeRng.StandardNormal(x.ShapeTensor(), Scalar(7L), Scalar(9L), Scalar(0L));
}

/// <summary>
/// Coverage for the in-graph counter-based runtime RNG (<see cref="RuntimeRng"/>): the ONNX-op
/// Threefry subgraph must reproduce the host generator (<see cref="Threefry2x32"/>) bit-for-bit
/// — proving the novel integer-op PRNG is correct and execution-provider-independent — and
/// produce well-distributed draws.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class RngRuntimeTests
{
    private static float[] RunDraw<TModule>(long rows, long cols)
    {
        var g = (FastComputationGraph)typeof(TModule)
            .GetProperty("ComputationGraph")!.GetValue(null)!;
        var input = TensorData([rows, cols], Enumerable.Repeat(0f, (int)(rows * cols)).ToArray());
        var concrete = g.ToConcreteArchitecture(g.FromOrderedInputs([input])).ToConcreteModel();
        var outputs = ComputeContext.Default.Execute(concrete, input);
        return outputs[0].ToTensorData().As<float32>().AccessMemory().ToArray();
    }

    // Host reference for the runtime scheme: element i -> counter (i, drawBase);
    // uniform = low 24 bits of x0 * 2^-24.
    private static float HostUniform(long i, uint k0, uint k1, uint drawBase)
    {
        var (x0, _) = Threefry2x32.Bijection((uint)i, drawBase, k0, k1);
        return (x0 & 0x00FFFFFFu) * (1.0f / 16777216.0f);
    }

    [Fact]
    public void TestInGraphUniformMatchesHostBitExact()
    {
        var vals = RunDraw<RtUniformDraw>(4, 4);
        Assert.Equal(16, vals.Length);
        for (long i = 0; i < 16; i++)
            Assert.Equal(HostUniform(i, 123, 456, 0), vals[i]);
    }

    [Fact]
    public void TestInGraphUniformIsInRangeAndSpread()
    {
        var vals = RunDraw<RtUniformDraw>(8, 8);
        Assert.All(vals, v => Assert.InRange(v, 0.0f, 0.99999997f));
        Assert.InRange(vals.Average(), 0.4f, 0.6f);
    }

    [Fact]
    public void TestInGraphNormalHasStandardMoments()
    {
        var vals = RunDraw<RtNormalDraw>(40, 40);
        double mean = vals.Average();
        double variance = vals.Select(v => (v - mean) * (v - mean)).Average();
        Assert.InRange(mean, -0.1, 0.1);
        Assert.InRange(variance, 0.85, 1.15);
    }
}
