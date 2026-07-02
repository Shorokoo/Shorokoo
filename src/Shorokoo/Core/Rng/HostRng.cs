using System;

namespace Shorokoo.Core.Rng;

/// <summary>
/// Host-side draw engine over <see cref="Threefry2x32"/>: turns a stream key
/// <c>(k0, k1)</c> plus a counter base into arrays of standard uniform / normal
/// floats. Used for trainable-parameter initialization (a one-time, host-mediated
/// step). The bit→float and normal conventions here are chosen to match the
/// in-graph runtime lowering exactly, so a value drawn host-side and one drawn as an
/// ONNX subgraph agree for the same (key, counter).
///
/// <para>A single "draw index" <c>d</c> addresses the stream: each Threefry block
/// yields two 32-bit words, so <c>d</c> maps to block <c>d/2</c>, word <c>d%2</c> —
/// giving O(1) random access.</para>
///
/// <para>Uniform: low 24 bits × 2⁻²⁴ ∈ [0, 1). Normal: Box–Muller over uniform
/// pairs (radius = √(−2·ln(1−u₁)) so the log argument is never 0).</para>
/// </summary>
internal sealed class HostRng
{
    private const float TwoPow24Inv = 1.0f / 16777216.0f; // 2^-24

    private readonly uint _k0;
    private readonly uint _k1;
    private readonly ulong _counterBase;

    public HostRng(uint k0, uint k1, ulong counterBase = 0)
    {
        _k0 = k0;
        _k1 = k1;
        _counterBase = counterBase;
    }

    /// <summary>The uniform draw in [0, 1) at draw index <paramref name="d"/>.</summary>
    private float UniformAt(ulong d)
    {
        ulong block = _counterBase + (d >> 1);
        var (w0, w1) = Threefry2x32.Bijection(
            (uint)(block & 0xFFFFFFFF), (uint)(block >> 32), _k0, _k1);
        uint bits = (d & 1) == 0 ? w0 : w1;
        return (bits & 0x00FFFFFFu) * TwoPow24Inv;
    }

    /// <summary><paramref name="count"/> i.i.d. draws from U(0, 1).</summary>
    public float[] StandardUniform(long count)
    {
        var result = new float[count];
        for (long i = 0; i < count; i++)
            result[i] = UniformAt((ulong)i);
        return result;
    }

    /// <summary><paramref name="count"/> i.i.d. draws from N(0, 1) (Box–Muller).</summary>
    public float[] StandardNormal(long count)
    {
        var result = new float[count];
        long pairs = (count + 1) / 2;
        for (long p = 0; p < pairs; p++)
        {
            float u1 = UniformAt((ulong)(2 * p));
            float u2 = UniformAt((ulong)(2 * p + 1));
            float radius = MathF.Sqrt(-2.0f * MathF.Log(1.0f - u1)); // 1-u1 ∈ (0,1] ⇒ no log(0)
            float theta = 2.0f * MathF.PI * u2;
            long j = 2 * p;
            result[j] = radius * MathF.Cos(theta);
            if (j + 1 < count)
                result[j + 1] = radius * MathF.Sin(theta);
        }
        return result;
    }

    /// <summary><paramref name="count"/> i.i.d. draws from U(low, high).</summary>
    public float[] Uniform(long count, float low, float high)
    {
        var result = StandardUniform(count);
        float span = high - low;
        for (long i = 0; i < count; i++)
            result[i] = low + result[i] * span;
        return result;
    }

    /// <summary><paramref name="count"/> i.i.d. draws from N(mean, std).</summary>
    public float[] Normal(long count, float mean, float std)
    {
        var result = StandardNormal(count);
        for (long i = 0; i < count; i++)
            result[i] = mean + result[i] * std;
        return result;
    }
}
