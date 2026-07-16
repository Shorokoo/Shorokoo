using System;

namespace Shorokoo.Core.Rng;

/// <summary>
/// Host-side draw engine over <see cref="Threefry2x32"/>: turns a stream key
/// <c>(k0, k1)</c> plus a counter base into arrays of standard uniform / normal
/// floats. Used for trainable-parameter initialization (a one-time, host-mediated
/// step). The underlying Threefry bijection and the bit→float / Box–Muller
/// conventions are bit-identical to the in-graph <see cref="RuntimeRng"/>.
///
/// <para>The two do NOT realize the same value sequence for a given key, however:
/// this host engine packs two draws per Threefry block (both output words, via the
/// draw index below), whereas the in-graph runtime path uses one word per element
/// with counter <c>(elementIndex, drawBase)</c>. That is by design and harmless —
/// initialization (host) and runtime feeds (in-graph) are separate streams keyed
/// from different sub-masters and are never compared. The one host↔graph value
/// agreement that IS relied on — index-based key splitting — matches exactly
/// (<c>HostFold</c> == <c>SHRK_RNG_SPLIT</c>), since both take both words of
/// <c>Bijection(counter: (index, 0), key)</c>.</para>
///
/// <para>A single "draw index" <c>d</c> addresses the stream: each Threefry block
/// yields two 32-bit words, so <c>d</c> maps to block <c>d/2</c>, word <c>d%2</c> —
/// giving O(1) random access. The fill loops run one bijection per block and consume
/// both words.</para>
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
    private readonly int _rounds;

    public HostRng(uint k0, uint k1, ulong counterBase = 0, int rounds = Threefry2x32.Rounds)
    {
        _k0 = k0;
        _k1 = k1;
        _counterBase = counterBase;
        _rounds = rounds;
    }

    /// <summary>The uniform pair in [0, 1)² at draw indices <c>(2·pair, 2·pair + 1)</c>:
    /// one bijection over block <paramref name="pair"/>, both output words consumed.</summary>
    private (float Even, float Odd) UniformPairAt(ulong pair)
    {
        ulong block = _counterBase + pair;
        var (w0, w1) = Threefry2x32.Bijection(
            (uint)(block & 0xFFFFFFFF), (uint)(block >> 32), _k0, _k1, _rounds);
        return ((w0 & 0x00FFFFFFu) * TwoPow24Inv, (w1 & 0x00FFFFFFu) * TwoPow24Inv);
    }

    /// <summary><paramref name="count"/> i.i.d. draws from U(0, 1).</summary>
    public float[] StandardUniform(long count)
    {
        var result = new float[count];
        long pairs = (count + 1) / 2;
        for (long p = 0; p < pairs; p++)
        {
            var (even, odd) = UniformPairAt((ulong)p);
            long j = 2 * p;
            result[j] = even;
            if (j + 1 < count)
                result[j + 1] = odd;
        }
        return result;
    }

    /// <summary><paramref name="count"/> i.i.d. draws from N(0, 1) (Box–Muller).</summary>
    public float[] StandardNormal(long count)
    {
        var result = new float[count];
        long pairs = (count + 1) / 2;
        for (long p = 0; p < pairs; p++)
        {
            var (u1, u2) = UniformPairAt((ulong)p);
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
