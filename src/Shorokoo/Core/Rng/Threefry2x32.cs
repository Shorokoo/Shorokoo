namespace Shorokoo.Core.Rng;

/// <summary>
/// Threefry-2x32, the counter-based PRNG (Salmon, Moraes, Dror &amp; Shaw,
/// "Parallel Random Numbers: As Easy as 1, 2, 3", SC'11 — the Random123 family)
/// used as Shorokoo's default bit generator.
///
/// <para>A counter-based PRNG is a keyed bijection <c>bits = f(key, counter)</c>: it
/// has no mutable stream state, so any (key, counter) pair is O(1) random-access into
/// a stream of period 2^64, and distinct keys give independent streams. That is what
/// lets Shorokoo derive a per-parameter / per-site stream from a name and index a draw
/// by (step, element) without threading generator state — and lets the identical
/// integer math run host-side (initialization) or as an ONNX subgraph (runtime) with
/// bit-for-bit agreement.</para>
///
/// <para>This is the 20-round variant (the Random123 safety-margin default; 13 rounds
/// is the minimum Crush-resistant form). Validated against the Random123 known-answer
/// test vectors — see <c>RngCoreTests</c>.</para>
/// </summary>
internal static class Threefry2x32
{
    /// <summary>Number of Feistel rounds. 20 is the Random123 safety-margin default.</summary>
    public const int Rounds = 20;

    /// <summary>The minimum Crush-resistant round count (Random123 <c>threefry2x32x13</c>):
    /// still passes BigCrush, ~35% fewer rounds than the default. A faster, lower-margin variant.</summary>
    public const int Rounds13 = 13;

    // Skein key-schedule parity constant for 32-bit words.
    private const uint SkeinParity = 0x1BD11BDAu;

    // Rotation constants for Threefry-2x32, cycled every 8 rounds (Random123 threefry.h).
    private static readonly int[] Rot = [13, 15, 26, 6, 17, 29, 16, 24];

    /// <summary>
    /// Applies the keyed bijection to a 64-bit counter <c>(c0, c1)</c> under key
    /// <c>(k0, k1)</c>, returning 64 pseudo-random bits as two 32-bit words.
    /// </summary>
    public static (uint x0, uint x1) Bijection(uint c0, uint c1, uint k0, uint k1)
        => Bijection(c0, c1, k0, k1, Rounds);

    /// <summary>
    /// The keyed bijection with an explicit <paramref name="rounds"/> count (Random123
    /// Threefry-2x32-R). The per-4-rounds key injection keys off the round index, so any R
    /// is the correct Random123 variant — R=13 injects after rounds 4/8/12 with no trailing
    /// injection, R=20 injects after 4/8/12/16/20.
    /// </summary>
    public static (uint x0, uint x1) Bijection(uint c0, uint c1, uint k0, uint k1, int rounds)
    {
        // Key schedule: ks[2] = parity ^ k0 ^ k1.
        uint ks0 = k0, ks1 = k1, ks2 = SkeinParity ^ k0 ^ k1;

        uint x0 = c0 + ks0;
        uint x1 = c1 + ks1;

        for (int r = 0; r < rounds; r++)
        {
            x0 += x1;
            int s = Rot[r & 7];
            x1 = (x1 << s) | (x1 >> (32 - s));
            x1 ^= x0;

            // Inject the key schedule after every 4th round.
            if ((r & 3) == 3)
            {
                int inject = (r >> 2) + 1;           // 1, 2, 3, ...
                uint kA = KeyWord(ks0, ks1, ks2, inject % 3);
                uint kB = KeyWord(ks0, ks1, ks2, (inject + 1) % 3);
                x0 += kA;
                x1 += kB + (uint)inject;
            }
        }

        return (x0, x1);
    }

    private static uint KeyWord(uint ks0, uint ks1, uint ks2, int i)
        => i == 0 ? ks0 : i == 1 ? ks1 : ks2;
}
