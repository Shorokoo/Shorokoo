using Shorokoo;
using Shorokoo.Core.Nodes.NodeDefinitions;
using static Shorokoo.Globals;

namespace Shorokoo.Core.Rng;

/// <summary>
/// In-graph counter-based RNG: builds an ONNX-op subgraph computing Threefry-2x32 over a
/// per-element counter, entirely from ordinary integer/float graph math. Because it uses no
/// ONNX random op, the result is deterministic and identical across execution providers and
/// the Quick Execution Engine, and an exported model's randomness is self-contained — unlike
/// ONNX's <c>RandomUniformLike</c>, whose value depends on the runtime, EP, platform, and
/// session lifetime.
///
/// <para>The 32-bit Threefry words are carried in <c>int64</c> tensors kept in the range
/// <c>[0, 2^32)</c> by an explicit <c>Mod 2^32</c> after every add/rotate. Shifts are done
/// arithmetically (<c>x&lt;&lt;s == (x*2^s) mod 2^32</c>, <c>x&gt;&gt;s == x/2^(bits)</c>) rather than with
/// ONNX <c>BitShift</c> (which is defined for unsigned types only); the only genuinely bitwise
/// op is <c>BitwiseXor</c> (the Feistel mix), which ONNX defines for signed integers. This
/// mirrors <see cref="Threefry2x32"/> bit-for-bit (validated against the Random123 known-answer
/// vectors — see <c>RngRuntimeTests</c>).</para>
///
/// <para>Per element <c>i</c> the counter is <c>(i, drawBase)</c>: <c>i</c> (the flat element
/// index) is the low counter word, <c>drawBase</c> (a per-execution value, e.g. the training
/// step) the high word, so successive executions draw fresh values while any fixed
/// <c>(key, drawBase, i)</c> replays exactly. Bit→float is the low 24 bits × 2⁻²⁴;
/// the normal transform is Box–Muller with radius = √(−2·ln(1−u₁)).</para>
/// </summary>
internal static class RuntimeRng
{
    private static readonly int[] Rot = [13, 15, 26, 6, 17, 29, 16, 24];
    private const long Pow2_32 = 0x1_0000_0000L;
    private const long SkeinParity = 0x1BD11BDAL;
    private const float TwoPow24Inv = 1.0f / 16777216.0f;

    /// <summary>Wraps a non-negative int64 tensor back into <c>[0, 2^32)</c> (== <c>&amp; 0xFFFFFFFF</c>).</summary>
    private static Tensor<int64> Mask32(Tensor<int64> x) => OnnxOp.Mod(x, Scalar(Pow2_32)).int64();

    /// <summary>32-bit left rotate by <paramref name="s"/> via arithmetic shift + recombine.
    /// The high part <c>(x*2^s) mod 2^32</c> occupies bits [s,32) and the low part <c>x/2^(32-s)</c>
    /// occupies bits [0,s); being disjoint, their sum equals their bitwise-or.</summary>
    private static Tensor<int64> RotL(Tensor<int64> x, int s)
    {
        var hi = Mask32(x * Scalar(1L << s));
        var lo = x / Scalar(1L << (32 - s));
        return hi + lo;
    }

    /// <summary>Threefry-2x32 over the per-element counter <c>(c0, drawBase)</c> with an explicit
    /// <paramref name="rounds"/> count (default <see cref="Threefry2x32.Rounds"/>;
    /// <see cref="Threefry2x32.Rounds13"/> is the Crush-resistant fast variant).
    /// <paramref name="c0"/> is the flat element-index tensor <c>[N]</c>; the other words are scalars.
    /// Bit-for-bit identical to <see cref="Threefry2x32.Bijection(uint, uint, uint, uint, int)"/>.</summary>
    public static (Tensor<int64> x0, Tensor<int64> x1) Bijection(
        Tensor<int64> c0, Scalar<int64> drawBase, Scalar<int64> k0, Scalar<int64> k1, int rounds = Threefry2x32.Rounds)
        => BijectionCore(c0, drawBase, k0, k1, rounds);

    /// <summary>
    /// The bijection with <b>per-element keys</b>: <paramref name="k0"/>/<paramref name="k1"/> are
    /// tensors broadcast-compatible with the counter, so N independent (key, counter) pairs are
    /// transformed in ONE pass. The math is elementwise, so this is the same computation the
    /// scalar-key overload performs — that one simply broadcasts a single key over the counter.
    /// Used to fold a whole key-tree level at once (see <see cref="BatchSplitKeys"/>).
    /// </summary>
    private static (Tensor<int64> x0, Tensor<int64> x1) BijectionCore(
        Tensor<int64> c0, Tensor<int64> drawBase, Tensor<int64> k0, Tensor<int64> k1, int rounds)
    {
        // Key schedule. ks2 = parity ^ k0 ^ k1.
        Tensor<int64> ks0 = k0, ks1 = k1, ks2 = OnnxOp.BitwiseXor(OnnxOp.BitwiseXor(Scalar(SkeinParity), k0).int64(), k1).int64();

        var x0 = Mask32(c0 + ks0);                       // [N]
        var x1 = Mask32((c0 - c0) + drawBase + ks1);     // broadcast drawBase to [N]

        for (int r = 0; r < rounds; r++)
        {
            x0 = Mask32(x0 + x1);
            x1 = RotL(x1, Rot[r & 7]);
            x1 = OnnxOp.BitwiseXor(x1, x0).int64();

            if ((r & 3) == 3)
            {
                int inject = (r >> 2) + 1;
                Tensor<int64> kA = KeyWord(ks0, ks1, ks2, inject % 3);
                Tensor<int64> kB = KeyWord(ks0, ks1, ks2, (inject + 1) % 3);
                x0 = Mask32(x0 + kA);
                x1 = Mask32(x1 + kB + Scalar((long)inject));
            }
        }
        return (x0, x1);
    }

    private static Tensor<int64> KeyWord(Tensor<int64> ks0, Tensor<int64> ks1, Tensor<int64> ks2, int i)
        => i == 0 ? ks0 : i == 1 ? ks1 : ks2;

    /// <summary>
    /// Index-based key split: child key words = Bijection(counter: (index, 0), key).
    /// Random access — computing child <paramref name="index"/> never computes any sibling.
    /// The counter word is the index's LOW 32 BITS (<c>Mask32</c>, matching the test oracle's
    /// <c>uint</c> cast), so indices <c>i</c> and
    /// <c>i + 2^32</c> alias to the same child key. Unreachable in practice: split indices
    /// are ModelId slots and iteration indices, which are <c>int</c>-typed.
    /// </summary>
    public static (Scalar<int64> k0, Scalar<int64> k1) SplitKey(
        Scalar<int64> k0, Scalar<int64> k1, Scalar<int64> index)
    {
        Vector<int64> ctr = [index];
        var (x0, x1) = Bijection(Mask32(ctr), Scalar(0L), k0, k1);
        return (x0.Vec()[0], x1.Vec()[0]);
    }

    /// <summary>
    /// Splits a whole key-tree <b>level</b> in one pass: <paramref name="keys"/> is a
    /// <c>[2, M]</c> block (row 0 = the k0 words, row 1 = the k1 words) of M independent parent
    /// keys, <paramref name="indices"/> the M per-key split counters, and the result is the
    /// <c>[2, M]</c> block of child keys in the same layout — so levels chain directly, output
    /// into input.
    ///
    /// <para>Element <c>i</c> computes exactly <see cref="SplitKey"/> of <c>(keys[.., i],
    /// indices[i])</c>: the same <see cref="BijectionCore"/> over the same counter
    /// <c>(Mask32(index), 0)</c>, differing only in that the key words arrive as tensors rather
    /// than broadcast scalars. Folding M streams costs ONE bijection instead of M.</para>
    /// </summary>
    public static Tensor<int64> BatchSplitKeys(Tensor<int64> keys, Vector<int64> indices)
    {
        var k0s = OnnxOp.Gather(keys, Scalar(0L), 0L).int64();   // [M]
        var k1s = OnnxOp.Gather(keys, Scalar(1L), 0L).int64();   // [M]
        var (x0, x1) = BijectionCore(Mask32(indices), Scalar(0L), k0s, k1s, Threefry2x32.Rounds);
        Vector<int64> row = [Scalar(1L), Scalar(-1L)];
        return OnnxOp.Concat([x0.Reshape(row), x1.Reshape(row)], 0L).int64();   // [2, M]
    }

    /// <summary>A [0,1) uniform from a 32-bit word: low 24 bits × 2⁻²⁴.</summary>
    private static Tensor<float32> ToUniform(Tensor<int64> word)
        => OnnxOp.Mod(word, Scalar(0x0100_0000L)).int64().Cast<float32>() * Scalar(TwoPow24Inv);

    /// <summary>The per-element flat index counter <c>[prod(shape)]</c> as int64.</summary>
    private static Tensor<int64> Counter(Vector<int64> shape)
    {
        Scalar<int64> n = shape.Reduce(ReduceKind.Prod);
        return OnnxOp.Range(Scalar(0L), n, Scalar(1L)).int64();   // [N]
    }

    /// <summary>Standard uniform U(0,1) of the given shape (bit generator: Threefry-2x32-<paramref name="rounds"/>).</summary>
    public static Tensor<float32> StandardUniform(
        Vector<int64> shape, Scalar<int64> k0, Scalar<int64> k1, Scalar<int64> drawBase, int rounds = Threefry2x32.Rounds)
    {
        var (x0, _) = Bijection(Counter(shape), drawBase, k0, k1, rounds);
        return ToUniform(x0).Reshape(shape);
    }

    /// <summary>Standard normal N(0,1) of the given shape (per-element Box–Muller over Threefry-2x32-<paramref name="rounds"/>).</summary>
    public static Tensor<float32> StandardNormal(
        Vector<int64> shape, Scalar<int64> k0, Scalar<int64> k1, Scalar<int64> drawBase, int rounds = Threefry2x32.Rounds)
    {
        var (x0, x1) = Bijection(Counter(shape), drawBase, k0, k1, rounds);
        var u1 = ToUniform(x0);
        var u2 = ToUniform(x1);
        var radius = ((-u1 + Scalar(1.0f)).Ln() * Scalar(-2.0f)).Sqrt();   // √(−2·ln(1−u₁))
        var theta = u2 * Scalar(2.0f * System.MathF.PI);
        return (radius * theta.Cos()).Reshape(shape);
    }

    /// <summary>U(low, high) of the given shape.</summary>
    public static Tensor<float32> Uniform(
        Vector<int64> shape, Scalar<int64> k0, Scalar<int64> k1, Scalar<int64> drawBase,
        Scalar<float32> low, Scalar<float32> high, int rounds = Threefry2x32.Rounds)
        => StandardUniform(shape, k0, k1, drawBase, rounds) * (high - low) + low;

    /// <summary>N(mean, scale) of the given shape.</summary>
    public static Tensor<float32> Normal(
        Vector<int64> shape, Scalar<int64> k0, Scalar<int64> k1, Scalar<int64> drawBase,
        Scalar<float32> mean, Scalar<float32> scale, int rounds = Threefry2x32.Rounds)
        => StandardNormal(shape, k0, k1, drawBase, rounds) * scale + mean;

    // ── Raw random bits ─────────────────────────────────────────────────────────────────
    // One generator draw per element (counter (i, drawBase), like uniform/normal). The
    // generator's low word x0 is a uniformly-random 32-bit value; the narrow widths take its
    // low W bits (all bits are equidistributed), U32 takes it whole, and U64 concatenates both
    // words x0 | (x1 << 32). U64 is computed in uint64 because x0 | (x1<<32) exceeds the
    // non-negative int64 range the rest of the generator stays within.

    /// <summary>Raw uniform bits, U8 (the low 8 bits of the generator word), of the given shape.</summary>
    public static Tensor<uint8> BitsU8(
        Vector<int64> shape, Scalar<int64> k0, Scalar<int64> k1, Scalar<int64> drawBase, int rounds = Threefry2x32.Rounds)
    {
        var (x0, _) = Bijection(Counter(shape), drawBase, k0, k1, rounds);
        return OnnxOp.Mod(x0, Scalar(0x100L)).int64().Cast<uint8>().Reshape(shape);
    }

    /// <summary>Raw uniform bits, U16 (the low 16 bits of the generator word), of the given shape.</summary>
    public static Tensor<uint16> BitsU16(
        Vector<int64> shape, Scalar<int64> k0, Scalar<int64> k1, Scalar<int64> drawBase, int rounds = Threefry2x32.Rounds)
    {
        var (x0, _) = Bijection(Counter(shape), drawBase, k0, k1, rounds);
        return OnnxOp.Mod(x0, Scalar(0x1_0000L)).int64().Cast<uint16>().Reshape(shape);
    }

    /// <summary>Raw uniform bits, U32 (the whole generator word, already in [0, 2^32)), of the given shape.</summary>
    public static Tensor<uint32> BitsU32(
        Vector<int64> shape, Scalar<int64> k0, Scalar<int64> k1, Scalar<int64> drawBase, int rounds = Threefry2x32.Rounds)
    {
        var (x0, _) = Bijection(Counter(shape), drawBase, k0, k1, rounds);
        return x0.Cast<uint32>().Reshape(shape);
    }

    /// <summary>Raw uniform bits, U64 (both generator words, x0 | (x1 &lt;&lt; 32)), of the given shape.</summary>
    public static Tensor<uint64> BitsU64(
        Vector<int64> shape, Scalar<int64> k0, Scalar<int64> k1, Scalar<int64> drawBase, int rounds = Threefry2x32.Rounds)
    {
        var (x0, x1) = Bijection(Counter(shape), drawBase, k0, k1, rounds);
        var lo = x0.Cast<uint64>();
        var hi = x1.Cast<uint64>();
        var shifted = OnnxOp.BitShift(hi, Scalar(32UL), BitShiftDirection.Left);
        return OnnxOp.BitwiseOr(lo, shifted).uint64().Reshape(shape);
    }
}
