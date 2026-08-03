using Shorokoo;
using Shorokoo.Core.Nodes.NodeDefinitions;
using static Shorokoo.Globals;

namespace Shorokoo.Core.Rng;

/// <summary>
/// In-graph counter-based RNG: builds an ONNX-op subgraph computing Threefry-2x32 over a
/// per-element counter, entirely from ordinary integer graph math. Because it uses no ONNX
/// random op, the result is deterministic and identical across execution providers and the
/// Quick Execution Engine, and an exported model's randomness is self-contained — unlike ONNX's
/// <c>RandomUniformLike</c>, whose value depends on the runtime, EP, platform, and session
/// lifetime.
///
/// <para><b>Types carry the contract.</b> A key, a split index and a draw position are whole
/// <c>uint64</c> values; Threefry's 32-bit lanes are <c>uint32</c> tensors. Nothing packs 32 bits
/// of data into a wider type, and nothing silently narrows a wider value to fit: the split
/// between words happens once, explicitly, at the <see cref="Words"/>/<see cref="Pack"/>
/// boundary, and is an implementation detail of this file. Working in real <c>uint32</c> also
/// means modular wraparound is the type's own semantics rather than an explicit mask after every
/// operation, and a rotate is a genuine pair of shifts.</para>
///
/// <para>A draw is keyed by <c>(key, drawBase)</c> and indexed by the flat element index
/// <c>i</c>: <c>drawBase</c> (a per-execution value, e.g. the training step) folds into the key
/// and <c>i</c> occupies the whole counter, so successive executions draw fresh values while any
/// fixed <c>(key, drawBase, i)</c> replays exactly. Bit→float is the low 24 bits × 2⁻²⁴;
/// the normal transform is Box–Muller with radius = √(−2·ln(1−u₁)). Mirrors
/// <see cref="Threefry2x32"/> bit-for-bit (validated against the Random123 known-answer
/// vectors — see <c>RngRuntimeTests</c>).</para>
/// </summary>
internal static class RuntimeRng
{
    private static readonly int[] Rot = [13, 15, 26, 6, 17, 29, 16, 24];
    private const uint SkeinParity = 0x1BD11BDAu;
    private const float TwoPow24Inv = 1.0f / 16777216.0f;

    // ── uint64 <-> the two uint32 lanes Threefry works in ───────────────────────────────
    // The ONLY place the word split exists. Everything above this boundary speaks uint64.

    /// <summary>The low and high 32-bit words of a 64-bit value.</summary>
    private static (Tensor<uint32> lo, Tensor<uint32> hi) Words(Tensor<uint64> v)
        => (v.Cast<uint32>(),
            OnnxOp.BitShift(v, Scalar(32UL), BitShiftDirection.Right).uint64().Cast<uint32>());

    /// <summary>The 64-bit value with the given low and high 32-bit words.</summary>
    private static Tensor<uint64> Pack(Tensor<uint32> lo, Tensor<uint32> hi)
        => OnnxOp.BitwiseOr(
            lo.Cast<uint64>(),
            OnnxOp.BitShift(hi.Cast<uint64>(), Scalar(32UL), BitShiftDirection.Left).uint64()).uint64();

    /// <summary>32-bit left rotate — a real rotate, since the words are a real 32-bit type.</summary>
    private static Tensor<uint32> RotL(Tensor<uint32> x, int s)
        => OnnxOp.BitwiseOr(
            OnnxOp.BitShift(x, Scalar((uint)s), BitShiftDirection.Left).uint32(),
            OnnxOp.BitShift(x, Scalar((uint)(32 - s)), BitShiftDirection.Right).uint32()).uint32();

    /// <summary>
    /// Threefry-2x32 over the counter <c>(c0, c1)</c> under the key <c>(k0, k1)</c>, with an
    /// explicit <paramref name="rounds"/> count. All four are <c>uint32</c> tensors and are
    /// broadcast together, so N independent (key, counter) pairs transform in one pass — a
    /// single key simply broadcasts over a counter tensor. Wraparound is the type's semantics.
    /// Bit-for-bit identical to <see cref="Threefry2x32.Bijection(uint, uint, uint, uint, int)"/>.
    /// </summary>
    public static (Tensor<uint32> x0, Tensor<uint32> x1) Bijection(
        Tensor<uint32> c0, Tensor<uint32> c1, Tensor<uint32> k0, Tensor<uint32> k1,
        int rounds = Threefry2x32.Rounds)
    {
        Tensor<uint32> ks0 = k0, ks1 = k1,
            ks2 = OnnxOp.BitwiseXor(OnnxOp.BitwiseXor(Scalar(SkeinParity), k0).uint32(), k1).uint32();

        var x0 = c0 + ks0;
        var x1 = c1 + ks1;

        for (int r = 0; r < rounds; r++)
        {
            x0 = x0 + x1;
            x1 = RotL(x1, Rot[r & 7]);
            x1 = OnnxOp.BitwiseXor(x1, x0).uint32();

            if ((r & 3) == 3)
            {
                int inject = (r >> 2) + 1;
                Tensor<uint32> kA = KeyWord(ks0, ks1, ks2, inject % 3);
                Tensor<uint32> kB = KeyWord(ks0, ks1, ks2, (inject + 1) % 3);
                x0 = x0 + kA;
                x1 = x1 + kB + Scalar((uint)inject);
            }
        }
        return (x0, x1);
    }

    private static Tensor<uint32> KeyWord(Tensor<uint32> ks0, Tensor<uint32> ks1, Tensor<uint32> ks2, int i)
        => i == 0 ? ks0 : i == 1 ? ks1 : ks2;

    /// <summary>
    /// Index-based key split: <c>child = Bijection(counter: index, key)</c>. Random access —
    /// computing child <paramref name="index"/> never computes any sibling. The index is a whole
    /// <b>64-bit</b> value occupying both counter words, so distinct indices give distinct
    /// children over the entire range (no aliasing).
    /// </summary>
    public static Scalar<uint64> SplitKey(Scalar<uint64> key, Scalar<uint64> index)
        => BatchSplitKeys(key, index).Scalar();

    /// <summary>
    /// Splits a whole key-tree <b>level</b> in one pass: M parent keys and M split indices in,
    /// the M child keys out, elementwise. Folding M streams costs ONE bijection instead of M;
    /// element <c>i</c> computes exactly <see cref="SplitKey"/> of <c>(keys[i], indices[i])</c>.
    /// </summary>
    public static Tensor<uint64> BatchSplitKeys(Tensor<uint64> keys, Tensor<uint64> indices)
    {
        var (k0, k1) = Words(keys);
        var (c0, c1) = Words(indices);
        var (x0, x1) = Bijection(c0, c1, k0, k1);
        return Pack(x0, x1);
    }

    /// <summary>A [0,1) uniform from a 32-bit word: low 24 bits × 2⁻²⁴.</summary>
    private static Tensor<float32> ToUniform(Tensor<uint32> word)
        => OnnxOp.BitwiseAnd(word, Scalar(0x00FF_FFFFu)).uint32().Cast<float32>() * Scalar(TwoPow24Inv);

    /// <summary>The per-element flat element index <c>[prod(shape)]</c>.</summary>
    private static Tensor<uint64> ElementIndex(Vector<int64> shape)
    {
        Scalar<int64> n = shape.Reduce(ReduceKind.Prod);
        return OnnxOp.Range(Scalar(0L), n, Scalar(1L)).int64().Cast<uint64>();   // [N]
    }

    /// <summary>
    /// The generator words for a draw of the given shape under a whole 64-bit key.
    ///
    /// <para>The draw position is folded <b>into the key</b> — one bijection over scalars —
    /// rather than spending a counter word on it. That leaves BOTH counter words for the
    /// element index, so a draw position and an element index are each a whole 64-bit value
    /// and neither aliases: the 2³²'th execution draws a fresh stream rather than repeating
    /// the first, and a tensor of more than 2³² elements does not repeat within itself. The
    /// fold is the same primitive as a key split, so <c>drawBase = d</c> draws exactly the
    /// stream <c>split(key, d)</c> does at <c>drawBase = 0</c>.</para>
    /// </summary>
    private static (Tensor<uint32> x0, Tensor<uint32> x1) Draw(
        Vector<int64> shape, Scalar<uint64> key, Scalar<uint64> drawBase, int rounds)
    {
        var (k0, k1) = Words(key);
        var (d0, d1) = Words(drawBase);
        var (dk0, dk1) = Bijection(d0, d1, k0, k1, rounds);

        var (c0, c1) = Words(ElementIndex(shape));
        return Bijection(c0, c1, dk0, dk1, rounds);
    }

    /// <summary>Standard uniform U(0,1) of the given shape (bit generator: Threefry-2x32-<paramref name="rounds"/>).</summary>
    public static Tensor<float32> StandardUniform(
        Vector<int64> shape, Scalar<uint64> key, Scalar<uint64> drawBase, int rounds = Threefry2x32.Rounds)
    {
        var (x0, _) = Draw(shape, key, drawBase, rounds);
        return ToUniform(x0).Reshape(shape);
    }

    /// <summary>Standard normal N(0,1) of the given shape (per-element Box–Muller over Threefry-2x32-<paramref name="rounds"/>).</summary>
    public static Tensor<float32> StandardNormal(
        Vector<int64> shape, Scalar<uint64> key, Scalar<uint64> drawBase, int rounds = Threefry2x32.Rounds)
    {
        var (x0, x1) = Draw(shape, key, drawBase, rounds);
        var u1 = ToUniform(x0);
        var u2 = ToUniform(x1);
        var radius = ((-u1 + Scalar(1.0f)).Ln() * Scalar(-2.0f)).Sqrt();   // √(−2·ln(1−u₁))
        var theta = u2 * Scalar(2.0f * System.MathF.PI);
        return (radius * theta.Cos()).Reshape(shape);
    }

    /// <summary>U(low, high) of the given shape.</summary>
    public static Tensor<float32> Uniform(
        Vector<int64> shape, Scalar<uint64> key, Scalar<uint64> drawBase,
        Scalar<float32> low, Scalar<float32> high, int rounds = Threefry2x32.Rounds)
        => StandardUniform(shape, key, drawBase, rounds) * (high - low) + low;

    /// <summary>N(mean, scale) of the given shape.</summary>
    public static Tensor<float32> Normal(
        Vector<int64> shape, Scalar<uint64> key, Scalar<uint64> drawBase,
        Scalar<float32> mean, Scalar<float32> scale, int rounds = Threefry2x32.Rounds)
        => StandardNormal(shape, key, drawBase, rounds) * scale + mean;

    // ── Raw random bits ─────────────────────────────────────────────────────────────────
    // One generator draw per element. The generator's low word x0 is a uniformly-random 32-bit
    // value; the narrow widths take its low W bits (all bits are equidistributed), U32 takes it
    // whole, and U64 is both words packed.

    /// <summary>Raw uniform bits, U8 (the low 8 bits of the generator word), of the given shape.</summary>
    public static Tensor<uint8> BitsU8(
        Vector<int64> shape, Scalar<uint64> key, Scalar<uint64> drawBase, int rounds = Threefry2x32.Rounds)
    {
        var (x0, _) = Draw(shape, key, drawBase, rounds);
        return x0.Cast<uint8>().Reshape(shape);
    }

    /// <summary>Raw uniform bits, U16 (the low 16 bits of the generator word), of the given shape.</summary>
    public static Tensor<uint16> BitsU16(
        Vector<int64> shape, Scalar<uint64> key, Scalar<uint64> drawBase, int rounds = Threefry2x32.Rounds)
    {
        var (x0, _) = Draw(shape, key, drawBase, rounds);
        return x0.Cast<uint16>().Reshape(shape);
    }

    /// <summary>Raw uniform bits, U32 (the whole generator word), of the given shape.</summary>
    public static Tensor<uint32> BitsU32(
        Vector<int64> shape, Scalar<uint64> key, Scalar<uint64> drawBase, int rounds = Threefry2x32.Rounds)
    {
        var (x0, _) = Draw(shape, key, drawBase, rounds);
        return x0.Reshape(shape);
    }

    /// <summary>Raw uniform bits, U64 (both generator words), of the given shape.</summary>
    public static Tensor<uint64> BitsU64(
        Vector<int64> shape, Scalar<uint64> key, Scalar<uint64> drawBase, int rounds = Threefry2x32.Rounds)
    {
        var (x0, x1) = Draw(shape, key, drawBase, rounds);
        return Pack(x0, x1).Reshape(shape);
    }
}
