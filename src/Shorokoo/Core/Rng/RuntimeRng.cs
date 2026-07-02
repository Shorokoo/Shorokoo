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
/// <c>(key, drawBase, i)</c> replays exactly. The bit→float and Box–Muller conventions match
/// <see cref="HostRng"/> (low 24 bits × 2⁻²⁴; radius = √(−2·ln(1−u₁))).</para>
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

    /// <summary>Threefry-2x32 (20 rounds) over the per-element counter <c>(c0, drawBase)</c>.
    /// <paramref name="c0"/> is the flat element-index tensor <c>[N]</c>; the other words are scalars.</summary>
    public static (Tensor<int64> x0, Tensor<int64> x1) Bijection(
        Tensor<int64> c0, Scalar<int64> drawBase, Scalar<int64> k0, Scalar<int64> k1)
    {
        // Key schedule (host-foldable: keys are scalars). ks2 = parity ^ k0 ^ k1.
        Scalar<int64> ks0 = k0, ks1 = k1, ks2 = Scalar(SkeinParity) ^ k0 ^ k1;

        var x0 = Mask32(c0 + ks0);                       // [N]
        var x1 = Mask32((c0 - c0) + drawBase + ks1);     // broadcast drawBase to [N]

        for (int r = 0; r < 20; r++)
        {
            x0 = Mask32(x0 + x1);
            x1 = RotL(x1, Rot[r & 7]);
            x1 = OnnxOp.BitwiseXor(x1, x0).int64();

            if ((r & 3) == 3)
            {
                int inject = (r >> 2) + 1;
                Scalar<int64> kA = KeyWord(ks0, ks1, ks2, inject % 3);
                Scalar<int64> kB = KeyWord(ks0, ks1, ks2, (inject + 1) % 3);
                x0 = Mask32(x0 + kA);
                x1 = Mask32(x1 + kB + Scalar((long)inject));
            }
        }
        return (x0, x1);
    }

    private static Scalar<int64> KeyWord(Scalar<int64> ks0, Scalar<int64> ks1, Scalar<int64> ks2, int i)
        => i == 0 ? ks0 : i == 1 ? ks1 : ks2;

    /// <summary>A [0,1) uniform from a 32-bit word: low 24 bits × 2⁻²⁴.</summary>
    private static Tensor<float32> ToUniform(Tensor<int64> word)
        => OnnxOp.Mod(word, Scalar(0x0100_0000L)).int64().Cast<float32>() * Scalar(TwoPow24Inv);

    /// <summary>The per-element flat index counter <c>[prod(shape)]</c> as int64.</summary>
    private static Tensor<int64> Counter(Vector<int64> shape)
    {
        Scalar<int64> n = shape.Reduce(ReduceKind.Prod);
        return OnnxOp.Range(Scalar(0L), n, Scalar(1L)).int64();   // [N]
    }

    /// <summary>Standard uniform U(0,1) of the given shape.</summary>
    public static Tensor<float32> StandardUniform(
        Vector<int64> shape, Scalar<int64> k0, Scalar<int64> k1, Scalar<int64> drawBase)
    {
        var (x0, _) = Bijection(Counter(shape), drawBase, k0, k1);
        return ToUniform(x0).Reshape(shape);
    }

    /// <summary>Standard normal N(0,1) of the given shape (per-element Box–Muller).</summary>
    public static Tensor<float32> StandardNormal(
        Vector<int64> shape, Scalar<int64> k0, Scalar<int64> k1, Scalar<int64> drawBase)
    {
        var (x0, x1) = Bijection(Counter(shape), drawBase, k0, k1);
        var u1 = ToUniform(x0);
        var u2 = ToUniform(x1);
        var radius = ((-u1 + Scalar(1.0f)).Ln() * Scalar(-2.0f)).Sqrt();   // √(−2·ln(1−u₁))
        var theta = u2 * Scalar(2.0f * System.MathF.PI);
        return (radius * theta.Cos()).Reshape(shape);
    }

    /// <summary>U(low, high) of the given shape.</summary>
    public static Tensor<float32> Uniform(
        Vector<int64> shape, Scalar<int64> k0, Scalar<int64> k1, Scalar<int64> drawBase,
        Scalar<float32> low, Scalar<float32> high)
        => StandardUniform(shape, k0, k1, drawBase) * (high - low) + low;

    /// <summary>N(mean, scale) of the given shape.</summary>
    public static Tensor<float32> Normal(
        Vector<int64> shape, Scalar<int64> k0, Scalar<int64> k1, Scalar<int64> drawBase,
        Scalar<float32> mean, Scalar<float32> scale)
        => StandardNormal(shape, k0, k1, drawBase) * scale + mean;
}
