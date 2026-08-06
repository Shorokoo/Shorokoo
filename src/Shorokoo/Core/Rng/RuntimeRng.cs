using Shorokoo;
using Shorokoo.Core.Nodes.NodeDefinitions;
using static Shorokoo.Globals;

namespace Shorokoo.Core.Rng;

/// <summary>
/// In-graph counter-based RNG: builds an ONNX-op subgraph computing Threefry-2x32 over a
/// per-element counter, entirely from ordinary integer/float graph math — the bit generator and
/// the uniform transform are integer and exact, while the normal transform's Box-Muller step runs
/// float32 Ln/Sqrt/Cos kernels (which is why the normal goldens carry a tolerance and the uniform
/// ones do not). Because it uses no ONNX
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
/// <para>A draw is keyed by <c>(key, substreamIndex)</c> and indexed by the stream position
/// <c>p</c>, and <see cref="Draw"/> yields one whole <c>uint64</c> of generator output per
/// position. <c>substreamIndex</c> selects which substream of the consumer's key to draw from — the
/// execution counter for a runtime feed (so each execution draws fresh), the draw's ordinal within
/// an initializer when one initializer draws more than once. It folds into the key
/// and <c>p</c> occupies the whole counter, so successive executions draw fresh values while any
/// fixed <c>(key, substreamIndex, p)</c> replays exactly. Bit→float is the low 24 bits × 2⁻²⁴;
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

    /// <summary>A [0,1) uniform from the low 32-bit lane of a generator value: low 24 bits × 2⁻²⁴.</summary>
    private static Tensor<float32> ToUniform(Tensor<uint64> v)
        => OnnxOp.BitwiseAnd(v, Scalar(0x00FF_FFFFUL)).uint64().Cast<float32>() * Scalar(TwoPow24Inv);

    /// <summary>Shifts <paramref name="shift"/> bits of a generator value away, bringing the lane
    /// above them into the low bits. Broadcasts, so a vector of lane offsets extracts every lane
    /// of every value in one node.</summary>
    private static Tensor<uint64> ShiftDown(Tensor<uint64> v, Variable shift)
        => OnnxOp.BitShift(v, shift, BitShiftDirection.Right).uint64();

    /// <summary>The number of elements a draw of the given shape produces.</summary>
    private static Scalar<int64> ElementCount(Vector<int64> shape) => shape.Reduce(ReduceKind.Prod);

    /// <summary>The counter positions <c>[0, count)</c>.</summary>
    private static Tensor<uint64> Positions(Scalar<int64> count)
        => OnnxOp.Range(Scalar(0L), count, Scalar(1L)).int64().Cast<uint64>();

    /// <summary>
    /// The generator's output at the first <paramref name="positionCount"/> counter positions of
    /// the stream under a whole 64-bit key — one whole <c>uint64</c> per position.
    ///
    /// <para>The substream index is folded <b>into the key</b> — one bijection over scalars —
    /// rather than spending a counter word on it. That leaves BOTH counter words for the
    /// stream position, so a substream index and a position are each a whole 64-bit value
    /// and neither aliases <em>as counters</em>: the 2³²'th execution draws a fresh stream rather
    /// than repeating the first, and the generator's word pair stays distinct across more than 2³²
    /// positions. (Distinct generator words, not distinct floats — <see cref="ToUniform"/> keeps 24
    /// bits, so drawn values collide by pigeonhole long before that.)</para>
    ///
    /// <para>The fold reuses the bijection, but it is <b>not</b> the key tree's split: it runs at
    /// this algorithm's <paramref name="rounds"/>, whereas a key split is pinned to the default
    /// algorithm so switching generators never re-keys a stream. Nor is <c>substreamIndex = d</c> the
    /// same as drawing at <c>substreamIndex = 0</c> under <c>split(key, d)</c> — that would fold twice
    /// (<c>B(0, B(d, key))</c>), and <c>B(0, ·)</c> is not the identity. The draw simply runs
    /// under the folded key <c>B(d, key)</c>.</para>
    /// </summary>
    private static Vector<uint64> Draw(
        Scalar<int64> positionCount, Scalar<uint64> key, Scalar<uint64> substreamIndex, int rounds)
    {
        var (k0, k1) = Words(key);
        var (d0, d1) = Words(substreamIndex);
        var (dk0, dk1) = Bijection(d0, d1, k0, k1, rounds);

        var (c0, c1) = Words(Positions(positionCount));
        var (x0, x1) = Bijection(c0, c1, dk0, dk1, rounds);
        return Pack(x0, x1).Vec();
    }

    // ── Packing ─────────────────────────────────────────────────────────────────────────
    // A position costs one bijection and yields 64 bits, and the two words are inseparable —
    // Threefry's rounds feed x0 and x1 through each other, so x1 is already computed by the
    // time x0 exists and discarding it saves nothing. Every consumer therefore takes as many
    // elements from a position as its element width allows: E = 64/W lanes, low lane first,
    // so element i is bits [i*W, (i+1)*W) of the draw's bit stream. The counter is a linear
    // index into that stream rather than resumable state, so this is only a choice of the
    // elements-per-position ratio; every bit the generator produces is equidistributed, so a
    // lane is as uniform as the whole value.

    /// <summary>
    /// <c>prod(shape)</c> lanes of <paramref name="width"/> bits, packed E = 64/<paramref
    /// name="width"/> per generator value, each carrying its value in the low bits (whatever
    /// rides above is the caller's to mask or narrow away).
    /// </summary>
    private static Vector<uint64> PackedLanes(
        Vector<int64> shape, Scalar<uint64> key, Scalar<uint64> substreamIndex, int width, int rounds)
    {
        long lanes = 64 / width;
        Scalar<int64> n = ElementCount(shape);
        var v = Draw((n + Scalar(lanes - 1)) / Scalar(lanes), key, substreamIndex, rounds);   // [ceil(N/E)]

        // The [M,1] values against the [E] lane offsets broadcast to [M,E], whose row-major
        // flatten lands lane l of value j at element j*E + l — the low-lane-first convention.
        var perLane = ShiftDown(v.Reshape(Vector(-1L, 1L)), VectorRange(0UL, 64UL, (ulong)width));
        return perLane.Reshape(Vector(-1L)).Vec().Slice(Scalar(0L), n);
    }

    /// <summary>Standard uniform U(0,1) of the given shape (bit generator: Threefry-2x32-<paramref name="rounds"/>).
    /// A uniform keeps 24 bits, so two fit in a position's 64 — one per 32-bit lane.</summary>
    public static Tensor<float32> StandardUniform(
        Vector<int64> shape, Scalar<uint64> key, Scalar<uint64> substreamIndex, int rounds = Threefry2x32.Rounds)
        => ToUniform(PackedLanes(shape, key, substreamIndex, 32, rounds)).Reshape(shape);

    /// <summary>Standard normal N(0,1) of the given shape (Box–Muller over Threefry-2x32-<paramref name="rounds"/>).
    /// Box–Muller turns a position's two uniforms into a <em>pair</em> of independent normals — the
    /// cosine and sine arms — so a position yields two elements: element 2j is the cosine arm of
    /// position j and element 2j+1 the sine arm.</summary>
    public static Tensor<float32> StandardNormal(
        Vector<int64> shape, Scalar<uint64> key, Scalar<uint64> substreamIndex, int rounds = Threefry2x32.Rounds)
    {
        Scalar<int64> n = ElementCount(shape);
        var v = Draw((n + Scalar(1L)) / Scalar(2L), key, substreamIndex, rounds);   // [ceil(N/2)]
        var u1 = ToUniform(v);                                  // low 32-bit lane
        var u2 = ToUniform(ShiftDown(v, Scalar(32UL)));         // high 32-bit lane
        var radius = ((-u1 + Scalar(1.0f)).Ln() * Scalar(-2.0f)).Sqrt();   // √(−2·ln(1−u₁))
        var theta = u2 * Scalar(2.0f * System.MathF.PI);

        var arms = (radius * theta.Cos()).Reshape(Vector(-1L, 1L))
            .Concat(1, (radius * theta.Sin()).Reshape(Vector(-1L, 1L)));   // [M,2]
        return arms.Reshape(Vector(-1L)).Vec().Slice(Scalar(0L), n).Reshape(shape);
    }

    /// <summary>U(low, high) of the given shape.</summary>
    public static Tensor<float32> Uniform(
        Vector<int64> shape, Scalar<uint64> key, Scalar<uint64> substreamIndex,
        Scalar<float32> low, Scalar<float32> high, int rounds = Threefry2x32.Rounds)
        => StandardUniform(shape, key, substreamIndex, rounds) * (high - low) + low;

    /// <summary>N(mean, scale) of the given shape.</summary>
    public static Tensor<float32> Normal(
        Vector<int64> shape, Scalar<uint64> key, Scalar<uint64> substreamIndex,
        Scalar<float32> mean, Scalar<float32> scale, int rounds = Threefry2x32.Rounds)
        => StandardNormal(shape, key, substreamIndex, rounds) * scale + mean;

    // ── Raw random bits ─────────────────────────────────────────────────────────────────
    // Raw bits are lanes straight out of the packing above, narrowed to the requested width:
    // N/8 positions for U8, N/4 for U16, N/2 for U32. U64 is one whole value per element and
    // has nothing to pack.

    /// <summary>Raw uniform bits, U8 (8 elements packed per generator value), of the given shape.</summary>
    public static Tensor<uint8> BitsU8(
        Vector<int64> shape, Scalar<uint64> key, Scalar<uint64> substreamIndex, int rounds = Threefry2x32.Rounds)
        => PackedLanes(shape, key, substreamIndex, 8, rounds).Cast<uint8>().Reshape(shape);

    /// <summary>Raw uniform bits, U16 (4 elements packed per generator value), of the given shape.</summary>
    public static Tensor<uint16> BitsU16(
        Vector<int64> shape, Scalar<uint64> key, Scalar<uint64> substreamIndex, int rounds = Threefry2x32.Rounds)
        => PackedLanes(shape, key, substreamIndex, 16, rounds).Cast<uint16>().Reshape(shape);

    /// <summary>Raw uniform bits, U32 (2 elements packed per generator value), of the given shape.</summary>
    public static Tensor<uint32> BitsU32(
        Vector<int64> shape, Scalar<uint64> key, Scalar<uint64> substreamIndex, int rounds = Threefry2x32.Rounds)
        => PackedLanes(shape, key, substreamIndex, 32, rounds).Cast<uint32>().Reshape(shape);

    /// <summary>Raw uniform bits, U64 (one whole generator value per element), of the given shape.</summary>
    public static Tensor<uint64> BitsU64(
        Vector<int64> shape, Scalar<uint64> key, Scalar<uint64> substreamIndex, int rounds = Threefry2x32.Rounds)
        => Draw(ElementCount(shape), key, substreamIndex, rounds).Reshape(shape);
}
