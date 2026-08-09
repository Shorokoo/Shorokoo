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
/// fixed <c>(key, substreamIndex, p)</c> replays exactly. Bit→float is the geometric draw (a
/// geometric octave times a 23-bit mantissa fraction — see <see cref="GeometricUniform"/>);
/// the normal transform is Box–Muller with radius = √(−2·ln w), w that geometric draw. Mirrors
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

    // ── Geometric uniform (Walker 1974; Reynolds' 41+23 form) ────────────────────────────
    // Walker, "Fast Generation of Uniformly Distributed Pseudorandom Numbers with Floating-Point
    // Representation" (1974) — independently rederived by Downey (2007). The 41-bit exponent field
    // plus 23-bit significand split of a single 64-bit draw is Marc Reynolds' practical form.
    // The 24-bit ToUniform grid above is what Box–Muller consumes; the PUBLIC uniform instead
    // draws the octave geometrically so it reaches the full float32 precision near zero.
    //
    // A uniform on [0,1) is, per octave, an even grid: [2^-1,1) carries half the mass, [2^-2,2^-1)
    // a quarter, and so on, each octave holding 2^23 equally-spaced floats. So draw the octave from
    // a geometric distribution — the count of leading zeros of a 41-bit field, P(octave e) = 2^e —
    // and fill the 23-bit mantissa uniformly. The value is a [1,2) fraction times a power-of-two
    // octave scale: both are exact in float32 (the fraction has a 24-bit significand, the scale only
    // shifts the exponent), so the whole draw is EXACT — no Cast rounding, no transcendental, hence
    // EP-independent. Reaches ~2^-41 near zero (vs the 24-bit grid's 2^-24), at one 64-bit
    // generator value per element (the packing/precision trade — the bits buy resolution).

    private static readonly float[] GeoOctaveScales = BuildGeoOctaveScales();
    private const int GeoExpBits = 41;                          // leading-zero window; reaches 2^-(GeoExpBits)
    private static float[] BuildGeoOctaveScales()               // [2^-1, 2^-2, …, 2^-GeoExpBits], all exact
    {
        float[] t = new float[GeoExpBits];
        for (int i = 0; i < GeoExpBits; i++) t[i] = (float)System.Math.Pow(2.0, -1 - i);
        return t;
    }

    /// <summary>
    /// One geometric uniform per 64-bit generator value: geometric octave (leading-zero count of the
    /// top 41 bits) × uniform 23-bit mantissa. Every produced <em>value</em> is exact — the fraction
    /// and the octave scale are both exact in float32, so no rounding occurs anywhere — and therefore
    /// EP-independent.
    ///
    /// <para>The <em>distribution</em> is a uniform truncated at 2⁻⁴¹: an all-zero exponent field
    /// falls into the same bucket as a field of 1, so the deepest octave carries double mass
    /// (2⁻⁴⁰ instead of 2⁻⁴¹) and nothing below 2⁻⁴¹ is produced. The support is also open at both
    /// ends — the range is [2⁻⁴¹, 1−2⁻²⁴], so exact 0 is never returned (the old 24-bit grid returned
    /// it with probability 2⁻²⁴).</para>
    /// </summary>
    private static Tensor<float32> GeometricUniform(Vector<uint64> v)
    {
        // Mantissa: low 23 bits → frac in [1,2). m < 2^23 casts to float32 without rounding, and
        // 1 + m·2⁻²³ lands on the exact float grid of [1,2) (ULP there is 2⁻²³).
        var m = OnnxOp.BitwiseAnd(v, Scalar((1UL << 23) - 1)).uint64();
        var frac = m.Cast<float32>() * Scalar(1.0f / 8388608.0f) + Scalar(1.0f);

        // Highest-set-bit position p∈[0,40] of the 41-bit exponent field, branchless (a field of 0
        // leaves p=0 → deepest octave). Selection is arithmetic, not Where — ORT has no Where for
        // uint64 — so every step is BitShift/Greater/Mul/Add: pure integer, identical on every EP.
        var ef = OnnxOp.BitwiseAnd(ShiftDown(v, Scalar(23UL)), Scalar((1UL << GeoExpBits) - 1)).uint64();
        Tensor<uint64> x = ef;
        Tensor<uint64> p = ShiftDown(ef, Scalar(63UL));               // zeros [N]
        foreach (int s in (int[])[32, 16, 8, 4, 2, 1])
        {
            // add = s where the top half is non-empty, else 0; then shift it away and tally it.
            var add = OnnxOp.Greater(ShiftDown(x, Scalar((ulong)s)), Scalar(0UL)).Cast<uint64>() * Scalar((ulong)s);
            p = (p + add).uint64();
            x = OnnxOp.BitShift(x, add, BitShiftDirection.Right).uint64();
        }
        // Octave index = leadingZeros = (GeoExpBits-1) - p; scale = 2^(-1-index); value = frac·scale, exact.
        var index = Scalar((long)(GeoExpBits - 1)) - p.Cast<int64>();
        var scale = (Tensor<float32>)OnnxOp.Gather(Vector(GeoOctaveScales), index, axis: 0);
        return frac * scale;
    }

    /// <summary>Standard uniform U(0,1) of the given shape (Walker's geometric draw over
    /// Threefry-2x32-<paramref name="rounds"/>): full-precision near zero, exact, EP-independent.
    /// One 64-bit generator value per element.</summary>
    public static Tensor<float32> StandardUniform(
        Vector<int64> shape, Scalar<uint64> key, Scalar<uint64> substreamIndex, int rounds = Threefry2x32.Rounds)
        => GeometricUniform(Draw(ElementCount(shape), key, substreamIndex, rounds)).Reshape(shape);

    /// <summary>Standard normal N(0,1) of the given shape (Box–Muller over Threefry-2x32-<paramref name="rounds"/>).
    /// Box–Muller turns a (radius, angle) pair into a <em>pair</em> of independent normals — the cosine
    /// and sine arms — so element 2j is the cosine arm of pair j and element 2j+1 the sine arm.
    ///
    /// <para>The radius is <c>√(−2·ln w)</c> where <c>w</c> is the <b>geometric</b> uniform (fine near
    /// zero, reaching ~2⁻⁴¹) rather than the 24-bit grid's <c>1−u₁</c> (floored at 2⁻²⁴). That deepens
    /// the reachable tail from ±5.77σ to ~±7.54σ and resolves it finely, and since <c>w > 0</c> always
    /// there is no <c>ln(0)</c>. The angle stays a 24-bit uniform — an even grid is what a uniform angle
    /// wants. Each pair spends two generator values: an even position for the radius's geometric draw,
    /// the odd next one for the angle.</para></summary>
    public static Tensor<float32> StandardNormal(
        Vector<int64> shape, Scalar<uint64> key, Scalar<uint64> substreamIndex, int rounds = Threefry2x32.Rounds)
    {
        Scalar<int64> n = ElementCount(shape);
        Scalar<int64> pairs2 = (n + Scalar(1L)) / Scalar(2L) * Scalar(2L);          // 2·ceil(N/2)
        var block = Draw(pairs2, key, substreamIndex, rounds);                       // [2M] positions
        var w  = GeometricUniform(block.Slice(Scalar(0L), pairs2, Scalar(2L)));      // even → radius draw
        var u2 = ToUniform(block.Slice(Scalar(1L), pairs2, Scalar(2L)));             // odd  → 24-bit angle
        var radius = (w.Ln() * Scalar(-2.0f)).Sqrt();                               // √(−2·ln w)
        var theta = u2 * Scalar(2.0f * System.MathF.PI);

        var arms = (radius * theta.Cos()).Reshape(Vector(-1L, 1L))
            .Concat(1, (radius * theta.Sin()).Reshape(Vector(-1L, 1L)));   // [M,2]
        return arms.Reshape(Vector(-1L)).Vec().Slice(Scalar(0L), n).Reshape(shape);
    }

    // ── Dense arbitrary-range uniform (region table) ─────────────────────────────────────
    // Walker/Reynolds generalized off [0,1) onto an arbitrary range: every float in [low, high)
    // is reachable with probability proportional to its ulp, from ONE 64-bit generator value per
    // element and with no rejection. The host oracle (tests: RngDenseUniformOracle) is the
    // contract; this rebuilds the same table in ONNX ops and must agree with it bit for bit.
    //
    // Floats are addressed by SIGNED ORDINAL z — the bit pattern for x >= 0, negated for x < 0 —
    // which is strictly monotone in the real value, so any range (straddling or not) is one
    // interval [zLow, zHigh) and the sign needs no separate draw. Float z owns [V(z), V(z+1)),
    // whose width is the ulp of weight class max(1, exponent field). The interval splits into
    // REGIONS of 2^bits equally-weighted floats: above the truncation floor the float grid itself,
    // cut into class groups and then into descending power-of-two blocks; below it a coarse even
    // lattice of spacing 2^(floorClass-150) carrying the remaining mass honestly.
    //
    // Weights are counted in units of the SHALLOWEST KEPT class's ulp (not the smallest
    // subnormal), which is what keeps the cumulative table inside int64 and pins the truncation
    // depth at DenseClasses = 38: the total never exceeds 2^62. Thresholds are floor(C*2^41/total)
    // by restoring binary long division, integer-only — 2^41*C overflows int64, and binary64 is
    // unusable because the Quick Execution Engine evaluates every float dtype at binary32
    // (Shorokoo#157), which would diverge silently between engines rather than fail.
    //
    // The table is built ONCE PER CALL (a [128] tensor), not per element.

    private const int DenseP = 23;                            // significand bits
    private const long DenseBinade = 1L << DenseP;            // floats per class group
    private const int DenseBias = 127;
    private const int DenseClasses = 38;                      // truncation depth K
    private const int DenseSelectorBits = 64 - DenseP;        // 41: the region-selector field
    private const int DenseMaxWeightExp = 62;                 // the accumulator's ceiling

    // The static [128]-slot layout. Every slot is a closed form of (zLow, zHigh) and the slot
    // index, so the table needs no data-dependent iteration; absent slots carry weight 0 and are
    // unreachable (a zero-weight slot's threshold equals the NEXT non-empty slot's, so the binary
    // search below — which takes the LAST slot whose threshold is <= the selector — always skips
    // past it, and the trailing fillers sit at threshold 2^41, above every selector).
    //
    // Slots run in ascending ordinal order, which is what makes the cumulative thresholds line up
    // with the oracle's. 126 of the 128 are used, because complementary families overlay: an
    // endpoint above the truncation floor spends its 24-slot family on the partial class group at
    // that endpoint and contributes exactly one full 2^P lattice block to the band; an endpoint
    // inside the band has no partial class group and spends the family on the partial lattice run
    // plus its stub instead.
    private const int DenseSlots = 128;
    private const int DenseLowAt = 0, DenseEndpointN = 24;    // partial family at zLow
    private const int DenseNegFullAt = 24, DenseFullN = 38;   // full class groups, negative ray
    private const int DenseNegBandAt = 62;                    // full lattice block [-2^P, 0)
    private const int DensePosBandAt = 63;                    // full lattice block [0, 2^P)
    private const int DensePosFullAt = 64;                    // full class groups, positive ray
    private const int DenseHighAt = 102;                      // partial family at zHigh
    private const int DensePadN = 2;

    private static readonly long[] DensePow2 = BuildDensePow2();
    private static readonly float[] DenseScale = BuildDenseScale();
    private static readonly float[] DenseSpacing = BuildDenseSpacing();
    private static readonly float[] DenseBoundary = BuildDenseBoundary();

    private static long[] BuildDensePow2()
    {
        long[] t = new long[DenseMaxWeightExp + 1];
        for (int i = 0; i < t.Length; i++) t[i] = 1L << i;
        return t;
    }

    /// <summary>2^(e-127) for a biased exponent field e; index 0 is unused (never gathered).</summary>
    private static float[] BuildDenseScale()
    {
        float[] t = new float[255];
        for (int e = 1; e < 255; e++) t[e] = System.MathF.ScaleB(1f, e - DenseBias);
        return t;
    }

    /// <summary>The lattice spacing 2^(c-150) of weight class c — the ulp of the shallowest kept
    /// class, i.e. one weight unit.</summary>
    private static float[] BuildDenseSpacing()
    {
        float[] t = new float[255];
        for (int c = 1; c < 255; c++) t[c] = System.MathF.ScaleB(1f, c - DenseBias - DenseP);
        return t;
    }

    /// <summary>Binade lower bounds for the exponent search: 2^(k-127), with the last entry
    /// +infinity so a finite magnitude never selects field 255.</summary>
    private static float[] BuildDenseBoundary()
    {
        float[] t = new float[256];
        for (int k = 1; k < 255; k++) t[k] = System.MathF.ScaleB(1f, k - DenseBias);
        t[255] = float.PositiveInfinity;
        return t;
    }

    /// <summary>A predicate as 0/1, so case analysis is arithmetic rather than a <c>Where</c>
    /// cascade (both of whose arms would evaluate anyway — ONNX has no lazy select outside
    /// <c>If</c>), and so no case ever depends on a build-time constant.</summary>
    private static Tensor<int64> Ind(Variable condition) => ((Tensor<bit>)condition).Cast<int64>();

    /// <summary>2^e, clamped to the accumulator's range so an absent slot's nonsense exponent can
    /// never gather out of bounds.</summary>
    private static Tensor<int64> DensePow(Tensor<int64> e)
        => OnnxOp.Gather(Vector(DensePow2), e.Max(Scalar(0L)).Min(Scalar((long)DenseMaxWeightExp)), axis: 0).int64();

    /// <summary>The signed ordinal of a finite float32, decoded arithmetically: ONNX has no
    /// bit-reinterpretation op at opset 21 (<c>BitCast</c> is opset 26 and throws), so the
    /// exponent field comes from a binary search over the binade boundaries and the significand
    /// from an exact power-of-two division. Every cast stays under 2^24, so <c>Cast</c>'s
    /// unspecified int-to-float rounding is never exercised (C-2/C-8, Shorokoo#156).</summary>
    private static Tensor<int64> DenseOrdinal(Tensor<float32> x)
    {
        var magnitude = x.Abs();
        Tensor<int64> field = Scalar(0L);
        foreach (long step in (long[])[128L, 64L, 32L, 16L, 8L, 4L, 2L, 1L])
        {
            var bound = OnnxOp.Gather(Vector(DenseBoundary), field + Scalar(step), axis: 0);
            field = field + Ind(OnnxOp.GreaterOrEqual(magnitude, bound)) * Scalar(step);
        }
        var implicitBit = Ind(OnnxOp.GreaterOrEqual(field, Scalar(1L)));
        var scale = (Tensor<float32>)OnnxOp.Gather(Vector(DenseScale), field.Max(Scalar(1L)), axis: 0);
        var significand = ((magnitude / scale - implicitBit.Cast<float32>()) * Scalar((float)DenseBinade))
            .Cast<int64>();
        var ordinal = field * Scalar(DenseBinade) + significand;
        return ordinal - Scalar(2L) * Ind(OnnxOp.Less(x, Scalar(0f))) * ordinal;
    }

    /// <summary>|V(z)| divided by the lattice spacing, truncated, and whether it divided exactly —
    /// the band's floor/ceil primitive. Pure int64: the significand carries at most 24 bits and
    /// the spacing only shifts.</summary>
    private static (Tensor<int64> Quotient, Tensor<int64> Exact) DenseOverSpacing(
        Tensor<int64> magnitude, Tensor<int64> floorExponent)
    {
        var subnormal = Ind(OnnxOp.Less(magnitude, Scalar(DenseBinade)));
        var binade = magnitude / Scalar(DenseBinade);
        var exponent = (Scalar(1L) - subnormal) * (binade - Scalar((long)DenseBias))
                     + subnormal * Scalar(1L - DenseBias);
        var significand = (Scalar(1L) - subnormal) * (Scalar(DenseBinade) + magnitude - binade * Scalar(DenseBinade))
                        + subnormal * magnitude;
        var step = DensePow(floorExponent - exponent);
        var quotient = significand / step;
        return (quotient, Ind(OnnxOp.Equal(quotient * step, significand)));
    }

    /// <summary>One endpoint family: the greedy binary decomposition of the run
    /// <c>[from, to)</c> into descending power-of-two blocks, laid out so slot <c>j</c> holds the
    /// block of size 2^(P-offset-j) — ascending in base, which is what keeps the cumulative
    /// thresholds aligned with the oracle's slot order. <paramref name="offset"/> shifts the run
    /// down one slot to leave the family's last slot for a trailing stub; that is safe precisely
    /// because a stub only exists when the run is shorter than a full class group, so its bit-P
    /// block is absent.</summary>
    private static (Tensor<int64> Base, Tensor<int64> Bits, Tensor<int64> Weight) DenseRunFamily(
        Tensor<int64> slot, Tensor<int64> from, Tensor<int64> to, Tensor<int64> lattice,
        Tensor<int64> weightClass, Tensor<int64> floorClass, Tensor<int64> offset)
    {
        var length = (to - from).Max(Scalar(0L));
        var bits = Scalar((long)DenseP) - offset - slot;
        var size = DensePow(bits);
        var half = length / size;
        var present = (half - half / Scalar(2L) * Scalar(2L)) * Ind(OnnxOp.GreaterOrEqual(bits, Scalar(0L)));
        var above = DensePow(bits + Scalar(1L));
        var weight = (Scalar(1L) - lattice) * DensePow(bits + weightClass - floorClass) + lattice * size;
        return (present * (from + length / above * above), present * bits, present * weight);
    }

    /// <summary>
    /// The region table for <c>[low, high)</c>: [128] slots of (threshold, base, index bits,
    /// lattice flag) plus the call-level lattice spacing, and the fixed result that replaces the
    /// draw for a NaN bound or an empty range. <paramref name="low"/> and <paramref name="high"/>
    /// are graph inputs, so every case below is data-driven — <c>RngAlgorithms</c> caches one
    /// shared uniform <c>Function</c> and cannot specialize on their values.
    /// </summary>
    private static (Tensor<int64> Threshold, Tensor<int64> Base, Tensor<int64> Bits,
                    Tensor<int64> Lattice, Tensor<float32> Spacing,
                    Tensor<bit> UseFixed, Tensor<float32> Fixed) BuildDenseTable(
        Tensor<float32> low, Tensor<float32> high)
    {
        const long binade = DenseBinade;

        // Non-finite bounds: NaN anywhere yields NaN, infinities clamp to the finite extremes.
        // +infinity as the upper bound is the one ordinal past the largest finite float, so the
        // whole finite domain stays reachable.
        var notANumber = (Tensor<bit>)OnnxOp.Or(OnnxOp.IsNaN(low), OnnxOp.IsNaN(high));
        var finiteLow = low.Max(Scalar(-float.MaxValue)).Min(Scalar(float.MaxValue));
        var finiteHigh = high.Max(Scalar(-float.MaxValue)).Min(Scalar(float.MaxValue));
        var zLow = DenseOrdinal((Tensor<float32>)OnnxOp.Where(notANumber, Scalar(0f), finiteLow));
        var zHighRaw = DenseOrdinal((Tensor<float32>)OnnxOp.Where(notANumber, Scalar(1f), finiteHigh))
                     + Ind(OnnxOp.Greater(high, Scalar(float.MaxValue)));

        // An inverted or empty range yields `low` — one rule covering low == high and low > high.
        // The table is still built over a one-float range so no downstream arithmetic degenerates.
        var empty = Ind(OnnxOp.LessOrEqual(zHighRaw, zLow));
        var useFixed = (Tensor<bit>)OnnxOp.Or(notANumber, OnnxOp.Greater(empty, Scalar(0L)));
        var fixedValue = (Tensor<float32>)OnnxOp.Where(notANumber, Scalar(float.NaN), finiteLow);
        var zHigh = zHighRaw.Max(zLow + Scalar(1L));

        // ── Truncation floor, band and lattice ──────────────────────────────────────────
        var magnitudeLow = zLow.Max(Scalar(0L) - zLow - Scalar(1L));
        var magnitudeTop = (zHigh - Scalar(1L)).Max(Scalar(1L) - zHigh);
        var topClass = (magnitudeLow / Scalar(binade)).Max(magnitudeTop / Scalar(binade)).Max(Scalar(1L));
        var floorClass = (topClass - Scalar((long)(DenseClasses - 1))).Max(Scalar(1L));
        var floorExponent = floorClass - Scalar((long)DenseBias);
        var latticeShift = floorClass - Scalar(1L);
        var zFloor = floorClass * Scalar(binade);
        var bandLow = (Scalar(0L) - zFloor).Max(zLow).Min(zHigh);
        var bandHigh = zFloor.Max(zLow).Min(zHigh);

        var (quotientLow, exactLow) = DenseOverSpacing(bandLow.Max(Scalar(0L) - bandLow), floorExponent);
        var (quotientHigh, exactHigh) = DenseOverSpacing(bandHigh.Max(Scalar(0L) - bandHigh), floorExponent);
        var negativeLow = Ind(OnnxOp.Less(bandLow, Scalar(0L)));
        var negativeHigh = Ind(OnnxOp.Less(bandHigh, Scalar(0L)));
        var latticeFrom = (Scalar(1L) - negativeLow) * (quotientLow + Scalar(1L) - exactLow)
                        - negativeLow * quotientLow;
        var latticeTo = (Scalar(1L) - negativeHigh) * quotientHigh
                      - negativeHigh * (quotientHigh + Scalar(1L) - exactHigh);

        var bandNonEmpty = Ind(OnnxOp.Less(bandLow, bandHigh));
        // A band too narrow to hold a lattice point degenerates to its stub alone.
        var narrow = bandNonEmpty * Ind(OnnxOp.Greater(latticeFrom, latticeTo));
        var lattices = bandNonEmpty * (Scalar(1L) - narrow);
        var leadStub = bandNonEmpty * (Scalar(1L) - exactLow);
        var trailStub = lattices * (Scalar(1L) - exactHigh);
        var negativeTo = latticeTo.Min(Scalar(0L));
        var positiveFrom = latticeFrom.Max(Scalar(0L));
        var negativeFull = lattices * Ind(OnnxOp.Equal(latticeFrom, Scalar(-binade)))
                                    * Ind(OnnxOp.Equal(negativeTo, Scalar(0L)));
        var positiveFull = lattices * Ind(OnnxOp.Equal(positiveFrom, Scalar(0L)))
                                    * Ind(OnnxOp.Equal(latticeTo, Scalar(binade)));

        var lowAbove = Ind(OnnxOp.Less(zLow, bandLow));
        var lowInBand = (Scalar(1L) - lowAbove) * bandNonEmpty;
        var highAbove = Ind(OnnxOp.Less(bandHigh, zHigh));
        var highInBand = (Scalar(1L) - highAbove) * bandNonEmpty;
        var lowTakesNegative = lowInBand * lattices * Ind(OnnxOp.Less(latticeFrom, Scalar(0L)))
                             * (Scalar(1L) - negativeFull);
        var lowTakesPositive = lowInBand * lattices * Ind(OnnxOp.Greater(latticeFrom, Scalar(0L)))
                             * Ind(OnnxOp.Equal(latticeTo, Scalar(binade)));

        // ── The family at zLow ──────────────────────────────────────────────────────────
        var negativeZLow = Ind(OnnxOp.Less(zLow, Scalar(0L)));
        var groupLow = magnitudeLow / Scalar(binade);
        var groupEndLow = (Scalar(1L) - Scalar(2L) * negativeZLow) * groupLow * Scalar(binade)
                        + (Scalar(1L) - negativeZLow) * Scalar(binade);
        var lowRunEnd = lowAbove * bandLow + (Scalar(1L) - lowAbove) * zHigh;
        var lowOrdinalTo = groupEndLow.Min(lowRunEnd);
        var lowFrom = lowInBand * (lowTakesNegative * latticeFrom + lowTakesPositive * positiveFrom)
                    + (Scalar(1L) - lowInBand) * zLow;
        var lowTo = lowInBand * (lowTakesNegative * negativeTo + lowTakesPositive * latticeTo)
                  + (Scalar(1L) - lowInBand) * lowOrdinalTo;

        var slot = OnnxOp.Range(Scalar(0L), Scalar((long)DenseEndpointN), Scalar(1L)).int64();
        var (lowBase, lowBits, lowWeight) = DenseRunFamily(
            slot, lowFrom, lowTo, lowInBand, (Scalar(1L) - lowInBand) * groupLow, floorClass, Scalar(0L));
        var atFirst = Ind(OnnxOp.Equal(slot, Scalar(0L))) * leadStub;
        var lowLattice = lowInBand * (Scalar(1L) - atFirst) + Scalar(0L) * slot;
        lowBase = lowBase * (Scalar(1L) - atFirst) + atFirst * bandLow;
        lowBits = lowBits * (Scalar(1L) - atFirst);
        lowWeight = lowWeight * (Scalar(1L) - atFirst) + atFirst;

        // ── Full class groups, both rays ────────────────────────────────────────────────
        var group = OnnxOp.Range(Scalar(0L), Scalar((long)DenseFullN), Scalar(1L)).int64();
        var negativeBase = Scalar(0L) - (groupLow - group) * Scalar(binade);
        var negativePresent = lowAbove * Ind(OnnxOp.LessOrEqual(negativeBase + Scalar(binade), bandLow));
        var negativeWeight = negativePresent
            * DensePow(Scalar((long)DenseP) + groupLow - Scalar(1L) - group - floorClass);

        var lowOrdinalPositive = (Scalar(1L) - lowInBand) * (Scalar(1L) - lowAbove);
        var positiveStart = lowOrdinalPositive * groupEndLow + (Scalar(1L) - lowOrdinalPositive) * bandHigh;
        var positiveBase = positiveStart + group * Scalar(binade);
        var positivePresent = highAbove * Ind(OnnxOp.LessOrEqual(positiveBase + Scalar(binade), zHigh));
        var positiveWeight = positivePresent
            * DensePow(Scalar((long)DenseP) + positiveBase / Scalar(binade) - floorClass);

        // ── The family at zHigh ─────────────────────────────────────────────────────────
        var magnitudeHigh = magnitudeTop;
        var negativeZHigh = Ind(OnnxOp.Less(zHigh - Scalar(1L), Scalar(0L)));
        var groupHigh = magnitudeHigh / Scalar(binade);
        var groupStartHigh = (Scalar(1L) - Scalar(2L) * negativeZHigh) * groupHigh * Scalar(binade)
                           - negativeZHigh * Scalar(binade);
        var highRunFrom = highAbove * bandHigh + (Scalar(1L) - highAbove) * zLow;
        var highOrdinalFrom = groupStartHigh.Max(highRunFrom);
        // The trailing class group is a slot of its own only when it is PARTIAL and not already
        // covered by the family at zLow — a full one is a plain class group above.
        var covered = (Scalar(1L) - lowInBand) * Ind(OnnxOp.GreaterOrEqual(lowOrdinalTo, zHigh));
        var trailing = (Scalar(1L) - covered) * Ind(OnnxOp.Less(zHigh - groupStartHigh, Scalar(binade)));
        var highOrdinalTo = highOrdinalFrom + trailing * (zHigh - highOrdinalFrom);
        var highTakesNegative = lattices * (Scalar(1L) - negativeFull) * (Scalar(1L) - lowTakesNegative)
                              * Ind(OnnxOp.Less(latticeFrom, negativeTo));
        var highTakesPositive = lattices * (Scalar(1L) - positiveFull) * (Scalar(1L) - lowTakesPositive)
                              * Ind(OnnxOp.Less(positiveFrom, latticeTo));
        var highFrom = highInBand * (highTakesNegative * latticeFrom + highTakesPositive * positiveFrom)
                     + (Scalar(1L) - highInBand) * highOrdinalFrom;
        var highTo = highInBand * (highTakesNegative * negativeTo + highTakesPositive * latticeTo)
                   + (Scalar(1L) - highInBand) * highOrdinalTo;

        var (highBase, highBits, highWeight) = DenseRunFamily(
            slot, highFrom, highTo, highInBand, (Scalar(1L) - highInBand) * groupHigh, floorClass, trailStub);
        var atLast = Ind(OnnxOp.Equal(slot, Scalar((long)(DenseEndpointN - 1)))) * trailStub;
        var highLattice = highInBand + Scalar(0L) * slot;
        highBase = highBase * (Scalar(1L) - atLast) + atLast * latticeTo;
        highBits = highBits * (Scalar(1L) - atLast);
        highWeight = highWeight * (Scalar(1L) - atLast) + atLast;

        // ── Assemble the [128] table ────────────────────────────────────────────────────
        var one = Vector(1L);
        var pad = Vector(0L, 0L);
        var bandBits = Scalar((long)DenseP).Reshape(one);
        var bases = lowBase
            .Concat(0, negativePresent * negativeBase, (negativeFull * Scalar(-binade)).Reshape(one),
                    Scalar(0L).Reshape(one), positivePresent * positiveBase, highBase, pad);
        var bits = lowBits
            .Concat(0, negativePresent * Scalar((long)DenseP), negativeFull.Reshape(one) * bandBits,
                    positiveFull.Reshape(one) * bandBits, positivePresent * Scalar((long)DenseP), highBits, pad);
        var lattice = lowLattice
            .Concat(0, Scalar(0L) * group, negativeFull.Reshape(one), positiveFull.Reshape(one),
                    Scalar(0L) * group, highLattice, pad);
        var weights = lowWeight
            .Concat(0, negativeWeight, (negativeFull * Scalar(1L << DenseP)).Reshape(one),
                    (positiveFull * Scalar(1L << DenseP)).Reshape(one), positiveWeight, highWeight, pad);

        var total = weights.Reduce(ReduceKind.Sum, Vector(0L), keepDims: false).Max(Scalar(1L));
        var cumulative = weights.CumSum(Scalar(0L), exclusive: true);
        // Past the last non-empty slot the cumulative equals the total: divide there would double
        // a value already at 2^62 and wrap, so those slots take the top threshold directly.
        var below = Ind(OnnxOp.Less(cumulative, total));
        var threshold = DenseLongDivide(cumulative * below, total)
                      + (Scalar(1L) - below) * Scalar(1L << DenseSelectorBits);
        var spacing = (Tensor<float32>)OnnxOp.Gather(Vector(DenseSpacing), floorClass, axis: 0);
        return (threshold, bases, bits, lattice, spacing, useFixed, fixedValue);
    }

    /// <summary>floor(numerator·2^41 / denominator), exactly, in int64, by restoring binary long
    /// division. Doubling is a MULTIPLY, not a shift: ONNX constrains <c>BitShift</c> to unsigned
    /// types, and ORT has no <c>uint64</c> <c>Where</c>. Every intermediate stays under 2^63
    /// because the denominator never exceeds 2^62 — which is what pins the truncation depth.</summary>
    private static Tensor<int64> DenseLongDivide(Tensor<int64> numerator, Tensor<int64> denominator)
    {
        var remainder = numerator;
        var quotient = numerator * Scalar(0L);
        for (int i = 0; i < DenseSelectorBits; i++)
        {
            remainder = remainder * Scalar(2L);
            var fits = Ind(OnnxOp.GreaterOrEqual(remainder, denominator));
            remainder = remainder - fits * denominator;
            quotient = quotient * Scalar(2L) + fits;
        }
        return quotient;
    }

    /// <summary>
    /// U(low, high) of the given shape, drawn densely: every float in the range is reachable with
    /// probability proportional to its ulp, from one 64-bit generator value per element, with no
    /// rejection and a static node count. The draw splits as region selector (the top 41 bits)
    /// against the cumulative threshold table, and index within the region (the low 23 bits).
    ///
    /// <para>The interval is half-open — <c>high</c> is not attainable, matching PyTorch
    /// <c>uniform_</c>, Keras <c>RandomUniform</c> and ONNX <c>RandomUniform</c>. An inverted or
    /// empty range yields <c>low</c>; a NaN bound yields NaN; infinite bounds clamp to the finite
    /// extremes.</para>
    ///
    /// <para>Selection is a 7-round binary search over the [128] table. It needs no bounds clamp —
    /// starting at 0, the steps 64+32+…+1 total 127 — and no <c>T[0] &lt;= s</c> guard, because
    /// slot 0's threshold is structurally 0 (its cumulative weight is). Both facts are load-bearing
    /// if the table's shape is ever changed.</para>
    /// </summary>
    public static Tensor<float32> Uniform(
        Vector<int64> shape, Scalar<uint64> key, Scalar<uint64> substreamIndex,
        Scalar<float32> low, Scalar<float32> high, int rounds = Threefry2x32.Rounds)
    {
        var (threshold, bases, bits, lattice, spacing, useFixed, fixedValue) = BuildDenseTable(low, high);
        var draw = Draw(ElementCount(shape), key, substreamIndex, rounds);
        var selector = ShiftDown(draw, Scalar((ulong)DenseP)).Cast<int64>();
        var index = OnnxOp.BitwiseAnd(draw, Scalar((1UL << DenseP) - 1)).uint64().Cast<int64>();

        Tensor<int64> found = selector * Scalar(0L);
        foreach (long step in (long[])[64L, 32L, 16L, 8L, 4L, 2L, 1L])
        {
            var probe = OnnxOp.Gather(threshold, found + Scalar(step), axis: 0);
            found = found + Ind(OnnxOp.LessOrEqual(probe, selector)) * Scalar(step);
        }
        var regionBase = OnnxOp.Gather(bases, found, axis: 0).int64();
        var regionBits = OnnxOp.Gather(bits, found, axis: 0).int64();
        var regionLattice = OnnxOp.Gather(lattice, found, axis: 0).int64();
        var value = regionBase + index / DensePow(Scalar((long)DenseP) - regionBits);

        // The lattice decode is the one float step: a lattice point is n·spacing with |n| <= 2^23,
        // so the cast is exact and the scaling only moves the exponent. Multiplying by the lattice
        // flag first keeps the cast in range on an ordinal region too.
        var latticeValue = (value * regionLattice).Cast<float32>() * spacing;

        // The ordinal decode reassembles (1 + m·2^-23)·2^(e-127) — every step exact in float32,
        // and the same value the oracle assembles bitwise.
        var magnitude = value.Max(Scalar(0L) - value);
        var field = (magnitude / Scalar(DenseBinade)).Min(Scalar(254L));
        var significand = magnitude - field * Scalar(DenseBinade);
        var scale = (Tensor<float32>)OnnxOp.Gather(Vector(DenseScale), field.Max(Scalar(1L)), axis: 0);
        var fraction = significand.Cast<float32>() * Scalar(1.0f / DenseBinade)
                     + Ind(OnnxOp.GreaterOrEqual(field, Scalar(1L))).Cast<float32>();
        var sign = Scalar(1.0f) - Scalar(2.0f) * Ind(OnnxOp.Less(value, Scalar(0L))).Cast<float32>();
        var ordinalValue = sign * fraction * scale;

        var drawn = (Tensor<float32>)OnnxOp.Where(
            (Tensor<bit>)OnnxOp.Greater(regionLattice, Scalar(0L)), latticeValue, ordinalValue);
        return ((Tensor<float32>)OnnxOp.Where(useFixed, fixedValue, drawn)).Reshape(shape);
    }

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
