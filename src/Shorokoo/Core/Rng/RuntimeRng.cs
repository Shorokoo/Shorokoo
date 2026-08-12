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

    // ── Dense arbitrary-range uniform (weight blocks) ────────────────────────────────────
    // Walker/Reynolds generalized off [0,1) onto an arbitrary range, from ONE 64-bit generator
    // value per element, with no rejection and a static node count. The host oracle (tests:
    // RngDenseUniformOracle) is the contract; this rebuilds it in ONNX ops and must agree with it
    // bit for bit.
    //
    // Floats are addressed by SIGNED ORDINAL z — the bit pattern for x >= 0, negated for x < 0 —
    // which is strictly monotone in the real value, so any range (straddling or not) is one
    // interval [zLow, zHigh) and the sign needs no separate draw. Float z owns [V(z), V(z+1)),
    // whose width is the ulp of weight class max(1, |z|'s pattern >> P) — the max(1, ...) is what
    // puts the subnormals and the smallest normal binade in one class. Note the asymmetry the
    // convention forces: a negative ordinal's class comes from the magnitude pattern BELOW it, so
    // class c on the negative side is the magnitudes (c<<P, (c+1)<<P], one ordinal off from the
    // binade. The classes still tile without gaps, which is all the decode needs.
    //
    // The interval is partitioned into seven BLOCKS, not a table. The weight axis need not run in
    // value order, so each kind of material becomes one contiguous block with a closed-form decode
    // and nothing is looked up: the lattice, the whole classes present on both signs, the whole
    // classes present on one, a partial class at each end of the range, and a stub at each end of
    // the lattice. Any block may be empty. Three decode forms cover all seven — lattice point,
    // geometric (whole classes), ordinal run — because a stub is a one-element run and a one-point
    // lattice respectively.
    //
    // TRUNCATION is the one approximation, and it is RANGE-DEPENDENT: 41 weight classes are
    // resolved as floats, 40 when the range straddles zero, since a straddling range carries both
    // signs of every class and 41 of them would total 2^65. Below the floor 2^(floorClass-127) a
    // coarse even lattice of spacing 2^(floorClass-150) carries the remaining mass, as one block
    // spanning both signs, so the floats down there are NOT individually reachable — 33.1% of the
    // floats in [0,1) are, 16.1% over the whole finite domain. What is exact is the MASS: every
    // block keeps the probability its width earns, so the draw stays uniform in VALUE. It is
    // resolution that is spent, not fairness.
    //
    // SELECTION spends the whole 64-bit draw and rounds NOTHING — no block is quantized, unlike a
    // scheme that maps blocks onto a fixed-width selector field. A block's threshold is its exact
    // cumulative weight in uint64 (six adds; ORT has no uint64 CumSum kernel), and the draw scales
    // onto the weight axis by floor(draw*total / 2^64), the high half of the 128-bit product —
    // Lemire's multiply-shift, so no division either. Both depths above top the total out at
    // exactly 2^64, which does not fit: it WRAPS to 0, and 0 is therefore the sentinel for it. That
    // costs one comparison in the scaling (where the wrap makes the scaling the identity) and one
    // in the block search (where a threshold must not be mistaken for the wrapped total); no
    // threshold can ever carry the value 2^64 itself, so nothing else needs 65 bits.
    //
    // Within a whole-class block the member index is the offset's LOW bits, not offset >> shift.
    // Both are exactly weight-preserving — a run of n indices each of weight 2^s spans n*2^s units,
    // so every residue mod n is hit exactly 2^s times — but the low-bits form is what drops the
    // mantissa out of the draw's low bits, which is what Walker/Reynolds does.
    //
    // So on [0,1) this draw IS Walker/Reynolds above the truncation floor, bit for bit: the range
    // is one-sided so it keeps 41 classes, the total is exactly 2^64 and the scaling is the
    // identity, the lattice occupies [0, 2^P) so the class block's offset is the draw itself, the
    // leading-bit search is Walker's leading-zero count over a 41-bit field, and the low P bits are
    // his mantissa. Below the floor the two part ways BY CONSTRUCTION, and must: the lattice
    // reaches exact zero, where Walker stops at 2^-41 and doubles his bottom binade's mass. Closing
    // that last 2^-41 would mean reproducing a wrong answer, so the difference stays. Sampling
    // cannot reach it either way, so RngRuntimeTests asserts both sides directly.
    //
    // The blocks are built ONCE PER CALL (seven-element columns), not per element.

    private const int DenseP = 23;                            // significand bits
    private const long DenseBinade = 1L << DenseP;            // floats per weight class, one sign
    private const int DenseBias = 127;
    private const int DenseMaxClasses = 41;                   // truncation depth off the straddle
    private const int DenseBlocks = 7;                        // lattice, 2-sided, 1-sided, 4 partials
    private const int DenseMaxShift = 62;                     // the int64 power table's top exponent
    private const int DenseMaxWeight = 63;                    // the uint64 one's

    private static readonly long[] DensePow2 = BuildDensePow2();
    private static readonly ulong[] DensePow2U = BuildDensePow2U();
    private static readonly float[] DenseScale = BuildDenseScale();
    private static readonly float[] DenseSpacing = BuildDenseSpacing();
    private static readonly float[] DenseBoundary = BuildDenseBoundary();

    private static long[] BuildDensePow2()
    {
        long[] t = new long[DenseMaxShift + 1];
        for (int i = 0; i < t.Length; i++) t[i] = 1L << i;
        return t;
    }

    private static ulong[] BuildDensePow2U()
    {
        ulong[] t = new ulong[DenseMaxWeight + 1];
        for (int i = 0; i < t.Length; i++) t[i] = 1UL << i;
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

    /// <summary><see cref="Ind"/> for the unsigned side of the table arithmetic.</summary>
    private static Tensor<uint64> IndU(Variable condition) => ((Tensor<bit>)condition).Cast<uint64>();

    /// <summary>2^e, clamped to the table's range so an empty block's nonsense exponent can never
    /// gather out of bounds.</summary>
    private static Tensor<int64> DensePow(Tensor<int64> e)
        => OnnxOp.Gather(Vector(DensePow2), e.Max(Scalar(0L)).Min(Scalar((long)DenseMaxShift)), axis: 0).int64();

    /// <summary><see cref="DensePow"/> in the weight arithmetic's uint64, which needs the one
    /// exponent int64 cannot hold: a single block may weigh 2^63.</summary>
    private static Tensor<uint64> DensePowU(Tensor<int64> e)
        => OnnxOp.Gather(Vector(DensePow2U), e.Max(Scalar(0L)).Min(Scalar((long)DenseMaxWeight)), axis: 0).uint64();

    /// <summary>The weight class of an ordinal: max(1, its magnitude pattern >> P).</summary>
    private static Tensor<int64> DenseClassOf(Tensor<int64> z)
        => (z.Max(Scalar(0L) - z - Scalar(1L)) / Scalar(DenseBinade)).Max(Scalar(1L));

    /// <summary>The per-block scalars laid out as one [7] column, in block order.</summary>
    private static Tensor<T> DenseColumn<T>(params Tensor<T>[] blocks) where T : IVarType
    {
        Tensor<T>[] rows = new Tensor<T>[blocks.Length];
        for (int i = 0; i < blocks.Length; i++) rows[i] = blocks[i].Reshape(Vector(1L));
        return rows[0].Concat(0, rows[1..]);
    }

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

    /// <summary>
    /// One sign's ordinal material <c>[from, to)</c> decomposed into a partial class at each end
    /// and the whole classes C0..C1 between — the oracle's <c>SplitRun</c>. A partial that happens
    /// to cover its whole class folds into the whole range instead, so the geometric block stays
    /// maximal; an absent run reports C1 &lt; C0 and both counts 0.
    ///
    /// <para><paramref name="negative"/> is a plain C# bool, resolved while the graph is built: the
    /// two rays differ only in which end of a class run an ordinal starts at and in which direction
    /// the classes ascend, so each call site emits its own straight-line arithmetic and no case
    /// depends on graph data.</para>
    /// </summary>
    private static (Tensor<int64> LowBase, Tensor<int64> LowCount, Tensor<int64> LowClass,
                    Tensor<int64> HighBase, Tensor<int64> HighCount, Tensor<int64> HighClass,
                    Tensor<int64> First, Tensor<int64> Last) DenseSplitRun(
        Tensor<int64> from, Tensor<int64> to, bool negative)
    {
        // Class c holds 2^P consecutive ordinals. Negative classes are the magnitudes
        // (c<<P, (c+1)<<P], so the run ends one ordinal below -(c<<P).
        Tensor<int64> RunStart(Tensor<int64> c)
            => negative ? Scalar(0L) - (c + Scalar(1L)) * Scalar(DenseBinade) : c * Scalar(DenseBinade);
        Tensor<int64> RunEnd(Tensor<int64> c)
            => negative ? Scalar(0L) - c * Scalar(DenseBinade) : (c + Scalar(1L)) * Scalar(DenseBinade);

        var present = Ind(OnnxOp.Less(from, to));
        var classFrom = DenseClassOf(from);
        var classTo = DenseClassOf(to - Scalar(1L));
        var lowEnd = RunEnd(classFrom).Min(to);
        var highStart = RunStart(classTo).Max(from);
        var lowWhole = Ind(OnnxOp.Equal(from, RunStart(classFrom)))
                     * Ind(OnnxOp.Equal(lowEnd, RunEnd(classFrom)));
        var highWhole = Ind(OnnxOp.Equal(to, RunEnd(classTo)))
                      * Ind(OnnxOp.Equal(highStart, RunStart(classTo)));

        // One class ends the run at both ends, so its partial is the low one alone.
        var lowCount = present * (Scalar(1L) - lowWhole) * (lowEnd - from);
        var highCount = present * (Scalar(1L) - highWhole) * (to - highStart)
                      * (Scalar(1L) - Ind(OnnxOp.Equal(classTo, classFrom)));
        var first = present * (negative ? classTo + (Scalar(1L) - highWhole)
                                        : classFrom + (Scalar(1L) - lowWhole));
        var last = present * (negative ? classFrom - (Scalar(1L) - lowWhole)
                                       : classTo - (Scalar(1L) - highWhole))
                 - (Scalar(1L) - present);
        return (from, lowCount, classFrom, highStart, highCount, classTo, first, last);
    }

    /// <summary>
    /// The seven weight blocks of <c>[low, high)</c>: the threshold column plus the columns their
    /// decodes read, the truncation floor and the lattice spacing, the summed weight, and the fixed
    /// result that replaces the draw for a NaN bound or an empty range. <paramref name="low"/> and
    /// <paramref name="high"/> are graph inputs, so every case below is data-driven —
    /// <c>RngAlgorithms</c> caches one shared uniform <c>Function</c> and cannot specialize on
    /// their values.
    ///
    /// <para>The blocks run in a FIXED order — lattice, both-sign classes, one-sign classes, the
    /// partial class at each end of the range, the stub at each end of the lattice — and any of
    /// them may weigh 0. An empty block's threshold is the next block's, which is exactly what
    /// keeps it unreachable, and the trailing ones carry the total.</para>
    ///
    /// <para>Internal rather than private so a test can hold the threshold column against the
    /// oracle's blocks. No amount of sampling covers that column: a block can own a single code out
    /// of 2^64, so a draw-based test agrees with a table that has dropped it.</para>
    /// </summary>
    internal static (Tensor<uint64> Threshold, Tensor<int64> Base, Tensor<int64> Class,
                    Tensor<int64> Width, Tensor<int64> Shift, Tensor<int64> Negative,
                    Tensor<int64> Geometric, Tensor<int64> Lattice, Tensor<int64> FloorClass,
                    Tensor<float32> Spacing, Tensor<bit> UseFixed, Tensor<float32> Fixed,
                    Tensor<uint64> Total) BuildDenseTable(
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
        // The blocks are still built over a one-float range so no downstream arithmetic degenerates.
        var empty = Ind(OnnxOp.LessOrEqual(zHighRaw, zLow));
        var useFixed = (Tensor<bit>)OnnxOp.Or(notANumber, OnnxOp.Greater(empty, Scalar(0L)));
        var fixedValue = (Tensor<float32>)OnnxOp.Where(notANumber, Scalar(float.NaN), finiteLow);
        var zHigh = zHighRaw.Max(zLow + Scalar(1L));

        // ── Truncation floor, band and lattice ──────────────────────────────────────────
        // A straddling range carries both signs of every class, so it keeps one class fewer: either
        // depth tops the total out at exactly 2^64, and neither may exceed it.
        var magnitudeLow = zLow.Max(Scalar(0L) - zLow - Scalar(1L));
        var magnitudeTop = (zHigh - Scalar(1L)).Max(Scalar(0L) - zHigh);
        var topClass = (magnitudeLow / Scalar(binade)).Max(magnitudeTop / Scalar(binade)).Max(Scalar(1L));
        var straddles = Ind(OnnxOp.Less(zLow, Scalar(0L))) * Ind(OnnxOp.Greater(zHigh, Scalar(0L)));
        var floorClass = (topClass - Scalar((long)DenseMaxClasses - 1L) + straddles).Max(Scalar(1L));
        var floorExponent = floorClass - Scalar((long)DenseBias);
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

        // The lattice takes the points the span wholly contains. Whichever end the range cuts
        // mid-cell leaves a sliver worth less than one weight unit, and it is dropped rather than
        // rounded up: the axis cannot express a part of a unit, and paying it a whole one
        // over-weighted that sliver's region by up to 2^191 — by far the worst distortion the
        // scheme had. At most one end is ever cut, since the other is a floor boundary.
        var bandNonEmpty = Ind(OnnxOp.Less(bandLow, bandHigh));
        var lattices = bandNonEmpty * (Scalar(1L) - Ind(OnnxOp.Greater(latticeFrom, latticeTo)));
        var latticeCount = lattices * (latticeTo - latticeFrom);

        // ── Whole classes, and the partial class at each end of the range ───────────────
        var (negLowBase, negLowCount, negLowClass, negHighBase, negHighCount, negHighClass,
             negFirst, negLast) = DenseSplitRun(zLow, bandLow, negative: true);
        var (posLowBase, posLowCount, posLowClass, posHighBase, posHighCount, posHighClass,
             posFirst, posLast) = DenseSplitRun(bandHigh, zHigh, negative: false);

        // Each ray's two partials get their own block. Folding them into one pair needs an argument
        // about which can coexist, and the obvious one is wrong: a range that straddles zero and
        // stops INSIDE the floor class on the positive side has a real partial at the low end of
        // BOTH rays, and collapsing them drops floats outright. Four blocks cost two compares.
        var negPresent = Ind(OnnxOp.Less(zLow, bandLow));
        var posPresent = Ind(OnnxOp.Less(bandHigh, zHigh));

        // The classes both rays hold whole become one block; whatever the deeper ray keeps above
        // them becomes the other, on whichever sign that is.
        var negWhole = negPresent * Ind(OnnxOp.GreaterOrEqual(negLast, negFirst));
        var posWhole = posPresent * Ind(OnnxOp.GreaterOrEqual(posLast, posFirst));
        var both = negWhole * posWhole;
        var shared = negLast.Min(posLast);
        var twoFirst = both * negFirst.Max(posFirst);
        var twoLast = both * shared;
        var oneFromNeg = negWhole * (Scalar(1L) - both) + both * Ind(OnnxOp.Greater(negLast, shared));
        var oneFromPos = posWhole * (Scalar(1L) - both)
                       + both * Ind(OnnxOp.LessOrEqual(negLast, shared)) * Ind(OnnxOp.Greater(posLast, shared));
        var onePresent = oneFromNeg + oneFromPos;
        var oneFirst = both * (shared + Scalar(1L))
                     + (Scalar(1L) - both) * (oneFromNeg * negFirst + oneFromPos * posFirst);
        var oneLast = oneFromNeg * negLast + oneFromPos * posLast;

        // ── Block weights, in units of the shallowest kept class's ulp ─────────────────
        // A class weighs twice the one below it, so a run of whole classes sums geometrically:
        // 2^(width + first - floorClass) * (2^(last - first + 1) - 1), width counting both signs at
        // P+1 and one at P. Every product is provably under 2^64, the total being at most that.
        var twoWeight = both.Cast<uint64>()
            * DensePowU(Scalar((long)DenseP + 1L) + twoFirst - floorClass)
            * (DensePowU(twoLast - twoFirst + Scalar(1L)) - Scalar(1UL));
        var oneWeight = onePresent.Cast<uint64>()
            * DensePowU(Scalar((long)DenseP) + oneFirst - floorClass)
            * (DensePowU(oneLast - oneFirst + Scalar(1L)) - Scalar(1UL));
        var negLowWeight = negLowCount.Cast<uint64>() * DensePowU(negLowClass - floorClass);
        var negHighWeight = negHighCount.Cast<uint64>() * DensePowU(negHighClass - floorClass);
        var posLowWeight = posLowCount.Cast<uint64>() * DensePowU(posLowClass - floorClass);
        var posHighWeight = posHighCount.Cast<uint64>() * DensePowU(posHighClass - floorClass);

        // ── Thresholds ─────────────────────────────────────────────────────────────────
        // A block's threshold IS its cumulative weight — no scaling, no division, no rounding — so
        // six adds do what a CumSum would, which is just as well: ORT has no uint64 kernel for one.
        // The total may reach exactly 2^64 and WRAP to 0, which is the sentinel for it; no
        // threshold can carry that value, so only the total ever means something other than itself.
        Tensor<uint64>[] weight =
        [
            latticeCount.Cast<uint64>(), twoWeight, oneWeight,
            negLowWeight, negHighWeight, posLowWeight, posHighWeight,
        ];
        Tensor<uint64>[] cumulative = new Tensor<uint64>[DenseBlocks];
        cumulative[0] = Scalar(0UL);
        for (int i = 1; i < DenseBlocks; i++) cumulative[i] = cumulative[i - 1] + weight[i - 1];
        var total = cumulative[DenseBlocks - 1] + weight[DenseBlocks - 1];

        // ── The block columns ──────────────────────────────────────────────────────────
        // Base is a lattice index for the lattice and its stub, an ordinal for the partials and the
        // low stub, and unread by the geometric blocks; Shift is the partials' weight per ordinal.
        Tensor<int64> zero = Scalar(0L);
        var spacing = (Tensor<float32>)OnnxOp.Gather(Vector(DenseSpacing), floorClass, axis: 0);
        return (DenseColumn(cumulative),
                DenseColumn(latticeFrom, zero, zero,
                            negLowBase, negHighBase, posLowBase, posHighBase),
                DenseColumn(zero, twoFirst, oneFirst, zero, zero, zero, zero),
                Vector(0L, DenseP + 1L, DenseP, 0L, 0L, 0L, 0L),
                DenseColumn(zero, zero, zero, negLowClass - floorClass, negHighClass - floorClass,
                            posLowClass - floorClass, posHighClass - floorClass),
                DenseColumn(zero, zero, oneFromNeg, zero, zero, zero, zero),
                Vector(0L, 1L, 1L, 0L, 0L, 0L, 0L),
                Vector(1L, 0L, 0L, 0L, 0L, 0L, 0L),
                floorClass, spacing, useFixed, fixedValue, total);
    }

    /// <summary>The high half of the 128-bit product a·b — Lemire's multiply-shift, which scales a
    /// draw onto [0, b) without a division. ONNX has no 128-bit type, so this is the schoolbook
    /// 32-bit split; every intermediate stays under 2^64. <paramref name="b"/> is the call's total
    /// weight, so all four products broadcast a tensor against a scalar.</summary>
    private static Tensor<uint64> DenseMulHigh(Tensor<uint64> a, Tensor<uint64> b)
    {
        var mask = Scalar(0xFFFF_FFFFUL);
        var (a0, a1) = (OnnxOp.BitwiseAnd(a, mask).uint64(), ShiftDown(a, Scalar(32UL)));
        var (b0, b1) = (OnnxOp.BitwiseAnd(b, mask).uint64(), ShiftDown(b, Scalar(32UL)));
        var (low, mid1, mid2) = (a0 * b0, a1 * b0, a0 * b1);
        var carry = ShiftDown(low, Scalar(32UL))
                  + OnnxOp.BitwiseAnd(mid1, mask).uint64() + OnnxOp.BitwiseAnd(mid2, mask).uint64();
        return a1 * b1 + ShiftDown(mid1, Scalar(32UL)) + ShiftDown(mid2, Scalar(32UL))
             + ShiftDown(carry, Scalar(32UL));
    }

    /// <summary>
    /// U(low, high) of the given shape, drawn densely: from one 64-bit generator value per element,
    /// with no rejection and a static node count. The whole draw scales onto the weight axis, the
    /// blocks' cumulative thresholds pick the block, and the offset above the winning threshold
    /// decodes in closed form — a lattice point, a run of ordinals, or a member of a run of whole
    /// weight classes.
    ///
    /// <para><b>Reachable floats.</b> Every float within the top 41 weight classes of the range —
    /// 40 when it straddles zero — is reachable with probability exactly proportional to its ulp,
    /// since selection rounds nothing. Below that truncation floor an even lattice carries the
    /// mass, and those floats are <b>not</b> individually reachable — 33.1% of the floats in [0,1)
    /// are reachable, 16.1% over the whole finite domain. The draw is still uniform in value there;
    /// it is resolution that is spent, not fairness.</para>
    ///
    /// <para>The interval is half-open — <c>high</c> is not attainable, matching PyTorch
    /// <c>uniform_</c>, Keras <c>RandomUniform</c> and ONNX <c>RandomUniform</c>. An inverted or
    /// empty range yields <c>low</c>; a NaN bound yields NaN; infinite bounds clamp to the finite
    /// extremes.</para>
    ///
    /// <para>Selection needs no search: over seven blocks, counting the thresholds at or below the
    /// scaled draw is cheaper than walking a tree, and the count IS the block index because an
    /// empty block carries the following block's threshold and so is never counted alone.</para>
    /// </summary>
    public static Tensor<float32> Uniform(
        Vector<int64> shape, Scalar<uint64> key, Scalar<uint64> substreamIndex,
        Scalar<float32> low, Scalar<float32> high, int rounds = Threefry2x32.Rounds)
    {
        var (threshold, bases, classes, widths, shifts, negatives, geometrics, lattices,
             floorClass, spacing, useFixed, fixedValue, total) = BuildDenseTable(low, high);
        var draw = Draw(ElementCount(shape), key, substreamIndex, rounds);
        // The whole draw, scaled onto the weight axis: floor(draw*total / 2^64). A total of 0 is
        // the 2^64 sentinel, where the scaling is the identity — mulhi by 0 is 0, so adding the
        // draw back covers that case without a branch.
        var scaled = DenseMulHigh(draw, total) + IndU(OnnxOp.Equal(total, Scalar(0UL))) * draw;

        // The owning block, as the count of thresholds at or below the scaled draw. The second
        // factor drops the trailing empty blocks, whose threshold IS the total: that may have
        // wrapped to 0, which would otherwise sit at or below every draw.
        Tensor<int64> found = scaled.Cast<int64>() * Scalar(0L);
        for (long block = 1; block < DenseBlocks; block++)
        {
            var probe = OnnxOp.Gather(threshold, Scalar(block), axis: 0).uint64();
            found = found + (Scalar(1L) - Ind(OnnxOp.Greater(probe, scaled)))
                          * (Scalar(1L) - Ind(OnnxOp.Equal(probe, total)));
        }
        var blockBase = OnnxOp.Gather(bases, found, axis: 0).int64();
        var blockClass = OnnxOp.Gather(classes, found, axis: 0).int64();
        var blockWidth = OnnxOp.Gather(widths, found, axis: 0).int64();
        var blockShift = OnnxOp.Gather(shifts, found, axis: 0).int64().Cast<uint64>();
        var blockNegative = OnnxOp.Gather(negatives, found, axis: 0).int64();
        var geometric = OnnxOp.Gather(geometrics, found, axis: 0).int64();
        var lattice = OnnxOp.Gather(lattices, found, axis: 0).int64();
        var offset = scaled - OnnxOp.Gather(threshold, found, axis: 0).uint64();

        // A run of whole classes decodes in closed form, because each class weighs twice the one
        // below it: offset + 2^m carries the class in the position of its leading bit — m is the
        // first class's own weight, in [P, 63], so the sum stays under 2^64 — and the member in its
        // low `width` bits, which is where the draw's own low bits are.
        var m = blockWidth + blockClass - floorClass;
        var shifted = offset + DensePowU(m);
        Tensor<int64> lead = Scalar(0L);
        foreach (long step in (long[])[32L, 16L, 8L, 4L, 2L, 1L])
            lead = lead + Scalar(step) * Ind(OnnxOp.Greater(
                ShiftDown(shifted, (lead + Scalar(step)).Cast<uint64>()), Scalar(0UL)));
        var index = OnnxOp.BitwiseAnd(shifted, DensePowU(blockWidth) - Scalar(1UL)).uint64();
        var mantissa = OnnxOp.BitwiseAnd(index, Scalar((ulong)DenseBinade - 1UL)).uint64().Cast<int64>();

        // A both-signs block spends the index's top bit on the sign; a one-sign block took its own.
        var twoSided = Ind(OnnxOp.Equal(blockWidth, Scalar((long)DenseP + 1L)));
        var negative = twoSided * ShiftDown(index, Scalar((ulong)DenseP)).Cast<int64>()
                     + (Scalar(1L) - twoSided) * blockNegative;
        // Negative class c is the magnitudes (c<<P, (c+1)<<P], so its mant'th member sits at
        // ordinal -((c+1)<<P) + mant rather than the positive side's (c<<P) + mant.
        var classIndex = blockClass + lead - m;
        var classOrdinal = (Scalar(1L) - Scalar(2L) * negative) * classIndex * Scalar(DenseBinade)
                         - negative * Scalar(DenseBinade) + mantissa;

        // Everything else is a run: base + offset >> shift, an ordinal for the partial classes and
        // the low stub, a lattice index for the lattice and its stub. Zeroing it in uint64 keeps a
        // whole-width geometric offset away from the int64 cast.
        var run = ((Scalar(1UL) - geometric.Cast<uint64>()) * ShiftDown(offset, blockShift)).Cast<int64>();
        var value = geometric * classOrdinal + (Scalar(1L) - geometric) * blockBase + run;

        // The lattice decode is the one float step: a lattice point is n·spacing with |n| <= 2^23,
        // so the cast is exact and the scaling only moves the exponent. Multiplying by the lattice
        // flag first keeps the cast in range on an ordinal block too.
        var latticeValue = (value * lattice).Cast<float32>() * spacing;

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
            (Tensor<bit>)OnnxOp.Greater(lattice, Scalar(0L)), latticeValue, ordinalValue);
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
