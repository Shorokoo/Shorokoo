using Shorokoo.Core.Inference.Abstractions;

namespace Shorokoo.Tests;

/// <summary>
/// Coverage-purpose tests for <see cref="Vector{T}"/> and <see cref="Scalar{T}"/>
/// in <c>Core/Vector.cs</c>, <c>Core/Vector.Index.cs</c>, and <c>Core/Scalar.cs</c>.
/// The static Unit/Empty per-DType arms, operator overloads (including
/// <see cref="PrimitiveParam"/> and bare-primitive variants),
/// ONNX op shortcut wrappers, and indexer overloads are exercised inside the
/// <c>Inline</c> bodies of <c>[Module]</c>-attributed classes — the source
/// generator invokes Inline once at type-init to build the ComputationGraph,
/// so coverage is captured even though the produced nodes are orphans and
/// get pruned during concretization.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class VectorScalarCoverageTests
{
    [Fact]
    public void TestVectorScalarCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<VectorUnitEmptyDispatchModel>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [4L])]));
        Assert.True(AutoTest.AdvancedTestGraph<VectorOperatorsModel>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [4L])]));
        Assert.True(AutoTest.AdvancedTestGraph<VectorOnnxOpsModel>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [4L])]));
        Assert.True(AutoTest.AdvancedTestGraph<VectorIndexerModel>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [4L])]));
        Assert.True(AutoTest.AdvancedTestGraph<ScalarUnitDispatchModel>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [4L])]));
        Assert.True(AutoTest.AdvancedTestGraph<ScalarOperatorsModel>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [4L])]));
        Assert.True(AutoTest.AdvancedTestGraph<ScalarImplicitPrimitiveConversionModel>(
            hyperparamInputs: [],
            runtimeInputs: []));
    }
}

/// <summary>
/// Exercises the per-DType dispatch arms of <see cref="Vector{T}.Unit"/> and
/// <see cref="Vector{T}.Empty"/>. Each static-property getter caches its value
/// per closed generic, so one access per DType is sufficient.
/// </summary>
[Module]
public partial class VectorUnitEmptyDispatchModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        _ = Vector<bit>.Unit;       _ = Vector<bit>.Empty;
        _ = Vector<int8>.Unit;      _ = Vector<int8>.Empty;
        _ = Vector<int16>.Unit;     _ = Vector<int16>.Empty;
        _ = Vector<int32>.Unit;     _ = Vector<int32>.Empty;
        _ = Vector<int64>.Unit;     _ = Vector<int64>.Empty;
        _ = Vector<uint8>.Unit;     _ = Vector<uint8>.Empty;
        _ = Vector<uint16>.Unit;    _ = Vector<uint16>.Empty;
        _ = Vector<uint32>.Unit;    _ = Vector<uint32>.Empty;
        _ = Vector<uint64>.Unit;    _ = Vector<uint64>.Empty;
        _ = Vector<bfloat16>.Unit;  _ = Vector<bfloat16>.Empty;
        _ = Vector<float16>.Unit;   _ = Vector<float16>.Empty;
        _ = Vector<float32>.Unit;   _ = Vector<float32>.Empty;
        _ = Vector<float64>.Unit;   _ = Vector<float64>.Empty;
        return input * Scalar(1.0f);
    }
}

/// <summary>
/// Exercises every operator overload on <see cref="Vector{T}"/>:
/// the arithmetic/bitwise/shift/unary/comparison forms in both
/// <c>Vector op Vector</c> and <c>Vector op Scalar</c> shapes,
/// across <c>float32</c> (arithmetic+comparison) and <c>int64</c>
/// (bitwise+shift) sub-vectors.
/// </summary>
[Module]
public partial class VectorOperatorsModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        // Vector<float32> — arithmetic & comparison.
        var v1 = Vector(1f, 2f, 3f, 4f);
        var v2 = Vector(5f, 6f, 7f, 8f);
        Scalar<float32> sp = 2f;

        _ = v1 + v2;  _ = sp + v1;  _ = v1 + sp;
        _ = v1 - v2;  _ = sp - v1;  _ = v1 - sp;
        _ = v1 * v2;  _ = sp * v1;  _ = v1 * sp;
        _ = v1 / v2;  _ = sp / v1;  _ = v1 / sp;
        _ = v1 % v2;  _ = sp % v1;  _ = v1 % sp;
        _ = -v1;

        _ = v1 >  v2; _ = sp >  v1; _ = v1 >  sp;
        _ = v1 >= v2; _ = sp >= v1; _ = v1 >= sp;
        _ = v1 <  v2; _ = sp <  v1; _ = v1 <  sp;
        _ = v1 <= v2; _ = sp <= v1; _ = v1 <= sp;
        _ = v1 == v2; _ = sp == v1; _ = v1 == sp;
        _ = v1 != v2; _ = sp != v1; _ = v1 != sp;

        // Vector<int64> — bitwise & shift.
        var iv1 = Vector(1L, 2L, 3L, 4L);
        var iv2 = Vector(5L, 6L, 7L, 8L);
        Scalar<int64> isp = 1L;

        _ = iv1 ^ iv2;  _ = isp ^ iv1;  _ = iv1 ^ isp;
        _ = iv1 & iv2;  _ = isp & iv1;  _ = iv1 & isp;
        _ = iv1 | iv2;  _ = isp | iv1;  _ = iv1 | isp;
        _ = iv1 << iv2; _ = iv1 << isp;
        _ = iv1 >> iv2; _ = iv1 >> isp;
        _ = !iv1;

        return input * Scalar(1.0f);
    }
}

/// <summary>
/// Exercises the ONNX-op shortcut wrappers on <see cref="Vector{T}"/> that
/// add a <c>.Vec()</c> reinterpret on top of the base <see cref="Tensor{T}"/>
/// implementation. Each call constructs an orphan graph node.
/// </summary>
[Module]
public partial class VectorOnnxOpsModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var v = Vector(1f, 2f, 3f, 4f);
        var iv = Vector(0L, 1L);
        var splits = Vector(2L, 2L);
        var sizes = Vector(8L);
        var scales = Vector(2f);
        var pad1 = Vector(1L);
        var pad2 = Vector(1L, 1L);
        var startV = Vector(0L);
        var endV = Vector(2L);
        var startS = Scalar(0L);
        var endS = Scalar(2L);
        var sFill = Scalar(0f);
        var iv1 = Vector(0L);
        var values = Vector(9f);
        var clipMin = Scalar(-10f);
        var clipMax = Scalar(10f);
        var condition = Vector(true, false, true, false);

        _ = v.Split(2);
        _ = v.Split(new long[] { 2L, 2L });
        _ = v.Split(splits, 2L);

        _ = v.Resize(sizes);
        _ = v.Rescale(scales);

        var v1 = Vector(42f);
        _ = v1.Squeeze();

        _ = v.Slice(startV, endV);
        _ = v.Slice(startS, endS);
        _ = v.Softmax();
        _ = v.GatherND(iv, batchDims: 0);

        _ = v.Cast<int32>();
        _ = v.Reduce(ReduceKind.Sum);
        _ = v.ReduceKeepDims(ReduceKind.Sum);
        _ = v.Tile(Vector(2L));

        _ = v.Min();
        _ = v.Max();
        _ = v.Floor();
        _ = v.Abs();
        _ = v.Acos();
        _ = v.Acosh();
        _ = v.Asin();
        _ = v.Asinh();
        _ = v.Atan();
        _ = v.Atanh();

        _ = v.ArgMaxReduce();
        _ = v.ArgMaxKeepdims();
        _ = v.ArgMinReduce();
        _ = v.ArgMinKeepdims();

        _ = v.Bernoulli();
        _ = v.Bernoulli<float32>();

        _ = v.Celu();
        _ = v.Ceiling();
        _ = v.Cos();
        _ = v.Cosh();
        _ = v.Sin();
        _ = v.Sinh();
        _ = v.Tan();
        _ = v.Tanh();
        _ = v.Pow(v);
        _ = v.Ln();
        _ = v.Sqrt();
        Vector<float32> vReciprocal = v.Reciprocal();   // Vector.Reciprocal() narrows to Vector<T>
        _ = vReciprocal;
        Vector<float32> vErf = v.Erf();                  // Vector.Erf() narrows to Vector<T>
        _ = vErf;
        _ = v.Sign();
        _ = v.Concat(v);

        // Pad — 4 overloads.
        _ = v.Pad(PadMode.Constant, startS, endS, sFill);
        _ = v.Pad(PadMode.Constant, pad1, pad1, sFill);
        _ = v.Pad(PadMode.Constant, pad1, pad1, sFill, axes: null);
        _ = v.Pad(PadMode.Constant, pad2, sFill);
        _ = v.Pad(PadMode.Constant, pad2);

        _ = v.Elu();
        _ = v.Gelu();
        _ = v.LeakyRelu();
        _ = v.Relu();
        _ = v.Selu();
        _ = v.Sigmoid();
        _ = v.TopK(2);
        _ = v.ScatterND(iv1, values);
        _ = v.Clip(clipMin, clipMax);
        _ = v.Compress(condition, axis: 0);
        _ = v.Exp();

        // Equals/GetHashCode (line 426-428).
        _ = v.Equals((object?)v);
        _ = v.GetHashCode();

        // IEnumerable<VectorExpressionHelper> contract (lines 133, 135-138).
        // The duck-typed foreach uses the public GetEnumerator; the explicit
        // IEnumerable.GetEnumerator requires going through the interface.
        foreach (var _vh in v) { }
        foreach (var _vh in (System.Collections.IEnumerable)v) { }

        return input * Scalar(1.0f);
    }
}

/// <summary>
/// Exercises the <see cref="VectorIndexerParam"/> implicit conversions, the
/// <see cref="VectorIndexerResult{T}"/> and <see cref="ScalarIndexerResult{T}"/>
/// implicit conversions and <c>Set</c> methods, and the <c>Vector&lt;T&gt;</c>
/// indexer overloads in <c>Core/Vector.Index.cs</c>.
/// </summary>
[Module]
public partial class VectorIndexerModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var v = Vector(1f, 2f, 3f, 4f);

        // VectorIndexerParam implicit operators — each builds a param struct.
        VectorIndexerParam p1 = 1..3;
        VectorIndexerParam p2 = (0..4, 1L);
        VectorIndexerParam p3 = (0L, 4L, 1L);
        VectorIndexerParam p4 = Vector(0L, 1L);
        VectorIndexerParam p5 = new long[] { 0L, 1L };
        VectorIndexerParam p6 = Range.All;

        // Vector<T> indexers: every overload.
        _ = v[p1]; _ = v[p2]; _ = v[p3]; _ = v[p4]; _ = v[p5];
        _ = v[1..3];                           // Range conversion
        _ = v[Vector(0L, 1L)];                 // Vector<int64>
        _ = v[new long[] { 0L }];              // long[]
        _ = v[Scalar(0L)];                     // Scalar<int64>
        _ = v[0L];                             // long
        _ = v[1];                              // int
        _ = v[Index.End];                      // Index

        // VectorIndexerResult conversions and Set.
        Vector<float32> sliced = v[1..3];                            // Slice path
        Vector<float32> indexGathered = v[Vector(0L, 1L)];           // GatherND path
        Vector<float32> fullRange = v[p6];                           // IsFullRange path

        _ = sliced; _ = indexGathered; _ = fullRange;

        // VectorIndexerResult.Set — IsFullRange + GatherND-index + no-step Slice + step Slice.
        _ = v[p6].Set(Vector(10f, 20f, 30f, 40f));
        _ = v[Vector(0L)].Set(Vector(99f));
        _ = v[1..3].Set(Vector(8f, 9f));
        _ = v[(0..4, 2L)].Set(Vector(8f, 9f));

        // ScalarIndexerResult conversion + Set + .T accessor.
        Scalar<float32> elemAtLong = v[0L];
        Scalar<float32> elemAtScalar = v[Scalar(0L)];
        _ = elemAtLong; _ = elemAtScalar;

        _ = v[Scalar(0L)].T;
        _ = v[1..3].T;

        _ = v[0L].Set(Scalar(99f));

        return input * Scalar(1.0f);
    }
}

/// <summary>
/// Exercises the per-DType dispatch arms of <see cref="Scalar{T}.Unit"/>.
/// </summary>
[Module]
public partial class ScalarUnitDispatchModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        _ = Scalar<bit>.Unit;
        _ = Scalar<int8>.Unit;
        _ = Scalar<int16>.Unit;
        _ = Scalar<int32>.Unit;
        _ = Scalar<int64>.Unit;
        _ = Scalar<uint8>.Unit;
        _ = Scalar<uint16>.Unit;
        _ = Scalar<uint32>.Unit;
        _ = Scalar<uint64>.Unit;
        _ = Scalar<bfloat16>.Unit;
        _ = Scalar<float16>.Unit;
        _ = Scalar<float32>.Unit;
        _ = Scalar<float64>.Unit;
        return input * Scalar(1.0f);
    }
}

/// <summary>
/// Exercises every operator overload on <see cref="Scalar{T}"/> —
/// <c>Scalar op Scalar</c> arithmetic/bitwise/shift/unary/comparison,
/// plus the full <see cref="PrimitiveParam"/> family
/// (<c>Scalar op PrimitiveParam</c> and <c>PrimitiveParam op Scalar</c>),
/// plus the ONNX-op shortcut wrappers and <c>Unsqueeze</c>.
/// </summary>
[Module]
public partial class ScalarOperatorsModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        // Scalar<float32> — arithmetic + comparison + PrimitiveParam variants.
        var s1 = Scalar(1f);
        var s2 = Scalar(2f);

        _ = s1 + s2; _ = s1 - s2; _ = s1 * s2; _ = s1 / s2; _ = s1 % s2;
        _ = -s1;
        _ = s1 >  s2; _ = s1 >= s2; _ = s1 <  s2; _ = s1 <= s2;
        _ = s1 == s2; _ = s1 != s2;

        _ = s1 + 1f; _ = 1f + s1;
        _ = s1 - 1f; _ = 1f - s1;
        _ = s1 * 1f; _ = 1f * s1;
        _ = s1 / 1f; _ = 1f / s1;
        _ = s1 % 1f; _ = 1f % s1;
        _ = s1 >  1f; _ = (PrimitiveParam)1f >  s1;
        _ = s1 >= 1f; _ = (PrimitiveParam)1f >= s1;
        _ = s1 <  1f; _ = (PrimitiveParam)1f <  s1;
        _ = s1 <= 1f; _ = (PrimitiveParam)1f <= s1;
        _ = s1 == 1f; _ = (PrimitiveParam)1f == s1;
        _ = s1 != 1f; _ = (PrimitiveParam)1f != s1;

        // Scalar<int64> — bitwise + shift + PrimitiveParam variants.
        var is1 = Scalar(1L);
        var is2 = Scalar(2L);
        _ = is1 ^ is2; _ = is1 & is2; _ = is1 | is2;
        _ = is1 << is2; _ = is1 >> is2;
        _ = !is1;

        _ = is1 ^ 1L; _ = (PrimitiveParam)1L ^ is1;
        _ = is1 & 1L; _ = (PrimitiveParam)1L & is1;
        _ = is1 | 1L; _ = (PrimitiveParam)1L | is1;
        _ = is1 << (PrimitiveParam)1L;
        _ = is1 >> (PrimitiveParam)1L;

        // PrimitiveParam -> Scalar<T> implicit conversion (Scalar.cs:68).
        // Casting via Tensor<T> + operator forces use of Scalar<T>(PrimitiveParam).
        _ = s1 + (PrimitiveParam)2.5f;

        // Scalar ONNX shortcuts.
        _ = s1.Cast<int32>();
        _ = s1.Min(s2);
        _ = s1.Max(s2);
        _ = s1.Floor();
        _ = s1.Abs();
        _ = s1.Acos();
        _ = s1.Acosh();
        _ = s1.Asin();
        _ = s1.Asinh();
        _ = s1.Atan();
        _ = s1.Atanh();
        _ = s1.Bernoulli();
        _ = s1.Bernoulli<float32>();
        _ = s1.Celu();
        _ = s1.Ceiling();
        _ = s1.Cos();
        _ = s1.Cosh();
        _ = s1.Sin();
        _ = s1.Sinh();
        _ = s1.Tan();
        _ = s1.Tanh();
        _ = s1.Pow(s2);
        _ = s1.Ln();
        _ = s1.Sqrt();
        Scalar<float32> sReciprocal = s1.Reciprocal();   // Scalar.Reciprocal() narrows to Scalar<T>
        _ = sReciprocal;
        Scalar<float32> sErf = s1.Erf();                 // Scalar.Erf() narrows to Scalar<T>
        _ = sErf;
        _ = s1.Sign();
        _ = s1.Elu();
        _ = s1.Gelu();
        _ = s1.LeakyRelu();
        _ = s1.Relu();
        _ = s1.Selu();
        _ = s1.Sigmoid();
        _ = s1.Clip(Scalar(-10f), Scalar(10f));
        _ = s1.Exp();
        _ = s1.Unsqueeze();
        // Scalar.Unsqueeze(long) narrows rank-0 -> rank-1: the typed target enforces that it
        // returns Vector<T> (not Tensor<T>), so this fails to compile if the override regresses.
        Vector<float32> unsqueezedAxis = s1.Unsqueeze(0L);
        _ = unsqueezedAxis;

        // Scalar.Equals/GetHashCode (Scalar.cs:71-75).
        _ = s1.Equals((object?)s2);
        _ = s1.GetHashCode();

        return input * Scalar(1.0f);
    }
}

/// <summary>
/// Self-checking module for the implicit <c>primitive -&gt; Scalar&lt;T&gt;</c> conversions
/// (Scalar.cs "Primitive value conversions" region). Verifies, at runtime, that a bare
/// primitive binds to a <see cref="Scalar{T}"/> parameter and that the element type is
/// taken from the contextually-required <c>T</c> (not the literal's C# type), converting
/// the value via <see cref="Globals.Scalar{T}(object)"/>.
///
/// <para>
/// <c>DoSomething(Scalar&lt;float32&gt;, Scalar&lt;bit&gt;)</c> below stands in for a
/// user-defined API: the intended call <c>DoSomething(1.1f, true)</c> compiles, and the
/// type-mismatched call <c>DoSomething(true, 1.1f)</c> also compiles as a best-effort
/// conversion (true -&gt; 1.0f, 1.1f -&gt; true) — the conversions are generic over T, so
/// they cannot be restricted to "matching" element types.
/// </para>
/// </summary>
[Module]
public partial class ScalarImplicitPrimitiveConversionModel
{
    private static void DoSomething(Scalar<float32> myParam, Scalar<bit> flag) { _ = myParam; _ = flag; }

    public static Scalar<bit> Inline()
    {
        // Intended call shape: each literal lands on the matching element type.
        Scalar<float32> floatToF32 = 1.1f;
        Scalar<bit> boolToBit = true;
        DoSomething(1.1f, true);

        // Mismatched call shape: still compiles, best-effort value conversion.
        Scalar<float32> boolToF32 = true;   // Convert.ToSingle(true)  == 1.0f
        Scalar<bit> floatToBit = 1.1f;      // Convert.ToBoolean(1.1f) == true
        DoSomething(true, 1.1f);

        // Context-driven targeting: the same literal 5 takes its element type from context.
        Scalar<int32> intTo32 = 5;
        Scalar<int64> intTo64 = 5;
        Scalar<float32> intToF32 = 5;

        // The remaining integer / unsigned / floating source types.
        Scalar<int64> longTo64 = 5L;
        Scalar<float64> doubleToF64 = 2.5;
        Scalar<int32> sbyteTo32 = (sbyte)7;
        Scalar<int32> shortTo32 = (short)9;
        Scalar<int64> byteTo64 = (byte)11;
        Scalar<int32> ushortTo32 = (ushort)13;
        Scalar<uint32> uintToU32 = 15u;
        Scalar<uint64> ulongToU64 = 17UL;

        // Half-precision sources: BFloat16 / Float16 -> Scalar<T>. Exercised at module-build
        // time and kept orphan (not folded into the returned bit) since not every backend
        // executes bf16/f16 equality. These let a raw half value stand in for a Scalar<T>
        // wherever one is expected — e.g. as a Vector<bfloat16> operand.
        Scalar<bfloat16> bf16FromHalf = (BFloat16)0.5f; _ = bf16FromHalf;
        Scalar<float16> f16FromHalf = (Float16)0.5f; _ = f16FromHalf;

        return
            (floatToF32 == Scalar(1.1f)) &
            (boolToBit == Scalar(true)) &
            (boolToF32 == Scalar(1.0f)) &
            (floatToBit == Scalar(true)) &
            (intTo32 == Scalar(5)) &
            (intTo64 == Scalar(5L)) &
            (intToF32 == Scalar(5.0f)) &
            (longTo64 == Scalar(5L)) &
            (doubleToF64 == Scalar(2.5)) &
            (sbyteTo32 == Scalar(7)) &
            (shortTo32 == Scalar(9)) &
            (byteTo64 == Scalar(11L)) &
            (ushortTo32 == Scalar(13)) &
            (uintToU32 == Scalar(15u)) &
            (ulongToU64 == Scalar(17UL));
    }
}
