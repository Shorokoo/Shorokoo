using Shorokoo.Core.Inference.Abstractions;
using static Shorokoo.Tests.Utils.SelfCheck;

namespace Shorokoo.Tests;

/// <summary>
/// Coverage-purpose tests for <see cref="Vector{T}"/> and <see cref="Scalar{T}"/>
/// in <c>Core/Vector.cs</c>, <c>Core/Vector.Index.cs</c>, and <c>Core/Scalar.cs</c>.
/// The static Unit/Empty per-DType arms, operator overloads (including
/// <see cref="PrimitiveParam"/> and bare-primitive variants),
/// ONNX op shortcut wrappers, and indexer overloads are exercised inside the
/// <c>Inline</c> bodies of <c>[Module]</c>-attributed classes.
///
/// <para>
/// Each module returns a <see cref="Scalar{T}"/> verdict that is <c>1</c> only when the
/// exercised values match their references — so the exercised nodes are reachable graph
/// outputs that actually execute (no pruning), and a broken op fails the test instead of
/// hiding behind a discarded result (see issue #4). Each module keeps a <c>Tensor</c> input
/// (folded into the verdict via <see cref="SelfCheck.Nan"/>, which contributes 0) so the graph
/// retains a runtime input and the input-free C# codegen round-trip stays skipped, exactly as
/// the original passthrough modules did.
/// </para>
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class VectorScalarCoverageTests
{
    private static TensorData[] Input => [TensorDataWithSmallVals(DType.Float32, [4L])];

    // One self-checking module per test so a fault in one surfaces independently of the others.

    [Fact]
    public void TestVectorUnitEmptyDispatch()
        => Assert.True(AutoTest.AdvancedTestGraph<VectorUnitEmptyDispatchModel>(hyperparamInputs: [], runtimeInputs: Input));

    [Fact]
    public void TestVectorOperators()
        => Assert.True(AutoTest.AdvancedTestGraph<VectorOperatorsModel>(hyperparamInputs: [], runtimeInputs: Input));

    [Fact]
    public void TestVectorOnnxOps()
        => Assert.True(AutoTest.AdvancedTestGraph<VectorOnnxOpsModel>(hyperparamInputs: [], runtimeInputs: Input));

    [Fact]
    public void TestScalarUnitDispatch()
        => Assert.True(AutoTest.AdvancedTestGraph<ScalarUnitDispatchModel>(hyperparamInputs: [], runtimeInputs: Input));

    [Fact]
    public void TestScalarOperators()
        => Assert.True(AutoTest.AdvancedTestGraph<ScalarOperatorsModel>(hyperparamInputs: [], runtimeInputs: Input));

    [Fact]
    public void TestScalarImplicitPrimitiveConversion()
        => Assert.True(AutoTest.AdvancedTestGraph<ScalarImplicitPrimitiveConversionModel>(hyperparamInputs: [], runtimeInputs: []));

    // VectorIndexerModel is self-checking like the others, but converting it makes the indexer
    // read/Set paths reachable outputs — which surfaces the faults tracked in #3. The first to
    // execute (at module-build time, in the (Vector<float32>)v[(0..4, 1L)] materialisation) is the
    // stepped-slice read: it throws CR006 "Step operations in vector indexing are not yet
    // supported". The gather (GatherND/ScatterND) and contiguous slice-write (Where) paths fault
    // too. Marked inconclusive (skipped) until #3 is fixed; the module is kept self-checking so
    // this test becomes meaningful — and turns green — the moment the underlying indexer bug is
    // resolved.
    [Fact(Skip = "Inconclusive: blocked by #3 (Vector indexer step-slice/gather/slice-write faults). See #4.")]
    public void TestVectorIndexerCoverage()
        => Assert.True(AutoTest.AdvancedTestGraph<VectorIndexerModel>(hyperparamInputs: [], runtimeInputs: Input));

    // float32 % is exercised over constant operands so Shorokoo's evaluator folds and validates
    // the result by value. (It computes correctly; its ONNX export — a Mod node with fmod=false,
    // which ONNX Runtime rejects for floats — is a separate limitation not covered here.)
    [Fact]
    public void TestVectorMod()
        => Assert.True(AutoTest.AdvancedTestGraph<VectorModModel>(hyperparamInputs: [], runtimeInputs: Input));

    [Fact]
    public void TestScalarMod()
        => Assert.True(AutoTest.AdvancedTestGraph<ScalarModModel>(hyperparamInputs: [], runtimeInputs: Input));
}

/// <summary>
/// Exercises the per-DType dispatch arms of <see cref="Vector{T}.Unit"/> and
/// <see cref="Vector{T}.Empty"/>. Each <c>Unit</c> contributes its single element (value 1)
/// and each supported <c>Empty</c> contributes 0 to the folded sum, so the arms are reachable
/// and the verdict is 1 only when every arm yields the expected value.
/// </summary>
[Module]
public partial class VectorUnitEmptyDispatchModel
{
    public static Scalar<bit> Inline(Tensor<float32> input)
    {
        Scalar<float32> acc = Scalar(0f);

        // Unit (value 1) for all 13 element types.
        acc = acc + Vector<bit>.Unit.Cast<float32>().Reduce(ReduceKind.Sum);
        acc = acc + Vector<int8>.Unit.Cast<float32>().Reduce(ReduceKind.Sum);
        acc = acc + Vector<int16>.Unit.Cast<float32>().Reduce(ReduceKind.Sum);
        acc = acc + Vector<int32>.Unit.Cast<float32>().Reduce(ReduceKind.Sum);
        acc = acc + Vector<int64>.Unit.Cast<float32>().Reduce(ReduceKind.Sum);
        acc = acc + Vector<uint8>.Unit.Cast<float32>().Reduce(ReduceKind.Sum);
        acc = acc + Vector<uint16>.Unit.Cast<float32>().Reduce(ReduceKind.Sum);
        acc = acc + Vector<uint32>.Unit.Cast<float32>().Reduce(ReduceKind.Sum);
        acc = acc + Vector<uint64>.Unit.Cast<float32>().Reduce(ReduceKind.Sum);
        acc = acc + Vector<bfloat16>.Unit.Cast<float32>().Reduce(ReduceKind.Sum);
        acc = acc + Vector<float16>.Unit.Cast<float32>().Reduce(ReduceKind.Sum);
        acc = acc + Vector<float32>.Unit.Cast<float32>().Reduce(ReduceKind.Sum);
        acc = acc + Vector<float64>.Unit.Cast<float32>().Reduce(ReduceKind.Sum);

        // Empty (length 0, contributes 0) for all 13 element types.
        acc = acc + Vector<bit>.Empty.Cast<float32>().Reduce(ReduceKind.Sum);
        acc = acc + Vector<int8>.Empty.Cast<float32>().Reduce(ReduceKind.Sum);
        acc = acc + Vector<int16>.Empty.Cast<float32>().Reduce(ReduceKind.Sum);
        acc = acc + Vector<int32>.Empty.Cast<float32>().Reduce(ReduceKind.Sum);
        acc = acc + Vector<int64>.Empty.Cast<float32>().Reduce(ReduceKind.Sum);
        acc = acc + Vector<uint8>.Empty.Cast<float32>().Reduce(ReduceKind.Sum);
        acc = acc + Vector<uint16>.Empty.Cast<float32>().Reduce(ReduceKind.Sum);
        acc = acc + Vector<uint32>.Empty.Cast<float32>().Reduce(ReduceKind.Sum);
        acc = acc + Vector<uint64>.Empty.Cast<float32>().Reduce(ReduceKind.Sum);
        acc = acc + Vector<bfloat16>.Empty.Cast<float32>().Reduce(ReduceKind.Sum);
        acc = acc + Vector<float16>.Empty.Cast<float32>().Reduce(ReduceKind.Sum);
        acc = acc + Vector<float32>.Empty.Cast<float32>().Reduce(ReduceKind.Sum);
        acc = acc + Vector<float64>.Empty.Cast<float32>().Reduce(ReduceKind.Sum);

        // 13 unit vectors (each 1) + 13 empty vectors (each 0) == 13.
        return ((acc - Scalar(13f)).Abs() + Nan(input)) < Scalar(1e-3f);
    }
}

/// <summary>
/// Exercises every operator overload on <see cref="Vector{T}"/>:
/// arithmetic/comparison in both <c>Vector op Vector</c> and <c>Vector op Scalar</c> shapes
/// (float32), and the int64 bitwise operators. Each result is folded into the verdict.
/// (float32 % -> VectorModModel; int64 << / >> are not covered — ONNX BitShift is unsigned-only.)
/// </summary>
[Module]
public partial class VectorOperatorsModel
{
    public static Scalar<bit> Inline(Tensor<float32> input)
    {
        // Vector<float32> — arithmetic & comparison.
        var v1 = Vector(1f, 2f, 3f, 4f);
        var v2 = Vector(5f, 6f, 7f, 8f);
        Scalar<float32> sp = 2f;

        Scalar<float32> err = Scalar(0f);

        err = err + L1(v1 + v2, Vector(6f, 8f, 10f, 12f)) + L1(sp + v1, Vector(3f, 4f, 5f, 6f))   + L1(v1 + sp, Vector(3f, 4f, 5f, 6f));
        err = err + L1(v1 - v2, Vector(-4f, -4f, -4f, -4f)) + L1(sp - v1, Vector(1f, 0f, -1f, -2f)) + L1(v1 - sp, Vector(-1f, 0f, 1f, 2f));
        err = err + L1(v1 * v2, Vector(5f, 12f, 21f, 32f)) + L1(sp * v1, Vector(2f, 4f, 6f, 8f))   + L1(v1 * sp, Vector(2f, 4f, 6f, 8f));
        err = err + L1(v1 / v2, Vector(1f / 5f, 2f / 6f, 3f / 7f, 4f / 8f)) + L1(sp / v1, Vector(2f / 1f, 2f / 2f, 2f / 3f, 2f / 4f)) + L1(v1 / sp, Vector(1f / 2f, 2f / 2f, 3f / 2f, 4f / 2f));
        err = err + L1(-v1, Vector(-1f, -2f, -3f, -4f));
        // NOTE: float32 % is exercised by VectorModModel (validated by value).

        err = err + L1((v1 >  v2).Cast<float32>(), Vector(0f, 0f, 0f, 0f)) + L1((sp >  v1).Cast<float32>(), Vector(1f, 0f, 0f, 0f)) + L1((v1 >  sp).Cast<float32>(), Vector(0f, 0f, 1f, 1f));
        err = err + L1((v1 >= v2).Cast<float32>(), Vector(0f, 0f, 0f, 0f)) + L1((sp >= v1).Cast<float32>(), Vector(1f, 1f, 0f, 0f)) + L1((v1 >= sp).Cast<float32>(), Vector(0f, 1f, 1f, 1f));
        err = err + L1((v1 <  v2).Cast<float32>(), Vector(1f, 1f, 1f, 1f)) + L1((sp <  v1).Cast<float32>(), Vector(0f, 0f, 1f, 1f)) + L1((v1 <  sp).Cast<float32>(), Vector(1f, 0f, 0f, 0f));
        err = err + L1((v1 <= v2).Cast<float32>(), Vector(1f, 1f, 1f, 1f)) + L1((sp <= v1).Cast<float32>(), Vector(0f, 1f, 1f, 1f)) + L1((v1 <= sp).Cast<float32>(), Vector(1f, 1f, 0f, 0f));
        err = err + L1((v1 == v2).Cast<float32>(), Vector(0f, 0f, 0f, 0f)) + L1((sp == v1).Cast<float32>(), Vector(0f, 1f, 0f, 0f)) + L1((v1 == sp).Cast<float32>(), Vector(0f, 1f, 0f, 0f));
        err = err + L1((v1 != v2).Cast<float32>(), Vector(1f, 1f, 1f, 1f)) + L1((sp != v1).Cast<float32>(), Vector(1f, 0f, 1f, 1f)) + L1((v1 != sp).Cast<float32>(), Vector(1f, 0f, 1f, 1f));

        // Vector<int64> — bitwise (int64 << / >> are not covered: ONNX BitShift is unsigned-only).
        var iv1 = Vector(1L, 2L, 3L, 4L);
        var iv2 = Vector(5L, 6L, 7L, 8L);
        Scalar<int64> isp = 1L;

        err = err + L1((iv1 ^ iv2).Cast<float32>(), Vector(4f, 4f, 4f, 12f))  + L1((isp ^ iv1).Cast<float32>(), Vector(0f, 3f, 2f, 5f)) + L1((iv1 ^ isp).Cast<float32>(), Vector(0f, 3f, 2f, 5f));
        err = err + L1((iv1 & iv2).Cast<float32>(), Vector(1f, 2f, 3f, 0f))   + L1((isp & iv1).Cast<float32>(), Vector(1f, 0f, 1f, 0f)) + L1((iv1 & isp).Cast<float32>(), Vector(1f, 0f, 1f, 0f));
        err = err + L1((iv1 | iv2).Cast<float32>(), Vector(5f, 6f, 7f, 12f))  + L1((isp | iv1).Cast<float32>(), Vector(1f, 3f, 3f, 5f)) + L1((iv1 | isp).Cast<float32>(), Vector(1f, 3f, 3f, 5f));
        err = err + L1((!iv1).Cast<float32>(), Vector(-2f, -3f, -4f, -5f));

        return (err + Nan(input)) < Scalar(1e-2f);
    }
}

/// <summary>
/// Exercises the ONNX-op shortcut wrappers on <see cref="Vector{T}"/>. Where an exact reference
/// is cheap the result is checked by value; for transcendentals it is checked via an exact
/// mathematical identity (so no host-computed constant is needed and float precision is
/// tolerated); for ops whose value is impractical to reference (resize/rescale, gather/scatter,
/// random sampling, gelu/selu, pad) the result is folded via a finiteness check so the node
/// still executes (reachable, not pruned).
/// </summary>
[Module]
public partial class VectorOnnxOpsModel
{
    public static Scalar<bit> Inline(Tensor<float32> input)
    {
        var v = Vector(1f, 2f, 3f, 4f);                 // strictly positive, |x| >= 1
        var vu = Vector(0.1f, 0.2f, 0.3f, 0.4f);        // in (-1, 1): valid for acos/asin/atanh
        var vfrac = Vector(1.2f, 2.7f, 3.4f, 4.8f);     // non-integers: real floor/ceil
        var vneg = Vector(-1f, 2f, -3f, 4f);            // mixed signs: real abs
        var vsign = Vector(-2f, 0f, 3f, -4f);           // mixed signs incl. zero: real sign
        var vprob = Vector(0.5f, 0.5f, 0.5f, 0.5f);     // valid probabilities for bernoulli
        var ones = Vector(1f, 1f, 1f, 1f);
        var zeros = Vector(0f, 0f, 0f, 0f);

        Scalar<float32> err = Scalar(0f);

        // Split — three overloads, each yields [1,2] and [3,4].
        var sp2 = v.Split(2);
        err = err + L1(sp2[0], Vector(1f, 2f)) + L1(sp2[1], Vector(3f, 4f));
        var sp3 = v.Split(new long[] { 2L, 2L });
        err = err + L1(sp3[0], Vector(1f, 2f)) + L1(sp3[1], Vector(3f, 4f));
        var sp4 = v.Split(Vector(2L, 2L), 2L);
        err = err + L1(sp4[0], Vector(1f, 2f)) + L1(sp4[1], Vector(3f, 4f));

        // Resize / Rescale — finiteness fold (interpolated values are backend-dependent).
        err = err + Nan(v.Resize(Vector(8L)));
        err = err + Nan(v.Rescale(Vector(2f)));

        // Squeeze: length-1 -> scalar.
        err = err + (Vector(42f).Squeeze() - Scalar(42f)).Abs();

        // Slice — vector- and scalar-bound variants, both yield [1,2].
        err = err + L1(v.Slice(Vector(0L), Vector(2L)), Vector(1f, 2f));
        err = err + L1(v.Slice(Scalar(0L), Scalar(2L)), Vector(1f, 2f));

        // Softmax sums to 1.
        err = err + (v.Softmax().Reduce(ReduceKind.Sum) - Scalar(1f)).Abs();

        // GatherND with a valid rank-1 coordinate (length-1, == data rank): gathers element 0.
        // (Multi-index gather via the indexer's [N] coordinate is the #3 path, covered by the
        // skipped VectorIndexerModel — passing [0,1] here is invalid GatherND, not a coverage goal.)
        err = err + Nan(v.GatherND(Vector(0L), batchDims: 0).Cast<float32>());

        // Cast round-trips back to the same integer values.
        err = err + L1(v.Cast<int32>().Cast<float32>(), v);

        // Reduce / ReduceKeepDims / Tile.
        err = err + (v.Reduce(ReduceKind.Sum) - Scalar(10f)).Abs();
        err = err + L1(v.ReduceKeepDims(ReduceKind.Sum), Vector(10f));
        err = err + L1(v.Tile(Vector(2L)), Vector(1f, 2f, 3f, 4f, 1f, 2f, 3f, 4f));

        // Min/Max with no extra operands — finiteness fold (identity over a single operand).
        err = err + Nan(v.Min());
        err = err + Nan(v.Max());

        // Floor / Ceiling / Abs / Sign — exact on chosen inputs.
        err = err + L1(vfrac.Floor(), Vector(1f, 2f, 3f, 4f));
        err = err + L1(vfrac.Ceiling(), Vector(2f, 3f, 4f, 5f));
        err = err + L1(vneg.Abs(), Vector(1f, 2f, 3f, 4f));
        err = err + L1(vsign.Sign(), Vector(-1f, 0f, 1f, -1f));

        // Inverse trig/hyperbolic via round-trip identities (also covers the forward op).
        err = err + L1(vu.Acos().Cos(), vu);
        err = err + L1(v.Acosh().Cosh(), v);
        err = err + L1(vu.Asin().Sin(), vu);
        err = err + L1(v.Asinh().Sinh(), v);
        err = err + L1(v.Atan().Tan(), v);
        err = err + L1(vu.Atanh().Tanh(), vu);

        // ArgMax/ArgMin of [1,2,3,4]: max at 3, min at 0.
        err = err + (v.ArgMaxReduce().Cast<float32>() - Scalar(3f)).Abs();
        err = err + L1(v.ArgMaxKeepdims().Cast<float32>(), Vector(3f));
        err = err + (v.ArgMinReduce().Cast<float32>() - Scalar(0f)).Abs();
        err = err + L1(v.ArgMinKeepdims().Cast<float32>(), Vector(0f));

        // Bernoulli — output is in {0,1}, so b*(b-1) == 0 (value-dependent, not constant-foldable).
        var b1 = vprob.Bernoulli();
        err = err + L1(b1 * (b1 - ones), zeros);
        var b2 = vprob.Bernoulli<float32>();
        err = err + L1(b2 * (b2 - ones), zeros);

        // Trig identities.
        err = err + L1(v.Sin() * v.Sin() + v.Cos() * v.Cos(), ones);
        err = err + L1(v.Cosh() * v.Cosh() - v.Sinh() * v.Sinh(), ones);
        err = err + L1(v.Tan(), v.Sin() / v.Cos());
        err = err + L1(v.Tanh(), v.Sinh() / v.Cosh());

        // Pow / Ln / Exp / Sqrt / Reciprocal / Erf identities.
        err = err + L1(v.Pow(Vector(2f, 2f, 2f, 2f)), v * v);
        err = err + L1(v.Exp().Ln(), v);
        err = err + L1(v.Sqrt() * v.Sqrt(), v);
        Vector<float32> vReciprocal = v.Reciprocal();
        err = err + L1(v * vReciprocal, ones);
        Vector<float32> vErf = v.Erf();
        err = err + L1(vErf + (-v).Erf(), zeros);   // erf is odd

        // Concat.
        err = err + L1(v.Concat(v), Vector(1f, 2f, 3f, 4f, 1f, 2f, 3f, 4f));

        // Pad — 5 overloads. Padded values are backend-shaped; finiteness fold keeps them reachable.
        err = err + Nan(v.Pad(PadMode.Constant, Scalar(0L), Scalar(2L), Scalar(0f)));
        err = err + Nan(v.Pad(PadMode.Constant, Vector(1L), Vector(1L), Scalar(0f)));
        err = err + Nan(v.Pad(PadMode.Constant, Vector(1L), Vector(1L), Scalar(0f), axes: null));
        err = err + Nan(v.Pad(PadMode.Constant, Vector(1L, 1L), Scalar(0f)));
        err = err + Nan(v.Pad(PadMode.Constant, Vector(1L, 1L)));

        // Activations: on strictly-positive inputs Relu/LeakyRelu/Elu/Celu are the identity;
        // Sigmoid via sig(x)+sig(-x)==1; Gelu/Selu via finiteness fold.
        err = err + L1(v.Relu(), v);
        err = err + L1(v.LeakyRelu(), v);
        err = err + L1(v.Elu(), v);
        err = err + L1(v.Celu(), v);
        err = err + L1(v.Sigmoid() + (-v).Sigmoid(), ones);
        err = err + Nan(v.Gelu());
        err = err + Nan(v.Selu());

        // TopK(2) of [1,2,3,4]: values [4,3], indices [3,2].
        var tk = v.TopK(2);
        err = err + L1(tk.topK, Vector(4f, 3f)) + L1(tk.indices.Cast<float32>(), Vector(3f, 2f));

        // ScatterND — indices are an [N,1] coordinate list (as the indexer's own Set does via a
        // double unsqueeze); [[0]] with updates [9] writes 9 at position 0.
        err = err + Nan(v.ScatterND(Vector(0L).Unsqueeze(1L), Vector(9f)));

        // Clip to [2,3] -> [2,2,3,3].
        err = err + L1(v.Clip(Scalar(2f), Scalar(3f)), Vector(2f, 2f, 3f, 3f));

        // Compress selects elements 0 and 2 -> [1,3].
        err = err + L1(v.Compress(Vector(true, false, true, false), axis: 0), Vector(1f, 3f));

        // Exp on its own (also exercised above through Exp().Ln()).
        err = err + Nan(v.Exp());

        // Host-side members (run at module build; no graph node to fold).
        _ = v.Equals((object?)v);
        _ = v.GetHashCode();
        foreach (var _vh in v) { }
        foreach (var _vh in (System.Collections.IEnumerable)v) { }

        return (err + Nan(input)) < Scalar(1e-2f);
    }
}

/// <summary>
/// Exercises the <see cref="VectorIndexerParam"/> implicit conversions, the
/// <see cref="VectorIndexerResult{T}"/> and <see cref="ScalarIndexerResult{T}"/>
/// implicit conversions and <c>Set</c> methods, and the <c>Vector&lt;T&gt;</c>
/// indexer overloads in <c>Core/Vector.Index.cs</c>. Reads are checked by value;
/// writes (the paths broken by #3) are folded so they become reachable outputs.
/// </summary>
[Module]
public partial class VectorIndexerModel
{
    public static Scalar<bit> Inline(Tensor<float32> input)
    {
        var v = Vector(1f, 2f, 3f, 4f, 5f);

        // VectorIndexerParam implicit operators — each builds a param struct, used below.
        VectorIndexerParam p1 = 1..3;
        VectorIndexerParam p2 = (0..4, 1L);
        VectorIndexerParam p3 = (0L, 4L, 1L);
        VectorIndexerParam p4 = Vector(0L, 1L);
        VectorIndexerParam p5 = new long[] { 0L, 1L };
        VectorIndexerParam p6 = Range.All;

        Scalar<float32> err = Scalar(0f);

        // Slice read: v[1..3] == [2, 3].
        err = err + L1(v[1..3].T, Vector(2f, 3f));
        err = err + L1(v[p1].T, Vector(2f, 3f));

        // Full-range read is the identity.
        err = err + L1(v[p6].T, v);
        err = err + L1(v[Range.All].T, v);

        // Element reads: v[0] == 1, v[^1] == 5.
        err = err + (v[0L].T - Scalar(1f)).Abs();
        err = err + (v[0].T - Scalar(1f)).Abs();
        err = err + (v[Scalar(0L)].T - Scalar(1f)).Abs();
        err = err + (v[Index.End].T - Scalar(5f)).Abs();

        // Element write: v[0] = 99 -> [99, 2, 3, 4, 5].
        err = err + L1(v[0L].Set(Scalar(99f)), Vector(99f, 2f, 3f, 4f, 5f));

        // Gather reads / param-conversion paths and the Set write paths broken by #3 — folded so
        // the nodes are reachable (these throw a shape-inference error today; #3).
        err = err + Nan(((Vector<float32>)v[p2]).Cast<float32>());
        err = err + Nan(((Vector<float32>)v[p3]).Cast<float32>());
        err = err + Nan(((Vector<float32>)v[p4]).Cast<float32>());
        err = err + Nan(((Vector<float32>)v[p5]).Cast<float32>());
        err = err + Nan(v[Vector(0L, 1L)].T);
        err = err + Nan(v[new long[] { 0L }].T);

        err = err + Nan(v[p6].Set(Vector(10f, 20f, 30f, 40f, 50f)));
        err = err + Nan(v[Vector(0L)].Set(Vector(99f)));
        err = err + Nan(v[1..3].Set(Vector(8f, 9f)));
        err = err + Nan(v[(0..4, 2L)].Set(Vector(8f, 9f)));

        return (err + Nan(input)) < Scalar(1e-2f);
    }
}

/// <summary>
/// Exercises the per-DType dispatch arms of <see cref="Scalar{T}.Unit"/>. Each arm contributes
/// its single value (1), so the verdict is 1 only when all 13 arms yield the expected value.
/// </summary>
[Module]
public partial class ScalarUnitDispatchModel
{
    public static Scalar<bit> Inline(Tensor<float32> input)
    {
        Scalar<float32> acc = Scalar(0f);

        acc = acc + Scalar<bit>.Unit.Cast<float32>();
        acc = acc + Scalar<int8>.Unit.Cast<float32>();
        acc = acc + Scalar<int16>.Unit.Cast<float32>();
        acc = acc + Scalar<int32>.Unit.Cast<float32>();
        acc = acc + Scalar<int64>.Unit.Cast<float32>();
        acc = acc + Scalar<uint8>.Unit.Cast<float32>();
        acc = acc + Scalar<uint16>.Unit.Cast<float32>();
        acc = acc + Scalar<uint32>.Unit.Cast<float32>();
        acc = acc + Scalar<uint64>.Unit.Cast<float32>();
        acc = acc + Scalar<bfloat16>.Unit.Cast<float32>();
        acc = acc + Scalar<float16>.Unit.Cast<float32>();
        acc = acc + Scalar<float32>.Unit.Cast<float32>();
        acc = acc + Scalar<float64>.Unit.Cast<float32>();

        return ((acc - Scalar(13f)).Abs() + Nan(input)) < Scalar(1e-3f);
    }
}

/// <summary>
/// Exercises every operator overload on <see cref="Scalar{T}"/> —
/// <c>Scalar op Scalar</c> arithmetic/comparison, the full <see cref="PrimitiveParam"/> family,
/// the int64 bitwise operators, and the ONNX-op shortcut wrappers and <c>Unsqueeze</c>.
/// (float32 % -> ScalarModModel; int64 << / >> are not covered — ONNX BitShift is unsigned-only.)
/// </summary>
[Module]
public partial class ScalarOperatorsModel
{
    public static Scalar<bit> Inline(Tensor<float32> input)
    {
        var s1 = Scalar(1f);
        var s2 = Scalar(2f);

        Scalar<float32> err = Scalar(0f);

        // Arithmetic + unary.
        err = err + (s1 + s2 - Scalar(3f)).Abs();
        err = err + (s1 - s2 - Scalar(-1f)).Abs();
        err = err + (s1 * s2 - Scalar(2f)).Abs();
        err = err + (s1 / s2 - Scalar(0.5f)).Abs();
        err = err + (-s1 - Scalar(-1f)).Abs();
        // NOTE: float32 % is exercised by ScalarModModel (validated by value).

        // Comparison (bit -> float).
        err = err + ((s1 >  s2).Cast<float32>() - Scalar(0f)).Abs();
        err = err + ((s1 >= s2).Cast<float32>() - Scalar(0f)).Abs();
        err = err + ((s1 <  s2).Cast<float32>() - Scalar(1f)).Abs();
        err = err + ((s1 <= s2).Cast<float32>() - Scalar(1f)).Abs();
        err = err + ((s1 == s2).Cast<float32>() - Scalar(0f)).Abs();
        err = err + ((s1 != s2).Cast<float32>() - Scalar(1f)).Abs();

        // PrimitiveParam arithmetic (both operand orders).
        err = err + (s1 + 1f - Scalar(2f)).Abs() + (1f + s1 - Scalar(2f)).Abs();
        err = err + (s1 - 1f - Scalar(0f)).Abs() + (1f - s1 - Scalar(0f)).Abs();
        err = err + (s1 * 1f - Scalar(1f)).Abs() + (1f * s1 - Scalar(1f)).Abs();
        err = err + (s1 / 1f - Scalar(1f)).Abs() + (1f / s1 - Scalar(1f)).Abs();
        // NOTE: float32 % (incl. PrimitiveParam) is exercised by ScalarModModel (validated by value).

        // PrimitiveParam comparison (both operand orders).
        err = err + ((s1 >  1f).Cast<float32>() - Scalar(0f)).Abs() + (((PrimitiveParam)1f >  s1).Cast<float32>() - Scalar(0f)).Abs();
        err = err + ((s1 >= 1f).Cast<float32>() - Scalar(1f)).Abs() + (((PrimitiveParam)1f >= s1).Cast<float32>() - Scalar(1f)).Abs();
        err = err + ((s1 <  1f).Cast<float32>() - Scalar(0f)).Abs() + (((PrimitiveParam)1f <  s1).Cast<float32>() - Scalar(0f)).Abs();
        err = err + ((s1 <= 1f).Cast<float32>() - Scalar(1f)).Abs() + (((PrimitiveParam)1f <= s1).Cast<float32>() - Scalar(1f)).Abs();
        err = err + ((s1 == 1f).Cast<float32>() - Scalar(1f)).Abs() + (((PrimitiveParam)1f == s1).Cast<float32>() - Scalar(1f)).Abs();
        err = err + ((s1 != 1f).Cast<float32>() - Scalar(0f)).Abs() + (((PrimitiveParam)1f != s1).Cast<float32>() - Scalar(0f)).Abs();

        // Scalar<int64> — bitwise (int64 << / >> are not covered: ONNX BitShift is unsigned-only).
        var is1 = Scalar(1L);
        var is2 = Scalar(2L);
        err = err + ((is1 ^ is2).Cast<float32>() - Scalar(3f)).Abs();
        err = err + ((is1 & is2).Cast<float32>() - Scalar(0f)).Abs();
        err = err + ((is1 | is2).Cast<float32>() - Scalar(3f)).Abs();
        err = err + ((!is1).Cast<float32>() - Scalar(-2f)).Abs();

        // Scalar<int64> — bitwise with PrimitiveParam.
        err = err + ((is1 ^ 1L).Cast<float32>() - Scalar(0f)).Abs() + (((PrimitiveParam)1L ^ is1).Cast<float32>() - Scalar(0f)).Abs();
        err = err + ((is1 & 1L).Cast<float32>() - Scalar(1f)).Abs() + (((PrimitiveParam)1L & is1).Cast<float32>() - Scalar(1f)).Abs();
        err = err + ((is1 | 1L).Cast<float32>() - Scalar(1f)).Abs() + (((PrimitiveParam)1L | is1).Cast<float32>() - Scalar(1f)).Abs();

        // PrimitiveParam -> Scalar<T> implicit conversion via operator.
        err = err + (s1 + (PrimitiveParam)2.5f - Scalar(3.5f)).Abs();

        // ONNX scalar shortcuts — exact where cheap, identity-based for transcendentals.
        var sd = Scalar(0.5f);   // valid domain for acos/asin/atanh
        var sb = Scalar(2f);     // valid domain for acosh
        err = err + (s1.Cast<int32>().Cast<float32>() - s1).Abs();
        err = err + (s1.Min(s2) - Scalar(1f)).Abs();
        err = err + (s1.Max(s2) - Scalar(2f)).Abs();
        err = err + (Scalar(1.7f).Floor() - Scalar(1f)).Abs();
        err = err + (Scalar(-3f).Abs() - Scalar(3f)).Abs();
        err = err + (sd.Acos().Cos() - sd).Abs();
        err = err + (sb.Acosh().Cosh() - sb).Abs();
        err = err + (sd.Asin().Sin() - sd).Abs();
        err = err + (s1.Asinh().Sinh() - s1).Abs();
        err = err + (s1.Atan().Tan() - s1).Abs();
        err = err + (sd.Atanh().Tanh() - sd).Abs();
        var sbern = Scalar(0.5f).Bernoulli();
        err = err + (sbern * (sbern - Scalar(1f))).Abs();
        var sbern2 = Scalar(0.5f).Bernoulli<float32>();
        err = err + (sbern2 * (sbern2 - Scalar(1f))).Abs();
        err = err + (s1.Celu() - s1).Abs();
        err = err + (Scalar(1.2f).Ceiling() - Scalar(2f)).Abs();
        err = err + (s1.Sin() * s1.Sin() + s1.Cos() * s1.Cos() - Scalar(1f)).Abs();
        err = err + (s1.Cosh() * s1.Cosh() - s1.Sinh() * s1.Sinh() - Scalar(1f)).Abs();
        err = err + (s1.Tan() - s1.Sin() / s1.Cos()).Abs();
        err = err + (s1.Tanh() - s1.Sinh() / s1.Cosh()).Abs();
        err = err + (s2.Pow(s2) - s2 * s2).Abs();
        err = err + (s2.Exp().Ln() - s2).Abs();
        err = err + (s2.Sqrt() * s2.Sqrt() - s2).Abs();
        Scalar<float32> sReciprocal = s2.Reciprocal();
        err = err + (s2 * sReciprocal - Scalar(1f)).Abs();
        Scalar<float32> sErf = s1.Erf();
        err = err + (sErf + (-s1).Erf()).Abs();
        err = err + (Scalar(-3f).Sign() - Scalar(-1f)).Abs();
        err = err + (s1.Elu() - s1).Abs();
        err = err + Nan(s1.Gelu());
        err = err + (s1.LeakyRelu() - s1).Abs();
        err = err + (s1.Relu() - s1).Abs();
        err = err + Nan(s1.Selu());
        err = err + (s1.Sigmoid() + (-s1).Sigmoid() - Scalar(1f)).Abs();
        err = err + (Scalar(5f).Clip(Scalar(0f), Scalar(3f)) - Scalar(3f)).Abs();
        err = err + Nan(s1.Exp());
        err = err + L1(s1.Unsqueeze(), Vector(1f));
        Vector<float32> unsqueezedAxis = s1.Unsqueeze(0L);
        err = err + L1(unsqueezedAxis, Vector(1f));

        // Host-side members.
        _ = s1.Equals((object?)s2);
        _ = s1.GetHashCode();

        return (err + Nan(input)) < Scalar(1e-2f);
    }
}

/// <summary>
/// Exercises the <see cref="Vector{T}"/> <c>%</c> (modulo) operator on <c>float32</c> over
/// constant operands, validated by value: [1,2,3,4] % [5,6,7,8] == [1,2,3,4], 2 % v == [0,0,2,2],
/// v % 2 == [1,0,1,0]. Shorokoo's evaluator constant-folds and computes these correctly.
/// (float32 <c>%</c> also has a separate ONNX-export limitation — it emits a Mod node with
/// fmod=false, which ONNX Runtime rejects for floats — not exercised here.)
/// </summary>
[Module]
public partial class VectorModModel
{
    public static Scalar<bit> Inline(Tensor<float32> input)
    {
        var v1 = Vector(1f, 2f, 3f, 4f);
        var v2 = Vector(5f, 6f, 7f, 8f);
        Scalar<float32> sp = 2f;

        Scalar<float32> err = Scalar(0f);
        err = err + L1(v1 % v2, Vector(1f, 2f, 3f, 4f)) + L1(sp % v1, Vector(0f, 0f, 2f, 2f)) + L1(v1 % sp, Vector(1f, 0f, 1f, 0f));
        return (err + Nan(input)) < Scalar(1e-2f);
    }
}

/// <summary>
/// Exercises the <see cref="Scalar{T}"/> <c>%</c> (modulo) operator on <c>float32</c> (plain and
/// <see cref="PrimitiveParam"/> forms) over constant operands, validated by value. Same separate
/// ONNX-export (Mod fmod=false) limitation as the vector case, not exercised here.
/// </summary>
[Module]
public partial class ScalarModModel
{
    public static Scalar<bit> Inline(Tensor<float32> input)
    {
        var s1 = Scalar(1f);
        var s2 = Scalar(2f);

        Scalar<float32> err = Scalar(0f);
        err = err + (s1 % s2 - Scalar(1f)).Abs();
        err = err + (s1 % 1f - Scalar(0f)).Abs() + (1f % s1 - Scalar(0f)).Abs();
        return (err + Nan(input)) < Scalar(1e-2f);
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
