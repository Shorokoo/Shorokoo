using System.Reflection;
using Shorokoo.Runtime;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Inference;
using Shorokoo.Core.Inference.Helpers;
using static Shorokoo.Tests.FoldForcing;

namespace Shorokoo.Tests;

/// <summary>Runtime-valued operands that force an otherwise-constant sub-chain through the host
/// folder — FastFoldConstants only materializes a constant that a non-constant node consumes.</summary>
public static class FoldForcing
{
    public static Tensor<int32> Runtime32(Tensor<float32> x)
        => OnnxOp.Range(Scalar(0L), x.ShapeTensor().Reduce(ReduceKind.Prod), Scalar(1L)).int64().Cast<int32>();
    public static Tensor<uint32> RuntimeU32(Tensor<float32> x)
        => OnnxOp.Range(Scalar(0L), x.ShapeTensor().Reduce(ReduceKind.Prod), Scalar(1L)).int64().Cast<uint32>();
}

/// <summary>
/// QuickExecutionEngine op-handler paths that no <c>Qee*Audit</c> module reaches — everything
/// else these tests once covered is subsumed by the self-checking audit suites, which drive
/// the same handlers with in-graph value/shape assertions.
/// </summary>
[Trait("Domain", "Inference")]
[Trait("Purpose", "Coverage")]
public class QeeOpsCoverageTests
{
    /// <summary>Lowers a module exactly like AdvancedTestGraph but runs only the
    /// QuickExecutionEngine validation pass — for graphs holding Shorokoo-internal ops that
    /// have no ONNX op-set registration.</summary>
    private static bool QeeOnly<TModule>(TensorData[] runtimeInputs)
    {
        var prop = typeof(TModule).GetProperty("ComputationGraph", BindingFlags.Public | BindingFlags.Static)
            ?? throw new InvalidOperationException($"{typeof(TModule).FullName} has no public static ComputationGraph property");
        var moduleGraph = ((ComputationGraph)prop.GetValue(null)!).ToInternal();
        var concreteArch = moduleGraph.ToConcreteArchitecture(moduleGraph.FromOrderedInputs([.. runtimeInputs]));
        var concreteModel = concreteArch.ToConcreteModel();
        var qee = new QuickExecutionEngine();
        var store = runtimeInputs.Length == 0 ? qee.Run(concreteModel) : qee.Run(concreteModel, runtimeInputs);
        foreach (var outKey in concreteModel.Outputs)
            if (!store.TryGetValue(outKey, out var rt) || rt.DType == DType.Invalid) return false;
        return true;
    }

    [Fact]
    public void TestQeeIntegerArithmeticAndBitwiseCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<QeeIntArithmeticOpsCheck>(
            hyperparamInputs: [], runtimeInputs: [
                TensorData(DType.Int64, [3L], -2L, 0L, 5L),
                TensorData(DType.Int64, [3L], 2L, 3L, 4L),
                TensorData(DType.Int64, [3L], 1L, 2L, 3L)]));
        Assert.True(AutoTest.AdvancedTestGraph<QeeBitwiseNotInt64Check>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Int64, [3L], 0b1100L, 0b1010L, 0b1111L)]));
    }

    [Fact]
    public void TestQeeConstantEyeLikeAndCastToBoolCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<QeeConstantOpsCheck>(hyperparamInputs: [], runtimeInputs: []));
        Assert.True(AutoTest.AdvancedTestGraph<QeeEyeLikeOpsCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorDataWithDefaultVals(DType.Float32, [3L, 3L])]));
        Assert.True(AutoTest.AdvancedTestGraph<QeeCastToBoolOpsCheck>(
            hyperparamInputs: [], runtimeInputs: [
                TensorData(DType.Int64, [3L], 0L, 1L, 2L),
                TensorData(DType.Bool, [3L], true, false, true)]));
    }

    [Fact]
    public void TestQeeEinsumTraceSizeAndMultinomialCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<QeeEinsumTraceCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [4L, 4L],
                1f, 0f, 0f, 0f, 0f, 2f, 0f, 0f, 0f, 0f, 3f, 0f, 0f, 0f, 0f, 4f)]));
        Assert.True(AutoTest.AdvancedTestGraph<QeeSizeCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [4L], -1.5f, 0.4f, 1.6f, 2.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<QeeMultinomialCheck>(
            hyperparamInputs: [], runtimeInputs: [
                TensorData(DType.Float32, [2L, 3L], 0.1f, 0.4f, 0.5f, 0.3f, 0.3f, 0.4f)],
            testCsRoundtrip: false));
    }

    [Fact]
    public void TestQeeDtypeIdentityOpsCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<QeeDtypeIdentitySignedOpsCheck>(
            hyperparamInputs: [], runtimeInputs: [
                TensorData(DType.Float64, [3L], 1.0, 2.0, 3.0),
                TensorData(DType.Int32, [3L], 1, 2, 3),
                TensorData(DType.Int16, [3L], (short)1, (short)2, (short)3),
                TensorData(DType.Int8, [3L], (sbyte)1, (sbyte)2, (sbyte)3)]));
        Assert.True(AutoTest.AdvancedTestGraph<QeeDtypeIdentityUnsignedOpsCheck>(
            hyperparamInputs: [], runtimeInputs: [
                TensorData(DType.UInt8, [3L], (byte)1, (byte)2, (byte)3),
                TensorData(DType.UInt16, [3L], (ushort)1, (ushort)2, (ushort)3),
                TensorData(DType.UInt32, [3L], (uint)1, (uint)2, (uint)3),
                TensorData(DType.UInt64, [3L], (ulong)1, (ulong)2, (ulong)3),
                TensorData(DType.Bool, [3L], true, false, true)]));
    }

    [Fact]
    public void TestQeeInternalSequenceOpsCoverage() =>
        Assert.True(QeeOnly<QeeInternalSequenceOpsCheck>([
            TensorData(DType.Float32, [], 1f),
            TensorData(DType.Float32, [], 2f),
            TensorData(DType.Float32, [], 3f)]));
}

/// <summary>A <c>uint32</c> constant that has to survive host-side constant folding as a
/// <c>uint32</c>: the folded value feeds an <c>Add</c> whose other operand is runtime-valued,
/// so the Add's type constraint sees the folded constant's dtype directly.</summary>
[Module]
public partial class QeeFoldedUnsignedConstant
{
    public static Tensor<uint32> Inline(Tensor<float32> x)
    {
        var runtime = OnnxOp.Range(Scalar(0L), x.ShapeTensor().Reduce(ReduceKind.Prod), Scalar(1L))
            .int64().Cast<uint32>();
        return runtime + Scalar(7UL).Cast<uint32>();
    }
}

/// <summary>A <c>uint32</c> chain that host-constant-folds only in part: a fully-constant
/// Threefry bijection keys a second bijection over a runtime-shaped counter, so the folded
/// values re-enter the graph and a later right shift exposes any bits above the declared
/// 32-bit width.</summary>
[Module]
public partial class QeePartiallyFoldedUInt32Chain
{
    public static Tensor<uint32> Inline(Tensor<float32> x)
    {
        var (a0, a1) = Shorokoo.Core.Rng.RuntimeRng.Bijection(
            Scalar(0u), Scalar(0u), Scalar(123u), Scalar(456u));
        var c0 = OnnxOp.Range(Scalar(0L), x.ShapeTensor().Reduce(ReduceKind.Prod), Scalar(1L))
            .int64().Cast<uint32>();
        var (b0, _) = Shorokoo.Core.Rng.RuntimeRng.Bijection(c0, Scalar(0u), a0, a1);
        return b0;
    }
}

/// <summary>A uint32 add that overflows, then a right shift of the sum — added to a
/// runtime-valued tensor so the constant sub-chain is actually forced through the host folder
/// (FastFoldConstants only materializes a constant a non-constant node consumes).</summary>
[Module]
public partial class QeeOverflowThenShift
{
    public static Tensor<uint32> Inline(Tensor<float32> x)
    {
        var folded = OnnxOp.BitShift(Scalar(4_294_967_295u) + Scalar(1u), Scalar(4u),
            BitShiftDirection.Right).uint32();
        var runtime = OnnxOp.Range(Scalar(0L), x.ShapeTensor().Reduce(ReduceKind.Prod), Scalar(1L))
            .int64().Cast<uint32>();
        return runtime + folded;
    }
}

/// <summary>A constant float64 sub-chain feeding a runtime-valued float64 tensor. QEE's float
/// buffer is float32, so it must decline to fold rather than retype or round the value.</summary>
[Module]
public partial class QeeFoldedFloat64Constant
{
    public static Tensor<float64> Inline(Tensor<float32> x)
    {
        var runtime = x.Reshape([Scalar(-1L)]).Cast<float64>();
        return runtime + (Scalar(3.0) * Scalar(2.0));
    }
}

/// <summary>Same shape, but with a constant beyond float32's range: 1e300 survives only if the
/// chain is never routed through QEE's float32 buffer.</summary>
[Module]
public partial class QeeWideFloat64Constant
{
    public static Tensor<float64> Inline(Tensor<float32> x)
    {
        var runtime = x.Reshape([Scalar(-1L)]).Cast<float64>() * Scalar(0.0);
        return runtime + (Scalar(1e300) * Scalar(1.0));
    }
}

/// <summary>
/// The QuickExecutionEngine keeps every integer width in one <c>long</c> buffer, so an integer
/// tensor's width lives ONLY in <see cref="RuntimeTensor.DType"/> and an op's result must be
/// narrowed to its tensor's DECLARED width. These tests pin that materializing a runtime tensor
/// back to <see cref="TensorData"/> keeps that width, that host constant folding neither
/// retypes nor mis-rounds a folded constant, and that unsigned ops wrap at the width boundary.
/// </summary>
[Trait("Domain", "Inference")]
[Trait("Purpose", "Coverage")]
public class QeeIntegerWidthTests
{
    [Fact]
    public void TestRuntimeTensorRoundTripKeepsIntegerWidth()
    {
        (DType dtype, object[] vals)[] cases = [
            (DType.Int8,   [(sbyte)-3, (sbyte)7]),
            (DType.Int16,  [(short)-300, (short)700]),
            (DType.Int32,  [-70000, 70000]),
            (DType.Int64,  [-5_000_000_000L, 5_000_000_000L]),
            (DType.UInt8,  [(byte)3, (byte)250]),
            (DType.UInt16, [(ushort)7, (ushort)65000]),
            (DType.UInt32, [7u, 4_000_000_000u]),
            (DType.UInt64, [7UL, 18_000_000_000_000_000_000UL]),
        ];
        foreach (var (dtype, vals) in cases)
        {
            var td = TensorData(dtype, [2L], vals);
            var back = TensorDataConverter.ToTensorData(TensorDataConverter.ToRuntimeTensor(td, maxElements: 16));
            Assert.NotNull(back);
            Assert.Equal(dtype, back!.DType);
            Assert.Equal(td.AccessRawMemory().ToArray(), back.AccessRawMemory().ToArray());
        }
    }

    private static TensorData Fold<TModule>(TensorData input)
    {
        var g = ((ComputationGraph)typeof(TModule)
            .GetProperty("ComputationGraph", BindingFlags.Public | BindingFlags.Static)!.GetValue(null)!).ToInternal();
        var concrete = g.ToConcreteArchitecture(g.FromOrderedInputs([input])).ToConcreteModel();
        return ComputeContext.Default.Execute(concrete, input)[0].ToTensorData();
    }

    [Fact]
    public void TestHostConstantFoldingKeepsDTypeAndWidth()
    {
        var zeros = TensorData([2L, 2L], 0f, 0f, 0f, 0f);
        var ramp = TensorData([2L, 2L], 1f, 2f, 3f, 4f);

        var folded = Fold<QeeFoldedUnsignedConstant>(zeros);
        Assert.Equal(DType.UInt32, folded.DType);
        Assert.Equal((uint[])[7u, 8u, 9u, 10u], folded.As<uint32>().AccessMemory().ToArray());

        // (2^32 - 1) + 1 is 0 in uint32, so >> 4 is 0; carrying the sum as 2^32 yields 2^28.
        Assert.Equal((uint[])[0u, 1u, 2u, 3u],
            Fold<QeeOverflowThenShift>(zeros).As<uint32>().AccessMemory().ToArray());

        var (a0, a1) = Shorokoo.Core.Rng.Threefry2x32.Bijection(0u, 0u, 123u, 456u);
        Assert.Equal(
            System.Linq.Enumerable.Range(0, 4)
                .Select(i => Shorokoo.Core.Rng.Threefry2x32.Bijection((uint)i, 0u, a0, a1).Item1).ToArray(),
            Fold<QeePartiallyFoldedUInt32Chain>(zeros).As<uint32>().AccessMemory().ToArray());

        var f64 = Fold<QeeFoldedFloat64Constant>(ramp);
        Assert.Equal(DType.Float64, f64.DType);
        Assert.Equal((double[])[7.0, 8.0, 9.0, 10.0], f64.As<float64>().AccessMemory().ToArray());

        // 1e300 folded through the float32 buffer would return Infinity.
        var wide = Fold<QeeWideFloat64Constant>(ramp).As<float64>().AccessMemory().ToArray();
        Assert.All(wide, v => Assert.True(double.IsFinite(v)));
        Assert.Equal(1e300, wide[0]);
    }

    // QEE may decline to fold (no data — the backend computes it instead); it must never fold wrong.
    private static void Qee<TModule>(DType dtype, params ulong[] expected)
    {
        var g = ((ComputationGraph)typeof(TModule)
            .GetProperty("ComputationGraph", BindingFlags.Public | BindingFlags.Static)!.GetValue(null)!).ToInternal();
        var concrete = g.ToConcreteArchitecture(g.FromOrderedInputs([])).ToConcreteModel();
        var rt = new QuickExecutionEngine().Run(concrete)[concrete.Outputs[0]];
        Assert.Equal(dtype, rt.DType);
        if (rt is RuntimeTensor { IntData: { } d })
            Assert.Equal(expected, d.Select(v => unchecked((ulong)v)));
    }

    [Fact]
    public void TestUnsignedOpsWrapAtTheWidthBoundary()
    {
        Qee<QeeU32Add>(DType.UInt32, 0, 1, 2147483650, 4294967294, 1);
        Qee<QeeU32Sub>(DType.UInt32, 4294967295, 4294967294, 4294967295, 0);
        Qee<QeeU32Mul>(DType.UInt32, 0, 4294967294, 0, 1);
        Qee<QeeU32Shift>(DType.UInt32, 2147483648, 65536, 4294967294, 65535, 1, 0);
        Qee<QeeU64Add>(DType.UInt64, 0, 1, 9223372036854775808, 18446744073709551614);
        Qee<QeeU64Sub>(DType.UInt64, 18446744073709551615, 18446744073709551614, 9223372036854775809, 0);
        Qee<QeeU64Mul>(DType.UInt64, 0, 18446744073709551614, 0, 1);
        Qee<QeeU64Shift>(DType.UInt64, 9223372036854775808, 4294967296, 4294967295, 1);
        Qee<QeeU64Bitwise>(DType.UInt64,
            9223372036854775808, 9223372036854775808, 4294967295,
            9223372036854775807, 0, 18446744069414584320,
            18446744073709551615, 9223372036854775808, 18446744073709551615);
        Qee<QeeU64Cast>(DType.UInt64, 4294967295, 0, 7);
    }

    [Fact]
    public void TestUnsignedOpsReadTheSignBitAsMagnitude()
    {
        Qee<QeeU64Div>(DType.UInt64, 4611686018427387904, 6148914691236517205, 1);
        Qee<QeeU64Mod>(DType.UInt64, 808, 5);
        Qee<QeeU64Compare>(DType.UInt64, 0, 0, 1, 1, 0, 0, 1, 1);
        Qee<QeeU64SignAbs>(DType.UInt64, 1, 0, 1, 9223372036854775808, 0, 18446744073709551615);
        Qee<QeeU64MinMax>(DType.UInt64, 7, 9223372036854775807, 9223372036854775808, 18446744073709551615);
        Qee<QeeU64Clip>(DType.UInt64, 10, 9223372036854775808, 18446744073709551614);
        Qee<QeeU64ReduceMax>(DType.UInt64, 9223372036854775808);
        Qee<QeeU64ReduceMin>(DType.UInt64, 2);
        Qee<QeeU64ReduceMean>(DType.UInt64, 3074457345618258604);
        Qee<QeeU64ReduceL1>(DType.UInt64, 9223372036854775814);
        Qee<QeeU64ReduceSum>(DType.UInt64, 9223372036854775814);
        Qee<QeeU64ReduceProd>(DType.UInt64, 9223372036854775808);
        Qee<QeeU64ArgExtreme>(DType.Int64, 2, 3);
        Qee<QeeU64TopK>(DType.UInt64, 18446744073709551615, 9223372036854775808);
        Qee<QeeU64Unique>(DType.UInt64, 4, 9223372036854775808, 18446744073709551615);
        Qee<QeeU64Pow>(DType.UInt64, 9223372036854775808, 12157665459056928801);
        Qee<QeeU64ToFloatAndBack>(DType.UInt64, 4611686018427387904, 2);
    }

    // Mean is the one reduction whose accumulator must re-enter the declared width before the
    // final step: truncation commutes with the sum but not with the divide. int64 has no narrower
    // width to re-enter, so it guards the other direction — that the narrowing is not applied there.
    [Fact]
    public void TestFoldedIntegerReduceMeanMatchesTheBackend()
    {
        Assert.Equal(Backend<QeeI32ReduceMeanRuntime>(TensorData(DType.Int32, [2L], 2147483647, 2147483647)),
                     Folded<QeeI32ReduceMeanFolded>(DType.Int32));
        Assert.Equal(-1L, Backend<QeeI64ReduceMeanRuntime>(TensorData(DType.Int64, [2L], long.MaxValue, long.MaxValue)));
        Qee<QeeI64ReduceMeanFolded>(DType.Int64, unchecked((ulong)-1L));
    }

    // The widths ONNX's Mean type constraint rejects, so there is no backend to compare against —
    // the declared-width rule is what keeps them self-consistent. Sums overflow every width here.
    [Fact]
    public void TestFoldedNarrowReduceMeanStaysInTheDeclaredWidth()
    {
        Qee<QeeU8ReduceMeanFolded>(DType.UInt8, 127);
        Qee<QeeI8ReduceMeanFolded>(DType.Int8, unchecked((ulong)-1L));
        Qee<QeeU16ReduceMeanFolded>(DType.UInt16, 32767);
    }

    // The width-boundary cases where folding and the backend must agree: +, -, * and Sum all
    // commute with truncation to the declared width, so the 64-bit buffer is free to overflow.
    [Fact]
    public void TestFoldedIntegerArithmeticMatchesTheBackend()
    {
        TensorData I32(int v) => TensorData(DType.Int32, [1L], v);
        TensorData U32(uint v) => TensorData(DType.UInt32, [1L], v);

        Assert.Equal(Backend<QeeI32AddRuntime>(I32(2147483647), I32(1)), Folded<QeeI32AddFolded>(DType.Int32));
        Assert.Equal(Backend<QeeI32SubRuntime>(I32(-2147483648), I32(1)), Folded<QeeI32SubFolded>(DType.Int32));
        Assert.Equal(Backend<QeeU32AddRuntime>(U32(4294967295), U32(1)), Folded<QeeU32AddFolded>(DType.UInt32));
        Assert.Equal(Backend<QeeU32SubRuntime>(U32(0), U32(1)), Folded<QeeU32SubFolded>(DType.UInt32));
        Assert.Equal(Backend<QeeU32MulRuntime>(U32(65536), U32(65536)), Folded<QeeU32MulFolded>(DType.UInt32));
        Assert.Equal(Backend<QeeI32ReduceSumRuntime>(TensorData(DType.Int32, [2L], 2147483647, 2147483647)),
                     Folded<QeeI32ReduceSumFolded>(DType.Int32));
        Assert.Equal(Backend<QeeI32AddThenDivRuntime>(I32(2147483647), I32(1), I32(5)),
                     Folded<QeeI32AddThenDivFolded>(DType.Int32));
        Assert.Equal(Backend<QeeU32AddThenDivRuntime>(U32(4294967295), U32(2), U32(5)),
                     Folded<QeeU32AddThenDivFolded>(DType.UInt32));
    }

    private static long Backend<TModule>(params TensorData[] inputs)
    {
        var g = ((ComputationGraph)typeof(TModule)
            .GetProperty("ComputationGraph", BindingFlags.Public | BindingFlags.Static)!.GetValue(null)!).ToInternal();
        var concrete = g.ToConcreteArchitecture(g.FromOrderedInputs([.. inputs])).ToConcreteModel();
        var outData = ComputeContext.Default.Execute(concrete, inputs)[0].ToTensorData();
        return outData.DType switch
        {
            var d when d == DType.UInt32 => outData.As<uint32>().AccessMemory().ToArray()[0],
            var d when d == DType.Int64 => outData.As<int64>().AccessMemory().ToArray()[0],
            _ => outData.As<int32>().AccessMemory().ToArray()[0],
        };
    }

    private static long Folded<TModule>(DType dtype)
    {
        var g = ((ComputationGraph)typeof(TModule)
            .GetProperty("ComputationGraph", BindingFlags.Public | BindingFlags.Static)!.GetValue(null)!).ToInternal();
        var x = TensorData([2L], 0f, 0f);
        var concrete = g.ToConcreteArchitecture(g.FromOrderedInputs([x])).ToConcreteModel();
        foreach (var kv in new QuickExecutionEngine().Run(concrete))
            if (kv.Value.DType == dtype && kv.Value is RuntimeTensor { IntData: { Length: 1 } m })
                return dtype == DType.UInt32 ? (uint)m[0] : (int)m[0];
        throw new InvalidOperationException("the constant sub-chain did not fold");
    }
}

/// <summary>A uint64 divide whose operands are constants, consumed by a runtime-valued tensor so
/// the divide is forced through host constant folding. The dividend is above long.MaxValue.</summary>
[Module]
public partial class QeeUInt64SignedDivide
{
    public static Tensor<uint64> Inline(Tensor<float32> x)
    {
        var runtime = OnnxOp.Range(Scalar(0L), x.ShapeTensor().Reduce(ReduceKind.Prod), Scalar(1L))
            .int64().Cast<uint64>();
        var folded = OnnxOp.Div(Scalar(9223372036854775808UL), Scalar(2UL)).uint64();
        return runtime + folded;
    }
}

/// <summary>A uint64 modulo whose operands are constants, forced through host constant folding the
/// same way as <see cref="QeeUInt64SignedDivide"/>. Both dividends are above long.MaxValue.</summary>
[Module]
public partial class QeeUInt64SignedModulo
{
    public static Tensor<uint64> Inline(Tensor<float32> x)
    {
        var runtime = OnnxOp.Range(Scalar(0L), x.ShapeTensor().Reduce(ReduceKind.Prod), Scalar(1L))
            .int64().Cast<uint64>();
        // 2^63 % 1000 == 808 unsigned; the signed floored modulo of -2^63 gives 192 instead.
        var folded = OnnxOp.Mod(Scalar(9223372036854775808UL), Scalar(1000UL)).uint64();
        return runtime + folded;
    }
}

/// <summary>A uint64 divide whose dividend is <c>ulong.MaxValue</c> — the all-ones bit pattern,
/// which signed division reads as <c>-1</c> and so collapses to 0 for any divisor &gt; 1.</summary>
[Module]
public partial class QeeUInt64SignedDivideMaxValue
{
    public static Tensor<uint64> Inline(Tensor<float32> x)
    {
        var runtime = OnnxOp.Range(Scalar(0L), x.ShapeTensor().Reduce(ReduceKind.Prod), Scalar(1L))
            .int64().Cast<uint64>();
        var folded = OnnxOp.Div(Scalar(18446744073709551615UL), Scalar(3UL)).uint64();
        return runtime + folded;
    }
}

/// <summary>
/// QEE holds every integer width in one <c>long</c> buffer, so a <c>uint64</c> above
/// <c>long.MaxValue</c> is a negative bit-pattern long. Host constant folding runs the kernels over
/// that buffer and bakes the result into the graph, so a signed operator there persists a wrong
/// value rather than merely displaying one.
/// </summary>
[Trait("Domain", "Inference")]
[Trait("Purpose", "Coverage")]
public class QeeUInt64SignedOperatorTests
{
    [Fact]
    public void TestFoldedUInt64DivideUsesUnsignedSemantics()
    {
        var g = ((ComputationGraph)typeof(QeeUInt64SignedDivide)
            .GetProperty("ComputationGraph")!.GetValue(null)!).ToInternal();
        var input = TensorData([2L, 2L], 0f, 0f, 0f, 0f);
        var concrete = g.ToConcreteArchitecture(g.FromOrderedInputs([input])).ToConcreteModel();
        var got = ComputeContext.Default.Execute(concrete, input)[0]
            .ToTensorData().As<uint64>().AccessMemory().ToArray();

        // 2^63 / 2 == 2^62, plus the element index. Signed division of the bit pattern gives
        // -2^62, i.e. 2^64 - 2^62 = 13835058055282163712.
        const ulong half = 4611686018427387904UL;   // 2^62
        Assert.Equal((ulong[])[half, half + 1, half + 2, half + 3], got);
    }

    // Same fault, Mod rather than Div — pinned separately because #133's bits packing is specified
    // as `(word / 2^(W*l)) mod 2^W`, so a literal implementation reaches BOTH operators.
    [Fact]
    public void TestFoldedUInt64ModuloUsesUnsignedSemantics()
    {
        var g = ((ComputationGraph)typeof(QeeUInt64SignedModulo)
            .GetProperty("ComputationGraph")!.GetValue(null)!).ToInternal();
        var input = TensorData([2L, 2L], 0f, 0f, 0f, 0f);
        var concrete = g.ToConcreteArchitecture(g.FromOrderedInputs([input])).ToConcreteModel();
        var got = ComputeContext.Default.Execute(concrete, input)[0]
            .ToTensorData().As<uint64>().AccessMemory().ToArray();

        // 2^63 % 1000 == 808, plus the element index. Signed reads the bit pattern as -2^63, and
        // ONNX Mod (fmod=0) is FLOORED rather than truncated, so it returns 1000 - 808 == 192 —
        // a plausible-looking small remainder, which is what makes this one easy to miss.
        const ulong rem = 808UL;
        Assert.Equal((ulong[])[rem, rem + 1, rem + 2, rem + 3], got);
    }

    // The all-ones dividend: signed division reads ulong.MaxValue as -1, so ANY divisor > 1
    // collapses the result to 0 — the most destructive shape of this bug, since it survives every
    // "is it roughly right?" eyeball check.
    [Fact]
    public void TestFoldedUInt64DivideOfMaxValueUsesUnsignedSemantics()
    {
        var g = ((ComputationGraph)typeof(QeeUInt64SignedDivideMaxValue)
            .GetProperty("ComputationGraph")!.GetValue(null)!).ToInternal();
        var input = TensorData([2L, 2L], 0f, 0f, 0f, 0f);
        var concrete = g.ToConcreteArchitecture(g.FromOrderedInputs([input])).ToConcreteModel();
        var got = ComputeContext.Default.Execute(concrete, input)[0]
            .ToTensorData().As<uint64>().AccessMemory().ToArray();

        // (2^64 - 1) / 3 == 6148914691236517205, plus the element index. Signed gives -1 / 3 == 0.
        const ulong third = 6148914691236517205UL;
        Assert.Equal((ulong[])[third, third + 1, third + 2, third + 3], got);
    }

    // FAILING — this pins an open fault in ONNX RUNTIME rather than guarding a regression, and
    // Shorokoo cannot fix it. Reproduced with a bare ONNX graph holding one Max node on 1.26.0 and
    // 1.28.0, opsets 13 and 21. Max/Min behave as a 64-bit compare split into 32-bit halves with
    // the LOW half compared signed, so they mis-order a pair exactly when the high halves are
    // equal and one low half has bit 31 set. Greater and Add on the same operands are correct.
    // Nothing in the product is exposed: every int64 Max operand stays inside (-2^31, 2^31), where
    // both halves are same-signed. No issue is open on Shorokoo/Shorokoo to skip against.
    [Fact]
    public void TestInt64MaxMisordersWhenTheLowHalvesStraddleBitThirtyOne()
    {
        var g = ((ComputationGraph)typeof(QeeI64MaxLowHalfSigned)
            .GetProperty("ComputationGraph")!.GetValue(null)!).ToInternal();
        // Pairs whose high halves are equal; the low halves straddle bit 31 in all but the first.
        long[] a = [5L, 1L << 31, (1L << 32) - 1, (1L << 32) + (1L << 31), (1L << 40) + (1L << 31) + 9,
                    3 * (1L << 32) + ((1L << 32) - 1), -2147483648L];
        long[] b = [9L, 1L, 1L, (1L << 32) + 5, (1L << 40) + 5, 3 * (1L << 32) + 1, -4294967291L];
        var inputA = TensorData(DType.Int64, [(long)a.Length], [.. a.Select(o => (object)o)]);
        var inputB = TensorData(DType.Int64, [(long)b.Length], [.. b.Select(o => (object)o)]);
        var concrete = g.ToConcreteArchitecture(g.FromOrderedInputs([inputA, inputB])).ToConcreteModel();
        var got = ComputeContext.Default.Execute(concrete, inputA, inputB)[0]
            .ToTensorData().As<int64>().AccessMemory().ToArray();
        Assert.Equal([.. a.Zip(b, Math.Max)], got);
    }
}

[Module] public partial class QeeI64MaxLowHalfSigned {
    public static Tensor<int64> Inline(Tensor<int64> a, Tensor<int64> b) => a.Max(b); }

[Module] public partial class QeeU32Add { public static Tensor<uint32> Inline()
    => Vector(4294967295u, 0u, 2147483648u, 4294967295u, 4294967295u) + Vector(1u, 1u, 2u, 4294967295u, 2u); }

[Module] public partial class QeeU32Sub { public static Tensor<uint32> Inline()
    => Vector(0u, 0u, 1u, 4294967295u) - Vector(1u, 2u, 2u, 4294967295u); }

[Module] public partial class QeeU32Mul { public static Tensor<uint32> Inline()
    => Vector(2147483648u, 4294967295u, 65536u, 4294967295u) * Vector(2u, 2u, 65536u, 4294967295u); }

[Module] public partial class QeeU32Shift { public static Tensor<uint32> Inline()
    => (Tensor<uint32>)OnnxOp.Concat([
        OnnxOp.BitShift(Vector(1u, 1u, 4294967295u), Vector(31u, 16u, 1u), BitShiftDirection.Left),
        OnnxOp.BitShift(Vector(4294967295u, 2147483648u, 1u), Vector(16u, 31u, 1u), BitShiftDirection.Right)], axis: 0); }

[Module] public partial class QeeU64Add { public static Tensor<uint64> Inline()
    => Vector(18446744073709551615UL, 0UL, 9223372036854775808UL, 18446744073709551615UL)
     + Vector(1UL, 1UL, 0UL, 18446744073709551615UL); }

[Module] public partial class QeeU64Sub { public static Tensor<uint64> Inline()
    => Vector(0UL, 0UL, 9223372036854775808UL, 18446744073709551615UL)
     - Vector(1UL, 2UL, 18446744073709551615UL, 18446744073709551615UL); }

[Module] public partial class QeeU64Mul { public static Tensor<uint64> Inline()
    => Vector(9223372036854775808UL, 18446744073709551615UL, 4294967296UL, 18446744073709551615UL)
     * Vector(2UL, 2UL, 4294967296UL, 18446744073709551615UL); }

[Module] public partial class QeeU64Shift { public static Tensor<uint64> Inline()
    => (Tensor<uint64>)OnnxOp.Concat([
        OnnxOp.BitShift(Vector(1UL, 1UL), Vector(63UL, 32UL), BitShiftDirection.Left),
        OnnxOp.BitShift(Vector(18446744073709551615UL, 9223372036854775808UL), Vector(32UL, 63UL),
            BitShiftDirection.Right)], axis: 0); }

[Module] public partial class QeeU64Bitwise { public static Tensor<uint64> Inline()
{
    var hi = Vector(18446744073709551615UL, 9223372036854775808UL, 18446744073709551615UL);
    var lo = Vector(9223372036854775808UL, 9223372036854775808UL, 4294967295UL);
    return (Tensor<uint64>)OnnxOp.Concat(
        [OnnxOp.BitwiseAnd(hi, lo), OnnxOp.BitwiseXor(hi, lo), OnnxOp.BitwiseOr(hi, lo)], axis: 0);
} }

[Module] public partial class QeeU64Cast { public static Tensor<uint64> Inline()
    => Vector(18446744073709551615UL, 9223372036854775808UL, 4294967303UL).Cast<uint32>().Cast<uint64>(); }

// The uint64 sign-bit family: every kernel below reads a lane above long.MaxValue, which the
// shared long buffer holds as a negative bit pattern. Consumed by
// TestUnsignedOpsReadTheSignBitAsMagnitude.

[Module] public partial class QeeU64Div { public static Tensor<uint64> Inline()
    => OnnxOp.Div(Vector(9223372036854775808UL, 18446744073709551615UL, 18446744073709551615UL),
                  Vector(2UL, 3UL, 18446744073709551615UL)).uint64(); }

[Module] public partial class QeeU64Mod { public static Tensor<uint64> Inline()
    => OnnxOp.Mod(Vector(9223372036854775808UL, 18446744073709551615UL), Vector(1000UL, 10UL)).uint64(); }

[Module] public partial class QeeU64Compare { public static Tensor<uint64> Inline()
{
    var hi = Vector(9223372036854775808UL, 18446744073709551615UL);
    var one = Vector(1UL, 1UL);
    return OnnxOp.Cast(OnnxOp.Concat([
        OnnxOp.Less(hi, one), OnnxOp.Greater(hi, one),
        OnnxOp.LessOrEqual(hi, one), OnnxOp.GreaterOrEqual(hi, one)], axis: 0), null, DType.UInt64).uint64();
} }

[Module] public partial class QeeU64SignAbs { public static Tensor<uint64> Inline()
{
    var v = Vector(9223372036854775808UL, 0UL, 18446744073709551615UL);
    return (Tensor<uint64>)OnnxOp.Concat([OnnxOp.Sign(v), OnnxOp.Abs(v)], axis: 0);
} }

[Module] public partial class QeeU64MinMax { public static Tensor<uint64> Inline()
{
    var a = Vector(9223372036854775808UL, 18446744073709551615UL);
    var b = Vector(7UL, 9223372036854775807UL);
    return (Tensor<uint64>)OnnxOp.Concat([OnnxOp.Min(a, b), OnnxOp.Max(a, b)], axis: 0);
} }

[Module] public partial class QeeU64Clip { public static Tensor<uint64> Inline()
    => OnnxOp.Clip(Vector(1UL, 9223372036854775808UL, 18446744073709551615UL),
                   Scalar(10UL), Scalar(18446744073709551614UL)).uint64(); }

[Module] public partial class QeeU64ReduceMax { public static Tensor<uint64> Inline()
    => Vector(9223372036854775808UL, 4UL, 2UL).Reduce(ReduceKind.Max); }

[Module] public partial class QeeU64ReduceMin { public static Tensor<uint64> Inline()
    => Vector(9223372036854775808UL, 4UL, 2UL).Reduce(ReduceKind.Min); }

[Module] public partial class QeeU64ReduceMean { public static Tensor<uint64> Inline()
    => Vector(9223372036854775808UL, 4UL, 2UL).Reduce(ReduceKind.Mean); }

[Module] public partial class QeeU64ReduceL1 { public static Tensor<uint64> Inline()
    => Vector(9223372036854775808UL, 4UL, 2UL).Reduce(ReduceKind.L1); }

[Module] public partial class QeeU64ReduceSum { public static Tensor<uint64> Inline()
    => Vector(9223372036854775808UL, 4UL, 2UL).Reduce(ReduceKind.Sum); }

[Module] public partial class QeeU64ReduceProd { public static Tensor<uint64> Inline()
    => Vector(9223372036854775808UL, 3UL).Reduce(ReduceKind.Prod); }

[Module] public partial class QeeU64ArgExtreme { public static Tensor<int64> Inline()
{
    var v = Vector(4UL, 9223372036854775808UL, 18446744073709551615UL, 1UL);
    return (Tensor<int64>)OnnxOp.Concat([
        OnnxOp.ArgMax(v, 0, true, false), OnnxOp.ArgMin(v, 0, true, false)], axis: 0);
} }

[Module] public partial class QeeU64TopK { public static Tensor<uint64> Inline()
    => OnnxOp.TopK(Vector(4UL, 9223372036854775808UL, 18446744073709551615UL, 1UL), Vector(2L)).values.uint64(); }

[Module] public partial class QeeU64Unique { public static Tensor<uint64> Inline()
    => OnnxOp.Unique(Vector(18446744073709551615UL, 4UL, 9223372036854775808UL, 4UL), sorted: true).y.uint64(); }

[Module] public partial class QeeU64Pow { public static Tensor<uint64> Inline()
    => OnnxOp.Pow(Vector(2UL, 3UL), Vector(63UL, 40UL)).uint64(); }

[Module] public partial class QeeU64ToFloatAndBack { public static Tensor<uint64> Inline()
    => OnnxOp.Div(Vector(9223372036854775808UL, 4UL).Cast<float32>(), Scalar(2f)).float32().Cast<uint64>(); }

// The reduce accumulator pair: the same reduction as a folded constant and as a backend-executed
// runtime value. Consumed by TestFoldedIntegerReduceMeanMatchesTheBackend.

[Module] public partial class QeeI32ReduceMeanFolded { public static Tensor<int32> Inline(Tensor<float32> x)
{
    var runtime = OnnxOp.Range(Scalar(0L), x.ShapeTensor().Reduce(ReduceKind.Prod), Scalar(1L))
        .int64().Cast<int32>();
    return runtime * Vector(2147483647, 2147483647).Reduce(ReduceKind.Mean, null, true).int32();
} }

[Module] public partial class QeeI32ReduceMeanRuntime { public static Tensor<int32> Inline(Tensor<int32> v)
    => v.Reduce(ReduceKind.Mean, null, true); }

[Module] public partial class QeeI32ReduceSumFolded { public static Tensor<int32> Inline(Tensor<float32> x)
{
    var runtime = OnnxOp.Range(Scalar(0L), x.ShapeTensor().Reduce(ReduceKind.Prod), Scalar(1L))
        .int64().Cast<int32>();
    return runtime * Vector(2147483647, 2147483647).Reduce(ReduceKind.Sum, null, true).int32();
} }

[Module] public partial class QeeI32ReduceSumRuntime { public static Tensor<int32> Inline(Tensor<int32> v)
    => v.Reduce(ReduceKind.Sum, null, true); }

// Width-boundary arithmetic, each as a folded constant and as a backend-executed runtime value.
// Consumed by TestFoldedIntegerArithmeticMatchesTheBackend.

[Module] public partial class QeeI32AddFolded { public static Tensor<int32> Inline(Tensor<float32> x)
    => Runtime32(x) * (Vector(2147483647) + Vector(1)); }
[Module] public partial class QeeI32AddRuntime { public static Tensor<int32> Inline(Tensor<int32> a, Tensor<int32> b) => a + b; }

[Module] public partial class QeeI32SubFolded { public static Tensor<int32> Inline(Tensor<float32> x)
    => Runtime32(x) * (Vector(-2147483648) - Vector(1)); }
[Module] public partial class QeeI32SubRuntime { public static Tensor<int32> Inline(Tensor<int32> a, Tensor<int32> b) => a - b; }

[Module] public partial class QeeU32AddFolded { public static Tensor<uint32> Inline(Tensor<float32> x)
    => RuntimeU32(x) * (Vector(4294967295u) + Vector(1u)); }
[Module] public partial class QeeU32AddRuntime { public static Tensor<uint32> Inline(Tensor<uint32> a, Tensor<uint32> b) => a + b; }

[Module] public partial class QeeU32SubFolded { public static Tensor<uint32> Inline(Tensor<float32> x)
    => RuntimeU32(x) * (Vector(0u) - Vector(1u)); }
[Module] public partial class QeeU32SubRuntime { public static Tensor<uint32> Inline(Tensor<uint32> a, Tensor<uint32> b) => a - b; }

[Module] public partial class QeeU32MulFolded { public static Tensor<uint32> Inline(Tensor<float32> x)
    => RuntimeU32(x) * (Vector(65536u) * Vector(65536u)); }
[Module] public partial class QeeU32MulRuntime { public static Tensor<uint32> Inline(Tensor<uint32> a, Tensor<uint32> b) => a * b; }

// An add that overflows the declared width feeding a divide. Two ops, so the narrowing tail runs
// between them and the divide sees the wrapped value.
// Consumed by TestFoldedIntegerArithmeticMatchesTheBackend.

[Module] public partial class QeeI32AddThenDivFolded { public static Tensor<int32> Inline(Tensor<float32> x)
    => Runtime32(x) * OnnxOp.Div(Vector(2147483647) + Vector(1), Vector(5)).int32(); }
[Module] public partial class QeeI32AddThenDivRuntime {
    public static Tensor<int32> Inline(Tensor<int32> a, Tensor<int32> b, Tensor<int32> c)
        => OnnxOp.Div(a + b, c).int32(); }

[Module] public partial class QeeU32AddThenDivFolded { public static Tensor<uint32> Inline(Tensor<float32> x)
    => RuntimeU32(x) * OnnxOp.Div(Vector(4294967295u) + Vector(2u), Vector(5u)).uint32(); }
[Module] public partial class QeeU32AddThenDivRuntime {
    public static Tensor<uint32> Inline(Tensor<uint32> a, Tensor<uint32> b, Tensor<uint32> c)
        => OnnxOp.Div(a + b, c).uint32(); }

// Mean at the widths the declared-width rule has to reach.
// Consumed by TestFoldedIntegerReduceMeanMatchesTheBackend / …StaysInTheDeclaredWidth.

[Module] public partial class QeeI64ReduceMeanFolded { public static Tensor<int64> Inline()
    => Vector(9223372036854775807L, 9223372036854775807L).Reduce(ReduceKind.Mean); }
[Module] public partial class QeeI64ReduceMeanRuntime { public static Tensor<int64> Inline(Tensor<int64> v)
    => v.Reduce(ReduceKind.Mean, null, true); }

[Module] public partial class QeeU8ReduceMeanFolded { public static Tensor<uint8> Inline()
    => Vector(255u, 255u).Cast<uint8>().Reduce(ReduceKind.Mean); }

[Module] public partial class QeeI8ReduceMeanFolded { public static Tensor<int8> Inline()
    => Vector(127, 127).Cast<int8>().Reduce(ReduceKind.Mean); }

[Module] public partial class QeeU16ReduceMeanFolded { public static Tensor<uint16> Inline()
    => Vector(65535u, 65535u).Cast<uint16>().Reduce(ReduceKind.Mean); }
