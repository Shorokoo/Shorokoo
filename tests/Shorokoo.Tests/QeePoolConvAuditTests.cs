using Shorokoo.Core.Backends;
using Shorokoo.Core.Interpreter;
using static Shorokoo.Tests.Utils.QeeAudit;

namespace Shorokoo.Tests;

/// <summary>
/// QEE audit batch: pooling and convolution families (ONNX opset 21). Each
/// module in QeePoolConvAuditModules.cs compares the ShapeTensor() of every op result
/// against the spec-expected dims and returns a single Scalar&lt;bit&gt;;
/// <see cref="QeeAudit.Check{TModule}"/> validates that bit under both real ONNX Runtime
/// execution and the <see cref="QuickExecutionEngine"/>'s own shape inference.
/// </summary>
[Trait("Domain", "Inference")]
[Trait("Purpose", "Coverage")]
public class QeePoolConvAuditTests
{
    private static TensorData Image1x1x10x10 => F32Wave([1L, 1L, 10L, 10L]);
    private static TensorData Image1x3x10x10 => F32Wave([1L, 3L, 10L, 10L]);

    [Fact]
    public void TestQeePoolingShapeAudits()
    {
        Assert.True(QeeAudit.Check<QeeMaxPoolShapeAuditCheck>(Image1x1x10x10));
        Assert.True(QeeAudit.Check<QeeAveragePoolShapeAuditCheck>(Image1x1x10x10));
        Assert.True(QeeAudit.Check<QeeLpPoolGlobalPoolShapeAuditCheck>(Image1x3x10x10));
        Assert.True(QeeAudit.Check<QeeMaxRoiPoolShapeAuditCheck>(
            F32Wave([1L, 2L, 8L, 8L]),
            F32([2L, 5L], 0f, 0f, 0f, 7f, 7f, 0f, 1f, 1f, 6f, 6f)));
        Assert.True(QeeAudit.Check<QeeMaxUnpoolShapeAuditCheck>(
            F32([1L, 1L, 2L, 2L], 6f, 8f, 14f, 16f),
            I64([1L, 1L, 2L, 2L], 0L, 2L, 5L, 7L)));
        Assert.True(QeeAudit.Check<QeePoolVariantsShapeAuditCheck>(
            F32Wave([1L, 2L, 11L]), F32Wave([1L, 2L, 9L, 8L]), F32Wave([1L, 1L, 5L, 6L, 4L])));
    }

    [Fact]
    public void TestQeeConvolutionShapeAudits()
    {
        Assert.True(QeeAudit.Check<QeeConvShapeAuditCheck>(F32Wave([1L, 4L, 9L, 9L])));
        Assert.True(QeeAudit.Check<QeeConvTransposeShapeAuditCheck>(F32Wave([1L, 2L, 5L, 5L])));
        Assert.True(QeeAudit.Check<QeeQuantizedConvShapeAuditCheck>(
            I8Zeros([1L, 1L, 7L, 7L]), I8Zeros([1L, 1L, 3L, 3L]),
            I8([], (sbyte)0), I8([], (sbyte)0),
            F32([], 0.5f), F32([], 0.25f), F32([], 0.5f), I8([], (sbyte)0)));
        Assert.True(QeeAudit.CheckWith<QeeDeformConvShapeAuditCheck>(
            [F32Wave([1L, 1L, 4L, 4L]),
             F32([1L, 1L, 2L, 2L], 1f, 0f, 0f, 1f),
             F32Wave([1L, 8L, 3L, 2L]),
             F32([1L], 0f)],
            testOnnxRoundtrip: false, testCsRoundtrip: false));
        Assert.True(QeeAudit.Check<QeeConvVariantsShapeAuditCheck>(
            F32Wave([2L, 2L, 9L]), F32Wave([1L, 4L, 7L, 6L]), F32Wave([1L, 2L, 5L, 4L, 4L]), F32Wave([1024L])));
    }

    [Fact]
    public void TestSameAutoPadWithDilationsPadsForTheDilatedKernel()
    {
        var x = F32([1L, 1L, 10L], [.. Enumerable.Range(0, 10).Select(i => (float)i)]);
        Assert.True(AutoTest.AdvancedTestGraph<SameDilatedMaxPoolValues>([], [x],
            expected: [3, 5, 7, 9, 9, 2, 4, 6, 8, 8]));
        Assert.True(AutoTest.AdvancedTestGraph<SameDilatedLpPoolValues>([], [x],
            expected: [3.1622777, 5.9160798, 9.1104336, 12.4498996, 11.4017543, 2, 4.4721360, 7.4833148, 10.7703296, 10]));
        Assert.True(AutoTest.AdvancedTestGraph<SameDilatedAveragePoolValues>([], [x],
            expected: [2, 3, 5, 7, 8, 1, 2, 4, 6, 7]));
        Assert.True(AutoTest.AdvancedTestGraph<SameDilatedPoolOddLengthValues>([], [F32([1L, 1L, 9L], [.. Enumerable.Range(0, 9).Select(i => (float)i)])],
            expected: [2, 4, 6, 8, 8, 2 / 3.0, 2, 4, 6, 14 / 3.0, 2, 3, 4, 5, 6, 7, 8, 7, 8]));
        Assert.True(AutoTest.AdvancedTestGraph<SameDilatedMaxPoolWithIndicesValues>([], [F32([1L, 1L, 5L, 6L], [.. Enumerable.Range(0, 30).Select(i => (float)(i * 7 % 30))])],
            expected: [7, 21, 29, 25, 25, 29, 25, 25, 29, 1, 3, 17, 25, 25, 17, 25, 25, 17]));
    }

    [Fact]
    public void TestPaddingAsLargeAsTheKernelIsPooled()
    {
        var x = F32([1L, 1L, 9L], [.. Enumerable.Range(-4, 9).Select(i => (float)i)]);
        Assert.True(AutoTest.AdvancedTestGraph<PadsReachingTheKernelPoolValues>([], [x],
            expected: [-3, -2, -1, 0, 1, 2, 3, 4, 2, -3, -2, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 2, -1.5, -1, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 1,
                3, 2, 4.1231056, 3, 2.236068, 2.236068, 3, 4.1231056, 2, -3, 0, 3]));
        Assert.True(AutoTest.AdvancedTestGraph<SameWideDilationPoolValues>([], [x],
            expected: [-2, 0, 2, 4, 3, -3, -2, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 2, -5 / 3.0, -1 / 3.0, 0, 1 / 3.0, 5 / 3.0, 2, 4, 2.8284271, 4, 2]));
        Assert.True(AutoTest.AdvancedTestGraph<PadsReachingTheKernelMaxPoolWithIndicesValues>([],
            [F32([1L, 2L, 4L, 4L], [.. Enumerable.Range(0, 32).Select(i => (float)(i * 7 % 32 - 16))])],
            expected: [12, -6, 4, -2, 10, 10, 14, 14, 4, 6, 12, 2, 22, 22, 18, 18, 12, -6, 4, -2, 10, 10, 14, 14, 1, 9, 3, 8, 25, 25, 24, 24,
                -10, 15, -6, -13, 6, -1, 10, 3, 10, 9, 6, 5, 26, 25, 22, 21]));
        Assert.True(AutoTest.AdvancedTestGraph<PadsReachingTheKernelMaxPoolElementTypeValues>([], [x],
            expected: [-3, -2, -1, 0, 1, 2, 3, 4, 2, 1, 2, 3, 4, 5, 6, 7, 8, 6, -3, -2, -1, 0, 1, 2, 3, 4, 2]));
        Assert.True(AutoTest.AdvancedTestGraph<PadsReachingTheKernelThreeAxisPoolValues>([],
            [F32([1L, 2L, 3L, 4L, 3L], [.. Enumerable.Range(0, 72).Select(i => (float)(i * 7 % 11 - 5))])],
            expected: [4, 4, 2, -2, 4, 4, 0, -4, 2, 2, 5, 5, 0, 0, 3, 3, 4, 5, 5, 1, 4, 4, 3, -1, 5, 5, 1, -3, 3, 3, -1, -5, 1, 1, 4, 4,
                -1, -1, 2, 2, 5, 5, 4, 0, 3, 3, 2, -2, 0.125, 0.125, 0.25, 0.25, 0, -0.25, 0, -0.125, 0.125, 0.125, 0.25, 0.25, 0, -0.25,
                0, -0.125, 0, -0.125, 0, -0.25, -0.25, 0.625, -0.125, -0.375, 0, -0.125, 0, -0.25, -0.25, 0.625, -0.125, -0.375]));
    }

    [Fact]
    public void TestCeilModeKeepsTheSpecsWindowsWhenPadsReachTheKernel()
        => Assert.True(AutoTest.AdvancedTestGraph<CeilModePadsReachingTheKernelPoolValues>([],
            [F32([1L, 1L, 9L], [.. Enumerable.Range(-4, 9).Select(i => (float)i)])],
            expected: [-3.5, -0.5, 2.5, 0, 5, 1, 3.6055513, 0, -3, -1, 1, 3, 3, -4, -1, 2, 3, -3.5, -2, 0, 2, 3.5]));

    [Fact]
    public void TestNegativeSamePaddingShiftsTheWindowsAsTheSpecDoes()
        => Assert.True(AutoTest.AdvancedTestGraph<NegativeSamePaddingPoolValues>([],
            [F32([1L, 1L, 8L], 5f, -3f, 8f, 1f, -7f, 2f, 6f, -4f)],
            expected: [1, 2, 8, 6, 2.5, 1, 1, 4, 3, 7, 4, 8, 6, -3, -7, -4]));

    [Fact]
    public void TestNegativeSamePaddingMaxPoolIndicesPointIntoTheInput()
        => Assert.True(AutoTest.AdvancedTestGraph<NegativeSamePaddingMaxPoolIndicesValues>([],
            [F32([1L, 2L, 5L, 6L], [.. Enumerable.Range(0, 60).Select(i => (float)(i * 7 % 11 - 5))])],
            expected: [4, -1, 5, 4, 5, 0, -1, 5, 6, 10, 25, 28, 36, 40, 54, 58, 2, 1, 5, 4, 3, 2, -3, 5, 1, 4, 25, 28, 31, 34, 49, 58]));

    [Fact]
    public void TestMaxPoolIndexOfALowestValuedWindowIsItsFirstPosition()
        => Assert.True(AutoTest.AdvancedTestGraph<LowestValueWindowMaxPoolIndicesValues>([],
            [F32([1L, 1L, 5L], 0f, 0f, 5f, 0f, 0f)],
            expected: [0, 2, 2, 3, 0, 2, 2, 3, 0, 1, 2, 1, 2, 3, 4]));

    [Fact]
    public void TestPaddedMaxPoolIndicesPointAtTheFirstMaximumOfTheirWindow()
        => Assert.True(AutoTest.AdvancedTestGraph<PaddedMaxPoolIndicesPointAtTheirValues>([],
            [F32([1L, 1L, 5L], 0f, 0f, 5f, 0f, 0f)]));

    [Fact]
    public void TestIndexedSmallIntegerMaxPoolGivesThePlainPoolsValuesOverWholePaddingWindows()
        => Assert.True(AutoTest.AdvancedTestGraph<IndexedSmallIntegerMaxPoolMatchesThePlainPool>([],
            [F32([1L, 1L, 4L], 3f, 0f, 7f, 1f)]));

    private const double N = double.NegativeInfinity;
    private const double L = double.MinValue;
    private static readonly double[] FloatWindows = [N, N, 1, N, N, N, L, N, N, N, L, L];
    private static readonly double[] FloatWindowMaxima = [N, 1, L, L, N, 1, L, L, N, N, 1, N, 1, N, N, N, N, N, L, N, L, N, N, N, L, L, L, L];
    private static readonly long[] FloatWindowPositions = [0, 2, 6, 10, 0, 4, 6, 9, 0, 1, 2, 1, 2, 3, 4, 3, 4, 5, 6, 7, 6, 7, 8, 9, 10, 11, 10, 11];
    private static readonly double[] IntegerWindows = [L, L, 1, L, L, L, L, L, L, L, L, L];
    private static readonly double[] IntegerWindowMaxima = [L, 1, L, L, L, 1, L, L, L, L, 1, L, 1, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L];
    private static readonly long[] IntegerWindowPositions = [0, 2, 6, 7, 0, 4, 6, 8, 0, 1, 2, 1, 2, 3, 4, 3, 4, 5, 6, 7, 6, 7, 8, 9, 10, 9, 10, 11];

    private static object AsFloat16(double v) => v == L ? Float16.MinValue : (Float16)(float)v;
    private static object AsFloat32(double v) => v == L ? float.MinValue : (float)v;
    private static object AsFloat64(double v) => v;
    private static object AsInt8(double v) => v == L ? sbyte.MinValue : (sbyte)v;
    private static object AsUInt8(double v) => v == L ? byte.MinValue : (byte)v;

    private static TensorData Typed(DType t, Func<double, object> of, long[] dims, double[] vals)
        => Globals.TensorData(t, dims, [.. vals.Select(of)]);

    private static bool LowestWindows(DType t, Func<double, object> of, double[] x, double[] y, long[] indices)
        => AutoTest.AdvancedTestGraph<LowestValueWindowMaxPoolCheck>([],
            [Typed(t, of, [1L, 2L, 2L, 3L], x), Typed(t, of, [y.Length], y), I64([indices.Length], indices)],
            genericTypes: new() { ["T"] = t });

    private static bool PlainMaxPool(DType t, Func<double, object> of, double[] x, params double[] y)
        => AutoTest.AdvancedTestGraph<MaxPoolValuesCheck>([], [Typed(t, of, [1L, 1L, x.Length], x), Typed(t, of, [y.Length], y)],
            genericTypes: new() { ["T"] = t });

    [Fact]
    public void TestSmallIntegerMaxPoolWindowsAtTheLowestValueTakeTheirFirstMaximum()
    {
        Assert.True(LowestWindows(DType.Int8, AsInt8, IntegerWindows, IntegerWindowMaxima, IntegerWindowPositions));
        Assert.True(LowestWindows(DType.UInt8, AsUInt8, IntegerWindows, IntegerWindowMaxima, IntegerWindowPositions));
    }

    // #437: ONNX Runtime's float MaxPool gives a window at or below the lowest finite value a wrong index and value
    [Fact(Skip = "#437: ONNX Runtime's float MaxPool gives a window at or below the lowest finite value a wrong index and value")]
    public void TestFloatMaxPoolWindowsAtOrBelowTheLowestFiniteValueTakeTheirFirstMaximum()
    {
        Assert.True(LowestWindows(DType.Float16, AsFloat16, FloatWindows, FloatWindowMaxima, FloatWindowPositions));
        Assert.True(LowestWindows(DType.Float32, AsFloat32, FloatWindows, FloatWindowMaxima, FloatWindowPositions));
        Assert.True(LowestWindows(DType.Float64, AsFloat64, FloatWindows, FloatWindowMaxima, FloatWindowPositions));
        Assert.True(AutoTest.AdvancedTestGraph<NegativeInfinityWindowMaxPoolIndicesValues>([],
            [F32([1L, 1L, 5L], float.NegativeInfinity, float.NegativeInfinity, 5f, float.NegativeInfinity, float.NegativeInfinity)],
            expected: [0, 0, 2, 2, 3, 4]));
    }

    // #426: ONNX Runtime's MaxPool gives a window of only -inf the lowest finite value instead of -inf
    [Fact(Skip = "#426: ONNX Runtime's MaxPool gives a window of only -inf the lowest finite value instead of -inf")]
    public void TestMaxPoolWithoutIndicesGivesANegativeInfinityWindowNegativeInfinity()
    {
        Assert.True(PlainMaxPool(DType.Float16, AsFloat16, [N, N, L, N], N, L, L));
        Assert.True(PlainMaxPool(DType.Float32, AsFloat32, [N, N, L, N], N, L, L));
        Assert.True(PlainMaxPool(DType.Float64, AsFloat64, [N, N, L, N], N, L, L));
    }

    internal static readonly double[] ConvTransposeSameStridedPastTheKernel = [0.5, 1.5, 0.5, 0.5, 2.5, 0.5, 0.5, 3.5, 0.5,
        0.5, 1.5, 0.5, 0.5, 2.5, 0.5, 0.5, 3.5, 0.5, 0.5, 0.5, 1.5, 0.5, 0.5, 0.5, 2.5, 0.5, 0.5, 0.5, 3.5, 0.5,
        0.5, 1.5, 0.5, 0.5, 0.5, 2.5, 0.5, 0.5, 0.5, 3.5, 0.5, 0.5, 0.5, 1.5, 10.5, 0.5, 0.5, 2.5, 20.5, 0.5, 0.5, 3.5, 30.5, 0.5,
        1.5, 10.5, 0.5, 0.5, 2.5, 20.5, 0.5, 0.5, 3.5, 30.5, 0.5, 0.5];

    [Fact]
    public void TestConvTransposeSameStridedPastTheKernelExtendsTheOutputAsTheSpecDoes()
        => Assert.True(AutoTest.AdvancedTestGraph<ConvTransposeSameStridedPastTheKernelValues>([],
            [F32([1L, 1L, 3L], 1f, 2f, 3f)], expected: ConvTransposeSameStridedPastTheKernel));

    [Fact]
    public void TestConvTransposeOutputShapeBeyondTheFullExtentIsRefused()
    {
        var ones = F32([1L, 1L, 2L, 2L], 1f, 1f, 1f, 1f);
        var ex = Assert.Throws<OnnxNodeException>(
            () => AutoTest.AdvancedTestGraph<ConvTransposeOversizedOutputShapeValues>([], [ones, ones, F32([1L], 0f)]));
        Assert.Contains("ConvTranspose", ex.Message);
        Assert.Contains("output_shape [6, 6]", ex.Message);
        Assert.Contains("full extent [4, 4]", ex.Message);
    }

    [Fact]
    public void TestConvTransposeOutputShapeOnePastTheFullExtentZeroExtendsTheEnd()
        => Assert.True(AutoTest.AdvancedTestGraph<ConvTransposeOutputShapeOnePastTheFullExtentValues>([],
            [F32([1L, 1L, 3L, 3L], [.. Enumerable.Range(0, 9).Select(i => (float)i)]), F32([1L, 1L, 3L, 3L], [.. Enumerable.Repeat(1f, 9)])],
            expected: [0, 0, 1, 1, 3, 2, 2, 0, 0, 0, 1, 1, 3, 2, 2, 0, 0, 0, 1, 1, 3, 2, 2, 0,
                3, 3, 7, 4, 9, 5, 5, 0, 3, 3, 7, 4, 9, 5, 5, 0, 3, 3, 7, 4, 9, 5, 5, 0,
                6, 6, 13, 7, 15, 8, 8, 0, 6, 6, 13, 7, 15, 8, 8, 0, 6, 6, 13, 7, 15, 8, 8, 0,
                0, 0, 0, 0, 0, 0, 0, 0]));
}
