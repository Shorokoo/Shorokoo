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
    }

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
