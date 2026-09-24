using Shorokoo.Core.Interpreter;
using static Shorokoo.Tests.Utils.QeeAudit;

namespace Shorokoo.Tests;

/// <summary>
/// QEE audit batch: normalization, softmax, linear-algebra and quantization
/// families (ONNX opset 21). Each module in QeeNormLinalgAuditModules.cs compares every
/// audited op's output values (and inferred shapes via ShapeTensor) against spec-expected
/// constants and returns a single Scalar&lt;bit&gt;; <see cref="QeeAudit.Check{TModule}"/>
/// validates that bit under both real ONNX Runtime execution and the
/// <see cref="QuickExecutionEngine"/>.
/// </summary>
[Trait("Domain", "Inference")]
[Trait("Purpose", "Coverage")]
public class QeeNormLinalgAuditTests
{
    [Fact]
    public void TestQeeNormSoftmaxLinalgAndQuantizationValueAudits()
    {
        Assert.True(QeeAudit.Check<QeeNormalizationAuditCheck>(F32([2L, 2L], 1f, 2f, 3f, 4f)));
        Assert.True(QeeAudit.Check<QeeSoftmaxFamilyValueAuditCheck>(
            F32([2L, 3L], 1f, 2f, 3f, 3f, 2f, 1f)));
        Assert.True(QeeAudit.Check<QeeLossDropoutAuditCheck>(
            F32([2L, 3L], 1f, 2f, 3f, 4f, 5f, 6f), I64([2L], 0L, 2L)));
        Assert.True(QeeAudit.Check<QeeMatMulGemmValueAuditCheck>(F32([2L, 3L], 1f, 2f, 3f, 4f, 5f, 6f)));
        Assert.True(QeeAudit.Check<QeeEinsumDetAuditCheck>(F32([2L, 3L], 1f, 2f, 3f, 4f, 5f, 6f)));
        Assert.True(QeeAudit.Check<QeeQuantizationValueAuditCheck>(
            F32([2L, 2L], 1.25f, -0.5f, 0.6f, 3.1f)));
        Assert.True(QeeAudit.Check<QeeNormLossVariantsAuditCheck>(
            F32Wave([2L, 4L, 3L, 2L]), I64([2L, 3L, 2L], 0L, 1L, 2L, 3L, 2L, 1L, 3L, 3L, 0L, 2L, 1L, 2L)));
    }

    [Fact]
    public void TestQuantizationRuntimeValueAudits()
    {
        Assert.True(QeeAudit.OrtOnly<QeeQuantizationRuntimeValueAuditCheck>(
            F32([2L, 4L], 0.625f, -1.3f, 2.2f, 40f, -0.1f, 3.75f, -50f, 0.875f),
            I8([2L, 4L], 10, -20, 30, -128, 127, 0, -3, 64)));
        Assert.True(QeeAudit.OrtOnly<QeeQLinearValueAuditCheck>(
            I8([2L, 3L], 1, -2, 3, 4, 0, -5),
            U8([2L, 3L], 125, 118, 140, 100, 130, 121),
            U8([1L, 2L, 6L, 6L], [.. Enumerable.Range(0, 72).Select(i => (byte)(i * 37 % 256))])));
    }

    [Fact]
    public void TestDequantizeLinearOfInt32WithoutZeroPointKeepsItsValuesThroughAReshape()
    {
        Assert.True(QeeAudit.Check<QeeDequantizeInt32ReshapeAuditCheck>(I32([3L], 1000, -6, 2)));
    }

    [Fact]
    public void TestDequantizeLinearWithoutZeroPointKeepsItsValuesThroughAReshapeOrATransposeForEveryInputType()
    {
        Assert.True(QeeAudit.Check<QeeDequantizeWithoutZeroPointReshapeTransposeAuditCheck>(
            I8([3L], 100, -6, 2), TensorData([3L], (short)1000, (short)-6, (short)2),
            TensorData([3L], (ushort)1000, (ushort)6, (ushort)2), I32([3L], 1000, -6, 2)));
        Assert.True(QeeAudit.Check<QeeDequantizeInt32VectorScaleReshapeAuditCheck>(I32([1L, 3L], 1000, -6, 2)));
        Assert.True(QeeAudit.Check<QeeDequantizeInt32PerAxisAuditCheck>(I32([2L, 3L], 10, -6, 2, 4, 0, -8)));
    }

    [Fact]
    public void TestLayerNormalizationOfRowsWithALargeMeanAgreesWithItsFunctionBody()
        => Assert.True(AutoTest.AdvancedTestGraph<LayerNormalizationOfALargeMeanCheck>([], LargeMeanRows));

    internal static TensorData[] LargeMeanRows =>
        [F32([4L, 256L], [.. Enumerable.Range(0, 1024).Select(i => 100f + 0.1f * MathF.Sin(1.7f * i))]),
         F32([256L], [.. Enumerable.Repeat(1f, 256)])];
}
