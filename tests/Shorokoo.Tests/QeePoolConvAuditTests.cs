using Shorokoo.Core.Inference;
using static Shorokoo.Tests.Utils.QeeAudit;

namespace Shorokoo.Tests;

/// <summary>
/// Phase 4 QEE-A1 audit batch: pooling and convolution families (ONNX opset 21). Each
/// module in QeePoolConvAuditModules.cs compares the ShapeTensor() of every op result
/// against the spec-expected dims and returns a single Scalar&lt;bit&gt;;
/// <see cref="QeeAudit.Check{TModule}"/> validates that bit under both real ONNX Runtime
/// execution and the <see cref="QuickExecutionEngine"/>'s own shape inference.
/// </summary>
[Trait("Domain", "Inference")]
[Trait("Purpose", "Coverage")]
public class QeePoolConvAuditTests
{
    private static readonly TensorData Image1x1x10x10 = F32Zeros([1L, 1L, 10L, 10L]);
    private static readonly TensorData Image1x3x10x10 = F32Zeros([1L, 3L, 10L, 10L]);

    [Fact]
    public void TestQeePoolingShapeAudits()
    {
        Assert.True(QeeAudit.Check<QeeMaxPoolShapeAuditCheck>(Image1x1x10x10));
        Assert.True(QeeAudit.Check<QeeAveragePoolShapeAuditCheck>(Image1x1x10x10));
        Assert.True(QeeAudit.Check<QeeLpPoolGlobalPoolShapeAuditCheck>(Image1x3x10x10));
        Assert.True(QeeAudit.Check<QeeMaxRoiPoolShapeAuditCheck>(
            F32Zeros([1L, 2L, 8L, 8L]),
            F32([2L, 5L], 0f, 0f, 0f, 7f, 7f, 0f, 1f, 1f, 6f, 6f)));
        Assert.True(QeeAudit.Check<QeeMaxUnpoolShapeAuditCheck>(
            F32([1L, 1L, 2L, 2L], 6f, 8f, 14f, 16f),
            I64([1L, 1L, 2L, 2L], 0L, 2L, 5L, 7L)));
    }

    [Fact]
    public void TestQeeConvolutionShapeAudits()
    {
        Assert.True(QeeAudit.Check<QeeConvShapeAuditCheck>(F32Zeros([1L, 4L, 9L, 9L])));
        Assert.True(QeeAudit.Check<QeeConvTransposeShapeAuditCheck>(F32Zeros([1L, 2L, 5L, 5L])));
        Assert.True(QeeAudit.Check<QeeQuantizedConvShapeAuditCheck>(
            I8Zeros([1L, 1L, 7L, 7L]), I8Zeros([1L, 1L, 3L, 3L]),
            I8([], (sbyte)0), I8([], (sbyte)0),
            F32([], 0.5f), F32([], 0.25f), F32([], 0.5f), I8([], (sbyte)0)));
        Assert.True(QeeAudit.CheckWith<QeeDeformConvShapeAuditCheck>(
            [F32Zeros([1L, 1L, 4L, 4L]),
             F32([1L, 1L, 2L, 2L], 1f, 0f, 0f, 1f),
             F32Zeros([1L, 8L, 3L, 2L]),
             F32([1L], 0f)],
            testOnnxRoundtrip: false, testCsRoundtrip: false));
    }
}
