using Shorokoo.Core.Inference;
using static Shorokoo.Tests.Utils.QeeAudit;

namespace Shorokoo.Tests;

/// <summary>
/// Phase 4 QEE-A4 audit batch: normalization, softmax, linear-algebra and quantization
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
    }
}
