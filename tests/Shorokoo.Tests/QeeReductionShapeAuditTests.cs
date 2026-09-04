using Shorokoo.Core.Inference;
using static Shorokoo.Tests.Utils.QeeAudit;

namespace Shorokoo.Tests;

/// <summary>
/// Phase 4 QEE-A3 audit batch: reductions plus the shape/data-movement family (ONNX opset
/// 21). Each module in QeeReductionShapeAuditModules.cs compares every audited op's output
/// values (and inferred shapes via ShapeTensor) against spec-expected constants and returns
/// a single Scalar&lt;bit&gt;; <see cref="QeeAudit.Check{TModule}"/> validates that bit
/// under both real ONNX Runtime execution and the <see cref="QuickExecutionEngine"/>.
/// </summary>
[Trait("Domain", "Inference")]
[Trait("Purpose", "Coverage")]
public class QeeReductionShapeAuditTests
{
    [Fact]
    public void TestQeeReduceArgCumSumAndReshapeFamilyValueAudits()
    {
        Assert.True(QeeAudit.Check<QeeReduceValueAuditCheck>(
            F32([2L, 3L], 1f, 2f, 3f, 4f, 5f, 6f),
            I64([2L, 3L], 1L, -2L, 3L, 4L, 5L, -6L)));
        Assert.True(QeeAudit.Check<QeeArgCumSumValueAuditCheck>(F32([2L, 3L], 1f, 3f, 3f, 2f, 0f, 2f)));
        Assert.True(QeeAudit.Check<QeeReshapeFamilyValueAuditCheck>(
            F32([2L, 3L, 4L], [.. Enumerable.Range(0, 24).Select(i => (float)i)])));
        Assert.True(QeeAudit.Check<QeeSliceGatherValueAuditCheck>(F32([3L, 4L],
            0f, 1f, 2f, 3f, 10f, 11f, 12f, 13f, 20f, 21f, 22f, 23f)));
    }

    // Fails: the QuickExecutionEngine does not resolve the audit bit for a graph that slices a
    // tensor to zero elements and reduces it. ONNX Runtime runs the same graph correctly
    // (QeeAudit.OrtOnly passes), so the reduction's identity element is not in question.
    [Fact]
    public void TestQeeFoldsAReductionOverAnEmptyTensorToTheIdentity()
        => Assert.True(QeeAudit.Check<QeeEmptyReduceIdentityCheck>(
            F32([2L, 3L], 1f, 2f, 3f, 4f, 5f, 6f)));

    [Fact]
    public void TestQeeScatterPadSplitConcatTileAndOneHotValueAudits()
    {
        Assert.True(QeeAudit.Check<QeeScatterPadValueAuditCheck>(F32([2L, 3L], 1f, 2f, 3f, 4f, 5f, 6f)));
        Assert.True(QeeAudit.Check<QeeSplitConcatTileSpaceValueAuditCheck>(
            F32([7L], 1f, 2f, 3f, 4f, 5f, 6f, 7f)));
        Assert.True(QeeAudit.Check<QeeOneHotTriluNonZeroValueAuditCheck>(I64([4L], 1L, 3L, -2L, 5L)));
    }
}
