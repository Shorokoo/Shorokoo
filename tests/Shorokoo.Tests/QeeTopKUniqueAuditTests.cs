using Shorokoo.Core.Inference;
using static Shorokoo.Tests.Utils.QeeAudit;

namespace Shorokoo.Tests;

/// <summary>
/// Phase-4 follow-up coverage for TopK and Unique, which the family audit batches missed.
/// The value case runs under both ORT and the <see cref="QuickExecutionEngine"/>; the
/// Unique axis form is data-dependent, so its shape case is ORT-validated only.
/// </summary>
[Trait("Domain", "Inference")]
[Trait("Purpose", "Coverage")]
public class QeeTopKUniqueAuditTests
{
    [Fact]
    public void TestQeeTopKUniqueValueAndShapeAudits()
    {
        Assert.True(QeeAudit.Check<QeeTopKUniqueValueAuditCheck>(
            F32([2L, 4L], 3f, 1f, 4f, 1f, 5f, 9f, 2f, 6f)));
        Assert.True(QeeAudit.OrtOnly<QeeTopKUniqueShapeAuditCheck>(
            F32([3L, 2L], 1f, 2f, 1f, 2f, 3f, 4f)));
    }
}
