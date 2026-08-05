using Shorokoo.Core.Inference;
using static Shorokoo.Tests.Utils.QeeAudit;

namespace Shorokoo.Tests;

/// <summary>
/// Phase 4 QEE-A6 audit batch: sequence, optional, string, signal and control-flow
/// families (ONNX opset 21). Each module in QeeSeqStringSignalAuditModules.cs is
/// self-checking on values (where QEE computes them) and on inferred shapes (via
/// ShapeTensor). Modules built on Shorokoo-internal op codes or @string runtime inputs
/// have no ORT-comparable data path, so they run through
/// <see cref="QeeAudit.QeeOnlyTyped{TModule}"/>.
/// </summary>
[Trait("Domain", "Inference")]
[Trait("Purpose", "Coverage")]
public class QeeSeqStringSignalAuditTests
{
    private static readonly TensorData FloatMat2x3 = F32([2L, 3L], 1f, 2f, 3f, 4f, 5f, 6f);

    [Fact]
    public void TestQeeSequenceOptionalAndReverseSequenceAudits()
    {
        Assert.True(QeeAudit.Check<QeeSequenceCoreAuditCheck>(FloatMat2x3, I64([], 0L)));
        Assert.True(QeeAudit.Check<QeeSplitToSeqConcatAuditCheck>(FloatMat2x3, I64([], 0L)));
        Assert.True(QeeAudit.QeeOnly<QeeSplitKeepdimsInteractionAuditCheck>(FloatMat2x3, I64([], 0L)));
        Assert.True(QeeAudit.Check<QeeReverseSequenceAuditCheck>(F32([4L, 4L],
            1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f)));
        Assert.True(QeeAudit.Check<QeeOptionalAuditCheck>(F32([3L], 1f, 2f, 3f)));
    }

    [Fact]
    public void TestQeeSignalStringAndInternalControlFlowAudits()
    {
        Assert.True(QeeAudit.Check<QeeWindowValueAuditCheck>(I64([], 8L)));
        Assert.True(QeeAudit.Check<QeeDftStftMelAuditCheck>(
            F32([1L, 4L, 1L], 1f, 2f, 3f, 4f),
            F32([1L, 16L, 1L], 0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f),
            I64([], 4L),
            F32([4L], 1f, 1f, 1f, 1f)));
        Assert.True(QeeAudit.Check<QeeTfIdfShapeAuditCheck>(I64([4L], 1L, 2L, 3L, 4L)));
        Assert.True(QeeAudit.QeeOnlyTyped<QeeStringOpsAuditCheck>(
            Strs([2L], "Hello World", "the quick fox"), Strs([2L], "A", "B")));
        Assert.True(QeeAudit.QeeOnlyTyped<QeeInternalControlFlowAuditCheck>(
            FloatMat2x3,
            F32([1L, 1L, 4L, 4L], 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f),
            F32([1L, 1L, 2L, 2L], 1f, 0f, 0f, 1f),
            F32([1L], 0f)));
    }
}
