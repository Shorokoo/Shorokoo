using Shorokoo.Core.Inference;
using static Shorokoo.Tests.Utils.QeeAudit;

namespace Shorokoo.Tests;

/// <summary>
/// Phase 4 QEE-A2 audit batch: elementwise, comparison, logical and bitwise families
/// (ONNX opset 21). Each module in QeeElementwiseAuditModules.cs compares every audited
/// op's output against spec-expected constants and returns a single Scalar&lt;bit&gt;;
/// <see cref="QeeAudit.Check{TModule}"/> validates that bit under both real ONNX Runtime
/// execution and the <see cref="QuickExecutionEngine"/>.
/// </summary>
[Trait("Domain", "Inference")]
[Trait("Purpose", "Coverage")]
public class QeeElementwiseAuditTests
{
    [Fact]
    public void TestQeeUnaryAndBinaryElementwiseValueAudits()
    {
        Assert.True(QeeAudit.Check<QeeTrigExpLogValueAuditCheck>(
            F32([3L], 0.5f, -0.25f, 0.75f), F32([3L], 1f, 2f, 4f)));
        Assert.True(QeeAudit.Check<QeeUnaryRoundingValueAuditCheck>(
            F32([5L], -1.5f, -0.5f, 0.5f, 1.5f, 2.5f)));
        Assert.True(QeeAudit.Check<QeeActivationValueAuditCheck>(
            F32([5L], -2f, -0.5f, 0f, 0.5f, 2f), F32([4L], -2.7f, -1f, 0.5f, 2.7f)));
        Assert.True(QeeAudit.Check<QeeBinaryArithValueAuditCheck>(
            F32([3L], 7.5f, -5.5f, 9.25f), F32([3L], 2f, 3f, -4f),
            I64([3L], 7L, -7L, 9L), I64([3L], 2L, 2L, -4L)));
    }

    [Fact]
    public void TestQeeCompareLogicBitwiseWhereAndSliceValueAudits()
    {
        Assert.True(QeeAudit.Check<QeeCompareLogicValueAuditCheck>(
            F32([3L], 1f, 2f, 3f), F32([3L], 2f, 2f, 2f),
            Bits([4L], true, false, true, false), Bits([4L], true, true, false, false)));
        Assert.True(QeeAudit.Check<QeeBitwiseValueAuditCheck>(
            I64([3L], 12L, 10L, 15L), I64([3L], 10L, 5L, 3L)));
        Assert.True(QeeAudit.Check<QeeMiscElementwiseValueAuditCheck>(
            F32([4L], 1f, -1f, 0f, 2f), F32([4L], 0f, 0f, 0f, 1f),
            Bits([4L], true, false, true, false)));
        Assert.True(QeeAudit.QeeOnly<QeeWhereBoolValueAuditCheck>(
            Bits([4L], true, false, true, false)));
        Assert.True(QeeAudit.Check<QeeSliceReverseValueAuditCheck>(F32([3L], 1f, 2f, 3f)));
    }
}

/// <summary>
/// Regression pins for wrapper-API value bugs once present in Scalar.cs / Vector.cs:
/// Scalar's left-shift-by-primitive operator delegating to the right shift, and
/// Scalar/Vector Min/Max ignoring their <c>params others</c> argument. Both are fixed;
/// each pin runs a self-checking module through <see cref="QuickExecutionEngine"/> and
/// asserts the value the public wrapper must produce.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class ScalarVectorWrapperBugPinTests
{
    [Fact]
    public void TestScalarShiftLeftByPrimitiveAndScalarVectorMinMaxForwardOthers()
    {
        Assert.True(QeeAudit.QeeOnly<ScalarShiftLeftPrimitiveBugPinCheck>(I64([], 4L)));
        Assert.True(QeeAudit.QeeOnly<ScalarMinMaxOthersBugPinCheck>(F32([], 5f), F32([], 2f)));
        Assert.True(QeeAudit.QeeOnly<VectorMinMaxOthersBugPinCheck>(
            F32([2L], 1f, 5f), F32([2L], 3f, 2f)));
    }
}
