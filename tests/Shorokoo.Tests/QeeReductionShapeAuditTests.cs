using Shorokoo.Core.Interpreter;
using static Shorokoo.Tests.Utils.QeeAudit;

namespace Shorokoo.Tests;

/// <summary>
/// QEE audit batch: reductions plus the shape/data-movement family (ONNX opset
/// 21). Each module in QeeReductionShapeAuditModules.cs compares every audited op's output
/// values (and inferred shapes via ShapeTensor) against spec-expected constants and returns
/// a single Scalar&lt;bit&gt;; <see cref="QeeAudit.Check{TModule}"/> validates that bit
/// under both real ONNX Runtime execution and the <see cref="QuickExecutionEngine"/>.
/// <see cref="QeeAudit.OrtOnly{TModule}"/> drives the one module whose bit QEE cannot fold
/// by construction, and still rejects an output QEE leaves untyped.
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

    [Fact]
    public void TestQeeReductionsWithoutAnIdentityDeclineAnEmptyAxisWithoutInvalidatingTheOutput()
        => Assert.True(QeeAudit.OrtOnly<QeeEmptyReduceNoIdentityCheck>(
            F32([2L, 3L], 1f, 2f, 3f, 4f, 5f, 6f)));

    [Fact]
    public void TestQeeScatterPadSplitConcatTileAndOneHotValueAudits()
    {
        Assert.True(QeeAudit.Check<QeeScatterPadValueAuditCheck>(F32([2L, 3L], 1f, 2f, 3f, 4f, 5f, 6f)));
        Assert.True(QeeAudit.Check<QeeSplitConcatTileSpaceValueAuditCheck>(
            F32([7L], 1f, 2f, 3f, 4f, 5f, 6f, 7f)));
        Assert.True(QeeAudit.Check<QeeOneHotTriluNonZeroValueAuditCheck>(I64([4L], 1L, 3L, -2L, 5L)));
        Assert.True(QeeAudit.Check<QeeScatterGatherNdEdgeValueAuditCheck>(F32([2L, 3L], 1f, 2f, 3f, 4f, 5f, 6f)));
    }

    [Fact]
    public void TestInt64ReduceMaxAndMinOverAnEmptyAxisYieldTheTypeExtremes()
        => Assert.True(AutoTest.AdvancedTestGraph<EmptyInt64ReduceMaxMinValues>([],
            [F32([2L, 3L], 1f, 2f, 3f, 4f, 5f, 6f)],
            expected: [long.MinValue, long.MaxValue]));

    [Fact]
    public void TestIntegerAndBoolReduceMaxAndMinOverAnEmptyGroupYieldTheTypeExtremes()
        => Assert.True(AutoTest.AdvancedTestGraph<EmptyIntegerReduceMaxMinValues>([],
            [F32([2L, 3L], 1f, 2f, 3f, 4f, 5f, 6f)],
            expected: [int.MinValue, int.MaxValue, sbyte.MinValue, sbyte.MaxValue, byte.MinValue, byte.MaxValue, 0, 1,
                long.MinValue, long.MaxValue, .. Enumerable.Repeat((double)long.MinValue, 6), .. Enumerable.Repeat((double)long.MaxValue, 6),
                long.MinValue, long.MaxValue, 3, 6, 1, 4]));

    // #409: ONNX Runtime ignores noop_with_empty_axes on an empty input and reduces every axis.
    [Fact(Skip = "#409: ONNX Runtime ignores noop_with_empty_axes on an empty input and reduces every axis")]
    public void TestANoopReductionPassesAnEmptyInputThrough()
        => Assert.True(AutoTest.AdvancedTestGraph<EmptyNoopReduceShape>([],
            [F32([2L, 3L], 1f, 2f, 3f, 4f, 5f, 6f)],
            expected: [2, 0]));

    // #411: ONNX Runtime's float16 ReduceSumSquare and ReduceL1 crash the process on an empty input with no axes.
    [Fact(Skip = "#411: ONNX Runtime's float16 ReduceSumSquare and ReduceL1 over an empty input with no axes crash the process")]
    public void TestFloat16ReduceSumSquareAndL1OverAnEmptyInputGiveZero()
        => Assert.True(AutoTest.AdvancedTestGraph<EmptyFloat16ReduceAllValues>([],
            [F32([2L, 3L], 1f, 2f, 3f, 4f, 5f, 6f)],
            expected: [0, 0]));
}
