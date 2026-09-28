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
    public void TestIntegerAndBoolReduceMaxAndMinOverAnEmptyGroupYieldTheTypeExtremes()
        => Assert.True(AutoTest.AdvancedTestGraph<EmptyIntegerReduceMaxMinCheck>([], [F32([2L, 3L], 1f, 2f, 3f, 4f, 5f, 6f)]));

    [Fact]
    public void TestAGenericReduceMaxAndMinSpecialisedToAnIntegerOrBoolYieldTheTypeExtremesOverAnEmptyGroup()
    {
        Assert.True(GenericMaxMin(DType.Int64, I64([1L, 0L]), long.MinValue, long.MaxValue));
        Assert.True(GenericMaxMin(DType.Int64, I64([2L, 2L], 1L, 5L, 3L, 2L), 5L, 3L, 1L, 2L));
        Assert.True(GenericMaxMin(DType.Int32, I32([1L, 0L]), int.MinValue, int.MaxValue));
        Assert.True(GenericMaxMin(DType.Int8, I8([1L, 0L]), sbyte.MinValue, sbyte.MaxValue));
        Assert.True(GenericMaxMin(DType.UInt8, U8([1L, 0L]), byte.MinValue, byte.MaxValue));
        Assert.True(GenericMaxMin(DType.Bool, Bits([1L, 0L]), 0L, 1L));
        Assert.True(GenericMaxMin(DType.Bool, Bits([2L, 2L], true, false, false, false), 1L, 0L, 0L, 0L));
    }

    [Fact]
    public void TestAReduceMaxOrMinThatCannotMeetAnEmptyGroupBuildsThePlainOperator()
    {
        Tensor<float32> f = InputTensor<float32>("f", rank: 2);
        Tensor<float16> h = InputTensor<float16>("h", rank: 2);
        Tensor<int64> i = InputTensor<int64>("i", rank: 2);
        Tensor<int64> c = Tensor([2L, 2L], 1L, 2L, 3L, 4L);
        Tensor<bit> b = Tensor([2L], true, false);
        Assert.True(IsPlain(f, f.Reduce(ReduceKind.Max, Vector(1L))));
        Assert.True(IsPlain(f, f.Reduce(ReduceKind.Min)));
        Assert.True(IsPlain(h, h.Reduce(ReduceKind.Max, Vector(-1L), keepDims: true)));
        Assert.True(IsPlain(c, c.Reduce(ReduceKind.Min, Vector(1L))));
        Assert.True(IsPlain(b, b.Reduce(ReduceKind.Max)));
        Assert.True(IsPlain(i, NN.Reduce(ReduceKind.Max, i, null, true, true)));
    }

    [Fact]
    public void TestAReductionOverANegativeAxisOfAnEmptyInputHasTheSpecShape()
        => Assert.True(AutoTest.AdvancedTestGraph<EmptyReduceNegativeAxisShapes>([], [F32([2L, 3L], 1f, 2f, 3f, 4f, 5f, 6f)],
            expected: [3, 1, 0, 3, 1, 0]));

    [Fact]
    public void TestTheRawReductionOverANegativeAxisOfAnEmptyInputHasTheSpecShape()
        => Assert.True(AutoTest.AdvancedTestGraph<EmptyRawReduceNegativeAxisShapes>([], [F32([2L, 3L], 1f, 2f, 3f, 4f, 5f, 6f)],
            expected: [3, 1, 0, 3, 1, 0]));

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

    private static bool GenericMaxMin(DType t, TensorData x, params long[] expected)
        => AutoTest.AdvancedTestGraph<GenericReduceMaxMinCheck>([], [x, I64([expected.Length], expected)],
            genericTypes: new() { ["T"] = t });

    private static bool IsPlain<T>(Tensor<T> x, Tensor<T> reduced) where T : IVarType
        => ((Variable)reduced).OwningNode is { OpCode: OpCodes.REDUCE_MAX or OpCodes.REDUCE_MIN } n && n.Inputs[0] == (Variable)x;
}
