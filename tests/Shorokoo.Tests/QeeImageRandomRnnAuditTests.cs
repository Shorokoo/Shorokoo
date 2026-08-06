using Shorokoo.Runtime;
using Shorokoo.Core.Inference;
using static Shorokoo.Tests.Utils.QeeAudit;

namespace Shorokoo.Tests;

/// <summary>
/// Phase 4 QEE-A5 audit batch: image/geometry, random/generator and recurrent families
/// (ONNX opset 21). Each module in QeeImageRandomRnnAuditModules.cs is self-checking
/// (single Scalar&lt;bit&gt;). Outputs whose shapes legitimately stay unknown at QEE time
/// (NonMaxSuppression's data-dependent n, ImageDecoder's data-dependent H/W, Constant
/// string tensors) are asserted by direct <see cref="RuntimeTensor"/> inspection instead —
/// the audit contract is that they degrade to a null shape with the correct rank/dtype,
/// never to guessed or negative dims.
/// </summary>
[Trait("Domain", "Inference")]
[Trait("Purpose", "Coverage")]
public class QeeImageRandomRnnAuditTests
{
    private static readonly TensorData NmsBoxes = F32([1L, 4L, 4L],
        0.0f, 0.0f, 1.0f, 1.0f,
        0.0f, 0.1f, 1.0f, 1.1f,
        5.0f, 5.0f, 6.0f, 6.0f,
        5.0f, 5.1f, 6.0f, 6.1f);

    private static readonly TensorData NmsScores = F32([1L, 1L, 4L], 0.9f, 0.8f, 0.7f, 0.6f);

    private static readonly TensorData RecurrentX = F32Zeros([4L, 2L, 3L]);

    [Fact]
    public void TestQeeImageGeometryShapeAudits()
    {
        var x8 = F32Zeros([1L, 1L, 8L, 8L]);
        Assert.True(QeeAudit.Check<QeeResizeShapeAuditCheck>(x8));
        Assert.True(QeeAudit.QeeOnly<QeeResizeNegativeAxesAuditCheck>(x8));
        Assert.True(QeeAudit.Check<QeeUpsampleAffineGridSampleAuditCheck>(F32Zeros([1L, 2L, 4L, 4L])));
        Assert.True(QeeAudit.Check<QeeAffineGridSample5DAuditCheck>(F32Zeros([1L, 1L, 3L, 4L, 4L])));
        Assert.True(QeeAudit.Check<QeeRoiAlignShapeAuditCheck>(
            F32Zeros([1L, 2L, 8L, 8L]),
            F32([3L, 4L], 0f, 0f, 4f, 4f, 1f, 1f, 6f, 6f, 2f, 2f, 7f, 7f),
            I64([3L], 0L, 0L, 0L)));
        Assert.True(QeeAudit.Check<QeeCol2ImCenterCropPadAuditCheck>(
            F32Zeros([1L, 8L, 12L]),
            F32([3L, 5L], 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f)));
    }

    [Fact]
    public void TestQeeNonMaxSuppressionAndImageDecoderDegradeToRankOnly()
    {
        Assert.True(QeeAudit.OrtOnly<QeeNmsOrtShapeAuditCheck>(NmsBoxes, NmsScores));
        Assert.True(QeeAudit.Check<QeeNmsEmptyAuditCheck>(NmsBoxes, NmsScores));

        var nms = Assert.IsType<RuntimeTensor>(
            QeeAudit.Outputs<QeeNmsRankOnlyCheck>(NmsBoxes, NmsScores).Single());
        Assert.Equal(DType.Int64, nms.DType);
        Assert.Null(nms.Shape);
        Assert.Equal(2, nms.Rank);
        Assert.Equal(2, nms.MaxRank);
        Assert.NotNull(nms.MaxShape);
        Assert.Equal([2L, 3L], nms.MaxShape!.Dims);

        var img = Assert.IsType<RuntimeTensor>(
            QeeAudit.Outputs<QeeImageDecoderCheck>(U8([4L], (byte)0, (byte)0, (byte)0, (byte)0)).Single());
        Assert.Equal(DType.UInt8, img.DType);
        Assert.Null(img.Shape);
        Assert.Equal(3, img.Rank);
        Assert.Equal(3, img.MaxRank);
    }

    [Fact]
    public void TestQeeRandomGeneratorAndRecurrentShapeAudits()
    {
        Assert.True(QeeAudit.CheckWith<QeeRandomFamilyAuditCheck>(
            [F32([2L, 3L], 0.1f, 0.5f, 0.9f, 0.3f, 0.7f, 0.2f),
             F32([2L, 4L], 0.1f, 0.4f, 0.3f, 0.2f, 0.25f, 0.25f, 0.25f, 0.25f)],
            testCsRoundtrip: false));
        Assert.True(QeeAudit.CheckWith<QeeRandomSeededDeterminismCheck>(
            [], qee: QeeStrictness.None, testCsRoundtrip: false));
        Assert.True(QeeAudit.Check<QeeRangeConstantOfShapeAuditCheck>());

        var strings = QeeAudit.Outputs<QeeConstantStringCheck>();
        var cs = Assert.IsType<RuntimeTensor>(strings[0]);
        Assert.Equal(DType.String, cs.DType);
        Assert.Empty(cs.Shape!.Dims);
        Assert.Equal(["hello"], cs.StringData!.Value.ToArray());
        var css = Assert.IsType<RuntimeTensor>(strings[1]);
        Assert.Equal(DType.String, css.DType);
        Assert.Equal([3L], css.Shape!.Dims);
        Assert.Equal(["a", "b", "c"], css.StringData!.Value.ToArray());

        Assert.True(QeeAudit.Check<QeeRnnShapeAuditCheck>(RecurrentX));
        Assert.True(QeeAudit.QeeOnly<QeeRecurrentQeeOnlyShapeAuditCheck>(RecurrentX));
        Assert.True(QeeAudit.Check<QeeGruShapeAuditCheck>(RecurrentX));
        Assert.True(QeeAudit.Check<QeeLstmShapeAuditCheck>(RecurrentX, I32([2L], 4, 4)));
    }
}
