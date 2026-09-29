using Shorokoo.Runtime;
using Shorokoo.Core.Factory;
using Shorokoo.Core.Interpreter;
using Shorokoo.PyTorch.Cpu;
using static Shorokoo.Tests.Utils.QeeAudit;

namespace Shorokoo.Tests;

/// <summary>
/// QEE audit batch: image/geometry, random/generator and recurrent families
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
    private static TensorData NmsBoxes => F32([1L, 4L, 4L],
        0.0f, 0.0f, 1.0f, 1.0f,
        0.0f, 0.1f, 1.0f, 1.1f,
        5.0f, 5.0f, 6.0f, 6.0f,
        5.0f, 5.1f, 6.0f, 6.1f);

    private static TensorData NmsScores => F32([1L, 1L, 4L], 0.9f, 0.8f, 0.7f, 0.6f);

    private static TensorData RecurrentX => F32Zeros([4L, 2L, 3L]);

    private static TensorData SeqLens => I32([2L], 4, 2);

    private static TensorData EmptySequence => I32([2L], 4, 0);

    [Fact]
    public void TestQeeImageGeometryShapeAudits()
    {
        var x8 = F32Wave([1L, 1L, 8L, 8L]);
        Assert.True(QeeAudit.Check<QeeResizeShapeAuditCheck>(x8));
        Assert.True(QeeAudit.Check<QeeResizeNegativeAxesAuditCheck>(x8));
        Assert.True(QeeAudit.Check<QeeUpsampleAffineGridSampleAuditCheck>(F32Wave([1L, 2L, 4L, 4L])));
        Assert.True(QeeAudit.Check<QeeAffineGridSample5DAuditCheck>(F32Wave([1L, 1L, 3L, 4L, 4L])));
        Assert.True(QeeAudit.Check<QeeRoiAlignShapeAuditCheck>(
            F32Wave([1L, 2L, 8L, 8L]),
            F32([3L, 4L], 0f, 0f, 4f, 4f, 1f, 1f, 6f, 6f, 2f, 2f, 7f, 7f),
            I64([3L], 0L, 0L, 0L)));
        Assert.True(QeeAudit.Check<QeeCol2ImCenterCropPadAuditCheck>(
            F32Wave([1L, 8L, 12L]),
            F32([3L, 5L], 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f)));
        Assert.True(QeeAudit.Check<QeeResizeModesShapeAuditCheck>(F32Wave([1L, 2L, 5L, 7L])));
        Assert.True(QeeAudit.OrtOnly<QeeResizeUInt8ValueAuditCheck>(U8([1L, 1L, 1L, 8L], 0, 255, 255, 0, 0, 255, 0, 0)));
        Assert.True(QeeAudit.Check<QeeSamplingVariantsShapeAuditCheck>(
            F32Wave([1L, 2L, 5L, 6L]), F32Wave([1L, 8L, 2L, 3L]), F32Wave([1L, 12L, 12L])));
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

    private const string Png = "iVBORw0KGgoAAAANSUhEUgAAAAMAAAACCAIAAAASFvFNAAAAHUlEQVR42mP4z8DA8J+BgeE/E7eI3InpKf//MwAAPP4G/q61Bd4AAAAASUVORK5CYII=";

    private const string Jpeg = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/2wBDAQICAgICAgUDAwUKBwYHCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgr/wAARCAAIABADAREAAhEBAxEB/8QAFAABAAAAAAAAAAAAAAAAAAAABf/EABQQAQAAAAAAAAAAAAAAAAAAAAD/xAAUAQEAAAAAAAAAAAAAAAAAAAAI/8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAwDAQACEQMRAD8APFc2CBUBO//Z";

    private const string GreyJpeg = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/wAALCAAIAAgBAREA/8QAFAABAAAAAAAAAAAAAAAAAAAACP/EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8AGb//2Q==";

    private static TensorData Encoded(string base64)
    {
        var bytes = Convert.FromBase64String(base64);
        return U8([bytes.Length], bytes);
    }

    private static double[] Repeat(int times, params double[] run) => [.. Enumerable.Repeat(run, times).SelectMany(r => r)];

    [Fact]
    public void TestImageDecoderDecodesPngAndJpegToHwcInEveryPixelFormatOnTorch()
    {
        Assert.True(AutoTest.AdvancedTestGraph<QeeImageDecoderValueCheck>([],
            [Encoded(Png), Encoded(Jpeg), Encoded(GreyJpeg)],
            context: new ComputeContext(new TorchCpuBackend()),
            tolerance: 0,
            expected:
            [
                255, 0, 0, 0, 255, 0, 0, 0, 255, 10, 20, 30, 200, 150, 100, 255, 255, 255,
                0, 0, 255, 0, 255, 0, 255, 0, 0, 30, 20, 10, 100, 150, 200, 255, 255, 255,
                76, 150, 29, 18, 159, 255,
                .. Repeat(8, [.. Repeat(8, 200, 100, 50), .. Repeat(8, 128, 128, 128)]),
                .. Repeat(8, [.. Repeat(8, 124), .. Repeat(8, 128)]),
                .. Repeat(64, 77, 77, 77),
            ]));
    }

    [Fact]
    public void TestAnInt64RangeCountsItsElementsExactly() => Int64RangesCountTheirElementsExactly(ComputeContext.Default);

    internal static void Int64RangesCountTheirElementsExactly(ComputeContext c)
    {
        Assert.True(Ranges(c, 0L, (1L << 62) + 1L, 1L << 61, 0L, 1L << 61, 1L << 62));
        Assert.True(Ranges(c, 0L, -(1L << 62) - 1L, -(1L << 61), 0L, -(1L << 61), -(1L << 62)));
        Assert.True(Ranges(c, long.MinValue, long.MaxValue, 1L << 62, long.MinValue, -(1L << 62), 0L, 1L << 62));
        Assert.True(Ranges(c, long.MaxValue, long.MinValue, long.MinValue, long.MaxValue, -1L));
        Assert.True(Ranges(c, 9007199254740993L, 9007199254740995L, 1L, 9007199254740993L, 9007199254740994L));
        Assert.True(Ranges(c, 0L, 7L, 2L, 0L, 2L, 4L, 6L));
        Assert.True(Ranges(c, 5L, 5L, 1L));
        Assert.True(Ranges(c, 0L, 1L << 62, -1L));
        Assert.True(Ranges(c, 1L << 62, long.MinValue, 1L << 61));
    }

    [Fact]
    public void TestAnInt64RangeWithAUnitStepCountsItsElementsExactly() => Int64UnitStepRangesCountTheirElementsExactly(ComputeContext.Default);

    [Fact]
    public void TestAnIntegerRangeOfConstantsCountsItsElementsExactly() => Assert.True(AutoTest.AdvancedTestGraph<IntegerRangeOfConstantsCheck>([], []));

    internal static void Int64UnitStepRangesCountTheirElementsExactly(ComputeContext c)
    {
        Assert.True(UnitRanges(c, -3L, 2L, [-3L, -2L, -1L, 0L, 1L], [2L, 1L, 0L, -1L, -2L]));
        Assert.True(FromAConstantStart(c, (1L << 62) + 2L, 1L << 62, (1L << 62) + 1L));
        Assert.ThrowsAny<Exception>(() => Ranges(c, long.MinValue, long.MaxValue, 1L));
    }

    internal static void UnitStepRangesWhoseSpanWrapsTheirTypeAreEmpty(ComputeContext c)
    {
        Assert.True(UnitRanges(c, long.MaxValue - 2L, long.MinValue + 2L, [], []));
        Assert.True(UnitRanges(c, long.MaxValue, long.MinValue, [], []));
        Assert.True(FromAConstantStart(c, long.MinValue));
        Assert.True(UnitRanges32(c, int.MaxValue - 2, int.MinValue + 2, [], []));
        Assert.True(UnitRanges32(c, int.MaxValue, int.MinValue, [], []));
        Assert.True(AutoTest.AdvancedTestGraph<UnitStepRangeOfConstantsWrappingItsTypeCheck>([], [], context: c));
    }

    [Fact]
    public void TestTheQuickEngineGivesNoShapeToARangeOfMoreElementsThanAnInt64Counts()
    {
        var (s, l) = (InputScalar<int64>("s"), InputScalar<int64>("l"));
        var g = new InternalComputationGraph([s, l], [OnnxOp.Range(s, l, Scalar(1L))]);
        Shape? Count(long start, long limit) => ((RuntimeTensor)new QuickExecutionEngine().Run(g, TensorData(DType.Int64, [], start), TensorData(DType.Int64, [], limit))[g.Outputs[0]]).Shape;
        Assert.Null(Count(long.MinValue, long.MaxValue));
        Assert.Null(Count(-2L, long.MaxValue));
        Assert.Equal([long.MaxValue], Count(-1L, long.MaxValue - 1L)!.Dims);
    }

    [Fact]
    public void TestAnInt32RangeCountsItsElementsExactly() => Int32RangesCountTheirElementsExactly(ComputeContext.Default);

    internal static void Int32RangesCountTheirElementsExactly(ComputeContext c)
    {
        Assert.True(Ranges32(c, int.MinValue, int.MaxValue, 1 << 30, int.MinValue, -(1 << 30), 0, 1 << 30));
        Assert.True(Ranges32(c, int.MaxValue, int.MinValue, int.MinValue, int.MaxValue, -1));
        Assert.True(Ranges32(c, int.MaxValue - 2, int.MinValue + 2, 1));
        Assert.True(Ranges32(c, int.MinValue, int.MaxValue, -1));
        Assert.True(Ranges32(c, 5, -5, -3, 5, 2, -1, -4));
        Assert.True(UnitRanges32(c, -3, 2, [-3, -2, -1, 0, 1], [2, 1, 0, -1, -2]));
    }

    private static bool Ranges32(ComputeContext c, int start, int limit, int delta, params int[] expected)
        => AutoTest.AdvancedTestGraph<Int32RangeCheck>([],
            [I32([], start), I32([], limit), I32([], delta), I32([expected.Length], expected)], context: c);

    private static bool UnitRanges32(ComputeContext c, int start, int limit, int[] up, int[] down)
        => AutoTest.AdvancedTestGraph<Int32UnitStepRangeCheck>([],
            [I32([], start), I32([], limit), I32([up.Length], up), I32([down.Length], down)], context: c);

    private static bool UnitRanges(ComputeContext c, long start, long limit, long[] up, long[] down)
        => AutoTest.AdvancedTestGraph<Int64UnitStepRangeCheck>([],
            [I64([], start), I64([], limit), I64([up.Length], up), I64([down.Length], down)], context: c);

    private static bool FromAConstantStart(ComputeContext c, long limit, params long[] expected)
        => AutoTest.AdvancedTestGraph<Int64RangeFromAConstantStartCheck>([], [I64([], limit), I64([expected.Length], expected)], context: c);

    private static bool Ranges(ComputeContext c, long start, long limit, long delta, params long[] expected)
        => AutoTest.AdvancedTestGraph<Int64RangeCheck>([],
            [I64([], start), I64([], limit), I64([], delta), I64([expected.Length], expected)], context: c);

    [Fact]
    public void TestQeeRandomGeneratorAndRecurrentShapeAudits()
    {
        Assert.True(QeeAudit.CheckWith<QeeRandomFamilyAuditCheck>(
            [F32([2L, 3L], 0.1f, 0.5f, 0.9f, 0.3f, 0.7f, 0.2f),
             F32([2L, 4L], 0.1f, 0.4f, 0.3f, 0.2f, 0.25f, 0.25f, 0.25f, 0.25f)],
            testCsRoundtrip: false));
        Assert.True(QeeAudit.CheckWith<QeeRandomSeededDeterminismCheck>(
            [], qee: QeeStrictness.None, testCsRoundtrip: false));
        Assert.True(QeeAudit.OrtOnly<QeeDropoutAuditCheck>(F32([4L], 1f, 2f, 3f, 4f)));
        Assert.True(QeeAudit.OrtOnly<QeeKeyedRngValueAuditCheck>(F32Zeros([3L, 5L])));
        Assert.True(QeeAudit.OrtOnly<RtLoweredUniform>(F32Zeros([3L, 5L])));
        Assert.True(QeeAudit.Check<QeeRangeConstantOfShapeAuditCheck>());

        var strings = QeeAudit.Outputs<QeeConstantStringCheck>();
        var cs = Assert.IsType<RuntimeTensor>(strings[0]);
        Assert.Equal(DType.Utf8, cs.DType);
        Assert.Empty(cs.Shape!.Dims);
        Assert.Equal(["hello"], cs.StringData!.Value.ToArray());
        var css = Assert.IsType<RuntimeTensor>(strings[1]);
        Assert.Equal(DType.Utf8, css.DType);
        Assert.Equal([3L], css.Shape!.Dims);
        Assert.Equal(["a", "b", "c"], css.StringData!.Value.ToArray());

        Assert.True(QeeAudit.Check<QeeRnnShapeAuditCheck>(RecurrentX));
        Assert.True(QeeAudit.QeeOnly<QeeRecurrentQeeOnlyShapeAuditCheck>(RecurrentX));
        Assert.True(QeeAudit.Check<QeeGruShapeAuditCheck>(RecurrentX));
        Assert.True(QeeAudit.Check<QeeLstmShapeAuditCheck>(RecurrentX, I32([2L], 4, 4)));
    }

    private static string Written(string op, bool bidirectional, string[] activations, float[]? alpha, float[]? beta, bool exported = false)
    {
        var x = InputTensor<float32>("x", rank: 3);
        var w = InputTensor<float32>("w", rank: 3);
        var r = InputTensor<float32>("r", rank: 3);
        var y = op switch
        {
            "RNN" => OnnxOp.Rnn(x, w, r, null, null, null, alpha, beta, activations, null,
                bidirectional ? RNNDirection.Bidirectional : RNNDirection.Forward, 5L, false).y,
            "GRU" => OnnxOp.Gru(x, w, r, null, null, null, alpha, beta, activations, null,
                bidirectional ? GRUDirection.Bidirectional : GRUDirection.Forward, 5L, false).y,
            _ => OnnxOp.Lstm(x, w, r, null, null, null, null, null, alpha, beta, activations, null,
                bidirectional ? LSTMDirection.Bidirectional : LSTMDirection.Forward, 5L, null, false).y,
        };
        var graph = new InternalComputationGraph([x, w, r], [y]);
        var node = (exported ? FastOnnxModelBuilder.BuildOnnxModel(graph) : FastOnnxModelBuilder.BuildInternalOnnxModel(graph, prepForOnnx: true))
            .Graph.Nodes.Single(n => n.OpType == op);
        string List(string name) => string.Join(" ", node.Attributes.SingleOrDefault(a => a.Name == name)?.Floats ?? []);
        return $"{List("activation_alpha")} | {List("activation_beta")}";
    }

    [Fact]
    public void TestRecurrentActivationArgumentsReachTheBackendOnePerConsumingActivationWithTheirDefaults()
    {
        Assert.Equal("0.3 0.7 | 0.2 0.2", Written("RNN", true, ["LeakyRelu", "Affine"], [0.3f, 0.7f], [0.2f]));
        Assert.Equal("0.3 0.3 | ", Written("RNN", true, ["Tanh", "LeakyRelu"], [0.3f], null));
        Assert.Equal("0.01 | ", Written("RNN", false, ["LeakyRelu"], null, null));
        Assert.Equal(" | ", Written("RNN", true, ["Tanh", "Relu"], [0.5f], [0.5f]));
        Assert.Equal("1 1 | 0", Written("GRU", false, ["Affine", "ThresholdedRelu"], null, null));
        Assert.Equal("2 1 | 3 4", Written("GRU", true, ["ScaledTanh", "Softsign", "Relu", "Affine"], [2f], [3f, 4f]));
        Assert.Equal("0.1 1 0.01 | 0.5", Written("LSTM", true, ["HardSigmoid", "Tanh", "Elu", "Sigmoid", "Relu", "LeakyRelu"], [0.1f], null));
        Assert.Equal("2 0 | 3 0", Written("LSTM", false, ["Affine", "ScaledTanh", "Tanh"], [2f], [3f]));
        Assert.Equal("0.3 0 | 0 0", Written("RNN", true, ["LeakyRelu", "ScaledTanh"], [0.3f], null));
        Assert.Equal("2 | 3", Written("LSTM", false, ["Affine", "ScaledTanh", "Tanh"], [2f], [3f], exported: true));
        Assert.Equal("0.3 | ", Written("RNN", true, ["LeakyRelu", "ScaledTanh"], [0.3f], null, exported: true));
    }

    [Fact]
    public void TestRecurrentValueAudits()
    {
        Assert.True(QeeAudit.OrtOnly<QeeRnnValueAuditCheck>(Wave(4, 2, 3), Wave(2, 5, 3), Wave(2, 5, 5), Wave(2, 10), Wave(2, 2, 5), SeqLens));
        Assert.True(QeeAudit.OrtOnly<QeeGruValueAuditCheck>(Wave(4, 2, 3), Wave(2, 15, 3), Wave(2, 15, 5), Wave(2, 30), Wave(2, 2, 5), SeqLens));
        Assert.True(QeeAudit.OrtOnly<QeeLstmValueAuditCheck>(Wave(4, 2, 3), Wave(2, 20, 3), Wave(2, 20, 5), Wave(2, 40), Wave(2, 2, 5), Wave(2, 2, 5), Wave(2, 15), SeqLens));
        Assert.True(QeeAudit.OrtOnly<QeeRecurrentActivationArgumentsValueCheck>(Wave(4, 2, 3), Wave(2, 20, 3), Wave(2, 20, 5), Wave(2, 40)));
    }

    [Fact]
    public void TestRecurrentNetworksEndAnEmptySequenceInZeroStates()
    {
        Assert.True(QeeAudit.OrtOnly<QeeRnnValueAuditCheck>(Wave(4, 2, 3), Wave(2, 5, 3), Wave(2, 5, 5), Wave(2, 10), Wave(2, 2, 5), EmptySequence));
        Assert.True(QeeAudit.OrtOnly<QeeGruValueAuditCheck>(Wave(4, 2, 3), Wave(2, 15, 3), Wave(2, 15, 5), Wave(2, 30), Wave(2, 2, 5), EmptySequence));
        Assert.True(QeeAudit.OrtOnly<QeeLstmValueAuditCheck>(Wave(4, 2, 3), Wave(2, 20, 3), Wave(2, 20, 5), Wave(2, 40), Wave(2, 2, 5), Wave(2, 2, 5), Wave(2, 15), EmptySequence));
    }

    [Fact]
    public void TestCropAndResizeAtScaleOneStillCropsToTheRoi()
        => Assert.True(AutoTest.AdvancedTestGraph<CropAndResizeAtScaleOneValues>([],
            [F32([1L, 1L, 1L, 5L], 0f, 1f, 2f, 3f, 4f)],
            expected: [2, 3, 4, -1, -1, 2, 3, 4, -1, -1]));

    [Fact]
    public void TestCropAndResizeVariantsCropToTheRoi()
        => Assert.True(AutoTest.AdvancedTestGraph<CropAndResizeVariantsValues>([],
            [F32([1L, 1L, 1L, 5L], 0f, 1f, 2f, 3f, 4f)],
            expected: [2, 3, 4, -1, -1, 2, 3, 4, -1, -1, -1, 0, 1, 2, 3, 2, 2.8, 3.6, -1, -1, -1,
                0, 5.5, 11, 16.5, 22, -1, -1, -1, -1, -1, 2, 3, 4, -1, -1,
                0, 0, 1, 1, 2, 2, 3, 3, 4, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]));

    [Fact]
    public void TestCropAndResizeAtAnUnchangedLengthReachesTheRoiEnd()
        => Assert.True(AutoTest.AdvancedTestGraph<CropAndResizeToTheRoiEndValues>([],
            [F32([1L, 1L, 1L, 4L], 1f, 2f, 3f, 4f)],
            expected: [2.5, 3, 3.5, 4, 2, 3, 3, 4, 2.5, 3, 3.59375, 4, 1.66796875, 2.5, 3.33203125, 4]));

    [Fact]
    public void TestCubicCropAndResizeAlongChannelsExtrapolatesWhereTheRoiLeavesTheInput()
        => Assert.True(AutoTest.AdvancedTestGraph<CubicCropAndResizeAlongChannelsValues>([],
            [F32([1L, 3L, 1L, 2L], 0f, 1f, 2f, 3f, 4f, 5f)],
            expected: [2, 3, 2.992, 3.992, 3.696, 4.696, -1, -1, -1, -1, -1, -1, 2, 3, 4, 5, -1, -1]));

    [Fact]
    public void TestCubicCropAndResizeScaledOnTheMiddleAxesExtrapolatesWhereTheRoiLeavesTheInput()
        => Assert.True(AutoTest.AdvancedTestGraph<CubicCropAndResizeChannelsLastValues>([],
            [F32([1L, 3L, 2L, 2L], [.. Enumerable.Range(0, 12).Select(i => (float)i)])],
            expected: [-1, -1, 4.7207031, 5.7207031, 6, 7, -1, -1, 6.7047029, 7.7047033, 7.984, 8.984, -1, -1, 8.1127014, 9.1127024,
                9.392, 10.392, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                4, 5, 6, 7, 5.984, 6.984, 7.984, 8.984, 7.392, 8.392, 9.392, 10.392, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                1.625, 2.625, 3.625, 4.625, 6.375, 7.375, 8.375, 9.375, -1, -1, -1, -1,
                4, 5, 4.6296296, 5.6296296, 5.3703742, 6.3703752, 6, 7, 5.984, 6.984, 6.6136298, 7.6136298, 7.3543763, 8.3543777,
                7.984, 8.984, 7.392, 8.392, 8.0216284, 9.0216293, 8.7623768, 9.7623787, 9.392, 10.392,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]));

    [Fact]
    public void TestCubicAntialiasedCropAndResizeKeepsThePolicyScale()
        => Assert.True(AutoTest.AdvancedTestGraph<CubicAntialiasedCropAndResizePolicyValues>([],
            [F32([1L, 4L, 6L, 2L], [.. Enumerable.Range(0, 48).Select(i => (float)(i % 7))])],
            expected: [-1, -1, -1, -1, -1, -1, -1, -1, 3.6422229, 4.6427097, 1.1374716, 2.3953724, 3.9555788, 3.4888892, 3.9883382, 1.1786468,
                -1, -1, -1, -1, -1, -1, -1, -1]));

    [Fact]
    public void TestResizeOverAxesBetweenTransposesMatchesTheResizeOverEveryAxis()
        => Assert.True(AutoTest.AdvancedTestGraph<ResizeOverAxesBetweenTransposesCheck>([],
            [F32([1L, 3L, 4L, 5L], [.. Enumerable.Range(0, 60).Select(i => (float)i)])]));

    [Fact]
    public void TestResizeOverAxesMatchesTheResizeOverEveryAxisInEveryModeAndOperand()
        => Assert.True(AutoTest.AdvancedTestGraph<ResizeOverAxesCheck>([],
            [F32([1L, 3L, 4L, 5L], [.. Enumerable.Range(0, 60).Select(i => (float)i)])]));

    [Fact]
    public void TestResizeOverAxesTakesAnEmptyRoiOrScalesAsAbsent()
        => Assert.True(AutoTest.AdvancedTestGraph<ResizeOverAxesWithEmptyOperandsCheck>([],
            [F32([1L, 3L, 4L, 5L], [.. Enumerable.Range(0, 60).Select(i => (float)i)])]));

    [Fact]
    public void TestResizePolicyOverAxesBetweenTransposesMatchesTheResizeOnTheInput()
        => Assert.True(AutoTest.AdvancedTestGraph<ResizePolicyOverAxesBetweenTransposesCheck>([],
            [F32([1L, 3L, 4L, 5L], [.. Enumerable.Range(0, 60).Select(i => (float)i)])]));

    [Fact]
    public void TestResizePolicyOverNegativeAxesMatchesTheResizeOverTheAxesCountedFromTheFront()
        => Assert.True(AutoTest.AdvancedTestGraph<ResizePolicyOverNegativeAxesCheck>([],
            [F32([1L, 3L, 4L, 5L], [.. Enumerable.Range(0, 60).Select(i => (float)i)])]));

    [Fact]
    public void TestOnnxRuntimeFailsToReturnARangeReturnedBesideTheGatherItDrives()
        => Assert.Equal(ErrorCodes.OU002, Assert.Throws<UnsupportedDTypeException>(
            () => AutoTest.AdvancedTestGraph<RangeReturnedBesideTheGatherItDrivesValues>([], [F32([3L, 2L], 0f, 1f, 2f, 3f, 4f, 5f)])).ErrorCode);

    [Fact]
    public void TestCol2ImOverOneSpatialAxisWithPadsAndStride()
        => Assert.True(AutoTest.AdvancedTestGraph<Col2Im1DPaddedValues>([],
            [F32([1L, 3L, 4L], [.. Enumerable.Range(0, 12).Select(i => (float)i)])],
            expected: [4, 9, 5, 11, 6, 13, 7, 11]));

    [Fact]
    public void TestCol2ImOverOneSpatialAxisWithoutPads()
        => Assert.True(AutoTest.AdvancedTestGraph<Col2Im1DUnpaddedValues>([],
            [F32([1L, 4L, 2L], [.. Enumerable.Range(1, 8).Select(i => (float)i)])],
            expected: [1, 5, 4, 5, 13, 8, 1, 5, 2, 6, 3, 7, 4, 8, 1, 7, 9, 11, 8, 1, 2, 8, 10, 7, 8]));
}
