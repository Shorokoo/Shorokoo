namespace Shorokoo.Tests.Modules
{
    // ===================================================================
    //  Modules that exercise the QEE shape-inference handlers added by
    //  the opset-21 op expansion batch (HammingWindow, HannWindow,
    //  HardSigmoid, HardSwish, LayerNormalization, LogSoftmax,
    //  MeanVarianceNormalization, Mish, NegativeLogLikelihoodLoss, PRelu,
    //  RegexFullMatch, STFT, Shrink, SoftmaxCrossEntropyLoss, Softplus,
    //  Softsign, ThresholdedRelu) plus the previously-existing QEE
    //  handlers that now have an OnnxOp factory method (Hardmax, IsInf,
    //  IsNaN, OneHot, QuantizeLinear, QLinearMatMul, QLinearConv, Round,
    //  Size, Multinomial, ImageDecoder, MelWeightMatrix, SplitToSequence,
    //  TfIdfVectorizer).
    //
    //  Mirrors the one-liner pattern in QeeOpsTestModules.cs and V2:
    //  each module chains several related ops so a single Coverage
    //  [Fact] driving one Module class widens QEE coverage across many
    //  branches that the existing AutoGrad and QEE suites never reach.
    // ===================================================================

    /// <summary>Unary float activations: Mish, Softplus, Softsign, HardSwish, HardSigmoid, ThresholdedRelu.</summary>
    [Module]
    public partial class QeeNewActivationsCheck
    {
        public static (Tensor<float32>, Tensor<float32>, Tensor<float32>, Tensor<float32>, Tensor<float32>, Tensor<float32>) Inline(Tensor<float32> x)
        {
            var mish = x.Mish();
            var softplus = x.Softplus();
            var softsign = x.Softsign();
            var hswish = x.HardSwish();
            var hsig = x.HardSigmoid(alpha: 0.2f, beta: 0.5f);
            var trelu = x.ThresholdedRelu(alpha: 1.0f);
            return (mish, softplus, softsign, hswish, hsig, trelu);
        }
    }

    /// <summary>IsInf + IsNaN — both produce bool outputs same-shape as the float input.</summary>
    [Module]
    public partial class QeeIsInfNaNCheck
    {
        public static (Tensor<bit>, Tensor<bit>) Inline(Tensor<float32> x)
        {
            var isinf = x.IsInf(detectNegative: true, detectPositive: true);
            var isnan = x.IsNaN();
            return (isinf, isnan);
        }
    }

    /// <summary>Normalization variants: Hardmax, LogSoftmax, MeanVarianceNormalization — all axis/axes-driven.</summary>
    [Module]
    public partial class QeeNormalizationVariantsCheck
    {
        public static (Tensor<float32>, Tensor<float32>, Tensor<float32>) Inline(Tensor<float32> x)
        {
            var hmax = x.Hardmax(axis: -1);
            var lsoft = x.LogSoftmax(axis: -1);
            var mvn = x.MeanVarianceNormalization(axes: new long[] { 0, 1 });
            return (hmax, lsoft, mvn);
        }
    }

    /// <summary>LayerNormalization — 3-output op (y + mean + inv_std_dev).</summary>
    [Module]
    public partial class QeeLayerNormalizationCheck
    {
        public static (Tensor<float32>, Tensor<float32>, Tensor<float32>) Inline(Tensor<float32> x, Tensor<float32> scale, Tensor<float32> bias)
        {
            var (y, mean, invStd) = OnnxOp.LayerNormalization(x, scale, bias, axis: -1, epsilon: 1e-5f, stashType: 1L);
            return ((Tensor<float32>)y, (Tensor<float32>)mean!, (Tensor<float32>)invStd!);
        }
    }

    /// <summary>PRelu — parametric ReLU; output is broadcast of x and slope.</summary>
    [Module]
    public partial class QeePReluCheck
    {
        public static Tensor<float32> Inline(Tensor<float32> x, Tensor<float32> slope)
            => (Tensor<float32>)OnnxOp.PRelu(x, slope);
    }

    /// <summary>NegativeLogLikelihoodLoss + SoftmaxCrossEntropyLoss — reduction-aware loss ops.</summary>
    [Module]
    public partial class QeeNewLossOpsCheck
    {
        public static (Tensor<float32>, Tensor<float32>) Inline(Tensor<float32> scores, Tensor<int64> labels)
        {
            var nll = (Tensor<float32>)OnnxOp.NegativeLogLikelihoodLoss(scores, labels, weight: null, ignoreIndex: -100L, reduction: "mean");
            var (sce, _) = OnnxOp.SoftmaxCrossEntropyLoss(scores, labels, weights: null, ignoreIndex: -100L, reduction: "mean");
            return (nll, (Tensor<float32>)sce);
        }
    }

    /// <summary>HammingWindow + HannWindow — analytic 1-D window vectors of length N.</summary>
    [Module]
    public partial class QeeNewWindowOpsCheck
    {
        public static (Tensor<float32>, Tensor<float32>) Inline(Scalar<int64> size)
        {
            var hamming = (Tensor<float32>)OnnxOp.HammingWindow(size, outputDatatype: DType.Float32, periodic: true);
            var hann = (Tensor<float32>)OnnxOp.HannWindow(size, outputDatatype: DType.Float32, periodic: true);
            return (hamming, hann);
        }
    }

    /// <summary>STFT — short-time Fourier transform; output is [batch, n_frames, n_dft, 2].</summary>
    [Module]
    public partial class QeeSTFTCheck
    {
        public static Tensor<float32> Inline(Tensor<float32> signal, Scalar<int64> frameStep, Vector<float32> window)
            => (Tensor<float32>)OnnxOp.STFT(signal, frameStep, window: window, frameLength: null, onesided: true);
    }

    /// <summary>MelWeightMatrix — five scalar attribute-like inputs producing a 2-D mel matrix.</summary>
    [Module]
    public partial class QeeMelWeightMatrixCheck
    {
        public static Tensor<float32> Inline(
            Scalar<int64> numMelBins, Scalar<int64> dftLength, Scalar<int64> sampleRate,
            Scalar<float32> lower, Scalar<float32> upper)
            => (Tensor<float32>)OnnxOp.MelWeightMatrix(numMelBins, dftLength, sampleRate, lower, upper,
                outputDatatype: DType.Float32);
    }

    /// <summary>Round + Shrink + Size — three small attribute-driven shape passes.</summary>
    [Module]
    public partial class QeeRoundShrinkSizeCheck
    {
        public static (Tensor<float32>, Tensor<float32>, Tensor<int64>) Inline(Tensor<float32> x)
        {
            var r = x.Round();
            var sh = x.Shrink(bias: 0.5f, lambd: 1f);
            var sz = (Tensor<int64>)OnnxOp.Size(x);
            return (r, sh, sz);
        }
    }

    /// <summary>OneHot — int index tensor + scalar depth + values vector → one-hot tensor.</summary>
    [Module]
    public partial class QeeOneHotCheck
    {
        public static Tensor<float32> Inline(Tensor<int64> indices, Scalar<int64> depth, Vector<float32> values)
            => (Tensor<float32>)OnnxOp.OneHot(indices, depth, values, axis: -1);
    }

    /// <summary>QuantizeLinear — float32 → int8 with explicit scale + zero point.</summary>
    [Module]
    public partial class QeeQuantizeLinearCheck
    {
        public static Tensor<int8> Inline(Tensor<float32> x, Tensor<float32> scale, Tensor<int8> zp)
            => (Tensor<int8>)OnnxOp.QuantizeLinear(x, scale, zp,
                axis: 1L, blockSize: null, outputDatatype: null, saturate: true, precision: null);
    }

    /// <summary>QLinearMatMul — quantized matmul with the full 8-input layout.</summary>
    [Module]
    public partial class QeeQLinearMatMulCheck
    {
        public static Tensor<int8> Inline(
            Tensor<int8> a, Scalar<float32> aScale, Scalar<int8> aZp,
            Tensor<int8> b, Scalar<float32> bScale, Scalar<int8> bZp,
            Scalar<float32> yScale, Scalar<int8> yZp)
            => (Tensor<int8>)OnnxOp.QLinearMatMul(a, aScale, aZp, b, bScale, bZp, yScale, yZp);
    }

    /// <summary>QLinearConv — quantized Conv with the standard 8-input layout (no bias).</summary>
    [Module]
    public partial class QeeQLinearConvCheck
    {
        public static Tensor<int8> Inline(
            Tensor<int8> x, Scalar<float32> xScale, Scalar<int8> xZp,
            Tensor<int8> w, Scalar<float32> wScale, Scalar<int8> wZp,
            Scalar<float32> yScale, Scalar<int8> yZp)
            => (Tensor<int8>)OnnxOp.QLinearConv(x, xScale, xZp, w, wScale, wZp, yScale, yZp, b: null,
                autoPad: AutoPad.NotSet, dilations: new long[] { 1, 1 }, group: 1L,
                kernelShape: new long[] { 2, 2 },
                pads: new long[] { 0, 0, 0, 0 }, strides: new long[] { 1, 1 });
    }

    /// <summary>Multinomial — random integer samples from a categorical input.</summary>
    [Module]
    public partial class QeeMultinomialCheck
    {
        public static Tensor<int64> Inline(Tensor<float32> input)
            => (Tensor<int64>)OnnxOp.Multinomial(input, dtype: DType.Int64, sampleSize: 5L, seed: 42f);
    }

    /// <summary>SplitToSequence followed by SequenceAt — extract one tensor from the sequence output.</summary>
    [Module]
    public partial class QeeSplitToSequenceCheck
    {
        public static Tensor<float32> Inline(Tensor<float32> x)
        {
            var seq = OnnxOp.SplitToSequence(x, split: null, axis: 0L, keepdims: 1L);
            return (Tensor<float32>)OnnxOp.SequenceAt(seq, Scalar(0L));
        }
    }

    /// <summary>ImageDecoder — encoded byte stream → decoded image tensor (shape inference only;
    /// the actual decode requires real PNG/JPEG bytes that ORT would otherwise reject).</summary>
    [Module]
    public partial class QeeImageDecoderCheck
    {
        public static Tensor<uint8> Inline(Vector<uint8> encoded)
            => (Tensor<uint8>)OnnxOp.ImageDecoder(encoded, pixelFormat: "RGB");
    }

    /// <summary>TfIdfVectorizer — n-gram tf-idf with ngram_indexes attribute setting output width.</summary>
    [Module]
    public partial class QeeTfIdfVectorizerCheck
    {
        public static Tensor<float32> Inline(Tensor<int64> x)
            => (Tensor<float32>)OnnxOp.TfIdfVectorizer(x,
                maxGramLength: 2L, maxSkipCount: 0L, minGramLength: 1L,
                mode: "TF",
                ngramCounts: new long[] { 0L, 2L },
                ngramIndexes: new long[] { 0L, 1L, 2L },
                poolInt64s: new long[] { 1L, 2L, 3L, 4L },
                poolStrings: null,
                weights: null);
    }

    /// <summary>
    /// StringConcat + RegexFullMatch — element-wise string concatenation feeds the regex
    /// match, chaining the two @string-input QEE handlers so one module drives both. The
    /// pattern uses ".*" to match anything (only shape inference matters for QEE coverage).
    /// </summary>
    [Module]
    public partial class QeeStringConcatRegexCheck
    {
        public static (Tensor<@string>, Tensor<bit>) Inline(Tensor<@string> x, Tensor<@string> y)
        {
            var concat = (Tensor<@string>)OnnxOp.StringConcat(x, y);
            var match = (Tensor<bit>)OnnxOp.RegexFullMatch(concat, pattern: ".*");
            return (concat, match);
        }
    }

    /// <summary>
    /// StringNormalizer + StringSplit — the two remaining @string-input QEE handlers,
    /// chained off a shared input. StringNormalizer carries every attribute branch
    /// (case_change_action, is_case_sensitive, locale, stopwords); StringSplit carries
    /// delimiter + maxsplit and returns both the [..., -1] split tensor and the per-element
    /// num_splits int64 tensor.
    /// </summary>
    [Module]
    public partial class QeeStringNormalizerSplitCheck
    {
        public static (Tensor<@string>, Tensor<@string>, Tensor<int64>) Inline(Tensor<@string> x)
        {
            var normalized = (Tensor<@string>)OnnxOp.StringNormalizer(x,
                caseChangeAction: "LOWER", isCaseSensitive: 0L,
                locale: "en_US", stopwords: new[] { "the", "a" });
            var (split, numSplits) = OnnxOp.StringSplit(x, delimiter: " ", maxsplit: 4L);
            return (normalized, (Tensor<@string>)split, (Tensor<int64>)numSplits);
        }
    }
}
