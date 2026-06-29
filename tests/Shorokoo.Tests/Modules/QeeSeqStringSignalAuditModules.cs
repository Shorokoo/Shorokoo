namespace Shorokoo.Tests.Modules
{
    // ===================================================================
    //  Self-checking audit modules for the Phase 4 QEE-A6 batch
    //  (sequence, optional, string, signal & control-flow family, ONNX
    //  opset 21). Like the A2–A5 modules, these compare the audited ops'
    //  computed VALUES (where QEE has a value path) and inferred SHAPES
    //  (via ShapeTensor) against spec-expected constants and return a
    //  single Scalar<bit>.
    //
    //  Driven by QeeSeqStringSignalAuditTests: ORT-runnable modules go
    //  through AdvancedTestGraph (validating the expectations against
    //  real ONNX Runtime) plus the strict QeeSelfCheck; modules built on
    //  Shorokoo-internal ops or @string runtime inputs (which carry no
    //  data into QEE/ORT result comparison) use the QeeOnly-style strict
    //  check instead.
    //
    //  Sequence positions are derived from a runtime int64 input so the
    //  SEQUENCE_* nodes survive FastFoldSequences and the QEE handlers
    //  (not the fold) produce the checked values.
    // ===================================================================

    /// <summary>SequenceConstruct (concrete 2-element sequence), SequenceLength,
    /// SequenceAt (runtime position 0 + NEGATIVE position −1), SequenceInsert at a runtime
    /// middle position, SequenceErase at position 0 AND with the position ABSENT (erase
    /// LAST per spec — the def's position input was wrongly required before this batch),
    /// SequenceEmpty (dtype attr) + SequenceLength 0 + insert-into-empty.
    /// Inputs x = [[1,2,3],[4,5,6]], p0 = 0.</summary>
    [Module]
    public partial class QeeSequenceCoreAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x, Scalar<int64> p0)
        {
            var p1 = p0 + Scalar(1L);
            var t1 = x;                          // 1..6
            var t2 = x + Scalar(10f);            // 11..16
            var t3 = x * Scalar(2f);             // 2..12

            var seq = (TensorSequence<float32>)OnnxOp.SequenceConstruct(t1, t2);
            var at0 = (Tensor<float32>)OnnxOp.SequenceAt(seq, p0);
            var atNeg = (Tensor<float32>)OnnxOp.SequenceAt(seq, p0 - Scalar(1L)); // −1 → t2

            var ins = (TensorSequence<float32>)OnnxOp.SequenceInsert(seq, t3, p1); // [t1,t3,t2]
            var insAt = (Tensor<float32>)OnnxOp.SequenceAt(ins, p1);

            var er = (TensorSequence<float32>)OnnxOp.SequenceErase(seq, p0);       // [t2]
            var erAt = (Tensor<float32>)OnnxOp.SequenceAt(er, p0);
            var erLast = (TensorSequence<float32>)OnnxOp.SequenceErase(ins);       // [t1,t3]
            var erLastAt = (Tensor<float32>)OnnxOp.SequenceAt(erLast, p1);

            var empty = Shorokoo.TensorSequence<float32>.CreateEmpty();
            var emptyIns = empty.InsertAt(t1, null);                               // append → [t1]

            var mismatch =
                IntMismatch1(seq.Count, 2L) +
                FloatMismatch(Flat(at0), Vector(1f, 2f, 3f, 4f, 5f, 6f)) +
                ShapeMismatch(at0, Vector(2L, 3L)) +
                FloatMismatch(Flat(atNeg), Vector(11f, 12f, 13f, 14f, 15f, 16f)) +
                IntMismatch1(ins.Count, 3L) +
                FloatMismatch(Flat(insAt), Vector(2f, 4f, 6f, 8f, 10f, 12f)) +
                IntMismatch1(er.Count, 1L) +
                FloatMismatch(Flat(erAt), Vector(11f, 12f, 13f, 14f, 15f, 16f)) +
                IntMismatch1(erLast.Count, 2L) +
                FloatMismatch(Flat(erLastAt), Vector(2f, 4f, 6f, 8f, 10f, 12f)) +
                IntMismatch1(empty.Count, 0L) +
                IntMismatch1(emptyIns.Count, 1L);
            return mismatch < Scalar(1L);
        }

        private static Tensor<float32> Flat(Tensor<float32> t) => t.Reshape(Vector(-1L));

        private static Scalar<int64> IntMismatch1(Scalar<int64> actual, long expected)
            => (actual.Reshape(Vector(1L)) - Vector(expected)).Abs()
                .Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        // NaN-safe: Not(<= tol) counts a NaN diff as a mismatch; a plain "> tol" would pass it (IEEE).
        private static Scalar<int64> FloatMismatch(Tensor<float32> actual, Vector<float32> expected)
            => ((Tensor<bit>)OnnxOp.Not((actual - expected).Abs() <= Scalar(1e-3f))).Cast<int64>()
                .Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        private static Scalar<int64> ShapeMismatch(ITensor t, Vector<int64> expected)
            => (t.TShape - expected).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
    }

    /// <summary>SplitToSequence: default (no split → size-1 chunks, keepdims=1 keeps the
    /// axis: [1,3] elements, count = 2), keepdims=0 (axis squeezed: [3] elements), scalar
    /// split=2 along axis 1 with UNEVEN tail ([2,2] then [2,1]), 1-D split [1,2].
    /// ConcatFromSequence: axis concat over the uneven-split sequence reassembles x exactly
    /// (per-element shapes legally disagree on the axis dim), and new_axis=1 stacks the
    /// default split into [2,1,3] == x values. Inputs x = [[1,2,3],[4,5,6]], p0 = 0.</summary>
    [Module]
    public partial class QeeSplitToSeqConcatAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x, Scalar<int64> p0)
        {
            var p1 = p0 + Scalar(1L);

            var stsDef = (TensorSequence<float32>)OnnxOp.SplitToSequence(x);
            var defAt0 = (Tensor<float32>)OnnxOp.SequenceAt(stsDef, p0);

            var stsNk = (TensorSequence<float32>)OnnxOp.SplitToSequence(x, split: null, axis: 0L, keepdims: 0L);
            var nkAt1 = (Tensor<float32>)OnnxOp.SequenceAt(stsNk, p1);

            // Scalar split 2 along axis 1 → [2,2] + [2,1].
            var stsS = (TensorSequence<float32>)OnnxOp.SplitToSequence(x, split: Scalar(2L), axis: 1L);
            var sAt1 = (Tensor<float32>)OnnxOp.SequenceAt(stsS, p1);

            var stsV = (TensorSequence<float32>)OnnxOp.SplitToSequence(x, split: Vector(1L, 2L), axis: 1L);
            var vAt1 = (Tensor<float32>)OnnxOp.SequenceAt(stsV, p1);

            var cfsUneven = stsS.Concat(axis: 1);            // [2,2]+[2,1] → [2,3] == x
            var cfsStack = stsDef.Concat(axis: 0, newAxis: true); // 2×[1,3] → [2,1,3]

            var mismatch =
                IntMismatch1(stsDef.Count, 2L) +
                ShapeMismatch(defAt0, Vector(1L, 3L)) +
                FloatMismatch(Flat(defAt0), Vector(1f, 2f, 3f)) +
                ShapeMismatch(nkAt1, Vector(3L)) +
                FloatMismatch(Flat(nkAt1), Vector(4f, 5f, 6f)) +
                IntMismatch1(stsS.Count, 2L) +
                ShapeMismatch(sAt1, Vector(2L, 1L)) +
                FloatMismatch(Flat(sAt1), Vector(3f, 6f)) +
                ShapeMismatch(vAt1, Vector(2L, 2L)) +
                FloatMismatch(Flat(vAt1), Vector(2f, 3f, 5f, 6f)) +
                ShapeMismatch(cfsUneven, Vector(2L, 3L)) +
                FloatMismatch(Flat(cfsUneven), Vector(1f, 2f, 3f, 4f, 5f, 6f)) +
                ShapeMismatch(cfsStack, Vector(2L, 1L, 3L)) +
                FloatMismatch(Flat(cfsStack), Vector(1f, 2f, 3f, 4f, 5f, 6f));
            return mismatch < Scalar(1L);
        }

        private static Tensor<float32> Flat(Tensor<float32> t) => t.Reshape(Vector(-1L));

        private static Scalar<int64> IntMismatch1(Scalar<int64> actual, long expected)
            => (actual.Reshape(Vector(1L)) - Vector(expected)).Abs()
                .Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        // NaN-safe: Not(<= tol) counts a NaN diff as a mismatch; a plain "> tol" would pass it (IEEE).
        private static Scalar<int64> FloatMismatch(Tensor<float32> actual, Vector<float32> expected)
            => ((Tensor<bit>)OnnxOp.Not((actual - expected).Abs() <= Scalar(1e-3f))).Cast<int64>()
                .Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        private static Scalar<int64> ShapeMismatch(ITensor t, Vector<int64> expected)
            => (t.TShape - expected).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
    }

    /// <summary>Spec conformance for the SplitToSequence keepdims/split interaction:
    /// "If input 'split' is specified, [keepdims] is ignored" — the element rank must stay 2
    /// ([2,1] here) even with keepdims=0 (QEE wrongly squeezed the axis before this batch).
    /// QEE-only: ORT deviates from the spec here — its SplitToSequence kernel APPLIES
    /// keepdims with split given and crashes reshaping a size-2 chunk to the squeezed shape
    /// ("Tensor size (4) != new size (2)"), so this combination can't be ORT-validated.
    /// Inputs x = [[1,2,3],[4,5,6]], p0 = 0.</summary>
    [Module]
    public partial class QeeSplitKeepdimsInteractionAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x, Scalar<int64> p0)
        {
            var sts = (TensorSequence<float32>)OnnxOp.SplitToSequence(x, split: Scalar(2L), axis: 1L, keepdims: 0L);
            var last = (Tensor<float32>)OnnxOp.SequenceAt(sts, p0 + Scalar(1L));
            var mismatch =
                ShapeMismatch(last, Vector(2L, 1L)) +
                FloatMismatch(last.Reshape(Vector(-1L)), Vector(3f, 6f));
            return mismatch < Scalar(1L);
        }

        // NaN-safe: Not(<= tol) counts a NaN diff as a mismatch; a plain "> tol" would pass it (IEEE).
        private static Scalar<int64> FloatMismatch(Tensor<float32> actual, Vector<float32> expected)
            => ((Tensor<bit>)OnnxOp.Not((actual - expected).Abs() <= Scalar(1e-3f))).Cast<int64>()
                .Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        private static Scalar<int64> ShapeMismatch(ITensor t, Vector<int64> expected)
            => (t.TShape - expected).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
    }

    /// <summary>ReverseSequence VALUES: batch_axis=0/time_axis=1 (the spec's "batchwise"
    /// example: lens [1,2,3,4] reverse each row's prefix) and the ATTR DEFAULTS
    /// batch_axis=1/time_axis=0 (the "timewise" layout: lens [4,3,2,1] reverse each
    /// column's prefix) — QEE used to compute no values and ignored both attrs.
    /// Input x4 = [[1..4],[5..8],[9..12],[13..16]].</summary>
    [Module]
    public partial class QeeReverseSequenceAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x4)
        {
            var revBatch = (Tensor<float32>)OnnxOp.ReverseSequence(
                x4, Vector(1L, 2L, 3L, 4L), batchAxis: 0, timeAxis: 1);
            // No attrs → spec defaults batch_axis=1, time_axis=0.
            var revTime = (Tensor<float32>)OnnxOp.ReverseSequence(x4, Vector(4L, 3L, 2L, 1L));

            var mismatch =
                ShapeMismatch(revBatch, Vector(4L, 4L)) +
                FloatMismatch(Flat(revBatch), Vector(
                    1f, 2f, 3f, 4f,
                    6f, 5f, 7f, 8f,
                    11f, 10f, 9f, 12f,
                    16f, 15f, 14f, 13f)) +
                FloatMismatch(Flat(revTime), Vector(
                    13f, 10f, 7f, 4f,
                    9f, 6f, 3f, 8f,
                    5f, 2f, 11f, 12f,
                    1f, 14f, 15f, 16f));
            return mismatch < Scalar(1L);
        }

        private static Tensor<float32> Flat(Tensor<float32> t) => t.Reshape(Vector(-1L));

        // NaN-safe: Not(<= tol) counts a NaN diff as a mismatch; a plain "> tol" would pass it (IEEE).
        private static Scalar<int64> FloatMismatch(Tensor<float32> actual, Vector<float32> expected)
            => ((Tensor<bit>)OnnxOp.Not((actual - expected).Abs() <= Scalar(1e-3f))).Cast<int64>()
                .Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        private static Scalar<int64> ShapeMismatch(ITensor t, Vector<int64> expected)
            => (t.TShape - expected).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
    }

    /// <summary>Optional family: Optional(x) → OptionalHasElement is concretely TRUE and
    /// OptionalGetElement returns x's exact values/shape; Optional(absent input, type attr)
    /// → OptionalHasElement is concretely FALSE. Input x = [1,2,3].</summary>
    [Module]
    public partial class QeeOptionalAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var opt = OnnxOp.Optional(x, DataStructure.Tensor, DType.Float32);
            var has = ((Scalar<bit>)OnnxOp.OptionalHasElement(opt)).Cast<int64>();
            var got = (Tensor<float32>)OnnxOp.OptionalGetElement(opt);

            var optNone = OnnxOp.Optional(null, DataStructure.Tensor, DType.Float32);
            var hasNone = ((Scalar<bit>)OnnxOp.OptionalHasElement(optNone)).Cast<int64>();

            var mismatch =
                (Scalar(1L) - has).Abs() +
                hasNone.Abs() +
                FloatMismatch(Flat(got), Vector(1f, 2f, 3f)) +
                ShapeMismatch(got, Vector(3L));
            return mismatch < Scalar(1L);
        }

        private static Tensor<float32> Flat(Tensor<float32> t) => t.Reshape(Vector(-1L));

        // NaN-safe: Not(<= tol) counts a NaN diff as a mismatch; a plain "> tol" would pass it (IEEE).
        private static Scalar<int64> FloatMismatch(Tensor<float32> actual, Vector<float32> expected)
            => ((Tensor<bit>)OnnxOp.Not((actual - expected).Abs() <= Scalar(1e-3f))).Cast<int64>()
                .Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        private static Scalar<int64> ShapeMismatch(ITensor t, Vector<int64> expected)
            => (t.TShape - expected).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
    }

    /// <summary>Window VALUES against the spec formulas (size is a runtime input so the
    /// nodes reach both QEE and ORT): HannWindow periodic N=8 + symmetric N=5,
    /// HammingWindow periodic N=4 (spec coefficients 25/46, 21/46) + symmetric N=5,
    /// BlackmanWindow periodic N=4 + symmetric N=5. Input size = 8.</summary>
    [Module]
    public partial class QeeWindowValueAuditCheck
    {
        public static Scalar<bit> Inline(Scalar<int64> size)
        {
            var n5 = size - Scalar(3L);
            var n4 = size - Scalar(4L);

            var hann8 = (Tensor<float32>)OnnxOp.HannWindow(size, outputDatatype: DType.Float32);
            var hannSym5 = (Tensor<float32>)OnnxOp.HannWindow(n5, outputDatatype: DType.Float32, periodic: false);
            var hamm4 = (Tensor<float32>)OnnxOp.HammingWindow(n4, outputDatatype: DType.Float32);
            var hammSym5 = (Tensor<float32>)OnnxOp.HammingWindow(n5, outputDatatype: DType.Float32, periodic: false);
            var black4 = (Tensor<float32>)OnnxOp.BlackmanWindow(n4, outputDatatype: DType.Float32, periodic: true);
            var blackSym5 = (Tensor<float32>)OnnxOp.BlackmanWindow(n5, outputDatatype: DType.Float32, periodic: false);

            var mismatch =
                ShapeMismatch(hann8, Vector(8L)) +
                FloatMismatch(Flat(hann8), Vector(
                    0f, 0.146447f, 0.5f, 0.853553f, 1f, 0.853553f, 0.5f, 0.146447f)) +
                FloatMismatch(Flat(hannSym5), Vector(0f, 0.5f, 1f, 0.5f, 0f)) +
                FloatMismatch(Flat(hamm4), Vector(0.086957f, 0.543478f, 1f, 0.543478f)) +
                FloatMismatch(Flat(hammSym5), Vector(0.086957f, 0.543478f, 1f, 0.543478f, 0.086957f)) +
                FloatMismatch(Flat(black4), Vector(0f, 0.34f, 1f, 0.34f)) +
                FloatMismatch(Flat(blackSym5), Vector(0f, 0.34f, 1f, 0.34f, 0f));
            return mismatch < Scalar(1L);
        }

        private static Tensor<float32> Flat(Tensor<float32> t) => t.Reshape(Vector(-1L));

        // NaN-safe: Not(<= tol) counts a NaN diff as a mismatch; a plain "> tol" would pass it (IEEE).
        private static Scalar<int64> FloatMismatch(Tensor<float32> actual, Vector<float32> expected)
            => ((Tensor<bit>)OnnxOp.Not((actual - expected).Abs() <= Scalar(1e-3f))).Cast<int64>()
                .Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        private static Scalar<int64> ShapeMismatch(ITensor t, Vector<int64> expected)
            => (t.TShape - expected).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
    }

    /// <summary>DFT VALUES (real [1,4,1] input, default axis −2): forward complex output
    /// [1,4,2], onesided (floor(4/2)+1 = 3 unique bins → [1,3,2] with the matching prefix
    /// values), inverse-of-forward roundtrip recovers the signal, dft_length=2 truncation
    /// with an explicit axis INPUT; STFT output shape [1, (16−4)/4+1, 4/2+1, 2] =
    /// [1,4,3,2]; MelWeightMatrix output shape [16/2+1, 8] = [9,8].
    /// Inputs sig=[1,2,3,4] as [1,4,1], stftSig=[1,16,1], frameStep=4, win=ones[4].</summary>
    [Module]
    public partial class QeeDftStftMelAuditCheck
    {
        public static Scalar<bit> Inline(
            Tensor<float32> sig, Tensor<float32> stftSig, Scalar<int64> frameStep, Vector<float32> win)
        {
            var dftF = (Tensor<float32>)OnnxOp.Dft(sig, null, null, inverse: false, onesided: false);
            var dftOne = (Tensor<float32>)OnnxOp.Dft(sig, null, null, inverse: false, onesided: true);
            var dftInv = (Tensor<float32>)OnnxOp.Dft(dftF, null, null, inverse: true, onesided: false);
            var dftLen = (Tensor<float32>)OnnxOp.Dft(sig, Scalar(2L), Scalar(1L), inverse: false, onesided: false);

            var stft = (Tensor<float32>)OnnxOp.STFT(stftSig, frameStep, window: win, frameLength: null, onesided: true);
            var mel = (Tensor<float32>)OnnxOp.MelWeightMatrix(
                Scalar(8L), Scalar(16L), Scalar(16000L), Scalar(0f), Scalar(8000f),
                outputDatatype: DType.Float32);

            var mismatch =
                ShapeMismatch(dftF, Vector(1L, 4L, 2L)) +
                FloatMismatch(Flat(dftF), Vector(10f, 0f, -2f, 2f, -2f, 0f, -2f, -2f)) +
                ShapeMismatch(dftOne, Vector(1L, 3L, 2L)) +
                FloatMismatch(Flat(dftOne), Vector(10f, 0f, -2f, 2f, -2f, 0f)) +
                ShapeMismatch(dftInv, Vector(1L, 4L, 2L)) +
                FloatMismatch(Flat(dftInv), Vector(1f, 0f, 2f, 0f, 3f, 0f, 4f, 0f)) +
                ShapeMismatch(dftLen, Vector(1L, 2L, 2L)) +
                FloatMismatch(Flat(dftLen), Vector(3f, 0f, -1f, 0f)) +
                ShapeMismatch(stft, Vector(1L, 4L, 3L, 2L)) +
                ShapeMismatch(mel, Vector(9L, 8L));
            return mismatch < Scalar(1L);
        }

        private static Tensor<float32> Flat(Tensor<float32> t) => t.Reshape(Vector(-1L));

        // NaN-safe: Not(<= tol) counts a NaN diff as a mismatch; a plain "> tol" would pass it (IEEE).
        private static Scalar<int64> FloatMismatch(Tensor<float32> actual, Vector<float32> expected)
            => ((Tensor<bit>)OnnxOp.Not((actual - expected).Abs() <= Scalar(1e-3f))).Cast<int64>()
                .Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        private static Scalar<int64> ShapeMismatch(ITensor t, Vector<int64> expected)
            => (t.TShape - expected).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
    }

    /// <summary>TfIdfVectorizer output extent = max(ngram_indexes) + 1 (= 5 here), NOT the
    /// pool length (= 3) — pins this batch's shape fix, ORT-validated. 1-D [4] → [5] and
    /// 2-D [1,4] → [1,5]. Pool: unigrams {1},{2} + bigram (3,4); mode TF.
    /// Input xi = [1,2,3,4] int64.</summary>
    [Module]
    public partial class QeeTfIdfShapeAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<int64> xi)
        {
            var v1 = (Tensor<float32>)OnnxOp.TfIdfVectorizer(xi,
                maxGramLength: 2L, maxSkipCount: 0L, minGramLength: 1L, mode: "TF",
                ngramCounts: new long[] { 0L, 2L },
                ngramIndexes: new long[] { 2L, 0L, 4L },
                poolInt64s: new long[] { 1L, 2L, 3L, 4L },
                poolStrings: null, weights: null);
            var v2 = (Tensor<float32>)OnnxOp.TfIdfVectorizer(xi.Reshape(Vector(1L, 4L)),
                maxGramLength: 2L, maxSkipCount: 0L, minGramLength: 1L, mode: "TF",
                ngramCounts: new long[] { 0L, 2L },
                ngramIndexes: new long[] { 2L, 0L, 4L },
                poolInt64s: new long[] { 1L, 2L, 3L, 4L },
                poolStrings: null, weights: null);

            var mismatch =
                ShapeMismatch(v1, Vector(5L)) +
                ShapeMismatch(v2, Vector(1L, 5L));
            return mismatch < Scalar(1L);
        }

        private static Scalar<int64> ShapeMismatch(ITensor t, Vector<int64> expected)
            => (t.TShape - expected).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
    }

    /// <summary>@string-input ops (QeeOnly-strict: string runtime data never reaches QEE or
    /// the result comparator, so these are shape/dtype checks): StringConcat broadcast
    /// shape, StringNormalizer WITHOUT stopwords (pure case change → exact passthrough
    /// shape; used to degrade the last dim), RegexFullMatch (bool, same shape), StringSplit
    /// num_splits (int64, input shape). The data-dependent outputs (stopword-filtered
    /// normalizer, split tokens) are returned raw — their exact shape is unknowable and the
    /// strict check only requires a valid dtype. Inputs x,y = string [2].</summary>
    [Module]
    public partial class QeeStringOpsAuditCheck
    {
        public static (Scalar<bit>, Tensor<@string>, Tensor<@string>) Inline(Tensor<@string> x, Tensor<@string> y)
        {
            var concat = (Tensor<@string>)OnnxOp.StringConcat(x, y);
            var norm = (Tensor<@string>)OnnxOp.StringNormalizer(x, caseChangeAction: "LOWER");
            var normStop = (Tensor<@string>)OnnxOp.StringNormalizer(x,
                caseChangeAction: "LOWER", isCaseSensitive: 0L, locale: "en_US",
                stopwords: new[] { "the" });
            var regex = (Tensor<bit>)OnnxOp.RegexFullMatch(concat, pattern: ".*");
            var (splitY, numSplits) = OnnxOp.StringSplit(x, delimiter: " ", maxsplit: 2L);

            var mismatch =
                ShapeMismatch(concat, Vector(2L)) +
                ShapeMismatch(norm, Vector(2L)) +
                ShapeMismatch(regex, Vector(2L)) +
                ShapeMismatch((Tensor<int64>)numSplits, Vector(2L));
            return (mismatch < Scalar(1L), normStop, (Tensor<@string>)splitY);
        }

        private static Scalar<int64> ShapeMismatch(ITensor t, Vector<int64> expected)
            => (t.TShape - expected).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
    }

    /// <summary>Shorokoo-internal control-flow/lowering ops (QeeOnly-strict — ORT has no
    /// kernels for these op codes): StateUpdateLink passes the UPDATED state (input 1)
    /// through with values; WithStateDeps passes the MAIN output (input 0) through with
    /// values; ShrkConv infers [1,1,3,3] from geometry-as-inputs (resolved constants);
    /// ShrkRandomNormal/ShrkRandomUniform get their shape from the shape input's VALUES
    /// (here a ShapeTensor chain); LoopIndexVariable is a rank-0 int64. LoopFakeInput
    /// (dtype attr + rank-only) and LoopScanVariable (rank+1, unknown iteration dim) have
    /// no computable shape and are returned raw for dtype validation.
    /// Inputs x = [[1,2,3],[4,5,6]], xImg = 4×4 NCHW, w = 2×2 kernel, b = [0].</summary>
    [Module]
    public partial class QeeInternalControlFlowAuditCheck
    {
        public static (Scalar<bit>, Tensor<float32>, Tensor<float32>) Inline(
            Tensor<float32> x, Tensor<float32> xImg, Tensor<float32> w, Vector<float32> b)
        {
            var updated = x + Scalar(1f);
            var link = (Tensor<float32>)InternalOp.StateUpdateLink(x, updated);
            var main = (Tensor<float32>)InternalOp.WithStateDeps(x, link);

            var conv = (Tensor<float32>)InternalOp.Conv(xImg, w, b, AutoPad.NotSet,
                pads: Vector(0L, 0L, 0L, 0L),
                strides: Vector(1L, 1L),
                dilations: Vector(1L, 1L),
                kernelShape: Vector(2L, 2L),
                group: Scalar(1L));

            var srn = (Tensor<float32>)InternalOp.RandomNormal(x.ShapeTensor(), mean: 0f, scale: 1f, seed: 7f);
            var sru = (Tensor<float32>)InternalOp.RandomUniform(x.ShapeTensor(), high: 1f, low: 0f, seed: 8f);

            var loopIdx = (Tensor<int64>)OnnxOp.LoopIndexVariable();
            var fakeInput = (Tensor<float32>)OnnxOp.LoopFakeInput(DType.Float32, rank: 2, DataStructure.Tensor);
            var scanVar = (Tensor<float32>)OnnxOp.LoopScanZombie(x);

            var mismatch =
                FloatMismatch(Flat(link), Vector(2f, 3f, 4f, 5f, 6f, 7f)) +
                ShapeMismatch(link, Vector(2L, 3L)) +
                FloatMismatch(Flat(main), Vector(1f, 2f, 3f, 4f, 5f, 6f)) +
                ShapeMismatch(conv, Vector(1L, 1L, 3L, 3L)) +
                ShapeMismatch(srn, Vector(2L, 3L)) +
                ShapeMismatch(sru, Vector(2L, 3L)) +
                // LoopIndexVariable: rank 0 → shape-of-shape is [0].
                IntMismatch(loopIdx.ShapeTensor().ShapeTensor(), Vector(0L));
            return (mismatch < Scalar(1L), fakeInput, scanVar);
        }

        private static Tensor<float32> Flat(Tensor<float32> t) => t.Reshape(Vector(-1L));

        // NaN-safe: Not(<= tol) counts a NaN diff as a mismatch; a plain "> tol" would pass it (IEEE).
        private static Scalar<int64> FloatMismatch(Tensor<float32> actual, Vector<float32> expected)
            => ((Tensor<bit>)OnnxOp.Not((actual - expected).Abs() <= Scalar(1e-3f))).Cast<int64>()
                .Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        private static Scalar<int64> IntMismatch(Tensor<int64> actual, Vector<int64> expected)
            => (actual - expected).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        private static Scalar<int64> ShapeMismatch(ITensor t, Vector<int64> expected)
            => (t.TShape - expected).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
    }
}
