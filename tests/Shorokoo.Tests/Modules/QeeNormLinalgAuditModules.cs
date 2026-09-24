using static Shorokoo.Tests.Modules.QeeAuditVerdicts;

namespace Shorokoo.Tests.Modules
{
    // ===================================================================
    //  Self-checking audit modules for the QEE audit of the
    //  normalization, softmax, linear-algebra & quantization family (ONNX
    //  opset 21). Like the elementwise and reduction audit modules, these
    //  compare the audited
    //  ops' computed VALUES (where QEE has a value path) and inferred
    //  SHAPES (via ShapeTensor) against spec-expected constants and
    //  return a single Scalar<bit>.
    //
    //  Driven two ways by QeeNormLinalgAuditTests: AdvancedTestGraph
    //  validates the expected values/shapes against real ONNX Runtime
    //  execution, and the QeeAudit strict-QEE bit-check validates that
    //  QuickExecutionEngine computes the same concrete values (a wrong
    //  or missing QEE value/shape flips the bit or leaves it uncomputed,
    //  failing the test). Ops whose QEE path is shape-only (losses,
    //  Einsum, Det, QLinearMatMul, most norm ops) appear with shape
    //  checks only, so the bit stays QEE-computable.
    // ===================================================================

    /// <summary>BatchNormalization (inference VALUES: per-channel affine; training_mode=1
    /// 3-output shapes: y + running stats [C]), LayerNormalization (negative axis −2 and
    /// default −1: Y passthrough + Mean/InvStdDev with normalized dims set to 1),
    /// InstanceNormalization / GroupNormalization (num_groups) / LRN (size etc.) /
    /// LpNormalization (axis, p) / MeanVarianceNormalization (axes) shape passthrough.
    /// Input x = [[1,2],[3,4]].</summary>
    [Module]
    public partial class QeeNormalizationAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var scale = Vector(2f, 0.5f);
            var bias = Vector(1f, -1f);
            var mean = Vector(1f, 2f);
            var variance = Vector(1f, 4f);

            // Inference mode (training_mode=0): y = (x − mean)/√(var+ε)·scale + bias.
            var bnInfer = x.BatchNormalization(scale, bias, mean, variance);
            // Training mode: 3 outputs; running stats are per-channel [C].
            var (bnTrain, runMean, runVar) = x.BatchNormalizationFullOuputs(
                scale, bias, mean, variance, trainingMode: true);

            var x3 = Vector(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f,
                13f, 14f, 15f, 16f, 17f, 18f, 19f, 20f, 21f, 22f, 23f, 24f)
                .Reshape(Vector(2L, 3L, 4L));
            var scale34 = Vector(1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f).Reshape(Vector(3L, 4L));
            var bias34 = Vector(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f).Reshape(Vector(3L, 4L));
            // Negative axis −2: normalization dims [3,4] → stats [2,1,1].
            var (lnY, lnMean, lnInv) = NN.LayerNormalizationFullOutputs(x3, scale34, bias34, axis: -2);
            // Default axis (−1): stats [2,3,1].
            var (lnY2, lnMean2, lnInv2) = NN.LayerNormalizationFullOutputs(
                x3, Vector(1f, 1f, 1f, 1f), Vector(0f, 0f, 0f, 0f));

            var xi = Vector(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f).Reshape(Vector(1L, 2L, 2L, 2L));
            var instNorm = (Tensor<float32>)OnnxOp.InstanceNormalization(
                xi, Vector(1f, 2f), Vector(0f, 1f), epsilon: 1e-5f);

            var xg = Vector(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f)
                .Reshape(Vector(1L, 4L, 2L, 2L));
            var groupNorm = NN.GroupNormalization(xg,
                Vector(1f, 1f, 1f, 1f), Vector(0f, 0f, 0f, 0f), numGroups: 2);
            var lrn = (Tensor<float32>)OnnxOp.Lrn(xg, alpha: 1e-4f, beta: 0.75f, bias: 1f, size: 3L);
            var lpNorm = (Tensor<float32>)OnnxOp.LpNormalization(x, axis: 0, p: 1);
            var mvn = x3.MeanVarianceNormalization((long[])[0, 2]);

            var mismatch =
                FloatMismatch(Flat(bnInfer), Vector(1f, -1f, 5f, -0.5f)) +
                ShapeMismatch(bnInfer, Vector(2L, 2L)) +
                ShapeMismatch(bnTrain, Vector(2L, 2L)) +
                ShapeMismatch(runMean, Vector(2L)) +
                ShapeMismatch(runVar, Vector(2L)) +
                ShapeMismatch(lnY, Vector(2L, 3L, 4L)) +
                ShapeMismatch(lnMean!, Vector(2L, 1L, 1L)) +
                ShapeMismatch(lnInv!, Vector(2L, 1L, 1L)) +
                ShapeMismatch(lnY2, Vector(2L, 3L, 4L)) +
                ShapeMismatch(lnMean2!, Vector(2L, 3L, 1L)) +
                ShapeMismatch(lnInv2!, Vector(2L, 3L, 1L)) +
                ShapeMismatch(instNorm, Vector(1L, 2L, 2L, 2L)) +
                ShapeMismatch(groupNorm, Vector(1L, 4L, 2L, 2L)) +
                ShapeMismatch(lrn, Vector(1L, 4L, 2L, 2L)) +
                ShapeMismatch(lpNorm, Vector(2L, 2L)) +
                ShapeMismatch(mvn, Vector(2L, 3L, 4L));
            return mismatch < Scalar(1L);
        }

        private static Tensor<float32> Flat(Tensor<float32> t) => t.Reshape(Vector(-1L));
    }

    /// <summary>Softmax (DEFAULT axis −1 since opset 13, explicit axis 0), LogSoftmax
    /// (default axis), Hardmax (default axis, negative axis, tie → FIRST max) — all with
    /// concrete value checks. Input s = [[1,2,3],[3,2,1]].</summary>
    [Module]
    public partial class QeeSoftmaxFamilyValueAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> s)
        {
            var tie = Vector(2f, 2f, 1f, 1f).Reshape(Vector(2L, 2L));
            var mismatch =
                // Default axis is −1 (the LAST axis), not 1-as-in-opset-11.
                FloatMismatch(Flat(s.Softmax()),
                    Vector(0.090031f, 0.244728f, 0.665241f, 0.665241f, 0.244728f, 0.090031f)) +
                FloatMismatch(Flat(s.Softmax(0)),
                    Vector(0.119203f, 0.5f, 0.880797f, 0.880797f, 0.5f, 0.119203f)) +
                FloatMismatch(Flat(s.LogSoftmax()),
                    Vector(-2.407606f, -1.407606f, -0.407606f, -0.407606f, -1.407606f, -2.407606f)) +
                FloatMismatch(Flat(s.Hardmax()), Vector(0f, 0f, 1f, 1f, 0f, 0f)) +
                // Negative axis −2 ≡ axis 0; the col-1 tie (2 vs 2) resolves to the FIRST row.
                FloatMismatch(Flat(s.Hardmax(-2)), Vector(0f, 1f, 1f, 1f, 0f, 0f)) +
                // Tie along the default axis: first occurrence wins per spec.
                FloatMismatch(Flat(tie.Hardmax()), Vector(1f, 0f, 1f, 0f)) +
                ShapeMismatch(s.Softmax(), Vector(2L, 3L));
            return mismatch < Scalar(1L);
        }

        private static Tensor<float32> Flat(Tensor<float32> t) => t.Reshape(Vector(-1L));
    }

    /// <summary>SoftmaxCrossEntropyLoss (reduction="none" → loss has the LABELS' shape;
    /// weights input; log_prob output keeps the scores' shape; explicit reduction="mean"
    /// → rank-0 scalar — exercises the def variant fixed in this batch),
    /// NegativeLogLikelihoodLoss (none / explicit sum), and Dropout (ratio +
    /// training_mode INPUTS; inference mode is an identity with an all-true mask).
    /// Losses are shape-checked only (QEE is shape-only for them); Dropout values are
    /// checked. Inputs: scores [2,3], labels [2] int64.</summary>
    [Module]
    public partial class QeeLossDropoutAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> scores, Tensor<int64> labels)
        {
            var weights = Vector(1f, 0.5f, 2f);
            var (lossNone, logProbNone) = NN.SoftmaxCrossEntropyLoss(
                scores, labels, weights, reduction: "none");
            var (lossMean, logProbMean) = NN.SoftmaxCrossEntropyLoss(
                scores, labels, reduction: "mean");
            var nllNone = NN.NegativeLogLikelihoodLoss(scores, labels, reduction: "none");
            var nllSum = NN.NegativeLogLikelihoodLoss(scores, labels, reduction: "sum");

            // Dropout without ratio/training_mode (inference): identity + all-true mask.
            var (drop1, mask1) = OnnxOp.Dropout(scores, null, null);
            // Dropout with explicit ratio + training_mode=false: still an identity.
            var (drop2, mask2) = OnnxOp.Dropout(scores, Scalar(0.5f), Scalar(false));

            var flatScores = Vector(1f, 2f, 3f, 4f, 5f, 6f);
            var mismatch =
                ShapeMismatch(lossNone, Vector(2L)) +
                ShapeMismatch(logProbNone!, Vector(2L, 3L)) +
                // reduction="mean" → scalar: rank (shape-of-shape) must be 0.
                IntMismatch(lossMean.ShapeTensor().ShapeTensor(), Vector(0L)) +
                ShapeMismatch(logProbMean!, Vector(2L, 3L)) +
                ShapeMismatch(nllNone, Vector(2L)) +
                IntMismatch(nllSum.ShapeTensor().ShapeTensor(), Vector(0L)) +
                FloatMismatch(Flat((Tensor<float32>)drop1), flatScores) +
                IntMismatch(FlatI(((Tensor<bit>)mask1!).Cast<int64>()), Vector(1L, 1L, 1L, 1L, 1L, 1L)) +
                ShapeMismatch((Tensor<bit>)mask1!, Vector(2L, 3L)) +
                FloatMismatch(Flat((Tensor<float32>)drop2), flatScores) +
                IntMismatch(FlatI(((Tensor<bit>)mask2!).Cast<int64>()), Vector(1L, 1L, 1L, 1L, 1L, 1L));
            return mismatch < Scalar(1L);
        }

        private static Tensor<float32> Flat(Tensor<float32> t) => t.Reshape(Vector(-1L));
        private static Tensor<int64> FlatI(Tensor<int64> t) => t.Reshape(Vector(-1L));
    }

    /// <summary>Normalization and loss variants on non-trivial data: SoftmaxCrossEntropyLoss over
    /// spatial dims with weights + ignore_index (mean, with log_prob) and without (sum, none),
    /// NegativeLogLikelihoodLoss with weights + ignore_index (mean) and without (none);
    /// training-mode BatchNormalization with a momentum; LayerNormalization over axis 1 without
    /// bias; InstanceNormalization, GroupNormalization with one channel per group, LRN of even
    /// size, LpNormalization p=2, MeanVarianceNormalization over one axis. Inputs: s [2,4,3,2],
    /// labels [2,3,2] int64.</summary>
    [Module]
    public partial class QeeNormLossVariantsAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> s, Tensor<int64> labels)
        {
            var weights = Vector(0.5f, 1f, 2f, 1.5f);
            var (sceMean, logProb) = NN.SoftmaxCrossEntropyLoss(s, labels, weights, ignoreIndex: 1L, reduction: "mean");
            var (sceSum, _) = NN.SoftmaxCrossEntropyLoss(s, labels, reduction: "sum");
            var (sceNone, _) = NN.SoftmaxCrossEntropyLoss(s, labels, reduction: "none");
            var nllMean = NN.NegativeLogLikelihoodLoss(s, labels, weights, ignoreIndex: 2L, reduction: "mean");
            var nllNone = NN.NegativeLogLikelihoodLoss(s, labels, reduction: "none");
            var scale = Vector(1.5f, -0.5f, 2f, 1f);
            var bias = Vector(0.1f, 0.2f, -0.3f, 0f);
            var (bn, runMean, runVar) = s.BatchNormalizationFullOuputs(scale, bias, Vector(0.2f, -0.1f, 0f, 0.3f),
                Vector(1f, 2f, 0.5f, 1.5f), momentum: 0.7f, trainingMode: true);
            var scaleLn = s.Slice(Vector(0L), Vector(1L)).Reshape(Vector(4L, 3L, 2L));
            var (ln, lnMean, lnInv) = NN.LayerNormalizationFullOutputs(s, scaleLn, axis: 1);
            var inst = (Tensor<float32>)OnnxOp.InstanceNormalization(s, scale, bias, epsilon: 1e-3f);
            var group = NN.GroupNormalization(s, scale, bias, numGroups: 4);
            var lrn = (Tensor<float32>)OnnxOp.Lrn(s, alpha: 0.02f, beta: 0.6f, bias: 2f, size: 5L);
            var lp = (Tensor<float32>)OnnxOp.LpNormalization(s, axis: -1, p: 2);
            var mvn = s.MeanVarianceNormalization((long[])[1]);

            var mismatch =
                IntMismatch(sceMean.ShapeTensor().ShapeTensor(), Vector(0L)) +
                ShapeMismatch(logProb!, Vector(2L, 4L, 3L, 2L)) +
                IntMismatch(sceSum.ShapeTensor().ShapeTensor(), Vector(0L)) +
                ShapeMismatch(sceNone, Vector(2L, 3L, 2L)) +
                IntMismatch(nllMean.ShapeTensor().ShapeTensor(), Vector(0L)) +
                ShapeMismatch(nllNone, Vector(2L, 3L, 2L)) +
                ShapeMismatch(bn, Vector(2L, 4L, 3L, 2L)) +
                ShapeMismatch(runMean, Vector(4L)) +
                ShapeMismatch(runVar, Vector(4L)) +
                ShapeMismatch(ln, Vector(2L, 4L, 3L, 2L)) +
                ShapeMismatch(lnMean!, Vector(2L, 1L, 1L, 1L)) +
                ShapeMismatch(lnInv!, Vector(2L, 1L, 1L, 1L)) +
                ShapeMismatch(inst, Vector(2L, 4L, 3L, 2L)) +
                ShapeMismatch(group, Vector(2L, 4L, 3L, 2L)) +
                ShapeMismatch(lrn, Vector(2L, 4L, 3L, 2L)) +
                ShapeMismatch(lp, Vector(2L, 4L, 3L, 2L)) +
                ShapeMismatch(mvn, Vector(2L, 4L, 3L, 2L));
            return mismatch < Scalar(1L);
        }
    }

    /// <summary>MatMul numpy semantics with VALUES: 2-D×2-D, the 1-D edge cases
    /// ([K]×[K,N] → [N], [M,K]×[K] → [M], [K]×[K] → scalar — ranks checked via
    /// shape-of-shape), batched 3-D×2-D broadcast; Gemm transA / transB / alpha+beta /
    /// C unidirectional broadcast (row [N] and scalar) with values; MatMulInteger
    /// (zero points → int32 values, omitted optional zero points, 1-D edge);
    /// QLinearMatMul shape. Input g = [[1,2,3],[4,5,6]].</summary>
    [Module]
    public partial class QeeMatMulGemmValueAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> g)
        {
            var m32 = Vector(7f, 8f, 9f, 10f, 11f, 12f).Reshape(Vector(3L, 2L));
            var v3 = Vector(1f, 2f, 3f);
            var b3 = Vector(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f).Reshape(Vector(2L, 2L, 3L));
            var bT = Vector(7f, 9f, 11f, 8f, 10f, 12f).Reshape(Vector(2L, 3L));
            var aT = Vector(1f, 4f, 2f, 5f, 3f, 6f).Reshape(Vector(3L, 2L));

            var mm = g.MatMul(m32);
            var mv = (Tensor<float32>)OnnxOp.MatMul(v3, m32);   // [K]×[K,N] → [N]
            var vm = (Tensor<float32>)OnnxOp.MatMul(g, v3);     // [M,K]×[K] → [M]
            var vv = (Tensor<float32>)OnnxOp.MatMul(v3, v3);    // [K]×[K] → scalar
            var bm = (Tensor<float32>)OnnxOp.MatMul(b3, m32);   // batched broadcast

            var gemmC = (Tensor<float32>)OnnxOp.Gemm(g, m32, Vector(1f, 2f));
            var gemmTB = (Tensor<float32>)OnnxOp.Gemm(g, bT, null, transB: 1);
            var gemmTA = (Tensor<float32>)OnnxOp.Gemm(aT, m32, null, transA: 1);
            var gemmAB = (Tensor<float32>)OnnxOp.Gemm(g, m32, Scalar(1f), alpha: 0.5f, beta: 2f);

            var aq = Vector((sbyte)1, (sbyte)2, (sbyte)3, (sbyte)4).Reshape(Vector(2L, 2L));
            var bq = Vector((sbyte)5, (sbyte)6, (sbyte)7, (sbyte)8).Reshape(Vector(2L, 2L));
            var mmi = (Tensor<int32>)OnnxOp.MatMulInteger(aq, bq, Scalar((sbyte)1), Scalar((sbyte)2));
            var mmiNoZp = (Tensor<int32>)OnnxOp.MatMulInteger(aq, bq);
            var mmiVec = (Tensor<int32>)OnnxOp.MatMulInteger(Vector((sbyte)1, (sbyte)2), bq);

            var qa = Vector((sbyte)1, (sbyte)2, (sbyte)3, (sbyte)4, (sbyte)5, (sbyte)6).Reshape(Vector(2L, 3L));
            var qb = Vector((sbyte)1, (sbyte)0, (sbyte)0, (sbyte)1, (sbyte)1, (sbyte)1).Reshape(Vector(3L, 2L));
            var qlmm = (Tensor<int8>)OnnxOp.QLinearMatMul(
                qa, Scalar(0.5f), Scalar((sbyte)0),
                qb, Scalar(0.25f), Scalar((sbyte)0),
                Scalar(0.5f), Scalar((sbyte)0));

            var mismatch =
                ShapeMismatch(mm, Vector(2L, 2L)) +
                FloatMismatch(Flat(mm), Vector(58f, 64f, 139f, 154f)) +
                FloatMismatch(Flat(mv), Vector(58f, 64f)) +
                IntMismatch(mv.ShapeTensor().ShapeTensor(), Vector(1L)) +
                FloatMismatch(Flat(vm), Vector(14f, 32f)) +
                IntMismatch(vm.ShapeTensor().ShapeTensor(), Vector(1L)) +
                FloatMismatch(Flat(vv), Vector(14f)) +
                IntMismatch(vv.ShapeTensor().ShapeTensor(), Vector(0L)) +
                ShapeMismatch(bm, Vector(2L, 2L, 2L)) +
                FloatMismatch(Flat(bm), Vector(58f, 64f, 139f, 154f, 220f, 244f, 301f, 334f)) +
                FloatMismatch(Flat(gemmC), Vector(59f, 66f, 140f, 156f)) +
                ShapeMismatch(gemmC, Vector(2L, 2L)) +
                FloatMismatch(Flat(gemmTB), Vector(58f, 64f, 139f, 154f)) +
                FloatMismatch(Flat(gemmTA), Vector(58f, 64f, 139f, 154f)) +
                FloatMismatch(Flat(gemmAB), Vector(31f, 34f, 71.5f, 79f)) +
                IntMismatch(FlatI(mmi.Cast<int64>()), Vector(5L, 6L, 21L, 26L)) +
                IntMismatch(FlatI(mmiNoZp.Cast<int64>()), Vector(19L, 22L, 43L, 50L)) +
                IntMismatch(FlatI(mmiVec.Cast<int64>()), Vector(19L, 22L)) +
                IntMismatch(mmiVec.ShapeTensor().ShapeTensor(), Vector(1L)) +
                ShapeMismatch(qlmm, Vector(2L, 2L));
            return mismatch < Scalar(1L);
        }

        private static Tensor<float32> Flat(Tensor<float32> t) => t.Reshape(Vector(-1L));
        private static Tensor<int64> FlatI(Tensor<int64> t) => t.Reshape(Vector(-1L));
    }

    /// <summary>Einsum equation parsing (shape-only in QEE): explicit matmul
    /// "ij,jk-&gt;ik", transpose "ij-&gt;ji", reduce "ij-&gt;i", diagonal "ii-&gt;i",
    /// implicit output "ij,jk", batched "bij,bjk-&gt;bik"; Det batch-dims output shape
    /// [..., M, M] → [...] (rank-0 for a single matrix). Input x = [2,3].</summary>
    [Module]
    public partial class QeeEinsumDetAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var k34 = Vector(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f).Reshape(Vector(3L, 4L));
            var sq33 = Vector(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f).Reshape(Vector(3L, 3L));
            var b223 = Vector(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f).Reshape(Vector(2L, 2L, 3L));
            var b232 = Vector(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f).Reshape(Vector(2L, 3L, 2L));
            var d233 = Vector(1f, 0f, 0f, 0f, 2f, 0f, 0f, 0f, 3f,
                2f, 1f, 0f, 0f, 1f, 0f, 1f, 0f, 1f).Reshape(Vector(2L, 3L, 3L));
            var d22 = Vector(3f, 1f, 1f, 2f).Reshape(Vector(2L, 2L));

            var eMatMul = (Tensor<float32>)OnnxOp.Einsum((Variable[])[ x, k34 ], "ij,jk->ik");
            var eTrans = (Tensor<float32>)OnnxOp.Einsum((Variable[])[ x ], "ij->ji");
            var eReduce = (Tensor<float32>)OnnxOp.Einsum((Variable[])[ x ], "ij->i");
            var eDiag = (Tensor<float32>)OnnxOp.Einsum((Variable[])[ sq33 ], "ii->i");
            var eImplicit = (Tensor<float32>)OnnxOp.Einsum((Variable[])[ x, k34 ], "ij,jk");
            var eBatch = (Tensor<float32>)OnnxOp.Einsum((Variable[])[ b223, b232 ], "bij,bjk->bik");
            var detBatch = NN.DeterminantMatrix(d233);
            var detSingle = NN.DeterminantMatrix(d22);

            var mismatch =
                ShapeMismatch(eMatMul, Vector(2L, 4L)) +
                ShapeMismatch(eTrans, Vector(3L, 2L)) +
                ShapeMismatch(eReduce, Vector(2L)) +
                ShapeMismatch(eDiag, Vector(3L)) +
                ShapeMismatch(eImplicit, Vector(2L, 4L)) +
                ShapeMismatch(eBatch, Vector(2L, 2L, 2L)) +
                ShapeMismatch(detBatch, Vector(2L)) +
                IntMismatch(detSingle.ShapeTensor().ShapeTensor(), Vector(0L));
            return mismatch < Scalar(1L);
        }
    }

    /// <summary>Quantization VALUES: QuantizeLinear per-tensor int8 (round-half-even:
    /// 2.5 → 2), per-axis (axis=0 scales + zero points), int8 saturation (±100/0.5 →
    /// 127/−128), uint8 default via output_dtype with no zero point; DequantizeLinear
    /// per-axis and per-tensor; DynamicQuantizeLinear 3 outputs (uint8 y, float scale,
    /// uint8 zero point) per the spec formula. Input q = [[1.25,−0.5],[0.6,3.1]].</summary>
    [Module]
    public partial class QeeQuantizationValueAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> q)
        {
            var qt = (Tensor<int8>)OnnxOp.QuantizeLinear(q, Scalar(0.5f), Scalar((sbyte)1));
            var qa = (Tensor<int8>)OnnxOp.QuantizeLinear(q,
                Vector(0.5f, 0.25f).Tensor(), Vector((sbyte)0, (sbyte)10), axis: 0);
            var qs = (Tensor<int8>)OnnxOp.QuantizeLinear(
                Vector(100f, -100f), Scalar(0.5f), Scalar((sbyte)0));
            // No zero point: output dtype comes from the output_dtype attribute (uint8),
            // negative values saturate at 0.
            var qu = NN.QuantizeLinear<float32, uint8>(q, Scalar(0.5f));

            var xq = Vector((sbyte)10, (sbyte)-20, (sbyte)30, (sbyte)40).Reshape(Vector(2L, 2L));
            var dqAxis = NN.DequantizeLinear(xq,
                Vector(0.5f, 0.25f).Tensor(), Vector((sbyte)0, (sbyte)20), axis: 0);
            var dqTensor = NN.DequantizeLinear(xq, Scalar(0.1f).Tensor(), null);

            var (dy, dScale, dZp) = NN.DynamicQuantizeLinear(Vector(0f, 2f, -3f, 5f));

            var mismatch =
                // 1.25/0.5 = 2.5 rounds HALF-EVEN to 2 (+zp 1 → 3).
                IntMismatch(FlatI(qt.Cast<int64>()), Vector(3L, 0L, 2L, 7L)) +
                ShapeMismatch(qt, Vector(2L, 2L)) +
                IntMismatch(FlatI(qa.Cast<int64>()), Vector(2L, -1L, 12L, 22L)) +
                IntMismatch(FlatI(qs.Cast<int64>()), Vector(127L, -128L)) +
                IntMismatch(FlatI(qu.Cast<int64>()), Vector(2L, 0L, 1L, 6L)) +
                FloatMismatch(Flat(dqAxis), Vector(5f, -10f, 2.5f, 5f)) +
                ShapeMismatch(dqAxis, Vector(2L, 2L)) +
                FloatMismatch(Flat(dqTensor), Vector(1f, -2f, 3f, 4f)) +
                // min −3 / max 5 (0 included): scale 8/255; zp round(95.625) = 96.
                IntMismatch(FlatI(dy.Cast<int64>()), Vector(96L, 160L, 0L, 255L)) +
                FloatMismatch(dScale.Reshape(Vector(1L)), Vector(8f / 255f)) +
                IntMismatch(dZp.Cast<int64>().Reshape(Vector(1L)), Vector(96L));
            return mismatch < Scalar(1L);
        }

        private static Tensor<float32> Flat(Tensor<float32> t) => t.Reshape(Vector(-1L));
        private static Tensor<int64> FlatI(Tensor<int64> t) => t.Reshape(Vector(-1L));
    }

    /// <summary>Quantization VALUES from runtime inputs, so that none is folded (QEE computes no
    /// blocked values, so this runs on ONNX Runtime only): QuantizeLinear
    /// per-tensor int8 with a tie (2.5 → 2) and saturation both ways, per-axis uint8 along
    /// axis 1, blocked (block 2 along axis 1), int16 and uint16 targets; DequantizeLinear
    /// per-tensor, per-axis, blocked and from int32; DynamicQuantizeLinear's three outputs.
    /// Inputs f = [[0.625,−1.3,2.2,40],[−0.1,3.75,−50,0.875]], i8 = [[10,−20,30,−128],[127,0,−3,64]].</summary>
    [Module]
    public partial class QeeQuantizationRuntimeValueAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> f, Tensor<int8> i8)
        {
            var blockScale = Vector(0.5f, 1f, 0.25f, 2f).Reshape(Vector(2L, 2L));
            var qT = (Tensor<int8>)OnnxOp.QuantizeLinear(f, Scalar(0.25f), Scalar((sbyte)3));
            var qAxis = (Tensor<uint8>)OnnxOp.QuantizeLinear(f, Vector(0.5f, 0.25f, 1f, 2f).Tensor(),
                Vector((byte)10, (byte)20, (byte)30, (byte)40), axis: 1);
            var qBlock = (Tensor<int8>)OnnxOp.QuantizeLinear(f, blockScale,
                Vector((sbyte)0, (sbyte)0, (sbyte)0, (sbyte)0).Reshape(Vector(2L, 2L)), axis: 1, blockSize: 2);
            var q16 = (Tensor<int16>)OnnxOp.QuantizeLinear(f, Scalar(0.01f), Scalar((short)-7));
            var qu16 = (Tensor<uint16>)OnnxOp.QuantizeLinear(f, Scalar(0.005f), Scalar((ushort)1000));

            var dq = (Tensor<float32>)OnnxOp.DequantizeLinear(i8, Scalar(0.5f), Scalar((sbyte)-2), null);
            var dqAxis = (Tensor<float32>)OnnxOp.DequantizeLinear(i8, Vector(0.5f, 0.25f, 1f, 2f),
                Vector((sbyte)1, (sbyte)2, (sbyte)3, (sbyte)4), axis: 1);
            var dqBlock = (Tensor<float32>)OnnxOp.DequantizeLinear(i8, blockScale, null, axis: 1, blockSize: 2);
            var dq32 = (Tensor<float32>)OnnxOp.DequantizeLinear(i8.Cast<int32>() * Scalar(1000), Scalar(0.001f), null, null);

            var (dy, dScale, dZp) = OnnxOp.DynamicQuantizeLinear(f);

            var mismatch =
                IntMismatch(FlatI(qT.Cast<int64>()), Vector(5L, -2L, 12L, 127L, 3L, 18L, -128L, 7L)) +
                IntMismatch(FlatI(qAxis.Cast<int64>()), Vector(11L, 15L, 32L, 60L, 10L, 35L, 0L, 40L)) +
                IntMismatch(FlatI(qBlock.Cast<int64>()), Vector(1L, -3L, 2L, 40L, 0L, 15L, -25L, 0L)) +
                IntMismatch(FlatI(q16.Cast<int64>()), Vector(55L, -137L, 213L, 3993L, -17L, 368L, -5007L, 81L)) +
                IntMismatch(FlatI(qu16.Cast<int64>()), Vector(1125L, 740L, 1440L, 9000L, 980L, 1750L, 0L, 1175L)) +
                FloatMismatch(Flat(dq), Vector(6f, -9f, 16f, -63f, 64.5f, 1f, -0.5f, 33f)) +
                FloatMismatch(Flat(dqAxis), Vector(4.5f, -5.5f, 27f, -264f, 63f, -0.5f, -6f, 120f)) +
                FloatMismatch(Flat(dqBlock), Vector(5f, -10f, 30f, -128f, 31.75f, 0f, -6f, 128f)) +
                FloatMismatch(Flat(dq32 - i8.Cast<float32>()), Vector(0f)) +
                IntMismatch(FlatI(((Tensor<uint8>)dy).Cast<int64>()), Vector(144L, 138L, 148L, 255L, 142L, 153L, 0L, 144L)) +
                FloatMismatch(((Tensor<float32>)dScale).Reshape(Vector(1L)), Vector(90f / 255f)) +
                IntMismatch(((Tensor<uint8>)dZp).Cast<int64>().Reshape(Vector(1L)), Vector(142L));
            return mismatch < Scalar(1L);
        }

        private static Tensor<float32> Flat(Tensor<float32> t) => t.Reshape(Vector(-1L));
        private static Tensor<int64> FlatI(Tensor<int64> t) => t.Reshape(Vector(-1L));
    }

    /// <summary>QLinearMatMul and QLinearConv VALUES (QEE computes neither, so this runs on ONNX
    /// Runtime only): int8 per-tensor and uint8 with a per-column b scale and zero point;
    /// QLinearConv uint8 × int8 with per-output-channel w scales, an int32 bias, group 2,
    /// asymmetric pads, strides and dilations, and SAME_LOWER. Inputs a8 = [[1,−2,3],[4,0,−5]],
    /// au = [[125,118,140],[100,130,121]], x [1,2,6,6] uint8 with x[i] = 37·i mod 256.</summary>
    [Module]
    public partial class QeeQLinearValueAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<int8> a8, Tensor<uint8> au, Tensor<uint8> x)
        {
            var b8 = Vector((sbyte)3, (sbyte)-1, (sbyte)0, (sbyte)2, (sbyte)5, (sbyte)-4).Reshape(Vector(3L, 2L));
            var bu = Vector((byte)130, (byte)120, (byte)128, (byte)140, (byte)100, (byte)129).Reshape(Vector(3L, 2L));
            var mm8 = (Tensor<int8>)OnnxOp.QLinearMatMul(
                a8, Scalar(0.05f), Scalar((sbyte)1), b8, Scalar(0.1f), Scalar((sbyte)0), Scalar(0.02f), Scalar((sbyte)-3));
            var mmu = (Tensor<uint8>)OnnxOp.QLinearMatMul(
                au, Scalar(0.02f), Scalar((byte)120), bu, Vector(0.05f, 0.03f), Vector((byte)128, (byte)125),
                Scalar(0.01f), Scalar((byte)100));

            var w = Vector((sbyte)1, (sbyte)-2, (sbyte)3, (sbyte)0, (sbyte)1, (sbyte)-1, (sbyte)2, (sbyte)1, (sbyte)-3,
                (sbyte)-1, (sbyte)0, (sbyte)2, (sbyte)1, (sbyte)1, (sbyte)0, (sbyte)-2, (sbyte)3, (sbyte)1,
                (sbyte)2, (sbyte)2, (sbyte)-1, (sbyte)0, (sbyte)-3, (sbyte)1, (sbyte)1, (sbyte)0, (sbyte)1,
                (sbyte)0, (sbyte)1, (sbyte)1, (sbyte)-1, (sbyte)2, (sbyte)0, (sbyte)3, (sbyte)-2, (sbyte)1).Reshape(Vector(4L, 1L, 3L, 3L));
            var wScale = Vector(0.1f, 0.05f, 0.2f, 0.08f);
            var bias = Vector(50, -30, 0, 200);
            var conv = (Tensor<uint8>)OnnxOp.QLinearConv(x, Scalar(0.04f), Scalar((byte)110), w, wScale, Scalar((sbyte)0),
                Scalar(0.03f), Scalar((byte)128), bias,
                autoPad: AutoPad.NotSet, dilations: [1L, 2L], group: 2L, kernelShape: [3L, 3L],
                pads: [1L, 2L, 0L, 1L], strides: [2L, 1L]);
            var same = (Tensor<uint8>)OnnxOp.QLinearConv(x, Scalar(0.04f), Scalar((byte)110), w, wScale, Scalar((sbyte)0),
                Scalar(0.03f), Scalar((byte)128), null,
                autoPad: AutoPad.SameLower, dilations: null, group: 2L, kernelShape: [3L, 3L], pads: null, strides: [2L, 2L]);

            var mismatch =
                IntMismatch(FlatI(mm8.Cast<int64>()), Vector(0L, -7L, -8L, 2L)) +
                IntMismatch(FlatI(mmu.Cast<int64>()), Vector(45L, 102L, 93L, 115L)) +
                ShapeMismatch(conv, Vector(1L, 4L, 3L, 5L)) +
                IntMismatch(FlatI(conv.Cast<int64>()), Vector(
                    168L, 124L, 144L, 75L, 122L, 60L, 225L, 213L, 155L, 143L, 137L, 30L, 58L, 239L, 184L,
                    136L, 98L, 88L, 132L, 136L, 139L, 139L, 131L, 109L, 93L, 141L, 124L, 142L, 138L, 135L,
                    174L, 164L, 130L, 130L, 51L, 0L, 148L, 124L, 154L, 108L, 216L, 148L, 2L, 236L, 140L,
                    155L, 159L, 139L, 151L, 149L, 136L, 93L, 176L, 223L, 141L, 169L, 180L, 140L, 105L, 173L)) +
                ShapeMismatch(same, Vector(1L, 4L, 3L, 3L)) +
                IntMismatch(FlatI(same.Cast<int64>()), Vector(
                    181L, 98L, 98L, 24L, 192L, 109L, 135L, 37L, 194L, 136L, 119L, 139L, 133L, 138L, 128L, 136L, 132L, 140L,
                    154L, 120L, 120L, 1L, 144L, 135L, 138L, 90L, 149L, 129L, 122L, 146L, 134L, 182L, 140L, 167L, 146L, 185L));
            return mismatch < Scalar(1L);
        }

        private static Tensor<int64> FlatI(Tensor<int64> t) => t.Reshape(Vector(-1L));
    }

    /// <summary>DequantizeLinear of an int32 tensor with no zero point, read through a Reshape:
    /// [1000, −6, 2] × 0.5 = [500, −3, 1].</summary>
    [Module]
    public partial class QeeDequantizeInt32ReshapeAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<int32> x)
        {
            var dq = (Tensor<float32>)OnnxOp.DequantizeLinear(x, Scalar(0.5f), null, null);
            return FloatMismatch(dq.Reshape(Vector(-1L)), Vector(500f, -3f, 1f)) < Scalar(1L);
        }
    }

    /// <summary>DequantizeLinear with no zero point, of int8, int16, uint16 and int32 tensors, each
    /// read through a Reshape (× 0.5) and through a Transpose (× 0.25).</summary>
    [Module]
    public partial class QeeDequantizeWithoutZeroPointReshapeTransposeAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<int8> a, Tensor<int16> b, Tensor<uint16> c, Tensor<int32> d)
        {
            var mismatch =
                Reshaped(OnnxOp.DequantizeLinear(a, Scalar(0.5f), null, null), Vector(50f, -3f, 1f)) +
                Transposed(OnnxOp.DequantizeLinear(a.Reshape(Vector(1L, 3L)), Scalar(0.25f), null, null), Vector(25f, -1.5f, 0.5f)) +
                Reshaped(OnnxOp.DequantizeLinear(b, Scalar(0.5f), null, null), Vector(500f, -3f, 1f)) +
                Transposed(OnnxOp.DequantizeLinear(b.Reshape(Vector(1L, 3L)), Scalar(0.25f), null, null), Vector(250f, -1.5f, 0.5f)) +
                Reshaped(OnnxOp.DequantizeLinear(c, Scalar(0.5f), null, null), Vector(500f, 3f, 1f)) +
                Transposed(OnnxOp.DequantizeLinear(c.Reshape(Vector(1L, 3L)), Scalar(0.25f), null, null), Vector(250f, 1.5f, 0.5f)) +
                Transposed(OnnxOp.DequantizeLinear(d.Reshape(Vector(1L, 3L)), Scalar(0.25f), null, null), Vector(250f, -1.5f, 0.5f));
            return mismatch < Scalar(1L);
        }

        private static Scalar<int64> Reshaped(Variable dq, Vector<float32> expected)
            => FloatMismatch(((Tensor<float32>)dq).Reshape(Vector(-1L)), expected);

        private static Scalar<int64> Transposed(Variable dq, Vector<float32> expected)
            => FloatMismatch(((Tensor<float32>)dq).Transpose().Reshape(Vector(-1L)), expected);
    }

    /// <summary>DequantizeLinear of an int32 [1, 3] tensor with no zero point and a one-element
    /// scale along axis 0, read through a Reshape: [1000, −6, 2] × 0.5 = [500, −3, 1].</summary>
    [Module]
    public partial class QeeDequantizeInt32VectorScaleReshapeAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<int32> x)
        {
            var dq = (Tensor<float32>)OnnxOp.DequantizeLinear(x, Vector(0.5f), null, 0L);
            return FloatMismatch(dq.Reshape(Vector(-1L)), Vector(500f, -3f, 1f)) < Scalar(1L);
        }
    }

    /// <summary>DequantizeLinear of an int32 [2, 3] tensor per axis — along axis 1 with a zero
    /// point, and along axis −2 without one, of the tensor reshaped so its rank is known — read
    /// through a Transpose. x = [[10, −6, 2], [4, 0, −8]].</summary>
    [Module]
    public partial class QeeDequantizeInt32PerAxisAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<int32> x)
        {
            var columns = (Tensor<float32>)OnnxOp.DequantizeLinear(x, Vector(0.5f, 2f, 0.25f), Vector(2, -2, 0), 1L);
            var rows = (Tensor<float32>)OnnxOp.DequantizeLinear(x.Reshape(Vector(2L, 3L)), Vector(0.5f, 0.25f), null, -2L);
            var mismatch =
                FloatMismatch(columns.Transpose().Reshape(Vector(-1L)), Vector(4f, 1f, -8f, 4f, 0.5f, -2f)) +
                FloatMismatch(rows.Transpose().Reshape(Vector(-1L)), Vector(5f, 1f, -3f, 0f, 1f, -2f));
            return mismatch < Scalar(1L);
        }
    }
}
