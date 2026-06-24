using System.Collections.Immutable;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// Shared shape/value machinery for the MatMul family (MatMul, MatMulInteger,
/// QLinearMatMul), implementing the full numpy-style semantics from the ONNX spec:
///   - 1-D <c>a</c>: a 1 is PREPENDED to its shape ([K] → [1,K]) and the corresponding
///     output dim removed after the multiply;
///   - 1-D <c>b</c>: a 1 is APPENDED ([K] → [K,1]) and the corresponding output dim removed;
///   - batch dims broadcast multidirectionally; unknown (−1) dims never guess a size.
/// </summary>
internal static class MatMulHelpers
{
    /// <summary>Dims of <c>a</c> padded for the multiply (prepend 1 when 1-D).</summary>
    public static long[] PadA(long[] aDims) => aDims.Length == 1 ? new[] { 1L, aDims[0] } : aDims;

    /// <summary>Dims of <c>b</c> padded for the multiply (append 1 when 1-D).</summary>
    public static long[] PadB(long[] bDims) => bDims.Length == 1 ? new[] { bDims[0], 1L } : bDims;

    /// <summary>
    /// Output shape per the ONNX MatMul spec, or null when it cannot be safely determined
    /// (unknown input shape, rank-0 input — invalid per spec — or an inner-dim/batch-dim
    /// mismatch, which would make any concrete claim wrong).
    /// </summary>
    public static Shape? InferShape(Shape? aShape, Shape? bShape)
    {
        if (aShape is null || bShape is null) return null;
        var aDims = aShape.Dims;
        var bDims = bShape.Dims;
        if (aDims.Length == 0 || bDims.Length == 0) return null;

        var a = PadA(aDims);
        var b = PadB(bDims);
        // The contracted dims must agree when both are known.
        if (a[^1] >= 0 && b[^2] >= 0 && a[^1] != b[^2]) return null;

        var outRank = Math.Max(a.Length, b.Length);
        var padded = new long[outRank];
        for (int i = 0; i < outRank - 2; i++)
        {
            var ai = i - (outRank - a.Length);
            var bi = i - (outRank - b.Length);
            long ad = ai >= 0 ? a[ai] : 1;
            long bd = bi >= 0 ? b[bi] : 1;
            long d;
            if (ad == bd) d = ad;
            else if (ad == 1) d = bd;
            else if (bd == 1) d = ad;
            else if (ad < 0) d = bd;  // unknown vs known >1: the known dim wins
            else if (bd < 0) d = ad;
            else return null;          // both known, both >1, different → invalid model
            padded[i] = d;
        }
        padded[^2] = a[^2];
        padded[^1] = b[^1];

        // Strip the dims introduced by 1-D padding. The appended b-dim is the last
        // position; the prepended a-dim is the second-to-last of the padded result
        // (or the last once the b-dim has been removed).
        var dims = new List<long>(padded);
        if (bDims.Length == 1) dims.RemoveAt(dims.Count - 1);
        if (aDims.Length == 1) dims.RemoveAt(dims.Count - (bDims.Length == 1 ? 1 : 2));
        return new Shape(dims.ToArray());
    }

    /// <summary>
    /// Drives the matmul accumulation loop over the (padded) output cells. For every
    /// output cell and every contraction index k, calls
    /// <paramref name="term"/>(flatIndexIntoA, flatIndexIntoB, flatOutIndex). The flat
    /// output layout is identical to the final (1-D-stripped) output layout, so callers
    /// can accumulate straight into the result buffer. Returns false when any dim is
    /// unknown (no terms are emitted).
    /// </summary>
    public static bool Accumulate(long[] aDimsRaw, long[] bDimsRaw, Action<int, int, int> term)
    {
        var a = PadA(aDimsRaw);
        var b = PadB(bDimsRaw);
        if (a.Any(d => d < 0) || b.Any(d => d < 0)) return false;

        int outRank = Math.Max(a.Length, b.Length);
        var oDims = new long[outRank];
        for (int i = 0; i < outRank - 2; i++)
        {
            var ai = i - (outRank - a.Length);
            var bi = i - (outRank - b.Length);
            long ad = ai >= 0 ? a[ai] : 1;
            long bd = bi >= 0 ? b[bi] : 1;
            oDims[i] = Math.Max(ad, bd);
        }
        long m = a[^2], k = a[^1], n = b[^1];
        oDims[^2] = m;
        oDims[^1] = n;

        var aStr = RowMajorStrides(a);
        var bStr = RowMajorStrides(b);
        // Batch strides aligned to the output dims; 0 where the input broadcasts (dim 1).
        var aBatch = new long[outRank - 2];
        var bBatch = new long[outRank - 2];
        for (int i = 0; i < outRank - 2; i++)
        {
            var ai = i - (outRank - a.Length);
            var bi = i - (outRank - b.Length);
            aBatch[i] = ai >= 0 && a[ai] != 1 ? aStr[ai] : 0;
            bBatch[i] = bi >= 0 && b[bi] != 1 ? bStr[bi] : 0;
        }

        long batchCount = 1;
        for (int i = 0; i < outRank - 2; i++) batchCount *= oDims[i];

        var idx = new long[outRank - 2];
        long outPos = 0;
        for (long batch = 0; batch < batchCount; batch++)
        {
            long aBase = 0, bBase = 0;
            for (int d = 0; d < idx.Length; d++)
            {
                aBase += idx[d] * aBatch[d];
                bBase += idx[d] * bBatch[d];
            }
            for (long mi = 0; mi < m; mi++)
            {
                for (long ni = 0; ni < n; ni++)
                {
                    for (long ki = 0; ki < k; ki++)
                        term((int)(aBase + mi * aStr[^2] + ki * aStr[^1]),
                             (int)(bBase + ki * bStr[^2] + ni * bStr[^1]),
                             (int)outPos);
                    outPos++;
                }
            }
            for (int d = idx.Length - 1; d >= 0; d--)
            {
                if (++idx[d] < oDims[d]) break;
                idx[d] = 0;
            }
        }
        return true;
    }

    private static long[] RowMajorStrides(long[] dims)
    {
        var strides = new long[dims.Length];
        long s = 1;
        for (int i = dims.Length - 1; i >= 0; i--) { strides[i] = s; s *= dims[i]; }
        return strides;
    }
}

/// <summary>
/// QEE kernel for ONNX <c>MatMul</c>: numpy-style shape semantics (incl. the 1-D
/// prepend/append-dim edge cases) plus concrete float/int value computation for small
/// results.
/// </summary>
internal sealed class MatMulOp : QuickOp
{
    public override string OpCode => OpCodes.MATMUL;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var a = inputs.Length > 0 ? inputs[0] : null;
        var b = inputs.Length > 1 ? inputs[1] : null;
        var dtype = a?.DType ?? b?.DType ?? DType.Float32;
        var shape = MatMulHelpers.InferShape(a?.Shape, b?.Shape);
        var rt = RuntimeTensorFactory.Create(dtype, shape);
        if (shape is null || !RuntimeTensorFactory.ShouldStoreData(shape, maxDataElements))
            return [rt];

        if (a!.FloatData is { } af && b!.FloatData is { } bf)
        {
            var buf = new float[shape.Count];
            if (MatMulHelpers.Accumulate(a.Shape!.Dims, b.Shape!.Dims,
                    (ia, ib, io) => buf[io] += af[ia] * bf[ib]))
                return [rt with { FloatData = ImmutableArray.Create(buf) }];
        }
        else if (a.IntData is { } ai && b!.IntData is { } bi)
        {
            var buf = new long[shape.Count];
            if (MatMulHelpers.Accumulate(a.Shape!.Dims, b.Shape!.Dims,
                    (ia, ib, io) => buf[io] += ai[ia] * bi[ib]))
                return [rt with { IntData = ImmutableArray.Create(buf) }];
        }
        return [rt];
    }
}
