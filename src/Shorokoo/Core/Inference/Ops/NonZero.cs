using System.Collections.Immutable;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Onnx;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// QEE kernel for ONNX <c>NonZero</c>: output is int64 of shape [rank, n] holding the
/// row-major coordinates of the nonzero elements. n is data-dependent, so the shape (and
/// values) are only produced when the input data is concretely known; otherwise only the
/// output rank (always 2) is reported.
/// </summary>
internal sealed class NonZeroOp : QuickOp
{
    public override string OpCode => OpCodes.NON_ZERO;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs[0];

        if (x?.Shape is { } inShape && x.Shape.Dims.Length >= 1
            && (x.FloatData is not null || x.IntData is not null || x.BoolData is not null))
        {
            var dims = inShape.Dims;
            var rank = dims.Length;
            long total = inShape.Count;

            bool IsNonZero(long flat) =>
                x.FloatData is { } fd ? fd[(int)flat] != 0f :
                x.IntData is { } id ? id[(int)flat] != 0L :
                x.BoolData!.Value[(int)flat];

            var hits = new List<long>();
            for (long flat = 0; flat < total; flat++)
                if (IsNonZero(flat)) hits.Add(flat);

            var outShape = new Shape(new[] { (long)rank, (long)hits.Count });
            var rtKnown = RuntimeTensorFactory.Create(DType.Int64, outShape);
            if (!RuntimeTensorFactory.ShouldStoreData(outShape, maxDataElements)) return [rtKnown];

            // buf[d, k] = coordinate d of the k-th nonzero element (row-major order).
            var buf = new long[rank * hits.Count];
            for (int kk = 0; kk < hits.Count; kk++)
            {
                long rem = hits[kk];
                for (int d = rank - 1; d >= 0; d--)
                {
                    buf[d * hits.Count + kk] = dims[d] == 0 ? 0 : rem % dims[d];
                    rem /= dims[d] == 0 ? 1 : dims[d];
                }
            }
            return [rtKnown with { IntData = ImmutableArray.Create(buf) }];
        }

        // NonZero output shape is [rank, num_nonzero]. Number of nonzeros is data-dependent so we
        // can only pin down the first dimension (the rank).
        int? outRank = x?.Shape is not null ? 2 : null;
        return [new RuntimeTensor
        {
            DType = DType.Int64,
            Shape = null,
            Rank = outRank,
            MaxRank = outRank,
        }];
    }
}
