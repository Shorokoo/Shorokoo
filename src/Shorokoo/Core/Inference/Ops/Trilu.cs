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
/// QEE kernel for ONNX <c>Trilu</c>: shape/dtype pass through; values keep the upper
/// (<c>upper=1</c>, default: <c>j ≥ i + k</c>) or lower (<c>j ≤ i + k</c>) triangle of each
/// trailing [N, M] matrix and zero the rest. The diagonal offset <c>k</c> is the optional
/// second input (default 0; unknown-but-connected blocks value computation).
/// </summary>
internal sealed class TriluOp : QuickOp
{
    public override string OpCode => OpCodes.TRILU;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs[0];
        var dtype = x?.DType ?? DType.Float32;
        var rt = RuntimeTensorFactory.Create(dtype, x?.Shape);
        if (x?.Shape is null || !RuntimeTensorFactory.ShouldStoreData(x.Shape, maxDataElements))
            return [rt];
        if (x.FloatData is null && x.IntData is null && x.BoolData is null) return [rt];

        var dims = x.Shape.Dims;
        var rank = dims.Length;
        if (rank < 2) return [rt]; // spec requires rank ≥ 2

        long k = 0;
        var kIn = inputs.Length > 1 ? inputs[1] : null;
        if (kIn is not null)
        {
            if (kIn.IntData is { Length: > 0 } kd) k = kd[0];
            else return [rt]; // k connected but unknown — values unknowable
        }

        var upper = AttrAccess.GetLong(attrs, OnnxOpAttributeNames.AttrUpper, 1) != 0;
        long n = dims[rank - 2], m = dims[rank - 1];
        long matSize = n * m;
        long batch = matSize == 0 ? 0 : x.Shape.Count / matSize;

        bool Keep(long i, long j) => upper ? j - i >= k : j - i <= k;

        if (x.FloatData is { } fd)
        {
            var buf = fd.ToArray();
            ZeroOutside(buf, batch, n, m, Keep, 0f);
            return [rt with { FloatData = ImmutableArray.Create(buf) }];
        }
        if (x.IntData is { } id)
        {
            var buf = id.ToArray();
            ZeroOutside(buf, batch, n, m, Keep, 0L);
            return [rt with { IntData = ImmutableArray.Create(buf) }];
        }
        if (x.BoolData is { } bd)
        {
            var buf = bd.ToArray();
            ZeroOutside(buf, batch, n, m, Keep, false);
            return [rt with { BoolData = ImmutableArray.Create(buf) }];
        }
        return [rt];
    }

    private static void ZeroOutside<T>(T[] buf, long batch, long n, long m, Func<long, long, bool> keep, T zero)
    {
        for (long b = 0; b < batch; b++)
            for (long i = 0; i < n; i++)
                for (long j = 0; j < m; j++)
                    if (!keep(i, j))
                        buf[(b * n + i) * m + j] = zero;
    }
}
