using System.Collections.Immutable;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Onnx;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Interpreter.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;

namespace Shorokoo.Core.Interpreter.Ops;

/// <summary>
/// QEE kernel for ONNX <c>Range</c>: 1-D output with
/// <c>max(ceil((limit - start) / delta), 0)</c> elements <c>start + i*delta</c>. The element
/// count for the integer dtypes uses exact integer ceiling division (no double roundtrip), and a
/// count beyond int64, which no dimension holds, is refused. Unknown start/limit/delta values degrade to an unknown shape of rank 1.
/// </summary>
internal sealed class RangeOp : QuickOp
{
    public override string OpCode => OpCodes.RANGE;

    /// <summary>Exact ceil(a / b) for integers (b != 0), in 128 bits, which hold the span of any
    /// two int64s.</summary>
    private static Int128 CeilDiv(Int128 a, Int128 b)
    {
        var q = a / b;
        if (a % b != 0 && (a < 0) == (b < 0)) q++;
        return q;
    }

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var start = inputs.Length > 0 ? inputs[0] : null;
        var limit = inputs.Length > 1 ? inputs[1] : null;
        var delta = inputs.Length > 2 ? inputs[2] : null;
        var dtype = start?.DType ?? DType.Int64;

        if (start is null || limit is null || delta is null)
            return [RuntimeTensorFactory.CreateRankOnly(dtype, 1)];

        if (start.IntData is { Length: > 0 } si && limit.IntData is { Length: > 0 } li && delta.IntData is { Length: > 0 } di)
        {
            var d = di[0] == 0 ? 1 : di[0];
            var exact = CeilDiv((Int128)li[0] - si[0], d);
            if (exact > long.MaxValue)
                throw new InvalidOperationException(
                    $"Range({si[0]}, {li[0]}, {di[0]}) has {exact} elements, more than a tensor's dimension holds.");
            var count = (long)Int128.Max(exact, 0);
            var rt = RuntimeTensorFactory.Create(dtype, new Shape(new[] { count }));
            if (RuntimeTensorFactory.ShouldStoreData(rt.Shape, maxDataElements))
            {
                var buf = new long[count];
                for (int i = 0; i < count; i++) buf[i] = si[0] + i * d;
                return [rt with { IntData = ImmutableArray.Create(buf) }];
            }
            return [rt];
        }
        if (start.FloatData is { Length: > 0 } sf && limit.FloatData is { Length: > 0 } lf && delta.FloatData is { Length: > 0 } df)
        {
            var d = df[0] == 0 ? 1 : df[0];
            var count = Math.Max(0, (long)Math.Ceiling((lf[0] - sf[0]) / d));
            var rt = RuntimeTensorFactory.Create(dtype, new Shape(new[] { count }));
            if (RuntimeTensorFactory.ShouldStoreData(rt.Shape, maxDataElements))
            {
                var buf = new float[count];
                for (int i = 0; i < count; i++) buf[i] = sf[0] + i * d;
                return [rt with { FloatData = ImmutableArray.Create(buf) }];
            }
            return [rt];
        }
        return [RuntimeTensorFactory.CreateRankOnly(dtype, 1)];
    }
}
