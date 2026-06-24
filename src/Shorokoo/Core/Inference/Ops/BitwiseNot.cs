using System.Collections.Immutable;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// ONNX <c>BitwiseNot</c>. QEE stores all integers as int64, so for the narrower
/// UNSIGNED dtypes the complement must be masked to the dtype's bit width (e.g.
/// ~12 as uint32 is 4294967283, not -13). int8/16/32/64 and uint64 use the plain
/// 64-bit complement (uint64 shares int64's bit pattern).
/// </summary>
internal sealed class BitwiseNotOp : QuickOp
{
    public override string OpCode => OpCodes.BITWISE_NOT;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs.Length > 0 ? inputs[0] : null;
        var dtype = x?.DType ?? DType.Int64;
        var rt = new RuntimeTensor
        {
            DType = dtype,
            Shape = x?.Shape,
            MaxShape = x?.MaxShape ?? x?.Shape,
            Rank = x?.Rank,
            MaxRank = x?.MaxRank ?? x?.Rank,
        };

        if (x?.IntData is { } id && RuntimeTensorFactory.ShouldStoreData(x.Shape, maxDataElements))
        {
            long? mask =
                dtype == DType.UInt8 ? 0xFFL :
                dtype == DType.UInt16 ? 0xFFFFL :
                dtype == DType.UInt32 ? 0xFFFFFFFFL :
                null;
            var buf = new long[id.Length];
            for (int i = 0; i < buf.Length; i++)
                buf[i] = mask is { } m ? ~id[i] & m : ~id[i];
            return [rt with { IntData = ImmutableArray.Create(buf) }];
        }
        return [rt];
    }
}
