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
/// QEE kernel for ONNX <c>DepthToSpace</c>: rearranges <c>[N, C*B², H, W]</c> into
/// <c>[N, C, H*B, W*B]</c> — the inverse of <see cref="SpaceToDepthOp"/>. The value path
/// honors the <c>mode</c> attribute: DCR decodes the input channel as <c>(b1, b2, c)</c>
/// (depth-column-row, c fastest), CRD as <c>(c, b1, b2)</c> (column-row-depth).
/// </summary>
internal sealed class DepthToSpaceOp : QuickOp
{
    public override string OpCode => OpCodes.DEPTH_TO_SPACE;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs.Length > 0 ? inputs[0] : null;
        var dtype = x?.DType ?? DType.Float32;
        if (x?.Shape is null) return [RuntimeTensorFactory.Create(dtype, null)];

        var dims = x.Shape.Dims;
        if (dims.Length != 4) return [RuntimeTensorFactory.Create(dtype, null)];

        var bs = attrs.GetLongVal(OnnxOpAttributeNames.AttrBlocksize)
              ?? attrs.GetLongVal(OnnxOpAttributeNames.AttrBlockSize)
              ?? 2;
        if (bs <= 0 || dims[1] % (bs * bs) != 0) return [RuntimeTensorFactory.Create(dtype, null)];

        var outDims = new long[4];
        outDims[0] = dims[0];
        outDims[1] = dims[1] / (bs * bs);
        outDims[2] = dims[2] * bs;
        outDims[3] = dims[3] * bs;
        var outShape = new Shape(outDims);
        var rt = RuntimeTensorFactory.Create(dtype, outShape);

        if (!RuntimeTensorFactory.ShouldStoreData(outShape, maxDataElements)) return [rt];
        if (x.FloatData is null && x.IntData is null && x.BoolData is null) return [rt];

        bool isCrd = ResolveIsCrd(attrs);
        long cOut = outDims[1], hIn = dims[2], wIn = dims[3];

        long outCount = outShape.Count;
        long[]? intBuf = x.IntData is not null ? new long[outCount] : null;
        float[]? floatBuf = x.FloatData is not null ? new float[outCount] : null;
        bool[]? boolBuf = x.BoolData is not null ? new bool[outCount] : null;
        long pos = 0;
        for (long n = 0; n < outDims[0]; n++)
            for (long c = 0; c < cOut; c++)
                for (long oh = 0; oh < outDims[2]; oh++)
                    for (long ow = 0; ow < outDims[3]; ow++)
                    {
                        long h = oh / bs, b1 = oh % bs;
                        long w = ow / bs, b2 = ow % bs;
                        long inC = isCrd
                            ? (c * bs + b1) * bs + b2   // CRD: channel decomposed as (c, b1, b2)
                            : (b1 * bs + b2) * cOut + c; // DCR: channel decomposed as (b1, b2, c)
                        long src = ((n * dims[1] + inC) * hIn + h) * wIn + w;
                        if (intBuf is not null) intBuf[pos] = x.IntData!.Value[(int)src];
                        else if (floatBuf is not null) floatBuf[pos] = x.FloatData!.Value[(int)src];
                        else if (boolBuf is not null) boolBuf[pos] = x.BoolData!.Value[(int)src];
                        pos++;
                    }

        if (intBuf is not null) return [rt with { IntData = ImmutableArray.Create(intBuf) }];
        if (floatBuf is not null) return [rt with { FloatData = ImmutableArray.Create(floatBuf) }];
        if (boolBuf is not null) return [rt with { BoolData = ImmutableArray.Create(boolBuf) }];
        return [rt];
    }

    /// <summary>Tolerantly resolves the <c>mode</c> attribute (enum or wire-form string);
    /// the spec default is DCR.</summary>
    private static bool ResolveIsCrd(OnnxCSharpAttributes attrs)
    {
        if (!attrs.IsAttributeDefined(OnnxOpAttributeNames.AttrMode)) return false;
        var obj = attrs.GetAttributeObj(OnnxOpAttributeNames.AttrMode);
        return obj switch
        {
            DepthColumnRowMode m => m == DepthColumnRowMode.CRD,
            string s => s.Equals("CRD", StringComparison.OrdinalIgnoreCase),
            _ => false,
        };
    }
}
