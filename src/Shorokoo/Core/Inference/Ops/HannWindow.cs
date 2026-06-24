using System;
using System.Collections.Immutable;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// ONNX <c>HannWindow</c>. Output: <c>w[n] = 0.5 · (1 − cos(2π·n/D))</c> for n = 0..N-1,
/// with <c>D = N</c> (periodic, default) or <c>N - 1</c> (symmetric).
/// </summary>
internal sealed class HannWindowOp : QuickOp
{
    public override string OpCode => OpCodes.HANN_WINDOW;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var dtype = attrs.GetDTypeVal(OnnxOpAttributeNames.AttrOutputDatatype) ?? DType.Float32;
        var periodic = attrs.GetBoolVal(OnnxOpAttributeNames.AttrPeriodic) ?? true;

        long? size = inputs[0]?.IntData is { Length: > 0 } sd ? sd[0] : null;
        if (size is not { } n || n <= 0)
            return [RuntimeTensorFactory.Create(dtype, null) with { Rank = 1, MaxRank = 1 }];

        var shape = new Shape(new[] { n });
        var rt = RuntimeTensorFactory.Create(dtype, shape);

        if (!DTypeHelpers.IsFloat(dtype) || !RuntimeTensorFactory.ShouldStoreData(shape, maxDataElements))
            return [rt];

        var denom = periodic ? n : Math.Max(1, n - 1);
        var buf = new float[n];
        for (int i = 0; i < n; i++)
        {
            double t = i / (double)denom;
            buf[i] = (float)(0.5 * (1.0 - Math.Cos(2 * Math.PI * t)));
        }
        return [rt with { FloatData = ImmutableArray.Create(buf) }];
    }
}
