using System;
using System.Collections.Immutable;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// ONNX <c>HammingWindow</c>. Input is a scalar <c>size</c> N. Output is a length-N
/// rank-1 tensor of Hamming-window coefficients: <c>w[n] = 0.54347 − 0.45653·cos(2π·n/D)</c>
/// where <c>D = N</c> (periodic, default) or <c>N - 1</c> (symmetric).
/// </summary>
internal sealed class HammingWindowOp : QuickOp
{
    public override string OpCode => OpCodes.HAMMING_WINDOW;

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
        const double alpha = 25.0 / 46.0;      // ≈ 0.54347
        const double beta = 1.0 - alpha;       // ≈ 0.45653
        for (int i = 0; i < n; i++)
        {
            double t = i / (double)denom;
            buf[i] = (float)(alpha - beta * Math.Cos(2 * Math.PI * t));
        }
        return [rt with { FloatData = ImmutableArray.Create(buf) }];
    }
}
