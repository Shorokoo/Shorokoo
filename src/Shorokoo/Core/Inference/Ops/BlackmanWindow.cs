using System;
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
/// ONNX <c>BlackmanWindow</c> op. Input is a scalar <c>size</c> (N). Output is a rank-1
/// tensor of length N filled with the Blackman window coefficients.
///
/// <para>Periodic (default, <c>periodic = 1</c>):
///   <c>w[n] = 0.42 − 0.5·cos(2π·n/N) + 0.08·cos(4π·n/N)</c> for n = 0..N-1</para>
/// <para>Symmetric (<c>periodic = 0</c>): same formula but with <c>N - 1</c> in the
/// denominator.</para>
/// </summary>
internal sealed class BlackmanWindowOp : QuickOp
{
    public override string OpCode => OpCodes.BLACKMAN_WINDOW;

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
            buf[i] = (float)(0.42 - 0.5 * Math.Cos(2 * Math.PI * t) + 0.08 * Math.Cos(4 * Math.PI * t));
        }
        return [rt with { FloatData = ImmutableArray.Create(buf) }];
    }
}
