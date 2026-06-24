using System.Collections.Immutable;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// QEE kernel for ONNX <c>Gemm</c>: <c>Y = alpha·A'·B' + beta·C</c> where A'/B' honor
/// <c>transA</c>/<c>transB</c>, A and B must be 2-D per spec (anything else degrades to
/// an unknown shape rather than guessing), and the optional C broadcasts
/// unidirectionally to [M, N]. Concrete float values are computed for small results;
/// the int path only runs with the default alpha/beta of 1 (fractional scaling of
/// integer tensors would need rounding semantics QEE doesn't model).
/// </summary>
internal sealed class GemmOp : QuickOp
{
    public override string OpCode => OpCodes.GEMM;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var a = inputs.Length > 0 ? inputs[0] : null;
        var b = inputs.Length > 1 ? inputs[1] : null;
        var c = inputs.Length > 2 ? inputs[2] : null;
        var transA = AttrAccess.GetBool(attrs, OnnxOpAttributeNames.AttrTransA, false);
        var transB = AttrAccess.GetBool(attrs, OnnxOpAttributeNames.AttrTransB, false);
        var dtype = a?.DType ?? b?.DType ?? DType.Float32;
        if (a?.Shape is null || b?.Shape is null
            || a.Shape.Dims.Length != 2 || b.Shape.Dims.Length != 2)
            return [RuntimeTensorFactory.Create(dtype, null)];

        var aDims = a.Shape.Dims;
        var bDims = b.Shape.Dims;
        var m = transA ? aDims[1] : aDims[0];
        var kA = transA ? aDims[0] : aDims[1];
        var kB = transB ? bDims[1] : bDims[0];
        var n = transB ? bDims[0] : bDims[1];
        // The contracted dims must agree when both are known.
        if (kA >= 0 && kB >= 0 && kA != kB)
            return [RuntimeTensorFactory.Create(dtype, null)];

        var shape = new Shape(new[] { m, n });
        var rt = RuntimeTensorFactory.Create(dtype, shape);
        if (m < 0 || n < 0 || kA < 0 || !RuntimeTensorFactory.ShouldStoreData(shape, maxDataElements))
            return [rt];

        var alpha = AttrAccess.GetFloat(attrs, OnnxOpAttributeNames.AttrAlpha, 1f);
        var beta = AttrAccess.GetFloat(attrs, OnnxOpAttributeNames.AttrBeta, 1f);

        // Validate C: when connected it must have known shape/data and broadcast
        // unidirectionally to [M, N]; otherwise block value computation (never guess).
        long[]? cDims = null;
        if (c is not null)
        {
            if (c.Shape is null) return [rt];
            cDims = c.Shape.Dims;
            if (cDims.Length > 2) return [rt];
            var cPad = new long[2];
            cPad[1] = cDims.Length > 0 ? cDims[^1] : 1;
            cPad[0] = cDims.Length > 1 ? cDims[^2] : 1;
            if ((cPad[0] != 1 && cPad[0] != m) || (cPad[1] != 1 && cPad[1] != n)) return [rt];
            cDims = cPad;
        }

        long k = kA;
        if (a.FloatData is { } af && b.FloatData is { } bf
            && (c is null || c.FloatData is not null))
        {
            var cf = c?.FloatData;
            var buf = new float[m * n];
            for (long mi = 0; mi < m; mi++)
            {
                for (long ni = 0; ni < n; ni++)
                {
                    float acc = 0;
                    for (long ki = 0; ki < k; ki++)
                        acc += af[(int)(transA ? ki * m + mi : mi * k + ki)]
                             * bf[(int)(transB ? ni * k + ki : ki * n + ni)];
                    float cv = cf is null ? 0f
                        : cf.Value[(int)((cDims![0] == 1 ? 0 : mi) * cDims[1] + (cDims[1] == 1 ? 0 : ni))];
                    buf[mi * n + ni] = alpha * acc + (cf is null ? 0f : beta * cv);
                }
            }
            return [rt with { FloatData = ImmutableArray.Create(buf) }];
        }
        if (alpha == 1f && beta == 1f
            && a.IntData is { } ai && b.IntData is { } bi
            && (c is null || c.IntData is not null))
        {
            var ci = c?.IntData;
            var buf = new long[m * n];
            for (long mi = 0; mi < m; mi++)
            {
                for (long ni = 0; ni < n; ni++)
                {
                    long acc = 0;
                    for (long ki = 0; ki < k; ki++)
                        acc += ai[(int)(transA ? ki * m + mi : mi * k + ki)]
                             * bi[(int)(transB ? ni * k + ki : ki * n + ni)];
                    if (ci is not null)
                        acc += ci.Value[(int)((cDims![0] == 1 ? 0 : mi) * cDims[1] + (cDims[1] == 1 ? 0 : ni))];
                    buf[mi * n + ni] = acc;
                }
            }
            return [rt with { IntData = ImmutableArray.Create(buf) }];
        }
        return [rt];
    }
}
