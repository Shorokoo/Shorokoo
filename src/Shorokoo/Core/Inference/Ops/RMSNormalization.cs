using System.Collections.Immutable;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Inference.Helpers;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// QEE kernel for ONNX <c>RMSNormalization</c> (opset 23+).
/// Y has X's shape and — per the spec's type groups (Y: V, the scale group) — the
/// scale input's dtype. Values are computed for small float tensors:
/// <c>y = x / sqrt(mean(x², suffix axes from `axis`) + epsilon) * scale</c>, with
/// scale broadcast right-aligned (unidirectional, per the spec). A non-broadcastable
/// scale or out-of-range axis degrades to shape-only. <c>stash_type</c> only selects
/// the internal computation precision and is ignored here (QEE computes in float32).
/// </summary>
internal sealed class RMSNormalizationOp : QuickOp
{
    public override string OpCode => OpCodes.RMS_NORMALIZATION;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs.Length > 0 ? inputs[0] : null;
        var scale = inputs.Length > 1 ? inputs[1] : null;
        var dtype = scale?.DType ?? x?.DType ?? DType.Float32;
        var rt = RuntimeTensorFactory.Create(dtype, x?.Shape);

        if (x?.Shape is null || x.FloatData is not { } xd || scale?.Shape is null || scale.FloatData is not { } sd
            || !RuntimeTensorFactory.ShouldStoreData(x.Shape, maxDataElements))
            return [rt];

        var dims = x.Shape.Dims;
        var rank = dims.Length;
        var axisAttr = AttrAccess.GetLong(attrs, OnnxOpAttributeNames.AttrAxis, -1);
        var axis = (int)(axisAttr < 0 ? axisAttr + rank : axisAttr);
        if (axis < 0 || axis >= rank) return [rt];
        var epsilon = AttrAccess.GetFloat(attrs, OnnxOpAttributeNames.AttrEpsilon, 1e-5f);

        // Right-aligned (numpy-style) broadcast strides for scale against x's shape;
        // unidirectional broadcasting means x's shape wins. Bail to shape-only when
        // a scale dim is neither 1 nor equal to x's dim.
        var sDims = scale.Shape.Dims;
        if (sDims.Length > rank) return [rt];
        var sStrides = new long[rank];
        long stride = 1;
        for (int i = rank - 1; i >= 0; i--)
        {
            int sIdx = i - (rank - sDims.Length);
            if (sIdx < 0) { sStrides[i] = 0; continue; }
            if (sDims[sIdx] == 1) { sStrides[i] = 0; }
            else if (sDims[sIdx] == dims[i]) { sStrides[i] = stride; }
            else return [rt];
            stride *= sDims[sIdx];
        }

        long suffix = 1;
        for (int i = axis; i < rank; i++) suffix *= dims[i];
        long prefix = 1;
        for (int i = 0; i < axis; i++) prefix *= dims[i];
        if (suffix <= 0 || xd.Length != prefix * suffix) return [rt];

        var buf = new float[xd.Length];
        var idx = new long[rank];
        for (long p = 0; p < prefix; p++)
        {
            long baseOff = p * suffix;
            double sumSq = 0;
            for (long q = 0; q < suffix; q++)
            {
                var v = xd[(int)(baseOff + q)];
                sumSq += (double)v * v;
            }
            var invRms = 1.0 / Math.Sqrt(sumSq / suffix + epsilon);
            for (long q = 0; q < suffix; q++)
            {
                long flat = baseOff + q;
                // Decompose the flat offset into per-dim indices to find scale's offset.
                long rem = flat;
                for (int i = rank - 1; i >= 0; i--) { idx[i] = rem % dims[i]; rem /= dims[i]; }
                long sOff = 0;
                for (int i = 0; i < rank; i++) sOff += idx[i] * sStrides[i];
                buf[(int)flat] = (float)(xd[(int)flat] * invRms) * sd[(int)sOff];
            }
        }
        return [rt with { FloatData = ImmutableArray.Create(buf) }];
    }
}
