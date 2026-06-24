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
/// QEE kernel for ONNX <c>OneHot</c>. Inputs are (indices, depth, values); the output
/// inserts a new dimension of size <c>depth</c> at <c>axis</c> in the indices' shape, with
/// dtype taken from <c>values</c> and concrete <c>off/on</c> values when all inputs carry
/// data (negative indices count from the end; out-of-range indices yield all-off rows).
/// </summary>
internal sealed class OneHotOp : QuickOp
{
    public override string OpCode => OpCodes.ONE_HOT;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var indices = inputs.Length > 0 ? inputs[0] : null;
        var depth = inputs.Length > 1 ? inputs[1] : null;
        var values = inputs.Length > 2 ? inputs[2] : null;
        var dtype = values?.DType ?? DType.Float32;

        if (indices?.Shape is null) return [RuntimeTensorFactory.Create(dtype, null)];

        long depthVal = -1;
        if (depth?.IntData is { Length: > 0 } di) depthVal = di[0];
        else if (depth?.FloatData is { Length: > 0 } df) depthVal = (long)df[0];
        if (depthVal < 0) return [RuntimeTensorFactory.Create(dtype, null)];

        var inDims = indices.Shape.Dims;
        var axis = AttrAccess.GetLong(attrs, OnnxOpAttributeNames.AttrAxis, -1);
        var rank = inDims.Length + 1;
        var normAxis = (int)(axis < 0 ? axis + rank : axis);

        if (normAxis < 0 || normAxis >= rank) return [RuntimeTensorFactory.Create(dtype, null)];

        var outDims = new long[rank];
        for (int i = 0, j = 0; i < rank; i++)
        {
            if (i == normAxis) outDims[i] = depthVal;
            else outDims[i] = inDims[j++];
        }
        var outShape = new Shape(outDims);
        var rt = RuntimeTensorFactory.Create(dtype, outShape);

        if (!RuntimeTensorFactory.ShouldStoreData(outShape, maxDataElements)) return [rt];

        // Value path: out[..., d, ...] = (index == d) ? on : off, with negative indices
        // counting from the end ([-depth, depth-1]); anything out of range yields off.
        long[]? idxVals = null;
        if (indices.IntData is { } ii) idxVals = ii.ToArray();
        else if (indices.FloatData is { } fi) idxVals = fi.Select(v => (long)v).ToArray();
        if (idxVals is null || values is null) return [rt];

        long innerCount = 1;
        for (int d = normAxis; d < inDims.Length; d++) innerCount *= inDims[d];
        long outerCount = 1;
        for (int d = 0; d < normAxis; d++) outerCount *= inDims[d];

        long outCount = outShape.Count;
        if (values.FloatData is { Length: >= 2 } vf)
        {
            var buf = FillOneHot(idxVals, outerCount, innerCount, depthVal, vf[0], vf[1]);
            return [rt with { FloatData = ImmutableArray.Create(buf) }];
        }
        if (values.IntData is { Length: >= 2 } vi)
        {
            var buf = FillOneHot(idxVals, outerCount, innerCount, depthVal, vi[0], vi[1]);
            return [rt with { IntData = ImmutableArray.Create(buf) }];
        }
        if (values.BoolData is { Length: >= 2 } vb)
        {
            var buf = FillOneHot(idxVals, outerCount, innerCount, depthVal, vb[0], vb[1]);
            return [rt with { BoolData = ImmutableArray.Create(buf) }];
        }
        return [rt];
    }

    /// <summary>Builds the one-hot buffer in output layout: the new depth axis sits between
    /// the indices' outer dims (product <paramref name="outerCount"/>) and inner dims
    /// (product <paramref name="innerCount"/>).</summary>
    private static T[] FillOneHot<T>(long[] idxVals, long outerCount, long innerCount, long depth, T off, T on)
    {
        var buf = new T[outerCount * depth * innerCount];
        for (long i = 0; i < buf.LongLength; i++) buf[i] = off;
        for (long flat = 0; flat < idxVals.LongLength; flat++)
        {
            long ix = idxVals[flat];
            if (ix < 0) ix += depth;
            if (ix < 0 || ix >= depth) continue; // out-of-range index → all-off row
            long outer = flat / innerCount;
            long inner = flat % innerCount;
            buf[(outer * depth + ix) * innerCount + inner] = on;
        }
        return buf;
    }
}
