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

internal sealed class ConcatOp : QuickOp
{
    public override string OpCode => OpCodes.CONCAT;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var axis = (int)(attrs.GetLongVal(OnnxOpAttributeNames.AttrAxis) ?? 0);
        var first = inputs.FirstOrDefault(i => i is not null);
        var dtype = first?.DType ?? DType.Float32;
        if (first?.Shape is null)
            return [RuntimeTensorFactory.Create(dtype, null)];

        if (axis < 0) axis += first.Shape.Dims.Length;
        if (axis < 0 || axis >= first.Shape.Dims.Length)
            return [RuntimeTensorFactory.Create(dtype, null)];
        var outDims = first.Shape.Dims.ToArray();
        outDims[axis] = 0;
        var missingShape = false;
        foreach (var inp in inputs)
        {
            if (inp?.Shape is null) { missingShape = true; break; }
            outDims[axis] += inp.Shape.Dims[axis];
        }
        if (missingShape)
            return [RuntimeTensorFactory.Create(dtype, null)];

        var rt = RuntimeTensorFactory.Create(dtype, new Shape(outDims));
        return [WithConcatValues(rt, inputs, axis, maxDataElements)];
    }

    /// <summary>
    /// General concat value path for any axis/rank: per outer block (dims before axis), copy
    /// each input's contiguous (axisDim × inner) chunk. Shape-composition chains (1-D int64,
    /// axis 0) are the most common case but float/bool and higher ranks fold the same way.
    /// Every input must have a known shape of <paramref name="rt"/>'s rank; shared with
    /// <see cref="ConcatFromSequenceOp"/>.
    /// </summary>
    internal static RuntimeTensor WithConcatValues(
        RuntimeTensor rt, RuntimeTensor?[] inputs, int axis, int maxDataElements)
    {
        if (!RuntimeTensorFactory.ShouldStoreData(rt.Shape, maxDataElements)) return rt;

        var outDims = rt.Shape!.Dims;
        long outerCount = 1;
        for (int d = 0; d < axis; d++) outerCount *= outDims[d];
        long innerCount = 1;
        for (int d = axis + 1; d < outDims.Length; d++) innerCount *= outDims[d];

        if (inputs.All(i => i?.IntData is not null))
        {
            var buf = new long[rt.Shape!.Count];
            FillConcat(inputs, axis, outerCount, innerCount, (inp, src, dst) => buf[dst] = inp.IntData!.Value[(int)src]);
            return rt with { IntData = ImmutableArray.Create(buf) };
        }
        if (inputs.All(i => i?.FloatData is not null))
        {
            var buf = new float[rt.Shape!.Count];
            FillConcat(inputs, axis, outerCount, innerCount, (inp, src, dst) => buf[dst] = inp.FloatData!.Value[(int)src]);
            return rt with { FloatData = ImmutableArray.Create(buf) };
        }
        if (inputs.All(i => i?.BoolData is not null))
        {
            var buf = new bool[rt.Shape!.Count];
            FillConcat(inputs, axis, outerCount, innerCount, (inp, src, dst) => buf[dst] = inp.BoolData!.Value[(int)src]);
            return rt with { BoolData = ImmutableArray.Create(buf) };
        }

        return rt;
    }

    private static void FillConcat(
        RuntimeTensor?[] inputs, int axis, long outerCount, long innerCount,
        Action<RuntimeTensor, long, long> copy)
    {
        long dstPos = 0;
        for (long outer = 0; outer < outerCount; outer++)
        {
            foreach (var inp in inputs)
            {
                long chunk = inp!.Shape!.Dims[axis] * innerCount;
                long srcBase = outer * chunk;
                for (long e = 0; e < chunk; e++) copy(inp, srcBase + e, dstPos++);
            }
        }
    }
}
