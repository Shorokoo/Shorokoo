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
/// QEE kernel for ONNX <c>Compress</c>. Selects entries from <c>input</c> along
/// <c>axis</c> (or from the flattened tensor when no axis is given) according to a 1-D
/// boolean <c>condition</c>. The condition may be SHORTER than the selected extent (the
/// missing tail counts as false) or longer (the overrun is ignored). Output dim along
/// <c>axis</c> equals the number of selected entries when the mask is known; otherwise
/// the shape is unknown.
/// </summary>
internal sealed class CompressOp : QuickOp
{
    public override string OpCode => OpCodes.COMPRESS;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs.Length > 0 ? inputs[0] : null;
        var cond = inputs.Length > 1 ? inputs[1] : null;
        var dtype = x?.DType ?? DType.Float32;
        if (x?.Shape is null) return [RuntimeTensorFactory.Create(dtype, null)];

        var axisAttr = attrs.GetLongVal(OnnxOpAttributeNames.AttrAxis);
        var inDims = x.Shape.Dims;
        var rank = inDims.Length;

        int axis = -1; // -1 → flattened
        long extent = x.Shape.Count;
        if (axisAttr is { } a)
        {
            axis = (int)(a < 0 ? a + rank : a);
            if (axis < 0 || axis >= rank) return [RuntimeTensorFactory.Create(dtype, null)];
            extent = inDims[axis];
        }

        // The selected positions (within the axis extent — condition entries beyond the
        // extent are ignored, missing entries count as false). Known only when the
        // condition mask has concrete data.
        List<long>? selected = null;
        if (cond?.BoolData is { } bd)
        {
            selected = new List<long>();
            for (int i = 0; i < bd.Length && i < extent; i++) if (bd[i]) selected.Add(i);
        }
        else if (cond?.IntData is { } id)
        {
            selected = new List<long>();
            for (int i = 0; i < id.Length && i < extent; i++) if (id[i] != 0) selected.Add(i);
        }
        if (selected is null) return [RuntimeTensorFactory.Create(dtype, null)];

        Shape outShape;
        if (axis < 0)
        {
            outShape = new Shape(new[] { (long)selected.Count });
        }
        else
        {
            var outDims = inDims.ToArray();
            outDims[axis] = selected.Count;
            outShape = new Shape(outDims);
        }
        var rt = RuntimeTensorFactory.Create(dtype, outShape);

        if (!RuntimeTensorFactory.ShouldStoreData(outShape, maxDataElements)) return [rt];
        if (x.FloatData is null && x.IntData is null && x.BoolData is null) return [rt];

        long outerCount = 1, innerCount = 1;
        if (axis >= 0)
        {
            for (int d = 0; d < axis; d++) outerCount *= inDims[d];
            for (int d = axis + 1; d < rank; d++) innerCount *= inDims[d];
        }
        long axisDim = axis >= 0 ? inDims[axis] : extent;

        long outCount = outShape.Count;
        long[]? intBuf = x.IntData is not null ? new long[outCount] : null;
        float[]? floatBuf = x.FloatData is not null ? new float[outCount] : null;
        bool[]? boolBuf = x.BoolData is not null ? new bool[outCount] : null;
        long pos = 0;
        for (long outer = 0; outer < outerCount; outer++)
            foreach (var sel in selected)
                for (long inner = 0; inner < innerCount; inner++)
                {
                    long src = (outer * axisDim + sel) * innerCount + inner;
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
}
