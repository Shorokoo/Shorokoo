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

internal sealed class GatherOp : QuickOp
{
    public override string OpCode => OpCodes.GATHER;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs[0];
        var indices = inputs[1];
        var axis = (int)(attrs.GetLongVal(OnnxOpAttributeNames.AttrAxis) ?? 0);
        var dtype = x?.DType ?? DType.Float32;

        if (x?.Shape is null || indices?.Shape is null)
            return [RuntimeTensorFactory.Create(dtype, null)];

        if (axis < 0) axis += x.Shape.Dims.Length;
        var outDimsList = new List<long>();
        for (int d = 0; d < axis; d++) outDimsList.Add(x.Shape.Dims[d]);
        foreach (var d in indices.Shape.Dims) outDimsList.Add(d);
        for (int d = axis + 1; d < x.Shape.Dims.Length; d++) outDimsList.Add(x.Shape.Dims[d]);
        var outShape = new Shape(outDimsList.ToArray());
        var rt = RuntimeTensorFactory.Create(dtype, outShape);

        // Propagate concrete data when both source and indices are available. This is critical for
        // shape construction patterns like SHAPE → GATHER → CONCAT.
        if (!RuntimeTensorFactory.ShouldStoreData(outShape, maxDataElements)) return [rt];
        if (indices.IntData is not { } idxData) return [rt];
        bool hasInt = x.IntData is not null;
        bool hasFloat = x.FloatData is not null;
        bool hasBool = x.BoolData is not null;
        if (!hasInt && !hasFloat && !hasBool) return [rt];

        var inDims = x.Shape.Dims;
        long axisSize = inDims[axis];
        long outerCount = 1;
        for (int d = 0; d < axis; d++) outerCount *= inDims[d];
        long innerCount = 1;
        for (int d = axis + 1; d < inDims.Length; d++) innerCount *= inDims[d];
        long indexCount = idxData.Length;
        long totalOut = outerCount * indexCount * innerCount;

        long[]? intBuf = hasInt ? new long[totalOut] : null;
        float[]? floatBuf = hasFloat ? new float[totalOut] : null;
        bool[]? boolBuf = hasBool ? new bool[totalOut] : null;

        long outPos = 0;
        for (long o = 0; o < outerCount; o++)
        {
            for (int k = 0; k < idxData.Length; k++)
            {
                long ix = idxData[k];
                if (ix < 0) ix += axisSize;
                long srcBase = (o * axisSize + ix) * innerCount;
                for (long inner = 0; inner < innerCount; inner++)
                {
                    long src = srcBase + inner;
                    if (intBuf is not null) intBuf[outPos] = x.IntData!.Value[(int)src];
                    else if (floatBuf is not null) floatBuf[outPos] = x.FloatData!.Value[(int)src];
                    else if (boolBuf is not null) boolBuf[outPos] = x.BoolData!.Value[(int)src];
                    outPos++;
                }
            }
        }

        if (intBuf is not null) return [rt with { IntData = ImmutableArray.Create(intBuf) }];
        if (floatBuf is not null) return [rt with { FloatData = ImmutableArray.Create(floatBuf) }];
        if (boolBuf is not null) return [rt with { BoolData = ImmutableArray.Create(boolBuf) }];
        return [rt];
    }
}
