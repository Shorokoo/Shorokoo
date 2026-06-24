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
/// QEE kernel for ONNX <c>CumSum</c>. Output shape and dtype match the input; concrete
/// values are computed when the input data is available and the result fits the engine's
/// element-count limit.
/// </summary>
internal sealed class CumSumOp : QuickOp
{
    public override string OpCode => OpCodes.CUM_SUM;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs.Length > 0 ? inputs[0] : null;
        var axisIn = inputs.Length > 1 ? inputs[1] : null;
        var dtype = x?.DType ?? DType.Float32;
        var rt = RuntimeTensorFactory.Create(dtype, x?.Shape);

        if (x?.Shape is null || !RuntimeTensorFactory.ShouldStoreData(x.Shape, maxDataElements))
            return [rt];

        // axis is required by the ONNX spec; if missing, just propagate the shape.
        long axisVal = 0;
        if (axisIn?.IntData is { Length: > 0 } ax) axisVal = ax[0];
        else return [rt];

        var dims = x.Shape.Dims;
        var rank = dims.Length;
        var axis = (int)(axisVal < 0 ? axisVal + rank : axisVal);
        if (axis < 0 || axis >= rank) return [rt];

        var exclusive = AttrAccess.GetBool(attrs, OnnxOpAttributeNames.AttrExclusive);
        var reverse = AttrAccess.GetBool(attrs, OnnxOpAttributeNames.AttrReverse);

        var strides = new long[rank];
        long s = 1;
        for (int i = rank - 1; i >= 0; i--) { strides[i] = s; s *= dims[i]; }
        long axisDim = dims[axis];
        long axisStride = strides[axis];
        long outerCount = 1;
        for (int i = 0; i < axis; i++) outerCount *= dims[i];
        long innerCount = 1;
        for (int i = axis + 1; i < rank; i++) innerCount *= dims[i];

        if (x.FloatData is { } fd)
        {
            var buf = new float[fd.Length];
            for (long outer = 0; outer < outerCount; outer++)
            {
                long outerOff = outer * axisDim * axisStride;
                for (long inner = 0; inner < innerCount; inner++)
                {
                    float acc = 0;
                    if (!reverse)
                    {
                        for (long k = 0; k < axisDim; k++)
                        {
                            long off = outerOff + k * axisStride + inner;
                            float v = fd[(int)off];
                            if (exclusive) { buf[(int)off] = acc; acc += v; }
                            else { acc += v; buf[(int)off] = acc; }
                        }
                    }
                    else
                    {
                        for (long k = axisDim - 1; k >= 0; k--)
                        {
                            long off = outerOff + k * axisStride + inner;
                            float v = fd[(int)off];
                            if (exclusive) { buf[(int)off] = acc; acc += v; }
                            else { acc += v; buf[(int)off] = acc; }
                        }
                    }
                }
            }
            return [rt with { FloatData = ImmutableArray.Create(buf) }];
        }
        if (x.IntData is { } id)
        {
            var buf = new long[id.Length];
            for (long outer = 0; outer < outerCount; outer++)
            {
                long outerOff = outer * axisDim * axisStride;
                for (long inner = 0; inner < innerCount; inner++)
                {
                    long acc = 0;
                    if (!reverse)
                    {
                        for (long k = 0; k < axisDim; k++)
                        {
                            long off = outerOff + k * axisStride + inner;
                            long v = id[(int)off];
                            if (exclusive) { buf[(int)off] = acc; acc += v; }
                            else { acc += v; buf[(int)off] = acc; }
                        }
                    }
                    else
                    {
                        for (long k = axisDim - 1; k >= 0; k--)
                        {
                            long off = outerOff + k * axisStride + inner;
                            long v = id[(int)off];
                            if (exclusive) { buf[(int)off] = acc; acc += v; }
                            else { acc += v; buf[(int)off] = acc; }
                        }
                    }
                }
            }
            return [rt with { IntData = ImmutableArray.Create(buf) }];
        }
        return [rt];
    }
}
