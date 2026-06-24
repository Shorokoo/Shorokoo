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
using System.Collections.Immutable;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// QEE kernel for ONNX <c>CenterCropPad</c>. The output dims along the listed <c>axes</c>
/// (default: every dim — the <c>shape</c> input must then have one entry per input dim) are
/// taken from the <c>shape</c> input's values; remaining dims pass through. A <c>shape</c> input that is connected but
/// value-unknown at QEE time degrades to an unknown shape with the input's rank (the output
/// dims along the cropped/padded axes differ from the input's, so passing the input shape
/// through would be wrong). Values are computed for small tensors: per axis, a larger input
/// is center-cropped (offset <c>(in-out)/2</c>, extra row cropped from the end) and a smaller
/// one is zero-padded (begin pad <c>(out-in)/2</c>, extra pad at the end), matching the ONNX
/// reference semantics.
/// </summary>
internal sealed class CenterCropPadOp : QuickOp
{
    public override string OpCode => OpCodes.CENTER_CROP_PAD;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs.Length > 0 ? inputs[0] : null;
        var shape = inputs.Length > 1 ? inputs[1] : null;
        var dtype = x?.DType ?? DType.Float32;
        if (x?.Shape is null)
            return [RuntimeTensorFactory.CreateRankOnly(dtype, x?.Rank)];

        var inDims = x.Shape.Dims;
        var rank = inDims.Length;
        if (shape?.IntData is not { } sd)
            return [RuntimeTensorFactory.CreateRankOnly(dtype, rank)];

        var outDims = inDims.ToArray();
        var axesAttr = attrs.GetLongsVal(OnnxOpAttributeNames.AttrAxes);
        int[] axes;
        if (axesAttr is null)
        {
            // Default per spec: all axes — shape must then have one entry per input dim.
            if (sd.Length != rank) return [RuntimeTensorFactory.CreateRankOnly(dtype, rank)];
            axes = new int[sd.Length];
            for (int i = 0; i < sd.Length; i++) axes[i] = i;
        }
        else
        {
            if (axesAttr.Length != sd.Length) return [RuntimeTensorFactory.CreateRankOnly(dtype, rank)];
            axes = new int[axesAttr.Length];
            var seen = new bool[rank];
            for (int i = 0; i < axesAttr.Length; i++)
            {
                var a = axesAttr[i] < 0 ? axesAttr[i] + rank : axesAttr[i];
                if (a < 0 || a >= rank || seen[a])
                    return [RuntimeTensorFactory.CreateRankOnly(dtype, rank)];
                seen[a] = true;
                axes[i] = (int)a;
            }
        }

        for (int i = 0; i < axes.Length; i++)
        {
            if (sd[i] < 0) return [RuntimeTensorFactory.CreateRankOnly(dtype, rank)];
            outDims[axes[i]] = sd[i];
        }

        var outShape = new Shape(outDims);
        var rt = RuntimeTensorFactory.Create(dtype, outShape);
        if (!RuntimeTensorFactory.ShouldStoreData(outShape, maxDataElements) || !x.HasAnyData
            || inDims.Any(d => d < 0))
            return [rt];

        // Per-axis input offset: crop start (in > out) or negative begin-pad (in < out).
        var offset = new long[rank];
        for (int d = 0; d < rank; d++)
            offset[d] = inDims[d] >= outDims[d]
                ? (inDims[d] - outDims[d]) / 2
                : -((outDims[d] - inDims[d]) / 2);

        var count = (int)outShape.Count;
        var coords = new long[rank];
        bool TryMapToInput(int outFlat, out long inFlat)
        {
            // Decompose outFlat into coords (row-major), shift by offset, recompose.
            long rem = outFlat;
            for (int d = rank - 1; d >= 0; d--)
            {
                coords[d] = outDims[d] == 0 ? 0 : rem % outDims[d];
                rem = outDims[d] == 0 ? 0 : rem / outDims[d];
            }
            inFlat = 0;
            for (int d = 0; d < rank; d++)
            {
                var inCoord = coords[d] + offset[d];
                if (inCoord < 0 || inCoord >= inDims[d]) { inFlat = -1; return false; }
                inFlat = inFlat * inDims[d] + inCoord;
            }
            return true;
        }

        if (x.FloatData is { } fd)
        {
            var buf = new float[count];
            for (int i = 0; i < count; i++) buf[i] = TryMapToInput(i, out var src) ? fd[(int)src] : 0f;
            return [rt with { FloatData = ImmutableArray.Create(buf) }];
        }
        if (x.IntData is { } id)
        {
            var buf = new long[count];
            for (int i = 0; i < count; i++) buf[i] = TryMapToInput(i, out var src) ? id[(int)src] : 0L;
            return [rt with { IntData = ImmutableArray.Create(buf) }];
        }
        if (x.BoolData is { } bd)
        {
            var buf = new bool[count];
            for (int i = 0; i < count; i++) buf[i] = TryMapToInput(i, out var src) && bd[(int)src];
            return [rt with { BoolData = ImmutableArray.Create(buf) }];
        }
        return [rt];
    }
}
