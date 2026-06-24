using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Onnx;

namespace Shorokoo.Core.Inference.Helpers;

internal static class ShapeHelpers
{
    /// <summary>
    /// Performs ONNX-style multidirectional broadcasting on two shapes.
    /// Returns null if either shape is null.
    /// </summary>
    public static Shape? Broadcast(Shape? a, Shape? b)
    {
        if (a is null || b is null) return null;
        var aDims = a.Dims;
        var bDims = b.Dims;
        var rank = Math.Max(aDims.Length, bDims.Length);
        var result = new long[rank];
        for (int i = 0; i < rank; i++)
        {
            var da = i < aDims.Length ? aDims[aDims.Length - 1 - i] : 1;
            var db = i < bDims.Length ? bDims[bDims.Length - 1 - i] : 1;
            if (da == 1) result[rank - 1 - i] = db;
            else if (db == 1) result[rank - 1 - i] = da;
            else result[rank - 1 - i] = Math.Max(da, db);
        }
        return new Shape(result);
    }

    public static Shape? Broadcast(params Shape?[] shapes)
    {
        Shape? acc = null;
        bool started = false;
        foreach (var s in shapes)
        {
            if (s is null) return null;
            if (!started) { acc = s; started = true; }
            else acc = Broadcast(acc, s);
        }
        return acc;
    }

    /// <summary>
    /// Computes one spatial output dim for the ONNX pooling family (MaxPool / AveragePool /
    /// LpPool) per the opset-21 spec: effective kernel <c>(k - 1) * dilation + 1</c>,
    /// <c>auto_pad</c> handling (SAME_* → <c>ceil(in / stride)</c>, VALID → no padding), and
    /// <c>ceil_mode</c> (ceil instead of floor division, clamped so the last pooling window
    /// starts inside the input or its begin padding rather than entirely in the end padding).
    /// </summary>
    public static long PoolOutputSize(
        long inSize, long kernel, long stride, long dilation,
        long padBegin, long padEnd, AutoPad autoPad, bool ceilMode)
    {
        if (autoPad == AutoPad.SameUpper || autoPad == AutoPad.SameLower)
            return (inSize + stride - 1) / stride;

        if (autoPad == AutoPad.Valid) { padBegin = 0; padEnd = 0; }
        var effectiveKernel = (kernel - 1) * dilation + 1;
        var numerator = inSize + padBegin + padEnd - effectiveKernel;
        if (!ceilMode) return numerator / stride + 1;

        var outSize = (numerator + stride - 1) / stride + 1;
        // ONNX requires the last window to start inside the input (or its begin padding);
        // a window starting entirely inside the end padding is dropped.
        if ((outSize - 1) * stride >= inSize + padBegin) outSize--;
        return outSize;
    }
}
