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

namespace Shorokoo.Core.Inference.Helpers;

/// <summary>
/// Base class for elementwise unary float operators. Handles shape/dtype propagation and applies
/// the supplied scalar function to every element when the input data is available.
/// </summary>
internal abstract class UnaryFloatOp : QuickOp
{
    protected abstract float Apply(float x);

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs.Length > 0 ? inputs[0] : null;
        var dtype = x?.DType ?? DType.Float32;

        ImmutableArray<float>? data = null;
        if (x?.FloatData is { } fd && RuntimeTensorFactory.ShouldStoreData(x.Shape, maxDataElements))
        {
            var buf = new float[fd.Length];
            for (int i = 0; i < buf.Length; i++) buf[i] = Apply(fd[i]);
            data = ImmutableArray.Create(buf);
        }

        return [new RuntimeTensor
        {
            DType = dtype,
            Shape = x?.Shape,
            MaxShape = x?.MaxShape ?? x?.Shape,
            Rank = x?.Rank,
            MaxRank = x?.MaxRank ?? x?.Rank,
            FloatData = data,
        }];
    }
}

/// <summary>
/// Base class for elementwise unary ops that accept both floats and integers and preserve the
/// input's dtype. Derived classes supply per-category scalar functions.
/// </summary>
internal abstract class UnaryNumericOp : QuickOp
{
    protected virtual float ApplyFloat(float x) => x;
    protected virtual long ApplyInt(long x) => x;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs.Length > 0 ? inputs[0] : null;
        var dtype = x?.DType ?? DType.Float32;

        ImmutableArray<float>? fData = null;
        ImmutableArray<long>? iData = null;
        var shouldStore = RuntimeTensorFactory.ShouldStoreData(x?.Shape, maxDataElements);
        if (x?.FloatData is { } fd && shouldStore)
        {
            var buf = new float[fd.Length];
            for (int i = 0; i < buf.Length; i++) buf[i] = ApplyFloat(fd[i]);
            fData = ImmutableArray.Create(buf);
        }
        else if (x?.IntData is { } id && shouldStore)
        {
            var buf = new long[id.Length];
            for (int i = 0; i < buf.Length; i++) buf[i] = ApplyInt(id[i]);
            iData = ImmutableArray.Create(buf);
        }

        return [new RuntimeTensor
        {
            DType = dtype,
            Shape = x?.Shape,
            MaxShape = x?.MaxShape ?? x?.Shape,
            Rank = x?.Rank,
            MaxRank = x?.MaxRank ?? x?.Rank,
            FloatData = fData,
            IntData = iData,
        }];
    }
}

/// <summary>
/// Base class for binary elementwise numeric operators with full ONNX broadcasting.
/// </summary>
internal abstract class BinaryNumericOp : QuickOp
{
    protected virtual float ApplyFloat(float a, float b) => a;
    protected virtual long ApplyInt(long a, long b) => a;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var a = inputs.Length > 0 ? inputs[0] : null;
        var b = inputs.Length > 1 ? inputs[1] : null;
        var dtype = a?.DType ?? b?.DType ?? DType.Float32;
        var shape = ShapeHelpers.Broadcast(a?.Shape, b?.Shape);

        ImmutableArray<float>? fData = null;
        ImmutableArray<long>? iData = null;
        if (shape is not null && RuntimeTensorFactory.ShouldStoreData(shape, maxDataElements)
            && a is not null && b is not null)
        {
            if (a.FloatData is { } af && b.FloatData is { } bf)
                fData = ImmutableArray.Create(ElementwiseBroadcast.Float(af, a.Shape!, bf, b.Shape!, shape, ApplyFloat));
            else if (a.IntData is { } ai && b.IntData is { } bi)
                iData = ImmutableArray.Create(ElementwiseBroadcast.Int(ai, a.Shape!, bi, b.Shape!, shape, ApplyInt));
        }

        return [RuntimeTensorFactory.Create(dtype, shape) with { FloatData = fData, IntData = iData }];
    }
}

/// <summary>
/// Base class for comparison operators: broadcasts shapes and produces a bool tensor.
/// </summary>
internal abstract class CompareOp : QuickOp
{
    protected virtual bool CompareFloat(float a, float b) => false;
    protected virtual bool CompareInt(long a, long b) => false;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var a = inputs.Length > 0 ? inputs[0] : null;
        var b = inputs.Length > 1 ? inputs[1] : null;
        var shape = ShapeHelpers.Broadcast(a?.Shape, b?.Shape);

        ImmutableArray<bool>? bData = null;
        if (shape is not null && RuntimeTensorFactory.ShouldStoreData(shape, maxDataElements)
            && a is not null && b is not null)
        {
            if (a.FloatData is { } af && b.FloatData is { } bf)
                bData = ImmutableArray.Create(ElementwiseBroadcast.BoolFromFloat(af, a.Shape!, bf, b.Shape!, shape, CompareFloat));
            else if (a.IntData is { } ai && b.IntData is { } bi)
                bData = ImmutableArray.Create(ElementwiseBroadcast.BoolFromInt(ai, a.Shape!, bi, b.Shape!, shape, CompareInt));
            else if (a.BoolData is { } ab && b.BoolData is { } bb)
                // Equal supports bool tensors (opset 19+): compare them as 0/1 integers.
                bData = ImmutableArray.Create(ElementwiseBroadcast.Bool(ab, a.Shape!, bb, b.Shape!, shape,
                    (x, y) => CompareInt(x ? 1 : 0, y ? 1 : 0)));
        }

        return [RuntimeTensorFactory.Create(DType.Bool, shape) with { BoolData = bData }];
    }
}

/// <summary>
/// Shared implementation for ONNX-style reductions. Subclasses override the accumulator.
/// </summary>
internal abstract class ReduceOpBase : QuickOp
{
    protected abstract float Reduce(IEnumerable<float> values);
    protected virtual long ReduceInt(IEnumerable<long> values) => (long)Reduce(values.Select(v => (float)v));

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs.Length > 0 ? inputs[0] : null;
        if (x is null) return [RuntimeTensorFactory.Create(DType.Float32, null)];

        var keepDims = AttrAccess.GetBool(attrs, OnnxOpAttributeNames.AttrKeepdims, true);
        var noopWithEmptyAxes = AttrAccess.GetBool(attrs, OnnxOpAttributeNames.AttrNoopWithEmptyAxes, false);

        long[]? axes = null;
        var axesInput = inputs.Length > 1 ? inputs[1] : null;
        if (axesInput?.IntData is { } idxData) axes = idxData.ToArray();
        else if (axesInput is not null)
            // The axes input is connected but its values are unknown at QEE time — which dims
            // get reduced is unknowable, so degrade to an unknown shape (never guess).
            return [RuntimeTensorFactory.Create(x.DType, null)];
        if (axes is null)
            axes = AttrAccess.GetLongs(attrs, OnnxOpAttributeNames.AttrAxes);

        Shape? resultShape = null;
        if (x.Shape is not null)
        {
            var dims = x.Shape.Dims.ToArray();
            if (axes is null || axes.Length == 0)
            {
                if (noopWithEmptyAxes)
                    resultShape = x.Shape;
                else
                    resultShape = keepDims
                        ? new Shape(Enumerable.Repeat(1L, dims.Length).ToArray())
                        : new Shape(Array.Empty<long>());
            }
            else
            {
                var normalizedAxes = axes.Select(a => a < 0 ? a + dims.Length : a).OrderByDescending(a => a).ToArray();
                if (keepDims)
                {
                    foreach (var ax in normalizedAxes) dims[(int)ax] = 1;
                    resultShape = new Shape(dims);
                }
                else
                {
                    var dimsList = dims.ToList();
                    foreach (var ax in normalizedAxes) dimsList.RemoveAt((int)ax);
                    resultShape = new Shape(dimsList.ToArray());
                }
            }
        }

        var rt = RuntimeTensorFactory.Create(x.DType, resultShape);

        // Propagate concrete data when the input has it and the result is small enough to store.
        // Needed so shape-construction chains like Shape → Slice → ReduceProd stay concrete.
        if (resultShape is null || !RuntimeTensorFactory.ShouldStoreData(resultShape, maxDataElements))
            return [rt];

        // noop_with_empty_axes with no axes: the op is an identity — pass the data through
        // unchanged (the reduce path below would wrongly fold everything into one element).
        if ((axes is null || axes.Length == 0) && noopWithEmptyAxes)
            return [rt with { FloatData = x.FloatData, IntData = x.IntData, BoolData = x.BoolData }];

        var inDims = x.Shape!.Dims;
        var normalizedAxesAll = (axes is null || axes.Length == 0)
            ? Enumerable.Range(0, inDims.Length).Select(i => (long)i).ToArray()
            : axes.Select(a => a < 0 ? a + inDims.Length : a).ToArray();
        var axisSet = new HashSet<long>(normalizedAxesAll);

        long keptCount = 1;
        for (int d = 0; d < inDims.Length; d++) if (!axisSet.Contains(d)) keptCount *= inDims[d];

        long[] inStrides = new long[inDims.Length];
        long s = 1;
        for (int i = inDims.Length - 1; i >= 0; i--) { inStrides[i] = s; s *= inDims[i]; }

        if (x.FloatData is { } fdata)
        {
            var outBuf = new float[keptCount];
            ReduceInto(inDims, inStrides, axisSet, keptCount, (long srcBase, int[] axisShape, long[] axisStrides) =>
            {
                return Reduce(EnumerateGroup(fdata, srcBase, axisShape, axisStrides).Select(v => v));
            }, outBuf);
            return [rt with { FloatData = ImmutableArray.Create(outBuf) }];
        }
        if (x.IntData is { } idata)
        {
            var outBuf = new long[keptCount];
            ReduceInto(inDims, inStrides, axisSet, keptCount, (long srcBase, int[] axisShape, long[] axisStrides) =>
            {
                return ReduceInt(EnumerateGroup(idata, srcBase, axisShape, axisStrides));
            }, outBuf);
            return [rt with { IntData = ImmutableArray.Create(outBuf) }];
        }
        if (x.BoolData is { } bdata)
        {
            // Reduce bools by mapping to 0/1 and delegating to ReduceInt. Works for Max
            // (→ any), Min (→ all), Sum (→ count), Prod (→ all). Result mapped back to bool
            // by checking >0 for anything that semantically means "at least one true".
            var idataFromBool = ImmutableArray.CreateRange(bdata.Select(v => v ? 1L : 0L));
            var outBuf = new bool[keptCount];
            ReduceInto(inDims, inStrides, axisSet, keptCount, (long srcBase, int[] axisShape, long[] axisStrides) =>
            {
                return ReduceInt(EnumerateGroup(idataFromBool, srcBase, axisShape, axisStrides)) != 0;
            }, outBuf);
            return [rt with { BoolData = ImmutableArray.Create(outBuf) }];
        }

        return [rt];
    }

    private static void ReduceInto<T>(long[] inDims, long[] inStrides, HashSet<long> axisSet, long keptCount,
        Func<long, int[], long[], T> compute, T[] outBuf)
    {
        var keptAxes = new List<int>();
        var redAxes = new List<int>();
        for (int d = 0; d < inDims.Length; d++) (axisSet.Contains(d) ? redAxes : keptAxes).Add(d);

        int[] redShape = redAxes.Select(d => (int)inDims[d]).ToArray();
        long[] redStrides = redAxes.Select(d => inStrides[d]).ToArray();

        var keptIdx = new int[keptAxes.Count];
        for (long flat = 0; flat < keptCount; flat++)
        {
            long srcBase = 0;
            for (int k = 0; k < keptAxes.Count; k++) srcBase += keptIdx[k] * inStrides[keptAxes[k]];
            outBuf[flat] = compute(srcBase, redShape, redStrides);

            for (int k = keptAxes.Count - 1; k >= 0; k--)
            {
                keptIdx[k]++;
                if (keptIdx[k] < (int)inDims[keptAxes[k]]) break;
                keptIdx[k] = 0;
            }
        }
    }

    private static IEnumerable<float> EnumerateGroup(ImmutableArray<float> data, long srcBase, int[] axisShape, long[] axisStrides)
    {
        if (axisShape.Length == 0) { yield return data[(int)srcBase]; yield break; }
        var idx = new int[axisShape.Length];
        while (true)
        {
            long off = srcBase;
            for (int d = 0; d < axisShape.Length; d++) off += idx[d] * axisStrides[d];
            yield return data[(int)off];

            int k = axisShape.Length - 1;
            while (k >= 0 && ++idx[k] == axisShape[k]) { idx[k] = 0; k--; }
            if (k < 0) yield break;
        }
    }

    private static IEnumerable<long> EnumerateGroup(ImmutableArray<long> data, long srcBase, int[] axisShape, long[] axisStrides)
    {
        if (axisShape.Length == 0) { yield return data[(int)srcBase]; yield break; }
        var idx = new int[axisShape.Length];
        while (true)
        {
            long off = srcBase;
            for (int d = 0; d < axisShape.Length; d++) off += idx[d] * axisStrides[d];
            yield return data[(int)off];

            int k = axisShape.Length - 1;
            while (k >= 0 && ++idx[k] == axisShape[k]) { idx[k] = 0; k--; }
            if (k < 0) yield break;
        }
    }
}

/// <summary>
/// Folds the variadic elementwise ops (Min / Max / Sum / Mean) over any number of inputs
/// with full multidirectional broadcasting. Used by their QuickOps to propagate concrete
/// values when every input carries data and the broadcast result is small.
/// </summary>
internal static class VariadicElementwise
{
    /// <summary>
    /// Left-fold of <paramref name="op"/> over the inputs' FloatData, broadcast into
    /// <paramref name="outShape"/>. Returns null when any input lacks shape or float data.
    /// </summary>
    public static float[]? FoldFloat(RuntimeTensor?[] inputs, Shape outShape, Func<float, float, float> op)
    {
        var present = new List<RuntimeTensor>();
        foreach (var i in inputs)
        {
            if (i?.Shape is null || i.FloatData is null) return null;
            present.Add(i);
        }
        if (present.Count == 0) return null;

        // Broadcast the first input into the output shape via an identity combine.
        var acc = ElementwiseBroadcast.Float(
            present[0].FloatData!.Value, present[0].Shape!,
            present[0].FloatData!.Value, present[0].Shape!, outShape, (x, _) => x);
        for (int k = 1; k < present.Count; k++)
            acc = ElementwiseBroadcast.Float(
                ImmutableArray.Create(acc), outShape,
                present[k].FloatData!.Value, present[k].Shape!, outShape, op);
        return acc;
    }

    /// <summary>Integer counterpart of <see cref="FoldFloat"/>.</summary>
    public static long[]? FoldInt(RuntimeTensor?[] inputs, Shape outShape, Func<long, long, long> op)
    {
        var present = new List<RuntimeTensor>();
        foreach (var i in inputs)
        {
            if (i?.Shape is null || i.IntData is null) return null;
            present.Add(i);
        }
        if (present.Count == 0) return null;

        var acc = ElementwiseBroadcast.Int(
            present[0].IntData!.Value, present[0].Shape!,
            present[0].IntData!.Value, present[0].Shape!, outShape, (x, _) => x);
        for (int k = 1; k < present.Count; k++)
            acc = ElementwiseBroadcast.Int(
                ImmutableArray.Create(acc), outShape,
                present[k].IntData!.Value, present[k].Shape!, outShape, op);
        return acc;
    }
}

/// <summary>
/// Utility routines for broadcasting two immutable buffers into a destination of a given shape.
/// </summary>
internal static class ElementwiseBroadcast
{
    public static float[] Float(ImmutableArray<float> a, Shape aShape, ImmutableArray<float> b, Shape bShape, Shape outShape, Func<float, float, float> op)
    {
        var dst = new float[outShape.Count];
        Iterate(aShape, bShape, outShape, (ia, ib, i) => dst[i] = op(a[ia], b[ib]));
        return dst;
    }

    public static long[] Int(ImmutableArray<long> a, Shape aShape, ImmutableArray<long> b, Shape bShape, Shape outShape, Func<long, long, long> op)
    {
        var dst = new long[outShape.Count];
        Iterate(aShape, bShape, outShape, (ia, ib, i) => dst[i] = op(a[ia], b[ib]));
        return dst;
    }

    public static bool[] BoolFromFloat(ImmutableArray<float> a, Shape aShape, ImmutableArray<float> b, Shape bShape, Shape outShape, Func<float, float, bool> op)
    {
        var dst = new bool[outShape.Count];
        Iterate(aShape, bShape, outShape, (ia, ib, i) => dst[i] = op(a[ia], b[ib]));
        return dst;
    }

    public static bool[] BoolFromInt(ImmutableArray<long> a, Shape aShape, ImmutableArray<long> b, Shape bShape, Shape outShape, Func<long, long, bool> op)
    {
        var dst = new bool[outShape.Count];
        Iterate(aShape, bShape, outShape, (ia, ib, i) => dst[i] = op(a[ia], b[ib]));
        return dst;
    }

    /// <summary>Broadcast-combine two bool tensors via an element-wise bool op.</summary>
    public static bool[] Bool(ImmutableArray<bool> a, Shape aShape, ImmutableArray<bool> b, Shape bShape, Shape outShape, Func<bool, bool, bool> op)
    {
        var dst = new bool[outShape.Count];
        Iterate(aShape, bShape, outShape, (ia, ib, i) => dst[i] = op(a[ia], b[ib]));
        return dst;
    }

    /// <summary>
    /// Broadcast three bool tensors (condition, thenValue, elseValue) into a single bool output
    /// using <see cref="Shorokoo.Core.Inference.Ops.WhereOp"/>'s element-wise
    /// selection rule: <c>result[i] = cond[i] ? thenValue[i] : elseValue[i]</c>.
    /// </summary>
    public static bool[] BoolWhere(
        ImmutableArray<bool> cond, Shape condShape,
        ImmutableArray<bool> thenV, Shape thenShape,
        ImmutableArray<bool> elseV, Shape elseShape,
        Shape outShape)
    {
        var dst = new bool[outShape.Count];
        Iterate3(condShape, thenShape, elseShape, outShape, (ic, it, ie, i) =>
            dst[i] = cond[ic] ? thenV[it] : elseV[ie]);
        return dst;
    }

    /// <summary>Float counterpart of <see cref="BoolWhere"/> (3-way broadcast select).</summary>
    public static float[] FloatWhere(
        ImmutableArray<bool> cond, Shape condShape,
        ImmutableArray<float> thenV, Shape thenShape,
        ImmutableArray<float> elseV, Shape elseShape,
        Shape outShape)
    {
        var dst = new float[outShape.Count];
        Iterate3(condShape, thenShape, elseShape, outShape, (ic, it, ie, i) =>
            dst[i] = cond[ic] ? thenV[it] : elseV[ie]);
        return dst;
    }

    /// <summary>Integer counterpart of <see cref="BoolWhere"/> (3-way broadcast select).</summary>
    public static long[] IntWhere(
        ImmutableArray<bool> cond, Shape condShape,
        ImmutableArray<long> thenV, Shape thenShape,
        ImmutableArray<long> elseV, Shape elseShape,
        Shape outShape)
    {
        var dst = new long[outShape.Count];
        Iterate3(condShape, thenShape, elseShape, outShape, (ic, it, ie, i) =>
            dst[i] = cond[ic] ? thenV[it] : elseV[ie]);
        return dst;
    }

    private static void Iterate3(Shape aShape, Shape bShape, Shape cShape, Shape outShape, Action<int, int, int, int> write)
    {
        var aDims = aShape.Dims;
        var bDims = bShape.Dims;
        var cDims = cShape.Dims;
        var oDims = outShape.Dims;
        var total = (int)outShape.Count;
        var rank = oDims.Length;

        var aStride = BroadcastStrides(aDims, oDims);
        var bStride = BroadcastStrides(bDims, oDims);
        var cStride = BroadcastStrides(cDims, oDims);

        var idx = new int[rank];
        for (int i = 0; i < total; i++)
        {
            int ia = 0, ib = 0, ic = 0;
            for (int d = 0; d < rank; d++)
            {
                ia += idx[d] * aStride[d];
                ib += idx[d] * bStride[d];
                ic += idx[d] * cStride[d];
            }
            write(ia, ib, ic, i);
            for (int d = rank - 1; d >= 0; d--)
            {
                if (++idx[d] < oDims[d]) break;
                idx[d] = 0;
            }
        }
    }

    private static void Iterate(Shape aShape, Shape bShape, Shape outShape, Action<int, int, int> write)
    {
        var aDims = aShape.Dims;
        var bDims = bShape.Dims;
        var oDims = outShape.Dims;
        var total = (int)outShape.Count;
        var rank = oDims.Length;

        var aStride = BroadcastStrides(aDims, oDims);
        var bStride = BroadcastStrides(bDims, oDims);

        var idx = new int[rank];
        for (int i = 0; i < total; i++)
        {
            int ia = 0, ib = 0;
            for (int d = 0; d < rank; d++)
            {
                ia += idx[d] * aStride[d];
                ib += idx[d] * bStride[d];
            }
            write(ia, ib, i);
            for (int d = rank - 1; d >= 0; d--)
            {
                if (++idx[d] < oDims[d]) break;
                idx[d] = 0;
            }
        }
    }

    private static int[] BroadcastStrides(long[] srcDims, long[] outDims)
    {
        var strides = new int[outDims.Length];
        int stride = 1;
        int srcOffset = outDims.Length - srcDims.Length;
        for (int d = srcDims.Length - 1; d >= 0; d--)
        {
            var odIdx = d + srcOffset;
            strides[odIdx] = srcDims[d] == 1 ? 0 : stride;
            stride *= (int)srcDims[d];
        }
        return strides;
    }
}
