using System.Collections.Immutable;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// Shared helpers for the QuantizeLinear / DequantizeLinear value paths: scale/zero-point
/// layout resolution (per-tensor scalar vs per-axis 1-D along <c>axis</c>, default 1) and
/// the saturation range of each representable quantized dtype. Blocked quantization
/// (block_size != 0, opset 21) keeps correct shapes/dtypes but blocks value computation.
/// </summary>
internal static class QuantizeHelpers
{
    /// <summary>Saturation range for the quantized dtypes representable in-framework.
    /// (float8/int4 dtypes are not supported in-framework, so no range for them.)</summary>
    public static (long Min, long Max)? QuantRange(DType dtype)
    {
        if (dtype == DType.Int8) return (sbyte.MinValue, sbyte.MaxValue);
        if (dtype == DType.UInt8) return (byte.MinValue, byte.MaxValue);
        if (dtype == DType.Int16) return (short.MinValue, short.MaxValue);
        if (dtype == DType.UInt16) return (ushort.MinValue, ushort.MaxValue);
        if (dtype == DType.Int32) return (int.MinValue, int.MaxValue);
        if (dtype == DType.UInt32) return (uint.MinValue, uint.MaxValue);
        return null;
    }

    /// <summary>
    /// Resolves the per-element scale/zero-point index function. Returns false when the
    /// layout can't be safely determined (unknown dims, scale length not matching the
    /// axis dim, axis out of range). <paramref name="scaleCount"/> is 1 for per-tensor.
    /// </summary>
    public static bool TryResolveLayout(long[] dims, long axisAttr, int scaleCount, out Func<long, int> channelOf)
    {
        channelOf = _ => 0;
        if (scaleCount == 1) return true;
        if (dims.Any(d => d < 0)) return false;
        var axis = (int)(axisAttr < 0 ? axisAttr + dims.Length : axisAttr);
        if (axis < 0 || axis >= dims.Length) return false;
        if (dims[axis] != scaleCount) return false;
        long inner = 1;
        for (int d = axis + 1; d < dims.Length; d++) inner *= dims[d];
        long axisDim = dims[axis];
        var innerC = inner;
        channelOf = i => (int)(i / innerC % axisDim);
        return true;
    }

    /// <summary>Round-half-to-even, as the ONNX quantization ops specify.</summary>
    public static float RoundHalfEven(float v) => MathF.Round(v, MidpointRounding.ToEven);
}

/// <summary>
/// QEE kernel for ONNX <c>QuantizeLinear</c> (opset 21). Output shape matches the input;
/// dtype is the <c>y_zero_point</c> input's dtype when present, else the
/// <c>output_dtype</c> attribute, else uint8. Concrete values
/// (<c>saturate(round_half_even(x / y_scale) + y_zero_point)</c>) are computed for the
/// per-tensor and per-axis layouts; blocked quantization (block_size != 0) is
/// shape/dtype-only. The <c>saturate</c> attribute only affects float8 targets, which
/// aren't representable in-framework.
/// </summary>
internal sealed class QuantizeLinearOp : QuickOp
{
    public override string OpCode => OpCodes.QUANTIZE_LINEAR;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs.Length > 0 ? inputs[0] : null;
        var scale = inputs.Length > 1 ? inputs[1] : null;
        var zeroPoint = inputs.Length > 2 ? inputs[2] : null;
        var dtype = zeroPoint?.DType
                 ?? (attrs.IsAttributeDefined(OnnxOpAttributeNames.AttrOutputDtype)
                        ? attrs.GetDTypeVal(OnnxOpAttributeNames.AttrOutputDtype) : null)
                 ?? DType.UInt8;
        var rt = RuntimeTensorFactory.Create(dtype, x?.Shape);

        if (x?.Shape is null || x.FloatData is not { } xd
            || !RuntimeTensorFactory.ShouldStoreData(x.Shape, maxDataElements))
            return [rt];
        if (AttrAccess.GetLong(attrs, OnnxOpAttributeNames.AttrBlockSize, 0) != 0) return [rt];
        if (QuantizeHelpers.QuantRange(dtype) is not { } range) return [rt];
        if (scale?.FloatData is not { } sd || sd.Length == 0) return [rt];
        if (sd.Any(s => s == 0f)) return [rt];

        var axis = AttrAccess.GetLong(attrs, OnnxOpAttributeNames.AttrAxis, 1);
        if (!QuantizeHelpers.TryResolveLayout(x.Shape.Dims, axis, sd.Length, out var channelOf))
            return [rt];

        ImmutableArray<long>? zd = null;
        if (zeroPoint is not null)
        {
            if (zeroPoint.IntData is not { } z || z.Length != sd.Length) return [rt];
            zd = z;
        }

        var buf = new long[xd.Length];
        for (int i = 0; i < buf.Length; i++)
        {
            int c = channelOf(i);
            double v = QuantizeHelpers.RoundHalfEven(xd[i] / sd[c]) + (zd?[c] ?? 0);
            buf[i] = (long)Math.Clamp(v, (double)range.Min, (double)range.Max);
        }
        return [rt with { IntData = ImmutableArray.Create(buf) }];
    }
}
