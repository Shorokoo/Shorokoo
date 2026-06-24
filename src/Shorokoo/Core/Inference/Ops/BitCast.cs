using System.Collections.Immutable;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Inference.Helpers;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// QEE kernel for ONNX <c>BitCast</c> (opset 26+): bitwise reinterpretation to a
/// same-width dtype. Per the spec the target must have the same bit-width, so the
/// output shape always equals the input shape; the dtype comes from the <c>to</c>
/// attribute. Values are computed for the pairs QEE's storage model can represent
/// exactly:
/// <list type="bullet">
///   <item>float32 ↔ int32/uint32 (FloatData is true float32 storage);</item>
///   <item>signed ↔ unsigned integer reinterpret at widths 8/16/32/64 (IntData
///         masked to the width).</item>
/// </list>
/// Other pairs (float64/float16/bfloat16 sources or targets, bool) propagate
/// shape+dtype only — QEE stores float data as float32 and bool as bool, so their
/// exact bit patterns are not representable.
/// </summary>
internal sealed class BitCastOp : QuickOp
{
    public override string OpCode => OpCodes.BIT_CAST;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs.Length > 0 ? inputs[0] : null;
        var toDType = attrs.GetDTypeVal(OnnxOpAttributeNames.AttrTo) ?? x?.DType ?? DType.Float32;
        var rt = RuntimeTensorFactory.Create(toDType, x?.Shape);

        if (x is null || x.DType == DType.Invalid || !RuntimeTensorFactory.ShouldStoreData(rt.Shape, maxDataElements))
            return [rt];

        var from = x.DType;

        // float32 → int32/uint32
        if (from == DType.Float32 && (toDType == DType.Int32 || toDType == DType.UInt32) && x.FloatData is { } fd)
        {
            var buf = new long[fd.Length];
            for (int i = 0; i < buf.Length; i++)
            {
                int bits = BitConverter.SingleToInt32Bits(fd[i]);
                buf[i] = toDType == DType.Int32 ? bits : (uint)bits;
            }
            return [rt with { IntData = ImmutableArray.Create(buf) }];
        }

        // int32/uint32 → float32
        if ((from == DType.Int32 || from == DType.UInt32) && toDType == DType.Float32 && x.IntData is { } id32)
        {
            var buf = new float[id32.Length];
            for (int i = 0; i < buf.Length; i++)
                buf[i] = BitConverter.Int32BitsToSingle(unchecked((int)id32[i]));
            return [rt with { FloatData = ImmutableArray.Create(buf) }];
        }

        // signed ↔ unsigned integer reinterpret at the same width
        if (x.IntData is { } id && IntWidth(from) is { } w && IntWidth(toDType) == w)
        {
            var buf = new long[id.Length];
            for (int i = 0; i < buf.Length; i++)
                buf[i] = Reinterpret(id[i], w, signedTarget: IsSignedInt(toDType));
            return [rt with { IntData = ImmutableArray.Create(buf) }];
        }

        return [rt];
    }

    private static int? IntWidth(DType dtype)
    {
        if (dtype == DType.Int8 || dtype == DType.UInt8) return 8;
        if (dtype == DType.Int16 || dtype == DType.UInt16) return 16;
        if (dtype == DType.Int32 || dtype == DType.UInt32) return 32;
        if (dtype == DType.Int64 || dtype == DType.UInt64) return 64;
        return null;
    }

    private static bool IsSignedInt(DType dtype)
        => dtype == DType.Int8 || dtype == DType.Int16 || dtype == DType.Int32 || dtype == DType.Int64;

    /// <summary>Reinterprets the low <paramref name="width"/> bits of <paramref name="v"/>
    /// as a signed or unsigned integer of that width (uint64 values beyond long.MaxValue
    /// wrap, matching QEE's 64-bit-storage convention for unsigned data).</summary>
    private static long Reinterpret(long v, int width, bool signedTarget)
    {
        if (width == 64) return v;
        long mask = (1L << width) - 1;
        long bits = v & mask;
        if (signedTarget && (bits & (1L << (width - 1))) != 0)
            bits -= 1L << width;
        return bits;
    }
}
