using System.Collections.Immutable;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// QEE kernel for ONNX <c>DynamicQuantizeLinear</c>. Three outputs:
///   (y: uint8 tensor matching input shape, y_scale: float32 scalar, y_zero_point: uint8 scalar).
/// Concrete values follow the spec formula:
///   y_scale = (max(0, max(x)) − min(0, min(x))) / 255,
///   y_zero_point = round_half_even(saturate(0 − min(0, min(x)) / y_scale)),
///   y = saturate(round_half_even(x / y_scale) + y_zero_point).
/// A zero scale (all-zero input) blocks value computation rather than dividing by zero.
/// </summary>
internal sealed class DynamicQuantizeLinearOp : QuickOp
{
    public override string OpCode => OpCodes.DYNAMIC_QUANTIZE_LINEAR;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs.Length > 0 ? inputs[0] : null;
        var scalarShape = new Shape(Array.Empty<long>());
        var y = RuntimeTensorFactory.Create(DType.UInt8, x?.Shape);
        var yScale = RuntimeTensorFactory.Create(DType.Float32, scalarShape);
        var yZeroPoint = RuntimeTensorFactory.Create(DType.UInt8, scalarShape);

        if (x?.Shape is not null && x.FloatData is { Length: > 0 } xd
            && RuntimeTensorFactory.ShouldStoreData(x.Shape, maxDataElements))
        {
            float min = 0f, max = 0f;
            foreach (var v in xd)
            {
                if (v < min) min = v;
                if (v > max) max = v;
            }
            float scale = (max - min) / 255f;
            if (scale > 0f && float.IsFinite(scale))
            {
                float zpF = QuantizeHelpers.RoundHalfEven(Math.Clamp(0f - min / scale, 0f, 255f));
                long zp = (long)zpF;
                var buf = new long[xd.Length];
                for (int i = 0; i < buf.Length; i++)
                    buf[i] = (long)Math.Clamp(QuantizeHelpers.RoundHalfEven(xd[i] / scale) + zp, 0f, 255f);

                y = y with { IntData = ImmutableArray.Create(buf) };
                yScale = yScale with { FloatData = ImmutableArray.Create(scale) };
                yZeroPoint = yZeroPoint with { IntData = ImmutableArray.Create(zp) };
            }
        }

        return [y, yScale, yZeroPoint];
    }
}
