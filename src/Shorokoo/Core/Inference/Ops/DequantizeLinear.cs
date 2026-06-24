using System.Collections.Immutable;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// QEE kernel for ONNX <c>DequantizeLinear</c> (opset 21). Output shape matches the
/// input; dtype is taken from <c>x_scale</c> (defaults to float32). Concrete values
/// (<c>(x − x_zero_point) * x_scale</c>) are computed for the per-tensor and per-axis
/// (1-D scale along <c>axis</c>, default 1) layouts; blocked dequantization
/// (block_size != 0) is shape/dtype-only.
/// </summary>
internal sealed class DequantizeLinearOp : QuickOp
{
    public override string OpCode => OpCodes.DEQUANTIZE_LINEAR;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs.Length > 0 ? inputs[0] : null;
        var scale = inputs.Length > 1 ? inputs[1] : null;
        var zeroPoint = inputs.Length > 2 ? inputs[2] : null;
        // opset-23 output_dtype (TensorProto dtype number) overrides the
        // x_scale-derived output type when set.
        var outputDtypeNum = attrs.GetLongVal(OnnxOpAttributeNames.AttrOutputDtype);
        var dtype = outputDtypeNum is { } n ? (DType)(int)n
            : scale?.DType ?? DType.Float32;
        var rt = RuntimeTensorFactory.Create(dtype, x?.Shape);

        if (x?.Shape is null || x.IntData is not { } xd
            || !RuntimeTensorFactory.ShouldStoreData(x.Shape, maxDataElements))
            return [rt];
        if (AttrAccess.GetLong(attrs, OnnxOpAttributeNames.AttrBlockSize, 0) != 0) return [rt];
        if (scale?.FloatData is not { } sd || sd.Length == 0) return [rt];

        var axis = AttrAccess.GetLong(attrs, OnnxOpAttributeNames.AttrAxis, 1);
        if (!QuantizeHelpers.TryResolveLayout(x.Shape.Dims, axis, sd.Length, out var channelOf))
            return [rt];

        ImmutableArray<long>? zd = null;
        if (zeroPoint is not null)
        {
            if (zeroPoint.IntData is not { } z || z.Length != sd.Length) return [rt];
            zd = z;
        }

        var buf = new float[xd.Length];
        for (int i = 0; i < buf.Length; i++)
        {
            int c = channelOf(i);
            buf[i] = (xd[i] - (zd?[c] ?? 0)) * sd[c];
        }
        return [rt with { FloatData = ImmutableArray.Create(buf) }];
    }
}
