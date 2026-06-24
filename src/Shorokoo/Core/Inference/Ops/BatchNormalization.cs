using System.Collections.Immutable;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// QEE kernel for ONNX <c>BatchNormalization</c> (opset 15: epsilon / momentum /
/// training_mode attributes; with training_mode=1 the op has the optional running_mean /
/// running_var outputs). All three outputs are always emitted; the engine drops any
/// beyond the graph's declared output count. In inference mode (training_mode=0, the
/// default) concrete y values are computed as the per-channel affine
/// <c>(x − input_mean) / sqrt(input_var + epsilon) * scale + B</c>; training mode uses
/// batch statistics, so values are not computed there.
/// </summary>
internal sealed class BatchNormalizationOp : QuickOp
{
    public override string OpCode => OpCodes.BATCH_NORMALIZATION;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs.Length > 0 ? inputs[0] : null;
        var scale = inputs.Length > 1 ? inputs[1] : null;
        var bias = inputs.Length > 2 ? inputs[2] : null;
        var inputMean = inputs.Length > 3 ? inputs[3] : null;
        var inputVar = inputs.Length > 4 ? inputs[4] : null;

        var yDType = x?.DType ?? DType.Float32;
        var statDType = inputMean?.DType ?? inputVar?.DType ?? yDType;

        Shape? channelShape = x?.Shape is not null && x.Shape.Dims.Length >= 2
            ? new Shape(new[] { x.Shape.Dims[1] })
            : null;

        var y = RuntimeTensorFactory.Create(yDType, x?.Shape);

        var trainingMode = AttrAccess.GetBool(attrs, OnnxOpAttributeNames.AttrTrainingMode, false);
        if (!trainingMode
            && x?.Shape is not null && x.Shape.Dims.Length >= 2 && x.HasDefiniteShape
            && RuntimeTensorFactory.ShouldStoreData(x.Shape, maxDataElements)
            && x.FloatData is { } xd)
        {
            var dims = x.Shape.Dims;
            long channels = dims[1];
            long inner = 1;
            for (int d = 2; d < dims.Length; d++) inner *= dims[d];

            if (scale?.FloatData is { } sd && sd.Length == channels
                && bias?.FloatData is { } bd && bd.Length == channels
                && inputMean?.FloatData is { } md && md.Length == channels
                && inputVar?.FloatData is { } vd && vd.Length == channels)
            {
                var epsilon = AttrAccess.GetFloat(attrs, OnnxOpAttributeNames.AttrEpsilon, 1e-5f);
                var buf = new float[xd.Length];
                for (int i = 0; i < buf.Length; i++)
                {
                    int c = (int)(i / inner % channels);
                    buf[i] = (xd[i] - md[c]) / MathF.Sqrt(vd[c] + epsilon) * sd[c] + bd[c];
                }
                y = y with { FloatData = ImmutableArray.Create(buf) };
            }
        }

        return [
            y,
            RuntimeTensorFactory.Create(statDType, channelShape),
            RuntimeTensorFactory.Create(statDType, channelShape),
        ];
    }
}
