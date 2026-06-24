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
/// Shape inference for the deprecated ONNX <c>Upsample</c> op. Output dim along each
/// axis is <c>floor(input_dim * scale)</c>. <c>Upsample</c> was superseded by
/// <c>Resize</c>, but older models still rely on it.
/// </summary>
internal sealed class UpsampleOp : QuickOp
{
    public override string OpCode => OpCodes.UPSAMPLE;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs.Length > 0 ? inputs[0] : null;
        var scales = inputs.Length > 1 ? inputs[1] : null;
        var dtype = x?.DType ?? DType.Float32;
        if (x?.Shape is null) return [RuntimeTensorFactory.Create(dtype, null)];

        var xDims = x.Shape.Dims;
        if (scales?.FloatData is { } sd && sd.Length == xDims.Length)
        {
            var outDims = new long[xDims.Length];
            for (int i = 0; i < xDims.Length; i++)
                outDims[i] = (long)System.Math.Floor(xDims[i] * sd[i]);
            return [RuntimeTensorFactory.Create(dtype, new Shape(outDims))];
        }
        return [RuntimeTensorFactory.Create(dtype, null)];
    }
}
