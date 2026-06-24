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
/// Shape inference for ONNX <c>Det</c>. Input <c>[..., M, M]</c> reduces to <c>[...]</c>;
/// the last two square-matrix dimensions are removed.
/// </summary>
internal sealed class DetOp : QuickOp
{
    public override string OpCode => OpCodes.DET;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs.Length > 0 ? inputs[0] : null;
        var dtype = x?.DType ?? DType.Float32;
        if (x?.Shape is null) return [RuntimeTensorFactory.Create(dtype, null)];

        var dims = x.Shape.Dims;
        if (dims.Length < 2) return [RuntimeTensorFactory.Create(dtype, null)];
        // The last two dims must be a square matrix; a known mismatch means the model is
        // invalid — degrade to unknown rather than claiming a shape.
        if (dims[^1] >= 0 && dims[^2] >= 0 && dims[^1] != dims[^2])
            return [RuntimeTensorFactory.Create(dtype, null)];

        var outDims = new long[dims.Length - 2];
        for (int i = 0; i < outDims.Length; i++) outDims[i] = dims[i];
        return [RuntimeTensorFactory.Create(dtype, new Shape(outDims))];
    }
}
