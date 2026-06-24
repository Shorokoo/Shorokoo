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
/// Shape inference for ONNX <c>Multinomial</c>. Input <c>[batch, class_count]</c> samples
/// <c>sample_size</c> draws (default 1) per batch row, producing
/// <c>[batch, sample_size]</c>. dtype defaults to int32 per spec. Values are random — never
/// computed.
/// </summary>
internal sealed class MultinomialOp : QuickOp
{
    public override string OpCode => OpCodes.MULTINOMIAL;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs.Length > 0 ? inputs[0] : null;
        var dtype = attrs.GetDTypeVal(OnnxOpAttributeNames.AttrDtype) ?? DType.Int32;
        var sampleSize = attrs.GetLongVal(OnnxOpAttributeNames.AttrSampleSize) ?? 1;

        // Input must be 2-D per spec; degrade to a rank-2 unknown otherwise.
        if (x?.Shape?.Dims is not { Length: 2 } xDims || xDims[0] < 0 || sampleSize < 0)
            return [RuntimeTensorFactory.CreateRankOnly(dtype, 2)];

        return [RuntimeTensorFactory.Create(dtype, new Shape(new[] { xDims[0], sampleSize }))];
    }
}
