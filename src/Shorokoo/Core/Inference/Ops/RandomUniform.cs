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

internal sealed class RandomUniformOp : QuickOp
{
    public override string OpCode => OpCodes.RANDOM_UNIFORM;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var shapeArr = attrs.GetLongsVal(OnnxOpAttributeNames.AttrShape);
        var dtype = attrs.GetDTypeVal(OnnxOpAttributeNames.AttrDtype) ?? DType.Float32;
        Shape? shape = shapeArr is null ? null : new Shape(shapeArr);
        return [RuntimeTensorFactory.Create(dtype, shape)];
    }
}
