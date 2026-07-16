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
/// Shape inference for the internal #ModelParamIdRef# op. Inputs are
/// [modelIndexId, iterationIndices, ...initializerParams] and the output dtype + shape are
/// fully known a priori: dtype comes from the ShrkAttrDtype attribute and shape comes from the
/// first initializer param (inputs[2]) which holds the param's shape vector.
///
/// Without this op QEE would leave MODEL_PARAM_ID_REF outputs shapeless, breaking the
/// downstream Conv/Shape/Gather chains that ExtractModelIdInfosFromStore relies on to capture
/// every model id.
/// </summary>
internal sealed class ModelParamIdRefOp : QuickOp
{
    public override string OpCode => InternalOpCodes.MODEL_PARAM_ID_REF;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var dtype = attrs.GetDTypeVal(OnnxOpAttributeNames.ShrkAttrDtype) ?? DType.Float32;

        Shape? shape = null;
        if (inputs.Length > 2 && inputs[2] is { } shapeVec && shapeVec.IntData is { } shapeDims)
            shape = new Shape(shapeDims.ToArray());

        return [RuntimeTensorFactory.Create(dtype, shape)];
    }
}
