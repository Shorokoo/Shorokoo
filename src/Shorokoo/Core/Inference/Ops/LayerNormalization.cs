using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// Shape inference for ONNX <c>LayerNormalization</c>. Three outputs:
///   y: same shape and dtype as input;
///   mean / inv_std_dev: same rank as input but with the normalization dims
///   (<c>axis</c> through the end, axis default −1) set to 1, typed per
///   <c>stash_type</c> (default 1 → float32 — NOT the input dtype, which matters for
///   fp16/bf16 inputs).
/// An out-of-range axis degrades the stat shapes to unknown rather than guessing.
/// </summary>
internal sealed class LayerNormalizationOp : QuickOp
{
    public override string OpCode => OpCodes.LAYER_NORMALIZATION;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs.Length > 0 ? inputs[0] : null;
        var dtype = x?.DType ?? DType.Float32;

        // stash_type picks the dtype of the mean / inv_std_dev outputs (a TensorProto
        // dtype number; spec constrains it to a float type — anything else keeps the
        // float32 default).
        var stashType = AttrAccess.GetLong(attrs, OnnxOpAttributeNames.AttrStashType, 1);
        var statDType = DType.Float32;
        try
        {
            var stash = DType.FromProtoTypeNum((int)stashType);
            if (DTypeHelpers.IsFloat(stash)) statDType = stash;
        }
        catch (ArgumentException) { /* unknown proto num — keep float32 */ }

        Shape? statShape = null;
        if (x?.Shape is not null)
        {
            var axis = (int)AttrAccess.GetLong(attrs, OnnxOpAttributeNames.AttrAxis, -1);
            var dims = x.Shape.Dims.ToArray();
            if (axis < 0) axis += dims.Length;
            if (axis >= 0 && axis < dims.Length)
            {
                for (int i = axis; i < dims.Length; i++) dims[i] = 1;
                statShape = new Shape(dims);
            }
        }

        return [
            RuntimeTensorFactory.Create(dtype, x?.Shape),
            RuntimeTensorFactory.Create(statDType, statShape),
            RuntimeTensorFactory.Create(statDType, statShape),
        ];
    }
}
