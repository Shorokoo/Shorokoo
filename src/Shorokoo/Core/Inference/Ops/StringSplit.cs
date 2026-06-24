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
/// Shape inference for ONNX <c>StringSplit</c>. Two outputs:
///   y: split tokens with a new trailing dim of length = max token count across elements —
///      data-dependent, so the exact shape is unknown; the rank is input rank + 1 and, when
///      <c>maxsplit</c> is set, the trailing dim is bounded by maxsplit + 1;
///   num_splits: per-element token count (int64), same shape as the input.
/// String VALUES never reach QEE (see TensorDataConverter), so y stays shape/dtype-only.
/// </summary>
internal sealed class StringSplitOp : QuickOp
{
    public override string OpCode => OpCodes.STRING_SPLIT;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs.Length > 0 ? inputs[0] : null;
        var dtype = x?.DType ?? DType.String;
        if (x?.Shape is null)
            return [
                RuntimeTensorFactory.Create(dtype, null),
                RuntimeTensorFactory.Create(DType.Int64, null),
            ];

        var dims = x.Shape.Dims;
        var yRank = dims.Length + 1;
        var y = RuntimeTensorFactory.Create(dtype, null) with { Rank = yRank, MaxRank = yRank };

        // maxsplit bounds the trailing dim: at most maxsplit splits → maxsplit + 1 tokens.
        var maxsplit = attrs.GetLongVal(OnnxOpAttributeNames.AttrMaxsplit);
        if (maxsplit is > 0)
        {
            var maxDims = new long[yRank];
            for (int i = 0; i < dims.Length; i++) maxDims[i] = dims[i];
            maxDims[yRank - 1] = maxsplit.Value + 1;
            y = y with { MaxShape = new Shape(maxDims) };
        }

        return [y, RuntimeTensorFactory.Create(DType.Int64, x.Shape)];
    }
}
