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
/// ONNX <c>CastLike</c>: casts input to the dtype of target_type; shape follows input.
/// Value conversion shares <see cref="CastOp.WithConvertedData"/>. The <c>saturate</c>
/// attribute only affects float8 targets, which Shorokoo does not support (N/A).
/// </summary>
internal sealed class CastLikeOp : QuickOp
{
    public override string OpCode => OpCodes.CAST_LIKE;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs[0];
        var target = inputs.Length > 1 ? inputs[1] : null;
        var dtype = target?.DType ?? x?.DType ?? DType.Float32;
        var rt = new RuntimeTensor
        {
            DType = dtype,
            Shape = x?.Shape,
            MaxShape = x?.MaxShape ?? x?.Shape,
            Rank = x?.Rank,
            MaxRank = x?.MaxRank ?? x?.Rank,
        };

        if (x is null || !RuntimeTensorFactory.ShouldStoreData(rt.Shape, maxDataElements))
            return [rt];

        return [CastOp.WithConvertedData(rt, x, dtype)];
    }
}
