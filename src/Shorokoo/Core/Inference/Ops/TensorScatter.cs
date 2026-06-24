using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Inference.Helpers;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// QEE kernel for ONNX <c>TensorScatter</c> (opset 24+). present_cache always has
/// past_cache's shape and dtype regardless of the axis/mode attributes and the
/// write_indices values, so this is an exact shape/dtype pass-through. Value
/// computation is intentionally not implemented (shape-only): the windowed
/// linear/circular write is exercised through the ORT execution path.
/// </summary>
internal sealed class TensorScatterOp : QuickOp
{
    public override string OpCode => OpCodes.TENSOR_SCATTER;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var pastCache = inputs.Length > 0 ? inputs[0] : null;
        var dtype = pastCache?.DType ?? DType.Float32;
        if (pastCache?.Shape is not null)
            return [RuntimeTensorFactory.Create(dtype, pastCache.Shape)];
        return [RuntimeTensorFactory.CreateRankOnly(dtype, pastCache?.Rank)];
    }
}
