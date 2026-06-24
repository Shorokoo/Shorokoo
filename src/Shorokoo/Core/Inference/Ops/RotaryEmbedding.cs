using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Inference.Helpers;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// QEE kernel for ONNX <c>RotaryEmbedding</c> (opset 23+). Y always has X's shape
/// and dtype (the rotation is a shape-preserving pairwise mix of the head
/// dimension), so this is an exact shape/dtype pass-through. Value computation is
/// intentionally not implemented (shape-only): the rotation depends on cache
/// gathering via position_ids, interleaved layout, and partial-rotation slicing —
/// real values come from the ORT execution path.
/// </summary>
internal sealed class RotaryEmbeddingOp : QuickOp
{
    public override string OpCode => OpCodes.ROTARY_EMBEDDING;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs.Length > 0 ? inputs[0] : null;
        var dtype = x?.DType ?? DType.Float32;
        if (x?.Shape is not null)
            return [RuntimeTensorFactory.Create(dtype, x.Shape)];
        return [RuntimeTensorFactory.CreateRankOnly(dtype, x?.Rank)];
    }
}
