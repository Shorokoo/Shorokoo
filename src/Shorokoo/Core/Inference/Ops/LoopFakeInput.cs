using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Onnx;
using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// Placeholder loop input emitted by Shorokoo's Looper when a loop variable's initial value
/// cannot be determined at graph construction time. Produces a tensor with only its declared
/// dtype and rank known.
/// </summary>
internal sealed class LoopFakeInputOp : QuickOp
{
    public override string OpCode => OpCodes.LOOP_FAKE_INPUT;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var dtype = attrs.GetDTypeVal(OnnxOpAttributeNames.AttrDtype) ?? DType.Float32;
        var rank = attrs.GetLongVal(OnnxOpAttributeNames.InternalAttrRank);

        var rt = RuntimeTensorFactory.Create(dtype, null);
        if (rank.HasValue)
            rt = rt with { Rank = (int)rank.Value, MaxRank = (int)rank.Value };
        return [rt];
    }
}
