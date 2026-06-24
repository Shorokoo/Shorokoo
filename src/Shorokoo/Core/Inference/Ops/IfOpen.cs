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
/// Open side of an <c>If</c> node pair. Produces no regular outputs (the pair's outputs live on
/// <see cref="IfCloseOp"/>); this implementation is a no-op that returns an empty array.
/// </summary>
internal sealed class IfOpenOp : QuickOp
{
    public override string OpCode => OpCodes.IF_OPEN;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        return Array.Empty<RuntimeTensor>();
    }
}
