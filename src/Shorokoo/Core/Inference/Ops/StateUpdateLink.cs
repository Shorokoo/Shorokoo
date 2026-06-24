using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// Pass-through op for the internal #StateUpdateLink# marker. Takes
/// (original_state, updated_state) and returns the updated_state tensor — the
/// "link" is a graph-level annotation, not a computation, so the output value
/// equals input[1] entirely (same shape, dtype, and data).
/// </summary>
internal sealed class StateUpdateLinkOp : QuickOp
{
    public override string OpCode => InternalOpCodes.STATE_UPDATE_LINK;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var updated = inputs.Length > 1 ? inputs[1] : null;
        if (updated is null)
            return [RuntimeTensorFactory.Create(DType.Invalid, null)];
        return [updated];
    }
}
