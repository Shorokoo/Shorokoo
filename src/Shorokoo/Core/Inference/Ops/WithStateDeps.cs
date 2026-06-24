using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// Pass-through op for the internal #WithStateDeps# marker. Takes
/// (main_output, ...state_deps) and returns the main output verbatim — the
/// trailing state-dep inputs only carry graph dependencies, not data.
/// </summary>
internal sealed class WithStateDepsOp : QuickOp
{
    public override string OpCode => InternalOpCodes.WITH_STATE_DEPS;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var main = inputs.Length > 0 ? inputs[0] : null;
        if (main is null)
            return [RuntimeTensorFactory.Create(DType.Invalid, null)];
        return [main];
    }
}
