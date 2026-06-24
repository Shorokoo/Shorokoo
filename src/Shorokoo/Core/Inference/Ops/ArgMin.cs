using Shorokoo.Core.Nodes.NodeDefinitions;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>ONNX <c>ArgMin</c> — see <see cref="ArgExtremeOpBase"/>.</summary>
internal sealed class ArgMinOp : ArgExtremeOpBase
{
    public override string OpCode => OpCodes.ARG_MIN;
    protected override bool Beats(float candidate, float best) => candidate < best;
    protected override bool BeatsInt(long candidate, long best) => candidate < best;
}
