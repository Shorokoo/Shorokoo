using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Inference.Ops;

// Relu accepts integer tensors too since opset 14, so it derives from UnaryNumericOp
// (dtype passthrough + both float and int value paths).
internal sealed class ReluOp : UnaryNumericOp
{
    public override string OpCode => OpCodes.RELU;
    protected override float ApplyFloat(float x) => x < 0 ? 0 : x;
    protected override long ApplyInt(long x) => x < 0 ? 0 : x;
}
