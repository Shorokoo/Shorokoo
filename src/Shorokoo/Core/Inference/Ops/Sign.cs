using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Inference.Ops;

internal sealed class SignOp : UnaryNumericOp
{
    public override string OpCode => OpCodes.SIGN;
    protected override float ApplyFloat(float x) => x > 0 ? 1f : x < 0 ? -1f : 0f;
    protected override long ApplyInt(long x) => x > 0 ? 1L : x < 0 ? -1L : 0L;
}
