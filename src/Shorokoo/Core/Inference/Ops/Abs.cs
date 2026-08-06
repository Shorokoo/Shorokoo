using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Inference.Ops;

internal sealed class AbsOp : UnaryNumericOp
{
    public override string OpCode => OpCodes.ABS;
    protected override float ApplyFloat(float x) => MathF.Abs(x);
    // Math.Abs(long.MinValue) throws; numpy wraps, and a throw here degrades the node to
    // unfolded rather than producing the value.
    protected override long ApplyInt(long x) => x < 0 ? unchecked(-x) : x;
    protected override ulong ApplyUInt(ulong x) => x;
}
