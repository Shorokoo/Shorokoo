using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Inference.Ops;

internal sealed class GreaterOp : CompareOp
{
    public override string OpCode => OpCodes.GREATER;
    protected override bool CompareFloat(float a, float b) => a > b;
    protected override bool CompareInt(long a, long b) => a > b;
}
