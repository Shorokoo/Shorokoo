using Shorokoo.Core.Interpreter.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Interpreter.Ops;

internal sealed class ReduceL2Op : ReduceOpBase
{
    public override string OpCode => OpCodes.REDUCE_L2;
    protected override bool CanFoldEmptyGroup => true;
    protected override float Reduce(IEnumerable<float> values) => MathF.Sqrt(values.Select(v => v * v).Sum());

    // The sum of squares wraps at the operand's width, as an integer sum does; its square root is
    // taken in float64 and truncated.
    protected override long ReduceInt(IEnumerable<long> values, DType dtype)
    {
        long s = 0;
        foreach (var v in values) unchecked { s += v * v; }
        return (long)Math.Sqrt(IntSemantics.NarrowToWidth(dtype, s));
    }

    protected override ulong ReduceUInt(IEnumerable<ulong> values, DType dtype)
    {
        ulong s = 0;
        foreach (var v in values) unchecked { s += v * v; }
        return (ulong)Math.Sqrt(IntSemantics.U(IntSemantics.NarrowToWidth(dtype, IntSemantics.S(s))));
    }
}
