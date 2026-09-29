using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.NodeDefinitions;

namespace Shorokoo.Core.Lowering.KernelWorkarounds.OnnxRuntime;

using static OnnxOp;
using static OpCodes;

/// <summary>
/// An <c>Add</c>, <c>Sub</c>, <c>Mul</c> or <c>Div</c> with an empty <c>Constant</c> operand,
/// whose result is the empty tensor of the operands' broadcast shape (Shorokoo/Shorokoo#454).
///
/// <para>ONNX Runtime's NoopElimination takes a constant operand whose every element is the
/// identity of the operation, 0 for either operand of <c>Add</c> and the right one of <c>Sub</c>,
/// 1 for either operand of <c>Mul</c> and the right one of <c>Div</c>, as leaving the other operand
/// unchanged, and removes the call; an empty constant has no element that is not, so the result
/// is the other operand. The empty operand is rebuilt from the other one, as its first no
/// elements viewed with the constant's shape, a value ONNX Runtime does not know when it builds the
/// session, so the call keeps its broadcast. Only a call with such an operand and another that is
/// not a <c>Constant</c> is rewritten; a left operand of <c>Sub</c> or <c>Div</c> is computed as the
/// spec says. An empty operand ONNX Runtime computes from constants when it builds the session is
/// not a <c>Constant</c> of the model, and its call is left as it stands: that result is accepted
/// as ONNX Runtime's.</para>
/// </summary>
internal sealed class ArithmeticEmptyConstantWorkaround : KernelWorkaround
{
    public override IReadOnlySet<string> OpCodes { get; } = new HashSet<string>([ADD, SUB, MUL, DIV], StringComparer.Ordinal);

    public override bool Applies(WorkaroundSite site)
        => IsEmptyConstant(site, 1) ? site.ConstantShapeOf(0) is null
            : site.OpCode is ADD or MUL && IsEmptyConstant(site, 0) && site.ConstantShapeOf(1) is null;

    private static bool IsEmptyConstant(WorkaroundSite site, int slot) => site.ConstantShapeOf(slot) is { } shape && shape.Dims.Contains(0);

    public override Variable?[] Rewrite(WorkaroundSite site, Variable?[] inputs)
    {
        var (a, b) = (inputs[0]!, inputs[1]!);
        Variable Rebuilt(Variable empty, Variable other)
            => Reshape(Slice(Reshape(other, Globals.Vector(-1L), allowZero: false), Globals.Vector(0L), Globals.Vector(0L)), Shape(empty), allowZero: true);
        if (IsEmptyConstant(site, 1)) b = Rebuilt(b, a);
        else a = Rebuilt(a, b);
        return [NodeBuilder.BuildNodeSingleOut(site.OpCode, [a, b], [])];
    }
}
