using System.Globalization;

namespace Shorokoo.PyTorch.Translation.Operators;

// Control flow; the semantics are in shorokoo_torch/ops_control.py. A branch or a loop body is
// written as a Python function nested where the node stands, so it reads the enclosing graph's
// values by closure.
internal static partial class OperatorTable
{
    static partial void RegisterControl(Registry table)
    {
        table.Custom("If", If, returnsTuple: true);
        table.Custom("Loop", Loop, returnsTuple: true);
    }

    private static string If(NodeContext node)
    {
        var thenBranch = node.Subgraph("then_branch");
        var elseBranch = node.Subgraph("else_branch");
        return $"ops_control.if_({node.Input(0)}, {thenBranch}, {elseBranch})";
    }

    private static string Loop(NodeContext node)
    {
        var body = node.Graph("body");
        var carried = node.Node.Inputs.Count - 2;
        var scanDtypes = body.Outputs.Skip(1 + carried)
            .Select(o => (o.Type?.TensorType?.ElemType is > 0 and var type ? type : 1).ToString(CultureInfo.InvariantCulture));
        var values = Enumerable.Range(2, carried).Select(node.Input);
        var bodyFunction = node.Subgraph("body");
        return $"ops_control.loop({node.Input(0)}, {node.Input(1)}, {PyLiteral.List(values)}, "
            + $"{bodyFunction}, {PyLiteral.List(scanDtypes)})";
    }
}
