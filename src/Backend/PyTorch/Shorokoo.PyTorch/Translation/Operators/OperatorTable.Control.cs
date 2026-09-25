using System.Globalization;
using Shorokoo.Core.Factory.IR;

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
        var scans = body.Outputs.Skip(1 + carried).Select(output => ScanType(node, output)).ToList();
        var values = Enumerable.Range(2, carried).Select(node.Input);
        var bodyFunction = node.Subgraph("body");
        return $"ops_control.loop({node.Input(0)}, {node.Input(1)}, {PyLiteral.List(values)}, "
            + $"{bodyFunction}, {PyLiteral.List(scans)})";
    }

    /// <summary>
    /// The element type and shape a scan output of a loop that runs no iteration is made empty with,
    /// as ONNX Runtime makes it: the body's declared element type, and the shape <c>[0, …]</c> of its
    /// declared dims, one it does not know being 0 — or <c>[0]</c> where it declares no shape.
    /// Refused where the body declares no element type: there is nothing to make it of.
    /// </summary>
    private static string ScanType(NodeContext node, ValueInfoProto output)
    {
        if (output.Type?.TensorType is not { ElemType: > 0 } tensor)
            throw node.Unsupported($"its body's scan output '{output.Name}' declares no tensor element type, "
                + "which a loop that runs no iteration hands it out empty of");
        var dims = tensor.Shape is { } shape
            ? PyLiteral.List(shape.Dims.Select(d => PyLiteral.Int(d.ShouldSerializeDimValue() ? d.DimValue : 0)))
            : "None";
        return $"({tensor.ElemType.ToString(CultureInfo.InvariantCulture)}, {dims})";
    }
}
