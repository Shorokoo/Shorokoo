using System.Globalization;

namespace Shorokoo.PythonTranslation.Operators;

// Sequences and optionals; the semantics are in each support package's ops_sequence.py. A SequenceMap body is
// written as a Python function nested where the node stands, as a Loop body is.
internal static partial class OperatorTable
{
    static partial void RegisterSequence(Registry table)
    {
        const string M = "ops_sequence.";
        table.Map("SequenceConstruct", M + "sequence_construct");
        table.Map("SequenceEmpty", M + "sequence_empty", ["dtype"], gradient: GradientRule.NotDifferentiable);
        table.Map("SequenceInsert", M + "sequence_insert");
        table.Map("SequenceErase", M + "sequence_erase");
        table.Map("SequenceAt", M + "sequence_at");
        table.Map("SequenceLength", M + "sequence_length", gradient: GradientRule.NotDifferentiable);
        table.Map("SplitToSequence", M + "split_to_sequence", ["axis", "keepdims"]);
        table.Map("ConcatFromSequence", M + "concat_from_sequence", ["axis", "new_axis"]);
        table.Custom("SequenceMap", SequenceMap, returnsTuple: true);
        // The type attribute only declares what an Optional without an input holds; the value
        // itself needs nothing of it.
        table.Custom("Optional", node => $"{M}optional({node.Input(0)})");
        table.Map("OptionalHasElement", M + "optional_has_element", gradient: GradientRule.NotDifferentiable);
        table.Map("OptionalGetElement", M + "optional_get_element");
    }

    private static string SequenceMap(NodeContext node)
    {
        var body = node.Subgraph("body");
        var inputs = Enumerable.Range(0, node.InputCount).Select(node.Input);
        return $"ops_sequence.sequence_map({body}, {string.Join(", ", inputs)}, "
            + $"_outputs={node.Graph("body").Outputs.Count.ToString(CultureInfo.InvariantCulture)})";
    }
}
