namespace Shorokoo.PyTorch.Translation.Operators;

// Indexing into a tensor; the semantics are in shorokoo_torch/ops_indexing.py.
internal static partial class OperatorTable
{
    static partial void RegisterIndexing(Registry table)
    {
        const string M = "ops_indexing.";
        table.Map("Gather", M + "gather", ["axis"]);
        table.Map("GatherElements", M + "gather_elements", ["axis"]);
        table.Map("Slice", M + "slice_", ["starts", "ends", "axes"]);
    }
}
