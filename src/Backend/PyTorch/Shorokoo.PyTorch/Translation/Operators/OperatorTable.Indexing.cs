namespace Shorokoo.PyTorch.Translation.Operators;

// Indexing into a tensor; the semantics are in shorokoo_torch/ops_indexing.py.
internal static partial class OperatorTable
{
    static partial void RegisterIndexing(Registry table)
    {
        const string M = "ops_indexing.";
        table.Map("Gather", M + "gather", ["axis"]);
        table.Map("GatherElements", M + "gather_elements", ["axis"]);
        table.Map("GatherND", M + "gather_nd", ["batch_dims"]);
        table.Map("Slice", M + "slice_", ["starts", "ends", "axes"]);
        table.Map("Compress", M + "compress", ["axis"]);
        table.Map("ScatterElements", M + "scatter_elements", ["axis", "reduction"]);
        table.Map("ScatterND", M + "scatter_nd", ["reduction"]);
        table.Map("TopK", M + "top_k", ["axis", "largest", "sorted"], tuple: true);
        table.Map("Unique", M + "unique", ["axis", "sorted"], tuple: true);
        table.Map("NonZero", M + "non_zero", gradient: TorchGradient.NotDifferentiable);
    }
}
