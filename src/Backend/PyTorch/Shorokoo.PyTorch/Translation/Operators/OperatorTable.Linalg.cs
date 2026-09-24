namespace Shorokoo.PyTorch.Translation.Operators;

// Matrix products; the semantics are in shorokoo_torch/ops_linalg.py.
internal static partial class OperatorTable
{
    static partial void RegisterLinalg(Registry table)
    {
        const string M = "ops_linalg.";
        table.Map("MatMul", M + "matmul");
        table.Map("Gemm", M + "gemm", ["alpha", "beta", "transA", "transB"]);
    }
}
