namespace Shorokoo.PythonTranslation.Operators;

// Matrix products and linear algebra; the semantics are in each support package's ops_linalg.py.
internal static partial class OperatorTable
{
    static partial void RegisterLinalg(Registry table)
    {
        const string M = "ops_linalg.";
        table.Map("MatMul", M + "matmul");
        table.Map("Gemm", M + "gemm", ["alpha", "beta", "transA", "transB"]);
        table.Map("MatMulInteger", M + "matmul_integer", gradient: GradientRule.NotDifferentiable);
        table.Map("Einsum", M + "einsum", ["equation"]);
        table.Map("Det", M + "det");
    }
}
