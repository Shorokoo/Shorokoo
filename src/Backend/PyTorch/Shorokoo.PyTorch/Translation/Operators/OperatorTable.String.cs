namespace Shorokoo.PyTorch.Translation.Operators;

// Strings and text; the semantics go in shorokoo_torch/ops_string.py. None is translated yet, so a model
// using one is refused when its session is created, naming the operator.
internal static partial class OperatorTable
{
    static partial void RegisterString(Registry table)
    {
    }
}
