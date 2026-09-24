namespace Shorokoo.PyTorch.Translation.Operators;

// Sequences and optionals; the semantics go in shorokoo_torch/ops_sequence.py. None is translated yet, so a model
// using one is refused when its session is created, naming the operator.
internal static partial class OperatorTable
{
    static partial void RegisterSequence(Registry table)
    {
    }
}
