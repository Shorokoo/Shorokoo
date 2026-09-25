namespace Shorokoo.PyTorch.Translation.Operators;

// Comparisons, logic, bitwise operators and Where; the semantics are in shorokoo_torch/ops_logic.py.
internal static partial class OperatorTable
{
    static partial void RegisterLogic(Registry table)
    {
        const string M = "ops_logic.";
        foreach (var (op, function) in (ReadOnlySpan<(string, string)>)
        [
            ("Equal", "equal"), ("Greater", "greater"), ("GreaterOrEqual", "greater_or_equal"),
            ("Less", "less"), ("LessOrEqual", "less_or_equal"), ("And", "and_"), ("Or", "or_"),
            ("Xor", "xor"), ("Not", "not_"), ("IsNaN", "is_nan"), ("BitwiseAnd", "bitwise_and"),
            ("BitwiseOr", "bitwise_or"), ("BitwiseXor", "bitwise_xor"), ("BitwiseNot", "bitwise_not"),
        ])
            table.Map(op, M + function, gradient: TorchGradient.NotDifferentiable);

        table.Map("IsInf", M + "is_inf", ["detect_negative", "detect_positive"], gradient: TorchGradient.NotDifferentiable);
        table.Map("BitShift", M + "bit_shift", ["direction"], gradient: TorchGradient.NotDifferentiable);
        table.Map("Where", M + "where");
    }
}
