namespace Shorokoo.PyTorch.Translation.Operators;

// Reductions; the semantics are in shorokoo_torch/ops_reduction.py.
internal static partial class OperatorTable
{
    static partial void RegisterReduction(Registry table)
    {
        const string M = "ops_reduction.";
        string[] reduceAttributes = ["axes", "keepdims", "noop_with_empty_axes"];
        foreach (var (op, function) in (ReadOnlySpan<(string, string)>)
        [
            ("ReduceSum", "reduce_sum"), ("ReduceMean", "reduce_mean"), ("ReduceMax", "reduce_max"),
            ("ReduceMin", "reduce_min"), ("ReduceProd", "reduce_prod"), ("ReduceL1", "reduce_l1"),
            ("ReduceL2", "reduce_l2"), ("ReduceSumSquare", "reduce_sum_square"),
            ("ReduceLogSum", "reduce_log_sum"), ("ReduceLogSumExp", "reduce_log_sum_exp"),
        ])
            table.Map(op, M + function, reduceAttributes);

        string[] argAttributes = ["axis", "keepdims", "select_last_index"];
        table.Map("ArgMax", M + "arg_max", argAttributes, gradient: TorchGradient.NotDifferentiable);
        table.Map("ArgMin", M + "arg_min", argAttributes, gradient: TorchGradient.NotDifferentiable);
    }
}
