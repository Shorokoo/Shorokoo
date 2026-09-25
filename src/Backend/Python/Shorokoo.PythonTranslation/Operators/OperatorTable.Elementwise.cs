namespace Shorokoo.PythonTranslation.Operators;

// Elementwise math and activations; the semantics are in each support package's ops_elementwise.py.
internal static partial class OperatorTable
{
    static partial void RegisterElementwise(Registry table)
    {
        const string M = "ops_elementwise.";
        foreach (var (op, function) in (ReadOnlySpan<(string, string)>)
        [
            ("Add", "add"), ("Sub", "sub"), ("Mul", "mul"), ("Div", "div"), ("Pow", "pow_"),
            ("Neg", "neg"), ("Abs", "abs_"), ("Sign", "sign"), ("Reciprocal", "reciprocal"),
            ("Floor", "floor"), ("Ceil", "ceil"), ("Round", "round_"), ("Sqrt", "sqrt"),
            ("Exp", "exp"), ("Log", "log"), ("Sin", "sin"), ("Cos", "cos"), ("Tan", "tan"),
            ("Asin", "asin"), ("Acos", "acos"), ("Atan", "atan"), ("Sinh", "sinh"), ("Cosh", "cosh"),
            ("Tanh", "tanh"), ("Asinh", "asinh"), ("Acosh", "acosh"), ("Atanh", "atanh"), ("Erf", "erf"),
            ("Max", "max_"), ("Min", "min_"), ("Sum", "sum_"), ("Mean", "mean"),
            ("Sigmoid", "sigmoid"), ("Relu", "relu"), ("HardSwish", "hard_swish"), ("Softplus", "softplus"),
            ("Softsign", "softsign"), ("Mish", "mish"), ("PRelu", "prelu"),
        ])
            table.Map(op, M + function);

        table.Map("Mod", M + "mod", ["fmod"]);
        table.Map("Clip", M + "clip", ["min", "max"]);
        table.Map("CumSum", M + "cumsum", ["exclusive", "reverse"]);
        table.Map("CumProd", M + "cumprod", ["exclusive", "reverse"]);
        table.Map("LeakyRelu", M + "leaky_relu", ["alpha"]);
        table.Map("Elu", M + "elu", ["alpha"]);
        table.Map("Selu", M + "selu", ["alpha", "gamma"]);
        table.Map("Celu", M + "celu", ["alpha"]);
        table.Map("ThresholdedRelu", M + "thresholded_relu", ["alpha"]);
        table.Map("HardSigmoid", M + "hard_sigmoid", ["alpha", "beta"]);
        table.Map("Gelu", M + "gelu", ["approximate"]);
        table.Map("Swish", M + "swish", ["alpha"]);
        table.Map("Shrink", M + "shrink", ["lambd", "bias"]);
        table.Map("Softmax", M + "softmax", ["axis"], opset: true);
        table.Map("LogSoftmax", M + "log_softmax", ["axis"], opset: true);
        table.Map("Hardmax", M + "hardmax", ["axis"], opset: true, gradient: GradientRule.NotDifferentiable);
        table.Map("Cast", M + "cast", ["to", "saturate", "round_mode"]);
        table.Map("CastLike", M + "cast_like", ["saturate", "round_mode"]);
    }
}
