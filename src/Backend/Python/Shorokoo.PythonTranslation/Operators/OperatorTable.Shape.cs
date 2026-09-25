namespace Shorokoo.PythonTranslation.Operators;

// Shape and data movement without indexing, and Constant; the semantics are in
// shorokoo_torch/ops_shape.py.
internal static partial class OperatorTable
{
    static partial void RegisterShape(Registry table)
    {
        const string M = "ops_shape.";
        table.Map("Reshape", M + "reshape", ["allowzero"]);
        table.Map("Transpose", M + "transpose", ["perm"]);
        table.Map("Concat", M + "concat", ["axis"]);
        table.Map("Split", M + "split", ["axis", "num_outputs", "split"], outputs: true);
        table.Map("Squeeze", M + "squeeze", ["axes"]);
        table.Map("Unsqueeze", M + "unsqueeze", ["axes"]);
        table.Map("Flatten", M + "flatten", ["axis"]);
        table.Map("Expand", M + "expand");
        table.Map("Tile", M + "tile");
        table.Map("Identity", M + "identity");
        table.Map("Trilu", M + "trilu", ["upper"]);
        table.Map("Pad", M + "pad", ["mode", "pads", "value"]);
        table.Map("Shape", M + "shape", ["start", "end"], gradient: GradientRule.NotDifferentiable);
        table.Map("Size", M + "size", gradient: GradientRule.NotDifferentiable);
        table.Map("ConstantOfShape", M + "constant_of_shape", ["value"], gradient: GradientRule.NotDifferentiable);
        table.Map("Range", M + "range_", gradient: GradientRule.NotDifferentiable);
        table.Custom("Constant", Constant, gradient: GradientRule.NotDifferentiable);
        table.Map("OneHot", M + "one_hot", ["axis"], gradient: GradientRule.NotDifferentiable);
        table.Map("EyeLike", M + "eye_like", ["dtype", "k"], gradient: GradientRule.NotDifferentiable);
        table.Map("ReverseSequence", M + "reverse_sequence", ["batch_axis", "time_axis"]);
        table.Map("TensorScatter", M + "tensor_scatter", ["mode", "axis"]);
    }

    /// <summary>A Constant node reads its value from whichever of its attributes it carries.</summary>
    private static string Constant(NodeContext node)
    {
        if (node.Node.Attributes.Count != 1)
            throw node.Unsupported("it must carry exactly one value attribute");
        var attribute = node.Node.Attributes[0];
        return attribute.Name switch
        {
            "value" => node.Constant(PythonConstant.FromTensor(node.Dialect, attribute.T, node.Node.OpType)),
            "value_float" => node.Constant(PythonConstant.Scalar(attribute.F)),
            "value_floats" => node.Constant(PythonConstant.Vector(attribute.Floats ?? [])),
            "value_int" => node.Constant(PythonConstant.Scalar(attribute.I)),
            "value_ints" => node.Constant(PythonConstant.Vector(attribute.Ints ?? [])),
            "value_string" => node.Constant(PythonConstant.StringsOf(node.Dialect, [], [attribute.S ?? []], node.AttributeDescription(attribute), node.Node.OpType)),
            "value_strings" => node.Constant(PythonConstant.StringsOf(node.Dialect, [attribute.Strings.Count], attribute.Strings, node.AttributeDescription(attribute), node.Node.OpType)),
            var other => throw node.Unsupported($"its value attribute '{other}' is not one the translation reads"),
        };
    }
}
