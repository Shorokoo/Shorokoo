namespace Shorokoo.PythonTranslation.Operators;

// Quantization; the semantics are in each support package's ops_quant.py. MatMulInteger and ConvInteger
// are registered with the matrix products and the convolutions.
internal static partial class OperatorTable
{
    static partial void RegisterQuant(Registry table)
    {
        const string M = "ops_quant.";
        table.Map("QuantizeLinear", M + "quantize_linear", ["axis", "block_size", "output_dtype", "saturate"],
            gradient: GradientRule.NotDifferentiable);
        table.Map("DequantizeLinear", M + "dequantize_linear", ["axis", "block_size", "output_dtype"]);
        table.Map("DynamicQuantizeLinear", M + "dynamic_quantize_linear", outputs: true,
            gradient: GradientRule.NotDifferentiable);
        table.Map("QLinearMatMul", M + "qlinear_matmul", gradient: GradientRule.NotDifferentiable);
        table.Map("QLinearConv", M + "qlinear_conv", ["auto_pad", "dilations", "group", "kernel_shape", "pads", "strides"],
            gradient: GradientRule.NotDifferentiable);
    }
}
