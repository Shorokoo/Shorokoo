namespace Shorokoo.PyTorch.Translation.Operators;

// Quantization; the semantics are in shorokoo_torch/ops_quant.py. MatMulInteger and ConvInteger
// are registered with the matrix products and the convolutions.
internal static partial class OperatorTable
{
    static partial void RegisterQuant(Registry table)
    {
        const string M = "ops_quant.";
        table.Map("QuantizeLinear", M + "quantize_linear", ["axis", "block_size", "output_dtype", "saturate"],
            gradient: TorchGradient.NotDifferentiable);
        table.Map("DequantizeLinear", M + "dequantize_linear", ["axis", "block_size", "output_dtype"]);
        table.Map("DynamicQuantizeLinear", M + "dynamic_quantize_linear", outputs: true,
            gradient: TorchGradient.NotDifferentiable);
        table.Map("QLinearMatMul", M + "qlinear_matmul", gradient: TorchGradient.NotDifferentiable);
        table.Map("QLinearConv", M + "qlinear_conv", ["auto_pad", "dilations", "group", "kernel_shape", "pads", "strides"],
            gradient: TorchGradient.NotDifferentiable);
    }
}
