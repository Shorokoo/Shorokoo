using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// QEE kernel for ONNX <c>QLinearMatMul</c>. Shares the numpy-style shape semantics of
/// <see cref="MatMulOp"/> (incl. the 1-D edge cases). Output dtype matches the
/// <c>y_zero_point</c> input (input[7]) when available, else falls back to input
/// <c>a</c>'s dtype. Values are not computed (quantized arithmetic; shape/dtype only).
/// </summary>
internal sealed class QLinearMatMulOp : QuickOp
{
    public override string OpCode => OpCodes.QLINEAR_MATMUL;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        // Inputs: [a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point]
        var a = inputs.Length > 0 ? inputs[0] : null;
        var b = inputs.Length > 3 ? inputs[3] : null;
        var yZeroPoint = inputs.Length > 7 ? inputs[7] : null;
        var dtype = yZeroPoint?.DType ?? a?.DType ?? DType.UInt8;
        var shape = MatMulHelpers.InferShape(a?.Shape, b?.Shape);
        return [RuntimeTensorFactory.Create(dtype, shape)];
    }
}
