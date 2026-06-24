using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Onnx;
using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// ONNX <c>OptionalGetElement</c>: unwraps an optional to its inner tensor. When the optional
/// carries a known <see cref="RuntimeOptionalTensor.ValueTensor"/> we return it; since opset
/// 18 the input may also be a plain tensor or a sequence, which passes through unchanged.
/// When presence is unknown (or known-false — invalid per spec) only the element dtype is
/// surfaced.
/// </summary>
internal sealed class OptionalGetElementOp : QuickOp
{
    public override string OpCode => OpCodes.OPTIONAL_GET_ELEMENT;

    protected override IRuntimeTensor[] Compute(IRuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var src = inputs.Length > 0 ? inputs[0] : null;

        if (src is RuntimeOptionalTensor opt && opt.ValueTensor is not null)
            return [opt.ValueTensor];

        // Opset 18+: a non-optional input is returned as-is (the op is then an identity).
        if (src is RuntimeTensor plain) return [plain];
        if (src is RuntimeSequenceTensor seq) return [seq];

        // Optional with known-false HasValue or entirely unknown: return a placeholder tensor
        // with just the dtype carried over from the optional's declared element type.
        var dtype = src?.DType ?? DType.Invalid;
        return [RuntimeTensorFactory.Create(dtype, null)];
    }
}
