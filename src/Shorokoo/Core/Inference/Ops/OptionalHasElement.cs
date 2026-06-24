using System.Collections.Immutable;
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
/// ONNX <c>OptionalHasElement</c>: produces a scalar bool indicating whether the optional
/// holds a value. When the input is a <see cref="RuntimeOptionalTensor"/> with a known
/// <see cref="RuntimeOptionalTensor.HasValue"/>, the result's <c>BoolData</c> carries it;
/// otherwise only the scalar-bool shape/dtype is known.
/// </summary>
internal sealed class OptionalHasElementOp : QuickOp
{
    public override string OpCode => OpCodes.OPTIONAL_HAS_ELEMENT;

    protected override IRuntimeTensor[] Compute(IRuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var src = inputs.Length > 0 ? inputs[0] : null;
        var rt = RuntimeTensorFactory.Create(DType.Bool, new Shape(Array.Empty<long>()));
        // Opset 18+: a plain tensor/sequence input is trivially present, and an absent
        // (optional, opset-18) input is trivially absent.
        var known = src switch
        {
            RuntimeOptionalTensor opt => opt.HasValue,
            RuntimeTensor or RuntimeSequenceTensor => true,
            null => false,
            _ => (bool?)null,
        };
        if (known is bool has)
            rt = rt with { BoolData = ImmutableArray.Create(has) };
        return [rt];
    }
}
