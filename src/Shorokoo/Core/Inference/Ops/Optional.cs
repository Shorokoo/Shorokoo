using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Onnx;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// ONNX <c>Optional</c>: wraps its optional tensor input into a <see cref="RuntimeOptionalTensor"/>.
/// If the input is provided, <see cref="RuntimeOptionalTensor.HasValue"/> is true; if it is
/// absent (null in the graph), HasValue is false.
/// </summary>
internal sealed class OptionalOp : QuickOp
{
    public override string OpCode => OpCodes.OPTIONAL;

    protected override IRuntimeTensor[] Compute(IRuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs.Length > 0 ? inputs[0] : null;

        // An empty optional's element dtype comes from the `type` attribute (stored by
        // OnnxOp.Optional as a (DataStructure, DType) tuple).
        var attrDType = attrs.GetTypeProtoVal(OnnxOpAttributeNames.AttrType)?.dtype ?? DType.Invalid;

        RuntimeOptionalTensor opt = x switch
        {
            RuntimeTensor tensor => new RuntimeOptionalTensor { HasValue = true, ValueTensor = tensor, DType = tensor.DType },
            null => new RuntimeOptionalTensor { HasValue = false, DType = attrDType },
            _ => new RuntimeOptionalTensor { HasValue = null, DType = x.DType },
        };
        return [opt];
    }
}
