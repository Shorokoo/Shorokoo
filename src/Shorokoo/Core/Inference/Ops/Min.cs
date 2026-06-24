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
/// Variadic elementwise <c>Min</c> with multidirectional broadcasting over all inputs.
/// Computes concrete float/int values when every input carries data.
/// </summary>
internal sealed class MinOp : QuickOp
{
    public override string OpCode => OpCodes.MIN;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var shapes = inputs.Where(i => i is not null).Select(i => i!.Shape).ToArray();
        var shape = ShapeHelpers.Broadcast(shapes);
        var dtype = inputs.FirstOrDefault(i => i is not null)?.DType ?? DType.Float32;
        var rt = RuntimeTensorFactory.Create(dtype, shape);

        if (shape is not null && RuntimeTensorFactory.ShouldStoreData(shape, maxDataElements))
        {
            if (VariadicElementwise.FoldFloat(inputs, shape, MathF.Min) is { } f)
                return [rt with { FloatData = System.Collections.Immutable.ImmutableArray.Create(f) }];
            if (VariadicElementwise.FoldInt(inputs, shape, Math.Min) is { } l)
                return [rt with { IntData = System.Collections.Immutable.ImmutableArray.Create(l) }];
        }
        return [rt];
    }
}
