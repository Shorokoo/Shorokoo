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
/// Variadic elementwise <c>Sum</c> with multidirectional broadcasting over all inputs.
/// Sum is float-only per spec, so only the float data path computes values.
/// </summary>
internal sealed class SumOp : QuickOp
{
    public override string OpCode => OpCodes.SUM;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var shapes = inputs.Where(i => i is not null).Select(i => i!.Shape).ToArray();
        var shape = ShapeHelpers.Broadcast(shapes);
        var dtype = inputs.FirstOrDefault(i => i is not null)?.DType ?? DType.Float32;
        var rt = RuntimeTensorFactory.Create(dtype, shape);

        if (shape is not null && RuntimeTensorFactory.ShouldStoreData(shape, maxDataElements)
            && VariadicElementwise.FoldFloat(inputs, shape, (a, b) => a + b) is { } f)
            return [rt with { FloatData = System.Collections.Immutable.ImmutableArray.Create(f) }];
        return [rt];
    }
}
