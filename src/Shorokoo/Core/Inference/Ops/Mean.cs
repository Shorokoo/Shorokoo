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
/// Variadic elementwise <c>Mean</c> with multidirectional broadcasting over all inputs:
/// the elementwise sum divided by the input count. Mean is float-only per spec.
/// </summary>
internal sealed class MeanOp : QuickOp
{
    public override string OpCode => OpCodes.MEAN;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var shapes = inputs.Where(i => i is not null).Select(i => i!.Shape).ToArray();
        var shape = ShapeHelpers.Broadcast(shapes);
        var dtype = inputs.FirstOrDefault(i => i is not null)?.DType ?? DType.Float32;
        var rt = RuntimeTensorFactory.Create(dtype, shape);

        if (shape is not null && RuntimeTensorFactory.ShouldStoreData(shape, maxDataElements)
            && VariadicElementwise.FoldFloat(inputs, shape, (a, b) => a + b) is { } f)
        {
            var n = inputs.Count(i => i is not null);
            for (int i = 0; i < f.Length; i++) f[i] /= n;
            return [rt with { FloatData = System.Collections.Immutable.ImmutableArray.Create(f) }];
        }
        return [rt];
    }
}
