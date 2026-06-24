using System.Collections.Immutable;
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

internal sealed class SizeOp : QuickOp
{
    public override string OpCode => OpCodes.SIZE;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs[0];
        var rt = RuntimeTensorFactory.Create(DType.Int64, new Shape(Array.Empty<long>()));
        if (x?.Shape is not null && x.Shape.Dims.All(d => d >= 0))
            return [rt with { IntData = ImmutableArray.Create(x.Shape.Count) }];
        return [rt];
    }
}
