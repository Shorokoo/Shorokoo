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

internal sealed class NotOp : QuickOp
{
    public override string OpCode => OpCodes.NOT;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs[0];
        var rt = RuntimeTensorFactory.Create(DType.Bool, x?.Shape);
        if (x?.BoolData is { } bd && RuntimeTensorFactory.ShouldStoreData(rt.Shape, maxDataElements))
        {
            var d = new bool[bd.Length];
            for (int i = 0; i < d.Length; i++) d[i] = !bd[i];
            return [rt with { BoolData = ImmutableArray.Create(d) }];
        }
        return [rt];
    }
}
