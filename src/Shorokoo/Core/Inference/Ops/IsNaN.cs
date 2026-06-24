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

/// <summary>
/// Shape inference for ONNX <c>IsNaN</c>. Output is a bool tensor with the same shape as input.
/// </summary>
internal sealed class IsNaNOp : QuickOp
{
    public override string OpCode => OpCodes.IS_NAN;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs.Length > 0 ? inputs[0] : null;
        var rt = RuntimeTensorFactory.Create(DType.Bool, x?.Shape);

        if (x?.FloatData is { } fd && RuntimeTensorFactory.ShouldStoreData(x.Shape, maxDataElements))
        {
            var buf = new bool[fd.Length];
            for (int i = 0; i < buf.Length; i++) buf[i] = float.IsNaN(fd[i]);
            return [rt with { BoolData = ImmutableArray.Create(buf) }];
        }
        return [rt];
    }
}
