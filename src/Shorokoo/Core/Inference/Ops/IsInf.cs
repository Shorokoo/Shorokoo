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
/// Shape inference for ONNX <c>IsInf</c>. Output is a bool tensor with the same shape as input.
/// </summary>
internal sealed class IsInfOp : QuickOp
{
    public override string OpCode => OpCodes.IS_INF;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs.Length > 0 ? inputs[0] : null;
        var rt = RuntimeTensorFactory.Create(DType.Bool, x?.Shape);

        if (x?.FloatData is { } fd && RuntimeTensorFactory.ShouldStoreData(x.Shape, maxDataElements))
        {
            // detect_positive and detect_negative both default to 1 per spec.
            var detectPos = AttrAccess.GetBool(attrs, OnnxOpAttributeNames.AttrDetectPositive, true);
            var detectNeg = AttrAccess.GetBool(attrs, OnnxOpAttributeNames.AttrDetectNegative, true);

            var buf = new bool[fd.Length];
            for (int i = 0; i < buf.Length; i++)
            {
                var v = fd[i];
                buf[i] = float.IsInfinity(v)
                    && ((v > 0 && detectPos) || (v < 0 && detectNeg));
            }
            return [rt with { BoolData = ImmutableArray.Create(buf) }];
        }
        return [rt];
    }
}
