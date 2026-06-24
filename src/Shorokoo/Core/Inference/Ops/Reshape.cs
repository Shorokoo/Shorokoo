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

internal sealed class ReshapeOp : QuickOp
{
    public override string OpCode => OpCodes.RESHAPE;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs[0];
        var shapeIn = inputs.Length > 1 ? inputs[1] : null;
        var dtype = x?.DType ?? DType.Float32;

        if (shapeIn?.IntData is not { } shapeData)
            return [RuntimeTensorFactory.Create(dtype, null)];

        var newDims = shapeData.ToArray();
        var allowZero = AttrAccess.GetBool(attrs, OnnxOpAttributeNames.AttrAllowzero, false);

        if (x?.Shape is null)
        {
            // Shape values are known but the input shape isn't. The output shape is still
            // fully determined when no dim needs input information (-1 always does; 0 does
            // unless allowzero). Otherwise degrade to unknown — never guess.
            if (newDims.Any(d => d < 0) || (!allowZero && newDims.Any(d => d == 0)))
                return [RuntimeTensorFactory.Create(dtype, null)];
            return [RuntimeTensorFactory.Create(dtype, new Shape(newDims))];
        }

        // Handle 0-dims (copy from input unless allowzero).
        if (!allowZero)
        {
            for (int i = 0; i < newDims.Length && i < x.Shape.Dims.Length; i++)
                if (newDims[i] == 0) newDims[i] = x.Shape.Dims[i];
        }
        // Handle -1.
        var negIdx = Array.IndexOf(newDims, -1L);
        if (negIdx >= 0)
        {
            long known = 1;
            foreach (var d in newDims) if (d >= 0) known *= d;
            if (known <= 0 || x.Shape.Count % known != 0)
                return [RuntimeTensorFactory.Create(dtype, null)]; // invalid combination — don't guess
            newDims[negIdx] = x.Shape.Count / known;
        }
        if (newDims.Any(d => d < 0))
            return [RuntimeTensorFactory.Create(dtype, null)];
        var outShape = new Shape(newDims);
        var rt = RuntimeTensorFactory.Create(dtype, outShape);

        var retainData = RuntimeTensorFactory.ShouldStoreData(outShape, maxDataElements);
        if (retainData && x.FloatData is { } fd && fd.Length == outShape.Count)
            return [rt with { FloatData = fd }];
        if (retainData && x.IntData is { } id && id.Length == outShape.Count)
            return [rt with { IntData = id }];
        if (retainData && x.BoolData is { } bd && bd.Length == outShape.Count)
            return [rt with { BoolData = bd }];
        return [rt];
    }
}
