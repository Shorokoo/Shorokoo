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
using System.Collections.Immutable;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// QEE kernel for ONNX <c>EyeLike</c>. Output has the input's (2-D) shape; the dtype is the
/// <c>dtype</c> attribute when set, else the input's. Values depend only on the shape and the
/// <c>k</c> diagonal offset (<c>out[i, j] = 1 iff j == i + k</c>), so they're computed for
/// small tensors regardless of whether the input's data is known. A known input shape whose
/// rank isn't 2 is invalid per spec and degrades to unknown.
/// </summary>
internal sealed class EyeLikeOp : QuickOp
{
    public override string OpCode => OpCodes.EYE_LIKE;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs.Length > 0 ? inputs[0] : null;
        var dtype = attrs.GetDTypeVal(OnnxOpAttributeNames.AttrDtype) ?? x?.DType ?? DType.Float32;
        if (x?.Shape is null)
            return [RuntimeTensorFactory.CreateRankOnly(dtype, 2)];
        if (x.Shape.Dims.Length != 2)
            return [RuntimeTensorFactory.CreateRankOnly(dtype, 2)];

        var rt = RuntimeTensorFactory.Create(dtype, x.Shape);
        var dims = x.Shape.Dims;
        if (dims[0] < 0 || dims[1] < 0 || !RuntimeTensorFactory.ShouldStoreData(x.Shape, maxDataElements))
            return [rt];

        var k = attrs.GetLongVal(OnnxOpAttributeNames.AttrK) ?? 0;
        var rows = (int)dims[0];
        var cols = (int)dims[1];
        if (DTypeHelpers.IsFloat(dtype))
        {
            var buf = new float[rows * cols];
            for (int i = 0; i < rows; i++)
            {
                var j = i + k;
                if (j >= 0 && j < cols) buf[i * cols + (int)j] = 1f;
            }
            return [rt with { FloatData = ImmutableArray.Create(buf) }];
        }
        if (DTypeHelpers.IsInt(dtype))
        {
            var buf = new long[rows * cols];
            for (int i = 0; i < rows; i++)
            {
                var j = i + k;
                if (j >= 0 && j < cols) buf[i * cols + (int)j] = 1L;
            }
            return [rt with { IntData = ImmutableArray.Create(buf) }];
        }
        if (DTypeHelpers.IsBool(dtype))
        {
            var buf = new bool[rows * cols];
            for (int i = 0; i < rows; i++)
            {
                var j = i + k;
                if (j >= 0 && j < cols) buf[i * cols + (int)j] = true;
            }
            return [rt with { BoolData = ImmutableArray.Create(buf) }];
        }
        return [rt];
    }
}
