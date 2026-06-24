using System.Collections.Immutable;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// Shared QEE implementation for the softmax family (Softmax / LogSoftmax / Hardmax,
/// opset 13+ single-axis semantics): output shape and dtype follow the input, the
/// <c>axis</c> attribute defaults to −1, and concrete float values are computed by
/// transforming each 1-D slice along the (normalized) axis. An out-of-range axis keeps
/// the (always-correct) passthrough shape but blocks value computation.
/// </summary>
internal abstract class SoftmaxFamilyOpBase : QuickOp
{
    /// <summary>Transforms one gathered slice along the softmax axis in place.</summary>
    protected abstract void TransformSlice(float[] slice);

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs.Length > 0 ? inputs[0] : null;
        var dtype = x?.DType ?? DType.Float32;
        var rt = new RuntimeTensor
        {
            DType = dtype,
            Shape = x?.Shape,
            MaxShape = x?.MaxShape ?? x?.Shape,
            Rank = x?.Rank,
            MaxRank = x?.MaxRank ?? x?.Rank,
        };

        if (x?.Shape is null || x.FloatData is not { } fd
            || !RuntimeTensorFactory.ShouldStoreData(x.Shape, maxDataElements))
            return [rt];

        var dims = x.Shape.Dims;
        var rank = dims.Length;
        var axis = (int)AttrAccess.GetLong(attrs, OnnxOpAttributeNames.AttrAxis, -1);
        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank) return [rt];

        long axisDim = dims[axis];
        if (axisDim <= 0) return [rt with { FloatData = fd }];
        long innerCount = 1;
        for (int d = axis + 1; d < rank; d++) innerCount *= dims[d];
        long outerCount = 1;
        for (int d = 0; d < axis; d++) outerCount *= dims[d];

        var buf = new float[fd.Length];
        var slice = new float[axisDim];
        for (long outer = 0; outer < outerCount; outer++)
        {
            long outerOff = outer * axisDim * innerCount;
            for (long inner = 0; inner < innerCount; inner++)
            {
                for (long k = 0; k < axisDim; k++)
                    slice[k] = fd[(int)(outerOff + k * innerCount + inner)];
                TransformSlice(slice);
                for (long k = 0; k < axisDim; k++)
                    buf[(int)(outerOff + k * innerCount + inner)] = slice[k];
            }
        }
        return [rt with { FloatData = ImmutableArray.Create(buf) }];
    }
}

/// <summary>QEE kernel for ONNX <c>Softmax</c>: exp(x − max) / Σ exp(x − max) along the axis.</summary>
internal sealed class SoftmaxOp : SoftmaxFamilyOpBase
{
    public override string OpCode => OpCodes.SOFTMAX;

    protected override void TransformSlice(float[] slice)
    {
        float max = float.NegativeInfinity;
        foreach (var v in slice) if (v > max) max = v;
        float sum = 0;
        for (int i = 0; i < slice.Length; i++)
        {
            slice[i] = MathF.Exp(slice[i] - max);
            sum += slice[i];
        }
        for (int i = 0; i < slice.Length; i++) slice[i] /= sum;
    }
}
