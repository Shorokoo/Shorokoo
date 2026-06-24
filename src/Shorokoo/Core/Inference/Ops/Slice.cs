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

internal sealed class SliceOp : QuickOp
{
    public override string OpCode => OpCodes.SLICE;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs[0];
        var dtype = x?.DType ?? DType.Float32;

        var startsIn = inputs.Length > 1 ? inputs[1]?.IntData : null;
        var endsIn = inputs.Length > 2 ? inputs[2]?.IntData : null;
        if (x?.Shape is null || startsIn is null || endsIn is null)
            return [RuntimeTensorFactory.Create(dtype, null)];

        var starts = startsIn.Value;
        var ends = endsIn.Value;
        // axes/steps connected but with unknown values: which dims get sliced (or by what
        // stride) is unknowable, so degrade to an unknown shape rather than guess defaults.
        if (inputs.Length > 3 && inputs[3] is not null && inputs[3]!.IntData is null)
            return [RuntimeTensorFactory.Create(dtype, null)];
        if (inputs.Length > 4 && inputs[4] is not null && inputs[4]!.IntData is null)
            return [RuntimeTensorFactory.Create(dtype, null)];
        var axes = inputs.Length > 3 ? inputs[3]?.IntData : null;
        var steps = inputs.Length > 4 ? inputs[4]?.IntData : null;

        var inDims = x.Shape.Dims;
        var outDims = inDims.ToArray();

        // Per-axis effective starts/steps, defaulting to identity (whole axis, step 1) for axes
        // not mentioned in the slice. Used for both shape inference and concrete data slicing.
        var effStarts = new long[inDims.Length];
        var effSteps = new long[inDims.Length];
        var effLens = new long[inDims.Length];
        for (int i = 0; i < inDims.Length; i++)
        {
            effStarts[i] = 0;
            effSteps[i] = 1;
            effLens[i] = inDims[i];
        }

        for (int i = 0; i < starts.Length; i++)
        {
            var ax = axes is { } axArr ? (int)(axArr[i] < 0 ? axArr[i] + outDims.Length : axArr[i]) : i;
            var step = steps is { } stArr ? stArr[i] : 1;
            if (step == 0) return [RuntimeTensorFactory.Create(dtype, null)]; // invalid per spec
            var dimSize = inDims[ax];
            // Negative starts/ends count from the end; then clamp per the ONNX Slice spec:
            //   step > 0: start ∈ [0, dim],   end ∈ [0, dim]
            //   step < 0: start ∈ [0, dim-1], end ∈ [-1, dim-1]  (end = -1 walks through index 0)
            var s = starts[i] < 0 ? starts[i] + dimSize : starts[i];
            var e = ends[i] < 0 ? ends[i] + dimSize : ends[i];
            long len;
            if (dimSize == 0)
            {
                s = 0;
                len = 0;
            }
            else if (step > 0)
            {
                s = Math.Clamp(s, 0, dimSize);
                e = Math.Clamp(e, 0, dimSize);
                var span = Math.Max(0, e - s);
                len = (span + step - 1) / step;
            }
            else
            {
                s = Math.Clamp(s, 0, dimSize - 1);
                e = Math.Clamp(e, -1, dimSize - 1);
                var span = Math.Max(0, s - e);
                len = (span + (-step) - 1) / (-step);
            }
            outDims[ax] = len;
            effStarts[ax] = s;
            effSteps[ax] = step;
            effLens[ax] = len;
        }

        var outShape = new Shape(outDims);
        var rt = RuntimeTensorFactory.Create(dtype, outShape);

        if (!RuntimeTensorFactory.ShouldStoreData(outShape, maxDataElements)) return [rt];

        bool hasInt = x.IntData is not null;
        bool hasFloat = x.FloatData is not null;
        bool hasBool = x.BoolData is not null;
        if (!hasInt && !hasFloat && !hasBool) return [rt];

        // Compute strides for the input tensor (row-major).
        var inStrides = new long[inDims.Length];
        long stride = 1;
        for (int i = inDims.Length - 1; i >= 0; i--)
        {
            inStrides[i] = stride;
            stride *= inDims[i];
        }

        long outCount = 1;
        for (int i = 0; i < outDims.Length; i++) outCount *= outDims[i];
        if (outCount == 0)
        {
            if (hasInt) return [rt with { IntData = ImmutableArray<long>.Empty }];
            if (hasFloat) return [rt with { FloatData = ImmutableArray<float>.Empty }];
            if (hasBool) return [rt with { BoolData = ImmutableArray<bool>.Empty }];
            return [rt];
        }

        var idx = new long[outDims.Length];
        long[]? intBuf = hasInt ? new long[outCount] : null;
        float[]? floatBuf = hasFloat ? new float[outCount] : null;
        bool[]? boolBuf = hasBool ? new bool[outCount] : null;

        for (long flat = 0; flat < outCount; flat++)
        {
            long src = 0;
            for (int d = 0; d < outDims.Length; d++)
                src += (effStarts[d] + idx[d] * effSteps[d]) * inStrides[d];
            if (intBuf is not null) intBuf[flat] = x.IntData!.Value[(int)src];
            else if (floatBuf is not null) floatBuf[flat] = x.FloatData!.Value[(int)src];
            else if (boolBuf is not null) boolBuf[flat] = x.BoolData!.Value[(int)src];
            // Increment multi-dim index.
            for (int d = outDims.Length - 1; d >= 0; d--)
            {
                idx[d]++;
                if (idx[d] < outDims[d]) break;
                idx[d] = 0;
            }
        }

        if (intBuf is not null) return [rt with { IntData = ImmutableArray.Create(intBuf) }];
        if (floatBuf is not null) return [rt with { FloatData = ImmutableArray.Create(floatBuf) }];
        if (boolBuf is not null) return [rt with { BoolData = ImmutableArray.Create(boolBuf) }];
        return [rt];
    }
}
