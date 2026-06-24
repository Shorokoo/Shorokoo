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
/// Shape inference for the ONNX Resize op (opset 21). Computes the output shape from either
/// the explicit <c>sizes</c> input (input[3]) or the <c>scales</c> input (input[2]) applied to
/// the input tensor's shape, honoring the <c>axes</c> attribute (only the listed dims are
/// resized; default = all dims) and <c>keep_aspect_ratio_policy</c> (with <c>sizes</c>:
/// not_larger/not_smaller derive a single uniform scale = min/max(sizes[i]/in[i]) and round
/// every resized dim). The interpolation attributes (mode, nearest_mode,
/// coordinate_transformation_mode, cubic_coeff_a, antialias, exclude_outside,
/// extrapolation_value) and the roi input only affect values, not the shape. No pixel data is
/// computed — QEE only needs the output shape so downstream operators can propagate it. A
/// sizes/scales input that is connected but value-unknown at QEE time degrades to an unknown
/// shape (the rank is still the input's rank) rather than falling back to a wrong guess.
/// </summary>
internal sealed class ResizeOp : QuickOp
{
    public override string OpCode => OpCodes.RESIZE;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs.Length > 0 ? inputs[0] : null;
        var dtype = x?.DType ?? DType.Float32;

        if (x?.Shape is null)
            return [RuntimeTensorFactory.CreateRankOnly(dtype, x?.Rank)];

        var xDims = x.Shape.Dims;
        var rank = xDims.Length;

        // Resolve the axes attribute (default: all dims, in order). Negative axes count from
        // the back; out-of-range or duplicate axes invalidate the inference.
        var axesAttr = attrs.GetLongsVal(OnnxOpAttributeNames.AttrAxes);
        int[] axes;
        if (axesAttr is null)
        {
            axes = new int[rank];
            for (int i = 0; i < rank; i++) axes[i] = i;
        }
        else
        {
            axes = new int[axesAttr.Length];
            var seen = new bool[rank];
            for (int i = 0; i < axesAttr.Length; i++)
            {
                var a = axesAttr[i] < 0 ? axesAttr[i] + rank : axesAttr[i];
                if (a < 0 || a >= rank || seen[a])
                    return [RuntimeTensorFactory.CreateRankOnly(dtype, rank)];
                seen[a] = true;
                axes[i] = (int)a;
            }
        }

        // ONNX Resize: inputs are [X, roi, scales, sizes]. sizes takes precedence when wired.
        var sizes = inputs.Length > 3 ? inputs[3] : null;
        var scales = inputs.Length > 2 ? inputs[2] : null;

        if (sizes is not null)
        {
            // Connected but value-unknown, or a length that matches neither rank nor axes →
            // unknown shape (never a guessed one).
            if (sizes.IntData is not { } sizeData || sizeData.Length != axes.Length)
                return [RuntimeTensorFactory.CreateRankOnly(dtype, rank)];

            var outDims = xDims.ToArray();
            var policy = ResolveAspectRatioPolicy(attrs);
            if (policy == KeepAspectRatioPolicy.stretch)
            {
                for (int i = 0; i < axes.Length; i++) outDims[axes[i]] = sizeData[i];
            }
            else
            {
                // not_larger / not_smaller: one uniform scale across the resized axes.
                double? scale = null;
                for (int i = 0; i < axes.Length; i++)
                {
                    if (xDims[axes[i]] <= 0) return [RuntimeTensorFactory.CreateRankOnly(dtype, rank)];
                    var s = sizeData[i] / (double)xDims[axes[i]];
                    scale = scale is null ? s
                        : policy == KeepAspectRatioPolicy.not_larger ? Math.Min(scale.Value, s)
                        : Math.Max(scale.Value, s);
                }
                if (scale is null) return [RuntimeTensorFactory.CreateRankOnly(dtype, rank)];
                // ORT rounds half away from zero (std::round) when applying the policy.
                for (int i = 0; i < axes.Length; i++)
                    outDims[axes[i]] = (long)Math.Round(scale.Value * xDims[axes[i]], MidpointRounding.AwayFromZero);
            }
            return MakeResult(dtype, rank, outDims);
        }

        if (scales is not null)
        {
            if (scales.FloatData is not { } scaleData || scaleData.Length != axes.Length)
                return [RuntimeTensorFactory.CreateRankOnly(dtype, rank)];

            var outDims = xDims.ToArray();
            for (int i = 0; i < axes.Length; i++)
                outDims[axes[i]] = (long)Math.Floor(xDims[axes[i]] * scaleData[i]);
            return MakeResult(dtype, rank, outDims);
        }

        // Neither scales nor sizes wired — invalid per spec; degrade to unknown.
        return [RuntimeTensorFactory.CreateRankOnly(dtype, rank)];
    }

    private static RuntimeTensor[] MakeResult(DType dtype, int rank, long[] outDims)
    {
        for (int i = 0; i < outDims.Length; i++)
            if (outDims[i] < 0)
                return [RuntimeTensorFactory.CreateRankOnly(dtype, rank)];
        return [RuntimeTensorFactory.Create(dtype, new Shape(outDims))];
    }

    private static KeepAspectRatioPolicy ResolveAspectRatioPolicy(OnnxCSharpAttributes attrs)
    {
        if (!attrs.IsAttributeDefined(OnnxOpAttributeNames.AttrKeepAspectRatioPolicy))
            return KeepAspectRatioPolicy.stretch;
        var obj = attrs.GetAttributeObj(OnnxOpAttributeNames.AttrKeepAspectRatioPolicy);
        return obj switch
        {
            KeepAspectRatioPolicy p => p,
            string s when s.Equals("not_larger", StringComparison.OrdinalIgnoreCase) => KeepAspectRatioPolicy.not_larger,
            string s when s.Equals("not_smaller", StringComparison.OrdinalIgnoreCase) => KeepAspectRatioPolicy.not_smaller,
            _ => KeepAspectRatioPolicy.stretch,
        };
    }
}
