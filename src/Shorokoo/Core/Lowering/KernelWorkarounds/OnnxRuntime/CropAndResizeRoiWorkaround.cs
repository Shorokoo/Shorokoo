using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;

namespace Shorokoo.Core.Lowering.KernelWorkarounds.OnnxRuntime;

using static OnnxOp;
using static OnnxOpAttributeNames;
using static OpCodes;

/// <summary>
/// <c>Resize</c> in <c>tf_crop_and_resize</c> mode, rewritten so that ONNX Runtime applies the roi
/// along an axis whose length the resize leaves unchanged (Shorokoo/Shorokoo#380).
///
/// <para>ONNX Runtime's <c>Resize</c> kernel treats such an axis as untouched: it copies the
/// whole input through when no axis changes length, and reads an axis at scale 1 straight
/// across otherwise, before it looks at the coordinate transformation. For every other mode
/// that is the spec's result; for <c>tf_crop_and_resize</c> it is not, since the roi still
/// moves each output coordinate and one that lands outside the input takes
/// <c>extrapolation_value</c>. The input's shape may not be known when the model is built, so the
/// choice is made when it runs: an <c>If</c> takes the plain node unless some axis keeps its
/// length under a roi that moves it, and the sequence below otherwise. A constant roi of
/// <c>[0, 1]</c> along every axis moves nothing, and its call is left as it stands.</para>
///
/// <para>Each such axis is resized to <c>2L−1</c> elements under the same roi, and every second
/// element is kept. <c>tf_crop_and_resize</c> maps output coordinate <c>x</c> of an axis of input
/// length <c>L</c> and output length <c>M &gt; 1</c> to
/// <c>start·(L−1) + x·(end−start)·(L−1)/(M−1)</c>, and the mapping does not read the scale.
/// Output <c>2x</c> of <c>M = 2L−1</c> doubles both the index and the divisor of output
/// <c>x</c> of <c>M = L</c>; doubling is exact in floating point, so each kept coordinate is,
/// to the bit, the one the formula gives output <c>x</c> at <c>M = L</c>. An axis of length 1,
/// or under the roi <c>[0, 1]</c>, maps every coordinate onto itself and is left alone.</para>
///
/// <para>Which axes keep their length follows the output shape the operator derives:
/// <c>floor(L·scale)</c> from scales, the sizes themselves under <c>stretch</c>, and
/// <c>round(s·L)</c> for the common scale <c>s</c> of a <c>not_larger</c> or
/// <c>not_smaller</c> policy. Scales stay scales, with a grown axis given one that floors to
/// <c>2L−1</c>; sizes are written out under <c>stretch</c>.</para>
/// </summary>
internal sealed class CropAndResizeRoiWorkaround : KernelWorkaround
{
    public override IReadOnlySet<string> OpCodes { get; } = new HashSet<string>([RESIZE], StringComparer.Ordinal);

    public override bool Applies(WorkaroundSite site)
        => site.Attributes.GetEnumVal<CoordinateTransformationMode>(AttrCoordinateTransformationMode)
                == CoordinateTransformationMode.Tf_crop_and_resize
            && site.IsPresent(1)
            && (site.IsPresent(2) || site.IsPresent(3))
            && !IsIdentityRoi(site.ConstantOf(1));

    public override Variable?[] Rewrite(WorkaroundSite site, Variable?[] inputs)
    {
        var a = site.Attributes;
        return [CropAndResize(inputs[0]!, inputs[1]!, inputs[2], inputs[3],
            a.GetBoolVal(AttrAntialias), a.GetLongsVal(AttrAxes), a.GetFloatVal(AttrCubicCoeffA),
            a.GetBoolVal(AttrExcludeOutside), a.GetFloatVal(AttrExtrapolationValue),
            a.GetEnumVal<KeepAspectRatioPolicy>(AttrKeepAspectRatioPolicy),
            a.GetEnumVal<ResizeMode>(AttrMode), a.GetEnumVal<NearestMode>(AttrNearestMode))];
    }

    /// <summary>Whether <paramref name="roi"/> is a constant starting every axis at 0 and ending
    /// it at 1.</summary>
    private static bool IsIdentityRoi(TensorAttribute? roi)
    {
        if (roi is null) return false;
        double[] values;
        if (roi.DType == DType.Float32) values = [.. roi.Elements<float>().ToArray().Select(v => (double)v)];
        else if (roi.DType == DType.Float64) values = roi.Elements<double>().ToArray();
        else return false;
        int half = values.Length / 2;
        return values.Length % 2 == 0 && values.Take(half).All(v => v == 0) && values.Skip(half).All(v => v == 1);
    }

    private static Variable CropAndResize(Variable x, Variable roi, Variable? scales,
        Variable? sizes, bool? antialias, long[]? axes,
        float? cubicCoeffA, bool? excludeOutside,
        float? extrapolationValue, KeepAspectRatioPolicy? keepAspectRatioPolicy,
        ResizeMode? mode, NearestMode? nearestMode)
    {
        var shape = Shape(x);
        var dims = axes is null ? shape : Gather(shape, Globals.Vector(axes), axis: 0);
        var dimsF = Cast(dims, null, DType.Float32);

        Variable outDims;
        if (sizes is null)
            outDims = Cast(Floor(Mul(dimsF, scales!)), null, DType.Int64);
        else if (keepAspectRatioPolicy is null or KeepAspectRatioPolicy.stretch)
            outDims = sizes;
        else
        {
            var ratios = Where(Equal(dims, Globals.Scalar(0L)), Globals.Scalar(1f), Div(Cast(sizes, null, DType.Float32), dimsF));
            var common = keepAspectRatioPolicy == KeepAspectRatioPolicy.not_larger
                ? ReduceMin(ratios, keepdims: false)
                : ReduceMax(ratios, keepdims: false);
            outDims = Cast(Floor(Add(Mul(common, dimsF), Globals.Scalar(0.5f))), null, DType.Int64);
        }

        var bounds = Split(roi, null, axis: 0, numOutputs: 2, variadicOutputCount: 2);
        var (starts, ends) = (bounds[0], bounds[1]);
        var zero = CastLike(Globals.Scalar(0f), roi, saturate: null);
        var one = CastLike(Globals.Scalar(1f), roi, saturate: null);
        var grow = And(
            And(Equal(outDims, dims), Greater(dims, Globals.Scalar(1L))),
            Not(And(Equal(starts, zero), Equal(ends, one))));
        var anyGrows = Cast(ReduceMax(Cast(grow, null, DType.Float32), keepdims: false), null, DType.Bool);

        var twice = Cast(grow, null, DType.Int64);
        var grownDims = Add(outDims, Mul(twice, Sub(outDims, Globals.Scalar(1L))));
        var plain = Resize(x, roi, scales, sizes, antialias, axes, CoordinateTransformationMode.Tf_crop_and_resize,
            cubicCoeffA, excludeOutside, extrapolationValue, keepAspectRatioPolicy, mode, nearestMode);
        var grown = sizes is null
            ? Resize(x, roi, Where(grow, Div(Sub(Add(dimsF, dimsF), Globals.Scalar(0.5f)), dimsF), scales!), null,
                antialias, axes, CoordinateTransformationMode.Tf_crop_and_resize, cubicCoeffA, excludeOutside,
                extrapolationValue, null, mode, nearestMode)
            : Resize(x, roi, null, grownDims,
                antialias, axes, CoordinateTransformationMode.Tf_crop_and_resize, cubicCoeffA, excludeOutside,
                extrapolationValue, null, mode, nearestMode);
        var cropped = Slice(grown, Mul(outDims, Globals.Scalar(0L)), grownDims,
            axes is null ? Range(Globals.Scalar(0L), Size(dims), Globals.Scalar(1L)) : Globals.Vector(axes), Add(twice, Globals.Scalar(1L)));

        return Ops.IfElse((Scalar<bit>)anyGrows, cropped, plain);
    }
}
