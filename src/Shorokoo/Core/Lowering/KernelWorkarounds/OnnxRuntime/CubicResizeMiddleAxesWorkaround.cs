using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;

namespace Shorokoo.Core.Lowering.KernelWorkarounds.OnnxRuntime;

using static OnnxOp;
using static OnnxOpAttributeNames;
using static OpCodes;

/// <summary>
/// A cubic rank-4 <c>tf_crop_and_resize</c> <c>Resize</c> that ONNX Runtime computes on its
/// channels-last route, rewritten as the same resize over the last two axes of a regrouped input
/// (Shorokoo/Shorokoo#421).
///
/// <para>ONNX Runtime resizes a 4-D input in cubic mode along its last two axes when the scales
/// of axes 0 and 1 are 1, and takes a route of its own, which treats the input as NHWC, when the
/// scales of axes 0 and 3 are 1 and that of axis 1 is not. On that route an output coordinate a
/// <c>tf_crop_and_resize</c> roi carries outside the input takes <c>extrapolation_value</c> at
/// the wrong elements; every other coordinate transformation mode it computes as the spec does.
/// The scale it tests is the one the call gives: the <c>scales</c> input itself, <c>sizes</c>
/// over the input's extents, or the common ratio of a <c>not_larger</c> or <c>not_smaller</c>
/// policy on the axes it covers.</para>
///
/// <para>The rewrite brings axes 1 and 2 last and folds axis 3 into axis 0:
/// <c>[N, C, H, W]</c> becomes <c>[N·W, 1, C, H]</c> through <c>Transpose(x, [0, 3, 1, 2])</c>
/// and a <c>Reshape</c>, whose outer two axes stay at scale 1, so the resize takes the route along
/// the last two axes; the result is reshaped and transposed back. On that route axes 0 and 3 are
/// read straight across, which is what the call's scales of 1 give them, so the regrouped call
/// carries the roi and the scale or length of axes 1 and 2 only, each along its new position. A
/// <c>not_larger</c> or <c>not_smaller</c> policy is kept, over the new positions of the axes it
/// names; one that names axis 0 or 3 scales it with axis 1, so its call never takes the
/// channels-last route and is left as it stands. The regrouping keeps ONNX Runtime's
/// own graph optimizations from folding the transposes back into the call, which a transpose
/// pair around the resize alone does not.</para>
///
/// <para>A constant scales input decides the route when the model is built; otherwise an
/// <c>If</c> chooses between the plain call and the sequence when it runs, the sequence built
/// within its branch, and its test holding only for an input of rank 4 when the graph does not
/// tell the rank. A constant roi of
/// <c>[0, 1]</c> along every axis carries no coordinate outside the input, and its call is left as
/// it stands.</para>
/// </summary>
internal sealed class CubicResizeMiddleAxesWorkaround : KernelWorkaround
{
    private static readonly long[] ToChannelsSecond = [0, 3, 1, 2];
    private static readonly long[] ToChannelsLast = [0, 2, 3, 1];

    public override IReadOnlySet<string> OpCodes { get; } = new HashSet<string>([RESIZE], StringComparer.Ordinal);

    public override bool Applies(WorkaroundSite site)
    {
        var a = site.Attributes;
        if (a.GetEnumVal<ResizeMode>(AttrMode) != ResizeMode.Cubic
            || a.GetEnumVal<CoordinateTransformationMode>(AttrCoordinateTransformationMode)
                != CoordinateTransformationMode.Tf_crop_and_resize
            || !site.IsPresent(1)
            || !(site.IsPresent(2) || site.IsPresent(3))
            || KnownRank(site) is { } rank && rank != 4)
            return false;
        var axes = Axes(a.GetLongsVal(AttrAxes));
        if (!axes.Contains(1) || ResizeRois.IsIdentity(site.ConstantOf(1))) return false;
        if (a.GetEnumVal<KeepAspectRatioPolicy>(AttrKeepAspectRatioPolicy) is not (null or KeepAspectRatioPolicy.stretch)
            && (axes.Contains(0) || axes.Contains(3)))
            return false;
        if (ConstantScales(site) is not { } scales) return true;
        if (scales.Length != axes.Length) return false;
        float ScaleOf(long axis) => Array.IndexOf(axes, axis) is var k and >= 0 ? scales[k] : 1f;
        return ScaleOf(0) == 1f && ScaleOf(3) == 1f && ScaleOf(1) != 1f;
    }

    public override Variable?[] Rewrite(WorkaroundSite site, Variable?[] inputs)
    {
        var a = site.Attributes;
        var (x, roi, scales, sizes) = (inputs[0]!, inputs[1]!, inputs[2], inputs[3]);
        long[]? writtenAxes = a.GetLongsVal(AttrAxes);
        var axes = Axes(writtenAxes);
        bool? antialias = a.GetBoolVal(AttrAntialias);
        float? cubicCoeffA = a.GetFloatVal(AttrCubicCoeffA);
        bool? excludeOutside = a.GetBoolVal(AttrExcludeOutside);
        float? extrapolationValue = a.GetFloatVal(AttrExtrapolationValue);
        var policy = a.GetEnumVal<KeepAspectRatioPolicy>(AttrKeepAspectRatioPolicy);
        var nearestMode = a.GetEnumVal<NearestMode>(AttrNearestMode);
        Variable Regroup() => Regrouped(x, roi, sizes is null ? scales : null, sizes, policy,
            axes, antialias, cubicCoeffA, excludeOutside, extrapolationValue, nearestMode);
        var rank = KnownRank(site);
        if (rank == 4 && ConstantScales(site) is not null)
            return [Regroup()];

        // An input whose rank the graph does not tell has its extents, and a call over every axis
        // its scales or sizes, read as four entries, padded with ones and cut at four, at the axes
        // counted from the front for rank 4, so the test reads within them whatever the rank; it
        // holds only for rank 4.
        Variable? FourOf(Variable? operand, Variable ones) => operand is null ? null
            : Slice(Concat([operand, ones], axis: 0), Globals.Vector(0L), Globals.Vector(4L));
        bool padded = rank is null;
        var dims = padded ? FourOf(Shape(x), Globals.Vector(1L, 1L, 1L, 1L))! : Shape(x);
        var readAxes = padded ? axes : writtenAxes;
        var readScales = padded && writtenAxes is null ? FourOf(scales, Globals.Vector(1f, 1f, 1f, 1f)) : scales;
        var readSizes = padded && writtenAxes is null ? FourOf(sizes, Globals.Vector(1L, 1L, 1L, 1L)) : sizes;
        var channelsLast = Not(IsUnscaled(dims, readScales, readSizes, policy, readAxes, axes, 1)!);
        foreach (var axis in (long[])[0, 3])
            if (IsUnscaled(dims, readScales, readSizes, policy, readAxes, axes, axis) is { } unscaled)
                channelsLast = And(unscaled, channelsLast);
        if (rank is null)
            channelsLast = And(Equal(Size(Shape(x)), Globals.Scalar(4L)), channelsLast);
        var open = IfOpen(channelsLast);
        var regrouped = Regroup();
        var plain = Resize(x, roi, scales, sizes, antialias, writtenAxes, CoordinateTransformationMode.Tf_crop_and_resize,
            cubicCoeffA, excludeOutside, extrapolationValue, policy, ResizeMode.Cubic, nearestMode);
        return [IfClose([regrouped], [plain], open)[0]];
    }

    /// <summary>
    /// The resize of axes 1 and 2 of <paramref name="x"/> over the last two axes of
    /// <c>[N·W, 1, C, H]</c>, with the roi of each and either its scale from
    /// <paramref name="scales"/> or its length from <paramref name="sizes"/> (one per resized
    /// axis), and axes 0 and 3 read straight across. Under a <c>not_larger</c> or
    /// <c>not_smaller</c> <paramref name="policy"/> the call keeps its operands and names the new
    /// positions of its axes, all of them 1 or 2.
    /// </summary>
    private static Variable Regrouped(Variable x, Variable roi, Variable? scales, Variable? sizes, KeepAspectRatioPolicy? policy,
        long[] axes, bool? antialias, float? cubicCoeffA, bool? excludeOutside, float? extrapolationValue, NearestMode? nearestMode)
    {
        var moved = Transpose(x, ToChannelsSecond);
        var folded = Reshape(moved, Concat([Globals.Vector(-1L, 1L), Shape(moved, start: 2)], axis: 0), allowZero: false);
        Variable Unfolded(Variable resized)
            => Transpose(Reshape(resized, Concat([Shape(moved, end: 2), Shape(resized, start: 2)], axis: 0), allowZero: false), ToChannelsLast);

        if (policy is not (null or KeepAspectRatioPolicy.stretch))
            return Unfolded(Resize(folded, roi, null, sizes, antialias, [.. axes.Select(axis => axis + 1)],
                CoordinateTransformationMode.Tf_crop_and_resize, cubicCoeffA, excludeOutside, extrapolationValue, policy,
                ResizeMode.Cubic, nearestMode));

        long n = axes.Length;
        long? channels = Array.IndexOf(axes, 1L) is var c and >= 0 ? c : null;
        long? height = Array.IndexOf(axes, 2L) is var h and >= 0 ? h : null;

        long zero = 2 * n, one = 2 * n + 1;
        var roiOf = Gather(Concat([roi, CastLike(Globals.Vector(0f, 1f), roi, saturate: null)], axis: 0),
            Globals.Vector(zero, zero, channels ?? zero, height ?? zero, one, one, channels + n ?? one, height + n ?? one), axis: 0);
        var scalesOf = scales is null ? null
            : Gather(Concat([scales, Globals.Vector(1f)], axis: 0), Globals.Vector(n, n, channels ?? n, height ?? n), axis: 0);

        var sizesOf = sizes is null ? null
            : Concat([Shape(folded, end: 2), Gather(Concat([sizes, Shape(x)], axis: 0),
                Globals.Vector(channels ?? n + 1, height ?? n + 2), axis: 0)], axis: 0);

        return Unfolded(Resize(folded, roiOf, scalesOf, sizesOf, antialias, null, CoordinateTransformationMode.Tf_crop_and_resize,
            cubicCoeffA, excludeOutside, extrapolationValue, null, ResizeMode.Cubic, nearestMode));
    }

    private static Variable CommonScale(Variable ratios, KeepAspectRatioPolicy policy)
        => policy == KeepAspectRatioPolicy.not_larger
            ? ReduceMin(ratios, keepdims: false)
            : ReduceMax(ratios, keepdims: false);

    /// <summary>The resized axes of a 4-D input, each counted from the front:
    /// <paramref name="axes"/>, or all four when the call names none.</summary>
    private static long[] Axes(long[]? axes) => axes is null ? [0, 1, 2, 3] : [.. axes.Select(axis => axis < 0 ? axis + 4 : axis)];

    /// <summary>The input's rank, when the graph tells it or, for a call over every axis, the
    /// length of a constant roi, scales or sizes does.</summary>
    private static int? KnownRank(WorkaroundSite site)
    {
        if ((site.RankOf(0) ?? site.OutputRankOf(0)) is { } rank) return rank;
        if (site.Attributes.GetLongsVal(AttrAxes) is not null) return null;
        if (site.ConstantShapeOf(2) is { Dims: [> 0 and var scales] }) return (int)scales;
        if (site.ConstantShapeOf(3) is { Dims: [var sizes] }) return (int)sizes;
        if (site.ConstantShapeOf(1) is { Dims: [var roi] }) return (int)(roi / 2);
        return null;
    }

    /// <summary>The call's scales, when a constant gives them and no sizes input is present.</summary>
    private static float[]? ConstantScales(WorkaroundSite site)
        => !site.IsPresent(3) && site.ConstantOf(2) is { } scales && scales.DType == DType.Float32
            ? scales.Elements<float>().ToArray()
            : null;

    /// <summary>Whether ONNX Runtime reads <paramref name="axis"/> at scale 1, as a boolean
    /// scalar of the graph; null for an axis the call does not resize. <paramref name="dims"/>
    /// is read at the axes as written, so the graph holds for an input of any rank.</summary>
    private static Variable? IsUnscaled(Variable dims, Variable? scales, Variable? sizes, KeepAspectRatioPolicy? policy,
        long[]? writtenAxes, long[] axes, long axis)
    {
        int k = Array.IndexOf(axes, axis);
        if (k < 0) return null;
        if (sizes is null) return Equal(Gather(scales!, Globals.Scalar((long)k), axis: 0), Globals.Scalar(1f));
        if (policy is null or KeepAspectRatioPolicy.stretch)
            return Equal(Gather(sizes, Globals.Scalar((long)k), axis: 0), Gather(dims, Globals.Scalar(writtenAxes?[k] ?? axis), axis: 0));
        var covered = Cast(writtenAxes is null ? dims : Gather(dims, Globals.Vector(writtenAxes), axis: 0), null, DType.Float32);
        return Equal(CommonScale(Div(Cast(sizes, null, DType.Float32), covered), policy.Value), Globals.Scalar(1f));
    }

}
