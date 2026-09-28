using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;

namespace Shorokoo.Core.Lowering.KernelWorkarounds.OnnxRuntime;

using static OnnxOp;
using static OnnxOpAttributeNames;
using static OpCodes;

/// <summary>
/// A <c>Resize</c> with an <c>axes</c> attribute, rewritten over every axis of its input so that
/// ONNX Runtime's graph optimizations keep it intact (Shorokoo/Shorokoo#429).
///
/// <para>ONNX Runtime's transpose optimizer moves a <c>Transpose</c> through a <c>Resize</c> by
/// permuting its roi, scales and sizes as though each held one entry per axis of the input. A call
/// that names its axes holds one per named axis, so the permutation reads past their end and the
/// session fails, or reorders them wrongly when the named axes are all of them out of order. The
/// kernel itself computes such a call as the spec does, and it also refuses negative axes, which
/// the rewrite counts from the front.</para>
///
/// <para>The rewrite writes each operand out over every axis: the named axes keep their entries,
/// and every other axis takes scale 1, its own extent as size, and the roi <c>[0, 1]</c>, which
/// leave it as it is in every mode. A constant operand of an input whose rank the graph tells is
/// written out as a constant; otherwise the operand is placed with <c>ScatterElements</c> over a
/// vector as long as the input's rank.</para>
///
/// <para>A <c>not_larger</c> or <c>not_smaller</c> policy takes its common scale over the named
/// axes alone, which no call over every axis reproduces. A call that names every axis is
/// reordered; one that names some keeps its axes and reads its input through
/// <c>OptionalGetElement(Optional(x))</c>, the same tensor, which the transpose optimizer cannot
/// move a <c>Transpose</c> through and constant folding leaves in place.</para>
/// </summary>
internal sealed class ResizeAxesSubsetWorkaround : KernelWorkaround
{
    public override IReadOnlySet<string> OpCodes { get; } = new HashSet<string>([RESIZE], StringComparer.Ordinal);

    public override bool Applies(WorkaroundSite site)
    {
        var a = site.Attributes;
        if (a.GetLongsVal(AttrAxes) is not { } axes) return false;
        return true;
    }

    public override Variable?[] Rewrite(WorkaroundSite site, Variable?[] inputs)
    {
        var a = site.Attributes;
        var (x, roi, scales, sizes) = (inputs[0]!, inputs[1], inputs[2], inputs[3]);
        var axes = a.GetLongsVal(AttrAxes)!;
        var rank = Rank(site);
        var policy = a.GetEnumVal<KeepAspectRatioPolicy>(AttrKeepAspectRatioPolicy);

        if (policy is not (null or KeepAspectRatioPolicy.stretch)
            && !(rank is { } known && Normalized(axes, known).Distinct().Count() == known))
            return [Resize(OptionalGetElement(Optional(x, DataStructure.Tensor, site.DTypeOf(0))),
                roi, scales, sizes, a.GetBoolVal(AttrAntialias), rank is { } r ? Normalized(axes, r) : axes,
                a.GetEnumVal<CoordinateTransformationMode>(AttrCoordinateTransformationMode), a.GetFloatVal(AttrCubicCoeffA),
                a.GetBoolVal(AttrExcludeOutside), a.GetFloatVal(AttrExtrapolationValue), policy,
                a.GetEnumVal<ResizeMode>(AttrMode), a.GetEnumVal<NearestMode>(AttrNearestMode))];

        Variable? Spread(int slot, Variable? operand, Func<Variable> filler, Func<Variable, Variable> scattered)
        {
            if (operand is null) return null;
            if (rank is not { } r) return scattered(operand);
            var positions = Normalized(axes, r);
            bool isRoi = slot == 1, isSizes = slot == 3;
            if (site.ConstantOf(slot) is { } constant && Written(constant, positions, r, isRoi, isSizes) is { } written)
                return written;
            return Gather(Concat([operand, filler()], axis: 0), Globals.Vector(Indices(positions, r, isRoi, isSizes)), axis: 0);
        }

        var fullRoi = Spread(1, roi, () => CastLike(Globals.Vector(0f, 1f), roi!, saturate: null), operand => ScatteredRoi(x, operand, axes));
        var fullScales = Spread(2, scales, () => Globals.Vector(1f),
            operand => ScatterElements(Expand(Globals.Vector(1f), Shape(Shape(x))), Globals.Vector(axes), operand, axis: 0));
        var fullSizes = Spread(3, sizes, () => Shape(x), operand => ScatterElements(Shape(x), Globals.Vector(axes), operand, axis: 0));

        return [Resize(x, fullRoi, fullScales, fullSizes, a.GetBoolVal(AttrAntialias), null,
            a.GetEnumVal<CoordinateTransformationMode>(AttrCoordinateTransformationMode), a.GetFloatVal(AttrCubicCoeffA),
            a.GetBoolVal(AttrExcludeOutside), a.GetFloatVal(AttrExtrapolationValue),
            policy, a.GetEnumVal<ResizeMode>(AttrMode),
            a.GetEnumVal<NearestMode>(AttrNearestMode))];
    }

    private static int? Rank(WorkaroundSite site) => site.RankOf(0) ?? site.OutputRankOf(0);

    private static long[] Normalized(long[] axes, int rank) => [.. axes.Select(axis => axis < 0 ? axis + rank : axis)];

    /// <summary>
    /// For each axis of a rank-<paramref name="rank"/> input, the index of its entry in the call's
    /// operand followed by what stands for an unnamed axis: index <c>n</c> (scale 1) for scales,
    /// <c>n + axis</c> (the input's own extent) for sizes, and <c>2n</c> and <c>2n + 1</c> (0 and
    /// 1) for the start and end of a roi.
    /// </summary>
    private static long[] Indices(long[] positions, int rank, bool isRoi, bool isSizes)
    {
        long n = positions.Length;
        long EntryOf(long axis, long offset, long otherwise) => Array.IndexOf(positions, axis) is var k and >= 0 ? k + offset : otherwise;
        if (isRoi)
            return [.. Enumerable.Range(0, rank).Select(i => EntryOf(i, 0, 2 * n)),
                .. Enumerable.Range(0, rank).Select(i => EntryOf(i, n, 2 * n + 1))];
        return [.. Enumerable.Range(0, rank).Select(i => EntryOf(i, 0, isSizes ? n + i : n))];
    }

    /// <summary>A constant operand written out over every axis, or null when it cannot be at
    /// build time: sizes that leave an axis to the input's extent, or a dtype it does not hold.</summary>
    private static Variable? Written(TensorAttribute constant, long[] positions, int rank, bool isRoi, bool isSizes)
    {
        if (isSizes && positions.Distinct().Count() != rank
            || constant.Shape.Dims is not [var length] || length != (isRoi ? 2 : 1) * positions.Length)
            return null;
        var indices = Indices(positions, rank, isRoi, isSizes);
        T[] Pick<T>(T[] values, T zero, T one) => [.. indices.Select(i => i < values.Length ? values[i]
            : isRoi && i == 2 * positions.Length ? zero : one)];
        if (constant.DType == DType.Float32) return Globals.Vector(Pick(constant.Elements<float>().ToArray(), 0f, 1f));
        if (constant.DType == DType.Float64) return Globals.Vector(Pick(constant.Elements<double>().ToArray(), 0d, 1d));
        if (constant.DType == DType.Int64) return Globals.Vector(Pick(constant.Elements<long>().ToArray(), 0L, 1L));
        return null;
    }

    /// <summary>The roi written out over every axis of an input whose rank only the run tells:
    /// starts over zeros and ends over ones.</summary>
    private static Variable ScatteredRoi(Variable x, Variable roi, long[] axes)
    {
        long n = axes.Length;
        var rankShape = Shape(Shape(x));
        Variable Over(float fill, long from) => ScatterElements(
            Expand(CastLike(Globals.Vector(fill), roi, saturate: null), rankShape), Globals.Vector(axes),
            Slice(roi, Globals.Vector(from), Globals.Vector(from + n)), axis: 0);
        return Concat([Over(0f, 0), Over(1f, n)], axis: 0);
    }
}
