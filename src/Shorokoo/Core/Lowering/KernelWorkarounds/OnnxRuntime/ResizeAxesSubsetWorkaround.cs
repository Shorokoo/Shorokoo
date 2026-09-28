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
/// vector as long as the input's rank. An operand present but empty stands for an absent one: a
/// constant one is dropped, as is the scales or sizes beside a constant non-empty other, which
/// only one of them may be; ONNX Runtime refuses a scales and a sizes neither of which is a
/// constant, and a roi only the run tells is written out as an empty vector when it is empty,
/// without a branch.</para>
///
/// <para>A <c>not_larger</c> or <c>not_smaller</c> policy takes its common scale over the named
/// axes alone, which no call over every axis reproduces. A call that names every axis is
/// reordered; one that names some keeps its axes and reads its input through
/// <c>OptionalGetElement(Optional(x))</c>, the same tensor, which the transpose optimizer cannot
/// move a <c>Transpose</c> through and constant folding leaves in place. When the graph does not
/// tell the input's rank, negative axes are counted from the front of the input reshaped to its
/// trailing axes behind one leading axis holding the rest, and the result reshaped back.</para>
/// </summary>
internal sealed class ResizeAxesSubsetWorkaround : KernelWorkaround
{
    public override IReadOnlySet<string> OpCodes { get; } = new HashSet<string>([RESIZE], StringComparer.Ordinal);

    public override bool Applies(WorkaroundSite site) => site.Attributes.GetLongsVal(AttrAxes) is not null;

    public override Variable?[] Rewrite(WorkaroundSite site, Variable?[] inputs)
    {
        var a = site.Attributes;
        bool IsEmpty(int slot) => site.ConstantOf(slot) is { Shape.Dims: [0] };
        bool IsFilled(int slot) => site.ConstantOf(slot) is { Shape.Dims: [> 0] };
        var (x, roi, scales, sizes) = (inputs[0]!, IsEmpty(1) ? null : inputs[1],
            IsEmpty(2) || IsFilled(3) ? null : inputs[2], IsEmpty(3) || IsFilled(2) ? null : inputs[3]);
        var axes = a.GetLongsVal(AttrAxes)!;
        var rank = Rank(site);
        var policy = a.GetEnumVal<KeepAspectRatioPolicy>(AttrKeepAspectRatioPolicy);
        Variable Call(Variable input, Variable? r, Variable? sc, Variable? sz, long[]? named) => Resize(input, r, sc, sz,
            a.GetBoolVal(AttrAntialias), named,
            a.GetEnumVal<CoordinateTransformationMode>(AttrCoordinateTransformationMode), a.GetFloatVal(AttrCubicCoeffA),
            a.GetBoolVal(AttrExcludeOutside), a.GetFloatVal(AttrExtrapolationValue), policy,
            a.GetEnumVal<ResizeMode>(AttrMode), a.GetEnumVal<NearestMode>(AttrNearestMode));

        if (policy is not (null or KeepAspectRatioPolicy.stretch)
            && !(rank is { } known && Normalized(axes, known).Distinct().Count() == known))
        {
            if (rank is { } r)
                return [Call(OptionalGetElement(Optional(x, DataStructure.Tensor, site.DTypeOf(0))), roi, scales, sizes, Normalized(axes, r))];
            if (axes.Any(axis => axis >= 0))
                return [Call(OptionalGetElement(Optional(x, DataStructure.Tensor, site.DTypeOf(0))), roi, scales, sizes, axes)];
            long trailing = -axes.Min();
            var folded = Reshape(x, Concat([Globals.Vector(-1L), Shape(x, start: -trailing)], axis: 0), allowZero: false);
            var resized = Call(folded, roi, scales, sizes, [.. axes.Select(axis => axis + trailing + 1)]);
            return [Reshape(resized, Concat([Shape(x, end: -trailing), Shape(resized, start: 1)], axis: 0), allowZero: false)];
        }

        Variable? Spread(int slot, Variable? operand, Func<Variable> filler, Func<Variable, Variable> scattered)
        {
            if (operand is null) return null;
            bool isRoi = slot == 1, isSizes = slot == 3;
            var constant = site.ConstantOf(slot);
            bool mayBeEmpty = isRoi && constant is null;
            var entries = !mayBeEmpty ? operand
                : Slice(Concat([operand, Expand(CastLike(Globals.Vector(0f), operand, saturate: null), Globals.Vector(2L * axes.Length))], axis: 0),
                    Globals.Vector(0L), Globals.Vector(2L * axes.Length));
            Variable full;
            if (rank is not { } r)
                full = scattered(entries);
            else
            {
                var positions = Normalized(axes, r);
                if (constant is not null && Written(constant, positions, r, isRoi, isSizes) is { } written)
                    return written;
                full = Gather(Concat([entries, filler()], axis: 0), Globals.Vector(Indices(positions, r, isRoi, isSizes)), axis: 0);
            }
            return !mayBeEmpty ? full
                : Slice(full, Globals.Vector(0L), Mul(Shape(full), Min(Shape(operand), Globals.Vector(1L))));
        }

        var fullRoi = Spread(1, roi, () => CastLike(Globals.Vector(0f, 1f), roi!, saturate: null), operand => ScatteredRoi(x, operand, axes));
        var fullScales = Spread(2, scales, () => Globals.Vector(1f),
            operand => ScatterElements(Expand(Globals.Vector(1f), Shape(Shape(x))), Globals.Vector(axes), operand, axis: 0));
        var fullSizes = Spread(3, sizes, () => Shape(x), operand => ScatterElements(Shape(x), Globals.Vector(axes), operand, axis: 0));

        return [Call(x, fullRoi, fullScales, fullSizes, null)];
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
