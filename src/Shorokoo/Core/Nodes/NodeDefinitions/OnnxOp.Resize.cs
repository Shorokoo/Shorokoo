using static Shorokoo.Globals;
using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;

namespace Shorokoo.Core.Nodes.NodeDefinitions;

public static partial class OnnxOp
{
    /// <summary>
    /// <c>Resize</c> in <c>tf_crop_and_resize</c> mode, written so that every backend applies the
    /// roi along an axis whose length the resize leaves unchanged.
    ///
    /// <para>ONNX Runtime's <c>Resize</c> kernel treats such an axis as untouched: it copies the
    /// whole input through when no axis changes length, and reads an axis at scale 1 straight
    /// across otherwise, before it looks at the coordinate transformation. For every other mode
    /// that is the spec's result; for <c>tf_crop_and_resize</c> it is not, since the roi still
    /// moves each output coordinate and one that lands outside the input takes
    /// <c>extrapolation_value</c> (Shorokoo/Shorokoo#380). The input's shape is not known while
    /// the graph is built, so the choice is made when it runs: an <c>If</c> takes the plain node
    /// unless some axis keeps its length under a roi that moves it, and the sequence below
    /// otherwise.</para>
    ///
    /// <para>Each such axis is resized to one element more and the extra element is sliced off.
    /// <c>tf_crop_and_resize</c> maps output coordinate <c>x</c> of an axis of input length
    /// <c>L</c> and output length <c>M &gt; 1</c> to
    /// <c>start·(L−1) + x·(end−start)·(L−1)/(M−1)</c>; growing <c>M</c> from <c>L</c> to
    /// <c>L+1</c> and <c>end</c> to <c>start + (end−start)·L/(L−1)</c> keeps each of the first
    /// <c>L</c> coordinates where it was. The mapping does not read the scale. An axis of length
    /// 1, or under the roi <c>[0, 1]</c>, maps every coordinate onto itself and is left
    /// alone.</para>
    ///
    /// <para>Which axes keep their length follows the output shape the operator derives:
    /// <c>floor(L·scale)</c> from scales, the sizes themselves under <c>stretch</c>, and
    /// <c>round(s·L)</c> for the common scale <c>s</c> of a <c>not_larger</c> or
    /// <c>not_smaller</c> policy. Scales stay scales, with a grown axis given one that floors to
    /// <c>L+1</c>; sizes are written out under <c>stretch</c>.</para>
    ///
    /// <para>Generated C# writes the whole <c>If</c> back as the one call it came from; see
    /// <see cref="CropAndResizeSource"/>.</para>
    /// </summary>
    private static Variable CropAndResize(Variable x, Variable roi, Variable? scales,
        Variable? sizes, bool? antialias, long[]? axes,
        float? cubicCoeffA, bool? excludeOutside,
        float? extrapolationValue, KeepAspectRatioPolicy? keepAspectRatioPolicy,
        ResizeMode? mode, NearestMode? nearestMode)
    {
        if (scales is null && sizes is null)
            return ResizeNode(x, roi, scales, sizes, antialias, axes, CoordinateTransformationMode.Tf_crop_and_resize,
                cubicCoeffA, excludeOutside, extrapolationValue, keepAspectRatioPolicy, mode, nearestMode);

        var shape = Shape(x);
        var dims = axes is null ? shape : Gather(shape, Vector(axes), axis: 0);
        var dimsF = Cast(dims, null, DType.Float32);

        Variable outDims;
        if (sizes is null)
            outDims = Cast(Floor(Mul(dimsF, scales!)), null, DType.Int64);
        else if (keepAspectRatioPolicy is null or KeepAspectRatioPolicy.stretch)
            outDims = sizes;
        else
        {
            var ratios = Where(Equal(dims, Scalar(0L)), Scalar(1f), Div(Cast(sizes, null, DType.Float32), dimsF));
            var common = keepAspectRatioPolicy == KeepAspectRatioPolicy.not_larger
                ? ReduceMin(ratios, keepdims: false)
                : ReduceMax(ratios, keepdims: false);
            outDims = Cast(Floor(Add(Mul(common, dimsF), Scalar(0.5f))), null, DType.Int64);
        }

        var bounds = Split(roi, null, axis: 0, numOutputs: 2, variadicOutputCount: 2);
        var (starts, ends) = (bounds[0], bounds[1]);
        var zero = CastLike(Scalar(0f), roi, saturate: null);
        var one = CastLike(Scalar(1f), roi, saturate: null);
        var grow = And(
            And(Equal(outDims, dims), Greater(dims, Scalar(1L))),
            Not(And(Equal(starts, zero), Equal(ends, one))));
        var anyGrows = Cast(ReduceMax(Cast(grow, null, DType.Float32), keepdims: false), null, DType.Bool);

        var length = CastLike(dims, roi, saturate: null);
        var span = Where(Greater(length, one), Sub(length, one), one);
        var grownEnds = Add(starts, Div(Mul(Sub(ends, starts), length), span));
        var grownRoi = Concat([starts, Where(grow, grownEnds, ends)], 0);
        var plain = ResizeNode(x, roi, scales, sizes, antialias, axes, CoordinateTransformationMode.Tf_crop_and_resize,
            cubicCoeffA, excludeOutside, extrapolationValue, keepAspectRatioPolicy, mode, nearestMode);
        var grown = sizes is null
            ? ResizeNode(x, grownRoi, Where(grow, Div(Add(dimsF, Scalar(1.5f)), dimsF), scales!), null,
                antialias, axes, CoordinateTransformationMode.Tf_crop_and_resize, cubicCoeffA, excludeOutside,
                extrapolationValue, null, mode, nearestMode)
            : ResizeNode(x, grownRoi, null, Add(outDims, Cast(grow, null, DType.Int64)),
                antialias, axes, CoordinateTransformationMode.Tf_crop_and_resize, cubicCoeffA, excludeOutside,
                extrapolationValue, null, mode, nearestMode);
        var cropped = Slice(grown, Mul(outDims, Scalar(0L)), outDims,
            axes is null ? Range(Scalar(0L), Size(dims), Scalar(1L)) : Vector(axes));

        return Ops.IfElse((Scalar<bit>)anyGrows, cropped, plain);
    }

    /// <summary>
    /// The call <see cref="CropAndResize"/> built <paramref name="close"/> from, when
    /// <paramref name="close"/> is the <c>If</c> it builds: the plain <c>Resize</c> in its else
    /// branch, and in its then branch a <c>Slice</c> of the grown <c>Resize</c>, whose roi is
    /// <c>Concat(starts, Where(grow, grown ends, ends))</c> over the two halves the plain node's roi
    /// splits into. Null for any other node.
    /// </summary>
    internal static Node? CropAndResizeSource(Node close)
        => close is { OpCode: IF_CLOSE, Outputs.Length: 1 }
            && close.FullInputs.TryGetValue(OnnxOpAttributeNames.AttrElseBranch, out var otherwise)
            && otherwise is [{ OwningNode: { OpCode: RESIZE, Inputs: [_, { } roi, ..] } plain }]
            && close.FullInputs.TryGetValue(OnnxOpAttributeNames.AttrThenBranch, out var thenBranch)
            && thenBranch is [{ OwningNode: { OpCode: SLICE, Inputs: [{ OwningNode: { OpCode: RESIZE, Inputs: [_, { } grownRoi, ..] } }, ..] } }]
            && grownRoi.OwningNode is { OpCode: CONCAT, Inputs: [{ OwningNode: { OpCode: SPLIT } split }, { OwningNode: { OpCode: WHERE } whereNode }] }
            && whereNode.Inputs is [_, _, { OwningNode: { } whereSplit }]
            && whereSplit == split && split.Inputs is [{ } splitRoi, ..] && splitRoi == roi
                ? plain
                : null;
}
