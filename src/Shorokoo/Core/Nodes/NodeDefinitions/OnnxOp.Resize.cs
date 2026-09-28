using System.Collections;
using static Shorokoo.Globals;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;
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

        var twice = Cast(grow, null, DType.Int64);
        var grownDims = Add(outDims, Mul(twice, Sub(outDims, Scalar(1L))));
        var plain = ResizeNode(x, roi, scales, sizes, antialias, axes, CoordinateTransformationMode.Tf_crop_and_resize,
            cubicCoeffA, excludeOutside, extrapolationValue, keepAspectRatioPolicy, mode, nearestMode);
        var grown = sizes is null
            ? ResizeNode(x, roi, Where(grow, Div(Sub(Add(dimsF, dimsF), Scalar(0.5f)), dimsF), scales!), null,
                antialias, axes, CoordinateTransformationMode.Tf_crop_and_resize, cubicCoeffA, excludeOutside,
                extrapolationValue, null, mode, nearestMode)
            : ResizeNode(x, roi, null, grownDims,
                antialias, axes, CoordinateTransformationMode.Tf_crop_and_resize, cubicCoeffA, excludeOutside,
                extrapolationValue, null, mode, nearestMode);
        var cropped = Slice(grown, Mul(outDims, Scalar(0L)), grownDims,
            axes is null ? Range(Scalar(0L), Size(dims), Scalar(1L)) : Vector(axes), Add(twice, Scalar(1L)));

        return Ops.IfElse((Scalar<bit>)anyGrows, cropped, plain);
    }

    /// <summary>
    /// The plain <c>Resize</c> <see cref="CropAndResize"/> built <paramref name="close"/> around,
    /// when <paramref name="close"/> is the <c>If</c> it builds; null for any other node. The
    /// match reads what the construction guarantees and what lowering, constant folding and an
    /// ONNX round trip leave in place: the else branch is a <c>tf_crop_and_resize</c>
    /// <c>Resize</c>; the then branch is a <c>Slice</c> of a second <c>Resize</c> of the same input
    /// under the same roi and attributes; and one <c>grow</c> mask decides the condition, the
    /// <c>Slice</c>'s steps, and the grown scales or sizes.
    /// </summary>
    internal static Node? CropAndResizeSource(Node close)
    {
        if (close is not { OpCode: IF_CLOSE, Outputs.Length: 1, ConnectingTensor.ParentNode: { OpCode: IF_OPEN } open }
            || !close.FullInputs.TryGetValue(AttrElseBranch, out var otherwise)
            || !close.FullInputs.TryGetValue(AttrThenBranch, out var then)
            || otherwise is not [{ OwningNode: { OpCode: RESIZE, Inputs: [{ } x, { } roi, var scales, var sizes] } plain }]
            || then is not [{ OwningNode: { OpCode: SLICE, Inputs: [{ OwningNode: { OpCode: RESIZE } grown }, _, { } ends, _,
                { OwningNode: { OpCode: ADD, Inputs: [{ OwningNode: { OpCode: CAST, Inputs: [{ } grow] } } twice, _] } }] } }]
            || grown.Inputs is not [{ } grownX, { } grownRoi, var grownScales, var grownSizes]
            || grownX != x || grownRoi != roi
            || grow.OwningNode.OpCode != AND
            || !CropAndResizeAttributesMatch(plain, grown)
            || open.Inputs is not [{ OwningNode: { OpCode: IDENTITY, Inputs: [{ OwningNode: { OpCode: CAST, Inputs:
                [{ OwningNode: { OpCode: REDUCE_MAX, Inputs: [{ OwningNode: { OpCode: CAST, Inputs: [{ } condGrow] } }, ..] } }] } }] } }]
            || condGrow != grow)
            return null;

        return sizes is null
            ? scales is not null && grownSizes is null
                && grownScales is { OwningNode: { OpCode: WHERE, Inputs: [{ } whereGrow, _, { } whereScales] } }
                && whereGrow == grow && whereScales == scales
                ? plain : null
            : grownScales is null && grownSizes == ends
                && ends.OwningNode is { OpCode: ADD, Inputs: [{ } outDims, { OwningNode: { OpCode: MUL, Inputs: [{ } mulTwice, _] } }] }
                && mulTwice == twice
                && (outDims == sizes || plain.Attributes.GetAttributeVals().GetValueOrDefault(AttrKeepAspectRatioPolicy) is not null)
                ? plain : null;
    }

    private static bool CropAndResizeAttributesMatch(Node plain, Node grown)
    {
        var plainAttrs = plain.Attributes.GetAttributeVals();
        var grownAttrs = grown.Attributes.GetAttributeVals();
        return Equals(plainAttrs.GetValueOrDefault(AttrCoordinateTransformationMode), CoordinateTransformationMode.Tf_crop_and_resize)
            && grownAttrs.GetValueOrDefault(AttrKeepAspectRatioPolicy) is null
            && plainAttrs.Keys.Union(grownAttrs.Keys).Where(k => k != AttrKeepAspectRatioPolicy).All(k =>
                StructuralComparisons.StructuralEqualityComparer.Equals(plainAttrs.GetValueOrDefault(k), grownAttrs.GetValueOrDefault(k)));
    }
}
