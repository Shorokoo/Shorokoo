using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;

namespace Shorokoo.Core.Lowering.KernelWorkarounds.OnnxRuntime;

using static OnnxOp;
using static OnnxOpAttributeNames;
using static OpCodes;

/// <summary>
/// A <c>ConvTranspose</c> with <c>SAME_UPPER</c> or <c>SAME_LOWER</c> padding whose stride exceeds
/// the dilated kernel's extent plus the output padding along some axis, rewritten as the full
/// transposed convolution cut or extended to the spec's output.
///
/// <para>The spec makes such a call's output <c>in · stride</c> long along each spatial axis,
/// padding <c>total = output_padding + (k - 1) · dilation + 1 - stride</c> in all, of which
/// <c>floor(total / 2)</c> is at the start for <c>SAME_UPPER</c> and <c>total - floor(total / 2)</c>
/// for <c>SAME_LOWER</c>. That total does not depend on the input's length, and it is negative
/// exactly when the stride exceeds the kernel's extent plus the output padding: the output then
/// reaches past the full transposed convolution, with the bias alone there. ONNX Runtime takes
/// such a total as 0 and gives the full transposed convolution.</para>
///
/// <para>The rewrite is the call without padding or bias, which is the full transposed
/// convolution and its output padding, then a <c>Pad</c> by the negated padding along each axis,
/// which crops a positive one and extends by zeros a negative one, then the bias. With the
/// kernel's extent known from <c>kernel_shape</c> or a constant weight, the padding is a constant
/// and the rewrite fires only where the spec's padding is negative; otherwise it is computed from
/// the weight's shape, for every call whose stride exceeds the output padding plus 1.</para>
/// </summary>
internal sealed class ConvTransposeSamePaddingWorkaround : KernelWorkaround
{
    public override IReadOnlySet<string> OpCodes { get; } = new HashSet<string>([CONV_TRANSPOSE], StringComparer.Ordinal);

    public override bool Applies(WorkaroundSite site)
    {
        var a = site.Attributes;
        if (a.GetEnumVal<AutoPad>(AttrAutoPad) is not (AutoPad.SameUpper or AutoPad.SameLower)
            || a.GetLongsVal(AttrOutputShape) is not null
            || Axes(site) is not { } n)
            return false;
        var (strides, dilations, outputPadding) = Spacing(a, n);
        if (Kernel(site, n) is { } kernel)
            return Enumerable.Range(0, n).Any(i => outputPadding[i] + (kernel[i] - 1) * dilations[i] + 1 < strides[i]);
        return Enumerable.Range(0, n).Any(i => outputPadding[i] + 1 < strides[i]);
    }

    public override Variable?[] Rewrite(WorkaroundSite site, Variable?[] inputs)
    {
        var a = site.Attributes;
        var (x, w, b) = (inputs[0]!, inputs[1]!, inputs[2]);
        int n = Axes(site)!.Value;
        var (strides, dilations, outputPadding) = Spacing(a, n);
        bool upper = a.GetEnumVal<AutoPad>(AttrAutoPad) == AutoPad.SameUpper;

        Variable before, after;
        if (Kernel(site, n) is { } kernel)
        {
            long[] total = [.. Enumerable.Range(0, n).Select(i => outputPadding[i] + (kernel[i] - 1) * dilations[i] + 1 - strides[i])];
            long[] start = [.. total.Select(t => upper ? FloorHalf(t) : t - FloorHalf(t))];
            before = Constant([.. start.Select(p => -p)]);
            after = Constant([.. Enumerable.Range(0, n).Select(i => start[i] - total[i])]);
        }
        else
        {
            // total = output_padding + 1 - stride + (k - 1) * dilation, at least 1 - stride; its
            // floor half as a truncating division of it raised by twice the stride.
            var total = Add(Constant([.. Enumerable.Range(0, n).Select(i => outputPadding[i] + 1 - strides[i])]),
                Mul(Sub(Shape(w, start: 2L), Constant([.. Enumerable.Repeat(1L, n)])), Constant(dilations)));
            var floorHalf = Sub(Div(Add(total, Constant([.. strides.Select(s => 2 * s)])), Constant([.. Enumerable.Repeat(2L, n)])),
                Constant(strides));
            var start = upper ? floorHalf : Sub(total, floorHalf);
            before = Neg(start);
            after = Sub(start, total);
        }

        var full = ConvTranspose(x, w, null!, AutoPad.NotSet, a.GetLongsVal(AttrDilations), a.GetLongVal(AttrGroup) ?? 1L,
            a.GetLongsVal(AttrKernelShape), a.GetLongsVal(AttrOutputPadding), null, null, a.GetLongsVal(AttrStrides));
        var zeros = Constant([0L, 0L]);
        var y = Pad(full, Concat([zeros, before, zeros, after], axis: 0), CastLike(Globals.Scalar(0f), full, null));
        if (b is not null)
            y = Add(y, Unsqueeze(b, Constant([.. Enumerable.Range(1, n).Select(i => (long)i)])));
        return [y];
    }

    private static long FloorHalf(long t) => t >= 0 ? t / 2 : -((1 - t) / 2);

    /// <summary>The number of spatial axes, from the attributes or the ranks the graph tells.</summary>
    private static int? Axes(WorkaroundSite site)
    {
        var a = site.Attributes;
        if (a.GetLongsVal(AttrStrides) is { Length: > 0 } s) return s.Length;
        if (a.GetLongsVal(AttrKernelShape) is { Length: > 0 } k) return k.Length;
        if (site.RankOf(0) is { } rank) return rank - 2;
        if (site.RankOf(1) is { } weightRank) return weightRank - 2;
        return null;
    }

    private static (long[] Strides, long[] Dilations, long[] OutputPadding) Spacing(OnnxCSharpAttributes a, int n)
    {
        long[] ones = [.. Enumerable.Repeat(1L, n)];
        return (a.GetLongsVal(AttrStrides) is { } s && s.Length == n ? s : ones,
            a.GetLongsVal(AttrDilations) is { } d && d.Length == n ? d : ones,
            a.GetLongsVal(AttrOutputPadding) is { } p && p.Length == n ? p : new long[n]);
    }

    /// <summary>The kernel's spatial extents, from <c>kernel_shape</c> or a constant weight.</summary>
    private static long[]? Kernel(WorkaroundSite site, int n)
    {
        if (site.Attributes.GetLongsVal(AttrKernelShape) is { } k && k.Length == n) return k;
        if (site.ConstantOf(1) is { } w && w.Shape.Dims.Length == n + 2) return w.Shape.Dims[2..];
        return null;
    }
}
