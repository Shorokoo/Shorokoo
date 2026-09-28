using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.NodeDefinitions;

namespace Shorokoo.Core.Lowering.KernelWorkarounds.OnnxRuntime;

using static OnnxOp;
using static OnnxOpAttributeNames;
using static OpCodes;

/// <summary>
/// <c>Col2Im</c> over one spatial axis, which ONNX Runtime's kernel computes wrongly for some
/// strides and pads, padded or not, and differently from run to run (Shorokoo/Shorokoo#381). It
/// computes the two-axis form as the spec does, so the call is rewritten as the same
/// <c>Col2Im</c> over two axes, the second of extent 1, followed by a <c>Squeeze</c> of that
/// axis. The two are equal element for element — a unit block along a unit axis with no pads and
/// unit stride and dilation covers exactly one position, so the column layout
/// <c>[N, C·K, L]</c> reads the same either way.
///
/// <para>A call is over one axis when its output has rank 3, its <c>block_shape</c> is a constant
/// of one element, or <c>pads</c>, <c>dilations</c> or <c>strides</c> name one axis; and none of
/// them names more.</para>
/// </summary>
internal sealed class Col2ImOneAxisWorkaround : KernelWorkaround
{
    public override IReadOnlySet<string> OpCodes { get; } = new HashSet<string>([COL2IM], StringComparer.Ordinal);

    public override bool Applies(WorkaroundSite site)
    {
        var dilations = site.Attributes.GetLongsVal(AttrDilations);
        var pads = site.Attributes.GetLongsVal(AttrPads);
        var strides = site.Attributes.GetLongsVal(AttrStrides);
        return (site.OutputRankOf(0) == 3 || site.ConstantOf(2) is { Shape.Dims: [1] }
                || pads is { Length: 2 } || dilations is { Length: 1 } || strides is { Length: 1 })
            && pads is null or { Length: 2 } && dilations is null or { Length: 1 } && strides is null or { Length: 1 };
    }

    public override Variable?[] Rewrite(WorkaroundSite site, Variable?[] inputs)
    {
        var dilations = site.Attributes.GetLongsVal(AttrDilations);
        var pads = site.Attributes.GetLongsVal(AttrPads);
        var strides = site.Attributes.GetLongsVal(AttrStrides);
        var unit = Globals.Vector(1L);
        long[] liftedDilations = [dilations?[0] ?? 1L, 1L];
        long[] liftedPads = [pads?[0] ?? 0L, 0L, pads?[1] ?? 0L, 0L];
        long[] liftedStrides = [strides?[0] ?? 1L, 1L];
        var lifted = Col2Im(inputs[0]!, Concat([inputs[1]!, unit], 0), Concat([inputs[2]!, unit], 0),
            liftedDilations, liftedPads, liftedStrides);
        return [Squeeze(lifted, Globals.Vector(-1L))];
    }
}
