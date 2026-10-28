using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;

namespace Shorokoo.Core.Lowering.KernelWorkarounds.OnnxRuntime;

using static OnnxOp;
using static OnnxOpAttributeNames;
using static OpCodes;

/// <summary>
/// A <c>Resize</c> in <c>tf_crop_and_resize</c> mode whose <c>roi</c> a <c>Constant</c> gives,
/// rewritten to read that <c>roi</c> as a value the card computes when the model runs
/// (Shorokoo/Shorokoo#539).
///
/// <para>ONNX Runtime's CUDA execution provider computes such a call wrongly when its <c>roi</c> is
/// an initializer of the session: its kernel returns zeros, in every mode and at every scale, with
/// no error. Given the same <c>roi</c> as a value computed when the session runs, the kernel crops
/// to it as the spec says. So the call reads <c>roi + 0</c>, the zero a sum over no element of the
/// call's own input, which no graph optimization folds into a constant, and the <c>roi</c> a copy of
/// the call's own, read by nothing else. A <c>roi</c> the graph computes is read as it
/// stands.</para>
///
/// <para>Part of the CUDA set only (<see cref="Shorokoo.Core.Backends.KernelWorkaroundSets.OnnxRuntimeCuda"/>),
/// last, so the calls earlier rewrites build are rewritten too: the CPU provider's kernel crops to
/// a constant <c>roi</c>.</para>
/// </summary>
internal sealed class CudaCropAndResizeConstantRoiWorkaround : KernelWorkaround
{
    public override IReadOnlySet<string> OpCodes { get; } = new HashSet<string>([RESIZE], StringComparer.Ordinal);

    public override bool Applies(WorkaroundSite site)
        => site.Attributes.GetEnumVal<CoordinateTransformationMode>(AttrCoordinateTransformationMode)
                == CoordinateTransformationMode.Tf_crop_and_resize
           && site.ConstantShapeOf(1) is { } roi && !roi.Dims.Contains(0);

    public override Variable?[] Rewrite(WorkaroundSite site, Variable?[] inputs)
    {
        var x = inputs[0]!;
        var roi = Constant(site.ConstantOf(1)!);
        var zero = ReduceSum(Cast(Slice(Reshape(x, Globals.Vector(-1L), allowZero: false), Globals.Vector(0L), Globals.Vector(0L)),
            null, DType.Float32), keepdims: true);
        var a = site.Attributes;
        return [Resize(x, Add(roi, CastLike(zero, roi, saturate: null)), inputs[2], inputs[3], a.GetBoolVal(AttrAntialias), a.GetLongsVal(AttrAxes),
            CoordinateTransformationMode.Tf_crop_and_resize, a.GetFloatVal(AttrCubicCoeffA), a.GetBoolVal(AttrExcludeOutside),
            a.GetFloatVal(AttrExtrapolationValue), a.GetEnumVal<KeepAspectRatioPolicy>(AttrKeepAspectRatioPolicy),
            a.GetEnumVal<ResizeMode>(AttrMode), a.GetEnumVal<NearestMode>(AttrNearestMode))];
    }
}
