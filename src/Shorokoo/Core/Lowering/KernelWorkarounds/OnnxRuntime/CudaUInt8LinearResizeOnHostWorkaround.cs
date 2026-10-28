using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;

namespace Shorokoo.Core.Lowering.KernelWorkarounds.OnnxRuntime;

using static OnnxOpAttributeNames;
using static OpCodes;

/// <summary>
/// A linear <c>Resize</c> of uint8 without <c>antialias</c>, kept on the host (Shorokoo/Shorokoo#557).
///
/// <para>ONNX Runtime's CUDA execution provider computes such a call wrongly, with no error: a row
/// <c>[0, 255, 255, 0, 0, 255, 0, 0]</c> resized by 1.7 comes out all zeros, by 0.6 as
/// <c>[0, 255, 0, 0]</c> where the spec gives <c>[84, 255, 0, 170]</c>. With <c>antialias</c> it
/// computes the call as the spec does, and so it does every other type. Kept on the host, the call
/// runs on the CPU provider, at the cost of copying its operands to the host and its result back.</para>
///
/// <para>Part of the CUDA set only (<see cref="Shorokoo.Core.Backends.KernelWorkaroundSets.OnnxRuntimeCuda"/>),
/// last, so the calls earlier rewrites build are kept on the host too. The regrouping of a resize
/// on the channels-last route (<see cref="CudaResizeMiddleAxesWorkaround"/>) leaves such a call as
/// it stands: on the host the CPU provider computes the call as written, and a call kept on the
/// host inside the <c>If</c> that regrouping builds for a call whose route is told only when it
/// runs stops the process while its session is built.</para>
/// </summary>
internal sealed class CudaUInt8LinearResizeOnHostWorkaround : KernelWorkaround
{
    public override IReadOnlySet<string> OpCodes { get; } = new HashSet<string>([RESIZE], StringComparer.Ordinal);

    public override bool KeepsOnHost => true;

    public override bool Applies(WorkaroundSite site) => Keeps(site);

    /// <summary>Whether the call at <paramref name="site"/> is one this workaround keeps on the
    /// host.</summary>
    internal static bool Keeps(WorkaroundSite site)
        => site.DTypeOf(0).IsSameElementTypeAs(DType.UInt8)
           && site.Attributes.GetEnumVal<ResizeMode>(AttrMode) == ResizeMode.Linear
           && site.Attributes.GetBoolVal(AttrAntialias) != true;

    public override Variable?[] Rewrite(WorkaroundSite site, Variable?[] inputs)
        => throw new InvalidOperationException("A call kept on the host is not rewritten.");
}
