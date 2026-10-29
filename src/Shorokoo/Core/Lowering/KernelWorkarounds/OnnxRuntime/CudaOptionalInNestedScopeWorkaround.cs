using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.NodeDefinitions;

namespace Shorokoo.Core.Lowering.KernelWorkarounds.OnnxRuntime;

using static OnnxOp;
using static OpCodes;

/// <summary>
/// The <c>Optional</c> of a tensor that something other than a read straight back out of it
/// reads, built in a scope of its own: an <c>If</c> on a constant <c>true</c> whose branches are
/// each that <c>Optional</c>, over the tensor from outside them (Shorokoo/Shorokoo#540).
///
/// <para>ONNX Runtime's <c>Optional</c> kernel declares its output an alias of its input, and its
/// session lets the two share one buffer when the input is a value computed in the same graph
/// and the <c>Optional</c> is not a graph output; the kernel then copies the one onto the other,
/// which the CUDA execution provider's copy refuses for a buffer on the card. A value a branch
/// reads from outside it is not one the branch computed, so the session gives the branch's
/// <c>Optional</c> of it a buffer of its own, the copy is between two buffers, and the optional
/// comes out of the <c>If</c> as the call's. That covers an <c>Optional</c> a branch builds and returns,
/// as the sum of an optional's gradients is, and one an <c>Identity</c> or any other reader takes.
/// The condition is a constant on the host, which the provider's <c>If</c> reads with no copy.
/// <see cref="CudaOptionalRoundTripWorkaround"/>, before this, has already written out every
/// tensor read straight back out of its <c>Optional</c>, and an <c>Optional</c> of nothing is
/// left as it stands: the kernel copies nothing for it.</para>
///
/// <para>Part of the CUDA set only (<see cref="Shorokoo.Core.Backends.KernelWorkaroundSets.OnnxRuntimeCuda"/>):
/// the CPU provider's copy skips a buffer shared with itself.</para>
/// </summary>
internal sealed class CudaOptionalInNestedScopeWorkaround : KernelWorkaround
{
    public override IReadOnlySet<string> OpCodes { get; } = new HashSet<string>([OPTIONAL], StringComparer.Ordinal);

    public override bool Applies(WorkaroundSite site) => site.IsPresent(0);

    public override Variable?[] Rewrite(WorkaroundSite site, Variable?[] inputs)
    {
        var x = inputs[0]!;
        var open = IfOpen(Globals.Scalar(true));
        return [IfClose([Optional(x, DataStructure.Tensor, site.DTypeOf(0))], [Optional(x, DataStructure.Tensor, site.DTypeOf(0))], open)[0]];
    }
}
