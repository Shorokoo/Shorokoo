using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.NodeDefinitions;

namespace Shorokoo.Core.Lowering.KernelWorkarounds.OnnxRuntime;

using static OnnxOp;
using static OpCodes;

/// <summary>
/// The <c>Optional</c> of a tensor that something other than a read straight back out of it or the
/// model's outputs reads, built in a scope of its own: an <c>If</c> whose branches are each that
/// <c>Optional</c>, over the tensor from outside them, on a condition the card computes when the
/// model runs (Shorokoo/Shorokoo#540).
///
/// <para>ONNX Runtime's <c>Optional</c> kernel declares its output an alias of its input, and its
/// session lets the two share one buffer when the input is a value computed in the same graph
/// and the <c>Optional</c> is not a graph output; the kernel then copies the one onto the other,
/// which the CUDA execution provider's copy refuses for a buffer on the card. A value a branch
/// reads from outside it is not one the branch computed, so the session gives the branch's
/// <c>Optional</c> of it a buffer of its own, the copy is between two buffers, and the optional
/// comes out of the <c>If</c> as the call's. That covers an <c>Optional</c> a branch builds and returns,
/// as the sum of an optional's gradients is, one in a loop body, and one an <c>Identity</c> or any
/// other reader takes.</para>
///
/// <para>The session's optimizations inline an <c>If</c> on a constant condition into the graph
/// around it, which puts the <c>Optional</c> back beside the tensor it reads. So the condition is
/// <c>true</c> computed as a sum over no element of the tensor equal to zero, which no optimization
/// folds into a constant, and the <c>If</c> stands at every optimization level the session is built
/// with.</para>
///
/// <para><see cref="CudaOptionalRoundTripWorkaround"/>, before this, has already written out every
/// tensor read straight back out of its <c>Optional</c>. An <c>Optional</c> of nothing is left as
/// it stands, the kernel copying nothing for it, and so is one only the model's outputs read: the
/// session gives a graph output a buffer of its own.</para>
///
/// <para>Part of the CUDA set only (<see cref="Shorokoo.Core.Backends.KernelWorkaroundSets.OnnxRuntimeCuda"/>):
/// the CPU provider's copy skips a buffer shared with itself.</para>
/// </summary>
internal sealed class CudaOptionalInNestedScopeWorkaround : KernelWorkaround
{
    public override IReadOnlySet<string> OpCodes { get; } = new HashSet<string>([OPTIONAL], StringComparer.Ordinal);

    public override bool Applies(WorkaroundSite site) => site.IsPresent(0) && !site.IsOutputReadOnlyAsAGraphOutput(0);

    public override Variable?[] Rewrite(WorkaroundSite site, Variable?[] inputs)
    {
        var x = inputs[0]!;
        var none = ReduceSum(Cast(Slice(Reshape(x, Globals.Vector(-1L), allowZero: false), Globals.Vector(0L), Globals.Vector(0L)),
            null, DType.Float32), keepdims: false);
        var open = IfOpen(Equal(none, Globals.Scalar(0f)));
        return [IfClose([Optional(x, DataStructure.Tensor, site.DTypeOf(0))], [Optional(x, DataStructure.Tensor, site.DTypeOf(0))], open)[0]];
    }
}
