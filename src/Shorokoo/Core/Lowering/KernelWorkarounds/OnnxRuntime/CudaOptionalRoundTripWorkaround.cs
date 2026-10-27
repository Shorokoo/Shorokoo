using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.NodeDefinitions;

namespace Shorokoo.Core.Lowering.KernelWorkarounds.OnnxRuntime;

using static OnnxOp;
using static OpCodes;

/// <summary>
/// An <c>OptionalGetElement</c> or <c>OptionalHasElement</c> of the <c>Optional</c> of a tensor,
/// rewritten without the <c>Optional</c> (Shorokoo/Shorokoo#540).
///
/// <para>ONNX Runtime's <c>Optional</c> kernel hands its input on as its output and lets the two
/// share one buffer; it then copies the one onto the other, which the CUDA execution provider's
/// copy refuses for a buffer on the card. So an <c>Optional</c> of a tensor the graph computes on
/// the card fails when it runs. The element read straight back out of it is the tensor itself, and
/// it is present: <c>OptionalHasElement</c> becomes true, and <c>OptionalGetElement</c> the tensor
/// taken through a one-element sequence (<c>SequenceAt(SequenceConstruct(x), 0)</c>), which copies
/// it once on the card. The sequence keeps what the round trip is there for where it is written as
/// a barrier: ONNX Runtime's transpose optimizer moves no <c>Transpose</c> through it, as it moves
/// none through the <c>Optional</c>. An <c>Optional</c> nothing else reads is dropped.</para>
///
/// <para>Part of the CUDA set only (<see cref="Shorokoo.Core.Backends.KernelWorkaroundSets.OnnxRuntimeCuda"/>):
/// the CPU provider's copy skips a buffer shared with itself.</para>
/// </summary>
internal sealed class CudaOptionalRoundTripWorkaround : KernelWorkaround
{
    public override IReadOnlySet<string> OpCodes { get; } =
        new HashSet<string>([OPTIONAL_GET_ELEMENT, OPTIONAL_HAS_ELEMENT], StringComparer.Ordinal);

    public override string? LooksThrough => OPTIONAL;

    public override bool Applies(WorkaroundSite site) => true;

    public override Variable?[] Rewrite(WorkaroundSite site, Variable?[] inputs)
        => [site.OpCode == OPTIONAL_GET_ELEMENT ? SequenceAt(SequenceConstruct(inputs[0]!), Globals.Scalar(0L)) : Globals.Scalar(true)];
}
