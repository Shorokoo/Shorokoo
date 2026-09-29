using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.NodeDefinitions;

namespace Shorokoo.Core.Lowering.KernelWorkarounds.OnnxRuntime;

using static OnnxOp;
using static OpCodes;

/// <summary>
/// A bitwise call — <c>BitwiseAnd</c>, <c>BitwiseOr</c>, <c>BitwiseXor</c>, <c>BitwiseNot</c> or
/// <c>BitShift</c> — inside a loop or branch body, over an int32, int64, uint32 or uint64 value the
/// body did not compute: one from an enclosing scope, or one of the body's own inputs, such as a
/// loop-carried value.
///
/// <para>ONNX Runtime's CUDA execution provider has no kernel for the bitwise operators, so it runs
/// such a call on the CPU and copies its operands off the card. Inside a body, the copy of a value
/// the body did not compute is not ordered after whatever wrote it: now and then the CPU kernel
/// reads the host buffer before the copy has filled it, and the call computes from what that memory
/// held before. Measured, a keyed draw's chunk loop comes out wrong in a few runs in a thousand,
/// and a model initialized from one seed on the card comes out different from build to build. The
/// same copy of a value the body computed itself is ordered, and so is a value that is already in
/// host memory when the body starts.</para>
///
/// <para>So each such operand is first taken through <c>Max(v, v)</c>, which is <c>v</c>, has a CUDA
/// kernel for these four types, and is computed inside the body: the copy the CPU call needs is
/// then of a value the body computed. ONNX Runtime's graph optimizations remove an <c>Identity</c>,
/// a <c>Cast</c> to the same type and an <c>Add</c> of zero, and leave this. Only the operands from
/// outside the body are taken through it, so a chain of bitwise calls over values the body
/// computed — most of a keyed draw — gains no copy.</para>
///
/// <para>Part of the CUDA set only (<see cref="Shorokoo.Core.Backends.KernelWorkaroundSets.OnnxRuntimeCuda"/>):
/// on the CPU provider every operand is already in host memory.</para>
/// </summary>
internal sealed class CudaHostFallbackBodyInputWorkaround : KernelWorkaround
{
    public override IReadOnlySet<string> OpCodes { get; } =
        new HashSet<string>([BITWISE_AND, BITWISE_OR, BITWISE_XOR, BITWISE_NOT, BIT_SHIFT], StringComparer.Ordinal);

    public override bool Applies(WorkaroundSite site)
        => Enumerable.Range(0, site.InputCount).Any(slot => Taken(site, slot));

    public override Variable?[] Rewrite(WorkaroundSite site, Variable?[] inputs)
    {
        var operands = inputs.Select((input, slot) => Taken(site, slot) ? Max([input!, input!]) : input).ToArray();
        return [site.OpCode switch
        {
            BITWISE_AND => BitwiseAnd(operands[0]!, operands[1]!),
            BITWISE_OR => BitwiseOr(operands[0]!, operands[1]!),
            BITWISE_XOR => BitwiseXor(operands[0]!, operands[1]!),
            BITWISE_NOT => BitwiseNot(operands[0]!),
            _ => BitShift(operands[0]!, operands[1]!,
                site.Attributes.IsAttributeDefined(OnnxOpAttributeNames.AttrDirection)
                    ? site.Attributes.GetAttributeObj(OnnxOpAttributeNames.AttrDirection) switch
                    {
                        BitShiftDirection d => d,
                        string s when s.Equals("RIGHT", StringComparison.OrdinalIgnoreCase) => BitShiftDirection.Right,
                        _ => BitShiftDirection.Left,
                    }
                    : null),
        }];
    }

    /// <summary>Whether the operand in <paramref name="slot"/> is taken through <c>Max</c>.</summary>
    private static bool Taken(WorkaroundSite site, int slot)
        => site.IsPresent(slot) && site.IsFromOutsideBody(slot) && HasCudaMax(site.DTypeOf(slot));

    private static bool HasCudaMax(DType dtype)
        => dtype.IsSameElementTypeAs(DType.Int32) || dtype.IsSameElementTypeAs(DType.Int64)
        || dtype.IsSameElementTypeAs(DType.UInt32) || dtype.IsSameElementTypeAs(DType.UInt64);
}
