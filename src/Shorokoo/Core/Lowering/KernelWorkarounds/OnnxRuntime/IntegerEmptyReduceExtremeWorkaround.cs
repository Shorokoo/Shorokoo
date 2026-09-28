using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.NodeDefinitions;

namespace Shorokoo.Core.Lowering.KernelWorkarounds.OnnxRuntime;

using static OnnxOpAttributeNames;
using static OpCodes;

/// <summary>
/// <c>ReduceMax</c> or <c>ReduceMin</c> over an integer or boolean input, rewritten so that an
/// empty group takes the spec's value: the type's minimum for ReduceMax and its maximum for
/// ReduceMin (false and true for bool) (Shorokoo/Shorokoo#382).
///
/// <para>ONNX Runtime's CPU kernels give such a group 0 for every integer type, and throw when a
/// reduced axis of a boolean input has extent 0; for a floating-point type they give -inf and
/// +inf, as the spec does. Only an empty input holds an empty group, so the plain operator is
/// right for every nonempty input, and 0 is already the identity of an unsigned ReduceMax. A call
/// is left as it stands for a floating-point or unsigned-ReduceMax input, for a scalar input or a
/// <c>Constant</c> one that is not empty, and when nothing is reduced (no axes with
/// <c>noop_with_empty_axes</c> set).</para>
///
/// <para>A boolean call never reaches the boolean kernel: ReduceMax becomes the uint8 ReduceMax of
/// the input cast to uint8, cast back, and ReduceMin the negation of that over the negated input.
/// An empty uint8 group gives 0, which casts to false, so both are exact on every input, with no
/// branch. Every other call becomes an <c>If</c> on the input's element count being 0, counted as
/// the product of its shape so that ONNX Runtime folds the <c>If</c> away when the model states
/// the input's dimensions: the other side is the plain operator, and the empty side expands the
/// identity to the shape the plain operator gives the input.</para>
/// </summary>
internal sealed class IntegerEmptyReduceExtremeWorkaround : KernelWorkaround
{
    public override IReadOnlySet<string> OpCodes { get; } = new HashSet<string>([REDUCE_MAX, REDUCE_MIN], StringComparer.Ordinal);

    public override bool Applies(WorkaroundSite site)
    {
        var dtype = site.DTypeOf(0);
        bool max = site.OpCode == REDUCE_MAX;
        if (site.RankOf(0) == 0) return false;
        if (!site.IsPresent(1) && site.Attributes.GetBoolVal(AttrNoopWithEmptyAxes) == true) return false;
        if (!dtype.IsSameElementTypeAs(DType.Bool) && ReductionIdentity(dtype, max) is null) return false;
        if (max && IsUnsigned(dtype)) return false;
        return Reductions.InputMayBeEmpty(site);
    }

    public override Variable?[] Rewrite(WorkaroundSite site, Variable?[] inputs)
    {
        var input = inputs[0]!;
        var axes = inputs[1];
        var keepDims = site.Attributes.GetBoolVal(AttrKeepdims);
        var noOp = site.Attributes.GetBoolVal(AttrNoopWithEmptyAxes);
        bool max = site.OpCode == REDUCE_MAX;
        var dtype = site.DTypeOf(0);

        if (dtype.IsSameElementTypeAs(DType.Bool))
        {
            Variable Any(Variable bits)
                => OnnxOp.Cast(OnnxOp.ReduceMax(OnnxOp.Cast(bits, null, DType.UInt8), axes, keepDims, noOp), null, DType.Bool);
            return [max ? Any(input) : OnnxOp.Not(Any(OnnxOp.Not(input)))];
        }

        Variable Plain(Variable data)
            => max ? OnnxOp.ReduceMax(data, axes, keepDims, noOp) : OnnxOp.ReduceMin(data, axes, keepDims, noOp);

        // The plain operator on the empty side gives the output's shape, over an input it runs on.
        return [Ops.IfElse(Reductions.IsEmpty(input),
            OnnxOp.Expand(ReductionIdentity(dtype, max)!, OnnxOp.Shape(Plain(input), null, null)), Plain(input))];
    }

    private static bool IsUnsigned(DType dtype)
        => dtype.IsSameElementTypeAs(DType.UInt8) || dtype.IsSameElementTypeAs(DType.UInt16)
            || dtype.IsSameElementTypeAs(DType.UInt32) || dtype.IsSameElementTypeAs(DType.UInt64);

    /// <summary>The identity of ReduceMax (<paramref name="max"/>) or ReduceMin over the integer
    /// type <paramref name="dtype"/>, as a constant; null for every other type.</summary>
    private static Variable? ReductionIdentity(DType dtype, bool max)
    {
        if (dtype.IsSameElementTypeAs(DType.Int8)) return Globals.Scalar(max ? sbyte.MinValue : sbyte.MaxValue);
        if (dtype.IsSameElementTypeAs(DType.Int16)) return Globals.Scalar(max ? short.MinValue : short.MaxValue);
        if (dtype.IsSameElementTypeAs(DType.Int32)) return Globals.Scalar(max ? int.MinValue : int.MaxValue);
        if (dtype.IsSameElementTypeAs(DType.Int64)) return Globals.Scalar(max ? long.MinValue : long.MaxValue);
        if (dtype.IsSameElementTypeAs(DType.UInt8)) return Globals.Scalar(max ? byte.MinValue : byte.MaxValue);
        if (dtype.IsSameElementTypeAs(DType.UInt16)) return Globals.Scalar(max ? ushort.MinValue : ushort.MaxValue);
        if (dtype.IsSameElementTypeAs(DType.UInt32)) return Globals.Scalar(max ? uint.MinValue : uint.MaxValue);
        if (dtype.IsSameElementTypeAs(DType.UInt64)) return Globals.Scalar(max ? ulong.MinValue : ulong.MaxValue);
        return null;
    }
}
