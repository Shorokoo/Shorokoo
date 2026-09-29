using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.NodeDefinitions;

namespace Shorokoo.Core.Lowering.KernelWorkarounds.OnnxRuntime;

using static OnnxOp;
using static OpCodes;

/// <summary>
/// A <c>MatMul</c> with an operand that may be empty, rewritten so that its result is the spec's
/// wherever ONNX Runtime's kernels get it wrong and the choice costs nothing (Shorokoo/Shorokoo#451).
///
/// <para>Where the contraction dimension is 0 the product is all zeros. ONNX Runtime's
/// <c>MatMul</c> kernel, for every element type, leaves its output unwritten when a matrix is
/// multiplied with a batch of matrices, and when a batch of matrices or a matrix is multiplied
/// with a vector; and, where the left operand's batch dimension is 1 and the right one's is not,
/// it gives the output the left operand's batch dimension. ONNX Runtime fuses a <c>Transpose</c>
/// feeding a <c>MatMul</c> into a <c>com.microsoft</c> <c>FusedMatMul</c>, whose transposed
/// operand also leaves a batched output unwritten. A matrix, or a vector, multiplied with a
/// matrix, and a vector with a vector, are computed as the spec says.</para>
///
/// <para>Every other call is rewritten only where the choice is made when the session is built.
/// A <c>Constant</c> operand with a dimension of 0 makes the product zeros of its shape, and the
/// call becomes those zeros (<see cref="ZerosOfTheProduct"/>); two <c>Constant</c> operands that
/// are not empty keep the call as it stands. When the model states every input's dimensions and
/// the call is not in a loop body (<see cref="WorkaroundSite.ShapesAreConcrete"/>), the call's
/// result goes through an <c>If</c> on either operand or the product having a dimension of 0,
/// which gives zeros of the product's shape where one has and the result where none has. ONNX
/// Runtime folds that <c>If</c> away when it builds the session wherever the shapes follow from
/// those dimensions. Where an operand's shape is computed from the data — the output of a
/// <c>NonZero</c>, of a <c>TopK</c> with a computed <c>k</c>, or of a <c>Reshape</c> to a computed
/// shape — it is not folded, and costs a few shape operations and a pass-through of the product on
/// every run. Otherwise the call is left as it stands: the kernels' result is accepted where the
/// shapes are known only when the model runs.</para>
///
/// <para>The call runs before the <c>If</c>, which reads the operands' shapes and the product, and
/// never the operands' memory: a training step is still marked to write an updated parameter over
/// the one the call reads. Asking whether the product is empty, which it is only where an operand
/// is, keeps the call out of the branch. With the call inside the branch, ONNX Runtime fails to
/// build some sessions in which it folds that <c>If</c> (a value the branch reads goes missing
/// from the graph), models with every operand nonempty among them. So the kernels still run on
/// empty operands, and what they do there that no result can correct is accepted: refusing an
/// empty batch against an operand with no batch dimension or one of 1, and stopping the process
/// in a <c>FusedMatMul</c> that moves the batch axis of rank-3 operands.</para>
/// </summary>
internal sealed class MatMulEmptyOperandWorkaround : KernelWorkaround
{
    public override IReadOnlySet<string> OpCodes { get; } = new HashSet<string>([MATMUL], StringComparer.Ordinal);

    public override bool Applies(WorkaroundSite site)
        => (site.RankOf(0), site.RankOf(1)) is not ((1 or 2, 2) or (1, 1))
           && (HasAnEmptyConstant(site)
               || site.ShapesAreConcrete && (site.ConstantShapeOf(0) is null || site.ConstantShapeOf(1) is null));

    private static bool HasAnEmptyConstant(WorkaroundSite site)
        => site.ConstantShapeOf(0) is { } a && a.Dims.Contains(0) || site.ConstantShapeOf(1) is { } b && b.Dims.Contains(0);

    public override Variable?[] Rewrite(WorkaroundSite site, Variable?[] inputs)
    {
        var (a, b) = (inputs[0]!, inputs[1]!);
        var (shapeOfA, shapeOfB) = (Shape(a), Shape(b));
        if (HasAnEmptyConstant(site))
            return [ZerosOfTheProduct(shapeOfA, shapeOfB, site.RankOf(0), site.RankOf(1), a, site.ConstantShapeOf(0) is null ? a : b)];
        var product = MatMul(a, b);
        return [Ops.IfElse(HasNoElement(shapeOfA, shapeOfB, Shape(product)),
            ZerosOfTheProduct(shapeOfA, shapeOfB, site.RankOf(0), site.RankOf(1), product, product), product)];
    }

    /// <summary>Whether a tensor of one of <paramref name="shapes"/> is empty: whether a dimension
    /// of one is 0.</summary>
    private static Scalar<bit> HasNoElement(params Variable[] shapes)
        => (Scalar<bit>)Equal(ReduceMin(Concat(shapes, 0), keepdims: false), Globals.Scalar(0L));

    /// <summary>
    /// Zeros like <paramref name="like"/> of the shape of the product of operands of shapes
    /// <paramref name="shapeOfA"/> and <paramref name="shapeOfB"/>: a zero expanded to the left
    /// operand's shape less its last axis, then to the right operand's less its contraction axis,
    /// the second-to-last or, for a vector, its only one. Each keeps an axis of 1 in place of the
    /// other's, as the batch dimensions broadcast, unless the other operand is a vector, whose axis
    /// the product drops. With a rank unknown, that axis is kept or not at run time. The zero is
    /// <see cref="BranchValues.Zero"/> of <paramref name="source"/>, so the zeros are never a
    /// constant ONNX Runtime computes when it builds the session.
    /// </summary>
    private static Variable ZerosOfTheProduct(Variable shapeOfA, Variable shapeOfB, int? rankA, int? rankB, Variable like, Variable source)
    {
        var (none, one) = (Globals.Vector(Array.Empty<long>()), Globals.Vector(1L));
        Variable Axes(Variable shape, long from, long to) => Slice(shape, Globals.Vector(from), Globals.Vector(to));
        Variable UnlessVector(Variable axes, int? rank, Variable shape) => rank switch
        {
            1 => none,
            null => Slice(axes, Globals.Vector(0L), Sub(Min([Shape(shape), Globals.Vector(2L)]), one)),
            _ => axes,
        };

        var left = Concat([Axes(shapeOfA, 0L, -1L), UnlessVector(one, rankB, shapeOfB)], 0);
        var right = Concat([Axes(shapeOfB, 0L, -2L), UnlessVector(one, rankA, shapeOfA),
            UnlessVector(Axes(shapeOfB, -1L, long.MaxValue), rankB, shapeOfB)], 0);
        return Expand(Expand(CastLike(BranchValues.Zero(source), like, saturate: null), left), right);
    }
}
