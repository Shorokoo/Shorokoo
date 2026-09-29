using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.NodeDefinitions;

namespace Shorokoo.Core.Lowering.KernelWorkarounds.OnnxRuntime;

using static OnnxOp;
using static OpCodes;

/// <summary>
/// A <c>MatMul</c> with an operand that may be empty, rewritten so that its result is the spec's
/// wherever ONNX Runtime's kernels get it wrong (Shorokoo/Shorokoo#451).
///
/// <para>Where the contraction dimension is 0 the product is all zeros. ONNX Runtime's
/// <c>MatMul</c> kernel, for every element type, leaves its output unwritten when a matrix is
/// multiplied with a batch of matrices, and when a batch of matrices or a matrix is multiplied
/// with a vector; and, where the left operand's batch dimension is 1 and the right one's is not,
/// it gives the output the left operand's batch dimension. ONNX Runtime fuses a <c>Transpose</c>
/// feeding a <c>MatMul</c>, directly or through a <c>Cast</c>, into a <c>com.microsoft</c>
/// <c>FusedMatMul</c>, whose transposed operand also leaves a batched output unwritten; its
/// transpose optimizer moves a <c>Transpose</c> through the ops between it and the
/// <c>MatMul</c>, so the fusion is not decided by the graph as written. Between two operands of
/// rank 3, a permutation that moves the batch axis fuses to <c>transBatchA</c> or
/// <c>transBatchB</c>, and that kernel stops the process on some empty operands. A matrix, or a
/// vector, multiplied with a matrix, and a vector with a vector, are computed as the spec
/// says.</para>
///
/// <para>Every other call becomes an <c>If</c> on either operand being empty, a dimension of its
/// shape being 0: zeros of the product's shape where one is, computed from the operands'
/// shapes (<see cref="ZerosOfTheProduct"/>). Where both operands may have a batch, rank 3 or more
/// or unknown, the <c>If</c> holds the call in its other branch, so no kernel runs on an empty
/// operand, nor gives the product a wrong shape. Otherwise the call runs before the <c>If</c>,
/// which hands its result on, and so reads the operands' shapes and not their memory: a training
/// step still writes an updated parameter over the one such a call reads, a matrix a batch is
/// multiplied with, and the call keeps its fusion. That <c>If</c> also asks whether the product is
/// empty, which it is only where an operand is, and which keeps the call out of its branch. When
/// the model states every input's dimensions, ONNX Runtime folds the <c>If</c> when it builds the
/// session wherever the operands' shapes follow from them. Two <c>Constant</c> operands that are
/// not empty keep the call as it stands.</para>
/// </summary>
internal sealed class MatMulEmptyOperandWorkaround : KernelWorkaround
{
    public override IReadOnlySet<string> OpCodes { get; } = new HashSet<string>([MATMUL], StringComparer.Ordinal);

    public override bool Applies(WorkaroundSite site)
        => (site.RankOf(0), site.RankOf(1)) is not ((1 or 2, 2) or (1, 1))
           && (MayBeEmpty(site, 0) || MayBeEmpty(site, 1));

    private static bool MayBeEmpty(WorkaroundSite site, int slot)
        => site.ConstantShapeOf(slot) is not { } shape || shape.Dims.Contains(0);

    public override Variable?[] Rewrite(WorkaroundSite site, Variable?[] inputs)
    {
        var (a, b) = (inputs[0]!, inputs[1]!);
        var (shapeOfA, shapeOfB) = (Shape(a), Shape(b));
        Variable Zeros(Variable source) => ZerosOfTheProduct(shapeOfA, shapeOfB, site.RankOf(0), site.RankOf(1), source);
        if (site.RankOf(0) is not < 3 && site.RankOf(1) is not < 3)
            return [Ops.IfElse(HasNoElement(shapeOfA, shapeOfB), Zeros(a), MatMul(a, b))];
        var product = MatMul(a, b);
        return [Ops.IfElse(HasNoElement(shapeOfA, shapeOfB, Shape(product)), Zeros(product), product)];
    }

    /// <summary>Whether a tensor of one of <paramref name="shapes"/> is empty: whether a dimension
    /// of one is 0.</summary>
    private static Scalar<bit> HasNoElement(params Variable[] shapes)
        => (Scalar<bit>)Equal(ReduceMin(Concat(shapes, 0), keepdims: false), Globals.Scalar(0L));

    /// <summary>
    /// Zeros of the shape of the product of operands of shapes <paramref name="shapeOfA"/> and
    /// <paramref name="shapeOfB"/>: a zero expanded to the left operand's shape less its last axis,
    /// then to the right operand's less its contraction axis, the second-to-last or, for a vector,
    /// its only one. Each keeps an axis of 1 in place of the other's, as the batch dimensions
    /// broadcast, unless the other operand is a vector, whose axis the product drops. With a rank
    /// unknown, that axis is kept or not at run time.
    ///
    /// <para>The zero is the sum of no element of <paramref name="source"/>, of the product's
    /// element type: a value ONNX Runtime does not compute when it builds the session, so the zeros
    /// are never a constant it folds. It fails to build a session in which it has folded an
    /// <c>If</c> to a branch holding a constant of 128 bytes or more.</para>
    /// </summary>
    private static Variable ZerosOfTheProduct(Variable shapeOfA, Variable shapeOfB, int? rankA, int? rankB, Variable source)
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
        var zero = ReduceSum(Slice(Reshape(source, Globals.Vector(-1L), allowZero: false), Globals.Vector(0L), Globals.Vector(0L)), keepdims: false);
        return Expand(Expand(zero, left), right);
    }
}
