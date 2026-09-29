using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.NodeDefinitions;

namespace Shorokoo.Core.Lowering.KernelWorkarounds.OnnxRuntime;

using static OnnxOp;

/// <summary>
/// What the workarounds that build an <c>If</c> share. ONNX Runtime fails to build a session in
/// which it has folded an <c>If</c> to a branch holding a constant of 128 bytes or more
/// (Shorokoo/Shorokoo#455), and it computes a value when it builds the session wherever the value
/// follows from constants and the shapes the model states. A branch value built from
/// <see cref="Zero"/> follows from neither, so it is never such a constant, and it costs a few
/// nodes on the side of the <c>If</c> that holds it alone. It also fails to build a session in
/// which it has folded an <c>If</c> whose branches alone read a value outside it that a
/// <c>Shape</c> it folds also reads (<see cref="Held"/>).
/// </summary>
internal static class BranchValues
{
    /// <summary>
    /// An int64 scalar 0: the sum of no element of <paramref name="source"/>, cast to int64. ONNX
    /// Runtime does not compute it when it builds the session unless <paramref name="source"/> is
    /// itself a constant.
    /// </summary>
    public static Variable Zero(Variable source)
        => ReduceSum(Cast(Slice(Reshape(source, Globals.Vector(-1L), allowZero: false), Globals.Vector(0L), Globals.Vector(0L)), null, DType.Int64),
            keepdims: false);

    /// <summary><paramref name="shape"/>, an int64 shape, plus <see cref="Zero"/> of
    /// <paramref name="source"/>: the same shape, which ONNX Runtime does not compute when it builds
    /// the session.</summary>
    public static Variable Unfolded(Variable shape, Variable source) => Add(shape, Zero(source));

    /// <summary>
    /// <paramref name="x"/> through a <c>Reshape</c> to its own shape, which ONNX Runtime computes
    /// without a copy: the value an <c>If</c> and its condition read in place of
    /// <paramref name="x"/>. When ONNX Runtime folds an <c>If</c>, the nodes of the branch it keeps
    /// read <paramref name="x"/> without the graph recording it until that pass ends; folding a
    /// later <c>Shape</c> of <paramref name="x"/> in the same pass then removes the node computing
    /// <paramref name="x"/> where nothing else outside the <c>If</c> reads it, and the session fails
    /// to build. The <c>Reshape</c> is such a reader, and the <c>If</c> and its condition are the
    /// only readers of the value it gives.
    /// </summary>
    public static Variable Held(Variable x) => Reshape(x, Shape(x), allowZero: true);
}
