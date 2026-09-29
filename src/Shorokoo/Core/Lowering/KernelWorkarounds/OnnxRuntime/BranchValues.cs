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
/// nodes on the side of the <c>If</c> that holds it alone.
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
}
