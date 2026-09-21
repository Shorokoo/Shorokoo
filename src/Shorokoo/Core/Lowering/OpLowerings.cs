using Shorokoo.Core.Nodes.NodeDefinitions;
using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;
using static Shorokoo.Globals;

namespace Shorokoo.Core.Lowering;

/// <summary>
/// Every operator the framework knows how to compute out of simpler ones, one
/// <see cref="OpLoweringAttribute"/>-marked method each.
///
/// <para>A lowering is ordinary Shorokoo code, written exactly as a <c>[Module]</c> function or an
/// <c>[AutoDiff]</c> gradient rule is: it takes the operator's inputs and attributes and combines
/// them with normal Shorokoo operations. It may therefore branch and loop over ranks or
/// attributes like any other C#. What it must not do is compute: the values it returns are graph
/// values, and which engine evaluates them — the QuickExecutionEngine on the spot, the autodiff
/// engine as nodes to differentiate — is not its concern.</para>
/// </summary>
internal static class OpLowerings
{
    /// <summary>
    /// <c>Softsign(x) = x / (1 + |x|)</c>.
    ///
    /// <para>The <c>1</c> is cast to x's type rather than built at x's type: casting is a runtime
    /// step, so it follows whatever dtype x actually turns out to have. Reading
    /// <c>x.Type</c> in C# instead would fix the constant at whatever the engine's stand-in for x
    /// happened to carry, which on the autodiff path is float32 for every input — a float64
    /// Softsign would then add a float32 one to a float64 magnitude.</para>
    /// </summary>
    [OpLowering(SOFTSIGN)]
    public static Variable?[] Softsign<T>(Tensor<T> x) where T : IVarType
        => [x / (OneLike(x) + x.Abs())];

    /// <summary>
    /// <c>TensorScatter</c> — ONNX opset 24's KV-cache window write — as a mask and a gather.
    ///
    /// <para>Call the batch coordinate <c>b</c>, the sequence axis's coordinate <c>p</c>, the
    /// cache's length along that axis <c>S</c> and the update's <c>L</c>. Which element of the
    /// update window lands on <c>p</c> is <c>rel = p - write_indices[b]</c>: in <c>linear</c>
    /// mode the window is <c>0 &lt;= rel &lt; L</c>, in <c>circular</c> mode the same test on
    /// <c>rel</c> taken modulo <c>S</c>. The spec's loop is therefore one expression,
    /// <c>present[… p …] = inWindow ? update[… rel …] : past[… p …]</c> — a
    /// <c>GatherElements</c> along the axis, whose index tensor carries <c>rel</c>, chosen
    /// between by a <c>Where</c> on the mask. Both index and mask depend on <c>b</c> and
    /// <c>p</c> only, so they are built at <c>[batch, sequence]</c> and broadcast over the
    /// other dimensions.</para>
    ///
    /// <para><c>B</c>, <c>S</c>, <c>L</c> and the rank all come from <c>Shape</c> at runtime.
    /// The rank in particular is not read off the input in C#: what a lowering is handed is the
    /// engine's stand-in for the operand, so a rank read there is a fact about the stand-in
    /// rather than about the tensor. The vector the <c>[batch, sequence]</c> mask is reshaped
    /// by — <c>B</c> at dim 0, <c>S</c> at the sequence axis, 1 everywhere else — is assembled
    /// from <c>Shape(past_cache)</c> instead, which makes the decomposition rank-agnostic. The
    /// only C# branches are on the operator's own attributes and on whether the optional
    /// <c>write_indices</c> slot is filled; every domain states both truthfully.</para>
    ///
    /// <para>The spec's own preconditions are taken as given rather than enforced:
    /// <c>L &lt;= S</c>, and, in <c>linear</c> mode, <c>write_indices + L &lt;= S</c>. A linear
    /// window that runs past the end writes only the part that fits. <c>axis</c> may not name
    /// the batch dimension, which the spec forbids outright.</para>
    /// </summary>
    [OpLowering(TENSOR_SCATTER)]
    public static Variable?[] TensorScatter<T>(
        Tensor<T> pastCache, Tensor<T> update, Tensor<int64>? writeIndices,
        long? axis, TensorScatterMode? mode) where T : IVarType
    {
        var seqAxis = axis ?? -2L;
        if (seqAxis == 0)
            throw new ArgumentOutOfRangeException(nameof(axis),
                "TensorScatter's axis names the sequence dimension and cannot be 0, the batch dimension.");

        var pastShape = OnnxOp.Shape(pastCache);
        var rank = OnnxOp.Size(pastShape);
        // Mod against the rank rather than a C# rank: fmod is left at its default 0, whose
        // remainder takes the divisor's sign, so a negative axis lands in [0, rank).
        var seqAxisPos = OnnxOp.Mod(OnnxOp.Add(Scalar(seqAxis), rank), rank);
        var cacheLen = OnnxOp.Gather(pastShape, seqAxisPos);
        var windowLen = OnnxOp.Gather(OnnxOp.Shape(update), seqAxisPos);

        var positions = OnnxOp.Reshape(
            OnnxOp.Range(Scalar(0L), cacheLen, Scalar(1L)), Vector(1L, -1L), allowZero: false);
        Variable writeStart = writeIndices is { } indices
            ? OnnxOp.Reshape(indices, Vector(-1L, 1L), allowZero: false)
            : Scalar(0L);
        var rel = OnnxOp.Sub(positions, writeStart);
        var lastInWindow = OnnxOp.Sub(windowLen, Scalar(1L));

        Variable inWindow, source;
        if (mode == TensorScatterMode.Circular)
        {
            var wrapped = OnnxOp.Mod(rel, cacheLen);
            inWindow = OnnxOp.Less(wrapped, windowLen);
            // Already non-negative, so only the upper clamp is needed to keep the masked-out
            // positions' indices inside update's axis — GatherElements rejects the rest.
            source = OnnxOp.Min(wrapped, lastInWindow);
        }
        else
        {
            inWindow = OnnxOp.And(
                OnnxOp.GreaterOrEqual(rel, Scalar(0L)), OnnxOp.Less(rel, windowLen));
            source = OnnxOp.Max(OnnxOp.Min(rel, lastInWindow), Scalar(0L));
        }

        var dims = OnnxOp.Range(Scalar(0L), rank, Scalar(1L));
        var batchLen = OnnxOp.Gather(OnnxOp.Shape(rel), Scalar(0L));
        var spread = OnnxOp.Where(OnnxOp.Equal(dims, seqAxisPos), cacheLen,
            OnnxOp.Where(OnnxOp.Equal(dims, Scalar(0L)), batchLen, Scalar(1L)));

        var gathered = OnnxOp.GatherElements(
            update, OnnxOp.Expand(OnnxOp.Reshape(source, spread, allowZero: true), pastShape), seqAxis);
        return [OnnxOp.Where(OnnxOp.Reshape(inWindow, spread, allowZero: true), gathered, pastCache)];
    }

    private static Tensor<T> OneLike<T>(Tensor<T> like) where T : IVarType
        => OnnxOp.CastLike(Scalar(1.0f), like, saturate: null);
}
