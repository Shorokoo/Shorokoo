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
    /// <c>TensorScatter</c> — ONNX opset 24's KV-cache window write — as one gather from the
    /// cache and the update laid end to end.
    ///
    /// <para>Call the batch coordinate <c>b</c>, the sequence axis's coordinate <c>p</c>, the
    /// cache's length along that axis <c>S</c> and the update's <c>L</c>. Which element of the
    /// update window lands on <c>p</c> is <c>rel = p - write_indices[b]</c>: in <c>linear</c>
    /// mode the window is <c>0 &lt;= rel &lt; L</c>, in <c>circular</c> mode the same test on
    /// <c>rel</c> taken modulo <c>S</c>. Concatenating <c>update</c> onto <c>past_cache</c>
    /// along the axis puts both candidates for <c>p</c> in one tensor — <c>past</c>'s at
    /// <c>p</c>, <c>update</c>'s at <c>S + rel</c> — so the spec's loop becomes a single
    /// <c>GatherElements</c> whose index is <c>inWindow ? S + rel : p</c>. The index depends on
    /// <c>b</c> and <c>p</c> only, so it is built at <c>[batch, sequence]</c> and broadcast over
    /// the other dimensions.</para>
    ///
    /// <para>Selecting on the index rather than on the gathered values is what keeps the
    /// decomposition element-type agnostic: the element type reaches <c>Concat</c> and
    /// <c>GatherElements</c> and nothing else. A <c>Where</c> over the values would put the
    /// selection on the element type instead, and ONNX Runtime's CPU provider registers
    /// <c>Where</c> for six element types only — the exported file would then refuse to run for
    /// the seven others the fused operator accepts, bool and bfloat16 among them. It also
    /// removes the need to clamp: an out-of-window index is never the one selected, so no index
    /// has to be kept in range for a gather that will discard it.</para>
    ///
    /// <para><c>B</c>, <c>S</c>, <c>L</c> and the rank all come from <c>Shape</c> at runtime.
    /// The rank in particular is not read off the input in C#: what a lowering is handed is the
    /// engine's stand-in for the operand, so a rank read there is a fact about the stand-in
    /// rather than about the tensor. The vector the <c>[batch, sequence]</c> index is reshaped
    /// by — <c>B</c> at dim 0, <c>S</c> at the sequence axis, 1 everywhere else — is assembled
    /// from <c>Shape(past_cache)</c> instead, which makes the decomposition rank-agnostic. The
    /// only C# branches are on the operator's own attributes and on whether the optional
    /// <c>write_indices</c> slot is filled; every domain states both truthfully.</para>
    ///
    /// <para>What this covers is every input the spec calls valid, checked element for element
    /// against ONNX Runtime's own opset-24 kernel by
    /// <c>QeeOpset26AuditTests.TestTensorScatterLoweringMatchesTheOrtKernel*</c>: both modes,
    /// <c>write_indices</c> present or absent, every legal <c>axis</c> at ranks 2 to 4, window
    /// lengths from 1 to <c>S</c>, an empty batch, an empty window, an empty cache and every
    /// element type Shorokoo can express. What it does not cover is what the spec
    /// places outside its domain, and the companion test holds that same reference kernel to
    /// refusing each one: <c>L &gt; S</c>; in <c>linear</c> mode <c>write_indices + L &gt; S</c>;
    /// a negative or out-of-range <c>write_indices</c>; and an <c>axis</c> that does not
    /// normalize into <c>[1, rank)</c> — the spec forbids the batch dimension outright, so a
    /// rank-2 cache has to name axis 1 or −1 rather than take the default −2. Only the literal
    /// <c>axis == 0</c> is refused here, since the rank a negative axis normalizes against is
    /// not known until the graph runs.</para>
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

        Variable inWindow, offset;
        if (mode == TensorScatterMode.Circular)
        {
            // An empty cache has no position to take a remainder for, but ONNX Runtime reads the
            // scalar divisor before the element loop, so it has to be kept off zero regardless.
            offset = OnnxOp.Mod(rel, OnnxOp.Max(cacheLen, Scalar(1L)));
            inWindow = OnnxOp.Less(offset, windowLen);
        }
        else
        {
            offset = rel;
            inWindow = OnnxOp.And(
                OnnxOp.GreaterOrEqual(rel, Scalar(0L)), OnnxOp.Less(rel, windowLen));
        }
        var source = OnnxOp.Where(inWindow, OnnxOp.Add(cacheLen, offset), positions);

        var dims = OnnxOp.Range(Scalar(0L), rank, Scalar(1L));
        var batchLen = OnnxOp.Gather(OnnxOp.Shape(rel), Scalar(0L));
        var spread = OnnxOp.Where(OnnxOp.Equal(dims, seqAxisPos), cacheLen,
            OnnxOp.Where(OnnxOp.Equal(dims, Scalar(0L)), batchLen, Scalar(1L)));

        return [OnnxOp.GatherElements(
            OnnxOp.Concat([pastCache, update], seqAxis),
            OnnxOp.Expand(OnnxOp.Reshape(source, spread, allowZero: true), pastShape), seqAxis)];
    }

    private static Tensor<T> OneLike<T>(Tensor<T> like) where T : IVarType
        => OnnxOp.CastLike(Scalar(1.0f), like, saturate: null);
}
