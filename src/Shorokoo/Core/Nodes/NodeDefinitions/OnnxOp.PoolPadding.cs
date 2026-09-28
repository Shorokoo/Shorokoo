using Shorokoo.Core.Nodes.OnnxNodes;

namespace Shorokoo.Core.Nodes.NodeDefinitions;

public static partial class OnnxOp
{
    /// <summary>What a pool computes over its window, which decides how its padding may be
    /// written out as data.</summary>
    private enum PoolKind
    {
        /// <summary>MaxPool.</summary>
        Max,
        /// <summary>LpPool.</summary>
        Lp,
        /// <summary>AveragePool with <c>count_include_pad</c> 1.</summary>
        AverageIncludingPad,
        /// <summary>AveragePool with <c>count_include_pad</c> 0.</summary>
        AverageExcludingPad,
    }

    /// <summary>Builds the plain pool over <c>Input</c> with explicit <c>Pads</c>, <c>Strides</c>,
    /// <c>CeilMode</c>, and, for an AveragePool, <c>IncludePad</c> in place of its own
    /// <c>count_include_pad</c>.</summary>
    private delegate Variable[] PlainPool(Variable input, long[] pads, long[] strides, bool ceilMode, bool includePad);

    /// <summary>
    /// Whether a pool's padding is one ONNX Runtime's pooling kernels do not pool as the spec does,
    /// so the builder writes it out itself (<see cref="PaddedPool"/>): <c>SAME_UPPER</c> or
    /// <c>SAME_LOWER</c> with a dilation above 1, which ONNX Runtime pads for the undilated kernel;
    /// <c>SAME_UPPER</c> or <c>SAME_LOWER</c> with a stride above the kernel along some axis,
    /// whose padding is negative for some input lengths and which ONNX Runtime then splits
    /// between the ends otherwise than the spec does; or explicit pads as large as the kernel
    /// along some axis, which it refuses.
    /// </summary>
    private static bool NeedsWrittenPadding(AutoPad? autoPad, long[]? dilations, long[]? kernelShape, long[]? pads, long[]? strides)
        => kernelShape is { Length: > 0 } kernel && (autoPad switch
        {
            AutoPad.SameUpper or AutoPad.SameLower =>
                dilations?.Length == kernel.Length && dilations.Any(d => d > 1)
                || strides?.Length == kernel.Length && Enumerable.Range(0, kernel.Length).Any(a => strides[a] > kernel[a]),
            null or AutoPad.NotSet => pads?.Length == 2 * kernel.Length
                && Enumerable.Range(0, pads.Length).Any(i => pads[i] >= kernel[i % kernel.Length]),
            _ => false,
        });

    /// <summary>
    /// A pool whose padding <see cref="NeedsWrittenPadding"/> holds, built from pools ONNX Runtime
    /// computes as the spec does.
    ///
    /// <para><b>SAME.</b> Along each spatial axis ONNX pads it for the dilated kernel: the output
    /// has <c>ceil(in / stride)</c> elements, and <c>(out - 1) * stride + (k - 1) * d + 1 - in</c>
    /// — <c>(k - 1) * d - (in - 1) mod stride</c> — padding is split between the ends,
    /// <c>floor(total / 2)</c> of it at the start for <c>SAME_UPPER</c> and <c>ceil(total / 2)</c>
    /// for <c>SAME_LOWER</c>, the rest at the end. With a stride above the kernel's extent the
    /// total can be negative, and a negative start padding starts the first window past the
    /// input's first element. With every stride 1 the total is <c>(k - 1) * d</c> whatever the
    /// input's length, and the pool takes it as explicit pads.
    /// Otherwise it depends on a length the input's shape may leave open until it runs, so the
    /// pool is built at stride 1 with the most padding any length asks for, <c>(k - 1) * d</c>:
    /// each window the spec takes is one of that pool's outputs, and a <c>Slice</c> computed from
    /// the input's shape picks them out, from the difference between that and the needed start
    /// padding, <c>ceil(in / stride)</c> of them, <c>stride</c> apart.</para>
    ///
    /// <para><b>Explicit pads.</b> ONNX Runtime takes pads smaller than the kernel, so up to
    /// <c>k - 1</c> of each stays the pool's and the rest is written into the input with
    /// <c>Pad</c>, as a value the window's result cannot tell from padding: the element type's
    /// lowest value for a MaxPool, 0 for an LpPool and for an AveragePool that counts its padding.
    /// An AveragePool that does not count it pools the zero-padded input and a zero-padded mask of
    /// ones alike, counting padding, and divides the one by the other — the window's sum over the
    /// number of input elements in it. The pool's windows are the spec's, so its outputs are too,
    /// save two: MaxPool's indices, which count positions in the padded input and are carried
    /// back to the input's, in the pool's <c>storage_order</c>, an index of -1 staying -1; and,
    /// with <c>ceil_mode</c>, the output's length: the spec's is
    /// <c>ceil((in + pads - (k - 1) * d - 1) / stride) + 1</c>, less one when that last window
    /// starts at or past <c>in + start pad</c>, and the pool, whose written padding moves where
    /// its end padding starts, can keep one window more, so it is sliced to the spec's.</para>
    ///
    /// <para>The lowest value is taken at run time, whatever the element type: the least of
    /// <c>-inf</c>, <c>-128</c> and <c>0</c> cast to it, which is <c>-inf</c> for a floating type,
    /// -128 for int8 and 0 for uint8 — the types MaxPool takes.</para>
    /// </summary>
    private static Variable[] PaddedPool(Variable x, PoolKind kind, AutoPad? autoPad, bool? ceilMode,
        long[]? dilations, long[] kernelShape, long[]? pads, long[]? strides, long? storageOrder, PlainPool pool)
    {
        int n = kernelShape.Length;
        long[] ones = [.. Enumerable.Repeat(1L, n)];
        long[] stride = strides?.Length == n ? strides : ones;
        long[] dilation = dilations?.Length == n ? dilations : ones;
        if (autoPad is not (AutoPad.SameUpper or AutoPad.SameLower))
            return ExplicitlyPaddedPool(x, kind, kernelShape, dilation, pads!, stride, ceilMode ?? false, storageOrder ?? 0L, pool);

        bool upper = autoPad == AutoPad.SameUpper;
        long[] reach = [.. Enumerable.Range(0, n).Select(a => (kernelShape[a] - 1) * dilation[a])];
        long[] head = [.. reach.Select(r => upper ? r / 2 : (r + 1) / 2)];
        long[] tail = [.. Enumerable.Range(0, n).Select(a => reach[a] - head[a])];

        var pooled = ExplicitlyPaddedPool(x, kind, kernelShape, dilation, [.. head, .. tail], ones, false, storageOrder ?? 0L, pool);
        if (stride.All(s => s == 1)) return pooled;

        var one = Constant(ones);
        var steps = Constant(stride);
        var length = Shape(x, start: 2L);
        var total = Sub(Constant(reach), Mod(Sub(length, one), steps));
        // floor(total / 2) or ceil(total / 2) of a total at least 1 - stride, as a truncating
        // division of the total raised by twice the stride.
        var raised = Add(total, Constant([.. stride.Select(s => upper ? 2 * s : 2 * s + 1)]));
        var needed = Sub(Div(raised, Constant([.. Enumerable.Repeat(2L, n)])), steps);
        var starts = Sub(Constant(head), needed);
        var count = Div(Add(length, Constant([.. stride.Select(s => s - 1)])), steps);
        var ends = Add(starts, Add(Mul(Sub(count, one), steps), one));
        var axes = SpatialAxes(n);
        return [.. pooled.Select(p => Slice(p, starts, ends, axes, steps))];
    }

    private static Variable[] ExplicitlyPaddedPool(Variable x, PoolKind kind, long[] kernelShape, long[] dilations,
        long[] pads, long[] strides, bool ceilMode, long storageOrder, PlainPool pool)
    {
        int n = kernelShape.Length;
        long[] written = [.. Enumerable.Range(0, 2 * n).Select(i => Math.Max(0L, pads[i] - (kernelShape[i % n] - 1)))];
        if (written.All(w => w == 0))
            return pool(x, pads, strides, ceilMode, kind == PoolKind.AverageIncludingPad);
        long[] kept = [.. Enumerable.Range(0, 2 * n).Select(i => pads[i] - written[i])];
        var padding = Constant([0L, 0L, .. written[..n], 0L, 0L, .. written[n..]]);

        Variable[] pooled;
        if (kind == PoolKind.AverageExcludingPad)
        {
            var sums = pool(Pad(x, padding, CastLike(Globals.Scalar(0f), x, null)), kept, strides, ceilMode, true)[0];
            var mask = Expand(CastLike(Globals.Scalar(1f), x, null), Concat([Constant([1L, 1L]), Shape(x, start: 2L)], 0L));
            var counts = pool(Pad(mask, padding, CastLike(Globals.Scalar(0f), x, null)), kept, strides, ceilMode, true)[0];
            pooled = [Div(sums, counts)];
        }
        else
        {
            var value = kind == PoolKind.Max
                ? Min(CastLike(Globals.Scalar(float.NegativeInfinity), x, null),
                    CastLike(Globals.Scalar(-128f), x, null), CastLike(Globals.Scalar(0f), x, null))
                : CastLike(Globals.Scalar(0f), x, null);
            var padded = Pad(x, padding, value);
            pooled = pool(padded, kept, strides, ceilMode, kind == PoolKind.AverageIncludingPad);
            if (pooled.Length > 1)
                pooled[1] = UnpaddedIndices(pooled[1], x, padded, written[..n], storageOrder);
        }

        if (!ceilMode || written[n..].All(w => w == 0)) return pooled;
        var one = Constant([.. Enumerable.Repeat(1L, n)]);
        var steps = Constant(strides);
        var length = Shape(x, start: 2L);
        var room = Add(length, Constant([.. Enumerable.Range(0, n)
            .Select(a => pads[a] + pads[a + n] - (kernelShape[a] - 1) * dilations[a] - 1 + strides[a] - 1)]));
        var windows = Add(Div(room, steps), one);
        var startsInEndPadding = GreaterOrEqual(Mul(Sub(windows, one), steps), Add(length, Constant(pads[..n])));
        var ends = Sub(windows, Cast(startsInEndPadding, null, DType.Int64));
        return [.. pooled.Select(p => Slice(p, Constant(new long[n]), ends, SpatialAxes(n)))];
    }

    /// <summary>Carries MaxPool indices counted in <paramref name="padded"/> back to
    /// <paramref name="x"/>, which it extends by <paramref name="before"/> at the start of each
    /// spatial axis: the leading batch-and-channel block is kept, and the spatial position is
    /// split into its coordinates, shifted, and recombined over <paramref name="x"/>'s extents —
    /// the last axis varying fastest for <c>storage_order</c> 0, the first for 1. An index of -1,
    /// which ONNX Runtime gives a window whose every value is the element type's lowest, stays
    /// -1.</summary>
    private static Variable UnpaddedIndices(Variable indices, Variable x, Variable padded, long[] before, long storageOrder)
    {
        int n = before.Length;
        var paddedShape = Shape(padded);
        var shape = Shape(x);
        Variable Extent(Variable s, int axis) => Gather(s, Constant((long)(axis + 2)), 0L);
        var paddedArea = ReduceProd(Shape(padded, start: 2L), null, false, null);
        var area = ReduceProd(Shape(x, start: 2L), null, false, null);

        var block = Div(indices, paddedArea);
        var rest = Mod(indices, paddedArea);
        var order = storageOrder == 0 ? Enumerable.Range(0, n).Reverse().ToArray() : Enumerable.Range(0, n).ToArray();
        var coordinates = new Variable[n];
        foreach (var axis in order)
        {
            var extent = Extent(paddedShape, axis);
            coordinates[axis] = Sub(Mod(rest, extent), Constant(before[axis]));
            rest = Div(rest, extent);
        }
        Variable position = Constant(0L);
        foreach (var axis in order.Reverse())
            position = Add(Mul(position, Extent(shape, axis)), coordinates[axis]);
        return Where(Less(indices, Constant(0L)), indices, Add(Mul(block, area), position));
    }

    private static Variable SpatialAxes(int n) => Constant([.. Enumerable.Range(2, n).Select(a => (long)a)]);
}
