namespace Shorokoo.Core.Backends;

/// <summary>
/// The strides torch gives a view, as far as a model of a translation's memory asks: whether a
/// reshape of a view is a view, or a copy torch makes because the view's strides cannot be reshaped,
/// and which operands a matrix product copies.
/// </summary>
internal static class TorchStrides
{
    /// <summary>The strides of a value of <paramref name="shape"/> laid out row by row.</summary>
    internal static long[] Contiguous(IReadOnlyList<long> shape)
    {
        var strides = new long[shape.Count];
        long step = 1;
        for (int d = shape.Count - 1; d >= 0; d--)
        {
            strides[d] = step;
            step *= Math.Max(shape[d], 1);
        }
        return strides;
    }

    /// <summary>
    /// The strides of an operator's output as a view of an input of <paramref name="shape"/> and
    /// <paramref name="strides"/>: permuted by a transpose (<paramref name="perm"/>, reversing the
    /// axes where null), none along an axis an expansion broadcasts, torch's own for a reshape —
    /// or null where torch's reshape cannot view the input and copies it — and the input's own for
    /// every other view of its shape.
    /// </summary>
    internal static long[]? OfView(string op, IReadOnlyList<long>? perm, IReadOnlyList<long> shape, long[] strides, IReadOnlyList<long> target)
    {
        switch (op)
        {
            case "Transpose":
            {
                var order = perm is { Count: > 0 } ? perm : [.. Enumerable.Range(0, strides.Length).Reverse().Select(d => (long)d)];
                return order.Count == strides.Length ? [.. order.Select(d => strides[d])] : null;
            }
            case "Expand":
            {
                var result = new long[target.Count];
                var lead = target.Count - shape.Count;
                for (int d = 0; d < result.Length; d++)
                    result[d] = d < lead || (shape[d - lead] == 1 && target[d] != 1) ? 0 : strides[d - lead];
                return result;
            }
            case "Reshape" or "Flatten" or "Squeeze" or "Unsqueeze":
                return Reshaped(shape, strides, target);
            default:
                return shape.SequenceEqual(target) ? strides : Contiguous(target);
        }
    }

    /// <summary>
    /// The elements torch's <c>matmul</c> copies of its operands, of <paramref name="aShape"/> and
    /// <paramref name="bShape"/> laid out by <paramref name="aStrides"/> and
    /// <paramref name="bStrides"/>, before it multiplies them, as ATen's <c>_matmul_impl</c> and its
    /// BLAS calls do: a product of matrices with more than two axes whose leading axes it cannot fold
    /// into rows as a view is a batched one, each operand broadcast to the batch and reshaped into
    /// one batch axis — a copy where its strides cannot be reshaped so; and on a card, cuBLAS reads a
    /// matrix only where one of its axes steps by one element and the other steps past it, and torch
    /// copies one it cannot read so.
    /// </summary>
    internal static long MatMulCopies(IReadOnlyList<long> aShape, IReadOnlyList<long> aStrides,
        IReadOnlyList<long> bShape, IReadOnlyList<long> bStrides, bool onCard)
    {
        if (aShape.Count < 2 || bShape.Count < 2 || aStrides.Count != aShape.Count || bStrides.Count != bShape.Count) return 0;
        static long Count(IEnumerable<long> shape) => shape.Aggregate(1L, (a, d) => a * d);
        long Unread(long rows, long columns, long rowStep, long columnStep, long elements)
            => !onCard || (columnStep == 1 || columns == 1) && (rowStep >= Math.Max(1, columns) || rows == 1)
                || (rowStep == 1 || rows == 1) && (columnStep >= Math.Max(1, rows) || columns == 1) ? 0 : elements;
        long Matrix(IReadOnlyList<long> shape, IReadOnlyList<long> strides)
            => Unread(shape[^2], shape[^1], strides[^2], strides[^1], Count(shape));
        if (aShape.Count == 2 && bShape.Count == 2) return Matrix(aShape, aStrides) + Matrix(bShape, bStrides);
        if (bShape.Count == 2 && Enumerable.Range(0, aShape.Count - 2).All(d => aStrides[d] == aStrides[d + 1] * aShape[d + 1]))
            return Unread(Count(aShape.Take(aShape.Count - 1)), aShape[^1], aStrides[^2], aStrides[^1], Count(aShape))
                + Matrix(bShape, bStrides);
        var rank = Math.Max(aShape.Count, bShape.Count) - 2;
        var batch = new long[rank];
        for (int d = 0; d < rank; d++)
        {
            long Axis(IReadOnlyList<long> shape) => d - (rank - (shape.Count - 2)) is var at && at >= 0 ? shape[at] : 1;
            batch[d] = Math.Max(Axis(aShape), Axis(bShape));
        }
        var product = Count(batch);
        long Batched(IReadOnlyList<long> shape, IReadOnlyList<long> strides)
        {
            long[] expandedShape = [.. batch, shape[^2], shape[^1]];
            var expanded = OfView("Expand", null, shape, [.. strides], expandedShape)!;
            long[] merged = [product, shape[^2], shape[^1]];
            return Reshaped(expandedShape, expanded, merged) is { } viewed
                ? Matrix(merged, viewed)
                : Count(merged);
        }
        return Batched(aShape, aStrides) + Batched(bShape, bStrides);
    }

    /// <summary>The strides torch gives a view of a tensor of <paramref name="shape"/> and
    /// <paramref name="strides"/> reshaped to <paramref name="target"/>, or null where it cannot
    /// view it and copies it (ATen's <c>computeStride</c>).</summary>
    internal static long[]? Reshaped(IReadOnlyList<long> shape, IReadOnlyList<long> strides, IReadOnlyList<long> target)
    {
        if (shape.Count == 0) return [.. target.Select(_ => 1L)];
        if (shape.Aggregate(1L, (a, d) => a * d) == 0) return shape.SequenceEqual(target) ? [.. strides] : Contiguous(target);
        var result = new long[target.Count];
        var viewD = target.Count - 1;
        var chunkBaseStride = strides[^1];
        long tensorNumel = 1, viewNumel = 1;
        for (int tensorD = shape.Count - 1; tensorD >= 0; tensorD--)
        {
            tensorNumel *= shape[tensorD];
            if (tensorD == 0 || (shape[tensorD - 1] != 1 && strides[tensorD - 1] != tensorNumel * chunkBaseStride))
            {
                while (viewD >= 0 && (viewNumel < tensorNumel || target[viewD] == 1))
                {
                    result[viewD] = viewNumel * chunkBaseStride;
                    viewNumel *= target[viewD];
                    viewD--;
                }
                if (viewNumel != tensorNumel) return null;
                if (tensorD > 0)
                {
                    chunkBaseStride = strides[tensorD - 1];
                    tensorNumel = 1;
                    viewNumel = 1;
                }
            }
        }
        return viewD == -1 ? result : null;
    }
}
