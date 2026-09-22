namespace Shorokoo.Core.Backends;

/// <summary>
/// How many bytes an element type lays down, and how many a tensor of it covers — the question
/// every byte-wise path has to answer before it can allocate a buffer or copy one back.
///
/// <para>Public because a backend written outside this repository needs exactly this to implement
/// <see cref="IShorokooBackend.CreateTensorInBackendMemory(ShorokooTensorElementType, byte[], long[], DeviceMemorySettings)"/>
/// and the uninitialized
/// allocation beside it. Without it every such backend writes the table out again, and a table
/// written twice is a table that drifts: a dtype added to one copy and not the other is not a
/// compile error but a buffer of the wrong size.</para>
///
/// <para>The table is kept in <see cref="ShorokooTensorElementType"/>'s own vocabulary rather than
/// being routed through <c>DType</c>'s. The two agree on every fixed-stride type and disagree
/// elsewhere — this enum follows ONNX's numbering, where <c>UInt4</c> is 21, while <c>DType</c>
/// encodes it as -2 — so a cast between them is not a translation.</para>
/// </summary>
public static class TensorElementLayout
{
    /// <summary>The bytes one element of <paramref name="elementType"/> occupies.</summary>
    /// <exception cref="NotSupportedException">The element type has no fixed byte stride —
    /// <see cref="ShorokooTensorElementType.String"/> is variable-length, so
    /// <see cref="IShorokooBackend.CreateStringTensor"/> is what builds one — or is not
    /// one the byte-wise paths handle at all.</exception>
    public static int ElementSizeInBytes(ShorokooTensorElementType elementType) => elementType switch
    {
        ShorokooTensorElementType.Int8 or ShorokooTensorElementType.UInt8
            or ShorokooTensorElementType.Bool => 1,
        ShorokooTensorElementType.Int16 or ShorokooTensorElementType.UInt16
            or ShorokooTensorElementType.Float16 or ShorokooTensorElementType.BFloat16 => 2,
        ShorokooTensorElementType.Float or ShorokooTensorElementType.Int32
            or ShorokooTensorElementType.UInt32 => 4,
        ShorokooTensorElementType.Double or ShorokooTensorElementType.Int64
            or ShorokooTensorElementType.UInt64 => 8,
        ShorokooTensorElementType.String => throw new NotSupportedException(
            "String tensors are variable-length and not byte-stride; use CreateStringTensor instead."),
        _ => throw new NotSupportedException(
            $"A {elementType} tensor has no fixed byte stride, so it cannot be built or read as bytes."),
    };

    /// <summary>The bytes a tensor of <paramref name="elementType"/> and <paramref name="shape"/>
    /// covers: the size of the buffer on either side of a copy, and of the one an uninitialized
    /// allocation hands out.</summary>
    /// <exception cref="ArgumentNullException"><paramref name="shape"/> is null.</exception>
    /// <exception cref="NotSupportedException">The element type has no fixed byte stride, or
    /// <paramref name="shape"/> has no known element count.</exception>
    /// <exception cref="OverflowException">The tensor covers more bytes than an <see cref="int"/>
    /// holds, which is more than any buffer here can address.</exception>
    public static int ByteCount(ShorokooTensorElementType elementType, long[] shape)
    {
        ArgumentNullException.ThrowIfNull(shape);
        var elements = 1L;
        foreach (var dim in shape)
        {
            // Refused per dimension rather than on the product, which would read a pair of them
            // as a count of 1 and lay out a buffer that fits nothing. A negative count would in
            // turn reach `new byte[...]` through the uninitialized allocation the interface
            // defaults to, where the fault is a shape nobody can see rather than this one.
            if (dim < 0)
                throw new NotSupportedException(
                    $"A [{string.Join(", ", shape)}] tensor has no known element count: a negative "
                    + "dimension stands for one the runtime works out while it runs, not for a "
                    + "size, so there is no buffer to lay out. Size it from the arrangement the "
                    + "run resolved.");
            elements *= dim;
        }
        return checked((int)(elements * ElementSizeInBytes(elementType)));
    }
}
