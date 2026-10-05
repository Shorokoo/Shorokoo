namespace Shorokoo.Core.Backends;

/// <summary>
/// Moves a tensor value from the backend that produced it onto another one.
///
/// <para>Two backends sharing a native ONNX Runtime — a CPU backend and a CUDA backend over one
/// loaded runtime — need none of this: a value either of them made is a value the other's
/// sessions accept, and the runtime does whatever copying the device needs. This is for the
/// case where the backends do <i>not</i> share a runtime, which
/// <see cref="IsolatedBackend"/> creates: there, a value is a pointer into allocations only its
/// own runtime knows about, so it has to be rebuilt on the other side from its contents.</para>
///
/// <para>Rebuilding reads the source, so only a host-resident value can cross. A value an
/// execution provider left in its own memory (<see cref="IShorokooTensorValue.IsHostAccessible"/>)
/// cannot: no path leads from one runtime's device allocation to another's, and the copy that
/// would make it possible is one the owning backend has to be asked for.</para>
/// </summary>
public static class BackendTransfer
{
    /// <summary>
    /// An independent copy of <paramref name="value"/>, built by <paramref name="target"/> on
    /// storage of its own. Tensors are copied through their raw bytes — a piece at a time where
    /// they are more than one managed array holds — string tensors through their elements, and a
    /// sequence element by element.
    /// </summary>
    /// <exception cref="ArgumentNullException">Either argument is null.</exception>
    /// <exception cref="InvalidOperationException"><paramref name="value"/> lives in an
    /// execution provider's own memory, or is neither a tensor nor a sequence.</exception>
    public static IShorokooTensorValue CopyTo(
        IShorokooBackend target, IShorokooTensorValue value)
    {
        ArgumentNullException.ThrowIfNull(target);
        ArgumentNullException.ThrowIfNull(value);

        return value.ValueType switch
        {
            ShorokooOnnxValueType.Tensor => CopyTensor(target, value),
            ShorokooOnnxValueType.Sequence => CopySequence(target, value),
            var other => throw new InvalidOperationException(
                $"A {other} cannot be moved between backends: only tensors and sequences of them "
                + "have contents this can rebuild on the other side."),
        };
    }

    private static IShorokooTensorValue CopyTensor(
        IShorokooBackend target, IShorokooTensorValue value)
    {
        if (!value.IsHostAccessible)
            throw new InvalidOperationException(
                $"This tensor ({string.Join('x', value.Shape)}:{value.ElementType}) lives in its "
                + "own backend's device memory, so it cannot be handed to " +
                $"{target.Description}. Only a host-resident value can cross between backends "
                + "that do not share a native runtime, so bring it back to the host on the backend "
                + "that owns it first -- TensorData.ToHost() does that, and TrainingCheckpoint.ToHost() "
                + "for a whole training state.");

        if (value.ElementType == ShorokooTensorElementType.String)
            return target.CreateStringTensor(value.GetStringTensorData(), value.Shape);

        if (BytesPastOneArray(value) is { } length) return CopyByThePiece(target, value, length);

        var copy = target.CreateTensorFromRawBytes(
            value.ElementType, value.GetTensorDataAsSpan<byte>().ToArray(), value.Shape);
        // Taking the span is the source's last read, so without this the JIT may retire it before
        // ToArray has copied out of the buffer it points at (Shorokoo/Shorokoo#178).
        GC.KeepAlive(value);
        return copy;
    }

    /// <summary>The bytes <paramref name="value"/> covers where they are more than one managed array
    /// holds, and null otherwise.</summary>
    private static long? BytesPastOneArray(IShorokooTensorValue value)
    {
        if (TensorElementLayout.FixedElementSize(value.ElementType) is not { } size) return null;
        long bytes = size;
        foreach (var dim in value.Shape) bytes *= dim;
        return bytes > Array.MaxLength ? bytes : null;
    }

    /// <summary>
    /// A copy of a host value of <paramref name="length"/> bytes, more than one managed array holds,
    /// in host memory <paramref name="target"/> allocates
    /// (<see cref="IShorokooBackend.CreateUninitializedHostTensor"/>), each piece of the source's
    /// buffer copied straight into it (<see cref="IShorokooBackend.TryCopyHostToTensorRange"/>).
    /// </summary>
    private static IShorokooTensorValue CopyByThePiece(IShorokooBackend target, IShorokooTensorValue value, long length)
    {
        var copy = target.CreateUninitializedHostTensor(value.ElementType, value.Shape);
        try
        {
            for (long offset = 0; offset < length;)
            {
                var count = (int)Math.Min(length - offset, StagedReadBack.StagingBytes);
                if (!target.TryCopyHostToTensorRange(copy, offset, value.HostPiece(offset, count)))
                    throw new InvalidOperationException(
                        $"{target.Description} cannot write part of a tensor in its host memory, and this "
                        + $"one ({string.Join('x', value.Shape)}:{value.ElementType}) holds more bytes than one "
                        + "managed array does, so it cannot be copied whole either.");
                offset += count;
            }
            // The pieces are the source's last read (Shorokoo/Shorokoo#178).
            GC.KeepAlive(value);
            return copy;
        }
        catch
        {
            copy.Dispose();
            throw;
        }
    }

    private static IShorokooTensorValue CopySequence(
        IShorokooBackend target, IShorokooTensorValue value)
    {
        var count = value.GetValueCount();
        var copies = new List<IShorokooTensorValue>(count);
        try
        {
            for (int i = 0; i < count; i++)
            {
                // GetValue hands back a value of its own, which is this method's to release
                // whether or not the copy of it succeeds.
                using var element = value.GetValue(i);
                copies.Add(CopyTo(target, element));
            }
        }
        catch
        {
            foreach (var copy in copies) copy.Dispose();
            throw;
        }
        // Outside the catch on purpose: CreateSequence takes the copies over, and releases them
        // itself if it cannot. Inside, a failure there would free each of them twice.
        return target.CreateSequence(copies);
    }
}
