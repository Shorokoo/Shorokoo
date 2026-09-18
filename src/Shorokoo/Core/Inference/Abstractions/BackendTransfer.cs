namespace Shorokoo.Core.Inference.Abstractions;

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
    /// storage of its own. Tensors are copied through their raw bytes, string tensors through
    /// their elements, and a sequence element by element.
    /// </summary>
    /// <exception cref="ArgumentNullException">Either argument is null.</exception>
    /// <exception cref="InvalidOperationException"><paramref name="value"/> lives in an
    /// execution provider's own memory, or is neither a tensor nor a sequence.</exception>
    public static IShorokooTensorValue CopyTo(
        IShorokooInferenceBackend target, IShorokooTensorValue value)
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
        IShorokooInferenceBackend target, IShorokooTensorValue value)
    {
        if (!value.IsHostAccessible)
            throw new InvalidOperationException(
                $"This tensor ({string.Join('x', value.Shape)}:{value.ElementType}) lives in its "
                + "own backend's device memory, so it cannot be handed to " +
                $"{target.Description}. Only a host-resident value can cross between backends "
                + "that do not share a native runtime, so bring it back to the host on the backend "
                + "that owns it first -- ResidentTrainingRun.StepToCheckpoint is what does that "
                + "for a resident training run.");

        if (value.ElementType == ShorokooTensorElementType.String)
            return target.CreateStringTensor(value.GetStringTensorData(), value.Shape);

        var copy = target.CreateTensorFromRawBytes(
            value.ElementType, value.GetTensorDataAsSpan<byte>().ToArray(), value.Shape);
        // Taking the span is the source's last read, so without this the JIT may retire it before
        // ToArray has copied out of the buffer it points at (Shorokoo/Shorokoo#178).
        GC.KeepAlive(value);
        return copy;
    }

    private static IShorokooTensorValue CopySequence(
        IShorokooInferenceBackend target, IShorokooTensorValue value)
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
