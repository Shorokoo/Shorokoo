namespace Shorokoo.Core.Inference.Abstractions;

// Implemented once per platform DLL. The platform DLL's identity (WinCPU /
// WinGPU / LinuxCPU / LinuxGPU) determines the EP; there is no EP parameter here.
// What it chose is reported by Description, so a caller need not reflect on the
// backend's assembly name to learn which device its sessions run on.
public interface IShorokooInferenceBackend
{
    // What this backend is: its assembly, its device, and the CUDA device it
    // allocates on. Every session it creates runs there.
    BackendDescription Description { get; }

    // Where this backend's tensors live. Derived from Description by default, which is right for
    // every backend that allocates on the device it computes on -- i.e. all of them so far -- so
    // an existing backend need not implement it. Two backends reporting the same space can hand
    // tensors to each other without copying; see MemorySpace.
    MemorySpace MemorySpace => Description.Device switch
    {
        ComputeDevice.Cuda => MemorySpace.Cuda(Description.CudaDeviceId ?? 0),
        ComputeDevice.Cpu => MemorySpace.Host,
        // A backend on some other execution provider -- DirectML, ROCm, CoreML -- allocates
        // somewhere this has no name for, and saying "host" would be a guess with teeth: a device
        // value would report IsHost, so the transfer code would share it with any context at all
        // and the accessors would dereference a device address as a host one. Unknown is refused
        // cleanly instead, which is the honest answer until such a backend names its own space.
        _ => MemorySpace.UnknownDevice,
    };

    // deviceMemory configures the arena this one session allocates in. It is a parameter, not
    // process state, because that is what ORT's own shape is: each session gets its own arena,
    // built from the values read here and kept for the session's life.
    IShorokooInferenceSession CreateSession(
        ReadOnlyMemory<byte> modelBytes,
        ShorokooGraphOptimization graphOptimization,
        ShorokooLogSeverity logSeverity,
        DeviceMemorySettings deviceMemory);

    IShorokooTensorValue CreateTensor<T>(T[] data, long[] shape) where T : unmanaged;

    IShorokooTensorValue CreateTensorFromRawBytes(
        ShorokooTensorElementType elementType,
        byte[] data,
        long[] shape);

    // String tensors don't fit the raw-bytes path: each element is variable-length
    // UTF-8 and reference-typed, so they get their own constructor.
    IShorokooTensorValue CreateStringTensor(IReadOnlyList<string> data, long[] shape);

    // Takes <paramref name="values"/> over: on success the sequence owns them and the caller must
    // not dispose them, and on failure this method disposes them before it throws. Both halves are
    // load-bearing -- BackendTransfer and the sequence builder both call this outside the catch
    // that would otherwise free the elements, precisely because a failure here has already freed
    // them -- so a backend that throws without disposing leaks every element it was handed.
    IShorokooTensorValue CreateSequence(IReadOnlyList<IShorokooTensorValue> values);

    // This value's contents as host bytes, whatever memory it is in. The default serves a value
    // the host can already read; a backend whose execution provider keeps values in its own memory
    // overrides it with the copy only that backend can make, since the allocation is its runtime's
    // and nothing outside knows how to reach it.
    byte[] CopyTensorToHost(IShorokooTensorValue value)
    {
        ArgumentNullException.ThrowIfNull(value);
        if (!value.IsHostAccessible)
            throw new InvalidOperationException(
                $"{Description} cannot read a tensor back from its execution provider's own "
                + "memory: it does not implement CopyTensorToHost.");
        var bytes = value.GetTensorDataAsSpan<byte>().ToArray();
        // Taking the span is the value's last read, so without this the JIT may retire it before
        // ToArray has copied out of the buffer it points at (Shorokoo/Shorokoo#178).
        GC.KeepAlive(value);
        return bytes;
    }

    // A tensor of this backend holding `data`, allocated where this backend's tensors live -- the
    // mirror of CopyTensorToHost above, and the one call that puts host bytes into MemorySpace.
    // For a host backend that is host memory; for a CUDA one it is the card's own memory, which is
    // where a tensor belonging to a CUDA context is supposed to be.
    //
    // The default builds it wherever CreateTensorFromRawBytes does, which is already the right
    // place for a host backend and for any backend whose values the host can read. A backend that
    // computes in memory of its own overrides it with the allocation only that backend can make,
    // exactly as it overrides CopyTensorToHost -- the two are one pair, and a backend answering
    // one and not the other can bring a tensor home but not send one out.
    //
    // Left to the default on a device backend, a tensor "moved onto the card" is host bytes with a
    // device context's name on them, which the execution provider then copies over on every single
    // run: the per-run copy that giving a tensor a context exists to remove.
    IShorokooTensorValue CreateTensorInBackendMemory(
        ShorokooTensorElementType elementType,
        byte[] data,
        long[] shape)
        => CreateTensorFromRawBytes(elementType, data, shape);

    // The same tensor as CreateTensorInBackendMemory builds -- same element type, same shape, same
    // memory -- with nothing put into it: the buffer holds whatever was there, and whoever asked
    // for it writes the contents. It is for a producer that fills a tensor element by element
    // rather than copying one it is already holding, which is the case where the managed array a
    // copy starts from is a second copy of the whole tensor, live for as long as both are
    // (Shorokoo/Shorokoo#359).
    //
    // The default fills it after all, from a zeroed buffer through the member above, and so buys
    // nothing: it is here because this interface is an ABI, and a member without a body is a
    // backend outside this repository that no longer compiles. A backend that does not override
    // this keeps paying the copy it always paid; overriding it is what stops paying.
    //
    // Sizing that buffer is TensorElementLayout's table -- the same one a backend's own byte-wise
    // constructor reads -- so the element types CreateTensorFromRawBytes turns away are turned
    // away here too, and in the same words. The two paths differing on which types exist would be
    // a worse answer than either.
    IShorokooTensorValue CreateUninitializedTensorInBackendMemory(
        ShorokooTensorElementType elementType,
        long[] shape)
        => CreateTensorInBackendMemory(
            elementType, new byte[TensorElementLayout.ByteCount(elementType, shape)], shape);
}
