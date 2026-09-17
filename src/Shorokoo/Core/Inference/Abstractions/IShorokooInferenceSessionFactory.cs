namespace Shorokoo.Core.Inference.Abstractions;

// Implemented once per platform DLL. The platform DLL's identity (WinCPU /
// WinGPU / LinuxCPU / LinuxGPU) determines the EP; there is no EP parameter here.
// What it chose is reported by Description, so a caller need not reflect on the
// factory's assembly name to learn which device its sessions run on.
public interface IShorokooInferenceSessionFactory
{
    // The backend this factory is: its assembly, its device, and the CUDA device it
    // allocates on. Every session it creates runs there.
    BackendDescription Description { get; }

    // Where this backend's tensors live. Derived from Description by default, which is right for
    // every backend that allocates on the device it computes on -- i.e. all of them so far -- so
    // an existing backend need not implement it. Two backends reporting the same space can hand
    // tensors to each other without copying; see MemorySpace.
    MemorySpace MemorySpace => Description.Device switch
    {
        ComputeDevice.Cuda => MemorySpace.Cuda(Description.CudaDeviceId ?? 0),
        _ => MemorySpace.Host,
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
}
