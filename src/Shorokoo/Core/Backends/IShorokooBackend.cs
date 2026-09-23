namespace Shorokoo.Core.Backends;

// Implemented once per platform DLL. The platform DLL's identity (WinCPU /
// WinGPU / LinuxCPU / LinuxGPU) determines the EP; there is no EP parameter here.
// What it chose is reported by Description, so a caller need not reflect on the
// backend's assembly name to learn which device its sessions run on.
public interface IShorokooBackend
{
    // What this backend is: its assembly, its device, and the CUDA device it
    // allocates on. Every session it creates runs there.
    BackendDescription Description { get; }

    // Where this backend's tensors live. Derived from Description by default, which is right for
    // every backend that allocates on the device it computes on -- i.e. all of them so far -- so
    // an existing backend need not implement it. The same space is necessary for one backend to
    // read another's allocation in place, and not sufficient: see RuntimeIdentity and CanAddress.
    MemorySpace MemorySpace => Description.Device switch
    {
        ComputeDevice.Cuda => MemorySpace.Cuda(Description.CudaDeviceId ?? 0),
        ComputeDevice.Cpu => MemorySpace.Host,
        // A backend on some other execution provider -- DirectML, ROCm, CoreML -- allocates
        // somewhere this has no name for, and saying "host" would be a guess with teeth: a device
        // value would report IsHost, so it would be read in place by any host backend at all and
        // the accessors would dereference a device address as a host one. Unknown is refused
        // cleanly instead, which is the honest answer until such a backend names its own space.
        _ => MemorySpace.UnknownDevice,
    };

    // The runtime this backend's allocations belong to, as an object compared by reference: two
    // backends answer the same object exactly when an allocation one of them makes is one the
    // other's sessions can be handed as it stands. It is the second half of a tensor's
    // MemoryLocation, recorded from the allocating backend when the tensor is made.
    //
    // The default is the backend itself, which is the answer that is never wrong: a backend can
    // always read what it allocated, and a backend that says nothing about sharing its runtime
    // shares it with nobody. A backend over a runtime that several backends can load together --
    // ONNX Runtime serving a CPU and a CUDA backend from one native -- overrides this with
    // something every such backend shares.
    object RuntimeIdentity => this;

    // Whether this backend's sessions can read memory at `location` as it stands, without a copy:
    // the question To(context) asks of the context's backend before deciding between handing the
    // tensor over and copying it. The answer is "same device and same runtime" -- and the
    // framework's own managed host memory, which every backend on the host reads, counts as every
    // host backend's runtime. It is asked of the target backend rather than decided by the core,
    // because only the backend knows what it can address.
    //
    // A location whose space is unknown is never addressable: two such allocations compare equal
    // as spaces without being anywhere in particular, so sharing one would be a guess.
    bool CanAddress(MemoryLocation location)
        => location.Space.IsKnown
           && location.Space == MemorySpace
           && (location.IsManaged || ReferenceEquals(location.Runtime, RuntimeIdentity));

    // Releases a value this backend allocated. Every release of a tensor's memory comes here, to
    // the backend that made it, whichever contexts the tensor was attached to -- or none -- so a
    // backend that has something to do when its memory comes back has one place to do it. The
    // default disposes the value, which is what releasing one has always meant.
    void Release(IShorokooTensorValue value)
    {
        ArgumentNullException.ThrowIfNull(value);
        value.Dispose();
    }

    // deviceMemory configures the arena this one session allocates in. It is a parameter, not
    // process state, because that is what ORT's own shape is: each session gets its own arena,
    // built from the values read here and kept for the session's life.
    IShorokooSession CreateSession(
        ReadOnlyMemory<byte> modelBytes,
        ShorokooGraphOptimization graphOptimization,
        ShorokooLogSeverity logSeverity,
        DeviceMemorySettings deviceMemory);

    // The same session, told what the caller wants recorded about it -- today, whether it keeps a
    // per-node record of which execution provider ran what. Like deviceMemory this is read while
    // the session is built and kept for its life, so it is a parameter rather than process state.
    //
    // The default drops the diagnostics and builds the ordinary session, so a backend outside this
    // repository keeps compiling. That is not papering over: a backend that does not implement
    // this records nothing, and a session that records nothing is exactly what
    // IShorokooSession.ReadNodePlacement's own default answers -- null.
    IShorokooSession CreateSession(
        ReadOnlyMemory<byte> modelBytes,
        ShorokooGraphOptimization graphOptimization,
        ShorokooLogSeverity logSeverity,
        DeviceMemorySettings deviceMemory,
        DiagnosticSettings diagnostics)
        => CreateSession(modelBytes, graphOptimization, logSeverity, deviceMemory);

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

    // The same tensor, told the arena settings of the context the result will belong to. This is
    // the whole of what makes a tensor moved onto a card answer to a budget: a device allocation
    // comes out of an arena, an arena is built with these values and keeps them for life, so the
    // only way to bound one is to say which arena to take it from before it is taken. Everything
    // in this repository that places a tensor in a backend's own memory calls this one.
    //
    // The settings named here are those of the context the copy is being made for -- the target of
    // the CopyTo or the To. That settles which arena the allocation comes out of for the tensor's
    // life: handing a tensor to a second context that can already address it copies nothing, so it
    // cannot re-home the allocation either.
    //
    // The default drops the settings and asks the member above, which is what a backend written
    // before this member existed implements -- so such a backend keeps compiling AND keeps being
    // asked, byte for byte as it was. That is honest rather than papering over: it never had a way
    // to honour a budget here, and nothing is lost by saying so.
    IShorokooTensorValue CreateTensorInBackendMemory(
        ShorokooTensorElementType elementType,
        byte[] data,
        long[] shape,
        DeviceMemorySettings deviceMemory)
        => CreateTensorInBackendMemory(elementType, data, shape);

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

    // The same tensor, under the arena settings of the context it will belong to, for the same
    // reason the copying constructor above takes them: a tensor a context allocates for itself is
    // as much of that context's device footprint as one it was handed, and ComputeContext's own
    // AllocateUninitialized is the caller.
    //
    // Its default asks the member above, dropping the settings, so a backend that overrode that one
    // to stop paying the fill goes on not paying it. A backend that means to honour a budget
    // overrides this one; overriding only the budgeted copy above does not reach here, because the
    // two are separate allocations and only the backend knows whether its own uninitialized path
    // goes through the other.
    IShorokooTensorValue CreateUninitializedTensorInBackendMemory(
        ShorokooTensorElementType elementType,
        long[] shape,
        DeviceMemorySettings deviceMemory)
        => CreateUninitializedTensorInBackendMemory(elementType, shape);

    // The arena that tensors placed in this backend's memory under deviceMemory come out of, as
    // its runtime reports it -- or null on a backend that reports none, and on one that has been
    // asked for no such tensor yet. It is the transfer half of what CompiledGraph's own arena
    // figures are for a session: between them a context's whole device footprint can be read
    // rather than guessed at, which a budget that bounds only the sessions could never be.
    //
    // Shared with every context carrying the same settings on the same device, since that is
    // exactly what shares the arena.
    ArenaStatistics? ReadTransferArenaStatistics(DeviceMemorySettings deviceMemory) => null;
}
