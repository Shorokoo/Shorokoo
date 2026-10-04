namespace Shorokoo.Core.Backends;

// A backend: the sessions that run models, the values they read, and the moves between host memory
// and the memory the backend computes in. Implemented once per platform DLL. The platform DLL's
// identity (WinCPU / WinGPU / LinuxCPU / LinuxGPU) determines the EP; there is no EP parameter
// here. What it chose is reported by Description, so a caller need not reflect on the backend's
// assembly name to learn which device its sessions run on.
//
// What every backend provides:
//
// - Its run memory. RunMemoryOf answers, per element type, the memory its sessions read a tensor
//   in and leave one in, and SequenceRunMemory answers the same for a sequence. That memory belongs
//   to this backend: another backend on the same device does not necessarily know how to use it
//   (CanAddress and RuntimeIdentity say which can).
// - Sessions (CreateSession) that read every input in that memory, refuse any value outside it,
//   and leave every output there (see IShorokooSession). A session never moves a value: the
//   framework places each input in the run memory before the run, and a value goes anywhere else
//   only when it is moved there.
// - The moves, each an operation of its own, which the framework calls itself -- to place an input
//   before a run, to put a tensor on a context, and to bring one home when its caller asks.
//   Into this backend's memory: CreateTensorInBackendMemory (host bytes, whole),
//   CreateUninitializedTensorInBackendMemory (an allocation a later move fills) and
//   TryCopyHostToTensorRange (host bytes into part of a tensor). Out of it: CopyTensorToHost (a
//   whole tensor as host bytes) and TryCopyTensorRangeToHost (part of one). StagedUpload and
//   StagedReadBack stream a tensor through the range copies, one bounded buffer at a time. The
//   defaults serve a backend whose memory the host reads; a backend that computes in memory of its
//   own overrides every one of them.
// - Values in its runtime's host memory: CreateTensor and CreateTensorFromRawBytes for tensors,
//   CreateStringTensor for strings and CreateSequence for sequences -- the run memory of strings
//   and sequences unless the backend answers otherwise.
// - Release, the one path by which memory it allocated goes back.
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

    // Whether memory at `location` is this backend's as it stands, needing no copy to be its own:
    // the question To(context) asks of the context's backend before deciding between handing the
    // tensor over and copying it, and a run asks before handing a session a tensor's own value. The
    // answer is "same device and same runtime" -- and the framework's own managed host memory, which
    // every backend on the host reads, counts as every host backend's runtime, so To hands a tensor
    // there over as it is; a run still feeds such a tensor through a copy its backend builds in its
    // run memory, a session being handed runtime values only. It is asked of the target backend
    // rather than decided by the core, because only the backend knows what it can address.
    //
    // A location whose space is unknown is addressable only by the one backend that allocated it:
    // two such allocations compare equal as spaces without being anywhere in particular, so sharing
    // one between backends would be a guess, while a backend always reads what it allocated. That
    // one backend is known here only where the runtime is the backend itself -- the default
    // identity; a runtime several backends share says nothing about which of them made an allocation
    // it names, and the framework, which knows the backend that allocated each tensor, hands such a
    // backend its own allocations there itself, and no other backend's.
    bool CanAddress(MemoryLocation location)
        => location.Space == MemorySpace
           && (location.Space.IsKnown
               ? location.IsManaged || ReferenceEquals(location.Runtime, RuntimeIdentity)
               : ReferenceEquals(location.Runtime, this));

    // Where a run on this backend reads a tensor of `elementType` it is fed, and where it leaves one
    // it produces: the memory every such input is in when the session is handed it -- as it stands,
    // where a tensor is there already, and otherwise through a copy the framework makes there with
    // the moves below -- and the memory every such output comes back in. To(context) answers by it
    // too, so the two agree. The default is this backend's own memory in its own runtime, and the
    // host memory of that runtime for a string tensor, which every runtime so far keeps there
    // whatever its device. The framework's own managed host memory is never a run's: a session is
    // handed runtime values only, so a run on a host backend reads such a tensor through a copy its
    // backend builds (see CanAddress).
    MemoryLocation RunMemoryOf(ShorokooTensorElementType elementType)
        => new(elementType == ShorokooTensorElementType.String ? MemorySpace.Host : MemorySpace, RuntimeIdentity);

    // Where a run on this backend reads a sequence it is fed, as RunMemoryOf answers for a tensor,
    // and where its runs leave the sequences they produce, which the framework records them as in.
    // The default is the host memory of this backend's runtime: ONNX Runtime reads a sequence's
    // elements through the host whatever its provider (see CreateSequence), and leaves a sequence a
    // run produces there.
    MemoryLocation SequenceRunMemory => new(MemorySpace.Host, RuntimeIdentity);

    // Releases a value this backend allocated. Every release the framework makes of a tensor's
    // memory comes here, to the backend that made it, whichever contexts the tensor was attached
    // to -- or none -- so a backend that has something to do when its memory comes back has one
    // place to do it. The default disposes the value, which is what releasing one has always meant.
    //
    // Memory a run consumed is the one exception, since the framework hands it over rather than
    // releasing it: the session it was handed to releases it (IShorokooSession.RunConsuming). That
    // session is the running backend's, and what it is handed is a value in that backend's run
    // memory, so it shares the allocating backend's runtime, but may be another instance of it; and
    // a session that leaves RunConsuming to the interface's default has it disposed without coming
    // here. A backend that counts its releases implements RunConsuming and releases there through
    // this.
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

    // The same session, told which of its outputs it may write into the memory of which of its
    // inputs (output aliasing, see OutputAlias): pairs the model's lowering proved -- nothing reads
    // the input after the output is written, in the model as handed over. A session may then write
    // such an output into the input's memory on a run that consumed that input (see
    // IShorokooSession.RunConsuming), where the two agree in memory, shape and element type.
    //
    // The proof is over the model as handed over, or over that model without the rewrites of the
    // backend's own KernelWorkaroundSet (see there). A backend that rewrites the graph before it runs
    // it -- fusing nodes, and so changing which of them read an input -- binds only the pairs its
    // rewritten graph still proves: OutputAliasProof answers for a serialized model.
    //
    // The default drops the pairs and builds the ordinary session, which aliases nothing: always
    // correct, and exactly what a backend that does not implement this should do.
    IShorokooSession CreateSession(
        ReadOnlyMemory<byte> modelBytes,
        ShorokooGraphOptimization graphOptimization,
        ShorokooLogSeverity logSeverity,
        DeviceMemorySettings deviceMemory,
        DiagnosticSettings diagnostics,
        IReadOnlyList<OutputAlias> outputAliases)
        => CreateSession(modelBytes, graphOptimization, logSeverity, deviceMemory, diagnostics);

    // The same session, told how many threads one run of it may spread an operator over:
    // intraOpThreads of 1 runs each operator on the calling thread alone, for a caller that runs
    // several sessions side by side and would otherwise have every one of them claim every core;
    // 0 leaves the count to the backend. Read while the session is built, like deviceMemory.
    //
    // The default drops the count and builds the ordinary session: a backend with no thread pool
    // of its own to size, or one outside this repository, runs as it always does.
    IShorokooSession CreateSession(
        ReadOnlyMemory<byte> modelBytes,
        ShorokooGraphOptimization graphOptimization,
        ShorokooLogSeverity logSeverity,
        DeviceMemorySettings deviceMemory,
        DiagnosticSettings diagnostics,
        IReadOnlyList<OutputAlias> outputAliases,
        int intraOpThreads)
        => CreateSession(modelBytes, graphOptimization, logSeverity, deviceMemory, diagnostics, outputAliases);

    // The same session, with some of the model's initializers supplied as values already in this
    // backend's memory -- weights loaded straight onto the card, which the model's bytes then need
    // not carry (Shorokoo/Shorokoo#436). The model declares each one as SuppliedInitializers
    // describes, and the session reads the value where it is for as long as it lives, which the
    // caller guarantees.
    //
    // The default refuses any it is given: a backend that cannot use a value in place says so, and
    // the caller then puts the values' bytes into the model instead (SuppliesInitializers).
    IShorokooSession CreateSession(
        ReadOnlyMemory<byte> modelBytes,
        ShorokooGraphOptimization graphOptimization,
        ShorokooLogSeverity logSeverity,
        DeviceMemorySettings deviceMemory,
        DiagnosticSettings diagnostics,
        IReadOnlyList<OutputAlias> outputAliases,
        int intraOpThreads,
        IReadOnlyList<SuppliedInitializer> suppliedInitializers)
    {
        ArgumentNullException.ThrowIfNull(suppliedInitializers);
        if (suppliedInitializers.Count > 0)
            throw new NotSupportedException(
                $"{Description} cannot take a model's initializers as values it already holds.");
        return CreateSession(modelBytes, graphOptimization, logSeverity, deviceMemory, diagnostics, outputAliases, intraOpThreads);
    }

    // The same session, told the floating-point precision it computes in: whether a CUDA card may
    // compute float32 products, convolutions and recurrent layers in TensorFloat-32
    // (PrecisionSettings.AllowTensorFloat32). Read while the session is built, like deviceMemory.
    // Every other overload builds a session in PrecisionSettings.Default -- float32 in full float32
    // precision -- so a session computes in anything less only where this was asked for it.
    //
    // The default drops the setting and builds the ordinary session. That is a backend computing in
    // full precision where it was allowed less, which is always within what was asked: the setting
    // allows TensorFloat-32 and never requires it.
    IShorokooSession CreateSession(
        ReadOnlyMemory<byte> modelBytes,
        ShorokooGraphOptimization graphOptimization,
        ShorokooLogSeverity logSeverity,
        DeviceMemorySettings deviceMemory,
        DiagnosticSettings diagnostics,
        IReadOnlyList<OutputAlias> outputAliases,
        int intraOpThreads,
        IReadOnlyList<SuppliedInitializer> suppliedInitializers,
        PrecisionSettings precision)
    {
        ArgumentNullException.ThrowIfNull(precision);
        return CreateSession(
            modelBytes, graphOptimization, logSeverity, deviceMemory, diagnostics, outputAliases, intraOpThreads,
            suppliedInitializers);
    }

    // Whether CreateSession takes supplied initializers. A decorator forwards it.
    bool SuppliesInitializers => false;

    // Whether this backend's sessions run a training step handed over in `format` (one of
    // TrainingFormats). Every backend runs TrainingFormats.Onnx -- a step whose gradient Shorokoo
    // has already written out as ordinary operators -- so that is the default and the only answer
    // a backend that computes no gradients of its own should give. A backend that differentiates
    // itself also accepts TrainingFormats.OnnxAutoGrad, whose one ai.shorokoo.training::AutoGrad
    // node it must then run. A rig asked to leave the gradient to a backend that does not accept
    // the format refuses at build, before anything is composed.
    //
    // A decorator forwards this, as it forwards every member with a default body: answering from
    // the default would say "no" for a backend that says "yes".
    bool AcceptsTrainingFormat(string format) => format == TrainingFormats.Onnx;

    // The kernel workarounds this backend's sessions are built with: one of KernelWorkaroundSets,
    // or null for none. The set names rewrites of operator calls this backend's kernels compute
    // otherwise than the ONNX spec says, each into an equivalent call they compute as the spec
    // does. They are applied to the model built for this backend's sessions only; the graph, an
    // exported file, generated C# and .srk keep every operator as written.
    //
    // The output-alias pairs a session is built with (see CreateSession) may be proved over the
    // model built without the set, where the model built with it proves fewer: a rewrite that
    // decides at run time reads its operands inside an If branch, which the proof refuses, though
    // the runtime folds most such Ifs away. A backend naming a set therefore binds only the pairs
    // the graph it runs proves again, as it does for a rewrite of its own.
    //
    // A decorator forwards this, as it forwards every member with a default body.
    string? KernelWorkaroundSet => null;

    // The most a run of `model` holds at once beyond its inputs -- with the outputs of
    // `outputAliases` written into the inputs they are paired with, computing in `precision` -- as
    // this backend lays a run's values out, or null where this backend has no such model. `model`'s
    // inputs state their shapes in full. The training rig's memory-aware pass judges the steps it weighs by this where it is
    // not null, so that a step it hands a backend holds least on that backend, rather than in the
    // pass's own model of a run, which charges the graph it hands over in the order it predicts the
    // backend takes, where the backend may rewrite that graph and run it in another.
    //
    // A decorator forwards this, as it forwards every member with a default body.
    internal long? ModelledRunPeak(Shorokoo.Core.Factory.IR.ModelProto model, IReadOnlyList<OutputAlias> outputAliases,
        PrecisionSettings precision) => null;

    // Whether ModelledRunPeak answers in about the time the memory-aware pass takes to evaluate a
    // graph itself -- building nothing of this backend's own, a session say -- so that the pass may
    // weigh the many candidates its search leaves on a plateau by it, rather than only the steps
    // each strategy takes.
    //
    // A decorator forwards this, as it forwards every member with a default body.
    internal bool ModelsARunQuickly => false;

    // How this backend lays a run's values out in memory, as the training rig's memory-aware pass
    // charges them while it searches (see ModelledRunPeak for how it then judges). ONNX Runtime's
    // allocation plan is the default; ONNX Runtime on the host adds what its CPU kernels hold beside
    // their outputs, and a backend running a model's translation lays them out as the translation
    // does.
    //
    // A decorator forwards this, as it forwards every member with a default body.
    internal Shorokoo.Core.AutoDiffCheckpointing.RunLayout RunLayout => Shorokoo.Core.AutoDiffCheckpointing.RunLayout.OnnxRuntime;

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

    // This value's contents as host bytes, whatever memory it is in: the move out of this backend's
    // memory, which the framework makes to bring a tensor home, and to read one that a run on
    // another backend is to be fed. The default serves a value
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

    // Copies `destination.Length` bytes of this value's contents, starting `byteOffset` bytes in,
    // into `destination` -- a piece of what CopyTensorToHost returns whole, into a buffer the caller
    // owns and reuses. It is how a tensor is written out of the provider's own memory without its
    // whole contents ever sitting in host memory at once: a save moves it through one bounded
    // buffer, piece by piece.
    //
    // Returns false, having copied nothing, where this backend cannot copy part of a value; the
    // caller then falls back to CopyTensorToHost. The default serves a value the host can read
    // itself, of any size, out of the piece of its buffer the value addresses (HostPiece), and a
    // backend whose provider keeps values in its own memory overrides it where it can reach an
    // arbitrary range of the allocation.
    bool TryCopyTensorRangeToHost(IShorokooTensorValue value, long byteOffset, Span<byte> destination)
    {
        ArgumentNullException.ThrowIfNull(value);
        if (!value.IsHostAccessible) return false;
        value.HostPiece(byteOffset, destination.Length).CopyTo(destination);
        // The piece is the value's last read (Shorokoo/Shorokoo#178).
        GC.KeepAlive(value);
        return true;
    }

    // Copies `source` into this value's contents, starting `byteOffset` bytes in -- the mirror of
    // TryCopyTensorRangeToHost, and how a tensor is loaded into the provider's own memory without
    // its whole contents ever sitting in host memory at once: a load moves it from the file through
    // one bounded buffer, piece by piece, into a tensor allocated where it is to live.
    //
    // Returns false, having copied nothing, where this backend cannot write part of a value; the
    // caller then falls back to CreateTensorInBackendMemory over the whole contents. The default
    // serves a value the host can write itself, of any size, into the piece of its buffer the value
    // addresses (HostPiece).
    bool TryCopyHostToTensorRange(IShorokooTensorValue value, long byteOffset, ReadOnlySpan<byte> source)
    {
        ArgumentNullException.ThrowIfNull(value);
        if (!value.IsHostAccessible) return false;
        source.CopyTo(value.HostPiece(byteOffset, source.Length));
        // The piece is the value's last read (Shorokoo/Shorokoo#178).
        GC.KeepAlive(value);
        return true;
    }

    // A tensor of this backend holding `data`, allocated where this backend's tensors live -- the
    // mirror of CopyTensorToHost above, and the one call that puts host bytes into MemorySpace.
    // For a host backend that is host memory; for a CUDA one it is the card's own memory, which is
    // where a tensor belonging to a CUDA context is supposed to be. It is the move the framework
    // makes to place a run's input in the run memory, wherever the input was.
    //
    // The default builds it wherever CreateTensorFromRawBytes does, which is already the right
    // place for a host backend and for any backend whose values the host can read. A backend that
    // computes in memory of its own overrides it with the allocation only that backend can make,
    // exactly as it overrides CopyTensorToHost -- the two are one pair, and a backend answering
    // one and not the other can bring a tensor home but not send one out.
    //
    // Left to the default on a device backend, a tensor "moved onto the card" is host bytes with a
    // device context's name on them, which that backend's own sessions refuse to be fed.
    //
    // It takes no device-memory settings, and needs none: a compute context's budget is kept by the
    // context, which counts the tensors attached to it and refuses a copy that would take it past
    // its budget before this is ever asked for the memory. A backend allocates; it does not decide
    // whose budget an allocation is on.
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
