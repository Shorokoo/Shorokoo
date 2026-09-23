using Microsoft.ML.OnnxRuntime;
using Shorokoo.Core.Backends;
using TensorElementType = Microsoft.ML.OnnxRuntime.Tensors.TensorElementType;

namespace Shorokoo.OnnxRuntime;

internal sealed class OrtSession : IShorokooSession
{
    private readonly InferenceSession _session;
    private readonly int? _cudaDeviceId;
    private readonly IShorokooBackend _backend;

    // What one probe of the session's outputs answers: ORT's memory info for this session's own
    // (non-host) output memory, or null when it produces everything on the host, and where the
    // outputs land as a whole. One Lazy for both, because one native call answers both and two
    // would build two OrtMemoryInfo collections for one answer. Held in a field, not a local:
    // OrtMemoryInfo owns a native handle and the binding below takes it as a bare pointer. Probed
    // on first ask, because most sessions never retain anything and a session is a common object
    // here -- parameter initialization and every eager Eval build one.
    private readonly Lazy<OutputMemory> _outputMemory;

    // This session's own allocator for the arena the figures come from, and the memory info naming
    // it. Both in fields for the reason the one above is: they own native handles that go over as
    // bare pointers. Built on first ask, because no session needs them unless something asked for
    // run statistics.
    private readonly Lazy<OrtAllocator?> _arenaAllocator;
    private OrtMemoryInfo? _ownedArenaMemoryInfo;

    // The same pair for the pinned host arena a CUDA session stages its crossings through. A
    // second allocator rather than a second reading of the first: they are different arenas with
    // different figures, and a session with no CUDA provider has no second one at all.
    private readonly Lazy<OrtAllocator?> _pinnedAllocator;
    private OrtMemoryInfo? _ownedPinnedMemoryInfo;

    // The folder ORT writes this session's profile into, or null when it was not built to record
    // one -- which is the default. Deleted with the session.
    private readonly string? _profileDirectory;
    private readonly object _profileGate = new();
    private NodePlacement? _nodePlacement;
    private bool _profilingEnded;

    // The outputs this session may write into the memory of the input each is paired with, on a run
    // that consumed that input: the pairs its own graph proved (see OrtBackend.CreateSession), each
    // with the shape and element type ORT inferred for the output. A pair whose output shape ORT
    // could not settle when the session was built is not here, since nothing could be bound to it
    // without knowing the output would fit -- a symbolic dim is only known once the run is under
    // way, and a buffer bound to the wrong shape fails the run.
    private readonly IReadOnlyList<AliasSlot> _aliases;

    /// <summary>An output this session may write into an input's memory, and what the input's value
    /// has to be for that to happen.</summary>
    private sealed record AliasSlot(string Output, string Input, long[] Shape, TensorElementType ElementType);

    public OrtSession(
        InferenceSession session, int? cudaDeviceId, IShorokooBackend backend)
        : this(session, cudaDeviceId, backend, profileDirectory: null, outputAliases: [])
    {
    }

    public OrtSession(
        InferenceSession session,
        int? cudaDeviceId,
        IShorokooBackend backend,
        string? profileDirectory,
        IReadOnlyList<OutputAlias> outputAliases)
    {
        _session = session;
        _cudaDeviceId = cudaDeviceId;
        _backend = backend;
        _profileDirectory = profileDirectory;
        _aliases = Slots(session, outputAliases);
        // Nothing to clean up, so nothing to finalize -- every untraced session would otherwise
        // join the finalization queue to run an early return.
        if (profileDirectory is null) GC.SuppressFinalize(this);
        _outputMemory = new Lazy<OutputMemory>(() => DiscoverOutputMemory(session));
        _arenaAllocator = new Lazy<OrtAllocator?>(CreateArenaAllocator);
        _pinnedAllocator = new Lazy<OrtAllocator?>(CreatePinnedAllocator);
    }

    /// <summary>
    /// The pairs of <paramref name="outputAliases"/> this session can bind: the output is one of its
    /// tensors, and not a string one; the input is one of its inputs; and ORT settled the output's
    /// shape in full when it built the session.
    /// </summary>
    private static List<AliasSlot> Slots(InferenceSession session, IReadOnlyList<OutputAlias> outputAliases)
    {
        var slots = new List<AliasSlot>(outputAliases.Count);
        if (outputAliases.Count == 0) return slots;
        var inputs = new HashSet<string>(session.InputNames, StringComparer.Ordinal);
        var outputs = session.OutputMetadata;
        foreach (var alias in outputAliases)
        {
            if (!inputs.Contains(alias.Input) || !outputs.TryGetValue(alias.Output, out var output)) continue;
            // A string tensor's elements are objects rather than bytes in a buffer of its own, and
            // nothing here proves writing one over another sound, so it is never bound.
            if (!output.IsTensor || output.ElementDataType == TensorElementType.String) continue;
            if (output.Dimensions.Any(d => d <= 0)) continue;
            slots.Add(new AliasSlot(
                alias.Output, alias.Input, [.. output.Dimensions.Select(d => (long)d)], output.ElementDataType));
        }
        return slots;
    }

    /// <summary>
    /// The ORT value behind <paramref name="value"/>, which this session may feed.
    ///
    /// <para>An <see cref="OrtTensorValue"/> <i>of this assembly</i> is one of ours already. That
    /// test is exact rather than approximate: a backend loaded by
    /// <see cref="IsolatedBackend"/> gets a private copy of this assembly, so its
    /// <c>OrtTensorValue</c> is a different type from this one — and its handles point into a
    /// native runtime this session knows nothing about, which is precisely when feeding them
    /// would be a wild pointer rather than a mistake ORT could catch. Two backends over one
    /// loaded runtime share this assembly and so share the type, which is right too: their values
    /// are interchangeable and ORT moves them to the device itself.</para>
    ///
    /// <para>Anything else is rebuilt here, by this backend, from the source's contents. The copy
    /// belongs to nobody, so it is added to <paramref name="borrowed"/> for the caller to release
    /// once the run has read it. That list is created only when there is something to put in it:
    /// every feed of an ordinary single-backend run takes the first branch, and a training loop
    /// calls this once per input per step.</para>
    /// </summary>
    private OrtValue Unwrap(IShorokooTensorValue value, ref List<IShorokooTensorValue>? borrowed)
    {
        if (value is OrtTensorValue own) return own.Inner;
        var copy = BackendTransfer.CopyTo(_backend, value);
        (borrowed ??= []).Add(copy);
        return ((OrtTensorValue)copy).Inner;
    }

    public IReadOnlyList<string> InputNames => _session.InputNames;
    public IReadOnlyList<string> OutputNames => _session.OutputNames;

    public bool HasDeviceMemory => _outputMemory.Value.DeviceMemoryInfo is not null;

    /// <summary>
    /// Runs the session with <paramref name="consumed"/> handed over: each is this backend's from
    /// here, on every path, and is released through it in the <c>finally</c> below — after the
    /// native run, which ONNX Runtime does not let go of its inputs before, and before this returns
    /// the outputs or rethrows a failure. Nothing the caller holds points into one of them any
    /// more: the run's outputs are values of their own.
    ///
    /// <para>ONNX Runtime keeps every input until the run ends — its memory planner gives each
    /// feed an extra use so a caller can read it after <c>Run</c> returns — so there is no earlier
    /// point at which a consumed input could go.</para>
    /// </summary>
    public IReadOnlyList<IShorokooTensorValue> RunConsuming(
        IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
        IReadOnlyCollection<IShorokooTensorValue> consumed,
        IReadOnlyList<string> outputNames,
        IReadOnlySet<string> retainedOutputNames,
        RunSettings runSettings)
    {
        ArgumentNullException.ThrowIfNull(consumed);
        try
        {
            return retainedOutputNames.Count == 0
                ? Run(inputs, outputNames, runSettings)
                : RunRetainingOutputs(inputs, outputNames, retainedOutputNames, runSettings);
        }
        finally
        {
            // Through the backend, which is the one release path for memory it allocated. Its
            // release is a disposal, which does not throw, so none of them can be skipped.
            foreach (var value in consumed) _backend.Release(value);
        }
    }

    /// <summary>
    /// <see cref="RunConsuming(IReadOnlyDictionary{string, IShorokooTensorValue}, IReadOnlyCollection{IShorokooTensorValue}, IReadOnlyList{string}, IReadOnlySet{string}, RunSettings)"/>,
    /// writing each output this session was built to alias into the memory of the consumed value
    /// its input was fed, wherever that can be done (see <see cref="OutputsIntoConsumed"/>), and
    /// saying which it did in <paramref name="aliasedInputs"/>.
    ///
    /// <para>An aliased output is bound to the consumed value itself, so the node that produces it
    /// writes straight into that memory rather than into a block of the arena — which is the whole
    /// saving, since ONNX Runtime holds the value until the run ends either way. What comes back is
    /// a value of its own over the same memory: ORT counts the references to a buffer, so releasing
    /// the consumed value below, as every run does, leaves the output whole.</para>
    /// </summary>
    public IReadOnlyList<IShorokooTensorValue> RunConsuming(
        IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
        IReadOnlyCollection<IShorokooTensorValue> consumed,
        IReadOnlyList<string> outputNames,
        IReadOnlySet<string> retainedOutputNames,
        RunSettings runSettings,
        out IReadOnlyList<string?> aliasedInputs)
    {
        ArgumentNullException.ThrowIfNull(consumed);
        var aliased = new string?[outputNames.Count];
        aliasedInputs = aliased;
        try
        {
            if (OutputsIntoConsumed(inputs, consumed, outputNames, retainedOutputNames) is not { } into)
                return retainedOutputNames.Count == 0
                    ? Run(inputs, outputNames, runSettings)
                    : RunRetainingOutputs(inputs, outputNames, retainedOutputNames, runSettings);

            var results = RunBound(inputs, outputNames, retainedOutputNames, into, runSettings);
            for (int i = 0; i < outputNames.Count; i++)
                if (into.TryGetValue(outputNames[i], out var target)) aliased[i] = target.Input;
            return results;
        }
        finally
        {
            foreach (var value in consumed) _backend.Release(value);
        }
    }

    /// <summary>A consumed value an output is written into, and the input it was fed as.</summary>
    private readonly record struct AliasTarget(string Input, OrtTensorValue Value);

    /// <summary>
    /// The outputs this run writes into the memory of a value it consumed, by output name, or null
    /// when there are none. A pair this session was built with (<see cref="_aliases"/>) is bound
    /// only where every one of these holds, and the output is produced as usual otherwise:
    /// <list type="bullet">
    /// <item>the run consumed the value its input was fed — memory it was only lent is the
    /// caller's, and writing into it would change a tensor the caller still reads;</item>
    /// <item>no other input is fed the same value, since the proof was about this input alone;</item>
    /// <item>it is a value of this runtime, of the output's element type and shape;</item>
    /// <item>it is in the memory the output is produced in: this session's card for an output the
    /// run retains there, and the host for any other.</item>
    /// </list>
    /// </summary>
    private Dictionary<string, AliasTarget>? OutputsIntoConsumed(
        IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
        IReadOnlyCollection<IShorokooTensorValue> consumed,
        IReadOnlyList<string> outputNames,
        IReadOnlySet<string> retainedOutputNames)
    {
        if (_aliases.Count == 0 || consumed.Count == 0) return null;
        var handed = new HashSet<IShorokooTensorValue>(consumed, ReferenceEqualityComparer.Instance);
        var fedAs = new Dictionary<IShorokooTensorValue, int>(ReferenceEqualityComparer.Instance);
        foreach (var value in inputs.Values) fedAs[value] = fedAs.GetValueOrDefault(value) + 1;
        var requested = new HashSet<string>(outputNames, StringComparer.Ordinal);
        var deviceMemory = _outputMemory.Value.DeviceMemoryInfo;

        Dictionary<string, AliasTarget>? into = null;
        foreach (var slot in _aliases)
        {
            if (!requested.Contains(slot.Output) || !inputs.TryGetValue(slot.Input, out var value)) continue;
            if (!handed.Contains(value) || fedAs[value] != 1 || value is not OrtTensorValue own) continue;
            var onDevice = deviceMemory is not null && retainedOutputNames.Contains(slot.Output);
            if (!Fits(own, slot, onDevice ? deviceMemory : null)) continue;
            (into ??= new Dictionary<string, AliasTarget>(StringComparer.Ordinal))[slot.Output] =
                new AliasTarget(slot.Input, own);
        }
        return into;
    }

    /// <summary>
    /// Whether <paramref name="value"/> can take <paramref name="slot"/>'s output: a tensor of its
    /// element type and shape, in <paramref name="deviceMemory"/>'s device memory where the output is
    /// produced there, and in host memory where it is null.
    /// </summary>
    private static bool Fits(OrtTensorValue value, AliasSlot slot, OrtMemoryInfo? deviceMemory)
    {
        if (value.ValueType != ShorokooOnnxValueType.Tensor) return false;
        if ((int)value.ElementType != (int)slot.ElementType || !value.Shape.AsSpan().SequenceEqual(slot.Shape))
            return false;
        if (deviceMemory is null) return value.IsHostAccessible;
        using var info = value.Inner.GetTensorMemoryInfo();
        // After the reads, for the reason OrtTensorValue.ProbeHostAccessible gives: the info points
        // into the native value rather than owning anything.
        var here = info.Name == deviceMemory.Name && info.Id == deviceMemory.Id;
        GC.KeepAlive(value);
        return here;
    }

    public IReadOnlyList<IShorokooTensorValue> Run(
        IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
        IReadOnlyList<string> outputNames,
        RunSettings runSettings)
    {
        ArgumentNullException.ThrowIfNull(runSettings);
        var abortToken = runSettings.CancellationToken;
        abortToken.ThrowIfCancellationRequested();
        List<IShorokooTensorValue>? borrowed = null;
        try
        {
            var ortInputs = new Dictionary<string, OrtValue>(inputs.Count);
            foreach (var (k, v) in inputs)
                ortInputs[k] = Unwrap(v, ref borrowed);

            using var runOptions = new RunOptions();
            ConfigureRun(runOptions, runSettings);
            using var abort = AbortWhenCancelled(runOptions, abortToken);

            IDisposableReadOnlyCollection<OrtValue> results;
            try
            {
                results = _session.Run(runOptions, ortInputs, outputNames);
            }
            catch (OnnxRuntimeException cause) when (WasStopped(cause, abortToken))
            {
                throw Aborted(cause, abortToken);
            }
            finally
            {
                // ORT snapshots each input's handle into an IntPtr[] and keeps no reference to the OrtValue
                // wrappers, so from that point on `ortInputs` is their only root -- and the JIT retires it
                // at the call. OrtValue has an ordinary finalizer that calls OrtReleaseValue, so a GC inside
                // the native Run would free the feeds while it is still reading them. `_session` is rooted
                // by this instance and `runOptions` by the using; the inputs need this. In the finally
                // rather than after the call, so it is reached however the run ends -- a terminated one
                // is still reading those buffers right up to the moment it gives up.
                GC.KeepAlive(ortInputs);
            }

            // `results` is deliberately not disposed. It is a container whose Dispose would dispose
            // the values inside it, and those are exactly what this returns: each one is handed to an
            // OrtTensorValue, and from there to the TensorData that owns it and releases it when
            // disposed (Shorokoo/Shorokoo#180). The container itself holds nothing else to release.
            var wrapped = new List<IShorokooTensorValue>(results.Count);
            foreach (var r in results) wrapped.Add(new OrtTensorValue(r));
            return wrapped;
        }
        finally
        {
            // Copies made for feeds that came from another backend's runtime. They exist only for
            // the duration of the run: the outputs above are this session's own values, so nothing
            // the caller keeps points into one of these.
            if (borrowed is not null) foreach (var copy in borrowed) copy.Dispose();
        }
    }

    public IReadOnlyList<IShorokooTensorValue> RunRetainingOutputs(
        IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
        IReadOnlyList<string> outputNames,
        IReadOnlySet<string> retainedOutputNames,
        RunSettings runSettings)
    {
        ArgumentNullException.ThrowIfNull(runSettings);
        var abortToken = runSettings.CancellationToken;
        abortToken.ThrowIfCancellationRequested();

        // Nothing to retain, or nowhere to retain it: an unbound Run is the same thing and
        // costs one native call less.
        var deviceMemoryInfo = _outputMemory.Value.DeviceMemoryInfo;
        if (deviceMemoryInfo is null || retainedOutputNames.Count == 0)
            return Run(inputs, outputNames, runSettings);
        return RunBound(inputs, outputNames, retainedOutputNames, into: null, runSettings);
    }

    /// <summary>
    /// Runs the session through an I/O binding: the outputs in <paramref name="into"/> written into
    /// the consumed values named there, those in <paramref name="retainedOutputNames"/> left in this
    /// session's device memory, and every other one fetched back to the host.
    /// </summary>
    private IReadOnlyList<IShorokooTensorValue> RunBound(
        IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
        IReadOnlyList<string> outputNames,
        IReadOnlySet<string> retainedOutputNames,
        Dictionary<string, AliasTarget>? into,
        RunSettings runSettings)
    {
        ArgumentNullException.ThrowIfNull(runSettings);
        var abortToken = runSettings.CancellationToken;
        abortToken.ThrowIfCancellationRequested();
        var deviceMemoryInfo = _outputMemory.Value.DeviceMemoryInfo;

        List<IShorokooTensorValue>? borrowed = null;
        try
        {
            using var binding = _session.CreateIoBinding();
            foreach (var (k, v) in inputs)
                binding.BindInput(k, Unwrap(v, ref borrowed));

            // An output bound to a value is written into that value's memory by the node that
            // produces it -- a consumed input's, for an aliased one. An output bound to a device is
            // allocated there by ORT and left there; one bound to the host allocator is fetched back
            // exactly as an unbound Run fetches it. ORT sizes both of those itself, so a shape it
            // only learns while running is fine; an aliased one was checked to fit before it got
            // here.
            var hostMemoryInfo = OrtMemoryInfo.DefaultInstance;
            foreach (var name in outputNames)
            {
                if (into is not null && into.TryGetValue(name, out var target))
                    binding.BindOutput(name, target.Value.Inner);
                else
                    binding.BindOutputToDevice(
                        name, deviceMemoryInfo is not null && retainedOutputNames.Contains(name)
                            ? deviceMemoryInfo : hostMemoryInfo);
            }

            using var runOptions = new RunOptions();
            ConfigureRun(runOptions, runSettings);
            using var abort = AbortWhenCancelled(runOptions, abortToken);

            IDisposableReadOnlyCollection<OrtValue> results;
            try
            {
                results = _session.RunWithBoundResults(runOptions, binding);
            }
            catch (OnnxRuntimeException cause) when (WasStopped(cause, abortToken))
            {
                throw Aborted(cause, abortToken);
            }
            finally
            {
                // Same rooting hazard as Run: the binding holds the feeds' raw handles, not the managed
                // wrappers, so nothing but `inputs` keeps them alive across the native run. `borrowed`
                // roots any feed that had to be rebuilt here, which `inputs` does not hold.
                GC.KeepAlive(inputs);
                GC.KeepAlive(borrowed);
            }

            // RunWithBoundResults returns the bound outputs in the binding's own order, which is the
            // order they were bound in -- ask it rather than assume, and hand them back in the order
            // the caller named.
            var boundNames = binding.GetOutputNames();
            // Nothing owns these values until each is wrapped and handed to a TensorData, and the
            // collection is deliberately not disposed, so anything that goes wrong between here and the
            // return leaks a device allocation apiece. Establish the shape first, and dispose the lot
            // if it is not what it must be.
            if (boundNames.Length != results.Count || results.Count != outputNames.Count)
            {
                foreach (var value in results) value.Dispose();
                throw new InvalidOperationException(
                    $"The run bound {boundNames.Length} outputs and returned {results.Count} values for "
                    + $"{outputNames.Count} requested names; they must agree one for one.");
            }

            var byName = new Dictionary<string, OrtValue>(results.Count);
            for (int i = 0; i < boundNames.Length; i++)
                byName[boundNames[i]] = results[i];

            var wrapped = new List<IShorokooTensorValue>(outputNames.Count);
            foreach (var name in outputNames)
            {
                // Same reason as the count check above, and the same handling: a name that does not
                // come back is a bad run, not an excuse to drop every device allocation it made.
                if (!byName.TryGetValue(name, out var value))
                {
                    foreach (var orphan in results) orphan.Dispose();
                    throw new InvalidOperationException(
                        $"The run bound no output named '{name}'; it bound "
                        + $"{string.Join(", ", boundNames)}.");
                }
                wrapped.Add(new OrtTensorValue(value));
            }
            return wrapped;
        }
        finally
        {
            if (borrowed is not null) foreach (var copy in borrowed) copy.Dispose();
        }
    }

    /// <summary>What one probe of the session's output memory answers: the memory a retained
    /// output can be bound to, and where the outputs land.</summary>
    private readonly record struct OutputMemory(
        OrtMemoryInfo? DeviceMemoryInfo, SessionOutputPlacement Placement);

    /// <summary>
    /// The memory the session's execution provider produces its outputs in, when that is not host
    /// memory — null when every output lands on the host (a CPU provider), and null too when the
    /// native build does not answer the question, which costs the retention and never correctness
    /// — and, from the same infos, where the outputs land as a whole. The infos ORT reports are
    /// owned by the collection it returns, so this copies the one it keeps rather than outliving
    /// its source.
    ///
    /// <para>The placement is the cheap fallback signal: a session that reports host memory for
    /// some outputs and its own for others has a graph ORT partitioned across the two, and one on
    /// a device backend reporting host memory for all of them ran the whole graph there.</para>
    /// </summary>
    private static OutputMemory DiscoverOutputMemory(InferenceSession session)
    {
        try
        {
            OrtMemoryInfo? device = null;
            var host = 0;
            var onDevice = 0;
            using var infos = session.GetMemoryInfosForOutputs();
            foreach (var info in infos)
            {
                if (OrtTensorValue.IsHostAllocator(info.Name)) { host++; continue; }
                onDevice++;
                device ??= CopyOf(info);
            }

            // The session is a bare argument whose last read is the call above, so without this
            // the JIT may retire it before the native call returns and a GC on any thread runs
            // ~InferenceSession underneath it -- a CompiledGraph is held weakly by its context, so
            // there is no strong reference above this one to fall back on.
            //
            // Measured rather than assumed, and the measurement is worth recording: removing this
            // does NOT reproduce a fault. Four hundred probes of a freshly compiled graph held only
            // in a local, in Release on a card, against a thread collecting and draining
            // finalizers, all came back with the placement and none took the process down -- because
            // the caller reaches this through a Lazy whose factory closure holds the session for as
            // long as the factory runs. That is what roots it today. This line is what makes the
            // rooting the method's own rather than a property of how it happens to be invoked
            // (Shorokoo/Shorokoo#178), which is exactly the kind of thing a refactor takes away
            // silently.
            GC.KeepAlive(session);

            var placement = (host, onDevice) switch
            {
                (0, 0) => SessionOutputPlacement.Unknown,
                (_, 0) => SessionOutputPlacement.Host,
                (0, _) => SessionOutputPlacement.Device,
                _ => SessionOutputPlacement.Mixed,
            };
            return new OutputMemory(device, placement);
        }
        // Catching broadly is the point: the doc above promises a failed probe costs the retention
        // and nothing else, and Lazy caches an escaping exception and rethrows it on every later
        // access -- which would fail every run of this session rather than fall back to the host.
        catch (Exception) { }
        return new OutputMemory(null, SessionOutputPlacement.Unknown);
    }

    /// <summary>An info of our own with the same contents, since the one ORT handed over belongs
    /// to the collection it came in and dies with it.</summary>
    private static OrtMemoryInfo CopyOf(OrtMemoryInfo info)
    {
        return new OrtMemoryInfo(info.Name, info.GetAllocatorType(), info.Id, info.GetMemoryType());
    }

    public SessionOutputPlacement OutputPlacement => _outputMemory.Value.Placement;

    /// <summary>
    /// This session's own allocator for the arena the figures come from: the CUDA device arena on
    /// a GPU backend, the session's CPU arena otherwise. It has to come from the session —
    /// <c>OrtAllocator.DefaultInstance</c> is the plain CPU allocator, which implements no
    /// statistics and answers with nothing at all.
    ///
    /// <para>Null on a build that has no such allocator to give, which is what a CUDA arena asked
    /// for on a session with no CUDA provider is: ORT refuses it by name, and that refusal is a
    /// backend without the figures rather than a failure of the run.</para>
    /// </summary>
    private OrtAllocator? CreateArenaAllocator()
    {
        OrtMemoryInfo? owned = null;
        try
        {
            if (_cudaDeviceId is { } device) owned = CudaArenaMemoryInfo(device);
            var allocator = new OrtAllocator(_session, owned ?? OrtMemoryInfo.DefaultInstance);
            // The field assignment is what roots `owned` across the constructor above: ORT takes
            // the info as a bare IntPtr, so without a read of the local after the call the JIT
            // retires it at the .Handle read and its critical finalizer can free the info while
            // OrtCreateAllocator is still reading it. It is also why it is assigned only here --
            // a refusal above leaves nothing for the disposal to have to skip -- and never
            // DefaultInstance, which is ORT's own shared singleton and not ours to release.
            _ownedArenaMemoryInfo = owned;
            return allocator;
        }
        catch (Exception)
        {
            owned?.Dispose();
            return null;
        }
    }

    /// <summary>ORT's memory info for a CUDA device arena. <c>CudaPinned</c> is the pinned host
    /// arena that host-to-device copies stage through and is a different allocator —
    /// <see cref="CudaPinnedArenaMemoryInfo"/>.</summary>
    private static OrtMemoryInfo CudaArenaMemoryInfo(int deviceId)
    {
        return new OrtMemoryInfo("Cuda", OrtAllocatorType.ArenaAllocator, deviceId, OrtMemType.Default);
    }

    /// <summary>ORT's memory info for the pinned host arena of the same device. The memory type is
    /// what tells it from the device arena above; the name is spelled the way ORT reports it on an
    /// output it serves from there.</summary>
    private static OrtMemoryInfo CudaPinnedArenaMemoryInfo(int deviceId)
    {
        return new OrtMemoryInfo(
            "CudaPinned", OrtAllocatorType.ArenaAllocator, deviceId, OrtMemType.CpuOutput);
    }

    /// <summary>
    /// The pinned host arena of this session's device, or null on a session with no CUDA provider
    /// — which has no such arena rather than an empty one.
    /// </summary>
    private OrtAllocator? CreatePinnedAllocator()
    {
        if (_cudaDeviceId is not { } device) return null;
        OrtMemoryInfo? owned = null;
        try
        {
            owned = CudaPinnedArenaMemoryInfo(device);
            var allocator = new OrtAllocator(_session, owned);
            // The field assignment roots `owned` across the constructor, exactly as above.
            _ownedPinnedMemoryInfo = owned;
            return allocator;
        }
        catch (Exception)
        {
            owned?.Dispose();
            return null;
        }
    }

    public ArenaStatistics? ReadArenaStatistics()
        => _arenaAllocator.Value is { } allocator ? OrtArenaStats.Read(allocator) : null;

    public ArenaStatistics? ReadPinnedArenaStatistics()
        => _pinnedAllocator.Value is { } allocator ? OrtArenaStats.Read(allocator) : null;

    /// <summary>
    /// Which execution provider ran each node, read out of the profile ORT has been writing since
    /// this session was built — or null when it was not built to write one.
    ///
    /// <para>Reading it ends the profiling: that is ORT's own shape, since the events are buffered
    /// and the file is only complete once profiling stops. So the trace covers every run up to
    /// this call, later runs are not in it, and the answer is kept so a second call gets the same
    /// one rather than asking a session that is no longer recording.</para>
    /// </summary>
    public NodePlacement? ReadNodePlacement()
    {
        if (_profileDirectory is null) return null;
        lock (_profileGate)
        {
            if (_profilingEnded) return _nodePlacement;
            _profilingEnded = true;
            try
            {
                _nodePlacement = OrtProfile.Read(_session.EndProfiling());
            }
            catch (Exception) { _nodePlacement = null; }
            return _nodePlacement;
        }
    }

    /// <summary>
    /// Arms <paramref name="runOptions"/> so that cancelling <paramref name="token"/> sets ORT's
    /// terminate flag on it, which is what makes a run abortable at all: ORT reads that flag
    /// before each node, and a run's options are otherwise a local no other thread can reach.
    /// Disarmed by disposing the registration, which is why the caller's <c>using</c> for it sits
    /// <i>after</i> the one for the options: disposal runs in reverse, so the registration is gone
    /// — and a callback already running on the cancelling thread waited out, which
    /// <see cref="CancellationTokenRegistration.Dispose"/> does — before the native handle it
    /// writes to is released.
    ///
    /// <para>The registration also roots <paramref name="runOptions"/> for its own lifetime, since
    /// the token source holds it as the callback's state. That is belt and braces: the caller's
    /// <c>using</c> reads the local in its <c>finally</c>, which already keeps it alive across the
    /// native call.</para>
    ///
    /// <para>A token that can never be cancelled gets no registration at all — the default
    /// <see cref="CancellationTokenRegistration"/> disposes to nothing — so an ordinary run pays
    /// nothing for this.</para>
    /// </summary>
    private static CancellationTokenRegistration AbortWhenCancelled(
        RunOptions runOptions, CancellationToken token)
        => token.CanBeCanceled
            ? token.Register(static state => ((RunOptions)state!).Terminate = true, runOptions)
            : default;

    /// <summary>
    /// Whether <paramref name="cause"/> is ORT reporting the run <paramref name="token"/> stopped,
    /// rather than a failure that merely happened while that token was cancelled. A stopped run and
    /// a broken model come back as the same exception type, so what tells them apart is ORT naming
    /// the flag; reading every failure as a cancellation because one was asked for reports a model
    /// that cannot run as a run the caller stopped, and a caller told that retries rather than
    /// fixing the model.
    ///
    /// <para>The window is narrow — ORT reads the flag between nodes and stops there, so a genuine
    /// failure can only outrun a cancellation from the kernel that was already running — which is
    /// also why nothing pins this from the outside: reaching the failing node with the flag already
    /// set is the race itself.</para>
    /// </summary>
    private static bool WasStopped(OnnxRuntimeException cause, CancellationToken token)
        => token.IsCancellationRequested
            && cause.Message.Contains("terminate flag", StringComparison.OrdinalIgnoreCase);

    /// <summary>
    /// What a run that was stopped throws. ORT reports a terminated run as a plain failed one —
    /// an <see cref="OnnxRuntimeException"/> naming the flag — which is indistinguishable at a
    /// glance from a model that is broken, so it is translated here into the one exception .NET
    /// gives this meaning. The cause is kept as the inner exception, and the token is carried so
    /// that a caller racing several runs can tell which cancellation stopped this one.
    /// </summary>
    private static OperationCanceledException Aborted(Exception cause, CancellationToken token)
        => new(
            "The run was stopped before it finished: the RunSettings.CancellationToken it was "
            + "given was cancelled while it was running, so it produced no outputs.",
            cause,
            token);

    /// <summary>Applies what <i>this</i> run runs with. The settings arrive per call rather than
    /// being held by the session, which is ORT's own shape for them: turning arena shrinkage on
    /// takes effect on a session that is already compiled, and on that run alone. Applied by both
    /// run paths, because a retaining run is the one that most wants the arena it keeps its state
    /// in bounded. It configures options the caller owns rather than returning new ones, so the
    /// handle stays inside a `using` at each call site.</summary>
    private void ConfigureRun(RunOptions runOptions, RunSettings runSettings)
    {
        if (OrtBackend.ArenaShrinkageRunConfig(_cudaDeviceId, runSettings.ShrinkArenaAfterRun)
            is { } arena)
            runOptions.AddRunConfigEntry("memory.enable_memory_arena_shrinkage", arena);
    }

    public void Dispose()
    {
        // Off the probe's own Lazy, not off whichever question was asked of it: the info is built
        // by the probe, so a session asked only where its outputs land has one to release too.
        if (_outputMemory.IsValueCreated) _outputMemory.Value.DeviceMemoryInfo?.Dispose();
        if (_arenaAllocator.IsValueCreated)
        {
            // The allocator first: it was built over this memory info and takes it as a bare
            // pointer, so the info outlives it by one statement rather than the other way round.
            _arenaAllocator.Value?.Dispose();
            _ownedArenaMemoryInfo?.Dispose();
        }
        if (_pinnedAllocator.IsValueCreated)
        {
            _pinnedAllocator.Value?.Dispose();
            _ownedPinnedMemoryInfo?.Dispose();
        }
        _session.Dispose();
        // After the session, which is what closes the profile file it has been writing.
        DeleteProfileDirectory();
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Deletes the trace folder for a session nobody disposed. A compiled graph is held weakly by
    /// the context that made it, so one that is used and dropped is collected with no disposal
    /// ever running — and each traced session owns a folder, so without this a program that
    /// compiles as it goes leaves one behind per compile for as long as it runs.
    ///
    /// <para>Only the folder. Nothing native is touched here: the session and everything built
    /// over it own their own handles and are finalized in their own time, and reaching for one of
    /// them from this thread is exactly the use-after-free this backend takes such care to avoid.
    /// A session built without tracing suppresses this in its constructor rather than joining the
    /// finalization queue to do nothing.</para>
    /// </summary>
    ~OrtSession() => DeleteProfileDirectory();

    private void DeleteProfileDirectory()
    {
        if (_profileDirectory is null) return;
        try { Directory.Delete(_profileDirectory, recursive: true); }
        // A temp folder that will not delete is not worth failing a disposal over; the platform
        // reclaims it, and there is nothing a caller could do here.
        catch (Exception) { }
    }
}
