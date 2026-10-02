using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using Python.Runtime;
using Shorokoo.Core.Backends;
using Shorokoo.Core.Factory.IR;
using Shorokoo.PythonHost;
using Shorokoo.PythonTranslation;

namespace Shorokoo.PyTorch;

/// <summary>
/// A model translated to Python and loaded into the interpreter, with its constants on the
/// backend's device: calling it runs the model.
///
/// <para>The translation's compiled code is cached by the translation's hash, so a second session
/// over the same model and pairs — a context compiling the same graph again — compiles nothing; each
/// session holds its own constants.</para>
///
/// <para>Every output is handed over as memory of its own — never an input's, a constant's or
/// another output's, even where the model's graph returns one of those as it is (an
/// <c>Identity</c>, a <c>Reshape</c> view), since the caller owns what it is handed and may write
/// to it — with two exceptions, both on a run that consumed inputs: output aliasing, an output the
/// session was built to write into an input's memory (<see cref="OutputAlias"/>); and placement, an
/// output the run wrote into a range of a consumed input's memory proved free for it
/// (<see cref="TorchPlacements"/>), handed over as a value standing on that memory. See
/// <see cref="RunConsuming(IReadOnlyDictionary{string, IShorokooTensorValue}, IReadOnlyCollection{IShorokooTensorValue}, IReadOnlyList{string}, RunSettings, out IReadOnlyList{string?})"/>.</para>
///
/// <para><b>Precision.</b> Whether torch computes <c>float32</c> products in TensorFloat-32 on a card,
/// and cuDNN convolutions and recurrent layers likewise, is decided by switches of the whole process
/// (<c>torch.backends.cuda.matmul.allow_tf32</c>, <c>torch.backends.cudnn.allow_tf32</c>), which torch
/// reads as it launches each kernel. So a run on a card sets both from its session's
/// <see cref="PrecisionSettings"/> as it starts, under the interpreter lock: off, which is full
/// <c>float32</c> precision, unless the session was built allowing TensorFloat-32. torch releases
/// that lock inside each operator, so runs on cards that set the switches differently do not run at
/// once: a run that allows TensorFloat-32 is the only torch run on a card while it runs, and runs in
/// full precision run beside one another. A run on the CPU reads neither switch and sets neither.</para>
/// </summary>
internal sealed class TorchSession : IShorokooSession
{
    private readonly TorchBackend _backend;
    private readonly TorchRuntime _runtime;
    private readonly string[] _inputNames;
    private readonly string[] _outputNames;
    private readonly ShorokooTensorElementType[] _outputSequenceTypes;
    private readonly Dictionary<string, int> _outputIndex;
    private readonly ShorokooLogSeverity _logSeverity;
    private readonly long? _limitBytes;
    private readonly bool _tensorFloat32;
    private readonly NodePlacement? _nodePlacement;
    private readonly AliasSlot[] _aliases;
    private readonly bool[] _bindable;

    // The loaded model's entry point and its constants, and what the run needs to keep outputs off
    // the constants' memory. Held in fields for the session's life: they are what the model is.
    private readonly PyObject _main;
    private readonly PyObject _constants;
    private readonly PyObject _constantStorages;
    private readonly PyObject _constantIds;
    private readonly TorchPlacements? _placements;
    private int _disposed;

    // torch's CUDA caching allocator is the whole process's, so a cap on it is too: a run under a
    // limit has its device to itself, so that no two runs set the cap at once and no run without
    // one is held to another's; runs without one share it.
    private static readonly ConcurrentDictionary<int, ReaderWriterLockSlim> DeviceRuns = new();

    // torch's TensorFloat-32 switches are the whole process's, so runs on cards that set them
    // differently must not overlap: a run allowing TensorFloat-32 holds this exclusively, and runs in
    // full precision share it. Taken before a device's lock, by every run on a card and no other.
    private static readonly ReaderWriterLockSlim Float32Runs = new(LockRecursionPolicy.SupportsRecursion);

    private TorchSession(
        TorchBackend backend, TorchRuntime runtime, TranslatedModel model, ShorokooLogSeverity logSeverity,
        long? limitBytes, bool tensorFloat32, NodePlacement? nodePlacement, SessionOutputPlacement outputPlacement,
        PyObject main, PyObject constants, PyObject constantStorages, PyObject constantIds,
        ModelProto proto, int modelBytes, IReadOnlyList<OutputAlias> outputAliases)
    {
        _backend = backend;
        _runtime = runtime;
        _inputNames = model.InputNames;
        _outputNames = model.OutputNames;
        _outputSequenceTypes = model.OutputSequenceElementTypes;
        _outputIndex = new Dictionary<string, int>(StringComparer.Ordinal);
        for (int i = 0; i < _outputNames.Length; i++) _outputIndex.TryAdd(_outputNames[i], i);
        _logSeverity = logSeverity;
        _limitBytes = limitBytes;
        _tensorFloat32 = tensorFloat32;
        _nodePlacement = nodePlacement;
        OutputPlacement = outputPlacement;
        // Every slot of the translation's plan, in its numbering, which the translated code's writes
        // name; and of those, the pairs a run can bind at all: an output of the graph's own, by one of
        // main's inputs, that the graph writes into the input's memory -- where both are, the run
        // memory.
        _aliases = [.. model.Aliases];
        _bindable =
        [
            .. _aliases.Select(slot => slot.InputIndex >= 0 && _outputIndex.ContainsKey(slot.Output)
                                       && slot.WrittenByTheGraph),
        ];
        BindableAliases = [.. _aliases.Where((_, slot) => _bindable[slot]).Select(slot => new OutputAlias(slot.Output, slot.Input))];
        _main = main;
        _constants = constants;
        _constantStorages = constantStorages;
        _constantIds = constantIds;
        _placements = TorchPlacements.For(proto, modelBytes, outputAliases, _inputNames, _outputIndex, runtime, constants, model.Constants.Count);
    }

    /// <summary>Whether a session of <paramref name="backend"/> built with <paramref name="precision"/>
    /// has its runs allow TensorFloat-32: on a card, where the settings allow it. A run on the CPU sets
    /// neither of torch's switches, so there it is false whatever the settings say.</summary>
    internal static bool TensorFloat32(TorchBackend backend, PrecisionSettings precision)
        => backend.OnCuda && precision.AllowTensorFloat32;

    /// <summary>Translates <paramref name="modelBytes"/> and loads it.</summary>
    public static TorchSession Create(
        TorchBackend backend, ReadOnlyMemory<byte> modelBytes, ShorokooLogSeverity logSeverity,
        DeviceMemorySettings deviceMemory, DiagnosticSettings diagnostics, IReadOnlyList<OutputAlias> outputAliases,
        PrecisionSettings precision)
    {
        ArgumentNullException.ThrowIfNull(deviceMemory);
        ArgumentNullException.ThrowIfNull(diagnostics);
        ArgumentNullException.ThrowIfNull(outputAliases);
        ArgumentNullException.ThrowIfNull(precision);
        ModelProto proto;
        using (var stream = new MemoryStream(modelBytes.ToArray(), writable: false))
            proto = ProtoBuf.Serializer.Deserialize<ModelProto>(stream);
        // Translated before torch is started, so that a model this backend cannot run is refused
        // without first provisioning an environment to not run it in.
        var model = OnnxToPythonTranslator.Translate(proto, outputAliases, TorchDialect.Instance);
        // The translation's own hash names its compiled code: what it writes depends on the pairs the
        // session was asked for as well as on the model.
        var hash = Convert.ToHexStringLower(SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(model.Source)))[..32];
        var placement = diagnostics.TraceNodePlacement ? Placement(proto.Graph!, backend.DeviceName) : null;

        var runtime = backend.Runtime;
        using (PythonRuntime.Gil())
        {
            var constants = new PyList();
            try
            {
                foreach (var constant in model.Constants)
                {
                    using var value = ConstantValue(runtime, constant, backend.DeviceName);
                    constants.Append(value);
                }
                var main = PyCall.Invoke(runtime.LoadModel, model.Source, $"<shorokoo-model-{hash}>", constants);
                return new TorchSession(backend, runtime, model, logSeverity,
                    backend.OnCuda ? deviceMemory.LimitBytes : null, TensorFloat32(backend, precision),
                    placement, OutputPlacementOf(proto.Graph!, backend.OnCuda),
                    main, constants, runtime.ConstantStorages.Invoke(constants), runtime.ConstantIds.Invoke(constants),
                    proto, modelBytes.Length, outputAliases);
            }
            catch (PythonException ex)
            {
                constants.Dispose();
                throw new InvalidOperationException(
                    $"{backend.Description} could not load the translated model: {ex.Format()}", ex);
            }
        }
    }

    private static unsafe PyObject ConstantValue(TorchRuntime runtime, PythonConstant constant, string device)
    {
        using var shape = TorchBackend.Shape(constant.Shape);
        if (constant.Strings is { } strings)
        {
            using var values = new PyList();
            foreach (var value in strings) PyCall.Append(values, value);
            return runtime.Strings.Invoke(values, shape);
        }
        var bytes = constant.Bytes!;
        var byteCount = PythonElementTypes.ByteCount(constant.ElementType, constant.Shape);
        fixed (byte* source = bytes)
            return PyCall.Invoke(runtime.FromHost, (long)source, (long)byteCount, (int)constant.ElementType, shape, device);
    }

    /// <summary>
    /// Where a session over <paramref name="graph"/> produces its outputs: every tensor on the
    /// session's device, and strings and sequences — which a torch session keeps on the host
    /// whatever its device — on the host, so a CUDA session with some of each is
    /// <see cref="SessionOutputPlacement.Mixed"/>.
    /// </summary>
    internal static SessionOutputPlacement OutputPlacementOf(GraphProto graph, bool onCuda)
    {
        if (!onCuda) return SessionOutputPlacement.Host;
        var host = graph.Outputs.Count(output => output.Type is { } type
            && (type.SequenceType is not null
                || type.TensorType?.ElemType == (int)TensorProto.DataType.String));
        return host == 0 ? SessionOutputPlacement.Device
            : host == graph.Outputs.Count ? SessionOutputPlacement.Host
            : SessionOutputPlacement.Mixed;
    }

    /// <summary>
    /// Every node of <paramref name="graph"/>, run on <paramref name="device"/>: a torch session runs
    /// its whole graph where the session is, with no provider to fall back to, so which device ran a
    /// node is known before any run. torch keeps no per-node record of the bytes each moved, so
    /// those are zero.
    /// </summary>
    internal static NodePlacement Placement(GraphProto graph, string device)
        => new([.. graph.Nodes.Select((node, index) => new NodeExecution(node.Name, node.OpType, device, index, 0, 0, 0))]);

    public IReadOnlyList<string> InputNames => _inputNames;

    public IReadOnlyList<string> OutputNames => _outputNames;

    public SessionOutputPlacement OutputPlacement { get; }

    public IReadOnlyList<OutputAlias> BindableAliases { get; }

    public NodePlacement? ReadNodePlacement() => _nodePlacement;

    /// <summary>Where this session's consuming runs place their values; null where it places none.</summary>
    internal TorchPlacements? Placements => _placements;

    /// <summary>
    /// On CUDA, torch's caching allocator on this session's device, as <c>torch.cuda.memory_stats</c>
    /// reports it — which is the whole process's allocator on that device and not this session's
    /// own, since torch has one per device; null on the CPU, where torch keeps no such figures.
    /// <see cref="ArenaStatistics.LimitBytes"/> is the limit this session's runs get, or -1;
    /// <see cref="ArenaStatistics.MaxAllocSizeBytes"/> and <see cref="ArenaStatistics.ReserveCount"/>
    /// are zero, since torch records neither.
    /// </summary>
    public ArenaStatistics? ReadArenaStatistics()
    {
        if (!_backend.OnCuda || Volatile.Read(ref _disposed) != 0) return null;
        using (PythonRuntime.Gil())
        {
            using var figures = PyCall.Invoke(_runtime.ArenaStatistics, _backend.DeviceName);
            if (figures.IsNone()) return null;
            long At(int index)
            {
                using var item = figures[index];
                return item.As<long>();
            }
            return new ArenaStatistics(At(0), _limitBytes ?? -1, At(1), At(2), At(3), At(4), At(5), At(6), At(7), At(8));
        }
    }

    /// <summary>Runs the model on <paramref name="inputs"/>, every one of them where this backend's
    /// runs read it, and leaves every output there: a tensor on this session's device, and a string
    /// tensor or a sequence in host memory.</summary>
    public IReadOnlyList<IShorokooTensorValue> Run(
        IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
        IReadOnlyList<string> outputNames,
        RunSettings runSettings)
        => RunCore(inputs, null, outputNames, runSettings, out _);

    /// <summary>The overload below, for a caller that does not ask which outputs went into consumed
    /// memory; a session built with pairs writes them there all the same.</summary>
    public IReadOnlyList<IShorokooTensorValue> RunConsuming(
        IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
        IReadOnlyCollection<IShorokooTensorValue> consumed,
        IReadOnlyList<string> outputNames,
        RunSettings runSettings)
        => RunConsuming(inputs, consumed, outputNames, runSettings, out _);

    /// <summary>
    /// Runs with <paramref name="consumed"/> handed over: each is released through the backend,
    /// exactly once, however the run ends — returning, throwing, stopped, refused before it starts.
    ///
    /// <para>An output this session was built to write into the memory of an input
    /// (<see cref="BindableAliases"/>) is written there where the run consumed that input's value, fed
    /// it under no other name, and the value is a torch tensor of this runtime in memory of its own:
    /// by the node that produces it, when that is an <c>Add</c>, <c>Sub</c>, <c>Mul</c> or
    /// <c>Div</c> of the value's floating-point type and shape on the run's device — torch writing
    /// the result into the value rather than into memory it allocates, and so the output costing no
    /// memory at all. The node declines where anything the rest of the run still reads could be the
    /// value under another name: torch's views are more than ONNX Runtime's, and the proof behind the
    /// pair knows only those. <paramref name="aliasedInputs"/> names, per output, the input it was
    /// written into.</para>
    ///
    /// <para>Values of the run are written into ranges of the consumed values' memory where the
    /// graph proves those ranges free for them (<see cref="TorchPlacements"/>): an output written so
    /// is handed over standing on the consumed value's memory, a <see cref="SharedBlock"/> it holds a
    /// lease on, beside every other output of the run standing on it.</para>
    ///
    /// <para>The consumed values are released as the run returns. An output written into one is a
    /// value of its own holding a reference to the same tensor, so it outlives the release.</para>
    /// </summary>
    public IReadOnlyList<IShorokooTensorValue> RunConsuming(
        IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
        IReadOnlyCollection<IShorokooTensorValue> consumed,
        IReadOnlyList<string> outputNames,
        RunSettings runSettings,
        out IReadOnlyList<string?> aliasedInputs)
    {
        ArgumentNullException.ThrowIfNull(consumed);
        try
        {
            return RunCore(inputs, consumed, outputNames, runSettings, out aliasedInputs);
        }
        finally
        {
            foreach (var value in consumed) _backend.Release(value);
        }
    }

    private IReadOnlyList<IShorokooTensorValue> RunCore(
        IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
        IReadOnlyCollection<IShorokooTensorValue>? consumed,
        IReadOnlyList<string> outputNames,
        RunSettings runSettings,
        out IReadOnlyList<string?> aliasedInputs)
    {
        aliasedInputs = [];
        ArgumentNullException.ThrowIfNull(inputs);
        ArgumentNullException.ThrowIfNull(outputNames);
        ArgumentNullException.ThrowIfNull(runSettings);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        var token = runSettings.CancellationToken;
        token.ThrowIfCancellationRequested();

        var wanted = new int[outputNames.Count];
        for (int i = 0; i < wanted.Length; i++)
            wanted[i] = _outputIndex.TryGetValue(outputNames[i], out var index)
                ? index
                : throw new ArgumentException($"The model has no output '{outputNames[i]}'.", nameof(outputNames));

        var stop = IntPtr.Zero;
        CancellationTokenRegistration registration = default;
        try
        {
            var feeds = new TorchTensorValue[_inputNames.Length];
            for (int i = 0; i < feeds.Length; i++)
            {
                if (!inputs.TryGetValue(_inputNames[i], out var value))
                    throw new ArgumentException($"The model's input '{_inputNames[i]}' was not fed.", nameof(inputs));
                feeds[i] = Fed(_inputNames[i], value);
            }
            var targets = AliasTargets(inputs, consumed, feeds);
            var entry = consumed is null || _placements is null
                ? null
                : _placements.EntryFor(feeds, TorchPlacements.Blocks(_inputNames, inputs, consumed, feeds, targets), outputNames);
            if (entry?.Main is null) entry = null;

            // A flag in native memory the run reads before every node, and the token's callback
            // sets: it needs no interpreter lock to set, so a cancellation lands while the run holds
            // it. Freed only after the registration is disposed, which waits out a callback running.
            if (token.CanBeCanceled)
            {
                stop = Marshal.AllocHGlobal(sizeof(int));
                Marshal.WriteInt32(stop, 0);
                registration = token.Register(static flag => Marshal.WriteInt32((IntPtr)flag!, 1), stop);
            }

            var device = _backend.OnCuda
                ? DeviceRuns.GetOrAdd(_backend.CudaDeviceId, static _ => new ReaderWriterLockSlim(LockRecursionPolicy.SupportsRecursion))
                : null;
            if (device is null) return Invoke(feeds, wanted, outputNames, targets, entry, stop, runSettings, out aliasedInputs);
            if (_tensorFloat32) Float32Runs.EnterWriteLock();
            else Float32Runs.EnterReadLock();
            try
            {
                var capped = _limitBytes is not null;
                if (capped) device.EnterWriteLock();
                else device.EnterReadLock();
                try
                {
                    return Invoke(feeds, wanted, outputNames, targets, entry, stop, runSettings, out aliasedInputs);
                }
                finally
                {
                    if (capped) device.ExitWriteLock();
                    else device.ExitReadLock();
                }
            }
            finally
            {
                if (_tensorFloat32) Float32Runs.ExitWriteLock();
                else Float32Runs.ExitReadLock();
            }
        }
        finally
        {
            registration.Dispose();
            if (stop != IntPtr.Zero) Marshal.FreeHGlobal(stop);
        }
    }

    /// <summary>
    /// <paramref name="value"/>, fed as input <paramref name="name"/>, where it is one of this
    /// runtime's in the memory this session's runs read it in: a tensor on this session's device,
    /// and a string tensor or a sequence in host memory. Refused otherwise, before the run starts:
    /// nothing here moves a value, the framework placing every input there before it hands it over
    /// (<see cref="IShorokooBackend.RunMemoryOf"/>).
    /// </summary>
    /// <exception cref="InvalidOperationException">The value is not this runtime's, or not where this
    /// session's runs read it.</exception>
    private TorchTensorValue Fed(string name, IShorokooTensorValue value)
    {
        if (value is TorchTensorValue own)
        {
            var hostRead = own.ValueType != ShorokooOnnxValueType.Tensor || own.ElementType == ShorokooTensorElementType.String;
            if ((hostRead || !_backend.OnCuda) ? own.IsHostAccessible : own.CudaDevice == _backend.CudaDeviceId)
                return own;
        }
        throw new InvalidOperationException(
            $"Input '{name}' was handed to a session of {_backend.Description} as "
            + (value is TorchTensorValue torch
                ? $"a {torch.ValueType} of {torch.ElementType} in "
                  + (torch.IsHostAccessible ? "host memory" : $"cuda:{torch.CudaDevice}'s memory")
                : $"a {value.GetType().Name}, which is not a value of this runtime")
            + ", outside the memory its runs read it in. A session takes only values its backend's runs "
            + "read where they are, and moves none: place the input there first, through the backend's "
            + "own moves (IShorokooBackend.CreateTensorInBackendMemory).");
    }

    /// <summary>
    /// Per slot of the translation's plan, the position of the input whose consumed value the output
    /// may be written into, or -1: where the session binds the slot's pair, the run consumed the very
    /// value it is fed as that input, fed it under no other name, and it is a tensor of this runtime.
    /// What else it takes is settled in the run.
    /// </summary>
    private int[] AliasTargets(
        IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
        IReadOnlyCollection<IShorokooTensorValue>? consumed,
        TorchTensorValue[] feeds)
    {
        if (BindableAliases.Count == 0 || consumed is null || consumed.Count == 0) return [];
        var handed = new HashSet<IShorokooTensorValue>(consumed, ReferenceEqualityComparer.Instance);
        var fedAs = new Dictionary<IShorokooTensorValue, int>(ReferenceEqualityComparer.Instance);
        foreach (var value in inputs.Values) fedAs[value] = fedAs.GetValueOrDefault(value) + 1;
        var targets = new int[_aliases.Length];
        for (int slot = 0; slot < targets.Length; slot++)
        {
            var alias = _aliases[slot];
            var value = _bindable[slot] ? inputs.GetValueOrDefault(alias.Input) : null;
            targets[slot] = value is TorchTensorValue { ValueType: ShorokooOnnxValueType.Tensor } own
                            && own.ElementType != ShorokooTensorElementType.String
                            && handed.Contains(own) && fedAs[own] == 1 && ReferenceEquals(feeds[alias.InputIndex], own)
                ? alias.InputIndex
                : -1;
        }
        return targets;
    }

    private IReadOnlyList<IShorokooTensorValue> Invoke(
        TorchTensorValue[] feeds, int[] wanted, IReadOnlyList<string> outputNames,
        int[] targets, TorchPlacements.Entry? placing, IntPtr stop, RunSettings runSettings, out IReadOnlyList<string?> aliasedInputs)
    {
        aliasedInputs = [];
        using (PythonRuntime.Gil())
        {
            using var args = new PyList();
            foreach (var feed in feeds) args.Append(feed.Value);
            using var wantedList = new PyList();
            foreach (var index in wanted) PyCall.Append(wantedList, index);
            using var aliases = new PyList();
            for (int slot = 0; slot < targets.Length; slot++)
            {
                var alias = _aliases[slot];
                using var output = new PyInt(_bindable[slot] ? _outputIndex[alias.Output] : -1);
                using var target = new PyInt(targets[slot]);
                using var entry = new PyTuple([output, target]);
                aliases.Append(entry);
            }

            PyObject results;
            try
            {
                using var noPlacements = new PyList();
                results = PyCall.Invoke(_runtime.Run,
                    placing?.Main ?? _main, args, wantedList, _backend.DeviceName, _constantStorages, _constantIds,
                    stop.ToInt64(), (int)_logSeverity, aliases, _limitBytes ?? -1L, runSettings.ShrinkArenaAfterRun,
                    _tensorFloat32, placing?.Slots ?? noPlacements);
            }
            catch (PythonException ex) when (ex.Type.Name == TorchRuntime.RunStopped && runSettings.CancellationToken.IsCancellationRequested)
            {
                throw new OperationCanceledException(
                    "The run was stopped before it finished: the RunSettings.CancellationToken it was given was "
                    + "cancelled while it was running, so it produced no outputs.", ex, runSettings.CancellationToken);
            }
            catch (PythonException ex) when (_limitBytes is { } limit && ex.Type.Name == "OutOfMemoryError")
            {
                throw new InvalidOperationException(
                    $"Running the model on {_backend.Description} needed more device memory than the {limit} bytes "
                    + $"this session's runs may allocate on {_backend.DeviceName} beyond what torch's allocator held "
                    + $"there when the run started (DeviceMemorySettings.LimitBytes): {ex.Format()}", ex);
            }
            catch (PythonException ex)
            {
                throw new InvalidOperationException(
                    $"Running the model on {_backend.Description} failed in PyTorch: {ex.Format()}", ex);
            }
            // The feeds' wrappers are what hold the tensors the run just read; the list above holds
            // references too, but these are the objects a caller's span may point into.
            GC.KeepAlive(feeds);

            using (results)
            {
                var outputs = new List<IShorokooTensorValue>(wanted.Length);
                string?[]? aliased = null;
                Dictionary<int, (SharedBlock Block, long Offset)>? blocks = null;
                try
                {
                    for (int i = 0; i < wanted.Length; i++)
                    {
                        using var triple = results[i];
                        using var description = triple[1];
                        using var written = triple[2];
                        var output = TorchTensorValue.Wrap(triple[0], description, _outputSequenceTypes[wanted[i]]);
                        outputs.Add(output);
                        if (PyTuple.IsTupleType(written))
                        {
                            using var slot = written[0];
                            var placement = placing!.Plan[slot.As<int>()];
                            var input = Array.IndexOf(_inputNames, placement.Block);
                            blocks ??= [];
                            if (!blocks.TryGetValue(input, out var block))
                                blocks[input] = block = BlockOf(feeds[input]);
                            output.StandOn(new BlockRange(block.Block, block.Offset + placement.Offset, placement.Bytes));
                        }
                        else if (written.IsTrue())
                            (aliased ??= new string?[wanted.Length])[i] = _aliases.Where((_, slot) => _bindable[slot]).First(a => a.Output == outputNames[i]).Input;
                    }
                }
                catch
                {
                    foreach (var output in outputs) output.Dispose();
                    throw;
                }
                if (aliased is not null) aliasedInputs = aliased;
                return outputs;
            }
        }
    }

    /// <summary>
    /// The block the outputs a run placed in consumed value <paramref name="consumed"/> stand on, and
    /// where its memory starts in it: the block the value itself stands on, where it does — its range
    /// is that block's to give out again — or else a block of the value's own memory, which torch
    /// frees once the last tensor reading it is gone, so letting go of the block has nothing to do.
    /// </summary>
    private static (SharedBlock Block, long Offset) BlockOf(TorchTensorValue consumed)
        => consumed.Range is { } range
            ? (range.Block, range.Offset)
            : (new SharedBlock(TorchPlacements.BytesOf(consumed), static () => { }), 0);

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _placements?.Dispose();
        using (PythonRuntime.Gil())
        {
            _main.Dispose();
            _constants.Dispose();
            _constantStorages.Dispose();
            _constantIds.Dispose();
        }
    }
}
