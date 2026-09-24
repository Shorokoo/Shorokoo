using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using Python.Runtime;
using Shorokoo.Core.Backends;
using Shorokoo.Core.Factory.IR;
using Shorokoo.PythonHost;
using Shorokoo.PyTorch.Translation;

namespace Shorokoo.PyTorch;

/// <summary>
/// A model translated to Python and loaded into the interpreter, with its constants on the
/// backend's device: calling it runs the model.
///
/// <para>The translation's compiled code is cached by the model's hash, so a second session over
/// the same model — a context compiling the same graph again — compiles nothing; each session holds
/// its own constants.</para>
///
/// <para>Every output is handed over as memory of its own — never an input's, a constant's or
/// another output's, even where the model's graph returns one of those as it is (an
/// <c>Identity</c>, a <c>Reshape</c> view), since the caller owns what it is handed and may write
/// to it — with one exception, which is output aliasing: an output the session was built to write
/// into an input's memory (<see cref="OutputAlias"/>), on a run that consumed that input. See
/// <see cref="RunConsuming(IReadOnlyDictionary{string, IShorokooTensorValue}, IReadOnlyCollection{IShorokooTensorValue}, IReadOnlyList{string}, IReadOnlySet{string}, RunSettings, out IReadOnlyList{string?})"/>.</para>
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
    private readonly NodePlacement? _nodePlacement;
    private readonly AliasSlot[] _aliases;
    private readonly bool[] _bindable;

    // The loaded model's entry point and its constants, and what the run needs to keep outputs off
    // the constants' memory. Held in fields for the session's life: they are what the model is.
    private readonly PyObject _main;
    private readonly PyObject _constants;
    private readonly PyObject _constantStorages;
    private readonly PyObject _constantIds;
    private int _disposed;

    // torch's CUDA caching allocator is the whole process's, so a cap on it is too: runs under a
    // limit are serialized per device, so that no two of them set the cap at once.
    private static readonly ConcurrentDictionary<int, object> CappedRuns = new();

    private TorchSession(
        TorchBackend backend, TorchRuntime runtime, TranslatedModel model, ShorokooLogSeverity logSeverity,
        long? limitBytes, NodePlacement? nodePlacement, SessionOutputPlacement outputPlacement,
        PyObject main, PyObject constants, PyObject constantStorages, PyObject constantIds)
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
        _nodePlacement = nodePlacement;
        OutputPlacement = outputPlacement;
        // Every slot of the translation's plan, in its numbering, which the translated code's writes
        // name; and of those, the pairs a run can bind at all: an output of the graph's own, by one of
        // main's inputs. Written by the graph, a pair binds wherever the output ends up in the memory
        // its input is in; copied home into it, only on a CUDA session, for an output fetched back.
        _aliases = [.. model.Aliases];
        _bindable =
        [
            .. _aliases.Select(slot => slot.InputIndex >= 0 && _outputIndex.ContainsKey(slot.Output)
                                       && (slot.WrittenByTheGraph || backend.OnCuda)),
        ];
        BindableAliases = [.. _aliases.Where((_, slot) => _bindable[slot]).Select(slot => new OutputAlias(slot.Output, slot.Input))];
        _main = main;
        _constants = constants;
        _constantStorages = constantStorages;
        _constantIds = constantIds;
    }

    /// <summary>Translates <paramref name="modelBytes"/> and loads it.</summary>
    public static TorchSession Create(
        TorchBackend backend, ReadOnlyMemory<byte> modelBytes, ShorokooLogSeverity logSeverity,
        DeviceMemorySettings deviceMemory, DiagnosticSettings diagnostics, IReadOnlyList<OutputAlias> outputAliases)
    {
        ArgumentNullException.ThrowIfNull(deviceMemory);
        ArgumentNullException.ThrowIfNull(diagnostics);
        ArgumentNullException.ThrowIfNull(outputAliases);
        ModelProto proto;
        using (var stream = new MemoryStream(modelBytes.ToArray(), writable: false))
            proto = ProtoBuf.Serializer.Deserialize<ModelProto>(stream);
        // Translated before torch is started, so that a model this backend cannot run is refused
        // without first provisioning an environment to not run it in.
        var model = OnnxToPythonTranslator.Translate(proto, outputAliases);
        var hash = Convert.ToHexStringLower(SHA256.HashData(modelBytes.Span))[..32];
        var placement = diagnostics.TraceNodePlacement ? Placement(proto.Graph!, backend.DeviceName) : null;

        var runtime = backend.Runtime;
        using (PythonRuntime.Gil())
        {
            var constants = new PyList();
            try
            {
                foreach (var constant in model.Constants)
                    constants.Append(ConstantValue(runtime, constant, backend.DeviceName));
                var main = runtime.LoadModel.Invoke(
                    new PyString(model.Source), new PyString($"<shorokoo-model-{hash}>"), constants);
                return new TorchSession(backend, runtime, model, logSeverity,
                    backend.OnCuda ? deviceMemory.LimitBytes : null, placement, OutputPlacementOf(proto.Graph!, backend.OnCuda),
                    main, constants, runtime.ConstantStorages.Invoke(constants), runtime.ConstantIds.Invoke(constants));
            }
            catch (PythonException ex)
            {
                constants.Dispose();
                throw new InvalidOperationException(
                    $"{backend.Description} could not load the translated model: {ex.Format()}", ex);
            }
        }
    }

    private static unsafe PyObject ConstantValue(TorchRuntime runtime, TorchConstant constant, string device)
    {
        var shape = TorchBackend.Shape(constant.Shape);
        if (constant.Strings is { } strings)
        {
            using var values = new PyList();
            foreach (var value in strings) values.Append(new PyString(value));
            return runtime.Strings.Invoke(values, shape);
        }
        var bytes = constant.Bytes!;
        var byteCount = TorchElementTypes.ByteCount(constant.ElementType, constant.Shape);
        fixed (byte* source = bytes)
            return runtime.FromHost.Invoke(
                new PyInt((long)source), new PyInt(byteCount), new PyInt((int)constant.ElementType),
                shape, new PyString(device));
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

    /// <summary>A CUDA session computes on the card, and can leave outputs there.</summary>
    public bool HasDeviceMemory => _backend.OnCuda;

    public SessionOutputPlacement OutputPlacement { get; }

    public IReadOnlyList<OutputAlias> BindableAliases { get; }

    public NodePlacement? ReadNodePlacement() => _nodePlacement;

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
            using var figures = _runtime.ArenaStatistics.Invoke(new PyString(_backend.DeviceName));
            if (figures.IsNone()) return null;
            long At(int index)
            {
                using var item = figures[index];
                return item.As<long>();
            }
            return new ArenaStatistics(At(0), _limitBytes ?? -1, At(1), At(2), At(3), At(4), At(5), At(6), At(7));
        }
    }

    public IReadOnlyList<IShorokooTensorValue> Run(
        IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
        IReadOnlyList<string> outputNames,
        RunSettings runSettings)
        => RunCore(inputs, null, outputNames, EmptySet, runSettings, out _);

    public IReadOnlyList<IShorokooTensorValue> RunRetainingOutputs(
        IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
        IReadOnlyList<string> outputNames,
        IReadOnlySet<string> retainedOutputNames,
        RunSettings runSettings)
        => RunCore(inputs, null, outputNames, retainedOutputNames, runSettings, out _);

    /// <summary>The overload below, for a caller that does not ask which outputs went into consumed
    /// memory; a session built with pairs writes them there all the same.</summary>
    public IReadOnlyList<IShorokooTensorValue> RunConsuming(
        IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
        IReadOnlyCollection<IShorokooTensorValue> consumed,
        IReadOnlyList<string> outputNames,
        IReadOnlySet<string> retainedOutputNames,
        RunSettings runSettings)
        => RunConsuming(inputs, consumed, outputNames, retainedOutputNames, runSettings, out _);

    /// <summary>
    /// Runs with <paramref name="consumed"/> handed over: each is released through the backend,
    /// exactly once, however the run ends — returning, throwing, stopped, refused before it starts.
    ///
    /// <para>An output this session was built to write into the memory of an input
    /// (<see cref="BindableAliases"/>) is written there where the run consumed that input's value, fed
    /// it under no other name, and the value is a torch tensor of this runtime in memory of its own:
    /// by the node that produces it, when that is an <c>Add</c>, <c>Sub</c>, <c>Mul</c> or
    /// <c>Div</c> of the value's floating-point type and shape on the device the output ends up on —
    /// torch writing the result into the value rather than into memory it allocates, and so the
    /// output costing no memory at all — or, on a CUDA session, by copying an output the run fetches
    /// back into a consumed value in host memory rather than into host memory of its own. The node
    /// declines where anything the rest of the run still reads could be the value under another
    /// name: torch's views are more than ONNX Runtime's, and the proof behind the pair knows only
    /// those. <paramref name="aliasedInputs"/> names, per output, the input it was written into.</para>
    ///
    /// <para>The consumed values are released as the run returns. An output written into one is a
    /// value of its own holding a reference to the same tensor, so it outlives the release.</para>
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
        try
        {
            return RunCore(inputs, consumed, outputNames, retainedOutputNames, runSettings, out aliasedInputs);
        }
        finally
        {
            foreach (var value in consumed) _backend.Release(value);
        }
    }

    private static readonly IReadOnlySet<string> EmptySet = new HashSet<string>();

    private IReadOnlyList<IShorokooTensorValue> RunCore(
        IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
        IReadOnlyCollection<IShorokooTensorValue>? consumed,
        IReadOnlyList<string> outputNames,
        IReadOnlySet<string> retainedOutputNames,
        RunSettings runSettings,
        out IReadOnlyList<string?> aliasedInputs)
    {
        aliasedInputs = [];
        ArgumentNullException.ThrowIfNull(inputs);
        ArgumentNullException.ThrowIfNull(outputNames);
        ArgumentNullException.ThrowIfNull(retainedOutputNames);
        ArgumentNullException.ThrowIfNull(runSettings);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        var token = runSettings.CancellationToken;
        token.ThrowIfCancellationRequested();

        var wanted = new int[outputNames.Count];
        for (int i = 0; i < wanted.Length; i++)
            wanted[i] = _outputIndex.TryGetValue(outputNames[i], out var index)
                ? index
                : throw new ArgumentException($"The model has no output '{outputNames[i]}'.", nameof(outputNames));

        List<IShorokooTensorValue>? borrowed = null;
        var stop = IntPtr.Zero;
        CancellationTokenRegistration registration = default;
        try
        {
            var feeds = new TorchTensorValue[_inputNames.Length];
            for (int i = 0; i < feeds.Length; i++)
            {
                if (!inputs.TryGetValue(_inputNames[i], out var value))
                    throw new ArgumentException($"The model's input '{_inputNames[i]}' was not fed.", nameof(inputs));
                if (value is TorchTensorValue own)
                {
                    feeds[i] = own;
                    continue;
                }
                // A value of another runtime is rebuilt on this one for the run, and released after it.
                var copy = (TorchTensorValue)BackendTransfer.CopyTo(_backend, value);
                (borrowed ??= []).Add(copy);
                feeds[i] = copy;
            }
            var targets = AliasTargets(inputs, consumed, feeds, retainedOutputNames);

            // A flag in native memory the run reads before every node, and the token's callback
            // sets: it needs no interpreter lock to set, so a cancellation lands while the run holds
            // it. Freed only after the registration is disposed, which waits out a callback running.
            if (token.CanBeCanceled)
            {
                stop = Marshal.AllocHGlobal(sizeof(int));
                Marshal.WriteInt32(stop, 0);
                registration = token.Register(static flag => Marshal.WriteInt32((IntPtr)flag!, 1), stop);
            }

            var capped = _limitBytes is not null ? CappedRuns.GetOrAdd(_backend.CudaDeviceId, static _ => new object()) : null;
            if (capped is not null) Monitor.Enter(capped);
            try
            {
                return Invoke(feeds, wanted, outputNames, retainedOutputNames, targets, stop, runSettings, out aliasedInputs);
            }
            finally
            {
                if (capped is not null) Monitor.Exit(capped);
            }
        }
        finally
        {
            registration.Dispose();
            if (stop != IntPtr.Zero) Marshal.FreeHGlobal(stop);
            if (borrowed is not null) foreach (var copy in borrowed) copy.Dispose();
        }
    }

    /// <summary>
    /// Per slot of the translation's plan, the position of the input whose consumed value the output
    /// may be written into, or -1: where the session binds the slot's pair, the run consumed the very
    /// value it is fed as that input, fed it under no other name, and it is a tensor of this runtime
    /// — a copy made for the run from another runtime's value is not the value consumed. What else it
    /// takes is settled in the run.
    /// </summary>
    private int[] AliasTargets(
        IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
        IReadOnlyCollection<IShorokooTensorValue>? consumed,
        TorchTensorValue[] feeds,
        IReadOnlySet<string> retainedOutputNames)
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
        TorchTensorValue[] feeds, int[] wanted, IReadOnlyList<string> outputNames, IReadOnlySet<string> retainedOutputNames,
        int[] targets, IntPtr stop, RunSettings runSettings, out IReadOnlyList<string?> aliasedInputs)
    {
        aliasedInputs = [];
        using (PythonRuntime.Gil())
        {
            using var args = new PyList();
            foreach (var feed in feeds) args.Append(feed.Value);
            using var wantedList = new PyList();
            using var retainedList = new PyList();
            foreach (var index in wanted)
            {
                wantedList.Append(new PyInt(index));
                retainedList.Append((_backend.OnCuda && retainedOutputNames.Contains(_outputNames[index])).ToPython());
            }
            using var aliases = new PyList();
            for (int slot = 0; slot < targets.Length; slot++)
            {
                var alias = _aliases[slot];
                using var entry = new PyTuple([
                    new PyInt(_bindable[slot] ? _outputIndex[alias.Output] : -1), new PyInt(targets[slot]),
                    (_backend.OnCuda && retainedOutputNames.Contains(alias.Output)).ToPython(),
                ]);
                aliases.Append(entry);
            }

            PyObject results;
            try
            {
                results = _runtime.Run.Invoke(
                    [_main, args, wantedList, retainedList, new PyString(_backend.DeviceName),
                     _constantStorages, _constantIds, new PyInt(stop.ToInt64()), new PyInt((int)_logSeverity),
                     aliases, new PyInt(_limitBytes ?? -1), runSettings.ShrinkArenaAfterRun.ToPython()]);
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
                    + $"this session's runs may allocate on {_backend.DeviceName} beyond what was allocated there "
                    + $"when the run started (DeviceMemorySettings.LimitBytes): {ex.Format()}", ex);
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
                try
                {
                    for (int i = 0; i < wanted.Length; i++)
                    {
                        using var triple = results[i];
                        using var description = triple[1];
                        using var written = triple[2];
                        outputs.Add(TorchTensorValue.Wrap(triple[0], description, _outputSequenceTypes[wanted[i]]));
                        if (written.IsTrue())
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

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        using (PythonRuntime.Gil())
        {
            _main.Dispose();
            _constants.Dispose();
            _constantStorages.Dispose();
            _constantIds.Dispose();
        }
    }
}
