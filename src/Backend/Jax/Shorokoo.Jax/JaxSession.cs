using System.Security.Cryptography;
using Python.Runtime;
using Shorokoo.Core.Backends;
using Shorokoo.Core.Factory.IR;
using Shorokoo.PythonHost;
using Shorokoo.PythonTranslation;

namespace Shorokoo.Jax;

/// <summary>
/// A model translated to Python and loaded into the interpreter, with its constants: calling it
/// runs the program XLA compiled from it for the shapes and element types of the inputs it is fed —
/// compiled the first time a run feeds that signature, and kept.
///
/// <para>Where every input of the model has a fixed shape and element type, the program for them is
/// compiled when the session is created, so that a model the backend cannot compile — one that
/// needs, as a number, something the graph computes from an input's values — is refused then.</para>
///
/// <para>Every output is handed over as memory of its own. A JAX array is never written in place,
/// so a run writes no output into a consumed input (<see cref="BindableAliases"/> is empty), and
/// one it hands home is a copy of its own; one it keeps on the card is an array no other value
/// shares a write to.</para>
///
/// <para><b>Stopping.</b> A run is one compiled program, which cannot be stopped part way: a run
/// whose token is cancelled before it starts is refused, and one cancelled while it runs finishes.</para>
/// </summary>
internal sealed class JaxSession : IShorokooSession
{
    private readonly JaxBackend _backend;
    private readonly JaxRuntime _runtime;
    private readonly string[] _inputNames;
    private readonly string[] _outputNames;
    private readonly Dictionary<string, int> _outputIndex;
    private readonly ShorokooLogSeverity _logSeverity;
    private readonly NodePlacement? _nodePlacement;

    // The loaded model: its code, its constants and the programs compiled from it. Held for the
    // session's life: it is what the model is.
    private readonly PyObject _model;
    private int _disposed;

    private JaxSession(
        JaxBackend backend, JaxRuntime runtime, TranslatedModel model, ShorokooLogSeverity logSeverity,
        NodePlacement? nodePlacement, PyObject loaded)
    {
        _backend = backend;
        _runtime = runtime;
        _inputNames = model.InputNames;
        _outputNames = model.OutputNames;
        _outputIndex = new Dictionary<string, int>(StringComparer.Ordinal);
        for (int i = 0; i < _outputNames.Length; i++) _outputIndex.TryAdd(_outputNames[i], i);
        _logSeverity = logSeverity;
        _nodePlacement = nodePlacement;
        _model = loaded;
    }

    /// <summary>Translates <paramref name="modelBytes"/>, loads it, and compiles it where its inputs'
    /// shapes are fixed.</summary>
    public static JaxSession Create(
        JaxBackend backend, ReadOnlyMemory<byte> modelBytes, ShorokooLogSeverity logSeverity, DiagnosticSettings diagnostics)
    {
        ArgumentNullException.ThrowIfNull(diagnostics);
        ModelProto proto;
        using (var stream = new MemoryStream(modelBytes.ToArray(), writable: false))
            proto = ProtoBuf.Serializer.Deserialize<ModelProto>(stream);
        // Translated before JAX is started, so that a model this backend cannot run is refused
        // without first provisioning an environment to not run it in.
        var model = OnnxToPythonTranslator.Translate(proto, [], JaxDialect.Instance);
        var hash = Convert.ToHexStringLower(SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(model.Source)))[..32];
        var placement = diagnostics.TraceNodePlacement ? Placement(proto.Graph!, backend.DeviceName) : null;
        var signature = FixedSignature(proto.Graph!, model.InputNames);

        var runtime = backend.Runtime;
        using (PythonRuntime.Gil())
        {
            PyObject? loaded = null;
            try
            {
                using var constants = new PyList();
                foreach (var constant in model.Constants)
                {
                    using var value = ConstantValue(runtime, constant);
                    constants.Append(value);
                }
                loaded = PyCall.Invoke(runtime.LoadModel, model.Source, $"<shorokoo-model-{hash}>", constants, backend.DeviceName);
                if (signature is not null)
                {
                    using var inputs = new PyList();
                    foreach (var (elementType, shape) in signature)
                    {
                        using var dims = JaxBackend.Shape(shape);
                        using var code = new PyInt((int)elementType);
                        using var entry = new PyTuple([code, dims]);
                        inputs.Append(entry);
                    }
                    PyCall.Invoke(runtime.Prepare, loaded, inputs).Dispose();
                }
                return new JaxSession(backend, runtime, model, logSeverity, placement, loaded);
            }
            catch (PythonException ex)
            {
                loaded?.Dispose();
                throw Refusal(backend, ex, "could not compile the translated model");
            }
        }
    }

    /// <summary>A failure JAX raised, as the backend reports it: a model that needs, as a number,
    /// something its inputs' values decide is one the backend cannot run
    /// (<see cref="JaxUnsupportedModelException"/>); anything else is a failure of the run.</summary>
    private static Exception Refusal(JaxBackend backend, PythonException ex, string what)
    {
        if (ex.Type.Name == JaxRuntime.DataDependentShape)
        {
            string? op = null;
            if (ex.Value is { } value && value.HasAttr("operator"))
            {
                using var name = value.GetAttr("operator");
                op = name.As<string>();
            }
            return new JaxUnsupportedModelException(JaxUnsupportedReason.UnsupportedUsage, "", op,
                $"The JAX backend cannot run the model: {ex.Message}", ex);
        }
        return new InvalidOperationException($"{backend.Description} {what}: {ex.Format()}", ex);
    }

    private static unsafe PyObject ConstantValue(JaxRuntime runtime, PythonConstant constant)
    {
        using var shape = JaxBackend.Shape(constant.Shape);
        var bytes = constant.Bytes!;
        var byteCount = PythonElementTypes.ByteCount(constant.ElementType, constant.Shape);
        fixed (byte* source = bytes)
            return PyCall.Invoke(runtime.FromHost, (long)source, (long)byteCount, (int)constant.ElementType, shape, "cpu");
    }

    /// <summary>The element type and shape of every input <c>main</c> takes, where the graph fixes
    /// them all; null where one has a symbolic or missing dimension, or no declared type.</summary>
    private static List<(ShorokooTensorElementType, long[])>? FixedSignature(GraphProto graph, string[] inputNames)
    {
        var signature = new List<(ShorokooTensorElementType, long[])>(inputNames.Length);
        foreach (var name in inputNames)
        {
            var tensor = graph.Inputs.First(i => i.Name == name).Type?.TensorType;
            if (tensor is not { ElemType: > 0, Shape: { } shape } || shape.Dims.Any(d => !d.ShouldSerializeDimValue() || d.DimValue < 0))
                return null;
            signature.Add(((ShorokooTensorElementType)tensor.ElemType, [.. shape.Dims.Select(d => d.DimValue)]));
        }
        return signature;
    }

    /// <summary>Every node of <paramref name="graph"/>, run on <paramref name="device"/>: a JAX
    /// session runs its whole graph where the session is.</summary>
    internal static NodePlacement Placement(GraphProto graph, string device)
        => new([.. graph.Nodes.Select((node, index) => new NodeExecution(node.Name, node.OpType, device, index, 0, 0, 0))]);

    public IReadOnlyList<string> InputNames => _inputNames;

    public IReadOnlyList<string> OutputNames => _outputNames;

    /// <summary>A CUDA session computes on the card, and can leave outputs there.</summary>
    public bool HasDeviceMemory => _backend.OnCuda;

    public SessionOutputPlacement OutputPlacement => _backend.OnCuda ? SessionOutputPlacement.Device : SessionOutputPlacement.Host;

    public IReadOnlyList<OutputAlias> BindableAliases => [];

    public NodePlacement? ReadNodePlacement() => _nodePlacement;

    /// <summary>
    /// On CUDA, the device allocator's figures as <c>jax.Device.memory_stats</c> reports them —
    /// which are the whole process's JAX allocator on that device, not this session's own; null on
    /// the CPU. <see cref="ArenaStatistics.LimitBytes"/> is the most the allocator may hold.
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
            return new ArenaStatistics(At(0), At(1), At(2), At(3), At(4), At(5), At(6), At(7), At(8));
        }
    }

    public IReadOnlyList<IShorokooTensorValue> Run(
        IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
        IReadOnlyList<string> outputNames,
        RunSettings runSettings)
        => RunCore(inputs, outputNames, EmptySet, runSettings);

    public IReadOnlyList<IShorokooTensorValue> RunRetainingOutputs(
        IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
        IReadOnlyList<string> outputNames,
        IReadOnlySet<string> retainedOutputNames,
        RunSettings runSettings)
        => RunCore(inputs, outputNames, retainedOutputNames, runSettings);

    /// <summary>Runs with <paramref name="consumed"/> handed over: each is released through the
    /// backend, exactly once, however the run ends. Nothing is written into them.</summary>
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
            return RunCore(inputs, outputNames, retainedOutputNames, runSettings);
        }
        finally
        {
            foreach (var value in consumed) _backend.Release(value);
        }
    }

    private static readonly IReadOnlySet<string> EmptySet = new HashSet<string>();

    private IReadOnlyList<IShorokooTensorValue> RunCore(
        IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
        IReadOnlyList<string> outputNames,
        IReadOnlySet<string> retainedOutputNames,
        RunSettings runSettings)
    {
        ArgumentNullException.ThrowIfNull(inputs);
        ArgumentNullException.ThrowIfNull(outputNames);
        ArgumentNullException.ThrowIfNull(retainedOutputNames);
        ArgumentNullException.ThrowIfNull(runSettings);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        runSettings.CancellationToken.ThrowIfCancellationRequested();

        var wanted = new int[outputNames.Count];
        for (int i = 0; i < wanted.Length; i++)
            wanted[i] = _outputIndex.TryGetValue(outputNames[i], out var index)
                ? index
                : throw new ArgumentException($"The model has no output '{outputNames[i]}'.", nameof(outputNames));

        List<IShorokooTensorValue>? borrowed = null;
        try
        {
            var feeds = new JaxTensorValue[_inputNames.Length];
            for (int i = 0; i < feeds.Length; i++)
            {
                if (!inputs.TryGetValue(_inputNames[i], out var value))
                    throw new ArgumentException($"The model's input '{_inputNames[i]}' was not fed.", nameof(inputs));
                if (value is JaxTensorValue own)
                {
                    feeds[i] = own;
                    continue;
                }
                // A value of another runtime is rebuilt on this one for the run, and released after it.
                var copy = (JaxTensorValue)BackendTransfer.CopyTo(_backend, value);
                (borrowed ??= []).Add(copy);
                feeds[i] = copy;
            }
            return Invoke(feeds, wanted, retainedOutputNames);
        }
        finally
        {
            if (borrowed is not null) foreach (var copy in borrowed) copy.Dispose();
        }
    }

    private IReadOnlyList<IShorokooTensorValue> Invoke(JaxTensorValue[] feeds, int[] wanted, IReadOnlySet<string> retainedOutputNames)
    {
        using (PythonRuntime.Gil())
        {
            using var args = new PyList();
            foreach (var feed in feeds) args.Append(feed.Value);
            using var wantedList = new PyList();
            using var retainedList = new PyList();
            foreach (var index in wanted)
            {
                PyCall.Append(wantedList, index);
                PyCall.Append(retainedList, _backend.OnCuda && retainedOutputNames.Contains(_outputNames[index]));
            }

            PyObject results;
            try
            {
                results = PyCall.Invoke(_runtime.Run, _model, args, wantedList, retainedList, (int)_logSeverity);
            }
            catch (PythonException ex)
            {
                throw Refusal(_backend, ex, "failed running the model in JAX");
            }
            GC.KeepAlive(feeds);

            using (results)
            {
                var outputs = new List<IShorokooTensorValue>(wanted.Length);
                try
                {
                    for (int i = 0; i < wanted.Length; i++)
                    {
                        using var pair = results[i];
                        using var description = pair[1];
                        outputs.Add(JaxTensorValue.Wrap(pair[0], description));
                    }
                }
                catch
                {
                    foreach (var output in outputs) output.Dispose();
                    throw;
                }
                return outputs;
            }
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        PythonRuntime.Release(_model);
    }
}
