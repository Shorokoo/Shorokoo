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
/// <para>Every output is handed over as memory of its own, where the backend's runs leave their
/// outputs: on a card, an array no other value shares a write to; on the CPU, a host array of its
/// own. A JAX array is never written in place, so a run writes no output into a consumed input
/// (<see cref="BindableAliases"/> is empty).</para>
///
/// <para><b>Stopping.</b> A run is one compiled program, which cannot be stopped part way: a run
/// whose token is cancelled before it starts is refused, and one cancelled while it runs finishes.</para>
///
/// <para><b>Precision.</b> The precision of every product and convolution is compiled into the
/// program, from the <see cref="PrecisionSettings"/> the session was built with: full <c>float32</c>
/// precision, unless it allows TensorFloat-32 and the session is on a card (see
/// <see cref="Float32Precision"/>).</para>
/// </summary>
internal sealed class JaxSession : IShorokooSession
{
    private readonly JaxBackend _backend;
    private readonly JaxRuntime _runtime;
    private readonly string[] _inputNames;
    private readonly string[] _outputNames;
    private readonly Dictionary<string, int> _outputIndex;
    private readonly NodePlacement? _nodePlacement;

    // The loaded model: its code, its constants and the programs compiled from it. Held for the
    // session's life: it is what the model is.
    private readonly PyObject _model;
    private int _disposed;

    private JaxSession(
        JaxBackend backend, JaxRuntime runtime, TranslatedModel model,
        NodePlacement? nodePlacement, PyObject loaded)
    {
        _backend = backend;
        _runtime = runtime;
        _inputNames = model.InputNames;
        _outputNames = model.OutputNames;
        _outputIndex = new Dictionary<string, int>(StringComparer.Ordinal);
        for (int i = 0; i < _outputNames.Length; i++) _outputIndex.TryAdd(_outputNames[i], i);
        _nodePlacement = nodePlacement;
        _model = loaded;
    }

    /// <summary>The precision XLA compiles the products and convolutions of a session of
    /// <paramref name="backend"/> in, as JAX names it: <c>HIGH</c> — TensorFloat-32 for a
    /// <c>float32</c> product on a card that has it — where <paramref name="precision"/> allows
    /// TensorFloat-32 and the backend is on a card, and otherwise <c>HIGHEST</c>, full
    /// precision.</summary>
    internal static string Float32Precision(JaxBackend backend, PrecisionSettings precision)
        => backend.OnCuda && precision.AllowTensorFloat32 ? "HIGH" : "HIGHEST";

    /// <summary>Translates <paramref name="modelBytes"/>, loads it, and compiles it where its inputs'
    /// shapes are fixed.</summary>
    public static JaxSession Create(
        JaxBackend backend, ReadOnlyMemory<byte> modelBytes, LogSettings log, DiagnosticSettings diagnostics,
        PrecisionSettings precision)
    {
        ArgumentNullException.ThrowIfNull(log);
        ArgumentNullException.ThrowIfNull(diagnostics);
        ArgumentNullException.ThrowIfNull(precision);
        ModelProto proto;
        using (var stream = new MemoryStream(modelBytes.ToArray(), writable: false))
            proto = Shorokoo.Onnx.OnnxProtobuf.ReadModel(stream);
        // Translated before JAX is started, so that a model this backend cannot run is refused
        // without first provisioning an environment to not run it in.
        var model = OnnxToPythonTranslator.Translate(proto, [], JaxDialect.Instance);
        var hash = Convert.ToHexStringLower(SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(model.Source)))[..32];
        var placement = diagnostics.TraceNodePlacement ? Placement(proto.Graph!, backend.DeviceName) : null;
        var signature = FixedSignature(proto.Graph!, model.InputNames);

        var runtime = backend.Runtime;
        RuntimeLogMessage[] warnings = [];
        try
        {
            return Load(backend, runtime, model, hash, placement, signature, precision, out warnings);
        }
        finally
        {
            PythonWarnings.Deliver(log, warnings);
        }
    }

    private static JaxSession Load(
        JaxBackend backend, JaxRuntime runtime, TranslatedModel model, string hash, NodePlacement? placement,
        List<(ShorokooTensorElementType, long[])>? signature, PrecisionSettings precision,
        out RuntimeLogMessage[] warnings)
    {
        warnings = [];
        using (PythonRuntime.Gil())
        {
            using var warned = new PyList();
            PyObject? loaded = null;
            try
            {
                using var constants = new PyList();
                foreach (var constant in model.Constants)
                {
                    using var value = ConstantValue(runtime, constant);
                    constants.Append(value);
                }
                loaded = PyCall.Invoke(runtime.LoadModel, model.Source, $"<shorokoo-model-{hash}>", constants, backend.DeviceName,
                    Float32Precision(backend, precision));
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
                    try
                    {
                        PyCall.Invoke(runtime.Prepare, loaded, inputs, warned).Dispose();
                    }
                    finally
                    {
                        warnings = PythonWarnings.Read(warned, JaxRuntime.Source);
                    }
                }
                return new JaxSession(backend, runtime, model, placement, loaded);
            }
            catch (PythonException ex)
            {
                loaded?.Dispose();
                throw Refusal(backend, ex, "could not compile the translated model");
            }
        }
    }

    /// <summary>A failure JAX raised, as the backend reports it: a model that needs, as a number,
    /// something its inputs' values decide, or anything else JAX does not implement — a mode, a
    /// gradient — is one the backend cannot run (<see cref="JaxUnsupportedModelException"/>);
    /// anything else is a failure of the run.</summary>
    private static Exception Refusal(JaxBackend backend, PythonException ex, string what)
    {
        if (ex.Type.Name == "NotImplementedError")
            return new JaxUnsupportedModelException(JaxUnsupportedReason.UnsupportedUsage, null, null,
                $"The JAX backend cannot run the model: {ex.Message}", ex);
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
            return new ArenaStatistics(At(0), At(1), At(2), At(3), At(4), At(5), At(6), At(7), At(8), At(9));
        }
    }

    /// <summary>Runs the model on <paramref name="inputs"/>, every one of them where this backend's
    /// runs read it, and leaves every output there: on the card for a CUDA session, and in host
    /// memory on the CPU.</summary>
    public IReadOnlyList<IShorokooTensorValue> Run(
        IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
        IReadOnlyList<string> outputNames,
        RunSettings runSettings)
    {
        ArgumentNullException.ThrowIfNull(inputs);
        ArgumentNullException.ThrowIfNull(outputNames);
        ArgumentNullException.ThrowIfNull(runSettings);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        runSettings.CancellationToken.ThrowIfCancellationRequested();

        var wanted = new int[outputNames.Count];
        for (int i = 0; i < wanted.Length; i++)
            wanted[i] = _outputIndex.TryGetValue(outputNames[i], out var index)
                ? index
                : throw new ArgumentException($"The model has no output '{outputNames[i]}'.", nameof(outputNames));

        var feeds = new JaxTensorValue[_inputNames.Length];
        for (int i = 0; i < feeds.Length; i++)
        {
            if (!inputs.TryGetValue(_inputNames[i], out var value))
                throw new ArgumentException($"The model's input '{_inputNames[i]}' was not fed.", nameof(inputs));
            feeds[i] = Fed(_inputNames[i], value);
        }
        return Invoke(feeds, wanted, runSettings.Log);
    }

    /// <summary>Runs with <paramref name="consumed"/> handed over: each is released through the
    /// backend, exactly once, however the run ends. Nothing is written into them.</summary>
    public IReadOnlyList<IShorokooTensorValue> RunConsuming(
        IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
        IReadOnlyCollection<IShorokooTensorValue> consumed,
        IReadOnlyList<string> outputNames,
        RunSettings runSettings)
    {
        ArgumentNullException.ThrowIfNull(consumed);
        try
        {
            return Run(inputs, outputNames, runSettings);
        }
        finally
        {
            foreach (var value in consumed) _backend.Release(value);
        }
    }

    /// <summary>
    /// <paramref name="value"/>, fed as input <paramref name="name"/>, where it is one of this
    /// runtime's in the memory this session's runs read it in: on this session's card for a CUDA
    /// session, and in host memory on the CPU. Refused otherwise, before the run starts: nothing here
    /// moves a value, the framework placing every input there before it hands it over
    /// (<see cref="IShorokooBackend.RunMemoryOf"/>).
    /// </summary>
    /// <exception cref="InvalidOperationException">The value is not this runtime's, or not where this
    /// session's runs read it.</exception>
    private JaxTensorValue Fed(string name, IShorokooTensorValue value)
    {
        if (value is JaxTensorValue own && (_backend.OnCuda ? own.Device == _backend.CudaDeviceId : own.IsHostAccessible))
            return own;
        throw new InvalidOperationException(
            $"Input '{name}' was handed to a session of {_backend.Description} as "
            + (value is JaxTensorValue jax
                ? $"a {jax.ElementType} tensor in " + (jax.IsHostAccessible ? "host memory" : $"device {jax.Device}'s memory")
                : $"a {value.GetType().Name}, which is not a value of this runtime")
            + ", outside the memory its runs read it in. A session takes only values its backend's runs "
            + "read where they are, and moves none: place the input there first, through the backend's "
            + "own moves (IShorokooBackend.CreateTensorInBackendMemory).");
    }

    /// <summary>The run, with the Python warnings it raised delivered to <paramref name="log"/> once
    /// it is over, however it ended (see <see cref="PythonWarnings"/>).</summary>
    private IReadOnlyList<IShorokooTensorValue> Invoke(JaxTensorValue[] feeds, int[] wanted, LogSettings log)
    {
        RuntimeLogMessage[] warnings = [];
        try
        {
            using (PythonRuntime.Gil())
            {
                using var warned = new PyList();
                try
                {
                    return InvokeWarning(feeds, wanted, warned);
                }
                finally
                {
                    warnings = PythonWarnings.Read(warned, JaxRuntime.Source);
                }
            }
        }
        finally
        {
            PythonWarnings.Deliver(log, warnings);
        }
    }

    private IReadOnlyList<IShorokooTensorValue> InvokeWarning(JaxTensorValue[] feeds, int[] wanted, PyList warned)
    {
        using (PythonRuntime.Gil())
        {
            using var args = new PyList();
            foreach (var feed in feeds) args.Append(feed.Value);
            using var wantedList = new PyList();
            foreach (var index in wanted) PyCall.Append(wantedList, index);

            PyObject results;
            try
            {
                results = PyCall.Invoke(_runtime.Run, _model, args, wantedList, warned);
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
