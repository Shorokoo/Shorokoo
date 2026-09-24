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
/// <para>Every output is handed over as memory of its own: never an input's, a constant's or
/// another output's, even where the model's graph returns one of those as it is (an
/// <c>Identity</c>, a <c>Reshape</c> view), since the caller owns what it is handed and may write
/// to it. Nothing is aliased into a consumed input; a run consumes by releasing.</para>
/// </summary>
internal sealed class TorchSession : IShorokooSession
{
    private readonly TorchBackend _backend;
    private readonly TorchRuntime _runtime;
    private readonly string[] _inputNames;
    private readonly string[] _outputNames;
    private readonly ShorokooTensorElementType[] _outputSequenceTypes;
    private readonly Dictionary<string, int> _outputIndex;

    // The loaded model's entry point and its constants, and what the run needs to keep outputs off
    // the constants' memory. Held in fields for the session's life: they are what the model is.
    private readonly PyObject _main;
    private readonly PyObject _constants;
    private readonly PyObject _constantStorages;
    private readonly PyObject _constantIds;
    private int _disposed;

    private TorchSession(
        TorchBackend backend, TorchRuntime runtime, TranslatedModel model,
        PyObject main, PyObject constants, PyObject constantStorages, PyObject constantIds)
    {
        _backend = backend;
        _runtime = runtime;
        _inputNames = model.InputNames;
        _outputNames = model.OutputNames;
        _outputSequenceTypes = model.OutputSequenceElementTypes;
        _outputIndex = new Dictionary<string, int>(StringComparer.Ordinal);
        for (int i = 0; i < _outputNames.Length; i++) _outputIndex.TryAdd(_outputNames[i], i);
        _main = main;
        _constants = constants;
        _constantStorages = constantStorages;
        _constantIds = constantIds;
    }

    /// <summary>Translates <paramref name="modelBytes"/> and loads it.</summary>
    public static TorchSession Create(TorchBackend backend, ReadOnlyMemory<byte> modelBytes)
    {
        ModelProto proto;
        using (var stream = new MemoryStream(modelBytes.ToArray(), writable: false))
            proto = ProtoBuf.Serializer.Deserialize<ModelProto>(stream);
        // Translated before torch is started, so that a model this backend cannot run is refused
        // without first provisioning an environment to not run it in.
        var model = OnnxToPythonTranslator.Translate(proto);
        var hash = Convert.ToHexStringLower(SHA256.HashData(modelBytes.Span))[..32];

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
                return new TorchSession(backend, runtime, model, main, constants,
                    runtime.ConstantStorages.Invoke(constants), runtime.ConstantIds.Invoke(constants));
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

    public IReadOnlyList<string> InputNames => _inputNames;

    public IReadOnlyList<string> OutputNames => _outputNames;

    /// <summary>A CUDA session computes on the card, and can leave outputs there.</summary>
    public bool HasDeviceMemory => _backend.OnCuda;

    public SessionOutputPlacement OutputPlacement
        => _backend.OnCuda ? SessionOutputPlacement.Device : SessionOutputPlacement.Host;

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

    /// <summary>
    /// Runs with <paramref name="consumed"/> handed over: each is released through the backend,
    /// exactly once, however the run ends — returning, throwing, refused before it starts. The
    /// run reads its inputs before it produces anything, and nothing it returns is in their memory,
    /// so releasing them as it returns frees nothing an output needs.
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
            return RunCore(inputs, outputNames, retainedOutputNames, runSettings);
        }
        finally
        {
            foreach (var value in consumed) _backend.Release(value);
        }
    }

    public IReadOnlyList<IShorokooTensorValue> RunConsuming(
        IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
        IReadOnlyCollection<IShorokooTensorValue> consumed,
        IReadOnlyList<string> outputNames,
        IReadOnlySet<string> retainedOutputNames,
        RunSettings runSettings,
        out IReadOnlyList<string?> aliasedInputs)
    {
        aliasedInputs = [];
        return RunConsuming(inputs, consumed, outputNames, retainedOutputNames, runSettings);
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
        // A run is one call into Python with no point to stop at part-way, so a cancellation is
        // honoured before it starts; one that arrives during the run is seen after it, which the
        // contract allows.
        runSettings.CancellationToken.ThrowIfCancellationRequested();

        var wanted = new int[outputNames.Count];
        for (int i = 0; i < wanted.Length; i++)
            wanted[i] = _outputIndex.TryGetValue(outputNames[i], out var index)
                ? index
                : throw new ArgumentException($"The model has no output '{outputNames[i]}'.", nameof(outputNames));

        List<IShorokooTensorValue>? borrowed = null;
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

                PyObject results;
                try
                {
                    results = _runtime.Run.Invoke(
                        _main, args, wantedList, retainedList, new PyString(_backend.DeviceName),
                        _constantStorages, _constantIds);
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
                    try
                    {
                        for (int i = 0; i < wanted.Length; i++)
                        {
                            using var pair = results[i];
                            using var description = pair[1];
                            outputs.Add(TorchTensorValue.Wrap(pair[0], description, _outputSequenceTypes[wanted[i]]));
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
        finally
        {
            if (borrowed is not null) foreach (var copy in borrowed) copy.Dispose();
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
