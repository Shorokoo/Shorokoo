using System.Runtime.InteropServices;
using Python.Runtime;
using Shorokoo.Core.Backends;
using Shorokoo.PythonHost;
using Shorokoo.PythonTranslation;

namespace Shorokoo.PyTorch;

/// <summary>
/// The <see cref="IShorokooBackend"/> implementation backed by PyTorch: it translates each model it
/// is handed (serialized ONNX, as every backend receives) into Python calling PyTorch, and runs it
/// in the process's embedded CPython. It is device-neutral and abstract — <c>Shorokoo.PyTorch.Cpu</c>
/// and <c>Shorokoo.PyTorch.Cuda</c> subclass it for their device.
///
/// <para><b>Always named, never discovered.</b> A torch backend is never the one
/// <see cref="DefaultBackend"/> picks for a program that named none: it is used by handing it to a
/// compute context — <c>new ComputeContext(new TorchCpuBackend())</c> — so it can sit beside an
/// ONNX Runtime backend in one deployment without making discovery ambiguous.</para>
///
/// <para><b>Python starts lazily.</b> Constructing a backend touches nothing; the first call that
/// needs torch resolves the environment (see <see cref="PythonEnvironmentResolver"/>), provisioning
/// it if nothing names one, and starts the interpreter. A failure there is a
/// <see cref="PythonEnvironmentException"/> saying what is missing. <see cref="Start"/> does it
/// up front, for a program that would rather fail at startup.</para>
/// </summary>
public abstract class TorchBackend : IShorokooBackend
{
    private readonly Func<PythonEnvironmentLock> _lockFile;
    private readonly PythonEnvironmentOptions _options;
    private readonly int? _cudaDeviceId;
    private readonly object _gate = new();
    private TorchRuntime? _runtime;

    /// <summary>Creates a backend over the environment the lock <paramref name="lockFile"/> answers
    /// describes, on the CPU or on CUDA device <paramref name="cudaDeviceId"/>. The lock is asked for
    /// when the backend starts, not here, so that a platform no lock exists for is refused the way
    /// every other reason the environment cannot be had is.</summary>
    protected TorchBackend(Func<PythonEnvironmentLock> lockFile, PythonEnvironmentOptions? options, int? cudaDeviceId)
    {
        ArgumentNullException.ThrowIfNull(lockFile);
        _lockFile = lockFile;
        _options = options ?? new PythonEnvironmentOptions();
        _cudaDeviceId = cudaDeviceId;
        Description = new BackendDescription(
            GetType().Assembly.GetName().Name ?? GetType().Name,
            cudaDeviceId is null ? ComputeDevice.Cpu : ComputeDevice.Cuda,
            cudaDeviceId);
        DeviceName = cudaDeviceId is { } id ? $"cuda:{id}" : "cpu";
    }

    /// <summary>The assembly the concrete backend lives in, and its device.</summary>
    public BackendDescription Description { get; }

    /// <summary>Shared by every torch backend in the process, CPU and CUDA alike: they run one torch,
    /// so a tensor either makes is one the other's sessions can be handed as it is.</summary>
    public object RuntimeIdentity => TorchRuntime.Identity;

    /// <summary>Both formats: a step Shorokoo has differentiated, and one whose
    /// <c>ai.shorokoo.training::AutoGrad</c> node torch autograd differentiates
    /// (<see cref="TrainingFormats.OnnxAutoGrad"/>) — what a rig built with
    /// <see cref="TrainingBackend.Native"/> hands over.</summary>
    public bool AcceptsTrainingFormat(string format)
        => format is TrainingFormats.Onnx or TrainingFormats.OnnxAutoGrad;

    /// <summary>torch's name for this backend's device: <c>cpu</c> or <c>cuda:N</c>.</summary>
    public string DeviceName { get; }

    internal bool OnCuda => _cudaDeviceId is not null;

    /// <summary>The CUDA device this backend runs on, or -1 on the CPU.</summary>
    internal int CudaDeviceId => _cudaDeviceId ?? -1;

    /// <summary>
    /// Resolves the Python environment and starts PyTorch in it now, rather than on the first call
    /// that needs it, and returns the environment it runs in.
    /// </summary>
    /// <exception cref="PythonEnvironmentException">No environment could be resolved or started,
    /// or it cannot serve this backend's device.</exception>
    public PythonEnvironment Start() => Runtime.Environment;

    internal TorchRuntime Runtime
    {
        get
        {
            if (Volatile.Read(ref _runtime) is { } started) return started;
            lock (_gate)
            {
                if (_runtime is not null) return _runtime;
                PythonEnvironmentLock lockFile;
                try
                {
                    lockFile = _lockFile();
                }
                catch (PlatformNotSupportedException ex)
                {
                    throw new PythonEnvironmentException(PythonEnvironmentFailure.UnsupportedPlatform,
                        $"{Description} cannot start: {ex.Message}", ex);
                }
                // Before the environment is resolved, since resolving it can mean provisioning several
                // gigabytes of CUDA libraries for a card the machine turns out not to have.
                if (_cudaDeviceId is not null && MissingDriver() is { } missing)
                    throw new PythonEnvironmentException(PythonEnvironmentFailure.DeviceUnavailable,
                        $"{Description} cannot start: {missing}");
                var runtime = TorchRuntime.Start(lockFile, _options);
                if (_cudaDeviceId is { } device && device >= runtime.CudaDeviceCount)
                    throw new PythonEnvironmentException(PythonEnvironmentFailure.DeviceUnavailable,
                        $"{Description} needs CUDA device {device}, and torch {runtime.TorchVersion} in "
                        + $"'{runtime.Environment.Directory}' sees {runtime.CudaDeviceCount}. "
                        + (runtime.TorchCudaVersion.Length == 0
                            ? "That is a CPU build of torch, which sees no card: where a CPU torch backend "
                              + "started first, it chose the environment for the whole process, so create the "
                              + "CUDA backend first or name a CUDA environment with "
                              + $"{PythonEnvironmentResolver.EnvironmentVariable}."
                            : $"It is built for CUDA {runtime.TorchCudaVersion}; the NVIDIA driver loads, so the "
                              + "device number is past the cards this machine has, or they are hidden from "
                              + "the process (CUDA_VISIBLE_DEVICES)."));
                Volatile.Write(ref _runtime, runtime);
                return runtime;
            }
        }
    }

    /// <summary>
    /// Why this machine's NVIDIA driver cannot serve this backend — none installed, too old for the
    /// CUDA its manifest names, or no device — or null where it can, or where the assembly cannot be
    /// read to ask (a single-file deployment has no path to probe), which leaves the answer to torch.
    /// </summary>
    private string? MissingDriver()
    {
        var location = GetType().Assembly.Location;
        if (string.IsNullOrEmpty(location)) return null;
        var probe = BackendPackage.Probe(location);
        return probe.Reason == BackendRejection.MissingCudaDriver ? probe.Detail : null;
    }

    /// <summary>Creates a session over a serialized ONNX model. The model is translated and
    /// checked here, so a model this backend cannot run is refused now, naming what it cannot
    /// run, rather than on its first run.</summary>
    /// <param name="modelBytes">The serialized ONNX model.</param>
    /// <param name="graphOptimization">Unused: torch runs the graph as translated.</param>
    /// <param name="logSeverity">The least severity at which a warning the session's runs raise
    /// in Python is shown; above <see cref="ShorokooLogSeverity.Warning"/> they are not.</param>
    /// <param name="deviceMemory">On CUDA, the limit each run's allocations get
    /// (<see cref="DeviceMemorySettings.LimitBytes"/>); ignored on the CPU.</param>
    /// <exception cref="TorchUnsupportedModelException">The model uses something this backend
    /// cannot run.</exception>
    public IShorokooSession CreateSession(
        ReadOnlyMemory<byte> modelBytes,
        ShorokooGraphOptimization graphOptimization,
        ShorokooLogSeverity logSeverity,
        DeviceMemorySettings deviceMemory)
        => TorchSession.Create(this, modelBytes, logSeverity, deviceMemory, DiagnosticSettings.Default, []);

    /// <summary>The same session, recording which device ran each node where
    /// <paramref name="diagnostics"/> asks (<see cref="DiagnosticSettings.TraceNodePlacement"/>):
    /// every node runs on this backend's device, so the record costs the runs nothing.</summary>
    public IShorokooSession CreateSession(
        ReadOnlyMemory<byte> modelBytes,
        ShorokooGraphOptimization graphOptimization,
        ShorokooLogSeverity logSeverity,
        DeviceMemorySettings deviceMemory,
        DiagnosticSettings diagnostics)
        => TorchSession.Create(this, modelBytes, logSeverity, deviceMemory, diagnostics, []);

    /// <summary>
    /// The same session, writing the outputs <paramref name="outputAliases"/> names into the memory
    /// of the inputs it pairs them with on a run that consumed those inputs, where the model proves
    /// the pair (<see cref="OutputAliasProof"/> — torch runs the graph as handed over, so the proof
    /// over it stands) and torch can make the write: see <see cref="TorchSession"/>.
    /// </summary>
    public IShorokooSession CreateSession(
        ReadOnlyMemory<byte> modelBytes,
        ShorokooGraphOptimization graphOptimization,
        ShorokooLogSeverity logSeverity,
        DeviceMemorySettings deviceMemory,
        DiagnosticSettings diagnostics,
        IReadOnlyList<OutputAlias> outputAliases)
    {
        ArgumentNullException.ThrowIfNull(outputAliases);
        return TorchSession.Create(this, modelBytes, logSeverity, deviceMemory, diagnostics, outputAliases);
    }

    public IShorokooTensorValue CreateTensor<T>(T[] data, long[] shape) where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(data);
        return FromHost(PythonElementTypes.Of<T>(), MemoryMarshal.AsBytes(data.AsSpan()), shape, "cpu");
    }

    /// <summary>A tensor of these bytes, in host memory whatever this backend's device —
    /// <see cref="CreateTensorInBackendMemory"/> is the one that builds it on the device.</summary>
    public IShorokooTensorValue CreateTensorFromRawBytes(
        ShorokooTensorElementType elementType, byte[] data, long[] shape)
    {
        ArgumentNullException.ThrowIfNull(data);
        return FromHost(elementType, data, shape, "cpu");
    }

    /// <summary>A tensor of these bytes on this backend's device: host memory on the CPU, the
    /// card's own on CUDA.</summary>
    public IShorokooTensorValue CreateTensorInBackendMemory(
        ShorokooTensorElementType elementType, byte[] data, long[] shape)
    {
        ArgumentNullException.ThrowIfNull(data);
        return FromHost(elementType, data, shape, DeviceName);
    }

    /// <summary>An uninitialized tensor on this backend's device.</summary>
    public IShorokooTensorValue CreateUninitializedTensorInBackendMemory(
        ShorokooTensorElementType elementType, long[] shape)
    {
        ArgumentNullException.ThrowIfNull(shape);
        PythonElementTypes.ByteCount(elementType, shape);
        var runtime = Runtime;
        using (PythonRuntime.Gil())
        {
            using var dims = Shape(shape);
            var tensor = PyCall.Invoke(runtime.Empty, (int)elementType, dims, DeviceName);
            return TorchTensorValue.Wrap(runtime, tensor, elementType);
        }
    }

    private unsafe TorchTensorValue FromHost(
        ShorokooTensorElementType elementType, ReadOnlySpan<byte> bytes, long[] shape, string device)
    {
        ArgumentNullException.ThrowIfNull(shape);
        if (elementType == ShorokooTensorElementType.String)
            throw new NotSupportedException(
                "String tensors are variable-length and not byte-stride; use CreateStringTensor instead.");
        var byteCount = PythonElementTypes.ByteCount(elementType, shape);
        if (bytes.Length < byteCount)
            throw new ArgumentException(
                $"Supplied data of {bytes.Length} bytes is less than shape size {byteCount} bytes.", nameof(bytes));
        var runtime = Runtime;
        using (PythonRuntime.Gil())
        {
            fixed (byte* source = bytes)
            {
                // The bytes are pinned for exactly the call that copies them: from_host copies into
                // a tensor torch allocated, so nothing keeps pointing into the managed buffer after.
                using var dims = Shape(shape);
                var tensor = PyCall.Invoke(runtime.FromHost, (long)source, (long)byteCount, (int)elementType, dims, device);
                return TorchTensorValue.Wrap(runtime, tensor, elementType);
            }
        }
    }

    public IShorokooTensorValue CreateStringTensor(IReadOnlyList<string> data, long[] shape)
    {
        ArgumentNullException.ThrowIfNull(data);
        ArgumentNullException.ThrowIfNull(shape);
        var runtime = Runtime;
        using (PythonRuntime.Gil())
        {
            using var values = new PyList();
            foreach (var value in data)
            {
                ArgumentNullException.ThrowIfNull(value, nameof(data));
                PyCall.Append(values, value);
            }
            using var dims = Shape(shape);
            var array = runtime.Strings.Invoke(values, dims);
            return TorchTensorValue.Wrap(runtime, array, ShorokooTensorElementType.String);
        }
    }

    /// <summary>
    /// A sequence of <paramref name="values"/>, which it takes over: on success the sequence holds
    /// them and each value handed over refuses every read from then on; on failure every one is
    /// released before this throws. A value of another runtime is copied in, and released too.
    /// </summary>
    public IShorokooTensorValue CreateSequence(IReadOnlyList<IShorokooTensorValue> values)
    {
        ArgumentNullException.ThrowIfNull(values);
        try
        {
            var elementType = values.Count > 0 ? values[0].ElementType : ShorokooTensorElementType.Float;
            var runtime = Runtime;
            var own = new List<TorchTensorValue>(values.Count);
            List<IShorokooTensorValue> copies = [];
            try
            {
                foreach (var value in values)
                {
                    if (value is TorchTensorValue torch) { own.Add(torch); continue; }
                    var copy = (TorchTensorValue)BackendTransfer.CopyTo(this, value);
                    copies.Add(copy);
                    own.Add(copy);
                }
                PyObject list;
                using (PythonRuntime.Gil())
                {
                    list = new PyList();
                    foreach (var value in own) ((PyList)list).Append(value.Value);
                }
                var sequence = WrapSequence(runtime, list, elementType);
                foreach (var value in own) value.HandedOver();
                return sequence;
            }
            finally
            {
                foreach (var copy in copies) copy.Dispose();
            }
        }
        finally
        {
            // Every value handed over is released here, success or not: on success its tensor lives
            // on in the sequence's list, which holds a reference of its own.
            foreach (var value in values) value.Dispose();
        }
    }

    private static TorchTensorValue WrapSequence(TorchRuntime runtime, PyObject list, ShorokooTensorElementType elementType)
    {
        using (PythonRuntime.Gil())
            return TorchTensorValue.Wrap(runtime, list, elementType);
    }

    /// <summary>Releases a value this backend made: drops its reference to the torch object, under
    /// the interpreter lock. Every release of a torch backend's memory comes here, a consumed run
    /// input's included.</summary>
    public void Release(IShorokooTensorValue value)
    {
        ArgumentNullException.ThrowIfNull(value);
        value.Dispose();
    }

    /// <summary>A tensor's contents as host bytes, whatever memory it is in: a CUDA tensor is
    /// brought home with torch's own copy.</summary>
    public byte[] CopyTensorToHost(IShorokooTensorValue value)
    {
        ArgumentNullException.ThrowIfNull(value);
        if (value is not TorchTensorValue torch)
        {
            if (!value.IsHostAccessible)
                throw new InvalidOperationException(
                    $"A {value.GetType().Name} in device memory did not come from {Description}, so it cannot be read here.");
            var bytes = value.GetTensorDataAsSpan<byte>().ToArray();
            GC.KeepAlive(value);
            return bytes;
        }
        if (torch.ValueType != ShorokooOnnxValueType.Tensor || torch.ElementType == ShorokooTensorElementType.String)
            throw new InvalidOperationException(
                $"Only a fixed-stride tensor can be read back as bytes; this is a {torch.ValueType} of {torch.ElementType}.");
        if (torch.IsHostAccessible)
        {
            var bytes = torch.GetTensorDataAsSpan<byte>().ToArray();
            GC.KeepAlive(torch);
            return bytes;
        }
        var runtime = Runtime;
        using (PythonRuntime.Gil())
        {
            using var host = TorchTensorValue.Wrap(runtime, runtime.HostCopy.Invoke(torch.Value), torch.ElementType);
            return host.GetTensorDataAsSpan<byte>().ToArray();
        }
    }

    internal static PyList Shape(long[] shape)
    {
        var list = new PyList();
        foreach (var dim in shape) PyCall.Append(list, dim);
        return list;
    }
}
