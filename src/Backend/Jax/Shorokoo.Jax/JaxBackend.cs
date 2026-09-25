using System.Runtime.InteropServices;
using Python.Runtime;
using Shorokoo.Core.Backends;
using Shorokoo.PythonHost;
using Shorokoo.PythonTranslation;

namespace Shorokoo.Jax;

/// <summary>
/// The <see cref="IShorokooBackend"/> implementation backed by JAX: it translates each model it is
/// handed (serialized ONNX, as every backend receives) into Python calling <c>jax.numpy</c> and
/// <c>jax.lax</c>, and runs it in the process's embedded CPython, compiled by XLA once per signature
/// of input shapes and element types. It is device-neutral and abstract — <c>Shorokoo.Jax.Cpu</c>
/// and <c>Shorokoo.Jax.Cuda</c> subclass it for their device.
///
/// <para><b>Training.</b> It accepts a training step whose gradient is left to it
/// (<see cref="TrainingFormats.OnnxAutoGrad"/>, what a rig built with
/// <see cref="TrainingBackend.Native"/> hands over): the gradient is <c>jax.value_and_grad</c> of the
/// step's forward pass, and XLA compiles the forward pass, the backward pass and the optimizer
/// update as one program.</para>
///
/// <para><b>Always named, never discovered.</b> A JAX backend is never the one
/// <see cref="DefaultBackend"/> picks for a program that named none: it is used by handing it to a
/// compute context — <c>new ComputeContext(new JaxCpuBackend())</c>.</para>
///
/// <para><b>Python starts lazily.</b> Constructing a backend touches nothing; the first call that
/// needs JAX resolves the environment (see <see cref="PythonEnvironmentResolver"/>), provisioning it
/// if nothing names one, and starts the interpreter. <see cref="Start"/> does it up front.</para>
/// </summary>
public abstract class JaxBackend : IShorokooBackend
{
    private readonly Func<PythonEnvironmentLock> _lockFile;
    private readonly PythonEnvironmentOptions _options;
    private readonly int? _cudaDeviceId;
    private readonly object _gate = new();
    private JaxRuntime? _runtime;

    /// <summary>Creates a backend over the environment the lock <paramref name="lockFile"/> answers
    /// describes, on the CPU or on CUDA device <paramref name="cudaDeviceId"/>. The lock is asked for
    /// when the backend starts, not here.</summary>
    protected JaxBackend(Func<PythonEnvironmentLock> lockFile, PythonEnvironmentOptions? options, int? cudaDeviceId)
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

    /// <summary>Shared by every JAX backend in the process, CPU and CUDA alike: they run one JAX,
    /// so a value either makes is one the other's sessions can be handed as it is.</summary>
    public object RuntimeIdentity => JaxRuntime.Identity;

    /// <summary>Both formats: a step Shorokoo has differentiated, and one whose
    /// <c>ai.shorokoo.training::AutoGrad</c> node JAX differentiates
    /// (<see cref="TrainingFormats.OnnxAutoGrad"/>).</summary>
    public bool AcceptsTrainingFormat(string format)
        => format is TrainingFormats.Onnx or TrainingFormats.OnnxAutoGrad;

    /// <summary>The backend's device as the support package names it: <c>cpu</c> or <c>cuda:N</c>.</summary>
    public string DeviceName { get; }

    internal bool OnCuda => _cudaDeviceId is not null;

    /// <summary>
    /// Resolves the Python environment and starts JAX in it now, rather than on the first call that
    /// needs it, and returns the environment it runs in.
    /// </summary>
    /// <exception cref="PythonEnvironmentException">No environment could be resolved or started,
    /// or it cannot serve this backend's device.</exception>
    public PythonEnvironment Start() => Runtime.Environment;

    internal JaxRuntime Runtime
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
                if (_cudaDeviceId is not null && Probe() is { Reason: var reason and (BackendRejection.WrongOperatingSystem or BackendRejection.MissingCudaDriver) } probe)
                    throw new PythonEnvironmentException(
                        reason == BackendRejection.WrongOperatingSystem ? PythonEnvironmentFailure.UnsupportedPlatform : PythonEnvironmentFailure.DeviceUnavailable,
                        $"{Description} cannot start: {probe.Detail}");
                var runtime = JaxRuntime.Start(lockFile, _options);
                if (_cudaDeviceId is { } device && device >= runtime.CudaDeviceCount)
                    throw new PythonEnvironmentException(PythonEnvironmentFailure.DeviceUnavailable,
                        $"{Description} needs CUDA device {device}, and JAX {runtime.JaxVersion} in "
                        + $"'{runtime.Environment.Directory}' sees {runtime.CudaDeviceCount}. Where it sees none, the "
                        + "environment has no JAX CUDA plugin -- a CPU environment, which a CPU backend that started "
                        + "first chose for the whole process: create the CUDA backend first, or name a CUDA "
                        + $"environment with {PythonEnvironmentResolver.EnvironmentVariable} -- or the card is hidden "
                        + "from the process (CUDA_VISIBLE_DEVICES).");
                Volatile.Write(ref _runtime, runtime);
                return runtime;
            }
        }
    }

    /// <summary>What the probe of the concrete backend's assembly says of this machine — which
    /// operating systems it runs on, and whether the NVIDIA driver can serve it — or null where the
    /// assembly cannot be read to ask (a single-file deployment has no path to probe), which leaves
    /// the answer to JAX.</summary>
    private BackendProbe? Probe()
    {
        var location = GetType().Assembly.Location;
        return string.IsNullOrEmpty(location) ? null : BackendPackage.Probe(location);
    }

    /// <summary>Creates a session over a serialized ONNX model. The model is translated and checked
    /// here, and where its inputs have fixed shapes compiled too, so a model this backend cannot
    /// run is refused now, naming what it cannot run.</summary>
    /// <param name="modelBytes">The serialized ONNX model.</param>
    /// <param name="graphOptimization">Unused: XLA optimizes the program it compiles.</param>
    /// <param name="logSeverity">The least severity at which a warning the session's runs raise in
    /// Python is shown.</param>
    /// <param name="deviceMemory">Unused: JAX's device allocator is the whole process's and takes no
    /// per-run limit.</param>
    /// <exception cref="JaxUnsupportedModelException">The model uses something this backend cannot
    /// run.</exception>
    public IShorokooSession CreateSession(
        ReadOnlyMemory<byte> modelBytes,
        ShorokooGraphOptimization graphOptimization,
        ShorokooLogSeverity logSeverity,
        DeviceMemorySettings deviceMemory)
        => JaxSession.Create(this, modelBytes, logSeverity, DiagnosticSettings.Default);

    /// <summary>The same session, recording which device ran each node where
    /// <paramref name="diagnostics"/> asks: every node runs on this backend's device.</summary>
    public IShorokooSession CreateSession(
        ReadOnlyMemory<byte> modelBytes,
        ShorokooGraphOptimization graphOptimization,
        ShorokooLogSeverity logSeverity,
        DeviceMemorySettings deviceMemory,
        DiagnosticSettings diagnostics)
        => JaxSession.Create(this, modelBytes, logSeverity, diagnostics);

    /// <summary>The same session: a JAX array is never written in place, so the session binds none
    /// of <paramref name="outputAliases"/> (<see cref="IShorokooSession.BindableAliases"/> is
    /// empty) and every output is memory of its own.</summary>
    public IShorokooSession CreateSession(
        ReadOnlyMemory<byte> modelBytes,
        ShorokooGraphOptimization graphOptimization,
        ShorokooLogSeverity logSeverity,
        DeviceMemorySettings deviceMemory,
        DiagnosticSettings diagnostics,
        IReadOnlyList<OutputAlias> outputAliases)
    {
        ArgumentNullException.ThrowIfNull(outputAliases);
        return JaxSession.Create(this, modelBytes, logSeverity, diagnostics);
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

    /// <summary>A tensor on this backend's device whose contents are unspecified: zeros, since a JAX
    /// array is never uninitialized.</summary>
    public IShorokooTensorValue CreateUninitializedTensorInBackendMemory(
        ShorokooTensorElementType elementType, long[] shape)
    {
        ArgumentNullException.ThrowIfNull(shape);
        RefuseStrings(elementType);
        PythonElementTypes.ByteCount(elementType, shape);
        var runtime = Runtime;
        using (PythonRuntime.Gil())
        {
            using var dims = Shape(shape);
            var tensor = PyCall.Invoke(runtime.Empty, (int)elementType, dims, DeviceName);
            return JaxTensorValue.Wrap(runtime, tensor);
        }
    }

    private unsafe JaxTensorValue FromHost(
        ShorokooTensorElementType elementType, ReadOnlySpan<byte> bytes, long[] shape, string device)
    {
        ArgumentNullException.ThrowIfNull(shape);
        RefuseStrings(elementType);
        var byteCount = PythonElementTypes.ByteCount(elementType, shape);
        if (bytes.Length < byteCount)
            throw new ArgumentException(
                $"Supplied data of {bytes.Length} bytes is less than shape size {byteCount} bytes.", nameof(bytes));
        var runtime = Runtime;
        using (PythonRuntime.Gil())
        {
            fixed (byte* source = bytes)
            {
                // The bytes are pinned for exactly the call that copies them: from_host copies into an
                // array it allocated, so nothing keeps pointing into the managed buffer after.
                using var dims = Shape(shape);
                var tensor = PyCall.Invoke(runtime.FromHost, (long)source, (long)byteCount, (int)elementType, dims, device);
                return JaxTensorValue.Wrap(runtime, tensor);
            }
        }
    }

    private static void RefuseStrings(ShorokooTensorElementType elementType)
    {
        if (elementType == ShorokooTensorElementType.String)
            throw new NotSupportedException("JAX has no string tensors, so the JAX backend cannot hold one.");
    }

    /// <summary>Refused: JAX has no string tensors.</summary>
    public IShorokooTensorValue CreateStringTensor(IReadOnlyList<string> data, long[] shape)
        => throw new NotSupportedException("JAX has no string tensors, so the JAX backend cannot hold one.");

    /// <summary>Refused: JAX has no sequences. Every value handed over is released before this
    /// throws, as a backend that takes a sequence's values over releases them on failure.</summary>
    public IShorokooTensorValue CreateSequence(IReadOnlyList<IShorokooTensorValue> values)
    {
        ArgumentNullException.ThrowIfNull(values);
        foreach (var value in values) value.Dispose();
        throw new NotSupportedException("JAX has no sequences, so the JAX backend cannot hold one.");
    }

    /// <summary>Releases a value this backend made: drops its reference to the array, under the
    /// interpreter lock.</summary>
    public void Release(IShorokooTensorValue value)
    {
        ArgumentNullException.ThrowIfNull(value);
        value.Dispose();
    }

    /// <summary>A tensor's contents as host bytes, whatever memory it is in: a device array is
    /// brought home by JAX.</summary>
    public byte[] CopyTensorToHost(IShorokooTensorValue value)
    {
        ArgumentNullException.ThrowIfNull(value);
        if (value is not JaxTensorValue jax)
        {
            if (!value.IsHostAccessible)
                throw new InvalidOperationException(
                    $"A {value.GetType().Name} in device memory did not come from {Description}, so it cannot be read here.");
            var bytes = value.GetTensorDataAsSpan<byte>().ToArray();
            GC.KeepAlive(value);
            return bytes;
        }
        if (jax.IsHostAccessible)
        {
            var bytes = jax.GetTensorDataAsSpan<byte>().ToArray();
            GC.KeepAlive(jax);
            return bytes;
        }
        var runtime = Runtime;
        using (PythonRuntime.Gil())
        {
            using var host = JaxTensorValue.Wrap(runtime, runtime.HostCopy.Invoke(jax.Value));
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
