namespace Shorokoo.Core.Inference.Abstractions;

/// <summary>The kind of device a backend executes on.</summary>
public enum ComputeDevice
{
    /// <summary>The host CPU — ONNX Runtime's default execution provider.</summary>
    Cpu,

    /// <summary>An NVIDIA GPU, through ONNX Runtime's CUDA execution provider.</summary>
    Cuda,

    /// <summary>
    /// Some other execution provider — DirectML, ROCm, CoreML, OpenVINO. None of the four
    /// shipped backends is one; a factory driving one says so here rather than passing itself
    /// off as the CPU, so that <see cref="InferenceBackend.RequireDevice"/> refuses it.
    /// </summary>
    Other,
}

/// <summary>
/// What a backend is: the assembly that supplies it, the device it executes on, and — for a
/// CUDA backend — the device it allocates on. <see cref="CudaDeviceId"/> is non-null exactly
/// when <see cref="Device"/> is <see cref="ComputeDevice.Cuda"/>, which the constructor
/// enforces.
///
/// <para>Read it off the live backend with <see cref="InferenceBackend.Describe"/>, or off the
/// context that will run your work with <c>ComputeContext.Backend</c>.</para>
/// </summary>
public readonly record struct BackendDescription
{
    private readonly string? _name;

    /// <param name="name">The assembly supplying the factory, e.g. <c>Shorokoo.WinGPU</c>.</param>
    /// <param name="device">The kind of device its sessions execute on.</param>
    /// <param name="cudaDeviceId">The CUDA device its sessions allocate on — required of a
    /// <see cref="ComputeDevice.Cuda"/> backend, and refused of any other.</param>
    /// <exception cref="ArgumentException"><paramref name="cudaDeviceId"/> disagrees with
    /// <paramref name="device"/>.</exception>
    public BackendDescription(string name, ComputeDevice device, int? cudaDeviceId)
    {
        ArgumentNullException.ThrowIfNull(name);
        if ((device == ComputeDevice.Cuda) != (cudaDeviceId is not null))
            throw new ArgumentException(
                $"A CUDA backend names the device it allocates on and every other kind names none, "
                + $"so {device} with {(cudaDeviceId is { } id ? $"device {id}" : "no device")} is not "
                + "a backend that could exist.",
                nameof(cudaDeviceId));
        _name = name;
        Device = device;
        CudaDeviceId = cudaDeviceId;
    }

    /// <summary>The assembly supplying the factory, e.g. <c>Shorokoo.WinGPU</c>.</summary>
    public string Name => _name ?? "";

    /// <summary>The kind of device this backend's sessions execute on.</summary>
    public ComputeDevice Device { get; }

    /// <summary>The CUDA device its sessions allocate on, or null on any non-CUDA backend.</summary>
    public int? CudaDeviceId { get; }

    /// <summary>One line naming the backend and the device, for a log or an error.</summary>
    public override string ToString() => Device switch
    {
        ComputeDevice.Cuda => $"{Name} (CUDA device {CudaDeviceId})",
        ComputeDevice.Cpu => $"{Name} (CPU)",
        _ => $"{Name} ({Device})",
    };
}
