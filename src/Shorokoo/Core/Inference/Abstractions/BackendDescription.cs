namespace Shorokoo.Core.Inference.Abstractions;

/// <summary>The kind of device a backend executes on.</summary>
public enum ComputeDevice
{
    /// <summary>The host CPU — ONNX Runtime's default execution provider.</summary>
    Cpu,

    /// <summary>An NVIDIA GPU, through ONNX Runtime's CUDA execution provider.</summary>
    Cuda,
}

/// <summary>
/// What a backend is: the assembly that supplies it, the device it executes on, and — for a
/// CUDA backend — the device it allocates on. <see cref="CudaDeviceId"/> is non-null exactly
/// when <see cref="Device"/> is <see cref="ComputeDevice.Cuda"/>.
///
/// <para>Read it off the live backend with <see cref="InferenceBackend.Describe"/>, or off the
/// context that will run your work with <c>ComputeContext.Backend</c>.</para>
/// </summary>
/// <param name="Name">The assembly supplying the factory, e.g. <c>Shorokoo.WinGPU</c>.</param>
/// <param name="Device">The kind of device its sessions execute on.</param>
/// <param name="CudaDeviceId">The CUDA device its sessions allocate on, or null on a CPU backend.</param>
public readonly record struct BackendDescription(string Name, ComputeDevice Device, int? CudaDeviceId)
{
    /// <summary>One line naming the backend and the device, for a log or an error.</summary>
    public override string ToString()
        => CudaDeviceId is { } id ? $"{Name} (CUDA device {id})" : $"{Name} (CPU)";
}
