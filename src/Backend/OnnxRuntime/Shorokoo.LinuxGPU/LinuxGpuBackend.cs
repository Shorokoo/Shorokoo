using Shorokoo.OnnxRuntime;

namespace Shorokoo.LinuxGPU;

/// <summary>
/// The Shorokoo inference backend for Linux x64 on an NVIDIA GPU. It creates ONNX Runtime
/// sessions with the <b>CUDA execution provider appended on device 0</b>; ORT falls back
/// to the CPU provider for any node CUDA cannot run. It ships in the
/// <c>Shorokoo.LinuxGPU</c> package, which brings the CUDA-flavored native ONNX Runtime
/// and needs a CUDA 12.x runtime installed on the machine.
///
/// <para>Naming it in your code is <b>optional</b>. Referencing the package copies
/// <c>Shorokoo.LinuxGPU.dll</c> next to <c>Shorokoo.dll</c>, where
/// <see cref="Shorokoo.Core.Inference.Abstractions.InferenceBackend"/> auto-discovers it
/// on the first inference call. Assign it explicitly only to name the backend when a
/// deployment holds more than one (which discovery otherwise refuses rather than choosing
/// between), to get a failure at startup instead of on the first inference call, or when
/// the backend DLL is not deployed next to <c>Shorokoo.dll</c>:</para>
/// <code>
/// InferenceBackend.Default = new LinuxGpuBackend();
/// </code>
/// <para>Assigning it names the <i>default</i> backend. To run this one alongside
/// another — a CPU context and a CUDA context in one process — hand it to a compute
/// context instead, and leave the default to whichever should have it:</para>
/// <code>
/// var context = new ComputeContext(new LinuxGpuBackend());
/// </code>
/// <para>Mind the casing: the package, assembly and namespace spell the device
/// <c>GPU</c> while the type name spells it <c>Gpu</c>, so the fully qualified name is
/// <c>Shorokoo.LinuxGPU.LinuxGpuBackend</c>.</para>
/// </summary>
public sealed class LinuxGpuBackend : OrtBackend
{
    /// <summary>
    /// Creates the backend. Every session it builds gets the CUDA execution provider on
    /// device 0, configured with the
    /// <see cref="Shorokoo.Core.Inference.Abstractions.DeviceMemorySettings"/> that session is
    /// built with.
    /// </summary>
    public LinuxGpuBackend() : base(cudaDeviceId: 0) { }
}
