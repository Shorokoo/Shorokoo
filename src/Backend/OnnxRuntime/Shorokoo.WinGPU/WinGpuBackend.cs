using Shorokoo.OnnxRuntime;

namespace Shorokoo.WinGPU;

/// <summary>
/// The Shorokoo backend for Windows x64 on an NVIDIA GPU. It creates ONNX
/// Runtime sessions with the <b>CUDA execution provider appended on device 0</b>; ORT
/// falls back to the CPU provider for any node CUDA cannot run. It ships in the
/// <c>Shorokoo.WinGPU</c> package, which brings the CUDA-flavored native ONNX Runtime and
/// needs a CUDA 12.x runtime installed on the machine.
///
/// <para>Naming it in your code is <b>optional</b>. Referencing the package copies
/// <c>Shorokoo.WinGPU.dll</c> next to <c>Shorokoo.dll</c>, where
/// <see cref="Shorokoo.Core.Backends.DefaultBackend"/> auto-discovers it
/// on the first run. Assign it explicitly only to name the backend when a
/// deployment holds more than one (which discovery otherwise refuses rather than choosing
/// between), to get a failure at startup instead of on the first run, or when
/// the backend DLL is not deployed next to <c>Shorokoo.dll</c>:</para>
/// <code>
/// DefaultBackend.Instance = new WinGpuBackend();
/// </code>
/// <para>Assigning it names the <i>default</i> backend. To run this one alongside
/// another — a CPU context and a CUDA context in one process — hand it to a compute
/// context instead, and leave the default to whichever should have it:</para>
/// <code>
/// var context = new ComputeContext(new WinGpuBackend());
/// </code>
/// <para>Mind the casing: the package, assembly and namespace spell the device
/// <c>GPU</c> while the type name spells it <c>Gpu</c>, so the fully qualified name is
/// <c>Shorokoo.WinGPU.WinGpuBackend</c>.</para>
/// </summary>
public sealed class WinGpuBackend : OrtBackend
{
    /// <summary>
    /// Creates the backend. Every session it builds gets the CUDA execution provider on
    /// device 0, configured with the
    /// <see cref="Shorokoo.Core.Backends.DeviceMemorySettings"/> that session is
    /// built with.
    /// </summary>
    public WinGpuBackend() : base(cudaDeviceId: 0) { }
}
