using Shorokoo.Core.Backends;
using Shorokoo.OnnxRuntime;

namespace Shorokoo.LinuxCPU;

/// <summary>
/// The Shorokoo backend for Linux x64 on the CPU. It creates ONNX Runtime
/// sessions on ORT's default (CPU) execution provider — the CPU EP needs no
/// configuration, so this backend adds none. It ships in the <c>Shorokoo.LinuxCPU</c>
/// package, which also brings the native ONNX Runtime for the platform.
///
/// <para>Naming it in your code is <b>optional</b>. Referencing the package copies
/// <c>Shorokoo.LinuxCPU.dll</c> next to <c>Shorokoo.dll</c>, where
/// <see cref="Shorokoo.Core.Backends.DefaultBackend"/> auto-discovers it
/// on the first run. Assign it explicitly only to name the backend when a
/// deployment holds more than one (which discovery otherwise refuses rather than choosing
/// between), to get a failure at startup instead of on the first run, or when
/// the backend DLL is not deployed next to <c>Shorokoo.dll</c>:</para>
/// <code>
/// DefaultBackend.Instance = new LinuxCpuBackend();
/// </code>
/// <para>Assigning it names the <i>default</i> backend. To run this one alongside
/// another — a CPU context and a CUDA context in one process — hand it to a compute
/// context instead, and leave the default to whichever should have it:</para>
/// <code>
/// var context = new ComputeContext(new LinuxCpuBackend());
/// </code>
/// <para>Mind the casing: the package, assembly and namespace spell the device
/// <c>CPU</c> while the type name spells it <c>Cpu</c>, so the fully qualified name is
/// <c>Shorokoo.LinuxCPU.LinuxCpuBackend</c>.</para>
/// </summary>
public sealed class LinuxCpuBackend : OrtBackend
{
    /// <summary>
    /// Creates the backend. CPU is ORT's default execution provider, so its
    /// execution-provider step does nothing; sessions still get the usual log-severity
    /// and graph-optimization options.
    /// </summary>
    public LinuxCpuBackend() : base(static (_, _) => { }, ComputeDevice.Cpu, cudaDeviceId: null) { }
}
