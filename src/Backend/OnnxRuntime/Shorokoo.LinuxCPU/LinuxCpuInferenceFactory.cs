using Shorokoo.OnnxRuntime;

namespace Shorokoo.LinuxCPU;

/// <summary>
/// The Shorokoo inference backend for Linux x64 on the CPU. It creates ONNX Runtime
/// sessions on ORT's default (CPU) execution provider — the CPU EP needs no
/// configuration, so this factory adds none. It ships in the <c>Shorokoo.LinuxCPU</c>
/// package, which also brings the native ONNX Runtime for the platform.
///
/// <para>Naming it in your code is <b>optional</b>. Referencing the package copies
/// <c>Shorokoo.LinuxCPU.dll</c> next to <c>Shorokoo.dll</c>, where
/// <see cref="Shorokoo.Core.Inference.Abstractions.InferenceBackend"/> auto-discovers it
/// on the first inference call. Assign it explicitly only to override the choice when
/// several backends are deployed side by side, to get a failure at startup instead of on
/// the first inference call, or when the backend DLL is not deployed next to
/// <c>Shorokoo.dll</c>:</para>
/// <code>
/// InferenceBackend.Factory = new LinuxCpuInferenceFactory();
/// </code>
/// <para>Mind the casing: the package, assembly and namespace spell the device
/// <c>CPU</c> while the type name spells it <c>Cpu</c>, so the fully qualified name is
/// <c>Shorokoo.LinuxCPU.LinuxCpuInferenceFactory</c>.</para>
/// </summary>
public sealed class LinuxCpuInferenceFactory : OrtSessionFactory
{
    /// <summary>
    /// Creates the factory. CPU is ORT's default execution provider, so this factory's
    /// execution-provider step does nothing; sessions still get the usual log-severity
    /// and graph-optimization options.
    /// </summary>
    public LinuxCpuInferenceFactory() : base(static _ => { }) { }
}
