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
/// on the first inference call. Assign it explicitly only to override the choice when
/// several backends are deployed side by side, to get a failure at startup instead of on
/// the first inference call, or when the backend DLL is not deployed next to
/// <c>Shorokoo.dll</c>:</para>
/// <code>
/// InferenceBackend.Factory = new LinuxGpuInferenceFactory();
/// </code>
/// <para>Mind the casing: the package, assembly and namespace spell the device
/// <c>GPU</c> while the type name spells it <c>Gpu</c>, so the fully qualified name is
/// <c>Shorokoo.LinuxGPU.LinuxGpuInferenceFactory</c>.</para>
/// </summary>
public sealed class LinuxGpuInferenceFactory : OrtSessionFactory
{
    /// <summary>
    /// Creates the factory. Every session it builds gets the CUDA execution provider on
    /// device 0.
    /// </summary>
    public LinuxGpuInferenceFactory() : base(static opts => opts.AppendExecutionProvider_CUDA(0)) { }
}
