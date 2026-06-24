using Shorokoo.OnnxRuntime;

namespace Shorokoo.LinuxGPU;

public sealed class LinuxGpuInferenceFactory : OrtSessionFactory
{
    public LinuxGpuInferenceFactory() : base(static opts => opts.AppendExecutionProvider_CUDA(0)) { }
}
