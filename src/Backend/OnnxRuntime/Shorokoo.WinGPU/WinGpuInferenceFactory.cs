using Shorokoo.OnnxRuntime;

namespace Shorokoo.WinGPU;

public sealed class WinGpuInferenceFactory : OrtSessionFactory
{
    public WinGpuInferenceFactory() : base(static opts => opts.AppendExecutionProvider_CUDA(0)) { }
}
