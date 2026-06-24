using Shorokoo.OnnxRuntime;

namespace Shorokoo.WinCPU;

public sealed class WinCpuInferenceFactory : OrtSessionFactory
{
    // CPU is ORT's default EP -- no SessionOptions configuration needed.
    public WinCpuInferenceFactory() : base(static _ => { }) { }
}
