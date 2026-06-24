using Shorokoo.OnnxRuntime;

namespace Shorokoo.LinuxCPU;

public sealed class LinuxCpuInferenceFactory : OrtSessionFactory
{
    // CPU is ORT's default EP -- no SessionOptions configuration needed.
    public LinuxCpuInferenceFactory() : base(static _ => { }) { }
}
