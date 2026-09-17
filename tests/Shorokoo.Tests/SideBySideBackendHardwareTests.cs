using System.Runtime.InteropServices;
using Shorokoo.Core.Inference.Abstractions;
using Shorokoo.Runtime;

namespace Shorokoo.Tests;

/// <summary>
/// The pairing <see cref="SideBySideBackendCoverageTests"/> stands in for, on hardware: one model
/// run on the CPU and on the card, in one process, from two <see cref="ComputeContext"/>s.
///
/// <para>Excluded from the coverage suite — it needs a CUDA machine <i>and</i> a deployment that
/// carries the CUDA backend's assembly and its provider libraries under <c>ort/cuda/</c>, which
/// the ordinary build leaves out because <c>libonnxruntime_providers_cuda.so</c> alone is some
/// 340 MB. Build with <c>-p:ShorokooDeployGpuBackend=true</c> to get them, then run with
/// <c>--filter "Purpose=Hardware"</c>. Without that deployment each test here skips and says
/// so rather than failing.</para>
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Hardware")]
public class SideBySideBackendHardwareTests
{
    private static readonly string CudaBackendDirectory =
        Path.Combine(AppContext.BaseDirectory, "ort", "cuda");

    private static bool Windows => RuntimeInformation.IsOSPlatform(OSPlatform.Windows);

    /// <summary>Why the CUDA backend cannot be loaded from this deployment, or null when it can.
    /// Names the missing file and the build switch that deploys it, so a skip is actionable.</summary>
    internal static string? DeploymentGap()
    {
        string[] required =
        [
            Windows ? "onnxruntime.dll" : "libonnxruntime.so",
            Windows ? "onnxruntime_providers_cuda.dll" : "libonnxruntime_providers_cuda.so",
            Windows ? "onnxruntime_providers_shared.dll" : "libonnxruntime_providers_shared.so",
            (Windows ? "Shorokoo.WinGPU" : "Shorokoo.LinuxGPU") + ".dll",
        ];
        var missing = required.Where(f => !File.Exists(Path.Combine(CudaBackendDirectory, f))).ToList();
        return missing.Count == 0 ? null
            : $"The CUDA backend is not deployed: {string.Join(", ", missing)} missing from "
              + $"'{CudaBackendDirectory}'. Rebuild with -p:ShorokooDeployGpuBackend=true.";
    }

    private static IShorokooInferenceSessionFactory LoadCuda() => IsolatedBackend.Load(
        new IsolatedBackendSpec
        {
            Name = "cuda:0",
            FactoryAssembly = Windows ? "Shorokoo.WinGPU" : "Shorokoo.LinuxGPU",
            NativeRuntimePath = Path.Combine(
                CudaBackendDirectory, Windows ? "onnxruntime.dll" : "libonnxruntime.so"),
            ProbeDirectory = CudaBackendDirectory,
        });

    [SideBySideCudaFact]
    public void TestOneModelRunsOnTheCpuAndOnTheCardInOneProcess()
    {
        var a = InputVector<float32>("a");
        var b = InputVector<float32>("b");
        var graph = new InternalComputationGraph([a, b], [a * b + a]);
        float[] av = [1f, 2f, 3f, 4f];
        float[] bv = [10f, 20f, 30f, 40f];
        var (ta, tb) = (TensorData([4], av), TensorData([4], bv));
        float[] expected = [.. av.Zip(bv, (x, y) => x * y + x)];

        var cpu = new ComputeContext();
        var cuda = new ComputeContext(LoadCuda());

        Assert.Equal(ComputeDevice.Cpu, cpu.Backend.Device);
        Assert.Equal(ComputeDevice.Cuda, cuda.Backend.Device);
        Assert.Equal(0, cuda.Backend.CudaDeviceId);

        // The same graph and the same two tensors, on the host and on the card, back and forth.
        Assert.Equal(expected, Floats(cpu.Execute(graph, ta, tb)[0]));
        Assert.Equal(expected, Floats(cuda.Execute(graph, ta, tb)[0]));
        Assert.Equal(expected, Floats(cpu.Execute(graph, ta, tb)[0]));

        // A compiled session stays on the backend that built it, and re-runs there.
        var onCard = cuda.Compile(graph);
        Assert.Equal("cuda:0", onCard.Backend.Name);
        Assert.Equal(expected, Floats(onCard.Execute(ta, tb)[0]));
        Assert.Equal(expected, Floats(onCard.Execute(ta, tb)[0]));

        // What the card produced feeds the host context, and the other way round.
        var fromCard = cuda.Execute(graph, ta, tb)[0].ToTensorData();
        var fromHost = cpu.Execute(graph, ta, tb)[0].ToTensorData();
        float[] twice = [.. expected.Zip(bv, (x, y) => x * y + x)];
        Assert.Equal(twice, Floats(cpu.Execute(graph, fromCard, tb)[0]));
        Assert.Equal(twice, Floats(cuda.Execute(graph, fromHost, tb)[0]));
    }

    private static float[] Floats(NamedModelParam param)
        => [.. param.ToTensorData().As<float32>().AccessMemory<float>()];
}

/// <summary>
/// A <see cref="FactAttribute"/> that skips unless this deployment carries the CUDA backend
/// side by side with the default one. Distinct from <see cref="CudaFactAttribute"/>, which asks
/// whether the <i>process-wide</i> backend is a GPU one — the question here is whether a second
/// backend can be loaded next to a CPU one, which is a property of the deployment.
/// </summary>
public sealed class SideBySideCudaFactAttribute : FactAttribute
{
    public SideBySideCudaFactAttribute()
    {
        if (SideBySideBackendHardwareTests.DeploymentGap() is { } gap) Skip = gap;
    }
}
