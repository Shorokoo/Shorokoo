using System.Runtime.InteropServices;
using Shorokoo.Core.Backends;
using Shorokoo.Jax.Cuda;
using Shorokoo.Runtime;

namespace Shorokoo.Tests;

/// <summary>
/// The JAX CUDA backend on a card. It runs in the CUDA 13 environment, which the first test
/// provisions (some 7 GB) unless <c>SHOROKOO_PYTHON_ENV</c> names one; and a process has one
/// Python environment, so run this class in a process where no CPU backend of either family started
/// first -- the Purpose=Hardware filter does.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Hardware")]
public class JaxCudaHardwareTests
{
    private static readonly Lazy<JaxCudaBackend> Cuda = new(() => new JaxCudaBackend());

    [JaxCudaFact]
    public void TestATensorLivesOnTheCardAndComesHomeByCopy()
    {
        byte[] bytes = [.. MemoryMarshal.AsBytes<float>([1f, 2f, 3f])];
        using var onCard = Cuda.Value.CreateTensorInBackendMemory(ShorokooTensorElementType.Float, bytes, [3]);
        using var onHost = Cuda.Value.CreateTensorFromRawBytes(ShorokooTensorElementType.Float, bytes, [3]);

        Assert.False(onCard.IsHostAccessible);
        Assert.True(onHost.IsHostAccessible);
        Assert.Equal(bytes, Cuda.Value.CopyTensorToHost(onCard));
        Assert.Throws<InvalidOperationException>(() => onCard.GetTensorDataAsSpan<float>().Length);
    }

    [JaxCudaFact]
    public void TestAModelRunsOnTheCardAndAgreesWithTheHost()
    {
        var (model, input) = SideBySideModel.Concrete();
        var onHost = SideBySideModel.Floats(new ComputeContext().Execute(model, input.Shared())[0]);
        using var context = new ComputeContext(Cuda.Value);

        SideBySideModel.AssertAgree(onHost, SideBySideModel.Floats(context.Execute(model, input.Shared())[0]), SideBySideModel.DeviceTolerance);
        SideBySideModel.AssertAgree(onHost, SideBySideModel.Floats(context.Compile(model).Execute(input)[0]), SideBySideModel.DeviceTolerance);
    }

    [JaxCudaFact]
    public void TestARetainedOutputStaysOnTheCardTheRestComeHomeAndTheSessionReadsTheAllocator()
    {
        using var session = Cuda.Value.CreateSession(PyTorchBackendCoverageTests.Onnx("Neg", 1), default, default, DeviceMemorySettings.Default);
        using var x = Cuda.Value.CreateTensor([1f, -2f], [2]);
        var inputs = new Dictionary<string, IShorokooTensorValue> { ["x0"] = x };
        using var kept = session.RunRetainingOutputs(inputs, ["y"], new HashSet<string> { "y" }, RunSettings.Default)[0];
        using var fetched = session.Run(inputs, ["y"], RunSettings.Default)[0];

        Assert.True(session.HasDeviceMemory);
        Assert.Equal(SessionOutputPlacement.Device, session.OutputPlacement);
        Assert.False(kept.IsHostAccessible);
        Assert.Equal([-1f, 2f], fetched.GetTensorDataAsSpan<float>().ToArray());
        Assert.Equal(Cuda.Value.CopyTensorToHost(fetched), Cuda.Value.CopyTensorToHost(kept));
        Assert.True(session.ReadArenaStatistics()!.Value.InUseBytes > 0);
    }

    [JaxCudaFact]
    public void TestTheCardsDriverIsEnoughToProbeAndStartTheBackend()
    {
        Assert.Equal(BackendRejection.None, BackendPackage.Probe(typeof(JaxCudaBackend).Assembly.Location).Reason);
        Assert.Equal(3, Cuda.Value.Start().PythonVersion.Major);
    }
}

/// <summary>Runs a test only on Linux with an NVIDIA driver: where there is a card for the JAX CUDA
/// backend, whose CUDA plugin is built for Linux alone.</summary>
public sealed class JaxCudaFactAttribute : FactAttribute
{
    public JaxCudaFactAttribute()
    {
        if (!OperatingSystem.IsLinux())
            Skip = "JAX's CUDA plugin is built for Linux only.";
        else if (!NativeLibrary.TryLoad("libcuda.so.1", out var driver))
            Skip = "No NVIDIA driver on this machine, so there is no card for the JAX CUDA backend to run on.";
        else
            NativeLibrary.Free(driver);
    }
}
