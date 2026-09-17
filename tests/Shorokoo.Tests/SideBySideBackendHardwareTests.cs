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

    /// <summary>
    /// The one above runs a graph assembled here out of two inputs and an expression. This runs a
    /// Shorokoo <i>model</i>: <see cref="SideBySideMlp"/>, a <c>[Module]</c> the source generator
    /// lowered to a <c>ComputationGraph</c>, concretized once so its two weight matrices are
    /// sampled and baked in -- and then put on the host and on the card, in one process, from two
    /// <see cref="ComputeContext"/>s.
    /// </summary>
    [SideBySideCudaFact]
    public void TestAShorokooModelRunsOnTheCpuAndOnTheCardInOneProcess()
    {
        var (model, input) = SideBySideModel.Concrete();
        var cpu = new ComputeContext();
        var cuda = new ComputeContext(LoadCuda());

        Assert.Equal(ComputeDevice.Cpu, cpu.Backend.Device);
        Assert.Equal(ComputeDevice.Cuda, cuda.Backend.Device);

        var onCpu = SideBySideModel.Floats(cpu.Execute(model, input)[0]);
        var onCard = SideBySideModel.Floats(cuda.Execute(model, input)[0]);

        // One model, so the weights are the same weights: they were sampled when it was
        // concretized, above, and both contexts were handed the graph carrying them. Pin that the
        // host run repeats before holding the card's answer against it -- two different models
        // agreeing, or failing to, would say nothing about the backends.
        Assert.Equal(onCpu, SideBySideModel.Floats(cpu.Execute(model, input)[0]));
        SideBySideModel.AssertAgree(onCpu, onCard, SideBySideModel.DeviceTolerance);

        // And there is something to agree on: a forward pass that came out constant would read
        // the same off any two backends, working or not.
        Assert.True(onCpu.Distinct().Count() > 1, "the model's output is constant");

        // Back to the host afterwards, so neither run left the other's runtime unable to serve.
        SideBySideModel.AssertAgree(onCpu, SideBySideModel.Floats(cpu.Execute(model, input)[0]));

        // A session compiled on the card stays there, and re-runs there.
        var onCardCompiled = cuda.Compile(model);
        Assert.Equal("cuda:0", onCardCompiled.Backend.Name);
        SideBySideModel.AssertAgree(
            onCpu, SideBySideModel.Floats(onCardCompiled.Execute(input)[0]),
            SideBySideModel.DeviceTolerance);
        SideBySideModel.AssertAgree(
            onCpu, SideBySideModel.Floats(onCardCompiled.Execute(input)[0]),
            SideBySideModel.DeviceTolerance);

        // And what the card computed feeds the host's context, and the other way round. The
        // subject is the crossing itself -- a tensor from one runtime is a type the other knows
        // nothing about, and BackendTransfer is what makes it feedable -- so each arm is held
        // against the host running the model on its own output, which is the answer both are
        // approximating. Not against each other: that comparison differs in its input and in its
        // arithmetic at once, and so measures the card's mantissa rather than the transfer.
        var fromCard = cuda.Execute(model, input)[0].ToTensorData();
        var fromHost = cpu.Execute(model, input)[0].ToTensorData();
        var secondPass = SideBySideModel.Floats(cpu.Execute(model, fromHost)[0]);
        SideBySideModel.AssertAgree(
            secondPass, SideBySideModel.Floats(cpu.Execute(model, fromCard)[0]),
            SideBySideModel.DeviceTolerance);
        SideBySideModel.AssertAgree(
            secondPass, SideBySideModel.Floats(cuda.Execute(model, fromHost)[0]),
            SideBySideModel.DeviceTolerance);
    }

    /// <summary>
    /// The whole of part 2's promise, on the card: a tensor the execution provider kept in device
    /// memory, moved to host memory by one call, handed between host contexts without moving
    /// again, and run on.
    /// </summary>
    [SideBySideCudaFact]
    public void TestATensorLeftOnTheCardMovesToHostMemoryAndRunsThere()
    {
        var a = InputVector<float32>("a");
        var b = InputVector<float32>("b");
        var graph = new InternalComputationGraph([a, b], [a * b + a]);
        float[] av = [1f, 2f, 3f, 4f];
        float[] bv = [10f, 20f, 30f, 40f];
        var (ta, tb) = (TensorData([4L], av), TensorData([4L], bv));
        float[] expected = [.. av.Zip(bv, (x, y) => x * y + x)];

        var cuda = new ComputeContext(LoadCuda());
        var compiled = cuda.Compile(graph);
        Assert.True(compiled.HasDeviceMemory);

        // Left where the provider put it: this is the one kind of tensor the host cannot read.
        var onCard = compiled.Execute([ta, tb], [true])[0].ToTensorData();
        Assert.Equal(MemoryKind.Cuda, onCard.Space.Kind);
        Assert.False(onCard.IsHostResident);
        Assert.Throws<InvalidOperationException>(() => onCard.As<float32>().AccessMemory<float>());

        // One call brings it home, through the backend that owns the allocation.
        var firstHost = new ComputeContext();
        var onHost = onCard.TransferTo(firstHost);

        Assert.Equal(MemorySpace.Host, onHost.Space);
        Assert.True(onHost.OwnsMemory);
        SideBySideModel.AssertAgree(expected, Floats(onHost), SideBySideModel.DeviceTolerance);

        // The move spent the source, which is what a move across spaces means.
        Assert.True(onCard.IsDisposed);

        // Between two host contexts nothing moves but the ownership...
        var secondHost = new ComputeContext();
        var shared = onHost.TransferTo(secondHost);
        Assert.False(onHost.OwnsMemory);
        Assert.True(shared.OwnsMemory);

        // ...and the result runs on the second one, which is where it now lives. Fed back through
        // the same graph, so the answer is the model applied twice rather than the first answer.
        float[] twice = [.. expected.Zip(bv, (x, y) => x * y + x)];
        SideBySideModel.AssertAgree(
            twice, Floats(secondHost.Execute(graph, shared, tb)[0].ToTensorData()),
            SideBySideModel.DeviceTolerance);
    }

    private static float[] Floats(TensorData data)
        => [.. data.As<float32>().AccessMemory<float>()];

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
