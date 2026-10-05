using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using Shorokoo.Core.Backends;
using Shorokoo.Modules.Losses;
using Shorokoo.Modules.Optimizers;
using Shorokoo.PyTorch.Cuda;
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
///
/// <para><b>Without <c>-p:ShorokooGpuTests=true</c>, though</b>: that one makes the process-wide
/// default backend a CUDA one, which is what <c>GpuExecutionTests</c> needs and the opposite of
/// what the host half of this pairing needs. The two classes therefore want different builds, and
/// each skips out of the other's.</para>
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
        if (missing.Count > 0)
            return $"The CUDA backend is not deployed: {string.Join(", ", missing)} missing from "
                + $"'{CudaBackendDirectory}'. Rebuild with -p:ShorokooDeployGpuBackend=true.";

        // The deployment is only half of what these need. Without this, a machine with the files
        // and no card ran every test here and failed inside ORT's session creation instead of
        // skipping -- which says nothing about the product and reads like a real regression.
        if (DeviceMemory.Read() is null)
            return "No CUDA device answers on this machine, so the card half of the CPU-and-CUDA "
                + "pairing cannot run. The deployment is in place; run this on a CUDA machine.";

        // And the host half needs the process-wide default backend to be the host one, which is
        // what -p:ShorokooGpuTests=true takes away: under it `new ComputeContext()` is a CUDA
        // context, so the pairing is a card against a card and four of these fail on assertions
        // about where a tensor lives. That switch is for GpuExecutionTests; this class wants the
        // deployment without it.
        var backend = DefaultBackend.Instance.GetType().Assembly.GetName().Name ?? "";
        return backend.EndsWith("GPU", StringComparison.OrdinalIgnoreCase)
            ? $"The process-wide default backend is '{backend}', so the host half of the "
              + "CPU-and-CUDA pairing would run on the card as well. Build without "
              + "-p:ShorokooGpuTests=true to run these."
            : null;
    }

    private static IShorokooBackend LoadCuda() => IsolatedBackend.Load(
        new IsolatedBackendSpec
        {
            Name = "cuda:0",
            BackendAssembly = Windows ? "Shorokoo.WinGPU" : "Shorokoo.LinuxGPU",
            NativeRuntimePath = Path.Combine(
                CudaBackendDirectory, Windows ? "onnxruntime.dll" : "libonnxruntime.so"),
            ProbeDirectory = CudaBackendDirectory,
        });

    /// <summary>Whether the copy of the ONNX Runtime backend that built <paramref name="session"/>
    /// made its runtime's environment with thread pools its sessions share.</summary>
    private static bool SharesThreadPools(IShorokooSession session)
        => (bool)session.GetType().Assembly.GetType("Shorokoo.OnnxRuntime.OrtEnvironment")!
            .GetProperty("SharedThreadPools", BindingFlags.Static | BindingFlags.NonPublic)!.GetValue(null)!;

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
        Assert.Equal(expected, Floats(cpu.Execute(graph, ta.Shared(), tb.Shared())[0]));
        Assert.Equal(expected, Floats(cuda.Execute(graph, ta.Shared(), tb.Shared())[0]));
        Assert.Equal(expected, Floats(cpu.Execute(graph, ta.Shared(), tb.Shared())[0]));

        // A compiled session stays on the backend that built it, and re-runs there.
        var onCard = cuda.Compile(graph);
        Assert.Equal("cuda:0", onCard.Backend.Name);
        Assert.True(SharesThreadPools(onCard.Session));
        Assert.True(SharesThreadPools(cpu.Compile(graph).Session));
        Assert.Equal(expected, Floats(onCard.Execute(ta.Shared(), tb.Shared())[0]));
        Assert.Equal(expected, Floats(onCard.Execute(ta.Shared(), tb.Shared())[0]));

        // What the card produced feeds the host context, and the other way round.
        var fromCard = cuda.Execute(graph, ta.Shared(), tb.Shared())[0].ToTensorData();
        var fromHost = cpu.Execute(graph, ta, tb.Shared())[0].ToTensorData();
        float[] twice = [.. expected.Zip(bv, (x, y) => x * y + x)];
        Assert.Equal(twice, Floats(cpu.Execute(graph, fromCard, tb.Shared())[0]));
        Assert.Equal(twice, Floats(cuda.Execute(graph, fromHost, tb)[0]));
    }

    [SideBySideCudaFact]
    public void TestAShorokooModelRunsOnTheCpuAndOnTheCardInOneProcess()
    {
        var (model, input) = SideBySideModel.Concrete();
        var cpu = new ComputeContext();
        var cuda = new ComputeContext(LoadCuda());

        Assert.Equal(ComputeDevice.Cpu, cpu.Backend.Device);
        Assert.Equal(ComputeDevice.Cuda, cuda.Backend.Device);

        var onCpu = SideBySideModel.Floats(cpu.Execute(model, input.Shared())[0]);
        var onCard = SideBySideModel.Floats(cuda.Execute(model, input.Shared())[0]);

        Assert.Equal(onCpu, SideBySideModel.Floats(cpu.Execute(model, input.Shared())[0]));
        SideBySideModel.AssertAgree(onCpu, onCard, SideBySideModel.DeviceTolerance);

        // And there is something to agree on: a forward pass that came out constant would read
        // the same off any two backends, working or not.
        Assert.True(onCpu.Distinct().Count() > 1);

        // Back to the host afterwards, so neither run left the other's runtime unable to serve.
        SideBySideModel.AssertAgree(onCpu, SideBySideModel.Floats(cpu.Execute(model, input.Shared())[0]));

        // A session compiled on the card stays there, and re-runs there.
        var onCardCompiled = cuda.Compile(model);
        Assert.Equal("cuda:0", onCardCompiled.Backend.Name);
        SideBySideModel.AssertAgree(
            onCpu, SideBySideModel.Floats(onCardCompiled.Execute(input.Shared())[0]),
            SideBySideModel.DeviceTolerance);
        SideBySideModel.AssertAgree(
            onCpu, SideBySideModel.Floats(onCardCompiled.Execute(input.Shared())[0]),
            SideBySideModel.DeviceTolerance);

        var fromCard = cuda.Execute(model, input.Shared())[0].ToTensorData();
        var fromHost = cpu.Execute(model, input)[0].ToTensorData();
        var secondPass = SideBySideModel.Floats(cpu.Execute(model, fromHost.Shared())[0]);
        SideBySideModel.AssertAgree(
            secondPass, SideBySideModel.Floats(cpu.Execute(model, fromCard)[0]),
            SideBySideModel.DeviceTolerance);
        SideBySideModel.AssertAgree(
            secondPass, SideBySideModel.Floats(cuda.Execute(model, fromHost)[0]),
            SideBySideModel.DeviceTolerance);
    }

    [SideBySideCudaFact]
    public void TestFloat32ProductsConvolutionsAndRecurrentLayersOnTheCardAreComputedInFullPrecisionUnlessTensorFloat32IsAllowed()
    {
        var host = SideBySideModel.LargeLayers(new ComputeContext());
        var cuda = LoadCuda();
        var strict = new ComputeContext(cuda);
        var allowed = new ComputeContext(cuda) { Precision = SideBySideModel.AllowingTensorFloat32 };

        SideBySideModel.AssertFullPrecision(host, SideBySideModel.LargeLayers(strict));
        SideBySideModel.AssertTensorFloat32(host, SideBySideModel.LargeLayers(allowed));
        SideBySideModel.AssertFullPrecision(host, SideBySideModel.LargeLayers(strict));
    }

    [SideBySideCudaFact]
    public void TestATensorLeftOnTheCardIsCopiedToHostMemoryAndRunsThere()
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

        // Left where the run put it, and read there through a host copy of its values.
        var onCard = compiled.Execute(ta, tb.Shared())[0].ToTensorData();
        Assert.Equal(MemoryKind.Cuda, onCard.Space.Kind);
        Assert.False(onCard.IsHostResident);
        SideBySideModel.AssertAgree(expected, onCard.As<float32>().AccessMemory<float>().ToArray(), SideBySideModel.DeviceTolerance);

        // One call brings it home, through the backend that made the allocation, and leaves the
        // source where it was.
        var firstHost = new ComputeContext();
        var onHost = onCard.To(firstHost);

        Assert.Equal(MemorySpace.Host, onHost.Space);
        SideBySideModel.AssertAgree(expected, Floats(onHost), SideBySideModel.DeviceTolerance);
        Assert.False(onCard.IsDisposed);
        Assert.Equal(MemoryKind.Cuda, onCard.Space.Kind);

        // Between two host contexts nothing moves: the second is handed the very same tensor...
        var secondHost = new ComputeContext();
        Assert.Same(onHost, onHost.To(secondHost));
        Assert.Contains(onHost, secondHost.Tensors);

        // ...and it runs there. Fed back through the same graph, so the answer is the model applied
        // twice rather than the first answer.
        float[] twice = [.. expected.Zip(bv, (x, y) => x * y + x)];
        SideBySideModel.AssertAgree(
            twice, Floats(secondHost.Execute(graph, onHost, tb)[0].ToTensorData()),
            SideBySideModel.DeviceTolerance);
    }

    [SideBySideCudaFact]
    public void TestTrainingStateLeftOnTheCardNamesTheDeviceAndComesHomeFromIt()
    {
        using var cuda = new ComputeContext(LoadCuda());
        var rig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, AdamWOptimizer.ComputationGraph,
            [TensorData([4L], [1f, 2f, 3f, 4f])],
            new AdamWOptimizerHyperparameters { LearningRate = 0.1f },
            runtimeContext: cuda);

        var checkpoint = rig.CreateInitialCheckpoint();
        var (input, target) = (TrainingRigHelpers.InBatch(1f, 2f, 3f, 4f),
                               TrainingRigHelpers.TargetBatch(2f, 4f, 6f, 8f));

        var compiled = rig.RuntimeContext.Compile(rig.TrainingStepPureGraph);

        var retained = compiled.Execute(
            ComputeContext.ExpandStructInputs(
                [checkpoint.TrainableParams.Shared(), checkpoint.ModelState.Shared(), checkpoint.OptimizerState.Shared(),
                 input.Shared(), target.Shared()]));

        Assert.NotEmpty(retained);
        Assert.NotNull(cuda.Backend.CudaDeviceId);
        var onCard = MemorySpace.Cuda(cuda.Backend.CudaDeviceId!.Value);
        foreach (var output in retained)
        {
            var state = output.ToTensorData();
            Assert.False(state.IsHostResident);
            Assert.True(state.Space.IsKnown);
            Assert.Equal(onCard, state.Space);
            Assert.Contains(state, cuda.Tensors);
        }

        var home = retained[0].ToTensorData().ToHost();
        Assert.Equal(MemorySpace.Host, home.Space);
        Assert.All(Floats(home), v => Assert.True(float.IsFinite(v)));

        // And the loop built on all this still trains, publishing its state where it is.
        using var run = rig.BeginResidentRun(checkpoint);
        run.Step(input.Shared(), target.Shared());
        var published = run.StepToCheckpoint(input, target);

        Assert.Equal(2, published.Step);
        Assert.All(published.TrainableParams.Fields.Values,
            f => Assert.Equal(onCard, ((TensorData)f).Space));
        Assert.All(published.ToHost().TrainableParams.Fields.Values,
            f => Assert.Equal(MemorySpace.Host, ((TensorData)f).Space));
        Assert.NotEmpty(TrainingRigHelpers.FlattenStruct(published.ToHost().TrainableParams));
    }

    [SideBySideCudaFact]
    public void TestATensorPutOnTheCardLivesInDeviceMemoryAndRunsThere()
    {
        var a = InputVector<float32>("a");
        var b = InputVector<float32>("b");
        var graph = new InternalComputationGraph([a, b], [a * b + a]);
        float[] av = [1f, 2f, 3f, 4f];
        float[] bv = [10f, 20f, 30f, 40f];
        float[] expected = [.. av.Zip(bv, (x, y) => x * y + x)];

        var cuda = new ComputeContext(LoadCuda());
        var onHost = TensorData([4L], av);
        Assert.Equal(MemorySpace.Host, onHost.Space);

        var onCard = onHost.To(cuda);

        Assert.Equal(MemorySpace.Cuda(0), onCard.Space);
        Assert.Contains(onCard, cuda.Tensors);
        Assert.Same(onCard, onCard.To(cuda));
        Assert.False(onCard.IsHostResident);
        Assert.Equal(av, onCard.As<float32>().AccessMemory<float>().ToArray());

        // A copy onto the card, and the source untouched.
        Assert.False(onHost.IsDisposed);
        Assert.Equal(av, Floats(onHost));

        var tb = TensorData([4L], bv);
        SideBySideModel.AssertAgree(
            expected, Floats(cuda.Execute(graph, onCard.Shared(), tb)[0]), SideBySideModel.DeviceTolerance);

        // Still there afterwards, and still the card's: fed .Shared(), a run reads a tensor rather
        // than consuming it.
        Assert.Equal(MemorySpace.Cuda(0), onCard.Space);

        var home = onCard.ToHost();
        Assert.Equal(MemorySpace.Host, home.Space);
        Assert.True(home.IsHostResident);
        Assert.Equal(av, Floats(home));
    }

    /// <summary>A tensor on the card, fed to a run on the host: read through one host copy, held
    /// and reused while it is read, and consumed through that copy when it is fed as it is — the
    /// card's own memory released at the feed.</summary>
    [SideBySideCudaFact]
    public void TestATensorOnTheCardIsReadByAHostRunThroughOneCopyAndConsumedThroughIt()
    {
        var a = InputVector<float32>("a");
        var b = InputVector<float32>("b");
        using var cuda = new ComputeContext(LoadCuda());
        using var cpu = new ComputeContext();
        var compiled = cpu.Compile(new InternalComputationGraph([a, b], [a * b + a]));
        TensorData Run(IData x) => compiled.Execute(x, TensorData([2L], 10f, 20f))[0].ToTensorData();
        var onCard = TensorData([2L], 1f, 2f).To(cuda);
        Assert.Equal(MemorySpace.Cuda(0), onCard.Space);

        var first = Run(onCard.Shared());
        var copy = Assert.Single(cpu.Tensors.Except([first, onCard]));
        Assert.Equal(MemorySpace.Host, copy.Space);
        var second = Run(onCard.Shared());
        Assert.Same(copy, Assert.Single(cpu.Tensors.Except([first, second, onCard])));

        var third = Run(onCard);
        Assert.True(onCard.IsDisposed && copy.IsDisposed);
        Assert.All((TensorData[])[first, second, third], t => Assert.Equal([11f, 42f], Floats(t)));
    }

    [SideBySideCudaFact]
    public void TestAnUninitializedTensorAllocatedOnTheCardLivesThereAndTakesWhatIsWrittenToIt()
    {
        var cuda = LoadCuda();
        long[] shape = [2L, 3L];
        var byteCount = TensorElementLayout.ByteCount(ShorokooTensorElementType.Float, shape);
        byte[] written = [.. Enumerable.Range(1, byteCount).Select(i => (byte)i)];

        using var device = cuda.CreateUninitializedTensorInBackendMemory(
            ShorokooTensorElementType.Float, shape);

        Assert.Equal(ShorokooTensorElementType.Float, device.ElementType);
        Assert.Equal(shape, device.Shape);
        Assert.False(device.IsHostAccessible);
        Assert.Throws<InvalidOperationException>(() => _ = device.GetTensorDataAsSpan<float>().Length);
        Assert.Throws<InvalidOperationException>(() => _ = device.GetTensorMutableDataAsSpan<float>().Length);

        Assert.NotNull(cuda.Description.CudaDeviceId);
        Assert.Equal(("Cuda", cuda.Description.CudaDeviceId!.Value), AllocatorOf(device));

        Assert.True(cuda.TryCopyHostToTensorRange(device, 0, written));
        var home = cuda.CopyTensorToHost(device);
        Assert.Equal(byteCount, home.Length);
        Assert.Equal(written, home);

        using var empty = cuda.CreateUninitializedTensorInBackendMemory(
            ShorokooTensorElementType.Float, [0L, 3L]);
        Assert.Equal((long[])[0L, 3L], empty.Shape);
        Assert.False(empty.IsHostAccessible);
        Assert.Equal(("Cuda", cuda.Description.CudaDeviceId!.Value), AllocatorOf(empty));
        Assert.Empty(cuda.CopyTensorToHost(empty));
    }

    [SideBySideCudaFact]
    public void TestASequenceValuedModelRunsOnTheCardAndItsElementsComeBackFromTheHost()
    {
        var x = InputVector<float32>("x");
        var split = new InternalComputationGraph(
            [x], [OnnxOp.SplitToSequence(x * Scalar(2f), split: Vector(2L, 2L), axis: 0)]);
        var pick = new InternalComputationGraph(
            [x], [OnnxOp.SequenceAt(OnnxOp.SplitToSequence(x, split: Vector(2L, 2L), axis: 0), Scalar(1L))]);
        var input = TensorData([4L], (float[])[1f, 2f, 3f, 4f]);

        using var cuda = new ComputeContext(LoadCuda());
        var sequence = cuda.Execute(split, input.Shared())[0].ToTensorDataSequence();

        Assert.IsType<OnnxTensorDataSequence<float32>>(sequence);
        Assert.Equal(2, sequence.Count);
        Assert.Equal([2f, 4f], Floats(sequence[0]));
        Assert.Equal([6f, 8f], Floats(sequence[1]));
        Assert.All(sequence, e => Assert.Equal(MemorySpace.Host, e.Space));
        Assert.Equal([3f, 4f], Floats(cuda.Execute(pick, input)[0]));

        // And the sequence crosses back to a host context, element by element.
        Assert.Equal([2f, 4f], Floats(sequence.CopyTo(new ComputeContext())[0]));
        Assert.Equal([2f, 4f], Floats(sequence.ToHost()[0]));
    }

    [SideBySideCudaFact]
    public void TestASequenceOutputOfACardRunComesBackInHostMemoryWhereItsElementsAreRead()
    {
        var x = InputVector<float32>("x");
        var pair = new InternalComputationGraph([x], [OnnxOp.SequenceConstruct(x, x + x)]);
        using var cuda = new ComputeContext(LoadCuda());

        var sequence = cuda.Compile(pair)
            .Execute(TensorData([2L], (float[])[1f, 2f]))[0].ToTensorDataSequence();

        Assert.Equal([1f, 2f], Floats(sequence[0]));
        Assert.Equal([2f, 4f], Floats(sequence[1]));
        Assert.All(sequence, e => Assert.Equal(MemorySpace.Host, e.Space));
    }

    [SideBySideCudaFact]
    public void TestASequenceRefusesTensorsTheCardHoldsRatherThanBecomingUnreadable()
    {
        var cuda = LoadCuda();
        IShorokooTensorValue OnCard() => cuda.CreateUninitializedTensorInBackendMemory(
            ShorokooTensorElementType.Float, [2L]);
        IShorokooTensorValue OnHost() => cuda.CreateTensor<float>([1f, 2f], [2L]);

        // A refusal takes over what it was handed, as the contract requires of any failure, so
        // every attempt is handed values of its own: reusing one reads a released value.
        IShorokooTensorValue[] handed = [OnHost(), OnCard()];
        var refused = Assert.Throws<InvalidOperationException>(() => cuda.CreateSequence(handed));
        Assert.Contains("CopyTensorToHost", refused.Message);
        Assert.All(handed, value => Assert.Throws<ObjectDisposedException>(() => value.ValueType));
        Assert.Throws<InvalidOperationException>(() => cuda.CreateSequence([OnCard()]));

        // The advice the message gives, followed on a tensor the card holds.
        using var onCard = OnCard();
        var home = cuda.CreateTensorFromRawBytes(
            ShorokooTensorElementType.Float, cuda.CopyTensorToHost(onCard), [2L]);

        using var sequence = cuda.CreateSequence([OnHost(), home]);
        Assert.Equal(2, sequence.GetValueCount());
        using var element = sequence.GetValue(0);
        Assert.Equal([1f, 2f], element.GetTensorDataAsSpan<float>().ToArray());
    }

    [SideBySideCudaFact]
    public void TestTheOnnxRuntimeAndPyTorchCudaBackendsShareOneProcessWhicheverStartsFirst()
    {
        Assert.Equal(0, InAChildProcess([], "cuda-backends", "onnxruntime-first"));
        Assert.Equal(0, InAChildProcess([], "cuda-backends", "pytorch-first"));
    }

    [SideBySideCudaFact]
    public void TestAConvolutionOnTheCardEndsInAResultOrAnExceptionWithNoOtherCudaMajorOnThePath()
    {
        Assert.Equal(0, InAChildProcess(WithNoOtherCudaMajorOnThePathAndNoPythonEnvironment(), "onnxruntime-convolution"));
    }

    [SideBySideCudaFact]
    public void TestACudaBackendStartedAfterTheProcessLoadedAnotherReleaseOfThePinnedCudnnRefusesNamingTheCopyHeld()
    {
        var other = Directory.CreateDirectory(Path.Combine(Path.GetTempPath(), "shorokoo-other-cudnn-" + Guid.NewGuid().ToString("N"))).FullName;
        try
        {
            Assert.Equal(0, InAChildProcess([], "onnxruntime-convolution-after-another-cudnn", other));
        }
        finally
        {
            Directory.Delete(other, recursive: true);
        }
    }

    /// <summary>What <c>dotnet Shorokoo.Tests.dll</c> runs: one case of a test that needs a process of
    /// its own, because what it covers is which native libraries a process loads, and from where.</summary>
    public static int Main(string[] args) => args switch
    {
        ["cuda-backends", var first] => BothCudaBackendsRun(pytorchFirst: first == "pytorch-first") ? 0 : 1,
        ["onnxruntime-convolution"] => AConvolutionEndsInAResultOrAnException(),
        ["onnxruntime-convolution-after-another-cudnn", var other] => AConvolutionAfterAnotherCudnnIsRefusedNamingIt(other),
        ["cuda-wheel-download", var root, var url, var served] => SideBySideBackendCoverageTests.DownloadEndedWithItsProcess(root, url, served),
        ["deep-model-routes", var path] => OnnxExternalDataTests.DeepModelRoutes(path),
        _ => 1,
    };

    internal static int InAChildProcess(Dictionary<string, string?> environment, params string[] args)
    {
        var dotnet = Path.Combine(RuntimeEnvironment.GetRuntimeDirectory(), "..", "..", "..", Windows ? "dotnet.exe" : "dotnet");
        var start = new ProcessStartInfo(dotnet, [typeof(SideBySideBackendHardwareTests).Assembly.Location, .. args])
            { RedirectStandardOutput = true, RedirectStandardError = true };
        foreach (var (name, value) in environment) start.Environment[name] = value;
        using var child = Process.Start(start)!;
        child.BeginOutputReadLine();
        child.BeginErrorReadLine();
        child.WaitForExit();
        return child.ExitCode;
    }

    private static Dictionary<string, string?> WithNoOtherCudaMajorOnThePathAndNoPythonEnvironment() => new()
    {
        ["PATH"] = string.Join(Path.PathSeparator, (Environment.GetEnvironmentVariable("PATH") ?? "")
            .Split(Path.PathSeparator)
            .Where(folder => !Directory.Exists(folder)
                || !Directory.EnumerateFiles(folder, "cublasLt64_*.dll").Any(f => !f.EndsWith("_13.dll")))),
        ["SHOROKOO_PYTHON_ENV"] = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString()),
    };

    private static bool ConvolutionRuns(IShorokooBackend backend)
    {
        var x = InputTensor<float32>("x", rank: 4);
        var w = InputTensor<float32>("w", rank: 4);
        var conv = new InternalComputationGraph([x, w], [OnnxOp.Conv(x, w, null!, AutoPad.NotSet,
            dilations: [1L, 1L], group: 1, kernelShape: [3L, 3L], pads: [0L, 0L, 0L, 0L], strides: [1L, 1L])]);
        float[] image = [.. Enumerable.Range(0, 16).Select(i => (float)i)];
        return Floats(new ComputeContext(backend).Execute(conv,
                TensorData([1L, 1L, 4L, 4L], image), TensorData([1L, 1L, 3L, 3L], Enumerable.Repeat(1f, 9).ToArray()))[0])
            .Zip((float[])[45f, 54f, 81f, 90f]).All(p => Math.Abs(p.First - p.Second) < 1e-3f);
    }

    private static bool BothCudaBackendsRun(bool pytorchFirst)
    {
        IShorokooBackend[] inOrder = pytorchFirst ? [new TorchCudaBackend(), LoadCuda()] : [LoadCuda(), new TorchCudaBackend()];
        return inOrder.All(ConvolutionRuns) && OneCopyOfEachCudaLibrary();
    }

    private static int AConvolutionEndsInAResultOrAnException()
    {
        try { return ConvolutionRuns(LoadCuda()) ? 0 : 1; }
        catch (Exception) { return 0; }
    }

    private static int AConvolutionAfterAnotherCudnnIsRefusedNamingIt(string other)
    {
        var cudnn = CudaLibraryPins.Current!.Libraries.Single(pin => pin.Name == "cudnn");
        var shim = cudnn.Files.Single(file => file.FileName is "cudnn64_9.dll" or "libcudnn.so.9").FileName;
        var copy = Path.Combine(other, shim);
        File.WriteAllBytes(copy, [.. File.ReadAllBytes(Path.Combine(CudaLibraryCache.DefaultRoot, cudnn.CacheKey, shim)), 0]);
        NativeLibrary.Load(copy);
        try { return ConvolutionRuns(LoadCuda()) ? 2 : 1; }
        catch (InvalidOperationException refusal) when (refusal.Message.Contains(shim) && refusal.Message.Contains(other)) { return 0; }
    }

    private static bool OneCopyOfEachCudaLibrary()
        => Process.GetCurrentProcess().Modules.Cast<ProcessModule>()
            .Where(m => ((string[])["cublas", "cudnn", "libcublas", "libcudnn"])
                .Any(family => m.ModuleName.StartsWith(family, StringComparison.OrdinalIgnoreCase)))
            .GroupBy(m => m.ModuleName, StringComparer.OrdinalIgnoreCase)
            .All(copies => copies.Select(m => m.FileName).Distinct().Count() == 1);

    /// <summary>The allocator ONNX Runtime made this value's buffer from, and the device it is on.
    /// Through the backend's own types, which an isolated backend loads privately, so the route to
    /// them is reflection rather than a cast.</summary>
    private static (string Allocator, int DeviceId) AllocatorOf(IShorokooTensorValue value)
    {
        var inner = value.GetType()
            .GetProperty("Inner", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(value)!;
        using var info = (IDisposable)inner.GetType()
            .GetMethod("GetTensorMemoryInfo")!.Invoke(inner, null)!;
        return ((string)info.GetType().GetProperty("Name")!.GetValue(info)!,
                (int)info.GetType().GetProperty("Id")!.GetValue(info)!);
    }

    private static float[] Floats(TensorData data)
        => data.As<float32>().CopyMemory<float>();

    private static float[] Floats(NamedModelParam param) => Floats(param.ToTensorData());
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
