using System.Reflection;
using System.Runtime.InteropServices;
using Shorokoo.Core.Backends;
using Shorokoo.Modules.Losses;
using Shorokoo.Modules.Optimizers;
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
        Assert.True(compiled.HasDeviceMemory);

        // Left where the provider put it: this is the one kind of tensor the host cannot read.
        var onCard = compiled.Execute([ta, tb.Shared()], [true])[0].ToTensorData();
        Assert.Equal(MemoryKind.Cuda, onCard.Space.Kind);
        Assert.False(onCard.IsHostResident);
        Assert.Throws<InvalidOperationException>(() => onCard.As<float32>().AccessMemory<float>());

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
            [new TensorDataModelParam("input", ModelParamType.InputParam, TensorData([4L], [1f, 2f, 3f, 4f]))],
            new AdamWOptimizerHyperparameters { LearningRate = 0.1f },
            runtimeContext: cuda);

        var checkpoint = rig.CreateInitialCheckpoint();
        var (input, target) = (TrainingRigHelpers.InBatch(1f, 2f, 3f, 4f),
                               TrainingRigHelpers.TargetBatch(2f, 4f, 6f, 8f));

        var compiled = rig.RuntimeContext.Compile(rig.TrainingStepPureGraph);
        Assert.True(compiled.HasDeviceMemory);

        var retained = compiled.Execute(
            ComputeContext.ExpandStructInputs(
                [checkpoint.TrainableParams.Shared(), checkpoint.ModelState.Shared(), checkpoint.OptimizerState.Shared(),
                 input.Shared(), target.Shared()]),
            [.. Enumerable.Repeat(true, compiled.OutputCount)]);

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

        // And the loop built on all this still trains, publishing state the host can read.
        using var run = rig.BeginResidentRun(checkpoint);
        run.Step(input.Shared(), target.Shared());
        var published = run.StepToCheckpoint(input, target);

        Assert.Equal(2, published.Step);
        Assert.All(published.TrainableParams.Fields.Values,
            f => Assert.Equal(MemorySpace.Host, ((TensorData)f).Space));
        Assert.NotEmpty(TrainingRigHelpers.FlattenStruct(published.TrainableParams));
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
        Assert.Throws<InvalidOperationException>(() => onCard.As<float32>().AccessMemory<float>());

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

        Assert.True(FillOnDevice(device, written));
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
    public void TestASequenceRefusesTensorsTheCardHoldsRatherThanBecomingUnreadable()
    {
        var cuda = LoadCuda();
        IShorokooTensorValue OnCard() => cuda.CreateUninitializedTensorInBackendMemory(
            ShorokooTensorElementType.Float, [2L]);
        IShorokooTensorValue OnHost() => cuda.CreateTensor<float>([1f, 2f], [2L]);

        // A refusal takes over what it was handed, as the contract requires of any failure, so
        // every attempt is handed values of its own: reusing one reads a released value.
        var refused = Assert.Throws<InvalidOperationException>(
            () => cuda.CreateSequence([OnHost(), OnCard()]));
        Assert.Contains("CopyTensorToHost", refused.Message);
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

    /// <summary>Writes <paramref name="bytes"/> across the bus into the value's own allocation,
    /// through the address and the copy the backend itself uses.</summary>
    private static bool FillOnDevice(IShorokooTensorValue value, byte[] bytes)
    {
        var backend = value.GetType().Assembly;
        var address = backend.GetType("Shorokoo.OnnxRuntime.OrtBackend")!
            .GetMethod("DevicePointer", BindingFlags.Static | BindingFlags.NonPublic)!
            .Invoke(null, [value])!;
        var copied = (bool)backend.GetType("Shorokoo.OnnxRuntime.CudaInterop")!
            .GetMethod("CopyHostToDevice", BindingFlags.Static | BindingFlags.Public)!
            .Invoke(null, [bytes, address, bytes.Length])!;
        GC.KeepAlive(value);
        return copied;
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
