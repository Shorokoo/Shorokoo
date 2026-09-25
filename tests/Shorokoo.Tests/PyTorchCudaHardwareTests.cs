using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Shorokoo.Core.Backends;
using Shorokoo.Core.Factory.IR;
using Shorokoo.PyTorch.Cuda;
using Shorokoo.Runtime;

namespace Shorokoo.Tests;

/// <summary>
/// The PyTorch CUDA backend on a card. It runs in the CUDA 13 environment, which the first test
/// provisions (some 6 GB) unless <c>SHOROKOO_PYTHON_ENV</c> names one; and a process has one
/// Python environment, so run this class in a process where no CPU torch backend started first --
/// the Purpose=Hardware filter does.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Hardware")]
public class PyTorchCudaHardwareTests
{
    private static readonly Lazy<TorchCudaBackend> Cuda = new(() => new TorchCudaBackend());

    [TorchCudaFact]
    public void TestATensorLivesOnTheCardAndComesHomeByCopy()
    {
        byte[] bytes = [.. MemoryMarshal.AsBytes<float>([1f, 2f, 3f])];
        using var onCard = Cuda.Value.CreateTensorInBackendMemory(ShorokooTensorElementType.Float, bytes, [3]);
        using var onHost = Cuda.Value.CreateTensorFromRawBytes(ShorokooTensorElementType.Float, bytes, [3]);
        using var blank = Cuda.Value.CreateUninitializedTensorInBackendMemory(ShorokooTensorElementType.Int64, [4]);

        Assert.False(onCard.IsHostAccessible);
        Assert.True(onHost.IsHostAccessible);
        Assert.Equal(bytes, Cuda.Value.CopyTensorToHost(onCard));
        Assert.Throws<InvalidOperationException>(() => onCard.GetTensorDataAsSpan<float>().Length);
        Assert.False(blank.IsHostAccessible);
    }

    [TorchCudaFact]
    public void TestAModelRunsOnTheCardAndAgreesWithTheHost()
    {
        var (model, input) = SideBySideModel.Concrete();
        var onHost = SideBySideModel.Floats(new ComputeContext().Execute(model, input.Shared())[0]);
        using var context = new ComputeContext(Cuda.Value);

        SideBySideModel.AssertAgree(onHost, SideBySideModel.Floats(context.Execute(model, input.Shared())[0]), SideBySideModel.DeviceTolerance);
        SideBySideModel.AssertAgree(onHost, SideBySideModel.Floats(context.Compile(model).Execute(input)[0]), SideBySideModel.DeviceTolerance);
    }

    [TorchCudaFact]
    public void TestARetainedOutputStaysOnTheCardAndTheRestComeHome()
    {
        using var session = Cuda.Value.CreateSession(NegModel(), default, default, DeviceMemorySettings.Default);
        using var x = Cuda.Value.CreateTensor([1f, -2f], [2]);
        var inputs = new Dictionary<string, IShorokooTensorValue> { ["x"] = x };
        using var kept = session.RunRetainingOutputs(inputs, ["y"], new HashSet<string> { "y" }, RunSettings.Default)[0];
        using var fetched = session.Run(inputs, ["y"], RunSettings.Default)[0];

        Assert.True(session.HasDeviceMemory);
        Assert.Equal(SessionOutputPlacement.Device, session.OutputPlacement);
        Assert.False(kept.IsHostAccessible);
        Assert.Equal([-1f, 2f], fetched.GetTensorDataAsSpan<float>().ToArray());
        Assert.Equal(Cuda.Value.CopyTensorToHost(fetched), Cuda.Value.CopyTensorToHost(kept));
    }

    [TorchCudaFact]
    public void TestTheCardsDriverIsEnoughToProbeAndStartTheBackend()
    {
        Assert.Equal(BackendRejection.None, BackendPackage.Probe(typeof(TorchCudaBackend).Assembly.Location).Reason);
        Assert.Equal(3, Cuda.Value.Start().PythonVersion.Major);
    }

    [TorchCudaFact]
    public void TestTheSessionReadsTheCardsAllocatorAndALimitCapsWhatItsRunsMayAllocate()
    {
        const long mebibyte = 1L << 20;
        using var free = Cuda.Value.CreateSession(NegModel(), default, default, DeviceMemorySettings.Default);
        using var tight = Cuda.Value.CreateSession(NegModel(), default, default, new DeviceMemorySettings { LimitBytes = mebibyte });
        using var roomy = Cuda.Value.CreateSession(NegModel(), default, default, new DeviceMemorySettings { LimitBytes = 64 * mebibyte });
        var before = free.ReadArenaStatistics()!.Value;
        using var x = Cuda.Value.CreateUninitializedTensorInBackendMemory(ShorokooTensorElementType.Float, [4 * mebibyte]);
        var feeds = new Dictionary<string, IShorokooTensorValue> { ["x"] = x };
        IShorokooTensorValue Kept(IShorokooSession session) => session.RunRetainingOutputs(feeds, ["y"], new HashSet<string> { "y" }, RunSettings.Default)[0];
        var after = free.ReadArenaStatistics()!.Value;

        Assert.Equal(-1, before.LimitBytes);
        Assert.True(after.InUseBytes >= before.InUseBytes + 16 * mebibyte);
        Assert.True(after.MaxInUseBytes >= after.InUseBytes && after.TotalAllocatedBytes >= after.InUseBytes);
        Assert.True(after.AllocationCount > before.AllocationCount);
        Assert.Equal(mebibyte, tight.ReadArenaStatistics()!.Value.LimitBytes);
        Assert.Contains("LimitBytes", Assert.Throws<InvalidOperationException>(() => Kept(tight)).Message);
        using (var y = Kept(roomy)) Assert.False(y.IsHostAccessible);
        using (var y = Kept(free)) Assert.False(y.IsHostAccessible);
    }

    [TorchCudaFact]
    public void TestALimitIsWhatARunMayAllocateBeyondWhatTheAllocatorHoldsWhereItHoldsMoreThanItHasHandedOut()
    {
        const long mebibyte = 1L << 20;
        var smalls = Enumerable.Range(0, 4096).Select(_ => Cuda.Value.CreateUninitializedTensorInBackendMemory(ShorokooTensorElementType.Float, [128])).ToList();
        foreach (var small in smalls.SkipLast(1)) small.Dispose();
        using var kept = smalls[^1];
        using var session = Cuda.Value.CreateSession(NegModel(), default, default, new DeviceMemorySettings { LimitBytes = 17 * mebibyte });
        using var x = Cuda.Value.CreateUninitializedTensorInBackendMemory(ShorokooTensorElementType.Float, [4 * mebibyte]);
        using var y = session.RunRetainingOutputs(new Dictionary<string, IShorokooTensorValue> { ["x"] = x }, ["y"], new HashSet<string> { "y" }, RunSettings.Default)[0];

        Assert.False(y.IsHostAccessible);
    }

    [TorchCudaFact]
    public void TestARunWithoutALimitIsNotHeldToTheLimitOfAnotherSessionsRunOnTheCard()
    {
        using var capped = Cuda.Value.CreateSession(NegModel(), default, default, new DeviceMemorySettings { LimitBytes = 1L << 20 });
        using var free = Cuda.Value.CreateSession(NegModel(), default, default, DeviceMemorySettings.Default);
        using var small = Cuda.Value.CreateUninitializedTensorInBackendMemory(ShorokooTensorElementType.Float, [1024]);
        using var large = Cuda.Value.CreateUninitializedTensorInBackendMemory(ShorokooTensorElementType.Float, [16L << 20]);
        var until = DateTime.UtcNow + TimeSpan.FromSeconds(2);
        void Run(IShorokooSession session, IShorokooTensorValue x)
            => session.RunRetainingOutputs(new Dictionary<string, IShorokooTensorValue> { ["x"] = x }, ["y"], new HashSet<string> { "y" }, RunSettings.Default)[0].Dispose();
        var cappedRuns = Task.Run(() => { while (DateTime.UtcNow < until) Run(capped, small); });
        var failures = 0;
        while (DateTime.UtcNow < until)
            try { Run(free, large); }
            catch (InvalidOperationException) { failures++; }
        cappedRuns.Wait();

        Assert.Equal(0, failures);
    }

    [TorchCudaFact]
    public void TestARunAskedToShrinkHandsTheCardsCachedBlocksBack()
    {
        using var session = Cuda.Value.CreateSession(PyTorchBackendCoverageTests.Serialize(
            ComputeContextLifetimeCoverageTests.GraphOf("x", "y", Op("Neg", "x", "n"), Op("Neg", "n", "m"), Op("Neg", "m", "y"))),
            default, default, DeviceMemorySettings.Default);
        using var x = Cuda.Value.CreateUninitializedTensorInBackendMemory(ShorokooTensorElementType.Float, [4L << 20]);
        var feeds = new Dictionary<string, IShorokooTensorValue> { ["x"] = x };
        ArenaStatistics After(bool shrink)
        {
            session.Run(feeds, ["y"], new RunSettings { ShrinkArenaAfterRun = shrink })[0].Dispose();
            return session.ReadArenaStatistics()!.Value;
        }
        var kept = After(shrink: false);
        var shrunk = After(shrink: true);

        Assert.True(shrunk.TotalAllocatedBytes < kept.TotalAllocatedBytes);
        Assert.True(shrunk.ArenaShrinkageCount > kept.ArenaShrinkageCount);
    }

    [TorchCudaFact]
    public void TestAnOutputIsWrittenIntoTheConsumedInputWhereverTheOutputEndsUp()
    {
        var sub = PyTorchBackendCoverageTests.Serialize(ComputeContextLifetimeCoverageTests.GraphOf("a:float[4] b:float[4]", "O:float[4]", Op("Sub", "a b", "O")));
        var matmul = PyTorchBackendCoverageTests.Serialize(ComputeContextLifetimeCoverageTests.GraphOf("a:float[2,2] b:float[2,2]", "O:float[2,2]", Op("MatMul", "b a", "O")));
        using var session = Cuda.Value.CreateSession(sub, default, default, DeviceMemorySettings.Default, DiagnosticSettings.Default, [new OutputAlias("O", "a")]);
        using var product = Cuda.Value.CreateSession(matmul, default, default, DeviceMemorySettings.Default, DiagnosticSettings.Default, [new OutputAlias("O", "a")]);
        byte[] Bytes(params float[] values) => [.. MemoryMarshal.AsBytes<float>(values)];
        (string? Input, IShorokooTensorValue O) Run(IShorokooSession on, IShorokooTensorValue a, IShorokooTensorValue b, bool keep)
        {
            var o = on.RunConsuming(new Dictionary<string, IShorokooTensorValue> { ["a"] = a, ["b"] = b }, [a], ["O"],
                keep ? new HashSet<string> { "O" } : ComputeContext.NoOutputsRetained, RunSettings.Default, out var aliased)[0];
            return (aliased.Count == 0 ? null : aliased[0], o);
        }
        using var b = Cuda.Value.CreateTensorInBackendMemory(ShorokooTensorElementType.Float, Bytes(1f, 2f, 3f, 4f), [4]);
        var (onCard, kept) = Run(session, Cuda.Value.CreateTensorInBackendMemory(ShorokooTensorElementType.Float, Bytes(10f, 20f, 30f, 40f), [4]), b, keep: true);
        var home = Cuda.Value.CreateTensor([10f, 20f, 30f, 40f], [4]);
        ref var homeMemory = ref MemoryMarshal.GetReference(home.GetTensorDataAsSpan<float>());
        var (onHost, fetched) = Run(session, home, b, keep: false);
        using var identity = Cuda.Value.CreateTensor([1f, 0f, 0f, 1f], [2, 2]);
        var (copied, square) = Run(product, Cuda.Value.CreateTensor([1f, 2f, 3f, 4f], [2, 2]), identity, keep: false);

        Assert.Equal(("a", "a", "a"), (onCard, onHost, copied));
        Assert.False(kept.IsHostAccessible);
        Assert.Equal(Bytes(9f, 18f, 27f, 36f), Cuda.Value.CopyTensorToHost(kept));
        Assert.Equal([9f, 18f, 27f, 36f], fetched.GetTensorDataAsSpan<float>().ToArray());
        Assert.True(Unsafe.AreSame(ref homeMemory, ref MemoryMarshal.GetReference(fetched.GetTensorDataAsSpan<float>())));
        Assert.Equal([1f, 2f, 3f, 4f], square.GetTensorDataAsSpan<float>().ToArray());
        Assert.Equal([new OutputAlias("O", "a")], product.BindableAliases);
        kept.Dispose();
        fetched.Dispose();
        square.Dispose();
    }

    [TorchCudaFact]
    public void TestARunOnTheCardIsStoppedWhenItsTokenIsCancelledAndItsNodesAreTracedOnTheCard()
    {
        using var session = Cuda.Value.CreateSession(PyTorchBackendCoverageTests.Serialize(PyTorchBackendCoverageTests.CountingLoop()),
            default, default, DeviceMemorySettings.Default, new DiagnosticSettings { TraceNodePlacement = true });
        using var m = Cuda.Value.CreateTensor([10_000_000L], []);
        using var v = Cuda.Value.CreateTensor([0f], []);
        using var later = new CancellationTokenSource(TimeSpan.FromMilliseconds(300));

        Assert.Equal(later.Token, Assert.Throws<OperationCanceledException>(() => session.Run(
            new Dictionary<string, IShorokooTensorValue> { ["m"] = m, ["v"] = v }, ["y"], new RunSettings { CancellationToken = later.Token })).CancellationToken);
        Assert.Equal([new ProviderShare("cuda:0", 1, 0, 0, 0)], session.ReadNodePlacement()!.Providers);
    }

    private static NodeProto Op(string op, string inputs, string outputs) => ComputeContextLifetimeCoverageTests.Op(op, inputs, outputs);

    private static byte[] NegModel()
    {
        var node = new Shorokoo.Core.Factory.IR.NodeProto { OpType = "Neg", Name = "neg" };
        node.Inputs.Add("x");
        node.Outputs.Add("y");
        var graph = new Shorokoo.Core.Factory.IR.GraphProto { Name = "g" };
        graph.Inputs.Add(new Shorokoo.Core.Factory.IR.ValueInfoProto { Name = "x" });
        graph.Outputs.Add(new Shorokoo.Core.Factory.IR.ValueInfoProto { Name = "y" });
        graph.Nodes.Add(node);
        var model = new Shorokoo.Core.Factory.IR.ModelProto { IrVersion = 10, Graph = graph };
        model.OpsetImports.Add(new Shorokoo.Core.Factory.IR.OperatorSetIdProto { Domain = "", Version = 21 });
        using var stream = new MemoryStream();
        ProtoBuf.Serializer.Serialize(stream, model);
        return stream.ToArray();
    }
}

/// <summary>Runs a test only where an NVIDIA driver answers: the torch CUDA backend brings its own
/// CUDA libraries, so the driver is all the machine has to supply.</summary>
public sealed class TorchCudaFactAttribute : FactAttribute
{
    public TorchCudaFactAttribute()
    {
        if (!NativeLibrary.TryLoad(OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so.1", out var driver))
            Skip = "No NVIDIA driver on this machine, so there is no card for the PyTorch CUDA backend to run on.";
        else
            NativeLibrary.Free(driver);
    }
}

/// <summary>Runs a test only where no NVIDIA driver answers: what the PyTorch CUDA backend does on a
/// machine it cannot run on.</summary>
public sealed class NoCudaDriverFactAttribute : FactAttribute
{
    public NoCudaDriverFactAttribute()
    {
        if (NativeLibrary.TryLoad(OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so.1", out var driver))
        {
            NativeLibrary.Free(driver);
            Skip = "An NVIDIA driver is installed, so this machine is not one the PyTorch CUDA backend refuses.";
        }
    }
}
