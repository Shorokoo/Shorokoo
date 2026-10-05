using System.Runtime.InteropServices;
using Shorokoo.Core.Backends;
using Shorokoo.Core.Factory.IR;
using Shorokoo.PyTorch;
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
[Collection(ProcessWideMemory.Name)]
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
    public void TestFloat32ProductsConvolutionsAndRecurrentLayersOnTheCardAreComputedInFullPrecisionUnlessTensorFloat32IsAllowed()
    {
        var host = SideBySideModel.LargeLayers(new ComputeContext());
        using var strict = new ComputeContext(Cuda.Value);
        using var allowed = new ComputeContext(Cuda.Value) { Precision = SideBySideModel.AllowingTensorFloat32 };

        SideBySideModel.AssertFullPrecision(host, SideBySideModel.LargeLayers(strict));
        SideBySideModel.AssertTensorFloat32(host, SideBySideModel.LargeLayers(allowed));
        SideBySideModel.AssertFullPrecision(host, SideBySideModel.LargeLayers(strict));
        SideBySideModel.AssertTensorFloat32(host, SideBySideModel.LargeLayers(allowed));
    }

    [TorchCudaFact]
    public void TestRunsOfAFullPrecisionSessionAndATensorFloat32SessionAtOnceEachComputeInTheirOwnPrecision()
    {
        var host = SideBySideModel.LargeLayers(new ComputeContext());
        using var strict = new ComputeContext(Cuda.Value);
        using var allowed = new ComputeContext(Cuda.Value) { Precision = SideBySideModel.AllowingTensorFloat32 };
        var strictRuns = Task.Run(() => Enumerable.Range(0, 4).Select(_ => SideBySideModel.LargeLayers(strict)).ToList());
        var allowedRuns = Task.Run(() => Enumerable.Range(0, 4).Select(_ => SideBySideModel.LargeLayers(allowed)).ToList());

        Assert.All(strictRuns.Result, card => SideBySideModel.AssertFullPrecision(host, card));
        Assert.All(allowedRuns.Result, card => SideBySideModel.AssertTensorFloat32(host, card));
    }

    [TorchCudaFact]
    public void TestTwoTensorFloat32RunsOnTheCardHoldTheirPrecisionAtOnce()
    {
        using var first = (TorchSession)TensorFloat32Session();
        using var second = (TorchSession)TensorFloat32Session();
        var arrived = 0;
        var together = new System.Collections.Concurrent.ConcurrentBag<bool>();
        void Holding(TorchSession session)
        {
            if (session != first && session != second) return;
            Interlocked.Increment(ref arrived);
            together.Add(SpinWait.SpinUntil(() => Volatile.Read(ref arrived) == 2, PyTorchBackendCoverageTests.Patience));
        }
        TorchSession.HoldingPrecision += Holding;
        try { Task.WaitAll(Task.Run(() => NegRun(first, default)), Task.Run(() => NegRun(second, default))); }
        finally { TorchSession.HoldingPrecision -= Holding; }
        Assert.Equal([true, true], together);
    }

    [TorchCudaFact]
    public void TestARunWaitingForItsPrecisionOnTheCardIsStoppedByItsToken()
    {
        using var session = TensorFloat32Session();
        using var stop = new CancellationTokenSource();
        Task run;
        bool stoppedWhileWaiting;
        TorchSession.Float32Runs.Enter(false);
        try
        {
            run = Task.Run(() => NegRun(session, stop.Token));
            Assert.True(SpinWait.SpinUntil(() => TorchSession.Float32Runs.Waiting(true) == 1, PyTorchBackendCoverageTests.Patience));
            stop.Cancel();
            stoppedWhileWaiting = ((IAsyncResult)run).AsyncWaitHandle.WaitOne(PyTorchBackendCoverageTests.Patience);
        }
        finally { TorchSession.Float32Runs.Exit(false); }
        Assert.True(((IAsyncResult)run).AsyncWaitHandle.WaitOne(PyTorchBackendCoverageTests.Patience));
        Assert.Equal((true, typeof(OperationCanceledException)), (stoppedWhileWaiting, run.Exception?.InnerException?.GetType()));
    }

    private static IShorokooSession TensorFloat32Session()
        => Cuda.Value.CreateSession(NegModel(), default, default, DeviceMemorySettings.Default, DiagnosticSettings.Default, [], 0, [], SideBySideModel.AllowingTensorFloat32);

    private static void NegRun(IShorokooSession session, CancellationToken token)
    {
        using var x = Cuda.Value.CreateTensorInBackendMemory(ShorokooTensorElementType.Float, [.. MemoryMarshal.AsBytes<float>([1f, -2f])], [2]);
        session.Run(new Dictionary<string, IShorokooTensorValue> { ["x"] = x }, ["y"], new RunSettings { CancellationToken = token })[0].Dispose();
    }

    [TorchCudaFact]
    public void TestEveryOutputStaysOnTheCardAndAnInputInHostMemoryIsRefused()
    {
        using var session = Cuda.Value.CreateSession(NegModel(), default, default, DeviceMemorySettings.Default);
        using var x = Cuda.Value.CreateTensorInBackendMemory(ShorokooTensorElementType.Float, [.. MemoryMarshal.AsBytes<float>([1f, -2f])], [2]);
        using var onHost = Cuda.Value.CreateTensor([1f, -2f], [2]);
        using var y = session.Run(new Dictionary<string, IShorokooTensorValue> { ["x"] = x }, ["y"], RunSettings.Default)[0];

        Assert.Equal(SessionOutputPlacement.Device, session.OutputPlacement);
        Assert.False(y.IsHostAccessible);
        Assert.Equal([.. MemoryMarshal.AsBytes<float>([-1f, 2f])], Cuda.Value.CopyTensorToHost(y));
        Assert.Throws<InvalidOperationException>(() => session.Run(new Dictionary<string, IShorokooTensorValue> { ["x"] = onHost }, ["y"], RunSettings.Default));
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
        IShorokooTensorValue Kept(IShorokooSession session) => session.Run(feeds, ["y"], RunSettings.Default)[0];
        var after = free.ReadArenaStatistics()!.Value;

        Assert.Equal(-1, before.LimitBytes);
        Assert.True(after.InUseBytes >= before.InUseBytes + 16 * mebibyte);
        Assert.True(after.MaxInUseBytes >= after.InUseBytes && after.TotalAllocatedBytes >= after.InUseBytes);
        Assert.True(after.RequestedInUseBytes >= before.RequestedInUseBytes + 16 * mebibyte && after.RequestedInUseBytes <= after.InUseBytes);
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
        using var y = session.Run(new Dictionary<string, IShorokooTensorValue> { ["x"] = x }, ["y"], RunSettings.Default)[0];

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
            => session.Run(new Dictionary<string, IShorokooTensorValue> { ["x"] = x }, ["y"], RunSettings.Default)[0].Dispose();
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
            ComputeContextLifetimeCoverageTests.GraphOf("x", "y", Op("Neg", "x", "n"), Op("Neg", "n", "m"), Op("Add", "n m", "y"))),
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
    public void TestAnOutputIsWrittenIntoTheConsumedInputWhereItsOperatorCanOverwriteItAndProducedAsUsualWhereItCannot()
    {
        var sub = PyTorchBackendCoverageTests.Serialize(ComputeContextLifetimeCoverageTests.GraphOf("a:float[4] b:float[4]", "O:float[4]", Op("Sub", "a b", "O")));
        var matmul = PyTorchBackendCoverageTests.Serialize(ComputeContextLifetimeCoverageTests.GraphOf("a:float[2,2] b:float[2,2]", "O:float[2,2]", Op("MatMul", "b a", "O")));
        using var session = Cuda.Value.CreateSession(sub, default, default, DeviceMemorySettings.Default, DiagnosticSettings.Default, [new OutputAlias("O", "a")]);
        using var product = Cuda.Value.CreateSession(matmul, default, default, DeviceMemorySettings.Default, DiagnosticSettings.Default, [new OutputAlias("O", "a")]);
        byte[] Bytes(params float[] values) => [.. MemoryMarshal.AsBytes<float>(values)];
        IShorokooTensorValue OnCard(long[] shape, params float[] values) => Cuda.Value.CreateTensorInBackendMemory(ShorokooTensorElementType.Float, Bytes(values), shape);
        long Address(IShorokooTensorValue value)
        {
            using (Shorokoo.PythonHost.PythonRuntime.Gil())
            using (var pointer = ((TorchTensorValue)value).Value.InvokeMethod("data_ptr"))
                return pointer.As<long>();
        }
        (string? Input, IShorokooTensorValue O) Run(IShorokooSession on, IShorokooTensorValue a, IShorokooTensorValue b)
        {
            var o = on.RunConsuming(new Dictionary<string, IShorokooTensorValue> { ["a"] = a, ["b"] = b }, [a], ["O"],
                RunSettings.Default, out var aliased)[0];
            return (aliased.Count == 0 ? null : aliased[0], o);
        }
        using var b = OnCard([4], 1f, 2f, 3f, 4f);
        var (onCard, kept) = Run(session, OnCard([4], 10f, 20f, 30f, 40f), b);
        var consumed = OnCard([4], 10f, 20f, 30f, 40f);
        var consumedMemory = Address(consumed);
        var (inPlace, written) = Run(session, consumed, b);
        using var identity = OnCard([2, 2], 1f, 0f, 0f, 1f);
        var operand = OnCard([2, 2], 1f, 2f, 3f, 4f);
        var operandMemory = Address(operand);
        var (produced, square) = Run(product, operand, identity);

        Assert.Equal(("a", "a", null), (onCard, inPlace, produced));
        Assert.False(kept.IsHostAccessible);
        Assert.Equal(Bytes(9f, 18f, 27f, 36f), Cuda.Value.CopyTensorToHost(kept));
        Assert.False(written.IsHostAccessible);
        Assert.Equal(Bytes(9f, 18f, 27f, 36f), Cuda.Value.CopyTensorToHost(written));
        Assert.Equal(consumedMemory, Address(written));
        Assert.Equal(Bytes(1f, 2f, 3f, 4f), Cuda.Value.CopyTensorToHost(square));
        Assert.NotEqual(operandMemory, Address(square));
        Assert.Empty(product.BindableAliases);
        kept.Dispose();
        written.Dispose();
        square.Dispose();
    }

    [TorchCudaFact]
    public void TestAPlacedRunComputesInItsContextsPrecision()
    {
        var host = ComputeContextLifetimeCoverageTests.ProductIntoConsumedOnTheHost();
        using var strictContext = new ComputeContext(Cuda.Value);
        using var allowedContext = new ComputeContext(Cuda.Value) { Precision = SideBySideModel.AllowingTensorFloat32 };
        var strict = ComputeContextLifetimeCoverageTests.RunProductIntoConsumed(strictContext);
        var allowed = ComputeContextLifetimeCoverageTests.RunProductIntoConsumed(allowedContext);

        Assert.Equal((true, true), (strict.Placed, allowed.Placed));
        SideBySideModel.AssertFullPrecision([host], [strict.Values]);
        SideBySideModel.AssertTensorFloat32([host], [allowed.Values]);
    }

    [TorchCudaFact]
    public void TestTheTwoHalvesScenarioWritesEveryValueIntoTheMemoryItConsumesOnTheCardAndItsOutputsOutliveTheSession()
    {
        const int Rows = 512, Columns = 1024;
        var (a, b, l) = ComputeContextLifetimeCoverageTests.TwoHalvesValues(Rows, Columns);
        NamedModelParam[] outputs = [];
        using (var context = new ComputeContext(Cuda.Value))
        {
            var compiled = context.Compile(ComputeContextLifetimeCoverageTests.TwoHalves());
            for (int run = 0; run < 2; run++)
            {
                outputs = compiled.Execute(TensorData([(long)Rows, Columns], a).To(context), TensorData([(long)Rows, Columns], b).To(context));
                Assert.True(l.Zip(SideBySideModel.Floats(outputs[0]), (x, y) => MathF.Abs(x - y) < 1e-5f).All(x => x));
                Assert.Equal(a[..(Rows / 2 * Columns)], SideBySideModel.Floats(outputs[1]));
                Assert.Equal(b[(Rows / 2 * Columns)..], SideBySideModel.Floats(outputs[2]));
            }
            var entry = Assert.Single(((TorchSession)compiled.Session).Placements!.Entries);
            Assert.Equal(10, entry.Plan.Count);
            Assert.All(outputs, o => Assert.NotNull(o.ToTensorData().Block));
            Assert.Same(outputs[1].ToTensorData().Block, outputs[2].ToTensorData().Block);
        }
        GC.Collect();
        GC.WaitForPendingFinalizers();
        Assert.True(l.Zip(SideBySideModel.Floats(outputs[0]), (x, y) => MathF.Abs(x - y) < 1e-5f).All(x => x));
        Assert.Equal(b[(Rows / 2 * Columns)..], SideBySideModel.Floats(outputs[2]));
    }

    [TorchCudaFact]
    public void TestARunWritesSoftmaxesNormalizationsClipsConvolutionsAndGemmsIntoTheMemoryItConsumesOnTheCardAllocatingNoneOfThem()
        => PyTorchBackendCoverageTests.SoftmaxesNormalizationsClipsConvolutionsAndGemmsArePlaced(Cuda.Value);

    [TorchCudaFact]
    public void TestASumOverTheLeadingAxesOnTheCardHoldsLittleBeyondItsResult()
    {
        Assert.Equal((8192f, true), LeadingSum([128, 128, 512], 0, 1));
        Assert.Equal((515.5f, true), LeadingSum([1031, 512], 0));
    }

    /// <summary>The first element of a sum over the leading axes <paramref name="axes"/> of a
    /// <paramref name="shape"/> of 0.5s, and whether the run held under a mebibyte beyond what it
    /// held as it started.</summary>
    private static (float First, bool Little) LeadingSum(long[] shape, params long[] axes)
    {
        var dims = string.Join(",", shape);
        var graph = ComputeContextLifetimeCoverageTests.WithInts(ComputeContextLifetimeCoverageTests.GraphOf($"x:float[{dims}]", "y",
            ComputeContextLifetimeCoverageTests.Op("ReduceSum", "x axes", "y")), "axes", axes);
        using var session = Cuda.Value.CreateSession(PyTorchBackendCoverageTests.Serialize(graph), default, default, DeviceMemorySettings.Default);
        using var x = Cuda.Value.CreateTensorInBackendMemory(ShorokooTensorElementType.Float,
            [.. MemoryMarshal.AsBytes<float>(Enumerable.Repeat(0.5f, (int)shape.Aggregate(1L, (a, d) => a * d)).ToArray())], shape);
        var feeds = new Dictionary<string, IShorokooTensorValue> { ["x"] = x };
        session.Run(feeds, ["y"], RunSettings.Default)[0].Dispose();
        var (peak, y) = PyTorchBackendCoverageTests.CardPeak(() => session.Run(feeds, ["y"], RunSettings.Default)[0]);
        using (y) return (BitConverter.ToSingle(Cuda.Value.CopyTensorToHost(y), 0), peak < 1 << 20);
    }

    [TorchCudaFact]
    public void TestASumOverTheLeadingAxesOfSixteenBitFloatsOnTheCardAddsUpInFloat32()
    {
        Assert.Equal(8f, SixteenBitSum(ShorokooTensorElementType.Float16));
        Assert.Equal(8f, SixteenBitSum(ShorokooTensorElementType.BFloat16));
    }

    /// <summary>The sum over its rows of a column of 1024 sixteen-bit floats whose partial sums of
    /// 32 rows each are no sixteen-bit float, and whose sum is 8.</summary>
    private static float SixteenBitSum(ShorokooTensorElementType type)
    {
        float[] values = [.. Enumerable.Range(0, 1024).Select(r => (r / 32, r % 32) switch
        {
            (0, var q) => q % 2 == 0 ? 1024f : -1024f,
            (var b, var q) => q % 2 == 0 ? 0.5f : b == 31 ? 0f : -0.5f,
        })];
        byte[] Bytes(float v) => type == ShorokooTensorElementType.Float16
            ? BitConverter.GetBytes(BitConverter.HalfToUInt16Bits((Half)v))
            : BitConverter.GetBytes((ushort)(BitConverter.SingleToUInt32Bits(v) >> 16));
        var graph = ComputeContextLifetimeCoverageTests.WithInts(ComputeContextLifetimeCoverageTests.GraphOf(
            $"x:{(type == ShorokooTensorElementType.Float16 ? "float16" : "bfloat16")}[1024,1]", "y",
            ComputeContextLifetimeCoverageTests.Op("ReduceSum", "x axes", "y")), "axes", 0);
        using var session = Cuda.Value.CreateSession(PyTorchBackendCoverageTests.Serialize(graph), default, default, DeviceMemorySettings.Default);
        using var x = Cuda.Value.CreateTensorInBackendMemory(type, [.. values.SelectMany(Bytes)], [1024, 1]);
        using var y = session.Run(new Dictionary<string, IShorokooTensorValue> { ["x"] = x }, ["y"], RunSettings.Default)[0];
        var bits = BitConverter.ToUInt16(Cuda.Value.CopyTensorToHost(y), 0);
        return type == ShorokooTensorElementType.Float16 ? (float)BitConverter.UInt16BitsToHalf(bits) : BitConverter.UInt32BitsToSingle((uint)bits << 16);
    }

    [TorchCudaFact]
    public void TestACardTensorPastTwoGibibytesIsSavedByThePiece()
        => Assert.Equal((StagedReadBack.StagingBytes, true), SavedPastTwoGibibytes(Cuda.Value));

    [TorchCudaFact]
    public void TestATensorPastTwoGibibytesIsLoadedOntoTheCardByThePiece()
        => Assert.Equal((StagedReadBack.StagingBytes, StagedReadBack.StagingBytes, true), LoadedPastTwoGibibytes(Cuda.Value));

    [TorchCudaFact]
    public void TestAPieceOfACardTensorIsWrittenAndReadAtAnyByteOffset()
        => Assert.Equal(PieceWrittenAtFive, PieceWrittenIntoACardTensor(Cuda.Value));

    /// <summary>Six bytes read three bytes into four zero floats after 1, 2, 3, 4 were written five
    /// bytes in, and then the whole of them.</summary>
    internal static readonly byte[] PieceWrittenAtFive = [0, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0];

    /// <summary>What <see cref="PieceWrittenAtFive"/> describes, of a tensor on
    /// <paramref name="backend"/>'s card, once a piece past its end is refused.</summary>
    internal static byte[] PieceWrittenIntoACardTensor(IShorokooBackend backend)
    {
        using var value = backend.CreateTensorInBackendMemory(ShorokooTensorElementType.Float, new byte[16], [4]);
        var read = new byte[6];
        Assert.True(backend.TryCopyHostToTensorRange(value, 5, [1, 2, 3, 4]));
        Assert.True(backend.TryCopyTensorRangeToHost(value, 3, read));
        Assert.Throws<ArgumentOutOfRangeException>(() => backend.TryCopyTensorRangeToHost(value, 14, read));
        return [.. read, .. backend.CopyTensorToHost(value)];
    }

    /// <summary>The int32s a tensor past 2 GiB holds: 0, 1, 2 and on.</summary>
    private const long CountedPastTwoGibibytes = 5L << 27;

    /// <summary>The most bytes one write of a save of a card tensor of <paramref name="backend"/>
    /// holding <see cref="CountedPastTwoGibibytes"/> int32s counting from 0 handed its stream, and
    /// whether the stream was handed exactly those.</summary>
    internal static (int Largest, bool Counted) SavedPastTwoGibibytes(IShorokooBackend backend)
    {
        const long Half = CountedPastTwoGibibytes / 2;
        var graph = ComputeContextLifetimeCoverageTests.GraphOf($"x:int32[{Half}] h", "y",
            ComputeContextLifetimeCoverageTests.Op("Add", "x h", "u"), ComputeContextLifetimeCoverageTests.Op("Concat", "x u", "y", attribute: ("axis", 0)));
        using var session = backend.CreateSession(PyTorchBackendCoverageTests.Serialize(graph), default, default, DeviceMemorySettings.Default);
        IShorokooTensorValue y;
        using (var x = backend.CreateTensorInBackendMemory(ShorokooTensorElementType.Int32, Counting(Half), [Half]))
        using (var h = backend.CreateTensorInBackendMemory(ShorokooTensorElementType.Int32, BitConverter.GetBytes((int)Half), []))
            y = session.Run(new Dictionary<string, IShorokooTensorValue> { ["x"] = x, ["h"] = h }, ["y"], RunSettings.Default)[0];
        var tensor = TensorData.Create(new Shape([CountedPastTwoGibibytes]), DType.Int32, y, backend);
        var saved = new CountingStream(CountedPastTwoGibibytes);
        try { tensor.WriteContentTo(saved); }
        finally { tensor.Delete(); }
        return (saved.Largest, saved.Counted);
    }

    /// <summary>The most bytes one read of a load onto <paramref name="backend"/>'s card of
    /// <see cref="CountedPastTwoGibibytes"/> int32s counting from 0 asked its stream for, the most
    /// one write of a save of what it loaded handed its own, and whether that save was handed exactly
    /// those int32s.</summary>
    internal static (int Read, int Written, bool Counted) LoadedPastTwoGibibytes(IShorokooBackend backend)
    {
        using var context = new ComputeContext(backend);
        var source = new CountingStream(CountedPastTwoGibibytes);
        var tensor = context.ReadTensor(new Shape([CountedPastTwoGibibytes]), DType.Int32, source);
        var saved = new CountingStream(CountedPastTwoGibibytes);
        try { tensor.WriteContentTo(saved); }
        finally { tensor.Delete(); }
        return (source.Largest, saved.Largest, saved.Counted);
    }

    private static byte[] Counting(long count)
    {
        var bytes = new byte[4 * count];
        var ints = MemoryMarshal.Cast<byte, int>(bytes.AsSpan());
        for (int i = 0; i < ints.Length; i++) ints[i] = i;
        return bytes;
    }

    /// <summary>A stream of the int32s 0, 1, 2 and on, as many as it is made with: read, it hands
    /// them out; written, it checks it is handed them, in order. It keeps the most bytes one call
    /// asked for or handed it.</summary>
    private sealed class CountingStream(long count) : Stream
    {
        private long _position;
        private bool _wrong;
        public int Largest { get; private set; }
        public bool Counted => !_wrong && _position == 4 * count;

        public override int Read(Span<byte> buffer)
        {
            Largest = Math.Max(Largest, buffer.Length);
            var length = (int)Math.Min(buffer.Length, 4 * count - _position) & ~3;
            var ints = MemoryMarshal.Cast<byte, int>(buffer[..length]);
            for (int i = 0; i < ints.Length; i++) ints[i] = (int)(_position / 4 + i);
            _position += length;
            return length;
        }

        public override void Write(ReadOnlySpan<byte> buffer)
        {
            Largest = Math.Max(Largest, buffer.Length);
            if (_position % 4 != 0 || buffer.Length % 4 != 0) _wrong = true;
            var ints = MemoryMarshal.Cast<byte, int>(buffer[..(buffer.Length & ~3)]);
            for (int i = 0; i < ints.Length && !_wrong; i++) _wrong = ints[i] != _position / 4 + i;
            _position += buffer.Length;
        }

        public override int Read(byte[] buffer, int offset, int count) => Read(buffer.AsSpan(offset, count));
        public override void Write(byte[] buffer, int offset, int count) => Write(buffer.AsSpan(offset, count));
        public override bool CanRead => true;
        public override bool CanSeek => false;
        public override bool CanWrite => true;
        public override long Length => throw new NotSupportedException();
        public override long Position { get => _position; set => throw new NotSupportedException(); }
        public override void Flush() { }
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
    }

    [TorchCudaFact]
    public void TestAConvolutionOfTwoTransposedViewsIsComputedAsTheWeightGradientItIsOnTheCard()
    {
        Assert.Equal("True True", PyTorchBackendCoverageTests.WeightGradient(batch: 4, sizes: 9, kernel: 3, stride: 1, dilation: 1, pad: 1, Cuda.Value));
        Assert.Equal("True True", PyTorchBackendCoverageTests.WeightGradient(batch: 2, sizes: 11, kernel: 3, stride: 2, dilation: 2, pad: 2, Cuda.Value));
    }

    [TorchCudaFact]
    public void TestAnElementWiseChainOnTheCardHoldsOneOfItsValuesAtATime()
    {
        const long Count = 1L << 22;
        Assert.Equal(Count * 4, ChainPeak($"x:float[{Count}]", ComputeContextLifetimeCoverageTests.Op("Neg", "x", "b"),
            ComputeContextLifetimeCoverageTests.Op("Abs", "b", "c"), ComputeContextLifetimeCoverageTests.Op("Sigmoid", "c", "d")));
        Assert.Equal(Count * 5, ChainPeak("x:float[1024,4096] h:float[1]", ComputeContextLifetimeCoverageTests.Op("Neg", "x", "b"),
            ComputeContextLifetimeCoverageTests.Op("Softmax", "b", "c"), ComputeContextLifetimeCoverageTests.Op("Greater", "c h", "m"),
            ComputeContextLifetimeCoverageTests.Op("Where", "m c h", "d")));
    }

    /// <summary>The most a run of <paramref name="nodes"/> holds on the card beyond its inputs, each
    /// input zeros but a 0.5 of one element, once its output's first two elements are shown to be
    /// 0.5.</summary>
    private static long ChainPeak(string inputs, params NodeProto[] nodes)
    {
        var graph = ComputeContextLifetimeCoverageTests.GraphOf(inputs, "d", nodes);
        using var session = Cuda.Value.CreateSession(PyTorchBackendCoverageTests.Serialize(graph), default, default, DeviceMemorySettings.Default);
        var feeds = new Dictionary<string, IShorokooTensorValue>();
        foreach (var input in graph.Inputs)
        {
            long[] shape = [.. input.Type.TensorType.Shape.Dims.Select(d => d.DimValue)];
            var count = shape.Aggregate(1L, (a, d) => a * d);
            feeds[input.Name] = Cuda.Value.CreateTensorInBackendMemory(ShorokooTensorElementType.Float,
                count == 1 ? [.. MemoryMarshal.AsBytes<float>([0.5f])] : new byte[count * 4], shape);
        }
        try
        {
            session.Run(feeds, ["d"], RunSettings.Default)[0].Dispose();
            var (peak, d) = PyTorchBackendCoverageTests.CardPeak(() => session.Run(feeds, ["d"], RunSettings.Default)[0]);
            using (d) Assert.Equal([.. MemoryMarshal.AsBytes<float>([0.5f, 0.5f])], Cuda.Value.CopyTensorToHost(d)[..8]);
            return peak;
        }
        finally
        {
            foreach (var feed in feeds.Values) feed.Dispose();
        }
    }

    [TorchCudaFact]
    public void TestClipHandsBackAZeroInsideItsBoundsAndTheLowerOfTwoEqualZeroBoundsBelowThemSignAndAllOnTheCard()
        => PyTorchBackendCoverageTests.ClipKeepsTheSignOfAZero(Cuda.Value);

    [TorchCudaFact]
    public void TestARunOnTheCardIsStoppedWhenItsTokenIsCancelledAndItsNodesAreTracedOnTheCard()
    {
        using var session = Cuda.Value.CreateSession(PyTorchBackendCoverageTests.Serialize(PyTorchBackendCoverageTests.CountingLoop()),
            default, default, DeviceMemorySettings.Default, new DiagnosticSettings { TraceNodePlacement = true });
        using var m = Cuda.Value.CreateTensorInBackendMemory(ShorokooTensorElementType.Int64, [.. MemoryMarshal.AsBytes<long>([10_000_000L])], []);
        using var v = Cuda.Value.CreateTensorInBackendMemory(ShorokooTensorElementType.Float, [.. MemoryMarshal.AsBytes<float>([0f])], []);
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
