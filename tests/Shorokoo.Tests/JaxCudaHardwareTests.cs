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
    public void TestFloat32ProductsConvolutionsAndRecurrentLayersOnTheCardAreComputedInFullPrecisionUnlessTensorFloat32IsAllowedWhichXlaChoosesForAConvolution()
    {
        var host = SideBySideModel.LargeLayers(new ComputeContext());
        using var strict = new ComputeContext(Cuda.Value);
        using var allowed = new ComputeContext(Cuda.Value) { Precision = SideBySideModel.AllowingTensorFloat32 };
        var tensorFloat32 = SideBySideModel.LargeLayers(allowed);

        SideBySideModel.AssertFullPrecision(host, SideBySideModel.LargeLayers(strict));
        SideBySideModel.AssertTensorFloat32([host[0], host[2]], [tensorFloat32[0], tensorFloat32[2]]);
        Assert.InRange(SideBySideModel.Deviation(host[1], tensorFloat32[1]), 0, SideBySideModel.TensorFloat32Tolerance);
    }

    [JaxCudaFact]
    public void TestEveryOutputStaysOnTheCardAnInputInHostMemoryIsRefusedAndTheSessionReadsTheAllocator()
    {
        using var session = Cuda.Value.CreateSession(PyTorchBackendCoverageTests.Onnx("Neg", 1), default, LogSettings.Default, DeviceMemorySettings.Default);
        using var x = Cuda.Value.CreateTensorInBackendMemory(ShorokooTensorElementType.Float, [.. MemoryMarshal.AsBytes<float>([1f, -2f])], [2]);
        using var onHost = Cuda.Value.CreateTensor([1f, -2f], [2]);
        using var y = session.Run(new Dictionary<string, IShorokooTensorValue> { ["x0"] = x }, ["y"], RunSettings.Default)[0];

        Assert.Equal(SessionOutputPlacement.Device, session.OutputPlacement);
        Assert.False(y.IsHostAccessible);
        Assert.Equal([.. MemoryMarshal.AsBytes<float>([-1f, 2f])], Cuda.Value.CopyTensorToHost(y));
        Assert.Throws<InvalidOperationException>(() => session.Run(new Dictionary<string, IShorokooTensorValue> { ["x0"] = onHost }, ["y"], RunSettings.Default));
        Assert.True(session.ReadArenaStatistics()!.Value.InUseBytes > 0);
    }

    [JaxCudaFact]
    public void TestACardTensorPastTwoGibibytesIsSavedByThePiece()
        => Assert.Equal((StagedReadBack.StagingBytes, true), PyTorchCudaHardwareTests.SavedPastTwoGibibytes(Cuda.Value));

    [JaxCudaFact]
    public void TestATensorPastTwoGibibytesIsLoadedOntoTheCardByThePiece()
        => Assert.Equal((StagedReadBack.StagingBytes, StagedReadBack.StagingBytes, true), PyTorchCudaHardwareTests.LoadedPastTwoGibibytes(Cuda.Value));

    [JaxCudaFact]
    public void TestAPieceOfACardTensorIsWrittenAndReadAtAnyByteOffset()
        => Assert.Equal(PyTorchCudaHardwareTests.PieceWrittenAtFive, PyTorchCudaHardwareTests.PieceWrittenIntoACardTensor(Cuda.Value));

    [JaxCudaFact]
    public void TestACardTensorReadWholeMakesNoCopyOfItOnTheCard()
        => Assert.Equal((0L, 0f, 16777215f), ReadWhole(1 << 24));

    [JaxCudaFact]
    public void TestASmallCardTensorIsSavedLoadedReadAndGatheredCompilingNothing()
        => Assert.Equal(0, Compiles(() => { foreach (var count in (int[])[3, 5, 7, 11]) RoundTrip(count); }));

    [JaxCudaFact]
    public void TestRowsGatheredFromALargeCardTensorCompileOneProgramPerPowerOfTwoOfTheirElements()
        => Assert.Equal(3, Compiles(() => GatherRuns(16384, 1024, 12)));

    private static (long Allocations, float First, float Last) ReadWhole(int count)
    {
        using var session = Cuda.Value.CreateSession(PyTorchBackendCoverageTests.Serialize(
            ComputeContextLifetimeCoverageTests.GraphOf("x", "y", ComputeContextLifetimeCoverageTests.Op("Neg", "x", "y"))), default, LogSettings.Default, DeviceMemorySettings.Default);
        var tensor = OnCard(count, 1);
        var before = session.ReadArenaStatistics()!.Value.AllocationCount;
        var read = tensor.CopyMemory<float>();
        var made = session.ReadArenaStatistics()!.Value.AllocationCount - before;
        tensor.Delete();
        return (made, read[0], read[^1]);
    }

    private static TensorData OnCard(int rows, int columns)
    {
        var values = new float[rows * columns];
        for (int i = 0; i < values.Length; i++) values[i] = i;
        return TensorData.Create(new Shape([rows, columns]), DType.Float32,
            Cuda.Value.CreateTensorInBackendMemory(ShorokooTensorElementType.Float, [.. MemoryMarshal.AsBytes<float>(values)], [rows, columns]), Cuda.Value);
    }

    private static void RoundTrip(int count)
    {
        using var context = new ComputeContext(Cuda.Value);
        var tensor = OnCard(count, 2);
        var saved = new MemoryStream();
        tensor.WriteContentTo(saved);
        tensor.CopyMemory<float>();
        tensor.TryCopyRows([1, 0], 8, new byte[16]);
        tensor.Delete();
        saved.Position = 0;
        context.ReadTensor(new Shape([count, 2]), DType.Float32, saved).Delete();
    }

    private static void GatherRuns(int rows, int columns, int longest)
    {
        var tensor = OnCard(rows, columns);
        for (int run = 1; run <= longest; run++)
            tensor.TryCopyRows([.. Enumerable.Range(run * longest, run)], 4 * columns, new byte[4 * columns * run]);
        tensor.Delete();
    }

    private static int Compiles(Action act)
    {
        Cuda.Value.Start();
        int Count()
        {
            using (Shorokoo.PythonHost.PythonRuntime.Gil())
            {
                using var scope = Python.Runtime.Py.CreateScope();
                scope.Exec("""
                    import sys, types, jax
                    if "shorokoo_compiles" not in sys.modules:
                        counter = types.ModuleType("shorokoo_compiles")
                        counter.count = 0
                        def heard(event, duration, **kwargs):
                            if event == "/jax/core/compile/backend_compile_duration":
                                counter.count += 1
                        jax.monitoring.register_event_duration_secs_listener(heard)
                        sys.modules["shorokoo_compiles"] = counter
                    count = sys.modules["shorokoo_compiles"].count
                    """);
                return scope.Get<int>("count");
            }
        }
        var before = Count();
        act();
        return Count() - before;
    }

    [JaxCudaFact]
    public void TestARunOnTheCardWritesItsOutputOverTheConsumedInputItDonatesDeletingItAndLeavesALentInputWhole()
    {
        var graph = ComputeContextLifetimeCoverageTests.GraphOf("a:float[2] b:float[2]", "O:float[2]", ComputeContextLifetimeCoverageTests.Op("Sub", "a b", "O"));
        using var session = Cuda.Value.CreateSession(PyTorchBackendCoverageTests.Serialize(graph), default, LogSettings.Default, DeviceMemorySettings.Default,
            DiagnosticSettings.Default, [new OutputAlias("O", "a")]);
        IShorokooTensorValue Card(params float[] values) => Cuda.Value.CreateTensorInBackendMemory(ShorokooTensorElementType.Float, [.. MemoryMarshal.AsBytes<float>(values)], [values.Length]);
        string Host(IShorokooTensorValue value) => string.Join(",", MemoryMarshal.Cast<byte, float>(Cuda.Value.CopyTensorToHost(value)).ToArray());
        var consumed = Card(5f, 7f);
        using var lent = Card(5f, 7f);
        using var b = Card(1f, 2f);
        var scope = Watching(("consumed", consumed), ("lent", lent));

        using var written = session.RunConsuming(new Dictionary<string, IShorokooTensorValue> { ["a"] = consumed, ["b"] = b }, [consumed], ["O"], RunSettings.Default, out var aliased)[0];
        using var computed = session.RunConsuming(new Dictionary<string, IShorokooTensorValue> { ["a"] = lent, ["b"] = b }, [], ["O"], RunSettings.Default, out var notAliased)[0];

        Assert.Equal(["a"], aliased);
        Assert.Empty(notAliased);
        Assert.Equal("4,5 4,5 5,7", $"{Host(written)} {Host(computed)} {Host(lent)}");
        Assert.Equal("True False", Deleted(scope, "consumed", "lent"));
    }

    [JaxCudaFact]
    public void TestAResidentRunOnTheCardWritesTheStateItOwnsInPlaceCopiesTheStateItWasLentAndKeepsEveryHeldCheckpoint()
    {
        var donating = TrainingRigHelpers.JaxDonatingRun(Cuda.Value, null, aliasing: true);
        var plain = TrainingRigHelpers.JaxDonatingRun(Cuda.Value, null, aliasing: false);
        var native = TrainingRigHelpers.JaxDonatingRun(Cuda.Value, TrainingBackend.Native, aliasing: true);

        Assert.Equal(plain.Trained, donating.Trained);
        Assert.Equal(donating.Kept, donating.KeptAfter);
        Assert.Equal(native.Kept, native.KeptAfter);
        Assert.Equal(16L, donating.Aliased);
        Assert.Equal(16L, native.Aliased);
        Assert.Equal(0L, plain.Aliased);
    }

    /// <summary>A scope holding the arrays of <paramref name="values"/> under their names, which
    /// outlive the values' own release, so a test can ask after the run what became of them.</summary>
    private static Python.Runtime.PyModule Watching(params (string Name, IShorokooTensorValue Value)[] values)
    {
        using (Shorokoo.PythonHost.PythonRuntime.Gil())
        {
            var scope = Python.Runtime.Py.CreateScope();
            foreach (var (name, value) in values) scope.Set(name, ((Shorokoo.Jax.JaxTensorValue)value).Value);
            return scope;
        }
    }

    /// <summary>Whether each array <paramref name="scope"/> holds under <paramref name="names"/> was
    /// deleted, and lets go of the scope.</summary>
    private static string Deleted(Python.Runtime.PyModule scope, params string[] names)
    {
        using (Shorokoo.PythonHost.PythonRuntime.Gil())
        using (scope)
        {
            scope.Exec($"deleted = ' '.join(str(a.is_deleted()) for a in [{string.Join(", ", names)}])");
            return scope.Get<string>("deleted");
        }
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
