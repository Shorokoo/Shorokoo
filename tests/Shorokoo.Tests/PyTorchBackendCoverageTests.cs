using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Python.Runtime;
using Shorokoo.Core.Backends;
using Shorokoo.Core.Factory.IR;
using Shorokoo.PythonHost;
using Shorokoo.PyTorch;
using Shorokoo.PyTorch.Cpu;
using Shorokoo.PyTorch.Cuda;
using Shorokoo.Runtime;

namespace Shorokoo.Tests;

[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class PyTorchBackendCoverageTests
{
    private static readonly TorchCpuBackend Torch = new();

    internal static readonly ShorokooTensorElementType[] EveryTorchType =
    [
        ShorokooTensorElementType.Float, ShorokooTensorElementType.UInt8, ShorokooTensorElementType.Int8,
        ShorokooTensorElementType.UInt16, ShorokooTensorElementType.Int16, ShorokooTensorElementType.Int32,
        ShorokooTensorElementType.Int64, ShorokooTensorElementType.Bool, ShorokooTensorElementType.Float16,
        ShorokooTensorElementType.Double, ShorokooTensorElementType.UInt32, ShorokooTensorElementType.UInt64,
        ShorokooTensorElementType.Complex64, ShorokooTensorElementType.Complex128, ShorokooTensorElementType.BFloat16,
        ShorokooTensorElementType.Float8E4M3FN, ShorokooTensorElementType.Float8E4M3FNUZ,
        ShorokooTensorElementType.Float8E5M2, ShorokooTensorElementType.Float8E5M2FNUZ,
    ];

    [Fact]
    public void TestTheEnvironmentIsTheExplicitOneThenTheVariablesThenTheProvisionedOne()
    {
        var running = Torch.Start();
        var explicitPath = new PythonEnvironmentOptions { EnvironmentPath = running.Directory };
        var elsewhere = new PythonEnvironmentOptions { EnvironmentPath = "/nonexistent" };

        Assert.Equal(PythonEnvironmentSource.Explicit, Resolve(explicitPath, running.Directory).Source);
        Assert.Equal(PythonEnvironmentSource.EnvironmentVariable, Resolve(new(), running.Directory).Source);
        Assert.Equal(running.Directory, Resolve(new(), running.Directory).Directory);
        Assert.Throws<PythonEnvironmentException>(() => Resolve(elsewhere, running.Directory));
        Assert.Equal(3, running.PythonVersion.Major);
        Assert.Equal(12, running.PythonVersion.Minor);
        Assert.True(File.Exists(running.LibPython));
        Assert.Equal(running.Directory, new TorchCpuBackend(new() { EnvironmentPath = running.Directory + Path.DirectorySeparatorChar }).Start().Directory);
        Assert.Equal(OperatingSystem.IsWindows() ? Path.Combine(running.SitePackages, "torch", "lib") : null, running.CudaLibraryDirectory);
    }

    [Fact]
    public void TestEveryWayOfNotHavingAnEnvironmentIsATypedFailure()
    {
        var root = Path.Combine(Path.GetTempPath(), "shorokoo-env-" + Guid.NewGuid().ToString("N"));
        var bare = Directory.CreateDirectory(Path.Combine(root, "bare")).FullName;
        var old = Directory.CreateDirectory(Path.Combine(root, "old")).FullName;
        File.WriteAllText(Path.Combine(old, "pyvenv.cfg"), "home = /usr/bin\nversion_info = 3.11.4\n");
        var homeless = Directory.CreateDirectory(Path.Combine(root, "homeless")).FullName;
        File.WriteAllText(Path.Combine(homeless, "pyvenv.cfg"), "home = /nonexistent/bin\nversion_info = 3.12.1\n");
        var twin = Directory.CreateDirectory(Path.Combine(root, "twin")).FullName;
        File.Copy(Path.Combine(Torch.Start().Directory, "pyvenv.cfg"), Path.Combine(twin, "pyvenv.cfg"));
        try
        {
            Assert.Equal(PythonEnvironmentFailure.EnvironmentNotFound, Failure(new() { EnvironmentPath = Path.Combine(root, "none") }));
            Assert.Equal(PythonEnvironmentFailure.NotAVirtualEnvironment, Failure(new() { EnvironmentPath = bare }));
            Assert.Equal(PythonEnvironmentFailure.WrongPythonVersion, Failure(new() { EnvironmentPath = old }));
            Assert.Equal(PythonEnvironmentFailure.LibPythonNotFound, Failure(new() { EnvironmentPath = homeless }));
            Assert.Equal(PythonEnvironmentFailure.UvNotFound, Failure(new() { CacheDirectory = root, UvPath = Path.Combine(root, "uv") }));
            Assert.Equal(PythonEnvironmentFailure.EnvironmentConflict,
                Assert.Throws<PythonEnvironmentException>(() => new TorchCpuBackend(new() { EnvironmentPath = twin }).Start()).Failure);
            Assert.True(PythonEnvironmentResolver.LooksLikeNetwork("error: Failed to fetch: `https://pypi.org/simple/torch/`"));
            Assert.False(PythonEnvironmentResolver.LooksLikeNetwork("error: No solution found"));
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void TestTheSharedCudaLibrariesAreTheReleasesTheCudaEnvironmentsPyTorchCarries()
    {
        var windows = CudaLibraryPins.ForPlatform("win-x64");
        var linux = CudaLibraryPins.ForPlatform("linux-x64");
        string Record(CudaLibraryFile file) => $"{file.FileName} {file.Sha256} {file.Size}";

        Assert.Equal(["cudnn", "cublas"], windows.Libraries.Select(library => library.Name));
        Assert.Equal(["cudnn", "cublas"], linux.Libraries.Select(library => library.Name));
        Assert.All(windows.Libraries.Concat(linux.Libraries), library => Assert.Equal(13, library.CudaMajor));
        Assert.Equal(windows.Libraries[0].Version, linux.Libraries[0].Version);
        Assert.Equal(windows.Libraries.SelectMany(library => library.Files).Select(Record).Order(), windows.Bundled.Select(Record).Order());
        Assert.Contains(windows.BundledWheelSha256!, LockedHashes("win-x64", windows.BundledBy!));
        Assert.All(linux.Libraries, library => Assert.Contains(library.WheelSha256, LockedHashes("linux-x64", $"{library.Package}=={library.Version}")));
    }

    private static string[] LockedHashes(string platform, string requirement)
    {
        var lines = PythonEnvironmentLock.ForPlatform("cu13", platform).Requirements.Split('\n');
        var at = Array.FindIndex(lines, line => line.StartsWith(requirement + " ", StringComparison.Ordinal));
        return at < 0 ? [] : [.. lines.Skip(at + 1).Select(line => line.Trim().TrimEnd('\\').Trim())
            .TakeWhile(line => line.StartsWith("--hash=sha256:", StringComparison.Ordinal)).Select(line => line["--hash=sha256:".Length..])];
    }

    [Fact]
    public void TestALockNamesItsCachedEnvironmentByItsContentsAndIndex()
    {
        var cpu = PythonEnvironmentLock.Cpu;
        var moved = new PythonEnvironmentLock("cpu", "3.12", cpu.Requirements, ["--index-url", "https://example.invalid/simple"]);

        Assert.StartsWith("cpu-", cpu.CacheKey);
        Assert.StartsWith("cu13-", PythonEnvironmentLock.Cu13.CacheKey);
        Assert.Contains("torch==2.14.0+cpu", cpu.Requirements);
        Assert.Contains("https://download.pytorch.org/whl/cpu", cpu.IndexArguments);
        Assert.NotEqual(cpu.Hash, moved.Hash);
        Assert.Equal(Path.Combine("/cache", "shorokoo", "python-envs"),
            PythonEnvironmentResolver.CacheRoot(new(), name => name == "XDG_CACHE_HOME" ? "/cache" : null, windows: false));
    }

    [Fact]
    public void TestEveryTorchElementTypeRoundTripsThroughATensorAndASession()
    {
        foreach (var type in EveryTorchType)
        {
            var bytes = Enumerable.Range(0, 3 * ElementSize(type)).Select(i => (byte)(type == ShorokooTensorElementType.Bool ? i % 2 : i * 7 % 64)).ToArray();
            using var session = Torch.CreateSession(Onnx("Identity", (int)type), default, default, DeviceMemorySettings.Default);
            using var host = Torch.CreateTensorFromRawBytes(type, bytes, [3]);
            using var device = Torch.CreateTensorInBackendMemory(type, bytes, [3]);
            var outputs = session.Run(new Dictionary<string, IShorokooTensorValue> { ["x0"] = host }, ["y"], RunSettings.Default);

            Assert.Equal(type, host.ElementType);
            Assert.Equal([3L], host.Shape);
            Assert.Equal(bytes, Torch.CopyTensorToHost(host));
            Assert.Equal(bytes, Torch.CopyTensorToHost(device));
            Assert.Equal(bytes, Torch.CopyTensorToHost(outputs[0]));
            Assert.Equal(type, outputs[0].ElementType);
            outputs[0].Dispose();
        }
    }

    [Fact]
    public void TestTypedTensorsSpansAndUninitializedTensorsAreTheTensorsTheySayTheyAre()
    {
        using var floats = Torch.CreateTensor([1f, 2f, 3f, 4f], [2, 2]);
        using var halves = Torch.CreateTensor([(Float16)1.5f], [1]);
        using var longs = Torch.CreateTensor([long.MaxValue, -1L], [2]);
        using var blank = Torch.CreateUninitializedTensorInBackendMemory(ShorokooTensorElementType.Int32, [2, 3]);
        using var empty = Torch.CreateTensor(Array.Empty<float>(), [0, 4]);
        floats.GetTensorMutableDataAsSpan<float>()[3] = 9f;

        Assert.Equal([1f, 2f, 3f, 9f], floats.GetTensorDataAsSpan<float>().ToArray());
        Assert.Equal(ShorokooTensorElementType.Float16, halves.ElementType);
        Assert.Equal(1.5f, (float)halves.GetTensorDataAsSpan<Float16>()[0]);
        Assert.Equal([long.MaxValue, -1L], longs.GetTensorDataAsSpan<long>().ToArray());
        Assert.Equal(6, blank.GetTensorDataAsSpan<int>().Length);
        Assert.Equal([0L, 4L], empty.Shape);
        Assert.Equal(0, empty.GetTensorDataAsSpan<float>().Length);
        Assert.True(floats.IsHostAccessible);
        Assert.Throws<NotSupportedException>(() => Torch.CreateTensorFromRawBytes(ShorokooTensorElementType.Int4, [0], [2]));
        Assert.Throws<NotSupportedException>(() => Torch.CreateTensorFromRawBytes(ShorokooTensorElementType.String, [0], [1]));
        Assert.Throws<ArgumentException>(() => Torch.CreateTensorFromRawBytes(ShorokooTensorElementType.Float, [0, 0], [1]));
    }

    [Fact]
    public void TestStringTensorsRoundTripAndRunThroughASession()
    {
        string[] words = ["plain", "", "naïve ✓", "quote ' and \\ and \n"];
        using var strings = Torch.CreateStringTensor(words, [2, 2]);
        using var session = Torch.CreateSession(Onnx("Identity", (int)ShorokooTensorElementType.String), default, default, DeviceMemorySettings.Default);
        var outputs = session.Run(new Dictionary<string, IShorokooTensorValue> { ["x0"] = strings }, ["y"], RunSettings.Default);

        Assert.Equal(ShorokooTensorElementType.String, strings.ElementType);
        Assert.Equal([2L, 2L], strings.Shape);
        Assert.Equal(words, strings.GetStringTensorData());
        Assert.Equal(words, outputs[0].GetStringTensorData());
        Assert.Throws<InvalidOperationException>(() => strings.GetTensorDataAsSpan<byte>().Length);
        outputs[0].Dispose();
    }

    [Fact]
    public void TestASequenceTakesItsElementsOverAndHandsOutCopies()
    {
        var first = Torch.CreateTensor([1f, 2f], [2]);
        var second = Torch.CreateTensor([3f], [1]);
        using var sequence = Torch.CreateSequence([first, second]);
        using var element = sequence.GetValue(0);
        element.GetTensorMutableDataAsSpan<float>()[0] = 7f;
        using var again = sequence.GetValue(0);
        using var none = Torch.CreateSequence([]);

        Assert.Equal(ShorokooOnnxValueType.Sequence, sequence.ValueType);
        Assert.Equal(2, sequence.GetValueCount());
        Assert.Equal(ShorokooTensorElementType.Float, sequence.GetSequenceElementType());
        Assert.Equal([1f, 2f], again.GetTensorDataAsSpan<float>().ToArray());
        Assert.Equal([1L], sequence.GetValue(1).Shape);
        Assert.Throws<ObjectDisposedException>(() => first.IsHostAccessible);
        Assert.Throws<ObjectDisposedException>(() => second.GetTensorDataAsSpan<float>().Length);
        Assert.Equal(0, none.GetValueCount());
        Assert.Throws<InvalidOperationException>(() => sequence.GetTensorDataAsSpan<float>().Length);
    }

    [Fact]
    public void TestAModelGraphRunsOnTorchAndAgreesWithOnnxRuntime()
    {
        var (model, input) = SideBySideModel.Concrete();
        var onOrt = SideBySideModel.Floats(new ComputeContext().Execute(model, input.Shared())[0]);
        using var context = new ComputeContext(Torch);
        var compiled = context.Compile(model);

        SideBySideModel.AssertAgree(onOrt, SideBySideModel.Floats(context.Execute(model, input.Shared())[0]));
        SideBySideModel.AssertAgree(onOrt, SideBySideModel.Floats(compiled.Execute(input.Shared())[0]));
        SideBySideModel.AssertAgree(onOrt, SideBySideModel.Floats(compiled.Execute(input)[0]));
        Assert.True(input.IsDisposed);
        Assert.Equal("Shorokoo.PyTorch.Cpu", compiled.Backend.Name);
    }

    [Fact]
    public void TestATorchAuditFailsOnAnOperatorTorchCannotRunAndOnADropoutFedOutOfTrainingThatDraws()
    {
        TensorData[] inputs = [QeeAudit.F32([4L], 1f, 2f, 3f, 4f), TensorData(DType.Bool, [], false)];
        Assert.True(Audited(inputs, static model => model));
        Assert.False(Audited(inputs, static model => Altered(model, dropout => dropout.OpType = "NoSuchOperator")));
        Assert.False(Audited(inputs, static model => Altered(model, dropout =>
        {
            model.Graph.Nodes.Insert(0, new NodeProto { OpType = "Constant", Name = "always_training", Outputs = { "always_training" },
                Attributes = { new AttributeProto { Name = "value", Type = AttributeProto.AttributeType.Tensor,
                    T = new TensorProto { data_type = (int)TensorProto.DataType.Bool, RawData = [1] } } } });
            dropout.Inputs[2] = "always_training";
        })));
    }

    private static bool Audited(TensorData[] inputs, Func<ModelProto, ModelProto> alterOnTorch)
        => QeeAuditOnTorch.Agrees<QeeDropoutFedModeAuditCheck>(QeeAudit.Lower<QeeDropoutFedModeAuditCheck>(inputs, null, null), inputs, alterOnTorch);

    private static ModelProto Altered(ModelProto model, Action<NodeProto> alterDropout)
    {
        alterDropout(model.Graph.Nodes.Single(n => n.OpType == "Dropout"));
        return model;
    }

    [Fact]
    public void TestControlFlowRunsOnTorch()
    {
        var torch = new ComputeContext(Torch);

        Assert.True(AutoTest.AdvancedTestGraph<AutoGradIfRuntimeConditionTrueCheck>([], [QeeAudit.F32([], 2f), QeeAudit.F32([], 3f)], context: torch));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradIfRuntimeConditionFalseCheck>([], [QeeAudit.F32([], -1f), QeeAudit.F32([], 3f)], context: torch));
    }

    [Fact]
    public void TestIntegerAndBoolReduceMaxAndMinOverAnEmptyGroupYieldTheTypeExtremesOnTorch()
        => EmptyIntegerReduceMaxAndMinYieldTheTypeExtremes(new ComputeContext(Torch));

    [Fact]
    public void TestPoolsComputeTheSpecValuesOnTorch() => PoolsComputeTheSpecValues(new ComputeContext(Torch));

    [Fact]
    public void TestMaxPoolIndicesComputeTheSpecValuesOnTorch() => MaxPoolIndicesComputeTheSpecValues(new ComputeContext(Torch));

    [Fact]
    public void TestConvTransposeSameStridedPastTheKernelComputesTheSpecValuesOnTorch()
        => ConvTransposeSameStridedPastTheKernelComputesTheSpecValues(new ComputeContext(Torch));

    [Fact]
    public void TestMaxPoolIndexOfALowestValuedWindowIsItsFirstInputPositionOnTorch()
        => LowestValuedWindowMaxPoolIndexIsItsFirstInputPosition(new ComputeContext(Torch));

    [Fact]
    public void TestMaxPoolIndexOfANegativeInfinityWindowNamesThePaddingBeforeItOnTorch()
        => Assert.True(Spec<NegativeInfinityWindowMaxPoolIndicesValues>(new ComputeContext(Torch),
            QeeAudit.F32([1L, 1L, 5L], float.NegativeInfinity, float.NegativeInfinity, 5f, float.NegativeInfinity, float.NegativeInfinity), -1, 0, 2, 2, 3, 4));

    [Fact]
    public void TestCropAndResizeAndCol2ImComputeTheSpecValuesOnTorch() => CropAndResizeAndCol2ImComputeTheSpecValues(new ComputeContext(Torch));

    [Fact]
    public void TestReductionsOfAnInputEmptyByItsValuesHaveTheSpecShapesAndValuesOnTorch()
    {
        var torch = new ComputeContext(Torch);
        var x = QeeAudit.F32([2L, 3L], 1f, 2f, 3f, 4f, 5f, 6f);
        Assert.True(Spec<EmptyReduceNegativeAxisShapes>(torch, x, 3, 1, 0, 3, 1, 0));
        Assert.True(Spec<EmptyRawReduceNegativeAxisShapes>(torch, x, 3, 1, 0, 3, 1, 0));
        Assert.True(Spec<EmptyNoopReduceShape>(torch, x, 2, 0));
        Assert.True(Spec<EmptyFloat16ReduceAllValues>(torch, x, 0, 0));
    }

    [Fact]
    public void TestANoopSumOrMeanHandsBackEachElementAsItIsANegativeZeroIncludedOnTorchAsOnOnnxRuntime()
    {
        foreach (var backend in (IShorokooBackend[])[DefaultBackend.Instance, Torch])
        {
            Assert.Equal("-0 1 -2", NoopReduced(backend, "ReduceSum"));
            Assert.Equal("-0 1 -2", NoopReduced(backend, "ReduceMean"));
            Assert.Equal("-0 1 -2", NoopReduced(backend, "ReduceMax"));
            Assert.Equal("+0 1 4", NoopReduced(backend, "ReduceSumSquare"));
        }
    }

    /// <summary>What <paramref name="op"/> reducing no axis, <c>noop_with_empty_axes</c> set, makes of
    /// -0, 1 and -2, a zero written with its sign.</summary>
    private static string NoopReduced(IShorokooBackend backend, string op)
    {
        var graph = Graph(["x"], ["y"], Node(op, ["x", "axes"], ["y"], attributes: Int("noop_with_empty_axes", 1)));
        graph.Inputs[0].Type = FloatTensor;
        graph.Outputs[0].Type = FloatTensor;
        graph.Initializers.Add(new TensorProto { Name = "axes", data_type = (int)TensorProto.DataType.Int64, Dims = [0] });
        using var session = backend.CreateSession(Serialize(graph), default, default, DeviceMemorySettings.Default);
        using var x = backend.CreateTensor([-0f, 1f, -2f], [3]);
        using var y = session.Run(new Dictionary<string, IShorokooTensorValue> { ["x"] = x }, ["y"], RunSettings.Default)[0];
        return string.Join(" ", MemoryMarshal.Cast<byte, float>(backend.CopyTensorToHost(y)).ToArray()
            .Select(v => v == 0 ? (float.IsNegative(v) ? "-0" : "+0") : v.ToString(System.Globalization.CultureInfo.InvariantCulture)));
    }

    [Fact]
    public void TestANoopReductionReducesEachElementAloneOnTorch()
    {
        NoopReductionsReduceEachElementAlone(new ComputeContext(Torch));
        Assert.True(AutoTest.AdvancedTestGraph<NoopReduceOfEachElementCheck>([], [QeeAudit.F32([2L, 3L], 1f, 2f, 3f, 4f, 5f, 6f)],
            context: new ComputeContext(Torch)));
    }

    [Fact]
    public void TestWhereSelectsOnEveryIntegerTypeAndBoolOnTorch() => WhereSelectsOnEveryIntegerTypeAndBool(new ComputeContext(Torch));

    [Fact]
    public void TestClipHandsBackAZeroInsideItsBoundsAndTheLowerOfTwoEqualZeroBoundsBelowThemSignAndAllOnTorchAsOnOnnxRuntime()
    {
        ClipKeepsTheSignOfAZero(DefaultBackend.Instance);
        ClipKeepsTheSignOfAZero(Torch);
    }

    internal static void ClipKeepsTheSignOfAZero(IShorokooBackend backend)
    {
        Assert.Equal("+0", Clipped(backend, -1f, 0f, -0f));
        Assert.Equal("-0", Clipped(backend, -1f, -0f, 0f));
        Assert.Equal("-0", Clipped(backend, -0f, 0f, null));
        Assert.Equal("+0", Clipped(backend, 0f, -0f, null));
        Assert.Equal("+0", Clipped(backend, 0f, null, -0f));
        Assert.Equal("-0", Clipped(backend, -0f, null, 0f));
        Assert.Equal("+0", Clipped(backend, -0f, 1f, 0f));
        Assert.Equal("-0", Clipped(backend, 0f, 1f, -0f));
        Assert.Equal("2", Clipped(backend, 3f, 1f, 2f));
        Assert.Equal("NaN", Clipped(backend, float.NaN, 1f, 2f));
    }

    /// <summary>What <c>Clip</c> makes of 67 elements equal to <paramref name="x"/> — enough to fill
    /// a vectorized kernel's loop and leave a tail — as each distinct result, a zero written with
    /// its sign.</summary>
    private static string Clipped(IShorokooBackend backend, float x, float? lo, float? hi)
    {
        string[] bounds = [lo is null ? "" : "lo", hi is null ? "" : "hi"];
        var graph = Graph(["x", .. bounds.Where(b => b.Length > 0)], ["y"], Node("Clip", ["x", .. bounds], ["y"]));
        foreach (var value in graph.Inputs.Concat(graph.Outputs)) value.Type = FloatTensor;
        using var session = backend.CreateSession(Serialize(graph), default, default, DeviceMemorySettings.Default);
        IShorokooTensorValue Tensor(float[] values, long[] shape)
            => backend.CreateTensorInBackendMemory(ShorokooTensorElementType.Float, [.. MemoryMarshal.AsBytes<float>(values)], shape);
        var feeds = new Dictionary<string, IShorokooTensorValue> { ["x"] = Tensor([.. Enumerable.Repeat(x, 67)], [67]) };
        if (lo is { } l) feeds["lo"] = Tensor([l], []);
        if (hi is { } h) feeds["hi"] = Tensor([h], []);
        using var y = session.Run(feeds, ["y"], RunSettings.Default)[0];
        foreach (var feed in feeds.Values) feed.Dispose();
        return string.Join(" ", MemoryMarshal.Cast<byte, float>(backend.CopyTensorToHost(y)).ToArray()
            .Select(v => v == 0 ? (float.IsNegative(v) ? "-0" : "+0") : v.ToString(System.Globalization.CultureInfo.InvariantCulture)).Distinct());
    }

    internal static void NoopReductionsReduceEachElementAlone(ComputeContext c)
    {
        var x = QeeAudit.F32([2L, 3L], 1f, 2f, 3f, 4f, 5f, 6f);
        Assert.True(AutoTest.AdvancedTestGraph<NoopReduceOfEachElementByShapeCheck>([], [x], context: c));
        Assert.True(AutoTest.AdvancedTestGraph<ElementwiseNoopReduceOfAConstantCheck>([], [x], context: c));
        Assert.True(AutoTest.AdvancedTestGraph<NoopLogSumExpOfInfinitiesCheck>([], [x], context: c));
        Assert.True(AutoTest.AdvancedTestGraph<LogSumExpOfInfiniteGroupsCheck>([], [x], context: c));
        Assert.True(AutoTest.AdvancedTestGraph<IntegerReduceL2BeyondFloat32Check>([], [x], context: c));
    }

    internal static void PoolsComputeTheSpecValues(ComputeContext c)
    {
        var ten = QeeAudit.F32([1L, 1L, 10L], [.. Enumerable.Range(0, 10).Select(i => (float)i)]);
        var nine = QeeAudit.F32([1L, 1L, 9L], [.. Enumerable.Range(-4, 9).Select(i => (float)i)]);
        Assert.True(Spec<SameDilatedMaxPoolValues>(c, ten, 3, 5, 7, 9, 9, 2, 4, 6, 8, 8));
        Assert.True(Spec<SameDilatedLpPoolValues>(c, ten, 3.1622777, 5.9160798, 9.1104336, 12.4498996, 11.4017543, 2, 4.4721360, 7.4833148, 10.7703296, 10));
        Assert.True(Spec<SameDilatedAveragePoolValues>(c, ten, 2, 3, 5, 7, 8, 1, 2, 4, 6, 7));
        Assert.True(Spec<SameDilatedPoolOddLengthValues>(c, QeeAudit.F32([1L, 1L, 9L], [.. Enumerable.Range(0, 9).Select(i => (float)i)]),
            2, 4, 6, 8, 8, 2 / 3.0, 2, 4, 6, 14 / 3.0, 2, 3, 4, 5, 6, 7, 8, 7, 8));
        Assert.True(Spec<PadsReachingTheKernelPoolValues>(c, nine, -3, -2, -1, 0, 1, 2, 3, 4, 2, -3, -2, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 2, -1.5, -1,
            -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 1, 3, 2, 4.1231056, 3, 2.236068, 2.236068, 3, 4.1231056, 2, -3, 0, 3));
        Assert.True(Spec<SameWideDilationPoolValues>(c, nine, -2, 0, 2, 4, 3, -3, -2, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 2, -5 / 3.0, -1 / 3.0, 0, 1 / 3.0, 5 / 3.0, 2, 4, 2.8284271, 4, 2));
        Assert.True(Spec<PadsReachingTheKernelMaxPoolElementTypeValues>(c, nine, -3, -2, -1, 0, 1, 2, 3, 4, 2, 1, 2, 3, 4, 5, 6, 7, 8, 6, -3, -2, -1, 0, 1, 2, 3, 4, 2));
        Assert.True(Spec<PadsReachingTheKernelThreeAxisPoolValues>(c, QeeAudit.F32([1L, 2L, 3L, 4L, 3L], [.. Enumerable.Range(0, 72).Select(i => (float)(i * 7 % 11 - 5))]),
            4, 4, 2, -2, 4, 4, 0, -4, 2, 2, 5, 5, 0, 0, 3, 3, 4, 5, 5, 1, 4, 4, 3, -1, 5, 5, 1, -3, 3, 3, -1, -5, 1, 1, 4, 4,
            -1, -1, 2, 2, 5, 5, 4, 0, 3, 3, 2, -2, 0.125, 0.125, 0.25, 0.25, 0, -0.25, 0, -0.125, 0.125, 0.125, 0.25, 0.25, 0, -0.25,
            0, -0.125, 0, -0.125, 0, -0.25, -0.25, 0.625, -0.125, -0.375, 0, -0.125, 0, -0.25, -0.25, 0.625, -0.125, -0.375));
        Assert.True(Spec<CeilModePadsReachingTheKernelPoolValues>(c, nine, -3.5, -0.5, 2.5, 0, 5, 1, 3.6055513, 0, -3, -1, 1, 3, 3, -4, -1, 2, 3, -3.5, -2, 0, 2, 3.5));
        Assert.True(Spec<NegativeSamePaddingPoolValues>(c, QeeAudit.F32([1L, 1L, 8L], 5f, -3f, 8f, 1f, -7f, 2f, 6f, -4f), 1, 2, 8, 6, 2.5, 1, 1, 4, 3, 7, 4, 8, 6, -3, -7, -4));
    }

    internal static void MaxPoolIndicesComputeTheSpecValues(ComputeContext c)
    {
        Assert.True(Spec<SameDilatedMaxPoolWithIndicesValues>(c, QeeAudit.F32([1L, 1L, 5L, 6L], [.. Enumerable.Range(0, 30).Select(i => (float)(i * 7 % 30))]),
            7, 21, 29, 25, 25, 29, 25, 25, 29, 1, 3, 17, 25, 25, 17, 25, 25, 17));
        Assert.True(Spec<PadsReachingTheKernelMaxPoolWithIndicesValues>(c, QeeAudit.F32([1L, 2L, 4L, 4L], [.. Enumerable.Range(0, 32).Select(i => (float)(i * 7 % 32 - 16))]),
            12, -6, 4, -2, 10, 10, 14, 14, 4, 6, 12, 2, 22, 22, 18, 18, 12, -6, 4, -2, 10, 10, 14, 14, 1, 9, 3, 8, 25, 25, 24, 24,
            -10, 15, -6, -13, 6, -1, 10, 3, 10, 9, 6, 5, 26, 25, 22, 21));
        Assert.True(AutoTest.AdvancedTestGraph<PaddedMaxPoolIndicesPointAtTheirValues>([], [QeeAudit.F32([1L, 1L, 5L], 0f, 0f, 5f, 0f, 0f)], context: c));
        Assert.True(Spec<NegativeSamePaddingMaxPoolIndicesValues>(c, QeeAudit.F32([1L, 2L, 5L, 6L], [.. Enumerable.Range(0, 60).Select(i => (float)(i * 7 % 11 - 5))]),
            4, -1, 5, 4, 5, 0, -1, 5, 6, 10, 25, 28, 36, 40, 54, 58, 2, 1, 5, 4, 3, 2, -3, 5, 1, 4, 25, 28, 31, 34, 49, 58));
    }

    internal static void ConvTransposeSameStridedPastTheKernelComputesTheSpecValues(ComputeContext c)
        => Assert.True(Spec<ConvTransposeSameStridedPastTheKernelValues>(c, QeeAudit.F32([1L, 1L, 3L], 1f, 2f, 3f),
            QeePoolConvAuditTests.ConvTransposeSameStridedPastTheKernel));

    internal static void LowestValuedWindowMaxPoolIndexIsItsFirstInputPosition(ComputeContext c)
        => Assert.True(Spec<LowestValueWindowMaxPoolIndicesValues>(c, QeeAudit.F32([1L, 1L, 5L], 0f, 0f, 5f, 0f, 0f), 0, 2, 2, 3, 0, 2, 2, 3, 0, 1, 2, 1, 2, 3, 4));

    internal static void NegativeInfinityWindowMaxPoolIndexIsItsFirstInputPosition(ComputeContext c)
        => Assert.True(Spec<NegativeInfinityWindowMaxPoolIndicesValues>(c,
            QeeAudit.F32([1L, 1L, 5L], float.NegativeInfinity, float.NegativeInfinity, 5f, float.NegativeInfinity, float.NegativeInfinity), 0, 0, 2, 2, 3, 4));

    internal static void CropAndResizeAndCol2ImComputeTheSpecValues(ComputeContext c)
    {
        var five = QeeAudit.F32([1L, 1L, 1L, 5L], 0f, 1f, 2f, 3f, 4f);
        Assert.True(Spec<CropAndResizeAtScaleOneValues>(c, five, 2, 3, 4, -1, -1, 2, 3, 4, -1, -1));
        Assert.True(Spec<CropAndResizeVariantsValues>(c, five, 2, 3, 4, -1, -1, 2, 3, 4, -1, -1, -1, 0, 1, 2, 3, 2, 2.8, 3.6, -1, -1, -1,
            0, 5.5, 11, 16.5, 22, -1, -1, -1, -1, -1, 2, 3, 4, -1, -1,
            0, 0, 1, 1, 2, 2, 3, 3, 4, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
        Assert.True(Spec<CropAndResizeToTheRoiEndValues>(c, QeeAudit.F32([1L, 1L, 1L, 4L], 1f, 2f, 3f, 4f),
            2.5, 3, 3.5, 4, 2, 3, 3, 4, 2.5, 3, 3.59375, 4, 1.66796875, 2.5, 3.33203125, 4));
        Assert.True(Spec<CubicCropAndResizeAlongChannelsValues>(c, QeeAudit.F32([1L, 3L, 1L, 2L], 0f, 1f, 2f, 3f, 4f, 5f),
            2, 3, 2.992, 3.992, 3.696, 4.696, -1, -1, -1, -1, -1, -1, 2, 3, 4, 5, -1, -1));
        Assert.True(Spec<Col2Im1DPaddedValues>(c, QeeAudit.F32([1L, 3L, 4L], [.. Enumerable.Range(0, 12).Select(i => (float)i)]), 4, 9, 5, 11, 6, 13, 7, 11));
        Assert.True(Spec<Col2Im1DUnpaddedValues>(c, QeeAudit.F32([1L, 4L, 2L], [.. Enumerable.Range(1, 8).Select(i => (float)i)]),
            1, 5, 4, 5, 13, 8, 1, 5, 2, 6, 3, 7, 4, 8, 1, 7, 9, 11, 8, 1, 2, 8, 10, 7, 8));
    }

    internal static void EmptyIntegerReduceMaxAndMinYieldTheTypeExtremes(ComputeContext c)
    {
        Assert.True(AutoTest.AdvancedTestGraph<EmptyIntegerReduceMaxMinCheck>([], [QeeAudit.F32([2L, 3L], 1f, 2f, 3f, 4f, 5f, 6f)], context: c));
        Assert.True(GenericMaxMin(c, DType.Int64, QeeAudit.I64([1L, 0L]), long.MinValue, long.MaxValue));
        Assert.True(GenericMaxMin(c, DType.Int8, QeeAudit.I8([1L, 0L]), sbyte.MinValue, sbyte.MaxValue));
        Assert.True(GenericMaxMin(c, DType.UInt8, QeeAudit.U8([1L, 0L]), byte.MinValue, byte.MaxValue));
        Assert.True(GenericMaxMin(c, DType.Bool, QeeAudit.Bits([1L, 0L]), 0L, 1L));
        Assert.True(GenericMaxMin(c, DType.Bool, QeeAudit.Bits([2L, 2L], true, false, false, false), 1L, 0L, 0L, 0L));
    }

    internal static void WhereSelectsOnEveryIntegerTypeAndBool(ComputeContext c)
    {
        Assert.True(AutoTest.AdvancedTestGraph<WhereOnEveryIntegerAndBoolTypeCheck>([], [QeeAudit.Bits([2L], true, false)], context: c));
        Assert.True(AutoTest.AdvancedTestGraph<WhereBroadcastsOnEveryTypeWithoutAKernelCheck>([], [QeeAudit.Bits([2L], true, false)], context: c));
        Assert.True(AutoTest.AdvancedTestGraph<WhereOnUInt64BeyondInt64Check>([], [QeeAudit.Bits([4L], true, false, true, false)], context: c));
        Assert.True(AutoTest.AdvancedTestGraph<WhereKeepsTheSignOfZeroCheck>([], [QeeAudit.Bits([2L], true, false)], context: c));
    }

    private static bool Spec<TModule>(ComputeContext c, TensorData x, params double[] expected)
        => AutoTest.AdvancedTestGraph<TModule>([], [x], context: c, expected: expected);

    private static bool GenericMaxMin(ComputeContext c, DType t, TensorData x, params long[] expected)
        => AutoTest.AdvancedTestGraph<GenericReduceMaxMinCheck>([], [x, QeeAudit.I64([expected.Length], expected)], context: c,
            genericTypes: new() { ["T"] = t });

    [Fact]
    public void TestAnInt64PadKeepsAConstantValueADoubleCannotHold()
        => Assert.True(AutoTest.AdvancedTestGraph<PadInt64WithWideValuesCheck>([], [QeeAudit.I64([2L], 1L, 2L)],
            context: new ComputeContext(Torch)));

    [Fact]
    public void TestAConstantPadFillsWithItsValueExactlyOnEveryTypeOnTorch() => ConstantPadsFillWithTheirValueExactly(Torch);

    [Fact]
    public void TestAnInt64RangeCountsItsElementsExactlyOnTorch() => QeeImageRandomRnnAuditTests.Int64RangesCountTheirElementsExactly(new ComputeContext(Torch));

    [Fact]
    public void TestAnInt64RangeWithAUnitStepCountsItsElementsExactlyOnTorch() => QeeImageRandomRnnAuditTests.Int64UnitStepRangesCountTheirElementsExactly(new ComputeContext(Torch));

    [Fact]
    public void TestAnInt32RangeCountsItsElementsExactlyOnTorch() => QeeImageRandomRnnAuditTests.Int32RangesCountTheirElementsExactly(new ComputeContext(Torch));

    [Fact]
    public void TestAUnitStepRangeWhoseSpanWrapsItsTypeIsEmptyOnTorch() => QeeImageRandomRnnAuditTests.UnitStepRangesWhoseSpanWrapsTheirTypeAreEmpty(new ComputeContext(Torch));

    internal static void ConstantPadsFillWithTheirValueExactly(IShorokooBackend backend)
    {
        Assert.True(PadsWith(backend, ShorokooTensorElementType.Int64, Raw(long.MaxValue)));
        Assert.True(PadsWith(backend, ShorokooTensorElementType.Int64, Raw(9007199254740993L)));
        Assert.True(PadsWith(backend, ShorokooTensorElementType.UInt64, Raw(ulong.MaxValue)));
        Assert.True(PadsWith(backend, ShorokooTensorElementType.UInt64, Raw((1UL << 63) + 1UL)));
        Assert.True(PadsWith(backend, ShorokooTensorElementType.Int32, Raw(int.MinValue)));
        Assert.True(PadsWith(backend, ShorokooTensorElementType.UInt32, Raw(uint.MaxValue)));
        Assert.True(PadsWith(backend, ShorokooTensorElementType.Int8, Raw(sbyte.MinValue)));
        Assert.True(PadsWith(backend, ShorokooTensorElementType.Bool, Raw(true)));
        Assert.True(PadsWith(backend, ShorokooTensorElementType.Float, Raw(-0f, float.NaN)));
        Assert.True(PadsWith(backend, ShorokooTensorElementType.Double, Raw(double.NegativeInfinity)));
        Assert.True(PadsWith(backend, ShorokooTensorElementType.Complex64, Raw(1.5f, -2f)));
    }

    private static byte[] Raw<T>(params T[] values) where T : unmanaged => MemoryMarshal.AsBytes(values.AsSpan()).ToArray();

    private static bool PadsWith(IShorokooBackend backend, ShorokooTensorElementType type, byte[] values)
    {
        var size = ElementSize(type);
        return Enumerable.Range(0, values.Length / size).All(i => PadFills(backend, type, values[(i * size)..((i + 1) * size)]));
    }

    private static bool PadFills(IShorokooBackend backend, ShorokooTensorElementType type, byte[] value)
    {
        var fill = new AttributeProto { Name = "value", Type = AttributeProto.AttributeType.Tensor, T = new TensorProto { data_type = (int)type, RawData = value } };
        var graph = Graph(["x0"], ["y"],
            Node("Constant", [], ["pads"], attributes: Tensor("value", 7, [2], [1, 1])),
            Node("Constant", [], ["fill"], attributes: fill),
            Node("Pad", ["x0", "pads", "fill"], ["y"]));
        using var session = backend.CreateSession(Serialize(graph), default, default, DeviceMemorySettings.Default);
        using var x = backend.CreateTensorFromRawBytes(type, new byte[value.Length], [1]);
        using var y = session.Run(Feeds(x), ["y"], RunSettings.Default)[0];
        return backend.CopyTensorToHost(y).SequenceEqual([.. value, .. new byte[value.Length], .. value]);
    }

    [Fact]
    public void TestBatchFirstRecurrentNetworksRunOnTorch()
    {
        Assert.True(AutoTest.AdvancedTestGraph<QeeRecurrentBatchFirstValueCheck>([],
            [QeeAudit.Wave(4, 2, 3), QeeAudit.Wave(2, 20, 3), QeeAudit.Wave(2, 20, 5), QeeAudit.Wave(2, 40), QeeAudit.Wave(2, 2, 5), QeeAudit.Wave(2, 2, 5), QeeAudit.Wave(2, 15), QeeAudit.I32([2L], 4, 2)],
            context: new ComputeContext(Torch)));
    }

    [Fact]
    public void TestAFunctionCalledInsideABranchRunsAsAPythonFunction()
    {
        var twice = new FunctionProto { Name = "Twice", Domain = "Functions" };
        twice.Inputs.Add("a");
        twice.Outputs.Add("b");
        twice.Nodes.Add(Node("Add", ["a", "a"], ["b"]));
        twice.OpsetImports.Add(new OperatorSetIdProto { Domain = "", Version = 21 });
        var thenBranch = Graph([], ["t"], Node("Twice", ["x"], ["t"], domain: "Functions"));
        var elseBranch = Graph([], ["e"], Node("Neg", ["x"], ["e"]));
        var branch = Node("If", ["c"], ["y"]);
        branch.Attributes.Add(new AttributeProto { Name = "then_branch", Type = AttributeProto.AttributeType.Graph, G = thenBranch });
        branch.Attributes.Add(new AttributeProto { Name = "else_branch", Type = AttributeProto.AttributeType.Graph, G = elseBranch });
        using var session = Torch.CreateSession(Serialize(Graph(["c", "x"], ["y"], branch), twice), default, default, DeviceMemorySettings.Default);

        Assert.Equal([2f, 4f], RunFloats(session, true, [1f, 2f]));
        Assert.Equal([-1f, -2f], RunFloats(session, false, [1f, 2f]));
    }

    [Fact]
    public void TestFunctionAttributesSequenceMapTensorScatterAndOmittedOutputsAgreeWithOnnxRuntime()
    {
        var scaled = new FunctionProto { Name = "Scaled", Domain = "Functions" };
        scaled.Inputs.Add("a");
        scaled.Outputs.Add("b");
        scaled.Attributes.Add("axis");
        scaled.AttributeProtoes.Add(new AttributeProto { Name = "alpha", Type = AttributeProto.AttributeType.Float, F = 2f });
        scaled.Nodes.AddRange([
            Node("Constant", [], ["c"], attributes: new AttributeProto { Name = "value_float", Type = AttributeProto.AttributeType.Float, RefAttrName = "alpha" }),
            Node("Mul", ["a", "c"], ["m"]),
            Node("Softmax", ["m"], ["b"], attributes: new AttributeProto { Name = "axis", Type = AttributeProto.AttributeType.Int, RefAttrName = "axis" })]);
        scaled.OpsetImports.Add(new OperatorSetIdProto { Domain = "", Version = 21 });
        var alpha3 = new AttributeProto { Name = "alpha", Type = AttributeProto.AttributeType.Float, F = 3f };
        var axis0 = new AttributeProto { Name = "axis", Type = AttributeProto.AttributeType.Int, I = 0 };

        var body = Graph(["e", "w"], ["p", "q"], Node("Add", ["e", "w"], ["p"]), Node("ReduceSum", ["e"], ["q"], attributes: Int("keepdims", 0)));
        foreach (var value in body.Inputs.Concat(body.Outputs)) value.Type = FloatTensor;

        Assert.True(AgreesWithOrt(Typed(21, ["x"], ["y1", "y2", "y3"],
            [Node("Scaled", ["x"], ["y1"], "Functions", alpha3, axis0), Node("Scaled", ["x"], ["y2"], "Functions"),
             Node("Scaled", ["x"], ["y3"], "Functions", axis0, alpha3)], scaled), X23));
        Assert.True(AgreesWithOrt(Typed(21, ["x", "w"], ["o1", "o2"],
            [Node("SplitToSequence", ["x"], ["s"]),
             Node("SequenceMap", ["s", "w"], ["ps", "qs"], attributes: new AttributeProto { Name = "body", Type = AttributeProto.AttributeType.Graph, G = body }),
             Node("ConcatFromSequence", ["ps"], ["o1"], attributes: Int("axis", 0)),
             Node("ConcatFromSequence", ["qs"], ["o2"], attributes: [Int("axis", 0), Int("new_axis", 1)])]),
            X23, ("w", [10f, 20f, 30f], [3])));
        foreach (var (mode, starts) in (ReadOnlySpan<(string, long[])>)[("linear", [1, 2]), ("circular", [3, 0])])
            Assert.True(AgreesWithOrt(Typed(24, ["past", "update"], ["present"],
                [Node("Constant", [], ["at"], attributes: Tensor("value", 7, [2], starts)),
                 Node("TensorScatter", ["past", "update", "at"], ["present"], attributes: [Str("mode", mode), Int("axis", 1)])]),
                ("past", new float[8], [2, 4, 1]), ("update", [1f, 2f, 3f, 4f], [2, 2, 1])));
        Assert.True(AgreesWithOrt(Typed(21, ["x"], ["u", "inv", "d"],
            [Node("Unique", ["x"], ["u", "", "inv"]),
             Node("Dropout", ["x"], ["d"])]), X23));
    }

    [Fact]
    public void TestAFunctionCallsShorokooAnnotationsAreNotArgumentsOfTheFunction()
    {
        var twice = new FunctionProto { Name = "Twice", Domain = "Functions" };
        twice.Inputs.Add("a");
        twice.Outputs.Add("b");
        twice.Nodes.Add(Node("Add", ["a", "a"], ["b"]));
        twice.OpsetImports.Add(new OperatorSetIdProto { Domain = "", Version = 21 });
        var call = Node("Twice", ["x"], ["y"], domain: "Functions");
        call.Attributes.Add(new AttributeProto { Name = "shrk_structure", Type = AttributeProto.AttributeType.Strings, Strings = { "Tensor"u8.ToArray() } });
        call.Attributes.Add(new AttributeProto { Name = "shrk_dtype", Type = AttributeProto.AttributeType.Ints, Ints = [1] });
        var bogus = Node("Twice", ["x"], ["y"], domain: "Functions");
        bogus.Attributes.Add(new AttributeProto { Name = "scale", Type = AttributeProto.AttributeType.Int, I = 2 });
        using var session = Torch.CreateSession(Serialize(Graph(["c", "x"], ["y"], call), twice), default, default, DeviceMemorySettings.Default);

        Assert.Equal([2f, 4f], RunFloats(session, true, [1f, 2f]));
        Assert.Throws<TorchUnsupportedModelException>(() => Torch.CreateSession(Serialize(Graph(["c", "x"], ["y"], bogus), twice), default, default, DeviceMemorySettings.Default));
    }

    [Fact]
    public void TestAnOutputIsMemoryOfItsOwnEvenWhereTheGraphReturnsItsInput()
    {
        using var session = Torch.CreateSession(Onnx("Identity", (int)ShorokooTensorElementType.Float), default, default, DeviceMemorySettings.Default);
        using var input = Torch.CreateTensor([1f, 2f], [2]);
        var output = session.Run(new Dictionary<string, IShorokooTensorValue> { ["x0"] = input }, ["y"], RunSettings.Default)[0];
        output.GetTensorMutableDataAsSpan<float>()[0] = 5f;

        Assert.Equal([1f, 2f], input.GetTensorDataAsSpan<float>().ToArray());
        Assert.Equal([5f, 2f], output.GetTensorDataAsSpan<float>().ToArray());
        output.Dispose();
    }

    [Fact]
    public void TestARunOnACardSetsTorchsTensorFloat32SwitchesFromItsSessionAndARunOnTheCpuSetsNeither()
    {
        var allowing = SideBySideModel.AllowingTensorFloat32;
        using var session = Torch.CreateSession(Onnx("Neg", (int)ShorokooTensorElementType.Float), default, default, DeviceMemorySettings.Default);
        using var x = Torch.CreateTensor([1f], [1]);
        const string Switches = "(torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)";
        const string Set = "__import__('shorokoo_torch.runtime', fromlist=['runtime']).float32_precision";

        Assert.Equal((false, false, true), (TorchSession.TensorFloat32(Torch, allowing), TorchSession.TensorFloat32(new TorchCudaBackend(), PrecisionSettings.Default),
            TorchSession.TensorFloat32(new TorchCudaBackend(), allowing)));
        Assert.Equal("(True, True)", Evaluated($"({Set}(True), {Switches})[1]"));
        session.Run(new Dictionary<string, IShorokooTensorValue> { ["x0"] = x }, ["y"], RunSettings.Default)[0].Dispose();
        Assert.Equal("(True, True)", Evaluated(Switches));
        Assert.Equal("(False, False)", Evaluated($"({Set}(False), {Switches})[1]"));
    }

    [Fact]
    public void TestRunsInOnePrecisionRunBesideOneAnotherAndRunsInTheOtherWaitTheirTurn()
    {
        Assert.True(Overlap(true, true));
        Assert.True(Overlap(false, false));
        Assert.False(Overlap(true, false));
        Assert.False(Overlap(false, true));
        Assert.False(JoinsPastAWaiter(true));
        Assert.False(JoinsPastAWaiter(false));
    }

    /// <summary>Whether a run in <paramref name="second"/>'s precision enters while one in
    /// <paramref name="first"/>'s holds the gate.</summary>
    private static bool Overlap(bool first, bool second)
    {
        var gate = new TorchPrecisionGate();
        gate.Enter(first);
        try { return Task.Run(() => EntersAndLeaves(gate, second)).Result; }
        finally { gate.Exit(first); }
    }

    /// <summary>Whether a run in <paramref name="held"/>'s precision joins one holding the gate
    /// while a run in the other waits, which enters once the holder leaves.</summary>
    private static bool JoinsPastAWaiter(bool held)
    {
        var gate = new TorchPrecisionGate();
        gate.Enter(held);
        var waiter = Task.Run(() => EntersAndLeaves(gate, !held, wait: true));
        SpinWait.SpinUntil(() => gate.Waiting(!held) == 1);
        var joined = Task.Run(() => EntersAndLeaves(gate, held)).Result;
        gate.Exit(held);
        Assert.True(waiter.Result);
        return joined;
    }

    private static bool EntersAndLeaves(TorchPrecisionGate gate, bool tensorFloat32, bool wait = false)
    {
        if (wait) gate.Enter(tensorFloat32);
        else if (!gate.TryEnter(tensorFloat32)) return false;
        gate.Exit(tensorFloat32);
        return true;
    }

    [Fact]
    public void TestAFailedImportIsACudaLibraryConflictWhereTheProcessHoldsAnotherReleaseAndAMissingPackageWhereNothingImports()
    {
        Assert.Equal(PythonEnvironmentFailure.CudaLibraryConflict, TorchRuntime.ImportFailure("ImportError", () => "cudnn64_9.dll", out _));
        Assert.Equal(PythonEnvironmentFailure.CudaLibraryConflict, TorchRuntime.ImportFailure("OSError", () => "cudnn64_9.dll", out _));
        Assert.Equal(PythonEnvironmentFailure.MissingPackage, TorchRuntime.ImportFailure("ModuleNotFoundError", () => "cudnn64_9.dll", out _));
        Assert.Equal(PythonEnvironmentFailure.MissingPackage, TorchRuntime.ImportFailure("ImportError", () => null, out _));
        Assert.Null(TorchRuntime.ImportFailure("OSError", () => null, out _));
    }

    [Fact]
    public void TestEveryConsumedFeedIsReleasedExactlyOnceHoweverTheRunEnds()
    {
        using var session = Torch.CreateSession(Onnx("Neg", (int)ShorokooTensorElementType.Float), default, default, DeviceMemorySettings.Default);
        using var cancelled = new CancellationTokenSource();
        cancelled.Cancel();

        Assert.Equal(1, ReleasesOf(feed => session.RunConsuming(Feeds(feed), [feed], ["y"], RunSettings.Default)));
        Assert.Equal(1, ReleasesOf(feed => session.RunConsuming(Feeds(feed), [feed], ["nope"], RunSettings.Default)));
        Assert.Equal(1, ReleasesOf(feed => session.RunConsuming(new Dictionary<string, IShorokooTensorValue>(), [feed], ["y"], RunSettings.Default)));
        Assert.Equal(1, ReleasesOf(feed => session.RunConsuming(Feeds(feed), [feed], ["y"], new RunSettings { CancellationToken = cancelled.Token })));
        Assert.Equal(1, ReleasesOf(feed => session.RunConsuming(Feeds(feed), [feed], ["y"], RunSettings.Default, out _)));
    }

    [Fact]
    public void TestAModelTheBackendCannotRunIsRefusedAtSessionCreationNamingTheOperator()
    {
        var unknown = Assert.Throws<TorchUnsupportedModelException>(() => Torch.CreateSession(Onnx("NoSuchOperator", 1), default, default, DeviceMemorySettings.Default));
        var foreign = Assert.Throws<TorchUnsupportedModelException>(() => Torch.CreateSession(Onnx("Relu", 1, domain: "com.example"), default, default, DeviceMemorySettings.Default));
        var attribute = Assert.Throws<TorchUnsupportedModelException>(() => Torch.CreateSession(Onnx("Relu", 1, attribute: new AttributeProto { Name = "bogus", Type = AttributeProto.AttributeType.Int, I = 1 }), default, default, DeviceMemorySettings.Default));

        Assert.Equal(("NoSuchOperator", TorchUnsupportedReason.UnknownOperator), (unknown.Operator, unknown.Reason));
        Assert.Equal(("com.example", TorchUnsupportedReason.UnknownOperator), (foreign.Domain, foreign.Reason));
        Assert.Equal(("Relu", TorchUnsupportedReason.UnsupportedUsage), (attribute.Operator, attribute.Reason));
        Assert.Contains("NoSuchOperator", unknown.Message);
        Assert.Contains("bogus", attribute.Message);
    }

    [Fact]
    public void TestAnAutoGradNodeIsTorchAutogradsGradientOfItsLoss()
    {
        float[] w = [0.5f, -1f, 2f], b = [0.1f, 0.2f, -0.3f], x = [1f, 2f, -0.5f], c = [0.3f, -0.2f, 0.1f];
        var step = Graph(["w", "b", "x", "c", "u"], ["w2", "gb", "gp", "gu", "loss"],
            Node("Exp", ["c"], ["p"]),
            Node("Mul", ["w", "x"], ["m"]),
            Node("Add", ["m", "b"], ["s"]),
            Node("Tanh", ["s"], ["t"]),
            Node("Mul", ["t", "p"], ["tp"]),
            Node("ReduceSum", ["tp"], ["loss"]),
            AutoGrad(["loss", "w", "b", "p", "u"], ["gw", "gb", "gp", "gu"]),
            Node("Sub", ["w", "gw"], ["w2"]));
        using var session = Torch.CreateSession(TrainingStep(step), ShorokooGraphOptimization.TrainingStep, default, DeviceMemorySettings.Default);
        var outputs = RunFloats(session, new() { ["w"] = w, ["b"] = b, ["x"] = x, ["c"] = c, ["u"] = [4f] }, ["w2", "gb", "gp", "gu", "loss"]);

        var t = w.Select((wi, i) => MathF.Tanh(wi * x[i] + b[i])).ToArray();
        var gb = t.Select((ti, i) => MathF.Exp(c[i]) * (1 - ti * ti)).ToArray();
        AssertNear(w.Select((wi, i) => wi - gb[i] * x[i]).ToArray(), outputs[0]);
        AssertNear(gb, outputs[1]);
        AssertNear(t, outputs[2]);
        Assert.Equal([0f], outputs[3]);
        AssertNear([t.Select((ti, i) => ti * MathF.Exp(c[i])).Sum()], outputs[4]);
        Assert.True(((IShorokooBackend)Torch).AcceptsTrainingFormat(TrainingFormats.OnnxAutoGrad));
        Assert.True(((IShorokooBackend)Torch).AcceptsTrainingFormat(TrainingFormats.Onnx));
        Assert.False(((IShorokooBackend)Torch).AcceptsTrainingFormat("onnx-autograd/2"));
    }

    [Fact]
    public void TestAnAutoGradNodeGivesZerosWhereTheLossIsConstantAndGradientsThroughExponentialActivationsAndAPadValue()
    {
        var padValue = Graph(["x", "pads", "w"], ["g"], Node("Pad", ["x", "pads", "w"], ["p"]), Node("Mul", ["p", "p"], ["q"]), Node("ReduceSum", ["q"], ["loss"]), AutoGrad(["loss", "w"], ["g"]));
        var constant = Graph(["w", "x"], ["g"], Node("ReduceSum", ["x"], ["loss"]), AutoGrad(["loss", "w"], ["g"]));
        var activations = Graph(["w"], ["g"],
            Node("Softplus", ["w"], ["a"]), Node("Elu", ["w"], ["e"]), Node("Selu", ["w"], ["s"]), Node("Celu", ["w"], ["c"]),
            Node("Sum", ["a", "e", "s", "c"], ["y"]), Node("ReduceSum", ["y"], ["loss"]), AutoGrad(["loss", "w"], ["g"]));
        using var constantSession = Torch.CreateSession(TrainingStep(constant), default, default, DeviceMemorySettings.Default);
        using var activationSession = Torch.CreateSession(TrainingStep(activations), default, default, DeviceMemorySettings.Default);
        using var padValueSession = Torch.CreateSession(TrainingStep(padValue), default, default, DeviceMemorySettings.Default);
        using var x = Torch.CreateTensor([1f, 2f], [2]);
        using var pads = Torch.CreateTensor([1L, 2L], [2]);
        using var w = Torch.CreateTensor([3f], []);
        using var padGradient = padValueSession.Run(new Dictionary<string, IShorokooTensorValue> { ["x"] = x, ["pads"] = pads, ["w"] = w }, ["g"], RunSettings.Default)[0];

        Assert.Equal([0f, 0f], RunFloats(constantSession, new() { ["w"] = [1f, 2f], ["x"] = [3f, 4f] }, ["g"])[0]);
        AssertNear([4.0507010f, 0f, 3.7817596f], RunFloats(activationSession, new() { ["w"] = [100f, -100f, 1f] }, ["g"])[0]);
        Assert.Equal([18f], padGradient.GetTensorDataAsSpan<float>().ToArray());
    }

    [Fact]
    public void TestAnAutoGradNodeTheBackendCannotDifferentiateIsRefusedAtSessionCreation()
    {
        var tanhOnPath = Graph(["w"], ["g"], Node("Tanh", ["w"], ["t"]), Node("ReduceSum", ["t"], ["loss"]), AutoGrad(["loss", "w"], ["g"]));
        var tanhOffPath = Graph(["w", "x"], ["g"], Node("Tanh", ["x"], ["t"]), Node("Mul", ["t", "w"], ["y"]), Node("ReduceSum", ["y"], ["loss"]), AutoGrad(["loss", "w"], ["g"]));
        var tanhPastAShape = Graph(["w"], ["g"], Node("Shape", ["w"], ["n"]), Cast("n", "f"), Node("Tanh", ["f"], ["t"]), Node("Mul", ["t", "w"], ["y"]), Node("ReduceSum", ["y"], ["loss"]), AutoGrad(["loss", "w"], ["g"]));
        var tanhInAFunction = Graph(["w"], ["g"], Node("Squash", ["w"], ["t"], domain: "Functions"), Node("ReduceSum", ["t"], ["loss"]), AutoGrad(["loss", "w"], ["g"]));
        var tanhInABranch = Graph(["c", "w"], ["g"], Branch("c", Graph([], ["t"], Node("Tanh", ["w"], ["t"])), Graph([], ["t"], Node("Neg", ["w"], ["t"]))), Node("ReduceSum", ["t"], ["loss"]), AutoGrad(["loss", "w"], ["g"]));
        var inABranch = Graph(["c", "w"], ["g"], Branch("c", Graph([], ["g"], Node("ReduceSum", ["w"], ["loss"]), AutoGrad(["loss", "w"], ["g"])), Graph([], ["g"], Node("Neg", ["w"], ["g"]))));
        var twice = Graph(["w"], ["g", "h"], Node("ReduceSum", ["w"], ["loss"]), AutoGrad(["loss", "w"], ["g"]), AutoGrad(["loss", "w"], ["h"]));
        var integer = Graph(["w"], ["g"], Node("ReduceSum", ["w"], ["loss"]), AutoGrad(["loss", "w"], ["g"]));
        integer.Inputs[0].Type = new TypeProto { TensorType = new TypeProto.Tensor { ElemType = (int)ShorokooTensorElementType.Int64 } };
        using var refused = OperatorTableGradient("Tanh", "Refused");

        Assert.Equal("Tanh", Refusal(tanhOnPath).Operator);
        Assert.Equal("Tanh", Refusal(tanhInAFunction).Operator);
        Assert.Equal("Tanh", Refusal(tanhInABranch).Operator);
        Torch.CreateSession(TrainingStep(tanhOffPath), default, default, DeviceMemorySettings.Default).Dispose();
        Torch.CreateSession(TrainingStep(tanhPastAShape), default, default, DeviceMemorySettings.Default).Dispose();
        Assert.Equal("AutoGrad", Refusal(inABranch).Operator);
        Assert.Equal("AutoGrad", Refusal(twice).Operator);
        Assert.Equal("AutoGrad", Refusal(integer).Operator);
        Assert.Equal("AutoGrad", Refusal(tanhOffPath, version: 2).Operator);
    }

    [Fact]
    public void TestGradientsAtTiesAndBoundsAreShorokoosOwnRules()
    {
        AssertNear([1f, 1f, 1f, 0f, 0f], Gradient("lambda x: E.clip(x, min=0.0, max=6.0)", "[0., 6., 3., -1., 7.]"));
        AssertNear([1f, 1f, 1f, 0f, 0f], Gradient("E.clip", "[0., 6., 3.]", "0.", "6."));
        AssertNear([1f / 3, 1f, 1f / 3, 0f, 1f / 3, 0f], Gradient("E.max_", "[1., 2.]", "[1., 0.]", "[1., 1.]"));
        AssertNear([1f / 3, 0f, 1f / 3, 1f, 1f / 3, 0f], Gradient("E.min_", "[1., 2.]", "[1., 0.]", "[1., 5.]"));
        AssertNear([0.1f, 1f, 0.1f], Gradient("lambda x: E.leaky_relu(x, alpha=0.1)", "[0., 2., -2.]"));
        AssertNear([0.25f, 0.25f, 1f, 0f, -2f, 0f], Gradient("E.prelu", "[0., -2., 3.]", "[0.25, 0.25, 0.25]"));
        AssertNear([0f, 0f, 0.2f], Gradient("E.hard_sigmoid", "[-2.5, 2.5, 0.]"));
        AssertNear([0f, 1f, 0.5f], Gradient("E.hard_swish", "[-3., 3., 0.]"));
        AssertNear([1e12f, 1e12f, 0.032f, -0.024f], Gradient("lambda x: N.lp_normalization(x, p=2)", "[[0., 0.], [3., 4.]]"));
        AssertNear([1e12f, 1e12f, 0.375f, 0.125f], Gradient("lambda x: N.lp_normalization(x, p=1)", "[[0., 0.], [1., -3.]]"));
    }

    [Fact]
    public void TestLayerNormalizationOfRowsWithALargeMeanAgreesWithItsFunctionBody()
        => Assert.True(AutoTest.AdvancedTestGraph<LayerNormalizationOfALargeMeanCheck>([], QeeNormLinalgAuditTests.LargeMeanRows,
            context: new ComputeContext(Torch)));

    [Fact]
    public void TestElementTypesTorchHasNoKernelForAreComputedExactly()
    {
        Assert.Equal("[0, 1, 1] torch.uint64", Evaluated("E.sign(u64(0, 5, 2**64 - 1))"));
        Assert.Equal("[2, 9223372036854775813, 7] torch.uint64", Evaluated("E.clip(u64(1, 2**64 - 1, 7), u64(2), u64(2**63 + 5))"));
        Assert.Equal("[5, 0] torch.uint32 [5, 0] torch.uint64", Evaluated("E.prelu(u32(5, 0), u32(2))") + " " + Evaluated("E.prelu(u64(5, 0), u64(2))"));
        Assert.Equal("[18446744073709551615] torch.uint64 [1] torch.uint64", Evaluated("R.reduce_max(u64(1, 2**64 - 1, 2**63))") + " " + Evaluated("R.reduce_min(u64(1, 2**64 - 1, 2**63))"));
        Assert.Equal("[2] torch.uint64", Evaluated("R.reduce_l1(u64(1, 2**64 - 1, 2))"));
        Assert.Equal("[1] torch.int64 [0] torch.int64", Evaluated("R.arg_max(u64(1, 2**64 - 1, 2**63))") + " " + Evaluated("R.arg_min(u64(1, 2**64 - 1, 2**63))"));
        foreach (var type in (ReadOnlySpan<string>)["uint16", "uint32", "uint64"])
            Assert.Equal($"[[1, 2], [0, 4]] torch.{type}", Evaluated($"S.trilu(torch.tensor(np.array([[1, 2], [3, 4]], np.{type})))"));
        foreach (var type in (ReadOnlySpan<string>)["uint32", "uint64"])
            Assert.Equal($"[[8, 11], [16, 23]] torch.{type}", Evaluated($"L.gemm(*[torch.tensor(np.array(v, np.{type})) for v in ([[1, 2], [3, 4]], [[1, 2], [3, 4]], [1])])"));
        Assert.Equal("-2.0 torch.float16", Evaluated("L.det(torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float16))"));
    }

    [Fact]
    public void TestFloat8CastsSaturateOrOverflowAsTheSpecificationSays()
    {
        Assert.Equal("[448.0, 448.0, nan, nan, nan, nan, nan] torch.float32",
            Evaluated("E.cast(E.cast(torch.tensor([448., 464., 465., -464.5, 1e5, float('inf'), float('nan')]), to=17, saturate=0), to=1)"));
        Assert.Equal("[57344.0, -57344.0, 57344.0] torch.float32",
            Evaluated("E.cast(E.cast(torch.tensor([float('inf'), float('-inf'), 1e5]), to=19), to=1)"));
    }

    [Fact]
    public void TestSplitIntoMoreOutputsThanAnEvenSplitFillsEndsInEmptyOnes()
        => Assert.Equal("[2, 2, 1, 0]", Evaluated("[len(t) for t in S.split(torch.arange(5.), num_outputs=4, _outputs=4)]"));

    [Fact]
    public void TestAnIntegerProductRunsOnTheCardOnlyWhereFloat64HoldsItExactly()
    {
        Assert.Equal("int64 int64", Evaluated("' '.join(L._integer_route([128, 128], n, 'cpu') for n in (1, 2**60))"));
        Assert.Equal("float64 cpu", Evaluated("' '.join(L._integer_route([128, 128], n, 'cuda') for n in (2**39, 2**39 + 1))"));
        Assert.Equal("float64 cpu", Evaluated("' '.join(L._integer_route([255, 255, 2], n, 'cuda') for n in (2**36, 2**37))"));
        Assert.Equal("128 255 32768 65535 2147483648 18446744073709551615 255 255",
            Evaluated("' '.join(str(L._largest(t, shifted)) for t, shifted in [(torch.int8, False), (torch.uint8, False), (torch.int16, False), (torch.uint16, False), (torch.int32, False), (torch.uint64, False), (torch.int8, True), (torch.uint8, True)])"));
        Assert.Equal("3 3 1 4", Evaluated("' '.join(str(L._summed_terms(e, s)) for e, s in [('ij,jk->ik', [(2, 3), (3, 4)]), ('ij,jk', [(2, 3), (3, 4)]), ('...i,...i->...i', [(5, 3), (5, 3)]), ('bij,bjk->b', [(7, 2, 2), (7, 2, 1)])])"));
    }

    private static float[] Gradient(string function, params string[] arguments)
        => [.. Evaluated($"""
            (lambda xs: ((({function})(*xs)).sum().backward(), [v for x in xs for v in (torch.zeros_like(x) if x.grad is None else x.grad).reshape(-1).tolist()])[1])(
                [torch.tensor(a, dtype=torch.float32, requires_grad=True) for a in ({string.Join(", ", arguments)},)])
            """).Trim('[', ']').Split(", ").Select(v => float.Parse(v, System.Globalization.CultureInfo.InvariantCulture))];

    private static string Evaluated(string expression, TorchBackend? backend = null)
    {
        (backend ?? Torch).Start();
        using (PythonRuntime.Gil())
        {
            using var scope = Py.CreateScope();
            scope.Exec($$"""
                import numpy as np
                import torch
                from shorokoo_torch import ops_elementwise as E, ops_linalg as L, ops_norm as N, ops_reduction as R, ops_shape as S
                u32 = lambda *v: torch.tensor(np.array(v, np.uint32))
                u64 = lambda *v: torch.tensor(np.array(v, np.uint64))
                value = {{expression}}
                result = f"{value.tolist()} {value.dtype}" if isinstance(value, torch.Tensor) else str(value)
                """);
            return scope.Get<string>("result");
        }
    }

    private static TorchUnsupportedModelException Refusal(GraphProto step, long version = 1)
        => Assert.Throws<TorchUnsupportedModelException>(() => Torch.CreateSession(TrainingStep(step, version), default, default, DeviceMemorySettings.Default));

    private static IDisposable OperatorTableGradient(string opType, string gradient)
        => Shorokoo.PythonTranslation.Operators.OperatorTable.OverrideGradient(
            opType, Enum.Parse<Shorokoo.PythonTranslation.Operators.GradientRule>(gradient));

    internal static NodeProto AutoGrad(string[] inputs, string[] outputs)
        => Node(TrainingFormats.AutoGradOpType, inputs, outputs, domain: TrainingFormats.AutoGradDomain);

    internal static NodeProto Cast(string input, string output)
    {
        var node = Node("Cast", [input], [output]);
        node.Attributes.Add(new AttributeProto { Name = "to", Type = AttributeProto.AttributeType.Int, I = (int)ShorokooTensorElementType.Float });
        return node;
    }

    internal static NodeProto Branch(string condition, GraphProto thenBranch, GraphProto elseBranch)
    {
        var node = Node("If", [condition], thenBranch.Outputs.Select(o => o.Name).ToArray());
        node.Attributes.Add(new AttributeProto { Name = "then_branch", Type = AttributeProto.AttributeType.Graph, G = thenBranch });
        node.Attributes.Add(new AttributeProto { Name = "else_branch", Type = AttributeProto.AttributeType.Graph, G = elseBranch });
        return node;
    }

    internal static byte[] TrainingStep(GraphProto graph, long version = 1, params FunctionProto[] functions)
    {
        var squash = new FunctionProto { Name = "Squash", Domain = "Functions" };
        squash.Inputs.Add("a");
        squash.Outputs.Add("b");
        squash.Nodes.Add(Node("Tanh", ["a"], ["b"]));
        squash.OpsetImports.Add(new OperatorSetIdProto { Domain = "", Version = 21 });
        var model = ProtoBuf.Serializer.Deserialize<ModelProto>(new MemoryStream(Serialize(graph, [squash, .. functions])));
        model.OpsetImports.Add(new OperatorSetIdProto { Domain = TrainingFormats.AutoGradDomain, Version = version });
        using var stream = new MemoryStream();
        ProtoBuf.Serializer.Serialize(stream, model);
        return stream.ToArray();
    }

    internal static float[][] RunFloats(IShorokooSession session, Dictionary<string, float[]> feeds, string[] outputs)
    {
        var inputs = feeds.ToDictionary(f => f.Key, f => Torch.CreateTensor(f.Value, [f.Value.Length]));
        try
        {
            var values = session.Run(inputs, outputs, RunSettings.Default);
            var floats = values.Select(v => v.GetTensorDataAsSpan<float>().ToArray()).ToArray();
            foreach (var value in values) value.Dispose();
            return floats;
        }
        finally
        {
            foreach (var input in inputs.Values) input.Dispose();
        }
    }

    internal static void AssertNear(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++) Assert.True(MathF.Abs(expected[i] - actual[i]) <= 1e-5f * MathF.Max(1f, MathF.Abs(expected[i])));
    }

    [Fact]
    public void TestEveryTorchBackendSharesOneRuntimeAndNamesItsOwnMemory()
    {
        var cuda = new TorchCudaBackend(1);

        Assert.Same(Torch.RuntimeIdentity, cuda.RuntimeIdentity);
        Assert.Same(Torch.RuntimeIdentity, new TorchCudaBackend().RuntimeIdentity);
        Assert.Equal(MemorySpace.Host, ((IShorokooBackend)Torch).MemorySpace);
        Assert.Equal(MemorySpace.Cuda(1), ((IShorokooBackend)cuda).MemorySpace);
        Assert.Equal(new BackendDescription("Shorokoo.PyTorch.Cuda", ComputeDevice.Cuda, 1), cuda.Description);
        Assert.Equal(new BackendDescription("Shorokoo.PyTorch.Cpu", ComputeDevice.Cpu, null), Torch.Description);
        Assert.Equal("cuda:1", cuda.DeviceName);
        Assert.True(((IShorokooBackend)Torch).CanAddress(new MemoryLocation(MemorySpace.Host, cuda.RuntimeIdentity)));
        Assert.False(((IShorokooBackend)Torch).CanAddress(new MemoryLocation(MemorySpace.Host, DefaultBackend.Instance.RuntimeIdentity)));
        Assert.Equal(new MemoryLocation(MemorySpace.Cuda(1), Torch.RuntimeIdentity), ((IShorokooBackend)cuda).RunMemoryOf(ShorokooTensorElementType.Float));
        Assert.Equal(new MemoryLocation(MemorySpace.Host, Torch.RuntimeIdentity), ((IShorokooBackend)cuda).RunMemoryOf(ShorokooTensorElementType.String));
        Assert.Equal(new MemoryLocation(MemorySpace.Host, Torch.RuntimeIdentity), ((IShorokooBackend)cuda).SequenceRunMemory);
        Assert.True(((IShorokooBackend)cuda).CanAddress(new MemoryLocation(MemorySpace.Cuda(1), Torch.RuntimeIdentity)));
        Assert.False(((IShorokooBackend)cuda).CanAddress(new MemoryLocation(MemorySpace.Cuda(0), Torch.RuntimeIdentity)));
        Assert.False(((IShorokooBackend)cuda).CanAddress(new MemoryLocation(MemorySpace.Host, Torch.RuntimeIdentity)));
        Assert.Throws<ArgumentOutOfRangeException>(() => new TorchCudaBackend(-1));
    }

    [Fact]
    public void TestEachPlatformHasItsOwnLockAndThisMachineRunsItsOwn()
    {
        var windowsCpu = PythonEnvironmentLock.ForPlatform("cpu", "win-x64");
        var windowsCuda = PythonEnvironmentLock.ForPlatform("cu13", "win-x64");
        var linuxCuda = PythonEnvironmentLock.ForPlatform("cu13", "linux-x64");

        Assert.Equal("linux-x64", PythonEnvironmentLock.PlatformOf(OSPlatform.Linux, Architecture.X64));
        Assert.Equal("win-x64", PythonEnvironmentLock.PlatformOf(OSPlatform.Windows, Architecture.X64));
        Assert.Null(PythonEnvironmentLock.PlatformOf(OSPlatform.OSX, Architecture.X64));
        Assert.Null(PythonEnvironmentLock.PlatformOf(OSPlatform.Linux, Architecture.Arm64));
        Assert.Equal(PythonEnvironmentLock.ForPlatform("cpu", PythonEnvironmentLock.CurrentPlatform()).Hash, PythonEnvironmentLock.Cpu.Hash);
        Assert.Contains("torch==2.14.0+cpu", windowsCpu.Requirements);
        Assert.Contains("https://download.pytorch.org/whl/cpu", windowsCpu.IndexArguments);
        Assert.Contains("torch==2.14.0+cu130", windowsCuda.Requirements);
        Assert.Contains("https://download.pytorch.org/whl/cu130", windowsCuda.IndexArguments);
        Assert.Contains("nvidia-cublas", linuxCuda.Requirements);
        foreach (var platform in PythonEnvironmentLock.Platforms)
            foreach (var requirements in ((string[])["cpu", "cu13"]).Select(name => PythonEnvironmentLock.ForPlatform(name, platform).Requirements))
                Assert.DoesNotContain(requirements.Split('\n'), line => line.Contains("==") && !line.TrimEnd().EndsWith('\\'));
        Assert.NotEqual(windowsCpu.Hash, PythonEnvironmentLock.ForPlatform("cpu", "linux-x64").Hash);
        Assert.Throws<PlatformNotSupportedException>(() => PythonEnvironmentLock.ForPlatform("cpu", "osx-arm64"));
        Assert.Throws<PlatformNotSupportedException>(() => PythonEnvironmentLock.ForPlatform("cpu", null));
        Assert.Throws<PlatformNotSupportedException>(() => PythonEnvironmentLock.ForPlatform("rocm", "linux-x64"));
    }

    [Fact]
    public void TestAWindowsEnvironmentIsReadTheWayWindowsLaysItOut()
    {
        var version = new System.Version(3, 12, 11);
        var home = Path.Combine("C:", "uv", "cpython-3.12.11");
        var env = Path.Combine("C:", "envs", "cpu");

        Assert.Equal(home, PythonEnvironment.PythonHomeOf(home + Path.DirectorySeparatorChar, windows: true));
        Assert.Equal([Path.Combine(home, "python312.dll")], PythonEnvironment.LibPythonCandidates(home, version, windows: true));
        Assert.Equal(Path.Combine(env, "Lib", "site-packages"), PythonEnvironment.SitePackagesOf(env, version, windows: true));
        var py = Path.Combine(Path.GetTempPath(), "py");
        Assert.Equal(py, PythonEnvironment.PythonHomeOf(Path.Combine(py, "bin"), windows: false));
        Assert.Contains(Path.Combine(py, "lib", "libpython3.12.so"), PythonEnvironment.LibPythonCandidates(py, version, windows: false));
        Assert.Equal(Path.Combine(env, "lib", "python3.12", "site-packages"), PythonEnvironment.SitePackagesOf(env, version, windows: false));
        Assert.Equal(Path.Combine("LocalAppData", "shorokoo", "python-envs"),
            PythonEnvironmentResolver.CacheRoot(new(), name => name == "LOCALAPPDATA" ? "LocalAppData" : "/xdg", windows: true));
        Assert.Equal(Path.Combine("/xdg", "shorokoo", "python-envs"),
            PythonEnvironmentResolver.CacheRoot(new(), name => name == "LOCALAPPDATA" ? "LocalAppData" : "/xdg", windows: false));
    }

    [NoCudaDriverFact]
    public void TestTheCudaBackendOnAMachineWithoutADriverRefusesToStartBeforeProvisioningAnything()
    {
        var cache = Path.Combine(Path.GetTempPath(), "shorokoo-nodriver-" + Guid.NewGuid().ToString("N"));
        var refusal = Assert.Throws<PythonEnvironmentException>(() => new TorchCudaBackend(0, new() { CacheDirectory = cache }).Start());

        Assert.Equal(PythonEnvironmentFailure.DeviceUnavailable, refusal.Failure);
        Assert.Contains("NVIDIA driver", refusal.Message);
        Assert.False(Directory.Exists(cache));
    }

    [Fact]
    public void TestARunIsStoppedBetweenNodesWhenItsTokenIsCancelledWhileItRuns()
    {
        using var session = Torch.CreateSession(Serialize(CountingLoop()), default, default, DeviceMemorySettings.Default);
        using var cancelled = new CancellationTokenSource();
        cancelled.Cancel();
        using var later = new CancellationTokenSource(TimeSpan.FromMilliseconds(300));
        IReadOnlyList<IShorokooTensorValue> Count(long iterations, CancellationToken token)
        {
            using var m = Torch.CreateTensor([iterations], []);
            using var start = Torch.CreateTensor([0f], []);
            return session.Run(new Dictionary<string, IShorokooTensorValue> { ["m"] = m, ["v"] = start }, ["y"], new RunSettings { CancellationToken = token });
        }

        Assert.Equal(later.Token, Assert.Throws<OperationCanceledException>(() => Count(10_000_000, later.Token)).CancellationToken);
        Assert.Equal(cancelled.Token, Assert.Throws<OperationCanceledException>(() => Count(1, cancelled.Token)).CancellationToken);
        Assert.Equal([5f], Count(5, new CancellationTokenSource().Token)[0].GetTensorDataAsSpan<float>().ToArray());
        Assert.Equal([5f], Count(5, CancellationToken.None)[0].GetTensorDataAsSpan<float>().ToArray());
    }

    [Fact]
    public void TestAWarningARunRaisesIsShownOnlyWhereItsSessionsLogSeverityAsksForWarnings()
    {
        Torch.Start();
        using (PythonRuntime.Gil())
        {
            using var scope = Py.CreateScope();
            scope.Exec("""
                import warnings
                from shorokoo_torch import runtime as rt
                shown = []
                original = rt._show_warning
                rt._show_warning = lambda message, *rest: shown.append(str(message))
                try:
                    for severity in (None, 0, 1, 2, 3, 4):
                        token = rt._warning_severity.set(severity)
                        try:
                            warnings.warn_explicit(f"at {severity}", UserWarning, "model", 1)
                        finally:
                            rt._warning_severity.reset(token)
                finally:
                    rt._show_warning = original
                """);
            Assert.Equal(["at None", "at 0", "at 1", "at 2"], scope.Get<string[]>("shown"));
        }
    }

    [Fact]
    public void TestACpuSessionHasNoArenaFiguresLeavesItsOutputsInHostMemoryAndRunsEveryNodeOnTheHost()
    {
        var graph = ComputeContextLifetimeCoverageTests.GraphOf("a:float[2] b:float[2]", "O:float[2]",
            ComputeContextLifetimeCoverageTests.Op("Sub", "a b", "t"), ComputeContextLifetimeCoverageTests.Op("Neg", "t", "O"));
        using var traced = Torch.CreateSession(Serialize(graph), default, default, DeviceMemorySettings.Default, new DiagnosticSettings { TraceNodePlacement = true });
        using var plain = Torch.CreateSession(Serialize(graph), default, default, DeviceMemorySettings.Default);
        using var a = Torch.CreateTensor([5f, 7f], [2]);
        using var b = Torch.CreateTensor([1f, 2f], [2]);
        var feeds = new Dictionary<string, IShorokooTensorValue> { ["a"] = a, ["b"] = b };
        using var kept = traced.Run(feeds, ["O"], RunSettings.Default)[0];

        Assert.True(kept.IsHostAccessible);
        Assert.Equal([-4f, -5f], kept.GetTensorDataAsSpan<float>().ToArray());
        Assert.Null(traced.ReadArenaStatistics());
        Assert.Null(traced.ReadPinnedArenaStatistics());
        Assert.Equal(SessionOutputPlacement.Host, traced.OutputPlacement);
        Assert.Null(plain.ReadNodePlacement());
        Assert.Equal([("Sub", "cpu"), ("Neg", "cpu")], traced.ReadNodePlacement()!.Nodes.Select(n => (n.OpType, n.Provider)));
        Assert.Equal([new ProviderShare("cpu", 2, 0, 0, 0)], traced.ReadNodePlacement()!.Providers);
    }

    [Fact]
    public void TestACudaSessionPlacesTensorOutputsOnTheCardAndStringsAndSequencesOnTheHost()
    {
        var tensors = ComputeContextLifetimeCoverageTests.GraphOf("a:float[2]", "x:float[2] y:float[2]");
        var mixed = ComputeContextLifetimeCoverageTests.GraphOf("a:float[2]", "x:float[2] s:string[2]");
        var strings = ComputeContextLifetimeCoverageTests.GraphOf("a:float[2]", "s:string[2]");
        var sequence = ComputeContextLifetimeCoverageTests.GraphOf("a:float[2]", "x:float[2]");
        sequence.Outputs.Add(new ValueInfoProto { Name = "q", Type = new TypeProto { SequenceType = new TypeProto.Sequence() } });

        Assert.Equal(SessionOutputPlacement.Device, TorchSession.OutputPlacementOf(tensors, onCuda: true));
        Assert.Equal(SessionOutputPlacement.Mixed, TorchSession.OutputPlacementOf(mixed, onCuda: true));
        Assert.Equal(SessionOutputPlacement.Host, TorchSession.OutputPlacementOf(strings, onCuda: true));
        Assert.Equal(SessionOutputPlacement.Mixed, TorchSession.OutputPlacementOf(sequence, onCuda: true));
        Assert.Equal(SessionOutputPlacement.Host, TorchSession.OutputPlacementOf(tensors, onCuda: false));
        Assert.Equal(["cuda:1"], TorchSession.Placement(ComputeContextLifetimeCoverageTests.GraphOf("a", "b", ComputeContextLifetimeCoverageTests.Op("Neg", "a", "b")), "cuda:1").Providers.Select(p => p.Provider));
    }

    [Fact]
    public void TestAnOutputIsWrittenIntoTheInputItsRunConsumedOnlyWhereNothingLaterCanStillReadThatMemory()
    {
        Assert.Equal(("a", "9 18 27 36"), Aliased("a:float[4] b:float[4]", "O:float[4]", [Op("Sub", "a b", "O")]));
        Assert.Equal(("a", "9 38 87 156"), Aliased("a:float[4] b:float[4]", "O:float[4]", [Op("Mul", "a b", "m"), Op("Neg", "b", "n"), Op("Add", "m n", "O")]));
        Assert.Equal(("a", "9 19 29 39"), Aliased("a:float[4] b:float[1]", "O:float[4]", [Op("Sub", "a b", "O")], b: [1f]));
        Assert.Equal((null, "9 18 27 36"), Aliased("a:float[4] b:float[4]", "O:float[4]", [Op("Sub", "a b", "O")], consume: false));
        Assert.Equal((null, "0 0 0 0"), Aliased("a:float[4] b:float[4]", "O:float[4]", [Op("Sub", "a b", "O")], feedTwice: true));
        Assert.Equal((null, "20 40 60 80"), Aliased("a:float[4] b:float[4]", "O:float[4]", [Op("Add", "a a", "O")]));
        Assert.Equal((null, "300"), Aliased("a:float[4] b:float[4]", "O", [Op("MatMul", "b a", "O")]));
        Assert.Equal((null, "-9 -18 -27 -36"), Aliased("a:float[4] b:float[4]", "O:float[4]", [Op("Sub", "a b", "t"), Op("Neg", "t", "O")]));
        Assert.Equal((null, "9 18 27 36"), Aliased("a:int64[4] b:int64[4]", "O:int64[4]", [Op("Sub", "a b", "O")], integers: true));
        Assert.Equal((null, "0 -20 -60 -120 -10 -20 -30 -40"), Aliased("a:float[4] b:float[4]", "O:float[4] T:float[4]", [Op("Transpose", "a", "t"), Op("Mul", "t b", "m"), Op("Neg", "t", "T"), Op("Sub", "a m", "O")]));
        Assert.Equal((null, "0 -20 -60 -120 10 20 30 40"), Aliased("a:float[4] b:float[4]", "O:float[4] T:float[4]", [Op("Transpose", "a", "T"), Op("Mul", "T b", "m"), Op("Sub", "a m", "O")]));
    }

    [Fact]
    public void TestAnOutputIsNotWrittenIntoItsInputWhereAValueWrittenOverThatInputIsStillRead()
    {
        Assert.Equal((null, "9 38 87 156 35400"), Aliased("a:float[4] b:float[4]", "O:float[4] M", [Op("Mul", "a b", "t"), Op("ReduceSumSquare", "t", "M"), Op("Sub", "t b", "O")]));
        Assert.Equal((null, "9 38 87 156 10 40 90 160"), Aliased("a:float[4] b:float[4]", "O:float[4] T:float[4]", [Op("Mul", "a b", "T"), Op("Sub", "T b", "O")]));
        Assert.Equal(("a", "9 38 87 156"), Aliased("a:float[4] b:float[4]", "O:float[4]", [Op("Mul", "a b", "t"), Op("Sub", "t b", "O")]));
    }

    [Fact]
    public void TestAnAliasedOutputHoldsTheConsumedMemoryWhichTheRunReleasesExactlyOnce()
    {
        var graph = ComputeContextLifetimeCoverageTests.GraphOf("a:float[4] b:float[4]", "O:float[4]", Op("Sub", "a b", "O"));
        using var session = Aliasing(graph);
        var a = Torch.CreateTensor([10f, 20f, 30f, 40f], [4]);
        using var b = Torch.CreateTensor([1f, 2f, 3f, 4f], [4]);
        ref var consumedMemory = ref MemoryMarshal.GetReference(a.GetTensorDataAsSpan<float>());
        var feeds = new Dictionary<string, IShorokooTensorValue> { ["a"] = a, ["b"] = b };
        using var o = session.RunConsuming(feeds, [a], ["O"], RunSettings.Default)[0];

        Assert.Equal([new OutputAlias("O", "a")], session.BindableAliases);
        Assert.Equal([9f, 18f, 27f, 36f], o.GetTensorDataAsSpan<float>().ToArray());
        Assert.True(Unsafe.AreSame(ref consumedMemory, ref MemoryMarshal.GetReference(o.GetTensorDataAsSpan<float>())));
        Assert.Throws<ObjectDisposedException>(() => a.IsHostAccessible);
        Assert.Empty(Aliasing(ComputeContextLifetimeCoverageTests.GraphOf("a:float[2,2] b:float[2,2]", "O:float[2,2]", Op("MatMul", "b a", "O"))).BindableAliases);
    }

    [Fact]
    public void TestACompiledGraphOnTorchWritesAMarkedOutputIntoTheTensorItsRunConsumed()
    {
        using var context = new ComputeContext(Torch);
        var a = InputVector<float32>("a");
        var b = InputVector<float32>("b");
        var compiled = context.Compile(new InternalComputationGraph([a, b], [a - b]), [[4L], [4L]], trainingStep: false, aliasCandidates: [(0, 0)]);
        long Aliased(IData x)
        {
            var before = context.AliasedOutputs;
            Assert.Equal([9f, 18f, 27f, 36f], compiled.Execute(x, TensorData([4L], 1f, 2f, 3f, 4f))[0].ToTensorData().As<float32>().CopyMemory<float>());
            return context.AliasedOutputs - before;
        }
        var kept = TensorData([4L], 10f, 20f, 30f, 40f);

        Assert.Equal(1, Aliased(TensorData([4L], 10f, 20f, 30f, 40f)));
        Assert.Equal(1, Aliased(TensorData([4L], 10f, 20f, 30f, 40f).To(context)));
        Assert.Equal(0, Aliased(kept.Shared()));
        Assert.Equal([10f, 20f, 30f, 40f], kept.As<float32>().CopyMemory<float>());
    }

    [Fact]
    public void TestEveryPairASessionBindsIsWrittenIntoItsOwnInputWhereASessionCannotBindAnEarlierPair()
    {
        var graph = ComputeContextLifetimeCoverageTests.GraphOf("a:float[3] b:float[3] c:float[3] g:float[3]", "O0:float[3] O1:float[3] O2:float[3]",
            Op("Mul", "a g", "t"), Op("Neg", "t", "O0"), Op("Sub", "b g", "O1"), Op("Add", "c g", "O2"));
        using var session = Torch.CreateSession(Serialize(graph), default, default, DeviceMemorySettings.Default, DiagnosticSettings.Default,
            [new OutputAlias("O0", "a"), new OutputAlias("O1", "b"), new OutputAlias("O2", "c")]);

        Assert.Equal("-,b,c -1 -1 -1 9 9 9 101 101 101", AliasedRun(session, graph));
        Assert.Equal([new OutputAlias("O1", "b"), new OutputAlias("O2", "c")], session.BindableAliases);
    }

    [Fact]
    public void TestTwoSessionsOverOneModelWithDifferentPairsEachRunTheirOwnTranslation()
    {
        var graph = ComputeContextLifetimeCoverageTests.GraphOf("b:float[3] c:float[3] g:float[3]", "O2:float[3] O1:float[3]", Op("Add", "c g", "O2"), Op("Sub", "b g", "O1"));
        using var first = Torch.CreateSession(Serialize(graph), default, default, DeviceMemorySettings.Default, DiagnosticSettings.Default, [new OutputAlias("O2", "c")]);
        using var second = Torch.CreateSession(Serialize(graph), default, default, DeviceMemorySettings.Default, DiagnosticSettings.Default, [new OutputAlias("O1", "b"), new OutputAlias("O2", "c")]);

        Assert.Equal("c,- 101 101 101 9 9 9", AliasedRun(first, graph));
        Assert.Equal("c,b 101 101 101 9 9 9", AliasedRun(second, graph));
    }

    [Fact]
    public void TestTheTwoHalvesScenarioWritesEveryValueIntoTheMemoryItConsumesOnTorchAndItsOutputsOutliveTheSession()
    {
        const int Rows = 512, Columns = 1024;
        var (a, b, l) = ComputeContextLifetimeCoverageTests.TwoHalvesValues(Rows, Columns);
        NamedModelParam[] outputs = [];
        using (var context = new ComputeContext(Torch))
        {
            var compiled = context.Compile(ComputeContextLifetimeCoverageTests.TwoHalves());
            for (int run = 0; run < 2; run++)
            {
                outputs = compiled.Execute(TensorData([(long)Rows, Columns], a), TensorData([(long)Rows, Columns], b));
                Assert.True(l.Zip(Floats(outputs[0]), (x, y) => MathF.Abs(x - y) < 1e-5f).All(x => x));
                Assert.Equal(a[..(Rows / 2 * Columns)], Floats(outputs[1]));
                Assert.Equal(b[(Rows / 2 * Columns)..], Floats(outputs[2]));
            }
            var entry = Assert.Single(((TorchSession)compiled.Session).Placements!.Entries);
            Assert.Equal(10, entry.Plan.Count);
            Assert.All(outputs, o => Assert.NotNull(o.ToTensorData().Block));
            Assert.Same(outputs[1].ToTensorData().Block, outputs[2].ToTensorData().Block);
        }
        GC.Collect();
        GC.WaitForPendingFinalizers();
        Assert.True(l.Zip(Floats(outputs[0]), (x, y) => MathF.Abs(x - y) < 1e-5f).All(x => x));
        Assert.Equal(b[(Rows / 2 * Columns)..], Floats(outputs[2]));
    }

    private static float[] Floats(NamedModelParam value) => [.. value.ToTensorData().As<float32>().AccessMemory<float>()];

    [Fact]
    public void TestARunWritesAValueIntoTheMemoryItConsumesOnlyWhereItIsProvedFreeAndTorchCanWriteItThere()
    {
        Assert.Equal("O@a+0", Placed(Graph("O", Op("Neg", "a", "O"))));
        Assert.Equal("O@-", Placed(Graph("O", Op("Neg", "a", "O")), consume: false));
        Assert.Equal("O@-", Placed(Graph("O", Op("Neg", "a", "O")), feedTwice: true));
        Assert.Equal("O@-", Placed(GraphOn("a:int64[131072] b:int64[131072]", "O", Op("Neg", "a", "O"))));
        Assert.Equal("x@a+0 O@a+1048576", Placed(Halved("x O", Op("Slice", "a zero half zero", "x"), Op("Neg", "b", "n"), Op("Slice", "n zero half zero", "O"))));
        Assert.Equal("x@a+1048576 O@a+0", Placed(Halved("x O", Op("Slice", "a half end zero", "x"), Op("Exp", "b", "e"), Op("Slice", "e zero half zero", "s"), Op("Neg", "s", "O"))));
        Assert.Equal("O@- Z@-", Placed(GraphOn("a:float[262144]", "O Z", ComputeContextLifetimeCoverageTests.Op("Cast", "a", "y", attribute: ("to", 1)), Op("Neg", "y", "O"), Op("Exp", "y", "Z"))));
        Assert.Equal("O@b+0", Placed(GraphOn("a:float[512,512] b:float[512,512]", "O", Op("Transpose", "a", "O"))));
        Assert.Equal("O@b+0", Placed(GraphOn("a:float[512,512] b:float[512,512] w:float[512,512]", "O", Op("MatMul", "a w", "O"))));
        Assert.Equal("O@a+0 Z@b+0", Placed(Graph("O Z", Op("Exp", "a", "e"), Op("Neg", "e", "O"), Op("Sigmoid", "b", "s"), Op("Add", "s e", "Z"))));
        Assert.Equal("O@-", Placed(GraphOn("a:float[64] b:float[64]", "O", Op("Neg", "a", "t"), Op("Exp", "t", "u"), Op("Add", "u b", "O"))));
        Assert.Equal("O@-", Placed(GraphOn("a:float[64] b:float[64]", "O", Op("Neg", "a", "t"), Op("Greater", "t b", "m"), Op("Where", "m b t", "O"))));
        Assert.Equal("O@-", Placed(GraphOn("a:float[8,8] s:float[8]", "O", Op("Softmax", "a", "t"), Op("LayerNormalization", "t s", "O"))));
    }

    [Fact]
    public void TestAValueTorchCannotWriteIntoARangeIsNotPlacedAndOneHandedARangeAnywayIsComputedAndCopiedThere()
    {
        Assert.Equal("O@-", Placed(GraphOn("a:float[262144] e:int64[262144]", "O", Op("Pow", "a e", "O"))));
        Assert.Equal("O@-", Placed(GraphOn("a:float[262144,4] v:float[4] b:float[262144]", "O", Op("MatMul", "a v", "O"))));
        Assert.Equal("(True, [8.0, 8.0, 8.0, 8.0])", PlacedInto("E.pow_, torch.full((4,), 2.0), torch.full((4,), 3)"));
        Assert.Equal("(True, [2.0, 2.0, 2.0, 2.0])", PlacedInto("L.matmul, torch.ones(4, 2), torch.ones(2)"));
    }

    /// <summary>Whether <c>place_into</c> called with <paramref name="call"/> for a slot over a range
    /// of four floats hands back that range, and what the range then holds.</summary>
    private static string PlacedInto(string call)
        => Evaluated("(lambda rt, t: __import__('contextvars').copy_context().run(lambda: (rt._placing.set(rt._Placing([t], [t], [(0, 0, 16, 1, [4], -1)], torch.device('cpu'), [])), "
                     + $"rt.place_into(0, False, {call}).data_ptr() == t.data_ptr(), t.tolist())[1:]))(__import__('shorokoo_torch.runtime', fromlist=['_']), torch.zeros(4))");

    [Fact]
    public void TestASessionStoppedFromPlacingWritesNothingOverTheInputsItsRunsConsumeAndOneBuiltForASingleRunStillDoes()
    {
        Assert.Equal([-1f, -2f, -3f, -4f], ConsumedAfterARun(null));
        Assert.Equal([1f, 2f, 3f, 4f], ConsumedAfterARun(null, stopPlacing: true));
        Assert.Equal([-1f, -2f, -3f, -4f], ConsumedAfterARun(new ComputeContext(Torch)));
        Assert.Equal([-1f, -2f, -3f, -4f], ConsumedAfterARun(new ComputeContext(Torch), placing: false));
        Assert.Equal([1f, 2f, 3f, 4f], ConsumedAfterARun(new ComputeContext(Torch) { ValuePlacement = false }));
        Assert.Equal([1f, 2f, 3f, 4f], ConsumedAfterARun(new ComputeContext(Torch) { OutputAliasing = false }));
    }

    /// <summary>What the memory of the input a run of <c>Neg</c> consumed holds once the run is over:
    /// a run of a session of the backend's own, stopped from placing where
    /// <paramref name="stopPlacing"/>, or of one <paramref name="context"/> builds, built for a single
    /// run where not <paramref name="placing"/>.</summary>
    private static float[] ConsumedAfterARun(ComputeContext? context, bool placing = true, bool stopPlacing = false)
    {
        var model = Serialize(GraphOn("x:float[4]", "O", Op("Neg", "x", "O")));
        using var owner = context;
        using var session = context?.BuildSession(Torch, model, ShorokooGraphOptimization.EnableAll, DeviceMemorySettings.Default, placing: placing)
            ?? Torch.CreateSession(model, default, default, DeviceMemorySettings.Default);
        if (stopPlacing) session.StopPlacing();
        var x = (TorchTensorValue)Torch.CreateTensor([1f, 2f, 3f, 4f], [4]);
        TorchTensorValue seen;
        using (PythonRuntime.Gil()) seen = TorchTensorValue.Wrap(Torch.Runtime, x.Value.InvokeMethod("detach"), ShorokooTensorElementType.Float);
        using (seen)
        {
            session.RunConsuming(new Dictionary<string, IShorokooTensorValue> { ["x"] = x }, [x], ["O"], RunSettings.Default)[0].Dispose();
            return seen.GetTensorDataAsSpan<float>().ToArray();
        }
    }

    [Fact]
    public void TestAShapeIsReadAsItsValueIsMadeSoThatTheValueIsHeldOnlyUntilItsContentsAreRead()
    {
        Assert.Equal("neg shape exp abs_ reshape", Calls(GraphOn("x:float[4]", "e", Op("Neg", "x", "b"), Op("Shape", "b", "s"), Op("Exp", "b", "c"), Op("Abs", "c", "d"), Op("Reshape", "d s", "e"))));
        Assert.Equal("size neg", Calls(GraphOn("x:float[4]", "b s", Op("Neg", "x", "b"), Op("Size", "x", "s"))));
        Assert.Equal("neg shape shape exp", Calls(GraphOn("x:float[4]", "c r", Op("Neg", "x", "b"), Op("Exp", "b", "c"), Op("Shape", "b", "s"), Op("Shape", "s", "r"))));
        Assert.Equal("neg exp shape", Calls(GraphOn("x:float[4]", "c s", Op("Neg", "x", "b"), Op("Exp", "b", "c"), Op("Shape", "c", "s"))));
    }

    /// <summary>The support functions the translation of <paramref name="graph"/> calls, in the order
    /// it calls them, each one written over an operand of it marked so.</summary>
    private static string Calls(GraphProto graph, bool over = false)
    {
        var model = ProtoBuf.Serializer.Deserialize<ModelProto>(new MemoryStream(Serialize(graph)));
        var source = Shorokoo.PythonTranslation.OnnxToPythonTranslator.Translate(model, [], TorchDialect.Instance, null,
            over ? TorchInPlace.Plan(model.Graph) : null).Source;
        return string.Join(" ", System.Text.RegularExpressions.Regex.Matches(source, @"(_over\(\w+, )?ops_\w+\.(\w+)[(,]")
            .Select(m => (m.Groups[1].Success ? "over:" : "") + m.Groups[2].Value));
    }

    [Fact]
    public void TestTheMemoryAwarePassHandsATorchContextNoStepHoldingMoreThanTheStepItWasHanded()
    {
        Assert.True(PassHoldsNoMoreOnTorch(Benchmarks.MemoryPassConv.ComputationGraph, [8L, 3L, 64L, 64L]));
        Assert.True(PassHoldsNoMoreOnTorch(Benchmarks.MemoryPassMlp.ComputationGraph, [64L, 256L]));
        Assert.True(PassHoldsNoMoreOnTorch(ChunkedSdpaMeanPoolModel.ComputationGraph, [2L, 4L, 256L, 32L]));
    }

    [Fact]
    public void TestTheMemoryAwarePassTakesATwoLayerEncodersStepOnATorchContextBelowTwoThirdsOfItsPeak()
    {
        using var context = new ComputeContext(Torch);
        var sample = TensorData([128L, 128L, 128L], new float[128 * 128 * 128]);
        var result = TrainingRig.FromScratch(Benchmarks.MemoryPassEncoder2.ComputationGraph, Shorokoo.Modules.Losses.L2Loss.ComputationGraph,
            Shorokoo.Modules.Optimizers.AdamWOptimizer.ComputationGraph, [sample],
            new Shorokoo.Modules.Optimizers.AdamWOptimizerHyperparameters { LearningRate = 0.001f }, runtimeContext: context).OptimizationResult;
        Assert.True(3 * result.Evaluation.PeakMemoryBytes <= 2 * result.AllStrategies[0].Evaluation.PeakMemoryBytes);
    }

    [Fact]
    public void TestTheMemoryAwarePassTakesATwoLayerEncodersStepAsLowAsTorchsModelOfACardRunSeesIt()
    {
        using var context = new ComputeContext(new JudgedAsOnACard());
        var sample = TensorData([128L, 128L, 128L], new float[128 * 128 * 128]);
        var result = TrainingRig.FromScratch(Benchmarks.MemoryPassEncoder2.ComputationGraph, Shorokoo.Modules.Losses.L2Loss.ComputationGraph,
            Shorokoo.Modules.Optimizers.AdamWOptimizer.ComputationGraph, [sample],
            new Shorokoo.Modules.Optimizers.AdamWOptimizerHyperparameters { LearningRate = 0.001f }, runtimeContext: context).OptimizationResult;
        var chosen = result.AllStrategies.Select(s => s.Graph).ToList().FindIndex(g => ReferenceEquals(g, result.OptimizedGraph));
        Assert.True(result.BackendPeakBytes![chosen] <= 217L << 20);
    }

    [Fact]
    public void TestTheMemoryAwarePassTakesARecurrentStepTorchModelsBelowItsPeakComputingWhatItWasHanded()
    {
        using var context = new ComputeContext(Torch);
        var step = TrainingRigFromScratchCoverageTests.RecurrentStep(context);
        Assert.True(step.Chosen < step.Handed);
        Assert.Equal(step.HandedComputes, step.ChosenComputes);
    }

    /// <summary>Torch on the CPU, its steps judged by torch's model of a run on a card.</summary>
    private sealed class JudgedAsOnACard() : TorchBackend(() => PythonEnvironmentLock.Cpu, null, cudaDeviceId: null), IShorokooBackend
    {
        long? IShorokooBackend.ModelledRunPeak(ModelProto model, IReadOnlyList<OutputAlias> outputAliases, PrecisionSettings precision)
            => TorchRunMemory.Peak(model, outputAliases, onHost: false);
    }

    [Fact]
    public void TestATorchRunIsModelledHoldingTheTemporariesItsKernelsMakeOnTheHost()
    {
        Assert.Equal(32768, RunPeak(GraphOn("x:float[64,64] s:float[64]", "O", Op("Exp", "x", "e"), Op("LayerNormalization", "e s", "O"))));
        Assert.Equal(65536, RunPeak(GraphOn("x:float[64,64] s:float[64]", "P", Op("Exp", "x", "e"), Op("LayerNormalization", "e s", "O"), Op("Add", "e O", "P"))));
        Assert.Equal(4096, RunPeak(GraphOn("x:float[2,3,8,8] w:float[4,3,3,3]", "O", ComputeContextLifetimeCoverageTests.With(Op("Conv", "x w", "O"), "pads", 1, 1, 1, 1))));
        Assert.Equal(688, RunPeak(GraphOn("x:float[1,4,4,4] w:float[4,3,3,3]", "O", Op("ConvTranspose", "x w", "O"))));
        Assert.Equal(7600, RunPeak(GraphOn("x:float[2,3,8,8] g:float[2,4,8,8]", "O", Op("Exp", "x", "a"), Op("Exp", "g", "b"),
            ComputeContextLifetimeCoverageTests.With(Op("Transpose", "a", "at"), "perm", 1, 0, 2, 3), ComputeContextLifetimeCoverageTests.With(Op("Transpose", "b", "bt"), "perm", 1, 0, 2, 3),
            ComputeContextLifetimeCoverageTests.With(Op("Conv", "at bt", "O"), "pads", 1, 1, 1, 1))));
    }

    [Fact]
    public void TestATorchRunIsModelledHoldingTheCopiesItsMatMulMakesOfOperandsItCannotReadInPlace()
    {
        var rows = ComputeContextLifetimeCoverageTests.WithInts(GraphOn("x:float[16,1,8] w:float[8,4]", "O", Op("Exp", "x", "e"), Op("Expand", "e s", "r"), Op("MatMul", "r w", "O")), "s", 16, 32, 8);
        Assert.Equal(25088, RunPeak(rows, onHost: false));
        Assert.Equal(8704, RunPeak(rows, onHost: true));
        var heads = GraphOn("x:float[2,8,4,4] w:float[2,4,4,8]", "O", Op("Exp", "x", "e"),
            ComputeContextLifetimeCoverageTests.With(Op("Transpose", "e", "t"), "perm", 0, 2, 1, 3), Op("MatMul", "t w", "O"));
        Assert.Equal(4096, RunPeak(heads, onHost: false));
        Assert.Equal(4096, RunPeak(heads, onHost: true));
        var folded = GraphOn("x:float[8,8,16] w:float[8,4]", "O", Op("Exp", "x", "e"),
            ComputeContextLifetimeCoverageTests.With(Op("Transpose", "e", "t"), "perm", 1, 2, 0), Op("MatMul", "t w", "O"));
        Assert.Equal(6144, RunPeak(folded, onHost: false));
    }

    [Fact]
    public void TestATorchRunIsModelledLayingAnElementWiseResultOutAsItsOperandsAreLaidOut()
    {
        NodeProto Heads() => ComputeContextLifetimeCoverageTests.With(Op("Transpose", "e", "t"), "perm", 0, 2, 1, 3);
        Assert.Equal(3072, RunPeak(ComputeContextLifetimeCoverageTests.WithInts(GraphOn("x:float[2,8,4,4] s:float[1]", "O", Op("Exp", "x", "e"), Heads(),
            Op("Mul", "t s", "m"), Op("Reshape", "m k", "r"), Op("Reshape", "e k", "v"), Op("Add", "r v", "O")), "k", 2, 4, 32)));
        Assert.Equal(3072, RunPeak(GraphOn("x:float[2,8,4,4] s:float[1]", "O", Op("Exp", "x", "e"), Heads(),
            Op("Mul", "t s", "m"), Op("Neg", "m", "n"), Op("Add", "n t", "O"))));
    }

    [Fact]
    public void TestATorchRunIsModelledHoldingWhatItsLoopsListsAndStacksHoldAndWhatItsRecurrentLayersStack()
    {
        Assert.Equal(768, RunPeak(ComputeContextLifetimeCoverageTests.WithInts(GraphOn("x:float[4,8]", "O", Op("SequenceEmpty", "", "s"),
            Loop("m s", "S", GraphOn("i c t", "k u", Op("Neg", "x", "n"), Op("SequenceInsert", "t n", "u"), Op("Identity", "c", "k"))),
            ComputeContextLifetimeCoverageTests.Op("ConcatFromSequence", "S", "O", attribute: ("axis", 0))), "m", 3)));
        Assert.Equal(512, RunPeak(ComputeContextLifetimeCoverageTests.WithInts(GraphOn("x:float[4,8]", "O",
            Loop("m", "O", GraphOn("i c", "k e", Op("Exp", "x", "e"), Op("Identity", "c", "k")))), "m", 2)));
        Assert.Null(RunPeak(ComputeContextLifetimeCoverageTests.WithInts(GraphOn("x:float[4,8]", "O",
            Loop("m x", "O", GraphOn("i c t", "k u", Op("Identity", "c", "k"), Op("Neg", "t", "u")))), "m", 50_000)));
        Assert.Equal(256, RunPeak(GraphOn("x:float[4,2,3] w:float[1,16,3] r:float[1,16,4]", "Y",
            ComputeContextLifetimeCoverageTests.Op("LSTM", "x w r", "Y", attribute: ("hidden_size", 4)))));
        var branch = ComputeContextLifetimeCoverageTests.Op("If", "c", "O", body: GraphOn("", "t", Op("Neg", "x", "t")));
        branch.Attributes.Add(new AttributeProto { Name = "else_branch", Type = AttributeProto.AttributeType.Graph, G = GraphOn("", "t", Op("Exp", "x", "t")) });
        Assert.Null(RunPeak(GraphOn("x:float[4,8] c:bool[1]", "O", branch)));
    }

    [Fact]
    public void TestATorchRunIsModelledHoldingTheResultOfEveryOperatorThatComputesOneOfItsOwn()
    {
        Assert.Equal(192, RunPeak(GraphOn("x:float[4,8] i:int64[2]", "O", Op("Exp", "x", "e"), Op("Gather", "e i", "O"))));
        Assert.Equal(384, RunPeak(GraphOn("x:float[4,8]", "O", Op("Exp", "x", "e"), ComputeContextLifetimeCoverageTests.Op("Cast", "e", "O", attribute: ("to", 7)))));
        Assert.Equal(256, RunPeak(GraphOn("x:float[4,8]", "O", Op("Exp", "x", "e"), ComputeContextLifetimeCoverageTests.Op("Cast", "e", "c", attribute: ("to", 1)), Op("Neg", "c", "O"))));
    }

    private static NodeProto Loop(string inputs, string outputs, GraphProto body) => ComputeContextLifetimeCoverageTests.Loop(inputs, outputs, body);

    [Fact]
    public void TestATorchRunIsModelledCopyingAReshapeOfAViewItsStridesCannotReshape()
    {
        Assert.Equal(4096, RunPeak(ComputeContextLifetimeCoverageTests.WithInts(GraphOn("x:float[4,8,16]", "O", Op("Exp", "x", "e"),
            ComputeContextLifetimeCoverageTests.With(Op("Transpose", "e", "t"), "perm", 1, 0, 2), Op("Reshape", "t shape", "O")), "shape", 8, 64)));
        Assert.Equal(2048, RunPeak(ComputeContextLifetimeCoverageTests.WithInts(GraphOn("x:float[4,8,16]", "O", Op("Exp", "x", "e"),
            ComputeContextLifetimeCoverageTests.With(Op("Transpose", "e", "t"), "perm", 1, 0, 2), Op("Reshape", "t shape", "O")), "shape", 8, 4, 4, 4)));
        Assert.Equal(2048, RunPeak(ComputeContextLifetimeCoverageTests.WithInts(GraphOn("x:float[4,8,16]", "O", Op("Exp", "x", "e"),
            Op("Reshape", "e shape", "O")), "shape", 32, 16)));
    }

    [Fact]
    public void TestATorchRunIsModelledFreeingEachValueAtItsLastReadWithItsViewsWritesOverDyingOperandsAndAliasedOutputsTakingNothing()
    {
        Assert.Equal(8192, RunPeak(GraphOn("a:float[1024]", "O", Op("Exp", "a", "e"), Op("Neg", "a", "f"), Op("Add", "e f", "O"))));
        Assert.Equal(4096, RunPeak(GraphOn("a:float[1024]", "O", Op("Exp", "a", "e"), Op("Neg", "e", "O"))));
        Assert.Equal(4096, RunPeak(GraphOn("a:float[32,32]", "O", Op("Exp", "a", "e"), Op("Transpose", "e", "O"))));
        Assert.Equal(0, RunPeak(GraphOn("a:float[1024] b:float[1024]", "O", Op("Add", "a b", "O")), new OutputAlias("O", "a")));
        Assert.Equal(4096, RunPeak(GraphOn("a:float[1024] b:float[1024]", "O", Op("Add", "a b", "O"))));
        Assert.Equal(263168, RunPeak(ComputeContextLifetimeCoverageTests.WithInts(GraphOn("a:float[256,256]", "O",
            Op("Exp", "a", "e"), Op("ReduceSum", "e axes", "s"), Op("Mul", "a s", "O")), "axes", 1)));
    }

    /// <summary>What <see cref="TorchRunMemory"/> models a run of <paramref name="graph"/> holding at
    /// once beyond its inputs, with <paramref name="aliases"/>' outputs written into their
    /// inputs.</summary>
    private static long? RunPeak(GraphProto graph, params OutputAlias[] aliases) => TorchRunMemory.Peak(new ModelProto { Graph = graph }, aliases);

    /// <summary><see cref="RunPeak(GraphProto, OutputAlias[])"/> on the host or a card.</summary>
    private static long? RunPeak(GraphProto graph, bool onHost) => TorchRunMemory.Peak(new ModelProto { Graph = graph }, [], onHost);

    /// <summary>Whether the training step a rig on a torch context runs holds no more at its peak
    /// than the step before the memory-aware pass would, each as the translation runs it: in its
    /// order, a view as its input's memory, and a node written over an operand dying there.</summary>
    private static bool PassHoldsNoMoreOnTorch(ComputationGraph model, long[] shape)
    {
        using var context = new ComputeContext(Torch);
        var sample = TensorData(shape, new float[shape.Aggregate(1L, (a, d) => a * d)]);
        var rig = TrainingRig.FromScratch(model, Shorokoo.Modules.Losses.L2Loss.ComputationGraph,
            Shorokoo.Modules.Optimizers.AdamWOptimizer.ComputationGraph, [sample],
            new Shorokoo.Modules.Optimizers.AdamWOptimizerHyperparameters { LearningRate = 0.001f }, runtimeContext: context);
        var state = rig.UpdatedParamFieldCount + rig.UpdatedStateFieldCount + rig.UpdatedOptimizerStateFieldCount;
        long Holds(ComputationGraph step)
        {
            var model = Benchmarks.MemoryPassBenchmarkTests.RigModel(step, rig.OptimizationInputShapes);
            var graph = model.Graph!;
            List<OutputAlias> written = [.. Enumerable.Range(0, state).Select(i => new OutputAlias(graph.Outputs[i].Name, graph.Inputs[i].Name))];
            return TorchRunMemory.Peak(model, OutputAliasProof.Prove(graph, written))!.Value;
        }
        return Holds(rig.TrainingStepPureGraph) <= Holds(rig.PreOptimizationGraph);
    }

    [Fact]
    public void TestAConvolutionOfTwoTransposedViewsIsComputedAsTheWeightGradientItIs()
    {
        Assert.Equal("True True", WeightGradient(batch: 4, sizes: 9, kernel: 3, stride: 1, dilation: 1, pad: 1));
        Assert.Equal("True True", WeightGradient(batch: 4, sizes: 10, kernel: 3, stride: 2, dilation: 1, pad: 1));
        Assert.Equal("True True", WeightGradient(batch: 2, sizes: 11, kernel: 3, stride: 2, dilation: 2, pad: 2));
        Assert.Equal("True True", WeightGradient(batch: 3, sizes: 7, kernel: 5, stride: 3, dilation: 1, pad: 0));
        Assert.Equal("True False", Evaluated("(lambda C, x: (str(C.conv(x, x).shape == (2, 2, 1, 1)) + ' ' + str(C._weight_gradient(x.transpose(0, 1).contiguous(), x, [1, 1], [0, 0], [1, 1], 1) is not None)))"
                                             + "(__import__('shorokoo_torch.ops_conv_pool', fromlist=['_']), torch.ones(2, 2, 3, 3))"));
    }

    /// <summary>Whether the translation's convolution of a convolution's input and output gradient,
    /// each with its batch and channel axes swapped, computes the convolution torch computes for
    /// them, and does it as torch's gradient of the convolution's weights.</summary>
    internal static string WeightGradient(int batch, int sizes, int kernel, int stride, int dilation, int pad, TorchBackend? backend = null)
        => Evaluated($$"""
            (lambda C, F, x, w: (lambda g: str((C.conv(x.transpose(0, 1), g.transpose(0, 1), strides=[{{dilation}}] * 2, dilations=[{{stride}}] * 2, pads=[{{pad}}] * 4)
                - F.conv2d(x.transpose(0, 1), g.transpose(0, 1), stride={{dilation}}, dilation={{stride}}, padding={{pad}})).abs().max().item() < 1e-12) + ' '
                + str(C._weight_gradient(x.transpose(0, 1), g.transpose(0, 1), [{{dilation}}] * 2, [{{pad}}] * 2, [{{stride}}] * 2, 1) is not None))(
                torch.randn(F.conv2d(x, w, stride={{stride}}, dilation={{dilation}}, padding={{pad}}).shape, dtype=torch.float64, device=x.device)))(
                __import__('shorokoo_torch.ops_conv_pool', fromlist=['_']), __import__('torch.nn.functional', fromlist=['_']),
                torch.randn({{batch}}, 3, {{sizes}}, {{sizes}}, dtype=torch.float64, device="{{(backend?.OnCuda == true ? "cuda" : "cpu")}}"),
                torch.randn(5, 3, {{kernel}}, {{kernel}}, dtype=torch.float64, device="{{(backend?.OnCuda == true ? "cuda" : "cpu")}}"))
            """.Replace("\n", " ").Replace("\r", " "), backend);

    [Fact]
    public void TestAnElementWiseNodeIsWrittenOverAnOperandNothingReadsAfterItAnInputOrAValueOfItsOwnMemory()
    {
        Assert.Equal("over:neg over:exp", Calls(GraphOn("x:float[4]", "c", Op("Neg", "x", "b"), Op("Exp", "b", "c")), over: true));
        Assert.Equal("over:exp", Calls(GraphOn("x:float[4]", "c", Op("Exp", "x", "c")), over: true));
        Assert.Equal("over:neg abs_ over:exp", Calls(GraphOn("x:float[4]", "c d", Op("Neg", "x", "b"), Op("Exp", "b", "c"), Op("Abs", "b", "d")), over: true));
        Assert.Equal("over:neg exp", Calls(GraphOn("x:float[4]", "b c", Op("Neg", "x", "b"), Op("Exp", "b", "c")), over: true));
        Assert.Equal("over:neg over:exp shape", Calls(GraphOn("x:float[4]", "c s", Op("Neg", "x", "b"), Op("Exp", "b", "c"), Op("Shape", "c", "s")), over: true));
        Assert.Equal("neg over:add over:mul", Calls(GraphOn("x:float[4]", "d", Op("Neg", "x", "b"), Op("Add", "x b", "c"), Op("Mul", "c c", "d")), over: true));
        Assert.Equal("over:neg over:softmax over:layer_normalization", Calls(GraphOn("x:float[4] s:float[4]", "d", Op("Neg", "x", "b"), Op("Softmax", "b", "c"), Op("LayerNormalization", "c s", "d")), over: true));
        Assert.Equal("over:neg identity abs_ over:exp", Calls(GraphOn("x:float[4]", "c d", Op("Neg", "x", "b"), Op("Identity", "b", "r"), Op("Exp", "b", "c"), Op("Abs", "r", "d")), over: true));
        Assert.Equal("over:neg transpose mul", Calls(GraphOn("x:float[2,2]", "c", Op("Neg", "x", "b"), Op("Transpose", "b", "t"), Op("Mul", "b t", "c")), over: true));
        Assert.Equal("neg over:add", Calls(GraphOn("x:float[4]", "c", Op("Neg", "x", "b"), Op("Add", "b x", "c")), over: true));
        Assert.Equal("over:neg greater over:where", Calls(GraphOn("x:float[4] y:float[4]", "c", Op("Neg", "x", "b"), Op("Greater", "b y", "m"), Op("Where", "m b y", "c")), over: true));
        Assert.Equal("over:neg greater where over:add", Calls(GraphOn("x:float[4]", "d", Op("Neg", "x", "b"), Op("Greater", "b b", "m"), Op("Where", "m b b", "c"), Op("Add", "c b", "d")), over: true));
    }

    [Fact]
    public void TestARunWritesSoftmaxesNormalizationsClipsConvolutionsAndGemmsIntoTheMemoryItConsumesComputingWhatAPlainRunComputes()
        => SoftmaxesNormalizationsClipsConvolutionsAndGemmsArePlaced(Torch);

    internal static void SoftmaxesNormalizationsClipsConvolutionsAndGemmsArePlaced(TorchBackend backend)
    {
        string Placed(GraphProto graph) => PlacedOn(backend, graph);
        Assert.Equal("O@b+0", Placed(Graph("O", Op("Softmax", "a", "O"))));
        Assert.Equal("O@b+0", Placed(Graph("O", Op("LogSoftmax", "a", "O"))));
        Assert.Equal("O@b+0", Placed(GraphOn("a:float[512,512] b:float[262144]", "O", ComputeContextLifetimeCoverageTests.Op("Softmax", "a", "O", attribute: ("axis", 0)))));
        Assert.Equal("O@a+0", Placed(Graph("O", Op("Gelu", "a", "O"))));
        Assert.Equal("O@a+0", Placed(GraphOn("a:float[262144] lo:float[1] hi:float[1]", "O", Op("Clip", "a lo hi", "O"))));
        Assert.Equal("O@a+0", Placed(GraphOn("a:float[262144] hi:float[1]", "O", Node("Clip", ["a", "", "hi"], ["O"]))));
        Assert.Equal("O@b+0", Placed(GraphOn("a:float[512,512] b:float[512,512] s:float[512] c:float[512]", "O", Op("LayerNormalization", "a s c", "O"))));
        Assert.Equal("O@b+0", Placed(GraphOn("x:float[4,64,32,32] b:float[4,64,32,32] s:float[64] c:float[64] m:float[64] v:float[64]", "O",
            Op("Abs", "v", "w"), Op("BatchNormalization", "x s c m w", "O"))));
        Assert.Equal("O@b+0", Placed(GraphOn("x:float[2,32,64,64] b:float[2,32,64,64] w:float[32,32,3,3]", "O",
            ComputeContextLifetimeCoverageTests.With(Op("Conv", "x w", "O"), "pads", 1, 1, 1, 1))));
        Assert.Equal("O@b+0", Placed(GraphOn("a:float[512,512] b:float[512,512] w:float[512,512] c:float[512]", "O", Op("Gemm", "a w c", "O"))));
    }

    [Fact]
    public void TestAPlacedValueGivenNoRangeIsComputedAsItIsAndCopiedOutOfItsOperandsWhereItWasProvedOutOfThem()
    {
        Assert.Equal("False", Evaluated("(lambda t: __import__('shorokoo_torch.runtime', fromlist=['_']).place_into(0, True, S.identity, t).data_ptr() == t.data_ptr())(torch.ones(4))"));
        Assert.Equal("True", Evaluated("(lambda t: __import__('shorokoo_torch.runtime', fromlist=['_']).place_into(0, False, S.identity, t).data_ptr() == t.data_ptr())(torch.ones(4))"));
        Assert.Equal("[-1.0, -1.0] torch.float32", Evaluated("__import__('shorokoo_torch.runtime', fromlist=['_']).place_into(0, True, E.neg, torch.ones(2))"));
    }

    [Fact]
    public void TestARunWritesIntoAConsumedInputOnlyWhereItsBytesAreItsOwnContiguousAndOnTheRunsDevice()
    {
        Assert.Equal("0", Blocks("[t, t[2:]]"));
        Assert.Equal("1", Blocks("[t[:2], t[2:]]"));
        Assert.Equal("0", Blocks("[t, t]"));
        Assert.Equal("0", Blocks("[t.reshape(2, 2).t()]"));
        Assert.Equal("0", Blocks("[t]", moved: "[t.clone()]"));
        Assert.Equal("0", Blocks("[t]", constants: "[t.untyped_storage().data_ptr()]"));
        Assert.Equal("1", Blocks("[t]"));
    }

    /// <summary>How many of the inputs a run is handed may take placed values, of the first:
    /// <paramref name="args"/> are the run's arguments, over a tensor <c>t</c> of four floats, and
    /// <paramref name="moved"/> the values it computes on, the arguments themselves by default.</summary>
    private static string Blocks(string args, string moved = "a", string constants = "[]")
        => Evaluated($"(lambda rt, t: (lambda a: len(rt._Placing(a, {moved}, [(0, 0, 8, 1, [2], -1)], torch.device('cpu'), {constants}).blocks))({args}))"
                     + "(__import__('shorokoo_torch.runtime', fromlist=['_']), torch.zeros(4))");

    private static GraphProto Graph(string outputs, params NodeProto[] nodes) => GraphOn("a:float[262144] b:float[262144]", outputs, nodes);

    private static GraphProto GraphOn(string inputs, string outputs, params NodeProto[] nodes) => ComputeContextLifetimeCoverageTests.GraphOf(inputs, outputs, nodes);

    private static GraphProto Halved(string outputs, params NodeProto[] nodes)
        => ComputeContextLifetimeCoverageTests.WithInts(ComputeContextLifetimeCoverageTests.WithInts(ComputeContextLifetimeCoverageTests.WithInts(
            GraphOn("a:float[524288] b:float[524288]", outputs, nodes), "zero", 0), "half", 262144), "end", 524288);

    /// <summary>
    /// Where each output of <paramref name="graph"/> stands after a run that consumes every input —
    /// <c>O@a+16</c> for 16 bytes into input a's memory, <c>O@-</c> for memory of its own — once the
    /// run is shown to compute what a run consuming nothing computes.
    /// </summary>
    private static string Placed(GraphProto graph, bool consume = true, bool feedTwice = false) => PlacedOn(Torch, graph, consume, feedTwice);

    /// <summary>
    /// <see cref="Placed"/> on <paramref name="backend"/>; on a card, with <c> allocating</c> after
    /// it where the consuming run did not allocate at least its outputs' bytes less there than the
    /// run consuming nothing.
    /// </summary>
    internal static string PlacedOn(TorchBackend backend, GraphProto graph, bool consume = true, bool feedTwice = false)
    {
        using var session = backend.CreateSession(Serialize(graph), default, default, DeviceMemorySettings.Default, DiagnosticSettings.Default, []);
        var names = graph.Outputs.Select(o => o.Name).ToArray();
        (string[] Values, string Where, long Allocated, long Bytes) Run(bool consuming)
        {
            var feeds = new Dictionary<string, IShorokooTensorValue>(StringComparer.Ordinal);
            foreach (var input in graph.Inputs.Where(i => i.Type?.TensorType is not null))
                feeds[input.Name] = feedTwice && feeds.Count > 0 ? feeds.Values.First() : Pattern(backend, input, feeds.Count);
            var fed = feeds.Values.Distinct().Cast<TorchTensorValue>().ToArray();
            var at = feeds.ToDictionary(f => f.Key, f => (((TorchTensorValue)f.Value).Address, TorchPlacements.BytesOf((TorchTensorValue)f.Value)));
            var fedBytes = fed.Select(value => Convert.ToBase64String(backend.CopyTensorToHost(value))).ToArray();
            var before = CardAllocated(backend);
            var results = session.RunConsuming(feeds, consuming ? fed : [], names, RunSettings.Default);
            var allocated = CardAllocated(backend) - before;
            if (!consuming)
            {
                Assert.Equal(fedBytes, fed.Select(value => Convert.ToBase64String(backend.CopyTensorToHost(value))));
                foreach (var value in fed) value.Dispose();
            }
            var where = names.Select((name, i) => ((TorchTensorValue)results[i]).Range is null
                ? $"{name}@-"
                : at.Where(f => ((TorchTensorValue)results[i]).Address >= f.Value.Address && ((TorchTensorValue)results[i]).Address < f.Value.Address + (nint)f.Value.Item2)
                    .Select(f => $"{name}@{f.Key}+{((TorchTensorValue)results[i]).Address - f.Value.Address}").Single());
            string[] values = [.. results.Select(r => Convert.ToBase64String(backend.CopyTensorToHost(r)))];
            var bytes = results.Sum(r => TorchPlacements.BytesOf((TorchTensorValue)r));
            foreach (var result in results) result.Dispose();
            return (values, string.Join(" ", where), allocated, bytes);
        }
        var plain = Run(consuming: false);
        var placed = Run(consume);
        Assert.Equal(plain.Values, placed.Values);
        return placed.Where + (backend.OnCuda && placed.Allocated + placed.Bytes > plain.Allocated ? " allocating" : "");
    }

    /// <summary>What <paramref name="run"/> answers, and the most torch's CUDA allocator held at once
    /// beyond what it held as the run began.</summary>
    internal static (long Peak, T Result) CardPeak<T>(Func<T> run)
    {
        using (PythonRuntime.Gil())
        {
            using var scope = Py.CreateScope();
            scope.Exec("import torch\ntorch.cuda.synchronize()\ntorch.cuda.reset_peak_memory_stats()\nbefore = torch.cuda.memory_allocated()");
            var result = run();
            scope.Exec("torch.cuda.synchronize()\npeak = torch.cuda.max_memory_allocated() - before");
            return (scope.Get<long>("peak"), result);
        }
    }

    /// <summary>Every byte torch's CUDA allocator has handed out in this process, or 0 off a card.</summary>
    private static long CardAllocated(TorchBackend backend)
    {
        if (!backend.OnCuda) return 0;
        using (PythonRuntime.Gil())
        {
            using var scope = Py.CreateScope();
            scope.Exec("import torch\nallocated = torch.cuda.memory_stats().get('allocated_bytes.all.allocated', 0)");
            return scope.Get<long>("allocated");
        }
    }

    private static IShorokooTensorValue Pattern(TorchBackend backend, ValueInfoProto input, int seed)
    {
        var shape = input.Type.TensorType.Shape.Dims.Select(d => d.DimValue).ToArray();
        var count = (int)shape.Aggregate(1L, (a, d) => a * d);
        return input.Type.TensorType.ElemType == (int)TensorProto.DataType.Int64
            ? backend.CreateTensorInBackendMemory(ShorokooTensorElementType.Int64,
                [.. MemoryMarshal.AsBytes<long>([.. Enumerable.Range(0, count).Select(i => (long)((i * 7 + seed) % 1000))])], shape)
            : backend.CreateTensorInBackendMemory(ShorokooTensorElementType.Float,
                [.. MemoryMarshal.AsBytes<float>([.. Enumerable.Range(0, count).Select(i => ((i * 7 + seed) % 1000) * 0.001f - 0.4f)])], shape);
    }

    [Fact]
    public void TestAnInitializerWhoseRawDataIsNotItsShapesSizeIsRefusedAtSessionCreation()
    {
        Assert.Throws<TorchUnsupportedModelException>(() => Torch.CreateSession(RawInitialized(4), default, default, DeviceMemorySettings.Default));
        Assert.Throws<TorchUnsupportedModelException>(() => Torch.CreateSession(RawInitialized(16), default, default, DeviceMemorySettings.Default));
        Assert.Throws<TorchUnsupportedModelException>(() => Torch.CreateSession(RawInitialized(0), default, default, DeviceMemorySettings.Default));
        Torch.CreateSession(RawInitialized(12), default, default, DeviceMemorySettings.Default).Dispose();
    }

    [Fact]
    public void TestProvisioningInstallsTheLockByHashIntoItsOwnEnvironmentAndGivesUpOnAUvThatHangs()
    {
        var installed = FakeUv("""
            echo "$@" >> LOG
            if [ "$1" = venv ]; then for last; do :; done; mkdir -p "$last"; fi
            """, """
            echo %*>>"LOG"
            if not "%~1"=="venv" exit /b 0
            for %%a in (%*) do set "last=%%~a"
            mkdir "%last%"
            """);
        var hung = FakeUv("sleep 20; exit 1", "ping -n 21 127.0.0.1 >nul & exit /b 1", TimeSpan.FromSeconds(2));
        var install = installed.Log.Single(line => line.StartsWith("pip install", StringComparison.Ordinal)).Replace("\"", "");

        Assert.Equal(PythonEnvironmentFailure.NotAVirtualEnvironment, installed.Failure);
        Assert.Contains($"--python {installed.Directory}", install);
        Assert.Contains("--require-hashes", install);
        Assert.Equal(PythonEnvironmentFailure.ProvisioningTimedOut, hung.Failure);
        Assert.True(hung.Took < TimeSpan.FromSeconds(15));
        Assert.Equal(["HOME", "HTTPS_PROXY", "SSL_CERT_FILE", "UV_CACHE_DIR", "UV_NATIVE_TLS", "UV_NO_PROGRESS"],
            PythonEnvironmentResolver.UvEnvironment(((string[])["HOME", "HTTPS_PROXY", "SSL_CERT_FILE", "UV_CACHE_DIR", "UV_NATIVE_TLS", "UV_SYSTEM_PYTHON",
                "UV_EXTRA_INDEX_URL", "UV_INDEX_STRATEGY", "UV_PYTHON", "VIRTUAL_ENV", "CONDA_PREFIX", "PYTHONPATH"]).Select(name => KeyValuePair.Create(name, "x"))).Keys.Order());
    }

    [Fact]
    public void TestProvisioningLinksTheCudaLibrariesWithinWhatIsLeftOfItsTimeout()
    {
        var cudnn = CudaLibraryPins.Current!.Libraries.Single(pin => pin.Name == "cudnn");
        var copy = Path.Combine(PythonEnvironment.SitePackagesOf("", System.Version.Parse(PythonEnvironmentLock.Cpu.PythonVersion), OperatingSystem.IsWindows()),
            "torch", "lib", cudnn.Files[^1].FileName);
        var linking = FakeUv($"""
            if [ "$1" = venv ]; then for last; do :; done; mkdir -p "$(dirname "$last/{copy}")"; echo x > "$last/{copy}"; fi
            """, $"""
            if not "%~1"=="venv" exit /b 0
            for %%a in (%*) do set "last=%%~a"
            mkdir "%last%\{Path.GetDirectoryName(copy)}"
            echo x>"%last%\{copy}"
            """, TimeSpan.FromSeconds(3), HoldingTheCudnnCacheAndForAWhileTheEnvironment);

        Assert.Equal(PythonEnvironmentFailure.NotAVirtualEnvironment, linking.Failure);
        Assert.True(linking.Took < TimeSpan.FromSeconds(4));
    }

    [Fact]
    public void TestAProcessWithoutUvWaitsForAnotherThatIsProvisioningAndUsesWhatItProvisioned()
    {
        var root = Directory.CreateDirectory(Path.Combine(Path.GetTempPath(), "shorokoo-wait-" + Guid.NewGuid().ToString("N"))).FullName;
        var directory = Path.Combine(root, PythonEnvironmentLock.Cpu.CacheKey);
        try
        {
            Task<PythonEnvironment> waiting;
            using (new FileStream(directory + ".lock", FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None))
            {
                waiting = Task.Run(() => PythonEnvironmentResolver.Resolve(PythonEnvironmentLock.Cpu, new() { CacheDirectory = root, UvPath = Path.Combine(root, "uv") }, _ => null));
                Thread.Sleep(500);
                Directory.CreateDirectory(directory);
                File.Copy(Path.Combine(Torch.Start().Directory, "pyvenv.cfg"), Path.Combine(directory, "pyvenv.cfg"));
                File.WriteAllText(Path.Combine(directory, ".shorokoo-provisioned"), PythonEnvironmentLock.Cpu.Hash);
            }

            Assert.Equal(PythonEnvironmentSource.Provisioned, waiting.Result.Source);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void TestABackendOnAPlatformWithoutALockIsConstructedAndRefusesToStartWithATypedFailure()
    {
        Assert.Equal(PythonEnvironmentFailure.UnsupportedPlatform, Assert.Throws<PythonEnvironmentException>(() => new OffPlatformBackend().Start()).Failure);
    }

    private sealed class OffPlatformBackend() : TorchBackend(() => PythonEnvironmentLock.ForPlatform("cpu", "osx-arm64"), null, null);

    [Fact]
    public void TestAFunctionCallOmittingTrailingInputsPassesNoneAndOneWithMoreInputsOrOutputsThanTheFunctionIsRefused()
    {
        var first = Function("First", ["a", "b"], ["y"], "", Node("Neg", ["a"], ["y"]));
        using var session = Torch.CreateSession(Serialize(Graph(["x"], ["y"], Node("First", ["x"], ["y"], "Functions")), first), default, default, DeviceMemorySettings.Default);

        Assert.Equal([-1f, -2f], RunFloats(session, new() { ["x"] = [1f, 2f] }, ["y"])[0]);
        Assert.Throws<TorchUnsupportedModelException>(() => Torch.CreateSession(Serialize(Graph(["x"], ["y"], Node("First", ["x", "x", "x"], ["y"], "Functions")), first), default, default, DeviceMemorySettings.Default));
        Assert.Throws<TorchUnsupportedModelException>(() => Torch.CreateSession(Serialize(Graph(["x"], ["y"], Node("First", ["x"], ["y", "z"], "Functions")), first), default, default, DeviceMemorySettings.Default));
    }

    [Fact]
    public void TestAFunctionCallRunsAndDifferentiatesTheOverloadItNames()
    {
        FunctionProto[] overloads = [Function("F", ["a"], ["y"], "neg", Node("Neg", ["a"], ["y"])), Function("F", ["a"], ["y"], "abs", Node("Abs", ["a"], ["y"]))];
        var call = Node("F", ["x"], ["y"], "Functions");
        call.Overload = "neg";
        var stepCall = Node("F", ["w"], ["t"], "Functions");
        stepCall.Overload = "neg";
        using var session = Torch.CreateSession(Serialize(Graph(["x"], ["y"], call), overloads), default, default, DeviceMemorySettings.Default);
        using var step = Torch.CreateSession(TrainingStep(Graph(["w"], ["g"], stepCall, Node("ReduceSum", ["t"], ["loss"]), AutoGrad(["loss", "w"], ["g"])), 1, overloads), default, default, DeviceMemorySettings.Default);

        Assert.Equal([-1f, 2f], RunFloats(session, new() { ["x"] = [1f, -2f] }, ["y"])[0]);
        Assert.Equal([-1f, -1f], RunFloats(step, new() { ["w"] = [1f, -2f] }, ["g"])[0]);
        Assert.Throws<TorchUnsupportedModelException>(() => Torch.CreateSession(Serialize(Graph(["x"], ["y"], call), overloads[0], overloads[0]), default, default, DeviceMemorySettings.Default));
    }

    [Fact]
    public void TestALoopThatRunsNoIterationHandsOutEachScanOutputEmptyOfItsBodysDeclaredTypeAndShapeAndOneOfNoTypeIsRefused()
    {
        using var session = Torch.CreateSession(Serialize(ScanLoop(typed: true)), default, default, DeviceMemorySettings.Default);
        using var m = Torch.CreateTensor([0L], []);
        using var v = Torch.CreateTensor([0f], []);
        using var scanned = session.Run(new Dictionary<string, IShorokooTensorValue> { ["m"] = m, ["v"] = v }, ["s"], RunSettings.Default)[0];

        Assert.Equal("Int64 0,2,3", $"{scanned.ElementType} {string.Join(",", scanned.Shape)}");
        Assert.Throws<TorchUnsupportedModelException>(() => Torch.CreateSession(Serialize(ScanLoop(typed: false)), default, default, DeviceMemorySettings.Default));
    }

    [Fact]
    public void TestAnAttributeWithNeitherATypeNorAValueOrWithTextThatIsNotUtf8IsRefusedAtSessionCreation()
    {
        Assert.Throws<TorchUnsupportedModelException>(() => Torch.CreateSession(Typed(11, ["x"], ["y"], [Node("ReduceSum", ["x"], ["y"], attributes: new AttributeProto { Name = "axes" })]), default, default, DeviceMemorySettings.Default));
        Assert.Throws<TorchUnsupportedModelException>(() => Torch.CreateSession(Typed(21, ["x"], ["y"], [Node("Gelu", ["x"], ["y"], attributes: new AttributeProto { Name = "approximate", Type = AttributeProto.AttributeType.String, S = [0x74, 0xFF] })]), default, default, DeviceMemorySettings.Default));
        Assert.Throws<TorchUnsupportedModelException>(() => Torch.CreateSession(Typed(21, ["x"], ["y"], [Node("Constant", [], ["y"], attributes: new AttributeProto { Name = "value_string", Type = AttributeProto.AttributeType.String, S = [0xC3] })]), default, default, DeviceMemorySettings.Default));
    }

    private static PythonEnvironment Resolve(PythonEnvironmentOptions options, string variable)
        => PythonEnvironmentResolver.Resolve(PythonEnvironmentLock.Cpu, options,
            name => name == PythonEnvironmentResolver.EnvironmentVariable ? variable : null);

    private static PythonEnvironmentFailure Failure(PythonEnvironmentOptions options)
        => Assert.Throws<PythonEnvironmentException>(
            () => PythonEnvironmentResolver.Resolve(PythonEnvironmentLock.Cpu, options, _ => null)).Failure;

    internal static int ElementSize(ShorokooTensorElementType type) => type switch
    {
        ShorokooTensorElementType.Complex64 => 8,
        ShorokooTensorElementType.Complex128 => 16,
        >= ShorokooTensorElementType.Float8E4M3FN and <= ShorokooTensorElementType.Float8E5M2FNUZ => 1,
        _ => TensorElementLayout.ElementSizeInBytes(type),
    };

    private static Dictionary<string, IShorokooTensorValue> Feeds(IShorokooTensorValue x) => new() { ["x0"] = x };

    private static int ReleasesOf(Action<IShorokooTensorValue> run)
    {
        var feed = new CountingValue();
        try { run(feed); }
        catch (Exception) { }
        return feed.Disposals;
    }

    private sealed class CountingValue : IShorokooTensorValue
    {
        private readonly float[] _data = [1f, 2f];
        public int Disposals;
        public ShorokooOnnxValueType ValueType => ShorokooOnnxValueType.Tensor;
        public ShorokooTensorElementType ElementType => ShorokooTensorElementType.Float;
        public long[] Shape => [2];
        public ReadOnlySpan<T> GetTensorDataAsSpan<T>() where T : unmanaged => MemoryMarshal.Cast<float, T>(_data);
        public Span<T> GetTensorMutableDataAsSpan<T>() where T : unmanaged => MemoryMarshal.Cast<float, T>(_data.AsSpan());
        public IReadOnlyList<string> GetStringTensorData() => throw new NotSupportedException();
        public int GetValueCount() => throw new NotSupportedException();
        public IShorokooTensorValue GetValue(int index) => throw new NotSupportedException();
        public ShorokooTensorElementType GetSequenceElementType() => throw new NotSupportedException();
        public void Dispose() => Disposals++;
    }

    private static NodeProto Op(string op, string inputs, string outputs) => ComputeContextLifetimeCoverageTests.Op(op, inputs, outputs);

    private static IShorokooSession Aliasing(GraphProto graph)
        => Torch.CreateSession(Serialize(graph), default, default, DeviceMemorySettings.Default, DiagnosticSettings.Default, [new OutputAlias("O", "a")]);

    private static (string? Input, string Outputs) Aliased(
        string inputs, string outputs, NodeProto[] nodes, float[]? b = null, bool consume = true, bool feedTwice = false, bool integers = false)
    {
        var graph = ComputeContextLifetimeCoverageTests.GraphOf(inputs, outputs, nodes);
        using var session = Aliasing(graph);
        IShorokooTensorValue Tensor(float[] values)
            => integers ? Torch.CreateTensor([.. values.Select(v => (long)v)], [values.Length]) : Torch.CreateTensor(values, [values.Length]);
        var a = Tensor([10f, 20f, 30f, 40f]);
        using var second = Tensor(b ?? [1f, 2f, 3f, 4f]);
        var feeds = new Dictionary<string, IShorokooTensorValue> { ["a"] = a, ["b"] = feedTwice ? a : second };
        var results = session.RunConsuming(feeds, consume ? [a] : [], [.. graph.Outputs.Select(o => o.Name)], RunSettings.Default, out var aliased);
        if (!consume) a.Dispose();
        float[] values = [.. results.SelectMany(r => integers ? r.GetTensorDataAsSpan<long>().ToArray().Select(v => (float)v) : r.GetTensorDataAsSpan<float>().ToArray())];
        foreach (var result in results) result.Dispose();
        return (aliased.Count == 0 ? null : aliased[0], string.Join(" ", values));
    }

    private static string AliasedRun(IShorokooSession session, GraphProto graph)
    {
        var feeds = graph.Inputs.ToDictionary(i => i.Name, i => Torch.CreateTensor(Enumerable.Repeat(i.Name switch { "a" => 1f, "b" => 10f, "c" => 100f, _ => 1f }, 3).ToArray(), [3]));
        using var kept = feeds["g"];
        var results = session.RunConsuming(feeds.ToDictionary(f => f.Key, f => (IShorokooTensorValue)f.Value), [.. feeds.Where(f => f.Key != "g").Select(f => f.Value)],
            [.. graph.Outputs.Select(o => o.Name)], RunSettings.Default, out var aliased);
        var values = string.Join(" ", results.SelectMany(r => r.GetTensorDataAsSpan<float>().ToArray()));
        foreach (var result in results) result.Dispose();
        return string.Join(",", results.Select((_, i) => aliased.Count == 0 ? "-" : aliased[i] ?? "-")) + " " + values;
    }

    private static byte[] RawInitialized(int bytes)
    {
        var graph = ComputeContextLifetimeCoverageTests.GraphOf("x:float[3]", "y:float[3]", Op("Add", "x w", "y"));
        graph.Initializers.Add(new TensorProto { Name = "w", data_type = (int)TensorProto.DataType.Float, Dims = [3], RawData = new byte[bytes] });
        return Serialize(graph);
    }

    private static (PythonEnvironmentFailure Failure, string[] Log, string Directory, TimeSpan Took) FakeUv(
        string sh, string cmd, TimeSpan? timeout = null, Func<string, IDisposable?>? holding = null)
    {
        var root = Directory.CreateDirectory(Path.Combine(Path.GetTempPath(), "shorokoo-uv-" + Guid.NewGuid().ToString("N"))).FullName;
        var (uv, log) = (Path.Combine(root, OperatingSystem.IsWindows() ? "uv.cmd" : "uv"), Path.Combine(root, "log"));
        File.WriteAllText(uv, OperatingSystem.IsWindows()
            ? "@echo off\r\n" + cmd.Replace("LOG", log).ReplaceLineEndings("\r\n") + "\r\n"
            : "#!/bin/sh\n" + sh.Replace("LOG", log) + "\n");
        if (!OperatingSystem.IsWindows()) File.SetUnixFileMode(uv, UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute);
        var held = holding?.Invoke(root);
        var clock = System.Diagnostics.Stopwatch.StartNew();
        try
        {
            var failure = Assert.Throws<PythonEnvironmentException>(() => PythonEnvironmentResolver.Resolve(PythonEnvironmentLock.Cpu,
                new() { CacheDirectory = root, UvPath = uv, ProvisioningTimeout = timeout ?? TimeSpan.FromMinutes(1) },
                name => name is "LOCALAPPDATA" or "XDG_CACHE_HOME" ? root : null)).Failure;
            return (failure, File.Exists(log) ? File.ReadAllLines(log) : [], Path.Combine(root, PythonEnvironmentLock.Cpu.CacheKey), clock.Elapsed);
        }
        finally
        {
            held?.Dispose();
            Directory.Delete(root, recursive: true);
        }
    }

    /// <summary>The lock of the CUDA library cache's cuDNN folder under <paramref name="root"/>, held
    /// until the result is disposed, and the lock of the environment provisioned there, held for a
    /// second and a half.</summary>
    private static IDisposable HoldingTheCudnnCacheAndForAWhileTheEnvironment(string root)
    {
        var cudnn = CudaLibraryPins.Current!.Libraries.Single(pin => pin.Name == "cudnn");
        var cuda = Directory.CreateDirectory(Path.Combine(root, "shorokoo", "cuda")).FullName;
        var cache = new FileStream(Path.Combine(cuda, cudnn.CacheKey + ".lock"), FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None);
        var environment = new FileStream(Path.Combine(root, PythonEnvironmentLock.Cpu.CacheKey + ".lock"), FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None);
        _ = Task.Delay(1500).ContinueWith(_ => environment.Dispose());
        return cache;
    }

    private static FunctionProto Function(string name, string[] inputs, string[] outputs, string overload, params NodeProto[] nodes)
    {
        var function = new FunctionProto { Name = name, Domain = "Functions", Overload = overload };
        function.Inputs.AddRange(inputs);
        function.Outputs.AddRange(outputs);
        function.Nodes.AddRange(nodes);
        function.OpsetImports.Add(new OperatorSetIdProto { Domain = "", Version = 21 });
        return function;
    }

    internal static GraphProto ScanLoop(bool typed)
    {
        var body = Graph(["i", "c", "x"], ["c2", "x2", "sc"], Node("Identity", ["c"], ["c2"]), Node("Identity", ["x"], ["x2"]),
            Node("Constant", [], ["sc"], attributes: Tensor("value", 7, [2, 3], [1, 2, 3, 4, 5, 6])));
        if (typed)
            body.Outputs[2].Type = new TypeProto { TensorType = new TypeProto.Tensor { ElemType = 7, Shape = new TensorShapeProto { Dims = { new() { DimValue = 2 }, new() { DimValue = 3 } } } } };
        return Graph(["m", "v"], ["y", "s"], Node("Loop", ["m", "", "v"], ["y", "s"], attributes: new AttributeProto { Name = "body", Type = AttributeProto.AttributeType.Graph, G = body }));
    }

    /// <summary>y = v + 1, m times over, by a Loop: one node per iteration to stop at.</summary>
    internal static GraphProto CountingLoop()
    {
        var one = new TensorProto { Name = "one", data_type = (int)TensorProto.DataType.Float, FloatDatas = [1f], Dims = [] };
        var body = Graph(["i", "c", "x"], ["c2", "x2"], Node("Identity", ["c"], ["c2"]), Node("Add", ["x", "one"], ["x2"]));
        body.Initializers.Add(one);
        var loop = Node("Loop", ["m", "", "v"], ["y"]);
        loop.Attributes.Add(new AttributeProto { Name = "body", Type = AttributeProto.AttributeType.Graph, G = body });
        return Graph(["m", "v"], ["y"], loop);
    }

    internal static float[] RunFloats(IShorokooSession session, bool condition, float[] x)
    {
        using var c = Torch.CreateTensor([condition], []);
        using var input = Torch.CreateTensor(x, [x.Length]);
        using var y = session.Run(new Dictionary<string, IShorokooTensorValue> { ["c"] = c, ["x"] = input }, ["y"], RunSettings.Default)[0];
        return y.GetTensorDataAsSpan<float>().ToArray();
    }

    internal static NodeProto Node(string opType, string[] inputs, string[] outputs, string domain = "", params AttributeProto[] attributes)
    {
        var node = new NodeProto { OpType = opType, Name = $"{opType}:{string.Join(",", outputs)}", Domain = domain };
        node.Inputs.AddRange(inputs);
        node.Outputs.AddRange(outputs);
        node.Attributes.AddRange(attributes);
        return node;
    }

    private static readonly (string, float[], long[]) X23 = ("x", [3f, 1f, 3f, 2f, 5f, 1f], [2, 3]);

    internal static TypeProto FloatTensor => new() { TensorType = new TypeProto.Tensor { ElemType = 1 } };

    internal static AttributeProto Int(string name, long value) => new() { Name = name, Type = AttributeProto.AttributeType.Int, I = value };

    internal static AttributeProto Str(string name, string value)
        => new() { Name = name, Type = AttributeProto.AttributeType.String, S = System.Text.Encoding.UTF8.GetBytes(value) };

    internal static AttributeProto Tensor(string name, int elementType, long[] dims, long[] values)
        => new() { Name = name, Type = AttributeProto.AttributeType.Tensor, T = new TensorProto { data_type = elementType, Dims = dims, Int64Datas = values } };

    internal static byte[] Typed(int opset, string[] inputs, string[] outputs, NodeProto[] nodes, params FunctionProto[] functions)
    {
        var graph = Graph(inputs, outputs, nodes);
        foreach (var input in graph.Inputs) input.Type = FloatTensor;
        var model = new ModelProto { IrVersion = 10, Graph = graph };
        model.OpsetImports.Add(new OperatorSetIdProto { Domain = "", Version = opset });
        model.OpsetImports.Add(new OperatorSetIdProto { Domain = "Functions", Version = 1 });
        model.Functions.AddRange(functions);
        using var stream = new MemoryStream();
        ProtoBuf.Serializer.Serialize(stream, model);
        return stream.ToArray();
    }

    private static bool AgreesWithOrt(byte[] model, params (string Name, float[] Data, long[] Shape)[] feeds)
    {
        var outputs = ProtoBuf.Serializer.Deserialize<ModelProto>(new MemoryStream(model)).Graph.Outputs.Select(o => o.Name).ToArray();
        var onOrt = Run(DefaultBackend.Instance, model, feeds, outputs);
        var onTorch = Run(Torch, model, feeds, outputs);
        return onOrt.Zip(onTorch).All(p => p.First.Type == p.Second.Type && p.First.Shape == p.Second.Shape
            && p.First.Values.Zip(p.Second.Values).All(v => Math.Abs(v.First - v.Second) <= AutoTest.Tolerance * Math.Max(1.0, Math.Abs(v.First))));
    }

    private static (ShorokooTensorElementType Type, string Shape, double[] Values)[] Run(
        IShorokooBackend backend, byte[] model, (string Name, float[] Data, long[] Shape)[] feeds, string[] outputs)
    {
        using var session = backend.CreateSession(model, default, default, DeviceMemorySettings.Default);
        var inputs = feeds.ToDictionary(f => f.Name, f => backend.CreateTensor(f.Data, f.Shape));
        var values = session.Run(inputs, outputs, RunSettings.Default);
        (ShorokooTensorElementType, string, double[])[] results = [.. values.Select(v => (v.ElementType, string.Join(",", v.Shape),
            v.ElementType == ShorokooTensorElementType.Float
                ? v.GetTensorDataAsSpan<float>().ToArray().Select(f => (double)f).ToArray()
                : v.GetTensorDataAsSpan<long>().ToArray().Select(i => (double)i).ToArray()))];
        foreach (var value in values.Concat(inputs.Values)) value.Dispose();
        return results;
    }

    internal static GraphProto Graph(string[] inputs, string[] outputs, params NodeProto[] nodes)
    {
        var graph = new GraphProto { Name = "g" };
        graph.Inputs.AddRange(inputs.Select(name => new ValueInfoProto { Name = name }));
        graph.Outputs.AddRange(outputs.Select(name => new ValueInfoProto { Name = name }));
        graph.Nodes.AddRange(nodes);
        return graph;
    }

    internal static byte[] Serialize(GraphProto graph, params FunctionProto[] functions)
    {
        var model = new ModelProto { IrVersion = 10, Graph = graph };
        model.OpsetImports.Add(new OperatorSetIdProto { Domain = "", Version = 21 });
        model.OpsetImports.Add(new OperatorSetIdProto { Domain = "Functions", Version = 1 });
        model.Functions.AddRange(functions);
        using var stream = new MemoryStream();
        ProtoBuf.Serializer.Serialize(stream, model);
        return stream.ToArray();
    }

    internal static byte[] Onnx(string opType, int elementType, string domain = "", AttributeProto? attribute = null)
    {
        static ValueInfoProto Value(string name, int type)
            => new() { Name = name, Type = new TypeProto { TensorType = new TypeProto.Tensor { ElemType = type } } };
        var node = new NodeProto { OpType = opType, Name = "node", Domain = domain };
        node.Inputs.Add("x0");
        node.Outputs.Add("y");
        if (attribute is not null) node.Attributes.Add(attribute);
        var graph = new GraphProto { Name = "g" };
        graph.Inputs.Add(Value("x0", elementType));
        graph.Outputs.Add(Value("y", elementType));
        graph.Nodes.Add(node);
        return Serialize(graph);
    }
}

[Module]
public partial class NegativeInfinityWindowMaxPoolIndicesValues
{
    public static Tensor<float32> Inline(Tensor<float32> x)
        => ((Tensor<int64>)OnnxOp.MaxPoolWithIndices(x, AutoPad.NotSet, false, null, [2L], [1L, 1L], 0L, null).indices).Cast<float32>();
}
