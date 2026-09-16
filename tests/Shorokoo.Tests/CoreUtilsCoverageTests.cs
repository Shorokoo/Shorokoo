using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using Shorokoo.Core.Factory.OpsFactories;
using Shorokoo.Core.Inference;
using Shorokoo.Core.Inference.Abstractions;
using Shorokoo.OnnxRuntime;
using Shorokoo.Runtime;

namespace Shorokoo.Tests;

/// <summary>
/// Coverage for the small framework utilities the graph-level suites never touch directly: the
/// internal LINQ-ish helpers in <c>Shorokoo.Core.Utils.Extensions</c>, the <see cref="NodeKey"/> /
/// <see cref="TensorKey"/> identity structs, the <see cref="ShorokooException"/> hierarchy, the
/// OpsFactories <see cref="Helpers"/> dtype sets and attribute-type mapping, the
/// <see cref="InferenceBackend"/> deployment-folder discovery and selection policy, the
/// description a live backend answers with and the device assertion built on it, the
/// <see cref="DeviceMemory"/> settings the CUDA backends map onto ORT's arena options, the typed
/// value-handle conversions, <c>ShapeUtils</c>' argument validation for <c>Reshape</c>'s
/// <c>keepAxes</c>, the <see cref="AtomicFileWriter"/> temp-and-rename commit protocol
/// (crash-window fault injection, stale-temp sweep, retain-last-N rotation), the
/// <see cref="DebugRequests"/> snapshot hook firing at every <see cref="GraphCreationPoint"/>,
/// and the public-API shape guard against a <c>params</c> array sitting behind an optional
/// parameter.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
[Collection(DeviceMemorySettings.Name)]
public class CoreUtilsCoverageTests
{
    private static InternalComputationGraph BoolGraph(IValue only) => new([], [only.ToVariable()]);

    [Fact]
    public void TestReshapeRejectsNegativeAndRepeatedKeepAxes()
    {
        var x = Vector(1f, 2f, 3f, 4f).Reshape(Vector(2L, 2L));
        Assert.Equal("keepAxes", Assert.Throws<ArgumentOutOfRangeException>(
            () => x.Reshape(Vector(-1L), keepAxes: [-1])).ParamName);
        Assert.Equal("keepAxes", Assert.Throws<ArgumentException>(
            () => x.Reshape(Vector(-1L), keepAxes: [0, 0])).ParamName);
    }

    [Fact]
    public void TestSelfCheckingGraphConventionRejectsAFalseBitAtAnyRank()
    {
        Assert.True(AutoTest.TestGraph(BoolGraph(Scalar(true))));
        Assert.False(AutoTest.TestGraph(BoolGraph(Scalar(false))));
        Assert.True(AutoTest.TestGraph(BoolGraph(Vector(true))));
        Assert.False(AutoTest.TestGraph(BoolGraph(Vector(false))));
        Assert.False(AutoTest.TestGraph(BoolGraph(Vector(true, false))));
    }

    /// <summary>QEE stub that hands back its first input untouched, degrading the op it replaces
    /// to an identity so the two engines disagree on purpose.</summary>
    private sealed class QeeIdentityStub : QuickOp
    {
        private readonly string _opCode;
        public QeeIdentityStub(string opCode) { _opCode = opCode; }
        public override string OpCode => _opCode;
        protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
            => [inputs[0]!];
    }

    private static InternalComputationGraph NotFalseGraph() => new([], [OnnxOp.Not(Scalar(false).ToVariable())]);

    [Fact]
    public void TestSelfCheckingGraphConventionIsEnforcedOnTheQuickEngineToo()
    {
        Assert.True(AutoTest.TestGraph(NotFalseGraph(), testQuickEngineExecution: true));

        using (OpRegistry.Override(new QeeIdentityStub(OpCodes.NOT)))
        {
            Assert.False(AutoTest.TestGraph(NotFalseGraph(), testQuickEngineExecution: true));
            Assert.True(AutoTest.TestGraph(NotFalseGraph(), testQuickEngineExecution: false));
        }
    }

    [Fact]
    public void TestExtensionsCoverage()
    {
        var list = new List<int> { 1 };
        list.AddAll([2, 3]);
        Assert.Equal([1, 2, 3], list);

        (string, int)[] valueTuples = [("a", 1), ("b", 2)];
        var dctTuples = new Dictionary<string, int>();
        dctTuples.AddAll(valueTuples);
        Assert.Equal(2, dctTuples.Count);

        Tuple<string, int>[] refTuples = [Tuple.Create("x", 9)];
        var dctRefTuples = new Dictionary<string, int>();
        dctRefTuples.AddAll(refTuples);
        Assert.Equal(9, dctRefTuples["x"]);

        string[] ab = ["a", "b"];
        string[] abc = ["a", "b", "c"];
        Assert.Equal([("a", 0), ("b", 1)], ab.Iterate().ToArray());
        Assert.Equal((Index)1, abc.FindIndexOf("b"));
        Assert.Equal(^0, abc.FindIndexOf("z"));

        string?[] withNulls = ["a", null, "b"];
        string?[] noNulls = ["x"];
        Assert.Equal(ab, withNulls.NotNulls().ToArray());
        Assert.Equal<int>([1, 3], Extensions.NotNulls<int>([1, null, 3]).ToArray());
        Assert.Equal<int>([2, 4], Extensions.NotNull<int>([null, 2, 4]).ToArray());
        Assert.Equal(["x"], noNulls.AssertNotNulls().ToArray());

        Assert.Equal("ok", ((string?)"ok").AssertNotNull());
        Assert.Equal(7, ((int?)7).AssertNotNull());
        Assert.Equal("ok", ((string?)"ok").NotNull());
        Assert.Equal(7, ((int?)7).NotNull());
        Assert.Throws<InvalidTensorOperationException>(() => ((string?)null).NotNull());
        Assert.Throws<InvalidTensorOperationException>(() => ((int?)null).NotNull());

        ITuple tuple = (1, 2, 3);
        Assert.Equal(3, tuple.ToEnumerable().Cast<object>().Count());
        Assert.Equal<int>([1, 2, 3], tuple.Cast<int>().ToArray());

        // Convert: identity + the widening/narrowing iterators + unsupported target, per source type.
        int[] ints = [1, 2];
        Assert.Same(ints, ints.Convert<int>());
        Assert.Equal<uint>([1, 2], ints.Convert<uint>().ToArray());
        Assert.Equal<long>([1, 2], ints.Convert<long>().ToArray());
        Assert.Equal<ulong>([1, 2], ints.Convert<ulong>().ToArray());
        Assert.Throws<UnsupportedDTypeException>(() => ints.Convert<float>());

        uint[] uints = [3, 4];
        Assert.Equal<int>([3, 4], uints.Convert<int>().ToArray());
        Assert.Same(uints, uints.Convert<uint>());
        Assert.Equal<long>([3, 4], uints.Convert<long>().ToArray());
        Assert.Equal<ulong>([3, 4], uints.Convert<ulong>().ToArray());
        Assert.Throws<UnsupportedDTypeException>(() => uints.Convert<float>());

        long[] longs = [5, 6];
        Assert.Equal<int>([5, 6], longs.Convert<int>().ToArray());
        Assert.Equal<uint>([5, 6], longs.Convert<uint>().ToArray());
        Assert.Same(longs, longs.Convert<long>());
        Assert.Equal<ulong>([5, 6], longs.Convert<ulong>().ToArray());
        Assert.Throws<UnsupportedDTypeException>(() => longs.Convert<float>());

        ulong[] ulongs = [7, 8];
        Assert.Equal<int>([7, 8], ulongs.Convert<int>().ToArray());
        Assert.Equal<uint>([7, 8], ulongs.Convert<uint>().ToArray());
        Assert.Equal<long>([7, 8], ulongs.Convert<long>().ToArray());
        Assert.Same(ulongs, ulongs.Convert<ulong>());
        Assert.Throws<UnsupportedDTypeException>(() => ulongs.Convert<float>());

        // Generic Convert<TIn, TOut>: implicit-operator path (long → Shape) and the
        // Convert.ChangeType fallback path (int → double).
        long[] dims = [2L, 3L];
        var shapes = dims.Convert<long, Shape>().ToArray();
        Assert.Equal(new Shape(2L), shapes[0]);
        Assert.Equal(new Shape(3L), shapes[1]);
        Assert.Equal<double>([1.0, 2.0], ints.Convert<int, double>().ToArray());
    }

    [Fact]
    public void TestNodeAndTensorKeyCoverage()
    {
        var key = NodeKey.New();
        Assert.False(key.IsEmpty);
        Assert.True(NodeKey.Empty.IsEmpty);

        var parsed = NodeKey.Parse(key.Id.ToString());
        Assert.Equal(key, parsed);
        Assert.True(key == parsed);
        Assert.False(key != parsed);
        Assert.True(key.Equals((object)parsed));
        Assert.Equal(key.GetHashCode(), parsed.GetHashCode());
        Assert.Equal(0, key.CompareTo(parsed));
        Assert.Equal(key.Id.ToString("N"), key.ToString());

        Assert.True(NodeKey.TryParse(key.Id.ToString(), out var tryParsed));
        Assert.Equal(key, tryParsed);
        Assert.False(NodeKey.TryParse("not-a-guid", out var failed));
        Assert.True(failed.IsEmpty);

        var node = NodeKey.New();
        var tensor = new TensorKey(node, 2);
        Assert.False(tensor.IsEmpty);
        Assert.False(tensor.IsConnectingTensor);
        Assert.True(TensorKey.Empty.IsEmpty);
        Assert.True(TensorKey.ForConnectingTensor(node).IsConnectingTensor);

        var roundTripped = TensorKey.Parse(tensor.ToString());
        Assert.Equal(tensor, roundTripped);
        Assert.True(tensor == roundTripped);
        Assert.False(tensor != roundTripped);
        Assert.True(tensor.Equals((object)roundTripped));
        Assert.Equal(tensor.GetHashCode(), roundTripped.GetHashCode());
        Assert.Equal(0, tensor.CompareTo(roundTripped));
        Assert.True(tensor.CompareTo(new TensorKey(node, 3)) < 0);

        Assert.Throws<FormatException>(() => TensorKey.Parse("missing-colon"));
        Assert.True(TensorKey.TryParse(tensor.ToString(), out var tensorTryParsed));
        Assert.Equal(tensor, tensorTryParsed);
        Assert.False(TensorKey.TryParse(null, out _));
        Assert.False(TensorKey.TryParse("", out _));
        Assert.False(TensorKey.TryParse("a:b:c", out _));
        Assert.False(TensorKey.TryParse("nope:1", out _));
    }

    [Fact]
    public void TestShorokooExceptionsCoverage()
    {
        var inner = new InvalidOperationException("inner");

        var dtype = new UnsupportedDTypeException("E001", "float99", "Cast", "extra context");
        Assert.Equal("E001", dtype.ErrorCode);
        Assert.Equal("float99", dtype.DTypeName);
        Assert.Equal("Cast", dtype.Operation);
        Assert.Contains("[E001]", dtype.Message);

        var tensorOp = new InvalidTensorOperationException("E002", "Reshape", "t0", "bad dims");
        Assert.Equal("Reshape", tensorOp.Operation);
        Assert.Equal("t0", tensorOp.TensorInfo);

        var node = new OnnxNodeException("E003", "Add", "add_1", "boom");
        Assert.Equal("Add", node.NodeType);
        Assert.Equal("add_1", node.NodeName);

        var module = new ModuleException("E004", "MyModule", "broken");
        Assert.Equal("MyModule", module.ModuleName);
        var moduleInner = new ModuleException("E004", "MyModule", "broken", inner);
        Assert.Same(inner, moduleInner.InnerException);

        var ctx = new ComputeContextException("E005", "cpu", "session failed");
        Assert.Equal("cpu", ctx.ContextInfo);
        var ctxInner = new ComputeContextException("E005", "cpu", "session failed", inner);
        Assert.Same(inner, ctxInner.InnerException);

        var model = new ModelException("E006", "model.onnx", "load failed");
        Assert.Equal("model.onnx", model.ModelInfo);
        var modelInner = new ModelException("E006", "model.onnx", "load failed", inner);
        Assert.Same(inner, modelInner.InnerException);

        var autodiff = new AutoDiffNotSupportedException("E007", "Det", "no gradient");
        Assert.Equal("Det", autodiff.OpName);
        Assert.Contains("Det", autodiff.Message);

        var reflection = new ReflectionException("E008", "Invoke", "MyType", "missing method");
        Assert.Equal("Invoke", reflection.MethodInfo);
        Assert.Equal("MyType", reflection.TypeInfo);
    }

    [Fact]
    public void TestOpsFactoriesHelpersCoverage()
    {
        // The dtype-set fields are consumed by op factories; touching each runs the
        // static initializers.
        Assert.Contains(DType.Float32, Helpers.Numeric14);
        Assert.Contains(DType.Float32, Helpers.Numeric13);
        Assert.Contains(DType.Float32, Helpers.Numeric6);
        Assert.Contains(DType.Float32, Helpers.Numeric1);
        Assert.Contains(DType.String, Helpers.All2);
        Assert.Contains(DType.BFloat16, Helpers.All13);

        Assert.Equal(Core.Factory.IR.AttributeProto.AttributeType.Int, AttributeType.Bool.ToProto());
        Assert.Equal(Core.Factory.IR.AttributeProto.AttributeType.Ints, AttributeType.Bools.ToProto());
        Assert.Equal(Core.Factory.IR.AttributeProto.AttributeType.Int, AttributeType.Long.ToProto());
        Assert.Equal(Core.Factory.IR.AttributeProto.AttributeType.Ints, AttributeType.Longs.ToProto());
        Assert.Equal(Core.Factory.IR.AttributeProto.AttributeType.Int, AttributeType.DType.ToProto());
        Assert.Equal(Core.Factory.IR.AttributeProto.AttributeType.Ints, AttributeType.DTypes.ToProto());
        Assert.Equal(Core.Factory.IR.AttributeProto.AttributeType.Float, AttributeType.Float.ToProto());
        Assert.Equal(Core.Factory.IR.AttributeProto.AttributeType.Floats, AttributeType.Floats.ToProto());
        Assert.Equal(Core.Factory.IR.AttributeProto.AttributeType.Graph, AttributeType.Graph.ToProto());
        Assert.Equal(Core.Factory.IR.AttributeProto.AttributeType.String, AttributeType.String.ToProto());
        Assert.Equal(Core.Factory.IR.AttributeProto.AttributeType.Strings, AttributeType.Strings.ToProto());
        Assert.Equal(Core.Factory.IR.AttributeProto.AttributeType.String, AttributeType.Enum.ToProto());
        Assert.Equal(Core.Factory.IR.AttributeProto.AttributeType.Strings, AttributeType.Enums.ToProto());
        Assert.Equal(Core.Factory.IR.AttributeProto.AttributeType.Tensor, AttributeType.Tensor.ToProto());
        Assert.Equal(Core.Factory.IR.AttributeProto.AttributeType.TypeProto, AttributeType.TypeProto.ToProto());
        Assert.Throws<UnsupportedDTypeException>(() => ((AttributeType)(-1)).ToProto());
    }

    [Fact]
    public void TestInferenceBackendDiscoveryAndSelectionPolicyCoverage()
    {
        // No backend is set explicitly in this suite, so accessing Factory exercises the
        // deployment-folder auto-discovery fallback; the platform backend is derived from the
        // running OS, so this holds on Windows and Linux alike.
        var factory = InferenceBackend.Factory;
        Assert.NotNull(factory);
        var name = factory.GetType().Assembly.GetName().Name ?? "";
        Assert.StartsWith(RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "Shorokoo.Win" : "Shorokoo.Linux", name);

        // Discovery is sticky, and the discovered backend actually executes.
        Assert.Same(factory, InferenceBackend.Factory);
        Assert.Equal(5f, OnnxEngine.Eval(Scalar(2f) + Scalar(3f)).As<float32>().AccessMemory()[0]);

        // The multi-candidate selection policy (the suite ships only one backend, so drive it
        // directly): nothing deployed → no choice; a single backend is taken as-is; several are
        // refused by name rather than guessed between.
        var cpu = ("Shorokoo.LinuxCPU", false);
        var gpu = ("Shorokoo.LinuxGPU", true);
        Assert.Null(InferenceBackend.SelectBackend([], "deployed in '/app'"));
        Assert.Equal(cpu, InferenceBackend.SelectBackend([cpu], "deployed in '/app'")!.Value);
        Assert.Equal(gpu, InferenceBackend.SelectBackend([gpu], "deployed in '/app'")!.Value);
        var refused = Assert.Throws<InvalidOperationException>(
            () => InferenceBackend.SelectBackend([cpu, gpu], "deployed in '/app'"));
        Assert.Contains("Shorokoo.LinuxCPU (CPU), Shorokoo.LinuxGPU (CUDA)", refused.Message);
        Assert.Contains("deployed in '/app'", refused.Message);
    }

    [Fact]
    public void TestTheLiveBackendNamesItsDeviceAndCanBeRequired()
    {
        var live = InferenceBackend.Describe();
        Assert.Same(InferenceBackend.Factory, InferenceBackend.Current);
        Assert.StartsWith(
            RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "Shorokoo.Win" : "Shorokoo.Linux", live.Name);
        Assert.Equal(live.Name.EndsWith("GPU", StringComparison.Ordinal) ? ComputeDevice.Cuda : ComputeDevice.Cpu,
            live.Device);
        Assert.Equal(live, ComputeContext.Default.Backend);
        Assert.Equal(live, new ComputeContext().Backend);

        InferenceBackend.RequireDevice(live.Device);
        var other = live.Device == ComputeDevice.Cpu ? ComputeDevice.Cuda : ComputeDevice.Cpu;
        var refusal = Assert.Throws<InvalidOperationException>(
            () => InferenceBackend.RequireDevice(other)).Message;
        Assert.Contains(live.ToString(), refusal);
        Assert.Contains(other == ComputeDevice.Cuda ? "requires a CUDA backend" : "requires a CPU backend", refusal);

        var cuda = new CudaFactoryProbe(3).Description;
        Assert.Equal(new BackendDescription("Shorokoo.Tests", ComputeDevice.Cuda, 3), cuda);
        Assert.Equal("Shorokoo.Tests (CUDA device 3)", cuda.ToString());
        Assert.Equal("Shorokoo.LinuxCPU (CPU)",
            new BackendDescription("Shorokoo.LinuxCPU", ComputeDevice.Cpu, null).ToString());
    }

    [Fact]
    public void TestTheProjectsAModelLibraryReferencesCarryNoBackendOrOnnxRuntime()
    {
        string[] projects = ["src/Shorokoo/Shorokoo.csproj", "src/Shorokoo.Modules/Shorokoo.Modules.csproj"];
        foreach (var project in projects)
        {
            var text = File.ReadAllText(Path.Combine(RepoRoot(), project.Replace('/', Path.DirectorySeparatorChar)));
            Assert.DoesNotContain(
                Regex.Matches(text, @"<(?:Package|Project)Reference\s+Include=""([^""]+)""")
                    .Select(m => Path.GetFileNameWithoutExtension(m.Groups[1].Value)),
                r => r.Contains("OnnxRuntime", StringComparison.OrdinalIgnoreCase)
                     || Regex.IsMatch(r, "(Win|Linux)(CPU|GPU)$"));
        }

        Assembly[] backendFree = [typeof(InferenceBackend).Assembly, typeof(global::Shorokoo.Modules.Layers.Linear).Assembly];
        foreach (var assembly in backendFree)
            Assert.DoesNotContain(assembly.GetReferencedAssemblies(), a =>
                (a.Name ?? "").Contains("OnnxRuntime", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void TestABackendDescriptionCannotContradictItself()
    {
        Assert.Throws<ArgumentException>(() => new BackendDescription("X", ComputeDevice.Cuda, null));
        Assert.Throws<ArgumentException>(() => new BackendDescription("X", ComputeDevice.Cpu, 0));
        Assert.Throws<ArgumentException>(() => new BackendDescription("X", ComputeDevice.Other, 0));
        Assert.Equal("X (CUDA device 2)", new BackendDescription("X", ComputeDevice.Cuda, 2).ToString());
        Assert.Equal("X (CPU)", new BackendDescription("X", ComputeDevice.Cpu, null).ToString());
        Assert.Equal("X (Other)", new BackendDescription("X", ComputeDevice.Other, null).ToString());
        Assert.Equal("", default(BackendDescription).Name);
    }

    [Fact]
    public void TestABackendOnAnotherProviderCanSaySoAndIsNotTakenForTheCpu()
    {
        IShorokooInferenceSessionFactory other = new OtherDeviceFactoryProbe();
        Assert.Equal(ComputeDevice.Other, other.Description.Device);
        Assert.Null(other.Description.CudaDeviceId);
    }

    [Fact]
    public void TestALoadedBackendForAnotherOsIsNotACandidate()
    {
        var windows = RuntimeInformation.IsOSPlatform(OSPlatform.Windows);
        (string, bool)[] thisOs = windows ? [("Shorokoo.WinCPU", false)] : [("Shorokoo.LinuxCPU", false)];
        string[] bothOses = ["System.Private.CoreLib", "Shorokoo.WinCPU", "Shorokoo.LinuxCPU"];
        string[] bothDevices = windows
            ? ["Shorokoo.WinCPU", "Shorokoo.WinGPU"]
            : ["Shorokoo.LinuxCPU", "Shorokoo.LinuxGPU"];
        Assert.Equal(thisOs, InferenceBackend.LoadedCandidates(bothOses));
        Assert.Empty(InferenceBackend.LoadedCandidates(["System.Private.CoreLib"]));
        Assert.Equal(2, InferenceBackend.LoadedCandidates(bothDevices).Count);
    }

    /// <summary>A CUDA <see cref="OrtSessionFactory"/> without a card: it describes itself from its
    /// constructor arguments, and no session is built here.</summary>
    private sealed class CudaFactoryProbe : OrtSessionFactory
    {
        public CudaFactoryProbe(int device) : base(device) { }
    }

    /// <summary>The subclass the base class invites: a different execution provider, on no CUDA
    /// device, which is neither the CPU nor something the base class can name for it.</summary>
    private sealed class OtherDeviceFactoryProbe : OrtSessionFactory
    {
        public OtherDeviceFactoryProbe() : base(_ => { }, cudaDeviceId: null) { }

        public override BackendDescription Description
            => new("Shorokoo.Tests", ComputeDevice.Other, null);
    }

    [Fact]
    public void TestDeviceMemorySettingsMapOntoTheCudaArenaOptions()
    {
        var uncapped = OrtSessionFactory.CudaProviderOptions(0, null, ArenaExtendStrategy.NextPowerOfTwo);
        Assert.Equal("0", uncapped["device_id"]);
        Assert.Equal("kNextPowerOfTwo", uncapped["arena_extend_strategy"]);
        Assert.False(uncapped.ContainsKey("gpu_mem_limit"));

        var budgeted = OrtSessionFactory.CudaProviderOptions(
            1, 16L * 1024 * 1024 * 1024, ArenaExtendStrategy.SameAsRequested);
        Assert.Equal("1", budgeted["device_id"]);
        Assert.Equal("kSameAsRequested", budgeted["arena_extend_strategy"]);
        Assert.Equal("17179869184", budgeted["gpu_mem_limit"]);
        Assert.Throws<ArgumentOutOfRangeException>(
            () => OrtSessionFactory.CudaProviderOptions(0, null, (ArenaExtendStrategy)7));

        Assert.Equal("gpu:0", OrtSessionFactory.ArenaShrinkageRunConfig(0, shrinkArenaAfterRun: true));
        Assert.Equal("gpu:3", OrtSessionFactory.ArenaShrinkageRunConfig(3, shrinkArenaAfterRun: true));
        Assert.Null(OrtSessionFactory.ArenaShrinkageRunConfig(0, shrinkArenaAfterRun: false));
        Assert.Null(OrtSessionFactory.ArenaShrinkageRunConfig(null, shrinkArenaAfterRun: true));
    }

    /// <summary>
    /// The shipped defaults, and the ORT options a GPU session built with them asks for. The
    /// arena strategy is deliberately not ORT's own: exact-size extension holds materially less
    /// on a run that feeds one shape to one compiled step, which is what a training run is.
    /// </summary>
    [Fact]
    public void TestDeviceMemoryDefaultsToTheExactSizeArenaAndRejectsAnEmptyBudget()
    {
        Assert.Equal(ArenaExtendStrategy.SameAsRequested, DeviceMemory.ArenaExtend);
        Assert.Null(DeviceMemory.LimitBytes);
        Assert.False(DeviceMemory.ShrinkArenaAfterRun);

        var shipped = OrtSessionFactory.CudaProviderOptions(0, DeviceMemory.LimitBytes, DeviceMemory.ArenaExtend);
        Assert.Equal("kSameAsRequested", shipped["arena_extend_strategy"]);
        Assert.False(shipped.ContainsKey("gpu_mem_limit"));

        Assert.Throws<ArgumentOutOfRangeException>(() => DeviceMemory.LimitBytes = 0);
        Assert.Throws<ArgumentOutOfRangeException>(() => DeviceMemory.LimitBytes = -1);
        Assert.Throws<ArgumentOutOfRangeException>(() => DeviceMemory.ArenaExtend = (ArenaExtendStrategy)7);
        Assert.Null(DeviceMemory.LimitBytes);
        Assert.Equal(ArenaExtendStrategy.SameAsRequested, DeviceMemory.ArenaExtend);

        // A limit set and cleared leaves nothing behind: null is the absence, not a zero.
        DeviceMemory.LimitBytes = 4096;
        Assert.Equal(4096L, DeviceMemory.LimitBytes);
        DeviceMemory.LimitBytes = null;
        Assert.Null(DeviceMemory.LimitBytes);
    }

    /// <summary>A reading is null on a machine with no CUDA runtime and a real one where there is
    /// a card, so the first assertion holds either way; the fold into the peak is driven through
    /// the seam so it is pinned on both.</summary>
    [Fact]
    public void TestDeviceMemoryReadsTheCardWhenThereIsOneAndSampleFoldsIntoThePeak()
    {
        Assert.True(DeviceMemory.Read() is not { TotalBytes: <= 0 });

        DeviceMemory.ResetPeak();
        Assert.Equal(0L, DeviceMemory.PeakUsedBytes);
        Assert.Null(DeviceMemory.SampleFrom(null));
        Assert.Equal(0L, DeviceMemory.PeakUsedBytes);

        var reading = new DeviceMemoryReading(4096, 1024, 5120);
        Assert.Equal(reading, DeviceMemory.SampleFrom(reading));
        Assert.Equal(4096L, DeviceMemory.PeakUsedBytes);
        Assert.Equal(4096L, DeviceMemory.ObservePeak(512));
        Assert.Equal(8192L, DeviceMemory.ObservePeak(8192));
        Assert.Equal(8192L, DeviceMemory.PeakUsedBytes);
        DeviceMemory.ResetPeak();
        Assert.Equal(0L, DeviceMemory.PeakUsedBytes);
    }

    /// <summary>
    /// The settings are reachable from a GPU session, which no test on a CPU box can observe by
    /// running one. What it can observe is that the product still calls the wiring: a backend that
    /// stopped passing its device id, stopped reading <see cref="DeviceMemory"/> when it builds the
    /// provider options, or stopped putting the shrinkage entry on its run options would leave
    /// every setting dead with every other test still green.
    /// </summary>
    [Fact]
    public void TestTheGpuBackendsStillCarryTheDeviceMemorySettingsIntoOrt()
    {
        var backend = Path.Combine(ProductSourceRoot(), "Backend", "OnnxRuntime");
        string Source(params string[] parts) =>
            StripCommentsAndStrings(File.ReadAllText(Path.Combine(backend, Path.Combine(parts))));

        string[] gpuFactories = ["Shorokoo.LinuxGPU/LinuxGpuInferenceFactory.cs", "Shorokoo.WinGPU/WinGpuInferenceFactory.cs"];
        foreach (var gpu in gpuFactories)
            Assert.Matches(@"base\s*\(\s*cudaDeviceId\s*:\s*0\s*\)", Source(gpu.Split('/')));
        string[] cpuFactories = ["Shorokoo.LinuxCPU/LinuxCpuInferenceFactory.cs", "Shorokoo.WinCPU/WinCpuInferenceFactory.cs"];
        foreach (var cpu in cpuFactories)
            Assert.Matches(@"cudaDeviceId\s*:\s*null", Source(cpu.Split('/')));

        var factory = Source("Shorokoo.OnnxRuntime", "OrtSessionFactory.cs");
        Assert.Matches(@"new\s+OrtInferenceSession\s*\(\s*session\s*,\s*_cudaDeviceId\s*\)", factory);
        Assert.Matches(@"AppendExecutionProvider_CUDA\s*\(\s*cuda\s*\)", factory);
        Assert.Contains("DeviceMemory.LimitBytes", factory);
        Assert.Contains("DeviceMemory.ArenaExtend", factory);

        var session = Source("Shorokoo.OnnxRuntime", "OrtInferenceSession.cs");
        Assert.Contains("memory.enable_memory_arena_shrinkage", File.ReadAllText(
            Path.Combine(backend, "Shorokoo.OnnxRuntime", "OrtInferenceSession.cs")));
        Assert.Matches(@"ArenaShrinkageRunConfig\s*\(\s*_cudaDeviceId\s*,\s*DeviceMemory\.ShrinkArenaAfterRun\s*\)", session);
        Assert.Matches(@"AddRunConfigEntry\s*\(", session);

        // Every path that runs the session has to apply it, not just one: the retaining path is
        // the loop a GPU user is steered into, and it is where an unbounded arena costs most.
        var runPaths = Regex.Matches(session, @"_session\s*\.\s*Run\w*\s*\(").Count;
        Assert.Equal(runPaths, Regex.Matches(session, @"ConfigureRun\s*\(\s*runOptions\s*\)").Count);
    }

    /// <summary>Two shapes the guard's exemptions once let through: a span consumed by a call
    /// inside a returned expression, which the caller never sees, and one "rooted" by a mention of
    /// the same identifier in a later member — with names like `data`, `value` or `t`, nearly
    /// always true. The return exemption now needs the span to be the returned expression itself,
    /// and the widening stops at the member the span was taken in.</summary>
    [Fact]
    public void TestTheSpanGuardCatchesASpanConsumedInsideAReturnOrRootedByAnotherMember()
    {
        Assert.NotEmpty(SpansUsedWithoutKeepingTheTensorAlive(
            "class C { string K(TensorData data) => Hex(data.AccessRawMemory()); }"));
        Assert.NotEmpty(SpansUsedWithoutKeepingTheTensorAlive(
            "class C { string K(TensorData data) { return Hex(data.AccessRawMemory()).Trim(); } }"));
        Assert.NotEmpty(SpansUsedWithoutKeepingTheTensorAlive(
            "class C { void K(TensorData data) { Use(data.AccessRawMemory(), 1); } void Other() { Log(data); } }"));
        // An expression-bodied member has no braces to bound the search, which is the form the
        // instance in the tree had.
        Assert.NotEmpty(SpansUsedWithoutKeepingTheTensorAlive(
            "class C { string K(TensorData d) => Hex(d.AccessRawMemory()).Trim(); void O() { var d = 1; } }"));
        // ...and a generic constraint ending in `class` is not a type header, so it must not cut
        // the search short and flag a rooted read.
        Assert.Empty(SpansUsedWithoutKeepingTheTensorAlive(
            "class C { void M<T>(TensorData t) where T : class { if (c) { var d = t.AccessRawMemory(); b.CopyTo(d); } GC.KeepAlive(t); } }"));
    }

    /// <summary>
    /// No filter in <c>release.yml</c> selects a <c>Purpose=Benchmark</c> class implicitly — each
    /// needs a step naming it, and each must precede the <c>Purpose=Gate</c> step, whose MSBuild
    /// workers wreck any measurement sharing the runner. Two classes were once added without one
    /// and never ran at release (Shorokoo/Shorokoo#277).
    /// </summary>
    [Fact]
    public void TestEveryBenchmarkClassHasItsOwnReleaseStepBeforeTheGate()
    {
        var workflow = File.ReadAllText(Path.Combine(RepoRoot(), ".github", "workflows", "release.yml"));
        var benchmarks = typeof(Shorokoo.Tests.Benchmarks.MemoryPassBenchmarkTests).Assembly.GetTypes()
            .Where(t => t.GetCustomAttributesData().Any(a =>
                a.AttributeType == typeof(Xunit.TraitAttribute) &&
                a.ConstructorArguments.Count == 2 &&
                (string?)a.ConstructorArguments[0].Value == "Purpose" &&
                (string?)a.ConstructorArguments[1].Value == "Benchmark"))
            .Select(t => t.Name)
            .ToArray();
        Assert.NotEmpty(benchmarks);

        var gate = workflow.IndexOf("\"Purpose=Gate\"", StringComparison.Ordinal);
        Assert.True(gate > 0);
        foreach (var name in benchmarks)
        {
            var step = workflow.IndexOf($"FullyQualifiedName~{name}\"", StringComparison.Ordinal);
            Assert.True(step > 0);
            Assert.True(step < gate);
        }
    }

    private static string RepoRoot() => Ancestor(d => File.Exists(Path.Combine(d, "Shorokoo.sln")));

    private static string ProductSourceRoot() =>
        Path.Combine(Ancestor(d => Directory.Exists(Path.Combine(d, "src", "Shorokoo"))), "src");

    private static string Ancestor(Func<string, bool> holds)
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !holds(dir.FullName))
            dir = dir.Parent;
        Assert.NotNull(dir);
        return dir!.FullName;
    }

    // Every way the product comes by one of ORT's SafeHandle types: the constructors, and the
    // SessionOptions factories (MakeSessionOptionWithCudaProvider and friends) that return one with
    // no `new` in it. The provider-options and arena types are here for the same reason as the rest
    // -- ORT takes each as a bare IntPtr and does no ref-counting, so an unrooted one is freed by
    // its critical finalizer mid-call. Test sources are deliberately out of scope: they use the
    // options-factory shape (build, configure, return), which this guard's stricter
    // `using`-or-field rule cannot express.
    private static readonly Regex OrtSafeHandleSource = new(
        @"new\s+(SessionOptions|RunOptions|OrtCUDAProviderOptions|OrtArenaCfg|OrtMemoryInfo)\s*\("
        + @"|SessionOptions\s*\.\s*Make\w*\s*\(", RegexOptions.Compiled);

    // The shapes that actually root the handle across a native call: the resource of a `using`, a
    // field, which lives as long as its owner, and a bare `return` of the handle itself, where
    // ownership passes to the caller and the callee never touches it again. The handle must be the
    // WHOLE initializer -- `using var s = new InferenceSession(b, new SessionOptions())` roots the
    // session and leaves the options collectible, which is the exact bug this guard exists for, and
    // `return Wrap(new SessionOptions())` consumes the handle rather than handing it back.
    private const string OrtSafeHandleTypes =
        "SessionOptions|RunOptions|OrtCUDAProviderOptions|OrtArenaCfg|OrtMemoryInfo";

    private static readonly Regex RootedInitializer = new(
        @"^\s*(using\s*\(?\s*(var|" + OrtSafeHandleTypes + @")\s+\w+\s*=\s*"
        + @"|(public|private|protected|internal)[\w\s]*?(" + OrtSafeHandleTypes + @")\s+\w+\s*=\s*"
        + @"|return\s*)$",
        RegexOptions.Compiled);

    // Strings go before line comments: a literal containing "//" would otherwise blank the rest of
    // its line and hide a construction sitting after it.
    private static string StripCommentsAndStrings(string source)
    {
        source = Regex.Replace(source, @"/\*.*?\*/", " ", RegexOptions.Singleline);
        source = Regex.Replace(source, @"@""(?:[^""]|"""")*""", " ");
        source = Regex.Replace(source, @"""(?:\\.|[^""\\])*""", " ");
        source = Regex.Replace(source, @"//[^\n]*", " ");
        return source;
    }

    // The second rooting shape, which no `using` can express because the value has to be
    // returned: a bare span taken out of a tensor (or the runtime value under one) and then read
    // or written THROUGH. Taking the span is the tensor's last read, so the JIT may retire it
    // before the span is used, and a collection on any thread then frees the buffer mid-copy.
    // Forwarding the span straight out (`return Inner.GetTensorDataAsSpan<T>();`) is not this
    // shape — the caller owns the lifetime from there. CopyMemory / CopyRawMemory / ValueAt do
    // the copy with the tensor kept alive, and are what a call site should reach for instead.
    private static readonly Regex SpanOutOfTensor = new(
        @"\.\s*(GetTensorMutableRawData|GetTensorDataAsSpan|GetTensorMutableDataAsSpan"
        + @"|AccessRawMemory|AccessModifiableRawMemory|AccessMemory|AccessModifiableMemory)"
        + @"\s*(<[^>()]*>)?\s*\(\s*\)", RegexOptions.Compiled);

    // The leading identifier of the receiver expression: `t`, `t.Field`, `run[0].ToTensorData()`.
    private static readonly Regex ReceiverRoot = new(
        @"([A-Za-z_]\w*)(?:\s*(?:\[[^\]]*\]|\.\s*\w+\s*(?:<[^>()]*>)?\s*(?:\([^()]*\))?))*\s*$",
        RegexOptions.Compiled);

    private static string[] SpansUsedWithoutKeepingTheTensorAlive(string source)
    {
        var code = StripCommentsAndStrings(source);
        var flagged = new List<string>();
        foreach (Match m in SpanOutOfTensor.Matches(code))
        {
            var before = code[(code.LastIndexOfAny([';', '{', '}'], m.Index) + 1)..m.Index];
            var after = code[(m.Index + m.Length)..];
            // Handed straight back to the caller, who owns the lifetime from there. The span has to
            // BE what is returned: `return Hex(t.AccessRawMemory());` consumes it inside a call and
            // hands back something else, leaving the tensor retired at the read -- the bug, not the
            // exemption. Requiring `;` immediately after the span is what separates them, since a
            // call around it closes with `)` first; the `return`/`=>` test only establishes that
            // this is a return at all.
            if (Regex.IsMatch(after, @"^\s*;") && Regex.IsMatch(before, @"(\breturn\b|=>)[^;{}]*$"))
                continue;
            var receiver = ReceiverRoot.Match(before.TrimEnd());
            if (!receiver.Success) continue;
            var name = receiver.Groups[1].Value;
            if (name is "this" or "base") continue;
            // A `using` compiles to try/finally, so the resource is reachable for the whole block.
            var enclosing = code[BlockStartBefore(code, m.Index)..m.Index];
            if (Regex.IsMatch(enclosing, @"\busing\s+(var|[\w<>\[\],\s]+)\s+" + Regex.Escape(name) + @"\b"))
                continue;
            if (!RootedAfter(code, m.Index + m.Length, name))
                flagged.Add(m.Value.Trim());
        }
        return [.. flagged];
    }

    // Rooted if the receiver is named again after the span is taken — a later read keeps it
    // reachable — or kept alive explicitly. Scoped to the enclosing block, then widened outwards
    // so a `GC.KeepAlive` after the `if`/`try` that uses the span still counts, but stopping
    // before the type body so one method's keep-alive never excuses another's.
    private static bool RootedAfter(string code, int from, string name)
    {
        var mention = new Regex(@"\b" + Regex.Escape(name) + @"\b");
        // The member the span was taken in. Widening past it would let a later sibling's mention
        // of the same identifier -- `data`, `value`, `t` -- root a read in this one, which with
        // names that common is nearly always true and exempts everything.
        int limit = MemberEndFrom(code, from);
        int cursor = from;
        for (int depth = 0; depth < 8; depth++)
        {
            int end = Math.Min(BlockEndFrom(code, cursor), limit);
            if (mention.IsMatch(code[cursor..end])) return true;
            if (end >= limit || end >= code.Length || IsTypeBody(code, end)) return false;
            cursor = end;
        }
        return false;
    }

    /// <summary>Index just past the closing brace of the member — method, property, local
    /// function — containing <paramref name="from"/>: the outermost block enclosing it whose
    /// header is not a type's.</summary>
    private static int MemberEndFrom(string code, int from)
    {
        // The opening braces enclosing `from`, innermost first.
        var opens = new List<int>();
        for (int inside = BlockStartBefore(code, from); inside > 0 && opens.Count < 32;
             inside = BlockStartBefore(code, inside - 1))
            opens.Add(inside - 1);

        int memberOpen = -1;
        foreach (var open in opens)
        {
            var header = code[Math.Max(0, open - 240)..open];
            if (OpensATypeBody(header)) break;
            memberOpen = open;
        }
        // No enclosing block below the type means an expression-bodied member, which has no braces
        // to bound it. Its statement does: widening past the `;` would reach the whole rest of the
        // file, which is the unbounded search this limit exists to stop.
        return memberOpen < 0
            ? StatementEndFrom(code, from)
            : BlockEndFrom(code, memberOpen + 1);
    }

    /// <summary>Whether a block's header text opens a type body. A real one names the type, which
    /// is what keeps a generic constraint — <c>where T : class</c>, ending in the same keyword —
    /// from reading as one and cutting a method's limit short.</summary>
    private static bool OpensATypeBody(string header) =>
        Regex.IsMatch(header, @"\b(class|struct|record|interface|enum|namespace)\s+\w[^;{}]*$");

    /// <summary>Index just past the `;` ending the statement containing <paramref name="from"/>,
    /// ignoring any inside nested brackets so a lambda body does not end it early.</summary>
    private static int StatementEndFrom(string code, int from)
    {
        int depth = 0;
        for (int i = from; i < code.Length; i++)
        {
            char c = code[i];
            if (c is '(' or '[' or '{') depth++;
            else if (c is ')' or ']' or '}') depth--;
            else if (c == ';' && depth <= 0) return i + 1;
        }
        return code.Length;
    }

    /// <summary>Index just inside the opening brace of the block containing <paramref name="index"/>.</summary>
    private static int BlockStartBefore(string code, int index)
    {
        int depth = 0;
        for (int i = index - 1; i >= 0; i--)
        {
            if (code[i] == '}') depth++;
            else if (code[i] == '{')
            {
                if (depth == 0) return i + 1;
                depth--;
            }
        }
        return 0;
    }

    /// <summary>Index just past the closing brace of the block containing <paramref name="from"/>.</summary>
    private static int BlockEndFrom(string code, int from)
    {
        int depth = 0;
        for (int i = from; i < code.Length; i++)
        {
            if (code[i] == '{') depth++;
            else if (code[i] == '}')
            {
                if (depth == 0) return i + 1;
                depth--;
            }
        }
        return code.Length;
    }

    /// <summary>True when the block just closed at <paramref name="closeEnd"/> was a type or
    /// namespace body — the point past which widening would see unrelated members.</summary>
    private static bool IsTypeBody(string code, int closeEnd)
    {
        int open = MatchingOpen(code, closeEnd - 1);
        if (open < 0) return true;
        var header = code[Math.Max(0, open - 240)..open];
        return OpensATypeBody(header);
    }

    private static int MatchingOpen(string code, int closeIndex)
    {
        int depth = 0;
        for (int i = closeIndex; i >= 0; i--)
        {
            if (code[i] == '}') depth++;
            else if (code[i] == '{')
            {
                depth--;
                if (depth == 0) return i;
            }
        }
        return -1;
    }

    private static string[] UnrootedOrtSafeHandles(string source)
    {
        char[] statementEnds = [';', '{', '}', ')'];
        var code = StripCommentsAndStrings(source);
        return OrtSafeHandleSource.Matches(code)
            .Where(m =>
            {
                var before = code[(code.LastIndexOfAny(statementEnds, m.Index) + 1)..m.Index];
                if (!RootedInitializer.IsMatch(before)) return true;
                // `return` roots the handle only when the handle IS what is returned. Anything
                // after its closing paren means the caller gets something else and the handle is
                // a temporary retired at the read that produced it -- the bug, not the exemption.
                return Regex.IsMatch(before, @"^\s*return\s*$") && !ReturnsTheHandleItself(code, m);
            })
            .Select(m => m.Value.Trim())
            .ToArray();
    }

    /// <summary>Whether the construction starting at <paramref name="m"/> is the whole of its
    /// return statement — its closing parenthesis followed by nothing but `;`.</summary>
    private static bool ReturnsTheHandleItself(string code, Match m)
    {
        int depth = 0;
        for (int i = m.Index + m.Length - 1; i < code.Length; i++)
        {
            if (code[i] == '(') depth++;
            else if (code[i] == ')' && --depth == 0)
                return Regex.IsMatch(code[(i + 1)..], @"^\s*;");
        }
        return false;
    }

    // A params array behind an optional parameter lets a positional argument bind to the optional
    // instead: the value lands in the wrong parameter wherever the types allow it, and fails
    // resolution somewhere unhelpful wherever they do not.
    private const BindingFlags DeclaredMembers = BindingFlags.Public | BindingFlags.NonPublic |
        BindingFlags.Instance | BindingFlags.Static | BindingFlags.DeclaredOnly;

    private static IEnumerable<string> ParamsBehindOptional(Type type) =>
        from member in type.GetMethods(DeclaredMembers).Cast<MethodBase>()
            .Concat(type.GetConstructors(DeclaredMembers))
        where member.IsPublic || member.IsFamily || member.IsFamilyOrAssembly
        let ps = member.GetParameters()
        where ps.Length > 1 && ps[..^1].Any(p => p.IsOptional)
              && (ps[^1].IsDefined(typeof(ParamArrayAttribute), false)
                  || ps[^1].IsDefined(typeof(ParamCollectionAttribute), false))
        select $"{type.FullName}.{member.Name}";

    private class ParamsBehindOptionalBait
    {
        public ParamsBehindOptionalBait(int a, int b = 0, params int[] rest) { }
        public void InstanceTrap(int a, int b = 0, params int[] rest) { }
        public static void StaticTrap(int a, int b = 0, params int[] rest) { }
        protected static void CollectionTrap(int a, int b = 0, params List<int> rest) { }
    }

    [Fact]
    public void TestNoPublicApiPlacesAParamsArrayBehindAnOptionalParameter()
    {
        var shipped = Directory.EnumerateFiles(AppContext.BaseDirectory, "Shorokoo*.dll")
            .Where(f => Path.GetFileName(f) is not ("Shorokoo.Tests.dll" or "Shorokoo.CodeGen.dll"))
            .Select(Assembly.LoadFrom).ToArray();
        var exported = shipped.SelectMany(a => a.GetExportedTypes()).ToArray();

        Assert.Equal(4, ParamsBehindOptional(typeof(ParamsBehindOptionalBait)).Count());
        Assert.True(shipped.Length >= 4 && exported.Length > 100);
        Assert.Empty(exported.SelectMany(ParamsBehindOptional));
    }

    [Fact]
    public void TestOrtSafeHandlesAreUsingScopedAndTheGuardStillDetectsEveryEvasion()
    {
        var files = Directory
            .EnumerateFiles(ProductSourceRoot(), "*.cs", SearchOption.AllDirectories)
            .Where(f => !f.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}") &&
                        !f.Contains($"{Path.DirectorySeparatorChar}bin{Path.DirectorySeparatorChar}"))
            .ToArray();
        Assert.True(files.Length > 100);

        var sources = files.Select(File.ReadAllText).ToArray();
        Assert.Contains(sources, s => OrtSafeHandleSource.IsMatch(StripCommentsAndStrings(s)));
        Assert.Empty(sources.SelectMany(UnrootedOrtSafeHandles));

        string[] mustFlag =
        [
            "var o = new SessionOptions();",
            "SessionOptions o = new SessionOptions();",
            "_ = new RunOptions();",
            "return Wrap(new SessionOptions());",
            "using var s = new InferenceSession(b, new SessionOptions());",
            "var o = SessionOptions.MakeSessionOptionWithCudaProvider(0);",
            "_prefix = \"https://x\"; var o = new SessionOptions();",
            "var cuda = new OrtCUDAProviderOptions();",
            "options.AppendExecutionProvider_CUDA(new OrtCUDAProviderOptions());",
            "var cfg = new OrtArenaCfg(limit, 1, 1024, -1);",
            "env.CreateAndRegisterAllocator(new OrtMemoryInfo(\"Cpu\", t, 0, m), cfg);",
            "return Wrap(new OrtMemoryInfo(n, t, 0, m));",
            "return new OrtMemoryInfo(n, t, 0, m).GetAllocatorType();",
            "return new SessionOptions().WithSomething();",
        ];
        string[] mustNotFlag =
        [
            "using var o = new SessionOptions();",
            "using (var o = new RunOptions()) { }",
            "using SessionOptions o = new SessionOptions();",
            "private readonly SessionOptions _o = new SessionOptions();",
            "using var o = SessionOptions.MakeSessionOptionWithCudaProvider(0);",
            "using var cuda = new OrtCUDAProviderOptions();",
            "using OrtCUDAProviderOptions cuda = new OrtCUDAProviderOptions();",
            "using var cfg = new OrtArenaCfg(limit, 1, 1024, -1);",
            "private readonly OrtMemoryInfo _info = new OrtMemoryInfo(\"Cpu\", t, 0, m);",
            "return new OrtMemoryInfo(info.Name, info.GetAllocatorType(), info.Id, info.GetMemoryType());",
        ];
        Assert.All(mustFlag, s => Assert.NotEmpty(UnrootedOrtSafeHandles(s)));
        Assert.All(mustNotFlag, s => Assert.Empty(UnrootedOrtSafeHandles(s)));

        Assert.Contains(sources, s => SpanOutOfTensor.IsMatch(StripCommentsAndStrings(s)));
        Assert.Empty(sources.SelectMany(SpansUsedWithoutKeepingTheTensorAlive));

        string[] spansMustFlag =
        [
            "void M() { var d = v.GetTensorMutableRawData(); b.CopyTo(d); }",
            "byte[] M() { return v.GetTensorDataAsSpan<byte>().ToArray(); }",
            "long[] M() { return t.As<int64>().AccessMemory().ToArray(); }",
            "float M() { return t.AccessMemory<float>()[0]; }",
            "void M() { var s = t.AccessModifiableRawMemory(); s[0] = 1; }",
            "void M() { GC.KeepAlive(v); } void N() { var d = v.GetTensorMutableRawData(); b.CopyTo(d); }",
            "void M() { var a = x.AccessRawMemory().ToArray(); GC.KeepAlive(x); var b2 = y.AccessRawMemory().ToArray(); }",
            "void M() { GC.KeepAlive(v); var d = v.GetTensorMutableRawData(); b.CopyTo(d); }",
            "void M() { if (c) { GC.KeepAlive(other); } var d = v.GetTensorMutableRawData(); b.CopyTo(d); }",
        ];
        string[] spansMustNotFlag =
        [
            "ReadOnlySpan<T> M() => Inner.GetTensorDataAsSpan<T>();",
            "ReadOnlySpan<T> M() { return Inner.GetTensorDataAsSpan<T>(); }",
            "Span<double> M(TensorData<float64> d) => d.AccessModifiableMemory<double>();",
            "byte[] M() { var a = v.GetTensorDataAsSpan<byte>().ToArray(); GC.KeepAlive(v); return a; }",
            "void M() { var d = v.GetTensorMutableRawData(); b.CopyTo(d); GC.KeepAlive(v); }",
            "void M() { using var v = Make(); var d = v.GetTensorMutableRawData(); b.CopyTo(d); }",
            "void M() { if (c) { var d = v.GetTensorMutableRawData(); b.CopyTo(d); } GC.KeepAlive(v); }",
            "void M() { try { var d = v.GetTensorMutableRawData(); b.CopyTo(d); } finally { Q(); } GC.KeepAlive(v); }",
            "byte[] M() { var a = t.AccessRawMemory().ToArray(); return Use(t, a); }",
        ];
        Assert.All(spansMustFlag, s => Assert.NotEmpty(SpansUsedWithoutKeepingTheTensorAlive(s)));
        Assert.All(spansMustNotFlag, s => Assert.Empty(SpansUsedWithoutKeepingTheTensorAlive(s)));
    }

    [Fact]
    public void TestVariableHandleConversionCoverage()
    {
        // A graph node carries the structural kind, runtime dtype and rank; wrapping it in a typed
        // value handle must enforce that all three are compatible.
        Variable scalarNode = InputScalar<float32>("a");
        Variable vectorNode = InputVector<float32>("b");
        Variable rank2Node = InputTensor<float32>("c", rank: 2);
        Variable seqNode = OnnxOp.SequenceEmpty(DType.Float32);
        Variable optNode = OnnxOp.Optional(null, DataStructure.Tensor, DType.Float32);

        Assert.Equal(0, ((Variable)(Scalar<float32>)scalarNode).Rank);
        Assert.Equal(1, ((Variable)(Vector<float32>)vectorNode).Rank);
        Assert.Equal(2, ((Variable)(Tensor<float32>)rank2Node).Rank);
        Assert.Equal(DataStructure.Sequence, ((Variable)(TensorSequence<float32>)seqNode).Structure());
        Assert.Equal(DataStructure.Optional, ((Variable)(OptionalTensor<float32>)optNode).Structure());

        // Structure must always match.
        Assert.Throws<InvalidTensorOperationException>(() => (object)(Tensor<float32>)seqNode);
        Assert.Throws<InvalidTensorOperationException>(() => (object)(TensorSequence<float32>)scalarNode);
        Assert.Throws<InvalidTensorOperationException>(() => (object)(OptionalTensor<float32>)scalarNode);

        // No implicit dtype reinterpretation (use Cast to convert).
        Assert.Throws<InvalidTensorOperationException>(() => (object)(Tensor<float64>)scalarNode);
        Assert.Throws<InvalidTensorOperationException>(() => (object)(Scalar<int64>)scalarNode);

        // A known-mismatching rank is an error.
        Assert.Throws<InvalidTensorOperationException>(() => (object)(Scalar<float32>)rank2Node);
        Assert.Throws<InvalidTensorOperationException>(() => (object)(Vector<float32>)rank2Node);
        Assert.Throws<InvalidTensorOperationException>(() => (object)(Scalar<float32>)vectorNode);

        // An UNKNOWN-rank node is adapted with an Identity rank-conversion node.
        Variable unranked = InputTensor<float32>("u"); Assert.Null(unranked.Rank);
        var sFromNull = (Variable)(Scalar<float32>)unranked;
        Assert.Equal(0, sFromNull.Rank); Assert.Equal(OpCodes.IDENTITY, sFromNull.OwningNode.OpCode);
        Variable unranked2 = InputTensor<float32>("u2");
        var vFromNull = (Variable)(Vector<float32>)unranked2;
        Assert.Equal(1, vFromNull.Rank); Assert.Equal(OpCodes.IDENTITY, vFromNull.OwningNode.OpCode);

        // Vec()/Scalar() reinterpret a tensor handle, validating rank exactly like the
        // Variable→handle operators.
        Assert.Throws<InvalidTensorOperationException>(() => (object)((Tensor<float32>)rank2Node).Vec());
        Assert.Throws<InvalidTensorOperationException>(() => (object)((Tensor<float32>)rank2Node).Scalar());
        Assert.Throws<InvalidTensorOperationException>(() => (object)((Scalar<float32>)scalarNode).Vec());
        Assert.Throws<InvalidTensorOperationException>(() => (object)((Vector<float32>)vectorNode).Scalar());
        Assert.Equal(1, ((Variable)((Tensor<float32>)vectorNode).Vec()).Rank);
        Assert.Equal(0, ((Variable)((Tensor<float32>)scalarNode).Scalar()).Rank);
        Variable unrankedVec = InputTensor<float32>("uv");
        var vecAdapted = (Variable)((Tensor<float32>)unrankedVec).Vec();
        Assert.Equal(1, vecAdapted.Rank); Assert.Equal(OpCodes.IDENTITY, vecAdapted.OwningNode.OpCode);

        // Cast<V> is the explicit dtype CONVERSION (inserts a Cast node); there is no reinterpret.
        Assert.Equal(DType.Float64, ((Variable)scalarNode.Cast<float64>()).Type);

        // ITensorSequence.InsertAt accepts any ITensor: a Vector<T>/Scalar<T> is an ITensor but not
        // a Tensor<T>, so the element must convert through its backing Variable — a direct
        // (Tensor<T>)element unbox would throw InvalidCastException.
        ITensorSequence seq = TensorSequence<float32>(InputTensor<float32>("e0", rank: 1));
        ITensor vectorElem = InputVector<float32>("v");
        ITensor scalarElem = InputScalar<float32>("s");
        var afterBoth = seq.InsertAt(vectorElem, Scalar(0L)).InsertAt(scalarElem, Scalar(0L));
        Assert.Equal(DataStructure.Sequence, afterBoth.Structure());
    }

    // ---- AtomicFileWriter: temp-and-rename commit, stale sweep, retain-last-N rotation ----

    private static string NewScratchDir()
    {
        var dir = Path.Combine(Path.GetTempPath(), $"shrk_atomic_{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        return dir;
    }

    private static void WriteVia(string path, string content, Action<string>? onWarning = null) =>
        AtomicFileWriter.WriteFile(
            path, s => s.Write(System.Text.Encoding.UTF8.GetBytes(content)), onWarning);

    private static string WriteSeriesMember(
        string dir, string prefix, int index, string suffix, int keep, Action<string>? onWarning = null)
    {
        var path = Path.Combine(dir, $"{prefix}{index}{suffix}");
        AtomicFileWriter.WriteFile(
            path,
            s => s.Write(System.Text.Encoding.UTF8.GetBytes($"content-{index}")),
            AtomicFileWriter.RetainPolicy.KeepLast(keep, prefix, suffix),
            onWarning);
        return path;
    }

    private static int[] SurvivingIndices(string dir, string prefix, string suffix) =>
        Directory.GetFiles(dir, $"{prefix}*{suffix}")
            .Select(Path.GetFileName)
            .Where(n => n!.StartsWith(prefix, StringComparison.Ordinal) && n.EndsWith(suffix, StringComparison.Ordinal))
            .Select(n => n!.Substring(prefix.Length, n.Length - prefix.Length - suffix.Length))
            .Where(t => t.Length > 0 && t.All(char.IsAsciiDigit))
            .Select(int.Parse)
            .OrderBy(i => i)
            .ToArray();

    [Fact]
    public void TestAtomicFileWriterRotationCoverage()
    {
        var dir = NewScratchDir();
        const string prefix = "ckpt-";
        const string suffix = ".safetensors";
        try
        {
            // Sequential saves with keep=3 leave exactly {9,10,11} — the 9-vs-10 boundary is where
            // a lexicographic ("10" < "9") sort would delete the wrong file.
            for (int i = 0; i <= 11; i++)
                Assert.True(File.Exists(WriteSeriesMember(dir, prefix, i, suffix, keep: 3)));
            int[] afterSequential = [9, 10, 11];
            Assert.Equal(afterSequential, SurvivingIndices(dir, prefix, suffix));

            foreach (var f in Directory.GetFiles(dir, $"{prefix}*{suffix}")) File.Delete(f);

            // Hostile ordering: pre-plant a mix, then one rotating write of index 20 with keep=3.
            int[] planted = [2, 9, 10, 12];
            foreach (var idx in planted)
                File.WriteAllText(Path.Combine(dir, $"{prefix}{idx}{suffix}"), "planted");
            var otherPrefix  = Path.Combine(dir, $"other-5{suffix}");
            var nonNumeric   = Path.Combine(dir, $"{prefix}abc{suffix}");
            var emptyToken   = Path.Combine(dir, $"{prefix}{suffix}");
            var stagedTemp   = Path.Combine(dir, $".tmp-{prefix}7{suffix}");
            var wrongSuffix  = Path.Combine(dir, $"{prefix}8.other");
            string[] nonMembers = [otherPrefix, nonNumeric, emptyToken, stagedTemp, wrongSuffix];
            foreach (var p in nonMembers) File.WriteAllText(p, "not-a-member");

            var committed = WriteSeriesMember(dir, prefix, 20, suffix, keep: 3);

            int[] afterHostile = [10, 12, 20];
            Assert.Equal(afterHostile, SurvivingIndices(dir, prefix, suffix));
            Assert.True(File.Exists(committed));
            Assert.False(File.Exists(Path.Combine(dir, $"{prefix}9{suffix}")));
            Assert.False(File.Exists(Path.Combine(dir, $"{prefix}2{suffix}")));
            foreach (var p in nonMembers) Assert.True(File.Exists(p));

            // A rotation failure never fails the save: the new file is already committed, so the
            // fault leaves it in place, prunes nothing, and surfaces only through onWarning.
            foreach (var f in Directory.GetFiles(dir)) File.Delete(f);
            for (int i = 0; i < 3; i++) WriteSeriesMember(dir, prefix, i, suffix, keep: 10);

            var warnings = new List<string>();
            AtomicFileWriter.RotationFaultInjection = p =>
            {
                if (p.StartsWith(dir, StringComparison.Ordinal)) throw new IOException("injected rotation crash");
            };
            string afterFault;
            try
            {
                afterFault = WriteSeriesMember(dir, prefix, 3, suffix, keep: 1, onWarning: warnings.Add);
            }
            finally { AtomicFileWriter.RotationFaultInjection = null; }

            Assert.True(File.Exists(afterFault));
            int[] unpruned = [0, 1, 2, 3];
            Assert.Equal(unpruned, SurvivingIndices(dir, prefix, suffix));
            Assert.Contains(warnings, w => w.Contains("rotation failed") && w.Contains("injected rotation crash"));
        }
        finally { Directory.Delete(dir, recursive: true); }
    }

    [Fact]
    public void TestAtomicFileWriterCommitAndValidationCoverage()
    {
        var dir = NewScratchDir();
        var target = Path.Combine(dir, "state.bin");
        try
        {
            WriteVia(target, "v1");
            Assert.Equal("v1", File.ReadAllText(target));
            Assert.Empty(Directory.GetFileSystemEntries(dir, ".tmp-*"));

            // Crash in the commit window (after write+flush, before rename): the old content
            // survives and the writer disposes its own staged copy. The hook filters on the
            // scratch dir so concurrent tests elsewhere are unaffected.
            AtomicFileWriter.CommitFaultInjection = p =>
            {
                if (p.StartsWith(dir, StringComparison.Ordinal)) throw new IOException("injected crash");
            };
            try
            {
                Assert.Throws<IOException>(() => WriteVia(target, "v2"));
            }
            finally { AtomicFileWriter.CommitFaultInjection = null; }
            Assert.Equal("v1", File.ReadAllText(target));
            Assert.Empty(Directory.GetFileSystemEntries(dir, ".tmp-*"));

            // A failure inside the content writer behaves the same way.
            Assert.Throws<InvalidOperationException>(() => AtomicFileWriter.WriteFile(
                target, _ => throw new InvalidOperationException("writer boom")));
            Assert.Equal("v1", File.ReadAllText(target));

            // Stale temps (planted here matching ".tmp-<target>-<32-hex-guid>") are swept on the
            // next successful save of the same target — but only precise matches.
            var staleOurs   = Path.Combine(dir, $".tmp-state.bin-{Guid.NewGuid():N}");
            var staleOther  = Path.Combine(dir, $".tmp-other.bin-{Guid.NewGuid():N}");
            var stalePrefix = Path.Combine(dir, $".tmp-state.bin-2-{Guid.NewGuid():N}");
            var staleNonGuid = Path.Combine(dir, ".tmp-state.bin-notahexguidxxxxxxxxxxx");
            string[] survivors = [staleOther, stalePrefix, staleNonGuid];
            File.WriteAllText(staleOurs, "partial");
            foreach (var p in survivors) File.WriteAllText(p, "partial");
            WriteVia(target, "v2");
            Assert.Equal("v2", File.ReadAllText(target));
            Assert.False(File.Exists(staleOurs));
            foreach (var p in survivors) Assert.True(File.Exists(p));

            // A concurrent writer's in-flight temp (held open with a deny-all share) is never
            // swept: the sweep can't acquire it, so it skips rather than destroying live data.
            var liveTemp = Path.Combine(dir, $".tmp-state.bin-{Guid.NewGuid():N}");
            using (new FileStream(liveTemp, FileMode.CreateNew, FileAccess.Write, FileShare.None))
            {
                WriteVia(target, "v3");
                Assert.True(File.Exists(liveTemp));
            }
            Assert.Equal("v3", File.ReadAllText(target));
        }
        finally { Directory.Delete(dir, recursive: true); }

        // Up-front validation fails before any data is written — a fresh directory stays empty.
        var clean = NewScratchDir();
        try
        {
            var missingDir = Path.Combine(clean, "does-not-exist");
            var ex = Assert.Throws<DirectoryNotFoundException>(
                () => WriteVia(Path.Combine(missingDir, "state.bin"), "v1"));
            Assert.Contains("does not exist", ex.Message);
            Assert.False(Directory.Exists(missingDir));

            Assert.Throws<ArgumentException>(() => WriteVia("", "v1"));
            Assert.Throws<ArgumentException>(() => WriteVia("   ", "v1"));
            Assert.Throws<ArgumentException>(() => WriteVia(Path.Combine(clean, ".tmp-state.bin"), "v1"));
            Assert.Throws<ArgumentNullException>(() => AtomicFileWriter.WriteFile(Path.Combine(clean, "s.bin"), null!));
            Assert.Empty(Directory.GetFileSystemEntries(clean));

            Assert.True(AtomicFileWriter.IsTempName(".tmp-run-42"));
            Assert.False(AtomicFileWriter.IsTempName("run-42"));
        }
        finally { Directory.Delete(clean, recursive: true); }
    }

    [Fact]
    public void TestEveryDebugRequestPointProducesItsSnapshot()
    {
        var dir = Path.Combine(Path.GetTempPath(), $"shrk_debugpoints_{Guid.NewGuid():N}");
        try
        {
            var points = Enum.GetValues<GraphCreationPoint>();
            var model = ScalarMultiplyModel.ComputationGraph;
            var hints = new ModelParamList(
                [new KeyValuePair<string, TensorData>(
                    model.ToInternal().Inputs[0].ToString(), TensorData([4L], new float[4]))],
                ModelParamType.InputParam);

            model.ToConcreteArchitecture(hints, new ComputeContext(),
                new DebugRequests(points.Select(p => (p, Path.Combine(dir, $"{p}.cs")))));

            bool Written(GraphCreationPoint p) => new FileInfo(Path.Combine(dir, $"{p}.cs")) is { Exists: true, Length: > 0 };

            Assert.Equal(points, points.Where(Written).ToArray());
        }
        finally { if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true); }
    }

    /// <summary>A struct's list surface reads its definition: the indexer and the enumerator both
    /// walk the definition's fields, so the count must too. A key beyond the definition is not a
    /// field of the struct, and counting it made Count disagree with both.</summary>
    [Fact]
    public void TestAStructCountsTheFieldsItsDefinitionDeclares()
    {
        var def = new TensorStructDef(
            [new TensorStructFieldDef("declared", DataStructure.Tensor, 1, DType.Float32)], "Counting");
        var s = new TensorDataStruct(def,
            [new("declared", Globals.TensorData([1L], [1f])), new("stray", Globals.TensorData([1L], [2f]))]);

        Assert.Equal(1, s.Count);
        Assert.Equal(s.Count, s.Count());
        Assert.Equal(s[0], s.Single());
    }

    /// <summary>A field's value must be the kind its definition declares. Nothing else can check it
    /// afterwards: the consumers disagree — the checkpoint writer refuses a non-tensor field by name,
    /// while binding drops it and fails two layers down on a dictionary lookup.</summary>
    [Fact]
    public void TestAStructFieldValueMustBeTheKindItsDefinitionDeclares()
    {
        var inner = new TensorStructDef(
            [new TensorStructFieldDef("inner", DataStructure.Tensor, 1, DType.Float32)], "Inner");
        var innerValue = new TensorDataStruct(inner, [new("inner", Globals.TensorData([1L], [42f]))]);
        var tensorValue = Globals.TensorData([1L], [1f]);

        TensorStructDef Declaring(DataStructure structure, DType elementType) => new(
            [new TensorStructFieldDef("f", structure, structure == DataStructure.Tensor ? 1 : null, elementType)],
            "Declaring");
        var declaresTensor = Declaring(DataStructure.Tensor, DType.Float32);
        var declaresStruct = Declaring(DataStructure.TensorStruct, DType.GetOrCreateForTensorStruct(inner));

        var declaresOptional = Declaring(DataStructure.Optional, DType.Float32);
        var declaresSequence = Declaring(DataStructure.Sequence, DType.Float32);
        var optionalValue = OptionalTensorData.Some(Globals.TensorData([1L], [3f]));

        IData[] accepted = [tensorValue, innerValue, optionalValue, tensorValue];
        TensorStructDef[] declaring = [declaresTensor, declaresStruct, declaresOptional, declaresOptional];
        Assert.All(declaring.Zip(accepted),
            p => Assert.Equal(1, new TensorDataStruct(p.First, [new("f", p.Second)]).Count));

        static string Refusal(TensorStructDef def, IData value) => Assert.Throws<ArgumentException>(
            () => new TensorDataStruct(def, [new("f", value)])).Message;
        Assert.Contains("Tensor", Refusal(declaresTensor, innerValue));
        Assert.Contains("TensorStruct", Refusal(declaresStruct, tensorValue));
        Assert.Contains("Sequence", Refusal(declaresSequence, tensorValue));
        Assert.Contains("Optional", Refusal(declaresOptional, innerValue));
        Assert.Contains("unsupported value type", Refusal(declaresTensor, null!));
    }
}
