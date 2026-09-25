using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using Microsoft.ML.OnnxRuntime;
using Shorokoo.Core.Factory;
using Shorokoo.Core.Factory.OpsFactories;
using Shorokoo.Core.Interpreter;
using Shorokoo.Core.Backends;
using Shorokoo.OnnxRuntime;
using Shorokoo.Runtime;

namespace Shorokoo.Tests;

/// <summary>
/// Coverage for the small framework utilities the graph-level suites never touch directly: the
/// internal LINQ-ish helpers in <c>Shorokoo.Core.Utils.Extensions</c>, the <see cref="NodeKey"/> /
/// <see cref="TensorKey"/> identity structs, the <see cref="ShorokooException"/> hierarchy, the
/// OpsFactories <see cref="Helpers"/> dtype sets and attribute-type mapping, the
/// <see cref="DefaultBackend"/> deployment-folder discovery and selection policy, the
/// description a live backend answers with and the device assertion built on it, the
/// uninitialised tensor allocation on the backend ABI and the zero-filling default behind it, the
/// <see cref="DeviceMemory"/> settings the CUDA backends map onto ORT's arena options, the
/// reflection that reaches ORT's per-session arena figures and the
/// <see cref="ArenaStatistics"/>/<see cref="RunStatistics"/>/<see cref="NodePlacement"/> surface
/// built on it, the per-run abort token and what both run paths do with one, the typed
/// value-handle conversions, <c>ShapeUtils</c>' argument validation for <c>Reshape</c>'s
/// <c>keepAxes</c>, the <see cref="AtomicFileWriter"/> temp-and-rename commit protocol
/// (crash-window fault injection, stale-temp sweep, retain-last-N rotation), the
/// <see cref="DebugRequests"/> snapshot hook firing at every <see cref="GraphCreationPoint"/>,
/// the public-API shape guard against a <c>params</c> array sitting behind an optional
/// parameter, and the guard that every <c>using</c> the documentation shows names public API.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
[Collection(DeviceMemoryPeak.Name)]
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
        Assert.Contains(DType.Utf8, Helpers.All2);
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
    public void TestDefaultBackendDiscoveryAndSelectionPolicyCoverage()
    {
        // No backend is set explicitly in this suite, so reading Instance exercises the
        // deployment-folder auto-discovery fallback; the platform backend is derived from the
        // running OS, so this holds on Windows and Linux alike.
        var backend = DefaultBackend.Instance;
        Assert.NotNull(backend);
        var name = backend.GetType().Assembly.GetName().Name ?? "";
        Assert.StartsWith(RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "Shorokoo.Win" : "Shorokoo.Linux", name);

        // Discovery is sticky, and the discovered backend actually executes.
        Assert.Same(backend, DefaultBackend.Instance);
        Assert.Equal(5f, OnnxEngine.Eval(Scalar(2f) + Scalar(3f)).As<float32>().AccessMemory()[0]);

        // The candidate policy, driven directly since the suite ships one backend: nothing
        // deployed → no choice, one is taken as-is, several are refused by name, and a backend
        // for another OS is not a candidate at all.
        var cpu = ("Shorokoo.LinuxCPU", false);
        var gpu = ("Shorokoo.LinuxGPU", true);
        Assert.Null(DefaultBackend.SelectBackend([], "deployed in '/app'"));
        Assert.Equal(cpu, DefaultBackend.SelectBackend([cpu], "deployed in '/app'")!.Value);
        Assert.Equal(gpu, DefaultBackend.SelectBackend([gpu], "deployed in '/app'")!.Value);
        var refused = Assert.Throws<InvalidOperationException>(
            () => DefaultBackend.SelectBackend([cpu, gpu], "deployed in '/app'"));
        Assert.Contains("Shorokoo.LinuxCPU (CPU), Shorokoo.LinuxGPU (CUDA)", refused.Message);
        Assert.Contains("deployed in '/app'", refused.Message);

        var windows = RuntimeInformation.IsOSPlatform(OSPlatform.Windows);
        (string, bool)[] thisOsOnly = windows ? [("Shorokoo.WinCPU", false)] : [("Shorokoo.LinuxCPU", false)];
        (string, bool)[] bothDevicesInOrder = windows
            ? [("Shorokoo.WinCPU", false), ("Shorokoo.WinGPU", true)]
            : [("Shorokoo.LinuxCPU", false), ("Shorokoo.LinuxGPU", true)];
        Assert.Equal(thisOsOnly, DefaultBackend.LoadedCandidates(
            ["System.Private.CoreLib", "Shorokoo.WinCPU", "Shorokoo.LinuxCPU"]));
        Assert.Empty(DefaultBackend.LoadedCandidates(["System.Private.CoreLib"]));
        Assert.Equal(bothDevicesInOrder, DefaultBackend.LoadedCandidates(
            windows ? ["Shorokoo.WinGPU", "Shorokoo.WinCPU"] : ["Shorokoo.LinuxGPU", "Shorokoo.LinuxCPU"]));
    }

    [Fact]
    public void TestTheLiveBackendNamesItsDeviceAndCanBeRequired()
    {
        var live = DefaultBackend.Describe();
        Assert.Same(DefaultBackend.Instance, DefaultBackend.Current);
        Assert.StartsWith(
            RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "Shorokoo.Win" : "Shorokoo.Linux", live.Name);
        Assert.Equal(live.Name.EndsWith("GPU", StringComparison.Ordinal) ? ComputeDevice.Cuda : ComputeDevice.Cpu,
            live.Device);
        Assert.Equal(live, ComputeContext.Default.Backend);

        DefaultBackend.RequireDevice(live.Device);
        var other = live.Device == ComputeDevice.Cpu ? ComputeDevice.Cuda : ComputeDevice.Cpu;
        Assert.Contains(live.ToString(), Assert.Throws<InvalidOperationException>(
            () => DefaultBackend.RequireDevice(other)).Message);
        Assert.Throws<ArgumentOutOfRangeException>(() => DefaultBackend.RequireDevice((ComputeDevice)7));
    }

    private static readonly ShorokooTensorElementType[] FixedStrideElementTypes =
    [
        ShorokooTensorElementType.Bool,
        ShorokooTensorElementType.Int8, ShorokooTensorElementType.UInt8,
        ShorokooTensorElementType.Int16, ShorokooTensorElementType.UInt16,
        ShorokooTensorElementType.Float16, ShorokooTensorElementType.BFloat16,
        ShorokooTensorElementType.Int32, ShorokooTensorElementType.UInt32,
        ShorokooTensorElementType.Float,
        ShorokooTensorElementType.Int64, ShorokooTensorElementType.UInt64,
        ShorokooTensorElementType.Double,
    ];

    [Fact]
    public void TestTheFixedStrideTableSizesEveryElementTypeTheByteWisePathsAccept()
    {
        Assert.Equal(
            (int[])[1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 8, 8, 8],
            [.. FixedStrideElementTypes.Select(TensorElementLayout.ElementSizeInBytes)]);

        Assert.Equal(24, TensorElementLayout.ByteCount(ShorokooTensorElementType.Float, [2L, 3L]));
        Assert.Equal(16, TensorElementLayout.ByteCount(ShorokooTensorElementType.Int64, [2L]));
        Assert.Equal(0, TensorElementLayout.ByteCount(ShorokooTensorElementType.Double, [0L, 3L]));
        Assert.Equal(1, TensorElementLayout.ByteCount(ShorokooTensorElementType.Bool, []));

        Assert.Contains("CreateStringTensor", Assert.Throws<NotSupportedException>(
            () => TensorElementLayout.ElementSizeInBytes(ShorokooTensorElementType.String)).Message);
        Assert.Throws<NotSupportedException>(
            () => TensorElementLayout.ElementSizeInBytes(ShorokooTensorElementType.Complex64));
        Assert.Throws<NotSupportedException>(
            () => TensorElementLayout.ByteCount(ShorokooTensorElementType.UInt4, [8L]));
        Assert.Throws<ArgumentNullException>(
            () => TensorElementLayout.ByteCount(ShorokooTensorElementType.Float, null!));
        Assert.Throws<OverflowException>(
            () => TensorElementLayout.ByteCount(ShorokooTensorElementType.Float, [long.MaxValue]));
        Assert.All((long[][])[[-1L], [2L, -1L], [-1L, -1L], [-2L]],
            shape => Assert.Throws<NotSupportedException>(
                () => TensorElementLayout.ByteCount(ShorokooTensorElementType.Float, shape)));
    }

    [Fact]
    public void TestAnUninitializedTensorIsWhatTheCopyingPathBuildsWithoutTheCopy()
    {
        var backend = DefaultBackend.Instance;
        long[][] shapes = [[4L], [2L, 3L], [2L, 1L, 5L]];

        foreach (var elementType in FixedStrideElementTypes)
            foreach (var shape in shapes)
            {
                using var copied = backend.CreateTensorInBackendMemory(
                    elementType, new byte[Elements(shape) * sizeof(double)], shape);
                using var fresh = backend.CreateUninitializedTensorInBackendMemory(elementType, shape);
                Assert.Equal(copied.ElementType, fresh.ElementType);
                Assert.Equal(copied.Shape, fresh.Shape);
                Assert.Equal(
                    copied.GetTensorDataAsSpan<byte>().Length, fresh.GetTensorDataAsSpan<byte>().Length);
            }

        Assert.Equal(
            (float[])[1.5f, -2.5f, 3f],
            Written(backend, ShorokooTensorElementType.Float, [3L], (float[])[1.5f, -2.5f, 3f]));
        Assert.Equal(
            (long[])[1L, -2L], Written(backend, ShorokooTensorElementType.Int64, [1L, 2L], (long[])[1L, -2L]));
        Assert.Equal(
            (byte[])[7, 9, 0], Written(backend, ShorokooTensorElementType.UInt8, [3L], (byte[])[7, 9, 0]));

        Assert.Equal(
            Assert.Throws<NotSupportedException>(() => backend.CreateTensorFromRawBytes(
                ShorokooTensorElementType.String, [], [2L])).Message,
            Assert.Throws<NotSupportedException>(() => backend.CreateUninitializedTensorInBackendMemory(
                ShorokooTensorElementType.String, [2L])).Message);
        Assert.Throws<NotSupportedException>(() => backend.CreateUninitializedTensorInBackendMemory(
            ShorokooTensorElementType.Complex64, [2L]));
        Assert.Throws<ArgumentNullException>(() => backend.CreateUninitializedTensorInBackendMemory(
            ShorokooTensorElementType.Float, null!));
    }

    [Fact]
    public void TestABackendThatDoesNotOverrideTheUninitializedAllocationStillGetsAZeroFilledOne()
    {
        IShorokooBackend defaulting = new ByteWiseOnlyBackend();

        Assert.Equal(new float[6], Zeroed<float>(defaulting, ShorokooTensorElementType.Float, [2L, 3L]));
        Assert.Equal(new long[3], Zeroed<long>(defaulting, ShorokooTensorElementType.Int64, [3L]));
        Assert.Equal(new byte[5], Zeroed<byte>(defaulting, ShorokooTensorElementType.UInt8, [5L]));
        Assert.Equal(new short[4], Zeroed<short>(defaulting, ShorokooTensorElementType.Int16, [2L, 2L]));
        Assert.Equal(new double[2], Zeroed<double>(defaulting, ShorokooTensorElementType.Double, [2L]));

        Assert.Contains("CreateStringTensor", Assert.Throws<NotSupportedException>(
            () => defaulting.CreateUninitializedTensorInBackendMemory(
                ShorokooTensorElementType.String, [2L])).Message);
        Assert.Throws<NotSupportedException>(() => defaulting.CreateUninitializedTensorInBackendMemory(
            ShorokooTensorElementType.Complex64, [2L]));
        Assert.Throws<ArgumentNullException>(() => defaulting.CreateUninitializedTensorInBackendMemory(
            ShorokooTensorElementType.Float, null!));
        Assert.All((long[][])[[-1L], [2L, -1L], [-1L, -1L]],
            shape => Assert.Throws<NotSupportedException>(
                () => defaulting.CreateUninitializedTensorInBackendMemory(
                    ShorokooTensorElementType.Float, shape)));
    }

    private static long Elements(long[] shape)
    {
        var elements = 1L;
        foreach (var dim in shape) elements *= dim;
        return elements;
    }

    private static T[] Written<T>(
        IShorokooBackend backend, ShorokooTensorElementType elementType, long[] shape, T[] values)
        where T : unmanaged
    {
        using var fresh = backend.CreateUninitializedTensorInBackendMemory(elementType, shape);
        values.CopyTo(fresh.GetTensorMutableDataAsSpan<T>());
        return [.. fresh.GetTensorDataAsSpan<T>()];
    }

    private static T[] Zeroed<T>(
        IShorokooBackend backend, ShorokooTensorElementType elementType, long[] shape)
        where T : unmanaged
    {
        using var value = backend.CreateUninitializedTensorInBackendMemory(elementType, shape);
        Assert.Equal(elementType, value.ElementType);
        Assert.Equal(shape, value.Shape);
        return [.. value.GetTensorDataAsSpan<T>()];
    }

    /// <summary>A backend answering only the byte-wise constructor, so what serves everything built
    /// on it is the interface's own default bodies rather than a backend's.</summary>
    private sealed class ByteWiseOnlyBackend : IShorokooBackend
    {
        public BackendDescription Description { get; } = new("byte-wise-only", ComputeDevice.Cpu, null);

        public IShorokooTensorValue CreateTensorFromRawBytes(
            ShorokooTensorElementType elementType, byte[] data, long[] shape)
            => DefaultBackend.Instance.CreateTensorFromRawBytes(elementType, data, shape);

        public IShorokooSession CreateSession(
            ReadOnlyMemory<byte> modelBytes, ShorokooGraphOptimization graphOptimization,
            ShorokooLogSeverity logSeverity, DeviceMemorySettings deviceMemory)
            => throw new NotSupportedException();

        public IShorokooTensorValue CreateTensor<T>(T[] data, long[] shape) where T : unmanaged
            => throw new NotSupportedException();

        public IShorokooTensorValue CreateStringTensor(IReadOnlyList<string> data, long[] shape)
            => throw new NotSupportedException();

        public IShorokooTensorValue CreateSequence(IReadOnlyList<IShorokooTensorValue> values)
            => throw new NotSupportedException();
    }

    [Fact]
    public void TestABackendDescribesItselfOrCannotBeBuiltAtAll()
    {
        Assert.Throws<ArgumentNullException>(() => new BackendDescription(null!, ComputeDevice.Cpu, null));
        Assert.Throws<ArgumentOutOfRangeException>(() => new BackendDescription("X", (ComputeDevice)7, null));
        Assert.Throws<ArgumentOutOfRangeException>(() => new BackendDescription("X", ComputeDevice.Cuda, -1));
        Assert.Throws<ArgumentException>(() => new BackendDescription("X", ComputeDevice.Cuda, null));
        Assert.Throws<ArgumentException>(() => new BackendDescription("X", ComputeDevice.Cpu, 0));
        Assert.Throws<ArgumentException>(() => new BackendDescription("X", ComputeDevice.Other, 0));

        Assert.Equal("X (CUDA device 2)", new BackendDescription("X", ComputeDevice.Cuda, 2).ToString());
        Assert.Equal("X (CPU)", new BackendDescription("X", ComputeDevice.Cpu, null).ToString());
        Assert.Equal("X (Other)", new BackendDescription("X", ComputeDevice.Other, null).ToString());

        var nameless = new BackendDescription("", ComputeDevice.Cpu, null);
        Assert.Equal("", default(BackendDescription).Name);
        Assert.Equal(nameless, default(BackendDescription));
        Assert.Equal(nameless.GetHashCode(), default(BackendDescription).GetHashCode());

        Assert.Equal(new BackendDescription("Shorokoo.Tests", ComputeDevice.Cuda, 3), new CudaBackendProbe(3).Description);
        Assert.Equal(new BackendDescription("Shorokoo.Tests", ComputeDevice.Cpu, null), new CpuBackendProbe().Description);
        Assert.Equal(new BackendDescription("Shorokoo.Tests", ComputeDevice.Other, null), new OtherDeviceBackendProbe().Description);
    }

    [Fact]
    public void TestARequiredDeviceIsRefusedByEveryBackendThatIsNotIt()
    {
        var cpu = new BackendDescription("B", ComputeDevice.Cpu, null);
        var cuda = new BackendDescription("B", ComputeDevice.Cuda, 0);
        var dml = new BackendDescription("B", ComputeDevice.Other, null);
        Assert.Null(DefaultBackend.DeviceRefusal(cpu, ComputeDevice.Cpu));
        Assert.Null(DefaultBackend.DeviceRefusal(cuda, ComputeDevice.Cuda));
        Assert.Null(DefaultBackend.DeviceRefusal(dml, ComputeDevice.Other));
        Assert.Contains("requires a CUDA backend, but B (CPU) is live",
            DefaultBackend.DeviceRefusal(cpu, ComputeDevice.Cuda));
        Assert.Contains("requires a CPU backend, but B (CUDA device 0) is live",
            DefaultBackend.DeviceRefusal(cuda, ComputeDevice.Cpu));
        Assert.Contains("requires a CPU backend, but B (Other) is live",
            DefaultBackend.DeviceRefusal(dml, ComputeDevice.Cpu));
        Assert.Contains("Shorokoo.{WinCPU,WinGPU,LinuxCPU,LinuxGPU}",
            DefaultBackend.DeviceRefusal(dml, ComputeDevice.Cpu));
        var noShippedBackend = DefaultBackend.DeviceRefusal(cpu, ComputeDevice.Other)!;
        Assert.Contains("requires a backend on some other execution provider", noShippedBackend);
        Assert.DoesNotContain("Shorokoo.{WinCPU,WinGPU,LinuxCPU,LinuxGPU}", noShippedBackend);
    }

    [Fact]
    public void TestTheProjectsAModelLibraryReferencesCarryNoBackendOrOnnxRuntime()
    {
        Assert.True(IsBackendOrOnnxRuntime("Microsoft.ML.OnnxRuntime"));
        Assert.True(IsBackendOrOnnxRuntime("Microsoft.ML.OnnxRuntime.Gpu.Linux"));
        Assert.True(IsBackendOrOnnxRuntime("Shorokoo.OnnxRuntime"));
        Assert.True(IsBackendOrOnnxRuntime("Shorokoo.LinuxGPU"));
        Assert.True(IsBackendOrOnnxRuntime("Shorokoo.WinCPU"));
        Assert.False(IsBackendOrOnnxRuntime("Shorokoo.Modules"));
        Assert.False(IsBackendOrOnnxRuntime("Newtonsoft.Json"));

        string[] backendFree = ["src/Shorokoo", "src/Shorokoo.Modules", "src/Shorokoo.Meta"];
        foreach (var project in backendFree)
            Assert.DoesNotContain(RestoredDependencies(project), IsBackendOrOnnxRuntime);
        Assert.Contains(RestoredDependencies("src/Backend/OnnxRuntime/Shorokoo.LinuxCPU"), IsBackendOrOnnxRuntime);
    }

    private static bool IsBackendOrOnnxRuntime(string dependency)
        => dependency.Contains("OnnxRuntime", StringComparison.OrdinalIgnoreCase)
           || Regex.IsMatch(dependency, @"^Shorokoo\.(Win|Linux)(CPU|GPU)$", RegexOptions.IgnoreCase);

    // The restored dependency graph rather than the project XML: it is what NuGet resolved, so it
    // carries transitive packages and anything Directory.Build.props injected, and is blind to how
    // the reference happened to be spelled.
    private static IEnumerable<string> RestoredDependencies(string projectDirectory)
    {
        var assets = Path.Combine(
            RepoRoot(), projectDirectory.Replace('/', Path.DirectorySeparatorChar), "obj", "project.assets.json");
        Assert.True(File.Exists(assets));
        var dependencies = Regex.Matches(File.ReadAllText(assets), @"""([^""/]+)/[^""]*"": \{\s*""type""")
            .Select(m => m.Groups[1].Value).ToList();
        Assert.NotEmpty(dependencies);
        return dependencies;
    }

    private sealed class CudaBackendProbe : OrtBackend
    {
        public CudaBackendProbe(int device) : base(device) { }
    }

    private sealed class CpuBackendProbe : OrtBackend
    {
        public CpuBackendProbe() : base(static (_, _) => { }, ComputeDevice.Cpu, cudaDeviceId: null) { }
    }

    private sealed class OtherDeviceBackendProbe : OrtBackend
    {
        public OtherDeviceBackendProbe() : base(static (_, _) => { }, ComputeDevice.Other, cudaDeviceId: null) { }
    }

    /// <summary>A CPU backend whose execution-provider step records the settings it is handed,
    /// which is where a GPU backend would read the arena budget out of them.</summary>
    private sealed class CapturingBackendProbe : OrtBackend
    {
        public CapturingBackendProbe(List<DeviceMemorySettings> seen)
            : base((_, mem) => seen.Add(mem), ComputeDevice.Cpu, cudaDeviceId: null) { }
    }

    /// <summary>Records the settings each run was handed. No outputs, so both run paths return
    /// nothing and every overload can be driven without a model.</summary>
    private sealed class RunSettingsRecorder : IShorokooSession
    {
        public List<RunSettings> Seen { get; } = [];
        public IReadOnlyList<string> InputNames => [];
        public IReadOnlyList<string> OutputNames => [];

        public IReadOnlyList<IShorokooTensorValue> Run(
            IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
            IReadOnlyList<string> outputNames,
            RunSettings runSettings)
        {
            Seen.Add(runSettings);
            return [];
        }

        public IReadOnlyList<IShorokooTensorValue> RunRetainingOutputs(
            IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
            IReadOnlyList<string> outputNames,
            IReadOnlySet<string> retainedOutputNames,
            RunSettings runSettings)
        {
            Seen.Add(runSettings);
            return [];
        }

        public void Dispose() { }
    }

    [Fact]
    public void TestDeviceMemorySettingsMapOntoTheCudaArenaOptions()
    {
        var uncapped = OrtBackend.CudaProviderOptions(0, null, ArenaExtendStrategy.NextPowerOfTwo);
        Assert.Equal("0", uncapped["device_id"]);
        Assert.Equal("kNextPowerOfTwo", uncapped["arena_extend_strategy"]);
        Assert.False(uncapped.ContainsKey("gpu_mem_limit"));

        var budgeted = OrtBackend.CudaProviderOptions(
            1, 16L * 1024 * 1024 * 1024, ArenaExtendStrategy.SameAsRequested);
        Assert.Equal("1", budgeted["device_id"]);
        Assert.Equal("kSameAsRequested", budgeted["arena_extend_strategy"]);
        Assert.Equal("17179869184", budgeted["gpu_mem_limit"]);
        Assert.Throws<ArgumentOutOfRangeException>(
            () => OrtBackend.CudaProviderOptions(0, null, (ArenaExtendStrategy)7));

        Assert.Equal("gpu:0", OrtBackend.ArenaShrinkageRunConfig(0, shrinkArenaAfterRun: true));
        Assert.Equal("gpu:3", OrtBackend.ArenaShrinkageRunConfig(3, shrinkArenaAfterRun: true));
        Assert.Null(OrtBackend.ArenaShrinkageRunConfig(0, shrinkArenaAfterRun: false));
        Assert.Null(OrtBackend.ArenaShrinkageRunConfig(null, shrinkArenaAfterRun: true));
    }

    /// <summary>
    /// The shipped defaults, and the ORT options a GPU session built with them asks for. The
    /// arena strategy is chosen per session rather than shipped as one value, and ORT is only
    /// ever handed a strategy it has — an unresolved one is refused rather than guessed at.
    /// </summary>
    [Fact]
    public void TestDeviceMemoryDefaultsToAutoAndRejectsAnEmptyBudget()
    {
        Assert.Equal(ArenaExtendStrategy.Auto, DeviceMemorySettings.Default.ArenaExtend);
        Assert.Null(DeviceMemorySettings.Default.LimitBytes);
        Assert.False(RunSettings.Default.ShrinkArenaAfterRun);
        Assert.Equal(DeviceMemorySettings.Default, new ComputeContext().DeviceMemory);
        Assert.Equal(RunSettings.Default, new ComputeContext().RunSettings);

        var shipped = OrtBackend.CudaProviderOptions(
            0,
            DeviceMemorySettings.Default.LimitBytes,
            DeviceMemorySettings.Default.Resolve(reusedAcrossShapes: false).ArenaExtend);
        Assert.Equal("kSameAsRequested", shipped["arena_extend_strategy"]);
        Assert.False(shipped.ContainsKey("gpu_mem_limit"));

        Assert.Throws<ArgumentOutOfRangeException>(
            () => OrtBackend.CudaProviderOptions(0, null, ArenaExtendStrategy.Auto));

        Assert.Throws<ArgumentOutOfRangeException>(() => new DeviceMemorySettings { LimitBytes = 0 });
        Assert.Throws<ArgumentOutOfRangeException>(() => new DeviceMemorySettings { LimitBytes = -1 });
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new DeviceMemorySettings { ArenaExtend = (ArenaExtendStrategy)7 });

        // A refused assignment leaves the shared default as it was: a record cannot be edited in
        // place, so no caller can spoil it for another.
        Assert.Null(DeviceMemorySettings.Default.LimitBytes);
        Assert.Equal(ArenaExtendStrategy.Auto, DeviceMemorySettings.Default.ArenaExtend);

        var capped = DeviceMemorySettings.Default with { LimitBytes = 4096 };
        Assert.Equal(4096L, capped.LimitBytes);
        Assert.Null(DeviceMemorySettings.Default.LimitBytes);
        Assert.Equal(DeviceMemorySettings.Default, capped with { LimitBytes = null });
    }

    /// <summary>
    /// Auto departs from exact-size extension only where Shorokoo knows a session is reused
    /// across differing shapes; a session it knows nothing about keeps exact-size extension
    /// rather than being guessed at. A named strategy is carried through untouched, and the
    /// budget rides along either way.
    /// </summary>
    [Fact]
    public void TestAutoTakesTheDoublingOnlyForASessionKnownToBeReusedAcrossShapes()
    {
        ArenaExtendStrategy Auto(bool reused) => DeviceMemorySettings.Default.Resolve(reused).ArenaExtend;

        Assert.Equal(ArenaExtendStrategy.SameAsRequested, Auto(reused: false));
        Assert.Equal(ArenaExtendStrategy.NextPowerOfTwo, Auto(reused: true));

        var named = new DeviceMemorySettings { ArenaExtend = ArenaExtendStrategy.NextPowerOfTwo, LimitBytes = 4096 };
        Assert.Same(named, named.Resolve(reusedAcrossShapes: false));
        Assert.Same(named, named.Resolve(reusedAcrossShapes: true));

        var budgeted = new DeviceMemorySettings { LimitBytes = 8192 };
        Assert.Equal(
            new DeviceMemorySettings { LimitBytes = 8192, ArenaExtend = ArenaExtendStrategy.SameAsRequested },
            budgeted.Resolve(reusedAcrossShapes: false));
    }

    /// <summary>
    /// The strategy a compiled graph really ends up with. A symbolic graph is not by itself
    /// evidence that shapes vary — an ordinary compiled inference graph is symbolic and is the
    /// case exact-size extension wins widest — so only the caller's own assertion moves it.
    /// </summary>
    [Fact]
    public void TestOnlyAnAssertedShapeReuseMovesACompiledGraphOffExactSizeExtension()
    {
        var x = InputTensor<float32>("x", rank: 1);
        var graph = new InternalComputationGraph([x], [x + x]);
        var ctx = new ComputeContext();

        ArenaExtendStrategy Compiled(IReadOnlyList<long[]?>? dims, bool trainingStep, bool reused)
            => ctx.Compile(graph, dims, trainingStep, reused).DeviceMemory.ArenaExtend;

        long[]?[] pinned = [[4L]];

        Assert.Equal(ArenaExtendStrategy.SameAsRequested, Compiled(pinned, trainingStep: true, reused: false));
        Assert.Equal(ArenaExtendStrategy.SameAsRequested, Compiled(null, trainingStep: true, reused: false));
        Assert.Equal(ArenaExtendStrategy.SameAsRequested, Compiled(null, trainingStep: false, reused: false));
        Assert.Equal(ArenaExtendStrategy.SameAsRequested, ctx.Compile(graph).DeviceMemory.ArenaExtend);
        Assert.Equal(ArenaExtendStrategy.NextPowerOfTwo, Compiled(null, trainingStep: true, reused: true));
    }

    /// <summary>
    /// The settings a session is built with reach the execution-provider step that configures its
    /// arena, already resolved. Nothing else on a box without a card observes an arena at all, so
    /// without this the whole per-session surface could be assembled, reported by
    /// <see cref="CompiledGraph.DeviceMemory"/>, and dropped on the way to ORT with the suite green.
    /// </summary>
    [Fact]
    public void TestASessionsSettingsReachTheExecutionProviderStepResolved()
    {
        var x = InputTensor<float32>("x", rank: 1);
        var proto = FastOnnxModelBuilder.BuildInternalOnnxModel(
            new InternalComputationGraph([x], [x + x]), prepForOnnx: true);
        var model = new MemoryStream();
        ProtoBuf.Serializer.Serialize(model, proto);

        var seen = new List<DeviceMemorySettings>();
        var probe = new CapturingBackendProbe(seen);
        var budget = new DeviceMemorySettings { LimitBytes = 8L << 30, ArenaExtend = ArenaExtendStrategy.NextPowerOfTwo };

        using (probe.CreateSession(model.ToArray(), ShorokooGraphOptimization.EnableAll, ShorokooLogSeverity.Fatal, budget)) { }
        Assert.Equal(budget, Assert.Single(seen));

        seen.Clear();
        using (probe.CreateSession(
            model.ToArray(), ShorokooGraphOptimization.EnableAll, ShorokooLogSeverity.Fatal,
            DeviceMemorySettings.Default.Resolve(reusedAcrossShapes: true))) { }
        Assert.Equal(ArenaExtendStrategy.NextPowerOfTwo, Assert.Single(seen).ArenaExtend);

        Assert.Throws<ArgumentNullException>(() => probe.CreateSession(
            model.ToArray(), ShorokooGraphOptimization.EnableAll, ShorokooLogSeverity.Fatal, null!));
    }

    /// <summary>
    /// The settings belong to the session and the run, not to the process: a context configures
    /// the sessions it compiles from then on, a second context is unaffected, and a graph already
    /// compiled keeps what it was built with. This is what stops a later change reaching a rig
    /// that is already running.
    /// </summary>
    [Fact]
    public void TestSessionAndRunSettingsAreScopedToTheSessionAndTheRunAndNotToTheProcess()
    {
        var budget = new DeviceMemorySettings { LimitBytes = 8L << 30, ArenaExtend = ArenaExtendStrategy.NextPowerOfTwo };
        var shrinking = new RunSettings { ShrinkArenaAfterRun = true };
        var configured = new ComputeContext { DeviceMemory = budget, RunSettings = shrinking };

        Assert.Equal(budget, configured.DeviceMemory);
        Assert.Equal(shrinking, configured.RunSettings);
        Assert.Equal(DeviceMemorySettings.Default, new ComputeContext().DeviceMemory);
        Assert.Equal(RunSettings.Default, new ComputeContext().RunSettings);

        var x = InputTensor<float32>("x", rank: 1);
        var inner = new InternalComputationGraph([x], [x + x]);
        Shorokoo.Core.Graph.RepresentativeInputShapes.Set(inner.FindNode(inner.Inputs[0].FastNodeKey)!, [2L]);
        var graph = new ComputationGraph(inner, GraphKind.ConcreteModel);
        var compiled = configured.Compile(graph);
        Assert.Equal(budget, compiled.DeviceMemory);
        Assert.Equal(shrinking, compiled.DefaultRunSettings);

        // Nothing a caller does afterwards can reach that session: the settings it was built with
        // are its own, and a differently configured context builds a differently configured one.
        // A default context resolves Auto rather than carrying it into the session.
        Assert.Equal(ArenaExtendStrategy.SameAsRequested, new ComputeContext().Compile(graph).DeviceMemory.ArenaExtend);
        Assert.Null(new ComputeContext().Compile(graph).DeviceMemory.LimitBytes);

        Assert.Throws<ArgumentNullException>(() => new ComputeContext { DeviceMemory = null! });
        Assert.Throws<ArgumentNullException>(() => new ComputeContext { RunSettings = null! });
        Assert.Throws<ArgumentNullException>(() => compiled.Run([], null!));
    }

    /// <summary>
    /// Every run entry point runs under what the call named, and under the compiling context's
    /// default when it named nothing. Without this a per-run override could be accepted and
    /// quietly dropped, which no GPU-free test would otherwise notice.
    /// </summary>
    [Fact]
    public void TestEveryRunEntryPointCarriesTheCallsRunSettingsAndOtherwiseTheContextsDefault()
    {
        var shrinking = new RunSettings { ShrinkArenaAfterRun = true };
        var session = new RunSettingsRecorder();
        var context = new ComputeContext();
        var compiled = new CompiledGraph(
            session, context.ResolvedBackend, [], [], ShorokooGraphOptimization.EnableAll,
            DeviceMemorySettings.Default, shrinking, context);

        compiled.Execute();
        compiled.Execute([], RunSettings.Default);
        compiled.Run();
        compiled.Run([], RunSettings.Default);
        compiled.Execute([], []);
        compiled.Execute([], [], RunSettings.Default);

        RunSettings[] expected = [shrinking, RunSettings.Default, shrinking, RunSettings.Default, shrinking, RunSettings.Default];
        Assert.Equal(expected, session.Seen);
        Assert.Equal(shrinking, compiled.DefaultRunSettings);
    }

    /// <summary>
    /// The abort seam, from both ends. A run whose token is already cancelled is refused before
    /// anything is fed or spent — so one ORT would itself have failed never reaches it — and a
    /// token that is never cancelled leaves every run as it was and holds nothing once the run
    /// returns. Without that last part a per-run callback would outlive the options it writes to.
    /// </summary>
    [Fact]
    public void TestACancelledRunIsRefusedBeforeItRunsAndAnUnfiredTokenIsHeldNoLongerThanTheRun()
    {
        var x = InputTensor<float32>("x", rank: 1);
        using var context = new ComputeContext();
        var compiled = context.Compile(new InternalComputationGraph([x], [x + x]));
        float[] values = [1f, 2f, 3f];
        float[] square = [1f, 2f, 3f, 4f];
        var input = TensorData([3L], values);
        var wrongRank = TensorData([2L, 2L], square);
        float[] expected = [2f, 4f, 6f];

        float[] Doubled(NamedModelParam[] outputs)
            => [.. outputs[0].ToTensorData().As<float32>().AccessMemory<float>()];

        Assert.False(RunSettings.Default.CancellationToken.CanBeCanceled);
        Assert.Equal(RunSettings.Default, new RunSettings { CancellationToken = CancellationToken.None });
        Assert.Equal(RunSettings.Default, context.RunSettings);

        using var unfired = new CancellationTokenSource();
        var watched = new RunSettings { CancellationToken = unfired.Token };
        Assert.NotEqual(RunSettings.Default, watched);
        Assert.Equal(expected, Doubled(compiled.Execute([input.Shared()], RunSettings.Default)));
        Assert.Equal(expected, Doubled(compiled.Execute([input.Shared()], watched)));
        Assert.Equal(expected, Doubled(compiled.Execute([input.Shared()], watched)));
        Assert.Equal(expected, Doubled(compiled.Execute([input.Shared()], [false], watched)));
        unfired.Cancel();

        using var cancelled = new CancellationTokenSource();
        cancelled.Cancel();
        var stopped = new RunSettings { CancellationToken = cancelled.Token };
        var refused = Assert.Throws<OperationCanceledException>(() => compiled.Execute([input], stopped));
        Assert.Equal(cancelled.Token, refused.CancellationToken);
        Assert.Null(refused.InnerException);
        Assert.Throws<OperationCanceledException>(() => compiled.Execute([input], [false], stopped));
        Assert.Throws<OperationCanceledException>(() => compiled.Execute([input], [true], stopped));
        Assert.Throws<OperationCanceledException>(() => compiled.Execute([wrongRank], stopped));
        Assert.IsType<OnnxRuntimeException>(
            Record.Exception(() => compiled.Execute([wrongRank], RunSettings.Default)));
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

    private static byte[] DoublingModel()
    {
        var x = InputTensor<float32>("x", rank: 1);
        var proto = FastOnnxModelBuilder.BuildInternalOnnxModel(
            new InternalComputationGraph([x], [x + x]), prepForOnnx: true);
        var model = new MemoryStream();
        ProtoBuf.Serializer.Serialize(model, proto);
        return model.ToArray();
    }

    private static CompiledGraph Doubling(ComputeContext context)
    {
        var x = InputTensor<float32>("x", rank: 1);
        return context.Compile(new InternalComputationGraph([x], [x + x]));
    }

    private static TensorData<float32> ThreeFloats() => TensorData([3L], 1f, 2f, 3f);

    /// <summary>
    /// The arena figures are reached through reflection, so nothing but this says the reflection
    /// still lands where it thinks it does. The hazard it guards is not a missing feature but a
    /// wrong one — a field that moved hands back some other pointer, which the binding then calls
    /// as a function — so it names every step: the internal type holding the one <c>OrtApi</c>, the
    /// field on it, each entry point by name and type, and the nine figures a real session
    /// allocator answers with. ORT's own default allocator implements none of them, which is the
    /// other half of why the allocator has to come from the session.
    /// </summary>
    [Fact]
    public void TestTheOrtArenaStatisticsBindingStillResolvesAndAnswers()
    {
        var holder = typeof(OrtAllocator).Assembly.GetType(OrtArenaStats.ApiHolderTypeName);
        Assert.NotNull(holder);
        var field = holder.GetField(
            OrtArenaStats.ApiFieldName, BindingFlags.Static | BindingFlags.NonPublic | BindingFlags.Public);
        Assert.NotNull(field);
        var api = field.GetValue(null);
        Assert.NotNull(api);
        foreach (var entry in OrtArenaStats.ApiEntryPointNames)
        {
            var pointer = api.GetType().GetField(entry);
            Assert.NotNull(pointer);
            Assert.Equal(typeof(IntPtr), pointer.FieldType);
            Assert.NotEqual(IntPtr.Zero, Assert.IsType<IntPtr>(pointer.GetValue(api)));
        }
        Assert.True(OrtArenaStats.IsBound);

        using var session = new InferenceSession(DoublingModel());
        using var allocator = new OrtAllocator(session, OrtMemoryInfo.DefaultInstance);
        var pairs = OrtArenaStats.ReadRaw(allocator);
        Assert.NotNull(pairs);
        Assert.Equal(OrtArenaStats.StatisticNames.Order(), pairs.Keys.Order());
        Assert.Empty(OrtArenaStats.ReadRaw(OrtAllocator.DefaultInstance)!);

        var figures = Assert.IsType<ArenaStatistics>(OrtArenaStats.Read(allocator));
        Assert.Equal(-1L, figures.LimitBytes);
        Assert.Equal(0L, figures.MaxInUseBytes);
    }

    /// <summary>
    /// A session's own arena, read back through the public surface: zeroed before it has run, and
    /// carrying what the run took afterwards. A backend answering the interface's default reports
    /// nothing rather than zeroes, which is the difference between "no figures" and "no memory".
    /// </summary>
    [Fact]
    public void TestAHostSessionReportsItsOwnArenaButNoPinnedOneAndABackendWithoutOneReportsNothing()
    {
        using var context = new ComputeContext();
        var compiled = Doubling(context);

        var before = Assert.IsType<ArenaStatistics>(compiled.ReadArenaStatistics());
        Assert.Equal(0L, before.MaxInUseBytes);
        Assert.Equal(0L, before.AllocationCount);
        Assert.Equal(-1L, before.LimitBytes);

        compiled.Execute(ThreeFloats());
        var after = Assert.IsType<ArenaStatistics>(compiled.ReadArenaStatistics());
        Assert.True(after.MaxInUseBytes > 0);
        Assert.True(after.AllocationCount > 0);
        Assert.True(after.TotalAllocatedBytes >= after.MaxInUseBytes);
        Assert.True(after.MaxAllocSizeBytes > 0);
        Assert.Null(compiled.ReadPinnedArenaStatistics());

        IShorokooSession unanswering = new RunSettingsRecorder();
        Assert.Null(unanswering.ReadArenaStatistics());
        Assert.Null(unanswering.ReadPinnedArenaStatistics());
        Assert.Null(unanswering.ReadNodePlacement());
        Assert.Equal(SessionOutputPlacement.Unknown, unanswering.OutputPlacement);

        compiled.Dispose();
        Assert.Throws<ObjectDisposedException>(() => compiled.ReadArenaStatistics());
        Assert.Throws<ObjectDisposedException>(() => compiled.ReadPinnedArenaStatistics());
        Assert.Throws<ObjectDisposedException>(() => compiled.ReadNodePlacement());
    }

    /// <summary>
    /// A session's weights come out of the same arena these figures read, so a session is already
    /// holding them before it has run anything and the first run's peak is weights plus whatever
    /// that run added. The record carries the mark the run found, which is what separates the two.
    /// </summary>
    [Fact]
    public void TestASessionHoldsItsWeightsInTheArenaBeforeItsFirstRunAndTheRecordSeparatesThem()
    {
        using var context = new ComputeContext
        {
            Diagnostics = new DiagnosticSettings { CollectRunStatistics = true },
        };
        var compiled = ArenaProbeModels.Weighted(context);

        var built = Assert.IsType<ArenaStatistics>(compiled.ReadArenaStatistics());
        Assert.Equal(ArenaProbeModels.WeightBytes, built.MaxInUseBytes);
        Assert.Equal(ArenaProbeModels.WeightBytes, built.InUseBytes);
        Assert.Equal(ArenaProbeModels.WeightBytes, built.MaxAllocSizeBytes);
        Assert.Equal(1L, built.AllocationCount);
        Assert.Equal(1L, built.ReserveCount);
        Assert.Equal(0L, built.ArenaExtensionCount);

        compiled.Execute(ArenaProbeModels.WeightedInput());
        var run = Assert.Single(context.RunStats.RecentRuns);
        Assert.Equal(ArenaProbeModels.WeightBytes, run.PriorPeakBytes);
        Assert.Equal(MemoryFigureKind.Measured, run.PeakKind);
        Assert.True(run.PeakBytes > run.PriorPeakBytes);
        Assert.True(run.PeakBytes - run.PriorPeakBytes < ArenaProbeModels.WeightBytes / 16);
        Assert.Equal(run.PeakBytes, context.RunStats.PeakBytes);
    }

    [Fact]
    public void TestTheFilledProbeSumsAsManyOnesAsTheShapeItIsFedAsksFor()
    {
        using var context = new ComputeContext();
        var filled = ArenaProbeModels.Filled(context);
        Assert.Equal(1000f, ArenaProbeModels.Sum(filled.Execute(ArenaProbeModels.FilledShape(1000))));
        Assert.Equal(4096f, ArenaProbeModels.Sum(filled.Execute(ArenaProbeModels.FilledShape(4096))));
    }

    /// <summary>
    /// Collection is off until it is asked for, and then every run lands in the aggregates. The
    /// per-run peak says which of the two things it is: the run that pushed the arena's high-water
    /// mark up measured its own peak, and one that stayed under it is bounded by a mark some
    /// earlier run set.
    /// </summary>
    [Fact]
    public void TestRunStatisticsAreOffUntilAskedForAndThenRecordEveryRun()
    {
        using var silent = new ComputeContext();
        Doubling(silent).Execute(ThreeFloats());
        Assert.Equal(RunStatistics.Empty, silent.RunStats);
        Assert.Equal(DiagnosticSettings.Default, silent.Diagnostics);
        Assert.False(DiagnosticSettings.Default.CollectRunStatistics);
        Assert.False(DiagnosticSettings.Default.TraceNodePlacement);
        Assert.Equal(1000, DiagnosticSettings.Default.RecentRunCapacity);

        using var counting = new ComputeContext
        {
            Diagnostics = new DiagnosticSettings { CollectRunStatistics = true },
        };
        var compiled = Doubling(counting);
        for (int run = 0; run < 3; run++) compiled.Execute(ThreeFloats());

        var stats = counting.RunStats;
        Assert.Equal(3L, stats.RunCount);
        Assert.True(stats.PeakBytes > 0);
        Assert.True(stats.ArenaBytes >= stats.PeakBytes);
        Assert.True(stats.LargestAllocationBytes > 0);
        Assert.True(stats.AllocationCount > 0);
        Assert.Equal(0L, stats.ArenaShrinkageCount);
        Assert.Equal([1L, 2L, 3L], stats.RecentRuns.Select(run => run.RunNumber));
        Assert.Equal(MemoryFigureKind.Measured, stats.RecentRuns[0].PeakKind);
        Assert.Equal(0L, stats.RecentRuns[0].PriorPeakBytes);
        Assert.Equal([.. stats.RecentRuns.SkipLast(1).Select(run => run.PeakBytes)],
            stats.RecentRuns.Skip(1).Select(run => run.PriorPeakBytes));
        Assert.Equal(stats.PeakBytes, stats.RecentRuns[2].PeakBytes);
        Assert.Equal(stats.PeakBytes, stats.RecentRuns[2].Arena.MaxInUseBytes);
        Assert.Equal([.. stats.RecentRuns.Select(run => run.PeakBytes).Order()],
            stats.RecentRuns.Select(run => run.PeakBytes));

        // The snapshot is one moment, not a live view of a context that keeps running.
        compiled.Execute(ThreeFloats());
        Assert.Equal(3L, stats.RunCount);
        Assert.Equal(4L, counting.RunStats.RunCount);

        Assert.Throws<ArgumentNullException>(() => new ComputeContext { Diagnostics = null! });
        Assert.Throws<ArgumentOutOfRangeException>(() => new DiagnosticSettings { RecentRunCapacity = -1 });
    }

    /// <summary>
    /// The ring bounds the detail and nothing else: the aggregates are folded as each run finishes,
    /// so they stay exact over every run a context ever made while the retained window holds only
    /// the last N. A capacity of zero keeps no detail and still counts.
    /// </summary>
    [Fact]
    public void TestTheRecentRunRingIsBoundedWhileTheAggregatesStayExactOverEveryRun()
    {
        static ArenaStatistics Arena(long maxInUse, long allocs) =>
            new(0, -1, 16, maxInUse, allocs, allocs, 0, 0, maxInUse * 2);

        static RunStatistics Fold(int capacity, params long[] peaks)
        {
            var collector = new RunStatisticsCollector(capacity);
            long high = 0;
            for (int i = 0; i < peaks.Length; i++)
            {
                var before = Arena(high, i);
                high = Math.Max(high, peaks[i]);
                collector.Record(before, Arena(high, i + 1));
            }
            return collector.Snapshot();
        }

        var bounded = Fold(3, 10, 40, 20, 30, 50);
        Assert.Equal(5L, bounded.RunCount);
        Assert.Equal(50L, bounded.PeakBytes);
        Assert.Equal(100L, bounded.ArenaBytes);
        Assert.Equal(5L, bounded.AllocationCount);
        Assert.Equal(5L, bounded.ArenaExtensionCount);
        Assert.Equal(16L, bounded.LargestAllocationBytes);
        Assert.Equal([3L, 4L, 5L], bounded.RecentRuns.Select(run => run.RunNumber));
        Assert.Equal([40L, 40L, 50L], bounded.RecentRuns.Select(run => run.PeakBytes));
        Assert.Equal([40L, 40L, 40L], bounded.RecentRuns.Select(run => run.PriorPeakBytes));
        MemoryFigureKind[] kinds = [MemoryFigureKind.UpperBound, MemoryFigureKind.UpperBound, MemoryFigureKind.Measured];
        Assert.Equal(kinds, bounded.RecentRuns.Select(run => run.PeakKind));

        var detailless = Fold(0, 10, 40, 20);
        Assert.Equal(3L, detailless.RunCount);
        Assert.Equal(40L, detailless.PeakBytes);
        Assert.Empty(detailless.RecentRuns);

        Assert.Equal(1L, Fold(4, 7).RunCount);
        Assert.Equal(7L, Assert.Single(Fold(4, 7).RecentRuns).PeakBytes);
        Assert.Equal(0L, RunStatistics.Empty.RunCount);
        Assert.Empty(RunStatistics.Empty.RecentRuns);
    }

    /// <summary>
    /// Where a session's outputs land, and — only when asked for — which provider ran each node.
    /// The placement is free and always there; the trace costs the session a profiler for its whole
    /// life, so a context that did not ask gets null rather than an empty trace it might read as
    /// "nothing ran on the host".
    /// </summary>
    [Fact]
    public void TestOutputPlacementIsFreeAndTheNodeTraceOnlyArrivesWhenItIsAskedFor()
    {
        using var plain = new ComputeContext();
        var untraced = Doubling(plain);
        untraced.Execute(ThreeFloats());
        Assert.Equal(SessionOutputPlacement.Host, untraced.OutputPlacement);
        Assert.False(untraced.HasDeviceMemory);
        Assert.Null(untraced.ReadNodePlacement());

        using var traced = new ComputeContext
        {
            Diagnostics = new DiagnosticSettings { TraceNodePlacement = true },
        };
        var compiled = Doubling(traced);
        compiled.Execute(ThreeFloats());
        compiled.Execute(ThreeFloats());

        var placement = Assert.IsType<NodePlacement>(compiled.ReadNodePlacement());
        Assert.NotEmpty(placement.Nodes);
        var share = Assert.Single(placement.Providers);
        Assert.Equal("CPUExecutionProvider", share.Provider);
        Assert.Equal(placement.Nodes.Count, share.NodeCount);
        Assert.Equal(placement.Nodes.Count, placement.NodesOn("CPUExecutionProvider").Count);
        Assert.Empty(placement.NodesOn("CUDAExecutionProvider"));
        Assert.True(share.OutputBytes > 0);
        Assert.Same(placement, compiled.ReadNodePlacement());
    }

    /// <summary>
    /// The grouping, on a shape no CPU-only machine can produce: a graph ORT split across two
    /// providers. Busiest provider first, byte counts summed per provider, and the nodes in
    /// execution order.
    /// </summary>
    [Fact]
    public void TestNodePlacementGroupsEveryNodeUnderTheProviderThatRanIt()
    {
        static NodeExecution Node(string name, string provider, long index, long output)
            => new(name, "Add", provider, index, output * 2, 8, output);

        // Given in the order they ran, with the fused node carrying the high graph index a fusion
        // pass hands out -- sorting by that index is what would move it to the end.
        var placement = new NodePlacement(
        [
            Node("a", "CUDAExecutionProvider", 0, 100),
            Node("c", "CUDAExecutionProvider", 6, 400),
            Node("b", "CPUExecutionProvider", 1, 200),
        ]);

        Assert.Equal(["a", "c", "b"], placement.Nodes.Select(node => node.Name));
        Assert.Equal(["CUDAExecutionProvider", "CPUExecutionProvider"], placement.Providers.Select(p => p.Provider));
        Assert.Equal([2, 1], placement.Providers.Select(p => p.NodeCount));
        Assert.Equal([500L, 200L], placement.Providers.Select(p => p.OutputBytes));
        Assert.Equal([1000L, 400L], placement.Providers.Select(p => p.ActivationBytes));
        Assert.Equal([16L, 8L], placement.Providers.Select(p => p.ParameterBytes));
        Assert.Equal(["b"], placement.NodesOn("CPUExecutionProvider").Select(node => node.Name));
        Assert.Empty(new NodePlacement([]).Providers);
        Assert.Throws<ArgumentNullException>(() => new NodePlacement(null!));
        Assert.Throws<ArgumentNullException>(() => placement.NodesOn(null!));
    }

    /// <summary>
    /// The profiler's output prefix is read when profiling is switched on and a later change is
    /// ignored, so a session built to trace its nodes has to set the prefix first or write its
    /// profile into whatever directory the program happens to be running from — one stray file per
    /// session, under ORT's own default name, that nothing then deletes.
    /// </summary>
    [Fact]
    public void TestATracedSessionWritesItsProfileWhereItWasToldAndNotIntoTheWorkingDirectory()
    {
        string[] Strays() => [.. Directory.GetFiles(
            Directory.GetCurrentDirectory(), "onnxruntime_profile_*.json").Order()];
        var before = Strays();
        using (var traced = new ComputeContext
        {
            Diagnostics = new DiagnosticSettings { TraceNodePlacement = true },
        })
        {
            var compiled = Doubling(traced);
            compiled.Execute(ThreeFloats());
            Assert.NotNull(compiled.ReadNodePlacement());
        }
        Assert.Equal(before, Strays());

        using var options = new SessionOptions();
        var directory = Path.Combine(Path.GetTempPath(), "shorokoo-profile-order-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            options.ProfileOutputPathPrefix = Path.Combine(directory, "profile");
            options.EnableProfiling = true;
            using var session = new InferenceSession(DoublingModel(), options);
            Assert.StartsWith(directory, session.EndProfiling(), StringComparison.Ordinal);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    /// <summary>
    /// The settings are reachable from a GPU session, which no test on a CPU box can observe by
    /// running one. What it can observe is that the product still calls the wiring: a backend that
    /// stopped passing its device id, stopped carrying the session's
    /// <see cref="DeviceMemorySettings"/> into the provider options, or stopped putting the
    /// shrinkage entry on its run options would leave every setting dead with every other test
    /// still green. It also pins where those values come from — the parameter the session or run
    /// was given, never process-wide state.
    /// </summary>
    [Fact]
    public void TestTheGpuBackendsStillCarryTheDeviceMemorySettingsIntoOrt()
    {
        var backend = Path.Combine(ProductSourceRoot(), "Backend", "OnnxRuntime");
        string Source(params string[] parts) =>
            StripCommentsAndStrings(File.ReadAllText(Path.Combine(backend, Path.Combine(parts))));

        string[] gpuFactories = ["Shorokoo.LinuxGPU/LinuxGpuBackend.cs", "Shorokoo.WinGPU/WinGpuBackend.cs"];
        foreach (var gpu in gpuFactories)
            Assert.Matches(@"base\s*\(\s*cudaDeviceId\s*:\s*0\s*\)", Source(gpu.Split('/')));
        string[] cpuFactories = ["Shorokoo.LinuxCPU/LinuxCpuBackend.cs", "Shorokoo.WinCPU/WinCpuBackend.cs"];
        foreach (var cpu in cpuFactories)
            Assert.DoesNotMatch(@"\bbase\s*\(", Source(cpu.Split('/')));

        var source = Source("Shorokoo.OnnxRuntime", "OrtBackend.cs");
        Assert.Matches(@"protected\s+OrtBackend\s*\(\s*\)\s*:\s*this\s*\([^;]*cudaDeviceId\s*:\s*null", source);
        // Trailing [,)] rather than a closing paren: what this pins is that the device id still
        // reaches the session, not how many other things travel with it.
        Assert.Matches(@"new\s+OrtSession\s*\(\s*session\s*,\s*_cudaDeviceId\s*[,)]", source);
        Assert.Matches(@"AppendExecutionProvider_CUDA\s*\(\s*cuda\s*\)", source);
        Assert.Matches(@"CudaProviderOptions\s*\(\s*deviceId\s*,\s*deviceMemory\.LimitBytes\s*,\s*deviceMemory\.ArenaExtend\s*\)", source);
        Assert.Matches(@"_configureExecutionProvider\s*\(\s*options\s*,\s*deviceMemory\s*\)", source);

        var context = StripCommentsAndStrings(File.ReadAllText(
            Path.Combine(ProductSourceRoot(), "Shorokoo", "Core", "ComputeContext.cs")));
        Assert.Matches(@"BuildSession\s*\(\s*backend\s*,\s*modelData\s*,\s*optimization\s*,\s*deviceMemory\s*[,)]", context);
        Assert.Matches(@"backend\.CreateSession\s*\(\s*modelData\s*,\s*optimization\s*,\s*ShorokooLogSeverity\.Fatal\s*,\s*deviceMemory\s*,", context);
        Assert.Matches(@"DeviceMemory\.Resolve\s*\(\s*reusedAcrossShapes\s*\)", context);

        var session = Source("Shorokoo.OnnxRuntime", "OrtSession.cs");
        Assert.Contains("memory.enable_memory_arena_shrinkage", File.ReadAllText(
            Path.Combine(backend, "Shorokoo.OnnxRuntime", "OrtSession.cs")));
        Assert.Matches(@"ArenaShrinkageRunConfig\s*\(\s*_cudaDeviceId\s*,\s*runSettings\.ShrinkArenaAfterRun\s*\)", session);
        Assert.Matches(@"AddRunConfigEntry\s*\(", session);

        // Every path that runs the session has to apply it, not just one: the retaining path is
        // the loop a GPU user is steered into, and it is where an unbounded arena costs most.
        var runPaths = Regex.Matches(session, @"_session\s*\.\s*Run\w*\s*\(").Count;
        Assert.Equal(runPaths, Regex.Matches(session, @"ConfigureRun\s*\(\s*runOptions\s*,\s*runSettings\s*\)").Count);
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

    // One root namespace per per-platform backend package. Only the running platform's assembly is
    // in the test output, so the rest cannot be reflected over and are named here instead.
    private static readonly string[] PlatformBackendNamespaces =
        ["Shorokoo.WinCPU", "Shorokoo.WinGPU", "Shorokoo.LinuxCPU", "Shorokoo.LinuxGPU"];

    /// <summary>
    /// Every <c>using Shorokoo…;</c> the documentation shows must name a namespace carrying public
    /// API — a public type, for <c>using static</c>. The compiler accepts an internals-only
    /// namespace, so only a test catches an import that resolves, reads as documentation, and hands
    /// the reader nothing it can call.
    /// </summary>
    [Fact]
    public void TestEveryDocumentationUsingNamesPublicApi()
    {
        Type[] exported = [.. ShippedAssemblies().SelectMany(a => a.GetExportedTypes())];
        var namespaces = exported.Select(t => t.Namespace).OfType<string>()
            .Concat(PlatformBackendNamespaces).ToHashSet(StringComparer.Ordinal);
        var types = exported.Select(t => t.FullName?.Replace('+', '.')).OfType<string>()
            .ToHashSet(StringComparer.Ordinal);

        var usings = Directory.GetFiles(Path.Combine(RepoRoot(), "Documentation"), "*.md")
            .SelectMany(f => Regex.Matches(File.ReadAllText(f), @"(?<![\w.])using (static )?(Shorokoo[\w.]*)\s*;")
                .Select(m => (At: $"{Path.GetFileName(f)}: {m.Value}", Static: m.Groups[1].Success, Name: m.Groups[2].Value)))
            .Distinct()
            .ToArray();

        Assert.True(usings.Length >= 10);
        Assert.Empty(usings.Where(u => !(u.Static ? types : namespaces).Contains(u.Name)).Select(u => u.At));
    }

    /// <summary>
    /// A member the documentation spells out — <c>NN.Conv</c>, <c>OnnxOp.Attention</c> — must exist
    /// on a public type of that name. Citing one that never existed sends the reader looking for an
    /// API to call, and no other test in the suite reads the prose.
    /// </summary>
    [Fact]
    public void TestEveryDocumentationMemberReferenceExists()
    {
        var byName = ShippedAssemblies().SelectMany(a => a.GetExportedTypes())
            .Where(t => t.Name is "Ops" or "OnnxOp" or "NN")
            .ToLookup(t => t.Name, StringComparer.Ordinal);

        var cited = Directory.GetFiles(Path.Combine(RepoRoot(), "Documentation"), "*.md")
            .SelectMany(f => Regex.Matches(File.ReadAllText(f), @"`(Ops|OnnxOp|NN)\.([A-Za-z0-9_]+)`")
                .Select(m => (At: $"{Path.GetFileName(f)}: {m.Value}", Type: m.Groups[1].Value, Member: m.Groups[2].Value)))
            .Distinct()
            .ToArray();

        Assert.True(cited.Length >= 10);
        Assert.Empty(cited.Where(c => !byName[c.Type].Any(t => t.GetMember(c.Member).Length != 0)).Select(c => c.At));
    }

    private static Assembly[] ShippedAssemblies() =>
        [.. Directory.EnumerateFiles(AppContext.BaseDirectory, "Shorokoo*.dll")
            .Where(f => Path.GetFileName(f) is not ("Shorokoo.Tests.dll" or "Shorokoo.CodeGen.dll"))
            .Select(Assembly.LoadFrom)];

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
    // The leading `\.` is a real limitation, not an oversight: it keys on a receiver, so a span
    // taken through an implicit `this` inside TensorData itself -- `write(AccessModifiableMemory
    // <V>())` in WriteMemory -- is invisible to this guard. Relaxing the dot matches every
    // declaration of those members too. Those helpers keep their tensor alive by convention and
    // by review; removing a GC.KeepAlive(this) from one of them leaves the suite green.
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
        var shipped = ShippedAssemblies();
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

    /// <summary>
    /// Every <c>RunOptions</c> the product builds is armed to abort. ONNX Runtime reads its
    /// terminate flag before each node and a run's options are otherwise a local no other thread
    /// can reach, so an unarmed one is a run nothing can stop — and nothing behavioural notices,
    /// because a lease makes a deliberate delete slow rather than unsafe. Deleting the
    /// registration therefore leaves the whole suite green, which is what this is here for.
    /// </summary>
    [Fact]
    public void TestEveryRunOptionsIsArmedToAbortAndTheGuardStillDetectsEveryEvasion()
    {
        var sources = ProductSources();
        Assert.Contains(sources, s => RunOptionsConstruction.IsMatch(StripCommentsAndStrings(s)));
        Assert.Empty(sources.SelectMany(UnarmedRunOptions));

        string[] mustFlag =
        [
            "class C { void M() { using var o = new RunOptions(); _s.Run(o, i, n); } }",
            "class C { void M() { using var o = new RunOptions(); using var a = AbortWhenCancelled(other, t); } }",
            "class C { void M() { using var o = new RunOptions(); } void N() { using var a = AbortWhenCancelled(o, t); } }",
            "class C { void M() { _s.Run(new RunOptions(), i, n); } }",
            "class C { void M() { using var o = new RunOptions(); } /* AbortWhenCancelled(o, t) */ }",
        ];
        string[] mustNotFlag =
        [
            "class C { void M() { using var o = new RunOptions(); using var a = AbortWhenCancelled(o, t); } }",
            "class C { void M() { using RunOptions o = new RunOptions(); using var a = AbortWhenCancelled(o, token); } }",
            "class C { void M() { using var o = new RunOptions(); if (c) { using var a = AbortWhenCancelled(o, t); } } }",
        ];
        Assert.All(mustFlag, s => Assert.NotEmpty(UnarmedRunOptions(s)));
        Assert.All(mustNotFlag, s => Assert.Empty(UnarmedRunOptions(s)));
    }

    private static readonly Regex RunOptionsConstruction = new(
        @"new\s+RunOptions\s*\(", RegexOptions.Compiled);

    // Armed means the registration names THIS options object and sits in the same member: one
    // naming another local writes the flag on options no run is using, and one in a neighbouring
    // member never runs for this construction at all.
    private static string[] UnarmedRunOptions(string source)
    {
        var code = StripCommentsAndStrings(source);
        var flagged = new List<string>();
        foreach (Match m in RunOptionsConstruction.Matches(code))
        {
            var named = Regex.Match(code[..m.Index], @"(\w+)\s*=\s*$");
            var member = code[m.Index..MemberEndFrom(code, m.Index)];
            if (named.Success && Regex.IsMatch(
                    member,
                    @"AbortWhenCancelled\s*\(\s*" + Regex.Escape(named.Groups[1].Value) + @"\s*,"))
                continue;
            flagged.Add(member[..Math.Min(member.Length, 80)].Trim());
        }
        return [.. flagged];
    }

    /// <summary>
    /// Every call into ONNX Runtime through an <c>OrtValue</c> keeps the value alive across it.
    /// ORT takes the value as a bare handle, so the JIT retires the wrapper at the handle read —
    /// before the native call starts — and <c>OrtValue</c> is a plain class whose ordinary
    /// finalizer releases the native value, so a collection on any thread during the call frees it
    /// underneath. Nothing behavioural notices: the callers happen to hold these values reachable
    /// today, which is safety by reachability rather than by construction, and a lifetime changed
    /// anywhere above them takes it away silently and only in Release.
    /// </summary>
    [Fact]
    public void TestEveryNativeCallThroughAnOrtValueKeepsItAliveAndTheGuardStillDetectsEveryEvasion()
    {
        var sources = ProductSources();
        Assert.Contains(sources, s => OrtValueCall.IsMatch(StripCommentsAndStrings(s)));
        Assert.Empty(sources.SelectMany(OrtValuesUsedWithoutKeepingThemAlive));

        string[] mustFlag =
        [
            "class C { bool M() { using var i = Inner.GetTensorMemoryInfo(); return i.Name == \"Cpu\"; } }",
            "class C { int M() => (int)Inner.GetTensorTypeAndShape().ElementDataType; }",
            "class C { long[] M() => Inner.GetTensorTypeAndShape().Shape; }",
            "class C { void M() { Inner.Dispose(); } }",
            "class C { Span<T> M() => Cast<U, T>(Inner.GetTensorDataAsSpan<U>()); }",
            "class C { int M() { var c = Inner.GetValueCount(); return c; } }",
            "class C { void M() { GC.KeepAlive(Inner); var c = Inner.GetValueCount(); } }",
            "class C { void M() { var s = ort.Inner.GetTensorMutableRawData(); Use(s); } }",
            "class C { void M() { GC.KeepAlive(ort); } void N() { var s = ort.Inner.GetTensorMutableRawData(); } }",
            "class C { bool M() { using var i = Inner.GetTensorMemoryInfo(); GC.KeepAlive(Inner); return i.Name == \"Cpu\"; } }",
        ];
        string[] mustNotFlag =
        [
            "class C { int M() { var c = Inner.GetValueCount(); GC.KeepAlive(Inner); return c; } }",
            "class C { ReadOnlySpan<T> M() => Inner.GetTensorDataAsSpan<T>(); }",
            "class C { ReadOnlySpan<T> M() { return Inner.GetTensorDataAsSpan<T>(); } }",
            "class C { void M() { var s = ort.Inner.GetTensorMutableRawData(); Use(s); GC.KeepAlive(ort); } }",
            "class C { void M() { if (c) { var s = Inner.GetValue(0); } GC.KeepAlive(Inner); } }",
            "class C { void M() { foreach (var v in xs) inner.Add(((OrtTensorValue)v).Inner); } }",
            "class C { bool M() { using var i = Inner.GetTensorMemoryInfo(); var h = i.Name == \"Cpu\"; GC.KeepAlive(Inner); return h; } }",
        ];
        Assert.All(mustFlag, s => Assert.NotEmpty(OrtValuesUsedWithoutKeepingThemAlive(s)));
        Assert.All(mustNotFlag, s => Assert.Empty(OrtValuesUsedWithoutKeepingThemAlive(s)));
    }

    [Fact]
    public void TestAReleasedRuntimeValueRefusesEveryReadRatherThanReadingFreedMemory()
    {
        var value = DefaultBackend.Instance.CreateTensor<float>([1f, 2f], [2L]);
        value.Dispose();
        Action[] reads =
        [
            () => _ = value.ValueType, () => _ = value.ElementType, () => _ = value.Shape,
            () => _ = value.IsHostAccessible, () => value.GetTensorDataAsSpan<float>(),
            () => value.GetTensorMutableDataAsSpan<float>(), () => value.GetStringTensorData(),
            () => value.GetValueCount(), () => value.GetValue(0), () => value.GetSequenceElementType(),
        ];

        Assert.All(reads, read => Assert.Throws<ObjectDisposedException>(read));
        value.Dispose();
    }

    [Fact]
    public void TestASequenceReleasesEveryValueItIsHandedWhetherItIsBuiltOrRefused()
    {
        var backend = DefaultBackend.Instance;
        IShorokooTensorValue Pair() => backend.CreateTensor<float>([1f, 2f], [2L]);
        static bool Released(IShorokooTensorValue value)
        {
            try { _ = ((OrtTensorValue)value).Inner; return false; }
            catch (ObjectDisposedException) { return true; }
        }

        IShorokooTensorValue[] built = [Pair(), Pair()];
        using (var sequence = backend.CreateSequence(built))
            Assert.Equal([1f, 2f], sequence.GetValue(1).GetTensorDataAsSpan<float>().ToArray());
        Assert.All(built, value => Assert.True(Released(value)));

        IShorokooTensorValue[] mixed = [Pair(), backend.CreateTensor<long>([1L], [1L])];
        Assert.Throws<OnnxRuntimeException>(() => backend.CreateSequence(mixed));
        Assert.All(mixed, value => Assert.True(Released(value)));

        var foreign = new ForeignValue();
        IShorokooTensorValue[] beside = [Pair(), foreign, Pair()];
        Assert.Throws<InvalidCastException>(() => backend.CreateSequence(beside));
        Assert.True(Released(beside[0]) && foreign.Disposed && Released(beside[2]));

        IShorokooTensorValue[] afterARelease = [Pair(), Pair()];
        afterARelease[0].Dispose();
        Assert.Throws<ObjectDisposedException>(() => backend.CreateSequence(afterARelease));
        Assert.True(Released(afterARelease[1]));
    }

    [Fact]
    public void TestABackendThatBuildsNoSequenceReleasesTheValuesItIsHandedForOne()
    {
        foreach (var backend in (IShorokooBackend[])[HostBackend.Instance, UnrecordedBackend.Instance])
        {
            var handed = new ForeignValue();
            Assert.Throws<NotSupportedException>(() => backend.CreateSequence([handed]));
            Assert.True(handed.Disposed);
        }
    }

    /// <summary>A value no backend made, which a sequence of the ONNX Runtime backend cannot
    /// hold.</summary>
    private sealed class ForeignValue : IShorokooTensorValue
    {
        internal bool Disposed { get; private set; }

        public ShorokooOnnxValueType ValueType => ShorokooOnnxValueType.Tensor;
        public ShorokooTensorElementType ElementType => ShorokooTensorElementType.Float;
        public long[] Shape => [2L];
        public ReadOnlySpan<T> GetTensorDataAsSpan<T>() where T : unmanaged => throw new NotSupportedException();
        public Span<T> GetTensorMutableDataAsSpan<T>() where T : unmanaged => throw new NotSupportedException();
        public IReadOnlyList<string> GetStringTensorData() => throw new NotSupportedException();
        public int GetValueCount() => throw new NotSupportedException();
        public IShorokooTensorValue GetValue(int index) => throw new NotSupportedException();
        public ShorokooTensorElementType GetSequenceElementType() => throw new NotSupportedException();
        public void Dispose() => Disposed = true;
    }

    // A call made on the OrtValue a wrapper holds, reached bare (`Inner.X()`) or through the
    // wrapper (`ort.Inner.X()`). The receiver root is what has to stay reachable: rooting the
    // wrapper roots the value it holds, which is why both spellings are read the same way.
    private static readonly Regex OrtValueCall = new(
        @"(?:\b([A-Za-z_]\w*)\s*\.\s*)?\bInner\s*\.\s*\w+\s*(?:<[^<>()]*>)?\s*\(",
        RegexOptions.Compiled);

    private static string[] OrtValuesUsedWithoutKeepingThemAlive(string source)
    {
        var code = StripCommentsAndStrings(source);
        var flagged = new List<string>();
        foreach (Match m in OrtValueCall.Matches(code))
        {
            var before = code[(code.LastIndexOfAny([';', '{', '}'], m.Index) + 1)..m.Index];
            // Handed straight back, so the caller owns the lifetime from there. The call has to BE
            // the return: anything after its closing paren reads the result here, with the value
            // already retired at the handle.
            if (Regex.IsMatch(before, @"(\breturn\b|=>)\s*$") && ReturnsTheHandleItself(code, m))
                continue;
            var name = m.Groups[1].Success ? m.Groups[1].Value : "Inner";
            if (!RootedAfter(code, LastUseOfResult(code, m), name)) flagged.Add(m.Value.Trim());
        }
        return [.. flagged];
    }

    // Where the rooting has to reach: past the call, and past every later read of a RESOURCE the
    // call handed back. Some of what ORT returns is borrowed rather than owned --
    // GetTensorMemoryInfo gives a non-owning pointer into the value's own state -- so each read of
    // it is another native read through the value, and a keep-alive before the last of them roots
    // nothing that matters. Measuring from the call alone accepted exactly that shape.
    //
    // `using` is the discriminator, and it is the right one: it marks the results that are
    // resources rather than copies. A plain `var c = Inner.GetValueCount()` hands back an int, and
    // reading c later goes nowhere near the value, so a keep-alive straight after that call is
    // correctly placed.
    private static int LastUseOfResult(string code, Match m)
    {
        int after = m.Index + m.Length;
        var before = code[(code.LastIndexOfAny([';', '{', '}'], m.Index) + 1)..m.Index];
        var bound = Regex.Match(
            before, @"\busing\s+(?:\bvar\b|[\w<>\[\],]+)\s+([A-Za-z_]\w*)\s*=\s*$");
        if (!bound.Success) return after;
        int limit = MemberEndFrom(code, after);
        if (limit <= after) return after;
        int last = after;
        foreach (Match use in Regex.Matches(
            code[after..limit], @"\b" + Regex.Escape(bound.Groups[1].Value) + @"\b"))
            last = after + use.Index + use.Length;
        return last;
    }

    /// <summary>
    /// The predicate two signals rest on: whether a graph is reported as having partly run on the
    /// host, and whether a tensor may be an element of a sequence. Both invert if a pinned host
    /// arena stops counting as host, and neither says so from a machine without a card.
    /// </summary>
    [Fact]
    public void TestThePinnedHostArenasCountAsHostMemoryAndTheProvidersOwnDoesNot()
    {
        Assert.True(OrtTensorValue.IsHostAllocator("Cpu"));
        Assert.True(OrtTensorValue.IsHostAllocator("CudaPinned"));
        Assert.True(OrtTensorValue.IsHostAllocator("HipPinned"));
        Assert.False(OrtTensorValue.IsHostAllocator("Cuda"));
        Assert.False(OrtTensorValue.IsHostAllocator("Hip"));
        Assert.False(OrtTensorValue.IsHostAllocator(null));
        Assert.False(OrtTensorValue.IsHostAllocator(""));
        Assert.False(OrtTensorValue.IsHostAllocator("cpu"));
    }

    private static string[] ProductSources() =>
        [.. Directory
            .EnumerateFiles(ProductSourceRoot(), "*.cs", SearchOption.AllDirectories)
            .Where(f => !f.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}") &&
                        !f.Contains($"{Path.DirectorySeparatorChar}bin{Path.DirectorySeparatorChar}"))
            .Select(File.ReadAllText)];

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

    // ---- AllocationFailureReport: host/device classification and the limit-vs-card discriminator ----

    // The two failures Shorokoo/Shorokoo#330 and #332 report as indistinguishable, verbatim.
    private const string ArenaFailure =
        "[ErrorCode:Fail] E:\\_work\\1\\s\\onnxruntime\\core\\framework\\bfc_arena.cc:358 "
        + "onnxruntime::BFCArena::AllocateRawInternal Failed to allocate memory for requested buffer of size 2359296";
    private const string HostFailure = "[ErrorCode:Fail] bad allocation";

    private static ProcessMemoryFacts Facts(long commitGiB, long limitGiB) => new(
        WorkingSetBytes: commitGiB * (1L << 30),
        CommitBytes: commitGiB * (1L << 30),
        ManagedHeapBytes: 3L << 30,
        LimitBytes: limitGiB * (1L << 30),
        LimitIsExplicit: true);

    private static string Report(
        string failure, ProcessMemoryFacts facts, string backend, DeviceMemoryReading? device = null) =>
        AllocationFailureReport.Render(
            "the training step at step 1",
            AllocationFailureReport.Classify(
                new InvalidOperationException(failure), AllocationFailureReport.IsGpuBackend(backend)),
            new DeviceFacts(AllocationFailureReport.IsGpuBackend(backend), device, null, backend),
            [new TensorInventorySection("trainable parameters",
                [new TensorInventoryEntry("wte", "Float32", [50257L, 384L], 50257L * 384 * 4)])],
            facts,
            failure);

    [Fact]
    public void TestAllocationFailuresAreRecognizedAndClassifiedByPool()
    {
        Assert.True(AllocationFailureReport.IsAllocationFailure(new InvalidOperationException(ArenaFailure)));
        Assert.True(AllocationFailureReport.IsAllocationFailure(new InvalidOperationException(HostFailure)));
        Assert.True(AllocationFailureReport.IsAllocationFailure(new OutOfMemoryException()));
        Assert.True(AllocationFailureReport.IsAllocationFailure(
            new InvalidOperationException("outer", new InvalidOperationException(ArenaFailure))));
        Assert.True(AllocationFailureReport.IsAllocationFailure(
            new InvalidOperationException("CUDA_ERROR_OUT_OF_MEMORY")));

        Assert.False(AllocationFailureReport.IsAllocationFailure(
            new InvalidOperationException("[ErrorCode:InvalidGraph] Node () Op (Foo) is not supported")));
        Assert.False(AllocationFailureReport.IsAllocationFailure(new InvalidOperationException("bad input shape")));

        Assert.Equal(AllocationPool.Device,
            AllocationFailureReport.Classify(new InvalidOperationException(ArenaFailure), gpuBackend: true));
        Assert.Equal(AllocationPool.Device, AllocationFailureReport.Classify(
            new InvalidOperationException(HostFailure, new InvalidOperationException(ArenaFailure)),
            gpuBackend: true));
        Assert.Equal(AllocationPool.Host,
            AllocationFailureReport.Classify(new InvalidOperationException(HostFailure), gpuBackend: true));
        Assert.Equal(AllocationPool.Host,
            AllocationFailureReport.Classify(new OutOfMemoryException(), gpuBackend: true));
        Assert.Equal(AllocationPool.Host,
            AllocationFailureReport.Classify(new InvalidOperationException("something else"), gpuBackend: false));
        Assert.Equal(AllocationPool.Unknown,
            AllocationFailureReport.Classify(new InvalidOperationException("something else"), gpuBackend: true));

        Assert.True(AllocationFailureReport.IsGpuBackend("Shorokoo.LinuxGPU"));
        Assert.True(AllocationFailureReport.IsGpuBackend("Shorokoo.WinGPU"));
        Assert.False(AllocationFailureReport.IsGpuBackend("Shorokoo.LinuxCPU"));
        Assert.False(AllocationFailureReport.IsGpuBackend(null));
    }

    [Fact]
    public void TestAnArenaFailureOnASessionWithNoDeviceMemoryIsHostNotAnAbsentAccelerator()
    {
        Assert.Equal(AllocationPool.Host,
            AllocationFailureReport.Classify(new InvalidOperationException(ArenaFailure), gpuBackend: false));
        Assert.Equal(AllocationPool.Device,
            AllocationFailureReport.Classify(new InvalidOperationException(ArenaFailure), gpuBackend: true));

        var cpu = Report(ArenaFailure, Facts(12, 128), "Shorokoo.LinuxCPU");
        Assert.Contains("HOST memory", cpu);
        Assert.DoesNotContain("DEVICE", cpu);
        Assert.DoesNotContain("accelerator", cpu);
    }

    [Fact]
    public void TestAHostLimitAndAFullDeviceNoLongerReadTheSame()
    {
        var capped = Report(ArenaFailure, Facts(commitGiB: 27, limitGiB: 28), "Shorokoo.LinuxGPU");
        var roomy = Report(ArenaFailure, Facts(commitGiB: 12, limitGiB: 128), "Shorokoo.LinuxGPU");
        Assert.NotEqual(capped, roomy);

        Assert.Contains("DEVICE memory", capped);
        Assert.Contains("28 GiB", capped);
        Assert.Contains("96% used", capped);
        Assert.Contains("backed by system commit", capped);
        Assert.Contains("re-run with it raised or removed", capped);

        Assert.Contains("DEVICE memory", roomy);
        Assert.DoesNotContain("backed by system commit", roomy);
        Assert.Contains("well inside its host memory limit", roomy);

        var host = Report(HostFailure, Facts(commitGiB: 27, limitGiB: 28), "Shorokoo.LinuxGPU");
        Assert.Contains("HOST memory", host);
        Assert.DoesNotContain("The failing allocation was for DEVICE memory", host);
        Assert.NotEqual(capped, host);

        string[] reports = [capped, roomy, host];
        foreach (var report in reports)
        {
            Assert.Contains("73.62 MiB", report);
            Assert.Contains("Shorokoo.LinuxGPU", report);
            Assert.Contains("Underlying failure:", report);
            Assert.Contains("the training step at step 1", report);
        }

        Assert.Contains("no memory limit could be read",
            AllocationFailureReport.Render("the training step at step 0", AllocationPool.Host,
                new DeviceFacts(false, null, null, null), [],
                new ProcessMemoryFacts(null, null, 0, null, false), "boom"));
    }

    [Fact]
    public void TestAnUnknownTensorSizeLeavesTheInventoryTotalUnknown()
    {
        TensorInventorySection Section(params long[] sizes) => new("trainable parameters",
            [.. sizes.Select((b, i) => new TensorInventoryEntry($"w{i}", "Float32", [1L], b))]);

        var known = AllocationFailureReport.Render("an op", AllocationPool.Host,
            new DeviceFacts(false, null, null, null), [Section(1024, 2048)],
            new ProcessMemoryFacts(null, null, 0, null, false), "boom");
        Assert.Contains("3 KiB in total", known);

        var partly = AllocationFailureReport.Render("an op", AllocationPool.Host,
            new DeviceFacts(false, null, null, null), [Section(1024, -1)],
            new ProcessMemoryFacts(null, null, 0, null, false), "boom");
        Assert.Contains("unknown size in total", partly);
        Assert.DoesNotContain("TiB", partly);
    }

    [Fact]
    public void TestTheDeviceReadingSeparatesAFullCardFromAnArenaThatCouldNotExtend()
    {
        var roomy = new DeviceMemoryReading(
            UsedBytes: 13L << 30, FreeBytes: 11L << 30, TotalBytes: 24L << 30);
        var full = new DeviceMemoryReading(
            UsedBytes: 24L << 30, FreeBytes: 64L << 20, TotalBytes: 24L << 30);

        var cappedHost = Report(ArenaFailure, Facts(27, 28), "Shorokoo.WinGPU", roomy);
        Assert.Contains("11 GiB free", cappedHost);
        Assert.Contains("The device has room", cappedHost);
        Assert.Contains("This is the limit, not the model", cappedHost);

        var fullCard = Report(ArenaFailure, Facts(12, 128), "Shorokoo.WinGPU", full);
        Assert.Contains("64 MiB free", fullCard);
        Assert.Contains("the accelerator itself running out", fullCard);
        Assert.DoesNotContain("The device has room", fullCard);

        var arenaStuck = Report(ArenaFailure, Facts(12, 128), "Shorokoo.WinGPU", roomy);
        Assert.Contains("not the card being out of memory", arenaStuck);
        Assert.Contains("ShrinkArenaAfterRun", arenaStuck);

        Assert.NotEqual(cappedHost, fullCard);
        Assert.NotEqual(cappedHost, arenaStuck);
        Assert.NotEqual(fullCard, arenaStuck);

        Assert.DoesNotContain("Device:", Report(ArenaFailure, Facts(12, 128), "Shorokoo.LinuxCPU"));

        Assert.Contains("arena is capped at 8 GiB", AllocationFailureReport.Render(
            "the training step at step 1", AllocationPool.Device,
            new DeviceFacts(true, roomy, 8L << 30, "Shorokoo.WinGPU"), [],
            Facts(12, 128), ArenaFailure));
    }

    [Fact]
    public void TestReadProcessMemoryReportsThisProcessWithoutThrowing()
    {
        var facts = AllocationFailureReport.ReadProcessMemory();
        Assert.True(facts.ManagedHeapBytes > 0);
        Assert.True(facts.WorkingSetBytes is null or > 0);
        Assert.True(facts.CommitBytes is null or > 0);
        Assert.True(facts.LimitBytes is null or > 0);
        Assert.False(facts.LimitBytes is null && facts.LimitIsExplicit);
    }

    [Fact]
    public void TestByteSizesRenderAtTheLargestUnitThatFits()
    {
        Assert.Equal("512 B", AllocationFailureReport.Bytes(512));
        Assert.Equal("1 KiB", AllocationFailureReport.Bytes(1024));
        Assert.Equal("1.5 KiB", AllocationFailureReport.Bytes(1536));
        Assert.Equal("2 GiB", AllocationFailureReport.Bytes(2L << 30));
        Assert.Equal("1 TiB", AllocationFailureReport.Bytes(1L << 40));
        Assert.Equal("0 B", AllocationFailureReport.Bytes(0));
        Assert.Equal("unknown size", AllocationFailureReport.Bytes(-1));
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

    /// <summary>Spends time inside whatever calls it and returns how much, by its own clock: a
    /// phase of a save report that contains the call is at least this long, however busy the
    /// machine is.</summary>
    private static TimeSpan Slept()
    {
        var clock = System.Diagnostics.Stopwatch.StartNew();
        Thread.Sleep(50);
        return clock.Elapsed;
    }

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
    public void TestAtomicFileWriterReportsWhatEachWriteCost()
    {
        var dir = NewScratchDir();
        try
        {
            var payload = new byte[1 << 20];
            var target = Path.Combine(dir, "state.bin");
            AtomicFileWriter.WriteFile(target, s => s.Write(payload));

            var clock = System.Diagnostics.Stopwatch.StartNew();
            var report = AtomicFileWriter.WriteFile(target, s => s.Write(payload));
            var outer = clock.Elapsed;
            Assert.Equal(payload.Length, report.BytesWritten);
            Assert.Equal(new FileInfo(target).Length, report.BytesWritten);
            Assert.True(report.Write > TimeSpan.Zero
                && report.Flush > TimeSpan.Zero && report.Commit > TimeSpan.Zero);
            Assert.True(report.Elapsed <= outer);

            var producing = TimeSpan.Zero;
            var delayed = AtomicFileWriter.WriteFile(
                Path.Combine(dir, "slow.bin"),
                s => { producing = Slept(); s.Write(payload); });
            Assert.True(delayed.Write >= producing);

            var rotating = TimeSpan.Zero;
            AtomicFileWriter.RotationFaultInjection = _ => rotating = Slept();
            SaveReport rotated;
            clock = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                rotated = AtomicFileWriter.WriteFile(
                    Path.Combine(dir, "ckpt-7.bin"), s => s.Write(payload),
                    AtomicFileWriter.RetainPolicy.KeepLast(1, "ckpt-", ".bin"));
            }
            finally { AtomicFileWriter.RotationFaultInjection = null; }
            var rotatedOuter = clock.Elapsed;
            Assert.Equal(payload.Length, rotated.BytesWritten);
            Assert.True(rotated.Commit >= rotating);
            Assert.True(rotated.Elapsed <= rotatedOuter);

            Assert.Equal(0.0, default(SaveReport).BytesPerSecond);
            Assert.Equal(2_000_000.0, new SaveReport(
                1_000_000, TimeSpan.FromSeconds(0.2), TimeSpan.FromSeconds(0.2),
                TimeSpan.FromSeconds(0.1)).BytesPerSecond);
            Assert.Equal(
                "200,000,077 bytes in 0.252s (757 MiB/s): write 0.051s, flush 0.194s, commit 0.007s",
                new SaveReport(200_000_077, TimeSpan.FromSeconds(0.051),
                    TimeSpan.FromSeconds(0.194), TimeSpan.FromSeconds(0.007)).ToString());
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
                    model.InputNames[0]!, TensorData([4L], new float[4]))],
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

/// <summary>
/// The graphs the memory-statistics and placement checks are asked of, shared by the coverage
/// tests above and by <see cref="GpuExecutionTests"/>, which asks the same questions of a card.
/// </summary>
internal static class ArenaProbeModels
{
    /// <summary>The weight's side, so that <see cref="Weighted"/> carries four mebibytes of
    /// parameter and nothing else of any size and an arena figure either side of construction is
    /// unambiguous.</summary>
    internal const int WeightSide = 1024;

    /// <inheritdoc cref="WeightSide"/>
    internal const long WeightBytes = (long)WeightSide * WeightSide * sizeof(float);

    /// <inheritdoc cref="WeightSide"/>
    internal static CompiledGraph Weighted(ComputeContext context)
    {
        var x = InputTensor<float32>("x", rank: 2);
        var w = Tensor([(long)WeightSide, WeightSide], new float[WeightSide * WeightSide]);
        return context.Compile(
            new InternalComputationGraph([x], [(Tensor<float32>)OnnxOp.MatMul(x, w)]));
    }

    /// <inheritdoc cref="WeightSide"/>
    internal static TensorData<float32> WeightedInput() =>
        TensorData([1L, WeightSide], new float[WeightSide]);

    /// <summary>
    /// A graph a device provider has to split: <c>Det</c> has no CUDA kernel, so the square runs
    /// on the card, its result crosses to the host for the determinant, and the doubling stays
    /// where it was computed. One output on each side, and a copy node between the two providers.
    /// </summary>
    internal static CompiledGraph Partitioned(ComputeContext context)
    {
        var x = InputTensor<float32>("x", rank: 2);
        var squared = x * x;
        return context.Compile(
            new InternalComputationGraph([x], [OnnxOp.Det(squared), squared + squared]));
    }

    /// <summary>The same shape with the device half taken away: a determinant of the input and
    /// nothing else, so there is no node a CUDA provider can run at all.</summary>
    internal static CompiledGraph HostOnly(ComputeContext context)
    {
        var x = InputTensor<float32>("x", rank: 2);
        return context.Compile(new InternalComputationGraph([x], [OnnxOp.Det(x)]));
    }

    /// <inheritdoc cref="Partitioned"/>
    internal static TensorData<float32> Square() => TensorData([2L, 2L], 4f, 1f, 2f, 3f);

    /// <summary>Both operands fed rather than one held as a weight, so nothing of the arena stays
    /// in use between runs and a shrinking run has blocks to hand back.</summary>
    internal static CompiledGraph MatMul(ComputeContext context)
    {
        var x = InputTensor<float32>("x", rank: 2);
        var w = InputTensor<float32>("w", rank: 2);
        return context.Compile(new InternalComputationGraph([x, w], [OnnxOp.MatMul(x, w)]));
    }

    /// <inheritdoc cref="MatMul"/>
    internal static TensorData<float32> MatMulOperand(int side) =>
        TensorData([(long)side, side], new float[side * side]);

    /// <summary>
    /// A graph whose arena need is set by what it is fed: a shape, filled with ones by
    /// <c>Expand</c> and summed. Its one large allocation is the fill, which the session's arena
    /// makes, while what it is fed is an element per dimension — so the arena a run needs is
    /// chosen without anything of that size being fed to it.
    /// </summary>
    internal static CompiledGraph Filled(ComputeContext context)
    {
        var shape = InputVector<int64>("shape");
        return context.Compile(new InternalComputationGraph(
            [shape], [OnnxOp.ReduceSum(OnnxOp.Expand(Vector(1f), shape), keepdims: false)]));
    }

    /// <inheritdoc cref="Filled"/>
    internal static TensorData<int64> FilledShape(long elements) => TensorData([1L], elements);

    /// <summary>What a run of <see cref="Filled"/> summed.</summary>
    internal static float Sum(NamedModelParam[] outputs)
        => outputs[0].ToTensorData().As<float32>().ValueAt<float>(0);
}
