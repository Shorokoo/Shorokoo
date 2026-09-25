using Shorokoo.Core.Backends;
using Shorokoo.Core.Factory.IR;
using Shorokoo.Jax;
using Shorokoo.Jax.Cpu;
using Shorokoo.Jax.Cuda;
using Shorokoo.PythonHost;
using Shorokoo.PyTorch.Cpu;
using Shorokoo.Runtime;
using static Shorokoo.Tests.PyTorchBackendCoverageTests;

namespace Shorokoo.Tests;

[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class JaxBackendCoverageTests
{
    private static readonly JaxCpuBackend Jax = new();

    [Fact]
    public void TestEveryJaxElementTypeRoundTripsThroughATensorAndASession()
    {
        foreach (var type in EveryTorchType)
        {
            var bytes = Enumerable.Range(0, 3 * ElementSize(type)).Select(i => (byte)(type == ShorokooTensorElementType.Bool ? i % 2 : i * 7 % 64)).ToArray();
            using var session = Jax.CreateSession(PyTorchBackendCoverageTests.Onnx("Identity", (int)type), default, default, DeviceMemorySettings.Default);
            using var host = Jax.CreateTensorFromRawBytes(type, bytes, [3]);
            using var device = Jax.CreateTensorInBackendMemory(type, bytes, [3]);
            var outputs = session.Run(new Dictionary<string, IShorokooTensorValue> { ["x0"] = host }, ["y"], RunSettings.Default);

            Assert.Equal(type, host.ElementType);
            Assert.Equal([3L], host.Shape);
            Assert.Equal(bytes, Jax.CopyTensorToHost(host));
            Assert.Equal(bytes, Jax.CopyTensorToHost(device));
            Assert.Equal(bytes, Jax.CopyTensorToHost(outputs[0]));
            Assert.Equal(type, outputs[0].ElementType);
            outputs[0].Dispose();
        }
    }

    [Fact]
    public void TestTypedTensorsSpansAndUninitializedTensorsAreTheTensorsTheySayTheyAre()
    {
        using var floats = Jax.CreateTensor([1f, 2f, 3f, 4f], [2, 2]);
        using var longs = Jax.CreateTensor([long.MaxValue, -1L], [2]);
        using var blank = Jax.CreateUninitializedTensorInBackendMemory(ShorokooTensorElementType.Int32, [2, 3]);
        using var empty = Jax.CreateTensor(Array.Empty<float>(), [0, 4]);
        floats.GetTensorMutableDataAsSpan<float>()[3] = 9f;

        Assert.Equal([1f, 2f, 3f, 9f], floats.GetTensorDataAsSpan<float>().ToArray());
        Assert.Equal([long.MaxValue, -1L], longs.GetTensorDataAsSpan<long>().ToArray());
        Assert.Equal(6, blank.GetTensorDataAsSpan<int>().Length);
        Assert.Equal([0L, 4L], empty.Shape);
        Assert.True(floats.IsHostAccessible);
        Assert.Throws<NotSupportedException>(() => Jax.CreateTensorFromRawBytes(ShorokooTensorElementType.Int4, [0], [2]));
        Assert.Throws<NotSupportedException>(() => Jax.CreateStringTensor(["a"], [1]));
        Assert.Throws<NotSupportedException>(() => Jax.CreateSequence([]));
        Assert.Throws<ArgumentException>(() => Jax.CreateTensorFromRawBytes(ShorokooTensorElementType.Float, [0, 0], [1]));
    }

    [Fact]
    public void TestAModelGraphRunsOnJaxAndAgreesWithOnnxRuntime()
    {
        var (model, input) = SideBySideModel.Concrete();
        var onOrt = SideBySideModel.Floats(new ComputeContext().Execute(model, input.Shared())[0]);
        using var context = new ComputeContext(Jax);
        var compiled = context.Compile(model);

        SideBySideModel.AssertAgree(onOrt, SideBySideModel.Floats(context.Execute(model, input.Shared())[0]));
        SideBySideModel.AssertAgree(onOrt, SideBySideModel.Floats(compiled.Execute(input.Shared())[0]));
        SideBySideModel.AssertAgree(onOrt, SideBySideModel.Floats(compiled.Execute(input)[0]));
        Assert.True(input.IsDisposed);
        Assert.Equal("Shorokoo.Jax.Cpu", compiled.Backend.Name);
    }

    [Fact]
    public void TestAnAutoGradNodeIsJaxsGradientOfItsLoss()
    {
        float[] w = [0.5f, -1f, 2f], b = [0.1f, 0.2f, -0.3f], x = [1f, 2f, -0.5f], c = [0.3f, -0.2f, 0.1f];
        var step = PyTorchBackendCoverageTests.Graph(["w", "b", "x", "c", "u"], ["w2", "gb", "gp", "gu", "loss"],
            Node("Exp", ["c"], ["p"]),
            Node("Mul", ["w", "x"], ["m"]),
            Node("Add", ["m", "b"], ["s"]),
            Node("Tanh", ["s"], ["t"]),
            Node("Mul", ["t", "p"], ["tp"]),
            Node("ReduceSum", ["tp"], ["loss"]),
            AutoGrad(["loss", "w", "b", "p", "u"], ["gw", "gb", "gp", "gu"]),
            Node("Sub", ["w", "gw"], ["w2"]));
        using var session = Jax.CreateSession(TrainingStep(step), ShorokooGraphOptimization.TrainingStep, default, DeviceMemorySettings.Default);
        var outputs = RunFloats(session, new() { ["w"] = w, ["b"] = b, ["x"] = x, ["c"] = c, ["u"] = [4f] }, ["w2", "gb", "gp", "gu", "loss"]);

        var t = w.Select((wi, i) => MathF.Tanh(wi * x[i] + b[i])).ToArray();
        var gb = t.Select((ti, i) => MathF.Exp(c[i]) * (1 - ti * ti)).ToArray();
        AssertNear(w.Select((wi, i) => wi - gb[i] * x[i]).ToArray(), outputs[0]);
        AssertNear(gb, outputs[1]);
        AssertNear(t, outputs[2]);
        Assert.Equal([0f], outputs[3]);
        AssertNear([t.Select((ti, i) => ti * MathF.Exp(c[i])).Sum()], outputs[4]);
        Assert.True(((IShorokooBackend)Jax).AcceptsTrainingFormat(TrainingFormats.OnnxAutoGrad));
        Assert.True(((IShorokooBackend)Jax).AcceptsTrainingFormat(TrainingFormats.Onnx));
        Assert.False(((IShorokooBackend)Jax).AcceptsTrainingFormat("onnx-autograd/2"));
    }

    [Fact]
    public void TestWhatJaxCannotHoldIsRefusedAtSessionCreationNamingIt()
    {
        var reshapedByValue = Shaped(PyTorchBackendCoverageTests.Graph(["x", "s"], ["y"], Node("Reshape", ["x", "s"], ["y"])), ("x", 1, [4]), ("s", 7, [2]));
        var reshapedByShape = Shaped(PyTorchBackendCoverageTests.Graph(["x"], ["y"], Node("Shape", ["x"], ["s"]), Node("Reshape", ["x", "s"], ["y"])), ("x", 1, [4]));

        Assert.Equal(("NonZero", JaxUnsupportedReason.UnknownOperator), Refusal(PyTorchBackendCoverageTests.Onnx("NonZero", 1)));
        Assert.Equal(("SequenceLength", JaxUnsupportedReason.UnknownOperator), Refusal(PyTorchBackendCoverageTests.Onnx("SequenceLength", 1)));
        Assert.Equal(((string?)null, JaxUnsupportedReason.UnsupportedModel), Refusal(PyTorchBackendCoverageTests.Onnx("Identity", (int)ShorokooTensorElementType.String)));
        Assert.Equal(("Reshape", JaxUnsupportedReason.UnsupportedUsage), Refusal(Serialize(reshapedByValue)));
        Jax.CreateSession(Serialize(reshapedByShape), default, default, DeviceMemorySettings.Default).Dispose();
    }

    [Fact]
    public void TestAModelOfUnfixedShapeIsCompiledForEachShapeItIsFed()
    {
        using var session = Jax.CreateSession(Serialize(PyTorchBackendCoverageTests.Graph(["x"], ["y"], Node("Shape", ["x"], ["s"]), Node("Cast", ["s"], ["f"], attributes: Int("to", 1)), Node("Mul", ["x", "f"], ["y"]))), default, default, DeviceMemorySettings.Default);

        Assert.Equal([2f, 4f], RunFloats(session, new() { ["x"] = [1f, 2f] }, ["y"])[0]);
        Assert.Equal([3f, 6f, 9f], RunFloats(session, new() { ["x"] = [1f, 2f, 3f] }, ["y"])[0]);
    }

    [Fact]
    public void TestControlFlowOnAnInputsValueCompilesAsXlaControlFlow()
    {
        var then = PyTorchBackendCoverageTests.Graph([], ["y"], Node("Add", ["x", "x"], ["y"]));
        var otherwise = PyTorchBackendCoverageTests.Graph([], ["e"], Node("Neg", ["x"], ["e"]));
        var longer = PyTorchBackendCoverageTests.Graph([], ["e"], Node("Concat", ["x", "x"], ["e"], attributes: Int("axis", 0)));
        using var branch = Jax.CreateSession(Serialize(PyTorchBackendCoverageTests.Graph(["c", "x"], ["y"], Branch("c", then, otherwise))), default, default, DeviceMemorySettings.Default);
        using var counting = Jax.CreateSession(Serialize(CountingLoop()), default, default, DeviceMemorySettings.Default);
        using var scanning = Jax.CreateSession(Serialize(ScanLoop(typed: true)), default, default, DeviceMemorySettings.Default);
        using var uneven = Jax.CreateSession(Serialize(PyTorchBackendCoverageTests.Graph(["c", "x"], ["y"], Branch("c", then, longer))), default, default, DeviceMemorySettings.Default);

        Assert.Equal([2f, 4f], Branched(branch, true));
        Assert.Equal([-1f, -2f], Branched(branch, false));
        Assert.Equal([5f], Counted(counting, "y", 5));
        Assert.Equal("If", Assert.Throws<JaxUnsupportedModelException>(() => Branched(uneven, true)).Operator);
        Assert.Equal("Loop", Assert.Throws<JaxUnsupportedModelException>(() => Counted(scanning, "s", 2)).Operator);
    }

    [Fact]
    public void TestARunCancelledBeforeItStartsIsRefusedAndOneWithALiveTokenRuns()
    {
        using var session = Jax.CreateSession(PyTorchBackendCoverageTests.Onnx("Neg", 1), default, default, DeviceMemorySettings.Default);
        using var x = Jax.CreateTensor([1f], [1]);
        using var cancelled = new CancellationTokenSource();
        cancelled.Cancel();

        Assert.Throws<OperationCanceledException>(() => session.Run(new Dictionary<string, IShorokooTensorValue> { ["x0"] = x }, ["y"], new RunSettings { CancellationToken = cancelled.Token }));
        using var y = session.Run(new Dictionary<string, IShorokooTensorValue> { ["x0"] = x }, ["y"], new RunSettings { CancellationToken = new CancellationTokenSource().Token })[0];
        Assert.Equal([-1f], y.GetTensorDataAsSpan<float>().ToArray());
    }

    [Fact]
    public void TestACpuSessionHasNoArenaFiguresBindsNoAliasRetainsNothingAndRunsEveryNodeOnTheHost()
    {
        var graph = ComputeContextLifetimeCoverageTests.GraphOf("a:float[2] b:float[2]", "O:float[2]",
            ComputeContextLifetimeCoverageTests.Op("Sub", "a b", "t"), ComputeContextLifetimeCoverageTests.Op("Neg", "t", "O"));
        using var traced = Jax.CreateSession(Serialize(graph), default, default, DeviceMemorySettings.Default, new DiagnosticSettings { TraceNodePlacement = true }, [new OutputAlias("O", "a")]);
        using var a = Jax.CreateTensor([5f, 7f], [2]);
        using var b = Jax.CreateTensor([1f, 2f], [2]);
        var consumed = Jax.CreateTensor([5f, 7f], [2]);
        var feeds = new Dictionary<string, IShorokooTensorValue> { ["a"] = consumed, ["b"] = b };
        using var kept = traced.RunConsuming(feeds, [consumed], ["O"], new HashSet<string> { "O" }, RunSettings.Default)[0];

        Assert.False(traced.HasDeviceMemory);
        Assert.True(kept.IsHostAccessible);
        Assert.Equal([-4f, -5f], kept.GetTensorDataAsSpan<float>().ToArray());
        Assert.Throws<ObjectDisposedException>(() => consumed.IsHostAccessible);
        Assert.Empty(traced.BindableAliases);
        Assert.Null(traced.ReadArenaStatistics());
        Assert.Equal(SessionOutputPlacement.Host, traced.OutputPlacement);
        Assert.Equal([("Sub", "cpu"), ("Neg", "cpu")], traced.ReadNodePlacement()!.Nodes.Select(n => (n.OpType, n.Provider)));
    }

    [Fact]
    public void TestEveryJaxBackendSharesOneRuntimeApartFromTorchsAndNamesItsOwnMemory()
    {
        var cuda = new JaxCudaBackend(1);

        Assert.Same(Jax.RuntimeIdentity, cuda.RuntimeIdentity);
        Assert.NotSame(Jax.RuntimeIdentity, new TorchCpuBackend().RuntimeIdentity);
        Assert.Equal(MemorySpace.Host, ((IShorokooBackend)Jax).MemorySpace);
        Assert.Equal(MemorySpace.Cuda(1), ((IShorokooBackend)cuda).MemorySpace);
        Assert.Equal("cuda:1", cuda.DeviceName);
        Assert.Equal(Jax.Start().Directory, new TorchCpuBackend().Start().Directory);
        Assert.Contains("jax==0.11.2", PythonEnvironmentLock.Cpu.Requirements);
        Assert.Contains("jax-cuda13-plugin==0.11.2", PythonEnvironmentLock.ForPlatform("cu13", "linux-x64").Requirements);
        Assert.DoesNotContain("jax-cuda13-plugin", PythonEnvironmentLock.ForPlatform("cu13", "win-x64").Requirements);
    }

    [NoCudaDriverFact]
    public void TestTheCudaBackendOnAMachineWithoutADriverRefusesToStartBeforeProvisioningAnything()
    {
        var cache = Path.Combine(Path.GetTempPath(), "shorokoo-nodriver-" + Guid.NewGuid().ToString("N"));
        var refusal = Assert.Throws<PythonEnvironmentException>(() => new JaxCudaBackend(0, new() { CacheDirectory = cache }).Start());

        Assert.Equal(PythonEnvironmentFailure.DeviceUnavailable, refusal.Failure);
        Assert.False(Directory.Exists(cache));
    }

    private static (string? Operator, JaxUnsupportedReason Reason) Refusal(byte[] model)
    {
        var refused = Assert.Throws<JaxUnsupportedModelException>(() => Jax.CreateSession(model, default, default, DeviceMemorySettings.Default));
        return (refused.Operator, refused.Reason);
    }

    private static GraphProto Shaped(GraphProto graph, params (string Name, int Type, long[] Dims)[] inputs)
    {
        foreach (var (name, type, dims) in inputs)
        {
            var shape = new TensorShapeProto();
            shape.Dims.AddRange(dims.Select(d => new TensorShapeProto.Dimension { DimValue = d }));
            graph.Inputs.Single(i => i.Name == name).Type = new TypeProto { TensorType = new TypeProto.Tensor { ElemType = type, Shape = shape } };
        }
        return graph;
    }

    private static float[] Branched(IShorokooSession session, bool condition)
    {
        using var c = Jax.CreateTensor([condition], []);
        using var x = Jax.CreateTensor([1f, 2f], [2]);
        using var y = session.Run(new Dictionary<string, IShorokooTensorValue> { ["c"] = c, ["x"] = x }, ["y"], RunSettings.Default)[0];
        return y.GetTensorDataAsSpan<float>().ToArray();
    }

    private static float[] Counted(IShorokooSession session, string output, long iterations)
    {
        using var m = Jax.CreateTensor([iterations], []);
        using var v = Jax.CreateTensor([0f], []);
        using var y = session.Run(new Dictionary<string, IShorokooTensorValue> { ["m"] = m, ["v"] = v }, [output], RunSettings.Default)[0];
        return y.GetTensorDataAsSpan<float>().ToArray();
    }

    private static float[][] RunFloats(IShorokooSession session, Dictionary<string, float[]> feeds, string[] outputs)
    {
        var inputs = feeds.ToDictionary(f => f.Key, f => Jax.CreateTensor(f.Value, [f.Value.Length]));
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
}
