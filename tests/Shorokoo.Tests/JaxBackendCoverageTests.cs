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

    [Fact]
    public void TestDrawsInsideCompiledControlFlowDifferEveryIterationAndLeaveTheDrawsAfterThemRunnable()
    {
        var hundred = new TensorProto { Name = "hundred", data_type = (int)TensorProto.DataType.Int64, Int64Datas = [100], Dims = [] };
        var body = PyTorchBackendCoverageTests.Graph(["i", "c", "x"], ["c2", "x2", "r"], Node("Identity", ["c"], ["c2"]), Node("Identity", ["x"], ["x2"]), Node("RandomUniform", [], ["r"], attributes: Ints("shape", 1)));
        body.Outputs[2].Type = FloatTensor;
        var scanned = PyTorchBackendCoverageTests.Graph(["v"], ["y", "s"], Node("Loop", ["hundred", "", "v"], ["y", "s"], attributes: new AttributeProto { Name = "body", Type = AttributeProto.AttributeType.Graph, G = body }));
        scanned.Initializers.Add(hundred);
        var drawn = PyTorchBackendCoverageTests.Graph([], ["t"], Node("RandomUniform", [], ["t"], attributes: Ints("shape", 2)));
        var negated = PyTorchBackendCoverageTests.Graph([], ["e"], Node("Neg", ["x"], ["e"]));
        var branched = PyTorchBackendCoverageTests.Graph(["c", "x"], ["y"], Branch("c", drawn, negated), Node("RandomUniform", [], ["u"], attributes: Ints("shape", 2)), Node("Add", ["t", "u"], ["y"]));
        branched.Nodes[0].Outputs[0] = "t";
        using var loop = Jax.CreateSession(Serialize(scanned), default, default, DeviceMemorySettings.Default);
        using var branch = Jax.CreateSession(Serialize(branched), default, default, DeviceMemorySettings.Default);

        Assert.Equal(100, RunFloats(loop, new() { ["v"] = [0f] }, ["s"])[0].Distinct().Count());
        Assert.Equal(2, Branched(branch, true).Length);
    }

    [Fact]
    public void TestAGradientFlowsThroughEveryPathButThroughTheTensorsItIsTakenWithRespectTo()
    {
        var step = PyTorchBackendCoverageTests.Graph(["w"], ["gw", "gv"],
            Node("Relu", ["w"], ["a"]), Node("Neg", ["a"], ["v"]), Node("Mul", ["a", "v"], ["p"]), Node("ReduceSum", ["p"], ["loss"]),
            AutoGrad(["loss", "w", "v"], ["gw", "gv"]));
        using var onJax = Jax.CreateSession(TrainingStep(step), default, default, DeviceMemorySettings.Default);
        using var onTorch = new TorchCpuBackend().CreateSession(TrainingStep(step), default, default, DeviceMemorySettings.Default);

        Assert.Equal([[-1f, -2f], [1f, 2f]], RunFloats(onJax, new() { ["w"] = [1f, 2f] }, ["gw", "gv"]));
        Assert.Equal([[-1f, -2f], [1f, 2f]], PyTorchBackendCoverageTests.RunFloats(onTorch, new() { ["w"] = [1f, 2f] }, ["gw", "gv"]));
    }

    [Fact]
    public void TestWhatJaxCannotDifferentiateOrHoldIsRefusedAsAModelItCannotRun()
    {
        var counting = CountingLoop();
        counting.Nodes.Add(Node("ReduceSum", ["y"], ["loss"]));
        counting.Nodes.Add(AutoGrad(["loss", "v"], ["g"]));
        counting.Outputs[0].Name = "g";
        var sequence = PyTorchBackendCoverageTests.Graph(["s"], ["t"], Node("Identity", ["s"], ["t"]));
        sequence.Inputs[0].Type = new TypeProto { SequenceType = new TypeProto.Sequence { ElemType = FloatTensor } };
        using var differentiated = Jax.CreateSession(TrainingStep(counting), default, default, DeviceMemorySettings.Default);

        Assert.Equal("Loop", Assert.Throws<JaxUnsupportedModelException>(() => Counted(differentiated, "g", 3)).Operator);
        Assert.Equal(((string?)null, JaxUnsupportedReason.UnsupportedModel), Refusal(Serialize(sequence)));
        Assert.Equal(("Cast", JaxUnsupportedReason.UnsupportedUsage), Refusal(PyTorchBackendCoverageTests.Onnx("Cast", 1, attribute: Int("to", (int)ShorokooTensorElementType.String))));
        Assert.Equal(("Cast", JaxUnsupportedReason.UnsupportedUsage), Refusal(PyTorchBackendCoverageTests.Onnx("Cast", 1, attribute: Int("to", (int)ShorokooTensorElementType.Int4))));
    }

    [Fact]
    public void TestOperatorsComputeWhatTorchComputesWhetherTheirInputsAreFedOrConstant()
    {
        (string, ShorokooTensorElementType, long[], double[]) x = ("x", ShorokooTensorElementType.Float, [3], [double.NaN, -1, 1]);
        foreach (var constant in (bool[])[false, true])
        {
            Assert.True(AgreesWithTorch(constant, Node("Relu", ["x"], ["y"]), x));
            Assert.True(AgreesWithTorch(constant, Node("Gemm", ["a", "b", "c"], ["y"], attributes: [Float("alpha", 0.5f), Float("beta", 0.5f)]), ("a", ShorokooTensorElementType.Int32, [1, 1], [3]), ("b", ShorokooTensorElementType.Int32, [1, 1], [1]), ("c", ShorokooTensorElementType.Int32, [1, 1], [3])));
            Assert.True(AgreesWithTorch(constant ? ["x", "axis"] : ["axis"], Node("CumSum", ["x", "axis"], ["y"]), ("x", ShorokooTensorElementType.BFloat16, [3000], [.. Enumerable.Repeat(1.0, 3000)]), ("axis", ShorokooTensorElementType.Int64, [], [0])));
            Assert.True(AgreesWithTorch(constant, Node("HardSigmoid", ["x"], ["y"]), ("x", ShorokooTensorElementType.BFloat16, [3], [-4, 0.5, 4])));
            Assert.True(AgreesWithTorch(constant, Node("Pow", ["x", "p"], ["y"]), ("x", ShorokooTensorElementType.Int32, [3], [2, 1, 3]), ("p", ShorokooTensorElementType.Int32, [3], [-1, -2, 2])));
            Assert.True(AgreesWithTorch(constant, Node("Einsum", ["a", "b"], ["y"], attributes: Str("equation", "ij,jk->ik")), ("a", ShorokooTensorElementType.BFloat16, [2, 2], [1, 2, 3, 4]), ("b", ShorokooTensorElementType.BFloat16, [2, 2], [1, 0, 0, 1])));
            Assert.True(AgreesWithTorch(constant, Node("Trilu", ["x", "k"], ["y"]), ("x", ShorokooTensorElementType.Float, [2, 3], [1, 2, 3, 4, 5, 6]), ("k", ShorokooTensorElementType.Int64, [], [1])));
        }
        Assert.True(AgreesWithTorch(["s", "z"], Node("QuantizeLinear", ["x", "s", "z"], ["y"]), ("x", ShorokooTensorElementType.Float, [2], [-12.15, 3.05]), ("s", ShorokooTensorElementType.Float, [], [0.1]), ("z", ShorokooTensorElementType.UInt8, [], [128])));
        Assert.True(AgreesWithTorch(["r"], Node("MaxRoiPool", ["x", "r"], ["y"], attributes: [Ints("pooled_shape", 2, 3), Float("spatial_scale", 0.9f)]), ("x", ShorokooTensorElementType.Float, [2, 4, 7, 8], [.. Enumerable.Range(0, 448).Select(i => Math.Round(Math.Sin(0.37 * i), 3))]), ("r", ShorokooTensorElementType.Float, [1, 5], [1, 0, 0, 7, 6])));
    }

    [Fact]
    public void TestGradientsAtTheJoinsOfTheExponentialActivationsAndThroughALoopWhoseConditionIsAValueAreShorokoosRules()
    {
        var three = new TensorProto { Name = "three", data_type = (int)TensorProto.DataType.Int64, Int64Datas = [3], Dims = [] };
        var big = new TensorProto { Name = "big", data_type = (int)TensorProto.DataType.Float, FloatDatas = [1e30f], Dims = [] };
        var rate = new TensorProto { Name = "rate", data_type = (int)TensorProto.DataType.Float, FloatDatas = [1.01f], Dims = [] };
        var body = PyTorchBackendCoverageTests.Graph(["i", "c", "x"], ["c2", "x2"], Node("Less", ["x", "big"], ["c2"]), Node("Mul", ["x", "rate"], ["x2"]));
        body.Initializers.AddRange([big, rate]);
        var looped = PyTorchBackendCoverageTests.Graph(["v"], ["g"], Node("Loop", ["three", "", "v"], ["y"], attributes: new AttributeProto { Name = "body", Type = AttributeProto.AttributeType.Graph, G = body }),
            Node("ReduceSum", ["y"], ["loss"]), AutoGrad(["loss", "v"], ["g"]));
        looped.Initializers.Add(three);
        var torch = new TorchCpuBackend();

        foreach (var backend in (IShorokooBackend[])[Jax, torch])
        {
            Assert.Equal([1f], Gradient(backend, "Elu", 0f));
            Assert.Equal([1.7580993f], Gradient(backend, "Selu", 0f));
            Assert.Equal([1f], Gradient(backend, "Celu", 0f));
            AssertNear([1.030301f], GradientOf(backend, looped, 1f));
        }
    }

    [Fact]
    public void TestALoopTripCountOfOneElementIsReadAsItsCountAndAGradientJaxCannotTakeIsRefused()
    {
        var scatter = PyTorchBackendCoverageTests.Graph(["w"], ["g"], Node("Constant", [], ["i"], attributes: Tensor("value", 7, [2], [0, 0])),
            Node("ScatterElements", ["w", "i", "w"], ["s"], attributes: Str("reduction", "mul")), Node("ReduceSum", ["s"], ["loss"]), AutoGrad(["loss", "w"], ["g"]));
        using var counting = Jax.CreateSession(Serialize(CountingLoop()), default, default, DeviceMemorySettings.Default);
        using var m = Jax.CreateTensor([5L], [1]);
        using var v = Jax.CreateTensor([0f], []);
        using var y = counting.Run(new Dictionary<string, IShorokooTensorValue> { ["m"] = m, ["v"] = v }, ["y"], RunSettings.Default)[0];
        using var scattered = Jax.CreateSession(TrainingStep(scatter), default, default, DeviceMemorySettings.Default);

        Assert.Equal([5f], y.GetTensorDataAsSpan<float>().ToArray());
        Assert.IsType<JaxUnsupportedModelException>(Assert.ThrowsAny<NotSupportedException>(() => RunFloats(scattered, new() { ["w"] = [2f, 3f] }, ["g"])));
    }

    private static float[] Gradient(IShorokooBackend backend, string activation, float at)
        => GradientOf(backend, PyTorchBackendCoverageTests.Graph(["v"], ["g"], Node(activation, ["v"], ["a"]), Node("ReduceSum", ["a"], ["loss"]), AutoGrad(["loss", "v"], ["g"])), at);

    private static float[] GradientOf(IShorokooBackend backend, GraphProto step, float at)
    {
        using var session = backend.CreateSession(TrainingStep(step), default, default, DeviceMemorySettings.Default);
        using var v = backend.CreateTensor([at], [1]);
        using var g = session.Run(new Dictionary<string, IShorokooTensorValue> { ["v"] = v }, ["g"], RunSettings.Default)[0];
        return g.GetTensorDataAsSpan<float>().ToArray();
    }

    /// <summary>One node run on JAX and on torch, its inputs fed or held as the model's constants, with
    /// the same output bytes, element type and shape.</summary>
    private static bool AgreesWithTorch(bool constant, NodeProto node, params (string Name, ShorokooTensorElementType Type, long[] Shape, double[] Values)[] inputs)
        => AgreesWithTorch(constant ? [.. inputs.Select(i => i.Name)] : [], node, inputs);

    private static bool AgreesWithTorch(string[] constants, NodeProto node, params (string Name, ShorokooTensorElementType Type, long[] Shape, double[] Values)[] inputs)
    {
        var fed = inputs.Where(i => !constants.Contains(i.Name)).ToArray();
        var graph = PyTorchBackendCoverageTests.Graph([.. fed.Select(i => i.Name)], ["y"], node);
        graph.Initializers.AddRange(inputs.Where(i => constants.Contains(i.Name)).Select(i => new TensorProto { Name = i.Name, data_type = (int)i.Type, Dims = i.Shape, RawData = Bytes(i.Type, i.Values) }));
        var model = Serialize(graph);
        var torch = new TorchCpuBackend();
        var (onJax, onTorch) = (Run(Jax, model, fed), Run(torch, model, fed));
        return onJax.Type == onTorch.Type && onJax.Shape.SequenceEqual(onTorch.Shape) && onJax.Bytes.SequenceEqual(onTorch.Bytes);
    }

    private static (ShorokooTensorElementType Type, long[] Shape, byte[] Bytes) Run(
        IShorokooBackend backend, byte[] model, (string Name, ShorokooTensorElementType Type, long[] Shape, double[] Values)[] inputs)
    {
        using var session = backend.CreateSession(model, default, default, DeviceMemorySettings.Default);
        var feeds = inputs.ToDictionary(i => i.Name, i => backend.CreateTensorFromRawBytes(i.Type, Bytes(i.Type, i.Values), i.Shape));
        try
        {
            using var y = session.Run(feeds, ["y"], RunSettings.Default)[0];
            return (y.ElementType, y.Shape, backend.CopyTensorToHost(y));
        }
        finally
        {
            foreach (var feed in feeds.Values) feed.Dispose();
        }
    }

    private static byte[] Bytes(ShorokooTensorElementType type, double[] values) => type switch
    {
        ShorokooTensorElementType.Float => [.. values.SelectMany(v => BitConverter.GetBytes((float)v))],
        ShorokooTensorElementType.BFloat16 => [.. values.SelectMany(v => BitConverter.GetBytes((ushort)(BitConverter.SingleToUInt32Bits((float)v) >> 16)))],
        ShorokooTensorElementType.Int32 => [.. values.SelectMany(v => BitConverter.GetBytes((int)v))],
        ShorokooTensorElementType.Int64 => [.. values.SelectMany(v => BitConverter.GetBytes((long)v))],
        ShorokooTensorElementType.UInt8 => [.. values.Select(v => (byte)v)],
        _ => throw new NotSupportedException(type.ToString()),
    };

    private static AttributeProto Float(string name, float value) => new() { Name = name, Type = AttributeProto.AttributeType.Float, F = value };

    private static AttributeProto Ints(string name, params long[] values) => new() { Name = name, Type = AttributeProto.AttributeType.Ints, Ints = values };

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
