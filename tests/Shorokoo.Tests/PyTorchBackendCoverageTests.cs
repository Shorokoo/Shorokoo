using System.Runtime.InteropServices;
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

    private static readonly ShorokooTensorElementType[] EveryTorchType =
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
            PythonEnvironmentResolver.CacheRoot(new(), name => name == "XDG_CACHE_HOME" ? "/cache" : null));
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
    public void TestControlFlowRunsOnTorch()
    {
        var torch = new ComputeContext(Torch);

        Assert.True(AutoTest.AdvancedTestGraph<AutoGradIfRuntimeConditionTrueCheck>([], [QeeAudit.F32([], 2f), QeeAudit.F32([], 3f)], context: torch));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradIfRuntimeConditionFalseCheck>([], [QeeAudit.F32([], -1f), QeeAudit.F32([], 3f)], context: torch));
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
    public void TestEveryConsumedFeedIsReleasedExactlyOnceHoweverTheRunEnds()
    {
        using var session = Torch.CreateSession(Onnx("Neg", (int)ShorokooTensorElementType.Float), default, default, DeviceMemorySettings.Default);
        using var cancelled = new CancellationTokenSource();
        cancelled.Cancel();

        Assert.Equal(1, ReleasesOf(feed => session.RunConsuming(Feeds(feed), [feed], ["y"], new HashSet<string>(), RunSettings.Default)));
        Assert.Equal(1, ReleasesOf(feed => session.RunConsuming(Feeds(feed), [feed], ["nope"], new HashSet<string>(), RunSettings.Default)));
        Assert.Equal(1, ReleasesOf(feed => session.RunConsuming(new Dictionary<string, IShorokooTensorValue>(), [feed], ["y"], new HashSet<string>(), RunSettings.Default)));
        Assert.Equal(1, ReleasesOf(feed => session.RunConsuming(Feeds(feed), [feed], ["y"], new HashSet<string>(), new RunSettings { CancellationToken = cancelled.Token })));
        Assert.Equal(1, ReleasesOf(feed => session.RunConsuming(Feeds(feed), [feed], ["y"], new HashSet<string>(), RunSettings.Default, out _)));
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

    private static TorchUnsupportedModelException Refusal(GraphProto step, long version = 1)
        => Assert.Throws<TorchUnsupportedModelException>(() => Torch.CreateSession(TrainingStep(step, version), default, default, DeviceMemorySettings.Default));

    private static IDisposable OperatorTableGradient(string opType, string gradient)
        => Shorokoo.PyTorch.Translation.Operators.OperatorTable.OverrideGradient(
            opType, Enum.Parse<Shorokoo.PyTorch.Translation.Operators.TorchGradient>(gradient));

    private static NodeProto AutoGrad(string[] inputs, string[] outputs)
        => Node(TrainingFormats.AutoGradOpType, inputs, outputs, domain: TrainingFormats.AutoGradDomain);

    private static NodeProto Cast(string input, string output)
    {
        var node = Node("Cast", [input], [output]);
        node.Attributes.Add(new AttributeProto { Name = "to", Type = AttributeProto.AttributeType.Int, I = (int)ShorokooTensorElementType.Float });
        return node;
    }

    private static NodeProto Branch(string condition, GraphProto thenBranch, GraphProto elseBranch)
    {
        var node = Node("If", [condition], thenBranch.Outputs.Select(o => o.Name).ToArray());
        node.Attributes.Add(new AttributeProto { Name = "then_branch", Type = AttributeProto.AttributeType.Graph, G = thenBranch });
        node.Attributes.Add(new AttributeProto { Name = "else_branch", Type = AttributeProto.AttributeType.Graph, G = elseBranch });
        return node;
    }

    private static byte[] TrainingStep(GraphProto graph, long version = 1)
    {
        var squash = new FunctionProto { Name = "Squash", Domain = "Functions" };
        squash.Inputs.Add("a");
        squash.Outputs.Add("b");
        squash.Nodes.Add(Node("Tanh", ["a"], ["b"]));
        squash.OpsetImports.Add(new OperatorSetIdProto { Domain = "", Version = 21 });
        var model = ProtoBuf.Serializer.Deserialize<ModelProto>(new MemoryStream(Serialize(graph, squash)));
        model.OpsetImports.Add(new OperatorSetIdProto { Domain = TrainingFormats.AutoGradDomain, Version = version });
        using var stream = new MemoryStream();
        ProtoBuf.Serializer.Serialize(stream, model);
        return stream.ToArray();
    }

    private static float[][] RunFloats(IShorokooSession session, Dictionary<string, float[]> feeds, string[] outputs)
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

    private static void AssertNear(float[] expected, float[] actual)
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
        Assert.Throws<ArgumentOutOfRangeException>(() => new TorchCudaBackend(-1));
    }

    private static PythonEnvironment Resolve(PythonEnvironmentOptions options, string variable)
        => PythonEnvironmentResolver.Resolve(PythonEnvironmentLock.Cpu, options,
            name => name == PythonEnvironmentResolver.EnvironmentVariable ? variable : null);

    private static PythonEnvironmentFailure Failure(PythonEnvironmentOptions options)
        => Assert.Throws<PythonEnvironmentException>(
            () => PythonEnvironmentResolver.Resolve(PythonEnvironmentLock.Cpu, options, _ => null)).Failure;

    private static int ElementSize(ShorokooTensorElementType type) => type switch
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

    private static float[] RunFloats(IShorokooSession session, bool condition, float[] x)
    {
        using var c = Torch.CreateTensor([condition], []);
        using var input = Torch.CreateTensor(x, [x.Length]);
        using var y = session.Run(new Dictionary<string, IShorokooTensorValue> { ["c"] = c, ["x"] = input }, ["y"], RunSettings.Default)[0];
        return y.GetTensorDataAsSpan<float>().ToArray();
    }

    private static NodeProto Node(string opType, string[] inputs, string[] outputs, string domain = "")
    {
        var node = new NodeProto { OpType = opType, Name = opType, Domain = domain };
        node.Inputs.AddRange(inputs);
        node.Outputs.AddRange(outputs);
        return node;
    }

    private static GraphProto Graph(string[] inputs, string[] outputs, params NodeProto[] nodes)
    {
        var graph = new GraphProto { Name = "g" };
        graph.Inputs.AddRange(inputs.Select(name => new ValueInfoProto { Name = name }));
        graph.Outputs.AddRange(outputs.Select(name => new ValueInfoProto { Name = name }));
        graph.Nodes.AddRange(nodes);
        return graph;
    }

    private static byte[] Serialize(GraphProto graph, params FunctionProto[] functions)
    {
        var model = new ModelProto { IrVersion = 10, Graph = graph };
        model.OpsetImports.Add(new OperatorSetIdProto { Domain = "", Version = 21 });
        model.OpsetImports.Add(new OperatorSetIdProto { Domain = "Functions", Version = 1 });
        model.Functions.AddRange(functions);
        using var stream = new MemoryStream();
        ProtoBuf.Serializer.Serialize(stream, model);
        return stream.ToArray();
    }

    private static byte[] Onnx(string opType, int elementType, string domain = "", AttributeProto? attribute = null)
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
