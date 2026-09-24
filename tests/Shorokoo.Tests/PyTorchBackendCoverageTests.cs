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
        Assert.Equal(running.Directory, new TorchCpuBackend(new() { EnvironmentPath = running.Directory + Path.DirectorySeparatorChar }).Start().Directory);
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

    private static byte[] TrainingStep(GraphProto graph, long version = 1, params FunctionProto[] functions)
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
        Assert.Equal(Path.Combine("/opt", "py"), PythonEnvironment.PythonHomeOf(Path.Combine("/opt", "py", "bin"), windows: false));
        Assert.Contains(Path.Combine("/opt", "py", "lib", "libpython3.12.so"), PythonEnvironment.LibPythonCandidates(Path.Combine("/opt", "py"), version, windows: false));
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
    public void TestACpuSessionHasNoArenaFiguresRetainsNothingAndRunsEveryNodeOnTheHost()
    {
        var graph = ComputeContextLifetimeCoverageTests.GraphOf("a:float[2] b:float[2]", "O:float[2]",
            ComputeContextLifetimeCoverageTests.Op("Sub", "a b", "t"), ComputeContextLifetimeCoverageTests.Op("Neg", "t", "O"));
        using var traced = Torch.CreateSession(Serialize(graph), default, default, DeviceMemorySettings.Default, new DiagnosticSettings { TraceNodePlacement = true });
        using var plain = Torch.CreateSession(Serialize(graph), default, default, DeviceMemorySettings.Default);
        using var a = Torch.CreateTensor([5f, 7f], [2]);
        using var b = Torch.CreateTensor([1f, 2f], [2]);
        var feeds = new Dictionary<string, IShorokooTensorValue> { ["a"] = a, ["b"] = b };
        using var kept = traced.RunRetainingOutputs(feeds, ["O"], new HashSet<string> { "O" }, RunSettings.Default)[0];

        Assert.False(traced.HasDeviceMemory);
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
        Assert.Equal((null, "-9 -18 -27 -36"), Aliased("a:float[4] b:float[4]", "O:float[4]", [Op("Sub", "a b", "t"), Op("Neg", "t", "O")]));
        Assert.Equal((null, "9 18 27 36"), Aliased("a:int64[4] b:int64[4]", "O:int64[4]", [Op("Sub", "a b", "O")], integers: true));
        Assert.Equal((null, "0 -20 -60 -120 -10 -20 -30 -40"), Aliased("a:float[4] b:float[4]", "O:float[4] T:float[4]", [Op("Transpose", "a", "t"), Op("Mul", "t b", "m"), Op("Sub", "a m", "O"), Op("Neg", "t", "T")]));
        Assert.Equal((null, "0 -20 -60 -120 10 20 30 40"), Aliased("a:float[4] b:float[4]", "O:float[4] T:float[4]", [Op("Transpose", "a", "T"), Op("Mul", "T b", "m"), Op("Sub", "a m", "O")]));
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
        using var o = session.RunConsuming(feeds, [a], ["O"], ComputeContext.NoOutputsRetained, RunSettings.Default)[0];

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
    public void TestAnInitializerWhoseRawDataIsNotItsShapesSizeIsRefusedAtSessionCreation()
    {
        Assert.Throws<TorchUnsupportedModelException>(() => Torch.CreateSession(RawInitialized(4), default, default, DeviceMemorySettings.Default));
        Assert.Throws<TorchUnsupportedModelException>(() => Torch.CreateSession(RawInitialized(16), default, default, DeviceMemorySettings.Default));
        Torch.CreateSession(RawInitialized(12), default, default, DeviceMemorySettings.Default).Dispose();
    }

    [Fact]
    public void TestProvisioningInstallsTheLockByHashIntoItsOwnEnvironmentAndGivesUpOnAUvThatHangs()
    {
        var installed = FakeUv("""
            echo "$@" >> LOG
            if [ "$1" = venv ]; then for last; do :; done; mkdir -p "$last"; fi
            """);
        var hung = FakeUv("sleep 20; exit 1", TimeSpan.FromSeconds(2));
        var install = installed.Log.Single(line => line.StartsWith("pip install", StringComparison.Ordinal));

        Assert.Equal(PythonEnvironmentFailure.NotAVirtualEnvironment, installed.Failure);
        Assert.Contains($"--python {installed.Directory}", install);
        Assert.Contains("--require-hashes", install);
        Assert.Equal(PythonEnvironmentFailure.ProvisioningTimedOut, hung.Failure);
        Assert.True(hung.Took < TimeSpan.FromSeconds(15));
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
        Assert.Throws<PythonEnvironmentException>(() => new OffPlatformBackend().Start());
    }

    private sealed class OffPlatformBackend() : TorchBackend(PythonEnvironmentLock.ForPlatform("cpu", "osx-arm64"), null, null);

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
        var results = session.RunConsuming(feeds, consume ? [a] : [], [.. graph.Outputs.Select(o => o.Name)], ComputeContext.NoOutputsRetained, RunSettings.Default, out var aliased);
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
            [.. graph.Outputs.Select(o => o.Name)], ComputeContext.NoOutputsRetained, RunSettings.Default, out var aliased);
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

    private static (PythonEnvironmentFailure Failure, string[] Log, string Directory, TimeSpan Took) FakeUv(string script, TimeSpan? timeout = null)
    {
        var root = Directory.CreateDirectory(Path.Combine(Path.GetTempPath(), "shorokoo-uv-" + Guid.NewGuid().ToString("N"))).FullName;
        var (uv, log) = (Path.Combine(root, "uv"), Path.Combine(root, "log"));
        File.WriteAllText(uv, "#!/bin/sh\n" + script.Replace("LOG", log) + "\n");
        File.SetUnixFileMode(uv, UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute);
        var clock = System.Diagnostics.Stopwatch.StartNew();
        try
        {
            var failure = Assert.Throws<PythonEnvironmentException>(() => PythonEnvironmentResolver.Resolve(PythonEnvironmentLock.Cpu,
                new() { CacheDirectory = root, UvPath = uv, ProvisioningTimeout = timeout ?? TimeSpan.FromMinutes(1) }, _ => null)).Failure;
            return (failure, File.Exists(log) ? File.ReadAllLines(log) : [], Path.Combine(root, PythonEnvironmentLock.Cpu.CacheKey), clock.Elapsed);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
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

    private static GraphProto ScanLoop(bool typed)
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

    private static float[] RunFloats(IShorokooSession session, bool condition, float[] x)
    {
        using var c = Torch.CreateTensor([condition], []);
        using var input = Torch.CreateTensor(x, [x.Length]);
        using var y = session.Run(new Dictionary<string, IShorokooTensorValue> { ["c"] = c, ["x"] = input }, ["y"], RunSettings.Default)[0];
        return y.GetTensorDataAsSpan<float>().ToArray();
    }

    private static NodeProto Node(string opType, string[] inputs, string[] outputs, string domain = "", params AttributeProto[] attributes)
    {
        var node = new NodeProto { OpType = opType, Name = $"{opType}:{string.Join(",", outputs)}", Domain = domain };
        node.Inputs.AddRange(inputs);
        node.Outputs.AddRange(outputs);
        node.Attributes.AddRange(attributes);
        return node;
    }

    private static readonly (string, float[], long[]) X23 = ("x", [3f, 1f, 3f, 2f, 5f, 1f], [2, 3]);

    private static TypeProto FloatTensor => new() { TensorType = new TypeProto.Tensor { ElemType = 1 } };

    private static AttributeProto Int(string name, long value) => new() { Name = name, Type = AttributeProto.AttributeType.Int, I = value };

    private static AttributeProto Str(string name, string value)
        => new() { Name = name, Type = AttributeProto.AttributeType.String, S = System.Text.Encoding.UTF8.GetBytes(value) };

    private static AttributeProto Tensor(string name, int elementType, long[] dims, long[] values)
        => new() { Name = name, Type = AttributeProto.AttributeType.Tensor, T = new TensorProto { data_type = elementType, Dims = dims, Int64Datas = values } };

    private static byte[] Typed(int opset, string[] inputs, string[] outputs, NodeProto[] nodes, params FunctionProto[] functions)
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

    private static GraphProto Graph(string[] inputs, string[] outputs, params NodeProto[] nodes)
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
