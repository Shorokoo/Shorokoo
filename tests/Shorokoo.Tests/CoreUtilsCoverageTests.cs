using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using Shorokoo.Core.Factory.OpsFactories;
using Shorokoo.Core.Inference;
using Shorokoo.Core.Inference.Abstractions;

namespace Shorokoo.Tests;

/// <summary>
/// Coverage for the small framework utilities the graph-level suites never touch directly: the
/// internal LINQ-ish helpers in <c>Shorokoo.Core.Utils.Extensions</c>, the <see cref="NodeKey"/> /
/// <see cref="TensorKey"/> identity structs, the <see cref="ShorokooException"/> hierarchy, the
/// OpsFactories <see cref="Helpers"/> dtype sets and attribute-type mapping, the
/// <see cref="InferenceBackend"/> deployment-folder discovery and selection policy, the typed
/// value-handle conversions, and the <see cref="AtomicFileWriter"/> temp-and-rename commit
/// protocol (crash-window fault injection, stale-temp sweep, retain-last-N rotation).
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
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
        // directly): nothing deployed → no choice; a single backend is taken as-is regardless of
        // CUDA; with both accessible CUDA presence decides.
        var cpu = ("Shorokoo.LinuxCPU", false);
        var gpu = ("Shorokoo.LinuxGPU", true);
        Assert.Null(InferenceBackend.SelectBackend([], cudaAvailable: true));
        Assert.Equal(cpu, InferenceBackend.SelectBackend([cpu], cudaAvailable: true)!.Value);
        Assert.Equal(gpu, InferenceBackend.SelectBackend([gpu], cudaAvailable: false)!.Value);
        Assert.Equal(gpu, InferenceBackend.SelectBackend([cpu, gpu], cudaAvailable: true)!.Value);
        Assert.Equal(cpu, InferenceBackend.SelectBackend([cpu, gpu], cudaAvailable: false)!.Value);
    }

    private static string ProductSourceRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !Directory.Exists(Path.Combine(dir.FullName, "src", "Shorokoo")))
            dir = dir.Parent;
        Assert.NotNull(dir);
        return Path.Combine(dir!.FullName, "src");
    }

    // Every way to come by one of ORT's SafeHandle types: the constructors, and the SessionOptions
    // factories (MakeSessionOptionWithCudaProvider and friends) that return one with no `new` in it.
    private static readonly Regex OrtSafeHandleSource = new(
        @"new\s+(SessionOptions|RunOptions)\s*\(|SessionOptions\s*\.\s*Make\w*\s*\(", RegexOptions.Compiled);

    // The two shapes that actually root the handle across a native call: the resource of a `using`,
    // and a field, which lives as long as its owner. The handle must be the WHOLE initializer --
    // `using var s = new InferenceSession(b, new SessionOptions())` roots the session and leaves the
    // options collectible, which is the exact bug this guard exists for.
    private static readonly Regex RootedInitializer = new(
        @"^\s*(using\s*\(?\s*(var|SessionOptions|RunOptions)\s+\w+\s*=\s*"
        + @"|(public|private|protected|internal)[\w\s]*?(SessionOptions|RunOptions)\s+\w+\s*=\s*)$",
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

    private static string[] UnrootedOrtSafeHandles(string source)
    {
        char[] statementEnds = [';', '{', '}', ')'];
        var code = StripCommentsAndStrings(source);
        return OrtSafeHandleSource.Matches(code)
            .Where(m => !RootedInitializer.IsMatch(code[(code.LastIndexOfAny(statementEnds, m.Index) + 1)..m.Index]))
            .Select(m => m.Value.Trim())
            .ToArray();
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
        ];
        string[] mustNotFlag =
        [
            "using var o = new SessionOptions();",
            "using (var o = new RunOptions()) { }",
            "using SessionOptions o = new SessionOptions();",
            "private readonly SessionOptions _o = new SessionOptions();",
            "using var o = SessionOptions.MakeSessionOptionWithCudaProvider(0);",
        ];
        Assert.All(mustFlag, s => Assert.NotEmpty(UnrootedOrtSafeHandles(s)));
        Assert.All(mustNotFlag, s => Assert.Empty(UnrootedOrtSafeHandles(s)));
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
}
