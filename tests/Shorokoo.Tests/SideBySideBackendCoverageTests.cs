using System.Runtime.InteropServices;
using System.Runtime.Loader;
using Shorokoo.Core.Inference.Abstractions;
using Shorokoo.Runtime;

namespace Shorokoo.Tests;

/// <summary>
/// Two backends live in one process: a <see cref="ComputeContext"/> per backend, each compiling
/// and running on its own native ONNX Runtime, with one model and one set of inputs shared
/// between them.
///
/// <para>The pairing here is the default backend — the stock CPU build, discovered as usual —
/// against an <see cref="IsolatedBackend"/> on <c>ort/cuda/libonnxruntime.so</c>, which the test
/// project deploys. That file is the CUDA-flavoured ONNX Runtime build: it carries the CUDA
/// execution provider <i>and</i> the CPU one, so it is a genuinely different runtime from the
/// stock build (different binary, different provider set) while still running on a machine with
/// no card. On a machine with one it is the same file the CUDA backend binds — see
/// <c>SideBySideBackendHardwareTests</c>, which runs the CPU-and-CUDA pairing this stands in
/// for.</para>
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class SideBySideBackendCoverageTests
{
    private static readonly string BackendRoot = Path.Combine(AppContext.BaseDirectory, "ort");

    /// <summary>The native the isolated backend binds, deployed by the test project's
    /// ShorokooBackendNatives items.</summary>
    internal static string AltRuntimePath => Path.Combine(
        BackendRoot, "cuda",
        RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "onnxruntime.dll" : "libonnxruntime.so");

    internal static string PlatformFactoryAssembly(bool gpu)
        => RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
            ? (gpu ? "Shorokoo.WinGPU" : "Shorokoo.WinCPU")
            : (gpu ? "Shorokoo.LinuxGPU" : "Shorokoo.LinuxCPU");

    // One isolated backend for the whole class: loading pins a native library for the life of the
    // process, and Load caches on the spec, so a per-test load would return this one anyway.
    private static readonly Lazy<IShorokooInferenceSessionFactory> Alt = new(() =>
        IsolatedBackend.Load(new IsolatedBackendSpec
        {
            Name = "alt-runtime",
            FactoryAssembly = PlatformFactoryAssembly(gpu: false),
            NativeRuntimePath = AltRuntimePath,
        }));

    /// <summary>y = a * b + a, over four elements — enough that a wrong feed shows up as wrong
    /// numbers rather than a lucky coincidence.</summary>
    private static (InternalComputationGraph Graph, TensorData A, TensorData B, float[] Expected) Model()
    {
        var a = InputVector<float32>("a");
        var b = InputVector<float32>("b");
        var graph = new InternalComputationGraph([a, b], [a * b + a]);
        float[] av = [1f, 2f, 3f, 4f];
        float[] bv = [10f, 20f, 30f, 40f];
        return (graph, TensorData([4], av), TensorData([4], bv),
            [.. av.Zip(bv, (x, y) => x * y + x)]);
    }

    private static float[] Floats(NamedModelParam param)
        => [.. param.ToTensorData().As<float32>().AccessMemory<float>()];

    [Fact]
    public void TestTwoBackendsRunOneModelOnOneSetOfInputsInOneProcess()
    {
        var (graph, a, b, expected) = Model();
        var first = new ComputeContext();
        var second = new ComputeContext(Alt.Value);

        // The whole point: the same graph and the same two tensors, run on each backend in turn.
        Assert.Equal(expected, Floats(first.Execute(graph, a, b)[0]));
        Assert.Equal(expected, Floats(second.Execute(graph, a, b)[0]));
        // And back again, so neither run left the other's runtime unable to serve.
        Assert.Equal(expected, Floats(first.Execute(graph, a, b)[0]));

        // A compiled session belongs to the backend that built it, and re-runs there.
        var compiled = second.Compile(graph);
        Assert.Equal(expected, Floats(compiled.Execute(a, b)[0]));
        Assert.Equal(expected, Floats(compiled.Execute(a, b)[0]));
        Assert.Equal("alt-runtime", compiled.Backend.Name);
    }

    [Fact]
    public void TestEachContextNamesItsOwnBackendAndTheDefaultIsUndisturbed()
    {
        var defaultContext = new ComputeContext();
        var alt = new ComputeContext(Alt.Value);

        Assert.Equal(InferenceBackend.Describe(), defaultContext.Backend);
        Assert.Equal("alt-runtime", alt.Backend.Name);
        Assert.NotEqual(defaultContext.Backend, alt.Backend);
        Assert.Equal(ComputeDevice.Cpu, alt.Backend.Device);
        Assert.Null(alt.Backend.CudaDeviceId);

        // Naming a backend on one context does not move the process default, which is what
        // everything that never asked for a backend goes on using.
        Assert.Same(InferenceBackend.Factory, InferenceBackend.Current);
        Assert.Equal(InferenceBackend.Describe(), ComputeContext.Default.Backend);
    }

    [Fact]
    public void TestTheRenamingWrapperForwardsEveryMemberOfTheFactoryInterface()
    {
        var wrapper = typeof(IsolatedBackend)
            .GetNestedType("RenamedFactory", System.Reflection.BindingFlags.NonPublic);
        Assert.NotNull(wrapper);

        var map = wrapper!.GetInterfaceMap(typeof(IShorokooInferenceSessionFactory));
        for (int i = 0; i < map.InterfaceMethods.Length; i++)
            Assert.Equal(wrapper, map.TargetMethods[i].DeclaringType);
    }

    [Fact]
    public void TestTheIsolatedBackendIsASecondNativeRuntimeRatherThanTheSameOneTwice()
    {
        var alt = Alt.Value;
        var loadedType = alt.GetType();

        // The renaming wrapper lives in the core assembly; the factory it forwards to is the one
        // loaded in isolation, and that is the type whose context is the question.
        var inner = loadedType
            .GetField("_inner", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!
            .GetValue(alt)!;
        var innerContext = AssemblyLoadContext.GetLoadContext(inner.GetType().Assembly);
        var defaultContext = AssemblyLoadContext.GetLoadContext(
            InferenceBackend.Factory.GetType().Assembly);

        Assert.NotNull(innerContext);
        Assert.NotSame(AssemblyLoadContext.Default, innerContext);
        Assert.Same(AssemblyLoadContext.Default, defaultContext);

        // Same assembly by name, two instances: that is what gives the ONNX Runtime wrapper a
        // second copy of its static native binding, and so a second native runtime.
        Assert.Equal(
            InferenceBackend.Factory.GetType().Assembly.GetName().Name,
            inner.GetType().Assembly.GetName().Name);
        Assert.NotSame(InferenceBackend.Factory.GetType().Assembly, inner.GetType().Assembly);

        var stock = AvailableProviders(AssemblyLoadContext.Default);
        Assert.Contains("CPUExecutionProvider", stock);
        Assert.DoesNotContain("CUDAExecutionProvider", stock);
        Assert.Contains("CUDAExecutionProvider", AvailableProviders(innerContext));

        // Loading the same spec again is the same backend, not a third runtime.
        Assert.Same(alt, IsolatedBackend.Load(new IsolatedBackendSpec
        {
            Name = "alt-runtime",
            FactoryAssembly = PlatformFactoryAssembly(gpu: false),
            NativeRuntimePath = AltRuntimePath,
        }));
    }

    [Fact]
    public void TestLoadingAnIsolatedBackendLeavesDiscoveryWithExactlyOneCandidate()
    {
        var alt = Alt.Value;
        var isolated = alt.GetType()
            .GetField("_inner", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!
            .GetValue(alt)!.GetType().Assembly;
        var loaded = AppDomain.CurrentDomain.GetAssemblies();

        Assert.Contains(isolated, loaded);
        Assert.DoesNotContain(isolated, InferenceBackend.DiscoverableAssemblies(loaded));
        Assert.Contains(
            InferenceBackend.Factory.GetType().Assembly,
            InferenceBackend.DiscoverableAssemblies(loaded));

        // Which is what discovery needs: one candidate, so it still has an unambiguous answer.
        var candidates = InferenceBackend.LoadedCandidates(
            InferenceBackend.DiscoverableAssemblies(loaded).Select(a => a.GetName().Name ?? ""));
        Assert.Single(candidates);
        Assert.Equal(InferenceBackend.Factory.GetType().Assembly.GetName().Name, candidates[0].Assembly);
    }

    [Fact]
    public void TestAModelGraphRunsOnTwoBackendsInOneProcess()
    {
        var (model, input) = SideBySideModel.Concrete();
        var first = new ComputeContext();
        var second = new ComputeContext(Alt.Value);

        var onFirst = SideBySideModel.Floats(first.Execute(model, input)[0]);
        var onSecond = SideBySideModel.Floats(second.Execute(model, input)[0]);

        Assert.Equal(onFirst, SideBySideModel.Floats(first.Execute(model, input)[0]));
        SideBySideModel.AssertAgree(onFirst, onSecond);

        // And the model is doing something. A forward pass that came out constant would be
        // agreed on by any two backends, working or not.
        Assert.True(onFirst.Distinct().Count() > 1);

        // A compiled session belongs to the backend that built it, and re-runs there.
        var compiled = second.Compile(model);
        Assert.Equal("alt-runtime", compiled.Backend.Name);
        SideBySideModel.AssertAgree(onFirst, SideBySideModel.Floats(compiled.Execute(input)[0]));
    }

    /// <summary>
    /// The execution providers compiled into the native ONNX Runtime that <paramref name="context"/>'s
    /// copy of the wrapper bound. Reached by reflection because it is a property of the native
    /// build rather than of a backend, so nothing in Shorokoo's own surface reports it.
    /// </summary>
    private static string[] AvailableProviders(AssemblyLoadContext context)
    {
        var ort = context.Assemblies.Single(a => a.GetName().Name == "Microsoft.ML.OnnxRuntime");
        var ortEnv = ort.GetType("Microsoft.ML.OnnxRuntime.OrtEnv")!;
        const System.Reflection.BindingFlags Any = System.Reflection.BindingFlags.Public
            | System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.Instance;
        var instance = ortEnv.GetMethod("Instance", Any, [])!.Invoke(null, null);
        var providers = ortEnv.GetMethod("GetAvailableProviders", Any, [])!;
        return (string[])providers.Invoke(providers.IsStatic ? null : instance, null)!;
    }

    [Fact]
    public void TestALiteralFedToANonDefaultBackendIsBuiltByThatBackendsRuntime()
    {
        var (graph, a, b, expected) = Model();
        var recorder = new RecordingBackend(Alt.Value);
        var onAlt = new ComputeContext(recorder);

        Assert.Equal(expected, Floats(onAlt.Execute(graph, a, b)[0]));

        // Two literals went in, and each reached the session as a value of the isolated runtime --
        // the one that was about to read it.
        var isolated = AltLoadContext();
        Assert.Equal(2, recorder.Fed.Count);
        Assert.All(recorder.Fed, v => Assert.Same(isolated, LoadContextOf(v)));

        // Each literal kept what it built, so feeding the same backend again costs nothing: a
        // tensor that rebuilt itself per run would be the same waste by another route.
        var built = recorder.Fed.ToArray();
        recorder.Fed.Clear();
        Assert.Equal(expected, Floats(onAlt.Execute(graph, a, b)[0]));
        Assert.Equal(built, recorder.Fed);

        Assert.Equal(expected, Floats(new ComputeContext().Execute(graph, a, b)[0]));
        foreach (var literal in (TensorData[])[a, b])
        {
            Assert.Same(AssemblyLoadContext.Default, LoadContextOf(literal.ToTensorValue()));
            Assert.DoesNotContain(literal.ToTensorValue(), built);
        }
    }

    /// <summary>The load context the isolated backend's own assemblies live in. It is what tells a
    /// value that backend built apart from one the default backend built, the two being the same
    /// type name in two loads of one assembly.</summary>
    private static AssemblyLoadContext AltLoadContext()
    {
        var inner = Alt.Value.GetType()
            .GetField("_inner", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!
            .GetValue(Alt.Value)!;
        return AssemblyLoadContext.GetLoadContext(inner.GetType().Assembly)!;
    }

    private static AssemblyLoadContext LoadContextOf(IShorokooTensorValue value)
        => AssemblyLoadContext.GetLoadContext(value.GetType().Assembly)!;

    /// <summary>
    /// A backend that is the backend it wraps in every respect, and keeps the values each session
    /// it makes was fed. Everything forwards, including the interface's default-bodied members:
    /// what reaches the session has to be what the wrapped runtime would have produced, or the
    /// recording says nothing about which runtime that was.
    /// </summary>
    private sealed class RecordingBackend(IShorokooInferenceSessionFactory inner)
        : IShorokooInferenceSessionFactory
    {
        internal List<IShorokooTensorValue> Fed { get; } = [];

        public BackendDescription Description => inner.Description;

        public MemorySpace MemorySpace => inner.MemorySpace;

        public IShorokooInferenceSession CreateSession(
            ReadOnlyMemory<byte> modelBytes, ShorokooGraphOptimization graphOptimization,
            ShorokooLogSeverity logSeverity)
            => new RecordingSession(inner.CreateSession(modelBytes, graphOptimization, logSeverity), Fed);

        public IShorokooTensorValue CreateTensor<T>(T[] data, long[] shape) where T : unmanaged
            => inner.CreateTensor(data, shape);

        public IShorokooTensorValue CreateTensorFromRawBytes(
            ShorokooTensorElementType elementType, byte[] data, long[] shape)
            => inner.CreateTensorFromRawBytes(elementType, data, shape);

        public IShorokooTensorValue CreateStringTensor(IReadOnlyList<string> data, long[] shape)
            => inner.CreateStringTensor(data, shape);

        public IShorokooTensorValue CreateSequence(IReadOnlyList<IShorokooTensorValue> values)
            => inner.CreateSequence(values);

        public byte[] CopyTensorToHost(IShorokooTensorValue value) => inner.CopyTensorToHost(value);

        public IShorokooTensorValue CreateTensorInBackendMemory(
            ShorokooTensorElementType elementType, byte[] data, long[] shape)
            => inner.CreateTensorInBackendMemory(elementType, data, shape);
    }

    /// <summary>A session that notes what it was fed on its way to running it.</summary>
    private sealed class RecordingSession(
        IShorokooInferenceSession inner, List<IShorokooTensorValue> fed) : IShorokooInferenceSession
    {
        public IReadOnlyList<string> InputNames => inner.InputNames;
        public IReadOnlyList<string> OutputNames => inner.OutputNames;
        public bool HasDeviceMemory => inner.HasDeviceMemory;

        public IReadOnlyList<IShorokooTensorValue> Run(
            IReadOnlyDictionary<string, IShorokooTensorValue> inputs, IReadOnlyList<string> outputNames)
        {
            fed.AddRange(inputs.Values);
            return inner.Run(inputs, outputNames);
        }

        public IReadOnlyList<IShorokooTensorValue> RunRetainingOutputs(
            IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
            IReadOnlyList<string> outputNames, IReadOnlySet<string> retainedOutputNames)
        {
            fed.AddRange(inputs.Values);
            return inner.RunRetainingOutputs(inputs, outputNames, retainedOutputNames);
        }

        public void Dispose() => inner.Dispose();
    }

    [Fact]
    public void TestDataCrossesBetweenBackendsInBothDirectionsAndSurvivesARoundTrip()
    {
        var (graph, a, b, expected) = Model();
        var first = new ComputeContext();
        var second = new ComputeContext(Alt.Value);

        // An output is the producing backend's own value. Feeding it to the other backend's
        // session is the case BackendTransfer exists for.
        var fromAlt = second.Execute(graph, a, b)[0].ToTensorData();
        var throughDefault = Floats(first.Execute(graph, fromAlt, b)[0]);
        Assert.Equal([.. expected.Zip([10f, 20f, 30f, 40f], (x, y) => x * y + x)], throughDefault);

        var fromDefault = first.Execute(graph, a, b)[0].ToTensorData();
        Assert.Equal(expected, [.. fromDefault.As<float32>().AccessMemory<float>()]);
        Assert.Equal(
            [.. expected.Zip([10f, 20f, 30f, 40f], (x, y) => x * y + x)],
            Floats(second.Execute(graph, fromDefault, b)[0]));

        // Crossing copies rather than moves: the source is still readable, and still its own.
        Assert.Equal(expected, [.. fromAlt.As<float32>().AccessMemory<float>()]);
    }

    [Fact]
    public void TestBackendTransferRebuildsATensorAStringTensorAndASequenceOnTheOtherBackend()
    {
        var target = Alt.Value;
        var source = InferenceBackend.Factory;

        float[] floats = [1.5f, -2.5f, 3.5f];
        var tensor = source.CreateTensor(floats, [3L]);
        var copiedTensor = BackendTransfer.CopyTo(target, tensor);
        Assert.Equal(floats, copiedTensor.GetTensorDataAsSpan<float>().ToArray());
        Assert.Equal([3L], copiedTensor.Shape);
        Assert.NotSame(tensor, copiedTensor);

        string[] strings = ["alpha", "", "gamma"];
        var stringTensor = source.CreateStringTensor(strings, [3L]);
        Assert.Equal(strings, BackendTransfer.CopyTo(target, stringTensor).GetStringTensorData());

        var sequence = source.CreateSequence(
            [source.CreateTensor<float>([1f, 2f], [2L]), source.CreateTensor<float>([3f, 4f], [2L])]);
        var copiedSequence = BackendTransfer.CopyTo(target, sequence);
        Assert.Equal(ShorokooOnnxValueType.Sequence, copiedSequence.ValueType);
        Assert.Equal(2, copiedSequence.GetValueCount());
        Assert.Equal([1f, 2f], copiedSequence.GetValue(0).GetTensorDataAsSpan<float>().ToArray());
        Assert.Equal([3f, 4f], copiedSequence.GetValue(1).GetTensorDataAsSpan<float>().ToArray());

        Assert.Throws<ArgumentNullException>(() => BackendTransfer.CopyTo(target, null!));
        Assert.Throws<ArgumentNullException>(() => BackendTransfer.CopyTo(null!, tensor));
    }

    [Fact]
    public void TestLoadingAnIsolatedBackendRefusesEveryWayOfNamingOneThatIsNotThere()
    {
        var factoryAssembly = PlatformFactoryAssembly(gpu: false);
        IsolatedBackendSpec Spec(string name, string assembly, string native, string? probe = null)
            => new() { Name = name, FactoryAssembly = assembly, NativeRuntimePath = native, ProbeDirectory = probe };

        Assert.Throws<ArgumentNullException>(() => IsolatedBackend.Load(null!));
        Assert.Throws<ArgumentException>(() => IsolatedBackend.Load(Spec(" ", factoryAssembly, AltRuntimePath)));
        Assert.Throws<ArgumentException>(() => IsolatedBackend.Load(Spec("x", "", AltRuntimePath)));
        Assert.Throws<ArgumentException>(() => IsolatedBackend.Load(Spec("x", factoryAssembly, "")));

        var missingNative = Path.Combine(BackendRoot, "nowhere", "libonnxruntime.so");
        Assert.Contains("nowhere", Assert.Throws<FileNotFoundException>(
            () => IsolatedBackend.Load(Spec("x", factoryAssembly, missingNative))).Message);

        Assert.Contains("Shorokoo.NotABackend", Assert.Throws<FileNotFoundException>(
            () => IsolatedBackend.Load(Spec("x", "Shorokoo.NotABackend", AltRuntimePath))).Message);

        // An assembly that is there but carries no factory is a different failure from one that
        // is not there at all, and says so.
        Assert.Contains("exposes no concrete", Assert.Throws<InvalidOperationException>(
            () => IsolatedBackend.Load(Spec("x", "Shorokoo", AltRuntimePath))).Message);

        Assert.Contains("elsewhere", Assert.Throws<FileNotFoundException>(
            () => IsolatedBackend.Load(Spec("x", factoryAssembly, AltRuntimePath, "/elsewhere"))).Message);
    }
}

/// <summary>
/// The Shorokoo model the side-by-side tests run, and what it takes to hold one backend's answer
/// against another's. Shared by the coverage pairing and the hardware one so that both put the
/// same graph over the same inputs on two backends.
/// </summary>
internal static class SideBySideModel
{
    /// <summary>
    /// <see cref="SideBySideMlp"/> as a graph a context can execute: the module's
    /// <c>ComputationGraph</c>, concretized against the shape it is about to be fed, which is
    /// what resolves its parameter shapes and samples its weights. One call, one model, so the
    /// graph handed to the second backend is the graph the first one ran.
    /// </summary>
    internal static (ComputationGraph Model, TensorData Input) Concrete()
    {
        float[] xv = [0.5f, -1.5f, 2.0f, 0.25f, -0.75f, 1.25f, -0.5f, 3.0f];
        var input = TensorData([2L, 4L], xv);
        var module = SideBySideMlp.ComputationGraph;
        var model = module
            .ToConcreteArchitecture(module.FromOrderedInputs([input]))
            .ToConcreteModel();
        return (model, input);
    }

    internal static float[] Floats(NamedModelParam param)
        => [.. param.ToTensorData().As<float32>().AccessMemory<float>()];

    /// <summary>
    /// What two runtimes running on the <i>same</i> device may differ by, which is very little:
    /// they are two builds of one library doing one arithmetic.
    /// </summary>
    internal const double RuntimeTolerance = AutoTest.Tolerance;

    /// <summary>
    /// What the host and the card may differ by, which is a good deal more. An NVIDIA card from
    /// Ampere on puts an fp32 MatMul through its tensor cores in TF32 unless told otherwise --
    /// ten mantissa bits against fp32's twenty-three -- so its answer sits a long way from a CPU
    /// that really did the sum in fp32, and no amount of correctness on either side closes that
    /// gap.
    ///
    /// <para>Measured rather than guessed: this model deviates by 8.3e-4 over one pass on an
    /// RTX 4090, and by 2.5e-3 where a value the card produced is fed back through it, the input
    /// error and the arithmetic error compounding. Running the same test under
    /// <c>NVIDIA_TF32_OVERRIDE=0</c> brings every arm back inside
    /// <see cref="RuntimeTolerance"/>, which is what identifies TF32 as the cause. The bound is
    /// set clear of the worst of those with room to spare, and is still four orders of magnitude
    /// tighter than a backend that had genuinely miscomputed the model would land.</para>
    /// </summary>
    internal const double DeviceTolerance = 5e-3;

    /// <summary>
    /// Asserts two backends answered alike, to <paramref name="tolerance"/> -- absolute near
    /// zero, relative above it. Not exact equality: two runtimes need not agree bit for bit, and
    /// two <i>devices</i> need not agree to anything like fp32's precision.
    /// </summary>
    internal static void AssertAgree(float[] expected, float[] actual, double tolerance = RuntimeTolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        foreach (var (want, got) in expected.Zip(actual))
            Assert.True(Math.Abs(want - got) <= tolerance * Math.Max(1.0, Math.Abs(want)));
    }
}
