using System.Runtime.InteropServices;
using System.Runtime.Loader;
using Shorokoo.Core.Backends;
using Shorokoo.Modules.Losses;
using Shorokoo.Modules.Optimizers;
using Shorokoo.Runtime;
using Shorokoo.Tests.Modules;

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

    internal static string PlatformBackendAssembly(bool gpu)
        => RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
            ? (gpu ? "Shorokoo.WinGPU" : "Shorokoo.WinCPU")
            : (gpu ? "Shorokoo.LinuxGPU" : "Shorokoo.LinuxCPU");

    // One isolated backend for the whole class: loading pins a native library for the life of the
    // process, and Load caches on the spec, so a per-test load would return this one anyway.
    private static readonly Lazy<IShorokooBackend> Alt = new(() =>
        IsolatedBackend.Load(new IsolatedBackendSpec
        {
            Name = "alt-runtime",
            BackendAssembly = PlatformBackendAssembly(gpu: false),
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

    [Fact]
    public void TestAnIsolatedBackendsSessionWritesAnOutputIntoTheInputItConsumed()
    {
        using var context = new ComputeContext(Alt.Value);
        var a = InputVector<float32>("a");
        var b = InputVector<float32>("b");
        var compiled = context.Compile(new InternalComputationGraph([a, b], [a - b]), [[4L], [4L]],
            trainingStep: false, aliasCandidates: [(0, 0)]);

        var result = compiled.Execute(TensorData([4L], 10f, 20f, 30f, 40f), TensorData([4L], 1f, 2f, 3f, 4f));

        Assert.Equal([9f, 18f, 27f, 36f], result[0].ToTensorData().As<float32>().CopyMemory<float>());
        Assert.Equal(1L, context.AliasedOutputs);
    }

    [Fact]
    public void TestTwoBackendsRunOneModelOnOneSetOfInputsInOneProcess()
    {
        var (graph, a, b, expected) = Model();
        var first = new ComputeContext();
        var second = new ComputeContext(Alt.Value);

        Assert.Equal(expected, SideBySideModel.Floats(first.Execute(graph, a.Shared(), b.Shared())[0]));
        Assert.Equal(expected, SideBySideModel.Floats(second.Execute(graph, a.Shared(), b.Shared())[0]));
        Assert.Equal(expected, SideBySideModel.Floats(first.Execute(graph, a.Shared(), b.Shared())[0]));

        var compiled = second.Compile(graph);
        Assert.Equal(expected, SideBySideModel.Floats(compiled.Execute(a.Shared(), b.Shared())[0]));
        Assert.Equal(expected, SideBySideModel.Floats(compiled.Execute(a, b)[0]));
        Assert.True(a.IsDisposed);
        Assert.Equal("alt-runtime", compiled.Backend.Name);
    }

    [Fact]
    public void TestEachContextNamesItsOwnBackendAndTheDefaultIsUndisturbed()
    {
        var defaultContext = new ComputeContext();
        var alt = new ComputeContext(Alt.Value);

        Assert.Equal(DefaultBackend.Describe(), defaultContext.Backend);
        Assert.Equal("alt-runtime", alt.Backend.Name);
        Assert.NotEqual(defaultContext.Backend, alt.Backend);
        Assert.Equal(ComputeDevice.Cpu, alt.Backend.Device);
        Assert.Null(alt.Backend.CudaDeviceId);

        Assert.Same(DefaultBackend.Instance, DefaultBackend.Current);
        Assert.Equal(DefaultBackend.Describe(), ComputeContext.Default.Backend);
    }

    [Fact]
    public void TestTheRenamingWrapperForwardsEveryMemberOfTheBackendInterface()
    {
        var wrapper = typeof(IsolatedBackend)
            .GetNestedType("RenamedBackend", System.Reflection.BindingFlags.NonPublic);
        Assert.NotNull(wrapper);

        var map = wrapper!.GetInterfaceMap(typeof(IShorokooBackend));
        for (int i = 0; i < map.InterfaceMethods.Length; i++)
            Assert.Equal(wrapper, map.TargetMethods[i].DeclaringType);
    }

    [Fact]
    public void TestTheIsolatedBackendIsASecondNativeRuntimeRatherThanTheSameOneTwice()
    {
        var alt = Alt.Value;
        var loadedType = alt.GetType();

        var inner = loadedType
            .GetField("_inner", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!
            .GetValue(alt)!;
        var innerContext = AssemblyLoadContext.GetLoadContext(inner.GetType().Assembly);
        var defaultContext = AssemblyLoadContext.GetLoadContext(
            DefaultBackend.Instance.GetType().Assembly);

        Assert.NotNull(innerContext);
        Assert.NotSame(AssemblyLoadContext.Default, innerContext);
        Assert.Same(AssemblyLoadContext.Default, defaultContext);

        Assert.Equal(
            DefaultBackend.Instance.GetType().Assembly.GetName().Name,
            inner.GetType().Assembly.GetName().Name);
        Assert.NotSame(DefaultBackend.Instance.GetType().Assembly, inner.GetType().Assembly);

        var stock = AvailableProviders(AssemblyLoadContext.Default);
        Assert.Contains("CPUExecutionProvider", stock);
        Assert.DoesNotContain("CUDAExecutionProvider", stock);
        Assert.Contains("CUDAExecutionProvider", AvailableProviders(innerContext));

        Assert.Same(alt, IsolatedBackend.Load(new IsolatedBackendSpec
        {
            Name = "alt-runtime",
            BackendAssembly = PlatformBackendAssembly(gpu: false),
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
        Assert.DoesNotContain(isolated, DefaultBackend.DiscoverableAssemblies(loaded));
        Assert.Contains(
            DefaultBackend.Instance.GetType().Assembly,
            DefaultBackend.DiscoverableAssemblies(loaded));

        var candidates = DefaultBackend.LoadedCandidates(
            DefaultBackend.DiscoverableAssemblies(loaded).Select(a => a.GetName().Name ?? ""));
        Assert.Single(candidates);
        Assert.Equal(DefaultBackend.Instance.GetType().Assembly.GetName().Name, candidates[0].Assembly);
    }

    [Fact]
    public void TestAModelGraphRunsOnTwoBackendsInOneProcess()
    {
        var (model, input) = SideBySideModel.Concrete();
        var first = new ComputeContext();
        var second = new ComputeContext(Alt.Value);

        var onFirst = SideBySideModel.Floats(first.Execute(model, input.Shared())[0]);
        var onSecond = SideBySideModel.Floats(second.Execute(model, input.Shared())[0]);

        Assert.Equal(onFirst, SideBySideModel.Floats(first.Execute(model, input.Shared())[0]));
        SideBySideModel.AssertAgree(onFirst, onSecond);

        Assert.True(onFirst.Distinct().Count() > 1);

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

        Assert.Equal(expected, SideBySideModel.Floats(onAlt.Execute(graph, a.Shared(), b.Shared())[0]));

        var isolated = AltLoadContext();
        Assert.Equal(2, recorder.Fed.Count);
        Assert.All(recorder.Fed, v => Assert.Same(isolated, LoadContextOf(v)));

        var built = recorder.Fed.ToArray();
        recorder.Fed.Clear();
        Assert.Equal(expected, SideBySideModel.Floats(onAlt.Execute(graph, a.Shared(), b.Shared())[0]));
        Assert.Equal(built, recorder.Fed);

        Assert.Equal(expected, SideBySideModel.Floats(new ComputeContext().Execute(graph, a.Shared(), b.Shared())[0]));
        foreach (var literal in (TensorData[])[a, b])
        {
            Assert.Same(AssemblyLoadContext.Default, LoadContextOf(literal.ToTensorValue()));
            Assert.DoesNotContain(literal.ToTensorValue(), built);
        }
    }

    [Fact]
    public void TestTwoBackendsOverOneRuntimeReadEachOthersMemoryAndAnIsolatedRuntimeCopiesIt()
    {
        var (graph, a, b, expected) = Model();
        var secondDefault = (IShorokooBackend)Activator.CreateInstance(DefaultBackend.Instance.GetType())!;
        using var first = new ComputeContext();
        using var sameRuntime = new ComputeContext(secondDefault);
        using var isolated = new ComputeContext(Alt.Value);

        Assert.Same(DefaultBackend.Instance.RuntimeIdentity, secondDefault.RuntimeIdentity);
        Assert.NotSame(DefaultBackend.Instance.RuntimeIdentity, Alt.Value.RuntimeIdentity);

        var fromFirst = first.Execute(graph, a.Shared(), b.Shared())[0].ToTensorData();
        var fromIsolated = isolated.Execute(graph, a.Shared(), b.Shared())[0].ToTensorData();

        Assert.Same(fromFirst, fromFirst.To(sameRuntime));
        Assert.Same(fromIsolated, fromIsolated.To(isolated));
        Assert.Same(a, a.To(isolated));
        Assert.NotSame(fromFirst, fromFirst.To(isolated));
        var copied = fromIsolated.To(first);
        Assert.NotSame(fromIsolated, copied);
        Assert.Equal(expected, [.. copied.As<float32>().AccessMemory<float>()]);
    }

    [Fact]
    public void TestAFedTensorsRuntimeCopyIsReleasedThroughTheBackendThatBuiltIt()
    {
        var (graph, a, b, _) = Model();
        var recorder = new RecordingBackend(Alt.Value);
        using var onAlt = new ComputeContext(recorder);

        onAlt.Execute(graph, a.Shared(), b.Shared());
        var built = recorder.Fed.ToArray();
        Assert.Empty(recorder.Released);

        onAlt.Dispose();
        Assert.Empty(recorder.Released);

        a.Delete();
        b.Delete();
        Assert.True(built.SequenceEqual(recorder.Released, ReferenceEqualityComparer.Instance));
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
    /// <summary>A value that reports device residency, and optionally a type that is neither a
    /// tensor nor a sequence — the two things BackendTransfer refuses.</summary>
    private sealed class UnreadableValue : IShorokooTensorValue
    {
        internal bool Map { get; init; }

        public bool IsHostAccessible => false;
        public ShorokooOnnxValueType ValueType =>
            Map ? ShorokooOnnxValueType.Map : ShorokooOnnxValueType.Tensor;
        public ShorokooTensorElementType ElementType => ShorokooTensorElementType.Float;
        public long[] Shape => [2L];

        public ReadOnlySpan<T> GetTensorDataAsSpan<T>() where T : unmanaged
            => throw new NotSupportedException();
        public Span<T> GetTensorMutableDataAsSpan<T>() where T : unmanaged
            => throw new NotSupportedException();
        public IReadOnlyList<string> GetStringTensorData() => throw new NotSupportedException();
        public int GetValueCount() => throw new NotSupportedException();
        public IShorokooTensorValue GetValue(int index) => throw new NotSupportedException();
        public ShorokooTensorElementType GetSequenceElementType() => throw new NotSupportedException();
        public void Dispose() { }
    }

    private sealed class RecordingBackend(IShorokooBackend inner)
        : IShorokooBackend
    {
        internal List<IShorokooTensorValue> Fed { get; } = [];

        internal List<ShorokooGraphOptimization> Sessions { get; } = [];

        internal List<IShorokooTensorValue> Released { get; } = [];

        public BackendDescription Description => inner.Description;

        public MemorySpace MemorySpace => inner.MemorySpace;

        public object RuntimeIdentity => inner.RuntimeIdentity;

        public bool CanAddress(MemoryLocation location) => inner.CanAddress(location);

        public bool AcceptsTrainingFormat(string format) => inner.AcceptsTrainingFormat(format);

        public void Release(IShorokooTensorValue value)
        {
            Released.Add(value);
            inner.Release(value);
        }

        public IShorokooSession CreateSession(
            ReadOnlyMemory<byte> modelBytes, ShorokooGraphOptimization graphOptimization,
            ShorokooLogSeverity logSeverity, DeviceMemorySettings deviceMemory)
        {
            Sessions.Add(graphOptimization);
            return new RecordingSession(
                inner.CreateSession(modelBytes, graphOptimization, logSeverity, deviceMemory), Fed);
        }

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

        public IShorokooTensorValue CreateUninitializedTensorInBackendMemory(
            ShorokooTensorElementType elementType, long[] shape)
            => inner.CreateUninitializedTensorInBackendMemory(elementType, shape);
    }

    /// <summary>A session that notes what it was fed on its way to running it.</summary>
    private sealed class RecordingSession(
        IShorokooSession inner, List<IShorokooTensorValue> fed) : IShorokooSession
    {
        public IReadOnlyList<string> InputNames => inner.InputNames;
        public IReadOnlyList<string> OutputNames => inner.OutputNames;
        public bool HasDeviceMemory => inner.HasDeviceMemory;

        public IReadOnlyList<IShorokooTensorValue> Run(
            IReadOnlyDictionary<string, IShorokooTensorValue> inputs, IReadOnlyList<string> outputNames,
            RunSettings runSettings)
        {
            fed.AddRange(inputs.Values);
            return inner.Run(inputs, outputNames, runSettings);
        }

        public IReadOnlyList<IShorokooTensorValue> RunRetainingOutputs(
            IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
            IReadOnlyList<string> outputNames, IReadOnlySet<string> retainedOutputNames,
            RunSettings runSettings)
        {
            fed.AddRange(inputs.Values);
            return inner.RunRetainingOutputs(
                inputs, outputNames, retainedOutputNames, runSettings);
        }

        public void Dispose() => inner.Dispose();
    }

    [Fact]
    public void TestDataCrossesBetweenBackendsInBothDirectionsAndSurvivesARoundTrip()
    {
        var (graph, a, b, expected) = Model();
        var first = new ComputeContext();
        var second = new ComputeContext(Alt.Value);

        var fromAlt = second.Execute(graph, a.Shared(), b.Shared())[0].ToTensorData();
        var throughDefault = SideBySideModel.Floats(first.Execute(graph, fromAlt.Shared(), b.Shared())[0]);
        Assert.Equal([.. expected.Zip([10f, 20f, 30f, 40f], (x, y) => x * y + x)], throughDefault);

        var fromDefault = first.Execute(graph, a, b.Shared())[0].ToTensorData();
        Assert.Equal(expected, [.. fromDefault.As<float32>().AccessMemory<float>()]);
        Assert.Equal(
            [.. expected.Zip([10f, 20f, 30f, 40f], (x, y) => x * y + x)],
            SideBySideModel.Floats(second.Execute(graph, fromDefault, b)[0]));

        Assert.Equal(expected, [.. fromAlt.As<float32>().AccessMemory<float>()]);
        Assert.True(fromDefault.IsDisposed);
    }

    [Fact]
    public void TestBackendTransferRebuildsATensorAStringTensorAndASequenceOnTheOtherBackend()
    {
        var target = Alt.Value;
        var source = DefaultBackend.Instance;

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

        // A value the execution provider kept cannot cross: rebuilding it reads the source, and
        // there is no path from one runtime's device allocation to another's. This is the refusal
        // inference.md promises users, and the message has to name the way home.
        var refused = Assert.Throws<InvalidOperationException>(
            () => BackendTransfer.CopyTo(target, new UnreadableValue()));
        Assert.Contains("StepToCheckpoint", refused.Message);

        // And a value that is neither a tensor nor a sequence has no contents to rebuild at all.
        Assert.Throws<InvalidOperationException>(
            () => BackendTransfer.CopyTo(target, new UnreadableValue { Map = true }));
    }

    [Fact]
    public void TestLoadingAnIsolatedBackendRefusesEveryWayOfNamingOneThatIsNotThere()
    {
        var factoryAssembly = PlatformBackendAssembly(gpu: false);
        IsolatedBackendSpec Spec(string name, string assembly, string native, string? probe = null)
            => new() { Name = name, BackendAssembly = assembly, NativeRuntimePath = native, ProbeDirectory = probe };

        Assert.Throws<ArgumentNullException>(() => IsolatedBackend.Load(null!));
        Assert.Throws<ArgumentException>(() => IsolatedBackend.Load(Spec(" ", factoryAssembly, AltRuntimePath)));
        Assert.Throws<ArgumentException>(() => IsolatedBackend.Load(Spec("x", "", AltRuntimePath)));
        Assert.Throws<ArgumentException>(() => IsolatedBackend.Load(Spec("x", factoryAssembly, "")));

        var missingNative = Path.Combine(BackendRoot, "nowhere", "libonnxruntime.so");
        Assert.Contains("nowhere", Assert.Throws<FileNotFoundException>(
            () => IsolatedBackend.Load(Spec("x", factoryAssembly, missingNative))).Message);

        Assert.Contains("Shorokoo.NotABackend", Assert.Throws<FileNotFoundException>(
            () => IsolatedBackend.Load(Spec("x", "Shorokoo.NotABackend", AltRuntimePath))).Message);

        Assert.Contains("exposes no concrete", Assert.Throws<InvalidOperationException>(
            () => IsolatedBackend.Load(Spec("x", "Shorokoo", AltRuntimePath))).Message);

        Assert.Contains("elsewhere", Assert.Throws<FileNotFoundException>(
            () => IsolatedBackend.Load(Spec("x", factoryAssembly, AltRuntimePath, "/elsewhere"))).Message);
    }

    [Fact]
    public void TestARigMergesOnOneBackendAndTrainsOnAnother()
    {
        // Recording backends, because the claim is about WHERE each phase ran. Asserting that the
        // rig kept the two contexts and that the loss fell says nothing about it: a rig that used
        // RuntimeContext for the merge as well, or ignored MergeContext entirely, passes both.
        var mergeBackend = new RecordingBackend(DefaultBackend.Instance);
        var runtimeBackend = new RecordingBackend(Alt.Value);
        using var merge = new ComputeContext(mergeBackend);
        using var runtime = new ComputeContext(runtimeBackend);
        var rig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, AdamWOptimizer.ComputationGraph,
            [new TensorDataModelParam("input", ModelParamType.InputParam, TensorData([4L], [1f, 2f, 3f, 4f]))],
            new AdamWOptimizerHyperparameters { LearningRate = 0.1f },
            rngConfig: null, mergeContext: merge, runtimeContext: runtime);

        Assert.Same(merge, rig.MergeContext);
        Assert.Same(runtime, rig.RuntimeContext);
        Assert.NotEqual(merge.Backend.Name, runtime.Backend.Name);

        var first = rig.TrainStep(
            rig.CreateInitialCheckpoint(),
            TrainingRigHelpers.InBatch(1f, 2f, 3f, 4f), TrainingRigHelpers.TargetBatch(2f, 4f, 6f, 8f));
        var second = rig.TrainStep(
            first, TrainingRigHelpers.InBatch(1f, 2f, 3f, 4f), TrainingRigHelpers.TargetBatch(2f, 4f, 6f, 8f));

        Assert.True(second.Loss < first.Loss);

        // Each phase built sessions on its own backend, and neither built any on the other's.
        Assert.NotEmpty(mergeBackend.Sessions);
        Assert.NotEmpty(runtimeBackend.Sessions);
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
