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

        // And the second copy bound the native the spec named, not another one the ordinary
        // probing would have found. Two contexts prove nothing on their own -- a LoadUnmanagedDll
        // that declined would leave both on the same file and every test above would still pass.
        // The runtimes say which they are: the stock build carries no CUDA execution provider,
        // the CUDA-flavoured build does. On the provider set rather than on a whole-list equality
        // because the stock build's set is not the same on every platform -- the Windows one
        // carries the Azure provider beside the CPU one, the Linux one carries the CPU alone --
        // whereas CUDA's absence from the one and presence in the other is what distinguishes
        // the runtimes, on either platform.
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

    /// <summary>
    /// Loading a backend must not cost the program its default one. Discovery refuses two backend
    /// assemblies, and an isolated backend's is a backend assembly loaded into the process — so
    /// without the load-context filter, loading one here makes the <i>next</i> first read of
    /// <see cref="InferenceBackend.Factory"/> throw, anywhere in the process. The suite only
    /// survives it when something has already resolved the default, which is an ordering it does
    /// not control.
    /// </summary>
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
