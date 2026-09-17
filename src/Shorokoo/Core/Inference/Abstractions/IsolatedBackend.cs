using System.Collections.Concurrent;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Loader;

namespace Shorokoo.Core.Inference.Abstractions;

/// <summary>
/// Which backend to load, and off which native ONNX Runtime. Two specs differing in
/// <see cref="NativeRuntimePath"/> name two runtimes that share nothing.
/// </summary>
public sealed record IsolatedBackendSpec
{
    /// <summary>
    /// What this backend is called in <see cref="BackendDescription.Name"/> — in a log, in
    /// <see cref="InferenceBackend.Describe"/>, in an error. It replaces the factory assembly's
    /// name, which cannot tell two loads of one assembly apart. Give it something a reader can
    /// act on: <c>cuda:0</c>, <c>cpu</c>, <c>ort-1.22</c>.
    /// </summary>
    public required string Name { get; init; }

    /// <summary>
    /// The simple name of the assembly holding the factory to load — <c>Shorokoo.LinuxGPU</c>,
    /// <c>Shorokoo.WinCPU</c>, or your own. It is loaded privately to this backend, as is the
    /// ONNX Runtime glue beneath it, so its types are this backend's alone.
    /// </summary>
    public required string FactoryAssembly { get; init; }

    /// <summary>
    /// The full path to the native ONNX Runtime this backend binds to
    /// (<c>libonnxruntime.so</c> / <c>onnxruntime.dll</c>). Two backends wanting separate
    /// runtimes need two separate <i>files</i>: the loader keys on the path, so pointing both
    /// at one file gets one runtime with two wrappers over it.
    ///
    /// <para>ONNX Runtime finds an execution provider's own library — CUDA's
    /// <c>libonnxruntime_providers_cuda.so</c>, say — next to this one, so a backend's whole
    /// native set belongs in one folder of its own.</para>
    /// </summary>
    public required string NativeRuntimePath { get; init; }

    /// <summary>
    /// Where to find the managed assemblies this backend loads privately. Null probes the
    /// folder holding the core Shorokoo assembly, which is where a normal build puts them.
    /// </summary>
    public string? ProbeDirectory { get; init; }
}

/// <summary>
/// A backend loaded into isolation of its own, so that it binds a native ONNX Runtime no
/// other backend in the process shares.
///
/// <para>
/// Most programs that want two devices at once do not need this. One ONNX Runtime build
/// serves every execution provider compiled into it — the CUDA-flavoured native carries the
/// CPU provider too — so two ordinary factories over the one loaded runtime already give a
/// CPU context and a CUDA context side by side:
/// </para>
/// <code>
/// var cpu  = new ComputeContext(new LinuxCpuInferenceFactory());
/// var cuda = new ComputeContext(new LinuxGpuInferenceFactory());
/// </code>
/// <para>
/// What needs isolation is a <i>second runtime</i>: two ONNX Runtime builds, or two versions,
/// in one process. The obstacle there is not ONNX Runtime — two of its natives load and run
/// side by side quite happily — but that its managed wrapper binds the whole native API into
/// <c>static</c> state on first use, once per copy of the assembly. So this gives each backend
/// its own copy, in an <see cref="AssemblyLoadContext"/> that resolves the unmanaged
/// <c>onnxruntime</c> to the path the spec names:
/// </para>
/// <code>
/// var cpu  = IsolatedBackend.Load(new IsolatedBackendSpec {
///     Name = "cpu",  FactoryAssembly = "Shorokoo.LinuxCPU",
///     NativeRuntimePath = Path.Combine(AppContext.BaseDirectory, "ort/cpu/libonnxruntime.so") });
/// var cuda = IsolatedBackend.Load(new IsolatedBackendSpec {
///     Name = "cuda", FactoryAssembly = "Shorokoo.LinuxGPU",
///     NativeRuntimePath = Path.Combine(AppContext.BaseDirectory, "ort/cuda/libonnxruntime.so") });
/// </code>
/// <para>
/// Only the ONNX Runtime wrapper, the glue over it and the factory assembly are private to a
/// backend. Everything else — the core Shorokoo assembly, the abstractions in this namespace,
/// the framework types your model is written in — resolves to the one copy the program already
/// loaded, so a <see cref="BackendDescription"/> or an <see cref="IShorokooTensorValue"/> is
/// the same type whichever backend produced it. That is what lets a
/// <c>ComputeContext</c> on one backend be handed a tensor from another: it is a type it
/// understands, from a runtime it does not share, and the session copies it across (host
/// memory only — see <see cref="BackendTransfer"/>).
/// </para>
/// <para>
/// A loaded backend lasts for the life of the process. Its native runtime holds thread pools,
/// arenas and allocators that outlive any object here, and nothing unloads it; loading the same
/// spec twice returns the same backend rather than a second copy.
/// </para>
/// </summary>
public static class IsolatedBackend
{
    // The assemblies a backend must not share, because each of them either holds ONNX Runtime's
    // static native binding or is compiled against it. Everything else resolves to the copy the
    // program already loaded, which is what keeps the abstraction types common.
    private const string OrtManagedAssembly = "Microsoft.ML.OnnxRuntime";
    private const string OrtGlueAssembly = "Shorokoo.OnnxRuntime";

    // The bare name ONNX Runtime's wrapper imports every entry point from.
    private const string OrtNativeLibrary = "onnxruntime";

    private static readonly ConcurrentDictionary<IsolatedBackendSpec, IShorokooInferenceSessionFactory> _loaded = new();

    /// <summary>
    /// Loads the backend <paramref name="spec"/> names, bound to the native ONNX Runtime at
    /// <see cref="IsolatedBackendSpec.NativeRuntimePath"/>, and returns its factory —
    /// ready to hand to a <c>ComputeContext</c>. Loading the same spec again returns the same
    /// backend.
    /// </summary>
    /// <exception cref="ArgumentNullException"><paramref name="spec"/> is null.</exception>
    /// <exception cref="ArgumentException">A required field of the spec is blank.</exception>
    /// <exception cref="FileNotFoundException">The native runtime, or the factory assembly,
    /// is not where the spec says.</exception>
    /// <exception cref="InvalidOperationException">The factory assembly holds no usable
    /// factory.</exception>
    public static IShorokooInferenceSessionFactory Load(IsolatedBackendSpec spec)
    {
        ArgumentNullException.ThrowIfNull(spec);
        Require(spec.Name, nameof(IsolatedBackendSpec.Name));
        Require(spec.FactoryAssembly, nameof(IsolatedBackendSpec.FactoryAssembly));
        Require(spec.NativeRuntimePath, nameof(IsolatedBackendSpec.NativeRuntimePath));

        // Keyed on the resolved paths rather than on the spec as written, so that two spellings
        // of one file -- a relative path and an absolute one -- are one backend rather than two
        // wrappers over the same native.
        var key = spec with
        {
            NativeRuntimePath = Path.GetFullPath(spec.NativeRuntimePath),
            ProbeDirectory = spec.ProbeDirectory is { Length: > 0 } dir ? Path.GetFullPath(dir) : null,
        };

        // Not GetOrAdd: loading is expensive, throws for several distinct reasons, and pins a
        // native library for the life of the process. GetOrAdd may run its factory more than
        // once under contention, and a second load of the same spec is exactly the waste this
        // cache exists to prevent.
        if (_loaded.TryGetValue(key, out var cached)) return cached;
        lock (_loaded)
        {
            if (_loaded.TryGetValue(key, out cached)) return cached;
            var loaded = LoadUncached(key);
            _loaded[key] = loaded;
            return loaded;
        }
    }

    private static void Require(string value, string field)
    {
        if (string.IsNullOrWhiteSpace(value))
            throw new ArgumentException($"An isolated backend's {field} is required.", nameof(IsolatedBackendSpec));
    }

    private static IShorokooInferenceSessionFactory LoadUncached(IsolatedBackendSpec spec)
    {
        var native = Path.GetFullPath(spec.NativeRuntimePath);
        if (!File.Exists(native))
            throw new FileNotFoundException(
                $"The isolated backend '{spec.Name}' binds the native ONNX Runtime at '{native}', " +
                "which is not there. Each backend needs its own copy of the native in a folder of " +
                "its own, alongside any execution-provider libraries it loads.",
                native);

        var probeDirectory = spec.ProbeDirectory is { Length: > 0 } given
            ? Path.GetFullPath(given)
            : ProbeDirectory();
        var factoryPath = Path.Combine(probeDirectory, spec.FactoryAssembly + ".dll");
        if (!File.Exists(factoryPath))
            throw new FileNotFoundException(
                $"The isolated backend '{spec.Name}' loads its factory from " +
                $"'{spec.FactoryAssembly}', which is not in '{probeDirectory}'. Reference that " +
                "backend package, or set IsolatedBackendSpec.ProbeDirectory to where it is deployed.",
                factoryPath);

        // The glue and the ONNX Runtime wrapper are looked for beside the factory first and beside
        // the core assembly second, because a backend deployed in a folder of its own normally
        // carries only its factory there and shares the rest with the ordinary build.
        var context = new BackendLoadContext(spec.Name, [probeDirectory, ProbeDirectory()], native);
        BindNativeRuntime(context, native);
        var factory = InstantiateFactory(context.LoadFromAssemblyPath(factoryPath))
            ?? throw new InvalidOperationException(
                $"'{spec.FactoryAssembly}' was loaded from '{factoryPath}' for the isolated " +
                $"backend '{spec.Name}' but exposes no concrete " +
                $"{nameof(IShorokooInferenceSessionFactory)} with a parameterless constructor.");

        var loaded = new RenamedFactory(factory, spec.Name);
        // A backend loaded here is a backend this process has, so it counts towards the one the
        // default context is built on -- under the same rule, where a CPU backend wins.
        InferenceBackend.Remember(loaded);
        return loaded;
    }

    /// <summary>
    /// Points this backend's private copy of the ONNX Runtime wrapper at the native the spec
    /// names, by registering a resolver for its <c>DllImport</c>s before anything can trigger
    /// them.
    ///
    /// <para>The load context's own <c>LoadUnmanagedDll</c> is not enough on its own, and the
    /// order is the whole reason. ONNX Runtime's wrapper registers a
    /// <see cref="System.Runtime.InteropServices.NativeLibrary.SetDllImportResolver"/> of its own
    /// the first time its native methods are touched, and a resolver registered on the assembly
    /// runs <i>before</i> the load context is consulted. Its resolver probes next to the assembly
    /// — where a normal build has already deployed a runtime — so it would find that one and the
    /// context's hook would never be asked. Registering first means ours is the one that answers:
    /// an assembly takes only one resolver, so the wrapper's own attempt then fails, which it
    /// expects and handles (it is how a host is meant to take this over).</para>
    ///
    /// <para>A backend whose factory does not sit on ONNX Runtime has no wrapper to bind here, and
    /// its native is left to the load context to resolve.</para>
    /// </summary>
    private static void BindNativeRuntime(BackendLoadContext context, string nativeRuntimePath)
    {
        if (context.PrivatePathOf(OrtManagedAssembly) is not { } ortPath) return;

        var ort = context.LoadFromAssemblyPath(ortPath);
        NativeLibrary.SetDllImportResolver(ort, (name, _, _) =>
            name.Equals(OrtNativeLibrary, StringComparison.OrdinalIgnoreCase)
                ? NativeLibrary.Load(nativeRuntimePath)
                : IntPtr.Zero);
    }

    private static string ProbeDirectory()
    {
        var location = typeof(IsolatedBackend).Assembly.Location;
        return !string.IsNullOrEmpty(location) && Path.GetDirectoryName(location) is { Length: > 0 } dir
            ? dir
            : AppContext.BaseDirectory;
    }

    private static IShorokooInferenceSessionFactory? InstantiateFactory(Assembly asm)
    {
        var type = asm.GetExportedTypes().FirstOrDefault(t =>
            typeof(IShorokooInferenceSessionFactory).IsAssignableFrom(t)
            && !t.IsAbstract
            && t.GetConstructor(Type.EmptyTypes) is not null);
        // Unlike the discovery path in InferenceBackend, a failure to construct is not swallowed:
        // there is no other candidate to fall through to, and the caller named this one.
        return type is null ? null : (IShorokooInferenceSessionFactory)Activator.CreateInstance(type)!;
    }

    /// <summary>
    /// The load context one backend lives in: it resolves the ONNX Runtime wrapper, the glue
    /// over it and the factory assembly to its own private copies, lets everything else resolve
    /// to the copy the program already loaded, and answers the unmanaged <c>onnxruntime</c> with
    /// this backend's native.
    /// </summary>
    private sealed class BackendLoadContext : AssemblyLoadContext
    {
        private readonly string[] _probeDirectories;
        private readonly string _nativeRuntimePath;

        // Not collectible. Unloading would reclaim the managed assemblies while leaving the
        // native runtime loaded with its thread pools, arenas and allocators still running --
        // and the finalizers releasing ORT handles would be racing an assembly teardown.
        internal BackendLoadContext(string name, string[] probeDirectories, string nativeRuntimePath)
            : base($"Shorokoo backend '{name}'", isCollectible: false)
        {
            _probeDirectories = probeDirectories;
            _nativeRuntimePath = nativeRuntimePath;
        }

        /// <summary>Where this backend's own copy of <paramref name="name"/> is, or null if it is
        /// not deployed anywhere this context probes.</summary>
        internal string? PrivatePathOf(string name)
            => _probeDirectories
                .Select(dir => Path.Combine(dir, name + ".dll"))
                .FirstOrDefault(File.Exists);

        protected override Assembly? Load(AssemblyName assemblyName)
        {
            if (assemblyName.Name is not { Length: > 0 } name || !IsPrivate(name)) return null;
            // Null falls through to the default context, which would quietly hand this backend the
            // copy the program already loaded -- and with it that copy's native runtime, which is
            // the one thing isolation exists to avoid. So say so instead.
            return PrivatePathOf(name) is { } path
                ? LoadFromAssemblyPath(path)
                : throw new FileNotFoundException(
                    $"The isolated backend {Name} needs its own copy of '{name}', which is not in "
                    + string.Join(" or ", _probeDirectories.Select(d => $"'{d}'"))
                    + ". Sharing the program's copy would share the native ONNX Runtime bound to "
                    + "it, which is what loading this backend in isolation is for.",
                    name + ".dll");
        }

        // The factory assembly resolves through LoadFromAssemblyPath in LoadUncached rather than
        // through here, so this names only what the factory pulls in behind it.
        private static bool IsPrivate(string name)
            => name.Equals(OrtManagedAssembly, StringComparison.OrdinalIgnoreCase)
               || name.Equals(OrtGlueAssembly, StringComparison.OrdinalIgnoreCase);

        protected override IntPtr LoadUnmanagedDll(string unmanagedDllName)
        {
            // The backstop behind BindNativeRuntime's resolver, for a backend whose wrapper this
            // context loaded without one being registered on it. Everything else -- a CUDA library
            // the provider pulls in, say -- goes through the ordinary search.
            if (!unmanagedDllName.Equals(OrtNativeLibrary, StringComparison.OrdinalIgnoreCase))
                return IntPtr.Zero;
            return NativeLibrary.Load(_nativeRuntimePath);
        }
    }

    /// <summary>
    /// A factory reporting the name its spec gave it, forwarding everything else to the one
    /// loaded in isolation. Without it two backends off one factory assembly describe
    /// themselves identically, and a log saying which device a run used stops saying it.
    ///
    /// <para>It forwards rather than copies: every tensor and session still comes from the inner
    /// factory, so they carry that backend's runtime types — which is how a session recognises a
    /// value as its own or as another backend's.</para>
    /// </summary>
    private sealed class RenamedFactory : IShorokooInferenceSessionFactory
    {
        private readonly IShorokooInferenceSessionFactory _inner;

        internal RenamedFactory(IShorokooInferenceSessionFactory inner, string name)
        {
            _inner = inner;
            var described = inner.Description;
            Description = new BackendDescription(name, described.Device, described.CudaDeviceId);
        }

        public BackendDescription Description { get; }

        public IShorokooInferenceSession CreateSession(
            ReadOnlyMemory<byte> modelBytes,
            ShorokooGraphOptimization graphOptimization,
            ShorokooLogSeverity logSeverity)
            => _inner.CreateSession(modelBytes, graphOptimization, logSeverity);

        public IShorokooTensorValue CreateTensor<T>(T[] data, long[] shape) where T : unmanaged
            => _inner.CreateTensor(data, shape);

        public IShorokooTensorValue CreateTensorFromRawBytes(
            ShorokooTensorElementType elementType, byte[] data, long[] shape)
            => _inner.CreateTensorFromRawBytes(elementType, data, shape);

        public IShorokooTensorValue CreateStringTensor(IReadOnlyList<string> data, long[] shape)
            => _inner.CreateStringTensor(data, shape);

        // Forwarded rather than inherited. Every member of the interface has to be, including the
        // ones with a default body: a default implementation is the wrapper's own, so leaving one
        // alone would answer for the backend instead of asking it -- and this one is exactly the
        // question only the backend that made the allocation can answer.
        public MemorySpace MemorySpace => _inner.MemorySpace;

        public byte[] CopyTensorToHost(IShorokooTensorValue value) => _inner.CopyTensorToHost(value);

        public IShorokooTensorValue CreateTensorInBackendMemory(
            ShorokooTensorElementType elementType, byte[] data, long[] shape)
            => _inner.CreateTensorInBackendMemory(elementType, data, shape);

        public IShorokooTensorValue CreateSequence(IReadOnlyList<IShorokooTensorValue> values)
            => _inner.CreateSequence(values);
    }
}
