using System.Collections.Concurrent;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Loader;

namespace Shorokoo.Core.Backends;

/// <summary>
/// Which backend to load, and off which native ONNX Runtime. Two specs differing in
/// <see cref="NativeRuntimePath"/> name two runtimes that share nothing.
/// </summary>
public sealed record IsolatedBackendSpec
{
    /// <summary>
    /// What this backend is called in <see cref="BackendDescription.Name"/> — in a log, in
    /// <see cref="DefaultBackend.Describe"/>, in an error. It replaces the backend assembly's
    /// name, which cannot tell two loads of one assembly apart. Give it something a reader can
    /// act on: <c>cuda:0</c>, <c>cpu</c>, <c>ort-1.22</c>.
    /// </summary>
    public required string Name { get; init; }

    /// <summary>
    /// The simple name of the assembly holding the backend to load — <c>Shorokoo.LinuxGPU</c>,
    /// <c>Shorokoo.WinCPU</c>, or your own. It is loaded privately to this backend, as is the
    /// ONNX Runtime glue beneath it, so its types are this backend's alone.
    /// </summary>
    public required string BackendAssembly { get; init; }

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
/// CPU provider too — so two ordinary backends over the one loaded runtime already give a
/// CPU context and a CUDA context side by side:
/// </para>
/// <code>
/// var cpu  = new ComputeContext(new LinuxCpuBackend());
/// var cuda = new ComputeContext(new LinuxGpuBackend());
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
///     Name = "cpu",  BackendAssembly = "Shorokoo.LinuxCPU",
///     NativeRuntimePath = Path.Combine(AppContext.BaseDirectory, "ort/cpu/libonnxruntime.so") });
/// var cuda = IsolatedBackend.Load(new IsolatedBackendSpec {
///     Name = "cuda", BackendAssembly = "Shorokoo.LinuxGPU",
///     NativeRuntimePath = Path.Combine(AppContext.BaseDirectory, "ort/cuda/libonnxruntime.so") });
/// </code>
/// <para>
/// Only the ONNX Runtime wrapper, the glue over it and the backend assembly are private to a
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

    private static readonly ConcurrentDictionary<IsolatedBackendSpec, IShorokooBackend> _loaded = new();

    /// <summary>
    /// Loads the backend <paramref name="spec"/> names, bound to the native ONNX Runtime at
    /// <see cref="IsolatedBackendSpec.NativeRuntimePath"/>, and returns it —
    /// ready to hand to a <c>ComputeContext</c>. Loading the same spec again returns the same
    /// backend.
    /// </summary>
    /// <exception cref="ArgumentNullException"><paramref name="spec"/> is null.</exception>
    /// <exception cref="ArgumentException">A required field of the spec is blank.</exception>
    /// <exception cref="FileNotFoundException">The native runtime, or the backend assembly,
    /// is not where the spec says.</exception>
    /// <exception cref="InvalidOperationException">The backend assembly holds no usable
    /// backend.</exception>
    public static IShorokooBackend Load(IsolatedBackendSpec spec)
    {
        ArgumentNullException.ThrowIfNull(spec);
        Require(spec.Name, nameof(IsolatedBackendSpec.Name));
        Require(spec.BackendAssembly, nameof(IsolatedBackendSpec.BackendAssembly));
        Require(spec.NativeRuntimePath, nameof(IsolatedBackendSpec.NativeRuntimePath));

        // Keyed on the resolved paths rather than on the spec as written, so that two spellings
        // of one file -- a relative path and an absolute one -- are one backend rather than two
        // wrappers over the same native. Name is cleared out of the key for the same reason: it is
        // a label for logs, not part of what is loaded, and leaving it in meant two names over one
        // file silently built two load contexts, two private copies of the ONNX Runtime wrapper and
        // two OrtCreateEnv calls against a single dlopen'd runtime.
        var key = spec with
        {
            Name = "",
            NativeRuntimePath = Path.GetFullPath(spec.NativeRuntimePath),
            // Resolved rather than left null, because LoadUncached resolves null to exactly this
            // directory: leaving the two spellings distinct made a TryLoad of a backend beside the
            // core assembly and a hand-written Load of the same file two entries and two contexts.
            ProbeDirectory = spec.ProbeDirectory is { Length: > 0 } dir
                ? Path.GetFullPath(dir) : ProbeDirectory(),
        };

        // Not GetOrAdd: loading is expensive, throws for several distinct reasons, and pins a
        // native library for the life of the process. GetOrAdd may run its value factory more than
        // once under contention, and a second load of the same spec is exactly the waste this
        // cache exists to prevent.
        if (_loaded.TryGetValue(key, out var cached)) return Named(cached, spec);
        lock (_loaded)
        {
            if (_loaded.TryGetValue(key, out cached)) return Named(cached, spec);
            var loaded = LoadUncached(key with { Name = spec.Name });
            _loaded[key] = loaded;
            return loaded;
        }
    }

    /// <summary>
    /// The already-loaded backend, refusing a second name for it. Two names would be two backends
    /// to every caller — a context reports one, a log records one — while being one runtime, so the
    /// distinction they are asking for does not exist. Saying so beats handing back a backend whose
    /// name is not the one that was asked for.
    /// </summary>
    private static IShorokooBackend Named(
        IShorokooBackend loaded, IsolatedBackendSpec spec)
        => loaded.Description.Name == spec.Name ? loaded
            : throw new InvalidOperationException(
                $"The native ONNX Runtime at '{Path.GetFullPath(spec.NativeRuntimePath)}' is already "
                + $"loaded as the backend '{loaded.Description.Name}', so it cannot also be loaded "
                + $"as '{spec.Name}'. One file is one runtime; give the second backend a native of "
                + "its own, or use the name it already has.");

    private static void Require(string value, string field)
    {
        if (string.IsNullOrWhiteSpace(value))
            throw new ArgumentException($"An isolated backend's {field} is required.", nameof(IsolatedBackendSpec));
    }

    private static IShorokooBackend LoadUncached(IsolatedBackendSpec spec)
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
        var backendPath = Path.Combine(probeDirectory, spec.BackendAssembly + ".dll");
        if (!File.Exists(backendPath))
            throw new FileNotFoundException(
                $"The isolated backend '{spec.Name}' loads its backend type from " +
                $"'{spec.BackendAssembly}', which is not in '{probeDirectory}'. Reference that " +
                "backend package, or set IsolatedBackendSpec.ProbeDirectory to where it is deployed.",
                backendPath);

        // The glue and the ONNX Runtime wrapper are looked for beside the backend first and beside
        // the core assembly second, because a backend deployed in a folder of its own normally
        // carries only its backend there and shares the rest with the ordinary build.
        // A load context cannot be unloaded -- it is deliberately non-collectible, because
        // finalizers releasing ORT handles must not race an assembly teardown -- so one built for a
        // load that then fails is resident for the life of the process, holding its private copies
        // of the wrapper and the glue, with nothing referencing it and no way to reclaim it. That
        // matters because failing is routine here: TryLoad catches everything so a caller can walk
        // a folder of candidates, and ten unloadable files used to mean ten permanent load contexts.
        // So the contexts are kept and reused per key, whether or not the backend behind one came
        // up, and a retry of a spec that failed reuses the context its first attempt built. Walking
        // a folder of distinct candidates still costs one per candidate that gets this far, which
        // is inherent: finding out whether a file is a backend means loading it. Probe turns most
        // away from metadata before any of this.
        var context = LoadContextFor(spec, probeDirectory, native);
        var backend = InstantiateBackend(context.LoadFromAssemblyPath(backendPath))
            ?? throw new InvalidOperationException(
                $"'{spec.BackendAssembly}' was loaded from '{backendPath}' for the isolated " +
                $"backend '{spec.Name}' but exposes no concrete " +
                $"{nameof(IShorokooBackend)} with a parameterless constructor.");

        // Deliberately not remembered. A backend loaded here is one the program named, and
        // discovery's question is which backend a program that named none meant -- so this is no
        // answer to it, as Documentation/inference.md says under Auto-discovery. Remembering it
        // made whichever isolated backend happened to load first the one every unnamed context
        // ran on, and, since a CPU one is never displaced, permanently.
        return new RenamedBackend(backend, spec.Name);
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
    /// <para>A backend that does not sit on ONNX Runtime has no wrapper to bind here, and
    /// its native is left to the load context to resolve.</para>
    /// </summary>
    // Keyed on the same thing the backend cache is, and held for the same reason a loaded native
    // is: it can never be released. LoadContexts is the count of them, which is the only way a test
    // can see that a failed load did not build a second one.
    private static readonly Dictionary<string, BackendLoadContext> _contexts = [];

    internal static int LoadContexts { get { lock (_loaded) return _contexts.Count; } }

    private static BackendLoadContext LoadContextFor(
        IsolatedBackendSpec spec, string probeDirectory, string native)
    {
        // The backend assembly is part of the key, not just the native and the probe directory. A
        // load that failed may have loaded assemblies into its context first -- a private copy of
        // the core assembly, say -- and handing that context to a different backend gives it types
        // that are not the program's, so the cast to the backend interface fails. Only a retry of
        // the identical spec can safely reuse one, which is exactly the case that was unbounded.
        var key = $"{native}|{probeDirectory}|{spec.BackendAssembly}";
        if (_contexts.TryGetValue(key, out var existing)) return existing;
        var created = new BackendLoadContext(spec.Name, [probeDirectory, ProbeDirectory()], native);
        BindNativeRuntime(created, native);
        _contexts[key] = created;
        return created;
    }

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

    private static IShorokooBackend? InstantiateBackend(Assembly asm)
    {
        var type = asm.GetExportedTypes().FirstOrDefault(t =>
            typeof(IShorokooBackend).IsAssignableFrom(t)
            && !t.IsAbstract
            && t.GetConstructor(Type.EmptyTypes) is not null);
        // Unlike the discovery path in DefaultBackend, a failure to construct is not swallowed:
        // there is no other candidate to fall through to, and the caller named this one.
        return type is null ? null : (IShorokooBackend)Activator.CreateInstance(type)!;
    }

    /// <summary>
    /// The load context one backend lives in: it resolves the ONNX Runtime wrapper, the glue
    /// over it and the backend assembly to its own private copies, lets everything else resolve
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

        // The backend assembly resolves through LoadFromAssemblyPath in LoadUncached rather than
        // through here, so this names only what the backend pulls in behind it.
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
    /// A backend reporting the name its spec gave it, forwarding everything else to the one
    /// loaded in isolation. Without it two backends off one assembly describe themselves
    /// identically, and a log saying which device a run used stops saying it.
    ///
    /// <para>It forwards rather than copies: every tensor and session still comes from the inner
    /// backend, so they carry that backend's runtime types — which is how a session recognises a
    /// value as its own or as another backend's.</para>
    /// </summary>
    private sealed class RenamedBackend : IShorokooBackend
    {
        private readonly IShorokooBackend _inner;

        internal RenamedBackend(IShorokooBackend inner, string name)
        {
            _inner = inner;
            var described = inner.Description;
            Description = new BackendDescription(name, described.Device, described.CudaDeviceId);
        }

        public BackendDescription Description { get; }

        public IShorokooSession CreateSession(
            ReadOnlyMemory<byte> modelBytes,
            ShorokooGraphOptimization graphOptimization,
            ShorokooLogSeverity logSeverity,
            DeviceMemorySettings deviceMemory)
            => _inner.CreateSession(modelBytes, graphOptimization, logSeverity, deviceMemory);

        public IShorokooSession CreateSession(
            ReadOnlyMemory<byte> modelBytes,
            ShorokooGraphOptimization graphOptimization,
            ShorokooLogSeverity logSeverity,
            DeviceMemorySettings deviceMemory,
            DiagnosticSettings diagnostics)
            => _inner.CreateSession(modelBytes, graphOptimization, logSeverity, deviceMemory, diagnostics);

        public IShorokooSession CreateSession(
            ReadOnlyMemory<byte> modelBytes,
            ShorokooGraphOptimization graphOptimization,
            ShorokooLogSeverity logSeverity,
            DeviceMemorySettings deviceMemory,
            DiagnosticSettings diagnostics,
            IReadOnlyList<OutputAlias> outputAliases)
            => _inner.CreateSession(
                modelBytes, graphOptimization, logSeverity, deviceMemory, diagnostics, outputAliases);

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

        public IShorokooTensorValue CreateUninitializedTensorInBackendMemory(
            ShorokooTensorElementType elementType, long[] shape)
            => _inner.CreateUninitializedTensorInBackendMemory(elementType, shape);

        public IShorokooTensorValue CreateSequence(IReadOnlyList<IShorokooTensorValue> values)
            => _inner.CreateSequence(values);

        // The runtime is the inner backend's, which is the whole reason this wrapper forwards
        // rather than answers: a tensor allocated through the wrapper records the wrapper as its
        // allocating backend, and its location has to name the runtime the allocation really
        // belongs to, or the inner backend -- asked below whether it can address it -- would not
        // recognise its own memory.
        public object RuntimeIdentity => _inner.RuntimeIdentity;

        public bool CanAddress(MemoryLocation location) => _inner.CanAddress(location);

        public MemoryLocation RunMemoryOf(ShorokooTensorElementType elementType) => _inner.RunMemoryOf(elementType);

        public MemoryLocation SequenceRunMemory => _inner.SequenceRunMemory;

        public void Release(IShorokooTensorValue value) => _inner.Release(value);
    }
}
