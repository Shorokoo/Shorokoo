using System.Runtime.InteropServices;

namespace Shorokoo.OnnxRuntime;

/// <summary>
/// The native library ONNX Runtime holds Shorokoo's allocator through
/// (<c>native/shorokoo_ort_allocator.cpp</c>, built with this assembly): the <c>OrtAllocator</c>
/// whose entry points forward each request to a <c>CachingAllocator</c>, and which turns a request
/// that allocator refuses into the C++ exception ONNX Runtime's own allocators throw, so that the
/// call that asked fails rather than writing through a null block.
///
/// <para><b>Where it is found.</b> Beside this assembly, or under <c>runtimes/&lt;rid&gt;/native/</c>
/// there — the two places a build or a package deploys a native — and failing both, wherever the
/// host's own probing finds a native of this name (a <c>deps.json</c> naming it). Relative to this
/// assembly rather than to the program, so that a backend loaded from a folder of its own
/// (<c>IsolatedBackend</c>, <c>BackendPackage</c>) finds the copy deployed with the glue it
/// loaded. It links nothing of ONNX Runtime's, so one copy serves every runtime in a
/// process.</para>
/// </summary>
internal static unsafe class NativeAllocator
{
    /// <summary>The library's name, as a native import spells it.</summary>
    internal const string LibraryName = "shorokoo_ort_allocator";

    /// <summary>The one entry point it exports.</summary>
    internal const string CreateExport = "shorokoo_ort_allocator_create";

    /// <summary>The library's file on this operating system.</summary>
    internal static string FileName
        => OperatingSystem.IsWindows() ? LibraryName + ".dll" : "lib" + LibraryName + ".so";

    private static readonly Lazy<IntPtr> _create = new(Bind, LazyThreadSafetyMode.ExecutionAndPublication);

    private static readonly Lazy<string?> _located = new(Locate, LazyThreadSafetyMode.ExecutionAndPublication);

    /// <summary><see cref="Locate"/>, asked once: the path a session registers the library's
    /// operators from (<c>native/shorokoo_ort_ops.cpp</c>), or null where it is not deployed in
    /// either place.</summary>
    internal static string? Located => _located.Value;

    /// <summary>
    /// A native <c>OrtAllocator</c> over the managed allocator <paramref name="state"/> names,
    /// whose requests go to the entry points <paramref name="allocate"/>
    /// (<c>(state, size, reason, capacity)</c>, answering a block or null with the reason written
    /// down), <paramref name="allocateOnStream"/> (<c>(state, size, stream, reason, capacity)</c>,
    /// the same for a request ONNX Runtime makes on one of its streams) and <paramref name="free"/>
    /// (<c>(state, block)</c>), describing itself with the native memory info <paramref name="info"/>.
    /// It lives for the life of the process.
    /// </summary>
    /// <exception cref="DllNotFoundException">The library is not deployed.</exception>
    internal static IntPtr Create(IntPtr state, IntPtr allocate, IntPtr allocateOnStream, IntPtr free, IntPtr info)
    {
        var create = (delegate* unmanaged<IntPtr, IntPtr, IntPtr, IntPtr, IntPtr, IntPtr>)_create.Value;
        var made = create(state, allocate, allocateOnStream, free, info);
        return made != IntPtr.Zero ? made
            : throw new OutOfMemoryException("There was no memory for Shorokoo's native allocator itself.");
    }

    /// <summary>The CUDA build of the library's operators (<c>native/shorokoo_adam_update.cu</c>),
    /// which a CUDA session registers in place of the CPU one. It ships in the GPU backend
    /// packages, Shorokoo.WinGPU and Shorokoo.LinuxGPU, beside ONNX Runtime's CUDA provider.</summary>
    internal const string CudaOperatorsLibraryName = "shorokoo_ort_cuda_ops";

    /// <summary>The CUDA operators library's file on this operating system.</summary>
    internal static string CudaOperatorsFileName
        => OperatingSystem.IsWindows() ? CudaOperatorsLibraryName + ".dll" : "lib" + CudaOperatorsLibraryName + ".so";

    private static readonly Lazy<string?> _locatedCudaOperators = new(
        () => LocateFile(CudaOperatorsFileName) is { } path && NativeLibrary.TryLoad(path, out _) ? path : null,
        LazyThreadSafetyMode.ExecutionAndPublication);

    /// <summary>The path a CUDA session registers the operators from, found where the library is,
    /// or null where that build is not deployed or this system cannot load it (one built against a
    /// newer C library than the system's, say): a build from source makes it only where it finds a
    /// CUDA toolkit to make it with, and a CUDA session without it runs each update as the chain of
    /// operators it is written as.</summary>
    internal static string? LocatedCudaOperators => _locatedCudaOperators.Value;

    /// <summary>The path the library is deployed at, beside this assembly or under
    /// <c>runtimes/&lt;rid&gt;/native/</c> there, or in a folder the host probes for natives, or
    /// null where it is in none of them.</summary>
    internal static string? Locate() => LocateFile(FileName);

    private static string? LocateFile(string fileName)
    {
        var location = typeof(NativeAllocator).Assembly.Location;
        List<string> directories = [];
        if (!string.IsNullOrEmpty(location) && Path.GetDirectoryName(location) is { Length: > 0 } dir) directories.Add(dir);
        directories.Add(AppContext.BaseDirectory);
        // The folders the host itself probes for natives, a deps.json's included.
        if (AppContext.GetData("NATIVE_DLL_SEARCH_DIRECTORIES") is string probed)
            directories.AddRange(probed.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries));
        var rid = (OperatingSystem.IsWindows() ? "win-" : "linux-")
            + RuntimeInformation.ProcessArchitecture.ToString().ToLowerInvariant();
        foreach (var directory in directories)
        {
            var flat = Path.Combine(directory, fileName);
            if (File.Exists(flat)) return flat;
            var underRuntimes = Path.Combine(directory, "runtimes", rid, "native", fileName);
            if (File.Exists(underRuntimes)) return underRuntimes;
        }
        return null;
    }

    private static IntPtr Bind()
    {
        IntPtr library;
        if (Locate() is { } path)
            library = NativeLibrary.Load(path);
        else if (!NativeLibrary.TryLoad(LibraryName, typeof(NativeAllocator).Assembly, null, out library))
            throw new DllNotFoundException(
                $"Shorokoo's native allocator ({FileName}) is not deployed: not beside "
                + $"{typeof(NativeAllocator).Assembly.GetName().Name}.dll, not under runtimes/<rid>/native "
                + "there, and not anywhere the host probes. It ships in the Shorokoo.OnnxRuntime package, "
                + "and a build from source makes it from native/ in that project.");
        return NativeLibrary.GetExport(library, CreateExport);
    }
}
