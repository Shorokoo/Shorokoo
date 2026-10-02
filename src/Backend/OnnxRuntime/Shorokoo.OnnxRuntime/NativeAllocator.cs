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

    /// <summary>
    /// A native <c>OrtAllocator</c> over the managed allocator <paramref name="state"/> names,
    /// whose requests go to the entry points <paramref name="allocate"/>
    /// (<c>(state, size, reason, capacity)</c>, answering a block or null with the reason written
    /// down) and <paramref name="free"/> (<c>(state, block)</c>), describing itself with the native
    /// memory info <paramref name="info"/>. It lives for the life of the process.
    /// </summary>
    /// <exception cref="DllNotFoundException">The library is not deployed.</exception>
    internal static IntPtr Create(IntPtr state, IntPtr allocate, IntPtr free, IntPtr info)
    {
        var create = (delegate* unmanaged<IntPtr, IntPtr, IntPtr, IntPtr, IntPtr>)_create.Value;
        var made = create(state, allocate, free, info);
        return made != IntPtr.Zero ? made
            : throw new OutOfMemoryException("There was no memory for Shorokoo's native allocator itself.");
    }

    /// <summary>The path the library is deployed at, beside this assembly or under
    /// <c>runtimes/&lt;rid&gt;/native/</c> there, or null where it is in neither.</summary>
    internal static string? Locate()
    {
        var location = typeof(NativeAllocator).Assembly.Location;
        string[] directories = !string.IsNullOrEmpty(location) && Path.GetDirectoryName(location) is { Length: > 0 } dir
            ? [dir, AppContext.BaseDirectory]
            : [AppContext.BaseDirectory];
        var rid = (OperatingSystem.IsWindows() ? "win-" : "linux-")
            + RuntimeInformation.ProcessArchitecture.ToString().ToLowerInvariant();
        foreach (var directory in directories)
        {
            var flat = Path.Combine(directory, FileName);
            if (File.Exists(flat)) return flat;
            var underRuntimes = Path.Combine(directory, "runtimes", rid, "native", FileName);
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
