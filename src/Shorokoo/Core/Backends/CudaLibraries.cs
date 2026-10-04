using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;

namespace Shorokoo.Core.Backends;

/// <summary>
/// The cuDNN and cuBLAS every CUDA backend of a process runs on: one release of each per CUDA major,
/// pinned beside the lock of the CUDA Python environment — the releases its PyTorch carries — and
/// one copy of it, which every backend binds to.
///
/// <para>One release, because two do not mix. A library named without a path binds to the copy the
/// process already holds under that name, and cuDNN's sub-libraries import one another's internal
/// entry points by name, as cuBLAS imports cuBLASLt's, which another release does not all export.
/// One copy, because a process holding the same release twice holds every page of it twice.</para>
///
/// <para>The copy is the release's folder in a per-user cache, <c>%LOCALAPPDATA%\shorokoo\cuda\</c>
/// on Windows and <c>$XDG_CACHE_HOME/shorokoo/cuda/</c> (or <c>~/.cache/…</c>) elsewhere.
/// <see cref="Prepare"/> fills it the first time it is needed — from a copy that matches the pin
/// exactly, a provisioned Python environment's or an installed one, or else from the release's wheel
/// on PyPI — and loads it, by full path, before a backend loads anything by name. The ONNX Runtime
/// CUDA backend calls it before its provider loads; a provisioned Python environment's own copies are
/// hard links to the cache's files, so PyTorch, loading them by path, loads the very same files.</para>
/// </summary>
public static partial class CudaLibraries
{
    private static readonly object Gate = new();
    private static bool _prepared;

    /// <summary>
    /// Fills the shared cache with the pinned cuDNN and cuBLAS where it does not hold them yet, and
    /// loads them into this process; returns at once when they are loaded already. The ONNX Runtime
    /// CUDA backend calls it before its execution provider loads. Call it yourself at startup to fetch
    /// them then rather than on the first CUDA run, or before a backend of your own loads them.
    /// </summary>
    /// <exception cref="InvalidOperationException">A pinned library is in neither the cache nor a copy
    /// that matches it exactly, and its wheel could not be fetched; the message names the library,
    /// where it was looked for and the size of the download. Or the process already holds, under the
    /// name a pinned file is loaded by, a copy that is not that file — another release, which something
    /// loaded before this was called — to which the pinned libraries would bind; nothing is loaded, and
    /// the message names the copies held.</exception>
    /// <exception cref="InvalidDataException">The wheel fetched is not the one pinned.</exception>
    /// <exception cref="IOException">The cache folder could not be written: a full disk, say.</exception>
    /// <exception cref="TimeoutException">Another process has been filling the cache folder for longer
    /// than an hour.</exception>
    public static void Prepare()
    {
        if (CudaLibraryPins.Current is not { } pins) return;
        lock (Gate)
        {
            if (_prepared) return;
            var root = CudaLibraryCache.DefaultRoot;
            // Before anything is fetched or loaded. A pinned file loaded by its path beside another copy
            // of its name is a second library, and what imports that name binds to the copy loaded first:
            // where that copy lacks an entry point the load fails, and where it does not -- always on
            // Linux, whose loader binds by soname -- the backend runs on it, and says nothing.
            if (OtherReleasesHeld(pins, root, HeldUnder) is { } held)
                throw new InvalidOperationException(
                    $"This process already holds another release of the CUDA libraries every CUDA backend shares "
                    + $"({held}), loaded before the CUDA backend prepared the pinned "
                    + $"{string.Join(" and ", pins.Libraries)}. Those libraries load one another by name, so the "
                    + "pinned copies would bind to the ones held, and two releases do not mix. Call "
                    + $"{nameof(CudaLibraries)}.{nameof(Prepare)}() before anything else in the process loads "
                    + "cuDNN or cuBLAS: a CUDA session of your own, or another framework's CUDA backend.");
            // cuBLAS first: cuDNN loads cuBLASLt by name, and has to find this one.
            foreach (var pin in pins.Libraries.OrderBy(pin => pin.Name == "cudnn"))
            {
                var directory = Path.Combine(root, pin.CacheKey);
                if (!CudaLibraryCache.IsComplete(directory, pin))
                    directory = CudaLibraryCache.Provision(pin, root,
                        CudaLibraryCache.InstalledCandidates(pin, Environment.GetEnvironmentVariable, OperatingSystem.IsWindows()),
                        CudaLibraryCache.DefaultTimeout);
                // In the pinned order, which puts what a file imports by name ahead of it. Loaded for
                // the life of the process, like every library a backend binds.
                foreach (var file in pin.Files)
                    NativeLibrary.Load(Path.Combine(directory, file.FileName));
            }
            _prepared = true;
        }
    }

    /// <summary>
    /// The copies this process holds, under the names the files of <paramref name="pins"/> are loaded
    /// by, that are not those files — <paramref name="heldUnder"/> naming the copy a name binds to, or
    /// null where none is loaded — each named with its version and folder; or null where it holds none.
    /// A copy is the pinned file where it is the cache's under <paramref name="root"/>, by any name —
    /// a provisioned environment's links to it among them — or has the size and SHA-256 the pin gives
    /// it.
    /// </summary>
    internal static string? OtherReleasesHeld(CudaLibraryPins pins, string root, Func<string, string?> heldUnder)
    {
        var held = new List<string>();
        foreach (var pin in pins.Libraries)
            foreach (var file in pin.Files)
                if (heldUnder(file.FileName) is { Length: > 0 } path && !IsPinned(path, file, Path.Combine(root, pin.CacheKey, file.FileName)))
                    held.Add(Describe(path));
        return held.Count == 0 ? null : string.Join("; ", held);
    }

    private static bool IsPinned(string path, CudaLibraryFile file, string cached)
    {
        if (CudaLibraryCache.SameFile(path, cached)) return true;
        try
        {
            return CudaLibraryCache.Matches(path, file);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return false;
        }
    }

    /// <summary>The file a library <paramref name="name"/> names binds to in this process — the copy
    /// of that name its loader holds, the first loaded — or null where it holds none.</summary>
    internal static string? HeldUnder(string name)
        => OperatingSystem.IsWindows() ? HeldUnderOnWindows(name)
            : OperatingSystem.IsLinux() ? HeldUnderOnLinux(name)
            : null;

    private static unsafe string? HeldUnderOnWindows(string name)
    {
        var module = GetModuleHandle(name);
        if (module == IntPtr.Zero) return null;
        var path = stackalloc char[32768];
        var length = GetModuleFileName(module, path, 32768);
        return length == 0 ? null : new string(path, 0, (int)length);
    }

    private static unsafe string? HeldUnderOnLinux(string name)
    {
        var (open, info, close) = Loader.Value;
        var utf8 = Marshal.StringToCoTaskMemUTF8(name);
        try
        {
            // RTLD_NOLOAD: the object a load of this name would bind to, where one is loaded, and
            // nothing loaded where none is.
            var handle = ((delegate* unmanaged<IntPtr, int, IntPtr>)open)(utf8, RtldLazy | RtldNoLoad);
            if (handle == IntPtr.Zero) return null;
            try
            {
                IntPtr map;
                if (((delegate* unmanaged<IntPtr, int, IntPtr*, int>)info)(handle, RtldDiLinkMap, &map) != 0) return null;
                // A link_map's second field is the file's name, as it was found.
                return Marshal.PtrToStringUTF8(Marshal.ReadIntPtr(map, IntPtr.Size)) is { Length: > 0 } path ? path : null;
            }
            finally
            {
                ((delegate* unmanaged<IntPtr, int>)close)(handle);
            }
        }
        finally
        {
            Marshal.FreeCoTaskMem(utf8);
        }
    }

    private const int RtldLazy = 0x1;
    private const int RtldNoLoad = 0x4;
    private const int RtldDiLinkMap = 2;

    /// <summary><c>dlopen</c>, <c>dlinfo</c> and <c>dlclose</c>: in the C library, or in
    /// <c>libdl</c> where the C library is older than glibc 2.34.</summary>
    private static readonly Lazy<(IntPtr Open, IntPtr Info, IntPtr Close)> Loader = new(() =>
    {
        foreach (var library in (string[])["libc.so.6", "libdl.so.2"])
            if (NativeLibrary.TryLoad(library, out var handle)
                && NativeLibrary.TryGetExport(handle, "dlopen", out var open)
                && NativeLibrary.TryGetExport(handle, "dlinfo", out var info)
                && NativeLibrary.TryGetExport(handle, "dlclose", out var close))
                return (open, info, close);
        throw new EntryPointNotFoundException("dlopen, dlinfo and dlclose are in neither libc.so.6 nor libdl.so.2.");
    });

    [LibraryImport("kernel32.dll", EntryPoint = "GetModuleHandleW", StringMarshalling = StringMarshalling.Utf16)]
    private static partial IntPtr GetModuleHandle(string name);

    [LibraryImport("kernel32.dll", EntryPoint = "GetModuleFileNameW")]
    private static unsafe partial uint GetModuleFileName(IntPtr module, char* path, uint size);

    /// <summary>
    /// The pinned libraries <paramref name="directory"/> ships that this process already holds another
    /// release of, each named with that copy's folder and version, or null when it holds none: what
    /// makes a framework that loads that folder's libraries by path fail to load them.
    /// </summary>
    internal static string? Conflict(string directory)
    {
        if (!OperatingSystem.IsWindows() || CudaLibraryPins.Current is not { } pins) return null;
        return Conflict(directory, PinnedFileNames(pins), LoadedModules());
    }

    private static IEnumerable<string> PinnedFileNames(CudaLibraryPins pins)
        => pins.Libraries.SelectMany(pin => pin.Files).Select(file => file.FileName);

    /// <summary>The file of every module this process has loaded.</summary>
    private static List<string> LoadedModules()
    {
        using var process = Process.GetCurrentProcess();
        var loaded = new List<string>();
        foreach (ProcessModule module in process.Modules)
            using (module)
                loaded.Add(module.FileName);
        return loaded;
    }

    /// <summary>The same, for the files <paramref name="loaded"/> lists as the ones the process
    /// holds and the file names <paramref name="pinned"/> lists as the pinned ones.</summary>
    internal static string? Conflict(string directory, IEnumerable<string> pinned, IEnumerable<string> loaded)
    {
        if (!Directory.Exists(directory)) return null;
        var names = pinned.ToHashSet(StringComparer.OrdinalIgnoreCase);
        var held = loaded
            .Where(path => names.Contains(Path.GetFileName(path)))
            .Where(path => File.Exists(Path.Combine(directory, Path.GetFileName(path))))
            // A copy of the very same file, under another path, binds as well as the folder's own.
            .Where(path => !SameContents(path, Path.Combine(directory, Path.GetFileName(path))))
            .Select(Describe)
            .ToList();
        return held.Count == 0 ? null : string.Join("; ", held);
    }

    /// <summary>A library file as a message names it: its name, its version where it states one, and
    /// its folder.</summary>
    private static string Describe(string path)
    {
        string? version;
        try
        {
            version = FileVersionInfo.GetVersionInfo(path).FileVersion;
        }
        catch (FileNotFoundException)
        {
            version = null;
        }
        return $"{Path.GetFileName(path)}{(version is { Length: > 0 } ? " " + version : "")} from '{Path.GetDirectoryName(path)}'";
    }

    private static bool SameContents(string a, string b)
    {
        if (CudaLibraryCache.SameFile(a, b)) return true;
        using var first = File.OpenRead(a);
        using var second = File.OpenRead(b);
        return first.Length == second.Length && CudaLibraryCache.Sha256Of(first) == CudaLibraryCache.Sha256Of(second);
    }
}
