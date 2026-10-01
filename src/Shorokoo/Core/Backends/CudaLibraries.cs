using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;

namespace Shorokoo.Core.Backends;

/// <summary>
/// The NVIDIA libraries the CUDA backends of one Windows process share: one copy of each, loaded
/// once, that every backend binds to whichever of them starts first.
///
/// <para>Windows resolves a library named without a path — an import, or a <c>LoadLibrary</c> of a
/// bare name — to the copy the process already holds under that name, whatever folder it came
/// from. cuDNN is a family of libraries that import one another by name, and two releases of it do
/// not mix: the internal entry points one release's <c>cudnn_cnn64_9.dll</c> imports from
/// <c>cudnn_graph64_9.dll</c> are not all exported by another's. cuBLAS is the same pair over
/// <c>cublasLt</c>. So two backends bringing two copies of the family cannot both run in one
/// process: the second to load binds its own files to the first one's.</para>
///
/// <para>The copy is the one the PyTorch CUDA backend's Python environment ships, wherever that
/// backend is deployed and its environment exists. PyTorch loads every library in its
/// <c>torch\lib</c> by path, so it can run on no other copy, and NVIDIA keeps cuDNN 9 and the
/// CUDA 13 libraries backward compatible across minor releases, so the ONNX Runtime backend runs on
/// it as well as on the system's. Shorokoo loads it before any of its backends loads a CUDA
/// library by name, so whichever backend starts first, both bind to it. Where no such environment
/// is in reach, each backend loads what the system's DLL search finds, as it does alone.</para>
///
/// <para>Shorokoo's own backends call <see cref="Prepare"/> themselves. A backend of your own that
/// loads CUDA libraries calls it first too. On other systems this does nothing.</para>
/// </summary>
public static class CudaLibraries
{
    // The families more than one backend loads by name, and what those load by name in turn: the
    // CUDA runtime Shorokoo calls itself; cuBLAS and cuBLASLt, which the ONNX Runtime CUDA provider
    // imports; cuDNN, which it loads; cuFFT, which it loads for its FFT operators; and the
    // compilers cuDNN, cuBLASLt and cuFFT load at run time. Each prefix is one family, which comes
    // from one folder or not at all.
    private static readonly string[] Families = ["cudart64_", "cublas", "cudnn", "cufft64_", "nvrtc", "nvJitLink_"];

    // The library a folder must hold to be one built for the CUDA major Shorokoo's CUDA backends
    // are: the runtime CudaRuntime binds.
    private const string Runtime = "cudart64_13.dll";

    // The Python-based CUDA backends Shorokoo ships, and where the folder their environment's
    // libraries live in is answered: the host they share, asked without starting Python or
    // provisioning anything.
    private static readonly string[] PythonCudaBackends = ["Shorokoo.PyTorch.Cuda", "Shorokoo.Jax.Cuda"];
    private const string PythonHost = "Shorokoo.PythonHost";
    private const string PythonHostLocator = "Shorokoo.PythonHost.PythonEnvironmentResolver";
    private const string PythonHostLocatorMethod = "ExistingCudaLibraryDirectory";

    private static readonly object Gate = new();
    private static string? _directory;
    private static MethodInfo? _locator;
    private static bool _locatorResolved;

    /// <summary>
    /// The folder the CUDA libraries this process shares were loaded from, or null while none
    /// has been: then each backend has loaded what the system's DLL search finds.
    /// </summary>
    public static string? Directory { get { lock (Gate) return _directory; } }

    /// <summary>
    /// Loads the CUDA libraries the process shares, where it shares any, ahead of a backend's own
    /// loads by name; returns at once when they are loaded already. Call it before loading any
    /// CUDA library by name.
    /// </summary>
    public static void Prepare()
    {
        if (!OperatingSystem.IsWindows()) return;
        lock (Gate)
        {
            if (_directory is not null) return;
            if (Located() is { } directory) Share(directory);
        }
    }

    /// <summary>
    /// Offers the CUDA libraries in <paramref name="directory"/> for the process to share, and loads
    /// from it each family of them the process holds no copy of yet. A family the process already
    /// holds from another folder is left as it is. A folder that is not built for CUDA 13 is
    /// declined. This is what a backend that loads its own libraries by path calls before it does.
    /// </summary>
    /// <exception cref="ArgumentException"><paramref name="directory"/> is blank.</exception>
    public static void Share(string directory)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(directory);
        if (!OperatingSystem.IsWindows()) return;
        var full = Path.GetFullPath(directory);
        if (!File.Exists(Path.Combine(full, Runtime))) return;
        lock (Gate)
        {
            var loaded = LoadedModules();
            var shared = false;
            foreach (var family in SharedLibraries(LibrariesIn(full)).GroupBy(Family))
            {
                // A family already held from elsewhere stays as it is: loading the rest of it from
                // here would bind those files to the other release's.
                if (family.Any(name => loaded.TryGetValue(name, out var paths)
                        && paths.Any(path => !SameFile(path, Path.Combine(full, name)))))
                    continue;
                foreach (var name in family.Where(name => !loaded.ContainsKey(name)))
                    NativeLibrary.Load(Path.Combine(full, name));
                shared = true;
            }
            if (shared) _directory ??= full;
        }
    }

    /// <summary>
    /// The CUDA libraries <paramref name="directory"/> ships that the process holds a different
    /// copy of — named with the version and folder of the copy it holds — or null when it holds
    /// none: what makes a backend that loads that folder's libraries by path fail to load them.
    /// </summary>
    /// <exception cref="ArgumentException"><paramref name="directory"/> is blank.</exception>
    public static string? Conflict(string directory)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(directory);
        if (!OperatingSystem.IsWindows() || !System.IO.Directory.Exists(directory)) return null;
        var full = Path.GetFullPath(directory);
        var loaded = LoadedModules();
        var foreign = SharedLibraries(LibrariesIn(full))
            .SelectMany(name => loaded.TryGetValue(name, out var paths)
                ? paths.Where(path => !SameFile(path, Path.Combine(full, name)))
                : Enumerable.Empty<string>())
            .Select(path => $"{Path.GetFileName(path)} {FileVersionInfo.GetVersionInfo(path).FileVersion} from '{Path.GetDirectoryName(path)}'")
            .ToList();
        return foreign.Count == 0 ? null : string.Join("; ", foreign);
    }

    /// <summary>
    /// The libraries among <paramref name="fileNames"/> the process shares, in the order they are
    /// loaded: family by family, in the order of <see cref="Families"/>, and by name within one.
    /// </summary>
    internal static IReadOnlyList<string> SharedLibraries(IEnumerable<string> fileNames)
        => [.. fileNames
            .Where(name => name.EndsWith(".dll", StringComparison.OrdinalIgnoreCase) && FamilyIndex(name) >= 0)
            .OrderBy(FamilyIndex)
            .ThenBy(name => name, StringComparer.OrdinalIgnoreCase)];

    private static IEnumerable<string> LibrariesIn(string directory)
        => System.IO.Directory.EnumerateFiles(directory, "*.dll").Select(path => Path.GetFileName(path));

    private static int FamilyIndex(string fileName)
        => Array.FindIndex(Families, prefix => fileName.StartsWith(prefix, StringComparison.OrdinalIgnoreCase));

    private static string Family(string fileName) => Families[FamilyIndex(fileName)];

    /// <summary>
    /// The method that answers where the Python environment a Python-based CUDA backend of this
    /// process would run in keeps its CUDA libraries, or null where no such backend is deployed
    /// beside this assembly. Resolved once: what is deployed does not change.
    /// </summary>
    internal static MethodInfo? Locator
    {
        get
        {
            lock (Gate)
            {
                if (_locatorResolved) return _locator;
                _locatorResolved = true;
                var deployed = DeployedDirectory();
                if (!PythonCudaBackends.Any(name => File.Exists(Path.Combine(deployed, name + ".dll"))))
                    return null;
                try
                {
                    _locator = Assembly.Load(new AssemblyName(PythonHost)).GetType(PythonHostLocator)
                        ?.GetMethod(PythonHostLocatorMethod, BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic, Type.EmptyTypes);
                }
                // A host that cannot be loaded offers nothing, which leaves every backend loading
                // its own libraries, exactly as where none is deployed.
                catch (Exception ex) when (ex is FileNotFoundException or FileLoadException or BadImageFormatException) { }
                return _locator;
            }
        }
    }

    private static string? Located()
    {
        try
        {
            return Locator?.Invoke(null, null) as string;
        }
        catch (TargetInvocationException)
        {
            return null;
        }
    }

    private static string DeployedDirectory()
    {
        var location = typeof(CudaLibraries).Assembly.Location;
        return !string.IsNullOrEmpty(location) && Path.GetDirectoryName(location) is { Length: > 0 } dir
            ? dir
            : AppContext.BaseDirectory;
    }

    private static Dictionary<string, List<string>> LoadedModules()
    {
        var loaded = new Dictionary<string, List<string>>(StringComparer.OrdinalIgnoreCase);
        using var process = Process.GetCurrentProcess();
        foreach (ProcessModule module in process.Modules)
        {
            using (module)
            {
                if (!loaded.TryGetValue(module.ModuleName, out var paths))
                    loaded[module.ModuleName] = paths = [];
                paths.Add(module.FileName);
            }
        }
        return loaded;
    }

    private static bool SameFile(string a, string b)
        => string.Equals(Path.GetFullPath(a), Path.GetFullPath(b), StringComparison.OrdinalIgnoreCase);
}
