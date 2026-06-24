using System.Reflection;
using System.Runtime.InteropServices;

namespace Shorokoo.Core.Inference.Abstractions;

/// <summary>
/// Holds the single <see cref="IShorokooInferenceSessionFactory"/> used for all
/// inference in the process.
///
/// <para>
/// The core Shorokoo assembly does not reference ONNX Runtime; the concrete
/// factory lives in a platform package (Shorokoo.WinCPU, Shorokoo.WinGPU,
/// Shorokoo.LinuxCPU, or Shorokoo.LinuxGPU) that you add as a dependency. The
/// supported way to choose one is to set it explicitly at startup:
/// </para>
/// <code>
/// InferenceBackend.Factory = new LinuxCpuInferenceFactory();
/// </code>
/// <para>
/// If you never set one, the first inference call auto-discovers a backend by
/// looking <b>only</b> in the folder next to this assembly for the known
/// Shorokoo.{Platform} DLLs. When both a CPU and a GPU backend for the current
/// OS are deployed there, the GPU one is used if a CUDA 12.x runtime is present,
/// otherwise the CPU one. Only one backend is ever live per process; loading a
/// second native (e.g. comparing CPU vs CUDA) requires separate processes.
/// </para>
/// </summary>
public static class InferenceBackend
{
    private static IShorokooInferenceSessionFactory? _factory;
    private static readonly object _gate = new();

    /// <summary>
    /// The backend used for all inference. Assign once at startup; if left unset
    /// it is auto-discovered from the deployment folder on first access.
    /// </summary>
    public static IShorokooInferenceSessionFactory Factory
    {
        get
        {
            if (_factory is not null) return _factory;
            lock (_gate) { return _factory ??= Discover(); }
        }
        set
        {
            ArgumentNullException.ThrowIfNull(value);
            lock (_gate) { _factory = value; }
        }
    }

    // The backend DLLs Shorokoo ships: the OS each targets and whether it drives
    // the CUDA execution provider.
    private static readonly (string Assembly, OSPlatform Os, bool Gpu)[] KnownBackends =
    [
        ("Shorokoo.WinCPU",   OSPlatform.Windows, false),
        ("Shorokoo.WinGPU",   OSPlatform.Windows, true),
        ("Shorokoo.LinuxCPU", OSPlatform.Linux,   false),
        ("Shorokoo.LinuxGPU", OSPlatform.Linux,   true),
    ];

    private static IShorokooInferenceSessionFactory Discover()
    {
        // A backend already loaded in the process wins -- it avoids pulling a
        // second native in alongside one the consumer has already bound.
        var preLoaded = TryFindAlreadyLoadedFactory();
        if (preLoaded is not null) return preLoaded;

        var dir = ProbeDirectory();
        var osCandidates = KnownBackends
            .Where(b => RuntimeInformation.IsOSPlatform(b.Os)
                        && File.Exists(Path.Combine(dir, b.Assembly + ".dll")))
            .Select(b => (b.Assembly, b.Gpu))
            .ToList();

        var chosen = SelectBackend(osCandidates, IsCudaAvailable())
            ?? throw new InvalidOperationException(
                $"No Shorokoo inference backend is set and none was found in '{dir}'. " +
                "Set one at startup -- e.g. InferenceBackend.Factory = new " +
                "LinuxCpuInferenceFactory(); (or the factory from whichever " +
                "Shorokoo.{WinCPU,WinGPU,LinuxCPU,LinuxGPU} package you reference) -- " +
                "or add such a package as a dependency.");

        var path = Path.Combine(dir, chosen.Assembly + ".dll");
        return InstantiateFactory(Assembly.LoadFrom(path))
            ?? throw new InvalidOperationException(
                $"'{chosen.Assembly}' was found at '{path}' but exposes no concrete " +
                $"{nameof(IShorokooInferenceSessionFactory)}.");
    }

    /// <summary>
    /// Chooses one backend from those deployed for the current OS. With several to
    /// pick from, the GPU backend is used only when a CUDA runtime is present,
    /// otherwise the CPU one; with a single candidate it is taken as-is. Returns
    /// null when nothing is deployed. Pure (no I/O) so the policy is unit-testable.
    /// </summary>
    internal static (string Assembly, bool Gpu)? SelectBackend(
        IReadOnlyList<(string Assembly, bool Gpu)> osCandidates, bool cudaAvailable)
    {
        if (osCandidates.Count == 0) return null;
        return osCandidates.FirstOrDefault(c => c.Gpu == cudaAvailable, osCandidates[0]);
    }

    private static string ProbeDirectory()
    {
        var location = typeof(InferenceBackend).Assembly.Location;
        if (!string.IsNullOrEmpty(location)
            && Path.GetDirectoryName(location) is { Length: > 0 } dir)
            return dir;
        return AppContext.BaseDirectory;
    }

    private static IShorokooInferenceSessionFactory? TryFindAlreadyLoadedFactory()
    {
        var known = KnownBackends.Select(b => b.Assembly).ToHashSet(StringComparer.OrdinalIgnoreCase);
        foreach (var asm in AppDomain.CurrentDomain.GetAssemblies())
        {
            if (!known.Contains(asm.GetName().Name ?? "")) continue;
            if (InstantiateFactory(asm) is { } factory) return factory;
        }
        return null;
    }

    private static IShorokooInferenceSessionFactory? InstantiateFactory(Assembly asm)
    {
        Type? type;
        try
        {
            type = asm.GetExportedTypes().FirstOrDefault(t =>
                typeof(IShorokooInferenceSessionFactory).IsAssignableFrom(t)
                && !t.IsAbstract
                && t.GetConstructor(Type.EmptyTypes) is not null);
        }
        catch
        {
            return null;
        }
        return type is null ? null : (IShorokooInferenceSessionFactory)Activator.CreateInstance(type)!;
    }

    private static bool IsCudaAvailable()
    {
        // CUDA Toolkit 12.x ships its runtime under an OS-specific name --
        // cudart64_12.dll on Windows, libcudart.so.12 on Linux; presence implies
        // the runtime libs ORT's CUDA EP DT_NEEDs are installed and resolvable.
        var cudaRuntime = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
            ? "cudart64_12.dll"
            : "libcudart.so.12";
        if (NativeLibrary.TryLoad(cudaRuntime, out var h))
        {
            NativeLibrary.Free(h);
            return true;
        }
        return false;
    }
}
