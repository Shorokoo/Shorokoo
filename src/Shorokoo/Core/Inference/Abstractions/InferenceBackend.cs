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
/// Shorokoo.LinuxCPU, or Shorokoo.LinuxGPU) that you add as a dependency.
/// Naming one in code is optional — referencing the package is normally enough —
/// but you can set it explicitly at startup to override the discovered choice, or
/// to fail at startup rather than on the first inference call:
/// </para>
/// <code>
/// InferenceBackend.Factory = new LinuxCpuInferenceFactory();
/// </code>
/// <para>
/// If you never set one, the first inference call auto-discovers a backend in two
/// steps: a Shorokoo.{Platform} assembly already loaded in the process wins, and
/// only failing that is the folder next to this assembly probed for the known
/// Shorokoo.{Platform} DLLs. Both steps consider only backends targeting the running
/// OS, and both <b>refuse</b> two of them rather than choosing between them — see
/// <see cref="SelectBackend"/>. Only one backend is ever live per process; loading a
/// second native (e.g. comparing CPU vs CUDA) requires separate processes.
/// </para>
/// <para>
/// Which backend that turned out to be is answerable: <see cref="Current"/> peeks
/// without resolving one, <see cref="Describe"/> names the live one, and
/// <see cref="RequireDevice"/> asserts it is the device this program meant to run on.
/// </para>
/// </summary>
public static class InferenceBackend
{
    private static IShorokooInferenceSessionFactory? _factory;
    private static readonly object _gate = new();

    /// <summary>
    /// The backend used for all inference. Assigning one is optional; if left unset
    /// it is auto-discovered on first access — an already-loaded backend assembly
    /// first, otherwise the deployment folder.
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

    /// <summary>
    /// The backend if one is live, null if none has been assigned or discovered yet.
    /// Unlike <see cref="Factory"/>, reading this does not resolve one — so a startup
    /// path can tell "nothing chosen yet" from "already bound" without deciding the
    /// question by asking it.
    ///
    /// <para>It reports, it does not reserve: two threads can both read null and both
    /// assign, and the last assignment wins. Decide the backend on one startup path
    /// rather than racing to fill in a null.</para>
    /// </summary>
    public static IShorokooInferenceSessionFactory? Current => _factory;

    /// <summary>
    /// Names the live backend and the device it runs on, resolving one the way
    /// <see cref="Factory"/> would if none is live yet. This is what a run's log should
    /// record: nothing at a call site says which device the work went to.
    /// </summary>
    public static BackendDescription Describe() => Factory.Description;

    /// <summary>
    /// Throws unless the live backend runs on <paramref name="device"/>, resolving one the
    /// way <see cref="Factory"/> would if none is live yet. A program whose correctness
    /// depends on its device — a check that must not contend with a training run on the
    /// card, say — states that here and fails at startup rather than discovering it from a
    /// throughput figure.
    /// </summary>
    /// <exception cref="InvalidOperationException">The live backend runs on another device.</exception>
    public static void RequireDevice(ComputeDevice device)
    {
        var live = Describe();
        if (live.Device != device)
            throw new InvalidOperationException(
                $"This program requires a {Label(device)} backend, but {live} is live. Reference the " +
                "Shorokoo.{WinCPU,WinGPU,LinuxCPU,LinuxGPU} package for the device you want, or " +
                "assign InferenceBackend.Factory before the first inference call.");
    }

    private static string Label(ComputeDevice device) => device switch
    {
        ComputeDevice.Cuda => "CUDA",
        ComputeDevice.Cpu => "CPU",
        _ => device.ToString(),
    };

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

        var chosen = SelectBackend(osCandidates, $"deployed in '{dir}'")
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
    /// Chooses the one backend among those deployed for the current OS. Nothing deployed
    /// returns null; a single candidate is taken as-is. Several are <b>refused</b>: only one
    /// backend can be live in a process, and the CPU and GPU packages deliver their native
    /// ONNX Runtime at the same path, so a deployment carrying both is already ambiguous and
    /// guessing at it silently is how work lands on a device its author did not intend. Pure
    /// (no I/O) so the policy is unit-testable; <paramref name="origin"/> is where the
    /// candidates were found, for the message.
    /// </summary>
    /// <exception cref="InvalidOperationException">More than one backend is deployed.</exception>
    internal static (string Assembly, bool Gpu)? SelectBackend(
        IReadOnlyList<(string Assembly, bool Gpu)> osCandidates, string origin)
    {
        if (osCandidates.Count == 0) return null;
        if (osCandidates.Count == 1) return osCandidates[0];
        throw Ambiguous(osCandidates, origin);
    }

    private static InvalidOperationException Ambiguous(
        IReadOnlyList<(string Assembly, bool Gpu)> candidates, string origin)
    {
        var names = string.Join(", ", candidates.Select(
            c => c.Assembly + (c.Gpu ? " (CUDA)" : " (CPU)")));
        return new InvalidOperationException(
            $"Several Shorokoo inference backends are {origin}: {names}. Only one can be live " +
            "in a process, and each package brings its own native ONNX Runtime, so a build " +
            "carrying both is ambiguous. Reference exactly one backend package -- keeping model " +
            "code in a library that references no backend, and one executable per device -- or " +
            "assign InferenceBackend.Factory before the first inference call to say which of " +
            "these you mean.");
    }

    /// <summary>
    /// The backends among <paramref name="loadedAssemblyNames"/> that could serve this process:
    /// known backend assemblies targeting the running OS. A Windows backend loaded into a Linux
    /// process is not one — it cannot run, and its native collides with nothing here, so it must
    /// not make the choice ambiguous. Pure, in <c>KnownBackends</c> order, so the policy is
    /// unit-testable and the refusal message is stable.
    /// </summary>
    internal static IReadOnlyList<(string Assembly, bool Gpu)> LoadedCandidates(
        IEnumerable<string> loadedAssemblyNames)
    {
        var loaded = loadedAssemblyNames.ToHashSet(StringComparer.OrdinalIgnoreCase);
        return KnownBackends
            .Where(b => RuntimeInformation.IsOSPlatform(b.Os) && loaded.Contains(b.Assembly))
            .Select(b => (b.Assembly, b.Gpu))
            .ToList();
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
        var assemblies = AppDomain.CurrentDomain.GetAssemblies();
        var usable = LoadedCandidates(assemblies.Select(asm => asm.GetName().Name ?? ""))
            .Select(candidate => (candidate, Factory: assemblies
                .Where(asm => string.Equals(
                    asm.GetName().Name, candidate.Assembly, StringComparison.OrdinalIgnoreCase))
                .Select(InstantiateFactory)
                .FirstOrDefault(factory => factory is not null)))
            .Where(found => found.Factory is not null)
            .ToList();

        // Only what is usable can be ambiguous: a backend assembly exposing no concrete factory
        // is no more a choice than one that is not loaded, and the folder probe still gets its
        // turn. Hence the instantiation before the refusal, not after it.
        if (usable.Count == 0) return null;
        if (usable.Count > 1)
            throw Ambiguous([.. usable.Select(found => found.candidate)], "already loaded in this process");
        return usable[0].Factory;
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
}
