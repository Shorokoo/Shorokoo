using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Loader;
using System.Threading;

namespace Shorokoo.Core.Backends;

/// <summary>
/// Holds the default <see cref="IShorokooBackend"/>: the one work runs on when no
/// <c>ComputeContext</c> names another, and the one a tensor is built for when it is fed without
/// a context naming one.
///
/// <para>
/// The core Shorokoo assembly does not reference ONNX Runtime; the concrete
/// backend lives in a platform package (Shorokoo.WinCPU, Shorokoo.WinGPU,
/// Shorokoo.LinuxCPU, or Shorokoo.LinuxGPU) that you add as a dependency.
/// Naming one in code is optional — referencing the package is normally enough —
/// but you can set it explicitly at startup to override the discovered choice, or
/// to fail at startup rather than on the first run:
/// </para>
/// <code>
/// DefaultBackend.Instance = new LinuxCpuBackend();
/// </code>
/// <para>
/// If you never set one, the first run auto-discovers a backend in two
/// steps: a Shorokoo.{Platform} assembly already loaded in the process wins, and
/// only failing that is the folder next to this assembly probed for the known
/// Shorokoo.{Platform} DLLs. Both steps consider only backends targeting the running
/// OS, and both <b>refuse</b> two of them rather than choosing between them — see
/// <see cref="SelectBackend"/>. Discovery picks <i>one</i> because a program that has
/// not said which it means has not said it; it is not a limit on how many can run.
/// </para>
/// <para>
/// Several backends <b>can</b> be live at once. A <c>ComputeContext</c> constructed with a
/// backend compiles and runs on that one, so two contexts on two backends share a process
/// without sharing a device. Tensors stay out of it: a <c>TensorData</c> holds managed bytes
/// and names no backend until it is fed to one, and a session builds what it is fed on its own
/// runtime (<see cref="BackendTransfer"/>). Where the second backend needs its own native ONNX
/// Runtime rather than another execution provider on the same one,
/// <see cref="IsolatedBackend"/> loads it.
/// </para>
/// <para>
/// Which backend is live is answerable: <see cref="Current"/> peeks
/// without resolving one, <see cref="Describe"/> names the live one, and
/// <see cref="RequireDevice"/> asserts it is the device this program meant to run on.
/// </para>
/// </summary>
public static class DefaultBackend
{
    private static volatile IShorokooBackend? _instance;
    private static readonly object _gate = new();

    /// <summary>
    /// The default backend: the one a <c>ComputeContext</c> that names no backend of its own
    /// compiles and runs on. Assigning one
    /// is optional; if left unset it is auto-discovered on first access — an already-loaded
    /// backend assembly first, otherwise the deployment folder.
    /// </summary>
    public static IShorokooBackend Instance
    {
        get
        {
            if (_instanceReads.Value is { } counter) Interlocked.Increment(ref counter.Value);
            if (_instance is not null) return _instance;
            lock (_gate) { return _instance ??= Discover(); }
        }
        set
        {
            ArgumentNullException.ThrowIfNull(value);
            // Assigning here is a choice, not a load, so it settles the remembered slot outright
            // rather than going through Remember's first-CPU-wins rule -- which would otherwise
            // refuse the assignment the ambiguity error tells the caller to make, leaving the
            // default context on a backend the program has just said it did not want.
            lock (_gate) { _instance = value; _remembered = value; }
        }
    }

    /// <summary>
    /// The backend if one is live, null if none has been assigned or discovered yet.
    /// Unlike <see cref="Instance"/>, reading this does not resolve one — so a startup
    /// path can tell "nothing chosen yet" from "already bound" without deciding the
    /// question by asking it.
    ///
    /// <para>It reports, it does not reserve: two threads can both read null and both
    /// assign, and the last assignment wins. Decide the backend on one startup path
    /// rather than racing to fill in a null.</para>
    /// </summary>
    public static IShorokooBackend? Current => _instance;

    // Counts reads of Instance made inside a CountInstanceReads call, and nothing else -- the
    // same seam ComputeContext.Default carries, and for the same reason. The two are distinct paths to a
    // backend: a graph pass can reach one without ever touching a compute context (every
    // OnnxUtils.CreateTensorValue does), so counting only the context's reads left half the
    // question unasked.
    private static readonly AsyncLocal<System.Runtime.CompilerServices.StrongBox<int>?> _instanceReads = new();

    /// <summary>
    /// Runs <paramref name="work"/> and returns how many times it read <see cref="Instance"/>.
    /// The seam for holding the graph-building path to "requires no backend"; it counts
    /// asks rather than backends resolved, because every test host has one deployed and answers a
    /// wrongly eager read in silence.
    /// </summary>
    internal static int CountInstanceReads(Action work)
    {
        var outer = _instanceReads.Value;
        var counter = new System.Runtime.CompilerServices.StrongBox<int>(0);
        _instanceReads.Value = counter;
        try { work(); }
        finally { _instanceReads.Value = outer; }
        return counter.Value;
    }

    private static volatile IShorokooBackend? _remembered;

    /// <summary>
    /// The backend <see cref="Shorokoo.Runtime.ComputeContext.Default"/> is built on: the first
    /// one loaded, except that a CPU backend displaces a GPU one and is never displaced itself.
    ///
    /// <para>The asymmetry is deliberate. A program that has loaded both is running them side by
    /// side on purpose and will name the one it means for each context; what it should not get for
    /// the <i>unnamed</i> default is the card, where a stray convenience call quietly allocates
    /// device memory. The host is the safe default and the one every machine has.</para>
    /// </summary>
    public static IShorokooBackend? Remembered => _remembered;

    /// <summary>
    /// Records <paramref name="backend"/> as a loaded backend. A CPU backend always wins; a GPU
    /// backend is kept only while no CPU one has been seen. Called for every backend the process
    /// resolved for itself.
    ///
    /// <para>Not for one the program named. A backend it asked for by name is an answer to that
    /// question and to no other, so <see cref="Instance"/>'s setter records its choice directly and
    /// <see cref="IsolatedBackend"/> records nothing at all.</para>
    /// </summary>
    public static void Remember(IShorokooBackend backend)
    {
        ArgumentNullException.ThrowIfNull(backend);
        lock (_gate)
        {
            var incomingIsCpu = backend.Description.Device == ComputeDevice.Cpu;
            if (_remembered is null || incomingIsCpu)
            {
                if (_remembered is null
                    || incomingIsCpu && _remembered.Description.Device != ComputeDevice.Cpu)
                    _remembered = backend;
            }
        }
    }

    /// <summary>Forgets the remembered backend. Test hook: the rule is about what a process loaded,
    /// and a test process loads several.</summary>
    internal static void ForgetRemembered() { lock (_gate) _remembered = null; }

    /// <summary>
    /// Names the live backend and the device it runs on, resolving one the way
    /// <see cref="Instance"/> would if none is live yet. This is what a run's log should
    /// record: nothing at a call site says which device the work went to.
    /// </summary>
    public static BackendDescription Describe() => Instance.Description;

    /// <summary>
    /// Throws unless the live backend runs on <paramref name="device"/>, resolving one the
    /// way <see cref="Instance"/> would if none is live yet. A program whose correctness
    /// depends on its device — a check that must not contend with a training run on the
    /// card, say — states that here and fails at startup rather than discovering it from a
    /// throughput figure.
    /// </summary>
    /// <exception cref="InvalidOperationException">The live backend runs on another device.</exception>
    public static void RequireDevice(ComputeDevice device)
    {
        if (!Enum.IsDefined(device))
            throw new ArgumentOutOfRangeException(nameof(device), device, "Not a ComputeDevice.");
        if (DeviceRefusal(Describe(), device) is { } refusal)
            throw new InvalidOperationException(refusal);
    }

    /// <summary>
    /// Why <paramref name="live"/> does not satisfy a program requiring <paramref name="required"/>,
    /// or null when it does. Pure, so every pairing is testable without a process whose backend is
    /// the one under test.
    /// </summary>
    internal static string? DeviceRefusal(BackendDescription live, ComputeDevice required)
    {
        if (live.Device == required) return null;
        var remedy = required == ComputeDevice.Other
            // None of the shipped packages reports Other, so naming them here would be a remedy
            // the reader cannot follow.
            ? "No shipped backend runs on another provider, so assign DefaultBackend.Instance with "
              + "your own before the first run."
            : "Reference the Shorokoo.{WinCPU,WinGPU,LinuxCPU,LinuxGPU} package for the device you "
              + "want, or assign DefaultBackend.Instance before the first run.";
        return $"This program requires {Requirement(required)}, but {live} is live. {remedy}";
    }

    private static string Requirement(ComputeDevice device) => device switch
    {
        ComputeDevice.Cuda => "a CUDA backend",
        ComputeDevice.Cpu => "a CPU backend",
        _ => "a backend on some other execution provider",
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

    private static IShorokooBackend Discover()
    {
        // A backend already loaded in the process wins -- it avoids pulling a
        // second native in alongside one the consumer has already bound.
        var preLoaded = TryFindAlreadyLoadedBackend();
        if (preLoaded is not null) return Remembering(preLoaded)!;

        var dir = ProbeDirectory();
        var osCandidates = KnownBackends
            .Where(b => RuntimeInformation.IsOSPlatform(b.Os)
                        && File.Exists(Path.Combine(dir, b.Assembly + ".dll")))
            .Select(b => (b.Assembly, b.Gpu))
            .ToList();

        var chosen = SelectBackend(osCandidates, $"deployed in '{dir}'")
            ?? throw new InvalidOperationException(
                $"No Shorokoo backend is set and none was found in '{dir}'. " +
                "Set one at startup -- e.g. DefaultBackend.Instance = new " +
                "LinuxCpuBackend(); (or the backend from whichever " +
                "Shorokoo.{WinCPU,WinGPU,LinuxCPU,LinuxGPU} package you reference) -- " +
                "or add such a package as a dependency.");

        var path = Path.Combine(dir, chosen.Assembly + ".dll");
        return Remembering(InstantiateBackend(Assembly.LoadFrom(path)))
            ?? throw new InvalidOperationException(
                $"'{chosen.Assembly}' was found at '{path}' but exposes no concrete " +
                $"{nameof(IShorokooBackend)}.");
    }

    /// <summary>
    /// Chooses the one backend among those deployed for the current OS. Nothing deployed
    /// returns null; a single candidate is taken as-is. Several are <b>refused</b>: this is the
    /// path for a program that named no backend, and guessing silently between two is how work
    /// lands on a device its author did not intend. The refusal is about the silence, not about
    /// a limit -- a program that names its backends runs as many as it likes. Pure
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
            $"Several Shorokoo backends are {origin}: {names}. Discovery picks the " +
            "backend for a program that named none, and this deployment gives it no way to " +
            "choose. Say which you mean: assign DefaultBackend.Instance before the first " +
            "run to make one of them the default. To run several at once, give each " +
            "ComputeContext its own backend -- new ComputeContext(new LinuxGpuBackend()) " +
            "-- and where they need separate native ONNX Runtimes, load them with " +
            "IsolatedBackend.Load.");
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

    /// <summary>
    /// The assemblies of <paramref name="assemblies"/> that discovery may choose between: those in
    /// the default load context.
    ///
    /// <para>A backend loaded into isolation (<see cref="IsolatedBackend"/>) lives in a context of
    /// its own, and it got there by being named — that is the only way one loads. So it is not an
    /// answer to "which backend did this program mean", and counting it would mean that naming a
    /// second backend made the first one ambiguous.</para>
    ///
    /// <para>This covers the already-loaded step only. The folder probe that runs when that step
    /// finds nothing reads files rather than assemblies, and a file has no load context to filter
    /// on — so an isolated backend whose assembly sits in the probe directory is still a candidate
    /// there. Deploying it in a folder of its own, which is what
    /// <see cref="IsolatedBackendSpec.ProbeDirectory"/> is for and what the deployment guidance
    /// requires, is what keeps it out of that count.</para>
    /// </summary>
    internal static Assembly[] DiscoverableAssemblies(IEnumerable<Assembly> assemblies)
        => [.. assemblies.Where(
            asm => AssemblyLoadContext.GetLoadContext(asm) == AssemblyLoadContext.Default)];

    /// <summary>Records a backend as it is produced, and hands it straight back.</summary>
    private static IShorokooBackend? Remembering(IShorokooBackend? backend)
    {
        if (backend is not null) Remember(backend);
        return backend;
    }

    private static string ProbeDirectory()
    {
        var location = typeof(DefaultBackend).Assembly.Location;
        if (!string.IsNullOrEmpty(location)
            && Path.GetDirectoryName(location) is { Length: > 0 } dir)
            return dir;
        return AppContext.BaseDirectory;
    }

    private static IShorokooBackend? TryFindAlreadyLoadedBackend()
    {
        var assemblies = DiscoverableAssemblies(AppDomain.CurrentDomain.GetAssemblies());
        var usable = LoadedCandidates(assemblies.Select(asm => asm.GetName().Name ?? ""))
            .Select(candidate => (candidate, Backend: assemblies
                .Where(asm => string.Equals(
                    asm.GetName().Name, candidate.Assembly, StringComparison.OrdinalIgnoreCase))
                .Select(InstantiateBackend)
                .FirstOrDefault(backend => backend is not null)))
            .Where(found => found.Backend is not null)
            .ToList();

        // Only what is usable can be ambiguous: a backend assembly exposing no concrete backend
        // is no more a choice than one that is not loaded, and the folder probe still gets its
        // turn. Hence the instantiation before the refusal, not after it.
        if (usable.Count == 0) return null;
        if (usable.Count > 1)
            throw Ambiguous([.. usable.Select(found => found.candidate)], "already loaded in this process");
        return usable[0].Backend;
    }

    private static IShorokooBackend? InstantiateBackend(Assembly asm)
    {
        try
        {
            var type = asm.GetExportedTypes().FirstOrDefault(t =>
                typeof(IShorokooBackend).IsAssignableFrom(t)
                && !t.IsAbstract
                && t.GetConstructor(Type.EmptyTypes) is not null);
            return type is null ? null : (IShorokooBackend)Activator.CreateInstance(type)!;
        }
        // Constructing is as fallible as reflecting, and a candidate that throws on construction
        // is simply not a backend this process can use -- it must not abort the search, the more
        // so now that every candidate is constructed rather than only the first.
        catch
        {
            return null;
        }
    }
}
