using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Loader;

namespace Shorokoo.Core.Inference.Abstractions;

/// <summary>
/// Holds the default <see cref="IShorokooInferenceSessionFactory"/>: the backend every tensor
/// is built on, and the one inference runs on when no <c>ComputeContext</c> names another.
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
/// <see cref="SelectBackend"/>. Discovery picks <i>one</i> because a program that has
/// not said which it means has not said it; it is not a limit on how many can run.
/// </para>
/// <para>
/// Several backends <b>can</b> be live at once. A <c>ComputeContext</c> constructed with a
/// factory compiles and runs on that one, so two contexts on two backends share a process
/// without sharing a device. Tensors stay out of it: <see cref="IShorokooTensorValue"/> is
/// built by this default backend wherever it is built, and a session on another backend
/// converts what it is fed (<see cref="BackendTransfer"/>). Where the second backend needs its
/// own native ONNX Runtime rather than another execution provider on the same one,
/// <see cref="IsolatedBackend"/> loads it.
/// </para>
/// <para>
/// Which backend is live is answerable: <see cref="Current"/> peeks
/// without resolving one, <see cref="Describe"/> names the live one, and
/// <see cref="RequireDevice"/> asserts it is the device this program meant to run on.
/// </para>
/// </summary>
public static class InferenceBackend
{
    private static volatile IShorokooInferenceSessionFactory? _factory;
    private static readonly object _gate = new();

    /// <summary>
    /// The default backend: the one every tensor the framework builds is built by, and the one a
    /// <c>ComputeContext</c> that names no backend of its own compiles and runs on. Assigning one
    /// is optional; if left unset it is auto-discovered on first access — an already-loaded
    /// backend assembly first, otherwise the deployment folder.
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
            // Assigning here is a choice, not a load, so it settles the remembered slot outright
            // rather than going through Remember's first-CPU-wins rule -- which would otherwise
            // refuse the assignment the ambiguity error tells the caller to make, leaving the
            // default context on a backend the program has just said it did not want.
            lock (_gate) { _factory = value; _remembered = value; }
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

    private static volatile IShorokooInferenceSessionFactory? _remembered;

    /// <summary>
    /// The backend <see cref="Shorokoo.Runtime.ComputeContext.Default"/> is built on: the first
    /// one loaded, except that a CPU backend displaces a GPU one and is never displaced itself.
    ///
    /// <para>The asymmetry is deliberate. A program that has loaded both is running them side by
    /// side on purpose and will name the one it means for each context; what it should not get for
    /// the <i>unnamed</i> default is the card, where a stray convenience call quietly allocates
    /// device memory. The host is the safe default and the one every machine has.</para>
    /// </summary>
    public static IShorokooInferenceSessionFactory? Remembered => _remembered;

    /// <summary>
    /// Records <paramref name="factory"/> as a loaded backend. A CPU backend always wins; a GPU
    /// backend is kept only while no CPU one has been seen. Called for every backend the process
    /// resolved for itself.
    ///
    /// <para>Not for one the program named. A backend it asked for by name is an answer to that
    /// question and to no other, so <see cref="Factory"/>'s setter records its choice directly and
    /// <see cref="IsolatedBackend"/> records nothing at all.</para>
    /// </summary>
    public static void Remember(IShorokooInferenceSessionFactory factory)
    {
        ArgumentNullException.ThrowIfNull(factory);
        lock (_gate)
        {
            var incomingIsCpu = factory.Description.Device == ComputeDevice.Cpu;
            if (_remembered is null || incomingIsCpu)
            {
                if (_remembered is null
                    || incomingIsCpu && _remembered.Description.Device != ComputeDevice.Cpu)
                    _remembered = factory;
            }
        }
    }

    /// <summary>Forgets the remembered backend. Test hook: the rule is about what a process loaded,
    /// and a test process loads several.</summary>
    internal static void ForgetRemembered() { lock (_gate) _remembered = null; }

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
            ? "No shipped backend runs on another provider, so assign InferenceBackend.Factory with "
              + "your own before the first inference call."
            : "Reference the Shorokoo.{WinCPU,WinGPU,LinuxCPU,LinuxGPU} package for the device you "
              + "want, or assign InferenceBackend.Factory before the first inference call.";
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

    private static IShorokooInferenceSessionFactory Discover()
    {
        // A backend already loaded in the process wins -- it avoids pulling a
        // second native in alongside one the consumer has already bound.
        var preLoaded = TryFindAlreadyLoadedFactory();
        if (preLoaded is not null) return Remembering(preLoaded)!;

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
        return Remembering(InstantiateFactory(Assembly.LoadFrom(path)))
            ?? throw new InvalidOperationException(
                $"'{chosen.Assembly}' was found at '{path}' but exposes no concrete " +
                $"{nameof(IShorokooInferenceSessionFactory)}.");
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
            $"Several Shorokoo inference backends are {origin}: {names}. Discovery picks the " +
            "backend for a program that named none, and this deployment gives it no way to " +
            "choose. Say which you mean: assign InferenceBackend.Factory before the first " +
            "inference call to make one of them the default. To run several at once, give each " +
            "ComputeContext its own factory -- new ComputeContext(new LinuxGpuInferenceFactory()) " +
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
    /// second backend made the first one ambiguous: a program could not load one without losing
    /// the ability to leave the default undeclared.</para>
    /// </summary>
    internal static Assembly[] DiscoverableAssemblies(IEnumerable<Assembly> assemblies)
        => [.. assemblies.Where(
            asm => AssemblyLoadContext.GetLoadContext(asm) == AssemblyLoadContext.Default)];

    /// <summary>Records a factory as it is produced, and hands it straight back.</summary>
    private static IShorokooInferenceSessionFactory? Remembering(IShorokooInferenceSessionFactory? factory)
    {
        if (factory is not null) Remember(factory);
        return factory;
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
        var assemblies = DiscoverableAssemblies(AppDomain.CurrentDomain.GetAssemblies());
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
        try
        {
            var type = asm.GetExportedTypes().FirstOrDefault(t =>
                typeof(IShorokooInferenceSessionFactory).IsAssignableFrom(t)
                && !t.IsAbstract
                && t.GetConstructor(Type.EmptyTypes) is not null);
            return type is null ? null : (IShorokooInferenceSessionFactory)Activator.CreateInstance(type)!;
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
