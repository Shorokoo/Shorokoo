using System.Globalization;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Shorokoo.Core.Backends;
using Shorokoo.Core.Factory.IR;
using OrtFloat16 = Microsoft.ML.OnnxRuntime.Float16;
using OrtBFloat16 = Microsoft.ML.OnnxRuntime.BFloat16;
using ShoFloat16 = Shorokoo.Core.Backends.Float16;
using ShoBFloat16 = Shorokoo.Core.Backends.BFloat16;
using TensorElementType = Microsoft.ML.OnnxRuntime.Tensors.TensorElementType;

namespace Shorokoo.OnnxRuntime;

/// <summary>
/// The <see cref="IShorokooBackend"/> implementation backed by ONNX
/// Runtime: it builds ORT sessions and ORT-backed tensor values for Shorokoo's inference
/// pipeline. It is platform-neutral and abstract — each platform package
/// (<c>Shorokoo.WinCPU</c>, <c>Shorokoo.WinGPU</c>, <c>Shorokoo.LinuxCPU</c>,
/// <c>Shorokoo.LinuxGPU</c>) subclasses it through its CPU or its CUDA constructor.
///
/// <para>You do not normally reference this type, or the <c>Shorokoo.OnnxRuntime</c>
/// package that carries it, directly: reference one platform package instead and let
/// <see cref="Shorokoo.Core.Backends.DefaultBackend"/> find its backend.
/// Subclass this only to drive a different ONNX Runtime execution provider than the four
/// shipped packages offer, appending it through the constructor that takes a delegate.</para>
/// </summary>
public abstract class OrtBackend : IShorokooBackend
{
    private readonly Action<SessionOptions, DeviceMemorySettings, PrecisionSettings> _configureExecutionProvider;
    private readonly int? _cudaDeviceId;

    // Whether every session runs on ONNX Runtime's own CPU provider or its CUDA provider, the two
    // whose graph is known not to change for being written out (see CreateSession), rather than
    // on a provider a subclass appended.
    private readonly bool _stockProvider;

    /// <summary>
    /// The constructor for a subclass that appends an execution provider of its own. Its sessions
    /// are never built a second time to save memory, as those of the CPU and CUDA constructors
    /// may be (see <see cref="CreateSession(ReadOnlyMemory{byte}, ShorokooGraphOptimization, ShorokooLogSeverity, DeviceMemorySettings, DiagnosticSettings, IReadOnlyList{OutputAlias})"/>):
    /// nothing here knows what that provider does to a graph.
    /// </summary>
    /// <param name="configureExecutionProvider">
    /// Applied to the <see cref="SessionOptions"/> of every session this backend creates,
    /// after the log-severity and graph-optimization settings and before the session is
    /// constructed. This is where a subclass appends its execution provider. It is handed the
    /// <see cref="DeviceMemorySettings"/> and the <see cref="PrecisionSettings"/> of the session
    /// being built — the settings belong to that session, so they arrive with it rather than being
    /// read from anywhere else. A provider that can compute <c>float32</c> in less than full
    /// precision is to be configured to compute it in full precision unless
    /// <see cref="PrecisionSettings.AllowTensorFloat32"/> allows otherwise, as
    /// <see cref="AppendCuda"/> configures CUDA.
    /// </param>
    /// <param name="device">
    /// The kind of device those sessions run on. A subclass driving a provider that is neither
    /// the CPU nor CUDA — DirectML, ROCm, CoreML — passes <see cref="ComputeDevice.Other"/>.
    /// It is a parameter rather than something inferred from <paramref name="cudaDeviceId"/>
    /// precisely because such a subclass names no CUDA device: inferring would report it as the
    /// CPU, and <see cref="DefaultBackend.RequireDevice"/> would then wave work onto a card
    /// its author meant to stay off.
    /// </param>
    /// <param name="cudaDeviceId">
    /// The CUDA device the provider appended above allocates on, or <c>null</c> when it is
    /// not a CUDA provider. It names the device whose allocator the sessions allocate through and
    /// <see cref="RunSettings.ShrinkArenaAfterRun"/> hands memory back to: that card, or the host
    /// where it is <c>null</c>.
    /// </param>
    /// <exception cref="ArgumentException"><paramref name="cudaDeviceId"/> disagrees with
    /// <paramref name="device"/>, or is negative.</exception>
    protected OrtBackend(
        Action<SessionOptions, DeviceMemorySettings, PrecisionSettings> configureExecutionProvider,
        ComputeDevice device,
        int? cudaDeviceId)
        : this(configureExecutionProvider, device, cudaDeviceId, stockProvider: false) { }

    /// <summary>
    /// The CPU-backend constructor: every session runs on ONNX Runtime's own CPU execution
    /// provider, its default, so none is appended.
    /// </summary>
    protected OrtBackend()
        : this(static (_, _, _) => { }, ComputeDevice.Cpu, cudaDeviceId: null, stockProvider: true) { }

    /// <summary>
    /// The CUDA-backend constructor: every session gets the CUDA execution provider on
    /// <paramref name="cudaDeviceId"/>, configured from the <see cref="DeviceMemorySettings"/> and
    /// <see cref="PrecisionSettings"/> that session is built with, and honours
    /// <see cref="RunSettings.ShrinkArenaAfterRun"/> for what Shorokoo's allocator keeps on that
    /// device on each run.
    /// </summary>
    protected OrtBackend(int cudaDeviceId)
        : this((opts, mem, precision) => AppendCuda(opts, cudaDeviceId, mem, precision), ComputeDevice.Cuda, cudaDeviceId, stockProvider: true) { }

    // The one the others call. Internal rather than private so a test can stand for a stock
    // provider while it watches each session being built.
    internal OrtBackend(
        Action<SessionOptions, DeviceMemorySettings, PrecisionSettings> configureExecutionProvider,
        ComputeDevice device,
        int? cudaDeviceId,
        bool stockProvider)
    {
        // Built here rather than on each read of Description, so a backend that could only
        // describe itself incoherently cannot be constructed at all.
        Description = new BackendDescription(GetType().Assembly.GetName().Name ?? GetType().Name, device, cudaDeviceId);
        _configureExecutionProvider = configureExecutionProvider;
        _cudaDeviceId = cudaDeviceId;
        _stockProvider = stockProvider;
        // Before anything of this runtime's makes ONNX Runtime's environment without them: the
        // environment's thread pools are made with it or never (see SessionsShareThreadPools).
        OrtEnvironment.Environment();
    }

    /// <summary>
    /// This backend: the assembly the concrete backend lives in, and the device the constructor
    /// named. Fixed at construction, so every read agrees and none can contradict the provider
    /// the subclass actually appended.
    /// </summary>
    public BackendDescription Description { get; }

    // One per loaded copy of this assembly, which is one per native ONNX Runtime: every backend
    // over the runtime the program loaded shares this object, and a backend IsolatedBackend loads
    // gets a private copy of this assembly and so an object of its own. That is exactly the line
    // between two backends that can hand each other an allocation and two that cannot -- the same
    // line OrtSession.Unwrap draws by type identity when it is fed a value.
    private static readonly object LoadedRuntime = new();

    /// <summary>
    /// The native ONNX Runtime this backend is bound to, shared by every backend over it. Two
    /// backends over one loaded runtime — a CPU backend and a CUDA backend in one process — can
    /// read each other's allocations in place on a device they share; a backend loaded by
    /// <see cref="IsolatedBackend"/> has a runtime of its own and cannot.
    /// </summary>
    public object RuntimeIdentity => LoadedRuntime;

    /// <summary>
    /// Where this backend's runs read a tensor of <paramref name="elementType"/> and leave one:
    /// on a CUDA backend the card's own memory, for every tensor but a string one, which ONNX
    /// Runtime keeps in host memory whatever its provider; on any other, host memory. That is
    /// where a session of a provider a subclass appends finds its inputs too: everything this
    /// backend builds is in host memory, and so is every output of such a session's runs, the
    /// provider copying what it reads to where it computes as it runs.
    /// </summary>
    public MemoryLocation RunMemoryOf(ShorokooTensorElementType elementType)
        => new(_cudaDeviceId is { } device && elementType != ShorokooTensorElementType.String
            ? MemorySpace.Cuda(device)
            : MemorySpace.Host, LoadedRuntime);

    /// <summary>
    /// <see cref="KernelWorkaroundSets.OnnxRuntime"/>: the rewrites around ONNX Runtime's kernels,
    /// on every execution provider, since each rewrite computes what the operator it replaces
    /// computes; and on a CUDA backend <see cref="KernelWorkaroundSets.OnnxRuntimeCuda"/>, those
    /// and the rewrites around what the CUDA provider alone does otherwise.
    /// </summary>
    public string? KernelWorkaroundSet => _cudaDeviceId is null
        ? KernelWorkaroundSets.OnnxRuntime
        : KernelWorkaroundSets.OnnxRuntimeCuda;

    /// <summary>
    /// Creates an ORT inference session over a serialized ONNX model, on this backend's
    /// execution provider.
    /// </summary>
    /// <param name="modelBytes">The serialized ONNX model.</param>
    /// <param name="graphOptimization">The ORT graph-optimization level to apply.</param>
    /// <param name="logSeverity">The minimum severity ORT logs at.</param>
    /// <param name="deviceMemory">The device-memory settings this session is built with, which
    /// limit what it allocates from its first block on.</param>
    public IShorokooSession CreateSession(
        ReadOnlyMemory<byte> modelBytes,
        ShorokooGraphOptimization graphOptimization,
        ShorokooLogSeverity logSeverity,
        DeviceMemorySettings deviceMemory)
        => CreateSession(
            modelBytes, graphOptimization, logSeverity, deviceMemory, DiagnosticSettings.Default);

    /// <summary>
    /// <see cref="CreateSession(ReadOnlyMemory{byte}, ShorokooGraphOptimization, ShorokooLogSeverity, DeviceMemorySettings)"/>,
    /// also recording what <paramref name="diagnostics"/> asks for. ORT reads the profiler switch
    /// while the session is being created and the session keeps it for life, which is why it
    /// arrives here and not per run.
    /// </summary>
    /// <param name="modelBytes">The serialized ONNX model.</param>
    /// <param name="graphOptimization">The ORT graph-optimization level to apply.</param>
    /// <param name="logSeverity">The minimum severity ORT logs at.</param>
    /// <param name="deviceMemory">The device-memory settings this session is built with.</param>
    /// <param name="diagnostics">What the session records about itself, and whether it computes
    /// deterministically. Its default records nothing and leaves the runtime's kernels as they are,
    /// which is what every session gets unless a context asked otherwise.</param>
    public IShorokooSession CreateSession(
        ReadOnlyMemory<byte> modelBytes,
        ShorokooGraphOptimization graphOptimization,
        ShorokooLogSeverity logSeverity,
        DeviceMemorySettings deviceMemory,
        DiagnosticSettings diagnostics)
        => Build(modelBytes, graphOptimization, logSeverity, deviceMemory, diagnostics, outputAliases: null, intraOpThreads: 0, [],
            PrecisionSettings.Default);

    /// <summary>
    /// <see cref="CreateSession(ReadOnlyMemory{byte}, ShorokooGraphOptimization, ShorokooLogSeverity, DeviceMemorySettings, DiagnosticSettings)"/>,
    /// for a session that may write the outputs <paramref name="outputAliases"/> names into the
    /// memory of the inputs it pairs them with, on a run that consumed those inputs.
    ///
    /// <para>The pairs were proved over the model as handed over, and ONNX Runtime does not run that
    /// model: it rewrites it first, and a rewrite can change which nodes read an input. Measured on
    /// a training step, ORT fused one of the two <c>MatMul</c>s reading a weight through a
    /// <c>Transpose</c> into a <c>FusedMatMul</c> reading the weight itself — a new direct reader
    /// the handed-over model never had. So the session is built with ORT writing out the graph it
    /// will actually run, and a pair is kept only where <see cref="OutputAliasProof"/> proves it
    /// again over that graph. A pair the rewritten graph no longer proves is dropped, and a graph
    /// that cannot be read back keeps none.</para>
    ///
    /// <para>Writing the graph out costs the session more than the write: while it saves the
    /// graph, ORT keeps every initializer in it — folded constants included — for the session's
    /// life, where otherwise it lets them go once the session holds its own copy. Where that copy
    /// is a second one, the cost is the initializers over again: measured on the CPU, a session
    /// over a 256 MiB <c>MatMul</c> weight, which ORT repacks for its kernel, took 524 MiB of
    /// process memory built while writing and 270 MiB without, where one over a 256 MiB
    /// <c>Add</c> operand, which the kernel reads as it is, took 270 MiB either way. On a card
    /// every initializer is a second copy by construction, the host's beside the card's. So where
    /// the graph written out holds more than 16 MiB of initializers, in the file beside it and
    /// inline, the session is disposed and built again without writing, keeping the pairs proved
    /// over the graph the first build wrote: the same model with the same options makes the same
    /// graph, writing it out changing nothing else the CPU and CUDA providers do. Below that,
    /// keeping the copy costs less than a second build.</para>
    ///
    /// <para>That premise is known only of the providers that ship here, so a session of a
    /// backend that appends a provider of its own stays as first built. DirectML fuses its graph
    /// only when none is being written out, so a DirectML session built to alias runs unfused —
    /// and built again, it would run a graph the pairs were never proved over. A provider that
    /// compiles nodes cannot have its graph written out at all, so a session built to alias on
    /// one is built twice, the second time aliasing nothing; and where no folder for the graph can
    /// be made in the temp folder, the session is built aliasing nothing from the start.</para>
    /// </summary>
    public IShorokooSession CreateSession(
        ReadOnlyMemory<byte> modelBytes,
        ShorokooGraphOptimization graphOptimization,
        ShorokooLogSeverity logSeverity,
        DeviceMemorySettings deviceMemory,
        DiagnosticSettings diagnostics,
        IReadOnlyList<OutputAlias> outputAliases)
        => CreateSession(
            modelBytes, graphOptimization, logSeverity, deviceMemory, diagnostics, outputAliases,
            intraOpThreads: 0);

    /// <summary>
    /// <see cref="CreateSession(ReadOnlyMemory{byte}, ShorokooGraphOptimization, ShorokooLogSeverity, DeviceMemorySettings, DiagnosticSettings, IReadOnlyList{OutputAlias})"/>,
    /// with ONNX Runtime's intra-op thread pool sized to <paramref name="intraOpThreads"/>: 1 runs
    /// every operator on the thread that called the run, for sessions run side by side; 0 leaves
    /// ONNX Runtime's own default, a thread per core.
    /// </summary>
    public IShorokooSession CreateSession(
        ReadOnlyMemory<byte> modelBytes,
        ShorokooGraphOptimization graphOptimization,
        ShorokooLogSeverity logSeverity,
        DeviceMemorySettings deviceMemory,
        DiagnosticSettings diagnostics,
        IReadOnlyList<OutputAlias> outputAliases,
        int intraOpThreads)
    {
        ArgumentNullException.ThrowIfNull(outputAliases);
        ArgumentOutOfRangeException.ThrowIfNegative(intraOpThreads);
        return Build(
            modelBytes, graphOptimization, logSeverity, deviceMemory, diagnostics,
            outputAliases.Count == 0 ? null : outputAliases, intraOpThreads, [], PrecisionSettings.Default);
    }

    /// <summary>
    /// <see cref="CreateSession(ReadOnlyMemory{byte}, ShorokooGraphOptimization, ShorokooLogSeverity, DeviceMemorySettings, DiagnosticSettings, IReadOnlyList{OutputAlias}, int)"/>,
    /// with the initializers <paramref name="suppliedInitializers"/> names taken as the values they
    /// are, where they are — on a card, weights loaded straight into its memory, which ONNX Runtime
    /// then reads in place rather than copying from the model (Shorokoo/Shorokoo#436).
    /// </summary>
    public IShorokooSession CreateSession(
        ReadOnlyMemory<byte> modelBytes,
        ShorokooGraphOptimization graphOptimization,
        ShorokooLogSeverity logSeverity,
        DeviceMemorySettings deviceMemory,
        DiagnosticSettings diagnostics,
        IReadOnlyList<OutputAlias> outputAliases,
        int intraOpThreads,
        IReadOnlyList<SuppliedInitializer> suppliedInitializers)
        => CreateSession(
            modelBytes, graphOptimization, logSeverity, deviceMemory, diagnostics, outputAliases, intraOpThreads,
            suppliedInitializers, PrecisionSettings.Default);

    /// <summary>
    /// <see cref="CreateSession(ReadOnlyMemory{byte}, ShorokooGraphOptimization, ShorokooLogSeverity, DeviceMemorySettings, DiagnosticSettings, IReadOnlyList{OutputAlias}, int, IReadOnlyList{SuppliedInitializer})"/>,
    /// computing in <paramref name="precision"/>, which reaches the execution-provider step with
    /// <paramref name="deviceMemory"/>. On a CUDA backend
    /// <see cref="PrecisionSettings.AllowTensorFloat32"/> is the CUDA provider's <c>use_tf32</c>
    /// (<see cref="CudaProviderOptions"/>); a CPU session computes <c>float32</c> in full precision
    /// whatever it says. The other overloads build in <see cref="PrecisionSettings.Default"/>.
    /// </summary>
    public IShorokooSession CreateSession(
        ReadOnlyMemory<byte> modelBytes,
        ShorokooGraphOptimization graphOptimization,
        ShorokooLogSeverity logSeverity,
        DeviceMemorySettings deviceMemory,
        DiagnosticSettings diagnostics,
        IReadOnlyList<OutputAlias> outputAliases,
        int intraOpThreads,
        IReadOnlyList<SuppliedInitializer> suppliedInitializers,
        PrecisionSettings precision)
    {
        ArgumentNullException.ThrowIfNull(outputAliases);
        ArgumentNullException.ThrowIfNull(suppliedInitializers);
        ArgumentNullException.ThrowIfNull(precision);
        ArgumentOutOfRangeException.ThrowIfNegative(intraOpThreads);
        return Build(
            modelBytes, graphOptimization, logSeverity, deviceMemory, diagnostics,
            outputAliases.Count == 0 ? null : outputAliases, intraOpThreads, suppliedInitializers, precision);
    }

    /// <summary>
    /// For tests that hold Shorokoo's allocator to ONNX Runtime's own: every session this backend
    /// builds allocates through the arena ONNX Runtime makes for it, as a session not handed the
    /// environment's allocators does, and reports that arena's figures. A kept output then holds its
    /// session's arena, so nothing but such a comparison builds one.
    /// </summary>
    internal bool SessionsUseOrtArena { get; init; }

    /// <summary>
    /// Whether the sessions this backend builds run their operators on ONNX Runtime's thread pools
    /// of the process, which every such session shares, rather than each on an intra-op pool of its
    /// own; true unless set otherwise. A pool's threads spin for a while after the work they were
    /// given, waiting for more, so two sessions with pools of their own run one after the other —
    /// a compiled graph's session and the one a run that consumes its input places its values
    /// through, or two compiled graphs — each run contending with the other pool's spinning threads:
    /// measured on the host, a two-layer encoder ran a consuming and a shared run in 385–450 ms
    /// that way against 113–118 on shared pools, and a session run alone took as long either way.
    /// A session built with an intra-op thread count of its own — one run side by side with others
    /// on a thread each — keeps a pool of its own of that size whatever this says. Where something
    /// else made the process's ONNX Runtime environment first, every session keeps its own.
    /// </summary>
    public bool SessionsShareThreadPools { get; init; } = true;

    /// <summary>Whether this backend's sessions run on a CUDA card.</summary>
    internal bool OnCard => _cudaDeviceId is not null;

    /// <summary>This backend's sessions take supplied initializers.</summary>
    public bool SuppliesInitializers => true;

    // The most bytes of initializers a session built while writing its graph out keeps as it was
    // built, holding them twice, rather than being built again without writing (see CreateSession).
    private const long InitializersKeptTwice = 16L << 20;

    private IShorokooSession Build(
        ReadOnlyMemory<byte> modelBytes,
        ShorokooGraphOptimization graphOptimization,
        ShorokooLogSeverity logSeverity,
        DeviceMemorySettings deviceMemory,
        DiagnosticSettings diagnostics,
        IReadOnlyList<OutputAlias>? outputAliases,
        int intraOpThreads,
        IReadOnlyList<SuppliedInitializer> suppliedInitializers,
        PrecisionSettings precision)
    {
        ArgumentNullException.ThrowIfNull(deviceMemory);
        ArgumentNullException.ThrowIfNull(diagnostics);
        // One copy for however many sessions are built from it: ORT takes the model as an array.
        var model = modelBytes.ToArray();
        if (!SessionPlacing.Suppressed && WeightsToShare(model, outputAliases, suppliedInitializers) is { } weights
            && BuildSharingWeights(model, weights, graphOptimization, logSeverity, deviceMemory, diagnostics, intraOpThreads, suppliedInitializers, precision) is { } sharing)
            return sharing;
        BuiltSession New(string? optimizedDirectory) => NewSession(
            model, graphOptimization, logSeverity, deviceMemory, diagnostics, optimizedDirectory, intraOpThreads,
            suppliedInitializers, precision);
        var placing = _stockProvider && !SessionsUseOrtArena && !SessionPlacing.Suppressed;
        var session = BuildSession(model, New, outputAliases, placing, out var written);
        // A session of a stock provider can place the values of a run that consumes inputs in the
        // memory of what it consumes (see OrtPlacements), through sessions of its own built over the
        // same model the same way, or over the graph the session runs as ONNX Runtime wrote it out.
        // They charge the session's own allocator accounts, so that what a run of the session takes
        // is read, budgeted and limited as one session's, whichever of them ran it. Not where its
        // sessions allocate through ONNX Runtime's own arena, a comparison's alone: what a placed run
        // saves is measured on Shorokoo's allocator.
        if (placing)
            session.Placements = new OrtPlacements(
                model,
                written,
                (variant, directory, externalData, shared) => Wrap(NewSession(
                    variant, externalData is null ? graphOptimization : ShorokooGraphOptimization.DisableAll, logSeverity,
                    deviceMemory, diagnostics with { TraceNodePlacement = false },
                    directory, intraOpThreads, suppliedInitializers, precision,
                    accounts: shared ? (session.HostAccount, session.CardAccount) : null, externalDataDirectory: externalData), []),
                this,
                () => session.HeldBytes);
        return session;
    }

    /// <summary>
    /// The weights of <paramref name="model"/> a session on a card reads from copies of this
    /// backend's rather than the model's, so that the sessions it places values through read them
    /// too, holding no copy of their own: on a card of a stock provider, a model over
    /// <see cref="OrtPlacements.ModelBytesKept"/> — whose second session would otherwise hold a
    /// second copy of the weights it carries — built with no output written into an input. Its
    /// initializers and constants of a mebibyte or more of fixed-width elements that it carries
    /// itself; null for any other model, or where it carries none. On the host such a copy saves
    /// nothing: ONNX Runtime
    /// packs a product's constant weight into memory of the session's own, and keeps a weight it was
    /// handed beside its packed copy.
    /// </summary>
    private IReadOnlyList<TensorProto>? WeightsToShare(
        byte[] model, IReadOnlyList<OutputAlias>? outputAliases, IReadOnlyList<SuppliedInitializer> suppliedInitializers)
    {
        if (!_stockProvider || SessionsUseOrtArena || _cudaDeviceId is null || outputAliases is not null
            || model.Length <= OrtPlacements.ModelBytesKept)
            return null;
        ModelProto parsed;
        using (var stream = new MemoryStream(model, writable: false))
            parsed = ProtoBuf.Serializer.Deserialize<ModelProto>(stream);
        var supplied = suppliedInitializers.Select(s => s.Name).ToHashSet(StringComparer.Ordinal);
        // Its initializers, and the tensors of its Constant nodes, which ONNX Runtime makes
        // initializers of the nodes' outputs as it loads the model.
        var tensors = (parsed.Graph?.Initializers ?? []).Concat((parsed.Graph?.Nodes ?? [])
            .Where(n => n.OpType == "Constant" && n.Domain is null or "" && n.Outputs.Count == 1)
            .SelectMany(n => n.Attributes.Where(a => a.Name == "value" && a.T is not null)
                .Select(a => new TensorProto { Name = n.Outputs[0], data_type = a.T.data_type, Dims = a.T.Dims, RawData = a.T.RawData })));
        List<TensorProto> weights = [.. tensors.Where(t =>
            t.RawData is { Length: >= 1 << 20 } && !supplied.Contains(t.Name)
            && PlacementShapes.ElementBytes(t.data_type) > 0
            && (t.Dims ?? []).Aggregate(1L, (a, d) => a * d) * PlacementShapes.ElementBytes(t.data_type) == t.RawData.Length)];
        return weights.Count == 0 ? null : weights;
    }

    /// <summary>
    /// A session over <paramref name="model"/> reading <paramref name="weights"/> from copies of
    /// this backend's (<see cref="WeightsToShare"/>), with the placements of its runs built over the
    /// graph it writes out as it is built — its larger initializers in a file beside it, so that
    /// writing it holds no second copy of them — and handed the copies its graph still reads; those
    /// ONNX Runtime's rewrites left unread are let go of. Null where the session cannot be built that
    /// way — the folder to write into cannot be made, or the build or the graph it writes fails — for
    /// the caller to build the session as any other.
    /// </summary>
    private OrtSession? BuildSharingWeights(
        byte[] model, IReadOnlyList<TensorProto> weights, ShorokooGraphOptimization graphOptimization, ShorokooLogSeverity logSeverity,
        DeviceMemorySettings deviceMemory, DiagnosticSettings diagnostics, int intraOpThreads,
        IReadOnlyList<SuppliedInitializer> suppliedInitializers, PrecisionSettings precision)
    {
        var directory = TempDirectory("shorokoo-runs-");
        try
        {
            Directory.CreateDirectory(directory);
        }
        catch (Exception unwritable) when (unwritable is IOException or UnauthorizedAccessException)
        {
            return null;
        }
        OrtSession session;
        ModelProto runs;
        List<(string Name, OrtTensorValue Value)> kept = [];
        try
        {
            var built = NewSession(model, graphOptimization, logSeverity, deviceMemory, diagnostics, directory, intraOpThreads,
                suppliedInitializers, precision, weightsToShare: weights);
            try
            {
                runs = OrtPlacements.ReadOptimized(directory);
            }
            catch
            {
                Discard(built);
                throw;
            }
            var read = runs.Graph!.Initializers.Select(i => i.Name).ToHashSet(StringComparer.Ordinal);
            for (int i = 0; i < weights.Count; i++)
                if (read.Contains(weights[i].Name)) kept.Add((weights[i].Name, built.SharedWeights[i]));
                else built.SharedWeights[i].Dispose();
            session = Wrap(built with { SharedWeights = [.. kept.Select(k => k.Value)] }, []);
        }
        catch (Exception failure) when (failure is not OutOfMemoryException)
        {
            // Sharing the weights is a saving and never a requirement: a build that cannot write its
            // graph out -- a full disk -- is built again as any other, where a fault of the model's
            // own fails it again, as it would have anyway.
            DeleteDirectory(directory);
            return null;
        }
        IReadOnlyList<SuppliedInitializer> handed = [.. suppliedInitializers, .. kept.Select(k => new SuppliedInitializer(k.Name, k.Value))];
        session.Placements = new OrtPlacements(
            directory, runs, kept.Select(k => k.Name).ToHashSet(StringComparer.Ordinal),
            (variant, optimized, externalData, shared) => Wrap(NewSession(
                variant, externalData is null ? graphOptimization : ShorokooGraphOptimization.DisableAll, logSeverity,
                deviceMemory, diagnostics with { TraceNodePlacement = false },
                optimized, intraOpThreads, handed, precision,
                accounts: shared ? (session.HostAccount, session.CardAccount) : null, externalDataDirectory: externalData), []),
            this,
            () => session.HeldBytes);
        return session;
    }

    /// <summary>
    /// A session over <paramref name="model"/>, binding the pairs of <paramref name="outputAliases"/>
    /// the graph ONNX Runtime runs still proves. Where it wrote that graph out to prove them and the
    /// session will place values (<paramref name="keep"/>), <paramref name="written"/> answers the
    /// folder it is in and the graph, for the placements to build from in place of a build of their
    /// own; null otherwise, the folder then deleted.
    /// </summary>
    private OrtSession BuildSession(
        byte[] model, Func<string?, BuiltSession> New, IReadOnlyList<OutputAlias>? outputAliases, bool keep,
        out OrtPlacements.Written? written)
    {
        written = null;
        if (outputAliases is null) return Wrap(New(null), []);

        var optimizedDirectory = TempDirectory("shorokoo-optimized-");
        try
        {
            // Made here, apart from the build, so that a temp folder this process cannot write in
            // is told apart from a failure of the build itself: aliasing is a saving and never a
            // requirement, so a session whose graph has nowhere to be written out is built as one
            // that aliases nothing, as it would be with no aliasing asked for. Only the folder is
            // made here: the runtime failing to write into it -- a disk filling as it writes --
            // reports nothing that tells it apart from any other failure of the build, and fails
            // the build like one.
            try
            {
                Directory.CreateDirectory(optimizedDirectory);
            }
            catch (Exception unwritable) when (unwritable is IOException or UnauthorizedAccessException)
            {
                return Wrap(New(null), []);
            }

            BuiltSession built;
            try
            {
                built = New(optimizedDirectory);
            }
            // ONNX Runtime cannot write out a graph holding nodes an execution provider compiled
            // (TensorRT, OpenVINO and the like), and refuses to build a session asked to. Aliasing
            // is a saving and never a requirement, so such a session is built again as one that
            // aliases nothing. That refusal alone: any other failure is the build's own and goes
            // to the caller, where caught here it would have been paid for twice when the model
            // cannot be built at all, and when it could -- an allocation failing while the session
            // initialized, say -- left the graph a session that never aliases, for its whole life
            // and without a word.
            catch (OnnxRuntimeException refusal) when (RefusesToWriteCompiledNodes(refusal))
            {
                return Wrap(New(null), []);
            }

            var (proved, initializerBytes, graph) = ProvedAgain(optimizedDirectory, outputAliases);
            if (initializerBytes > InitializersKeptTwice && _stockProvider)
            {
                Discard(built);
                built = New(null);
            }
            var session = Wrap(built, proved);
            // The graph a build of the same model with the same options writes, whether or not the
            // session was built again without writing it.
            if (keep && graph is not null) written = new OrtPlacements.Written(optimizedDirectory, graph);
            return session;
        }
        finally
        {
            if (written is null) DeleteDirectory(optimizedDirectory);
        }
    }

    /// <summary>
    /// Whether <paramref name="failure"/> is ONNX Runtime refusing to write out a graph because an
    /// execution provider compiled some of its nodes. It says so in this message and in no other
    /// way; were a later version to word it otherwise, a session on such a provider would fail to
    /// build rather than quietly alias nothing, which is the direction to fail in.
    /// </summary>
    private static bool RefusesToWriteCompiledNodes(OnnxRuntimeException failure)
        => failure.Message.Contains("contains compiled nodes", StringComparison.Ordinal);

    /// <summary>An ONNX Runtime session, and the folder it writes its profile into, if it keeps
    /// one: what <see cref="OrtSession"/> is made of.</summary>
    internal readonly record struct BuiltSession(
        InferenceSession Session,
        string? ProfileDirectory,
        IReadOnlyList<OrtValue> SuppliedViews,
        CachingAllocator.Account HostAccount,
        CachingAllocator.Account? CardAccount,
        bool OwnsAccounts = true)
    {
        /// <summary>The weights the session reads from copies of this backend's (see
        /// <see cref="WeightsToShare"/>), which the session owns.</summary>
        internal IReadOnlyList<OrtTensorValue> SharedWeights { get; init; } = [];
    }

    /// <summary>
    /// An ONNX Runtime session over <paramref name="model"/>, writing the graph it will run into
    /// <paramref name="optimizedDirectory"/> where one is named, and charging what it allocates to
    /// <paramref name="accounts"/> where they are given — another session's, which keeps them —
    /// or to accounts of its own.
    /// </summary>
    internal BuiltSession NewSession(
        byte[] model,
        ShorokooGraphOptimization graphOptimization,
        ShorokooLogSeverity logSeverity,
        DeviceMemorySettings deviceMemory,
        DiagnosticSettings diagnostics,
        string? optimizedDirectory,
        int intraOpThreads,
        IReadOnlyList<SuppliedInitializer> suppliedInitializers,
        PrecisionSettings precision,
        (CachingAllocator.Account Host, CachingAllocator.Account? Card)? accounts = null,
        string? externalDataDirectory = null,
        IReadOnlyList<TensorProto>? weightsToShare = null)
    {
        try
        {
            return NewSessionOnce(model, graphOptimization, logSeverity, deviceMemory, diagnostics, optimizedDirectory,
                intraOpThreads, suppliedInitializers, precision, accounts, externalDataDirectory, weightsToShare);
        }
        catch (OnnxRuntimeException refused) when (refused.Message.Contains("CreateEnvWithGlobalThreadPools", StringComparison.Ordinal))
        {
            // An environment made elsewhere with no pools of its own refuses a session asked to run
            // on them: from now on every session keeps its own.
            OrtEnvironment.NoSharedThreadPools();
            return NewSessionOnce(model, graphOptimization, logSeverity, deviceMemory, diagnostics, optimizedDirectory,
                intraOpThreads, suppliedInitializers, precision, accounts, externalDataDirectory, weightsToShare);
        }
    }

    private BuiltSession NewSessionOnce(
        byte[] model,
        ShorokooGraphOptimization graphOptimization,
        ShorokooLogSeverity logSeverity,
        DeviceMemorySettings deviceMemory,
        DiagnosticSettings diagnostics,
        string? optimizedDirectory,
        int intraOpThreads,
        IReadOnlyList<SuppliedInitializer> suppliedInitializers,
        PrecisionSettings precision,
        (CachingAllocator.Account Host, CachingAllocator.Account? Card)? accounts,
        string? externalDataDirectory,
        IReadOnlyList<TensorProto>? weightsToShare)
    {
        // The `using` is load-bearing, not tidiness. SessionOptions is a SafeHandle, so it
        // carries a critical finalizer that calls OrtReleaseSessionOptions, and ORT takes its
        // handle as a bare IntPtr -- the P/Invoke does no SafeHandle ref-counting, and the
        // session does not retain the options the caller passes. Left as a plain local, the
        // options are unreachable the instant that IntPtr is read, so a GC landing inside
        // session creation frees them while ORT is still walking sess_options->provider_factories
        // (core/session/utils.cc, InitializeSession) -- a use-after-free that segfaults the
        // process. Disposing in a finally keeps them rooted across the constructor.
        using var options = new SessionOptions();
        Configure(options, graphOptimization, logSeverity);
        if (intraOpThreads > 0) options.IntraOpNumThreads = intraOpThreads;
        // A session with no thread count of its own runs its operators on the process's pools, so
        // that two sessions run one after another -- a session and the one it places values
        // through, or two compiles -- do not each keep a pool whose threads spin against the
        // other's (see SessionsShareThreadPools).
        var sharedPools = SessionsShareThreadPools && intraOpThreads == 0 && OrtEnvironment.SharedThreadPools;
        if (sharedPools) options.DisablePerSessionThreads();
        if (diagnostics.DeterministicCompute) UseDeterministicCompute(options);
        // Named before anything can throw, and made inside the try, by the call that points the
        // options into it: a setter there throwing after the folder was made would otherwise leave
        // it with no name for the catch to delete it by. The folder the graph is written into is
        // the caller's, made and deleted there.
        var profileDirectory = diagnostics.TraceNodePlacement ? TempDirectory("shorokoo-node-placement-") : null;
        // A model whose initializers lie in files beside it is read with them from the caller's
        // folder, which also takes a supplied initializer's placeholder; otherwise a placeholder
        // gets a folder of its own, made here and deleted below.
        var placeholderDirectory = suppliedInitializers.Count > 0 && externalDataDirectory is null ? TempDirectory("shorokoo-supplied-") : null;
        List<OrtValue> views = [];
        List<OrtTensorValue> shared = [];
        // What the session allocates -- its weights as it is built, and everything its runs take --
        // comes from Shorokoo's allocators, charged to accounts of its own (see CachingAllocator):
        // ONNX Runtime writes every output into memory its session's allocator gives it, so an
        // output can hold only its own block only where that allocator is Shorokoo's. On a card, the
        // account carries the session's limit, which ONNX Runtime has no say in.
        var host = accounts?.Host ?? RuntimeAllocator.ForHost().Shared.Open("session");
        var card = accounts is { } given ? given.Card
            : _cudaDeviceId is { } device ? RuntimeAllocator.ForCard(device).Shared.Open("session") : null;
        if (card is not null && accounts is null) card.Limit = deviceMemory.LimitBytes;
        if (!SessionsUseOrtArena) options.AddSessionConfigEntry("session.use_env_allocators", "1");
        try
        {
            if (profileDirectory is not null) EnableProfiling(options, profileDirectory);
            if (optimizedDirectory is not null) WriteOptimizedModel(options, optimizedDirectory);
            if (placeholderDirectory is not null) Supply(options, placeholderDirectory, suppliedInitializers, views);
            else if (externalDataDirectory is not null)
            {
                if (suppliedInitializers.Count > 0) Supply(options, externalDataDirectory, suppliedInitializers, views);
                else options.AddSessionConfigEntry("session.model_external_initializers_file_folder_path", externalDataDirectory);
            }
            _configureExecutionProvider(options, deviceMemory, precision);
            // Copies of the weights in this backend's memory, charged to the session as its own
            // weights would be, which the session reads in place of copying the model's.
            if (weightsToShare is not null)
                using (CachingAllocator.Charge(host, card))
                    foreach (var weight in weightsToShare)
                    {
                        var copy = (OrtTensorValue)CreateTensorInBackendMemory((ShorokooTensorElementType)weight.data_type, weight.RawData, weight.Dims);
                        shared.Add(copy);
                        using var memory = copy.Inner.GetTensorMemoryInfo();
                        var view = OrtValue.CreateTensorValueWithData(
                            memory, (TensorElementType)weight.data_type, weight.Dims, DevicePointer(copy), weight.RawData.Length);
                        GC.KeepAlive(copy);
                        views.Add(view);
                        options.AddInitializer(weight.Name, view);
                    }
            InferenceSession session;
            using (CachingAllocator.Charge(host, card))
                session = new InferenceSession(model, options);
            // The values themselves are the caller's to keep alive for the session's life; this
            // keeps them reachable across the constructor, which takes them as bare handles.
            GC.KeepAlive(suppliedInitializers);
            return new BuiltSession(session, profileDirectory, views, host, card, accounts is null) { SharedWeights = shared };
        }
        catch
        {
            foreach (var view in views) view.Dispose();
            foreach (var copy in shared) copy.Dispose();
            if (accounts is null)
            {
                host.Allocator.Close(host);
                card?.Allocator.Close(card);
            }
            // No session to own the folder, so nothing would ever delete it.
            DeleteDirectory(profileDirectory);
            throw;
        }
        finally
        {
            // ONNX Runtime checks the placeholder exists as it builds the session and never reads
            // it, the values arriving from the options instead.
            DeleteDirectory(placeholderDirectory);
        }
    }

    /// <summary><paramref name="built"/> as this backend's session, binding the pairs of
    /// <paramref name="outputAliases"/> it can.</summary>
    internal OrtSession Wrap(BuiltSession built, IReadOnlyList<OrtSession.ProvedAlias> outputAliases)
    {
        var (session, profileDirectory, views, host, card, owns) = built;
        try
        {
            // The session keeps this backend to release what its runs consume through it, and to
            // name it in a refusal.
            return new OrtSession(session, _cudaDeviceId, this, profileDirectory, outputAliases, host, card, owns)
            {
                SuppliedViews = views,
                SharedWeights = built.SharedWeights,
                OnOrtArena = SessionsUseOrtArena,
            };
        }
        catch
        {
            Discard(built);
            throw;
        }
    }

    /// <summary>Releases a session nothing will own, and its profile folder.</summary>
    private static void Discard(BuiltSession built)
    {
        built.Session.Dispose();
        foreach (var weight in built.SharedWeights) weight.Dispose();
        if (built.OwnsAccounts)
        {
            built.HostAccount.Allocator.Close(built.HostAccount);
            built.CardAccount?.Allocator.Close(built.CardAccount);
        }
        foreach (var view in built.SuppliedViews) view.Dispose();
        DeleteDirectory(built.ProfileDirectory);
    }

    /// <summary>A folder of its own in the temp folder, named and not yet made.</summary>
    private static string TempDirectory(string prefix)
        => Path.Combine(Path.GetTempPath(), prefix + Guid.NewGuid().ToString("N"));

    // What the optimized model is called inside the folder it is written to, and the file its
    // larger initializers go to beside it: they are the constants ORT folded, which the proof does
    // not read, so they are kept out of the model it parses.
    internal const string OptimizedModelFile = "optimized.onnx";
    private const string OptimizedInitializersFile = "initializers.bin";

    /// <summary>
    /// Has ONNX Runtime write the graph it will run — after its rewrites, with the nodes that
    /// actually execute — into <paramref name="directory"/>, which the caller has made. The
    /// initializers above a kibibyte go to a file beside it rather than into the model, so a folded
    /// constant the size of a tensor costs a write and not a parse.
    /// </summary>
    private static void WriteOptimizedModel(SessionOptions options, string directory)
    {
        options.OptimizedModelFilePath = Path.Combine(directory, OptimizedModelFile);
        options.AddSessionConfigEntry(
            "session.optimized_model_external_initializers_file_name", OptimizedInitializersFile);
        options.AddSessionConfigEntry(
            "session.optimized_model_external_initializers_min_size_in_bytes", "1024");
    }

    /// <summary>
    /// The pairs of <paramref name="outputAliases"/> the graph ONNX Runtime wrote into
    /// <paramref name="directory"/> still proves, each with the shape that graph states for its
    /// output, and the bytes of initializers the graph holds, in the file beside it and inline.
    /// Where it wrote nothing that can be read, no pairs — a pair this cannot prove is not bound,
    /// which costs the memory and never the result — initializers of any size, which rules out
    /// keeping them twice unseen, and no graph.
    /// </summary>
    private static (IReadOnlyList<OrtSession.ProvedAlias> Proved, long InitializerBytes, ModelProto? Graph) ProvedAgain(
        string directory, IReadOnlyList<OutputAlias> outputAliases)
    {
        try
        {
            ModelProto model;
            using (var stream = File.OpenRead(Path.Combine(directory, OptimizedModelFile)))
                model = ProtoBuf.Serializer.Deserialize<ModelProto>(stream);
            if (model.Graph is not { } graph) return ([], long.MaxValue, null);
            var stated = new Dictionary<string, TypeProto?>(StringComparer.Ordinal);
            foreach (var output in graph.Outputs) stated.TryAdd(output.Name, output.Type);
            var aside = new FileInfo(Path.Combine(directory, OptimizedInitializersFile));
            return (
                [.. OutputAliasProof.Prove(graph, outputAliases).Select(alias =>
                    new OrtSession.ProvedAlias(alias, StatedShape(stated.GetValueOrDefault(alias.Output))))],
                (aside.Exists ? aside.Length : 0) + graph.Initializers.Sum(InlineBytes),
                model);
        }
        catch (Exception)
        {
            return ([], long.MaxValue, null);
        }
    }

    /// <summary>The bytes of <paramref name="initializer"/>'s contents the model holds inline:
    /// none for one ONNX Runtime wrote to the file beside it.</summary>
    private static long InlineBytes(TensorProto initializer)
        => (initializer.RawData?.LongLength ?? 0)
           + 4L * ((initializer.FloatDatas?.Length ?? 0) + (initializer.Int32Datas?.Length ?? 0))
           + 8L * ((initializer.Int64Datas?.Length ?? 0) + (initializer.DoubleDatas?.Length ?? 0)
                   + (initializer.Uint64Datas?.Length ?? 0))
           + initializer.StringDatas.Sum(bytes => (long)bytes.Length);

    /// <summary>
    /// The shape <paramref name="type"/> states, where it states one in full — every dimension a
    /// known positive number, and none for a scalar — and null where it states no shape at all or
    /// leaves a dimension open.
    /// </summary>
    private static long[]? StatedShape(TypeProto? type)
    {
        if (type?.TensorType?.Shape is not { } shape) return null;
        var dims = new long[shape.Dims.Count];
        for (int i = 0; i < dims.Length; i++)
        {
            if (shape.Dims[i].DimValue <= 0 || shape.Dims[i].DimParam.Length > 0) return null;
            dims[i] = shape.Dims[i].DimValue;
        }
        return dims;
    }

    /// <summary>
    /// Turns ORT's profiler on for a session asked to trace where its nodes ran, writing into
    /// <paramref name="directory"/>, which this makes. A folder of its own per session: two
    /// sessions profiling at once would otherwise agree on a prefix, and ORT tells files apart by
    /// timestamp alone.
    ///
    /// <para><b>The prefix is set before the switch is thrown, and the order is load-bearing.</b>
    /// ORT reads the prefix at the moment profiling is enabled and ignores any later change, so
    /// setting it afterwards writes the profile into the process's working directory under ORT's
    /// own default name — a stray file per session, in whatever folder the program happens to be
    /// running from, that nothing then cleans up.</para>
    /// </summary>
    private static void EnableProfiling(SessionOptions options, string directory)
    {
        Directory.CreateDirectory(directory);
        options.ProfileOutputPathPrefix = Path.Combine(directory, "profile");
        options.EnableProfiling = true;
    }

    /// <summary>
    /// Has the session <paramref name="options"/> build run the deterministic kernel wherever an
    /// operator has one (<see cref="DiagnosticSettings.DeterministicCompute"/>).
    ///
    /// <para>ONNX Runtime's C API has the switch, <c>SetDeterministicCompute</c>, and its managed
    /// surface does not wrap it, so it is called through the managed runtime's own table of that
    /// API: the table belongs to the ONNX Runtime assembly this backend is bound to, and so to the
    /// native runtime the options were made by. A backend that <see cref="IsolatedBackend"/> loads
    /// beside another reaches its own runtime's table the same way, where calling the entry point
    /// by name could reach the other's.</para>
    /// </summary>
    /// <exception cref="NotSupportedException">The ONNX Runtime assembly this backend is bound to
    /// holds no such table, or no such entry in it.</exception>
    /// <exception cref="InvalidOperationException">The runtime refused the setting.</exception>
    private static unsafe void UseDeterministicCompute(SessionOptions options)
    {
        var api = NativeApi.Value;
        var status = ((delegate* unmanaged<IntPtr, byte, IntPtr>)api.SetDeterministicCompute)(
            options.DangerousGetHandle(), 1);
        // The handle was the options' last read, and ORT takes it as a bare IntPtr (see NewSession).
        GC.KeepAlive(options);
        if (status == IntPtr.Zero) return;
        var message = Marshal.PtrToStringUTF8(
            ((delegate* unmanaged<IntPtr, IntPtr>)api.GetErrorMessage)(status));
        ((delegate* unmanaged<IntPtr, void>)api.ReleaseStatus)(status);
        throw new InvalidOperationException($"ONNX Runtime refused deterministic compute: {message}");
    }

    /// <summary>The entries of ONNX Runtime's C API <see cref="UseDeterministicCompute"/> calls.</summary>
    private readonly record struct DeterministicComputeApi(
        IntPtr SetDeterministicCompute, IntPtr GetErrorMessage, IntPtr ReleaseStatus);

    private static readonly Lazy<DeterministicComputeApi> NativeApi = new(() =>
    {
        const System.Reflection.BindingFlags Any = System.Reflection.BindingFlags.Public
            | System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static
            | System.Reflection.BindingFlags.Instance;
        var assembly = typeof(SessionOptions).Assembly;
        var table = assembly.GetType("Microsoft.ML.OnnxRuntime.NativeMethods")?.GetField("api_", Any)?.GetValue(null);
        IntPtr Entry(string name) => table?.GetType().GetField(name, Any)?.GetValue(table) is IntPtr p && p != IntPtr.Zero
            ? p
            : throw new NotSupportedException(
                $"Deterministic compute needs ONNX Runtime's {name}, which {assembly.GetName()} does not expose.");
        return new(Entry("SetDeterministicCompute"), Entry("GetErrorMessage"), Entry("ReleaseStatus"));
    });

    /// <summary>
    /// Hands <paramref name="suppliedInitializers"/> to the session as the values of the
    /// initializers they name, and points the model's placeholder location into
    /// <paramref name="directory"/>, which this makes with the one empty file ONNX Runtime checks
    /// for. Measured on ONNX Runtime 1.30 with a 1 GiB weight in CUDA memory: the session read it
    /// where it was, growing the card's use by 12 MiB and the process's by 29 MiB where building
    /// the weight from the model grew both by the weight again.
    ///
    /// <para>ONNX Runtime takes an initializer only over memory it does not own, and this
    /// backend's values are its allocations; so each goes over as a view of the same buffer,
    /// which <paramref name="views"/> collects for the session to release after itself.</para>
    /// </summary>
    private static void Supply(
        SessionOptions options, string directory, IReadOnlyList<SuppliedInitializer> suppliedInitializers,
        List<OrtValue> views)
    {
        Directory.CreateDirectory(directory);
        File.WriteAllBytes(Path.Combine(directory, SuppliedInitializer.PlaceholderLocation), []);
        options.AddSessionConfigEntry("session.model_external_initializers_file_folder_path", directory);
        foreach (var supplied in suppliedInitializers)
        {
            if (supplied.Value is not OrtTensorValue ort)
                throw new InvalidOperationException(
                    $"Initializer '{supplied.Name}' was supplied as a {supplied.Value.GetType().Name}, which "
                    + "did not come from this backend.");
            using var memory = ort.Inner.GetTensorMemoryInfo();
            var view = OrtValue.CreateTensorValueWithData(
                memory, (TensorElementType)(int)ort.ElementType, ort.Shape, DevicePointer(ort),
                ort.Inner.GetTensorSizeInBytes());
            views.Add(view);
            options.AddInitializer(supplied.Name, view);
            // The memory info is borrowed from the value, and read through it until here.
            GC.KeepAlive(ort);
        }
    }

    private static void DeleteDirectory(string? directory)
    {
        if (directory is null) return;
        try { Directory.Delete(directory, recursive: true); }
        catch (Exception) { }
    }

    /// <summary>
    /// Applies the settings every session this backend creates runs with — the log severity
    /// and the graph-optimization level, plus the session configuration entry that
    /// <see cref="ShorokooGraphOptimization.TrainingStep"/> stands for, ONNX Runtime's memory pattern
    /// off, and its nodes run one at a time — to <paramref name="options"/>. Public so a diagnostic can build an ORT session
    /// with exactly the product's configuration plus its own (profiling, an optimized-model dump).
    ///
    /// <para><b>The memory pattern is off</b> (<c>EnableMemoryPattern</c>). With it on, ONNX Runtime
    /// traces a session's first run for each set of input shapes and from the second allocates the
    /// run's planned tensors as one block laid out by that trace, in place of a block per tensor.
    /// The block is never smaller than the most the planner's own reuse has in use at once, and
    /// measured it was larger on every graph tried, on the host and on a card, through Shorokoo's
    /// allocator: the memory-reuse scenario (two consumed 64 MiB inputs sliced, a fill through a
    /// unary chain, a concatenation through another) reached 256–272 MiB beyond its inputs against
    /// 160–192 without it; training steps of the 12-layer, MLP, convolution, encoder and LSTM
    /// families 3–24% higher; inference graphs with symbolic shapes 29–37% higher, and with static ones
    /// 7–13%. What the pattern saves is allocations — a step of the 12-layer stack makes 2 rather than
    /// 282 on the host — and an allocation and its free cost Shorokoo's allocator under 200 cycles:
    /// training steps of the linear, MLP, 12-layer and encoder families measured without the pattern
    /// within 2% of their time with it, on the host and on a card, the 4×2 linear's within the 8% its
    /// own runs vary by.</para>
    ///
    /// <para><b>Sequential execution</b> (<c>ExecutionMode.ORT_SEQUENTIAL</c>, ONNX Runtime's
    /// default, set so that nothing else can be): a session runs its nodes one at a time, on each of
    /// its streams in the order of the graph it writes out (<c>OptimizedModelFilePath</c>) — measured
    /// against the order its kernels ran in, profiled, over the training and inference graphs of
    /// several model families, on the host and on a card, built from a model and from a graph ONNX
    /// Runtime wrote out alike. A run that writes its values into the memory of the inputs it
    /// consumes is proved in that order, which parallel execution would not keep.</para>
    ///
    /// <para>For <see cref="ShorokooGraphOptimization.TrainingStep"/>:
    /// <c>optimization.disable_specified_optimizers</c> = CommonSubexpressionElimination;
    /// everything else in ORT_ENABLE_ALL — constant folding, the MatMul/Gelu/LayerNorm fusions,
    /// layout transforms — stays on.</para>
    ///
    /// <para>Deliberately absent: <c>session.set_denormal_as_zero</c>. ORT applies that entry to
    /// the constructing thread once per process (first session wins) by setting FTZ/DAZ in its
    /// MXCSR, which then flushes every later float and double operation on that thread — the
    /// caller's own managed code included, for the life of the thread. It was weighed for a
    /// measured speedup on denormal attention gradients; those gradients turned out to be an
    /// artefact of a profiling harness that fed two weight tensors identical values, so there is
    /// no speedup to set against the leak. Nothing asserts its absence: a guard on it was
    /// deleted deliberately, having needed a <c>dotnet test</c> invocation of its own to observe
    /// a process's first session.</para>
    /// </summary>
    public static void Configure(
        SessionOptions options,
        ShorokooGraphOptimization graphOptimization,
        ShorokooLogSeverity logSeverity)
    {
        options.LogSeverityLevel = (OrtLoggingLevel)(int)logSeverity;
        options.EnableMemoryPattern = false;
        // One node at a time, in the order of the graph the session writes out, on each stream:
        // what a run placing its values relies on (see OrtPlacements).
        options.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
        if (graphOptimization == ShorokooGraphOptimization.TrainingStep)
        {
            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            options.AddSessionConfigEntry("optimization.disable_specified_optimizers", "CommonSubexpressionElimination");
        }
        else
            options.GraphOptimizationLevel = (GraphOptimizationLevel)(int)graphOptimization;
    }

    /// <summary>
    /// Appends the CUDA execution provider on <paramref name="deviceId"/>, computing in
    /// <paramref name="precision"/> (<see cref="CudaProviderOptions"/>). This is what the GPU backends
    /// pass as their execution-provider step. <paramref name="deviceMemory"/> does not reach the
    /// provider: a session of this backend allocates on the card through Shorokoo's allocator, which
    /// enforces <see cref="DeviceMemorySettings.LimitBytes"/> itself (see
    /// <see cref="CachingAllocator"/>), rather than through the provider's arena.
    /// </summary>
    public static void AppendCuda(
        SessionOptions options, int deviceId, DeviceMemorySettings deviceMemory, PrecisionSettings precision)
    {
        ArgumentNullException.ThrowIfNull(deviceMemory);
        ArgumentNullException.ThrowIfNull(precision);
        // Before the provider is loaded: it imports cuBLAS and loads cuDNN by name, so they bind to
        // the pinned copies every CUDA backend of the process shares rather than to whatever PATH offers.
        CudaLibraries.Prepare();
        // OrtCUDAProviderOptions is a SafeHandle that ORT takes as a bare IntPtr, exactly like
        // the SessionOptions above, so it needs the same `using`: the options are read during
        // AppendExecutionProvider_CUDA, well after the JIT has retired the local at its .Handle
        // read, and a GC there would run the critical finalizer under the native call.
        using var cuda = new OrtCUDAProviderOptions();
        cuda.UpdateOptions(CudaProviderOptions(deviceId, precision));
        options.AppendExecutionProvider_CUDA(cuda);
    }

    /// <summary>
    /// The CUDA execution-provider options for a device and a precision, in ORT's own
    /// <c>provider_options</c> spelling. Pure, and public alongside <see cref="Configure"/> so the
    /// mapping can be read and asserted without a CUDA machine to build a session on.
    ///
    /// <para><c>use_tf32</c> is always named, and is <c>1</c> only where
    /// <see cref="PrecisionSettings.AllowTensorFloat32"/> is set. The provider's own default is
    /// <c>1</c>, which computes <c>float32</c> products, convolutions and recurrent layers in
    /// TensorFloat-32, so a session that left it out would compute in less than full precision
    /// without being asked to.</para>
    /// </summary>
    public static Dictionary<string, string> CudaProviderOptions(int deviceId, PrecisionSettings precision)
    {
        ArgumentNullException.ThrowIfNull(precision);
        return new()
        {
            ["device_id"] = deviceId.ToString(CultureInfo.InvariantCulture),
            ["use_tf32"] = precision.AllowTensorFloat32 ? "1" : "0",
        };
    }

    /// <summary>
    /// Copies a flat managed array into an ORT tensor of the given shape. Shorokoo's
    /// <c>Float16</c>/<c>BFloat16</c> are reinterpreted as ORT's own half types; every
    /// other unmanaged element type is copied through as-is.
    /// </summary>
    public IShorokooTensorValue CreateTensor<T>(T[] data, long[] shape) where T : unmanaged
    {
        if (typeof(T) == typeof(ShoFloat16))
            return Allocate(TensorElementType.Float16, MemoryMarshal.AsBytes(data.AsSpan()), shape);
        if (typeof(T) == typeof(ShoBFloat16))
            return Allocate(TensorElementType.BFloat16, MemoryMarshal.AsBytes(data.AsSpan()), shape);
        return Allocate(ElementTypeOf<T>(), MemoryMarshal.AsBytes(data.AsSpan()), shape);
    }

    /// <summary>
    /// Builds an ORT tensor of <paramref name="elementType"/> and
    /// <paramref name="shape"/> by reinterpreting a fixed-stride byte buffer.
    ///
    /// <para>In host memory, whatever device this backend computes on: Shorokoo's host allocator's,
    /// as every fixed-width host tensor of this backend is. <see cref="CreateTensorInBackendMemory"/> is the one
    /// that builds it where this backend's tensors are meant to live.</para>
    /// </summary>
    /// <exception cref="NotSupportedException">
    /// The element type has no fixed byte stride — <see cref="ShorokooTensorElementType.String"/>
    /// is variable-length, so use <see cref="CreateStringTensor"/> for it — or is not one this
    /// method handles at all.
    /// </exception>
    public IShorokooTensorValue CreateTensorFromRawBytes(
        ShorokooTensorElementType elementType,
        byte[] data,
        long[] shape)
        => Allocate(
            FixedStrideElementType(elementType, nameof(CreateTensorFromRawBytes)),
            data.AsSpan(),
            shape);

    /// <summary>
    /// A tensor of this backend holding <paramref name="data"/>, in the memory this backend's
    /// tensors live in.
    ///
    /// <para>On a host backend that is where <see cref="CreateTensorFromRawBytes"/> already builds
    /// it, so this defers to it. On a CUDA backend it is the card's own memory, where this backend's
    /// sessions read every tensor they are fed: the buffer comes from Shorokoo's allocator for that
    /// device and the bytes cross the bus once, here.</para>
    ///
    /// <para>The card's memory comes out of one allocator per device, shared by every compute
    /// context on it (<see cref="CachingAllocator"/>), so nothing here bounds it: a context's
    /// device-memory budget is kept by the context, which refuses a copy that would take it past
    /// its budget before asking for the memory at all.</para>
    /// </summary>
    /// <exception cref="ArgumentException"><paramref name="data"/> holds fewer bytes than
    /// <paramref name="shape"/> covers.</exception>
    /// <exception cref="InvalidOperationException">The CUDA runtime is not available to make the
    /// copy with.</exception>
    /// <exception cref="NotSupportedException">The element type has no fixed byte stride.</exception>
    public IShorokooTensorValue CreateTensorInBackendMemory(
        ShorokooTensorElementType elementType,
        byte[] data,
        long[] shape)
    {
        ArgumentNullException.ThrowIfNull(data);
        ArgumentNullException.ThrowIfNull(shape);
        if (_cudaDeviceId is null)
            return CreateTensorFromRawBytes(elementType, data, shape);

        var ortElementType = FixedStrideElementType(elementType, nameof(CreateTensorInBackendMemory));
        var byteCount = TensorElementLayout.ByteCount(elementType, shape);
        // A caller may hand over a buffer longer than the shape covers -- the node-definition
        // tables do -- in which case the surplus was never part of the tensor. Shorter is a
        // mistake, and on this path it would leave the tail of a device allocation unwritten.
        if (data.Length < byteCount)
            throw new ArgumentException(
                $"Supplied data of {data.Length} bytes is less than shape size {byteCount} bytes.",
                nameof(data));

        var wrapped = AllocateInBackendMemory(ortElementType, shape);
        try
        {
            // ORT's managed surface has no host-to-device copy, so this goes through the CUDA
            // runtime, to the address the value carries -- which is the one thing that may be done
            // with a device allocation here. An empty tensor has nothing to copy, and CUDA is
            // entitled to refuse the address ORT hands back for a zero-byte one.
            var copied = byteCount == 0
                || CudaInterop.CopyHostToDevice(data, DevicePointer(wrapped), byteCount);
            GC.KeepAlive(wrapped);
            if (!copied)
                throw new InvalidOperationException(
                    $"Filling this tensor ({string.Join('x', shape)}:{elementType}) in "
                    + $"{Description}'s device memory failed. The CUDA runtime is what performs the "
                    + "copy, so a machine without it cannot put a tensor on the card.");
        }
        catch
        {
            // The value owns a device allocation from the moment ORT returns it, and nothing else
            // has a reference to free it by.
            wrapped.Dispose();
            throw;
        }
        return wrapped;
    }

    /// <summary>
    /// A tensor of <paramref name="elementType"/> and <paramref name="shape"/> in the memory this
    /// backend's tensors live in, with nothing written into it: the buffer holds whatever its memory
    /// last held, and the caller fills it.
    ///
    /// <para>The same allocation <see cref="CreateTensorInBackendMemory"/> makes, and no copy —
    /// which is the point. That one starts from a managed array, so the tensor exists twice for as
    /// long as the caller holds the array it was built from; a producer writing the buffer itself
    /// never has the second copy at all (Shorokoo/Shorokoo#359).</para>
    ///
    /// <para>On a host backend the buffer is host memory and
    /// <see cref="IShorokooTensorValue.GetTensorMutableDataAsSpan{T}"/> is what fills it. On a CUDA
    /// backend it is the card's own memory, which the host cannot write through a span at all:
    /// filling it means a copy across the bus, or a run that produces into it.</para>
    /// </summary>
    /// <exception cref="ArgumentNullException"><paramref name="shape"/> is null.</exception>
    /// <exception cref="NotSupportedException">The element type has no fixed byte stride.</exception>
    public IShorokooTensorValue CreateUninitializedTensorInBackendMemory(
        ShorokooTensorElementType elementType,
        long[] shape)
    {
        ArgumentNullException.ThrowIfNull(shape);
        return AllocateInBackendMemory(
            FixedStrideElementType(elementType, nameof(CreateUninitializedTensorInBackendMemory)),
            shape);
    }

    /// <summary>
    /// The ORT element type <paramref name="elementType"/> is laid down as, refusing the ones a
    /// flat byte buffer cannot express. <paramref name="operation"/> names the caller, so a
    /// refusal says which byte-wise constructor was asked.
    /// </summary>
    /// <exception cref="NotSupportedException">
    /// The element type has no fixed byte stride — <see cref="ShorokooTensorElementType.String"/>
    /// is variable-length, so use <see cref="CreateStringTensor"/> for it — or is not one these
    /// methods handle at all.
    /// </exception>
    private static TensorElementType FixedStrideElementType(
        ShorokooTensorElementType elementType, string operation)
    {
        return elementType switch
        {
            ShorokooTensorElementType.Float or ShorokooTensorElementType.UInt8
                or ShorokooTensorElementType.Int8 or ShorokooTensorElementType.UInt16
                or ShorokooTensorElementType.Int16 or ShorokooTensorElementType.Int32
                or ShorokooTensorElementType.Int64 or ShorokooTensorElementType.Bool
                or ShorokooTensorElementType.Float16 or ShorokooTensorElementType.Double
                or ShorokooTensorElementType.UInt32 or ShorokooTensorElementType.UInt64
                or ShorokooTensorElementType.BFloat16
                => (TensorElementType)(int)elementType,
            ShorokooTensorElementType.String => throw new NotSupportedException(
                "String tensors are variable-length and not byte-stride; use CreateStringTensor instead."),
            _ => throw new NotSupportedException(
                $"{operation} does not support element type {elementType}."),
        };
    }

    /// <summary>
    /// Builds an ORT string tensor of the given shape, filling it element by element in
    /// row-major order from <paramref name="data"/>.
    /// </summary>
    public IShorokooTensorValue CreateStringTensor(IReadOnlyList<string> data, long[] shape)
    {
        var ortValue = OrtValue.CreateTensorWithEmptyStrings(OrtAllocator.DefaultInstance, shape);
        try
        {
            for (int i = 0; i < data.Count; i++)
                ortValue.StringTensorSetElementAt(data[i].AsSpan(), i);
        }
        catch
        {
            // A null element, or more elements than the shape covers, throws part-way through --
            // and nothing references the value yet, so it would sit on the finalizer queue. Every
            // sibling on this path already brackets its fill this way.
            ortValue.Dispose();
            throw;
        }
        return new OrtTensorValue(ortValue);
    }

    /// <summary>
    /// Packs already-created tensor values into an ORT sequence value, in order.
    ///
    /// <para>This <b>takes ownership</b> of <paramref name="values"/>: ORT moves them into the
    /// sequence's own member list and the sequence frees them when it is disposed, so a caller
    /// that still needs them must pass copies. Shorokoo's <c>TensorDataSequence.Create</c> does
    /// exactly that. Each value handed over refuses every read from then on, whether the sequence
    /// is built or refused.</para>
    ///
    /// <para><b>Every element must be in host memory.</b> ONNX Runtime will happily pack a tensor
    /// the execution provider left on a card into a sequence, and then cannot read it back out:
    /// its <c>GetValue</c> copies an element with a plain host <c>memcpy</c> whatever allocator it
    /// is given, so the first read of one dereferences a device address from the host and takes
    /// the process down with an access violation that nothing can catch. Such a sequence is
    /// write-only, so this refuses to make one (Shorokoo/Shorokoo#368).</para>
    /// </summary>
    /// <exception cref="InvalidOperationException">An element is in the execution provider's own
    /// memory rather than the host's. Every value handed over is disposed first, as this method's
    /// contract requires of any failure; the tensor the caller copied them from is untouched, and
    /// it is that one the message's advice is about.</exception>
    public IShorokooTensorValue CreateSequence(IReadOnlyList<IShorokooTensorValue> values)
    {
        // A sequence holds its elements' ORT values and frees them with itself, and a value standing
        // on a shared block owns none of its memory: such an element goes in as a copy of its own,
        // and the value it was made from is released here, its lease with it.
        if (values.Any(v => v is OrtTensorValue { Range: not null }))
        {
            var owned = new IShorokooTensorValue[values.Count];
            var made = 0;
            try
            {
                for (; made < values.Count; made++)
                    owned[made] = values[made] is OrtTensorValue { Range: not null, IsInDeviceMemory: false } view ? Owned(view) : values[made];
            }
            catch
            {
                // As any failure does, this lets go of everything it was handed: the copies made so
                // far in place of their values, the values not reached yet, and -- released by the
                // copy that failed -- the value it was copying.
                for (var i = 0; i < made; i++) owned[i].Dispose();
                for (var i = made + 1; i < values.Count; i++) values[i].Dispose();
                throw;
            }
            values = owned;
        }
        var inner = new List<OrtValue>(values.Count);
        try
        {
            // Before the ownership transfer below, so the message can still name the offending
            // tensor's shape and type. A pattern match rather than a cast, so a value that is not
            // this backend's still fails where it did, on the cast below. Inside the try with the
            // rest: this method's contract is that a failure disposes what it was handed, and a
            // refusal is a failure like any other -- as is a value already released, which throws
            // when it is asked where it is. What the caller keeps is the tensor these were copied
            // from, which is what it has to move.
            foreach (var v in values)
                if (v is OrtTensorValue { IsInDeviceMemory: true } onDevice)
                    throw new InvalidOperationException(
                        $"A tensor ({string.Join('x', onDevice.Shape)}:{onDevice.ElementType}) in "
                        + $"{Description.Name}'s own device memory cannot be an element of a sequence: "
                        + "ONNX Runtime can pack it into one but reads an element back with a host "
                        + "copy, so nothing could ever read it again. Bring the tensor into host "
                        + "memory first -- CopyTensorToHost does that, and TensorData.ToHost() "
                        + "is the same move on a tensor.");

            // This method documents itself as taking ownership, so an element that is not this
            // backend's value -- one from another runtime, or a foreign implementation -- throws on
            // the cast with the earlier elements already unwrapped and the caller already committed
            // to having given them up.
            foreach (var v in values) inner.Add(((OrtTensorValue)v).Inner);
            var sequence = OrtValue.CreateSequence(inner);
            // The sequence holds the values now and frees them with itself, so each wrapper is
            // marked released without freeing what it held. Left live, a wrapper would go on
            // handing ORT a value the sequence owns, and once the sequence is gone a freed one.
            foreach (var v in values) ((OrtTensorValue)v).HandedOver();
            return new OrtTensorValue(sequence);
        }
        catch
        {
            // ORT hands the values back on failure -- it empties the list only on success -- so
            // they are freed here, every one this was handed rather than those unwrapped before the
            // failure, and through their wrappers. A value freed behind a wrapper that still reports
            // itself live reaches ORT on its next read as a handle that is gone: an access
            // violation, not an exception.
            foreach (var v in values) v.Dispose();
            throw;
        }
    }

    /// <summary>A host copy of <paramref name="view"/>, a value standing on a shared block, which is
    /// released.</summary>
    private OrtTensorValue Owned(OrtTensorValue view)
    {
        try
        {
            return Allocate((TensorElementType)(int)view.ElementType, view.GetTensorDataAsSpan<byte>(), view.Shape);
        }
        finally
        {
            view.Dispose();
        }
    }

    /// <summary>An ORT value over <paramref name="bytes"/> bytes of <paramref name="owner"/>'s memory
    /// from byte <paramref name="offset"/>, owning none of it and holding no lease: for a run to
    /// write a value into, that nothing outlives the run.</summary>
    internal static OrtValue Over(OrtTensorValue owner, long offset, ShorokooTensorElementType elementType, long[] shape, long bytes)
    {
        var inner = owner.Inner;
        using var memory = inner.GetTensorMemoryInfo();
        var view = OrtValue.CreateTensorValueWithData(
            memory, (TensorElementType)(int)elementType, shape, (IntPtr)((long)AddressOf(inner) + offset), bytes);
        GC.KeepAlive(owner);
        return view;
    }

    /// <summary>
    /// A value over <paramref name="bytes"/> bytes of <paramref name="owner"/>'s memory from byte
    /// <paramref name="offset"/>, of <paramref name="elementType"/> and <paramref name="shape"/>,
    /// standing on <paramref name="block"/> — the block <paramref name="owner"/> is, or stands on —
    /// at <paramref name="offsetInBlock"/>, with a lease of its own on it. The memory is where the
    /// owner's is, on the host or a card, and the value owns none of it.
    /// </summary>
    internal static OrtTensorValue View(
        OrtTensorValue owner, long offset, ShorokooTensorElementType elementType, long[] shape, long bytes,
        SharedBlock block, long offsetInBlock)
    {
        block.Lease(offsetInBlock, bytes);
        try
        {
            var inner = owner.Inner;
            using var memory = inner.GetTensorMemoryInfo();
            var view = OrtValue.CreateTensorValueWithData(
                memory, (TensorElementType)(int)elementType, shape, (IntPtr)((long)AddressOf(inner) + offset), bytes);
            GC.KeepAlive(owner);
            return new OrtTensorValue(view, new BlockRange(block, offsetInBlock, bytes));
        }
        catch
        {
            block.Release(offsetInBlock, bytes);
            throw;
        }
    }

    /// <summary>
    /// A block over <paramref name="owner"/>'s memory, of its <paramref name="bytes"/>, for values to
    /// stand on, which <paramref name="letGo"/> lets go of with the last of them. Where that memory is
    /// a block Shorokoo's allocator carved from one of its arenas (<see cref="RangesGoBack(OrtTensorValue)"/>)
    /// — on the host one of 64 KiB or more, a run's or a tensor's made from host data; on a card any
    /// block, where the driver offers virtual memory management — a part of it no value stands on any
    /// more goes back to that allocator (<see cref="CachingAllocator.ReleaseRange"/>) while the rest is
    /// still in use.
    /// </summary>
    internal static SharedBlock BlockOver(OrtTensorValue owner, long bytes, Action letGo)
    {
        if (RangesGoBack(owner, out var allocator, out var address))
            return new SharedBlock(bytes, letGo, (offset, length, toTheEnd) => allocator.ReleaseRange(address, offset, length, toTheEnd));
        return new SharedBlock(bytes, letGo);
    }

    /// <summary>
    /// Whether a part of <paramref name="value"/>'s memory no value stands on any more goes back
    /// while the rest is in use: for a value standing on a block, whether that block gives back;
    /// for one in memory of its own, whether that memory is a block Shorokoo's allocator carved
    /// from an account's arena — where <paramref name="allocator"/> takes the part back at
    /// <paramref name="address"/>.
    /// </summary>
    internal static bool RangesGoBack(OrtTensorValue value, out CachingAllocator allocator, out IntPtr address)
    {
        allocator = null!;
        address = IntPtr.Zero;
        if (value.Range is { } range) return range.Block.GivesBack;
        if (value.CudaDevice is { } device) allocator = CachingAllocator.ForCard(device);
        else if (value.InHostMemory) allocator = CachingAllocator.ForHost();
        else return false;
        address = AddressOf(value.Inner);
        return allocator.ReleasesRanges(address);
    }

    /// <summary>Whether a part of <paramref name="value"/>'s memory no value stands on any more goes
    /// back while the rest is in use (<see cref="RangesGoBack(OrtTensorValue, out CachingAllocator, out IntPtr)"/>),
    /// asked once of each value.</summary>
    internal static bool RangesGoBack(OrtTensorValue value) => value.PartsGoBack ??= RangesGoBack(value, out _, out _);

    /// <summary>
    /// <paramref name="value"/>'s contents as host bytes, including when it is in the execution
    /// provider's own memory and so cannot be read here at all.
    ///
    /// <para>ONNX Runtime's managed surface has no device-to-host copy to call here: a value it
    /// left on the card hands out a pointer and no way to read it. So the copy is made through the
    /// CUDA runtime directly, from the address the value carries — the same address the runtime
    /// would use, since there is only one allocation and this backend is the one that made it.</para>
    /// </summary>
    public byte[] CopyTensorToHost(IShorokooTensorValue value)
    {
        ArgumentNullException.ThrowIfNull(value);

        if (value.IsHostAccessible)
        {
            var hostBytes = value.GetTensorDataAsSpan<byte>().ToArray();
            GC.KeepAlive(value);
            return hostBytes;
        }

        if (value.ValueType != ShorokooOnnxValueType.Tensor)
            throw new InvalidOperationException(
                $"Only a tensor can be read back from device memory; this is a {value.ValueType}.");

        var destination = new byte[TensorElementLayout.ByteCount(value.ElementType, value.Shape)];
        var copied = CudaInterop.CopyDeviceToHost(DevicePointer(value), destination);
        GC.KeepAlive(value);
        if (!copied)
            throw new InvalidOperationException(
                $"Reading this tensor ({string.Join('x', value.Shape)}:{value.ElementType}) back "
                + $"from {Description}'s device memory failed. The CUDA runtime is what performs "
                + "the copy, so a machine without it cannot bring a device-resident value home.");
        return destination;
    }

    /// <summary>
    /// A range of <paramref name="value"/>'s contents, copied into <paramref name="destination"/>
    /// — through the CUDA runtime from the device address plus <paramref name="byteOffset"/> where
    /// the value is on the card, so a tensor streams off it through one reused buffer instead of
    /// arriving whole. False only where the CUDA runtime is absent, for the caller to fall back to
    /// <see cref="CopyTensorToHost"/>, which then reports that.
    /// </summary>
    public bool TryCopyTensorRangeToHost(IShorokooTensorValue value, long byteOffset, Span<byte> destination)
    {
        ArgumentNullException.ThrowIfNull(value);

        if (value is OrtTensorValue { IsHostAccessible: true } host)
        {
            HostPiece(host, byteOffset, destination.Length).CopyTo(destination);
            GC.KeepAlive(host);
            return true;
        }

        if (value.IsHostAccessible)
        {
            var source = value.GetTensorDataAsSpan<byte>();
            if (byteOffset < 0 || byteOffset > source.Length - destination.Length)
                throw RangeOutside(byteOffset, destination.Length, source.Length);
            source.Slice((int)byteOffset, destination.Length).CopyTo(destination);
            GC.KeepAlive(value);
            return true;
        }

        if (value.ValueType != ShorokooOnnxValueType.Tensor)
            throw new InvalidOperationException(
                $"Only a tensor can be read back from device memory; this is a {value.ValueType}.");

        var length = TensorElementLayout.ByteLength(value.ElementType, value.Shape);
        if (byteOffset < 0 || byteOffset > length - destination.Length)
            throw RangeOutside(byteOffset, destination.Length, length);
        var status = CudaInterop.CopyDeviceToHost(DevicePointer(value) + (nint)byteOffset, destination);
        GC.KeepAlive(value);
        if (status is null) return false;
        if (status != 0)
            throw new InvalidOperationException(
                $"Copying {destination.Length} bytes at offset {byteOffset} of this tensor "
                + $"({string.Join('x', value.Shape)}:{value.ElementType}) out of {Description}'s device "
                + $"memory failed with CUDA error {status}.");
        return true;
    }

    /// <summary>
    /// <paramref name="source"/> copied into <paramref name="value"/>'s contents at
    /// <paramref name="byteOffset"/> — through the CUDA runtime to the device address plus the
    /// offset where the value is on the card, so a tensor streams onto it through one reused buffer
    /// instead of arriving whole. False only where the CUDA runtime is absent, for the caller to fall
    /// back to <see cref="CreateTensorInBackendMemory"/>.
    /// </summary>
    public bool TryCopyHostToTensorRange(IShorokooTensorValue value, long byteOffset, ReadOnlySpan<byte> source)
    {
        ArgumentNullException.ThrowIfNull(value);

        if (value is OrtTensorValue { IsHostAccessible: true } host)
        {
            source.CopyTo(HostPiece(host, byteOffset, source.Length));
            GC.KeepAlive(host);
            return true;
        }

        if (value.IsHostAccessible)
        {
            var destination = value.GetTensorMutableDataAsSpan<byte>();
            if (byteOffset < 0 || byteOffset > destination.Length - source.Length)
                throw RangeOutside(byteOffset, source.Length, destination.Length);
            source.CopyTo(destination.Slice((int)byteOffset, source.Length));
            GC.KeepAlive(value);
            return true;
        }

        if (value.ValueType != ShorokooOnnxValueType.Tensor)
            throw new InvalidOperationException(
                $"Only a tensor can be written into device memory; this is a {value.ValueType}.");

        var length = TensorElementLayout.ByteLength(value.ElementType, value.Shape);
        if (byteOffset < 0 || byteOffset > length - source.Length)
            throw RangeOutside(byteOffset, source.Length, length);
        var status = CudaInterop.CopyHostToDevice(source, DevicePointer(value) + (nint)byteOffset);
        GC.KeepAlive(value);
        if (status is null) return false;
        if (status != 0)
            throw new InvalidOperationException(
                $"Copying {source.Length} bytes into this tensor ({string.Join('x', value.Shape)}:"
                + $"{value.ElementType}) at offset {byteOffset} in {Description}'s device memory failed "
                + $"with CUDA error {status}.");
        return true;
    }

    /// <summary>
    /// The <paramref name="count"/> bytes at <paramref name="byteOffset"/> into a host-accessible
    /// value's buffer, addressed from where the buffer starts rather than sliced out of a span over
    /// all of it: a span's length is an int, so ORT cannot make one over a buffer above 2 GiB, and a
    /// piece of such a buffer is exactly what a load streams through. The caller keeps
    /// <paramref name="value"/> alive for as long as it uses the piece.
    /// </summary>
    private static unsafe Span<byte> HostPiece(OrtTensorValue value, long byteOffset, int count)
    {
        var length = TensorElementLayout.ByteLength(value.ElementType, value.Shape);
        if (byteOffset < 0 || byteOffset > length - count)
            throw RangeOutside(byteOffset, count, length);
        return new Span<byte>((byte*)(DevicePointer(value) + (nint)byteOffset), count);
    }

    private static ArgumentOutOfRangeException RangeOutside(long byteOffset, int count, long length)
        => new(nameof(byteOffset), $"{count} bytes from offset {byteOffset} run past the value's {length}.");

    /// <summary>
    /// The address the value's buffer is at, without reading it — which for a device allocation is
    /// the one thing that may be done with it here.
    ///
    /// <para>Through the ORT value rather than through <see cref="IShorokooTensorValue"/>, whose
    /// span accessors refuse a value the provider kept precisely so that nobody dereferences a
    /// device address as a host one. Taking the address is not dereferencing it, and this is the
    /// backend that made the allocation.</para>
    /// </summary>
    private static IntPtr DevicePointer(IShorokooTensorValue value)
    {
        if (value is not OrtTensorValue ort)
            throw new InvalidOperationException(
                $"A {value.GetType().Name} did not come from this backend, so its device memory "
                + "cannot be read here.");
        var address = AddressOf(ort.Inner);
        // The value's last read here, so without this the JIT may retire the local and a collection
        // on any thread free the allocation before the address is used. The caller keeps it alive
        // across the copy itself (Shorokoo/Shorokoo#178).
        GC.KeepAlive(ort);
        return address;
    }

    /// <summary>
    /// The address of <paramref name="value"/>'s buffer, without reading it: <see cref="DevicePointer"/>
    /// for an ORT value. The caller keeps <paramref name="value"/> alive for as long as it uses the
    /// address.
    /// </summary>
    internal static unsafe IntPtr AddressOf(OrtValue value)
    {
        // A Span's length is an int, so ORT cannot make one over an allocation larger than 2 GiB;
        // above that the address comes from its C API directly.
        IntPtr address;
        if (value.GetTensorSizeInBytes() <= int.MaxValue)
        {
            var span = value.GetTensorMutableRawData();
            fixed (byte* p = span) address = (IntPtr)p;
        }
        else
        {
            address = OrtTensorAddress.Read(value) ?? throw new InvalidOperationException(
                "The address of a tensor larger than 2 GiB could not be read from ONNX Runtime.");
        }
        GC.KeepAlive(value);
        return address;
    }

    /// <summary>
    /// An ORT-allocated buffer of this element type and shape in the memory this backend's tensors
    /// live in, with nothing written into it: host memory on a host backend, the card's own on a
    /// CUDA one.
    /// </summary>
    private OrtTensorValue AllocateInBackendMemory(TensorElementType elementType, long[] shape)
    {
        // The one place the host-or-card decision is made for a tensor placed in this backend's
        // memory, so the two constructors that build there cannot come to differ on it. A CUDA
        // backend allocating from the default allocator would hand back host memory wearing the
        // card's name, which the execution provider then copies over on every run.
        if (_cudaDeviceId is not { } deviceId)
            return new(OrtValue.CreateAllocatedTensorValue(RuntimeAllocator.ForHost().Managed, elementType, shape));
        return new(OrtValue.CreateAllocatedTensorValue(RuntimeAllocator.ForCard(deviceId).Managed, elementType, shape));
    }

    /// <summary>
    /// Builds an ORT tensor on a buffer of Shorokoo's host allocator, the memory a session's own
    /// host tensors come from, and copies <paramref name="bytes"/> into it: a block whose part no
    /// tensor stands on any more goes back on its own (<see cref="CachingAllocator.ReleaseRange"/>).
    /// Measured against ONNX Runtime's default allocator on Windows, making and deleting a tensor
    /// took as long either way up to 4 MiB, and up to a fifth longer at 16 and 64 MiB, where both take
    /// fresh pages from the system each time; a tensor of a mebibyte or more goes back to the system
    /// as it goes either way (<see cref="CachingAllocator.Account.KeepsLargeBlocks"/>).
    ///
    /// <para>The obvious alternative — <c>OrtValue.CreateTensorValueFromMemory</c> over a managed
    /// array — is why this is a copy. That API pins the array for the value's lifetime and releases
    /// the pin only from <c>Dispose</c>: the release sits behind the <c>disposing</c> guard, so an
    /// <c>OrtValue</c> reclaimed by its finalizer never runs it. A tensor value is as often collected
    /// as disposed, so a tensor built that way would pin its bytes for the life of the process — a
    /// training loop feeding a fresh batch each step would leak one batch per step, permanently, and
    /// no collection could get it back. An ORT-allocated buffer is released by the value's
    /// finalizer along with the value, so it behaves like every other tensor the runtime hands
    /// back — which is why <see cref="CreateTensorInBackendMemory"/> allocates device memory the
    /// same way rather than calling <c>cudaMalloc</c> and owning the result itself.</para>
    /// </summary>
    private static OrtTensorValue Allocate(TensorElementType elementType, ReadOnlySpan<byte> bytes, long[] shape)
    {
        var wrapped = new OrtTensorValue(
            OrtValue.CreateAllocatedTensorValue(RuntimeAllocator.ForHost().Managed, elementType, shape));
        try
        {
            var destination = wrapped.Inner.GetTensorMutableRawData();
            if (bytes.Length < destination.Length)
                throw new ArgumentException(
                    $"Supplied data of {bytes.Length} bytes is less than shape size {destination.Length} bytes.",
                    nameof(bytes));
            // A caller may hand over a buffer longer than the shape covers — the node-definition
            // tables do — in which case the surplus is not part of the tensor and is not read.
            bytes.Slice(0, destination.Length).CopyTo(destination);
            // The destination span is a bare pointer into the value's buffer: reading the value
            // for it is the value's last read, after which the JIT may retire the local and a
            // collection on any thread can run the value's finalizer and free the buffer out from
            // under the copy. Scope does not root anything (Shorokoo/Shorokoo#178); this does.
            GC.KeepAlive(wrapped);
        }
        catch
        {
            wrapped.Dispose();
            throw;
        }
        return wrapped;
    }

    /// <summary>The ORT element type of the storage primitive <typeparamref name="T"/>.</summary>
    private static TensorElementType ElementTypeOf<T>() where T : unmanaged
    {
        if (typeof(T) == typeof(float)) return TensorElementType.Float;
        if (typeof(T) == typeof(double)) return TensorElementType.Double;
        if (typeof(T) == typeof(bool)) return TensorElementType.Bool;
        if (typeof(T) == typeof(sbyte)) return TensorElementType.Int8;
        if (typeof(T) == typeof(byte)) return TensorElementType.UInt8;
        if (typeof(T) == typeof(short)) return TensorElementType.Int16;
        if (typeof(T) == typeof(ushort)) return TensorElementType.UInt16;
        if (typeof(T) == typeof(int)) return TensorElementType.Int32;
        if (typeof(T) == typeof(uint)) return TensorElementType.UInt32;
        if (typeof(T) == typeof(long)) return TensorElementType.Int64;
        if (typeof(T) == typeof(ulong)) return TensorElementType.UInt64;
        if (typeof(T) == typeof(OrtFloat16)) return TensorElementType.Float16;
        if (typeof(T) == typeof(OrtBFloat16)) return TensorElementType.BFloat16;
        throw new NotSupportedException($"CreateTensor does not support element type {typeof(T).Name}.");
    }
}
