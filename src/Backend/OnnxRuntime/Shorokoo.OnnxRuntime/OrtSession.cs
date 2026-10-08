using Microsoft.ML.OnnxRuntime;
using Shorokoo.Core.Backends;
using TensorElementType = Microsoft.ML.OnnxRuntime.Tensors.TensorElementType;

namespace Shorokoo.OnnxRuntime;

internal sealed class OrtSession : IShorokooSession
{
    private readonly InferenceSession _session;
    private readonly int? _cudaDeviceId;
    private readonly OrtBackend _backend;

    // What this session allocates, through Shorokoo's allocators (see CachingAllocator), is charged
    // to these: on the host, and on its card for a CUDA session. Every block its construction and its
    // runs take -- weights, intermediates, the outputs ONNX Runtime writes -- is one of its own in one
    // of them, and the card's carries the limit a device-memory budget leaves the session's runs.
    private readonly CachingAllocator.Account _hostAccount;
    private readonly CachingAllocator.Account? _cardAccount;

    // Closes those accounts once the session is gone -- disposed, or collected without -- so what
    // they keep cached goes where the allocator can hand it back. Null for a session charging
    // another's accounts, which that one closes: a session carrying out another's placements.
    private readonly AccountsCloser? _accounts;

    private sealed class AccountsCloser(CachingAllocator.Account host, CachingAllocator.Account? card)
    {
        internal void Close()
        {
            CloseBoth();
            GC.SuppressFinalize(this);
        }

        ~AccountsCloser() => CloseBoth();

        private void CloseBoth()
        {
            host.Allocator.Close(host);
            card?.Allocator.Close(card);
        }
    }

    // Where the session's runtime computes its outputs, probed on first ask: a session is a common
    // object here -- parameter initialization and every eager Eval build one -- and most are never
    // asked.
    private readonly Lazy<SessionOutputPlacement> _placement;

    // The memory a run of a CUDA session reads a tensor in and leaves one in -- its card's, in the
    // form ONNX Runtime binds an output to -- or null on any other session, whose run memory is the
    // host's. Held in a field, not a local: OrtMemoryInfo owns a native handle and the binding takes
    // it as a bare pointer.
    private readonly OrtMemoryInfo? _cardMemory;

    // The pinned host arena a CUDA session stages its crossings through, which ONNX Runtime keeps
    // for itself, and the memory info naming it -- both in fields for the reason the one above is.
    // Built on first ask, because no session needs them unless something asked for its figures; a
    // session with no CUDA provider has no such arena at all.
    private readonly Lazy<OrtAllocator?> _pinnedAllocator;
    private OrtMemoryInfo? _ownedPinnedMemoryInfo;

    // The folder ORT writes this session's profile into, or null when it was not built to record
    // one -- which is the default. Deleted with the session.
    private readonly string? _profileDirectory;

    /// <summary>The views of supplied initializers this session was built over (see
    /// <c>OrtBackend.Supply</c>), which ONNX Runtime requires to outlive it: released after it.</summary>
    internal IReadOnlyList<OrtValue> SuppliedViews { get; init; } = [];

    /// <summary>The copies of its model's weights in the backend's memory this session reads in
    /// place of the model's, which it owns, and its variants read too.</summary>
    internal IReadOnlyList<OrtTensorValue> SharedWeights { get; init; } = [];

    /// <summary>Where this session's runs that consume inputs place their values in the memory of
    /// what they consume (see <see cref="OrtPlacements"/>), or null for a session that places
    /// nothing. Set once, as the backend builds the session.</summary>
    internal OrtPlacements? Placements { get; set; }

    /// <summary>The route ONNX Runtime names this session's logger by: what it logs goes to the
    /// settings the session was built with, as does what a run logs whose
    /// <see cref="RunSettings.Log"/> is the same. A session placing another's values logs through
    /// that one's.</summary>
    internal required RuntimeLogRoutes.Route Log { get; init; }

    // Whether this session closes Log when it is disposed: the session a backend built, and not one
    // it places values through.
    private bool _ownsLogRoute;

    /// <summary>Makes this session the one that closes <see cref="Log"/>.</summary>
    internal void OwnLogRoute() => _ownsLogRoute = true;

    void IShorokooSession.StopPlacing()
    {
        Placements?.Dispose();
        Placements = null;
    }

    private readonly object _profileGate = new();
    private NodePlacement? _nodePlacement;
    private bool _profilingEnded;

    // The outputs this session may write into the memory of the input each is paired with, on a run
    // that consumed that input: the pairs its own graph proved (see OrtBackend.CreateSession), each
    // with the shape and element type ORT inferred for the output. A pair whose output shape ORT
    // could not settle when the session was built is not here, since nothing could be bound to it
    // without knowing the output would fit -- a symbolic dim is only known once the run is under
    // way, and a buffer bound to the wrong shape fails the run. Keyed by output: the proof pairs each
    // output with one input at most.
    private readonly Dictionary<string, AliasSlot> _aliases;

    /// <summary>An output this session may write into an input's memory, and what the input's value
    /// has to be for that to happen.</summary>
    private sealed record AliasSlot(string Output, string Input, long[] Shape, TensorElementType ElementType);

    /// <summary>
    /// A pair the graph ONNX Runtime runs proves, and the shape that graph states for the output —
    /// null where it states none, or leaves a dimension open.
    /// </summary>
    internal readonly record struct ProvedAlias(OutputAlias Alias, long[]? StatedShape);

    public IReadOnlyList<OutputAlias> BindableAliases { get; }

    public OrtSession(
        InferenceSession session,
        int? cudaDeviceId,
        OrtBackend backend,
        string? profileDirectory,
        IReadOnlyList<ProvedAlias> outputAliases,
        CachingAllocator.Account hostAccount,
        CachingAllocator.Account? cardAccount,
        bool ownsAccounts = true)
    {
        _session = session;
        _cudaDeviceId = cudaDeviceId;
        _backend = backend;
        _profileDirectory = profileDirectory;
        _hostAccount = hostAccount;
        _cardAccount = cardAccount;
        _accounts = ownsAccounts ? new AccountsCloser(hostAccount, cardAccount) : null;
        _aliases = Slots(session, outputAliases);
        BindableAliases = [.. _aliases.Values.Select(slot => new OutputAlias(slot.Output, slot.Input))];
        // Nothing to clean up, so nothing to finalize -- every untraced session would otherwise
        // join the finalization queue to run an early return.
        if (profileDirectory is null) GC.SuppressFinalize(this);
        _placement = new Lazy<SessionOutputPlacement>(() => DiscoverPlacement(session));
        _hostMemoryNames = new Lazy<(IReadOnlySet<string> Inputs, IReadOnlySet<string> Outputs)?>(ReadHostMemoryNames);
        // The info names the card's memory as a binding wants it: ONNX Runtime binds an output to
        // the device an info names, and allocates it there from the session's allocator for it.
        _cardMemory = cudaDeviceId is { } device ? CudaMemoryInfo(device) : null;
        _pinnedAllocator = new Lazy<OrtAllocator?>(CreatePinnedAllocator);
    }

    /// <summary>
    /// The pairs of <paramref name="outputAliases"/> this session can bind: the output is one of its
    /// tensors, and not a string one; the input is one of its inputs; and ORT settled the output's
    /// shape in full when it built the session.
    ///
    /// <para>Settled means the graph ORT wrote out states the shape, and the session reports the
    /// same one. The session alone cannot say: ORT reports an output it has no shape for as having
    /// no dimensions, which is what a scalar has too, so a scalar consumed value would pass for an
    /// output the run then makes <c>[1]</c>, and the run fail on the binding. The graph tells the
    /// two apart — a scalar's shape is there, and empty.</para>
    /// </summary>
    private static Dictionary<string, AliasSlot> Slots(InferenceSession session, IReadOnlyList<ProvedAlias> outputAliases)
    {
        var slots = new Dictionary<string, AliasSlot>(outputAliases.Count, StringComparer.Ordinal);
        if (outputAliases.Count == 0) return slots;
        var inputs = new HashSet<string>(session.InputNames, StringComparer.Ordinal);
        var outputs = session.OutputMetadata;
        foreach (var (alias, shape) in outputAliases)
        {
            if (shape is null || !inputs.Contains(alias.Input) || !outputs.TryGetValue(alias.Output, out var output)) continue;
            // A string tensor's elements are objects rather than bytes in a buffer of its own, and
            // nothing here proves writing one over another sound, so it is never bound.
            if (!output.IsTensor || output.ElementDataType == TensorElementType.String) continue;
            if (!output.Dimensions.Select(d => (long)d).SequenceEqual(shape)) continue;
            slots.TryAdd(alias.Output, new AliasSlot(alias.Output, alias.Input, shape, output.ElementDataType));
        }
        return slots;
    }

    /// <summary>
    /// The ORT value behind <paramref name="value"/>, fed as input <paramref name="name"/>: one of
    /// this runtime's, in the memory this session's runs read it in — refused otherwise, before the
    /// run starts. Nothing here moves a value; the framework places every input in the run memory
    /// of this session's backend before it hands it over (<see cref="IShorokooBackend.RunMemoryOf"/>).
    ///
    /// <para>An <see cref="OrtTensorValue"/> <i>of this assembly</i> is a value of this runtime. That
    /// test is exact rather than approximate: a backend loaded by <see cref="IsolatedBackend"/> gets
    /// a private copy of this assembly, so its <c>OrtTensorValue</c> is a different type from this
    /// one — and its handles point into a native runtime this session knows nothing about, which is
    /// precisely when feeding them would be a wild pointer rather than a mistake ORT could
    /// catch.</para>
    /// </summary>
    /// <exception cref="InvalidOperationException">The value is not this runtime's, or not in this
    /// session's run memory.</exception>
    private OrtValue Fed(string name, IShorokooTensorValue value)
    {
        if (value is OrtTensorValue own && InRunMemory(own)) return own.Inner;
        throw new InvalidOperationException(
            $"Input '{name}' was handed to a session of {_backend.Description} as {Describe(value)}, "
            + $"outside the memory its runs read it in ({RunMemoryName(value)}). A session takes only "
            + "values its backend's runs read where they are, and moves none: place the input there "
            + "first, through the backend's own moves (IShorokooBackend.CreateTensorInBackendMemory).");
    }

    /// <summary>
    /// Whether <paramref name="value"/> is where a run of this session reads it: a tensor on this
    /// session's card, on a CUDA session, unless it is a string tensor; any other tensor in host
    /// memory; and a sequence, which this runtime builds in host memory only
    /// (<see cref="OrtBackend.CreateSequence"/>).
    /// </summary>
    private bool InRunMemory(OrtTensorValue value)
    {
        if (value.ValueType == ShorokooOnnxValueType.Sequence) return true;
        if (value.ValueType != ShorokooOnnxValueType.Tensor) return false;
        if (_cudaDeviceId is { } device && value.ElementType != ShorokooTensorElementType.String)
            return value.IsOnCudaDevice(device);
        return !value.IsInDeviceMemory;
    }

    /// <summary>Where this session's runs read a value like <paramref name="value"/>, in the words a
    /// refusal uses.</summary>
    private string RunMemoryName(IShorokooTensorValue value)
        => _cudaDeviceId is { } device && value.ValueType == ShorokooOnnxValueType.Tensor
           && value.ElementType != ShorokooTensorElementType.String
            ? $"CUDA device {device}'s memory"
            : "host memory";

    /// <summary>What a refusal calls <paramref name="value"/>.</summary>
    private static string Describe(IShorokooTensorValue value)
        => value is OrtTensorValue own
            ? own.ValueType == ShorokooOnnxValueType.Tensor
                ? $"a {string.Join('x', own.Shape)}:{own.ElementType} tensor in {own.MemoryName}"
                : $"a {own.ValueType}"
            : $"a {value.GetType().Name}, which is not a value of this runtime";

    public IReadOnlyList<string> InputNames => _session.InputNames;
    public IReadOnlyList<string> OutputNames => _session.OutputNames;

    /// <summary>
    /// The overload below, for a caller that does not ask which outputs went into consumed memory.
    /// A session built with pairs writes them there all the same. Every run the framework makes
    /// calls the overload below, so this one serves a caller outside it.
    /// </summary>
    public IReadOnlyList<IShorokooTensorValue> RunConsuming(
        IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
        IReadOnlyCollection<IShorokooTensorValue> consumed,
        IReadOnlyList<string> outputNames,
        RunSettings runSettings)
        => RunConsuming(inputs, consumed, outputNames, runSettings, out _);

    /// <summary>
    /// Runs the session with <paramref name="consumed"/> handed over, writing each output this
    /// session was built to alias into the memory of the consumed value its input was fed,
    /// wherever that can be done (see <see cref="OutputsIntoConsumed"/>), and saying which it did in
    /// <paramref name="aliasedInputs"/>; and, where its placements adopted a plan for the run's
    /// signature, writing values into ranges of the other consumed values (see
    /// <see cref="OrtPlacements"/>).
    ///
    /// <para>Each consumed value is this backend's from here, on every path, and is released
    /// through it in the <c>finally</c> below — after the native run, which ONNX Runtime does not
    /// let go of its inputs before, and before this returns the outputs or rethrows a failure —
    /// except one an output was placed in: that one became a block (<see cref="SharedBlock"/>)
    /// the outputs placed in it stand on, each over its own range, and the block releases it with
    /// the last of them. Nothing else the caller holds points into a consumed value: an aliased
    /// output is a value of its own over the same memory. ONNX Runtime keeps every input until the
    /// run ends — its memory planner gives each feed an extra use so a caller can read it after
    /// <c>Run</c> returns — so there is no earlier point at which a consumed input could go.</para>
    ///
    /// <para>An aliased output is bound to the consumed value itself, so the node that produces it
    /// writes straight into that memory rather than into a block of the arena — which is the whole
    /// saving, since ONNX Runtime holds the value until the run ends either way. What comes back is
    /// a value of its own over the same memory: ORT counts the references to a buffer, so releasing
    /// the consumed value below, as every run does, leaves the output whole.</para>
    ///
    /// <para>Writing into a consumed value is only safe because it is memory of its own — nothing
    /// else reads it — and the values runs consume are mostly outputs of earlier runs. So this
    /// rests on ONNX Runtime never handing out a session's own memory as an output, and it does
    /// not, measured: an output it folded to a constant, and an initializer or a view of
    /// one handed out as an output, each comes back in a buffer of its own on every run, and a
    /// write into one reaches neither the session nor any other run's output.
    /// <c>ComputeContextLifetimeCoverageTests.TestAnOutputTheRuntimeFoldsToAConstantIsMemoryOfItsOwnThatAWriteDoesNotCarryIntoAnotherRun</c>
    /// fails if the runtime shares one.</para>
    /// </summary>
    public IReadOnlyList<IShorokooTensorValue> RunConsuming(
        IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
        IReadOnlyCollection<IShorokooTensorValue> consumed,
        IReadOnlyList<string> outputNames,
        RunSettings runSettings,
        out IReadOnlyList<string?> aliasedInputs)
    {
        ArgumentNullException.ThrowIfNull(consumed);
        aliasedInputs = [];
        HashSet<IShorokooTensorValue>? kept = null;
        try
        {
            ArgumentNullException.ThrowIfNull(runSettings);
            var into = OutputsIntoConsumed(inputs, consumed, outputNames);
            IReadOnlyList<IShorokooTensorValue> results;
            var intoInputs = into?.Values.Select(t => t.Input).ToHashSet(StringComparer.Ordinal) ?? [];
            var blocks = Placements is null || consumed.Count == 0 ? null : OrtPlacements.Blocks(inputs, consumed, intoInputs);
            if (Placements is { } placements && blocks is not null && placements.EntryFor(inputs, blocks, outputNames, intoInputs) is { } entry)
                results = placements.Run(
                    this, entry, inputs, blocks, outputNames,
                    () => RunBound(inputs, outputNames, into, runSettings),
                    (variant, placed) => variant.RunBound(inputs, outputNames, into, runSettings, placed),
                    // An output written into its own input is not placed in another.
                    (IReadOnlyCollection<string>?)into?.Keys ?? [],
                    runSettings.CancellationToken,
                    out kept);
            else
                results = RunBound(inputs, outputNames, into, runSettings);
            if (into is not null)
            {
                var aliased = new string?[outputNames.Count];
                for (int i = 0; i < outputNames.Count; i++)
                    if (into.TryGetValue(outputNames[i], out var target)) aliased[i] = target.Input;
                aliasedInputs = aliased;
            }
            return results;
        }
        finally
        {
            // Through the backend, which is the one release path for memory it allocated. Its
            // release is a disposal, which does not throw, so none of them can be skipped. A value an
            // output was placed in is the block's to release, with the last output standing on it.
            foreach (var value in consumed)
                if (kept is null || !kept.Contains(value)) ((IShorokooBackend)_backend).Release(value);
            // Last, so that what the run let go of is handed back with the rest: what it consumed, and
            // what of a block no placed output stands on.
            HandBackIfAsked(runSettings);
        }
    }

    /// <summary>A consumed value an output is written into, and the input it was fed as.</summary>
    private readonly record struct AliasTarget(string Input, OrtTensorValue Value);

    /// <summary>
    /// The outputs this run writes into the memory of a value it consumed, by output name, or null
    /// when there are none. A pair this session was built with (<see cref="_aliases"/>) is bound
    /// only where every one of these holds, and the output is produced as usual otherwise:
    /// <list type="bullet">
    /// <item>the run consumed the value its input was fed — memory it was only lent is the
    /// caller's, and writing into it would change a tensor the caller still reads;</item>
    /// <item>no other input is fed the same value, since the proof was about this input alone;</item>
    /// <item>it is a value of this runtime, of the output's element type and shape.</item>
    /// </list>
    /// The value and the output are in the same memory by construction: the run memory, where every
    /// input is fed (<see cref="Fed"/>) and every output left (<see cref="RunBound"/>).
    /// </summary>
    private Dictionary<string, AliasTarget>? OutputsIntoConsumed(
        IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
        IReadOnlyCollection<IShorokooTensorValue> consumed,
        IReadOnlyList<string> outputNames)
    {
        if (_aliases.Count == 0 || consumed.Count == 0) return null;
        var handed = new HashSet<IShorokooTensorValue>(consumed, ReferenceEqualityComparer.Instance);
        var fedAs = new Dictionary<IShorokooTensorValue, int>(ReferenceEqualityComparer.Instance);
        foreach (var value in inputs.Values) fedAs[value] = fedAs.GetValueOrDefault(value) + 1;

        Dictionary<string, AliasTarget>? into = null;
        foreach (var name in outputNames)
        {
            if (!_aliases.TryGetValue(name, out var slot) || !inputs.TryGetValue(slot.Input, out var value)) continue;
            if (!handed.Contains(value) || fedAs[value] != 1 || value is not OrtTensorValue own) continue;
            if (!Fits(own, slot)) continue;
            (into ??= new Dictionary<string, AliasTarget>(StringComparer.Ordinal))[slot.Output] =
                new AliasTarget(slot.Input, own);
        }
        return into;
    }

    /// <summary>Whether <paramref name="value"/> can take <paramref name="slot"/>'s output: a tensor
    /// of its element type and shape.</summary>
    private static bool Fits(OrtTensorValue value, AliasSlot slot)
        => value.ValueType == ShorokooOnnxValueType.Tensor
           && (int)value.ElementType == (int)slot.ElementType
           && value.Shape.AsSpan().SequenceEqual(slot.Shape);

    /// <summary>
    /// Runs the session, every output left in the run memory (<see cref="RunBound"/>). A run asked
    /// to hand back its memory has the allocators hand back what they keep cached for this session
    /// once it ends (<see cref="CachingAllocator.ReleaseCached"/>).
    /// </summary>
    public IReadOnlyList<IShorokooTensorValue> Run(
        IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
        IReadOnlyList<string> outputNames,
        RunSettings runSettings)
    {
        ArgumentNullException.ThrowIfNull(runSettings);
        try
        {
            return RunBound(inputs, outputNames, into: null, runSettings);
        }
        finally
        {
            HandBackIfAsked(runSettings);
        }
    }

    /// <summary>Has the allocators hand back what they keep cached for this session — and for the
    /// sessions carrying out its placements, which charge its accounts — where
    /// <paramref name="runSettings"/> asks a run to hand back its memory: as the run is over, once
    /// everything it let go of has gone back to them.</summary>
    private void HandBackIfAsked(RunSettings? runSettings)
    {
        if (runSettings is null) return;
        if (runSettings.ShrinkArenaAfterRun)
        {
            if (_cardAccount is { } card) card.Allocator.ReleaseCached(card);
            _hostAccount.Allocator.ReleaseCached(_hostAccount);
        }
    }

    /// <summary>
    /// Runs the session, every output left in the run memory — on a CUDA session a tensor on this
    /// session's card, and a string tensor or a sequence in host memory; on any other session, host
    /// memory — where ONNX Runtime wrote it: in a block of its own that this session's allocator
    /// gave the run (<see cref="CachingAllocator"/>), or in the consumed value <paramref name="into"/>
    /// names for it. Nothing is copied after the run, and an output kept alive holds its own block
    /// and nothing else of the session's, idle or disposed.
    ///
    /// <para>Through an I/O binding where it has to be — to write into what the run consumed, or to
    /// leave an output on the card — and as ONNX Runtime runs a session plain otherwise, since a
    /// binding costs every run a few microseconds more.</para>
    ///
    /// <para>Under a device-memory budget, each output on the card is handed over as the run
    /// returns (<see cref="CachingAllocator.HandOver"/>): the context counts it with the tensors
    /// attached to it from then on, so the limit the session's later runs get is no longer spent on
    /// it. What the allocators keep cached is handed back by the caller, once the run has let go of
    /// everything it will (<see cref="HandBackIfAsked"/>).</para>
    /// </summary>
    private IReadOnlyList<IShorokooTensorValue> RunBound(
        IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
        IReadOnlyList<string> outputNames,
        Dictionary<string, AliasTarget>? into,
        RunSettings runSettings,
        IReadOnlyDictionary<string, PlacedBinding>? placed = null)
    {
        ArgumentNullException.ThrowIfNull(runSettings);
        runSettings.CancellationToken.ThrowIfCancellationRequested();
        var outputs = into is null && _cardMemory is null && placed is null
            ? RunPlain(inputs, outputNames, runSettings)
            : RunThroughABinding(inputs, outputNames, into, runSettings, placed);
        if (_cardAccount?.Limit is not null) HandOver(outputs);
        return outputs;
    }

    /// <summary>The allocator account this session charges its host memory to, which the sessions
    /// carrying out its placements charge too (<see cref="OrtPlacements"/>).</summary>
    internal CachingAllocator.Account HostAccount => _hostAccount;

    /// <summary>The account this session charges its card memory to, or null on the host.</summary>
    internal CachingAllocator.Account? CardAccount => _cardAccount;

    /// <summary>
    /// What <paramref name="run"/> answers, with <paramref name="peak"/> the most it asked of this
    /// session's allocator beyond what the session held as it started — on its card for a CUDA
    /// session, of host memory otherwise — its outputs included.
    /// </summary>
    internal T Measured<T>(Func<T> run, out long peak)
    {
        var account = _cardAccount ?? _hostAccount;
        account.BeginPeak();
        var result = run();
        peak = account.Peak;
        return result;
    }

    /// <summary>What this session holds of its allocators' memory now: what it was built with and
    /// what its callers keep of its runs.</summary>
    internal long HeldBytes
        => _hostAccount.Allocator.Statistics(_hostAccount).InUseBytes
           + (_cardAccount is { } card ? card.Allocator.Statistics(card).InUseBytes : 0);

    /// <summary>Hands the outputs this session's allocator gave the run on the card over to the
    /// caller, as far as its limit is concerned (<see cref="CachingAllocator.HandOver"/>).</summary>
    private void HandOver(IReadOnlyList<IShorokooTensorValue> outputs)
    {
        var card = _cardAccount!;
        foreach (var output in outputs)
        {
            if (output is not OrtTensorValue own || own.ValueType != ShorokooOnnxValueType.Tensor) continue;
            var value = own.Inner;
            if (value.GetTensorSizeInBytes() == 0) continue;
            card.Allocator.HandOver(card, OrtBackend.AddressOf(value));
        }
    }

    /// <summary>
    /// The run <paramref name="run"/> makes, with run options armed to stop when the token of
    /// <paramref name="runSettings"/> is cancelled, and everything it allocates charged to this
    /// session (<see cref="CachingAllocator.Charge"/>). <paramref name="feeds"/> is what holds the
    /// values the run reads: ORT takes their handles as bare pointers and keeps no reference to the
    /// managed wrappers, so without this a collection inside the native run could free the feeds
    /// while it is still reading them. Kept alive in the finally rather than after the call, so it is
    /// reached however the run ends — a terminated one is still reading those buffers right up to the
    /// moment it gives up.
    /// </summary>
    private T Invoke<T>(RunSettings runSettings, object feeds, Func<RunOptions, T> run)
    {
        var abortToken = runSettings.CancellationToken;
        using var runOptions = new RunOptions();
        using var runLog = RouteRun(runOptions, runSettings.Log);
        using var inProgress = Log.Enter(runSettings.Log);
        using var abort = AbortWhenCancelled(runOptions, abortToken);
        using var charge = CachingAllocator.Charge(_hostAccount, _cardAccount);
        var finished = false;
        try
        {
            var result = run(runOptions);
            finished = true;
            return result;
        }
        catch (OnnxRuntimeException cause) when (WasStopped(cause, abortToken))
        {
            throw Aborted(cause, abortToken);
        }
        finally
        {
            // A run that fails or is stopped returns without draining the card's stream, so the
            // memory it let go of waits for the card before anything else may have it.
            if (!finished) charge.Unfinished();
            GC.KeepAlive(feeds);
        }
    }

    /// <summary>
    /// Sets <paramref name="runOptions"/> to log where <paramref name="log"/> says: through the
    /// session's own logger where they are the settings it was built with, and otherwise through a
    /// logger of the run's own, named by a route the caller closes when the run is over — ONNX
    /// Runtime names a run's logger by the session's id and the run's tag, so what the run logs
    /// reaches this run's settings and no other run's, of this session or another, however many run
    /// at once, on whichever thread. What ONNX Runtime logs during the run through the session's
    /// logger instead — most of what it says about a run, an output of another shape than the model
    /// states included — reaches the run's settings through the run being in progress on the
    /// calling thread (<see cref="RuntimeLogRoutes.Route.Enter(LogSettings)"/>), from the level the
    /// session's logger was built at (<see cref="OrtBackend.LoggerSeverity"/>).
    /// </summary>
    private RuntimeLogRoutes.Route? RouteRun(RunOptions runOptions, LogSettings log)
    {
        ArgumentNullException.ThrowIfNull(log);
        if (log.Equals(Log.Log))
        {
            runOptions.LogSeverityLevel = (OrtLoggingLevel)OrtBackend.LoggerSeverity(Log.Log);
            return null;
        }
        var route = RuntimeLogRoutes.Open(log);
        runOptions.LogId = route.Id;
        runOptions.LogSeverityLevel = (OrtLoggingLevel)OrtBackend.LoggerSeverity(log);
        return route;
    }

    /// <summary>A run of a session whose outputs all stay in host memory, writing nothing into what
    /// it consumed: ONNX Runtime makes each output where it leaves it.</summary>
    private IReadOnlyList<IShorokooTensorValue> RunPlain(
        IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
        IReadOnlyList<string> outputNames,
        RunSettings runSettings)
    {
        var ortInputs = new Dictionary<string, OrtValue>(inputs.Count);
        foreach (var (name, value) in inputs)
            ortInputs[name] = Fed(name, value);
        // `results` is deliberately not disposed. It is a container whose Dispose would dispose the
        // values inside it, and those are exactly what this returns: each one is handed to an
        // OrtTensorValue, and from there to the TensorData that owns it and releases it when disposed
        // (Shorokoo/Shorokoo#180). The container itself holds nothing else to release.
        var results = Invoke(runSettings, inputs, runOptions => _session.Run(runOptions, ortInputs, outputNames));
        var wrapped = new IShorokooTensorValue[results.Count];
        for (int i = 0; i < wrapped.Length; i++) wrapped[i] = new OrtTensorValue(results[i]);
        return wrapped;
    }

    /// <summary>
    /// What a value placed in the memory of an input the run consumed is bound to for one run:
    /// <see cref="Bound"/>, an ORT value over its range, and, where the caller asked for it as an
    /// output, <see cref="Returned"/>, the value handed back for it — a view over the same range
    /// holding a lease on the block (see <see cref="SharedBlock"/>). An intermediate is bound to be
    /// written there and handed back to nobody.
    /// </summary>
    internal readonly record struct PlacedBinding(OrtValue Bound, OrtTensorValue? Returned);

    /// <summary>A run through an I/O binding: each output bound to the consumed value
    /// <paramref name="into"/> names for it, or to the range <paramref name="placed"/> gives it, and
    /// otherwise to the memory it is left in. A placed value the caller did not ask for is bound
    /// as well, and what the run hands back for it let go of.</summary>
    private IReadOnlyList<IShorokooTensorValue> RunThroughABinding(
        IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
        IReadOnlyList<string> outputNames,
        Dictionary<string, AliasTarget>? into,
        RunSettings runSettings,
        IReadOnlyDictionary<string, PlacedBinding>? placed = null)
    {
        using var binding = _session.CreateIoBinding();
        foreach (var (k, v) in inputs)
            binding.BindInput(k, Fed(k, v));

        // An output bound to a value is written into that value's memory by the node that produces
        // it -- a consumed input's, for an aliased one, which was checked to fit before it got here,
        // or a range of one, for a placed one, which the placement proof put there. An output bound
        // to a memory is allocated there by ORT, from this session's allocator for it, copied there
        // by ORT where the node that produces it ran elsewhere (Shorokoo/Shorokoo#493); ORT sizes
        // those itself, so a shape it only learns while running is fine. A string tensor and a
        // sequence are left in host memory: ORT keeps strings there whatever its provider, and reads a
        // sequence's elements one by one through its host allocator, which reads an element left in
        // device memory as though it were host memory, and the process faults.
        var hostMemoryInfo = OrtMemoryInfo.DefaultInstance;
        List<string> bound = [.. outputNames];
        if (placed is not null)
            foreach (var name in placed.Keys)
                if (!bound.Contains(name)) bound.Add(name);
        foreach (var name in bound)
        {
            if (placed is not null && placed.TryGetValue(name, out var place))
                binding.BindOutput(name, place.Bound);
            else if (into is not null && into.TryGetValue(name, out var target))
                binding.BindOutput(name, target.Value.Inner);
            else
                binding.BindOutputToDevice(name, OnTheCard(name) ? _cardMemory! : hostMemoryInfo);
        }

        // The binding holds the feeds' raw handles, which `inputs` holds the wrappers of.
        var results = Invoke(runSettings, inputs, runOptions => _session.RunWithBoundResults(runOptions, binding));

        // RunWithBoundResults returns the bound outputs in the binding's own order, which is the
        // order they were bound in -- ask it rather than assume, and hand them back in the order the
        // caller named.
        var boundNames = binding.GetOutputNames();
        // Nothing owns these values until each is wrapped and handed to a TensorData, and the
        // collection is deliberately not disposed -- its Dispose would dispose the values in it,
        // which are what this returns -- so anything that goes wrong between here and the return
        // leaks an allocation apiece. Establish the shape first, and dispose the lot if it is not
        // what it must be.
        if (boundNames.Length != results.Count || results.Count != bound.Count)
        {
            foreach (var value in results) value.Dispose();
            throw new InvalidOperationException(
                $"The run bound {boundNames.Length} outputs and returned {results.Count} values for "
                + $"{bound.Count} bound names; they must agree one for one.");
        }

        // Bound in the caller's order, the outputs come back in it, before any placed value the
        // caller did not ask for; each is looked up by name where they do not.
        Dictionary<string, OrtValue>? byName = null;
        for (int i = 0; i < outputNames.Count && byName is null; i++)
            if (!string.Equals(boundNames[i], outputNames[i], StringComparison.Ordinal))
            {
                byName = new Dictionary<string, OrtValue>(results.Count, StringComparer.Ordinal);
                for (int b = 0; b < boundNames.Length; b++) byName[boundNames[b]] = results[b];
            }
        // Same reason as the count check above, and the same handling: a name that does not come
        // back is a bad run, not an excuse to drop every allocation it made.
        if (byName is not null)
            foreach (var name in outputNames)
                if (!byName.ContainsKey(name))
                {
                    foreach (var orphan in results) orphan.Dispose();
                    throw new InvalidOperationException(
                        $"The run bound no output named '{name}'; it bound {string.Join(", ", boundNames)}.");
                }

        var wrapped = new IShorokooTensorValue[outputNames.Count];
        // Of a placed run, the values handed on, the others let go of below.
        var handed = placed is null ? null : new HashSet<OrtValue>(ReferenceEqualityComparer.Instance);
        for (int i = 0; i < wrapped.Length; i++)
        {
            var value = byName is null ? results[i] : byName[outputNames[i]];
            if (placed is not null && placed.TryGetValue(outputNames[i], out var place) && place.Returned is { } view)
            {
                // The view over the range, which holds the lease, is what is handed back; the value
                // the run returned for it holds nothing.
                wrapped[i] = view;
                continue;
            }
            handed?.Add(value);
            // An output written into a consumed value that stands on a block stands on it too, over
            // the same range: it takes a lease of its own, since the consumed value's goes as the
            // consumed value is released.
            wrapped[i] = into is not null && into.TryGetValue(outputNames[i], out var target) && target.Value.Range is { } range
                ? Leased(value, range)
                : new OrtTensorValue(value);
        }
        if (handed is not null)
            foreach (var value in results)
                if (!handed.Contains(value)) value.Dispose();
        return wrapped;
    }

    /// <summary><paramref name="value"/>, which stands on <paramref name="range"/>, holding a lease
    /// of its own there.</summary>
    private static OrtTensorValue Leased(OrtValue value, BlockRange range)
    {
        range.Lease();
        return new OrtTensorValue(value, range);
    }

    /// <summary>Whether output <paramref name="name"/> is left on this session's card: on a CUDA
    /// session, every tensor output but a string one.</summary>
    private bool OnTheCard(string name)
    {
        if (_cardMemory is null) return false;
        var output = _session.OutputMetadata[name];
        return output.IsTensor && output.ElementDataType != TensorElementType.String;
    }

    /// <summary>
    /// Where the session's execution provider computes its outputs, as a whole: Unknown where the
    /// native build does not answer. A session that reports host memory for some outputs and its
    /// own for others has a graph ORT partitioned across the two, and one on a device backend
    /// reporting host memory for all of them ran the whole graph there.
    /// </summary>
    private static SessionOutputPlacement DiscoverPlacement(InferenceSession session)
    {
        try
        {
            var host = 0;
            var onDevice = 0;
            using var infos = session.GetMemoryInfosForOutputs();
            foreach (var info in infos)
            {
                if (OrtTensorValue.IsHostAllocator(info.Name)) host++;
                else onDevice++;
            }

            // The session is a bare argument whose last read is the call above, so without this
            // the JIT may retire it before the native call returns and a GC on any thread runs
            // ~InferenceSession underneath it -- a CompiledGraph is held weakly by its context, so
            // there is no strong reference above this one to fall back on.
            //
            // Measured rather than assumed, and the measurement is worth recording: removing this
            // does NOT reproduce a fault. Four hundred probes of a freshly compiled graph held only
            // in a local, in Release on a card, against a thread collecting and draining
            // finalizers, all came back with the placement and none took the process down -- because
            // the caller reaches this through a Lazy whose factory closure holds the session for as
            // long as the factory runs. That is what roots it today. This line is what makes the
            // rooting the method's own rather than a property of how it happens to be invoked
            // (Shorokoo/Shorokoo#178), which is exactly the kind of thing a refactor takes away
            // silently.
            GC.KeepAlive(session);

            return (host, onDevice) switch
            {
                (0, 0) => SessionOutputPlacement.Unknown,
                (_, 0) => SessionOutputPlacement.Host,
                (0, _) => SessionOutputPlacement.Device,
                _ => SessionOutputPlacement.Mixed,
            };
        }
        // Catching broadly is the point: the doc above promises a failed probe answers Unknown and
        // nothing else, and Lazy caches an escaping exception and rethrows it on every later access.
        catch (Exception) { }
        return SessionOutputPlacement.Unknown;
    }

    public SessionOutputPlacement OutputPlacement => _placement.Value;

    /// <summary>
    /// The inputs this session's runs read in host memory, and the outputs they leave there, as
    /// ONNX Runtime says where it reads and leaves each one; null where it does not say. Asked once.
    /// </summary>
    internal (IReadOnlySet<string> Inputs, IReadOnlySet<string> Outputs)? HostMemoryNames => _hostMemoryNames.Value;

    private readonly Lazy<(IReadOnlySet<string> Inputs, IReadOnlySet<string> Outputs)?> _hostMemoryNames;

    private (IReadOnlySet<string>, IReadOnlySet<string>)? ReadHostMemoryNames()
    {
        try
        {
            static HashSet<string> InHost(IReadOnlyList<string> names, IReadOnlyCollection<OrtMemoryInfo> infos)
                => [.. names.Zip(infos).Where(p => OrtTensorValue.IsHostAllocator(p.Second.Name)).Select(p => p.First)];
            using var inputs = _session.GetMemoryInfosForInputs();
            using var outputs = _session.GetMemoryInfosForOutputs();
            (IReadOnlySet<string>, IReadOnlySet<string>) names = (InHost(_session.InputNames, inputs), InHost(_session.OutputNames, outputs));
            // After the names are read, for the reason DiscoverPlacement gives: each info points into
            // the session rather than owning anything.
            GC.KeepAlive(_session);
            return names;
        }
        // Not said is answered as such, and the caller takes nothing for read on the card.
        catch (Exception) { return null; }
    }

    /// <summary>ORT's memory info for a CUDA device's memory, as a binding names the device an
    /// output is left on. <c>CudaPinned</c> is the pinned host arena that host-to-device copies stage
    /// through and is a different allocator — <see cref="CudaPinnedArenaMemoryInfo"/>.</summary>
    private static OrtMemoryInfo CudaMemoryInfo(int deviceId)
    {
        return new OrtMemoryInfo("Cuda", OrtAllocatorType.DeviceAllocator, deviceId, OrtMemType.Default);
    }


    /// <summary>ORT's memory info for the pinned host arena of the same device. The memory type is
    /// what tells it from the device arena above; the name is spelled the way ORT reports it on an
    /// output it serves from there.</summary>
    private static OrtMemoryInfo CudaPinnedArenaMemoryInfo(int deviceId)
    {
        return new OrtMemoryInfo(
            "CudaPinned", OrtAllocatorType.ArenaAllocator, deviceId, OrtMemType.CpuOutput);
    }

    /// <summary>
    /// The pinned host arena of this session's device, or null on a session with no CUDA provider
    /// — which has no such arena rather than an empty one.
    /// </summary>
    private OrtAllocator? CreatePinnedAllocator()
    {
        if (_cudaDeviceId is not { } device) return null;
        OrtMemoryInfo? owned = null;
        try
        {
            owned = CudaPinnedArenaMemoryInfo(device);
            var allocator = new OrtAllocator(_session, owned);
            // The constructor takes the info as a bare handle; reading `owned` again here, to keep
            // it, keeps it reachable until the constructor has returned.
            _ownedPinnedMemoryInfo = owned;
            return allocator;
        }
        catch (Exception)
        {
            owned?.Dispose();
            return null;
        }
    }

    /// <summary>
    /// The figures of what this session's allocator account holds (<see cref="CachingAllocator"/>):
    /// on the card for a CUDA session, in host memory otherwise. Every block its construction and
    /// its runs took and have not let go of is in use, its weights and the outputs a caller keeps
    /// alike; what it let go of and the allocator keeps for its next runs is allocated besides.
    /// </summary>
    public ArenaStatistics? ReadArenaStatistics()
    {
        if (OnOrtArena) return ReadOrtArenaStatistics();
        var account = _cardAccount ?? _hostAccount;
        return account.Allocator.Statistics(account);
    }

    /// <summary>Whether this session allocates through the arena ONNX Runtime made for it rather than
    /// Shorokoo's allocator (<see cref="OrtBackend.SessionsUseOrtArena"/>).</summary>
    internal bool OnOrtArena { get; init; }

    /// <summary>Whether this session runs its operators on the process's thread pools rather than
    /// pools of its own (<see cref="OrtBackend.SessionsShareThreadPools"/>).</summary>
    internal bool OnSharedThreadPools { get; init; }

    /// <summary>The figures of the arena ONNX Runtime made for this session: the card's for a CUDA
    /// session, the host's otherwise.</summary>
    private ArenaStatistics? ReadOrtArenaStatistics()
    {
        if (_cudaDeviceId is not { } device) return ReadOrtArenaStatistics(OrtMemoryInfo.DefaultInstance);
        using var card = new OrtMemoryInfo(OrtMemoryInfo.allocatorCUDA, OrtAllocatorType.ArenaAllocator, device, OrtMemType.Default);
        return ReadOrtArenaStatistics(card);
    }

    private ArenaStatistics? ReadOrtArenaStatistics(OrtMemoryInfo memory)
    {
        using var allocator = new OrtAllocator(_session, memory);
        return OrtArenaStats.Read(allocator);
    }

    /// <summary>
    /// Sets the most this session's runs may hold on its card to <paramref name="limitBytes"/>,
    /// which the card's allocator enforces as each block is asked for (<see cref="CachingAllocator.Account.Limit"/>),
    /// with no need to build the session again. A session with no card has nothing to limit.
    /// </summary>
    public bool TryLimitDeviceMemory(long limitBytes)
    {
        if (_cardAccount is not { } card) return false;
        card.Limit = limitBytes;
        return true;
    }

    public ArenaStatistics? ReadPinnedArenaStatistics()
        => _pinnedAllocator.Value is { } allocator ? OrtArenaStats.Read(allocator) : null;

    /// <summary>
    /// Which execution provider ran each node, read out of the profile ORT has been writing since
    /// this session was built — or null when it was not built to write one.
    ///
    /// <para>Reading it ends the profiling: that is ORT's own shape, since the events are buffered
    /// and the file is only complete once profiling stops. So the trace covers every run up to
    /// this call, later runs are not in it, and the answer is kept so a second call gets the same
    /// one rather than asking a session that is no longer recording.</para>
    /// </summary>
    public NodePlacement? ReadNodePlacement()
    {
        if (_profileDirectory is null) return null;
        lock (_profileGate)
        {
            if (_profilingEnded) return _nodePlacement;
            _profilingEnded = true;
            try
            {
                _nodePlacement = OrtProfile.Read(_session.EndProfiling());
            }
            catch (Exception) { _nodePlacement = null; }
            return _nodePlacement;
        }
    }

    /// <summary>
    /// Arms <paramref name="runOptions"/> so that cancelling <paramref name="token"/> sets ORT's
    /// terminate flag on it, which is what makes a run abortable at all: ORT reads that flag
    /// before each node, and a run's options are otherwise a local no other thread can reach.
    /// Disarmed by disposing the registration, which is why the caller's <c>using</c> for it sits
    /// <i>after</i> the one for the options: disposal runs in reverse, so the registration is gone
    /// — and a callback already running on the cancelling thread waited out, which
    /// <see cref="CancellationTokenRegistration.Dispose"/> does — before the native handle it
    /// writes to is released.
    ///
    /// <para>The registration also roots <paramref name="runOptions"/> for its own lifetime, since
    /// the token source holds it as the callback's state. That is belt and braces: the caller's
    /// <c>using</c> reads the local in its <c>finally</c>, which already keeps it alive across the
    /// native call.</para>
    ///
    /// <para>A token that can never be cancelled gets no registration at all — the default
    /// <see cref="CancellationTokenRegistration"/> disposes to nothing — so an ordinary run pays
    /// nothing for this.</para>
    /// </summary>
    private static CancellationTokenRegistration AbortWhenCancelled(
        RunOptions runOptions, CancellationToken token)
        => token.CanBeCanceled
            ? token.Register(static state => ((RunOptions)state!).Terminate = true, runOptions)
            : default;

    /// <summary>
    /// Whether <paramref name="cause"/> is ORT reporting the run <paramref name="token"/> stopped,
    /// rather than a failure that merely happened while that token was cancelled. A stopped run and
    /// a broken model come back as the same exception type, so what tells them apart is ORT naming
    /// the flag; reading every failure as a cancellation because one was asked for reports a model
    /// that cannot run as a run the caller stopped, and a caller told that retries rather than
    /// fixing the model.
    ///
    /// <para>The window is narrow — ORT reads the flag between nodes and stops there, so a genuine
    /// failure can only outrun a cancellation from the kernel that was already running — which is
    /// also why nothing pins this from the outside: reaching the failing node with the flag already
    /// set is the race itself.</para>
    /// </summary>
    private static bool WasStopped(OnnxRuntimeException cause, CancellationToken token)
        => token.IsCancellationRequested
            && cause.Message.Contains("terminate flag", StringComparison.OrdinalIgnoreCase);

    /// <summary>
    /// What a run that was stopped throws. ORT reports a terminated run as a plain failed one —
    /// an <see cref="OnnxRuntimeException"/> naming the flag — which is indistinguishable at a
    /// glance from a model that is broken, so it is translated here into the one exception .NET
    /// gives this meaning. The cause is kept as the inner exception, and the token is carried so
    /// that a caller racing several runs can tell which cancellation stopped this one.
    /// </summary>
    private static OperationCanceledException Aborted(Exception cause, CancellationToken token)
        => new(
            "The run was stopped before it finished: the RunSettings.CancellationToken it was "
            + "given was cancelled while it was running, so it produced no outputs.",
            cause,
            token);

    public void Dispose()
    {
        if (_pinnedAllocator.IsValueCreated)
        {
            _pinnedAllocator.Value?.Dispose();
            _ownedPinnedMemoryInfo?.Dispose();
        }
        Placements?.Dispose();
        _session.Dispose();
        // After the session, which may log as it is released.
        if (_ownsLogRoute) Log.Dispose();
        foreach (var weight in SharedWeights) weight.Dispose();
        // After the session, whose release lets go of its weights through them.
        _accounts?.Close();
        _cardMemory?.Dispose();
        foreach (var view in SuppliedViews) view.Dispose();
        // After the session, which is what closes the profile file it has been writing.
        DeleteProfileDirectory();
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Deletes the trace folder for a session nobody disposed. A compiled graph is held weakly by
    /// the context that made it, so one that is used and dropped is collected with no disposal
    /// ever running — and each traced session owns a folder, so without this a program that
    /// compiles as it goes leaves one behind per compile for as long as it runs.
    ///
    /// <para>Only the folder. Nothing native is touched here: the session and everything built
    /// over it own their own handles and are finalized in their own time, and reaching for one of
    /// them from this thread is exactly the use-after-free this backend takes such care to avoid.
    /// A session built without tracing suppresses this in its constructor rather than joining the
    /// finalization queue to do nothing.</para>
    /// </summary>
    ~OrtSession() => DeleteProfileDirectory();

    private void DeleteProfileDirectory()
    {
        if (_profileDirectory is null) return;
        try { Directory.Delete(_profileDirectory, recursive: true); }
        // A temp folder that will not delete is not worth failing a disposal over; the platform
        // reclaims it, and there is nothing a caller could do here.
        catch (Exception) { }
    }
}
