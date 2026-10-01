using Microsoft.ML.OnnxRuntime;
using Shorokoo.Core.Backends;
using TensorElementType = Microsoft.ML.OnnxRuntime.Tensors.TensorElementType;

namespace Shorokoo.OnnxRuntime;

internal sealed class OrtSession : IShorokooSession
{
    private readonly InferenceSession _session;
    private readonly int? _cudaDeviceId;
    private readonly OrtBackend _backend;

    // The outputs whose shape ONNX Runtime settled when it built this session, by name: each run
    // allocates them in memory of their own before it starts and binds them there (see RunBound).
    private readonly Dictionary<string, SettledOutput> _settled;

    // The shape of each settled output whose every dimension is a count, which is the shape of it on
    // every run: one array, which ONNX Runtime only reads, rather than one a run.
    private readonly Dictionary<string, long[]> _fixedShapes;

    // The allocators this session's outputs come from, on the host and on its card, resolved on
    // first use (see RuntimeAllocators).
    private OrtAllocator? _hostOutputs;
    private OrtAllocator? _cardOutputs;

    // Where the session's runtime computes its outputs, probed on first ask: a session is a common
    // object here -- parameter initialization and every eager Eval build one -- and most are never
    // asked.
    private readonly Lazy<SessionOutputPlacement> _placement;

    // The memory a run of a CUDA session reads a tensor in and leaves one in -- its card's, in the
    // form ONNX Runtime binds an output to -- or null on any other session, whose run memory is the
    // host's. Held in a field, not a local: OrtMemoryInfo owns a native handle and the binding takes
    // it as a bare pointer.
    private readonly OrtMemoryInfo? _cardMemory;

    // This session's own allocator for the arena the figures come from, and the memory info naming
    // it. Both in fields for the reason the one above is: they own native handles that go over as
    // bare pointers. Built on first ask, because no session needs them unless something asked for
    // run statistics.
    private readonly Lazy<OrtAllocator?> _arenaAllocator;
    private OrtMemoryInfo? _ownedArenaMemoryInfo;

    // The same pair for the pinned host arena a CUDA session stages its crossings through. A
    // second allocator rather than a second reading of the first: they are different arenas with
    // different figures, and a session with no CUDA provider has no second one at all.
    private readonly Lazy<OrtAllocator?> _pinnedAllocator;
    private OrtMemoryInfo? _ownedPinnedMemoryInfo;

    // The folder ORT writes this session's profile into, or null when it was not built to record
    // one -- which is the default. Deleted with the session.
    private readonly string? _profileDirectory;

    /// <summary>The views of supplied initializers this session was built over (see
    /// <c>OrtBackend.Supply</c>), which ONNX Runtime requires to outlive it: released after it.</summary>
    internal IReadOnlyList<OrtValue> SuppliedViews { get; init; } = [];

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

    public IReadOnlyList<SettledOutput> SettledOutputs { get; }

    public OrtSession(
        InferenceSession session,
        int? cudaDeviceId,
        OrtBackend backend,
        string? profileDirectory,
        IReadOnlyList<ProvedAlias> outputAliases)
    {
        _session = session;
        _cudaDeviceId = cudaDeviceId;
        _backend = backend;
        _profileDirectory = profileDirectory;
        _aliases = Slots(session, outputAliases);
        BindableAliases = [.. _aliases.Values.Select(slot => new OutputAlias(slot.Output, slot.Input))];
        _settled = Settled(session);
        SettledOutputs = [.. session.OutputNames.Where(_settled.ContainsKey).Select(name => _settled[name])];
        _fixedShapes = _settled.Values
            .Where(output => output.Dimensions.All(dimension => dimension.Input is null))
            .ToDictionary(output => output.Name, output => output.Dimensions.Select(dimension => dimension.Count).ToArray(), StringComparer.Ordinal);
        // Nothing to clean up, so nothing to finalize -- every untraced session would otherwise
        // join the finalization queue to run an early return.
        if (profileDirectory is null) GC.SuppressFinalize(this);
        _placement = new Lazy<SessionOutputPlacement>(() => DiscoverPlacement(session));
        // The arena's info names the card's memory as a binding wants it: ONNX Runtime binds an
        // output to the device an info names, and allocates it from the session's own allocator there.
        _cardMemory = cudaDeviceId is { } device ? CudaArenaMemoryInfo(device) : null;
        _arenaAllocator = new Lazy<OrtAllocator?>(CreateArenaAllocator);
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
    /// The outputs of <paramref name="session"/> a run can be handed memory for before it starts: a
    /// tensor whose elements have a fixed width — a string tensor's are objects ONNX Runtime makes as
    /// it writes them — and whose every dimension ONNX Runtime settled when it built the session
    /// (<see cref="OrtOutputShapes"/>): a count, or one it leaves open whose symbolic name is that of
    /// a dimension of an input, as the runtime names an output dimension it has proved equal to that
    /// input's. A product of two inputs of open shapes, say, is as long as the first's rows and as wide
    /// as the second's columns, which each run tells. None where the runtime's answer could not be
    /// read, which leaves every output to be copied out of the arena instead (<see cref="RunBound"/>).
    /// </summary>
    private static Dictionary<string, SettledOutput> Settled(InferenceSession session)
    {
        var settled = new Dictionary<string, SettledOutput>(StringComparer.Ordinal);
        if (OrtOutputShapes.Read(session) is not { } shapes) return settled;

        // Each input dimension's symbolic name, and where it is.
        var inputDimensions = new Dictionary<string, (string Input, int Axis)>(StringComparer.Ordinal);
        foreach (var (input, metadata) in session.InputMetadata)
        {
            if (!metadata.IsTensor) continue;
            var symbols = metadata.SymbolicDimensions;
            for (int axis = 0; axis < symbols.Length; axis++)
                if (!string.IsNullOrEmpty(symbols[axis])) inputDimensions.TryAdd(symbols[axis], (input, axis));
        }

        for (int i = 0; i < shapes.Length; i++)
        {
            var name = session.OutputNames[i];
            var output = session.OutputMetadata[name];
            if (shapes[i] is not { } shape || !output.IsTensor) continue;
            var elementType = (ShorokooTensorElementType)(int)output.ElementDataType;
            if (!HasFixedWidth(elementType)) continue;
            var symbols = output.SymbolicDimensions;
            var dimensions = new OutputDimension[shape.Length];
            var known = true;
            for (int d = 0; d < shape.Length && known; d++)
            {
                if (shape[d] > 0)
                    dimensions[d] = OutputDimension.Of(shape[d]);
                else if (shape[d] < 0 && d < symbols.Length && symbols[d] is { Length: > 0 } symbol
                         && inputDimensions.TryGetValue(symbol, out var from))
                    dimensions[d] = OutputDimension.OfInput(from.Input, from.Axis);
                else
                    known = false;
            }
            if (known) settled[name] = new SettledOutput(name, elementType, dimensions);
        }
        return settled;
    }

    private static bool HasFixedWidth(ShorokooTensorElementType elementType)
    {
        try
        {
            TensorElementLayout.ElementSizeInBytes(elementType);
            return true;
        }
        catch (NotSupportedException)
        {
            return false;
        }
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
    /// <paramref name="aliasedInputs"/>.
    ///
    /// <para>Each consumed value is this backend's from here, on every path, and is released
    /// through it in the <c>finally</c> below — after the native run, which ONNX Runtime does not
    /// let go of its inputs before, and before this returns the outputs or rethrows a failure.
    /// Nothing the caller holds points into one of them any more: the run's outputs are values of
    /// their own. ONNX Runtime keeps every input until the run ends — its memory planner gives each
    /// feed an extra use so a caller can read it after <c>Run</c> returns — so there is no earlier
    /// point at which a consumed input could go.</para>
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
        try
        {
            if (OutputsIntoConsumed(inputs, consumed, outputNames) is not { } into)
                return Run(inputs, outputNames, runSettings);

            var results = RunBound(inputs, outputNames, into, runSettings);
            var aliased = new string?[outputNames.Count];
            for (int i = 0; i < outputNames.Count; i++)
                if (into.TryGetValue(outputNames[i], out var target)) aliased[i] = target.Input;
            aliasedInputs = aliased;
            return results;
        }
        finally
        {
            // Through the backend, which is the one release path for memory it allocated. Its
            // release is a disposal, which does not throw, so none of them can be skipped.
            foreach (var value in consumed) ((IShorokooBackend)_backend).Release(value);
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
    /// Runs the session, every output left in the run memory and none of it in the session's own
    /// (<see cref="RunBound"/>).
    /// </summary>
    public IReadOnlyList<IShorokooTensorValue> Run(
        IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
        IReadOnlyList<string> outputNames,
        RunSettings runSettings)
        => RunBound(inputs, outputNames, into: null, runSettings);

    /// <summary>
    /// Runs the session, every output left in the run memory — on a CUDA session a tensor on this
    /// session's card, and a string tensor or a sequence in host memory; on any other session, host
    /// memory — and in memory of its own: what comes back holds its own bytes and nothing of the
    /// arena the run computed in, which stays this session's, for its runs. So an output kept alive
    /// keeps no arena alive, idle or disposed, and a run that hands its arena's unused blocks back
    /// hands back everything but the session's weights. Each output gets there one of three ways:
    /// <list type="bullet">
    /// <item>written into the consumed value <paramref name="into"/> names for it, if any;</item>
    /// <item>written into memory the run is handed for it before it starts, where the session settled
    /// its shape when it was built (<see cref="SettledOutputs"/>): out of this runtime's own allocator
    /// for the card or the host (<see cref="RuntimeAllocators"/>);</item>
    /// <item>copied out of the arena into the same memory as the run returns, for any other — a
    /// shape known only once the run is under way can be sized by nothing but the runtime, which
    /// makes such an output in the arena (<see cref="Owned"/>).</item>
    /// </list>
    /// A run asked to hand its arena's unused blocks back has that allocator hand back what it holds
    /// unused as well (<see cref="RuntimeAllocators.Shrink"/>), and does so as it starts, before
    /// any of its outputs is placed: an output placed in a block some released tensor left behind
    /// would keep that whole block from going back — measured on the card, a 256-byte output placed
    /// in the 64 MiB block a released tensor left kept all 64 MiB.
    ///
    /// <para>Through an I/O binding where it has to be, and as ONNX Runtime runs a session plain
    /// where it need not, since a binding costs every run a few microseconds more: measured on a
    /// 64-by-64 product on the CPU, 14 microseconds bound against 11 run plain, and 9 with the
    /// output's memory handed over. A run writing nothing into what it consumed runs plain where
    /// every output is settled, or, on a session whose outputs all stay in host memory, where none
    /// is.</para>
    /// </summary>
    private IReadOnlyList<IShorokooTensorValue> RunBound(
        IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
        IReadOnlyList<string> outputNames,
        Dictionary<string, AliasTarget>? into,
        RunSettings runSettings)
    {
        ArgumentNullException.ThrowIfNull(runSettings);
        var abortToken = runSettings.CancellationToken;
        abortToken.ThrowIfCancellationRequested();

        if (runSettings.ShrinkArenaAfterRun) RuntimeAllocators.Shrink(_cudaDeviceId, abortToken);
        var shapes = ShapesToPlace(inputs, outputNames, into);
        var placeable = 0;
        foreach (var shape in shapes)
            if (shape is not null) placeable++;
        return into is null && placeable == outputNames.Count
            ? RunPlaced(inputs, outputNames, shapes, runSettings)
            : into is null && placeable == 0 && _cardMemory is null
                ? RunUnbound(inputs, outputNames, runSettings)
                : RunThroughABinding(inputs, outputNames, into, shapes, runSettings);
    }

    /// <summary>
    /// The run <paramref name="run"/> makes, with run options set up for
    /// <paramref name="runSettings"/> and armed to stop when its token is cancelled.
    /// <paramref name="feeds"/> is what holds the values the run reads: ORT takes their handles as
    /// bare pointers and keeps no reference to the managed wrappers, so without this a collection
    /// inside the native run could free the feeds while it is still reading them. Kept alive in the
    /// finally rather than after the call, so it is reached however the run ends — a terminated one
    /// is still reading those buffers right up to the moment it gives up.
    /// </summary>
    private T Invoke<T>(RunSettings runSettings, object feeds, Func<RunOptions, T> run)
    {
        var abortToken = runSettings.CancellationToken;
        using var runOptions = new RunOptions();
        ConfigureRun(runOptions, runSettings);
        using var abort = AbortWhenCancelled(runOptions, abortToken);
        try
        {
            return run(runOptions);
        }
        catch (OnnxRuntimeException cause) when (WasStopped(cause, abortToken))
        {
            throw Aborted(cause, abortToken);
        }
        finally
        {
            GC.KeepAlive(feeds);
        }
    }

    /// <summary>
    /// The shape of each output in <paramref name="outputNames"/> this run is handed memory for
    /// before it starts: one that is settled (<see cref="SettledOutputs"/>), with any dimension it
    /// takes from an input read off the value that input is fed — or null, for one written into a
    /// consumed value (<paramref name="into"/>) or that the run makes as it goes.
    /// </summary>
    private long[]?[] ShapesToPlace(
        IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
        IReadOnlyList<string> outputNames,
        Dictionary<string, AliasTarget>? into)
    {
        var shapes = new long[]?[outputNames.Count];
        // Each input's shape, read once however many outputs take a dimension from it.
        Dictionary<string, long[]?>? fed = null;
        for (int i = 0; i < shapes.Length; i++)
        {
            var name = outputNames[i];
            if (into is not null && into.ContainsKey(name)) continue;
            if (_fixedShapes.TryGetValue(name, out var fixedShape))
                shapes[i] = fixedShape;
            else if (_settled.TryGetValue(name, out var settled))
                shapes[i] = ShapeFor(settled, inputs, fed ??= new Dictionary<string, long[]?>(StringComparer.Ordinal));
        }
        return shapes;
    }

    /// <summary>The shape <paramref name="settled"/> takes on a run fed <paramref name="inputs"/>, each
    /// input's shape read into <paramref name="fed"/> the first time it is asked for.</summary>
    private static long[]? ShapeFor(
        SettledOutput settled, IReadOnlyDictionary<string, IShorokooTensorValue> inputs, Dictionary<string, long[]?> fed)
        => settled.ShapeFor(input =>
        {
            if (!fed.TryGetValue(input, out var shape))
                fed[input] = shape =
                    !inputs.TryGetValue(input, out var value) || value.ValueType != ShorokooOnnxValueType.Tensor ? null
                    : value is OrtTensorValue ort ? ort.ReadShape
                    : value.Shape;
            return shape;
        });

    /// <summary>The allocator memory of its own for an output comes from: this runtime's for the
    /// card where <paramref name="onCard"/>, and for the host otherwise.</summary>
    private OrtAllocator OutputAllocator(bool onCard)
        => onCard
            ? _cardOutputs ??= _backend.OutputAllocator(onCard: true)
            : _hostOutputs ??= _backend.OutputAllocator(onCard: false);

    /// <summary>Memory of its own, of <paramref name="shape"/>, for output <paramref name="name"/>,
    /// which the run writes it into.</summary>
    private OrtValue Placed(string name, long[] shape)
        => OrtValue.CreateAllocatedTensorValue(
            OutputAllocator(onCard: OnTheCard(name)),
            (TensorElementType)(int)_settled[name].ElementType, shape);

    /// <summary>A run whose every output is handed memory before it starts, of the shape
    /// <paramref name="shapes"/> gives it, and written there.</summary>
    private IReadOnlyList<IShorokooTensorValue> RunPlaced(
        IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
        IReadOnlyList<string> outputNames,
        long[]?[] shapes,
        RunSettings runSettings)
    {
        var names = new string[inputs.Count];
        var values = new OrtValue[inputs.Count];
        var i = 0;
        foreach (var (name, value) in inputs)
        {
            names[i] = name;
            values[i++] = Fed(name, value);
        }
        var placed = new OrtValue[outputNames.Count];
        try
        {
            for (int o = 0; o < placed.Length; o++) placed[o] = Placed(outputNames[o], shapes[o]!);
            Invoke(runSettings, inputs, runOptions =>
            {
                _session.Run(runOptions, names, values, outputNames, placed);
                return placed;
            });
        }
        catch
        {
            foreach (var memory in placed) memory?.Dispose();
            throw;
        }
        var owned = new IShorokooTensorValue[placed.Length];
        for (int o = 0; o < placed.Length; o++) owned[o] = new OrtTensorValue(placed[o]);
        return owned;
    }

    /// <summary>A run of a session whose outputs all stay in host memory, none of them settled: the
    /// runtime makes each in the arena, and each is copied out (<see cref="Owned"/>).</summary>
    private IReadOnlyList<IShorokooTensorValue> RunUnbound(
        IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
        IReadOnlyList<string> outputNames,
        RunSettings runSettings)
    {
        var ortInputs = new Dictionary<string, OrtValue>(inputs.Count);
        foreach (var (name, value) in inputs)
            ortInputs[name] = Fed(name, value);
        // Not disposed: a container whose Dispose would dispose the values in it, which Owned takes
        // over.
        var results = Invoke(runSettings, inputs, runOptions => _session.Run(runOptions, ortInputs, outputNames));
        return Owned(outputNames, [.. results], written: null);
    }

    /// <summary>A run through an I/O binding: each output bound to the consumed value
    /// <paramref name="into"/> names for it, to memory of its own of the shape
    /// <paramref name="shapes"/> gives it, if any, and otherwise to the memory it is left in.</summary>
    private IReadOnlyList<IShorokooTensorValue> RunThroughABinding(
        IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
        IReadOnlyList<string> outputNames,
        Dictionary<string, AliasTarget>? into,
        long[]?[] shapes,
        RunSettings runSettings)
    {
        using var binding = _session.CreateIoBinding();
        foreach (var (k, v) in inputs)
            binding.BindInput(k, Fed(k, v));

        // An output bound to a value is written into that value's memory by the node that
        // produces it: a consumed input's, for an aliased one, and memory of its own for one whose
        // shape is settled -- an aliased one was checked to fit before it got here. An output bound
        // to a memory is allocated there by ORT, in the session's arena, copied there by ORT where
        // the node that produces it ran elsewhere (Shorokoo/Shorokoo#493); ORT sizes those itself,
        // so a shape it only learns while running is fine. A string tensor and a sequence are left
        // in host memory: ORT keeps strings there whatever its provider, and reads a sequence's
        // elements one by one through its host allocator, which reads an element left in device
        // memory as though it were host memory, and the process faults.
        var hostMemoryInfo = OrtMemoryInfo.DefaultInstance;
        // The memory handed to the settled outputs, let go of once the run is over: the values the
        // run hands back are over the same buffers, and ORT counts the references to a buffer.
        List<OrtValue>? placed = null;
        // Which outputs, by position, are bound to a value, and come back as they are rather than
        // copied.
        bool[]? written = null;
        IDisposableReadOnlyCollection<OrtValue> results;
        try
        {
            for (int o = 0; o < outputNames.Count; o++)
            {
                var name = outputNames[o];
                if (into is not null && into.TryGetValue(name, out var target))
                    binding.BindOutput(name, target.Value.Inner);
                else if (shapes[o] is { } shape)
                {
                    var memory = Placed(name, shape);
                    (placed ??= []).Add(memory);
                    binding.BindOutput(name, memory);
                }
                else
                {
                    binding.BindOutputToDevice(name, OnTheCard(name) ? _cardMemory! : hostMemoryInfo);
                    continue;
                }
                (written ??= new bool[outputNames.Count])[o] = true;
            }

            // The binding holds the feeds' raw handles, which `inputs` holds the wrappers of.
            results = Invoke(runSettings, inputs, runOptions => _session.RunWithBoundResults(runOptions, binding));
        }
        finally
        {
            if (placed is not null)
                foreach (var memory in placed) memory.Dispose();
        }

        // RunWithBoundResults returns the bound outputs in the binding's own order, which is the
        // order they were bound in -- ask it rather than assume, and hand them back in the order
        // the caller named.
        var boundNames = binding.GetOutputNames();
        // Nothing owns these values until each is wrapped and handed to a TensorData, and the
        // collection is deliberately not disposed -- its Dispose would dispose the values in it,
        // which are what this returns -- so anything that goes wrong between here and the return
        // leaks an allocation apiece. Establish the shape first, and dispose the lot if it is not
        // what it must be.
        if (boundNames.Length != results.Count || results.Count != outputNames.Count)
        {
            foreach (var value in results) value.Dispose();
            throw new InvalidOperationException(
                $"The run bound {boundNames.Length} outputs and returned {results.Count} values for "
                + $"{outputNames.Count} requested names; they must agree one for one.");
        }

        // Bound in the caller's order, they come back in it; each is looked up by name where it does
        // not.
        Dictionary<string, OrtValue>? byName = null;
        var values = new OrtValue[outputNames.Count];
        for (int i = 0; i < values.Length; i++)
        {
            if (string.Equals(boundNames[i], outputNames[i], StringComparison.Ordinal))
            {
                values[i] = results[i];
                continue;
            }
            if (byName is null)
            {
                byName = new Dictionary<string, OrtValue>(results.Count, StringComparer.Ordinal);
                for (int b = 0; b < boundNames.Length; b++)
                    byName[boundNames[b]] = results[b];
            }
            // Same reason as the count check above, and the same handling: a name that does not
            // come back is a bad run, not an excuse to drop every allocation it made.
            if (!byName.TryGetValue(outputNames[i], out var value))
            {
                foreach (var orphan in results) orphan.Dispose();
                throw new InvalidOperationException(
                    $"The run bound no output named '{outputNames[i]}'; it bound "
                    + $"{string.Join(", ", boundNames)}.");
            }
            values[i] = value;
        }
        return Owned(outputNames, values, written);
    }

    /// <summary>
    /// <paramref name="values"/>, the run's outputs in <paramref name="names"/>' order, as values
    /// of their own: each one bound to a value of its own (<paramref name="written"/>) as it is, and
    /// every other one copied out of the session's arena, on either device, which is then let go of.
    /// A tensor is copied byte for byte where it is, into this runtime's own allocator for its device
    /// (<see cref="OrtBackend.OutputAllocator"/>); a string tensor element by element, and a sequence
    /// as a sequence of copies of its elements, in host memory where the run left them.
    ///
    /// <para>The copies on the card are made on the CUDA runtime's own stream, which nothing orders
    /// the session's next use of the block after: so they are all waited for, once, before any block
    /// they read is let go of.</para>
    /// </summary>
    /// <exception cref="InvalidOperationException">A copy failed. Every value is released, the
    /// copies made so far with them.</exception>
    private IReadOnlyList<IShorokooTensorValue> Owned(
        IReadOnlyList<string> names, OrtValue[] values, bool[]? written)
    {
        var owned = new OrtTensorValue[values.Length];
        var copied = new bool[values.Length];
        var waitForTheCard = false;
        try
        {
            for (int i = 0; i < values.Length; i++)
            {
                var copy = written?[i] == true
                    ? null
                    : CopyOut(values[i], OnTheCard(names[i]), ref waitForTheCard);
                copied[i] = copy is not null;
                owned[i] = copy ?? new OrtTensorValue(values[i]);
            }
            if (waitForTheCard && CudaInterop.Synchronize() is not 0)
                throw new InvalidOperationException(
                    $"Copying a run's outputs out of the arena of a session of {_backend.Description} "
                    + "failed while the copies were waited for.");
        }
        catch
        {
            for (int i = 0; i < values.Length; i++)
            {
                if (copied[i]) owned[i].Dispose();
                values[i].Dispose();
            }
            throw;
        }
        for (int i = 0; i < values.Length; i++)
            if (copied[i]) values[i].Dispose();
        return owned;
    }

    /// <summary>
    /// A copy of <paramref name="value"/> in memory of its own, or null for a value that is neither a
    /// tensor nor a sequence, which is left as it is: on the card where the run left it there
    /// (<paramref name="card"/>), and in host memory otherwise. <paramref name="onCard"/> is set where
    /// the copy was made on the card, and has to be waited for.
    /// </summary>
    private OrtTensorValue? CopyOut(OrtValue value, bool card, ref bool onCard)
    {
        if (value.OnnxType == OnnxValueType.ONNX_TYPE_SEQUENCE) return CopySequence(value);
        if (value.OnnxType != OnnxValueType.ONNX_TYPE_TENSOR) return null;
        var info = value.GetTensorTypeAndShape();
        return info.ElementDataType == TensorElementType.String
            ? CopyStrings(value, info.Shape)
            : CopyTensor(value, info.ElementDataType, info.Shape, card, ref onCard);
    }

    private unsafe OrtTensorValue CopyTensor(
        OrtValue value, TensorElementType elementType, long[] shape, bool card, ref bool onCard)
    {
        var copy = new OrtTensorValue(OrtValue.CreateAllocatedTensorValue(OutputAllocator(card), elementType, shape));
        try
        {
            var bytes = checked((long)value.GetTensorSizeInBytes());
            if (bytes > 0)
            {
                var from = OrtBackend.AddressOf(value);
                var to = OrtBackend.AddressOf(copy.Inner);
                if (!card)
                    Buffer.MemoryCopy((void*)from, (void*)to, bytes, bytes);
                else if (CudaInterop.CopyDeviceToDevice(to, from, bytes) is not 0)
                    throw new InvalidOperationException(
                        $"Copying an output ({string.Join('x', shape)}:{elementType}) out of the arena of a "
                        + $"session of {_backend.Description} failed. The CUDA runtime is what performs the "
                        + "copy.");
                onCard |= card;
            }
            // The addresses were the values' last reads, so the JIT may retire them before the copy
            // is done, and a collection free a buffer under it (Shorokoo/Shorokoo#178).
            GC.KeepAlive(value);
            GC.KeepAlive(copy);
            return copy;
        }
        catch
        {
            copy.Dispose();
            throw;
        }
    }

    private static OrtTensorValue CopyStrings(OrtValue value, long[] shape)
    {
        var strings = value.GetStringTensorAsArray();
        GC.KeepAlive(value);
        var copy = OrtValue.CreateTensorWithEmptyStrings(OrtAllocator.DefaultInstance, shape);
        try
        {
            for (int i = 0; i < strings.Length; i++)
                copy.StringTensorSetElementAt(strings[i].AsSpan(), i);
        }
        catch
        {
            copy.Dispose();
            throw;
        }
        return new OrtTensorValue(copy);
    }

    /// <summary>A sequence of copies of <paramref name="value"/>'s elements, each read out into
    /// memory of its own (<see cref="OrtTensorValue.GetValue"/>) — or null for an empty one, which
    /// holds no memory to copy.</summary>
    private static OrtTensorValue? CopySequence(OrtValue value)
    {
        var count = value.GetValueCount();
        if (count == 0) return null;
        var elements = new List<OrtValue>(count);
        try
        {
            for (int i = 0; i < count; i++) elements.Add(value.GetValue(i, OrtAllocator.DefaultInstance));
            GC.KeepAlive(value);
            // ORT moves the elements into the sequence, which frees them with itself, and empties the
            // list only where it succeeds -- so what is left in it is what a failure leaves here.
            return new OrtTensorValue(OrtValue.CreateSequence(elements));
        }
        catch
        {
            foreach (var element in elements) element.Dispose();
            throw;
        }
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
    /// This session's own allocator for the arena the figures come from: the CUDA device arena on
    /// a GPU backend, the session's CPU arena otherwise. It has to come from the session —
    /// <c>OrtAllocator.DefaultInstance</c> is the plain CPU allocator, which implements no
    /// statistics and answers with nothing at all.
    ///
    /// <para>Null on a build that has no such allocator to give, which is what a CUDA arena asked
    /// for on a session with no CUDA provider is: ORT refuses it by name, and that refusal is a
    /// backend without the figures rather than a failure of the run.</para>
    /// </summary>
    private OrtAllocator? CreateArenaAllocator()
    {
        OrtMemoryInfo? owned = null;
        try
        {
            if (_cudaDeviceId is { } device) owned = CudaArenaMemoryInfo(device);
            var allocator = new OrtAllocator(_session, owned ?? OrtMemoryInfo.DefaultInstance);
            // The field assignment is what roots `owned` across the constructor above: ORT takes
            // the info as a bare IntPtr, so without a read of the local after the call the JIT
            // retires it at the .Handle read and its critical finalizer can free the info while
            // OrtCreateAllocator is still reading it. It is also why it is assigned only here --
            // a refusal above leaves nothing for the disposal to have to skip -- and never
            // DefaultInstance, which is ORT's own shared singleton and not ours to release.
            _ownedArenaMemoryInfo = owned;
            return allocator;
        }
        catch (Exception)
        {
            owned?.Dispose();
            return null;
        }
    }

    /// <summary>ORT's memory info for a CUDA device arena. <c>CudaPinned</c> is the pinned host
    /// arena that host-to-device copies stage through and is a different allocator —
    /// <see cref="CudaPinnedArenaMemoryInfo"/>.</summary>
    private static OrtMemoryInfo CudaArenaMemoryInfo(int deviceId)
    {
        return new OrtMemoryInfo("Cuda", OrtAllocatorType.ArenaAllocator, deviceId, OrtMemType.Default);
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
            // The field assignment roots `owned` across the constructor, exactly as above.
            _ownedPinnedMemoryInfo = owned;
            return allocator;
        }
        catch (Exception)
        {
            owned?.Dispose();
            return null;
        }
    }

    public ArenaStatistics? ReadArenaStatistics()
        => _arenaAllocator.Value is { } allocator ? OrtArenaStats.Read(allocator) : null;

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
    internal static CancellationTokenRegistration AbortWhenCancelled(
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

    /// <summary>Applies what <i>this</i> run runs with. The settings arrive per call rather than
    /// being held by the session, which is ORT's own shape for them: turning arena shrinkage on
    /// takes effect on a session that is already compiled, and on that run alone. Applied by both
    /// run paths, because a run that leaves its outputs on a card is the one that most wants the
    /// arena it keeps them in bounded. It configures options the caller owns rather than returning
    /// new ones, so the handle stays inside a `using` at each call site.</summary>
    private void ConfigureRun(RunOptions runOptions, RunSettings runSettings)
    {
        if (OrtBackend.ArenaShrinkageRunConfig(_cudaDeviceId, runSettings.ShrinkArenaAfterRun)
            is { } arena)
            runOptions.AddRunConfigEntry("memory.enable_memory_arena_shrinkage", arena);
    }

    public void Dispose()
    {
        if (_arenaAllocator.IsValueCreated)
        {
            // The allocator first: it was built over this memory info and takes it as a bare
            // pointer, so the info outlives it by one statement rather than the other way round.
            _arenaAllocator.Value?.Dispose();
            _ownedArenaMemoryInfo?.Dispose();
        }
        if (_pinnedAllocator.IsValueCreated)
        {
            _pinnedAllocator.Value?.Dispose();
            _ownedPinnedMemoryInfo?.Dispose();
        }
        _session.Dispose();
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
