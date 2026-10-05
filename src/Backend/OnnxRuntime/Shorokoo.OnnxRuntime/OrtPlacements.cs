using System.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using Shorokoo.Core.Backends;
using Shorokoo.Core.Factory.IR;

namespace Shorokoo.OnnxRuntime;

/// <summary>
/// Where a session's runs that consume inputs place their values in the memory of what they consume
/// (<see cref="PlacementProof"/>), and the sessions that carry the placements out.
///
/// <para><b>Per run signature.</b> A placement needs the set of inputs a run consumes and the shape
/// of every input, which only a run knows, so placements are planned per signature — those, and the
/// outputs asked for — and kept: a session of its own (a <i>variant</i>) for each signature that
/// places anything, built over the model with every placed value made a graph output, so a binding can
/// hand ONNX Runtime the range each one goes to, and charging the session's own allocator accounts.</para>
///
/// <para><b>Built from the graph that runs.</b> The plan is made over the graph ONNX Runtime runs
/// for this session — written out once, with its initializers, by the session itself as it was built
/// where it wrote that graph out anyway — to prove the outputs it writes into its inputs, or to share
/// the weights it carries with its variants (<see cref="OrtBackend"/>, a model over
/// <see cref="ModelBytesKept"/> on a card) — and otherwise by a probe build of the model. Where
/// every value the plan places is a value of the model as handed over, the variant is first built
/// over that model, optimized as the plain session is, which runs its nodes in the plain session's
/// order; otherwise, or where that variant does not hold the plan, it is built from the graph that
/// runs itself, with ONNX Runtime's optimizations off: its fusions are made already, so a value only
/// a fusion makes (a fused layer normalization's sum) is a value the variant can bind, and nothing is
/// rewritten again. Either way the plan is proved again over the graph the variant writes out as it
/// is built, and the variant's compute nodes must be the plain session's, operator for
/// operator.</para>
///
/// <para><b>The model is kept</b> to build the variants, and the probe where there is one, from: in memory up to
/// <see cref="ModelBytesKept"/>, and a larger one in a file of its own, written as the session is
/// built and deleted with it, so that its weights are not held a second time — or not at all where
/// the session wrote the graph it runs itself. A variant of a larger model is built from the graph
/// that runs alone, which reads its weights from the files written beside that graph rather than from
/// a copy of the model, or from the copies the session shares with it.</para>
///
/// <para><b>In the order the session runs.</b> ONNX Runtime runs a session's nodes one at a time
/// (<see cref="OrtBackend.Configure"/> asks for sequential execution), on each of its streams in the
/// order of the graph the session writes out: the CPU provider's on one stream, and on a card every
/// node of the CUDA provider's — every node that reads or writes the card's memory, a copy to or from
/// the host included — on another, the provider's few host nodes on a stream of their own. A
/// session's graph is read from what it wrote out, the variant's from its own build, so the proof
/// takes a node listed before another as having run before it, and a value goes into a range once
/// everything reading what the range held has run, whether or not the graph's edges order the two.
/// The graph handed over is ordered afresh by ONNX Runtime and is planned over only to tell whether
/// anything could be placed at all.</para>
///
/// <para><b>Decided before the first run.</b> Whether a plan pays is read off the proof's model of
/// the memory a run holds (<see cref="PlacementProof.ModelledPeak"/>), its kernels' own scratch
/// counted as the allocator counts it (<see cref="OrtRunMemory"/>): with every placed value in its
/// block against with none, less what the variants hold of their own. A signature's first run is
/// placed already where that pays, so a run that fits only placed runs on its first call. The first
/// placed run is measured, and a plan whose run, with what its variant holds, did not save the
/// margin against the modelled plain run is let go of for the runs after it.</para>
/// </summary>
internal sealed class OrtPlacements : IDisposable
{
    /// <summary>The largest model a session keeps in memory to build variants from; a larger one it
    /// keeps in a file.</summary>
    internal const int ModelBytesKept = 16 << 20;

    /// <summary>The file in a folder of a session's placements that marks it as theirs.</summary>
    internal const string KeptLockFile = "kept.lock";

    /// <summary>How many signatures a session plans for; a run of any other runs unplaced.</summary>
    private const int MostSignatures = 8;

    /// <summary>
    /// Builds a session over a model — <paramref name="model"/>, or where that is null the one in
    /// <paramref name="modelFile"/>, which ONNX Runtime reads itself — writing the graph it will run
    /// into the folder <paramref name="optimizedDirectory"/> names where one is. A model read with
    /// its initializers from files in <paramref name="externalDataDirectory"/> — a graph ONNX Runtime
    /// wrote out — is built with ONNX Runtime's optimizations off. A <paramref name="variant"/>
    /// charges the session's own allocator accounts; a probe, built and let go of, accounts of its
    /// own.
    /// </summary>
    internal delegate OrtSession VariantBuilder(
        byte[]? model, string? modelFile, string? optimizedDirectory, string? externalDataDirectory, bool variant);

    // The model as handed over: in memory where it is small, and otherwise in a file of its own.
    private readonly byte[]? _model;
    // The files kept on disk to build from: the model, where it is large, and the graph ONNX Runtime
    // runs with its initializers.
    private readonly KeptFiles _files = new();
    private readonly object _sources = new();
    private readonly VariantBuilder _build;
    private readonly OrtBackend _backend;
    // What the session and its variants hold of the allocator accounts they share, read either side
    // of building a variant to tell what the variant holds of its own.
    private readonly Func<long> _held;
    private readonly object _gate = new();
    private readonly Dictionary<string, Entry> _entries = new(StringComparer.Ordinal);
    private ModelProto? _original;
    // The graph ONNX Runtime runs for the plain session, as the session's build or a probe build wrote
    // it out; the folder it is in with its initializers is kept for the session's life to build
    // variants from.
    private ModelProto? _runs;
    private string? _broken;
    private bool _disposed;

    /// <summary>
    /// The placements of a session that wrote the graph it runs into <paramref name="runsDirectory"/>
    /// as it was built (<paramref name="runs"/>), its larger initializers in a file beside it, and
    /// reads the initializers <paramref name="sharedWeights"/> names from copies the variants are
    /// handed too: the variants are built from that graph, with no probe, and the folder is kept,
    /// and deleted with these.
    /// </summary>
    internal OrtPlacements(
        string runsDirectory, ModelProto runs, IReadOnlySet<string> sharedWeights, VariantBuilder build, OrtBackend backend,
        Func<long> held)
    {
        _files.KeepRuns(runsDirectory);
        _runs = runs;
        _sharedWeights = sharedWeights;
        _build = build;
        _backend = backend;
        _held = held;
    }

    // The initializers the session and its variants read from copies the session holds, which no
    // variant holds of its own.
    private readonly IReadOnlySet<string> _sharedWeights = new HashSet<string>();

    /// <summary>A graph ONNX Runtime wrote out as it built a session, and the folder it is in with its
    /// larger initializers.</summary>
    internal sealed record Written(string Directory, ModelProto Runs);

    /// <summary>
    /// The placements of a session built from <paramref name="model"/>: the variants are built from
    /// it, or from the graph ONNX Runtime runs for the session — <paramref name="written"/> where the
    /// session's build wrote it out, the folder then kept, and deleted with these; otherwise written
    /// out by a probe build the first time a plan asks for it.
    /// </summary>
    internal OrtPlacements(byte[] model, Written? written, VariantBuilder build, OrtBackend backend, Func<long> held)
    {
        _build = build;
        if (written is not null)
        {
            _files.KeepRuns(written.Directory);
            _runs = written.Runs;
        }
        if (model.Length <= ModelBytesKept)
            _model = model;
        else
            try
            {
                _files.KeepModel(model);
            }
            catch (Exception unwritable) when (unwritable is IOException or UnauthorizedAccessException)
            {
                // Placing is a saving and never a requirement: a model that cannot be kept is not
                // placed in.
                _broken = $"the model could not be kept in a file: {unwritable.Message}";
                _files.Delete();
            }
        _backend = backend;
        _held = held;
    }

    /// <summary>Where a signature is: not yet planned, or settled — its runs placed, or plain.</summary>
    internal enum Stage { Unplanned, Adopted, Refused }

    /// <summary>One run signature's placements, what the model said of them, and how they
    /// measured.</summary>
    internal sealed class Entry
    {
        internal Stage Stage;
        // The most the run holds at once beyond its inputs, as the proof models it: plain, and with
        // every placement in its block.
        internal long PredictedPlainPeak;
        internal long PredictedPlacedPeak;
        // The most the first placed run asked of the session's allocator, measured.
        internal long PlacedPeak;
        internal long VariantHeld;
        internal IReadOnlyList<Placement> Plan = [];
        internal Dictionary<string, (long[] Shape, ShorokooTensorElementType Type)> Shapes = new(StringComparer.Ordinal);
        internal OrtSession? Variant;
        internal string? Refusal;
        // Taken while the signature is planned, so that planning holds back its own runs alone; how
        // many runs are on the variant now; and whether its first placed run was measured.
        internal readonly object Planning = new();
        internal int Running;
        internal bool Measured;

        // What the entry was planned over, for a measurement to read: the graph ONNX Runtime runs
        // for the plain session, the shapes the run's inputs came in, and its blocks.
        internal GraphProto? Graph;
        internal GraphProto? VariantGraph;
        // Whether the variant was built from the graph ONNX Runtime wrote out rather than from the
        // model as handed over.
        internal bool VariantFromRuns;
        internal IReadOnlyDictionary<string, (long[] Shape, int ElementType)> Given = new Dictionary<string, (long[], int)>();
        internal IReadOnlyDictionary<string, long> BlockBytes = new Dictionary<string, long>();
    }

    /// <summary>Told of every signature as it is settled — adopted or refused — for a measurement to
    /// read; null where nothing listens.</summary>
    internal static Action<Entry>? Settled;

    /// <summary>For a test: asked as a placed run of this thread is about to run on its variant, and
    /// what it answers is thrown there, as a failure of the run would be; null on every thread but one
    /// a test sets it on.</summary>
    [ThreadStatic]
    internal static Func<Exception?>? PlacedRunFault;

    /// <summary>For a test: asked as a signature of this thread's run starts to be planned, and what
    /// it answers is thrown there, as a failure of the planning would be; null on every thread but
    /// one a test sets it on.</summary>
    [ThreadStatic]
    internal static Func<Exception?>? PlanningFault;

    /// <summary>Every signature planned so far, for a test to read.</summary>
    internal IReadOnlyList<Entry> Entries
    {
        get { lock (_gate) return [.. _entries.Values]; }
    }

    /// <summary>
    /// The memory a run may place values in, by input name: each value the run consumed that it was
    /// fed under that one name, a tensor of this runtime of fixed-width elements, into which no
    /// marked output is written, of at least <see cref="PlacementProof.Smallest"/> bytes — the least
    /// a placed value is, so a smaller one holds none. A run consuming none of those plans nothing.
    /// </summary>
    internal static Dictionary<string, OrtTensorValue> Blocks(
        IReadOnlyDictionary<string, IShorokooTensorValue> inputs, IReadOnlyCollection<IShorokooTensorValue> consumed,
        IReadOnlySet<string> aliasedInputs)
    {
        var blocks = new Dictionary<string, OrtTensorValue>(StringComparer.Ordinal);
        if (consumed.Count == 0) return blocks;
        var handed = new HashSet<IShorokooTensorValue>(consumed, ReferenceEqualityComparer.Instance);
        var fedAs = new Dictionary<IShorokooTensorValue, int>(ReferenceEqualityComparer.Instance);
        foreach (var value in inputs.Values) fedAs[value] = fedAs.GetValueOrDefault(value) + 1;
        foreach (var (name, value) in inputs)
        {
            if (!handed.Contains(value) || fedAs[value] != 1 || aliasedInputs.Contains(name)) continue;
            if (value is not OrtTensorValue { ValueType: ShorokooOnnxValueType.Tensor } own) continue;
            if (PlacementShapes.ElementBytes((int)own.ElementType) == 0 || BytesOf(own) < PlacementProof.Smallest) continue;
            blocks[name] = own;
        }
        return blocks;
    }

    private static long BytesOf(OrtTensorValue value)
        => value.ReadShape.Aggregate(1L, (a, d) => a * d) * PlacementShapes.ElementBytes((int)value.ElementType);

    /// <summary>
    /// The entry for a run that may place values in <paramref name="blocks"/>: the one its signature
    /// has, or one made the first time its signature is seen; null for a signature not seen before
    /// where this session has stopped planning or plans for as many signatures as it will.
    /// <paramref name="aliasedInputs"/> names the inputs the run writes an output into, which no plan
    /// places.
    /// </summary>
    internal Entry? EntryFor(
        IReadOnlyDictionary<string, IShorokooTensorValue> inputs, Dictionary<string, OrtTensorValue> blocks,
        IReadOnlyList<string> outputNames, IReadOnlySet<string> aliasedInputs)
    {
        if (blocks.Count == 0) return null;
        var key = new System.Text.StringBuilder();
        foreach (var (name, value) in inputs.OrderBy(i => i.Key, StringComparer.Ordinal))
        {
            // A block whose parts go back once no output stands on them plans otherwise than one
            // held whole, so the two are signatures of their own; and an input an output is written
            // into plans otherwise than one only read, whose output a plan may place.
            key.Append(name).Append(
                blocks.TryGetValue(name, out var block) ? OrtBackend.RangesGoBack(block) ? "!!" : "!"
                : aliasedInputs.Contains(name) ? "=" : ":");
            if (value is OrtTensorValue { ValueType: ShorokooOnnxValueType.Tensor } tensor)
                key.Append((int)tensor.ElementType).Append('[').AppendJoin(',', tensor.ReadShape).Append(']');
            key.Append(';');
        }
        key.Append("->").AppendJoin(';', outputNames);
        lock (_gate)
        {
            if (_disposed) return null;
            // A session that stopped planning still runs what it settled: an adopted signature
            // placed, on the variant it holds for it.
            if (_entries.TryGetValue(key.ToString(), out var entry)) return entry;
            if (_broken is not null || _entries.Count >= MostSignatures) return null;
            return _entries[key.ToString()] = new Entry();
        }
    }

    /// <summary>
    /// Runs <paramref name="owner"/> on a signature <paramref name="entry"/> stands for, planning it
    /// first where nothing has: placed on its variant once adopted, plain once refused. Planning a
    /// signature holds back only its own runs, and stops where <paramref name="cancellation"/> is
    /// cancelled, leaving the signature to be planned by a run after it. The first placed run to end
    /// is measured; a placed run that fails refuses the signature, so the runs after it run plain,
    /// and a refused signature's variant goes once no run is on it. <paramref name="aliasedOutputs"/>
    /// names the outputs the run writes into the inputs they are marked for, which no plan places.
    /// <paramref name="kept"/> answers the consumed values a placed output stands on, which the run
    /// does not release: the block each became releases it with its last output.
    /// </summary>
    internal IReadOnlyList<IShorokooTensorValue> Run(
        OrtSession owner, Entry entry, IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
        Dictionary<string, OrtTensorValue> blocks, IReadOnlyList<string> outputNames,
        Func<IReadOnlyList<IShorokooTensorValue>> plain,
        Func<OrtSession, IReadOnlyDictionary<string, OrtSession.PlacedBinding>, IReadOnlyList<IShorokooTensorValue>> placed,
        IReadOnlyCollection<string> aliasedOutputs, CancellationToken cancellation, out HashSet<IShorokooTensorValue>? kept)
    {
        kept = null;
        lock (entry.Planning)
        {
            if (entry.Stage == Stage.Unplanned)
            {
                cancellation.ThrowIfCancellationRequested();
                string? broken;
                lock (_gate) broken = _broken;
                // A session that stopped planning plans no signature left unplanned either.
                if (broken is not null) Refuse(entry, broken);
                else Prepare(owner, entry, inputs, blocks, outputNames, aliasedOutputs, cancellation);
                Settled?.Invoke(entry);
            }
        }
        OrtSession? variant = null;
        var first = false;
        lock (_gate)
        {
            if (entry.Stage == Stage.Adopted && entry.Variant is { } adopted)
            {
                variant = adopted;
                entry.Running++;
                // Taken by the one run measuring, as its allocator account reads one run at a time.
                first = !entry.Measured;
                entry.Measured = true;
            }
        }
        if (variant is null) return plain();
        try
        {
            var results = RunPlaced(variant, entry, blocks, outputNames, placed, first, out kept, out var peak);
            if (first)
                lock (_gate)
                {
                    entry.PlacedPeak = Math.Max(peak, 1);
                    // The model was wrong where the placed run asked for more than it said the plain
                    // one would — both with what the kernels take for their own use besides the
                    // values: the runs after this one run plain.
                    if (entry.Stage == Stage.Adopted && peak + entry.VariantHeld + Margin(entry.PredictedPlainPeak) > entry.PredictedPlainPeak)
                        Refuse(entry, $"placed, the run asked {peak} bytes against {entry.PredictedPlainPeak} modelled plain");
                }
            return results;
        }
        catch (OperationCanceledException) when (first)
        {
            // A run stopped before it ended measured nothing: the next one is measured instead.
            lock (_gate) entry.Measured = false;
            throw;
        }
        catch (Exception failure) when (failure is not OperationCanceledException)
        {
            lock (_gate)
                if (entry.Stage == Stage.Adopted)
                    Refuse(entry, $"the placed run failed: {failure.GetType().Name}: {failure.Message}");
            throw;
        }
        finally
        {
            lock (_gate)
            {
                entry.Running--;
                if (entry.Running == 0 && (entry.Stage != Stage.Adopted || _disposed)) LetGoOfVariant(entry);
            }
        }
    }

    /// <summary>Releases <paramref name="value"/> through <paramref name="backend"/>: what a block
    /// over the value's memory does with its last lease. Made apart from any run so that it holds
    /// those two alone — a placed output keeps its block, and so this, for as long as it lives.</summary>
    private static Action ReleaseThrough(IShorokooBackend backend, OrtTensorValue value) => () => backend.Release(value);

    /// <summary>The least a plan must save to be kept: a mebibyte, or a sixty-fourth of the plain
    /// run where that is more.</summary>
    private static long Margin(long plain) => Math.Max(1L << 20, plain / 64);

    /// <summary>A run on <paramref name="variant"/> with every placement of
    /// <paramref name="entry"/> bound to its range, measuring the most it asked of the variant's
    /// allocator.</summary>
    private IReadOnlyList<IShorokooTensorValue> RunPlaced(
        OrtSession variant, Entry entry, Dictionary<string, OrtTensorValue> blocks, IReadOnlyList<string> outputNames,
        Func<OrtSession, IReadOnlyDictionary<string, OrtSession.PlacedBinding>, IReadOnlyList<IShorokooTensorValue>> placed,
        bool measure, out HashSet<IShorokooTensorValue>? kept, out long peak)
    {
        peak = 0;
        kept = new HashSet<IShorokooTensorValue>(ReferenceEqualityComparer.Instance);
        var asked = new HashSet<string>(outputNames, StringComparer.Ordinal);
        var shared = new Dictionary<string, (SharedBlock Block, long Base)>(StringComparer.Ordinal);
        var bindings = new Dictionary<string, OrtSession.PlacedBinding>(StringComparer.Ordinal);
        var temporaries = new List<OrtValue>();
        try
        {
            foreach (var placement in entry.Plan)
            {
                var owner = blocks[placement.Block];
                var (shape, type) = entry.Shapes[placement.Value];
                if (asked.Contains(placement.Value))
                {
                    if (!shared.TryGetValue(placement.Block, out var block))
                    {
                        block = owner.Range is { } range
                            ? (range.Block, range.Offset)
                            : (OrtBackend.BlockOver(owner, BytesOf(owner), ReleaseThrough(_backend, owner)), 0L);
                        shared[placement.Block] = block;
                    }
                    var view = OrtBackend.View(owner, placement.Offset, type, shape, placement.Bytes, block.Block, block.Base + placement.Offset);
                    if (owner.Range is null) kept.Add(owner);
                    bindings[placement.Value] = new OrtSession.PlacedBinding(view.Inner, view);
                }
                else
                {
                    var over = OrtBackend.Over(owner, placement.Offset, type, shape, placement.Bytes);
                    temporaries.Add(over);
                    bindings[placement.Value] = new OrtSession.PlacedBinding(over, null);
                }
            }
            if (PlacedRunFault?.Invoke() is { } fault) throw fault;
            var results = measure ? variant.Measured(() => placed(variant, bindings), out peak) : placed(variant, bindings);
            // The run is over: what of each block no output it handed back stands on goes back.
            foreach (var (block, _) in shared.Values) block.Settle();
            return results;
        }
        catch
        {
            // The views handed back for outputs hold the leases: letting them go lets each block go,
            // and with it the consumed value it releases.
            foreach (var binding in bindings.Values) binding.Returned?.Dispose();
            throw;
        }
        finally
        {
            foreach (var value in temporaries) value.Dispose();
        }
    }

    /// <summary>
    /// Plans <paramref name="entry"/> and, where the plan pays, builds its variant: the plan over the
    /// graph ONNX Runtime runs, weighed by the proof's model of the memory a run holds, the variant
    /// built over the model as handed over where it can bind the plan, and from the graph that runs
    /// where it cannot or did not hold it, and the plan proved again over the graph the variant runs,
    /// until the two agree. Refuses the entry, for good, where nothing can be placed, where placing
    /// saves too little, where the variant's compute nodes differ from the plain session's, or where
    /// anything fails on the way — the run then runs as it would have with no placing — and stops
    /// planning for the session where it ran out of memory, as planning another signature would; the
    /// signatures adopted already go on placed.
    /// Leaves the entry to be planned again where <paramref name="cancellation"/> stopped it. Under
    /// the lock.
    /// </summary>
    private void Prepare(
        OrtSession owner, Entry entry, IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
        Dictionary<string, OrtTensorValue> blocks, IReadOnlyList<string> outputNames, IReadOnlyCollection<string> aliasedOutputs,
        CancellationToken cancellation)
    {
        try
        {
            if (PlanningFault?.Invoke() is { } fault) throw fault;
            var given = new Dictionary<string, (long[] Shape, int ElementType)>(StringComparer.Ordinal);
            foreach (var (name, value) in inputs)
                if (value is OrtTensorValue { ValueType: ShorokooOnnxValueType.Tensor } tensor)
                    given[name] = (tensor.ReadShape, (int)tensor.ElementType);
            var blockBytes = blocks.ToDictionary(b => b.Key, b => BytesOf(b.Value), StringComparer.Ordinal);
            var givingBack = blocks.Where(b => OrtBackend.RangesGoBack(b.Value)).Select(b => b.Key).ToHashSet(StringComparer.Ordinal);
            entry.Given = given;
            entry.BlockBytes = blockBytes;
            // The model as handed over first, which costs no build: what it places nothing in, the
            // graph ONNX Runtime makes of it does not either, short of a rewrite freeing a range --
            // and a session builds nothing for such a run. Not where the graph that runs is at hand
            // already and the model is kept in a file, which reading would copy for nothing.
            var original = HandedOverToPlan();
            if (original is not null
                && (original.Graph is not { } handed
                    || handed.Nodes.Count > PlacementProof.MostNodes
                    || new PlacementProof(handed, blockBytes, PlacementShapes.Evaluate(handed, given), memory: LayoutOf(handed, owner).Memory, runsInOrder: true)
                        .Plan(PlacementProof.Smallest, PlacementProof.IdleOutputBytes, givingBack).Count == 0))
            {
                Refuse(entry, "nothing to place in the graph handed over");
                return;
            }
            cancellation.ThrowIfCancellationRequested();
            var runsModel = RunGraph();
            var runs = runsModel.Graph!;
            entry.Graph = runs;
            if (runs.Nodes.Count > PlacementProof.MostNodes)
            {
                Refuse(entry, "too large a graph to place in");
                return;
            }
            var shapes = PlacementShapes.Evaluate(runs, given);
            var runsOutputs = runs.Outputs.Select(o => o.Name).ToHashSet(StringComparer.Ordinal);
            var (memory, scratch) = LayoutOf(runs, owner);
            var proof = new PlacementProof(runs, blockBytes, shapes, memory: memory, runsInOrder: true, scratch: node => scratch(node, shapes));
            var plan = proof.Prove(proof.Plan(PlacementProof.Smallest, PlacementProof.IdleOutputBytes, givingBack)
                .Where(p => (!runsOutputs.Contains(p.Value) || outputNames.Contains(p.Value)) && !aliasedOutputs.Contains(p.Value)));
            if (plan.Count == 0)
            {
                Refuse(entry, "nothing to place in the graph ONNX Runtime runs");
                return;
            }
            entry.PredictedPlainPeak = proof.ModelledPeak([]);
            entry.PredictedPlacedPeak = proof.ModelledPeak(plan);
            // What a variant would hold of its own: the initializers it reads from the graph's files,
            // every one but those the session is handed.
            var predictedHeld = OwnInitializerBytes(runsModel);
            if (!Pays(entry, predictedHeld))
            {
                Refuse(entry, $"placing saves too little: {entry.PredictedPlainPeak} bytes modelled plain, "
                              + $"{entry.PredictedPlacedPeak} placed, {predictedHeld} held by the variant");
                return;
            }
            // A variant over the model as handed over, optimized as the plain session is, runs its
            // nodes in the order the plain session does; it can only bind values of that model, and
            // where exposing them stops a fusion, it runs other operators than the plain session. A
            // variant over the graph ONNX Runtime wrote out binds any value of it, a fusion's too,
            // but ONNX Runtime orders that graph afresh as it loads it. So the first is tried where it
            // can be, and the second where it cannot or did not hold.
            var originalValues = original?.Graph?.Nodes.SelectMany(n => n.Outputs).ToHashSet(StringComparer.Ordinal);
            bool[] sources = _model is not null && originalValues is not null && plan.All(p => originalValues.Contains(p.Value)) ? [false, true] : [true];
            foreach (var fromRuns in sources)
            {
                var candidate = plan;
                for (int attempt = 0; attempt < 3; attempt++)
                {
                    cancellation.ThrowIfCancellationRequested();
                    var outcome = TryVariant(entry, fromRuns ? runsModel : original!, fromRuns, candidate, runs, runsOutputs, shapes, given, blockBytes);
                    if (outcome.Plan is null) break;
                    if (entry.Stage != Stage.Unplanned) return;
                    candidate = outcome.Plan;
                }
                if (entry.Stage != Stage.Unplanned) return;
            }
            Refuse(entry, entry.Refusal ?? "no variant held the plan");
        }
        catch (OperationCanceledException) when (cancellation.IsCancellationRequested)
        {
            // Stopped rather than failed: a run after this one plans the signature.
            throw;
        }
        catch (OutOfMemoryException exhausted)
        {
            // Placing is a saving and never a requirement: the run goes on as it would have with no
            // placing, its memory what it was before planning began.
            lock (_gate) _broken ??= $"planning ran out of memory: {exhausted.Message}";
            Refuse(entry, $"planning failed: {exhausted.GetType().Name}: {exhausted.Message}");
        }
        catch (Exception failure)
        {
            Refuse(entry, $"planning failed: {failure.GetType().Name}: {failure.Message}");
        }
    }

    /// <summary>
    /// Builds a variant over <paramref name="model"/> — the graph ONNX Runtime wrote out where
    /// <paramref name="fromRuns"/>, the model as handed over otherwise — binding <paramref name="plan"/>,
    /// and settles <paramref name="entry"/> where it can: adopted with the variant where the plan is
    /// proved again in full over the graph the variant runs and pays in the order it runs it,
    /// refused where it does not pay. Answers the placements it proved where only some held, to
    /// try again with; null where this variant cannot hold the plan — what it runs differs from the
    /// plain session, or it proves none — with <see cref="Entry.Refusal"/> saying why.
    /// </summary>
    private (IReadOnlyList<Placement>? Plan, bool _) TryVariant(
        Entry entry, ModelProto model, bool fromRuns, IReadOnlyList<Placement> plan, GraphProto runs, HashSet<string> runsOutputs,
        IReadOnlyDictionary<string, PlacementShapes.Value> shapes, IReadOnlyDictionary<string, (long[] Shape, int ElementType)> given,
        IReadOnlyDictionary<string, long> blockBytes)
    {
        var exposed = plan.Where(p => !runsOutputs.Contains(p.Value)).Select(p => p.Value).ToList();
        var directory = Path.Combine(Path.GetTempPath(), "shorokoo-placed-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        OrtSession? variant = null;
        try
        {
            var heldBefore = _held();
            variant = _build(WithOutputs(model, exposed, shapes), null, directory, fromRuns ? _files.Runs : null, true);
            var variantHeld = _held() - heldBefore;
            var graph = ReadOptimized(directory).Graph!;
            entry.VariantGraph = graph;
            if (!SameComputeNodes(runs, graph))
            {
                entry.Refusal = "exposing the placed values changed what ONNX Runtime runs";
                return (null, false);
            }
            var variantShapes = PlacementShapes.Evaluate(graph, given);
            var (memory, scratch) = LayoutOf(graph, variant);
            var variantProof = new PlacementProof(graph, blockBytes, variantShapes, runsOutputs, memory, runsInOrder: true,
                scratch: node => scratch(node, variantShapes));
            var proved = variantProof.Prove(plan
                .Where(p => variantShapes.TryGetValue(p.Value, out var v) && v.Bytes == p.Bytes && StatedAgrees(graph, p.Value, v, given)));
            if (proved.Count == 0)
            {
                entry.Refusal = "the variant's own graph proves none of the placements";
                return (null, false);
            }
            if (proved.Count < plan.Count) return (proved, false);
            entry.Plan = plan;
            entry.Shapes = plan.ToDictionary(
                p => p.Value, p => (variantShapes[p.Value].Shape, (ShorokooTensorElementType)variantShapes[p.Value].ElementType),
                StringComparer.Ordinal);
            entry.VariantHeld = variantHeld;
            // The variant's own order decides what it holds at once, which may not be the plain
            // session's: weighed again in it.
            entry.PredictedPlacedPeak = variantProof.ModelledPeak(plan);
            entry.VariantFromRuns = fromRuns;
            if (!Pays(entry, variantHeld))
            {
                Refuse(entry, $"placing saves too little: {entry.PredictedPlainPeak} bytes modelled plain, "
                              + $"{entry.PredictedPlacedPeak} placed in the variant's order, {variantHeld} held by the variant");
                return (null, false);
            }
            lock (_gate)
            {
                // A session let go of while this was planned keeps no variant: the finally below lets
                // go of it.
                if (_disposed)
                {
                    entry.Stage = Stage.Refused;
                    entry.Refusal = "the session was let go of";
                    return (null, false);
                }
                entry.Variant = variant;
                entry.Stage = Stage.Adopted;
            }
            variant = null;
            return (plan, true);
        }
        finally
        {
            variant?.Dispose();
            try { Directory.Delete(directory, recursive: true); } catch (Exception) { }
        }
    }

    /// <summary>Whether <paramref name="entry"/>'s plan saves, by the model, at least the margin
    /// beyond <paramref name="held"/>, what its variant holds of its own.</summary>
    private bool Pays(Entry entry, long held)
        => entry.PredictedPlainPeak - entry.PredictedPlacedPeak - held - OtherVariantsHeld(entry) >= Margin(entry.PredictedPlainPeak);

    /// <summary>What the variants of the signatures other than <paramref name="entry"/> adopted hold
    /// of their own: the session holds them all at once, so a plan pays for them too.</summary>
    private long OtherVariantsHeld(Entry entry)
    {
        lock (_gate) return _entries.Values.Where(e => e != entry && e.Stage == Stage.Adopted).Sum(e => e.VariantHeld);
    }

    /// <summary>The bytes of the initializers of <paramref name="model"/> a session built over it
    /// reads in itself: every one but those the session is handed (<see cref="SuppliedInitializer"/>),
    /// and those it reads from copies the session shares with its variants.</summary>
    private long OwnInitializerBytes(ModelProto model)
    {
        var handed = model.Graph!.Inputs.Select(i => i.Name).ToHashSet(StringComparer.Ordinal);
        long bytes = 0;
        foreach (var initializer in model.Graph.Initializers)
            if (!handed.Contains(initializer.Name) && !_sharedWeights.Contains(initializer.Name))
                bytes += (initializer.Dims ?? []).Aggregate(1L, (a, d) => a * d) * PlacementShapes.ElementBytes(initializer.data_type);
        return bytes;
    }

    /// <summary>Refuses <paramref name="entry"/>: its runs run plain from now on, and its variant
    /// goes as the last run on it ends, at once where none is.</summary>
    private static void Refuse(Entry entry, string why)
    {
        entry.Stage = Stage.Refused;
        entry.Refusal = why;
        if (entry.Running == 0) LetGoOfVariant(entry);
    }

    private static void LetGoOfVariant(Entry entry)
    {
        entry.Variant?.Dispose();
        entry.Variant = null;
    }

    /// <summary>
    /// How <paramref name="session"/> — this session or a variant of it — lays out the values of
    /// <paramref name="graph"/>, the graph it runs, as far as placing them goes, and what its kernels
    /// take for their own use while each node runs, given the shapes of the run's values. Memory
    /// ONNX Runtime's way, where on a card a node that runs on the host (<see cref="HostNodes"/>)
    /// writes into no range — its output is host memory, which a binding to a range on the card
    /// receives only as the run ends, over whatever the run wrote there since. Scratch as
    /// <see cref="OrtRunMemory"/> measures it in the memory the session's runs are measured in: of a
    /// card's nodes on a card, a host node taking none of the card's, and of every node on the host —
    /// so the run the model weighs takes what a run measured on the session's allocator does.
    /// </summary>
    private (PlacementMemory Memory, Func<NodeProto, IReadOnlyDictionary<string, PlacementShapes.Value>, long> Scratch) LayoutOf(
        GraphProto graph, OrtSession session)
    {
        if (!_backend.OnCard) return (PlacementMemory.OnnxRuntime, OrtRunMemory.HostScratch);
        var names = session.HostMemoryNames;
        var host = HostNodes(graph, names?.Inputs ?? graph.Inputs.Select(i => i.Name).ToHashSet(StringComparer.Ordinal), names?.Outputs);
        return (PlacementMemory.OnnxRuntime with { Writes = (node, _) => !host.Contains(node) },
            (node, shapes) => host.Contains(node) ? 0 : OrtRunMemory.CardScratch(node, shapes));
    }

    // The CUDA provider's operators reading nothing of the card's memory to write their output there:
    // a fill of a shape, a range, random values.
    private static readonly HashSet<string> CardGenerators = new(StringComparer.Ordinal)
    {
        "ConstantOfShape", "Range", "RandomNormal", "RandomUniform",
    };

    /// <summary>
    /// The nodes of <paramref name="graph"/>, a graph a session on a card runs, whose outputs are in
    /// host memory: a copy to the host, a shape or a size, and every node of the CPU provider's. ONNX
    /// Runtime hands a node values in its provider's memory: across a copy node where one provider's
    /// node reads another's output, and for a graph input, by copying the feed as the run starts —
    /// to the host where only CPU-provider nodes read it, and where nodes of both read it, to the
    /// host too, the card's readers then reading it through a copy onto the card.
    /// <paramref name="hostInputs"/> names the inputs the session reads in host memory, as it says
    /// where it reads each one; where it is not said, those a copy onto the card reads.
    /// <paramref name="hostOutputs"/> names the outputs it leaves in host memory, as it says, where
    /// the node writing one is a node of neither kind above.
    ///
    /// <para>So a node whose output a copy onto the card reads, or that writes an output left in
    /// host memory, is the CPU provider's, and so is every node writing what such a node reads; a
    /// node reading the output of a node of the card's, an input read on the card, or a copy onto the
    /// card is the CUDA provider's; and of the nodes reading none of those, only its generators,
    /// where no copy onto the card takes what they write.</para>
    /// </summary>
    internal static HashSet<NodeProto> HostNodes(GraphProto graph, IReadOnlySet<string>? hostInputs = null, IReadOnlySet<string>? hostOutputs = null)
    {
        var copiedOn = graph.Nodes.Where(n => n.OpType == "MemcpyFromHost").SelectMany(n => n.Inputs).ToHashSet(StringComparer.Ordinal);
        var producer = new Dictionary<string, NodeProto>(StringComparer.Ordinal);
        foreach (var node in graph.Nodes)
            foreach (var output in node.Outputs)
                if (output.Length > 0) producer[output] = node;
        static bool Copies(NodeProto node) => node.OpType is "MemcpyFromHost" or "MemcpyToHost" or "Shape" or "Size";
        var host = new HashSet<NodeProto>(ReferenceEqualityComparer.Instance);
        var pending = new Stack<NodeProto>(graph.Nodes.Where(n =>
            (n.OpType != "MemcpyToHost" && n.Outputs.Any(copiedOn.Contains))
            || (!Copies(n) && hostOutputs is not null && n.Outputs.Any(hostOutputs.Contains))));
        while (pending.TryPop(out var node))
        {
            if (!host.Add(node)) continue;
            foreach (var input in node.Inputs)
                if (input.Length > 0 && producer.TryGetValue(input, out var writer) && !Copies(writer))
                    pending.Push(writer);
        }
        var onCard = graph.Inputs.Select(i => i.Name)
            .Where(i => !copiedOn.Contains(i) && hostInputs?.Contains(i) != true).ToHashSet(StringComparer.Ordinal);
        foreach (var node in graph.Nodes)
        {
            var card = !host.Contains(node) && node.OpType switch
            {
                "MemcpyFromHost" => true,
                "MemcpyToHost" or "Shape" or "Size" => false,
                _ => node.Inputs.Any(onCard.Contains)
                     || (CardGenerators.Contains(node.OpType) && !node.Outputs.Any(copiedOn.Contains)),
            };
            if (card) onCard.UnionWith(node.Outputs);
            else host.Add(node);
        }
        return host;
    }

    /// <summary>The model as handed to the backend — or, where the session kept none, the graph it
    /// runs as it wrote it out — for a measurement to read.</summary>
    internal ModelProto OriginalModel
    {
        get { lock (_sources) return OriginalUnderLock(); }
    }

    /// <summary>The model as handed to the backend for a plan to look at before anything is built,
    /// parsed once — or null where the graph that runs is at hand already and the model is kept in
    /// no memory of the session's: reading it would cost a copy of the model for nothing.</summary>
    private ModelProto? HandedOverToPlan()
    {
        lock (_sources) return _model is null && _runs is not null ? null : OriginalUnderLock();
    }

    /// <summary>The model as handed to the backend, parsed once — or, where the session kept none,
    /// the graph it runs as it wrote it out. One kept in a file is kept parsed
    /// without the contents of its larger tensors, which the proof does not read: it is a graph to
    /// plan over, and never one to build from.</summary>
    private ModelProto OriginalUnderLock()
    {
        if (_original is not null) return _original;
        if (_model is null && _files.Model is null) return _original = _runs!;
        if (_model is not null)
        {
            using var stream = new MemoryStream(_model, writable: false);
            return _original = ProtoBuf.Serializer.Deserialize<ModelProto>(stream);
        }
        using var file = File.OpenRead(_files.Model!);
        return _original = WithoutLargeTensorContents(file);
    }

    /// <summary>The most bytes of a tensor's contents a model read to plan over keeps.</summary>
    private const int TensorContentsKept = 1024;

    /// <summary>
    /// The model <paramref name="source"/> holds, read without the contents of any of its tensors
    /// over <see cref="TensorContentsKept"/> bytes — its initializers', its constants', and those of
    /// its subgraphs — which are skipped over in the stream, never read: what is read is the size of
    /// the graph, not of the model's weights.
    /// </summary>
    internal static ModelProto WithoutLargeTensorContents(Stream source)
    {
        var lean = new MemoryStream();
        CopyLean(source, source.Length, lean, Message.Model);
        lean.Position = 0;
        return ProtoBuf.Serializer.Deserialize<ModelProto>(lean);
    }

    /// <summary>The ONNX messages <see cref="CopyLean"/> looks into, on the way to tensors.</summary>
    private enum Message { Model, Graph, Node, Attribute, Tensor }

    /// <summary>The message field <paramref name="field"/> of a <paramref name="parent"/> holds, where
    /// it is one on the way to a tensor: a model's graph; a graph's nodes and initializers; a node's
    /// attributes; an attribute's tensors and graphs.</summary>
    private static Message? Holds(Message parent, ulong field) => (parent, field) switch
    {
        (Message.Model, 7) => Message.Graph,
        (Message.Graph, 1) => Message.Node,
        (Message.Graph, 5) => Message.Tensor,
        (Message.Node, 5) => Message.Attribute,
        (Message.Attribute, 5 or 10) => Message.Tensor,
        (Message.Attribute, 6 or 11) => Message.Graph,
        _ => null,
    };

    /// <summary>Copies the fields of a <paramref name="message"/> from <paramref name="source"/> up to
    /// <paramref name="end"/> into <paramref name="target"/>, but for a tensor's contents — raw, or
    /// as packed numbers — over <see cref="TensorContentsKept"/> bytes.</summary>
    private static void CopyLean(Stream source, long end, Stream target, Message message)
    {
        while (source.Position < end)
        {
            var tag = ReadVarint(source);
            var (field, wire) = (tag >> 3, (int)(tag & 7));
            switch (wire)
            {
                case 0:
                    WriteVarint(target, tag);
                    WriteVarint(target, ReadVarint(source));
                    break;
                case 1 or 5:
                    WriteVarint(target, tag);
                    CopyBytes(source, target, wire == 1 ? 8 : 4);
                    break;
                case 2:
                    var length = (long)ReadVarint(source);
                    if (message == Message.Tensor && field is 4 or 5 or 7 or 9 or 10 or 11 && length > TensorContentsKept)
                    {
                        source.Seek(length, SeekOrigin.Current);
                        break;
                    }
                    WriteVarint(target, tag);
                    if (Holds(message, field) is { } inner)
                    {
                        var nested = new MemoryStream();
                        CopyLean(source, source.Position + length, nested, inner);
                        WriteVarint(target, (ulong)nested.Length);
                        nested.WriteTo(target);
                    }
                    else
                    {
                        WriteVarint(target, (ulong)length);
                        CopyBytes(source, target, length);
                    }
                    break;
                default:
                    throw new InvalidDataException($"The model holds a field of wire type {wire}, which ONNX does not use.");
            }
        }
    }

    private static ulong ReadVarint(Stream source)
    {
        ulong value = 0;
        for (int shift = 0; shift < 64; shift += 7)
        {
            var b = source.ReadByte();
            if (b < 0) throw new EndOfStreamException("The model ends inside a field.");
            value |= (ulong)(b & 0x7F) << shift;
            if (b < 0x80) return value;
        }
        throw new InvalidDataException("The model holds a number longer than ten bytes.");
    }

    private static void WriteVarint(Stream target, ulong value)
    {
        for (; value >= 0x80; value >>= 7) target.WriteByte((byte)(value | 0x80));
        target.WriteByte((byte)value);
    }

    private static void CopyBytes(Stream source, Stream target, long count)
    {
        Span<byte> buffer = stackalloc byte[4096];
        while (count > 0)
        {
            var read = source.Read(buffer[..(int)Math.Min(count, buffer.Length)]);
            if (read == 0) throw new EndOfStreamException("The model ends inside a field.");
            target.Write(buffer[..read]);
            count -= read;
        }
    }

    /// <summary>
    /// The files a session's placements keep on disk — a copy of a model too large to keep in
    /// memory, and the folder of the graph ONNX Runtime runs — deleted as the placements are let go
    /// of, or, where the session is collected without being disposed, as this is finalized: a
    /// compiled graph held weakly goes that way, and each would otherwise leave a model's worth of
    /// files behind. A file still mapped by a variant ONNX Runtime has not released yet stays.
    ///
    /// <para><b>Claimed while kept.</b> A process that ends without letting go of them — killed, or
    /// crashed — leaves them behind, so each is held open while it is kept: the model's copy itself,
    /// and in the folder a file of its own (<see cref="KeptLockFile"/>). What is left in the
    /// temporary folder by a process that has ended is claimed by nothing, and is deleted by the next
    /// process to keep files there (<see cref="SweepStale"/>).</para>
    /// </summary>
    private sealed class KeptFiles
    {
        private FileStream? _modelClaim;
        private FileStream? _runsClaim;

        internal string? Model { get; private set; }
        internal string? Runs { get; private set; }

        /// <summary>Keeps <paramref name="model"/> in a file of its own, claimed.</summary>
        internal void KeepModel(byte[] model)
        {
            (SweepOnThisThread ?? ProcessSweep).Start();
            Model = Path.Combine(Path.GetTempPath(), ModelFilePrefix + Guid.NewGuid().ToString("N") + ".onnx");
            File.WriteAllBytes(Model, model);
            _modelClaim = new FileStream(Model, FileMode.Open, FileAccess.Read, FileShare.Read);
        }

        /// <summary>Keeps <paramref name="directory"/>, claimed — kept unclaimed where its claim
        /// cannot be made, as a folder nothing else could take over a sweep is.</summary>
        internal void KeepRuns(string directory)
        {
            (SweepOnThisThread ?? ProcessSweep).Start();
            Runs = directory;
            try
            {
                _runsClaim = new FileStream(Path.Combine(directory, KeptLockFile), FileMode.CreateNew, FileAccess.ReadWrite, FileShare.Read);
            }
            catch (Exception unclaimed) when (unclaimed is IOException or UnauthorizedAccessException) { }
        }

        ~KeptFiles() => Delete();

        internal void Delete()
        {
            try { _modelClaim?.Dispose(); } catch (Exception) { }
            try { _runsClaim?.Dispose(); } catch (Exception) { }
            _modelClaim = _runsClaim = null;
            if (Model is { } model)
                try { File.Delete(model); } catch (Exception) { }
            if (Runs is { } runs)
                try { Directory.Delete(runs, recursive: true); } catch (Exception) { }
            Model = null;
            Runs = null;
        }

    }

    /// <summary>The sweep of what ended processes left (<see cref="SweepStale"/>), made once in the
    /// background as this process first keeps files.</summary>
    private static readonly StaleSweep ProcessSweep = new(SweepStale);

    /// <summary>For a test: the sweep this thread's placements start as they first keep files, in place
    /// of the process's own; null on every thread but one a test sets it on.</summary>
    [ThreadStatic]
    internal static StaleSweep? SweepOnThisThread;

    /// <summary>A sweep made once, in the background, the first time it is started.</summary>
    internal sealed class StaleSweep(Action sweep)
    {
        private readonly Lazy<Task> _task = new(() => Task.Run(sweep));

        /// <summary>Starts the sweep where it has not been, answering it.</summary>
        internal Task Start() => _task.Value;

        /// <summary>Whether the sweep has been started.</summary>
        internal bool Started => _task.IsValueCreated;
    }

    private const string ModelFilePrefix = "shorokoo-model-";

    /// <summary>The prefixes of the folders the sessions and their placements make in the temporary
    /// folder: the graph that runs, kept; and those made and deleted as a session or a variant is
    /// built.</summary>
    private static readonly string[] FolderPrefixes = ["shorokoo-runs-", "shorokoo-optimized-", "shorokoo-placed-"];

    /// <summary>How long a file or folder has gone unwritten before a sweep may take it for one an
    /// ended process left: none is in the making that long.</summary>
    private static readonly TimeSpan StaleAfter = TimeSpan.FromDays(1);

    /// <summary>
    /// Deletes what processes that have ended left in the temporary folder: each model copy (as
    /// <see cref="KeptFiles"/> keeps them) and each folder of a session's or its placements' that
    /// has gone a day unwritten and that no live process claims — a model copy no process holds open,
    /// a folder whose <see cref="KeptLockFile"/> no process holds open, or that has none. Anything it
    /// cannot read or delete it leaves.
    /// </summary>
    internal static void SweepStale()
    {
        var stale = DateTime.UtcNow - StaleAfter;
        string[] entries;
        try
        {
            entries = Directory.GetFileSystemEntries(Path.GetTempPath(), "shorokoo-*");
        }
        catch (Exception unreadable) when (unreadable is IOException or UnauthorizedAccessException or ArgumentException) { return; }
        foreach (var entry in entries)
            try
            {
                var name = Path.GetFileName(entry);
                if (name.StartsWith(ModelFilePrefix, StringComparison.Ordinal) && name.EndsWith(".onnx", StringComparison.Ordinal))
                {
                    if (File.GetLastWriteTimeUtc(entry) < stale) DeleteUnclaimed(entry);
                }
                else if (FolderPrefixes.Any(p => name.StartsWith(p, StringComparison.Ordinal)) && Directory.Exists(entry)
                         && Directory.GetLastWriteTimeUtc(entry) < stale)
                {
                    var claim = Path.Combine(entry, KeptLockFile);
                    if (!File.Exists(claim) || DeleteUnclaimed(claim)) Directory.Delete(entry, recursive: true);
                }
            }
            catch (Exception locked) when (locked is IOException or UnauthorizedAccessException) { }
    }

    /// <summary>Deletes <paramref name="file"/> where no process holds it open, which opening it for
    /// itself alone tells, and answers whether it did.</summary>
    private static bool DeleteUnclaimed(string file)
    {
        try
        {
            using (new FileStream(file, FileMode.Open, FileAccess.ReadWrite, FileShare.None, 1, FileOptions.DeleteOnClose)) { }
            return true;
        }
        catch (Exception claimed) when (claimed is IOException or UnauthorizedAccessException)
        {
            return false;
        }
    }

    /// <summary>The graph ONNX Runtime runs for the plain session, with its initializers in files
    /// beside it: written out by the session as it was built, or else by a probe build of the model,
    /// once, into a folder kept until the session goes.</summary>
    private ModelProto RunGraph()
    {
        lock (_sources) return RunGraphUnderLock();
    }

    /// <summary><see cref="RunGraph"/>, under the lock. A probe of a model kept in a file is built
    /// from the file, which ONNX Runtime reads itself, so that no copy of the model is made here
    /// besides the one it builds from.</summary>
    private ModelProto RunGraphUnderLock()
    {
        if (_runs is not null) return _runs;
        var directory = Path.Combine(Path.GetTempPath(), "shorokoo-runs-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            _build(_model, _model is null ? _files.Model : null, directory, null, false).Dispose();
            _runs = ReadOptimized(directory);
            _files.KeepRuns(directory);
            return _runs;
        }
        catch (Exception failure)
        {
            _broken = failure.Message;
            try { Directory.Delete(directory, recursive: true); } catch (Exception) { }
            throw;
        }
    }

    /// <summary>The model a build wrote into <paramref name="directory"/>, its larger initializers
    /// left in the file beside it.</summary>
    internal static ModelProto ReadOptimized(string directory)
    {
        using var stream = File.OpenRead(Path.Combine(directory, OrtBackend.OptimizedModelFile));
        var model = ProtoBuf.Serializer.Deserialize<ModelProto>(stream);
        return model.Graph is null ? throw new InvalidOperationException("ONNX Runtime wrote out a model with no graph.") : model;
    }

    /// <summary><paramref name="model"/> with <paramref name="exposed"/> made graph outputs, each
    /// stating its type and shape for the run.</summary>
    private static byte[] WithOutputs(ModelProto model, IReadOnlyList<string> exposed, IReadOnlyDictionary<string, PlacementShapes.Value> shapes)
    {
        var copy = ProtoBuf.Serializer.DeepClone(model);
        foreach (var name in exposed)
        {
            var value = shapes[name];
            var tensor = new TypeProto.Tensor { ElemType = value.ElementType, Shape = new TensorShapeProto() };
            foreach (var dim in value.Shape) tensor.Shape.Dims.Add(new TensorShapeProto.Dimension { DimValue = dim });
            copy.Graph!.Outputs.Add(new ValueInfoProto { Name = name, Type = new TypeProto { TensorType = tensor } });
        }
        using var written = new MemoryStream();
        ProtoBuf.Serializer.Serialize(written, copy);
        return written.ToArray();
    }

    /// <summary>Whether two graphs compute with the same nodes: the same operators, each as many
    /// times.</summary>
    private static bool SameComputeNodes(GraphProto a, GraphProto b)
    {
        static Dictionary<string, int> Count(GraphProto graph)
        {
            var counts = new Dictionary<string, int>(StringComparer.Ordinal);
            foreach (var node in graph.Nodes)
            {
                var key = node.Domain + "::" + node.OpType;
                counts[key] = counts.GetValueOrDefault(key) + 1;
            }
            return counts;
        }
        var (x, y) = (Count(a), Count(b));
        return x.Count == y.Count && x.All(pair => y.TryGetValue(pair.Key, out var n) && n == pair.Value);
    }

    /// <summary>Whether what <paramref name="graph"/> states of <paramref name="name"/>'s shape — its
    /// dimensions that are numbers, and those named after an input dimension — agrees with
    /// <paramref name="value"/>, as the placement assumes. A shape it does not state agrees.</summary>
    private static bool StatedAgrees(
        GraphProto graph, string name, PlacementShapes.Value value, IReadOnlyDictionary<string, (long[] Shape, int ElementType)> given)
    {
        var info = graph.ValueInfoes.Concat(graph.Outputs).FirstOrDefault(i => i.Name == name);
        if (info?.Type?.TensorType is not { } tensor) return true;
        if (tensor.ElemType != 0 && tensor.ElemType != value.ElementType) return false;
        if (tensor.Shape is not { } shape) return true;
        if (shape.Dims.Count != value.Shape.Length) return false;
        var symbols = new Dictionary<string, long>(StringComparer.Ordinal);
        foreach (var input in graph.Inputs)
            if (input.Type?.TensorType?.Shape is { } declared && given.TryGetValue(input.Name, out var g) && declared.Dims.Count == g.Shape.Length)
                for (int i = 0; i < g.Shape.Length; i++)
                    if (declared.Dims[i].DimParam is { Length: > 0 } symbol) symbols.TryAdd(symbol, g.Shape[i]);
        for (int i = 0; i < value.Shape.Length; i++)
        {
            var dim = shape.Dims[i];
            if (dim.DimParam is { Length: > 0 } symbol)
            {
                if (symbols.TryGetValue(symbol, out var bound) && bound != value.Shape[i]) return false;
            }
            else if (dim.ShouldSerializeDimValue() && dim.DimValue != value.Shape[i]) return false;
        }
        return true;
    }

    public void Dispose()
    {
        lock (_gate)
        {
            _disposed = true;
            // A variant a run is on goes as that run ends.
            foreach (var entry in _entries.Values)
                if (entry.Running == 0) LetGoOfVariant(entry);
        }
        lock (_sources)
        {
            _files.Delete();
            GC.SuppressFinalize(_files);
        }
    }
}
