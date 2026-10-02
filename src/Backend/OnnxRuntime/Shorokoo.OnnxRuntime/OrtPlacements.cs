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
/// for this session — written out once, with its initializers, by a probe build of the model. Where
/// every value the plan places is a value of the model as handed over, the variant is first built
/// over that model, optimized as the plain session is, which runs its nodes in the plain session's
/// order; otherwise, or where that variant does not hold the plan, it is built from the graph that
/// runs itself, with ONNX Runtime's optimizations off: its fusions are made already, so a value only
/// a fusion makes (a fused layer normalization's sum) is a value the variant can bind, and nothing is
/// rewritten again. Either way the plan is proved again over the graph the variant writes out as it
/// is built, and the variant's compute nodes must be the plain session's, operator for
/// operator.</para>
///
/// <para><b>The model is kept</b> to build the probe and the variants from: in memory up to
/// <see cref="ModelBytesKept"/>, and a larger one in a file of its own, written as the session is
/// built and deleted with it, so that its weights are not held a second time. A variant of such a
/// model is built from the graph that runs alone, which reads its weights from the files the probe
/// wrote rather than from a copy of the model.</para>
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
/// the memory a run holds (<see cref="PlacementProof.ModelledPeak"/>): with every placed value in its
/// block against with none, less what the variant holds of its own. A signature's first run is
/// placed already where that pays, so a run that fits only placed runs on its first call. The first
/// placed run is measured, and a plan whose run asked for more than the model said the plain one
/// would is let go of for the runs after it.</para>
/// </summary>
internal sealed class OrtPlacements : IDisposable
{
    /// <summary>The largest model a session keeps in memory to build variants from; a larger one it
    /// keeps in a file.</summary>
    internal const int ModelBytesKept = 16 << 20;

    /// <summary>How many signatures a session plans for; a run of any other runs unplaced.</summary>
    private const int MostSignatures = 8;

    /// <summary>
    /// Builds a session over a model, writing the graph it will run into the folder
    /// <paramref name="optimizedDirectory"/> names where one is. A model read with its initializers
    /// from files in <paramref name="externalDataDirectory"/> — a graph ONNX Runtime wrote out — is
    /// built with ONNX Runtime's optimizations off. A <paramref name="variant"/> charges the
    /// session's own allocator accounts; a probe, built and let go of, accounts of its own.
    /// </summary>
    internal delegate OrtSession VariantBuilder(byte[] model, string? optimizedDirectory, string? externalDataDirectory, bool variant);

    // The model as handed over: in memory where it is small, and otherwise in a file of its own.
    private readonly byte[]? _model;
    private string? _modelFile;
    private readonly VariantBuilder _build;
    private readonly OrtBackend _backend;
    // What the session and its variants hold of the allocator accounts they share, read either side
    // of building a variant to tell what the variant holds of its own.
    private readonly Func<long> _held;
    private readonly object _gate = new();
    private readonly Dictionary<string, Entry> _entries = new(StringComparer.Ordinal);
    private ModelProto? _original;
    // The graph ONNX Runtime runs for the plain session, as a probe build wrote it out, and the folder
    // it is in with its initializers, kept for the session's life to build variants from.
    private ModelProto? _runs;
    private string? _runsDirectory;
    private string? _broken;
    private bool _disposed;

    internal OrtPlacements(byte[] model, VariantBuilder build, OrtBackend backend, Func<long> held)
    {
        _build = build;
        if (model.Length <= ModelBytesKept)
            _model = model;
        else
            try
            {
                _modelFile = Path.Combine(Path.GetTempPath(), "shorokoo-model-" + Guid.NewGuid().ToString("N") + ".onnx");
                File.WriteAllBytes(_modelFile, model);
            }
            catch (Exception unwritable) when (unwritable is IOException or UnauthorizedAccessException)
            {
                // Placing is a saving and never a requirement: a model that cannot be kept is not
                // placed in.
                _broken = $"the model could not be kept in a file: {unwritable.Message}";
                DeleteModelFile();
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

    /// <summary>Every signature planned so far, for a test to read.</summary>
    internal IReadOnlyList<Entry> Entries
    {
        get { lock (_gate) return [.. _entries.Values]; }
    }

    /// <summary>
    /// The memory a run may place values in, by input name: each value the run consumed that it was
    /// fed under that one name, a tensor of this runtime of fixed-width elements, into which no
    /// marked output is written.
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
            if (PlacementShapes.ElementBytes((int)own.ElementType) == 0 || BytesOf(own) <= 0) continue;
            blocks[name] = own;
        }
        return blocks;
    }

    private static long BytesOf(OrtTensorValue value)
        => value.ReadShape.Aggregate(1L, (a, d) => a * d) * PlacementShapes.ElementBytes((int)value.ElementType);

    /// <summary>
    /// The entry for a run that may place values in <paramref name="blocks"/>, made the first time
    /// its signature is seen; null where this session has stopped planning or plans for as many
    /// signatures as it will.
    /// </summary>
    internal Entry? EntryFor(
        IReadOnlyDictionary<string, IShorokooTensorValue> inputs, Dictionary<string, OrtTensorValue> blocks,
        IReadOnlyList<string> outputNames)
    {
        if (blocks.Count == 0) return null;
        var key = new System.Text.StringBuilder();
        foreach (var (name, value) in inputs.OrderBy(i => i.Key, StringComparer.Ordinal))
        {
            // A block whose parts go back once no output stands on them plans otherwise than one
            // held whole, so the two are signatures of their own.
            key.Append(name).Append(blocks.TryGetValue(name, out var block) ? OrtBackend.RangesGoBack(block, out _, out _) ? "!!" : "!" : ":");
            if (value is OrtTensorValue { ValueType: ShorokooOnnxValueType.Tensor } tensor)
                key.Append((int)tensor.ElementType).Append('[').AppendJoin(',', tensor.ReadShape).Append(']');
            key.Append(';');
        }
        key.Append("->").AppendJoin(';', outputNames);
        lock (_gate)
        {
            if (_disposed || _broken is not null) return null;
            if (_entries.TryGetValue(key.ToString(), out var entry)) return entry;
            if (_entries.Count >= MostSignatures) return null;
            return _entries[key.ToString()] = new Entry();
        }
    }

    /// <summary>
    /// Runs <paramref name="owner"/> on a signature <paramref name="entry"/> stands for, the way the
    /// entry's stage says: plain while it is measured or once refused, placed on its variant while
    /// that is tried or once adopted. <paramref name="kept"/> answers the consumed values a placed
    /// output stands on, which the run does not release: the block each became releases it with its
    /// last output.
    /// </summary>
    internal IReadOnlyList<IShorokooTensorValue> Run(
        OrtSession owner, Entry entry, IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
        Dictionary<string, OrtTensorValue> blocks, IReadOnlyList<string> outputNames,
        Func<IReadOnlyList<IShorokooTensorValue>> plain,
        Func<OrtSession, IReadOnlyDictionary<string, OrtSession.PlacedBinding>, IReadOnlyList<IShorokooTensorValue>> placed,
        out HashSet<IShorokooTensorValue>? kept)
    {
        kept = null;
        OrtSession? variant;
        bool first;
        lock (_gate)
        {
            if (entry.Stage == Stage.Unplanned)
            {
                Prepare(entry, inputs, blocks, outputNames);
                Settled?.Invoke(entry);
            }
            variant = entry.Stage == Stage.Adopted ? entry.Variant : null;
            first = entry.PlacedPeak == 0;
        }
        if (variant is null) return plain();
        var results = RunPlaced(variant, entry, blocks, outputNames, placed, out kept, out var peak);
        if (first)
            lock (_gate)
            {
                if (entry.Stage == Stage.Adopted && entry.PlacedPeak == 0)
                {
                    entry.PlacedPeak = Math.Max(peak, 1);
                    // The model was wrong where the placed run asked for more than it said the plain
                    // one would: the runs after this one run plain.
                    if (peak + entry.VariantHeld + Margin(entry.PredictedPlainPeak) > entry.PredictedPlainPeak)
                        Refuse(entry, $"placed, the run asked {peak} bytes against {entry.PredictedPlainPeak} modelled plain");
                }
            }
        return results;
    }

    /// <summary>The least a plan must save to be kept: a mebibyte, or a sixty-fourth of the plain
    /// run where that is more.</summary>
    private static long Margin(long plain) => Math.Max(1L << 20, plain / 64);

    /// <summary>A run on <paramref name="variant"/> with every placement of
    /// <paramref name="entry"/> bound to its range, measuring the most it asked of the variant's
    /// allocator.</summary>
    private IReadOnlyList<IShorokooTensorValue> RunPlaced(
        OrtSession variant, Entry entry, Dictionary<string, OrtTensorValue> blocks, IReadOnlyList<string> outputNames,
        Func<OrtSession, IReadOnlyDictionary<string, OrtSession.PlacedBinding>, IReadOnlyList<IShorokooTensorValue>> placed,
        out HashSet<IShorokooTensorValue>? kept, out long peak)
    {
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
                            : (OrtBackend.BlockOver(owner, BytesOf(owner), () => ((IShorokooBackend)_backend).Release(owner)), 0L);
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
            var results = variant.Measured(() => placed(variant, bindings), out peak);
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
    /// built from that graph and the plan proved again over the graph the variant runs, until the
    /// two agree. Refuses the entry, for good, where nothing can be placed, where placing saves too
    /// little, where the variant's compute nodes differ from the plain session's, or where anything
    /// fails on the way. Under the lock.
    /// </summary>
    private void Prepare(
        Entry entry, IReadOnlyDictionary<string, IShorokooTensorValue> inputs, Dictionary<string, OrtTensorValue> blocks,
        IReadOnlyList<string> outputNames)
    {
        try
        {
            var original = Original();
            var given = new Dictionary<string, (long[] Shape, int ElementType)>(StringComparer.Ordinal);
            foreach (var (name, value) in inputs)
                if (value is OrtTensorValue { ValueType: ShorokooOnnxValueType.Tensor } tensor)
                    given[name] = (tensor.ReadShape, (int)tensor.ElementType);
            var blockBytes = blocks.ToDictionary(b => b.Key, b => BytesOf(b.Value), StringComparer.Ordinal);
            var givingBack = blocks.Where(b => OrtBackend.RangesGoBack(b.Value, out _, out _)).Select(b => b.Key).ToHashSet(StringComparer.Ordinal);
            entry.Given = given;
            entry.BlockBytes = blockBytes;
            // The model as handed over first, which costs no build: what it places nothing in, the
            // graph ONNX Runtime makes of it does not either, short of a rewrite freeing a range --
            // and a session builds nothing for such a run.
            if (original.Graph is not { } handed
                || handed.Nodes.Count > PlacementProof.MostNodes
                || new PlacementProof(handed, blockBytes, PlacementShapes.Evaluate(handed, given), runsInOrder: true)
                    .Plan(PlacementProof.Smallest, PlacementProof.IdleOutputBytes, givingBack).Count == 0)
            {
                Refuse(entry, "nothing to place in the graph handed over");
                return;
            }
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
            var proof = new PlacementProof(runs, blockBytes, shapes, runsInOrder: true);
            var plan = proof.Prove(proof.Plan(PlacementProof.Smallest, PlacementProof.IdleOutputBytes, givingBack)
                .Where(p => !runsOutputs.Contains(p.Value) || outputNames.Contains(p.Value)));
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
            // where exposing them stops a fusion, what it runs is no longer the plain session's. A
            // variant over the graph ONNX Runtime wrote out binds any value of it, a fusion's too,
            // but ONNX Runtime orders that graph afresh as it loads it. So the first is tried where it
            // can be, and the second where it cannot or did not hold.
            var originalValues = original.Graph.Nodes.SelectMany(n => n.Outputs).ToHashSet(StringComparer.Ordinal);
            bool[] sources = _model is not null && plan.All(p => originalValues.Contains(p.Value)) ? [false, true] : [true];
            foreach (var fromRuns in sources)
            {
                var candidate = plan;
                for (int attempt = 0; attempt < 3; attempt++)
                {
                    var outcome = TryVariant(entry, fromRuns ? runsModel : original, fromRuns, candidate, runs, runsOutputs, shapes, given, blockBytes);
                    if (outcome.Plan is null) break;
                    if (entry.Stage != Stage.Unplanned) return;
                    candidate = outcome.Plan;
                }
                if (entry.Stage != Stage.Unplanned) return;
            }
            Refuse(entry, entry.Refusal ?? "no variant held the plan");
        }
        catch (Exception failure) when (failure is not OutOfMemoryException)
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
            variant = _build(WithOutputs(model, exposed, shapes), directory, fromRuns ? _runsDirectory : null, true);
            var variantHeld = _held() - heldBefore;
            var graph = ReadOptimized(directory).Graph!;
            entry.VariantGraph = graph;
            if (!SameComputeNodes(runs, graph))
            {
                entry.Refusal = "exposing the placed values changed what ONNX Runtime runs";
                return (null, false);
            }
            var variantShapes = PlacementShapes.Evaluate(graph, given);
            var variantProof = new PlacementProof(graph, blockBytes, variantShapes, runsOutputs, runsInOrder: true);
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
            entry.Variant = variant;
            entry.Stage = Stage.Adopted;
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
    private static bool Pays(Entry entry, long held)
        => entry.PredictedPlainPeak - entry.PredictedPlacedPeak - held >= Margin(entry.PredictedPlainPeak);

    /// <summary>The bytes of the initializers of <paramref name="model"/> a session built over it
    /// reads in itself: every one but those the session is handed (<see cref="SuppliedInitializer"/>).</summary>
    private long OwnInitializerBytes(ModelProto model)
    {
        var handed = model.Graph!.Inputs.Select(i => i.Name).ToHashSet(StringComparer.Ordinal);
        long bytes = 0;
        foreach (var initializer in model.Graph.Initializers)
            if (!handed.Contains(initializer.Name))
                bytes += (initializer.Dims ?? []).Aggregate(1L, (a, d) => a * d) * PlacementShapes.ElementBytes(initializer.data_type);
        return bytes;
    }

    private static void Refuse(Entry entry, string why)
    {
        entry.Stage = Stage.Refused;
        entry.Refusal = why;
        entry.Variant?.Dispose();
        entry.Variant = null;
    }

    /// <summary>The model as handed to the backend, for a measurement to read.</summary>
    internal ModelProto OriginalModel
    {
        get { lock (_gate) return Original(); }
    }

    /// <summary>The model as handed to the backend, parsed once. One kept in a file is kept parsed
    /// without the contents of its larger tensors, which the proof does not read: it is a graph to
    /// plan over, and never one to build from.</summary>
    private ModelProto Original()
    {
        if (_original is not null) return _original;
        if (_model is not null)
        {
            using var stream = new MemoryStream(_model, writable: false);
            return _original = ProtoBuf.Serializer.Deserialize<ModelProto>(stream);
        }
        using (var file = File.OpenRead(_modelFile!))
            _original = ProtoBuf.Serializer.Deserialize<ModelProto>(file);
        foreach (var tensor in _original.Graph?.Initializers ?? [])
            if (tensor.RawData is { Length: > 1024 }) tensor.RawData = [];
        foreach (var node in _original.Graph?.Nodes ?? [])
            foreach (var attribute in node.Attributes)
                if (attribute.T?.RawData is { Length: > 1024 }) attribute.T.RawData = [];
        return _original;
    }

    /// <summary>The model's bytes, as handed to the backend.</summary>
    private byte[] ModelBytes() => _model ?? File.ReadAllBytes(_modelFile!);

    private void DeleteModelFile()
    {
        if (_modelFile is null) return;
        try { File.Delete(_modelFile); } catch (Exception) { }
        _modelFile = null;
    }

    /// <summary>The graph ONNX Runtime runs for the plain session, with its initializers in files
    /// beside it: written out by a probe build of the model, once, into a folder kept until the
    /// session goes.</summary>
    private ModelProto RunGraph()
    {
        if (_runs is not null) return _runs;
        var directory = Path.Combine(Path.GetTempPath(), "shorokoo-runs-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            _build(ModelBytes(), directory, null, false).Dispose();
            _runs = ReadOptimized(directory);
            _runsDirectory = directory;
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
    private static ModelProto ReadOptimized(string directory)
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
            foreach (var entry in _entries.Values)
            {
                entry.Variant?.Dispose();
                entry.Variant = null;
            }
            if (_runsDirectory is not null)
            {
                try { Directory.Delete(_runsDirectory, recursive: true); } catch (Exception) { }
                _runsDirectory = null;
            }
            DeleteModelFile();
        }
    }
}
