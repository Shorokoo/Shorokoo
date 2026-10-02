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
/// <para><b>Proved over the graph that runs.</b> The plan is made over the graph ONNX Runtime runs
/// for this session (written out by a probe build of the model), and proved again over the graph the
/// variant runs, written out as it is built: a value its rewrites folded into a constant, a node a
/// fusion removed, or a reader a fusion introduced, refuses the placement. The variant's compute nodes
/// must be the plain session's, operator for operator: exposing a value that blocks a fusion changes
/// what runs, and such a plan is not used.</para>
///
/// <para><b>Measured before it is kept.</b> A signature's first run runs as the session always runs,
/// and the most it asks of the session's allocator is recorded; its second runs placed, and the
/// placements are kept only where that run asked for less, by more than what the variant holds of
/// its own. Every later run of the signature runs the way that chose.</para>
/// </summary>
internal sealed class OrtPlacements : IDisposable
{
    /// <summary>The largest model a session keeps to build variants from.</summary>
    internal const int ModelBytesKept = 16 << 20;

    /// <summary>How many signatures a session plans for; a run of any other runs unplaced.</summary>
    private const int MostSignatures = 8;

    /// <summary>Builds a session over a model, charging the session's own allocator accounts, writing
    /// the graph it will run into the folder named where one is.</summary>
    internal delegate OrtSession VariantBuilder(byte[] model, string? optimizedDirectory);

    private readonly byte[] _model;
    private readonly VariantBuilder _build;
    private readonly OrtBackend _backend;
    // What the session and its variants hold of the allocator accounts they share, read either side
    // of building a variant to tell what the variant holds of its own.
    private readonly Func<long> _held;
    private readonly object _gate = new();
    private readonly Dictionary<string, Entry> _entries = new(StringComparer.Ordinal);
    private ModelProto? _original;
    private HashSet<string>? _originalValues;
    private GraphProto? _runs;
    private string? _broken;
    private bool _disposed;

    internal OrtPlacements(byte[] model, VariantBuilder build, OrtBackend backend, Func<long> held)
    {
        _model = model;
        _build = build;
        _backend = backend;
        _held = held;
    }

    /// <summary>Where a signature is: run as the session runs to measure it, run placed to measure
    /// that, or settled one way or the other.</summary>
    internal enum Stage { MeasurePlain, TryPlaced, Adopted, Refused }

    /// <summary>One run signature's placements and how they measured.</summary>
    internal sealed class Entry
    {
        internal Stage Stage;
        internal long PlainPeak;
        internal long PlacedPeak;
        internal long VariantHeld;
        internal IReadOnlyList<Placement> Plan = [];
        internal Dictionary<string, (long[] Shape, ShorokooTensorElementType Type)> Shapes = new(StringComparer.Ordinal);
        internal OrtSession? Variant;
        internal string? Refusal;

        // What the entry was planned over, for a measurement to read: the graph ONNX Runtime runs
        // for the plain session, the shapes the run's inputs came in, and its blocks.
        internal GraphProto? Graph;
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
            key.Append(name).Append(blocks.ContainsKey(name) ? "!" : ":");
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
        Stage stage;
        lock (_gate) stage = entry.Stage;
        switch (stage)
        {
            case Stage.MeasurePlain:
            {
                var results = owner.Measured(plain, out var peak);
                lock (_gate)
                {
                    if (entry.Stage == Stage.MeasurePlain)
                    {
                        entry.PlainPeak = peak;
                        entry.Stage = Stage.TryPlaced;
                    }
                }
                return results;
            }
            case Stage.TryPlaced:
            {
                lock (_gate)
                    if (entry.Variant is null && entry.Stage == Stage.TryPlaced) Prepare(entry, inputs, blocks, outputNames);
                if (entry.Stage != Stage.TryPlaced || entry.Variant is not { } variant) return plain();
                var results = RunPlaced(variant, entry, blocks, outputNames, placed, out kept, out var peak);
                lock (_gate)
                {
                    if (entry.Stage == Stage.TryPlaced)
                    {
                        entry.PlacedPeak = peak;
                        var margin = Math.Max(1L << 20, entry.PlainPeak / 64);
                        if (peak + entry.VariantHeld + margin <= entry.PlainPeak)
                        {
                            entry.Stage = Stage.Adopted;
                            Settled?.Invoke(entry);
                        }
                        else Refuse(entry, $"placed, the run asked {peak} bytes against {entry.PlainPeak} plain");
                    }
                }
                return results;
            }
            case Stage.Adopted when entry.Variant is { } variant:
                return RunPlaced(variant, entry, blocks, outputNames, placed, out kept, out _);
            default:
                return plain();
        }
    }

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
                            : (new SharedBlock(BytesOf(owner), () => ((IShorokooBackend)_backend).Release(owner)), 0L);
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
            return variant.Measured(() => placed(variant, bindings), out peak);
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
    /// Plans <paramref name="entry"/> and builds its variant: the plan over the graph ONNX Runtime
    /// runs, proved again over the graph the variant runs, until the two agree. Refuses the entry,
    /// for good, where nothing can be placed, the variant's compute nodes differ from the plain
    /// session's, or anything fails on the way. Under the lock.
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
            entry.Given = given;
            entry.BlockBytes = blockBytes;
            // The model as handed over first, which costs no build: what it places nothing in, the
            // graph ONNX Runtime makes of it does not either, short of a rewrite freeing a range --
            // and a session builds nothing for such a run.
            if (original.Graph is not { } handed
                || handed.Nodes.Count > PlacementProof.MostNodes
                || new PlacementProof(handed, blockBytes, PlacementShapes.Evaluate(handed, given)).Plan(PlacementProof.Smallest, PlacementProof.IdleOutputBytes).Count == 0)
            {
                Refuse(entry, "nothing to place in the graph handed over");
                return;
            }
            var runs = RunGraph();
            entry.Graph = runs;
            if (runs.Nodes.Count > PlacementProof.MostNodes)
            {
                Refuse(entry, "too large a graph to place in");
                return;
            }
            var shapes = PlacementShapes.Evaluate(runs, given);
            var proof = new PlacementProof(runs, blockBytes, shapes);
            var originalOutputs = original.Graph.Outputs.Select(o => o.Name).ToHashSet(StringComparer.Ordinal);
            var planned = proof.Plan(PlacementProof.Smallest, PlacementProof.IdleOutputBytes);
            var plan = proof.Prove(planned
                .Where(p => _originalValues!.Contains(p.Value) && (originalOutputs.Contains(p.Value) ? outputNames.Contains(p.Value) : true)));
            for (int attempt = 0; attempt < 3; attempt++)
            {
                if (plan.Count == 0)
                {
                    Refuse(entry, attempt > 0 ? "the variant's own graph proves none of the placements"
                        : planned.Count == 0 ? "nothing to place in the graph ONNX Runtime runs"
                        : $"the graph ONNX Runtime runs places only values its rewrites made ({string.Join(", ", planned.Select(p => p.Value))})");
                    return;
                }
                var exposed = plan.Where(p => !originalOutputs.Contains(p.Value)).Select(p => p.Value).ToList();
                var model = WithOutputs(exposed, shapes);
                var directory = Path.Combine(Path.GetTempPath(), "shorokoo-placed-" + Guid.NewGuid().ToString("N"));
                Directory.CreateDirectory(directory);
                OrtSession? variant = null;
                try
                {
                    var heldBefore = _held();
                    variant = _build(model, directory);
                    var variantHeld = _held() - heldBefore;
                    var graph = ReadOptimized(directory);
                    var variantShapes = PlacementShapes.Evaluate(graph, given);
                    if (!SameComputeNodes(runs, graph))
                    {
                        Refuse(entry, "exposing the placed values changed what ONNX Runtime runs");
                        return;
                    }
                    var proved = new PlacementProof(graph, blockBytes, variantShapes, originalOutputs).Prove(plan
                        .Where(p => variantShapes.TryGetValue(p.Value, out var v) && v.Bytes == p.Bytes && StatedAgrees(graph, p.Value, v, given)));
                    if (proved.Count == plan.Count)
                    {
                        entry.Plan = plan;
                        entry.Shapes = plan.ToDictionary(
                            p => p.Value, p => (variantShapes[p.Value].Shape, (ShorokooTensorElementType)variantShapes[p.Value].ElementType),
                            StringComparer.Ordinal);
                        entry.VariantHeld = variantHeld;
                        entry.Variant = variant;
                        variant = null;
                        return;
                    }
                    plan = proved;
                }
                finally
                {
                    variant?.Dispose();
                    try { Directory.Delete(directory, recursive: true); } catch (Exception) { }
                }
            }
            Refuse(entry, "the placements proved kept changing as the variant was built");
        }
        catch (Exception failure) when (failure is not OutOfMemoryException)
        {
            Refuse(entry, $"planning failed: {failure.GetType().Name}: {failure.Message}");
        }
    }

    private static void Refuse(Entry entry, string why)
    {
        entry.Stage = Stage.Refused;
        entry.Refusal = why;
        entry.Variant?.Dispose();
        entry.Variant = null;
        Settled?.Invoke(entry);
    }

    /// <summary>The model as handed to the backend, for a measurement to read.</summary>
    internal ModelProto OriginalModel
    {
        get { lock (_gate) return Original(); }
    }

    /// <summary>The model as handed to the backend, parsed once.</summary>
    private ModelProto Original()
    {
        if (_original is not null) return _original;
        using var stream = new MemoryStream(_model, writable: false);
        var model = ProtoBuf.Serializer.Deserialize<ModelProto>(stream);
        _originalValues = new HashSet<string>(StringComparer.Ordinal);
        foreach (var node in model.Graph?.Nodes ?? [])
            foreach (var output in node.Outputs)
                if (output.Length > 0) _originalValues.Add(output);
        return _original = model;
    }

    /// <summary>The graph ONNX Runtime runs for the plain session: written out by a probe build of
    /// the model, once.</summary>
    private GraphProto RunGraph()
    {
        if (_runs is not null) return _runs;
        var directory = Path.Combine(Path.GetTempPath(), "shorokoo-plain-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            _build(_model, directory).Dispose();
            return _runs = ReadOptimized(directory);
        }
        catch (Exception failure)
        {
            _broken = failure.Message;
            throw;
        }
        finally
        {
            try { Directory.Delete(directory, recursive: true); } catch (Exception) { }
        }
    }

    /// <summary>The graph a build wrote into <paramref name="directory"/>, its larger initializers
    /// left in the file beside it.</summary>
    private static GraphProto ReadOptimized(string directory)
    {
        using var stream = File.OpenRead(Path.Combine(directory, OrtBackend.OptimizedModelFile));
        return ProtoBuf.Serializer.Deserialize<ModelProto>(stream).Graph
               ?? throw new InvalidOperationException("ONNX Runtime wrote out a model with no graph.");
    }

    /// <summary>The model with <paramref name="exposed"/> made graph outputs, each stating its type
    /// and shape for the run.</summary>
    private byte[] WithOutputs(IReadOnlyList<string> exposed, IReadOnlyDictionary<string, PlacementShapes.Value> shapes)
    {
        using var source = new MemoryStream(_model, writable: false);
        var model = ProtoBuf.Serializer.Deserialize<ModelProto>(source);
        foreach (var name in exposed)
        {
            var value = shapes[name];
            var tensor = new TypeProto.Tensor { ElemType = value.ElementType, Shape = new TensorShapeProto() };
            foreach (var dim in value.Shape) tensor.Shape.Dims.Add(new TensorShapeProto.Dimension { DimValue = dim });
            model.Graph.Outputs.Add(new ValueInfoProto { Name = name, Type = new TypeProto { TensorType = tensor } });
        }
        using var written = new MemoryStream();
        ProtoBuf.Serializer.Serialize(written, model);
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
        }
    }
}
