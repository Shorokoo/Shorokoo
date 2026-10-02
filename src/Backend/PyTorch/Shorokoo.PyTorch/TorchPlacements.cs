using System.Security.Cryptography;
using Python.Runtime;
using Shorokoo.Core.Backends;
using Shorokoo.Core.Factory.IR;
using Shorokoo.PythonHost;
using Shorokoo.PythonTranslation;

namespace Shorokoo.PyTorch;

/// <summary>
/// Where a torch session's runs that consume inputs write their values in the memory of what they
/// consume (<see cref="PlacementProof"/>, under <see cref="PlacementMemory.PyTorch"/>), and the
/// translations that do the writing.
///
/// <para><b>Per run signature.</b> A placement needs the set of inputs a run consumes and the shape
/// of every input, which only a run knows, so placements are planned per signature — those, and the
/// outputs asked for — and kept, each with a translation of its own: the model's, with each placed
/// value's node made a write into its range (<c>_into</c>). The graph torch runs is the graph handed
/// over, as the translation writes it node for node, so the proof is over the graph that runs.</para>
///
/// <para><b>Kept without measuring.</b> A placed value is written where it goes by torch's own
/// operator writing into it, or by a copy of a view that would otherwise be copied out on the way
/// out; nothing a placement does allocates, so a plan never asks for more than the plain run and
/// is kept as soon as it is made.</para>
/// </summary>
internal sealed class TorchPlacements : IDisposable
{
    /// <summary>The largest model a session keeps to translate placements from.</summary>
    internal const int ModelBytesKept = 16 << 20;

    /// <summary>How many signatures a session plans for; a run of any other runs unplaced.</summary>
    private const int MostSignatures = 8;

    private const string TrainingDomain = "ai.shorokoo.training";

    private readonly ModelProto _model;
    private readonly IReadOnlyList<OutputAlias> _aliases;
    private readonly string[] _inputNames;
    private readonly IReadOnlyDictionary<string, int> _outputIndex;
    private readonly TorchRuntime _runtime;
    private readonly PyObject _constants;
    private readonly int _constantCount;
    private readonly object _gate = new();
    private readonly Dictionary<string, Entry> _entries = new(StringComparer.Ordinal);
    private bool _disposed;

    private TorchPlacements(
        ModelProto model, IReadOnlyList<OutputAlias> aliases, string[] inputNames, IReadOnlyDictionary<string, int> outputIndex,
        TorchRuntime runtime, PyObject constants, int constantCount)
    {
        _model = model;
        _aliases = aliases;
        _inputNames = inputNames;
        _outputIndex = outputIndex;
        _runtime = runtime;
        _constants = constants;
        _constantCount = constantCount;
    }

    /// <summary>
    /// The placements of a session over <paramref name="model"/>, or null where it places nothing: a
    /// model larger than <see cref="ModelBytesKept"/>, or a training step whose gradient torch
    /// computes — autograd keeps what the forward pass made for the backward pass, and a value
    /// written over in place would be read changed.
    /// </summary>
    internal static TorchPlacements? For(
        ModelProto model, int modelBytes, IReadOnlyList<OutputAlias> aliases, string[] inputNames,
        IReadOnlyDictionary<string, int> outputIndex, TorchRuntime runtime, PyObject constants, int constantCount)
    {
        if (modelBytes > ModelBytesKept || model.Graph is not { } graph) return null;
        if (graph.Nodes.Any(node => node.Domain == TrainingDomain)) return null;
        return new TorchPlacements(model, aliases, inputNames, outputIndex, runtime, constants, constantCount);
    }

    /// <summary>One run signature's placements and the translation that writes them.</summary>
    internal sealed class Entry
    {
        internal IReadOnlyList<Placement> Plan = [];
        internal string? Refusal;
        internal PyObject? Main;
        internal PyObject? Slots;
        internal int[] Blocks = [];
    }

    /// <summary>Told of every signature as it is planned, for a measurement to read; null where
    /// nothing listens.</summary>
    internal static Action<Entry>? Settled;

    /// <summary>Every signature planned so far, for a test to read.</summary>
    internal IReadOnlyList<Entry> Entries
    {
        get { lock (_gate) return [.. _entries.Values]; }
    }

    /// <summary>
    /// The inputs a run may place values in, by position among the session's inputs: each value the
    /// run consumed that it was fed under that one name, a tensor of fixed-width elements holding
    /// something, into which no alias pair writes an output — <paramref name="aliasTargets"/> names
    /// those.
    /// </summary>
    internal static SortedSet<int> Blocks(
        string[] inputNames, IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
        IReadOnlyCollection<IShorokooTensorValue> consumed, TorchTensorValue[] feeds, IReadOnlyCollection<int> aliasTargets)
    {
        var blocks = new SortedSet<int>();
        if (consumed.Count == 0) return blocks;
        var handed = new HashSet<IShorokooTensorValue>(consumed, ReferenceEqualityComparer.Instance);
        var fedAs = new Dictionary<IShorokooTensorValue, int>(ReferenceEqualityComparer.Instance);
        foreach (var value in inputs.Values) fedAs[value] = fedAs.GetValueOrDefault(value) + 1;
        for (int i = 0; i < feeds.Length; i++)
        {
            var feed = feeds[i];
            if (!handed.Contains(feed) || fedAs[feed] != 1 || aliasTargets.Contains(i)) continue;
            if (feed.ValueType != ShorokooOnnxValueType.Tensor || PlacementShapes.ElementBytes((int)feed.ElementType) == 0) continue;
            if (BytesOf(feed) <= 0) continue;
            blocks.Add(i);
        }
        return blocks;
    }

    internal static long BytesOf(TorchTensorValue value)
        => value.Shape.Aggregate(1L, (a, d) => a * d) * PlacementShapes.ElementBytes((int)value.ElementType);

    /// <summary>
    /// The entry for a run feeding <paramref name="feeds"/> that may place values in the inputs at
    /// <paramref name="blocks"/> and reads <paramref name="outputNames"/>, planned the first time its
    /// signature is seen; null where this session plans for as many signatures as it will. An entry
    /// whose plan places nothing has no translation, and its runs run plain.
    /// </summary>
    internal Entry? EntryFor(TorchTensorValue[] feeds, SortedSet<int> blocks, IReadOnlyList<string> outputNames)
    {
        if (blocks.Count == 0) return null;
        var key = new System.Text.StringBuilder();
        for (int i = 0; i < feeds.Length; i++)
        {
            var feed = feeds[i];
            key.Append(blocks.Contains(i) ? '!' : ':').Append((int)feed.ValueType).Append('/').Append((int)feed.ElementType)
                .Append('[').AppendJoin(',', feed.Shape).Append("];");
        }
        key.Append("->").AppendJoin(';', outputNames);
        lock (_gate)
        {
            if (_disposed) return null;
            if (_entries.TryGetValue(key.ToString(), out var entry)) return entry;
            if (_entries.Count >= MostSignatures) return null;
            entry = new Entry { Blocks = [.. blocks] };
            Prepare(entry, feeds, blocks, outputNames);
            Settled?.Invoke(entry);
            return _entries[key.ToString()] = entry;
        }
    }

    /// <summary>Plans <paramref name="entry"/> over the model's graph with the run's shapes, and
    /// translates and loads the model with its placed values written into their ranges.</summary>
    private void Prepare(Entry entry, TorchTensorValue[] feeds, SortedSet<int> blocks, IReadOnlyList<string> outputNames)
    {
        var graph = _model.Graph!;
        var given = new Dictionary<string, (long[] Shape, int ElementType)>(StringComparer.Ordinal);
        for (int i = 0; i < feeds.Length; i++)
            if (feeds[i].ValueType == ShorokooOnnxValueType.Tensor && feeds[i].ElementType != ShorokooTensorElementType.String)
                given[_inputNames[i]] = (feeds[i].Shape, (int)feeds[i].ElementType);
        var blockBytes = blocks.ToDictionary(i => _inputNames[i], i => BytesOf(feeds[i]), StringComparer.Ordinal);
        PlacementProof proof;
        Dictionary<string, PlacementShapes.Value> shapes;
        try
        {
            shapes = PlacementShapes.Evaluate(graph, given);
            proof = new PlacementProof(graph, blockBytes, shapes, outputNames.ToHashSet(StringComparer.Ordinal), PlacementMemory.PyTorch);
        }
        catch (ArgumentException ex)
        {
            entry.Refusal = ex.Message;
            return;
        }
        var plan = proof.Plan(PlacementProof.Smallest, PlacementProof.IdleOutputBytes);
        if (plan.Count == 0)
        {
            entry.Refusal = "nothing to place";
            return;
        }

        var placed = new Dictionary<string, PlacedValue>(StringComparer.Ordinal);
        for (int slot = 0; slot < plan.Count; slot++)
            placed[plan[slot].Value] = new PlacedValue(slot, proof.LeavesAChain(plan[slot].Value));
        // A node is written over its operand as in the plain translation, unless either is placed:
        // the plan relies on a placed value's range alone, and on the memory of every value it does
        // not place being its own.
        var writers = graph.Nodes.SelectMany(n => n.Outputs.Where(o => o.Length > 0).Select(o => (Output: o, Node: n)))
            .GroupBy(p => p.Output, StringComparer.Ordinal).ToDictionary(g => g.Key, g => g.First().Node, StringComparer.Ordinal);
        var over = TorchInPlace.Plan(graph)
            .Where(o => !placed.ContainsKey(o.Key) && !placed.ContainsKey(writers[o.Key].Inputs[o.Value]))
            .ToDictionary(o => o.Key, o => o.Value, StringComparer.Ordinal);
        TranslatedModel translated;
        try
        {
            translated = OnnxToPythonTranslator.Translate(_model, _aliases, TorchDialect.Instance, placed, over);
        }
        catch (NotSupportedException ex)
        {
            entry.Refusal = ex.Message;
            return;
        }
        if (translated.Constants.Count != _constantCount)
        {
            entry.Refusal = "the placed translation reads other constants";
            return;
        }
        var hash = Convert.ToHexStringLower(SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(translated.Source)))[..32];
        using (PythonRuntime.Gil())
        {
            var slots = new PyList();
            try
            {
                foreach (var placement in plan)
                {
                    var shape = shapes[placement.Value];
                    using var index = new PyInt(Array.IndexOf(_inputNames, placement.Block));
                    using var offset = new PyInt(placement.Offset);
                    using var bytes = new PyInt(placement.Bytes);
                    using var code = new PyInt(shape.ElementType);
                    using var dims = TorchBackend.Shape(shape.Shape);
                    using var output = new PyInt(_outputIndex.TryGetValue(placement.Value, out var o) ? o : -1);
                    using var tuple = new PyTuple([index, offset, bytes, code, dims, output]);
                    slots.Append(tuple);
                }
                entry.Main = PyCall.Invoke(_runtime.LoadModel, translated.Source, $"<shorokoo-model-{hash}>", _constants);
                entry.Slots = slots;
                entry.Plan = plan;
            }
            catch (PythonException ex)
            {
                slots.Dispose();
                entry.Refusal = ex.Format();
            }
        }
    }

    public void Dispose()
    {
        lock (_gate)
        {
            if (_disposed) return;
            _disposed = true;
            using (PythonRuntime.Gil())
                foreach (var entry in _entries.Values)
                {
                    entry.Main?.Dispose();
                    entry.Slots?.Dispose();
                }
        }
    }
}
