using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.Processors.Fast;
using Shorokoo.Graph;

namespace Shorokoo.Core.Lowering.KernelWorkarounds;

/// <summary>
/// What a <see cref="KernelWorkaround"/> may know about one call when the model is built: the
/// operator, its attributes, the dtype and rank of each input and output slot, which slots are
/// present, which outputs anything reads, and the value of an input a <c>Constant</c> produces.
///
/// <para>Reading a constant through <see cref="ConstantOf"/> is recorded, so a replacement built
/// from one call's constant is reused only for calls whose constant in that slot is the same.</para>
/// </summary>
internal sealed class WorkaroundSite
{
    private readonly FastNode node;
    private readonly FastTensorKey?[] inputKeys;
    private readonly FastTensorKey?[] outputKeys;
    private readonly (DType DType, int? Rank)?[] inputs;
    private readonly (DType DType, int? Rank)?[] outputs;
    private readonly IReadOnlyDictionary<FastTensorKey, FastNode> producers;
    private readonly IReadOnlySet<FastTensorKey> read;
    private readonly SortedDictionary<int, string> constantsRead = [];
    private readonly bool shapesAreConcrete;

    private WorkaroundSite(
        FastNode node,
        FastTensorKey?[] inputKeys,
        FastTensorKey?[] outputKeys,
        (DType DType, int? Rank)?[] inputs,
        (DType DType, int? Rank)?[] outputs,
        IReadOnlyDictionary<FastTensorKey, FastNode> producers,
        IReadOnlySet<FastTensorKey> read,
        bool shapesAreConcrete)
    {
        this.shapesAreConcrete = shapesAreConcrete;
        this.node = node;
        this.inputKeys = inputKeys;
        this.outputKeys = outputKeys;
        this.inputs = inputs;
        this.outputs = outputs;
        this.producers = producers;
        this.read = read;
    }

    /// <summary>
    /// The site of <paramref name="node"/>, or null when the graph does not tell every present
    /// input's dtype, or some input is not a tensor: a call the pass leaves as it stands.
    /// <paramref name="shapesAreConcrete"/> is <see cref="ShapesAreConcrete"/>.
    /// </summary>
    internal static WorkaroundSite? TryCreate(
        FastNode node,
        IReadOnlyDictionary<FastTensorKey, FastTensorInfo> tensorInfo,
        IReadOnlyDictionary<FastTensorKey, FastNode> producers,
        IReadOnlySet<FastTensorKey> read,
        bool shapesAreConcrete = false)
    {
        FastTensorKey?[] inputKeys = [.. node.Inputs.Select(k => k is { IsEmpty: false } ? k : null)];
        FastTensorKey?[] outputKeys = [.. node.Outputs.Select(k => k is { IsEmpty: false } ? k : null)];

        var inputs = new (DType DType, int? Rank)?[inputKeys.Length];
        for (int i = 0; i < inputKeys.Length; i++)
        {
            if (inputKeys[i] is not { } key) continue;
            if (!tensorInfo.TryGetValue(key, out var info) || info.Structure != DataStructure.Tensor
                || info.DType == DType.Invalid || info.DType.IsGenericType)
                return null;
            inputs[i] = (info.DType, info.Rank);
        }

        var outputs = new (DType DType, int? Rank)?[outputKeys.Length];
        for (int i = 0; i < outputKeys.Length; i++)
            if (outputKeys[i] is { } key && tensorInfo.TryGetValue(key, out var info))
                outputs[i] = (info.DType, info.Rank);

        return new WorkaroundSite(node, inputKeys, outputKeys, inputs, outputs, producers, read, shapesAreConcrete);
    }

    /// <summary>
    /// Whether the model is built with every graph input's dimensions stated, so the backend knows
    /// every shape the graph computes from them when it builds the session, and folds what is
    /// computed from those shapes alone. A replacement that decides by shape can then leave the
    /// decision to that fold. The same for every call of one build.
    /// </summary>
    public bool ShapesAreConcrete => shapesAreConcrete;

    /// <summary>The operator's op code.</summary>
    public string OpCode => node.OpCode;

    /// <summary>The operator's attributes.</summary>
    public OnnxCSharpAttributes Attributes => node.Attributes;

    /// <summary>The number of input slots, present or not.</summary>
    public int InputCount => inputKeys.Length;

    /// <summary>The number of output slots, present or not.</summary>
    public int OutputCount => outputKeys.Length;

    /// <summary>Whether input slot <paramref name="slot"/> is given.</summary>
    public bool IsPresent(int slot) => slot < inputKeys.Length && inputKeys[slot] is not null;

    /// <summary>The dtype of input slot <paramref name="slot"/>; <see cref="DType.Invalid"/> for
    /// an absent one.</summary>
    public DType DTypeOf(int slot) => slot < inputs.Length && inputs[slot] is { } d ? d.DType : DType.Invalid;

    /// <summary>The rank of input slot <paramref name="slot"/>, when the graph tells it.</summary>
    public int? RankOf(int slot) => slot < inputs.Length && inputs[slot] is { } d ? d.Rank : null;

    /// <summary>Whether output slot <paramref name="slot"/> is produced.</summary>
    public bool IsOutputPresent(int slot) => slot < outputKeys.Length && outputKeys[slot] is not null;

    /// <summary>The dtype of output slot <paramref name="slot"/>; <see cref="DType.Invalid"/>
    /// when the graph does not tell it.</summary>
    public DType OutputDTypeOf(int slot) => slot < outputs.Length && outputs[slot] is { } d ? d.DType : DType.Invalid;

    /// <summary>The rank of output slot <paramref name="slot"/>, when the graph tells it.</summary>
    public int? OutputRankOf(int slot) => slot < outputs.Length && outputs[slot] is { } d ? d.Rank : null;

    /// <summary>Whether anything in the graph reads output slot <paramref name="slot"/>.</summary>
    public bool IsOutputUsed(int slot) => slot < outputKeys.Length && outputKeys[slot] is { } key && read.Contains(key);

    /// <summary>
    /// The value of input slot <paramref name="slot"/> when a <c>Constant</c> produces it; null
    /// for every other input. The read is recorded for the plan cache.
    /// </summary>
    public TensorAttribute? ConstantOf(int slot)
    {
        var value = PeekConstant(slot);
        constantsRead[slot] = Fingerprint(value);
        return value;
    }

    /// <summary>The descriptor of each input slot, null for an absent one.</summary>
    internal (DType DType, int? Rank)?[] InputDescriptors => inputs;

    /// <summary>The key of each output slot, null for an absent one.</summary>
    internal FastTensorKey?[] OutputKeys => outputKeys;

    /// <summary>The input slots <see cref="ConstantOf"/> has read, with what it found.</summary>
    internal IReadOnlyDictionary<int, string> ConstantsRead => constantsRead;

    /// <summary>What <see cref="ConstantOf"/> would find in <paramref name="slot"/>, as the
    /// plan cache compares it, without recording the read.</summary>
    internal string FingerprintOf(int slot) => Fingerprint(PeekConstant(slot));

    /// <summary>Everything besides constants that a replacement may depend on: the dtype and rank
    /// of each output and whether it is read.</summary>
    internal string OutputFingerprint()
        => string.Join(";", Enumerable.Range(0, outputKeys.Length).Select(i =>
            outputKeys[i] is null ? "~" : $"{OutputDTypeOf(i)},{OutputRankOf(i) ?? -1},{(IsOutputUsed(i) ? 'u' : '-')}"));

    private TensorAttribute? PeekConstant(int slot)
    {
        if (slot >= inputKeys.Length || inputKeys[slot] is not { } key
            || !producers.TryGetValue(key, out var producer) || producer.OpCode != OpCodes.CONSTANT)
            return null;
        var attributes = producer.Attributes.GetAttributeVals();
        object? Get(string name) => attributes.GetValueOrDefault(name);
        return Get(OnnxOpAttributeNames.AttrValue) is TensorAttribute { HasValues: true } value ? value
            : Get(OnnxOpAttributeNames.AttrValueInts) is long[] ints ? TensorAttribute.Create(new Shape((long)ints.Length), ints)
            : Get(OnnxOpAttributeNames.AttrValueInt) is long single ? TensorAttribute.Create(new Shape(), single)
            : Get(OnnxOpAttributeNames.AttrValueFloats) is float[] floats ? TensorAttribute.Create(new Shape((long)floats.Length), floats)
            : Get(OnnxOpAttributeNames.AttrValueFloat) is float scalar ? TensorAttribute.Create(new Shape(), scalar)
            : null;
    }

    private static string Fingerprint(TensorAttribute? value)
    {
        if (value is null) return "~";
        var head = $"{value.DType}[{string.Join(",", value.Shape.Dims)}]";
        return value.DType == DType.Utf8
            ? head + string.Join("\u0001", value.Values.Select(v => $"{v.Length}:{v}"))
            : head + Convert.ToHexString(value.Bytes);
    }
}
