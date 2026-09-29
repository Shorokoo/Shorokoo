using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.Processors.Fast;
using Shorokoo.Graph;

namespace Shorokoo.Core.Lowering.KernelWorkarounds;

/// <summary>
/// What a <see cref="KernelWorkaround"/> may know about one call when the model is built: the
/// operator, its attributes, the dtype and rank of each input and output slot, which slots are
/// present, which inputs come from outside the loop or branch body the call is in, which outputs
/// anything reads, and the value of an input a <c>Constant</c> produces.
///
/// <para>Reading a constant is recorded, so a replacement built from one call's constant is reused
/// only for calls whose constant agrees in what was read: its value, through
/// <see cref="ConstantOf"/>; only its shape, through <see cref="ConstantShapeOf"/>. A workaround
/// that needs no more than the shape asks for no more, so calls whose constants differ only in
/// value share one replacement.</para>
/// </summary>
internal sealed class WorkaroundSite
{
    private readonly FastNode node;
    private readonly FastTensorKey?[] inputKeys;
    private readonly FastTensorKey?[] outputKeys;
    private readonly (DType DType, int? Rank)?[] inputs;
    private readonly (DType DType, int? Rank)?[] outputs;
    private readonly bool[] outsideBody;
    private readonly IReadOnlyDictionary<FastTensorKey, FastNode> producers;
    private readonly IReadOnlySet<FastTensorKey> read;
    private readonly SortedDictionary<int, ConstantRead> constantsRead = [];
    private readonly bool shapesAreConcrete;
    private readonly bool inLoopBody;

    private WorkaroundSite(
        FastNode node,
        FastTensorKey?[] inputKeys,
        FastTensorKey?[] outputKeys,
        (DType DType, int? Rank)?[] inputs,
        (DType DType, int? Rank)?[] outputs,
        bool[] outsideBody,
        IReadOnlyDictionary<FastTensorKey, FastNode> producers,
        IReadOnlySet<FastTensorKey> read,
        bool shapesAreConcrete,
        bool inLoopBody)
    {
        this.outsideBody = outsideBody;
        this.shapesAreConcrete = shapesAreConcrete && !inLoopBody;
        this.inLoopBody = inLoopBody;
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
    /// <paramref name="shapesAreConcrete"/> says whether the model is built with every graph
    /// input's dimensions stated, <paramref name="inLoopBody"/> is <see cref="IsInLoopBody"/>, and
    /// <paramref name="outsideBody"/> tells, of an input, <see cref="IsFromOutsideBody"/>; without
    /// it, no input is.
    /// </summary>
    internal static WorkaroundSite? TryCreate(
        FastNode node,
        IReadOnlyDictionary<FastTensorKey, FastTensorInfo> tensorInfo,
        IReadOnlyDictionary<FastTensorKey, FastNode> producers,
        IReadOnlySet<FastTensorKey> read,
        bool shapesAreConcrete = false,
        bool inLoopBody = false,
        Func<FastTensorKey, bool>? outsideBody = null)
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

        bool[] outside = [.. inputKeys.Select(k => k is { } key && outsideBody is not null && outsideBody(key))];
        return new WorkaroundSite(node, inputKeys, outputKeys, inputs, outputs, outside, producers, read, shapesAreConcrete, inLoopBody);
    }

    /// <summary>
    /// Whether the model is built with every graph input's dimensions stated and the call is not
    /// in a loop body (<see cref="IsInLoopBody"/>), so the backend knows the shapes the call reads
    /// that the graph computes from those dimensions when it builds the session, and folds what is
    /// computed from those shapes alone. A replacement that decides by shape can then leave the
    /// decision to that fold. A shape computed from the data, a <c>NonZero</c>'s or one read from
    /// a tensor's values by a <c>Reshape</c> or <c>Expand</c>, say, is known only when the model
    /// runs, and a decision on it is made on every run.
    /// </summary>
    public bool ShapesAreConcrete => shapesAreConcrete;

    /// <summary>
    /// Whether the call is in the body of a <c>Loop</c> or a <c>SequenceMap</c>, which runs once
    /// per iteration or element. A body's inputs are typed by rank alone, so the backend does not
    /// fold a decision on their shapes when it builds the session, and such a decision would be
    /// made again on every iteration: <see cref="ShapesAreConcrete"/> is false there.
    /// </summary>
    public bool IsInLoopBody => inLoopBody;

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

    /// <summary>
    /// Whether the call is in a loop or branch body and input slot <paramref name="slot"/> is a
    /// value that body did not compute: one from an enclosing scope, or one of the body's own
    /// inputs, such as the iteration number or a loop-carried value. False for an absent slot, and
    /// for every slot of a call outside any body.
    /// </summary>
    public bool IsFromOutsideBody(int slot) => slot < outsideBody.Length && outsideBody[slot];

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
        constantsRead[slot] = new ConstantRead(value, Values: true);
        return value;
    }

    /// <summary>
    /// The shape of input slot <paramref name="slot"/> when a <c>Constant</c> produces it; null
    /// for every other input. Its dtype is <see cref="DTypeOf"/>. The read is recorded for the plan
    /// cache as a read of the shape alone.
    /// </summary>
    public Shape? ConstantShapeOf(int slot)
    {
        var value = PeekConstant(slot);
        if (!constantsRead.TryGetValue(slot, out var earlier) || !earlier.Values)
            constantsRead[slot] = new ConstantRead(value, Values: false);
        return value?.Shape;
    }

    /// <summary>The descriptor of each input slot, null for an absent one.</summary>
    internal (DType DType, int? Rank)?[] InputDescriptors => inputs;

    /// <summary>The key of each output slot, null for an absent one.</summary>
    internal FastTensorKey?[] OutputKeys => outputKeys;

    /// <summary>
    /// What one site read of the constant in one slot: the constant (null for an input no
    /// <c>Constant</c> produces), and whether its <paramref name="Values"/> were read or only its
    /// shape.
    /// </summary>
    internal readonly record struct ConstantRead(TensorAttribute? Constant, bool Values);

    /// <summary>The input slots whose constant has been read, with what was found.</summary>
    internal IReadOnlyDictionary<int, ConstantRead> ConstantsRead => constantsRead;

    /// <summary>
    /// Whether this site's constants agree with <paramref name="reads"/>, another site's, in all
    /// it read: in each slot, a constant or none; its dtype and shape; and, where the values were
    /// read, every value. Nothing is recorded.
    /// </summary>
    internal bool Reads(IReadOnlyDictionary<int, ConstantRead> reads)
    {
        foreach (var (slot, read) in reads)
        {
            var mine = PeekConstant(slot);
            if (mine is null || read.Constant is not { } theirs)
            {
                if (mine is not null || read.Constant is not null) return false;
                continue;
            }
            if (mine.DType != theirs.DType || !mine.Shape.Equals(theirs.Shape)) return false;
            if (!read.Values || ReferenceEquals(mine, theirs)) continue;
            if (mine.DType == DType.Utf8 ? !mine.Values.SequenceEqual(theirs.Values, StringComparer.Ordinal)
                : !mine.Bytes.SequenceEqual(theirs.Bytes))
                return false;
        }
        return true;
    }

    /// <summary>Everything besides constants and the input descriptors that a replacement may depend
    /// on: which inputs come from outside the body, the dtype and rank of each output and whether it
    /// is read, and <see cref="ShapesAreConcrete"/>.</summary>
    internal string OutputFingerprint()
        => new string([.. outsideBody.Select(o => o ? 'o' : '-')]) + "|" + (shapesAreConcrete ? "c;" : "s;")
        + string.Join(";", Enumerable.Range(0, outputKeys.Length).Select(i =>
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

}
