namespace Shorokoo.Core.Backends;

/// <summary>
/// An output whose element type and shape a session settled when it was built — every dimension a
/// count, or the length of a dimension of one of its inputs — so that every run of it is handed
/// the output's memory before it starts: memory of the output's own, in the run memory of the
/// session's backend and outside the session's arena, rather than memory the run finds as it goes.
/// That memory is what a device-memory budget has to leave room for outside the arena for the
/// length of a run (<see cref="IShorokooSession.SettledOutputs"/>).
/// </summary>
/// <param name="Name">The output, as the session names it.</param>
/// <param name="ElementType">Its element type, one of a fixed width.</param>
/// <param name="Dimensions">Its dimensions, in order; none for a scalar.</param>
public sealed record SettledOutput(
    string Name, ShorokooTensorElementType ElementType, IReadOnlyList<OutputDimension> Dimensions)
{
    /// <summary>An output of <paramref name="shape"/>, every dimension a count.</summary>
    public SettledOutput(string name, ShorokooTensorElementType elementType, long[] shape)
        : this(name, elementType, [.. shape.Select(OutputDimension.Of)])
    {
    }

    /// <summary>
    /// The output's shape for a run fed inputs whose shapes <paramref name="inputShape"/> answers,
    /// by the session's input name — null for an input that is not a tensor, or not fed — or null
    /// where a dimension comes from such an input.
    /// </summary>
    public long[]? ShapeFor(Func<string, long[]?> inputShape)
    {
        ArgumentNullException.ThrowIfNull(inputShape);
        var shape = new long[Dimensions.Count];
        for (int i = 0; i < shape.Length; i++)
        {
            var dimension = Dimensions[i];
            if (dimension.Input is null)
                shape[i] = dimension.Count;
            else if (inputShape(dimension.Input) is { } fed && dimension.Axis < fed.Length)
                shape[i] = fed[dimension.Axis];
            else
                return null;
        }
        return shape;
    }

    /// <summary>The bytes the output takes in a run fed inputs whose shapes
    /// <paramref name="inputShape"/> answers, or null where its shape cannot be told
    /// (<see cref="ShapeFor"/>).</summary>
    public long? ByteCountFor(Func<string, long[]?> inputShape)
        => ShapeFor(inputShape) is { } shape ? TensorElementLayout.ByteLength(ElementType, shape) : null;
}

/// <summary>
/// One dimension of a <see cref="SettledOutput"/>: a count, or the length of dimension
/// <see cref="Axis"/> of input <see cref="Input"/>, whatever that is on a given run.
/// </summary>
/// <param name="Count">The dimension's length, where it is a count.</param>
/// <param name="Input">The input whose dimension it is, by the session's name for it; null where it
/// is a count.</param>
/// <param name="Axis">Which of that input's dimensions it is.</param>
public readonly record struct OutputDimension(long Count, string? Input, int Axis)
{
    /// <summary>A dimension of <paramref name="count"/>.</summary>
    public static OutputDimension Of(long count) => new(count, null, 0);

    /// <summary>A dimension as long as dimension <paramref name="axis"/> of input
    /// <paramref name="input"/>.</summary>
    public static OutputDimension OfInput(string input, int axis) => new(0, input, axis);
}
