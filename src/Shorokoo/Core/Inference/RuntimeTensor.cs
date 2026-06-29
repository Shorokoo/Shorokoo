using System.Collections.Immutable;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Onnx;

namespace Shorokoo.Core.Inference;

/// <summary>
/// A plain runtime tensor. Tracks the data type, (possibly partial) shape/rank information, a
/// reference to the original graph tensor, and optionally the concrete element values. Values
/// are only populated for small tensors (see <see cref="QuickExecutionEngine.MaxDataElements"/>).
///
/// Immutable: produce modified copies with C#'s <c>with</c> expression. Array-valued payloads
/// are <see cref="ImmutableArray{T}"/> so no consumer can mutate them.
/// </summary>
internal sealed record class RuntimeTensor : IRuntimeTensor
{
    /// <summary>The logical data type of this tensor. Always known (never null).</summary>
    public DType DType { get; init; } = DType.Invalid;

    /// <summary>
    /// The exact shape of this tensor when fully known. Null when only an upper bound is known.
    /// </summary>
    public Shape? Shape { get; init; }

    /// <summary>
    /// A per-dimension upper bound for the shape. Null when not known. Equal to Shape when the
    /// shape is fully known.
    /// </summary>
    public Shape? MaxShape { get; init; }

    /// <summary>
    /// Exact rank of the tensor when known. Null when only an upper bound is known.
    /// </summary>
    public int? Rank { get; init; }

    /// <summary>
    /// Upper bound on rank when known. Equal to Rank when rank is exactly known.
    /// </summary>
    public int? MaxRank { get; init; }

    /// <inheritdoc />
    public Variable? ReferenceTensor { get; init; }

    /// <summary>Data for float-like dtypes. Null if unknown or too large.</summary>
    public ImmutableArray<float>? FloatData { get; init; }

    /// <summary>Data for integer-like dtypes. Null if unknown or too large.</summary>
    public ImmutableArray<long>? IntData { get; init; }

    /// <summary>Data for string dtypes. Null if unknown or too large.</summary>
    public ImmutableArray<string>? StringData { get; init; }

    /// <summary>Data for boolean dtype. Null if unknown or too large.</summary>
    public ImmutableArray<bool>? BoolData { get; init; }

    /// <inheritdoc />
    public ImmutableArray<long>? IterationIndices { get; init; }

    /// <inheritdoc />
    public ImmutableArray<IRuntimeTensor>? History { get; init; }

    public bool HasAnyData =>
        FloatData is not null || IntData is not null || StringData is not null || BoolData is not null;

    public bool HasDefiniteShape => Shape is not null && Shape.Dims.All(d => d >= 0);

    public long? ElementCount => Shape is null ? null : Shape.Count;

    public override string ToString()
    {
        var shapeStr = Shape?.ToString() ?? (MaxShape is null ? "?" : $"≤{MaxShape}");
        return $"RT[{DType} {shapeStr}]";
    }
}
