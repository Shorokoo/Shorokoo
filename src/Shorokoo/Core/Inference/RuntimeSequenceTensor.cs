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
/// Runtime wrapper for an ONNX tensor sequence. Two representations are supported and
/// exactly one is active at a time (invariant: <see cref="Tensors"/> xor
/// <see cref="TemplateTensor"/>). Immutable; produce copies via <c>with</c>.
/// </summary>
internal sealed record class RuntimeSequenceTensor : IRuntimeTensor
{
    /// <inheritdoc />
    public DType DType { get; init; } = DType.Invalid;

    /// <inheritdoc />
    public Variable? ReferenceTensor { get; init; }

    /// <inheritdoc />
    public ImmutableArray<long>? IterationIndices { get; init; }

    /// <inheritdoc />
    public ImmutableArray<IRuntimeTensor>? History { get; init; }

    /// <summary>Number of tensors in the sequence when known.</summary>
    public long? Count { get; init; }

    /// <summary>
    /// Concrete list of per-element tensors (each entry non-null). When populated,
    /// <see cref="TemplateTensor"/> must be null.
    /// </summary>
    public ImmutableArray<RuntimeTensor>? Tensors { get; init; }

    /// <summary>
    /// Summary tensor that carries information true for every element the sequence contains
    /// (or could contain). When populated, <see cref="Tensors"/> must be null.
    /// </summary>
    public RuntimeTensor? TemplateTensor { get; init; }

    public override string ToString()
    {
        var countStr = Count?.ToString() ?? "?";
        if (Tensors is not null) return $"Seq[{countStr}, concrete]";
        if (TemplateTensor is not null) return $"Seq[{countStr}, template={TemplateTensor}]";
        return $"Seq[{countStr}, empty]";
    }
}
