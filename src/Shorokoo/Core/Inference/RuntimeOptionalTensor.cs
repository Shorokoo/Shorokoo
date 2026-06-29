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
/// Runtime wrapper for an ONNX optional tensor. Immutable; produce copies via <c>with</c>.
/// </summary>
internal sealed record class RuntimeOptionalTensor : IRuntimeTensor
{
    /// <inheritdoc />
    public DType DType { get; init; } = DType.Invalid;

    /// <inheritdoc />
    public Variable? ReferenceTensor { get; init; }

    /// <inheritdoc />
    public ImmutableArray<long>? IterationIndices { get; init; }

    /// <inheritdoc />
    public ImmutableArray<IRuntimeTensor>? History { get; init; }

    /// <summary>
    /// Whether this optional actually contains a value. Null when not statically known.
    /// </summary>
    public bool? HasValue { get; init; }

    /// <summary>
    /// The tensor held by this optional. Null when <see cref="HasValue"/> is known-false.
    /// </summary>
    public RuntimeTensor? ValueTensor { get; init; }

    public override string ToString()
    {
        var state = HasValue switch { true => "some", false => "none", _ => "?" };
        return $"Opt[{state} {ValueTensor}]";
    }
}
