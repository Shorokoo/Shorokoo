using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Graph;
using static Shorokoo.Globals;

namespace Shorokoo.Modules.Losses;

/// <summary>
/// How a per-element loss tensor is aggregated into the reported loss, matching
/// the PyTorch / ONNX <c>reduction</c> semantics:
/// <list type="bullet">
/// <item><see cref="None"/> — no aggregation; the per-element loss tensor is
///   returned unchanged (a non-scalar result, so it does not satisfy the
///   <c>TrainingRig</c> 2-input/scalar loss contract).</item>
/// <item><see cref="Mean"/> — the (weighted) average over elements (the default).</item>
/// <item><see cref="Sum"/> — the sum over all elements.</item>
/// </list>
/// This is a build-time C# enum (baked into the graph at build), not a
/// <c>[Hyper]</c>: the value is architectural and <see cref="None"/> changes the
/// output shape, so it cannot be a scalar hyperparameter.
/// </summary>
public enum LossReduction
{
    /// <summary>Return the per-element loss tensor unchanged (non-scalar).</summary>
    None,

    /// <summary>Return the (weighted) average over all elements.</summary>
    Mean,

    /// <summary>Return the sum over all elements.</summary>
    Sum,
}

/// <summary>
/// Internal helpers mapping <see cref="LossReduction"/> onto the core-op
/// reduction string (for the cross-entropy family, which reduces inside the
/// ONNX op) and onto a per-element tensor reduction (for the regression / BCE
/// losses, which reduce in C#).
/// </summary>
internal static class LossReductionExtensions
{
    /// <summary>Maps to the ONNX loss-op <c>reduction</c> attribute string.</summary>
    internal static string ToOnnxReduction(this LossReduction reduction) => reduction switch
    {
        LossReduction.None => "none",
        LossReduction.Sum => "sum",
        _ => "mean",
    };

    /// <summary>
    /// Applies the reduction to an already-computed per-element loss tensor:
    /// <see cref="LossReduction.None"/> returns the tensor unchanged,
    /// <see cref="LossReduction.Mean"/> / <see cref="LossReduction.Sum"/> reduce
    /// over all elements to a rank-0 tensor.
    /// </summary>
    internal static Tensor<float32> ApplyToPerElement(
        this LossReduction reduction, Tensor<float32> perElement) => reduction switch
    {
        LossReduction.None => perElement,
        LossReduction.Sum => perElement.Reduce(ReduceKind.Sum, keepDims: false),
        _ => perElement.Reduce(ReduceKind.Mean, keepDims: false),
    };
}
