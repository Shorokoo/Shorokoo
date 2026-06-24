using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Onnx;
using Shorokoo.Core.Nodes.AutoDiff;
using static Shorokoo.Globals;

namespace Shorokoo.Modules.Losses;

/// <summary>
/// L2 (Mean Squared Error) loss function module.
/// Computes: loss = mean((predictions - targets)^2) over ALL elements, for
/// predictions of any rank. Flattens before a ReduceMean with explicit axis 0
/// for clean gradient computation.
/// <para>
/// The 2-input <see cref="Inline"/> is the rig-safe default. <see cref="Reduced"/>
/// (scalar) and <see cref="PerElement"/> (tensor) add the <c>reduction</c> knob.
/// </para>
/// </summary>
[Module]
public partial class L2Loss
{
    public static Scalar<float32> Inline(Tensor<float32> predictions, Tensor<float32> targets)
    {
        var diff = predictions - targets;
        var squared = (diff * diff).Reshape(Vector(-1L));
        var axes = Vector(0L);
        var reduced = (Tensor<float32>)OnnxOp.ReduceMean((ITensor)squared, axes, keepdims: false);
        return reduced.Scalar();
    }

    /// <summary>
    /// MSE with a configurable reduction, reduced to a scalar. <c>mean</c>/<c>sum</c>
    /// reduce the flattened squared error over all elements (as <see cref="Inline"/> does).
    /// </summary>
    /// <param name="predictions">Predicted values.</param>
    /// <param name="targets">Target values.</param>
    /// <param name="reduction"><see cref="LossReduction.Mean"/> (default) or
    /// <see cref="LossReduction.Sum"/>; use <see cref="PerElement"/> for
    /// <see cref="LossReduction.None"/>.</param>
    public static Scalar<float32> Reduced(
        Tensor<float32> predictions,
        Tensor<float32> targets,
        LossReduction reduction = LossReduction.Mean)
    {
        if (reduction == LossReduction.None)
            throw new System.ArgumentException(
                "LossReduction.None returns a per-element tensor; call PerElement instead.",
                nameof(reduction));
        var diff = predictions - targets;
        var squared = (diff * diff).Reshape(Vector(-1L));
        return reduction.ApplyToPerElement(squared).Scalar();
    }

    /// <summary>
    /// Per-element squared error <c>(predictions − targets)²</c>
    /// (<see cref="LossReduction.None"/> semantics), keeping the input shape.
    /// </summary>
    public static Tensor<float32> PerElement(Tensor<float32> predictions, Tensor<float32> targets)
    {
        var diff = predictions - targets;
        return diff * diff;
    }
}
