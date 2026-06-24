using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using System;

namespace Shorokoo;

/// <summary>
/// Lowers a high-level training graph (produced by
/// <see cref="Shorokoo.Core.Training.TrainingGraphBuilder.PrepareForTrainingAsFast(FastComputationGraph, FastComputationGraph)"/>)
/// into an autograd-flattened executable form, for use by the
/// AutoDiffCheckpointing tests that score graph-optimization strategies.
///
/// <para>
/// The production training path (<see cref="TrainingRig"/>)
/// has its own equivalent lowering inside <c>BuildTrainingStepPureGraph</c> that
/// additionally handles loop iteration-count folding; this helper is the
/// minimal autograd-only version kept around for the optimizer-evaluation
/// tests.
/// </para>
/// </summary>
public static class TrainingLoop
{
    /// <summary>
    /// Lowers a high-level training graph (autograd nodes + struct inputs/outputs)
    /// into an executable graph by expanding struct outputs, unpacking struct
    /// inputs, simplifying, then running the Fast-native autograd processor.
    /// </summary>
    public static FastComputationGraph LowerTrainingGraph(FastComputationGraph highLevelGraph)
    {
        if (highLevelGraph is null) throw new ArgumentNullException(nameof(highLevelGraph));

        var fast = highLevelGraph.Clone();
        Shorokoo.Core.Nodes.Processors.Fast.FastExpandStructOutputs.Process(fast);
        Shorokoo.Core.Nodes.Processors.Fast.FastUnpackTensorStructs.Process(fast);
        Shorokoo.Core.Nodes.Processors.Fast.FastSimplify.Process(fast);

        // Lower attribute-tensorized variant ops (e.g. SHRK_CONV) to standard ONNX ops before
        // autograd — they have no gradient rule and the simplify above resolves their geometry.
        Shorokoo.Core.Nodes.Processors.Fast.FastLowerAttributeTensorOps.Process(fast);

        Shorokoo.Core.Nodes.Processors.AutoGrad.FastProcessAutoGradProcessor.Process(fast);

        Shorokoo.Core.Nodes.Processors.Fast.FastSimplify.Process(fast);
        return fast;
    }
}
