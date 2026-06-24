using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Onnx;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;
using System.Collections.Immutable;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// Shape inference for ONNX <c>NonMaxSuppression</c>. Output is a 2-D int64 tensor with shape
/// <c>[num_selected_indices, 3]</c> (columns: batch_index, class_index, box_index). The
/// leading dim is data-dependent, so only the rank (2) is reported — never a placeholder dim
/// that would leak through Shape-op value chains. When the scores shape and the
/// max_output_boxes_per_class value are known, a per-dimension upper bound
/// (<c>MaxShape = [batches * classes * min(spatial, max_boxes), 3]</c>) is also produced.
/// One exact case exists: an absent max_output_boxes_per_class input defaults to 0 per spec
/// ("no output"), making the output exactly <c>[0, 3]</c>.
/// </summary>
internal sealed class NonMaxSuppressionOp : QuickOp
{
    public override string OpCode => OpCodes.NON_MAX_SUPPRESSION;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var scores = inputs.Length > 1 ? inputs[1] : null;
        var maxBoxes = inputs.Length > 2 ? inputs[2] : null;

        // Absent max_output_boxes_per_class defaults to 0 → no boxes are ever selected, and
        // the same holds for a connected input whose value is concretely 0.
        var maxBoxesKnown = maxBoxes is null ? 0L
            : maxBoxes.IntData is { Length: > 0 } mb ? mb[0]
            : (long?)null;
        if (maxBoxesKnown is 0 || maxBoxesKnown < 0)
        {
            return [RuntimeTensorFactory.Create(DType.Int64, new Shape(new long[] { 0, 3 }))
                with { IntData = ImmutableArray<long>.Empty }];
        }

        Shape? maxShape = null;
        if (scores?.Shape?.Dims is { Length: 3 } sDims && sDims[0] >= 0 && sDims[1] >= 0 && sDims[2] >= 0)
        {
            var perClass = maxBoxesKnown is { } cap ? Math.Min(sDims[2], cap) : sDims[2];
            maxShape = new Shape(new[] { sDims[0] * sDims[1] * perClass, 3 });
        }

        return [new RuntimeTensor
        {
            DType = DType.Int64,
            Shape = null,
            MaxShape = maxShape,
            Rank = 2,
            MaxRank = 2,
        }];
    }
}
