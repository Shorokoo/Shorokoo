using System;
using System.Collections.Generic;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;
using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;
using static Shorokoo.Globals;

namespace Shorokoo.Core.Nodes.NodeDefinitions
{
    // Out-of-line (runtime-only) test data for the MQ op group. Moved here so the structural op
    // definitions in Definitions.MQ.cs stay free of runtime-typed, test-only values. Applied to each
    // maker post-construction, keyed by op code, by Definitions.NodeDefinitions.
    internal static partial class Definitions
    {
        static partial void RegisterMQTestData(Dictionary<string, Action<NodeDefinitionMaker>> r)
        {
            r[MATMUL] = m => m
                .InputTestShapes("a", [[1, 2, 3, 4], [2, 1, 5, 2]])
                .InputTestShapes("b", [[3, 1, 4, 2], [2, 2]]);

            r[MATMUL_INTEGER] = m => m
                .InputTestShapes("a", [[1, 2, 3, 4], [2, 1, 5, 2]])
                .InputTestShapes("b", [[3, 1, 4, 2], [2, 2]])
                .InputTestShapes("a_zero_point", [[], []])
                .InputTestShapes("b_zero_point", [[], []]);

            r[MAX] = m => m
                .VariadicInputTestShapes([[[3, 1, 2], [1, 2, 2]], [[2, 3], [2, 1, 3], [2, 2, 3]], [[1, 2, 3, 4], [3, 2, 3, 1], [3, 1, 1, 1], []]]);

            // MaxPool declares two variants; the original chain set test values for each in order
            // (last write wins on the maker's single test-data dict), reproduced here verbatim.
            r[MAX_POOL] = m => m
                .AttributeTestValues(InternalAttrHasOptionalOutputs, [1L])
                .AttributeTestValues(AttrAutoPad, [AutoPad.NotSet])
                .AttributeTestValues(AttrCeilMode, [0L])
                .AttributeTestValues(AttrDilations, (long[]?[])[[1, 1]])
                .AttributeTestValues(AttrKernelShape, (long[]?[])[[3, 3]])
                .AttributeTestValues(AttrPads, (long[]?[])[[0, 0, 0, 0]])
                .AttributeTestValues(AttrStorageOrder, [0L])
                .AttributeTestValues(AttrStrides, (long[]?[])[[1, 1]])
                .AttributeTestValues(InternalAttrHasOptionalOutputs, [0L])
                .AttributeTestValues(AttrAutoPad, [AutoPad.NotSet])
                .AttributeTestValues(AttrCeilMode, [0L])
                .AttributeTestValues(AttrDilations, (long[]?[])[[1, 1]])
                .AttributeTestValues(AttrKernelShape, (long[]?[])[[3, 3]])
                .AttributeTestValues(AttrPads, (long[]?[])[[0, 0, 0, 0]])
                .AttributeTestValues(AttrStorageOrder, [0L])
                .AttributeTestValues(AttrStrides, (long[]?[])[[1, 1]]);

            r[MAX_UNPOOL] = m => m
                .InputTestShapes("X", [[1L, 1L, 2L, 2L]])
                .InputTestShapes("I", [[1L, 1L, 2L, 2L]])
                .AttributeTestValues(AttrKernelShape, (long[][])[[2L, 2L]])
                .AttributeTestValues(AttrStrides, (long[][])[[2L, 2L]]);

            r[MAX_ROI_POOL] = m => m
                .InputTestShapes("X", [[1L, 1L, 4L, 4L]])
                .InputTestValues("rois", [TensorData([1, 5], 0f, 0f, 0f, 3f, 3f)])
                .AttributeTestValues(AttrPooledShape, (long[][])[[2L, 2L]])
                .AttributeTestValues(AttrSpatialScale, [1f]);

            r[MIN] = m => m
                .VariadicInputTestShapes([[[3, 1, 2], [1, 2, 2]], [[2, 3], [2, 1, 3], [2, 2, 3]], [[1, 2, 3, 4], [3, 2, 3, 1], [3, 1, 1, 1], []]]);

            r[MEAN] = m => m
                .VariadicInputTestShapes([[[3, 1, 2], [1, 2, 2]], [[2, 3], [2, 1, 3], [2, 2, 3]]]);

            r[MOD] = m => m.WithBroadcastTestShapes();
            r[MUL] = m => m.WithBroadcastTestShapes();
            r[OR] = m => m.WithBroadcastTestShapes();
            r[POW] = m => m.WithBroadcastTestShapes();

            r[NON_MAX_SUPPRESSION] = m => m
                .AttributeTestValues(AttrCenterPointBox, [0L])
                .InputTestValues("boxes", [TensorData([1, 3, 4], 0f, 0f, 1f, 1f, 0.1f, 0.1f, 0.8f, 0.8f, 0.2f, 0.2f, 0.7f, 0.7f)])
                .InputTestValues("scores", [TensorData([1, 1, 3], 0.9f, 0.75f, 0.6f)])
                .InputTestValues("max_output_boxes_per_class", [TensorData([], 2L)])
                .InputTestValues("iou_threshold", [TensorData([], 0.5f)])
                .InputTestValues("score_threshold", [TensorData([], 0.3f)]);

            r[PAD] = m => m
                .InputTestShapes("data", [[1, 2, 3], [3], [3, 4, 1]])
                .InputTestValues("pads", [TensorData([2], 2L, 3L), TensorData([2], 0L, 5L), TensorData([4], 0L, 1L, 2L, 3L)])
                .InputTestValues("axes", [TensorData([1], 1L), TensorData([1], 0L), TensorData([2], 0L, 2L)]);
        }
    }
}
