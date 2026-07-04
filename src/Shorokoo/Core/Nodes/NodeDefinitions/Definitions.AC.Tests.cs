using System.Collections.Generic;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;
using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;
using static Shorokoo.Globals;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Modules;

namespace Shorokoo.Core.Nodes.NodeDefinitions
{
    // Out-of-line (runtime-only) test data for the AC op group; see Definitions.MQ.Tests.cs.
    internal static partial class Definitions
    {
        static partial void RegisterACTestData(System.Collections.Generic.Dictionary<string, System.Action<NodeDefinitionMaker>> r)
        {
            r[ADD] = m => m
                .WithBroadcastTestShapes();
            r[AFFINE_GRID] = m => m
                .InputTestShapes("theta", [[2, 2, 3], [1, 3, 4]])
                .InputTestValues("size", [TensorData(4, 2L, 3L, 4L, 2L), TensorData(5, 1L, 2L, 3L, 4L, 3L)]);
            r[ARG_MAX] = m => m
                .AttributeTestValues(AttrAxis, [0L, 1L, 2L])
                .InputTestShapes("data", [[5],[3,4],[2,3,4]])
                .AttributeTestValues(AttrAxis, [0L, 1L, 2L])
                .InputTestShapes("data", [[5],[3,4],[2,3,4]]);
            r[ARG_MIN] = m => m
                .AttributeTestValues(AttrAxis, [0L, 1L, 2L])
                .InputTestShapes("data", [[5],[3,4],[2,3,4]])
                .AttributeTestValues(AttrAxis, [0L, 1L, 2L])
                .InputTestShapes("data", [[5],[3,4],[2,3,4]]);
            r[ATTENTION] = m => m
                .AttributeTestValues(InternalAttrHasOptionalOutputs, [1L])
                .InputTestShapes("Q", [[1, 2, 3, 4]])
                .InputTestShapes("K", [[1, 2, 5, 4]])
                .InputTestShapes("V", [[1, 2, 5, 4]])
                .InputTestShapes("attn_mask", [[3, 7]])
                .InputTestShapes("past_key", [[1, 2, 2, 4]])
                .InputTestShapes("past_value", [[1, 2, 2, 4]])
                .AttributeTestValues(InternalAttrHasOptionalOutputs, [0L])
                .InputTestShapes("Q", [[1, 2, 3, 4]])
                .InputTestShapes("K", [[1, 2, 5, 4]])
                .InputTestShapes("V", [[1, 2, 5, 4]])
                .InputTestShapes("attn_mask", [[3, 5]])
                .InputTestValues("nonpad_kv_seqlen", [TensorData([1], 4L)]);
            r[AVERAGE_POOL] = m => m
                .AttributeTestValues(AttrDilations, (long[][])[[1L,1L],[2L,2L]])
                .AttributeTestValues(AttrKernelShape, (long[][])[[2L,2L],[3L,3L]])
                .AttributeTestValues(AttrPads, (long[][])[[0L,0L,0L,0L],[1L,1L,1L,1L]])
                .AttributeTestValues(AttrStrides, (long[][])[[2L,2L],[2L,2L]])
                .InputTestShapes("x", [[1L,2L,5L,3L],[1L,3L,2L,5L]])
                .AttributeTestValues(AttrDilations, (long[][])[[1L,1L],[2L,2L]])
                .AttributeTestValues(AttrKernelShape, (long[][])[[2L,2L],[3L,3L]])
                .AttributeTestValues(AttrPads, (long[]?[])[null, null])
                .AttributeTestValues(AttrStrides, (long[][])[[2L,2L],[2L,2L]])
                .InputTestShapes("x", [[1L,2L,5L,3L],[1L,3L,2L,5L]]);
            r[BATCH_NORMALIZATION] = m => m
                .InputTestShapes("x", [[3L,2L],[3L,2L,4L]])
                .InputTestShapes("scale", [[2L],[2L]])
                .InputTestShapes("b", [[2L],[2L]])
                .InputTestShapes("inputMean", [[2L],[2L]])
                .InputTestShapes("inputVariance", [[2L],[2L]])
                .InputTestShapes("x", [[3L,2L],[3L,2L,4L]])
                .InputTestShapes("scale", [[2L],[2L]])
                .InputTestShapes("b", [[2L],[2L]])
                .InputTestShapes("inputMean", [[2L],[2L]])
                .InputTestShapes("inputVariance", [[2L],[2L]]);
            r[BIT_CAST] = m => m
                .InputTestShapes("input", [[2, 3]]);
            r[BIT_SHIFT] = m => m
                .WithBroadcastTestShapes()
                .WithBroadcastTestShapes();
            r[BITWISE_AND] = m => m
                .WithBroadcastTestShapes();
            r[BITWISE_OR] = m => m
                .WithBroadcastTestShapes();
            r[BITWISE_XOR] = m => m
                .WithBroadcastTestShapes();
            r[BLACKMAN_WINDOW] = m => m
                .InputTestValues("size", [TensorData([], 3L), TensorData([], 20L), TensorData([], 6L), TensorData([], 32L)])
                .InputTestValues("size", [TensorData([], 3L), TensorData([], 20L), TensorData([], 6L), TensorData([], 32L)]);
            r[CENTER_CROP_PAD] = m => m
                .AttributeTestValues(AttrAxes, (long[]?[])[null, [-1L, 1L], [0L, 2L]])
                .InputTestValues("input_data", [TensorData([2,3], 1f, 2f, 3f, 4f, 5f, 6f), TensorData([1, 2, 3], 1L, 2L, 3L, 4L, 5L, 6L), TensorData([2,1,3], 1d, 2d, 3d, 4d, 5d, 6d)])
                .InputTestValues("shape", [TensorData([2], 1L, 2L), TensorData([2], 2L, 2L), TensorData([2], 1L, 2L)]);
            r[COL2IM] = m => m
                .InputTestShapes("input", [[1, 2, 15]])
                .InputTestValues("image_shape", [TensorData([2], 3L, 5L)])
                .InputTestValues("block_shape", [TensorData([2], 1L, 1L)])
                .AttributeTestValues(AttrDilations, (long[][])[[1L,1L]])
                .AttributeTestValues(AttrPads, (long[][])[[0L, 0L, 0L, 0L]])
                .AttributeTestValues(AttrStrides, (long[][])[[1L,1L]]);
            r[CONSTANT] = m => m
                .AttributeTestValues(AttrValue, [TensorData([2,3], 1f, 2f, 3f, 4f, 5f, 6f)]);
            r[CONSTANT_OF_SHAPE] = m => m
                .InputTestValues("shape", [TensorData([3], 1L, 2L, 3L), TensorData([2], 2L, 3L), TensorData([1], 3L), TensorData([1], 3L), TensorData([0], (long[])[])])
                .InputTestValues("shape", [TensorData([1], 1L, 2L, 3L), TensorData([1], 2L, 3L), TensorData([1], 3L), TensorData([1], 3L), TensorData([0L], (long[])[])])
                .AttributeTestValues(AttrValue, [TensorData([1], 1f), TensorData([1], 2L), TensorData([1], -1.01d), TensorData([1], (uint)2), TensorData([1], true)]);
            r[CONV] = m => m
                .InputTestShapes("X", [[1L,1L,5L,5L],[1L,1L,5L,5L]])
                .InputTestShapes("W", [[1L,1L,2L,2L],[1L,1L,3L,3L]])
                .InputTestShapes("B", [[1L],[1L]])
                .AttributeTestValues(AttrAutoPad, ["NOTSET", "SAME_UPPER"])
                .AttributeTestValues(AttrDilations, (long[][])[[1L,1L], [1L,1L]])
                .AttributeTestValues(AttrGroup, [1L, 1L])
                .AttributeTestValues(AttrKernelShape, (long[][])[[2L,2L], [3L,3L]])
                .AttributeTestValues(AttrPads, (long[]?[])[[0L,0L,0L,0L], null])
                .AttributeTestValues(AttrStrides, (long[][])[[1L,1L], [1L,1L]]);
            r[CONV_INTEGER] = m => m
                .InputTestShapes("X", [[1L,1L,5L,5L],[1L,1L,5L,5L]])
                .InputTestShapes("W", [[1L,1L,2L,2L],[1L,1L,3L,3L]])
                .AttributeTestValues(AttrAutoPad, ["NOTSET", "SAME_UPPER"])
                .AttributeTestValues(AttrDilations, (long[][])[[1L,1L], [1L,1L]])
                .AttributeTestValues(AttrGroup, [1L, 1L])
                .AttributeTestValues(AttrKernelShape, (long[][])[[2L,2L], [3L,3L]])
                .AttributeTestValues(AttrPads, (long[]?[])[[0L,0L,0L,0L], null])
                .AttributeTestValues(AttrStrides, (long[][])[[1L,1L], [1L,1L]]);
            r[CONV_TRANSPOSE] = m => m
                .InputTestShapes("X", [[1L, 1L, 5L, 5L], [1L, 1L, 5L, 5L]])
                .InputTestShapes("W", [[1L, 1L, 2L, 2L], [1L, 1L, 3L, 3L]])
                .InputTestShapes("B", [[1L], [1L]])
                .AttributeTestValues(AttrAutoPad, ["NOTSET"])
                .AttributeTestValues(AttrDilations, (long[][])[[1L, 1L]])
                .AttributeTestValues(AttrGroup, [1L])
                .AttributeTestValues(AttrKernelShape, (long[][])[[3L, 3L]])
                .AttributeTestValues(AttrOutputPadding, (long[][])[[0,0,0,0]])
                .AttributeTestValues(AttrOutputShape, (long[][])[[11,11]])
                .AttributeTestValues(AttrPads, (long[]?[])[[0,0,0,0]])
                .AttributeTestValues(AttrStrides, (long[][])[[2L, 2L]]);
            r[CUM_PROD] = m => m
                .InputTestShapes("x", [[5], [3, 4], [2, 3, 4]])
                .InputTestValues("axis", [TensorData([], 0L), TensorData([], 1L), TensorData([], 2L)]);
            r[CUM_SUM] = m => m
                .InputTestShapes("x", [[5],[3,4],[2,3,4]])
                .InputTestValues("axis", [TensorData([], 0L), TensorData([], 1L), TensorData([], 2L)]);
        }
    }
}
