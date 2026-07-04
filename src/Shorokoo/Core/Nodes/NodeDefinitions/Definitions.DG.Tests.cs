using System.Collections.Generic;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;
using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;
using static Shorokoo.Globals;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Modules;

namespace Shorokoo.Core.Nodes.NodeDefinitions
{
    // Out-of-line (runtime-only) test data for the DG op group; see Definitions.MQ.Tests.cs.
    internal static partial class Definitions
    {
        static partial void RegisterDGTestData(System.Collections.Generic.Dictionary<string, System.Action<NodeDefinitionMaker>> r)
        {
            r[DFT] = m => m
                .InputTestShapes("input", [[2, 2, 2], [2, 2, 3, 1]])
                .InputTestValues("dft_length", [TensorData([], 1L), TensorData([], 4L)])
                .InputTestValues("axis", [TensorData([], 1L), TensorData([], 2L)])
                .AttributeTestValues(AttrInverse, [1L, 0L])
                .AttributeTestValues(AttrOnesided, [0L, 1L]);
            r[DEFORM_CONV] = m => m
                .InputTestShapes("X", [[1L,1L,5L,5L],[1L,1L,5L,5L]])
                .InputTestShapes("W", [[1L,1L,2L,2L],[1L,1L,3L,3L]])
                .InputTestShapes("offset", [[1L,8L,4L,4L],[1L,18L,5L,5L]])
                .InputTestShapes("B", [[1L],[1L]])
                .InputTestShapes("mask", [[1, 4, 4, 4],[1, 9, 5, 5]])
                .AttributeTestValues(AttrDilations, (long[][])[[1L,1L], [1L,1L]])
                .AttributeTestValues(AttrGroup, [1L, 1L])
                .AttributeTestValues(AttrKernelShape, (long[][])[[2L,2L], [3L,3L]])
                .AttributeTestValues(AttrOffsetGroup, [1L, 1L])
                .AttributeTestValues(AttrPads, (long[]?[])[[0L,0L,0L,0L], [0L,0L,0L,0L]])
                .AttributeTestValues(AttrStrides, (long[][])[[1L,1L], [1L,1L]]);
            r[DEPTH_TO_SPACE] = m => m
                .InputTestShapes("input", [[1, 16, 2, 2], [1, 18, 2, 2]])
                .AttributeTestValues(AttrBlocksize, [2L, 3L])
                .AttributeTestValues(AttrMode, ["DCR", "CRD"]);
            r[DEQUANTIZE_LINEAR] = m => m
                .InputTestShapes("x", [[2, 6]])
                .InputTestShapes("x_scale", [[6]])
                .InputTestShapes("x_zero_point", [[6]])
                .AttributeTestValues(AttrAxis, [1L])
                .AttributeTestValues(AttrBlockSize, [2L])
                .InputTestShapes("x", [[2, 3, 4]])
                .InputTestShapes("x_scale", [[3]])
                .InputTestShapes("x_zero_point", [[3]])
                .AttributeTestValues(AttrAxis, [1L])
                .AttributeTestValues(AttrBlockSize, [null])
                .InputTestShapes("x", [[2, 3, 4]])
                .InputTestShapes("x_scale", [ [3] ])
                .InputTestShapes("x_zero_point", [ [3] ])
                .AttributeTestValues(AttrAxis, [null])
                .AttributeTestValues(AttrBlockSize, [null]);
            r[DET] = m => m
                .InputTestShapes("X", [[2, 2, 2], [2, 1, 4, 4]]);
            r[DIV] = m => m
                .WithBroadcastTestShapes();
            r[EQUAL] = m => m
                .WithBroadcastTestShapes();
            r[EXPAND] = m => m
                .InputTestShapes("input", [[2, 3], [3, 1, 2]])
                .InputTestValues("shape", [TensorData([2], (long[])[2, 3]), TensorData([4], (long[])[2, 1, 3, 2])]);
            r[EYE_LIKE] = m => m
                .InputTestShapes("input", [[2, 2], [4, 4]])
                .InputTestShapes("input", [[2, 2], [4, 4]]);
            r[FLATTEN] = m => m
                .InputTestShapes("input", [[2, 3, 4], [2, 3, 4, 5]])
                .AttributeTestValues(AttrAxis, [1L, 2L]);
            r[GLOBAL_AVERAGE_POOL] = m => m
                .InputTestShapes("X", [[1, 3, 4, 5], [1, 3, 5, 5]]);
            r[GLOBAL_LP_POOL] = m => m
                .InputTestShapes("X", [[1, 3, 4, 5], [1, 3, 5, 5]])
                .AttributeTestValues(AttrP, [1L, 2L, 3L]);
            r[GLOBAL_MAX_POOL] = m => m
                .InputTestShapes("X", [[1, 3, 4, 5], [1, 3, 5, 5]]);
            r[GREATER] = m => m
                .WithBroadcastTestShapes();
            r[GREATER_OR_EQUAL] = m => m
                .WithBroadcastTestShapes();
            r[GRID_SAMPLE] = m => m
                .InputTestShapes("X", [[1, 3, 4, 5], [1, 3, 5, 5]])
                .InputTestShapes("grid", [[1, 4, 5, 2], [1, 5, 5, 2]]);
            r[GROUP_NORMALIZATION] = m => m
                .InputTestShapes("X", [[1, 4, 4, 5], [1, 3, 6, 5]])
                .InputTestShapes("scale", [[4], [6]])
                .InputTestShapes("bias", [[4], [6]])
                .AttributeTestValues(AttrEpsilon, [1e-5f, 1e-6f])
                .AttributeTestValues(AttrNumGroups, [2L, 3L])
                .AttributeTestValues(AttrStashType, [1L, 2L]);
            r[GEMM] = m => m
                .InputTestShapes("A", [[2, 3]])
                .InputTestShapes("B", [[3, 2]])
                .InputTestShapes("C", [[2, 2]]);
        }
    }
}
