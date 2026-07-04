using System.Collections.Generic;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;
using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;
using static Shorokoo.Globals;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Modules;

namespace Shorokoo.Core.Nodes.NodeDefinitions
{
    // Out-of-line (runtime-only) test data for the S op group; see Definitions.MQ.Tests.cs.
    internal static partial class Definitions
    {
        static partial void RegisterSTestData(System.Collections.Generic.Dictionary<string, System.Action<NodeDefinitionMaker>> r)
        {
            r[SCATTER_ELEMENTS] = m => m
                .InputTestShapes("data", [[3, 3]])
                .InputTestValues("indices", [TensorData([2, 3], 1L, 0L, 2L, 0L, 2L, 1L)])
                .InputTestShapes("updates", [[2, 3]]);
            r[SCATTER_ND] = m => m
                .InputTestShapes("data", [[5]])
                .InputTestValues("indices", [TensorData([2,1], 1L, 3L)])
                .InputTestShapes("updates", [[2]]);
            r[SEQUENCE_AT] = m => m
                .VariadicInputTestShapes([[[1,2,3],[2,3]], [[1],[]]])
                .InputTestValues("position", [TensorData([], 1L), TensorData([], 0L)]);
            r[SEQUENCE_CONSTRUCT] = m => m
                .AttributeTestValues(ShrkAttrFunctionName, [null, null])
                .AttributeTestValues(ShrkAttrDomainName, [null, null])
                .VariadicInputTestShapes([[[1,2,3],[2,3]], [[1],[]]]);
            r[SEQUENCE_EMPTY] = m => m
                .AttributeTestValues(ShrkAttrFunctionName, [null, null])
                .AttributeTestValues(ShrkAttrDomainName, [null, null]);
            r[SEQUENCE_ERASE] = m => m
                .VariadicInputTestShapes([[[1,2,3],[2,3]], [[1],[]]])
                .InputTestValues("position", [TensorData([], 1L), TensorData([], 0L)]);
            r[SEQUENCE_INSERT] = m => m
                .VariadicInputTestShapes([[[1,2,3],[2,3]], [[1],[]]])
                .InputTestValues("position", [TensorData([], 1L), TensorData([], 0L)]);
            r[SEQUENCE_LENGTH] = m => m
                .VariadicInputTestShapes([[[1,2,3],[2,3]], [[1],[]]]);
            r[SLICE] = m => m
                .InputTestShapes("x", [[2,8,3],[1,3,4,5,2]])
                .InputTestValues("starts", [TensorData([2], 1L,2L), TensorData([3], 1L, 2L, 2L)])
                .InputTestValues("ends", [TensorData([2], 8L,3L), TensorData([3], 3L, 3L, 3L)])
                .InputTestValues("axes", [TensorData([2], 1L,2L), TensorData([3], 1L,2L,3L)])
                .InputTestValues("steps", [TensorData([2], 2L,1L), TensorData([3], 1L,1L,1L)]);
            r[SOFTMAX] = m => m
                .InputTestShapes("input", [[2,3,4],[1,2,2,2,2],[3]])
                .AttributeTestValues(AttrAxis, [2L,3L,null]);
            r[SPLIT] = m => m
                .AttributeTestValues(AttrAxis, [2L, 1L])
                .AttributeTestValues(AttrNumOutputs, [null, 3L])
                .InputTestShapes("input", [[2,1,3],[1,6,2]])
                .InputTestValues("split", [TensorData([2], 1L,2L), null])
                .VariadicTestCounts("V", [2, 3]);
            r[SQUEEZE] = m => m
                .InputTestShapes("data", [[2,1,3,1,4,1],[1,2,2,2,1,1,1]])
                .InputTestValues("axes", [TensorData([2], 1L,3L), TensorData([3], 6L, 5L, 4L)]);
            r[SUB] = m => m
                .WithBroadcastTestShapes();
            r[SUM] = m => m
                .VariadicInputTestShapes([[[3,1,2],[1,2,2]],[[2,3],[2,1,3],[2,2,3]]]);
            r[SWISH] = m => m
                .InputTestShapes("X", [[5], [2, 3]])
                .AttributeTestValues(AttrAlpha, [1.0f, 2.0f]);
            r[SPACE_TO_DEPTH] = m => m
                .InputTestShapes("input", [[1, 1, 4, 4]])
                .AttributeTestValues(AttrBlocksize, [2L]);
        }
    }
}
