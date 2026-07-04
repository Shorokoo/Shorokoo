using System.Collections.Generic;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;
using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;
using static Shorokoo.Globals;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Modules;

namespace Shorokoo.Core.Nodes.NodeDefinitions
{
    // Out-of-line (runtime-only) test data for the TZ op group; see Definitions.MQ.Tests.cs.
    internal static partial class Definitions
    {
        static partial void RegisterTZTestData(System.Collections.Generic.Dictionary<string, System.Action<NodeDefinitionMaker>> r)
        {
            r[TILE] = m => m
                .InputTestShapes("input", [[2,3,4],[1,3,2,2]])
                .InputTestValues("repeats", [TensorData([3], 2L, 3L, 2L), TensorData([4], 3L, 1L, 1L, 2L)]);
            r[TOPK] = m => m
                .InputTestShapes("X", [[2,3,4],[1,3,2,2,6]])
                .InputTestValues("K", [TensorData([1], 2L), TensorData([1], 3L)])
                .AttributeTestValues(AttrAxis, [2L, 4L]);
            r[TRANSPOSE] = m => m
                .InputTestShapes("data", [[2,3,4],[1,2,2,2,2]])
                .AttributeTestValues(AttrPerm, (long[]?[])[[1,2,0], [3,4,2,1,0]]);
            r[UNSQUEEZE] = m => m
                .InputTestShapes("data", [[2,3,4],[1,2,2,2]])
                .InputTestValues("axes", [TensorData([2], 1L,3L), TensorData([3], 6L, 5L, 4L)]);
            r[XOR] = m => m
                .WithBroadcastTestShapes();
            r[TRILU] = m => m
                .InputTestShapes("input", [[3, 3]]);
            r[UNIQUE] = m => m
                .InputTestShapes("X", [[6], [3, 4]])
                .AttributeTestValues(AttrSorted, [true, true]);
            r[TENSOR_SCATTER] = m => m
                .InputTestShapes("past_cache", [[2, 1, 4, 5]])
                .InputTestShapes("update", [[2, 1, 1, 5]])
                .InputTestValues("write_indices", [TensorData([2], 1L, 2L)]);
        }
    }
}
