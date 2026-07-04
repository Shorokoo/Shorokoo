using System.Collections.Generic;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;
using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Modules;

namespace Shorokoo.Core.Nodes.NodeDefinitions
{
    // Out-of-line (runtime-only) test data for the HL op group; see Definitions.MQ.Tests.cs.
    internal static partial class Definitions
    {
        static partial void RegisterHLTestData(System.Collections.Generic.Dictionary<string, System.Action<NodeDefinitionMaker>> r)
        {
            r[LESS] = m => m
                .WithBroadcastTestShapes();
            r[LESS_OR_EQUAL] = m => m
                .WithBroadcastTestShapes();
            r[INSTANCE_NORMALIZATION] = m => m
                .InputTestShapes("input", [[1, 3, 4, 5]])
                .InputTestShapes("scale", [[3]])
                .InputTestShapes("B", [[3]]);
            r[LP_POOL] = m => m
                .AttributeTestValues(AttrDilations, (long[][])[[1L,1L],[1L,1L]])
                .AttributeTestValues(AttrKernelShape, (long[][])[[2L,2L],[3L,3L]])
                .AttributeTestValues(AttrP, [2L, 2L])
                .AttributeTestValues(AttrPads, (long[][])[[0L,0L,0L,0L],[1L,1L,1L,1L]])
                .AttributeTestValues(AttrStrides, (long[][])[[2L,2L],[1L,1L]])
                .InputTestShapes("X", [[1L,1L,5L,5L],[1L,1L,6L,4L]])
                .AttributeTestValues(AttrDilations, (long[][])[[1L,1L],[1L,1L]])
                .AttributeTestValues(AttrKernelShape, (long[][])[[2L,2L],[3L,3L]])
                .AttributeTestValues(AttrP, [2L, 2L])
                .AttributeTestValues(AttrPads, (long[]?[])[null, null])
                .AttributeTestValues(AttrStrides, (long[][])[[2L,2L],[1L,1L]])
                .InputTestShapes("X", [[1L,1L,5L,5L],[1L,1L,6L,4L]]);
        }
    }
}
