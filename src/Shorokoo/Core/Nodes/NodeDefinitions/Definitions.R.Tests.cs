using System.Collections.Generic;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;
using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;
using static Shorokoo.Globals;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Modules;

namespace Shorokoo.Core.Nodes.NodeDefinitions
{
    // Out-of-line (runtime-only) test data for the R op group; see Definitions.MQ.Tests.cs.
    internal static partial class Definitions
    {
        static partial void RegisterRTestData(System.Collections.Generic.Dictionary<string, System.Action<NodeDefinitionMaker>> r)
        {
            r[RANGE] = m => m
                .InputTestValues("start", [TensorData([], 1L), TensorData([], 2f), TensorData([], 3d)])
                .InputTestValues("limit", [TensorData([], 12L), TensorData([], 34f), TensorData([], 3954d)])
                .InputTestValues("delta", [TensorData([], 2L), TensorData([], 3.6f), TensorData([], 203d)]);
            r[REDUCE_L1] = m => m
                .InputTestShapes("data", [[3,1,2],[1,2,2]])
                .InputTestValues("axes", [TensorData([1], 1L), TensorData([1], 2L)])
                .InputTestShapes("data", [[3,1,2],[1,2,2]])
                .InputTestValues("axes", [TensorData([1], 1L), TensorData([1], 2L)]);
            r[REDUCE_L2] = m => m
                .InputTestShapes("data", [[3,1,2],[1,2,2]])
                .InputTestValues("axes", [TensorData([1], 1L), TensorData([1], 2L)])
                .InputTestShapes("data", [[3,1,2],[1,2,2]])
                .InputTestValues("axes", [TensorData([1], 1L), TensorData([1], 2L)]);
            r[REDUCE_LOG_SUM] = m => m
                .InputTestShapes("data", [[3,1,2],[1,2,2]])
                .InputTestValues("axes", [TensorData([1], 1L), TensorData([1], 2L)])
                .InputTestShapes("data", [[3,1,2],[1,2,2]])
                .InputTestValues("axes", [TensorData([1], 1L), TensorData([1], 2L)]);
            r[REDUCE_LOG_SUM_EXP] = m => m
                .InputTestShapes("data", [[3,1,2],[1,2,2]])
                .InputTestValues("axes", [TensorData([1], 1L), TensorData([1], 2L)])
                .InputTestShapes("data", [[3,1,2],[1,2,2]])
                .InputTestValues("axes", [TensorData([1], 1L), TensorData([1], 2L)]);
            r[REDUCE_MAX] = m => m
                .InputTestShapes("data", [[3,1,2],[1,2,2]])
                .InputTestValues("axes", [TensorData([1], 1L), TensorData([1], 2L)])
                .InputTestShapes("data", [[3,1,2],[1,2,2]])
                .InputTestValues("axes", [TensorData([1], 1L), TensorData([1], 2L)]);
            r[REDUCE_MEAN] = m => m
                .InputTestShapes("data", [[3,1,2],[1,2,2]])
                .InputTestValues("axes", [TensorData([1], 1L), TensorData([1], 2L)])
                .InputTestShapes("data", [[3,1,2],[1,2,2]])
                .InputTestValues("axes", [TensorData([1], 1L), TensorData([1], 2L)]);
            r[REDUCE_MIN] = m => m
                .InputTestShapes("data", [[3,1,2],[1,2,2]])
                .InputTestValues("axes", [TensorData([1], 1L), TensorData([1], 2L)])
                .InputTestShapes("data", [[3,1,2],[1,2,2]])
                .InputTestValues("axes", [TensorData([1], 1L), TensorData([1], 2L)]);
            r[REDUCE_PROD] = m => m
                .InputTestShapes("data", [[3,1,2],[1,2,2]])
                .InputTestValues("axes", [TensorData([1], 1L), TensorData([1], 2L)])
                .InputTestShapes("data", [[3,1,2],[1,2,2]])
                .InputTestValues("axes", [TensorData([1], 1L), TensorData([1], 2L)]);
            r[REDUCE_SUM] = m => m
                .InputTestShapes("data", [[3,1,2],[1,2,2]])
                .InputTestValues("axes", [TensorData([1], 1L), TensorData([1], 2L)])
                .InputTestShapes("data", [[3,1,2],[1,2,2]])
                .InputTestValues("axes", [TensorData([1], 1L), TensorData([1], 2L)]);
            r[REDUCE_SUM_SQUARE] = m => m
                .InputTestShapes("data", [[3,1,2],[1,2,2]])
                .InputTestValues("axes", [TensorData([1], 1L), TensorData([1], 2L)])
                .InputTestShapes("data", [[3,1,2],[1,2,2]])
                .InputTestValues("axes", [TensorData([1], 1L), TensorData([1], 2L)]);
            r[RESHAPE] = m => m
                .InputTestShapes("input", [[2,3,4],[6,4]])
                .InputTestValues("shape", [TensorData([2], 12L, 2L), TensorData([3], 3L, 4L, 2L)]);
            r[REVERSE_SEQUENCE] = m => m
                .InputTestShapes("input", [[4, 4], [2, 4, 3]])
                .InputTestValues("sequence_lens", [TensorData([4], 1L, 2L, 3L, 4L), TensorData([2], 3L, 2L)])
                .AttributeTestValues(AttrBatchAxis, [0L, 0L])
                .AttributeTestValues(AttrTimeAxis, [1L, 1L]);
            r[RESIZE] = m => m
                .AttributeTestValues(AttrAntialias, [1L])
                .AttributeTestValues(AttrAxes, (long[]?[])[null])
                .AttributeTestValues(AttrCoordinateTransformationMode, ["half_pixel_symmetric"])
                .AttributeTestValues(AttrCubicCoeffA, [null])
                .AttributeTestValues(AttrExcludeOutside, [1L])
                .AttributeTestValues(AttrExtrapolationValue, [3f])
                .AttributeTestValues(AttrKeepAspectRatioPolicy, ["stretch"])
                .AttributeTestValues(AttrMode, ["linear"])
                .AttributeTestValues(AttrNearestMode, ["round_prefer_ceil"])
                .InputTestShapes("X", [[3,4,5]])
                .InputTestValues("roi", [null])
                .InputTestValues("scales", [TensorData([3], 1.4f, 0.5f, 1.2f)])
                .InputTestValues("sizes", [null]);
            r[ROI_ALIGN] = m => m
                .InputTestShapes("X", [[1L,1L,4L,4L]])
                .InputTestValues("rois", [TensorData([1, 4], 0f, 0f, 3f, 3f)])
                .InputTestValues("batch_indices", [TensorData([1], 0L)])
                .AttributeTestValues(AttrCoordinateTransformationMode, ["half_pixel"])
                .AttributeTestValues(AttrMode, ["avg"])
                .AttributeTestValues(AttrOutputHeight, [2L])
                .AttributeTestValues(AttrOutputWidth, [2L])
                .AttributeTestValues(AttrSamplingRatio, [2L])
                .AttributeTestValues(AttrSpatialScale, [1f]);
            r[RMS_NORMALIZATION] = m => m
                .InputTestShapes("X", [[2, 4], [2, 3, 4]])
                .InputTestShapes("scale", [[4], [4]])
                .AttributeTestValues(AttrAxis, [-1L, 2L])
                .AttributeTestValues(AttrEpsilon, [1e-5f, 1e-5f])
                .AttributeTestValues(AttrStashType, [1L, 1L]);
            r[ROTARY_EMBEDDING] = m => m
                .InputTestShapes("X", [[1, 2, 3, 4]])
                .InputTestShapes("cos_cache", [[8, 2]])
                .InputTestShapes("sin_cache", [[8, 2]])
                .InputTestValues("position_ids", [TensorData([1, 3], 0L, 1L, 2L)]);
        }
    }
}
