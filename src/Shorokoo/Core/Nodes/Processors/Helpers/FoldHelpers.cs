using Shorokoo.Graph;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.OnnxNodes;

namespace Shorokoo.Core.Nodes.Processors.Helpers
{
    /// <summary>
    /// Shared helper methods for fold processors.
    /// </summary>
    internal static class FoldHelpers
    {
        public static int[] MaxModelIdCounts(IEnumerable<ModelId> modelIds)
        {
            var maxLength = modelIds.Max(m => m.ToLongVals().Length);
            var maxIds = Enumerable.Range(0, maxLength).Select(index => modelIds.Max(x => index >= x.Vals.Length ? 0 : x.Vals[index]) + 1).ToArray();

            return maxIds;
        }

        public static long[] IndexToFlattenedIndexTransform(int[] tensorDims)
        {
            int r = tensorDims.Length;
            long[] transform = new long[r];

            long stride = 1;
            for (int i = r - 1; i >= 0; i--)
            {
                transform[i] = stride;
                stride *= tensorDims[i];
            }

            return transform;
        }

        public static long TransformModelIdToFlattenedIndex(ModelId modelId, long[] transformArray)
        {
            var modelIdVals = modelId.ToLongVals().Concat(Enumerable.Repeat(0L, transformArray.Length)).Take(transformArray.Length).ToArray();

            var retval = 0L;
            for (var i = 0; i < modelIdVals.Length; i++)
                retval += modelIdVals[i] * transformArray[i];

            return retval;
        }
    }
}
