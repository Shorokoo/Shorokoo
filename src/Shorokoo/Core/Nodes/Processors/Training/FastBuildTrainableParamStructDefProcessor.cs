using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.OnnxNodes;
using System.Collections.Generic;
using System.Collections.Immutable;

namespace Shorokoo.Core.Nodes.Processors.Training
{
    /// <summary>
    /// Builds a <see cref="TensorStructDef"/> from a sequence of
    /// <see cref="FastDiscoveredParamInfo"/> records. Each parameter becomes a field whose
    /// dtype, rank, and structure are taken straight from the captured Fast-side info — no
    /// <see cref="Variable"/> reflection (and therefore no CG round-trip) is needed.
    /// </summary>
    internal static class FastBuildTrainableParamStructDefProcessor
    {
        public static TensorStructDef Process(ImmutableArray<FastDiscoveredParamInfo> paramInfos, string? typeName = null)
        {
            var fields = new List<TensorStructFieldDef>(paramInfos.Length);
            foreach (var p in paramInfos)
                fields.Add(new TensorStructFieldDef(p.Name, p.Structure, p.Rank, p.DType));
            return new TensorStructDef(fields, typeName);
        }
    }
}
