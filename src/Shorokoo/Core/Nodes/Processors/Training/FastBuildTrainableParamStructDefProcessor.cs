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
            var names = new HashSet<string>(paramInfos.Length);
            foreach (var p in paramInfos)
            {
                // Field names are the struct's lookup keys, and TrainingRig pairs initial values
                // into dictionaries keyed by them, so a repeat would leave two parameters sharing
                // one value and one gradient while every count still agreed — the silent shape of
                // Shorokoo/Shorokoo#284. Concretization gives each parameter its own ModelId path
                // and hence its own name; say so loudly if that ever stops holding.
                if (!names.Add(p.Name))
                    throw new InvalidOperationException(
                        $"Parameter discovery produced two fields named '{p.Name}'. Each parameter "
                        + "must have its own name: the name is the struct's lookup key and how "
                        + "initial values and gradients are paired to it.");
                fields.Add(new TensorStructFieldDef(p.Name, p.Structure, p.Rank, p.DType));
            }
            return new TensorStructDef(fields, typeName);
        }
    }
}
