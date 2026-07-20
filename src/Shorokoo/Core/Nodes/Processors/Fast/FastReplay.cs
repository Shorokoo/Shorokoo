using Shorokoo.Graph;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using System;
using System.Collections.Generic;
using Shorokoo.Core.Nodes.Processors.AutoGrad;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// Inlines a source <see cref="InternalComputationGraph"/> into a target graph by cloning the
    /// source's non-input nodes (with fresh keys), remapping every reference to a source-input
    /// key onto a caller-provided <see cref="FastTensorKey"/>, and appending the cloned nodes
    /// to <see cref="InternalComputationGraph.Nodes"/>.
    ///
    /// <para>
    /// Replaces the CG-side <c>ReplayLossGraph</c> in <c>TrainingGraphBuilder</c> and
    /// <c>ReplayGraph</c> in <c>TrainingRig</c>. The source graph is not modified; the target
    /// graph gains exactly one cloned, rewired body of the source per call.
    /// </para>
    /// </summary>
    internal static class FastReplay
    {
        /// <summary>
        /// Splices a clone of <paramref name="source"/> into <paramref name="target"/>:
        /// every node of <paramref name="source"/> except its input-producing nodes is cloned,
        /// re-keyed, and appended to <paramref name="target"/>.<see cref="InternalComputationGraph.Nodes"/>.
        /// References to <paramref name="source"/>'s input keys are rewritten to point at the
        /// matching entry of <paramref name="mappedInputs"/>.
        /// </summary>
        /// <param name="target">Graph to splice cloned nodes into. Mutated in place.</param>
        /// <param name="source">Graph whose body to clone. Not modified.</param>
        /// <param name="mappedInputs">
        /// One entry per <paramref name="source"/>.<see cref="InternalComputationGraph.Inputs"/>;
        /// each entry is the <see cref="FastTensorKey"/> in <paramref name="target"/> that the
        /// corresponding source input should resolve to.
        /// </param>
        /// <returns>
        /// The keys (in <paramref name="target"/>) that <paramref name="source"/>'s outputs
        /// resolve to after splicing — for outputs that are passthroughs of an input, these are
        /// the matching <paramref name="mappedInputs"/> entries; for computed outputs they are
        /// the rekeyed body-node output keys.
        /// </returns>
        public static FastTensorKey[] ReplayInto(
            InternalComputationGraph target,
            InternalComputationGraph source,
            FastTensorKey[] mappedInputs)
        {
            if (target is null) throw new ArgumentNullException(nameof(target));
            if (source is null) throw new ArgumentNullException(nameof(source));
            if (mappedInputs is null) throw new ArgumentNullException(nameof(mappedInputs));

            if (source.Inputs.Count != mappedInputs.Length)
                throw new ArgumentException(
                    $"Input count mismatch: source has {source.Inputs.Count} inputs, " +
                    $"but {mappedInputs.Length} mapped inputs were provided.",
                    nameof(mappedInputs));

            // 1. Deep-clone the source so we can rekey freely without touching it.
            var clone = source.Clone();
            FastProcessorHelper.RekeySubgraph(clone);

            // 2. Build remap from cloned-input keys onto caller-provided keys. The
            //    remap is also consulted when mapping the cloned outputs back onto
            //    target keys, so passthrough outputs (source.Outputs[i] == source.Inputs[j])
            //    surface as the matching mappedInputs[j].
            var remap = new Dictionary<FastTensorKey, FastTensorKey>(clone.Inputs.Count);
            for (int i = 0; i < clone.Inputs.Count; i++)
                remap[clone.Inputs[i]] = mappedInputs[i];

            // 3. Identify input-producing nodes; these are dropped during splicing because
            //    the caller's mappedInputs already supply those values from target's side.
            var inputNodeKeys = new HashSet<FastNodeKey>();
            foreach (var ik in clone.Inputs)
                if (!ik.IsEmpty) inputNodeKeys.Add(ik.FastNodeKey);

            // 4. Splice body nodes into target, rewriting references to source inputs.
            foreach (var node in clone.Nodes)
            {
                if (inputNodeKeys.Contains(node.Key))
                    continue;

                foreach (var (_, slots) in node.FullInputs)
                {
                    for (int j = 0; j < slots.Count; j++)
                    {
                        if (slots[j] is FastTensorKey k && remap.TryGetValue(k, out var mapped))
                            slots[j] = mapped;
                    }
                }

                target.Nodes.Add(node);
            }

            // 5. Map cloned outputs through the remap.
            var outputs = new FastTensorKey[clone.Outputs.Count];
            for (int i = 0; i < clone.Outputs.Count; i++)
            {
                var k = clone.Outputs[i];
                outputs[i] = remap.TryGetValue(k, out var mapped) ? mapped : k;
            }
            return outputs;
        }
    }
}
