using Shorokoo.Graph;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Rng;
using System;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// Binds an <see cref="RngConfig"/> to a graph's runtime random feeds by STAMPING, never
    /// rewriting: every top-level <c>SHRK_RANDOM_UNIFORM/NORMAL</c> site (id-bearing since
    /// creation, its ModelId made absolute by module inlining) gets a
    /// <c>shrk_rng_explicit_key</c> attribute carrying its resolved stream key — the runtime
    /// master folded host-side along the feed's ModelId path (the RNG key tree IS the ModelId
    /// tree) — plus the config's algorithm name. Nothing else changes: no opcode, no inputs,
    /// no new nodes, so the pass is idempotent and re-running it with a different config
    /// re-binds the whole graph's randomness in place (works on a concrete architecture, a
    /// concrete model, or a training-rig step graph alike — the shared stamp point is
    /// concretization). Only the ONNX-prep lowering (<see cref="FastLowerRandomOps"/>) reads
    /// the stamp and rewrites the feed to the keyed deterministic draw; an unstamped feed
    /// falls back to the ONNX random ops.
    ///
    /// <para>Feeds inside loops are left unstamped (ONNX fallback): a loop body's node
    /// executes once per iteration with the same key and drawBase, which under the keyed path
    /// would repeat the same values every iteration; the fallback preserves
    /// fresh-per-iteration draws until per-iteration key splitting is plumbed.</para>
    /// </summary>
    internal static class FastApplyRngKeys
    {
        public static void Process(FastComputationGraph graph, RngConfig rngConfig)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));
            if (rngConfig is null) throw new ArgumentNullException(nameof(rngConfig));

            string algorithmName = rngConfig.Algorithm switch
            {
                RngAlgorithm.Threefry2x32 => RngAlgorithms.Default,
                _ => throw new ArgumentOutOfRangeException(
                    nameof(rngConfig), rngConfig.Algorithm, "Unknown RNG algorithm."),
            };

            int loopDepth = 0;
            foreach (var node in graph.Nodes)
            {
                if (node.OpCode == OpCodes.LOOP_OPEN) loopDepth++;
                else if (node.OpCode == OpCodes.LOOP_CLOSE) loopDepth--;

                bool isUniform = node.OpCode == InternalOpCodes.SHRK_RANDOM_UNIFORM;
                bool isNormal = node.OpCode == InternalOpCodes.SHRK_RANDOM_NORMAL;
                if ((!isUniform && !isNormal) || loopDepth > 0)
                    continue;

                var idVals = node.Attributes.GetIntsVal(ShrkAttrLocalModelId);
                if (idVals is null || idVals.Length == 0)
                {
                    // Not id-bearing (e.g. built by a path that bypasses id assignment):
                    // leave unstamped for the ONNX fallback lowering.
                    continue;
                }

                var (k0, k1) = rngConfig.FoldRunKey(idVals);
                node.Attributes = node.Attributes.SetAttributes(
                    (ShrkAttrRngExplicitKey, (long[])[k0, k1]),
                    (ShrkAttrRngAlgorithm, algorithmName));
            }
        }
    }
}
