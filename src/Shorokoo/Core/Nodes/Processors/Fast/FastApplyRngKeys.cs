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
    /// <para>Feeds under loops carry the <c>-1</c> iteration placeholder in their ModelId
    /// (one per enclosing loop level, exactly like params). For those the stamp carries the
    /// key of the path PREFIX up to the first <c>-1</c>; the remaining slots — runtime
    /// iteration indices and the concrete slots after them — are realized in-graph at ONNX
    /// prep by SHRK_RNG_SPLIT folds on the feed's iteration-indices input (so every loop
    /// iteration draws from its own stream, and an unrolled copy of the same loop
    /// constant-folds to bit-identical keys). A feed inside a loop scope whose id has no
    /// <c>-1</c> (no iteration-indices plumbing — an older build path) stays unstamped: a
    /// single key would repeat identical values every iteration, so the ONNX fallback
    /// keeps draws fresh instead.</para>
    /// </summary>
    internal static class FastApplyRngKeys
    {
        public static void Process(FastComputationGraph graph, RngConfig rngConfig)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));
            if (rngConfig is null) throw new ArgumentNullException(nameof(rngConfig));

            string algorithmName = RngAlgorithms.NameOf(rngConfig.Algorithm);

            int loopDepth = 0;
            foreach (var node in graph.Nodes)
            {
                if (node.OpCode == OpCodes.LOOP_OPEN) loopDepth++;
                else if (node.OpCode == OpCodes.LOOP_CLOSE) loopDepth--;

                bool isUniform = node.OpCode == InternalOpCodes.SHRK_RANDOM_UNIFORM;
                bool isNormal = node.OpCode == InternalOpCodes.SHRK_RANDOM_NORMAL;
                if (!isUniform && !isNormal)
                    continue;

                var idVals = node.Attributes.GetIntsVal(ShrkAttrLocalModelId);
                if (idVals is null || idVals.Length == 0)
                {
                    // Not id-bearing (e.g. built by a path that bypasses id assignment):
                    // leave unstamped for the ONNX fallback lowering.
                    continue;
                }

                int firstIterationSlot = Array.IndexOf(idVals, -1);
                if (firstIterationSlot < 0 && loopDepth > 0)
                {
                    // In a loop scope but no iteration slot in the id: the feed has no
                    // iteration-indices plumbing, so a stamped key would repeat the same
                    // values every iteration. Leave it to the ONNX fallback.
                    continue;
                }

                // Fold up to the first iteration placeholder (the whole path when none):
                // the rest of the path is realized at ONNX prep by in-graph splits on the
                // runtime iteration index.
                var foldVals = firstIterationSlot < 0 ? idVals : idVals[..firstIterationSlot];
                var (k0, k1) = rngConfig.FoldRunKey(foldVals);
                node.Attributes = node.Attributes.SetAttributes(
                    (ShrkAttrRngExplicitKey, (long[])[k0, k1]),
                    (ShrkAttrRngAlgorithm, algorithmName));
            }
        }
    }
}
