using Shorokoo.Graph;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Rng;
using System;
using System.Linq;
using System.Collections.Generic;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// Binds an <see cref="RngConfig"/> to a graph: validates it against the graph's realized
    /// stream set, writes the config's randomness state as the graph's single
    /// <c>SHRK_RNG_KEY_VECTOR</c> carrier (the recorded identity — see
    /// <see cref="RngConfig.BuildKeyVector"/>), and runs the RNG key initializers
    /// (<see cref="FastMaterializeRngKeys"/>): every feed site's <c>SHRK_RNG_KEY_PARAM</c> entity
    /// gets its key-table value materialized from the identity, exactly as trainable
    /// parameters get their values by running their initializers. Re-binding re-runs the key
    /// initializers only — parameter values are untouched — so a concrete model can be
    /// re-seeded or switched to another algorithm in place, bit-exactly.
    ///
    /// <para>Validation is fail-loud, per the concreteness contract: every id-bearing feed
    /// must be wired to the key entity created at concretization (see
    /// <c>ToConcreteArchitecture</c>), a loop-body feed without per-iteration identity is an
    /// error (a single key would repeat identical values every iteration), and a
    /// <see cref="RngCollection.Runtime"/> override that matches no realized stream throws
    /// instead of silently doing nothing. (<see cref="RngCollection.Params"/> overrides are
    /// validated where they are consumed: parameter initialization.)</para>
    /// </summary>
    internal static class FastBindRngConfig
    {
        public static void Process(FastComputationGraph graph, RngConfig rngConfig)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));
            if (rngConfig is null) throw new ArgumentNullException(nameof(rngConfig));

            string algorithmName = RngAlgorithms.NameOf(rngConfig.Algorithm);
            var realizedPaths = new HashSet<string>();

            int loopDepth = 0;
            foreach (var node in graph.Nodes)
            {
                if (node.OpCode == OpCodes.LOOP_OPEN) loopDepth++;
                else if (node.OpCode == OpCodes.LOOP_CLOSE) loopDepth--;

                if (node.OpCode == InternalOpCodes.SHRK_RNG_KEY_PARAM)
                {
                    // The site's key entity owns the realized stream set.
                    var siteId = node.Attributes.GetIntsVal(ShrkAttrLocalModelId);
                    var realizedFlat = node.Attributes.GetIntsVal(ShrkAttrRngRealizedIds);
                    if (siteId is { Length: > 0 } && realizedFlat is { Length: > 0 })
                    {
                        int idLen = siteId.Length;
                        for (int r = 0; r < realizedFlat.Length / idLen; r++)
                            realizedPaths.Add(string.Join(",", realizedFlat[(r * idLen)..((r + 1) * idLen)]));
                    }
                    continue;
                }

                bool isUniform = node.OpCode == InternalOpCodes.SHRK_RANDOM_UNIFORM;
                bool isNormal = node.OpCode == InternalOpCodes.SHRK_RANDOM_NORMAL;
                if (!isUniform && !isNormal)
                    continue;

                var idVals = node.Attributes.GetIntsVal(ShrkAttrLocalModelId);
                if (idVals is null || idVals.Length == 0)
                {
                    // Not id-bearing (e.g. built by a path that bypasses id assignment): it has
                    // no stream identity, so it stays on the ONNX fallback at lowering.
                    continue;
                }

                if (loopDepth > 0 && Array.IndexOf(idVals, -1) < 0)
                {
                    // A loop-body feed without an iteration slot has no per-iteration stream
                    // identity: a deterministic key would repeat the same values every
                    // iteration. Under a bound config that is an error, not a silent
                    // nondeterministic fallback.
                    throw new InvalidOperationException(
                        $"ApplyRngConfig: the runtime random feed at ModelId [{string.Join(", ", idVals)}] " +
                        "sits inside a loop but carries no iteration slot (no per-iteration " +
                        "stream identity), so it cannot draw deterministically per iteration.");
                }

                if (node.Inputs.Count < 4 || node.Inputs[3] is null)
                    throw new InvalidOperationException(
                        $"ApplyRngConfig: the runtime random feed at ModelId [{string.Join(", ", idVals)}] " +
                        "carries no realized stream ids (no key entity is wired). RNG streams " +
                        "are enumerated at concretization (ToConcreteArchitecture) — bind the " +
                        "config to a concrete architecture, concrete model, or training-rig " +
                        "step graph.");
            }

            // Fail-loud override validation: a Runtime override that matches no stream of this
            // graph would otherwise be a silent no-op — exactly the re-keying hazard explicit
            // seeding exists to prevent.
            var unmatched = rngConfig.OverrideKeys
                .Where(k => k.collection == RngCollection.Runtime && !realizedPaths.Contains(k.pathKey))
                .Select(k => $"[{k.pathKey}]")
                .ToArray();
            if (unmatched.Length > 0)
                throw new InvalidOperationException(
                    "RngConfig.Override(Runtime, ...) matches no runtime stream of this graph: " +
                    string.Join(", ", unmatched) +
                    ". Available stream paths are listed by GetRngStreamReport(); overrides must " +
                    "use a reported path exactly.");

            InjectKeyVector(graph, rngConfig, algorithmName);
            FastMaterializeRngKeys.Process(graph, rngConfig);
        }

        /// <summary>
        /// Injects (or replaces) the model's compact RNG key vector — a single parameter-like
        /// int64 tensor carrying the config's randomness state (see
        /// <see cref="RngConfig.BuildKeyVector"/>; override records are path-keyed and
        /// self-describing, so no stream enumeration is stored). Lowered to a plain CONSTANT
        /// at ONNX prep; re-binding with a different config replaces it.
        /// </summary>
        private static void InjectKeyVector(
            FastComputationGraph graph, RngConfig rngConfig, string algorithmName)
        {
            var vector = rngConfig.BuildKeyVector();
            var data = new OnnxTensorData<int64>(
                new Shape(vector.Length),
                Core.Utils.OnnxUtils.CreateTensorValue(new Shape(vector.Length), vector));

            graph.Nodes.RemoveAll(n => n.OpCode == InternalOpCodes.SHRK_RNG_KEY_VECTOR);
            var nodeKey = FastNodeKey.New();
            var attrDefs = Definitions.NodeDefinitions[InternalOpCodes.SHRK_RNG_KEY_VECTOR].AttributeDefs;
            graph.Nodes.Add(new FastNode
            {
                Key = nodeKey,
                OpCode = InternalOpCodes.SHRK_RNG_KEY_VECTOR,
                Attributes = OnnxCSharpAttributes.FromCSharpVals(
                    new Dictionary<string, object?>
                    {
                        [AttrValue] = data,
                        [ShrkAttrRngAlgorithm] = algorithmName,
                    }, attrDefs),
                FullInputs = new Dictionary<string, List<FastTensorKey?>>(),
                FullOutputs = { [""] = new List<FastTensorKey?> { new FastTensorKey(nodeKey, 0) } },
            });
        }
    }
}
