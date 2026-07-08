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
    /// Binds an <see cref="RngConfig"/> to a graph by validating it against the graph's
    /// realized stream set and writing the config's randomness state as the graph's single
    /// <c>SHRK_RNG_KEY_VECTOR</c> carrier (see <see cref="RngConfig.BuildKeyVector"/>) — the
    /// <b>source of truth</b> every key derivation reads. Binding writes NO per-node state:
    /// the ONNX-prep lowering (<see cref="FastLowerRandomOps"/>) decodes the carrier and
    /// derives each feed's keys from its structural attributes (ModelId + realized ids) on the
    /// export clone, so re-binding a different config is replacing one node, the pre-export
    /// graph keeps its structure, and there is no second representation to drift out of sync.
    /// A graph with no carrier lowers its feeds to the ONNX random fallback.
    ///
    /// <para>Validation is fail-loud, per the concreteness contract: every id-bearing feed
    /// must carry realized stream ids (enumerated at concretization — see
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

                var realizedFlat = node.Attributes.GetIntsVal(ShrkAttrRngRealizedIds);
                if (realizedFlat is not { Length: > 0 })
                    throw new InvalidOperationException(
                        $"ApplyRngConfig: the runtime random feed at ModelId [{string.Join(", ", idVals)}] " +
                        "carries no realized stream ids. RNG streams are enumerated at " +
                        "concretization (ToConcreteArchitecture) — bind the config to a " +
                        "concrete architecture, concrete model, or training-rig step graph.");

                int idLen = idVals.Length;
                for (int r = 0; r < realizedFlat.Length / idLen; r++)
                    realizedPaths.Add(string.Join(",", realizedFlat[(r * idLen)..((r + 1) * idLen)]));
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
