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
                    // Not id-bearing (e.g. built by a path that bypasses id assignment):
                    // leave unstamped for the ONNX fallback lowering.
                    continue;
                }

                // Realized-streams path (concrete architectures): the QEE enumeration stamped
                // the feed's full stream ids at concretization. Every stream's key is resolved
                // host-side over its FULL realized path, so per-stream overrides — including a
                // single loop iteration's — take effect. When any of the feed's streams is
                // overridden, the resolved keys are stamped as a table the lowering selects
                // from by flattened iteration index; otherwise the legacy prefix stamp is kept
                // (the in-graph split derivation folds to bit-identical keys).
                var realizedFlat = node.Attributes.GetIntsVal(ShrkAttrRngRealizedIds);
                if (realizedFlat is { Length: > 0 })
                {
                    int idLen = idVals.Length;
                    int n = realizedFlat.Length / idLen;
                    bool anyOverride = false;
                    var table = new long[2 * n];
                    for (int r = 0; r < n; r++)
                    {
                        var path = realizedFlat[(r * idLen)..((r + 1) * idLen)];
                        realizedPaths.Add(string.Join(",", path));
                        if (rngConfig.HasOverride(RngCollection.Runtime, path)) anyOverride = true;
                        var (rk0, rk1) = rngConfig.FoldRunKey(path);
                        table[2 * r] = rk0;
                        table[2 * r + 1] = rk1;
                    }

                    if (anyOverride)
                    {
                        node.Attributes = node.Attributes.SetAttributes(
                            (ShrkAttrRngKeyTable, table),
                            (ShrkAttrRngExplicitKey, (long[])[table[0], table[1]]),
                            (ShrkAttrRngAlgorithm, algorithmName));
                    }
                    else
                    {
                        int prefixEnd = Array.IndexOf(idVals, -1);
                        var prefix = prefixEnd < 0 ? idVals : idVals[..prefixEnd];
                        var (pk0, pk1) = rngConfig.FoldRunKey(prefix);
                        node.Attributes = node.Attributes.SetAttributes(
                            (ShrkAttrRngKeyTable, (long[])[]),
                            (ShrkAttrRngExplicitKey, (long[])[pk0, pk1]),
                            (ShrkAttrRngAlgorithm, algorithmName));
                    }
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

                // Legacy (non-enumerated) path: fold up to the first iteration placeholder
                // (the whole path when none); the rest of the path is realized at ONNX prep
                // by in-graph splits on the runtime iteration index.
                var foldVals = firstIterationSlot < 0 ? idVals : idVals[..firstIterationSlot];
                var (k0, k1) = rngConfig.FoldRunKey(foldVals);
                realizedPaths.Add(string.Join(",", foldVals));
                node.Attributes = node.Attributes.SetAttributes(
                    (ShrkAttrRngExplicitKey, (long[])[k0, k1]),
                    (ShrkAttrRngAlgorithm, algorithmName));
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

            InjectKeyVector(graph, rngConfig, algorithmName, realizedPaths);
        }

        private static int ComparePaths(int[] a, int[] b)
        {
            for (int i = 0; i < Math.Min(a.Length, b.Length); i++)
                if (a[i] != b[i]) return a[i].CompareTo(b[i]);
            return a.Length.CompareTo(b.Length);
        }

        /// <summary>
        /// Injects (or replaces) the model's compact RNG key vector — a single parameter-like
        /// int64 tensor carrying the config's randomness state in the smallest of three tiers
        /// (see <see cref="RngConfig.BuildKeyVector"/>). The tier-3 expansion enumerates init
        /// streams (trainable-param ids) then runtime streams (realized feed paths), each
        /// sorted lexicographically — the same canonical order reconstruction uses. Lowered to
        /// a plain CONSTANT at ONNX prep; re-stamping with a different config replaces it.
        /// </summary>
        private static void InjectKeyVector(
            FastComputationGraph graph, RngConfig rngConfig, string algorithmName,
            HashSet<string> runtimePathKeys)
        {
            var initPaths = graph.Nodes
                .Where(n => n.OpCode == InternalOpCodes.TRAINABLE_PARAM)
                .Select(n => n.Attributes.GetIntsVal(ShrkAttrLocalModelId))
                .Where(v => v is { Length: > 0 })
                .Select(v => v!)
                .OrderBy(v => v, Comparer<int[]>.Create(ComparePaths))
                .ToList();
            var runPaths = runtimePathKeys
                .Select(k => k.Split(',').Select(int.Parse).ToArray())
                .OrderBy(v => v, Comparer<int[]>.Create(ComparePaths))
                .ToList();

            var vector = rngConfig.BuildKeyVector(initPaths, runPaths);
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
                    new System.Collections.Generic.Dictionary<string, object?>
                    {
                        [AttrValue] = data,
                        [ShrkAttrRngAlgorithm] = algorithmName,
                        [ShrkAttrRngInitStreamCount] = (long)initPaths.Count,
                    }, attrDefs),
                FullInputs = new System.Collections.Generic.Dictionary<string, System.Collections.Generic.List<FastTensorKey?>>(),
                FullOutputs = { [""] = new System.Collections.Generic.List<FastTensorKey?> { new FastTensorKey(nodeKey, 0) } },
            });
        }
    }
}
