using Shorokoo.Graph;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Rng;
using Shorokoo.Core.Utils;
using Shorokoo.Onnx;
using System;
using System.Collections.Generic;
using System.Linq;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// Wires every id-bearing runtime random feed's <b>in-graph key derivation chain</b>: a
    /// <c>SHRK_RNG_SPLIT</c> chain rooted at the model's <c>RngSeed</c> parameter (the ordinary
    /// non-trainable int64 parameter at reserved ModelId [0] holding the runtime RNG identity —
    /// see <see cref="RngRuntimeIdentity"/>), one split per ModelId path element. A static path
    /// element enters as a constant split counter; a loop-iteration slot (<c>-1</c>) enters as
    /// the feed's <b>runtime iteration index</b> (an element of its iteration-indices input) —
    /// so per-iteration streams need no enumeration, no key tables, and no stride arithmetic:
    /// iteration <c>i</c> derives <c>fold(..., i, ...)</c> on the fly, bit-identically to the
    /// host-side <see cref="RngConfig.FoldRunKey"/>.
    ///
    /// <para><b>Per-stream overrides route structurally.</b> An override record replaces the
    /// fully folded key of exactly one realized stream, so an overridden site's chain selects
    /// (via <c>Where</c> on the runtime iteration indices, or unconditionally when the site has
    /// no iteration slots) a <c>Gather</c> of the record's fixed key-word offset in the
    /// canonical identity vector instead of the folded chain. Changing override <em>values</em>
    /// is thereafter a parameter write; changing the override <em>set</em> re-runs this pass
    /// (the orphaned chain nodes are swept by ordinary reachability).</para>
    ///
    /// <para>Because the chain roots at a <c>MODEL_PARAM</c>/<c>MODEL_PARAM_DATA</c> node —
    /// which Shorokoo's constant folding refuses to fold — persisted graphs keep the symbolic
    /// chains, and re-binding a loaded model is a parameter write that changes every draw by
    /// construction. ORT's session-build constant folding collapses the constant segments to
    /// literal keys, recovering the pre-computation at the session layer.</para>
    /// </summary>
    internal static class FastWireRngKeyDerivation
    {
        /// <summary>The <c>RngSeed</c> parameter's name (its identifier-template param part).</summary>
        public const string RngSeedName = "RngSeed";

        /// <summary>The <c>RngSeed</c> parameter's identifier template at reserved ModelId [0].</summary>
        public static readonly string RngSeedIdentifierTemplate =
            ModelParamIdentifierTemplate.LocalTrainableParam(
                new ModelId(0), RngSeedName, 0, System.Collections.Immutable.ImmutableArray<int>.Empty).ToString();

        /// <summary>
        /// The graph's <c>RngSeed</c> node: a value-less <c>MODEL_PARAM</c> on a concrete
        /// architecture, a <c>MODEL_PARAM_DATA</c> holding the identity vector once bound.
        /// Null when the graph has no runtime random surface (or predates concretization).
        /// </summary>
        public static FastNode? FindRngSeedNode(InternalComputationGraph graph)
            => graph.Nodes.FirstOrDefault(n =>
                (n.OpCode == InternalOpCodes.MODEL_PARAM || n.OpCode == InternalOpCodes.MODEL_PARAM_DATA)
                && n.IdentifierTemplate == RngSeedIdentifierTemplate);

        /// <summary>The id-bearing runtime random feeds (the graph's runtime RNG sites).</summary>
        public static IEnumerable<FastNode> IdBearingFeeds(InternalComputationGraph graph)
            => graph.Nodes.Where(n =>
                (n.OpCode == InternalOpCodes.SHRK_RANDOM_UNIFORM ||
                 n.OpCode == InternalOpCodes.SHRK_RANDOM_NORMAL) &&
                n.Attributes.GetIntsVal(ShrkAttrLocalModelId) is { Length: > 0 });

        /// <summary>
        /// Creates the value-less <c>RngSeed</c> <c>MODEL_PARAM</c> at ModelId [0] and wires
        /// every feed's derivation chain (no overrides — the architecture-time wiring; a bind
        /// with overrides re-runs the wiring). Runs at concretization, and only when the graph
        /// has at least one id-bearing runtime feed: a model without random computation carries
        /// no <c>RngSeed</c>, no chains, and nothing RNG-related — persisted or otherwise.
        /// </summary>
        public static void CreateRngSeedAndWireChains(InternalComputationGraph graph)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));
            if (!IdBearingFeeds(graph).Any()) return;

            var attrDefs = Definitions.NodeDefinitions[InternalOpCodes.MODEL_PARAM].AttributeDefs;
            var nodeKey = FastNodeKey.New();
            var seedNode = new FastNode
            {
                Key = nodeKey,
                OpCode = InternalOpCodes.MODEL_PARAM,
                Attributes = OnnxCSharpAttributes.FromCSharpVals(
                    new Dictionary<string, object?>
                    {
                        [ShrkAttrLocalModelId] = (long[])[0L],
                        [ShrkAttrDtype] = DType.Int64,
                        [ShrkAttrRank] = 1L,
                        [ShrkAttrIsTrainable] = false,
                    }, attrDefs),
                IdentifierTemplate = RngSeedIdentifierTemplate,
                FriendlyName = RngSeedName,
                FullInputs = new Dictionary<string, List<FastTensorKey?>>(),
                FullOutputs = { [""] = new List<FastTensorKey?> { new FastTensorKey(nodeKey, 0) } },
            };
            graph.Nodes.Insert(0, seedNode);

            Process(graph, Array.Empty<(int[] path, int keyOffset)>(), validateStructure: true);
        }

        /// <summary>
        /// (Re-)wires every id-bearing feed's key chain against the current identity layout.
        /// <paramref name="overrideRecords"/> lists the runtime override records — realized
        /// stream path + the record's key-word offset in the identity vector (see
        /// <see cref="RngRuntimeIdentity"/>) — in canonical order. Previously wired chain nodes
        /// become unreachable and are swept by the caller. <paramref name="validateStructure"/>
        /// enables the concretization-time slot check (an iteration-slot count that the feed's
        /// iteration-indices input cannot fill is a hard build error, never a silent fallback);
        /// bind-time re-wiring skips it, since simplification may have reshaped the input's
        /// producer while leaving the vector itself intact.
        /// </summary>
        public static void Process(
            InternalComputationGraph graph,
            IReadOnlyList<(int[] path, int keyOffset)> overrideRecords,
            bool validateStructure = false)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));

            var seedNode = FindRngSeedNode(graph)
                ?? throw new InvalidOperationException(
                    "FastWireRngKeyDerivation: the graph has no RngSeed parameter at ModelId [0] " +
                    "to root the key derivation chains at.");
            var seedOut = seedNode.Outputs[0]!.Value;

            var nodeByKey = graph.Nodes.ToDictionary(n => n.Key);
            var newNodes = new List<FastNode>(graph.Nodes.Count);

            foreach (var node in graph.Nodes)
            {
                bool isFeed = node.OpCode == InternalOpCodes.SHRK_RANDOM_UNIFORM ||
                              node.OpCode == InternalOpCodes.SHRK_RANDOM_NORMAL;
                var idVals = isFeed ? node.Attributes.GetIntsVal(ShrkAttrLocalModelId) : null;
                if (idVals is not { Length: > 0 })
                {
                    newNodes.Add(node);
                    continue;
                }

                int depth = idVals.Count(v => v == -1);
                var iterIndices = node.Inputs.Count > 2 ? node.Inputs[2] : null;

                if (depth > 0 && iterIndices is null)
                    throw new InvalidOperationException(
                        $"FastWireRngKeyDerivation: the runtime random feed at ModelId " +
                        $"[{string.Join(", ", idVals)}] has an iteration slot but carries no " +
                        "iteration-indices input to derive its per-iteration stream key from.");

                if (validateStructure && depth > 0)
                {
                    // The feed's iteration-indices input supplies one element per enclosing
                    // loop; a site id with more -1 slots than that has no per-iteration stream
                    // identity — a corrupted inventory, not a zero-trip loop. Hard error, per
                    // the concreteness contract.
                    int available = CountIterationIndexElements(iterIndices!.Value, nodeByKey);
                    if (available != depth)
                        throw new FastPipelineUnsupportedException(
                            "FastWireRngKeyDerivation: the runtime random feed at ModelId " +
                            $"[{string.Join(", ", idVals)}] has {depth} iteration slot(s) but its " +
                            $"iteration-indices input supplies {available} element(s), so its " +
                            "per-iteration stream keys cannot be derived. A concrete architecture " +
                            "requires a derivable stream set — never a silent fallback.");
                }

                var keyKey = BuildChain(idVals, iterIndices, seedOut, overrideRecords, newNodes);

                var feedInputs = node.FullInputs[""];
                while (feedInputs.Count < 4) feedInputs.Add(null);
                feedInputs[3] = keyKey;
                newNodes.Add(node);
            }

            graph.Nodes = newNodes;
        }

        /// <summary>Counts the elements of a feed's iteration-indices input by walking its
        /// producer (the per-level CONCAT built at trace time; a CONSTANT means none).</summary>
        private static int CountIterationIndexElements(
            FastTensorKey iterIndices, Dictionary<FastNodeKey, FastNode> nodeByKey)
        {
            if (!nodeByKey.TryGetValue(iterIndices.FastNodeKey, out var producer)) return 0;
            if (producer.OpCode == OpCodes.CONCAT) return producer.Inputs.Count;
            if (producer.OpCode == OpCodes.CONSTANT)
            {
                var data = producer.Attributes.GetTensorVal(AttrValue);
                return data is null ? 0 : (int)data.Shape.Count;
            }
            return 0;
        }

        /// <summary>
        /// One feed's chain: root = the identity's master key words; one SHRK_RNG_SPLIT per
        /// path element (constant counter for a static slot, the runtime iteration index for a
        /// -1 slot); then, for each override record whose path matches the site, a selector
        /// that roots the key at the record's key words instead.
        /// </summary>
        private static FastTensorKey BuildChain(
            int[] idVals, FastTensorKey? iterIndices, FastTensorKey seedOut,
            IReadOnlyList<(int[] path, int keyOffset)> overrideRecords, List<FastNode> newNodes)
        {
            var key = AppendGatherPair(seedOut, RngRuntimeIdentity.RunKeyIndex, newNodes);

            int iterSlot = 0;
            var iterElementKeys = new Dictionary<int, FastTensorKey>();   // iter position -> scalar index
            for (int i = 0; i < idVals.Length; i++)
            {
                FastTensorKey counter;
                if (idVals[i] == -1)
                {
                    counter = AppendGatherScalar(iterIndices!.Value, iterSlot, newNodes);
                    iterElementKeys[iterSlot] = counter;
                    iterSlot++;
                }
                else
                {
                    counter = AppendScalarInt64(idVals[i], newNodes);
                }
                key = AppendSplit(key, counter, newNodes);
            }

            foreach (var (path, keyOffset) in overrideRecords)
            {
                if (!PathMatchesSite(path, idVals)) continue;
                var overrideKey = AppendGatherPair(seedOut, keyOffset, newNodes);
                if (iterElementKeys.Count == 0)
                {
                    // Static site: the override replaces the fully folded key unconditionally.
                    key = overrideKey;
                    continue;
                }
                // In-loop site: the override applies to exactly one realized iteration —
                // select at runtime on the iteration indices.
                FastTensorKey? cond = null;
                int slot = 0;
                for (int i = 0; i < idVals.Length; i++)
                {
                    if (idVals[i] != -1) continue;
                    var eq = AppendBinaryOp(OpCodes.EQUAL,
                        iterElementKeys[slot], AppendScalarInt64(path[i], newNodes), newNodes);
                    cond = cond is null ? eq : AppendBinaryOp(OpCodes.AND, cond.Value, eq, newNodes);
                    slot++;
                }
                key = AppendWhere(cond!.Value, overrideKey, key, newNodes);
            }

            return key;
        }

        /// <summary>Whether an override's realized path addresses this site: same length, equal
        /// static slots; a -1 (iteration) slot matches any non-negative realized value.</summary>
        internal static bool PathMatchesSite(int[] path, int[] siteVals)
        {
            if (path.Length != siteVals.Length) return false;
            for (int i = 0; i < path.Length; i++)
            {
                if (siteVals[i] == -1) { if (path[i] < 0) return false; }
                else if (path[i] != siteVals[i]) return false;
            }
            return true;
        }

        /// <summary>Gathers the [offset, offset+1] element pair of a rank-1 int64 vector — a key's two words.</summary>
        private static FastTensorKey AppendGatherPair(
            FastTensorKey vector, int offset, List<FastNode> newNodes)
        {
            var indices = new OnnxTensorData<int64>(
                new Shape(2), OnnxUtils.CreateTensorValue(new Shape(2), (long[])[offset, offset + 1]));
            var indicesKey = AppendConstant(indices, newNodes);
            return AppendBinaryOpWithAttrs(OpCodes.GATHER,
                new Dictionary<string, object?> { [AttrAxis] = 0L }, vector, indicesKey, newNodes);
        }

        /// <summary>Gathers element <paramref name="index"/> of a rank-1 int64 vector as a rank-0 scalar.</summary>
        private static FastTensorKey AppendGatherScalar(
            FastTensorKey vector, int index, List<FastNode> newNodes)
        {
            var indexConst = AppendScalarInt64(index, newNodes);
            return AppendBinaryOpWithAttrs(OpCodes.GATHER,
                new Dictionary<string, object?> { [AttrAxis] = 0L }, vector, indexConst, newNodes);
        }

        private static FastTensorKey AppendSplit(
            FastTensorKey key, FastTensorKey counter, List<FastNode> newNodes)
        {
            // The split (key-tree fold) is deliberately algorithm-independent — see
            // RngAlgorithms.GetFunction — so the chain is wired with the default name once
            // and never re-wired on an algorithm switch.
            return AppendBinaryOpWithAttrs(InternalOpCodes.SHRK_RNG_SPLIT,
                new Dictionary<string, object?> { [ShrkAttrRngAlgorithm] = RngAlgorithms.Default },
                key, counter, newNodes);
        }

        private static FastTensorKey AppendBinaryOp(
            string opCode, FastTensorKey a, FastTensorKey b, List<FastNode> newNodes)
            => AppendBinaryOpWithAttrs(opCode, new Dictionary<string, object?>(), a, b, newNodes);

        private static FastTensorKey AppendBinaryOpWithAttrs(
            string opCode, Dictionary<string, object?> attrs,
            FastTensorKey a, FastTensorKey b, List<FastNode> newNodes)
        {
            var nodeKey = FastNodeKey.New();
            var outKey = new FastTensorKey(nodeKey, 0);
            var attrDefs = Definitions.NodeDefinitions[opCode].AttributeDefs;
            newNodes.Add(new FastNode
            {
                Key = nodeKey,
                OpCode = opCode,
                Attributes = OnnxCSharpAttributes.FromCSharpVals(attrs, attrDefs),
                FullInputs = { [""] = new List<FastTensorKey?> { a, b } },
                FullOutputs = { [""] = new List<FastTensorKey?> { outKey } },
            });
            return outKey;
        }

        private static FastTensorKey AppendWhere(
            FastTensorKey cond, FastTensorKey ifTrue, FastTensorKey ifFalse, List<FastNode> newNodes)
        {
            var nodeKey = FastNodeKey.New();
            var outKey = new FastTensorKey(nodeKey, 0);
            var attrDefs = Definitions.NodeDefinitions[OpCodes.WHERE].AttributeDefs;
            newNodes.Add(new FastNode
            {
                Key = nodeKey,
                OpCode = OpCodes.WHERE,
                Attributes = OnnxCSharpAttributes.FromCSharpVals(new Dictionary<string, object?>(), attrDefs),
                FullInputs = { [""] = new List<FastTensorKey?> { cond, ifTrue, ifFalse } },
                FullOutputs = { [""] = new List<FastTensorKey?> { outKey } },
            });
            return outKey;
        }

        private static FastTensorKey AppendScalarInt64(long value, List<FastNode> newNodes)
        {
            var data = new OnnxTensorData<int64>(
                new Shape(Array.Empty<long>()),
                OnnxUtils.CreateTensorValue(new Shape(Array.Empty<long>()), (long[])[value]));
            return AppendConstant(data, newNodes);
        }

        private static FastTensorKey AppendConstant(TensorData data, List<FastNode> newNodes)
        {
            var constAttrDefs = Definitions.NodeDefinitions[OpCodes.CONSTANT].AttributeDefs;
            var key = FastNodeKey.New();
            var outKey = new FastTensorKey(key, 0);
            newNodes.Add(new FastNode
            {
                Key = key,
                OpCode = OpCodes.CONSTANT,
                Attributes = OnnxCSharpAttributes.FromCSharpVals(
                    new Dictionary<string, object?> { [AttrValue] = data }, constAttrDefs),
                FullInputs = new Dictionary<string, List<FastTensorKey?>>(),
                FullOutputs = { [""] = new List<FastTensorKey?> { outKey } },
            });
            return outKey;
        }
    }
}
