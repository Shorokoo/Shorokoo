using Shorokoo.Core.AutoDiffCheckpointing;
using Shorokoo.Core.Graph;
using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using System.Collections.Generic;
using System.Linq;

namespace Shorokoo.Core.AutoDiffCheckpointing;

/// <summary>
/// Identifies tensors that should be recomputed rather than stored in memory, then
/// transforms the graph by inserting recomputation nodes for those tensors.
///
/// <para>Recomputation only pays when it recomputes a <b>chain</b> back to tensors that are
/// alive at the recompute point anyway. Cloning a single producer and letting it read its
/// original inputs — which is all this did before — forces those inputs to stay alive to the
/// recompute point, and for a MatMul or any elementwise op the inputs are at least as large
/// as the output. The lifetime extended cancels the lifetime shortened, so single-node
/// recompute has a net saving of about zero and an honest evaluator rejects every candidate
/// it proposes. That is checkpointing's whole trick: walk back to a frontier that costs
/// nothing to keep, and free everything between.</para>
///
/// <para>The walk stops at a tensor that is live across the recompute point for its own
/// reasons, or that is a graph input, parameter or constant. Everything strictly inside the
/// walk is recomputed, so nothing's lifetime is extended by construction — the saving is the
/// bytes of the interior, and the cost is the interior's compute. A transformer's attention
/// is the archetype: the softmax output and the scores it came from both span the forward-to-
/// backward gap, and both recompute from Q and K, which the gradient needs anyway.</para>
/// </summary>
internal class Rematerializer
{
    private readonly GraphEvaluator _evaluator;
    private readonly ComputeMemoryObjective _objective;
    private readonly int _maxIterations;

    public Rematerializer(
        ComputeMemoryObjective objective,
        int maxIterations = MemoryAwareGraphOptimizer.DefaultRematerializationIterations,
        GraphEvaluator? evaluator = null)
    {
        _evaluator = evaluator ?? new GraphEvaluator();
        _objective = objective;
        _maxIterations = maxIterations;
    }

    /// <summary>
    /// Computes the combined metric for a given evaluation result.
    /// </summary>
    public double ComputeCombinedMetric(GraphEvaluationResult eval) => _objective.Score(eval);

    /// <summary>
    /// Applies rematerialization to the graph, returning the rewritten graph together with
    /// shape information that covers it.
    ///
    /// <para>Returning the shape info is not a convenience. Every recomputed tensor gets a
    /// fresh key, absent from the caller's original <paramref name="shapeInfo"/>, and
    /// <see cref="GraphEvaluator"/> skips an output it has no shape for while the per-op
    /// estimators cost a null output shape at zero. Evaluating the rewritten graph against
    /// the ORIGINAL shape info therefore prices every clone this pass just inserted at zero
    /// bytes and zero time — which makes rematerialization look free and lets the caller
    /// accept transforms that are strictly worse on both axes.</para>
    /// </summary>
    public (InternalComputationGraph Graph, ShapeInferenceResult ShapeInfo) Apply(
        InternalComputationGraph graph, ShapeInferenceResult shapeInfo)
    {
        var currentGraph = graph;
        var currentShapeInfo = shapeInfo;
        var currentEval = _evaluator.Evaluate(currentGraph, currentShapeInfo);
        var currentScore = _objective.Score(currentEval);

        // Peak memory is a MAXIMUM over the schedule, and that shapes the search. Freeing one
        // tensor lowers one node and exposes the next-highest, so committing chains singly can
        // stall on a plateau where only the group wins; committing the group can lose the other
        // way, because every chain rebuilds its own intermediates and a hundred of them cost
        // more than the liveness they buy. So each round tries the whole batch first and then
        // falls back to walking the ranked list, keeping whatever individually improves.
        var evaluations = 0;
        while (evaluations < MaxEvaluationsPerCall)
        {
            var chains = FindRematerializationChains(currentGraph.Nodes, currentShapeInfo, currentEval);
            if (chains.Count == 0)
                break;

            var accepted = false;

            bool Commit(List<RematChain> batch)
            {
                var (candidateGraph, tensorMapping) = ApplyChains(currentGraph, batch);
                if (tensorMapping.Count == 0)
                    return false;

                var candidateShapeInfo = AugmentShapeInfo(currentShapeInfo, tensorMapping);
                var candidateEval = _evaluator.Evaluate(candidateGraph, candidateShapeInfo);
                evaluations++;

                var candidateScore = _objective.Score(candidateEval);
                if (candidateScore >= currentScore)
                    return false;

                currentGraph = candidateGraph;
                currentShapeInfo = candidateShapeInfo;
                currentEval = candidateEval;
                currentScore = candidateScore;
                return true;
            }

            if (chains.Count > 1 && Commit(chains))
                accepted = true;

            if (!accepted)
            {
                foreach (var chain in chains)
                {
                    if (evaluations >= MaxEvaluationsPerCall)
                        break;
                    if (Commit([chain]))
                        accepted = true;
                }
            }

            if (!accepted)
                break;
        }

        return (currentGraph, currentShapeInfo);
    }

    private static ShapeInferenceResult AugmentShapeInfo(
        ShapeInferenceResult baseInfo,
        Dictionary<FastTensorKey, FastTensorKey> newToOriginalMapping)
    {
        if (newToOriginalMapping.Count == 0)
            return baseInfo;

        var builder = baseInfo.TensorInfos.ToBuilder();
        foreach (var (newKey, originalKey) in newToOriginalMapping)
        {
            var originalInfo = baseInfo.GetTensorInfo(originalKey);
            if (originalInfo is not null)
                builder[newKey] = originalInfo;
        }

        return new ShapeInferenceResult(builder.ToImmutable());
    }

    /// <summary>Longest producer chain the walk will recompute. A chain that reaches this
    /// deep is either not anchored to live tensors or is duplicating too much work to be
    /// worth it, and every extra link multiplies both the clone cost and the search.</summary>
    private const int MaxChainNodes = 12;

    /// <summary>
    /// Ceiling on full-graph evaluations spent per call. Each committed chain costs one
    /// evaluation and each rejected trial costs one too, so this bounds the pass's cost to
    /// something proportional to graph size rather than to how many candidates happen to look
    /// profitable — which on a transformer is hundreds.
    /// </summary>
    private const int MaxEvaluationsPerCall = 48;

    /// <summary>
    /// A tensor to recompute just before one late consumer, together with the producer chain
    /// that rebuilds it from tensors already live at that point.
    /// </summary>
    private sealed class RematChain
    {
        public required FastTensorKey Target { get; init; }
        public required FastNode Consumer { get; init; }
        /// <summary>Producers to clone, in topological order — deepest dependency first.</summary>
        public required List<FastNode> Chain { get; init; }
        public required long FreedBytes { get; init; }
        public required double ExtraCompute { get; init; }
        public double ExpectedMetricDelta { get; init; }
    }

    private List<RematChain> FindRematerializationChains(
        IList<FastNode> graphNodes,
        ShapeInferenceResult shapeInfo,
        GraphEvaluationResult eval)
    {
        // Every position below is a position in the evaluation's walk — ORT's execution order,
        // not the linear order — so that "spans the peak" and "live at the recompute point"
        // mean what they will mean at runtime.
        IList<FastNode> nodes = eval.NodeDetails.Count == graphNodes.Count
            ? eval.NodeDetails.Select(d => graphNodes[d.NodeIndex]).ToList()
            : graphNodes;

        int peakNodeIdx = 0;
        long peakMem = 0;
        for (int i = 0; i < eval.NodeDetails.Count && i < nodes.Count; i++)
        {
            if (eval.NodeDetails[i].CurrentMemoryBytes > peakMem)
            {
                peakMem = eval.NodeDetails[i].CurrentMemoryBytes;
                peakNodeIdx = i;
            }
        }

        var producerIdx = new Dictionary<FastTensorKey, int>();
        var producerNode = new Dictionary<FastTensorKey, FastNode>();
        var lastUse = new Dictionary<FastTensorKey, int>();
        var rooted = new HashSet<FastTensorKey>();
        var nodeCost = new Dictionary<FastNodeKey, double>();

        for (int i = 0; i < nodes.Count; i++)
        {
            var node = nodes[i];
            nodeCost[node.Key] = i < eval.NodeDetails.Count ? eval.NodeDetails[i].ComputeTime : 0;

            foreach (var input in node.Inputs)
                if (input is not null) lastUse[input.Value] = i;

            foreach (var output in node.Outputs)
            {
                if (output is null) continue;
                producerIdx[output.Value] = i;
                producerNode[output.Value] = node;
                // A graph input, parameter or constant costs nothing to keep: it is resident
                // for the whole step regardless, so the walk can anchor on it for free.
                if (node.IsModelInput() || node.IsModelParamData() || node.Inputs.Count == 0)
                    rooted.Add(output.Value);
            }
        }

        // Live at index i for its own reasons, so anchoring on it extends nothing.
        bool FreeToKeep(FastTensorKey key, int at)
            => rooted.Contains(key)
            || (lastUse.TryGetValue(key, out var last) && last >= at)
            || !producerIdx.ContainsKey(key);

        var chains = new List<RematChain>();
        var seen = new HashSet<(FastNodeKey, FastTensorKey)>();

        for (int consumerIdx = 0; consumerIdx < nodes.Count; consumerIdx++)
        {
            var consumer = nodes[consumerIdx];
            foreach (var input in consumer.Inputs)
            {
                if (input is null) continue;
                var target = input.Value;

                if (!producerIdx.TryGetValue(target, out var pIdx)) continue;
                // Only tensors whose life spans the peak are worth shortening.
                if (!(pIdx <= peakNodeIdx && consumerIdx >= peakNodeIdx)) continue;
                if (!seen.Add((consumer.Key, target))) continue;

                var chain = BuildChain(target, consumerIdx, producerNode, producerIdx, lastUse,
                    nodeCost, shapeInfo, FreeToKeep);
                if (chain is null) continue;

                // A chain frees bytes spread over a long lifetime but pays for them in one
                // burst, rebuilding every intermediate at the recompute point. If that point
                // is already near the peak the burst simply moves the peak there and the whole
                // transform is wasted, so require headroom for it before proposing the chain
                // at all. Without this the ranking is dominated by chains that free the most
                // and raise the peak, and the evaluation budget is spent rediscovering that.
                var memoryAtRecompute = consumerIdx < eval.NodeDetails.Count
                    ? eval.NodeDetails[consumerIdx].CurrentMemoryBytes : peakMem;
                if (memoryAtRecompute + chain.FreedBytes > peakMem) continue;

                var delta = _objective.TradeDelta(chain.ExtraCompute, chain.FreedBytes);
                if (delta >= 0) continue;

                chains.Add(new RematChain
                {
                    Target = chain.Target,
                    Consumer = consumer,
                    Chain = chain.Chain,
                    FreedBytes = chain.FreedBytes,
                    ExtraCompute = chain.ExtraCompute,
                    ExpectedMetricDelta = delta,
                });
            }
        }

        chains.Sort((a, b) => a.ExpectedMetricDelta.CompareTo(b.ExpectedMetricDelta));
        return chains;
    }

    /// <summary>
    /// Walks back from <paramref name="target"/> collecting the producers needed to rebuild it
    /// at <paramref name="recomputeAt"/>, stopping at any tensor that is free to keep there.
    /// Returns null when the chain cannot be built, runs past <see cref="MaxChainNodes"/>, or
    /// bottoms out on something that cannot be cloned.
    /// </summary>
    private static RematChain? BuildChain(
        FastTensorKey target,
        int recomputeAt,
        Dictionary<FastTensorKey, FastNode> producerNode,
        Dictionary<FastTensorKey, int> producerIdx,
        Dictionary<FastTensorKey, int> lastUse,
        Dictionary<FastNodeKey, double> nodeCost,
        ShapeInferenceResult shapeInfo,
        Func<FastTensorKey, int, bool> freeToKeep)
    {
        var order = new List<FastNode>();
        var visited = new HashSet<FastNodeKey>();
        long freed = 0;
        double cost = 0;
        var ok = true;

        void Walk(FastTensorKey key)
        {
            if (!ok) return;
            if (freeToKeep(key, recomputeAt)) return;
            if (!producerNode.TryGetValue(key, out var producer)) { ok = false; return; }
            if (!visited.Add(producer.Key)) return;
            if (CloneProducerForRecompute(producer) is null) { ok = false; return; }
            if (order.Count >= MaxChainNodes) { ok = false; return; }

            foreach (var input in producer.Inputs)
                if (input is not null) Walk(input.Value);

            if (!ok) return;
            order.Add(producer);
            cost += nodeCost.GetValueOrDefault(producer.Key, 0);
            freed += shapeInfo.GetTensorInfo(key)?.MemoryBytes ?? 0;
        }

        // The target itself always needs recomputing, even though it is live at recomputeAt
        // (that is the consumption we are replacing), so start from its producer directly.
        if (!producerNode.TryGetValue(target, out var targetProducer)) return null;
        if (CloneProducerForRecompute(targetProducer) is null) return null;
        visited.Add(targetProducer.Key);
        foreach (var input in targetProducer.Inputs)
            if (input is not null) Walk(input.Value);
        if (!ok) return null;
        order.Add(targetProducer);
        cost += nodeCost.GetValueOrDefault(targetProducer.Key, 0);
        freed += shapeInfo.GetTensorInfo(target)?.MemoryBytes ?? 0;

        if (order.Count == 0 || freed <= 0) return null;

        return new RematChain
        {
            Target = target,
            Consumer = targetProducer,   // replaced by the caller
            Chain = order,
            FreedBytes = freed,
            ExtraCompute = cost,
        };
    }

    /// <summary>
    /// Inserts each chain's cloned producers immediately after the target's original
    /// producer, and rewires the consumer that asked for them to read the clone. Returns the
    /// rewritten graph and a map from every minted key to the tensor it replicates, so the
    /// caller can extend shape information to cover them.
    ///
    /// <para>The linear position decides when ORT runs the clones, and it is not "as late as
    /// possible" that puts them late: ORT visits a node's producers highest-index first, so a
    /// chain placed just before its consumer would be recomputed BEFORE the consumer's other
    /// inputs (the gradients of the backward pass) and then held across all of them. Placed
    /// low, next to the original producer, the chain is the consumer's lowest-index producer
    /// and runs last — right before the consumer, which is the point of recomputing.</para>
    /// </summary>
    private static (InternalComputationGraph graph, Dictionary<FastTensorKey, FastTensorKey> tensorMapping) ApplyChains(
        InternalComputationGraph graph, List<RematChain> chains)
    {
        var copy = graph.Clone();
        var byProducer = new Dictionary<FastNodeKey, List<RematChain>>();
        foreach (var chain in chains)
        {
            if (!byProducer.TryGetValue(chain.Chain[^1].Key, out var list))
                byProducer[chain.Chain[^1].Key] = list = new List<RematChain>();
            list.Add(chain);
        }

        var newToOriginal = new Dictionary<FastTensorKey, FastTensorKey>();
        var newNodes = new List<FastNode>(copy.Nodes.Count);
        var rewires = new Dictionary<FastNodeKey, List<(FastTensorKey Target, FastTensorKey Recomputed)>>();

        foreach (var node in copy.Nodes)
        {
            if (rewires.TryGetValue(node.Key, out var pending))
                foreach (var (target, recomputed) in pending)
                    foreach (var slot in node.FullInputs.Values)
                        for (int i = 0; i < slot.Count; i++)
                            if (slot[i] is FastTensorKey k && k.Equals(target))
                                slot[i] = recomputed;

            newNodes.Add(node);

            if (byProducer.TryGetValue(node.Key, out var here))
            {
                foreach (var chain in here)
                {
                    // Fresh key per cloned producer; interior references are remapped so the
                    // chain reads its own copies and only its frontier reads the original graph.
                    var remap = new Dictionary<FastTensorKey, FastTensorKey>();
                    var emitted = new List<FastNode>(chain.Chain.Count);
                    var built = true;

                    foreach (var producer in chain.Chain)
                    {
                        var clone = CloneProducerForRecompute(producer);
                        if (clone is null) { built = false; break; }

                        foreach (var slot in clone.FullInputs.Values)
                            for (int i = 0; i < slot.Count; i++)
                                if (slot[i] is FastTensorKey k && remap.TryGetValue(k, out var mapped))
                                    slot[i] = mapped;

                        foreach (var (slotName, slot) in producer.FullOutputs)
                        {
                            var cloneSlot = clone.FullOutputs[slotName];
                            for (int i = 0; i < slot.Count && i < cloneSlot.Count; i++)
                                if (slot[i] is FastTensorKey orig && cloneSlot[i] is FastTensorKey fresh)
                                {
                                    remap[orig] = fresh;
                                    newToOriginal[fresh] = orig;
                                }
                        }

                        emitted.Add(clone);
                    }

                    if (!built || !remap.TryGetValue(chain.Target, out var recomputed))
                        continue;

                    newNodes.AddRange(emitted);
                    if (!rewires.TryGetValue(chain.Consumer.Key, out var list))
                        rewires[chain.Consumer.Key] = list = new List<(FastTensorKey, FastTensorKey)>();
                    list.Add((chain.Target, recomputed));
                }
            }
        }

        copy.Nodes = newNodes;
        return (copy, newToOriginal);
    }

    /// <summary>
    /// Mirror of <see cref="SimpleBackpropOptimizer"/>'s clone logic. Inlined here so the two
    /// optimizers don't share private state, but the algorithm is identical: produce a fresh
    /// FastNode for the same op with new keys, preserving inputs verbatim.
    /// </summary>
    private static FastNode? CloneProducerForRecompute(FastNode producer)
    {
        if (producer.IsOpenNode() || producer.IsCloseNode() ||
            producer.IsFunction() || producer.IsModelInput() ||
            producer.IsModelParamData())
            return null;

        var outputCount = producer.FullOutputs.Values.Sum(s => s.Count(k => k is not null));
        if (outputCount != 1) return null;

        var freshKey = FastNodeKey.New();

        var newFullInputs = new Dictionary<string, List<FastTensorKey?>>();
        foreach (var (slotName, slot) in producer.FullInputs)
            newFullInputs[slotName] = new List<FastTensorKey?>(slot);

        var newFullOutputs = new Dictionary<string, List<FastTensorKey?>>();
        foreach (var (slotName, slot) in producer.FullOutputs)
        {
            var remapped = new List<FastTensorKey?>(slot.Count);
            foreach (var k in slot)
            {
                if (k is null) { remapped.Add(null); continue; }
                if (k.Value.IsEmpty) { remapped.Add(k); continue; }
                remapped.Add(new FastTensorKey(freshKey, k.Value.OutputIndex));
            }
            newFullOutputs[slotName] = remapped;
        }

        return new FastNode
        {
            Key = freshKey,
            OpCode = producer.OpCode,
            Attributes = producer.Attributes,
            FullInputs = newFullInputs,
            FullOutputs = newFullOutputs,
            FriendlyName = producer.FriendlyName,
            StackTrace = producer.StackTrace,
        };
    }
}
