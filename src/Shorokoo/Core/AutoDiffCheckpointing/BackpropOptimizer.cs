using Shorokoo.Core.AutoDiffCheckpointing;
using Shorokoo.Core.Graph;
using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using System.Collections.Generic;
using System.Linq;

namespace Shorokoo.Core.AutoDiffCheckpointing;

/// <summary>
/// Result of a backprop optimization pass, containing the optimized
/// <see cref="InternalComputationGraph"/> and its evaluation metrics.
/// </summary>
internal class BackpropOptimizationResult
{
    /// <summary>The optimized <see cref="InternalComputationGraph"/>.</summary>
    public required InternalComputationGraph OptimizedGraph { get; init; }

    /// <summary>
    /// The combined metric score for the optimized graph
    /// (lower is better: computeFactor * computeTime + memoryFactor * peakMemoryBytes).
    /// </summary>
    public double CombinedMetric { get; init; }

    /// <summary>
    /// The evaluation result obtained by running the evaluator on the optimized computation graph.
    /// </summary>
    public required GraphEvaluationResult Evaluation { get; init; }

    public override string ToString()
        => $"CombinedMetric={CombinedMetric:F4}, " +
           $"Compute={Evaluation.TotalComputeTime:F2}, PeakMemory={Evaluation.PeakMemoryBytes / (1024.0 * 1024.0):F2} MB";
}

/// <summary>
/// Optimizes backpropagation graph execution using iterative replay: greedily selects
/// tensors to recompute instead of store to reduce peak memory, repeating up to
/// maxIterations times as long as the combined metric improves.
/// </summary>
internal class SimpleBackpropOptimizer
{
    private readonly GraphEvaluator _evaluator;
    private readonly double _computeFactor;
    private readonly double _memoryFactor;
    private readonly int _maxIterations;

    public SimpleBackpropOptimizer(
        double computeFactor = 1.0,
        double memoryFactor = 1e-6,
        int maxIterations = 20,
        GraphEvaluator? evaluator = null)
    {
        _evaluator = evaluator ?? new GraphEvaluator();
        _computeFactor = computeFactor;
        _memoryFactor = memoryFactor;
        _maxIterations = maxIterations;
    }

    /// <summary>
    /// Computes the combined metric for a given evaluation result.
    /// </summary>
    public double ComputeCombinedMetric(GraphEvaluationResult eval)
        => _computeFactor * eval.TotalComputeTime + _memoryFactor * eval.PeakMemoryBytes;

    /// <summary>
    /// Optimizes the given graph using iterative replay and returns the result.
    /// </summary>
    public BackpropOptimizationResult Optimize(InternalComputationGraph graph, ShapeInferenceResult shapeInfo)
    {
        var optimizedGraph = BuildOptimizedGraph(graph, shapeInfo);

        var evaluation = _evaluator.Evaluate(optimizedGraph, shapeInfo);
        var metric = ComputeCombinedMetric(evaluation);

        return new BackpropOptimizationResult
        {
            OptimizedGraph = optimizedGraph,
            CombinedMetric = metric,
            Evaluation = evaluation,
        };
    }

    private InternalComputationGraph BuildOptimizedGraph(InternalComputationGraph graph, ShapeInferenceResult shapeInfo)
    {
        var baseEval = _evaluator.Evaluate(graph, shapeInfo);
        var replayedTensors = IdentifyTensorsToReplay(graph.Nodes, shapeInfo, baseEval);

        if (replayedTensors.Count == 0)
            return graph;

        return ApplyReplayTransformation(graph, replayedTensors);
    }

    /// <summary>
    /// Transforms the graph by inserting recomputation nodes for replayed tensors.
    /// For each replayed tensor with multiple consumers, the first consumer keeps the
    /// original tensor and subsequent consumers get a freshly recomputed copy.
    /// Returns <paramref name="graph"/> unchanged (same reference) when no recomputation
    /// nodes were actually inserted.
    /// </summary>
    internal static InternalComputationGraph ApplyReplayTransformation(
        InternalComputationGraph graph, HashSet<FastTensorKey> replayedTensors)
    {
        // Map: replayed tensor key → its producer node (read off the source graph)
        var tensorProducerNode = new Dictionary<FastTensorKey, FastNode>();
        foreach (var node in graph.Nodes)
        {
            foreach (var output in node.Outputs)
            {
                if (output is null) continue;
                if (replayedTensors.Contains(output.Value))
                    tensorProducerNode[output.Value] = node;
            }
        }

        // First, count how many recomputation insertions would actually happen — if none,
        // return the original graph reference so no-op rebuilds reuse the source instance.
        var firstSeen = new HashSet<FastTensorKey>();
        bool anyRecompute = false;
        foreach (var node in graph.Nodes)
        {
            foreach (var input in node.Inputs)
            {
                if (input is null) continue;
                if (!replayedTensors.Contains(input.Value)) continue;
                if (!firstSeen.Add(input.Value))
                {
                    anyRecompute = true;
                    break;
                }
            }
            if (anyRecompute) break;
        }
        if (!anyRecompute) return graph;

        // Now do the actual transformation on a clone.
        var copy = graph.Clone();
        // Re-derive producer map on the clone so node references are clone-local.
        var clonedProducerNode = new Dictionary<FastTensorKey, FastNode>();
        foreach (var node in copy.Nodes)
        {
            foreach (var output in node.Outputs)
            {
                if (output is null) continue;
                if (replayedTensors.Contains(output.Value))
                    clonedProducerNode[output.Value] = node;
            }
        }

        var firstConsumed = new HashSet<FastTensorKey>();
        var newNodes = new List<FastNode>(copy.Nodes.Count);
        foreach (var node in copy.Nodes)
        {
            foreach (var (slotName, slot) in node.FullInputs)
            {
                for (int i = 0; i < slot.Count; i++)
                {
                    var k = slot[i];
                    if (k is null) continue;
                    if (!replayedTensors.Contains(k.Value)) continue;
                    if (firstConsumed.Add(k.Value)) continue;

                    if (!clonedProducerNode.TryGetValue(k.Value, out var producer))
                        continue;

                    var clone = CloneProducerForRecompute(producer);
                    if (clone is null) continue;

                    int outIdx = k.Value.OutputIndex;
                    var newOutputKey = FindOutputAt(clone, outIdx);
                    if (newOutputKey is null) continue;

                    newNodes.Add(clone);
                    slot[i] = newOutputKey.Value;
                }
            }

            newNodes.Add(node);
        }

        copy.Nodes = newNodes;
        return copy;
    }

    /// <summary>
    /// Clones <paramref name="producer"/> with a fresh node key and fresh output keys (matching
    /// the original output structure). Returns null if the producer is structural / unsupported
    /// (open/close, model input, etc.). Inputs are preserved verbatim; the clone re-references
    /// the same upstream tensors.
    /// </summary>
    private static FastNode? CloneProducerForRecompute(FastNode producer)
    {
        if (producer.IsOpenNode() || producer.IsCloseNode() ||
            producer.IsFunction() || producer.IsModelInput() ||
            producer.IsModelParamData())
            return null;

        // Single-output check: the rebuild logic below handles multi-output nodes too,
        // but the rest of the rematerialization machinery only requests single-output replay.
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
            // GraphOpenNodeKey and TargetFunction are intentionally not copied — those are
            // structural / function-call concerns ruled out above.
        };
    }

    private static FastTensorKey? FindOutputAt(FastNode node, int outputIndex)
    {
        foreach (var slot in node.FullOutputs.Values)
            foreach (var k in slot)
                if (k is FastTensorKey key && key.OutputIndex == outputIndex)
                    return key;
        return null;
    }

    /// <summary>
    /// Iterative replay: greedily identifies tensors whose removal from memory (replaced
    /// with recomputation) would most reduce peak memory.
    /// </summary>
    private HashSet<FastTensorKey> IdentifyTensorsToReplay(
        IList<FastNode> nodes,
        ShapeInferenceResult shapeInfo,
        GraphEvaluationResult baseEval)
    {
        var tensorInfo = BuildTensorReplayInfo(nodes, shapeInfo, baseEval);

        var replayedTensors = new HashSet<FastTensorKey>();

        double currentComputeTime = baseEval.TotalComputeTime;
        long currentPeakMemory = baseEval.PeakMemoryBytes;
        double bestMetric = _computeFactor * currentComputeTime + _memoryFactor * currentPeakMemory;

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            FastTensorKey? bestCandidate = null;
            double bestCandidateMetricDelta = 0;
            long bestMemorySaving = 0;
            double bestExtraCompute = 0;

            foreach (var (key, info) in tensorInfo)
            {
                if (replayedTensors.Contains(key)) continue;
                if (info.MemoryBytes <= 0) continue;
                if (info.IsGraphInput) continue;

                var memorySaving = info.IsLiveAtPeak ? info.MemoryBytes : 0;
                if (memorySaving <= 0) continue;

                var extraCompute = (info.UseCount - 1) * info.ProducerComputeCost;
                if (extraCompute < 0) extraCompute = 0;

                var metricDelta = _computeFactor * extraCompute - _memoryFactor * memorySaving;

                if (metricDelta < bestCandidateMetricDelta ||
                    (bestCandidate is null && metricDelta < 0))
                {
                    bestCandidate = key;
                    bestCandidateMetricDelta = metricDelta;
                    bestMemorySaving = memorySaving;
                    bestExtraCompute = extraCompute;
                }
            }

            if (bestCandidate is null)
                break;

            replayedTensors.Add(bestCandidate.Value);
            currentComputeTime += bestExtraCompute;
            currentPeakMemory -= bestMemorySaving;

            var newMetric = _computeFactor * currentComputeTime + _memoryFactor * currentPeakMemory;
            if (newMetric >= bestMetric)
            {
                replayedTensors.Remove(bestCandidate.Value);
                currentComputeTime -= bestExtraCompute;
                currentPeakMemory += bestMemorySaving;
                break;
            }
            bestMetric = newMetric;
        }

        return replayedTensors;
    }

    private record TensorReplayInfo(
        long MemoryBytes,
        double ProducerComputeCost,
        int UseCount,
        bool IsGraphInput,
        bool IsLiveAtPeak);

    private static Dictionary<FastTensorKey, TensorReplayInfo> BuildTensorReplayInfo(
        IList<FastNode> nodes,
        ShapeInferenceResult shapeInfo,
        GraphEvaluationResult baseEval)
    {
        var result = new Dictionary<FastTensorKey, TensorReplayInfo>();

        var useCount = new Dictionary<FastTensorKey, int>();
        foreach (var node in nodes)
        {
            foreach (var input in node.Inputs)
            {
                if (input is null) continue;
                useCount.TryGetValue(input.Value, out var c);
                useCount[input.Value] = c + 1;
            }
        }

        var tensorProducerCost = new Dictionary<FastTensorKey, double>();
        var graphInputKeys = new HashSet<FastTensorKey>();

        for (int i = 0; i < nodes.Count; i++)
        {
            var node = nodes[i];
            if (node.IsModelInput() || node.IsModelParamData())
            {
                foreach (var output in node.Outputs)
                    if (output is not null) graphInputKeys.Add(output.Value);
            }

            var nodeCost = baseEval.NodeDetails[i].ComputeTime;
            foreach (var output in node.Outputs)
            {
                if (output is not null)
                    tensorProducerCost[output.Value] = nodeCost;
            }
        }

        // Find peak memory node index
        int peakNodeIdx = 0;
        long peakMem = 0;
        for (int i = 0; i < baseEval.NodeDetails.Count; i++)
        {
            if (baseEval.NodeDetails[i].CurrentMemoryBytes > peakMem)
            {
                peakMem = baseEval.NodeDetails[i].CurrentMemoryBytes;
                peakNodeIdx = i;
            }
        }

        var tensorLastUse = new Dictionary<FastTensorKey, int>();
        for (int i = 0; i < nodes.Count; i++)
        {
            foreach (var input in nodes[i].Inputs)
                if (input is not null) tensorLastUse[input.Value] = i;
        }

        var tensorFirstProduce = new Dictionary<FastTensorKey, int>();
        for (int i = 0; i < nodes.Count; i++)
        {
            foreach (var output in nodes[i].Outputs)
            {
                if (output is not null && !tensorFirstProduce.ContainsKey(output.Value))
                    tensorFirstProduce[output.Value] = i;
            }
        }

        var allTensorKeys = new HashSet<FastTensorKey>();
        foreach (var node in nodes)
        {
            foreach (var output in node.Outputs)
                if (output is not null) allTensorKeys.Add(output.Value);
        }

        foreach (var key in allTensorKeys)
        {
            var info = shapeInfo.GetTensorInfo(key);
            long memBytes = info?.MemoryBytes ?? 0;
            double producerCost = tensorProducerCost.GetValueOrDefault(key, 0);
            int uses = useCount.GetValueOrDefault(key, 0);
            bool isInput = graphInputKeys.Contains(key);

            bool isLiveAtPeak = false;
            if (tensorFirstProduce.TryGetValue(key, out var firstProd) &&
                tensorLastUse.TryGetValue(key, out var lastUse))
            {
                isLiveAtPeak = firstProd <= peakNodeIdx && lastUse >= peakNodeIdx;
            }

            result[key] = new TensorReplayInfo(memBytes, producerCost, uses, isInput, isLiveAtPeak);
        }

        return result;
    }
}
