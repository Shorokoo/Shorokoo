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
/// </summary>
internal class Rematerializer
{
    private readonly GraphEvaluator _evaluator;
    private readonly double _computeFactor;
    private readonly double _memoryFactor;
    private readonly int _maxIterations;

    public Rematerializer(
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
    /// Applies rematerialization to the graph.
    /// </summary>
    public FastComputationGraph Apply(FastComputationGraph graph, ShapeInferenceResult shapeInfo)
    {
        var currentGraph = graph;
        var currentShapeInfo = shapeInfo;
        var currentEval = _evaluator.Evaluate(currentGraph, currentShapeInfo);

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            var candidates = FindRematerializationCandidates(
                currentGraph.Nodes, currentShapeInfo, currentEval);

            if (candidates.Count == 0)
                break;

            var tensorsToReplay = new HashSet<FastTensorKey>();
            foreach (var candidate in candidates)
                tensorsToReplay.Add(candidate.TensorKey);

            if (tensorsToReplay.Count == 0)
                break;

            var (candidateGraph, tensorMapping) = ApplyRematerialization(currentGraph, tensorsToReplay);

            if (ReferenceEquals(candidateGraph, currentGraph))
                break;

            var augmentedShapeInfo = AugmentShapeInfo(currentShapeInfo, tensorMapping);

            currentGraph = candidateGraph;
            currentShapeInfo = augmentedShapeInfo;
            currentEval = _evaluator.Evaluate(currentGraph, augmentedShapeInfo);
        }

        return currentGraph;
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

    private List<RematerializationCandidate> FindRematerializationCandidates(
        IList<FastNode> nodes,
        ShapeInferenceResult shapeInfo,
        GraphEvaluationResult eval)
    {
        // Find peak memory node index
        int peakNodeIdx = 0;
        long peakMem = 0;
        for (int i = 0; i < eval.NodeDetails.Count; i++)
        {
            if (eval.NodeDetails[i].CurrentMemoryBytes > peakMem)
            {
                peakMem = eval.NodeDetails[i].CurrentMemoryBytes;
                peakNodeIdx = i;
            }
        }

        var tensorProducerIdx = new Dictionary<FastTensorKey, int>();
        var tensorProducerCost = new Dictionary<FastTensorKey, double>();
        var graphInputKeys = new HashSet<FastTensorKey>();
        var useCount = new Dictionary<FastTensorKey, int>();

        for (int i = 0; i < nodes.Count; i++)
        {
            var node = nodes[i];
            if (node.IsModelInput() || node.IsModelParamData())
            {
                foreach (var output in node.Outputs)
                    if (output is not null) graphInputKeys.Add(output.Value);
            }

            foreach (var input in node.Inputs)
            {
                if (input is null) continue;
                useCount.TryGetValue(input.Value, out var c);
                useCount[input.Value] = c + 1;
            }

            var nodeCost = eval.NodeDetails[i].ComputeTime;
            foreach (var output in node.Outputs)
            {
                if (output is null) continue;
                tensorProducerIdx[output.Value] = i;
                tensorProducerCost[output.Value] = nodeCost;
            }
        }

        var tensorLastUse = new Dictionary<FastTensorKey, int>();
        for (int i = 0; i < nodes.Count; i++)
        {
            foreach (var input in nodes[i].Inputs)
                if (input is not null) tensorLastUse[input.Value] = i;
        }

        var candidates = new List<RematerializationCandidate>();
        foreach (var node in nodes)
        {
            foreach (var output in node.Outputs)
            {
                if (output is null) continue;
                var key = output.Value;

                if (graphInputKeys.Contains(key)) continue;

                var uses = useCount.GetValueOrDefault(key, 0);
                if (uses <= 1) continue;

                var producerIdx = tensorProducerIdx.GetValueOrDefault(key, -1);
                var lastUse = tensorLastUse.GetValueOrDefault(key, -1);
                if (producerIdx < 0 || lastUse < 0) continue;
                if (!(producerIdx <= peakNodeIdx && lastUse >= peakNodeIdx)) continue;

                var info = shapeInfo.GetTensorInfo(key);
                long memBytes = info?.MemoryBytes ?? 0;
                if (memBytes <= 0) continue;

                var producerNode = nodes[producerIdx];
                if (producerNode.IsOpenNode() || producerNode.IsCloseNode() ||
                    producerNode.IsFunction() || producerNode.IsModelInput() ||
                    producerNode.IsModelParamData())
                    continue;

                if (producerNode.Outputs.Count(o => o is not null) != 1)
                    continue;

                var recomputeCost = tensorProducerCost.GetValueOrDefault(key, 0);
                var extraCompute = (uses - 1) * recomputeCost;
                var memorySaving = memBytes;

                var expectedDelta = _computeFactor * extraCompute - _memoryFactor * memorySaving;
                if (expectedDelta >= 0) continue;

                candidates.Add(new RematerializationCandidate
                {
                    TensorKey = key,
                    MemoryBytes = memBytes,
                    ExtraComputeCost = extraCompute,
                    MemorySaving = memorySaving,
                    ExpectedMetricDelta = expectedDelta,
                    UseCount = uses,
                });
            }
        }

        candidates.Sort((a, b) => a.ExpectedMetricDelta.CompareTo(b.ExpectedMetricDelta));
        return candidates;
    }

    /// <summary>
    /// Transforms the graph by inserting recomputation nodes for the specified tensors.
    /// Returns the modified graph and a mapping from new tensor keys to the original
    /// tensor keys they replicate.
    /// </summary>
    private static (FastComputationGraph graph, Dictionary<FastTensorKey, FastTensorKey> tensorMapping) ApplyRematerialization(
        FastComputationGraph graph, HashSet<FastTensorKey> replayedTensors)
    {
        var copy = graph.Clone();

        var tensorProducerNode = new Dictionary<FastTensorKey, FastNode>();
        foreach (var node in copy.Nodes)
        {
            foreach (var output in node.Outputs)
            {
                if (output is null) continue;
                if (replayedTensors.Contains(output.Value))
                    tensorProducerNode[output.Value] = node;
            }
        }

        var firstConsumed = new HashSet<FastTensorKey>();
        var newToOriginal = new Dictionary<FastTensorKey, FastTensorKey>();

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

                    if (!tensorProducerNode.TryGetValue(k.Value, out var producer))
                        continue;

                    var clone = SimpleBackpropOptimizer_CloneProducerForRecompute(producer);
                    if (clone is null) continue;

                    int outIdx = k.Value.OutputIndex;
                    FastTensorKey? newOutputKey = null;
                    foreach (var s in clone.FullOutputs.Values)
                        foreach (var ck in s)
                            if (ck is FastTensorKey cKey && cKey.OutputIndex == outIdx)
                            { newOutputKey = cKey; break; }

                    if (newOutputKey is null) continue;

                    newNodes.Add(clone);
                    newToOriginal[newOutputKey.Value] = k.Value;
                    slot[i] = newOutputKey.Value;
                }
            }

            newNodes.Add(node);
        }

        copy.Nodes = newNodes;
        return (copy, newToOriginal);
    }

    /// <summary>
    /// Mirror of <see cref="SimpleBackpropOptimizer"/>'s clone logic. Inlined here so the two
    /// optimizers don't share private state, but the algorithm is identical: produce a fresh
    /// FastNode for the same op with new keys, preserving inputs verbatim.
    /// </summary>
    private static FastNode? SimpleBackpropOptimizer_CloneProducerForRecompute(FastNode producer)
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

    private class RematerializationCandidate
    {
        public FastTensorKey TensorKey { get; init; }
        public long MemoryBytes { get; init; }
        public double ExtraComputeCost { get; init; }
        public long MemorySaving { get; init; }
        public double ExpectedMetricDelta { get; init; }
        public int UseCount { get; init; }
    }
}
