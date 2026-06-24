using Shorokoo.Core.AutoDiffCheckpointing.OpsPerf;
using Shorokoo.Core.Graph;
using Shorokoo.Core.AutoDiffCheckpointing;
using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using System.Collections.Generic;
using System.Linq;

namespace Shorokoo.Core.AutoDiffCheckpointing;

/// <summary>
/// Evaluates a <see cref="FastComputationGraph"/>'s performance by walking through nodes in
/// execution order, tracking cumulative compute time and peak memory usage.
///
/// Memory tracking:
/// - A tensor is not loaded into memory until it first appears as an input to a node.
/// - Once a tensor is used and no subsequent node requires it, its memory is freed.
/// - In-place buffer reuse is considered where ops support it.
///
/// Uses ShapeInference data and per-op performance models to produce estimates.
/// </summary>
internal class GraphEvaluator
{
    private readonly OpPerfRegistry _perfRegistry;

    public GraphEvaluator(OpPerfRegistry? perfRegistry = null)
    {
        _perfRegistry = perfRegistry ?? new OpPerfRegistry();
    }

    /// <summary>
    /// Evaluates the given <see cref="FastComputationGraph"/> using shape inference data.
    /// </summary>
    public GraphEvaluationResult Evaluate(FastComputationGraph graph, ShapeInferenceResult shapeInfo)
    {
        var nodes = graph.Nodes;

        // Build a map of tensor key → last node index that uses it as input.
        var tensorLastUseIndex = BuildTensorLastUseIndex(nodes);

        // Track which tensors are currently in memory and their sizes
        var liveMemory = new Dictionary<FastTensorKey, long>();
        long currentMemoryBytes = 0;
        long peakMemoryBytes = 0;
        double cumulativeComputeTime = 0;

        var nodeDetails = new List<NodeEvaluationInfo>(nodes.Count);

        for (int nodeIdx = 0; nodeIdx < nodes.Count; nodeIdx++)
        {
            var node = nodes[nodeIdx];

            // Skip pure metadata nodes
            if (node.IsModelInput())
            {
                // Model input tensors are allocated but we don't count them
                // until first use — they'll be added when first consumed.
                nodeDetails.Add(new NodeEvaluationInfo
                {
                    OpCode = node.OpCode,
                    ComputeTime = 0,
                    ExtraMemoryBytes = 0,
                    CurrentMemoryBytes = currentMemoryBytes,
                    CumulativeComputeTime = cumulativeComputeTime
                });
                continue;
            }

            var nodeInputs = node.Inputs;
            var nodeOutputs = node.Outputs;

            // Step 1: Load input tensors into memory if not already loaded
            foreach (var input in nodeInputs)
            {
                if (input is null) continue;
                var key = input.Value;
                if (!liveMemory.ContainsKey(key))
                {
                    var info = shapeInfo.GetTensorInfo(key);
                    if (info is not null)
                    {
                        liveMemory[key] = info.MemoryBytes;
                        currentMemoryBytes += info.MemoryBytes;
                    }
                }
            }

            // Step 2: Compute op performance
            var perfInput = BuildOpPerfInput(node, nodeIdx, shapeInfo, tensorLastUseIndex);
            var perfResult = _perfRegistry.Estimate(perfInput);

            // Add extra workspace memory temporarily
            var peakDuringOp = currentMemoryBytes + perfResult.ExtraMemoryBytes;

            // Step 3: Add output tensor memory (accounting for in-place reuse)
            var inPlaceReuse = perfResult.InPlaceBufferReuse;

            for (int outIdx = 0; outIdx < nodeOutputs.Count; outIdx++)
            {
                var output = nodeOutputs[outIdx];
                if (output is null) continue;

                var outputInfo = shapeInfo.GetTensorInfo(output.Value);
                if (outputInfo is null) continue;

                if (inPlaceReuse.TryGetValue(outIdx, out var reusedInputIdx)
                    && reusedInputIdx >= 0 && reusedInputIdx < nodeInputs.Count)
                {
                    // In-place: output reuses the buffer of the specified input.
                    var reusedInputKey = nodeInputs[reusedInputIdx];
                    if (reusedInputKey is not null && liveMemory.ContainsKey(reusedInputKey.Value))
                    {
                        // Transfer the memory tracking from input to output
                        var inputMem = liveMemory[reusedInputKey.Value];
                        liveMemory.Remove(reusedInputKey.Value);
                        liveMemory[output.Value] = outputInfo.MemoryBytes;
                        // Adjust if output size differs from input size
                        currentMemoryBytes += outputInfo.MemoryBytes - inputMem;
                    }
                    else
                    {
                        liveMemory[output.Value] = outputInfo.MemoryBytes;
                        currentMemoryBytes += outputInfo.MemoryBytes;
                    }
                }
                else
                {
                    liveMemory[output.Value] = outputInfo.MemoryBytes;
                    currentMemoryBytes += outputInfo.MemoryBytes;
                }
            }

            // Update peak including workspace
            var peakAfterOutputs = currentMemoryBytes + perfResult.ExtraMemoryBytes;
            peakDuringOp = System.Math.Max(peakDuringOp, peakAfterOutputs);
            if (peakDuringOp > peakMemoryBytes)
                peakMemoryBytes = peakDuringOp;

            // Step 4: Free input tensors that are no longer needed
            foreach (var input in nodeInputs)
            {
                if (input is null) continue;
                var key = input.Value;

                // Check if this was an in-place reuse — already handled above
                if (!liveMemory.ContainsKey(key)) continue;

                // Check if any later node still needs this tensor
                if (tensorLastUseIndex.TryGetValue(key, out var lastIdx) && lastIdx <= nodeIdx)
                {
                    currentMemoryBytes -= liveMemory[key];
                    liveMemory.Remove(key);
                }
            }

            // Update peak after frees
            if (currentMemoryBytes > peakMemoryBytes)
                peakMemoryBytes = currentMemoryBytes;

            cumulativeComputeTime += perfResult.ComputeTime;

            nodeDetails.Add(new NodeEvaluationInfo
            {
                OpCode = node.OpCode,
                ComputeTime = perfResult.ComputeTime,
                ExtraMemoryBytes = perfResult.ExtraMemoryBytes,
                CurrentMemoryBytes = currentMemoryBytes,
                CumulativeComputeTime = cumulativeComputeTime
            });
        }

        return new GraphEvaluationResult
        {
            TotalComputeTime = cumulativeComputeTime,
            PeakMemoryBytes = peakMemoryBytes,
            NodeDetails = nodeDetails,
        };
    }

    /// <summary>
    /// Builds a map of tensor key → last node index where that tensor is used as input.
    /// </summary>
    private static Dictionary<FastTensorKey, int> BuildTensorLastUseIndex(
        System.Collections.Generic.IList<FastNode> nodes)
    {
        var lastUse = new Dictionary<FastTensorKey, int>();
        for (int i = 0; i < nodes.Count; i++)
        {
            foreach (var input in nodes[i].Inputs)
            {
                if (input is not null)
                    lastUse[input.Value] = i;
            }
        }
        return lastUse;
    }

    /// <summary>
    /// Builds the OpPerfInput for a given node.
    /// </summary>
    private static OpPerfInput BuildOpPerfInput(
        FastNode node,
        int nodeIdx,
        ShapeInferenceResult shapeInfo,
        Dictionary<FastTensorKey, int> tensorLastUseIndex)
    {
        var nodeInputs = node.Inputs;
        var nodeOutputs = node.Outputs;

        var inputShapes = new TensorShapeInfo?[nodeInputs.Count];
        var inputMustRemainIntact = new bool[nodeInputs.Count];

        for (int i = 0; i < nodeInputs.Count; i++)
        {
            var input = nodeInputs[i];
            if (input is null) continue;

            inputShapes[i] = shapeInfo.GetTensorInfo(input.Value);

            // Input must remain intact if it's used by any later node
            if (tensorLastUseIndex.TryGetValue(input.Value, out var lastIdx))
                inputMustRemainIntact[i] = lastIdx > nodeIdx;
            else
                inputMustRemainIntact[i] = true; // Conservative: keep alive if unknown
        }

        var outputShapes = new TensorShapeInfo?[nodeOutputs.Count];
        for (int i = 0; i < nodeOutputs.Count; i++)
        {
            var output = nodeOutputs[i];
            if (output is not null)
                outputShapes[i] = shapeInfo.GetTensorInfo(output.Value);
        }

        // Extract attributes as dictionary
        var attrVals = node.Attributes.GetAttributeVals();
        var attrs = new Dictionary<string, object?>();
        foreach (var kvp in attrVals)
        {
            if (!kvp.Key.StartsWith("shrk_"))
                attrs[kvp.Key] = kvp.Value;
        }

        return new OpPerfInput
        {
            InputShapes = inputShapes,
            OutputShapes = outputShapes,
            InputMustRemainIntact = inputMustRemainIntact,
            OpCode = node.OpCode,
            Attributes = attrs,
        };
    }
}
