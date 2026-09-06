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
/// Which execution order <see cref="GraphEvaluator"/> walks a graph in.
/// </summary>
internal enum EvaluationOrder
{
    /// <summary>
    /// The order ONNX Runtime will actually run — see <see cref="OrtExecutionOrder"/>. This is
    /// the order that decides what gets allocated, so it is the default and the one the
    /// memory-aware pass optimizes.
    /// </summary>
    OrtOrder,

    /// <summary>
    /// The graph's own linear order (<see cref="InternalComputationGraph.Nodes"/>, the order
    /// NodeProtos are emitted in). ORT does not run this order; it is kept so the two can be
    /// compared.
    /// </summary>
    ProtoOrder,
}

/// <summary>
/// Evaluates a <see cref="InternalComputationGraph"/>'s performance by walking through nodes in
/// execution order, tracking cumulative compute time and peak memory usage.
///
/// Memory tracking:
/// - A tensor is not loaded into memory until it first appears as an input to a node.
/// - Once a tensor is used and no subsequent node requires it, its memory is freed; an output
///   nothing consumes (and that is not a graph output) is freed at the node that produced it.
/// - In-place buffer reuse is modelled as a shared buffer: the output aliases the input's
///   buffer, which is charged once and freed when every key aliasing it is past its last use.
///   An op may only write in place when no other live key shares that buffer.
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
    /// Evaluates the given <see cref="InternalComputationGraph"/> using shape inference data,
    /// walking it in <paramref name="order"/>.
    /// </summary>
    public GraphEvaluationResult Evaluate(
        InternalComputationGraph graph,
        ShapeInferenceResult shapeInfo,
        EvaluationOrder order = EvaluationOrder.OrtOrder)
    {
        var nodes = graph.Nodes;
        var ortWalk = OrtExecutionOrder.Compute(nodes);
        var walk = order == EvaluationOrder.OrtOrder ? ortWalk : Enumerable.Range(0, nodes.Count).ToArray();

        // Tensor key → last walk position that reads it.
        var tensorLastUse = BuildTensorLastUse(nodes, walk);
        var graphOutputs = new HashSet<FastTensorKey>(graph.Outputs);

        var live = new LiveBuffers();
        long peakMemoryBytes = 0;
        double cumulativeComputeTime = 0;

        var nodeDetails = new List<NodeEvaluationInfo>(nodes.Count);

        for (int pos = 0; pos < walk.Length; pos++)
        {
            var nodeIdx = walk[pos];
            var node = nodes[nodeIdx];

            // Model input tensors are allocated but not counted until first use.
            if (node.IsModelInput())
            {
                nodeDetails.Add(new NodeEvaluationInfo
                {
                    OpCode = node.OpCode,
                    NodeIndex = nodeIdx,
                    ComputeTime = 0,
                    ExtraMemoryBytes = 0,
                    CurrentMemoryBytes = live.CurrentBytes,
                    CumulativeComputeTime = cumulativeComputeTime
                });
                continue;
            }

            var nodeInputs = node.Inputs;
            var nodeOutputs = node.Outputs;

            // Step 1: Load input tensors into memory if not already loaded
            foreach (var input in nodeInputs)
            {
                if (input is null || live.Contains(input.Value)) continue;
                var info = shapeInfo.GetTensorInfo(input.Value);
                if (info is not null)
                    live.Allocate(input.Value, info.MemoryBytes);
            }

            // Step 2: Compute op performance
            var perfInput = BuildOpPerfInput(node, pos, shapeInfo, tensorLastUse);
            var perfResult = _perfRegistry.Estimate(perfInput);

            var peakDuringOp = live.CurrentBytes + perfResult.ExtraMemoryBytes;

            // Step 3: Add output tensor memory (accounting for in-place reuse)
            var inPlaceReuse = perfResult.InPlaceBufferReuse;

            for (int outIdx = 0; outIdx < nodeOutputs.Count; outIdx++)
            {
                var output = nodeOutputs[outIdx];
                if (output is null || live.Contains(output.Value)) continue;

                var outputInfo = shapeInfo.GetTensorInfo(output.Value);
                if (outputInfo is null) continue;

                FastTensorKey? reused = null;
                if (inPlaceReuse.TryGetValue(outIdx, out var reusedInputIdx)
                    && reusedInputIdx >= 0 && reusedInputIdx < nodeInputs.Count
                    && nodeInputs[reusedInputIdx] is FastTensorKey candidate
                    && live.Contains(candidate))
                {
                    // A view (the input stays intact) always shares; an in-place write may only
                    // land in a buffer no other live key still reads.
                    var inputStaysLive = tensorLastUse.TryGetValue(candidate, out var last) && last > pos;
                    if (inputStaysLive || live.AliasCount(candidate) == 1)
                        reused = candidate;
                }

                if (reused is FastTensorKey source)
                    live.Alias(source, output.Value, outputInfo.MemoryBytes);
                else
                    live.Allocate(output.Value, outputInfo.MemoryBytes);
            }

            var peakAfterOutputs = live.CurrentBytes + perfResult.ExtraMemoryBytes;
            peakDuringOp = System.Math.Max(peakDuringOp, peakAfterOutputs);
            if (peakDuringOp > peakMemoryBytes)
                peakMemoryBytes = peakDuringOp;

            // Step 4: Free input tensors that are no longer needed
            foreach (var input in nodeInputs)
            {
                if (input is null || !live.Contains(input.Value)) continue;
                if (tensorLastUse.TryGetValue(input.Value, out var lastPos) && lastPos <= pos)
                    live.Release(input.Value);
            }

            // Step 5: Free outputs nothing will ever read
            foreach (var output in nodeOutputs)
            {
                if (output is null || !live.Contains(output.Value)) continue;
                if (!tensorLastUse.ContainsKey(output.Value) && !graphOutputs.Contains(output.Value))
                    live.Release(output.Value);
            }

            if (live.CurrentBytes > peakMemoryBytes)
                peakMemoryBytes = live.CurrentBytes;

            cumulativeComputeTime += perfResult.ComputeTime;

            nodeDetails.Add(new NodeEvaluationInfo
            {
                OpCode = node.OpCode,
                NodeIndex = nodeIdx,
                ComputeTime = perfResult.ComputeTime,
                ExtraMemoryBytes = perfResult.ExtraMemoryBytes,
                CurrentMemoryBytes = live.CurrentBytes,
                CumulativeComputeTime = cumulativeComputeTime
            });
        }

        return new GraphEvaluationResult
        {
            TotalComputeTime = cumulativeComputeTime,
            PeakMemoryBytes = peakMemoryBytes,
            OrderFidelity = OrtExecutionOrder.Fidelity(nodes, ortWalk),
            NodeDetails = nodeDetails,
        };
    }

    /// <summary>
    /// The live tensors as shared buffers: several keys may alias one buffer (a reshape of a
    /// tensor that is also read directly, an in-place output), which is charged once and
    /// released when its last alias is.
    /// </summary>
    private sealed class LiveBuffers
    {
        private readonly Dictionary<FastTensorKey, int> _bufferOf = new();
        private readonly Dictionary<int, (long Bytes, int Aliases)> _buffers = new();
        private int _nextId;

        public long CurrentBytes { get; private set; }

        public bool Contains(FastTensorKey key) => _bufferOf.ContainsKey(key);

        public int AliasCount(FastTensorKey key) => _buffers[_bufferOf[key]].Aliases;

        public void Allocate(FastTensorKey key, long bytes)
        {
            var id = _nextId++;
            _buffers[id] = (bytes, 1);
            _bufferOf[key] = id;
            CurrentBytes += bytes;
        }

        public void Alias(FastTensorKey source, FastTensorKey alias, long aliasBytes)
        {
            var id = _bufferOf[source];
            var (bytes, aliases) = _buffers[id];
            if (aliasBytes > bytes)
            {
                CurrentBytes += aliasBytes - bytes;
                bytes = aliasBytes;
            }
            _buffers[id] = (bytes, aliases + 1);
            _bufferOf[alias] = id;
        }

        public void Release(FastTensorKey key)
        {
            var id = _bufferOf[key];
            _bufferOf.Remove(key);
            var (bytes, aliases) = _buffers[id];
            if (aliases == 1)
            {
                _buffers.Remove(id);
                CurrentBytes -= bytes;
            }
            else
            {
                _buffers[id] = (bytes, aliases - 1);
            }
        }
    }

    /// <summary>
    /// Builds a map of tensor key → last walk position where that tensor is used as input.
    /// </summary>
    private static Dictionary<FastTensorKey, int> BuildTensorLastUse(IList<FastNode> nodes, int[] walk)
    {
        var lastUse = new Dictionary<FastTensorKey, int>();
        for (int pos = 0; pos < walk.Length; pos++)
        {
            foreach (var input in nodes[walk[pos]].Inputs)
            {
                if (input is not null)
                    lastUse[input.Value] = pos;
            }
        }
        return lastUse;
    }

    /// <summary>
    /// Builds the OpPerfInput for a given node at walk position <paramref name="pos"/>.
    /// </summary>
    private static OpPerfInput BuildOpPerfInput(
        FastNode node,
        int pos,
        ShapeInferenceResult shapeInfo,
        Dictionary<FastTensorKey, int> tensorLastUse)
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
            if (tensorLastUse.TryGetValue(input.Value, out var lastPos))
                inputMustRemainIntact[i] = lastPos > pos;
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
