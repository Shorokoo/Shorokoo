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
/// Memory tracking follows ORT's allocation plan (see <c>AllocationPlan</c>):
/// - A tensor is not loaded into memory until it first appears as an input to a node.
/// - Once a tensor is used and no subsequent node requires it, its buffer is dead — but it is
///   returned only if no later output of the identical shape takes it over; until then it
///   stays occupied. An output nothing consumes (and that is not a graph output) dies at the
///   node that produced it.
/// - In-place buffer reuse is modelled as a shared buffer: the output aliases the input's
///   buffer, which is charged once. An op may only write in place when no other live key
///   shares that buffer, and never into a fed input.
/// - Shape/Size read metadata only; ORT folds them under static shapes, so they neither load
///   nor hold their input.
///
/// Uses ShapeInference data and per-op performance models to produce estimates.
/// </summary>
internal class GraphEvaluator
{
    private readonly OpPerfRegistry _perfRegistry;
    private readonly bool _modelOrtBufferReuse;

    /// <param name="perfRegistry">The per-op estimators to price nodes with; the default registry when null.</param>
    /// <param name="modelOrtBufferReuse">Model ORT's static buffer reuse (see <c>AllocationPlan</c>).
    /// False counts plain liveness — what an allocator that returned every dead buffer at once
    /// would need — which is a lower bound ORT's plan never reaches on a real training step.</param>
    public GraphEvaluator(OpPerfRegistry? perfRegistry = null, bool modelOrtBufferReuse = true)
    {
        _perfRegistry = perfRegistry ?? new OpPerfRegistry();
        _modelOrtBufferReuse = modelOrtBufferReuse;
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
        var consumerOpCodes = BuildConsumerOpCodes(nodes);
        var graphOutputs = new HashSet<FastTensorKey>(graph.Outputs);
        var graphInputs = new HashSet<FastTensorKey>(graph.Inputs);
        foreach (var node in nodes)
            if (node.IsModelInput())
                foreach (var output in node.Outputs)
                    if (output is not null) graphInputs.Add(output.Value);

        var plan = new AllocationPlan(graphOutputs, graphInputs, _modelOrtBufferReuse);
        var extraAtPos = new long[walk.Length];
        var computeAtPos = new double[walk.Length];
        var opCodeAtPos = new string[walk.Length];
        double cumulativeComputeTime = 0;

        for (int pos = 0; pos < walk.Length; pos++)
        {
            var nodeIdx = walk[pos];
            var node = nodes[nodeIdx];
            opCodeAtPos[pos] = node.OpCode;

            // Model input tensors are allocated but not counted until first use.
            if (node.IsModelInput())
                continue;

            var nodeInputs = node.Inputs;
            var nodeOutputs = node.Outputs;
            var readsMetadataOnly = IsMetadataOnly(node);

            // Step 1: Load input tensors into memory if not already loaded. A metadata-only
            // reader (Shape/Size) is folded away by ORT under static shapes, so it neither loads
            // nor holds its input.
            if (!readsMetadataOnly)
                foreach (var input in nodeInputs)
                {
                    if (input is null || plan.Contains(input.Value)) continue;
                    var info = shapeInfo.GetTensorInfo(input.Value);
                    if (info is not null)
                        plan.Allocate(input.Value, info, pos);
                }

            // Step 2: Compute op performance
            var perfInput = BuildOpPerfInput(node, pos, shapeInfo, tensorLastUse, consumerOpCodes);
            var perfResult = _perfRegistry.Estimate(perfInput);
            extraAtPos[pos] = perfResult.ExtraMemoryBytes;

            // Step 3: Add output tensor memory (accounting for in-place reuse)
            var inPlaceReuse = perfResult.InPlaceBufferReuse;

            for (int outIdx = 0; outIdx < nodeOutputs.Count; outIdx++)
            {
                var output = nodeOutputs[outIdx];
                if (output is null || plan.Contains(output.Value)) continue;

                var outputInfo = shapeInfo.GetTensorInfo(output.Value);
                if (outputInfo is null) continue;

                FastTensorKey? reused = null;
                if (!readsMetadataOnly
                    && inPlaceReuse.TryGetValue(outIdx, out var reusedInputIdx)
                    && reusedInputIdx >= 0 && reusedInputIdx < nodeInputs.Count
                    && nodeInputs[reusedInputIdx] is FastTensorKey candidate
                    && plan.Contains(candidate))
                {
                    // A view (the input stays intact) always shares; an in-place write may only
                    // land in a buffer no other live key still reads, and never in a fed input.
                    var inputStaysLive = tensorLastUse.TryGetValue(candidate, out var last) && last > pos;
                    if (inputStaysLive || (plan.AliasCount(candidate) == 1 && !plan.IsGraphInput(candidate)))
                        reused = candidate;
                }

                if (reused is FastTensorKey source)
                    plan.Alias(source, output.Value, outputInfo.MemoryBytes);
                else
                    plan.Allocate(output.Value, outputInfo, pos);
            }

            // Step 4: Free input tensors that are no longer needed
            if (!readsMetadataOnly)
                foreach (var input in nodeInputs)
                {
                    if (input is null || !plan.Contains(input.Value)) continue;
                    if (tensorLastUse.TryGetValue(input.Value, out var lastPos) && lastPos <= pos)
                        plan.Release(input.Value, pos);
                }

            // Step 5: Free outputs nothing will ever read
            foreach (var output in nodeOutputs)
            {
                if (output is null || !plan.Contains(output.Value)) continue;
                if (!tensorLastUse.ContainsKey(output.Value))
                    plan.Release(output.Value, pos);
            }

            cumulativeComputeTime += perfResult.ComputeTime;
            computeAtPos[pos] = perfResult.ComputeTime;
        }

        // The plan is only known once the walk is complete: a buffer handed down a reuse
        // chain is occupied from its first allocation to the death of the chain's last value.
        var occupancy = plan.Occupancy(walk.Length);
        long peakMemoryBytes = 0;
        var nodeDetails = new List<NodeEvaluationInfo>(walk.Length);
        double cumulative = 0;
        for (int pos = 0; pos < walk.Length; pos++)
        {
            var during = occupancy[pos] + extraAtPos[pos];
            if (during > peakMemoryBytes) peakMemoryBytes = during;
            cumulative += computeAtPos[pos];
            nodeDetails.Add(new NodeEvaluationInfo
            {
                OpCode = opCodeAtPos[pos],
                NodeIndex = walk[pos],
                ComputeTime = computeAtPos[pos],
                ExtraMemoryBytes = extraAtPos[pos],
                CurrentMemoryBytes = occupancy[pos],
                CumulativeComputeTime = cumulative,
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

    /// <summary>Reads only its input's metadata; ORT folds it away under static shapes.</summary>
    private static bool IsMetadataOnly(FastNode node)
        => node.OpCode is Shorokoo.Core.Nodes.NodeDefinitions.OpCodes.SHAPE or Shorokoo.Core.Nodes.NodeDefinitions.OpCodes.SIZE;

    /// <summary>
    /// ORT's memory plan for a sequential run, which is what decides how much is allocated.
    ///
    /// <para>Several keys may alias one buffer (a reshape of a tensor that is also read
    /// directly, an in-place output); the buffer is charged once. When its last alias dies the
    /// buffer is not returned: ORT's planner puts it on a free list and hands it to the next
    /// output of the <b>identical static shape and type</b> (most recently freed first), and it
    /// stays occupied across the gap — a buffer is released only when the last value of its
    /// reuse chain dies, or at its own death if nothing ever reuses it. A fed input is never
    /// recycled and a graph output is never released. Measured against a run's actual
    /// mmap/munmap trace this reproduces ORT's allocation count and the size histogram at the
    /// peak to within one buffer; plain free-at-last-use undercounts the optimized graphs by
    /// a third, because an early free buys nothing until a same-shape tensor is born.</para>
    /// </summary>
    private sealed class AllocationPlan
    {
        private sealed class Buffer
        {
            public long Bytes;
            public string Shape = "";
            public int Aliases;
            public int Start;
            public int End = int.MaxValue;
            public bool IsGraphInput;
            public bool IsGraphOutput;
        }

        private readonly Dictionary<FastTensorKey, int> _bufferOf = new();
        private readonly List<Buffer> _buffers = new();
        private readonly List<int> _free = new();
        private readonly HashSet<FastTensorKey> _graphOutputs;
        private readonly HashSet<FastTensorKey> _graphInputs;
        private readonly bool _reuse;

        public AllocationPlan(HashSet<FastTensorKey> graphOutputs, HashSet<FastTensorKey> graphInputs, bool reuse)
        {
            _graphOutputs = graphOutputs;
            _graphInputs = graphInputs;
            _reuse = reuse;
        }

        public bool Contains(FastTensorKey key) => _bufferOf.ContainsKey(key);

        public int AliasCount(FastTensorKey key) => _buffers[_bufferOf[key]].Aliases;

        public bool IsGraphInput(FastTensorKey key) => _buffers[_bufferOf[key]].IsGraphInput;

        private static string ShapeKey(TensorShapeInfo info)
            => info.DType + "[" + string.Join(",", info.Shape.Dims) + "]";

        public void Allocate(FastTensorKey key, TensorShapeInfo info, int pos)
        {
            var isInput = _graphInputs.Contains(key);
            var shape = ShapeKey(info);
            if (!isInput)
            {
                for (int i = _free.Count - 1; i >= 0; i--)
                {
                    var id = _free[i];
                    var candidate = _buffers[id];
                    if (candidate.Bytes != info.MemoryBytes || candidate.Shape != shape) continue;
                    _free.RemoveAt(i);
                    candidate.Aliases = 1;
                    candidate.End = int.MaxValue;
                    candidate.IsGraphOutput = _graphOutputs.Contains(key);
                    _bufferOf[key] = id;
                    return;
                }
            }

            _buffers.Add(new Buffer
            {
                Bytes = info.MemoryBytes,
                Shape = shape,
                Aliases = 1,
                Start = pos,
                IsGraphInput = isInput,
                IsGraphOutput = _graphOutputs.Contains(key),
            });
            _bufferOf[key] = _buffers.Count - 1;
        }

        public void Alias(FastTensorKey source, FastTensorKey alias, long aliasBytes)
        {
            var id = _bufferOf[source];
            var buffer = _buffers[id];
            if (aliasBytes > buffer.Bytes) buffer.Bytes = aliasBytes;
            buffer.Aliases++;
            if (_graphOutputs.Contains(alias)) buffer.IsGraphOutput = true;
            _bufferOf[alias] = id;
        }

        public void Release(FastTensorKey key, int pos)
        {
            var id = _bufferOf[key];
            _bufferOf.Remove(key);
            var buffer = _buffers[id];
            buffer.Aliases--;
            if (buffer.Aliases > 0 || buffer.IsGraphOutput) return;
            buffer.End = pos;
            if (_reuse && !buffer.IsGraphInput) _free.Add(id);
        }

        /// <summary>Occupied bytes after each walk position.</summary>
        public long[] Occupancy(int positions)
        {
            var delta = new long[positions + 1];
            foreach (var buffer in _buffers)
            {
                var end = buffer.End == int.MaxValue ? positions - 1 : buffer.End;
                delta[buffer.Start] += buffer.Bytes;
                delta[end + 1] -= buffer.Bytes;
            }
            var occupancy = new long[positions];
            long current = 0;
            for (int pos = 0; pos < positions; pos++)
            {
                current += delta[pos];
                occupancy[pos] = current;
            }
            return occupancy;
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
            if (IsMetadataOnly(nodes[walk[pos]])) continue;
            foreach (var input in nodes[walk[pos]].Inputs)
            {
                if (input is not null)
                    lastUse[input.Value] = pos;
            }
        }
        return lastUse;
    }

    /// <summary>Tensor key → op codes of the nodes that read it (ORT fuses some producer/consumer pairs).</summary>
    private static Dictionary<FastTensorKey, List<string>> BuildConsumerOpCodes(IList<FastNode> nodes)
    {
        var consumers = new Dictionary<FastTensorKey, List<string>>();
        foreach (var node in nodes)
            foreach (var input in node.Inputs)
            {
                if (input is null) continue;
                if (!consumers.TryGetValue(input.Value, out var list))
                    consumers[input.Value] = list = new List<string>();
                list.Add(node.OpCode);
            }
        return consumers;
    }

    /// <summary>
    /// Builds the OpPerfInput for a given node at walk position <paramref name="pos"/>.
    /// </summary>
    private static OpPerfInput BuildOpPerfInput(
        FastNode node,
        int pos,
        ShapeInferenceResult shapeInfo,
        Dictionary<FastTensorKey, int> tensorLastUse,
        Dictionary<FastTensorKey, List<string>> consumerOpCodes)
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

        List<string>? consumers = null;
        foreach (var output in nodeOutputs)
            if (output is not null && consumerOpCodes.TryGetValue(output.Value, out var list))
                (consumers ??= new List<string>()).AddRange(list);

        return new OpPerfInput
        {
            InputShapes = inputShapes,
            OutputShapes = outputShapes,
            InputMustRemainIntact = inputMustRemainIntact,
            OpCode = node.OpCode,
            Attributes = attrs,
            ConsumerOpCodes = consumers ?? [],
        };
    }
}
