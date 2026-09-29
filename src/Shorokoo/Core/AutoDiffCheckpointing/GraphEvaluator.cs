using Shorokoo.Core.AutoDiffCheckpointing.OpsPerf;
using Shorokoo.Core.Backends;
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
/// The state a graph carries from one run to the next — a training step's parameters, model state
/// and optimizer state — as <see cref="GraphEvaluator"/> charges it.
///
/// <para>Each pair names, by position, an output that is the updated value of a state field and
/// the input it replaces. The caller feeds each such input and keeps it until the run ends: ONNX
/// Runtime holds every fed input for the whole run, and a consumed one cannot come back part-way
/// through. So a state input is charged from the first position to the last, read or not. Where
/// <see cref="WrittenInPlace"/> holds and <see cref="OutputAliasProof"/> proves a pair over the
/// graph evaluated — and the two agree in shape and element type, as a backend requires before it
/// binds them — the output is written into the input's buffer (<see cref="OutputAlias"/>) and the
/// pair is charged that one buffer; any other pair holds both, the input for the run and the
/// output from its writer on. The proof is asked of each graph evaluated, so a rewrite that
/// breaks one — a reader of the input moved past the output's writer, a recomputation reading it
/// late — is charged the output it can no longer write in place.</para>
/// </summary>
/// <param name="Pairs">Each updated state output with the state input it replaces, by position:
/// output <c>Output</c> of the graph, input <c>Input</c>.</param>
/// <param name="WrittenInPlace">Whether the run writes a proven pair's output into its input's
/// memory: what the compute context the graph runs on does
/// (<see cref="Shorokoo.Runtime.ComputeContext.OutputAliasing"/>).</param>
internal sealed record StepState(IReadOnlyList<(int Output, int Input)> Pairs, bool WrittenInPlace);

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
/// - The state a step carries (<see cref="StepState"/>) is held for the whole run, and an updated
///   state output the step provably writes over the state it replaces shares that state's buffer.
///
/// Uses ShapeInference data and per-op performance models to produce estimates.
/// </summary>
internal class GraphEvaluator
{
    private readonly OpPerfRegistry _perfRegistry;
    private readonly bool _modelOrtBufferReuse;
    private readonly StepState? _state;

    /// <param name="perfRegistry">The per-op estimators to price nodes with; the default registry when null.</param>
    /// <param name="modelOrtBufferReuse">Model ORT's static buffer reuse (see <c>AllocationPlan</c>).
    /// False counts plain liveness — what an allocator that returned every dead buffer at once
    /// would need — which is a lower bound ORT's plan never reaches on a real training step.</param>
    /// <param name="state">The state every graph this evaluates carries across runs, charged as
    /// <see cref="StepState"/> describes; none when null. A pass that hands this evaluator to each
    /// of its strategies scores every candidate with the state it will run with, so a candidate
    /// that keeps a pair written in place outscores one that loses it.</param>
    public GraphEvaluator(OpPerfRegistry? perfRegistry = null, bool modelOrtBufferReuse = true, StepState? state = null)
    {
        _perfRegistry = perfRegistry ?? new OpPerfRegistry();
        _modelOrtBufferReuse = modelOrtBufferReuse;
        _state = state;
    }

    /// <summary>
    /// The state pairs of this evaluator's <see cref="StepState"/> that <paramref name="graph"/>
    /// writes in place, by position: those <see cref="OutputAliasProof"/> proves over it whose
    /// output and input <paramref name="shapeInfo"/> gives one shape and element type. None where
    /// the evaluator carries no state or the state is not written in place.
    /// </summary>
    internal IReadOnlyList<(int Output, int Input)> StatePairsWrittenInPlace(
        InternalComputationGraph graph, ShapeInferenceResult shapeInfo)
    {
        if (_state is not { WrittenInPlace: true, Pairs.Count: > 0 }) return [];
        var inputs = graph.Inputs;
        var outputs = graph.Outputs;
        var inPlace = new List<(int Output, int Input)>();
        foreach (var pair in OutputAliasProof.Prove(graph, _state.Pairs))
            if (shapeInfo.GetTensorInfo(outputs[pair.Output]) is { } output
                && shapeInfo.GetTensorInfo(inputs[pair.Input]) is { } input
                && output.MemoryBytes == input.MemoryBytes
                && AllocationPlan.ShapeKey(output) == AllocationPlan.ShapeKey(input))
                inPlace.Add(pair);
        return inPlace;
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
        // The output nodes run nothing and hold nothing: a graph output is never released anyway.
        var walk = order == EvaluationOrder.OrtOrder
            ? ortWalk
            : Enumerable.Range(0, graph.BodyEnd).ToArray();

        // Tensor key → last walk position that reads it.
        var tensorLastUse = BuildTensorLastUse(nodes, walk);
        var consumerOpCodes = BuildConsumerOpCodes(nodes);
        var graphOutputs = new HashSet<FastTensorKey>(graph.Outputs);
        // Never recycled into and never written in place: the fed inputs, and the initializers
        // ORT keeps (constants, parameter data). A state input is held for the whole run, as ORT
        // holds it (see StepState); every other one's buffer is released at its last read, so the
        // model charges it only while it is read, not for the whole run as ORT in fact does. That
        // understates a step whose other inputs and initializers are large next to its
        // activations, and is the next fidelity gap to close after the two the benchmark names.
        var inputs = graph.Inputs;
        var graphInputs = new HashSet<FastTensorKey>(inputs);
        foreach (var node in nodes)
            if (node.IsModelInput() || node.IsModelParamData() || node.OpCode == Shorokoo.Core.Nodes.NodeDefinitions.OpCodes.CONSTANT)
                foreach (var output in node.Outputs)
                    if (output is not null) graphInputs.Add(output.Value);

        var plan = new AllocationPlan(graphOutputs, graphInputs, _modelOrtBufferReuse);

        // The state: each input held from the first position, and each output its step writes in
        // place bound to the buffer of the input it replaces, to be placed there by its writer.
        var heldBuffer = new Dictionary<FastTensorKey, int>();
        var writtenInPlace = new Dictionary<FastTensorKey, int>();
        var statePairsInPlace = new List<(int Output, int Input)>();
        if (_state is not null)
        {
            var outputs = graph.Outputs;
            foreach (var (output, input) in _state.Pairs)
                if (input >= 0 && input < inputs.Count && output >= 0 && output < outputs.Count
                    && !heldBuffer.ContainsKey(inputs[input])
                    && shapeInfo.GetTensorInfo(inputs[input]) is { } info)
                    heldBuffer[inputs[input]] = plan.Hold(inputs[input], info);
            foreach (var (output, input) in StatePairsWrittenInPlace(graph, shapeInfo))
                if (heldBuffer.TryGetValue(inputs[input], out var buffer))
                {
                    writtenInPlace[outputs[output]] = buffer;
                    statePairsInPlace.Add((output, input));
                }
        }
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

                if (writtenInPlace.TryGetValue(output.Value, out var stateBuffer))
                {
                    plan.Place(stateBuffer, output.Value, outputInfo.MemoryBytes);
                    continue;
                }

                FastTensorKey? reused = null;
                if (!readsMetadataOnly
                    && inPlaceReuse.TryGetValue(outIdx, out var reusedInputIdx)
                    && reusedInputIdx >= 0 && reusedInputIdx < nodeInputs.Count
                    && nodeInputs[reusedInputIdx] is FastTensorKey candidate
                    && plan.Contains(candidate))
                {
                    // A view (the input stays intact) always shares; an in-place write may only
                    // land in a buffer no other live key still reads, never in a fed input, and
                    // never into or out of a graph output — ORT gives an output its own buffer
                    // and never overwrites one (measured: 127 MB against the 64 MB in-place would
                    // read, either way round).
                    var inputStaysLive = tensorLastUse.TryGetValue(candidate, out var last) && last > pos;
                    var touchesGraphOutput = graphOutputs.Contains(output.Value) || plan.IsGraphOutput(candidate);
                    if (!touchesGraphOutput && (inputStaysLive || (plan.AliasCount(candidate) == 1 && !plan.IsGraphInput(candidate))))
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
            StateWrittenInPlace = statePairsInPlace,
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
    ///
    /// <para>A held buffer — a state input (<see cref="StepState"/>) — is occupied from the first
    /// position to the last whoever reads it, and the state output written in place is placed in
    /// it rather than given one of its own.</para>
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
            public bool IsHeld;
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

        public bool IsGraphOutput(FastTensorKey key) => _buffers[_bufferOf[key]].IsGraphOutput;

        public static string ShapeKey(TensorShapeInfo info)
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
            => Place(_bufferOf[source], alias, aliasBytes);

        /// <summary>A fed input's buffer, held from the first position to the last; its id.</summary>
        public int Hold(FastTensorKey key, TensorShapeInfo info)
        {
            _buffers.Add(new Buffer
            {
                Bytes = info.MemoryBytes,
                Shape = ShapeKey(info),
                Aliases = 1,
                Start = 0,
                IsGraphInput = true,
                IsGraphOutput = _graphOutputs.Contains(key),
                IsHeld = true,
            });
            _bufferOf[key] = _buffers.Count - 1;
            return _buffers.Count - 1;
        }

        /// <summary>Puts <paramref name="key"/> in buffer <paramref name="id"/>, whether or not
        /// any key still names it.</summary>
        public void Place(int id, FastTensorKey key, long bytes)
        {
            var buffer = _buffers[id];
            if (bytes > buffer.Bytes) buffer.Bytes = bytes;
            buffer.Aliases++;
            _bufferOf[key] = id;
        }

        public void Release(FastTensorKey key, int pos)
        {
            var id = _bufferOf[key];
            _bufferOf.Remove(key);
            var buffer = _buffers[id];
            buffer.Aliases--;
            if (buffer.Aliases > 0 || buffer.IsGraphOutput || buffer.IsHeld) return;
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
        foreach (var node in nodes.Where(n => !Shorokoo.Core.Nodes.NodeDefinitions.InternalOpCodes.IsGraphOutputOp(n.OpCode)))
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
