using System.Collections.Immutable;
using Shorokoo.Core.Graph;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Onnx;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Nodes.Processors.AutoGrad;
using Shorokoo.Core.Nodes.Processors.Fast;
using Shorokoo.Core.Interpreter.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Lowering;

namespace Shorokoo.Core.Interpreter;

/// <summary>
/// Interprets a Shorokoo <see cref="InternalComputationGraph"/> by walking it node-by-node and invoking
/// a dedicated operator for each node. Every tensor produced by the graph is kept until the
/// end — nothing is evicted — so callers have full visibility into every intermediate tensor's
/// shape/dtype and, when small, its values.
///
/// Design highlights:
///   - Every tensor is represented as an <see cref="IRuntimeTensor"/> — plain
///     <see cref="RuntimeTensor"/>, <see cref="RuntimeOptionalTensor"/> for ONNX optionals, or
///     <see cref="RuntimeSequenceTensor"/> for ONNX tensor sequences.
///   - Concrete values are only stored for tensors with at most <see cref="MaxDataElements"/>
///     elements. All larger tensors keep only shape information.
///   - Operators live as standalone classes under <c>Ops/</c>, one per op code, auto-discovered
///     via <see cref="OpRegistry"/>. An op code this engine cannot compute may still be run, out
///     of ones it can: every op code on <see cref="LoweredOpCodes"/> is rewritten by
///     <see cref="FastLowerRegisteredOps"/> into its registered <see cref="OpLoweringRegistry"/>
///     decomposition, on a private copy of the graph, before the walk starts.
///   - <c>If</c> is supported by recursing into its subgraph when the condition value is known
///     and merging both branches' shapes when it is not.
///   - <c>Loop</c> is executed as a real iteration: the engine walks the body, then the close
///     op decides whether to loop again. On a loop-back, the close op's results are mapped onto
///     the open node's outputs (the in-body loop vars) and the engine jumps back to the node
///     after the open node. Tensors produced inside a loop accumulate per-iteration history.
///   - An instance carries no mutable state: everything a walk accumulates lives in the
///     <see cref="QuickRunState"/> that each <c>Run</c> creates for itself. One engine is
///     therefore reusable across runs — including after a run that gave up part-way — and
///     safe to share between threads.
///   - It consumes nothing. A tensor fed as it is is read, not taken as a compute context's run
///     would take it, and a <see cref="SharedInput"/> is read as it is, whatever its mode: this is
///     a reference evaluator, not a run, so it takes ownership of nothing it is given and every
///     input is the caller's, alive and unchanged, when it returns.
/// </summary>
public sealed class QuickExecutionEngine
{
    /// <summary>
    /// Size threshold: tensors with more than this many elements keep only shape/dtype
    /// information. Tensors at or below this count have their actual values filled in.
    /// </summary>
    public const int DefaultMaxDataElements = 256;

    /// <summary>
    /// Instance-visible accessor for the element threshold.
    /// </summary>
    public int MaxDataElements { get; init; } = DefaultMaxDataElements;

    /// <summary>
    /// Readonly-graph overload of <see cref="Run(InternalComputationGraph, TensorData[])"/>.
    /// Requires a concretized graph and interprets a private copy of it.
    /// </summary>
    public Dictionary<FastTensorKey, IRuntimeTensor> Run(Shorokoo.Graph.ComputationGraph graph, params TensorData[] sampleInputs)
    {
        graph.RequireConcretized("QuickExecutionEngine.Run");
        return Run(graph.ToInternal(), sampleInputs);
    }

    /// <summary>Readonly-graph overload of <see cref="Run(InternalComputationGraph, IData[])"/>.</summary>
    public Dictionary<FastTensorKey, IRuntimeTensor> Run(Shorokoo.Graph.ComputationGraph graph, params IData[] inputs)
    {
        graph.RequireConcretized("QuickExecutionEngine.Run");
        return Run(graph.ToInternal(), inputs);
    }

    /// <summary>Readonly-graph overload of <see cref="Execute(InternalComputationGraph, IData[])"/>.</summary>
    public IData[] Execute(Shorokoo.Graph.ComputationGraph graph, params IData[] inputs)
    {
        graph.RequireConcretized("QuickExecutionEngine.Execute");
        return Execute(graph.ToInternal(), inputs);
    }

    /// <summary>
    /// Convenience overload that takes <see cref="TensorData"/> samples for each graph input in
    /// order. Each sample is converted into a <see cref="RuntimeTensor"/> (respecting the size
    /// threshold) and bound to the matching graph input.
    /// </summary>
    public Dictionary<FastTensorKey, IRuntimeTensor> Run(InternalComputationGraph graph, params TensorData[] sampleInputs)
    {
        var graphInputs = graph.Inputs;
        if (sampleInputs.Length != graphInputs.Count)
            throw new ArgumentException(
                $"Expected {graphInputs.Count} sample inputs but got {sampleInputs.Length}.");

        var initial = new Dictionary<FastTensorKey, IRuntimeTensor>();
        for (int i = 0; i < graphInputs.Count; i++)
            initial[graphInputs[i]] = TensorDataConverter.ToRuntimeTensor(sampleInputs[i], MaxDataElements);
        return Run(graph, initial);
    }

    /// <summary>
    /// Convenience overload taking input <see cref="IData"/> values (plain <see cref="TensorData"/>
    /// and/or <see cref="OptionalTensorData"/>) for each graph input in order, binding each to the
    /// matching graph input.
    /// </summary>
    public Dictionary<FastTensorKey, IRuntimeTensor> Run(InternalComputationGraph graph, params IData[] inputs)
    {
        var graphInputs = graph.Inputs;
        if (inputs.Length != graphInputs.Count)
            throw new ArgumentException($"Expected {graphInputs.Count} inputs but got {inputs.Length}.");

        var initial = new Dictionary<FastTensorKey, IRuntimeTensor>();
        for (int i = 0; i < graphInputs.Count; i++)
            initial[graphInputs[i]] = TensorDataConverter.ToRuntimeInput(inputs[i], MaxDataElements);
        return Run(graph, initial);
    }

    /// <summary>
    /// Executes the graph with the ordered input <see cref="IData"/> values and returns each graph
    /// output as an <see cref="IData"/> — a <see cref="TensorData"/>, or an
    /// <see cref="OptionalTensorData"/> for an OptionalTensor output. Optional inputs are supplied as
    /// <see cref="OptionalTensorData"/> (present or absent); this is the optional-aware,
    /// pure-managed counterpart of the ONNX-Runtime <c>ComputeContext.Execute</c> path.
    /// </summary>
    public IData[] Execute(InternalComputationGraph graph, params IData[] inputs)
    {
        var store = Run(graph, inputs);
        var outputs = new IData[graph.Outputs.Count];
        for (int i = 0; i < graph.Outputs.Count; i++)
        {
            if (!store.TryGetValue(graph.Outputs[i], out var rt))
                throw new InvalidTensorOperationException(ErrorCodes.CR006, "Execute", $"output #{i}",
                    "Graph output was not produced by execution");
            outputs[i] = TensorDataConverter.ToOutputData(rt)
                ?? throw new InvalidTensorOperationException(ErrorCodes.CR006, "Execute", $"output #{i}",
                    "Graph output has no concrete data (raise MaxDataElements, or check the inputs)");
        }
        return outputs;
    }

    /// <summary>
    /// Runs the engine across the given <see cref="InternalComputationGraph"/>: walks nodes in topo
    /// order, dispatches each to its <see cref="OpRegistry"/>-registered operator, and returns
    /// the populated tensor store keyed by <see cref="FastTensorKey"/>.
    /// </summary>
    public Dictionary<FastTensorKey, IRuntimeTensor> Run(
        InternalComputationGraph graph,
        Dictionary<FastTensorKey, IRuntimeTensor>? initialInputs = null)
    {
        var store = new Dictionary<FastTensorKey, IRuntimeTensor>();
        if (initialInputs is not null)
            foreach (var kvp in initialInputs)
                store[kvp.Key] = kvp.Value;

        // An operator this engine cannot compute but has a registered lowering for is rewritten
        // into the operators it can, on a copy this run owns. The copy is not hygiene: a caller
        // may hand over a graph whose FastNodes are its own live nodes — the constant folder
        // assembles one out of the nodes it is folding — and lowering in place would rewrite that
        // caller's graph. Cloning preserves every key, so the store this returns is keyed exactly
        // as the caller's graph is, and the caller's Softsign stays a Softsign.
        if (FastLowerRegisteredOps.HasLowerableOp(graph, LoweredOpCodes))
        {
            graph = graph.Clone();
            FastLowerRegisteredOps.Process(graph, LoweredOpCodes);
        }

        var nodeByKey = FastProcessorHelper.BuildNodeByKey(graph);
        var state = new QuickRunState();
        var nodes = graph.Nodes;
        int i = 0;
        while (i < nodes.Count)
        {
            var node = nodes[i];
            var nextIndex = ProcessNode(node, i, graph, nodeByKey, store, state);
            i = nextIndex ?? i + 1;
        }

        return store;
    }

    /// <summary>
    /// What this engine lowers before it walks a graph: the operators it cannot compute itself.
    /// <c>Softsign</c> and <c>TensorScatter</c> are both here because no <see cref="QuickOp"/>
    /// implements either — TensorScatter's used to, but only well enough to infer a shape, and a
    /// kernel that produces no values leaves the engine as unable to compute the operator as no
    /// kernel at all. The list is stated rather than derived from <see cref="OpRegistry"/> for
    /// exactly that case — a kernel's presence does not mean the kernel computes the operator.
    /// </summary>
    internal static readonly ImmutableHashSet<string> LoweredOpCodes =
        ImmutableHashSet.Create(StringComparer.Ordinal, OpCodes.SOFTSIGN, OpCodes.TENSOR_SCATTER);

    /// <summary>
    /// Processes one node and returns an optional next-node index. A null return means "proceed
    /// to <paramref name="nodeIndex"/> + 1" (the normal case). A non-null return means "jump to
    /// that index" — used by LOOP_CLOSE on a loop-back.
    /// </summary>
    internal int? ProcessNode(
        FastNode node, int nodeIndex, InternalComputationGraph graph,
        Dictionary<FastNodeKey, FastNode> nodeByKey,
        Dictionary<FastTensorKey, IRuntimeTensor> store,
        QuickRunState state)
    {
        var outputKeys = node.Outputs;

        if (IsFastModelInputOpCode(node.OpCode))
        {
            var outKey = outputKeys.FirstOrDefault(k => k is not null);
            if (outKey is not null && !store.ContainsKey(outKey.Value))
            {
                var dtype = node.Attributes.GetDTypeVal(OnnxOpAttributeNames.AttrDtype) ?? DType.Invalid;
                store[outKey.Value] = RuntimeTensorFactory.Create(dtype, null);
            }
            return null;
        }

        if (node.OpCode == InternalOpCodes.MODEL_PARAM_DATA)
        {
            var outKey = outputKeys.FirstOrDefault(k => k is not null);
            if (outKey is null) return null;
            var data = node.Attributes.GetAttributeVal(OnnxOpAttributeNames.ShrkAttrTensorData);
            store[outKey.Value] = data is not null
                ? TensorDataConverter.ToRuntimeTensor(data, MaxDataElements)
                : RuntimeTensorFactory.Create(DType.Invalid, null);
            return null;
        }

        if (node.OpCode == OpCodes.LOOP_OPEN)
            state.LoopStack.Add(new FastLoopFrame(node, nodeIndex));

        var op = OpRegistry.Get(node.OpCode);
        if (op is null)
        {
            WriteDeclaredOutputs(node, store);
            PopLoopFrame(node, state);
            return null;
        }

        IRuntimeTensor[] results;
        bool loopBack;
        try
        {
            (results, loopBack) = op.Execute(node, graph, nodeByKey, store, MaxDataElements, state);
        }
        catch
        {
            WriteDeclaredOutputs(node, store);
            PopLoopFrame(node, state);
            return null;
        }

        if (loopBack)
        {
            FastNode? openNode = null;
            if (node.GraphOpenNodeKey is FastNodeKey openKey && !openKey.IsEmpty)
                nodeByKey.TryGetValue(openKey, out openNode);

            if (openNode is null)
            {
                WriteDeclaredOutputs(node, store);
                PopLoopFrame(node, state);
                return null;
            }

            StoreResults(openNode.Outputs, results, store, state);

            var frameIdx = state.LoopStack.FindLastIndex(f => f.OpenNode.Key == openNode.Key);
            if (frameIdx < 0)
            {
                StoreResults(outputKeys, results, store, state);
                return null;
            }
            return state.LoopStack[frameIdx].OpenNodeIndex + 1;
        }

        PopLoopFrame(node, state);
        StoreResults(outputKeys, results, store, state);
        return null;
    }

    /// <summary>
    /// Drops the frame the paired <c>LOOP_OPEN</c> pushed, for a walk leaving a
    /// <c>LOOP_CLOSE</c> for good. Every exit from a close node except a loop-back has to do
    /// this, the ones that gave up included: a frame left behind tells
    /// <see cref="StoreResults"/> that every later node in the run was produced inside a loop,
    /// so their tensors pick up an iteration tag and a history they never had. A close node
    /// naming no open node is a malformed pair, and the innermost frame is then the only one
    /// it can have been closing.
    /// </summary>
    private static void PopLoopFrame(FastNode node, QuickRunState state)
    {
        if (node.OpCode != OpCodes.LOOP_CLOSE || state.LoopStack.Count == 0) return;

        var frameIdx = node.GraphOpenNodeKey is FastNodeKey ck && !ck.IsEmpty
            ? state.LoopStack.FindLastIndex(f => f.OpenNode.Key == ck)
            : state.LoopStack.Count - 1;
        if (frameIdx >= 0) state.LoopStack.RemoveAt(frameIdx);
    }

    private static void StoreResults(
        System.Collections.Generic.List<FastTensorKey?> outputKeys,
        IRuntimeTensor[] results,
        Dictionary<FastTensorKey, IRuntimeTensor> store,
        QuickRunState state)
    {
        ImmutableArray<long>? iterTemplate = state.LoopStack.Count > 0
            ? ImmutableArray.Create(new long[state.LoopStack.Count])
            : null;

        for (int i = 0; i < outputKeys.Count; i++)
        {
            var k = outputKeys[i];
            if (k is null) continue;

            IRuntimeTensor rt = i < results.Length && results[i] is not null
                ? results[i]
                : new RuntimeTensor { DType = DType.Invalid };

            var newIter = iterTemplate;
            ImmutableArray<IRuntimeTensor>? newHistory = null;
            bool setHistory = false;
            if (state.LoopStack.Count > 0 && store.TryGetValue(k.Value, out var prior))
            {
                newHistory = prior.History is { } h ? h.Add(prior) : ImmutableArray.Create(prior);
                setHistory = true;
            }

            rt = rt switch
            {
                RuntimeTensor r => r with
                {
                    IterationIndices = newIter,
                    History = setHistory ? newHistory : r.History,
                },
                RuntimeOptionalTensor o => o with
                {
                    IterationIndices = newIter,
                    History = setHistory ? newHistory : o.History,
                },
                RuntimeSequenceTensor s => s with
                {
                    IterationIndices = newIter,
                    History = setHistory ? newHistory : s.History,
                },
                _ => rt,
            };

            store[k.Value] = rt;
        }
    }

    /// <summary>
    /// InternalComputationGraph placeholder writer for ops that couldn't run: the tensors we produce
    /// have no known dtype or rank (this info used to come from the graph's TensorInfos side
    /// dictionary, but QEE no longer reads from it). Downstream ops that tolerate invalid
    /// dtype/shape inputs will keep progressing; those that don't will fall back into this same
    /// placeholder path.
    /// </summary>
    private static void WriteDeclaredOutputs(
        FastNode node,
        Dictionary<FastTensorKey, IRuntimeTensor> store)
    {
        foreach (var k in node.Outputs)
        {
            if (k is null) continue;
            if (store.ContainsKey(k.Value)) continue;
            store[k.Value] = new RuntimeTensor { DType = DType.Invalid };
        }
    }

    private static bool IsFastModelInputOpCode(string opCode) =>
        opCode == InternalOpCodes.MODEL_TENSOR_INPUT ||
        opCode == InternalOpCodes.MODEL_OPTIONAL_INPUT ||
        opCode == InternalOpCodes.MODEL_SEQUENCE_INPUT ||
        opCode == InternalOpCodes.MODEL_TENSORSTRUCT_INPUT ||
        opCode == InternalOpCodes.GENERIC_TYPE_INPUT;
}
