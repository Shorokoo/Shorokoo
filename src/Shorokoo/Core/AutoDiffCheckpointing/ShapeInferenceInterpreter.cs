using Shorokoo;
using Shorokoo.Core.Graph;
using Shorokoo.Runtime;
using Shorokoo.Core;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Training;
using Shorokoo.Modules;
using Shorokoo.Graph;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Nodes.Processors.AutoGrad;
using Shorokoo.Core.Nodes.Processors.Fast;
using Shorokoo.Core.Inference;
using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.NodeDefinitions;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;
using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;

namespace Shorokoo.Core.AutoDiffCheckpointing;

/// <summary>
/// Runs shape inference on a concrete <see cref="FastComputationGraph"/> by first
/// executing the graph through the <see cref="QuickExecutionEngine"/> (pure C#) and
/// then falling back to ONNX Runtime per-node execution only for tensors whose shape
/// QEE could not resolve.
///
/// <para>
/// QEE-first means the vast majority of graphs need zero ONNX Runtime sessions, which
/// avoids native segfaults on large graphs and is dramatically faster on small ones.
/// ORT remains available as a per-node fallback for ops QEE doesn't model.
/// </para>
///
/// <para>
/// The caller declares which tensor keys are actually needed via the
/// <c>requiredKeys</c> parameter on <see cref="Infer(FastComputationGraph, IReadOnlyCollection{FastTensorKey}?, TensorData[])"/>;
/// only those drive the ORT fallback. The convenience overload defaults to "every output
/// tensor of every node" so existing callers (which iterate the whole graph) continue
/// to receive shapes for everything.
/// </para>
/// </summary>
internal class ShapeInferenceInterpreter
{
    /// <summary>
    /// Element-count threshold used when materializing data into
    /// <see cref="TensorShapeInfo.Data"/>. Tensors at or below this size keep their
    /// concrete values; larger tensors carry shape/dtype only. The same threshold is
    /// passed to the underlying <see cref="QuickExecutionEngine"/>.
    /// </summary>
    public const int MaxSmallTensorElements = 1024;

    private readonly ComputeContext _computeContext;

    public ShapeInferenceInterpreter(ComputeContext? computeContext = null)
    {
        _computeContext = computeContext ?? new ComputeContext();
    }

    /// <summary>
    /// Convenience overload: infers shapes for every output tensor of every node in
    /// the graph. See <see cref="Infer(FastComputationGraph, IReadOnlyCollection{FastTensorKey}?, TensorData[])"/>
    /// for the explicit form.
    /// </summary>
    public ShapeInferenceResult Infer(FastComputationGraph graph, params TensorData[] sampleInputs)
        => Infer(graph, requiredKeys: null, sampleInputs);

    /// <summary>
    /// Runs shape inference on a concrete <see cref="FastComputationGraph"/> using the
    /// provided sample inputs.
    /// </summary>
    /// <param name="graph">A concrete computation graph (post-ToConcreteArchitecture/ToConcreteModel).</param>
    /// <param name="requiredKeys">
    /// Tensor keys the caller actually needs shapes for. Drives the ORT fallback —
    /// only keys that QEE couldn't resolve AND are required will trigger per-node
    /// ONNX Runtime execution. Pass <c>null</c> to require every output tensor of every node.
    /// </param>
    /// <param name="sampleInputs">Sample input tensor data, one per graph input, in order.</param>
    /// <returns>Shape inference results containing per-tensor shape information.</returns>
    public ShapeInferenceResult Infer(
        FastComputationGraph graph,
        IReadOnlyCollection<FastTensorKey>? requiredKeys,
        params TensorData[] sampleInputs)
    {
        var graphInputs = graph.Inputs;
        if (sampleInputs.Length != graphInputs.Count)
            throw new ArgumentException(
                $"Expected {graphInputs.Count} sample inputs but got {sampleInputs.Length}.");

        var tensorStore = new Dictionary<FastTensorKey, TensorShapeInfo>();

        // Step 1: QEE-based pure-C# execution. Covers ~all ONNX ops (138 op
        // implementations under Inference/QuickExecutionEngine/Ops), plus Shorokoo
        // internals like MODEL_PARAM_DATA. Produces shape, dtype, and small-tensor
        // values in one pass without spinning up ORT sessions.
        Dictionary<FastTensorKey, IRuntimeTensor> qeeStore;
        try
        {
            var qee = new QuickExecutionEngine { MaxDataElements = MaxSmallTensorElements };
            qeeStore = qee.Run(graph, sampleInputs);
        }
        catch
        {
            // QEE itself catches per-op exceptions, so reaching here means a structural
            // problem (e.g., bad initial-input mapping). Fall through with empty store —
            // ORT will pick up everything required.
            qeeStore = new Dictionary<FastTensorKey, IRuntimeTensor>();
        }

        foreach (var kvp in qeeStore)
        {
            if (TryConvertToShapeInfo(kvp.Value) is { } info)
                tensorStore[kvp.Key] = info;
        }

        // Step 2: identify required keys still missing after QEE and run the ORT
        // fallback only for the producing nodes. Topological order over graph.Nodes
        // guarantees each fallback node sees populated inputs.
        var required = requiredKeys ?? CollectAllOutputKeys(graph);
        HashSet<FastNodeKey>? nodesToFallback = null;
        foreach (var key in required)
        {
            if (tensorStore.ContainsKey(key)) continue;
            if (TryFindProducer(graph, key) is { } producer)
            {
                nodesToFallback ??= new HashSet<FastNodeKey>();
                nodesToFallback.Add(producer.Key);
            }
        }

        if (nodesToFallback is not null)
        {
            for (int i = 0; i < graph.Nodes.Count; i++)
            {
                var node = graph.Nodes[i];
                if (!nodesToFallback.Contains(node.Key)) continue;
                FallbackResolveNode(node, tensorStore);
            }
        }

        return new ShapeInferenceResult(tensorStore.ToImmutableDictionary());
    }

    /// <summary>
    /// Resolves a single node's outputs through ORT, used only when QEE failed to
    /// populate one of its required outputs. Subgraph-bearing nodes (LOOP/IF/SCAN)
    /// can't run as standalone mini-graphs, so we leave them unresolved rather than
    /// guessing — downstream callers can detect the missing key.
    /// </summary>
    private void FallbackResolveNode(FastNode node, Dictionary<FastTensorKey, TensorShapeInfo> tensorStore)
    {
        if (node.IsModelInput())
            return; // graph inputs should already be in tensorStore from QEE's initial map

        if (node.IsModelParamData())
        {
            ProcessModelParamData(node, tensorStore);
            return;
        }

        if (node.OpCode == OpCodes.CONSTANT)
        {
            ProcessConstant(node, tensorStore);
            return;
        }

        if (node.HasGraphAttribute())
            return; // can't ORT-execute a subgraph in isolation; leave unresolved

        ExecuteNode(node, tensorStore);
    }

    /// <summary>
    /// Converts a QEE <see cref="IRuntimeTensor"/> into a <see cref="TensorShapeInfo"/>.
    /// Returns null for sequence/optional tensors or invalid/unshaped tensors. Drops the
    /// data payload when QEE's stored data length doesn't match the shape's element count
    /// (which QEE allows for cases like broadcast-pending scalars).
    /// </summary>
    private static TensorShapeInfo? TryConvertToShapeInfo(IRuntimeTensor rt)
    {
        if (rt is not RuntimeTensor r) return null; // optional/sequence — no flat shape
        if (r.Shape is null) return null;
        if (r.DType == DType.Invalid) return null;

        var elementCount = r.Shape.Count;
        var dataMatchesShape =
            (r.FloatData is { } fd && fd.Length == elementCount) ||
            (r.IntData is { } id && id.Length == elementCount) ||
            (r.BoolData is { } bd && bd.Length == elementCount);
        var data = dataMatchesShape ? TensorDataConverter.ToTensorData(r) : null;
        return new TensorShapeInfo(r.Shape, r.DType, data);
    }

    private static List<FastTensorKey> CollectAllOutputKeys(FastComputationGraph graph)
    {
        var keys = new List<FastTensorKey>();
        foreach (var node in graph.Nodes)
            foreach (var k in node.Outputs)
                if (k is not null) keys.Add(k.Value);
        return keys;
    }

    private static FastNode? TryFindProducer(FastComputationGraph graph, FastTensorKey key)
    {
        // Linear scan; nothing in the optimization stack calls Infer in a hot loop so
        // building a producer map up front is unnecessary unless many keys are missing.
        foreach (var node in graph.Nodes)
            foreach (var k in node.Outputs)
                if (k is not null && k.Value.Equals(key))
                    return node;
        return null;
    }

    private void ProcessModelParamData(FastNode node, Dictionary<FastTensorKey, TensorShapeInfo> tensorStore)
    {
        var tensorData = node.Attributes.GetTensorVal(ShrkAttrTensorData);
        if (tensorData is null)
            return;

        var output = FirstOutputKey(node);
        if (output is null) return;
        StoreTensorInfo(tensorStore, output.Value, tensorData);
    }

    private void ProcessConstant(FastNode node, Dictionary<FastTensorKey, TensorShapeInfo> tensorStore)
    {
        var output = FirstOutputKey(node);
        if (output is null) return;

        // Try to get TensorData from the value attribute
        var tensorData = node.Attributes.GetTensorVal(AttrValue);
        if (tensorData is not null)
        {
            StoreTensorInfo(tensorStore, output.Value, tensorData);
            return;
        }

        // Handle scalar constant types
        if (!node.Attributes.IsDefaultValue("value_int"))
        {
            var intVal = node.Attributes.GetLongVal("value_int")!.Value;
            var data = TensorData.CreateFromRawBytes(new Shape(Array.Empty<long>()), DType.Int64, BitConverter.GetBytes(intVal));
            StoreTensorInfo(tensorStore, output.Value, data);
            return;
        }
        if (!node.Attributes.IsDefaultValue("value_float"))
        {
            var floatVal = node.Attributes.GetFloatVal("value_float")!.Value;
            var data = TensorData.CreateFromRawBytes(new Shape(Array.Empty<long>()), DType.Float32, BitConverter.GetBytes(floatVal));
            StoreTensorInfo(tensorStore, output.Value, data);
            return;
        }
        // Fallback: execute the constant node through ONNX Runtime.
        ExecuteNode(node, tensorStore);
    }

    private void ExecuteNode(FastNode node, Dictionary<FastTensorKey, TensorShapeInfo> tensorStore)
    {
        // Build a tiny FastComputationGraph: [N CONSTANT nodes] → [op clone] → outputs.
        var inputs = node.Inputs;
        var miniGraph = new FastComputationGraph();
        var constantOutputKeys = new FastTensorKey?[inputs.Count];

        for (int i = 0; i < inputs.Count; i++)
        {
            var input = inputs[i];
            if (input is null)
            {
                constantOutputKeys[i] = null;
                continue;
            }

            if (!tensorStore.TryGetValue(input.Value, out var info))
                return; // missing input shape — can't run ORT, leave node unresolved

            var inputData = info.Data ?? CreateZeroTensorData(info.Shape, info.DType);
            var constNode = Shorokoo.Core.Nodes.Processors.Fast.FastInternalOp.Constant(inputData);
            miniGraph.Nodes.Add(constNode);
            constantOutputKeys[i] = constNode.FullOutputs[""][0]!.Value;
        }

        // Clone the source node with a fresh key and remap inputs to point to constant outputs.
        var freshKey = FastNodeKey.New();
        var newFullInputs = new Dictionary<string, List<FastTensorKey?>>();
        // Walk source FullInputs in deterministic order (matches node.Inputs flat order).
        int flatIdx = 0;
        foreach (var (slotName, slot) in node.FullInputs.OrderBy(x => x.Key, System.StringComparer.Ordinal))
        {
            var remapped = new List<FastTensorKey?>(slot.Count);
            foreach (var k in slot)
            {
                if (k is null) remapped.Add(null);
                else remapped.Add(flatIdx < constantOutputKeys.Length ? constantOutputKeys[flatIdx] : null);
                flatIdx++;
            }
            newFullInputs[slotName] = remapped;
        }

        var newFullOutputs = new Dictionary<string, List<FastTensorKey?>>();
        // Allocate fresh output keys mirroring the original output structure.
        var freshOutputKeys = new List<FastTensorKey?>();
        foreach (var (slotName, slot) in node.FullOutputs)
        {
            var remapped = new List<FastTensorKey?>(slot.Count);
            for (int i = 0; i < slot.Count; i++)
            {
                if (slot[i] is null) { remapped.Add(null); continue; }
                var newKey = new FastTensorKey(freshKey, slot[i]!.Value.OutputIndex);
                remapped.Add(newKey);
                freshOutputKeys.Add(newKey);
            }
            newFullOutputs[slotName] = remapped;
        }

        // Strip out graph-typed attributes (already filtered earlier via HasGraphAttribute, but
        // this is defensive — also drop Shorokoo-internal metadata that ORT can't consume).
        var attrDefs = node.Attributes.AttributeDefs;
        var attrVals = node.Attributes.GetAttributeVals();
        var keptAttrPairs = new List<(string, object?)>();
        foreach (var def in attrDefs)
        {
            if (def.AttributeName.StartsWith("shrk_")) continue;
            if (def.Type == AttributeType.Graph) continue;
            if (attrVals.TryGetValue(def.AttributeName, out var v))
                keptAttrPairs.Add((def.AttributeName, v));
        }
        var clonedAttrs = OnnxCSharpAttributes.FromCSharpVals(
            keptAttrPairs.ToDictionary(p => p.Item1, p => p.Item2),
            attrDefs.Where(d => !d.AttributeName.StartsWith("shrk_") && d.Type != AttributeType.Graph).ToImmutableList());

        var clonedNode = new FastNode
        {
            Key = freshKey,
            OpCode = node.OpCode,
            Attributes = clonedAttrs,
            FullInputs = newFullInputs,
            FullOutputs = newFullOutputs,
        };
        miniGraph.Nodes.Add(clonedNode);

        // Output strategy: when the cloned op has multiple outputs, bundle them all
        // into a single SEQUENCE_CONSTRUCT and expose just that one sequence as the
        // graph output. Each per-node ORT session then returns a single result regardless
        // of the original op's output count, which sidesteps multi-output handling
        // fragility and dramatically reduces ORT plumbing per node.
        //
        // SEQUENCE_CONSTRUCT requires all bundled tensors to share a dtype, so for
        // ops with mixed-dtype outputs (TopK values+indices, Dropout output+mask, etc.)
        // we fall back to the legacy direct-outputs path. We don't know dtypes upfront,
        // so we attempt the sequence path first; if ORT rejects it (dtype mismatch or
        // any other reason) we retry without the wrap.
        var realOutputKeys = freshOutputKeys.Where(k => k.HasValue).Select(k => k!.Value).ToList();
        bool useSequenceWrap = realOutputKeys.Count > 1;

        FastTensorKey? sequenceOutKey = null;
        if (useSequenceWrap)
        {
            var seqNodeKey = FastNodeKey.New();
            var seqNode = FastNodeConstructionUtils.CreateSequenceConstruct(
                seqNodeKey, realOutputKeys.Cast<FastTensorKey?>());
            miniGraph.Nodes.Add(seqNode);
            sequenceOutKey = new FastTensorKey(seqNodeKey, 0);
            miniGraph.Outputs.Add(sequenceOutKey.Value);
        }
        else
        {
            foreach (var k in freshOutputKeys)
                if (k.HasValue) miniGraph.Outputs.Add(k.Value);
        }

        Shorokoo.NamedModelParam[] results;
        try
        {
            results = _computeContext.Execute(miniGraph);
        }
        catch (Exception) when (CatchShapeInferenceErrors())
        {
            // Sequence-wrap path failed (typically due to mixed-dtype outputs that
            // SEQUENCE_CONSTRUCT rejects). Retry without the wrap.
            if (useSequenceWrap)
            {
                miniGraph.Nodes.RemoveAt(miniGraph.Nodes.Count - 1);
                miniGraph.Outputs.Clear();
                foreach (var k in freshOutputKeys)
                    if (k.HasValue) miniGraph.Outputs.Add(k.Value);
                useSequenceWrap = false;
                try { results = _computeContext.Execute(miniGraph); }
                catch (Exception) when (CatchShapeInferenceErrors())
                {
                    return; // ORT couldn't resolve this node; leave it unresolved
                }
            }
            else
            {
                return; // ORT couldn't resolve this node; leave it unresolved
            }
        }

        var srcOutputs = node.Outputs;
        if (useSequenceWrap)
        {
            // Exactly one result expected — the sequence. Unwrap it into per-output tensors.
            TensorDataSequence? seq;
            try { seq = results[0].ToTensorDataSequence(); }
            catch (Exception) when (CatchShapeInferenceErrors()) { return; }
            if (seq is null) return;
            for (int i = 0; i < srcOutputs.Count && i < seq.Count; i++)
            {
                var srcOut = srcOutputs[i];
                if (srcOut is null) continue;
                StoreTensorInfo(tensorStore, srcOut.Value, seq[i]);
            }
        }
        else
        {
            // Direct path: each result is its own NamedModelParam. Skip non-tensor
            // results (sequences/optionals) — TensorShapeInfo only models a flat tensor.
            for (int i = 0; i < srcOutputs.Count && i < results.Length; i++)
            {
                var srcOut = srcOutputs[i];
                if (srcOut is null) continue;

                TensorData? resultData;
                try { resultData = results[i].ToTensorData(); }
                catch (Exception) when (CatchShapeInferenceErrors()) { continue; }
                if (resultData is null) continue;
                StoreTensorInfo(tensorStore, srcOut.Value, resultData);
            }
        }
    }

    /// <summary>Filter for shape-inference error catch blocks. Returns true for recoverable errors.</summary>
    private static bool CatchShapeInferenceErrors() => true;

    private static FastTensorKey? FirstOutputKey(FastNode node)
    {
        foreach (var slot in node.FullOutputs.Values)
            foreach (var k in slot)
                if (k is FastTensorKey key && !key.IsEmpty)
                    return key;
        return null;
    }

    private static void StoreTensorInfo(
        Dictionary<FastTensorKey, TensorShapeInfo> store,
        FastTensorKey key,
        TensorData data)
    {
        var isSmall = data.Shape.Count <= MaxSmallTensorElements;
        store[key] = new TensorShapeInfo(
            data.Shape,
            data.DType,
            isSmall ? data : null);
    }

    private static TensorData CreateZeroTensorData(Shape shape, DType dtype)
    {
        var bitsPerElement = dtype.EncodingBitCount;
        if (bitsPerElement < 8)
            throw new NotSupportedException(
                $"Data type {dtype} with {bitsPerElement}-bit encoding is not supported for zero tensor creation.");

        var bytesPerElement = bitsPerElement / 8;
        var totalBytes = shape.Count * bytesPerElement;
        var zeroBytes = new byte[totalBytes];
        return TensorData.CreateFromRawBytes(shape, dtype, zeroBytes);
    }
}
