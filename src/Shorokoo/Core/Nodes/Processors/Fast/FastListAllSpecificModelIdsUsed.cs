// PART 1: Entry point. Orchestrates mask-graph construction, QEE run, and result extraction.
//
// The class is split across 4 files that are concatenated into
// src/Shorokoo/Framework/Nodes/Processors/Fast/FastListAllSpecificModelIdsUsed.cs.

using Shorokoo.Core;
using Shorokoo.Core.Graph;
using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Inference;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;
using Shorokoo.Core.Nodes.Processors.AutoGrad;

namespace Shorokoo.Core.Nodes.Processors.Fast;

/// <summary>
/// Given a <see cref="FastComputationGraph"/>, a list of candidate specific model IDs and
/// a set of hint inputs, returns the subset of candidate model IDs that actually contribute
/// to a graph output — i.e. whose corresponding <c>TRAINABLE_PARAM_ID_REF</c> node is
/// reachable from the graph outputs through data edges, with IF branches correctly gated
/// on their (already-folded-to-constant) conditions and LOOP iterations unioned through
/// scan-variable masks.
///
/// <para>
/// Mechanism: clones the input graph and appends a parallel mask-computation subgraph —
/// every tensor <c>t</c> gets an associated <c>Vector&lt;bit&gt;</c> mask
/// <c>mask(t)</c> of length <c>numModelIds</c>. Propagation rules:
/// </para>
/// <list type="bullet">
///   <item>Default op: <c>mask(output) = OR(mask(inputs))</c>.</item>
///   <item><c>TRAINABLE_PARAM_ID_REF</c>: adds a one-hot bit for its specific model ID to
///     the combined input mask.</item>
///   <item><c>IF_CLOSE</c>: <c>mask(output_i) = condMask | Where(cond, thenMask_i, elseMask_i)</c>.
///     Because the IF conditions are already folded to CONSTANT booleans by
///     <see cref="FastUnpackModelStruct"/> replacing each <c>MODEL_HYPERPARAM</c> with the
///     caller's hyperparam tensor, QEE evaluates the condition-gated mask to one
///     branch-or-the-other.</item>
///   <item><c>LOOP_OPEN</c> / <c>LOOP_CLOSE</c>: carry variables get paired mask carries;
///     scan variables get paired mask scan variables that are reduced (MAX along axis 0)
///     into a single per-output mask vector after the close node.</item>
/// </list>
///
/// <para>
/// After construction, the combined mask of all graph outputs is set as the new (single)
/// graph output and the whole extended graph is run through <see cref="QuickExecutionEngine"/>.
/// The bits set in the final mask tensor name the live model IDs.
/// </para>
///
/// <para>
/// This replaces a previous CG round-trip through <c>FastProcessorHelper.Apply</c>
/// onto a now-deleted CG-side equivalent, which was prohibitively memory-heavy on bigger
/// models (ResNet50 / RetinaNet crashed the test host).
/// </para>
/// </summary>
internal static partial class FastListAllSpecificModelIdsUsed
{
    /// <summary>
    /// Vector length cap for the mask tensors. QEE's default element cap (256) is far too
    /// small for ResNet-scale model-ID spaces, so the instance used for this analysis is
    /// configured with a much bigger cap. If a graph actually asks for more model IDs than
    /// this, we conservatively return all candidates as "used" rather than produce a
    /// silently-truncated mask.
    /// </summary>
    private const int MaxMaskVectorLength = 1_000_000;

    public static ImmutableArray<ModelId> Process(
        FastComputationGraph graph,
        ModelParamList inputHints,
        ImmutableArray<ModelId> candidateModelIds)
    {
        if (candidateModelIds.IsEmpty)
            return ImmutableArray<ModelId>.Empty;

        var tensorDims = FoldHelpers.MaxModelIdCounts(candidateModelIds);
        long numModelIds = 1L;
        for (int i = 0; i < tensorDims.Length; i++) numModelIds *= tensorDims[i];
        if (numModelIds <= 0 || numModelIds > MaxMaskVectorLength)
            return candidateModelIds; // Safety: fall back to "all used" on unreasonable sizes.

        var transformArr = FoldHelpers.IndexToFlattenedIndexTransform(tensorDims);
        long transformLen = transformArr.Length;

        // Work on a clone so the caller's graph isn't mutated.
        var workGraph = graph.Clone();

        var context = new MaskBuildContext
        {
            NumModelIds = numModelIds,
            TransformLen = transformLen,
            TransformArr = transformArr,
            TensorMasks = new Dictionary<FastTensorKey, FastTensorKey>(),
            NewNodes = new List<FastNode>(),
            NodeByKey = new Dictionary<FastNodeKey, FastNode>(workGraph.Nodes.Count),
        };

        foreach (var n in workGraph.Nodes) context.NodeByKey[n.Key] = n;

        // Pre-create shared constant nodes (empty mask, indices 0..N-1, transform vector).
        BuildSharedConstants(context);

        // Graph inputs have empty masks (nothing has contributed to them).
        foreach (var inputKey in workGraph.Inputs)
            context.TensorMasks[inputKey] = context.EmptyMaskKey;

        // Walk the nodes in their current topological order, emitting mask-computation nodes
        // after each original node. For LOOP_OPEN / LOOP_CLOSE we modify the node in place
        // to add extra carry / scan slots for the mask tensors.
        var finalNodes = new List<FastNode>(workGraph.Nodes.Count + context.NewNodes.Count + 128);
        finalNodes.AddRange(context.NewNodes); // shared constants come first
        context.NewNodes.Clear();

        foreach (var node in workGraph.Nodes)
        {
            // LOOP_OPEN / LOOP_CLOSE mutate the node in place AND add mask-computing nodes
            // that become new INPUTS of the same (possibly mutated) node. Those new nodes
            // must be placed BEFORE the node itself in topological order so QEE can
            // evaluate them first. Every other op just reads the node's already-computed
            // outputs, so its mask-computing nodes can safely land AFTER the node.
            bool isLoopBoundary = node.OpCode == OpCodes.LOOP_OPEN
                                || node.OpCode == OpCodes.LOOP_CLOSE;

            DispatchNode(node, context, workGraph);

            if (isLoopBoundary)
            {
                // Mask-computing ops feeding into the loop boundary's new carry/scan input
                // slots come first; the loop node itself must follow them.
                // (Post-loop REDUCE_MAX nodes for scan masks are only ever emitted by
                // HandleLoopClose and land AFTER the close node — we split them out below.)
                if (node.OpCode == OpCodes.LOOP_CLOSE)
                {
                    // HandleLoopClose may emit both pre-close mask ops (for the new input
                    // slots) and post-close ReduceMax ops (for scan-output masks). Split at
                    // the first ReduceMax and place the rest after the close node. Inputs to
                    // ReduceMax post-close are the LOOP_CLOSE's own outputs, which only exist
                    // after it runs.
                    int splitIdx = context.NewNodes.FindIndex(n => n.OpCode == OpCodes.REDUCE_MAX);
                    if (splitIdx < 0) splitIdx = context.NewNodes.Count;
                    for (int k = 0; k < splitIdx; k++) finalNodes.Add(context.NewNodes[k]);
                    finalNodes.Add(node);
                    for (int k = splitIdx; k < context.NewNodes.Count; k++) finalNodes.Add(context.NewNodes[k]);
                }
                else
                {
                    finalNodes.AddRange(context.NewNodes);
                    finalNodes.Add(node);
                }
                context.NewNodes.Clear();
            }
            else
            {
                finalNodes.Add(node);
                if (context.NewNodes.Count > 0)
                {
                    finalNodes.AddRange(context.NewNodes);
                    context.NewNodes.Clear();
                }
            }
        }

        // Combined mask of all graph outputs.
        var combinedOutputMask = CombineMasks(
            workGraph.Outputs.Select(o => context.TensorMasks.TryGetValue(o, out var m) ? m : context.EmptyMaskKey),
            context, finalNodes);

        workGraph.Nodes = finalNodes;
        workGraph.Outputs = new List<FastTensorKey> { combinedOutputMask };

        // Run QEE. Use an oversized MaxDataElements so mask vectors (length numModelIds) are
        // fully materialized — the default 256 would drop data on anything but tiny models.
        var engine = new QuickExecutionEngine { MaxDataElements = MaxMaskVectorLength };
        var initialInputs = BuildInitialInputs(inputHints, graph.Inputs, engine);
        Dictionary<FastTensorKey, IRuntimeTensor> store;
        try
        {
            store = engine.Run(workGraph, initialInputs);
        }
        catch
        {
            // If QEE can't run the extended graph for any reason, fall back to "all candidates
            // used" rather than risk pruning a live param.
            return candidateModelIds;
        }

        if (!store.TryGetValue(combinedOutputMask, out var resultRaw)
            || resultRaw is not RuntimeTensor resultRt
            || resultRt.BoolData is not { } bits)
        {
            // Shape-only or missing tensor — can't decide liveness reliably. Treat every
            // candidate as used to avoid pruning a live param.
            return candidateModelIds;
        }

        var bitsArr = bits;
        var usedIndices = new HashSet<int>();
        for (int i = 0; i < bitsArr.Length; i++)
            if (bitsArr[i]) usedIndices.Add(i);

        return candidateModelIds
            .Where(m => usedIndices.Contains((int)FoldHelpers.TransformModelIdToFlattenedIndex(m, transformArr)))
            .ToImmutableArray();
    }

    /// <summary>
    /// Dispatch table: pick the right handler based on the node's op code.
    /// </summary>
    private static void DispatchNode(FastNode node, MaskBuildContext ctx, FastComputationGraph graph)
    {
        switch (node.OpCode)
        {
            case OpCodes.LOOP_OPEN:
                HandleLoopOpen(node, ctx);
                break;
            case OpCodes.LOOP_CLOSE:
                HandleLoopClose(node, ctx);
                break;
            case OpCodes.IF_OPEN:
                // IF_OPEN's outputs are structural; no mask handling needed beyond the default
                // (the condition's mask propagates via IF_CLOSE).
                HandleDefault(node, ctx);
                break;
            case OpCodes.IF_CLOSE:
                HandleIfClose(node, ctx);
                break;
            case InternalOpCodes.TRAINABLE_PARAM_ID_REF:
                HandleTrainableParamIdRef(node, ctx);
                break;
            default:
                HandleDefault(node, ctx);
                break;
        }
    }

    /// <summary>
    /// Shared state threaded through every handler — pre-built constants, the tensor→mask
    /// lookup, and the output-accumulator list for newly-emitted mask-computing FastNodes.
    /// </summary>
    private sealed class MaskBuildContext
    {
        public long NumModelIds;
        public long TransformLen;
        public long[] TransformArr = null!;
        public FastTensorKey EmptyMaskKey;
        public FastTensorKey IndicesKey;          // [0, 1, ..., numModelIds-1]
        public FastTensorKey TransformVecKey;     // stride vector for model ID → flat index
        public FastTensorKey ZeroScalarLongKey;   // scalar 0L used as Pad fill value
        public FastTensorKey ZeroUnsqueezedKey;   // [0L] (vector of one) for left-pad amount
        public Dictionary<FastTensorKey, FastTensorKey> TensorMasks = null!;
        public List<FastNode> NewNodes = null!;
        public Dictionary<FastNodeKey, FastNode> NodeByKey = null!;
    }
}

// PART 2: Default / IF_CLOSE / TRAINABLE_PARAM_ID_REF handlers.

internal static partial class FastListAllSpecificModelIdsUsed
{
    /// <summary>
    /// Default-op mask: <c>mask(output_i) = OR(mask(inputs))</c>.
    ///
    /// Handles any op not specifically recognized as LOOP/IF/TRAINABLE_PARAM_ID_REF. If the
    /// node has no inputs at all (e.g. CONSTANT, MODEL_TENSOR_INPUT outside the graph inputs
    /// set), all outputs get the empty mask.
    /// </summary>
    private static void HandleDefault(FastNode node, MaskBuildContext ctx)
    {
        var inputMasks = new List<FastTensorKey>();
        foreach (var kvp in node.FullInputs)
        {
            foreach (var input in kvp.Value)
            {
                if (input is FastTensorKey ik && !ik.IsEmpty && ctx.TensorMasks.TryGetValue(ik, out var im))
                    inputMasks.Add(im);
            }
        }

        FastTensorKey combined;
        if (inputMasks.Count == 0)
            combined = ctx.EmptyMaskKey;
        else
            combined = CombineMasks(inputMasks, ctx, ctx.NewNodes);

        foreach (var kvp in node.FullOutputs)
            foreach (var output in kvp.Value)
                if (output is FastTensorKey ok && !ok.IsEmpty)
                    ctx.TensorMasks[ok] = combined;
    }

    /// <summary>
    /// <c>TRAINABLE_PARAM_ID_REF</c> mask: the union of input masks plus a one-hot bit at
    /// this node's specific model-ID index. Input[0] is the specific model-ID vector, which
    /// we convert to a flat index (right-pad with zeros to the transform vector's length,
    /// multiply element-wise, reduce-sum) and compare against the precomputed <c>indices</c>
    /// vector via Equal to get the one-hot mask.
    /// </summary>
    private static void HandleTrainableParamIdRef(FastNode node, MaskBuildContext ctx)
    {
        var modelIdInputKey = node.FullInputs.TryGetValue("", out var inputs) && inputs.Count > 0
            ? inputs[0]
            : null;
        var outputKey = node.FullOutputs.TryGetValue("", out var outputs) && outputs.Count > 0
            ? outputs[0]
            : null;
        if (outputKey is null || outputKey.Value.IsEmpty) return;

        var inputMasks = new List<FastTensorKey>();
        foreach (var kvp in node.FullInputs)
            foreach (var input in kvp.Value)
                if (input is FastTensorKey ik && !ik.IsEmpty && ctx.TensorMasks.TryGetValue(ik, out var im))
                    inputMasks.Add(im);

        FastTensorKey oneHotMask;
        if (modelIdInputKey is FastTensorKey mk && !mk.IsEmpty)
            oneHotMask = BuildOneHotMaskFromModelIdTensor(mk, ctx, ctx.NewNodes);
        else
            oneHotMask = ctx.EmptyMaskKey;

        var all = new List<FastTensorKey>(inputMasks) { oneHotMask };
        var combined = CombineMasks(all, ctx, ctx.NewNodes);
        ctx.TensorMasks[outputKey.Value] = combined;
    }

    /// <summary>
    /// <c>IF_CLOSE</c> mask: <c>mask(output_i) = condMask | Where(cond, thenMask_i, elseMask_i)</c>.
    ///
    /// FullInputs are grouped by branch: <c>"then_branch"</c> and <c>"else_branch"</c>. The
    /// condition comes from the paired <c>IF_OPEN</c>'s Input[0]; its mask propagates into
    /// every output so that a condition computed from a trainable param would count it live.
    /// The actual branch-select uses the original-graph condition tensor (already present in
    /// the graph, pointed to by the cloned IF_OPEN's inputs), broadcast across the mask
    /// vector length via <c>Where</c>.
    /// </summary>
    private static void HandleIfClose(FastNode node, MaskBuildContext ctx)
    {
        FastNode? openNode = null;
        if (node.GraphOpenNodeKey is FastNodeKey openKey && !openKey.IsEmpty)
            ctx.NodeByKey.TryGetValue(openKey, out openNode);
        if (openNode is null) { HandleDefault(node, ctx); return; }

        FastTensorKey? condKey = null;
        if (openNode.FullInputs.TryGetValue("", out var openInputs) && openInputs.Count > 0)
            condKey = openInputs[0];

        FastTensorKey condMask = condKey is FastTensorKey ck && !ck.IsEmpty && ctx.TensorMasks.TryGetValue(ck, out var cm)
            ? cm : ctx.EmptyMaskKey;

        node.FullInputs.TryGetValue("then_branch", out var thenInputs);
        node.FullInputs.TryGetValue("else_branch", out var elseInputs);
        thenInputs ??= new List<FastTensorKey?>();
        elseInputs ??= new List<FastTensorKey?>();

        if (node.FullOutputs.TryGetValue("", out var outputs))
        {
            int n = outputs.Count;
            for (int i = 0; i < n; i++)
            {
                var ok = outputs[i];
                if (ok is null || ok.Value.IsEmpty) continue;

                FastTensorKey thenMask = i < thenInputs.Count && thenInputs[i] is FastTensorKey tk && !tk.IsEmpty
                    && ctx.TensorMasks.TryGetValue(tk, out var tm) ? tm : ctx.EmptyMaskKey;
                FastTensorKey elseMask = i < elseInputs.Count && elseInputs[i] is FastTensorKey ek && !ek.IsEmpty
                    && ctx.TensorMasks.TryGetValue(ek, out var em) ? em : ctx.EmptyMaskKey;

                FastTensorKey selected;
                if (condKey is FastTensorKey condTk && !condTk.IsEmpty)
                    selected = CreateWhere(condTk, thenMask, elseMask, ctx.NewNodes);
                else
                    selected = thenMask; // No condition — treat both as potentially used.

                ctx.TensorMasks[ok.Value] = CombineMasks(new[] { condMask, selected }, ctx, ctx.NewNodes);
            }
        }
    }
}

// PART 3: LOOP_OPEN / LOOP_CLOSE handlers. These mutate the cloned node in place to add
// extra carry / scan slots so QEE carries the mask tensors through iterations alongside
// the real data.

internal static partial class FastListAllSpecificModelIdsUsed
{
    /// <summary>
    /// LOOP_OPEN layout (in the empty-group):
    /// <code>
    ///   Inputs  = [maxIter, cond, ...N carryInits]
    ///   Outputs = [iterIdx, condOut, ...N carryPassthrough]   (inside the "body" group)
    /// </code>
    /// Modified to carry N parallel mask tensors:
    /// <code>
    ///   Inputs  = [maxIter, cond, ...N carryInits, ...N carryInitMasks]
    ///   Outputs = [iterIdx, condOut, ...N carryPassthrough, ...N carryPassthroughMasks]
    /// </code>
    /// The mask initializers come from <c>ctx.TensorMasks</c> of the original carry inits.
    /// The passthrough mask outputs are assigned TensorKeys under the LOOP_OPEN's FastNodeKey
    /// with fresh output indices (appended after the originals).
    ///
    /// QEE's <see cref="Shorokoo.Core.Inference.Ops.LoopOpenOp"/> treats any
    /// Inputs past index 1 as loop variables and mirrors them into body outputs — so the
    /// extra carries get handled correctly without any QEE changes.
    /// </summary>
    private static void HandleLoopOpen(FastNode node, MaskBuildContext ctx)
    {
        if (!node.FullInputs.TryGetValue("", out var inputs)) return;
        if (!node.FullOutputs.TryGetValue("body", out var outputs)) return;

        int numLoopVars = inputs.Count - 2;
        if (numLoopVars <= 0)
        {
            // No loop variables — just record empty mask for iter_idx / cond_continue.
            foreach (var output in outputs)
                if (output is FastTensorKey o && !o.IsEmpty)
                    ctx.TensorMasks[o] = ctx.EmptyMaskKey;
            return;
        }

        // The iteration-count/cond-mask contributes to every loop variable's mask: if the
        // iter count or break condition is computed from a trainable param, that param is
        // "used" by the loop as a whole.
        var countCondMasks = new List<FastTensorKey>();
        for (int i = 0; i < 2; i++)
        {
            var maybe = inputs[i];
            if (maybe is FastTensorKey tk && !tk.IsEmpty && ctx.TensorMasks.TryGetValue(tk, out var m))
                countCondMasks.Add(m);
        }
        FastTensorKey countCondMask = countCondMasks.Count == 0
            ? ctx.EmptyMaskKey
            : CombineMasks(countCondMasks, ctx, ctx.NewNodes);

        // For each carry init, the body-side initial mask is carryInitMask | countCondMask.
        var carryInitMaskKeys = new List<FastTensorKey?>(numLoopVars);
        for (int i = 0; i < numLoopVars; i++)
        {
            var init = inputs[2 + i];
            if (init is null || init.Value.IsEmpty)
            {
                carryInitMaskKeys.Add(null);
                continue;
            }
            var initMask = ctx.TensorMasks.TryGetValue(init.Value, out var mm) ? mm : ctx.EmptyMaskKey;
            var orMask = CombineMasks(new[] { initMask, countCondMask }, ctx, ctx.NewNodes);
            carryInitMaskKeys.Add(orMask);
        }

        // Allocate new output slots on this LOOP_OPEN for mask passthrough.
        int oldOutputCount = outputs.Count;
        var maskOutputKeys = new List<FastTensorKey?>(numLoopVars);
        for (int i = 0; i < numLoopVars; i++)
        {
            var passthroughKey = new FastTensorKey(node.Key, oldOutputCount + i);
            outputs.Add(passthroughKey);
            maskOutputKeys.Add(passthroughKey);
        }

        // Append the mask initializers to the node's inputs (must stay in the empty group to
        // be treated as loop-variable inits by QEE).
        for (int i = 0; i < numLoopVars; i++)
            inputs.Add(carryInitMaskKeys[i]);

        // Record masks for the node's existing outputs.
        // outputs[0] = iterIdx, outputs[1] = condOut — both derive from maxIter/cond, so
        // their mask is countCondMask.
        if (outputs.Count > 0 && outputs[0] is FastTensorKey iterIdxKey && !iterIdxKey.IsEmpty)
            ctx.TensorMasks[iterIdxKey] = countCondMask;
        if (outputs.Count > 1 && outputs[1] is FastTensorKey condOutKey && !condOutKey.IsEmpty)
            ctx.TensorMasks[condOutKey] = countCondMask;

        for (int i = 0; i < numLoopVars; i++)
        {
            var oldCarryOutput = outputs[2 + i];
            var newMaskOutput = maskOutputKeys[i];
            if (oldCarryOutput is FastTensorKey co && !co.IsEmpty && newMaskOutput is FastTensorKey mo)
                ctx.TensorMasks[co] = mo;
        }
    }

    /// <summary>
    /// LOOP_CLOSE layout (in the "body" group for inputs, empty group for outputs):
    /// <code>
    ///   Inputs["body"] = [continueWhen, ...N carryUpdates, ...M scanVars]
    ///   Outputs[""]    = [...N finalCarries, ...M scanOutputs]
    /// </code>
    /// Modified to:
    /// <code>
    ///   Inputs["body"] = [continueWhen, ...N carryUpdates, ...N carryUpdateMasks,
    ///                     ...M scanVars,     ...M scanVarMasks]
    ///   Outputs[""]    = [...N finalCarries, ...N finalCarryMasks,
    ///                     ...M scanOutputs,  ...M scanOutputMasks]
    /// </code>
    /// Crucially, the carry and scan slots must remain paired with LOOP_OPEN's carries (the
    /// open has been modified to also have 2N carry inits), so the layout above keeps all 2N
    /// carry updates together before any scan variables. <c>N_loop</c> in QEE is derived as
    /// <c>openNode.Inputs.Count - 2</c>, which is now 2N thanks to <see cref="HandleLoopOpen"/>.
    ///
    /// After the LOOP_CLOSE, each scan-output mask is still a per-iteration tensor of shape
    /// [numIterations, numModelIds]; we emit a post-loop <c>ReduceMax</c> along axis 0 to
    /// collapse it to the single final Vector&lt;bit&gt; mask used by downstream consumers.
    /// </summary>
    private static void HandleLoopClose(FastNode node, MaskBuildContext ctx)
    {
        if (!node.FullInputs.TryGetValue("body", out var inputs)) return;
        if (!node.FullOutputs.TryGetValue("", out var outputs)) return;

        FastNode? openNode = null;
        if (node.GraphOpenNodeKey is FastNodeKey openKey && !openKey.IsEmpty)
            ctx.NodeByKey.TryGetValue(openKey, out openNode);
        if (openNode is null) { HandleDefault(node, ctx); return; }

        // N_loop is determined from the (already-modified) LOOP_OPEN. HandleLoopOpen ran first
        // for this pair in topological order, so openNode's inputs.Count is already 2 + 2*N
        // — the true carry count (N) is half of that minus-two-halved.
        if (!openNode.FullInputs.TryGetValue("", out var openInputs)) return;
        int halfInputs = openInputs.Count - 2;
        int numOrigLoopVars = halfInputs / 2; // after HandleLoopOpen doubled the carries
        if (numOrigLoopVars * 2 != halfInputs)
        {
            // HandleLoopOpen must have bailed out (no carry vars); nothing to do here.
            HandleDefault(node, ctx);
            return;
        }

        // Split current close inputs into: [continueWhen, ...origCarryUpdates, ...carryMaskUpdatesFromOpenMirror, ...origScanVars, ...scanMaskVars]
        // But HandleLoopOpen already grew openInputs. LOOP_CLOSE's CURRENT inputs haven't been
        // modified yet — they still hold:
        //   [continueWhen, ...origCarryUpdates, ...origScanVars]
        //
        // After modification they must hold:
        //   [continueWhen, ...origCarryUpdates, ...carryMaskUpdates,
        //                                       ...origScanVars,    ...scanMaskVars]
        //
        // i.e. we insert N carry-mask-updates between the carry updates and the scan vars,
        // and append N scan-mask-vars at the end, where N = numOrigLoopVars.
        var origCarryUpdates = new List<FastTensorKey?>();
        var origScanVars = new List<FastTensorKey?>();
        for (int i = 0; i < numOrigLoopVars && (1 + i) < inputs.Count; i++)
            origCarryUpdates.Add(inputs[1 + i]);
        for (int i = 1 + numOrigLoopVars; i < inputs.Count; i++)
            origScanVars.Add(inputs[i]);

        int numScan = origScanVars.Count;

        // continueWhen's mask gets OR'd into every loop-var & scan-var update mask.
        FastTensorKey continueWhenMask = ctx.EmptyMaskKey;
        if (inputs.Count > 0 && inputs[0] is FastTensorKey cwK && !cwK.IsEmpty
            && ctx.TensorMasks.TryGetValue(cwK, out var cwM))
            continueWhenMask = cwM;

        var carryUpdateMasks = new List<FastTensorKey?>(numOrigLoopVars);
        foreach (var cu in origCarryUpdates)
        {
            if (cu is null || cu.Value.IsEmpty) { carryUpdateMasks.Add(null); continue; }
            var baseMask = ctx.TensorMasks.TryGetValue(cu.Value, out var m) ? m : ctx.EmptyMaskKey;
            carryUpdateMasks.Add(CombineMasks(new[] { baseMask, continueWhenMask }, ctx, ctx.NewNodes));
        }
        var scanVarMasks = new List<FastTensorKey?>(numScan);
        foreach (var sv in origScanVars)
        {
            if (sv is null || sv.Value.IsEmpty) { scanVarMasks.Add(null); continue; }
            var baseMask = ctx.TensorMasks.TryGetValue(sv.Value, out var m) ? m : ctx.EmptyMaskKey;
            scanVarMasks.Add(CombineMasks(new[] { baseMask, continueWhenMask }, ctx, ctx.NewNodes));
        }

        // Rebuild the close's input list in the new layout.
        var newInputs = new List<FastTensorKey?>(1 + 2 * numOrigLoopVars + 2 * numScan);
        newInputs.Add(inputs.Count > 0 ? inputs[0] : null); // continueWhen
        newInputs.AddRange(origCarryUpdates);
        newInputs.AddRange(carryUpdateMasks);
        newInputs.AddRange(origScanVars);
        newInputs.AddRange(scanVarMasks);
        node.FullInputs["body"] = newInputs;

        // Build the close's new output list. Original layout is [N finalCarries, M scanOutputs];
        // the new layout is [N finalCarries, N finalCarryMasks, M scanOutputs, M scanOutputMasks].
        var origFinalCarries = new List<FastTensorKey?>();
        var origScanOutputs = new List<FastTensorKey?>();
        for (int i = 0; i < numOrigLoopVars && i < outputs.Count; i++)
            origFinalCarries.Add(outputs[i]);
        for (int i = numOrigLoopVars; i < outputs.Count; i++)
            origScanOutputs.Add(outputs[i]);

        // Allocate fresh output indices for mask outputs. Start right after all existing
        // outputs so the existing TensorKeys (which may be referenced elsewhere) don't shift.
        int nextOutputIdx = outputs.Count;
        var finalCarryMaskKeys = new List<FastTensorKey?>(numOrigLoopVars);
        for (int i = 0; i < numOrigLoopVars; i++)
            finalCarryMaskKeys.Add(new FastTensorKey(node.Key, nextOutputIdx++));
        var scanOutputMaskKeys = new List<FastTensorKey?>(numScan);
        for (int i = 0; i < numScan; i++)
            scanOutputMaskKeys.Add(new FastTensorKey(node.Key, nextOutputIdx++));

        var newOutputs = new List<FastTensorKey?>(origFinalCarries.Count + finalCarryMaskKeys.Count
                                            + origScanOutputs.Count + scanOutputMaskKeys.Count);
        newOutputs.AddRange(origFinalCarries);
        newOutputs.AddRange(finalCarryMaskKeys);
        newOutputs.AddRange(origScanOutputs);
        newOutputs.AddRange(scanOutputMaskKeys);
        node.FullOutputs[""] = newOutputs;

        // Record tensor masks for the final-carry outputs (direct assignment).
        for (int i = 0; i < numOrigLoopVars; i++)
        {
            if (origFinalCarries[i] is FastTensorKey fc && !fc.IsEmpty && finalCarryMaskKeys[i] is FastTensorKey fcm)
                ctx.TensorMasks[fc] = fcm;
        }

        // Scan-output masks come out as [numIterations, numModelIds] tensors. Reduce-max
        // along axis 0 collapses them to the per-output Vector<bit> mask.
        for (int i = 0; i < numScan; i++)
        {
            if (origScanOutputs[i] is FastTensorKey so && !so.IsEmpty && scanOutputMaskKeys[i] is FastTensorKey som)
            {
                var reduced = CreateReduceMaxAxis0(som, ctx.NewNodes);
                ctx.TensorMasks[so] = reduced;
            }
        }
    }
}

// PART 4: helpers — constant builders, op builders, one-hot construction, input preparation.

internal static partial class FastListAllSpecificModelIdsUsed
{
    /// <summary>
    /// Builds the shared constant FastNodes used by every mask computation: the empty-mask
    /// vector (all-false), the indices vector <c>[0, 1, …, numModelIds-1]</c>, the
    /// stride / transform vector used to flatten a specific model ID into a linear index,
    /// the scalar 0 used as a Pad fill value, and the [0] unsqueezed scalar used as the
    /// left-pad amount.
    /// </summary>
    private static void BuildSharedConstants(MaskBuildContext ctx)
    {
        long n = ctx.NumModelIds;

        var emptyBools = new bool[n];
        var emptyData = Shorokoo.Globals.TensorData(new long[] { n }, emptyBools);
        var emptyKey = FastNodeKey.New();
        ctx.EmptyMaskKey = new FastTensorKey(emptyKey, 0);
        ctx.NewNodes.Add(CreateConstantTensorDataNode(emptyKey, emptyData));

        var indicesArr = new long[n];
        for (int i = 0; i < n; i++) indicesArr[i] = i;
        var indicesData = Shorokoo.Globals.TensorData(new long[] { n }, indicesArr);
        var idxKey = FastNodeKey.New();
        ctx.IndicesKey = new FastTensorKey(idxKey, 0);
        ctx.NewNodes.Add(CreateConstantTensorDataNode(idxKey, indicesData));

        var transformData = Shorokoo.Globals.TensorData(
            new long[] { ctx.TransformLen }, ctx.TransformArr);
        var trKey = FastNodeKey.New();
        ctx.TransformVecKey = new FastTensorKey(trKey, 0);
        ctx.NewNodes.Add(CreateConstantTensorDataNode(trKey, transformData));

        // Scalar 0L (shape []) for Pad's constant_value input.
        var zeroScalarData = Shorokoo.Globals.TensorData(new long[0], new long[] { 0L });
        var zsKey = FastNodeKey.New();
        ctx.ZeroScalarLongKey = new FastTensorKey(zsKey, 0);
        ctx.NewNodes.Add(CreateConstantTensorDataNode(zsKey, zeroScalarData));

        // [0L] (shape [1]) used to build the Pad's pads vector (a 1-D tensor).
        var zeroVecData = Shorokoo.Globals.TensorData(new long[] { 1 }, new long[] { 0L });
        var zuKey = FastNodeKey.New();
        ctx.ZeroUnsqueezedKey = new FastTensorKey(zuKey, 0);
        ctx.NewNodes.Add(CreateConstantTensorDataNode(zuKey, zeroVecData));
    }

    /// <summary>
    /// OR-combines a sequence of mask tensor keys into a single mask. Empty input yields
    /// the empty mask; a singleton is returned as-is; otherwise we emit a left-fold of
    /// binary OR nodes.
    /// </summary>
    private static FastTensorKey CombineMasks(
        IEnumerable<FastTensorKey> maskKeys, MaskBuildContext ctx, List<FastNode> newNodes)
    {
        FastTensorKey? acc = null;
        foreach (var m in maskKeys)
        {
            if (acc is null) { acc = m; continue; }
            acc = CreateBinaryOp(OpCodes.OR, acc.Value, m, newNodes);
        }
        return acc ?? ctx.EmptyMaskKey;
    }

    /// <summary>
    /// Emits <c>Where(cond, thenMask, elseMask)</c> with scalar <c>cond</c> broadcast across
    /// both mask vectors. ONNX's Where handles the broadcasting; we don't need to pre-expand
    /// the condition.
    /// </summary>
    private static FastTensorKey CreateWhere(
        FastTensorKey cond, FastTensorKey thenK, FastTensorKey elseK, List<FastNode> newNodes)
    {
        var nodeKey = FastNodeKey.New();
        var outputKey = new FastTensorKey(nodeKey, 0);
        newNodes.Add(FastNodeCreationHelpers.CreateFastNode(
            nodeKey, OpCodes.WHERE,
            new Dictionary<string, object?>(),
            new FastTensorKey?[] { cond, thenK, elseK }));
        return outputKey;
    }

    /// <summary>
    /// Builds a Vector&lt;bit&gt; of length <c>numModelIds</c> with exactly one bit set at
    /// the flat index corresponding to the specific model ID in
    /// <paramref name="modelIdKey"/>.
    ///
    /// Flat index computation: right-pad the model ID vector with zeros out to the transform
    /// vector's length, multiply element-wise by the transform vector, reduce-sum to a scalar.
    /// The one-hot mask is then <c>Equal(indices, flatIdx)</c> with broadcasting over the vector.
    /// </summary>
    private static FastTensorKey BuildOneHotMaskFromModelIdTensor(
        FastTensorKey modelIdKey, MaskBuildContext ctx, List<FastNode> newNodes)
    {
        // "Size" op isn't registered in Definitions.NodeDefinitions, but Shape + ReduceProd
        // gives the same scalar result — Shape yields a 1-D vector of dims, ReduceProd over
        // an empty axes input reduces over all axes → scalar total size.
        var modelIdSize = CreateShapeSize(modelIdKey, newNodes);
        var transformSize = CreateShapeSize(ctx.TransformVecKey, newNodes);
        // padRight = transformSize - modelIdSize   (scalar)
        var padRightScalar = CreateBinaryOp(OpCodes.SUB, transformSize, modelIdSize, newNodes);
        // Unsqueeze padRight to shape [1] so we can concat it with [0] into the Pad pads arg.
        var padRightVec = CreateUnsqueezeScalarToVec(padRightScalar, newNodes);
        // pads = Concat([0], [padRight]) -> [0, padRight] (shape [2]).
        var padsVec = CreateConcatAxis0(new[] { ctx.ZeroUnsqueezedKey, padRightVec }, newNodes);
        // padded = Pad(modelIdKey, padsVec, zeroScalar)   using default PadMode.Constant (mode=0).
        var padded = CreatePad(modelIdKey, padsVec, ctx.ZeroScalarLongKey, newNodes);
        // multiplied = padded * transformVec
        var multiplied = CreateBinaryOp(OpCodes.MUL, padded, ctx.TransformVecKey, newNodes);
        // flatIdx = ReduceSum(multiplied, axes=[0], keepdims=0)  -> scalar int64
        var flatIdx = CreateReduceSumAxis0Scalar(multiplied, newNodes);
        // oneHotMask = Equal(indices, flatIdx)   -> [numModelIds] bool vector
        return CreateBinaryOp(OpCodes.EQUAL, ctx.IndicesKey, flatIdx, newNodes);
    }

    /// <summary>
    /// <c>Unsqueeze</c>-to-rank-1: maps a scalar to a 1-element vector. Uses a fresh axes
    /// CONSTANT <c>[0]</c> so each unsqueeze is independent (no shared-axis pitfalls).
    /// </summary>
    private static FastTensorKey CreateUnsqueezeScalarToVec(FastTensorKey scalarKey, List<FastNode> newNodes)
    {
        var axesKey = FastNodeKey.New();
        var axesTk = new FastTensorKey(axesKey, 0);
        newNodes.Add(CreateConstantTensorDataNode(axesKey,
            Shorokoo.Globals.TensorData(new long[] { 1 }, new long[] { 0L })));

        var unsqKey = FastNodeKey.New();
        var unsqTk = new FastTensorKey(unsqKey, 0);
        newNodes.Add(FastNodeCreationHelpers.CreateFastNode(
            unsqKey, OpCodes.UNSQUEEZE,
            new Dictionary<string, object?>(),
            new FastTensorKey?[] { scalarKey, axesTk }));
        return unsqTk;
    }

    /// <summary>
    /// Concats N 1-D int64 tensors along axis 0.
    /// </summary>
    private static FastTensorKey CreateConcatAxis0(IEnumerable<FastTensorKey> tensors, List<FastNode> newNodes)
    {
        var nodeKey = FastNodeKey.New();
        var outputKey = new FastTensorKey(nodeKey, 0);
        newNodes.Add(FastNodeCreationHelpers.CreateFastNode(
            nodeKey, OpCodes.CONCAT,
            new Dictionary<string, object?> { [AttrAxis] = 0L },
            tensors.Select(t => (FastTensorKey?)t).ToArray()));
        return outputKey;
    }

    /// <summary>
    /// Pad with mode=constant (the ONNX Pad op's <c>mode</c> attribute as a string);
    /// <paramref name="padsVec"/> is a 1-D int64 tensor [padLeft, padRight] and
    /// <paramref name="constantValue"/> is a scalar int64 0 used as the fill.
    /// </summary>
    private static FastTensorKey CreatePad(
        FastTensorKey data, FastTensorKey padsVec, FastTensorKey constantValue, List<FastNode> newNodes)
    {
        var nodeKey = FastNodeKey.New();
        var outputKey = new FastTensorKey(nodeKey, 0);
        newNodes.Add(FastNodeCreationHelpers.CreateFastNode(
            nodeKey, OpCodes.PAD,
            new Dictionary<string, object?> { [AttrMode] = PadMode.Constant },
            new FastTensorKey?[] { data, padsVec, constantValue }));
        return outputKey;
    }

    /// <summary>
    /// ReduceSum along axis 0 with keepdims=0 — scalar output from a 1-D input.
    /// </summary>
    private static FastTensorKey CreateReduceSumAxis0Scalar(FastTensorKey input, List<FastNode> newNodes)
    {
        var axesKey = FastNodeKey.New();
        var axesTk = new FastTensorKey(axesKey, 0);
        newNodes.Add(CreateConstantTensorDataNode(axesKey,
            Shorokoo.Globals.TensorData(new long[] { 1 }, new long[] { 0L })));

        var nodeKey = FastNodeKey.New();
        var outputKey = new FastTensorKey(nodeKey, 0);
        newNodes.Add(FastNodeCreationHelpers.CreateFastNode(
            nodeKey, OpCodes.REDUCE_SUM,
            new Dictionary<string, object?> { [AttrKeepdims] = false },
            new FastTensorKey?[] { input, axesTk }));
        return outputKey;
    }

    /// <summary>
    /// ReduceMax along axis 0 with keepdims=0 — collapses a [numIterations, numModelIds]
    /// scan-output mask tensor to the Vector&lt;bit&gt; mask tracked downstream.
    /// </summary>
    private static FastTensorKey CreateReduceMaxAxis0(FastTensorKey input, List<FastNode> newNodes)
    {
        var axesKey = FastNodeKey.New();
        var axesTk = new FastTensorKey(axesKey, 0);
        newNodes.Add(CreateConstantTensorDataNode(axesKey,
            Shorokoo.Globals.TensorData(new long[] { 1 }, new long[] { 0L })));

        var nodeKey = FastNodeKey.New();
        var outputKey = new FastTensorKey(nodeKey, 0);
        newNodes.Add(FastNodeCreationHelpers.CreateFastNode(
            nodeKey, OpCodes.REDUCE_MAX,
            new Dictionary<string, object?> { [AttrKeepdims] = false },
            new FastTensorKey?[] { input, axesTk }));
        return outputKey;
    }

    /// <summary>
    /// Scalar-size of a tensor: <c>ReduceProd(Shape(t))</c>. (The "Size" op code isn't
    /// registered in this codebase's Definitions table, so we compose Shape+ReduceProd
    /// which gives the same result: total element count as a scalar int64.)
    /// </summary>
    private static FastTensorKey CreateShapeSize(FastTensorKey input, List<FastNode> newNodes)
    {
        var shapeKey = FastNodeKey.New();
        var shapeTk = new FastTensorKey(shapeKey, 0);
        newNodes.Add(FastNodeCreationHelpers.CreateFastNode(
            shapeKey, OpCodes.SHAPE,
            new Dictionary<string, object?>(),
            new FastTensorKey?[] { input }));

        // ReduceProd with `noop_with_empty_axes=false` and no axes input reduces over all
        // axes — yields a scalar = product of all dims of `input`.
        var prodKey = FastNodeKey.New();
        var prodTk = new FastTensorKey(prodKey, 0);
        newNodes.Add(FastNodeCreationHelpers.CreateFastNode(
            prodKey, OpCodes.REDUCE_PROD,
            new Dictionary<string, object?>
            {
                [AttrKeepdims] = false,
                [AttrNoopWithEmptyAxes] = false,
            },
            new FastTensorKey?[] { shapeTk, null }));
        return prodTk;
    }

    /// <summary>Creates a single-input, single-output op FastNode.</summary>
    private static FastTensorKey CreateUnaryOp(string opCode, FastTensorKey input, List<FastNode> newNodes)
    {
        var nodeKey = FastNodeKey.New();
        var outputKey = new FastTensorKey(nodeKey, 0);
        newNodes.Add(FastNodeCreationHelpers.CreateFastNode(
            nodeKey, opCode,
            new Dictionary<string, object?>(),
            new FastTensorKey?[] { input }));
        return outputKey;
    }

    /// <summary>Creates a two-input, single-output op FastNode.</summary>
    private static FastTensorKey CreateBinaryOp(
        string opCode, FastTensorKey a, FastTensorKey b, List<FastNode> newNodes)
    {
        var nodeKey = FastNodeKey.New();
        var outputKey = new FastTensorKey(nodeKey, 0);
        newNodes.Add(FastNodeCreationHelpers.CreateFastNode(
            nodeKey, opCode,
            new Dictionary<string, object?>(),
            new FastTensorKey?[] { a, b }));
        return outputKey;
    }

    /// <summary>Builds a CONSTANT FastNode producing the given TensorData as its output.</summary>
    private static FastNode CreateConstantTensorDataNode(FastNodeKey nodeKey, TensorData td)
    {
        var tensorKey = new FastTensorKey(nodeKey, 0);
        var attrDefs = Definitions.NodeDefinitions[OpCodes.CONSTANT].AttributeDefs;
        var attrs = OnnxCSharpAttributes.FromCSharpVals(
            new Dictionary<string, object?> { [AttrValue] = td }, attrDefs);

        return new FastNode
        {
            Key = nodeKey,
            OpCode = OpCodes.CONSTANT,
            Attributes = attrs,
            FullOutputs = { [""] = new List<FastTensorKey?> { tensorKey } },
        };
    }

    /// <summary>
    /// Prepares the initial-input dictionary passed to QEE. Mirrors the existing logic in
    /// <see cref="FastConvertTrainableParamIdRefToTrainableParam"/>: for every graph input
    /// that has a matching entry in <paramref name="inputHints"/>, feed QEE the runtime
    /// tensor so shape-dependent ops inside the graph can evaluate fully.
    /// </summary>
    private static Dictionary<FastTensorKey, IRuntimeTensor>? BuildInitialInputs(
        ModelParamList inputHints,
        IReadOnlyList<FastTensorKey> graphInputKeys,
        QuickExecutionEngine engine)
    {
        if (inputHints is null || inputHints.ModelParams.Length == 0)
            return null;

        var dict = new Dictionary<FastTensorKey, IRuntimeTensor>();
        int limit = System.Math.Min(graphInputKeys.Count, inputHints.ModelParams.Length);
        for (int i = 0; i < limit; i++)
        {
            var td = inputHints.ModelParams[i].ToTensorData();
            if (td is null) continue;
            dict[graphInputKeys[i]] = Shorokoo.Core.Inference.Helpers.TensorDataConverter.ToRuntimeTensor(
                td, engine.MaxDataElements);
        }
        return dict.Count == 0 ? null : dict;
    }
}
