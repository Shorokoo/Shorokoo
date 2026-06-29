using Shorokoo.Core;
using Shorokoo.Core.Graph;
using Shorokoo.Runtime;
using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Inference;
using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics;
using System.Linq;
using Shorokoo.Core.Nodes.Processors.AutoGrad;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// Fast-graph counterparts of the processors used by
    /// <c>ComputationGraph.ToConcreteArchitecture</c>. Each <c>Process</c> method
    /// takes a <see cref="FastComputationGraph"/> and mutates it in place. Processors are
    /// implemented natively on <see cref="FastComputationGraph"/>.
    /// </summary>
    internal static class FastProcessorHelper
    {
        /// <summary>
        /// Builds a FastNodeKey → FastNode lookup that also maps output FastTensorKey.FastNodeKey
        /// entries. This handles LOOP_OPEN carry variables whose output TensorKeys have a
        /// FastNodeKey different from the LOOP_OPEN's own Key.
        /// </summary>
        public static Dictionary<FastNodeKey, FastNode> BuildNodeByKey(FastComputationGraph graph)
        {
            var nodeByKey = new Dictionary<FastNodeKey, FastNode>(graph.Nodes.Count);
            foreach (var n in graph.Nodes)
            {
                nodeByKey[n.Key] = n;
                foreach (var kvp in n.FullOutputs)
                    foreach (var ok in kvp.Value)
                        if (ok is not null && !ok.Value.IsEmpty)
                            nodeByKey.TryAdd(ok.Value.FastNodeKey, n);
            }
            return nodeByKey;
        }

        /// <summary>
        /// Thin wrapper around <see cref="FastComputationGraph.IsLinearOrderValid"/>.
        /// Kept as a separate name so existing pass call sites read the way they did
        /// when this was a Kahn re-sort.
        /// </summary>
        public static void EnsureTopologicalOrder(FastComputationGraph graph)
            => System.Diagnostics.Debug.Assert(graph.IsLinearOrderValid(), "graph.IsLinearOrderValid()");

        /// <summary>
        /// Removes nodes from <see cref="FastComputationGraph.Nodes"/> that are not
        /// reachable from the graph's outputs. This cleans up dead nodes left behind
        /// by native in-place processors that disconnect nodes without removing them.
        /// </summary>
        public static void RemoveUnreachableNodes(FastComputationGraph graph)
        {
            // Build output FastTensorKey → FastNodeKey mapping.
            var tensorToNodeKey = new Dictionary<FastTensorKey, FastNodeKey>();
            var nodesByKey = new Dictionary<FastNodeKey, FastNode>();
            foreach (var node in graph.Nodes)
            {
                nodesByKey[node.Key] = node;
                foreach (var kvp in node.FullOutputs)
                    foreach (var ok in kvp.Value)
                        if (ok is not null && !ok.Value.IsEmpty)
                            tensorToNodeKey[ok.Value] = node.Key;
            }

            // Walk backwards from outputs to find all reachable nodes.
            var reachable = new HashSet<FastNodeKey>();
            var worklist = new Queue<FastNodeKey>();

            void EnqueueTensor(FastTensorKey tk)
            {
                if (tk.IsEmpty) return;
                if (tensorToNodeKey.TryGetValue(tk, out var nk) && reachable.Add(nk))
                    worklist.Enqueue(nk);
            }

            foreach (var outputKey in graph.Outputs)
                EnqueueTensor(outputKey);

            // Graph inputs must keep their producer (MODEL_*INPUT) nodes alive even
            // if no path from any output reaches them — otherwise graph.Inputs ends
            // up referencing a tensor produced by a node that no longer exists.
            foreach (var inputKey in graph.Inputs)
                EnqueueTensor(inputKey);

            while (worklist.Count > 0)
            {
                var nk = worklist.Dequeue();
                if (!nodesByKey.TryGetValue(nk, out var node)) continue;

                foreach (var kvp in node.FullInputs)
                    foreach (var inputKey in kvp.Value)
                        if (inputKey is not null)
                            EnqueueTensor(inputKey.Value);

                if (node.GraphOpenNodeKey is FastNodeKey openKey && !openKey.IsEmpty && reachable.Add(openKey))
                    worklist.Enqueue(openKey);
            }

            if (reachable.Count == graph.Nodes.Count) return;

            graph.Nodes.RemoveAll(n => !reachable.Contains(n.Key));
        }

        /// <summary>
        /// Re-keys all nodes and tensor keys in a <see cref="FastComputationGraph"/> so that
        /// every node and tensor key is globally unique. Necessary because
        /// <see cref="Function.GetFastFlattenedGraph"/> is cached and shared across call
        /// sites, so cloning it multiple times produces subgraphs with identical keys.
        /// Without re-keying, inserting the same function's subgraph more than once causes
        /// key collisions.
        /// </summary>
        public static void RekeySubgraph(FastComputationGraph sub)
        {
            // Phase 1: assign new NodeKeys and build the mapping.
            var nodeKeyMap = new Dictionary<FastNodeKey, FastNodeKey>();
            foreach (var node in sub.Nodes)
            {
                var oldNodeKey = node.Key;
                var newNodeKey = FastNodeKey.New();
                nodeKeyMap[oldNodeKey] = newNodeKey;
                node.Key = newNodeKey;
            }

            // Phase 1b: some TensorKeys reference NodeKeys that don't belong to any
            // node in this subgraph — these are "synthetic" cross-node references
            // used by ops like LOOP_OPEN whose iter_idx / cond-carry tensors have a
            // dedicated FastNodeKey that isn't itself a node. Those synthetic NodeKeys
            // must also be rekeyed per clone; otherwise two independent copies of
            // the same cached flattened graph would share the same iter_idx key in
            // the main graph, causing the cycle detector to trace parent edges to
            // the wrong loop. Subgraph-input TensorKeys stay as-is — they're
            // placeholders the splice step replaces with caller-side keys.
            var subgraphInputNodeKeys = new HashSet<FastNodeKey>();
            foreach (var ikey in sub.Inputs)
                if (!ikey.IsEmpty) subgraphInputNodeKeys.Add(ikey.FastNodeKey);

            void RegisterSynthetic(FastTensorKey tk)
            {
                if (tk.IsEmpty) return;
                if (nodeKeyMap.ContainsKey(tk.FastNodeKey)) return;
                if (subgraphInputNodeKeys.Contains(tk.FastNodeKey)) return;
                nodeKeyMap[tk.FastNodeKey] = FastNodeKey.New();
            }
            foreach (var node in sub.Nodes)
            {
                foreach (var kvp in node.FullOutputs)
                    foreach (var tk in kvp.Value)
                        if (tk is FastTensorKey v) RegisterSynthetic(v);
                foreach (var kvp in node.FullInputs)
                    foreach (var tk in kvp.Value)
                        if (tk is FastTensorKey v) RegisterSynthetic(v);
            }

            // Phase 2: remap all TensorKeys in outputs. A FastTensorKey's FastNodeKey may refer
            // to ANY node in the subgraph (not necessarily the producing node — e.g.
            // LOOP_OPEN carry variables have TensorKeys whose FastNodeKey points to loop body
            // nodes). We must look up the old FastNodeKey in our map.
            var tensorKeyMap = new Dictionary<FastTensorKey, FastTensorKey>();
            foreach (var node in sub.Nodes)
            {
                foreach (var kvp in node.FullOutputs)
                {
                    for (int i = 0; i < kvp.Value.Count; i++)
                    {
                        if (kvp.Value[i] is FastTensorKey oldTk && !oldTk.IsEmpty)
                        {
                            FastNodeKey mappedNodeKey;
                            if (!nodeKeyMap.TryGetValue(oldTk.FastNodeKey, out mappedNodeKey))
                                mappedNodeKey = oldTk.FastNodeKey; // truly external (e.g. subgraph input), keep as-is

                            var newTk = new FastTensorKey(mappedNodeKey, oldTk.OutputIndex);
                            tensorKeyMap[oldTk] = newTk;
                            kvp.Value[i] = newTk;
                        }
                    }
                }
            }

            // Phase 3: remap all inputs and GraphOpenNodeKey
            foreach (var node in sub.Nodes)
            {
                foreach (var kvp in node.FullInputs)
                {
                    for (int i = 0; i < kvp.Value.Count; i++)
                    {
                        if (kvp.Value[i] is FastTensorKey oldTk && tensorKeyMap.TryGetValue(oldTk, out var newTk))
                            kvp.Value[i] = newTk;
                    }
                }

                if (node.GraphOpenNodeKey is FastNodeKey openKey && !openKey.IsEmpty)
                {
                    if (nodeKeyMap.TryGetValue(openKey, out var newOpenKey))
                        node.GraphOpenNodeKey = newOpenKey;
                }
            }

            // Phase 4: remap graph-level inputs and outputs
            for (int i = 0; i < sub.Inputs.Count; i++)
            {
                if (tensorKeyMap.TryGetValue(sub.Inputs[i], out var newTk))
                    sub.Inputs[i] = newTk;
            }
            for (int i = 0; i < sub.Outputs.Count; i++)
            {
                if (tensorKeyMap.TryGetValue(sub.Outputs[i], out var newTk))
                    sub.Outputs[i] = newTk;
            }

            // Validation: all GraphOpenNodeKey references must resolve to a node in the subgraph.
            var subKeys = new HashSet<FastNodeKey>();
            foreach (var node in sub.Nodes)
                subKeys.Add(node.Key);
            foreach (var node in sub.Nodes)
            {
                if (node.GraphOpenNodeKey is FastNodeKey openKey && !openKey.IsEmpty)
                {
                    if (!subKeys.Contains(openKey))
                        throw new InvalidOperationException(
                            $"RekeySubgraph: {node.OpCode} (Key={node.Key}) GraphOpenNodeKey {openKey} not found in subgraph nodes ({sub.Nodes.Count} nodes).");
                }
            }

            // Validation: all node keys must be unique.
            if (subKeys.Count != sub.Nodes.Count)
                throw new InvalidOperationException(
                    $"RekeySubgraph: duplicate node keys detected. {sub.Nodes.Count} nodes but only {subKeys.Count} unique keys.");
        }
    }

    /// <summary>
    /// Native fast-graph processor that mutates <see cref="FastNode.Attributes"/> and
    /// <see cref="FastNode.IdentifierTemplate"/> in place for nodes that carry a
    /// <c>shrk_local_model_id</c> attribute.
    /// </summary>
    internal static class FastApplyIdentifierTemplates
    {
        public static void Process(FastComputationGraph graph)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));

            // Build FastNodeKey → FastNode lookup for navigating input chains.
            var nodeByKey = FastProcessorHelper.BuildNodeByKey(graph);

            var tensorInfos = FastTensorInfoProcessor.BuildTensorInfoLookup(graph);

            var idPickerEnumerator = FindNextSpot().GetEnumerator();
            var loopModelIds = new Dictionary<FastTensorKey, (ModelId loopModelId, IEnumerator<int> loopContentsModelIdPicker)>();
            var loopDedupeIdsMap = new Dictionary<FastTensorKey, int>();
            var moduleNameDedupeIdMap = new Dictionary<string, int>();

            foreach (var fastNode in graph.Nodes)
            {
                if (!fastNode.Attributes.IsAttributeDefined(OnnxOpAttributeNames.ShrkAttrLocalModelId))
                    continue;

                Debug.Assert(fastNode.OpCode == InternalOpCodes.TRAINABLE_PARAM_REF ||
                             fastNode.OpCode == InternalOpCodes.MODULE_SET_HYPERPARAMS);

                var toIdentifyKey = fastNode.Outputs[0]!.Value;

                var loopIterationIndices = GetIterationIndices(fastNode, nodeByKey);
                var modelIdToUse = AllocateLoopModelIds(loopIterationIndices, idPickerEnumerator, loopModelIds);

                // Update model ID in attributes.
                var dctRebuiltAttributes = fastNode.Attributes.GetAttributeVals().ToDictionary();
                dctRebuiltAttributes[OnnxOpAttributeNames.ShrkAttrLocalModelId] =
                    modelIdToUse.Vals.Select(x => (long)x).ToArray();

                // Compute loop dedupe IDs.
                foreach (var loopIndex in loopIterationIndices)
                {
                    if (!loopDedupeIdsMap.ContainsKey(loopIndex))
                        loopDedupeIdsMap.Add(loopIndex, loopDedupeIdsMap.Count);
                }
                var loopDedupeIds = loopIterationIndices.Select(x => loopDedupeIdsMap[x]).ToImmutableArray();

                // Get the module function from tensor info or node's target function.
                var tensorInfo = tensorInfos[toIdentifyKey];
                var modelFn = (tensorInfo.ModuleFn ?? fastNode.TargetFunction)!;
                var moduleName = modelFn.DefaultName!;

                // Track module name dedupe IDs.
                if (moduleNameDedupeIdMap.TryGetValue(moduleName, out var existing))
                    moduleNameDedupeIdMap[moduleName] = existing + 1;
                else
                    moduleNameDedupeIdMap[moduleName] = 0;
                var moduleNameDedupeId = moduleNameDedupeIdMap[moduleName];

                // Build identifier template.
                ModelParamIdentifierTemplate? identifierTemplate = null;
                if (modelFn.FunctionType == FunctionType.Module)
                    identifierTemplate = ModelParamIdentifierTemplate.LocalModule(
                        modelIdToUse, moduleName, moduleNameDedupeId, loopDedupeIds);
                else if (modelFn.FunctionType == FunctionType.TrainableParamInitializer ||
                         modelFn.FunctionType == FunctionType.StateParamInitializer)
                    identifierTemplate = ModelParamIdentifierTemplate.LocalTrainableParam(
                        modelIdToUse, moduleName, moduleNameDedupeId, loopDedupeIds);

                // Mutate in place.
                fastNode.Attributes = OnnxCSharpAttributes.FromCSharpVals(
                    dctRebuiltAttributes, fastNode.Attributes.AttributeDefs);
                fastNode.IdentifierTemplate = identifierTemplate?.ToString();
            }
        }

        /// <summary>
        /// Navigates from a TRAINABLE_PARAM_REF / MODULE_SET_HYPERPARAMS node backwards
        /// through its iteration-indices input chain (CONCAT → IDENTITY → UNSQUEEZE → LOOP_OPEN)
        /// and returns the TensorKeys of the LOOP_OPEN iteration outputs.
        /// </summary>
        private static ImmutableArray<FastTensorKey> GetIterationIndices(
            FastNode node, Dictionary<FastNodeKey, FastNode> nodeByKey)
        {
            int inputIndex;
            if (node.OpCode == InternalOpCodes.TRAINABLE_PARAM_ID_REF ||
                node.OpCode == InternalOpCodes.TRAINABLE_PARAM_MODEL_REF ||
                node.OpCode == InternalOpCodes.MODULE_SET_HYPERPARAMS)
                inputIndex = 1;
            else if (node.OpCode == InternalOpCodes.TRAINABLE_PARAM_REF)
                inputIndex = 0;
            else
                throw new InvalidOperationException(
                    $"FastApplyIdentifierTemplates: unexpected OpCode '{node.OpCode}'");

            var inputs = node.Inputs;
            var iterationIndicesKey = inputs[inputIndex];
            if (iterationIndicesKey is null || iterationIndicesKey.Value.IsEmpty)
                return [];

            var producingNode = nodeByKey[iterationIndicesKey.Value.FastNodeKey];

            if (producingNode.OpCode == OpCodes.CONSTANT)
                return [];

            Debug.Assert(producingNode.OpCode == OpCodes.CONCAT);

            var concatInputs = producingNode.Inputs;
            var builder = ImmutableArray.CreateBuilder<FastTensorKey>(concatInputs.Count);

            foreach (var concatInput in concatInputs)
            {
                Debug.Assert(concatInput is not null);
                var currentKey = concatInput.Value;
                var currentNode = nodeByKey[currentKey.FastNodeKey];

                if (currentNode.OpCode == OpCodes.IDENTITY)
                {
                    currentKey = currentNode.Inputs[0]!.Value;
                    currentNode = nodeByKey[currentKey.FastNodeKey];
                }

                if (currentNode.OpCode == OpCodes.UNSQUEEZE)
                {
                    currentKey = currentNode.Inputs[0]!.Value;
                    currentNode = nodeByKey[currentKey.FastNodeKey];
                }

                Debug.Assert(currentNode.OpCode == OpCodes.LOOP_OPEN);
                builder.Add(currentKey);
            }

            return builder.ToImmutable();
        }

        private static ModelId AllocateLoopModelIds(
            ImmutableArray<FastTensorKey> iterationIndices,
            IEnumerator<int> globalScopeIdPicker,
            Dictionary<FastTensorKey, (ModelId loopModelId, IEnumerator<int> loopContentsModelIdPicker)> loopModelIds)
        {
            if (iterationIndices.Length == 0)
            {
                globalScopeIdPicker.MoveNext();
                return new ModelId(globalScopeIdPicker.Current);
            }

            var parentLoopModelIdPicker = globalScopeIdPicker;
            var parentLoopModelId = new ModelId((int[])[]);

            foreach (var iterationIndex in iterationIndices)
            {
                if (loopModelIds.TryGetValue(iterationIndex, out var entry))
                {
                    (parentLoopModelId, parentLoopModelIdPicker) = entry;
                }
                else
                {
                    parentLoopModelIdPicker.MoveNext();
                    var loopModelId = parentLoopModelIdPicker.Current;
                    var currentLoopModelId = new ModelId(parentLoopModelId, new ModelId(loopModelId, -1));
                    var currentLoopModelIdPicker = FindNextSpot().GetEnumerator();

                    loopModelIds[iterationIndex] = (currentLoopModelId, currentLoopModelIdPicker);

                    parentLoopModelIdPicker = currentLoopModelIdPicker;
                    parentLoopModelId = currentLoopModelId;
                }
            }

            parentLoopModelIdPicker.MoveNext();
            return new ModelId(parentLoopModelId, new ModelId(parentLoopModelIdPicker.Current));
        }

        private static IEnumerable<int> FindNextSpot()
        {
            for (int spot = 1; ; spot++)
                yield return spot;
        }
    }

    /// <summary>
    /// Native fast-graph implementation of <c>InlineModulesAndFunctions</c>.
    /// The main graph stays as <see cref="FastComputationGraph"/> throughout and no
    /// <c>ComputationGraph</c> round-trip occurs. Function subgraphs are
    /// obtained from <see cref="Function.GetFastFlattenedGraph"/>, which recursively
    /// invokes this pass on sub-function bodies and caches the result.
    ///
    /// Process iterates until no <c>MODEL_INVOKE</c> / <c>FUNCTION_INVOKE</c> can be
    /// further inlined: when a freshly-spliced body references a caller's
    /// MODULE_SET_HYPERPARAMS for what used to be a signature-only model variable,
    /// the next sweep sees the new producer and completes the inlining.
    /// </summary>
    internal static class FastInlineModulesAndFunctions
    {
        public static void Process(FastComputationGraph graph)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));
            while (ProcessOnePass(graph)) { }
        }

        private static bool ProcessOnePass(FastComputationGraph graph)
        {
            var nodeByKey = FastProcessorHelper.BuildNodeByKey(graph);

            var tensorInfos = FastTensorInfoProcessor.BuildTensorInfoLookup(graph);

            var newNodes = new List<FastNode>(graph.Nodes.Count);
            var outputRemap = new Dictionary<FastTensorKey, FastTensorKey>();
            bool anyInlined = false;

            foreach (var fastNode in graph.Nodes)
            {
                bool isFunction = fastNode.OpCode == InternalOpCodes.FUNCTION_INVOKE;
                bool isModuleCall = fastNode.OpCode == InternalOpCodes.MODEL_INVOKE;

                if (!isFunction && !isModuleCall)
                {
                    newNodes.Add(fastNode);
                    continue;
                }

                FastComputationGraph subFastGraph;
                var hyperparamNodeKeys = new List<FastTensorKey?>();

                if (isFunction)
                {
                    var targetFunction = fastNode.TargetFunction!;
                    subFastGraph = targetFunction.GetFastFlattenedGraph().Clone();
                    FastProcessorHelper.RekeySubgraph(subFastGraph);
                }
                else
                {
                    // MODULE_INVOKE: inputs[0] = model, inputs[1:] = function args
                    var modelKey = fastNode.Inputs[0]!.Value;

                    // Find the direct MODULE_SET_HYPERPARAMS node producing the model
                    FastNode? directModelCreation = FindDirectModuleCreation(modelKey, nodeByKey);

                    // Get the module function from the tensor info
                    var moduleFn = tensorInfos[modelKey].ModuleFn!;

                    // Skip inlining for non-hyper model-type parameters that only have a type
                    // signature. These MODEL_INVOKE nodes will be resolved later when the
                    // concrete model is substituted during the caller's inlining pass.
                    if (directModelCreation is null && moduleFn.FunctionType == FunctionType.ModuleSignature)
                    {
                        var producingNode = nodeByKey[modelKey.FastNodeKey];
                        bool isHyperModelParam = IsModelInputNode(producingNode) &&
                            producingNode.Attributes.GetEnumVal<InputType>(OnnxOpAttributeNames.ShrkAttrInputType) == InputType.Hyperparam;
                        if (!isHyperModelParam)
                        {
                            newNodes.Add(fastNode);
                            continue;
                        }
                    }

                    subFastGraph = moduleFn.GetFastFlattenedGraph().Clone();
                    FastProcessorHelper.RekeySubgraph(subFastGraph);

                    // Reparent the subgraph
                    if (directModelCreation is not null)
                    {
                        var parentIdTemplate = new ModelParamIdentifierTemplate(directModelCreation.IdentifierTemplate!);
                        var parentModelIdVals = directModelCreation.Attributes.GetIntsVal(OnnxOpAttributeNames.ShrkAttrLocalModelId)!;
                        var parentModelId = new ModelId(parentModelIdVals);
                        var parentIterIndicesKey = directModelCreation.Inputs[1]; // FastTensorKey in main graph

                        FastReparentToCallSite(subFastGraph, parentIdTemplate, parentModelId, parentIterIndicesKey, graph);
                    }
                    else
                    {
                        FastReparentToModelVariable(subFastGraph, modelKey);
                    }

                    // Create MODEL_HYPERPARAM fast nodes for each hyperparam input
                    for (int i = 0; i < moduleFn.HyperparamInputs.Length; i++)
                    {
                        var hyperparam = moduleFn.HyperparamInputs[i];
                        var hyperparamNodeKey = FastNodeKey.New();
                        var hyperparamTensorKey = new FastTensorKey(hyperparamNodeKey, 0);

                        var attrVals = new Dictionary<string, object?>
                        {
                            [OnnxOpAttributeNames.ShrkAttrHyperparamIndex] = (long)i,
                            [OnnxOpAttributeNames.ShrkAttrDtype] = hyperparam.DType,
                            [OnnxOpAttributeNames.ShrkAttrRank] = (int?)hyperparam.Rank
                        };

                        var hyperparamNode = FastNodeCreationHelpers.CreateFastNode(
                            hyperparamNodeKey, InternalOpCodes.MODEL_HYPERPARAM,
                            attrVals, new FastTensorKey?[] { modelKey });

                        newNodes.Add(hyperparamNode);
                        nodeByKey[hyperparamNodeKey] = hyperparamNode;
                        hyperparamNodeKeys.Add(hyperparamTensorKey);
                    }
                }

                // Build caller input keys: [hyperparamKeys..., node.Inputs.Skip(1)...]
                // Skip(1) skips the model variable for MODULE_INVOKE, and matches
                // the non-fast InlineModulesAndFunctions behaviour for FUNCTION_INVOKE.
                var callerInputKeys = new List<FastTensorKey?>();
                callerInputKeys.AddRange(hyperparamNodeKeys);
                callerInputKeys.AddRange(fastNode.Inputs.Skip(1).ToList());

                Debug.Assert(callerInputKeys.Count == subFastGraph.Inputs.Count,
                    $"FastInlineModulesAndFunctions: caller inputs ({callerInputKeys.Count}) != " +
                    $"subgraph inputs ({subFastGraph.Inputs.Count}) for {fastNode.OpCode}");

                // Build input remap: subgraph input key → caller input key
                var inputRemap = new Dictionary<FastTensorKey, FastTensorKey>();
                for (int i = 0; i < subFastGraph.Inputs.Count && i < callerInputKeys.Count; i++)
                {
                    if (callerInputKeys[i] is FastTensorKey callerKey)
                        inputRemap[subFastGraph.Inputs[i]] = callerKey;
                }

                // Insert subgraph nodes with remapped inputs
                foreach (var subNode in subFastGraph.Nodes)
                {
                    foreach (var kvp in subNode.FullInputs)
                    {
                        var inputList = kvp.Value;
                        for (int j = 0; j < inputList.Count; j++)
                        {
                            if (inputList[j] is FastTensorKey key && inputRemap.TryGetValue(key, out var replacement))
                                inputList[j] = replacement;
                        }
                    }
                    newNodes.Add(subNode);
                    nodeByKey[subNode.Key] = subNode;
                }

                // Map invoke node's outputs → subgraph's outputs
                var invokeOutputs = fastNode.Outputs;
                for (int i = 0; i < invokeOutputs.Count && i < subFastGraph.Outputs.Count; i++)
                {
                    if (invokeOutputs[i] is FastTensorKey invokeOutKey)
                        outputRemap[invokeOutKey] = subFastGraph.Outputs[i];
                }

                // Don't add the invoke node itself — it's been replaced by the subgraph.
                anyInlined = true;
            }

            graph.Nodes = newNodes;

            // Apply output remaps to all node inputs and graph outputs
            if (outputRemap.Count > 0)
            {
                // Resolve transitive remap chains
                foreach (var key in outputRemap.Keys.ToList())
                {
                    var target = outputRemap[key];
                    while (outputRemap.TryGetValue(target, out var next))
                        target = next;
                    outputRemap[key] = target;
                }

                for (int i = 0; i < graph.Nodes.Count; i++)
                {
                    foreach (var kvp in graph.Nodes[i].FullInputs)
                    {
                        var inputList = kvp.Value;
                        for (int j = 0; j < inputList.Count; j++)
                        {
                            if (inputList[j] is FastTensorKey key && outputRemap.TryGetValue(key, out var replacement))
                                inputList[j] = replacement;
                        }
                    }
                }

                for (int i = 0; i < graph.Outputs.Count; i++)
                {
                    if (outputRemap.TryGetValue(graph.Outputs[i], out var replacement))
                        graph.Outputs[i] = replacement;
                }
            }

            return anyInlined;
        }

        /// <summary>
        /// Follows Identity chains backwards from a model FastTensorKey to find the
        /// MODULE_SET_HYPERPARAMS node that directly creates the model, if any.
        /// </summary>
        private static FastNode? FindDirectModuleCreation(FastTensorKey modelKey, Dictionary<FastNodeKey, FastNode> nodeByKey)
        {
            var currentKey = modelKey;
            var visited = new HashSet<FastTensorKey>();
            while (visited.Add(currentKey))
            {
                if (!nodeByKey.TryGetValue(currentKey.FastNodeKey, out var node))
                    return null;

                if (node.OpCode == OpCodes.IDENTITY)
                {
                    var input = node.Inputs[0];
                    if (input is null) return null;
                    currentKey = input.Value;
                    continue;
                }

                if (node.OpCode == InternalOpCodes.MODULE_SET_HYPERPARAMS)
                    return node;

                return null;
            }
            return null;
        }

        private static bool IsModelInputNode(FastNode node)
        {
            return node.OpCode == InternalOpCodes.MODEL_TENSOR_INPUT ||
                   node.OpCode == InternalOpCodes.MODEL_OPTIONAL_INPUT ||
                   node.OpCode == InternalOpCodes.MODEL_SEQUENCE_INPUT ||
                   node.OpCode == InternalOpCodes.MODEL_TENSORSTRUCT_INPUT ||
                   node.OpCode == InternalOpCodes.GENERIC_TYPE_INPUT;
        }

        /// <summary>
        /// Native reparenting of a function subgraph to a call site. Updates TRAINABLE_PARAM_REF
        /// and MODULE_SET_HYPERPARAMS nodes in the subgraph: prepends the parent model ID,
        /// composes identifier templates, and combines iteration indices.
        /// </summary>
        private static void FastReparentToCallSite(
            FastComputationGraph subGraph,
            ModelParamIdentifierTemplate parentIdTemplate,
            ModelId parentModelId,
            FastTensorKey? parentIterIndicesKey,
            FastComputationGraph mainGraph)
        {
            var subNodeByKey = new Dictionary<FastNodeKey, FastNode>(subGraph.Nodes.Count);
            foreach (var n in subGraph.Nodes) subNodeByKey[n.Key] = n;

            var nodesToInsert = new List<(int insertBeforeIndex, FastNode node)>();

            for (int i = 0; i < subGraph.Nodes.Count; i++)
            {
                var node = subGraph.Nodes[i];

                if (node.OpCode == InternalOpCodes.TRAINABLE_PARAM_REF)
                {
                    // TRAINABLE_PARAM_REF inputs: [iterationIndices, ...initializerParams]
                    var childIterIndicesKey = node.Inputs[0];

                    var combinedIterIndicesKey = CombineIterationIndices(
                        parentIterIndicesKey, childIterIndicesKey,
                        subGraph, mainGraph, subNodeByKey, nodesToInsert, i);

                    // Update model ID: prepend parent
                    var dctAttributes = node.Attributes.GetAttributeVals().ToDictionary();
                    var childModelIdVals = (long[])dctAttributes[OnnxOpAttributeNames.ShrkAttrLocalModelId]!;
                    var combinedModelId = new ModelId(parentModelId, ModelId.FromLongVals(childModelIdVals));
                    dctAttributes[OnnxOpAttributeNames.ShrkAttrLocalModelId] =
                        combinedModelId.Vals.Select(x => (long)x).ToArray();

                    // Compose identifier templates
                    var childTemplate = new ModelParamIdentifierTemplate(node.IdentifierTemplate!);
                    var combinedTemplate = new ModelParamIdentifierTemplate(parentIdTemplate, childTemplate);

                    node.Attributes = OnnxCSharpAttributes.FromCSharpVals(dctAttributes, node.Attributes.AttributeDefs);
                    node.IdentifierTemplate = combinedTemplate.ToString();

                    // Update iteration indices input
                    var inputs = node.FullInputs[""];
                    inputs[0] = combinedIterIndicesKey;
                }
                else if (node.OpCode == InternalOpCodes.MODULE_SET_HYPERPARAMS)
                {
                    // MODULE_SET_HYPERPARAMS inputs: [inputModule, iterationIndices, ...hyperParams]
                    var childIterIndicesKey = node.Inputs[1];

                    var combinedIterIndicesKey = CombineIterationIndices(
                        parentIterIndicesKey, childIterIndicesKey,
                        subGraph, mainGraph, subNodeByKey, nodesToInsert, i);

                    var dctAttributes = node.Attributes.GetAttributeVals().ToDictionary();
                    var childModelIdVals = (long[])dctAttributes[OnnxOpAttributeNames.ShrkAttrLocalModelId]!;
                    var combinedModelId = new ModelId(parentModelId, ModelId.FromLongVals(childModelIdVals));
                    dctAttributes[OnnxOpAttributeNames.ShrkAttrLocalModelId] =
                        combinedModelId.Vals.Select(x => (long)x).ToArray();

                    var childTemplate = new ModelParamIdentifierTemplate(node.IdentifierTemplate!);
                    var combinedTemplate = new ModelParamIdentifierTemplate(parentIdTemplate, childTemplate);

                    node.Attributes = OnnxCSharpAttributes.FromCSharpVals(dctAttributes, node.Attributes.AttributeDefs);
                    node.IdentifierTemplate = combinedTemplate.ToString();

                    // Update iteration indices input (position 1 in the flat inputs)
                    var inputs = node.FullInputs[""];
                    inputs[1] = combinedIterIndicesKey;
                }
            }

            // Insert new nodes (CONCAT for combined iteration indices) in order
            int offset = 0;
            foreach (var (insertIdx, newNode) in nodesToInsert.OrderBy(x => x.insertBeforeIndex))
            {
                subGraph.Nodes.Insert(insertIdx + offset, newNode);
                offset++;
            }
        }

        /// <summary>
        /// Native reparenting to a model variable. Converts TRAINABLE_PARAM_REF nodes
        /// to TRAINABLE_PARAM_MODEL_REF by prepending the model variable as the first input.
        /// </summary>
        private static void FastReparentToModelVariable(
            FastComputationGraph subGraph, FastTensorKey modelKey)
        {
            for (int i = 0; i < subGraph.Nodes.Count; i++)
            {
                var node = subGraph.Nodes[i];

                if (node.OpCode == InternalOpCodes.TRAINABLE_PARAM_REF)
                {
                    // Current TRAINABLE_PARAM_REF inputs: [iterationIndices, ...initializerParams]
                    // New TRAINABLE_PARAM_MODEL_REF inputs: [model, iterationIndices, ...initializerParams]
                    var currentInputs = node.FullInputs[""];
                    var newInputs = new List<FastTensorKey?> { modelKey };
                    newInputs.AddRange(currentInputs);

                    // Convert LocalModelId → RelativeModelId
                    var dctAttributes = node.Attributes.GetAttributeVals().ToDictionary();
                    var modelIdVals = (long[])dctAttributes[OnnxOpAttributeNames.ShrkAttrLocalModelId]!;
                    dctAttributes.Remove(OnnxOpAttributeNames.ShrkAttrLocalModelId);
                    dctAttributes[OnnxOpAttributeNames.ShrkAttrRelativeModelId] =
                        modelIdVals.Select(x => (int)x).ToArray();

                    var modelRefAttrDefs = Definitions.NodeDefinitions[InternalOpCodes.TRAINABLE_PARAM_MODEL_REF].AttributeDefs;
                    node.Attributes = OnnxCSharpAttributes.FromCSharpVals(dctAttributes, modelRefAttrDefs);
                    node.OpCode = InternalOpCodes.TRAINABLE_PARAM_MODEL_REF;
                    node.FullInputs = new Dictionary<string, List<FastTensorKey?>> { [""] = newInputs };
                }
                else if (node.OpCode == InternalOpCodes.MODULE_SET_HYPERPARAMS)
                {
                    throw new InvalidOperationException(
                        "FastReparentToModelVariable: MODULE_SET_HYPERPARAMS not supported (matches non-fast behaviour).");
                }
            }
        }

        /// <summary>
        /// Combines parent and child iteration-indices TensorKeys. Returns the key to use
        /// for the combined iteration indices. Creates a flat CONCAT FastNode when both
        /// parent and child are non-empty: the new CONCAT's inputs are the union of the
        /// parent CONCAT's inputs and the child CONCAT's inputs, preserving the
        /// <see cref="FastNodeCreationHelpers.GetIterationIndexScalars"/> walker's
        /// invariant that the producing node is a CONCAT whose inputs each resolve to
        /// LOOP_OPEN / CONSTANT after at most one IDENTITY + UNSQUEEZE step. Nesting the
        /// halves as <c>CONCAT(parentCONCAT, childCONCAT)</c> would trip the walker's
        /// Debug.Assert at the inner CONCAT.
        /// </summary>
        private static FastTensorKey? CombineIterationIndices(
            FastTensorKey? parentKey, FastTensorKey? childKey,
            FastComputationGraph subGraph, FastComputationGraph mainGraph,
            Dictionary<FastNodeKey, FastNode> subNodeByKey,
            List<(int, FastNode)> nodesToInsert, int currentIndex)
        {
            bool parentIsEmpty = parentKey is null || IsEmptyConstantVector(parentKey.Value, mainGraph);
            bool childIsEmpty = childKey is null || IsEmptyConstantVector(childKey.Value, subGraph, subNodeByKey);

            if (parentIsEmpty && childIsEmpty)
                return childKey ?? parentKey;

            if (parentIsEmpty) return childKey;
            if (childIsEmpty) return parentKey;

            // Build a flat CONCAT whose inputs are the union of parent's and child's
            // scalar inputs (in that order). Both halves are themselves CONCATs by the
            // GetIterationIndexScalars convention, so naively nesting them would break
            // that walker. After splicing the subgraph into the main graph, every
            // referenced scalar input becomes accessible — parent's inputs are upstream
            // of the splice point, child's are inside the spliced section.
            var combinedInputs = new List<FastTensorKey?>();
            AppendIterIndexInputs(parentKey!.Value, mainGraph, combinedInputs);
            AppendIterIndexInputs(childKey!.Value, subGraph, subNodeByKey, combinedInputs);

            var concatNodeKey = FastNodeKey.New();
            var concatTensorKey = new FastTensorKey(concatNodeKey, 0);
            var concatNode = FastNodeCreationHelpers.CreateFastNode(
                concatNodeKey, OpCodes.CONCAT,
                new Dictionary<string, object?> { [OnnxOpAttributeNames.AttrAxis] = 0L },
                combinedInputs.ToArray());

            nodesToInsert.Add((currentIndex, concatNode));
            return concatTensorKey;
        }

        /// <summary>
        /// Appends the iteration-index scalar inputs feeding the given key into
        /// <paramref name="dest"/>. If the key's producing node is a CONCAT (the normal
        /// case for a non-empty iter-index chain), copies its inputs verbatim;
        /// otherwise (degenerate: UNSQUEEZE / IDENTITY / LOOP_OPEN / CONSTANT producing
        /// the key directly) appends the key itself so the new CONCAT references it as
        /// a single scalar slot.
        /// </summary>
        private static void AppendIterIndexInputs(FastTensorKey key, FastComputationGraph graph, List<FastTensorKey?> dest)
        {
            foreach (var n in graph.Nodes)
            {
                if (n.Key != key.FastNodeKey) continue;
                if (n.OpCode == OpCodes.CONCAT)
                {
                    foreach (var input in n.Inputs) dest.Add(input);
                    return;
                }
                break;
            }
            dest.Add(key);
        }

        private static void AppendIterIndexInputs(FastTensorKey key, FastComputationGraph graph,
            Dictionary<FastNodeKey, FastNode> nodeByKey, List<FastTensorKey?> dest)
        {
            if (nodeByKey.TryGetValue(key.FastNodeKey, out var n) && n.OpCode == OpCodes.CONCAT)
            {
                foreach (var input in n.Inputs) dest.Add(input);
                return;
            }
            dest.Add(key);
        }

        private static bool IsEmptyConstantVector(FastTensorKey key, FastComputationGraph graph)
        {
            // Check via tensor info or by finding the producing node in the graph
            for (int i = 0; i < graph.Nodes.Count; i++)
            {
                var n = graph.Nodes[i];
                if (n.Key == key.FastNodeKey && n.OpCode == OpCodes.CONSTANT)
                {
                    var tensorVal = n.Attributes.GetTensorVal(OnnxOpAttributeNames.AttrValue);
                    return tensorVal != null && tensorVal.Shape.Count == 0;
                }
            }
            return false;
        }

        private static bool IsEmptyConstantVector(FastTensorKey key, FastComputationGraph graph, Dictionary<FastNodeKey, FastNode> nodeByKey)
        {
            if (!nodeByKey.TryGetValue(key.FastNodeKey, out var node)) return false;
            if (node.OpCode != OpCodes.CONSTANT) return false;
            var tensorVal = node.Attributes.GetTensorVal(OnnxOpAttributeNames.AttrValue);
            return tensorVal != null && tensorVal.Shape.Count == 0;
        }
    }

    /// <summary>
    /// Native fast-graph processor that collects identifier templates from FastNodes
    /// without converting to <c>ComputationGraph</c>.
    /// </summary>
    internal static class FastExtractIdentifierTemplates
    {
        public struct IdentifierTemplateInfos
        {
            public readonly ImmutableDictionary<ModelId, ModelParamIdentifierTemplate> FullTemplates { get; init; }
            public readonly ImmutableDictionary<ModelId, ModelParamIdentifierTemplate> RelativeTemplates { get; init; }
            public readonly ImmutableDictionary<ModelId, ModelParamIdentifierTemplate> BaseModuleTemplates { get; init; }
            public readonly ImmutableDictionary<ModelId, ModelParamIdentifierTemplate> RelativeModuleTemplates { get; init; }
        }

        public static IdentifierTemplateInfos Process(FastComputationGraph graph)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));

            var dctFullTemplates = new Dictionary<ModelId, ModelParamIdentifierTemplate>();
            var dctRelativeTemplates = new Dictionary<ModelId, ModelParamIdentifierTemplate>();
            var dctBaseModuleTemplates = new Dictionary<ModelId, ModelParamIdentifierTemplate>();
            var dctRelativeModuleTemplates = new Dictionary<ModelId, ModelParamIdentifierTemplate>();

            foreach (var fastNode in graph.Nodes)
            {
                if (fastNode.IdentifierTemplate is null)
                    continue;

                if (fastNode.OpCode == InternalOpCodes.MODULE_SET_HYPERPARAMS)
                {
                    var idTemplate = new ModelParamIdentifierTemplate(fastNode.IdentifierTemplate).ToGeneralizedTemplate();
                    dctBaseModuleTemplates[idTemplate.ModelIdTemplate] = idTemplate;
                }
                else if (fastNode.OpCode == InternalOpCodes.TRAINABLE_PARAM_REF)
                {
                    var idTemplate = new ModelParamIdentifierTemplate(fastNode.IdentifierTemplate).ToGeneralizedTemplate();
                    dctFullTemplates[idTemplate.ModelIdTemplate] = idTemplate;
                }
                else if (fastNode.OpCode == InternalOpCodes.TRAINABLE_PARAM_MODEL_REF)
                {
                    var idTemplate = new ModelParamIdentifierTemplate(fastNode.IdentifierTemplate).ToGeneralizedTemplate();
                    dctRelativeTemplates[idTemplate.ModelIdTemplate] = idTemplate;
                }
            }

            return new IdentifierTemplateInfos
            {
                FullTemplates = dctFullTemplates.ToImmutableDictionary(),
                RelativeTemplates = dctRelativeTemplates.ToImmutableDictionary(),
                BaseModuleTemplates = dctBaseModuleTemplates.ToImmutableDictionary(),
                RelativeModuleTemplates = dctRelativeModuleTemplates.ToImmutableDictionary()
            };
        }
    }

    /// <summary>
    /// Converts TRAINABLE_PARAM_REF and TRAINABLE_PARAM_MODEL_REF nodes into
    /// TRAINABLE_PARAM_ID_REF nodes by building model-ID vectors from iteration indices
    /// and identifier templates. New CONSTANT, UNSQUEEZE, CONCAT and GET_MODEL_ID FastNodes
    /// are created and inserted in topological order.
    /// </summary>
    internal static class FastConvertToIdRefTrainableParams
    {
        public static void Process(FastComputationGraph graph)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));

            int unprocessedCount = CountUnprocessed(graph);
            while (unprocessedCount > 0)
            {
                InternalProcess(graph);
                int newCount = CountUnprocessed(graph);
                if (newCount >= unprocessedCount)
                    throw new InvalidOperationException(
                        $"FastConvertToIdRefTrainableParams: no progress. {newCount} TRAINABLE_PARAM_REF/MODEL_REF remain.");
                unprocessedCount = newCount;
            }
        }

        private static int CountUnprocessed(FastComputationGraph graph)
            => graph.Nodes.Count(n =>
                n.OpCode == InternalOpCodes.TRAINABLE_PARAM_REF ||
                n.OpCode == InternalOpCodes.TRAINABLE_PARAM_MODEL_REF);

        private static void InternalProcess(FastComputationGraph graph)
        {
            var nodeByKey = FastProcessorHelper.BuildNodeByKey(graph);

            var newNodes = new List<FastNode>(graph.Nodes.Count + 64);

            foreach (var fastNode in graph.Nodes)
            {
                if (fastNode.OpCode == InternalOpCodes.TRAINABLE_PARAM_REF)
                {
                    ProcessTrainableParamRef(fastNode, nodeByKey, newNodes);
                }
                else if (fastNode.OpCode == InternalOpCodes.TRAINABLE_PARAM_MODEL_REF)
                {
                    ProcessTrainableParamModelRef(fastNode, nodeByKey, newNodes);
                }

                newNodes.Add(fastNode);
            }

            graph.Nodes = newNodes;
        }

        private static void ProcessTrainableParamRef(
            FastNode fastNode,
            Dictionary<FastNodeKey, FastNode> nodeByKey, List<FastNode> newNodes)
        {
            // TRAINABLE_PARAM_REF inputs: [iterationIndices, ...initializerParams]
            var inputs = fastNode.Inputs;
            var iterationIndicesKey = inputs[0]; // may be null

            // Get iteration index TensorKeys (the scalar LOOP_OPEN outputs).
            var iterationIndexKeys = GetIterationIndexScalars(iterationIndicesKey, nodeByKey);

            // Parse IdentifierTemplate to get the ModelIdTemplate.
            var idTemplate = new ModelParamIdentifierTemplate(fastNode.IdentifierTemplate!);
            var genericModelId = idTemplate.ModelIdTemplate;

            // Build the specific model-ID vector as FastNodes.
            var fullModelIdKey = BuildSpecificModelIdFastNodes(
                genericModelId, iterationIndexKeys, baseModelIdKey: null,
                newNodes);

            // Build new attributes: remove ShrkAttrLocalModelId, switch to ID_REF defs.
            var dctAttributes = fastNode.Attributes.GetAttributeVals().ToDictionary();
            dctAttributes.Remove(OnnxOpAttributeNames.ShrkAttrLocalModelId);
            var idRefAttrDefs = Definitions.NodeDefinitions[InternalOpCodes.TRAINABLE_PARAM_ID_REF].AttributeDefs;
            var newAttributes = OnnxCSharpAttributes.FromCSharpVals(dctAttributes, idRefAttrDefs);

            // TRAINABLE_PARAM_ID_REF inputs: [modelId, iterationIndices, ...initializerParams]
            var initializerParamKeys = inputs.Skip(1).ToList();
            var newInputs = new List<FastTensorKey?> { fullModelIdKey };
            newInputs.Add(iterationIndicesKey);
            newInputs.AddRange(initializerParamKeys);

            // Mutate in place.
            fastNode.OpCode = InternalOpCodes.TRAINABLE_PARAM_ID_REF;
            fastNode.Attributes = newAttributes;
            fastNode.FullInputs = new Dictionary<string, List<FastTensorKey?>> { [""] = newInputs };
        }

        private static void ProcessTrainableParamModelRef(
            FastNode fastNode,
            Dictionary<FastNodeKey, FastNode> nodeByKey, List<FastNode> newNodes)
        {
            // TRAINABLE_PARAM_MODEL_REF inputs: [model, iterationIndices, ...initializerParams]
            var inputs = fastNode.Inputs;
            var modelKey = inputs[0]!.Value;
            var iterationIndicesKey = inputs[1];

            // Create GET_MODEL_ID node for the model variable → Vector<int64> base model ID.
            var getModelIdKey = FastNodeKey.New();
            var baseModelIdTensorKey = new FastTensorKey(getModelIdKey, 0);
            var getModelIdNode = FastNodeCreationHelpers.CreateFastNode(getModelIdKey, InternalOpCodes.GET_MODEL_ID,
                new Dictionary<string, object?>(), new[] { (FastTensorKey?)modelKey });
            newNodes.Add(getModelIdNode);

            // Get iteration index scalars.
            var iterationIndexKeys = GetIterationIndexScalars(iterationIndicesKey, nodeByKey);

            // Parse IdentifierTemplate to get the relative model ID template.
            var relativeTemplate = fastNode.IdentifierTemplate is not null
                ? new ModelParamIdentifierTemplate(fastNode.IdentifierTemplate)
                : null;
            var genericModelId = relativeTemplate?.ModelIdTemplate ?? new ModelId();

            // Build full model-ID vector, prepending the base model ID.
            var fullModelIdKey = BuildSpecificModelIdFastNodes(
                genericModelId, iterationIndexKeys, baseModelIdTensorKey,
                newNodes);

            // Compose identifier template: base (from model's owning node) + relative.
            var modelOwnerNode = nodeByKey[modelKey.FastNodeKey];
            var baseTemplateObj = modelOwnerNode.IdentifierTemplate is not null
                ? new ModelParamIdentifierTemplate(modelOwnerNode.IdentifierTemplate)
                : null;
            ModelParamIdentifierTemplate? fullTemplate =
                (baseTemplateObj is not null && relativeTemplate is not null)
                    ? new ModelParamIdentifierTemplate(baseTemplateObj, relativeTemplate)
                    : relativeTemplate;

            // Build new attributes: remove ShrkAttrRelativeModelId, switch to ID_REF defs.
            var dctAttributes = fastNode.Attributes.GetAttributeVals().ToDictionary();
            dctAttributes.Remove(OnnxOpAttributeNames.ShrkAttrRelativeModelId);
            var idRefAttrDefs = Definitions.NodeDefinitions[InternalOpCodes.TRAINABLE_PARAM_ID_REF].AttributeDefs;
            var newAttributes = OnnxCSharpAttributes.FromCSharpVals(dctAttributes, idRefAttrDefs);

            // TRAINABLE_PARAM_ID_REF inputs: [modelId, iterationIndices, ...initializerParams]
            var initializerParamKeys = inputs.Skip(2).ToList();
            var newInputs = new List<FastTensorKey?> { fullModelIdKey };
            newInputs.Add(iterationIndicesKey);
            newInputs.AddRange(initializerParamKeys);

            // Mutate in place.
            fastNode.OpCode = InternalOpCodes.TRAINABLE_PARAM_ID_REF;
            fastNode.Attributes = newAttributes;
            fastNode.IdentifierTemplate = fullTemplate?.ToString();
            fastNode.FullInputs = new Dictionary<string, List<FastTensorKey?>> { [""] = newInputs };
        }

        // Helper methods delegated to shared FastNodeCreationHelpers.

        private static ImmutableArray<FastTensorKey> GetIterationIndexScalars(
            FastTensorKey? iterationIndicesKey, Dictionary<FastNodeKey, FastNode> nodeByKey)
            => FastNodeCreationHelpers.GetIterationIndexScalars(iterationIndicesKey, nodeByKey);

        private static FastTensorKey BuildSpecificModelIdFastNodes(
            ModelId genericModelId,
            ImmutableArray<FastTensorKey> iterationIndexKeys, FastTensorKey? baseModelIdKey,
            List<FastNode> newNodes)
            => FastNodeCreationHelpers.BuildSpecificModelIdFastNodes(genericModelId, iterationIndexKeys, baseModelIdKey, newNodes);
    }

    /// <summary>
    /// Unpacks the Model struct entirely on <see cref="FastComputationGraph"/>:
    /// MODULE_SET_HYPERPARAMS / MODEL_HYPERPARAM / GET_MODEL_ID / IDENTITY for Model
    /// pass-through, plus per-field parallel SEQUENCE_* and LOOP_OPEN / LOOP_CLOSE /
    /// IF_CLOSE expansion for Model tensors carried through sequences and control flow.
    /// Works for every hyperparam field layout (Tensor / Optional / Sequence) because the
    /// emitted per-field SEQUENCE_* ops just carry the field's original tensor key — no
    /// IfElse / OptionalHasElement / SequenceSlice gadgets needed.
    /// </summary>
    internal static class FastUnpackModelStruct
    {
        public static void Process(FastComputationGraph graph)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));

            // Non-Tensor-layout hyperparam fields (Optional / Sequence) don't need any
            // special gadgets on the Fast path: the per-field parallel SEQUENCE_CONSTRUCT /
            // SEQUENCE_AT we already emit happily carry Optional- and Sequence-typed tensor
            // keys as their element type. (Formerly gated to the CG round-trip as "B1a"; see
            // TestOptionalHypersSequenceCalled / TestSeqHypersSequenceCalled for coverage.)

            // Shrink every loop scope before the native pass. Any node that is
            // positionally between a LOOP_OPEN/LOOP_CLOSE pair but does not transitively
            // depend on the loop's body outputs is hoisted to just before the OPEN. This
            // matches CG's implicit HoistLoopInvariantNodes and prevents QEE from
            // attaching per-iteration History to loop-invariant tensor values — the
            // positional pathology behind the former B1b fall-back.
            FastScopeHelper.ShrinkAllScopes(graph);

            // Run the native pass. We walk the graph in topological
            // order; each handler records its generated nodes on ctx.NewNodes, which we
            // snapshot before/after each dispatch so we can interleave the new nodes into
            // the final list at the position of the node that produced them — preserving
            // topological order for downstream round-trips (FastTensorInfoProcessor asserts
            // it).
            var ctx = new FastModelStructContext(graph);
            var originalNodes = graph.Nodes;
            var finalNodes = new List<FastNode>(originalNodes.Count + 32);

            foreach (var fastNode in originalNodes)
            {
                int newNodesBefore = ctx.NewNodes.Count;
                DispatchNode(fastNode, ctx);
                int newNodesAfter = ctx.NewNodes.Count;

                if (!ctx.NodesToRemove.Contains(fastNode.Key))
                    finalNodes.Add(fastNode);

                // Some handlers (MODULE_SET_HYPERPARAMS / SEQUENCE_AT / ...) append
                // producer nodes (e.g. the per-field SEQUENCE_AT or UNSQUEEZE / CONCAT for
                // model-ID building) that must appear before the mutated node's downstream
                // consumers. Since handlers append *after* the original node in our final
                // list, downstream ops still see the producers in topological order.
                //
                // For LOOP_OPEN / LOOP_CLOSE / IF_CLOSE (in-place mutations) no new nodes
                // are emitted — only the node's own inputs/outputs change — so this block
                // is a no-op for them.
                for (int i = newNodesBefore; i < newNodesAfter; i++)
                    finalNodes.Add(ctx.NewNodes[i]);
            }

            ApplyRemapAndCommit(graph, ctx, finalNodes);
        }

        private static void DispatchNode(FastNode fastNode, FastModelStructContext ctx)
        {
            switch (fastNode.OpCode)
            {
                case InternalOpCodes.MODULE_SET_HYPERPARAMS:
                    HandleModuleSetHyperparams(fastNode, ctx);
                    return;

                case InternalOpCodes.MODEL_HYPERPARAM:
                    HandleModelHyperparam(fastNode, ctx);
                    return;

                case InternalOpCodes.GET_MODEL_ID:
                    HandleGetModelId(fastNode, ctx);
                    return;

                case OpCodes.IDENTITY:
                    HandleIdentity(fastNode, ctx);
                    return;

                case OpCodes.SEQUENCE_EMPTY:
                    if (fastNode.TargetFunction is not null)
                        FastModelSequenceOpHandlers.HandleSequenceEmpty(fastNode, ctx);
                    return;

                case OpCodes.SEQUENCE_CONSTRUCT:
                    if (AnyInputIsUnpackedStruct(fastNode, ctx))
                        FastModelSequenceOpHandlers.HandleSequenceConstruct(fastNode, ctx);
                    return;

                case OpCodes.SEQUENCE_AT:
                    if (InputIsUnpackedStructSequence(fastNode, 0, ctx))
                        FastModelSequenceOpHandlers.HandleSequenceAt(fastNode, ctx);
                    return;

                case OpCodes.SEQUENCE_ERASE:
                    if (InputIsUnpackedStructSequence(fastNode, 0, ctx))
                        FastModelSequenceOpHandlers.HandleSequenceErase(fastNode, ctx);
                    return;

                case OpCodes.SEQUENCE_INSERT:
                    if (InputIsUnpackedStructSequence(fastNode, 0, ctx))
                        FastModelSequenceOpHandlers.HandleSequenceInsert(fastNode, ctx);
                    return;

                case OpCodes.SEQUENCE_LENGTH:
                    if (InputIsUnpackedStructSequence(fastNode, 0, ctx))
                        FastModelSequenceOpHandlers.HandleSequenceLength(fastNode, ctx);
                    return;

                case OpCodes.LOOP_OPEN:
                    if (AnyVariadicInputIsUnpackedModel(fastNode, variadicStart: 2, ctx))
                        FastModelControlFlowOpHandlers.HandleLoopOpen(fastNode, ctx);
                    return;

                case OpCodes.LOOP_CLOSE:
                    if (LoopCloseHasUnpackedLoopVar(fastNode, ctx))
                        FastModelControlFlowOpHandlers.HandleLoopClose(fastNode, ctx);
                    return;

                case OpCodes.IF_CLOSE:
                    if (IfCloseHasUnpackedBranchSlot(fastNode, ctx))
                        FastModelControlFlowOpHandlers.HandleIfClose(fastNode, ctx);
                    return;
            }
        }

        private static void HandleModuleSetHyperparams(FastNode fastNode, FastModelStructContext ctx)
        {
            // Inputs: [inputModule, iterationIndices, ...hyperParams]
            var inputs = fastNode.Inputs;
            var iterationIndicesKey = inputs[1];
            var hyperParamKeys = inputs.Skip(2).ToList();

            var iterationIndexScalars = FastNodeCreationHelpers.GetIterationIndexScalars(
                iterationIndicesKey, ctx.NodeByKey);
            var idTemplate = new ModelParamIdentifierTemplate(fastNode.IdentifierTemplate!);
            var genericModelId = idTemplate.ModelIdTemplate;
            var modelIdNewNodes = new List<FastNode>();
            var modelIdKey = FastNodeCreationHelpers.BuildSpecificModelIdFastNodes(
                genericModelId, iterationIndexScalars, baseModelIdKey: null, modelIdNewNodes);

            foreach (var n in modelIdNewNodes) ctx.RecordNewNode(n);

            var structFields = new List<FastTensorKey?> { iterationIndicesKey };
            structFields.AddRange(hyperParamKeys);
            structFields.Add(modelIdKey);

            var outputKey = fastNode.Outputs[0]!.Value;
            ctx.UnpackedStructs[outputKey] = structFields;
            ctx.NodesToRemove.Add(fastNode.Key);
        }

        private static void HandleModelHyperparam(FastNode fastNode, FastModelStructContext ctx)
        {
            var modelInputKey = fastNode.Inputs[0]!.Value;
            var resolvedKey = ctx.ResolveModelKey(modelInputKey);
            var hyperparamIndex = (int)fastNode.Attributes.GetLongVal(
                OnnxOpAttributeNames.ShrkAttrHyperparamIndex)!.Value;

            if (ctx.UnpackedStructs.TryGetValue(resolvedKey, out var fields))
            {
                // Skip iterationIndices at index 0: field at hyperparamIndex + 1
                var fieldKey = fields[hyperparamIndex + 1];
                if (fieldKey is not null)
                {
                    var outputKey = fastNode.Outputs[0]!.Value;
                    ctx.Remap[outputKey] = fieldKey.Value;
                    ctx.NodesToRemove.Add(fastNode.Key);
                }
            }
        }

        private static void HandleGetModelId(FastNode fastNode, FastModelStructContext ctx)
        {
            var modelInputKey = fastNode.Inputs[0]!.Value;
            var resolvedKey = ctx.ResolveModelKey(modelInputKey);

            if (ctx.UnpackedStructs.TryGetValue(resolvedKey, out var fields))
            {
                var modelIdKey = fields.Last();
                if (modelIdKey is not null)
                {
                    var outputKey = fastNode.Outputs[0]!.Value;
                    ctx.Remap[outputKey] = modelIdKey.Value;
                    ctx.NodesToRemove.Add(fastNode.Key);
                }
            }
        }

        private static void HandleIdentity(FastNode fastNode, FastModelStructContext ctx)
        {
            var inputKey = fastNode.Inputs[0];
            if (inputKey is null) return;

            var resolvedKey = ctx.ResolveModelKey(inputKey.Value);
            if (ctx.UnpackedStructs.TryGetValue(resolvedKey, out var fields))
            {
                var outputKey = fastNode.Outputs[0]!.Value;
                ctx.UnpackedStructs[outputKey] = fields;
                ctx.NodesToRemove.Add(fastNode.Key);
            }
            else if (ctx.UnpackedStructSequences.TryGetValue(resolvedKey, out var seqs))
            {
                var outputKey = fastNode.Outputs[0]!.Value;
                ctx.UnpackedStructSequences[outputKey] = seqs;
                ctx.NodesToRemove.Add(fastNode.Key);
            }
        }

        private static bool AnyInputIsUnpackedStruct(FastNode node, FastModelStructContext ctx)
        {
            foreach (var inputKey in node.Inputs)
            {
                if (inputKey is null) continue;
                var resolved = ctx.ResolveModelKey(inputKey.Value);
                if (ctx.UnpackedStructs.ContainsKey(resolved)) return true;
            }
            return false;
        }

        private static bool InputIsUnpackedStructSequence(FastNode node, int slot, FastModelStructContext ctx)
        {
            if (node.Inputs.Count <= slot) return false;
            var key = node.Inputs[slot];
            if (key is null) return false;
            var resolved = ctx.ResolveModelKey(key.Value);
            return ctx.UnpackedStructSequences.ContainsKey(resolved);
        }

        private static bool AnyVariadicInputIsUnpackedModel(
            FastNode node, int variadicStart, FastModelStructContext ctx)
        {
            var inputs = node.Inputs;
            for (int i = variadicStart; i < inputs.Count; i++)
            {
                var key = inputs[i];
                if (key is null) continue;
                var resolved = ctx.ResolveModelKey(key.Value);
                if (ctx.UnpackedStructs.ContainsKey(resolved) ||
                    ctx.UnpackedStructSequences.ContainsKey(resolved))
                    return true;
            }
            return false;
        }

        private static bool LoopCloseHasUnpackedLoopVar(FastNode node, FastModelStructContext ctx)
        {
            if (node.GraphOpenNodeKey is not FastNodeKey openKey) return false;
            // If the open node got expanded we'll have stashed its pre-expansion variadic
            // count; that's a direct signal that this close also needs expansion.
            int originalLoopVarCount;
            try { originalLoopVarCount = ctx.GetLoopOpenOriginalVariadicCount(openKey); }
            catch { return false; }

            var bodyInputs = node.FullInputs.TryGetValue(OnnxOpAttributeNames.AttrBody, out var b)
                ? b : null;
            if (bodyInputs is null) return false;

            for (int i = 0; i < originalLoopVarCount; i++)
            {
                var key = bodyInputs[1 + i];
                if (key is null) continue;
                var resolved = ctx.ResolveModelKey(key.Value);
                if (ctx.UnpackedStructs.ContainsKey(resolved) ||
                    ctx.UnpackedStructSequences.ContainsKey(resolved))
                    return true;
            }
            return false;
        }

        private static bool IfCloseHasUnpackedBranchSlot(FastNode node, FastModelStructContext ctx)
        {
            if (!node.FullInputs.TryGetValue(OnnxOpAttributeNames.AttrElseBranch, out var elseInputs))
                return false;
            foreach (var key in elseInputs)
            {
                if (key is null) continue;
                var resolved = ctx.ResolveModelKey(key.Value);
                if (ctx.UnpackedStructs.ContainsKey(resolved) ||
                    ctx.UnpackedStructSequences.ContainsKey(resolved))
                    return true;
            }
            return false;
        }

        private static void ApplyRemapAndCommit(
            FastComputationGraph graph, FastModelStructContext ctx, List<FastNode> finalNodes)
        {
            // Resolve transitive remap chains: if A→B and B→C, collapse to A→C.
            foreach (var key in ctx.Remap.Keys.ToList())
            {
                var target = ctx.Remap[key];
                while (ctx.Remap.TryGetValue(target, out var next)) target = next;
                ctx.Remap[key] = target;
            }

            foreach (var node in finalNodes)
            {
                foreach (var kvp in node.FullInputs)
                {
                    var inputList = kvp.Value;
                    for (int j = 0; j < inputList.Count; j++)
                    {
                        if (inputList[j] is FastTensorKey key && ctx.Remap.TryGetValue(key, out var replacement))
                            inputList[j] = replacement;
                    }
                }
            }

            for (int i = 0; i < graph.Outputs.Count; i++)
            {
                if (ctx.Remap.TryGetValue(graph.Outputs[i], out var replacement))
                    graph.Outputs[i] = replacement;
            }

            graph.Nodes = finalNodes;
        }
    }

    /// <summary>
    /// Lowers TensorStruct operations on a <see cref="FastComputationGraph"/>, handling:
    /// <list type="bullet">
    ///   <item><c>MODEL_TENSORSTRUCT_INPUT</c> (B2a): expanded into per-field
    ///     <c>MODEL_TENSOR_INPUT</c> nodes.</item>
    ///   <item><c>TENSOR_STRUCT_CREATE</c> / <c>TENSOR_STRUCT_GETFIELD</c>: rewired
    ///     via a per-field unpack map, no CG round-trip.</item>
    ///   <item><c>SEQUENCE_EMPTY</c> / <c>SEQUENCE_CONSTRUCT</c> /
    ///     <c>SEQUENCE_AT</c> / <c>SEQUENCE_ERASE</c> / <c>SEQUENCE_INSERT</c> /
    ///     <c>SEQUENCE_LENGTH</c> carrying TensorStruct-typed I/O (B2b):
    ///     expanded into per-field parallel sequence ops.</item>
    ///   <item><c>LOOP_OPEN</c> / <c>LOOP_CLOSE</c> / <c>IF_CLOSE</c> carrying
    ///     TensorStruct-typed loop-vars / branch slots (B2b): each slot
    ///     expanded into its parallel per-field slots.</item>
    /// </list>
    /// Non-Tensor field structures (Optional / Sequence / nested TensorStruct) are
    /// carried through the per-field parallel <c>SEQUENCE_*</c> ops by their
    /// element type — same approach <see cref="FastUnpackModelStruct"/> takes for
    /// non-Tensor hyperparam fields.
    /// </summary>
    internal static class FastUnpackTensorStructs
    {
        public static void Process(FastComputationGraph graph)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));

            // Op-code-only pre-scan. A graph that is entirely TensorStruct-free can skip
            // both the native pass and the expensive BuildTensorInfoLookup (which itself
            // round-trips via FastComputationGraphConverter.ToComputationGraph). The three
            // op codes below are the only ways a TensorStruct can enter the graph:
            //   - TENSOR_STRUCT_CREATE / TENSOR_STRUCT_GETFIELD produce / consume one,
            //   - MODEL_TENSORSTRUCT_INPUT is a TensorStruct graph input.
            bool hasCreate = false, hasGetField = false, hasStructInput = false;

            for (int i = 0; i < graph.Nodes.Count; i++)
            {
                var opCode = graph.Nodes[i].OpCode;
                if (opCode == InternalOpCodes.TENSOR_STRUCT_CREATE) hasCreate = true;
                else if (opCode == InternalOpCodes.TENSOR_STRUCT_GETFIELD) hasGetField = true;
                else if (opCode == InternalOpCodes.MODEL_TENSORSTRUCT_INPUT) hasStructInput = true;
            }

            // No TensorStruct node of any kind — nothing for this processor to do, and
            // no need to pay for BuildTensorInfoLookup.
            if (!hasCreate && !hasGetField && !hasStructInput)
                return;

            // Build FastTensorKey → TensorStructDef map directly from producer nodes'
            // AttrDtype attribute. Both TENSOR_STRUCT_CREATE and MODEL_TENSORSTRUCT_INPUT
            // carry the struct's DType on AttrDtype; reading from the producer node is
            // free and avoids a BuildTensorInfoLookup round-trip for the GETFIELD
            // field-index resolution below.
            var structDtypeByKey = new Dictionary<FastTensorKey, DType>();
            foreach (var node in graph.Nodes)
            {
                if (node.OpCode != InternalOpCodes.TENSOR_STRUCT_CREATE &&
                    node.OpCode != InternalOpCodes.MODEL_TENSORSTRUCT_INPUT)
                    continue;
                if (node.Outputs.Count == 0 || node.Outputs[0] is not FastTensorKey ok || ok.IsEmpty) continue;
                var dtype = node.Attributes.GetDTypeVal(OnnxOpAttributeNames.AttrDtype);
                if (dtype is null) continue;
                structDtypeByKey[ok] = dtype;
            }

            // --- Native in-place implementation for simple CREATE/GETFIELD patterns ---

            // Maps a TENSOR_STRUCT_CREATE output FastTensorKey → ordered list of field input TensorKeys.
            var structFields = new Dictionary<FastTensorKey, List<FastTensorKey?>>();

            // Maps IDENTITY output → what it points to, for following through Identity chains.
            var identityMap = new Dictionary<FastTensorKey, FastTensorKey>();

            // FastTensorKey remap: GETFIELD output → the actual field's FastTensorKey.
            var remap = new Dictionary<FastTensorKey, FastTensorKey>();

            // Nodes to remove (CREATE, GETFIELD, and struct-input nodes become dead after rewiring).
            var nodesToRemove = new HashSet<FastNodeKey>();

            // B2a native unpacking: expand every `MODEL_TENSORSTRUCT_INPUT` node in
            // the graph's inputs into one `MODEL_TENSOR_INPUT` per struct field, then
            // register the struct output's FastTensorKey → field-input-FastTensorKey list in
            // `structFields` so the existing GETFIELD resolution (second pass below)
            // rewires consumers to the correct field input.
            if (hasStructInput)
            {
                // Build a FastTensorKey → producer lookup restricted to struct-input
                // producers so we can flag graph inputs that come from a
                // MODEL_TENSORSTRUCT_INPUT node and find that node's attributes.
                var structInputProducerByOutput = new Dictionary<FastTensorKey, FastNode>();
                foreach (var node in graph.Nodes)
                {
                    if (node.OpCode != InternalOpCodes.MODEL_TENSORSTRUCT_INPUT) continue;
                    foreach (var kvp in node.FullOutputs)
                        foreach (var ok in kvp.Value)
                            if (ok is FastTensorKey tk && !tk.IsEmpty)
                                structInputProducerByOutput[tk] = node;
                }

                var newGraphInputs = new List<FastTensorKey>(graph.Inputs.Count);
                var newNodes = new List<FastNode>();

                foreach (var inputKey in graph.Inputs)
                {
                    if (!structInputProducerByOutput.TryGetValue(inputKey, out var structInputNode))
                    {
                        newGraphInputs.Add(inputKey);
                        continue;
                    }

                    // Lookup the struct's TensorStructDef from its DType attribute.
                    var structDType = structInputNode.Attributes.GetDTypeVal(OnnxOpAttributeNames.AttrDtype);
                    var structDef = structDType?.TensorStructDef;
                    Debug.Assert(structDef is not null,
                        "FastUnpackTensorStructs.Process: MODEL_TENSORSTRUCT_INPUT producer is missing "
                        + "a TensorStructDef on its AttrDtype. Framework emission paths always populate "
                        + "this — a null structDef here means the node was constructed by hand or its "
                        + "DType lost its TensorStructDef somewhere upstream.");

                    // One MODEL_TENSOR_INPUT FastNode per field. Matches CG's
                    // `InternalOp.RuntimeInput(field.ElementType, field.Rank, defaultName)`.
                    var fieldKeys = new List<FastTensorKey?>(structDef.Fields.Length);
                    var tensorInputDefs = Definitions.NodeDefinitions[InternalOpCodes.MODEL_TENSOR_INPUT].AttributeDefs;
                    foreach (var field in structDef.Fields)
                    {
                        var fieldNodeKey = FastNodeKey.New();
                        var fieldTensorKey = new FastTensorKey(fieldNodeKey, 0);
                        var fieldAttrs = OnnxCSharpAttributes.FromCSharpVals(
                            new Dictionary<string, object?>
                            {
                                [OnnxOpAttributeNames.AttrDtype] = field.ElementType,
                                [OnnxOpAttributeNames.ShrkAttrRank] = (long?)field.Rank,
                            },
                            tensorInputDefs);
                        var fieldNode = new FastNode
                        {
                            Key = fieldNodeKey,
                            OpCode = InternalOpCodes.MODEL_TENSOR_INPUT,
                            Attributes = fieldAttrs,
                            FullOutputs = { [""] = new List<FastTensorKey?> { fieldTensorKey } },
                        };
                        newNodes.Add(fieldNode);
                        fieldKeys.Add(fieldTensorKey);
                        newGraphInputs.Add(fieldTensorKey);
                    }

                    structFields[inputKey] = fieldKeys;
                    nodesToRemove.Add(structInputNode.Key);
                }

                // Insert field-input nodes at the front of graph.Nodes. Their
                // outputs are graph inputs (no producer edges visible to
                // `EnsureTopologicalOrder` because graph inputs are treated as
                // pre-available), so placing them at the top keeps graph.Nodes
                // in a topologically-valid order without needing a re-sort
                // pass just for the new MODEL_TENSOR_INPUT entries.
                graph.Nodes.InsertRange(0, newNodes);
                graph.Inputs.Clear();
                foreach (var k in newGraphInputs) graph.Inputs.Add(k);
            }

            // Maps a TensorStruct-typed sequence's output FastTensorKey to the list of
            // per-field parallel sequence TensorKeys. Populated by
            // SEQUENCE_EMPTY/CONSTRUCT/ERASE/INSERT handlers and consumed by
            // SEQUENCE_AT/ERASE/INSERT/LENGTH + LOOP/IF expansion handlers.
            var unpackedStructSequences = new Dictionary<FastTensorKey, List<FastTensorKey>>();

            // LOOP_OPEN.Key → original (pre-expansion) variadic loop-var count. The
            // matching LOOP_CLOSE handler reads this to recover the loopVar /
            // scanVar split from its own still-pre-expansion body inputs.
            var loopOpenOriginalVariadicCounts = new Dictionary<FastNodeKey, int>();

            // New nodes produced by per-op handlers (SEQUENCE_* per-field expansions,
            // SEQUENCE_LENGTH CONSTANT replacements), keyed by the origin node they
            // were emitted for. At the end of the pass we splice each origin's
            // extras right after the origin in the new node list, which keeps the
            // graph topologically ordered + properly nested by construction.
            var extrasByOriginNode = new Dictionary<FastNodeKey, List<FastNode>>();
            FastNodeKey currentOriginKey = default;
            void AddExtra(FastNode n)
            {
                if (!extrasByOriginNode.TryGetValue(currentOriginKey, out var list))
                    extrasByOriginNode[currentOriginKey] = list = new List<FastNode>();
                list.Add(n);
            }
            int totalExtras = 0;

            // Resolve a FastTensorKey through identity chains to find the originating
            // key — stops as soon as we hit a key that's already registered in
            // structFields or unpackedStructSequences (so a consumer of IDENTITY(A)
            // picks up A's unpack state).
            FastTensorKey ResolveKey(FastTensorKey key)
            {
                var visited = new HashSet<FastTensorKey>();
                while (visited.Add(key))
                {
                    if (structFields.ContainsKey(key) || unpackedStructSequences.ContainsKey(key))
                        return key;
                    if (!identityMap.TryGetValue(key, out var next)) return key;
                    key = next;
                }
                return key;
            }

            // Unified topological walk. graph.Nodes is in topological order by the
            // time this processor runs (earlier Fast passes preserve it), so every
            // consumer sees its TensorStruct-typed inputs already registered in
            // structFields / unpackedStructSequences by the time we visit it.
            foreach (var node in graph.Nodes)
            {
                currentOriginKey = node.Key;
                switch (node.OpCode)
                {
                    case InternalOpCodes.TENSOR_STRUCT_CREATE:
                    {
                        if (node.Outputs.Count == 0 || node.Outputs[0] is not FastTensorKey cok || cok.IsEmpty) break;
                        structFields[cok] = node.Inputs.ToList();
                        nodesToRemove.Add(node.Key);
                        break;
                    }
                    case OpCodes.IDENTITY:
                    {
                        if (node.Outputs.Count == 0 || node.Outputs[0] is not FastTensorKey iok || iok.IsEmpty) break;
                        if (node.Inputs.Count == 0 || node.Inputs[0] is not FastTensorKey iik || iik.IsEmpty) break;
                        identityMap[iok] = iik;
                        // Propagate unpack state across the IDENTITY so downstream
                        // ResolveKey calls on `iok` find the struct/sequence.
                        var rk = ResolveKey(iik);
                        if (structFields.TryGetValue(rk, out var idFields))
                            structFields[iok] = idFields;
                        else if (unpackedStructSequences.TryGetValue(rk, out var idSeqs))
                            unpackedStructSequences[iok] = idSeqs;
                        if (structDtypeByKey.TryGetValue(rk, out var idDType))
                            structDtypeByKey[iok] = idDType;
                        break;
                    }
                    case OpCodes.SEQUENCE_EMPTY:
                    {
                        var dtype = node.Attributes.GetDTypeVal(OnnxOpAttributeNames.AttrDtype);
                        if (dtype?.TensorStructDef is not { } sEmptyDef) break;
                        if (node.Outputs.Count == 0 || node.Outputs[0] is not FastTensorKey eok || eok.IsEmpty) break;

                        var fieldSeqKeys = new List<FastTensorKey>(sEmptyDef.Fields.Length);
                        foreach (var field in sEmptyDef.Fields)
                        {
                            var newKey = FastNodeKey.New();
                            AddExtra(FastNodeConstructionUtils.CreateSequenceEmpty(newKey, field.ElementType));
                            fieldSeqKeys.Add(new FastTensorKey(newKey, 0));
                        }
                        unpackedStructSequences[eok] = fieldSeqKeys;
                        structDtypeByKey[eok] = dtype;
                        nodesToRemove.Add(node.Key);
                        break;
                    }
                    case OpCodes.SEQUENCE_CONSTRUCT:
                    {
                        // Only fires when every element is a registered TensorStruct.
                        // A SEQUENCE_CONSTRUCT over plain tensors isn't our concern.
                        if (node.Outputs.Count == 0 || node.Outputs[0] is not FastTensorKey ccok || ccok.IsEmpty) break;
                        var elementInputs = node.Inputs;
                        if (elementInputs.Count == 0) break;

                        var elementFieldsList = new List<List<FastTensorKey?>>(elementInputs.Count);
                        bool allStruct = true;
                        foreach (var ik in elementInputs)
                        {
                            if (ik is null) { allStruct = false; break; }
                            var rk = ResolveKey(ik.Value);
                            if (!structFields.TryGetValue(rk, out var fList)) { allStruct = false; break; }
                            elementFieldsList.Add(fList);
                        }
                        if (!allStruct) break;

                        int numFields = elementFieldsList[0].Count;
                        if (!elementFieldsList.All(e => e.Count == numFields)) break;

                        var fieldSeqKeys = new List<FastTensorKey>(numFields);
                        for (int f = 0; f < numFields; f++)
                        {
                            var elements = elementFieldsList.Select(e => e[f]).ToList();
                            var newKey = FastNodeKey.New();
                            AddExtra(FastNodeConstructionUtils.CreateSequenceConstruct(newKey, elements));
                            fieldSeqKeys.Add(new FastTensorKey(newKey, 0));
                        }
                        unpackedStructSequences[ccok] = fieldSeqKeys;
                        // Inherit the element struct dtype from the first input.
                        if (elementInputs[0] is FastTensorKey firstElKey
                            && structDtypeByKey.TryGetValue(ResolveKey(firstElKey), out var elDType))
                            structDtypeByKey[ccok] = elDType;
                        nodesToRemove.Add(node.Key);
                        break;
                    }
                    case OpCodes.SEQUENCE_AT:
                    {
                        if (node.Inputs.Count < 2 || node.Inputs[0] is not FastTensorKey satSeq || satSeq.IsEmpty) break;
                        if (node.Inputs[1] is not FastTensorKey satIdx || satIdx.IsEmpty) break;
                        var rk = ResolveKey(satSeq);
                        if (!unpackedStructSequences.TryGetValue(rk, out var satSeqs)) break;
                        if (node.Outputs.Count == 0 || node.Outputs[0] is not FastTensorKey satOut || satOut.IsEmpty) break;

                        var perFieldOut = new List<FastTensorKey?>(satSeqs.Count);
                        foreach (var seqKey in satSeqs)
                        {
                            var newKey = FastNodeKey.New();
                            AddExtra(FastNodeConstructionUtils.CreateSequenceAt(newKey, seqKey, satIdx));
                            perFieldOut.Add(new FastTensorKey(newKey, 0));
                        }
                        structFields[satOut] = perFieldOut;
                        // Element's struct dtype matches the sequence's.
                        if (structDtypeByKey.TryGetValue(rk, out var satDType))
                            structDtypeByKey[satOut] = satDType;
                        nodesToRemove.Add(node.Key);
                        break;
                    }
                    case OpCodes.SEQUENCE_ERASE:
                    {
                        if (node.Inputs.Count < 2 || node.Inputs[0] is not FastTensorKey serSeq || serSeq.IsEmpty) break;
                        if (node.Inputs[1] is not FastTensorKey serIdx || serIdx.IsEmpty) break;
                        var rk = ResolveKey(serSeq);
                        if (!unpackedStructSequences.TryGetValue(rk, out var serSeqs)) break;
                        if (node.Outputs.Count == 0 || node.Outputs[0] is not FastTensorKey serOut || serOut.IsEmpty) break;

                        var perFieldSeq = new List<FastTensorKey>(serSeqs.Count);
                        foreach (var seqKey in serSeqs)
                        {
                            var newKey = FastNodeKey.New();
                            AddExtra(FastNodeConstructionUtils.CreateSequenceErase(newKey, seqKey, serIdx));
                            perFieldSeq.Add(new FastTensorKey(newKey, 0));
                        }
                        unpackedStructSequences[serOut] = perFieldSeq;
                        if (structDtypeByKey.TryGetValue(rk, out var serDType))
                            structDtypeByKey[serOut] = serDType;
                        nodesToRemove.Add(node.Key);
                        break;
                    }
                    case OpCodes.SEQUENCE_INSERT:
                    {
                        if (node.Inputs.Count < 2 || node.Inputs[0] is not FastTensorKey sisSeq || sisSeq.IsEmpty) break;
                        if (node.Inputs[1] is not FastTensorKey sisEl || sisEl.IsEmpty) break;
                        var rkSeq = ResolveKey(sisSeq);
                        var rkEl = ResolveKey(sisEl);
                        if (!unpackedStructSequences.TryGetValue(rkSeq, out var sisSeqs)) break;
                        if (!structFields.TryGetValue(rkEl, out var sisElFields)) break;
                        if (sisSeqs.Count != sisElFields.Count) break;
                        if (node.Outputs.Count == 0 || node.Outputs[0] is not FastTensorKey sisOut || sisOut.IsEmpty) break;
                        FastTensorKey? sisPos = node.Inputs.Count > 2 ? node.Inputs[2] : null;

                        var perFieldSeq = new List<FastTensorKey>(sisSeqs.Count);
                        for (int f = 0; f < sisSeqs.Count; f++)
                        {
                            var seqKey = sisSeqs[f];
                            var elFieldKey = sisElFields[f];
                            if (elFieldKey is null) break;
                            var newKey = FastNodeKey.New();
                            AddExtra(FastNodeConstructionUtils.CreateSequenceInsert(
                                newKey, seqKey, elFieldKey.Value, sisPos));
                            perFieldSeq.Add(new FastTensorKey(newKey, 0));
                        }
                        if (perFieldSeq.Count != sisSeqs.Count) break;

                        unpackedStructSequences[sisOut] = perFieldSeq;
                        if (structDtypeByKey.TryGetValue(rkSeq, out var sisDType))
                            structDtypeByKey[sisOut] = sisDType;
                        nodesToRemove.Add(node.Key);
                        break;
                    }
                    case OpCodes.SEQUENCE_LENGTH:
                    {
                        if (node.Inputs.Count < 1 || node.Inputs[0] is not FastTensorKey slSeq || slSeq.IsEmpty) break;
                        var rk = ResolveKey(slSeq);
                        if (!unpackedStructSequences.TryGetValue(rk, out var slSeqs) || slSeqs.Count == 0) break;
                        if (node.Outputs.Count == 0 || node.Outputs[0] is not FastTensorKey slOut || slOut.IsEmpty) break;

                        // All parallel field sequences share the same length, so
                        // emit a single SEQUENCE_LENGTH on the first field and remap.
                        var newKey = FastNodeKey.New();
                        AddExtra(FastNodeConstructionUtils.CreateSequenceLength(newKey, slSeqs[0]));
                        remap[slOut] = new FastTensorKey(newKey, 0);
                        nodesToRemove.Add(node.Key);
                        break;
                    }
                    case OpCodes.LOOP_OPEN:
                    {
                        // LOOP_OPEN flat inputs = [maxIter, cond, ...loopVars]. Only
                        // expand when at least one loopVar is a registered TensorStruct
                        // or TensorStruct sequence.
                        if (!HasRegisteredStructLoopVar(node, 2, structFields, unpackedStructSequences, ResolveKey)) break;
                        ExpandLoopOpenStructLoopVars(node, structFields, unpackedStructSequences, structDtypeByKey, remap, loopOpenOriginalVariadicCounts, ResolveKey);
                        break;
                    }
                    case OpCodes.LOOP_CLOSE:
                    {
                        // LOOP_CLOSE body inputs = [break, ...loopVars, ...scanVars].
                        // LoopVar inputs may be TensorStruct — scan vars are always
                        // plain tensors per the op definition.
                        if (!HasRegisteredStructLoopCloseVar(node, structFields, unpackedStructSequences, loopOpenOriginalVariadicCounts, ResolveKey)) break;
                        ExpandLoopCloseStructLoopVars(node, structFields, unpackedStructSequences, structDtypeByKey, remap, loopOpenOriginalVariadicCounts, ResolveKey);
                        break;
                    }
                    case OpCodes.IF_CLOSE:
                    {
                        if (!HasRegisteredStructIfBranch(node, structFields, unpackedStructSequences, ResolveKey)) break;
                        ExpandIfCloseStructBranches(node, structFields, unpackedStructSequences, structDtypeByKey, remap, ResolveKey);
                        break;
                    }
                    case InternalOpCodes.TENSOR_STRUCT_GETFIELD:
                    {
                        if (node.Inputs.Count == 0 || node.Inputs[0] is not FastTensorKey gstruct || gstruct.IsEmpty) break;
                        var rk = ResolveKey(gstruct);
                        if (!structFields.TryGetValue(rk, out var gFields)) break;
                        var fieldName = node.Attributes.GetStringVal(OnnxOpAttributeNames.ShrkAttrFieldName);
                        if (!structDtypeByKey.TryGetValue(rk, out var gDType) ||
                            gDType.TensorStructDef is not { } gStructDef || fieldName is null) break;
                        int gFieldIdx = -1;
                        for (int f = 0; f < gStructDef.Fields.Length; f++)
                            if (gStructDef.Fields[f].Name == fieldName) { gFieldIdx = f; break; }
                        if (gFieldIdx < 0 || gFieldIdx >= gFields.Count || gFields[gFieldIdx] is null) break;
                        if (node.Outputs.Count == 0 || node.Outputs[0] is not FastTensorKey gOut || gOut.IsEmpty) break;
                        remap[gOut] = gFields[gFieldIdx]!.Value;
                        nodesToRemove.Add(node.Key);
                        break;
                    }
                }
            }

            // Every SEQUENCE_AT / LOOP_OPEN / LOOP_CLOSE / IF_CLOSE output that
            // produces a TensorStruct flows through `structFields` (for struct
            // values) or `unpackedStructSequences` (for struct sequences). For
            // GETFIELD resolution of a struct produced by LOOP/IF, we also need
            // a TensorStructDef lookup — populate it eagerly for LOOP/IF outputs.
            // The original struct FastTensorKey is the same key the control-flow
            // handler stored in the unpack dict, so we carry its dtype from the
            // pre-expansion input.
            //
            // (Already handled for CREATE / MODEL_TENSORSTRUCT_INPUT in
            // structDtypeByKey above. For LOOP/IF-propagated struct keys, the
            // handlers themselves record the dtype by re-seeding
            // structDtypeByKey when they see an incoming struct key — see the
            // helper methods.)

            foreach (var kv in extrasByOriginNode) totalExtras += kv.Value.Count;

            Debug.Assert(remap.Count > 0 || nodesToRemove.Count > 0 || totalExtras > 0,
                "FastUnpackTensorStructs.Process: reached the rewire stage without any work to do. "
                + "The initial hasCreate/hasGetField/hasStructInput check above gates this stage, "
                + "and each of those op codes produces at least one remap/removal/extra by "
                + "construction in well-formed graphs.");

            // Resolve transitive remap chains (nested struct fields).
            foreach (var key in remap.Keys.ToList())
            {
                var target = remap[key];
                while (remap.TryGetValue(target, out var next))
                    target = next;
                remap[key] = target;
            }

            // Apply the remap: replace all input references to GETFIELD outputs with the actual field keys.
            for (int i = 0; i < graph.Nodes.Count; i++)
            {
                var node = graph.Nodes[i];
                foreach (var kvp in node.FullInputs)
                {
                    var inputList = kvp.Value;
                    for (int j = 0; j < inputList.Count; j++)
                    {
                        if (inputList[j] is FastTensorKey key && remap.TryGetValue(key, out var replacement))
                            inputList[j] = replacement;
                    }
                }
            }

            // Also remap graph outputs.
            for (int i = 0; i < graph.Outputs.Count; i++)
            {
                if (remap.TryGetValue(graph.Outputs[i], out var replacement))
                    graph.Outputs[i] = replacement;
            }

            // Rebuild graph.Nodes: drop nodesToRemove, splice each origin node's
            // extras directly after the origin (or in place of it, when the origin
            // is being removed). Origin extras are constructed from the origin's
            // outputs and live in the same enclosing scope, so this preserves
            // both topological order and OPEN/CLOSE nesting without a Kahn pass.
            var rebuiltUnpack = new List<FastNode>(graph.Nodes.Count + totalExtras);
            foreach (var n in graph.Nodes)
            {
                if (!nodesToRemove.Contains(n.Key)) rebuiltUnpack.Add(n);
                if (extrasByOriginNode.TryGetValue(n.Key, out var extras))
                    rebuiltUnpack.AddRange(extras);
            }
            graph.Nodes = rebuiltUnpack;
            System.Diagnostics.Debug.Assert(graph.IsLinearOrderValid(), "graph.IsLinearOrderValid()");
        }

        // -----------------------------------------------------------------
        // B2b control-flow expansion helpers. All three drive off the
        // `structFields` / `unpackedStructSequences` tables populated by the
        // main topological walk; none consult tensor-info. The LOOP_OPEN /
        // LOOP_CLOSE pair communicates the pre-expansion loop-var count via
        // `loopOpenOriginalVariadicCounts` so the CLOSE handler can recover
        // the loopVar / scanVar split of its own still-pre-expansion body
        // inputs (we've already rewritten OPEN's inputs by then).
        // -----------------------------------------------------------------

        /// <summary>
        /// Returns true when the LOOP_OPEN's variadic loop-var slice
        /// (inputs[variadicStart..]) contains at least one key whose resolved
        /// identity is a registered TensorStruct or TensorStruct sequence.
        /// </summary>
        private static bool HasRegisteredStructLoopVar(
            FastNode openNode, int variadicStart,
            Dictionary<FastTensorKey, List<FastTensorKey?>> structFields,
            Dictionary<FastTensorKey, List<FastTensorKey>> unpackedStructSequences,
            Func<FastTensorKey, FastTensorKey> resolveKey)
        {
            var inputs = openNode.Inputs;
            for (int i = variadicStart; i < inputs.Count; i++)
            {
                if (inputs[i] is not FastTensorKey tk || tk.IsEmpty) continue;
                var rk = resolveKey(tk);
                if (structFields.ContainsKey(rk) || unpackedStructSequences.ContainsKey(rk))
                    return true;
            }
            return false;
        }

        /// <summary>
        /// Returns true when LOOP_CLOSE has at least one loop-var body input
        /// (not scan-var, since scan-vars are always plain tensors) that is a
        /// registered TensorStruct or TensorStruct sequence. We recover the
        /// loopVar count via `loopOpenOriginalVariadicCounts` because OPEN has
        /// already been rewritten to the expanded variadic arity by the time
        /// CLOSE visits.
        /// </summary>
        private static bool HasRegisteredStructLoopCloseVar(
            FastNode closeNode,
            Dictionary<FastTensorKey, List<FastTensorKey?>> structFields,
            Dictionary<FastTensorKey, List<FastTensorKey>> unpackedStructSequences,
            Dictionary<FastNodeKey, int> loopOpenOriginalVariadicCounts,
            Func<FastTensorKey, FastTensorKey> resolveKey)
        {
            if (closeNode.GraphOpenNodeKey is not FastNodeKey openKey) return false;
            if (!loopOpenOriginalVariadicCounts.TryGetValue(openKey, out int nLoopVars)) return false;

            if (!closeNode.FullInputs.TryGetValue(OnnxOpAttributeNames.AttrBody, out var bodyInputs))
                return false;

            for (int i = 0; i < nLoopVars && 1 + i < bodyInputs.Count; i++)
            {
                if (bodyInputs[1 + i] is not FastTensorKey tk || tk.IsEmpty) continue;
                var rk = resolveKey(tk);
                if (structFields.ContainsKey(rk) || unpackedStructSequences.ContainsKey(rk))
                    return true;
            }
            return false;
        }

        /// <summary>
        /// Returns true when IF_CLOSE has at least one branch-slot input
        /// (else-branch is authoritative; then-branch is asserted to match) that
        /// is a registered TensorStruct or TensorStruct sequence.
        /// </summary>
        private static bool HasRegisteredStructIfBranch(
            FastNode ifCloseNode,
            Dictionary<FastTensorKey, List<FastTensorKey?>> structFields,
            Dictionary<FastTensorKey, List<FastTensorKey>> unpackedStructSequences,
            Func<FastTensorKey, FastTensorKey> resolveKey)
        {
            if (!ifCloseNode.FullInputs.TryGetValue(OnnxOpAttributeNames.AttrElseBranch, out var elseInputs))
                return false;
            foreach (var ik in elseInputs)
            {
                if (ik is not FastTensorKey tk || tk.IsEmpty) continue;
                var rk = resolveKey(tk);
                if (structFields.ContainsKey(rk) || unpackedStructSequences.ContainsKey(rk))
                    return true;
            }
            return false;
        }

        /// <summary>
        /// LOOP_OPEN variadic expansion. Each TensorStruct loop-var slot becomes
        /// N parallel slots (one per field); each TensorStruct-sequence loop-var
        /// slot likewise becomes N parallel sequence slots. Plain-tensor slots
        /// pass through with a fresh output key (their slot index shifts as
        /// struct slots before them expand). Records the new per-field output
        /// list in `structFields` / `unpackedStructSequences` so downstream
        /// body consumers resolve the expanded slot directly. Stores the
        /// pre-expansion variadic count in `loopOpenOriginalVariadicCounts`
        /// for the matching LOOP_CLOSE handler.
        /// </summary>
        private static void ExpandLoopOpenStructLoopVars(
            FastNode openNode,
            Dictionary<FastTensorKey, List<FastTensorKey?>> structFields,
            Dictionary<FastTensorKey, List<FastTensorKey>> unpackedStructSequences,
            Dictionary<FastTensorKey, DType> structDtypeByKey,
            Dictionary<FastTensorKey, FastTensorKey> remap,
            Dictionary<FastNodeKey, int> loopOpenOriginalVariadicCounts,
            Func<FastTensorKey, FastTensorKey> resolveKey)
        {
            // FullInputs[""]       = [maxIter, cond, ...loopVars]
            // FullOutputs["body"]  = [iterIdx, vestigialTrue, ...loopVars]
            // (LoopAPI.ProcessNode collapses LOOP_OPEN's three declared output
            // groups into a single AttrBody group.)
            const string openOutputGroup = OnnxOpAttributeNames.AttrBody;
            var origDirectInputs = openNode.FullInputs[""];
            var origBodyOutputs = openNode.FullOutputs[openOutputGroup];
            int origLoopVarInputCount = Math.Max(0, origDirectInputs.Count - 2);

            var newDirectInputs = new List<FastTensorKey?>(origDirectInputs.Count)
            { origDirectInputs[0], origDirectInputs[1] };
            var newBodyOutputs = new List<FastTensorKey?>(origBodyOutputs.Count)
            { origBodyOutputs[0], origBodyOutputs[1] };

            int slot = 2;
            for (int i = 0; i < origLoopVarInputCount; i++)
            {
                var origIn = origDirectInputs[2 + i];
                var origOut = origBodyOutputs[2 + i];
                Debug.Assert(origIn is not null && origOut is not null,
                    "ExpandLoopOpenStructLoopVars: LOOP_OPEN loop-var slot has a null "
                    + "input/output. LoopAPI binds every loop-var initializer + body "
                    + "output by construction, so null pass-through is unreachable.");

                var rk = resolveKey(origIn.Value);
                if (structFields.TryGetValue(rk, out var fields))
                {
                    var perFieldOut = new List<FastTensorKey?>(fields.Count);
                    foreach (var f in fields)
                    {
                        newDirectInputs.Add(f);
                        var newOut = new FastTensorKey(openNode.Key, slot++);
                        newBodyOutputs.Add(newOut);
                        perFieldOut.Add(newOut);
                    }
                    structFields[origOut.Value] = perFieldOut;
                    if (structDtypeByKey.TryGetValue(rk, out var dt))
                        structDtypeByKey[origOut.Value] = dt;
                }
                else if (unpackedStructSequences.TryGetValue(rk, out var seqs))
                {
                    var perFieldSeq = new List<FastTensorKey>(seqs.Count);
                    foreach (var s in seqs)
                    {
                        newDirectInputs.Add(s);
                        var newOut = new FastTensorKey(openNode.Key, slot++);
                        newBodyOutputs.Add(newOut);
                        perFieldSeq.Add(newOut);
                    }
                    unpackedStructSequences[origOut.Value] = perFieldSeq;
                    if (structDtypeByKey.TryGetValue(rk, out var dt))
                        structDtypeByKey[origOut.Value] = dt;
                }
                else
                {
                    newDirectInputs.Add(origIn);
                    var newOut = new FastTensorKey(openNode.Key, slot++);
                    newBodyOutputs.Add(newOut);
                    if (!newOut.Equals(origOut.Value)) remap[origOut.Value] = newOut;
                }
            }

            openNode.FullInputs[""] = newDirectInputs;
            openNode.FullOutputs[openOutputGroup] = newBodyOutputs;
            loopOpenOriginalVariadicCounts[openNode.Key] = origLoopVarInputCount;
        }

        /// <summary>
        /// LOOP_CLOSE variadic expansion. Mirror of <see cref="ExpandLoopOpenStructLoopVars"/>:
        /// each TensorStruct loop-var body-input slot expands into its per-field
        /// parallel slots and the corresponding output slot expands the same way.
        /// Scan-var inputs pass through verbatim (scan-vars are always plain
        /// tensors per the op definition); their output slot indices shift by
        /// the same delta as loopVars' expansion and are remapped so downstream
        /// consumers still resolve correctly.
        /// </summary>
        private static void ExpandLoopCloseStructLoopVars(
            FastNode closeNode,
            Dictionary<FastTensorKey, List<FastTensorKey?>> structFields,
            Dictionary<FastTensorKey, List<FastTensorKey>> unpackedStructSequences,
            Dictionary<FastTensorKey, DType> structDtypeByKey,
            Dictionary<FastTensorKey, FastTensorKey> remap,
            Dictionary<FastNodeKey, int> loopOpenOriginalVariadicCounts,
            Func<FastTensorKey, FastTensorKey> resolveKey)
        {
            var openKey = closeNode.GraphOpenNodeKey!.Value;
            int nLoopVars = loopOpenOriginalVariadicCounts[openKey];

            // FullInputs["body"] = [break, ...loopVars, ...scanVars]
            // FullOutputs[""]    = [...loopedVars, ...scannedVars]
            // (LoopAPI.ProcessNode collapses loopedVariables / scannedVariables
            // into the default group.)
            var origBodyInputs = closeNode.FullInputs[OnnxOpAttributeNames.AttrBody];
            var origFlatOutputs = closeNode.FullOutputs[""];

            var newBodyInputs = new List<FastTensorKey?>(origBodyInputs.Count);
            newBodyInputs.Add(origBodyInputs[0]); // break
            var newFlatOutputs = new List<FastTensorKey?>();
            int slot = 0;

            // LoopVar slots (variadic).
            for (int i = 0; i < nLoopVars; i++)
            {
                var origIn = origBodyInputs[1 + i];
                var origOut = origFlatOutputs[i];
                Debug.Assert(origIn is not null && origOut is not null,
                    "ExpandLoopCloseStructLoopVars: LOOP_CLOSE loop-var slot has a null "
                    + "body input/flat output. LoopAPI binds every loop-var on both "
                    + "sides by construction, so null pass-through is unreachable.");

                var rk = resolveKey(origIn.Value);
                if (structFields.TryGetValue(rk, out var fields))
                {
                    var perFieldOut = new List<FastTensorKey?>(fields.Count);
                    foreach (var f in fields)
                    {
                        newBodyInputs.Add(f);
                        var newOut = new FastTensorKey(closeNode.Key, slot++);
                        newFlatOutputs.Add(newOut);
                        perFieldOut.Add(newOut);
                    }
                    structFields[origOut.Value] = perFieldOut;
                    if (structDtypeByKey.TryGetValue(rk, out var dt))
                        structDtypeByKey[origOut.Value] = dt;
                }
                else if (unpackedStructSequences.TryGetValue(rk, out var seqs))
                {
                    var perFieldSeq = new List<FastTensorKey>(seqs.Count);
                    foreach (var s in seqs)
                    {
                        newBodyInputs.Add(s);
                        var newOut = new FastTensorKey(closeNode.Key, slot++);
                        newFlatOutputs.Add(newOut);
                        perFieldSeq.Add(newOut);
                    }
                    unpackedStructSequences[origOut.Value] = perFieldSeq;
                    if (structDtypeByKey.TryGetValue(rk, out var dt))
                        structDtypeByKey[origOut.Value] = dt;
                }
                else
                {
                    newBodyInputs.Add(origIn);
                    var newOut = new FastTensorKey(closeNode.Key, slot++);
                    newFlatOutputs.Add(newOut);
                    if (!newOut.Equals(origOut.Value)) remap[origOut.Value] = newOut;
                }
            }

            // ScanVar slots pass through: inputs verbatim, outputs with shifted slots.
            for (int j = 1 + nLoopVars; j < origBodyInputs.Count; j++)
                newBodyInputs.Add(origBodyInputs[j]);
            for (int j = nLoopVars; j < origFlatOutputs.Count; j++)
            {
                var origOut = origFlatOutputs[j];
                Debug.Assert(origOut is not null,
                    "ExpandLoopCloseStructLoopVars: LOOP_CLOSE scan output slot is null. "
                    + "LoopAPI binds every scan output to a body value by construction, "
                    + "so null pass-through is unreachable.");
                var newOut = new FastTensorKey(closeNode.Key, slot++);
                newFlatOutputs.Add(newOut);
                if (!newOut.Equals(origOut.Value)) remap[origOut.Value] = newOut;
            }

            closeNode.FullInputs[OnnxOpAttributeNames.AttrBody] = newBodyInputs;
            closeNode.FullOutputs[""] = newFlatOutputs;
        }

        /// <summary>
        /// IF_CLOSE branch expansion. For each branch slot, both the then- and
        /// else-side inputs must be unpacked the same way (struct vs sequence
        /// vs plain tensor) — asserted and required. Struct / sequence slots
        /// expand into N parallel slots, one per field. Plain slots pass
        /// through with a fresh output key.
        /// </summary>
        private static void ExpandIfCloseStructBranches(
            FastNode ifCloseNode,
            Dictionary<FastTensorKey, List<FastTensorKey?>> structFields,
            Dictionary<FastTensorKey, List<FastTensorKey>> unpackedStructSequences,
            Dictionary<FastTensorKey, DType> structDtypeByKey,
            Dictionary<FastTensorKey, FastTensorKey> remap,
            Func<FastTensorKey, FastTensorKey> resolveKey)
        {
            var origElse = ifCloseNode.FullInputs[OnnxOpAttributeNames.AttrElseBranch];
            var origThen = ifCloseNode.FullInputs[OnnxOpAttributeNames.AttrThenBranch];
            var origOuts = ifCloseNode.FullOutputs[""];

            var newElse = new List<FastTensorKey?>();
            var newThen = new List<FastTensorKey?>();
            var newOuts = new List<FastTensorKey?>();
            int slot = 0;

            for (int i = 0; i < origElse.Count; i++)
            {
                var origEl = origElse[i];
                var origTh = origThen[i];
                var origOut = origOuts[i];
                Debug.Assert(origEl is not null && origTh is not null && origOut is not null,
                    "ExpandIfCloseStructBranches: IF_CLOSE branch slot has a null then/else "
                    + "input or output. condition.IfElse binds every branch slot on both "
                    + "sides by construction, so null pass-through is unreachable.");

                var rkEl = resolveKey(origEl.Value);
                var rkTh = resolveKey(origTh.Value);

                bool elIsStruct = structFields.ContainsKey(rkEl);
                bool thIsStruct = structFields.ContainsKey(rkTh);
                bool elIsSeq = unpackedStructSequences.ContainsKey(rkEl);
                bool thIsSeq = unpackedStructSequences.ContainsKey(rkTh);

                Debug.Assert(elIsStruct == thIsStruct && elIsSeq == thIsSeq,
                    "IF_CLOSE branch slot must be unpacked the same way on both sides.");

                if (elIsStruct)
                {
                    var elFields = structFields[rkEl];
                    var thFields = structFields[rkTh];
                    Debug.Assert(elFields.Count == thFields.Count);
                    var perFieldOut = new List<FastTensorKey?>(elFields.Count);
                    for (int f = 0; f < elFields.Count; f++)
                    {
                        newElse.Add(elFields[f]);
                        newThen.Add(thFields[f]);
                        var newOut = new FastTensorKey(ifCloseNode.Key, slot++);
                        newOuts.Add(newOut);
                        perFieldOut.Add(newOut);
                    }
                    structFields[origOut.Value] = perFieldOut;
                    if (structDtypeByKey.TryGetValue(rkEl, out var dt))
                        structDtypeByKey[origOut.Value] = dt;
                }
                else if (elIsSeq)
                {
                    var elSeqs = unpackedStructSequences[rkEl];
                    var thSeqs = unpackedStructSequences[rkTh];
                    Debug.Assert(elSeqs.Count == thSeqs.Count);
                    var perFieldSeq = new List<FastTensorKey>(elSeqs.Count);
                    for (int f = 0; f < elSeqs.Count; f++)
                    {
                        newElse.Add(elSeqs[f]);
                        newThen.Add(thSeqs[f]);
                        var newOut = new FastTensorKey(ifCloseNode.Key, slot++);
                        newOuts.Add(newOut);
                        perFieldSeq.Add(newOut);
                    }
                    unpackedStructSequences[origOut.Value] = perFieldSeq;
                    if (structDtypeByKey.TryGetValue(rkEl, out var dt))
                        structDtypeByKey[origOut.Value] = dt;
                }
                else
                {
                    newElse.Add(origEl);
                    newThen.Add(origTh);
                    var newOut = new FastTensorKey(ifCloseNode.Key, slot++);
                    newOuts.Add(newOut);
                    if (!newOut.Equals(origOut.Value)) remap[origOut.Value] = newOut;
                }
            }

            ifCloseNode.FullInputs[OnnxOpAttributeNames.AttrElseBranch] = newElse;
            ifCloseNode.FullInputs[OnnxOpAttributeNames.AttrThenBranch] = newThen;
            ifCloseNode.FullOutputs[""] = newOuts;
        }
    }

    /// <summary>
    /// Converts TRAINABLE_PARAM_ID_REF nodes to TRAINABLE_PARAM nodes.
    /// Template composition is performed natively on the <see cref="FastComputationGraph"/>.
    /// Model ID extraction runs <see cref="QuickExecutionEngine"/> directly over the
    /// FastComputationGraph — no CG round-trip needed. Node rewriting (step 4 equivalent)
    /// is also implemented natively, avoiding any CG↔FastCG conversion.
    /// </summary>
    internal static class FastConvertTrainableParamIdRefToTrainableParam
    {
        public static void Process(
            FastComputationGraph graph,
            FastExtractIdentifierTemplates.IdentifierTemplateInfos identifierTemplatesInfo,
            ModelParamList inputHints,
            ComputeContext? computeContext = null)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));

            bool hasIdRef = false;
            for (int i = 0; i < graph.Nodes.Count; i++)
            {
                if (graph.Nodes[i].OpCode == InternalOpCodes.TRAINABLE_PARAM_ID_REF)
                {
                    hasIdRef = true;
                    break;
                }
            }
            if (!hasIdRef) return;

            // Template composition: purely structural, no CG needed.
            var composedTemplates = new Dictionary<ModelId, ModelParamIdentifierTemplate>();
            foreach (var kvp in identifierTemplatesInfo.FullTemplates)
                composedTemplates[kvp.Key] = kvp.Value;
            foreach (var relKvp in identifierTemplatesInfo.RelativeTemplates)
            {
                foreach (var baseKvp in identifierTemplatesInfo.BaseModuleTemplates)
                {
                    var composed = new ModelParamIdentifierTemplate(baseKvp.Value, relKvp.Value);
                    composedTemplates[composed.ModelIdTemplate] = composed;
                }
            }
            var paramIdentifierTemplates = new IdTemplateInfos
            {
                IdTemplates = composedTemplates.ToImmutableDictionary()
            };

            // Run QEE over the FastGraph to collect per-node tensor values (including loop history).
            // Provide sample inputs so shape-dependent initializer params (e.g., conv weight shapes
            // computed from ChannelsNCHW) can be fully evaluated.
            var engine = new QuickExecutionEngine();
            Dictionary<FastTensorKey, IRuntimeTensor>? initialInputs = null;
            if (inputHints is not null && inputHints.ModelParams.Length > 0)
            {
                initialInputs = new Dictionary<FastTensorKey, IRuntimeTensor>();
                var graphInputKeys = graph.Inputs;
                for (int i = 0; i < Math.Min(graphInputKeys.Count, inputHints.ModelParams.Length); i++)
                {
                    var td = inputHints.ModelParams[i].ToTensorData();
                    if (td is not null)
                    {
                        initialInputs[graphInputKeys[i]] = TensorDataConverter.ToRuntimeTensor(
                            td, engine.MaxDataElements);
                    }
                }
            }
            var store = engine.Run(graph, initialInputs);
            var candidateModelIdInfos = ExtractModelIdInfosFromStore(graph, store);

            // If QEE couldn't resolve every TRAINABLE_PARAM_ID_REF node's model ID (e.g., the
            // ID flows through ops whose integer data QEE can't track — Sequence ops populated
            // inside loops, etc.), throw the sentinel exception. There is no longer a CG
            // fallback, so callers see a hard failure for these unsupported graph shapes.
            if (!AllIdRefModelIdsResolved(graph, store))
                throw new FastPipelineUnsupportedException(
                    "FastConvertTrainableParamIdRefToTrainableParam: QEE could not resolve all " +
                    "TRAINABLE_PARAM_ID_REF model IDs (integer data unavailable).");

            if (candidateModelIdInfos.IsEmpty) return;

            // Liveness filter: ExtractModelIdInfosFromStore returns every model ID a
            // TRAINABLE_PARAM_ID_REF node could produce, even ones whose value never reaches
            // the graph output (e.g. a shortcut Conv param on a BasicBlockS11 call whose
            // `downsample` hyperparam folds the IfElse-gate to the `x` branch — the shortcut
            // Conv is dead but its ID_REF node is still in the graph). Without this filter,
            // the resulting concrete architecture carries dead trainable-param initializers
            // that inflate the serialized model.
            //
            // Uses the native fast-graph port of `ListAllSpecificModelIdsUsed` — builds a
            // parallel mask-computation subgraph on the FastComputationGraph, runs it
            // through QEE, and reads back the set of live model IDs. Replaces a prior
            // CG round-trip through `FastProcessorHelper.Apply` that allocated enough
            // intermediate CG state to crash the testhost on big models.
            var candidateIds = candidateModelIdInfos.Select(x => x.SpecificModelId).ToImmutableArray();
            var liveModelIds = FastListAllSpecificModelIdsUsed.Process(
                graph, inputHints ?? new ModelParamList(), candidateIds).ToHashSet();
            var liveModelIdInfos = candidateModelIdInfos
                .Where(x => liveModelIds.Contains(x.SpecificModelId))
                .ToImmutableArray();
            // Pruned (non-live) candidates still have ID_REF nodes in the graph (e.g. an
            // IfElse whose condition only QEE/shape-inference can fold to constant — the
            // dead branch's ID_REF survives FastFoldConstants). Their slots in the param
            // sequence must be filled with shape-correct zero CONSTANTs so the surviving
            // SequenceAt resolves to a tensor that broadcasts cleanly with the dead
            // branch's eager downstream consumers (e.g. an Add/Mul outside the IF body).
            // Without this they'd default to the dtype's empty (shape (0,)) filler and
            // crash ONNX Runtime with a broadcast error.
            var deadModelIdInfos = candidateModelIdInfos
                .Where(x => !liveModelIds.Contains(x.SpecificModelId))
                .ToImmutableArray();

            // Proceed whenever there is anything to convert — INCLUDING the all-dead
            // case (live empty, dead non-empty). That arises when a module's only
            // trainable params sit in a branch that a baked hyperparameter folds away
            // — e.g. an affine normalizer's gamma/beta in `affine.IfElse(x*g+b, x)`
            // with `affine = false`, where gamma/beta are the graph's ONLY params.
            // NativeConvertTrainableParamIdRef handles dead-only by emitting
            // shape-correct zero CONSTANTs and replacing every surviving
            // TRAINABLE_PARAM_ID_REF; returning early here would instead leave those
            // dead id-refs in the graph, tripping the forbidden-op validation. Since
            // `candidateModelIdInfos` is non-empty above, this guard only fires when
            // there is genuinely nothing to do.
            if (liveModelIdInfos.IsEmpty && deadModelIdInfos.IsEmpty) return;

            NativeConvertTrainableParamIdRef(graph, liveModelIdInfos, deadModelIdInfos, paramIdentifierTemplates);
        }

        /// <summary>
        /// Discovery-only variant of the QEE-based model-id resolution that <see cref="Process"/>
        /// performs internally. Runs <see cref="QuickExecutionEngine"/> over <paramref name="graph"/>
        /// with the given <paramref name="inputHints"/> bound to graph inputs, then returns one
        /// <see cref="TrainableParamInfo"/> per distinct specific model ID resolved from the
        /// graph's TRAINABLE_PARAM_ID_REF nodes (including per-iteration values for loops via the
        /// QEE History field). Does not mutate the graph, does not run the liveness filter, and
        /// does not throw if some model IDs are unresolved — callers performing shape-discovery
        /// for default-checkpoint initialization use this entry point.
        /// </summary>
        public static ImmutableArray<TrainableParamInfo> DiscoverTrainableParamInfos(
            FastComputationGraph graph,
            ModelParamList? inputHints = null)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));

            var engine = new QuickExecutionEngine();
            Dictionary<FastTensorKey, IRuntimeTensor>? initialInputs = null;
            if (inputHints is not null && inputHints.ModelParams.Length > 0)
            {
                initialInputs = new Dictionary<FastTensorKey, IRuntimeTensor>();
                var graphInputKeys = graph.Inputs;
                for (int i = 0; i < Math.Min(graphInputKeys.Count, inputHints.ModelParams.Length); i++)
                {
                    var td = inputHints.ModelParams[i].ToTensorData();
                    if (td is not null)
                    {
                        initialInputs[graphInputKeys[i]] = TensorDataConverter.ToRuntimeTensor(td, engine.MaxDataElements);
                    }
                }
            }
            var store = engine.Run(graph, initialInputs);
            return ExtractModelIdInfosFromStore(graph, store);
        }

        /// <summary>
        /// Returns true iff every TRAINABLE_PARAM_ID_REF node's model-ID input resolves to a
        /// RuntimeTensor whose integer data (or per-iteration history) is available.
        /// </summary>
        private static bool AllIdRefModelIdsResolved(FastComputationGraph graph, Dictionary<FastTensorKey, IRuntimeTensor> store)
        {
            for (int i = 0; i < graph.Nodes.Count; i++)
            {
                var node = graph.Nodes[i];
                if (node.OpCode != InternalOpCodes.TRAINABLE_PARAM_ID_REF) continue;

                if (node.Inputs.Count == 0 || node.Inputs[0] is null) return false;
                if (!store.TryGetValue(node.Inputs[0]!.Value, out var raw)) return false;
                if (raw is not RuntimeTensor rt) return false;

                bool topLevelHasInt = rt.IntData.HasValue;
                bool anyIterHasInt = false;
                if (rt.History.HasValue)
                {
                    foreach (var h in rt.History.Value)
                        if (h is RuntimeTensor hrt && hrt.IntData.HasValue) { anyIterHasInt = true; break; }
                }
                if (!topLevelHasInt && !anyIterHasInt) return false;
            }
            return true;
        }

        private static void NativeConvertTrainableParamIdRef(
            FastComputationGraph graph,
            ImmutableArray<TrainableParamInfo> liveParamInfos,
            ImmutableArray<TrainableParamInfo> deadParamInfos,
            IdTemplateInfos idTemplateInfos)
        {
            // Index space must cover every candidate ID — including pruned ones — because
            // their ID_REF nodes are still in the graph and produce runtime IDs whose
            // dimension values can lie outside the live-only range.
            var allIdsForDims = liveParamInfos.Concat(deadParamInfos).Select(x => x.SpecificModelId);
            var maxIdCounts = FoldHelpers.MaxModelIdCounts(allIdsForDims);
            var maxSize = Enumerable.Aggregate(maxIdCounts, 1, (a, c) => a * c);
            var transformArray = FoldHelpers.IndexToFlattenedIndexTransform(maxIdCounts);

            var newNodes = new List<FastNode>();

            // --- TRAINABLE_PARAM nodes + CONSTANT initializer params ---
            var trainableParamKeysByType = liveParamInfos.Concat(deadParamInfos)
                .Select(x => x.TargetFn.Outputs[0].DType).Distinct()
                .ToDictionary(x => x, x => new FastTensorKey?[maxSize]);

            foreach (var paramInfo in liveParamInfos)
            {
                var modelId = paramInfo.SpecificModelId;
                var index = FoldHelpers.TransformModelIdToFlattenedIndex(modelId, transformArray);
                var dtype = paramInfo.TargetFn.Outputs[0].DType;
                var rank = paramInfo.TargetFn.OutputRankOverrides[0];
                var idTemplateString = idTemplateInfos.GetSpecificIdentifierTemplate(modelId).ToString();

                var initializerParamKeys = new List<FastTensorKey?>();
                foreach (var td in paramInfo.TrainableParamInputParamValues)
                {
                    var constKey = FastNodeKey.New();
                    newNodes.Add(CreateConstantTensorDataNode(constKey, td));
                    initializerParamKeys.Add(new FastTensorKey(constKey, 0));
                }

                var tpKey = FastNodeKey.New();
                var tpTensorKey = new FastTensorKey(tpKey, 0);
                var tpAttrVals = new Dictionary<string, object?>
                {
                    [OnnxOpAttributeNames.ShrkAttrLocalModelId] = paramInfo.SpecificModelId.Vals.ToArray(),
                    [OnnxOpAttributeNames.ShrkAttrDtype] = dtype,
                    [OnnxOpAttributeNames.ShrkAttrRank] = (long?)rank,
                    [OnnxOpAttributeNames.ShrkAttrShape] = paramInfo.Shape.Dims,
                    [OnnxOpAttributeNames.ShrkAttrIsTrainable] = (paramInfo.TargetFn.FunctionType == FunctionType.TrainableParamInitializer),
                    [OnnxOpAttributeNames.ShrkAttrFunctionName] = paramInfo.TargetFn.DefaultName,
                    [OnnxOpAttributeNames.ShrkAttrDomainName] = (string?)"Functions",
                };
                var tpAttrDefs = Definitions.NodeDefinitions[InternalOpCodes.TRAINABLE_PARAM].AttributeDefs;
                var tpAttrs = OnnxCSharpAttributes.FromCSharpVals(tpAttrVals, tpAttrDefs);

                newNodes.Add(new FastNode
                {
                    Key = tpKey,
                    OpCode = InternalOpCodes.TRAINABLE_PARAM,
                    Attributes = tpAttrs,
                    FullInputs = { [""] = initializerParamKeys },
                    FullOutputs = { [""] = new List<FastTensorKey?> { tpTensorKey } },
                    TargetFunction = paramInfo.TargetFn,
                    IdentifierTemplate = idTemplateString,
                });
                trainableParamKeysByType[dtype][index] = tpTensorKey;
            }

            // Dead candidates — emit shape-correct zero CONSTANTs (not TRAINABLE_PARAMs,
            // so they don't bloat the persisted ModelParamList). Their value is never
            // observed: the SequenceAt result feeds a dead branch whose output the IF
            // discards — only its shape matters, so it broadcasts cleanly with peers.
            foreach (var paramInfo in deadParamInfos)
            {
                var modelId = paramInfo.SpecificModelId;
                var index = FoldHelpers.TransformModelIdToFlattenedIndex(modelId, transformArray);
                var dtype = paramInfo.TargetFn.Outputs[0].DType;
                var zeroKey = FastNodeKey.New();
                newNodes.Add(CreateConstantTensorDataNode(zeroKey,
                    Globals.TensorDataWithDefaultVals(dtype, paramInfo.Shape.Dims)));
                trainableParamKeysByType[dtype][index] = new FastTensorKey(zeroKey, 0);
            }

            // --- Empty vector fillers for sequence slots that no ID_REF can ever hit
            // (combinations of dim values not present in any candidate). Safe to leave
            // shape-(0,) since they're never read. ---
            var emptyVectorKeys = new Dictionary<DType, FastTensorKey>();
            foreach (var dtype in trainableParamKeysByType.Keys)
            {
                var evKey = FastNodeKey.New();
                newNodes.Add(CreateConstantTensorDataNode(evKey, Globals.TensorData(dtype)));
                emptyVectorKeys[dtype] = new FastTensorKey(evKey, 0);
            }

            // --- SEQUENCE_CONSTRUCT nodes (one per DType) ---
            var sequenceKeys = new Dictionary<DType, FastTensorKey>();
            foreach (var kvp in trainableParamKeysByType)
            {
                var dtype = kvp.Key;
                var seqKey = FastNodeKey.New();
                var seqTensorKey = new FastTensorKey(seqKey, 0);

                var seqInputs = kvp.Value
                    .Select(e => e ?? (FastTensorKey?)emptyVectorKeys[dtype])
                    .ToList();

                var seqAttrDefs = Definitions.NodeDefinitions[OpCodes.SEQUENCE_CONSTRUCT].AttributeDefs;
                var seqAttrs = OnnxCSharpAttributes.FromCSharpVals(new Dictionary<string, object?>(), seqAttrDefs);

                newNodes.Add(new FastNode
                {
                    Key = seqKey,
                    OpCode = OpCodes.SEQUENCE_CONSTRUCT,
                    Attributes = seqAttrs,
                    FullInputs = { [""] = seqInputs },
                    FullOutputs = { [""] = new List<FastTensorKey?> { seqTensorKey } },
                });
                sequenceKeys[dtype] = seqTensorKey;
            }

            // --- Shared index computation nodes ---

            var transformVecKey = FastNodeKey.New();
            var transformVecTK = new FastTensorKey(transformVecKey, 0);
            newNodes.Add(CreateConstantTensorDataNode(transformVecKey,
                Globals.TensorData(new long[] { transformArray.Length }, transformArray)));

            var unsqAxesKey = FastNodeKey.New();
            var unsqAxesTK = new FastTensorKey(unsqAxesKey, 0);
            newNodes.Add(CreateConstantTensorDataNode(unsqAxesKey,
                Globals.TensorData(new long[] { 1 }, -1L)));

            var transformShapeKey = FastNodeKey.New();
            var transformShapeTK = new FastTensorKey(transformShapeKey, 0);
            newNodes.Add(FastNodeCreationHelpers.CreateFastNode(
                transformShapeKey, OpCodes.SHAPE,
                new Dictionary<string, object?>(),
                new FastTensorKey?[] { transformVecTK }));

            var reduceAttrs = new Dictionary<string, object?>
            {
                [OnnxOpAttributeNames.AttrKeepdims] = false,
                [OnnxOpAttributeNames.AttrNoopWithEmptyAxes] = false,
            };

            var transformSizeKey = FastNodeKey.New();
            var transformSizeTK = new FastTensorKey(transformSizeKey, 0);
            newNodes.Add(FastNodeCreationHelpers.CreateFastNode(
                transformSizeKey, OpCodes.REDUCE_PROD,
                new Dictionary<string, object?>(reduceAttrs),
                new FastTensorKey?[] { transformShapeTK, null }));

            var scalar0Key = FastNodeKey.New();
            var scalar0TK = new FastTensorKey(scalar0Key, 0);
            newNodes.Add(CreateConstantTensorDataNode(scalar0Key,
                Globals.TensorData(new long[] { }, 0L)));

            var unsqueezed0Key = FastNodeKey.New();
            var unsqueezed0TK = new FastTensorKey(unsqueezed0Key, 0);
            newNodes.Add(FastNodeCreationHelpers.CreateFastNode(
                unsqueezed0Key, OpCodes.UNSQUEEZE,
                new Dictionary<string, object?>(),
                new FastTensorKey?[] { scalar0TK, unsqAxesTK }));

            // --- Per TRAINABLE_PARAM_ID_REF: index computation + SEQUENCE_AT ---
            // Each ID_REF gets a small graph that's spliced in at the original
            // ID_REF's position so the result is topologically ordered + nested
            // by construction.
            var remap = new Dictionary<FastTensorKey, FastTensorKey>();
            var perIdRefReplacements = new Dictionary<FastNodeKey, List<FastNode>>();
            var emptyAttrs = new Dictionary<string, object?>();

            foreach (var fastNode in graph.Nodes)
            {
                if (fastNode.OpCode != InternalOpCodes.TRAINABLE_PARAM_ID_REF) continue;

                var idRefOutputKey = fastNode.Outputs[0]!.Value;
                var dtype = fastNode.Attributes.GetDTypeVal(OnnxOpAttributeNames.ShrkAttrDtype)!;
                var modelIdInputKey = fastNode.Inputs[0]!.Value;

                var perReplacement = new List<FastNode>(9);

                // SHAPE(modelIdInput)
                var modelIdShapeKey = FastNodeKey.New();
                perReplacement.Add(FastNodeCreationHelpers.CreateFastNode(
                    modelIdShapeKey, OpCodes.SHAPE,
                    new Dictionary<string, object?>(),
                    new FastTensorKey?[] { modelIdInputKey }));

                // REDUCE_PROD(modelIdShape) → modelId size
                var modelIdSizeKey = FastNodeKey.New();
                perReplacement.Add(FastNodeCreationHelpers.CreateFastNode(
                    modelIdSizeKey, OpCodes.REDUCE_PROD,
                    new Dictionary<string, object?>(reduceAttrs),
                    new FastTensorKey?[] { new FastTensorKey(modelIdShapeKey, 0), null }));

                // SUB(transformSize, modelIdSize) → padAmount
                var padAmountKey = FastNodeKey.New();
                perReplacement.Add(FastNodeCreationHelpers.CreateFastNode(
                    padAmountKey, OpCodes.SUB,
                    new Dictionary<string, object?>(),
                    new FastTensorKey?[] { transformSizeTK, new FastTensorKey(modelIdSizeKey, 0) }));

                // UNSQUEEZE(padAmount) → 1-element vector
                var unsqPadAmtKey = FastNodeKey.New();
                perReplacement.Add(FastNodeCreationHelpers.CreateFastNode(
                    unsqPadAmtKey, OpCodes.UNSQUEEZE,
                    new Dictionary<string, object?>(),
                    new FastTensorKey?[] { new FastTensorKey(padAmountKey, 0), unsqAxesTK }));

                // CONCAT([unsqueezed0, unsqueezedPadAmount]) → pads vector
                var padsKey = FastNodeKey.New();
                perReplacement.Add(FastNodeCreationHelpers.CreateFastNode(
                    padsKey, OpCodes.CONCAT,
                    new Dictionary<string, object?> { [OnnxOpAttributeNames.AttrAxis] = 0L },
                    new FastTensorKey?[] { unsqueezed0TK, new FastTensorKey(unsqPadAmtKey, 0) }));

                // PAD(modelIdInput, pads, scalar0, mode=Constant)
                var paddedKey = FastNodeKey.New();
                perReplacement.Add(FastNodeCreationHelpers.CreateFastNode(
                    paddedKey, OpCodes.PAD,
                    new Dictionary<string, object?> { [OnnxOpAttributeNames.AttrMode] = PadMode.Constant },
                    new FastTensorKey?[] { modelIdInputKey, new FastTensorKey(padsKey, 0), scalar0TK, null }));

                // MUL(padded, transformVector)
                var multipliedKey = FastNodeKey.New();
                perReplacement.Add(FastNodeCreationHelpers.CreateFastNode(
                    multipliedKey, OpCodes.MUL,
                    new Dictionary<string, object?>(),
                    new FastTensorKey?[] { new FastTensorKey(paddedKey, 0), transformVecTK }));

                // REDUCE_SUM(multiplied) → flat index scalar
                var flatIndexKey = FastNodeKey.New();
                perReplacement.Add(FastNodeCreationHelpers.CreateFastNode(
                    flatIndexKey, OpCodes.REDUCE_SUM,
                    new Dictionary<string, object?>(reduceAttrs),
                    new FastTensorKey?[] { new FastTensorKey(multipliedKey, 0), null }));

                // SEQUENCE_AT(sequence[dtype], flatIndex)
                var seqAtKey = FastNodeKey.New();
                var seqAtTK = new FastTensorKey(seqAtKey, 0);
                perReplacement.Add(FastNodeCreationHelpers.CreateFastNode(
                    seqAtKey, OpCodes.SEQUENCE_AT,
                    new Dictionary<string, object?>(),
                    new FastTensorKey?[] { sequenceKeys[dtype], new FastTensorKey(flatIndexKey, 0) }));

                remap[idRefOutputKey] = seqAtTK;
                perIdRefReplacements[fastNode.Key] = perReplacement;
            }

            // Splice the new graph: shared nodes (TRAINABLE_PARAMs + sequence
            // construction + shared index helpers) go at the front — they have no
            // dependencies on the original graph. Each ID_REF gets replaced
            // in-place by its per-replacement subgraph, which inherits the ID_REF's
            // surrounding scope (so nesting is preserved). All other original
            // nodes stay in their existing positions.
            var rebuilt = new List<FastNode>(graph.Nodes.Count + newNodes.Count);
            rebuilt.AddRange(newNodes);
            foreach (var node in graph.Nodes)
            {
                if (perIdRefReplacements.TryGetValue(node.Key, out var replacement))
                    rebuilt.AddRange(replacement);
                else
                    rebuilt.Add(node);
            }
            graph.Nodes = rebuilt;

            // Apply remap to all node inputs and graph outputs.
            if (remap.Count > 0)
            {
                foreach (var key in remap.Keys.ToList())
                {
                    var target = remap[key];
                    while (remap.TryGetValue(target, out var next))
                        target = next;
                    remap[key] = target;
                }

                for (int i = 0; i < graph.Nodes.Count; i++)
                {
                    foreach (var kvp in graph.Nodes[i].FullInputs)
                    {
                        var inputList = kvp.Value;
                        for (int j = 0; j < inputList.Count; j++)
                        {
                            if (inputList[j] is FastTensorKey tk && remap.TryGetValue(tk, out var replacement))
                                inputList[j] = replacement;
                        }
                    }
                }

                for (int i = 0; i < graph.Outputs.Count; i++)
                {
                    if (remap.TryGetValue(graph.Outputs[i], out var replacement))
                        graph.Outputs[i] = replacement;
                }
            }

            System.Diagnostics.Debug.Assert(graph.IsLinearOrderValid(), "graph.IsLinearOrderValid()");
        }

        private static FastNode CreateConstantTensorDataNode(FastNodeKey nodeKey, TensorData td)
        {
            var tensorKey = new FastTensorKey(nodeKey, 0);
            var attrDefs = Definitions.NodeDefinitions[OpCodes.CONSTANT].AttributeDefs;
            var attrs = OnnxCSharpAttributes.FromCSharpVals(
                new Dictionary<string, object?> { [OnnxOpAttributeNames.AttrValue] = td }, attrDefs);

            return new FastNode
            {
                Key = nodeKey,
                OpCode = OpCodes.CONSTANT,
                Attributes = attrs,
                FullOutputs = { [""] = new List<FastTensorKey?> { tensorKey } },
            };
        }

        /// <summary>
        /// Reads all TRAINABLE_PARAM_ID_REF nodes from the QEE store, extracting each unique
        /// model ID and its corresponding initializer parameter values. The QEE History field
        /// carries per-iteration values from loops, so no graph restructuring is needed.
        /// </summary>
        private static ImmutableArray<TrainableParamInfo> ExtractModelIdInfosFromStore(
            FastComputationGraph graph,
            Dictionary<FastTensorKey, IRuntimeTensor> store)
        {
            var result = new Dictionary<ModelId, TrainableParamInfo>();

            foreach (var node in graph.Nodes)
            {
                if (node.OpCode != InternalOpCodes.TRAINABLE_PARAM_ID_REF) continue;

                var fn = node.TargetFunction;
                if (fn is null) continue;

                var inputs = node.Inputs;
                if (inputs.Count == 0 || inputs[0] is null) continue;
                if (!store.TryGetValue(inputs[0]!.Value, out var modelIdRaw)) continue;
                if (modelIdRaw is not RuntimeTensor modelIdRt) continue;

                var modelIdIterations = CollectAllIterations(modelIdRt);

                // Gather all iterations for each initializer param (inputs[2+]).
                var initParamIterations = new List<List<RuntimeTensor?>>();
                for (int i = 2; i < inputs.Count; i++)
                {
                    if (inputs[i] is null || !store.TryGetValue(inputs[i]!.Value, out var paramRaw))
                    {
                        initParamIterations.Add([]);
                        continue;
                    }
                    initParamIterations.Add(
                        CollectAllIterations(paramRaw is RuntimeTensor prt ? prt : null));
                }

                for (int iterIdx = 0; iterIdx < modelIdIterations.Count; iterIdx++)
                {
                    var modelIdIter = modelIdIterations[iterIdx];
                    if (modelIdIter?.IntData is not { } idData) continue;

                    var filteredId = idData.Where(x => x != -2).ToArray();
                    if (filteredId.Length == 0) continue;
                    var modelId = ModelId.FromLongVals(filteredId);
                    if (result.ContainsKey(modelId)) continue;

                    var paramValues = ImmutableArray.CreateBuilder<TensorData>(initParamIterations.Count);
                    bool allAvailable = true;
                    foreach (var paramList in initParamIterations)
                    {
                        // A param with Count==1 is a global constant (outside any loop); reuse it for
                        // every iteration rather than treating missing indices as unavailable.
                        RuntimeTensor? paramRt = iterIdx < paramList.Count ? paramList[iterIdx]
                            : paramList.Count == 1 ? paramList[0]
                            : null;
                        var td = paramRt is not null ? TensorDataConverter.ToTensorData(paramRt) : null;
                        if (td is null) { allAvailable = false; break; }
                        paramValues.Add(td);
                    }
                    if (!allAvailable) continue;

                    result[modelId] = new TrainableParamInfo
                    {
                        SpecificModelId = modelId,
                        TrainableParamInputParamValues = paramValues.ToImmutable(),
                        TargetFn = fn,
                    };
                }
            }

            return [.. result.Values];
        }

        private static List<RuntimeTensor?> CollectAllIterations(RuntimeTensor? rt)
        {
            var list = new List<RuntimeTensor?>();
            if (rt?.History is { } history)
                foreach (var h in history)
                    list.Add(h as RuntimeTensor);
            list.Add(rt);
            return list;
        }
    }

    /// <summary>
    /// Runs a fully-native convergence loop over <see cref="FastFoldConstants"/>,
    /// <see cref="FastFoldConstantConditionBranches"/>,
    /// <see cref="FastFoldSequences"/>, and
    /// <see cref="FastFoldConstantIterationLoops"/> — four pure
    /// <see cref="FastComputationGraph"/> passes, no <c>ComputationGraph</c>
    /// round-trip anywhere.
    /// <para>
    /// Native equivalents cover every fold case the (now-deleted) CG Simplify
    /// pipeline did: full INSERT/ERASE/LENGTH/AT coverage on `knownSequences`,
    /// scan vars, dynamic cond chain via AND/WHERE, nested LOOP/IF with
    /// `GraphOpenNodeKey` rewiring, per-iteration identifier-template /
    /// model-id updates, and loop-invariant body-node sharing.
    /// </para>
    /// </summary>
    internal static class FastSimplify
    {
        public static void Process(FastComputationGraph graph, ComputeContext? context = null)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));

            FastGraphCycleDetector.AssertAcyclic(graph, "FastSimplify entry");

            // Outer convergence loop: run the native folding pipeline to a fixed
            // point, then run `FastFoldConstantIterationLoops` to unroll every
            // eligible CONSTANT-maxIter LOOP. Unrolling typically turns the
            // `LOOP_OPEN`'s iteration-index output into a scalar CONSTANT, opening
            // up new native folding work, so we re-enter the inner native loop
            // after each unroll pass. Historically this called the full CG
            // `Simplify`; then narrowed to CG `FoldConstantIterationLoops` +
            // `FoldSequencesProcessor`; then just CG `FoldConstantIterationLoops`;
            // it is now fully native.
            const int MAX_OUTER_ITERATIONS = 1000;
            int outerIteration = 0;
            bool outerChanged = true;
            while (outerChanged)
            {
                if (++outerIteration > MAX_OUTER_ITERATIONS)
                    throw new InvalidOperationException(
                        $"FastSimplify: Exceeded maximum outer iteration limit ({MAX_OUTER_ITERATIONS}). " +
                        "This suggests an infinite loop in the graph simplification process.");

                outerChanged = false;

                // Inner native convergence: alternate FastFoldConstants (to its own
                // fixed point) with one pass each of FastFoldConstantConditionBranches
                // and FastFoldSequences. Folding an IF can expose new constants in
                // the previously-dead branch's surroundings; folding SEQUENCE_AT /
                // SEQUENCE_LENGTH can collapse a SEQUENCE_CONSTRUCT (or upstream
                // INSERT/ERASE chain) whose only remaining consumers just disappeared.
                const int MAX_INNER_ITERATIONS = 1000;
                int innerIteration = 0;
                bool innerChanged = true;
                while (innerChanged)
                {
                    if (++innerIteration > MAX_INNER_ITERATIONS)
                        throw new InvalidOperationException(
                            $"FastSimplify: Exceeded maximum inner iteration limit ({MAX_INNER_ITERATIONS}). " +
                            "This suggests an infinite loop in the graph folding process.");

                    innerChanged = false;
                    while (FastFoldConstants.Process(graph))
                    {
                        innerChanged = true;
                        FastGraphCycleDetector.AssertAcyclic(graph, $"FastSimplify after FastFoldConstants (outer {outerIteration}, inner {innerIteration})");
                    }

                    if (FastFoldConstantConditionBranches.Process(graph))
                    {
                        innerChanged = true;
                        FastGraphCycleDetector.AssertAcyclic(graph, $"FastSimplify after FastFoldConstantConditionBranches (outer {outerIteration}, inner {innerIteration})");
                    }

                    if (FastFoldSequences.Process(graph))
                    {
                        innerChanged = true;
                        FastGraphCycleDetector.AssertAcyclic(graph, $"FastSimplify after FastFoldSequences (outer {outerIteration}, inner {innerIteration})");
                    }
                }

                // Native loop-unroll pass: handles the simple subset of
                // FoldConstantIterationLoops (no scan vars, no cond chain, no
                // nested control flow, no identifier-template updates). Anything
                // it unrolls is gone before the CG round-trip runs, so the
                // round-trip only sees loops that need the full CG machinery.
                if (FastFoldConstantIterationLoops.Process(graph))
                {
                    FastGraphCycleDetector.AssertAcyclic(graph, $"FastSimplify after FastFoldConstantIterationLoops (outer {outerIteration})");
                    outerChanged = true;
                }

                // No more CG round-trip here. Stages A–G of
                // `FastFoldConstantIterationLoops` cover every loop the CG
                // `FoldConstantIterationLoops` pass could unroll, plus a superset:
                // scan vars, dynamic cond chains, nested LOOP / IF control flow,
                // identifier-template / model-id updates per iteration. The two
                // residual native disqualifiers (N==0 with scan vars; scan vars
                // combined with a dynamic cond chain) leave the loop in place,
                // which is correct for downstream passes — they just don't get
                // unrolled statically. No test in the Quick+Standard tiers hits
                // either residual case.
            }
        }
    }

    /// <summary>
    /// Folds every <c>IF_OPEN</c> / <c>IF_CLOSE</c> pair whose condition input is produced by
    /// a <c>CONSTANT</c> node. The winning branch's input tensor keys are spliced
    /// into every consumer of the <c>IF_CLOSE</c>'s outputs (and into
    /// <see cref="FastComputationGraph.Outputs"/> when applicable); the
    /// <c>IF_OPEN</c> / <c>IF_CLOSE</c> nodes themselves and the losing branch's
    /// now-unreachable body are then dropped from
    /// <see cref="FastComputationGraph.Nodes"/>.
    /// </summary>
    internal static class FastFoldConstantConditionBranches
    {
        /// <summary>
        /// Runs one folding pass. Returns true if any IF was folded.
        /// </summary>
        public static bool Process(FastComputationGraph graph)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));

            // Quick exit: no IF_CLOSE → nothing to do.
            bool hasIfClose = false;
            for (int i = 0; i < graph.Nodes.Count; i++)
            {
                if (graph.Nodes[i].OpCode == OpCodes.IF_CLOSE) { hasIfClose = true; break; }
            }
            if (!hasIfClose) return false;

            var nodeByKey = FastProcessorHelper.BuildNodeByKey(graph);
            var producerByOutput = new Dictionary<FastTensorKey, FastNode>(graph.Nodes.Count * 2);
            foreach (var producer in graph.Nodes)
            {
                foreach (var kvp in producer.FullOutputs)
                    foreach (var output in kvp.Value)
                        if (output is FastTensorKey tk && !tk.IsEmpty)
                            producerByOutput[tk] = producer;
            }

            var remap = new Dictionary<FastTensorKey, FastTensorKey>();
            var nodesToRemove = new HashSet<FastNodeKey>();

            foreach (var closeNode in graph.Nodes)
            {
                if (closeNode.OpCode != OpCodes.IF_CLOSE) continue;
                if (closeNode.GraphOpenNodeKey is not FastNodeKey openKey || openKey.IsEmpty) continue;
                if (!nodeByKey.TryGetValue(openKey, out var openNode)) continue;

                var openInputs = openNode.Inputs;
                if (openInputs.Count == 0 || openInputs[0] is not FastTensorKey condKey || condKey.IsEmpty)
                    continue;

                if (!producerByOutput.TryGetValue(condKey, out var producer)) continue;
                if (producer.OpCode != OpCodes.CONSTANT) continue;

                var tensorVal = producer.Attributes.GetTensorVal(OnnxOpAttributeNames.AttrValue);
                if (tensorVal is null || tensorVal.DType != DType.Bool) continue;

                bool boolVal = tensorVal.As<bit>().AccessMemory()[0];

                if (!closeNode.FullInputs.TryGetValue(OnnxOpAttributeNames.AttrThenBranch, out var thenInputs)) continue;
                if (!closeNode.FullInputs.TryGetValue(OnnxOpAttributeNames.AttrElseBranch, out var elseInputs)) continue;
                var winningInputs = boolVal ? thenInputs : elseInputs;

                // IF_CLOSE's outputs live under the default key; one output per branch slot.
                if (!closeNode.FullOutputs.TryGetValue("", out var outputs)) continue;
                Debug.Assert(outputs.Count == winningInputs.Count,
                    "IF_CLOSE output count must match branch input count.");

                for (int i = 0; i < outputs.Count; i++)
                {
                    if (outputs[i] is not FastTensorKey outKey || outKey.IsEmpty) continue;
                    if (winningInputs[i] is not FastTensorKey winKey || winKey.IsEmpty) continue;
                    remap[outKey] = winKey;
                }

                nodesToRemove.Add(openKey);
                nodesToRemove.Add(closeNode.Key);
            }

            if (nodesToRemove.Count == 0) return false;

            // Resolve transitive remap chains (an IF_CLOSE output may itself be the
            // input to another IF_CLOSE we just folded).
            foreach (var key in remap.Keys.ToList())
            {
                var target = remap[key];
                while (remap.TryGetValue(target, out var next)) target = next;
                remap[key] = target;
            }

            // Apply remap to every node's inputs and to the graph's outputs.
            foreach (var node in graph.Nodes)
            {
                foreach (var kvp in node.FullInputs)
                {
                    var inputList = kvp.Value;
                    for (int j = 0; j < inputList.Count; j++)
                    {
                        if (inputList[j] is FastTensorKey key && remap.TryGetValue(key, out var replacement))
                            inputList[j] = replacement;
                    }
                }
            }
            for (int i = 0; i < graph.Outputs.Count; i++)
            {
                if (remap.TryGetValue(graph.Outputs[i], out var replacement))
                    graph.Outputs[i] = replacement;
            }

            // Drop the folded IF_OPEN / IF_CLOSE pairs. The losing branch's body
            // nodes are now unreachable from graph.Outputs and from any surviving
            // node's inputs; sweep them via RemoveUnreachableNodes so they don't
            // linger as dead weight in subsequent passes.
            graph.Nodes.RemoveAll(n => nodesToRemove.Contains(n.Key));
            FastProcessorHelper.RemoveUnreachableNodes(graph);

            return true;
        }
    }

    /// <summary>
    /// Tracks "known" sequences — any tensor whose producing chain flows through
    /// <c>SEQUENCE_EMPTY</c> / <c>SEQUENCE_CONSTRUCT</c>, optionally modified by
    /// <c>SEQUENCE_INSERT</c> / <c>SEQUENCE_ERASE</c> with constant indices — and
    /// folds its consumers:
    /// <list type="bullet">
    ///   <item><c>SEQUENCE_AT(known_seq, const_idx)</c> → remap to the element
    ///     <c>known_seq[const_idx]</c>'s <see cref="FastTensorKey"/>. Once every
    ///     consumer is folded the producing <c>SEQUENCE_CONSTRUCT</c> becomes
    ///     unreachable and any upstream trainable-param nodes it uniquely kept
    ///     alive (e.g. those of an eliminated IF branch) are swept by
    ///     <see cref="FastProcessorHelper.RemoveUnreachableNodes"/>.</item>
    ///   <item><c>SEQUENCE_LENGTH(known_seq)</c> → emit a fresh int64 scalar
    ///     <c>CONSTANT</c> and remap.</item>
    /// </list>
    /// Folding is pure: a rewrite only substitutes a sequence consumer with a
    /// reference to an element already in the graph (or a trivially-valued
    /// constant). Sequences produced inside a loop body but consumed inside the
    /// same body fold correctly because each iteration's re-evaluated consumer
    /// simply sees the element tensor of that iteration; sequences consumed
    /// outside the body are protected by the fact that a <c>LOOP_CLOSE</c>'s
    /// sequence output is a fresh <see cref="FastTensorKey"/> that was never added
    /// to the <c>knownSequences</c> map.
    /// </summary>
    internal static class FastFoldSequences
    {
        /// <summary>
        /// Runs one folding pass. Returns true if any sequence consumer was folded.
        /// </summary>
        public static bool Process(FastComputationGraph graph)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));

            // Quick exit: no foldable consumers → nothing to do.
            bool hasConsumer = false;
            for (int i = 0; i < graph.Nodes.Count; i++)
            {
                var op = graph.Nodes[i].OpCode;
                if (op == OpCodes.SEQUENCE_AT || op == OpCodes.SEQUENCE_LENGTH)
                {
                    hasConsumer = true;
                    break;
                }
            }
            if (!hasConsumer) return false;

            var producerByOutput = new Dictionary<FastTensorKey, FastNode>(graph.Nodes.Count * 2);
            foreach (var producer in graph.Nodes)
            {
                foreach (var kvp in producer.FullOutputs)
                    foreach (var output in kvp.Value)
                        if (output is FastTensorKey tk && !tk.IsEmpty)
                            producerByOutput[tk] = producer;
            }

            // Walk in topological order (FastSimplify's inner loop keeps graph.Nodes
            // topologically ordered) and build a map from sequence-output FastTensorKey
            // → ordered element list. An entry here means every position in the
            // sequence is statically resolvable to an existing FastTensorKey.
            var knownSequences = new Dictionary<FastTensorKey, List<FastTensorKey?>>();
            foreach (var node in graph.Nodes)
            {
                switch (node.OpCode)
                {
                    case OpCodes.SEQUENCE_CONSTRUCT:
                    {
                        if (node.Outputs.Count == 0) break;
                        if (node.Outputs[0] is not FastTensorKey outKey || outKey.IsEmpty) break;
                        knownSequences[outKey] = new List<FastTensorKey?>(node.Inputs);
                        break;
                    }
                    case OpCodes.SEQUENCE_EMPTY:
                    {
                        if (node.Outputs.Count == 0) break;
                        if (node.Outputs[0] is not FastTensorKey outKey || outKey.IsEmpty) break;
                        knownSequences[outKey] = new List<FastTensorKey?>();
                        break;
                    }
                    case OpCodes.SEQUENCE_INSERT:
                    {
                        if (node.Inputs.Count < 2) break;
                        if (node.Inputs[0] is not FastTensorKey seqKey || seqKey.IsEmpty) break;
                        if (!knownSequences.TryGetValue(seqKey, out var elements)) break;
                        long length = elements.Count;
                        long idx;
                        // Position is slot 2 and is optional; when missing/null, append.
                        FastTensorKey? posKey = node.Inputs.Count >= 3 ? node.Inputs[2] : null;
                        if (posKey is null || posKey.Value.IsEmpty)
                        {
                            idx = length;
                        }
                        else
                        {
                            if (!producerByOutput.TryGetValue(posKey.Value, out var posProducer)) break;
                            if (posProducer.OpCode != OpCodes.CONSTANT) break;
                            var v = ReadConstantLong(posProducer);
                            if (v is null) break;
                            idx = v.Value;
                            if (idx < -length || idx > length) break;
                            if (idx < 0) idx += length;
                        }
                        var element = node.Inputs[1];
                        var newList = new List<FastTensorKey?>(elements);
                        newList.Insert((int)idx, element);
                        if (node.Outputs.Count == 0) break;
                        if (node.Outputs[0] is not FastTensorKey outKey || outKey.IsEmpty) break;
                        knownSequences[outKey] = newList;
                        break;
                    }
                    case OpCodes.SEQUENCE_ERASE:
                    {
                        if (node.Inputs.Count < 2) break;
                        if (node.Inputs[0] is not FastTensorKey seqKey || seqKey.IsEmpty) break;
                        if (!knownSequences.TryGetValue(seqKey, out var elements)) break;
                        long length = elements.Count;
                        if (node.Inputs[1] is not FastTensorKey idxKey || idxKey.IsEmpty) break;
                        if (!producerByOutput.TryGetValue(idxKey, out var idxProducer)) break;
                        if (idxProducer.OpCode != OpCodes.CONSTANT) break;
                        var v = ReadConstantLong(idxProducer);
                        if (v is null) break;
                        long idx = v.Value;
                        if (idx < -length || idx >= length) break;
                        if (idx < 0) idx += length;
                        var newList = new List<FastTensorKey?>(elements);
                        newList.RemoveAt((int)idx);
                        if (node.Outputs.Count == 0) break;
                        if (node.Outputs[0] is not FastTensorKey outKey || outKey.IsEmpty) break;
                        knownSequences[outKey] = newList;
                        break;
                    }
                }
            }

            var remap = new Dictionary<FastTensorKey, FastTensorKey>();
            var nodesToRemove = new HashSet<FastNodeKey>();
            var nodesToAdd = new List<FastNode>();

            foreach (var node in graph.Nodes)
            {
                switch (node.OpCode)
                {
                    case OpCodes.SEQUENCE_AT:
                    {
                        if (node.Inputs.Count < 2) break;
                        if (node.Inputs[0] is not FastTensorKey seqKey || seqKey.IsEmpty) break;
                        if (!knownSequences.TryGetValue(seqKey, out var elements)) break;
                        if (node.Inputs[1] is not FastTensorKey idxKey || idxKey.IsEmpty) break;
                        if (!producerByOutput.TryGetValue(idxKey, out var idxProducer)) break;
                        if (idxProducer.OpCode != OpCodes.CONSTANT) break;
                        var v = ReadConstantLong(idxProducer);
                        if (v is null) break;
                        long length = elements.Count;
                        long idx = v.Value;
                        if (idx < -length || idx >= length) break;
                        if (idx < 0) idx += length;
                        if (elements[(int)idx] is not FastTensorKey winKey || winKey.IsEmpty) break;
                        if (node.Outputs.Count == 0) break;
                        if (node.Outputs[0] is not FastTensorKey atOut || atOut.IsEmpty) break;
                        remap[atOut] = winKey;
                        nodesToRemove.Add(node.Key);
                        break;
                    }
                    case OpCodes.SEQUENCE_LENGTH:
                    {
                        if (node.Inputs.Count < 1) break;
                        if (node.Inputs[0] is not FastTensorKey seqKey || seqKey.IsEmpty) break;
                        if (!knownSequences.TryGetValue(seqKey, out var elements)) break;
                        if (node.Outputs.Count == 0) break;
                        if (node.Outputs[0] is not FastTensorKey lenOut || lenOut.IsEmpty) break;
                        var newNodeKey = FastNodeKey.New();
                        var constNode = FastNodeCreationHelpers.CreateConstantNode(
                            newNodeKey, new long[0], (long)elements.Count);
                        nodesToAdd.Add(constNode);
                        remap[lenOut] = new FastTensorKey(newNodeKey, 0);
                        nodesToRemove.Add(node.Key);
                        break;
                    }
                }
            }

            if (nodesToRemove.Count == 0 && nodesToAdd.Count == 0)
                return false;

            // Resolve transitive remap chains.
            foreach (var key in remap.Keys.ToList())
            {
                var target = remap[key];
                while (remap.TryGetValue(target, out var next)) target = next;
                remap[key] = target;
            }

            // Apply remap to every node's inputs and to graph outputs.
            foreach (var node in graph.Nodes)
            {
                foreach (var kvp in node.FullInputs)
                {
                    var inputList = kvp.Value;
                    for (int j = 0; j < inputList.Count; j++)
                    {
                        if (inputList[j] is FastTensorKey key && remap.TryGetValue(key, out var replacement))
                            inputList[j] = replacement;
                    }
                }
            }
            for (int i = 0; i < graph.Outputs.Count; i++)
            {
                if (remap.TryGetValue(graph.Outputs[i], out var replacement))
                    graph.Outputs[i] = replacement;
            }

            // Drop the folded consumer nodes; prepend any freshly-minted scalar
            // CONSTANTs (for SEQUENCE_LENGTH) — they have no inputs, so the front
            // is a topologically valid placement and CONSTANTs carry no scope of
            // their own. Sweep any producers whose only consumers just
            // disappeared (a SEQUENCE_CONSTRUCT whose AT/LENGTH consumers are all
            // folded, or an upstream trainable-param kept alive solely by the
            // construct).
            if (nodesToAdd.Count > 0)
            {
                var combined = new List<FastNode>(graph.Nodes.Count + nodesToAdd.Count);
                combined.AddRange(nodesToAdd);
                combined.AddRange(graph.Nodes);
                graph.Nodes = combined;
            }
            graph.Nodes.RemoveAll(n => nodesToRemove.Contains(n.Key));
            FastProcessorHelper.RemoveUnreachableNodes(graph);
            System.Diagnostics.Debug.Assert(graph.IsLinearOrderValid(), "graph.IsLinearOrderValid()");

            return true;
        }

        /// <summary>
        /// Reads a long value from a <c>CONSTANT</c> node, supporting both the
        /// <c>value_int</c> attribute form and the <c>value</c> tensor form (int64
        /// scalar). Returns null if neither matches.
        /// </summary>
        private static long? ReadConstantLong(FastNode constNode)
        {
            var attrs = constNode.Attributes;
            if (!attrs.IsDefaultValue(OnnxOpAttributeNames.AttrValueInt))
                return attrs.GetLongVal(OnnxOpAttributeNames.AttrValueInt);

            var tensorVal = attrs.GetTensorVal(OnnxOpAttributeNames.AttrValue);
            if (tensorVal is null) return null;
            if (tensorVal.DType != DType.Int64) return null;
            if (tensorVal.Shape.Dims.Length != 0) return null; // require scalar
            return tensorVal.As<int64>().AccessMemory()[0];
        }
    }

    /// <summary>
    /// Unrolls a <c>LOOP_OPEN</c> / <c>LOOP_CLOSE</c> pair whose <c>maxIter</c> input is a
    /// <c>CONSTANT</c>, cloning the body nodes with fresh <see cref="FastNodeKey"/>s per
    /// iteration and wiring each iteration's loop-variable inputs to the previous
    /// iteration's body outputs (or to the <c>LOOP_OPEN</c>'s initializers on iteration 0).
    /// <para>
    /// Scope — simple loops only. A loop is only unrolled natively when every one of
    /// these holds; otherwise <see cref="FastSimplify"/> leaves the loop in place.
    /// <list type="bullet">
    ///   <item>The <c>maxIter</c> producer is a <c>CONSTANT</c> with a non-negative
    ///     int64 scalar value.</item>
    ///   <item>The <c>LOOP_OPEN</c>'s continue-condition input (slot 1) is either
    ///     absent / null, or is a <c>CONSTANT(true)</c>. No <c>IfElse</c>-chained
    ///     early-termination logic is needed.</item>
    ///   <item>The <c>LOOP_CLOSE</c> has no scan variables: the number of close outputs
    ///     equals the number of loop variables declared on the <c>LOOP_OPEN</c>
    ///     (<c>open.Inputs.Count - 2</c>, or 0 when the loop has no loop vars).</item>
    ///   <item>The body contains no nested <c>LOOP_OPEN</c> / <c>LOOP_CLOSE</c> /
    ///     <c>IF_OPEN</c> / <c>IF_CLOSE</c> — nested control flow would require
    ///     rewiring <see cref="FastNode.GraphOpenNodeKey"/> on cloned close nodes.</item>
    ///   <item>No body node has a non-null <see cref="FastNode.IdentifierTemplate"/>
    ///     and no body node's attributes include <c>ShrkAttrLocalModelId</c> or
    ///     <c>ShrkAttrRelativeModelId</c> — the CG path rewrites these per unrolled
    ///     copy, and the Fast path does not yet (serialize / substitute / reserialize
    ///     the identifier template and update the model-id attribute arrays).</item>
    ///   <item>Every output of every body node is consumed only by other body nodes
    ///     or by the matching <c>LOOP_CLOSE</c> — no external references into the
    ///     body. (Also disqualifies body-produced graph outputs, which in a
    ///     well-formed loop don't occur.)</item>
    /// </list>
    /// </para>
    /// <para>
    /// When eligible, the unroll runs in place: the <c>LOOP_OPEN</c>, <c>LOOP_CLOSE</c>,
    /// and original body nodes are removed; N fresh copies of the body (iteration 0
    /// reuses the originals with remapped inputs to avoid churn; iterations 1..N-1
    /// are cloned with fresh keys); consumers of the <c>LOOP_CLOSE</c>'s loop-var
    /// outputs are remapped to the final iteration's corresponding body outputs.
    /// A zero-iteration loop (N == 0) is a special case — the CLOSE's loop-var
    /// outputs are remapped directly to the OPEN's initializer inputs and the whole
    /// body is dropped.
    /// </para>
    /// </summary>
    internal static class FastFoldConstantIterationLoops
    {
        /// <summary>
        /// Runs one unroll pass. Returns true if any eligible loop was unrolled.
        /// Non-eligible loops (scan vars, cond chain, nested control flow, identifier-template
        /// updates needed, etc.) are left untouched for <see cref="FastSimplify"/>'s CG
        /// round-trip to handle.
        ///
        /// <para>
        /// Precondition: no loop body may contain a <c>TRAINABLE_PARAM_REF</c>,
        /// <c>TRAINABLE_PARAM_ID_REF</c>, <c>TRAINABLE_PARAM_MODEL_REF</c>, or
        /// <c>MODULE_SET_HYPERPARAMS</c> node. The full
        /// <see cref="FastComputationGraphExtensions.ToConcreteArchitecture"/> pipeline
        /// resolves all four into plain <c>TRAINABLE_PARAM</c> (or, for
        /// <c>MODULE_SET_HYPERPARAMS</c>, removes the node via
        /// <see cref="FastUnpackModelStruct"/>) before <see cref="FastSimplify"/> calls
        /// in here, so the per-iteration identifier-template / model-id rewrite that
        /// used to run on those op-codes is dead in production and has been removed.
        /// <see cref="UnrollOne"/> asserts this invariant on every cloned body node.
        /// </para>
        /// </summary>
        public static bool Process(FastComputationGraph graph)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));

            // Quick exit: no LOOP_OPEN at all — nothing to do.
            bool hasLoopOpen = false;
            for (int i = 0; i < graph.Nodes.Count; i++)
            {
                if (graph.Nodes[i].OpCode == OpCodes.LOOP_OPEN) { hasLoopOpen = true; break; }
            }
            if (!hasLoopOpen) return false;

            // Build producer / consumer indices used by every eligibility check and the
            // unroll itself. Recomputed once here; the unroll mutates graph.Nodes so if
            // we find more than one eligible loop in a single pass we iterate and re-enter.
            bool anyUnrolled = false;

            while (true)
            {
                int openIdx = -1;
                int closeIdx = -1;
                FastNode? openNode = null;
                FastNode? closeNode = null;
                long iterCount = -1;

                // Re-scan each pass so positions are current after the previous rewrite.
                var producerByOutput = new Dictionary<FastTensorKey, FastNode>(graph.Nodes.Count * 2);
                foreach (var producer in graph.Nodes)
                {
                    foreach (var kvp in producer.FullOutputs)
                        foreach (var output in kvp.Value)
                            if (output is FastTensorKey tk && !tk.IsEmpty)
                                producerByOutput[tk] = producer;
                }

                for (int i = 0; i < graph.Nodes.Count; i++)
                {
                    var n = graph.Nodes[i];
                    if (n.OpCode != OpCodes.LOOP_OPEN) continue;
                    var inputs = n.Inputs;
                    if (inputs.Count == 0 || inputs[0] is not FastTensorKey maxIterKey || maxIterKey.IsEmpty)
                        continue;
                    if (!producerByOutput.TryGetValue(maxIterKey, out var maxIterProducer)) continue;
                    if (maxIterProducer.OpCode != OpCodes.CONSTANT) continue;
                    var v = ReadConstantLong(maxIterProducer);
                    if (v is null || v.Value < 0) continue;

                    // Locate the matching CLOSE by scanning forward.
                    FastNode? candidateClose = null;
                    int candidateCloseIdx = -1;
                    for (int j = i + 1; j < graph.Nodes.Count; j++)
                    {
                        var cn = graph.Nodes[j];
                        if (cn.OpCode != OpCodes.LOOP_CLOSE) continue;
                        if (cn.GraphOpenNodeKey is FastNodeKey ok && ok == n.Key)
                        {
                            candidateClose = cn;
                            candidateCloseIdx = j;
                            break;
                        }
                    }
                    if (candidateClose is null) continue;

                    // Eligibility gate.
                    if (!IsEligibleForNativeUnroll(graph, n, i, candidateClose, candidateCloseIdx, producerByOutput))
                        continue;

                    openNode = n; openIdx = i;
                    closeNode = candidateClose; closeIdx = candidateCloseIdx;
                    iterCount = v.Value;
                    break;
                }

                if (openNode is null || closeNode is null) break;

                UnrollOne(graph, openNode, openIdx, closeNode, closeIdx, iterCount);
                anyUnrolled = true;

                FastGraphCycleDetector.AssertAcyclic(graph, "FastFoldConstantIterationLoops after UnrollOne");
                // Loop back: there may be another eligible loop to unroll.
            }

            if (anyUnrolled)
            {
                // UnrollOne splices clones in place of the loop body, so the graph
                // remains topologically ordered and properly nested by construction
                // — no Kahn re-sort needed here.
                FastProcessorHelper.RemoveUnreachableNodes(graph);
                System.Diagnostics.Debug.Assert(graph.IsLinearOrderValid(), "graph.IsLinearOrderValid()");
            }
            return anyUnrolled;
        }

        private static bool IsEligibleForNativeUnroll(
            FastComputationGraph graph,
            FastNode openNode, int openIdx,
            FastNode closeNode, int closeIdx,
            Dictionary<FastTensorKey, FastNode> producerByOutput)
        {
            // (1) Continue-condition: any FastTensorKey is acceptable — dynamic or
            // constant. When present, UnrollOne emits an AND / WHERE chain that
            // "freezes" the loop-var outputs at the iteration where the body's
            // break condition first goes false. The scan-var + cond-chain combo
            // is still disqualified (see the Stage-E scan gate below) because
            // the shape-frozen output semantics for scan vars under early
            // termination would need more machinery.
            var openInputs = openNode.Inputs;

            // (2) Loop-var / scan-var structure. LOOP_OPEN declares the loop-var count
            // via its inputs beyond [maxIter, cond]. LOOP_CLOSE has inputs
            // [break, ...loopVarBodyOuts, ...scanVarBodyOuts] and outputs
            // [...loopVarCloseOuts, ...scanVarCloseOuts]. The loop-var and scan-var
            // counts must be consistent between the two sides.
            int nLoop = Math.Max(0, openInputs.Count - 2);
            int nScan = closeNode.Outputs.Count - nLoop;
            if (nScan < 0) return false;
            if (closeNode.Inputs.Count != 1 + nLoop + nScan) return false;

            // (2b) Scan vars require an iteration to produce a concrete shape; a
            // zero-iteration loop with scan outputs would need a dtype-aware empty
            // tensor, which the native pass doesn't have without a TensorInfo
            // round-trip. Disqualify and let the CG path handle that degenerate case.
            long iterCountPreview = 0;
            if (openInputs.Count > 0 && openInputs[0] is FastTensorKey miKey && !miKey.IsEmpty
                && producerByOutput.TryGetValue(miKey, out var miProducer)
                && miProducer.OpCode == OpCodes.CONSTANT
                && ReadConstantLong(miProducer) is long mi) iterCountPreview = mi;
            if (iterCountPreview == 0 && nScan > 0) return false;

            // (2c) The "cond chain is dynamic" signal lives in CLOSE.Inputs[0] — the
            // body's per-iteration continue-when output. OPEN.Inputs[1] is just the
            // initial cond (usually a CONSTANT(true)) and doesn't tell us whether
            // the body drives a dynamic break. Mixing scan vars with a dynamic
            // body-break is not supported natively — the shape of a scan output
            // under early termination is tricky to express statically. Anything
            // with nScan > 0 needs a static-true body break; otherwise CG handles it.
            bool bodyBreakIsStaticTrue = IsStaticTrue(closeNode.Inputs[0], producerByOutput);
            if (nScan > 0 && !bodyBreakIsStaticTrue) return false;

            // (3) Body: collect nodes positionally between OPEN and CLOSE. Nested
            // LOOP / IF control flow is allowed: when cloning a nested close node
            // whose GraphOpenNodeKey refers to a body OPEN also being cloned this
            // iteration, UnrollOne rewires GraphOpenNodeKey to the cloned OPEN's
            // new FastNodeKey (Stage G). Structural validity invariant: every nested
            // close node in the body must have its matching open node also inside
            // the body (no cross-body OPEN/CLOSE pairs). If we see a stray close
            // node whose open is outside the body, we bail — that would mean a
            // broken graph that the Fast pipeline should've rejected earlier.
            var bodyNodeSet = new HashSet<FastNodeKey>();
            for (int j = openIdx + 1; j < closeIdx; j++)
            {
                var b = graph.Nodes[j];
                bodyNodeSet.Add(b.Key);
            }
            for (int j = openIdx + 1; j < closeIdx; j++)
            {
                var b = graph.Nodes[j];
                if ((b.OpCode == OpCodes.LOOP_CLOSE || b.OpCode == OpCodes.IF_CLOSE)
                    && b.GraphOpenNodeKey is FastNodeKey gok && !gok.IsEmpty
                    && !bodyNodeSet.Contains(gok))
                    return false;
            }

            // (4) Body outputs must be consumed only by other body nodes, the CLOSE, or
            // nowhere. A body output that leaks to a non-body consumer would break after
            // the unroll (iteration 0 reuses the originals in place, but iterations 1..N-1
            // produce fresh keys the leaking consumer wouldn't see).
            var bodyOutputKeys = new HashSet<FastTensorKey>();
            for (int j = openIdx + 1; j < closeIdx; j++)
            {
                foreach (var kvp in graph.Nodes[j].FullOutputs)
                    foreach (var ok in kvp.Value)
                        if (ok is FastTensorKey tk && !tk.IsEmpty) bodyOutputKeys.Add(tk);
            }

            for (int j = 0; j < graph.Nodes.Count; j++)
            {
                if (j > openIdx && j < closeIdx) continue; // body node — self-consumption fine
                if (j == closeIdx) continue; // CLOSE allowed
                var c = graph.Nodes[j];
                foreach (var kvp in c.FullInputs)
                    foreach (var ik in kvp.Value)
                        if (ik is FastTensorKey ikt && bodyOutputKeys.Contains(ikt)) return false;
            }
            foreach (var outK in graph.Outputs)
                if (bodyOutputKeys.Contains(outK)) return false;

            // OPEN outputs may be consumed by body nodes OR by the matching CLOSE
            // (CLOSE's break input typically threads through OPEN's vestigal-true
            // output, and loop-var pass-through in degenerate zero-body loops). No
            // other consumer may reference them — post-loop code should use CLOSE
            // outputs, not OPEN outputs.
            var openOutputKeys = new HashSet<FastTensorKey>();
            foreach (var kvp in openNode.FullOutputs)
                foreach (var ok in kvp.Value)
                    if (ok is FastTensorKey tk && !tk.IsEmpty) openOutputKeys.Add(tk);

            for (int j = 0; j < graph.Nodes.Count; j++)
            {
                if (j == openIdx) continue;
                if (j > openIdx && j < closeIdx) continue; // body
                if (j == closeIdx) continue; // CLOSE allowed
                var c = graph.Nodes[j];
                foreach (var kvp in c.FullInputs)
                    foreach (var ik in kvp.Value)
                        if (ik is FastTensorKey ikt && openOutputKeys.Contains(ikt)) return false;
            }
            foreach (var outK in graph.Outputs)
                if (openOutputKeys.Contains(outK)) return false;

            return true;
        }

        /// <summary>
        /// Returns true when <paramref name="condKey"/> is either null/empty (meaning
        /// "no cond", i.e. always true by convention) or a <c>CONSTANT</c> bool scalar
        /// whose value is true. Used to short-circuit the cond-chain machinery when
        /// no dynamic termination is possible.
        /// </summary>
        private static bool IsStaticTrue(FastTensorKey? condKey, Dictionary<FastTensorKey, FastNode> producerByOutput)
        {
            if (condKey is null || condKey.Value.IsEmpty) return true;
            if (!producerByOutput.TryGetValue(condKey.Value, out var prod)) return false;
            if (prod.OpCode != OpCodes.CONSTANT) return false;
            var tv = prod.Attributes.GetTensorVal(OnnxOpAttributeNames.AttrValue);
            if (tv is null || tv.DType != DType.Bool) return false;
            if (tv.Shape.Dims.Length != 0) return false;
            return tv.As<bit>().AccessMemory()[0];
        }

        /// <summary>
        /// Re-pass invariant probe for the Stage G loop-dependency analysis. Returns
        /// true if any not-yet-marked body node has an input in
        /// <paramref name="loopDependentTensors"/>. In a well-formed graph this is
        /// always false after the first pass + scope-pair propagation +
        /// output-promotion — see the Debug.Assert caller in <see cref="UnrollOne"/>
        /// for the full reasoning.
        /// </summary>
        private static bool AnyUnmarkedBodyNodeHasLoopDepInput(
            List<FastNode> bodyNodes,
            HashSet<FastNodeKey> mustCloneBodyKeys,
            HashSet<FastTensorKey> loopDependentTensors)
        {
            foreach (var b in bodyNodes)
            {
                if (mustCloneBodyKeys.Contains(b.Key)) continue;
                foreach (var kvp in b.FullInputs)
                    foreach (var ik in kvp.Value)
                        if (ik is FastTensorKey ikt && loopDependentTensors.Contains(ikt))
                            return true;
            }
            return false;
        }

        private static void UnrollOne(
            FastComputationGraph graph,
            FastNode openNode, int openIdx,
            FastNode closeNode, int closeIdx,
            long iterCount)
        {
            var openInputs = openNode.Inputs;
            int nLoop = Math.Max(0, openInputs.Count - 2);
            int nScan = closeNode.Outputs.Count - nLoop;
            var openOutputs = openNode.Outputs; // [iterIndex, vestigalTrue, ...loopVars]

            // Collect body node references before we mutate graph.Nodes.
            var bodyNodes = new List<FastNode>(Math.Max(0, closeIdx - openIdx - 1));
            for (int j = openIdx + 1; j < closeIdx; j++) bodyNodes.Add(graph.Nodes[j]);

            // Partition body nodes into loop-dependent (must be cloned per iteration
            // — their outputs vary across iterations) and loop-invariant (share a
            // single value across all iterations — clone once or keep in place).
            // Seed the loop-dependent tensor-key set with every OPEN output, then
            // expand by walking body nodes in positional (topological) order. A node
            // is loop-dependent if any of its inputs is a loop-dependent FastTensorKey.
            //
            // Scope propagation (Stage G): nested control-flow pairs inside the
            // body must be cloned as an atomic unit. If either side of an
            // OPEN/CLOSE pair is loop-dependent, both sides plus every body node
            // positionally between them must be cloned — otherwise a cloned CLOSE
            // would end up pointing at the un-cloned OPEN (wrong scope) or vice
            // versa. The CLOSE's effective inputs include the OPEN's inputs (via
            // the GraphOpenNodeKey linkage) for loop-dep purposes; we treat that
            // edge explicitly below so an IF whose only loop-varying signal is its
            // condition still triggers scope-wide cloning.
            //
            // Without this partitioning, naively cloning every body node N times
            // produces N duplicate copies of nodes whose value never changes —
            // notably fully-resolved `TRAINABLE_PARAM` nodes placed into the body
            // by `FastConvertTrainableParamIdRefToTrainableParam`. The resulting
            // graph has `N * K` trainable-param nodes instead of the expected `K`,
            // breaking downstream accounting in `GetConcreteModelParamInfos`.
            var loopDependentTensors = new HashSet<FastTensorKey>();
            foreach (var openOut in openNode.Outputs)
                if (openOut is FastTensorKey ot && !ot.IsEmpty) loopDependentTensors.Add(ot);

            var bodyByKey = new Dictionary<FastNodeKey, int>(bodyNodes.Count);
            for (int bi = 0; bi < bodyNodes.Count; bi++) bodyByKey[bodyNodes[bi].Key] = bi;

            var mustCloneBodyKeys = new HashSet<FastNodeKey>();
            foreach (var b in bodyNodes)
            {
                bool anyLoopDepIn = false;
                foreach (var kvp in b.FullInputs)
                    foreach (var ik in kvp.Value)
                        if (ik is FastTensorKey ikt && loopDependentTensors.Contains(ikt)) { anyLoopDepIn = true; break; }

                // For a CLOSE, also treat its matching OPEN's inputs as implicit
                // inputs — mirrors the fact that a close's output semantics depend
                // on its open's condition / iter-count / initializers even though
                // that edge isn't in FullInputs.
                if (!anyLoopDepIn && (b.OpCode == OpCodes.LOOP_CLOSE || b.OpCode == OpCodes.IF_CLOSE)
                    && b.GraphOpenNodeKey is FastNodeKey gok
                    && bodyByKey.TryGetValue(gok, out var openIdxInBody))
                {
                    var pairedOpen = bodyNodes[openIdxInBody];
                    foreach (var kvp in pairedOpen.FullInputs)
                        foreach (var ik in kvp.Value)
                            if (ik is FastTensorKey ikt && loopDependentTensors.Contains(ikt)) { anyLoopDepIn = true; break; }
                }

                if (!anyLoopDepIn) continue;
                mustCloneBodyKeys.Add(b.Key);
                foreach (var kvp in b.FullOutputs)
                    foreach (var ok in kvp.Value)
                        if (ok is FastTensorKey okt && !okt.IsEmpty) loopDependentTensors.Add(okt);
            }

            // Scope-pair propagation: for each marked OPEN, mark its matching CLOSE
            // plus every body node between them (they all live inside the same
            // atomic scope). For each marked CLOSE, mark its matching OPEN + same.
            // Iterate to fix-point in case a newly-marked interior node promotes
            // its own enclosing scope.
            bool changed = true;
            while (changed)
            {
                changed = false;
                for (int bi = 0; bi < bodyNodes.Count; bi++)
                {
                    var b = bodyNodes[bi];
                    if (!mustCloneBodyKeys.Contains(b.Key)) continue;

                    if (b.OpCode == OpCodes.LOOP_OPEN || b.OpCode == OpCodes.IF_OPEN)
                    {
                        for (int j = bi + 1; j < bodyNodes.Count; j++)
                        {
                            var cand = bodyNodes[j];
                            bool isMatchingClose =
                                (cand.OpCode == OpCodes.LOOP_CLOSE || cand.OpCode == OpCodes.IF_CLOSE)
                                && cand.GraphOpenNodeKey is FastNodeKey gk && gk == b.Key;
                            if (isMatchingClose)
                            {
                                for (int k2 = bi; k2 <= j; k2++)
                                    if (mustCloneBodyKeys.Add(bodyNodes[k2].Key)) changed = true;
                                break;
                            }
                        }
                    }
                    else if ((b.OpCode == OpCodes.LOOP_CLOSE || b.OpCode == OpCodes.IF_CLOSE)
                        && b.GraphOpenNodeKey is FastNodeKey gok2
                        && bodyByKey.TryGetValue(gok2, out var openBi))
                    {
                        for (int k2 = openBi; k2 <= bi; k2++)
                            if (mustCloneBodyKeys.Add(bodyNodes[k2].Key)) changed = true;
                    }
                }

                // Every body node now known to be cloned also becomes loop-dep
                // (its outputs vary across iterations after cloning), so
                // downstream consumers pick up loop-dep on the next pass.
                foreach (var b in bodyNodes)
                    if (mustCloneBodyKeys.Contains(b.Key))
                        foreach (var kvp in b.FullOutputs)
                            foreach (var ok in kvp.Value)
                                if (ok is FastTensorKey okt && !okt.IsEmpty && loopDependentTensors.Add(okt))
                                    changed = true;

                // Invariant pin: scope-pair propagation should leave no
                // not-yet-marked body node with a loop-dep input. Every key in
                // `loopDependentTensors` is produced by a node in
                // `mustCloneBodyKeys`; downstream consumers in the same scope
                // were marked by scope-pair propagation, and consumers outside
                // the scope go through the matching CLOSE (which was already
                // marked at first-pass time — that mark is what made
                // scope-pair propagation fire in the first place). The
                // "re-run the forward pass" branch this assertion replaces
                // had 0 hits across the full Quick+Standard suite. If this
                // assertion fires, the re-pass needs to be restored.
                Debug.Assert(
                    !AnyUnmarkedBodyNodeHasLoopDepInput(bodyNodes, mustCloneBodyKeys, loopDependentTensors),
                    "Stage G loop-dep analysis: scope-pair propagation left an unmarked "
                    + "body node with a loop-dep input.");
            }

            // N == 0: body never runs. Route CLOSE's loop-var outputs directly to the
            // initializers from OPEN's inputs[2..] and drop OPEN, CLOSE, and body.
            // (The gate already disqualified the N==0 case when nScan > 0, since
            // producing an empty scan tensor natively would need dtype inference.)
            if (iterCount == 0)
            {
                var remap0 = new Dictionary<FastTensorKey, FastTensorKey>();
                for (int k = 0; k < nLoop; k++)
                {
                    if (closeNode.Outputs[k] is not FastTensorKey closeOut || closeOut.IsEmpty) continue;
                    if (openInputs[2 + k] is not FastTensorKey init || init.IsEmpty) continue;
                    remap0[closeOut] = init;
                }
                ApplyRemapToGraph(graph, remap0);

                var removeKeys = new HashSet<FastNodeKey> { openNode.Key, closeNode.Key };
                foreach (var b in bodyNodes) removeKeys.Add(b.Key);
                graph.Nodes.RemoveAll(n => removeKeys.Contains(n.Key));
                return;
            }

            // Per-iteration iteration-index CONSTANT. Built once per iter, added to graph
            // at the end via newNodes.
            var newNodes = new List<FastNode>();
            var remap = new Dictionary<FastTensorKey, FastTensorKey>();

            // Track each iteration's output-key map: body_node_output → current-iter
            // materialised FastTensorKey. Every iteration clones the body with fresh keys
            // (the original body nodes are slated for removal anyway), so later
            // iterations can still read the original body inputs without seeing any
            // in-place mutation a "reuse iter 0 originals" shortcut would have caused.
            Dictionary<FastTensorKey, FastTensorKey>? prevIterBodyOutputMap = null;

            // Per-scan-var accumulator: iteration → cloned FastTensorKey of the body node
            // that feeds CLOSE.Inputs[1 + nLoop + k]. After all iterations run we
            // emit one Unsqueeze(axes=[0]) per iteration plus a Concat(axis=0) that
            // produces the final scan output.
            var scanIterationKeys = new List<List<FastTensorKey?>>(nScan);
            for (int k = 0; k < nScan; k++) scanIterationKeys.Add(new List<FastTensorKey?>(Math.Max((int)iterCount, 0)));

            // Cond-chain scaffolding (Stage E). If the body's per-iteration
            // continue-when output (CLOSE.Inputs[0]) is a dynamic value (not a
            // CONSTANT(true) / null), we need an AND+WHERE chain to "freeze" each
            // loop-var output at the iteration where the body-break first goes
            // false. OPEN.Inputs[1] is just the *initial* cond (usually a
            // CONSTANT(true) placed by the DSL) and by itself doesn't imply a
            // dynamic chain. The AND/WHERE emissions are skipped for static-true
            // cases so the previous no-cond path stays a straight-line unroll.
            //
            // We rebuild a one-shot producer lookup here because the caller's
            // (Process) producerByOutput lookup is out of scope and the set of
            // nodes is small enough that the cost is negligible next to the
            // clone work below.
            var producersLocal = new Dictionary<FastTensorKey, FastNode>(graph.Nodes.Count * 2);
            foreach (var producer in graph.Nodes)
            {
                foreach (var kvp in producer.FullOutputs)
                    foreach (var output in kvp.Value)
                        if (output is FastTensorKey tk && !tk.IsEmpty)
                            producersLocal[tk] = producer;
            }
            bool hasCondChain = !IsStaticTrue(closeNode.Inputs[0], producersLocal);

            // Per-loop-var "current final output" tensor key, tracked through the
            // WHERE chain. Seeded from the initializers so a fully-broken cond
            // (never enters the body) still returns the right value.
            var finalLoopVarKey = new FastTensorKey?[nLoop];
            for (int k = 0; k < nLoop; k++)
            {
                finalLoopVarKey[k] = openInputs.Count > 2 + k && openInputs[2 + k] is FastTensorKey initK && !initK.IsEmpty
                    ? initK
                    : (FastTensorKey?)null;
            }

            // Running AND-chain FastTensorKey (bool scalar). Seeded from the initial
            // cond on OPEN.Inputs[1] when present; if that slot is absent/null but
            // the body break is dynamic, we emit a fresh CONSTANT(true) as seed so
            // iter 0's AND has two operands. Only used when hasCondChain is true.
            FastTensorKey? allPrevCondKey = null;
            if (hasCondChain)
            {
                if (openInputs.Count >= 2 && openInputs[1] is FastTensorKey initCondKey && !initCondKey.IsEmpty)
                    allPrevCondKey = initCondKey;
                else
                {
                    var seedKey = FastNodeKey.New();
                    var seedAttrs = OnnxCSharpAttributes.FromCSharpVals(
                        new Dictionary<string, object?>
                        {
                            [OnnxOpAttributeNames.AttrValue] = Globals.TensorData(new long[0], true),
                        },
                        Definitions.NodeDefinitions[OpCodes.CONSTANT].AttributeDefs);
                    var seedNode = new FastNode
                    {
                        Key = seedKey,
                        OpCode = OpCodes.CONSTANT,
                        Attributes = seedAttrs,
                        FullOutputs = { [""] = new List<FastTensorKey?> { new FastTensorKey(seedKey, 0) } },
                    };
                    newNodes.Add(seedNode);
                    allPrevCondKey = new FastTensorKey(seedKey, 0);
                }
            }

            for (long iter = 0; iter < iterCount; iter++)
            {
                // Build iterInputRemap: body-visible TensorKeys whose source was an OPEN
                // output get remapped to the iteration-specific source. This covers
                //   openOutputs[0]   (iteration index)     → CONSTANT(iter) node
                //   openOutputs[1]   (vestigal true bool)  → CONSTANT(true) node
                //   openOutputs[2+k] (loop-var in-body)    → prev iter's body-close-input[1+k]
                //                                            (initializer on iter 0)
                var iterInputRemap = new Dictionary<FastTensorKey, FastTensorKey>();

                // iteration-index CONSTANT per iteration — a fresh int64 scalar node.
                var iterIdxNodeKey = FastNodeKey.New();
                newNodes.Add(FastNodeCreationHelpers.CreateConstantNode(
                    iterIdxNodeKey, new long[0], iter));
                var iterIdxKey = new FastTensorKey(iterIdxNodeKey, 0);

                // vestigal true bool CONSTANT per iteration. AttrValueInt can't represent
                // a bool, so go through the AttrValue tensor form.
                var trueNodeKey = FastNodeKey.New();
                var trueAttrs = OnnxCSharpAttributes.FromCSharpVals(
                    new Dictionary<string, object?>
                    {
                        [OnnxOpAttributeNames.AttrValue] = Globals.TensorData(new long[0], true),
                    },
                    Definitions.NodeDefinitions[OpCodes.CONSTANT].AttributeDefs);
                var trueNode = new FastNode
                {
                    Key = trueNodeKey,
                    OpCode = OpCodes.CONSTANT,
                    Attributes = trueAttrs,
                    FullOutputs = { [""] = new List<FastTensorKey?> { new FastTensorKey(trueNodeKey, 0) } },
                };
                newNodes.Add(trueNode);
                var trueKey = new FastTensorKey(trueNodeKey, 0);

                if (openOutputs.Count > 0 && openOutputs[0] is FastTensorKey iterOut && !iterOut.IsEmpty)
                    iterInputRemap[iterOut] = iterIdxKey;
                if (openOutputs.Count > 1 && openOutputs[1] is FastTensorKey vtOut && !vtOut.IsEmpty)
                    iterInputRemap[vtOut] = trueKey;

                for (int k = 0; k < nLoop; k++)
                {
                    if (openOutputs.Count <= 2 + k) break;
                    if (openOutputs[2 + k] is not FastTensorKey bodyIn || bodyIn.IsEmpty) continue;

                    // Iteration i's in-body loop-var value is whatever we currently
                    // track as "final output so far" — seeded from the initializer
                    // and (without cond chain) advanced through each iteration's
                    // body-close-input, or (with cond chain) through the emitted
                    // WHERE node. Either way `finalLoopVarKey[k]` is the single
                    // source of truth here.
                    var src = finalLoopVarKey[k];
                    if (src is not null) iterInputRemap[bodyIn] = src.Value;
                }

                // Clone every body node with a fresh FastNodeKey. curTensorMap subsumes
                // iterInputRemap (OPEN-produced keys) and extends with body-internal
                // keys as they are cloned; curBodyOutputMap records original → new
                // output key for this iteration's clone of each body node.
                var curTensorMap = new Dictionary<FastTensorKey, FastTensorKey>(iterInputRemap);
                var curBodyOutputMap = new Dictionary<FastTensorKey, FastTensorKey>();

                // Clones are produced in two passes per iteration:
                //   Pass 1 — basic clone with remapped FullInputs and fresh FullOutputs.
                //   Pass 2 — Stage F: for any cloned node carrying an IdentifierTemplate
                //     or model-id placeholder, walk the newly-remapped iteration-index
                //     input chain, extract the concrete iteration indices (our loop's
                //     CONSTANT(iter), outer loops' LOOP_OPEN output → -1), and rewrite
                //     Attributes.ShrkAttrLocalModelId / ShrkAttrRelativeModelId plus
                //     IdentifierTemplate to match.
                // Two passes because the iter-index chain walk needs the freshly-cloned
                // CONCAT / UNSQUEEZE nodes for this iteration to be resolvable, which
                // means the clone's inputs must have been remapped first.
                //
                // Loop-invariant body nodes (no transitive dep on an OPEN output) are
                // NOT cloned — they are shared across every iteration via identity
                // mappings in curTensorMap / curBodyOutputMap, so their consumers in
                // this iteration's clones point at the original (shared) FastTensorKey.
                //
                // Nested OPEN/CLOSE pairs inside the body are cloned like any other
                // node; a cloned CLOSE's GraphOpenNodeKey is rewired to the matching
                // cloned OPEN's fresh key via `nodeKeyRemap` below.
                var clonedThisIter = new List<(FastNode original, FastNode cloned)>(bodyNodes.Count);
                var nodeKeyRemap = new Dictionary<FastNodeKey, FastNodeKey>();

                foreach (var b in bodyNodes)
                {
                    if (!mustCloneBodyKeys.Contains(b.Key))
                    {
                        // Loop-invariant — leave in place, share via identity.
                        foreach (var kvp in b.FullOutputs)
                            foreach (var ok in kvp.Value)
                                if (ok is FastTensorKey okt && !okt.IsEmpty)
                                {
                                    curTensorMap.TryAdd(okt, okt);
                                    curBodyOutputMap.TryAdd(okt, okt);
                                }
                        continue;
                    }

                    var newKey = FastNodeKey.New();
                    nodeKeyRemap[b.Key] = newKey;

                    // Rewire GraphOpenNodeKey for nested close nodes: if the matching
                    // open was also cloned this iteration, point the clone at the
                    // cloned open; otherwise keep the original reference. Body
                    // topological order guarantees the OPEN clone was processed before
                    // its CLOSE (so nodeKeyRemap[gok] is populated by the time we look
                    // it up).
                    FastNodeKey? clonedGraphOpenKey = b.GraphOpenNodeKey;
                    if (b.GraphOpenNodeKey is FastNodeKey gok && !gok.IsEmpty
                        && nodeKeyRemap.TryGetValue(gok, out var clonedOpen))
                        clonedGraphOpenKey = clonedOpen;

                    var cloned = new FastNode
                    {
                        Key = newKey,
                        OpCode = b.OpCode,
                        Attributes = b.Attributes,
                        FriendlyName = b.FriendlyName,
                        StackTrace = b.StackTrace,
                        GraphOpenNodeKey = clonedGraphOpenKey,
                        IdentifierTemplate = b.IdentifierTemplate,
                        TargetFunction = b.TargetFunction,
                    };

                    foreach (var kvp in b.FullInputs)
                    {
                        var list = new List<FastTensorKey?>(kvp.Value.Count);
                        foreach (var ik in kvp.Value)
                        {
                            if (ik is null) { list.Add(null); continue; }
                            if (curTensorMap.TryGetValue(ik.Value, out var rep)) list.Add(rep);
                            else list.Add(ik.Value);
                        }
                        cloned.FullInputs[kvp.Key] = list;
                    }

                    foreach (var kvp in b.FullOutputs)
                    {
                        var list = new List<FastTensorKey?>(kvp.Value.Count);
                        foreach (var ok in kvp.Value)
                        {
                            if (ok is null) { list.Add(null); continue; }
                            var newTk = new FastTensorKey(newKey, ok.Value.OutputIndex);
                            list.Add(newTk);
                            curTensorMap[ok.Value] = newTk;
                            curBodyOutputMap[ok.Value] = newTk;
                        }
                        cloned.FullOutputs[kvp.Key] = list;
                    }

                    newNodes.Add(cloned);
                    clonedThisIter.Add((b, cloned));
                }

                // Precondition assert: by the time FastSimplify invokes Process,
                // FastConvertToIdRefTrainableParams + FastUnpackModelStruct +
                // FastConvertTrainableParamIdRefToTrainableParam have all run, so the
                // four op-codes that used to drive a per-iteration identifier-template
                // rewrite here are gone. See the Process docstring for the full chain.
                foreach (var (_, cloned) in clonedThisIter)
                {
                    Debug.Assert(
                        cloned.OpCode != InternalOpCodes.TRAINABLE_PARAM_REF
                        && cloned.OpCode != InternalOpCodes.TRAINABLE_PARAM_ID_REF
                        && cloned.OpCode != InternalOpCodes.TRAINABLE_PARAM_MODEL_REF
                        && cloned.OpCode != InternalOpCodes.MODULE_SET_HYPERPARAMS,
                        $"UnrollOne: cloned body node has OpCode {cloned.OpCode}, but the "
                        + "full ToConcreteArchitecture pipeline rewrites all four "
                        + "Stage-F op-codes to TRAINABLE_PARAM (or removes them) before "
                        + "FastSimplify runs the unroll.");
                }

                prevIterBodyOutputMap = curBodyOutputMap;

                // Compute each loop-var's "candidate" FastTensorKey for this iteration —
                // the body-close-input remapped through this iter's clone. If no
                // cond chain, the candidate becomes the new finalLoopVarKey
                // directly. With cond chain, we emit a WHERE(allPrev AND bodyCond,
                // candidate, carriedFinal) and thread the result forward.
                var iterCandidateLoopVarKey = new FastTensorKey?[nLoop];
                for (int k = 0; k < nLoop; k++)
                {
                    var closeBodyIn = closeNode.Inputs[1 + k];
                    if (closeBodyIn is not FastTensorKey ckey || ckey.IsEmpty) { iterCandidateLoopVarKey[k] = null; continue; }
                    iterCandidateLoopVarKey[k] = curBodyOutputMap.TryGetValue(ckey, out var mappedK) ? mappedK : ckey;
                }

                if (hasCondChain)
                {
                    // allPrev_i gates iter_i's candidate — it tracks "did we enter
                    // iter i at all?". Per-loop-var WHERE(allPrev_i, candidate_i,
                    // carriedFinal): if we did, use this iteration's value;
                    // otherwise keep the carried prior final.
                    var selectorKey = allPrevCondKey;
                    Debug.Assert(selectorKey is not null && !selectorKey.Value.IsEmpty,
                        "FastFoldConstantIterationLoops.UnrollOne: hasCondChain is true but the "
                        + "running selector is null/empty. The seed block above always gives "
                        + "allPrevCondKey a non-empty value (initial OPEN cond or freshly emitted "
                        + "CONSTANT(true)), and the AND-chain update preserves that invariant.");
                    for (int k = 0; k < nLoop; k++)
                    {
                        var cand = iterCandidateLoopVarKey[k];
                        var carried = finalLoopVarKey[k];
                        if (cand is null || carried is null) continue;

                        var whereNodeKey = FastNodeKey.New();
                        var whereOutKey = new FastTensorKey(whereNodeKey, 0);
                        newNodes.Add(FastNodeCreationHelpers.CreateFastNode(
                            whereNodeKey, OpCodes.WHERE, new Dictionary<string, object?>(),
                            new FastTensorKey?[] { selectorKey.Value, cand.Value, carried.Value }));
                        finalLoopVarKey[k] = whereOutKey;
                    }

                    // Update allPrev for the NEXT iteration: bodyCond_i AND allPrev_i.
                    // This body cond determines whether iter_{i+1} would run.
                    FastTensorKey? iterBodyCondKey = null;
                    var breakIn = closeNode.Inputs[0];
                    if (breakIn is FastTensorKey bk && !bk.IsEmpty)
                        iterBodyCondKey = curBodyOutputMap.TryGetValue(bk, out var mappedBk) ? mappedBk : bk;

                    if (iterBodyCondKey is not null && allPrevCondKey is not null
                        && !iterBodyCondKey.Value.IsEmpty && !allPrevCondKey.Value.IsEmpty)
                    {
                        var andNodeKey = FastNodeKey.New();
                        var andOutKey = new FastTensorKey(andNodeKey, 0);
                        newNodes.Add(FastNodeCreationHelpers.CreateFastNode(
                            andNodeKey, OpCodes.AND, new Dictionary<string, object?>(),
                            new FastTensorKey?[] { allPrevCondKey.Value, iterBodyCondKey.Value }));
                        allPrevCondKey = andOutKey;
                    }
                }
                else
                {
                    // No cond chain: the candidate is the new final directly.
                    for (int k = 0; k < nLoop; k++)
                    {
                        if (iterCandidateLoopVarKey[k] is not null)
                            finalLoopVarKey[k] = iterCandidateLoopVarKey[k];
                    }
                }

                // Record per-scan-var the cloned FastTensorKey this iteration produced for
                // CLOSE.Inputs[1 + nLoop + k]. LoopAPI's third pass binds every scan
                // output to a body-produced value, so each scan input is a non-empty
                // FastTensorKey that resolves through curBodyOutputMap.
                for (int k = 0; k < nScan; k++)
                {
                    var scanInKey = closeNode.Inputs[1 + nLoop + k];
                    Debug.Assert(scanInKey is FastTensorKey sk0 && !sk0.IsEmpty,
                        "FastFoldConstantIterationLoops.UnrollOne: scan input is null/empty. "
                        + "LoopAPI binds every scan output to a body value by construction.");
                    var sk = (FastTensorKey)scanInKey!;
                    var hasMapped = curBodyOutputMap.TryGetValue(sk, out var mapped);
                    Debug.Assert(hasMapped,
                        "FastFoldConstantIterationLoops.UnrollOne: scan input is not body-produced. "
                        + "LoopAPI's third pass maps every scan output to a body node's output.");
                    scanIterationKeys[k].Add(mapped);
                }
            }

            // Original body nodes are now dead — consumers have been remapped to the
            // cloned iterations, and the CLOSE-output remap below re-routes the loop
            // outputs to the final iteration's clones.

            // After N iterations, CLOSE's loop-var outputs route to `finalLoopVarKey`
            // — the WHERE-chain result when a dynamic cond is present, or simply
            // the last iteration's body value when there isn't one. The single
            // source of truth unifies both paths.
            for (int k = 0; k < nLoop; k++)
            {
                if (closeNode.Outputs[k] is not FastTensorKey closeOut || closeOut.IsEmpty) continue;
                if (finalLoopVarKey[k] is FastTensorKey finalK && !finalK.IsEmpty)
                    remap[closeOut] = finalK;
            }

            // Scan outputs: each iteration's scan body value becomes a rank+1 row of
            // the final output. Emit Unsqueeze(axes=[0]) per iteration (lifting each
            // value to rank+1 with leading dim 1) and a Concat(axis=0) over them
            // (stacking into leading dim N). The resulting Concat output is the
            // CLOSE scan output's producer.
            for (int k = 0; k < nScan; k++)
            {
                if (closeNode.Outputs[nLoop + k] is not FastTensorKey scanOut || scanOut.IsEmpty) continue;
                var iterKeys = scanIterationKeys[k];
                if (iterKeys.Count == 0 || iterKeys.Any(x => x is null || x.Value.IsEmpty)) continue;

                // Shared axes CONSTANT for every Unsqueeze in this scan chain — a
                // vector<int64>([0]) telling UNSQUEEZE to prepend a new axis 0.
                var axesNodeKey = FastNodeKey.New();
                var axesKey = new FastTensorKey(axesNodeKey, 0);
                newNodes.Add(FastNodeCreationHelpers.CreateConstantNode(
                    axesNodeKey, new long[] { 1 }, 0L));

                var unsqueezedKeys = new List<FastTensorKey?>(iterKeys.Count);
                foreach (var iterKey in iterKeys)
                {
                    var unsqNodeKey = FastNodeKey.New();
                    var unsqKey = new FastTensorKey(unsqNodeKey, 0);
                    newNodes.Add(FastNodeCreationHelpers.CreateFastNode(
                        unsqNodeKey, OpCodes.UNSQUEEZE,
                        new Dictionary<string, object?>(),
                        new FastTensorKey?[] { iterKey!.Value, axesKey }));
                    unsqueezedKeys.Add(unsqKey);
                }

                var concatNodeKey = FastNodeKey.New();
                var concatOutKey = new FastTensorKey(concatNodeKey, 0);
                newNodes.Add(FastNodeCreationHelpers.CreateFastNode(
                    concatNodeKey, OpCodes.CONCAT,
                    new Dictionary<string, object?> { [OnnxOpAttributeNames.AttrAxis] = 0L },
                    unsqueezedKeys.ToArray()));

                remap[scanOut] = concatOutKey;
            }

            // Splice the cloned iterations in where the loop body used to live, so
            // the result is already topologically ordered and properly nested:
            //   [ pre-loop nodes ]
            //   [ surviving (loop-invariant) body nodes, in their original order ]
            //   [ cloned iterations, in iteration order — each block internally nested ]
            //   [ post-loop nodes ]
            // Pre-loop nodes only feed the loop and surviving body nodes (never the
            // other way around); post-loop nodes consume CLOSE outputs which we
            // remap onto keys produced inside `newNodes`. So this layout satisfies
            // every data edge without any further reordering.
            ApplyRemapToGraph(graph, remap);

            var mustClone = new HashSet<FastNodeKey>();
            foreach (var b in bodyNodes)
                if (mustCloneBodyKeys.Contains(b.Key)) mustClone.Add(b.Key);

            var spliced = new List<FastNode>(graph.Nodes.Count - 2 - mustClone.Count + newNodes.Count);
            for (int i = 0; i < openIdx; i++) spliced.Add(graph.Nodes[i]);
            for (int j = openIdx + 1; j < closeIdx; j++)
            {
                var b = graph.Nodes[j];
                if (!mustClone.Contains(b.Key)) spliced.Add(b);
            }
            spliced.AddRange(newNodes);
            for (int i = closeIdx + 1; i < graph.Nodes.Count; i++) spliced.Add(graph.Nodes[i]);
            graph.Nodes = spliced;
        }

        private static void ApplyRemapToGraph(FastComputationGraph graph, Dictionary<FastTensorKey, FastTensorKey> remap)
        {
            if (remap.Count == 0) return;

            // Resolve transitive chains first.
            foreach (var key in remap.Keys.ToList())
            {
                var target = remap[key];
                while (remap.TryGetValue(target, out var next)) target = next;
                remap[key] = target;
            }

            foreach (var node in graph.Nodes)
            {
                foreach (var kvp in node.FullInputs)
                {
                    var list = kvp.Value;
                    for (int j = 0; j < list.Count; j++)
                    {
                        if (list[j] is FastTensorKey k && remap.TryGetValue(k, out var rep))
                            list[j] = rep;
                    }
                }
            }
            for (int i = 0; i < graph.Outputs.Count; i++)
                if (remap.TryGetValue(graph.Outputs[i], out var rep))
                    graph.Outputs[i] = rep;
        }

        private static long? ReadConstantLong(FastNode constNode)
        {
            var attrs = constNode.Attributes;
            if (!attrs.IsDefaultValue(OnnxOpAttributeNames.AttrValueInt))
                return attrs.GetLongVal(OnnxOpAttributeNames.AttrValueInt);

            var tensorVal = attrs.GetTensorVal(OnnxOpAttributeNames.AttrValue);
            if (tensorVal is null) return null;
            if (tensorVal.DType != DType.Int64) return null;
            if (tensorVal.Shape.Dims.Length != 0) return null;
            return tensorVal.As<int64>().AccessMemory()[0];
        }
    }

    /// <summary>
    /// Native FastComputationGraph constant folding. Uses
    /// <see cref="QuickExecutionEngine"/> in-process to evaluate the constant subgraph;
    /// constants the engine cannot resolve are simply left unfolded (so the outer
    /// Simplify loop converges on the next iteration once nothing changes).
    /// </summary>
    internal static class FastFoldConstants
    {
        /// <summary>
        /// Runs one folding pass over <paramref name="graph"/>. Returns true if any
        /// constant was folded into a CONSTANT node.
        /// </summary>
        public static bool Process(FastComputationGraph graph)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));

            // Identify which tensor keys are "constant" — i.e., produced by a chain of ops
            // where every input is itself constant. Excludes model inputs, model param
            // data, trainable params, and open/close structural nodes.
            var constantTensors = new HashSet<FastTensorKey>();
            var unfoldableConstantOutputs = new HashSet<FastTensorKey>();
            var requiredConstantTensors = new HashSet<FastTensorKey>();
            var constantProducingNodes = new List<FastNode>();

            foreach (var node in graph.Nodes)
            {
                var op = node.OpCode;

                if (IsModelInputOpCode(op) || op == InternalOpCodes.MODEL_PARAM_DATA)
                    continue;

                if (op == InternalOpCodes.TRAINABLE_PARAM
                    || op.EndsWith("#OPEN") || op.EndsWith("#CLOSE"))
                {
                    // For IF_OPEN, the condition input drives FoldConstantConditionBranches;
                    // mark it as a required constant if foldable.
                    if (op == OpCodes.IF_OPEN)
                    {
                        var inputs = node.Inputs;
                        if (inputs.Count > 0 && inputs[0] is FastTensorKey cond
                            && constantTensors.Contains(cond)
                            && !unfoldableConstantOutputs.Contains(cond))
                            requiredConstantTensors.Add(cond);
                    }
                    continue;
                }

                bool allConst = true;
                foreach (var inp in node.Inputs)
                {
                    if (inp is null) continue; // omitted optional
                    if (!constantTensors.Contains(inp.Value))
                    {
                        allConst = false;
                        break;
                    }
                }

                if (allConst)
                {
                    constantProducingNodes.Add(node);
                    foreach (var output in node.Outputs)
                    {
                        if (output is null) continue;
                        constantTensors.Add(output.Value);

                        // Already a CONSTANT node — re-folding accomplishes nothing.
                        if (op == OpCodes.CONSTANT)
                            unfoldableConstantOutputs.Add(output.Value);
                        else if (IsSequenceProducingOpCode(op))
                            unfoldableConstantOutputs.Add(output.Value);
                    }
                }
                else
                {
                    foreach (var inp in node.Inputs)
                    {
                        if (inp is null) continue;
                        var k = inp.Value;
                        if (constantTensors.Contains(k) && !unfoldableConstantOutputs.Contains(k))
                            requiredConstantTensors.Add(k);
                    }
                }
            }

            if (requiredConstantTensors.Count == 0) return false;

            // Build a minimal subgraph containing only constant-producing nodes plus any
            // upstream MODEL_PARAM_DATA they may depend on (none, by definition above —
            // MODEL_PARAM_DATA is excluded from constants — but include for safety).
            // Running QEE on this subgraph avoids touching the thousands of non-constant
            // ops (Conv, MatMul, …) in the full graph.
            var subgraph = new FastComputationGraph
            {
                Nodes = constantProducingNodes,
            };
            System.Diagnostics.Debug.Assert(subgraph.IsLinearOrderValid(), "subgraph.IsLinearOrderValid()");

            var store = new QuickExecutionEngine().Run(subgraph);

            var foldedTensorData = new Dictionary<FastTensorKey, TensorData>();
            foreach (var key in requiredConstantTensors)
            {
                if (store.TryGetValue(key, out var rt)
                    && rt is RuntimeTensor plain
                    && TensorDataConverter.ToTensorData(plain) is { } td)
                {
                    foldedTensorData[key] = td;
                }
            }

            if (foldedTensorData.Count == 0) return false;

            // Materialize each folded constant as a fresh CONSTANT FastNode and remap
            // consumers from the old FastTensorKey to the new one.
            var oldToNewKey = new Dictionary<FastTensorKey, FastTensorKey>(foldedTensorData.Count);
            var nodesToAdd = new List<FastNode>(foldedTensorData.Count);
            foreach (var kvp in foldedTensorData)
            {
                var oldKey = kvp.Key;
                var td = kvp.Value;
                var newNodeKey = FastNodeKey.New();
                var newNode = CreateConstantTensorDataNode(newNodeKey, td);
                var newKey = newNode.Outputs[0]!.Value;

                nodesToAdd.Add(newNode);
                oldToNewKey[oldKey] = newKey;
            }

            // Prepend the new CONSTANT nodes. They have no inputs, so the front of
            // the node list is a topologically valid position — every existing
            // consumer (which used to read from the original constant subgraph)
            // now reads from a freshly-prepended CONSTANT that comes before it.
            // Constants carry no scope of their own, so this also keeps OPEN/CLOSE
            // nesting intact.
            var combined = new List<FastNode>(graph.Nodes.Count + nodesToAdd.Count);
            combined.AddRange(nodesToAdd);
            combined.AddRange(graph.Nodes);
            graph.Nodes = combined;

            foreach (var node in graph.Nodes)
            {
                foreach (var kvp in node.FullInputs)
                {
                    var list = kvp.Value;
                    for (int j = 0; j < list.Count; j++)
                    {
                        if (list[j] is FastTensorKey tk && oldToNewKey.TryGetValue(tk, out var rep))
                            list[j] = rep;
                    }
                }
            }

            for (int i = 0; i < graph.Outputs.Count; i++)
            {
                if (oldToNewKey.TryGetValue(graph.Outputs[i], out var rep))
                    graph.Outputs[i] = rep;
            }

            FastProcessorHelper.RemoveUnreachableNodes(graph);
            System.Diagnostics.Debug.Assert(graph.IsLinearOrderValid(), "graph.IsLinearOrderValid()");
            return true;
        }

        private static bool IsModelInputOpCode(string opCode) =>
            opCode == InternalOpCodes.MODEL_TENSOR_INPUT ||
            opCode == InternalOpCodes.MODEL_OPTIONAL_INPUT ||
            opCode == InternalOpCodes.MODEL_SEQUENCE_INPUT ||
            opCode == InternalOpCodes.MODEL_TENSORSTRUCT_INPUT ||
            opCode == InternalOpCodes.GENERIC_TYPE_INPUT;

        // Ops whose output is a DataStructure.Sequence. MODULE_INVOKE/FUNCTION_INVOKE and the
        // model-struct ops (MODEL_HYPERPARAM, MODULE_SET_HYPERPARAMS, GET_MODEL_ID) can also
        // carry sequence outputs but are guaranteed to be gone before FoldConstants runs (the
        // ComputationGraph → fast pipeline asserts them removed upstream). MODEL_SEQUENCE_INPUT
        // is covered separately by IsModelInputOpCode and LOOP_CLOSE scan outputs are covered
        // by the "#CLOSE" suffix skip above.
        private static bool IsSequenceProducingOpCode(string opCode) =>
            opCode == OpCodes.SEQUENCE_EMPTY ||
            opCode == OpCodes.SEQUENCE_CONSTRUCT ||
            opCode == OpCodes.SEQUENCE_ERASE ||
            opCode == OpCodes.SEQUENCE_INSERT ||
            opCode == InternalOpCodes.SEQUENCE_CONCAT ||
            opCode == InternalOpCodes.SEQUENCE_SLICE;

        private static FastNode CreateConstantTensorDataNode(FastNodeKey nodeKey, TensorData td)
        {
            var tensorKey = new FastTensorKey(nodeKey, 0);
            var attrDefs = Definitions.NodeDefinitions[OpCodes.CONSTANT].AttributeDefs;
            var attrs = OnnxCSharpAttributes.FromCSharpVals(
                new Dictionary<string, object?> { [OnnxOpAttributeNames.AttrValue] = td }, attrDefs);

            return new FastNode
            {
                Key = nodeKey,
                OpCode = OpCodes.CONSTANT,
                Attributes = attrs,
                FullOutputs = { [""] = new List<FastTensorKey?> { tensorKey } },
            };
        }
    }

    /// <summary>
    /// Shared helper methods for creating FastNodes (CONSTANT, UNSQUEEZE, CONCAT, Identity)
    /// that mirror the node structures produced by the normal ComputationGraph API.
    /// Used by <see cref="FastConvertToIdRefTrainableParams"/> and <see cref="FastUnpackModelStruct"/>.
    /// </summary>
    internal static class FastNodeCreationHelpers
    {
        /// <summary>
        /// Navigates the iteration-indices input backwards through CONCAT→IDENTITY→UNSQUEEZE→LOOP_OPEN
        /// and returns the TensorKeys of the scalar iteration index outputs (one per loop nesting level).
        /// </summary>
        public static ImmutableArray<FastTensorKey> GetIterationIndexScalars(
            FastTensorKey? iterationIndicesKey, Dictionary<FastNodeKey, FastNode> nodeByKey)
        {
            if (iterationIndicesKey is null || iterationIndicesKey.Value.IsEmpty)
                return [];

            var producingNode = nodeByKey[iterationIndicesKey.Value.FastNodeKey];

            if (producingNode.OpCode == OpCodes.CONSTANT)
                return [];

            Debug.Assert(producingNode.OpCode == OpCodes.CONCAT);

            var concatInputs = producingNode.Inputs;
            var builder = ImmutableArray.CreateBuilder<FastTensorKey>(concatInputs.Count);

            foreach (var concatInput in concatInputs)
            {
                Debug.Assert(concatInput is not null);
                var key = concatInput.Value;
                var node = nodeByKey[key.FastNodeKey];

                if (node.OpCode == OpCodes.IDENTITY)
                {
                    key = node.Inputs[0]!.Value;
                    node = nodeByKey[key.FastNodeKey];
                }

                if (node.OpCode == OpCodes.UNSQUEEZE)
                {
                    key = node.Inputs[0]!.Value;
                    node = nodeByKey[key.FastNodeKey];
                }

                Debug.Assert(node.OpCode == OpCodes.LOOP_OPEN || node.OpCode == OpCodes.CONSTANT);
                builder.Add(key);
            }

            return builder.ToImmutable();
        }

        /// <summary>
        /// Creates CONSTANT / UNSQUEEZE / Identity / CONCAT FastNodes that produce a model-ID
        /// <c>Vector&lt;int64&gt;</c>, optionally prepended with <paramref name="baseModelIdKey"/>.
        /// Returns the FastTensorKey of the final Identity output.
        /// </summary>
        public static FastTensorKey BuildSpecificModelIdFastNodes(
            ModelId genericModelId,
            ImmutableArray<FastTensorKey> iterationIndexKeys,
            FastTensorKey? baseModelIdKey,
            List<FastNode> newNodes)
        {
            var unsqueezedKeys = new List<FastTensorKey>();

            if (baseModelIdKey is not null)
                unsqueezedKeys.Add(baseModelIdKey.Value);

            int iterIdx = 0;
            foreach (var val in genericModelId.Vals)
            {
                FastTensorKey scalarKey;
                if (val == -1)
                {
                    scalarKey = iterationIndexKeys[iterIdx++];
                }
                else
                {
                    var constNodeKey = FastNodeKey.New();
                    scalarKey = new FastTensorKey(constNodeKey, 0);
                    newNodes.Add(CreateConstantNode(constNodeKey, new long[0], (long)val));
                }

                // Fresh axes CONSTANT per UNSQUEEZE (matches Tensor.Unsqueeze() which
                // creates a new Vector(-1L) constant each time).
                var axesNodeKey = FastNodeKey.New();
                var axesKey = new FastTensorKey(axesNodeKey, 0);
                newNodes.Add(CreateConstantNode(axesNodeKey, new long[] { 1 }, -1L));

                var unsqueezeNodeKey = FastNodeKey.New();
                var unsqueezeTensorKey = new FastTensorKey(unsqueezeNodeKey, 0);
                newNodes.Add(CreateFastNode(unsqueezeNodeKey, OpCodes.UNSQUEEZE,
                    new Dictionary<string, object?>(),
                    new FastTensorKey?[] { scalarKey, axesKey }));

                // Identity node matching .Vec() call in Scalar.Unsqueeze().
                var vecNodeKey = FastNodeKey.New();
                var vecTensorKey = new FastTensorKey(vecNodeKey, 0);
                newNodes.Add(CreateFastNode(vecNodeKey, OpCodes.IDENTITY,
                    new Dictionary<string, object?>(),
                    new FastTensorKey?[] { unsqueezeTensorKey }));
                unsqueezedKeys.Add(vecTensorKey);
            }

            // CONCAT all unsqueezed vectors (even for single element, matching normal path).
            var concatNodeKey = FastNodeKey.New();
            var concatTensorKey = new FastTensorKey(concatNodeKey, 0);
            newNodes.Add(CreateFastNode(concatNodeKey, OpCodes.CONCAT,
                new Dictionary<string, object?> { [OnnxOpAttributeNames.AttrAxis] = 0L },
                unsqueezedKeys.Select(k => (FastTensorKey?)k).ToArray()));

            return concatTensorKey;
        }

        /// <summary>Creates a CONSTANT FastNode producing a scalar or vector int64 tensor.</summary>
        public static FastNode CreateConstantNode(FastNodeKey nodeKey, long[] dims, long value)
        {
            var tensorKey = new FastTensorKey(nodeKey, 0);
            var td = Globals.TensorData(dims, value);
            var attrDefs = Definitions.NodeDefinitions[OpCodes.CONSTANT].AttributeDefs;
            var attrs = OnnxCSharpAttributes.FromCSharpVals(
                new Dictionary<string, object?> { [OnnxOpAttributeNames.AttrValue] = td }, attrDefs);

            return new FastNode
            {
                Key = nodeKey,
                OpCode = OpCodes.CONSTANT,
                Attributes = attrs,
                FullOutputs = { [""] = new List<FastTensorKey?> { tensorKey } },
            };
        }

        /// <summary>Creates a generic single-output FastNode.</summary>
        public static FastNode CreateFastNode(
            FastNodeKey nodeKey, string opCode,
            Dictionary<string, object?> attrVals, FastTensorKey?[] inputs)
        {
            var tensorKey = new FastTensorKey(nodeKey, 0);
            var attrDefs = Definitions.NodeDefinitions[opCode].AttributeDefs;
            var attrs = OnnxCSharpAttributes.FromCSharpVals(attrVals, attrDefs);

            return new FastNode
            {
                Key = nodeKey,
                OpCode = opCode,
                Attributes = attrs,
                FullInputs = { [""] = new List<FastTensorKey?>(inputs) },
                FullOutputs = { [""] = new List<FastTensorKey?> { tensorKey } },
            };
        }
    }

    /// <summary>
    /// Thrown by a Fast pipeline processor when the graph relies on behaviour the native Fast
    /// implementation cannot (yet) handle. The CG fallback was removed; this exception now
    /// propagates to the caller as a hard failure indicating an unsupported graph shape.
    /// </summary>
    internal sealed class FastPipelineUnsupportedException : Exception
    {
        public FastPipelineUnsupportedException(string message) : base(message) { }
    }
}
