using Shorokoo.Graph;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Nodes.Processors.AutoGrad;
using Shorokoo.Core.Nodes.Processors.Fast;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using Shorokoo.Core.Utils;
using Shorokoo.Onnx;
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;

namespace Shorokoo.Core.Nodes.Processors.Training
{
    /// <summary>
    /// Fast-native rewrite step used by <see cref="Shorokoo.Core.Training.TrainingGraphBuilder.PrepareForTrainingAsFast(InternalComputationGraph, InternalComputationGraph)"/>.
    /// Mutates the supplied <see cref="InternalComputationGraph"/> in place: every model input producer node is replaced
    /// (via key remap) by the matching <c>TENSOR_STRUCT_GETFIELD</c> output of a
    /// <c>model_inputs</c> struct input; every non-trainable param node is replaced by the
    /// matching <c>TENSOR_STRUCT_GETFIELD</c> output of a <c>model_state</c> struct input;
    /// every <c>STATE_UPDATE_LINK</c> is unwrapped to its updated-state input (which is
    /// also collected as a new graph output); every <c>WITH_STATE_DEPS</c> is unwrapped to
    /// its main input. The replaced nodes themselves are dropped from <c>graph.Nodes</c>,
    /// the original input keys are dropped from <c>graph.Inputs</c>, and the new struct
    /// inputs are appended.
    /// </summary>
    internal static class FastRebuildModelInputsForTrainingProcessor
    {
        public sealed class ProcessResult
        {
            public FastTensorKey RebuiltModelOutput { get; }
            public FastTensorKey[] RebuiltParamFieldKeys { get; }
            public FastTensorKey RebuiltTrainableParamStructInput { get; }
            public ImmutableArray<FastTensorKey> StateUpdateOutputs { get; }

            internal ProcessResult(
                FastTensorKey rebuiltModelOutput,
                FastTensorKey[] rebuiltParamFieldKeys,
                FastTensorKey rebuiltTrainableParamStructInput,
                ImmutableArray<FastTensorKey> stateUpdateOutputs)
            {
                RebuiltModelOutput = rebuiltModelOutput;
                RebuiltParamFieldKeys = rebuiltParamFieldKeys;
                RebuiltTrainableParamStructInput = rebuiltTrainableParamStructInput;
                StateUpdateOutputs = stateUpdateOutputs;
            }
        }

        public static ProcessResult Process(
            InternalComputationGraph graph,
            FastNodeKey[] originalModelInputNodeKeys,
            FastTensorKey[] modelInputFieldKeys,
            FastTensorKey modelInputStructInputKey,
            FastNodeKey[] stateParamNodeKeys,
            FastTensorKey[] stateFieldKeys,
            FastTensorKey? stateStructInputKey,
            FastTensorKey trainableParamStructInputKey,
            FastTensorKey[] paramFieldKeys,
            FastTensorKey modelOutputKey)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));
            if (originalModelInputNodeKeys.Length != modelInputFieldKeys.Length)
                throw new ArgumentException("originalModelInputNodeKeys and modelInputFieldKeys must have the same length.");
            if (stateParamNodeKeys.Length != stateFieldKeys.Length)
                throw new ArgumentException("stateParamNodeKeys and stateFieldKeys must have the same length.");

            var modelInputReplacementByNodeKey = new Dictionary<FastNodeKey, FastTensorKey>(originalModelInputNodeKeys.Length);
            for (int i = 0; i < originalModelInputNodeKeys.Length; i++)
                modelInputReplacementByNodeKey[originalModelInputNodeKeys[i]] = modelInputFieldKeys[i];

            var stateReplacementByNodeKey = new Dictionary<FastNodeKey, FastTensorKey>(stateParamNodeKeys.Length);
            for (int i = 0; i < stateParamNodeKeys.Length; i++)
                stateReplacementByNodeKey[stateParamNodeKeys[i]] = stateFieldKeys[i];

            var remap = new Dictionary<FastTensorKey, FastTensorKey>();
            var stateUpdateOutputs = new List<FastTensorKey>();
            var nodesToRemove = new HashSet<FastNodeKey>();

            // Walk in stored (topological) order so STATE_UPDATE_LINK / WITH_STATE_DEPS
            // can resolve their inputs through any remap entries already added by upstream nodes.
            foreach (var node in graph.Nodes)
            {
                if (modelInputReplacementByNodeKey.TryGetValue(node.Key, out var modelInputFieldKey))
                {
                    var outputKey = GetSingleOutputKey(node);
                    remap[outputKey] = modelInputFieldKey;
                    nodesToRemove.Add(node.Key);
                    continue;
                }

                if (stateReplacementByNodeKey.TryGetValue(node.Key, out var stateFieldKey))
                {
                    var outputKey = GetSingleOutputKey(node);
                    remap[outputKey] = stateFieldKey;
                    nodesToRemove.Add(node.Key);
                    continue;
                }

                if (node.OpCode == InternalOpCodes.STATE_UPDATE_LINK)
                {
                    var inputs = node.FullInputs[""];
                    var updatedStateInput = inputs[1].AssertNotNull();
                    var resolvedUpdatedState = ResolveRemap(remap, updatedStateInput);
                    stateUpdateOutputs.Add(resolvedUpdatedState);
                    var outputKey = GetSingleOutputKey(node);
                    remap[outputKey] = resolvedUpdatedState;
                    nodesToRemove.Add(node.Key);
                    continue;
                }

                if (node.OpCode == InternalOpCodes.WITH_STATE_DEPS)
                {
                    var inputs = node.FullInputs[""];
                    var mainInput = inputs[0].AssertNotNull();
                    var resolvedMain = ResolveRemap(remap, mainInput);
                    var outputKey = GetSingleOutputKey(node);
                    remap[outputKey] = resolvedMain;
                    nodesToRemove.Add(node.Key);
                    continue;
                }
            }

            // Rewire every input slot of every surviving node, replacing keys that match
            // a remap entry with the chain's terminus.
            foreach (var node in graph.Nodes)
            {
                if (nodesToRemove.Contains(node.Key)) continue;
                foreach (var (groupName, slots) in node.FullInputs)
                {
                    for (int i = 0; i < slots.Count; i++)
                    {
                        var k = slots[i];
                        if (k is FastTensorKey tk && remap.ContainsKey(tk))
                            slots[i] = ResolveRemap(remap, tk);
                    }
                }
            }

            // Rewire graph outputs.
            for (int i = 0; i < graph.Outputs.Count; i++)
            {
                if (remap.ContainsKey(graph.Outputs[i]))
                    graph.Outputs[i] = ResolveRemap(remap, graph.Outputs[i]);
            }

            graph.Nodes.RemoveAll(n => nodesToRemove.Contains(n.Key));

            // Move the struct input nodes and their per-field GETFIELD producers to the
            // front of graph.Nodes. Without this, those nodes sit wherever the converter
            // placed them (typically at the end, since the temp CG used to seed the Fast
            // graph only references them via dangling outputs) — but their tensors are
            // now consumed by every former model-input / state-param consumer, which
            // would leave the node list out of topological order.
            var preludeKeys = new HashSet<FastNodeKey> { modelInputStructInputKey.FastNodeKey };
            foreach (var k in modelInputFieldKeys) preludeKeys.Add(k.FastNodeKey);
            if (stateStructInputKey is FastTensorKey ssk)
            {
                preludeKeys.Add(ssk.FastNodeKey);
                foreach (var k in stateFieldKeys) preludeKeys.Add(k.FastNodeKey);
            }

            var preludeNodes = new List<FastNode>(preludeKeys.Count);
            var nonPreludeNodes = new List<FastNode>(graph.Nodes.Count - preludeKeys.Count);
            foreach (var n in graph.Nodes)
            {
                if (preludeKeys.Contains(n.Key)) preludeNodes.Add(n);
                else nonPreludeNodes.Add(n);
            }

            // Order the prelude as [struct input, ...GETFIELDs] for each struct, so each
            // GETFIELD's struct-input dependency is satisfied.
            var orderedPrelude = new List<FastNode>(preludeKeys.Count);
            AppendInOrder(orderedPrelude, preludeNodes, modelInputStructInputKey.FastNodeKey, modelInputFieldKeys);
            if (stateStructInputKey is FastTensorKey sskOrder)
                AppendInOrder(orderedPrelude, preludeNodes, sskOrder.FastNodeKey, stateFieldKeys);

            graph.Nodes.Clear();
            graph.Nodes.AddRange(orderedPrelude);
            graph.Nodes.AddRange(nonPreludeNodes);

            // Replace graph inputs with the new struct inputs.
            // Preserve InputUniqueNames for kept inputs — only the original model input
            // keys are dropped; the trainable param struct input was added by the prior
            // FastReplaceTrainableParamsWithInputProcessor pass and stays.
            var keptInputs = new List<FastTensorKey>();
            var keptInputNames = new List<string?>();
            for (int i = 0; i < graph.Inputs.Count; i++)
            {
                if (modelInputReplacementByNodeKey.ContainsKey(graph.Inputs[i].FastNodeKey))
                    continue;
                keptInputs.Add(graph.Inputs[i]);
                keptInputNames.Add(i < graph.InputUniqueNames.Count ? graph.InputUniqueNames[i] : null);
            }

            // Prepend the new struct inputs ahead of the kept (trainable param struct) input
            // to mirror the CG-side rebuild's [model_inputs_struct, state_struct?, trainable_param_struct]
            // ordering. The output names are looked up from the producing struct-input nodes.
            var newInputs = new List<FastTensorKey> { modelInputStructInputKey };
            var newNames = new List<string?> { LookupInputName(graph, modelInputStructInputKey) };
            if (stateStructInputKey is FastTensorKey sk)
            {
                newInputs.Add(sk);
                newNames.Add(LookupInputName(graph, sk));
            }
            newInputs.AddRange(keptInputs);
            newNames.AddRange(keptInputNames);

            graph.Inputs = newInputs;
            graph.InputUniqueNames = newNames;

            // Append state-update outputs to the graph outputs, mirroring the CG version.
            foreach (var su in stateUpdateOutputs)
                graph.Outputs.Add(su);

            FastProcessorHelper.RemoveUnreachableNodes(graph);

            var rebuiltModelOutput = ResolveRemap(remap, modelOutputKey);
            var rebuiltParamFieldKeys = paramFieldKeys.Select(k => ResolveRemap(remap, k)).ToArray();
            var rebuiltTrainableParamStructInput = ResolveRemap(remap, trainableParamStructInputKey);

            return new ProcessResult(
                rebuiltModelOutput,
                rebuiltParamFieldKeys,
                rebuiltTrainableParamStructInput,
                stateUpdateOutputs.ToImmutableArray());
        }

        private static void AppendInOrder(
            List<FastNode> dest,
            List<FastNode> source,
            FastNodeKey structInputKey,
            FastTensorKey[] fieldKeys)
        {
            foreach (var n in source)
                if (n.Key == structInputKey) { dest.Add(n); break; }

            foreach (var fk in fieldKeys)
                foreach (var n in source)
                    if (n.Key == fk.FastNodeKey) { dest.Add(n); break; }
        }

        private static FastTensorKey GetSingleOutputKey(FastNode node)
        {
            foreach (var slot in node.FullOutputs.Values)
                foreach (var k in slot)
                    if (k is FastTensorKey tk && !tk.IsEmpty)
                        return tk;
            throw new InvalidOperationException(
                $"FastRebuildModelInputsForTrainingProcessor: node '{node.OpCode}' (Key={node.Key}) has no non-empty output key.");
        }

        private static FastTensorKey ResolveRemap(Dictionary<FastTensorKey, FastTensorKey> remap, FastTensorKey key)
        {
            while (remap.TryGetValue(key, out var next))
                key = next;
            return key;
        }

        private static string? LookupInputName(InternalComputationGraph graph, FastTensorKey inputKey)
        {
            for (int i = 0; i < graph.Nodes.Count; i++)
            {
                var node = graph.Nodes[i];
                if (node.Key != inputKey.FastNodeKey) continue;
                return node.FriendlyName;
            }
            return null;
        }
    }
}
