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
            var nodesToRemove = new HashSet<FastNodeKey>();
            // The value each state field currently holds, keyed by that field's GETFIELD output.
            // It starts as the field itself — the value fed in for this step — and each
            // STATE_UPDATE_LINK for that field advances it. A state parameter reached from several
            // call sites is one field with several links (one model handle called twice inlines its
            // body, and its update, once per call), so the updates have to chain: the second body's
            // read must see the first's result, and the struct gets the LAST value, not the first
            // (Shorokoo/Shorokoo#306).
            var currentByStateField = new Dictionary<FastTensorKey, FastTensorKey>(stateFieldKeys.Length);
            foreach (var fieldKey in stateFieldKeys) currentByStateField[fieldKey] = fieldKey;

            // The key a slot should now read: the remap chain's terminus, then that state field's
            // current value if it is one.
            FastTensorKey CurrentValueOf(FastTensorKey key)
            {
                var resolved = remap.ContainsKey(key) ? ResolveRemap(remap, key) : key;
                return currentByStateField.TryGetValue(resolved, out var current) ? current : resolved;
            }

            // Which state field each link updates, resolved BEFORE any rewiring so the keys are
            // still the ones the graph was built with. StateUpdate validates that its original-state
            // argument is produced by the parameter node itself, but inlining wraps values in
            // Identity, so the link's input reaches the parameter through a chain of those.
            var fieldByStateUpdateLink = new Dictionary<FastNodeKey, FastTensorKey>();
            if (stateFieldKeys.Length > 0)
            {
                var nodeByOutput = new Dictionary<FastTensorKey, FastNode>();
                foreach (var node in graph.Nodes)
                    foreach (var (_, outs) in node.FullOutputs)
                        foreach (var outKey in outs)
                            if (outKey is FastTensorKey ok && !ok.IsEmpty) nodeByOutput[ok] = node;

                foreach (var node in graph.Nodes)
                {
                    if (node.OpCode != InternalOpCodes.STATE_UPDATE_LINK) continue;
                    var cursor = node.FullInputs[""][0];
                    while (cursor is FastTensorKey step && nodeByOutput.TryGetValue(step, out var producer))
                    {
                        if (stateReplacementByNodeKey.TryGetValue(producer.Key, out var field))
                        {
                            fieldByStateUpdateLink[node.Key] = field;
                            break;
                        }
                        if (producer.OpCode != OpCodes.IDENTITY) break;
                        var producerInputs = producer.FullInputs[""];
                        if (producerInputs.Count != 1) break;
                        cursor = producerInputs[0];
                    }
                }
            }

            // One ordered walk, rewiring each node's inputs AT ITS OWN POSITION rather than in a
            // second uniform pass. Position is what makes chaining work: the same rewrite applied
            // to the whole graph at once would give every read of a state parameter the same key,
            // so a read after an update could not differ from one before it.
            foreach (var node in graph.Nodes)
            {
                if (modelInputReplacementByNodeKey.TryGetValue(node.Key, out var modelInputFieldKey))
                {
                    remap[GetSingleOutputKey(node)] = modelInputFieldKey;
                    nodesToRemove.Add(node.Key);
                    continue;
                }

                if (stateReplacementByNodeKey.TryGetValue(node.Key, out var stateFieldKey))
                {
                    remap[GetSingleOutputKey(node)] = stateFieldKey;
                    nodesToRemove.Add(node.Key);
                    continue;
                }

                foreach (var (_, slots) in node.FullInputs)
                {
                    for (int i = 0; i < slots.Count; i++)
                    {
                        if (slots[i] is not FastTensorKey tk) continue;
                        var rewired = CurrentValueOf(tk);
                        if (rewired != tk) slots[i] = rewired;
                    }
                }

                if (node.OpCode == InternalOpCodes.STATE_UPDATE_LINK)
                {
                    var resolvedUpdatedState = node.FullInputs[""][1].AssertNotNull();
                    if (fieldByStateUpdateLink.TryGetValue(node.Key, out var field))
                        currentByStateField[field] = resolvedUpdatedState;
                    remap[GetSingleOutputKey(node)] = resolvedUpdatedState;
                    nodesToRemove.Add(node.Key);
                    continue;
                }

                if (node.OpCode == InternalOpCodes.WITH_STATE_DEPS)
                {
                    remap[GetSingleOutputKey(node)] = node.FullInputs[""][0].AssertNotNull();
                    nodesToRemove.Add(node.Key);
                    continue;
                }
            }

            // One output per state FIELD, in field order, so the updated-state struct lines up with
            // the struct def the caller built from the same order. A field nobody updated keeps the
            // value it was fed, which is what pass-through state means.
            var stateUpdateOutputs = new List<FastTensorKey>(stateFieldKeys.Length);
            foreach (var fieldKey in stateFieldKeys) stateUpdateOutputs.Add(currentByStateField[fieldKey]);

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
