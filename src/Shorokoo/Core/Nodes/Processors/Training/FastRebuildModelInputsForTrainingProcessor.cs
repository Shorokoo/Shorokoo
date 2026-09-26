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
    /// its main input. The replaced nodes themselves — the original model inputs among them —
    /// are dropped from <c>graph.Nodes</c>, and the new struct inputs lead the inputs.
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

            // The inputs that stay: all but the original model inputs, which the struct replaces,
            // and the new struct inputs, which lead.
            var keptInputs = graph.Inputs
                .Where(k => !originalModelInputNodeKeys.Contains(k.FastNodeKey)
                         && k != modelInputStructInputKey && k != stateStructInputKey)
                .ToList();

            var modelInputReplacementByNodeKey = new Dictionary<FastNodeKey, FastTensorKey>(originalModelInputNodeKeys.Length);
            for (int i = 0; i < originalModelInputNodeKeys.Length; i++)
                modelInputReplacementByNodeKey[originalModelInputNodeKeys[i]] = modelInputFieldKeys[i];

            var stateReplacementByNodeKey = new Dictionary<FastNodeKey, FastTensorKey>(stateParamNodeKeys.Length);
            for (int i = 0; i < stateParamNodeKeys.Length; i++)
                stateReplacementByNodeKey[stateParamNodeKeys[i]] = stateFieldKeys[i];

            var remap = new Dictionary<FastTensorKey, FastTensorKey>();
            var nodesToRemove = new HashSet<FastNodeKey>();

            // One updated value per state FIELD, not per link. A model called twice keeps one
            // state parameter and one update per call, so appending per link hands the struct more
            // values than it has fields and the surplus is dropped (Shorokoo/Shorokoo#306). Track
            // what each field currently holds instead: a link updating a field advances it, and the
            // last value is what the field carries out. Chaining has already pointed a later call's
            // link at the earlier call's, so the final value carries every update in call order.
            var currentByField = new FastTensorKey[stateFieldKeys.Length];
            var heldByField = new HashSet<FastTensorKey>[stateFieldKeys.Length];
            for (int i = 0; i < stateFieldKeys.Length; i++)
            {
                currentByField[i] = stateFieldKeys[i];
                heldByField[i] = [stateFieldKeys[i]];
            }
            var nodeByKeyForState = new Dictionary<FastNodeKey, FastNode>(graph.Nodes.Count);
            foreach (var n in graph.Nodes) nodeByKeyForState[n.Key] = n;

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

                    // Which field this advances is what its original-state input names: the field
                    // itself for the first call, an earlier call's updated value after. Any value
                    // the field has held counts, not only the latest — the arms of an IfElse each
                    // read the value the field held entering the branch. Inlining wraps that value
                    // in Identity, so the walk has to see through them; a remap lookup alone finds
                    // nothing and the field goes unidentified.
                    var resolvedOriginalState = ResolveStateValue(
                        inputs[0].AssertNotNull(), remap, nodeByKeyForState);
                    int field = System.Array.FindIndex(heldByField, held => held.Contains(resolvedOriginalState));
                    if (field < 0)
                        throw new InvalidOperationException(
                            "FastRebuildModelInputsForTrainingProcessor: a state update reads a value that is "
                            + "not one this state field has held, so the field it updates cannot be "
                            + "identified and its update would be dropped.");
                    currentByField[field] = resolvedUpdatedState;
                    heldByField[field].Add(resolvedUpdatedState);

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

            // Rewire every input slot of every surviving node (the output nodes among them),
            // replacing keys that match a remap entry with the chain's terminus.
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

            graph.Nodes.RemoveAll(n => nodesToRemove.Contains(n.Key));

            // Settle the inputs: the new struct inputs ahead of the kept ones (the trainable param
            // struct input the prior FastReplaceTrainableParamsWithInputProcessor pass added), to
            // mirror the CG-side rebuild's [model_inputs_struct, state_struct?, trainable_param_struct]
            // ordering. The original model inputs were removed above with the nodes they replace.
            // The struct inputs were appended at the tail for convenience, and so were their
            // per-field GETFIELDs — which every former model-input / state-param consumer now
            // reads — so those move to the start of the body, each struct's in field order.
            List<FastTensorKey> newInputs = [modelInputStructInputKey];
            if (stateStructInputKey is FastTensorKey sk) newInputs.Add(sk);
            newInputs.AddRange(keptInputs);
            graph.SetInputs(newInputs);

            List<FastTensorKey> preludeKeys = [.. modelInputFieldKeys, .. stateFieldKeys];
            var preludeNodeKeys = preludeKeys.Select(k => k.FastNodeKey).ToHashSet();
            var byKey = graph.Nodes.Where(n => preludeNodeKeys.Contains(n.Key)).ToDictionary(n => n.Key);
            graph.Nodes.RemoveAll(n => preludeNodeKeys.Contains(n.Key));
            graph.InsertAtBodyStart(preludeKeys.Select(k => byKey[k.FastNodeKey]));

            // Append one state output per field — the value that field ends the forward holding.
            foreach (var su in currentByField)
                graph.AddOutput(su);

            FastProcessorHelper.RemoveUnreachableNodes(graph);

            var rebuiltModelOutput = ResolveRemap(remap, modelOutputKey);
            var rebuiltParamFieldKeys = paramFieldKeys.Select(k => ResolveRemap(remap, k)).ToArray();
            var rebuiltTrainableParamStructInput = ResolveRemap(remap, trainableParamStructInputKey);

            return new ProcessResult(
                rebuiltModelOutput,
                rebuiltParamFieldKeys,
                rebuiltTrainableParamStructInput,
                currentByField.ToImmutableArray());
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

        /// <summary>
        /// What a state-update's original-state input actually names, seeing through both the remap
        /// this pass builds and the Identity wrappers inlining leaves between a link and the
        /// parameter it updates.
        /// </summary>
        private static FastTensorKey ResolveStateValue(
            FastTensorKey key,
            Dictionary<FastTensorKey, FastTensorKey> remap,
            Dictionary<FastNodeKey, FastNode> nodeByKey)
        {
            var current = ResolveRemap(remap, key);
            var visited = new HashSet<FastTensorKey>();
            while (visited.Add(current))
            {
                if (!nodeByKey.TryGetValue(current.FastNodeKey, out var node)) return current;
                if (node.OpCode != OpCodes.IDENTITY) return current;
                if (node.Inputs.Count == 0 || node.Inputs[0] is not FastTensorKey inner) return current;
                current = ResolveRemap(remap, inner);
            }
            return current;
        }

        private static FastTensorKey ResolveRemap(Dictionary<FastTensorKey, FastTensorKey> remap, FastTensorKey key)
        {
            while (remap.TryGetValue(key, out var next))
                key = next;
            return key;
        }
    }
}
