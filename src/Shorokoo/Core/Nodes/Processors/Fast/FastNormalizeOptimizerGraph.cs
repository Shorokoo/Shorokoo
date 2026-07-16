using Shorokoo.Graph;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Utils;
using Shorokoo.Modules;
using Shorokoo.Onnx;
using Shorokoo.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using Shorokoo.Core.Nodes.Processors.AutoGrad;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// Result of <see cref="FastNormalizeOptimizerGraph.Process"/>: the optimizer graph's
    /// hyperparameter/state structure plus the graph used to compute initial state values.
    /// </summary>
    internal sealed class NormalizedOptimizerGraphInfo
    {
        /// <summary>Number of leading hyperparameter inputs (everything before currentParam and grad).</summary>
        public int HyperparamCount { get; }

        /// <summary>Number of optimizer state tensors (per trainable parameter).</summary>
        public int StateCount { get; }

        /// <summary>
        /// Graph that computes the initial value of every optimizer state tensor. Inputs are the
        /// optimizer's original inputs <c>[hyperparam_0, ..., currentParam, grad]</c>; outputs are
        /// the state initializer results, one per state in state order. Execute it via
        /// <see cref="FastNormalizeOptimizerGraph.RunStateInitGraph"/> once per trainable
        /// parameter, binding that parameter's initial value (and a zero gradient). Null when
        /// <see cref="StateCount"/> is 0.
        /// </summary>
        public FastComputationGraph? StateInitGraph { get; }

        /// <summary>Element type of each state tensor, in state order.</summary>
        public DType[] StateDTypes { get; }

        /// <summary>Declared rank of each state tensor (null when the initializer's rank is dynamic).</summary>
        public int?[] StateRanks { get; }

        internal NormalizedOptimizerGraphInfo(
            int hyperparamCount,
            int stateCount,
            FastComputationGraph? stateInitGraph,
            DType[] stateDTypes,
            int?[] stateRanks)
        {
            HyperparamCount = hyperparamCount;
            StateCount = stateCount;
            StateInitGraph = stateInitGraph;
            StateDTypes = stateDTypes;
            StateRanks = stateRanks;
        }
    }

    /// <summary>
    /// Normalizes an optimizer <see cref="FastComputationGraph"/> from its authored form into the
    /// explicit input/output convention the TrainingRig's per-parameter replay loop expects.
    ///
    /// <para>
    /// Authored form (what a <c>[Module]</c> optimizer's <c>Inline</c> produces):
    /// inputs are <c>[hyperparam_0, ..., currentParam, grad]</c> only — optimizer state never
    /// appears in the method signature. Each state tensor is created inside the body by an
    /// optimizer-owned <c>[StateInitializer]</c>'s <c>Init</c> call (a <c>MODEL_PARAM_REF</c>
    /// node with <c>shrk_is_trainable=false</c>), and its per-step update is registered via
    /// <see cref="Shorokoo.Globals.StateUpdate{T}(T, T)"/> (a <c>STATE_UPDATE_LINK</c> node, with
    /// the module output wrapped in <c>WITH_STATE_DEPS</c>).
    /// </para>
    ///
    /// <para>
    /// Normalized form: each state's <c>MODEL_PARAM_REF</c> is replaced by a fresh graph
    /// input appended after <c>grad</c> (so inputs become
    /// <c>[hyperparam_0, ..., currentParam, grad, state_0, ...]</c>), <c>WITH_STATE_DEPS</c>
    /// wrappers are unwrapped, and each state's updated tensor is appended as a graph output in
    /// state order (outputs become <c>[updated_param, updated_state_0, ...]</c>). The state
    /// initializer calls themselves are split into a separate state-init graph (see
    /// <see cref="NormalizedOptimizerGraphInfo.StateInitGraph"/>) that the rig executes per
    /// trainable parameter to produce the initial optimizer-state values — the exact same
    /// initializer-execution mechanism used for trainable parameters
    /// (cf. <see cref="FastInitializeModelParams"/>).
    /// </para>
    ///
    /// <see cref="Process"/> mutates the supplied graph in place and returns the discovered structure.
    /// </summary>
    internal static class FastNormalizeOptimizerGraph
    {
        public static NormalizedOptimizerGraphInfo Process(FastComputationGraph graph)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));
            if (graph.Inputs.Count < 2)
                throw new InvalidOperationException(
                    $"Optimizer graph must have at least (currentParam, grad) inputs; it has {graph.Inputs.Count}.");

            const int mandatoryParamAndGradInputs = 2;
            int hyperparamCount = graph.Inputs.Count - mandatoryParamAndGradInputs;

            // Pass 1: discover the state-parameter nodes ([StateInitializer] Init results) in
            // linear (declaration) order, and collect the StateUpdate machinery nodes.
            var stateNodes = new List<FastNode>();
            var stateUpdateLinkNodes = new List<FastNode>();
            bool hasWithStateDeps = false;

            foreach (var node in graph.Nodes)
            {
                switch (node.OpCode)
                {
                    case InternalOpCodes.MODEL_PARAM_REF:
                    case InternalOpCodes.MODEL_PARAM:
                    case InternalOpCodes.MODEL_PARAM_ID_REF:
                    case InternalOpCodes.MODEL_PARAM_MODEL_REF:
                    case InternalOpCodes.MODEL_PARAM_DATA:
                    {
                        var isTrainable = node.Attributes.GetBoolVal(OnnxOpAttributeNames.ShrkAttrIsTrainable) ?? true;
                        if (isTrainable)
                            throw new InvalidOperationException(
                                "Optimizer graph declares a trainable parameter (a [TrainableParamInitializer] " +
                                "Init call or embedded weight). Optimizers receive the current parameter and its " +
                                "gradient as inputs; trainable parameters belong to the model graph. Use a " +
                                "[StateInitializer(Ownership = StateOwnership.OptimizerOwned)] initializer for " +
                                "optimizer state instead.");

                        if (node.OpCode != InternalOpCodes.MODEL_PARAM_REF)
                            throw new InvalidOperationException(
                                $"Optimizer graph contains a '{node.OpCode}' state-parameter node, which is not " +
                                "supported. Pass the optimizer module's raw ComputationGraph to the TrainingRig " +
                                "(optimizer graphs are normalized by the rig itself and must not be pre-processed " +
                                "through the model concretization pipeline).");

                        if (node.TargetFunction is null)
                            throw new InvalidOperationException(
                                "Optimizer state-parameter node has no initializer function; optimizer state must " +
                                "be created via a [StateInitializer] class's Init method.");

                        if (node.TargetFunction.StateOwnership != StateOwnership.OptimizerOwned)
                            throw new InvalidOperationException(
                                $"Optimizer graph creates state via '{node.TargetFunction.DefaultName}', a " +
                                "module-owned state initializer. State created inside an optimizer module is " +
                                "replicated per trainable parameter and carried in the rig's optimizer-state " +
                                "struct, so its initializer must be declared with " +
                                "[StateInitializer(Ownership = StateOwnership.OptimizerOwned)].");

                        stateNodes.Add(node);
                        break;
                    }

                    case InternalOpCodes.STATE_UPDATE_LINK:
                        stateUpdateLinkNodes.Add(node);
                        break;

                    case InternalOpCodes.WITH_STATE_DEPS:
                        hasWithStateDeps = true;
                        break;
                }
            }

            // Stateless optimizer (e.g. plain SGD): nothing to normalize.
            if (stateNodes.Count == 0 && stateUpdateLinkNodes.Count == 0 && !hasWithStateDeps)
                return new NormalizedOptimizerGraphInfo(
                    hyperparamCount, 0, null, Array.Empty<DType>(), Array.Empty<int?>());

            // Pass 2: split off the state-init graph before mutating the main graph. Each state
            // node is rewritten into a FUNCTION_INVOKE of its initializer (dropping the leading
            // iterationIndices input that MODEL_PARAM_REF carries), the invoke results become
            // the graph outputs, and everything not feeding them is pruned. The initializer-arg
            // subgraphs (e.g. ShapeTensor(currentParam)) stay live, so the graph's inputs remain
            // the optimizer's original [hyperparams..., currentParam, grad].
            FastComputationGraph? stateInitGraph = null;
            if (stateNodes.Count > 0)
            {
                stateInitGraph = graph.Clone();
                var initNodesByKey = stateInitGraph.Nodes.ToDictionary(n => n.Key);
                var functionInvokeAttrDefs = Definitions.NodeDefinitions[InternalOpCodes.FUNCTION_INVOKE].AttributeDefs;
                var initOutputs = new List<FastTensorKey>(stateNodes.Count);

                foreach (var stateNode in stateNodes)
                {
                    var initNode = initNodesByKey[stateNode.Key];

                    var dtype = stateNode.Attributes.GetDTypeVal(OnnxOpAttributeNames.ShrkAttrDtype).AssertNotNull();
                    var rank = stateNode.Attributes.GetLongVal(OnnxOpAttributeNames.ShrkAttrRank) ?? -1;

                    initNode.OpCode = InternalOpCodes.FUNCTION_INVOKE;
                    initNode.Attributes = OnnxCSharpAttributes.FromCSharpVals(
                        new Dictionary<string, object?>
                        {
                            [OnnxOpAttributeNames.ShrkAttrStructure] = new[] { DataStructure.Tensor },
                            [OnnxOpAttributeNames.ShrkAttrDtype] = new[] { dtype },
                            [OnnxOpAttributeNames.ShrkAttrRank] = new[] { rank },
                            [OnnxOpAttributeNames.ShrkAttrGenericTypeArgs] = null,
                        },
                        functionInvokeAttrDefs);
                    initNode.IdentifierTemplate = null;

                    // MODEL_PARAM_REF inputs are [iterationIndices, ...initializerParams];
                    // FUNCTION_INVOKE expects just the initializer params. TargetFunction (the
                    // initializer fn) is preserved by the clone.
                    var slots = initNode.FullInputs[""];
                    initNode.FullInputs[""] = slots.Skip(1).ToList();

                    initOutputs.Add(initNode.FullOutputs[""][0]!.Value);
                }

                stateInitGraph.Outputs = initOutputs;
                stateInitGraph.OutputUniqueNames = initOutputs.Select(_ => (string?)null).ToList();
                stateInitGraph.OutputRankOverrides = null;
                FastProcessorHelper.RemoveUnreachableNodes(stateInitGraph);
            }

            // Pass 3: replace each state node in the main graph with a fresh runtime input
            // appended after grad, in state (declaration) order.
            var stateDTypes = new DType[stateNodes.Count];
            var stateRanks = new int?[stateNodes.Count];
            var remap = new Dictionary<FastTensorKey, FastTensorKey>(stateNodes.Count);
            var stateIndexByInputKey = new Dictionary<FastTensorKey, int>(stateNodes.Count);
            var stateNodeKeys = new HashSet<FastNodeKey>(stateNodes.Select(n => n.Key));

            for (int i = 0; i < stateNodes.Count; i++)
            {
                var stateNode = stateNodes[i];
                stateDTypes[i] = stateNode.Attributes.GetDTypeVal(OnnxOpAttributeNames.ShrkAttrDtype).AssertNotNull();
                var rankLong = stateNode.Attributes.GetLongVal(OnnxOpAttributeNames.ShrkAttrRank);
                stateRanks[i] = rankLong is long r ? checked((int)r) : null;

                var inputName = $"optimizer_state_{i}";
                var inputNode = FastInternalOp.RuntimeInput(stateDTypes[i], stateRanks[i], inputName);
                graph.Nodes.Insert(i, inputNode);
                var inputKey = new FastTensorKey(inputNode.Key, 0);

                remap[stateNode.FullOutputs[""][0]!.Value] = inputKey;
                stateIndexByInputKey[inputKey] = i;
                graph.Inputs.Add(inputKey);
                graph.InputUniqueNames.Add(inputName);
            }

            graph.Nodes.RemoveAll(n => stateNodeKeys.Contains(n.Key));

            // Rewire every consumer of a replaced state node onto the new input.
            if (remap.Count > 0)
            {
                foreach (var node in graph.Nodes)
                {
                    foreach (var (_, slots) in node.FullInputs)
                    {
                        for (int j = 0; j < slots.Count; j++)
                        {
                            if (slots[j] is FastTensorKey k && remap.TryGetValue(k, out var mapped))
                                slots[j] = mapped;
                        }
                    }
                }
            }

            // Pass 4: match each STATE_UPDATE_LINK to its state input. The link's original-state
            // input either references the state input directly or through the Identity nodes that
            // rank casts like .Vec() insert; trace through those.
            var producerByOutput = new Dictionary<FastTensorKey, FastNode>();
            foreach (var node in graph.Nodes)
            {
                foreach (var (_, outs) in node.FullOutputs)
                {
                    foreach (var ok in outs)
                    {
                        if (ok is FastTensorKey k && !k.IsEmpty)
                            producerByOutput[k] = node;
                    }
                }
            }

            FastTensorKey ResolveThroughIdentities(FastTensorKey key)
            {
                while (producerByOutput.TryGetValue(key, out var producer)
                       && producer.OpCode == OpCodes.IDENTITY
                       && producer.FullInputs.TryGetValue("", out var idSlots)
                       && idSlots.Count > 0
                       && idSlots[0] is FastTensorKey inner
                       && !inner.IsEmpty)
                {
                    key = inner;
                }
                return key;
            }

            var originalInputPositions = new Dictionary<FastTensorKey, int>();
            for (int i = 0; i < hyperparamCount + mandatoryParamAndGradInputs; i++)
                originalInputPositions[graph.Inputs[i]] = i;

            var updatedStateKeys = new FastTensorKey?[stateNodes.Count];
            foreach (var linkNode in stateUpdateLinkNodes)
            {
                // Both link inputs were already rewired onto the new state inputs by the remap
                // loop above (covering the degenerate StateUpdate(state, state) shape too).
                var linkInputs = linkNode.FullInputs[""];
                if (linkInputs.Count < 2
                    || linkInputs[0] is not FastTensorKey origKey || origKey.IsEmpty
                    || linkInputs[1] is not FastTensorKey updKey || updKey.IsEmpty)
                    continue;

                var resolvedOrig = ResolveThroughIdentities(origKey);
                if (!stateIndexByInputKey.TryGetValue(resolvedOrig, out var stateIndex))
                {
                    if (originalInputPositions.ContainsKey(resolvedOrig))
                        throw new InvalidOperationException(
                            "The optimizer's StateUpdate targets a graph input. Optimizer state must not be " +
                            "declared in the optimizer's Inline method signature — method parameters are runtime " +
                            "inputs, not state. Create the state inside the body via an optimizer-owned " +
                            "[StateInitializer]'s Init method (e.g. " +
                            "OptimizerStateZeros.Init(currentParam.ShapeTensor())) and pass that to StateUpdate.");

                    throw new InvalidOperationException(
                        "The optimizer's StateUpdate targets a tensor that is not a state variable. Optimizer " +
                        "state must be created via an optimizer-owned [StateInitializer]'s Init method (e.g. " +
                        "OptimizerStateZeros.Init(currentParam.ShapeTensor())); the unmodified result of that " +
                        "Init call is what StateUpdate's first argument must be.");
                }

                if (updatedStateKeys[stateIndex] is not null)
                    throw new InvalidOperationException(
                        $"Optimizer state #{stateIndex} receives more than one StateUpdate; each state variable " +
                        "must be updated exactly once per step (combine the updates into a single value, e.g. " +
                        "with IfElse, and register it with one StateUpdate call).");

                updatedStateKeys[stateIndex] = updKey;
            }

            for (int i = 0; i < updatedStateKeys.Length; i++)
            {
                if (updatedStateKeys[i] is null)
                    throw new InvalidOperationException(
                        $"Optimizer state #{i} (initializer '{stateNodes[i].TargetFunction!.DefaultName}') is " +
                        "created but never updated. Every optimizer state variable must receive exactly one " +
                        "Globals.StateUpdate(state, newValue) call so the rig can round-trip it between steps.");
            }

            // Pass 5: unwrap WITH_STATE_DEPS from the primary outputs and append the updated-state
            // tensors as outputs in state order. The link / deps nodes become dead and are pruned.
            var newOutputs = new List<FastTensorKey>(graph.Outputs.Count + stateNodes.Count);
            foreach (var outKey in graph.Outputs)
            {
                if (producerByOutput.TryGetValue(outKey, out var producer)
                    && producer.OpCode == InternalOpCodes.WITH_STATE_DEPS
                    && producer.FullInputs.TryGetValue("", out var wsdSlots)
                    && wsdSlots.Count > 0
                    && wsdSlots[0] is FastTensorKey wsdMain
                    && !wsdMain.IsEmpty)
                {
                    newOutputs.Add(wsdMain);
                }
                else
                {
                    newOutputs.Add(outKey);
                }
            }

            int primaryOutputCount = newOutputs.Count;
            foreach (var upd in updatedStateKeys)
                newOutputs.Add(upd!.Value);

            var newOutputNames = new List<string?>(newOutputs.Count);
            for (int i = 0; i < newOutputs.Count; i++)
                newOutputNames.Add(i < primaryOutputCount && i < graph.OutputUniqueNames.Count
                    ? graph.OutputUniqueNames[i] : null);

            graph.Outputs = newOutputs;
            graph.OutputUniqueNames = newOutputNames;
            graph.OutputRankOverrides = null;

            FastProcessorHelper.RemoveUnreachableNodes(graph);

            return new NormalizedOptimizerGraphInfo(
                hyperparamCount, stateNodes.Count, stateInitGraph, stateDTypes, stateRanks);
        }

        /// <summary>
        /// Executes a <see cref="NormalizedOptimizerGraphInfo.StateInitGraph"/> with the given
        /// input bindings (one <see cref="TensorData"/> per graph input, in input order:
        /// hyperparameter seed values, then the parameter's initial value, then a zero gradient)
        /// and returns the initial state values in state order. Mirrors
        /// <see cref="FastInitializeModelParams"/>: inputs are baked as constants and the
        /// initializer FUNCTION_INVOKEs run through <see cref="ComputeContext.Run(FastComputationGraph, NamedModelParam[])"/>.
        /// </summary>
        internal static TensorData[] RunStateInitGraph(
            FastComputationGraph stateInitGraph,
            ComputeContext computeContext,
            TensorData[] boundInputs)
        {
            if (stateInitGraph.Inputs.Count != boundInputs.Length)
                throw new ArgumentException(
                    $"State-init graph expects {stateInitGraph.Inputs.Count} inputs but {boundInputs.Length} " +
                    "values were bound.", nameof(boundInputs));

            var workGraph = stateInitGraph.Clone();
            var nodesByKey = workGraph.Nodes.ToDictionary(n => n.Key);
            var constantAttrDefs = Definitions.NodeDefinitions[OpCodes.CONSTANT].AttributeDefs;

            // Bake each graph input as a constant in place (preserving the node's output key, so
            // no consumer rewiring is needed), then run the now-closed graph.
            for (int i = 0; i < workGraph.Inputs.Count; i++)
            {
                var inputNode = nodesByKey[workGraph.Inputs[i].FastNodeKey];
                inputNode.OpCode = OpCodes.CONSTANT;
                inputNode.Attributes = OnnxCSharpAttributes.FromCSharpVals(
                    new Dictionary<string, object?> { [OnnxOpAttributeNames.AttrValue] = boundInputs[i] },
                    constantAttrDefs);
                inputNode.FullInputs = new Dictionary<string, List<FastTensorKey?>>();
                inputNode.FriendlyName = null;
            }

            workGraph.Inputs = new List<FastTensorKey>();
            workGraph.InputUniqueNames = new List<string?>();

            var results = computeContext.Run(workGraph);
            return results.Select(r => r.ToTensorData()).ToArray();
        }
    }
}
