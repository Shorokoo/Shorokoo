using Shorokoo.Graph;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Nodes.Processors.Training;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using Shorokoo.Core.Utils;
using Shorokoo.Onnx;
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Reflection;
using Shorokoo;
using Shorokoo.Core;

namespace Shorokoo.Core.Training;

/// <summary>
/// Builds computation graphs suitable for training by composing a model's forward pass,
/// a loss function, and automatic differentiation into a single unified graph.
/// 
/// The resulting graph takes model input (TIn), targets (TOut), and trainable parameter
/// fields as inputs, and outputs the loss value and gradient fields representing ∂loss/∂params.
/// </summary>
public static class TrainingGraphBuilder
{
    /// <summary>
    /// Prepares a model for training by composing it with a loss function (provided as a
    /// module's Inline method reference) and automatic differentiation.
    ///
    /// The loss function Func must reference the Inline method of a [Module]-annotated class.
    /// The module class's ComputationGraph property is used to obtain the loss computation graph.
    /// </summary>
    /// <typeparam name="TOut">The model output / loss input type (e.g., Tensor&lt;float32&gt;)</typeparam>
    /// <typeparam name="TLoss">The loss output type (e.g., Scalar&lt;float32&gt;)</typeparam>
    /// <param name="modelGraph">The computation graph for the model (from a module's ComputationGraph property)</param>
    /// <param name="lossFunction">A Func referencing a loss module's Inline method (2 inputs → 1 output)</param>
    /// <returns>A high-level <see cref="FastComputationGraph"/> containing AutoGrad nodes, with inputs
    /// [model_inputs..., targets, param_struct] and outputs [loss, gradient_struct]</returns>
    public static FastComputationGraph PrepareForTrainingAsFast<TOut, TLoss>(
        FastComputationGraph modelGraph,
        Func<TOut, TOut, TLoss> lossFunction)
        where TOut : IValue
        where TLoss : IValue
    {
        if (modelGraph is null) throw new ArgumentNullException(nameof(modelGraph));
        if (lossFunction is null) throw new ArgumentNullException(nameof(lossFunction));

        var lossGraph = ExtractFastGraphFromDelegate(lossFunction);
        return PrepareForTrainingAsFast(modelGraph, lossGraph);
    }

    /// <summary>
    /// Prepares a model for training by composing it with a loss function and automatic
    /// differentiation. Returns a high-level <see cref="FastComputationGraph"/> containing
    /// <c>AUTO_GRAD</c> nodes, with inputs [model_inputs_struct, targets, param_struct,
    /// state_struct?] and outputs [loss, gradient_struct, state_struct].
    ///
    /// <para>
    /// <paramref name="modelGraph"/> may be in raw or already-concrete form. When raw, the
    /// legacy minimal-processing path runs (no input-aware liveness filter). When the caller
    /// has pre-routed the graph through
    /// <see cref="Shorokoo.Graph.FastComputationGraphExtensions.ToConcreteArchitecture"/>
    /// — which TrainingRig.FromScratch does — those minimal passes are no-ops on the already-
    /// concrete graph, and downstream trainable-param discovery picks up exactly the live
    /// (post-liveness-filter) TRAINABLE_PARAM nodes.
    /// </para>
    /// </summary>
    public static FastComputationGraph PrepareForTrainingAsFast(
        FastComputationGraph modelGraph,
        FastComputationGraph lossGraph)
    {
        if (modelGraph is null) throw new ArgumentNullException(nameof(modelGraph));
        if (lossGraph is null) throw new ArgumentNullException(nameof(lossGraph));

        // Validate loss graph shape first so we fail fast.
        if (lossGraph.Inputs.Count != 2)
            throw new ArgumentException(
                $"Loss graph must have exactly 2 inputs (predictions, targets), but has {lossGraph.Inputs.Count}.",
                nameof(lossGraph));
        if (lossGraph.Outputs.Count != 1)
            throw new ArgumentException(
                $"Loss graph must have exactly 1 output (loss), but has {lossGraph.Outputs.Count}.",
                nameof(lossGraph));

        // Step 1+2: Process model graph for training and replace trainable params with a
        // TensorStruct input — both passes happen on a single FastComputationGraph that
        // becomes the host graph for the rest of this function. If the caller has already
        // routed the graph through ToConcreteArchitecture (TrainingRig.FromScratch does
        // this so it can run input-aware liveness filtering once for the whole pipeline),
        // skip ProcessGraphForTrainingOnFast — its first pass, FastApplyIdentifierTemplates,
        // asserts on the TRAINABLE_PARAM nodes that the concrete-arch pipeline produces.
        var fastGraph = modelGraph.Clone();
        if (!IsAlreadyConcretized(fastGraph))
            ProcessGraphForTrainingOnFast(fastGraph);
        var fastReplaceResult = Nodes.Processors.Training.FastReplaceTrainableParamsWithInputProcessor.Process(fastGraph);

        var trainableParamStructInputKey = fastReplaceResult.TrainableParamStructInputKey;
        var paramFieldKeys = fastReplaceResult.ParamFieldKeys;
        var trainableParamStructDef = fastReplaceResult.TrainableParamStructDef;

        // Identify model's original inputs (everything except the new param struct input).
        var originalModelInputKeys = new List<FastTensorKey>();
        var originalModelInputNames = new List<string?>();
        for (int i = 0; i < fastGraph.Inputs.Count; i++)
        {
            if (fastGraph.Inputs[i] == trainableParamStructInputKey) continue;
            originalModelInputKeys.Add(fastGraph.Inputs[i]);
            originalModelInputNames.Add(i < fastGraph.InputUniqueNames.Count ? fastGraph.InputUniqueNames[i] : null);
        }

        // The model's single output (prediction).
        if (fastGraph.Outputs.Count != 1)
            throw new ArgumentException(
                $"Model graph must have exactly 1 output, but has {fastGraph.Outputs.Count}.",
                nameof(modelGraph));
        var modelOutputKey = fastGraph.Outputs[0];

        // Producer-by-output map for reading dtype/rank attributes off model-input nodes.
        var producerByOutput = BuildProducerByOutputMap(fastGraph);

        // Step 4: Discover state parameters (non-trainable model params like BatchNorm
        // running stats). Done on the Fast graph directly.
        var stateParamInfos = FastDiscoverStateParamsProcessor.Process(fastGraph);

        // Model state is module-owned (updated by the module's own forward logic). An
        // optimizer-owned state initializer inside a model would silently become model state
        // here, so reject it loudly instead.
        foreach (var stateParam in stateParamInfos)
        {
            if (stateParam.Node.TargetFunction?.StateOwnership == StateOwnership.OptimizerOwned)
                throw new ArgumentException(
                    $"Model graph creates state via '{stateParam.Node.TargetFunction.DefaultName}', an " +
                    "optimizer-owned state initializer. Optimizer-owned state ([StateInitializer(Ownership = " +
                    "StateOwnership.OptimizerOwned)]) is replicated per trainable parameter and may only be " +
                    "created inside an optimizer module; module state (e.g. running statistics) must use " +
                    "StateOwnership.ModuleOwned.",
                    nameof(modelGraph));
        }

        // Track input-style nodes we add so we can move them to the front of
        // fastGraph.Nodes at the end (in creation order). Each fastGraph.Nodes.Add
        // for an INPUT or its GETFIELD also records into headNodesInOrder.
        var headNodesInOrder = new List<FastNode>();

        // Step 5: Build state struct + GETFIELDs in fastGraph (if any state).
        TensorStructDef stateStructDef;
        FastTensorKey? stateStructInputKey = null;
        var stateFieldKeys = new FastTensorKey[stateParamInfos.Length];

        if (stateParamInfos.Length > 0)
        {
            stateStructDef = FastBuildTrainableParamStructDefProcessor.Process(stateParamInfos, "ModelState");
            var stateStructDType = DType.GetOrCreateForTensorStruct(stateStructDef);
            var stateStructNode = Nodes.Processors.Fast.FastInternalOp.TensorStructInput(stateStructDType, "model_state");
            fastGraph.Nodes.Add(stateStructNode);
            headNodesInOrder.Add(stateStructNode);
            stateStructInputKey = new FastTensorKey(stateStructNode.Key, 0);

            for (int i = 0; i < stateParamInfos.Length; i++)
            {
                var fieldDef = stateStructDef.Fields[i];
                var getField = Nodes.Processors.Fast.FastInternalOp.TensorStructGetField(
                    stateStructInputKey.Value, fieldDef.Name, fieldDef.ElementType, fieldDef.Rank, fieldDef.Structure);
                fastGraph.Nodes.Add(getField);
                headNodesInOrder.Add(getField);
                stateFieldKeys[i] = new FastTensorKey(getField.Key, 0);
            }
        }
        else
        {
            stateStructDef = new TensorStructDef(Array.Empty<TensorStructFieldDef>(), "ModelState");
        }

        // Step 6: Build model_inputs_struct + per-field GETFIELDs in fastGraph.
        var modelInputFields = new TensorStructFieldDef[originalModelInputKeys.Count];
        for (int i = 0; i < originalModelInputKeys.Count; i++)
        {
            var inputProducer = producerByOutput[originalModelInputKeys[i]];
            var dtype = inputProducer.Attributes.GetDTypeVal(OnnxOpAttributeNames.AttrDtype)
                ?? throw new InvalidOperationException(
                    $"Model input node {inputProducer.OpCode} (Key={inputProducer.Key}) has no AttrDtype attribute.");
            var rank = (int?)inputProducer.Attributes.GetLongVal(OnnxOpAttributeNames.ShrkAttrRank);
            var name = originalModelInputNames[i] ?? $"input_{i}";
            modelInputFields[i] = new TensorStructFieldDef(name, DataStructure.Tensor, rank, dtype);
        }
        var modelInputStructDef = new TensorStructDef(modelInputFields, "ModelInputs");
        var modelInputStructDType = DType.GetOrCreateForTensorStruct(modelInputStructDef);
        var modelInputStructNode = Nodes.Processors.Fast.FastInternalOp.TensorStructInput(modelInputStructDType, "model_inputs");
        fastGraph.Nodes.Add(modelInputStructNode);
        headNodesInOrder.Add(modelInputStructNode);
        var modelInputStructInputKey = new FastTensorKey(modelInputStructNode.Key, 0);

        var modelInputFieldKeys = new FastTensorKey[originalModelInputKeys.Count];
        for (int i = 0; i < originalModelInputKeys.Count; i++)
        {
            var fieldDef = modelInputStructDef.Fields[i];
            var getField = Nodes.Processors.Fast.FastInternalOp.TensorStructGetField(
                modelInputStructInputKey, fieldDef.Name, fieldDef.ElementType, fieldDef.Rank, fieldDef.Structure);
            fastGraph.Nodes.Add(getField);
            headNodesInOrder.Add(getField);
            modelInputFieldKeys[i] = new FastTensorKey(getField.Key, 0);
        }

        // Step 7 (was step 6): Rewire model-input and state-param nodes through the new
        // struct inputs. Mutates fastGraph in place; struct inputs replace the original
        // model inputs in fastGraph.Inputs.
        var fastRebuildResult = Nodes.Processors.Training.FastRebuildModelInputsForTrainingProcessor.Process(
            fastGraph,
            originalModelInputNodeKeys: originalModelInputKeys.Select(k => k.FastNodeKey).ToArray(),
            modelInputFieldKeys: modelInputFieldKeys,
            modelInputStructInputKey: modelInputStructInputKey,
            stateParamNodeKeys: stateParamInfos.Select(p => p.Node.Key).ToArray(),
            stateFieldKeys: stateFieldKeys,
            stateStructInputKey: stateStructInputKey,
            trainableParamStructInputKey: trainableParamStructInputKey,
            paramFieldKeys: paramFieldKeys,
            modelOutputKey: modelOutputKey);

        var rebuiltModelOutput = fastRebuildResult.RebuiltModelOutput;
        var rebuiltParamFieldKeys = fastRebuildResult.RebuiltParamFieldKeys;
        var rebuiltTrainableParamStructInput = fastRebuildResult.RebuiltTrainableParamStructInput;
        var stateUpdateOutputs = fastRebuildResult.StateUpdateOutputs;

        // Step 8 (was step 7): create target input.
        var (lossTargetType, lossTargetRank, lossTargetName) = ResolveFastInputDef(lossGraph, 1);
        var targetInputNode = Nodes.Processors.Fast.FastInternalOp.RuntimeInput(
            lossTargetType, lossTargetRank, lossTargetName ?? "targets");
        fastGraph.Nodes.Add(targetInputNode);
        headNodesInOrder.Add(targetInputNode);
        var targetInputKey = new FastTensorKey(targetInputNode.Key, 0);

        // Step 9 (was step 8): replay the loss graph into fastGraph with [prediction, target]
        // as the two source inputs.
        var lossReplayOutputs = Nodes.Processors.Fast.FastReplay.ReplayInto(
            fastGraph, lossGraph,
            mappedInputs: new[] { rebuiltModelOutput, targetInputKey });
        var lossOutputKey = lossReplayOutputs[0];

        // Step 10 (was step 9): emit AUTO_GRAD node.
        var autoGradNode = Nodes.Processors.Fast.FastInternalOp.AutoGrad(lossOutputKey, rebuiltParamFieldKeys);
        fastGraph.Nodes.Add(autoGradNode);
        var gradientKeys = new FastTensorKey[rebuiltParamFieldKeys.Length];
        for (int i = 0; i < rebuiltParamFieldKeys.Length; i++)
            gradientKeys[i] = new FastTensorKey(autoGradNode.Key, i);

        // Step 11 (was step 10): pack gradients into a TensorStruct.
        var gradientStructKey = Nodes.Processors.Fast.FastStructGradientPacker.PackGradients(
            fastGraph, trainableParamStructDef, gradientKeys);

        // Step 12 (was step 11): pack updated state into a struct (empty if no state).
        var stateOutputDType = DType.GetOrCreateForTensorStruct(stateStructDef);
        var updatedStateStructNode = Nodes.Processors.Fast.FastInternalOp.TensorStructCreate(
            stateOutputDType, stateUpdateOutputs);
        fastGraph.Nodes.Add(updatedStateStructNode);
        var updatedStateStructKey = new FastTensorKey(updatedStateStructNode.Key, 0);

        // Step 13 (was step 12): finalize fastGraph's inputs and outputs.
        // Desired input order: [model_inputs_struct, targets, param_struct, state_struct?]
        // After Step 7 fastGraph.Inputs is [model_inputs_struct, state_struct?, param_struct].
        var finalInputs = new List<FastTensorKey> { modelInputStructInputKey, targetInputKey, rebuiltTrainableParamStructInput };
        var finalNames = new List<string?> { "model_inputs", "targets", LookupInputName(fastGraph, rebuiltTrainableParamStructInput) };
        if (stateStructInputKey is FastTensorKey ssk)
        {
            finalInputs.Add(ssk);
            finalNames.Add("model_state");
        }
        fastGraph.Inputs = finalInputs;
        fastGraph.InputUniqueNames = finalNames;

        // Outputs: [loss, gradient_struct, state_struct].
        fastGraph.Outputs = new List<FastTensorKey> { lossOutputKey, gradientStructKey, updatedStateStructKey };
        fastGraph.OutputUniqueNames = new List<string?>(new string?[3]);
        fastGraph.OutputRankOverrides = null;

        Nodes.Processors.Fast.FastProcessorHelper.RemoveUnreachableNodes(fastGraph);

        // Move the input-style nodes added by this builder (struct inputs +
        // their GETFIELDs, runtime target input) to the front in the order they
        // were created — they were appended at the tail for convenience but
        // every body node consumes them, so they belong before the body in
        // topological order. INPUT/GETFIELD nodes carry no scope of their own
        // and the body is already nested by construction, so prepending these
        // doesn't break nesting and removes the need for a Kahn re-sort.
        var headKeys = new HashSet<FastNodeKey>(headNodesInOrder.Select(n => n.Key));
        var rebuilt = new List<FastNode>(fastGraph.Nodes.Count);
        rebuilt.AddRange(headNodesInOrder);
        foreach (var n in fastGraph.Nodes)
            if (!headKeys.Contains(n.Key)) rebuilt.Add(n);
        fastGraph.Nodes = rebuilt;
        System.Diagnostics.Debug.Assert(fastGraph.IsLinearOrderValid(), "fastGraph.IsLinearOrderValid()");

        return fastGraph;
    }

    private static Dictionary<FastTensorKey, FastNode> BuildProducerByOutputMap(FastComputationGraph graph)
    {
        var map = new Dictionary<FastTensorKey, FastNode>();
        foreach (var node in graph.Nodes)
        {
            foreach (var (_, outs) in node.FullOutputs)
            {
                foreach (var ok in outs)
                {
                    if (ok is FastTensorKey k && !k.IsEmpty)
                        map[k] = node;
                }
            }
        }
        return map;
    }

    private static string? LookupInputName(FastComputationGraph graph, FastTensorKey inputKey)
    {
        for (int i = 0; i < graph.Inputs.Count; i++)
            if (graph.Inputs[i] == inputKey)
                return i < graph.InputUniqueNames.Count ? graph.InputUniqueNames[i] : null;
        return null;
    }

    /// <summary>
    /// Processes a module's computation graph for training in place. Runs the same pipeline
    /// steps as ToConcreteArchitecture but stops before the ConvertTrainableParamIdRefToTrainableParam
    /// step (which requires graph execution and fails with symbolic model inputs).
    /// </summary>
    /// <summary>
    /// A graph is "concretized" (already through
    /// <see cref="Shorokoo.Graph.FastComputationGraphExtensions.ToConcreteArchitecture"/>)
    /// iff it has no high-level forms left: no MODEL_INVOKE, FUNCTION_INVOKE,
    /// TRAINABLE_PARAM_REF, or TRAINABLE_PARAM_MODEL_REF nodes. Used to decide whether
    /// <see cref="ProcessGraphForTrainingOnFast"/> needs to run.
    /// </summary>
    private static bool IsAlreadyConcretized(FastComputationGraph graph)
    {
        foreach (var node in graph.Nodes)
        {
            if (node.OpCode == InternalOpCodes.MODEL_INVOKE
                || node.OpCode == InternalOpCodes.FUNCTION_INVOKE
                || node.OpCode == InternalOpCodes.TRAINABLE_PARAM_REF
                || node.OpCode == InternalOpCodes.TRAINABLE_PARAM_MODEL_REF
                || node.OpCode == InternalOpCodes.TRAINABLE_PARAM_ID_REF)
                return false;
        }
        return true;
    }

    private static void ProcessGraphForTrainingOnFast(FastComputationGraph fastGraph)
    {
        Nodes.Processors.Fast.FastApplyIdentifierTemplates.Process(fastGraph);
        Nodes.Processors.Fast.FastInlineModulesAndFunctions.Process(fastGraph);
        Nodes.Processors.Fast.FastProcessorHelper.RemoveUnreachableNodes(fastGraph);
        Nodes.Processors.Fast.FastConvertToIdRefTrainableParams.Process(fastGraph);
        Nodes.Processors.Fast.FastUnpackModelStruct.Process(fastGraph);
        Nodes.Processors.Fast.FastUnpackTensorStructs.Process(fastGraph);

        // We intentionally skip ConvertTrainableParamIdRefToTrainableParam (requires execution)
        // and Simplify (can't handle TRAINABLE_PARAM_ID_REF nodes).
        // FastReplaceTrainableParamsWithInputProcessor handles TRAINABLE_PARAM_ID_REF directly.
    }

    /// <summary>
    /// Reads the dtype, rank and (optional) friendly name for the i-th input of a Fast graph
    /// directly from the producing MODEL_TENSOR_INPUT node.
    /// </summary>
    private static (DType type, int? rank, string? name) ResolveFastInputDef(FastComputationGraph graph, int index)
    {
        var inputKey = graph.Inputs[index];
        FastNode? producer = null;
        foreach (var n in graph.Nodes)
        {
            foreach (var (_, outs) in n.FullOutputs)
            {
                foreach (var ok in outs)
                {
                    if (ok is FastTensorKey k && k == inputKey)
                    {
                        producer = n;
                        break;
                    }
                }
                if (producer is not null) break;
            }
            if (producer is not null) break;
        }
        if (producer is null)
            throw new InvalidOperationException(
                $"PrepareForTrainingAsFast: input #{index} ({inputKey}) has no producer.");

        var dtype = producer.Attributes.GetDTypeVal(OnnxOpAttributeNames.AttrDtype)
            ?? throw new InvalidOperationException(
                $"PrepareForTrainingAsFast: input #{index} producer {producer.OpCode} has no AttrDtype.");
        var rank = (int?)producer.Attributes.GetLongVal(OnnxOpAttributeNames.ShrkAttrRank);
        string? name = index < graph.InputUniqueNames.Count ? graph.InputUniqueNames[index] : null;
        return (dtype, rank, name);
    }

    /// <summary>
    /// Reads a module's source-generated <c>ComputationGraph</c> static property reflectively.
    /// </summary>
    internal static FastComputationGraph ExtractFastGraphFromDelegate(Delegate func)
    {
        if (func is null) throw new ArgumentNullException(nameof(func));

        var method = func.Method;
        var declaringType = method.DeclaringType;

        if (declaringType is null)
            throw new ArgumentException(
                "The provided delegate does not have a declaring type. " +
                "It must reference the Inline method of a [Module]-annotated class.",
                nameof(func));

        if (method.Name != "Inline")
            throw new ArgumentException(
                $"The provided delegate references method '{method.Name}' on type '{declaringType.Name}'. " +
                "It must reference the 'Inline' method of a [Module]-annotated class.",
                nameof(func));

        var graphProperty = declaringType.GetProperty(
            "ComputationGraph",
            BindingFlags.Public | BindingFlags.Static | BindingFlags.FlattenHierarchy);

        if (graphProperty is null)
            throw new ArgumentException(
                $"Type '{declaringType.Name}' does not have a public static 'ComputationGraph' property. " +
                "Ensure it is a [Module]-annotated class with source-generated code.",
                nameof(func));

        var rawGraph = graphProperty.GetValue(null);
        if (rawGraph is null)
            throw new ArgumentException(
                $"The 'ComputationGraph' property on '{declaringType.Name}' returned null.",
                nameof(func));
        if (rawGraph is not FastComputationGraph fast)
            throw new ArgumentException(
                $"The 'ComputationGraph' property on '{declaringType.Name}' returned a value of type " +
                $"'{rawGraph.GetType().Name}', expected FastComputationGraph.",
                nameof(func));
        return fast;
    }
}
