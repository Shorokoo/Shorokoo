using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Graph;
using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Nodes.Processors.AutoGrad;
using Shorokoo.Core.Nodes.Processors.Fast;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using Shorokoo.Core.Utils;
using Shorokoo.Onnx;
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Reflection;

namespace Shorokoo.Core.Nodes.Processors.AutoGrad
{
    /// <summary>
    /// Fast-native autograd lowering. For each <c>AUTO_GRAD</c> node in the graph, walks the
    /// forward subgraph from <c>loss</c> back to the AUTO_GRAD parameter inputs in reverse
    /// topological order, calling each forward node's <c>[AutoDiff]</c> gradient method in
    /// Variable land with fresh stand-in inputs (one per slot per call), then splices the
    /// resulting gradient Variable subgraph back into the host Fast graph and rewires
    /// consumers of the AUTO_GRAD outputs to point at the new gradient keys.
    ///
    /// <para>
    /// No <c>[AutoDiff]</c> method needs to change. Built on the assumption that:
    /// <list type="bullet">
    ///   <item>Loss dtype is float32 (hardcoded; revisit when needed).</item>
    ///   <item>Per-tensor dtype is stripped during Fast conversion, so dtype mismatches
    ///         between the all-float32 Variable gradient subgraph and the host Fast graph
    ///         wash out at the splice boundary.</item>
    ///   <item>Functions are pre-inlined. <c>IF</c> pairs are differentiable via the
    ///         <c>IF_CLOSE</c> gradient; dynamic <c>LOOP</c> control flow has no backward
    ///         pass — when a loss→param path runs through one, lowering throws
    ///         <c>AutoDiffNotSupportedException</c> (AD003) rather than silently emitting
    ///         a zeros gradient. The same applies to any other unregistered op with a
    ///         parameter in its ancestry; unregistered ops with no parameter behind them
    ///         (Constant/Random subgraphs) are legitimate gradient leaves and are cut.</item>
    /// </list>
    /// </para>
    /// </summary>
    internal static class FastProcessAutoGradProcessor
    {
        /// <summary>
        /// Per-op MethodInfo for the <c>[AutoDiff]</c>-annotated gradient methods. Used to
        /// inspect each input slot's parameter type so we can pick a matching dtype for the
        /// fresh stand-in <c>Tensor&lt;T&gt;</c> we feed in (e.g. <c>Tensor&lt;int64&gt;</c>
        /// for ConstantOfShape's <c>shape</c> parameter rather than the default float32).
        /// </summary>
        private static readonly Dictionary<string, MethodInfo> gradientMethodInfos = BuildGradientMethodInfos();

        /// <summary>
        /// Maps every C# IVarType class (e.g. <c>typeof(int64)</c>) back to its <see cref="DType"/>
        /// so we can pick concrete-typed placeholders (e.g. <c>Tensor&lt;int64&gt;</c>) when a
        /// gradient method's parameter is non-generic.
        /// </summary>
        private static readonly Dictionary<Type, DType> dtypeByIVarType = BuildDTypeByIVarType();

        /// <summary>
        /// Lowers every <c>AUTO_GRAD</c> node in <paramref name="graph"/> in place. Removes
        /// the AUTO_GRAD nodes, runs <see cref="FastProcessorHelper.RemoveUnreachableNodes"/>
        /// and <see cref="FastProcessorHelper.EnsureTopologicalOrder"/>, then returns.
        /// </summary>
        public static void Process(FastComputationGraph graph)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));

            var autoGradNodes = graph.Nodes
                .Where(n => n.OpCode == InternalOpCodes.AUTO_GRAD)
                .ToList();
            if (autoGradNodes.Count == 0) return;

            foreach (var autoGradNode in autoGradNodes)
                ProcessOne(graph, autoGradNode);

            graph.Nodes.RemoveAll(n => n.OpCode == InternalOpCodes.AUTO_GRAD);
            FastProcessorHelper.RemoveUnreachableNodes(graph);

            System.Diagnostics.Debug.Assert(graph.IsLinearOrderValid(), "graph.IsLinearOrderValid()");
        }

        private static void ProcessOne(FastComputationGraph graph, FastNode autoGradNode)
        {
            // AUTO_GRAD inputs: [loss, ...params]; outputs: [grad_param_0, grad_param_1, ...]
            var allInputs = autoGradNode.Inputs;
            if (allInputs.Count < 1 || allInputs[0] is not FastTensorKey lossKey)
                throw new InvalidOperationException("FastProcessAutoGrad: AUTO_GRAD node missing loss input.");

            var paramKeys = new FastTensorKey[allInputs.Count - 1];
            for (int i = 0; i < paramKeys.Length; i++)
            {
                if (allInputs[i + 1] is not FastTensorKey pk)
                    throw new InvalidOperationException(
                        $"FastProcessAutoGrad: AUTO_GRAD param input #{i} is null.");
                paramKeys[i] = pk;
            }
            var paramKeySet = new HashSet<FastTensorKey>(paramKeys);
            var autoGradOutputs = autoGradNode.Outputs.Select(k => (FastTensorKey)k!).ToArray();

            // We splice the new gradient FastNodes into graph.Nodes at the position of the
            // AUTO_GRAD node so that the topological invariant is preserved end-to-end:
            // every input dep of a gradient node (model inputs, loss, params) sits at an
            // earlier index, and every consumer of the AUTO_GRAD outputs (e.g. optimizer
            // weight-update ops) sits at a later index. Appending to the end would put
            // gradient nodes after their consumers and break topological order.
            var autoGradIdx = graph.Nodes.IndexOf(autoGradNode);
            if (autoGradIdx < 0)
                throw new InvalidOperationException(
                    "FastProcessAutoGrad: AUTO_GRAD node not found in graph.Nodes.");

            // 1. Find the forward subgraph between loss and params, in topological order
            //    (leaves → loss). We then iterate it in reverse for gradient accumulation.
            var producerByOutput = BuildProducerByOutputMap(graph);
            if (!producerByOutput.TryGetValue(lossKey, out var lossProducer))
                throw new InvalidOperationException(
                    $"FastProcessAutoGrad: loss tensor {lossKey} has no producer.");

            var gradOpsMap = AutoDiffs.GetGradientOps();
            var paramDependent = ComputeParamDependentNodes(graph, paramKeySet, producerByOutput);
            var topoOrder = ComputeForwardTopoOrder(
                graph, lossProducer, paramKeySet, producerByOutput, gradOpsMap, paramDependent);

            // Lookup of FastNode by key — needed by ProcessNode to find IF_CLOSE's paired
            // open node when prepending the condition tensor for the gradient call.
            var nodesByKey = graph.Nodes.ToDictionary(n => n.Key);

            // Per-tensor metadata lookup. Used by ProcessNode to seed each fresh stand-in
            // input variable's Rank from the original host-graph tensor, so gradient ops
            // that branch on rank (e.g. Gather's non-zero-axis path) see the real value
            // rather than null.
            var tensorInfo = FastTensorInfoProcessor.BuildTensorInfoLookup(graph);

            // 2. Walk in reverse topo order, calling per-node gradient methods.
            var gradByKey = new Dictionary<FastTensorKey, Variable>();
            var freshInputBacking = new Dictionary<Variable, FastTensorKey>(ReferenceEqualityComparer.Instance);

            // Initialize loss gradient: a Scalar<float32> = 1.0. We hardcode float32 here;
            // see class doc for the rationale.
            gradByKey[lossKey] = (Variable)Globals.Scalar(1.0f);

            for (int i = topoOrder.Count - 1; i >= 0; i--)
            {
                ProcessNode(topoOrder[i], gradByKey, freshInputBacking, gradOpsMap, nodesByKey, tensorInfo);
            }

            // 3. Splice gradient Variable subgraphs into a temporary list. We maintain a
            //    single Variable→FastTensorKey map across all splices so that shared
            //    sub-expressions between different params' gradients only emit one FastNode
            //    (otherwise two splices would produce duplicate FastNodeKeys via FromCgKey
            //    for the same Variable Node).
            // 3. Lower the gradient Variable subgraph via the standard FastComputationGraph
            //    constructor, with externalInputKeys mapping each fresh stand-in Variable to
            //    the host-graph FastTensorKey it represents. The constructor sorts by
            //    OrderingHintNumber, drops the stand-in MODEL_TENSOR_INPUT nodes, and emits
            //    body FastNodes that already reference host keys directly.
            var standIns = freshInputBacking.Keys.ToImmutableArray();
            var gradHeads = paramKeys
                .Select(pk => gradByKey.TryGetValue(pk, out var gh) ? gh : null)
                .NotNulls()
                .ToImmutableArray();

            var newNodes = new List<FastNode>();
            FastComputationGraph? fastGradGraph = null;
            if (!gradHeads.IsEmpty)
            {
                fastGradGraph = new FastComputationGraph(
                    standIns, gradHeads, externalInputKeys: freshInputBacking);
                newNodes.AddRange(fastGradGraph.Nodes);
            }

            // Build per-param output mapping. Params with no gradient path get a
            // shape-matching zero via Sub(p, p), like the CG processor.
            var keyMappings = new Dictionary<FastTensorKey, FastTensorKey>();
            int gradHeadIdx = 0;
            for (int i = 0; i < paramKeys.Length; i++)
            {
                FastTensorKey gradKey;
                if (gradByKey.ContainsKey(paramKeys[i]))
                {
                    gradKey = fastGradGraph!.Outputs[gradHeadIdx++];
                }
                else
                {
                    gradKey = SpliceZerosLike(newNodes, paramKeys[i]);
                }
                keyMappings[autoGradOutputs[i]] = gradKey;
            }

            // 4. Insert the spliced gradient FastNodes at the AUTO_GRAD node's position so
            //    they sit before any AUTO_GRAD-output consumer in graph.Nodes.
            graph.Nodes.InsertRange(autoGradIdx, newNodes);

            // 5. Rewire consumers of AUTO_GRAD outputs to point at the new gradient keys.
            RewireConsumers(graph, keyMappings);
        }

        // ------------------------------------------------------------------------------------
        // Walk: forward topological order from leaves (paramKeys) up to the loss producer.
        // Graph leaves (MODEL_TENSOR_INPUT, CONSTANT, etc.) and any param-leaf are pruned:
        // their gradient (if any) flows into gradByKey via accumulation from their consumers,
        // never via direct invocation.
        //
        // Producers WITHOUT a registered [AutoDiff] gradient are handled by param-reachability:
        //   - If no AUTO_GRAD parameter lies anywhere in the producer's input ancestry
        //     (paramDependent), it is a legitimate gradient leaf (Constant / RandomNormal /
        //     a pure-constant subgraph) and the chain is cut silently — the correct behavior.
        //   - If a parameter IS reachable behind it, the node stays in the topo order so that
        //     ProcessNode throws AD003 when a real gradient flows into it. Cutting silently
        //     there would freeze the parameter at zero gradient — the worst failure mode for
        //     a training framework. (ProcessNode still skips it when all its output grads are
        //     null, i.e. it only feeds the loss through non-differentiable side paths such as
        //     Shape.)
        //
        // graph.Nodes is already topologically ordered, so we don't traverse the DAG: a single
        // reverse-order pass propagates the "reachable from loss along gradient edges" mark
        // from each visited node back to its inputs' producers (which sit at lower indices and
        // will be visited later in the pass). Then we collect the marked nodes in graph.Nodes
        // order, which the caller iterates in reverse for gradient accumulation.
        // ------------------------------------------------------------------------------------

        private static List<FastNode> ComputeForwardTopoOrder(
            FastComputationGraph graph,
            FastNode lossProducer,
            HashSet<FastTensorKey> paramKeySet,
            Dictionary<FastTensorKey, FastNode> producerByOutput,
            Dictionary<string, Func<Variable?[], Variable?[], OnnxCSharpAttributes, Variable?[]>> gradOpsMap,
            HashSet<FastNodeKey> paramDependent)
        {
            var reachable = new HashSet<FastNodeKey>();
            if (gradOpsMap.ContainsKey(lossProducer.OpCode) || paramDependent.Contains(lossProducer.Key))
                reachable.Add(lossProducer.Key);

            for (int i = graph.Nodes.Count - 1; i >= 0; i--)
            {
                var node = graph.Nodes[i];
                if (!reachable.Contains(node.Key)) continue;

                foreach (var (_, slots) in node.FullInputs)
                {
                    foreach (var slot in slots)
                    {
                        if (slot is not FastTensorKey k) continue;
                        if (paramKeySet.Contains(k)) continue;
                        if (!producerByOutput.TryGetValue(k, out var producer)) continue;
                        if (!gradOpsMap.ContainsKey(producer.OpCode)
                            && !paramDependent.Contains(producer.Key)) continue;
                        reachable.Add(producer.Key);
                    }
                }
            }

            return graph.Nodes.Where(n => reachable.Contains(n.Key)).ToList();
        }

        // ------------------------------------------------------------------------------------
        // Param reachability: marks every node whose input ancestry contains one of the
        // AUTO_GRAD parameter tensors. A single forward pass over the (topologically ordered)
        // node list suffices: a node is param-dependent iff one of its input slots is a param
        // key or is produced by an already-marked node.
        // ------------------------------------------------------------------------------------

        private static HashSet<FastNodeKey> ComputeParamDependentNodes(
            FastComputationGraph graph,
            HashSet<FastTensorKey> paramKeySet,
            Dictionary<FastTensorKey, FastNode> producerByOutput)
        {
            var paramDependent = new HashSet<FastNodeKey>();
            foreach (var node in graph.Nodes)
            {
                foreach (var (_, slots) in node.FullInputs)
                {
                    foreach (var slot in slots)
                    {
                        if (slot is not FastTensorKey k) continue;
                        if (paramKeySet.Contains(k)
                            || (producerByOutput.TryGetValue(k, out var producer)
                                && paramDependent.Contains(producer.Key)))
                        {
                            paramDependent.Add(node.Key);
                            goto nextNode;
                        }
                    }
                }
            nextNode: ;
            }
            return paramDependent;
        }

        // ------------------------------------------------------------------------------------
        // Per-node gradient computation: invoke the [AutoDiff] gradient method with fresh
        // Tensor<float32> stand-in inputs and the already-accumulated output gradients.
        // ------------------------------------------------------------------------------------

        private static void ProcessNode(
            FastNode node,
            Dictionary<FastTensorKey, Variable> gradByKey,
            Dictionary<Variable, FastTensorKey> freshInputBacking,
            Dictionary<string, Func<Variable?[], Variable?[], OnnxCSharpAttributes, Variable?[]>> gradOpsMap,
            Dictionary<FastNodeKey, FastNode> nodesByKey,
            Dictionary<FastTensorKey, FastTensorInfo> tensorInfo)
        {
            var outputs = node.Outputs;
            var outputGrads = new Variable?[outputs.Count];
            for (int i = 0; i < outputs.Count; i++)
            {
                if (outputs[i] is FastTensorKey k && gradByKey.TryGetValue(k, out var g))
                    outputGrads[i] = g;
            }

            // Skip nodes whose outputs all have no gradient (dead branch of multi-output op).
            if (outputGrads.All(g => g is null)) return;

            // A gradient flows into this node and an AUTO_GRAD parameter sits behind it
            // (ComputeForwardTopoOrder only admits unregistered nodes when a param is
            // reachable through them). Cutting silently here would freeze that parameter
            // with a zeros gradient, so fail loudly instead.
            if (!gradOpsMap.TryGetValue(node.OpCode, out var gradOp))
            {
                var isLoopOp = node.OpCode is OpCodes.LOOP_OPEN or OpCodes.LOOP_CLOSE
                    or OpCodes.LOOP_FAKE_INPUT or OpCodes.LOOP_SCAN_VARIABLE
                    or OpCodes.LOOP_INDEX_VARIABLE;
                throw new AutoDiffNotSupportedException(ErrorCodes.AD003, node.OpCode,
                    isLoopOp
                        ? "op lies on the differentiation path between the loss and a trainable "
                          + "parameter, but autodiff has no backward pass for dynamic loops — see "
                          + "Documentation/limitations.md. Unroll the loop statically (constant trip count) "
                          + "or keep trainable parameters out of dynamic-loop bodies."
                        : "op lies on the differentiation path between the loss and a trainable "
                          + "parameter, but has no registered gradient. Training through it would "
                          + "silently freeze the parameter, so this is an error. Register an "
                          + "[AutoDiff] gradient for the op or detach it from the loss path.");
            }

            // Build fresh stand-in IValues for each input slot. We pick the dtype from the
            // gradient method's parameter type when it's a concrete Variable subtype (e.g.
            // Tensor<int64> for shape/axes/indices); generic Tensor<T> parameters default to
            // float32, in line with the loss-grad-is-float32 assumption.
            //
            // For IF_CLOSE, the gradient (IfCloseGradient) expects inputs[0] = condition from
            // the paired IF_OPEN node. The Fast IF_CLOSE node carries only the branch tensors
            // in its Inputs (else_*, then_*), so we look up the paired open node and prepend
            // its condition input here, mirroring AutoDiffEngine's CG-side behavior.
            var inputs = node.Inputs;
            if (node.OpCode == OpCodes.IF_CLOSE
                && node.GraphOpenNodeKey is FastNodeKey openKey
                && nodesByKey.TryGetValue(openKey, out var openNode)
                && openNode.Inputs.Count > 0)
            {
                var withCond = new List<FastTensorKey?>(1 + inputs.Count) { openNode.Inputs[0] };
                withCond.AddRange(inputs);
                inputs = withCond;
            }

            // For DROPOUT with a wired training_mode input, the gradient needs the forward
            // mask OUTPUT (the dict-gradient signature only carries the node's inputs).
            // Append the mask's host key as a trailing extra stand-in slot — DropoutGradient
            // reads it as inputs[3] and builds a gradient that is correct for both runtime
            // values of training_mode. Without this slot, DropoutGradient throws AD003
            // rather than silently passing dy through a possibly-training-mode Dropout.
            if (node.OpCode == OpCodes.DROPOUT
                && node.Inputs.Count >= 3 && node.Inputs[2] is not null
                && node.Outputs.Count >= 2 && node.Outputs[1] is FastTensorKey)
            {
                var withMask = new List<FastTensorKey?>(inputs.Count + 1);
                withMask.AddRange(inputs);
                withMask.Add((FastTensorKey)node.Outputs[1]!);
                inputs = withMask;
            }
            var inputIValues = new Variable?[inputs.Count];
            gradientMethodInfos.TryGetValue(node.OpCode, out var methodInfo);
            var methodParams = methodInfo?.GetParameters();
            for (int i = 0; i < inputs.Count; i++)
            {
                if (inputs[i] is not FastTensorKey k)
                {
                    inputIValues[i] = null;
                    continue;
                }
                var (slotDtype, slotRank) = ResolveSlotDtype(methodParams, i);
                // Fall back to the host-graph tensor's actual Rank when the gradient method's
                // parameter type (typically Tensor<T>) doesn't pin one down — preserves rank
                // through the stand-in so rank-branching gradients see the real value.
                if (slotRank is null && tensorInfo.TryGetValue(k, out var info))
                    slotRank = info.Rank;
                var fresh = InternalOp.RuntimeInput(slotDtype, rank: slotRank);
                inputIValues[i] = fresh;
                freshInputBacking[fresh] = k;
            }

            Variable?[] inputGrads;
            try
            {
                inputGrads = gradOp(inputIValues, outputGrads, node.Attributes);
            }
            catch (AutoDiffNotSupportedException)
            {
                // Deliberate "this attribute combination has no gradient" guards (AD003)
                // must surface as-is so callers can catch the typed exception.
                throw;
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException(
                    $"FastProcessAutoGrad: gradient op for '{node.OpCode}' (Key={node.Key}) threw: {ex.Message}", ex);
            }

            // Accumulate into gradByKey, using AutoDiffEngine.AccumulateGradients on collisions.
            for (int i = 0; i < inputs.Count && i < inputGrads.Length; i++)
            {
                if (inputs[i] is not FastTensorKey k) continue;
                if (inputGrads[i] is null) continue;

                // Gradient ops already return graph nodes.
                var grad = inputGrads[i]!;

                if (gradByKey.TryGetValue(k, out var existing))
                    gradByKey[k] = AutoDiffEngine.AccumulateGradients(existing, grad);
                else
                    gradByKey[k] = grad;
            }
        }

        // ------------------------------------------------------------------------------------
        // Zero gradient fallback: emit Sub(p, p) for params with no gradient path from loss.
        // ------------------------------------------------------------------------------------

        private static FastTensorKey SpliceZerosLike(List<FastNode> newNodes, FastTensorKey paramKey)
        {
            // We can build a Sub(p, p) FastNode directly without going through Variable land,
            // since both inputs share the same FastTensorKey.
            var fastNode = new FastNode
            {
                Key = FastNodeKey.New(),
                OpCode = OpCodes.SUB,
                Attributes = OnnxCSharpAttributes.FromCSharpVals(
                    new Dictionary<string, object?>(),
                    Definitions.NodeDefinitions[OpCodes.SUB].AttributeDefs),
            };
            fastNode.FullInputs[""] = new List<FastTensorKey?> { paramKey, paramKey };
            fastNode.FullOutputs[""] = new List<FastTensorKey?> { new FastTensorKey(fastNode.Key, 0) };
            newNodes.Add(fastNode);
            return new FastTensorKey(fastNode.Key, 0);
        }

        // ------------------------------------------------------------------------------------
        // Rewire consumers of the AUTO_GRAD outputs (and graph.Outputs) to the new keys.
        // ------------------------------------------------------------------------------------

        private static void RewireConsumers(
            FastComputationGraph graph,
            Dictionary<FastTensorKey, FastTensorKey> keyMappings)
        {
            if (keyMappings.Count == 0) return;

            foreach (var node in graph.Nodes)
            {
                foreach (var (_, slots) in node.FullInputs)
                {
                    for (int i = 0; i < slots.Count; i++)
                    {
                        if (slots[i] is FastTensorKey k && keyMappings.TryGetValue(k, out var mapped))
                            slots[i] = mapped;
                    }
                }
            }

            for (int i = 0; i < graph.Outputs.Count; i++)
            {
                if (keyMappings.TryGetValue(graph.Outputs[i], out var mapped))
                    graph.Outputs[i] = mapped;
            }
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

        // ------------------------------------------------------------------------------------
        // Type / metadata setup. Built once at first use.
        // ------------------------------------------------------------------------------------

        private static Dictionary<string, MethodInfo> BuildGradientMethodInfos()
        {
            var map = new Dictionary<string, MethodInfo>();
            var methods = typeof(AutoDiffs).GetMethods(BindingFlags.Public | BindingFlags.Static)
                .Where(m => m.GetCustomAttribute<AutoDiffAttribute>() != null);
            foreach (var m in methods)
            {
                var attr = m.GetCustomAttribute<AutoDiffAttribute>()!;
                map[attr.OpName] = m;
            }
            return map;
        }

        private static Dictionary<Type, DType> BuildDTypeByIVarType()
        {
            // Mirror DType's known concrete dtypes back to their IVarType class. The set must
            // cover anything an [AutoDiff] method might declare as a concrete generic argument.
            var dtypes = new[]
            {
                DType.BFloat16, DType.Float16, DType.Float32, DType.Float64,
                DType.Int8, DType.Int16, DType.Int32, DType.Int64,
                DType.UInt8, DType.UInt16, DType.UInt32, DType.UInt64,
                DType.Bool,
            };
            var map = new Dictionary<Type, DType>();
            foreach (var d in dtypes)
            {
                try { map[d.ToIVarType()] = d; }
                catch { /* skip unsupported dtypes */ }
            }
            return map;
        }

        /// <summary>
        /// Returns the dtype and rank to use for the stand-in Variable at slot
        /// <paramref name="slotIdx"/>. When the gradient method's parameter at that index is a
        /// concrete <c>Scalar&lt;TConcrete&gt;</c>, <c>Vector&lt;TConcrete&gt;</c>, or
        /// <c>Tensor&lt;TConcrete&gt;</c>, returns <c>TConcrete</c>'s dtype and the
        /// matching rank (0 for Scalar, 1 for Vector, null for Tensor) so
        /// <see cref="Node.MakeOutput"/> picks the same concrete Variable subtype the gradient
        /// method declares — otherwise the eventual reflection invoke fails with a cast error.
        /// Defaults to (<see cref="DType.Float32"/>, null) when the parameter is generic or
        /// untyped.
        /// </summary>
        private static (DType Dtype, int? Rank) ResolveSlotDtype(ParameterInfo[]? methodParams, int slotIdx)
        {
            if (methodParams is null || slotIdx >= methodParams.Length) return (DType.Float32, null);

            // A value-struct handle parameter may be declared nullable (`Tensor<T>?` == Nullable<Tensor<T>>);
            // unwrap it to recover the underlying handle type.
            var paramType = Nullable.GetUnderlyingType(methodParams[slotIdx].ParameterType) ?? methodParams[slotIdx].ParameterType;
            if (paramType.ContainsGenericParameters) return (DType.Float32, null);

            if (paramType.IsGenericType)
            {
                var args = paramType.GetGenericArguments();
                if (args.Length >= 1 && dtypeByIVarType.TryGetValue(args[0], out var dt))
                {
                    var def = paramType.GetGenericTypeDefinition();
                    int? rank = def == typeof(Scalar<>) ? 0
                              : def == typeof(Vector<>) ? 1
                              : (int?)null;
                    return (dt, rank);
                }
            }
            return (DType.Float32, null);
        }
    }
}
