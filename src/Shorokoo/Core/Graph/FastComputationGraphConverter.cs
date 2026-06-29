using Shorokoo.Graph;
using Shorokoo.Core.Graph;
using Shorokoo.Core;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using Shorokoo.Core.Utils;
using Shorokoo.Onnx;
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text.RegularExpressions;
using Shorokoo.Core.Nodes.Processors.Helpers;

namespace Shorokoo.Graph
{
    /// <summary>
    /// FastCG-only helpers: build Variable / Node views of a
    /// <see cref="FastComputationGraph"/>, walk it for Functions, and populate a fresh
    /// FastCG from a topologically-ordered Node sequence (used by the FastCG
    /// constructor). All operations stay within the FastCG / Node / Variable triangle —
    /// no <c>ComputationGraph</c> wrapper is materialized.
    /// </summary>
    public static partial class FastComputationGraphConverter
    {
        /// <summary>
        /// Matches names produced by <c>FastUseUniqueNames</c>: a literal "N" followed
        /// by a non-negative decimal integer. Used by the <c>useSequentialIds</c> path
        /// of <see cref="PopulateFromNodes"/> to lift the integer id out of
        /// <see cref="Node.FriendlyName"/> into the resulting <see cref="FastNodeKey"/>.
        /// </summary>
        private static readonly Regex SequentialNamePattern = new Regex(@"^N(\d+)$", RegexOptions.Compiled);

        /// <summary>
        /// Populates an empty <paramref name="fastGraph"/> by lowering the provided
        /// topologically-ordered <see cref="Node"/> sequence to <see cref="FastNode"/>s
        /// and wiring up <paramref name="inputs"/> / <paramref name="outputs"/>. Used by
        /// the <see cref="FastComputationGraph(System.Collections.Immutable.ImmutableArray{Variable}, System.Collections.Immutable.ImmutableArray{Variable}, System.Collections.Immutable.ImmutableArray{int?}?, IReadOnlyDictionary{Variable, FastTensorKey}?)"/>
        /// constructor.
        /// </summary>
        internal static void PopulateFromNodes(
            FastComputationGraph fastGraph,
            IEnumerable<Node> topologicalOrderNodes,
            IEnumerable<Variable> inputs,
            IEnumerable<Variable> outputs,
            int?[] outputRankOverrides,
            bool useSequentialIds,
            IReadOnlyDictionary<Variable, FastTensorKey>? externalInputKeys = null)
        {
            fastGraph.OutputRankOverrides = outputRankOverrides;

            // When the source node sequence contains duplicate NodeKeys (which happens when
            // the same cached inner function is inlined multiple times), we must assign fresh
            // FastNodeKeys to the duplicates. We track the mapping from Variable objects
            // (unique by reference) to the correct FastTensorKey so that input wiring
            // follows the right producer even when keys would otherwise collide.
            var usedNodeKeys = new HashSet<NodeKey>();
            var usedFastIds = useSequentialIds ? new HashSet<UInt128>() : null;
            var variableToKey = new Dictionary<Variable, FastTensorKey>(ReferenceEqualityComparer.Instance);
            // For GraphOpenNodeKey: map from Node object to the assigned FastNodeKey.
            var nodeToAssignedKey = new Dictionary<Node, FastNodeKey>(ReferenceEqualityComparer.Instance);

            // Pre-seed variableToKey with externally-supplied mappings for stand-in
            // input IValues. Their owning Nodes are skipped below so the host-graph
            // FastTensorKey takes their place wherever they're referenced.
            if (externalInputKeys is not null)
                foreach (var (iv, key) in externalInputKeys)
                    variableToKey[iv] = key;

            foreach (var node in topologicalOrderNodes)
            {
                if (externalInputKeys is not null && node.Outputs.Length == 1 &&
                    node.Outputs[0] is Variable outIv && externalInputKeys.ContainsKey(outIv))
                {
                    // Stand-in input — host key already in variableToKey; don't emit a FastNode.
                    continue;
                }
                bool isDuplicate = !usedNodeKeys.Add(node.Key);

                FastNodeKey assignedKey;
                if (useSequentialIds)
                {
                    assignedKey = ParseSequentialKey(node);
                    if (!usedFastIds!.Add(assignedKey.Id))
                        throw new InvalidOperationException(
                            $"FastComputationGraphConverter: useSequentialIds requires unique N{{i}} names but '{node.FriendlyName}' is reused.");
                }
                else
                {
                    assignedKey = isDuplicate ? FastNodeKey.New() : FastNodeKey.FromCgKey(node.Key);
                }
                nodeToAssignedKey[node] = assignedKey;

                // Resolve GraphOpenNodeKey using the already-assigned key of the open node.
                FastNodeKey? graphOpenNodeKey = null;
                if (node.GraphOpenNode is not null)
                {
                    if (nodeToAssignedKey.TryGetValue(node.GraphOpenNode, out var openAssigned))
                        graphOpenNodeKey = openAssigned;
                    else if (useSequentialIds)
                        graphOpenNodeKey = ParseSequentialKey(node.GraphOpenNode);
                    else
                        graphOpenNodeKey = FastNodeKey.FromCgKey(node.GraphOpenNode.Key);
                }

                var fastNode = new FastNode
                {
                    Key = assignedKey,
                    OpCode = node.OpCode,
                    Attributes = node.Attributes,
                    FriendlyName = node.FriendlyName,
                    StackTrace = node.StackTrace,
                    GraphOpenNodeKey = graphOpenNodeKey,
                    IdentifierTemplate = node.IdentifierTemplate?.ToString(),
                    TargetFunction = node.TargetFunction,
                };

                foreach (var kvp in node.FullInputs)
                {
                    var slot = new List<FastTensorKey?>(kvp.Value.Length);
                    foreach (var input in kvp.Value)
                    {
                        if (input is null) { slot.Add(null); continue; }
                        if (variableToKey.TryGetValue(input, out var mappedKey))
                            slot.Add(mappedKey);
                        else
                            slot.Add(FastTensorKey.FromCgKey(input.Key));
                    }
                    fastNode.FullInputs[kvp.Key] = slot;
                }

                foreach (var kvp in node.FullOutputs)
                {
                    var slot = new List<FastTensorKey?>(kvp.Value.Length);
                    foreach (var output in kvp.Value)
                    {
                        if (output is null) { slot.Add(null); continue; }

                        FastTensorKey outputKey;
                        if (useSequentialIds)
                        {
                            // Under sequential ids the producing node's FastNodeKey already
                            // reflects the desired integer id; outputs always belong to that node.
                            outputKey = new FastTensorKey(assignedKey, output.Key.OutputIndex);
                        }
                        else if (isDuplicate && output.Key.NodeKey == node.Key)
                        {
                            outputKey = new FastTensorKey(assignedKey, output.Key.OutputIndex);
                        }
                        else if (isDuplicate)
                        {
                            // Cross-node reference (e.g. LOOP_OPEN carry variable).
                            // If the referenced node was also duplicated, remap.
                            outputKey = FastTensorKey.FromCgKey(output.Key);
                        }
                        else
                        {
                            outputKey = FastTensorKey.FromCgKey(output.Key);
                        }

                        variableToKey[output] = outputKey;
                        slot.Add(outputKey);
                    }
                    fastNode.FullOutputs[kvp.Key] = slot;
                }

                fastGraph.Nodes.Add(fastNode);
            }

            foreach (var input in inputs)
            {
                FastTensorKey key = variableToKey.TryGetValue(input, out var mk) ? mk : FastTensorKey.FromCgKey(input.Key);
                fastGraph.Inputs.Add(key);
                fastGraph.InputUniqueNames.Add(input.UniqueName);
            }
            foreach (var output in outputs)
            {
                FastTensorKey key = variableToKey.TryGetValue(output, out var mk) ? mk : FastTensorKey.FromCgKey(output.Key);
                fastGraph.Outputs.Add(key);
                fastGraph.OutputUniqueNames.Add(output.UniqueName);
            }
        }

        /// <summary>
        /// Pulls the integer suffix out of <paramref name="node"/>'s <see cref="Node.FriendlyName"/>
        /// (e.g. "N42" → 42). Throws if the name doesn't match — callers must run
        /// <c>FastUseUniqueNames</c> first.
        /// </summary>
        private static FastNodeKey ParseSequentialKey(Node node)
        {
            var name = node.FriendlyName;
            if (string.IsNullOrEmpty(name))
                throw new InvalidOperationException(
                    $"FastComputationGraphConverter: useSequentialIds requires every node to have an N{{i}} FriendlyName, but Node {node.Key} has none. Run UseUniqueNames before converting.");
            var m = SequentialNamePattern.Match(name);
            if (!m.Success)
                throw new InvalidOperationException(
                    $"FastComputationGraphConverter: useSequentialIds requires every node's FriendlyName to match ^N\\d+$, but Node {node.Key} has FriendlyName='{name}'. Run UseUniqueNames before converting.");
            return new FastNodeKey(UInt128.Parse(m.Groups[1].ValueSpan));
        }

        /// <summary>
        /// Builds the <see cref="FastTensorKey"/> → <see cref="Variable"/> mapping for
        /// <paramref name="fastGraph"/> by reconstructing the underlying
        /// <see cref="Node"/> / <see cref="Variable"/> objects. Use this when you only
        /// need per-tensor Variable metadata (dtype, structure, rank, unique name,
        /// owning module function).
        /// </summary>
        public static Dictionary<FastTensorKey, Variable> BuildTensorMapping(FastComputationGraph fastGraph)
        {
            if (fastGraph is null) throw new ArgumentNullException(nameof(fastGraph));
            return BuildNodesAndTensorMap(fastGraph).tensorsByKey;
        }

        /// <summary>
        /// Reconstructs the underlying <see cref="Node"/> / <see cref="Variable"/> objects
        /// for <paramref name="fastGraph"/> in topological (build) order, returning the
        /// rebuilt nodes alongside the inputs, outputs and FastTensorKey → Variable map.
        /// Use this when a caller wants Node/Variable views of the graph.
        /// </summary>
        public static (ImmutableArray<Node> nodesInTopoOrder,
                       ImmutableArray<Variable> inputs,
                       ImmutableArray<Variable> outputs,
                       Dictionary<FastTensorKey, Variable> tensorMapping)
            BuildNodes(FastComputationGraph fastGraph)
        {
            if (fastGraph is null) throw new ArgumentNullException(nameof(fastGraph));
            var built = BuildNodesAndTensorMap(fastGraph);
            return (built.nodesInTopoOrder, built.inputs, built.outputs, built.tensorsByKey);
        }

        /// <summary>
        /// Walks <paramref name="fastGraph"/> and returns every <see cref="Function"/> it
        /// reaches (directly via <see cref="FastNode.TargetFunction"/>, transitively via
        /// each function's <see cref="Function.OriginalFastGraph"/>) in dependency
        /// post-order: if function A calls B, B precedes A.
        /// </summary>
        public static ImmutableArray<Function> FunctionsPostOrder(FastComputationGraph fastGraph)
        {
            if (fastGraph is null) throw new ArgumentNullException(nameof(fastGraph));

            var fnDependencies = new Dictionary<Function, HashSet<Function>>();
            var toVisitQueue = new Queue<Function>(LocalFunctions(fastGraph));

            while (toVisitQueue.Count != 0)
            {
                var toVisit = toVisitQueue.Dequeue();
                if (fnDependencies.ContainsKey(toVisit))
                    continue;

                var newFns = LocalFunctions(toVisit.OriginalFastGraph);
                fnDependencies[toVisit] = newFns.ToHashSet();
                foreach (var fn in newFns)
                {
                    if (!fnDependencies.ContainsKey(fn))
                        toVisitQueue.Enqueue(fn);
                }
            }

            var retVal = new List<Function>();
            while (fnDependencies.Count != 0)
            {
                var toProcess = fnDependencies.Where(x => x.Value.Count == 0).Select(x => x.Key).ToList();
                if (toProcess.Count == 0)
                    throw new InvalidOperationException("Cyclic function dependencies detected in computation graph.");

                foreach (var fn in toProcess)
                {
                    retVal.Add(fn);
                    foreach (var fnVals in fnDependencies.Values)
                        fnVals.Remove(fn);
                    fnDependencies.Remove(fn);
                }
            }

            return retVal.ToImmutableArray();
        }

        /// <summary>
        /// Direct (non-transitive) Function references in <paramref name="fastGraph"/>:
        /// distinct <see cref="FastNode.TargetFunction"/> values in topological node order.
        /// </summary>
        public static ImmutableArray<Function> LocalFunctions(FastComputationGraph fastGraph)
        {
            if (fastGraph is null) throw new ArgumentNullException(nameof(fastGraph));
            return fastGraph.Nodes.Select(n => n.TargetFunction).NotNulls().Distinct().ToImmutableArray();
        }

        private static (ImmutableArray<Node> nodesInTopoOrder,
                        Dictionary<FastTensorKey, Variable> tensorsByKey,
                        ImmutableArray<Variable> inputs,
                        ImmutableArray<Variable> outputs)
            BuildNodesAndTensorMap(FastComputationGraph fastGraph)
        {
            // Map from the stored FastTensorKey to the freshly-created Variable we built while
            // rebuilding nodes in topological order.
            var tensorsByKey = new Dictionary<FastTensorKey, Variable>();
            var nodesByKey = new Dictionary<FastNodeKey, Node>();
            var nodesInOrder = new List<Node>(fastGraph.Nodes.Count);

            // Close nodes whose open node hasn't been built yet are deferred and resolved
            // after all nodes have been created. This handles graphs where the data-dependency
            // topological order places a close node before its open node (valid because
            // GraphOpenNodeKey is a structural cross-reference, not a data dependency).
            var deferredCloseNodes = new List<(Node closeNode, FastNodeKey openKey)>();

            foreach (var fastNode in fastGraph.Nodes)
            {
                var nodeDefResolver = Definitions.NodeDefinitions[fastNode.OpCode];
                var attributes = fastNode.Attributes
                                  ?? OnnxCSharpAttributes.FromCSharpVals(new Dictionary<string, object?>(), nodeDefResolver.AttributeDefs);
                var nodeDef = nodeDefResolver.Resolve(attributes.ToProto());

                var fullInputs = fastNode.FullInputs.ToImmutableDictionary(
                    kvp => kvp.Key,
                    kvp => kvp.Value.Select(k =>
                    {
                        if (k is null || k.Value.IsEmpty) return (Variable?)null;
                        if (!tensorsByKey.TryGetValue(k.Value, out var found))
                            throw new InvalidOperationException(
                                $"FastComputationGraphConverter: tensor {k.Value} referenced by node '{fastNode.OpCode}' (Key={fastNode.Key}) was not produced by any earlier node. " +
                                "Make sure FastComputationGraph.Nodes is in topological order.");
                        return (Variable?)found;
                    }).ToArray());

                Node? openNode = null;
                if (fastNode.GraphOpenNodeKey is FastNodeKey openKey && !openKey.IsEmpty)
                    nodesByKey.TryGetValue(openKey, out openNode);

                var dataProcessor = new InputNodeDataProcessor
                {
                    DefaultName = fastNode.FriendlyName,
                    NodeDef = nodeDef,
                    FullInputs = fullInputs,
                    ProtoAttributes = attributes.ToProto(),
                    StackTrace = fastNode.StackTrace,
                };

                ImmutableDictionary<string, int?>? knownVariadicCounts = null;
                if (openNode is not null)
                {
                    knownVariadicCounts = new InputNodeDataProcessor
                    {
                        DefaultName = openNode.DefaultName,
                        NodeDef = openNode.NodeDef,
                        FullInputs = openNode.FullInputs,
                        ProtoAttributes = openNode.Attributes.ToProto(),
                        StackTrace = openNode.StackTrace,
                    }.InferVariadicCounts(openNode.FullOutputs.Values.Select(x => x.Length).Max());
                }

                var primaryGroup = fastNode.FullOutputs.OrderBy(g => g.Key, StringComparer.Ordinal).FirstOrDefault();
                string?[]? outputNamesForInference = primaryGroup.Value is null
                    ? null
                    : primaryGroup.Value.Select(k =>
                        (k is null || k.Value.IsEmpty) ? string.Empty : (string?)null).ToArray();

                var inferredOutputs = dataProcessor.DeduceMultiOutputDefinitions(knownVariadicCounts, outputNamesForInference);

                var fullOutputs = fastNode.FullOutputs.ToImmutableDictionary(
                    kvp => kvp.Key,
                    kvp =>
                    {
                        if (!inferredOutputs.TryGetValue(kvp.Key, out var inferredSlots))
                            throw new InvalidOperationException(
                                $"FastComputationGraphConverter: type inference produced no outputs for group '{kvp.Key}' of node '{fastNode.OpCode}' (Key={fastNode.Key}).");

                        if (inferredSlots.Length != kvp.Value.Count)
                            throw new InvalidOperationException(
                                $"FastComputationGraphConverter: type inference produced {inferredSlots.Length} output(s) in group '{kvp.Key}' of node '{fastNode.OpCode}' (Key={fastNode.Key}) but the FastNode has {kvp.Value.Count}.");

                        return inferredSlots;
                    });

                var newNode = new Node(
                    nodeDef: nodeDef,
                    attributes: attributes,
                    inputs: fullInputs,
                    outputs: fullOutputs,
                    stackTrace: fastNode.StackTrace,
                    defaultName: fastNode.FriendlyName,
                    identifierTemplateString: fastNode.IdentifierTemplate,
                    targetFunction: fastNode.TargetFunction,
                    openNode: openNode,
                    existingKey: fastNode.Key.ToCgKey());

                // If this is a close node whose open node hasn't been built yet,
                // defer the GraphOpenNode / ConnectingTensor linkage.
                if (openNode is null && fastNode.GraphOpenNodeKey is FastNodeKey deferredOpenKey && !deferredOpenKey.IsEmpty)
                    deferredCloseNodes.Add((newNode, deferredOpenKey));

                nodesByKey[fastNode.Key] = newNode;
                nodesInOrder.Add(newNode);

                // Record each freshly-created output tensor against its FastTensorKey so that
                // subsequent nodes can wire up to it by key.
                foreach (var group in newNode.FullOutputs)
                {
                    foreach (var output in group.Value)
                    {
                        if (output is null) continue;
                        tensorsByKey[FastTensorKey.FromCgKey(output.Key)] = output;
                    }
                }
                if (newNode.ConnectingTensor is not null)
                    tensorsByKey[FastTensorKey.FromCgKey(newNode.ConnectingTensor.Key)] = newNode.ConnectingTensor;

                // Map original FastNode output FastTensorKeys to rebuilt outputs by position.
                var fastGroups = fastNode.FullOutputs.OrderBy(g => g.Key, StringComparer.Ordinal).ToList();
                var rebuiltGroups = newNode.FullOutputs.OrderBy(g => g.Key, StringComparer.Ordinal).ToList();
                for (int gi = 0; gi < fastGroups.Count && gi < rebuiltGroups.Count; gi++)
                {
                    var fastSlots = fastGroups[gi].Value;
                    var rebuiltSlots = rebuiltGroups[gi].Value;
                    for (int si = 0; si < fastSlots.Count && si < rebuiltSlots.Length; si++)
                    {
                        var origKey = fastSlots[si];
                        var rebuiltOutput = rebuiltSlots[si];
                        if (origKey is not null && !origKey.Value.IsEmpty && rebuiltOutput is not null)
                        {
                            var rebuiltAsFast = FastTensorKey.FromCgKey(rebuiltOutput.Key);
                            if (origKey.Value != rebuiltAsFast)
                                tensorsByKey[origKey.Value] = rebuiltOutput;
                        }
                    }
                }
            }

            // Resolve deferred close nodes: set GraphOpenNode and ConnectingTensor now that
            // all open nodes have been built.
            foreach (var (closeNode, openKey) in deferredCloseNodes)
            {
                if (!nodesByKey.TryGetValue(openKey, out var resolvedOpen))
                    throw new InvalidOperationException(
                        $"FastComputationGraphConverter: close node '{closeNode.OpCode}' (Key={closeNode.Key}) references open node {openKey} which was not found in the graph.");
                closeNode.GraphOpenNode = resolvedOpen;
                closeNode.ConnectingTensor = resolvedOpen.ConnectingTensor;
            }

            Variable LookupGraphTensor(FastTensorKey key, string role)
            {
                if (tensorsByKey.TryGetValue(key, out var v)) return v;
                throw new InvalidOperationException(
                    $"FastComputationGraphConverter: {role} tensor {key} was not produced by any node in the graph.");
            }

            var inputs = fastGraph.Inputs.Select(k => LookupGraphTensor(k, "input")).ToImmutableArray();
            var outputs = fastGraph.Outputs.Select(k => LookupGraphTensor(k, "output")).ToImmutableArray();

            // Restore original UniqueNames so that human-readable graph-input/output names
            // survive FastCG processor passes (loop unrolling, simplify, etc. otherwise
            // replace them with TensorKey.ToString()-style strings).
            ApplyOriginalNames(inputs, fastGraph.InputUniqueNames);
            ApplyOriginalNames(outputs, fastGraph.OutputUniqueNames);

            return (nodesInOrder.ToImmutableArray(), tensorsByKey, inputs, outputs);
        }

        private static void ApplyOriginalNames(ImmutableArray<Variable> variables, List<string?> names)
        {
            for (int i = 0; i < variables.Length && i < names.Count; i++)
            {
                if (names[i] is null) continue;
                // Every graph value is a non-generic Variable (previously this needed a generic-type
                // pattern + dynamic fallback when graph values were generic).
                variables[i]?.SetUniqueName(names[i]);
            }
        }
    }
}
