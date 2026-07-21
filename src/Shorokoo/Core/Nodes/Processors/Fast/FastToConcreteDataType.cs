using Shorokoo;
using Shorokoo.Core.Graph;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Onnx;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Utils;
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics;
using System.Linq;
using Shorokoo.Core.Nodes.Processors.AutoGrad;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// Fast equivalent of <c>ToConcreteDataTypeProcessor</c>.
    /// Converts a specialized generic <see cref="InternalComputationGraph"/> to a fully concrete
    /// graph by:
    /// <list type="number">
    /// <item>Building specialized variants of every reachable generic <see cref="Function"/>,
    /// one per unique set of type arguments (via <see cref="FastChangeGenericTypeSpecialization"/>).</item>
    /// <item>For each specialization, producing a concrete <see cref="Function"/> by stripping
    /// generic metadata from DType / DTypes / Tensor attributes, removing
    /// <see cref="InternalOpCodes.GENERIC_TYPE_INPUT"/> input nodes, and rewiring call-site
    /// <see cref="FastNode.TargetFunction"/> references to the matching concrete function.</item>
    /// <item>Applying the same concretization pass to a clone of the top-level graph.</item>
    /// </list>
    ///
    /// Returns a fresh <see cref="InternalComputationGraph"/>; the input is not mutated. Because
    /// Fast tensors don't carry per-tensor types, no re-inference step is required — type
    /// information is reconstructed at the next FastCG → CG boundary.
    /// </summary>
    internal static class FastToConcreteDataType
    {
        public static InternalComputationGraph Process(InternalComputationGraph graph)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));

            var specializedFunctions = PrepareSpecializedFunctions(graph);

            var concreteFunctions = new Dictionary<(Function genericFn, string argsKey), Function>();

            foreach (var genericFn in EnumerateGenericFunctionsPostOrder(graph))
            {
                if (!specializedFunctions.TryGetValue(genericFn, out var perArgs))
                    continue;

                foreach (var (argsKey, specializedFn) in perArgs)
                {
                    var concreteFast = ProcessTopLevelGraph(specializedFn.OriginalFastGraph.Clone(), concreteFunctions);
                    var concreteFunction = new Function(concreteFast, specializedFn.FunctionType,
                        defaultName: specializedFn.DefaultName,
                        friendlyName: specializedFn.FriendlyName);
                    concreteFunctions[(genericFn, argsKey)] = concreteFunction;
                }
            }

            return ProcessTopLevelGraph(graph.Clone(), concreteFunctions);
        }

        private static Dictionary<Function, Dictionary<string, Function>> PrepareSpecializedFunctions(InternalComputationGraph graph)
        {
            var specializedFunctions = new Dictionary<(Function, string), Function>();
            var requiredSpecializations = new Dictionary<Function, HashSet<(FastNode refNode, string argsKey)>>();

            foreach (var (refNode, argsKey) in EnumerateGenericCallSites(graph)
                            .DistinctBy(x => (x.refNode.TargetFunction!, x.argsKey)))
            {
                var fn = refNode.TargetFunction.AssertNotNull();
                if (!requiredSpecializations.ContainsKey(fn))
                    requiredSpecializations[fn] = new();
                requiredSpecializations[fn].Add((refNode, argsKey));
            }

            while (requiredSpecializations.Count != 0)
            {
                var fn = requiredSpecializations.Keys.First();
                var specialization = requiredSpecializations[fn].First();

                Debug.Assert(!specializedFunctions.ContainsKey((fn, specialization.argsKey)));

                var typeArgs = GetGenericTypeParamVals(specialization.refNode).AssertNotNull();

                // The function body's leading inputs are GENERIC_TYPE_INPUT nodes, one per
                // generic parameter, in source order. Pair each call-site type arg with the
                // corresponding GenericTypeParamName from the function body.
                var genericInputNodes = fn.OriginalFastGraph.Nodes
                    .Where(n => n.OpCode == InternalOpCodes.GENERIC_TYPE_INPUT)
                    .Take(typeArgs.Length)
                    .ToArray();
                Debug.Assert(genericInputNodes.Length == typeArgs.Length);

                var typeSpecializations = typeArgs.Zip(genericInputNodes).ToDictionary(
                    x => x.Second.Attributes.GetDTypeVal(OnnxOpAttributeNames.AttrDtype)!
                                .GenericTypeParamName.AssertNotNull(),
                    x => x.First.AssertNotNull().ToNonGenericType());

                var specializedFast = fn.OriginalFastGraph.Clone();
                FastChangeGenericTypeSpecialization.Process(specializedFast, typeSpecializations);

                var specializedFunction = new Function(
                    specializedFast, fn.FunctionType,
                    defaultName: fn.DefaultName,
                    friendlyName: fn.FriendlyName);

                specializedFunctions[(fn, specialization.argsKey)] = specializedFunction;

                requiredSpecializations[fn].Remove(specialization);
                if (requiredSpecializations[fn].Count == 0)
                    requiredSpecializations.Remove(fn);

                foreach (var (subRefNode, subArgsKey) in EnumerateGenericCallSites(specializedFast)
                                .DistinctBy(x => (x.refNode.TargetFunction!, x.argsKey)))
                {
                    var subFn = subRefNode.TargetFunction.AssertNotNull();
                    if (specializedFunctions.ContainsKey((subFn, subArgsKey)))
                        continue;
                    if (!requiredSpecializations.ContainsKey(subFn))
                        requiredSpecializations[subFn] = new();
                    requiredSpecializations[subFn].Add((subRefNode, subArgsKey));
                }
            }

            return specializedFunctions.GroupBy(x => x.Key.Item1).ToDictionary(
                g => g.Key,
                g => g.ToDictionary(y => y.Key.Item2, y => y.Value));
        }

        private static InternalComputationGraph ProcessTopLevelGraph(
            InternalComputationGraph graph,
            Dictionary<(Function genericFn, string argsKey), Function> concreteFunctions)
        {
            // Identify GENERIC_TYPE_INPUT nodes (and the input keys they produce) so they
            // can be removed in a single pass.
            var genericInputKeys = new HashSet<FastTensorKey>();
            var nodesToRemove = new HashSet<FastNodeKey>();
            foreach (var node in graph.Nodes)
            {
                if (node.OpCode != InternalOpCodes.GENERIC_TYPE_INPUT) continue;
                nodesToRemove.Add(node.Key);
                foreach (var k in node.Outputs.NotNulls())
                    genericInputKeys.Add(k);
            }

            var newNodes = new List<FastNode>(graph.Nodes.Count);
            foreach (var node in graph.Nodes)
            {
                if (nodesToRemove.Contains(node.Key)) continue;
                StripGenericsFromAttributesInPlace(node);
                RewireTargetFunctionInPlace(node, concreteFunctions);
                newNodes.Add(node);
            }
            graph.Nodes = newNodes;

            // Drop generic-typed entries from Inputs / InputUniqueNames in lockstep so they
            // stay positionally aligned.
            var newInputs = new List<FastTensorKey>(graph.Inputs.Count);
            var newInputNames = new List<string?>(graph.Inputs.Count);
            for (int i = 0; i < graph.Inputs.Count; i++)
            {
                if (genericInputKeys.Contains(graph.Inputs[i])) continue;
                newInputs.Add(graph.Inputs[i]);
                if (i < graph.InputUniqueNames.Count)
                    newInputNames.Add(graph.InputUniqueNames[i]);
            }
            graph.Inputs = newInputs;
            graph.InputUniqueNames = newInputNames;

            return graph;
        }

        private static void StripGenericsFromAttributesInPlace(FastNode node)
        {
            var attrs = node.Attributes;
            var rebuilt = attrs.GetAttributeVals().ToDictionary();
            bool any = false;

            foreach (var def in attrs.AttributeDefs)
            {
                switch (def.Type)
                {
                    case AttributeType.DType:
                    {
                        var dt = attrs.GetDTypeVal(def.AttributeName);
                        if (dt is null || !dt.IsGenericTypeReference) continue;
                        rebuilt[def.AttributeName] = dt.ToNonGenericType();
                        any = true;
                        break;
                    }
                    case AttributeType.DTypes:
                    {
                        var dts = attrs.GetDTypesVal(def.AttributeName);
                        if (dts is null || dts.All(x => !x.IsGenericTypeReference)) continue;
                        rebuilt[def.AttributeName] = dts.Select(x => x.ToNonGenericType()).ToArray();
                        any = true;
                        break;
                    }
                    case AttributeType.Tensor:
                    {
                        var td = attrs.GetTensorVal(def.AttributeName);
                        if (td is null || !td.DType.IsGenericTypeReference) continue;
                        rebuilt[def.AttributeName] =
                            TensorDataConversion.ConvertTensorDataType(td, td.DType.ToNonGenericType());
                        any = true;
                        break;
                    }
                }
            }

            if (any)
                node.Attributes = OnnxCSharpAttributes.FromCSharpVals(rebuilt, attrs.AttributeDefs);
        }

        private static void RewireTargetFunctionInPlace(
            FastNode node,
            Dictionary<(Function genericFn, string argsKey), Function> concreteFunctions)
        {
            var key = TryGetGenericCallSiteKey(node);
            if (key is null) return;
            if (concreteFunctions.TryGetValue((key.Value.refNode.TargetFunction.AssertNotNull(), key.Value.argsKey), out var concrete))
                node.TargetFunction = concrete;
        }

        private static IEnumerable<Function> EnumerateGenericFunctionsPostOrder(InternalComputationGraph graph)
        {
            // Mirrors ComputationGraph.FunctionsPostOrlder but walks via FastCG. Each function's
            // dependencies are the distinct TargetFunctions in its OriginalFastGraph nodes.
            var fnDependencies = new Dictionary<Function, HashSet<Function>>();
            var toVisit = new Queue<Function>(LocalFunctions(graph));

            while (toVisit.Count != 0)
            {
                var fn = toVisit.Dequeue();
                if (fnDependencies.ContainsKey(fn)) continue;
                var deps = LocalFunctions(fn.OriginalFastGraph).ToHashSet();
                fnDependencies[fn] = deps;
                foreach (var d in deps)
                    if (!fnDependencies.ContainsKey(d))
                        toVisit.Enqueue(d);
            }

            var result = new List<Function>();
            while (fnDependencies.Count != 0)
            {
                var ready = fnDependencies.Where(x => x.Value.Count == 0).Select(x => x.Key).ToList();
                if (ready.Count == 0)
                    throw new InvalidOperationException("Cyclic function dependencies detected in computation graph.");
                foreach (var fn in ready)
                {
                    result.Add(fn);
                    foreach (var deps in fnDependencies.Values) deps.Remove(fn);
                    fnDependencies.Remove(fn);
                }
            }
            return result;
        }

        private static IEnumerable<Function> LocalFunctions(InternalComputationGraph graph) =>
            graph.Nodes.Select(n => n.TargetFunction).NotNulls().Distinct();

        private static IEnumerable<(FastNode refNode, string argsKey)> EnumerateGenericCallSites(InternalComputationGraph graph)
        {
            foreach (var node in graph.Nodes)
            {
                var key = TryGetGenericCallSiteKey(node);
                if (key is not null) yield return key.Value;
            }
        }

        private static (FastNode refNode, string argsKey)? TryGetGenericCallSiteKey(FastNode node)
        {
            if (node.TargetFunction is null) return null;
            var typeArgs = GetGenericTypeParamVals(node);
            if (typeArgs is null || typeArgs.Value.Length == 0) return null;
            var argsKey = string.Join(",", typeArgs.Value.Select(t => t.ToNonGenericType().ToString()));
            return (node, argsKey);
        }

        private static ImmutableArray<DType>? GetGenericTypeParamVals(FastNode node)
        {
            if (!node.Attributes.IsAttributeDefined(OnnxOpAttributeNames.ShrkAttrGenericTypeArgs))
                return ImmutableArray<DType>.Empty;
            return node.Attributes.GetDTypesVal(OnnxOpAttributeNames.ShrkAttrGenericTypeArgs)?.ToImmutableArray();
        }
    }
}
