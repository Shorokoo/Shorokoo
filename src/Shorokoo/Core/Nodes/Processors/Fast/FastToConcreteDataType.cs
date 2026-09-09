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
    /// <item>Building a specialized body for every reachable <see cref="Function"/> that carries
    /// generic material or calls one that does, and is reached by a call site naming which
    /// specialization it wants — for a generic function one body per unique set of type arguments
    /// (via <see cref="FastChangeGenericTypeSpecialization"/>), for a non-generic caller a single
    /// body, since there only its callees change.</item>
    /// <item>For each body, producing a concrete <see cref="Function"/> by stripping
    /// generic metadata from DType / DTypes / Tensor attributes, removing
    /// <see cref="InternalOpCodes.GENERIC_TYPE_INPUT"/> input nodes, and rewiring call-site
    /// <see cref="FastNode.TargetFunction"/> references to the matching concrete function.</item>
    /// <item>Applying the same concretization pass to a clone of the top-level graph.</item>
    /// </list>
    ///
    /// <para>Walking the whole reachable closure is the point: a generic call site need not sit in
    /// the top-level graph nor in another generic body. A non-generic module that calls a generic
    /// one — and anything wrapping that module, to any depth — is rebuilt too, so its own call
    /// site points at the concrete callee. Seeding from the top-level graph and recursing only
    /// through the generic bodies left every deeper caller splicing a body that still carried its
    /// <c>GENERIC_TYPE_INPUT</c> slots against a call site supplying no values for them
    /// (Shorokoo/Shorokoo#286).</para>
    ///
    /// <para>A function with no generic material anywhere below it is left alone, keeping its
    /// identity and its flattened-body cache.</para>
    ///
    /// <para>Not every reference names a specialization: a model sequence tags its
    /// <c>SEQUENCE_CONSTRUCT</c> / <c>SEQUENCE_EMPTY</c> with the element module's own function
    /// and no type arguments. Such a reference builds no body of its own; it is bound to the sole
    /// specialization that the <em>typed</em> references built, since that is the only case where
    /// it cannot mean anything else. Where they built several it is left as it was, and a generic
    /// module held in a sequence alongside a second use at another type argument still fails to
    /// lower (Shorokoo/Shorokoo#296).</para>
    ///
    /// Returns a fresh <see cref="InternalComputationGraph"/>; the input is not mutated. Because
    /// Fast tensors don't carry per-tensor types, no re-inference step is required — type
    /// information is reconstructed at the next FastCG → CG boundary.
    /// </summary>
    internal static class FastToConcreteDataType
    {
        /// <summary>The specialization key of a non-generic function: it has exactly one body.</summary>
        private const string NonGenericArgsKey = "";

        public static InternalComputationGraph Process(InternalComputationGraph graph)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));

            var bodyFacts = new Dictionary<Function, BodyFacts>();
            var functionsPostOrder = EnumerateFunctionsPostOrder(graph, bodyFacts);
            var genericFunctions = new HashSet<Function>();
            var needsErasure = FunctionsNeedingErasure(functionsPostOrder, bodyFacts, genericFunctions);
            var specializedBodies = SpecializedBodies(graph, needsErasure, genericFunctions);

            var concreteFunctions = new Dictionary<(Function fn, string argsKey), Function>();
            var soleSpecialization = new Dictionary<Function, Function?>();

            foreach (var fn in functionsPostOrder)
            {
                if (!specializedBodies.TryGetValue(fn, out var perArgs))
                    continue;

                foreach (var (argsKey, specializedBody) in perArgs)
                {
                    var concreteFast = ConcretizeInPlace(specializedBody, concreteFunctions, soleSpecialization, genericFunctions);
                    var concrete = new Function(concreteFast, fn.FunctionType,
                        defaultName: fn.DefaultName,
                        friendlyName: fn.FriendlyName,
                        stateOwnership: fn.StateOwnership)
                    {
                        // Defensive: an RNG algorithm function only ever reaches a graph after
                        // this pass, or on reload of one already erased, so nothing here carries
                        // the tags today. Dropping them would silently make such a function
                        // inlinable, which is not a thing to leave to the pass ordering.
                        RngAlgorithm = fn.RngAlgorithm,
                        RngFunctionKind = fn.RngFunctionKind,
                    };
                    concreteFunctions[(fn, argsKey)] = concrete;
                    soleSpecialization[fn] = soleSpecialization.ContainsKey(fn) ? null : concrete;
                }
            }

            return ConcretizeInPlace(graph.Clone(), concreteFunctions, soleSpecialization, genericFunctions);
        }

        /// <summary>What one thaw of a function's body tells us about it. Read once and reused,
        /// because <see cref="Function.OriginalFastGraph"/> deep-thaws on every access.</summary>
        private readonly record struct BodyFacts(
            HashSet<Function> Callees, bool CarriesGenerics, bool DeclaresGenericParams);

        /// <summary>
        /// The reachable functions that have to be rebuilt: those whose own body carries generic
        /// material, and — transitively — their callers, whose call sites have to be repointed at
        /// the rebuilt callee. Callees precede callers in <paramref name="functionsPostOrder"/>,
        /// so one forward sweep reaches the fixpoint. Fills <paramref name="genericFunctions"/>
        /// with those that declare generic parameters.
        /// </summary>
        private static HashSet<Function> FunctionsNeedingErasure(
            IReadOnlyList<Function> functionsPostOrder,
            Dictionary<Function, BodyFacts> bodyFacts,
            HashSet<Function> genericFunctions)
        {
            var needsErasure = new HashSet<Function>();
            foreach (var fn in functionsPostOrder)
            {
                var facts = bodyFacts[fn];
                if (facts.DeclaresGenericParams)
                    genericFunctions.Add(fn);
                if (facts.CarriesGenerics || facts.Callees.Any(needsErasure.Contains))
                    needsErasure.Add(fn);
            }
            return needsErasure;
        }

        private static bool CarriesGenerics(FastNode node) =>
            node.OpCode == InternalOpCodes.GENERIC_TYPE_INPUT
            || (node.TargetFunction is not null && TypeArgs(node).Length != 0)
            || HasGenericAttributes(node);

        /// <summary>
        /// One body per (function, type arguments) pair over the whole reachable closure, walked
        /// breadth-first from <paramref name="graph"/> through the bodies as they are produced —
        /// a specialized body's own call sites name concrete type arguments, so they can only be
        /// enumerated once that body exists.
        /// </summary>
        private static Dictionary<Function, Dictionary<string, InternalComputationGraph>> SpecializedBodies(
            InternalComputationGraph graph, HashSet<Function> needsErasure, HashSet<Function> genericFunctions)
        {
            var bodies = new Dictionary<(Function fn, string argsKey), InternalComputationGraph>();
            var pending = new Queue<(FastNode refNode, Function fn, string argsKey)>();
            var queued = new HashSet<(Function fn, string argsKey)>();

            void EnqueueCallees(InternalComputationGraph body)
            {
                foreach (var (refNode, fn, argsKey) in EnumerateCallSites(body, genericFunctions))
                {
                    if (!needsErasure.Contains(fn)) continue;
                    if (!queued.Add((fn, argsKey))) continue;
                    pending.Enqueue((refNode, fn, argsKey));
                }
            }

            EnqueueCallees(graph);

            while (pending.Count != 0)
            {
                var (refNode, fn, argsKey) = pending.Dequeue();

                // OriginalFastGraph thaws a fresh mutable copy, so this body is ours to rewrite.
                var body = fn.OriginalFastGraph;
                if (genericFunctions.Contains(fn))
                    FastChangeGenericTypeSpecialization.Process(body, TypeSpecializations(body, TypeArgs(refNode)));

                bodies[(fn, argsKey)] = body;
                EnqueueCallees(body);
            }

            return bodies.GroupBy(x => x.Key.fn).ToDictionary(
                g => g.Key,
                g => g.ToDictionary(y => y.Key.argsKey, y => y.Value));
        }

        /// <summary>
        /// Pairs each call-site type argument with the generic parameter it binds. The body's
        /// leading inputs are <see cref="InternalOpCodes.GENERIC_TYPE_INPUT"/> nodes, one per
        /// generic parameter, in source order.
        /// </summary>
        private static Dictionary<string, DType> TypeSpecializations(
            InternalComputationGraph body, ImmutableArray<DType> typeArgs)
        {
            var genericInputNodes = body.Nodes
                .Where(n => n.OpCode == InternalOpCodes.GENERIC_TYPE_INPUT)
                .Take(typeArgs.Length)
                .ToArray();
            Debug.Assert(genericInputNodes.Length == typeArgs.Length);

            return typeArgs.Zip(genericInputNodes).ToDictionary(
                x => x.Second.Attributes.GetDTypeVal(OnnxOpAttributeNames.AttrDtype)!
                            .GenericTypeParamName.AssertNotNull(),
                x => x.First.AssertNotNull().ToNonGenericType());
        }

        private static InternalComputationGraph ConcretizeInPlace(
            InternalComputationGraph graph,
            Dictionary<(Function fn, string argsKey), Function> concreteFunctions,
            Dictionary<Function, Function?> soleSpecialization,
            HashSet<Function> genericFunctions)
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
                RewireTargetFunctionInPlace(node, concreteFunctions, soleSpecialization, genericFunctions);
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

        private static bool HasGenericAttributes(FastNode node)
        {
            var attrs = node.Attributes;
            foreach (var def in attrs.AttributeDefs)
            {
                switch (def.Type)
                {
                    case AttributeType.DType:
                        if (attrs.GetDTypeVal(def.AttributeName) is { IsGenericTypeReference: true }) return true;
                        break;
                    case AttributeType.DTypes:
                        if (attrs.GetDTypesVal(def.AttributeName)?.Any(x => x.IsGenericTypeReference) == true) return true;
                        break;
                    case AttributeType.Tensor:
                        if (attrs.GetTensorVal(def.AttributeName)?.DType.IsGenericTypeReference == true) return true;
                        break;
                }
            }
            return false;
        }

        private static void StripGenericsFromAttributesInPlace(FastNode node)
        {
            if (!HasGenericAttributes(node)) return;

            var attrs = node.Attributes;
            var rebuilt = attrs.GetAttributeVals().ToDictionary();

            foreach (var def in attrs.AttributeDefs)
            {
                switch (def.Type)
                {
                    case AttributeType.DType:
                    {
                        var dt = attrs.GetDTypeVal(def.AttributeName);
                        if (dt is null || !dt.IsGenericTypeReference) continue;
                        rebuilt[def.AttributeName] = dt.ToNonGenericType();
                        break;
                    }
                    case AttributeType.DTypes:
                    {
                        var dts = attrs.GetDTypesVal(def.AttributeName);
                        if (dts is null || dts.All(x => !x.IsGenericTypeReference)) continue;
                        rebuilt[def.AttributeName] = dts.Select(x => x.ToNonGenericType()).ToArray();
                        break;
                    }
                    case AttributeType.Tensor:
                    {
                        var td = attrs.GetTensorVal(def.AttributeName);
                        if (td is null || !td.DType.IsGenericTypeReference) continue;
                        rebuilt[def.AttributeName] =
                            TensorDataConversion.ConvertTensorDataType(td, td.DType.ToNonGenericType());
                        break;
                    }
                }
            }

            node.Attributes = OnnxCSharpAttributes.FromCSharpVals(rebuilt, attrs.AttributeDefs);
        }

        private static void RewireTargetFunctionInPlace(
            FastNode node,
            Dictionary<(Function fn, string argsKey), Function> concreteFunctions,
            Dictionary<Function, Function?> soleSpecialization,
            HashSet<Function> genericFunctions)
        {
            if (TryGetCallSiteKey(node, genericFunctions) is { } site)
            {
                if (concreteFunctions.TryGetValue((site.fn, site.argsKey), out var concrete))
                    node.TargetFunction = concrete;
                return;
            }

            // A node that carries a generic function without naming type arguments: a model
            // sequence tags its SEQUENCE_CONSTRUCT / SEQUENCE_EMPTY with the element module's
            // own Function, and that is where the sequence's models get their ModuleFn back
            // from. It cannot say which specialization it means, so bind it only when the graph
            // built exactly one — then there is nothing else it could be. Left alone, it keeps
            // the unspecialized function and the inliner splices a body still declaring type
            // slots against a call site supplying none.
            if (node.TargetFunction is { } fn
                    && soleSpecialization.TryGetValue(fn, out var only) && only is not null)
                node.TargetFunction = only;
        }

        private static IReadOnlyList<Function> EnumerateFunctionsPostOrder(
            InternalComputationGraph graph, Dictionary<Function, BodyFacts> bodyFacts)
        {
            // Mirrors ComputationGraph.FunctionsPostOrlder but walks via FastCG. Each function's
            // dependencies are the distinct TargetFunctions in its OriginalFastGraph nodes. This
            // is the one thaw of each body the analysis gets: everything later reads bodyFacts.
            var fnDependencies = new Dictionary<Function, HashSet<Function>>();
            var toVisit = new Queue<Function>(LocalFunctions(graph));

            while (toVisit.Count != 0)
            {
                var fn = toVisit.Dequeue();
                if (fnDependencies.ContainsKey(fn)) continue;
                var body = fn.OriginalFastGraph;
                var deps = LocalFunctions(body).ToHashSet();
                bodyFacts[fn] = new BodyFacts(deps,
                    body.Nodes.Any(CarriesGenerics),
                    body.Nodes.Any(n => n.OpCode == InternalOpCodes.GENERIC_TYPE_INPUT));

                // The topological sort below drains its own copy; bodyFacts keeps the callee set.
                fnDependencies[fn] = [.. deps];
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

        private static IEnumerable<(FastNode refNode, Function fn, string argsKey)> EnumerateCallSites(
            InternalComputationGraph graph, HashSet<Function> genericFunctions)
        {
            foreach (var node in graph.Nodes)
            {
                var site = TryGetCallSiteKey(node, genericFunctions);
                if (site is not null) yield return site.Value;
            }
        }

        /// <summary>
        /// The specialization a node selects, or <c>null</c> when it selects none. The key is read
        /// off the callee rather than off the node: a generic callee is keyed by the type arguments
        /// the node names, a non-generic one by <see cref="NonGenericArgsKey"/> whatever the node
        /// names. So a node that merely carries a generic function along (<c>SUBMODEL</c> naming no
        /// type arguments) selects nothing, rather than selecting an unspecialized body. What then
        /// happens to it is <see cref="RewireTargetFunctionInPlace"/>'s to decide.
        /// </summary>
        private static (FastNode refNode, Function fn, string argsKey)? TryGetCallSiteKey(
            FastNode node, HashSet<Function> genericFunctions)
        {
            if (node.TargetFunction is not { } fn) return null;
            if (!genericFunctions.Contains(fn)) return (node, fn, NonGenericArgsKey);

            var typeArgs = TypeArgs(node);
            if (typeArgs.Length == 0) return null;
            return (node, fn, string.Join(",", typeArgs.Select(t => t.ToNonGenericType().ToString())));
        }

        private static ImmutableArray<DType> TypeArgs(FastNode node)
        {
            if (!node.Attributes.IsAttributeDefined(OnnxOpAttributeNames.ShrkAttrGenericTypeArgs))
                return ImmutableArray<DType>.Empty;
            return node.Attributes.GetDTypesVal(OnnxOpAttributeNames.ShrkAttrGenericTypeArgs)
                        ?.ToImmutableArray() ?? ImmutableArray<DType>.Empty;
        }
    }
}
