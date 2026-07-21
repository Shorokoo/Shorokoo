using Shorokoo;
using Shorokoo.Core.Graph;
using Shorokoo.Runtime;
using Shorokoo.Graph;
using Shorokoo.Core.Inference;
using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Modules;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// Lowers Shorokoo-specific "attribute-tensorized" operator variants (those registered in
    /// <see cref="AttributeTensorOpRegistry"/>, e.g. <c>SHRK_CONV</c>) back to their standard ONNX
    /// operator. Each variant carries some attributes as int64 tensor inputs rather than static
    /// attributes; this pass resolves those inputs to constant values, writes them as static
    /// attributes on the standard op, swaps the op code and drops the now-consumed inputs.
    ///
    /// <para>Per attribute-source subgraph, the resolution cascade is:</para>
    /// <list type="number">
    /// <item>Constant-fold: run <see cref="QuickExecutionEngine"/> on an input-free pruned clone.</item>
    /// <item>QEE with the supplied <see cref="ModelParamList"/> sample inputs bound.</item>
    /// <item>ONNX Runtime (<see cref="ComputeContext.Execute(InternalComputationGraph, IData[])"/>) with the samples bound.</item>
    /// </list>
    /// Because this pass runs after the <c>FastSimplify</c> that already constant-folds and unrolls
    /// loops, compile-time-constant and loop-derived geometry is already resolved by strategy 1;
    /// strategies 2/3 cover geometry that is only computable from the sample inputs.
    ///
    /// <para>Variant ops can appear in the main graph and inside <see cref="Function"/> bodies;
    /// both are lowered (function bodies use strategy 1 only, since top-level sample inputs do not
    /// map to function parameters).</para>
    /// </summary>
    internal static class FastLowerAttributeTensorOps
    {
        public static void Process(
            InternalComputationGraph graph,
            ModelParamList? sampleInputs = null,
            ComputeContext? compute = null)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));
            compute ??= ComputeContext.Default;
            ProcessGraph(graph, sampleInputs, compute, new Dictionary<Function, Function>());
        }

        private static void ProcessGraph(
            InternalComputationGraph graph,
            ModelParamList? sampleInputs,
            ComputeContext compute,
            Dictionary<Function, Function> functionRemap)
        {
            // Lower every Function reachable from this graph (post-order, memoized per instance).
            foreach (var node in graph.Nodes)
                if (node.TargetFunction is { } fn)
                    LowerFunctionRecursive(fn, compute, functionRemap);

            // Lower variant nodes in the main graph by mutating them in place.
            foreach (var node in graph.Nodes)
                if (AttributeTensorOpRegistry.Specs.TryGetValue(node.OpCode, out var spec))
                    LowerNode(node, spec, graph, sampleInputs, compute);

            // Lowering can leave the attribute-source subgraphs unreferenced; sweep them.
            FastProcessorHelper.RemoveUnreachableNodes(graph);

            if (functionRemap.Count > 0)
                foreach (var node in graph.Nodes)
                    if (node.TargetFunction is { } fn && functionRemap.TryGetValue(fn, out var newFn))
                        node.TargetFunction = newFn;
        }

        private static void LowerFunctionRecursive(
            Function fn, ComputeContext compute, Dictionary<Function, Function> functionRemap)
        {
            if (functionRemap.ContainsKey(fn)) return;

            var bodyHasVariantOps = HasVariantOps(fn.OriginalFastGraph) ||
                                    fn.ReferencedFunctions.Any(x => HasVariantOps(x.OriginalFastGraph));
            if (!bodyHasVariantOps)
            {
                functionRemap[fn] = fn;
                return;
            }

            var bodyFast = fn.OriginalFastGraph.Clone();
            // Function bodies resolve via constant folding only — top-level sample inputs don't
            // correspond to function parameters.
            ProcessGraph(bodyFast, sampleInputs: null, compute, functionRemap);

            functionRemap[fn] = new Function(bodyFast, fn.FunctionType,
                defaultName: fn.DefaultName, friendlyName: fn.FriendlyName, fn.StateOwnership);
        }

        private static bool HasVariantOps(InternalComputationGraph graph) =>
            graph.Nodes.Any(node => AttributeTensorOpRegistry.Specs.ContainsKey(node.OpCode));

        private static void LowerNode(
            FastNode node,
            AttributeTensorSpec spec,
            InternalComputationGraph graph,
            ModelParamList? sampleInputs,
            ComputeContext compute)
        {
            var inputDefs = Definitions.NodeDefinitions[node.OpCode].VariantDefinitions[0].InputDefs;
            if (!node.FullInputs.TryGetValue("", out var slots))
                throw new InvalidOperationException($"{node.OpCode}: node has no default input group.");
            if (slots.Count != inputDefs.Count)
                throw new InvalidOperationException(
                    $"{node.OpCode}: input slot count {slots.Count} does not match definition input count {inputDefs.Count}.");

            var slotByName = new Dictionary<string, int>(inputDefs.Count);
            for (int i = 0; i < inputDefs.Count; i++)
                slotByName[inputDefs[i].ParamName] = i;

            // Gather the attribute-source tensor keys to resolve, in mapping order.
            var keysToResolve = new List<FastTensorKey>(spec.TensorAttributes.Length);
            foreach (var mapping in spec.TensorAttributes)
            {
                if (!slotByName.TryGetValue(mapping.InputName, out var idx))
                    throw new InvalidOperationException(
                        $"{node.OpCode}: registry references unknown input '{mapping.InputName}'.");
                var key = slots[idx]
                    ?? throw new InvalidOperationException(
                        $"{node.OpCode}: attribute-source input '{mapping.InputName}' is missing.");
                keysToResolve.Add(key);
            }

            var resolved = ResolveKeys(graph, keysToResolve, sampleInputs, compute, node.OpCode);

            // Build the standard op's attribute bag: carry over the variant's static attributes the
            // standard op also declares (and that aren't produced by a mapping), then add the
            // resolved geometry values.
            var standardDefs = Definitions.NodeDefinitions[spec.StandardOpCode].AttributeDefs;
            var attrSourceNames = spec.TensorAttributes.Select(m => m.AttributeName).ToHashSet();
            var variantVals = node.Attributes.GetAttributeVals();

            var newAttrs = new Dictionary<string, object?>();
            foreach (var def in standardDefs)
                if (!attrSourceNames.Contains(def.AttributeName)
                    && variantVals.TryGetValue(def.AttributeName, out var carried))
                    newAttrs[def.AttributeName] = carried;

            for (int i = 0; i < spec.TensorAttributes.Length; i++)
            {
                var mapping = spec.TensorAttributes[i];
                var longs = resolved[i].As<int64>().AccessMemory().ToArray();
                newAttrs[mapping.AttributeName] = mapping.IsScalar
                    ? (object)(longs.Length > 0 ? longs[0] : 0L)
                    : longs;
            }

            node.OpCode = spec.StandardOpCode;
            node.Attributes = OnnxCSharpAttributes.FromCSharpVals(newAttrs, standardDefs);

            // Keep only pass-through inputs (those without a tensor-attribute mapping), in def order.
            var mappedInputs = spec.TensorAttributes.Select(m => m.InputName).ToHashSet();
            var newSlots = new List<FastTensorKey?>();
            for (int i = 0; i < inputDefs.Count; i++)
                if (!mappedInputs.Contains(inputDefs[i].ParamName))
                    newSlots.Add(slots[i]);
            node.FullInputs = new Dictionary<string, List<FastTensorKey?>> { [""] = newSlots };
        }

        private static TensorData[] ResolveKeys(
            InternalComputationGraph graph,
            List<FastTensorKey> keys,
            ModelParamList? sampleInputs,
            ComputeContext compute,
            string opCodeForError)
        {
            var resolved = new TensorData?[keys.Count];

            // Strategy 1: constant-fold via QEE on an input-free pruned clone.
            {
                var resolver = graph.Clone();
                resolver.Inputs = new List<FastTensorKey>();
                resolver.InputUniqueNames = new List<string?>();
                resolver.Outputs = new List<FastTensorKey>(keys);
                resolver.OutputUniqueNames = new List<string?>(new string?[keys.Count]);
                resolver.OutputRankOverrides = null;
                FastProcessorHelper.RemoveUnreachableNodes(resolver);
                TryResolveWithQee(resolver, keys, resolved, samples: null);
            }

            // Strategies 2/3: QEE then ORT with the sample inputs bound.
            if (sampleInputs is not null && AnyUnresolved(resolved))
            {
                var resolver = graph.Clone();
                resolver.Outputs = new List<FastTensorKey>(keys);
                resolver.OutputUniqueNames = new List<string?>(new string?[keys.Count]);
                resolver.OutputRankOverrides = null;
                FastProcessorHelper.RemoveUnreachableNodes(resolver);

                var orderedSamples = OrderSamples(resolver, sampleInputs);
                if (orderedSamples is not null)
                {
                    TryResolveWithQee(resolver, keys, resolved, orderedSamples);
                    if (AnyUnresolved(resolved))
                        TryResolveWithOrt(resolver, keys, resolved, orderedSamples, compute);
                }
            }

            for (int i = 0; i < resolved.Length; i++)
                if (resolved[i] is null)
                    throw new InvalidOperationException(
                        $"FastLowerAttributeTensorOps: could not resolve a tensor-attribute input of '{opCodeForError}' " +
                        "to a constant value. Attribute-source inputs must be computable at lowering time " +
                        "(compile-time constant, or derivable from the supplied sample inputs).");

            return resolved!;
        }

        private static bool AnyUnresolved(TensorData?[] resolved)
        {
            foreach (var r in resolved)
                if (r is null) return true;
            return false;
        }

        private static void TryResolveWithQee(
            InternalComputationGraph resolver, List<FastTensorKey> keys, TensorData?[] resolved, TensorData[]? samples)
        {
            try
            {
                var store = samples is null
                    ? new QuickExecutionEngine().Run(resolver)
                    : new QuickExecutionEngine().Run(resolver, samples);

                for (int i = 0; i < keys.Count; i++)
                    if (resolved[i] is null
                        && store.TryGetValue(keys[i], out var rt)
                        && rt is RuntimeTensor plain
                        && TensorDataConverter.ToTensorData(plain) is { } td)
                        resolved[i] = td;
            }
            catch
            {
                // QEE catches per-op exceptions internally; reaching here is a structural failure.
                // Leave the slots for the next strategy.
            }
        }

        private static void TryResolveWithOrt(
            InternalComputationGraph resolver, List<FastTensorKey> keys, TensorData?[] resolved,
            TensorData[] samples, ComputeContext compute)
        {
            var results = compute.Execute(resolver, samples);
            for (int i = 0; i < keys.Count; i++)
                if (resolved[i] is null)
                    resolved[i] = results[i].ToTensorData();
        }

        /// <summary>
        /// Orders the supplied sample inputs to match the resolver graph's input slots by name.
        /// Returns null if any input slot has no matching sample (so the sample-based strategies
        /// are skipped and the caller relies on constant folding).
        /// </summary>
        private static TensorData[]? OrderSamples(InternalComputationGraph resolver, ModelParamList sampleInputs)
        {
            var byName = new Dictionary<string, TensorData>();
            foreach (var p in sampleInputs.ModelParams)
                byName[p.ParamName] = p.ToTensorData();

            var ordered = new TensorData[resolver.Inputs.Count];
            for (int i = 0; i < resolver.Inputs.Count; i++)
            {
                var name = i < resolver.InputUniqueNames.Count ? resolver.InputUniqueNames[i] : null;
                if (name is null || !byName.TryGetValue(name, out var td))
                    return null;
                ordered[i] = td;
            }
            return ordered;
        }
    }
}
