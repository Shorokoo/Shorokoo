using Shorokoo.Core;
using Shorokoo.Core.Graph;
using Shorokoo.Runtime;
using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using Shorokoo.Core.Utils;
using Shorokoo.Onnx;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Shorokoo.Core.Nodes.Processors.AutoGrad;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// Fast-native port of <c>InitializeModelParams</c>.
    /// Walks <c>graph</c> for every <c>MODEL_PARAM</c> node, rewrites it
    /// to a <c>FUNCTION_INVOKE</c> of its initializer <see cref="Function"/> (preserving
    /// the original initializer-param inputs, the output <see cref="FastTensorKey"/>, and
    /// the target function), then runs the resulting graph through
    /// <see cref="ComputeContext.Run(FastComputationGraph, NamedModelParam[])"/> with each
    /// initializer's output as a graph output. The decoded results are returned as a
    /// <see cref="ModelId"/> → <see cref="TensorData"/> dictionary.
    /// </summary>
    internal static class FastInitializeModelParams
    {
        public static ImmutableDictionary<ModelId, TensorData> Process(
            FastComputationGraph graph,
            ComputeContext? computeContext,
            RngConfig? rngConfig = null,
            ConcreteModelParamInfos? paramInfos = null)
        {
            // Keyed per-parameter initialization needs BOTH the config and the inventory:
            // with a config but no inventory, every parameter would silently skip the keyed
            // draw substitution and initialize through its un-keyed initializer function, whose
            // in-body draws carry no ModelId and lower through the ONNX fallback to backend
            // randomness — values not derived from the config at all, while the config
            // looks engaged (its override validation below still runs).
            if (rngConfig is not null && paramInfos is null)
                throw new System.ArgumentNullException(nameof(paramInfos),
                    "FastInitializeModelParams: an RngConfig was supplied without the parameter " +
                    "inventory, but keyed per-parameter initialization needs both — without the " +
                    "inventory every parameter would silently initialize un-keyed, from backend " +
                    "randomness not derived from the config. Pass GetConcreteModelParamInfos() " +
                    "of the same concrete architecture.");

            computeContext ??= ComputeContext.Default;

            var workGraph = graph.Clone();

            var functionInvokeAttrDefs = Definitions.NodeDefinitions[InternalOpCodes.FUNCTION_INVOKE].AttributeDefs;

            // Per-parameter initialization RNG: map each parameter's ModelId to its
            // canonical name + shape so a random initializer draws in-graph keyed noise on
            // that parameter's own stream (see FastInitKeyedDraws). Null config disables it.
            var infoById = rngConfig is null
                ? null
                : paramInfos!.ParamInfos.ToDictionary(x => x.ModelId);

            var collectedModelIds = new List<ModelId>();
            var collectedOutputKeys = new List<FastTensorKey>();

            foreach (var node in workGraph.Nodes)
            {
                if (node.OpCode != InternalOpCodes.MODEL_PARAM) continue;

                var dtype = node.Attributes.GetDTypeVal(OnnxOpAttributeNames.ShrkAttrDtype).AssertNotNull();
                var rank = node.Attributes.GetLongVal(OnnxOpAttributeNames.ShrkAttrRank) ?? -1;
                var modelIdVals = node.Attributes.GetIntsVal(OnnxOpAttributeNames.ShrkAttrLocalModelId).AssertNotNull();
                var modelId = new ModelId(modelIdVals);

                // Replace the (shared) initializer with a per-parameter keyed-draw clone
                // before the node is rewritten to FUNCTION_INVOKE (which preserves TargetFunction).
                if (infoById is not null)
                {
                    // The mirror of the unmatched-override check below: a parameter the bound
                    // config cannot key must fail loudly — skipping just this one would leave
                    // its initializer un-keyed (backend randomness) while its siblings stay
                    // keyed, with nothing reporting the mix.
                    if (!infoById.TryGetValue(modelId, out var info))
                        throw new System.InvalidOperationException(
                            "FastInitializeModelParams: the trainable parameter " +
                            $"'{node.IdentifierTemplate}' at ModelId [{string.Join(", ", modelId.Vals)}] " +
                            "is missing from the supplied parameter inventory, so it would silently " +
                            "initialize un-keyed (backend randomness not derived from the RngConfig) " +
                            "while the other parameters stay keyed. The inventory must be " +
                            "GetConcreteModelParamInfos() of this same graph.");

                    if (node.TargetFunction is { } initFn)
                    {
                        // Stream key = init master folded along the parameter's ModelId path —
                        // the RNG key tree IS the ModelId tree, host-side here (bit-identical
                        // to the in-graph SHRK_RNG_SPLIT chain), so a param's init stream is
                        // reconstructible offline from its ModelId alone.
                        var key = rngConfig!.FoldInitKey(modelId.Vals);
                        // Init draws under the configured algorithm's registry name (the key
                        // itself is algorithm-independent — see RngConfig.FoldInitKey), so a
                        // param's init values switch with the algorithm just like runtime feeds.
                        var injected = FastInitKeyedDraws.BuildKeyedDraws(
                            initFn, key, info.ToShorokooIdString(),
                            Core.Rng.RngAlgorithms.NameOf(rngConfig.Algorithm));
                        if (injected is not null)
                            node.TargetFunction = injected;
                    }
                }

                var newAttributes = OnnxCSharpAttributes.FromCSharpVals(
                    new Dictionary<string, object?>
                    {
                        [OnnxOpAttributeNames.ShrkAttrStructure] = new[] { DataStructure.Tensor },
                        [OnnxOpAttributeNames.ShrkAttrDtype] = new[] { dtype },
                        [OnnxOpAttributeNames.ShrkAttrRank] = new[] { rank },
                        [OnnxOpAttributeNames.ShrkAttrGenericTypeArgs] = null,
                    },
                    functionInvokeAttrDefs);

                node.OpCode = InternalOpCodes.FUNCTION_INVOKE;
                node.Attributes = newAttributes;
                node.IdentifierTemplate = null;
                // FullInputs and TargetFunction (the initializer fn) are preserved
                // unchanged: FUNCTION_INVOKE expects the same variadic input list and
                // a TargetFunction reference, matching what MODEL_PARAM stored.

                var outputKey = node.FullOutputs[""][0]!.Value;
                collectedModelIds.Add(modelId);
                collectedOutputKeys.Add(outputKey);
            }

            // Fail-loud override validation, mirroring the Runtime-side check at bind
            // (FastBindRngConfig): a Params override that matches no parameter of this graph
            // would otherwise be a silent no-op — the exact re-keying hazard explicit seeding
            // exists to prevent.
            if (rngConfig is not null)
            {
                var paramPaths = collectedModelIds
                    .Select(id => string.Join(",", id.Vals))
                    .ToHashSet();
                var unmatched = rngConfig.OverrideKeys
                    .Where(k => k.collection == RngCollection.Params && !paramPaths.Contains(k.pathKey))
                    .Select(k => $"[{k.pathKey}]")
                    .ToArray();
                if (unmatched.Length > 0)
                    throw new System.InvalidOperationException(
                        "RngConfig.Override(Params, ...) matches no trainable parameter of this " +
                        "graph: " + string.Join(", ", unmatched) +
                        ". Parameter stream paths are listed by GetRngStreamReport(); overrides " +
                        "must use a reported path exactly.");
            }

            if (collectedOutputKeys.Count == 0)
                return ImmutableDictionary<ModelId, TensorData>.Empty;

            // Replace graph inputs / outputs to mirror the legacy
            // `RebuildGraph(newInputs: [], newOutputs: [...])` call. Then sweep the
            // nodes that no longer feed any output (e.g. the original output-producing
            // chains and any inputs they pulled in).
            workGraph.Inputs = new List<FastTensorKey>();
            workGraph.InputUniqueNames = new List<string?>();
            workGraph.Outputs = new List<FastTensorKey>(collectedOutputKeys);
            workGraph.OutputUniqueNames = collectedOutputKeys.Select(_ => (string?)null).ToList();
            workGraph.OutputRankOverrides = null;

            FastProcessorHelper.RemoveUnreachableNodes(workGraph);

            var results = computeContext.Run(workGraph);

            return collectedModelIds.Zip(results)
                .ToImmutableDictionary(x => x.First, x => x.Second.ToTensorData());
        }
    }
}
