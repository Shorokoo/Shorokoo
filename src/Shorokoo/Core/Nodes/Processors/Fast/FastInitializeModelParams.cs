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
            computeContext ??= ComputeContext.Default;

            var workGraph = graph.Clone();

            var functionInvokeAttrDefs = Definitions.NodeDefinitions[InternalOpCodes.FUNCTION_INVOKE].AttributeDefs;

            // Per-parameter initialization RNG: map each parameter's ModelId to its
            // canonical name + shape so a random initializer draws host noise keyed by
            // that parameter's own stream (see FastInitRngNoise). Null config disables it.
            var infoById = rngConfig is null || paramInfos is null
                ? null
                : paramInfos.ParamInfos.ToDictionary(x => x.ModelId);

            var collectedModelIds = new List<ModelId>();
            var collectedOutputKeys = new List<FastTensorKey>();

            foreach (var node in workGraph.Nodes)
            {
                if (node.OpCode != InternalOpCodes.MODEL_PARAM) continue;

                // The RngSeed parameter at reserved ModelId [0] carries the runtime RNG
                // identity, not a weight: it has no initializer function to run —
                // ApplyRngConfig is its initialization.
                if (node.Attributes.GetIntsVal(OnnxOpAttributeNames.ShrkAttrLocalModelId) is [0])
                    continue;

                var dtype = node.Attributes.GetDTypeVal(OnnxOpAttributeNames.ShrkAttrDtype).AssertNotNull();
                var rank = node.Attributes.GetLongVal(OnnxOpAttributeNames.ShrkAttrRank) ?? -1;
                var modelIdVals = node.Attributes.GetIntsVal(OnnxOpAttributeNames.ShrkAttrLocalModelId).AssertNotNull();
                var modelId = new ModelId(modelIdVals);

                // Replace the (shared) initializer with a per-parameter noise-injected clone
                // before the node is rewritten to FUNCTION_INVOKE (which preserves TargetFunction).
                if (infoById is not null && node.TargetFunction is { } initFn &&
                    infoById.TryGetValue(modelId, out var info))
                {
                    long elementCount = 1;
                    foreach (var d in info.Shape.Dims) elementCount *= d;
                    if (elementCount > 0)
                    {
                        // Stream key = init master folded along the parameter's ModelId path —
                        // the RNG key tree IS the ModelId tree, host-side here (bit-identical
                        // to the in-graph SHRK_RNG_SPLIT chain), so a param's init stream is
                        // reconstructible offline from its ModelId alone.
                        var key = rngConfig!.FoldInitKey(modelId.Vals);
                        // Init noise uses the configured algorithm's draw round count (the key
                        // itself is algorithm-independent — see RngConfig.FoldInitKey), so a
                        // param's init values switch with the algorithm just like runtime feeds.
                        var injected = FastInitRngNoise.BuildNoiseInjected(
                            initFn, key, info.ToShorokooIdString(), elementCount,
                            Core.Rng.RngAlgorithms.DrawRoundsOf(rngConfig.Algorithm));
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
