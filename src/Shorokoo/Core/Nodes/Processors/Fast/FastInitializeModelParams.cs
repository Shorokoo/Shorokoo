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
    /// <see cref="ComputeContext.Run(InternalComputationGraph, NamedModelParam[])"/> with each
    /// initializer's output as a graph output. The decoded results are returned as a
    /// <see cref="ModelId"/> → <see cref="TensorData"/> dictionary.
    /// </summary>
    internal static class FastInitializeModelParams
    {
        public static ImmutableDictionary<ModelId, TensorData> Process(
            InternalComputationGraph graph,
            ComputeContext? computeContext,
            RngConfig? rngConfig = null,
            ConcreteModelParamInfos? paramInfos = null)
        {
            // Keyed per-parameter initialization needs BOTH the config and the inventory:
            // with a config but no inventory, every parameter would skip the keyed draw
            // substitution and initialize through its un-keyed initializer function — values
            // not derived from the config at all, while the config looks engaged (its override
            // validation below still runs).
            if (rngConfig is not null && paramInfos is null)
                throw new System.ArgumentNullException(nameof(paramInfos),
                    "FastInitializeModelParams: an RngConfig was supplied without the parameter " +
                    "inventory, but keyed per-parameter initialization needs both — without the " +
                    "inventory every parameter would initialize outside the keyed scheme, from " +
                    "values not derived from the config. Pass GetConcreteModelParamInfos() " +
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

            // Resolve every parameter's init key ONCE, up front, by executing one small graph of
            // split chains (RngKeyResolver) — the host still computes no RNG itself (#136). Each
            // initializer then embeds its key as a literal, as it always did.
            //
            // The alternative — emitting each parameter's split chain inside its own initializer
            // body — is what a naive "move the fold in-graph" does, and it is materially worse:
            // it multiplies THIS graph (which ORT must build and fold in one session) by the
            // ModelId depth of every parameter, and it makes the shared chain's placement
            // dependent on which control-flow scope the first draw happens to sit in.
            // Only parameters whose initializer actually draws need a key; a constant-filled
            // initializer (zeros/ones bias, etc.) would otherwise pay for a key nothing reads.
            var initKeys = infoById is null
                ? null
                : ResolveInitKeys(
                    workGraph.Nodes
                        .Where(n => n.OpCode == InternalOpCodes.MODEL_PARAM &&
                                    n.Attributes.GetIntsVal(OnnxOpAttributeNames.ShrkAttrLocalModelId) is not [0] &&
                                    n.TargetFunction is { } f && FastInitKeyedDraws.DrawsRandomness(f))
                        .Select(n => new ModelId(
                            n.Attributes.GetIntsVal(OnnxOpAttributeNames.ShrkAttrLocalModelId).AssertNotNull()))
                        .Distinct(),
                    rngConfig!, computeContext);

            var collectedModelIds = new List<ModelId>();
            var collectedOutputKeys = new List<FastTensorKey>();
            // What this call is initializing, kept for the failure message below: a native
            // allocation failure aborts the whole session and names nothing on its own.
            var collectedInventory = new List<(string? Template, ModelId Id, DType DType, long[]? Shape)>();

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
                var shape = node.Attributes.GetLongsVal(OnnxOpAttributeNames.ShrkAttrShape);
                // Both are cleared by the rewrite below; read them while they are still there.
                var identifierTemplate = node.IdentifierTemplate;

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
                        // the RNG key tree IS the ModelId tree — resolved above by executing the
                        // derivation, so a param's init stream stays reconstructible offline
                        // from its ModelId alone.
                        // A non-drawing initializer has no key (none was resolved); BuildKeyedDraws
                        // returns null for it anyway, so the value here is never consumed.
                        var key = initKeys!.TryGetValue(modelId, out var k) ? k : default;
                        // Init draws under the configured algorithm's registry name (the key
                        // tree itself is algorithm-independent — the split is always the default
                        // algorithm), so a param's init values switch with the algorithm just
                        // like runtime feeds.
                        var injected = FastInitKeyedDraws.BuildKeyedDraws(
                            initFn, key, info.ToShorokooIdString(),
                            Core.Rng.RngAlgorithms.NameOf(rngConfig!.Algorithm));
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
                collectedInventory.Add((identifierTemplate, modelId, dtype, shape));
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

            NamedModelParam[] results;
            try
            {
                results = computeContext.Run(workGraph);
            }
            catch (System.Exception ex)
            {
                // Every parameter's initializer runs in ONE session, so the native failure
                // (ORT reports an out-of-memory abort as a bare "bad allocation") carries no
                // parameter, shape or size — nothing separates "this parameter is too large"
                // from "this graph is malformed" (#208). Report what the session was asked to
                // allocate; the inner exception keeps the original diagnosis.
                throw new ComputeContextException(ErrorCodes.CR008, "FastInitializeModelParams",
                    DescribeInventory(collectedInventory) + " Underlying failure: " + ex.Message, ex);
            }

            return collectedModelIds.Zip(results)
                .ToImmutableDictionary(x => x.First, x => x.Second.ToTensorData());
        }

        /// <summary>
        /// Renders what one initialization session was asked to produce: the parameter count,
        /// the total element count and byte size, and the largest parameters by size.
        /// </summary>
        private static string DescribeInventory(
            List<(string? Template, ModelId Id, DType DType, long[]? Shape)> inventory)
        {
            static long ElementCount(long[]? shape)
            {
                if (shape is null) return -1;
                long n = 1;
                foreach (var d in shape)
                {
                    if (d < 0) return -1;
                    n *= d;
                }
                return n;
            }

            static long ByteCount(DType dtype, long elements)
            {
                if (elements < 0) return -1;
                try { return elements * dtype.EncodingBitCount / 8; }
                catch (UnsupportedDTypeException) { return -1; }
            }

            static string Bytes(long bytes) => bytes < 0
                ? "unknown size"
                : bytes >= 1L << 50 ? $"{bytes / (double)(1L << 50):F2} PiB"
                : bytes >= 1L << 40 ? $"{bytes / (double)(1L << 40):F2} TiB"
                : bytes >= 1L << 30 ? $"{bytes / (double)(1L << 30):F2} GiB"
                : bytes >= 1L << 20 ? $"{bytes / (double)(1L << 20):F2} MiB"
                : $"{bytes} bytes";

            var sized = inventory
                .Select(x =>
                {
                    var elements = ElementCount(x.Shape);
                    return (x.Template, x.Id, x.DType, x.Shape, Elements: elements,
                        Bytes: ByteCount(x.DType, elements));
                })
                .ToArray();

            static string Describe(
                (string? Template, ModelId Id, DType DType, long[]? Shape, long Elements, long Bytes) p)
                => $"'{p.Template ?? "<unnamed>"}' at ModelId [{string.Join(", ", p.Id.Vals)}] " +
                   $"{p.DType} [{(p.Shape is null ? "unknown shape" : string.Join(", ", p.Shape))}] " +
                   $"= {Bytes(p.Bytes)}";

            var totalBytes = sized.Any(x => x.Bytes < 0) ? -1 : sized.Sum(x => x.Bytes);
            var totalElements = sized.Any(x => x.Elements < 0) ? -1 : sized.Sum(x => x.Elements);
            var largest = sized.OrderByDescending(x => x.Bytes).Take(5).Select(Describe);

            return $"initializing {sized.Length} model parameter{(sized.Length == 1 ? "" : "s")} " +
                   $"({(totalElements < 0 ? "unknown" : totalElements.ToString("N0"))} elements, " +
                   $"{Bytes(totalBytes)}) failed. All initializers run in one session, so the " +
                   "underlying failure names no parameter of its own; the largest of them, in " +
                   "order: " + string.Join("; ", largest) + ".";
        }

        /// <summary>
        /// Resolves each parameter's init stream key by EXECUTING the in-graph derivation once
        /// for the whole model (#136: the host runs no RNG itself), in bounded chunks — instead of
        /// embedding a split chain per parameter in the much larger initialization graph.
        /// </summary>
        private static Dictionary<ModelId, ulong> ResolveInitKeys(
            IEnumerable<ModelId> modelIds, RngConfig rngConfig, ComputeContext? computeContext)
        {
            var ids = modelIds.ToArray();
            var resolved = Core.Rng.RngKeyResolver.Resolve(
                [.. ids.Select(id => rngConfig.InitKeySpec(id.Vals))], computeContext);
            var keys = new Dictionary<ModelId, ulong>(ids.Length);
            for (int i = 0; i < ids.Length; i++)
                keys[ids[i]] = resolved[i];
            return keys;
        }

    }
}
