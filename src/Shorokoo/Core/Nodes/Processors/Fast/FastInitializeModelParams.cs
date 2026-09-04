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
using System.Globalization;
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
    /// <see cref="ComputeContext.Run(InternalComputationGraph, NamedModelParam[])"/> — once per
    /// parameter, over the slice of the graph feeding that one initializer's output (see
    /// <see cref="ChunkFor"/>), each result copied off its session (see <see cref="Rehost"/>).
    /// The decoded results are returned as a
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
            var collectedInventory =
                new List<(string? Template, ConcreteModelParamInfo? Info, ModelId Id, DType DType, long[]? Shape)>();

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
                ConcreteModelParamInfo? paramInfo = null;

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

                    paramInfo = info;

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
                collectedInventory.Add((identifierTemplate, paramInfo, modelId, dtype, shape));
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

            // Mirror the legacy `RebuildGraph(newInputs: [], newOutputs: [...])` call: the
            // initialization graph takes no input at all, and each parameter's initializer
            // output becomes a graph output.
            workGraph.Inputs = new List<FastTensorKey>();
            workGraph.InputUniqueNames = new List<string?>();
            workGraph.OutputRankOverrides = null;

            var builder = ImmutableDictionary.CreateBuilder<ModelId, TensorData>();
            for (int i = 0; i < collectedOutputKeys.Count; i++)
            {
                var chunk = ChunkFor(workGraph, collectedOutputKeys[i]);
                try
                {
                    builder[collectedModelIds[i]] = Rehost(computeContext.Run(chunk)[0].ToTensorData());
                }
                catch (System.Exception ex) when (IsAllocationFailure(ex))
                {
                    // The backend reports an out-of-memory abort as a bare "bad allocation", with
                    // no parameter, shape or size — nothing to separate "this parameter is too
                    // large" from "this graph is malformed" (#208). Because each parameter now
                    // gets its own session, the one being initialized IS the one that failed, so
                    // name it, with the rest of the model as context; the inner exception keeps
                    // the original diagnosis. Only allocation failures are relabelled: everything
                    // else this call can raise (a missing backend package, an unsupported op, a
                    // malformed graph) already says what it is, and keeping its type keeps the
                    // catch clauses around this API working.
                    throw new ComputeContextException(ErrorCodes.CR008, "FastInitializeModelParams",
                        DescribeInventory(collectedInventory, failing: i) +
                        " Underlying failure: " + ex.Message, ex);
                }
            }
            return builder.ToImmutable();
        }

        /// <summary>
        /// Copies a freshly computed parameter value onto storage of its own, then releases the
        /// backend tensor it came out of.
        ///
        /// <para>The copy is the half that matters. A backend result tensor is allocated by ITS
        /// OWN session's allocator and keeps that allocator — and so the session's whole arena,
        /// sized to the draw's peak rather than to the parameter — alive for as long as the value
        /// is referenced. These values ARE referenced: they are the rig's initial weights, so no
        /// collection can reclaim them. Returned as-is, one session per parameter holds every
        /// session's arena at once: measured on the 12 x [384, 384] stack, 379-481 MiB still live
        /// after a forced collection, against 10-32 MiB re-hosted, for 6.75 MiB of actual
        /// parameter.</para>
        ///
        /// <para>Disposing is the smaller half: dropping the last reference already lets the
        /// finalizer free the value, and disposing only makes that deterministic rather than
        /// leaving it to a thread nobody controls. It has to be the BACKING VALUE — TensorData's
        /// own Dispose is the standard pattern with an empty body, since the runtime value owns
        /// the buffer, so disposing the wrapper frees nothing.</para>
        ///
        /// <para>This is what lets <see cref="ChunkFor"/> buy its speed for nothing: with the
        /// copy, slicing costs no measurable memory over one whole-model session; without it,
        /// several hundred MiB.</para>
        /// </summary>
        private static TensorData Rehost(TensorData data)
        {
            if (data.DType == DType.String) return data;
            // Read the bytes before disposing: that invalidates the buffer they came from.
            var copy = TensorData.CreateFromRawBytes(data.Shape, data.DType, data.AccessRawMemory().ToArray());
            if (data is IOnnxData backed) backed.Value.Dispose();
            return copy;
        }

        /// <summary>
        /// The single-parameter slice of the rewritten initialization graph: a copy whose only
        /// output is <paramref name="outputKey"/>, swept down to the nodes that actually feed it.
        ///
        /// <para>Initialization runs ONE SESSION PER PARAMETER rather than one session for the
        /// whole model. The parameters are mutually independent — each keyed draw hangs off its
        /// own stream key and reads nothing another initializer produces — so slicing changes no
        /// value; what it changes is the size of the graph any one session has to build. That
        /// matters because the backend's session build is SUPERLINEAR in graph size. Measured on
        /// an N-layer stack of [384, 384] normal draws, with constant folding already off (see
        /// <c>ComputeContext.IsFullyConstant</c>, the other half of this fix), building the
        /// whole-model init graph cost 0.23 s at N=1, 0.88 s at N=2, 3.4 s at N=4, 13.4 s at N=8
        /// and 32 s at N=12 — a clean 4x per doubling. Left whole, a GPT-sized model spends
        /// minutes of pure build before the first gradient (Shorokoo/Shorokoo#195).</para>
        ///
        /// <para>Slicing does not pay for that speed in memory — but only because of
        /// <see cref="Rehost"/>. N sessions would otherwise hold N arenas at once, each sized to a
        /// draw rather than to a parameter, for several hundred MiB more than the one-session
        /// graph. With each result copied off its session as it is produced, the N=12 model above
        /// peaks within noise of the one-session build (497 MiB against 492) while costing a
        /// quarter of its wall clock (8.6 s against 37.0). The order-of-magnitude memory win over
        /// the behaviour #194 reported is <c>IsFullyConstant</c>'s, not this.</para>
        ///
        /// <para>Sharing one function BODY across the parameters does not substitute for slicing:
        /// the backend inlines every call site, so the graph it builds is the same size either
        /// way (measured — it made no difference). What IS linear here is the backend session
        /// builds, which is where the cost lives; cloning and sweeping the graph once per
        /// parameter is itself quadratic in the parameter count, but it is host-side pointer work
        /// against a session build measured in tenths of a second.</para>
        /// </summary>
        private static InternalComputationGraph ChunkFor(
            InternalComputationGraph workGraph, FastTensorKey outputKey)
        {
            var chunk = workGraph.Clone();
            chunk.Outputs = new List<FastTensorKey> { outputKey };
            chunk.OutputUniqueNames = new List<string?> { null };
            FastProcessorHelper.RemoveUnreachableNodes(chunk);
            return chunk;
        }

        /// <summary>
        /// True for the failures #208 is about: a managed out-of-memory, or a native allocation
        /// abort the backend reports only as text ("bad allocation", "std::bad_alloc", ...).
        /// </summary>
        private static bool IsAllocationFailure(System.Exception ex)
        {
            string[] markers = ["bad alloc", "bad_alloc", "out of memory", "failed to allocate",
                "insufficient memory"];
            for (var e = ex; e is not null; e = e.InnerException)
            {
                if (e is System.OutOfMemoryException) return true;
                if (markers.Any(m => e.Message.Contains(m, System.StringComparison.OrdinalIgnoreCase)))
                    return true;
            }
            return false;
        }

        /// <summary>
        /// Renders the parameter at <paramref name="failing"/> — the one whose own initialization
        /// session just aborted — and then the model it belongs to as context: the parameter
        /// count, the total element count and byte size, and the largest parameters by size.
        /// Sizes saturate rather than wrap — the parameter that blew its session up is exactly
        /// the one whose element count can overflow Int64, and it has to stay at the top of the
        /// list.
        /// </summary>
        private static string DescribeInventory(
            List<(string? Template, ConcreteModelParamInfo? Info, ModelId Id, DType DType, long[]? Shape)> inventory,
            int failing)
        {
            const long Unknown = -1;

            static long AddSat(long a, long b) => a > long.MaxValue - b ? long.MaxValue : a + b;

            static long ElementCount(long[]? shape)
            {
                if (shape is null) return Unknown;
                long n = 1;
                foreach (var d in shape)
                {
                    if (d < 0) return Unknown;
                    if (d != 0 && n > long.MaxValue / d) return long.MaxValue;
                    n *= d;
                }
                return n;
            }

            static long ByteCount(DType dtype, long elements)
            {
                if (elements < 0) return Unknown;
                int bits;
                try { bits = dtype.EncodingBitCount; }
                catch (UnsupportedDTypeException) { return Unknown; }
                if (bits <= 0) return Unknown;
                return elements > long.MaxValue / bits ? long.MaxValue : elements * bits / 8;
            }

            static string Bytes(long bytes) => bytes < 0
                ? "unknown size"
                : bytes == long.MaxValue ? "more than 8 EiB"
                : bytes >= 1L << 60 ? Fmt(bytes / (double)(1L << 60), "EiB")
                : bytes >= 1L << 50 ? Fmt(bytes / (double)(1L << 50), "PiB")
                : bytes >= 1L << 40 ? Fmt(bytes / (double)(1L << 40), "TiB")
                : bytes >= 1L << 30 ? Fmt(bytes / (double)(1L << 30), "GiB")
                : bytes >= 1L << 20 ? Fmt(bytes / (double)(1L << 20), "MiB")
                : bytes.ToString("N0", CultureInfo.InvariantCulture) + " bytes";

            static string Fmt(double v, string unit) => v.ToString("F2", CultureInfo.InvariantCulture) + " " + unit;

            var sized = inventory
                .Select(x =>
                {
                    var elements = ElementCount(x.Shape);
                    return (x.Template, x.Info, x.Id, x.Shape, DType: x.DType, Elements: elements,
                        Bytes: ByteCount(x.DType, elements));
                })
                .ToArray();

            static string Describe(
                (string? Template, ConcreteModelParamInfo? Info, ModelId Id, long[]? Shape,
                 DType DType, long Elements, long Bytes) p)
                => $"'{p.Info?.ToShorokooIdString() ?? p.Template ?? "<unnamed>"}' " +
                   $"at ModelId [{string.Join(", ", p.Id.Vals)}] " +
                   $"{p.DType} [{(p.Shape is null ? "unknown shape" : string.Join(", ", p.Shape))}] " +
                   $"= {Bytes(p.Bytes)}";

            var totalBytes = sized.Any(x => x.Bytes < 0)
                ? Unknown : sized.Aggregate(0L, (t, x) => AddSat(t, x.Bytes));
            var totalElements = sized.Any(x => x.Elements < 0)
                ? Unknown : sized.Aggregate(0L, (t, x) => AddSat(t, x.Elements));
            var largest = sized.OrderByDescending(x => x.Bytes).Take(5).Select(Describe);

            return $"initializing the model parameter {Describe(sized[failing])} failed. " +
                   $"It is 1 of {sized.Length} " +
                   $"({(totalElements < 0 ? "unknown"
                        : totalElements == long.MaxValue ? "more than 9.2e18"
                        : totalElements.ToString("N0", CultureInfo.InvariantCulture))} " +
                   $"elements, {Bytes(totalBytes)} in total); the largest of them, in order: " +
                   string.Join("; ", largest) + ".";
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
