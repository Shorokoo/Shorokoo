using Shorokoo.Graph;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Rng;
using Shorokoo.Core.Utils;
using Shorokoo.Onnx;
using System;
using System.Collections.Generic;
using System.Linq;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// Random-op lowering pass (runs in the ONNX-export pre-passes). Pure rewriting: key
    /// DERIVATION does not happen here — a feed's key is its in-graph <c>SHRK_RNG_SPLIT</c>
    /// chain rooted at the <c>RngSeed</c> parameter, wired at concretization (see
    /// <see cref="FastWireRngKeyDerivation"/>).
    ///
    /// <para><b>Keyed feeds</b> (id-bearing, chain wired) rewrite to the keyed deterministic
    /// draw form <c>SHRK_RNG_UNIFORM/NORMAL</c> — inputs <c>[key, substreamIndex, shape, a, b]</c> —
    /// and then, like every keyed SHRK_RNG_* op (the chain splits included), to a call of the
    /// named algorithm's non-inlined function: the exported model calls tagged local
    /// FunctionProtos, so its randomness is deterministic, portable, and identifiable. The
    /// draw algorithm comes from the bound identity's algorithm id (<c>RngSeed[1]</c>); an
    /// unbound graph is bound to the DEFAULT identity here first — a concrete artifact is
    /// never unkeyed, and "no config" simply means the default deterministic identity.</para>
    ///
    /// <para><b>Feeds without stream identity</b> (no ModelId or no chain — e.g. draws
    /// inside un-run initializer function bodies, or graphs that never went through
    /// concretization) take the ONNX fallback: <c>ConstantOfShape +
    /// RandomUniformLike/RandomNormalLike</c>, with any user seed copied through and none
    /// synthesized.</para>
    /// </summary>
    internal static class FastLowerRandomOps
    {
        public static void Process(InternalComputationGraph graph)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));

            var functionRemap = new Dictionary<Function, Function>();
            ProcessGraph(graph, functionRemap);
        }

        private static void ProcessGraph(
            InternalComputationGraph graph, Dictionary<Function, Function> functionRemap)
        {
            // Lower every Function reachable from this graph first (memoized per Function instance).
            foreach (var node in graph.Nodes)
                if (node.TargetFunction is { } fn)
                    LowerFunctionRecursive(fn, functionRemap);

            // The graph's identity: the RngSeed parameter's bound value. A still-unbound
            // RngSeed (a concrete architecture exported without a config) binds to the
            // default identity here: a concrete artifact is never unkeyed. An unknown
            // algorithm id (a corrupt or hand-edited carrier) fails loudly —
            // lowering under a substitute would silently diverge from the recorded identity.
            string algorithm = RngAlgorithms.Default;
            if (FastWireRngKeyDerivation.FindRngSeedNode(graph) is { } seedNode)
            {
                if (seedNode.OpCode == InternalOpCodes.MODEL_PARAM)
                    WriteDefaultIdentity(seedNode);
                var rngSeedData = seedNode.Attributes.GetTensorVal(ShrkAttrTensorData)
                    ?.As<uint64>().AccessMemory().ToArray();
                if (rngSeedData is not null)
                {
                    var identity = RngRuntimeIdentity.Decode(rngSeedData);
                    var boundAlgorithm = identity.Algorithm
                        ?? throw new NotSupportedException(
                            "FastLowerRandomOps: the model's RngSeedData records the " +
                            $"unrecognized algorithm id {identity.AlgorithmId}. Lowering under " +
                            "a substitute algorithm " +
                            "would silently diverge from the recorded algorithm.");
                    algorithm = RngAlgorithms.NameOf(boundAlgorithm);
                }
            }

            var newNodes = new List<FastNode>(graph.Nodes.Count);

            foreach (var node in graph.Nodes)
            {
                if (node.OpCode == InternalOpCodes.SHRK_RNG_SPLIT ||
                    node.OpCode == InternalOpCodes.SHRK_RNG_UNIFORM ||
                    node.OpCode == InternalOpCodes.SHRK_RNG_NORMAL ||
                    node.OpCode == InternalOpCodes.SHRK_RNG_BITS)
                {
                    LowerKeyedRngToFunctionCall(node);
                    newNodes.Add(node);
                    continue;
                }

                bool isUniform = node.OpCode == InternalOpCodes.SHRK_RANDOM_UNIFORM;
                bool isNormal = node.OpCode == InternalOpCodes.SHRK_RANDOM_NORMAL;
                bool isBits = node.OpCode == InternalOpCodes.SHRK_RANDOM_BITS;
                if (!isUniform && !isNormal && !isBits)
                {
                    newNodes.Add(node);
                    continue;
                }

                var idVals = node.Attributes.GetIntsVal(ShrkAttrLocalModelId);
                // The key input is the last input slot: [shape, substreamIndex, iterationIndices, key].
                var keySource = node.Inputs.Count > 3 ? node.Inputs[3] : null;
                if (idVals is { Length: > 0 } && keySource is { } ks)
                {
                    if (isBits) RewriteBitsFeedToKeyedDraw(node, ks, algorithm, newNodes);
                    else RewriteFeedToKeyedDraw(node, isUniform, ks, algorithm, newNodes);
                    LowerKeyedRngToFunctionCall(node);
                    newNodes.Add(node);
                    continue;
                }

                // Raw bits have no unkeyed fallback: unlike a float draw, a bit pattern is only
                // meaningful under a stream key, and there is no ONNX bits-like op to defer to. So
                // a bits feed that lacks a keyed chain is always a hard error — but the two causes
                // want different diagnostics.
                if (isBits)
                {
                    if (idVals is { Length: > 0 })
                        // Id-bearing but chain-less — the graph was modified since concretization
                        // (the analogue of the float path's Debug.Assert corruption case).
                        throw new InvalidOperationException(
                            $"FastLowerRandomOps: the SHRK_RANDOM_BITS feed at ModelId " +
                            $"[{string.Join(", ", idVals)}] is id-bearing but has no key derivation " +
                            "chain — the graph was modified since concretization. Re-concretize " +
                            "(ToConcreteArchitecture) before lowering.");
                    throw new InvalidOperationException(
                        "FastLowerRandomOps: a SHRK_RANDOM_BITS feed reached lowering with no stream " +
                        "identity. Raw random bits require a keyed RNG identity and have no unkeyed " +
                        "fallback — draw them inside a concrete, id-bearing model.");
                }

                // A float feed without stream identity (no ModelId, or no chain — e.g. inside
                // an initializer function body): the ONNX fallback — ConstantOfShape +
                // RandomUniformLike/NormalLike. Every legitimate fallback case carries NO key
                // input; an id-bearing feed always got its chain at concretization, so a
                // missing chain here means the graph was modified since — and silently
                // lowering it would turn a keyed site into real backend randomness.
                System.Diagnostics.Debug.Assert(idVals is not { Length: > 0 },
                    $"FastLowerRandomOps: the feed at ModelId [{string.Join(", ", idVals ?? [])}] " +
                    "is id-bearing but has no key derivation chain wired — the graph was " +
                    "modified since concretization; lowering it to the ONNX random fallback " +
                    "would silently make a keyed site non-deterministic.");
                LowerToOnnxRandomLike(node, isUniform, newNodes);
                newNodes.Add(node);
            }

            graph.Nodes = newNodes;

            if (functionRemap.Count > 0)
                foreach (var node in graph.Nodes)
                    if (node.TargetFunction is { } fn && functionRemap.TryGetValue(fn, out var newFn))
                        node.TargetFunction = newFn;
        }

        /// <summary>Binds a still-unbound RngSeed MODEL_PARAM to the default identity in place
        /// (the export-time analogue of ToConcreteModel's default bind).</summary>
        private static void WriteDefaultIdentity(FastNode seedNode)
        {
            var identity = RngRuntimeIdentity.Build(RngConfig.Default);
            var data = new OnnxTensorData<uint64>(
                new Shape(identity.Length),
                OnnxUtils.CreateTensorValue(new Shape(identity.Length), identity));
            var attrDefs = Definitions.NodeDefinitions[InternalOpCodes.MODEL_PARAM_DATA].AttributeDefs;
            seedNode.OpCode = InternalOpCodes.MODEL_PARAM_DATA;
            seedNode.Attributes = OnnxCSharpAttributes.FromCSharpVals(
                new Dictionary<string, object?>
                {
                    [ShrkAttrTensorData] = data,
                    [ShrkAttrIsTrainable] = false,
                }, attrDefs);
            seedNode.FullInputs = new Dictionary<string, List<FastTensorKey?>>();
            seedNode.TargetFunction = null;
        }

        private static void LowerFunctionRecursive(
            Function fn, Dictionary<Function, Function> functionRemap)
        {
            if (functionRemap.ContainsKey(fn)) return;

            var bodyHasRandomOps = HasRandomOps(fn.OriginalFastGraph) ||
                                   fn.ReferencedFunctions.Any(x => HasRandomOps(x.OriginalFastGraph));
            if (!bodyHasRandomOps)
            {
                functionRemap[fn] = fn;   // visited, unchanged
                return;
            }

            var bodyFast = fn.OriginalFastGraph.Clone();
            ProcessGraph(bodyFast, functionRemap);

            functionRemap[fn] = new Function(bodyFast, fn.FunctionType,
                defaultName: fn.DefaultName, friendlyName: fn.FriendlyName, fn.StateOwnership);
        }

        private static bool HasRandomOps(InternalComputationGraph graph) =>
            graph.Nodes.Any(node =>
                node.OpCode == InternalOpCodes.SHRK_RANDOM_UNIFORM ||
                node.OpCode == InternalOpCodes.SHRK_RANDOM_NORMAL ||
                node.OpCode == InternalOpCodes.SHRK_RANDOM_BITS ||
                node.OpCode == InternalOpCodes.SHRK_RNG_SPLIT ||
                node.OpCode == InternalOpCodes.SHRK_RNG_UNIFORM ||
                node.OpCode == InternalOpCodes.SHRK_RNG_NORMAL ||
                node.OpCode == InternalOpCodes.SHRK_RNG_BITS);

        /// <summary>
        /// Rewrites a keyed SHRK_RNG_* node in place to a FUNCTION_INVOKE of the named
        /// algorithm's function of the matching kind. The node inputs already match the
        /// function's input order 1:1; RNG algorithm functions are never inlined and export
        /// as ONNX local FunctionProtos, so the call survives to the ONNX model as a
        /// Functions-domain call node (see FastOpsetResolver).
        /// </summary>
        private static void LowerKeyedRngToFunctionCall(FastNode node)
        {
            var algorithm = node.Attributes.GetStringVal(ShrkAttrRngAlgorithm)
                ?? RngAlgorithms.Default;
            // bits carries its output uint width in shrk_dtype; the other kinds have a fixed dtype.
            // A missing width is a hard error (never a silent default) — the attribute is always
            // set by the feed/factory, so its absence means graph corruption.
            DType? bitsDtype = node.OpCode == InternalOpCodes.SHRK_RNG_BITS
                ? node.Attributes.GetDTypeVal(ShrkAttrDtype)
                  ?? throw new InvalidOperationException(
                      "FastLowerRandomOps: a SHRK_RNG_BITS node is missing its shrk_dtype (output width) attribute.")
                : null;
            var (kind, dtype, rank) = node.OpCode switch
            {
                InternalOpCodes.SHRK_RNG_SPLIT => (RngAlgorithms.KindSplit, DType.UInt64, 0L),
                InternalOpCodes.SHRK_RNG_UNIFORM => (RngAlgorithms.KindUniform, DType.Float32, -1L),
                InternalOpCodes.SHRK_RNG_NORMAL => (RngAlgorithms.KindNormal, DType.Float32, -1L),
                InternalOpCodes.SHRK_RNG_BITS => (RngAlgorithms.KindBits, bitsDtype!, -1L),
                _ => throw new InvalidOperationException(
                    $"LowerKeyedRngToFunctionCall: unexpected opcode '{node.OpCode}'."),
            };
            var fn = RngAlgorithms.GetFunction(algorithm, kind, bitsDtype);

            var invokeAttrDefs = Definitions.NodeDefinitions[InternalOpCodes.FUNCTION_INVOKE].AttributeDefs;
            node.OpCode = InternalOpCodes.FUNCTION_INVOKE;
            node.Attributes = OnnxCSharpAttributes.FromCSharpVals(
                new Dictionary<string, object?>
                {
                    [ShrkAttrStructure] = (DataStructure[])[DataStructure.Tensor],
                    [ShrkAttrDtype] = (DType[])[dtype],
                    [ShrkAttrRank] = (long[])[rank],
                    [ShrkAttrGenericTypeArgs] = null,
                },
                invokeAttrDefs);
            node.IdentifierTemplate = null;
            node.TargetFunction = fn;
        }

        /// <summary>
        /// The uniform feed's optional in-graph bound inputs, at slots 4 and 5 (after
        /// <c>[shape, substreamIndex, iterationIndices, key]</c>). Present when the range was
        /// computed in-graph rather than given as literals; both arrive together or not at all.
        /// </summary>
        internal static (FastTensorKey low, FastTensorKey high)? TensorBounds(FastNode node)
        {
            var low = node.Inputs.Count > 4 ? node.Inputs[4] : null;
            var high = node.Inputs.Count > 5 ? node.Inputs[5] : null;
            if (low is null && high is null) return null;
            if (low is null || high is null)
                throw new InvalidOperationException(
                    "FastLowerRandomOps: a SHRK_RANDOM_UNIFORM feed carries only one of its two " +
                    "in-graph bound inputs. A tensor-bounded range needs both; lowering one of them " +
                    "against the other's attribute default would silently change the range.");
            return (low.Value, high.Value);
        }

        /// <summary>
        /// Rewrites an id-bearing SHRK_RANDOM_* feed in place to the SHRK_RNG_UNIFORM/NORMAL
        /// form (inputs <c>[key, substreamIndex, shape, a, b]</c>). The key is the feed's derivation
        /// chain (already wired as its key input); substreamIndex is the site's own counter input
        /// when wired (e.g. the injected per-execution state counter) else 0, and
        /// the distribution bounds come off the node's in-graph bound inputs when it has them,
        /// else off its attributes as f32 scalar constants.
        ///
        /// <para>The counter is the framework's own int64 execution ordinal, while a draw
        /// position on the RNG algorithm interface is a whole uint64 — so this boundary is
        /// where the two meet, and the cast happens here rather than leaking uint64 into the
        /// framework's state plumbing.</para>
        /// </summary>
        private static void RewriteFeedToKeyedDraw(
            FastNode node, bool isUniform, FastTensorKey keySource,
            string algorithm, List<FastNode> newNodes)
        {
            var shapeInput = node.Inputs[0]
                ?? throw new InvalidOperationException("SHRK_RANDOM_* has null shape input.");

            float a = isUniform
                ? node.Attributes.GetFloatVal(AttrLow) ?? 0.0f
                : node.Attributes.GetFloatVal(AttrMean) ?? 0.0f;
            float b = isUniform
                ? node.Attributes.GetFloatVal(AttrHigh) ?? 1.0f
                : node.Attributes.GetFloatVal(AttrScale) ?? 1.0f;

            var bounds = isUniform ? TensorBounds(node) : null;

            var substreamIndexKey = node.Inputs.Count > 1 && node.Inputs[1] is { } db
                ? AppendCastToUInt64(db, newNodes)
                : AppendScalarUInt64(0UL, newNodes);
            var aKey = bounds?.low ?? AppendScalarFloat32(a, newNodes);
            var bKey = bounds?.high ?? AppendScalarFloat32(b, newNodes);

            var newOp = isUniform ? InternalOpCodes.SHRK_RNG_UNIFORM : InternalOpCodes.SHRK_RNG_NORMAL;
            var attrDefs = Definitions.NodeDefinitions[newOp].AttributeDefs;
            node.OpCode = newOp;
            node.Attributes = OnnxCSharpAttributes.FromCSharpVals(
                new Dictionary<string, object?> { [ShrkAttrRngAlgorithm] = algorithm },
                attrDefs);
            node.FullInputs = new Dictionary<string, List<FastTensorKey?>>
            {
                [""] = new List<FastTensorKey?> { keySource, substreamIndexKey, shapeInput, aKey, bKey }
            };
        }

        /// <summary>
        /// Rewrites an id-bearing SHRK_RANDOM_BITS feed in place to the SHRK_RNG_BITS keyed draw
        /// form (inputs <c>[key, substreamIndex, shape]</c>), carrying the feed's output uint width
        /// (shrk_dtype) onto the keyed op. Bits carry no distribution bounds.
        /// </summary>
        private static void RewriteBitsFeedToKeyedDraw(
            FastNode node, FastTensorKey keySource, string algorithm, List<FastNode> newNodes)
        {
            var shapeInput = node.Inputs[0]
                ?? throw new InvalidOperationException("SHRK_RANDOM_BITS has null shape input.");
            var dtype = node.Attributes.GetDTypeVal(ShrkAttrDtype)
                ?? throw new InvalidOperationException(
                    "FastLowerRandomOps: a SHRK_RANDOM_BITS feed is missing its shrk_dtype (output width) attribute.");
            var substreamIndexKey = node.Inputs.Count > 1 && node.Inputs[1] is { } db
                ? AppendCastToUInt64(db, newNodes)
                : AppendScalarUInt64(0UL, newNodes);

            var attrDefs = Definitions.NodeDefinitions[InternalOpCodes.SHRK_RNG_BITS].AttributeDefs;
            node.OpCode = InternalOpCodes.SHRK_RNG_BITS;
            node.Attributes = OnnxCSharpAttributes.FromCSharpVals(
                new Dictionary<string, object?>
                {
                    [ShrkAttrRngAlgorithm] = algorithm,
                    [ShrkAttrDtype] = dtype,
                }, attrDefs);
            node.FullInputs = new Dictionary<string, List<FastTensorKey?>>
            {
                [""] = new List<FastTensorKey?> { keySource, substreamIndexKey, shapeInput }
            };
        }

        private static FastTensorKey AppendScalarUInt64(ulong value, List<FastNode> newNodes)
        {
            var data = new OnnxTensorData<uint64>(
                new Shape(Array.Empty<long>()),
                OnnxUtils.CreateTensorValue(new Shape(Array.Empty<long>()), (ulong[])[value]));
            return AppendConstant(data, newNodes);
        }

        /// <summary>Casts the framework's int64 execution counter to the RNG interface's
        /// uint64 draw position.</summary>
        private static FastTensorKey AppendCastToUInt64(FastTensorKey value, List<FastNode> newNodes)
        {
            var attrDefs = Definitions.NodeDefinitions[OpCodes.CAST].AttributeDefs;
            var nodeKey = FastNodeKey.New();
            var outKey = new FastTensorKey(nodeKey, 0);
            newNodes.Add(new FastNode
            {
                Key = nodeKey,
                OpCode = OpCodes.CAST,
                Attributes = OnnxCSharpAttributes.FromCSharpVals(
                    new Dictionary<string, object?> { [AttrTo] = DType.UInt64 }, attrDefs),
                FullInputs = { [""] = new List<FastTensorKey?> { value } },
                FullOutputs = { [""] = new List<FastTensorKey?> { outKey } },
            });
            return outKey;
        }

        private static FastTensorKey AppendScalarFloat32(float value, List<FastNode> newNodes)
        {
            var data = new OnnxTensorData<float32>(
                new Shape(Array.Empty<long>()),
                OnnxUtils.CreateTensorValue(new Shape(Array.Empty<long>()), (float[])[value]));
            return AppendConstant(data, newNodes);
        }

        private static FastTensorKey AppendConstant(TensorData data, List<FastNode> newNodes)
        {
            var constAttrDefs = Definitions.NodeDefinitions[OpCodes.CONSTANT].AttributeDefs;
            var key = FastNodeKey.New();
            var outKey = new FastTensorKey(key, 0);
            newNodes.Add(new FastNode
            {
                Key = key,
                OpCode = OpCodes.CONSTANT,
                Attributes = OnnxCSharpAttributes.FromCSharpVals(
                    new Dictionary<string, object?> { [AttrValue] = data }, constAttrDefs),
                FullInputs = new Dictionary<string, List<FastTensorKey?>>(),
                FullOutputs = { [""] = new List<FastTensorKey?> { outKey } },
            });
            return outKey;
        }

        private static readonly TensorData ZeroScalar = new OnnxTensorData<float32>(
            new Shape(1), OnnxUtils.CreateTensorValue((long[])[1], (float[])[0f]));

        /// <summary>
        /// Lowers an unkeyed SHRK_RANDOM_* node to <c>ConstantOfShape(shape, 0f)</c> +
        /// <c>RandomUniformLike/RandomNormalLike(placeholder)</c>, copying the distribution
        /// attrs and any user seed through (never synthesizing one).
        ///
        /// <para><c>RandomUniformLike</c> reads its bounds from ATTRIBUTES, so a feed whose range
        /// is in-graph cannot take this fallback: there is nowhere to put the bounds. Like an
        /// id-less bits feed, that combination is a hard build error rather than a silently
        /// dropped range.</para>
        /// </summary>
        private static void LowerToOnnxRandomLike(FastNode node, bool isUniform, List<FastNode> newNodes)
        {
            var shapeInput = node.Inputs[0]
                ?? throw new InvalidOperationException("SHRK_RANDOM_* has null shape input.");

            if (isUniform && TensorBounds(node) is not null)
                throw new InvalidOperationException(
                    "FastLowerRandomOps: a SHRK_RANDOM_UNIFORM feed with in-graph bound inputs " +
                    "reached lowering with no stream identity. The ONNX fallback " +
                    "(ConstantOfShape + RandomUniformLike) carries its bounds as attributes and " +
                    "cannot express a range computed in-graph, so the range would be silently " +
                    "dropped. Draw an in-graph range inside a concrete, id-bearing model (or pass " +
                    "literal bounds).");

            var placeholderKey = AppendConstantOfShape(shapeInput, newNodes);

            var dctAttrs = isUniform
                ? new Dictionary<string, object?>
                {
                    [AttrHigh] = node.Attributes.GetFloatVal(AttrHigh),
                    [AttrLow] = node.Attributes.GetFloatVal(AttrLow),
                    [AttrSeed] = node.Attributes.GetFloatVal(AttrSeed),
                }
                : new Dictionary<string, object?>
                {
                    [AttrMean] = node.Attributes.GetFloatVal(AttrMean),
                    [AttrScale] = node.Attributes.GetFloatVal(AttrScale),
                    [AttrSeed] = node.Attributes.GetFloatVal(AttrSeed),
                };
            var opCode = isUniform ? OpCodes.RANDOM_UNIFORM_LIKE : OpCodes.RANDOM_NORMAL_LIKE;
            var attrDefs = Definitions.NodeDefinitions[opCode].AttributeDefs;

            node.OpCode = opCode;
            node.Attributes = OnnxCSharpAttributes.FromCSharpVals(dctAttrs, attrDefs);
            node.FullInputs = new Dictionary<string, List<FastTensorKey?>>
            {
                [""] = new List<FastTensorKey?> { placeholderKey }
            };
        }

        private static FastTensorKey AppendConstantOfShape(FastTensorKey shapeInput, List<FastNode> newNodes)
        {
            var nodeKey = FastNodeKey.New();
            var outputKey = new FastTensorKey(nodeKey, 0);
            var attrDefs = Definitions.NodeDefinitions[OpCodes.CONSTANT_OF_SHAPE].AttributeDefs;
            var attrs = OnnxCSharpAttributes.FromCSharpVals(
                new Dictionary<string, object?> { [AttrValue] = ZeroScalar },
                attrDefs);

            newNodes.Add(new FastNode
            {
                Key = nodeKey,
                OpCode = OpCodes.CONSTANT_OF_SHAPE,
                Attributes = attrs,
                FullInputs = { [""] = new List<FastTensorKey?> { shapeInput } },
                FullOutputs = { [""] = new List<FastTensorKey?> { outputKey } },
            });

            return outputKey;
        }
    }
}
