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
    /// draw form <c>SHRK_RNG_UNIFORM/NORMAL</c> — inputs <c>[key, drawBase, shape, a, b]</c> —
    /// and then, like every keyed SHRK_RNG_* op (the chain splits included), to a call of the
    /// named algorithm's non-inlined function: the exported model calls tagged local
    /// FunctionProtos, so its randomness is deterministic, portable, and identifiable. The
    /// draw algorithm comes from the bound identity's algorithm id (<c>RngSeed[0]</c>); an
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
        public static void Process(FastComputationGraph graph)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));

            var functionRemap = new Dictionary<Function, Function>();
            ProcessGraph(graph, functionRemap);
        }

        private static void ProcessGraph(
            FastComputationGraph graph, Dictionary<Function, Function> functionRemap)
        {
            // Lower every Function reachable from this graph first (memoized per Function instance).
            foreach (var node in graph.Nodes)
                if (node.TargetFunction is { } fn)
                    LowerFunctionRecursive(fn, functionRemap);

            // The graph's identity: the RngSeed parameter's bound value. A still-unbound
            // RngSeed (a concrete architecture exported without a config) binds to the
            // default identity here: a concrete artifact is never unkeyed. An unknown
            // algorithm id (a file written by a newer framework version) fails loudly —
            // lowering under a substitute would silently diverge from the recorded identity.
            string algorithm = RngAlgorithms.Default;
            if (FastWireRngKeyDerivation.FindRngSeedNode(graph) is { } seedNode)
            {
                if (seedNode.OpCode == InternalOpCodes.MODEL_PARAM)
                    WriteDefaultIdentity(seedNode);
                var identityVec = seedNode.Attributes.GetTensorVal(ShrkAttrTensorData)
                    ?.As<int64>().AccessMemory().ToArray();
                if (identityVec is not null)
                {
                    var identity = RngRuntimeIdentity.Decode(identityVec);
                    var boundAlgorithm = identity.Algorithm
                        ?? throw new NotSupportedException(
                            "FastLowerRandomOps: the model's RngSeed identity records the " +
                            $"unknown algorithm id {identity.AlgorithmId} (likely written by a " +
                            "newer framework version). Lowering under a substitute algorithm " +
                            "would silently diverge from the recorded identity.");
                    algorithm = RngAlgorithms.NameOf(boundAlgorithm);
                }
            }

            var newNodes = new List<FastNode>(graph.Nodes.Count);

            foreach (var node in graph.Nodes)
            {
                if (node.OpCode == InternalOpCodes.SHRK_RNG_SPLIT ||
                    node.OpCode == InternalOpCodes.SHRK_RNG_UNIFORM ||
                    node.OpCode == InternalOpCodes.SHRK_RNG_NORMAL)
                {
                    LowerKeyedRngToFunctionCall(node);
                    newNodes.Add(node);
                    continue;
                }

                bool isUniform = node.OpCode == InternalOpCodes.SHRK_RANDOM_UNIFORM;
                bool isNormal = node.OpCode == InternalOpCodes.SHRK_RANDOM_NORMAL;
                if (!isUniform && !isNormal)
                {
                    newNodes.Add(node);
                    continue;
                }

                var idVals = node.Attributes.GetIntsVal(ShrkAttrLocalModelId);
                var keySource = node.Inputs.Count > 3 ? node.Inputs[3] : null;
                if (idVals is { Length: > 0 } && keySource is { } ks)
                {
                    RewriteFeedToKeyedDraw(node, isUniform, ks, algorithm, newNodes);
                    LowerKeyedRngToFunctionCall(node);
                    newNodes.Add(node);
                    continue;
                }

                // A feed without stream identity (no ModelId, or no chain — e.g. inside
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
            var data = new OnnxTensorData<int64>(
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
                defaultName: fn.DefaultName, friendlyName: fn.FriendlyName);
        }

        private static bool HasRandomOps(FastComputationGraph graph) =>
            graph.Nodes.Any(node =>
                node.OpCode == InternalOpCodes.SHRK_RANDOM_UNIFORM ||
                node.OpCode == InternalOpCodes.SHRK_RANDOM_NORMAL ||
                node.OpCode == InternalOpCodes.SHRK_RNG_SPLIT ||
                node.OpCode == InternalOpCodes.SHRK_RNG_UNIFORM ||
                node.OpCode == InternalOpCodes.SHRK_RNG_NORMAL);

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
            var (kind, dtype, rank) = node.OpCode switch
            {
                InternalOpCodes.SHRK_RNG_SPLIT => (RngAlgorithms.KindSplit, DType.Int64, 1L),
                InternalOpCodes.SHRK_RNG_UNIFORM => (RngAlgorithms.KindUniform, DType.Float32, -1L),
                _ => (RngAlgorithms.KindNormal, DType.Float32, -1L),
            };
            var fn = RngAlgorithms.GetFunction(algorithm, kind);

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
        /// Rewrites an id-bearing SHRK_RANDOM_* feed in place to the SHRK_RNG_UNIFORM/NORMAL
        /// form (inputs <c>[key, drawBase, shape, a, b]</c>). The key is the feed's derivation
        /// chain (already wired as its key input); drawBase is the site's own counter input
        /// when wired (e.g. the injected per-execution state counter) else a constant 0, and
        /// the distribution bounds come off the node's attributes as f32 scalar constants.
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

            var drawBaseKey = node.Inputs.Count > 1 && node.Inputs[1] is { } db
                ? db
                : AppendScalarInt64(0L, newNodes);
            var aKey = AppendScalarFloat32(a, newNodes);
            var bKey = AppendScalarFloat32(b, newNodes);

            var newOp = isUniform ? InternalOpCodes.SHRK_RNG_UNIFORM : InternalOpCodes.SHRK_RNG_NORMAL;
            var attrDefs = Definitions.NodeDefinitions[newOp].AttributeDefs;
            node.OpCode = newOp;
            node.Attributes = OnnxCSharpAttributes.FromCSharpVals(
                new Dictionary<string, object?> { [ShrkAttrRngAlgorithm] = algorithm },
                attrDefs);
            node.FullInputs = new Dictionary<string, List<FastTensorKey?>>
            {
                [""] = new List<FastTensorKey?> { keySource, drawBaseKey, shapeInput, aKey, bKey }
            };
        }

        private static FastTensorKey AppendScalarInt64(long value, List<FastNode> newNodes)
        {
            var data = new OnnxTensorData<int64>(
                new Shape(Array.Empty<long>()),
                OnnxUtils.CreateTensorValue(new Shape(Array.Empty<long>()), (long[])[value]));
            return AppendConstant(data, newNodes);
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
        /// </summary>
        private static void LowerToOnnxRandomLike(FastNode node, bool isUniform, List<FastNode> newNodes)
        {
            var shapeInput = node.Inputs[0]
                ?? throw new InvalidOperationException("SHRK_RANDOM_* has null shape input.");

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
