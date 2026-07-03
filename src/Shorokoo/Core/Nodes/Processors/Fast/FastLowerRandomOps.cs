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
    /// Random-op lowering pass (runs in the ONNX-export pre-passes).
    ///
    /// <para><b>Keyed draws</b> (<c>SHRK_RNG_SPLIT/UNIFORM/NORMAL</c>, produced by
    /// <see cref="FastApplyRngKeys"/> when an <see cref="RngConfig"/> is bound) lower to
    /// FUNCTION_INVOKEs of the named algorithm's non-inlined functions — the exported model
    /// calls tagged local FunctionProtos, so its randomness is deterministic, portable, and
    /// identifiable.</para>
    ///
    /// <para><b>Stamped feeds</b> (<c>SHRK_RANDOM_*</c> carrying the
    /// <c>shrk_rng_explicit_key</c> attribute <see cref="FastApplyRngKeys"/> stamps when an
    /// <see cref="RngConfig"/> is bound) are rewritten HERE — and only here — into the keyed
    /// draw: key/drawBase/bounds materialize as constants feeding the algorithm function
    /// call. Deferring the rewrite to ONNX prep is what makes config binding cheap and
    /// re-bindable: the pre-export graph keeps its structure, a re-stamp overrides a prior
    /// stamp, and constant folding of the key chain happens on the export clone.</para>
    ///
    /// <para><b>Unstamped draws</b> (no config bound, or in-loop feeds the stamping pass
    /// skips) take the ONNX fallback: <c>ConstantOfShape +
    /// RandomUniformLike/RandomNormalLike</c> — the pre-keyed-RNG behavior, non-reproducible
    /// by nature, with any user seed copied through and none synthesized.</para>
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

                if (node.Attributes.GetLongsVal(ShrkAttrRngExplicitKey) is { Length: 2 } stampedKey)
                {
                    // Config-stamped feed: materialize the stamped key (plus drawBase and
                    // distribution bounds) as constants and call the algorithm function.
                    RewriteStampedToKeyedDraw(node, isUniform, stampedKey, newNodes);
                    LowerKeyedRngToFunctionCall(node);
                    newNodes.Add(node);
                    continue;
                }

                // Unstamped random op (no RngConfig bound, or an in-loop feed FastApplyRngKeys
                // skipped): the ONNX fallback — ConstantOfShape + RandomUniformLike/NormalLike.
                // Non-portable/non-reproducible by nature; documented as the no-config behavior.
                LowerToOnnxRandomLike(node, isUniform, newNodes);
                newNodes.Add(node);
            }

            graph.Nodes = newNodes;

            if (functionRemap.Count > 0)
                foreach (var node in graph.Nodes)
                    if (node.TargetFunction is { } fn && functionRemap.TryGetValue(fn, out var newFn))
                        node.TargetFunction = newFn;
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
                    [ShrkAttrStructure] = new[] { DataStructure.Tensor },
                    [ShrkAttrDtype] = new[] { dtype },
                    [ShrkAttrRank] = new[] { rank },
                    [ShrkAttrGenericTypeArgs] = null,
                },
                invokeAttrDefs);
            node.IdentifierTemplate = null;
            node.TargetFunction = fn;
        }

        /// <summary>
        /// Rewrites a stamped SHRK_RANDOM_* node in place to the SHRK_RNG_UNIFORM/NORMAL
        /// form (inputs <c>[key, drawBase, shape, a, b]</c>): the stamped key becomes an
        /// int64[2] constant, drawBase is the site's own counter input when wired (e.g.
        /// Dropout's per-execution state counter) else a constant 0, and the distribution
        /// bounds come off the node's attributes as f32 scalar constants.
        /// </summary>
        private static void RewriteStampedToKeyedDraw(
            FastNode node, bool isUniform, long[] stampedKey, List<FastNode> newNodes)
        {
            var shapeInput = node.Inputs[0]
                ?? throw new InvalidOperationException("SHRK_RANDOM_* has null shape input.");

            float a = isUniform
                ? node.Attributes.GetFloatVal(AttrLow) ?? 0.0f
                : node.Attributes.GetFloatVal(AttrMean) ?? 0.0f;
            float b = isUniform
                ? node.Attributes.GetFloatVal(AttrHigh) ?? 1.0f
                : node.Attributes.GetFloatVal(AttrScale) ?? 1.0f;
            var algorithm = node.Attributes.GetStringVal(ShrkAttrRngAlgorithm)
                ?? RngAlgorithms.Default;

            var keyKey = AppendKeyConstant(stampedKey, newNodes);
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
                [""] = new List<FastTensorKey?> { keyKey, drawBaseKey, shapeInput, aKey, bKey }
            };
        }

        private static FastTensorKey AppendKeyConstant(long[] keyWords, List<FastNode> newNodes)
        {
            var data = new OnnxTensorData<int64>(
                new Shape(2),
                OnnxUtils.CreateTensorValue(new Shape(2), keyWords));
            return AppendConstant(data, newNodes);
        }

        private static FastTensorKey AppendScalarInt64(long value, List<FastNode> newNodes)
        {
            var data = new OnnxTensorData<int64>(
                new Shape(Array.Empty<long>()),
                OnnxUtils.CreateTensorValue(new Shape(Array.Empty<long>()), new[] { value }));
            return AppendConstant(data, newNodes);
        }

        private static FastTensorKey AppendScalarFloat32(float value, List<FastNode> newNodes)
        {
            var data = new OnnxTensorData<float32>(
                new Shape(Array.Empty<long>()),
                OnnxUtils.CreateTensorValue(new Shape(Array.Empty<long>()), new[] { value }));
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
            new Shape(1), OnnxUtils.CreateTensorValue(new long[] { 1 }, new[] { 0f }));

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
