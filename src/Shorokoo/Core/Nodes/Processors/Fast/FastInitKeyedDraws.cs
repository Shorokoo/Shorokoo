using Shorokoo.Graph;
using Shorokoo.Core;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Utils;
using Shorokoo.Onnx;
using System;
using System.Collections.Generic;
using System.Linq;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// Per-parameter initialization RNG. Rewrites the abstract <c>SHRK_RANDOM_*</c> draw
    /// inside a trainable-parameter's initializer to the keyed in-graph draw
    /// (<c>SHRK_RNG_UNIFORM/NORMAL</c>) on the parameter's own stream — the same
    /// counter-based Threefry lowering the runtime feeds use — so that same-shape
    /// parameters no longer receive identical values, initialization is reproducible for a
    /// config, and the draw executes on the compute backend (values are never staged as
    /// host-generated constants, so init is not bounded by the protobuf constant-size
    /// ceiling and parallelizes on GPU).
    ///
    /// <para>The draw node is rewritten in place: its key input is the parameter's folded
    /// init key as a <c>[2]</c> constant, its drawBase is the draw's ordinal within the
    /// initializer (a distinct sub-stream per draw; every shipping initializer has exactly
    /// one, so ordinal 0 in practice), and its shape input and declared distribution
    /// bounds carry over — the initializer's downstream scaling math is unchanged.
    /// <see cref="FastLowerRandomOps"/> later lowers the keyed node to a call of the named
    /// algorithm's exported function, exactly as for a runtime feed.</para>
    ///
    /// <para>The substitution runs on the initializer's <b>flattened</b> body
    /// (<see cref="Function.GetFastFlattenedGraph"/>), so a draw factored into a called
    /// function or sub-module is inlined to the top level and keyed like an inline draw —
    /// each inlined call site becomes its own node and its own sub-stream ordinal. A draw
    /// inside a call that survives flattening cannot be keyed and is rejected loudly
    /// rather than left to lower through the generic ONNX fallback into unkeyed,
    /// non-reproducible backend randomness.</para>
    /// </summary>
    internal static class FastInitKeyedDraws
    {
        /// <summary>
        /// Returns a new initializer <see cref="Function"/> whose random draws are rewritten
        /// to keyed in-graph draws on the stream <paramref name="streamKey"/> under the named
        /// <paramref name="algorithm"/>, or <c>null</c> if <paramref name="fn"/> contains no
        /// random ops (the caller then keeps the original). Draws nested in called
        /// functions/sub-modules are reached by flattening the body first; a draw inside a
        /// call that survives flattening throws, since it carries no ModelId or key and would
        /// otherwise silently resolve through the ONNX fallback to real backend randomness —
        /// no error, no entry in the RNG stream report.
        /// </summary>
        public static Function? BuildKeyedDraws(
            Function fn, (uint k0, uint k1) streamKey, string streamName, string algorithm)
        {
            // Flatten so a draw factored into a called function/sub-module becomes a
            // top-level node the substitution below can intercept. Shipping initializers
            // contain no calls, so their flattened body is node-identical to the original.
            var body = fn.GetFastFlattenedGraph().Clone();

            // Backstop: anything still invoked after flattening (a non-inlinable call, or
            // a nested parameter definition) must not smuggle a draw past the top-level
            // scan. Only nodes whose target function actually executes count — inlining
            // leaves dead ShrkCreateModule metadata behind, which still names the (now
            // spliced-in) module function.
            var nested = body.Nodes
                .Where(n => n.OpCode == InternalOpCodes.FUNCTION_INVOKE ||
                            n.OpCode == InternalOpCodes.MODEL_INVOKE ||
                            n.OpCode == InternalOpCodes.MODEL_PARAM ||
                            n.OpCode == InternalOpCodes.MODEL_PARAM_REF ||
                            n.OpCode == InternalOpCodes.MODEL_PARAM_ID_REF ||
                            n.OpCode == InternalOpCodes.MODEL_PARAM_MODEL_REF)
                .Select(n => n.TargetFunction)
                .Where(f => f is not null && f.RngFunctionKind is null)
                .SelectMany(f => f!.ReferencedFunctions.Concat([f]))
                .FirstOrDefault(f => f!.OriginalFastGraph.Nodes.Any(n =>
                    n.OpCode == InternalOpCodes.SHRK_RANDOM_UNIFORM ||
                    n.OpCode == InternalOpCodes.SHRK_RANDOM_NORMAL));
            if (nested is not null)
                throw new NotSupportedException(
                    $"Initializer '{fn.FriendlyName}' of parameter '{streamName}' draws randomness " +
                    $"inside the called function '{nested.FriendlyName}', which could not be inlined. " +
                    "The nested draw keeps no parameter key and would fall back to unkeyed, " +
                    "non-reproducible backend randomness. Move the RandomUniform/RandomNormal " +
                    "call directly into the initializer's body.");

            var (k0, k1) = streamKey;

            var newNodes = new List<FastNode>(body.Nodes.Count);
            int randomOrdinal = 0;

            foreach (var node in body.Nodes)
            {
                bool isUniform = node.OpCode == InternalOpCodes.SHRK_RANDOM_UNIFORM;
                bool isNormal = node.OpCode == InternalOpCodes.SHRK_RANDOM_NORMAL;
                if (!isUniform && !isNormal)
                {
                    newNodes.Add(node);
                    continue;
                }

                var shapeInput = node.Inputs[0]
                    ?? throw new InvalidOperationException("Random init node has null shape input.");

                float a = isUniform
                    ? node.Attributes.GetFloatVal(AttrLow) ?? 0.0f
                    : node.Attributes.GetFloatVal(AttrMean) ?? 0.0f;
                float b = isUniform
                    ? node.Attributes.GetFloatVal(AttrHigh) ?? 1.0f
                    : node.Attributes.GetFloatVal(AttrScale) ?? 1.0f;

                // The parameter's own stream key as a [2] constant, and a distinct
                // sub-stream (drawBase = ordinal) per draw within one initializer.
                var keyKey = AppendConstant(new OnnxTensorData<int64>(
                    new Shape(2), OnnxUtils.CreateTensorValue(new Shape(2), (long[])[k0, k1])), newNodes);
                var drawBaseKey = AppendConstant(new OnnxTensorData<int64>(
                    new Shape(Array.Empty<long>()),
                    OnnxUtils.CreateTensorValue(new Shape(Array.Empty<long>()), (long[])[randomOrdinal])), newNodes);
                var aKey = AppendConstant(new OnnxTensorData<float32>(
                    new Shape(Array.Empty<long>()),
                    OnnxUtils.CreateTensorValue(new Shape(Array.Empty<long>()), (float[])[a])), newNodes);
                var bKey = AppendConstant(new OnnxTensorData<float32>(
                    new Shape(Array.Empty<long>()),
                    OnnxUtils.CreateTensorValue(new Shape(Array.Empty<long>()), (float[])[b])), newNodes);

                // Rewrite the random node in place to the keyed draw (inputs
                // [key, drawBase, shape, a, b]), preserving its output key so downstream
                // consumers stay valid. FastLowerRandomOps lowers it to the algorithm's
                // function call at ONNX prep.
                var newOp = isUniform ? InternalOpCodes.SHRK_RNG_UNIFORM : InternalOpCodes.SHRK_RNG_NORMAL;
                node.OpCode = newOp;
                node.Attributes = OnnxCSharpAttributes.FromCSharpVals(
                    new Dictionary<string, object?> { [ShrkAttrRngAlgorithm] = algorithm },
                    Definitions.NodeDefinitions[newOp].AttributeDefs);
                node.FullInputs = new Dictionary<string, List<FastTensorKey?>>
                {
                    [""] = new List<FastTensorKey?> { keyKey, drawBaseKey, shapeInput, aKey, bKey }
                };
                newNodes.Add(node);
                randomOrdinal++;
            }

            if (randomOrdinal == 0)
                return null; // no random ops; caller keeps the shared original

            body.Nodes = newNodes;

            // Give the per-parameter initializer a unique name. The original name
            // ("KaimingUniform", ...) is shared across every parameter using that
            // initializer; leaving it unchanged makes ONNX function emission dedupe the
            // distinct per-parameter bodies (with their distinct keys) down to one, so
            // every same-initializer parameter would collapse to identical values.
            string suffix = new string(streamName.Select(c => char.IsLetterOrDigit(c) ? c : '_').ToArray());
            return new Function(body, fn.FunctionType,
                defaultName: fn.DefaultName + "__rng__" + suffix,
                friendlyName: fn.FriendlyName + "__rng__" + suffix,
                fn.StateOwnership);
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
    }
}
