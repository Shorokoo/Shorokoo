using Shorokoo.Graph;
using Shorokoo.Core;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Rng;
using Shorokoo.Core.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// Per-parameter initialization RNG. Replaces the seeded <c>SHRK_RANDOM_*</c> draw
    /// inside a trainable-parameter's initializer with host-generated noise (see
    /// <see cref="HostRng"/>) keyed by the parameter's own stream, so that
    /// same-shape parameters no longer receive identical values and initialization is
    /// reproducible and backend-independent (it no longer relies on ONNX Runtime's
    /// non-portable random ops).
    ///
    /// <para>The random node is rewritten to <c>Reshape(noiseConstant, shapeInput)</c>:
    /// the flat noise vector has one element per parameter element (which equals the
    /// random draw's element count for every shipping initializer, including
    /// <c>Orthogonal</c> whose draw is a flattened <c>[r, c]</c>), and the node's own
    /// shape input reshapes it, so no shape inference is needed. The noise already
    /// carries the node's declared distribution (low/high or mean/scale), so the
    /// initializer's downstream scaling math is unchanged.</para>
    ///
    /// <para>The substitution scans the initializer's own top-level nodes only — an
    /// initializer must draw inline in its own body. A draw nested inside a function the
    /// body calls cannot be intercepted here, and is rejected loudly (see
    /// <see cref="BuildNoiseInjected"/>) rather than left to lower through the generic
    /// ONNX fallback into unkeyed, non-reproducible backend randomness.</para>
    /// </summary>
    internal static class FastInitRngNoise
    {
        /// <summary>
        /// Returns a new initializer <see cref="Function"/> whose random draws are replaced
        /// by host noise keyed by <paramref name="streamName"/>, or <c>null</c> if
        /// <paramref name="fn"/> contains no random ops (the caller then keeps the original).
        /// Throws if a function called by the initializer body contains a random draw: such
        /// a draw is invisible to this top-level scan, carries no ModelId or key, and would
        /// otherwise silently resolve through the ONNX fallback to real backend randomness —
        /// with no error and no entry in the RNG stream report.
        /// </summary>
        public static Function? BuildNoiseInjected(
            Function fn, (uint k0, uint k1) streamKey, string streamName, long elementCount, int drawRounds = Threefry2x32.Rounds)
        {
            var nested = fn.ReferencedFunctions.FirstOrDefault(f => f.OriginalFastGraph.Nodes.Any(n =>
                n.OpCode == InternalOpCodes.SHRK_RANDOM_UNIFORM ||
                n.OpCode == InternalOpCodes.SHRK_RANDOM_NORMAL));
            if (nested is not null)
                throw new NotSupportedException(
                    $"Initializer '{fn.FriendlyName}' of parameter '{streamName}' draws randomness " +
                    $"inside the called function '{nested.FriendlyName}'. Keyed per-parameter " +
                    "initialization substitutes random draws at the top level of the initializer's " +
                    "own body only; a nested draw keeps no parameter key and would fall back to " +
                    "unkeyed, non-reproducible backend randomness. Move the RandomUniform/" +
                    "RandomNormal call directly into the initializer's body.");

            var body = fn.OriginalFastGraph.Clone();

            var (k0, k1) = streamKey;

            var constAttrDefs = Definitions.NodeDefinitions[OpCodes.CONSTANT].AttributeDefs;
            var reshapeAttrDefs = Definitions.NodeDefinitions[OpCodes.RESHAPE].AttributeDefs;

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

                // Distinct sub-stream per random node within one initializer (all shipping
                // initializers have exactly one, so this is ordinal 0 in practice).
                ulong counterBase = (ulong)randomOrdinal << 40;
                var rng = new HostRng(k0, k1, counterBase, drawRounds);

                float[] noise = isUniform
                    ? rng.Uniform(elementCount,
                        node.Attributes.GetFloatVal(AttrLow) ?? 0.0f,
                        node.Attributes.GetFloatVal(AttrHigh) ?? 1.0f)
                    : rng.Normal(elementCount,
                        node.Attributes.GetFloatVal(AttrMean) ?? 0.0f,
                        node.Attributes.GetFloatVal(AttrScale) ?? 1.0f);

                // Flat [N] noise constant.
                var constKey = FastNodeKey.New();
                var constOut = new FastTensorKey(constKey, 0);
                var noiseData = new OnnxTensorData<float32>(
                    new Shape(elementCount),
                    OnnxUtils.CreateTensorValue(new Shape(elementCount), noise));
                newNodes.Add(new FastNode
                {
                    Key = constKey,
                    OpCode = OpCodes.CONSTANT,
                    Attributes = OnnxCSharpAttributes.FromCSharpVals(
                        new Dictionary<string, object?> { [AttrValue] = noiseData }, constAttrDefs),
                    FullInputs = new Dictionary<string, List<FastTensorKey?>>(),
                    FullOutputs = { [""] = new List<FastTensorKey?> { constOut } },
                });

                // Rewrite the random node in place to Reshape(noise, originalShapeInput),
                // preserving its output key so downstream consumers stay valid.
                var shapeInput = node.Inputs[0]
                    ?? throw new InvalidOperationException("Random init node has null shape input.");
                node.OpCode = OpCodes.RESHAPE;
                node.Attributes = OnnxCSharpAttributes.FromCSharpVals(
                    new Dictionary<string, object?> { [AttrAllowzero] = false }, reshapeAttrDefs);
                node.FullInputs = new Dictionary<string, List<FastTensorKey?>>
                {
                    [""] = new List<FastTensorKey?> { constOut, shapeInput }
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
            // distinct per-parameter bodies (with their distinct noise) down to one, so
            // every same-initializer parameter would collapse to identical values.
            string suffix = new string(streamName.Select(c => char.IsLetterOrDigit(c) ? c : '_').ToArray());
            return new Function(body, fn.FunctionType,
                defaultName: fn.DefaultName + "__rng__" + suffix,
                friendlyName: fn.FriendlyName + "__rng__" + suffix,
                fn.StateOwnership);
        }
    }
}
